#!/usr/bin/env python3
"""bench_g130.py: 8-rank variant-vs-control bench + exactness + stall test. STOPPED WINDOW ONLY, GPU 0, one process per Spark,
started by ops/run-bench8.sh. Nothing installed. NOT RUN YET: written offline, only syntax-checked (no GPU was allowed).

All arms share ONE RdmaAR instance (same QPs, arena, proxy thread and core) and switch with set_variant at a quiet point, so a
0.35 us threshold is not lost in per-instance differences. A graph keeps the kernels it was captured with.
Phases (--phases, default X,S,L):
  X  exactness on the real fabric, per arm, eager + graph: acc = +0.0f, FADD r = 0..7, one RN (planted classes: -0, subnormal,
     cancel, order-dependent overflow, inf/NaN, absorption, order), gathers vs torch.cat, identical on all ranks. The reversed-order
     reference proves the test can see an order change.
  S  forced proxy stalls on real hardware, without any test hook in the .so: a busy process shares the proxy's core, so the
     scheduler parks the proxy for ms while the GPU runs on. Free-running mixed op list, eager, no sync, then bitwise checks.
  L  latency, ABBA-rotated: modes (b2b, l2pf) x (AG M=6, AR M=4, AR M=6). us/op, plus stage/net/tail/total and proxy hand-off / post
     from h.stats when RDMA_AR_TIMING=1 (--timing). Gate logic is in ops/verdict.py.
A control .so has no set_variant: only the arm ctl runs (the cross-check that flags off == control).
"""
import argparse
import ctypes
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import roce8_base as B  # noqa: E402

ARMS = {  # name: (split_post, hyg_proxy, pub_mask)
    "ctl": (False, False, 0), "split": (True, False, 0), "v1": (False, False, 1), "v2": (False, False, 2),
    "v1v2": (False, False, 3), "hygp": (False, True, 0), "hygg": (False, False, 4), "hyg": (False, True, 4),
    "all": (True, True, 7),
}
HIDDEN, QKV = B.HIDDEN, B.QKV_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--phases", default="X,S,L")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--modes", default="b2b,l2pf")
    ap.add_argument("--cells", default="ag:6,ar:4,ar:6")
    ap.add_argument("--n-ops", type=int, default=40)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--gap-us", type=float, default=100.0)
    ap.add_argument("--pf-mb", type=float, default=6.0)
    ap.add_argument("--pf-segs", type=int, default=2)
    ap.add_argument("--pin", default="x925", help="x925: proxy on an X925 core (lowest noise); none: unpinned, as serving")
    ap.add_argument("--stall-ops", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--timeout-ms", type=int, default=30000)
    a = ap.parse_args()

    import torch.distributed as dist
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    grp = dist.group.WORLD
    ext = B.load_ext(a.ext)
    core = B.X925[0] if a.pin == "x925" else -1
    before = B.tids()
    with B.inherit_affinity(core):
        h = ext.RdmaAR(rank, world, B.MAX_BYTES, B.MAX_BYTES, B.RDMA_HCAS, B.GID, 1, -1, a.timeout_ms, 0)
        infos = [None] * world
        dist.all_gather_object(infos, h.local_info(), group=grp)
        h.connect(infos)
    dist.barrier(group=grp)
    proxy_tids = B.pinned_tids(before, core)
    has_var = hasattr(h, "set_variant")
    arms = [n for n in a.arms.split(",") if n and (has_var or n == "ctl")]
    timing = os.environ.get("RDMA_AR_TIMING", "0") == "1"
    res = {"rank": rank, "world": world, "ext": a.ext, "ext_sha": B.file_sha(a.ext), "args": vars(a), "arms": arms, "timing": timing,
           "host": os.uname().nodename, "default_variant": list(h.variant()) if has_var else None,
           "env": {k: v for k, v in os.environ.items() if k.startswith(("RDMA_AR", "NCCL_", "MLX5"))}}
    if has_var and list(h.variant()) != [0, 0, 0]:
        raise SystemExit(f"flags must be unset for the bench (arms set them), got {list(h.variant())}")
    # no pingpong comparison across arms: HYG_PROXY delays ping by up to kHygSpin loads, so use the hand-off stat instead
    B.log0(f"init ok: ext={a.ext} ({res['ext_sha']}) arms={arms} timing={timing} proxy_tids={proxy_tids} core={core}")
    fails = 0

    def use(arm):
        if has_var:
            s, g, m = ARMS[arm]
            h.set_variant(s, g, m)

    def two(nb):
        return nb >= 32768  # the served policy (RDMA_AR_TWOSHOT_MIN_BYTES default)

    def ar(x):
        return h.all_reduce(x, two(x.numel() * x.element_size()))

    def ag(x):
        out = torch.empty(x.shape[0], world * x.shape[1], dtype=x.dtype, device=x.device)
        h.all_gather_cols(out, x, x.shape[1] * x.element_size())
        return out

    def graph_of(fn):
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            fn()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g

    # ================================================================ X ===
    if "X" in a.phases.split(","):
        xres = []
        for dtype, rows in ((torch.bfloat16, [1, 2, 4, 6, 12, 48]), (torch.float32, [1, 4, 6])):
            for M in rows:
                xc, _cl = B.gen_inputs(M, dtype, a.seed, world)
                x32 = xc.to(dev).float()
                want = B.ref_sum(x32, True).to(dtype)
                rev = B.ref_sum(x32, True, order=range(world - 1, -1, -1)).to(dtype)
                sens = int((B.bits_of(rev) != B.bits_of(want)).sum())
                if sens == 0:
                    fails += 1
                mine = xc[rank].to(dev).contiguous()
                for arm in arms:
                    use(arm)
                    out = ar(mine.clone())
                    torch.cuda.synchronize()
                    xin = mine.clone()
                    holder = {}
                    g = graph_of(lambda: holder.__setitem__("o", ar(xin)))
                    ok_g = True
                    for _ in range(3):
                        g.replay()
                        torch.cuda.synchronize()
                        ok_g &= torch.equal(B.bits_of(holder["o"]), B.bits_of(want))
                    ok_e = torch.equal(B.bits_of(out), B.bits_of(want))
                    hs = [None] * world
                    dist.all_gather_object(hs, B.sha(out), group=grp)
                    same = all(s == hs[0] for s in hs)
                    ok = ok_e and ok_g and same
                    fails += 0 if ok else 1
                    xres.append(dict(dtype=str(dtype), M=M, arm=arm, eager=ok_e, graph=ok_g, same=same, order_diffs=sens))
                    B.log0(f"[X] {str(dtype)[6:]:8s} M={M:2d} {arm:6s} eager {ok_e} graph {ok_g} ranks-identical {same} (reversed-order diffs {sens})")
        for M in (1, 6, 48):
            gq = torch.Generator().manual_seed(a.seed + 31 * M)
            allx = torch.randint(-32768, 32767, (world, M, QKV), generator=gq, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
            want = torch.cat([allx[r] for r in range(world)], dim=-1).to(dev)
            mine = allx[rank].to(dev).contiguous()
            for arm in arms:
                use(arm)
                out = ag(mine)
                torch.cuda.synchronize()
                ok = torch.equal(B.bits_of(out), B.bits_of(want))
                fails += 0 if ok else 1
                xres.append(dict(gather=M, arm=arm, ok=ok))
                B.log0(f"[X] gather M={M:2d} {arm:6s} bitwise == torch.cat: {ok}")
        use("ctl")
        res["X"] = xres

    # ================================================================ S ===
    if "S" in a.phases.split(",") and core >= 0:
        sres = []
        for ai, arm in enumerate(arms):
            use(arm)
            rng = random.Random(a.seed + ai)
            ops = []
            for _ in range(a.stall_ops):
                kind = rng.choice(["two", "two", "two", "one", "ag", "agc"])
                ops.append((kind, rng.choice([1, 2, 3]) if kind == "one" else rng.choice([4, 6, 8]) if kind == "two" else rng.choice([1, 6])))
            # inputs first: no CPU work between launches, so the GPU really runs ahead of a parked proxy
            inputs = []
            for i, (kind, M) in enumerate(ops):
                seed = a.seed * 7 + ai * 100003 + i
                if kind in ("two", "one"):
                    inputs.append(B.gen_inputs(M, torch.bfloat16, seed, world)[0][rank].to(dev).contiguous())
                else:
                    gq = torch.Generator().manual_seed(seed)
                    allx = torch.randint(-32768, 32767, (world, M, QKV), generator=gq, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
                    inputs.append(allx[rank].to(dev).contiguous())
            torch.cuda.synchronize()
            victims = {ai % world, (ai + 3) % world}
            hog = None
            if rank in victims:
                hog = subprocess.Popen([sys.executable, "-I", "-c", f"import os\nos.sched_setaffinity(0,{{{core}}})\nwhile True: pass\n"])
            time.sleep(0.2)
            dist.barrier(group=grp)
            h.stats(True)
            outs = []
            for i, (kind, M) in enumerate(ops):
                seed = a.seed * 7 + ai * 100003 + i
                mine = inputs[i]
                if kind in ("two", "one"):
                    outs.append((kind, M, seed, h.all_reduce(mine, kind == "two")))
                elif kind == "agc":
                    outs.append((kind, M, seed, ag(mine)))
                else:
                    out = torch.empty((world, M, QKV), dtype=torch.bfloat16, device=dev)
                    h.all_gather_into(out, mine)
                    outs.append((kind, M, seed, out))
            torch.cuda.synchronize()
            st = list(h.stats(True))
            if hog is not None:
                hog.kill()
                hog.wait()
            bad = 0
            for kind, M, seed, out in outs:
                if kind in ("two", "one"):
                    xc, _c = B.gen_inputs(M, torch.bfloat16, seed, world)
                    want = B.ref_sum(xc.to(dev).float(), True).to(torch.bfloat16)
                    bad += 0 if torch.equal(B.bits_of(out), B.bits_of(want)) else 1
                else:
                    gq = torch.Generator().manual_seed(seed)
                    allx = torch.randint(-32768, 32767, (world, M, QKV), generator=gq, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(dev)
                    want = allx if kind == "ag" else torch.cat([allx[r] for r in range(world)], dim=-1)
                    bad += 0 if torch.equal(B.bits_of(out), B.bits_of(want)) else 1
            del outs
            fails += 1 if bad else 0
            # px post max shows the stalls really happened; 0 stalls would make a pass meaningless (read it on the victims)
            sres.append(dict(arm=arm, ops=len(ops), bad=bad, victim=rank in victims, px_post_max_ns=st[8]))
            allb = [None] * world
            dist.all_gather_object(allb, (bad, rank in victims, st[8]), group=grp)
            B.log0(f"[S] {arm:6s} {len(ops)} ops, wrong outputs per rank {[x[0] for x in allb]}; max proxy post us on victims "
                   f"{[round(x[2] / 1000) for x in allb if x[1]]}, others {[round(x[2] / 1000) for x in allb if not x[1]]}")
        use("ctl")
        res["S"] = sres

    # ================================================================ L ===
    if "L" in a.phases.split(","):
        lib = ctypes.CDLL(B.LIVE_L2PF)
        lib.dsv41_l2pf.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p]
        lib.dsv41_l2pf.restype = ctypes.c_int
        lib.dsv41_spin.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p]
        lib.dsv41_spin.restype = ctypes.c_int
        side = torch.cuda.Stream()
        n = a.n_ops
        pf_bytes = int(a.pf_mb * B.MB) & ~15
        wbufs = [torch.empty(pf_bytes // 2, dtype=torch.bfloat16, device=dev) for _ in range(n)]
        for t in wbufs:
            t.normal_()
        segb = (pf_bytes // max(1, a.pf_segs)) & ~15
        tabs = [torch.tensor(sum(([t.data_ptr() + k * segb, segb] for k in range(a.pf_segs)), []), dtype=torch.int64, device=dev) for t in wbufs]

        def body(kind, M, mode, with_ops):
            xa = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=dev)
            xg = torch.randn(M, QKV, dtype=torch.bfloat16, device=dev)

            def fn():
                for i in range(n):
                    if mode != "b2b":
                        assert lib.dsv41_spin(int(a.gap_us * 1000), 8, torch.cuda.current_stream().cuda_stream) == 0
                    if mode == "l2pf":
                        side.wait_stream(torch.cuda.current_stream())
                        assert lib.dsv41_l2pf(tabs[i % n].data_ptr(), a.pf_segs, B.CH, side.cuda_stream) == 0
                    if with_ops:
                        ar(xa) if kind == "ar" else ag(xg)
                if mode == "l2pf":
                    torch.cuda.current_stream().wait_stream(side)
            return fn

        def timed(g, arm=None):
            dist.barrier(group=grp)
            if arm:
                use(arm)
            for _ in range(2):
                g.replay()
            torch.cuda.synchronize()
            dist.barrier(group=grp)
            h.stats(True)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(a.replays):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            st = list(h.stats(True))
            return e0.elapsed_time(e1) * 1000.0 / (a.replays * n), [v / 1000.0 for v in (st[1], st[2], st[3], st[4], st[6], st[7])]

        lres = []
        for mode in [s for s in a.modes.split(",") if s]:
            for cell in a.cells.split(","):
                kind, M = cell.split(":")
                M = int(M)
                gb = graph_of(body(kind, M, mode, False)) if mode != "b2b" else None
                graphs = {}
                for arm in arms:
                    use(arm)
                    graphs[arm] = graph_of(body(kind, M, mode, True))
                per = {x: [] for x in arms}
                parts = {x: [] for x in arms}
                for t in range(a.trials):
                    order = arms[t % len(arms):] + arms[:t % len(arms)]
                    if t % 2:
                        order = order[::-1]
                    b = timed(gb)[0] if gb is not None else 0.0
                    for arm in order:
                        us, pt = timed(graphs[arm], arm)
                        per[arm].append(us - b)
                        parts[arm].append(pt)
                use("ctl")
                med = {x: float(np.median(v)) for x, v in per.items()}
                lres.append(dict(mode=mode, kind=kind, M=M, us_op=med, trials=per,
                                 parts={x: np.median(np.array(v), axis=0).tolist() for x, v in parts.items()}))
                B.log0(f"[L] {mode:4s} {kind} M={M} " + " ".join(f"{x} {v:6.2f}" for x, v in med.items()))
                del graphs, gb
                torch.cuda.synchronize()
        res["L"] = lres

    allres = [None] * world
    dist.all_gather_object(allres, {"rank": rank, "fails": fails, "L": res.get("L", []), "S": res.get("S", [])}, group=grp)
    res["fails"] = fails
    with open(f"{a.out}.rank{rank}.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    if rank == 0:
        with open(f"{a.out}.allranks.json", "w") as f:
            json.dump(allres, f, indent=1, default=str)
    B.log0(f"DONE failures per rank: {[r['fails'] for r in allres]}")
    dist.barrier(group=grp)
    os._exit(0 if all(r["fails"] == 0 for r in allres) else 1)


if __name__ == "__main__":
    main()
