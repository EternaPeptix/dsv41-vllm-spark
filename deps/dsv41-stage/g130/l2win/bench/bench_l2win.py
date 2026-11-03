#!/usr/bin/env python3
"""bench_l2win.py (g130 l2win): 8-rank OFFLINE bench, real RDMA fabric, service STOPPED on all 8. Started from the Mac by
ops/run-bench8.sh. Uses the INSTALLED rdma_all_reduce.py and the production rdma_ar_ext.so as served, the live r4 burst
library and the installed g2 pacer library. Installs nothing, writes only <out>.* under results/. NOT RUN YET.

Window shapes are the ones the live plan builds with DSV41_L2_PREFETCH_MHC=0 (rank0.log, launcher 37757dde):
  gate   1 segment, 3.75 MiB         the post-attention AR window, consumer = bf16 router GEMM (Kernel2) at AR end +5.4 us
  short  2 segments, 0.03 + 1.09 MiB the post-MoE AR window, consumer = qkv_a-shard GEMM at AR end +6 us
  gather 4 segments, 0.16 + 5.00 + 0.16 + 0.69 MiB, the qkv_a gather window, consumer = q_b GEMM at AR end +1.9 us
  big    4 segments, 6 MiB           the > 8-row AR window (the 4-session class)

Mode names (what the driver flags do; ops/verdict_l2win.py maps a mode to its env):
  none                 no prefetch; the bitwise reference
  live                 r4 burst kernel, as served
  d<US>                burst after a chained dsv41_spin(US us), 1 CTA, on the same side stream
  p<R>_c<KiB>[_d<US>][_x<US>]      the paced issuer at R GB/s with that chunk, in-kernel delay, issue cap (default 150)
  s<R>_c<KiB>[+b<R2>|+d<US>]       windows <= 2 MiB paced at R; larger ones paced at R2, or burst after a spin, or burst
Gather windows stay `live` in every mode except `none` (phase B has a separate info-only row for gather arms).

Phases:
 C  issuer ceiling, rank-local, stats pointer: {gate, short, big} x chunk {16,32,64} KiB x request {85..175, 400} GB/s,
    eager, L2 flushed before each launch; then the served modes from a captured graph. Reports achieved GB/s, issued
    bytes, first/last issue. Prefetched buffers must stay bitwise unchanged.
 B  collective stretch in isolation: 40 x [fork window -> AR(M) or gather -> gap spin], ABBA. us/op and in-kernel
    stage / net / tail. Classes: gate and short at M = 2 (one-shot), 4, 6 (two-shot); gather (6,224) with --ag-modes.
 W  per-layer time with the consumers, ABBA. M <= 8: fork gate -> AR -> mHC-like fp32 GEMM on a side stream and router-like
    bf16 GEMM -> rest spin -> fork short -> AR -> shard GEMM -> fork gather -> gather cols -> q_b GEMM -> rest spin.
    M > 8: fork big -> AR -> qkv_a-like GEMM (9.17 MB) -> q_b-like GEMM (5.24 MB) -> rest spin.
    This is the only phase that can see bytes landing late.
 F  bitwise, same graphs as W, plus forced-overlap modes (--forced) whose pacer is delayed into the consumer GEMMs on
    every op. M = 1,2,4,6,8,12,16,24, >= 200 interleaved replays per mode per M, --inputs distinct input sets, AR inputs
    distinct per op and per rank. Every output of every mode is compared with `none` by uint8 view, and the AR digests
    across ranks. Any mismatch fails the bench.
Writes <out>.rank<R>.json; rank 0 prints tables. Exit 0 only if every bitwise check passed on every rank; a timing loss is
a result, not an error. GO / KILL is decided by ops/verdict_l2win.py on the rank-0 json.
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys

os.environ["DSV41_RDMA_AR"] = "1"
os.environ.setdefault("DSV41_RDMA_PARLOAD", "1")
os.environ["DSV41_RDMA_AG_COLS"] = "1"
os.environ["DSV41_L2_PREFETCH"] = "0"  # this bench drives the prefetch kernels itself
os.environ["RDMA_AR_TIMING"] = "1"
os.environ.setdefault("RDMA_AR_MAX_REDUCE_BYTES", "2097152")
os.environ["RDMA_AR_FINGERPRINT"] = "1"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

HIDDEN = 5120
MIB = 1 << 20
CH = 16384
SMALL_MAX = 2 * MIB  # DSV41_L2_PREFETCH_PACE_MAX_WIN_KB=2048
LIVE_L2PF = os.path.expanduser("~/dsv41-stage/r4/l2-prefetch/libdsv41_l2pf.so")
G2_L2PF = os.path.expanduser("~/rdma_ar_dsv41/build-g2ct/libdsv41_l2pf_g2.so")
PACER_SHA = "0451009d0b4732e9c4867ab2d3ec7130be24e517bf22228a83dfa6a222a9d622"
R4_SHA = "a09f85e8757e693f7c0bccce62b5717678a06f4c85cb00959daf64e38939bcbc"
DEFAULT_MODES = ("none,live,d0,d1,d3,p85_c16,p125_c64,p150_c64,p150_c64_x26,s85_c16,s85_c64+b150,s85_c16+d1,s85_c16+d3")
FORCED = "p85_c16_d40,p150_c64_d40"  # F only: the issuer runs across the consumer GEMMs on every op
OUT_SMALL = ("ar1", "mix", "gate", "ar2", "y", "g", "q")
OUT_BIG = ("ar1", "y", "q")


def parse_mode(s):
    """Mode name -> dict. Pure; tested by tests/test_l2win_plan_cpu.py."""
    if s in ("none", "live"):
        return {"kind": s}
    m = re.fullmatch(r"d(\d+(?:\.\d+)?)", s)
    if m:
        return {"kind": "d", "delay_us": float(m.group(1))}
    m = re.fullmatch(r"p(\d+(?:\.\d+)?)_c(\d+)(?:_d(\d+(?:\.\d+)?))?(?:_x(\d+(?:\.\d+)?))?", s)
    if m:
        return {"kind": "p", "gbps": float(m.group(1)), "chunk": int(m.group(2)) * 1024,
                "delay_us": float(m.group(3) or 0), "cap_us": float(m.group(4) or 150)}
    m = re.fullmatch(r"s(\d+(?:\.\d+)?)_c(\d+)(?:\+b(\d+(?:\.\d+)?)|\+d(\d+(?:\.\d+)?))?", s)
    if m:
        return {"kind": "s", "gbps": float(m.group(1)), "chunk": int(m.group(2)) * 1024,
                "big_gbps": float(m.group(3)) if m.group(3) else 0.0,
                "big_delay_us": float(m.group(4)) if m.group(4) else None, "cap_us": 150.0}
    raise ValueError(f"bad mode {s}")


def log0(*a):
    if dist.get_rank() == 0:
        print(*a, flush=True)


def fsha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def clock(tag):
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        log0(f"[clock {tag}] {r.stdout.strip() or r.stderr.strip()}")
    except Exception as e:  # noqa: BLE001 - informational
        log0(f"[clock {tag}] unavailable ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phases", default="C,B,W,F")
    ap.add_argument("--modes", default=DEFAULT_MODES)
    ap.add_argument("--forced", default=FORCED)
    ap.add_argument("--ag-modes", default="none,live,d0,d1,d3,p150_c64")
    ap.add_argument("--b-rows", default="2,4,6")
    ap.add_argument("--w-rows", default="1,2,4,6,8,12,16,24")
    ap.add_argument("--f-rows", default="1,2,4,6,8,12,16,24")
    ap.add_argument("--n-ops", type=int, default=40)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--gap-us", type=float, default=100.0)
    ap.add_argument("--attn-us", type=float, default=30.0)
    ap.add_argument("--rest-us", type=float, default=150.0)
    ap.add_argument("--ceil-reps", type=int, default=30)
    ap.add_argument("--bitwise-replays", type=int, default=204)
    ap.add_argument("--inputs", type=int, default=8)
    a = ap.parse_args()
    phases = set(a.phases.split(","))
    modes = [m for m in a.modes.split(",") if m]
    forced = [m for m in a.forced.split(",") if m]
    for m in modes + forced + a.ag_modes.split(","):
        parse_mode(m)
    if "none" not in modes or "live" not in modes:
        sys.exit("--modes must hold none and live")

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    grp = dist.group.WORLD
    from vllm.distributed.device_communicators import rdma_all_reduce as W

    rd = W.RdmaAllReduce(grp, rank, world, device)
    assert not rd.disabled, "rdma_ar failed to initialise (see log)"
    res = {"rank": rank, "world": world, "ext": W._EXT_PATH, "ext_sha": fsha(W._EXT_PATH), "args": vars(a),
           "env": {k: v for k, v in os.environ.items() if k.startswith(("RDMA_AR", "DSV41_", "NCCL_"))},
           "lib_sha": {"pacer": fsha(G2_L2PF), "r4": fsha(LIVE_L2PF)}}
    log0(f"init ok: ext={W._EXT_PATH} sha={res['ext_sha'][:8]} parload={rd.h.parload()} ag_cols={rd.ag_cols}")
    assert res["lib_sha"]["pacer"] == PACER_SHA, "pacer library is not the installed 0451009d"
    assert res["lib_sha"]["r4"] == R4_SHA, "burst library is not the live a09f85e8"
    assert rd.ag_cols, "column gathers off: wrong .so"

    live = ctypes.CDLL(LIVE_L2PF)
    live.dsv41_l2pf.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p]
    live.dsv41_l2pf.restype = ctypes.c_int
    live.dsv41_spin.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p]
    live.dsv41_spin.restype = ctypes.c_int
    g2 = ctypes.CDLL(G2_L2PF)
    g2.dsv41_l2pf_paced.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong,
                                    ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p]
    g2.dsv41_l2pf_paced.restype = ctypes.c_int
    side, side3, s_mhc = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    pend = {"b": None, "p": None}

    def spin(us):
        assert live.dsv41_spin(int(us * 1000), 8, torch.cuda.current_stream().cuda_stream) == 0

    def join_b():
        if pend["b"] is not None:
            torch.cuda.current_stream().wait_stream(pend["b"])
            pend["b"] = None

    def join_all():
        join_b()
        if pend["p"] is not None:
            torch.cuda.current_stream().wait_stream(pend["p"])
            pend["p"] = None

    def burst(t, delay_us=None):
        side.wait_stream(torch.cuda.current_stream())
        if delay_us is not None:  # = _fork_part mode "d" of the patched driver: 1 CTA
            assert live.dsv41_spin(int(delay_us * 1000), 1, side.cuda_stream) == 0
        assert live.dsv41_l2pf(t[0].data_ptr(), t[1], CH, side.cuda_stream) == 0
        pend["b"] = side

    def paced(t, gbps, chunk, delay_us=0.0, cap_us=150.0, stats=None):
        side3.wait_stream(torch.cuda.current_stream())
        assert g2.dsv41_l2pf_paced(t[0].data_ptr(), t[1], chunk, int(round(chunk * 1000.0 / gbps)),
                                   int(delay_us * 1000), int(cap_us * 1000), stats, side3.cuda_stream) == 0
        pend["p"] = side3

    def fork(cfg, t):
        k = cfg["kind"]
        if k == "none":
            return
        if k == "live":
            burst(t)
        elif k == "d":
            burst(t, cfg["delay_us"])
        elif k == "p":
            paced(t, cfg["gbps"], cfg["chunk"], cfg["delay_us"], cfg["cap_us"])
        elif t[2] <= SMALL_MAX:
            paced(t, cfg["gbps"], cfg["chunk"], 0.0, cfg["cap_us"])
        elif cfg["big_gbps"] > 0:
            paced(t, cfg["big_gbps"], cfg["chunk"], 0.0, cfg["cap_us"])
        else:
            burst(t, cfg["big_delay_us"])

    def tab(pieces):
        """[(tensor, nbytes|None)] -> (int64 table on device, nseg, bytes)"""
        segs = []
        for tn, nb in pieces:
            full = tn.numel() * tn.element_size()
            nb = (full if nb is None else min(nb, full)) & ~15
            if nb > 0:
                segs += [tn.data_ptr(), nb]
        return torch.tensor(segs, dtype=torch.int64, device=device), len(segs) // 2, sum(segs[1::2])

    def graph_of(fn, warm=2):
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(warm):
                fn()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g

    def time_graph(g, n_ops):
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier(group=grp)
        rd.h.stats(True)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(a.replays):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        st = rd.h.stats(True)
        return e0.elapsed_time(e1) * 1000.0 / (a.replays * n_ops), [v / 1000.0 for v in st[1:5]]

    def abba(graphs, n_ops):
        names = list(graphs)
        per = {k: [] for k in names}
        parts = {k: [] for k in names}
        for t in range(a.trials):
            order = names[t % len(names):] + names[:t % len(names)]
            for k in order + order[::-1]:
                us, pt = time_graph(graphs[k], n_ops)
                per[k].append(us)
                parts[k].append(pt)
        return {k: (statistics.median(per[k]), [statistics.median([p[i] for p in parts[k]]) for i in range(4)], per[k])
                for k in names}

    # ---- weights: distinct per op, so every op's window is L2-cold. About 23 MB x 40 per rank. ----
    nops = a.n_ops
    torch.manual_seed(4321 + rank)
    fn_w = [torch.randn(24, 20480, device=device) * 0.01 for _ in range(nops)]                         # read, not prefetched
    gate_w = [(torch.randn(384, HIDDEN, device=device) * 0.02).to(torch.bfloat16) for _ in range(nops)]  # 3.75 MiB
    shard_s = [torch.randint(0, 256, (35840,), dtype=torch.uint8, device=device) for _ in range(nops)]
    shard_w = [(torch.randn(224, 2560, device=device) * 0.02).to(torch.bfloat16) for _ in range(nops)]  # 1.09 MiB
    qb_s = [torch.randint(0, 256, (163840,), dtype=torch.uint8, device=device) for _ in range(nops)]
    qb_w = [(torch.randn(2560, 1024, device=device) * 0.02).to(torch.bfloat16) for _ in range(nops)]    # 5.00 MiB
    wob_s = [torch.randint(0, 256, (163840,), dtype=torch.uint8, device=device) for _ in range(nops)]
    wob_w = [torch.randint(0, 256, (723520,), dtype=torch.uint8, device=device) for _ in range(nops)]
    qkv_s = [torch.randint(0, 256, (286720,), dtype=torch.uint8, device=device) for _ in range(nops)]
    qkv_w = [(torch.randn(896, HIDDEN, device=device) * 0.02).to(torch.bfloat16) for _ in range(nops)]  # 8.75 MiB
    t_gate = [tab([(g_, None)]) for g_ in gate_w]
    t_short = [tab([(s, None), (w, None)]) for s, w in zip(shard_s, shard_w)]
    t_gather = [tab([(s, None), (w, None), (bs, None), (bw, None)]) for s, w, bs, bw in zip(qb_s, qb_w, wob_s, wob_w)]
    t_big = [tab([(shard_s[i], None), (shard_w[i], None), (qkv_s[i], None),
                  (qkv_w[i], 6 * MIB - 35840 - 1146880 - 286720)]) for i in range(nops)]
    log0(f"windows MiB: gate {t_gate[0][2] / MIB:.2f}/{t_gate[0][1]} short {t_short[0][2] / MIB:.2f}/{t_short[0][1]} "
         f"gather {t_gather[0][2] / MIB:.2f}/{t_gather[0][1]} big {t_big[0][2] / MIB:.2f}/{t_big[0][1]}")
    clock("start")
    fails = 0

    # ================================================================================= phase C ==
    if "C" in phases:
        stats = torch.zeros(4, dtype=torch.int64, device=device)
        scratch = torch.zeros(16 * MIB, dtype=torch.float32, device=device)  # 64 MB: evicts the 24 MiB L2

        def flush():
            scratch.add_(1)
            torch.cuda.synchronize()

        def read():
            torch.cuda.synchronize()
            iss, cnt, first, last = [int(v) for v in stats.tolist()]
            return iss, cnt, first / 1e3, last / 1e3

        before = [t.clone() for t in (gate_w[0], shard_w[0], shard_s[0], qkv_w[0])]
        tables = {"gate": t_gate[0], "short": t_short[0], "big": t_big[0]}
        cases = [(tn, c, r) for tn in tables for c in (16, 32, 64) for r in (85, 100, 116, 125, 150, 175, 400)]
        samp = {x: [] for x in cases}
        st_p = torch.cuda.Stream()
        for rep in range(a.ceil_reps):
            k = rep % len(cases)
            for x in cases[k:] + cases[:k]:
                tn, c, r = x
                t = tables[tn]
                flush()
                stats.zero_()
                torch.cuda.synchronize()
                ch = c * 1024
                assert g2.dsv41_l2pf_paced(t[0].data_ptr(), t[1], ch, int(round(ch * 1000.0 / r)), 0, 0, stats.data_ptr(),
                                           st_p.cuda_stream) == 0
                samp[x].append(read())

        def summarize(x, s, nb):
            ch = x[1] * 1024
            full = sum(v[0] == nb for v in s)
            mf, ml = statistics.median(v[2] for v in s), statistics.median(v[3] for v in s)
            ach = (nb - ch) / max(1e-9, ml - mf) / 1e3
            return dict(table=x[0], chunk_kib=x[1], req_gbps=x[2], full=full, n=len(s), first_us=round(mf, 2),
                        last_us=round(ml, 2), achieved_gbps=round(ach, 1))

        tabC = [summarize(x, samp[x], tables[x[0]][2]) for x in cases]
        for r_ in tabC:
            log0(f"[C] {r_['table']:6s} chunk {r_['chunk_kib']:3d} KiB req {r_['req_gbps']:4d}: achieved {r_['achieved_gbps']:6.1f} "
                 f"GB/s first {r_['first_us']:5.2f} last {r_['last_us']:6.2f} us full {r_['full']}/{r_['n']}")
        # the same from captured graphs, for the served shapes
        tabCG = []
        for tn, c, r in (("short", 16, 85), ("gate", 64, 125), ("gate", 64, 150), ("short", 64, 85), ("big", 64, 150)):
            t = tables[tn]
            ch = c * 1024
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                assert g2.dsv41_l2pf_paced(t[0].data_ptr(), t[1], ch, int(round(ch * 1000.0 / r)), 0, 0, stats.data_ptr(),
                                           torch.cuda.current_stream().cuda_stream) == 0
            torch.cuda.synchronize()
            s = []
            for _ in range(a.ceil_reps):
                flush()
                stats.zero_()
                g.replay()
                s.append(read())
            tabCG.append(summarize((tn, c, r), s, t[2]))
            r_ = tabCG[-1]
            log0(f"[C graph] {tn:6s} chunk {c:3d} KiB req {r:4d}: achieved {r_['achieved_gbps']:6.1f} GB/s last {r_['last_us']:6.2f} "
                 f"us full {r_['full']}/{r_['n']}")
            del g
        same = all(torch.equal(x, y) for x, y in zip(before, (gate_w[0], shard_w[0], shard_s[0], qkv_w[0])))
        if not same:
            fails += 1
        log0(f"[C] prefetched buffers bitwise unchanged: {same}")
        res["C"] = tabC
        res["CG"] = tabCG
        del scratch, before

    # ================================================================================= phase B ==
    if "B" in phases:
        tabB = []
        for cls, rows, mlist, tb in [("gate", int(r), modes, t_gate) for r in a.b_rows.split(",")] + \
                                    [("short", int(r), modes, t_short) for r in a.b_rows.split(",")] + \
                                    [("gather", 6, a.ag_modes.split(","), t_gather)]:
            if cls == "gather":
                x = (torch.randn(rows, 224, device=device) * 2).to(torch.bfloat16)
                op = lambda x=x: rd.all_gather_cols(x, 1)  # noqa: E731
            else:
                x = (torch.randn(rows, HIDDEN, device=device) * 2).to(torch.bfloat16)
                op = lambda x=x: rd.all_reduce(x)  # noqa: E731
            graphs = {}
            for name in mlist:
                def fn(cfg=parse_mode(name), op=op, tb=tb):
                    for i in range(nops):
                        join_b()
                        fork(cfg, tb[i])
                        op()
                        spin(a.gap_us)
                    join_all()
                graphs[name] = graph_of(fn)
            r = abba(graphs, nops)
            base, kbase = r["none"][0], r["none"][1][3]
            lb, klb = r["live"][0], r["live"][1][3]
            for name in mlist:
                us, pt, allv = r[name]
                tabB.append(dict(cls=cls, M=rows, mode=name, us_op=us - a.gap_us, stretch=us - base, vs_live=us - lb,
                                 stage=pt[0], net=pt[1], tail=pt[2], kernel=pt[3], kstretch=pt[3] - kbase,
                                 kvs_live=pt[3] - klb, all=allv))
                log0(f"[B] {cls:6s} M={rows} {name:14s} op {us - a.gap_us:7.2f} ({us - base:+6.2f} none, {us - lb:+6.2f} live) "
                     f"kernel {pt[3]:6.2f} ({pt[3] - klb:+6.2f}) = stage {pt[0]:5.2f} net {pt[1]:6.2f} tail {pt[2]:5.2f}")
            del graphs
        res["B"] = tabB

    # ======================================================================== shared W/F graph ==
    def small_body(cfg, s, out):
        """M <= 8 layer. Prefetch never changes which bytes a consumer reads."""
        def fn():
            for k in out:
                out[k].clear()
            for i in range(nops):
                spin(a.attn_us)
                join_b()
                fork(cfg, t_gate[i])
                ar1 = rd.all_reduce(s["xa"][2 * i])
                mix = torch.empty(s["x32"].shape[0], 24, dtype=torch.float32, device=device)
                s_mhc.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_mhc):
                    torch.matmul(s["x32"], fn_w[i].t(), out=mix)
                gate = s["x16"] @ gate_w[i].t()
                torch.cuda.current_stream().wait_stream(s_mhc)
                spin(a.rest_us)
                join_b()
                fork(cfg, t_short[i])
                ar2 = rd.all_reduce(s["xa"][2 * i + 1])
                y = s["xq"] @ shard_w[i].t()
                join_b()
                if cfg["kind"] != "none":
                    burst(t_gather[i])  # gather windows stay live
                g = rd.all_gather_cols(y, 1)
                q = s["xb"] @ qb_w[i].t()
                spin(a.rest_us)
                for k, v in zip(OUT_SMALL, (ar1, mix, gate, ar2, y, g, q)):
                    out[k].append(v)
            join_all()
        return fn

    def big_body(cfg, s, out):
        """M > 8 layer (the 4-session class): one 6 MiB AR window, then the qkv_a and q_b GEMMs."""
        def fn():
            for k in out:
                out[k].clear()
            for i in range(nops):
                spin(a.attn_us)
                join_b()
                fork(cfg, t_big[i])
                ar1 = rd.all_reduce(s["xa"][i])
                y = s["x16"] @ qkv_w[i].t()
                q = s["xb"] @ qb_w[i].t()
                spin(a.rest_us)
                for k, v in zip(OUT_BIG, (ar1, y, q)):
                    out[k].append(v)
            join_all()
        return fn

    def input_sets(m, big):
        K = max(1, a.inputs)
        sets = []
        for j in range(K):
            gen = torch.Generator(device="cpu").manual_seed(77 + 1000 * m + 100003 * j)
            gen_r = torch.Generator(device="cpu").manual_seed(91 + 1000 * m + 100003 * j + 7919 * rank)
            s = {"xa": (torch.randn((1 if big else 2) * nops, m, HIDDEN, generator=gen_r) * 2).to(torch.bfloat16).to(device),
                 "x16": (torch.randn(m, HIDDEN, generator=gen)).to(torch.bfloat16).to(device),
                 "xb": (torch.randn(m, 1024, generator=gen)).to(torch.bfloat16).to(device)}
            if not big:
                s["x32"] = torch.randn(m, 20480, generator=gen).to(device)
                s["xq"] = (torch.randn(m, 2560, generator=gen)).to(torch.bfloat16).to(device)
            sets.append(s)
        return sets

    if "W" in phases or "F" in phases:
        w_rows = [int(v) for v in a.w_rows.split(",")] if "W" in phases else []
        f_rows = [int(v) for v in a.f_rows.split(",")] if "F" in phases else []
        tabW = []
        f2 = {"replays": a.bitwise_replays, "inputs": a.inputs, "mismatches": {}, "rank_digest_bad": {}, "first": None}
        for m in sorted(set(w_rows) | set(f_rows)):
            big = m > 8
            keys = OUT_BIG if big else OUT_SMALL
            sets = input_sets(m, big)
            static = {k: v.clone() for k, v in sets[0].items()}
            names = list(modes) + (list(forced) if m in f_rows else [])

            def load(j):
                for k, v in sets[j].items():
                    static[k].copy_(v)

            graphs, outs = {}, {}
            for name in names:
                out = {k: [] for k in keys}
                graphs[name] = graph_of((big_body if big else small_body)(parse_mode(name), static, out))
                outs[name] = out

            if m in f_rows:
                def views(o):
                    return [t.reshape(-1).view(torch.uint8) for k in keys for t in o[k]]

                layout, nb = [], 0
                for k in keys:
                    for i, t in enumerate(outs["none"][k]):
                        layout.append((k, i, nb, t.numel() * t.element_size()))
                        nb += t.numel() * t.element_size()
                buf = torch.empty(nb, dtype=torch.uint8, device=device)
                refs, digs = [], []
                for j in range(len(sets)):
                    load(j)
                    graphs["none"].replay()
                    torch.cuda.synchronize()
                    refs.append(torch.cat(views(outs["none"])))
                    arb = torch.cat([t.reshape(-1).view(torch.uint8) for k in keys if k.startswith("ar")
                                     for t in outs["none"][k]]).cpu().numpy().tobytes()
                    digs.append(hashlib.sha256(arb).hexdigest()[:16])
                alld = [None] * world
                dist.all_gather_object(alld, digs, group=grp)
                bad = [j for j in range(len(sets)) if len({d[j] for d in alld}) != 1]
                if bad:
                    fails += len(bad)
                    f2["rank_digest_bad"][m] = bad
                    log0(f"[F] M={m}: AR digests differ across ranks for inputs {bad}: FAIL")
                mism = {k: 0 for k in names}
                for r in range(a.bitwise_replays):
                    j = r % len(sets)
                    for k in names[r % len(names):] + names[:r % len(names)]:
                        load(j)
                        graphs[k].replay()
                        torch.cat(views(outs[k]), out=buf)
                        if not torch.equal(buf, refs[j]):
                            mism[k] += 1
                            if f2["first"] is None:
                                d = {}
                                for key, i, o, n in layout:
                                    if not torch.equal(buf[o:o + n], refs[j][o:o + n]):
                                        d.setdefault(key, []).append(i)
                                f2["first"] = dict(rank=rank, M=m, mode=k, replay=r, input=j,
                                                   differs={kk: v[:8] for kk, v in d.items()})
                                print(f"[F] rank {rank} M={m} {k} replay {r}: differs {f2['first']['differs']}", flush=True)
                fails += sum(mism.values())
                f2["mismatches"][m] = mism
                log0(f"[F] M={m}: {a.bitwise_replays} replays x {len(names)} modes vs none: "
                     + ("0 mismatches (rank 0)" if not sum(mism.values()) else "MISMATCH " + json.dumps(mism)))
                del buf, refs

            if m in w_rows:
                load(0)
                torch.cuda.synchronize()
                r = abba({k: graphs[k] for k in modes}, nops)
                base = r["live"][0]
                for k in modes:
                    us, pt, allv = r[k]
                    tabW.append(dict(M=m, mode=k, us_layer=us, vs_live=us - base, ar_kernel=pt[3], all=allv))
                log0(f"[W] M={m}: " + " | ".join(f"{k} {r[k][0]:.1f} ({r[k][0] - base:+.1f})" for k in modes))
            del graphs, outs, sets, static
            torch.cuda.empty_cache()
        res["W"] = tabW
        res["F"] = f2

    clock("end")
    res["failures"] = fails
    allf = [None] * world
    dist.all_gather_object(allf, fails, group=grp)
    res["failures_all_ranks"] = allf
    with open(f"{a.out}.rank{rank}.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    ok = not any(allf)
    log0(f"DONE failures per rank: {allf} -> {'PASS' if ok else 'FAIL'}")
    dist.barrier(group=grp)
    os._exit(0 if ok else 1)  # proxy thread / QPs are left to process teardown


if __name__ == "__main__":
    sys.exit(main())
