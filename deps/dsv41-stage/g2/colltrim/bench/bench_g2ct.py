#!/usr/bin/env python3
"""bench_g2ct.py (g2 colltrim): 8-rank OFFLINE bench, real RDMA fabric, service STOPPED on all 8.
Started on all 8 Sparks by ops/run-bench8.sh (from the Mac): one process per Spark, env RANK, WORLD_SIZE,
MASTER_ADDR, MASTER_PORT. Uses ONLY staged files (site/ copies of rdma_all_reduce.py + cuda_communicator.py,
build/rdma_ar_ext.so, build/libdsv41_l2pf_g2.so) plus the live burst library; nothing is installed.
RDMA_AR_TIMING=1 is forced (the .so reads it at construction; same probe cost in every arm).

Phase A, column all-gather (DSV41_RDMA_AG_COLS), through the STAGED CudaCommunicator.all_gather method:
  A1 eager, parload and legacy kernels: cols path == today's path (stacked + movedim/reshape) == PyNccl
     all_gather along the same dim, as raw bytes; qkv_a (M,224), logits (M,16160), Engram (M,3,256),
     3-D dim 2, fp32, and the fallbacks (seg % 16 != 0 -> today's path, dim 0 untouched); the output sha
     is identical on all 8 ranks.
  A2 CUDA graph: 40 column gathers (M 1..8 x 224) + 2 all-reduces, --replays replays with fresh inputs,
     every output checked against PyNccl.
  A3 timing, ABBA x --trials: a decode-step-like graph of 40 x [all-reduce (M x 5120, served mode) +
     qkv_a gather (M,224)], today's path vs cols, M in --rows-a; plus 10 x logits gather (M,16160).
Phase B, prefetch stretch (L2 prefetch pacing): a graph of 40 x [fork prefetch of a cold 6 MB (or 1.6 MB)
  weight window on a side stream -> collective -> --gap-us spin (the compute until the next collective)],
  burst side joined at the next collective (as live), paced side joined at the end. Collectives: all-reduce rows in --rows-b (two-shot >= 32 KB as served) and the
  qkv_a gather (6,224). Configs: none, burst 1.6 MB, burst 6 MB (live kernel), paced 6 MB at --rates GB/s,
  paced 100 GB/s with a 10 us delay, paced 100 GB/s with a 1 MB burst head. Reports us/op and the
  in-kernel stage / net / tail / total (h.stats) per config.
Phase W, window emulation (what a re-plan / pacing arm buys per layer), per op i with distinct cold weights:
  W1 post-attention window: [30 us "attention" spin] -> AR(M) -> mhc-like fp32 GEMM [M,20480]x[20480,24]
     (1.97 MB) -> router-like bf16 GEMM [M,5120]x[5120,384] (3.93 MB). Modes: none | live (burst before the
     AR) | pace R + 2 MB head | attn (burst forked before the attention spin = REPLAN=attn) | attn paced R.
  W2 qkv / q_b window: AR(M) -> qkv_a-like GEMM [M,2560]x[2560,224] (1.15 MB) -> AG (M,224) cols ->
     q_b-like GEMM [M,1024]x[1024,2560] (5.24 MB). Modes: none | live (qkv_a in the AR window, q_b in the
     AG window) | qbprev | qbpost | qbpost paced R | ag0 (no AG window).
  us per op = per layer; x40 ~ per decode step.
Writes <out>.rank<R>.json; rank 0 prints the tables. Exit status 0 only if every bitwise check passed on
every rank.
"""
import argparse
import ctypes
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import statistics
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SITE = os.path.join(ROOT, "site", "vllm", "distributed", "device_communicators")
# env for the staged wrapper, BEFORE it is loaded (it reads env at import / construction)
os.environ["DSV41_RDMA_AR"] = "1"
os.environ["DSV41_RDMA_AG_COLS"] = "1"
os.environ.setdefault("DSV41_RDMA_PARLOAD", "1")
os.environ["DSV41_L2_PREFETCH"] = "0"  # the bench drives the prefetch kernels itself
os.environ["RDMA_AR_EXT"] = os.environ.get("G2CT_EXT", os.path.join(ROOT, "build", "rdma_ar_ext.so"))
os.environ["RDMA_AR_TIMING"] = "1"
os.environ.setdefault("RDMA_AR_MAX_REDUCE_BYTES", "2097152")
os.environ["RDMA_AR_FINGERPRINT"] = "1"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

HIDDEN = 5120
MB = 1 << 20
CH = 16384
LIVE_L2PF = os.path.expanduser("~/dsv41-stage/r4/l2-prefetch/libdsv41_l2pf.so")
G2_L2PF = os.environ.get("G2CT_L2PF_LIB", os.path.join(ROOT, "build", "libdsv41_l2pf_g2.so"))


def log0(*a):
    if dist.get_rank() == 0:
        print(*a, flush=True)


def load_as(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def same_on_all_ranks(h, group):
    hs = [None] * dist.get_world_size()
    dist.all_gather_object(hs, h, group=group)
    return all(x == hs[0] for x in hs)


class Nccl:
    def __init__(self, group, device):
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        self.c = PyNcclCommunicator(group=group, device=device)
        assert not self.c.disabled, "PyNccl unavailable"

    def all_gather_dim(self, x, dim, world):
        out = torch.empty((world,) + tuple(x.shape), dtype=x.dtype, device=x.device)
        self.c.all_gather(out.view(-1), x.contiguous().view(-1), stream=torch.cuda.current_stream())
        return out.movedim(0, dim).reshape(tuple(x.shape[:dim]) + (world * x.shape[dim],) + tuple(x.shape[dim + 1:]))

    def all_reduce(self, x):
        return self.c.all_reduce(x, stream=torch.cuda.current_stream())


def rand(shape, dtype, rank, salt, device):
    g = torch.Generator(device="cpu").manual_seed(1000003 * rank + 7919 * hash(tuple(shape)) % 1000003 + salt)
    if dtype.is_floating_point:
        return (torch.randn(shape, generator=g) * 2).to(dtype).to(device)
    return torch.randint(-2**31, 2**31 - 1, shape, generator=g, dtype=torch.int64).to(dtype).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phases", default="A,B,W")
    ap.add_argument("--rows-a", default="1,2,4,6,8")
    ap.add_argument("--rows-b", default="1,6,48")
    ap.add_argument("--rates", default="50,100,150,200")
    ap.add_argument("--w-rows", default="1,6")
    ap.add_argument("--w-rate", type=float, default=100.0)
    ap.add_argument("--n-ops", type=int, default=40)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--attn-us", type=float, default=30.0)
    ap.add_argument("--gap-us", type=float, default=100.0,
                    help="phase B: main-stream spin after each collective (the compute between collectives)")
    a = ap.parse_args()
    phases = set(a.phases.split(","))

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    grp = dist.group.WORLD
    res = {"rank": rank, "world": world, "env": {k: v for k, v in os.environ.items()
                                                 if k.startswith(("RDMA_AR", "DSV41_", "NCCL_", "G2CT"))}}
    nccl = Nccl(grp, device)
    W = load_as("g2ct_rdma_all_reduce", os.path.join(SITE, "rdma_all_reduce.py"))
    # a submodule name, so the file's relative imports resolve inside the installed package
    CC = load_as("vllm.distributed.device_communicators.g2ct_cuda_communicator",
                 os.path.join(SITE, "cuda_communicator.py"))
    rd = W.RdmaAllReduce(grp, rank, world, device)
    assert not rd.disabled, "rdma_ar failed to initialise (see log)"
    assert rd.ag_cols, "column gathers not enabled (wrong .so?)"
    res["variant"] = (getattr(W._EXT, "K1_VARIANT", None), getattr(W._EXT, "G2_VARIANT", None))
    res["pingpong_ns"] = rd.h.pingpong(2000)
    comm = types.SimpleNamespace(rdma_comm=rd, world_size=world, pynccl_comm=None)

    def gather(x, dim, cols=True):
        rd.ag_cols = cols
        try:
            return CC.CudaCommunicator.all_gather(comm, x, dim)
        finally:
            rd.ag_cols = True

    log0(f"init ok: ext={W._EXT_PATH} variant={res['variant']} parload={rd.h.parload()} "
         f"max_reduce={rd.max_reduce_bytes} pingpong={res['pingpong_ns']:.0f} ns")
    for _ in range(3):
        nccl.all_reduce(torch.ones(16, HIDDEN, dtype=torch.bfloat16, device=device))
    torch.cuda.synchronize()
    fails = 0

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
        """graphs: {name: graph}; rotated order per trial; returns {name: (median us/op, median parts)}."""
        names = list(graphs)
        per = {k: [] for k in names}
        parts = {k: [] for k in names}
        for t in range(a.trials):
            order = names[t % len(names):] + names[:t % len(names)]
            for k in order + order[::-1]:
                us, pt = time_graph(graphs[k], n_ops)
                per[k].append(us)
                parts[k].append(pt)
        return {k: (statistics.median(per[k]), [statistics.median([p[i] for p in parts[k]]) for i in range(4)],
                    per[k]) for k in names}

    # =================================================================================== phase A ==
    if "A" in phases:
        cases = [((1, 224), 1, torch.bfloat16), ((2, 224), 1, torch.bfloat16), ((4, 224), 1, torch.bfloat16),
                 ((6, 224), 1, torch.bfloat16), ((8, 224), 1, torch.bfloat16), ((6, 16160), 1, torch.bfloat16),
                 ((48, 16160), 1, torch.bfloat16), ((6, 3, 256), 1, torch.bfloat16),
                 ((192, 3, 256), 1, torch.bfloat16), ((6, 3, 512), 2, torch.bfloat16),
                 ((6, 640), 1, torch.float32), ((5, 8), 1, torch.bfloat16), ((6, 4), 1, torch.int32),
                 # fallbacks: seg % 16 != 0 (today's path), dim 0 (untouched)
                 ((6, 2), 1, torch.float32), ((6, 12), 1, torch.bfloat16), ((6, 224), 0, torch.bfloat16)]
        acc = []
        for shape, dim, dtype in cases:
            x = rand(shape, dtype, rank, 11, device)
            if not rd.should_gather(x):
                acc.append(dict(shape=shape, dim=dim, dtype=str(dtype), routed=False))
                continue
            ref = nccl.all_gather_dim(x, dim, world)
            for par in (True, False):
                rd.h.set_parload(par)
                old = gather(x, dim, cols=False)
                new = gather(x, dim, cols=True)
                torch.cuda.synchronize()
                b_ref = ref.contiguous().view(torch.uint8)
                ok_old = bool(torch.equal(old.contiguous().view(torch.uint8), b_ref))
                ok_new = bool(torch.equal(new.contiguous().view(torch.uint8), b_ref)) and new.is_contiguous() \
                    and tuple(new.shape) == tuple(ref.shape)
                allsame = same_on_all_ranks(sha(new), grp)
                ok = ok_old and ok_new and allsame
                fails += 0 if ok else 1
                acc.append(dict(shape=shape, dim=dim, dtype=str(dtype), parload=par, today_eq_ref=ok_old,
                                cols_eq_ref=ok_new, same_all_ranks=allsame))
        rd.h.set_parload(True)
        res["A1"] = acc
        nr = [c for c in acc if c.get("parload") is not None]
        log0(f"[A1] eager: {sum(1 for c in nr if c['today_eq_ref'] and c['cols_eq_ref'] and c['same_all_ranks'])}"
             f"/{len(nr)} routed cases bitwise (cols == today == PyNccl, identical on all ranks)")

        # A2 graph correctness
        Ms = [1 + (i % 8) for i in range(40)]
        ins = [torch.zeros((m, 224), dtype=torch.bfloat16, device=device) for m in Ms]
        ar_in = [torch.zeros((r, HIDDEN), dtype=torch.bfloat16, device=device) for r in (1, 6)]
        holder = {}

        def a2():
            outs = [gather(x, 1) for x in ins]
            ars = [rd.all_reduce(t) for t in ar_in]
            holder["o"], holder["ar"] = outs, ars

        g = graph_of(a2)
        gfail = 0
        for rep in range(a.replays):
            for i, (x, m) in enumerate(zip(ins, Ms)):
                x.copy_(rand((m, 224), torch.bfloat16, rank, 500 + 97 * rep + i, device))
            for t in ar_in:
                t.copy_(rand(tuple(t.shape), torch.bfloat16, rank, 700 + rep, device))
            g.replay()
            torch.cuda.synchronize()
            for x, o in zip(ins, holder["o"]):
                if not torch.equal(o.view(torch.int16), nccl.all_gather_dim(x, 1, world).view(torch.int16)):
                    gfail += 1
        fails += gfail
        res["A2"] = dict(replays=a.replays, failures=gfail)
        log0(f"[A2] graph: {a.replays} replays x 40 column gathers, {gfail} mismatches on rank 0")
        del g

        # A3 timing
        lat = []
        for m in map(int, a.rows_a.split(",")):
            xa = rand((m, HIDDEN), torch.bfloat16, rank, 21, device)
            xg = [rand((m, 224), torch.bfloat16, rank, 22 + i, device) for i in range(a.n_ops)]

            def step(cols, xa=xa, xg=xg):
                for x in xg:
                    rd.all_reduce(xa)
                    gather(x, 1, cols=cols)

            r = abba({"today": graph_of(lambda: step(False)), "cols": graph_of(lambda: step(True))}, a.n_ops)
            d = r["today"][0] - r["cols"][0]
            lat.append(dict(case=f"AR+AG M={m}", today_us_op=r["today"][0], cols_us_op=r["cols"][0], saved_us_per_gather=d,
                            today_all=r["today"][2], cols_all=r["cols"][2]))
            log0(f"[A3] step-like AR+AG M={m}: today {r['today'][0]:6.2f} us/pair, cols {r['cols'][0]:6.2f} "
                 f"-> {d:+5.2f} us per gather ({d * 40:+6.1f} us per 40-gather step)")
        for m in (2, 6, 12):
            xl = [rand((m, 16160), torch.bfloat16, rank, 31 + i, device) for i in range(10)]

            def lg(cols, xl=xl):
                for x in xl:
                    gather(x, 1, cols=cols)

            if not rd.should_gather(xl[0]):
                continue
            r = abba({"today": graph_of(lambda: lg(False)), "cols": graph_of(lambda: lg(True))}, 10)
            d = r["today"][0] - r["cols"][0]
            lat.append(dict(case=f"logits M={m}", today_us_op=r["today"][0], cols_us_op=r["cols"][0], saved_us_per_gather=d))
            log0(f"[A3] logits gather (M={m},16160): today {r['today'][0]:6.1f} us, cols {r['cols'][0]:6.1f} us -> {d:+5.1f} us")
        res["A3"] = lat

    # =================================================================================== phase B ==
    live = ctypes.CDLL(LIVE_L2PF) if os.path.exists(LIVE_L2PF) else ctypes.CDLL(G2_L2PF)
    live.dsv41_l2pf.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p]
    live.dsv41_l2pf.restype = ctypes.c_int
    g2 = ctypes.CDLL(G2_L2PF)
    g2.dsv41_l2pf_paced.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong,
                                    ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p]
    g2.dsv41_l2pf_paced.restype = ctypes.c_int
    side, side2, side3 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    pend = {"b": None, "a": None, "p": None}
    spin_lib = live if hasattr(live, "dsv41_spin") else g2
    spin_lib.dsv41_spin.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p]
    spin_lib.dsv41_spin.restype = ctypes.c_int

    def spin(us):
        if us > 0:
            assert spin_lib.dsv41_spin(int(us * 1000), 8, torch.cuda.current_stream().cuda_stream) == 0

    def join_b():
        if pend["b"] is not None:
            torch.cuda.current_stream().wait_stream(pend["b"])
            pend["b"] = None

    def join_all():
        join_b()
        for k in ("a", "p"):
            if pend[k] is not None:
                torch.cuda.current_stream().wait_stream(pend[k])
                pend[k] = None

    def burst_attn(tab, n):
        """g2 attn fork: burst on the WOA-like stream, joined only at the end of the forward."""
        side2.wait_stream(torch.cuda.current_stream())
        assert live.dsv41_l2pf(tab.data_ptr(), n, CH, side2.cuda_stream) == 0
        pend["a"] = side2

    def burst(tab, n):
        side.wait_stream(torch.cuda.current_stream())
        assert live.dsv41_l2pf(tab.data_ptr(), n, CH, side.cuda_stream) == 0
        pend["b"] = side

    def paced(tab, n, gbps, delay_us=0.0, max_us=150.0):
        side3.wait_stream(torch.cuda.current_stream())
        assert g2.dsv41_l2pf_paced(tab.data_ptr(), n, CH, int(round(CH * 1000.0 / gbps)), int(delay_us * 1000),
                                   int(max_us * 1000), None, side3.cuda_stream) == 0
        pend["p"] = side3

    def tab_of(ts, cap=None):
        """segment table over tensors (in order), optionally capped at `cap` bytes; returns (table, n, bytes)."""
        segs, left = [], cap if cap is not None else 1 << 62
        for t in ts:
            nb = min(t.numel() * t.element_size(), left) & ~15
            if nb <= 0:
                break
            segs += [t.data_ptr(), nb]
            left -= nb
        return torch.tensor(segs, dtype=torch.int64, device=device), len(segs) // 2, sum(segs[1::2])

    def split(ts, head):
        """(head table, tail table) over tensors: first `head` bytes / the rest (16-B pieces)."""
        h, tl, left = [], [], head
        for t in ts:
            p, nb = t.data_ptr(), t.numel() * t.element_size()
            k = min(nb, left) & ~15
            if k > 0:
                h += [p, k]
                left -= k
            if nb - k >= 16:
                tl += [p + k, (nb - k) & ~15]
        mk = (lambda s: (torch.tensor(s, dtype=torch.int64, device=device), len(s) // 2) if s else (None, 0))
        return mk(h), mk(tl)

    if "B" in phases:
        nops = a.n_ops
        w6 = [torch.empty(6 * MB // 2, dtype=torch.bfloat16, device=device) for _ in range(nops)]
        for t in w6:
            t.normal_()
        t6 = [tab_of([t])[:2] for t in w6]
        t16 = [tab_of([t], cap=int(1.6 * MB))[:2] for t in w6]
        hd = [split([t], 1 * MB) for t in w6]
        configs = [("none", None), ("burst1.6", ("b", t16)), ("burst6", ("b", t6))]
        configs += [(f"pace{int(r)}", ("p", t6, float(r), 0.0)) for r in map(float, a.rates.split(","))]
        configs += [("pace100d10", ("p", t6, 100.0, 10.0)), ("pace100h1M", ("h", hd, 100.0))]
        coll = [(f"ar{r}", r) for r in map(int, a.rows_b.split(","))] + [("ag6x224", None)]
        tabB = []
        for cname, rows in coll:
            if rows is not None:
                x = rand((rows, HIDDEN), torch.bfloat16, rank, 41, device)
                op = (lambda x=x: rd.all_reduce(x))
            else:
                x = rand((6, 224), torch.bfloat16, rank, 42, device)
                op = (lambda x=x: gather(x, 1))
            graphs = {}
            for name, cfg in configs:
                def fn(cfg=cfg, op=op):
                    for i in range(nops):
                        join_b()  # live: the previous window's burst is joined at the next collective
                        if cfg is not None:
                            if cfg[0] == "b":
                                burst(*cfg[1][i])
                            elif cfg[0] == "p":
                                paced(*cfg[1][i], cfg[2], cfg[3])
                            else:
                                (ht, hn), (tt, tn) = cfg[1][i]
                                if hn:
                                    burst(ht, hn)
                                if tn:
                                    paced(tt, tn, cfg[2])
                        op()
                        spin(a.gap_us)
                    join_all()
                graphs[name] = graph_of(fn)
            r = abba(graphs, nops)
            base = r["none"][0]
            kbase = r["none"][1][3]
            for name, _cfg in configs:
                us, parts, allv = r[name]
                tabB.append(dict(coll=cname, cfg=name, us_op=us - a.gap_us, stretch_us=us - base, stage=parts[0],
                                 net=parts[1], tail=parts[2], kernel_total=parts[3], kernel_stretch=parts[3] - kbase,
                                 gap_us=a.gap_us, all=allv))
                log0(f"[B] {cname:8s} {name:11s} op {us - a.gap_us:7.2f} us ({us - base:+6.2f})  in-kernel total "
                     f"{parts[3]:6.2f} ({parts[3] - kbase:+6.2f}) = stage {parts[0]:5.2f} + net {parts[1]:6.2f} "
                     f"+ tail {parts[2]:5.2f}")
            del graphs
        res["B"] = tabB
        del w6, t6, t16, hd
        torch.cuda.empty_cache()

    # =================================================================================== phase W ==
    if "W" in phases:
        nops = a.n_ops
        R = a.w_rate
        tabW = []
        def attention():
            spin(a.attn_us)

        fn_w = [torch.randn(24, 20480, device=device) * 0.01 for _ in range(nops)]          # 1.97 MB fp32
        gate_w = [(torch.randn(384, HIDDEN, device=device) * 0.01).to(torch.bfloat16) for _ in range(nops)]
        qkv_w = [(torch.randn(224, 2560, device=device) * 0.01).to(torch.bfloat16) for _ in range(nops)]
        qb_w = [(torch.randn(2560, 1024, device=device) * 0.01).to(torch.bfloat16) for _ in range(nops)]
        win1 = [tab_of([f, g_])[:2] for f, g_ in zip(fn_w, gate_w)]
        win1s = [split([f, g_], 2 * MB) for f, g_ in zip(fn_w, gate_w)]
        winq = [tab_of([q])[:2] for q in qkv_w]
        winb = [tab_of([b])[:2] for b in qb_w]
        winqb = [tab_of([q, b])[:2] for q, b in zip(qkv_w, qb_w)]
        for m in map(int, a.w_rows.split(",")):
            xa = rand((m, HIDDEN), torch.bfloat16, rank, 51, device)
            x32 = torch.randn(m, 20480, device=device)
            x16 = rand((m, HIDDEN), torch.bfloat16, rank, 52, device)
            xq = rand((m, 2560), torch.bfloat16, rank, 53, device)
            xb = rand((m, 1024), torch.bfloat16, rank, 54, device)
            sink = {}

            def w1(mode):
                """post-attention window. live: burst forked right before the AR (joined at the next
                collective); pacehead: 2 MB burst head + paced tail; attn: the same bytes forked before the
                attention stand-in on the attn stream (joined at the end); attnpace: paced from there."""
                def fn():
                    for i in range(nops):
                        if mode == "attn":
                            burst_attn(*win1[i])
                        elif mode == "attnpace":
                            paced(*win1[i], R)
                        attention()
                        join_b()
                        if mode == "live":
                            burst(*win1[i])
                        elif mode == "pacehead":
                            (ht, hn), (tt, tn) = win1s[i]
                            if hn:
                                burst(ht, hn)
                            if tn:
                                paced(tt, tn, R)
                        rd.all_reduce(xa)
                        sink["a"] = x32 @ fn_w[i].t()
                        sink["b"] = x16 @ gate_w[i].t()
                    join_all()
                return fn

            def w2(mode):
                """qkv / q_b window. live: qkv_a bytes in the AR window, q_b bytes in the AG window;
                qbprev: both in the AR window; qbpost: q_b forked right after the AR kernel (burst, joined at
                the AG); qbpostpace: the same paced; ag0: no AG window."""
                def fn():
                    for i in range(nops):
                        join_b()
                        if mode in ("live", "qbpost", "qbpostpace", "ag0"):
                            burst(*winq[i])
                        elif mode == "qbprev":
                            burst(*winqb[i])
                        rd.all_reduce(xa)
                        if mode == "qbpost":
                            burst(*winb[i])
                        elif mode == "qbpostpace":
                            paced(*winb[i], R)
                        y = xq @ qkv_w[i].t()
                        join_b()
                        if mode == "live":
                            burst(*winb[i])
                        sink["g"] = gather(y, 1)
                        sink["b"] = xb @ qb_w[i].t()
                    join_all()
                return fn

            r1 = abba({k: graph_of(w1(k)) for k in ("none", "live", "pacehead", "attn", "attnpace")}, nops)
            r2 = abba({k: graph_of(w2(k)) for k in ("none", "live", "qbprev", "qbpost", "qbpostpace", "ag0")}, nops)
            for tag, r in (("W1 post-attn", r1), ("W2 qkv/q_b", r2)):
                base = r["live"][0]
                line = []
                for k, (us, _p, allv) in r.items():
                    tabW.append(dict(window=tag, M=m, mode=k, us_layer=us, vs_live_us=us - base, all=allv))
                    line.append(f"{k} {us:6.1f} ({us - base:+5.1f})")
                log0(f"[W] {tag} M={m}: us/layer " + "  ".join(line) + f"   [x40 per step; pace R={R:.0f} GB/s]")
        res["W"] = tabW

    res["failures"] = fails
    allf = [None] * world
    dist.all_gather_object(allf, fails, group=grp)
    res["failures_all_ranks"] = allf
    with open(f"{a.out}.rank{rank}.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    log0(f"DONE failures per rank: {allf} -> {'PASS' if not any(allf) else 'FAIL'}")
    dist.barrier(group=grp)
    os._exit(0 if not any(allf) else 1)  # proxy thread / QPs are left to process teardown


if __name__ == "__main__":
    sys.exit(main())
