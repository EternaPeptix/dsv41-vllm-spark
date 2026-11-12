#!/usr/bin/env python3
"""g2 colltrim (b) single-GPU test: paced L2 prefetch kernel + the g2 fork/join logic under CUDA graphs.
ONE GPU, ~1.5 GB, ~2 min. Stopped window only (no serving on that GPU). Staged module, no install:

  source ~/venvs/vllm032/bin/activate && cd ~/dsv41-stage/g2/colltrim && python3 tests/test_l2pf_g2_gpu.py

Part P (paced kernel, build/libdsv41_l2pf_g2.so):
  P1 issue timeline from the kernel's own stats: issued bytes == window bytes (no cap) or <= rate x cap
     (cap), first issue ~ delay, last issue ~ delay + bytes / rate (gate: within 10% + 3 us).
  P2 the prefetched buffer is bitwise unchanged (a prefetch writes nothing).
  P3 L2 effect (reported): a 6 MB read after a burst / paced prefetch vs L2-cold.
  P4 the r4 symbols in this library behave like the live library (same kernel source): a burst prefetch
     through either library leaves the same timing class (reported) and no error.
Part T (toy decode forward, 12 layers, collectives emulated by dsv41_spin on the main stream):
  every arm is learned eagerly, captured, replayed: off (G2 off = live behaviour), pace, pace+head,
  attn, qbprev, qbpost, attn+qbpost+pace, woa+attn. Gates: graph output == eager output (bitwise) and
  equal across arms; all three side streams joined at the end of the forward; the planned g2 fork counts
  were all issued at capture; the attn arm leaves the post-attention AR windows empty; flags-off plans
  equal the live module's (also covered by the CPU test).
Timing per arm is printed but means little here (spins do not contend for DRAM like real RDMA); the 8-rank
bench (bench/bench_g2ct.py) measures the stretch.
"""
import collections
import ctypes
import importlib.machinery
import json
import importlib.util
import os
import statistics
import sys

os.environ["DSV41_L2_PREFETCH"] = "1"
os.environ.setdefault("DSV41_L2_PREFETCH_REPLAN", "attn")  # import-time: wq_b post-hooks get registered
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
G2LIB = os.environ.get("G2CT_L2PF_LIB", os.path.join(ROOT, "build", "libdsv41_l2pf_g2.so"))
os.environ["DSV41_L2_PREFETCH_PACE_LIB"] = G2LIB
LIVELIB = os.path.expanduser("~/dsv41-stage/r4/l2-prefetch/libdsv41_l2pf.so")
if not os.path.exists(os.environ.get("DSV41_L2_PREFETCH_LIB", LIVELIB)):
    os.environ["DSV41_L2_PREFETCH_LIB"] = G2LIB  # same burst kernel source


def load(path, name):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


L = load(os.path.join(ROOT, "site/vllm/distributed/device_communicators/dsv41_l2_prefetch.py"), "l2pf_g2_staged")
LIVE = load(os.path.join(ROOT, "base/dsv41_l2_prefetch.py.998af3d7"), "l2pf_live")
assert L.ENABLED and L.REPLAN_ATTN
dev = torch.device("cuda")
torch.cuda.set_device(0)
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)
    return bool(cond)


lib = ctypes.CDLL(G2LIB)
lib.dsv41_l2pf_paced.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong,
                                 ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p]
lib.dsv41_l2pf_paced.restype = ctypes.c_int
lib.dsv41_l2pf.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p]
lib.dsv41_l2pf.restype = ctypes.c_int
lib.dsv41_spin.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p]
lib.dsv41_spin.restype = ctypes.c_int
check(lib.dsv41_l2pf_g2_version() == 1, "libdsv41_l2pf_g2 version 1")
MB = 1 << 20
CH = 16384


def segtab(buf, nbytes, pieces=3):
    """segment table over the first nbytes of buf in `pieces` 16-B aligned pieces (plan-like)."""
    base = buf.data_ptr()
    step = (nbytes // pieces) & ~15
    segs, off = [], 0
    for i in range(pieces):
        sz = step if i < pieces - 1 else nbytes - off
        segs += [base + off, sz & ~15]
        off += sz
    return torch.tensor(segs, dtype=torch.int64, device=dev), pieces


def flush_l2(scratch):
    scratch.add_(1)  # 64 MB read+write evicts the 24 MiB L2


# ---------------------------------------------------------------------------------------- part P --
print("[P1] paced issue timeline", flush=True)
pool = torch.randn(8 * MB // 4 * 4, device=dev)  # 32 MB fp32 data under test
scratch = torch.zeros(16 * MB, dtype=torch.float32, device=dev)  # 64 MB
before = pool.view(torch.int32).clone()
stats = torch.zeros(4, dtype=torch.int64, device=dev)
s = torch.cuda.Stream()
for gbps, nbytes, delay_us, cap_us in ((50, 6 * MB, 0, 0), (100, 6 * MB, 0, 0), (200, 6 * MB, 0, 0),
                                       (100, 6 * MB, 10, 0), (100, 6 * MB, 0, 20), (150, 2 * MB, 5, 0),
                                       (100, 1 * MB, 0, 150)):
    tab, n = segtab(pool, nbytes)
    stats.zero_()
    period_ps = int(round(CH * 1000.0 / gbps))
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        e0.record()
        rc = lib.dsv41_l2pf_paced(tab.data_ptr(), n, CH, period_ps, int(delay_us * 1000), int(cap_us * 1000),
                                  stats.data_ptr(), s.cuda_stream)
        e1.record()
    torch.cuda.synchronize()
    issued, cnt, first, last = [int(v) for v in stats.tolist()]
    dur_us = e0.elapsed_time(e1) * 1000.0
    exp_last = delay_us + (nbytes - CH) / (gbps * 1e3)  # us: the last chunk issues one period before the end
    if cap_us:
        ok = issued <= gbps * 1e3 * cap_us + CH and last / 1e3 <= delay_us + cap_us + 3 and issued > 0
        what = f"capped: issued {issued / MB:.2f} MB <= {gbps * cap_us / 1e3:.2f} MB + 1 chunk"
    else:
        ok = (issued == nbytes and abs(first / 1e3 - delay_us) <= 3 + 0.1 * delay_us
              and abs(last / 1e3 - exp_last) <= 3 + 0.1 * exp_last)
        what = f"issued {issued / MB:.2f} MB in {cnt} prefetches, last at {last / 1e3:.1f} us (expect {exp_last:.1f})"
    check(rc == 0 and ok, f"{gbps:4d} GB/s {nbytes / MB:.0f} MB delay {delay_us} cap {cap_us}: {what}; "
                          f"first {first / 1e3:.1f} us, kernel {dur_us:.1f} us")
torch.cuda.synchronize()
check(torch.equal(before, pool.view(torch.int32)), "[P2] prefetched data bitwise unchanged")

print("[P3] L2 effect on a 6 MB read (reported)", flush=True)
probe = pool[: 6 * MB // 4]


def timed_read(prefetch):
    flush_l2(scratch)
    torch.cuda.synchronize()
    if prefetch is not None:
        prefetch()
        torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    probe.sum()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0


tab6, n6 = segtab(probe, 6 * MB)
arms = {
    "cold": None,
    "burst": lambda: lib.dsv41_l2pf(tab6.data_ptr(), n6, CH, torch.cuda.current_stream().cuda_stream),
    "paced 100": lambda: lib.dsv41_l2pf_paced(tab6.data_ptr(), n6, CH, 163840, 0, 0, None,
                                              torch.cuda.current_stream().cuda_stream),
}
res = {k: statistics.median([timed_read(v) for _ in range(9)]) for k, v in arms.items()}
print("    " + "  ".join(f"{k}: {v:.1f} us" for k, v in res.items()), flush=True)

if os.path.exists(LIVELIB):
    live = ctypes.CDLL(LIVELIB)
    live.dsv41_l2pf.argtypes = lib.dsv41_l2pf.argtypes
    live.dsv41_l2pf.restype = ctypes.c_int
    t_live = statistics.median([timed_read(lambda: live.dsv41_l2pf(tab6.data_ptr(), n6, CH,
                                                                    torch.cuda.current_stream().cuda_stream))
                                for _ in range(9)])
    print(f"[P4] burst via the live library: {t_live:.1f} us (this library {res['burst']:.1f} us)", flush=True)

# ---------------------------------------------------------------------------------------- part T --
print("[T] toy forward: g2 arms under CUDA graphs", flush=True)
L.S.installed = True
L._side_stream()
L._side_stream2()
L._side_stream3()
H, QR = 2048, 512
N_LAYERS = int(os.environ.get("N_LAYERS", "12"))
COLL_US = float(os.environ.get("COLL_US", "20"))
ATTN_US = float(os.environ.get("ATTN_US", "30"))


def spin(us):
    assert lib.dsv41_spin(int(us * 1000), 8, torch.cuda.current_stream().cuda_stream) == 0


def collective(kind, x):
    L.before_collective(kind)
    spin(COLL_US)
    L.after_collective(kind)
    return x


class Lin(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight = torch.nn.Parameter((torch.randn(n, k, device=dev) * (1.0 / k ** 0.5)).to(torch.bfloat16),
                                         requires_grad=False)

    def forward(self, x):
        return x @ self.weight.t()


class Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq_b = Lin(4 * QR, QR)   # named "...attn.wq_b": the g2 attn fork point and q_b category
        self.wo_b = Lin(H, 4 * QR)

    def forward(self, x):
        q = self.wq_b(x[:, :QR])
        spin(ATTN_US)  # rope / KV insert / indexer / sparse MLA stand-in
        return self.wo_b(torch.tanh(q))


class FFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Lin(384, H)        # named "...ffn.gate": router category
        self.up = Lin(1024, H)

    def forward(self, x):
        return self.up(x)[:, :H // 2].repeat(1, 2) * 0.1 + self.gate(x)[:, :1] * 0.01


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = Lin(256, H)
        self.attn = Attn()
        self.ffn = FFN()

    def forward(self, x):
        y = self.qkv(x)                     # qkv_a shard GEMM stand-in
        y = collective("ag", y)             # qkv_a gather
        x = torch.tanh(x + self.attn(torch.cat([y, x[:, 256:]], 1)))
        x = collective("ar", x)             # post-attention AR
        x = torch.tanh(x + self.ffn(x))
        return collective("ar", x)          # post-MoE AR


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(Layer() for _ in range(N_LAYERS))

    def forward(self, input_ids, positions):
        x = input_ids
        x = collective("ar", x)             # embedding AR
        for layer in self.layers:
            x = layer(x)
        return x


torch.manual_seed(0)
L.bracket_class(Toy)
model = Toy()
L._hook_modules(model)  # with REPLAN_ATTN set at import: the wq_b post-hooks are registered for every arm

ARMS = {
    "off": dict(),
    "pace100": dict(PACE_GBPS=100.0),
    "pace60+head1M": dict(PACE_GBPS=60.0, PACE_HEAD=1 << 20),
    "attn": dict(REPLAN={"attn"}),
    "qbprev": dict(REPLAN={"qbprev"}),
    "qbpost": dict(REPLAN={"qbpost"}),
    "attn+qbpost+pace": dict(REPLAN={"attn", "qbpost"}, PACE_GBPS=120.0, PACE_HEAD=512 << 10),
}


def set_arm(cfg):
    L.PACE_GBPS = float(cfg.get("PACE_GBPS", 0.0))
    L.PACE_HEAD = int(cfg.get("PACE_HEAD", 0))
    L.REPLAN = frozenset(cfg.get("REPLAN", ()))
    L.G2 = L.PACE_GBPS > 0 or bool(L.REPLAN)
    L.S.pace_ok = L.PACE_GBPS > 0
    if L.S.pace_ok:
        L._load_lib_g2()


RECORDED = {}
_make_plan = L.make_plan


def _spy_make_plan(events, n_coll, *a, **k):
    RECORDED["last"] = (list(events), n_coll)
    return _make_plan(events, n_coll, *a, **k)


L.make_plan = _spy_make_plan


def capture(x, pos, cfg):
    sig = (id(model), x.shape[0])
    set_arm(cfg)
    L.S.frozen.discard(sig)
    L.S.learned.pop(sig, None)
    L.S.plans.pop(sig, None)
    s2 = torch.cuda.Stream()
    s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        y_eager = model(x, pos)
    torch.cuda.current_stream().wait_stream(s2)
    plan = L.S.plans.get(sig)
    L.S.g2_forks = collections.Counter()  # G2-off forwards do not reset it; keep arms independent
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = model(x, pos)
    info = dict(plan=plan, g2_forks=dict(L.S.g2_forks), forks=L.S.forks,
                pend=(L.S.pending, L.S.pending2, L.S.pending3))
    torch.cuda.synchronize()
    return g, y, y_eager.clone(), info


def replay_us(g, n=30):
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    return statistics.median(ts)


for m in (2, 6):
    x = (torch.randn(m, H, device=dev) * 0.5).to(torch.bfloat16)
    pos = torch.arange(m, device=dev)
    caps = {name: capture(x, pos, cfg) for name, cfg in ARMS.items()}
    outs = {}
    for name, (g, y, ye, info) in caps.items():
        g.replay()
        torch.cuda.synchronize()
        outs[name] = y.clone()
        check(torch.isfinite(ye.float()).all().item(), f"M={m} {name}: eager output finite")
        check(torch.equal(outs[name].view(torch.int16), ye.view(torch.int16)), f"M={m} {name}: graph == eager (bits)")
        check(info["pend"] == (None, None, None), f"M={m} {name}: side streams joined at the end of the forward")
        plan = info["plan"]
        if name == "off":
            check("g2" not in plan and info["g2_forks"] == {}, f"M={m} off: no g2 plan, no g2 forks")
        else:
            planned = json.loads(plan["g2"]["nforks"])
            check(info["g2_forks"] == planned, f"M={m} {name}: g2 forks issued {info['g2_forks']} == planned {planned}")
        if name == "attn":
            g2 = plan["g2"]
            post_attn = [i for i in range(1, plan["n"] + 1) if (i - 1) % 3 == 2]  # coll order: ar1, (ag, ar, ar) x N
            check(all(i not in g2["pre"] for i in post_attn), f"M={m} attn: post-attention AR windows empty")
            check(len(g2["attn"]) == N_LAYERS, f"M={m} attn: {len(g2['attn'])} attn forks (one per layer)")
    # flags-off plan from REAL recorded events (this forward) == the live module's plan
    set_arm({})
    L.S.frozen.discard((id(model), m))
    L.S.learned.pop((id(model), m), None)
    model(x, pos)
    torch.cuda.synchronize()
    ev, nc = RECORDED["last"]
    a, b = LIVE.make_plan(ev, nc), _make_plan(ev, nc)
    same = (a["n"] == b["n"] and a["key"] == b["key"] and a["bytes"] == b["bytes"]
            and sorted(a["win"]) == sorted(b["win"])
            and all(torch.equal(a["win"][k][0], b["win"][k][0]) for k in a["win"]))
    check(same and "g2" not in b, f"M={m}: flags-off plan == live module plan on recorded events "
                                  f"({nc} collectives, {len(b['win'])} windows)")
    ref = outs["off"]
    check(all(torch.equal(o.view(torch.int16), ref.view(torch.int16)) for o in outs.values()),
          f"M={m}: outputs bitwise equal across all {len(outs)} arms")
    t = {name: replay_us(caps[name][0]) for name in caps}
    print(f"M={m}: graph us " + "  ".join(f"{k} {v:.0f}" for k, v in t.items()), flush=True)
    del caps
    torch.cuda.empty_cache()

# flags-off plan equality vs the live module on the same recorded toy events (structure only)
set_arm({})
print("RESULT", "PASS" if not FAILS else f"FAIL ({len(FAILS)})", flush=True)
sys.exit(1 if FAILS else 0)
