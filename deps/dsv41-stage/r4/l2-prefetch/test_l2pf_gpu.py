#!/usr/bin/env python3
"""r4 l2-prefetch GPU unit test + microbench (ONE GPU, ~1.5 GB, ~1 min; run only when GPU jobs are allowed,
ideally with the dsv41 service stopped on that host).

  source ~/venvs/vllm032/bin/activate
  DSV41_L2_PREFETCH=1 DSV41_L2_PREFETCH_LIB=~/dsv41-stage/r4/l2-prefetch/libdsv41_l2pf.so \
    python3 ~/dsv41-stage/r4/l2-prefetch/test_l2pf_gpu.py

Part A (bit-exact + capture): a toy 40-"layer" decoder whose layers look like the DSv4.1 decode step
  [fake AR (spin kernel, 8 CTAs, 25 us) -> mhc-like fp32 fn read (1.97 MB) -> router bf16 (384x5120)
   -> shared w13 bf16 (576x5120) -> fake AR -> qkv_a bf16 (1792x5120) -> q_b (4096x1280) -> o_b (5120x1024)]
  is learned eagerly, then CUDA-graph captured with the prefetch branches, and replayed. Outputs with
  DSV41 prefetch on/off must be bitwise equal (prefetch is only a cache hint). Checks the plan:
  collectives learned == collectives at capture, windows > 0, side-stream branch joined.
Part B (timing): graph replay time per step, prefetch off vs on, M = 1, 6, 24, 96 (rows).
  Expected on GB10: on < off by ~4-10 us per window at small M (the following GEMM's first ~6 MB
  come from L2). The real step has 85 windows (+8 in the drafter).
Exit 0 = PASS (bitwise equal, plan sane). Timing is reported, not gated.
"""
import os
import statistics
import sys
import types

os.environ.setdefault("DSV41_L2_PREFETCH", "1")
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# the module imports vllm.logger only; give it a stub when run outside the venv's vllm
try:
    import vllm.logger  # noqa: F401
except Exception:  # noqa: BLE001
    import logging
    vl = types.ModuleType("vllm.logger")
    vl.init_logger = logging.getLogger
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules["vllm.logger"] = vl

import dsv41_l2_prefetch as L  # noqa: E402  (staged copy next to this test)

assert L.ENABLED, "set DSV41_L2_PREFETCH=1"
assert L.init(), "native lib failed to load"
L.S.installed = True  # no vLLM model module here: bracket the toy class directly
lib = L._load_lib()
L._side_stream()  # as install_model_module does, outside any capture
dev = torch.device("cuda")
torch.manual_seed(0)
SPIN_NS = int(os.environ.get("SPIN_NS", "25000"))
N_LAYERS = int(os.environ.get("N_LAYERS", "40"))


def fake_ar(x):
    """Stands in for the RDMA all-reduce: same hook call, then an 8-CTA spin of SPIN_NS on the stream."""
    L.before_collective("ar")
    rc = lib.dsv41_spin(SPIN_NS, 8, torch.cuda.current_stream().cuda_stream)
    assert rc == 0
    return x


class Lin(torch.nn.Module):
    def __init__(self, n, k, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter((torch.randn(n, k, device=dev) * 0.02).to(dtype), requires_grad=False)

    def forward(self, x):
        return x @ self.weight.t()


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hc_fn = torch.nn.Parameter(torch.randn(24, 20480, device=dev) * 0.01, requires_grad=False)
        self.gate = Lin(384, 5120)
        self.w13 = Lin(576, 5120)
        self.w2 = Lin(5120, 288)
        self.qkv_a = Lin(1792, 5120)
        self.q_b = Lin(4096, 1280)
        self.o_b = Lin(5120, 1024)

    def forward(self, x):
        x = fake_ar(x)
        L._record(("hc_fn", (self.hc_fn,)))  # what the mHC wrapper records in vLLM
        mix = (x.float().repeat(1, 4) @ self.hc_fn.t()).sum(-1, keepdim=True)
        g = self.gate(x)
        h = self.w13(x)[:, :288]
        x = x + self.w2(torch.nn.functional.silu(h)) + (g[:, :1] + mix).to(x.dtype) * 1e-3
        x = fake_ar(x)
        a = self.qkv_a(x)
        q = self.q_b(a[:, :1280])
        x = x + self.o_b(q[:, :1024])
        return x


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(Layer() for _ in range(N_LAYERS))

    def forward(self, input_ids, positions):
        x = input_ids
        for layer in self.layers:
            x = layer(x)
        return fake_ar(x)


L.bracket_class(Toy)
model = Toy()


def run_case(m):
    x = torch.randn(m, 5120, device=dev, dtype=torch.bfloat16)
    pos = torch.arange(m, device=dev)
    # eager learn (as vLLM's warm-up before each capture)
    L.S.disabled = False
    y_eager = model(x, pos)
    sig = (id(model), m)
    plan = L.S.plans.get(sig)
    assert plan is not None, "no plan learned"
    n_coll = plan["n"]
    assert n_coll == 2 * N_LAYERS + 1, f"collectives learned {n_coll}"
    assert len(plan["win"]) >= 2 * N_LAYERS, f"windows {len(plan['win'])}"
    graphs = {}
    for on in (False, True):
        L.S.disabled = not on
        L.S.frozen.discard(sig)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            model(x, pos)  # warm-up on the capture stream (learns again when on)
        torch.cuda.current_stream().wait_stream(s)
        forks0 = L.S.forks
        with torch.cuda.graph(g):
            y = model(x, pos)
        if on:
            assert sig in L.S.frozen, "capture did not freeze the plan"
            assert L.S.pending is None, "side stream left open"
        graphs[on] = (g, y)
    outs = {}
    for on, (g, y) in graphs.items():
        g.replay()
        torch.cuda.synchronize()
        outs[on] = y.clone()
    exact = torch.equal(outs[False], outs[True])
    print(f"  (info) graph == eager: {torch.equal(outs[False], y_eager)}")
    # timing
    res = {}
    for on, (g, _) in graphs.items():
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        res[on] = statistics.median(ts)
    return exact, n_coll, len(plan["win"]), plan["bytes"], res


ok = True
for m in [int(v) for v in os.environ.get("MS", "1,6,24,96").split(",")]:
    exact, n, nw, nb, res = run_case(m)
    ok &= exact
    d = (res[False] - res[True]) * 1e3
    print(f"M={m:3d} exact={exact} collectives={n} windows={nw} prefetch={nb / 2**20:.1f} MB/fwd "
          f"graph off {res[False]:.3f} ms on {res[True]:.3f} ms  saved {d:.1f} us total, "
          f"{d / max(1, nw):.2f} us/window", flush=True)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
