#!/usr/bin/env python3
"""g2 colltrim (a) CPU test of the Python glue: no GPU, no CUDA context (CUDA_VISIBLE_DEVICES="").

  cd ~/dsv41-stage/g2/colltrim && CUDA_VISIBLE_DEVICES= nice -n 19 ~/venvs/vllm032/bin/python tests/test_cols_logic_cpu.py

Drives the STAGED cuda_communicator.CudaCommunicator.all_gather and rdma_all_reduce.RdmaAllReduce
(.all_gather_cols / .all_gather_stacked) against a fake extension that emulates the kernels on CPU tensors
(8 "peers" with distinct data; all_gather_into = stacked copy, all_gather_cols = the column layout the CUDA
kernel writes). Gates (exit 1 on failure):
  1 with DSV41_RDMA_AG_COLS on, every dim > 0 gather whose seg (elem * prod(shape[dim:])) is a multiple of
    16 B takes the column path exactly once and returns torch.cat(peers, dim) (bytes), contiguous;
  2 dim 0, seg % 16 != 0, or ag_cols off -> today's stacked path, same result;
  3 a declined column call (None) has no side effect: no audit, no L2 prefetch hook, no extension call;
  4 the L2 prefetch before/after hooks fire exactly once per gather on both paths (one "c" per collective).
"""
import importlib.machinery
import importlib.util
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["DSV41_L2_PREFETCH"] = "0"
os.environ["DSV41_RDMA_AG_COLS"] = "1"
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SITE = os.path.join(ROOT, "site", "vllm", "distributed", "device_communicators")
WORLD = 8
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL " + msg, flush=True)
    return bool(cond)


def load_as(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


W = load_as("g2ct_rdma_all_reduce_cpu", os.path.join(SITE, "rdma_all_reduce.py"))
CC = load_as("vllm.distributed.device_communicators.g2ct_cuda_communicator_cpu", os.path.join(SITE, "cuda_communicator.py"))
check(W._ag_cols_requested() and W._ext_path() == W._G2_EXT, "AG_COLS=1 selects the g2 build path")
os.environ["RDMA_AR_EXT"] = "/x/explicit.so"
check(W._ext_path() == "/x/explicit.so", "explicit RDMA_AR_EXT still wins")
del os.environ["RDMA_AR_EXT"]

CALLS = []


class FakeH:
    """emulates the kernels: peers[r] is rank r's input (this rank = RANK uses x itself)."""
    RANK = 3

    def __init__(self):
        self.peers = None

    def _peers(self, x):
        ps = [p.clone() for p in self.peers]
        ps[self.RANK] = x
        return ps

    def all_gather_into(self, out, x):
        CALLS.append("stacked")
        for r, p in enumerate(self._peers(x)):
            out[r].copy_(p)

    def all_gather_cols(self, out, x, seg_bytes):
        CALLS.append(("cols", seg_bytes))
        ps = self._peers(x)
        nb = x.numel() * x.element_size()
        assert seg_bytes % 16 == 0 and nb % seg_bytes == 0
        outer = nb // seg_bytes
        ob = out.view(torch.uint8).view(outer, WORLD * seg_bytes)
        for r, p in enumerate(ps):  # the CUDA kernel's destination rule, byte for byte
            ob[:, r * seg_bytes:(r + 1) * seg_bytes] = p.contiguous().view(torch.uint8).view(outer, seg_bytes)


class FakeL2:
    def __init__(self):
        self.ev = []

    def before_collective(self, kind):
        self.ev.append(("before", kind))

    def after_collective(self, kind):
        self.ev.append(("after", kind))


def make_rd(ag_cols, l2=None):
    rd = object.__new__(W.RdmaAllReduce)
    rd.disabled, rd.world, rd.rank = False, WORLD, FakeH.RANK
    rd.max_bytes = rd.max_reduce_bytes = 2 * 1024 * 1024
    rd.use_gather, rd.allow_fp32, rd.ag_cols = True, True, ag_cols
    rd._audit_left, rd._streams_seen, rd._xs_last = 0, set(), None
    rd.h = FakeH()
    rd._audit = lambda *a, **k: CALLS.append("audit")
    rd.should_gather = lambda x: x.is_contiguous() and 0 < x.numel() * x.element_size() <= rd.max_bytes \
        and (x.numel() * x.element_size()) % 16 == 0
    W._L2PF = l2
    W._L2PF_AFTER = l2.after_collective if l2 is not None else None
    return rd


CASES = [((6, 224), -1, torch.bfloat16, True), ((1, 224), 1, torch.bfloat16, True),
         ((48, 16160), -1, torch.bfloat16, True), ((6, 3, 256), 1, torch.bfloat16, True),
         ((6, 3, 512), 2, torch.bfloat16, True), ((5, 8), 1, torch.bfloat16, True),
         ((6, 640), 1, torch.float32, True), ((6, 4), -1, torch.int32, True),
         ((6, 2), -1, torch.float32, False),   # seg 8 B -> today's path
         ((6, 12), 1, torch.bfloat16, False),  # seg 24 B -> today's path
         ((6, 224), 0, torch.bfloat16, False),  # dim 0 -> today's path
         ((4, 6, 40), 1, torch.bfloat16, True)]
n = 0
for shape, dim, dtype, want_cols in CASES:
    g = torch.Generator().manual_seed(n)
    peers = [(torch.randn(shape, generator=g) * 3).to(dtype) if dtype.is_floating_point
             else torch.randint(-1000, 1000, shape, generator=g).to(dtype) for _ in range(WORLD)]
    d = dim if dim >= 0 else dim + len(shape)
    ref = torch.cat(peers, d)
    for cols_on in (True, False):
        l2 = FakeL2()
        rd = make_rd(cols_on, l2)
        rd.h.peers = peers
        comm = types.SimpleNamespace(rdma_comm=rd, world_size=WORLD, pynccl_comm=None)
        CALLS.clear()
        out = CC.CudaCommunicator.all_gather(comm, peers[FakeH.RANK].clone(), dim)
        used_cols = any(isinstance(c, tuple) and c[0] == "cols" for c in CALLS)
        exp_cols = want_cols and cols_on
        n += 1
        check(torch.equal(out.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)),
              f"{shape} dim{dim} {dtype} cols={cols_on}: result != torch.cat(peers, dim)")
        check(out.is_contiguous() or not exp_cols, f"{shape} dim{dim}: column result not contiguous")
        check(used_cols == exp_cols, f"{shape} dim{dim} {dtype} cols={cols_on}: column path used={used_cols}, expected {exp_cols}")
        n_ext = sum(1 for c in CALLS if c == "stacked" or isinstance(c, tuple))
        check(n_ext == 1, f"{shape} dim{dim} cols={cols_on}: {n_ext} extension calls (want exactly 1)")
        check(CALLS.count("audit") == 1, f"{shape} dim{dim} cols={cols_on}: {CALLS.count('audit')} audits (want 1)")
        check(l2.ev == [("before", "ag"), ("after", "ag")],
              f"{shape} dim{dim} cols={cols_on}: prefetch hooks {l2.ev} (want one before + one after)")
        if exp_cols:
            seg = [c for c in CALLS if isinstance(c, tuple)][0][1]
            esz = torch.tensor([], dtype=dtype).element_size()
            exp_seg = esz
            for s in shape[d:]:
                exp_seg *= s
            check(seg == exp_seg, f"{shape} dim{dim}: seg_bytes {seg} != {exp_seg}")

# a declined call has no side effect
l2 = FakeL2()
rd = make_rd(True, l2)
rd.h.peers = [torch.zeros(6, 2) for _ in range(WORLD)]
CALLS.clear()
r = rd.all_gather_cols(torch.zeros(6, 2), 1)
check(r is None and CALLS == [] and l2.ev == [], "declined column call had side effects")
r = rd.all_gather_cols(torch.zeros(6, 224, dtype=torch.bfloat16), 0)
check(r is None and CALLS == [] and l2.ev == [], "dim-0 column call not declined cleanly")
rd.ag_cols = False
r = rd.all_gather_cols(torch.zeros(6, 224, dtype=torch.bfloat16), 1)
check(r is None and CALLS == [] and l2.ev == [], "ag_cols off: column call not declined cleanly")
print(f"{n} gather calls checked", flush=True)
print("RESULT", "PASS" if not FAILS else f"FAIL ({len(FAILS)})", flush=True)
sys.exit(1 if FAILS else 0)
