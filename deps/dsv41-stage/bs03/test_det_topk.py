#!/usr/bin/env python3
"""test_det_topk.py: BS03 Design B gate. Runs on ONE stopped rank, no model. Default: every stage, an estimated 30-60 min.

Compares det_topk (site/dsv41_bs03_det_topk.py + build/vllm_bs03_det_topk_sm121.so) with the live decode chain
(persistent_topk, which falls back to top_k_per_row_decode at <= 8 rows, then dsv41_topk_ties.decode_hook, then
sparse_attn_indexer._canon_topk_), as integers, inside captured CUDA graphs, and times both.

Stages (--stages, in this order):
  probe   cudaOccupancyMaxActiveClusters for C=16,8,4,2, shared memory, registers, the cluster map.
  smoke   eager and one-graph compare per cluster path, the backend the live dispatcher resolves, try_decode gating.
  timing  det_topk vs chain at rows 2,4,6,12,16 and lengths 300,1100,4096,32768,131072,524288, alone and beside a 2-CTA
          64-thread l2pf co-runner, plus the G1-mix cells (lengths 300,512,700,1150 at rows 2,4,6).
  sweep   informational: det_topk per cluster size at rows 2,4,6. Not part of the verdict.
  exact   200 replays per shape: rows {1,2,4,6,8,9,12,16,24,48,128} x widths {524288,1048576} x 14 lengths x 6 input kinds.
          The first 4 draws of a shape are also compared with the integer reference and with the chain run eagerly.

Input kinds keep >= 512 finite values per row longer than 512 (below that the chain itself depends on the row count):
  cont continuous fp32 | q18 18 significant bits | tie512 ties at the 512th value | pmzero +-0 straddling the 512th value |
  denorm a few denormals, half of the rows with the 512th value among them | masked 8-column blocks, 2048 kept above
  16384 columns (>= 16377 finite values), half of the rows tied.
Columns past each row's end hold +inf or other rows' data, so a read past the end changes the answer.

Final line: VERDICT: GO | STOP | HOLD, and the exit code: 0 GO, 1 STOP, 2 HOLD, 3 refused or harness error.
  GO    weighted det_topk <= 9 us/layer (the worse of alone and beside the co-runner), no timed shape slower than the chain,
        zero differing rows, every stage complete.
  STOP  weighted >= 13 us/layer, or any differing row.
  HOLD  anything else: 9-13 us, a slower shape (slower = det > chain * 1.02 + 0.5 us, to keep timer noise out), an
        incomplete run, or a row where both arms agree and differ from the integer reference.

Options: --quick (small grid, about 3 min, never GO) --mock (CPU harness check, no GPU, safe beside the service; never a
real result) --stages a,b --replays N --seed N --out DIR --map SPEC --smem-cap BYTES --max-minutes M --force.
The test writes only under --out (default <kit>/out/<UTC>); its Triton cache is <kit>/out/triton-cache.
"""
from __future__ import annotations

import argparse
import ctypes
import faulthandler
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

KIT = os.path.dirname(os.path.abspath(__file__))
K = 512
GO_US, STOP_US = 9.0, 13.0
SLOW_RATIO, SLOW_ABS_US = 1.02, 0.5
MAX_SEQ_LEN = 1048576  # what the decode graphs capture at --max-model-len 1048576 (attn_metadata.max_seq_len)
REPEATS = 8  # arm runs per graph replay in the timing graphs
L2PF_LIB = os.path.expanduser("~/dsv41-stage/r4/l2-prefetch/libdsv41_l2pf.so")
L2PF_CHUNK = 16384  # DSV41_L2_PREFETCH_CHUNK default
L2PF_SEG_BYTES = 3 << 20  # two segments = the 6 MB prefetch budget of one window
EXPECT_LIVE = {
    "model_executor/layers/sparse_attn_indexer.py": ("005785f9", "ebe2ba62"),
    "model_executor/layers/dsv41_topk_ties.py": ("943961bc",),
    "model_executor/layers/indexer_topk.py": ("4dc91f83",),
}
KINDS = ("cont", "q18", "tie512", "pmzero", "denorm", "masked")
TIED_VALUE = -2.0917205810546875  # a real tied 512th value (README, g5det): 18 significant bits

GRIDS = {
    "full": dict(
        # The asked-for {2,4,6,8,9,24,48,128} plus 1, 12 and 16: production rows the hook would take, and the only rows on
        # the default map's C=2 path.
        rows=(1, 2, 4, 6, 8, 9, 12, 16, 24, 48, 128),
        widths=(524288, 1048576),
        lengths=(1, 511, 512, 513, 1163, 4096, 12288, 12289, 16384, 16385, 65536, 262144, 524288, 1048576),
        replays=200,
        t_rows=(2, 4, 6, 12, 16),
        t_lengths=(300, 1100, 4096, 32768, 131072, 524288),
        g1_lengths=(300, 512, 700, 1150),
        sweep_lengths=(300, 1150, 4096, 32768, 131072, 524288),
        volume_rotate=16_000_000,
        volume_draw=32_000_000,
    ),
    "quick": dict(
        rows=(2, 6, 12, 24),
        widths=(1048576,),
        lengths=(1, 512, 513, 1163, 12288, 12289, 16385, 65536),
        replays=20,
        t_rows=(2, 4, 6),
        t_lengths=(300, 1100, 32768),
        g1_lengths=(300, 512, 700, 1150),
        sweep_lengths=(300, 32768),
        volume_rotate=16_000_000,
        volume_draw=32_000_000,
    ),
    "mock": dict(
        rows=(1, 2, 6, 12),
        widths=(32768, 65536),
        lengths=(1, 511, 512, 513, 1163, 4096, 12288, 12289, 16384, 16385, 65536),
        replays=4,
        t_rows=(2, 4, 6),
        t_lengths=(300, 4096),
        g1_lengths=(300, 512, 700, 1150),
        sweep_lengths=(300, 4096),
        volume_rotate=200_000,
        volume_draw=200_000,
    ),
}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--mock", action="store_true")
ap.add_argument("--stages", default="probe,smoke,timing,sweep,exact")
ap.add_argument("--replays", type=int, default=0)
ap.add_argument("--seed", type=int, default=20261102)
ap.add_argument("--out", default="")
ap.add_argument("--map", default="", help="cluster map, e.g. 5:8,11:4,22:2 (sets DSV41_BS03_CLUSTER_MAP)")
ap.add_argument("--smem-cap", type=int, default=0, help="dynamic shared memory per CTA (sets DSV41_BS03_SMEM_CAP)")
ap.add_argument("--max-minutes", type=float, default=100.0, help="the exact stage stops here and the verdict becomes HOLD")
ap.add_argument("--force", action="store_true", help="skip the check that the GPU is idle")
ap.add_argument("--allow-base-mismatch", action="store_true", help="run although the live files are not the expected shas")
ap.add_argument("--mock-det-us", type=float, default=6.0, help="--mock: simulated det_topk time")
ap.add_argument("--mock-inject", default="", choices=("", "mismatch", "slowshape", "oracle"), help="--mock: exercise a verdict path")
args = ap.parse_args()
if args.mock_inject and not args.mock:
    ap.error("--mock-inject needs --mock")

GRID = GRIDS["mock" if args.mock else "quick" if args.quick else "full"]
REPLAYS = args.replays or GRID["replays"]
STAGES = [s for s in args.stages.split(",") if s]
stamp_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
OUT = args.out or os.path.join(KIT, "out", stamp_utc + ("-mock" if args.mock else "-quick" if args.quick else "-full"))
os.makedirs(OUT, exist_ok=True)

if args.map:
    os.environ["DSV41_BS03_CLUSTER_MAP"] = args.map
if args.smem_cap:
    os.environ["DSV41_BS03_SMEM_CAP"] = str(args.smem_cap)
# The live chain's flags, as in ~/dsv41-tp8.sh; the hook file is imported by path and the site flag stays off.
os.environ["DSV41_TOPK_CANON"] = "1"
os.environ["DSV41_TOPK_CANON_FUSED"] = "1"
os.environ["DSV41_TOPK_CANON_TIES"] = "all"
os.environ["DSV41_BS03_DET_TOPK"] = "0"
os.environ.pop("DSV41_DET_DUMP", None)
if args.mock:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
else:
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    os.environ["VLLM_USE_DEEP_GEMM"] = "1"
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(KIT, "out", "triton-cache")

import torch  # noqa: E402

if args.mock:
    torch.set_num_threads(1)

RESULTS = open(os.path.join(OUT, "results.jsonl"), "a", buffering=1)
SHAS: dict = {}


def log(*a):
    print(*a, flush=True)


def record(kind, **kw):
    RESULTS.write(json.dumps(dict(kind=kind, **kw)) + "\n")


def file_sha(p, n=None):
    with open(p, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    return h[:n] if n else h


# ------------------------------------------------------------------------------------------- the integer reference
def int_keys(x: torch.Tensor) -> torch.Tensor:
    """Order-preserving integer keys of fp32 values, from the bits only (-0 == +0, denormals distinct). CPU."""
    u = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    u = torch.where((u & 0x7FFFFFFF) == 0, torch.zeros_like(u), u)
    return torch.where((u >> 31) != 0, (~u) & 0xFFFFFFFF, u | 0x80000000)


def f_ref_row(x: torch.Tensor, n: int) -> torch.Tensor:
    """The canonical function of one row: <= 512 columns: 0..n-1, -1 fill. Else every key above the 512th largest, then the
    lowest columns on it, ascending."""
    out = torch.full((K,), -1, dtype=torch.int32)
    n = max(int(n), 0)
    if n <= K:
        out[:n] = torch.arange(n, dtype=torch.int32)
        return out
    key = int_keys(x[:n].cpu())
    kth = key.topk(K).values[-1]
    gt = (key > kth).nonzero()[:, 0]
    eq = (key == kth).nonzero()[:, 0]
    sel = torch.cat([gt, eq[: K - gt.numel()]]).sort().values
    out[: sel.numel()] = sel.to(torch.int32)
    return out


# ---------------------------------------------------------------------------------------------------- the backends
class Stub:
    pass


class MockGraph:
    def __init__(self, fn):
        self.fn = fn

    def replay(self):
        self.fn()


class MockBackend:
    """CPU stand-ins for every GPU primitive, so all harness paths and the verdict logic run without a GPU. Both arms are the
    integer reference; the clock is simulated (chain 30 us, det_topk --mock-det-us)."""

    name = "MOCK"

    def __init__(self):
        self.dev = torch.device("cpu")
        self.clock = 1000
        self.replay_index = 0
        self.inject = None
        self.gen = torch.Generator().manual_seed(args.seed)
        self._hook = None

    def hook(self):
        # The hook's map and gating, with stubs for its two vLLM imports.
        if self._hook is not None:
            return self._hook
        stubs = {
            "vllm": Stub(), "vllm.logger": Stub(), "vllm.model_executor": Stub(), "vllm.model_executor.layers": Stub(),
            "vllm.model_executor.layers.dsv41_topk_ties": Stub(),
        }
        import logging
        stubs["vllm.logger"].init_logger = logging.getLogger
        ties = stubs["vllm.model_executor.layers.dsv41_topk_ties"]
        ties.TIES_DECODE, ties.DUMP = True, False
        stubs["vllm.model_executor.layers"].dsv41_topk_ties = ties
        saved = {k: sys.modules.get(k) for k in stubs}
        sys.modules.update(stubs)
        try:
            spec = importlib.util.spec_from_file_location("dsv41_bs03_det_topk", os.path.join(KIT, "site", "dsv41_bs03_det_topk.py"))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self._hook = m
        return m

    def chain(self, logits, seq, next_n, out, ws, max_seq_len=MAX_SEQ_LEN):
        ends = seq.reshape(-1)
        for r in range(logits.shape[0]):
            out[r] = f_ref_row(logits[r], int(ends[r]))
        self.clock += 30_000

    def det(self, logits, seq, next_n, out, cluster=None):
        ends = seq.reshape(-1)
        for r in range(logits.shape[0]):
            out[r] = f_ref_row(logits[r], int(ends[r]))
        if self.inject and self.inject(logits.shape[0], int(ends[0])):
            out[0, 0] += 1
        self.clock += int(args.mock_det_us * 1000)
        if args.mock_inject == "slowshape" and logits.shape[0] == 6 and int(ends[0]) == 4096:
            self.clock += 40_000

    def capture(self, fn):
        return MockGraph(fn)

    def stamp(self, ts, i):
        self.clock += 2000  # the stamp itself
        ts[i] = self.clock

    def fork_corun(self, i):
        pass

    def join_corun(self):
        pass

    def make_workspace(self):
        return torch.zeros(1)

    def sync(self):
        pass

    def trim(self):
        pass

    def probe(self, c):
        return dict(cluster=c, optin=100000, static_smem=19936, dyn=65536, num_regs=64, max_active_clusters=48 // c)

    def smoke_extra(self, res):
        m = self.hook()
        cpu = torch.zeros(2, 8)
        why = m._why_not(cpu, torch.zeros((1, 2), dtype=torch.int32), torch.zeros(2, 512, dtype=torch.int32), 512, "auto", None)
        ok = why == "device"
        bad = [
            m._why_not(cpu, torch.zeros((1, 2), dtype=torch.int32), torch.zeros(2, 512, dtype=torch.int32), 512, "flashinfer", None),
            m._why_not(cpu, torch.zeros((1, 2), dtype=torch.int32), torch.zeros(2, 512, dtype=torch.int32), 1024, "auto", None),
            m._why_not(cpu, torch.zeros((1, 2), dtype=torch.int32), torch.zeros(2, 512, dtype=torch.int32), 512, "auto", cpu),
            m._why_not(cpu, torch.zeros((1, 2), dtype=torch.int64), torch.zeros(2, 512, dtype=torch.int32), 512, "auto", None),
            m._why_not(cpu, torch.zeros((1, 3), dtype=torch.int32), torch.zeros(2, 512, dtype=torch.int32), 512, "auto", None),
            m._why_not(torch.zeros(129, 8), torch.zeros((1, 129), dtype=torch.int32), torch.zeros(129, 512, dtype=torch.int32), 512, "auto", None),
        ]
        ok = ok and all(b is not None and b != "device" for b in bad)
        log("  hook gating (mock):", "ok" if ok else "BAD", why, bad)
        mp = [(r, m.cluster_for(r)) for r in (1, 2, 4, 5, 6, 8, 9, 11, 12, 16, 22, 23, 24, 48, 128)]
        log("  hook cluster map:", mp)
        if not ok:
            res["harness_errors"].append("hook gating")


class GpuBackend:
    name = "GPU"

    def __init__(self):
        import vllm._custom_ops  # noqa: F401  registers torch.ops._C
        from vllm.model_executor.layers import dsv41_topk_ties as ties
        from vllm.model_executor.layers import sparse_attn_indexer as sai
        from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE, SparseIndexerTopk

        if not (ties.TIES_DECODE and sai._DSV41_TOPK_CANON and sai._DSV41_TOPK_CANON_FUSED):
            raise RuntimeError("the live chain's flags did not take effect")
        self.ties, self.sai = ties, sai
        self.ws_bytes = RADIX_TOPK_WORKSPACE_SIZE
        self.dispatcher = SparseIndexerTopk("auto")
        self.dev = torch.device("cuda", 0)
        spec = importlib.util.spec_from_file_location("dsv41_bs03_det_topk", os.path.join(KIT, "site", "dsv41_bs03_det_topk.py"))
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        if not self.m.ACTIVE:
            raise RuntimeError("the hook is not ACTIVE under the chain's flags")
        self.m.ext()
        self.gen = torch.Generator(device=self.dev).manual_seed(args.seed)
        self.stamp_lib = ctypes.CDLL(os.path.join(KIT, "build", "libbs03_stamp.so"))
        self.stamp_lib.bs03_stamp.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.stamp_lib.bs03_stamp.restype = ctypes.c_int
        self.l2pf = ctypes.CDLL(L2PF_LIB)
        self.l2pf.dsv41_l2pf.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p]
        self.l2pf.dsv41_l2pf.restype = ctypes.c_int
        self.side = torch.cuda.Stream()
        # One 6 MB window per repeat, over more memory than L2 holds (25 MB), so the prefetches keep missing.
        self.pf_buf = torch.ones(REPEATS * 2 * L2PF_SEG_BYTES, dtype=torch.uint8, device=self.dev)
        a = self.pf_buf.data_ptr()
        segs = [[a + i * 2 * L2PF_SEG_BYTES, L2PF_SEG_BYTES, a + (i * 2 + 1) * L2PF_SEG_BYTES, L2PF_SEG_BYTES] for i in range(REPEATS)]
        self.segs = torch.tensor(segs, dtype=torch.int64, device=self.dev)

    def hook(self):
        return self.m

    def chain(self, logits, seq, next_n, out, ws, max_seq_len=MAX_SEQ_LEN):
        """indexer_topk.py L339-346 (the "auto" backend on SM12x), then sparse_attn_indexer.py L1050-1075 without DCP.
        A capture sees max_seq_len = max_model_len, so at <= 8 rows persistent_topk falls back to top_k_per_row_decode.
        An eager step sees the true maximum, which takes persistent_topk's own paths: the smoke stage runs those too."""
        torch.ops._C.persistent_topk(logits, seq, out, ws, K, max_seq_len)
        self.ties.decode_hook(logits, seq, next_n, out, None, None, {}, None)
        self.sai._canon_topk_(out)

    def det(self, logits, seq, next_n, out, cluster=None):
        self.m.decode(logits, seq, next_n, out, cluster)

    def capture(self, fn):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def stamp(self, ts, i):
        rc = self.stamp_lib.bs03_stamp(ts.data_ptr() + 8 * i, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"stamp: cuda error {rc}")

    def fork_corun(self, i):
        """The l2pf fork as dsv41_l2_prefetch._fork_part does it: the side stream waits for the main stream, then runs the
        real kernel (2 segments = 2 CTAs of 64 threads, burst of bulk prefetches)."""
        self.side.wait_stream(torch.cuda.current_stream())
        rc = self.l2pf.dsv41_l2pf(self.segs.data_ptr() + i * 4 * 8, 2, L2PF_CHUNK, self.side.cuda_stream)
        if rc:
            raise RuntimeError(f"l2pf: cuda error {rc}")

    def join_corun(self):
        torch.cuda.current_stream().wait_stream(self.side)

    def make_workspace(self):
        return torch.zeros(self.ws_bytes, dtype=torch.uint8, device=self.dev)

    def sync(self):
        torch.cuda.synchronize()

    def trim(self):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def probe(self, c):
        return self.m.ext().det_topk_probe(c, 0)

    def smoke_extra(self, res):
        rows, width = 6, 1048576
        logits = torch.zeros(rows, width, device=self.dev)
        seq = torch.full((1, rows), 1163, dtype=torch.int32, device=self.dev)
        out = torch.zeros(rows, K, dtype=torch.int32, device=self.dev)
        backend = self.dispatcher.resolve_backend(logits, K, rows)
        log("  live dispatcher resolves auto to:", backend, "(expected persistent)")
        if backend != "persistent":
            res["harness_errors"].append(f"the live dispatcher resolves auto to {backend}, this test times persistent_topk")
        m = self.m
        logits[:, :1163] = torch.randn(rows, 1163, device=self.dev)
        ok = m.try_decode(logits, seq, rows, out, K, "auto", None) is True
        refused = [
            m.try_decode(logits, seq, rows, out, K, "flashinfer", None),
            m.try_decode(logits, seq, rows, out, K, "auto", seq),
            m.try_decode(logits.to(torch.float64), seq, rows, out, K, "auto", None),
            m.try_decode(logits, seq.to(torch.int64), rows, out, K, "auto", None),
            m.try_decode(logits, seq[:, :3], rows, out, K, "auto", None),
            m.try_decode(logits, seq, rows, out[:, :256], 256, "auto", None),
        ]
        ok = ok and not any(refused)
        log("  hook gating:", "ok" if ok else "BAD", refused)
        log("  hook cluster map:", [(r, m.cluster_for(r)) for r in (1, 2, 4, 5, 6, 8, 9, 11, 12, 16, 22, 23, 24, 48, 128)])
        if not ok:
            res["harness_errors"].append("hook gating")
        del logits, seq, out


# ----------------------------------------------------------------------------------------------------- input vectors
def bits_to_float(i: torch.Tensor) -> torch.Tensor:
    """Float values from int32 bit patterns, with no arithmetic that could flush a denormal."""
    return i.view(torch.float32)


class Vectors:
    def __init__(self, be):
        self.be = be
        self.dev = be.dev
        self.g = be.gen

    def randn(self, *shape):
        return torch.randn(*shape, generator=self.g, device=self.dev)

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.dev)

    def randint(self, lo, hi, shape, dtype=torch.int64):
        return torch.randint(lo, hi, shape, generator=self.g, device=self.dev, dtype=dtype)

    @staticmethod
    def q18(x):
        """18 significant bits: the low 6 of the 23 mantissa bits cleared, so truncation toward zero, exact."""
        return (x.view(torch.int32) & -64).view(torch.float32)

    def place(self, x, a, b, above, eq, allowed=None):
        """Row r: its first a[r] random columns take above[r], its next b[r] take eq[r]. Everything else stays as it is."""
        rows, length = x.shape
        m = int((a + b).max())
        score = self.rand(rows, length)
        if allowed is not None:
            score = score.masked_fill(~allowed, -1.0)
        pos = score.topk(m, dim=1).indices
        rank = torch.arange(m, device=self.dev)[None, :]
        new = torch.gather(x, 1, pos)
        new = torch.where((rank >= a[:, None]) & (rank < (a + b)[:, None]), eq[:, :m], new)
        new = torch.where(rank < a[:, None], above[:, :m], new)
        x.scatter_(1, pos, new)

    def class_sizes(self, rows, length):
        """a columns above the boundary value and b on it, a <= 511 and a + b >= 513."""
        a = self.randint(0, K, (rows,))
        extra = self.randint(0, min(2048, length - K - 1) + 1, (rows,))
        b = K + 1 - a + extra
        return a, b

    def tied(self, x, allowed=None):
        rows, length = x.shape
        a, b = self.class_sizes(rows, length)
        m = int((a + b).max())
        v = torch.full((rows, 1), TIED_VALUE, device=self.dev)
        v[rows // 2 :] = self.q18(self.randn(rows - rows // 2, 1))
        step = 1.0 / 256
        x.copy_(self.q18(v - (self.randn(rows, length).abs() + step)))
        above = self.q18(v + (self.randn(rows, m).abs() + step))
        self.place(x, a, b, above, v.expand(rows, m), allowed)
        return x

    def pmzero(self, rows, length):
        step = 1.0 / 256
        x = -(self.randn(rows, length).abs() + step)
        a, b = self.class_sizes(rows, length)
        m = int((a + b).max())
        above = self.randn(rows, m).abs() + step
        signs = self.randint(0, 2, (rows, m), dtype=torch.int32) * (-2147483648)
        self.place(x, a, b, above, bits_to_float(signs))
        return x

    def denorm(self, rows, length):
        x = self.randn(rows, length)
        if length <= K:
            return x
        # All rows: a few denormals anywhere, either sign.
        d = int(self.randint(1, 65, (1,)))
        d = min(d, length)
        signs = self.randint(0, 2, (rows, d), dtype=torch.int32) * (-2147483648)
        vals = bits_to_float(self.randint(1, 1 << 23, (rows, d), dtype=torch.int32) | signs)
        self.place(x, torch.zeros(rows, dtype=torch.int64, device=self.dev), torch.full((rows,), d, device=self.dev),
                   vals, vals)
        # Half of the rows: the 512th value among a few denormals and zeros, so that their order decides the selection.
        half = rows // 2 if rows > 1 else int(self.rand(1) < 0.5)
        if half == 0:
            return x
        step = 1.0 / 256
        y = -(self.randn(half, length).abs() + step)
        a = self.randint(K - 7, K, (half,))
        b = torch.minimum(self.randint(8, 65, (half,)), length - a)
        m = int((a + b).max())
        above = self.randn(half, m).abs() + step
        small = self.rand(half, 1) < 0.5  # some rows from 3 values, so that denormals also tie among themselves
        mags = torch.where(small, self.randint(1, 4, (half, m), dtype=torch.int32), self.randint(1, 1 << 23, (half, m), dtype=torch.int32))
        z = self.rand(half, m) < 0.25
        mags = mags.masked_fill(z, 0)
        self.place(y, a, b, above, bits_to_float(mags))
        x[:half] = y
        return x

    def masked(self, rows, length):
        x = self.q18(self.randn(rows, length))
        nb = (length + 7) // 8
        col_keep = None
        if nb > 2048:
            score = self.rand(rows, nb)
            score[1::2, -1] = 2.0  # odd rows keep the row's last, possibly partial, block
            keep = torch.zeros(rows, nb, dtype=torch.bool, device=self.dev).scatter_(1, score.topk(2048, dim=1).indices, True)
            col_keep = keep.repeat_interleave(8, dim=1)[:, :length]
        if length > K:
            half = rows // 2
            if half:
                y = x[:half].clone()
                self.tied(y, None if col_keep is None else col_keep[:half])
                x[:half] = y
        if col_keep is not None:
            x = x.masked_fill(~col_keep, float("-inf"))
            fin = int(torch.isfinite(x).sum(dim=1).min())
            if fin < 16377:
                raise RuntimeError(f"masked vector: {fin} finite values")
        return x

    def draw(self, kind, rows, length):
        if kind == "cont":
            return self.randn(rows, length)
        if kind == "q18":
            return self.q18(self.randn(rows, length))
        if kind == "tie512":
            x = torch.empty(rows, length, device=self.dev)
            if length <= K:
                return self.q18(self.randn(rows, length))
            return self.tied(x)
        if kind == "pmzero":
            if length <= K:
                return self.randn(rows, length)
            return self.pmzero(rows, length)
        if kind == "denorm":
            return self.denorm(rows, length)
        if kind == "masked":
            return self.masked(rows, length)
        raise ValueError(kind)

    def coverage(self, x):
        """How many rows really hold what the kind promises: boundary ties, -0 on the boundary class, denormals."""
        rows, length = x.shape
        c = dict(tie_rows=0, pm0_rows=0, denormals=0, finite_min=int(torch.isfinite(x).sum(dim=1).min()))
        i = x.view(torch.int32)
        c["denormals"] = int((((i & 0x7F800000) == 0) & ((i & 0x7FFFFFFF) != 0)).sum())
        if length > K:
            kth = x.topk(K, dim=1).values[:, -1:]
            n_ge = (x >= kth).sum(dim=1)
            c["tie_rows"] = int((n_ge > K).sum())
            c["pm0_rows"] = int(((kth[:, 0] == 0) & (n_ge > K)).sum())
        return c


def rotate_into(logits, base, length, offset, gran):
    """logits[:, :length] = base rotated left by offset over its whole gran-sized blocks; a partial last block stays put."""
    region = (length // gran) * gran
    offset = offset % region if region else 0
    if offset == 0:
        logits[:, :length].copy_(base)
        return
    logits[:, : region - offset].copy_(base[:, offset:region])
    logits[:, region - offset : region].copy_(base[:, :offset])
    if region < length:
        logits[:, region:length].copy_(base[:, region:length])


def batch_shape(rows):
    for next_n in (6, 4, 3, 2):
        if rows % next_n == 0:
            return rows // next_n, next_n
    return rows, 1


def lens_bank(be, rows, length, kind, rng, widths_lengths):
    """Per-row lengths to cycle through: uniform, the (B, next_n) staircase of spec decode, ragged rows."""
    b, next_n = batch_shape(rows)
    bank = []

    def add(values):
        bank.append(torch.tensor(values, dtype=torch.int32, device=be.dev).reshape(b, next_n))

    add([length] * rows)
    add([length] * rows)
    add([max(length - next_n + 1 + j, 0) for _ in range(b) for j in range(next_n)])
    if kind != "masked":  # a shorter masked row could fall under 512 finite values, which is out of domain
        choices = [x for x in (0, -1) + tuple(widths_lengths) if x <= length]
        for _ in range(2):
            add([length if i % 2 == 0 else rng.choice(choices) for i in range(rows)])
    return bank


# -----------------------------------------------------------------------------------------------------------  stages
def fresh_res():
    return dict(
        exact_shapes=0, exact_replays=0, exact_diff_rows=0, oracle_bad_rows=0, smoke_diff_rows=0,
        timing={}, sweep={}, exact_complete=False, timing_complete=False, smoke_done=False, probe={}, harness_errors=[],
        mismatch_saved=0, exact_stopped_at=None,
    )


def stage_probe(be, res):
    log("== probe: occupancy, shared memory, registers (queries only)")
    for c in (16, 8, 4, 2):
        try:
            d = be.probe(c)
            res["probe"][str(c)] = dict(d)
            log(f"  C={c:<2d} cudaOccupancyMaxActiveClusters={d['max_active_clusters']:<3d} dyn_smem={d['dyn']} static_smem={d['static_smem']} "
                f"optin={d['optin']} regs={d['num_regs']}")
        except RuntimeError as e:
            res["probe"][str(c)] = dict(error=str(e))
            log(f"  C={c:<2d} probe error: {e}")
    m = be.hook()
    for rows in (2, 4, 5, 6, 8, 9, 11, 12, 16, 22, 24, 48, 128):
        c = m.cluster_for(rows)
        p = res["probe"].get(str(c), {})
        note = ""
        if c > 1 and p.get("max_active_clusters", 1 << 30) < rows:
            note = f"  NOTE: fewer active clusters ({p.get('max_active_clusters')}) than rows: clusters will queue"
        log(f"  rows {rows:<3d} -> C={c} ({c * rows} CTAs){note}")
    record("probe", probe=res["probe"], map=str(m.CLUSTER_MAP), smem_cap=m.SMEM_CAP, small=m.SMALL)


def compare(be, logits, seq, out_c, out_d, ctx, res, key, oracle=False):
    """Int-view compare. The sentinel -7 in unwritten slots counts as a difference too."""
    diff = (out_c != out_d).any(dim=1) | (out_d < -1).any(dim=1)
    n_diff = int(diff.sum())
    if n_diff:
        res[key] += n_diff
        ends = seq.reshape(-1)
        bad = diff.nonzero()[:, 0].tolist()
        log(f"  MISMATCH {ctx}: {n_diff} rows differ, first rows {bad[:4]}")
        for r in bad[:3]:
            n = int(ends[r])
            ref = f_ref_row(logits[r], n)
            c_ok = torch.equal(out_c[r].cpu(), ref)
            d_ok = torch.equal(out_d[r].cpu(), ref)
            nd = int((out_c[r] != out_d[r]).sum())
            log(f"    row {r} n={n}: {nd} slots differ; chain==reference {c_ok}, det_topk==reference {d_ok}")
            if res["mismatch_saved"] < 8:
                res["mismatch_saved"] += 1
                path = os.path.join(OUT, f"mismatch-{res['mismatch_saved']}.pt")
                torch.save(dict(ctx=ctx, row=r, n=n, x=logits[r, : max(n, 0)].cpu().clone(), chain=out_c[r].cpu().clone(),
                                det=out_d[r].cpu().clone(), reference=ref), path)
                log(f"    saved {path}")
        record("mismatch", ctx=ctx, rows=n_diff, first=bad[:8])
    if oracle:
        # The kernels against the integer reference, on two rows: guards against both arms being wrong the same way.
        ends = seq.reshape(-1)
        for r in {0, logits.shape[0] - 1}:
            ref = f_ref_row(logits[r], int(ends[r]))
            if not torch.equal(out_d[r].cpu(), ref):
                res["oracle_bad_rows"] += 1
                if res["oracle_bad_rows"] <= 8:
                    log(f"  ORACLE {ctx}: row {r} n={int(ends[r])} det_topk differs from the integer reference")
                    record("oracle", ctx=ctx, row=r, n=int(ends[r]))
    return n_diff


def make_buffers(be, rows, width):
    logits = torch.empty(rows, width, device=be.dev)
    b, next_n = batch_shape(rows)
    seq = torch.zeros((b, next_n), dtype=torch.int32, device=be.dev)
    out_c = torch.empty(rows, K, dtype=torch.int32, device=be.dev)
    out_d = torch.empty(rows, K, dtype=torch.int32, device=be.dev)
    return logits, seq, next_n, out_c, out_d, be.make_workspace()


def stage_smoke(be, vec, res):
    log("== smoke: eager and one-graph compare per cluster path")
    be.smoke_extra(res)
    width = max(GRID["widths"])
    for rows in (2, 6, 12, 24):
        logits, seq, next_n, out_c, out_d, ws = make_buffers(be, rows, width)
        logits.fill_(float("inf"))
        t0 = time.time()
        for length in [x for x in (300, 513, 1163, 12288, 12289, 65536) if x <= width]:
            x = vec.draw("tie512", rows, length)
            logits[:, :length].copy_(x)
            seq.fill_(length)
            out_c.fill_(-7)
            out_d.fill_(-7)
            be.chain(logits, seq, next_n, out_c, ws)
            be.det(logits, seq, next_n, out_d)
            be.sync()
            compare(be, logits, seq, out_c, out_d, f"smoke eager rows={rows} L={length}", res, "smoke_diff_rows", oracle=True)
            out_c.fill_(-7)
            be.chain(logits, seq, next_n, out_c, ws, max_seq_len=length)
            be.sync()
            compare(be, logits, seq, out_c, out_d, f"smoke eager true-max rows={rows} L={length}", res, "smoke_diff_rows")
        g = be.capture(lambda: (be.chain(logits, seq, next_n, out_c, ws), be.det(logits, seq, next_n, out_d)))
        for length in [x for x in (1, 512, 513, 12289) if x <= width]:
            x = vec.draw("q18", rows, length)
            logits[:, :length].copy_(x)
            seq.fill_(length)
            out_c.fill_(-7)
            out_d.fill_(-7)
            g.replay()
            be.sync()
            compare(be, logits, seq, out_c, out_d, f"smoke graph rows={rows} L={length}", res, "smoke_diff_rows", oracle=True)
        del g
        log(f"  rows {rows}: done in {time.time() - t0:.1f} s, differing rows so far {res['smoke_diff_rows']}")
        del logits, seq, out_c, out_d, ws
        be.trim()
    res["smoke_done"] = True


def build_timing_graph(be, arm, co, logits, seq, next_n, out, ws, ts, cluster=None):
    def body():
        for i in range(REPEATS):
            be.stamp(ts, 2 * i)
            if co:
                be.fork_corun(i)
            if arm == "chain":
                be.chain(logits, seq, next_n, out, ws)
            elif arm == "det":
                be.det(logits, seq, next_n, out, cluster)
            be.stamp(ts, 2 * i + 1)
            if co:
                be.join_corun()

    return be.capture(body)


def measure(be, g, ts, cal_us=0.0, seconds=0.25):
    """Median over replays of the mean arm time per replay, in us, from %globaltimer stamps inside the graph."""
    for _ in range(3):
        g.replay()
    be.sync()
    t0 = time.time()
    g.replay()
    v = ts.cpu()
    per = time.time() - t0
    n = max(20, min(200, int(seconds / max(per, 1e-6))))
    vals = []
    for _ in range(n):
        g.replay()
        v = ts.cpu().to(torch.int64)
        vals.append(float((v[1::2] - v[0::2]).to(torch.float64).mean()) / 1000.0)
    vals.sort()
    return dict(median=statistics.median(vals) - cal_us, p10=vals[len(vals) // 10] - cal_us, p90=vals[(len(vals) * 9) // 10] - cal_us, n=n)


def fill_for_timing(vec, logits, seq, rows, length):
    """Real logits are fp32 values with 18 significant bits. Past the row end: +inf."""
    logits.fill_(float("inf"))
    logits[:, :length].copy_(vec.draw("q18", rows, length))
    seq.fill_(length)


def nvsmi_clocks(be):
    if isinstance(be, MockBackend):
        return "mock"
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu", "--format=csv,noheader"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=20)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"nvidia-smi error {e!r}"


def g1_cells(widths):
    """(rows, width, length, weight). Rows: chains in the code + prose traces (q0.1, q0.2 on cf9d04b1): 2 x216, 4 x672, 6 x64.
    Lengths: about 57% of calls have <= 512 entries (5 ratio-1 and 3 ratio-2 layers per step, skeptic-gain arithmetic), split
    evenly over each side. Ratio-2 rows (W=524288) stop near 582 columns in G1, so they count at <= 512 only."""
    hi, lo = max(widths), min(widths)
    rows_w = {2: 216, 4: 672, 6: 64}
    total = sum(rows_w.values())
    cells = []
    for rows, rw in rows_w.items():
        for length, lw in ((300, 0.285), (512, 0.285), (700, 0.215), (1150, 0.215)):
            w = rw / total * lw
            if length <= 512 and hi != lo:
                cells.append((rows, hi, length, w * 5 / 8))
                cells.append((rows, lo, length, w * 3 / 8))
            else:
                cells.append((rows, hi, length, w))
    return cells


def stage_timing(be, vec, res):
    log("== timing: us per layer, median over replays, stamps inside the graph, q18 rows, past-the-end columns +inf")
    log(f"  clocks at start (sm, mem, power, temp): {nvsmi_clocks(be)}")
    m = be.hook()
    t_start = time.time()
    for width in GRID["widths"]:
        for rows in GRID["t_rows"]:
            logits, seq, next_n, out, _, ws = make_buffers(be, rows, width)
            ts = torch.zeros(2 * REPEATS, dtype=torch.int64, device=be.dev)
            fill_for_timing(vec, logits, seq, rows, min(1163, width))
            be.chain(logits, seq, next_n, out, ws)  # compiles the Triton kernels before any capture
            be.det(logits, seq, next_n, out)
            be.sync()
            graphs = {
                (arm, co): build_timing_graph(be, arm, co, logits, seq, next_n, out, ws, ts)
                for arm in ("chain", "det") for co in (False, True)
            }
            g_cal = build_timing_graph(be, "cal", False, logits, seq, next_n, out, ws, ts)
            lengths = list(GRID["t_lengths"]) + (list(GRID["g1_lengths"]) if rows in (2, 4, 6) else [])
            for length in sorted(set(x for x in lengths if x <= width)):
                fill_for_timing(vec, logits, seq, rows, length)
                cal = measure(be, g_cal, ts)["median"]
                row = {}
                for (arm, co), g in graphs.items():
                    r = measure(be, g, ts, cal)
                    row[(arm, co)] = r["median"]
                    res["timing"][f"{rows}/{width}/{length}/{arm}/{'co' if co else 'alone'}"] = r["median"]
                    record("timing", rows=rows, width=width, length=length, arm=arm, co=co, cal_us=cal, **r)
                log(f"  rows {rows:<2d} C={m.cluster_for(rows)} W={width} L={length:<7d} chain {row[('chain', False)]:8.1f} det {row[('det', False)]:8.1f} "
                    f"| beside l2pf chain {row[('chain', True)]:8.1f} det {row[('det', True)]:8.1f}")
            del graphs, g_cal, logits, seq, out, ws, ts
            be.trim()
    log(f"  clocks at end: {nvsmi_clocks(be)}; timing stage took {time.time() - t_start:.0f} s")
    res["timing_complete"] = True


def g1_weighted(res, arm, co):
    total = 0.0
    for rows, width, length, w in g1_cells(GRID["widths"]):
        total += w * res["timing"][f"{rows}/{width}/{length}/{arm}/{co}"]
    return total


def slower_shapes(res):
    out = []
    for key, det in res["timing"].items():
        rows, width, length, arm, co = key.split("/")
        if arm != "det":
            continue
        chain = res["timing"][f"{rows}/{width}/{length}/chain/{co}"]
        if det > chain * SLOW_RATIO + SLOW_ABS_US:
            out.append((f"rows={rows} W={width} L={length} {co}", round(det, 1), round(chain, 1)))
    return out


def stage_sweep(be, vec, res):
    log("== sweep (informational, no effect on the verdict): det_topk us per layer by CTAs per row, alone")
    width = max(GRID["widths"])
    for rows in (2, 4, 6):
        logits, seq, next_n, out, _, ws = make_buffers(be, rows, width)
        ts = torch.zeros(2 * REPEATS, dtype=torch.int64, device=be.dev)
        fill_for_timing(vec, logits, seq, rows, 1163)
        for c in (1, 2, 4, 8):
            be.det(logits, seq, next_n, out, c)
        be.sync()
        graphs = {c: build_timing_graph(be, "det", False, logits, seq, next_n, out, ws, ts, cluster=c) for c in (1, 2, 4, 8)}
        g_cal = build_timing_graph(be, "cal", False, logits, seq, next_n, out, ws, ts)
        for length in [x for x in GRID["sweep_lengths"] if x <= width]:
            fill_for_timing(vec, logits, seq, rows, length)
            cal = measure(be, g_cal, ts)["median"]
            row = {c: measure(be, g, ts, cal)["median"] for c, g in graphs.items()}
            res["sweep"][f"{rows}/{length}"] = row
            record("sweep", rows=rows, width=width, length=length, by_cluster=row, cal_us=cal)
            log(f"  rows {rows} L={length:<7d} " + " ".join(f"C{c}={row[c]:.1f}" for c in (1, 2, 4, 8)))
        del graphs, g_cal, logits, seq, out, ws, ts
        be.trim()


def stage_exact(be, vec, res):
    rng = random.Random(args.seed)
    widths, lengths, kinds, grows = GRID["widths"], GRID["lengths"], KINDS, GRID["rows"]
    total = sum(1 for w in widths for _ in grows for _ in kinds for x in lengths if x <= w)
    log(f"== exact: {total} shapes, {REPLAYS} replays each; int-view compare inside one captured graph per (rows, width)")
    t0 = time.time()
    done = 0
    deadline = t0 + args.max_minutes * 60
    mock_hit = dict(done=False)
    if args.mock_inject == "mismatch":
        def inject(rows, n):
            if rows == 2 and n == 513 and not mock_hit["done"]:
                mock_hit["done"] = True
                return True
            return False
        be.inject = inject
    for width in widths:
        for rows in grows:
            logits, seq, next_n, out_c, out_d, ws = make_buffers(be, rows, width)
            logits.fill_(0.0)
            seq.fill_(min(1163, width))
            logits[:, : min(1163, width)].copy_(vec.draw("q18", rows, min(1163, width)))
            be.chain(logits, seq, next_n, out_c, ws)  # compiles the Triton kernels before the capture
            be.det(logits, seq, next_n, out_d)
            be.sync()
            g = be.capture(lambda: (be.chain(logits, seq, next_n, out_c, ws), be.det(logits, seq, next_n, out_d)))
            for kind in kinds:
                logits.fill_(float("inf"))  # lengths ascend, so everything past the current length stays +inf
                for length in sorted(x for x in lengths if x <= width):
                    if time.time() > deadline:
                        res["exact_stopped_at"] = f"rows={rows} W={width} kind={kind} L={length}"
                        log(f"  --max-minutes reached at {res['exact_stopped_at']}")
                        return
                    volume = rows * length
                    distinct = max(1, min(REPLAYS, math.ceil(REPLAYS * GRID["volume_rotate"] / volume)))
                    draws = max(1, min(distinct, 3 if volume * 3 >= GRID["volume_draw"] else math.ceil(GRID["volume_draw"] / volume)))
                    rotate_every = math.ceil(REPLAYS / distinct)
                    draw_every = math.ceil(REPLAYS / draws)
                    bank = lens_bank(be, rows, length, kind, rng, lengths)
                    gran = 8 if kind == "masked" else 1
                    base = None
                    cover = None
                    d0 = res["exact_diff_rows"]
                    checks_left = 4  # per shape: the integer reference and the chain's eager paths, on the first draws
                    for i in range(REPLAYS):
                        new_draw = i % draw_every == 0
                        if new_draw:
                            base = vec.draw(kind, rows, length)
                        if new_draw or i % rotate_every == 0:
                            rotate_into(logits, base, length, 0 if new_draw else rng.randrange(0, max(length, 1)) // gran * gran, gran)
                        if new_draw and cover is None:
                            cover = vec.coverage(logits[:, :length])
                            if length > K and ((kind == "tie512" and cover["tie_rows"] != rows)
                                               or (kind == "pmzero" and cover["pm0_rows"] != rows)
                                               or (kind == "denorm" and cover["denormals"] == 0)):
                                res["harness_errors"].append(f"input kind {kind} rows={rows} L={length} lost its promise: {cover}")
                                log(f"  HARNESS: {res['harness_errors'][-1]}")
                        seq.copy_(bank[i % len(bank)])
                        out_c.fill_(-7)
                        out_d.fill_(-7)
                        g.replay()
                        check = new_draw and checks_left > 0
                        ctx = f"rows={rows} W={width} L={length} kind={kind} replay={i}"
                        compare(be, logits, seq, out_c, out_d, ctx, res, "exact_diff_rows", oracle=check)
                        if check:
                            # Eager steps see the true maximum length, so persistent_topk takes its own paths in production too.
                            checks_left -= 1
                            out_c.fill_(-7)
                            be.chain(logits, seq, next_n, out_c, ws, max_seq_len=max(1, int(seq.max())))
                            be.sync()
                            compare(be, logits, seq, out_c, out_d, ctx + " eager-true-max", res, "exact_diff_rows")
                        if args.mock_inject == "oracle" and new_draw and rows == 2 and length == 513:
                            res["oracle_bad_rows"] += 1
                    res["exact_replays"] += REPLAYS
                    res["exact_shapes"] += 1
                    done += 1
                    record("exact", rows=rows, width=width, length=length, input=kind, replays=REPLAYS, distinct_inputs=distinct, draws=draws,
                           diff_rows=res["exact_diff_rows"] - d0, coverage=cover)
                    log(f"  rows={rows:<3d} W={width} L={length:<7d} {kind:<6s} replays={REPLAYS} inputs={distinct} draws={draws} "
                        f"tie_rows={cover['tie_rows']}/{rows} pm0_rows={cover['pm0_rows']} denormals={cover['denormals']} "
                        f"finite_min={cover['finite_min']} diff_rows={res['exact_diff_rows'] - d0}")
            del g, logits, seq, out_c, out_d, ws
            be.trim()
            el = time.time() - t0
            log(f"  -- rows {rows} W={width} finished: {done}/{total} shapes, {el / 60:.1f} min, differing rows {res['exact_diff_rows']}, "
                f"integer-reference misses {res['oracle_bad_rows']}")
    res["exact_complete"] = True


# ---------------------------------------------------------------------------------------------------------- verdict
def verdict(res):
    tag = "(MOCK, not a measurement) " if args.mock else "(QUICK, can only STOP) " if args.quick else ""
    lines = []
    diff = res["exact_diff_rows"] + res["smoke_diff_rows"]
    lines.append(f"differing rows {diff} (smoke {res['smoke_diff_rows']}, exact {res['exact_diff_rows']}); "
                 f"shapes {res['exact_shapes']}, replays {res['exact_replays']}; integer-reference misses {res['oracle_bad_rows']}")
    x = None
    slower = []
    if res["timing_complete"]:
        wa, wc = g1_weighted(res, "det", "alone"), g1_weighted(res, "det", "co")
        ca, cc = g1_weighted(res, "chain", "alone"), g1_weighted(res, "chain", "co")
        x = max(wa, wc)
        slower = slower_shapes(res)
        lines.append(f"G1-mix weighted us/layer: det_topk alone {wa:.2f}, beside l2pf {wc:.2f}; chain alone {ca:.2f}, beside l2pf {cc:.2f}; "
                     f"estimated saving {8 * (ca - wa):.0f} us/step alone, {8 * (cc - wc):.0f} us/step beside l2pf")
        lines.append(f"timed shapes slower than the chain (det > chain * {SLOW_RATIO} + {SLOW_ABS_US} us): {len(slower)} {slower[:12]}")
    reasons = []
    if diff:
        v = "STOP"
        reasons.append(f"{diff} differing rows")
    elif x is not None and x >= STOP_US:
        v = "STOP"
        reasons.append(f"weighted {x:.2f} us >= {STOP_US}")
    elif res["harness_errors"]:
        v = "HOLD"
        reasons.append(f"{len(res['harness_errors'])} harness errors, first {res['harness_errors'][:3]}")
    elif args.force or args.allow_base_mismatch:
        v = "HOLD"
        reasons.append("--force or --allow-base-mismatch was used, so this run cannot say GO")
    elif args.quick:
        v = "HOLD"
        reasons.append("a --quick run never says GO")
    elif REPLAYS < 200 and not args.mock:
        v = "HOLD"
        reasons.append(f"{REPLAYS} replays per shape, the gate asks for 200")
    elif not (res["exact_complete"] and res["timing_complete"] and res["smoke_done"]):
        v = "HOLD"
        reasons.append(f"incomplete: smoke {res['smoke_done']}, timing {res['timing_complete']}, exact {res['exact_complete']}"
                       + (f" (stopped at {res['exact_stopped_at']})" if res["exact_stopped_at"] else ""))
    elif res["oracle_bad_rows"]:
        v = "HOLD"
        reasons.append(f"{res['oracle_bad_rows']} rows where the arms agree but differ from the integer reference")
    elif x > GO_US:
        v = "HOLD"
        reasons.append(f"weighted {x:.2f} us is between {GO_US} and {STOP_US}")
    elif slower:
        v = "HOLD"
        reasons.append(f"{len(slower)} timed shapes slower than the chain")
    else:
        v = "GO"
        reasons.append(f"weighted {x:.2f} us <= {GO_US}, no slower shape, zero differing rows")
    for line in lines:
        log(line)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(dict(verdict=v, reasons=reasons, lines=lines, res={k: res[k] for k in res if k != "timing"}, timing=res["timing"],
                       args=vars(args), shas=SHAS), f, indent=1, default=str)
    log(f"VERDICT: {tag}{v}: " + "; ".join(reasons) + f" [{res.get('config', '')}]")
    return {"GO": 0, "STOP": 1, "HOLD": 2}[v]


# -------------------------------------------------------------------------------------------------------- preflight
def preflight():
    site = os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
    bad = []
    for rel, shas in EXPECT_LIVE.items():
        got = file_sha(os.path.join(site, rel), 8)
        log(f"  {rel} {got}")
        if got not in shas:
            bad.append(f"{rel} {got} not in {shas}")
    log(f"  extension {file_sha(os.path.join(KIT, 'build', 'vllm_bs03_det_topk_sm121.so'), 8)}, "
        f"stamp {file_sha(os.path.join(KIT, 'build', 'libbs03_stamp.so'), 8)}, l2pf {file_sha(L2PF_LIB, 8)}")
    if bad and not args.allow_base_mismatch:
        log("REFUSED: live files differ from the expected shas: " + "; ".join(bad))
        sys.exit(3)
    if args.force:
        return
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "[v]llm serve|[V]LLM::"], stdout=subprocess.PIPE, text=True)
    if r.returncode == 0:
        log(f"REFUSED: vLLM processes are running on this node ({r.stdout.split()}). Stop the service first; this test uses the GPU.")
        sys.exit(3)
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30)
    if r.stdout.strip():
        log("REFUSED: the GPU has compute processes:\n" + r.stdout)
        sys.exit(3)


def main():
    faulthandler.dump_traceback_later(1200, repeat=True)
    SHAS.update(test=file_sha(os.path.abspath(__file__), 16), hook=file_sha(os.path.join(KIT, "site", "dsv41_bs03_det_topk.py"), 16))
    log(f"test_det_topk.py {'MOCK ' if args.mock else ''}{'QUICK ' if args.quick else ''}stages={STAGES} replays={REPLAYS} seed={args.seed} out={OUT}")
    log(f"host {os.uname().nodename}, UTC {time.strftime('%FT%TZ', time.gmtime())}, torch {torch.__version__}, test {file_sha(os.path.abspath(__file__), 8)}, "
        f"hook {file_sha(os.path.join(KIT, 'site', 'dsv41_bs03_det_topk.py'), 8)}")
    if args.mock:
        be = MockBackend()
    else:
        try:
            preflight()
            be = GpuBackend()
            p = torch.cuda.get_device_properties(0)
            log(f"device {p.name}, capability {torch.cuda.get_device_capability(0)}, SMs {p.multi_processor_count}, "
                f"memory {p.total_memory / 2**30:.1f} GiB; hook smem cap {be.m.SMEM_CAP}, small {be.m.SMALL}, map {be.m.CLUSTER_MAP}")
        except Exception as e:  # noqa: BLE001  exit 1 would read as STOP
            import traceback
            traceback.print_exc()
            log(f"HARNESS OR LAUNCH ERROR before any stage: {e!r}")
            sys.exit(3)
    vec = Vectors(be)
    res = fresh_res()
    hook = be.hook()
    res["config"] = f"map={hook.CLUSTER_MAP} smem_cap={hook.SMEM_CAP} small={hook.SMALL}"
    try:
        for stage, fn in (("probe", lambda: stage_probe(be, res)), ("smoke", lambda: stage_smoke(be, vec, res)),
                          ("timing", lambda: stage_timing(be, vec, res)), ("sweep", lambda: stage_sweep(be, vec, res)),
                          ("exact", lambda: stage_exact(be, vec, res))):
            if stage in STAGES:
                fn()
    except Exception as e:  # noqa: BLE001  a harness or launch failure is not a verdict on the kernel
        import traceback
        traceback.print_exc()
        res["harness_errors"].append(repr(e))
        log(f"HARNESS OR LAUNCH ERROR, stopping early: {e!r}")
    rc = verdict(res)
    log(f"results: {OUT}/results.jsonl, {OUT}/summary.json")
    if res["harness_errors"] and rc != 1:
        sys.exit(3)
    sys.exit(rc)


main()
