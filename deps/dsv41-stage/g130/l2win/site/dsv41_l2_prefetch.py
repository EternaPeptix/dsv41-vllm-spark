# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 r4 l2-prefetch: while an RDMA collective spins, prefetch the next weights into L2.

Gate: DSV41_L2_PREFETCH=1 (default off; rdma_all_reduce.py imports this module only then).

Why: in a TP8 decode step the target graph runs 85 RDMA all-reduces (+ the r3-dense all-gathers,
if installed). Each is an 8-16 CTA ar_kernel that polls flags for 20-90 us while the SMs and DRAM
sit idle. The kernels right after it are weight-bound: mhc_fused (reads the layer's hc fn, 24 x
20480 fp32 = 1.97 MB), then the bf16 router gate (3.93 MB) and the MXFP8 shared w1/w3 (3.0 MB)
after the post-attention all-reduce, or the replicated fused_wqa_wkv (9.5 MB) after the post-MoE
all-reduce. A side-stream kernel issues `cp.async.bulk.prefetch.L2` for the first
DSV41_L2_PREFETCH_MB (default 6) of those weights right before the collective is launched, so the
following kernels start on L2 hits. GB10 L2 = 24 MiB (nsys TARGET_INFO_GPU.l2CacheSize).
A prefetch is only a cache hint: every output is bit-identical by construction.

How the "next weights" are found (no hard-coded layer map, robust to other patches):
  * DeepseekV4Model.forward (target model only; the DSpark drafter is not bracketed) is wrapped.
    A decode-sized forward (<= DSV41_L2_PREFETCH_MAX_TOKENS tokens, not a PIECEWISE capture) that
    runs eagerly LEARNS: forward pre-hooks on every weight-holding submodule (experts, embeddings
    and lm-head excluded by size / name) and a wrapper on the mHC entry points record the order of
    weight reads, and this module's collective hook numbers the RDMA collectives.
    Window k = the first BUDGET bytes of weights read after collective k (until collective k+1).
  * vLLM captures every FULL graph right after an eager warm-up forward of the same size, so the
    capture replays exactly the learned sequence: at collective k the hook forks the side stream
    (side.wait_stream(main)), launches the prefetch there, and joins it back into the main stream
    at collective k+1 or at the end of the forward (the capture never ends with an open branch).
  * Plans are per (model, num_tokens); a size is frozen once captured. Eager (uncaptured) decode
    forwards relearn at most DSV41_L2_PREFETCH_LEARN_MAX times and never prefetch.
  * Collective count mismatch at capture (path changed between warm-up and capture) is logged and
    the remaining windows are skipped. Rank-local: no communication is added, so a rank that
    disables itself (missing .so, error) cannot desynchronise the RDMA sequence.

Env knobs:
  DSV41_L2_PREFETCH_MB            budget per window (6)
  DSV41_L2_PREFETCH_AG            1 = all-gather windows too (1)
  DSV41_L2_PREFETCH_MAX_TOKENS    largest bracketed forward (128)
  DSV41_L2_PREFETCH_MHC           1 = record the mHC fn read (the kernel right after the AR) (1)
  DSV41_L2_PREFETCH_SKIP_N        drop 2-D weights with <= N output rows from plans (0 = keep all)
  DSV41_L2_PREFETCH_MIN_PARAM_KB  ignore tensors smaller than this (32)
  DSV41_L2_PREFETCH_MAX_PARAM_MB  modules holding a tensor larger than this are not hooked (64)
  DSV41_L2_PREFETCH_LEARN_MAX     relearn cap per uncaptured size (8)
  DSV41_L2_PREFETCH_CHUNK         bytes per bulk-prefetch instruction (16384)
  DSV41_L2_PREFETCH_PRUNE_ROUTER  1 = omit recorded ffn.gate weights (0)
  DSV41_L2_PREFETCH_PRUNE_SHARED  1 = omit recorded ffn.shared_experts weights (0)
  DSV41_L2_PREFETCH_PRUNE_SHARED_POST_BUDGET 1 = remove selected shared hints without backfill (0)
  DSV41_L2_PREFETCH_PRUNE_STALE_QKV 1 = omit shard reads when r3 falls back by M (0)
  DSV41_L2_PREFETCH_PLAN_AUDIT    1 = log address-free per-window before/after plans (0)
  DSV41_L2_PREFETCH_AUDIT_TOKENS  comma-separated sizes for audit (4,6,12,24)
  DSV41_L2_PREFETCH_LIB           path of libdsv41_l2pf.so
                                  (~/dsv41-stage/r4/l2-prefetch/libdsv41_l2pf.so)

g1-stepovh WOA window (knapcio DSV41_L2_PREFETCH_WOA port, takuz0 #5; default OFF, needs the main gate):
  DSV41_L2_PREFETCH_WOA=1         after the LAST wq_b of each attention block (the indexer's wq_b in
                                  indexer layers, else the attention wq_b), a second side stream
                                  prefetches the wo_a bytes the o_proj will read (the sonar MXFP8 twin
                                  e8m0 scales + e4m3 codes when sonar takes the call, else the bf16
                                  weight) while rope / KV insert / indexer / sparse MLA run. The branch
                                  forks after the wq_b GEMM (side2.wait_stream(main)) and is joined at the
                                  end of the forward, so no main-stream kernel ever waits on it.
  DSV41_L2_PREFETCH_WOA_MB        wo_a budget per window (6; the TP8 twin is 4.3 MB)
  DSV41_L2_PREFETCH_WOB_MB        then this much of the weights read after wo_a, up to the next
                                  collective (wo_b), in the same window (0 = off; knapcio: flat)
  Collective windows are untouched by construction: wo_a is never added to them and the L / woa events
  carry no items, so with WOA on the collective plan is byte-identical to WOA off. Learned like the
  collective windows (eager warm-up before each capture), numbered per forward (wq_b post-hook
  ordinal); a count mismatch at capture skips the remaining WOA windows. Cache hints only.

g2 colltrim (b): stop the prefetch from stretching the RDMA collectives (default OFF; every arm needs the
main gate, changes cache hints only, and with all of them off the module behaves exactly as 998af3d7).
The k1 profile measured ~+10 us per post-MoE all-reduce (~1.6 MB windows), ~+18 us per post-attention
all-reduce (6 MB) and ~+8 us per qkv_a gather (6 MB of q_b + wo_b): the burst kernel saturates LPDDR5X
for the whole window while the collective's latency chain (send-slot staging, NIC DMA, sys-scope flag
polls, peer loads) queues behind it. Two levers, combinable ("G2 mode" = any of them on):
  DSV41_L2_PREFETCH_PACE_GBPS=R   issue the window bytes at R GB/s from a one-warp pacer
                                  (dsv41_l2pf_paced in DSV41_L2_PREFETCH_PACE_LIB) on a third side stream
                                  that is joined only at the end of the forward, so a pacer never blocks the
                                  main stream (0 = off: burst as before)
  DSV41_L2_PREFETCH_PACE_KINDS    fork kinds that are paced (ar,ag,post,attn); others stay burst
  DSV41_L2_PREFETCH_PACE_HEAD_KB  the first N KB of each paced fork are still issued as a burst on the
                                  usual side stream (the bytes the very next kernel reads, e.g. mhc fn) (0)
  DSV41_L2_PREFETCH_PACE_DELAY_US start the paced part this long after the fork (0)
  DSV41_L2_PREFETCH_PACE_MAX_US   a pacer stops issuing this long after its start (150; 0 = no cap)
  DSV41_L2_PREFETCH_PACE_CHUNK    bytes per paced prefetch instruction (16384)
  DSV41_L2_PREFETCH_PACE_LIB      ~/rdma_ar_dsv41/build-g2ct/libdsv41_l2pf_g2.so (missing -> burst, logged)
  g130 l2win (all default off; cache hints only):
  DSV41_L2_PREFETCH_PACE_MAX_WIN_KB  forks with more bytes than this are not paced at PACE_GBPS (0 = no limit)
  DSV41_L2_PREFETCH_PACE_BIG_GBPS    those larger forks are paced at this rate instead of staying burst (0 = burst)
  DSV41_L2_PREFETCH_BURST_DELAY_US   chain dsv41_spin(this long, 1 CTA) before the burst on the side stream (0 = off)
  DSV41_L2_PREFETCH_BURST_DELAY_KINDS fork kinds the delay applies to (ar)
  DSV41_L2_PREFETCH_REPLAN        comma list of cross-window re-plans (window bytes move, never grow):
    attn    the post-attention all-reduce window (mhc fn + router gate, read right after that AR) is
            forked at the last wq_b of the attention block instead (second side stream, like WOA), so it
            streams during rope / KV insert / indexer / sparse MLA and the AR runs with no prefetch.
            Registers the wq_b post-hooks (as WOA does); with WOA on, its wo_a bytes come first.
    qbprev  the q_b (wq_b) bytes of every qkv_a all-gather window move to the end of the previous
            collective's window (the post-MoE all-reduce), so the short gather window needs no burst
    qbpost  the same bytes are forked right AFTER the previous collective's kernel instead (a "post"
            fork: they overlap mhc / qkv_a GEMM, not a collective)
  DSV41_L2_PREFETCH_REPLAN_MB     cap on the bytes one re-plan adds to its receiving fork (16)
  Windows that lose bytes are not backfilled. Plans stay per (model, num_tokens), learned at the eager
  warm-up before each capture; rank-local, no collective is added or removed.
"""
from __future__ import annotations

import collections
import ctypes
import json
import os
import sys

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


ENABLED = _env("DSV41_L2_PREFETCH", "0").lower() in ("1", "on", "true")
BUDGET = int(float(_env("DSV41_L2_PREFETCH_MB", "6")) * (1 << 20))
USE_AG = _env("DSV41_L2_PREFETCH_AG", "1") != "0"
MAX_TOKENS = int(_env("DSV41_L2_PREFETCH_MAX_TOKENS", "128"))
MHC = _env("DSV41_L2_PREFETCH_MHC", "1") != "0"
R3_DENSE = _env("DSV41_L2_PREFETCH_R3", "1") != "0"  # r4 rebase: record r3-e4 qkv_a shard reads
SKIP_N = int(_env("DSV41_L2_PREFETCH_SKIP_N", "0"))
MIN_PARAM = int(float(_env("DSV41_L2_PREFETCH_MIN_PARAM_KB", "32")) * 1024)
MAX_PARAM = int(float(_env("DSV41_L2_PREFETCH_MAX_PARAM_MB", "64")) * (1 << 20))
LEARN_MAX = int(_env("DSV41_L2_PREFETCH_LEARN_MAX", "8"))
CHUNK = int(_env("DSV41_L2_PREFETCH_CHUNK", "16384"))
# Goal120 D01: independent, default-off cache-hint filters; never change tensor math.
PRUNE_ROUTER = _env("DSV41_L2_PREFETCH_PRUNE_ROUTER", "0") == "1"
PRUNE_SHARED = _env("DSV41_L2_PREFETCH_PRUNE_SHARED", "0") == "1"
PRUNE_SHARED_POST_BUDGET = _env("DSV41_L2_PREFETCH_PRUNE_SHARED_POST_BUDGET", "0") == "1"
PRUNE_STALE_QKV = _env("DSV41_L2_PREFETCH_PRUNE_STALE_QKV", "0") == "1"
PLAN_AUDIT = _env("DSV41_L2_PREFETCH_PLAN_AUDIT", "0") == "1"
AUDIT_TOKENS = frozenset(int(v) for v in _env(
    "DSV41_L2_PREFETCH_AUDIT_TOKENS", "4,6,12,24").split(",") if v.strip())
LIB_PATH = os.path.expanduser(
    _env("DSV41_L2_PREFETCH_LIB", "~/dsv41-stage/r4/l2-prefetch/libdsv41_l2pf.so")
)
# g1-stepovh WOA window (default off; see module doc)
WOA = _env("DSV41_L2_PREFETCH_WOA", "0").lower() in ("1", "on", "true")
WOA_BUDGET = int(float(_env("DSV41_L2_PREFETCH_WOA_MB", "6")) * (1 << 20))
WOB_BUDGET = int(float(_env("DSV41_L2_PREFETCH_WOB_MB", "0")) * (1 << 20))
# g2 colltrim (b) pacing / re-plan (default off; see module doc)
PACE_GBPS = float(_env("DSV41_L2_PREFETCH_PACE_GBPS", "0") or "0")
PACE_KINDS = frozenset(k.strip() for k in _env("DSV41_L2_PREFETCH_PACE_KINDS", "ar,ag,post,attn").split(",")
                       if k.strip())
PACE_HEAD = int(float(_env("DSV41_L2_PREFETCH_PACE_HEAD_KB", "0") or "0") * 1024) & ~15
PACE_DELAY_NS = int(float(_env("DSV41_L2_PREFETCH_PACE_DELAY_US", "0") or "0") * 1000)
PACE_MAX_NS = int(float(_env("DSV41_L2_PREFETCH_PACE_MAX_US", "150") or "0") * 1000)
PACE_CHUNK = max(16, int(_env("DSV41_L2_PREFETCH_PACE_CHUNK", "16384")) & ~15)
# g130 l2win (default off; see module doc)
PACE_MAX_WIN = int(float(_env("DSV41_L2_PREFETCH_PACE_MAX_WIN_KB", "0") or "0") * 1024)
PACE_BIG_GBPS = float(_env("DSV41_L2_PREFETCH_PACE_BIG_GBPS", "0") or "0")
BURST_DELAY_NS = int(float(_env("DSV41_L2_PREFETCH_BURST_DELAY_US", "0") or "0") * 1000)
BURST_DELAY_KINDS = frozenset(k.strip() for k in _env("DSV41_L2_PREFETCH_BURST_DELAY_KINDS", "ar").split(",")
                              if k.strip())
PACE_LIB_PATH = os.path.expanduser(
    _env("DSV41_L2_PREFETCH_PACE_LIB", "~/rdma_ar_dsv41/build-g2ct/libdsv41_l2pf_g2.so")
)
REPLAN = frozenset(k.strip() for k in _env("DSV41_L2_PREFETCH_REPLAN", "").split(",") if k.strip())
REPLAN_BUDGET = int(float(_env("DSV41_L2_PREFETCH_REPLAN_MB", "16")) * (1 << 20))
_REPLAN_KNOWN = frozenset(("attn", "qbprev", "qbpost"))
REPLAN_ATTN = "attn" in REPLAN
G2 = PACE_GBPS > 0 or bool(REPLAN) or BURST_DELAY_NS > 0
SONAR_MODULE = "vllm.models.deepseek_v4.nvidia.ops.sonar_bf16_bmm"
MODEL_MODULE = "vllm.models.deepseek_v41.nvidia.model"
_MAX_INSTALL_TRIES = 100000


class _State:
    def __init__(self) -> None:
        self.lib = None
        self.disabled = not ENABLED
        self.installed = False
        self.install_tries = 0
        self.side: dict[int, torch.cuda.Stream] = {}
        # per bracketed forward
        self.sig: tuple | None = None
        self.mode: str | None = None  # "learn" | "capture"
        self.idx = 0
        self.events: list | None = None
        self.pending: torch.cuda.Stream | None = None
        self.forks = 0
        # g1 WOA: wq_b post-hook ordinal, second side stream (joined at the end of the forward)
        self.lin = 0
        self.pending2: torch.cuda.Stream | None = None
        self.forks2 = 0
        self.side2: dict[int, torch.cuda.Stream] = {}
        # g2 colltrim: paced side stream (joined at the end of the forward) and fork counters
        self.lib_g2 = None
        self.pace_ok = False
        self.pending3: torch.cuda.Stream | None = None
        self.side3: dict[int, torch.cuda.Stream] = {}
        self.g2_forks = collections.Counter()
        # persistent
        self.plans: dict[tuple, dict] = {}
        self.learned: collections.Counter = collections.Counter()
        self.frozen: set[tuple] = set()
        self.hooked: set[int] = set()
        self.keepalive: list[torch.Tensor] = []
        self.logged: set = set()
        self.model_filter = None  # tests: callable(model) -> bool


S = _State()


# ----------------------------------------------------------------------------- native ------
def _load_lib():
    if S.lib is None:
        lib = ctypes.CDLL(LIB_PATH)
        lib.dsv41_l2pf.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p]
        lib.dsv41_l2pf.restype = ctypes.c_int
        lib.dsv41_spin.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p]
        lib.dsv41_spin.restype = ctypes.c_int
        S.lib = lib
    return S.lib


def _load_lib_g2():
    """g2 colltrim: the paced issuer (a second library; the burst path keeps using LIB_PATH)."""
    if S.lib_g2 is None:
        lib = ctypes.CDLL(PACE_LIB_PATH)
        lib.dsv41_l2pf_paced.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong,
                                         ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p]
        lib.dsv41_l2pf_paced.restype = ctypes.c_int
        S.lib_g2 = lib
    return S.lib_g2


def init() -> bool:
    """Called once by rdma_all_reduce.py at import when DSV41_L2_PREFETCH=1."""
    if not ENABLED:
        return False
    try:
        _load_lib()
    except OSError as e:
        S.disabled = True
        logger.warning("DSV41_L2_PREFETCH: cannot load %s (%s); prefetch disabled on this rank", LIB_PATH, e)
        return False
    if G2:
        bad = sorted(REPLAN - _REPLAN_KNOWN)
        if bad:
            raise ValueError(f"DSV41_L2_PREFETCH_REPLAN: unknown re-plan(s) {bad}; known {sorted(_REPLAN_KNOWN)}")
        if "qbprev" in REPLAN and "qbpost" in REPLAN:
            raise ValueError("DSV41_L2_PREFETCH_REPLAN: qbprev and qbpost are alternatives; pick one")
        if PACE_GBPS > 0:
            try:
                _load_lib_g2()
                S.pace_ok = True
            except (OSError, AttributeError) as e:
                logger.warning("DSV41_L2_PREFETCH_PACE: cannot load %s (%s); paced parts fall back to burst",
                               PACE_LIB_PATH, e)
        logger.info(
            "DSV41_L2_PREFETCH g2: pace=%s GB/s (kinds=%s head=%d KB delay=%.1f us max=%.0f us chunk=%d lib=%s) "
            "replan=%s (cap %.1f MB)",
            PACE_GBPS if S.pace_ok else 0, ",".join(sorted(PACE_KINDS)), PACE_HEAD // 1024,
            PACE_DELAY_NS / 1e3, PACE_MAX_NS / 1e3, PACE_CHUNK, PACE_LIB_PATH if S.pace_ok else "-",
            ",".join(sorted(REPLAN)) or "-", REPLAN_BUDGET / 2**20,
        )
        if PACE_MAX_WIN > 0 or BURST_DELAY_NS > 0:
            logger.info("DSV41_L2_PREFETCH l2win: pace_max_win=%d KB pace_big=%g GB/s burst_delay=%.1f us kinds=%s",
                        PACE_MAX_WIN // 1024, PACE_BIG_GBPS, BURST_DELAY_NS / 1e3, ",".join(sorted(BURST_DELAY_KINDS)))
    logger.info(
        "DSV41_L2_PREFETCH armed: budget %.1f MB/window ag=%s max_tokens=%d mhc=%s skip_n=%d lib=%s",
        BUDGET / 2**20, USE_AG, MAX_TOKENS, MHC, SKIP_N, LIB_PATH,
    )
    if WOA:
        logger.info("DSV41_L2_PREFETCH_WOA armed: wo_a %.1f MB + wo_b %.1f MB per attention window "
                    "(second side stream, joined at the end of the forward)",
                    WOA_BUDGET / 2**20, WOB_BUDGET / 2**20)
    return True


def _side_stream() -> torch.cuda.Stream:
    dev = torch.cuda.current_device()
    s = S.side.get(dev)
    if s is None:
        s = S.side[dev] = torch.cuda.Stream(device=dev)
    return s


def _side_stream2() -> torch.cuda.Stream:
    """g1 WOA: its own side stream, so a collective's join never waits on an attention window."""
    dev = torch.cuda.current_device()
    s = S.side2.get(dev)
    if s is None:
        s = S.side2[dev] = torch.cuda.Stream(device=dev)
    return s


def _side_stream3() -> torch.cuda.Stream:
    """g2 colltrim: the paced issuer's stream, joined only at the end of the forward."""
    dev = torch.cuda.current_device()
    s = S.side3.get(dev)
    if s is None:
        s = S.side3[dev] = torch.cuda.Stream(device=dev)
    return s


# ----------------------------------------------------------------------------- install -----
def _try_install() -> None:
    S.install_tries += 1
    mod = sys.modules.get(MODEL_MODULE)
    if mod is None:
        return
    install_model_module(mod)


def install_model_module(mod) -> None:
    """Wrap DeepseekV4Model.forward and the mHC entry points of the nvidia DSv4.1 model module."""
    cls = getattr(mod, "DeepseekV4Model", None)
    if cls is None:
        S.installed = True
        logger.warning("DSV41_L2_PREFETCH: %s has no DeepseekV4Model; nothing bracketed", MODEL_MODULE)
        return
    if not getattr(cls, "_dsv41_l2pf", False):
        bracket_class(cls)
    if MHC:
        _wrap_mhc(mod, "mhc_shifted_post_pre", 4)
        _wrap_mhc(mod, "mhc_pre_delayed_tilelang", 1)
    if R3_DENSE:
        _wrap_r3_dense()
    _side_stream()  # created outside any capture
    if WOA:
        _wrap_sonar()
        _side_stream2()
    if G2:
        _side_stream2()  # attn forks
        _side_stream3()  # paced parts
    S.installed = True
    logger.info("DSV41_L2_PREFETCH: DeepseekV4Model.forward bracketed (decode <= %d tokens)", MAX_TOKENS)


def bracket_class(cls) -> None:
    """Wrap cls.forward(self, input_ids, positions, ...) (public for the GPU unit test)."""
    orig = cls.forward

    def forward(self, *args, **kwargs):
        return _bracketed(orig, self, args, kwargs)

    forward.__wrapped__ = orig  # type: ignore[attr-defined]
    cls.forward = forward
    cls._dsv41_l2pf = True


def _wrap_mhc(mod, name: str, fn_pos: int) -> None:
    fn = getattr(mod, name, None)
    if fn is None or getattr(fn, "_dsv41_l2pf", False):
        return

    def wrapper(*args, **kwargs):
        if S.mode == "learn":
            t = args[fn_pos] if len(args) > fn_pos else kwargs.get("fn")
            if isinstance(t, torch.Tensor):
                _record(("mhc_fn", (t,)))
        return fn(*args, **kwargs)

    wrapper._dsv41_l2pf = True  # type: ignore[attr-defined]
    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    setattr(mod, name, wrapper)


def _wrap_r3_dense() -> None:
    """r4 rebase onto r3-e4: DSV41_QKV_A_SHARD reads a private [224, K] MXFP8 shard
    (attn._dsv41_qkv_shard) instead of calling the fused_wqa_wkv module, so no module
    pre-hook fires for it. Record the shard (scales first, then weight) right before the
    shard GEMM; its all-gather is the next collective, so it lands in the correct window.
    The first call per layer builds the shard and is not recorded (plan is learned again
    at every capture warm-up). indexer weights_proj (bmm32 path) is recorded the same way."""
    try:
        import importlib

        r3 = importlib.import_module("vllm.models.deepseek_v41.nvidia.dsv41_r3_dense")
    except Exception as e:  # noqa: BLE001
        logger.info("DSV41_L2_PREFETCH: r3 dense module not importable (%r); not wrapped", e)
        return
    fn = getattr(r3, "qkv_a_sharded", None)
    if fn is not None and not getattr(fn, "_dsv41_l2pf", False):

        def qkv_wrapper(attn, x, *a, **k):
            if S.mode == "learn":
                st = getattr(attn, "_dsv41_qkv_shard", None)
                if st is not None and hasattr(st, "w"):
                    name = str(getattr(attn, "prefix", "attn")) + ".qkv_a_shard"
                    # Match the live 993023f2 helper's M guard before its internal
                    # all-gather. Recording after fn() would move the event into
                    # the wrong window. Default-off preserves the original recorder.
                    stale = (PRUNE_STALE_QKV and
                             not 0 < r3._num_tokens(x) <= r3.QKV_MAX_M)
                    _record((name, (getattr(st, "s_rm", None), st.w)),
                            kind="prune_stale_qkv" if stale else "w")
            return fn(attn, x, *a, **k)

        qkv_wrapper._dsv41_l2pf = True  # type: ignore[attr-defined]
        qkv_wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        r3.qkv_a_sharded = qkv_wrapper
    fw = getattr(r3, "indexer_weights_proj", None)
    if fw is not None and not getattr(fw, "_dsv41_l2pf", False):

        def wproj_wrapper(linear, x, *a, **k):
            if S.mode == "learn":
                w = getattr(linear, "weight", None)
                if isinstance(w, torch.Tensor):
                    _record(("idx.weights_proj", (w,)))
            return fw(linear, x, *a, **k)

        wproj_wrapper._dsv41_l2pf = True  # type: ignore[attr-defined]
        wproj_wrapper.__wrapped__ = fw  # type: ignore[attr-defined]
        r3.indexer_weights_proj = wproj_wrapper
    logger.info("DSV41_L2_PREFETCH: r3 dense qkv_a shard / weights_proj reads recorded")


def woa_tensors(took: bool, wo_a, grouped_weight) -> tuple:
    """The wo_a bytes deep_gemm_fp8_o_proj's grouped projection reads, in read-priority order.
    took = sonar try_project's return value. Sonar fp8 mode streams the exact MXFP8 twin
    (wo_a._dsv41_sonar_fp8 = (e4m3 codes [G, N, K], e8m0 [G, N, K/32])): the scales first (every
    K tile needs them), then the codes. Sonar bf16 mode and the torch.bmm fallback read the bf16
    grouped weight (a view of wo_a.weight)."""
    tw = getattr(wo_a, "_dsv41_sonar_fp8", None) if took else None
    if isinstance(tw, tuple) and len(tw) == 2:
        return (tw[1], tw[0])
    return (grouped_weight,)


def _wrap_sonar() -> None:
    """g1 WOA: record the wo_a read. deep_gemm_fp8_o_proj looks `_sonar.try_project` up at call time
    (function-local import), so wrapping the module attribute catches every target o_proj call. The
    record is appended after the call returns: no collective and no wq_b ends inside it, so its
    position in the event order is the same as before the call."""
    try:
        import importlib

        sonar = importlib.import_module(SONAR_MODULE)
    except Exception as e:  # noqa: BLE001
        logger.warning("DSV41_L2_PREFETCH_WOA: %s not importable (%r); no wo_a windows", SONAR_MODULE, e)
        return
    fn = getattr(sonar, "try_project", None)
    if fn is None or getattr(fn, "_dsv41_l2pf", False):
        return

    def try_project(o_proj_input, wo_a, grouped_weight, z, *a, **k):
        took = fn(o_proj_input, wo_a, grouped_weight, z, *a, **k)
        if S.mode == "learn" and S.events is not None:
            name = str(getattr(wo_a, "prefix", "wo_a"))
            _record((name, woa_tensors(bool(took), wo_a, grouped_weight)), kind="woa")
        return took

    try_project._dsv41_l2pf = True  # type: ignore[attr-defined]
    try_project.__wrapped__ = fn  # type: ignore[attr-defined]
    sonar.try_project = try_project
    logger.info("DSV41_L2_PREFETCH_WOA: sonar o_proj wo_a reads recorded")


# ----------------------------------------------------------------------------- recording ---
def _tensor_items(tensors) -> list[tuple[int, int, int | None]]:
    out = []
    for t in tensors:
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            continue
        nbytes = t.numel() * t.element_size()
        ptr = t.data_ptr()
        if nbytes < MIN_PARAM or ptr % 16 or not t.is_contiguous():
            continue
        rows = int(t.shape[0]) if t.dim() == 2 else None
        out.append((ptr, nbytes, rows))
    return out


def _record(ev, kind="w") -> None:
    name, tensors = ev
    items = _tensor_items(tensors)
    if items and S.events is not None:
        S.events.append((kind, name, items))


def _module_tensors(m: torch.nn.Module) -> list[torch.Tensor] | None:
    """Direct (non-recursive) params + buffers in read order: small (scales) first, then weights.
    None = do not hook (no tensor worth prefetching, or holds a huge tensor: experts, vocab)."""
    ts = []
    for pname, p in list(m.named_parameters(recurse=False)) + list(m.named_buffers(recurse=False)):
        if p is None or not p.is_cuda:
            continue
        if pname.startswith("hc_"):  # mHC tensors: recorded at their real use by _wrap_mhc
            continue
        nbytes = p.numel() * p.element_size()
        if nbytes > MAX_PARAM:
            return None
        if nbytes >= MIN_PARAM:
            ts.append(p)
    if not ts:
        return None
    return sorted(ts, key=lambda t: t.numel() * t.element_size())


def _hook_modules(model: torch.nn.Module) -> None:
    if id(model) in S.hooked:
        return
    S.hooked.add(id(model))
    n_hooked = 0
    n_lin = 0
    for name, m in model.named_modules():
        if (WOA or REPLAN_ATTN) and (name == "wq_b" or name.endswith(".wq_b")):
            # g1 WOA: the wq_b GEMMs (attention and indexer) end the window that precedes wo_a
            m.register_forward_hook(_wq_b_post_hook)
            n_lin += 1
        if ".experts" in name and "shared_experts" not in name:
            continue
        ts = _module_tensors(m)
        if ts is None:
            continue
        def pre_hook(_mod, _args, _ts=ts, _name=name):
            if S.mode == "learn":
                _record((_name, _ts))

        m.register_forward_pre_hook(pre_hook)
        n_hooked += 1
    logger.info("DSV41_L2_PREFETCH: %d weight modules hooked in %s%s", n_hooked, type(model).__name__,
                f"; WOA/attn: {n_lin} wq_b post-hooks" if (WOA or REPLAN_ATTN) else "")


def _wq_b_post_hook(_mod, _args, _out) -> None:
    """g1 WOA: numbered linear boundary. Learn: record it. Capture: fork the planned wo_a window."""
    if S.sig is None or S.mode is None:
        return
    S.lin += 1
    if S.mode == "learn":
        if S.events is not None:
            S.events.append(("L", S.lin))
        return
    plan = S.plans.get(S.sig)
    if plan is None or S.lin > plan.get("nl", 0) or S.idx > plan["n"]:
        return
    g2 = plan.get("g2")
    if g2 is not None:
        for part in g2["attn"].get(S.lin, ()):
            _fork_part(part, "attn")
        return
    item = plan.get("lwin", {}).get(S.lin)
    if item is not None:
        _fork2(item)


# ----------------------------------------------------------------------------- bracket -----
def _in_piecewise() -> bool:
    try:
        from vllm.forward_context import in_piecewise_cudagraph

        return bool(in_piecewise_cudagraph())
    except Exception:  # noqa: BLE001
        return False


def _num_tokens(args, kwargs) -> int:
    pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
    if isinstance(pos, torch.Tensor):
        return int(pos.shape[-1]) if pos.dim() == 2 else int(pos.shape[0])  # mrope: [3, n]
    ids = kwargs.get("input_ids", args[0] if args else None)
    return int(ids.shape[0]) if isinstance(ids, torch.Tensor) else 0


def _bracketed(orig, model, args, kwargs):
    if S.disabled or S.sig is not None or (S.model_filter is not None and not S.model_filter(model)):
        return orig(model, *args, **kwargs)
    n = _num_tokens(args, kwargs)
    if n <= 0 or n > MAX_TOKENS or _in_piecewise():
        return orig(model, *args, **kwargs)
    sig = (id(model), n)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        mode = "capture" if sig in S.plans else None
    else:
        mode = "learn" if (sig not in S.frozen and S.learned[sig] < LEARN_MAX) else None
    if mode is None:
        return orig(model, *args, **kwargs)
    _hook_modules(model)
    S.sig, S.mode, S.idx, S.forks = sig, mode, 0, 0
    S.lin, S.forks2 = 0, 0
    if G2:
        S.g2_forks = collections.Counter()
    S.events = [] if mode == "learn" else None
    try:
        return orig(model, *args, **kwargs)
    finally:
        _join()
        _join2()
        _join3()
        try:
            _finish(sig, mode)
        finally:
            S.sig = S.mode = S.events = None


def _finish(sig, mode) -> None:
    if mode == "learn":
        S.learned[sig] += 1
        if S.idx == 0:
            return  # no RDMA collective in this forward (NCCL path): learn nothing
        plan = make_plan(S.events or [], S.idx)
        old = S.plans.get(sig)
        S.plans[sig] = plan
        if old is None or old["key"] != plan["key"]:
            first = sig[1] not in {k[1] for k in S.logged if k[0] == "plan"}
            S.logged.add(("plan", sig[1]))
            logger.info(
                "DSV41_L2_PREFETCH plan tokens=%d collectives=%d windows=%d prefetch=%.1f MB/forward%s%s",
                sig[1], plan["n"], len(plan["win"]), plan["bytes"] / 2**20,
                "" if old is None else " (relearned)",
                ("; first windows: " + plan["desc"]) if first else "",
            )
            if WOA:
                logger.info(
                    "DSV41_L2_PREFETCH_WOA plan tokens=%d wq_b=%d woa_windows=%d prefetch=%.1f MB/forward%s",
                    sig[1], plan.get("nl", 0), len(plan.get("lwin", {})), plan.get("lbytes", 0) / 2**20,
                    ("; first: " + plan.get("ldesc", "")) if first else "",
                )
            if "g2" in plan:
                logger.info("DSV41_L2_PREFETCH g2 plan tokens=%d %s%s", sig[1], plan["g2"]["summary"],
                            ("; first: " + plan["g2"]["desc"]) if first else "")
            if "pruning" in plan:
                logger.info(
                    "DSV41_L2_PREFETCH pruning tokens=%d %s",
                    sig[1], json.dumps(plan["pruning"]["totals"], sort_keys=True),
                )
                if PLAN_AUDIT and sig[1] in AUDIT_TOKENS:
                    logger.info(
                        "DSV41_L2_PREFETCH plan_audit tokens=%d %s",
                        sig[1], json.dumps(plan["pruning"]["windows"], sort_keys=True),
                    )
    else:
        S.frozen.add(sig)
        plan = S.plans.get(sig)
        ok = plan is not None and plan["n"] == S.idx
        if ("cap", sig[1]) not in S.logged or not ok:
            S.logged.add(("cap", sig[1]))
            logger.info(
                "DSV41_L2_PREFETCH captured tokens=%d forks=%d collectives=%d (learned %d)%s",
                sig[1], S.forks, S.idx, plan["n"] if plan else -1,
                "" if ok else " MISMATCH: windows past the mismatch were skipped",
            )
        if G2 and plan is not None and "g2" in plan and ("cap3", sig[1]) not in S.logged:
            S.logged.add(("cap3", sig[1]))
            logger.info("DSV41_L2_PREFETCH g2 captured tokens=%d forks %s (planned %s)", sig[1],
                        json.dumps(dict(sorted(S.g2_forks.items()))), plan["g2"]["nforks"])
        if WOA:
            ok2 = plan is not None and plan.get("nl", 0) == S.lin
            if ("cap2", sig[1]) not in S.logged or not ok2:
                S.logged.add(("cap2", sig[1]))
                logger.info(
                    "DSV41_L2_PREFETCH_WOA captured tokens=%d forks=%d wq_b=%d (learned %d)%s",
                    sig[1], S.forks2, S.lin, plan.get("nl", 0) if plan else -1,
                    "" if ok2 else " MISMATCH: WOA windows past the mismatch were skipped",
                )


def _weight_category(name: str) -> str:
    """Classify names actually emitted by named_modules / r3 / mHC recorders.

    DeepseekV4MoE assigns gate and shared_experts before its experts runner,
    so named_modules dedup uses layers.N.ffn.gate and
    layers.N.ffn.shared_experts.{gate_up_proj,down_proj}. Match components,
    never generic 'gate' substrings (Engram and attention gates must survive).
    """
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "ffn":
            if parts[i + 1:] == ["gate"]:
                return "router"
            if parts[i + 1] == "shared_experts":
                return "shared"
    if name == "mhc_fn":
        return "mhc"
    if parts[-1] == "qkv_a_shard":
        return "qkv_shard"
    if parts[-1] in ("wq_b", "q_b"):
        return "q_b"
    return "other"


def _drop_weight(name: str) -> bool:
    group = _weight_category(name)
    return ((PRUNE_ROUTER and group == "router") or
            (PRUNE_SHARED and group == "shared"))


def _selected_summary(items, budget):
    """Mirror budget allocation for diagnostics only; never expose device addresses."""
    left, groups, names = budget, collections.Counter(), collections.Counter()
    for _, nbytes, name in items:
        if left < 16:
            break
        take = min(nbytes, left) & ~15
        if take <= 0:
            continue
        groups[_weight_category(name)] += take
        names[name] += take
        left -= take
    return {"bytes": sum(groups.values()), "by_category": dict(groups),
            "by_name": dict(names)}


def _shared_post_budget_items(items, budget):
    """Keep only nonshared slices of the original selected window; never refill holes."""
    selected, left = [], budget
    for ptr, nbytes, name in items:
        if left < 16:
            break
        take = min(nbytes, left) & ~15
        if take <= 0:
            continue
        left -= take  # shared slices still consume their original budget positions
        if _weight_category(name) != "shared":
            selected.append((ptr, take, name))
    return selected


def make_plan(events, n_coll: int, budget: int = BUDGET, skip_n: int = SKIP_N, device="cuda") -> dict:
    """events: [("c", idx, kind) | ("w", name, [(ptr, nbytes, rows), ...])] in forward order.
    Window idx = the first `budget` bytes of the weights read after collective idx and before the
    next one. Returns {"n", "win": {idx: (segtable int64 [2*nseg] on device, nseg, bytes)}, ...}."""
    if PRUNE_SHARED_POST_BUDGET and (PRUNE_ROUTER or PRUNE_SHARED or PRUNE_STALE_QKV):
        raise ValueError("post-budget shared pruning is an isolated arm; disable all pre-budget pruning flags")
    windows: list[list] = []
    stale = set()
    cur = None
    for ev in events:
        if ev[0] == "c":
            cur = [ev[1], ev[2], []]
            windows.append(cur)
        elif ev[0] in ("L", "woa"):
            continue  # g1 WOA events add nothing to collective windows (make_woa_plan reads them)
        elif cur is not None:
            for it in ev[2]:
                if skip_n and it[2] is not None and it[2] <= skip_n:
                    continue
                cur[2].append((it[0], it[1], ev[1]))
                if ev[0] == "prune_stale_qkv":
                    stale.add((len(windows) - 1, len(cur[2]) - 1))
    win, total, desc = {}, 0, []
    sel = []  # g2 colltrim: every window's selected (ptr, take, name) in read order, by position
    audit = []
    diagnostic = PLAN_AUDIT or PRUNE_ROUTER or PRUNE_SHARED or PRUNE_STALE_QKV or PRUNE_SHARED_POST_BUDGET
    for window_idx, (idx, kind, items) in enumerate(windows):
        before = items
        # Retain event-specific stale markers, so a valid read of the same
        # pointer elsewhere in the forward cannot be accidentally suppressed.
        if PRUNE_ROUTER or PRUNE_SHARED or PRUNE_STALE_QKV:
            items = [it for item_idx, it in enumerate(items) if not _drop_weight(it[2])
                     and not (PRUNE_STALE_QKV and (window_idx, item_idx) in stale)]
        if PRUNE_SHARED_POST_BUDGET:
            items = _shared_post_budget_items(items, budget)
        if diagnostic:
            audit.append({"index": idx, "kind": kind,
                          "before": _selected_summary(before, budget),
                          "after": _selected_summary(items, budget)})
        segs, left, names = [], budget, []
        g2segs = []
        for ptr, nbytes, name in items:
            if left < 16:
                break
            take = min(nbytes, left) & ~15
            if take <= 0:
                continue
            segs.append((ptr, take))
            names.append(f"{name}:{take / 2**20:.2f}")
            if G2:
                g2segs.append((ptr, take, name))
            left -= take
        if G2:
            sel.append((idx, kind, g2segs))
        if not segs:
            continue
        flat = [v for s in segs for v in s]
        t = torch.tensor(flat, dtype=torch.int64, device=device)
        S.keepalive.append(t)  # a captured graph keeps reading this table
        nb = sum(s[1] for s in segs)
        win[idx] = (t, len(segs), nb)
        total += nb
        if len(desc) < 4:
            desc.append(f"[{kind}{idx} " + " ".join(names) + "]")
    key = (n_coll, tuple((i, w[1], w[2]) for i, w in sorted(win.items())),
           tuple(tuple(s[:2] for s in w[2]) for w in windows))
    result = {"n": n_coll, "win": win, "bytes": total, "desc": " ".join(desc), "key": key}
    if WOA:
        woa = make_woa_plan(events, device=device)
        result.update(woa)
        result["key"] = key + (woa["lkey"],)
    if G2:
        g2 = make_g2_plan(events, sel, device=device)
        result["g2"] = g2
        result["key"] = result["key"] + (g2["gkey"],)
        result.setdefault("nl", g2["nl"])  # wq_b ordinal bound for attn forks when WOA is off
    if diagnostic:
        before_groups, after_groups = collections.Counter(), collections.Counter()
        for window in audit:
            before_groups.update(window["before"]["by_category"])
            after_groups.update(window["after"]["by_category"])
        result["pruning"] = {"windows": audit, "totals": {
            "router": PRUNE_ROUTER, "shared": PRUNE_SHARED,
            "stale_qkv": PRUNE_STALE_QKV,
            "before_bytes": sum(before_groups.values()),
            "after_bytes": sum(after_groups.values()),
            "before_by_category": dict(before_groups),
            "after_by_category": dict(after_groups),
        }}
        if PRUNE_SHARED_POST_BUDGET:
            result["pruning"]["totals"]["shared_post_budget"] = True
    return result


def _take_items(items, budget: int) -> list[tuple[int, int]]:
    """First `budget` bytes of [(ptr, nbytes, name)] in order, every piece rounded down to 16 B."""
    segs, left = [], budget
    for ptr, nbytes, _name in items:
        if left < 16:
            break
        take = min(nbytes, left) & ~15
        if take <= 0:
            continue
        segs.append((ptr, take))
        left -= take
    return segs


def make_woa_plan(events, woa_budget: int | None = None, wob_budget: int | None = None,
                  device="cuda") -> dict:
    """g1 WOA: events -> {"lwin": {wq_b ordinal: (segtable, nseg, bytes)}, "nl", "lbytes", "ldesc", "lkey"}.
    A window opens at the LAST wq_b (L event) before a wo_a read (woa event) within one collective
    window, and takes the first woa_budget bytes of the wo_a items, then (wob_budget > 0) the first
    wob_budget bytes of the weights read after wo_a until the next collective or wq_b. A wo_a read with
    no wq_b since the last collective has no fork point and gets no window."""
    woa_budget = WOA_BUDGET if woa_budget is None else woa_budget
    wob_budget = WOB_BUDGET if wob_budget is None else wob_budget
    nl, last_l, cur, groups = 0, None, None, []
    for ev in events:
        k = ev[0]
        if k == "c":
            last_l, cur = None, None
        elif k == "L":
            nl = max(nl, ev[1])
            last_l, cur = ev[1], None
        elif k == "woa":
            if last_l is None:
                continue
            if cur is None or cur[0] != last_l:
                cur = [last_l, [], []]
                groups.append(cur)
            cur[1].extend((it[0], it[1], ev[1]) for it in ev[2])
        elif cur is not None:
            cur[2].extend((it[0], it[1], ev[1]) for it in ev[2])
    lwin, total, desc = {}, 0, []
    for lin, woa_items, after in groups:
        segs = _take_items(woa_items, woa_budget)
        if wob_budget > 0:
            segs += _take_items(after, wob_budget)
        if not segs:
            continue
        t = torch.tensor([v for s in segs for v in s], dtype=torch.int64, device=device)
        S.keepalive.append(t)  # a captured graph keeps reading this table
        nb = sum(s[1] for s in segs)
        lwin[lin] = (t, len(segs), nb)
        total += nb
        if len(desc) < 2:
            desc.append(f"[L{lin} {woa_items[0][2] if woa_items else '?'}:{nb / 2**20:.2f}]")
    lkey = (nl, tuple((i, w[1], w[2]) for i, w in sorted(lwin.items())),
            tuple(tuple(it[:2] for it in g[1]) for g in groups))
    return {"lwin": lwin, "nl": nl, "lbytes": total, "ldesc": " ".join(desc), "lkey": lkey}


# ----------------------------------------------------------------------------- g2 colltrim --
def _woa_segments(events, woa_budget: int | None = None, wob_budget: int | None = None) -> dict:
    """make_woa_plan's grouping and selection, returning {wq_b ordinal: [(ptr, take, name)]} (no tensors)."""
    woa_budget = WOA_BUDGET if woa_budget is None else woa_budget
    wob_budget = WOB_BUDGET if wob_budget is None else wob_budget
    last_l, cur, groups = None, None, []
    for ev in events:
        k = ev[0]
        if k == "c":
            last_l, cur = None, None
        elif k == "L":
            last_l, cur = ev[1], None
        elif k == "woa":
            if last_l is None:
                continue
            if cur is None or cur[0] != last_l:
                cur = [last_l, [], []]
                groups.append(cur)
            cur[1].extend((it[0], it[1], ev[1]) for it in ev[2])
        elif cur is not None:
            cur[2].extend((it[0], it[1], ev[1]) for it in ev[2])
    out = {}
    for lin, woa_items, after in groups:
        segs = _take_named(woa_items, woa_budget)
        if wob_budget > 0:
            segs += _take_named(after, wob_budget)
        if segs:
            out[lin] = segs
    return out


def _take_named(items, budget: int) -> list[tuple[int, int, str]]:
    """_take_items keeping names: first `budget` bytes of [(ptr, nbytes, name)], pieces rounded down to 16 B."""
    segs, left = [], budget
    for ptr, nbytes, name in items:
        if left < 16:
            break
        take = min(nbytes, left) & ~15
        if take <= 0:
            continue
        segs.append((ptr, take, name))
        left -= take
    return segs


def _cap_named(segs, cap: int) -> tuple[list, list]:
    """Split [(ptr, take, name)] into (first `cap` bytes, the rest), pieces rounded to 16 B."""
    head, tail, left = [], [], cap
    for ptr, take, name in segs:
        if left >= 16 and take > 0:
            t = min(take, left) & ~15
            if t > 0:
                head.append((ptr, t, name))
                left -= t
                if take - t >= 16:
                    tail.append((ptr + t, (take - t) & ~15, name))
                continue
        tail.append((ptr, take, name))
    return head, tail


def g2_fork_lists(events, sel, replan=None, replan_budget=None, woa=None) -> dict:
    """Pure-Python g2 re-plan (CPU-testable). sel = [(idx, kind, [(ptr, take, name)])] per collective window in
    order (make_plan's selection after pruning). Returns {"pre": {idx: segs}, "post": {idx: segs},
    "attn": {lin: segs}, "kind": {idx: kind}, "moved": {...bytes by re-plan...}}; segs in issue order."""
    replan = REPLAN if replan is None else replan
    replan_budget = REPLAN_BUDGET if replan_budget is None else replan_budget
    woa = WOA if woa is None else woa
    pre = {idx: list(segs) for idx, _kind, segs in sel}
    kinds = {idx: kind for idx, kind, _segs in sel}
    order = [idx for idx, _kind, _segs in sel]
    post: dict[int, list] = {}
    attn: dict[int, list] = {}
    moved = collections.Counter()
    # last wq_b ordinal inside each collective window (events between collective idx and the next one)
    last_l: dict[int, int] = {}
    cur, nl = None, 0
    for ev in events:
        if ev[0] == "c":
            cur = ev[1]
        elif ev[0] == "L":
            nl = max(nl, ev[1])
            if cur is not None:
                last_l[cur] = ev[1]
    if woa or "attn" in replan:
        attn = {lin: list(segs) for lin, segs in (_woa_segments(events).items() if woa else ())}
    if "attn" in replan:
        # the window after a collective window that holds a wq_b boundary is the post-attention AR window:
        # fork it at that (last) wq_b instead, after any WOA bytes
        for pos in range(1, len(order)):
            prev, idx = order[pos - 1], order[pos]
            lin = last_l.get(prev)
            if lin is None or kinds[idx] != "ar" or not pre.get(idx):
                continue
            take, keep = _cap_named(pre[idx], replan_budget)
            if not take:
                continue
            attn.setdefault(lin, []).extend(take)
            pre[idx] = keep
            moved["attn"] += sum(t for _p, t, _n in take)
    for mode in ("qbprev", "qbpost"):
        if mode not in replan:
            continue
        for pos in range(1, len(order)):
            prev, idx = order[pos - 1], order[pos]
            if kinds[idx] != "ag":
                continue
            qb = [sg for sg in pre.get(idx, ()) if _weight_category(sg[2]) == "q_b"]
            if not qb:
                continue
            take, rest = _cap_named(qb, replan_budget)
            if not take:
                continue
            taken = collections.Counter()
            for ptr, t, _n in take:
                taken[ptr] += t
            keep = []
            for ptr, t, nm in pre[idx]:  # drop the moved q_b bytes (the cap keeps any remainder in place)
                if _weight_category(nm) == "q_b" and taken[ptr] > 0:
                    used = min(t, taken[ptr])
                    taken[ptr] -= used
                    if t - used >= 16:
                        keep.append((ptr + used, (t - used) & ~15, nm))
                    continue
                keep.append((ptr, t, nm))
            pre[idx] = keep
            if mode == "qbprev":
                pre.setdefault(prev, []).extend(take)
            else:
                post.setdefault(prev, []).extend(take)
            moved[mode] += sum(t for _p, t, _n in take)
    return {"pre": {k: v for k, v in pre.items() if v}, "post": post, "attn": {k: v for k, v in attn.items() if v},
            "kind": kinds, "moved": dict(moved), "nl": nl}


def _parts(segs, fork_kind: str, device) -> list:
    """[(ptr, take, name)] -> [("b"|"p", table, nseg, bytes)]: burst head + paced tail when pacing applies."""
    pace = PACE_GBPS > 0 and S.pace_ok and fork_kind in PACE_KINDS
    pmode = "p"
    if pace and PACE_MAX_WIN > 0 and sum(tk for _p, tk, _n in segs) > PACE_MAX_WIN:
        pace = PACE_BIG_GBPS > 0
        pmode = "q"  # paced at PACE_BIG_GBPS
    if pace and PACE_HEAD > 0:
        head, tail = _cap_named(segs, PACE_HEAD)
    elif pace:
        head, tail = [], list(segs)
    else:
        head, tail = list(segs), []
    out = []
    bmode = "d" if BURST_DELAY_NS > 0 and fork_kind in BURST_DELAY_KINDS else "b"
    for mode, ss in ((bmode, head), (pmode, tail)):
        if not ss:
            continue
        t = torch.tensor([v for p, tk, _n in ss for v in (p, tk)], dtype=torch.int64, device=device)
        S.keepalive.append(t)  # a captured graph keeps reading this table
        out.append((mode, t, len(ss), sum(tk for _p, tk, _n in ss)))
    return out


def make_g2_plan(events, sel, device="cuda") -> dict:
    """g2 colltrim plan attached to make_plan's result as plan["g2"] (see the module doc)."""
    fl = g2_fork_lists(events, sel)
    kinds = fl["kind"]
    pre = {idx: _parts(segs, kinds[idx], device) for idx, segs in fl["pre"].items()}
    post = {idx: _parts(segs, "post", device) for idx, segs in fl["post"].items()}
    attn = {lin: _parts(segs, "attn", device) for lin, segs in fl["attn"].items()}
    nb = collections.Counter()
    nforks = collections.Counter()
    for where, d in (("pre", pre), ("post", post), ("attn", attn)):
        for parts in d.values():
            for mode, _t, _n, b in parts:
                nb[f"{where}_{mode}"] += b
                nforks[f"{where}_{mode}"] += 1
    summary = json.dumps({"bytes_mb": {k: round(v / 2**20, 2) for k, v in sorted(nb.items())},
                          "moved_mb": {k: round(v / 2**20, 2) for k, v in sorted(fl["moved"].items())},
                          "forks": dict(sorted(nforks.items()))}, sort_keys=True)
    desc = []
    for idx in sorted(fl["pre"])[:3]:
        desc.append(f"[{kinds[idx]}{idx} " + " ".join(f"{n}:{t / 2**20:.2f}" for _p, t, n in fl["pre"][idx]) + "]")
    for idx in sorted(fl["post"])[:1]:
        desc.append(f"[post{idx} " + " ".join(f"{n}:{t / 2**20:.2f}" for _p, t, n in fl["post"][idx]) + "]")
    for lin in sorted(fl["attn"])[:1]:
        desc.append(f"[L{lin} " + " ".join(f"{n}:{t / 2**20:.2f}" for _p, t, n in fl["attn"][lin]) + "]")
    gkey = ("g2",
            tuple(sorted((i, tuple(s[:2] for s in v)) for i, v in fl["pre"].items())),
            tuple(sorted((i, tuple(s[:2] for s in v)) for i, v in fl["post"].items())),
            tuple(sorted((i, tuple(s[:2] for s in v)) for i, v in fl["attn"].items())),
            PACE_GBPS if S.pace_ok else 0.0, PACE_HEAD, tuple(sorted(PACE_KINDS)),
            PACE_MAX_WIN, PACE_BIG_GBPS, BURST_DELAY_NS, tuple(sorted(BURST_DELAY_KINDS)))
    return {"pre": pre, "post": post, "attn": attn, "bytes": sum(nb.values()), "summary": summary,
            "desc": " ".join(desc), "gkey": gkey, "nforks": json.dumps(dict(sorted(nforks.items()))),
            "nl": fl["nl"]}


def _join3() -> None:
    p = S.pending3
    if p is not None:
        torch.cuda.current_stream().wait_stream(p)
        S.pending3 = None


def _fork_part(part, where: str) -> None:
    """g2: launch one planned part. Burst parts at collective forks use the usual side stream (joined at
    the next collective, as today); burst parts at attn forks use the WOA stream (joined at the end of
    the forward); paced parts use the pacer stream (joined at the end of the forward)."""
    mode, t, nseg, _nb = part
    main = torch.cuda.current_stream()
    if mode in ("p", "q") and S.pace_ok:
        side = _side_stream3()
        side.wait_stream(main)
        period_ps = int(round(PACE_CHUNK * 1000.0 / (PACE_BIG_GBPS if mode == "q" else PACE_GBPS)))
        rc = _load_lib_g2().dsv41_l2pf_paced(t.data_ptr(), nseg, PACE_CHUNK, period_ps, PACE_DELAY_NS,
                                             PACE_MAX_NS, None, side.cuda_stream)
        if rc:
            raise RuntimeError(f"DSV41_L2_PREFETCH_PACE: paced prefetch launch failed (cuda error {rc})")
        S.pending3 = side
    else:
        side = _side_stream2() if where == "attn" else _side_stream()
        side.wait_stream(main)
        if mode == "d":
            rc = _load_lib().dsv41_spin(BURST_DELAY_NS, 1, side.cuda_stream)
            if rc:
                raise RuntimeError(f"DSV41_L2_PREFETCH_BURST_DELAY: spin launch failed (cuda error {rc})")
        rc = _load_lib().dsv41_l2pf(t.data_ptr(), nseg, CHUNK, side.cuda_stream)
        if rc:
            raise RuntimeError(f"DSV41_L2_PREFETCH: prefetch launch failed (cuda error {rc})")
        if where == "attn":
            S.pending2 = side
        else:
            S.pending = side
    S.g2_forks[f"{where}_{mode}"] += 1


# ----------------------------------------------------------------------------- collective --
def _join() -> None:
    p = S.pending
    if p is not None:
        torch.cuda.current_stream().wait_stream(p)
        S.pending = None


def _join2() -> None:
    p = S.pending2
    if p is not None:
        torch.cuda.current_stream().wait_stream(p)
        S.pending2 = None


def _fork2(item) -> None:
    """g1 WOA: like _fork, on the second side stream; joined only at the end of the forward."""
    t, nseg, _ = item
    main = torch.cuda.current_stream()
    side = _side_stream2()
    side.wait_stream(main)
    rc = _load_lib().dsv41_l2pf(t.data_ptr(), nseg, CHUNK, side.cuda_stream)
    if rc:
        raise RuntimeError(f"DSV41_L2_PREFETCH_WOA: prefetch launch failed (cuda error {rc})")
    S.pending2 = side
    S.forks2 += 1


def _fork(item) -> None:
    t, nseg, _ = item
    main = torch.cuda.current_stream()
    side = _side_stream()
    side.wait_stream(main)
    rc = _load_lib().dsv41_l2pf(t.data_ptr(), nseg, CHUNK, side.cuda_stream)
    if rc:
        raise RuntimeError(f"DSV41_L2_PREFETCH: prefetch launch failed (cuda error {rc})")
    S.pending = side
    S.forks += 1


def before_collective(kind: str) -> None:
    """Called by RdmaAllReduce right before it launches an all-reduce ("ar") / all-gather ("ag"),
    on the stream the collective runs on."""
    if S.disabled:
        return
    if not S.installed:
        if S.install_tries < _MAX_INSTALL_TRIES:
            _try_install()
    if S.sig is None:
        return
    _join()
    S.idx += 1
    if S.mode == "learn":
        if S.events is not None:
            S.events.append(("c", S.idx, kind))
        return
    plan = S.plans.get(S.sig)
    if plan is None or S.idx > plan["n"]:
        return
    if kind == "ag" and not USE_AG:
        return
    g2 = plan.get("g2")
    if g2 is not None:
        for part in g2["pre"].get(S.idx, ()):
            _fork_part(part, "pre")
        return
    item = plan["win"].get(S.idx)
    if item is not None:
        _fork(item)


def after_collective(kind: str) -> None:
    """g2 colltrim: called by RdmaAllReduce right AFTER it launched collective S.idx (same stream). Forks
    the planned "post" parts, which then start when the collective kernel has finished."""
    if not G2 or S.disabled or S.sig is None or S.mode != "capture":
        return
    plan = S.plans.get(S.sig)
    if plan is None or S.idx > plan["n"]:
        return
    g2 = plan.get("g2")
    if g2 is None:
        return
    for part in g2["post"].get(S.idx, ()):
        _fork_part(part, "post")
