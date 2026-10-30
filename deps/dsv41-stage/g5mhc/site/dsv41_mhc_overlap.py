# SPDX-License-Identifier: Apache-2.0
"""DSV41_MHC_OVERLAP: bitwise mHC boundary overlap for the DSv4.1 decode graphs on sm_12x.

DEFAULT OFF. model.py imports this module only when DSV41_MHC_OVERLAP (or DSV41_MHC_OVERLAP_DUMP)
is set, so with the variable unset nothing here runs, no stream is created, and model.py
behaves exactly as its unpatched version (live 9c2d22ad).

Why it can be exact. Between two sublayers the served decode graph runs, on one stream,
    AR -> mhc_fused_tilelang (bf16 streams + fn projection partials) -> mhc_pre_big_fuse_with_norm
          ("fused": split sums, sigmoids, 20x Sinkhorn, collapse with the CARRIED pre-mix, RMSNorm, aux)
and only the bf16 streams, the collapse, the RMSNorm and the aux are needed by the next sublayer.
The projection and the Sinkhorn produce post/comb/next_pre, which are first read at the NEXT boundary.
This module keeps every arithmetic op of the served path and only moves the coefficient half onto a
side stream (the decoder layer already joins mhc_stream after each sublayer):

  mode "stock" (no new arithmetic; the served path's own kernels):
    caller stream : mhc_post_tilelang                      -> residual_cur (bf16 streams)
                    MHC_PRE_NORM_KERNEL split_mode="input"  -> layer_input (+ aux when capture_aux)
    side stream   : <= 32 tokens: mhc_fused_tilelang (the served projection; its bf16 copy is discarded)
                    >  32 tokens: DeepGEMM tf32 prenorm GEMM on residual_cur (exactly the served > 32 path)
                    MHC_PRE_NORM_KERNEL split_mode="stats"  -> post, comb, next_pre
    Bitwise iff three codegen equalities hold on the cluster's TileLang/nvcc build (tests/test_mhc_overlap_bitwise.py):
      S1  mhc_post_tilelang bf16 == mhc_fused_tilelang bf16 residual (<= 32 tokens; > 32 is the same kernel)
      S2  "input" collapse/RMSNorm/aux == "fused" collapse/RMSNorm/aux (same TileLang branch, other split mode)
      S3  "stats" post/comb/next_pre == "fused" coefficients (same TileLang branch, other split mode)
  mode "moet" (vLLM-Moet b0dc737 mhc_post_norm on the caller stream: one launch replicating S1+S2's
    arithmetic; Apache-2.0, sources in DSV41_MHC_OVERLAP_MOET_DIR). Only after the bitwise test passes on
    this build with the post-mix contraction mode it reports (DSV41_MHC_OVERLAP_MOET_MODE).

Knobs (all read once at import):
  DSV41_MHC_OVERLAP            off (default) | stock | moet
  DSV41_MHC_OVERLAP_MAX_TOKENS 64   FULL-graph captures with 1..N tokens take the overlap (stock vLLM: 16)
  DSV41_MHC_OVERLAP_PROJ       fused (default: the served kernel AND tiling, 96 CTAs/token)
                               narrow (same TileLang source, tile_n=24 -> 8 CTAs/token; per-output reduction
                                       order unchanged by construction, test S4)
                               moet (Moet mhc_proj replica, 8*ceil(T/4) CTAs; test S5)
  DSV41_MHC_OVERLAP_PRE        0 (default): first-layer / Engram / drafter-first mhc_pre stay served (empty fork)
                               1: they take vLLM's stock mhc_pre_delayed_overlap (same GEMM, split epilogue; S2/S3)
  DSV41_MHC_OVERLAP_JOIN       boundary (default: the decoder layer's own join after the sublayer)
                               ar (experimental: join right before the sublayer's all-reduce via the rdma_ar
                                   L2-prefetch before_collective hook, so the boundary kernel keeps one parent)
  DSV41_MHC_OVERLAP_MOET_DIR   ~/dsv41-stage/moet/src/vllm-moet-b0dc737/tools/dsv41_sm120/mhc_overlap
  DSV41_MHC_OVERLAP_MOET_MODE  1 (Moet's sm_120 default; use what the test reports for sm_121)
  DSV41_MHC_OVERLAP_DUMP       <dir>: test aid - torch.save eager (never captured) shifted-boundary inputs of
                               this rank for the recorded-input bitwise test; nothing else changes.
  DSV41_MHC_OVERLAP_DUMP_N     8 calls per rank (plus the first capture_aux call), DUMP_RANKS "0"
  DSV41_MHC_OVERLAP_DUMP_ARM   /dev/shm/dsv41-mhc-dump.arm: the dump runs only while this file exists
                               (g5mhc: so the startup profile and warm-up dummy forwards never dump)
  DSV41_MHC_OVERLAP_DUMP_ROWS  512 (first rows of each dumped call), DUMP_EVERY 9 (every 9th eligible call)

Not used here on purpose: fuse_mhc_all_reduce (TP4-only in vLLM 0.32; our TP8 rdma_ar stays in the linear
layers), stock mhc_shifted_post_pre(stream=...) (tf32 projection on bf16 streams + residual.mean aux: not
bitwise), Moet's P2P all-reduce consumer (we run rdma_ar).
"""

from __future__ import annotations

import functools
import os
import sys

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


MODE = {"": "off", "0": "off", "off": "off", "1": "stock", "stock": "stock", "moet": "moet"}.get(
    _env("DSV41_MHC_OVERLAP", "0"), "off"
)
ACTIVE = MODE != "off"
MAX_TOKENS = max(1, int(os.environ.get("DSV41_MHC_OVERLAP_MAX_TOKENS", "64") or 64))
PROJ = _env("DSV41_MHC_OVERLAP_PROJ", "fused")
if PROJ not in ("fused", "narrow", "moet"):
    PROJ = "fused"
OVERLAP_PRE = _env("DSV41_MHC_OVERLAP_PRE", "0") == "1"
JOIN = _env("DSV41_MHC_OVERLAP_JOIN", "boundary")
if JOIN not in ("boundary", "ar"):
    JOIN = "boundary"
MOET_DIR = os.path.expanduser(
    os.environ.get(
        "DSV41_MHC_OVERLAP_MOET_DIR",
        "~/dsv41-stage/moet/src/vllm-moet-b0dc737/tools/dsv41_sm120/mhc_overlap",
    )
)
MOET_MODE = int(os.environ.get("DSV41_MHC_OVERLAP_MOET_MODE", "1") or 1)
NARROW_TILE_N = int(os.environ.get("DSV41_MHC_OVERLAP_TILE_N", "24") or 24)
DUMP_DIR = os.environ.get("DSV41_MHC_OVERLAP_DUMP", "")
DUMP_N = int(os.environ.get("DSV41_MHC_OVERLAP_DUMP_N", "8") or 8)
DUMP_RANKS = {int(r) for r in os.environ.get("DSV41_MHC_OVERLAP_DUMP_RANKS", "0").split(",") if r.strip()}
# g5mhc: the dump is ARMED by a control file (so the startup profile / warm-up dummy forwards never dump),
# keeps the first DUMP_ROWS rows of a call and takes every DUMP_EVERY-th eligible call (spread over layers).
DUMP_ARM = os.environ.get("DSV41_MHC_OVERLAP_DUMP_ARM", "/dev/shm/dsv41-mhc-dump.arm")
DUMP_ROWS = int(os.environ.get("DSV41_MHC_OVERLAP_DUMP_ROWS", "512") or 512)
DUMP_EVERY = max(1, int(os.environ.get("DSV41_MHC_OVERLAP_DUMP_EVERY", "9") or 9))

_STREAMS: list[torch.cuda.Stream] = []
_MOET = None  # the mhc_post_norm_sm120 module once loaded
_DUMPED = {"n": 0, "aux": False, "seen": 0}
_JOIN_HOOKED = False


def configure(**kw) -> None:
    """Test hook: override the module knobs (MODE, PROJ, MAX_TOKENS, OVERLAP_PRE, JOIN, MOET_MODE, ...)."""
    g = globals()
    for k, v in kw.items():
        assert k in g, k
        g[k] = v
    g["ACTIVE"] = g["MODE"] != "off"


# --------------------------------------------------------------------------------------------- gating
def supported(vllm_config) -> bool:
    """The side stream is created only for the configuration this module was proven for."""
    if not ACTIVE:
        return False
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    from .ops.mhc import supports_mhc_all_reduce

    cfg = vllm_config.model_config.hf_config
    reasons = []
    if not (current_platform.is_cuda() and current_platform.is_device_capability_family(120)):
        reasons.append("not sm_12x")
    if cfg.hc_mult != 4 or cfg.hidden_size % 1024 or not 1024 <= cfg.hidden_size <= 5120:
        reasons.append(f"hc_mult={cfg.hc_mult} hidden={cfg.hidden_size}")
    if vllm_config.parallel_config.use_ubatching:
        reasons.append("ubatching")
    if not is_deep_gemm_supported():
        reasons.append("DeepGEMM off (the > 32-token projection is its tf32 prenorm GEMM)")
    try:
        if supports_mhc_all_reduce(vllm_config):
            reasons.append("the stock fused mHC all-reduce would engage (TP4); this path keeps it off")
    except Exception:  # noqa: BLE001 - TP group not up: TP != 4 returns before touching it
        pass
    if reasons:
        logger.warning("DSV41_MHC_OVERLAP=%s disabled: %s", MODE, "; ".join(reasons))
        return False
    return True


def make_stream(vllm_config) -> torch.cuda.Stream | None:
    """One side stream per model (target, drafter); kernels warmed here, never inside a capture."""
    try:
        cfg = vllm_config.model_config.hf_config
        _prepare(cfg.hidden_size, cfg.hc_mult)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DSV41_MHC_OVERLAP=%s disabled: warm-up failed: %r", MODE, exc)
        return None
    s = torch.cuda.Stream()
    s._dsv41_mhc_pending = None
    s._dsv41_mhc_open = False
    _STREAMS.append(s)
    _maybe_hook_join()
    logger.info(
        "DSV41_MHC_OVERLAP=%s: mHC coefficients on a side stream in FULL decode graphs of 1..%d tokens "
        "(proj=%s, pre=%s, join=%s%s)",
        MODE, MAX_TOKENS, PROJ, int(OVERLAP_PRE), JOIN,
        f", moet mode {MOET_MODE}" if MODE == "moet" else "",
    )
    return s


def _moet():
    global _MOET
    if _MOET is None:
        if MOET_DIR not in sys.path:
            sys.path.insert(0, MOET_DIR)
        import mhc_post_norm_sm120 as m  # vLLM-Moet b0dc737 (Apache-2.0)

        # g5mhc: Moet's _ext() (lru_cached, runs once) sets TORCH_CUDA_ARCH_LIST process-wide; restore it so no
        # later torch cpp_extension build in this worker (e.g. the m32-bw kernels) sees a changed arch list.
        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        try:
            m._ext()
        finally:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
        _MOET = m
    return _MOET


def _prepare(hidden: int, hc: int) -> None:
    """Compile what the capture will launch: TileLang JIT keys outside the warm-up registry and the Moet
    extension (whose [T, H] workspace must not be re-grown inside a capture)."""
    dev = torch.device("cuda", torch.cuda.current_device())
    mix = hc * (hc + 2)
    if PROJ == "narrow":
        for n in (1, 16):  # both served fused configs share n_thr=128, n_splits=8 at hidden 5120
            t = _dummy(n, hc, hidden, mix, dev)
            _proj_narrow(t["x"], t["residual"], t["post"], t["comb"], t["fn"], hc, hidden, mix)
    if MODE == "moet" or PROJ == "moet":
        m = _moet()
        t = _dummy(MAX_TOKENS, hc, hidden, mix, dev)
        if MODE == "moet":
            m.mhc_post_norm(t["x"], t["residual"], t["post"], t["comb"], t["pre"], t["w"], 1e-6,
                            capture_aux=True, mode=MOET_MODE)
        if PROJ == "moet":
            tt = _dummy(1, hc, hidden, mix, dev)
            m.mhc_proj(tt["x"], tt["residual"], tt["post"], tt["comb"], tt["fn"])
    torch.cuda.synchronize(dev)


def _dummy(n, hc, hidden, mix, dev):
    return dict(
        x=torch.zeros(n, hidden, dtype=torch.bfloat16, device=dev),
        residual=torch.zeros(n, hc, hidden, dtype=torch.bfloat16, device=dev),
        post=torch.zeros(n, hc, 1, dtype=torch.float32, device=dev),
        comb=torch.zeros(n, hc, hc, dtype=torch.float32, device=dev),
        pre=torch.full((n, hc), 0.25, dtype=torch.float32, device=dev),
        fn=torch.zeros(mix, hc * hidden, dtype=torch.float32, device=dev),
        w=torch.ones(hidden, dtype=torch.bfloat16, device=dev),
    )


# ------------------------------------------------------------------------------------- fork / join
def empty_fork(stream: torch.cuda.Stream) -> None:
    """Bring the side stream into the capture without work, so the decoder layer's join stays legal."""
    stream.wait_stream(torch.cuda.current_stream())
    stream._dsv41_mhc_open = True


def join(stream: torch.cuda.Stream) -> None:
    """The decoder layer's join (stock placement: after the sublayer). With JOIN=ar the rdma_ar hook may
    already have joined right before the sublayer's all-reduce; then nothing is left to join."""
    if JOIN == "ar" and not getattr(stream, "_dsv41_mhc_open", True):
        return
    torch.cuda.current_stream().wait_stream(stream)
    stream._dsv41_mhc_open = False


def served_pre(served_fn, *args, stream: torch.cuda.Stream, **kwargs):
    """DSV41_MHC_OVERLAP_PRE=0: the first-layer / Engram mhc_pre stays the served kernel pair."""
    empty_fork(stream)
    return served_fn(*args, **kwargs)


def overlap_pre(stock_overlap_fn, *args, stream: torch.cuda.Stream, **kwargs):
    """DSV41_MHC_OVERLAP_PRE=1: vLLM's own mhc_pre_delayed_overlap (it forks `stream` itself: the same
    tf32 prenorm GEMM as the served mhc_pre_delayed_tilelang, epilogue split into "input"/"stats").
    Caveats: it uses record_stream (blocks pinned until the capture ends: 3 boundaries per forward), and
    the L2-prefetch wrapper only wraps mhc_pre_delayed_tilelang, so the learned prefetch plan loses these
    three fn reads (cache hint only, not numerics)."""
    stream._dsv41_mhc_pending = None
    out = stock_overlap_fn(*args, stream=stream, **kwargs)
    stream._dsv41_mhc_open = True
    return out


def _maybe_hook_join() -> None:
    """JOIN=ar: join every open overlap stream right before the next all-reduce on the caller stream
    (rdma_all_reduce calls dsv41_l2_prefetch.before_collective('ar') right before launching it)."""
    global _JOIN_HOOKED
    if JOIN != "ar" or _JOIN_HOOKED:
        return
    try:
        from vllm.distributed.device_communicators import dsv41_l2_prefetch as l2pf
    except Exception as exc:  # noqa: BLE001
        logger.warning("DSV41_MHC_OVERLAP_JOIN=ar unavailable (%r): joining at the boundary", exc)
        return
    orig = l2pf.before_collective

    def before_collective(kind: str) -> None:
        if kind == "ar" and torch.cuda.is_current_stream_capturing():
            for s in _STREAMS:
                if getattr(s, "_dsv41_mhc_open", False):
                    torch.cuda.current_stream().wait_stream(s)
                    s._dsv41_mhc_open = False
        orig(kind)

    l2pf.before_collective = before_collective
    _JOIN_HOOKED = True


# ------------------------------------------------------------------------------------------ kernels
def _proj_narrow(x, residual, post_layer_mix, comb_res_mix, fn, hc, hidden, mix):
    """The served projection (mhc_fused_tilelang) with tile_n=NARROW_TILE_N outputs per CTA: the same
    k-split (n_splits, n_thr from the served config), the same thread->position mapping and the same
    per-output fma chain / warp butterfly / cross-warp order; only how many outputs one CTA owns changes.
    Returns (mixes [S, n, mix], sqrsum [S, n], residual copy [n, hc, H] that nobody reads)."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_fused_tilelang,
        require_fused_post_pre_config,
    )

    n = residual.shape[0]
    _served_tile_n, n_splits, n_thr = require_fused_post_pre_config(n, hidden, hc)
    yp = residual.new_empty((n_splits, n, mix), dtype=torch.float32)
    rp = residual.new_empty((n_splits, n), dtype=torch.float32)
    dup = residual.new_empty(residual.shape)
    mhc_fused_tilelang(
        comb_res_mix, residual, post_layer_mix.view(n, hc), x, fn.view(mix, hc, hidden),
        yp, rp, dup, hc, hidden, mix, n_thr=n_thr, tile_n=NARROW_TILE_N,
    )
    return yp, rp, dup


def applicable(x, residual, pre_mix, norm_weight) -> bool:
    if not isinstance(x, torch.Tensor) or pre_mix is None or norm_weight is None:
        return False
    if residual.dim() != 3 or x.dim() != 2:
        return False
    n, hc, hidden = residual.shape
    return (
        0 < n <= MAX_TOKENS
        and hc == 4
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and tuple(x.shape) == (n, hidden)
        and pre_mix.dtype == torch.float32
        and pre_mix.is_contiguous()
        and pre_mix.numel() == n * hc
        and norm_weight.dtype == torch.bfloat16
        and norm_weight.numel() == hidden
    )


def shifted_post_pre_overlap(
    x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix, norm_weight, norm_eps,
    capture_aux, stream,
):
    """The served mhc_fused_post_pre_delayed_tilelang, split across the caller stream and `stream`.

    Returns what the served function returns: (residual_cur, post [n,hc,1], comb [n,hc,hc],
    layer_input, next_pre, aux). Only residual_cur, layer_input and aux are ready on the caller stream;
    the caller joins `stream` before reading post/comb/next_pre (the decoder layer does, after the sublayer).
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_FUSED_TILELANG_KERNEL,
        _MHC_POST_TILELANG_KERNEL,
        mhc_fused_post_pre_split_config,
    )
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    n, hc, hidden = residual.shape
    input_size = hc * hidden
    mix = hc * (hc + 2)
    dev = residual.device
    post2d = post_layer_mix.view(n, hc)
    fused_config = mhc_fused_post_pre_split_config(n, hidden, hc)
    # The epilogue is compiled per split count: the served path's (8 for the fused kernel, else the
    # bucketed DeepGEMM split), so "input"/"stats" hit the keys the warm-up registered.
    n_splits = fused_config[1] if fused_config is not None else compute_mhc_pre_num_splits(input_size, n)
    post = torch.empty((n, hc), device=dev, dtype=torch.float32)
    comb = torch.empty((n, hc * hc), device=dev, dtype=torch.float32)
    next_pre = torch.empty_like(post)
    epi = dict(
        hidden_size=hidden, rms_eps=rms_eps, hc_pre_eps=hc_pre_eps, hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value, sinkhorn_repeat=sinkhorn_repeat, norm_eps=norm_eps,
        hc_mult=hc, use_pre_mix_in=True, save_pre_mix=True, rms_numel=input_size,
    )

    # ---- critical path, caller stream ----
    if MODE == "moet":
        residual_cur, layer_input, aux = _moet().mhc_post_norm(
            x, residual, post_layer_mix, comb_res_mix, pre_mix, norm_weight, norm_eps,
            capture_aux=capture_aux, mode=MOET_MODE,
        )
        unread = ()
    else:
        residual_cur = torch.empty_like(residual)
        _MHC_POST_TILELANG_KERNEL(comb_res_mix, residual, post2d, x, residual_cur, hc, hidden)
        layer_input = torch.empty((n, hidden), device=dev, dtype=torch.bfloat16)
        aux = torch.empty((n if capture_aux else 0, hidden), device=dev, dtype=torch.bfloat16)
        # "input" mode reads neither the split partials nor the coefficient outputs: shape-only stand-ins.
        mix_unread = torch.empty((n_splits, n, mix), device=dev, dtype=torch.float32)
        sqr_unread = torch.empty((n_splits, n), device=dev, dtype=torch.float32)
        MHC_PRE_NORM_KERNEL(
            mix_unread, sqr_unread, hc_scale, hc_base, residual_cur, post, comb, layer_input,
            norm_weight, pre_mix, next_pre, aux if capture_aux else layer_input,
            split_mode="input", write_aux=capture_aux, **epi,
        )
        unread = (mix_unread, sqr_unread)

    # ---- coefficients, side stream (forked AFTER the critical kernels: they stay the AR's only child) ----
    main = torch.cuda.current_stream()
    stream.wait_stream(main)
    stream._dsv41_mhc_open = True
    with torch.cuda.stream(stream):
        if fused_config is not None:
            if PROJ == "narrow":
                mixes, sqrsum, dup = _proj_narrow(x, residual, post_layer_mix, comb_res_mix, fn, hc, hidden, mix)
            elif PROJ == "moet" and _moet().mhc_proj_applicable(x, residual):
                mixes, sqrsum = _moet().mhc_proj(x, residual, post_layer_mix, comb_res_mix, fn)
                dup = None
            else:
                mixes, sqrsum, dup = _MHC_FUSED_TILELANG_KERNEL(
                    comb_res_mix, residual, post2d, x, fn.view(mix, hc, hidden), hc, hidden, mix
                )
        else:
            mixes = torch.empty((n_splits, n, mix), device=dev, dtype=torch.float32)
            sqrsum = torch.empty((n_splits, n), device=dev, dtype=torch.float32)
            tf32_hc_prenorm_gemm(residual_cur.view(n, input_size), fn, mixes, sqrsum, n_splits)
            dup = None
        MHC_PRE_NORM_KERNEL(
            mixes, sqrsum, hc_scale, hc_base, residual_cur, post, comb, layer_input,
            norm_weight, pre_mix, next_pre, layer_input,
            split_mode="stats", write_aux=False, **epi,
        )
    # Everything the side stream reads or writes stays referenced until the next boundary, by which point
    # the decoder layer has joined (record_stream would pin these blocks until the end of the capture).
    stream._dsv41_mhc_pending = (
        x, residual, post_layer_mix, comb_res_mix, pre_mix, residual_cur, layer_input,
        mixes, sqrsum, dup, post, comb, next_pre, unread,
    )
    return residual_cur, post.unsqueeze(-1), comb.view(n, hc, hc), layer_input, next_pre, aux


def wrap_shifted_post_pre(orig):
    """model.py rebinds its module-global mhc_shifted_post_pre to this dispatcher (so the L2-prefetch
    wrapper, which wraps that global by name, wraps the dispatcher; fn stays positional argument 4)."""

    @functools.wraps(orig)
    def mhc_shifted_post_pre(
        x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=None, norm_weight=None,
        norm_eps=1e-6, capture_aux=False, *, stream=None, reduce_results=False,
    ):
        if stream is None:
            if DUMP_DIR:
                _maybe_dump(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                            pre_mix, norm_weight, norm_eps, capture_aux,
                            (rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat))
            return orig(
                x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix, norm_weight, norm_eps,
                capture_aux, stream=None, reduce_results=reduce_results,
            )
        # A stream reaches this point only from a FULL-graph capture the decoder layer approved; the
        # previous boundary's side-stream inputs were joined after its sublayer and can be released.
        stream._dsv41_mhc_pending = None
        if not ACTIVE or reduce_results or not applicable(x, residual, pre_mix, norm_weight):
            empty_fork(stream)  # the served path; the caller's join stays legal
            return orig(
                x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix, norm_weight, norm_eps,
                capture_aux, stream=None, reduce_results=reduce_results,
            )
        return shifted_post_pre_overlap(
            x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix, norm_weight, norm_eps,
            capture_aux, stream,
        )

    mhc_shifted_post_pre._dsv41_mhc_overlap = True
    return mhc_shifted_post_pre


# -------------------------------------------------------------------------------------------- dump
def _maybe_dump(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, pre_mix, norm_weight,
                norm_eps, capture_aux, eps) -> None:
    """Recorded inputs for tests/test_mhc_overlap_bitwise.py --dump: eager prefill boundaries only."""
    if torch.cuda.is_current_stream_capturing() or not isinstance(x, torch.Tensor) or pre_mix is None:
        return
    if x.shape[0] < 64 or norm_weight is None:
        return
    if _DUMPED["n"] >= DUMP_N and _DUMPED["aux"]:
        return
    if not os.path.exists(DUMP_ARM):  # g5mhc: armed by a control file only (never during startup)
        return
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001
        rank = 0
    if rank not in DUMP_RANKS:
        return
    seen = _DUMPED["seen"]
    _DUMPED["seen"] += 1
    want = (_DUMPED["n"] < DUMP_N and seen % DUMP_EVERY == 0) or (capture_aux and not _DUMPED["aux"])
    if not want:
        return
    d = os.path.join(DUMP_DIR, f"rank{rank}")
    os.makedirs(d, exist_ok=True)
    k = _DUMPED["n"]
    r = min(int(x.shape[0]), DUMP_ROWS)

    def cpu(t):
        return t[:r].detach().contiguous().cpu()

    blob = dict(
        x=cpu(x), residual=cpu(residual), post_layer_mix=cpu(post_layer_mix),
        comb_res_mix=cpu(comb_res_mix), pre_mix=cpu(pre_mix), fn=fn.detach().cpu(),
        hc_scale=hc_scale.detach().cpu(), hc_base=hc_base.detach().cpu(), norm_weight=norm_weight.detach().cpu(),
        norm_eps=float(norm_eps), capture_aux=bool(capture_aux), eps=tuple(float(e) for e in eps[:4]) + (int(eps[4]),),
        call=seen, rows_total=int(x.shape[0]),
    )
    torch.save(blob, os.path.join(d, f"b{k:03d}_c{seen:03d}{'_aux' if capture_aux else ''}.pt"))
    _DUMPED["n"] += 1
    if capture_aux:
        _DUMPED["aux"] = True
