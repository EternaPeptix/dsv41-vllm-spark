# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import typing
from collections.abc import Callable, Iterable
from functools import partial
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.kernel import MEGA_MOE_BACKENDS, NATIVE_MEGA_MOE_BACKENDS
from vllm.distributed import (
    get_engram_dp_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import (
    get_forward_context,
    in_piecewise_cudagraph,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4ExpertsModular,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _unpack
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    drop_checkpoint_cache,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MoE as DeepseekV4MoEBase,
)
from vllm.models.deepseek_v4.nvidia.model import (
    MegaGateRoutingMetadata,
    make_deepseek_v4_expert_params_mapping,
    prepare_mega_gate_routing_metadata,
)
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.decoder_replay_layers import G4P_STEP, DecoderReplayLayers
from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
    DeepseekV4MegaAttnAttention,
)
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLAAttention,
    DeepseekV4FlashInferSM120Attention,
)
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from ..common.engram import (
    Engram,
    EngramBatch,
    EngramLayout,
    NgramHashState,
    can_share_engram_tables,
    gather_engram_hashes,
)
from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID, image_sentinel_mask
from .ops.mhc import (
    MHC_OVERLAP_MAX_TOKENS,
    init_mhc_all_reduce,
    mhc_pre_delayed_overlap,
    mhc_shifted_post_pre,
    supports_mhc_all_reduce,
    supports_mhc_overlap,
)

# DSV41_MHC_OVERLAP (default off): bitwise mHC boundary overlap on sm_12x, see
# dsv41_mhc_overlap.py. With DSV41_MHC_OVERLAP and DSV41_MHC_OVERLAP_DUMP unset nothing below
# runs: _DSV41_MHC_OVL stays None and every use of it in this file reduces to the unpatched code.
_DSV41_MHC_OVL = None
if os.environ.get("DSV41_MHC_OVERLAP", "0").strip().lower() not in (
    "",
    "0",
    "off",
) or os.environ.get("DSV41_MHC_OVERLAP_DUMP", ""):
    from . import dsv41_mhc_overlap as _DSV41_MHC_OVL

    # Rebinding the module global keeps the L2-prefetch wrapper (it wraps this name, fn at
    # position 4) in front of the dispatcher.
    mhc_shifted_post_pre = _DSV41_MHC_OVL.wrap_shifted_post_pre(mhc_shifted_post_pre)


def _dsv41_mhc_max_tokens() -> int:
    return MHC_OVERLAP_MAX_TOKENS if _DSV41_MHC_OVL is None else _DSV41_MHC_OVL.MAX_TOKENS

if typing.TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)

# r4 small-kernels flags (default off; see vllm/models/deepseek_v41/nvidia/dsv41_r4_small.py)
_DSV41_R4 = any(
    __import__("os").environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
    for k in ("DSV41_R4_CERT_HEAD", "DSV41_R4_MAINPROJ_SPLIT")
)

# ---------------------------------------------------------------------------
# r4 prefill-sp (default OFF). DSV41_PREFILL_SP=1: for eager forwards with at
# least DSV41_PREFILL_SP_MIN_ROWS rows (prefill chunks; never CUDA-graph decode
# or verify, never the drafter), layers [start, decoder_replay_start) keep the
# hc residual stream sharded by rows across TP ranks. Attention and the MoE
# still run on all rows (all-gathered input) but return reduce-scattered
# partial sums instead of all-reduced ones, so the per-row work between them
# (mHC post/pre + norms, Engram gate) runs on 1/TP of the rows. The state is
# all-gathered once before the decoder-replay layers. Same collective bytes
# as the all-reduce path (RS + AG == ring AR); the sum order of the
# reduce-scatter differs from the all-reduce, so results are not bit-identical.
# Port of knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 adapter/prefill_sp.py.
# ---------------------------------------------------------------------------
import contextlib
import threading

_DSV41_SP_MODE = {
    "1": "fast",
    "fast": "fast",
    "exact": "exact",
    "comm": "comm",
}.get(os.environ.get("DSV41_PREFILL_SP", "0").strip().lower(), "off")
_DSV41_PREFILL_SP = _DSV41_SP_MODE != "off"
try:
    _DSV41_PREFILL_SP_MIN_ROWS = max(
        int(os.environ.get("DSV41_PREFILL_SP_MIN_ROWS", "2048") or 2048), 1
    )
except ValueError:
    _DSV41_PREFILL_SP_MIN_ROWS = 2048
try:
    _DSV41_PREFILL_SP_LAYERS = max(
        int(os.environ.get("DSV41_PREFILL_SP_LAYERS", "0") or 0), 0
    )
except ValueError:
    _DSV41_PREFILL_SP_LAYERS = 0
try:
    _DSV41_PREFILL_SP_DEBUG = int(os.environ.get("DSV41_PREFILL_SP_DEBUG", "0") or 0)
except ValueError:
    _DSV41_PREFILL_SP_DEBUG = 0
_DSV41_PREFILL_SP_LOGICAL_ROWS = (
    os.environ.get("DSV41_PREFILL_SP_LOGICAL_ROWS", "1").strip().lower()
    not in ("0", "off", "false", "no")
)
_DSV41_PREFILL_SP_SELF_CHECK_RAW = (
    os.environ.get(
        "DSV41_PREFILL_SP_SELF_CHECK",
        "1" if _DSV41_SP_MODE == "exact" else "0",
    )
    .strip()
    .lower()
)
_DSV41_PREFILL_SP_SELF_CHECK = _DSV41_PREFILL_SP_SELF_CHECK_RAW not in (
    "0", "off", "false", "no"
)
# g4p: "once" keys the check by (layer, op) only -- a startup canary; the logical-row
# kernels are row-invariant, so the verdict does not depend on the row count.
_DSV41_PREFILL_SP_SELF_CHECK_ONCE = _DSV41_PREFILL_SP_SELF_CHECK_RAW == "once"
_DSV41_PREFILL_SP_WKV_SHARD = (
    os.environ.get("DSV41_PREFILL_SP_WKV", "full").strip().lower() == "shard"
)
_DSV41_PREFILL_SP_DEBUG_REPLAY_TOPK = (
    os.environ.get("DSV41_PREFILL_SP_DEBUG_REPLAY_TOPK", "1").strip().lower()
    not in ("0", "off", "false", "no")
)
SP_LOGICAL_ROWS = True


class _Dsv41SpCtx(threading.local):
    def __init__(self) -> None:
        self.active: bool = False
        self.logical_rows: int = 0
        self.shard_rows: int = 0
        self.inner: int = 0
        self.debug_topk: (
            dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] | None
        ) = None
        self.layer_id: int = -1


_sp_ctx = _Dsv41SpCtx()
_SP_SELF_CHECK_OK: dict[tuple[int, str, int], bool] = {}
_SP_HOOKS_INSTALLED: bool = False


@contextlib.contextmanager
def _sp_inner():
    _sp_ctx.inner += 1
    try:
        yield
    finally:
        _sp_ctx.inner -= 1


@contextlib.contextmanager
def _sp_region(logical_rows: int, shard_rows: int):
    prev = (_sp_ctx.active, _sp_ctx.logical_rows, _sp_ctx.shard_rows)
    _sp_ctx.active = True
    _sp_ctx.logical_rows = int(logical_rows)
    _sp_ctx.shard_rows = int(shard_rows)
    try:
        yield
    finally:
        _sp_ctx.active, _sp_ctx.logical_rows, _sp_ctx.shard_rows = prev


def sp_logical_rows(x: torch.Tensor | int) -> int:
    """Row count a kernel heuristic/gate must use for `x`: the full unsharded
    chunk row count `T` when `x` is this rank's shard inside an SP prefill
    region, else `x.shape[0]` (or `int(x)`). Cheap and inert when SP is off."""
    n = int(x.shape[0]) if isinstance(x, torch.Tensor) else int(x)
    if (
        not _DSV41_PREFILL_SP_LOGICAL_ROWS
        or not _sp_ctx.active
        or _sp_ctx.inner > 0
    ):
        return n
    return _sp_ctx.logical_rows if n == _sp_ctx.shard_rows else n


def _install_sp_logical_rows_hooks() -> None:
    """Patch mHC row-count heuristics to query `sp_logical_rows()` when SP is
    enabled. Never called when `DSV41_PREFILL_SP` is unset/off."""
    global _SP_HOOKS_INSTALLED
    if _SP_HOOKS_INSTALLED:
        return
    from vllm.model_executor.kernels.mhc import tilelang_kernels as _tlk
    from vllm.model_executor.kernels.mhc import warmup as _mhc_warmup

    orig_splits = _mhc_warmup.compute_mhc_pre_num_splits
    if not getattr(orig_splits, "_dsv41_sp_wrapped", False):

        def _compute_mhc_pre_num_splits(input_size: int, num_tokens: int) -> int:
            return orig_splits(input_size, sp_logical_rows(num_tokens))

        _compute_mhc_pre_num_splits.__wrapped__ = orig_splits  # type: ignore[attr-defined]
        _compute_mhc_pre_num_splits._dsv41_sp_wrapped = True  # type: ignore[attr-defined]
        _mhc_warmup.compute_mhc_pre_num_splits = _compute_mhc_pre_num_splits

    orig_fused_cfg = _tlk.mhc_fused_post_pre_split_config
    if not getattr(orig_fused_cfg, "_dsv41_sp_wrapped", False):

        def _mhc_fused_post_pre_split_config(
            num_tokens: int, hidden_size: int, hc_mult: int
        ) -> tuple[int, int, int] | None:
            return orig_fused_cfg(sp_logical_rows(num_tokens), hidden_size, hc_mult)

        _mhc_fused_post_pre_split_config.__wrapped__ = orig_fused_cfg  # type: ignore[attr-defined]
        _mhc_fused_post_pre_split_config._dsv41_sp_wrapped = True  # type: ignore[attr-defined]
        _tlk.mhc_fused_post_pre_split_config = _mhc_fused_post_pre_split_config

    orig_num_split = _tlk.compute_num_split
    if not getattr(orig_num_split, "_dsv41_sp_wrapped", False):

        def _compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
            if (
                _DSV41_PREFILL_SP_LOGICAL_ROWS
                and _sp_ctx.active
                and _sp_ctx.inner == 0
                and grid_size == cdiv(_sp_ctx.shard_rows, 64)
            ):
                grid_size = cdiv(_sp_ctx.logical_rows, 64)
            return orig_num_split(block_k, k, grid_size)

        _compute_num_split.__wrapped__ = orig_num_split  # type: ignore[attr-defined]
        _compute_num_split._dsv41_sp_wrapped = True  # type: ignore[attr-defined]
        _tlk.compute_num_split = _compute_num_split

    orig_gemm_disp = _tlk._HC_PRENORM_GEMM_TILELANG_KERNEL.dispatch
    if not getattr(orig_gemm_disp, "_dsv41_sp_wrapped", False):

        def _gemm_dispatch(*, num_tokens: int, **kw):
            return orig_gemm_disp(num_tokens=sp_logical_rows(num_tokens), **kw)

        _gemm_dispatch.__wrapped__ = orig_gemm_disp  # type: ignore[attr-defined]
        _gemm_dispatch._dsv41_sp_wrapped = True  # type: ignore[attr-defined]
        _tlk._HC_PRENORM_GEMM_TILELANG_KERNEL.dispatch = _gemm_dispatch  # type: ignore[method-assign]

    _SP_HOOKS_INSTALLED = True


if _DSV41_PREFILL_SP:
    _install_sp_logical_rows_hooks()
    logger.warning(
        "DSV41_PREFILL_SP armed: eager forwards with >= %d rows run layers "
        "before the decoder replay sequence-parallel (mode=%s layers=%s "
        "logical_rows=%s self_check=%s debug=%d)",
        _DSV41_PREFILL_SP_MIN_ROWS,
        _DSV41_SP_MODE,
        _DSV41_PREFILL_SP_LAYERS or "all",
        _DSV41_PREFILL_SP_LOGICAL_ROWS,
        _DSV41_PREFILL_SP_SELF_CHECK,
        _DSV41_PREFILL_SP_DEBUG,
    )


def _g4p_bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-for-bit equality (NaN payloads included), unlike torch.equal on floats."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    return torch.equal(
        a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)
    )


def _g4p_real_forward() -> bool:
    """A served step: not captured, not a FlashInfer autotune / profiling dummy."""
    if torch.cuda.is_current_stream_capturing() or not is_forward_context_available():
        return False
    try:
        from flashinfer.autotuner import AutoTuner

        if getattr(AutoTuner.get(), "is_tuning_mode", False):
            return False
    except Exception:  # noqa: BLE001
        pass
    attn_metadata = get_forward_context().attn_metadata
    return isinstance(attn_metadata, dict) and bool(attn_metadata)


def _dsv41_sp_agree_min(local_ok: bool, device: torch.device) -> bool:
    """Rank-agreed AND (ReduceOp.MIN) across the TP group."""
    try:
        from vllm.distributed import get_tp_group

        grp = get_tp_group()
        dev_grp = getattr(grp, "device_group", None)
        if dev_grp is not None and grp.world_size > 1 and device.type == "cuda":
            flag = torch.tensor(
                [1 if local_ok else 0], dtype=torch.int32, device=device
            )
            torch.distributed.all_reduce(
                flag, op=torch.distributed.ReduceOp.MIN, group=dev_grp
            )
            return bool(flag.item())
    except Exception:  # noqa: BLE001
        pass
    return bool(local_ok)


def _dsv41_sp_reduce(x: torch.Tensor) -> torch.Tensor:
    """This rank's rows of the TP sum of x. exact: the stock all-reduce (same
    reduction order and bits) then this rank's rows; else a reduce-scatter."""
    if _DSV41_SP_MODE == "exact":
        return sp_shard(tensor_model_parallel_all_reduce(x))
    return sp_reduce_scatter(x)


def _dsv41_sp_clean_forward(positions: torch.Tensor) -> bool:
    """Debug compares are valid only when no request reads state written by an
    earlier chunk (every request starts at position 0) and the forward is a
    real step: not captured, not a FlashInfer autotune or profiling dummy."""
    if torch.cuda.is_current_stream_capturing() or not is_forward_context_available():
        return False
    try:
        from flashinfer.autotuner import AutoTuner

        if getattr(AutoTuner.get(), "is_tuning_mode", False):
            return False
    except Exception:  # noqa: BLE001
        pass
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict) or not attn_metadata:
        return False
    qsl = None
    for md in attn_metadata.values():
        qsl = getattr(md, "query_start_loc", None)
        if qsl is not None:
            break
    if qsl is None:
        return False
    starts = qsl[:-1].long()
    starts = starts[starts < positions.shape[0]]
    return bool(starts.numel() > 0 and int(positions[starts].max().item()) == 0)


def _dsv41_sp_bounds(num_tokens: int) -> tuple[int, int, int]:
    """(chunk, start, valid) of this rank's row shard, matching sp_shard()."""
    tp_size = get_tensor_model_parallel_world_size()
    chunk = cdiv(num_tokens, tp_size)
    start = min(get_tensor_model_parallel_rank() * chunk, num_tokens)
    return chunk, start, max(min(chunk, num_tokens - start), 0)


def _dsv41_sp_gather(t: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if _DSV41_SP_MODE == "comm" or t.shape[0] == num_tokens:
        return t[:num_tokens]
    return sp_all_gather(t.contiguous())[:num_tokens]


def _dsv41_sp_pad_to_chunk(t: torch.Tensor, chunk: int) -> torch.Tensor:
    if t.shape[0] >= chunk:
        return t[:chunk]
    pad = (0, 0) * (t.ndim - 1) + (0, chunk - t.shape[0])
    return torch.nn.functional.pad(t, pad)


def _dsv41_sp_self_check_op(
    op_kind: str,
    num_tokens: int,
    shard_fn: Callable[[], tuple[torch.Tensor, ...]],
    full_fn: Callable[[], tuple[torch.Tensor, ...]],
    device: torch.device,
    layer_id: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Per-(layer_id, op_kind, M) bitwise self-check of a sharded per-row op
    against the stock full-M op on gathered inputs, agreed across TP ranks with
    MIN and falling back to full-M if any kernel/tactic ever diverges."""
    if not _DSV41_PREFILL_SP_SELF_CHECK or not _sp_ctx.active or _sp_ctx.inner > 0:
        return shard_fn()
    lid = _sp_ctx.layer_id if layer_id is None else int(layer_id)
    key = (lid, op_kind, -1 if _DSV41_PREFILL_SP_SELF_CHECK_ONCE else int(num_tokens))
    ok = _SP_SELF_CHECK_OK.get(key)
    chunk, start, valid = _dsv41_sp_bounds(num_tokens)
    if ok is None and not _g4p_real_forward():
        return shard_fn()
    if ok is None:
        ours = shard_fn()
        with _sp_inner():
            refs_full = full_fn()
        refs_slice = tuple(
            _dsv41_sp_pad_to_chunk(r[start : start + valid].contiguous(), chunk)
            for r in refs_full
        )
        local_ok = all(
            a[:valid].shape == b[:valid].shape
            and a.dtype == b.dtype
            and _g4p_bits_equal(a[:valid], b[:valid])
            for a, b in zip(ours, refs_slice)
        )
        ok = _SP_SELF_CHECK_OK[key] = _dsv41_sp_agree_min(local_ok, device)
        if get_tensor_model_parallel_rank() == 0 and (
            len(_SP_SELF_CHECK_OK) <= 16 or not ok
        ):
            logger.warning(
                "DSV41_PREFILL_SP self-check (L=%d, %s, M=%d, shard=%d): %s "
                "(rank0_local=%s)",
                lid,
                op_kind,
                num_tokens,
                chunk,
                "BIT-EXACT on all ranks -> SHARD ON"
                if ok
                else "DIVERGED -> FULL-M FALLBACK",
                local_ok,
            )
        return ours if ok else refs_slice
    if ok:
        return shard_fn()
    with _sp_inner():
        refs_full = full_fn()
    return tuple(
        _dsv41_sp_pad_to_chunk(r[start : start + valid].contiguous(), chunk)
        for r in refs_full
    )


_G4P_SP_DBG = {"replays": 0, "no_replay": 0}


def _dsv41_sp_attn(attn, positions: torch.Tensor, x_shard: torch.Tensor) -> torch.Tensor:
    """Attention on all rows, wo_b without its all-reduce, then reduce-scatter."""
    num_tokens = positions.shape[0]
    if _DSV41_SP_MODE == "comm":
        x = x_shard
    else:
        x = sp_all_gather(x_shard)[:num_tokens]
    wo_b = attn.wo_b
    prev = wo_b.reduce_results
    wo_b.reduce_results = False
    dbg_topk = _sp_ctx.debug_topk
    layer_id = _sp_ctx.layer_id
    saved_indexer_op = None
    if (
        dbg_topk is not None
        and _DSV41_PREFILL_SP_DEBUG_REPLAY_TOPK
        and layer_id in dbg_topk
        and getattr(attn, "indexer", None) is not None
    ):
        ref_x, ref_topk, ref_cand = dbg_topk[layer_id]
        if _g4p_bits_equal(x, ref_x):
            indexer = attn.indexer
            saved_indexer_op = indexer.indexer_op
            _ref_topk_n = int(ref_topk.shape[0]) if ref_topk.numel() else 0

            def _replay_indexer_op(*_a, **_kw):
                if indexer.topk_indices_buffer is not None and _ref_topk_n:
                    indexer.topk_indices_buffer[:_ref_topk_n].copy_(ref_topk)
                if ref_cand is not None and attn.candidate_block_buffer is not None:
                    attn.candidate_block_buffer[: ref_cand.shape[0]].copy_(ref_cand)
                return indexer.topk_indices_buffer

            saved_indexer_op.forward = _replay_indexer_op
            _G4P_SP_DBG["replays"] += 1
        else:
            _G4P_SP_DBG["no_replay"] += 1
    try:
        with _sp_inner():
            x = attn(positions, x, None)
    finally:
        wo_b.reduce_results = prev
        if saved_indexer_op is not None:
            del saved_indexer_op.forward
    if _DSV41_SP_MODE == "comm":
        # Transport only: every row stays on every rank.
        return _dsv41_sp_gather(sp_reduce_scatter(x), num_tokens)
    return _dsv41_sp_reduce(x)


def _dsv41_sp_ffn(
    ffn,
    x_shard: torch.Tensor,
    num_tokens: int,
    input_ids: torch.Tensor | None,
    mega_gate_metadata,
) -> torch.Tensor:
    """MoE on all rows, final all-reduce skipped, then reduce-scatter."""
    if _DSV41_SP_MODE == "comm":
        x = x_shard
    else:
        x = sp_all_gather(x_shard)[:num_tokens]
    runner = getattr(ffn, "experts", None)
    cfg = getattr(runner, "moe_config", None)
    if (
        cfg is None
        or getattr(ffn, "use_native_mega_moe", False)
        or getattr(runner, "_fused_output_is_reduced", False)
    ):
        # The kernel reduces by itself: keep it and take this rank's rows.
        with _sp_inner():
            out = ffn(x, input_ids, mega_gate_metadata)
        if _DSV41_SP_MODE == "comm":
            return out
        return sp_shard(out)
    prev = cfg.skip_final_all_reduce
    cfg.skip_final_all_reduce = True
    try:
        with _sp_inner():
            x = ffn(x, input_ids, mega_gate_metadata)
    finally:
        cfg.skip_final_all_reduce = prev
    if _DSV41_SP_MODE == "comm":
        return _dsv41_sp_gather(sp_reduce_scatter(x), num_tokens)
    return _dsv41_sp_reduce(x)


def _dsv41_sp_engram(
    engram,
    hidden_shard: torch.Tensor,
    hash_ids: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Engram.forward for a row shard of the stream (TP-sharded wkv kept).

    The wkv projection still runs on all rows (column-parallel + gather, as in
    the stock path); only the gate/injection kernel runs on this rank's rows.
    """
    import triton

    from ..common.engram import _fused_engram_post_wkv_kernel

    num_kv_all = hash_ids.shape[0]
    chunk, start, num_kv_tokens = _dsv41_sp_bounds(num_kv_all)
    if engram._batch is not None:
        with _sp_inner():
            kv = engram.wkv(engram._batch.wkv_input(engram, num_kv_all))
        kv = kv[start:]
    elif _DSV41_PREFILL_SP_WKV_SHARD:
        emb_full = engram.embed(hash_ids).flatten(-2)
        emb_shard = _dsv41_sp_pad_to_chunk(
            emb_full[start : start + num_kv_tokens].contiguous(), chunk
        )
        (kv,) = _dsv41_sp_self_check_op(
            "engram_wkv",
            num_kv_all,
            lambda: (engram.wkv(emb_shard),),
            lambda: (engram.wkv(emb_full),),
            hidden_shard.device,
            layer_id=engram.layer_hash_index,
        )
    else:
        with _sp_inner():
            kv = engram.wkv(engram.embed(hash_ids).flatten(-2))
        kv = kv[start:]
    num_tokens, hc_mult, dim = hidden_shard.shape
    assert num_tokens == chunk, (num_tokens, chunk)
    assert hc_mult == engram.hc_mult and dim == engram.dim
    output = torch.empty_like(hidden_shard)
    if num_tokens == 0:
        return output
    if token_mask is not None:
        token_mask = token_mask[start : start + num_kv_tokens]
    block_size = triton.next_power_of_2(dim)
    num_warps = 8 if block_size >= 2048 else 4
    mask = token_mask if token_mask is not None else hidden_shard
    _fused_engram_post_wkv_kernel[(num_tokens * hc_mult,)](
        hidden_shard,
        kv,
        engram.q_weight,
        engram.k_weight,
        mask,
        output,
        num_kv_tokens,
        hidden_shard.stride(0),
        hidden_shard.stride(1),
        hidden_shard.stride(2),
        kv.stride(0),
        kv.stride(1),
        engram.q_weight.stride(0),
        engram.q_weight.stride(1),
        engram.k_weight.stride(0),
        engram.k_weight.stride(1),
        token_mask.stride(0) if token_mask is not None else 0,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        engram.eps,
        engram.clamp_value,
        DIM=dim,
        HC_MULT=hc_mult,
        BLOCK_SIZE=block_size,
        HAS_MASK=token_mask is not None,
        num_warps=num_warps,
    )
    return output


class DeepseekV4MoE(DeepseekV4MoEBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sequence_parallel: bool = False,
        reduce_results: bool = True,
    ):
        config = vllm_config.model_config.hf_config
        n_routed_experts = config.n_routed_experts
        n_activated_experts = config.num_experts_per_tok
        if extract_layer_index(prefix) >= config.num_hidden_layers:
            n_routed_experts = (
                getattr(config, "dspark_n_routed_experts", 0) or n_routed_experts
            )
            n_activated_experts = (
                getattr(config, "dspark_num_experts_per_tok", 0) or n_activated_experts
            )
        super().__init__(
            vllm_config,
            prefix=prefix,
            use_sequence_parallel=use_sequence_parallel,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            num_hash_layers=0,
            reduce_results=reduce_results,
            image_sentinel_lo=IMAGE_SENTINEL_BASE_ID,
        )

    def defer_finalize(self) -> None:
        """Leave the routed top-k reduction to the next fused all-reduce + mHC.

        Only the modular TRTLLM MXFP4 experts can stop after GEMM2; the fused
        kernel takes as many tokens as the mHC overlap path.
        """
        moe_config = self.experts.moe_config
        experts_cls = getattr(
            self.experts.routed_experts.quant_method, "experts_cls", None
        )
        if (
            experts_cls is TrtLlmMxfp4ExpertsModular
            and self.experts.routed_scaling_factor == 1.0
        ):
            moe_config.defer_moe_finalize(MHC_OVERLAP_MAX_TOKENS)
            if moe_config.use_deferred_moe_finalize:
                logger.info_once(
                    "DSV4.1 mHC: MoE top-k finalize fused into the all-reduce for "
                    "up to %d tokens.",
                    MHC_OVERLAP_MAX_TOKENS,
                )

    def defers_finalize(self, num_tokens: int) -> bool:
        return (
            not self.use_native_mega_moe
            and self.experts.moe_config.should_defer_moe_finalize(num_tokens)
        )

    def forward_unfinalized(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> MoEOutput:
        """``forward`` with the routed top-k reduction and all-reduce left open."""
        # The runner's custom op returns tensors only, so run its body directly.
        shared_output, routed = _unpack(
            self.experts._forward_impl(
                hidden_states, hidden_states, hidden_states, input_ids
            )
        )
        assert shared_output is not None
        assert isinstance(routed, UnfinalizedMoEOutput)
        return MoEOutput(routed=routed, shared_output=shared_output)


def _select_dsv4_attn_cls(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
    """Pick the CUDA sparse-MLA attention class for the configured backend.

    The generic CUDA backend selector does not instantiate DSv4 layers directly,
    so map generic sparse-MLA choices to the DSv4-specialized attention class.
    Without an explicit backend: SM12 takes FlashInfer, SM100 takes mega
    attention where the topology allows it, and everything else keeps the
    FlashMLA path.
    """
    backend = vllm_config.attention_config.backend
    device_capability = current_platform.get_device_capability()
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
    ):
        raise ValueError(
            f"{backend.name} is not a DeepSeek V4.1 attention backend. "
            "Use FLASHINFER_MLA_SPARSE_DSV41 for DeepSeek V4.1 FlashInfer "
            "sparse MLA."
        )
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41,
    ):
        if device_capability is not None and device_capability.major == 12:
            return DeepseekV4FlashInferSM120Attention
        return DeepseekV4FlashInferMLAAttention
    if backend is AttentionBackendEnum.FLASHMLA_MEGA_ATTN_DSV41:
        return DeepseekV4MegaAttnAttention
    if backend in (
        AttentionBackendEnum.FLASHMLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV41,
    ):
        return DeepseekV4FlashMLAAttention

    if device_capability is not None and device_capability.major == 12:
        return DeepseekV4FlashInferSM120Attention
    # Mega attention is the SM100 default: it fuses Q RoPE, sparse attention,
    # the output's inverse RoPE and its FP8 cast into one launch, and brings
    # the 288 B NVFP4 compressed record -- the format the reference
    # implementation itself stores. It declines topologies it cannot serve
    # (non-SM100, TP that leaves fewer than WV_GROUP_SIZE heads per wo_a
    # group, a build without the kernel), which then fall through to FlashMLA.
    if DeepseekV4MegaAttnAttention.is_available_for(vllm_config):
        return DeepseekV4MegaAttnAttention
    return DeepseekV4FlashMLAAttention


def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:
    parallel_config = vllm_config.parallel_config
    moe_needs_token_sharded_input = (
        vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS
    )
    return (
        parallel_config.pipeline_parallel_size == 1
        and parallel_config.enable_expert_parallel
        and parallel_config.tensor_parallel_size > 1
        and (moe_needs_token_sharded_input or parallel_config.data_parallel_size > 1)
    )


def maybe_init_gemm_rs(vllm_config: VllmConfig, use_sequence_parallel: bool) -> bool:
    """Set up the fused ``wo_b`` GEMM + reduce-scatter when opted in.

    Sequence parallel is the only topology where ``wo_b`` ends in a
    reduce-scatter, so the kernel is bound to RS mode. The workspace is a
    process-wide singleton shared with any other model code in this worker
    (the DSpark drafter reuses it), and its NVLink multicast rendezvous is
    collective, so every TP rank must take the same decision here.
    """
    if not (use_sequence_parallel and envs.VLLM_ENABLE_GEMM_RS):
        return False

    # The kernel module pulls in cute_dsl, so import it only once opted in.
    from vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar import (
        maybe_init_gemm_rs_ar,
    )

    hidden_size = vllm_config.model_config.hf_config.hidden_size
    if not maybe_init_gemm_rs_ar(vllm_config, N=hidden_size, all_reduce=False):
        return False
    logger.info_once("To disable DeepSeek-V4.1 GEMM-RS, set VLLM_ENABLE_GEMM_RS=0.")
    return True


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
        engram_layout: EngramLayout | None = None,
        engram_prefetch_stream: torch.cuda.Stream | None = None,
        run_gemm_rs: bool = False,
        mhc_stream: torch.cuda.Stream | None = None,
        fuse_mhc_all_reduce: bool = False,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.mhc_stream = mhc_stream
        self.fuse_mhc_all_reduce = fuse_mhc_all_reduce
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)

        self.engram: Engram | None = None
        if engram_layout is not None:
            layer_id = extract_layer_index(prefix)
            if layer_id in engram_layout.layer_ids:
                self.engram = Engram(
                    config,
                    vllm_config.quant_config,
                    engram_layout,
                    engram_layout.layer_ids.index(layer_id),
                    use_sequence_parallel=self.use_sequence_parallel,
                    prefix=f"{prefix}.engram",
                    prefetch_stream=engram_prefetch_stream,
                )

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = _select_dsv4_attn_cls(vllm_config)(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
            candidate_block_buffer=candidate_block_buffer,
        )
        if self.use_sequence_parallel or fuse_mhc_all_reduce:
            self.attn.wo_b.reduce_results = False
            if run_gemm_rs:
                # Binds only when wo_b's kernel and shape qualify; otherwise
                # forward keeps the separate reduce-scatter below.
                self.attn.bind_gemm_rs()
        self.ffn = DeepseekV4MoE(
            vllm_config,
            prefix=f"{prefix}.ffn",
            use_sequence_parallel=self.use_sequence_parallel,
            reduce_results=not fuse_mhc_all_reduce,
        )

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        if vllm_config.kernel_config.enable_jit_warmup and current_platform.is_cuda():
            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                mhc_fused_post_pre_splits,
            )
            from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL

            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            if self.use_sequence_parallel:
                tp_size = vllm_config.parallel_config.tensor_parallel_size
                max_tokens = (max_tokens + tp_size - 1) // tp_size
            # The epilogue compiles per projection width and per pre-mix mode.
            # The first layer projects the broadcast embedding, so it reads one
            # hidden_size-wide row and selects stream zero; every later sublayer
            # projects the full hc stream, and collapses it with the pre-mix the
            # previous sublayer carried in. Those later shapes also arrive
            # through the fused post + pre-norm GEMM, which picks split-k
            # factors the token sweep below never produces.
            broadcast_embedding = {
                "rms_numel": self.hidden_size,
                "use_pre_mix_in": False,
                "extra_splits": (),
            }
            hc_stream = {
                "rms_numel": self.hc_mult * self.hidden_size,
                "extra_splits": mhc_fused_post_pre_splits(
                    self.hidden_size, self.hc_mult
                ),
            }
            variants = [
                broadcast_embedding,
                {**hc_stream, "use_pre_mix_in": False},
                {**hc_stream, "use_pre_mix_in": True},
            ]
            if vllm_config.speculative_config is not None:
                # Draft setups read aux hidden states out of the same collapse,
                # which is its own specialization of the kernel.
                variants.append(
                    {**hc_stream, "use_pre_mix_in": True, "write_aux": True}
                )
            # Shifted mHC splits the epilogue in two: the input collapse on the
            # caller stream and coefficient generation on a side stream.
            split_modes = (
                ("fused", "stats", "input") if mhc_stream is not None else ("fused",)
            )
            for variant in variants:
                for split_mode in split_modes:
                    MHC_PRE_NORM_KERNEL.register_warmup(
                        split_mode=split_mode,
                        max_tokens=(
                            max_tokens
                            if split_mode == "fused"
                            else min(max_tokens, _dsv41_mhc_max_tokens())
                        ),
                        hidden_size=self.hidden_size,
                        rms_eps=self.rms_norm_eps,
                        hc_pre_eps=self.hc_eps,
                        hc_sinkhorn_eps=self.hc_eps,
                        hc_post_mult_value=self.hc_post_alpha,
                        sinkhorn_repeat=self.hc_sinkhorn_iters,
                        norm_eps=self.rms_norm_eps,
                        hc_mult=self.hc_mult,
                        **variant,
                    )
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        x: torch.Tensor | MoEOutput,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        *,
        capture_previous_aux: bool = False,
        mega_gate_metadata: MegaGateRoutingMetadata | None = None,
        dsv41_sp: bool = False,
    ) -> tuple[
        torch.Tensor | MoEOutput,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        previous_aux: torch.Tensor | None = None
        mhc_stream = self.mhc_stream
        if mhc_stream is not None and (
            in_piecewise_cudagraph()
            or not 0 < positions.shape[0] <= _dsv41_mhc_max_tokens()
            or not torch.cuda.is_current_stream_capturing()
        ):
            # Use overlap only in FULL graphs; eager warmup initializes Mega mHC.
            mhc_stream = None
        if mhc_stream is not None and _DSV41_MHC_OVL is not None:
            # DSV41_MHC_OVERLAP: the first-layer / Engram mhc_pre stays the served kernel
            # pair (an empty fork keeps this layer's join legal) unless DSV41_MHC_OVERLAP_PRE=1,
            # which takes vLLM's own mhc_pre_delayed_overlap (same GEMM, split epilogue).
            mhc_pre = (
                partial(
                    _DSV41_MHC_OVL.overlap_pre, mhc_pre_delayed_overlap, stream=mhc_stream
                )
                if _DSV41_MHC_OVL.OVERLAP_PRE
                else partial(
                    _DSV41_MHC_OVL.served_pre, mhc_pre_delayed_tilelang, stream=mhc_stream
                )
            )
        else:
            mhc_pre = (
                partial(mhc_pre_delayed_overlap, stream=mhc_stream)
                if mhc_stream is not None
                else mhc_pre_delayed_tilelang
            )
        # The reference collapses each sublayer's input with the *previous*
        # sublayer's pre-mix: attention uses the pre-mix carried in (identity
        # for the first layer), the FFN uses this layer's attention pre-mix.
        num_tokens = positions.shape[0]
        sp_check = (
            dsv41_sp
            and _DSV41_SP_MODE != "comm"
            and _DSV41_PREFILL_SP_SELF_CHECK
            and mhc_stream is None
            and not self.fuse_mhc_all_reduce
        )
        if residual is None:
            assert isinstance(x, torch.Tensor)
            if x.dim() == 2:
                # First layer: the stream is the embedding broadcast to hc
                # copies and the identity pre-mix selects copy 0.
                assert self.hc_attn_fn_broadcast is not None
                x_in = x
                residual = x_in.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
                if sp_check:
                    post_mix, res_mix, x, attn_pre = _dsv41_sp_self_check_op(
                        "mhc_pre_bcast",
                        num_tokens,
                        lambda: mhc_pre(
                            residual,
                            self.hc_attn_fn_broadcast,
                            self.hc_attn_scale,
                            self.hc_attn_base,
                            self.rms_norm_eps,
                            self.hc_eps,
                            self.hc_eps,
                            self.hc_post_alpha,
                            self.hc_sinkhorn_iters,
                            x=x_in,
                            norm_weight=self.attn_norm.weight,
                            norm_eps=self.attn_norm.variance_epsilon,
                        ),
                        lambda: (
                            lambda x_full: mhc_pre(
                                x_full.unsqueeze(1)
                                .expand(-1, self.hc_mult, -1)
                                .contiguous(),
                                self.hc_attn_fn_broadcast,
                                self.hc_attn_scale,
                                self.hc_attn_base,
                                self.rms_norm_eps,
                                self.hc_eps,
                                self.hc_eps,
                                self.hc_post_alpha,
                                self.hc_sinkhorn_iters,
                                x=x_full,
                                norm_weight=self.attn_norm.weight,
                                norm_eps=self.attn_norm.variance_epsilon,
                            )
                        )(_dsv41_sp_gather(x_in, num_tokens)),
                        x_in.device,
                    )
                else:
                    post_mix, res_mix, x, attn_pre = mhc_pre(
                        residual,
                        self.hc_attn_fn_broadcast,
                        self.hc_attn_scale,
                        self.hc_attn_base,
                        self.rms_norm_eps,
                        self.hc_eps,
                        self.hc_eps,
                        self.hc_post_alpha,
                        self.hc_sinkhorn_iters,
                        x=x_in,
                        norm_weight=self.attn_norm.weight,
                        norm_eps=self.attn_norm.variance_epsilon,
                    )
            else:
                residual = x
                post_mix, res_mix, x, attn_pre = mhc_pre(
                    residual,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    pre_mix=pre_mix,
                    norm_weight=self.attn_norm.weight,
                    norm_eps=self.attn_norm.variance_epsilon,
                )
        elif self.engram is not None and engram_hashes is not None:
            # Engram injection happens between the previous sublayer's post
            # and this block's pre, on the full hc stream, so the mix
            # coefficients see the injected stream. The injection also keeps
            # the post out of the pre-norm GEMM's fused prologue. The layer
            # before an engram layer finalizes its own MoE.
            assert isinstance(x, torch.Tensor)
            if self.fuse_mhc_all_reduce:
                x = tensor_model_parallel_all_reduce(x)
            previous_post = mhc_post_tilelang(x, residual, post_mix, res_mix)
            if capture_previous_aux:
                previous_aux = previous_post.mean(dim=1)
            if dsv41_sp and _DSV41_SP_MODE != "comm":
                residual = _dsv41_sp_engram(
                    self.engram,
                    previous_post,
                    engram_hashes[:, self.engram.layer_hash_index],
                    engram_mask,
                )
            else:
                residual = self.engram(
                    previous_post,
                    engram_hashes[:, self.engram.layer_hash_index],
                    engram_mask,
                )
            if sp_check and pre_mix is not None:
                res_in, pre_in = residual, pre_mix
                post_mix, res_mix, x, attn_pre = _dsv41_sp_self_check_op(
                    "mhc_pre_engram",
                    num_tokens,
                    lambda: mhc_pre(
                        res_in,
                        self.hc_attn_fn,
                        self.hc_attn_scale,
                        self.hc_attn_base,
                        self.rms_norm_eps,
                        self.hc_eps,
                        self.hc_eps,
                        self.hc_post_alpha,
                        self.hc_sinkhorn_iters,
                        pre_mix=pre_in,
                        norm_weight=self.attn_norm.weight,
                        norm_eps=self.attn_norm.variance_epsilon,
                    ),
                    lambda: mhc_pre(
                        _dsv41_sp_gather(res_in, num_tokens),
                        self.hc_attn_fn,
                        self.hc_attn_scale,
                        self.hc_attn_base,
                        self.rms_norm_eps,
                        self.hc_eps,
                        self.hc_eps,
                        self.hc_post_alpha,
                        self.hc_sinkhorn_iters,
                        pre_mix=_dsv41_sp_gather(pre_in, num_tokens),
                        norm_weight=self.attn_norm.weight,
                        norm_eps=self.attn_norm.variance_epsilon,
                    ),
                    res_in.device,
                )
            else:
                post_mix, res_mix, x, attn_pre = mhc_pre(
                    residual,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    pre_mix=pre_mix,
                    norm_weight=self.attn_norm.weight,
                    norm_eps=self.attn_norm.variance_epsilon,
                )
        else:
            # The collapse already reads the post-mapped streams, so the mean
            # aux consumers want comes out of the same kernel.
            if (
                sp_check
                and isinstance(x, torch.Tensor)
                and pre_mix is not None
                and post_mix is not None
                and res_mix is not None
                and not capture_previous_aux
            ):
                x_in, res_in, pm_in, rm_in, pr_in = (
                    x,
                    residual,
                    post_mix,
                    res_mix,
                    pre_mix,
                )
                residual, post_mix, res_mix, x, attn_pre = _dsv41_sp_self_check_op(
                    "mhc_shifted_attn",
                    num_tokens,
                    lambda: mhc_shifted_post_pre(
                        x_in,
                        res_in,
                        pm_in,
                        rm_in,
                        self.hc_attn_fn,
                        self.hc_attn_scale,
                        self.hc_attn_base,
                        self.rms_norm_eps,
                        self.hc_eps,
                        self.hc_eps,
                        self.hc_post_alpha,
                        self.hc_sinkhorn_iters,
                        pre_mix=pr_in,
                        norm_weight=self.attn_norm.weight,
                        norm_eps=self.attn_norm.variance_epsilon,
                        capture_aux=False,
                        stream=None,
                        reduce_results=False,
                    )[:5],
                    lambda: mhc_shifted_post_pre(
                        _dsv41_sp_gather(x_in, num_tokens),
                        _dsv41_sp_gather(res_in, num_tokens),
                        _dsv41_sp_gather(pm_in, num_tokens),
                        _dsv41_sp_gather(rm_in, num_tokens),
                        self.hc_attn_fn,
                        self.hc_attn_scale,
                        self.hc_attn_base,
                        self.rms_norm_eps,
                        self.hc_eps,
                        self.hc_eps,
                        self.hc_post_alpha,
                        self.hc_sinkhorn_iters,
                        pre_mix=_dsv41_sp_gather(pr_in, num_tokens),
                        norm_weight=self.attn_norm.weight,
                        norm_eps=self.attn_norm.variance_epsilon,
                        capture_aux=False,
                        stream=None,
                        reduce_results=False,
                    )[:5],
                    x_in.device,
                )
            else:
                residual, post_mix, res_mix, x, attn_pre, aux = mhc_shifted_post_pre(
                    x,
                    residual,
                    post_mix,
                    res_mix,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    pre_mix=pre_mix,
                    norm_weight=self.attn_norm.weight,
                    norm_eps=self.attn_norm.variance_epsilon,
                    capture_aux=capture_previous_aux,
                    stream=mhc_stream,
                    reduce_results=self.fuse_mhc_all_reduce,
                )
                if capture_previous_aux:
                    previous_aux = aux

        if dsv41_sp:
            # r4 prefill-sp: x is this rank's rows; returns this rank's rows.
            x = _dsv41_sp_attn(self.attn, positions, x)
        else:
            if self.use_sequence_parallel:
                x = sp_all_gather(x)[: positions.shape[0]]

            x = self.attn(positions, x, None)
        # With GEMM-RS bound, the attention output is already this rank's
        # sequence-parallel shard (see DeepseekV4Attention._wo_b_proj).
        if self.use_sequence_parallel and self.attn.gemm_rs is None:
            x = sp_reduce_scatter(x)

        if mhc_stream is not None:
            if _DSV41_MHC_OVL is not None:
                _DSV41_MHC_OVL.join(mhc_stream)
            else:
                torch.cuda.current_stream().wait_stream(mhc_stream)
        if (
            sp_check
            and isinstance(x, torch.Tensor)
            and residual is not None
            and post_mix is not None
            and res_mix is not None
        ):
            x_in2, res_in2, pm_in2, rm_in2, ap_in2 = (
                x,
                residual,
                post_mix,
                res_mix,
                attn_pre,
            )
            residual, post_mix, res_mix, x, ffn_pre = _dsv41_sp_self_check_op(
                "mhc_shifted_ffn",
                num_tokens,
                lambda: mhc_shifted_post_pre(
                    x_in2,
                    res_in2,
                    pm_in2,
                    rm_in2,
                    self.hc_ffn_fn,
                    self.hc_ffn_scale,
                    self.hc_ffn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    pre_mix=ap_in2,
                    norm_weight=self.ffn_norm.weight,
                    norm_eps=self.ffn_norm.variance_epsilon,
                    stream=None,
                    reduce_results=False,
                )[:5],
                lambda: mhc_shifted_post_pre(
                    _dsv41_sp_gather(x_in2, num_tokens),
                    _dsv41_sp_gather(res_in2, num_tokens),
                    _dsv41_sp_gather(pm_in2, num_tokens),
                    _dsv41_sp_gather(rm_in2, num_tokens),
                    self.hc_ffn_fn,
                    self.hc_ffn_scale,
                    self.hc_ffn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    pre_mix=_dsv41_sp_gather(ap_in2, num_tokens),
                    norm_weight=self.ffn_norm.weight,
                    norm_eps=self.ffn_norm.variance_epsilon,
                    stream=None,
                    reduce_results=False,
                )[:5],
                x_in2.device,
            )
        else:
            residual, post_mix, res_mix, x, ffn_pre, _ = mhc_shifted_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                pre_mix=attn_pre,
                norm_weight=self.ffn_norm.weight,
                norm_eps=self.ffn_norm.variance_epsilon,
                stream=mhc_stream,
                reduce_results=self.fuse_mhc_all_reduce,
            )
        if dsv41_sp:
            x = _dsv41_sp_ffn(
                self.ffn, x, positions.shape[0], input_ids, mega_gate_metadata
            )
        elif self.fuse_mhc_all_reduce and self.ffn.defers_finalize(x.shape[0]):
            # The next layer's first boundary finalizes it with the all-reduce.
            x = self.ffn.forward_unfinalized(x, input_ids)
        else:
            x = self.ffn(x, input_ids, mega_gate_metadata)
        if mhc_stream is not None:
            if _DSV41_MHC_OVL is not None:
                _DSV41_MHC_OVL.join(mhc_stream)
            else:
                torch.cuda.current_stream().wait_stream(mhc_stream)
        return x, residual, post_mix, res_mix, ffn_pre, previous_aux

    def write_kv(
        self,
        x: torch.Tensor | MoEOutput,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        """Write the KV ``forward`` would for the replay layers' inputs, skipping
        the rest of the layer; the replay batch then reruns its rows."""
        # The replay batch rules out Engram and sequence parallel here.
        assert self.engram is None and not self.use_sequence_parallel
        *_, x, _, _ = mhc_shifted_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=pre_mix,
            norm_weight=self.attn_norm.weight,
            norm_eps=self.attn_norm.variance_epsilon,
            reduce_results=self.fuse_mhc_all_reduce,
        )
        self.attn.forward_kv(positions, x)


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        self.use_native_mega_moe = (
            vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS
        )
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)
        if (
            self.use_native_mega_moe
            and not vllm_config.parallel_config.enable_expert_parallel
        ):
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a different "
                "moe backend."
            )
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # Three aux streams: one per non-default input GEMM in
        # DeepseekV4Attention._run_parallel_input_projections
        # (compressor kv_score, indexer.weights_proj). fused_wqa_wkv stays on
        # the default stream.
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        # Keep mHC independent of the streams used inside attention.
        mhc_stream = torch.cuda.Stream() if supports_mhc_overlap(vllm_config) else None
        if (
            mhc_stream is None
            and _DSV41_MHC_OVL is not None
            and _DSV41_MHC_OVL.supported(vllm_config)
        ):
            # DSV41_MHC_OVERLAP: supported() refuses every config in which the stock fused mHC
            # all-reduce below would engage, so fuse_mhc_all_reduce stays False on this path.
            mhc_stream = _DSV41_MHC_OVL.make_stream(vllm_config)
        self.fuse_mhc_all_reduce = mhc_stream is not None and supports_mhc_all_reduce(
            vllm_config
        )
        if self.fuse_mhc_all_reduce:
            init_mhc_all_reduce(vllm_config)

        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        # Two-level candidate filtering: the indexer at
        # candidate_source_layer_id publishes the top candidate blocks of
        # compressed positions here; later ratio-1 indexers (24/28/32/36)
        # mask their scores with it.
        candidate_source_layer = getattr(config, "candidate_source_layer_id", -1)
        candidate_topk_blocks = getattr(config, "candidate_topk_blocks", 0)
        if candidate_source_layer >= 0 and candidate_topk_blocks > 0:
            self.candidate_block_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                candidate_topk_blocks,
                dtype=torch.int32,
            )
        else:
            self.candidate_block_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.engram_layout = EngramLayout.from_config(config)
        engram_config = vllm_config.engram_config
        self.vllm_config = vllm_config
        engram_prefetch_stream = (
            torch.cuda.Stream()
            if self.engram_layout is not None
            and engram_config is not None
            and engram_config.cpu_offload
            else None
        )
        if (
            self.engram_layout is not None
            and engram_config is not None
            and engram_config.dp_shared_memory
        ):
            engram_config.dp_shared_memory = can_share_engram_tables(self.engram_layout)

        if self.engram_layout is not None and engram_config and engram_config.use_thp:
            # Release old checkpoint cache before allocating the Engram host tables.
            model_config = vllm_config.model_config
            drop_checkpoint_cache(
                model_config.model_weights or model_config.model,
                revision=model_config.revision,
                cache_dir=vllm_config.load_config.download_dir,
            )

        # GEMM-RS uses NCCL symmetric-memory multicast, which requires all TP
        # ranks to belong to one NVLink domain. Collective: run before layers.
        self.run_gemm_rs = maybe_init_gemm_rs(vllm_config, self.use_sequence_parallel)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
                candidate_block_buffer=self.candidate_block_buffer,
                engram_layout=self.engram_layout,
                engram_prefetch_stream=engram_prefetch_stream,
                run_gemm_rs=self.run_gemm_rs,
                mhc_stream=mhc_stream,
                fuse_mhc_all_reduce=self.fuse_mhc_all_reduce,
            ),
            prefix=f"{prefix}.layers",
        )
        self.engram_batch = EngramBatch.create(
            [
                layer.engram
                for layer in islice(self.layers, self.start_layer, self.end_layer)
                if getattr(layer, "engram", None) is not None
            ],
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        if self.fuse_mhc_all_reduce:
            # A MoE's top-k finalize folds into the next layer's first mHC
            # boundary, which the last local layer lacks and an engram layer
            # replaces with its own all-reduce.
            local_layers = list(islice(self.layers, self.start_layer, self.end_layer))
            for layer, successor in zip(local_layers, local_layers[1:]):
                assert isinstance(layer, DeepseekV4DecoderLayer)
                assert isinstance(successor, DeepseekV4DecoderLayer)
                if successor.engram is None:
                    layer.ffn.defer_finalize()

        # Decoder-side SWA bounded replay: in eager prefill steps the layers past
        # the last KV source run on each request's trailing window only
        # (decoder_replay_layers.py).
        self.decoder_replay_layers: DecoderReplayLayers | None = None
        self.decoder_replay_start = self.end_layer
        cut = max(config.kv_source_layer_ids)
        if (
            cut < self.end_layer - 1
            and self._decoder_replay_supported(vllm_config, cut)
            and self.layers[cut].attn.swa_cache_layer.bounded_replay
        ):
            assert self.layers[cut].attn.compress_ratio <= 1
            # The last KV source writes every row's KV, then replays.
            self.decoder_replay_start = cut
            # The attention metadata keys the replay layers read.
            metadata_prefixes = set()
            for layer in islice(self.layers, cut, self.end_layer):
                attn = typing.cast(DeepseekV4DecoderLayer, layer).attn
                metadata_prefixes.add(attn.swa_cache_layer.prefix)
                if attn.compressed_cache_prefix is not None:
                    metadata_prefixes.add(attn.compressed_cache_prefix)
                if attn.indexer is not None:
                    metadata_prefixes.add(attn.indexer.k_cache.prefix)
            self.decoder_replay_layers = DecoderReplayLayers(
                config.sliding_window,
                self._run_replay_layers,
                self.layers[cut].write_kv,
                metadata_prefixes,
                self.layers[cut].attn.swa_cache_layer.prefix,
            )
            logger.info_once(
                "Decoder SWA bounded replay: in eager prefill steps, layer %d "
                "writes its KV for every token, and the rest of it and layers "
                "%d-%d run on each request's last %d tokens only.",
                cut,
                cut + 1,
                self.end_layer - 1,
                config.sliding_window,
            )

        # The n-gram hash needs a slot-keyed rolling store of compressed ids
        # (chunked prefill / decode lookback); key it off the first local
        # layer's sliding-window KV cache. Only PP ranks owning an engram
        # layer need it.
        self.engram_hash: NgramHashState | None = None
        self.engram_dp_shared_memory = bool(
            vllm_config.engram_config and vllm_config.engram_config.dp_shared_memory
        )
        self.engram_swa_prefix: str | None = None
        if self.engram_layout is not None:
            local_engram = any(
                isinstance(layer, DeepseekV4DecoderLayer) and layer.engram is not None
                for layer in islice(self.layers, self.start_layer, self.end_layer)
            )
            if local_engram:
                first_layer = next(
                    iter(islice(self.layers, self.start_layer, self.end_layer))
                )
                swa_cache_module = first_layer.attn.swa_cache_layer
                self.engram_hash = NgramHashState(
                    vllm_config, self.engram_layout, swa_cache_module
                )
                self.engram_swa_prefix = swa_cache_module.prefix

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if get_pp_group().is_last_rank and needs_mtp_hidden_states:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until the final hc collapse — plus the
        # (num_tokens, hc_mult) pre-mix the next rank's first layer needs
        # for its attention collapse.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "pre_mix": torch.zeros(
                    (batch_size, self.hc_mult),
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_native_mega_moe:
            input_ids = input_ids.to(torch.int64)

        # Engram n-gram hashes for the whole (flattened) batch, computed once
        # on the full token stream — before any sequence-parallel sharding —
        # and consumed by the engram layers (1 and 14) below. Skipped on
        # profile runs (KV cache unbound).
        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if (
            self.engram_hash is not None
            and input_ids is not None
            and is_forward_context_available()
        ):
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, list):
                attn_metadata = attn_metadata[dbo_current_ubatch_id()]
            if isinstance(attn_metadata, dict) and self.engram_hash.ensure_cache():
                assert self.engram_swa_prefix is not None
                swa_metadata = typing.cast(
                    "DeepseekSparseSWAMetadata", attn_metadata[self.engram_swa_prefix]
                )
                # Image-span tokens are dead: they break n-grams (hash op
                # takes True=dead) and their gate is zeroed (Engram.forward
                # takes True=keep).
                image_mask = image_sentinel_mask(input_ids)
                engram_mask = ~image_mask
                if lookback_token_ids is None:
                    if not self.engram_hash.use_slot_cache:
                        raise NotImplementedError(
                            "engram needs `lookback_token_ids` from the model "
                            "runner (the DBO/ubatch wrapper drops model kwargs)"
                        )
                    num_reqs = swa_metadata.num_decodes + swa_metadata.num_prefills
                    lookback_token_ids = input_ids.new_full(
                        (num_reqs, self.engram_hash.lookback_depth), -1
                    )
                engram_hashes = self.engram_hash(
                    input_ids,
                    positions,
                    swa_metadata.query_start_loc,
                    image_mask,
                    lookback_token_ids,
                    image_sentinel_mask(lookback_token_ids),
                    swa_metadata.slot_mapping,
                    swa_metadata.block_table,
                )
            elif not self.engram_dp_shared_memory and get_engram_dp_size() > 1:
                # DP-sharded lookups are collective, so a replica skipping the
                # hash still has to reach them.
                engram_hashes, engram_mask = self.engram_hash.dummy_hashes(input_ids)
            if engram_hashes is not None:
                # Gather all Engram rows before entering the decoder layers.
                # One gather feeds every layer sharing the DP-split table.
                if self.engram_batch is not None:
                    self.engram_batch.prepare_embeddings(engram_hashes)
                else:
                    gathered_hashes = gather_engram_hashes(
                        engram_hashes, dp_shared_memory=self.engram_dp_shared_memory
                    )
                    for layer in islice(self.layers, self.start_layer, self.end_layer):
                        engram = getattr(layer, "engram", None)
                        if engram is not None:
                            engram.prepare_embeddings(
                                gathered_hashes[:, engram.layer_hash_index]
                            )

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        # r4 prefill-sp: decided from the row count only (rank-identical).
        dsv41_sp = self._dsv41_prefill_sp_active(positions, intermediate_tensors)
        dsv41_sp_ref = None
        if dsv41_sp:
            global _DSV41_PREFILL_SP_DEBUG
            if (
                _DSV41_PREFILL_SP_DEBUG > 0
                and engram_hashes is not None
                and _dsv41_sp_clean_forward(positions)
            ):
                # Debug: run the stock (all-reduce) path first on the same
                # inputs; KV/indexer writes are overwritten by the SP pass.
                _DSV41_PREFILL_SP_DEBUG -= 1
                dsv41_sp_ref = self._dsv41_sp_run_debug_ref(
                    hidden_states.clone(),
                    positions,
                    input_ids,
                    engram_hashes,
                    engram_mask,
                )
            if _DSV41_SP_MODE != "comm":
                hidden_states = sp_shard(hidden_states)

        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = None
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
        aux_hidden_by_layer: dict[int, torch.Tensor] = {}
        hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(
            range(self.start_layer, self.decoder_replay_start),
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            aux_hidden_by_layer,
            engram_hashes,
            engram_mask,
            dsv41_sp=dsv41_sp,
        )
        if dsv41_sp:
            # Back to replicated rows for the decoder replay / collapse.
            assert isinstance(hidden_states, torch.Tensor)
            hidden_states = _dsv41_sp_gather(hidden_states, full_num_tokens)
            residual = _dsv41_sp_gather(residual, full_num_tokens)
            post_mix = _dsv41_sp_gather(post_mix, full_num_tokens)
            res_mix = _dsv41_sp_gather(res_mix, full_num_tokens)
            pre_mix = _dsv41_sp_gather(pre_mix, full_num_tokens)
            if dsv41_sp_ref is not None:
                names = ("hidden", "residual", "post_mix", "res_mix", "pre_mix")
                got = (hidden_states, residual, post_mix, res_mix, pre_mix)
                errs = []
                for name, a, b in zip(names, dsv41_sp_ref, got):
                    a = a.reshape(b.shape)
                    if a.dtype == b.dtype and _g4p_bits_equal(a, b):
                        errs.append(f"{name}: bitwise=1")
                        continue
                    af = a.float()
                    d = (af - b.float()).abs()
                    errs.append(
                        f"{name}: max_abs={d.max().item():.3e} "
                        f"rel_l2={(d.norm() / af.norm().clamp_min(1e-30)).item():.3e} bitwise=0"
                    )
                    del af, d
                logger.warning(
                    "DSV41_PREFILL_SP compare rows=%d rank=%d | %s | topk_replay=%d/%d"
                    " | selfcheck ok=%d diverged=%d",
                    full_num_tokens,
                    get_tensor_model_parallel_rank(),
                    " | ".join(errs),
                    _G4P_SP_DBG["replays"],
                    _G4P_SP_DBG["replays"] + _G4P_SP_DBG["no_replay"],
                    sum(1 for v in _SP_SELF_CHECK_OK.values() if v),
                    sum(1 for v in _SP_SELF_CHECK_OK.values() if not v),
                )
                _G4P_SP_DBG["replays"] = _G4P_SP_DBG["no_replay"] = 0
                del dsv41_sp_ref, got, errs
                _sp_ctx.debug_topk = None
                torch.cuda.empty_cache()
        late_aux: list[torch.Tensor] = []
        e7x = (
            self.decoder_replay_layers is not None
            and self.decoder_replay_layers.skip_post
        )
        if e7x:
            # One forward per mark; anything but the eager trimmed path runs stock.
            self.decoder_replay_layers.skip_post = False
            e7x = (
                self.decoder_replay_layers.replay_batch is not None
                and not in_piecewise_cudagraph()
                and get_pp_group().is_last_rank
            )
        if e7x:
            return self._dsv41_e7x_skip_post(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                aux_hidden_by_layer,
                full_num_tokens,
            )
        if self.decoder_replay_layers is not None:
            hidden_states, pre_mix, *late_aux = self.decoder_replay_layers(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
            )
        else:
            hidden_states = self._collapse(
                hidden_states,
                residual,
                post_mix,
                res_mix,
                aux_hidden_by_layer,
                full_num_tokens,
            )

        aux_hidden_states = [
            aux_hidden_by_layer[layer_id]
            for layer_id in self.aux_hidden_state_layers
            if layer_id in aux_hidden_by_layer
        ] + late_aux

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )

        # MTP needs full HC states; otherwise collapse and normalize locally
        # before gathering to reduce communication.
        if self._mtp_hidden_buffer is not None:
            if self.use_sequence_parallel:
                hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
                pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes — the mix the reference applies via
        # ``last_layer.hc_pre(h, pre_mix)`` (v4.1 has no learned hc_head).
        assert pre_mix is not None
        hidden_states = hc_collapse_triton(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if self.use_sequence_parallel and self._mtp_hidden_buffer is None:
            # Without MTP, gather only the collapsed and normalized hidden states.
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _dsv41_e7x_skip_post(
        self,
        hidden_states,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        full_num_tokens: int,
    ):
        """g4p E7X: layer 20's write_kv for every row and the replay layers on the
        replay batch run exactly as in stock (every KV and indexer-K write of the
        step happens); the scatter back, the MTP buffer copy, the collapse and the
        norm are skipped. The returned rows are zeros: this step's logits are
        discarded and the DSpark speculator skips its draft (model_state.py)."""
        assert self.decoder_replay_layers is not None
        self.decoder_replay_layers.run_rows_discard(
            hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual
        )
        out = residual.new_zeros((full_num_tokens, self.config.hidden_size))
        num_late = sum(
            1
            for layer_id in self.aux_hidden_state_layers
            if self.decoder_replay_start <= layer_id <= self.end_layer
        )
        aux_hidden_states = [
            aux_hidden_by_layer[layer_id]
            for layer_id in self.aux_hidden_state_layers
            if layer_id in aux_hidden_by_layer
        ] + [out] * num_late
        if len(aux_hidden_states) > 0:
            return out, aux_hidden_states
        return out

    def _dsv41_prefill_sp_active(
        self,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
    ) -> bool:
        """r4 prefill-sp gate; depends only on rank-identical inputs."""
        return bool(
            _DSV41_PREFILL_SP
            and not self.use_sequence_parallel
            and not self.fuse_mhc_all_reduce
            and not self.use_native_mega_moe
            and intermediate_tensors is None
            and get_pp_group().world_size == 1
            and get_tensor_model_parallel_world_size() > 1
            and positions.shape[0] >= _DSV41_PREFILL_SP_MIN_ROWS
            and self.decoder_replay_start > self.start_layer
            and not torch.cuda.is_current_stream_capturing()
            and not G4P_STEP["sp_off"]
        )

    def _mega_gate_metadata(
        self, input_ids: torch.Tensor | None
    ) -> MegaGateRoutingMetadata | None:
        if not self.use_native_mega_moe:
            return None
        assert input_ids is not None
        return prepare_mega_gate_routing_metadata(
            input_ids,
            has_hash_routing=False,
            image_sentinel_base_id=IMAGE_SENTINEL_BASE_ID
            if getattr(self.config, "vision_n_layers", 0) > 0
            else None,
        )

    def _run_layers(
        self,
        layer_ids: range,
        hidden_states: torch.Tensor | MoEOutput,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        dsv41_sp: bool = False,
    ) -> tuple[
        torch.Tensor | MoEOutput, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # Every layer's post runs inside the next layer's fused pre, so aux
        # hidden states are read back from there instead of recomputed.
        full_num_tokens = positions.shape[0]
        mega_gate_metadata = self._mega_gate_metadata(input_ids)
        sp_end = layer_ids.stop
        if dsv41_sp and _DSV41_PREFILL_SP_LAYERS > 0:
            sp_end = min(sp_end, self.start_layer + _DSV41_PREFILL_SP_LAYERS)
        chunk, _, _ = _dsv41_sp_bounds(full_num_tokens)
        region_ctx = (
            _sp_region(full_num_tokens, chunk)
            if (dsv41_sp and _DSV41_SP_MODE != "comm")
            else contextlib.nullcontext()
        )
        try:
            with region_ctx:
                for idx in layer_ids:
                    if dsv41_sp and idx == sp_end and idx > layer_ids.start:
                        assert isinstance(hidden_states, torch.Tensor)
                        if _DSV41_SP_MODE != "comm":
                            hidden_states = _dsv41_sp_gather(
                                hidden_states, full_num_tokens
                            )
                            residual = _dsv41_sp_gather(residual, full_num_tokens)
                            post_mix = _dsv41_sp_gather(post_mix, full_num_tokens)
                            res_mix = _dsv41_sp_gather(res_mix, full_num_tokens)
                            pre_mix = _dsv41_sp_gather(pre_mix, full_num_tokens)
                        _sp_ctx.active = False
                    layer_sp = dsv41_sp and idx < sp_end
                    _sp_ctx.layer_id = idx
                    (
                        hidden_states,
                        residual,
                        post_mix,
                        res_mix,
                        pre_mix,
                        previous_aux,
                    ) = self.layers[idx](
                        hidden_states,
                        positions,
                        input_ids,
                        pre_mix,
                        post_mix,
                        res_mix,
                        residual,
                        engram_hashes,
                        engram_mask,
                        capture_previous_aux=idx in self.aux_hidden_state_layers,
                        mega_gate_metadata=mega_gate_metadata,
                        dsv41_sp=layer_sp,
                    )
                    if previous_aux is not None:
                        # idx is the one-based id of the layer whose post this is.
                        if self.use_sequence_parallel or (
                            layer_sp and _DSV41_SP_MODE != "comm"
                        ):
                            previous_aux = sp_all_gather(previous_aux)[
                                :full_num_tokens
                            ]
                        aux_hidden_by_layer[idx] = previous_aux
        finally:
            _sp_ctx.layer_id = -1
            if dsv41_sp:
                _sp_ctx.debug_topk = None
        assert residual is not None and post_mix is not None
        assert res_mix is not None and pre_mix is not None
        return hidden_states, residual, post_mix, res_mix, pre_mix

    def _dsv41_sp_run_debug_ref(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        engram_hashes: torch.Tensor | None,
        engram_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | MoEOutput, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Run the stock reference pass for DSV41_PREFILL_SP_DEBUG and snapshot
        each index-source layer's input `x` and `topk_indices_buffer` so that
        cross-call `topKPerRowPrefill` atomic tie-breaking at T > 4,096 cannot
        confound the 20-layer SP comparison when `x` is bit-identical."""
        num_tokens = positions.shape[0]
        debug_topk: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        ] = {}
        hooks = []
        for idx in range(self.start_layer, self.decoder_replay_start):
            attn = self.layers[idx].attn
            if getattr(attn, "indexer", None) is not None:

                def _make_hook(layer_idx: int, attn_mod):
                    orig_attn = attn_mod.forward

                    def _wrapped_attn(pos, x_in, llama_4_scaling=None):
                        x_snap = x_in.clone()
                        out = orig_attn(pos, x_in, llama_4_scaling)
                        topk_snap = (
                            attn_mod.topk_indices_buffer[:num_tokens].clone()
                            if attn_mod.topk_indices_buffer is not None
                            else torch.empty(0, device=x_in.device, dtype=torch.int32)
                        )
                        cand_snap = (
                            attn_mod.candidate_block_buffer[:num_tokens].clone()
                            if attn_mod.candidate_block_buffer is not None
                            else None
                        )
                        debug_topk[layer_idx] = (x_snap, topk_snap, cand_snap)
                        return out

                    return attn_mod, orig_attn, _wrapped_attn

                m, orig_f, wrap_f = _make_hook(idx, attn)
                m.forward = wrap_f
                hooks.append((m, orig_f))
        try:
            ref = self._run_layers(
                range(self.start_layer, self.decoder_replay_start),
                hidden_states,
                positions,
                input_ids,
                None,
                None,
                None,
                None,
                {},
                engram_hashes,
                engram_mask,
                dsv41_sp=False,
            )
        finally:
            for m, orig_f in hooks:
                m.forward = orig_f
        _sp_ctx.debug_topk = debug_topk
        return ref

    def _collapse(
        self,
        hidden_states: torch.Tensor | MoEOutput,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        full_num_tokens: int,
    ) -> torch.Tensor:
        # Without a successor boundary, the last layer finalized its own MoE.
        assert isinstance(hidden_states, torch.Tensor)
        # The last layer has no successor to fold its post into.
        if self.fuse_mhc_all_reduce:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        if self.end_layer in self.aux_hidden_state_layers:
            final_aux = hidden_states.mean(dim=1)
            if self.use_sequence_parallel:
                final_aux = sp_all_gather(final_aux)[:full_num_tokens]
            aux_hidden_by_layer[self.end_layer] = final_aux
        return hidden_states

    def _run_replay_layers(
        self,
        hidden_states: torch.Tensor | MoEOutput,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Layers ``decoder_replay_start``.. on the given rows; returns their output,
        the last FFN's pre-mix and the aux hidden states they capture."""
        aux_hidden_by_layer: dict[int, torch.Tensor] = {}
        hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(
            range(self.decoder_replay_start, self.end_layer),
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            aux_hidden_by_layer,
        )
        hidden_states = self._collapse(
            hidden_states,
            residual,
            post_mix,
            res_mix,
            aux_hidden_by_layer,
            positions.shape[0],
        )
        return (
            hidden_states,
            pre_mix,
            *(
                aux_hidden_by_layer[layer_id]
                for layer_id in self.aux_hidden_state_layers
                if layer_id in aux_hidden_by_layer
            ),
        )

    def _decoder_replay_supported(self, vllm_config: VllmConfig, cut: int) -> bool:
        """Whether this rank may replay from layer ``cut`` on; warns when not."""
        parallel_config = vllm_config.parallel_config
        spec_config = vllm_config.speculative_config
        draft_config = spec_config.draft_model_config if spec_config else None
        draft_hf_config = getattr(draft_config, "hf_config", None)
        draft_window = getattr(draft_hf_config, "sliding_window", None)
        draft_layer_types = getattr(draft_hf_config, "layer_types", None) or ()
        window = self.config.sliding_window
        if self.start_layer > cut or self.end_layer < self.config.num_hidden_layers:
            reason = (
                "the pipeline stage holding the last KV source layer must also "
                "hold every layer after it"
            )
        elif (
            self.use_sequence_parallel
            or parallel_config.prefill_context_parallel_size > 1
            or parallel_config.use_ubatching
        ):
            reason = (
                "the replay-layer batch shrinks per rank, which sequence and "
                "prefill-context parallelism and microbatching cannot follow"
            )
        elif any(i >= cut for i in getattr(self.config, "engram_layer_ids", ())):
            reason = "an Engram layer sits at or after the last KV source layer"
        elif draft_config is not None and (
            draft_window is None
            or draft_window > window
            or any(t != "sliding_attention" for t in draft_layer_types)
        ):
            reason = (
                f"the drafter (sliding window {draft_window}) reads hidden states "
                f"outside the target's {window}-token window"
            )
        else:
            return True
        logger.warning_once("Decoder SWA bounded replay is off: %s.", reason)
        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        # Block-FP8 shared experts: pad the intermediate up to the TP-uniform
        # block count so the standard loaders below slice it evenly (trailing
        # ranks land on the zero pad). SP / unquantized ones need no padding.
        pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not self.use_sequence_parallel
        )

        for name, loaded_weight in weights:
            if name.startswith(("vision.", "aligner.", "image_")):
                # Vision weights are loaded by the outer multimodal wrapper.
                logger.warning_once("Skipping non-text weight: %s", name)
                continue
            if pad_shared_expert and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                if name not in params_dict:
                    head, _, leaf = name.rpartition(".")
                    suffixed = f"{head}.base_layer.{leaf}"
                    if suffixed in params_dict:
                        name = suffixed
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Non-LoRA params on a LoRA-wrapped module live at
                    # ``<head>.base_layer.<leaf>``; the checkpoint is plain.
                    if name not in params_dict:
                        head, _, leaf = name.rpartition(".")
                        suffixed = f"{head}.base_layer.{leaf}"
                        if suffixed in params_dict:
                            name = suffixed
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    @staticmethod
    def _pad_shared_expert_weight(
        quant_config: QuantizationConfig | None,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-pad a block-FP8 shared-expert weight/scale on its intermediate
        axis so the standard TP loaders split it into even, block-aligned shards
        (trailing ranks get the zero pad). gate (w1)/up (w3) [I, H] pad dim 0;
        down (w2 -> down_proj) [H, I] pads dim 1.
        """
        block_size = getattr(quant_config, "weight_block_size", None)
        assert block_size is not None
        # Round the intermediate axis up to a whole number of TP shards. The axis
        # is in elements for weights (step = block) and in blocks for scales.
        step = (
            1 if name.endswith(("weight_scale_inv", "weight_scale")) else block_size[0]
        )
        dim = 1 if ".down_proj." in name else 0
        mult = get_tensor_model_parallel_world_size() * step
        pad = cdiv(loaded_weight.shape[dim], mult) * mult - loaded_weight.shape[dim]
        if pad == 0:
            return loaded_weight
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = pad
        return torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=dim)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        first_layer = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
        if first_layer.ffn.use_native_mega_moe:
            return make_deepseek_v4_expert_params_mapping(self.config.n_routed_experts)
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def finalize_mega_moe_weights(self) -> None:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.ffn.finalize_mega_moe_weights()

    def finalize_mhc_broadcast_weights(self) -> None:
        if not get_pp_group().is_first_rank or self.start_layer >= self.end_layer:
            return
        layer = self.layers[self.start_layer]
        if isinstance(layer, DeepseekV4DecoderLayer):
            broadcast = (
                layer.hc_attn_fn.detach()
                .view(-1, layer.hc_mult, layer.hidden_size)
                .sum(dim=1)
            )
            if layer.hc_attn_fn_broadcast is None:
                layer.hc_attn_fn_broadcast = broadcast
            else:
                layer.hc_attn_fn_broadcast.copy_(broadcast)


def _linear_scale_param_name(vllm_config: VllmConfig, expert_dtype: str) -> str:
    """Parameter name the linear quant method registers for weight scales.

    Native MXFP8 checkpoints ([32, 32] blocks with MXFP4 experts) route linear
    layers through ModelOptLinearMethod, which registers ``weight_scale``;
    block-FP8 linear layers register ``weight_scale_inv``.
    """
    use_mxfp8 = (
        getattr(vllm_config.quant_config, "weight_block_size", None) == [32, 32]
        and expert_dtype == "fp4"
    )
    return "weight_scale" if use_mxfp8 else "weight_scale_inv"


def _make_deepseek_v4_weights_mapper(
    expert_dtype: str, linear_scale_name: str = "weight_scale_inv"
) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). Linear scales use the
        # parameter name registered by their quantization method.
        scale_regex = {
            # ``.base_layer.``-namespace variant (LoRA-wrapped experts).
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale",
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            # The ``embed.weight`` -> ``embed_tokens.weight`` suffix rule
            # renames the engram fp8 table but not its scale; route the
            # scale explicitly to the same module.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            # ``.base_layer.``-namespace variant of the above.
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale_inv",
            # Same engram reroute as the fp4 branch above.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            "mtp.": None,
            # The v4.1 checkpoint declares the VL arch even for text-only
            # use; the text backbone drops the vision-tower weights.
            "vision.": None,
            "aligner.": None,
            "image_": None,
        },
    )


class DeepseekV4MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list["DeepseekV4MoE"]

    def extract_moe_parameters(self, example_moe: "DeepseekV4MoE | None") -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class DeepseekV41LLMForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsEagle3,
    SupportsLoRA,
    DeepseekV4MixtureOfExperts,
):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    # The MTP draft head is not LoRA-adapted.
    lora_skip_prefixes = ["mtp."]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(
            expert_dtype, _linear_scale_param_name(vllm_config, expert_dtype)
        )

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.num_moe_layers = self.config.num_hidden_layers
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe: DeepseekV4MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, DeepseekV4DecoderLayer):
                continue
            if isinstance(layer.ffn, DeepseekV4MoE):
                example_moe = layer.ffn
                self.moe_mlp_layers.append(layer.ffn)
                self.moe_layers.append(layer.ffn.experts)

        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if _DSV41_R4:
            # r4 small-kernels: certified (screened) target head; None -> stock.
            from vllm.models.deepseek_v41.nvidia import dsv41_r4_small as _r4

            out = _r4.target_logits(self, hidden_states, skip_gather=False)
            if out is not None:
                return out
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if _DSV41_R4:
            from vllm.models.deepseek_v41.nvidia import dsv41_r4_small as _r4

            out = _r4.target_logits(self, hidden_states, skip_gather=True)
            if out is not None:
                return out
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        """Tokens before a chunk start the engram hash needs; the model runner
        passes them as `lookback_token_ids`."""
        engram_hash = self.model.engram_hash
        return engram_hash.lookback_depth if engram_hash is not None else 0

    @property
    def decoder_replay_layers(self) -> DecoderReplayLayers | None:
        return self.model.decoder_replay_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-collapse residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.process_weights_after_loading()
        config = self.model.vllm_config
        if config.engram_config and config.engram_config.use_thp:
            # Loading weights refills the file cache; release it before MADV_COLLAPSE.
            model_config = config.model_config
            drop_checkpoint_cache(
                model_config.model_weights or model_config.model,
                revision=model_config.revision,
                cache_dir=config.load_config.download_dir,
            )
            for layer in islice(
                self.model.layers, self.model.start_layer, self.model.end_layer
            ):
                if layer.engram is not None:
                    layer.engram.embed_tokens.collapse_huge_pages()
        return loaded_params

    def process_weights_after_loading(self) -> None:
        self.model.finalize_mega_moe_weights()
        self.model.finalize_mhc_broadcast_weights()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
