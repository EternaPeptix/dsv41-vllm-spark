# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import (
    CUDAGraphMode,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
)
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask as _apply_candidate_mask,
)
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    select_candidate_blocks as _select_candidate_blocks,
)
from vllm.model_executor.layers.indexer_topk import (
    RADIX_TOPK_WORKSPACE_SIZE,
    get_indexer_topk,
)
# g5det (goal window 5, 2026-10-27): boundary-tie canonicalization + debug dump of the indexer
# top-k; inert unless DSV41_TOPK_CANON_TIES or DSV41_DET_DUMP is set (see dsv41_topk_ties.py).
from vllm.model_executor.layers import dsv41_topk_ties as _g5det
# BS03 (2026-11-02): one canonical top-512 kernel replaces top-k + TIES + CANON on decode rows; inert unless
# DSV41_BS03_DET_TOPK=1 (the module is not even imported otherwise; see dsv41_bs03_det_topk.py).
if os.environ.get("DSV41_BS03_DET_TOPK", "0") == "1":
    from vllm.model_executor.layers import dsv41_bs03_det_topk as _bs03
else:
    _bs03 = None
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    tp_prefill_row_sharding_supported,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.pcp import maybe_gather_indexer_k
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


# dsv41 nondet: top-k kernels (persistent_topk fallback top_k_per_row_decode for
# rows > ~398K wide, histogram/radix paths for rows > 512) emit the right index SET
# in a run-dependent ORDER (atomics); sparse MLA sums in stored order, so greedy
# output drifts. DSV41_TOPK_CANON=1 sorts each row ascending with -1 last (equals
# the trivial-branch order for short rows). 0 = stock.
_DSV41_TOPK_CANON = os.environ.get("DSV41_TOPK_CANON", "0") == "1"
# g3 codex-adopt #1 (Codex R5 7fc2d1ab): DSV41_TOPK_CANON_FUSED=1 replaces the 5-launch
# CANON above (where + radix sort + eq + masked_fill + copy) by ONE Triton launch per call
# that sorts each 512-wide row in registers: negatives -> INT32_MAX, ascending sort,
# INT32_MAX -> -1. Same integer multiset per row -> bitwise the stock result. Needs
# DSV41_TOPK_CANON=1; warmed at startup (one compile key); other shapes/layouts/devices
# keep the stock path. 0/unset = stock (no kernel object, no warmup registration).
_DSV41_TOPK_CANON_FUSED = os.environ.get("DSV41_TOPK_CANON_FUSED", "0") == "1"
_I32_MAX = 2147483647


@triton.jit(
    do_not_specialize=["row_stride"],
    do_not_specialize_on_alignment=["t_ptr", "row_stride"],
)
def _dsv41_canon_topk512_kernel(t_ptr, row_stride):
    # Widen before multiplication: legal views can span more than 2**31
    # elements even though each row stride fits the one i32 signature.
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, 512)
    values = tl.load(t_ptr + row * row_stride + col)
    keys = tl.where(values < 0, 2147483647, values)
    ordered = tl.sort(keys, descending=False)
    # Match stock even when an input is positive INT32_MAX.
    result = tl.where(ordered == 2147483647, -1, ordered)
    tl.store(t_ptr + row * row_stride + col, result)


class CanonTopK512Kernel(VllmTritonJitKernel["CanonTopK512Kernel.CompileKey"]):
    """One integer-only, in-place sort per independent 512-element row."""

    @dataclass(frozen=True)
    class CompileKey:
        pass

    kernel = staticmethod(_dsv41_canon_topk512_kernel)

    def dispatch(self, t: torch.Tensor) -> CompileKey:
        return self.CompileKey()

    def get_warmup_keys(self) -> list[CompileKey]:
        return [self.CompileKey()]

    def warmup_inputs(self, compile_key: CompileKey) -> dict:
        return {
            "t": TritonWarmupTensor(
                torch.int32, aligned=False, shape=(1, 512), strides=(512, 1)
            )
        }

    @kernel_launcher
    def __call__(self, t: torch.Tensor) -> LaunchSpec:
        return (t.shape[0],), dict(row_stride=t.stride(0), num_warps=4)


_CANON_TOPK_512_KERNEL = CanonTopK512Kernel() if _DSV41_TOPK_CANON_FUSED else None


def _can_fuse_canon_topk(t: torch.Tensor) -> bool:
    # No host reads of values, pointer-identity cache, or across-forward state.
    # Unsupported types/layouts/platforms retain the stock implementation.
    return (
        _CANON_TOPK_512_KERNEL is not None
        and type(t) is torch.Tensor
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and t.device.type == "cuda"
        and t.device.index == 0
        and t.dtype == torch.int32
        and t.layout == torch.strided
        and not t.is_neg()
        and t.ndim == 2
        and t.shape[1] == 512
        and 0 < t.shape[0] <= _I32_MAX
        and t.stride(1) == 1
        and 512 <= t.stride(0) <= _I32_MAX
    )


_CANON_FUSED_FIRST_CALL = True


def _canon_topk_(t: torch.Tensor) -> None:
    global _CANON_FUSED_FIRST_CALL
    if t.numel() == 0:
        return
    if _CANON_TOPK_512_KERNEL is not None and _can_fuse_canon_topk(t):
        if _CANON_FUSED_FIRST_CALL:
            _CANON_FUSED_FIRST_CALL = False
            logger.info(
                "DSV41_TOPK_CANON_FUSED: fused CANON kernel in use (first call: "
                "rows=%d row_stride=%d)",
                t.shape[0],
                t.stride(0),
            )
        _CANON_TOPK_512_KERNEL(t)
        return
    key = torch.where(t < 0, _I32_MAX, t)
    s = torch.sort(key, dim=-1).values
    t.copy_(s.masked_fill_(s == _I32_MAX, -1))


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


def dcp_gather_kv_rows(
    local_padded: torch.Tensor,
    deinterleave_idx: torch.Tensor,
    gathered_buf: torch.Tensor,
    out_buf: torch.Tensor,
) -> torch.Tensor:
    """All-gather this rank's KV shard and return it in global token order."""
    gathered = gathered_buf[: get_dcp_group().world_size * local_padded.shape[0]]
    dist.all_gather_into_tensor(
        gathered, local_padded.contiguous(), group=get_dcp_group().device_group
    )
    return torch.index_select(
        gathered, 0, deinterleave_idx, out=out_buf[: deinterleave_idx.shape[0]]
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


class PrefillRowShard:
    """One TP rank's contiguous slice of the replicated prefill rows.

    Rows are independent, so each rank scores only its slice and the group
    exchanges final int32 top-k indices instead of the logits."""

    def __init__(self, num_decode_tokens: int, sizes: list[int], rank: int):
        self.num_decode_tokens = num_decode_tokens
        self.sizes = sizes
        self.start = num_decode_tokens + sum(sizes[:rank])
        self.stop = self.start + sizes[rank]

    @classmethod
    def from_metadata(
        cls, prefill_metadata, num_decode_tokens: int
    ) -> "PrefillRowShard | None":
        sizes = getattr(prefill_metadata, "row_shard_sizes", None)
        if sizes is None:
            return None
        return cls(num_decode_tokens, sizes, get_tensor_model_parallel_rank())

    def narrow(self, chunk) -> "tuple[int, int, torch.Tensor, torch.Tensor] | None":
        """Row window and narrowed causal bounds of ``chunk`` for this rank."""
        start = max(chunk.token_start, self.start)
        end = min(chunk.token_end, self.stop)
        if start >= end:
            return None
        lo, hi = start - chunk.token_start, end - chunk.token_start
        return start, end, chunk.cu_seqlen_ks[lo:hi], chunk.cu_seqlen_ke[lo:hi]

    def exchange_topk(self, topk_buffer: torch.Tensor, width: int) -> None:
        """Gather per-rank final indices back into global row order."""
        # The local input must outlive the async PyNCCL gather's enqueued copy.
        local = topk_buffer[self.start : self.stop, :width].contiguous()
        end = self.num_decode_tokens + sum(self.sizes)
        topk_buffer[self.num_decode_tokens : end, :width] = get_tp_group().all_gatherv(
            local, dim=0, sizes=self.sizes
        )


def _reserve_row_shard_exchange_transient(
    topk_indices_buffer: torch.Tensor, device: torch.device
) -> None:
    """Profiling/dummy runs return before ``exchange_topk`` executes; reserve
    its worst-case transient so the memory profile accounts for it.

    The exchange stages this rank's contiguous slice and the all_gatherv
    wrapper allocates the full gathered tensor; each is bounded by the
    buffer's row capacity times its (tile-padded) width.
    """
    cfg = get_current_vllm_config_or_none()
    if cfg is None:
        return
    parallel = cfg.parallel_config
    if not tp_prefill_row_sharding_supported(
        cfg,
        parallel.decode_context_parallel_size,
        parallel.prefill_context_parallel_size > 1,
        get_tp_group().world_size,
    ):
        return
    rows, width = topk_indices_buffer.shape[:2]
    _ = torch.empty(
        2 * rows * width * topk_indices_buffer.element_size(),
        dtype=torch.uint8,
        device=device,
    )


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    pcp_shard_decode_requests: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    topk_backend: str = "auto",
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    if candidate_blocks is not None:
        # Candidate blocks are request-local; the DCP-sharded logits layout
        # would need per-rank translation that is not implemented.
        assert dcp_world_size == 1, (
            "v4.1 two-level candidate filtering is not supported with DCP."
        )
        assert candidate_block_size > 0

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        profile_specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        ]
        if use_pcp and dcp_world_size > 1:
            # The PCP+DCP path takes an all-gather destination and a
            # de-interleaved result.
            gather_spec = _gather_workspace_shapes(
                total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
            )
            profile_specs.extend(gather_spec * 2)
        current_workspace_manager().get_simultaneous(*profile_specs)

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        # The dummy run returns before exchange_topk executes, so reserve its
        # worst-case transient for the memory profile.
        _reserve_row_shard_exchange_transient(topk_indices_buffer, hidden_states.device)

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_pcp,
            pcp_shard_decode_requests,
            dense_mha_metadata_layer_name,
            use_fp4_cache,
            candidate_blocks=candidate_blocks,
            candidate_block_size=candidate_block_size,
            candidate_write=candidate_write,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    # Keep PCP padding so every rank contributes the same all-gather shape.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        # MTP draft decodes are replicated and use an unexpanded slot mapping.
        if use_pcp and num_tokens > k.shape[0]:
            num_tokens //= get_pcp_group().world_size
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
            pcp_shard_decode_requests=pcp_shard_decode_requests,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping_for_cache,
            quant_block_size,
            scale_fmt,
        )
    # The indexer and main MLA may classify the same short extend differently
    # because they use independent decode thresholds. Only the main MLA route
    # can determine whether the top-k indices will be consumed.
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            prefill_metadata = getattr(mla_metadata, "prefill", None)
            if (
                getattr(prefill_metadata, "use_dense_mha", False)
                and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Deliberately leave the buffer untouched. Dense MHA does not
                # consume top-k indices for this batch; clearing it would be
                # unnecessary work.
                return topk_indices_buffer

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        # PCP + DCP needs two more pairs: the rank-major all-gather destination
        # and the de-interleaved result.
        pcp_chunks = [
            (c, c.pcp_deinterleave_idx)
            for c in prefill_metadata.chunks
            if c.pcp_deinterleave_idx is not None
        ]
        gather_specs: list[tuple[tuple[int, int], torch.dtype]] = []
        if pcp_chunks:
            gathered_rows = max(
                dcp_world_size * c.local_total_seq_lens for c, _ in pcp_chunks
            )
            deinterleaved_rows = max(idx.shape[0] for _, idx in pcp_chunks)
            for rows in (gathered_rows, deinterleaved_rows):
                gather_specs.extend(
                    _gather_workspace_shapes(rows, head_dim, fp8_dtype, use_fp4_cache)
                )
        k_quant_full, k_scale_full, *gather_bufs = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
            *gather_specs,
        )
        # One row window per rank; DSV4.1 candidate rows stay replicated.
        row_shard = (
            None
            if candidate_blocks is not None
            else PrefillRowShard.from_metadata(prefill_metadata, num_decode_tokens)
        )
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            # Narrow scoring to this rank; KV gathers remain unconditional for
            # continuation chunks that reuse the workspace.
            row_start, row_end = chunk.token_start, chunk.token_end
            if row_shard is not None:
                narrowed = row_shard.narrow(chunk)
                if narrowed is None:
                    continue
                row_start, row_end, cu_seqlen_ks, cu_seqlen_ke = narrowed

            # PCP + DCP KV all-gather.
            deinterleave_idx = chunk.pcp_deinterleave_idx
            if deinterleave_idx is not None:
                # local_total_seq_lens is the PCP-padded extent here, so every
                # rank contributes an identically shaped shard.
                gathered_values, gathered_scales, out_values, out_scales = gather_bufs
                shard = k_quant[: chunk.local_total_seq_lens]
                k_quant = dcp_gather_kv_rows(
                    shard, deinterleave_idx, gathered_values, out_values
                )
                k_scale = dcp_gather_kv_rows(
                    k_scale[: chunk.local_total_seq_lens],
                    deinterleave_idx,
                    gathered_scales,
                    out_scales,
                )

            q_slice = q_quant[row_start:row_end]
            q_scale_slice = q_scale[row_start:row_end] if q_scale is not None else None
            topk_indices = topk_indices_buffer[row_start:row_end, :topk_tokens]

            g4p_cand = False
            if chunk.local_total_seq_lens == 0 or q_slice.shape[0] == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                # g4p 06: candidate-only scoring of a consumer indexer's prefill chunk.
                g4p_cand = (
                    _G4P_CAND_MODE != "off"
                    and candidate_blocks is not None
                    and not candidate_write
                    and candidate_block_size == _CAND_BLOCK_SIZE
                    and not use_fp4_cache
                    and deinterleave_idx is None
                    and dcp_world_size == 1
                    and q_scale_slice is None
                    and topk_tokens == 512
                    and current_platform.is_cuda()
                    and not torch.cuda.is_current_stream_capturing()
                    and 1 <= q_slice_cast.shape[0] <= _CAND_MAX_PREFILL_ROWS
                    and k_quant_cast.shape[0] >= max(_CAND_CAPACITY, _CAND_MIN_K)
                    and candidate_blocks.shape[1] == _CAND_BLOCKS
                    and not _g4p_cand_step_off()
                )
                if g4p_cand:
                    logits = _g4p_cand_prefill(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[row_start:row_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        candidate_blocks[chunk.token_start : chunk.token_end],
                        topk_indices,
                        topk_tokens,
                    )
                elif current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[row_start:row_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[row_start:row_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                if candidate_blocks is not None and not g4p_cand:
                    # Two-level selection (v4.1): the candidate source
                    # publishes its top blocks; later indexers mask their
                    # scores to them. Both before the row top-k.
                    chunk_candidates = candidate_blocks[
                        chunk.token_start : chunk.token_end
                    ]
                    if candidate_write:
                        _select_candidate_blocks(
                            logits,
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            chunk_candidates.shape[1],
                            candidate_block_size,
                            chunk_candidates,
                        )
                    else:
                        _apply_candidate_mask(
                            logits,
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            chunk_candidates,
                            candidate_block_size,
                        )
                if not g4p_cand:
                    ops.top_k_per_row_prefill(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )
                    if _g5det.ENABLED:
                        _g5det.prefill_hook(
                            logits,
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            topk_indices,
                            q_slice_cast,
                            weights[row_start:row_end],
                            {
                                "hfp": hidden_states[row_start:row_end],
                                "kfp": k_quant_cast[: chunk.local_total_seq_lens],
                                "ksfp": k_scale_cast[: chunk.local_total_seq_lens],
                                "cand": -1
                                if candidate_blocks is None
                                else int(candidate_write),
                                "row_start": row_start,
                                "row_end": row_end,
                            },
                            forward_context,
                        )

            if deinterleave_idx is None:
                # Under the PCP path the top-k already ran over the whole
                # context.
                _merge_dcp_topk_global(
                    logits,
                    topk_indices,
                    topk_tokens,
                    dcp_rank,
                    dcp_world_size,
                    cp_kv_cache_interleave_size,
                    row_starts=cu_seqlen_ks,
                )
            if _DSV41_TOPK_CANON:
                _canon_topk_(topk_indices)

        if row_shard is not None:
            # Rows are already fully ranked, so this is a layout gather.
            row_shard.exchange_topk(topk_indices_buffer, topk_tokens)

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        # requires_padding can be computed False for ragged warmup/mixed
        # batches (seen on SM120 TP=2: 8 tokens over 6 seqs -> uniform
        # reshape below would crash). The token-count/batch-size divisibility
        # check is a necessary-but-not-sufficient proxy for "uniform batch"
        # (e.g. decode_lens=[1, 2, 3] divides evenly by 3 but isn't uniform);
        # it catches the specific case seen in practice without requiring a
        # device sync to inspect decode_lens directly, which would break
        # CUDA-graph-capture safety on this path. Shared by both the pack
        # branch below and the matching unpack branch further down — they
        # must agree on which path was taken.
        needs_padded_path = decode_metadata.requires_padding or (
            num_decode_tokens % decode_lens.shape[0] != 0
        )
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif needs_padded_path:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
                indices=decode_metadata.indices,
            )
        num_rows = logits.shape[0]
        if candidate_blocks is not None:
            # Two-level selection (v4.1) on the decode logits; columns are
            # request-local compressed positions. seq_lens is (B, next_n)
            # for native spec decode (per-row effective lens) and (B, 1)
            # otherwise.
            vis = seq_lens.reshape(-1)
            row_repeat = next_n if vis.numel() != num_rows else 1
            vis = vis[:num_rows]
            decode_candidates = candidate_blocks[:num_rows]
            if candidate_write:
                _select_candidate_blocks(
                    logits,
                    None,
                    vis,
                    decode_candidates.shape[1],
                    candidate_block_size,
                    decode_candidates,
                    row_repeat,
                )
            else:
                _apply_candidate_mask(
                    logits,
                    None,
                    vis,
                    decode_candidates,
                    candidate_block_size,
                    row_repeat,
                )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # BS03: True = the canonical kernel wrote topk_indices (top-k + TIES + CANON in one launch). False, always so
        # with the flag unset, = the chain below runs exactly as before. try_decode refuses DCP, so the merge is unaffected.
        bs03_done = _bs03 is not None and _bs03.try_decode(
            logits,
            seq_lens,
            next_n,
            topk_indices,
            topk_tokens,
            topk_backend,
            decode_metadata.global_seq_lens,
        )
        # The backend comes from the layer (config is only readable at model
        # construction); dispatchers are cached per backend.
        if not bs03_done:
            get_indexer_topk(topk_backend)(
                logits,
                seq_lens,
                next_n,
                topk_indices,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        if _g5det.ENABLED and not bs03_done:
            _g5det.decode_hook(
                logits,
                seq_lens,
                next_n,
                topk_indices,
                padded_q_quant_cast,
                weights[:num_padded_tokens],
                {
                    "hfp": hidden_states[:num_decode_tokens],
                    "cand": -1 if candidate_blocks is None else int(candidate_write),
                },
                forward_context,
            )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )
        if _DSV41_TOPK_CANON and not bs03_done:
            _canon_topk_(topk_indices)

        if needs_padded_path:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    pcp_shard_decode_requests: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    topk_backend: str = "auto",
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer", "candidate_blocks"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        compress_ratio: int = 1,
        candidate_blocks: torch.Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.compress_ratio = compress_ratio
        # v4.1 two-level selection: the candidate source indexer writes the
        # top candidate blocks here; later indexers mask their scores with it.
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        self.candidate_write = candidate_write
        self.dense_mha_metadata_layer_name = ""
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
        self._parallel_config = parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        self.pcp_shard_decode_requests = parallel_config.pcp_shard_decode_requests
        self._cp_kv_cache_interleave_size: int | None = None
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )

        if vllm_config.kernel_config.enable_jit_warmup:
            from vllm.v1.attention.ops.common import (
                _PACK_SEQ_TRITON_KERNEL,
                _UNPACK_SEQ_TRITON_KERNEL,
            )

            pack_dtype = torch.uint8 if use_fp4_cache else current_platform.fp8_dtype()
            _PACK_SEQ_TRITON_KERNEL.register_warmup(
                dtype=pack_dtype,
                pad_value=0 if use_fp4_cache else -float("inf"),
            )
            _UNPACK_SEQ_TRITON_KERNEL.register_warmup()

            if (
                _DSV41_TOPK_CANON
                and _CANON_TOPK_512_KERNEL is not None
                and current_platform.is_cuda()
                and current_platform.is_device_capability_family(120)
            ):
                _CANON_TOPK_512_KERNEL.register_warmup()
                logger.info_once(
                    "DSV41_TOPK_CANON_FUSED=1: fused width-512 CANON kernel "
                    "registered for JIT warmup"
                )
            elif _CANON_TOPK_512_KERNEL is not None:
                logger.warning_once(
                    "DSV41_TOPK_CANON_FUSED=1 is inert here (needs DSV41_TOPK_CANON=1 "
                    "on CUDA SM12x)"
                )

            if self.dcp_world_size > 1 and current_platform.is_cuda() and has_cutedsl():
                from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (  # noqa: E501
                    _PACK_DCP_TOPK_CANDIDATES_KERNEL,
                    _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL,
                )

                _PACK_DCP_TOPK_CANDIDATES_KERNEL.register_warmup()
                _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL.register_warmup()

    @property
    def cp_kv_cache_interleave_size(self) -> int:
        """With PD+DCP, the real value isn't known until block_size is finalized,
        which happens after this layer is built. Safe to cache after the first access,
        as long as the adjustment always runs before any forward pass
        (it's set up in Worker.initialize_from_config, ahead of warmup/serving).
        """
        if self._cp_kv_cache_interleave_size is None:
            value = self._parallel_config.cp_kv_cache_interleave_size
            if isinstance(get_forward_context().attn_metadata, dict):
                self._cp_kv_cache_interleave_size = value
            return value
        return self._cp_kv_cache_interleave_size

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            if self.use_pcp and self.dcp_world_size > 1:
                raise NotImplementedError(
                    "The ROCm sparse-indexer path does not support PCP+DCP."
                )
            return self.forward_hip(hidden_states, q_quant, k, weights)
        elif current_platform.is_cpu():
            return self.forward_cpu(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm, XPU and CPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_pcp,
            self.pcp_shard_decode_requests,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
            False,
            candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size,
            candidate_write=self.candidate_write,
            topk_backend=self.topk_backend,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        from vllm.platforms.rocm import on_gfx11, on_gfx950

        if (
            rocm_aiter_ops.is_enabled()
            or rocm_aiter_ops.is_rdna_aiter_enabled()
            or on_gfx11()
            # The so-called AITER sparse indexer op has a native gfx950 path:
            # its cache insert, MQA logits, and top-k fallbacks are implemented
            # by local Triton/C++ kernels and do not require the aiter package.
            or on_gfx950()
        ):
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
                compress_ratio=self.compress_ratio,
                candidate_blocks=self.candidate_blocks,
                candidate_block_size=self.candidate_block_size,
                candidate_write=self.candidate_write,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path requires AITER or a supported "
            "native architecture (gfx950/gfx11)."
        )

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        """CPU sparse attention indexer: cache write stays eager Python glue
        (own K-cache layout, not shared with the main attention cache
        write). PREFILL and DECODE both call the ported
        ``fp8_paged_mqa_logits_cpu``/``topk_transform_512_cpu`` kernels,
        which read the paged K-cache directly via ``page_table`` -- no
        eager gather step, no per-request Python loop.

        ``prefill_metadata.chunks`` always has exactly one entry here:
        ``DeepseekV4CPUIndexerMetadataBuilder`` overrides the base chunk
        split to always return the whole step's prefill batch as one
        chunk, since the base chunking only bounds CUDA/XPU's dense M*N
        logits tensor and flat K-gather workspace, neither of which this
        paged kernel allocates.
        """
        assert not self.use_fp4_cache, (
            "CPU sparse indexer doesn't support fp4 cache yet"
        )
        assert isinstance(q_quant, torch.Tensor), (
            "CPU sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        assert self.dcp_world_size <= 1 and not self.use_pcp, (
            "CPU sparse indexer doesn't support decode/prefill context parallelism yet."
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        attn_metadata_narrowed: DeepseekV32IndexerMetadata | None = None
        if isinstance(attn_metadata, dict):
            metadata = attn_metadata[self.k_cache.prefix]
            assert isinstance(metadata, DeepseekV32IndexerMetadata)
            attn_metadata_narrowed = metadata
        if attn_metadata_narrowed is None:
            # Profiling/dummy run: no real metadata to act on.
            return self.topk_indices_buffer

        kv_cache = self.k_cache.kv_cache
        topk_tokens = self.topk_tokens
        topk_indices_buffer = self.topk_indices_buffer
        slot_mapping = attn_metadata_narrowed.slot_mapping
        has_decode = attn_metadata_narrowed.num_decodes > 0
        has_prefill = attn_metadata_narrowed.num_prefills > 0
        num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

        num_tokens = slot_mapping.shape[0]
        if k is not None:
            k = k[:num_tokens]

        if not self.skip_k_cache_insert:
            # Only reachable via DeepseekV32Attention with
            # prefill_context_parallel_size > 1 on CPU -- set_k_cpu/set_s_cpu
            # (the kernels this used to call) have been removed as unused/
            # untested (csrc/cpu/sgl-kernels/store_cache.cpp).
            raise NotImplementedError(
                "SparseAttnIndexer.forward_cpu: skip_k_cache_insert=False "
                "(prefill context parallel on CPU) is not supported."
            )

        topk_indices_buffer[: hidden_states.shape[0]] = -1

        if has_prefill:
            assert topk_tokens == 512, (
                "topk_transform_512_cpu only supports index_topk == 512."
            )
            prefill_metadata = attn_metadata_narrowed.prefill
            assert prefill_metadata is not None
            assert len(prefill_metadata.chunks) == 1, (
                "forward_cpu expects the prefill metadata builder to always "
                "produce a single chunk -- see "
                "DeepseekV4CPUIndexerMetadataBuilder._split_indexer_prefill_chunks."
            )
            chunk = prefill_metadata.chunks[0]
            # kv_cache is a per-layer view into vLLM's shared multi-layer
            # cache allocation, so its block stride generally exceeds
            # block_size * page_width; fp8_paged_mqa_logits_cpu reads
            # kv_view.stride(0) explicitly, so no copy is needed here.
            kv_view = kv_cache.view(kv_cache.shape[0], -1)
            block_size = kv_cache.shape[1]
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if chunk.local_total_seq_lens == 0:
                topk_indices.fill_(-1)
            else:
                assert chunk.local_cu_seq_lens is not None
                # Each token's own (per-request, DCP-local) causal length.
                local_seq_lens = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
                # Recover each token's owning request from its row-start
                # tag. Ties (from zero-length requests) are harmless: those
                # tokens have local length 0 and never dereference
                # page_table.
                req_idx = (
                    torch.searchsorted(
                        chunk.local_cu_seq_lens, chunk.cu_seqlen_ks, right=True
                    )
                    - 1
                )
                page_table = chunk.block_table[req_idx]
                # The true max over this chunk's own rows, NOT
                # chunk.max_local_total_seq_lens (that field sums every
                # request's length in the chunk, bounding the old
                # flat-gather buffer this paged path no longer allocates).
                max_seq_len = int(local_seq_lens.max().item())

                logits = ops.fp8_paged_mqa_logits_cpu(
                    q_slice,
                    kv_view,
                    weights[chunk.token_start : chunk.token_end],
                    local_seq_lens,
                    page_table,
                    block_size,
                    max_seq_len,
                )
                out_page_scratch = torch.empty(
                    (q_slice.shape[0], topk_tokens),
                    dtype=torch.int32,
                    device=kv_cache.device,
                )
                ops.topk_transform_512_cpu(
                    logits,
                    local_seq_lens,
                    page_table,
                    out_page_scratch,
                    block_size,
                    topk_indices,
                )

        if has_decode:
            decode_metadata = attn_metadata_narrowed.decode
            assert decode_metadata is not None
            assert not decode_metadata.requires_padding, (
                "CPU sparse indexer decode path does not support speculative "
                "decoding (native MTP) yet."
            )
            batch_size = decode_metadata.decode_lens.shape[0]
            if batch_size > 0:
                # No native MTP on CPU (asserted above) => exactly one
                # query token per decode request, so the flat slice below
                # is already the batch-major layout
                # fp8_paged_mqa_logits_cpu wants.
                assert num_decode_tokens == batch_size, (
                    "CPU sparse indexer decode path expects exactly one query "
                    "token per decode request."
                )
                assert topk_tokens == 512, (
                    "topk_transform_512_cpu only supports index_topk == 512."
                )
                seq_lens = decode_metadata.seq_lens
                seq_lens = (
                    seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
                )
                block_table = decode_metadata.block_table[:batch_size]
                block_size = kv_cache.shape[1]
                kv_view = kv_cache.view(kv_cache.shape[0], -1)

                logits = ops.fp8_paged_mqa_logits_cpu(
                    q_quant[:num_decode_tokens],
                    kv_view,
                    weights[:num_decode_tokens],
                    seq_lens,
                    block_table,
                    block_size,
                    attn_metadata_narrowed.max_seq_len,
                )

                # out_page_indices is a required kernel output but unused:
                # the indexer's topk output must stay local/compressed-
                # context positions (resolved later by
                # DeepseekV4CPUAttention.forward_mqa via
                # map_local_to_global_slots_cpu).
                out_page_scratch = torch.empty(
                    (batch_size, topk_tokens),
                    dtype=torch.int32,
                    device=kv_cache.device,
                )
                ops.topk_transform_512_cpu(
                    logits,
                    seq_lens,
                    block_table,
                    out_page_scratch,
                    block_size,
                    topk_indices_buffer[:num_decode_tokens, :topk_tokens],
                )

        return topk_indices_buffer


# ---------------------------------------------------------------------------
# g4p item 06 (2026-10-27, default OFF): candidate-only prefill scoring for the
# consumer indexers (24/28/32/36), scatter arm. See the builder mk_g4p06.py.
# ---------------------------------------------------------------------------
from vllm.utils import deep_gemm as _dg  # noqa: E402


def _g4p_cand_mode() -> str:
    raw = os.environ.get("DSV41_CAND_ONLY_INDEXER", "off").strip().lower()
    if raw in ("scatter", "1", "on", "true"):
        return "scatter"
    if raw == "verify":
        return "verify"
    return "off"


_G4P_CAND_MODE = _g4p_cand_mode()
_CAND_BLOCKS = 2048
_CAND_BLOCK_SIZE = 8
_CAND_CAPACITY = _CAND_BLOCKS * _CAND_BLOCK_SIZE  # 16,384 tokens
_CAND_MAX_PREFILL_ROWS = 128
# q06 (spark-two, 128 rows, per consumer-layer call): stock 2.43 / 9.87 / 19.58 ms vs scatter 4.78 / 8.26 / 11.99 ms
# at 128K / 512K / 1M K rows -> break-even ~360K. Below this many K rows the stock path is kept.
try:
    _CAND_MIN_K = max(int(os.environ.get("DSV41_CAND_ONLY_MIN_K", "393216")), 0)
except ValueError:
    _CAND_MIN_K = 393216
_G4P_CAND_STATS = {"calls": 0, "verify": 0, "bad_cols": 0, "bad_calls": 0,
                   "topk_rows_diff": 0, "rows": 0}
_G4P_STEP_REF: dict | None = None
if _G4P_CAND_MODE != "off":
    logger.warning(
        "DSV41_CAND_ONLY_INDEXER=%s armed: consumer-indexer PREFILL chunks with <= %d "
        "rows and >= %d K rows score only their %d candidate blocks (decode unchanged)",
        _G4P_CAND_MODE, _CAND_MAX_PREFILL_ROWS, max(_CAND_CAPACITY, _CAND_MIN_K), _CAND_BLOCKS,
    )


def _g4p_cand_step_off() -> bool:
    """Per-request opt-out of this step (vllm_xargs dsv41_cand_off), set by model_state."""
    global _G4P_STEP_REF
    if _G4P_STEP_REF is None:
        try:
            from vllm.models.deepseek_v41.decoder_replay_layers import G4P_STEP

            _G4P_STEP_REF = G4P_STEP
        except Exception:  # noqa: BLE001
            _G4P_STEP_REF = {}
    return bool(_G4P_STEP_REF.get("cand_off", False))


@triton.jit(
    do_not_specialize=["candidate_stride", "N"],
    do_not_specialize_on_alignment=[
        "K",
        "S",
        "Candidates",
        "Starts",
        "Ends",
        "PackedK",
        "PackedS",
        "Map",
        "candidate_stride",
        "N",
    ],
)
def _pack_prefill_candidate_k_kernel(
    K,
    S,
    Candidates,
    Starts,
    Ends,
    PackedK,
    PackedS,
    Map,
    candidate_stride,
    N,
    CAP: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slots = tl.program_id(1) * TILE + tl.arange(0, TILE)
    offsets = slots % 8
    blocks = tl.load(
        Candidates + row * candidate_stride + slots // 8,
        slots < CAP,
        other=-1,
    ).to(tl.int64)
    start = tl.load(Starts + row).to(tl.int64)
    end = tl.load(Ends + row).to(tl.int64)
    first = start + blocks * 8
    edge = first >= N
    pos = tl.where(edge, N - 1, first + offsets)
    valid = (
        (slots < CAP)
        & (blocks >= 0)
        & (~edge | (offsets == 0))
        & (pos >= start)
        & (pos < end)
        & (pos < N)
    )
    tl.store(Map + row * CAP + slots, tl.where(valid, pos, -1), slots < CAP)
    dims = tl.arange(0, 128)
    values = tl.load(
        K + pos[:, None] * 128 + dims[None, :], valid[:, None], other=0
    )
    sf = tl.load(S + pos, valid, other=0)
    tl.store(
        PackedK + (row * CAP + slots[:, None]) * 128 + dims[None, :],
        values,
        (slots < CAP)[:, None],
    )
    tl.store(PackedS + row * CAP + slots, sf, slots < CAP)


@triton.jit(
    do_not_specialize=["out_stride"],
    do_not_specialize_on_alignment=["Scores", "Map", "Out", "out_stride"],
)
def _scatter_candidate_scores_kernel(
    Scores,
    Map,
    Out,
    score_stride: tl.constexpr,
    out_stride,
    CAP: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slots = tl.program_id(1) * TILE + tl.arange(0, TILE)
    pos = tl.load(Map + row * CAP + slots, slots < CAP, other=-1).to(tl.int64)
    bits = tl.load(
        Scores + row * score_stride + slots,
        (slots < CAP) & (pos >= 0),
        other=0,
    )
    tl.atomic_xchg(
        Out + row * out_stride + pos,
        bits,
        (slots < CAP) & (pos >= 0),
    )


def _mqa_logits_with_max_seqlen_k(
    q_tuple: tuple[torch.Tensor, torch.Tensor | None],
    kv_tuple: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    max_seqlen_k: int,
) -> torch.Tensor:
    _dg._lazy_init()
    if _dg._fp8_fp4_mqa_logits_impl is None:
        return fp8_fp4_mqa_logits(
            q_tuple, kv_tuple, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False
        )
    return _dg._fp8_fp4_mqa_logits_impl(
        q_tuple,
        kv_tuple,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
        max_seqlen_k=max_seqlen_k,
    )


def _g4p_cand_scatter_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Full-width [m, n] fp32 logits: -inf except the valid candidate columns, which
    hold the score bits the same DeepGEMM kernel computes on the packed candidates."""
    m, n = q.shape[0], k.shape[0]
    packed = torch.empty((m * _CAND_CAPACITY, 128), dtype=torch.uint8, device=q.device)
    packed_sf = torch.empty((m * _CAND_CAPACITY,), dtype=torch.int32, device=q.device)
    ids = torch.empty((m, _CAND_CAPACITY), dtype=torch.int32, device=q.device)
    _pack_prefill_candidate_k_kernel[(m, triton.cdiv(_CAND_CAPACITY, 128))](
        k.view(torch.uint8),
        scales.view(torch.int32),
        candidates,
        starts,
        ends,
        packed,
        packed_sf,
        ids,
        candidates.stride(0),
        n,
        _CAND_CAPACITY,
        128,
    )
    lo = torch.arange(m, dtype=torch.int32, device=q.device) * _CAND_CAPACITY
    hi = lo + _CAND_CAPACITY
    compact = _mqa_logits_with_max_seqlen_k(
        (q, None),
        (packed.view(q.dtype), packed_sf.view(torch.float32)),
        weights,
        lo,
        hi,
        max_seqlen_k=_CAND_CAPACITY,
    )
    stride = triton.cdiv(n + 128, 256) * 256
    padded_m = triton.cdiv(m, 4) * 4
    backing = torch.full(
        (padded_m, stride), -float("inf"), dtype=torch.float32, device=q.device
    )
    full_logits = backing[:m, :n]
    _scatter_candidate_scores_kernel[(m, triton.cdiv(_CAND_CAPACITY, 512))](
        compact.view(torch.int32),
        ids,
        full_logits.view(torch.int32),
        compact.stride(0),
        full_logits.stride(0),
        _CAND_CAPACITY,
        512,
    )
    return full_logits


def _g4p_cand_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    candidates: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> torch.Tensor:
    """Consumer-indexer prefill logits + row top-k (written to topk_indices), scatter or
    verify mode. Returns the logits the caller's DCP merge receives (DCP is off here)."""
    m = q.shape[0]
    _G4P_CAND_STATS["calls"] += 1
    _G4P_CAND_STATS["rows"] += m
    if _G4P_CAND_MODE == "scatter":
        full = _g4p_cand_scatter_logits(q, k, scales, weights, starts, ends, candidates)
        ops.top_k_per_row_prefill(
            full, starts, ends, topk_indices, m, full.stride(0), full.stride(1), topk_tokens
        )
        c = _G4P_CAND_STATS["calls"]
        if c in (1, 100, 1000, 10000, 100000):
            logger.info(
                "DSV41_CAND_ONLY_INDEXER scatter: %d consumer prefill calls (%d rows) on "
                "the candidate path (last: m=%d n=%d)", c, _G4P_CAND_STATS["rows"], m, k.shape[0],
            )
        return full
    # verify: serve stock, compare the scatter logits in shadow
    dense = fp8_fp4_mqa_logits((q, None), (k, scales), weights, starts, ends, clean_logits=False)
    _apply_candidate_mask(dense, starts, ends, candidates, _CAND_BLOCK_SIZE)
    ops.top_k_per_row_prefill(
        dense, starts, ends, topk_indices, m, dense.stride(0), dense.stride(1), topk_tokens
    )
    n = k.shape[0]
    full = _g4p_cand_scatter_logits(q, k, scales, weights, starts, ends, candidates)
    col = torch.arange(n, device=q.device, dtype=torch.int32)
    valid = (col[None, :] >= starts[:, None]) & (col[None, :] < ends[:, None])
    neq = dense[:, :n].view(torch.int32) != full.view(torch.int32)
    bad = (neq & valid).sum()
    bad_all = neq.sum()
    tk = torch.full((m, topk_tokens), -1, dtype=torch.int32, device=q.device)
    ops.top_k_per_row_prefill(
        full, starts, ends, tk, m, full.stride(0), full.stride(1), topk_tokens
    )
    c1 = topk_indices[:, :topk_tokens].clone()
    _canon_topk_(c1)
    _canon_topk_(tk)
    rows_diff = (c1 != tk).any(dim=1).sum()
    nbad, nrd, nall = int(bad.item()), int(rows_diff.item()), int(bad_all.item())
    st = _G4P_CAND_STATS
    st["verify"] += 1
    st["bad_cols"] += nbad
    st["bad_calls"] += int(nbad > 0)
    st["topk_rows_diff"] += nrd
    st["bad_cols_all"] = st.get("bad_cols_all", 0) + nall
    if nbad or nall or st["verify"] in (1, 10, 100) or st["verify"] % 200 == 0:
        logger.warning(
            "DSV41_CAND_ONLY_INDEXER verify: call %d m=%d n=%d bad_cols=%d bad_cols_all=%d "
            "topk_rows_diff=%d | totals calls=%d bad_cols=%d bad_cols_all=%d bad_calls=%d "
            "topk_rows_diff=%d rows=%d",
            st["verify"], m, n, nbad, nall, nrd, st["verify"], st["bad_cols"],
            st["bad_cols_all"], st["bad_calls"], st["topk_rows_diff"], st["rows"],
        )
    return dense
