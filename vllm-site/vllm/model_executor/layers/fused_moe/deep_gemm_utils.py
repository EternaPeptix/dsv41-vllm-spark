# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Taken from https://github.com/ModelTC/LightLLM/blob/8ed97c74c18f11505b048b1ba00ba5c0cef8bff6/lightllm/common/fused_moe/deepep_scatter_gather.py
and updated to fit vllm needs and terminology.
"""

import math
import os

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout
from vllm.utils.math_utils import round_up


def expert_num_tokens_round_up_and_sum(
    expert_num_tokens: torch.Tensor, alignment: int
) -> int:
    # Round up each element in expert_num_tokens to the nearest multiple of
    # alignment.
    ent = (expert_num_tokens.to(torch.int64) + (alignment - 1)) // alignment * alignment
    return torch.sum(ent).item()


def compute_aligned_M_and_alignment(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
) -> tuple[int, int]:
    """Return (M_sum, alignment_used).

    `alignment_used` may be smaller than the caller-supplied `alignment` on
    SM100/SM120 when DeepGEMM can JIT a smaller BLOCK_M for the per-call
    expected_m. Callers that index by block size (e.g. ``M_sum // block_m``)
    or assert workspace alignment must use the returned `alignment_used`,
    not their original `alignment` argument.

    Prefer this over the int-returning :func:`compute_aligned_M` when the
    GEMM call site needs to wrap itself in ``mk_alignment_scope`` or
    otherwise reason about the actual per-expert padding.
    """
    if (expert_tokens_meta is not None) and (
        expert_tokens_meta.expert_num_tokens_cpu is not None
    ):
        return (
            expert_num_tokens_round_up_and_sum(
                expert_tokens_meta.expert_num_tokens_cpu, alignment=alignment
            ),
            alignment,
        )

    # expert_num_tokens not on cpu. Cap padding by min(M*num_topk,
    # local_num_experts) — at batch=1 decode only `num_topk` experts can be
    # active, so the worst-case `local_num_experts*(align-1)` is too loose.
    # Also shrink `alignment` to DeepGEMM's per-call theoretical BLOCK_M on
    # SM100/SM120 when smaller.
    expected_m = M * num_topk
    try:
        from vllm.utils.deep_gemm import (
            get_theoretical_mk_alignment_for_contiguous_layout,
        )

        # num_groups=local_num_experts so the helper recovers per-expert em;
        # omitting it over-picks BLOCK_M on SM120 (heuristic assumes em is
        # already per-expert).
        per_call_align = get_theoretical_mk_alignment_for_contiguous_layout(
            expected_m=expected_m,
            num_groups=local_num_experts,
        )
        if per_call_align and per_call_align <= alignment:
            alignment = per_call_align
    except Exception:
        pass

    max_active_experts = min(M * num_topk, local_num_experts)
    M_sum = (M * num_topk) + max_active_experts * (alignment - 1)
    M_sum = round_up(M_sum, alignment)
    return M_sum, alignment


def compute_aligned_M(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
) -> int:
    """Return ``M_sum`` only (backward-compat wrapper).

    Equivalent to :func:`compute_aligned_M_and_alignment`'s first return
    value. Existing downstream callers and the warmup path that only size
    a workspace use this. Call sites that need the actual per-expert
    alignment (to wrap GEMMs in ``mk_alignment_scope``) should use
    :func:`compute_aligned_M_and_alignment` instead.
    """
    M_sum, _ = compute_aligned_M_and_alignment(
        M, num_topk, local_num_experts, alignment, expert_tokens_meta
    )
    return M_sum


@triton.jit
def apply_expert_map(expert_id, expert_map):
    if expert_id != -1:
        expert_id = tl.load(expert_map + expert_id).to(expert_id.dtype)
    return expert_id


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
    ALIGN_M: tl.constexpr,
    USE_PSUM_LAYOUT: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    # Round up to ALIGN_M so cumsum matches the workspace's per-expert slices.
    tokens_per_expert = ((tokens_per_expert + ALIGN_M - 1) // ALIGN_M) * ALIGN_M
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert

    # Extract this block's offset from the register vector (warp shuffle,
    # no global memory round-trip) then write it once to expert_start_loc.
    cur_expert_start = tl.sum(
        tl.where(offset_cumsum == cur_expert, cumsum, tl.zeros_like(cumsum))
    )
    tl.store(expert_start_loc + cur_expert, cur_expert_start)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

    if USE_PSUM_LAYOUT:
        # Logical ends include preceding experts' alignment gaps.
        tl.store(m_indices + cur_expert, cur_expert_start + cur_expert_token_num)
    else:
        m_indices_start_ptr = m_indices + cur_expert_start
        off_expert = tl.arange(0, BLOCK_E)

        # Unwritten padding retains the -1 initialization.
        for start_m in tl.range(0, cur_expert_token_num, BLOCK_E):
            offs = start_m + off_expert
            mask = offs < cur_expert_token_num
            tl.store(
                m_indices_start_ptr + offs,
                cur_expert,
                mask=mask,
            )


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
    PACK_UE8M0: tl.constexpr,
    SCALE_PACKED_SIZE: tl.constexpr,
    SCALE_PACKED_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    output_tensor_stride0 = output_tensor_stride0.to(tl.int64)

    if PACK_UE8M0:
        # One int32 per 4 consecutive 32-wide UE8M0 groups, stored MN-major.
        offs_pk = tl.arange(0, SCALE_PACKED_SIZE_PAD)
        mask_pk = offs_pk < SCALE_PACKED_SIZE
    else:
        offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
        mask_s = offset_in_s < SCALE_HIDDEN_SIZE

    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)

        if PACK_UE8M0:
            # Pack 4 UE8M0 bytes into one int32 (byte j = group 4*pk+j).
            base_s = recv_x_scale + token_id * recv_x_scale_stride0
            g0, g1 = offs_pk * 4, offs_pk * 4 + 1
            g2, g3 = offs_pk * 4 + 2, offs_pk * 4 + 3
            b0 = tl.load(
                base_s + g0 * recv_x_scale_stride1, mask=g0 < SCALE_HIDDEN_SIZE
            )
            b1 = tl.load(
                base_s + g1 * recv_x_scale_stride1, mask=g1 < SCALE_HIDDEN_SIZE
            )
            b2 = tl.load(
                base_s + g2 * recv_x_scale_stride1, mask=g2 < SCALE_HIDDEN_SIZE
            )
            b3 = tl.load(
                base_s + g3 * recv_x_scale_stride1, mask=g3 < SCALE_HIDDEN_SIZE
            )
            packed_s = (
                b0.to(tl.int32)
                | (b1.to(tl.int32) << 8)
                | (b2.to(tl.int32) << 16)
                | (b3.to(tl.int32) << 24)
            )
        else:
            to_copy_s = tl.load(
                recv_x_scale + token_id * recv_x_scale_stride0 + offset_in_s,
                mask=mask_s,
            )

        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)

            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)

            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index_i64 = dest_token_index.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index,
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index_i64 * output_tensor_stride0
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)

                output_tensor_scale_ptr = (
                    output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                )
                if PACK_UE8M0:
                    tl.store(
                        output_tensor_scale_ptr + offs_pk * output_tensor_scale_stride1,
                        packed_s,
                        mask=mask_pk,
                    )
                else:
                    tl.store(
                        output_tensor_scale_ptr + offset_in_s, to_copy_s, mask=mask_s
                    )


@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_map: torch.Tensor | None,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
    align_m: int = 128,
    block_size: int = 128,
    pack_ue8m0: bool = False,
    use_psum_layout: bool = False,
):
    # BLOCK_E is the m_indices fill-loop tile (masked), independent of align_m.
    BLOCK_E = 128
    BLOCK_D = block_size  # block size of activation-scale quantization
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts

    if use_psum_layout:
        assert m_indices.shape == (num_experts,)
    else:
        assert m_indices.shape[0] % align_m == 0
    assert expert_start_loc.shape[0] == num_experts

    # pack_ue8m0: scatter packs 4 UE8M0 bytes per int32; else copies scales as-is.
    scale_hidden_size = hidden_size // BLOCK_D
    scale_packed_size = (scale_hidden_size + 3) // 4 if pack_ue8m0 else 1

    _fwd_kernel_ep_scatter_1[(grid,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        ALIGN_M=align_m,
        USE_PSUM_LAYOUT=use_psum_layout,
    )

    grid = min(recv_topk.shape[0], 1024 * 8)

    _fwd_kernel_ep_scatter_2[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=scale_hidden_size,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(scale_hidden_size),
        PACK_UE8M0=pack_ue8m0,
        SCALE_PACKED_SIZE=scale_packed_size,
        SCALE_PACKED_SIZE_PAD=triton.next_power_of_2(scale_packed_size),
    )
    return


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)

    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(0, topk_num):
            expert_id = tl.load(
                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
            )

            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)

            if expert_id >= 0:
                source_token_index = tl.load(
                    input_index + cur_token * input_index_stride0 + topk_index
                )
                source_token_index_i64 = source_token_index.to(tl.int64)
                acc_weight = tl.load(
                    recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index
                )
                tmp = tl.load(
                    input_tensor
                    + source_token_index_i64 * input_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                accumulator += tmp.to(tl.float32) * acc_weight

        tl.store(
            output_tensor
            + tl.cast(cur_token, tl.int64) * output_tensor_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    expert_map: torch.Tensor | None,
    output_tensor: torch.Tensor,
):
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    BLOCK_D = math.gcd(hidden_size, 1024)
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))

    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return


# =====================================================================================================
# g3 mfront (2026-10-26, default off): fused M32 MoE permute (DSV41_MOE_FUSED_PERMUTE)
#
#   DSV41_MOE_FUSED_PERMUTE=0       stock (default): count_expert_num_tokens, torch.zeros(scales), torch.full(-1),
#                                   ep_scatter_1, ep_scatter_2, then DeepGEMM transpose_and_pack inside FC1
#                                   (6 kernels per target MoE layer, ~14 us/layer on the bench, ~22 us in the
#                                   4-user serving profile)
#   DSV41_MOE_FUSED_PERMUTE=1       target decode MoE on the M32 path only (E 384, topk 6, 32-row alignment,
#                                   FP8 activations with fp32 UE8M0-valued per-128 scales): 2 kernels
#                                     plan   : grouped_layout = -1 and packed scales = 0 over M_sum rows (what
#                                              torch.full / torch.zeros + transpose_and_pack leave), and the
#                                              32-aligned expert starts (same rounding/cumsum as ep_scatter_1)
#                                     scatter: one program per (token, k) slot: dest = atomic cursor (as
#                                              ep_scatter_2), copy the FP8 row, grouped_layout[dest] = expert,
#                                              inv_perm, and the 4 UE8M0 bytes per packed int32 word written
#                                              directly in DeepGEMM's MN-major TMA-aligned layout with the exact
#                                              transpose_and_pack_fp32_into_ue8m0 formula
#                                              (v0 >> 23) | (v1 >> 15) | (v2 >> 7) | (v3 << 1)
#                                   FC1 then receives the packed int32 scales (as FC2 already does from the fused
#                                   SiLU-quant kernel) and skips its transpose_and_pack launch.
#   DSV41_MOE_FUSED_PERMUTE=verify  serve the stock path; also run the fused kernels into shadow buffers and count
#                                   mismatches on the device (rows by slot, packed scale words, grouped_layout,
#                                   expert ends, dead-row scale words); report: touch ~/dsv41-logs/mfront-verify.trigger
#                                   (-> mfront-verify.<host>.<pid>.json).
#
# Exactness (bitwise MoE output): every live slot's FP8 row and packed UE8M0 scale bytes are byte copies / the same bit
# formula; grouped_layout, the per-expert 32-aligned segments, LIVE_ACT expert ends and the dead-row scale words are
# identical arrays. Only the order of rows INSIDE an expert segment can differ (it is atomic-arrival order in both
# paths); every FC1/FC2 output row depends on its own A row and scale only and ep_gather reads through inv_perm, so
# the MoE output is unchanged. tests/test_mfront_gpu.py checks the full MoE output bitwise.
# =====================================================================================================
_DSV41_FP_MODE = os.environ.get("DSV41_MOE_FUSED_PERMUTE", "0").strip().lower()
if _DSV41_FP_MODE in ("", "off"):
    _DSV41_FP_MODE = "0"
if _DSV41_FP_MODE not in ("0", "1", "verify"):
    raise ValueError(f"DSV41_MOE_FUSED_PERMUTE must be 0|1|verify, got {_DSV41_FP_MODE!r}")
_DSV41_FP_SLOTS = 1024  # max M * topk handled by one plan program (M32 path: M <= 96 -> 576)
_DSV41_FP_FILL_ROWS = 1024


@triton.jit
def _dsv41_fp_plan_kernel(
    topk_ids,
    ids_stride0,
    ids_stride1,
    cursor,          # [E] int32 out: 32-aligned expert start (advanced to the live end by the scatter kernel)
    layout,          # [M_sum] int32 out: -1
    sfp,             # packed int32 scales, element (row r, pack p) at sfp + p * mn_aligned + r: 0
    M,
    M_sum,
    mn_aligned,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    ALIGN: tl.constexpr,
    PACKS: tl.constexpr,
    SLOTS: tl.constexpr,
    E_PAD: tl.constexpr,
    FILL_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * FILL_ROWS + tl.arange(0, FILL_ROWS)
    rm = r < M_sum
    tl.store(layout + r, tl.full([FILL_ROWS], -1, tl.int32), mask=rm)
    zero = tl.zeros([FILL_ROWS], dtype=tl.int32)
    for p in tl.static_range(PACKS):
        tl.store(sfp + p * mn_aligned + r, zero, mask=rm)
    if pid == 0:
        s = tl.arange(0, SLOTS)
        sm = s < M * TOPK
        t = s // TOPK
        k = s % TOPK
        ids = tl.load(topk_ids + t * ids_stride0 + k * ids_stride1, mask=sm, other=-1).to(tl.int32)
        valid = sm & (ids >= 0) & (ids < E)
        cnt = tl.histogram(tl.where(valid, ids, E), E_PAD)
        b = tl.arange(0, E_PAD)
        cnt = tl.where(b < E, cnt, 0)
        aligned = (cnt + ALIGN - 1) // ALIGN * ALIGN
        start = tl.cumsum(aligned, axis=0) - aligned
        tl.store(cursor + b, start.to(tl.int32), mask=b < E)


@triton.jit
def _dsv41_fp_scatter_kernel(
    topk_ids,
    ids_stride0,
    ids_stride1,
    aq_words,        # [M, H/4] int32 view of the FP8 rows
    aq_row_words,
    sf_bits,         # [M, SF_K] int32 view of the fp32 scales (any strides)
    sf_stride0,
    sf_stride1,
    cursor,
    out_words,       # [M_sum, H/4] int32 view of the permuted FP8 rows
    out_row_words,
    sfp,
    mn_aligned,
    layout,
    inv_perm,
    inv_stride0,
    inv_stride1,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    H_WORDS: tl.constexpr,
    H_BLOCK: tl.constexpr,
    SF_K: tl.constexpr,
    PACKS: tl.constexpr,
    PACKS_PAD: tl.constexpr,
):
    slot = tl.program_id(0)
    t = slot // TOPK
    k = slot % TOPK
    e = tl.load(topk_ids + t * ids_stride0 + k * ids_stride1).to(tl.int32)
    if (e >= 0) & (e < E):
        dest = tl.atomic_add(cursor + e, 1)
        tl.store(inv_perm + t * inv_stride0 + k * inv_stride1, dest)
        tl.store(layout + dest, e)
        w = tl.arange(0, H_BLOCK)
        wm = w < H_WORDS
        row = tl.load(aq_words + t.to(tl.int64) * aq_row_words + w, mask=wm)
        tl.store(out_words + dest.to(tl.int64) * out_row_words + w, row, mask=wm)
        p = tl.arange(0, PACKS_PAD)
        pm = p < PACKS
        base = sf_bits + t.to(tl.int64) * sf_stride0
        g0 = p * 4
        v0 = tl.load(base + g0 * sf_stride1, mask=pm & (g0 < SF_K), other=0).to(tl.uint32, bitcast=True)
        v1 = tl.load(base + (g0 + 1) * sf_stride1, mask=pm & (g0 + 1 < SF_K), other=0).to(tl.uint32, bitcast=True)
        v2 = tl.load(base + (g0 + 2) * sf_stride1, mask=pm & (g0 + 2 < SF_K), other=0).to(tl.uint32, bitcast=True)
        v3 = tl.load(base + (g0 + 3) * sf_stride1, mask=pm & (g0 + 3 < SF_K), other=0).to(tl.uint32, bitcast=True)
        packed = (v0 >> 23) | (v1 >> 15) | (v2 >> 7) | (v3 << 1)
        tl.store(sfp + p * mn_aligned + dest, packed.to(tl.int32, bitcast=True), mask=pm)


def dsv41_fp_eligible(aq, aq_scale, topk_ids, local_num_experts, expert_map, expert_tokens_meta,
                      block_k, use_psum_layout, align_used) -> bool:
    """g3 mfront gate: the qualified target M32 decode geometry only."""
    if _DSV41_FP_MODE == "0":
        return False
    if expert_map is not None or use_psum_layout or align_used != 32:
        return False
    if expert_tokens_meta is not None and getattr(expert_tokens_meta, "expert_num_tokens", None) is not None:
        return False
    if local_num_experts != 384 or topk_ids.dim() != 2 or topk_ids.size(1) != 6:
        return False
    M, H = aq.shape
    if topk_ids.size(0) != M or M * 6 > _DSV41_FP_SLOTS or block_k != 128 or H % 128 or H % 4:
        return False
    if aq.dtype != torch.float8_e4m3fn or not aq.is_contiguous():
        return False
    if aq_scale.dtype != torch.float32 or aq_scale.dim() != 2 or tuple(aq_scale.shape) != (M, H // 128):
        return False
    return topk_ids.dtype in (torch.int32, torch.int64)


def dsv41_fused_permute(aq, aq_scale, topk_ids, local_num_experts, aq_out, M_sum, align_used):
    """Returns (aq_out, packed_scales_int32 [M_sum, packs] MN-major, grouped_layout, inv_perm, expert_ends)."""
    M, H = aq.shape
    topk = topk_ids.size(1)
    E = local_num_experts
    sf_k = H // 128
    packs = (sf_k + 3) // 4
    mn_aligned = round_up(M_sum, 4)
    device = aq.device
    if aq_out is None:
        aq_out = torch.empty((M_sum, H), device=device, dtype=aq.dtype)
    assert aq_out.shape == (M_sum, H) and aq_out.is_contiguous()
    sfp = torch.empty_strided((M_sum, packs), (1, mn_aligned), device=device, dtype=torch.int32)
    layout = torch.empty((M_sum,), device=device, dtype=torch.int32)
    inv_perm = torch.empty(topk_ids.shape, device=device, dtype=torch.int32)
    cursor = torch.empty((E,), device=device, dtype=torch.int32)
    _dsv41_fp_plan_kernel[(triton.cdiv(M_sum, _DSV41_FP_FILL_ROWS),)](
        topk_ids, topk_ids.stride(0), topk_ids.stride(1), cursor, layout, sfp, M, M_sum, mn_aligned,
        E=E, TOPK=topk, ALIGN=align_used, PACKS=packs, SLOTS=_DSV41_FP_SLOTS,
        E_PAD=triton.next_power_of_2(E + 1), FILL_ROWS=_DSV41_FP_FILL_ROWS, num_warps=4,
    )
    if M * topk > 0:
        aw = aq.view(torch.int32)
        ow = aq_out.view(torch.int32)
        sb = aq_scale.view(torch.int32)
        _dsv41_fp_scatter_kernel[(M * topk,)](
            topk_ids, topk_ids.stride(0), topk_ids.stride(1), aw, aw.stride(0), sb, sb.stride(0), sb.stride(1),
            cursor, ow, ow.stride(0), sfp, mn_aligned, layout, inv_perm, inv_perm.stride(0), inv_perm.stride(1),
            E=E, TOPK=topk, H_WORDS=H // 4, H_BLOCK=triton.next_power_of_2(H // 4), SF_K=sf_k, PACKS=packs,
            PACKS_PAD=triton.next_power_of_2(packs), num_warps=4,
        )
    return aq_out, sfp, layout, inv_perm, cursor


@triton.jit
def _dsv41_fp_compare_kernel(
    topk_ids, ids_stride0, ids_stride1,
    inv_s, inv_s0, inv_s1, inv_f, inv_f0, inv_f1,
    out_s, out_f, row_words,
    sf_s, sf_s0, sf_s1,
    sfp_f, mn_aligned,
    lay_s, lay_f, M_sum, cur_s, cur_f,
    counters,
    M,
    E: tl.constexpr, TOPK: tl.constexpr, H_WORDS: tl.constexpr, H_BLOCK: tl.constexpr, SF_K: tl.constexpr,
    PACKS: tl.constexpr, PACKS_PAD: tl.constexpr, LAY_BLOCK: tl.constexpr, E_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < M * TOPK:
        t = pid // TOPK
        k = pid % TOPK
        e = tl.load(topk_ids + t * ids_stride0 + k * ids_stride1).to(tl.int32)
        if (e >= 0) & (e < E):
            ds = tl.load(inv_s + t * inv_s0 + k * inv_s1).to(tl.int64)
            df = tl.load(inv_f + t * inv_f0 + k * inv_f1).to(tl.int64)
            w = tl.arange(0, H_BLOCK)
            wm = w < H_WORDS
            a = tl.load(out_s + ds * row_words + w, mask=wm, other=0)
            b = tl.load(out_f + df * row_words + w, mask=wm, other=0)
            bad = tl.sum(tl.where(wm & (a != b), 1, 0), axis=0)
            p = tl.arange(0, PACKS_PAD)
            pm = p < PACKS
            g0 = p * 4
            base = sf_s + ds * sf_s0
            v0 = tl.load(base + g0 * sf_s1, mask=pm & (g0 < SF_K), other=0).to(tl.uint32, bitcast=True)
            v1 = tl.load(base + (g0 + 1) * sf_s1, mask=pm & (g0 + 1 < SF_K), other=0).to(tl.uint32, bitcast=True)
            v2 = tl.load(base + (g0 + 2) * sf_s1, mask=pm & (g0 + 2 < SF_K), other=0).to(tl.uint32, bitcast=True)
            v3 = tl.load(base + (g0 + 3) * sf_s1, mask=pm & (g0 + 3 < SF_K), other=0).to(tl.uint32, bitcast=True)
            want = (v0 >> 23) | (v1 >> 15) | (v2 >> 7) | (v3 << 1)
            got = tl.load(sfp_f + p * mn_aligned + df, mask=pm, other=0).to(tl.uint32, bitcast=True)
            sbad = tl.sum(tl.where(pm & (want != got), 1, 0), axis=0)
            tl.atomic_add(counters + 1, 1)
            tl.atomic_add(counters + 2, bad.to(tl.int64))
            tl.atomic_add(counters + 3, sbad.to(tl.int64))
    else:
        nbad = tl.zeros([LAY_BLOCK], dtype=tl.int64)
        for r0 in range(0, M_sum, LAY_BLOCK):
            r = r0 + tl.arange(0, LAY_BLOCK)
            rm = r < M_sum
            ls = tl.load(lay_s + r, mask=rm, other=0)
            lf = tl.load(lay_f + r, mask=rm, other=0)
            nbad += tl.where(rm & (ls != lf), 1, 0)
            for q in tl.static_range(PACKS):  # dead rows: packed scale words must be 0 (as transpose_and_pack of 0.0)
                z = tl.load(sfp_f + q * mn_aligned + r, mask=rm & (lf < 0), other=0)
                nbad += tl.where(rm & (lf < 0) & (z != 0), 1, 0)
        b = tl.arange(0, E_PAD)
        bm = b < E
        cs = tl.load(cur_s + b, mask=bm, other=0)
        cf = tl.load(cur_f + b, mask=bm, other=0)
        tot = tl.sum(nbad, axis=0) + tl.sum(tl.where(bm & (cs != cf), 1, 0), axis=0).to(tl.int64)
        tl.atomic_add(counters + 0, 1)
        tl.atomic_add(counters + 4, tot)


class _Dsv41FpVerifier:
    """DSV41_MOE_FUSED_PERMUTE=verify: counters [layouts compared, slots compared, row-word mismatches,
    scale-word mismatches, layout/ends/dead-scale mismatches, calls skipped during capture]."""

    def __init__(self):
        assert not torch.cuda.is_current_stream_capturing()
        self.device = torch.cuda.current_device()
        self.counters = torch.zeros(8, dtype=torch.int64, device="cuda")
        self._host = torch.zeros(8, dtype=torch.int64, pin_memory=True)
        self._side = torch.cuda.Stream()
        self.skipped = 0
        self._start_reporter()

    def check(self, aq, aq_scale, topk_ids, E, M_sum, align_used, served):
        s_out, s_scale, s_layout, s_inv, s_cursor = served
        f_out = torch.empty_like(s_out)
        f_out, f_sfp, f_layout, f_inv, f_cursor = dsv41_fused_permute(aq, aq_scale, topk_ids, E, f_out, M_sum,
                                                                     align_used)
        M, H = aq.shape
        topk = topk_ids.size(1)
        sf_k = H // 128
        packs = (sf_k + 3) // 4
        ss = s_scale.view(torch.int32)
        so, fo = s_out.view(torch.int32), f_out.view(torch.int32)
        _dsv41_fp_compare_kernel[(M * topk + 1,)](
            topk_ids, topk_ids.stride(0), topk_ids.stride(1), s_inv, s_inv.stride(0), s_inv.stride(1),
            f_inv, f_inv.stride(0), f_inv.stride(1), so, fo, so.stride(0), ss, ss.stride(0), ss.stride(1),
            f_sfp, round_up(M_sum, 4), s_layout, f_layout, M_sum, s_cursor, f_cursor, self.counters, M,
            E=E, TOPK=topk, H_WORDS=H // 4, H_BLOCK=triton.next_power_of_2(H // 4), SF_K=sf_k, PACKS=packs,
            PACKS_PAD=triton.next_power_of_2(packs), LAY_BLOCK=1024, E_PAD=triton.next_power_of_2(E + 1),
            num_warps=4,
        )

    def _start_reporter(self):
        import json
        import socket
        import threading
        import time
        from pathlib import Path

        root = Path(os.environ.get("DSV41_MOE_FUSED_PERMUTE_VERIFY_DIR", str(Path.home() / "dsv41-logs")))
        delay = float(os.environ.get("DSV41_MOE_FUSED_PERMUTE_VERIFY_DELAY_S", "600"))
        trigger = root / "mfront-verify.trigger"
        out = root / f"mfront-verify.{socket.gethostname()}.{os.getpid()}.json"
        born = time.time()

        def loop():
            seen = None
            while True:
                time.sleep(2.0)
                try:
                    mtime = trigger.stat().st_mtime
                except OSError:
                    continue
                if mtime == seen or time.time() - born < delay:  # never touch CUDA during startup/capture
                    continue
                seen = mtime
                try:
                    with torch.cuda.device(self.device), torch.cuda.stream(self._side):
                        self._host.copy_(self.counters, non_blocking=True)
                        self._side.synchronize()
                    c = self._host.tolist()
                    bad = c[2] + c[3] + c[4]
                    rep = dict(host=socket.gethostname(), pid=os.getpid(), time=time.time(), mode="verify",
                               calls=c[0], slots=c[1], row_word_mismatch=c[2], scale_word_mismatch=c[3],
                               layout_ends_deadscale_mismatch=c[4], skipped_eager_capture=self.skipped,
                               status="MISMATCH" if bad else ("OK" if c[0] else "NO-CALLS"))
                    tmp = out.with_suffix(".tmp")
                    tmp.write_text(json.dumps(rep, indent=1))
                    tmp.replace(out)
                except Exception as exc:  # diagnostics must never kill serving
                    try:
                        out.with_suffix(".err").write_text(repr(exc))
                    except OSError:
                        pass

        threading.Thread(target=loop, name="mfront-verify-report", daemon=True).start()


_dsv41_fp_verifier: "_Dsv41FpVerifier | None" = None


def _dsv41_fp_verify(aq, aq_scale, topk_ids, E, M_sum, align_used, served):
    global _dsv41_fp_verifier
    if _dsv41_fp_verifier is None:
        if torch.cuda.is_current_stream_capturing():
            return  # first eligible call inside a capture: that graph serves stock unverified
        _dsv41_fp_verifier = _Dsv41FpVerifier()
    _dsv41_fp_verifier.check(aq, aq_scale, topk_ids, E, M_sum, align_used, served)


def deepgemm_moe_permute(
    aq: torch.Tensor,
    aq_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    aq_out: torch.Tensor | None = None,
    block_size: int | None = None,
    use_psum_layout: bool = False,
    alignment_m: int | None = None,
    return_expert_ends: bool = False,
):
    """Scatter ``aq`` into the per-expert aligned contiguous layout.

    Returns ``(aq_out, aq_scale_out, grouped_layout, inv_perm, align_used)``.
    With ``return_expert_ends=True`` (DSV41_MOE_LIVE_ACT, k1 exp 2) a sixth
    value is appended: ``expert_start_loc`` after ep_scatter. ``_fwd_kernel_
    ep_scatter_1`` writes each expert's aligned start and the atomic_add in
    ``_fwd_kernel_ep_scatter_2`` advances it once per routed row, so it ends
    as the exclusive end of the expert's live rows (DeepGEMM psum layout:
    ``roundup(end[e-1], align_used) == start[e]``). Default: unchanged.
    """
    assert aq.ndim == 2
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    H = aq.size(1)
    device = aq.device

    block_m, block_k = get_mk_alignment_for_contiguous_layout()
    if alignment_m is not None:
        if alignment_m not in (32, 64, 128) or alignment_m > block_m:
            raise ValueError(f"Invalid explicit grouped MoE M alignment: {alignment_m}")
        block_m = alignment_m
    # The activation-scale group size may differ from the M/K tile alignment
    # (e.g. MXFP8 uses a 32-element scale group while block_k stays 128).
    if block_size is not None:
        block_k = block_size

    M_sum, align_used = compute_aligned_M_and_alignment(
        M=topk_ids.size(0),
        num_topk=topk_ids.size(1),
        local_num_experts=local_num_experts,
        alignment=block_m,
        expert_tokens_meta=expert_tokens_meta,
    )

    # g3 mfront (DSV41_MOE_FUSED_PERMUTE): fused plan + scatter on the qualified M32 geometry.
    _fp = dsv41_fp_eligible(aq, aq_scale, topk_ids, local_num_experts, expert_map, expert_tokens_meta,
                            block_k, use_psum_layout, align_used)
    if _fp and _DSV41_FP_MODE == "1":
        f_out, f_sfp, f_layout, f_inv, f_ends = dsv41_fused_permute(
            aq, aq_scale, topk_ids, local_num_experts, aq_out, M_sum, align_used)
        if return_expert_ends:
            return f_out, f_sfp, f_layout, f_inv, align_used, f_ends
        return f_out, f_sfp, f_layout, f_inv, align_used

    expert_start_loc = torch.empty(
        (local_num_experts), device=device, dtype=torch.int32
    )

    assert aq_out is None or aq_out.shape == (M_sum, H)
    if aq_out is None:
        aq_out = torch.empty((M_sum, H), device=device, dtype=aq.dtype)

    # uint8 UE8M0 (MXFP8) -> scatter packs into DeepGEMM's int32 MN-major
    # TMA-aligned layout; float32 (FP8/FP4) scattered row-major as-is.
    pack_ue8m0 = aq_scale.dtype == torch.uint8
    sf_k = H // block_k
    if pack_ue8m0:
        packed_sf_k = (sf_k + 3) // 4
        tma_aligned_mn = round_up(M_sum, 4)
        aq_scale_out = torch.empty_strided(
            (M_sum, packed_sf_k),
            (1, tma_aligned_mn),
            device=device,
            dtype=torch.int32,
        )
    else:
        aq_scale_out = torch.zeros((M_sum, sf_k), device=device, dtype=torch.float32)

    if use_psum_layout:
        grouped_layout = torch.empty(
            (local_num_experts,), device=device, dtype=torch.int32
        )
    else:
        # Negative expert IDs mark padding for the contiguous scheduler.
        grouped_layout = torch.full(
            (M_sum,), fill_value=-1, device=device, dtype=torch.int32
        )
    inv_perm = torch.empty(topk_ids.shape, device=device, dtype=torch.int32)

    expert_num_tokens = None
    if expert_tokens_meta is not None:
        expert_num_tokens = expert_tokens_meta.expert_num_tokens
    if expert_num_tokens is None:
        # A cudagraph-decode carrier may hold only psum_recv_per_rank and leave
        # expert_num_tokens None; count locally as in the meta-absent case.
        expert_num_tokens = count_expert_num_tokens(
            topk_ids, local_num_experts, expert_map
        )

    ep_scatter(
        recv_x=aq,
        recv_x_scale=aq_scale,
        recv_topk=topk_ids,
        num_recv_tokens_per_expert=expert_num_tokens,
        expert_start_loc=expert_start_loc,
        expert_map=expert_map,
        output_tensor=aq_out,
        output_tensor_scale=aq_scale_out,
        m_indices=grouped_layout,
        output_index=inv_perm,
        align_m=align_used,
        block_size=block_k,
        pack_ue8m0=pack_ue8m0,
        use_psum_layout=use_psum_layout,
    )

    if _fp and _DSV41_FP_MODE == "verify":
        _dsv41_fp_verify(aq, aq_scale, topk_ids, local_num_experts, M_sum, align_used,
                         (aq_out, aq_scale_out, grouped_layout, inv_perm, expert_start_loc))

    if return_expert_ends:
        return aq_out, aq_scale_out, grouped_layout, inv_perm, align_used, expert_start_loc
    return aq_out, aq_scale_out, grouped_layout, inv_perm, align_used


def deepgemm_unpermute_and_reduce(
    a: torch.Tensor,  # Grouped gemm output
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    expert_map: torch.Tensor | None,
    output: torch.Tensor,
):
    return ep_gather(
        input_tensor=a,
        recv_topk_ids=topk_ids,
        recv_topk_weight=topk_weights,
        input_index=inv_perm,
        expert_map=expert_map,
        output_tensor=output,
    )
