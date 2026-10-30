# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verification for DSpark speculative decoding."""

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.async_utils import StepTimingSample, stream
from vllm.v1.worker.gpu.attn_utils import (
    get_query_lens_mismatch_unsupported_backend,
    get_varlen_cudagraph_unsupported_backend,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
_PROFILE_REPLAYS = 5


# ---------------------------------------------------------------------------
# DSV41 r4 verify-router: DSV41_VERIFY_CAP (default OFF).
#
# Port of knapcio's DSV41_VERIFY_CAP=conf:T to vLLM adaptive verification.
# The CPU budget (stale confidences) still fixes the verify layout and graph
# size. After the GPU assigns draft slots from the *fresh* confidences, every
# admitted draft whose fresh survival (running product of the DSpark
# confidence head) is below tau is declared dead:
#   * its verify row is marked as padding, so with DSV41_ROUTE_LIVE the MoE
#     router gives it no new experts (no extra expert weight reads);
#   * its draft is marked invalid for the rejection sampler (the upstream
#     grammar mechanism: draft id -1), so acceptance stops before it and the
#     next token comes from the last live row (greedy: target argmax; sampled:
#     plain target distribution, i.e. the dead position is q = 0).
# Survival at position i uses only drafts < i (DSpark's confidence head reads
# the Markov embedding of the previous token), so the cut is a stopping rule:
# exact for greedy and for sampling. Live rows only see earlier rows (causal
# attention, row-wise MoE/dense), so their logits are unchanged.
#   DSV41_VERIFY_CAP=conf:0.1    fixed tau (knapcio's production value)
#   DSV41_VERIFY_CAP=auto[:s]    tau = s * (marginal verify ms per row at the
#                                chosen width) * (expected tokens / step ms)
#   DSV41_VERIFY_CAP_MIN=1       drafts always kept per verifying request
#   DSV41_VERIFY_CAP_TAU_MAX=0.5 clamp for auto
#   DSV41_VERIFY_CAP_LOG=N       rank-0 log every N steps (one sync per N)
#   DSV41_VERIFY_CAP_TORCH=1     torch reference instead of the Triton kernel
# DSV41_AV_INTRA_BUCKET=f (0..1, default 0 = off): inside a cudagraph pad
# bucket, price a real row at f * the profiled per-row slope (padding rows
# are free once DSV41_ROUTE_LIVE skips them), so the trimmer may stop
# between captured sizes.
# ---------------------------------------------------------------------------
def _dsv41_parse_vcap() -> tuple[bool, bool, float]:
    """Returns (enabled, auto, value)."""
    spec = os.environ.get("DSV41_VERIFY_CAP", "").strip().lower()
    if spec in ("", "0", "off", "false"):
        return False, False, 0.0
    if spec.startswith("conf:"):
        return True, False, float(spec.split(":", 1)[1])
    if spec == "auto":
        return True, True, 1.0
    if spec.startswith("auto:"):
        return True, True, float(spec.split(":", 1)[1])
    raise ValueError(
        f"DSV41_VERIFY_CAP={spec!r}: use conf:<tau>, auto or auto:<scale>"
    )


@triton.jit
def _dsv41_live_cap_kernel(
    conf_ptr,  # [max_num_reqs, S] fp32, indexed by request-state slot
    conf_stride,
    idx_mapping_ptr,  # [num_reqs]
    cap_ptr,  # [num_reqs] admitted drafts
    cu_num_logits_ptr,  # [num_reqs + 1]
    logits_indices_ptr,  # [num_logits] token row of each logit row
    is_padding_ptr,  # [num_tokens_padded] uint8 view of the bool buffer
    dead_ptr,  # [num_logits] uint8 view of the bool output
    live_ptr,  # [num_reqs] int32 (stats)
    tau,
    kmin,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    r = tl.program_id(0)
    slot = tl.load(idx_mapping_ptr + r).to(tl.int64)
    cap = tl.load(cap_ptr + r).to(tl.int32)
    offs = tl.arange(0, BLOCK_S)
    m = offs < S
    c = tl.load(conf_ptr + slot * conf_stride + offs, mask=m, other=0.0).to(
        tl.float32
    )
    # Confidences are probabilities, so survival never increases along the
    # drafts and the drafts with survival >= tau form a prefix: count them.
    surv = tl.cumprod(c, axis=0)
    ok = ((surv >= tau) & m).to(tl.int32)
    keep = tl.sum(ok, axis=0)
    live = tl.minimum(tl.maximum(keep, kmin), cap)
    tl.store(live_ptr + r, live)
    start = tl.load(cu_num_logits_ptr + r).to(tl.int64)
    end = tl.load(cu_num_logits_ptr + r + 1).to(tl.int64)
    n = (end - start - 1).to(tl.int32)
    j = tl.arange(0, BLOCK_J)
    jm = j <= n
    dead = (j > live) & jm
    rows = tl.load(logits_indices_ptr + start + j, mask=jm, other=0).to(tl.int64)
    tl.store(is_padding_ptr + rows, dead.to(tl.uint8), mask=jm)
    tl.store(dead_ptr + start + j, dead.to(tl.uint8), mask=jm)


def dsv41_live_cap_torch(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    logits_indices: torch.Tensor,
    is_padding: torch.Tensor,
    tau: float,
    kmin: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for _dsv41_live_cap_kernel (also used by the unit test)."""
    surv = confidence_probs[idx_mapping].float().cumprod(dim=1)
    keep = (surv >= tau).to(torch.int32).cumprod(dim=1).sum(dim=1)
    live = torch.minimum(keep.clamp(min=kmin), capacities.to(torch.int32))
    live_by_slot = torch.zeros(
        confidence_probs.shape[0], dtype=torch.int32, device=live.device
    )
    live_by_slot[idx_mapping] = live.to(torch.int32)
    dead = expanded_local_pos > live_by_slot[expanded_idx_mapping]
    is_padding[logits_indices] = dead
    return dead, live


# ---------------------------------------------------------------------------
# DSV41 g4d item 09 (Codex #9, 2026-10-27): DSV41_FUSED_AV (default OFF).
#
# reallocate_drafts() runs between the drafter graph and the target graph. Live it
# issues 3 small H2D copies (query_start_loc in prepare_inputs, the scheduled draft
# counts, the non-draft token counts; plus cu_num_logits in prepare_inputs, which
# AV overwrites), the torch.compile'd budget assignment (6 kernels: survival scan,
# gatherTopK, bitonicSortKVInPlace, zeros_like, index_fill, sum) and two cumsum
# chains (2 zero_, 2 adds, 2 x (DeviceScanInit + DeviceScan), 1 fill): about 80 us
# of GPU time and launch gaps per decode step (2027f4d8 nsys, DT segment).
#   DSV41_FUSED_AV=1       one packed H2D + one Triton kernel produce the same
#                          capacities / num_non_draft / cu_num_logits /
#                          query_start_loc, and prepare_inputs skips the two
#                          H2D copies that reallocate_drafts overwrites.
#   DSV41_FUSED_AV=verify  serve the stock path; run the fused kernel into shadow
#                          buffers and count mismatching entries on the device
#                          (rank-0 log every DSV41_FUSED_AV_LOG steps, default 2000).
# Bitwise contract (integers):
#   * survival = the live inductor kernel's scan (triton_red_fused_arange_cumprod_
#     ge_index_masked_fill_unsqueeze_0, best config XBLOCK 4 / R0_BLOCK 8 /
#     num_warps 2, layout threadsPerWarp [4, 8]: one element per lane, Hillis-
#     Steele shfl.up 1/2/4 + mul.f32): s2 = c0*(c1*c2), s3 = (c0*c1)*(c2*c3),
#     s4 = c0*((c1*c2)*(c3*c4)), ...; slots past a request's scheduled drafts -inf;
#   * admission = torch.topk(draft_budget) on the flattened [num_reqs, S] scores:
#     radix key of the fp32 value (TopKTypeConfig<float>::convert, NaN largest),
#     ties to the lower flat index (gatherTopK writes equal values first-seen);
#   * capacities = admitted per request; cu_num_logits / query_start_loc = integer
#     prefix sums, query_start_loc[num_reqs + 1:] = num_tokens.
# ---------------------------------------------------------------------------
def _dsv41_fused_av_mode() -> int:
    v = os.environ.get("DSV41_FUSED_AV", "").strip().lower()
    if v in ("", "0", "off", "false"):
        return 0
    if v in ("1", "on", "true"):
        return 1
    if v == "verify":
        return 2
    raise ValueError(f"DSV41_FUSED_AV={v!r}: use 0, 1 or verify")


@triton.jit
def _dsv41_fav_keys(conf_ptr, conf_stride, idx_ptr, sched_ptr, num_reqs, r, s,
                    S: tl.constexpr):
    """Radix keys (torch.topk order, int64 in [0, 2^32)) of the slots (r, s), their
    validity (r < num_reqs, s < S) and the survival values (for the unit test)."""
    rv = r < num_reqs
    valid = rv & (s < S)
    slot = tl.load(idx_ptr + r, mask=rv, other=0).to(tl.int64)
    sched = tl.load(sched_ptr + r, mask=rv, other=0)
    base = conf_ptr + slot * conf_stride
    c0 = tl.load(base + 0, mask=rv & (0 < S), other=0.0).to(tl.float32)
    c1 = tl.load(base + 1, mask=rv & (1 < S), other=0.0).to(tl.float32)
    c2 = tl.load(base + 2, mask=rv & (2 < S), other=0.0).to(tl.float32)
    c3 = tl.load(base + 3, mask=rv & (3 < S), other=0.0).to(tl.float32)
    c4 = tl.load(base + 4, mask=rv & (4 < S), other=0.0).to(tl.float32)
    c5 = tl.load(base + 5, mask=rv & (5 < S), other=0.0).to(tl.float32)
    c6 = tl.load(base + 6, mask=rv & (6 < S), other=0.0).to(tl.float32)
    c7 = tl.load(base + 7, mask=rv & (7 < S), other=0.0).to(tl.float32)
    # Hillis-Steele over 8 lanes, exactly the live scan's shfl.up 1 / 2 / 4 steps.
    a1 = c0 * c1
    a2 = c1 * c2
    a3 = c2 * c3
    a4 = c3 * c4
    a5 = c4 * c5
    a6 = c5 * c6
    a7 = c6 * c7
    b2 = c0 * a2
    b3 = a1 * a3
    b4 = a2 * a4
    b5 = a3 * a5
    b6 = a4 * a6
    b7 = a5 * a7
    d4 = c0 * b4
    d5 = a1 * b5
    d6 = b2 * b6
    d7 = b3 * b7
    surv = tl.where(s == 0, c0,
           tl.where(s == 1, a1,
           tl.where(s == 2, b2,
           tl.where(s == 3, b3,
           tl.where(s == 4, d4,
           tl.where(s == 5, d5,
           tl.where(s == 6, d6, d7)))))))
    v = tl.where(valid & (s < sched), surv, float("-inf"))
    ib = v.to(tl.int32, bitcast=True).to(tl.int64)
    key = tl.where(ib < 0, -1 - ib, ib + 2147483648)
    key = tl.where(v == v, key, 4294967295)
    return key, valid, surv


@triton.jit(do_not_specialize=["num_reqs", "qsl_len", "draft_budget", "mode",
                               "num_tokens", "num_bonus"])
def _dsv41_fused_av_kernel(
    conf_ptr,  # [max_num_reqs, S] fp32 fresh confidences, by request-state slot
    conf_stride,
    idx_ptr,  # [num_reqs] int64 idx_mapping
    packed_ptr,  # [2 * num_reqs] int32: scheduled drafts, then non-draft tokens
    cap_ptr,  # out [num_reqs] int32 admitted drafts (_batch_draft_capacity)
    nd_ptr,  # out [num_reqs] int32 non-draft tokens (_num_non_draft_tokens)
    cu_ptr,  # out [num_reqs + 1] int32 cu_num_logits
    qsl_ptr,  # out [qsl_len] int32 query_start_loc (whole buffer)
    dbg_ptr,  # out [BLOCK_R, BLOCK_S] fp32 survival (DEBUG only)
    num_reqs,
    qsl_len,
    draft_budget,
    mode,  # 0: budget 0, 1: budget == scheduled total, 2: top-k admission
    num_tokens,
    num_bonus,
    S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_S: tl.constexpr,
    CJ: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    DEBUG: tl.constexpr,
):
    rr = tl.arange(0, BLOCK_R)
    rmask = rr < num_reqs
    sched = tl.load(packed_ptr + rr, mask=rmask, other=0)
    nd = tl.load(packed_ptr + num_reqs + rr, mask=rmask, other=0)
    cap = tl.where(mode == 1, sched, 0)
    if mode == 2:
        r2 = rr[:, None] + tl.zeros([BLOCK_R, BLOCK_S], dtype=tl.int32)
        s2 = tl.arange(0, BLOCK_S)[None, :] + tl.zeros([BLOCK_R, BLOCK_S], dtype=tl.int32)
        key_i, valid_i, surv_i = _dsv41_fav_keys(
            conf_ptr, conf_stride, idx_ptr, packed_ptr, num_reqs, r2, s2, S
        )
        if DEBUG:
            tl.store(dbg_ptr + r2 * BLOCK_S + s2, surv_i)
        fi = r2 * S + s2
        rank = tl.zeros([BLOCK_R, BLOCK_S], dtype=tl.int32)
        for j0 in range(0, num_reqs * BLOCK_S, CJ):
            jj = j0 + tl.arange(0, CJ)
            rj = jj // BLOCK_S
            sj = jj % BLOCK_S
            key_j, valid_j, _unused = _dsv41_fav_keys(
                conf_ptr, conf_stride, idx_ptr, packed_ptr, num_reqs, rj, sj, S
            )
            fj = rj * S + sj
            kj = key_j[None, None, :]
            ki = key_i[:, :, None]
            beats = valid_j[None, None, :] & (
                (kj > ki) | ((kj == ki) & (fj[None, None, :] < fi[:, :, None]))
            )
            rank += tl.sum(beats.to(tl.int32), axis=2)
        adm = valid_i & (rank < draft_budget)
        cap = tl.sum(adm.to(tl.int32), axis=1)
    cap = tl.where(rmask, cap, 0)
    tl.store(cap_ptr + rr, cap, mask=rmask)
    tl.store(nd_ptr + rr, nd, mask=rmask)
    cu = tl.cumsum(tl.where(rmask, cap + num_bonus, 0), axis=0)
    q = tl.cumsum(tl.where(rmask, cap + nd, 0), axis=0)
    tl.store(cu_ptr + 1 + rr, cu, mask=rmask)
    tl.store(qsl_ptr + 1 + rr, q, mask=rmask)
    qq = tl.arange(0, BLOCK_Q)
    zero = tl.zeros([BLOCK_Q], dtype=tl.int32)
    tl.store(cu_ptr + qq, zero, mask=qq == 0)
    tl.store(
        qsl_ptr + qq,
        tl.where(qq == 0, zero, zero + num_tokens),
        mask=(qq == 0) | ((qq > num_reqs) & (qq < qsl_len)),
    )


@triton.jit(do_not_specialize=["num_reqs", "qsl_len"])
def _dsv41_fused_av_cmp_kernel(
    a_cap, b_cap, a_nd, b_nd, a_cu, b_cu, a_q, b_q, cnt_ptr, num_reqs, qsl_len,
    BLOCK_Q: tl.constexpr,
):
    q = tl.arange(0, BLOCK_Q)
    m_r = q < num_reqs
    m_cu = q < num_reqs + 1
    m_q = q < qsl_len
    bad = tl.sum((tl.load(a_cap + q, mask=m_r, other=0)
                  != tl.load(b_cap + q, mask=m_r, other=0)).to(tl.int64), axis=0)
    bad += tl.sum((tl.load(a_nd + q, mask=m_r, other=0)
                   != tl.load(b_nd + q, mask=m_r, other=0)).to(tl.int64), axis=0)
    bad += tl.sum((tl.load(a_cu + q, mask=m_cu, other=0)
                   != tl.load(b_cu + q, mask=m_cu, other=0)).to(tl.int64), axis=0)
    bad += tl.sum((tl.load(a_q + q, mask=m_q, other=0)
                   != tl.load(b_q + q, mask=m_q, other=0)).to(tl.int64), axis=0)
    tl.store(cnt_ptr, tl.load(cnt_ptr) + bad)
    tl.store(cnt_ptr + 1, tl.load(cnt_ptr + 1) + 1)


def dsv41_fused_av_launch(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    packed_gpu: torch.Tensor,
    cap: torch.Tensor,
    nd: torch.Tensor,
    cu: torch.Tensor,
    qsl: torch.Tensor,
    num_reqs: int,
    draft_budget: int,
    mode: int,
    num_tokens: int,
    num_bonus: int,
    max_num_reqs: int,
    dbg: torch.Tensor | None = None,
) -> None:
    """One launch of the fused kernel (also the unit test's entry point)."""
    S = confidence_probs.shape[1]
    qsl_len = qsl.shape[0]
    _dsv41_fused_av_kernel[(1,)](
        confidence_probs,
        confidence_probs.stride(0),
        idx_mapping,
        packed_gpu,
        cap,
        nd,
        cu,
        qsl,
        dbg if dbg is not None else cap,
        num_reqs,
        qsl_len,
        draft_budget,
        mode,
        num_tokens,
        num_bonus,
        S=S,
        BLOCK_R=max(triton.next_power_of_2(max_num_reqs), 2),
        BLOCK_S=8,
        CJ=32,
        BLOCK_Q=max(triton.next_power_of_2(qsl_len), 2),
        DEBUG=dbg is not None,
        num_warps=4,
    )


def _dsv41_env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _dsv41_calibrate_curves(
    draft_curve: list[tuple[int, float]],
    verify_curve: list[tuple[int, float]],
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """DSV41 patch: optional override / rescale of the profiled cost curves.

    The startup profile feeds random token ids, which route to far more
    experts than real decode tokens, so verify cost grows ~2.5x too fast
    with width and the trimmer under-verifies (prose verifies ~1 draft).
    """
    raw = (list(draft_curve), list(verify_curve))
    path = os.environ.get("DSV41_AV_COST_JSON", "").strip()
    if path:
        with open(path) as f:
            pinned = json.load(f)
        if pinned.get("draft_ms"):
            draft_curve = [(int(n), float(ms)) for n, ms in pinned["draft_ms"]]
        if pinned.get("verify_ms"):
            verify_curve = [(int(n), float(ms)) for n, ms in pinned["verify_ms"]]
    a = _dsv41_env_float("DSV41_AV_VERIFY_SCALE", 1.0)
    n0 = _dsv41_env_float("DSV41_AV_SCALE_N0", 8.0)
    n1 = max(_dsv41_env_float("DSV41_AV_SCALE_N1", 48.0), n0 + 1.0)
    if a != 1.0 and verify_curve:
        base = min(verify_curve)[1]  # cost at the smallest profiled width
        out = []
        for n, ms in verify_curve:
            if n <= n0:
                s = a
            elif n >= n1:
                s = 1.0
            else:
                s = a + (1.0 - a) * (n - n0) / (n1 - n0)
            out.append((n, base + s * (ms - base)))
        verify_curve = out
    fixed = _dsv41_env_float("DSV41_AV_FIXED_MS", 0.0)
    if fixed:
        draft_curve = [(n, ms + fixed) for n, ms in draft_curve]
    if path or a != 1.0 or fixed:
        logger.info(
            "DSV41 adaptive-verification calibration (json=%s scale=%s n0=%s "
            "n1=%s fixed_ms=%s): draft_ms=%s verify_ms=%s",
            path or None, a, n0, n1, fixed, draft_curve, verify_curve,
        )
    dump = os.environ.get("DSV41_AV_DUMP_JSON", "").strip()
    if dump:
        try:
            if get_tp_group().rank_in_group == 0:
                with open(dump, "w") as f:
                    json.dump(
                        {
                            "raw_draft_ms": raw[0],
                            "raw_verify_ms": raw[1],
                            "draft_ms": draft_curve,
                            "verify_ms": verify_curve,
                        },
                        f,
                    )
        except Exception as e:  # diagnostics only
            logger.warning("DSV41_AV_DUMP_JSON failed: %s", e)
    return draft_curve, verify_curve

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.utils import AttentionGroup


def prepare_cost_profile_inputs(
    input_batch: "InputBatch", vocab_size: int, seed: int
) -> None:
    """Exercise MoE routing while retaining dummy cache safety and graph mode.

    Use an isolated CPU generator so every TP rank receives identical IDs,
    independent of rank-local RNG consumption. Vary seeds across repetitions
    to avoid repeatedly timing one unusually narrow set of experts. These
    synthetic tokens are a routing probe, not a representative prompt corpus.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(
        vocab_size,
        (input_batch.num_tokens,),
        generator=generator,
        dtype=input_batch.input_ids.dtype,
        device="cpu",
    )
    input_batch.input_ids.copy_(ids)
    input_batch.is_padding.fill_(False)


def resolve_adaptive_cudagraph_mode(
    mode: CUDAGraphMode, *, piecewise_capture_available: bool
) -> CUDAGraphMode:
    """Select a separate decode route for variable-length verification."""
    assert mode.has_full_cudagraphs()
    if mode == CUDAGraphMode.FULL_DECODE_ONLY or not piecewise_capture_available:
        return CUDAGraphMode.FULL_DECODE_ONLY
    return CUDAGraphMode.FULL_AND_PIECEWISE


def _assign_draft_token_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
    num_steps: int,
) -> None:
    """Admit the globally best `draft_budget` draft slots, in place.

    Every (request, step) slot is scored by its survival probability, the running
    product of that request's per-position confidences, and the highest scores win.
    Survival only decreases along a request, so a global top-k always admits
    continuously along steps with a request.

    `capacities` enters holding each request's scheduled draft count (which bounds its
    eligible slots) and leaves holding the admitted count. The caller only calls this
    when `draft_budget < capacities.sum()`, so every winner is a real draft slot.
    """
    survival = confidence_probs[idx_mapping].cumprod(dim=1)
    steps = torch.arange(num_steps, device=survival.device)
    # Out-of-range slots score -inf so they never outrank a real draft.
    out_of_range = steps[None, :] >= capacities[:, None]
    survival = survival.masked_fill(out_of_range, -float("inf"))
    flat = survival.flatten()
    winners = flat.topk(draft_budget).indices
    admitted = torch.zeros_like(flat, dtype=torch.bool).index_fill_(0, winners, True)
    torch.sum(admitted.view_as(survival), dim=1, dtype=capacities.dtype, out=capacities)


_assign_draft_token_budget_compiled = torch.compile(
    _assign_draft_token_budget, dynamic=True
)


def build_cost_tables_from_curves(
    draft_curve: list[tuple[int, float]],
    verify_curve: list[tuple[int, float]],
    max_num_reqs: int,
    max_batch_tokens: int,
    cudagraph_limit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cost tables: graph-padded below the capture limit, smooth above.

    Args:
        draft_curve: (size, cost) samples for the draft model.
        verify_curve: (size, cost) samples for the verify model.
        max_num_reqs: Largest request count to build a table for.
        max_batch_tokens: Largest token count to build a table for.
        cudagraph_limit: Largest cudagraph-captured size. At or below it,
            execution pads up to the next captured size, so cost is a step
            function. Above it there is no padding, so cost is continuous.

    """

    def build_table(limit: int, curve: list[tuple[int, float]]) -> np.ndarray:
        xs, ys = np.asarray(curve, dtype=np.float64).T
        ys = np.maximum.accumulate(ys)
        values = np.arange(limit + 1)
        # Execution pads to the next captured size, so cost is a step
        # function of the padded size: smooth interpolation would invent
        # marginal per-token costs that don't exist within a pad bucket.
        idx = np.searchsorted(xs, values, side="left")
        result = ys[np.minimum(idx, len(xs) - 1)]
        # Past the capture limit nothing pads, so cost really is continuous in
        # size; snapping to the next profiled point overestimates badly when
        # the profiled points are far apart. Interpolate only between points
        # that are themselves past the limit: crossing the limit loses
        # cudagraphs entirely, which is a genuine discontinuity, so the first
        # point above it must not be blended with the last one below.
        if cudagraph_limit:
            smooth = values > cudagraph_limit
            above = xs > cudagraph_limit
            if smooth.any() and above.any():
                result[smooth] = np.interp(values[smooth], xs[above], ys[above])
        if len(xs) > 1:
            after = values > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            result[after] = ys[-1] + slope * (values[after] - xs[-1])
        return result

    draft_table = np.maximum(build_table(max_num_reqs, draft_curve), 0.0)
    verify_table = np.maximum(build_table(max_batch_tokens, verify_curve), 1e-6)
    return draft_table, verify_table


class AdaptiveVerificationManager:
    def __init__(
        self,
        req_states: "RequestState",
        query_start_loc: torch.Tensor,
        num_bonus_tokens: int,
        max_total_logits: int,
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.num_bonus_tokens = num_bonus_tokens
        # Rejection sampling verifies logits in one contiguous chunk; the
        # chunked path indexes by scheduled (untrimmed) offsets and cannot
        # address the compacted layout, so the budget must fit one chunk.
        self._max_total_logits = max_total_logits
        self.query_start_loc = query_start_loc
        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        # Largest cudagraph-captured token count; above it nothing pads.
        self._cudagraph_limit = 0
        self._batch_budget: tuple[dict[str, int], dict[str, int], int] | None = None
        max_num_reqs = req_states.max_num_reqs
        # Current per-slot confidences
        self._confidence_probs = torch.empty(
            (max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._batch_draft_capacity = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._num_non_draft_tokens = torch.empty_like(query_start_loc[:-1])
        self._cu_num_logits = torch.empty_like(query_start_loc)

        # Two D2H slots preserve stale inputs for budget selection.
        self._stale_confidences = [
            CpuGpuBuffer(
                max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._pending_resets: list[int] = []
        self._stale_idx = 0
        # DSV41 r4 verify-router (DSV41_VERIFY_CAP); inert when unset.
        (
            self.dsv41_vcap_enabled,
            self._dsv41_vcap_auto,
            self._dsv41_vcap_value,
        ) = _dsv41_parse_vcap()
        self._dsv41_vcap_kmin = int(os.environ.get("DSV41_VERIFY_CAP_MIN", "1"))
        self._dsv41_vcap_tau_max = float(
            os.environ.get("DSV41_VERIFY_CAP_TAU_MAX", "0.5")
        )
        self._dsv41_vcap_torch = os.environ.get("DSV41_VERIFY_CAP_TORCH", "") == "1"
        self._dsv41_vcap_log = int(os.environ.get("DSV41_VERIFY_CAP_LOG", "0") or 0)
        self._dsv41_intra_bucket = float(
            os.environ.get("DSV41_AV_INTRA_BUCKET", "0") or 0.0
        )
        self._dsv41_tau = self._dsv41_vcap_value
        self._dsv41_slope: np.ndarray | None = None
        self._dsv41_vcap_steps = 0
        self._dsv41_vcap_tau_sum = 0.0
        self._dsv41_vcap_rows = 0
        if self.dsv41_vcap_enabled:
            if not envs.VLLM_MOE_SKIP_PADDING:
                raise ValueError(
                    "DSV41_VERIFY_CAP needs VLLM_MOE_SKIP_PADDING=1 (the padding "
                    "mask is refilled every step)."
                )
            self._dsv41_dead_buf = torch.zeros(
                max(1, max_total_logits), dtype=torch.bool, device=device
            )
            self._dsv41_live_buf = torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            )
            self._dsv41_dead_count = torch.zeros(
                (), dtype=torch.int64, device=device
            )
            logger.info(
                "DSV41 verify-cap: %s (kmin=%d tau_max=%s torch=%s); dead verify "
                "rows are padding for the router and invalid for acceptance.",
                ("auto x%s" % self._dsv41_vcap_value)
                if self._dsv41_vcap_auto
                else ("conf:%s" % self._dsv41_vcap_value),
                self._dsv41_vcap_kmin,
                self._dsv41_vcap_tau_max,
                self._dsv41_vcap_torch,
            )
        self._dsv41_log_every = int(os.environ.get("DSV41_AV_LOG_BUDGET", "0") or 0)
        self._dsv41_budget_hist: Counter = Counter()
        self._dsv41_budget_calls = 0
        # DSV41 g4d item 09 (DSV41_FUSED_AV): fused reallocate_drafts; inert when unset.
        self._dsv41_fav = _dsv41_fused_av_mode()
        if self._dsv41_fav and self.num_speculative_steps > 8:
            logger.warning(
                "DSV41_FUSED_AV: %d speculative steps > 8 is not supported; stock path",
                self.num_speculative_steps,
            )
            self._dsv41_fav = 0
        self.dsv41_fused_av_on = self._dsv41_fav == 1
        if self._dsv41_fav:
            self._dsv41_fav_packed = torch.zeros(
                2 * max_num_reqs, dtype=torch.int32, device=device
            )
            self._dsv41_fav_log = int(os.environ.get("DSV41_FUSED_AV_LOG", "2000") or 2000)
            self._dsv41_fav_steps = 0
            if self._dsv41_fav == 2:
                self._dsv41_fav_cap = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
                self._dsv41_fav_nd = torch.zeros_like(self._num_non_draft_tokens)
                self._dsv41_fav_cu = torch.zeros_like(self._cu_num_logits)
                self._dsv41_fav_qsl = torch.zeros_like(query_start_loc)
                self._dsv41_fav_cnt = torch.zeros(2, dtype=torch.int64, device=device)
            self._dsv41_fav_warmup(max_num_reqs, device)
            logger.info(
                "DSV41_FUSED_AV=%s: reallocate_drafts runs as one packed H2D + one "
                "Triton launch%s (S=%d, max_num_reqs=%d).",
                "1" if self._dsv41_fav == 1 else "verify",
                "" if self._dsv41_fav == 1 else " in shadow, stock path served",
                self.num_speculative_steps,
                max_num_reqs,
            )
        for slot in self._stale_confidences:
            slot.np.fill(1.0)

    def add_request(self, req_idx: int) -> None:
        self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
        self._pending_resets.append(req_idx)
        self._confidence_probs[req_idx].fill_(1.0)

    def batches_to_profile(self, capture_sizes: list[int]) -> Iterator[dict[str, int]]:
        """Dummy-run kwargs whose step timings seed the cost tables.

        Run these inside StepTimingCollector.collect(), then hand the block's
        timings to set_initial_cost_curves."""
        max_num_tokens = self.req_states.max_num_batched_tokens
        size = self._cudagraph_limit = capture_sizes[-1] if capture_sizes else 0
        # Also profile beyond the capture limit: real steps run there
        # (piecewise/eager) and linear extrapolation badly underestimates
        # them. These runs double as JIT warmup for the piecewise shapes.
        tail_sizes: set[int] = set()
        if size:
            tail_sizes.add(min(size + size // 2, max_num_tokens))
            while size < max_num_tokens:
                size = min(size * 2, max_num_tokens)
                tail_sizes.add(size)
            tail_sizes -= set(capture_sizes)
        for num_tokens in capture_sizes + sorted(tail_sizes):
            for replay in range(_PROFILE_REPLAYS):
                yield {
                    "num_tokens": num_tokens,
                    "context_len": envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN,
                    "cost_profile_seed": num_tokens * _PROFILE_REPLAYS + replay,
                }

    def set_initial_cost_curves(self, samples: list[StepTimingSample]) -> None:
        def median_curve(
            points: Iterable[tuple[int, float]],
        ) -> list[tuple[int, float]]:
            grouped: defaultdict[int, list[float]] = defaultdict(list)
            for key, value in points:
                grouped[key].append(value)
            return [(k, float(np.median(v))) for k, v in sorted(grouped.items())]

        # Draft curve: eager-target steps inflate drafter timings (the CPU is
        # still launching kernels, opening gaps between the drafter's events),
        # and request counts — unlike token counts — collide across execution
        # modes, so only graph-replay samples may price the draft curve.
        draft_curve = median_curve(
            (s.num_reqs, s.drafter_ms) for s in samples if s.full_cudagraph
        )
        verify_curve = median_curve(
            (s.num_target_tokens, s.forward_ms) for s in samples
        )
        logger.info(
            "Adaptive verification cost profile: draft_ms=%s verify_ms=%s "
            "full_graph_samples=%d total_samples=%d",
            draft_curve,
            verify_curve,
            sum(s.full_cudagraph for s in samples),
            len(samples),
        )
        self.set_cost_curves(draft_curve, verify_curve)

    def set_cost_curves(
        self,
        draft_curve: list[tuple[int, float]],
        verify_curve: list[tuple[int, float]],
    ) -> None:
        draft_curve, verify_curve = _dsv41_calibrate_curves(draft_curve, verify_curve)
        draft_curve, verify_curve = get_tp_group().broadcast_object(
            (draft_curve, verify_curve), src=0
        )
        if not draft_curve or not verify_curve:
            raise RuntimeError(
                "Adaptive verification could not profile step costs. Pass "
                "`enable_adaptive_verification=false` in the speculative config to "
                "verify a fixed number of drafts instead."
            )
        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
            self._cudagraph_limit,
        )
        self._dsv41_post_cost_tables(verify_curve)
        logger.debug("DSpark cost tables: %s", self.cost_tables)

    def _dsv41_post_cost_tables(self, verify_curve: list[tuple[int, float]]) -> None:
        """DSV41 r4: per-row verify slope (auto tau) and intra-bucket pricing."""
        if not (self.dsv41_vcap_enabled or self._dsv41_intra_bucket > 0):
            return
        assert self.cost_tables is not None
        xs, ys = np.asarray(verify_curve, dtype=np.float64).T
        ys = np.maximum.accumulate(ys)
        n = len(self.cost_tables[1])
        values = np.arange(n)
        if len(xs) > 1:
            i = np.clip(np.searchsorted(xs, values, side="left"), 1, len(xs) - 1)
            slope = (ys[i] - ys[i - 1]) / np.maximum(xs[i] - xs[i - 1], 1.0)
        else:
            slope = np.zeros(n)
        self._dsv41_slope = np.maximum(slope, 0.0)
        frac = min(max(self._dsv41_intra_bucket, 0.0), 1.0)
        if frac > 0 and len(xs) > 1:
            verify_table = self.cost_tables[1]
            limit = self._cudagraph_limit or int(xs[-1])
            for t in range(1, min(limit, n - 1) + 1):
                k = int(np.searchsorted(xs, t, side="left"))
                if k <= 0 or k >= len(xs) or xs[k] == t or xs[k] > limit:
                    continue
                hi, lo = xs[k], xs[k - 1]
                verify_table[t] = max(
                    ys[k] - frac * (ys[k] - ys[k - 1]) * (hi - t) / (hi - lo), 1e-6
                )
            logger.info(
                "DSV41 AV intra-bucket pricing f=%.2f: verify_ms[1..16]=%s",
                frac,
                np.round(verify_table[1:17], 2).tolist(),
            )

    def record_confidences(
        self, confidence_probs: torch.Tensor, input_batch: "InputBatch"
    ) -> None:
        """Publish this step's raw confidences for the ranking kernel and start
        copying them to the CPU, where a later step's budget reads them."""
        num_reqs = input_batch.num_reqs
        ready_idx = self._stale_idx ^ 1
        with gpu_sync_allowed():
            self._copy_events[ready_idx].synchronize()
        if self._pending_resets:
            self._stale_confidences[ready_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        # Last step's copy has landed: budgets read it, this step overwrites the
        # slot they were reading before.
        self._stale_idx, write_idx = ready_idx, self._stale_idx

        self._confidence_probs[input_batch.idx_mapping] = confidence_probs[:num_reqs]
        write_slot = self._stale_confidences[write_idx]
        write_slot.gpu.copy_(self._confidence_probs)

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            self._copy_events[write_idx].record()

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
    ) -> int:
        """Token count once the draft budget is trimmed to fit.

        Stashes the chosen budget in ``_batch_budget`` for the compaction and
        reallocation that follow in the same step.
        """
        assert self.cost_tables is not None
        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        scheduled_tokens = np.fromiter(
            num_tokens_per_req.values(), dtype=np.int32, count=num_reqs
        )
        scheduled_drafts = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = scheduled_tokens - scheduled_drafts
        slots = np.fromiter(
            map(self.req_states.req_id_to_index.__getitem__, req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        stale_confidences = self._stale_confidences[self._stale_idx].np[slots]
        survival_probability = np.cumprod(stale_confidences.astype(np.float64), axis=1)
        steps = np.arange(self.num_speculative_steps)
        valid = steps[None, :] < scheduled_drafts[:, None]
        scores = np.sort(survival_probability[valid])[::-1]
        num_non_draft_tokens_total = int(num_non_draft_tokens.sum())
        max_draft_budget = min(
            int(scheduled_drafts.sum()),
            max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
        )
        scores = scores[:max_draft_budget]
        draft_cost_ms, verify_cost_ms = self.cost_tables
        num_sampling_requests = np.count_nonzero(
            self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens
            >= self.req_states.prefill_len.np[slots]
        )
        num_tokens_to_estimated_accepted_tokens = np.concatenate(
            ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
        )
        costs = (
            draft_cost_ms[len(req_ids)]
            + verify_cost_ms[
                num_non_draft_tokens_total : num_non_draft_tokens_total
                + max_draft_budget
                + 1
            ]
        )
        num_drafts_per_req = {
            req_id: int(num_drafts)
            for req_id, num_drafts in zip(req_ids, scheduled_drafts, strict=True)
        }
        num_non_draft_tokens_per_req = {
            req_id: int(num_tokens)
            for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
        }
        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
        if self._dsv41_vcap_auto and self._dsv41_slope is not None:
            # Break-even survival for one more verify row: its marginal cost in
            # ms times the step's expected tokens per ms.
            width = min(
                num_non_draft_tokens_total + draft_budget, len(self._dsv41_slope) - 1
            )
            tau = (
                self._dsv41_vcap_value
                * float(self._dsv41_slope[width])
                * float(num_tokens_to_estimated_accepted_tokens[draft_budget])
                / max(float(costs[draft_budget]), 1e-6)
            )
            self._dsv41_tau = min(max(tau, 0.0), self._dsv41_vcap_tau_max)
        if self._dsv41_log_every > 0:
            self._dsv41_budget_hist[(num_reqs, draft_budget)] += 1
            self._dsv41_budget_calls += 1
            if self._dsv41_budget_calls % self._dsv41_log_every == 0:
                if get_tp_group().rank_in_group == 0:
                    logger.info(
                        "DSV41 AV budget histogram (num_reqs, drafts): %s",
                        sorted(self._dsv41_budget_hist.items()),
                    )
                self._dsv41_budget_hist.clear()
        self._batch_budget = (
            num_drafts_per_req,
            num_non_draft_tokens_per_req,
            draft_budget,
        )
        return sum(num_non_draft_tokens_per_req.values()) + draft_budget

    def compact_batch(
        self,
        num_draft_tokens_per_req: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        cu_num_logits_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compact the CPU batch to the chosen draft budget.

        Returns the compacted per-request token counts and the CPU cu_num_logits_np.
        If the draft budget is 0, we can know cu_num_logits_np exactly, otherwise
        its unchanged/an-upper-bound.
        """
        batch_budget = self._batch_budget
        assert batch_budget is not None
        _, _, draft_budget = batch_budget
        num_drafts = int(num_draft_tokens_per_req.sum())
        if draft_budget == num_drafts:
            return num_scheduled_tokens, cu_num_logits_np

        num_non_draft_tokens = num_scheduled_tokens - num_draft_tokens_per_req
        if draft_budget == 0:
            # The draft budget is 0, so we can know cu_num_logits_np exactly. This helps
            # when we would exceed the sampler logit chunk size.
            num_reqs = num_scheduled_tokens.shape[0]
            cu_num_logits_np = (
                np.arange(num_reqs + 1, dtype=cu_num_logits_np.dtype)
                * self.num_bonus_tokens
            )
            return num_non_draft_tokens, cu_num_logits_np

        is_verification_request = num_draft_tokens_per_req > 0
        num_verification_reqs = int(is_verification_request.sum())
        # sort_batch_req_ids keeps verification requests at the front.
        assert np.all(is_verification_request[:num_verification_reqs])
        # for the CPU side buffer we distribute draft tokens evenly
        draft_lens_cpu = np.zeros_like(num_non_draft_tokens)
        draft_lens_cpu[:num_verification_reqs] = draft_budget // num_verification_reqs
        draft_lens_cpu[: draft_budget % num_verification_reqs] += 1
        return num_non_draft_tokens + draft_lens_cpu, cu_num_logits_np

    def _dsv41_fav_warmup(self, max_num_reqs: int, device) -> None:
        """JIT-compile the fused (and compare) kernels once, at startup."""
        S = self.num_speculative_steps
        conf = torch.full((max_num_reqs, S), 0.5, dtype=torch.float32, device=device)
        idx = torch.zeros(1, dtype=torch.int64, device=device)
        packed = torch.tensor([S, 1], dtype=torch.int32, device=device)
        cap = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        nd = torch.zeros_like(cap)
        cu = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        qsl = torch.zeros_like(cu)
        dsv41_fused_av_launch(
            conf, idx, packed, cap, nd, cu, qsl, 1, 1, 2, 2, self.num_bonus_tokens,
            max_num_reqs,
        )
        if self._dsv41_fav == 2:
            cnt = torch.zeros(2, dtype=torch.int64, device=device)
            _dsv41_fused_av_cmp_kernel[(1,)](
                cap, cap, nd, nd, cu, cu, qsl, qsl, cnt, 1, qsl.shape[0],
                BLOCK_Q=max(triton.next_power_of_2(qsl.shape[0]), 2), num_warps=1,
            )
        torch.cuda.synchronize(device)

    def _dsv41_fav_run(
        self,
        idx_mapping: torch.Tensor,
        num_reqs: int,
        scheduled_drafts: np.ndarray,
        num_non_draft_tokens: np.ndarray,
        draft_budget: int,
        num_tokens: int,
    ) -> None:
        total = int(scheduled_drafts.sum())
        mode = 0 if draft_budget == 0 else (1 if draft_budget >= total else 2)
        packed = np.empty(2 * num_reqs, dtype=np.int32)
        packed[:num_reqs] = scheduled_drafts
        packed[num_reqs:] = num_non_draft_tokens
        packed_gpu = self._dsv41_fav_packed[: 2 * num_reqs]
        async_tensor_h2d(packed, out=packed_gpu)
        if self._dsv41_fav == 1:
            outs = (
                self._batch_draft_capacity,
                self._num_non_draft_tokens,
                self._cu_num_logits,
                self.query_start_loc,
            )
        else:
            outs = (
                self._dsv41_fav_cap,
                self._dsv41_fav_nd,
                self._dsv41_fav_cu,
                self._dsv41_fav_qsl,
            )
        dsv41_fused_av_launch(
            self._confidence_probs,
            idx_mapping,
            packed_gpu,
            *outs,
            num_reqs,
            draft_budget,
            mode,
            num_tokens,
            self.num_bonus_tokens,
            self.req_states.max_num_reqs,
        )

    def _dsv41_fav_check(self, num_reqs: int) -> None:
        qsl_len = self.query_start_loc.shape[0]
        _dsv41_fused_av_cmp_kernel[(1,)](
            self._dsv41_fav_cap,
            self._batch_draft_capacity,
            self._dsv41_fav_nd,
            self._num_non_draft_tokens,
            self._dsv41_fav_cu,
            self._cu_num_logits,
            self._dsv41_fav_qsl,
            self.query_start_loc,
            self._dsv41_fav_cnt,
            num_reqs,
            qsl_len,
            BLOCK_Q=max(triton.next_power_of_2(qsl_len), 2),
            num_warps=1,
        )
        self._dsv41_fav_steps += 1
        if self._dsv41_fav_steps % self._dsv41_fav_log == 0:
            bad, calls = self._dsv41_fav_cnt.tolist()  # one sync per N steps
            if get_tp_group().rank_in_group == 0:
                logger.info(
                    "DSV41_FUSED_AV verify: %d calls, %d mismatching entries "
                    "(capacities / non-draft / cu_num_logits / query_start_loc)",
                    calls,
                    bad,
                )

    def reallocate_drafts(
        self, req_ids: list[str], idx_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch_budget, self._batch_budget = self._batch_budget, None
        assert batch_budget is not None
        num_drafts_per_req, num_non_draft_tokens_per_req, draft_budget = batch_budget
        num_reqs = idx_mapping.shape[0]
        scheduled_drafts = np.fromiter(
            (num_drafts_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = np.fromiter(
            (num_non_draft_tokens_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_tokens = int(num_non_draft_tokens.sum()) + draft_budget
        if self._dsv41_fav:
            # DSV41 g4d 09: fused path (=1 serves it; =verify runs it in shadow).
            self._dsv41_fav_run(
                idx_mapping,
                num_reqs,
                scheduled_drafts,
                num_non_draft_tokens,
                draft_budget,
                num_tokens,
            )
            if self._dsv41_fav == 1:
                return (
                    self._cu_num_logits[: num_reqs + 1],
                    self.query_start_loc,
                    draft_budget,
                )

        # Rank draft slots by survival probability and admit the best prefix.
        # capacities enters holding each request's valid draft count (the kernel
        # uses it to bound eligible slots) and leaves holding the admitted count.
        capacities = self._batch_draft_capacity[:num_reqs]
        if draft_budget == 0:
            capacities.zero_()
        else:
            async_tensor_h2d(scheduled_drafts, out=capacities)
            if draft_budget < int(scheduled_drafts.sum()):
                _assign_draft_token_budget_compiled(
                    self._confidence_probs,
                    idx_mapping,
                    capacities,
                    draft_budget,
                    self.num_speculative_steps,
                )

        num_non_draft_tokens_gpu = self._num_non_draft_tokens[:num_reqs]
        async_tensor_h2d(num_non_draft_tokens, out=num_non_draft_tokens_gpu)
        self._cu_num_logits[:1].zero_()
        torch.cumsum(
            capacities + self.num_bonus_tokens,
            dim=0,
            out=self._cu_num_logits[1 : num_reqs + 1],
        )
        self.query_start_loc[:1].zero_()
        torch.cumsum(
            capacities + num_non_draft_tokens_gpu,
            dim=0,
            out=self.query_start_loc[1 : num_reqs + 1],
        )
        self.query_start_loc[num_reqs + 1 :].fill_(num_tokens)
        if self._dsv41_fav == 2:
            self._dsv41_fav_check(num_reqs)
        return (
            self._cu_num_logits[: num_reqs + 1],
            self.query_start_loc,
            draft_budget,
        )

    def dsv41_apply_live_cap(self, input_batch: "InputBatch") -> None:
        """DSV41 r4 verify-cap: mark dead verify rows (see module comment).

        Runs after reallocate_drafts and combine_sampled_and_draft_tokens, before
        the forward. Writes input_batch.is_padding for the dead token rows (the
        runner refills the mask every step) and attaches the per-logit-row dead
        mask as input_batch.dsv41_dead_drafts for the rejection sampler.
        """
        if not self.dsv41_vcap_enabled:
            return
        num_reqs = input_batch.num_reqs
        num_logits = int(input_batch.logits_indices.shape[0])
        if num_reqs == 0 or num_logits == 0:
            return
        tau = float(self._dsv41_tau)
        capacities = self._batch_draft_capacity[:num_reqs]
        dead = self._dsv41_dead_buf[:num_logits]
        live = self._dsv41_live_buf[:num_reqs]
        if self._dsv41_vcap_torch:
            d, lv = dsv41_live_cap_torch(
                self._confidence_probs,
                input_batch.idx_mapping,
                capacities,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                input_batch.logits_indices,
                input_batch.is_padding,
                tau,
                self._dsv41_vcap_kmin,
            )
            dead.copy_(d)
            live.copy_(lv)
        else:
            S = self.num_speculative_steps
            _dsv41_live_cap_kernel[(num_reqs,)](
                self._confidence_probs,
                self._confidence_probs.stride(0),
                input_batch.idx_mapping,
                capacities,
                input_batch.cu_num_logits,
                input_batch.logits_indices,
                input_batch.is_padding.view(torch.uint8),
                dead.view(torch.uint8),
                live,
                tau,
                self._dsv41_vcap_kmin,
                S=S,
                BLOCK_S=triton.next_power_of_2(S),
                BLOCK_J=triton.next_power_of_2(S + 1),
                num_warps=1,
            )
        input_batch.dsv41_dead_drafts = dead
        if self._dsv41_vcap_log > 0:
            self._dsv41_dead_count += dead.sum()
            self._dsv41_vcap_steps += 1
            self._dsv41_vcap_tau_sum += tau
            self._dsv41_vcap_rows += num_logits - num_reqs
            if self._dsv41_vcap_steps % self._dsv41_vcap_log == 0:
                n_dead = int(self._dsv41_dead_count.item())  # one sync per N steps
                if get_tp_group().rank_in_group == 0:
                    logger.info(
                        "DSV41 verify-cap: %d steps, mean tau %.3f, dead %d of %d "
                        "verified drafts (%.1f%%)",
                        self._dsv41_vcap_log,
                        self._dsv41_vcap_tau_sum / self._dsv41_vcap_log,
                        n_dead,
                        self._dsv41_vcap_rows,
                        100.0 * n_dead / max(self._dsv41_vcap_rows, 1),
                    )
                self._dsv41_dead_count.zero_()
                self._dsv41_vcap_tau_sum = 0.0
                self._dsv41_vcap_rows = 0


def maybe_create_adaptive_verification_manager(
    *,
    enable_adaptive_verification: bool,
    attn_groups: list[list["AttentionGroup"]],
    req_states: "RequestState",
    query_start_loc: torch.Tensor,
    num_bonus_tokens: int,
    max_total_logits: int,
    vllm_config: "VllmConfig",
    target_layer_names: set[str] | None = None,
    additional_attn_cg_support: tuple[AttentionCGSupport, str | None] | None = None,
) -> AdaptiveVerificationManager | None:
    if not enable_adaptive_verification:
        return None

    # The selector rejects unsupported backends, but models that
    # hard-wire theirs (e.g. DeepSeek-V4) never go through it.
    backend = get_query_lens_mismatch_unsupported_backend(
        attn_groups, checked_layer_names=target_layer_names
    )
    if backend is not None:
        raise ValueError(
            "Adaptive verification trims verification requests on device, which"
            f" the {backend} attention backend does not support. Pass "
            "enable_adaptive_verification=false in the speculative config, or "
            "use a backend that does."
        )

    # The runner's decode_query_len, the width varlen decode graphs capture.
    max_query_len = req_states.num_speculative_steps + num_bonus_tokens
    unsupported: tuple[str | None, int | None] | None = (
        get_varlen_cudagraph_unsupported_backend(
            attn_groups,
            vllm_config,
            max_query_len,
            checked_layer_names=target_layer_names,
        )
    )
    if unsupported is None and additional_attn_cg_support is not None:
        # Groups built outside init_attn_backend report only their support
        # level, and without a bound only ALWAYS replays varlen batches.
        additional_support, additional_backend = additional_attn_cg_support
        if additional_support != AttentionCGSupport.ALWAYS:
            unsupported = additional_backend, None
    if unsupported is not None:
        backend, bound = unsupported
        allowed = "none" if bound is None else f"at most {bound}"
        raise ValueError(
            "Adaptive verification replays decode cudagraphs whose per-request "
            f"query lengths vary up to {max_query_len}, but {backend} allows "
            f"{allowed}. Pass enable_adaptive_verification=false in the "
            "speculative config, or use a backend that does."
        )

    return AdaptiveVerificationManager(
        req_states,
        query_start_loc,
        num_bonus_tokens,
        max_total_logits=max_total_logits,
    )
