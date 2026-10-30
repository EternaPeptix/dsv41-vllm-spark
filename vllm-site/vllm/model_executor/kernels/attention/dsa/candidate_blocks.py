# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm.triton_utils import tl, triton

# dsv41 k1/1-indexer-mask (flag-gated; both default off = stock behaviour).
#   DSV41_CAND_LENAWARE=1  decode only (row_ks is None): the candidate flags and
#       the -inf mask stop at each row's end instead of the max_model_len-wide
#       logits row. Columns >= end are never read by any decode top-k backend
#       (the candidate-source layer already hands them unmasked to top-k), so
#       this is bitwise exact on everything that is read.
#   DSV41_CAND_LENSELECT=1 decode only: layer-20 block select = block scores
#       bounded to cdiv(end, block) + persistent_topk(k=2048, device lengths)
#       instead of torch.topk over all nblocks. Same candidate SET when a row
#       has <= k valid blocks (ctx <= 16384 compressed positions); above that
#       it can differ only on fp32 ties / NaN at the k-th block (greedy-eq).
#       Candidate order is not consumed (flags are order-insensitive).
#   DSV41_CAND_CTAS (default 384): target CTAs for the grid-stride kernels.
_DSV41_CAND_LENAWARE = os.environ.get("DSV41_CAND_LENAWARE", "0") == "1"
_DSV41_CAND_LENSELECT = os.environ.get("DSV41_CAND_LENSELECT", "0") == "1"
_DSV41_CAND_CTAS = max(1, int(os.environ.get("DSV41_CAND_CTAS", "384") or "384"))
_RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
_k1_ws_cache: dict = {}
if _DSV41_CAND_LENAWARE or _DSV41_CAND_LENSELECT:
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "dsv41 k1 candidate_blocks armed: LENAWARE=%d LENSELECT=%d CTAS=%d",
        _DSV41_CAND_LENAWARE,
        _DSV41_CAND_LENSELECT,
        _DSV41_CAND_CTAS,
    )


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        (blocks[:, None] < nblocks)
        & (offsets[None, :] < BLOCK_SIZE)
        & (cols < end)
        & (cols < width),
        other=-float("inf"),
    )
    reduced = tl.reduce(values, 1, _max_with_nan)
    reduced = tl.where(
        (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE),
        float("inf"),
        reduced,
    )
    tl.store(scores + row * nblocks + blocks, reduced, blocks < nblocks)


@triton.jit(do_not_specialize=["k"])
def _store_candidates_kernel(
    values,
    indices,
    output,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
    index = tl.load(indices + row * k + cols, cols < k, other=-1)
    # NaN scores can occur during warmup; only -inf denotes padding.
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(value != -float("inf"), index, -1),
        cols < OUT_K,
    )


@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(
    candidates,
    starts,
    flags,
    stride_row,
    stride_col,
    stride_start,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(nblocks + 1, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots <= nblocks)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    ).to(tl.int64)
    # Preserve the packed-column clamp for candidates beyond the logits width.
    block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    tl.debug_barrier()
    tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))


@triton.jit(do_not_specialize=["width", "nblocks"])
def _mask_candidates_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    tl.store(
        logits + row * stride_row + cols * stride_col,
        -float("inf"),
        (cols < width) & ~(valid & keep),
    )


# ---------------------------------------------------------------------------
# dsv41 k1: length-bounded variants (grid-stride over a static grid, so the
# launch shape depends only on (rows, width) and stays CUDA-graph safe; the
# per-row trip count comes from the device-side end).
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_bounded_kernel(
    candidates,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    # The mask reads flags only for blocks of columns in [start, min(end, width))
    # plus the edge slot, so only those slots need zeroing.
    span = tl.maximum(tl.minimum(end, width) - start, 0)
    nread = (span + BLOCK_SIZE - 1) // BLOCK_SIZE
    offsets = tl.arange(0, 1024)
    for tile in range(0, (nread + 1023) // 1024):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots < nread)
    tl.store(flags + row * (nblocks + 1) + nblocks, 0)
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    ).to(tl.int64)
    block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    tl.debug_barrier()
    tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))


@triton.jit(do_not_specialize=["width", "nblocks", "NPROG"])
def _mask_candidates_bounded_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    NPROG,
):
    row = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    lim = tl.maximum(tl.minimum(end, width), 0)
    ntiles = (lim + TILE - 1) // TILE
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    for t in range(pid, ntiles, NPROG):
        cols = t * TILE + tl.arange(0, TILE)
        valid = (cols >= start) & (cols < end) & (cols < width)
        block = (cols - start) // BLOCK_SIZE
        keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
        keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
        # Same values as the stock kernel on cols < min(end, width); columns
        # at or past the row end are left untouched (never read downstream).
        tl.store(
            logits + row * stride_row + cols * stride_col,
            -float("inf"),
            (cols < lim) & ~(valid & keep),
        )


@triton.jit(do_not_specialize=["width", "nblocks", "NPROG"])
def _block_scores_bounded_kernel(
    logits,
    starts,
    ends,
    scores,
    lens,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    NPROG,
):
    row = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    # Blocks the stock kernel can score above -inf: those holding a column in
    # [start, end) (incl. the pinned newest block), capped at nblocks.
    nvalid = tl.minimum((tl.maximum(end - start, 0) + BLOCK_SIZE - 1) // BLOCK_SIZE,
                        nblocks)
    if pid == 0:
        tl.store(lens + row, nvalid.to(tl.int32))
    ntiles = (nvalid + TILE - 1) // TILE
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    for t in range(pid, ntiles, NPROG):
        blocks = t * TILE + tl.arange(0, TILE)
        cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
        values = tl.load(
            logits + row * stride_row + cols * stride_col,
            (blocks[:, None] < nblocks)
            & (offsets[None, :] < BLOCK_SIZE)
            & (cols < end)
            & (cols < width),
            other=-float("inf"),
        )
        reduced = tl.reduce(values, 1, _max_with_nan)
        reduced = tl.where(
            (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE),
            float("inf"),
            reduced,
        )
        # Whole tiles are written, so slots in [nvalid, roundup(nvalid, TILE))
        # hold -inf (top-k only reads [0, nvalid) anyway).
        tl.store(scores + row * nblocks + blocks, reduced, blocks < nblocks)


@triton.jit
def _drop_neg_inf_candidates_kernel(
    scores,
    output,
    out_stride_row,
    nblocks,
    K: tl.constexpr,
    TILE: tl.constexpr,
):
    # Stock semantics: a selected block whose score is -inf is padding (-1).
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    m = cols < K
    index = tl.load(output + row * out_stride_row + cols, m, other=-1)
    ok = m & (index >= 0)
    value = tl.load(scores + row * nblocks + index, ok, other=-float("inf"))
    tl.store(
        output + row * out_stride_row + cols,
        tl.where(ok & (value != -float("inf")), index, -1),
        m,
    )


def _k1_nprog(rows: int, ntiles: int) -> int:
    return max(1, min(ntiles, triton.cdiv(_DSV41_CAND_CTAS, max(rows, 1))))


def _k1_topk_workspace(device: torch.device) -> torch.Tensor:
    # Same request the decode top-k makes, so the shared workspace never grows.
    try:
        from vllm.v1.worker.workspace import current_workspace_manager

        (ws,) = current_workspace_manager().get_simultaneous(
            ((_RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )
        return ws
    except Exception:
        key = str(device)
        ws = _k1_ws_cache.get(key)
        if ws is None:
            ws = torch.empty(
                (_RADIX_TOPK_WORKSPACE_SIZE,), dtype=torch.uint8, device=device
            )
            _k1_ws_cache[key] = ws
        return ws


def _k1_lenselect_ok(
    logits: torch.Tensor, row_ks, topk_blocks: int, out: torch.Tensor
) -> bool:
    rows = logits.shape[0]
    return (
        row_ks is None
        and logits.dtype == torch.float32
        and topk_blocks in (512, 1024, 2048)
        and out.dtype == torch.int32
        and out.dim() == 2
        and out.shape[0] == rows
        and out.shape[1] == topk_blocks
        and out.stride(1) == 1
        and out.stride(0) == topk_blocks
        and hasattr(torch.ops._C, "persistent_topk")
    )


def _select_candidate_blocks_lenselect(
    logits: torch.Tensor,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int,
) -> None:
    rows, width = logits.shape
    nblocks = triton.cdiv(width, block_size)
    scores = logits.new_empty((rows, nblocks))
    lens = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    nprog = _k1_nprog(rows, triton.cdiv(nblocks, 128))
    _block_scores_bounded_kernel[(rows, nprog)](
        logits,
        None,
        row_ke,
        scores,
        lens,
        *logits.stride(),
        0,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        False,
        row_repeat,
        128,
        nprog,
    )
    # Host max_seq_len = nblocks (static): per-row lengths come from `lens`;
    # row_bound = min(stride, max_seq_len) = nblocks never truncates a row.
    torch.ops._C.persistent_topk(
        scores, lens, out, _k1_topk_workspace(logits.device), topk_blocks, nblocks
    )
    _drop_neg_inf_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
        scores, out, out.stride(0), nblocks, topk_blocks, 256
    )


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
    lenselect: bool | None = None,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    if lenselect is None:
        lenselect = _DSV41_CAND_LENSELECT
    if lenselect and _k1_lenselect_ok(logits, row_ks, topk_blocks, out):
        _select_candidate_blocks_lenselect(
            logits, row_ke, topk_blocks, block_size, out, row_repeat
        )
        return
    nblocks = triton.cdiv(width, block_size)
    scores = logits.new_empty((rows, nblocks))
    _block_scores_kernel[(rows, triton.cdiv(nblocks, 128))](
        logits,
        row_ks,
        row_ke,
        scores,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        128,
    )
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
        top.values,
        top.indices,
        out,
        *out.stride(),
        top.values.shape[1],
        topk_blocks,
        256,
    )


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
    lenaware: bool | None = None,
) -> None:
    """Mask packed logits outside causal bounds and request-local candidates."""
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, nblocks + 1), device=logits.device, dtype=torch.uint8)
    start_stride = row_ks.stride(0) if row_ks is not None else 0
    if lenaware is None:
        # Decode only (prefill passes row_ks and keeps the stock kernels).
        lenaware = _DSV41_CAND_LENAWARE and row_ks is None
    if lenaware:
        _candidate_flags_bounded_kernel[(rows,)](
            candidate_blocks,
            row_ks,
            row_ke,
            flags,
            *candidate_blocks.stride(),
            start_stride,
            row_ke.stride(0),
            width,
            nblocks,
            block_size,
            candidate_blocks.shape[1],
            row_ks is not None,
            row_repeat,
        )
        nprog = _k1_nprog(rows, triton.cdiv(width, 1024))
        _mask_candidates_bounded_kernel[(rows, nprog)](
            logits,
            row_ks,
            row_ke,
            flags,
            *logits.stride(),
            start_stride,
            row_ke.stride(0),
            width,
            nblocks,
            block_size,
            row_ks is not None,
            row_repeat,
            1024,
            nprog,
        )
        return
    _candidate_flags_kernel[(rows,)](
        candidate_blocks,
        row_ks,
        flags,
        *candidate_blocks.stride(),
        start_stride,
        width,
        nblocks,
        block_size,
        candidate_blocks.shape[1],
        row_ks is not None,
        row_repeat,
    )
    _mask_candidates_kernel[(rows, triton.cdiv(width, 1024))](
        logits,
        row_ks,
        row_ke,
        flags,
        *logits.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        1024,
    )
