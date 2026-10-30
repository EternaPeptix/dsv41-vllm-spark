"""pd-d05 verify-mode compare kernel (DSV41_MOE_FC2_D05=verify only).

For an FC2 output D (served, original kernel) and a shadow S (D05 kernel,
pre-filled with POISON) over the same inputs and scatter32 grouped layout:

  * 8-row group g is "written" iff layout[8*(row//8)] >= 0. Every int16 word
    of a written group must equal D bitwise (live rows and the padding rows
    of a partially live group are computed from identical inputs).
  * A dead group must still hold POISON in S (the store was skipped).

Counter row (int32): [calls, live rows, mismatching words in written groups,
non-POISON words in dead groups]. Nothing here is on the served path.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

_N = 5120
_BLOCK_N = 1024


@triton.jit
def _d05_compare_kernel(d_ptr, s_ptr, layout_ptr, cnt_ptr, POISON: tl.constexpr,
                        N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    nb = tl.program_id(1)
    e = tl.load(layout_ptr + row)
    g = tl.load(layout_ptr + (row // 8) * 8)
    cols = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    base = row.to(tl.int64) * N
    a = tl.load(d_ptr + base + cols)
    b = tl.load(s_ptr + base + cols)
    if (row == 0) & (nb == 0):
        tl.atomic_add(cnt_ptr + 0, 1)
    if g >= 0:
        bad = tl.sum((a != b).to(tl.int32), axis=0)
        if bad > 0:
            tl.atomic_add(cnt_ptr + 2, bad)
        if (e >= 0) & (nb == 0):
            tl.atomic_add(cnt_ptr + 1, 1)
    else:
        touched = tl.sum((b != POISON).to(tl.int32), axis=0)
        if touched > 0:
            tl.atomic_add(cnt_ptr + 3, touched)


def compare(d: torch.Tensor, shadow: torch.Tensor, layout: torch.Tensor,
            counters: torch.Tensor, poison: int) -> None:
    m = d.shape[0]
    assert d.shape == shadow.shape and d.shape[1] == _N and m % 8 == 0
    assert layout.dtype == torch.int32 and layout.numel() == m
    assert counters.dtype == torch.int32 and counters.numel() >= 4
    grid = (m, _N // _BLOCK_N)
    _d05_compare_kernel[grid](
        d.view(torch.int16), shadow.view(torch.int16), layout, counters,
        POISON=poison, N=_N, BLOCK_N=_BLOCK_N, num_warps=4)
