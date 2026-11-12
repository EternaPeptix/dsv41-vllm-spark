# r5-glue device test helper (review N6): a PDL primary that lets its dependents launch at once and writes its output
# LATE. A dependent that touches that output before its griddepcontrol.wait reads the poison that was there before.
import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _r5g_late_copy(src, dst, n, spin, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()  # dependents (launched with launch_pdl) may start right now
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32) + pid.to(tl.float32)
    for i in range(spin):  # ~4 cycles per trip, dependent chain: spin=20000 is tens of microseconds
        acc = acc * 0.5 + 1.0
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(src + offs, mask=m, other=0)
    tl.store(dst + offs, tl.where(acc > -1.0, v, 0), mask=m)  # acc > -1 always: keeps the spin alive


def late_copy(src, dst, spin=20000):
    """dst <- src (raw bytes, both contiguous, size a multiple of 4 bytes), written after the spin."""
    s32 = src.contiguous().view(torch.int32).reshape(-1)
    d32 = dst.view(torch.int32).reshape(-1)
    assert d32.data_ptr() == dst.data_ptr() and s32.numel() == d32.numel()
    n = s32.numel()
    _r5g_late_copy[(triton.cdiv(n, BLOCK),)](s32, d32, n, spin, BLOCK=BLOCK, num_warps=4)
