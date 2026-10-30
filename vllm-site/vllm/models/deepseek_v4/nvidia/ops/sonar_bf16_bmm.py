# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV41 r3: skinny split-K grouped WO-A projection for SM12x decode.

Ported from dphnAI/sonar PR #1788
(aphrodite/models/deepseek_v4/nvidia/ops/_sm120_bf16_bmm.py) and extended with
an FP8-weight (W8A16) variant for GB10, where the bf16 bmm is bandwidth-bound.

Opt-in via env (read once, at import):
  DSV41_SONAR_OPROJ=0|1|bf16|fp8   0 (default) = off; 1/bf16 = the PR kernel on
                                   the live bf16 weight; fp8 = same kernel but it
                                   streams an exact MXFP8 re-encoding of the bf16
                                   weight (half the bytes; bf16 products are
                                   bit-identical, only the K summation order
                                   differs from cuBLAS, same as mode bf16).
  DSV41_SONAR_OPROJ_MIN_M          smallest token count routed here (default 1;
                                   the PR uses 2)
  DSV41_SONAR_OPROJ_MAX_M          largest token count routed here (default 64)
  DSV41_SONAR_OPROJ_SPLITS         force split-K count (default: 8 if M<=32 else 16)

In fp8 mode the FP8 copy (+~4.3 MB/layer, ~173 MB/rank for 40 layers) is built
on the first eager call of any size (vLLM's memory-profile run, which precedes
graph capture) and is asserted bit-exact against the bf16 weight; a layer that
fails the check or is first seen during capture uses the bf16 kernel instead.
M is a runtime argument (not constexpr), so odd prefill-chunk sizes do not
trigger a Triton JIT compile mid-serving.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_MODE_RAW = os.environ.get("DSV41_SONAR_OPROJ", "0").strip().lower()
MODE = {"1": "bf16", "bf16": "bf16", "fp8": "fp8"}.get(_MODE_RAW, "")
MIN_M = int(os.environ.get("DSV41_SONAR_OPROJ_MIN_M", "1"))
MAX_M = int(os.environ.get("DSV41_SONAR_OPROJ_MAX_M", "64"))
_FORCE_SPLITS = int(os.environ.get("DSV41_SONAR_OPROJ_SPLITS", "0"))
_MX_BLOCK = 32
_logged = False


def enabled() -> bool:
    return bool(MODE)


@triton.jit
def _sm120_wo_a_partial(
    A,
    W,
    S,
    P,
    M,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ASG: tl.int64,
    ASM: tl.int64,
    ASK: tl.int64,
    CHUNK: tl.constexpr,
    FP8W: tl.constexpr,
):
    group = tl.program_id(1) // tl.cdiv(M, 16)
    row = tl.program_id(1) % tl.cdiv(M, 16) * 16 + tl.arange(0, 16)
    col = tl.program_id(0) * 64 + tl.arange(0, 64)
    split = tl.program_id(2)
    acc = tl.zeros((16, 64), tl.float32)
    for step in range(CHUNK // 128):
        k = split * CHUNK + step * 128 + tl.arange(0, 128)
        a = tl.load(
            A + group * ASG + row[:, None] * ASM + k[None, :] * ASK,
            row[:, None] < M,
            other=0,
        )
        # N % 64 == 0 and K % (128 * splits) == 0 are checked by the caller,
        # so weight/scale tiles are always in bounds (no mask; fp8 loads cannot
        # take an integer `other`).
        if FP8W:
            q = tl.load(W + group * N * K + col[None, :] * K + k[:, None])
            e = tl.load(
                S + group * N * (K // 32) + col[None, :] * (K // 32) + k[:, None] // 32
            )
            # Same arithmetic as dequant_mxfp8_to_bf16: fp32 q * 2^(e-127),
            # rounded once to bf16 -> bit-identical to the live bf16 weight.
            # 2^(e-127) is built from its bits (e in 1..254), not ex2.approx.
            scl = (e.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            w = (q.to(tl.float32) * scl).to(tl.bfloat16)
        else:
            w = tl.load(W + group * N * K + col[None, :] * K + k[:, None])
        acc = tl.dot(a, w, acc)
    tl.store(
        P + ((split * G + group) * M + row[:, None]) * N + col[None, :],
        acc,
        (row[:, None] < M) & (col[None, :] < N),
    )


@triton.jit
def _sm120_wo_a_reduce(
    P, OUT, M, G: tl.constexpr, N: tl.constexpr, SPLITS: tl.constexpr
):
    pos = tl.program_id(0) * 256 + tl.arange(0, 256)
    split = tl.arange(0, SPLITS)
    total = G * M * N
    partial = tl.load(P + split[:, None] * total + pos[None, :], pos[None, :] < total, other=0)
    value = tl.sum(partial, axis=0)
    group, row, col = pos // (M * N), pos // N % M, pos % N
    tl.store(OUT + (row * G + group) * N + col, value, pos < total)


def _splits_for(m: int) -> int:
    if _FORCE_SPLITS > 0:
        return _FORCE_SPLITS
    return 8 if m <= 32 else 16


def grouped_bmm(
    a: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
) -> None:
    """out[m, g, :] = a[m, g, :] @ weight[g].T  (weight [G, N, K]).

    ``weight`` is bf16, or float8_e4m3fn with ``weight_scale`` uint8 E8M0
    [G, N, K // 32]. Input strides are preserved; ``out`` is contiguous
    [M, G, N] bf16.
    """
    m, groups, k = a.shape
    n = weight.shape[1]
    splits = _splits_for(m)
    fp8w = weight.dtype == torch.float8_e4m3fn
    partial = torch.empty((splits, groups, m, n), device=a.device, dtype=torch.float32)
    _sm120_wo_a_partial[(triton.cdiv(n, 64), groups * triton.cdiv(m, 16), splits)](
        a,
        weight,
        weight_scale if fp8w else weight,
        partial,
        m,
        groups,
        n,
        k,
        a.stride(1),
        a.stride(0),
        a.stride(2),
        k // splits,
        fp8w,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    _sm120_wo_a_reduce[(triton.cdiv(groups * m * n, 256),)](
        partial, out, m, groups, n, splits, num_warps=4
    )


@torch.no_grad()
def mxfp8_requant_exact(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Re-encode a bf16 weight that came from MXFP8 dequant, exactly.

    For each 32-wide K block pick the smallest e with amax / 2^e <= 448. That e
    is <= the checkpoint's own scale exponent, so every q' = w / 2^e is the
    original e4m3 code shifted up by a power of two (no bits lost). Returns
    (q fp8 [.., K], e8m0 uint8 [.., K//32]) or None if the round trip is not
    bit-exact (the caller then keeps the bf16 kernel for that layer).
    """
    shape = w.shape
    k = shape[-1]
    if k % _MX_BLOCK:
        return None
    wf = w.float().reshape(*shape[:-1], k // _MX_BLOCK, _MX_BLOCK)
    amax = wf.abs().amax(dim=-1)
    _, ex = torch.frexp(amax)  # amax = m * 2^ex, m in [0.5, 1)
    # smallest e with amax / 2^e <= 448 (e4m3 max): start at floor(log2 amax)-8
    # (quotient in [256, 512)) and bump by one where it exceeds 448. Dividing
    # by a power of two is exact, so the test is exact. Every value is then
    # its original e4m3 code times 2^(s - e) with s - e >= 0: exact.
    e = ex.to(torch.int32) - 1 - 8
    e = e + (amax / torch.ldexp(torch.ones_like(amax), e) > 448).to(torch.int32)
    e = torch.where(amax > 0, e, torch.zeros_like(e))
    e8 = (e + 127).clamp(1, 254)
    # exact 2^(e8-127) from its fp32 bit pattern (matches the kernel)
    descale = (e8.to(torch.int32) << 23).view(torch.float32)
    q = (wf / descale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    back = (q.float() * descale.unsqueeze(-1)).to(torch.bfloat16).reshape(shape)
    if not torch.equal(back.view(torch.int16), w.contiguous().view(torch.int16)):
        return None
    return q.reshape(shape).contiguous(), e8.to(torch.uint8).contiguous()


def _fp8_copy(wo_a, grouped_weight: torch.Tensor):
    cached = getattr(wo_a, "_dsv41_sonar_fp8", None)
    if cached is not None:
        return cached if cached is not False else None
    if torch.cuda.is_current_stream_capturing():
        return None  # never build weights inside a graph; bf16 kernel instead
    res = mxfp8_requant_exact(grouped_weight)
    if res is None:
        logger.warning(
            "DSV41 sonar o_proj: fp8 re-encode not exact for %s; using bf16 kernel",
            getattr(wo_a, "prefix", "?"),
        )
        wo_a._dsv41_sonar_fp8 = False
        return None
    wo_a._dsv41_sonar_fp8 = res
    return res


def try_project(o_proj_input: torch.Tensor, wo_a, grouped_weight: torch.Tensor, z: torch.Tensor) -> bool:
    """Run the skinny kernel if the call qualifies; return False to fall back."""
    global _logged
    tokens = o_proj_input.shape[0]
    n, k = grouped_weight.shape[1], grouped_weight.shape[2]
    splits = _splits_for(tokens)
    res = None
    if MODE == "fp8":
        # Built on the first eager call of any size (vLLM's memory-profile run
        # is eager and precedes graph capture), never inside a capture.
        res = _fp8_copy(wo_a, grouped_weight)
    if not (
        MIN_M <= tokens <= MAX_M
        and o_proj_input.dtype == grouped_weight.dtype == torch.bfloat16
        and grouped_weight.is_contiguous()
        and z.is_contiguous()
        and n % 64 == 0
        and k % (128 * splits) == 0
    ):
        return False
    weight, scale = grouped_weight, None
    if res is not None:
        weight, scale = res
    if not _logged:
        _logged = True
        logger.info(
            "DSV41 sonar o_proj enabled: mode=%s weight=%s M=[%d,%d] G=%d N=%d K=%d",
            MODE, weight.dtype, MIN_M, MAX_M, grouped_weight.shape[0], n, k,
        )
    grouped_bmm(o_proj_input, weight, z, scale)
    return True
