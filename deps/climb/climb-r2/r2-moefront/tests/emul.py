"""Bit-level numpy emulation of the live FC1-input quant (SASS of per_token_group_quant_8bit_register_kernel
<bf16, e4m3fn, false, true, 128>, sm_120 cubin in vllm/_C_stable_libtorch.abi3.so) and of the r2-moefront fused
kernel's quant (dsv41_routeprep._dsv41_rp_kernel), every step with explicit float32 rounding (numpy float32 ops are
IEEE single, round-to-nearest-even, no FTZ on this host; checked by `assert_ieee_host`).

Both functions take bf16 bit patterns (uint16) of shape [..., 128] (one quant group per last-axis row) and return
(e4m3 bytes uint8 [..., 128], fp32 scale bits uint32 [...]).
"""
import numpy as np

F32 = np.float32
U32 = np.uint32
I32 = np.int32

EPS = F32(1e-10)  # per_token_group_quant_fp8(eps=1e-10) -> C++ float
FP8_MAX = F32(448.0)
FP8_MIN = F32(-448.0)
FLOOR = F32(1e-10)  # SASS immediate 1.00000001335143196e-10 == float32(1e-10)


def assert_ieee_host():
    tiny = np.array([np.float32(2.0) ** -149], dtype=F32)
    assert (tiny * F32(0.5))[0] == F32(0.0) and (tiny * F32(1.0))[0] != 0, "host flushes denormals"
    assert (np.array([np.float32(2.0) ** -126], dtype=F32) * F32(0.75))[0] != 0, "host flushes denormals"


def bf16_to_f32(xb):
    return (xb.astype(U32) << U32(16)).view(F32)


def f32_bits(x):
    return np.ascontiguousarray(x, dtype=F32).view(U32)


def bits_f32(b):
    return np.ascontiguousarray(b, dtype=U32).view(F32)


# ----------------------------------------------------------------------------------------------- live (SASS order)
def live_quant(xb):
    """Live kernel: 8 lanes x 16 bf16 per group. Each lane: m = FMNMX(|x0|, eps), then FMNMX(m, |x_i|) i=1..15 in
    register order; SHFL.BFLY xor 4, 2, 1 with FMNMX; y = absmax / fp8_max (IEEE: MUFU.RCP + Newton + FCHK slow path
    = correctly rounded); y = FMNMX(|y|, 1e-10); scale bits = exp field (+1 ulp of exponent if any mantissa bit);
    inv = IEEE 1/scale; per element: FMUL x*inv; FMNMX(q, fp8_min); FMNMX(., fp8_max, min); c10 e4m3fn encode with
    the satfinite branches (0x43f00000, 0x3c7fffff, +0x407ffff + odd, FADD 16384)."""
    x = bf16_to_f32(xb)
    shp = x.shape[:-1]
    lanes = x.reshape(shp + (8, 16))
    m = np.fmax(np.abs(lanes[..., 0]), EPS)
    for i in range(1, 16):
        m = np.fmax(m, np.abs(lanes[..., i]))
    for d in (4, 2, 1):  # butterfly over lane index
        idx = np.arange(8) ^ d
        m = np.fmax(m, m[..., idx])
    absmax = m[..., 0]
    assert np.array_equal(f32_bits(m), np.repeat(f32_bits(absmax)[..., None], 8, axis=-1))
    y = (absmax / FP8_MAX).astype(F32)
    y = np.fmax(np.abs(y), FLOOR)
    yb = f32_bits(y)
    e = yb & U32(0x7F800000)
    sbits = np.where((yb & U32(0x7FFFFF)) != 0, e + U32(0x800000), e).astype(U32)
    scale = bits_f32(sbits)
    inv = (F32(1.0) / scale).astype(F32)
    q = (x * inv[..., None]).astype(F32)
    q = np.fmax(q, FP8_MIN)
    q = np.fmin(q, FP8_MAX)
    r27 = f32_bits(q)  # signed bits
    r28 = r27 & U32(0x7FFFFFFF)
    sign = (r27 >> U32(24)) & U32(0x80)
    # branch 1: >= 480.0 (satfinite: NaN -> 0x7f, else 0x7e)
    big = np.where(r28 > U32(0x7F800000), U32(0x7F), U32(0x7E))
    # branch 2: normal (> 0x3c7fffff): R9 = R27 + 0x407ffff + ((R27 >> 20) & 1), signed bits, uint32 wrap
    odd = (r27 >> U32(20)) & U32(1)
    r9 = (r27.astype(np.uint64) + 0x407FFFF + odd).astype(np.uint64) & 0xFFFFFFFF
    r9 = r9.astype(U32)
    norm = np.where((r9 & U32(0xFF00000)) != U32(0x7F00000), r9 >> U32(20), U32(0x7E))
    # branch 3: subnormal: FADD |q| + 16384 (fp32 RN), low byte
    sub = f32_bits((bits_f32(r28) + F32(16384.0)).astype(F32))
    mag = np.where(r28 >= U32(0x43F00000), big, np.where(r28 > U32(0x3C7FFFFF), norm, sub))
    byte = ((mag & U32(0xFF)) | sign).astype(np.uint8)
    return byte, sbits


# --------------------------------------------------------------------------------- fused kernel (Triton lines)
def new_quant(xb):
    """dsv41_routeprep._dsv41_rp_kernel, line by line: xf = bf16->fp32; amax = maximum(max(|xf|, axis=1), eps)
    (tree reduction, any order); y = div_rn(amax, fp8_max); y = maximum(|y|, 1e-10); sb = exp field + 0x800000 if
    mantissa; inv = div_rn(1, scale); q = xf * inv; q = minimum(maximum(q, fp8_min), fp8_max); encode with int32 ops
    on a = |q| bits (tn = a + 0x0407FFFF + odd; check bits 20..27 != 0x7f; fs = float(a) + 16384)."""
    x = bf16_to_f32(xb)
    ax = np.abs(x)
    # pairwise tree (a different order than the live kernel on purpose: max is exact and order-free)
    while ax.shape[-1] > 1:
        h = ax.shape[-1] // 2
        ax = np.maximum(ax[..., :h], ax[..., h:])  # Triton max.f32 == fmax for non-NaN
    amax = np.fmax(ax[..., 0], EPS)
    y = (amax / FP8_MAX).astype(F32)
    y = np.fmax(np.abs(y), F32(1e-10))
    yb = f32_bits(y).view(I32)
    sb = yb & I32(0x7F800000)
    sb = np.where((yb & I32(0x007FFFFF)) != 0, sb + I32(0x00800000), sb).astype(I32)
    scale = sb.view(F32)
    inv = (F32(1.0) / scale).astype(F32)
    q = (x * inv[..., None]).astype(F32)
    q = np.fmin(np.fmax(q, FP8_MIN), FP8_MAX)
    qb = f32_bits(q).view(I32)
    a = qb & I32(0x7FFFFFFF)
    sign = (qb >> 24) & I32(0x80)  # arithmetic shift, bit 31 lands in bit 7
    odd = (a >> 20) & I32(1)
    tn = (a.astype(np.int64) + 0x0407FFFF + odd).astype(I32)  # < 2^31 for a < 0x43f00000: no overflow
    mag_n = np.where((tn & I32(0x0FF00000)) != I32(0x07F00000), tn >> 20, I32(0x7E))
    fs = (a.view(F32) + F32(16384.0)).astype(F32)
    mag_s = fs.view(I32)
    mag_o = np.where(a > I32(0x7F800000), I32(0x7F), I32(0x7E))
    mag = np.where(a >= I32(0x43F00000), mag_o, np.where(a >= I32(0x3C800000), mag_n, mag_s))
    byte = ((mag & I32(0xFF)) | sign).astype(np.uint8)
    return byte, sb.view(U32)


# -------------------------------------------------------------------- independent reference (float64 + torch c10)
def ref_quant(xb):
    """Spec reference: y = fp32(absmax/448) (IEEE), scale = 2^ceil(log2(max(y, 1e-10f))) computed in float64,
    q = clamp(x * 2^-k) (exact power-of-two scaling, float64 then fp32), e4m3fn by torch's CPU cast (c10)."""
    import torch

    x = bf16_to_f32(xb)
    absmax = np.fmax(np.abs(x).max(axis=-1), EPS)
    y = (absmax / FP8_MAX).astype(F32)
    y = np.fmax(y, F32(1e-10)).astype(np.float64)
    k = np.ceil(np.log2(y))
    # log2 in float64 is exact enough at powers of two; fix any off-by-one near them explicitly
    k = np.where(np.exp2(k - 1) >= y, k - 1, k)
    k = np.where(np.exp2(k) < y, k + 1, k)
    scale = np.exp2(k).astype(F32)
    q = (x.astype(np.float64) * np.exp2(-k)[..., None]).astype(F32)  # fp32 rounding only below 2^-126 (-> 0 byte)
    q = np.clip(q, -448.0, 448.0).astype(F32)
    t = torch.from_numpy(q.copy()).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    return t, f32_bits(scale)


def pack_live(sbits4):
    """mfront scatter / transpose_and_pack: (v0 >> 23) | (v1 >> 15) | (v2 >> 7) | (v3 << 1) on uint32."""
    v = sbits4.astype(U32)
    return ((v[..., 0] >> U32(23)) | (v[..., 1] >> U32(15)) | (v[..., 2] >> U32(7)) | (v[..., 3] << U32(1))).astype(U32)


def pack_new(sbits4):
    """fused kernel: same formula on int32 (arithmetic shifts; scale bits are positive)."""
    v = sbits4.astype(U32).view(I32)
    w = (v[..., 0] >> 23) | (v[..., 1] >> 15) | (v[..., 2] >> 7) | (v[..., 3] << 1)
    return w.astype(I32).view(U32)
