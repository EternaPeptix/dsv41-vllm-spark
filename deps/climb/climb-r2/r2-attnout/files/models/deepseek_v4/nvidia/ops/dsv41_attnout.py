# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 climb r2-attnout, lever G19 (attn:L-OP): attention output path on SM12x decode.

Gate: DSV41_ATTNOUT_OP (read once at import by deepseek_v4/nvidia/ops/o_proj.py; unset/0 = this module
is never imported and the live path runs unchanged). Value = comma list of parts:

  rope   fold the bf16 inverse RoPE (fused_inv_rope_fp8_quant, quantize=False; trace name `kernel`, grid
         (M, 8)) into the A-tile load of the sonar split-K wo_a partial kernel. The inverse-RoPE launch,
         its out_buf allocation and one graph dependency gap disappear.
  rq     replace the split reduce (_sm120_wo_a_reduce) + flashinfer MXFP8QuantizeSwizzledKernel (the
         activation quant inside wo_b's FlashInferCutlassMxfp8LinearKernel) by ONE kernel that sums the
         32 splits in the live order, rounds to bf16, and quantizes with flashinfer's exact PTX; wo_b then
         receives a QuantizedActivation (the same path the SM100 fused WO-A uses). One launch + gap gone.
  st1    partial kernel with num_stages=1 (the K loop has exactly one 128-wide iteration at SPLITS=32,
         so stage 2 only costs shared memory: 20 KB -> occupancy 4 CTAs/SM).
  vs     partial kernel loads the E8M0 scale tile as [4, BN] and broadcasts it (live: 64 byte-gathers
         per thread of the same 256 bytes).
  bn128  partial kernel with a 16x128 output tile (256 CTAs instead of 512).
  aliases: 1|on|all = rope,rq ; tune = st1,vs

Every part keeps each output element's floating-point op sequence (see EXACTNESS.md in the kit):
  * same K split (SPLITS x CHUNK), same tl.dot (mma.sync m16n8k16 bf16, fp32 acc, kWidth 4 from the fp8
    weight upcast, 8 k16 steps ascending from acc=0), same dequant (q.f32 * 2^(e-127) -> bf16 RN),
  * inverse RoPE as the live PTX does it: ps = p*s (mul.rn), add = fma(x, c, ps), sub = fma(x, c, -ps),
    one bf16 RN; nope lanes are the untouched bf16 bits,
  * split sum = (p0+p2+...+p30) + (p1+p3+...+p31), each chain left to right (the live reduce layout
    [32,256] blocked [1,4]x[1,32]x[2,2]: 16 in-thread adds per warp row, then one cross-warp add),
  * quant = flashinfer cute-dsl helpers' PTX verbatim (float_to_ue8m0_fast, ue8m0_to_inv_scale_fast,
    bfloat2_to_fp8x2_scaled), amax*fp32(1/448), 128x4 swizzled scale layout, zero scales on pad rows.
On-device self-check: the first CHECK_CALLS eager (non-capturing) calls run the live kernels next to
the new ones (real weights + real activations, plus one synthetic hostile set on the first call) and
compare raw bits; a mismatch disables that part for the process and logs ERROR (live results are used).
Nothing is checked or built inside a CUDA-graph capture; a part that has not passed a check yet runs
the live kernels during capture.
"""
from __future__ import annotations

import os
import struct

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

TAG = "r2-attnout"
_RAW = os.environ.get("DSV41_ATTNOUT_OP", "0").strip().lower()
_ALIASES = {"1": ("rope", "rq"), "on": ("rope", "rq"), "all": ("rope", "rq"), "tune": ("st1", "vs")}
_KNOWN = {"rope", "rq", "st1", "vs", "bn128"}


def parse(raw: str) -> frozenset:
    raw = raw.strip().lower()
    if raw in ("", "0", "off"):
        return frozenset()
    parts: set[str] = set()
    for tok in raw.replace("+", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in _ALIASES:
            parts.update(_ALIASES[tok])
        elif tok in _KNOWN:
            parts.add(tok)
        else:
            raise ValueError(f"DSV41_ATTNOUT_OP={raw!r}: unknown part {tok!r} (known: "
                             f"{sorted(_KNOWN | set(_ALIASES))})")
    return frozenset(parts)


PARTS = parse(_RAW)
ROPE = "rope" in PARTS
RQ = "rq" in PARTS
ST1 = "st1" in PARTS
VS = "vs" in PARTS
BN128 = "bn128" in PARTS
TUNE = ST1 or VS or BN128
CHECK_CALLS = 16  # eager self-checks per process (each compares every enabled part)

# Per-part state: None = not yet checked on this device, True = passed >= 1 check, False = disabled.
_ok: dict[str, bool | None] = {"partial": None, "rq": None}
_checks_left = CHECK_CALLS
_synthetic_done = False
_logged = False

# fp32(1/448) bit pattern (flashinfer: Float32(INV_FLOAT8_E4M3_MAX), INV = 1.0 / 448.0, RN to fp32)
INV448_BITS = struct.unpack("<I", struct.pack("<f", 1.0 / 448.0))[0]

# ---- flashinfer quantization_cute_dsl_utils.py PTX, verbatim bodies --------------------------------
_ASM_UE8M0 = tl.constexpr("""
{
    .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;
    .reg .u32 bits, exp_biased, mantissa, bump, result;
    setp.le.f32 p_zero, $1, 0f00000000;
    mov.b32 bits, $1;
    shr.b32 exp_biased, bits, 23;
    and.b32 exp_biased, exp_biased, 255;
    and.b32 mantissa, bits, 0x7FFFFF;
    setp.ne.u32 p_has_mant, mantissa, 0;
    selp.u32 bump, 1, 0, p_has_mant;
    setp.eq.u32 p_exp_zero, exp_biased, 0;
    setp.le.u32 p_tiny_sub, mantissa, 0x400000;
    and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;
    @p_tiny_sub mov.u32 bump, 0;
    add.u32 result, exp_biased, bump;
    setp.gt.u32 p_ovf, result, 254;
    selp.u32 result, 254, result, p_ovf;
    selp.u32 $0, 0, result, p_zero;
}
""")
_ASM_INV_SCALE = tl.constexpr("""
{
    .reg .s32 new_exp;
    .reg .b32 float_bits;
    .reg .pred p_zero;
    setp.eq.u32 p_zero, $1, 0;
    sub.s32 new_exp, 254, $1;
    max.s32 new_exp, new_exp, 0;
    shl.b32 float_bits, new_exp, 23;
    mov.b32 $0, float_bits;
    @p_zero mov.b32 $0, 0;
}
""")
# bfloat2_to_fp8x2_scaled for ONE lane: the bf16 -> f32 widening (shl 16) is done by the caller's exact
# .to(tl.float32); mul.f32 / min / max / cvt.rn.satfinite.e4m3x2 are per lane, so lane f0 of the pair
# gives the same byte whatever its partner is (the partner here is +0, the result's high byte is dropped).
_ASM_FP8 = tl.constexpr("""
{
    .reg .f32 f0, f1;
    .reg .b16 fp8_pair;
    mul.f32 f0, $1, $2;
    mov.b32 f1, 0;
    min.f32 f0, f0, 0f43E00000;
    max.f32 f0, f0, 0fC3E00000;
    cvt.rn.satfinite.e4m3x2.f32 fp8_pair, f1, f0;
    cvt.u32.u16 $0, fp8_pair;
}
""")


# ---- kernels -----------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["M"])
def _r2_wo_a_partial(
    A,  # ROPE: attention output o [M, H, HEAD_DIM] bf16 (any token/head stride, unit dim stride)
    POS,  # ROPE: positions [M] int64
    CS,  # ROPE: cos_sin_cache [max_pos, 2*HALF_ROPE] fp32
    W,
    S,
    P,
    M,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ASG: tl.int64,  # not ROPE: live strides (group, token, k); ROPE: ASG unused
    ASM: tl.int64,  # token stride (both modes)
    ASK: tl.int64,  # not ROPE: k stride; ROPE: head stride
    CSS: tl.int64,  # cos_sin_cache row stride
    CHUNK: tl.constexpr,
    FP8W: tl.constexpr,
    ROPE: tl.constexpr,
    HPG: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    BN: tl.constexpr,
    VEC_SCALE: tl.constexpr,
):
    group = tl.program_id(1) // tl.cdiv(M, 16)
    row = tl.program_id(1) % tl.cdiv(M, 16) * 16 + tl.arange(0, 16)
    col = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    acc = tl.zeros((16, BN), tl.float32)
    for step in range(CHUNK // 128):
        k = split * CHUNK + step * 128 + tl.arange(0, 128)
        if ROPE:
            # A[row, k] = inverse-RoPE(o[row, group*HPG + k // HEAD_DIM, k % HEAD_DIM]) rounded to bf16,
            # with the live kernel's op sequence (fused_inv_rope_fp8_quant.py:120-138, PTX: mul.f32x2
            # p*s; fma.rn.f32x2 x*c + ps; neg; fma.rn.f32x2 x*c + (-ps); cvt.rn.bf16.f32; selp).
            rmask = row[:, None] < M
            head = group * HPG + k // HEAD_DIM
            d = k % HEAD_DIM
            base = A + row[:, None] * ASM + head[None, :] * ASK
            xb = tl.load(base + d[None, :], rmask, other=0)
            is_rope = d >= NOPE
            rl = d - NOPE
            rope2 = rmask & is_rope[None, :]
            xp = tl.load(base + (d ^ 1)[None, :], rope2, other=0).to(tl.float32)
            pos = tl.load(POS + row, row < M, other=0)
            ci = tl.maximum(rl >> 1, 0)
            cbase = CS + pos[:, None] * CSS + ci[None, :]
            c = tl.load(cbase, rope2, other=1.0)
            s = tl.load(cbase + HALF_ROPE, rope2, other=0.0)
            x = xb.to(tl.float32)
            ps = xp * s
            r_add = tl.fma(x, c, ps)
            r_sub = tl.fma(x, c, -ps)
            rot = tl.where(((rl & 1) == 0)[None, :], r_add, r_sub).to(tl.bfloat16)
            a = tl.where(rope2, rot, xb)
        else:
            a = tl.load(
                A + group * ASG + row[:, None] * ASM + k[None, :] * ASK,
                row[:, None] < M,
                other=0,
            )
        if FP8W:
            q = tl.load(W + group * N * K + col[None, :] * K + k[:, None])
            if VEC_SCALE:
                kb = (split * CHUNK + step * 128) // 32 + tl.arange(0, 4)
                e4 = tl.load(S + group * N * (K // 32) + col[None, :] * (K // 32) + kb[:, None])
                e = tl.reshape(tl.broadcast_to(e4[:, None, :], (4, 32, BN)), (128, BN))
            else:
                e = tl.load(
                    S + group * N * (K // 32) + col[None, :] * (K // 32) + k[:, None] // 32
                )
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


@triton.jit(do_not_specialize=["M"])
def _r2_reduce_quant(
    P,  # [SPLITS, 1, M, N] fp32
    Q,  # [M, N] uint8 (e4m3 codes)
    SF,  # [PADDED_M * SF_COLS] uint8, 128x4 swizzled UE8M0
    M,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    PADDED_M: tl.constexpr,
    SF_COLS: tl.constexpr,
    INV448: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_c = tl.program_id(0)
    r = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    cols = pid_c * 256 + tl.arange(0, 8)[:, None] * 32 + tl.arange(0, 32)[None, :]  # [8, 32]
    total = M * N
    base = P + r * N + cols
    # live _sm120_wo_a_reduce tree (SPLITS=32): warp row 0 chains splits 0,2,..,30, warp row 1 chains
    # 1,3,..,31 (each left to right, starting from the first element, no zero seed), then one add.
    ev = tl.load(base)
    od = tl.load(base + total)
    for i in tl.static_range(1, SPLITS // 2):
        ev = ev + tl.load(base + (2 * i) * total)
        od = od + tl.load(base + (2 * i + 1) * total)
    zb = (ev + od).to(tl.bfloat16)  # the live reduce's bf16 store (cvt.rn.bf16.f32)
    zf = zb.to(tl.float32)  # exact widening (= flashinfer's shl.b32 16)
    amax = tl.max(tl.abs(zf), axis=1)  # [8]; max is order independent for NaN-free input
    inv448 = tl.full([8], INV448, tl.int32).to(tl.float32, bitcast=True)
    normalized = amax * inv448
    ue = tl.inline_asm_elementwise(_ASM_UE8M0, "=r,f", [normalized], dtype=tl.int32, is_pure=True, pack=1)
    inv = tl.inline_asm_elementwise(_ASM_INV_SCALE, "=f,r", [ue], dtype=tl.float32, is_pure=True, pack=1)
    invb = tl.broadcast_to(inv[:, None], (8, 32))
    q = tl.inline_asm_elementwise(_ASM_FP8, "=r,f,f", [zf, invb], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(Q + r * N + cols, q.to(tl.uint8))
    c = pid_c * 8 + tl.arange(0, 8)
    sf_col = (c % 4) + (c // 4) * 512
    off = sf_col + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * SF_COLS)
    tl.store(SF + off, ue.to(tl.uint8))
    # flashinfer writes UE8M0 0 for every padding row M..PADDED_M-1; program row r covers M+r, M+r+M, ...
    for p in range(M + r, PADDED_M, M):
        offp = sf_col + (p % 32) * 16 + ((p % 128) // 32) * 4 + (p // 128) * (128 * SF_COLS)
        tl.store(SF + offp, tl.zeros([8], tl.uint8))


# ---- host side ---------------------------------------------------------------------------------------
def _sonar():
    from vllm.models.deepseek_v4.nvidia.ops import sonar_bf16_bmm

    return sonar_bf16_bmm


def _capturing() -> bool:
    return torch.cuda.is_current_stream_capturing()


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a = a.contiguous().view(torch.uint8)
    b = b.contiguous().view(torch.uint8)
    return bool(torch.equal(a, b))


def _first_diff(a: torch.Tensor, b: torch.Tensor) -> str:
    a8 = a.contiguous().view(torch.uint8).flatten()
    b8 = b.contiguous().view(torch.uint8).flatten()
    if a8.shape != b8.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    idx = torch.nonzero(a8 != b8)
    return "none" if idx.numel() == 0 else f"byte {int(idx[0])} of {a8.numel()} ({idx.numel()} differ)"


class Ctx:
    """Per-call context handed through sonar.try_project (and the L2-prefetch wrapper's *args)."""

    __slots__ = ("rope", "inv_args", "wo_b", "out")

    def __init__(self, rope, inv_args, wo_b):
        self.rope = rope  # (positions, cos_sin_cache, heads_per_group, nope_dim, half_rope) or None
        self.inv_args = inv_args  # kwargs to recompute the live inverse RoPE (self-check / fallback)
        self.wo_b = wo_b
        self.out = None

    def run(self, a, weight, scale, z) -> bool:
        return _run(self, a, weight, scale, z)


def _live_inv_rope(o, a):
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

    out, _ = fused_inv_rope_fp8_quant(o, **a)
    return out


def o_proj(o, positions, cos_sin_cache, wo_a, wo_b, *, n_groups, heads_per_group, nope_dim, rope_dim,
           o_lora_rank, quant_group_size, tma_aligned_scales):
    """deep_gemm_fp8_o_proj's bf16-wo_a branch with the enabled parts.

    Returns None before launching anything when nothing applies (the caller then runs the live code);
    once it launched work it always finishes the projection itself (live fallback included).
    """
    from vllm.platforms import current_platform

    sonar = _sonar()
    if not (sonar.enabled() and current_platform.is_device_capability_family(120)):
        return None
    use_rope = (
        ROPE
        and _ok["partial"] is not False
        and o.dtype == torch.bfloat16
        and o.dim() == 3
        and o.stride(2) == 1
        and o.shape[1] == n_groups * heads_per_group
        and o.shape[2] == nope_dim + rope_dim
        and o.shape[2] % 128 == 0
        and rope_dim % 2 == 0
        and nope_dim % 2 == 0
        and positions.dtype == torch.int64
        and positions.dim() == 1
        and positions.shape[0] >= o.shape[0]
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[-1] == rope_dim
        and cos_sin_cache.stride(1) == 1
    )
    if not (use_rope or (TUNE and _ok["partial"] is not False) or (RQ and _ok["rq"] is not False)):
        return None
    inv_args = dict(
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
        tma_aligned_scales=tma_aligned_scales,
        quantize=False,
    )
    grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
    if use_rope:
        inp = o
        rope = (positions, cos_sin_cache, heads_per_group, nope_dim, rope_dim // 2)
    else:
        inp = _live_inv_rope(o, inv_args)  # the live call, same arguments
        rope = None
    z = torch.empty((o.shape[0], n_groups, o_lora_rank), device=o.device, dtype=torch.bfloat16)
    ctx = Ctx(rope, inv_args, wo_b)
    # sonar.try_project may be the L2-prefetch recorder's wrapper (it forwards *args): one call per
    # projection, as on the live path, so the learned wo_a read sequence is unchanged.
    if sonar.try_project(inp, wo_a, grouped_weight, z, ctx):
        return wo_b(ctx.out)
    if use_rope:
        inp = _live_inv_rope(o, inv_args)
    torch.bmm(inp.transpose(0, 1), grouped_weight.transpose(1, 2), out=z.transpose(0, 1))
    return wo_b(z.flatten(1))


def _rq_layer_ok(wo_b) -> bool:
    """wo_b is DeepseekV4Attention._wo_b_proj of a layer whose wo_b runs FlashInferCutlassMxfp8 with a
    dynamic MXFP8 input and no GEMM-RS (so QuantizedActivation goes straight to apply_weights)."""
    layer = getattr(wo_b, "__self__", None)
    fn = getattr(wo_b, "__func__", None)
    if layer is None or fn is None or fn.__name__ != "_wo_b_proj":
        return False
    cached = getattr(layer, "_r2ao_rq", None)
    if cached is not None:
        return cached
    ok = False
    try:
        from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
        from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

        mod = layer.wo_b
        method = getattr(mod, "scheme", mod.quant_method)
        kern = getattr(method, "kernel", None)
        ok = (
            getattr(layer, "gemm_rs", None) is None
            and getattr(mod, "input_is_parallel", False) is True  # RowParallelLinear passes input_ through
            and get_input_quant_key(mod) == kMxfp8Dynamic
            and type(kern).__name__ == "FlashInferCutlassMxfp8LinearKernel"
            and type(getattr(method, "fmt", None)).__name__ in ("FormatScheme", "NoneType")
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] rq eligibility check failed for %s: %r", TAG, type(layer).__name__, e)
        ok = False
    layer._r2ao_rq = ok
    return ok


def _launch_partial(a, rope, weight, scale, partial, m, groups, n, k, splits, fp8w, new: bool):
    """new=False: the live _sm120_wo_a_partial launch of sonar.grouped_bmm, verbatim."""
    sonar = _sonar()
    if not new:
        sonar._sm120_wo_a_partial[(triton.cdiv(n, 64), groups * triton.cdiv(m, 16), splits)](
            a, weight, scale if fp8w else weight, partial, m, groups, n, k,
            a.stride(1), a.stride(0), a.stride(2), k // splits, fp8w,
            num_warps=4, num_stages=2, enable_fp_fusion=False,
        )
        return
    bn = 128 if (BN128 and n % 128 == 0) else 64
    if rope is not None:
        positions, cs, hpg, nope, half = rope
        args = (a, positions, cs)
        asg, asm, ask = 0, a.stride(0), a.stride(1)
        head_dim = a.shape[2]
        css = cs.stride(0)
    else:
        args = (a, a, a)  # POS/CS unused (ROPE constexpr False)
        asg, asm, ask = a.stride(1), a.stride(0), a.stride(2)
        hpg, head_dim, nope, half, css = 1, 128, 0, 1, 0
    _r2_wo_a_partial[(triton.cdiv(n, bn), groups * triton.cdiv(m, 16), splits)](
        *args, weight, scale if fp8w else weight, partial, m, groups, n, k,
        asg, asm, ask, css, k // splits, fp8w,
        rope is not None, hpg, head_dim, nope, half, bn, VS,
        num_warps=4, num_stages=1 if ST1 else 2, enable_fp_fusion=False,
    )


def _launch_live_reduce(partial, z, m, groups, n, splits):
    _sonar()._sm120_wo_a_reduce[(triton.cdiv(groups * m * n, 256),)](
        partial, z, m, groups, n, splits, num_warps=4
    )


def _launch_rq(partial, m, n, splits):
    padded_m = (m + 127) // 128 * 128
    sf_cols = n // 32
    q = torch.empty((m, n), dtype=torch.uint8, device=partial.device)
    sf = torch.empty((padded_m * sf_cols,), dtype=torch.uint8, device=partial.device)
    _r2_reduce_quant[(n // 256, m)](
        partial, q, sf, m, n, splits, padded_m, sf_cols, INV448_BITS, True,
        num_warps=4, enable_fp_fusion=False, launch_pdl=True,
    )
    return q.view(torch.float8_e4m3fn), sf


def _live_quant(z2d):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    return mxfp8_e4m3_quantize(z2d, is_sf_swizzled_layout=True)


def _qa(q, sf, m, n):
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    return QuantizedActivation(q, sf, torch.bfloat16, torch.Size((m, n)), kMxfp8Dynamic)


def _disable(part: str, why: str) -> None:
    _ok[part] = False
    logger.error("[%s] DSV41_ATTNOUT_OP part %r DISABLED for this process: device bits differ from the "
                 "live kernels (%s). The live kernels run instead.", TAG, part, why)


def _run(ctx: Ctx, a, weight, scale, z) -> bool:
    global _checks_left, _logged
    m = a.shape[0]
    groups, n, k = weight.shape
    sonar = _sonar()
    splits = sonar._splits_for(m)
    fp8w = weight.dtype == torch.float8_e4m3fn
    capturing = _capturing()
    want_new_partial = (ctx.rope is not None or TUNE) and _ok["partial"] is not False
    want_rq = (
        RQ
        and _ok["rq"] is not False
        and groups == 1
        and splits == 32
        and n % 256 == 0
        and z.shape == (m, 1, n)
        and _rq_layer_ok(ctx.wo_b)
    )
    can_check = (not capturing) and _checks_left > 0
    if not can_check:
        # A part runs unchecked only after it passed a check: inside a capture (never the first use) and
        # on eager calls after the check budget is spent, unchecked parts fall back to the live kernels.
        if want_new_partial and _ok["partial"] is not True:
            want_new_partial = False
        if want_rq and _ok["rq"] is not True:
            want_rq = False
    if ctx.rope is not None and not want_new_partial:
        a = _live_inv_rope(a, ctx.inv_args)  # the folded RoPE is not usable: run the live kernel
        rope = None
    else:
        rope = ctx.rope
    check = can_check and (want_new_partial or want_rq)
    if check:
        _checks_left -= 1
        _self_check_synthetic(ctx, a if rope is not None else None, weight, scale, rope, want_new_partial,
                              want_rq)
        if _ok["partial"] is False:
            if rope is not None:
                a = _live_inv_rope(a, ctx.inv_args)
                rope = None
            want_new_partial = False
        if _ok["rq"] is False:
            want_rq = False

    partial = torch.empty((splits, groups, m, n), device=a.device, dtype=torch.float32)
    if check and want_new_partial:
        # Real-data self-check; any exception fails closed (part disabled, live kernels for this call).
        a_ref = _live_inv_rope(a, ctx.inv_args) if rope is not None else a
        try:
            _launch_partial(a, rope, weight, scale, partial, m, groups, n, k, splits, fp8w, new=True)
            ref = torch.empty_like(partial)
            _launch_partial(a_ref, None, weight, scale, ref, m, groups, n, k, splits, fp8w, new=False)
            if _bits_equal(partial, ref):
                _ok["partial"] = True
            else:
                _disable("partial", f"real call M={m}: {_first_diff(partial, ref)}")
                partial = ref
        except Exception as e:  # noqa: BLE001
            _disable("partial", f"real call M={m} raised {e!r}")
            _launch_partial(a_ref, None, weight, scale, partial, m, groups, n, k, splits, fp8w, new=False)
    else:
        _launch_partial(a, rope, weight, scale, partial, m, groups, n, k, splits, fp8w, new=want_new_partial)
    if want_rq and check:
        try:
            q, sf = _launch_rq(partial, m, n, splits)
            _launch_live_reduce(partial, z, m, groups, n, splits)
            q_ref, sf_ref = _live_quant(z.flatten(1))
            if _bits_equal(q, q_ref) and _bits_equal(sf, sf_ref):
                _ok["rq"] = True
            else:
                _disable("rq", f"real call M={m}: q {_first_diff(q, q_ref)}, sf {_first_diff(sf, sf_ref)}")
        except Exception as e:  # noqa: BLE001
            _disable("rq", f"real call M={m} raised {e!r}")
        if _ok["rq"] is not True:
            _launch_live_reduce(partial, z, m, groups, n, splits)  # live z (recomputed: same kernel, same bits)
            ctx.out = z.flatten(1)  # wo_b quantizes z itself, as on the live path
            return True
        ctx.out = _qa(q, sf, m, n)
    elif want_rq:
        q, sf = _launch_rq(partial, m, n, splits)  # only reached after rq passed a check
        ctx.out = _qa(q, sf, m, n)
    else:
        _launch_live_reduce(partial, z, m, groups, n, splits)
        ctx.out = z.flatten(1)
    if not _logged and not capturing:
        _logged = True
        logger.info("[%s] DSV41_ATTNOUT_OP=%s active: rope=%s rq=%s st1=%s vs=%s bn128=%s (M=%d splits=%d "
                    "fp8w=%s); self-check budget %d eager calls", TAG, ",".join(sorted(PARTS)),
                    rope is not None, want_rq, ST1, VS, BN128, m, splits, fp8w, CHECK_CALLS)
    return True


def _self_check_synthetic(ctx, o, weight, scale, rope, want_new_partial, want_rq) -> None:
    """Once per process, first eager call: hostile synthetic inputs on the real weights / rope cache."""
    global _synthetic_done
    if _synthetic_done:
        return
    _synthetic_done = True
    try:
        groups, n, k = weight.shape
        fp8w = weight.dtype == torch.float8_e4m3fn
        dev = weight.device
        sonar = _sonar()
        for m in (1, 5, 16, 32):
            splits = sonar._splits_for(m)
            if k % (128 * splits):
                continue
            if want_new_partial and _ok["partial"] is not False:
                gen = torch.Generator(device="cpu").manual_seed(1234 + m)
                if rope is not None:
                    positions, cs, hpg, nope, half = rope
                    hd = nope + 2 * half
                    x = _hostile_bf16((m, groups * hpg, hd), gen).to(dev)
                    pos = torch.randint(0, cs.shape[0], (m,), generator=gen).to(dev)
                    rope_m = (pos, cs, hpg, nope, half)
                    inv = dict(ctx.inv_args, positions=pos)
                    a_ref = _live_inv_rope(x, inv)
                    a_new = x
                else:
                    a_ref = _hostile_bf16((m, groups, k), gen).to(dev)
                    a_new, rope_m = a_ref, None
                p_new = torch.empty((splits, groups, m, n), device=dev, dtype=torch.float32)
                p_ref = torch.empty_like(p_new)
                _launch_partial(a_new, rope_m, weight, scale, p_new, m, groups, n, k, splits, fp8w, True)
                _launch_partial(a_ref, None, weight, scale, p_ref, m, groups, n, k, splits, fp8w, False)
                if not _bits_equal(p_new, p_ref):
                    _disable("partial", f"synthetic M={m}: {_first_diff(p_new, p_ref)}")
            if want_rq and _ok["rq"] is not False and splits == 32 and groups == 1 and n % 256 == 0:
                gen = torch.Generator(device="cpu").manual_seed(4321 + m)
                part = _hostile_partials((splits, 1, m, n), gen).to(dev)
                z = torch.empty((m, 1, n), device=dev, dtype=torch.bfloat16)
                _launch_live_reduce(part, z, m, 1, n, splits)
                q_ref, sf_ref = _live_quant(z.flatten(1))
                q, sf = _launch_rq(part, m, n, splits)
                if not (_bits_equal(q, q_ref) and _bits_equal(sf, sf_ref)):
                    _disable("rq", f"synthetic M={m}: q {_first_diff(q, q_ref)}, sf {_first_diff(sf, sf_ref)}")
    except Exception as e:  # noqa: BLE001
        logger.error("[%s] synthetic self-check raised %r: disabling all parts", TAG, e)
        _ok["partial"] = False
        _ok["rq"] = False


def _hostile_bf16(shape, gen) -> torch.Tensor:
    """NaN/Inf-free bf16 with a wide exponent range, exact zeros, -0, ties and +-max-ish values."""
    x = torch.randn(shape, generator=gen) * torch.exp2(torch.randint(-20, 12, shape, generator=gen).float())
    flat = x.view(-1)
    nz = flat.numel()
    idx = torch.randint(0, nz, (max(1, nz // 50),), generator=gen)
    flat[idx] = 0.0
    idx = torch.randint(0, nz, (max(1, nz // 50),), generator=gen)
    flat[idx] = -0.0
    idx = torch.randint(0, nz, (max(1, nz // 100),), generator=gen)
    flat[idx] = 3.0e4
    return x.to(torch.bfloat16)


def _hostile_partials(shape, gen) -> torch.Tensor:
    """fp32 partials: wide range, cancellation, zero blocks, values that round to bf16 ties; no denormal
    block maxima (that corner depends on CuTe-DSL FTZ choices and is covered by device_test.py)."""
    x = torch.randn(shape, generator=gen) * torch.exp2(torch.randint(-12, 12, shape, generator=gen).float())
    # cancellation: split 1 = -split 0 on a quarter of the columns
    n = shape[-1]
    x[1, ..., : n // 4] = -x[0, ..., : n // 4]
    # an all-zero 32-block per row
    x[..., 32:64] = 0.0
    # bf16 tie values (exactly halfway between two bf16 numbers) on split 0, zero elsewhere
    x[:, ..., 64:96] = 0.0
    x[0, ..., 64:96] = 1.0 + 2.0 ** -8
    return x.float()
