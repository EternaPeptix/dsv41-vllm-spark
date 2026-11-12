# SPDX-License-Identifier: Apache-2.0
"""DSV41 shared-expert SiLU+MXFP8-quant fold — producer glue for the serving path.

Proven bitwise-equal to the stock chain (act_and_mul -> MXFP8QuantizeSwizzled) on
384/384 hostile cells incl. codes, scales and the down-GEMM output (GOAL-130 SS13):
  act   = bf16(fp32(bf16(g/(1+expf(-g)))) * up32)   # vLLM packed_silu_kernel double rounding
  amax  = max|act| per 32-block, floored at fp32-tiny
  e     = clamp(ceil(log2(amax/448))+127, 0, 254)
  q     = act * exp2(127-e)                          # FTZ-safe exact rescale (never divide)
  codes = q -> tl.float8e4nv (RN, saturating)
  scales= e bytes in the F8_128x4 swizzle the CUTLASS mxfp8 kernel consumes
Wiring: gated by DSV41_SHARED_ACT_QUANT_FOLD=1 (read once at import). With the flag
off, or when any guard fails (no gate_up_proj/act_fn/down_proj attrs, consumer key
!= kMxfp8Dynamic, unexpected shape/dtype), the stock layer path runs unchanged.
"""
import os

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

_FOLD_ENABLED = os.environ.get("DSV41_SHARED_ACT_QUANT_FOLD", "") == "1"
_logged_once: set[str] = set()


def _log_once(tag: str) -> None:
    if tag not in _logged_once:
        _logged_once.add(tag)
        print(f"[dsv41-shared-fold] {tag}", flush=True)


@triton.jit
def _shared_silu_quant_kernel(
    x_ptr, q_ptr, s_ptr, M,
    N: tl.constexpr, NUM_K_T: tl.constexpr, K_SCALE_PAD: tl.constexpr,
    LIMIT: tl.constexpr, FORWARD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs = pid_b * 32 + tl.arange(0, 32)
    base = x_ptr + pid_m.to(tl.int64) * (2 * N)
    gate = tl.load(base + offs).to(tl.float32)
    up = tl.load(base + N + offs).to(tl.float32)
    if FORWARD == 1:
        # SiluAndMulWithClamp (the serving shared-expert act, ckpt swiglu_limit=10.0):
        # clamp gate to +LIMIT and up to +-LIMIT in f32 (exact: bf16 inputs are
        # f32-exact and LIMIT is bf16-representable), then the vLLM packed chain:
        # silu (expf, division form), bf16 round, fp32 multiply, bf16 round.
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    # vLLM packed_silu_kernel exactly: fp32 division-form silu (expf), bf16 round,
    # fp32 multiply by up, bf16 round (double rounding, gate-proven).
    silu32 = gate / (1.0 + tl.math.exp(-gate))
    act = (silu32.to(tl.bfloat16).to(tl.float32) * up).to(tl.bfloat16).to(tl.float32)
    a = tl.abs(act)
    amax = tl.max(a, axis=0)
    amax = tl.maximum(amax, 1.1754943508222875e-38)
    le = tl.math.log2(amax / 448.0)
    e = tl.ceil(le) + 127.0
    e = tl.minimum(tl.maximum(e, 0.0), 254.0)
    q = act * tl.exp2(127.0 - e)  # FTZ-safe exact rescale
    qb = q.to(tl.float8e4nv)
    tl.store(q_ptr + pid_m.to(tl.int64) * N + offs, qb)
    mt = pid_m // 128
    i4 = (pid_m % 128) // 32
    i32 = pid_m % 32
    kt = pid_b // 4
    k4 = pid_b % 4
    soff = ((((mt * NUM_K_T + kt) * 32 + i32) * 4 + i4) * 4) + k4
    tl.store(s_ptr + soff, e.to(tl.uint8))


def _shared_silu_quant(x_gate_up: torch.Tensor, limit: float | None = None):
    """[M, 576] bf16 -> ([M, 288] e4m3 codes, flat uint8 F8_128x4 scales).

    limit=None: plain SiluAndMul (FORWARD=0). limit=L: SiluAndMulWithClamp
    semantics (clamp gate to +L, up to +-L) before the silu (FORWARD=1)."""
    M, two_n = x_gate_up.shape
    N = two_n // 2
    dev = x_gate_up.device
    codes = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=dev)
    num_k_t = (N // 32 + 3) // 4
    padded_m = ((M + 127) // 128) * 128          # F8_128x4 tile rows, any M
    scales = torch.zeros(padded_m * (num_k_t * 4), dtype=torch.uint8, device=dev)
    ll = float(limit)
    fwd = 1 if limit is not None else 0
    lim = ll if fwd else 0.0
    _shared_silu_quant_kernel[(M, N // 32)](
        x_gate_up, codes, scales, M,
        N=N, NUM_K_T=num_k_t, K_SCALE_PAD=num_k_t * 4,
        LIMIT=lim, FORWARD=fwd,
        num_warps=1, num_stages=1,
    )
    return codes, scales


def warm(dev: torch.device = None) -> None:
    """JIT-prime all decode Ms. Call OUTSIDE capture (install window or eager warmup)."""
    if not _FOLD_ENABLED:
        return
    dev = dev or torch.device("cuda", torch.cuda.current_device())
    for m in range(1, 9):
        _shared_silu_quant(torch.randn(m, 576, device=dev, dtype=torch.bfloat16))
    _log_once(f"JIT primed (shapes M=1..8) on {dev}")


def maybe_fold_forward(layer: torch.nn.Module, x: torch.Tensor):
    """Take the shared-expert MLP layer through the fused path, or None to run stock.

    All static guards (attrs, SiluAndMul act, down_proj consumer key) run BEFORE any
    compute, so the fallback is the pristine stock path with no recomputation. The
    only dynamic guard (gate_up output dtype/shape/contiguity) is checked after the
    gate_up GEMM against known decode shapes; a mismatch logs once and the stock
    layer forward re-runs (bitwise-identical result, only wasted compute on a path
    that should never match in production).
    """
    if not _FOLD_ENABLED:
        return None
    gate_up_proj = getattr(layer, "gate_up_proj", None)
    down_proj = getattr(layer, "down_proj", None)
    act_fn = getattr(layer, "act_fn", None)
    if (gate_up_proj is None or down_proj is None
            or not isinstance(act_fn, (SiluAndMul, SiluAndMulWithClamp))):
        _log_once(f"fold: layer attrs not recognised (act={type(act_fn).__name__}) -> stock path")
        return None
    limit = None
    if isinstance(act_fn, SiluAndMulWithClamp):
        if act_fn.alpha != 1.0 or act_fn.beta != 0.0:
            _log_once("fold: act alpha/beta not default -> stock path")
            return None
        limit = act_fn.swiglu_limit
    if get_input_quant_key(down_proj) != kMxfp8Dynamic:
        _log_once("fold: down_proj consumer key != kMxfp8Dynamic -> stock path")
        return None
    g0 = gate_up_proj(x)
    g = g0[0] if isinstance(g0, tuple) else g0   # linears return (output, bias)
    if not isinstance(g, torch.Tensor):
        _log_once("fold: gate_up output not a tensor -> stock path")
        return None
    two_n = g.shape[-1]
    if (g.dtype != torch.bfloat16 or two_n % 64 != 0
            or two_n // 2 % 32 != 0 or not g.is_contiguous()):
        _log_once(f"fold: unsupported gate_up {two_n}x{g.dtype} -> stock path")
        return None
    g2 = g.reshape(-1, two_n)
    codes, scales = _shared_silu_quant(g2, limit)
    _log_once(f"fold engaged (M={g2.shape[0]}, N={two_n}, clamp={limit})")
    d = two_n // 2
    act_shape = (*g.shape[:-1], d)
    qa = QuantizedActivation(
        data=codes if len(act_shape) == 2 else codes.view(act_shape),
        scale=scales,
        orig_dtype=torch.bfloat16,
        orig_shape=act_shape,
        quant_key=kMxfp8Dynamic,
    )
    o0 = down_proj(qa)
    out = o0[0] if isinstance(o0, tuple) else o0
    return out.view(g.shape[:-1] + (out.shape[-1],))
