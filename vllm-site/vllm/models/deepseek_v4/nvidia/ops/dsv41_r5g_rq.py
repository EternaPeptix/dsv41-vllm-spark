# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 climb r5 (r5-glue), lever RQPAD: _r2_reduce_quant without the per-row padding-scale loop.

Imported ONLY when DSV41_GLUE_RQPAD is on, by the tagged lines at the end of dsv41_attnout.py (which itself is
imported only when DSV41_ATTNOUT_OP is set). Those lines rebind dsv41_attnout._launch_rq to wrap(_launch_rq), so
every caller (_run, _self_check_synthetic) reaches this module; with the flag unset nothing here is imported and
the r2 kernel runs exactly as today.

What changes: the r2 kernel ends with

    for p in range(M + r, PADDED_M, M):          # one 8-byte UE8M0-zero store per padding row
        tl.store(SF + offp(p), zeros[8])

i.e. 127 / 63 / 31 / 25 / 21 dependent loop trips for M = 1 / 2 / 4 / 5 / 6 (PADDED_M = 128). The t2 traces show
the cost: kernel p50 8.70 / 5.34 / 4.16 / 3.46 / 3.55 us at M = 1 / 2 / 4 / 5 / 6. _r5g_reduce_quant writes the
same zero bytes to the same addresses as ONE masked [PAD_J, 8] store per PAD_J * M rows (one trip for every
M <= 128 with PAD_J = 128). Everything before that store is the r2 kernel's body verbatim (tests/test_ast_exact.py
checks it statement by statement), so every q byte and every non-padding scale byte comes from the identical op
sequence; padding bytes are constant zeros. Exact by construction; device_test.py proves the bytes.

Runtime self-check (eager, non-capturing calls only, like dsv41_attnout's):
  * synthetic sweep (review N2), once per process at the first eager call: r2 kernel vs this kernel at EVERY sonar
    M = 1..32 on hostile synthetic partials (dsv41_attnout._hostile_partials: wide range, cancellation, zero blocks,
    bf16 ties), q + every SF byte, this kernel's buffers 0xFF-poisoned. Production's first eager call is at M=2 and
    the decode sizes (M=4/6) then run only inside CUDA graphs, so without the sweep they would be served unchecked on
    the rank; with it the boot INFO line covers every M the graphs can contain;
  * real-data check: the first eager call at each M also runs both kernels on the real partials.
A mismatch or an exception disables RQPAD for the process (ERROR log) and the r2 kernel's outputs are returned. The
new kernel runs inside a CUDA-graph capture or on an unchecked eager call only after the sweep passed and no check
failed.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v4.nvidia.ops import dsv41_attnout as _ao
from vllm.models.deepseek_v4.nvidia.ops.dsv41_attnout import _ASM_FP8, _ASM_INV_SCALE, _ASM_UE8M0  # noqa: F401
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

TAG = "r5-glue"
_RAW = os.environ.get("DSV41_GLUE_RQPAD", "0").strip().lower()
if _RAW not in ("1", "on", "true"):
    raise ValueError(f"DSV41_GLUE_RQPAD={_RAW!r}: use 0 or 1")

PAD_J = 128  # padding rows per masked store: >= PADDED_M - M for every sonar M (1..32, PADDED_M = 128)
CHECK_CALLS = 32  # eager real-data self-checks per process (one per distinct M; sonar M range is 1..32)
SWEEP_M = tuple(range(1, 33))  # synthetic sweep: every sonar M (MAX_M = 32)
_POISON = 0xFF  # never a UE8M0 the kernel writes (max 254) and never a pad byte (0)

_ok: bool | None = None  # None = not checked yet, True = >= 1 check passed and none failed, False = disabled
_checked: set[int] = set()
_checks_left = CHECK_CALLS
_logged = False
_swept = False


@triton.jit(do_not_specialize=["M"])
def _r5g_reduce_quant(
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
    PAD_J: tl.constexpr,
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
    # r5-glue RQPAD: flashinfer writes UE8M0 0 for every padding row M..PADDED_M-1; program row r covers the rows
    # M+r, M+r+M, ... (the r2 loop's rows), here PAD_J of them per masked [PAD_J, 8] store.
    jj = tl.arange(0, PAD_J)
    for j0 in range(0, PADDED_M - M - r, PAD_J * M):
        p = M + r + j0 + jj * M
        offp = sf_col[None, :] + ((p % 32) * 16 + ((p % 128) // 32) * 4 + (p // 128) * (128 * SF_COLS))[:, None]
        tl.store(SF + offp, tl.zeros([PAD_J, 8], tl.uint8), mask=(p < PADDED_M)[:, None])


def _launch_new_into(partial, q, sf, m, n, splits):
    """The r5 kernel into caller buffers, with the r2 launch's exact grid / arguments / options (+ PAD_J)."""
    padded_m = (m + 127) // 128 * 128
    sf_cols = n // 32
    _r5g_reduce_quant[(n // 256, m)](
        partial, q, sf, m, n, splits, padded_m, sf_cols, _ao.INV448_BITS, True, PAD_J,
        num_warps=4, enable_fp_fusion=False, launch_pdl=True,
    )


def _launch_r2_into(partial, q, sf, m, n, splits):
    """dsv41_attnout._launch_rq's kernel launch, verbatim, into caller buffers (device_test / self-check)."""
    padded_m = (m + 127) // 128 * 128
    sf_cols = n // 32
    _ao._r2_reduce_quant[(n // 256, m)](
        partial, q, sf, m, n, splits, padded_m, sf_cols, _ao.INV448_BITS, True,
        num_warps=4, enable_fp_fusion=False, launch_pdl=True,
    )


def _alloc(partial, m, n):
    """The same two allocations, in the same order, as dsv41_attnout._launch_rq."""
    padded_m = (m + 127) // 128 * 128
    sf_cols = n // 32
    q = torch.empty((m, n), dtype=torch.uint8, device=partial.device)
    sf = torch.empty((padded_m * sf_cols,), dtype=torch.uint8, device=partial.device)
    return q, sf


def _compare(orig, partial, m, n, splits):
    """r2 outputs (via the original _launch_rq) vs this kernel on 0xFF-poisoned buffers -> (q_ref, sf_ref, q, sf, why);
    why == "" when every q and SF byte is identical."""
    q_ref, sf_ref = orig(partial, m, n, splits)  # today's launch, today's allocations
    q, sf = _alloc(partial, m, n)
    q.fill_(_POISON)
    sf.fill_(_POISON)
    try:
        _launch_new_into(partial, q, sf, m, n, splits)
        same = _ao._bits_equal(q, q_ref.view(torch.uint8)) and _ao._bits_equal(sf, sf_ref)
        why = "" if same else (f"q {_ao._first_diff(q, q_ref.view(torch.uint8))}, sf {_ao._first_diff(sf, sf_ref)}")
    except Exception as e:  # noqa: BLE001  fail closed
        why = f"raised {e!r}"[:400]
    return q_ref, sf_ref, q, sf, why


def _sweep(orig, dev, n, splits):
    """Once per process, first eager call: r2 vs this kernel at every M in SWEEP_M on hostile synthetic partials."""
    global _ok, _swept
    _swept = True
    for m in SWEEP_M:
        gen = torch.Generator(device="cpu").manual_seed(5150 + m)
        try:
            part = _ao._hostile_partials((splits, 1, m, n), gen).to(dev)
            why = _compare(orig, part, m, n, splits)[4]
        except Exception as e:  # noqa: BLE001  fail closed (also an r2-side failure: never serve the kit unproven)
            why = f"raised {e!r}"[:400]
        if why:
            _ok = False
            logger.error("[%s] DSV41_GLUE_RQPAD DISABLED for this process: synthetic sweep M=%d differs from "
                         "_r2_reduce_quant (%s). The r2 kernel runs instead.", TAG, m, why)
            return False
    if _ok is None:
        _ok = True
    logger.info("[%s] DSV41_GLUE_RQPAD=1: _r5g_reduce_quant bit-identical to _r2_reduce_quant at every M %d..%d "
                "(synthetic sweep, hostile partials, q + all SF bytes, N=%d, SPLITS=%d); pad-store kernel ON", TAG,
                SWEEP_M[0], SWEEP_M[-1], n, splits)
    return True


def _check(orig, partial, m, n, splits):
    """Eager only: r2 outputs (via the original _launch_rq) vs this kernel on poisoned buffers. Returns (q, sf)."""
    global _ok, _checks_left, _logged
    _checks_left -= 1
    _checked.add(m)
    q_ref, sf_ref, q, sf, why = _compare(orig, partial, m, n, splits)
    same = not why
    if not same:
        _ok = False
        logger.error("[%s] DSV41_GLUE_RQPAD DISABLED for this process: M=%d differs from _r2_reduce_quant (%s). "
                     "The r2 kernel runs instead.", TAG, m, why)
        return q_ref, sf_ref
    if _ok is None:
        _ok = True
    if not _logged:
        _logged = True
        logger.info("[%s] DSV41_GLUE_RQPAD=1: _r5g_reduce_quant bit-identical to _r2_reduce_quant at M=%d "
                    "(eager real-data self-check, %d left)", TAG, m, _checks_left)
    return q.view(torch.float8_e4m3fn), sf


def wrap(orig):
    """Replacement for dsv41_attnout._launch_rq (same signature and return contract)."""

    def _launch_rq(partial, m, n, splits):
        if _ok is False:
            return orig(partial, m, n, splits)
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing and not _swept and not _sweep(orig, partial.device, n, splits):
            return orig(partial, m, n, splits)
        if not capturing and _checks_left > 0 and m not in _checked:
            return _check(orig, partial, m, n, splits)
        if _ok is not True:  # never checked (e.g. first use inside a capture): today's kernel
            return orig(partial, m, n, splits)
        q, sf = _alloc(partial, m, n)
        _launch_new_into(partial, q, sf, m, n, splits)
        return q.view(torch.float8_e4m3fn), sf

    _launch_rq.__wrapped__ = orig
    _launch_rq.r5g = True
    return _launch_rq
