# SPDX-License-Identifier: Apache-2.0
"""climb r5 (r5-glue), lever RPSPLIT: a shorter _dsv41_rp_kernel (same bytes, same per-element op order).

Imported ONLY when DSV41_GLUE_RPSPLIT is on, by the tagged lines at the end of dsv41_routeprep.py (itself imported only
with DSV41_MOE_ROUTEPREP=1). Those lines rebind dsv41_routeprep._launch to wrap(_launch); fused_permute (the real launch)
and _precompile (the one-time warmup) both resolve _launch at call time, so they reach this module. With the flag unset
nothing here is imported and the r2 kernel runs exactly as today.

Why the r2 kernel is slow (t2 traces: 7.9-8.3 us p50 at every grid 6..36, 40 calls/step, all on the critical path, FC1
starts 0.38-0.45 us after it ends): every program is ONE route and walks a serial chain
    gdc_wait -> ids load -> histogram -> cumsum -> 4 reductions -> 5 x [load x chunk -> amax -> div -> encode -> store
    row bytes -> pack (smem layout conversions) -> store words] -> dead-row fill
The five chunk loads cannot move above the previous chunk's stores (x and the slot rows are distinct buffers, but the
compiler cannot prove it), so the five global-load latencies are serialised behind the plan and behind each other.

DSV41_GLUE_RPSPLIT=hoist   one program per route as today; the 5 chunk loads of the token row are issued right after
                           gdc_wait, before the plan, so they overlap the plan and each other. Chunks are then quantized
                           in the same order with the r2 loop body verbatim (_r5g_rp_chunk).
DSV41_GLUE_RPSPLIT=split   grid (M*TOPK, 5): program (route, ch) loads chunk ch right after gdc_wait, computes the same
                           plan, and quantizes only chunk ch. The route's other duties (inv_perm / layout entry,
                           segment padding of the expert's first route, strided dead-row fill, expert ends, optional FC1
                           prefetch) run only in the ch == 0 program, so every byte is written by exactly one program,
                           as before. The dead-row fill keeps nprog = num_programs(0), i.e. the r2 partition.
Per element nothing changes: each 128-element quant group is computed from its own 128 inputs by the same ops in the
same order (absmax, eps floor, IEEE div by 448, 1e-10 floor, ceil-pow2 bit math, IEEE reciprocal, one multiply, clamp,
integer e4m3 encode, packed-word shifts); the plan is integer math. Only the load issue point and the program a chunk
runs in move. tests/test_ast_exact.py checks the chunk body against the r2 loop statement by statement.

Scope: calls with M * TOPK <= 64 (one SLOTS bucket, M <= 10: every decode/verify size). Larger M (small prefill) keeps
the r2 kernel. Runtime self-check (eager, non-capturing only), each run comparing the r2 kernel and the new one written
into separately poisoned copies of all five outputs, every byte:
  * synthetic sweep (review N2), once per process at the first in-scope eager call: every M = 1..10 x 2 routings
    ("spread": distinct experts; "edge": only experts 0 and 383, segments longer than one 32-row block) on hostile x
    (per-group magnitudes 2^-30..2^30, a zero group, a tiny group), with the call's ids dtype. Production's first eager
    call is at M=2 and the decode sizes (M=4/6) then run only inside CUDA graphs: the sweep proves them on this rank;
  * real-data check: the first eager call at each M also compares both kernels on the call's own x / ids.
A mismatch or a compile/launch error disables RPSPLIT for the process (ERROR log) and the r2 kernel serves. The new
kernel runs inside a capture only after the sweep passed and no check failed.
"""
import os

import torch

from vllm.model_executor.layers.fused_moe import dsv41_routeprep as _rp
from vllm.model_executor.layers.fused_moe.dsv41_routeprep import _rp_prefetch_l2  # noqa: F401  (jit helper)
from vllm.triton_utils import tl, triton

TAG = "r5-glue"
_RAW = os.environ.get("DSV41_GLUE_RPSPLIT", "0").strip().lower()
if _RAW not in ("hoist", "split"):
    raise ValueError(f"DSV41_GLUE_RPSPLIT={_RAW!r}: use 0, hoist or split")
MODE = _RAW
SPLIT = MODE == "split"
NCH = _rp._H // (_rp._GB * _rp._GROUP)  # 5 chunks of 8 quant groups
MAX_SLOTS_NEW = 64  # M * TOPK <= 64 (M <= 10)
CHECK_CALLS = 16
SWEEP_M = tuple(range(1, MAX_SLOTS_NEW // _rp._TOPK + 1))  # 1..10
SWEEP_ROUTINGS = ("spread", "edge")
_POISON8 = 0xA5
_POISON32 = 0x5A5A5A5A

_ok = None  # None = unchecked, True = >= 1 check passed and none failed, False = disabled for the process
_checked = set()
_checks_left = CHECK_CALLS
_logged = False
_swept = False


def _log():
    from vllm.logger import init_logger

    return init_logger(__name__)


@triton.jit
def _r5g_offs(ch, GB: tl.constexpr, GROUP: tl.constexpr):
    g = tl.arange(0, GB)
    c = tl.arange(0, GROUP)
    return (ch * GB + g)[:, None] * GROUP + c[None, :]


@triton.jit
def _r5g_rp_chunk(xa, ch, orow, sfp, mn_aligned, dest, eps, fp8_min, fp8_max, GB: tl.constexpr,
                  GROUP: tl.constexpr):
    # dsv41_routeprep._dsv41_rp_kernel's `for ch in tl.static_range(...)` body for one chunk, verbatim except that the
    # bf16 tile xa was loaded earlier (same address: _r5g_offs(ch) == the r2 `offs`).
    g = tl.arange(0, GB)
    c = tl.arange(0, GROUP)
    col = tl.arange(0, 4)[None, :]
    fmax_v = tl.zeros([GB], dtype=tl.float32) + fp8_max
    one_v = tl.full([GB], 1.0, tl.float32)
    offs = (ch * GB + g)[:, None] * GROUP + c[None, :]
    xf = xa.to(tl.float32)
    # absmax with the eps floor (exact, order-free), IEEE div by fp8_max, 1e-10 floor
    amax = tl.maximum(tl.max(tl.abs(xf), axis=1), eps)
    y = tl.math.div_rn(amax, fmax_v)
    y = tl.maximum(tl.abs(y), 1e-10)
    # UE8M0 scale by bit-math ceil(log2): exponent field, +1 if any mantissa bit
    yb = y.to(tl.int32, bitcast=True)
    sb = yb & 0x7F800000
    sb = tl.where((yb & 0x007FFFFF) != 0, sb + 0x00800000, sb)
    scale = sb.to(tl.float32, bitcast=True)
    inv = tl.math.div_rn(one_v, scale)  # IEEE reciprocal of a power of two: exact 2^-k
    q = xf * inv[:, None]
    q = tl.minimum(tl.maximum(q, fp8_min), fp8_max)
    # e4m3fn encode, the live kernel's integer/fp32 sequence (satfinite c10-style RNE)
    qb = q.to(tl.int32, bitcast=True)
    a = qb & 0x7FFFFFFF
    sign = (qb >> 24) & 0x80
    odd = (a >> 20) & 1
    tn = a + 0x0407FFFF + odd
    mag_n = tl.where((tn & 0x0FF00000) != 0x07F00000, tn >> 20, 0x7E)
    fs = a.to(tl.float32, bitcast=True) + 16384.0
    mag_s = fs.to(tl.int32, bitcast=True)
    mag_o = tl.where(a > 0x7F800000, 0x7F, 0x7E)
    mag = tl.where(a >= 0x43F00000, mag_o, tl.where(a >= 0x3C800000, mag_n, mag_s))
    byte = (mag & 0xFF) | sign
    tl.store(orow + offs, byte.to(tl.uint8))
    # packed word = (v0 >> 23) | (v1 >> 15) | (v2 >> 7) | (v3 << 1), as transpose_and_pack / mfront
    sb2 = tl.reshape(sb, (GB // 4, 4))
    v0 = tl.sum(tl.where(col == 0, sb2, 0), axis=1)
    v1 = tl.sum(tl.where(col == 1, sb2, 0), axis=1)
    v2 = tl.sum(tl.where(col == 2, sb2, 0), axis=1)
    v3 = tl.sum(tl.where(col == 3, sb2, 0), axis=1)
    packed = (v0 >> 23) | (v1 >> 15) | (v2 >> 7) | (v3 << 1)
    p = ch * (GB // 4) + tl.arange(0, GB // 4)
    tl.store(sfp + p * mn_aligned + dest, packed)


@triton.jit
def _r5g_rp_kernel(
    x_ptr, x_row_stride, topk_ids, ids_stride0, ids_stride1, out_ptr, out_row_stride, sfp, mn_aligned, layout,
    inv_perm, inv_stride0, inv_stride1, ends, w1_ptr, w1_s0, w1_s1, sf_ptr, sf_s0, sf_s1, sf_s2, M, M_sum, eps,
    fp8_min, fp8_max,
    E: tl.constexpr, TOPK: tl.constexpr, ALIGN: tl.constexpr, H: tl.constexpr, GROUP: tl.constexpr,
    GB: tl.constexpr, PACKS: tl.constexpr, SLOTS: tl.constexpr, E_PAD: tl.constexpr, FILL_ROWS: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr, PF_KB: tl.constexpr, PF_N: tl.constexpr, PF_NPAD: tl.constexpr,
    PF_SPAD: tl.constexpr, PF_SF_STEP: tl.constexpr, KB_BYTES: tl.constexpr, launch_pdl: tl.constexpr,
    SPLIT: tl.constexpr,
):
    tl.static_assert(H // (GB * GROUP) == 5, "r5-glue RPSPLIT is written for 5 chunks (H=5120, GB=8, GROUP=128)")
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    y = tl.program_id(1)  # SPLIT: the chunk; hoist: grid dim 1 is 1, so y == 0
    is0 = y == 0
    if launch_pdl:
        # Everything this kernel reads (topk_ids from the top-k kernel, x) is final after this wait, and nothing is
        # written before it.
        tl.extra.cuda.gdc_wait()

    # ---- r5-glue: issue the token row's x loads first (their latency overlaps the plan below)
    t = pid // TOPK
    k = pid % TOPK
    xrow = x_ptr + t.to(tl.int64) * x_row_stride
    if SPLIT:
        xa = tl.load(xrow + _r5g_offs(y, GB, GROUP))
    else:
        xa0 = tl.load(xrow + _r5g_offs(0, GB, GROUP))
        xa1 = tl.load(xrow + _r5g_offs(1, GB, GROUP))
        xa2 = tl.load(xrow + _r5g_offs(2, GB, GROUP))
        xa3 = tl.load(xrow + _r5g_offs(3, GB, GROUP))
        xa4 = tl.load(xrow + _r5g_offs(4, GB, GROUP))

    # ---- plan: the same integer formulas as _dsv41_fp_plan_kernel (histogram, 32-aligned exclusive cumsum)
    s = tl.arange(0, SLOTS)
    sm = s < M * TOPK
    ids = tl.load(topk_ids + (s // TOPK) * ids_stride0 + (s % TOPK) * ids_stride1, mask=sm, other=-1).to(tl.int32)
    valid = sm & (ids >= 0) & (ids < E)
    cnt = tl.histogram(tl.where(valid, ids, E), E_PAD)
    b = tl.arange(0, E_PAD)
    cnt = tl.where(b < E, cnt, 0)
    aligned = (cnt + ALIGN - 1) // ALIGN * ALIGN
    start = tl.cumsum(aligned, axis=0) - aligned
    total = tl.sum(aligned, axis=0)

    e = tl.load(topk_ids + t * ids_stride0 + k * ids_stride1).to(tl.int32)
    live = (e >= 0) & (e < E)
    sel = b == e
    start_e = tl.sum(tl.where(sel, start, 0), axis=0)
    cnt_e = tl.sum(tl.where(sel, cnt, 0), axis=0)
    al_e = tl.sum(tl.where(sel, aligned, 0), axis=0)
    # deterministic slot: routes of expert e take its segment rows in (token, k) order
    rank = tl.sum(tl.where(valid & (s < pid) & (ids == e), 1, 0), axis=0)
    dest = start_e + rank
    first = live & (rank == 0)

    # ---- lever 2 (DSV41_MOE_FC1_FRONT=pf<N>): L2 prefetch of the selected expert's first FC1 k-blocks (ch-0 program)
    if PF_KB > 0:
        if first & is0:
            n = tl.minimum(tl.arange(0, PF_NPAD), PF_N - 1)
            wrow = w1_ptr + e.to(tl.int64) * w1_s0 + n.to(tl.int64) * w1_s1
            for j in tl.static_range(PF_KB):
                _rp_prefetch_l2(wrow + j * KB_BYTES)
            nsf = tl.minimum(tl.arange(0, PF_SPAD) * PF_SF_STEP, PF_N - 1)
            srow = sf_ptr + e.to(tl.int64) * sf_s0 + nsf.to(tl.int64) * sf_s1
            for j in tl.static_range(PF_KB):
                _rp_prefetch_l2(srow + j * sf_s2)
    if EARLY_TRIGGER:
        # FC1 (PDL-launched) may now be scheduled; its cudaGridDependencySynchronize still waits for this grid's
        # completion and memory flush before it reads layout / rows / scales.
        tl.extra.cuda.gdc_launch_dependents()

    # ---- live route: quantize token row t into slot dest (r2 op sequence per chunk, _r5g_rp_chunk)
    if live:
        if is0:
            tl.store(inv_perm + t * inv_stride0 + k * inv_stride1, dest)
            tl.store(layout + dest, e)
        orow = out_ptr + dest.to(tl.int64) * out_row_stride
        if SPLIT:
            _r5g_rp_chunk(xa, y, orow, sfp, mn_aligned, dest, eps, fp8_min, fp8_max, GB, GROUP)
        else:
            _r5g_rp_chunk(xa0, 0, orow, sfp, mn_aligned, dest, eps, fp8_min, fp8_max, GB, GROUP)
            _r5g_rp_chunk(xa1, 1, orow, sfp, mn_aligned, dest, eps, fp8_min, fp8_max, GB, GROUP)
            _r5g_rp_chunk(xa2, 2, orow, sfp, mn_aligned, dest, eps, fp8_min, fp8_max, GB, GROUP)
            _r5g_rp_chunk(xa3, 3, orow, sfp, mn_aligned, dest, eps, fp8_min, fp8_max, GB, GROUP)
            _r5g_rp_chunk(xa4, 4, orow, sfp, mn_aligned, dest, eps, fp8_min, fp8_max, GB, GROUP)

    # ---- dead rows only (no row is written twice): segment padding by the expert's first route, tail strided
    if first & is0:
        r = start_e + cnt_e + tl.arange(0, ALIGN)
        rm = r < start_e + al_e
        tl.store(layout + r, tl.full([ALIGN], -1, tl.int32), mask=rm)
        for pk in tl.static_range(PACKS):
            tl.store(sfp + pk * mn_aligned + r, tl.zeros([ALIGN], dtype=tl.int32), mask=rm)
    if is0:
        rr = tl.arange(0, FILL_ROWS)
        for r0 in range(total + pid * FILL_ROWS, M_sum, nprog * FILL_ROWS):
            r = r0 + rr
            rm = r < M_sum
            tl.store(layout + r, tl.full([FILL_ROWS], -1, tl.int32), mask=rm)
            for pk in tl.static_range(PACKS):
                tl.store(sfp + pk * mn_aligned + r, tl.zeros([FILL_ROWS], dtype=tl.int32), mask=rm)
    if (pid == 0) & is0:
        tl.store(ends + b, (start + cnt).to(tl.int32), mask=b < E)


def _launch_new(x, x_row_stride, ids, ids_s0, ids_s1, out_u8, out_row_stride, sfp, mn_aligned, layout, inv_perm,
                inv_s0, inv_s1, ends, w1, sf, M, M_sum, warmup=False, split=None):
    """dsv41_routeprep._launch's arguments / constexprs / options, verbatim, plus SPLIT and the (M*TOPK, 5) grid."""
    split = SPLIT if split is None else split
    pf_kb, pf_n, sf_step, w_args = _rp._pf_args(w1, sf, _rp._E, layout)
    grid = (M * _rp._TOPK, NCH if split else 1)
    args = (x, x_row_stride, ids, ids_s0, ids_s1, out_u8, out_row_stride, sfp, mn_aligned, layout, inv_perm, inv_s0,
            inv_s1, ends, *w_args, M, M_sum, 1e-10, -448.0, 448.0)
    kw = dict(E=_rp._E, TOPK=_rp._TOPK, ALIGN=_rp._ALIGN, H=_rp._H, GROUP=_rp._GROUP, GB=_rp._GB,
              PACKS=_rp._H // _rp._GROUP // 4, SLOTS=max(64, triton.next_power_of_2(M * _rp._TOPK)),
              E_PAD=triton.next_power_of_2(_rp._E + 1), FILL_ROWS=_rp._FILL_ROWS,
              EARLY_TRIGGER=_rp.EARLY_TRIGGER and _rp._PDL, PF_KB=pf_kb, PF_N=pf_n,
              PF_NPAD=triton.next_power_of_2(pf_n), PF_SPAD=triton.next_power_of_2(triton.cdiv(pf_n, sf_step)),
              PF_SF_STEP=sf_step, KB_BYTES=_rp._FC1_KB_BYTES, SPLIT=split, num_warps=4, enable_fp_fusion=False,
              launch_pdl=_rp._PDL)
    if warmup:
        _r5g_rp_kernel.warmup(*args, grid=grid, **kw)
    else:
        _r5g_rp_kernel[grid](*args, **kw)
    return pf_kb


def _disable(why):
    global _ok
    _ok = False
    _log().error("[%s] DSV41_GLUE_RPSPLIT=%s DISABLED for this process: %s. The r2 route-prep kernel runs instead.",
                 TAG, MODE, why)


def _poisoned_outputs(out_u8, sfp, layout, inv_perm, ends, mn_aligned):
    packs = _rp._H // _rp._GROUP // 4
    o = torch.full_like(out_u8, _POISON8)
    s = torch.full((packs * mn_aligned,), _POISON32, dtype=torch.int32, device=out_u8.device)
    lay = torch.full_like(layout, _POISON32)
    inv = torch.full_like(inv_perm, _POISON32)
    en = torch.full_like(ends, _POISON32)
    return o, s, lay, inv, en


def _compare(orig, a):
    """r2 kernel vs new kernel on separately poisoned copies of all five outputs -> list of differences ([] = same)."""
    (x, xs, ids, i0, i1, out_u8, os_, sfp, mn, layout, inv, v0, v1, ends, w1, sf, M, M_sum) = a
    try:
        ref = _poisoned_outputs(out_u8, sfp, layout, inv, ends, mn)
        new = _poisoned_outputs(out_u8, sfp, layout, inv, ends, mn)
        orig(x, xs, ids, i0, i1, ref[0], os_, ref[1], mn, ref[2], ref[3], v0, v1, ref[4], w1, sf, M, M_sum)
        _launch_new(x, xs, ids, i0, i1, new[0], os_, new[1], mn, new[2], new[3], v0, v1, new[4], w1, sf, M, M_sum)
        names = ("rows", "scale words", "layout", "inv_perm", "ends")
        return [f"{nm}: {_first_diff(p, q)}" for nm, p, q in zip(names, ref, new) if not torch.equal(p, q)]
    except Exception as e:  # noqa: BLE001  compile / launch failure: fail closed
        return [f"raised {e!r}"[:400]]


def _synthetic(M, routing, ids_dtype, dev, w1, sf):
    """_launch arguments for a synthetic route-prep call at M, shaped and strided as fused_permute builds them."""
    H, T, E, G, A = _rp._H, _rp._TOPK, _rp._E, _rp._GROUP, _rp._ALIGN
    gen = torch.Generator(device="cpu").manual_seed(7000 + 10 * M + SWEEP_ROUTINGS.index(routing))
    mag = torch.exp2(torch.randint(-30, 31, (M, H // G, 1), generator=gen).float())
    x = (torch.randn((M, H // G, G), generator=gen) * mag).reshape(M, H)
    x[:, :G] = 0.0  # all-zero group: the eps floor
    x[:, G:2 * G] *= 2.0 ** -40  # tiny group: e4m3 subnormal codes
    x = x.to(torch.bfloat16).to(dev)
    if routing == "spread":
        ids = torch.randperm(E, generator=gen)[: M * T].view(M, T)
    else:
        ids = torch.randint(0, 2, (M, T), generator=gen) * (E - 1)
    ids = ids.to(ids_dtype).to(dev).contiguous()
    m_sum = M * T + min(M * T, E) * (A - 1)
    m_sum = (m_sum + A - 1) // A * A  # the fused path's 32-aligned M_sum
    mn = (m_sum + 3) // 4 * 4
    out = torch.empty((m_sum, H), dtype=torch.uint8, device=dev)
    sfp = torch.empty((H // G // 4 * mn,), dtype=torch.int32, device=dev)
    layout = torch.empty((m_sum,), dtype=torch.int32, device=dev)
    inv = torch.empty((M, T), dtype=torch.int32, device=dev)
    ends = torch.empty((E,), dtype=torch.int32, device=dev)
    return (x, x.stride(0), ids, ids.stride(0), ids.stride(1), out, out.stride(0), sfp, mn, layout, inv, inv.stride(0),
            inv.stride(1), ends, w1, sf, M, m_sum)


def _sweep(orig, a):
    """Once per process, first in-scope eager call: r2 vs new kernel at every M in SWEEP_M x SWEEP_ROUTINGS."""
    global _ok, _swept
    _swept = True
    x, ids, w1, sf = a[0], a[2], a[14], a[15]
    for M in SWEEP_M:
        for routing in SWEEP_ROUTINGS:
            try:
                bad = _compare(orig, _synthetic(M, routing, ids.dtype, x.device, w1, sf))
            except Exception as e:  # noqa: BLE001  fail closed
                bad = [f"raised {e!r}"[:400]]
            if bad:
                _disable(f"synthetic sweep M={M} ({routing} routing): " + "; ".join(bad))
                return False
    if _ok is None:
        _ok = True
    _log().info("[%s] DSV41_GLUE_RPSPLIT=%s: _r5g_rp_kernel bit-identical to _dsv41_rp_kernel at every M %d..%d "
                "(synthetic sweep, routings %s, %s ids, all 5 outputs); RPSPLIT ON for M*TOPK <= %d", TAG, MODE,
                SWEEP_M[0], SWEEP_M[-1], "/".join(SWEEP_ROUTINGS), str(ids.dtype).replace("torch.", ""),
                MAX_SLOTS_NEW)
    return True


def _check(orig, a):
    """Eager only: r2 kernel vs new kernel on the call's own inputs (separately poisoned copies of all five outputs)."""
    global _ok, _checks_left, _logged
    M = a[16]
    _checks_left -= 1
    _checked.add(M)
    bad = _compare(orig, a)
    if bad:
        _disable(f"eager self-check M={M}: " + "; ".join(bad))
        return False
    if _ok is None:
        _ok = True
    if not _logged:
        _logged = True
        _log().info("[%s] DSV41_GLUE_RPSPLIT=%s: _r5g_rp_kernel bit-identical to _dsv41_rp_kernel at M=%d (eager "
                    "real-data self-check, all 5 outputs, %d left)", TAG, MODE, M, _checks_left)
    return True


def _first_diff(a, b):
    a8 = a.contiguous().view(torch.uint8).flatten()
    b8 = b.contiguous().view(torch.uint8).flatten()
    idx = torch.nonzero(a8 != b8)
    return "none" if idx.numel() == 0 else f"byte {int(idx[0])} of {a8.numel()} ({idx.numel()} differ)"


def wrap(orig):
    """Replacement for dsv41_routeprep._launch (same signature and return value)."""

    def _launch(x, x_row_stride, ids, ids_s0, ids_s1, out_u8, out_row_stride, sfp, mn_aligned, layout, inv_perm,
                inv_s0, inv_s1, ends, w1, sf, M, M_sum, warmup=False):
        a = (x, x_row_stride, ids, ids_s0, ids_s1, out_u8, out_row_stride, sfp, mn_aligned, layout, inv_perm,
             inv_s0, inv_s1, ends, w1, sf, M, M_sum)
        in_scope = M * _rp._TOPK <= MAX_SLOTS_NEW
        if warmup:
            pf_kb = orig(*a, warmup=True)  # today's precompile, unchanged (its failure handling is the caller's)
            if in_scope and _ok is not False:
                try:
                    _launch_new(*a, warmup=True)
                except Exception as e:  # noqa: BLE001
                    _disable(f"precompile raised {e!r}"[:600])
            return pf_kb
        if not in_scope or _ok is False:
            return orig(*a)
        if not torch.cuda.is_current_stream_capturing():
            if not _swept and not _sweep(orig, a):
                return orig(*a)
            if _checks_left > 0 and M not in _checked and not _check(orig, a):
                return orig(*a)
        if _ok is not True:
            return orig(*a)
        return _launch_new(*a)

    _launch.__wrapped__ = orig
    _launch.r5g = True
    return _launch
