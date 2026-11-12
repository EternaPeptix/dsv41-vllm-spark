# SPDX-License-Identifier: Apache-2.0
"""climb r2 (r2-moefront): fused target-MoE route prep (G04 Stage A) + FC1 front (G18 FC1 part).

Imported ONLY when DSV41_MOE_ROUTEPREP is set (deep_gemm_moe.py / deep_gemm_utils.py gate the import).

DSV41_MOE_ROUTEPREP=1 (lever G04, moe:L-A Stage A)
    Target MoE layers (E=384, top-6, H=5120, M32 path, mfront serving) skip the FC1-input quant in prepare()
    (DeepGemmFP4Experts.expects_unquantized_inputs) and run ONE kernel instead of three:
        today:  per_token_group_quant_8bit_register_kernel<bf16,e4m3,false,true,128>  (x -> x_q, x_s)
                _dsv41_fp_plan_kernel   (layout=-1, packed scales=0 over M_sum rows, 32-aligned expert starts)
                _dsv41_fp_scatter_kernel (atomic slot, FP8 row copy, layout, inv_perm, packed UE8M0 word)
        fused:  _dsv41_rp_kernel, one program per (token, k) route, PDL-launched after the top-k kernel:
                gdc_wait; recompute the plan (same histogram / 32-aligned cumsum); slot = start[e] + prefix rank;
                quantize the token row with the live kernel's exact op sequence (EXACTNESS.md) straight into the slot;
                packed UE8M0 words with the live formula; dead rows only get layout -1 / scale word 0.
    The top-k kernel is untouched (Stage C, top-k fusion, is not built: its fp32 weight_sum tree and approximate
    instructions would have to be re-proven).
    Every call that does not qualify (prefill M > 96, other geometry, mfront off) quantizes in apply() with the very
    function prepare() would have called (_quantize_input) and then runs today's code unchanged.

DSV41_MOE_FC1_FRONT (lever G18 FC1 part, moe:L-F; needs DSV41_MOE_ROUTEPREP=1)
    0 (default)  nothing
    pdl          the fused kernel calls griddepcontrol.launch_dependents right after its gdc_wait, so the
                 (already PDL-launched) FC1 grid is scheduled and runs its pre-sync prologue while the route prep
                 finishes. FC1 reads nothing of ours before cudaGridDependencySynchronize (full completion + flush).
    pf<N>        pdl + an L2 prefetch (prefetch.global.L2) of the first N FC1 k-blocks (64 B FP4 each) of every
                 weight row and of the matching w1_scale words for each SELECTED expert, issued by the first route
                 of that expert right after top-k is known. N in {1, 2, 4, 8}. Cache hint only.
"""

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

TAG = "r2-moefront"

_RP = os.environ.get("DSV41_MOE_ROUTEPREP", "0").strip().lower()
if _RP not in ("1", "on", "true"):
    raise ValueError(f"DSV41_MOE_ROUTEPREP={_RP!r}: use 0 or 1")

_FRONT = os.environ.get("DSV41_MOE_FC1_FRONT", "0").strip().lower()
if _FRONT in ("", "0", "off", "false"):
    EARLY_TRIGGER, PF_KB = False, 0
elif _FRONT == "pdl":
    EARLY_TRIGGER, PF_KB = True, 0
elif _FRONT in ("pf1", "pf2", "pf4", "pf8"):
    EARLY_TRIGGER, PF_KB = True, int(_FRONT[2:])
else:
    raise ValueError(f"DSV41_MOE_FC1_FRONT={_FRONT!r}: use 0, pdl, pf1, pf2, pf4 or pf8")

_E = 384
_TOPK = 6
_H = 5120
_GROUP = 128
_ALIGN = 32
_MAX_SLOTS = 1024  # same cap as mfront (_DSV41_FP_SLOTS)
_GB = 8  # quant groups per chunk (2 packed scale words)
_FILL_ROWS = 64
_FC1_KB_BYTES = 64  # one FC1 k-block (BLOCK_K=128) of an FP4 weight row

_PDL = bool(current_platform.is_arch_support_pdl())

# per-process constants of the live prepare() quant call, checked once (first eligible-looking call, eager)
_STATIC = None
# static check + one-time Triton precompile of every SLOTS bucket of the variant in use (first eligible call, eager):
# None = not decided yet, True = fused path on, False = fused path off for the whole process (today's code runs)
_READY = None
_PRECOMPILE_M = (1, 16, 32, 64, 96)  # SLOTS buckets 64, 128, 256, 512, 1024
_PF_LOGGED = False


class Deferred:
    """Stands in for a1q_scale between DeepGemmFP4Experts.apply and deepgemm_moe_permute (fused path)."""

    __slots__ = ("w1", "w1_scale")

    def __init__(self, w1, w1_scale):
        self.w1 = w1
        self.w1_scale = w1_scale


@triton.jit
def _rp_prefetch_l2(ptrs):
    # Cache hint only: no register result is used, no memory is written.
    return tl.inline_asm_elementwise(
        "prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;", "=r,l", [ptrs],
        dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _dsv41_rp_kernel(
    x_ptr,  # bf16 [M, H] contiguous (the MoE input prepare() would have quantized)
    x_row_stride,
    topk_ids,
    ids_stride0,
    ids_stride1,
    out_ptr,  # uint8 view of the FP8 permuted rows [M_sum, H] (workspace13)
    out_row_stride,
    sfp,  # int32 packed UE8M0 scales, element (row r, pack p) at sfp + p * mn_aligned + r
    mn_aligned,
    layout,  # int32 [M_sum]
    inv_perm,  # int32 [M, TOPK]
    inv_stride0,
    inv_stride1,
    ends,  # int32 [E]: start + count (what the mfront cursor holds after the scatter)
    w1_ptr,  # FC1 packed FP4 weights (int8 view), lever 2 only
    w1_s0,
    w1_s1,
    sf_ptr,  # FC1 int32 weight scales, lever 2 only
    sf_s0,
    sf_s1,
    sf_s2,
    M,
    M_sum,
    eps,
    fp8_min,
    fp8_max,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    ALIGN: tl.constexpr,
    H: tl.constexpr,
    GROUP: tl.constexpr,
    GB: tl.constexpr,
    PACKS: tl.constexpr,
    SLOTS: tl.constexpr,
    E_PAD: tl.constexpr,
    FILL_ROWS: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
    PF_KB: tl.constexpr,
    PF_N: tl.constexpr,
    PF_NPAD: tl.constexpr,
    PF_SPAD: tl.constexpr,
    PF_SF_STEP: tl.constexpr,
    KB_BYTES: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if launch_pdl:
        # Everything this kernel reads (topk_ids from the top-k kernel, x) is final after this wait, and nothing is
        # written before it.
        tl.extra.cuda.gdc_wait()

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

    t = pid // TOPK
    k = pid % TOPK
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

    # ---- lever 2 (DSV41_MOE_FC1_FRONT=pf<N>): L2 prefetch of the selected expert's first FC1 k-blocks
    if PF_KB > 0:
        if first:
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

    # ---- live route: quantize token row t into slot dest (live per_token_group_quant op sequence, EXACTNESS.md)
    if live:
        tl.store(inv_perm + t * inv_stride0 + k * inv_stride1, dest)
        tl.store(layout + dest, e)
        xrow = x_ptr + t.to(tl.int64) * x_row_stride
        orow = out_ptr + dest.to(tl.int64) * out_row_stride
        g = tl.arange(0, GB)
        c = tl.arange(0, GROUP)
        col = tl.arange(0, 4)[None, :]
        fmax_v = tl.zeros([GB], dtype=tl.float32) + fp8_max
        one_v = tl.full([GB], 1.0, tl.float32)
        for ch in tl.static_range(H // (GB * GROUP)):
            offs = (ch * GB + g)[:, None] * GROUP + c[None, :]
            xf = tl.load(xrow + offs).to(tl.float32)
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

    # ---- dead rows only (no row is written twice): segment padding by the expert's first route, tail strided
    if first:
        r = start_e + cnt_e + tl.arange(0, ALIGN)
        rm = r < start_e + al_e
        tl.store(layout + r, tl.full([ALIGN], -1, tl.int32), mask=rm)
        for pk in tl.static_range(PACKS):
            tl.store(sfp + pk * mn_aligned + r, tl.zeros([ALIGN], dtype=tl.int32), mask=rm)
    rr = tl.arange(0, FILL_ROWS)
    for r0 in range(total + pid * FILL_ROWS, M_sum, nprog * FILL_ROWS):
        r = r0 + rr
        rm = r < M_sum
        tl.store(layout + r, tl.full([FILL_ROWS], -1, tl.int32), mask=rm)
        for pk in tl.static_range(PACKS):
            tl.store(sfp + pk * mn_aligned + r, tl.zeros([FILL_ROWS], dtype=tl.int32), mask=rm)
    if pid == 0:
        tl.store(ends + b, (start + cnt).to(tl.int32), mask=b < E)


def _static_ok() -> bool:
    """Process constants of the live prepare() quant: UE8M0 on, e4m3fn +-448 (per_token_group_quant_fp8 defaults)."""
    global _STATIC
    if _STATIC is None:
        from vllm.model_executor.layers.fused_moe import deep_gemm_utils as dgu
        from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        _STATIC = bool(
            is_deep_gemm_e8m0_used()
            and tuple(get_fp8_min_max()) == (-448.0, 448.0)
            and current_platform.fp8_dtype() == torch.float8_e4m3fn
            and dgu._DSV41_FP_MODE == "1"  # mfront serves today: same packed-scale / layout contract
        )
    return _STATIC


def _m_sum(M, block_m=_ALIGN):
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M_and_alignment

    return compute_aligned_M_and_alignment(M, _TOPK, _E, block_m, None)


def _pf_args(w1, sf, E, unused):
    """Lever-2 arguments: (pf_kb, pf_n, sf_step, w_args). Prefetch is switched off (pf_kb 0) when the FC1 weight /
    scale geometry is not the expected one; `unused` fills the pointer slots that PF_KB == 0 never touches."""
    pf_kb = PF_KB
    if pf_kb and not (w1 is not None and w1.dim() == 3 and w1.size(0) == E and w1.element_size() == 1
                      and w1.stride(2) == 1 and sf is not None and sf.dim() == 3 and sf.size(0) == E
                      and sf.size(1) == w1.size(1) and sf.dtype == torch.int32
                      and w1.size(2) >= pf_kb * _FC1_KB_BYTES and sf.size(2) >= pf_kb):
        pf_kb = 0
    if pf_kb:
        sf_step = 32 if sf.stride(1) == 1 else 1  # one prefetch per 128-B line when N is the contiguous dim
        return pf_kb, w1.size(1), sf_step, (w1, w1.stride(0), w1.stride(1), sf, sf.stride(0), sf.stride(1),
                                             sf.stride(2))
    return 0, 1, 1, (unused, 0, 0, unused, 0, 0, 0)


def _launch(x, x_row_stride, ids, ids_s0, ids_s1, out_u8, out_row_stride, sfp, mn_aligned, layout, inv_perm, inv_s0,
            inv_s1, ends, w1, sf, M, M_sum, warmup=False):
    """The one place that builds the kernel arguments: the real launch and the startup precompile use it alike."""
    pf_kb, pf_n, sf_step, w_args = _pf_args(w1, sf, _E, layout)
    grid = (M * _TOPK,)
    args = (x, x_row_stride, ids, ids_s0, ids_s1, out_u8, out_row_stride, sfp, mn_aligned, layout, inv_perm, inv_s0,
            inv_s1, ends, *w_args, M, M_sum, 1e-10, -448.0, 448.0)
    kw = dict(E=_E, TOPK=_TOPK, ALIGN=_ALIGN, H=_H, GROUP=_GROUP, GB=_GB, PACKS=_H // _GROUP // 4,
              SLOTS=max(64, triton.next_power_of_2(M * _TOPK)), E_PAD=triton.next_power_of_2(_E + 1),
              FILL_ROWS=_FILL_ROWS, EARLY_TRIGGER=EARLY_TRIGGER and _PDL, PF_KB=pf_kb, PF_N=pf_n,
              PF_NPAD=triton.next_power_of_2(pf_n), PF_SPAD=triton.next_power_of_2(triton.cdiv(pf_n, sf_step)),
              PF_SF_STEP=sf_step, KB_BYTES=_FC1_KB_BYTES, num_warps=4, enable_fp_fusion=False, launch_pdl=_PDL)
    if warmup:  # compile only (Triton JITFunction.warmup): nothing is launched, no memory is read or written
        _dsv41_rp_kernel.warmup(*args, grid=grid, **kw)
    else:
        _dsv41_rp_kernel[grid](*args, **kw)
    return pf_kb


def _precompile(w1, sf, ids_dtype, device) -> int:
    """Compile every SLOTS bucket of the variant this process will launch (same _launch arguments, real strides and
    FC1 geometry), so that a Triton/ptxas failure turns the fused path off instead of failing a forward. The pointer
    arguments are tiny placeholders of the right dtype: warmup only reads their dtype and alignment."""
    x = torch.empty(64, dtype=torch.bfloat16, device=device)
    ids = torch.empty(64, dtype=ids_dtype, device=device)
    u8 = torch.empty(64, dtype=torch.uint8, device=device)
    i32 = torch.empty(64, dtype=torch.int32, device=device)
    n = 0
    for M in _PRECOMPILE_M:
        Ms, _ = _m_sum(M)
        mn_aligned = (Ms + 3) // 4 * 4
        _launch(x, _H, ids, _TOPK, 1, u8, _H, i32, mn_aligned, i32, i32, _TOPK, 1, i32, w1, sf, M, Ms, warmup=True)
        n += 1
    return n


def _ready(w1, sf, ids_dtype, device) -> bool:
    """Static check + one-time precompile, decided once per process at the first call that reaches it (vLLM's eager
    profile / warm-up runs precede every CUDA-graph capture). False keeps today's code for the whole process."""
    global _READY
    if _READY is None:
        why = "static check failed"
        ok = _static_ok()
        if ok:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return True  # not expected; undecided, the JIT compiles at launch exactly as without the precompile
            try:
                why = f"precompiled {_precompile(w1, sf, ids_dtype, device)} SLOTS variants"
            except Exception as exc:  # noqa: BLE001  a compile failure must never take the server down
                ok, why = False, f"precompile failed: {exc!r}"[:600]
        _READY = ok
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "DSV41_MOE_ROUTEPREP=1 (%s): fused route prep %s (%s); DSV41_MOE_FC1_FRONT=%s (trigger %s, prefetch %d "
            "k-blocks)", TAG, "eligible" if ok else "OFF", why, _FRONT or "0", EARLY_TRIGGER and _PDL, PF_KB)
    return _READY


def expects_unquantized_inputs(experts) -> bool:
    """Property body installed on DeepGemmFP4Experts: defer the FC1-input quant for the target layers only."""
    cfg = experts.moe_config
    return (
        getattr(experts, "_dsv41_target_m32", None) is not None
        and cfg.num_experts == _E
        and cfg.experts_per_token == _TOPK
        and cfg.hidden_dim == _H
    )


def _eligible(experts, x, w1, topk_ids, expert_map, expert_tokens_meta) -> bool:
    if not _static_ok():
        return False
    qc = experts.quant_config
    if (qc.quant_dtype != torch.float8_e4m3fn or qc.block_shape is None or qc.block_shape[1] != _GROUP
            or qc.a1_scale is not None or qc.per_act_token_quant):
        return False
    if expert_map is not None or getattr(experts, "_dsv41_force_m32", False):
        return False
    if expert_tokens_meta is not None and getattr(expert_tokens_meta, "expert_num_tokens", None) is not None:
        return False
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.size(1) != _H or not x.is_contiguous():
        return False
    if topk_ids.dim() != 2 or topk_ids.size(1) != _TOPK or topk_ids.dtype not in (torch.int32, torch.int64):
        return False
    M = topk_ids.size(0)
    if x.size(0) != M or M < 1 or M * _TOPK > _MAX_SLOTS or w1.size(0) != _E:
        return False
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M_and_alignment

    block_m = experts._workspace_m_alignment(M, w1.size(1), x.size(1), _TOPK, _E)
    _, align = compute_aligned_M_and_alignment(M, _TOPK, _E, block_m, expert_tokens_meta)
    return align == _ALIGN and _ready(w1, experts.w1_scale, topk_ids.dtype, x.device)


def front(experts, hidden_states, w1, topk_ids, expert_map, expert_tokens_meta):
    """apply() entry when prepare() deferred the quant: (bf16 x, Deferred) for the fused path, else (a1q, a1q_scale)
    from the exact call prepare() would have made."""
    if _eligible(experts, hidden_states, w1, topk_ids, expert_map, expert_tokens_meta):
        return hidden_states, Deferred(w1, experts.w1_scale)
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import _quantize_input

    return _quantize_input(hidden_states, experts.quant_config)


def fused_permute(aq, aq_scale, topk_ids, local_num_experts, expert_map, expert_tokens_meta, aq_out, M_sum,
                  align_used, block_k, use_psum_layout, return_expert_ends):
    """deepgemm_moe_permute's fused branch: same return contract as the mfront branch."""
    M, H = aq.shape
    ok = (isinstance(aq_scale, Deferred) and align_used == _ALIGN and block_k == _GROUP and not use_psum_layout
          and expert_map is None and local_num_experts == _E and H == _H and aq.dtype == torch.bfloat16
          and topk_ids.shape == (M, _TOPK) and M * _TOPK <= _MAX_SLOTS and aq_out is not None
          and tuple(aq_out.shape) == (M_sum, H) and aq_out.is_contiguous() and aq_out.dtype == torch.float8_e4m3fn
          and aq.is_contiguous() and topk_ids.dtype in (torch.int32, torch.int64))
    if ok:  # the kernel reads x rows while other programs write slot rows: they must not share memory
        a0, a1 = aq.data_ptr(), aq.data_ptr() + aq.numel() * aq.element_size()
        o0, o1 = aq_out.data_ptr(), aq_out.data_ptr() + aq_out.numel()
        ok = a1 <= o0 or o1 <= a0
    if not ok:  # front() and deepgemm_moe_permute disagree: never serve unquantized rows
        raise RuntimeError(f"DSV41_MOE_ROUTEPREP: fused route prep not applicable here (M={M}, align={align_used})")
    E = local_num_experts
    packs = H // _GROUP // 4
    mn_aligned = (M_sum + 3) // 4 * 4
    device = aq.device
    sfp = torch.empty_strided((M_sum, packs), (1, mn_aligned), device=device, dtype=torch.int32)
    layout = torch.empty((M_sum,), device=device, dtype=torch.int32)
    inv_perm = torch.empty(topk_ids.shape, device=device, dtype=torch.int32)
    ends = torch.empty((E,), device=device, dtype=torch.int32)
    w1 = aq_scale.w1
    sf = aq_scale.w1_scale
    pf_kb = _launch(aq, aq.stride(0), topk_ids, topk_ids.stride(0), topk_ids.stride(1), aq_out.view(torch.uint8),
                    aq_out.stride(0), sfp, mn_aligned, layout, inv_perm, inv_perm.stride(0), inv_perm.stride(1), ends,
                    w1, sf, M, M_sum)
    global _PF_LOGGED
    if not _PF_LOGGED:
        _PF_LOGGED = True
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "%s: first fused call M=%d M_sum=%d; FC1 front: trigger %s, prefetch %d k-blocks (requested %d)%s", TAG, M,
            M_sum, EARLY_TRIGGER and _PDL, pf_kb, PF_KB,
            f"; w1 {tuple(w1.shape)}/{w1.stride()} w1_scale {tuple(sf.shape)}/{sf.stride()} {sf.dtype}" if PF_KB and
            w1 is not None and sf is not None else "")
    if return_expert_ends:
        return aq_out, sfp, layout, inv_perm, align_used, ends
    return aq_out, sfp, layout, inv_perm, align_used
