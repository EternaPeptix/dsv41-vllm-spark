"""Bit-exact CPU emulation of the live skinny_mx A8 kernel (v1), skinny_mx v2 (A8 and fused quant) and the
FlashInfer CuTe-DSL MXFP8QuantizeSwizzled op sequence. numpy only; every fp32 operation is rounded explicitly:

  fma32(a, b, c)  single-rounding fp32 FMA: exact product in float64 (24+24 bits), TwoSum for the addition,
                  then round to fp32 with an explicit fix of the one double-rounding case (exact fp32 midpoint);
  add32 / mul32   float64 op then round to fp32 (innocuous double rounding for + and *, 53 >= 2*24+2);
  bf16_rn         round-to-nearest-even on the fp32 bits.

The simulators follow the kernels' index mapping lane by lane (which lane loads which 16-byte vector, which lane
computes which second-vector dot and which lane it is shuffled to, butterfly partners, combine order), so an
off-by-one in v2's mapping shows up as a bit difference against v1.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32


# ------------------------------------------------------------------ fp32 primitives
def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


def add32(a, b):
    return f32(np.asarray(a, np.float64) + np.asarray(b, np.float64))


def mul32(a, b, ftz=False):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if ftz:
        a = np.where(np.abs(a) < 2.0 ** -126, np.copysign(0.0, a), a)
        b = np.where(np.abs(b) < 2.0 ** -126, np.copysign(0.0, b), b)
    r = f32(a * b)
    if ftz:
        r = np.where(np.abs(r) < 2.0 ** -126, np.copysign(0.0, r), r)
    return r


def fma32(a, b, c):
    """fp32 fmaf(a, b, c), correctly rounded (inputs are fp32 values held in float64, finite)."""
    a, b, c = (np.asarray(v, np.float64) for v in (a, b, c))
    p = a * b  # exact
    s = p + c
    bb = s - p
    err = (p - (s - bb)) + (c - bb)  # TwoSum: s + err == p + c exactly
    r = s.astype(np.float32)
    rd = r.astype(np.float64)
    # the only case f32(s) can be wrong: s exactly halfway between two fp32 values and err != 0
    toward = np.where(rd < s, np.float32(np.inf), np.float32(-np.inf))
    r2 = np.nextafter(r, toward.astype(np.float32))
    mid = (rd != s) & (((rd + r2.astype(np.float64)) * 0.5) == s) & (err != 0)
    if np.any(mid):
        hi = np.maximum(rd, r2.astype(np.float64))
        lo = np.minimum(rd, r2.astype(np.float64))
        rd = np.where(mid, np.where(err > 0, hi, lo), rd)
    # exact zero sum keeps IEEE sign rules from the float64 addition (p + c), already in s
    return rd


def bf16_rn(x):
    """fp32 (float64 holder) -> bf16 RNE -> back as float64 (NaN-free inputs)."""
    u = f32(x).astype(np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    return (r.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def bf16_bits_to_f(bits):
    return (np.asarray(bits, np.uint32) << 16).view(np.float32).astype(np.float64)


def f_to_bf16_bits(x):
    u = f32(x).astype(np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


# e4m3fn decode table
def _e4m3_table():
    t = np.zeros(256)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 7
        if e == 15 and m == 7:
            t[b] = np.nan
        elif e == 0:
            t[b] = s * m * 2.0 ** -9
        else:
            t[b] = s * (1 + m / 8) * 2.0 ** (e - 7)
    return t


E4M3 = _e4m3_table()
_POS = E4M3[:127]  # codes 0..126 are 0 .. 448 ascending


def e4m3_rn_satfinite(v):
    """cvt.rn.satfinite.e4m3 of fp32 values (finite here: inputs are already clamped to +-448)."""
    v = np.asarray(v, np.float64)
    a = np.abs(v)
    a = np.minimum(a, 448.0)
    i = np.searchsorted(_POS, a)  # first code with value >= a
    i = np.clip(i, 0, 126)
    lo = np.clip(i - 1, 0, 126)
    vl, vh = _POS[lo], _POS[i]
    dl, dh = a - vl, vh - a
    pick = np.where(dl < dh, lo, np.where(dh < dl, i, np.where(lo % 2 == 0, lo, i)))  # ties to even code
    pick = np.where(a == vh, i, pick)
    sign = np.signbit(v)
    return (pick | np.where(sign, 0x80, 0)).astype(np.uint8)


def e8m0_to_float(e):
    e = np.asarray(e, np.int64)
    out = np.where(e == 0, 2.0 ** -127, np.ldexp(1.0, (e - 127).clip(-127, 127)))
    return np.where(e == 255, np.nan, out)


# ------------------------------------------------------------------ FlashInfer MXFP8 quant (PTX semantics)
INV448 = float(np.float32(1.0 / 448.0))


def fi_float_to_ue8m0_fast(v):
    v32 = f32(v).astype(np.float32)
    bits = v32.view(np.uint32).astype(np.int64)
    exp_b = (bits >> 23) & 255
    mant = bits & 0x7FFFFF
    bump = (mant != 0).astype(np.int64)
    tiny = (exp_b == 0) & (mant <= 0x400000)
    bump = np.where(tiny, 0, bump)
    res = exp_b + bump
    res = np.where(res > 254, 254, res)
    return np.where(v32 <= 0, 0, res)  # setp.le.f32 (NaN compares false)


def fi_inv_scale(e):
    e = np.asarray(e, np.int64)
    ne = np.maximum(254 - e, 0)
    r = (ne.astype(np.uint32) << 23).view(np.float32).astype(np.float64)
    return np.where(e == 0, 0.0, r)


def ptx_min(a, b):
    return np.fmin(a, b)  # min.f32: NaN operand -> the other


def ptx_max(a, b):
    return np.fmax(a, b)


def fi_quant_blocks(xbits, ftz=False):
    """xbits: uint16 bf16 bits [..., 32] (one MX block per last axis).
    Returns (codes uint8 [..., 32], ue8m0 int [...]). Structure of the 4T/SF swizzled kernel: 4 lanes x 8 values,
    bf16x2 abs/max tree, hmax to f32, xor-shuffle max over the 4 lanes."""
    xb = np.asarray(xbits, np.uint32)
    ab = xb & 0x7FFF
    af = bf16_bits_to_f(ab).reshape(*xb.shape[:-1], 4, 8)  # lane, 8 values
    # bfloat2_max_abs_4 tree on pairs (v0..v3 = values (0,1),(2,3),(4,5),(6,7)): lane-wise max per half
    p = af.reshape(*af.shape[:-1], 4, 2)  # [.., lane, pair, half]
    m01 = ptx_max(p[..., 0, :], p[..., 1, :])
    m23 = ptx_max(p[..., 2, :], p[..., 3, :])
    mx = ptx_max(m01, m23)  # [.., lane, half]
    local = ptx_max(mx[..., 0], mx[..., 1])  # bfloat2_hmax_reduce_to_f32
    # reduce_max_4threads: xor 1 then xor 2
    l1 = ptx_max(local, local[..., [1, 0, 3, 2]])
    g = ptx_max(l1, l1[..., [2, 3, 0, 1]])[..., 0]
    nm = mul32(g, INV448, ftz=ftz)
    ue = fi_float_to_ue8m0_fast(nm)
    inv = fi_inv_scale(ue)
    xf = bf16_bits_to_f(xb)
    v = mul32(xf, inv[..., None])
    v = ptx_max(ptx_min(v, 448.0), -448.0)
    return e4m3_rn_satfinite(v), ue


def v1_deq(codes, ue):
    """v1 A8 prologue: bf16_rn(float(e4m3) * e8m0_to_float(scale)) (fp32 multiply, no FTZ)."""
    sc = e8m0_to_float(ue)
    return bf16_rn(mul32(E4M3[codes], sc[..., None]))


# ------------------------------------------------------------------ GEMV simulators
WARPS = SPLIT = 8


def dot16(wf, xf):
    """wf, xf: [..., 16]; v1 order s = 0; s = fmaf(w_i, x_i, s), i = 0..15."""
    s = np.zeros(np.broadcast_shapes(wf.shape[:-1], xf.shape[:-1]))
    for i in range(16):
        s = fma32(wf[..., i], xf[..., i], s)
    return s


def v1_pick_split(N, K, grid):
    best, best_eff = 1, -1.0
    s = 1
    while s <= 8:
        if s > 1 and (K // 16) // s < 32:
            break
        iters = (N + (WARPS // s) - 1) // (WARPS // s)
        waves = (iters + grid - 1) // grid
        eff = iters / (waves * grid)
        if eff > best_eff + 0.02:
            best_eff, best = eff, s
        s *= 2
    return best


def v1_gemv(xdeq, wcodes, wsc, grid=48, ret_t=False):
    """skinny_mx_kernel<M, 4, SPLIT> (A8), xdeq [M, K] float64 (bf16 values), wcodes uint8 [N, K], wsc [N, K/32].
    Returns y as float64 holding bf16 values [M, N]."""
    M, K = xdeq.shape
    N = wcodes.shape[0]
    split = v1_pick_split(N, K, grid)
    assert split == 8, split
    kv = K // 16
    kv_per = (kv + split - 1) // split
    wf = E4M3[wcodes].reshape(N, kv, 16)
    xf = xdeq.reshape(M, kv, 16)
    s_all = e8m0_to_float(wsc)  # [N, K/32]
    y = np.zeros((M, N))
    rows = np.arange(N)
    part_sums = np.zeros((N, split, M))
    for part in range(split):
        kb, ke = part * kv_per, min(kv, part * kv_per + kv_per)
        lane_acc = np.zeros((N, 32, M))
        for lane in range(32):
            acc = np.zeros((N, M))
            k = kb + lane
            # UNROLL loop (k + 96 < ke) then tail: both are acc = fmaf(s_k, dot16(w_k, x_k), acc) in k order
            while k < ke:
                d = dot16(wf[rows, k][:, None, :], xf[:, k][None, :, :])  # [N, M]
                acc = fma32(s_all[rows, k >> 1][:, None], d, acc)
                k += 32
            lane_acc[:, lane] = acc
        v = lane_acc
        for o in (16, 8, 4, 2, 1):
            v = add32(v, v[:, np.arange(32) ^ o])
        part_sums[:, part] = v[:, 0]
    t = np.zeros((N, M))
    for q in range(split):
        t = add32(t, part_sums[:, q])
    y[:, :] = bf16_rn(t).T
    return (y, t.T.copy()) if ret_t else y


KVP, E = 40, 8
RPP = 32 // E


def v2_gemv(xdeq_fn, wcodes, wsc, grid=48, maxr=None, ret_t=False):
    """skinny_v2_kernel<M, MAXR, QF> lane-level simulator. xdeq_fn(part) -> [M, KVP, 16] activation slice the
    warp builds in its prologue (A8 or fused quant). Returns y [M, N]."""
    N, K = wcodes.shape
    kv = K // 16
    assert v1_pick_split(N, K, 48) == 8 and kv == 8 * KVP
    r = -(-N // grid)
    MAXR = maxr or (5 if r <= 5 else 8)
    assert r <= MAXR
    NPASS = (MAXR + RPP - 1) // RPP
    wf_all = E4M3[wcodes].reshape(N, kv, 16)
    s_all = e8m0_to_float(wsc)
    M = xdeq_fn(0).shape[0]
    y = np.full((M, N), np.nan)
    tt = np.full((M, N), np.nan)
    slices = [xdeq_fn(p) for p in range(WARPS)]  # [M, KVP, 16] each
    for b0 in range(grid):
        rows = [b0 + i * grid for i in range(MAXR)]
        valid = [row < N for row in rows]
        part_sums = np.zeros((MAXR, WARPS, M))
        for warp in range(WARPS):
            kb = warp * KVP
            xs = slices[warp]
            lanes = np.arange(32)
            # A: weights (zero for invalid rows)
            wa = np.zeros((MAXR, 32, 16))
            sa = np.zeros((MAXR, 32))
            so = np.zeros((MAXR, 32))
            for i, row in enumerate(rows):
                if valid[i]:
                    wa[i] = wf_all[row, kb + lanes]
                    sa[i] = s_all[row, (kb + lanes) >> 1]
                    own = lanes < E
                    so[i, own] = s_all[row, (kb + 32 + lanes[own]) >> 1]
            wx = np.zeros((NPASS, 32, 16))
            for p in range(NPASS):
                for lane in range(32):
                    i = p * RPP + lane // E
                    if i < MAXR and rows[i] < N:
                        wx[p, lane] = wf_all[rows[i], kb + 32 + (lane % E)]
            # C: first vectors
            acc = np.zeros((MAXR, 32, M))
            xa = xs[:, lanes, :]  # local vector index v = lane -> [M, 32, 16]
            for i in range(MAXR):
                d = dot16(wa[i][None, :, :], xa)  # [M, 32]
                acc[i] = fma32(sa[i][:, None], d.T, 0.0)  # fmaf(s0, dot16(w0, x), 0.f)
            # second vectors computed by all lanes, then shuffled to the owner
            dx = np.zeros((NPASS, 32, M))
            for p in range(NPASS):
                xv = xs[:, 32 + (lanes % E), :]  # [M, 32, 16]
                dx[p] = dot16(wx[p][None, :, :], xv).T
            for i in range(MAXR):
                src = (i % RPP) * E + (lanes % E)
                d = dx[i // RPP][src]  # __shfl_sync(FULL, dx[i/RPP][m], src)
                own = lanes < E
                acc[i, own] = fma32(so[i, own][:, None], d[own], acc[i, own])
            # D: butterfly, lane 0
            v = acc
            for o in (16, 8, 4, 2, 1):
                v = add32(v, v[:, lanes ^ o])
            part_sums[:, warp] = v[:, 0]
        for i, row in enumerate(rows):
            if valid[i]:
                t = np.zeros(M)
                for q in range(SPLIT):
                    t = add32(t, part_sums[i, q])
                y[:, row] = bf16_rn(t)
                tt[:, row] = t
    assert not np.isnan(y).any()
    return (y, tt) if ret_t else y


def a8_slices(xdeq):
    """A8 prologue: the warp's slice is v1's smem content for vectors [kb, kb + KVP)."""
    M, K = xdeq.shape
    xv = xdeq.reshape(M, K // 16, 16)
    return lambda p: xv[:, p * KVP:(p + 1) * KVP]


def fused_slices(xbits, ftz=False):
    """QF prologue, literally: task t = it * 32 + lane, quad bq = t >> 2, m = bq // BPP, blk = bq % BPP,
    elem = kb*16 + blk*32 + (t & 3)*8; the quad's 32 values are quantised together (fi_quant8); lane writes
    its 8 dequantised values to smem[m * kv8 + elem / 8] (8-value chunk index)."""
    M, K = xbits.shape
    BPP = KVP // 2
    kv8 = K // 8

    def build(p):
        kb = p * KVP
        smem = np.full((M * kv8, 8), np.nan)
        tasks = M * BPP * 4
        iters = -(-tasks // 32)
        for it in range(iters):
            for quad in range(8):
                t0 = it * 32 + quad * 4
                if t0 >= tasks:
                    continue  # whole quad invalid (tasks is a multiple of 16)
                bq = t0 >> 2
                m, blk = bq // BPP, bq % BPP
                elem0 = kb * 16 + blk * 32
                codes, ue = fi_quant_blocks(xbits[m, elem0:elem0 + 32][None, :], ftz=ftz)
                deq = v1_deq(codes, ue)[0]
                for sub in range(4):
                    elem = elem0 + sub * 8
                    smem[m * kv8 + (elem >> 3)] = deq[sub * 8:(sub + 1) * 8]
        sl = smem.reshape(M, K // 16, 16)[:, kb:kb + KVP]
        assert not np.isnan(sl).any()
        return sl

    return build
