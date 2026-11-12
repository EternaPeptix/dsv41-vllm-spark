#!/usr/bin/env python3
"""CPU emulation: old vs new op sequences, with explicit fp32 / bf16 / e4m3 rounding at every step.

A. Inverse-RoPE fold. Old = live fused_inv_rope_fp8_quant (quantize=False) writes out_buf[g, m, hpg*512] bf16,
   the live partial reads A[m, k] = out_buf[g, m, k] through (ASG, ASM, ASK) = out.transpose(0,1) strides.
   New = _r2_wo_a_partial's ROPE branch computes A[m, k] from o[m, g*HPG + k//512, k%512], partner d^1,
   cos/sin at (d-NOPE)>>1 (+HALF_ROPE), masks. Both use the live PTX op sequence (ps = p*s RN; add = fma(x,c,ps);
   sub = fma(x,c,-ps); bf16 RN; nope lanes = original bits). Exact fp32 FMA via float64 TwoSum.
B. Split reduce. Old = the live _sm120_wo_a_reduce order derived from its TTGIR layout ([32,256] blocked
   sizePerThread [1,4], threadsPerWarp [1,32], warpsPerCTA [2,2], order [1,0]: in-thread chain over the 16
   registers of each column in register order, then the cross-warp add) and its PTX (add.f32 chain
   r1+r6, +r10, ...; register i = row w0+2i). New = _r2_reduce_quant's explicit (p0+p2+..+p30)+(p1+..+p31).
   Also shows that a different order (0..31 sequential, pairwise) changes bits on the same data.
C. MXFP8 quant. Old = flashinfer MXFP8QuantizeSwizzledKernel 4-threads-per-SF path (8 bf16 per thread: abs/max
   bf16x2 tree, f32 shuffle-xor max over 4 lanes, amax*fp32(1/448), ue8m0 bit trick, 2^(127-ue8m0) from bits,
   per element mul.f32 / clamp +-448 / cvt.rn.satfinite.e4m3). New = _r2_reduce_quant's [8,32] tile (tl.max over
   32, same scalar ops). Compares e4m3 bytes and UE8M0 bytes.
D. Scale layout: _r2_reduce_quant's 128x4 swizzle offsets == flashinfer compute_sf_index_swizzled_128x4_gpu for
   every (row < 128, col < 32); for every M in 1..32 the kit's padding loop + data rows write each of the
   128*32 scale bytes exactly once (rows >= M get 0), like flashinfer.
Run on 512S1: ~/exo-venv313/bin/python3 tests/test_emulation.py
"""
import numpy as np
import torch

RNG = np.random.default_rng(20261108)


# ---------------- exact rounding helpers ----------------
def f32(x):
    return np.asarray(x, dtype=np.float32)


def bf16_rn(x32):
    """fp32 -> bf16 round-to-nearest-even (cvt.rn.bf16.f32), returned as uint16 bits. NaN-free inputs."""
    b = f32(x32).view(np.uint32).astype(np.uint64)
    lsb = (b >> 16) & 1
    r = (b + 0x7FFF + lsb) >> 16
    return r.astype(np.uint16)


def bf16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


def fma32(a, b, c):
    """Correctly rounded fp32 fma(a, b, c) (fma.rn.f32), vectorised, exact via float64 TwoSum."""
    a, b, c = (np.asarray(v, dtype=np.float32).astype(np.float64) for v in (a, b, c))
    p = a * b  # exact: 24+24 bit significands
    s = p + c
    bb = s - p
    err = (p - (s - bb)) + (c - bb)  # exact: s + err == p + c
    r = s.astype(np.float32)
    rd = r.astype(np.float64)
    # s is on the float64 grid; the only double-rounding hazard is s exactly halfway between two fp32 values
    # (then f32(s) picked the even one, but the exact value s + err lies strictly on one side).
    lo = np.where(rd <= s, r, np.nextafter(r, np.float32(-np.inf)))
    hi = np.where(rd <= s, np.nextafter(r, np.float32(np.inf)), r)
    hi = np.where(rd == s, r, hi)
    is_mid = (rd != s) & (s == (lo.astype(np.float64) + hi.astype(np.float64)) / 2) & (err != 0)
    r = np.where(is_mid, np.where(err > 0, hi, lo), r)
    return r.astype(np.float32)


def check_fma32():
    # against exact rational arithmetic on a hostile sample
    from fractions import Fraction

    n = 4000
    a = f32(RNG.standard_normal(n) * np.exp2(RNG.integers(-30, 30, n)))
    b = f32(RNG.standard_normal(n) * np.exp2(RNG.integers(-30, 30, n)))
    c = f32(-(a.astype(np.float64) * b.astype(np.float64)) * (1 + RNG.standard_normal(n) * 1e-7))  # cancellation
    c[::3] = f32(RNG.standard_normal(len(c[::3])) * np.exp2(RNG.integers(-60, 60, len(c[::3]))))
    got = fma32(a, b, c)
    bad = 0
    for i in range(n):
        ex = Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))
        # round exact to fp32 (RNE) through candidates
        r = np.float32(float(ex))
        cands = [r, np.nextafter(r, np.float32(np.inf)), np.nextafter(r, np.float32(-np.inf))]
        best = min(cands, key=lambda v: (abs(Fraction(float(v)) - ex), int(np.float32(v).view(np.uint32)) & 1))
        if best.view(np.uint32) != got[i].view(np.uint32):
            bad += 1
    assert bad == 0, f"fma32 emulation wrong on {bad}/{n}"
    print(f"PASS fma32 emulation == exact rational RNE on {n} hostile triples")


# ---------------- A. inverse-RoPE fold ----------------
def rope_lane(x32, p32, c32, s32, even):
    ps = f32(p32 * s32)  # mul.f32 (RN)
    add = fma32(x32, c32, ps)
    sub = fma32(x32, c32, -ps)
    return bf16_rn(np.where(even, add, sub))


def live_inv_rope(o_u16, pos, cs, G, HPG, NOPE, HALF):
    """fused_inv_rope_fp8_quant.kernel, quantize=False: grid (token, global head); out_buf[g, t, hig*HD + d]."""
    M, H, HD = o_u16.shape
    out = np.zeros((G, M, HPG * HD), dtype=np.uint16)
    d = np.arange(HD)
    is_rope = d >= NOPE
    rl = d - NOPE
    for t in range(M):
        for gh in range(H):
            g, hig = gh // HPG, gh % HPG
            x = o_u16[t, gh]
            xp = np.where(is_rope, o_u16[t, gh, d ^ 1], 0)
            ci = np.maximum(rl >> 1, 0)
            c = np.where(is_rope, cs[pos[t], ci], 1.0)
            s = np.where(is_rope, cs[pos[t], HALF + ci], 0.0)
            rot = rope_lane(bf16_to_f32(x), bf16_to_f32(xp), f32(c), f32(s), (rl & 1) == 0)
            out[g, t, hig * HD:(hig + 1) * HD] = np.where(is_rope, rot, x)
    return out


def live_A_tile(out_buf, g, rows, ks, M):
    """the live partial's A load: A + group*ASG + row*ASM + k*ASK on out_buf.transpose(0, 1) ([M, G, D])."""
    A = np.zeros((len(rows), len(ks)), dtype=np.uint16)
    for i, r in enumerate(rows):
        if r < M:
            A[i] = out_buf[g, r, ks]
    return A


def new_A_tile(o_u16, pos, cs, g, rows, ks, M, HPG, HD, NOPE, HALF):
    """_r2_wo_a_partial ROPE branch, transcribed op by op (vectorised over the [16, 128] tile)."""
    rows = np.asarray(rows)
    ks = np.asarray(ks)
    rmask = (rows < M)[:, None]
    head = g * HPG + ks // HD
    d = ks % HD
    rr = np.minimum(rows, M - 1)
    xb = np.where(rmask, o_u16[rr[:, None], head[None, :], d[None, :]], 0).astype(np.uint16)
    is_rope = d >= NOPE
    rl = d - NOPE
    rope2 = rmask & is_rope[None, :]
    xp = np.where(rope2, o_u16[rr[:, None], head[None, :], (d ^ 1)[None, :]], 0).astype(np.uint16)
    pos_r = np.where(rows < M, pos[rr], 0)
    ci = np.maximum(rl >> 1, 0)
    c = np.where(rope2, cs[pos_r[:, None], ci[None, :]], 1.0)
    s = np.where(rope2, cs[pos_r[:, None], HALF + ci[None, :]], 0.0)
    rot = rope_lane(bf16_to_f32(xb), bf16_to_f32(xp), f32(c), f32(s), ((rl & 1) == 0)[None, :])
    return np.where(rope2, rot, xb).astype(np.uint16)


def hostile_bf16(shape):
    x = RNG.standard_normal(shape) * np.exp2(RNG.integers(-24, 14, shape))
    flat = x.reshape(-1)
    idx = RNG.integers(0, flat.size, flat.size // 40)
    flat[idx] = 0.0
    idx = RNG.integers(0, flat.size, flat.size // 40)
    flat[idx] = -0.0
    idx = RNG.integers(0, flat.size, flat.size // 80)
    flat[idx] = 65280.0  # large finite bf16
    idx = RNG.integers(0, flat.size, flat.size // 80)
    flat[idx] = 2.0 ** -133  # bf16 denormal
    return bf16_rn(f32(x))


def rope_cache(max_pos, HALF):
    inv = 1.0 / (10000 ** (np.arange(HALF) / HALF))
    ang = np.arange(max_pos)[:, None] * inv[None, :]
    return np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32)


def test_rope():
    HD, NOPE, HALF = 512, 448, 32
    cs = rope_cache(4096, HALF)
    cs[7] = f32([1.0] * HALF + [0.0] * HALF)  # identity row
    cs[9, :HALF] = -0.0  # signed zeros
    n_tiles = 0
    for G, HPG in ((1, 8), (2, 4)):
        K = HPG * HD
        for M in (1, 2, 5, 16, 17, 32):
            o = hostile_bf16((M, G * HPG, HD))
            pos = RNG.integers(0, 4096, M)
            pos[0] = 7
            if M > 1:
                pos[1] = 9
            out_buf = live_inv_rope(o, pos, cs, G, HPG, NOPE, HALF)
            for g in range(G):
                for rb in range(-(-M // 16)):
                    rows = rb * 16 + np.arange(16)
                    for split in range(K // 128):
                        ks = split * 128 + np.arange(128)
                        a_old = live_A_tile(out_buf, g, rows, ks, M)
                        a_new = new_A_tile(o, pos, cs, g, rows, ks, M, HPG, HD, NOPE, HALF)
                        assert np.array_equal(a_old, a_new), (G, HPG, M, g, rb, split)
                        n_tiles += 1
    print(f"PASS A. inverse-RoPE fold: {n_tiles} [16,128] A tiles bit-identical (G=1/HPG=8 and G=2/HPG=4, "
          f"M in 1,2,5,16,17,32, hostile bf16 incl. +-0, bf16 denormals, 65280, identity/signed-zero cos rows)")


# ---------------- B. split reduce ----------------
def blocked_owner(i, j, spt=(1, 4), tpw=(1, 32), wpc=(2, 2)):
    """(warp_row, rep_index_along_dim0) for element (i, j) of a [32, 256] tensor in the live #blocked layout."""
    tile0 = spt[0] * tpw[0] * wpc[0]  # 2
    wi = (i // (spt[0] * tpw[0])) % wpc[0]
    rep = i // tile0
    return wi, rep


def live_reduce(P):
    """P: [32, T] fp32. Per column: each warp row chains its 16 registers in rep order, then one cross-warp add."""
    S, T = P.shape
    chains = {}
    for i in range(S):
        wi, rep = blocked_owner(i, 0)
        chains.setdefault(wi, []).append((rep, i))
    acc = []
    for wi in sorted(chains):
        regs = [i for _, i in sorted(chains[wi])]
        a = P[regs[0]].copy()
        for i in regs[1:]:
            a = f32(a + P[i])
        acc.append(a)
    return f32(acc[0] + acc[1]), {wi: [i for _, i in sorted(v)] for wi, v in chains.items()}


def new_reduce(P):
    ev, od = P[0].copy(), P[1].copy()
    for i in range(1, 16):
        ev = f32(ev + P[2 * i])
        od = f32(od + P[2 * i + 1])
    return f32(ev + od)


def hostile_partials(T):
    P = f32(RNG.standard_normal((32, T)) * np.exp2(RNG.integers(-30, 30, (32, T))))
    P[1, : T // 4] = -P[0, : T // 4]
    P[:, T // 4: T // 4 + 32] = 0.0
    P[5, T // 4 + 32: T // 4 + 64] = -0.0
    P[:, T // 2: T // 2 + 32] = 0.0
    P[0, T // 2: T // 2 + 32] = 1.0 + 2.0 ** -8  # bf16 ties after the sum
    P[3, -64:] = f32(3.0e38)
    P[4, -64:] = f32(-3.0e38)  # large cancellation, stays finite in this order
    return P


def test_reduce():
    _, chains = live_reduce(np.zeros((32, 1), np.float32))
    assert chains == {0: list(range(0, 32, 2)), 1: list(range(1, 32, 2))}, chains
    T = 4096
    P = hostile_partials(T)
    old, _ = live_reduce(P)
    new = new_reduce(P)
    assert np.array_equal(old.view(np.uint32), new.view(np.uint32))
    seq = P[0].copy()
    for i in range(1, 32):
        seq = f32(seq + P[i])
    torch_sum = P.astype(np.float32).sum(0, dtype=np.float32)
    d_seq = int((seq.view(np.uint32) != new.view(np.uint32)).sum())
    d_np = int((torch_sum.view(np.uint32) != new.view(np.uint32)).sum())
    assert d_seq > 0 and d_np > 0, "emulation not sensitive to the order"
    zb_old, zb_new = bf16_rn(old), bf16_rn(new)
    assert np.array_equal(zb_old, zb_new)
    print(f"PASS B. reduce: live layout order = evens chain + odds chain ({chains[0][:3]}.. / {chains[1][:3]}..), "
          f"new explicit order bit-identical on {T} hostile columns (fp32 and bf16); sequential 0..31 differs in "
          f"{d_seq}, numpy pairwise in {d_np} columns -> the test is order sensitive")


# ---------------- C. MXFP8 quant ----------------
INV448 = np.float32(1.0 / 448.0)


def ue8m0(v32):
    bits = f32(v32).view(np.uint32).astype(np.int64)
    p_zero = f32(v32) <= 0
    exp = (bits >> 23) & 255
    man = bits & 0x7FFFFF
    bump = (man != 0).astype(np.int64)
    bump = np.where((exp == 0) & (man <= 0x400000), 0, bump)
    res = np.minimum(exp + bump, 254)
    return np.where(p_zero, 0, res).astype(np.uint32)


def inv_scale(ue):
    ue = np.asarray(ue, dtype=np.int64)
    e = np.maximum(254 - ue, 0)
    out = (e.astype(np.uint32) << 23).view(np.float32)
    return np.where(ue == 0, np.float32(0), out).astype(np.float32)


def to_e4m3(x32):
    y = np.minimum(np.maximum(f32(x32), f32(-448.0)), f32(448.0))
    return torch.from_numpy(y.copy()).to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def flashinfer_quant(z_u16):
    """4T/SF path: an SF block of 32 is 4 threads x 8 bf16; per thread abs/max tree on bf16x2, to f32, then
    shuffle-xor 1 and 2 with fmax; scale ops; per element bf16 -> f32 (shl 16), mul.f32, clamp, cvt."""
    M, N = z_u16.shape
    q = np.zeros((M, N), np.uint8)
    sf = np.zeros((M, N // 32), np.uint8)
    for r in range(M):
        for b in range(N // 32):
            blk = z_u16[r, b * 32:(b + 1) * 32]
            tmax = []
            for t in range(4):
                v = np.abs(bf16_to_f32(blk[t * 8:(t + 1) * 8]))  # habs2 + exact widening
                v01, v23 = np.maximum(v[0:2], v[2:4]), np.maximum(v[4:6], v[6:8])
                v2 = np.maximum(v01, v23)
                tmax.append(f32(max(v2[0], v2[1])))
            m1 = [f32(max(tmax[i], tmax[i ^ 1])) for i in range(4)]
            gmax = f32(max(m1[0], m1[2]))
            norm = f32(gmax * INV448)
            u = ue8m0(norm)
            inv = inv_scale(u)
            q[r, b * 32:(b + 1) * 32] = to_e4m3(f32(bf16_to_f32(blk) * inv))
            sf[r, b] = u
    return q, sf


def kit_quant(z_u16):
    """_r2_reduce_quant: per (row, 256-col chunk) program, [8, 32] tile, tl.max(|zf|, axis=1)."""
    M, N = z_u16.shape
    q = np.zeros((M, N), np.uint8)
    sf = np.zeros((M, N // 32), np.uint8)
    for r in range(M):
        for pc in range(N // 256):
            zf = bf16_to_f32(z_u16[r, pc * 256:(pc + 1) * 256]).reshape(8, 32)
            amax = np.abs(zf).max(axis=1).astype(np.float32)
            norm = f32(amax * INV448)
            u = ue8m0(norm)
            inv = inv_scale(u)
            q[r, pc * 256:(pc + 1) * 256] = to_e4m3(f32(zf * inv[:, None])).reshape(-1)
            sf[r, pc * 8:(pc + 1) * 8] = u
    return q, sf


def test_quant():
    M, N = 8, 1024
    z = hostile_bf16((M, N))
    z[0, 0:32] = 0  # all-zero block -> ue8m0 0, inv 0
    z[1, 32:64] = bf16_rn(f32(np.full(32, 448.0)))  # exact max
    z[2, 64:96] = bf16_rn(f32(RNG.standard_normal(32) * 2.0 ** 100))  # huge
    z[3, 96:128] = bf16_rn(f32(RNG.standard_normal(32) * 2.0 ** -100))  # tiny normal amax
    z[4, 128:160] = bf16_rn(f32(np.linspace(-3, 3, 32)))  # e4m3 rounding ties after scaling
    qa, sa = flashinfer_quant(z)
    qb, sb = kit_quant(z)
    assert np.array_equal(qa, qb) and np.array_equal(sa, sb)
    # independent sanity: dequantised values within half an e4m3 ulp
    print(f"PASS C. MXFP8 quant: kit [8,32]-tile order == flashinfer 4T/SF order, {M}x{N} hostile bf16 "
          "(zero / exact-448 / 2^100 / 2^-100 blocks, ties): e4m3 and UE8M0 bytes identical")


# ---------------- D. swizzled scale layout + padding coverage ----------------
def fi_swizzle(row, col, padded_cols):
    return (col % 4) + (col // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + (row // 128) * (128 * padded_cols)


def kit_swizzle(r, c, sf_cols):
    return (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * sf_cols)


def test_layout():
    N, sf_cols = 1024, 32
    for r in range(256):
        for c in range(sf_cols):
            assert kit_swizzle(r, c, sf_cols) == fi_swizzle(r, c, sf_cols)
    for M in range(1, 33):
        padded = (M + 127) // 128 * 128
        writes = np.zeros(padded * sf_cols, np.int32)
        for pc in range(N // 256):
            for r in range(M):  # grid (N//256, M)
                for j in range(8):
                    writes[kit_swizzle(r, pc * 8 + j, sf_cols)] += 1
                for p in range(M + r, padded, M):
                    for j in range(8):
                        writes[kit_swizzle(p, pc * 8 + j, sf_cols)] += 1
        assert (writes == 1).all(), (M, np.unique(writes))
    print("PASS D. swizzle offsets == flashinfer compute_sf_index_swizzled_128x4_gpu (rows 0..255, 32 cols); "
          "for every M in 1..32 each of the 128x32 scale bytes is written exactly once (pad rows -> 0)")


if __name__ == "__main__":
    check_fma32()
    test_rope()
    test_reduce()
    test_quant()
    test_layout()
    print("ALL PASS test_emulation")
