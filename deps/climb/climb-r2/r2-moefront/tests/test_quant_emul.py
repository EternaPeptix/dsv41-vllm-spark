"""CPU emulation: live FC1-input quant (SASS op order) == fused kernel quant (Triton op order), bit for bit.

Exhaustive where the op sequence is elementwise:
  A. scale map over every non-negative finite bf16 absmax (+ eps floor): scale bits live == fused == float64 ref
  B. element map over every bf16 x (65,536 patterns incl. +-0, denormals, max, NaN/inf) for every reachable scale:
     e4m3 byte live == fused, and == torch c10 reference for every finite x with |x| <= 448 * scale
  C. packed UE8M0 words: mfront uint32 formula == fused int32 formula for every reachable scale in every position
  D. whole 128-groups / 5120-rows: random, recorded-like and hostile rows (dynamic range, ties, zeros, signed zeros,
     denormals, max magnitude, exact 448*2^k, RNE midpoints): bytes + scale bits identical, and absmax order-free.
Run: python3 tests/test_quant_emul.py  (numpy + torch CPU; ~1-2 min on 512S1)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emul import (F32, U32, assert_ieee_host, bits_f32, f32_bits, live_quant, new_quant, pack_live,  # noqa: E402
                  pack_new, ref_quant)

RNG = np.random.default_rng(20261108)
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAIL.append(name)


def first_diff(a, b):
    d = np.nonzero(a.reshape(-1) != b.reshape(-1))[0]
    return f"first mismatch at flat index {d[0]} ({a.reshape(-1)[d[0]]} vs {b.reshape(-1)[d[0]]}), {d.size} total" \
        if d.size else ""


def group_with_absmax(amax_bits, fill_bits=None):
    """[n, 128] bf16 groups whose absmax is amax_bits (element 37), others from fill or zero."""
    n = amax_bits.shape[0]
    g = np.zeros((n, 128), dtype=np.uint16) if fill_bits is None else fill_bits.copy()
    g[:, 37] = amax_bits
    return g


def test_scale_map():
    pos = np.arange(0, 0x7F80, dtype=np.uint32).astype(np.uint16)  # +0 .. max finite bf16
    neg = pos | np.uint16(0x8000)
    for name, bits in (("positive", pos), ("negative", neg)):
        g = group_with_absmax(bits)
        _, s_live = live_quant(g)
        _, s_new = new_quant(g)
        _, s_ref = ref_quant(g)
        check(f"A scale map, every {name} bf16 absmax ({bits.size})", np.array_equal(s_live, s_new),
              first_diff(s_live, s_new))
        check(f"A scale map vs float64 reference ({name})", np.array_equal(s_live, s_ref), first_diff(s_live, s_ref))
    scales = np.unique(s_live)
    e = scales >> U32(23)
    print(f"   reachable scale exponents: {e.min()}..{e.max()} ({scales.size} values; 1/scale normal: "
          f"{bool(((254 - e) >= 1).all() and ((254 - e) <= 254).all())})")
    return scales


def test_element_map(scales):
    xs = np.arange(0, 65536, dtype=np.uint32).astype(np.uint16)  # every bf16 pattern
    t0 = time.time()
    bad = 0
    bad_ref = 0
    first = ""
    n_ref = 0
    for sb in scales:
        # a group = 128 lanes; put the absmax that produces this scale in lane 37 and sweep x in the others
        amax_bits = None
        # find one bf16 absmax that maps to sb (from the scale map: amax = ~ scale * 448 * 0.75 works)
        target = float(bits_f32(np.array([sb], dtype=U32))[0]) * 448.0 * 0.75
        cand = (f32_bits(np.array([target], dtype=F32)) >> U32(16)).astype(np.uint16)
        for delta in range(-4, 5):
            c = np.array([int(cand[0]) + delta], dtype=np.uint16)
            if new_quant(group_with_absmax(c))[1][0] == sb:
                amax_bits = c
                break
        if amax_bits is None:  # eps floor scales: absmax 0
            c = np.array([0], dtype=np.uint16)
            if new_quant(group_with_absmax(c))[1][0] == sb:
                amax_bits = c
            else:
                continue
        # the element op only depends on (x, scale): emulate with the group's scale and every x
        sc = bits_f32(np.array([sb], dtype=U32))
        x = xs.reshape(512, 128).copy()
        # live and new per-element paths with the scale forced (the scale map was checked in A)
        bl = _elem_live(x, sc[0])
        bn = _elem_new(x, sc[0])
        if not np.array_equal(bl, bn):
            bad += 1
            first = first or f"scale bits {sb:#x}: " + first_diff(bl, bn)
        # torch c10 reference on the realistic domain |x| <= 448 * scale (finite)
        xf = (xs.astype(np.uint32) << 16).view(np.float32).reshape(512, 128)
        dom = np.isfinite(xf) & (np.abs(xf.astype(np.float64)) <= 448.0 * float(sc[0]))
        import torch
        q = (xf.astype(np.float64) * (1.0 / float(sc[0]))).astype(np.float32)
        r = torch.from_numpy(np.clip(q, -448, 448).astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        n_ref += int(dom.sum())
        if not np.array_equal(bl[dom], r[dom]):
            bad_ref += 1
            first = first or f"ref scale {sb:#x}: " + first_diff(bl[dom], r[dom])
    check(f"B element map, all 65536 bf16 x for each of {scales.size} reachable scales "
          f"({scales.size * 65536} cases, {time.time() - t0:.1f}s)", bad == 0, first)
    check(f"B element map vs torch c10 e4m3fn reference ({n_ref} in-domain cases)", bad_ref == 0, first)


def _elem_live(xb, scale):
    from emul import FP8_MAX, FP8_MIN, bf16_to_f32
    x = bf16_to_f32(xb)
    inv = (F32(1.0) / F32(scale)).astype(F32)
    q = np.fmin(np.fmax((x * inv).astype(F32), FP8_MIN), FP8_MAX)
    r27 = f32_bits(q)
    r28 = r27 & U32(0x7FFFFFFF)
    sign = (r27 >> U32(24)) & U32(0x80)
    big = np.where(r28 > U32(0x7F800000), U32(0x7F), U32(0x7E))
    odd = (r27 >> U32(20)) & U32(1)
    r9 = ((r27.astype(np.uint64) + 0x407FFFF + odd) & 0xFFFFFFFF).astype(U32)
    norm = np.where((r9 & U32(0xFF00000)) != U32(0x7F00000), r9 >> U32(20), U32(0x7E))
    sub = f32_bits((bits_f32(r28) + F32(16384.0)).astype(F32))
    mag = np.where(r28 >= U32(0x43F00000), big, np.where(r28 > U32(0x3C7FFFFF), norm, sub))
    return ((mag & U32(0xFF)) | sign).astype(np.uint8)


def _elem_new(xb, scale):
    from emul import FP8_MAX, FP8_MIN, I32, bf16_to_f32
    x = bf16_to_f32(xb)
    sb = f32_bits(np.array([scale], dtype=F32)).view(I32)
    inv = (F32(1.0) / sb.view(F32)).astype(F32)  # div_rn(1, scale)
    q = np.fmin(np.fmax((x * inv).astype(F32), FP8_MIN), FP8_MAX)
    qb = f32_bits(q).view(I32)
    a = qb & I32(0x7FFFFFFF)
    sign = (qb >> 24) & I32(0x80)
    odd = (a >> 20) & I32(1)
    tn = (a.astype(np.int64) + 0x0407FFFF + odd).astype(np.int64).astype(I32)
    mag_n = np.where((tn & I32(0x0FF00000)) != I32(0x07F00000), tn >> 20, I32(0x7E))
    mag_s = (a.view(F32) + F32(16384.0)).astype(F32).view(I32)
    mag_o = np.where(a > I32(0x7F800000), I32(0x7F), I32(0x7E))
    mag = np.where(a >= I32(0x43F00000), mag_o, np.where(a >= I32(0x3C800000), mag_n, mag_s))
    return ((mag & I32(0xFF)) | sign).astype(np.uint8)


def test_inv_exact(scales):
    inv = (F32(1.0) / bits_f32(scales)).astype(F32)
    e = scales >> U32(23)
    want = ((U32(254) - e) << U32(23)).astype(U32)
    check("B' IEEE 1/scale is the exact power of two 2^-k for every reachable scale",
          np.array_equal(f32_bits(inv), want), first_diff(f32_bits(inv), want))


def test_pack(scales):
    n = scales.size
    ok = True
    for pos in range(4):
        v = np.full((n, 4), scales[n // 2], dtype=U32)
        v[:, pos] = scales
        ok &= np.array_equal(pack_live(v), pack_new(v))
    v = scales[RNG.integers(0, n, size=(1_000_000, 4))]
    ok &= np.array_equal(pack_live(v), pack_new(v))
    check(f"C packed UE8M0 word, every scale in every position + 1e6 random 4-tuples", ok)


def hostile_rows(n):
    rows = []
    r = RNG.standard_normal((n, 5120)).astype(np.float32)
    rows.append(r)
    rows.append((r * np.exp2(RNG.integers(-60, 60, size=(n, 1)))).astype(np.float32))  # per-row dynamic range
    rows.append((r * np.exp2(RNG.integers(-130, 125, size=(n, 5120)))).astype(np.float32))  # per-element range
    z = np.zeros((n, 5120), dtype=np.float32)
    z[:, ::2] = -0.0
    rows.append(z)  # zeros / signed zeros (eps floor)
    t = np.full((n, 5120), 3.0, dtype=np.float32)
    rows.append(t)  # ties
    k = RNG.integers(-30, 30, size=(n, 40))
    m = np.repeat(448.0 * np.exp2(k), 128, axis=1).astype(np.float32)
    rows.append(m * RNG.choice([-1.0, 1.0], size=(n, 5120)).astype(np.float32))  # exact 448*2^k absmax
    big = np.full((n, 5120), 3.3895e38, dtype=np.float32)
    rows.append(big * RNG.choice([-1.0, 1.0], size=(n, 5120)).astype(np.float32))  # bf16 max magnitude
    den = (RNG.integers(1, 128, size=(n, 5120)) * 2.0 ** -133).astype(np.float32)
    rows.append(den)  # bf16 denormals only
    mid = (RNG.integers(16, 31, size=(n, 5120)) + 0.5).astype(np.float32) * np.exp2(
        RNG.integers(-8, 0, size=(n, 5120))).astype(np.float32)
    mid[:, ::128] = 448.0
    rows.append(mid)  # e4m3 RNE midpoints with scale 1
    x = np.concatenate(rows, axis=0)
    xb = (x.view(np.uint32) >> 16).astype(np.uint16)  # truncate to bf16 bit patterns (values themselves are hostile)
    # round-to-nearest bf16 version as well (what a bf16 producer writes)
    xr = ((x.view(np.uint32) + 0x7FFF + ((x.view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16)
    xr = np.where(np.isfinite(x), xr, xb)
    return np.concatenate([xb, xr], axis=0)


def test_rows():
    xb = hostile_rows(64)
    g = xb.reshape(-1, 40, 128)
    bl, sl = live_quant(g)
    bn, sn = new_quant(g)
    check(f"D whole rows ({xb.shape[0]} rows x 5120, random/hostile): e4m3 bytes", np.array_equal(bl, bn),
          first_diff(bl, bn))
    check("D whole rows: scale bits", np.array_equal(sl, sn), first_diff(sl, sn))
    perm = RNG.permutation(128)
    bp, sp = live_quant(g[..., perm])
    check("D absmax order-free (permuted lanes give the same scales)", np.array_equal(sp, sl))
    words_l = pack_live(sl.reshape(-1, 10, 4))
    words_n = pack_new(sn.reshape(-1, 10, 4))
    check("D packed words of whole rows", np.array_equal(words_l, words_n))
    finite = np.isfinite(bf16_to_f32_rows(xb)).all(axis=1)
    br, sr = ref_quant(g[finite])
    check(f"D vs float64 + torch c10 reference ({int(finite.sum())} finite rows)",
          np.array_equal(bl[finite], br) and np.array_equal(sl[finite], sr), first_diff(bl[finite], br))


def bf16_to_f32_rows(xb):
    return (xb.astype(np.uint32) << 16).view(np.float32)


if __name__ == "__main__":
    np.seterr(all="ignore")  # inf/NaN inputs are part of the sweep
    assert_ieee_host()
    t0 = time.time()
    scales = test_scale_map()
    test_inv_exact(scales)
    test_element_map(scales)
    test_pack(scales)
    test_rows()
    print(f"{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)} ({time.time() - t0:.1f}s)")
    sys.exit(1 if FAIL else 0)
