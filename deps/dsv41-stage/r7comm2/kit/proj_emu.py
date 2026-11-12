#!/usr/bin/env python3
"""proj_emu.py: exact CPU emulation of the served projection kernel dsv41_oblit._proj_rows (H = 5120, BLOCK = 8192,
num_warps = 8; op order read from its PTX/SASS, ref/triton/*/_proj_rows.{ptx,sass}, proven by gate/projgate.py), and of
the HT13 must-fail mutants. Pure Python (fractions for exact FMA rounding), no numpy needed for the core.

Served op order, per row (thread t = 0..255, its 32 elements e(c, j) = c * 2048 + t * 8 + j, c = 0..3, j = 0..7, an
element whose 8-vector starts at >= 5120 is masked: x = d = +0.0):
    acc = x[1] * d[1]; acc = fma(x[0], d[0], acc); acc = fma(x[j], d[j], acc) for j = 2..31     (fp32, RN)
    warp butterfly: acc += shfl_xor(acc, m) for m = 16, 8, 4, 2, 1
    block: lanes 0..7 of warp 0 take warp sums w[0..7]; v += shfl_xor(v, m) for m = 4, 2, 1; s = v of lane 0
    y[e] = bf16_rn( fma(-s, d[e], x[e]) )
NaN results are canonical (fp32 0x7FFFFFFF, bf16 0x7FFF), as the r6 AR emulator assumes for GB10."""
import math
import struct
from fractions import Fraction

H = 5120
F32_NAN = 0x7FFFFFFF


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def bits32(f):
    if f != f:
        return F32_NAN
    return struct.unpack("<I", struct.pack("<f", f))[0]


def _round_f32(q):
    """exact rational -> fp32 bits, round to nearest even, subnormals kept (no FTZ), overflow to +-Inf"""
    if q == 0:
        return 0
    neg = q < 0
    a = -q if neg else q
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    # a in [2^e, 2^(e+1)); fp32 normal exponent range -126..127, 23 fraction bits
    if e < -126:
        e = -126  # subnormal: fixed quantum 2^-149
    quantum = Fraction(2) ** (e - 23)
    n = a / quantum
    fl = n.numerator // n.denominator
    rem = n - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    val = fl * quantum
    if val >= Fraction(2) ** 128:
        b = 0x7F800000
    else:
        b = bits32(float(val))  # exact: val is representable in fp32 (and thus in float64)
    return b | (0x80000000 if neg else 0)


def _nonfinite(*xs):
    return any(math.isinf(x) or math.isnan(x) for x in xs)


def fma32(a, b, c):
    """fp32 bits of fma(a, b, c) with one rounding; a, b, c are Python floats holding fp32 values"""
    if _nonfinite(a, b, c):
        r = a * b + c  # IEEE special-value rules are the same in binary64; finite parts cannot round here
        return bits32(r) if r == r else F32_NAN
    q = Fraction(a) * Fraction(b) + Fraction(c)
    if q == 0:
        # exact zero: sign by IEEE rules (RN): +0 unless both the product and c are -0 / negative-zero sums
        p0 = math.copysign(1.0, a) * math.copysign(1.0, b)
        if a * b == 0 and c == 0:
            return 0x80000000 if (p0 < 0 and math.copysign(1.0, c) < 0) else 0
        return 0
    return _round_f32(q)


def fmul32(a, b):
    if _nonfinite(a, b):
        r = a * b
        return bits32(r) if r == r else F32_NAN
    q = Fraction(a) * Fraction(b)
    if q == 0:
        return 0x80000000 if math.copysign(1.0, a) * math.copysign(1.0, b) < 0 else 0
    return _round_f32(q)


def fadd32(a, b):
    if _nonfinite(a, b):
        r = a + b
        return bits32(r) if r == r else F32_NAN
    q = Fraction(a) + Fraction(b)
    if q == 0:
        return 0x80000000 if (math.copysign(1.0, a) < 0 and math.copysign(1.0, b) < 0) else 0
    return _round_f32(q)


def bf16_rn(fbits):
    if (fbits & 0x7F800000) == 0x7F800000 and (fbits & 0x7FFFFF):
        return 0x7FFF
    lsb = (fbits >> 16) & 1
    return ((fbits + 0x7FFF + lsb) >> 16) & 0xFFFF


def _elems(t):
    out = []
    for c in range(4):
        base = c * 2048 + t * 8
        for j in range(8):
            out.append((base + j, base < H))
    return out


VARIANTS = ("served", "mp_first", "mp_warp", "mp_l2", "mp_unfused", "mp_chunk")


def proj_row(xb, db, variant="served"):
    """xb: 5120 bf16 bit patterns (ints); db: 5120 fp32 bit patterns -> 5120 bf16 bit patterns of y"""
    assert len(xb) == H and len(db) == H
    xb = [int(v) for v in xb]  # numpy scalars -> Python ints
    db = [int(v) for v in db]
    xf = [f32((v & 0xFFFF) << 16) for v in xb]
    df = [f32(v) for v in db]
    if variant == "mp_chunk":  # x elements of the vector at 5112 never loaded
        for e in range(5112, 5120):
            xf[e] = 0.0
    accs = []
    for t in range(256):
        el = _elems(t)
        xs = [xf[e] if ok else 0.0 for e, ok in el]
        ds = [df[e] if ok else 0.0 for e, ok in el]
        if variant == "mp_first":
            a = f32(fmul32(xs[0], ds[0]))
            a = f32(fma32(xs[1], ds[1], a))
        else:
            a = f32(fmul32(xs[1], ds[1]))
            a = f32(fma32(xs[0], ds[0], a))
        for j in range(2, 32):
            a = f32(fma32(xs[j], ds[j], a))
        accs.append(a)
    masks1 = (1, 2, 4, 8, 16) if variant == "mp_warp" else (16, 8, 4, 2, 1)
    wsum = []
    for w in range(8):
        v = accs[32 * w:32 * w + 32]
        for m in masks1:
            v = [f32(fadd32(v[l], v[l ^ m])) for l in range(32)]
        wsum.append(v[0])
    masks2 = (1, 2, 4) if variant == "mp_l2" else (4, 2, 1)
    v = wsum[:]
    for m in masks2:
        v = [f32(fadd32(v[l], v[l ^ m])) for l in range(8)]
    s = v[0]
    ns = -s
    y = []
    for e in range(H):
        if variant == "mp_unfused":
            yb = fadd32(xf[e], f32(fmul32(ns, df[e])))
        else:
            yb = fma32(ns, df[e], xf[e])
        y.append(bf16_rn(yb))
    return y


def selftest():
    """unit checks of the exact helpers against known identities (CPU)"""
    import random
    rnd = random.Random(7)
    bad = 0
    # fma vs separately rounded mul+add must differ somewhere on random fp32 data, and agree when exact
    diff = 0
    for _ in range(2000):
        a, b, c = (f32(rnd.getrandbits(32) & 0xBF7FFFFF) for _ in range(3))
        if _nonfinite(a, b, c):
            continue
        if fma32(a, b, c) != fadd32(f32(fmul32(a, b)), c):
            diff += 1
    bad += diff == 0
    # fp32 add of representable values: compare with numpy if available
    try:
        import numpy as np
        for _ in range(3000):
            a, b = (np.float32(f32(rnd.getrandbits(32))) for _ in range(2))
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            with np.errstate(all="ignore"):
                ref = int(np.float32(a + b).view(np.uint32))
                refm = int(np.float32(a * b).view(np.uint32))
            if math.isnan(f32(ref)):
                continue
            bad += fadd32(float(a), float(b)) != ref
            if not math.isnan(f32(refm)):
                bad += fmul32(float(a), float(b)) != refm
    except ImportError:
        pass
    # bf16 RN-even ties
    bad += bf16_rn(0x3F808000) != 0x3F80  # tie to even (down)
    bad += bf16_rn(0x3F818000) != 0x3F82  # tie to even (up)
    bad += bf16_rn(0x7FC12345) != 0x7FFF
    return bad == 0, diff


if __name__ == "__main__":
    ok, d = selftest()
    print(f"proj_emu selftest {'PASS' if ok else 'FAIL'} (fma != mul+add on {d} of 2000 random triples)")


# ---------------------------------------------------------------------------------------------- planted rows -------
def _bf(v):
    """bf16 bits of an exactly representable value"""
    b = bits32(v)
    assert b & 0xFFFF == 0, v
    return b >> 16


def planted_row(a=0.5, b_probe=None):
    """(x bf16 bits, d fp32 bits) of one row on which EVERY served-vs-mutant difference shows in y (proj_emu VARIANTS):
    thread 2 makes the first-two-products order matter (x1*d1 rounds, x0*d0 cancels it), warp 7 the lane butterfly
    order (2^24 + 1 - 2^24), warps 1 / 4 / 5 the second-level order, a probe with x = 0, d = 1 shows -s in y, a
    near-cancelling probe (x ~ s * d with d = 1 + 2^-23) shows the FFMA single rounding, and x = 1, d = 0 at element
    5112 shows the last vector's mask. Every other element is +0.0."""
    x = [0] * H
    d = [0] * H
    def put(e, xv, dv):
        x[e] = _bf(xv)
        d[e] = bits32(dv)
    one_u = 1.0 + 2.0 ** -23
    put(16, -1.5 * 2 ** 22, 1.0)
    put(17, 1.5 * 2 ** 22, one_u)
    put(256, -4096.0, 4096.0)
    put(1280, 4096.0, 4096.0)
    put(1024, a, 1.0)
    put(1792, 4096.0, 4096.0)
    put(1920, 1.0, 1.0)
    put(1800, -4096.0, 4096.0)
    put(600, 0.0, 1.0)
    if b_probe is not None:
        put(700, b_probe, one_u)
        put(701, -b_probe, one_u)
    put(5112, 1.0, 0.0)
    return x, d


def find_planted():
    """search the two free values until every mutant differs from the served kernel on the planted row"""
    for a in (0.5, 0.75, 0.375, 0.625):
        x, d = planted_row(a)
        s_like = None
        y0 = proj_row(x, d)
        s_like = -f32(y0[600] << 16)
        for k in range(-3, 4):
            b = float(f32(_bf_round(s_like * (1.0 + 2.0 ** -23)) << 16)) + k * 2.0 ** -6
            try:
                x, d = planted_row(a, b)
            except AssertionError:
                continue
            y = proj_row(x, d)
            diffs = {v: sum(p != q for p, q in zip(y, proj_row(x, d, v))) for v in VARIANTS[1:]}
            if all(diffs.values()):
                return a, b, diffs
    return None


def _bf_round(v):
    return bf16_rn(bits32(v))
