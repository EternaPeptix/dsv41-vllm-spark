#!/usr/bin/env python3
"""CPU emulation test (no GPU): v1 vs v2 op sequences, bit for bit.

  python3 tests/test_emulation.py [--quick]

1. emulator self-tests: fma32 vs an exact Fraction reference; e4m3 RN-satfinite and bf16 RNE vs torch casts.
2. quant: FlashInfer swizzled-kernel op sequence vs the v2 fused prologue (literal task -> lane -> smem mapping),
   codes, scales and dequantised bf16; FTZ vs non-FTZ amax multiply differ only in the documented band.
3. GEMV N=224, K=5120: v1 (skinny_mx_kernel<M,4,8>) vs v2 A8 and v2 fused-quant, M = 1..8, random / hostile
   weights and activations, grids 48 (production, MAXR 5), 56, 75, 28 (MAXR 8).
4. negative controls: v2 with a wrong shuffle source lane, and v1 with SPLIT 4, must NOT match (the test has teeth).
Exit 0 only if every comparison that must match is bit-identical and every negative control mismatches.
"""
from __future__ import annotations

import os
import sys
import time
from fractions import Fraction

import numpy as np

np.seterr(all="ignore")  # NaN-bit casts in the bf16 table / huge-value quant rows are expected

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emu  # noqa: E402

QUICK = "--quick" in sys.argv
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def bits32(a):
    return emu.f32(a).astype(np.float32).view(np.uint32)


# ------------------------------------------------------------------ 1. emulator self-tests
def exact_fma32(a, b, c):
    v = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    if v == 0:  # IEEE RN: an exact zero sum is +0 unless a*b and c are both -0
        prod_neg_zero = (a == 0 or b == 0) and (np.signbit(a) != np.signbit(b))
        return -0.0 if (prod_neg_zero and c == 0 and np.signbit(c)) else 0.0
    # round Fraction to fp32 RNE via candidate neighbours
    f = np.float32(float(v))  # float(v) is RN to double; fix below with exact comparisons
    cands = {f, np.nextafter(f, np.float32(np.inf)), np.nextafter(f, np.float32(-np.inf))}
    best = None
    for cnd in cands:
        d = abs(Fraction(float(cnd)) - v)
        key = (d, int(np.float32(cnd).view(np.uint32)) & 1)
        if best is None or key < best[0]:
            best = (key, cnd)
    return float(best[1])


def selftest():
    rng = np.random.default_rng(1)
    n = 3000 if QUICK else 20000
    a = emu.f32(rng.standard_normal(n) * np.exp2(rng.integers(-40, 40, n)))
    b = emu.f32(rng.standard_normal(n) * np.exp2(rng.integers(-40, 40, n)))
    c = emu.f32(-(a * b) * (1 + rng.integers(-3, 4, n) * 2.0 ** -24))  # near-cancellation
    c[::5] = emu.f32(rng.standard_normal(n // 5) * np.exp2(rng.integers(-140, 40, n // 5)))
    # exact midpoints: a*b = m + half-ulp patterns
    a[::7] = 1.0 + 2.0 ** -23
    b[::7] = 1.0 + 2.0 ** -23
    c[::7] = 0.0
    # subnormal results
    a[3::11] = emu.f32(2.0 ** -70 * (1 + rng.random(len(a[3::11]))))
    b[3::11] = emu.f32(2.0 ** -70 * (1 + rng.random(len(b[3::11]))))
    got = emu.fma32(a, b, c)
    ref = np.array([exact_fma32(x, y, z) for x, y, z in zip(a, b, c)])
    check("fma32 == exact Fraction fp32 FMA", np.array_equal(bits32(got), bits32(ref)), f"({n} triples)")
    try:
        import torch
    except Exception:
        print("[SKIP] torch cast cross-checks (no torch)")
        return
    allbf = np.arange(65536, dtype=np.uint32)
    vals = emu.bf16_bits_to_f(allbf)
    fin = np.isfinite(vals)
    t = torch.from_numpy(vals[fin].astype(np.float32))
    ok_bf = np.array_equal(emu.f_to_bf16_bits(vals[fin]), t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
    check("bf16 RNE == torch on all finite bf16", ok_bf)
    xs = np.concatenate([emu.E4M3[:127], (emu.E4M3[:126] + emu.E4M3[1:127]) / 2, rng.random(200000) * 448.0])
    xs = np.concatenate([xs, -xs, [0.0, -0.0]])
    xs = emu.f32(xs)
    ours = emu.e4m3_rn_satfinite(xs)
    th = torch.from_numpy(xs.astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    check("e4m3 RN == torch float8_e4m3fn cast on [-448, 448] incl. all ties", np.array_equal(ours, th),
          f"({len(xs)} values)")


# ------------------------------------------------------------------ inputs
def hostile_bf16(M, K, seed, huge=True):
    """bf16 bits [M, K]: block magnitude spread, zeros and -0, subnormals, the FTZ band, e4m3 ties, ceil edges,
    near-max values (finite, NaN-free)."""
    rng = np.random.default_rng(seed)
    nb = K // 32
    x = rng.standard_normal((M, nb, 32)) * np.exp2(rng.integers(-30, 21, (M, nb, 1)))
    kind = rng.integers(0, 10, (M, nb, 1))
    lane = np.arange(32).reshape(1, 1, 32)
    e = rng.integers(-20, 10, (M, nb, 1)).astype(float)
    pos = rng.integers(0, 32, (M, nb, 1))
    k0 = np.where(lane % 3 == 1, -0.0, 0.0) * np.ones((M, nb, 32))
    k1 = rng.integers(-127, 128, (M, nb, 32)) * 2.0 ** -133
    band = 448.0 * 2.0 ** -127 * (1.0 + rng.random((M, nb, 1)) * 0.99)
    k2 = np.where(lane == pos, band, rng.random((M, nb, 32)) * band)
    k3 = (rng.integers(0, 240, (M, nb, 32)) + 0.5) * np.exp2(e - 8)
    k3 = np.where(lane == 0, 448.0 * np.exp2(e), k3)
    k4 = np.where(lane == 0, 448.0 * np.exp2(e) * (1 + 2.0 ** -7), x)
    top, spread = (3.3895313892515355e38, 2.0 ** 100) if huge else (2.0 ** 62 * 1.9921875, 2.0 ** 60)
    k5 = np.where(lane == 3, top, rng.standard_normal((M, nb, 32)) * spread)  # GEMV: keep fp32 sums finite
    k6 = np.where(lane % 2 == 0, -x, x)
    out = x
    for k, alt in enumerate((k0, k1, k2, k3, k4, k5, k6)):
        out = np.where(kind == k, alt, out)
    out = out.reshape(M, K)
    out[:, ::97] = 0.0
    return emu.f_to_bf16_bits(out)


def random_bf16(M, K, seed):
    rng = np.random.default_rng(seed)
    return emu.f_to_bf16_bits(rng.standard_normal((M, K)) * 0.05)


def weights(N, K, seed, hostile):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, (N, K)).astype(np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x7E  # no NaN codes (checkpoint weights are finite)
    if hostile:
        sc = rng.integers(100, 151, (N, K // 32))
        sc[:, ::7] = rng.integers(0, 4, sc[:, ::7].shape)  # 2^-127 .. 2^-124: subnormal products
        codes[:, ::13] = 0x80  # -0
        codes[::5, :] = np.where(rng.random((len(codes[::5]), K)) < 0.5, 0x00, codes[::5, :])
    else:
        sc = rng.integers(118, 124, (N, K // 32))
    return codes, sc


def a8_operands(xbits, ftz=False):
    M, K = xbits.shape
    codes, ue = emu.fi_quant_blocks(xbits.reshape(M, K // 32, 32), ftz=ftz)
    return emu.v1_deq(codes, ue).reshape(M, K)


def ybits(y):
    return emu.f_to_bf16_bits(y)


# ------------------------------------------------------------------ 2. quant
def quant_tests():
    K = 5120
    for name, xb in (("hostile", hostile_bf16(16, K, 5)), ("random", random_bf16(8, K, 6))):
        M = xb.shape[0]
        c0, u0 = emu.fi_quant_blocks(xb.reshape(M, K // 32, 32), ftz=False)
        d0 = emu.v1_deq(c0, u0).reshape(M, K)
        for ftz in (False, True):
            fs = emu.fused_slices(xb, ftz=ftz)
            got = np.concatenate([fs(p) for p in range(8)], axis=1).reshape(M, K)
            cf, uf = emu.fi_quant_blocks(xb.reshape(M, K // 32, 32), ftz=ftz)
            df = emu.v1_deq(cf, uf).reshape(M, K)
            check(f"fused prologue mapping == FlashInfer quant + v1 dequant ({name}, ftz={ftz})",
                  np.array_equal(ybits(got), ybits(df)))
        c1, u1 = emu.fi_quant_blocks(xb.reshape(M, K // 32, 32), ftz=True)
        diff = u0 != u1
        if diff.any():
            amax = np.abs(emu.bf16_bits_to_f(xb)).reshape(M, K // 32, 32).max(-1)
            inband = (amax[diff] > 448.0 * 2.0 ** -127) & (amax[diff] < 448.0 * 2.0 ** -126 * 1.0000001)
            check(f"FTZ-mode scale differences confined to the amax*(1/448) subnormal band ({name})", inband.all(),
                  f"({int(diff.sum())} blocks differ)")
        else:
            print(f"[INFO] {name}: FTZ and non-FTZ give identical scales on this input")
    # independent cross-check of the e4m3 step: torch cast of the clamped scaled values
    try:
        import torch
        xb = hostile_bf16(8, K, 9)
        M = 8
        blocks = xb.reshape(M, K // 32, 32)
        c0, u0 = emu.fi_quant_blocks(blocks)
        inv = emu.fi_inv_scale(u0)
        v = np.clip(emu.mul32(emu.bf16_bits_to_f(blocks), inv[..., None]), -448, 448)
        th = torch.from_numpy(v.astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        check("quant codes == torch e4m3 cast of the clamped scaled values", np.array_equal(c0, th))
    except ImportError:
        pass


# ------------------------------------------------------------------ 3. GEMV
def gemv_tests():
    N, K = 224, 5120
    Ms = [1, 2, 4, 6] if QUICK else list(range(1, 9))
    for wname, hostile in (("rand-w", False), ("hostile-w", True)):
        codes, sc = weights(N, K, 11 if hostile else 10, hostile)
        for M in Ms:
            for xname, xb in (("rand-x", random_bf16(M, K, 100 + M)), ("hostile-x", hostile_bf16(M, K, 200 + M, huge=False))):
                t0 = time.time()
                xdeq = a8_operands(xb)
                y1, t1 = emu.v1_gemv(xdeq, codes, sc, ret_t=True)
                ya, ta = emu.v2_gemv(emu.a8_slices(xdeq), codes, sc, grid=48, ret_t=True)
                yq, tq = emu.v2_gemv(emu.fused_slices(xb, ftz=False), codes, sc, grid=48, ret_t=True)
                ref = ybits(y1)
                # compare the fp32 pre-rounding sums too: bf16 rounding alone would hide small fp32 differences
                check(f"v2 A8 == v1  M={M} {wname} {xname} grid48 (bf16 out + fp32 sums)",
                      np.array_equal(ybits(ya), ref) and np.array_equal(bits32(ta), bits32(t1)))
                check(f"v2 QF == v1(fi_quant) M={M} {wname} {xname} grid48 (bf16 out + fp32 sums)",
                      np.array_equal(ybits(yq), ref) and np.array_equal(bits32(tq), bits32(t1)),
                      f"({time.time() - t0:.1f}s, out zeros {int((ref & 0x7FFF == 0).sum())}, "
                      f"-0 {int((ref == 0x8000).sum())}, fp32 subnormal sums {int(((np.abs(t1) < 2.0**-126) & (t1 != 0)).sum())})")
    codes, sc = weights(N, K, 12, True)
    M = 5
    xb = hostile_bf16(M, K, 77, huge=False)
    xdeq = a8_operands(xb)
    y1, t1 = emu.v1_gemv(xdeq, codes, sc, ret_t=True)
    ref = ybits(y1)
    for g in ((56, 75, 28) if not QUICK else (28,)):
        yg, tg = emu.v2_gemv(emu.a8_slices(xdeq), codes, sc, grid=g, ret_t=True)
        check(f"v2 A8 == v1  M=5 grid {g} (MAXR {5 if -(-N // g) <= 5 else 8}) (bf16 out + fp32 sums)",
              np.array_equal(ybits(yg), ref) and np.array_equal(bits32(tg), bits32(t1)))
    return codes, sc, xdeq, (ref, t1)


# ------------------------------------------------------------------ 4. negative controls
def negative_controls(codes, sc, xdeq, refs):
    ref, t1 = refs
    xs = emu.a8_slices(xdeq)

    def shifted(p):
        s = xs(p).copy()
        s[:, 32:40] = np.roll(s[:, 32:40], 1, axis=1)  # second vectors paired with the wrong activations
        return s

    got = ybits(emu.v2_gemv(shifted, codes, sc, grid=48))
    check("negative control: mis-mapped second vectors MISMATCH", not np.array_equal(got, ref),
          f"({int((got != ref).sum())} of {ref.size} differ)")
    M, K = xdeq.shape
    # SPLIT-4 K partition (different reduction tree): recompute via v1 with a forced 4-way partition
    kv = K // 16
    wf = emu.E4M3[codes].reshape(codes.shape[0], kv, 16)
    xf = xdeq.reshape(M, kv, 16)
    s_all = emu.e8m0_to_float(sc)
    N = codes.shape[0]
    rows = np.arange(N)
    parts = []
    for part in range(4):
        kb, ke = part * 80, part * 80 + 80
        lane_acc = np.zeros((N, 32, M))
        for lane in range(32):
            acc = np.zeros((N, M))
            k = kb + lane
            while k < ke:
                d = emu.dot16(wf[rows, k][:, None, :], xf[:, k][None, :, :])
                acc = emu.fma32(s_all[rows, k >> 1][:, None], d, acc)
                k += 32
            lane_acc[:, lane] = acc
        v = lane_acc
        for o in (16, 8, 4, 2, 1):
            v = emu.add32(v, v[:, np.arange(32) ^ o])
        parts.append(v[:, 0])
    t = np.zeros((N, M))
    for q in range(4):
        t = emu.add32(t, parts[q])
    got4 = ybits(emu.bf16_rn(t).T)
    t4 = bits32(t.T)
    check("negative control: SPLIT 4 (different reduction order) MISMATCHES v1 SPLIT 8 in the fp32 sums",
          not np.array_equal(t4, bits32(t1)),
          f"(fp32 sums: {int((t4 != bits32(t1)).sum())} of {t4.size} differ; bf16 outputs: "
          f"{int((got4 != ref).sum())} differ - why the device test compares on many inputs)")


if __name__ == "__main__":
    t0 = time.time()
    selftest()
    quant_tests()
    c, s, x, r = gemv_tests()
    negative_controls(c, s, x, r)
    print(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}  ({time.time() - t0:.0f}s)")
    sys.exit(1 if FAIL else 0)
