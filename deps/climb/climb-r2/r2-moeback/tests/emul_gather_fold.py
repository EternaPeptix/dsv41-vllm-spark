#!/usr/bin/env python3
"""G25 end-to-end CPU emulation: old op sequence vs new op sequence, element by element, explicit roundings.

OLD (production today, per output element (t, d)):
  acc = +0.0f; for k in 0..5 (fixed order): if id[t,k] >= 0: acc = fma.rn.f32(f32(x[src(t,k), d]), w[t,k], acc)
  routed = cvt.rn.bf16(acc)                                  # _gated_gather store
  out = c10 RNE( f32(shared[t,d]) + f32(routed) )           # torch CUDAFunctor_add<BFloat16>, next kernel
  (status failed: acc stays +0.0f, the loop is skipped)
NEW (_gated_gather_add):
  identical acc loop, routed = cvt.rn.bf16(acc), out = cvt.rn.bf16( add.rn.f32(f32(routed), f32(shared)) )
NEGATIVE CONTROL (what must NOT be built): out = cvt.rn.bf16( fma-chain(acc starting from f32(shared)) ),
  i.e. accumulating the shared value into the fp32 accumulator. Shown to differ, so the test is sensitive.

fma.rn.f32 is emulated exactly: the product of a bf16 (8 significant bits) and an fp32 (24) is exact in
float64; the sum with the fp32 accumulator is formed with an error-free TwoSum and the final rounding to fp32
corrects the only case where double rounding could bite (a float64 sum lying exactly on an fp32 midpoint).
Inputs: random (normal, wide dynamic range), recorded-like (top-k weights that sum to the routed scale),
and hostile (zeros of both signs, subnormals, max-magnitude, exact-tie constructions, cancellation,
status failed, -1 expert ids). numpy only; run anywhere (512S1: ~/exo-venv313/bin/python3).
"""
import sys

import numpy as np

from emul_add_exhaustive import bf16_bits_to_f32, c10_round_f32_to_bf16, rne_f64_to_bf16


def f32_to_bf16_bits_rne(x32):
    return rne_f64_to_bf16(np.asarray(x32, np.float32).astype(np.float64))


def fma_f32(v32, w32, acc32):
    """Exact emulation of fma.rn.f32(v, w, acc) for finite operands."""
    p = v32.astype(np.float64) * w32.astype(np.float64)          # exact (<= 32 significant bits)
    a = acc32.astype(np.float64)
    s = p + a
    bb = s - p
    e = (p - (s - bb)) + (a - bb)                                 # TwoSum: p + a == s + e exactly
    r = s.astype(np.float32)                                      # RNE f64 -> f32 (subnormals kept)
    up = np.nextafter(r, np.float32(np.inf))
    dn = np.nextafter(r, np.float32(-np.inf))
    r64 = r.astype(np.float64)
    lo = np.where(r64 > s, dn, r).astype(np.float64)              # f32 neighbours bracketing s
    hi = np.where(r64 > s, r, np.where(r64 < s, up, r)).astype(np.float64)
    mid = (lo + hi) / 2                                           # exact in f64
    on_mid = (s == mid) & (lo != hi) & (e != 0)
    fixed = np.where(e > 0, hi, lo).astype(np.float32)
    return np.where(on_mid, fixed, r)


def gather_acc(x_bits, ids, w, src, ok, init=None):
    """fp32 accumulator per (token, d) exactly as _gated_gather's loop (k ascending)."""
    m = ids.shape[0]
    d = x_bits.shape[1]
    acc = np.zeros((m, d), np.float32) if init is None else init.copy()
    if not ok:
        return acc
    x32 = bf16_bits_to_f32(x_bits)
    for k in range(ids.shape[1]):
        live = ids[:, k] >= 0
        v = x32[src[:, k]]
        new = fma_f32(v, np.broadcast_to(w[:, k:k + 1], v.shape).astype(np.float32), acc)
        acc = np.where(live[:, None], new, acc)
    return acc


def old_seq(x_bits, ids, w, src, shared_bits, ok):
    routed = f32_to_bf16_bits_rne(gather_acc(x_bits, ids, w, src, ok))
    with np.errstate(over="ignore", invalid="ignore"):
        s = (bf16_bits_to_f32(shared_bits) + bf16_bits_to_f32(routed)).astype(np.float32)   # torch: self + other
    return c10_round_f32_to_bf16(s)


def new_seq(x_bits, ids, w, src, shared_bits, ok):
    routed = f32_to_bf16_bits_rne(gather_acc(x_bits, ids, w, src, ok))
    with np.errstate(over="ignore", invalid="ignore"):
        s = (bf16_bits_to_f32(routed) + bf16_bits_to_f32(shared_bits)).astype(np.float32)   # add.rn.f32
    return f32_to_bf16_bits_rne(s)                                                         # cvt.rn.bf16


def wrong_seq(x_bits, ids, w, src, shared_bits, ok):
    return f32_to_bf16_bits_rne(gather_acc(x_bits, ids, w, src, ok, init=bf16_bits_to_f32(shared_bits)))


def to_bits(x):
    return f32_to_bf16_bits_rne(np.asarray(x, np.float32))


def make_case(rng, m, d, kind):
    topk = 6
    m_sum = ((m * topk + 31) // 32) * 32 + 64
    ids = rng.integers(0, 384, size=(m, topk)).astype(np.int32)
    src = rng.permutation(m_sum)[: m * topk].reshape(m, topk).astype(np.int64)
    w = rng.dirichlet(np.ones(topk), size=m).astype(np.float32) * np.float32(2.5)
    x = rng.standard_normal((m_sum, d)).astype(np.float32) * np.float32(0.05)
    sh = rng.standard_normal((m, d)).astype(np.float32) * np.float32(0.3)
    ok = True
    if kind == "wide":
        x *= np.exp2(rng.integers(-60, 60, size=(m_sum, d))).astype(np.float32)
        sh *= np.exp2(rng.integers(-60, 60, size=(m, d))).astype(np.float32)
    elif kind == "subnormal":
        x = (rng.standard_normal((m_sum, d)) * 2.0**-128).astype(np.float32)
        sh = (rng.standard_normal((m, d)) * 2.0**-130).astype(np.float32)
        w = np.abs(rng.standard_normal((m, topk))).astype(np.float32)
    elif kind == "zeros":
        x[:, ::3] = 0.0
        x[:, 1::3] = -0.0
        sh[:, ::2] = -0.0
        sh[:, 1::4] = 0.0
    elif kind == "maxmag":
        big = np.float32(bf16_bits_to_f32(np.array([0x7F7F], np.uint16))[0])
        x[:, ::5] = big * np.sign(x[:, ::5] + 1e-30)
        w = (w / np.float32(8.0)).astype(np.float32)
        sh[:, ::7] = big * np.float32(0.9) * np.sign(sh[:, ::7] + 1e-30)
    elif kind == "ties":
        # routed values in [1,2) and shared = 2^-8 * odd: f32 sums land exactly on bf16 midpoints
        x = (1.0 + rng.integers(0, 128, size=(m_sum, d)) / 128.0).astype(np.float32)
        ids[:, 1:] = -1
        w[:, 0] = 1.0
        sh = (rng.integers(0, 64, size=(m, d)) * 2 + 1).astype(np.float32) * np.float32(2.0**-8)
    elif kind == "cancel":
        sh = -bf16_bits_to_f32(to_bits(x[src[:, 0]] * w[:, :1]))
        ids[:, 1:] = -1
    elif kind == "neg_ids":
        ids[rng.random((m, topk)) < 0.3] = -1
    elif kind == "status_failed":
        ok = False
    return to_bits(x), ids, w, src, to_bits(sh), ok


def main():
    rng = np.random.default_rng(20261108)
    kinds = ["normal", "wide", "subnormal", "zeros", "maxmag", "ties", "cancel", "neg_ids", "status_failed"]
    total = bad = wrong_diff = 0
    for kind in kinds:
        for m in (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 48, 96):
            x, ids, w, src, sh, ok = make_case(rng, m, 256, kind)
            o = old_seq(x, ids, w, src, sh, ok)
            n = new_seq(x, ids, w, src, sh, ok)
            nan = ((o & 0x7F80) == 0x7F80) & ((o & 0x7F) != 0)
            diff = (o != n) & ~nan
            total += o.size
            bad += int(diff.sum())
            if diff.any():
                i = np.argwhere(diff)[0]
                print(f"MISMATCH kind={kind} m={m} first={tuple(i)} old={o[tuple(i)]:#06x} new={n[tuple(i)]:#06x}")
            wr = wrong_seq(x, ids, w, src, sh, ok)
            wrong_diff += int(((wr != o) & ~nan).sum())
        print(f"{kind:14s} ok")
    # FMA emulation self-check against exact rationals on a sample
    from fractions import Fraction
    v = bf16_bits_to_f32(rng.integers(0, 0x5F00, 4000).astype(np.uint16)) * np.where(rng.random(4000) < .5, -1, 1).astype(np.float32)
    ww = (rng.standard_normal(4000) * np.exp2(rng.integers(-20, 20, 4000))).astype(np.float32)
    aa = (rng.standard_normal(4000) * np.exp2(rng.integers(-40, 40, 4000))).astype(np.float32)
    got = fma_f32(v.astype(np.float32), ww, aa)
    for i in range(4000):
        exact = Fraction(float(v[i])) * Fraction(float(ww[i])) + Fraction(float(aa[i]))
        # correctly rounded fp32 of exact via float64 neighbours
        c = np.float32(float(exact))
        cands = [y for y in (np.nextafter(c, np.float32(-np.inf)), c, np.nextafter(c, np.float32(np.inf))) if np.isfinite(y)]
        best = min(cands, key=lambda y: (abs(Fraction(float(y)) - exact), int(np.array(y, np.float32).view(np.uint32)) & 1))
        if np.array(best, np.float32).view(np.uint32) != np.array(got[i], np.float32).view(np.uint32):
            print("FMA emulation error", v[i], ww[i], aa[i], got[i], best)
            sys.exit(2)
    print(f"fma emulation: 4000/4000 equal to exact-rational rounding")
    print(f"elements compared: {total}; OLD != NEW: {bad}; NEGATIVE CONTROL (shared folded into fp32 acc) "
          f"differs from OLD on {wrong_diff} elements")
    if bad or wrong_diff == 0:
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
