#!/usr/bin/env python3
"""G25 epilogue proof, exhaustive over every ordered pair of non-NaN bf16 values (2^32 minus NaN pairs).

Three lowerings of  out = bf16( f32(routed_bf16) + f32(shared_bf16) ):
  A  torch today: CUDAFunctor_add<BFloat16> = fp32 IEEE add (round-to-nearest-even, subnormals kept; the
     a + 1.0f*b form may be contracted by nvcc to fma.rn(1, b, a), which is the same single rounding), then
     c10::BFloat16 round_to_nearest_even on the fp32 bits (bias 0x7FFF + lsb, >> 16).
  B  Triton add.rn.f32 then cvt.rn.bf16.f32 (IEEE RNE conversion, subnormals kept): fp32 add as in A, then an
     independent RNE implementation (via float64, exact) of the f32 -> bf16 conversion.
  C  native add.bf16 / add.rn.bf16x2 (what LLVM may shrink A to): ONE rounding of the exact sum to bf16.
     The exact sum is formed in float64: when the exponent gap is <= 44 the float64 sum is exact; when it is
     larger the smaller operand is below 2^-37 of the larger, so both the float64 sum and the exact sum round
     to the larger operand in bf16 (no tie is possible), hence RN_bf16(f64 sum) == RN_bf16(exact sum) always.
Asserts A == B == C bit for bit on every pair (NaN results excluded; inf allowed) and prints the counts.
CPU only (numpy). Run on 512S1:  ~/exo-venv313/bin/python3 emul_add_exhaustive.py [--procs 8]
"""
import argparse
import multiprocessing as mp
import sys
import time

import numpy as np


def bf16_bits_to_f32(bits16: np.ndarray) -> np.ndarray:
    return (bits16.astype(np.uint32) << 16).view(np.float32)


def c10_round_f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """c10::BFloat16 round_to_nearest_even (non-NaN inputs)."""
    u = x.view(np.uint32)
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + bias) >> 16).astype(np.uint16)


def rne_f64_to_bf16(x: np.ndarray) -> np.ndarray:
    """Independent IEEE round-to-nearest-even from float64 to bf16 (8-bit significand, emin -126,
    subnormal quantum 2^-133, overflow to inf). Returns bf16 bit patterns."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.signbit(x)
    a = np.abs(x)
    m, e = np.frexp(a)                     # a = m * 2^e, m in [0.5, 1)
    # quantum: 2^(e-8) for normals (8 significant bits), 2^-133 for subnormals
    q_exp = np.maximum(e - 8, -133)
    scaled = np.ldexp(a, -q_exp)           # exact (power-of-two scaling)
    r = np.rint(scaled)                    # round half to even
    y = np.ldexp(r, q_exp)
    y = np.where(a == 0, 0.0, y)
    big = y > float(np.float32(bf16_bits_to_f32(np.array([0x7F7F], np.uint16))[0]))
    y32 = y.astype(np.float32)             # exact: y has <= 8 significant bits and is in bf16 range or inf
    y32 = np.where(big, np.float32(np.inf), y32)
    bits = (y32.view(np.uint32) >> 16).astype(np.uint16)
    lost = (y32.view(np.uint32) & 0xFFFF) != 0
    assert not lost.any(), "rounding produced a non-bf16 value"
    return np.where(sign, bits | np.uint16(0x8000), bits)


def chunk(args):
    a_lo, a_hi = args
    b_bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    b_nan = ((b_bits & 0x7F80) == 0x7F80) & ((b_bits & 0x7F) != 0)
    b_bits = b_bits[~b_nan]
    b32 = bf16_bits_to_f32(b_bits)
    b64 = b32.astype(np.float64)
    n = mism_ab = mism_ac = 0
    first = None
    with np.errstate(over="ignore", invalid="ignore"):
        for a in range(a_lo, a_hi):
            ab = np.uint16(a)
            if ((a & 0x7F80) == 0x7F80) and (a & 0x7F):
                continue
            a32 = bf16_bits_to_f32(np.array([ab], np.uint16))[0]
            s32 = (np.float32(a32) + b32).astype(np.float32)          # fp32 add, RNE, subnormals kept
            ok = ~np.isnan(s32)                                      # inf + -inf -> NaN: excluded
            A = c10_round_f32_to_bf16(s32[ok])
            B = rne_f64_to_bf16(s32[ok].astype(np.float64))
            s64 = np.float64(a32) + b64[ok]
            C = rne_f64_to_bf16(s64)
            n += int(ok.sum())
            dab = A != B
            dac = A != C
            mism_ab += int(dab.sum())
            mism_ac += int(dac.sum())
            if first is None and (dab.any() or dac.any()):
                i = int(np.flatnonzero(dab | dac)[0])
                first = (a, int(b_bits[ok][i]), int(A[i]), int(B[i]), int(C[i]))
    return n, mism_ab, mism_ac, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="only 4096 routed values (spread over all exponents)")
    args = ap.parse_args()
    t0 = time.time()
    if np.finfo(np.float32).tiny / 2 == 0:
        sys.exit("numpy float32 flushes subnormals here; cannot emulate")
    # Self-checks of the two rounding helpers against known values.
    probe = np.array([1.0, 1.0 + 2**-8, 1.0 + 3 * 2**-8, 2**-133, 2**-134, 3 * 2**-134, (2 - 2**-7) * 2.0**127,
                      (2 - 2**-8) * 2.0**127, 1e-45], dtype=np.float64)
    want = np.array([0x3F80, 0x3F80, 0x3F82, 0x0001, 0x0000, 0x0002, 0x7F7F, 0x7F80, 0x0000], np.uint16)
    got = rne_f64_to_bf16(probe)
    assert (got == want).all(), (got, want)
    assert (c10_round_f32_to_bf16(probe[:4].astype(np.float32)) == want[:4]).all()
    # Sensitivity: truncation (round-toward-zero) instead of RNE must be caught.
    s = (np.float32(1.0) + bf16_bits_to_f32(np.arange(0x3B00, 0x3C00, dtype=np.uint16))).astype(np.float32)
    assert (c10_round_f32_to_bf16(s) != (s.view(np.uint32) >> 16).astype(np.uint16)).any()
    np.seterr(over="ignore")
    ranges = [(lo, lo + 1024) for lo in range(0, 65536, 1024)]
    if args.quick:
        ranges = [(lo, lo + 64) for lo in range(0, 65536, 1024)]
    with mp.Pool(args.procs) as pool:
        res = pool.map(chunk, ranges)
    n = sum(r[0] for r in res)
    ab = sum(r[1] for r in res)
    ac = sum(r[2] for r in res)
    firsts = [r[3] for r in res if r[3] is not None]
    print(f"pairs compared (non-NaN result): {n}")
    print(f"A(torch f32 add + c10 RNE) != B(f32 add + IEEE cvt.rn): {ab}")
    print(f"A(torch) != C(native add.bf16, single rounding of exact sum): {ac}")
    if firsts:
        print("first mismatch (a_bits, b_bits, A, B, C):", [hex(x) for x in firsts[0]])
    print(f"elapsed {time.time() - t0:.1f}s  quick={args.quick}")
    if ab or ac:
        sys.exit(1)
    print("PASS: all three lowerings are bit-identical")


if __name__ == "__main__":
    main()
