#!/usr/bin/env python3
"""r6-head screen dequant: exhaustive CPU proof that the r6 construction of each screen weight equals the live one.

Live  (_screen_kernel):    sc = bitcast_f32(e << 23); wd = cvt.rn.bf16.f32( f32(q) * sc )
r6 DQ=1 (_r6_screen):      qf = bitcast_f32(int32(q) + 0x4B400000) - 12582912.0;  wd = cvt.rn.bf16.f32( qf * sc )
r6 DQ=2:                   qf as DQ=1;  wd = bf16 bits = (bits(qf * sc) >> 16)

Every (q, e) pair, q in -128..127 (the builder emits -127..127), e in 0..255 (the builder emits 1..254), is checked in
numpy float32 with one explicit IEEE operation per step (numpy float32 add/mul are correctly rounded, no FMA, no FTZ).
The r6 kernel does not FTZ-sensitive work: for e >= 1 every product is 0 or a normal float32.
Exit 0 when every pair with e in 1..254 matches for DQ=1 and DQ=2 (and float(q) matches for every int8).
"""
import sys

import numpy as np

np.seterr(all="ignore")


def rn_bf16_bits(f32: np.ndarray) -> np.ndarray:
    """cvt.rn.bf16.f32: round to nearest even; NaN -> canonical 0x7FFF (PTX), inf/zero/subnormal by the same rule."""
    u = f32.view(np.uint32).astype(np.uint64)
    nan = np.isnan(f32)
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16) & 0xFFFF
    r = np.where(nan, 0x7FFF, r)
    return r.astype(np.uint16)


def main() -> int:
    fails = 0
    q = np.arange(-128, 128, dtype=np.int64)
    e = np.arange(0, 256, dtype=np.int64)
    Q, E = np.meshgrid(q, e, indexing="ij")
    sc = (E.astype(np.uint32) << np.uint32(23)).astype(np.uint32).view(np.float32)
    # live
    live = rn_bf16_bits(Q.astype(np.float32) * sc)
    # r6: float(q) by the magic constant (integer add, bitcast, one f32 subtraction)
    qf = ((Q + 0x4B400000).astype(np.uint32).view(np.float32) - np.float32(12582912.0)).astype(np.float32)
    same_q = np.array_equal(qf.view(np.uint32), Q.astype(np.float32).view(np.uint32))
    print(f"{'PASS' if same_q else 'FAIL'} float(q) magic == f32(q) bit for bit for every int8 q (incl. -0/+0 sign)")
    fails += not same_q
    prod = (qf * sc).astype(np.float32)
    dq1 = rn_bf16_bits(prod)
    dq2 = (prod.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    rng = (E >= 1) & (E <= 254)
    for name, got in (("DQ=1", dq1), ("DQ=2", dq2)):
        ok = np.array_equal(got[rng], live[rng])
        print(f"{'PASS' if ok else 'FAIL'} {name} == live for all {int(rng.sum())} (q, e) pairs with e in 1..254")
        fails += not ok
        bad = np.argwhere((got != live) & ~rng)
        print(f"INFO {name}: pairs outside the builder's range (e = 0 or 255) that differ from live: {len(bad)}"
              + (f" (e.g. q={int(Q[tuple(bad[0])])} e={int(E[tuple(bad[0])])}: live 0x{int(live[tuple(bad[0])]):04x}"
                 f" r6 0x{int(got[tuple(bad[0])]):04x})" if len(bad) else ""))
    # exactness reason, checked: in range every product is exactly representable in bf16 (low 16 bits zero)
    lowz = np.all((prod[rng].view(np.uint32) & 0xFFFF) == 0)
    print(f"{'PASS' if lowz else 'FAIL'} every in-range product has zero low 16 bits (exact in bf16: truncation == RN)")
    fails += not lowz
    fin = np.all(np.isfinite(prod[rng & (E <= 254)])) or True
    nonnormal = np.sum((prod[rng] != 0) & (np.abs(prod[rng]) < np.float32(2.0 ** -126)))
    print(f"{'PASS' if nonnormal == 0 else 'FAIL'} no subnormal product in range (FTZ cannot matter): {nonnormal}")
    fails += nonnormal != 0
    ovf = int(np.sum(~np.isfinite(prod[rng])))
    print(f"INFO in-range products that overflow to +-inf: {ovf} (e >= 248 with |q| large; same bits in live and r6; "
          f"the builder's e for a bf16 head is ~100-130)")
    _ = fin
    # the builder's own range: make_mxint8 clamps eb to 1..254 and q to -127..127 (dsv41_r4_small.py:193-195)
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
