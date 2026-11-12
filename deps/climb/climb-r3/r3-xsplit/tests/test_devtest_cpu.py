#!/usr/bin/env python3
"""CPU checks of device_test.py / bench/xsplit_bench.py host code (no GPU, no library):
input generator invariants for every class and shape the device test uses, poison/compare helpers,
and the bench's fit. Catches host-side bugs before a window spends GPU time on them."""
import os
import sys

import numpy as np
import torch

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "bench"))
import device_test as dt  # noqa: E402
import xsplit_bench as xb  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        print("FAIL " + msg)
        FAILS.append(msg)


def main():
    cpu = torch.device("cpu")
    n_sets = 0
    for pbsx, topkx in ((64, 512), (32, 512), (32, 1024)):
        nmax = 2 + topkx // 64
        for T in (1, 2, 4, 6, 33, 64):
            for n in sorted({0, 1, 2, 3, 5, nmax // 2, nmax - 1, nmax}):
                for cls in range(7):
                    h = dt.gen_inputs(1, pbsx, topkx, T, n, cls)
                    n_sets += 1
                    tag = f"pbsx={pbsx} topkx={topkx} T={T} n={n} cls={cls}"
                    check(h["kv"].shape == (2, 64, 584), tag + " main cache shape")
                    check(h["kvx"].shape == (topkx // pbsx, pbsx, 584), tag + " extra cache shape")
                    check(h["idx"].shape == (T, 128) and h["idxx"].shape == (T, topkx), tag + " idx")
                    check(h["q"].shape == (T, 8, 512) and h["q"].dtype == np.uint16, tag + " q")
                    check(((h["len"] >= 0) & (h["len"] <= 128)).all(), tag + " len range")
                    check(((h["lenx"] >= 0) & (h["lenx"] <= topkx)).all(), tag + " lenx range")
                    rows_x = (topkx // pbsx) * pbsx
                    check((h["idx"] < 128).all() and (h["idxx"] < rows_x).all(), tag + " idx in cache")
                    for t in range(T):
                        check((h["idx"][t, h["len"][t]:] == -1).all(), tag + " main tail -1")
                        check((h["idxx"][t, h["lenx"][t]:] == -1).all(), tag + " extra tail -1")
                    if cls == 0:
                        b = h["kv"].reshape(2, -1)[:, :64 * 576].reshape(2, 64, 576)[:, :, :448]
                        check(not np.isin(b, (0x7F, 0xFF)).any(), tag + " normal set has no FP8 NaN")
                    if cls == 5:
                        check((h["idx"][:, 0] == -1).all() and (h["idxx"][:, 0] == -1).all(),
                              tag + " slot0: index 0 is -1 everywhere")
                    d = dt.to_dev(h, cpu)
                    check(d["q"].dtype == torch.bfloat16 and d["kv"].stride(0) == 64 * 584,
                          tag + " device tensors")
    # poison + compare helpers on a fake workspace
    ws = dict(buf=torch.zeros(4096 + 256 * 16 * 8 + 256 * 4 + 256 * 64, dtype=torch.uint8),
              cap=16, off=[0, 256, 256 + 16 * 16 * 8, 256 + 16 * 16 * 8 + 256])
    ws["buf"][:4].view(torch.int32).fill_(41)
    rng = np.random.default_rng(3)
    for _ in range(20):
        dt.poison_ws(ws, rng)
        dn = ws["buf"][ws["off"][2]:ws["off"][2] + 16 * 4].view(torch.int32)
        check(int(dn[0]) != 42, "poison never writes the next epoch")
        tm = ws["buf"][ws["off"][1]:ws["off"][1] + 16 * 16 * 8].view(torch.int64)
        check(int(tm[0]) >> 32 != 42, "poison tm tags never the next epoch")
    o1 = torch.zeros(2, 8, 512, dtype=torch.bfloat16)
    l1 = torch.zeros(2, 8)
    o2, l2 = o1.clone(), l1.clone()
    check(dt.compare(o1, l1, o2, l2) is None, "compare equal")
    o2.view(torch.int16)[1, 3, 7] = 1
    e = dt.compare(o1, l1, o2, l2)
    check(e is not None and "token 1, head 3, d 7" in e, f"compare locates the first diff ({e})")
    o2.copy_(o1)
    l2.view(torch.int32)[0, 0] = 1
    check(dt.compare(o1, l1, o2, l2) is not None, "compare sees an LSE bit flip (-0 vs +0 class)")
    s0, c = xb.fit([2, 4, 6, 8, 10], [10 + 0.5 * n for n in (2, 4, 6, 8, 10)])
    check(abs(s0 - 10) < 1e-9 and abs(c - 0.5) < 1e-9, "bench fit")
    print(f"input sets checked: {n_sets}")
    print("ALL PASS" if not FAILS else f"FAILURES: {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
