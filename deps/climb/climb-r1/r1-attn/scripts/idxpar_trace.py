#!/usr/bin/env python3
"""r1-attn: read the IDXPAR effect out of an nsys sqlite export (60-step decode window, --cuda-graph-trace=node).

    python3 idxpar_trace.py q0.4.sqlite [more.sqlite ...]

Per index-source layer (2/8/14/20/24/28/32/36, by the order of sm120_fp8_paged_mqa_logits launches) it reports,
in us relative to the end of _q_kv_norm_quant_kernel (t0):
  prolog   t0 -> logits start (the critical-path prologue IDXPAR shortens; today 52-55 kv-source, 42-45 index-only)
  mainS    main wq_b GEMM start (the 1x32 device_kernel on the stream of fusedDeepseekV4QNormRope...Insert)
  idxS/idxD indexer wq_b GEMM start / duration (the other 1x32 device_kernel; today it starts 23 / 14 us late)
  woaS     first l2pf_kernel start after t0 (the WOA prefetch fork; must come after the indexer wq_b ENDS)
  ovl      1 if the two GEMMs overlap in time
  streams  distinct streams carrying kernels in the window, l2pf included (stock 3 kv-source / 2 index-only;
           IDXPAR 4 / 3: aux1, high, (medium), l2pf -- main itself is idle until the logits)
Expect with DSV41_IDXPAR=1: idxS ~0-2 us, ovl=1, prolog ~30-37 us, woaS >= idxS+idxD, per-forward saving
~120-150 us; logits/attention kernels unchanged in count and grid.
"""
import collections
import sqlite3
import statistics as st
import sys

LAY = [2, 8, 14, 20, 24, 28, 32, 36]


def rows(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    S = dict(c.execute("select id,value from StringIds"))
    return [(a, b, S[n], gx, gy, s) for a, b, n, gx, gy, s in c.execute(
        "select start,end,shortName,gridX,gridY,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]


def analyse(path):
    k = rows(path)
    lg = [i for i, x in enumerate(k) if x[2] == "sm120_fp8_paged_mqa_logits"]
    res = collections.defaultdict(list)
    bad = 0
    for j, i in enumerate(lg):
        L = LAY[j % 8]
        q = i
        while q > 0 and k[q][2] != "_q_kv_norm_quant_kernel":
            q -= 1
        t0 = k[q][1]
        win = [x for x in k[q + 1:i] if x[0] >= k[q][0]]
        # IDXPAR may start a GEMM on another stream before / after kernels of other streams: take the window by time
        gem = [x for x in k[q + 1:i + 1] if x[2] == "device_kernel" and x[4] == 32 and x[0] < k[i][0]]
        ins = [x for x in win if x[2].startswith("fusedDeepseekV4QNormRope")]
        if len(gem) != 2 or not ins:
            bad += 1
            continue
        same = [g for g in gem if g[5] == ins[0][5] and g[0] <= ins[0][0]]
        mw = same[-1] if same else gem[0]
        iw = gem[1] if mw is gem[0] else gem[0]
        woa = [x for x in k[q + 1:i + 40] if x[2] == "l2pf_kernel" and x[0] >= t0]
        us = lambda t: (t - t0) / 1e3  # noqa: E731
        res[L].append((
            us(k[i][0]), us(mw[0]), us(iw[0]), (iw[1] - iw[0]) / 1e3,
            us(woa[0][0]) if woa else float("nan"),
            1.0 if (mw[0] < iw[1] and iw[0] < mw[1]) else 0.0,
            float(len({x[5] for x in win})),
        ))
    return res, len(lg), bad


def main():
    for p in sys.argv[1:]:
        res, n, bad = analyse(p)
        print(f"{p}: logits launches {n}, unparsed {bad}")
        print("  L    n  prolog   mainS    idxS    idxD    woaS   ovl  streams")
        tot = 0.0
        for L in LAY:
            v = res.get(L)
            if not v:
                continue
            m = [st.mean(c) for c in zip(*v)]
            tot += m[0]
            print(f" {L:2d} {len(v):4d} " + " ".join(f"{x:7.2f}" for x in m[:5]) + f" {m[5]:5.2f} {m[6]:7.1f}")
        print(f"  sum of prologues per forward: {tot:.1f} us")


if __name__ == "__main__":
    main()
