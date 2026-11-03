#!/usr/bin/env python3
# Read-only. (1) does AR end track burst end? binned AR dur vs burst dur per class. (2) offsets of the weight-bound
# consumers after AR end, and burst end vs consumer start. (3) gate M<24 two-shot per-iteration ordering.
import sqlite3, sys, re, collections, bisect
IT = 60
def q(x, f):
    if not x: return float('nan')
    x = sorted(x); return x[min(len(x) - 1, int(f * len(x)))]
def slope(xs, ys):
    n = len(xs)
    if n < 5: return float('nan'), float('nan')
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    return (sxy / sxx if sxx else float('nan')), (sxy / (sxx * syy) ** .5 if sxx and syy else float('nan'))
for arg in sys.argv[1:]:
    p, l = arg.split('=')
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    S = dict(c.execute("select id,value from StringIds"))
    rows = []
    for a, b, s, d, gx, tx in c.execute("select start,end,shortName,demangledName,gridX,blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
        rows.append((a, b, S.get(d) or S.get(s) or '', gx, tx))
    other = [r for r in rows if 'l2pf' not in r[2]]
    ostart = [r[0] for r in other]
    pf = [r for r in rows if 'l2pf' in r[2]]
    pfs = [r[0] for r in pf]
    ar = [r for r in rows if 'ar_kernel_pl' in r[2]]
    print(f"\n=== {l}")
    cl = collections.defaultdict(list)
    for r in ar:
        a, b, n, gx, tx = r
        m = re.search(r'ar_kernel_pl<\(int\)(\d+)', n)
        mode = m.group(1)
        i = bisect.bisect_right(pfs, a + 1000) - 1
        if i < 0 or not (pf[i][1] > a and abs(pf[i][0] - a) < 2000): continue
        pd = (pf[i][1] - pf[i][0]) / 1e3
        sz = 'S' if pd < 12 else ('M' if pd < 24 else 'L')
        cl[(mode, sz)].append((pd, (b - a) / 1e3, r, pf[i]))
    print("class  n/it  corr slope(AR on burst) | AR p50 by burst-dur quartile | burstdur quartile edges")
    for k in sorted(cl):
        v = cl[k]
        if len(v) / IT < 1: continue
        xs = [x[0] for x in v]; ys = [x[1] for x in v]
        sl, co = slope(xs, ys)
        ed = [q(xs, .25), q(xs, .5), q(xs, .75)]
        bins = [[], [], [], []]
        for x, y in zip(xs, ys):
            bins[sum(1 for e in ed if x > e)].append(y)
        print(f"{k[0]} {k[1]} {len(v)/IT:6.2f} {co:5.2f} {sl:5.2f} | " + " ".join(f"{q(b,.5):5.1f}" for b in bins) + " | " + " ".join(f"{e:5.1f}" for e in ed))
    print("--- kernel sequence after AR end (offset from AR end, us, p50; first 7 distinct non-l2pf), and burst end relative to AR end")
    for k in sorted(cl):
        v = cl[k]
        if len(v) / IT < 5: continue
        seqs = collections.defaultdict(list)
        keyc = collections.Counter()
        for pd, dur, r, f in v:
            i = bisect.bisect_left(ostart, r[1])
            seq = []
            while i < len(other) and len(seq) < 7 and other[i][0] - r[1] < 200000:
                seq.append((other[i][2][:30], (other[i][0] - r[1]) / 1e3, (other[i][1] - other[i][0]) / 1e3))
                i += 1
            key = tuple(s[0] for s in seq)
            keyc[key] += 1
            seqs[key].append(seq)
        key, cnt = keyc.most_common(1)[0]
        print(f"{k}: top seq n/it {cnt/IT:.1f} of {len(v)/IT:.1f}; burstend-AR-end p50 {q([(f[1]-r[1])/1e3 for _,_,r,f in v],.5):.1f}")
        for j, name in enumerate(key):
            print(f"    {name:32s} start +{q([s[j][1] for s in seqs[key]],.5):7.1f}  dur {q([s[j][2] for s in seqs[key]],.5):7.1f}")
    # the bf16 gate kernel relative to AR end for M class
    for k in sorted(cl):
        v = cl[k]
        if k[1] != 'M' or len(v) / IT < 5: continue
        kk = []; marg = []
        for pd, dur, r, f in v:
            i = bisect.bisect_left(ostart, r[1])
            while i < len(other) and other[i][0] - r[1] < 200000:
                if 'Kernel2' in other[i][2]:
                    kk.append((other[i][0] - r[1]) / 1e3); marg.append((other[i][0] - f[1]) / 1e3); break
                i += 1
        print(f"{k} first bf16 Kernel2 start after AR end: p10 {q(kk,.1):.1f} p50 {q(kk,.5):.1f} p90 {q(kk,.9):.1f} n/it {len(kk)/IT:.1f}; Kernel2 start minus burst end p10 {q(marg,.1):.1f} p50 {q(marg,.5):.1f}")
    for k in sorted(cl):
        v = cl[k]
        if k != ('3', 'L'): continue
        kk = []; marg = []
        for pd, dur, r, f in v:
            i = bisect.bisect_left(ostart, r[1])
            while i < len(other) and other[i][0] - r[1] < 200000:
                if 'device_kernel' in other[i][2]:
                    kk.append((other[i][0] - r[1]) / 1e3); marg.append((other[i][0] - f[1]) / 1e3); break
                i += 1
        print(f"gather L: first device_kernel (q_b GEMM) start after AR end: p10 {q(kk,.1):.1f} p50 {q(kk,.5):.1f} p90 {q(kk,.9):.1f}; minus burst end p10 {q(marg,.1):.1f} p50 {q(marg,.5):.1f} p90 {q(marg,.9):.1f}")
    # inter-kernel gap on a side stream: l2pf-after-spin proxy: minimum launch gap between consecutive same-stream kernels in graph
    # kernel start vs previous kernel end on the AR's next kernel (already 0.4us). Print gap distribution AR end -> next kernel
    gaps = []
    for r in ar:
        i = bisect.bisect_left(ostart, r[1])
        if i < len(other): gaps.append((other[i][0] - r[1]) / 1e3)
    print(f"AR end -> next kernel gap p10 {q(gaps,.1):.2f} p50 {q(gaps,.5):.2f} p90 {q(gaps,.9):.2f}")
