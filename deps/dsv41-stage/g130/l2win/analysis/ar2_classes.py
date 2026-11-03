#!/usr/bin/env python3
# Read-only. Classify rdma AR kernels by (mode template, gridX, paired l2pf grid/size) and print duration
# percentiles, burst overrun relative to AR end, next non-l2pf kernel gap, co-resident kernels during overrun.
import sqlite3, sys, re, collections, statistics as st, bisect
IT = 60
def q(x, f):
    if not x: return float('nan')
    x = sorted(x); return x[min(len(x) - 1, int(f * len(x)))]
for arg in sys.argv[1:]:
    p, l = arg.split('=')
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    S = dict(c.execute("select id,value from StringIds"))
    rows = []
    for a, b, s, d, gx, tx in c.execute("select start,end,shortName,demangledName,gridX,blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
        rows.append((a, b, S.get(d) or S.get(s) or '', gx, tx))
    other = [r for r in rows if 'l2pf' not in r[2] and 'spin_kernel' not in r[2]]
    ostart = [r[0] for r in other]
    pf = [r for r in rows if 'l2pf' in r[2]]
    pfs = [r[0] for r in pf]
    ar = [r for r in rows if 'ar_kernel' in r[2]]
    if l == 'prose':
        print("full names:")
        for n, k in collections.Counter(r[2] for r in ar).most_common(8):
            print(' ', k, n[:160])
    recs = []
    for r in ar:
        a, b, n, gx, tx = r
        m = re.search(r'ar_kernel_pl<\(int\)(\d+), \(bool\)(\d)', n)
        tmpl = m.group(1) + ('b' if m.group(2) == '1' else '') if m else n[:12]
        i = bisect.bisect_right(pfs, a + 1000) - 1
        mine = None
        if i >= 0 and pf[i][1] > a and abs(pf[i][0] - a) < 2000:
            mine = pf[i]
        k = bisect.bisect_left(ostart, b)
        nx = other[k] if k < len(other) else None
        recs.append(dict(t=tmpl, gx=gx, dur=(b - a) / 1e3, a=a, b=b,
                         pg=mine[3] if mine else 0,
                         pd=(mine[1] - mine[0]) / 1e3 if mine else None,
                         pe=(mine[1] - b) / 1e3 if mine else None,
                         ps=(mine[0] - a) / 1e3 if mine else None,
                         nx=nx[2][:34] if nx else '', ng=(nx[0] - b) / 1e3 if nx else None, pfr=mine))
    print(f"\n=== {l}: AR/it {len(ar)/IT:.1f} l2pf/it {len(pf)/IT:.1f} unpaired-pf {len(pf)-sum(1 for r in recs if r['pfr'])}")
    print("mode grid pfgrid sizeclass n/it  p10   p50   p90   | pfdur p50 | pfend-arend p10 p50 p90 frac>0 | next-gap p50 | us/step sum(dur)")
    by = collections.defaultdict(list)
    for r in recs:
        sz = 'none'
        if r['pd'] is not None:
            sz = 'S<12' if r['pd'] < 12 else ('M<24' if r['pd'] < 24 else 'L>=24')
        by[(r['t'], sz, r['pg'], r['gx'])].append(r)
    for k in sorted(by, key=lambda x: (x[0], x[1], x[2], x[3])):
        v = by[k]
        if len(v) / IT < 0.4: continue
        d = [x['dur'] for x in v]
        pd = [x['pd'] for x in v if x['pd'] is not None]
        pe = [x['pe'] for x in v if x['pe'] is not None]
        ng = [x['ng'] for x in v if x['ng'] is not None and x['ng'] < 500]
        fr = sum(1 for e in pe if e > 0) / len(pe) if pe else 0
        print(f"{k[0]:>4s} {k[3]:4d} {k[2]:4d} {k[1]:6s} {len(v)/IT:6.2f} {q(d,.1):6.1f} {q(d,.5):6.1f} {q(d,.9):6.1f} | {q(pd,.5):6.1f} | {q(pe,.1):6.1f} {q(pe,.5):6.1f} {q(pe,.9):6.1f} {fr:5.2f} | {q(ng,.5):6.1f} | {sum(d)/IT:7.0f}")
    # aggregated by mode + sizeclass
    print("--- aggregated mode/size")
    by2 = collections.defaultdict(list)
    for r in recs:
        sz = 'none'
        if r['pd'] is not None:
            sz = 'S<12' if r['pd'] < 12 else ('M<24' if r['pd'] < 24 else 'L>=24')
        by2[(r['t'], sz)].append(r)
    for k in sorted(by2):
        v = by2[k]
        d = [x['dur'] for x in v]
        print(f"{k[0]:>4s} {k[1]:6s} n/it {len(v)/IT:6.2f} p10 {q(d,.1):6.1f} p50 {q(d,.5):6.1f} p90 {q(d,.9):6.1f} grids {collections.Counter(x['gx'] for x in v).most_common(4)}")
    # next kernels and overruns
    print("--- next non-pf kernel after AR, with burst, by size class")
    nk = collections.defaultdict(list)
    for r in recs:
        if r['pd'] is None: continue
        sz = 'S<12' if r['pd'] < 12 else ('M<24' if r['pd'] < 24 else 'L>=24')
        nk[(r['t'], sz, r['nx'])].append(r['ng'])
    for k, v in sorted(nk.items(), key=lambda x: -len(x[1]))[:14]:
        print(f"  {k[0]:>4s} {k[1]:6s} {k[2]:36s} n/it {len(v)/IT:5.2f} gap p50 {q(v,.5):6.1f}")
    # co-residency: kernels overlapping the tail of the burst after AR end
    co = collections.Counter(); tot = 0; ovl = collections.Counter()
    for r in recs:
        if r['pfr'] is None or r['pe'] is None or r['pe'] <= 0: continue
        tot += 1
        i = bisect.bisect_left(ostart, r['b'])
        seen = set()
        while i < len(other) and other[i][0] < r['pfr'][1]:
            seen.add(other[i][2][:48]); i += 1
        for s in seen: co[s] += 1
        if not seen: ovl['none'] += 1
    print(f"--- bursts running past AR end: {tot/IT:.2f}/it; with no other kernel in overrun {ovl['none']/IT:.2f}/it; co-resident kernels (per it):")
    for s, n in co.most_common(10): print(f"   {n/IT:6.2f}  {s}")
