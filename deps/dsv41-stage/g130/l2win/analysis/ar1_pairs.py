#!/usr/bin/env python3
# Read-only: pair each rdma_ar kernel with l2pf kernels on rank-0 sqlite. Prints classes of AR calls.
import sqlite3, sys, re, collections, statistics as st, bisect, json
IT = 60
def med(x): return st.median(x) if x else float('nan')
for arg in sys.argv[1:]:
    p, l = arg.split('=')
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    S = dict(c.execute("select id,value from StringIds"))
    rows = []
    for a, b, s, d, sid, gx, gy, gz, tx in c.execute(
        "select start,end,shortName,demangledName,streamId,gridX,gridY,gridZ,blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
        rows.append((a, b, S.get(d) or S.get(s) or '', sid, gx, tx))
    ar = [r for r in rows if 'ar_kernel' in r[2]]
    pf = [r for r in rows if 'l2pf' in r[2]]
    names = collections.Counter(r[2][:50] for r in ar)
    print(f"\n=== {l}: rows {len(rows)} ar {len(ar)} ({len(ar)/IT:.1f}/iter) l2pf {len(pf)} ({len(pf)/IT:.1f}/iter)")
    print("ar names", names.most_common(10))
    print("pf names", collections.Counter(r[2][:40] for r in pf).most_common(5))
    print("pf grid", collections.Counter((r[4], r[5]) for r in pf).most_common(10))
    print("streams ar", collections.Counter(r[3] for r in ar).most_common(4), "pf", collections.Counter(r[3] for r in pf).most_common(4))
    # next kernels on main stream
    mainsid = collections.Counter(r[3] for r in ar).most_common(1)[0][0]
    main = [r for r in rows if r[3] == mainsid]
    mainstarts = [r[0] for r in main]
    pfstarts = [r[0] for r in pf]
    recs = []
    for r in ar:
        a, b, n, sid, gx, tx = r
        i = bisect.bisect_right(pfstarts, b) - 1
        ov = None
        # overlapping pf (scan back few)
        j = i
        cand = []
        while j >= 0 and j > i - 6:
            if pf[j][1] > a and pf[j][0] < b:
                cand.append(pf[j])
            j -= 1
        k = bisect.bisect_right(mainstarts, b - 1)
        nxt = main[k] if k < len(main) else None
        rec = dict(dur=(b - a) / 1e3, name=n, gx=gx, tx=tx, a=a, b=b, npf=len(cand))
        if cand:
            q = cand[0]
            rec.update(pfdur=(q[1] - q[0]) / 1e3, pfgrid=q[4], pf_s=(q[0] - a) / 1e3, pf_e=(q[1] - b) / 1e3)
        if nxt is not None:
            rec.update(nx=nxt[2][:40], nxgap=(nxt[0] - b) / 1e3)
        recs.append(rec)
    by = collections.defaultdict(list)
    for r in recs:
        by[(r['name'][:40], r['gx'], r['tx'], r['npf'] > 0)].append(r)
    print(f"{'kernel':42s} grid thr pf  n/it   p50dur  p25   p75  pfdur p50 grid  pfstart  pfend-arend")
    for k in sorted(by, key=lambda x: -len(by[x])):
        v = by[k]
        d = [x['dur'] for x in v]
        qs = st.quantiles(d, n=4) if len(d) > 3 else [0, med(d), 0]
        pd = [x['pfdur'] for x in v if 'pfdur' in x]
        pg = collections.Counter(x.get('pfgrid') for x in v).most_common(1)[0][0]
        ps = [x['pf_s'] for x in v if 'pf_s' in x]
        pe = [x['pf_e'] for x in v if 'pf_e' in x]
        print(f"{k[0]:42s} {k[1]:4d} {k[2]:3d} {int(k[3]):2d} {len(v)/IT:6.2f} {qs[1]:7.1f} {qs[0]:6.1f} {qs[2]:6.1f} {med(pd):7.1f} {str(pg):4s} {med(ps):7.1f} {med(pe):7.1f}")
    # next-kernel by class
    print("--- next kernel after AR (with prefetch) top")
    cnt = collections.defaultdict(list)
    for r in recs:
        if r['npf'] > 0 and 'nx' in r:
            cnt[(r['gx'], r['tx'], r.get('pfgrid'), r['nx'])].append(r['nxgap'])
    for k, v in sorted(cnt.items(), key=lambda x: -len(x[1]))[:16]:
        print(k, f"n/it {len(v)/IT:.2f} gap p50 {med(v):.1f}")
