# stepstart.py <sqlite>...: t2 re-base of the C3 step-start collectives (climb r6 step kit).
# Per step (target graph = segment with >1000 kernels; drafter graph = the next graph segment):
#  eager pre-target region: kernels, each duration and the gap before it, total span drafter-end -> target start
#  target graph: for each _fused_engram_post_wkv_kernel, the kernels between the previous mhc op and it
#  drafter graph: the first ar_kernel_pl (embedding AR) duration and its gaps
import sqlite3, sys, collections
md = lambda v: sorted(v)[len(v)//2] if v else float('nan')
p10 = lambda v: sorted(v)[len(v)//10] if v else float('nan')
for p in sys.argv[1:]:
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True); S = dict(c.execute("select id,value from StringIds"))
    rt = dict((r[0], S.get(r[1], '')) for r in c.execute("select correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME"))
    k = [(a, b, S[n], gx, cr, sx) for a, b, n, gx, cr, sx in c.execute(
        "select start,end,shortName,gridX,correlationId,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    segs = []; cur = None
    for x in k:
        src = x[4] if 'GraphLaunch' in rt.get(x[4], '') else 'E'
        if cur and cur[0] == src: cur[1].append(x)
        else: cur = [src, [x]]; segs.append(cur)
    T = [i for i, s in enumerate(segs) if s[0] != 'E' and len(s[1]) > 1000]
    pre = collections.defaultdict(list); span = []; eng = collections.defaultdict(list); dr = collections.defaultdict(list)
    for i in T:
        if i < 2 or segs[i-1][0] != 'E' or segs[i-2][0] == 'E': continue
        D = segs[i-2][1]; E = segs[i-1][1]; G = segs[i][1]
        prev_end = max(x[1] for x in D)
        span.append((G[0][0]-prev_end)/1e3)
        for j, x in enumerate(E):
            pe = prev_end if j == 0 else E[j-1][1]
            nm = x[2][:28] + '#%d' % j
            pre[nm].append(((x[1]-x[0])/1e3, (x[0]-pe)/1e3))
        pre['->graph gap'].append(((G[0][0]-E[-1][1])/1e3, 0))
        main = [x for x in G if not x[2].startswith('l2pf')]
        # engram: per layer, find _fused_engram_post_wkv_kernel; walk back to the 2 preceding ar_kernel_pl
        posts = [j for j, x in enumerate(main) if x[2].startswith('_fused_engram_post')]
        for L, j in enumerate(posts):
            ars = [q for q in range(j-1, max(0, j-40), -1) if main[q][2] == 'ar_kernel_pl'][:2]
            if len(ars) < 2: continue
            wag, lag = main[ars[0]], main[ars[1]]
            eng[f'L{L} wkvAG dur'].append((wag[1]-wag[0])/1e3)
            eng[f'L{L} wkvAG gap in'].append((wag[0]-main[ars[0]-1][1])/1e3)
            eng[f'L{L} wkvAG gap out'].append((main[ars[0]+1][0]-wag[1])/1e3)
            eng[f'L{L} lookAG dur'].append((lag[1]-lag[0])/1e3)
            eng[f'L{L} lookAG gap in'].append((lag[0]-main[ars[1]-1][1])/1e3)
            eng[f'L{L} lookAG gap out'].append((main[ars[1]+1][0]-lag[1])/1e3)
            seg = main[ars[1]+1:ars[0]]
            eng[f'L{L} between(look->wkv) n'].append(len(seg))
            eng[f'L{L} between span'].append((wag[0]-lag[1])/1e3)
            for q in seg:
                eng[f'L{L}   ' + q[2][:40]].append((q[1]-q[0])/1e3)
            eng[f'L{L} lookAG prod=' + main[ars[1]-1][2][:36]].append(0)
        if i+1 < len(segs):
            nx = [s for s in segs[i+1:i+4] if s[0] != 'E']
            if nx:
                Dg = [x for x in nx[0][1] if not x[2].startswith('l2pf')]
                a = [q for q, x in enumerate(Dg) if x[2] == 'ar_kernel_pl']
                if a:
                    q = a[0]; x = Dg[q]
                    dr['first AR dur'].append((x[1]-x[0])/1e3); dr['first AR grid'].append(x[3])
                    dr['gap in'].append((x[0]-Dg[q-1][1])/1e3 if q else 0)
                    dr['gap out'].append((Dg[q+1][0]-x[1])/1e3)
                    dr['prod=' + (Dg[q-1][2][:40] if q else '-')].append(0)
                    dr['cons=' + Dg[q+1][2][:40]].append(0)
                    dr['kernels before'].append(q)
    print(f'== {p.split("/")[-1]}  steps {len(span)}  drafter-end -> target-start span p50 {md(span):.1f} p10 {p10(span):.1f} us')
    for nm, v in pre.items():
        if len(v) >= len(span)//2:
            print(f'   E-pre {nm:34s} n {len(v):3d} dur p50 {md([a for a,_ in v]):7.2f}  gap-before p50 {md([b for _,b in v]):7.2f}')
    for nm, v in eng.items():
        if len(v) >= 5: print(f'   eng {nm:52s} n {len(v):3d} p50 {md(v):7.2f} p10 {p10(v):7.2f}')
    for nm, v in dr.items():
        print(f'   drafter {nm:50s} n {len(v):3d} p50 {md(v):7.2f}')
