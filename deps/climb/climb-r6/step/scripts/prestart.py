# prestart.py <sqlite>...: per step, the eager region between the drafter graph end and the target graph start.
# Reports: drafter end -> emb kernel start, emb AR duration, AR end -> target graph first kernel, and
# the host-side API timestamps (cudaGraphLaunch call time vs GPU start) to tell host- vs GPU-bound.
import sqlite3, sys, statistics as st
for p in sys.argv[1:]:
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True); S = dict(c.execute("select id,value from StringIds"))
    rtn = {}; rts = {}
    for cid, nid, s0, e0 in c.execute("select correlationId,nameId,start,end from CUPTI_ACTIVITY_KIND_RUNTIME"):
        rtn[cid] = S.get(nid, ''); rts[cid] = (s0, e0)
    k = [(a, b, S[n], gx, cr, stx) for a, b, n, gx, cr, stx in c.execute(
        "select start,end,shortName,gridX,correlationId,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    segs = []; cur = None
    for x in k:
        src = x[4] if 'GraphLaunch' in rtn.get(x[4], '') else 'E'
        if cur and cur[0] == src: cur[1].append(x)
        else: cur = [src, [x]]; segs.append(cur)
    T = [i for i, s in enumerate(segs) if s[0] != 'E' and len(s[1]) > 1000]
    rows = []
    for i in T:
        pre = segs[i-1]
        if pre[0] != 'E': continue
        emb = [x for x in pre[1] if x[2].startswith('vocab_parallel_embedding')]
        if not emb: continue
        e = emb[-1]; j = pre[1].index(e)
        ar = pre[1][j+1] if j+1 < len(pre[1]) and pre[1][j+1][2] == 'ar_kernel_pl' else None
        if ar is None: continue
        dg = segs[i-2][1][-1][1] if i >= 2 and segs[i-2][0] != 'E' else None
        g0 = segs[i][1][0]
        # host time of the graph launch API call for this graph
        hl = rts.get(segs[i][0], (0, 0))[0]
        # host time of the AR launch
        har = rts.get(ar[4], (0, 0))[0]
        rows.append(dict(d2emb=(e[0]-dg)/1e3 if dg else float('nan'), emb=(e[1]-e[0])/1e3,
                         emb2ar=(ar[0]-e[1])/1e3, ar=(ar[1]-ar[0])/1e3, ar2g=(g0[0]-ar[1])/1e3,
                         hl_minus_arend=(hl-ar[1])/1e3, gpu_g0_minus_hl=(g0[0]-hl)/1e3,
                         har_minus_embend=(har-e[1])/1e3, n_e=len(pre[1]), arg=ar[3]))
    print(p.split('/')[-1], 'steps', len(rows))
    for key in rows[0]:
        v = sorted(r[key] for r in rows if r[key] == r[key])
        if not v: continue
        print(f"  {key:18s} p10 {v[len(v)//10]:8.2f} p50 {v[len(v)//2]:8.2f} p90 {v[9*len(v)//10]:8.2f} mean {st.mean(v):8.2f}")
