# engpf.py <sqlite>...: l2pf (paced prefetch) kernels overlapping the Engram lookAG / wkvAG windows and the drafter
# embedding AR, t2. If a window carries prefetch, removing that collective can slow the GEMM it was feeding.
import sqlite3, sys, collections
md = lambda v: sorted(v)[len(v)//2] if v else float('nan')
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
    acc = collections.defaultdict(list)
    for i, s in enumerate(segs):
        if s[0] == 'E' or len(s[1]) < 1000: continue
        G = s[1]; main = [x for x in G if not x[2].startswith('l2pf')]; pf = [x for x in G if x[2].startswith('l2pf')]
        posts = [j for j, x in enumerate(main) if x[2].startswith('_fused_engram_post')]
        for L, j in enumerate(posts):
            ars = [q for q in range(j-1, max(0, j-40), -1) if main[q][2] == 'ar_kernel_pl'][:2]
            if len(ars) < 2: continue
            for nm, a in (('wkvAG', main[ars[0]]), ('lookAG', main[ars[1]])):
                ov = sum(max(0, min(a[1], q[1]) - max(a[0], q[0])) for q in pf) / 1e3
                acc[f'L{L} {nm} l2pf overlap us'].append(ov)
            gq = main[ars[0]-1]
            acc[f'L{L} wkv GEMM dur'].append((gq[1]-gq[0])/1e3)
            ov = sum(max(0, min(gq[1], q[1]) - max(gq[0], q[0])) for q in pf) / 1e3
            acc[f'L{L} wkv GEMM l2pf overlap us'].append(ov)
        if i + 1 < len(segs):
            nx = [t for t in segs[i+1:i+4] if t[0] != 'E']
            if nx:
                D = nx[0][1]; dm = [x for x in D if not x[2].startswith('l2pf')]; dpf = [x for x in D if x[2].startswith('l2pf')]
                a = [x for x in dm if x[2] == 'ar_kernel_pl']
                if a:
                    a = a[0]
                    acc['drafter emb AR l2pf overlap us'].append(sum(max(0, min(a[1], q[1]) - max(a[0], q[0])) for q in dpf) / 1e3)
    print(p.split('/')[-1], '  '.join(f'{k}: p50 {md(v):.1f}' for k, v in acc.items()))
