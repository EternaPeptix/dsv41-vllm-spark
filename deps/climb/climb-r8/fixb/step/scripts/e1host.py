# e1host.py <sqlite>...: E1 (target graph end -> drafter graph start) per step on t2.
# For each named milestone: GPU time from the target-graph end; the host slack (GPU start - host API end):
# slack >> 0 means the host was ahead (GPU-bound), slack ~ launch latency means the host paced the GPU.
import sqlite3, sys, statistics as st
MS = ['_screen_kernel', '_refine_kernel', '_sf_local_argmax_kernel', 'ar_kernel_pl#1', '_post_update_kernel',
      'CatArrayBatchedCopy_vectorized', 'kernel_cutlass_kernel_flashinferquantization', 'device_kernel',
      'ar_kernel_pl#2', 'rms_norm_kernel', '_prepare_dflash_inputs_kernel']
for p in sys.argv[1:]:
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True); S = dict(c.execute("select id,value from StringIds"))
    rtn = {}; rte = {}
    for cid, nid, s0, e0 in c.execute("select correlationId,nameId,start,end from CUPTI_ACTIVITY_KIND_RUNTIME"):
        rtn[cid] = S.get(nid, ''); rte[cid] = e0
    k = [(a, b, S[n], gx, cr, stx) for a, b, n, gx, cr, stx in c.execute(
        "select start,end,shortName,gridX,correlationId,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    segs = []; cur = None
    for x in k:
        src = x[4] if 'GraphLaunch' in rtn.get(x[4], '') else 'E'
        if cur and cur[0] == src: cur[1].append(x)
        else: cur = [src, [x]]; segs.append(cur)
    T = [i for i, s in enumerate(segs) if s[0] != 'E' and len(s[1]) > 1000]
    acc = {m: ([], [], []) for m in MS}; tot = []; gaps = []; win = []
    for i in T:
        if i + 1 >= len(segs) or segs[i+1][0] != 'E': continue
        t0 = segs[i][1][-1][1]; E = segs[i+1][1]
        seen = {}
        for x in E:
            nm = x[2]
            key = None
            for m in MS:
                base = m.split('#')[0]
                if nm.startswith(base):
                    if '#' in m:
                        n = seen.get(base, 0) + 1; seen[base] = n
                        if m.endswith('#%d' % n): key = m
                        elif m.endswith('#1') and n == 2: key = None
                    else:
                        key = m if m not in seen else None; seen[m] = 1
                    if key: break
            if key and key in acc:
                acc[key][0].append((x[0]-t0)/1e3); acc[key][1].append((x[1]-x[0])/1e3)
                acc[key][2].append((x[0]-rte.get(x[4], x[0]))/1e3)
        tot.append((E[-1][1]-t0)/1e3)
        g = sum(max(0, E[j][0]-E[j-1][1]) for j in range(1, len(E))) / 1e3; gaps.append(g)
        sc = [x for x in E if x[2].startswith('_screen_kernel')]; ca = [x for x in E if x[2].startswith('CatArray')]
        if sc and ca: win.append((ca[0][0]-sc[0][1])/1e3)
    md = lambda v: sorted(v)[len(v)//2] if v else float('nan')
    print(p.split('/')[-1], 'E1 steps', len(tot), f'E1 span p50 {md(tot):.1f} us, inter-kernel gaps p50 {md(gaps):.1f} us,'
          f' screen end -> CatArray start p50 {md(win):.1f} p10 {sorted(win)[len(win)//10]:.1f}')
    for m in MS:
        a, d, s = acc[m]
        if a: print(f"  {m:46s} n {len(a):3d} at p50 {md(a):7.1f}  dur p50 {md(d):6.2f}  host slack p50 {md(s):9.1f} p10 {sorted(s)[len(s)//10]:9.1f}")
