# prestage.py <dir> <w>: per step, the latency-bound window before the CERT screen on ranks 0/4:
#   fc2e  = end of the last DRAM-bound FFN GEMM (fc2 / sm120 fp8 gemm) before the screen -> screen start
#   ar    = duration of the last all-reduce before the screen (the final layer's MoE AR)
#   gap   = target-graph end -> first E1 kernel, and E1 start -> screen start
#   pf    = l2pf kernels overlapping [fc2e, screen start] (count, us)
import sqlite3,sys
import numpy as np
def load(p):
    c=sqlite3.connect(f'file:{p}?mode=ro',uri=True); S=dict(c.execute("select id,value from StringIds"))
    return [(a,b,S[n],st) for a,b,n,st in c.execute("select start,end,shortName,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
d,w=sys.argv[1],sys.argv[2]
for r in '04':
    k=load(f'{d}/q{r}.{w}.sqlite')
    rows=[]
    for i,x in enumerate(k):
        if x[2]!='_screen_kernel': continue
        s0=x[0]
        j=i-1
        while j>0 and not (k[j][2].startswith('athena_tp8_fc2') or k[j][2].startswith('sm120_fp8_fp4_gemm')): j-=1
        fc2e=k[j][1]
        ars=[y for y in k[j:i] if y[2]=='ar_kernel_pl']
        ar=(ars[-1][1]-ars[-1][0])/1e3 if ars else 0
        pf=[y for y in k[j-50:i] if 'l2pf' in y[2] and y[1]>fc2e and y[0]<s0]
        xn=[y for y in k[j:i] if y[2]=='_xnorm_kernel']
        rows.append(dict(win=(s0-fc2e)/1e3, ar=ar, npf=len(pf), pfus=sum((min(y[1],s0)-max(y[0],fc2e)) for y in pf)/1e3))
    rows=rows[2:]
    a=lambda kk: np.array([z[kk] for z in rows])
    print(f"w{w} r{r} n={len(rows)}: fc2-end->screen {a('win').mean():.1f}/{np.median(a('win')):.1f} us; last AR {a('ar').mean():.1f} us; l2pf kernels in window {a('npf').mean():.2f} ({a('pfus').mean():.1f} us)")
