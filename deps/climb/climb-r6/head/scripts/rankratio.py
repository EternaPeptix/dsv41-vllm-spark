# rankratio.py <dir> <w>: per kernel name (+grid), median duration on rank 0 and rank 4 and the r4/r0 ratio,
# for the kernels with the largest total time. Tells SM-clock-sensitive (compute/issue-bound) from DRAM-bound kernels.
import sqlite3,sys,collections
import numpy as np
def load(p):
    c=sqlite3.connect(f'file:{p}?mode=ro',uri=True); S=dict(c.execute("select id,value from StringIds"))
    D=collections.defaultdict(list)
    for a,b,n,gx in c.execute("select start,end,shortName,gridX from CUPTI_ACTIVITY_KIND_KERNEL"):
        D[(S[n][:44],gx)].append((b-a)/1e3)
    return D
d,w=sys.argv[1],sys.argv[2]
A,B=load(f'{d}/q0.{w}.sqlite'),load(f'{d}/q4.{w}.sqlite')
rows=[]
for key in A:
    if key in B and len(A[key])>=50 and len(B[key])>=50:
        rows.append((sum(A[key]),key,np.median(A[key]),np.median(B[key]),len(A[key])))
rows.sort(reverse=True)
for tot,key,a,b,n in rows[:32]:
    print(f"{key[0]:44s} g{key[1]:<5d} n{n:<6d} r0 {a:8.2f} r4 {b:8.2f} r4/r0 {b/a:6.3f}")
