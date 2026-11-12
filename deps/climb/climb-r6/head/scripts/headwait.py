# headwait.py <dir> <w>: per step on ranks 0 and 4, the eager head chain timings (us):
#   xs = xnorm start -> screen start, sc = screen dur, rf = refine dur, am = argmax dur, g = (max,id) gather dur,
#   arr = gather start relative to the gather end (= how early this rank arrived), te = target-graph end -> screen start.
# Then cross-rank: matched by step index (both ranks have the same step count), the arrival of the later of r0/r4.
import sqlite3,sys,collections
import numpy as np
def load(p):
    c=sqlite3.connect(f'file:{p}?mode=ro',uri=True); S=dict(c.execute("select id,value from StringIds"))
    return [(a,b,S[n],gx,st) for a,b,n,gx,st in c.execute("select start,end,shortName,gridX,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
d,w=sys.argv[1],sys.argv[2]
R={}
for r in '04':
    k=load(f'{d}/q{r}.{w}.sqlite')
    rows=[]
    for i,x in enumerate(k):
        if x[2]=='ar_kernel_pl' and i>3 and k[i-1][2]=='_sf_local_argmax_kernel' and k[i-2][2]=='_refine_kernel' and k[i-3][2]=='_screen_kernel':
            am,rf,sc=k[i-1],k[i-2],k[i-3]
            rows.append(dict(sc=(sc[1]-sc[0])/1e3, rf=(rf[1]-rf[0])/1e3, am=(am[1]-am[0])/1e3, g=(x[1]-x[0])/1e3,
                             gap1=(rf[0]-sc[1])/1e3, gap2=(am[0]-rf[1])/1e3, gap3=(x[0]-am[1])/1e3,
                             chain=(x[0]-sc[0])/1e3))
    R[r]=rows[2:]
    a=lambda key: np.array([z[key] for z in R[r]])
    print(f"w{w} r{r} n={len(R[r])}: "+"  ".join(f"{kk} {a(kk).mean():.1f}/{np.median(a(kk)):.1f}/{np.percentile(a(kk),90):.1f}" for kk in ('sc','rf','am','g','gap1','gap2','gap3','chain')))
    rf=a('rf'); print(f"   refine hist (us): "+" ".join(f"<{b}:{int(((rf<b)&(rf>=lo)).sum())}" for lo,b in ((0,20),(20,25),(25,30),(30,40),(40,60),(60,100),(100,150),(150,300),(300,1e9))))
n=min(len(R['0']),len(R['4']))
# chain = screen start -> gather start on each rank; gather g; (chain + g) ~ screen start -> common gather end.
c0=np.array([R['0'][i]['chain']+R['0'][i]['g'] for i in range(n)]); c4=np.array([R['4'][i]['chain']+R['4'][i]['g'] for i in range(n)])
print(f"   screen-start->gather-end r0 {c0.mean():.1f}  r4 {c4.mean():.1f}; own chain r0 {np.mean([z['chain'] for z in R['0']]):.1f} r4 {np.mean([z['chain'] for z in R['4']]):.1f}")
