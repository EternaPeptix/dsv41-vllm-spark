# FC1 fluid model on the real static M32 tile walk (scheduler per m32_sched_sim.py / bs08_sim.py). Read-only, no GPU. python3 -I.
import json,sys,gzip,re,collections,statistics as st,math
NSM=48; NN=9; KB=40; B=4352.0; C=230.0; F0=396.0; T0=10.0
cache={}
def lists(nmb,G):
    k=(nmb,G)
    if k in cache: return cache[k]
    out=[0]*NSM; total=nmb*NN; per=NN*8
    for c in range(NSM):
        it=0
        while True:
            b=it*NSM+c; it+=1
            if b>=total: break
            g=b//per; first=g*8; ing=b%per; nbig=min(8,nmb-first)
            m=first+ing%nbig
            if m>=G:
                if nbig==8: break
                continue
            out[c]+=1
    cache[k]=out; return out
def rr(G):
    out=[0]*NSM
    for l in range(G*NN): out[l%NSM]+=1
    return out
def sim(per,floor=F0,c=C,t0=T0,b=B):
    rem=[p*KB for p in per]; t=d=f=0.0; 
    while any(rem):
        act=[i for i,r in enumerate(rem) if r>0]; n=len(act)
        sd=b*n/c   # ns/stage dram-limited
        s=max(floor,sd)
        k=min(rem[i] for i in act)
        t+=s*k
        if sd>=floor: d+=s*k
        else: f+=s*k
        for i in act: rem[i]-=k
    return (t0+t/1000.0,d/1000.0,f/1000.0)
def T(G,nmb,floor=F0,bal=False,t0=T0):
    if G==0: return t0
    per=rr(G) if bal else lists(nmb,G)
    return sim(per,floor,t0=t0)[0]
if __name__=='__main__':
    print('Table: G  bytesMB floor238 | T live(M2 nmb12) (M4 nmb24) (M6 nmb36) | balanced | F=264 live | F=264 bal | F=198 live  busiest(nmb24)')
    for G in list(range(2,37)):
        mb=G*1.56672
        print(f"{G:3d} {mb:6.2f} {mb/238*1000:6.1f} | {T(G,12) if G<=12 else 0:6.1f} {T(G,24) if G<=24 else 0:6.1f} {T(G,36):6.1f} | {T(G,36,bal=True):6.1f} | {T(G,36,264):6.1f} {T(G,36,264,True):6.1f} {T(G,36,198):6.1f}  {max(lists(36,G))}")
    # calibration against gsweep
    rows=[json.loads(l) for l in open(sys.argv[1])]
    err=[]
    for r in rows:
        if r.get('event')=='g':
            nmb=r['m']*6
            err.append(r['fc1_us']-T(r['G'],nmb))
    print('gsweep calibration n',len(err),'rmse %.2f mean %.2f'%(math.sqrt(st.mean(e*e for e in err)),st.mean(err)))
