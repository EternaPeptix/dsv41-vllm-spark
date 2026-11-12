# e1tail.py <sqlite> <k>: all-stream kernels from 400 us before the end of the k-th target graph replay to the
# head (max,id) gather end. Prints start (us, rel. target end), dur, stream, grid, name.
import sqlite3,sys
p,kk=sys.argv[1],int(sys.argv[2])
c=sqlite3.connect(f'file:{p}?mode=ro',uri=True); S=dict(c.execute("select id,value from StringIds"))
rt=dict((r[0],S.get(r[1],'')) for r in c.execute("select correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME"))
k=[(a,b,S[n],gx,cr,st,bx) for a,b,n,gx,cr,st,bx in c.execute("select start,end,shortName,gridX,correlationId,streamId,blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
segs=[];cur=None
for x in k:
    src=x[4] if 'GraphLaunch' in rt.get(x[4],'') else 'E'
    if cur and cur[0]==src: cur[1].append(x)
    else:
        cur=[src,[x]]; segs.append(cur)
T=[i for i,s in enumerate(segs) if s[0]!='E' and len(s[1])>1000]
i=T[kk]; t0=segs[i][1][-1][1]
# find gather end after t0
ge=None
for j,x in enumerate(k):
    if x[0]>t0 and x[2]=='ar_kernel_pl' and k[j-1][2]=='_sf_local_argmax_kernel': ge=x[1]; break
for x in k:
    if x[1]<t0-400e3 or x[0]>ge: continue
    print(f"{(x[0]-t0)/1e3:9.2f} {(x[1]-x[0])/1e3:7.2f} s{x[5]:<6d} g{x[3]:<5d} b{x[6]:<4d} {x[2][:70]}")
