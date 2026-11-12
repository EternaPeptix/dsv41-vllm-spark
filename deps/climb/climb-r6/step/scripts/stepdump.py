# stepdump.py <sqlite> <k> [n_graph_kernels]: eager segment before the k-th target graph,
# the first n kernels of that graph, and the E1 segment after it (read-only).
import sqlite3, sys
p, kk = sys.argv[1], int(sys.argv[2]); ng = int(sys.argv[3]) if len(sys.argv) > 3 else 90
c = sqlite3.connect(f'file:{p}?mode=ro', uri=True); S = dict(c.execute("select id,value from StringIds"))
rt = dict((r[0], S.get(r[1], '')) for r in c.execute("select correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME"))
k = [(a, b, S[n], gx, cr, st) for a, b, n, gx, cr, st in c.execute(
    "select start,end,shortName,gridX,correlationId,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
segs = []; cur = None
for x in k:
    src = x[4] if 'GraphLaunch' in rt.get(x[4], '') else 'E'
    if cur and cur[0] == src: cur[1].append(x)
    else: cur = [src, [x]]; segs.append(cur)
T = [i for i, s in enumerate(segs) if s[0] != 'E' and len(s[1]) > 1000]
i = T[kk]; t0 = segs[i][1][0][0]
def pr(tag, xs):
    for x in xs:
        print(f"{tag:6s} {(x[0]-t0)/1e3:9.2f} {(x[1]-x[0])/1e3:7.2f} s{x[5]:<6d} g{x[3]:<5d} {x[2][:64]}")
pr('Epre', segs[i-1][1][-25:] if segs[i-1][0] == 'E' else [])
print('---- graph', len(segs[i][1]))
pr('G', segs[i][1][:ng])
