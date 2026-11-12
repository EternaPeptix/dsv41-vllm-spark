import sys, collections
sys.path.insert(0, 'anatomy')
from cp4 import load, segments, critical
p = sys.argv[1]
rows, rt = load(p); segs = segments(rows, rt)
groups = collections.defaultdict(list)
for x in segs:
    if x['kind'] == 'T': groups[x['key']].append(x)
shown = 0
for key, reps in groups.items():
    if len(reps) < 3: continue
    reps2, S, E, meta, main, paths = critical(reps)
    for r, path in enumerate(paths):
        js = [q[0] for q in path]
        for k, (j, g, ex, head) in enumerate(path):
            if 'act_and_mul' in meta[j][4] and g > 20e3 and shown < 3:
                shown += 1
                t0 = S[r].min()
                print(f'--- replay {r} key {key} act gap {g/1e3:.1f}')
                for jj, gg, ee, hh in path[max(0, k - 8):k + 10]:
                    m = meta[jj]
                    print(f'  {(S[r,jj]-t0)/1e3:9.2f} end {(E[r,jj]-t0)/1e3:9.2f} gap {gg/1e3:6.2f} s{m[2]:<6} {m[7]:<18} {m[4][:40]:<40} {m[5]}')
                # what else ran around: kernels on any stream overlapping [act_s-80, act_e+60]
                a0, a1 = S[r, j] - 120e3, E[r, j] + 80e3
                print('  all nodes in window:')
                for i in sorted(range(len(meta)), key=lambda i: S[r, i]):
                    if S[r, i] < a1 and E[r, i] > a0 and meta[i][8] != 'l2pf':
                        m = meta[i]
                        print(f'    {(S[r,i]-t0)/1e3:9.2f} end {(E[r,i]-t0)/1e3:9.2f} s{m[2]:<6} {m[7]:<18} {m[4][:40]:<40} {m[5]}')
