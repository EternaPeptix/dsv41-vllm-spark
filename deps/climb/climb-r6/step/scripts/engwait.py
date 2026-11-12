# engwait.py <sqlite>...: r6-review nonblocking item "PACK waits on layer 14's Engram lookup at layer 1".
# For every target forward (t2), slack = start of layer 1's rows all-gather (the 2nd ar_kernel_pl before the first
# _fused_engram_post_wkv_kernel, on that graph stream) - end of the LAST _engram_lookup_kernel before it (layer 14's
# lookup; both lookups run back to back at forward start). slack > 0 means the packed gather at layer 1 would not
# wait on layer 14's lookup (no new critical-path time).
import sqlite3, sys
q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))] if v else float('nan')
allv = []
for p in sys.argv[1:]:
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True); S = dict(c.execute("select id,value from StringIds"))
    k = [(a, b, S[n], sx) for a, b, n, sx in c.execute(
        "select start,end,shortName,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    looks = [x for x in k if x[2] == '_engram_lookup_kernel']
    posts = [i for i, x in enumerate(k) if x[2].startswith('_fused_engram_post')]
    v = []; nlook = []
    for j, i0 in enumerate(posts):
        prev_look = [x for x in looks if x[1] <= k[i0][0] and x[1] > k[i0][0] - 5e6]
        if not prev_look:
            continue
        last = max(prev_look, key=lambda x: x[1])
        if j > 0 and posts[j - 1] > 0 and k[posts[j - 1]][0] > last[1]:
            continue  # this is the 2nd Engram layer of the forward (its lookups were before the 1st layer)
        s0 = k[i0][3]
        ars = [x for x in k[max(0, i0 - 400):i0] if x[2] == 'ar_kernel_pl' and x[3] == s0][-2:]
        if len(ars) < 2:
            continue
        v.append((ars[0][0] - last[1]) / 1e3)
        nlook.append(len([x for x in prev_look if x[1] > ars[0][0] - 5e6]))
    allv += v
    print(f"{p.split('/')[-1]}: forwards {len(v)}  slack us (layer-1 rows AG start - last lookup end): "
          f"min {min(v):.1f}  p10 {q(v, .1):.1f}  p50 {q(v, .5):.1f}  p90 {q(v, .9):.1f}  "
          f"forwards with slack < 0: {sum(1 for s in v if s < 0)}")
print(f"ALL: {len(allv)} forwards, min slack {min(allv):.1f} us, p10 {q(allv, .1):.1f}, p50 {q(allv, .5):.1f}; "
      f"negative {sum(1 for s in allv if s < 0)}")
