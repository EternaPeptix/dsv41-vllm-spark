# r5-prio: re-base the three scheduling levers on the t2 traces of launcher 166d7aac.
# Run on 512S1 from ~/climb-traces/r5-prio:  python3 prio_t2.py <sqlite>=<label> ...
# Uses the round-4/5 anatomy loader + replay-consistent critical path (cp4.py, cls4.py), unchanged.
import sys, collections, statistics as st
import numpy as np
sys.path.insert(0, 'anatomy')
import cp4
from cp4 import load, segments, critical

def ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

def is_cmp(m):  # KV-source / compressor BF16 cuBLAS GEMM on aux stream 0 (torch.mm fused_wkv_wgate)
    return 'Kernel2' in m[4] and m[5] in ('8x8x16', '8x4x16', '8x3x16', '8x2x16')

def main():
    for arg in sys.argv[1:]:
        p, label = arg.split('=')
        rows, rt = load(p)
        segs = segments(rows, rt)
        Ti = [i for i, x in enumerate(segs) if x['kind'] == 'T']
        nsteps = sum(1 for a, b in zip(Ti, Ti[1:]) if segs[b]['s'] - segs[a]['s'] <= 60e6)
        groups = collections.defaultdict(list)
        for x in segs:
            if x['kind'] == 'T':
                groups[x['key']].append(x)
        A = collections.defaultdict(float)   # CMPPRIO accumulators (ns, summed over replays)
        B = collections.defaultdict(float)   # shared-expert accumulators
        C = collections.defaultdict(float)   # engram accumulators
        lay = []                              # per-layer shared records
        nrep = 0
        cmpgrid = collections.Counter()
        for key, reps in groups.items():
            if len(reps) < 3:
                continue
            reps2, S, E, meta, main, paths = critical(reps)
            N = len(meta)
            order = sorted(range(N), key=lambda j: S[0, j])
            bystream = collections.defaultdict(list)
            for j in order:
                bystream[meta[j][2]].append(j)
            prev_same = {}
            for s, v in bystream.items():
                for a, b in zip(v, v[1:]):
                    prev_same[b] = a
            for r, path in enumerate(paths):
                nrep += 1
                onp = {}
                prev = None
                for j, g, ex, head in path:
                    onp[j] = (prev, g, ex)
                    prev = j
                # ---- (A) qkv_a skinny: CP gap into it and what ran in the gap
                for j in range(N):
                    m = meta[j]
                    if 'skinny' not in m[4]:
                        continue
                    A['n'] += 1
                    pj = prev_same.get(j)
                    if pj is None:
                        continue
                    w = S[r, j] - E[r, pj]
                    conc = [i for i in range(N) if i != j and meta[i][2] != m[2]
                            and ov(S[r, i], E[r, i], E[r, pj], S[r, j]) > 0.5 * max(w, 1)]
                    has_cmp = any(is_cmp(meta[i]) for i in conc)
                    for i in conc:
                        if is_cmp(meta[i]):
                            cmpgrid[meta[i][5]] += 1
                    tag = 'cmp' if has_cmp else 'other'
                    if has_cmp:
                        c = next(i for i in conc if is_cmp(meta[i]))
                        cs = [i for i in bystream[meta[c][2]] if S[0, i] >= S[0, c]][:8]
                        A['cmp_dur'] += E[r, c] - S[r, c]
                        A['skinny_dur_cmp'] += E[r, j] - S[r, j]
                        A['cmpchain_cp'] += sum(onp[i][2] + onp[i][1] for i in cs if i in onp)
                        fs = [i for i in range(N) if 'fused_save_compress' in meta[i][4] and S[0, i] > S[0, c] and S[0, i] < S[0, c] + 200e3]
                        if fs:
                            i = min(fs, key=lambda q: S[0, q])
                            # model (graph-internal stream ids are executor-assigned, so locate nodes by name/time):
                            #   Q = quant end (skinny's predecessor), w = skinny wait, M = first _q_kv_norm end after skinny,
                            #   Cx = first splitKreduce end after the compressor GEMM, F = FSCN start.
                            DELTA = 2.5e3
                            sk_d = E[r, j] - S[r, j]
                            g = max(0.0, w - DELTA)
                            qn = [q for q in range(N) if '_q_kv_norm' in meta[q][4] and S[r, q] > S[r, j]]
                            sk = [q for q in range(N) if 'splitKreduce' in meta[q][4] and S[r, q] >= E[r, c] - 1e3]
                            if qn and sk:
                                Mx = E[r, min(qn, key=lambda q: S[r, q])]
                                Cx = E[r, min(sk, key=lambda q: S[r, q])]
                                cur = max(Mx, Cx)
                                late = S[r, i] - cur
                                Fp = max(Mx - g, Cx + sk_d) + late
                                Fs = max(Mx - g, (E[r, j] - g) + (Cx - S[r, c])) + late
                                gp = min(g, max(0.0, S[r, i] - Fp))
                                gs = min(g, max(0.0, S[r, i] - Fs))
                                oncp = (i in onp or j in onp)
                                A['g_prio'] += gp
                                A['g_serial'] += gs
                                A['g_prio_cp'] += gp if oncp else 0.0
                                A['g_serial_cp'] += gs if oncp else 0.0
                                A['g_main'] += g
                                A['cmp_minus_main'] += Cx - Mx
                                A['fscn_late'] += late
                            A['fscn_cp'] += (onp[i][2] + onp[i][1]) if i in onp else 0.0
                            A['fscn_n'] += 1
                            A['fscn_on'] += 1 if i in onp else 0
                    elif w > 3e3:
                        conc2 = collections.Counter(meta[i][7] for i in conc)
                        for k2 in conc2: A['oth:' + k2] += 1
                    A['wait_' + tag] += max(0.0, w)
                    A['nwait_' + tag] += 1 if w > 3e3 else 0
                    if j in onp:
                        A['cpgap_' + tag] += onp[j][1]
                # ---- (B) shared expert chain vs FC1
                seq = order
                last_n640 = None
                for j in seq:
                    m = meta[j]
                    if m[7] == 'gemm_mxfp8_N640':
                        last_n640 = j
                    if m[7] == 'fc1_moe' and last_n640 is not None:
                        n640 = last_n640
                        last_n640 = None
                        sh = meta[n640][2]
                        chain = [i for i in bystream[sh] if S[0, i] >= S[0, n640]]
                        act = next((i for i in chain if 'act_and_mul' in meta[i][4]), None)
                        down = next((i for i in chain if meta[i][7] == 'gemm_mxfp8_N5120'), None)
                        if act is None or down is None:
                            continue
                        pf = prev_same.get(j)
                        fc1_ready = E[r, pf] if pf is not None else S[r, j]
                        # the gather-add (join) after this FC1
                        ga = next((i for i in seq if S[0, i] > S[0, j] and meta[i][7] == 'moe_gather_add'), None)
                        fc2 = next((i for i in seq if S[0, i] > S[0, j] and meta[i][7] == 'fc2_moe'), None)
                        if ga is None or fc2 is None:
                            continue
                        rec = dict(
                            fc1_ready=fc1_ready, fc1_s=S[r, j], fc1_e=E[r, j], fc1_d=E[r, j] - S[r, j],
                            n640_s=S[r, n640], n640_e=E[r, n640], act_s=S[r, act], act_e=E[r, act],
                            down_s=S[r, down], down_e=E[r, down], fc2_e=E[r, fc2], ga_s=S[r, ga],
                            act_on=(act in onp), act_gap=(onp[act][1] if act in onp else 0.0),
                            down_on=(down in onp), ga_on=(ga in onp), rep=nrep, li=sum(1 for q in lay if q['rep']==nrep),
                            fc1_grid=meta[j][5])
                        lay.append(rec)
                # ---- (C) engram: wkv GEMM (N3200) + its quant, rows AG before, kv AG after
                for j in range(N):
                    m = meta[j]
                    if m[7] != 'gemm_mxfp8_N3200':
                        continue
                    C['n'] += 1
                    C['gemm_dur'] += E[r, j] - S[r, j]
                    if j in onp:
                        C['gemm_cp'] += onp[j][2] + onp[j][1]
                    pq = prev_same.get(j)
                    if pq is not None and meta[pq][7] == 'mxfp8_quant':
                        C['quant_dur'] += E[r, pq] - S[r, pq]
                        if pq in onp:
                            C['quant_cp'] += onp[pq][2] + onp[pq][1]
                        pa = prev_same.get(pq)
                        if pa is not None and meta[pa][7] == 'ar':
                            C['rowsag_dur'] += E[r, pa] - S[r, pa]
                    nx = [i for i in bystream[m[2]] if S[0, i] > S[0, j]]
                    if nx and meta[nx[0]][7] == 'ar':
                        C['kvag_dur'] += E[r, nx[0]] - S[r, nx[0]]
        f = lambda v: v / nsteps / 1e3
        print(f'== {label}: steps {nsteps}  T replays {nrep}')
        print(f'  (A) qkv_a skinny: {A["n"]/nsteps:.1f}/step; same-stream wait with a compressor GEMM running: '
              f'{f(A["wait_cmp"]):.1f} us/step ({A["nwait_cmp"]/nsteps:.2f} waits>3us/step), on-CP gap {f(A["cpgap_cmp"]):.1f} us/step; '
              f'other waits {f(A["wait_other"]):.1f} us/step (on-CP gap {f(A["cpgap_other"]):.1f})  cmp grids {dict(cmpgrid)}')
        print(f'      cmp GEMM mean {A["cmp_dur"]/max(1,A["nwait_cmp"])/1e3:.1f} us, skinny dur in cmp layers {A["skinny_dur_cmp"]/max(1,A["nwait_cmp"])/1e3:.1f} us; '
              f'compressor chain (cmp GEMM + next 7 on its stream) on CP {f(A["cmpchain_cp"]):.1f} us/step; '
              f'fused_save_compress_norm on CP {A["fscn_on"]}/{A["fscn_n"]} ({f(A["fscn_cp"]):.1f} us/step)')
        print(f'      MODEL per step: FSCN-ready earlier by prio {f(A["g_prio"]):.1f} (counted only where skinny or FSCN on CP: {f(A["g_prio_cp"]):.1f}) '
              f'| serial {f(A["g_serial"]):.1f} ({f(A["g_serial_cp"]):.1f}) us/step; FSCN starts {A["fscn_late"]/max(1,A["fscn_n"])/1e3:.1f} us after both inputs ready (mean); '
              f'skinny wait beyond {2.5} us {f(A["g_main"]):.1f} us/step; compressor-ready minus main-ready (Cx-M) mean {A["cmp_minus_main"]/max(1,A["fscn_n"])/1e3:.1f} us')
        print('      other waits>3us concurrent with: ' + str(sorted(((k[4:], round(v/nsteps,2)) for k, v in A.items() if k.startswith('oth:')), key=lambda z: -z[1])[:6]))
        if lay:
            L = lay
            xs = np.array([max(0.0, x['n640_e'] - x['fc1_s']) / 1e3 for x in L]); ys = np.array([x['fc1_d'] / 1e3 for x in L])
            for lo, hi in ((-1, 0.01), (0.01, 5), (5, 15), (15, 30), (30, 1e9)):
                sel = (xs > lo) & (xs <= hi)
                if sel.sum():
                    print(f'      FC1 dur vs N640 overlap at FC1 launch ({lo:g},{hi:g}] us: n/step {sel.sum()/nsteps:.2f}  FC1 p50 {np.median(ys[sel]):.1f} mean {ys[sel].mean():.1f}')
            starved = [x for x in L if x['act_s'] - x['n640_e'] > 10e3]
            ok = [x for x in L if x['act_s'] - x['n640_e'] <= 10e3]
            per = len(L) / nsteps
            print(f'  (B) shared chain: {per:.1f} MoE layers/step; starved (act starts >10us after N640) {len(starved)/nsteps:.2f}/step')
            def md(v):
                return st.median(v) / 1e3 if v else float('nan')
            act_d = md([x['act_e'] - x['act_s'] for x in ok]); down_d = md([x['down_e'] - x['down_s'] for x in ok])
            q_gap = md([x['down_s'] - x['act_e'] for x in ok])
            print(f'      unstarved medians: act {act_d:.2f} us, act_end->down_start {q_gap:.2f} us, down {down_d:.2f} us; '
                  f'N640 end->act start {md([x["act_s"]-x["n640_e"] for x in ok]):.2f} us')
            print(f'      starved: act wait mean {np.mean([x["act_s"]-x["n640_e"] for x in starved])/1e3 if starved else 0:.1f} us, '
                  f'act on CP in {sum(x["act_on"] for x in starved)}/{len(starved)}; CP gap into act {f(sum(x["act_gap"] for x in L)):.1f} us/step')
            # join wait at the gather-add: how late the shared chain is vs FC2 end
            jw = [max(0.0, x['ga_s'] - x['fc2_e']) for x in L]
            print(f'      gather-add start - FC2 end (join wait): {f(sum(jw)):.1f} us/step; >5us in {sum(1 for v in jw if v>5e3)/nsteps:.2f} layers/step')
            # ORDER model: FC1 waits for the shared chain's done event
            ev = 1.5e3
            cost = gain = 0.0
            slack = []
            for x in L:
                sh_done = x['n640_e'] + (act_d + q_gap + down_d) * 1e3 + 1.0e3  # chain without starvation
                if x['act_s'] - x['n640_e'] <= 10e3:
                    sh_done = x['down_e']  # observed (not starved)
                new_fc1 = max(x['fc1_s'], sh_done + ev)
                d_cost = new_fc1 - x['fc1_s']
                d_gain = max(0.0, x['ga_s'] - x['fc2_e'] - 1.0e3)
                cost += d_cost; gain += d_gain
                slack.append((x['fc1_ready'] - sh_done) / 1e3)
            print(f'      ORDER model (FC1 waits shared done): FC1 delay {f(cost):.1f} us/step, join wait removed {f(gain):.1f} us/step, '
                  f'net {f(gain-cost):+.1f} us/step;  slack fc1_ready - shared_done: p10 {np.percentile(slack,10):.1f} p50 {np.percentile(slack,50):.1f} p90 {np.percentile(slack,90):.1f} us')
            # ORDER-ACT model: FC1 waits for act_and_mul (+quant) only; down GEMM may still race FC1
            cost2 = 0.0
            for x in L:
                a_done = x['n640_e'] + (act_d + q_gap) * 1e3 + 0.5e3
                if x['act_s'] - x['n640_e'] <= 10e3:
                    a_done = x['down_s']
                cost2 += max(0.0, max(x['fc1_s'], a_done + ev) - x['fc1_s'])
            onc=[x for x in L if x['act_on']]
            li=collections.Counter()
            for x in onc:
                pass
            if onc:
                print(f'      act-on-CP cases {len(onc)/nsteps:.2f}/step: act_gap mean {np.mean([x["act_gap"] for x in onc])/1e3:.1f}; '
                      f'act_s-fc1_s mean {np.mean([x["act_s"]-x["fc1_s"] for x in onc])/1e3:.1f}; act_s-fc1_e mean {np.mean([x["act_s"]-x["fc1_e"] for x in onc])/1e3:.1f}; '
                      f'n640_e-fc1_s mean {np.mean([x["n640_e"]-x["fc1_s"] for x in onc])/1e3:.1f}; fc1_d {np.mean([x["fc1_d"] for x in onc])/1e3:.1f}; '
                      f'down_e-fc2_e {np.mean([x["down_e"]-x["fc2_e"] for x in onc])/1e3:.1f}; layer idx {collections.Counter(x["li"] for x in onc).most_common(4)}')
            sv=[x for x in L if x['act_s'] - x['n640_e'] > 10e3]
            print(f'      all starved: act_s-fc1_e mean {np.mean([x["act_s"]-x["fc1_e"] for x in sv])/1e3:.1f} (p10 {np.percentile([x["act_s"]-x["fc1_e"] for x in sv],10)/1e3:.1f} p90 {np.percentile([x["act_s"]-x["fc1_e"] for x in sv],90)/1e3:.1f}); '
                  f'n640_e - fc1_s mean {np.mean([x["n640_e"]-x["fc1_s"] for x in sv])/1e3:.1f}; down_e - fc2_e mean {np.mean([x["down_e"]-x["fc2_e"] for x in sv])/1e3:.1f} max {max(x["down_e"]-x["fc2_e"] for x in sv)/1e3:.1f}')
            print(f'      ORDER-ACT model (FC1 waits act+quant): FC1 delay {f(cost2):.1f} us/step (down GEMM still races FC1)')
        if lay:
            print(f'      FC1 grids {collections.Counter(x["fc1_grid"] for x in L).most_common(3)}')
        if C['n']:
            k = nsteps
            print(f'  (C) engram wkv: {C["n"]/k/ (nrep/nsteps):.1f}/replay; GEMM {C["gemm_dur"]/C["n"]/1e3:.1f} us each, on CP {f(C["gemm_cp"]):.1f} us/step; '
                  f'quant {f(C["quant_dur"]):.1f} (CP {f(C["quant_cp"]):.1f}); rows-AG {f(C["rowsag_dur"]):.1f}; kv-AG {f(C["kvag_dur"]):.1f} us/step')

if __name__ == '__main__':
    main()
