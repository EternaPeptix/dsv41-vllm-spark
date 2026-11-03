#!/usr/bin/env python3
"""pubmodel.py: exhaustive interleaving check of the rdma_ar publish protocol (plain Python, seconds of CPU, no build).

Each rank = a GPU program (one op after another on a single stream) + a proxy (pass: s1 then s2) with one atomic step per
shared-memory access. A post delivers payload + flag to one peer at once (RC in-order per QP is an assumption, not checked).
Explores EVERY interleaving and reports:
  TORN  the proxy posted a message with a descriptor from another op (silent wrong data)
  LOST  a publish overwrote one its proxy had not read
  DEAD  no step can run and not every rank finished (a timeout trap in the real kernel)
Guards, phase 1: legacy (posted >= send), v1 (posted >= seq-1), none. Phase 2: legacy, v2 (none), seqm1 (posted2 >= seq-1).
Limits: sequentially consistent memory, so it says nothing about GPU weak ordering or the NIC.
"""
import sys
from collections import deque

ONE, TWO = 1, 2


def explore(world, ops, g1, g2, peers_wait_p2=True, limit=4_000_000):
    # GPU pc: (k, stage). stages: 0 guard1, 1 desc1, 2 seq1, 3 wait1, 4 guard2, 5 desc2, 6 seq2, 7 wait2. k == len(ops): done
    # proxy pc: 0 read s1, 1 read desc1, 2.. post i (1-based peers) -> posted store; 20 read s2, 21 read desc2, 22.. post, store
    # rank = (k, st, send, desc1, posted, send2, desc2, posted2, last1, last2, ppc, cur, curdesc, npost)
    init = tuple((0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0) for _ in range(world))
    f1 = tuple(tuple(0 for _ in range(world)) for _ in range(world))
    f2 = f1
    start = (init, f1, f2)
    seen = {start}
    dq = deque([start])
    bad = {}
    n = 0

    def setf(fl, dst, src, v):
        row = list(fl[dst])
        row[src] = max(row[src], v)
        out = list(fl)
        out[dst] = tuple(row)
        return tuple(out)

    while dq:
        s = dq.popleft()
        ranks, fl1, fl2 = s
        n += 1
        if n > limit:
            return n, bad, True
        succ = []
        idle = []  # a proxy pass that finds nothing: explored (a stale read then a later pass is the real inversion), not progress
        for r in range(world):
            R = ranks[r]
            k, st, send, d1, posted, send2, d2, posted2, last1, last2, ppc, cur, curd, np_ = R

            def upd(**kw):
                names = ("k", "st", "send", "d1", "posted", "send2", "d2", "posted2", "last1", "last2", "ppc", "cur", "curd", "np_")
                vals = dict(zip(names, R))
                vals.update(kw)
                rr = list(ranks)
                rr[r] = tuple(vals[x] for x in names)
                return tuple(rr)

            # ---- GPU
            if k < len(ops):
                seq, mode = k + 1, ops[k]
                if st == 0:
                    ok = {"legacy": posted >= send, "v1": posted >= seq - 1, "none": True}[g1]
                    if ok:
                        succ.append((upd(st=1), fl1, fl2))
                elif st == 1:
                    succ.append((upd(st=2, d1=seq), fl1, fl2))
                elif st == 2:
                    if send != last1 and not (ppc in (1, 2, 3) and cur == send):
                        bad.setdefault("LOST", s)
                    succ.append((upd(st=3, send=seq), fl1, fl2))
                elif st == 3:
                    if all(q == r or fl1[r][q] >= seq for q in range(world)):
                        if mode == TWO:
                            succ.append((upd(st=4), fl1, fl2))
                        else:
                            succ.append((upd(k=k + 1, st=0), fl1, fl2))
                elif st == 4:
                    ok = {"legacy": posted2 >= send2, "v2": True, "seqm1": posted2 >= seq - 1}[g2]
                    if ok:
                        succ.append((upd(st=5), fl1, fl2))
                elif st == 5:
                    succ.append((upd(st=6, d2=seq), fl1, fl2))
                elif st == 6:
                    if send2 != last2 and not (ppc in (21, 22, 23) and cur == send2):
                        bad.setdefault("LOST", s)
                    succ.append((upd(st=7, send2=seq), fl1, fl2))
                elif st == 7:
                    if not peers_wait_p2 or all(q == r or fl2[r][q] >= seq for q in range(world)):
                        succ.append((upd(k=k + 1, st=0), fl1, fl2))
            # ---- proxy: pass = read s1, handle it if new, read s2, handle it if new
            if ppc == 0:
                if send != last1:
                    succ.append((upd(ppc=1, cur=send), fl1, fl2))
                else:
                    idle.append((upd(ppc=20), fl1, fl2))
            elif ppc == 1:
                if d1 != cur:
                    bad.setdefault("TORN", s)
                succ.append((upd(ppc=2, curd=d1, np_=0), fl1, fl2))
            elif ppc == 2:
                if np_ < world - 1:
                    q = (r + 1 + np_) % world
                    succ.append((upd(np_=np_ + 1), setf(fl1, q, r, cur), fl2))
                else:
                    succ.append((upd(ppc=3, posted=cur), fl1, fl2))
            elif ppc == 3:
                succ.append((upd(ppc=20, last1=cur, cur=0), fl1, fl2))
            elif ppc == 20:
                if send2 != last2:
                    succ.append((upd(ppc=21, cur=send2), fl1, fl2))
                else:
                    idle.append((upd(ppc=0), fl1, fl2))
            elif ppc == 21:
                if d2 != cur:
                    bad.setdefault("TORN", s)
                succ.append((upd(ppc=22, curd=d2, np_=0), fl1, fl2))
            elif ppc == 22:
                if np_ < world - 1:
                    q = (r + 1 + np_) % world
                    succ.append((upd(np_=np_ + 1), fl1, setf(fl2, q, r, cur)), )
                else:
                    succ.append((upd(ppc=23, posted2=cur), fl1, fl2))
            elif ppc == 23:
                succ.append((upd(ppc=0, last2=cur, cur=0), fl1, fl2))
        # DEAD: nothing but idle passes can run, every proxy has nothing new, and not all ranks finished
        quiet = all(x[2] == x[8] and x[5] == x[9] and x[10] in (0, 20) for x in ranks)
        if not succ and quiet and not all(x[0] == len(ops) for x in ranks):
            bad.setdefault("DEAD", s)
        for t in succ + idle:
            if t not in seen:
                seen.add(t)
                dq.append(t)
    return n, bad, False


def main():
    fails = 0
    cases = [
        # name, g1, g2, peers wait for phase 2, safe?
        ("production (legacy, legacy)", "legacy", "legacy", True, True),
        ("V1 only", "v1", "legacy", True, True),
        ("V2 only", "legacy", "v2", True, True),
        ("V1 + V2", "v1", "v2", True, True),
        ("mutant: no phase-1 guard", "none", "legacy", True, False),
        ("mutant: V1 compare on phase 2", "v1", "seqm1", True, False),
        ("mutant: V2, but peers skip the phase-2 wait", "v1", "v2", False, False),
    ]
    scopes = [(2, (TWO, ONE, TWO, TWO, ONE)), (3, (TWO, TWO)), (3, (TWO, ONE, TWO))]
    for name, g1, g2, pw, safe in cases:
        seen_bad = False
        for world, ops in scopes:
            n, bad, cut = explore(world, ops, g1, g2, pw)
            verdict = "STATE LIMIT HIT" if cut else ("clean" if not bad else "VIOLATION " + ",".join(sorted(bad)))
            seen_bad |= bool(bad)
            if safe and (bad or cut):
                fails += 1
            print(f"{name:46s} world {world} ops {''.join(str(o) for o in ops):6s} states {n:8d}  {verdict}", flush=True)
        if not safe and not seen_bad:
            print(f"  UNEXPECTED: mutant not caught in any scope")
            fails += 1
        print(f"  -> {'clean in every scope, as required' if safe and not seen_bad else 'caught, as required' if not safe and seen_bad else 'NOT AS EXPECTED'}", flush=True)
    print("MODEL", "PASS" if not fails else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
