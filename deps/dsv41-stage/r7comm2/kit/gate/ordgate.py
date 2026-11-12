#!/usr/bin/env python3
"""ordgate.py: the HT13 last-CTA ORDERING gate (review item 4), CPU only, from `cuobjdump -sass` text.

    python3 -I ordgate.py --cand sass/dev-r7.sass [--json out.json] [--mutant --expect REGEX]
    python3 -I ordgate.py --selftest [climb-r6/comm/sass/dev-r6.sass.gz]

projgate proves WHAT the projection computes; this gate proves WHEN its loads may run. In every non-P2 kR7Proj kernel
(ar_kernel_r3 mask & 512, mask & 32 clear: the one-shot path every rows <= 3 fold takes) the projecting CTA is the last
one through st->arrive3, and it must read p.out only after every CTA's p.out stores are visible at gpu scope:

  O1 release   the arrival atomic A (the ATOM*.ADD whose address is st + 0x54 = DevState.arrive3) is preceded, on every
               path, by a CTA barrier; at that barrier and at A every global write on the path is followed by an
               unpredicated MEMBAR of GPU or SYS scope (each thread fences its own stores before the barrier that
               orders them before thread 0's arrival; MEMBAR.*.CTA does not count);
  O2 last CTA  A's result feeds a shared-memory flag (STS); evaluated concretely for gridDim.x G in {2, 3, 16} and an
               arrival value v in {0, 1, G-2, G-1, G}: the flag is nonzero iff v == G - 1;
  O3 acquire   every global load of the projection (the LDGs in the y stores' dataflow slice, from projgate) is
               (a) unreachable when the flag read back (LDS of the same address) is 0: CFG edges whose condition is
                   decidably false under that value are removed, and the load's own guard must not be decidably true;
               (b) preceded on every path by that LDS and, after the last barrier on the path, by an unpredicated
                   MEMBAR of GPU or SYS scope.
Anything not modelled or not decidable FAILS (fail closed). The must-fail mutants mo_nofence (O1), mo_first (O2) and
mo_noacq (O3) are gate cases MO1..MO3 once compiled. --selftest needs no r7 build: a synthetic r7-shaped kernel and 5
must-fail edits of it (every rule), plus O1/O2 on the REAL last_cta() arrival of r6's ar_kernel (dev-r6 SASS, arrival
st + 0x08) and 2 must-fail edits of that real code."""
import argparse
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sassir as S  # noqa: E402
import fpflow as F  # noqa: E402

ST_TAG = (F.REGIONS.index("st") + 1) << 40
ARRIVE3 = 0x54   # offsetof(DevState, arrive3): seq 0, arrive 8, arrive2 12, depart 16, t_start 24 .. n_timed 72, entered 80
ARRIVE = 0x08    # offsetof(DevState, arrive) (r6 last_cta, selftest only)
GWRITE = ("STG", "ST", "RED", "REDG", "ATOMG", "ATOM")


class Fail(Exception):
    pass


def executes(x):
    return not (x.gpred == "PT" and x.gneg)


def unpred(x):
    return x.gpred == "PT" and not x.gneg


def is_fence(x):
    return x.base == "MEMBAR" and unpred(x) and any(m in ("GPU", "SYS") for m in x.mods)


def is_bar(x):
    return x.base == "BAR" and executes(x) and "ARV" not in x.mods


def ins_preds(k, i):
    x = k.ins[i]
    b = k.blocks[x.block]
    if i != b[0]:
        return [i - 1]
    return [k.blocks[p][-1] for p in k.pred[x.block] if p in k.reach]


def forward(k, entry, top, transfer, meet):
    """must-analysis to a fixpoint: state BEFORE every instruction of a reachable block"""
    outb = {b: top for b in k.reach}
    before = {}
    for _ in range(200):
        changed = False
        for b in k.rpo:
            ps = [outb[p] for p in k.pred[b] if p in k.reach]
            s = meet(ps + [entry]) if b == 0 else (meet(ps) if ps else top)
            for i in k.blocks[b]:
                before[i] = s
                s = transfer(k.ins[i], s)
            if outb[b] != s:
                outb[b] = s
                changed = True
        if not changed:
            return before
    raise Fail("dataflow did not converge")


def AND(ss):
    return tuple(all(v) for v in zip(*ss)) if isinstance(ss[0], tuple) else all(ss)


def addr_of(k, x, env):
    return S.addr_value(S.Evaluator(k, env), x.vals["addr"])


def check(k, env_of, arrive_off, gated=None, rules=("O1", "O2", "O3")):
    """env_of(tid, nctaid) -> S.Env. gated: instruction indices of the projection loads (required for O3)."""
    notes = []
    # ---- the arrival atomic: ATOM*.ADD at st + arrive_off
    cand = [x for x in k.ins if x.base in ("ATOMG", "ATOM") and "ADD" in x.mods and executes(x) and x.block in k.reach]
    A = []
    for x in cand:
        try:
            a = addr_of(k, x, env_of(0, 3))
        except Exception:  # noqa: BLE001
            continue
        if a == ST_TAG + arrive_off:
            A.append(x)
    if len(A) != 1:
        raise Fail(f"expected exactly one arrival atomic at st+{arrive_off:#x}, found {len(A)}")
    A = A[0]
    notes.append(f"arrival /*{A.addr:04x}*/ {A.text}")
    flags = [x for x in k.ins if x.base == "STS" and executes(x) and x.block in k.reach and x.vals and
             A.idx in k.slice_ins(x.vals["data"])]
    if not flags:
        raise Fail("the arrival atomic's result feeds no shared-memory flag (STS)")
    # ---- O1 release
    if "O1" in rules:
        def tr(x, s):
            if x.base in GWRITE and executes(x) and x is not A:
                return False
            if x.base == "CALL":
                return False
            if is_fence(x):
                return True
            return s
        st = forward(k, True, True, tr, AND)
        if not st.get(A.idx, False):
            raise Fail(f"O1: a global write reaches the arrival /*{A.addr:04x}*/ without a gpu-scope MEMBAR after it")
        bars, seen, stack = set(), set(), [A.idx]
        while stack:
            i = stack.pop()
            for p in ins_preds(k, i):
                if p in seen:
                    continue
                seen.add(p)
                x = k.ins[p]
                if is_bar(x):
                    bars.add(p)
                    continue
                if x.base in GWRITE and executes(x):
                    raise Fail(f"O1: global write /*{x.addr:04x}*/ {x.text} reaches the arrival with no barrier between")
                stack.append(p)
            if not ins_preds(k, i) and i == k.blocks[0][0]:
                raise Fail("O1: a path from the kernel entry reaches the arrival without a CTA barrier")
        if not bars:
            raise Fail("O1: no CTA barrier before the arrival")
        for b in sorted(bars):
            if not st.get(b, False):
                x = k.ins[b]
                raise Fail(f"O1: barrier /*{x.addr:04x}*/ before the arrival is reached with unfenced global writes "
                           f"(each thread must MEMBAR.GPU its stores before it)")
        notes.append(f"O1 ok: barrier(s) {[f'/*{k.ins[b].addr:04x}*/' for b in sorted(bars)]} fenced; arrival fenced")
    # ---- O2 last CTA
    if "O2" in rules:
        for G in (2, 3, 16):
            for v in sorted({0, 1, G - 2, G - 1, G}):
                env = env_of(0, G)
                env.loads[A.idx] = v
                ev = S.Evaluator(k, env)
                for x in flags:
                    g = ev.strict(x.vals["guard"])
                    val = ev.strict(x.vals["data"][0])
                    if g is None or val is None:
                        raise Fail(f"O2: flag /*{x.addr:04x}*/ not decidable for arrival {v} of {G}")
                    if g and (val != 0) != (v == G - 1):
                        raise Fail(f"O2: flag /*{x.addr:04x}*/ = {val} for arrival value {v} with gridDim.x {G} "
                                   f"(only the last CTA, v == {G - 1}, may project)")
        notes.append(f"O2 ok: flag(s) {[f'/*{x.addr:04x}*/' for x in flags]} nonzero iff arrival == gridDim.x - 1")
    # ---- O3 acquire
    if "O3" in rules:
        if not gated:
            raise Fail("O3: no projection loads given")
        sa = {addr_of(k, x, env_of(0, 3)) for x in flags}
        go = [x for x in k.ins if x.base == "LDS" and executes(x) and x.block in k.reach and x.vals and
              addr_of(k, x, env_of(1, 3)) in sa]
        if not go:
            raise Fail("O3: nobody reads the flag back (no LDS of its address)")
        env = env_of(1, 3)
        for x in go:
            env.loads[x.idx] = 0
        ev = S.Evaluator(k, env)
        live, stack = {0}, [0]
        while stack:
            b = stack.pop()
            for s_ in k.succ[b]:
                if s_ in live:
                    continue
                c = k.edge_cond.get((b, s_), k.TRUE)
                if ev.strict(c) == 0:
                    continue
                live.add(s_)
                stack.append(s_)
        for i in gated:
            x = k.ins[i]
            if x.block in live and ev.strict(x.vals["guard"]) != 0:
                raise Fail(f"O3: projection load /*{x.addr:04x}*/ {x.text} can run when the last-CTA flag is 0")
        goi = {x.idx for x in go}

        def tr3(x, s):
            f, g = s
            if is_bar(x):
                f = False
            if is_fence(x):
                f = True
            if x.idx in goi:
                g = True
            return (f, g)
        st3 = forward(k, (False, False), (True, True), tr3, AND)
        for i in gated:
            f, g = st3.get(i, (False, False))
            x = k.ins[i]
            if not g:
                raise Fail(f"O3: projection load /*{x.addr:04x}*/ reachable without reading the flag")
            if not f:
                raise Fail(f"O3: projection load /*{x.addr:04x}*/ not preceded by a gpu-scope MEMBAR after the last barrier")
        notes.append(f"O3 ok: {len(gated)} projection loads gated by the flag (LDS {[f'/*{x.addr:04x}*/' for x in go]}) "
                     f"and fenced after the barrier")
    return notes


def env_fp(tid, nctaid):
    env, _ = F.make_env(0, 1, tid=tid, cta=0, nctaid=nctaid)
    env.sr["SR_CgaCtaId"] = 0
    return env


def env_r7(tid, nctaid):
    import projgate as P
    env = P.env_r7(tid)
    env.cb[(0, 0x370)] = nctaid
    env.sr["SR_CgaCtaId"] = 0
    env.sr.pop("SR_WARPDIV", None)  # r8: both sides of a BRA.DIV stay possible here (fail closed)
    return env


def r7_kernels(fs):
    out = []
    for n, body in fs.items():
        m = re.search(r"ar_kernel_r3ILi(\d)ELb(\d)ELi8ELi(\d+)EE", n)
        if m and int(m.group(3)) & 512 and not int(m.group(3)) & 32:
            out.append((n, body))
    return out


def gated_of(name, body, proofs=None):
    """r8-comm2fix: O3 gates exactly the global loads that projgate PROVED to be the projection's x / d operands (in
    every one of its environments). r7 took every LDG in the y stores' syntactic register slice; a masked x word is
    `MOV Rx, RZ; @!P LDG Rx`, whose slice keeps Rx's older definition, i.e. the AR's own staging load of p.in
    (/*0960*/ etc.), which runs before the flag by design: 0/12 on a correct kernel. No proof -> fail closed."""
    import projgate as P
    pr = (proofs or P.prove_all([(name, body)]))[name]
    if not pr.get("ok"):
        raise Fail(f"O3: the projection is not proven by projgate, its operand loads are unknown ({pr.get('error')})")
    k = S.Kernel(name, body)
    g = sorted(pr["loads"])
    if not g or any(k.ins[i].base not in ("LDG", "LD") for i in g):
        raise Fail("O3: projgate's operand loads are not global loads")
    return k, g


# ------------------------------------------------------------------------------------------------ selftest -----
SYN = """
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   LDC.64 R2, c[0x0][0x388] ;
        /*0020*/                   LDC.64 R4, c[0x0][0x3d8] ;
        /*0030*/                   S2UR UR5, SR_CgaCtaId ;
        /*0040*/                   UMOV UR6, 0x400 ;
        /*0050*/                   ULEA UR6, UR5, UR6, 0x18 ;
        /*0060*/                   IMAD.WIDE.U32 R6, R0, 0x10, R2 ;
        /*0070*/                   STG.E.128 desc[UR4][R6.64], R8 ;
        /*0080*/                   MEMBAR.SC.GPU ;
        /*0090*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;
        /*00a0*/                   ISETP.NE.AND P0, PT, R0, RZ, PT ;
        /*00b0*/               @P0 BRA 0x150 ;
        /*00c0*/                   MOV R10, 0x1 ;
        /*00d0*/                   MEMBAR.ALL.GPU ;
        /*00e0*/                   ATOMG.E.ADD.STRONG.GPU PT, R11, desc[UR4][R4.64+0x54], R10 ;
        /*00f0*/                   LDC R12, c[0x0][0x370] ;
        /*0100*/                   IADD3 R12, PT, PT, R12, -0x1, RZ ;
        /*0110*/                   ISETP.NE.AND P1, PT, R11, R12, PT ;
        /*0120*/                   SEL R13, RZ, 0x1, P1 ;
        /*0130*/                   STS [UR6+0x8], R13 ;
        /*0140*/                   NOP ;
        /*0150*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;
        /*0160*/                   LDS R15, [UR6+0x8] ;
        /*0170*/                   ISETP.NE.AND P2, PT, R15, RZ, PT ;
        /*0180*/              @!P2 EXIT ;
        /*0190*/                   MEMBAR.SC.GPU ;
        /*01a0*/                   LDG.E.128.STRONG.GPU R16, desc[UR4][R6.64] ;
        /*01b0*/                   STG.E.128 desc[UR4][R6.64+0x10], R16 ;
        /*01c0*/                   EXIT ;
        /*01d0*/                   BRA 0x1d0;
"""
SYN_MUT = [  # (name, old, new, expected reason)
    ("nofence", "MEMBAR.SC.GPU ;\n        /*0090*/", "NOP ;\n        /*0090*/", "O1"),
    ("cta-fence", "MEMBAR.SC.GPU ;\n        /*0090*/", "MEMBAR.ALL.CTA ;\n        /*0090*/", "O1"),
    ("first", "IADD3 R12, PT, PT, R12, -0x1, RZ", "IADD3 R12, PT, PT, RZ, RZ, RZ", "O2"),
    ("unguarded", "@!P2 EXIT", "NOP", "O3"),
    ("noacq", "MEMBAR.SC.GPU ;\n        /*01a0*/", "NOP ;\n        /*01a0*/", "O3"),
]


def syn_kernel(text):
    body = S.split_functions("Function : syn\n" + text)["syn"]
    k = S.Kernel("syn", body)
    return k, [x.idx for x in k.ins if x.base == "LDG"]


def selftest(r6_path):
    res = []
    k, g = syn_kernel(SYN)
    try:
        check(k, env_fp, ARRIVE3, g)
        res.append(("synthetic r7 tail", True, "PASS"))
    except Fail as e:
        res.append(("synthetic r7 tail", False, f"base FAILS: {e}"))
    for name, old, new, why in SYN_MUT:
        assert SYN.count(old) == 1, name
        k, g = syn_kernel(SYN.replace(old, new))
        try:
            check(k, env_fp, ARRIVE3, g)
            res.append((f"synthetic mutant {name}", False, "NOT caught"))
        except Fail as e:
            res.append((f"synthetic mutant {name}", str(e).startswith(why), f"caught: {e}"))
    if r6_path:
        text = (gzip.open(r6_path, "rt") if r6_path.endswith(".gz") else open(r6_path)).read()
        fs = S.split_functions(text)
        nm = [n for n in fs if re.search(r"ar_kernelILi0ELb0EE", n)]
        if len(nm) != 1:
            res.append(("r6 ar_kernel last_cta", False, f"{len(nm)} one-shot bf16 ar_kernel in {r6_path}"))
        else:
            body = fs[nm[0]]
            try:
                notes = check(S.Kernel(nm[0], body), env_fp, ARRIVE, rules=("O1", "O2"))
                res.append(("real r6 ar_kernel last_cta (O1 O2)", True, "; ".join(notes)))
            except Fail as e:
                res.append(("real r6 ar_kernel last_cta (O1 O2)", False, f"base FAILS: {e}"))
            # must-fail edits of the REAL code: drop the fence before the arrival barrier; compare with 0
            ai = [i for i, (a, t) in enumerate(body) if t.startswith("ATOMG.E.ADD") and "-0x10" in t]
            bi = max(i for i in range(ai[0]) if body[i][1].startswith("BAR.SYNC")) if ai else None
            fi = [i for i in range(bi) if body[i][1].startswith("MEMBAR.SC.SYS")] if bi else []
            ui = [i for i in range(ai[0], ai[0] + 12) if body[i][1].startswith("UIADD3") and "-0x1" in body[i][1]] if ai else []
            for name, idx, repl, why in (("real nofence", fi[-1:] , "NOP", "O1"),
                                         ("real first", ui[:1], None, "O2")):
                if not idx:
                    res.append((name, False, "edit site not found"))
                    continue
                b2 = list(body)
                a_, t_ = b2[idx[0]]
                b2[idx[0]] = (a_, repl if repl else re.sub(r"UR\d+, -0x1, URZ", "URZ, URZ, URZ", t_))
                try:
                    check(S.Kernel(nm[0], b2), env_fp, ARRIVE, rules=("O1", "O2"))
                    res.append((name, False, "NOT caught"))
                except Fail as e:
                    res.append((name, str(e).startswith(why), f"caught: {e}"))
    for n, ok, m in res:
        print(f"selftest {n}: {'OK' if ok else 'BAD'} ({m[:200]})")
    good = all(ok for _, ok, _ in res)
    print(f"ORDGATE_SELFTEST {'PASS' if good else 'FAIL'} {sum(ok for _, ok, _ in res)}/{len(res)}")
    return 0 if good else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand")
    ap.add_argument("--json")
    ap.add_argument("--mutant", action="store_true")
    ap.add_argument("--expect", default=".")
    ap.add_argument("--selftest", nargs="?", const="", default=None)
    a = ap.parse_args()
    if a.selftest is not None:
        return selftest(a.selftest)
    fs = S.split_functions(F.load_text(a.cand))
    ks = r7_kernels(fs)
    rep = {}
    import projgate as P
    proofs = P.prove_all(ks)
    for n, body in ks:
        try:
            k, g = gated_of(n, body, proofs)
            rep[n] = {"ok": True, "notes": check(k, env_r7, ARRIVE3, g)}
        except Fail as e:
            rep[n] = {"ok": False, "why": str(e)}
        except Exception as e:  # noqa: BLE001  (fail closed)
            rep[n] = {"ok": False, "why": f"{type(e).__name__}: {e}"}
        print(f"{'PASS' if rep[n]['ok'] else 'FAIL'} {n[-60:]}: {rep[n].get('why') or rep[n]['notes'][-1]}")
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    if not ks:
        print("ORDGATE FAIL: no non-P2 kR7Proj kernel in the candidate")
        return 1
    if a.mutant:
        caught = [n for n, r in rep.items() if not r["ok"] and re.search(a.expect, r["why"])]
        print(f"ORD MUTANT: non-P2 kR7Proj kernels caught {len(caught)}/{len(ks)} (reason /{a.expect}/) -> "
              f"{'KILLED' if len(caught) == len(ks) else 'SURVIVED'}")
        return 0 if len(caught) == len(ks) else 1
    good = sum(r["ok"] for r in rep.values())
    print(f"ORDGATE {'PASS' if good == len(ks) else 'FAIL'}: {good}/{len(ks)} non-P2 kR7Proj kernels")
    return 0 if good == len(ks) else 1


if __name__ == "__main__":
    sys.exit(main())
