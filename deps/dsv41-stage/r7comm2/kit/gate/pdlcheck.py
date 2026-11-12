#!/usr/bin/env python3
"""pdlcheck.py --cand cand.sass [--live live.sass] [--mutant] [--json out.json]: the C2 (PDL) binary check. CPU only.

SASS of the PTX instructions: griddepcontrol.wait = ACQBULK, griddepcontrol.launch_dependents = PREEXIT (both seen in the
r6 build; the live and r3 binaries contain neither).

For every ar_kernel_r3<mode, f32, 8, mask> of the candidate:
  mask & 128 (kR6Pdl):
    W1  at least one ACQBULK, none predicated;
    W2  no EXIT, CALL or global memory instruction (LDG / STG / ATOMG / RED / LD / ST / ATOM) is reachable from the
        kernel entry without passing an ACQBULK, except loads whose address decodes (concrete evaluation, every rank,
        several parameter sets) to ctrl + 0x40 = Ctrl::posted_seq: the one access r6 allows before the wait;
        (an EXIT before the wait would let a CTA finish without waiting: transitivity for later PDL kernels)
    W3  mask & 256 (kR6Trig): at least one PREEXIT, none reachable without passing an ACQBULK; without 256: no PREEXIT.
  mask & 128 == 0: no ACQBULK and no PREEXIT (the r3 kernels are unchanged).
Every live kernel (ar_kernel_pl / ar_kernel / pingpong) has neither instruction.
--mutant (m_waitlate): exit 0 only if EVERY PDL kernel fails W2."""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpflow as F  # noqa: E402
import sassir as S  # noqa: E402

MEM = ("LDG", "STG", "ATOMG", "RED", "REDG", "LD", "ST", "ATOM")
CTRL_POSTED_SEQ = 0x40


def pre_wait(K):
    """instructions reachable from entry on some path that has not executed an ACQBULK (instruction granularity)."""
    seen_blocks, out, work = set(), [], [0]
    while work:
        bi = work.pop()
        if bi in seen_blocks:
            continue
        seen_blocks.add(bi)
        stopped = False
        for i in K.blocks[bi]:
            x = K.ins[i]
            if x.base == "ACQBULK" and x.gpred == "PT" and not x.gneg:
                stopped = True
                break
            out.append(x)
        if not stopped:
            work.extend(K.succ[bi])
    return out


def allowed_load(K, x):
    """True when x is a load whose address is ctrl + 0x40 in every environment where it executes."""
    if x.base not in ("LDG", "LD") or not x.vals or "addr" not in x.vals:
        return False
    n = 0
    for tid, cta in ((0, 0), (0, 1), (37, 0)):
        for rank in range(F.W):
            env, info = F.make_env(rank, 11 + tid + cta, tid=tid, cta=cta)
            ev = S.Evaluator(K, env, 1)
            try:
                if ev.reach(x.block) == 0:
                    continue
                if x.vals.get("guard") is not None and ev.strict(x.vals["guard"]) == 0:
                    continue
                d = F.decode(S.addr_value(ev, x.vals["addr"]), info)
            except Exception:  # noqa: BLE001 - anything not decodable is not the allowed load
                return False
            n += 1
            if d is None or d[0] != "ctrl" or d[3] != CTRL_POSTED_SEQ:
                return False
    return n > 0


def check(name, body):
    K = S.Kernel(name, body)
    m = re.search(r"ar_kernel_r3ILi(\d)ELb(\d)ELi8ELi(\d+)EE", name)
    errs, notes = [], []
    acq = [x for x in K.ins if x.base == "ACQBULK"]
    pre = [x for x in K.ins if x.base == "PREEXIT"]
    if not m:
        if acq or pre:
            errs.append(f"{len(acq)} ACQBULK / {len(pre)} PREEXIT in a non-r3 kernel")
        return K, errs, notes
    mask = int(m.group(3))
    if mask & 128:
        if not acq:
            errs.append("W1: PDL kernel without ACQBULK (griddepcontrol.wait)")
        for x in acq:
            if x.gpred != "PT" or x.gneg:
                errs.append(f"W1: predicated ACQBULK at /*{x.addr:04x}*/")
        early = pre_wait(K)
        n_ok = 0
        for x in early:
            if x.base in ("EXIT", "CALL"):
                errs.append(f"W2: {x.text} at /*{x.addr:04x}*/ reachable before the wait")
            elif x.base in MEM:
                if allowed_load(K, x):
                    n_ok += 1
                    notes.append(f"pre-wait load /*{x.addr:04x}*/ {x.text} = ctrl->posted_seq (allowed)")
                else:
                    errs.append(f"W2: {x.text} at /*{x.addr:04x}*/ reachable before the wait")
        early_idx = {x.idx for x in early}
        if mask & 256:
            if not pre:
                errs.append("W3: trigger kernel without PREEXIT (griddepcontrol.launch_dependents)")
            for x in pre:
                if x.idx in early_idx:
                    errs.append(f"W3: PREEXIT at /*{x.addr:04x}*/ reachable before the wait")
        elif pre:
            errs.append(f"W3: {len(pre)} PREEXIT in a kernel without the trigger bit")
        notes.append(f"{len(acq)} ACQBULK, {len(pre)} PREEXIT, {len(early)} instructions before the wait, "
                     f"{n_ok} allowed pre-wait load(s)")
    else:
        if acq or pre:
            errs.append(f"{len(acq)} ACQBULK / {len(pre)} PREEXIT in an r3 kernel without the PDL bit")
    return K, errs, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--live", default=None)
    ap.add_argument("--mutant", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    rep = {"cand": a.cand, "kernels": {}}
    ok = True
    if a.live:
        for n, body in S.split_functions(F.load_text(a.live)).items():
            _, errs, _ = check(n, body)
            if errs:
                ok = False
                print(f"live {F.kernel_key(n)}: {errs[0]}")
        print(f"live kernels: {'no' if ok else 'SOME'} griddepcontrol instructions")
    seen = {}
    for n, body in S.split_functions(F.load_text(a.cand)).items():
        k = F.kernel_key(n)
        if k in seen:
            continue
        _, errs, notes = check(n, body)
        seen[k] = (errs, notes)
        rep["kernels"][k] = {"errors": errs, "notes": notes}
    pdl = {k: v for k, v in seen.items() if re.search(r"ar_kernel_r3I.*ELi(\d+)EE", k)
           and int(re.search(r"ELi(\d+)EE$", k).group(1)) & 128}
    other = {k: v for k, v in seen.items() if k not in pdl}
    for k in sorted(pdl):
        e, nt = pdl[k]
        print(f"pdl  {k:44s} {'PASS' if not e else 'FAIL'}  {nt[-1] if nt else ''}" + (f"  first error: {e[0]}" if e else ""))
    bad_other = [k for k, (e, _) in other.items() if e]
    for k in bad_other:
        print(f"other {k}: {other[k][0][0]}")
    print(f"non-PDL kernels without griddepcontrol: {len(other) - len(bad_other)}/{len(other)}")
    if a.mutant:
        caught = [k for k, (e, _) in pdl.items() if any(s.startswith("W2") for s in e)]
        v = bool(pdl) and len(caught) == len(pdl) and not bad_other
        print(f"PDL MUTANT: PDL kernels caught by W2 {len(caught)}/{len(pdl)} -> {'KILLED' if v else 'SURVIVED'}")
        rep["verdict"] = "KILLED" if v else "SURVIVED"
        rc = 0 if v else 1
    else:
        v = ok and bool(pdl) and all(not e for e, _ in pdl.values()) and not bad_other
        print(f"PDLCHECK {'PASS' if v else 'FAIL'}: {sum(not e for e, _ in pdl.values())}/{len(pdl)} PDL kernels")
        rep["verdict"] = "PASS" if v else "FAIL"
        rc = 0 if v else 1
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, sort_keys=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
