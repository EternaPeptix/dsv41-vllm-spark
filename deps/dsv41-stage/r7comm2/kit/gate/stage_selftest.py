#!/usr/bin/env python3
"""stage_selftest.py <ref live.sass(.gz)> <cand r3/r6/r7.sass(.gz)>: must-fail evidence for the r7 staging check
(review item "make the binary gate's concrete environment enter the unrolled staging loop"), CPU only.

  S0  the unedited candidate passes the full fpflow gate (reduction proofs + staging copies, unrolled body reached);
  S1  SASS-level m_stage: in every grid-staging ar_kernel_r3 (no S1 / Sbuf bit), the data registers of the 3rd and
      4th send stores of the 4x-unrolled body are swapped (send[i + 2s] <- in[i + 3s] and vice versa); every such
      kernel must FAIL with a staging-copy error;
  S2  the same edit checked with only the r6 environments (3 CTAs, nvec 4000: the unrolled loop is never entered):
      the copy proof alone cannot see it, and the coverage assertion must turn that into a FAIL ("no environment
      executes the 4x-unrolled staging body") - i.e. the r6 environment set was blind, the r7 set is not.
The source-level twin of S1 is src/mutants/rdma_ar.m_stage.cu (needs a build; run through run_gate7.py in the window).
Exit 0 only if S0 passes and S1 / S2 are caught in every grid-staging kernel."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpflow as F  # noqa: E402
import sassir as S  # noqa: E402

GRID = re.compile(r"ar_kernel_r3ILi(\d)ELb(\d)ELi8ELi(\d+)EE")


def grid_kernels(funcs):
    out = []
    for n in funcs:
        m = GRID.search(n)
        if m and not (int(m.group(3)) & (16 | 64)):
            out.append(n)
    return sorted(out)


def swap_unrolled(body):
    """swap the data registers of the 3rd and 4th STG.E.128 of the first run of four unguarded STG.E.128 to the same
    base-register family inside the unrolled body; returns the edited (addr, text) list or None."""
    txt = [t for _, t in body]
    for i in range(len(txt) - 3):
        grp = []
        j = i
        while j < len(txt) and len(grp) < 4:
            if re.match(r"STG\.E\.128 ", txt[j]):
                grp.append(j)
            elif not re.match(r"(LDG|IMAD|IADD|LEA|MOV|SHF|LOP3|ISETP|UIADD|ULEA|NOP)", txt[j]):
                break
            j += 1
        if len(grp) == 4:
            a, b = grp[2], grp[3]
            ra = txt[a].rsplit(",", 1)
            rb = txt[b].rsplit(",", 1)
            new = list(body)
            new[a] = (body[a][0], ra[0] + "," + rb[1])
            new[b] = (body[b][0], rb[0] + "," + ra[1])
            return new
    return None


def run(name, body, envs=None):
    g = F.KernelGate(name, body)
    if envs is not None:
        g.STAGE_ENVS = envs
    g.run()
    return g


def main():
    ref, cand = sys.argv[1], sys.argv[2]
    funcs = S.split_functions(F.load_text(cand))
    ks = grid_kernels(funcs)
    ok = True
    r6_envs = tuple(e for e in F.KernelGate.STAGE_ENVS if e[2] == 3)
    print(f"grid-staging r3 kernels: {len(ks)}; r7 environments {len(F.KernelGate.STAGE_ENVS)}, r6 subset {len(r6_envs)}")
    for n in ks:
        key = F.kernel_key(n)
        g0 = run(n, funcs[n])
        s0 = not g0.errors and bool(g0.staging.get("unrolled_blocks"))
        mut = swap_unrolled(funcs[n])
        if mut is None:
            print(f"{key}: S0 {'PASS' if s0 else 'FAIL'}; S1/S2 could not locate the unrolled body: FAIL")
            ok = False
            continue
        g1 = run(n, mut)
        s1 = any("staging store" in e for e in g1.errors)
        g2 = run(n, mut, r6_envs)
        s2_copy_blind = not any("staging store" in e for e in g2.errors)
        s2 = any("unrolled staging body" in e for e in g2.errors)
        print(f"{key}: S0 {'PASS' if s0 else 'FAIL'} ({g0.staging.get('proven')} copies) | S1 m_stage "
              f"{'CAUGHT' if s1 else 'NOT CAUGHT'}" + (f" ({[e for e in g1.errors if 'staging' in e][0][:110]})" if s1 else "")
              + f" | S2 r6 envs: copy proof {'blind' if s2_copy_blind else 'sees it'}, coverage assertion "
              f"{'FAILS the kernel' if s2 else 'SILENT'}")
        ok &= s0 and s1 and s2
    print(f"STAGE_SELFTEST {'PASS' if ok and ks else 'FAIL'}")
    return 0 if ok and ks else 1


if __name__ == "__main__":
    raise SystemExit(main())
