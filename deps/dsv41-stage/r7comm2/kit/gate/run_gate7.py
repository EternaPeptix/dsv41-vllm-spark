#!/usr/bin/env python3
"""run_gate7.py [--sass DIR] [--sass6 DIR] [--out DIR]: the r7-comm2 binary gate suite (CPU only, stdlib python).

Every case must give its expected verdict. A case whose input SASS does not exist yet is MISSING, which fails the suite
(the dev-r7 / D7 inputs exist only after ops/dev8-r7.sh resp. ops/build8-r7.sh ran).

r6 regression (climb-r6/comm/sass, with the r7 sassir: CS2R.32 width, SHFL destination, 2-operand IADD, predicate-operand
BRA edges) - the r6 verdicts must not change:
  G0 G1 G2 G3 G4 G5 M1..M7         as climb-r6/comm/gate/run_gate.py (G0-G4 now also run the staging-copy proof)
CPU evidence that needs no new build:
  S7a  stage_selftest live vs r3 .so 7ec43164   STAGE_SELFTEST PASS (unrolled body proven; SASS-level m_stage caught;
                                                the r6 environments alone are blind to it)
  S7b  stage_selftest live vs dev-r6            STAGE_SELFTEST PASS
  P7   projgate --selftest on the served Triton _proj_rows SASS: PROJGATE_SELFTEST PASS (6 must-fail SASS edits caught)
  P7b  projgate records of the three served Triton cubins are equal (one record, f29c9ff9b85178f7)
  A7   acqcheck --selftest: every check's must-fail mutant caught (the N1/N2 verdicts are reported as INFO lines)
  O7   ordgate --selftest (review item 4): synthetic r7-shaped tail PASS + 5 must-fail edits caught (O1 release, O2 last
       CTA, O3 acquire) + O1/O2 on the REAL r6 ar_kernel last_cta() arrival (dev-r6) PASS and 2 must-fail edits caught
dev-r7 (device-only cubins of the r7 source and its mutants, ops/dev8-r7.sh, spark-eight while no lock exists):
  R0  fpflow live vs dev-r7                     FPFLOW_GATE PASS (identity 39/39 live kernels = the flag-off path;
                                                every r3/r6/r7 kernel's reduction proven + staging copies)
  R1  fpflow dev-ctl_r6 vs dev-r7               IDENTITY N/N (every r6 kernel instruction-identical in the r7 build)
  R2  pdlcheck dev-r7                           PDLCHECK PASS
  R3  projgate served Triton vs dev-r7          PROJGATE PASS (every kR7Proj kernel = the served projection dataflow)
  R4  ordgate dev-r7                            ORDGATE PASS (every non-P2 kR7Proj kernel: release, last CTA, acquire)
  MR1..MR5  fpflow --mutant m_rev / m_seed0 / m_negzero / m_ffma / m_tree   KILLED (MR4 reason: FFMA in the reduction chain)
  V1..V6    fpflow --mutant on the UNMUTATED dev-r7 with each MR1-5/MR8 reason: caught 0/N, SURVIVED (r8 review N1:
            no MR catch is already met by the base kernels)
  MR6 fpflow m_waitlate PASS; MR7 pdlcheck --mutant m_waitlate KILLED
  MR8 fpflow --mutant m_stage --expect staging --mutant-skip-mask 64         KILLED
  MP1..MP5 projgate --mutant mp_first / mp_warp / mp_l2 / mp_unfused / mp_chunk   KILLED
  MO1..MO3 ordgate --mutant mo_nofence (O1) / mo_first (O2) / mo_noacq (O3)        KILLED
D7 (the full build of exactly the .so KIT.json records: sass/r7-<sha8>.sass from ops/build8-r7.sh):
  D7_live_vs_r7so  FPFLOW PASS;  D7_pdl_r7so  PDLCHECK PASS;  D7_proj_r7so  PROJGATE PASS;  D7_ord_r7so  ORDGATE PASS;
  D7_devr7_vs_r7so IDENTITY N/N (device-TU SASS == .so SASS)
SUMMARY.txt carries "GATED_SO_SHA256 <sha>" only when every case passed including all five D7 cases; install.py refuses
any .so without that line."""
import argparse
import json
import os
import re
import subprocess
import sys
import time

H = os.path.dirname(os.path.abspath(__file__))
K = os.path.dirname(H)


def kit_so_sha(path):
    try:
        k = json.load(open(path))
    except OSError:
        return None
    for f in k.get("files", []):
        if f.get("path", "").endswith("r7comm2/build/rdma_ar_ext.so"):
            return f.get("sha256") or None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", default=os.path.join(K, "sass"))
    ap.add_argument("--sass6", default=os.path.join(K, "..", "..", "climb-r6", "comm", "sass"))
    ap.add_argument("--out", default=os.path.join(H, "results"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--kit-json", default=os.path.join(K, "KIT.json"))
    ap.add_argument("--skip-missing", action="store_true",
                    help="report MISSING cases but do not fail on them (pre-build CPU evidence runs only; never in a window)")
    a = ap.parse_args()

    def s(d, n):
        p = os.path.join(d, n)
        return p if os.path.exists(p) or not os.path.exists(p + ".gz") else p + ".gz"
    S6, S7 = a.sass6, a.sass
    live, r3 = s(S6, "live-14d049dd.sass"), s(S6, "r3-7ec43164.sass")
    py = [sys.executable, "-I"]
    fp, pc, pg = py + [os.path.join(H, "fpflow.py")], py + [os.path.join(H, "pdlcheck.py")], py + [os.path.join(H, "projgate.py")]
    og = py + [os.path.join(H, "ordgate.py")]
    tri = sorted(os.path.join(K, "ref", "triton", d, "_proj_rows.sass") for d in os.listdir(os.path.join(K, "ref", "triton")))
    cases = [
        ("G0_live_vs_r3so", fp + ["--ref", live, "--cand", r3], "FPFLOW_GATE PASS"),
        ("G1_live_vs_devctl", fp + ["--ref", live, "--cand", s(S6, "dev-ctl_r3.sass")], "FPFLOW_GATE PASS"),
        ("G2_r3so_vs_devctl", fp + ["--ref", r3, "--cand", s(S6, "dev-ctl_r3.sass")], "IDENTITY: 63/63"),
        ("G3_live_vs_devr6", fp + ["--ref", live, "--cand", s(S6, "dev-r6.sass")], "FPFLOW_GATE PASS"),
        ("G4_r3so_vs_devr6", fp + ["--ref", r3, "--cand", s(S6, "dev-r6.sass")], "IDENTITY: 63/63"),
        ("G5_pdl_devr6", pc + ["--cand", s(S6, "dev-r6.sass"), "--live", live], "PDLCHECK PASS"),
        ("M1_m_rev", fp + ["--ref", live, "--cand", s(S6, "dev-m_rev.sass"), "--mutant", "--expect", r"term \d for rank|rank order"], "-> KILLED"),
        ("M2_m_seed0", fp + ["--ref", live, "--cand", s(S6, "dev-m_seed0.sass"), "--mutant", "--expect", "seed"], "-> KILLED"),
        ("M3_m_negzero", fp + ["--ref", live, "--cand", s(S6, "dev-m_negzero.sass"), "--mutant", "--expect", "seed"], "-> KILLED"),
        ("M4_m_ffma", fp + ["--ref", live, "--cand", s(S6, "dev-m_ffma.sass"), "--mutant", "--expect", "FFMA"], "-> KILLED"),
        ("M5_m_tree", fp + ["--ref", live, "--cand", s(S6, "dev-m_tree.sass"), "--mutant", "--expect", "tree|seed"], "-> KILLED"),
        ("M6_m_waitlate_fp", fp + ["--ref", live, "--cand", s(S6, "dev-m_waitlate.sass")], "FPFLOW_GATE PASS"),
        ("M7_m_waitlate_pdl", pc + ["--cand", s(S6, "dev-m_waitlate.sass"), "--mutant"], "-> KILLED"),
        ("S7a_stage_r3so", py + [os.path.join(H, "stage_selftest.py"), live, r3], "STAGE_SELFTEST PASS"),
        ("S7b_stage_devr6", py + [os.path.join(H, "stage_selftest.py"), live, s(S6, "dev-r6.sass")], "STAGE_SELFTEST PASS"),
        ("P7_projgate_selftest", pg + ["--selftest", tri[0]], "PROJGATE_SELFTEST PASS"),
        ("P7b_projgate_served_variants", py + [os.path.join(H, "projgate.py"), "--records"] + tri, "RECORDS EQUAL"),
        ("A7_acqcheck_selftest", py + [os.path.join(H, "acqcheck.py"), "--sass", os.path.join(K, "audit", "out", "t2_sass.jsonl.gz"),
                                       "--pdl", os.path.join(K, "audit", "out", "t2_pdl.json"), "--selftest"], "ACQCHECK"),
        ("O7_ordgate_selftest", og + ["--selftest", s(S6, "dev-r6.sass")], "ORDGATE_SELFTEST PASS"),
        ("R0_live_vs_devr7", fp + ["--ref", live, "--cand", s(S7, "dev-r7.sass")], "FPFLOW_GATE PASS"),
        ("R1_devctlr6_vs_devr7", fp + ["--ref", s(S7, "dev-ctl_r6.sass"), "--cand", s(S7, "dev-r7.sass")], "IDENTITY: "),
        ("R2_pdl_devr7", pc + ["--cand", s(S7, "dev-r7.sass"), "--live", live], "PDLCHECK PASS"),
        ("R3_proj_devr7", pg + ["--ref", tri[0], "--cand", s(S7, "dev-r7.sass")], "PROJGATE PASS"),
        ("R4_ord_devr7", og + ["--cand", s(S7, "dev-r7.sass")], "ORDGATE PASS"),
    ]
    # r8-comm2fix review N1: under --mutant the UNMUTATED dev-r7 kR7Proj kernels fail with "stray FFMA" (the fold), which
    # /FFMA/ matched, so MR4 was vacuous on those 18 kernels; MR4 now needs the reduction-chain reason, and the V
    # controls below prove that no MR reason is already met by the unmutated dev-r7 (caught 0/N on the base)
    MR_REASONS = (("m_rev", r"term \d for rank|rank order", []), ("m_seed0", "seed", []), ("m_negzero", "seed", []),
                  ("m_ffma", r"FFMA at .* in the reduction chain", []), ("m_tree", "tree|seed", []))
    for i, (m, ex, extra) in enumerate(MR_REASONS, 1):
        cases.append((f"MR{i}_{m}", fp + ["--ref", live, "--cand", s(S7, f"dev-{m}.sass"), "--mutant", "--expect", ex] + extra,
                      "-> KILLED"))
    cases += [
        ("MR6_m_waitlate_fp", fp + ["--ref", live, "--cand", s(S7, "dev-m_waitlate.sass")], "FPFLOW_GATE PASS"),
        ("MR7_m_waitlate_pdl", pc + ["--cand", s(S7, "dev-m_waitlate.sass"), "--mutant"], "-> KILLED"),
        ("MR8_m_stage", fp + ["--ref", live, "--cand", s(S7, "dev-m_stage.sass"), "--mutant", "--expect", "staging",
                              "--mutant-skip-mask", "64"], "-> KILLED"),
    ]
    for i, (m, ex, extra) in enumerate(MR_REASONS + (("m_stage", "staging", ["--mutant-skip-mask", "64"]),), 1):
        cases.append((f"V{i}_base_vs_{m}", fp + ["--ref", live, "--cand", s(S7, "dev-r7.sass"), "--mutant", "--expect", ex]
                      + extra, "VACUITY"))
    # r8-comm2fix: each projection mutant must be caught for ITS reason in every kR7Proj kernel (r7 had no --expect,
    # and its base R3 failed every kernel, so "caught 18/18" proved nothing)
    for i, (m, ex) in enumerate((("mp_first", "record differs"), ("mp_warp", "tree level"), ("mp_l2", "tree level"),
                                 ("mp_unfused", "not an FFMA"), ("mp_chunk", r"y\[\d+\] = fma")), 1):
        cases.append((f"MP{i}_{m}", pg + ["--ref", tri[0], "--cand", s(S7, f"dev-{m}.sass"), "--mutant", "--expect", ex],
                      "-> KILLED"))
    for i, (m, ex) in enumerate((("mo_nofence", "^O1"), ("mo_first", "^O2"), ("mo_noacq", "^O3")), 1):
        cases.append((f"MO{i}_{m}", og + ["--cand", s(S7, f"dev-{m}.sass"), "--mutant", "--expect", ex], "-> KILLED"))
    so_sha = kit_so_sha(a.kit_json)
    if so_sha:
        f = s(S7, f"r7-{so_sha[:8]}.sass")
        cases += [("D7_live_vs_r7so", fp + ["--ref", live, "--cand", f], "FPFLOW_GATE PASS"),
                  ("D7_pdl_r7so", pc + ["--cand", f, "--live", live], "PDLCHECK PASS"),
                  ("D7_proj_r7so", pg + ["--ref", tri[0], "--cand", f], "PROJGATE PASS"),
                  ("D7_ord_r7so", og + ["--cand", f], "ORDGATE PASS"),
                  ("D7_devr7_vs_r7so", fp + ["--ref", s(S7, "dev-r7.sass"), "--cand", f], "IDENTITY: ")]
    os.makedirs(a.out, exist_ok=True)
    lines, bad, missing, info = [], 0, 0, []
    for name, cmd, want in cases:
        if a.only and a.only not in name:
            continue
        miss = [p for p in cmd if p.endswith((".sass", ".sass.gz", ".jsonl", ".jsonl.gz")) and not os.path.exists(p)]
        if miss:
            lines.append(f"{name:30s} MISSING input {[os.path.basename(m) for m in miss]}")
            missing += 1
            if not a.skip_missing:
                bad += 1
            print(lines[-1], flush=True)
            continue
        t0 = time.time()
        jcmd = cmd + (["--json", os.path.join(a.out, name + ".json")] if not name.startswith(("S7", "P7", "A7", "O7")) else [])
        r = subprocess.run(jcmd, capture_output=True, text=True)
        out = r.stdout + r.stderr
        open(os.path.join(a.out, name + ".txt"), "w").write(" ".join(os.path.relpath(c) if os.path.exists(c) else c
                                                                      for c in jcmd) + "\n" + out)
        last = [l for l in out.splitlines() if l.strip()][-1:] or ["(no output)"]
        idl = [l for l in out.splitlines() if l.startswith("IDENTITY:")]
        hit = want in out
        if want == "VACUITY":  # V controls: the unmutated dev-r7 must meet the MR reason in NO reduction kernel
            hit = bool(re.search(r"reduction kernels caught 0/[1-9]\d* .*-> SURVIVED", out))
            ok = hit and r.returncode == 1
        elif want == "IDENTITY: ":
            hit = bool(idl) and idl[0].split()[1].split("/")[0] == idl[0].split()[1].split("/")[1]
            ok = hit
        elif name.startswith("A7"):
            st = [l for l in out.splitlines() if l.startswith("selftest ")]
            ok = bool(st) and all(l.endswith(": OK") for l in st)
            info += [l for l in out.splitlines() if l.startswith("ACQCHECK")]
        else:
            ok = hit and (r.returncode == 0 or want.startswith("IDENTITY"))
        bad += not ok
        lines.append(f"{name:30s} {'OK ' if ok else 'BAD'} ({time.time() - t0:6.1f} s) {(idl[0] + ' | ') if idl else ''}{last[0][:150]}")
        print(lines[-1], flush=True)
    dl = [l for l in lines if l.startswith("D7_")]
    n = len(lines)
    for i in info:
        lines.append(f"INFO {i}  (N2 gates the C2 arm, N1 the trig arm; audit/out/acqcheck.txt)")
    if so_sha and not bad and len(dl) == 5 and all(l.split()[1] == "OK" for l in dl):
        lines.append(f"GATED_SO_SHA256 {so_sha} (every case OK incl. D7 on sass/r7-{so_sha[:8]}.sass)")
    elif not so_sha:
        lines.append("GATED_SO_SHA256 none (KIT.json records no r7 .so yet: no D7 cases; install.py refuses)")
    lines.append(f"RUN_GATE7 {'PASS' if not bad else 'FAIL'}: {n - bad - (missing if a.skip_missing else 0)}/{n} cases gave the "
                 f"expected verdict"
                 + (f" ({missing} MISSING inputs, tolerated by --skip-missing: NOT a gate pass)" if a.skip_missing and missing else ""))
    if a.skip_missing and missing:
        lines[-1] = lines[-1].replace("RUN_GATE7 PASS", "RUN_GATE7 PARTIAL")
    open(os.path.join(a.out, "SUMMARY.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[n:]))
    return 0 if not bad and not (a.skip_missing and missing) else 1


if __name__ == "__main__":
    raise SystemExit(main())
