#!/usr/bin/env python3
"""run_gate.py [--sass DIR] [--out DIR]: the whole r6-comm binary gate suite (CPU only; stdlib python).

Cases (each must give its expected verdict; any other outcome is a FAIL of the suite):
  G0  fpflow  live 14d049dd  vs  r3 .so 7ec43164 (the C1 build)               PASS  (identity 39/39 + 24 r3 kernels proven)
  G1  fpflow  live           vs  dev-ctl_r3 (r3 source rebuilt now, device TU)  PASS  (method control)
  G2  fpflow  r3 .so         vs  dev-ctl_r3                                    identity 63/63 (the device-TU SASS == the .so SASS)
  G3  fpflow  live           vs  dev-r6 (C1 + C2 source)                       PASS  (all 56 r3/r6 kernels proven)
  G4  fpflow  r3 .so         vs  dev-r6                                        identity 63/63 (r3 kernels unchanged by C2)
  G5  pdlcheck dev-r6 (+ live)                                                 PASS
  M1  fpflow --mutant m_rev      expect /rank|reads|term/   KILLED
  M2  fpflow --mutant m_seed0    expect /seed/              KILLED
  M3  fpflow --mutant m_negzero  expect /seed/              KILLED
  M4  fpflow --mutant m_ffma     expect /FFMA/              KILLED
  M5  fpflow --mutant m_tree     expect /tree|seed/         KILLED
  M6  fpflow          m_waitlate                            PASS  (no FP change: the dataflow gate must not object)
  M7  pdlcheck --mutant m_waitlate                          KILLED (W2: accesses before the wait)
  D*  the same checks on the full-build SASS of THE .so KIT.json records (sass/r6-<sha8>.sass from ops/build8-r6.sh,
      sha8 = KIT.json's r6 .so sha256[:8]; review N3: never "the last dump by name"). KIT.json records a .so but its
      dump is missing -> D cases MISSING (suite FAIL). No .so recorded yet -> no D cases. SUMMARY.txt then carries
      "GATED_SO_SHA256 <sha>" only when every D case passed; install.py refuses a .so without that line (review N4).
Writes <out>/<case>.txt and <out>/SUMMARY.txt; exit 0 only if every case gave its expected verdict."""
import argparse
import os
import subprocess
import sys
import time

H = os.path.dirname(os.path.abspath(__file__))


def kit_so_sha(path):
    """the r6 .so sha256 KIT.json records (None before the window's build)"""
    import json
    try:
        k = json.load(open(path))
    except OSError:
        return None
    for f in k.get("files", []):
        if f.get("path", "").endswith("r6comm/build/rdma_ar_ext.so"):
            return f.get("sha256") or None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", default=os.path.join(H, "..", "sass"))
    ap.add_argument("--out", default=os.path.join(H, "results"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--kit-json", default=os.path.join(H, "..", "KIT.json"))
    a = ap.parse_args()
    def s(n):  # the .sass dumps are kept gzipped in the kit; either form is accepted
        p = os.path.join(a.sass, n)
        return p if os.path.exists(p) or not os.path.exists(p + ".gz") else p + ".gz"
    live, r3 = s("live-14d049dd.sass"), s("r3-7ec43164.sass")
    fp = [sys.executable, "-I", os.path.join(H, "fpflow.py")]
    pc = [sys.executable, "-I", os.path.join(H, "pdlcheck.py")]
    cases = [
        ("G0_live_vs_r3so", fp + ["--ref", live, "--cand", r3], "FPFLOW_GATE PASS"),
        ("G1_live_vs_devctl", fp + ["--ref", live, "--cand", s("dev-ctl_r3.sass")], "FPFLOW_GATE PASS"),
        ("G2_r3so_vs_devctl", fp + ["--ref", r3, "--cand", s("dev-ctl_r3.sass")], "IDENTITY: 63/63"),
        ("G3_live_vs_devr6", fp + ["--ref", live, "--cand", s("dev-r6.sass")], "FPFLOW_GATE PASS"),
        ("G4_r3so_vs_devr6", fp + ["--ref", r3, "--cand", s("dev-r6.sass")], "IDENTITY: 63/63"),
        ("G5_pdl_devr6", pc + ["--cand", s("dev-r6.sass"), "--live", live], "PDLCHECK PASS"),
        ("M1_m_rev", fp + ["--ref", live, "--cand", s("dev-m_rev.sass"), "--mutant", "--expect", r"term \d for rank|rank order"], "-> KILLED"),
        ("M2_m_seed0", fp + ["--ref", live, "--cand", s("dev-m_seed0.sass"), "--mutant", "--expect", "seed"], "-> KILLED"),
        ("M3_m_negzero", fp + ["--ref", live, "--cand", s("dev-m_negzero.sass"), "--mutant", "--expect", "seed"], "-> KILLED"),
        ("M4_m_ffma", fp + ["--ref", live, "--cand", s("dev-m_ffma.sass"), "--mutant", "--expect", "FFMA"], "-> KILLED"),
        ("M5_m_tree", fp + ["--ref", live, "--cand", s("dev-m_tree.sass"), "--mutant", "--expect", "tree|seed"], "-> KILLED"),
        ("M6_m_waitlate_fp", fp + ["--ref", live, "--cand", s("dev-m_waitlate.sass")], "FPFLOW_GATE PASS"),
        ("M7_m_waitlate_pdl", pc + ["--cand", s("dev-m_waitlate.sass"), "--mutant"], "-> KILLED"),
    ]
    so_sha = kit_so_sha(a.kit_json)
    if so_sha:
        f = s(f"r6-{so_sha[:8]}.sass")  # the dump of exactly the recorded .so (a missing one is reported MISSING)
        cases += [("D3_live_vs_r6so", fp + ["--ref", live, "--cand", f], "FPFLOW_GATE PASS"),
                  ("D4_r3so_vs_r6so", fp + ["--ref", r3, "--cand", f], "IDENTITY: 63/63"),
                  ("D5_pdl_r6so", pc + ["--cand", f, "--live", live], "PDLCHECK PASS"),
                  ("D6_devr6_vs_r6so", fp + ["--ref", s("dev-r6.sass"), "--cand", f], "IDENTITY: ")]
    os.makedirs(a.out, exist_ok=True)
    lines, bad = [], 0
    for name, cmd, want in cases:
        if a.only and a.only not in name:
            continue
        miss = [p for p in cmd if p.endswith((".sass", ".sass.gz")) and not os.path.exists(p)]
        if miss:
            lines.append(f"{name:22s} MISSING input {miss}")
            bad += 1
            continue
        t0 = time.time()
        r = subprocess.run(cmd + ["--json", os.path.join(a.out, name + ".json")], capture_output=True, text=True)
        out = r.stdout + r.stderr
        open(os.path.join(a.out, name + ".txt"), "w").write(" ".join(os.path.relpath(c) if c.endswith((".sass", ".gz")) else c
                                                                      for c in cmd) + "\n" + out)
        last = [l for l in out.splitlines() if l.strip()][-1:] or ["(no output)"]
        idl = [l for l in out.splitlines() if l.startswith("IDENTITY:")]
        hit = want in out
        if name.startswith("D6"):
            hit = bool(idl) and idl[0].split()[1].split("/")[0] == idl[0].split()[1].split("/")[1]
        ok = hit and (r.returncode == 0 or want.startswith("IDENTITY"))
        bad += not ok
        lines.append(f"{name:22s} {'OK ' if ok else 'BAD'} ({time.time() - t0:5.1f} s) {(idl[0] + ' | ') if idl else ''}{last[0]}")
        print(lines[-1], flush=True)
    dl = [l for l in lines if l.startswith("D")]
    ncase = len(lines)
    if so_sha and not bad and len(dl) == 4 and all(l.split()[1] == "OK" for l in dl):
        lines.append(f"GATED_SO_SHA256 {so_sha} (D3-D6 OK on sass/r6-{so_sha[:8]}.sass)")
    elif not so_sha:
        lines.append("GATED_SO_SHA256 none (KIT.json records no r6 .so yet: no D cases; install.py refuses)")
    lines.append(f"RUN_GATE {'PASS' if not bad else 'FAIL'}: {ncase - bad}/{ncase} cases gave the expected verdict")
    open(os.path.join(a.out, "SUMMARY.txt"), "w").write("\n".join(lines) + "\n")
    print(lines[-1])
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
