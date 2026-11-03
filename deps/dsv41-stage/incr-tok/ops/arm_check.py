#!/usr/bin/env python3
"""arm_check.py <arm dir> <mode off|on|audit>: verdict for one arm.sh arm, ON spark-one, stdlib only.

Reads, in <arm dir>: exact.json, g1.jsonl, det.jsonl, ab.jsonl, rank0.incr (the DSV41_INCR_TOK lines of rank0.log),
errors.rank0 and the *.rc files. PASS needs: exact.py combined a8c3394af9d28249; G1 outputs_digest 9a18f802d930f6e7; all 9
long-prompt cases with the reference token hashes on every run; ttft_ab.py finished with prompt ids on every step; no
ERROR/Traceback in rank0.log; and the overlay state the flag asked for: off = no DSV41_INCR_TOK line at all; on/audit =
an "enabled" line, no "DISABLED", no "MISMATCH", no "attach failed". Exit 0 PASS, 1 FAIL.
"""
from __future__ import annotations

import collections
import json
import os
import sys

EXACT = "a8c3394af9d28249"
G1 = "9a18f802d930f6e7"
LONG_REF = {"big:2500": "3f4a9dd1", "big:8000": "d3e0f849", "big:32000": "7ba3d1a9", "big:128000": "89bd9111",
            "code:4000": "43c4b492", "prose:6000": "05e29037", "l3:B:T16-spam-reply:1": "37dd0438",
            "l3:B:T02-classify:1": "6b7032f2", "l3:A:T01-reviews:1": "6d8cf8a2"}
LONG_CASES = ",".join(LONG_REF)


def main(d: str, mode: str) -> int:
    fails = []

    def rd(name, default=""):
        p = os.path.join(d, name)
        return open(p).read() if os.path.exists(p) else default

    for name in ("exact", "g1", "det", "ab"):
        rc = rd(f"{name}.rc").strip()
        if rc != "0":
            fails.append(f"{name} exit {rc or 'missing'}")
    try:
        comb = json.loads(rd("exact.json", "{}")).get("combined", "")
    except ValueError:
        comb = ""
    print(f"exact {comb[:16]} (want {EXACT})")
    if not comb.startswith(EXACT):
        fails.append("exact anchor")
    dig = []
    for line in rd("g1.jsonl").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("metric") == "G1_summary":
            dig.append((r.get("outputs_digest", "")[:16], r.get("G1_mean_tok_s")))
    print(f"G1 {dig[-1] if dig else None} (want {G1})")
    if not dig or dig[-1][0] != G1:
        fails.append("G1 digest")
    C = collections.defaultdict(collections.Counter)
    for line in rd("det.jsonl").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "tok_sha" in r:
            C[r["case"]][r["tok_sha"][:8]] += 1
    bad = {c: dict(v) for c, v in C.items() if set(v) != {LONG_REF.get(c)}}
    print(f"long prompts {len(C)} of 9 cases, runs {sum(sum(v.values()) for v in C.values())}, "
          + ("all == reference" if not bad and len(C) == 9 else f"DIFF {bad}"))
    if bad or len(C) != 9:
        fails.append("long prompts")
    ab = [json.loads(x) for x in rd("ab.jsonl").splitlines() if x.strip()]
    if not ab or any(r.get("ids_sha") is None for r in ab):
        fails.append("ttft_ab incomplete")
    else:
        print(f"ttft_ab {len(ab)} steps, {ab[0]['prompt_tokens']} -> {ab[-1]['prompt_tokens']} tokens, "
              f"foreign traffic seen on {sum(1 for r in ab if r.get('foreign_after'))} steps")
    errs = rd("errors.rank0", "missing").strip()
    print(f"rank0 ERROR/Traceback lines: {errs}")
    if errs != "0":
        fails.append("rank0 errors")
    lines = [x for x in rd("rank0.incr").splitlines() if x.strip()]
    for x in lines[-6:]:
        print("  ", x[:240])
    if mode == "off":
        if lines:
            fails.append("flag off but the overlay logged")
    else:
        if not any(f"DSV41_INCR_TOK={'on' if mode == 'on' else 'audit'} enabled" in x for x in lines):
            fails.append("overlay not enabled")
        for word in ("DISABLED", "MISMATCH", "attach failed", "internal error"):
            if any(word in x for x in lines):
                fails.append(f"overlay logged {word}")
    print("ARM_CHECK", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("off", "on", "audit"):
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
