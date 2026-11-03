#!/usr/bin/env python3
"""ab_compare.py OFF.jsonl ON.jsonl [ON_incr-tok.jsonl]: compare two ttft_ab.py runs (flag off vs DSV41_INCR_TOK=1).

Gates (exit 1 on any):
  - same number of steps, and per step the same prompt_tokens and the same ids_sha (the server's prompt_token_ids,
    returned by return_token_ids): the cache must not change a single prompt id;
  - the overlay log (if given) has no MISMATCH / verify_failed record and at least 10 "hit" records.
Reported, not gated: out_sha agreement (greedy output; a difference with equal ids_sha points at decode nondeterminism,
not at this kit: stop and review), and the first-token wait: median ttft_s over steps >= 1, over hit steps and over
verified steps (labels from the overlay log, matched to steps by prompt length), with the per-step delta OFF - ON.
"""
from __future__ import annotations

import json
import statistics
import sys


def load(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 4) if xs else None


def main(argv) -> int:
    off, on = load(argv[1]), load(argv[2])
    log = load(argv[3]) if len(argv) > 3 else None
    rc = 0
    if len(off) != len(on) or not off:
        print(f"FAIL step counts differ or empty: off {len(off)} on {len(on)}")
        return 1
    bad_ids = [a["step"] for a, b in zip(off, on) if a["ids_sha"] != b["ids_sha"] or a["prompt_tokens"] != b["prompt_tokens"]]
    if bad_ids or any(a["ids_sha"] is None for a in off + on):
        print(f"FAIL prompt ids differ at steps {bad_ids[:10]}")
        rc = 1
    else:
        print(f"OK prompt ids identical on all {len(off)} steps ({off[0]['prompt_tokens']} -> {off[-1]['prompt_tokens']} tokens)")
    bad_out = [a["step"] for a, b in zip(off, on) if a["out_sha"] != b["out_sha"]]
    print(("NOTE outputs differ at steps %s (ids equal: decode nondeterminism, review before promoting)" % bad_out[:10])
          if bad_out else "OK greedy outputs identical on all steps")
    kind = {}
    if log is not None:
        mism = [r for r in log if r.get("kind") in ("MISMATCH", "verify_failed")]
        if mism:
            print(f"FAIL overlay log has {len(mism)} mismatch records: {mism[:2]}")
            rc = 1
        by_tokens = {}
        for r in log:
            by_tokens.setdefault(r.get("tokens"), r.get("kind"))
        for b in on:
            kind[b["step"]] = by_tokens.get(b["prompt_tokens"], "unlogged")
        n_hit = sum(1 for k in kind.values() if k == "hit")
        print(f"overlay kinds over the A/B steps: { {k: list(kind.values()).count(k) for k in set(kind.values())} }")
        if n_hit < 10:
            print(f"FAIL only {n_hit} hit steps: the A/B measured little of the cache (verification window not passed?)")
            rc = 1
        hits = [r for r in log if r.get("kind") == "hit"]
        ver = [r for r in log if r.get("kind") == "verified"]
        if ver:
            print(f"server-side, verified requests: full tokenize median {med(r.get('ms_full') for r in ver)} ms, "
                  f"incremental median {med(r.get('ms_incr') for r in ver)} ms (n={len(ver)})")
        if hits:
            print(f"server-side, hits: incremental median {med(r.get('ms_incr') for r in hits)} ms (n={len(hits)})")
    steps = [(a, b) for a, b in zip(off, on) if a["step"] >= 1]
    d_all = [a["ttft_s"] - b["ttft_s"] for a, b in steps]
    print(f"TTFT median over steps >= 1: off {med(a['ttft_s'] for a, _ in steps)} s, on {med(b['ttft_s'] for _, b in steps)} s, "
          f"median per-step saving {med(d_all)} s (n={len(steps)})")
    for k in ("hit", "verified"):
        sel = [(a, b) for a, b in steps if kind.get(b["step"]) == k]
        if sel:
            print(f"TTFT on {k} steps: off {med(a['ttft_s'] for a, _ in sel)} s, on {med(b['ttft_s'] for _, b in sel)} s, "
                  f"median saving {med(a['ttft_s'] - b['ttft_s'] for a, b in sel)} s (n={len(sel)}, "
                  f"{sel[0][1]['prompt_tokens']}-{sel[-1][1]['prompt_tokens']} tokens)")
    print("AB_COMPARE", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
