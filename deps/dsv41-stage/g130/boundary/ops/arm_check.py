#!/usr/bin/env python3
"""arm_check.py: the pass or fail of one arm, for arm.sh. Stdlib only, no GPU, no service contact. Missing evidence is a FAIL.

  serving <O> <conc 0|1> <reference det.jsonl>     on spark-one, after the benchmarks: exact, G1 digest, G4 ran, 9 long prompts, rank 0 errors, load32
  fleet <dir> <ctx mode> <mk mode> [VAR=VAL ...]   on the Mac: <host>.log and <host>.audit from all 8 say the flags did what the arm claims
  record <digest> <ctx mode> <mk mode> [VAR=VAL ...]   on the Mac: prints what a passed audit arm covered, for audit_pass.json
  pairing <audit_pass.json> <digest> <ctx mode> <mk mode> [VAR=VAL ...]   on the Mac, before a t arm restarts anything: a passed audit arm on this kit ran the kernels it would serve
"""
import json
import os
import re
import sys

EXACT = "a8c3394af9d28249"
DIGEST = "9a18f802d930f6e7"
HOSTS = ("spark-one", "spark-two", "spark-three", "spark-four", "spark-five", "spark-six", "spark-seven", "spark-eight")
FAILS = []


def report(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def rows(path):
    """JSON rows of a results file; empty when it is missing or unreadable."""
    text = read(path)
    out = []
    for line in (text or "").splitlines():
        if line.strip().startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def rc_ok(o, name):
    return (read(os.path.join(o, name + ".rc")) or "").strip() == "0"


def long_rows(path):
    """{case: [(tok_sha, lp_sha), ...]}, and the number of cases not repeatable within the file."""
    cases, unrepeatable = {}, 0
    for r in rows(path):
        if r.get("summary"):
            unrepeatable += r.get("distinct_tok") != 1 or r.get("distinct_lp") != 1
        elif "tok_sha" in r:
            cases.setdefault(r["case"], []).append((r["tok_sha"], r.get("lp_sha")))
    return cases, unrepeatable


def serving(o, conc, ref_path):
    exact = read(os.path.join(o, "exact.json"))
    try:
        combined = json.loads(exact)["combined"]
    except (TypeError, ValueError, KeyError):
        combined = ""
    report("exact", rc_ok(o, "exact") and combined.startswith(EXACT), f"{combined[:16] or 'no result'} (want {EXACT})")

    g1 = [r for r in rows(os.path.join(o, "g1.jsonl")) if r.get("metric") == "G1_summary"]
    d = str(g1[-1].get("outputs_digest", "")) if g1 else ""
    same = bool(g1) and g1[-1].get("all_reps_identical") is True
    report("g1", rc_ok(o, "g1") and d.startswith(DIGEST) and same,
           f"G1 {g1[-1].get('G1_mean_tok_s') if g1 else 'none'} digest {d[:16] or 'none'} (want {DIGEST}) reps identical {same}")

    g4 = [r for r in rows(os.path.join(o, "g4.jsonl")) if r.get("metric") == "G4_summary"]
    runs_ok = bool(g4) and g4[-1].get("runs") and g4[-1].get("runs_ok") == g4[-1].get("runs")
    report("g4", rc_ok(o, "g4") and bool(runs_ok),
           f"min {g4[-1].get('G4_per_session_min')} mean {g4[-1].get('G4_per_session_mean')} runs {g4[-1].get('runs_ok')}/{g4[-1].get('runs')}" if g4 else "no result")

    ref, _ = long_rows(ref_path)
    got, unrepeatable = long_rows(os.path.join(o, "det.jsonl"))
    bad = sorted(c for c in ref if not got.get(c) or len(got[c]) != 2 or set(got[c]) != set(ref[c]))
    report("long prompts", rc_ok(o, "det") and len(ref) == 9 and not bad and not unrepeatable and set(got) == set(ref),
           f"{len(got)} of {len(ref)} cases, differing or missing {bad}, not repeatable {int(unrepeatable)}")

    if conc:
        m = re.search(r"LOAD32 ok (\d+) /32", read(os.path.join(o, "load32.log")) or "")
        report("load32", rc_ok(o, "load32") and bool(m) and m.group(1) == "32", m.group(0) if m else "no result")

    errs = (read(os.path.join(o, "errors.rank0")) or "").strip()
    report("rank0 errors", errs == "0", errs or "no count")


def fleet(d, ctx, mk, flags):
    env = dict(kv.split("=", 1) for kv in flags)
    tiles = env.get("DSV41_MK_FOLD_TILES", "16")
    wants = []
    if ctx != "0":
        wants.append(f"DSV41_CTXKV_FOLD={ctx}: rows pinned")
    if mk != "0":
        wants.append(f"DSV41_MK_FOLD={mk} tiles {tiles}: self-check passed")
    audit = "audit" in (ctx, mk)
    for h in HOSTS:
        log = read(os.path.join(d, h + ".log"))
        why = []
        if log is None or "LOG_READ" not in log or "SSH_FAILED" in log or "LOG_MISSING" in log:
            why.append("rank log not read")
        log = log or ""
        for want in wants:
            if want not in log:
                why.append(f"no line '{want}': this rank measured stock")
        if not wants and re.search(r"rows pinned|self-check passed", log):
            why.append("fold line in a flags-off arm")
        for bad in ("disabled, stock", "kept on the stock path", "prebuild failed", "boundary audit MISMATCH", "report failed"):
            if bad in log:
                why.append(f"log says '{bad}'")
        m = re.search(r"kept stock (\[[^\]]*\])", log)
        if m and m.group(1) != "[]":
            why.append(f"kept stock {m.group(1)}")

        text = (read(os.path.join(d, h + ".audit")) or "").strip()
        if audit:
            try:
                a = json.loads(text)
            except ValueError:
                a = None
            if not a:
                why.append("no audit report")
            else:
                if a.get("status") != "OK":
                    why.append(f"audit status {a.get('status')}")
                for k in ("ctx_mismatching_words", "mk_pair_mismatching_words", "mk_token_mismatches", "mk_embed_mismatching_words"):
                    if a.get(k) != 0:
                        why.append(f"{k} {a.get(k)}")
                if ctx == "audit" and not a.get("ctx_calls"):
                    why.append("ctx_calls 0")
                if mk == "audit" and (a.get("mk_positions") or 0) < 10000:
                    why.append(f"mk_positions {a.get('mk_positions')} under 10000")
                print(h, {k: v for k, v in a.items() if k.startswith(("ctx_", "mk_"))}, "step", a.get("step"), flush=True)
        elif text:
            why.append("audit report in a non-audit arm")
        report(h, not why, "; ".join(why))


def shape(flags):
    """The flags that pick which kernels run, with the default dsv41_boundary.py applies when one is unset."""
    env = dict(kv.split("=", 1) for kv in flags)
    return env.get("DSV41_MK_FOLD_TILES", "16"), env.get("DSV41_CTXKV_TACTIC_MAP", "")


def record(digest, ctx, mk, flags):
    tiles, tactic_map = shape(flags)
    print(json.dumps({"digest": digest, "ctx": ctx, "mk": mk, "tiles": tiles, "tactic_map": tactic_map}))
    sys.exit(0)


def pairing(path, digest, ctx, mk, flags):
    tiles, tactic_map = shape(flags)
    try:
        a = json.loads(read(path) or "")
    except ValueError:
        a = None
    if not isinstance(a, dict):
        report("audit arm passed", False, "no passing audit arm recorded: run it first, with the kernels this arm will serve")
        return
    report("audit arm on this kit", a.get("digest") == digest, f"audit {a.get('digest')}, kit {digest}")
    if ctx != "0":
        report("context KV audited", a.get("ctx") == "audit", f"audit arm had DSV41_CTXKV_FOLD={a.get('ctx')}")
        # The audit shadow is always merge, so nothing covers a split arm.
        report("context KV shape audited", ctx != "split", "split has no audit path: it never saw real rows")
        report("tactic map same as audit", a.get("tactic_map") == tactic_map, f"audit '{a.get('tactic_map')}', arm '{tactic_map}'")
    if mk != "0":
        report("Markov audited", a.get("mk") == "audit", f"audit arm had DSV41_MK_FOLD={a.get('mk')}")
        report("tiles same as audit", a.get("tiles") == tiles, f"audit {a.get('tiles')}, arm {tiles}")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "serving" and len(sys.argv) == 5:
        serving(sys.argv[2], sys.argv[3] == "1", sys.argv[4])
    elif what == "fleet" and len(sys.argv) >= 5:
        fleet(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif what == "record" and len(sys.argv) >= 5:
        record(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif what == "pairing" and len(sys.argv) >= 6:
        pairing(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:])
    else:
        raise SystemExit(__doc__)
    print(f"arm_check {what}: {'FAIL ' + ', '.join(FAILS) if FAILS else 'PASS'}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
