#!/usr/bin/env python3
"""verdict_l2win.py <rank0.json>: GO / HOLD / KILL per mode from bench_l2win.py output. Pure; the decision functions are CPU-tested.

Thresholds (the same table is in WINDOW.md):
  BITWISE   any mismatch, any rank: KILL everything that shares the mode's flag. Not a noise question.
  RATE      a paced mode needs achieved >= 0.95 x requested on the windows it paces (phase C, table and chunk). Else that mode is KILL.
  LATE      per-layer time vs live (phase W): no M in 1,2,4,6,8 may be slower by more than 1.5 us; no M in 12,16,24 by more than 3 us.
  GAIN      saved_ms = 40 x weighted (live - mode) us/layer over M in 1,2,4,6,8, weights below. saved_ms >= 0.12: GO to a pinned pair.
            0.06 <= saved_ms < 0.12: HOLD (rider only, in a bundle). Below 0.06: KILL.
Weights come from the rank-0 traces of 2026-11-02: two-shot in about 5 of 6 of G1's AR calls (all of code, half of prose), and the grid split
of the two-shot class. Math has no trace, assumed like code. M per grid is inferred, not read. Projection: G1% = saved_ms x 4.2 x realisation
0.45..0.85 (followup-2; MHC=0 realised about 0.66). A GO here is not a promotion: the bar is a pinned pair.
"""
import json
import sys

WEIGHTS = {1: 0.0, 2: 0.17, 4: 0.33, 6: 0.40, 8: 0.10}
LATE_SMALL_US, LATE_BIG_US = 1.5, 3.0
GO_MS, HOLD_MS = 0.12, 0.06
RATE_FRAC = 0.95
P = "DSV41_L2_PREFETCH_"
ENV_ONLY = {"p"}


def mode_env(mode: str, parse) -> tuple:
    """(env string, needs the kit's patched module)."""
    c = parse(mode)
    k = c["kind"]
    if k == "d":
        return f"{P}BURST_DELAY_US={c['delay_us']:g}", True
    if k == "p":
        env = [f"{P}PACE_GBPS={c['gbps']:g}", f"{P}PACE_KINDS=ar"]
        if c["chunk"] != 16384:
            env.append(f"{P}PACE_CHUNK={c['chunk']}")
        if c["delay_us"]:
            env.append(f"{P}PACE_DELAY_US={c['delay_us']:g}")
        if c["cap_us"] != 150:
            env.append(f"{P}PACE_MAX_US={c['cap_us']:g}")
        return " ".join(env), False
    if k == "s":
        env = [f"{P}PACE_GBPS={c['gbps']:g}", f"{P}PACE_KINDS=ar", f"{P}PACE_MAX_WIN_KB=2048"]
        if c["chunk"] != 16384:
            env.append(f"{P}PACE_CHUNK={c['chunk']}")
        if c["big_gbps"]:
            env.append(f"{P}PACE_BIG_GBPS={c['big_gbps']:g}")
        if c["big_delay_us"] is not None:
            env.append(f"{P}BURST_DELAY_US={c['big_delay_us']:g}")
        return " ".join(env), True
    return "", False


def rate_ok(c, table_rows: list) -> tuple:
    """Every (table, rate) the mode paces must reach RATE_FRAC x rate at its chunk. Rows missing = not proven = False."""
    k = c["kind"]
    if k not in ("p", "s"):
        return True, ""
    need = []
    if k == "p":
        need = [("short", c["gbps"]), ("gate", c["gbps"]), ("big", c["gbps"])]
    else:
        need = [("short", c["gbps"])]
        if c["big_gbps"]:
            need += [("gate", c["big_gbps"]), ("big", c["big_gbps"])]
    idx = {(r["table"], r["chunk_kib"], r["req_gbps"]): r for r in table_rows}
    chunk = c["chunk"] // 1024
    for tb, rate in need:
        r = idx.get((tb, chunk, int(rate)))
        if r is None:
            return False, f"no phase C row {tb} c{chunk} r{rate:g}"
        if r["full"] != r["n"]:
            return False, f"{tb} c{chunk} r{rate:g}: issued all bytes in {r['full']}/{r['n']}"
        if r["achieved_gbps"] < RATE_FRAC * rate:
            return False, f"{tb} c{chunk} r{rate:g}: achieved {r['achieved_gbps']:g}"
    return True, ""


def decide(mode: str, c, us: dict, ceil_rows: list, mismatches: int) -> dict:
    """us[M] = us/layer of (mode - live). Returns the verdict for one mode."""
    if mismatches:
        return dict(mode=mode, verdict="KILL", why=f"{mismatches} bitwise mismatches", saved_ms=None)
    ok, why = rate_ok(c, ceil_rows)
    if not ok:
        return dict(mode=mode, verdict="KILL", why="rate: " + why, saved_ms=None)
    missing = [m for m in WEIGHTS if WEIGHTS[m] and m not in us]
    if missing:
        return dict(mode=mode, verdict="HOLD", why=f"phase W missing M={missing}", saved_ms=None)
    late = [m for m, v in us.items() if (m <= 8 and v > LATE_SMALL_US) or (m > 8 and v > LATE_BIG_US)]
    saved = sum(WEIGHTS[m] * -us[m] for m in WEIGHTS if WEIGHTS[m]) * 40 / 1000.0
    if late:
        return dict(mode=mode, verdict="KILL", why=f"late bytes: slower at M={sorted(late)}", saved_ms=round(saved, 3))
    v = "GO" if saved >= GO_MS else ("HOLD" if saved >= HOLD_MS else "KILL")
    return dict(mode=mode, verdict=v, why="", saved_ms=round(saved, 3),
                g1_pct=(round(saved * 4.2 * 0.45, 2), round(saved * 4.2 * 0.85, 2)))


def main() -> int:
    sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/bench")
    from bench_l2win import parse_mode  # noqa: E402 - needs torch at import; run with the venv python
    r = json.load(open(sys.argv[1]))
    allf = r.get("failures_all_ranks", [])
    print("failures per rank:", allf)
    modes = [m for m in r["args"]["modes"].split(",") if m not in ("none", "live")]
    forced = [m for m in r["args"].get("forced", "").split(",") if m]
    w = {}
    for row in r.get("W", []):
        w.setdefault(row["mode"], {})[row["M"]] = row["vs_live"]
    mism = {}
    for per in r.get("F", {}).get("mismatches", {}).values():
        for k, v in per.items():
            mism[k] = mism.get(k, 0) + v
    print("forced-overlap modes (co-residency):", {k: mism.get(k, "not run") for k in forced})
    if any(allf) or any(mism.get(k) for k in forced + ["none", "live"]):
        print("VERDICT: KILL, a bitwise check failed. See F.first in the json of the failing rank. Do not run a pinned pair.")
    out = []
    for m in modes:
        d = decide(m, parse_mode(m), w.get(m, {}), r.get("C", []), mism.get(m, 0) + (1 if any(allf) else 0))
        env, patched = mode_env(m, parse_mode)
        d.update(env=env, needs_patch=patched)
        out.append(d)
        print(f"{d['verdict']:5s} {m:16s} saved_ms {d['saved_ms']} g1% {d.get('g1_pct')} {d['why']}\n      env: {env}"
              f"{'  [kit module needed]' if patched else '  [env-only]'}")
    go = sorted((d for d in out if d["verdict"] == "GO"), key=lambda d: (d["needs_patch"], -d["saved_ms"]))
    print("BEST:", (go[0]["mode"] + " -> " + go[0]["env"]) if go else "none")
    print("Env-only first among GO. If the best is a p* mode at gate rate, check co-residency: F forced modes must be 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
