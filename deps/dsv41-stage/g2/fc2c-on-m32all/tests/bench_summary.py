#!/usr/bin/env python3
"""Summarize bench_g2.py JSON lines (CPU only): python3 -I bench_summary.py bench-<TS>.rank<R>.jsonl [...]"""
import json
import sys

for fn in sys.argv[1:]:
    rows, proj, setup = [], None, {}
    for line in open(fn):
        d = json.loads(line)
        if d.get("event") == "timing":
            rows.append(d)
        elif d.get("event") == "projection":
            proj = d
        elif d.get("event") == "setup":
            setup = d
    print(f"== {fn}  host={setup.get('host')} rank={setup.get('tp_rank')} layers={setup.get('layers')} "
          f"tails={setup.get('tail_bytes_total')} natives={setup.get('natives')}")
    print("  M  set      exp   moe live us   fc2c us (x, ms/step)        fc2c_all us (x, ms/step)     "
          "fc2-only x fc2c / all   fc2 GB/s live/fc2c   bitwise")
    for d in rows:
        mo, f2 = d.get("moe", {}), d.get("fc2", {})
        if not mo:
            continue

        def cell(n):
            r = mo.get(n)
            return f"{r['layer_us']:7.1f} ({r['speedup']:.3f}, {r['delta_ms_step']:+.2f})" if r else " " * 24
        f2s = "/".join(f"{f2[n]['speedup']:.3f}" if n in f2 else "-" for n in ("fc2c", "fc2c_all"))
        gb = "/".join(str(f2[n].get("weight_gbps", "-")) if n in f2 else "-" for n in ("live", "fc2c"))
        bw = all(mo[n].get("bitwise_vs_live", True) for n in mo if n != "live")
        print(f"  {d['m']:2d} {d['set']:8s} {d['distinct_mean']:5.1f}  {mo['live']['layer_us']:8.1f}   "
              f"{cell('fc2c'):26s} {cell('fc2c_all'):26s}  {f2s:21s} {gb:20s} {'OK' if bw else 'MISMATCH'}")
    if proj:
        print("  projection (bench delta x40; serving x0.96 1u / x0.90 4u; % at the step time given):")
        for r in proj["rows"]:
            print(f"    M={r['m']:2d} {r['set']:8s} serving ms/step {r['serving_delta_ms_step']}  "
                  f"tok/s % {r['tok_s_pct_at_step']} (step {r['step_ms']} ms)")
