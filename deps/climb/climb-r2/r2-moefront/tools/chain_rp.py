#!/usr/bin/env python3
"""Per-layer MoE front chain from an nsys sqlite export, control (stock 3-kernel chain) or r2-moefront arms.

    python3 chain_rp.py trace.sqlite=label [...]

For every target FC1 (sm120_fp8_fp4_gemm_1d1d_impl<0, 576, ...>) it walks back on the same stream to the router
splitKreduce and reports (mean / p10 / p50 / p90 / min, us; per-step sum in ms over the window's 60 steps):
  chain       splitKreduce end -> FC1 start (the critical-path segment the levers shorten)
  topk_post   splitKreduce end -> _dsv4_topk_kernel end
  control:    quant / plan / scatter durations and the 4 gaps (t-q, q-p, p-s, s-FC1)
  fused:      rp (= _dsv41_rp_kernel duration), g_trp (top-k end -> rp start; < 0 = PDL overlap),
              g_rpf (rp end -> FC1 start; < 0 = FC1 launched early, DSV41_MOE_FC1_FRONT=pdl/pf*)
  fc1         FC1 duration (pf arms: look for a lower mean / p10 here)
  slack       routed _gated_gather end - last shared-expert DeviceGemmMxfp8 end before the next ar_kernel
              (> 0: the routed chain is still the last arrival, so the saving is on the critical path)
Prints which layout each FC1 had, so a mixed window (prefill M > 96 falls back to the stock chain) is visible.
"""
import re
import sqlite3
import statistics as stt
import sys

FC1 = re.compile(r"sm120_fp8_fp4_gemm_1d1d_impl<\(unsigned int\)0, \(unsigned int\)576")
PATS = [("scatter", "_dsv41_fp_scatter"), ("plan", "_dsv41_fp_plan"), ("quant", "per_token_group_quant_8bit_register"),
        ("rp", "_dsv41_rp_kernel"), ("topk", "_dsv4_topk"), ("splitk", "splitKreduce")]


def q(v, f):
    v = sorted(v)
    return v[int(f * (len(v) - 1))]


def main():
    for arg in sys.argv[1:]:
        p, lab = arg.split("=")
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        S = dict(c.execute("select id,value from StringIds"))
        k = [(a, b, st, S.get(d) or S.get(s) or "") for a, b, st, d, s in c.execute(
            "select start,end,streamId,demangledName,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
        res = {}
        kinds = {"control": 0, "fused": 0, "other": 0}

        def add(key, v):
            res.setdefault(key, []).append(v)
        for i, r in enumerate(k):
            if not FC1.search(r[3]):
                continue
            st = r[2]
            seq = {}
            j = i - 1
            while j > 0 and i - j < 40:
                a, b, s2, n = k[j]
                if s2 == st:
                    for key, pat in PATS:
                        if key not in seq and pat in n:
                            seq[key] = (a, b)
                    if "splitk" in seq:
                        break
                j -= 1
            if "splitk" not in seq or "topk" not in seq:
                kinds["other"] += 1
                continue
            sk = seq["splitk"][1]
            add("chain", r[0] - sk)
            add("topk_post", seq["topk"][1] - sk)
            add("fc1", r[1] - r[0])
            if "rp" in seq and seq["rp"][0] > seq["topk"][0]:
                kinds["fused"] += 1
                add("rp", seq["rp"][1] - seq["rp"][0])
                add("g_trp", seq["rp"][0] - seq["topk"][1])
                add("g_rpf", r[0] - seq["rp"][1])
            elif all(x in seq for x in ("quant", "plan", "scatter")):
                kinds["control"] += 1
                for key in ("quant", "plan", "scatter"):
                    add(key, seq[key][1] - seq[key][0])
                add("g_tq", seq["quant"][0] - seq["topk"][1])
                add("g_qp", seq["plan"][0] - seq["quant"][1])
                add("g_ps", seq["scatter"][0] - seq["plan"][1])
                add("g_sf", r[0] - seq["scatter"][1])
            else:
                kinds["other"] += 1
            m, gend, send = i, None, None
            while m < len(k) and m - i < 60:
                a, b, s2, n = k[m]
                if "ar_kernel" in n and a > r[0]:
                    break
                if s2 == st and "_gated_gather" in n:
                    gend = b
                if s2 != st and "DeviceGemmMxfp8" in n:
                    send = b if send is None else max(send, b)
                m += 1
            if gend and send:
                add("slack", gend - send)
        n = len(res.get("chain", []))
        print(f"{lab}: target FC1 layers {n} ({n / 60:.1f}/step); layouts {kinds}")
        for key, v in res.items():
            print(f"   {key:10s} mean {stt.mean(v) / 1e3:7.2f} us  p10 {q(v, .1) / 1e3:7.2f}  p50 {q(v, .5) / 1e3:7.2f}"
                  f"  p90 {q(v, .9) / 1e3:7.2f}  min {min(v) / 1e3:7.2f}  per-step-sum {sum(v) / 60 / 1e6:6.3f} ms")


if __name__ == "__main__":
    main()
