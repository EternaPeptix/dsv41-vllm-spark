#!/usr/bin/env python3
"""Project m32bw_bench.py results to serving decode ms/step at 1 and 4 users (CPU only).

  python3 project.py bench-r0.jsonl [bench-r7.jsonl ...]

Per arm: bench saving per layer (us, vs live, same graph/trials) x 40 target layers, then scaled to serving with the
ratio serving/bench routed-GEMM time, because the variants remove bytes and the serving kernels are shorter (L2
prefetch, narrower average verify width):
  1 user : bench M=6 recorded windows (C1 coding verify width); serving FC1+FC2 = 172.6 + 104.8 us/layer at M=6
           (k1 trace new1, pre-D05) -> scale = 277.4 / bench live (FC1+FC2) at M=6 from the profile (or 0.96);
  4 users: bench c4rec M=24 (real 4-user batches; else the u6x4 sets); serving FC1+FC2 = 472 + 259 us/layer (k1
           trace, 4u) -> scale = 731 / bench live (FC1+FC2) at M=24 (or 0.90 if no profile).
Then step times: 1u C1 coding 32.1 ms (AL 5.23) and prose 27.6 ms; 4u Codex profile step 50.4 ms.
"""
import json
import statistics
import sys

SERV = {1: 277.4, 4: 731.0}
DEFAULT_SCALE = {1: 0.96, 4: 0.90}
STEP = {"1u_code": 32.1, "1u_prose": 27.6, "4u": 50.4}

timing, prof = {}, {}
for fn in sys.argv[1:]:
    for line in open(fn):
        d = json.loads(line)
        if d.get("event") == "timing":
            for arm, r in d["results"].items():
                timing.setdefault((d["m"], d["set"]), {}).setdefault(arm, []).append(r["layer_us_med"])
        elif d.get("event") == "profile":
            k = {x["k"]: x["us_per_layer"] for x in d["kernels"]}
            prof.setdefault((d["m"], d["arm"]), []).append(k.get("FC1", 0) + k.get("FC2", 0))


def saving(m, prefer):
    keys = [k for k in timing if k[0] == m and (prefer is None or k[1] == prefer)]
    if not keys and prefer is not None:
        keys = [k for k in timing if k[0] == m and k[1].startswith("u6x")]
    out = {}
    for k in keys:
        live = statistics.mean(timing[k]["live"])
        for arm, v in timing[k].items():
            out.setdefault(arm, []).append(live - statistics.mean(v))
    return {arm: statistics.mean(v) for arm, v in out.items()}, keys


for users, m, prefer in ((1, 6, None), (1, 1, None), (4, 24, "c4rec"), (4, 12, "c4rec")):
    sv, keys = saving(m, prefer)
    if not sv:
        continue
    live_gemm = prof.get((m, "live"))
    scale = SERV[users] / statistics.mean(live_gemm) if live_gemm else DEFAULT_SCALE[users]
    print(f"== {users} user(s), bench M={m} sets={sorted(set(k[1] for k in keys))} scale={scale:.2f}")
    for arm, us in sorted(sv.items(), key=lambda kv: -kv[1]):
        bench_ms = us * 40 / 1000
        serv_ms = bench_ms * scale
        if users == 1:
            extra = "  C1 code %.2f -> %.2f ms (%+.1f%% tok/s)  prose ~%+.1f%%" % (
                STEP["1u_code"], STEP["1u_code"] - serv_ms, 100 * (STEP["1u_code"] / (STEP["1u_code"] - serv_ms) - 1),
                100 * (STEP["1u_prose"] / (STEP["1u_prose"] - 0.6 * serv_ms) - 1))
        else:
            extra = "  4u step %.1f -> %.1f ms (%+.1f%% per-session tok/s)" % (
                STEP["4u"], STEP["4u"] - serv_ms, 100 * (STEP["4u"] / (STEP["4u"] - serv_ms) - 1))
        print(f"  {arm:5s} bench {us:+7.1f} us/layer = {bench_ms:+.2f} ms/step; serving ~{serv_ms:+.2f} ms/step{extra}")
