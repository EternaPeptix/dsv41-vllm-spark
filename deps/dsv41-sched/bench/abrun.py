#!/usr/bin/env python3
"""abrun.py <outdir> <plan.json> : in-instance A/B driver (run on spark-one).
plan = {"arms": {name: ctl-dict}, "seq": [[arm, kind, args...], ...]}
kind: starve128 | starve1m | solo128 | solo1m.  Writes ctl file, waits until the scheduler's
stats file reports the params, then runs sbench with label=arm into <outdir>/<kind>.jsonl."""
import json, os, subprocess, sys, time
out, plan = sys.argv[1], json.load(open(sys.argv[2]))
os.makedirs(out, exist_ok=True)
CTL = "/dev/shm/dsv41-sched.json"; ST = "/dev/shm/dsv41-sched-stats.json"
SB = os.path.expanduser("~/dsv41-sched/bench/sbench.py")
KIND = {"solo500": ["solo", "--long", "500k"],
        "starve128": ["starve", "--long", "128k", "--offsets", "2,5,8,11,14,17"],
        "starve1m": ["starve", "--long", "975k", "--offsets", "10,45,80,115,150,185"],
        "solo128": ["solo", "--long", "128k"], "solo1m": ["solo", "--long", "975k"]}
for step in plan["seq"]:
    arm, kind = step[0], step[1]
    ctl = plan["arms"][arm]
    json.dump(ctl, open(CTL + ".tmp", "w")); os.replace(CTL + ".tmp", CTL)
    # the scheduler reloads the ctl file on schedule() (<= 0.5 s); poke, then confirm via the rank0 log
    # ("dsv41_sched params now {...}"), since the stats file is not rewritten when enable=0.
    want = None
    for _ in range(30):
        subprocess.run(["curl", "-s", "-o", "/dev/null", "127.0.0.1:8000/v1/completions", "-H",
                        "Content-Type: application/json", "-d", '{"model":"dsv41","prompt":"hi","max_tokens":2}'])
        time.sleep(1)
        lines = [l for l in open(os.path.expanduser("~/dsv41-logs/rank0.log"), errors="replace")
                 if "dsv41_sched params now" in l or "dsv41_sched armed" in l]
        if lines:
            import ast
            d = ast.literal_eval(lines[-1][lines[-1].index("{"):lines[-1].index("}") + 1])
            if all(d.get(k) == v for k, v in ctl.items()):
                want = d; break
    if want is None:
        sys.exit(f"ctl not picked up for {arm}: {ctl}")
    cmd = ["python3", SB] + KIND[kind] + ["--label", arm, "--out", f"{out}/{kind}.jsonl"] + step[2:]
    t1 = time.time()
    st0 = open(ST).read() if os.path.exists(ST) else ""
    r = subprocess.run(cmd, capture_output=True, text=True)
    time.sleep(2.5)
    st1 = open(ST).read() if os.path.exists(ST) else ""
    temps = subprocess.run("for h in one two three four five six seven eight; do ssh -n -o ConnectTimeout=3 spark-$h "
                           "'cat /sys/class/thermal/thermal_zone0/temp' 2>/dev/null | head -c 3; echo -n ' '; done",
                           shell=True, capture_output=True, text=True).stdout
    with open(f"{out}/stats.jsonl", "a") as fh:
        fh.write(json.dumps({"arm": arm, "kind": kind, "t": t1, "before": st0, "after": st1, "temps": temps}) + "\n")
    with open(f"{out}/log.txt", "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {arm} {kind} {step[2:]} rc={r.returncode} {time.time()-t1:.0f}s\n{r.stdout[-2000:]}{r.stderr[-2000:]}\n")
open(f"{out}/DONE", "w").write("done\n")
