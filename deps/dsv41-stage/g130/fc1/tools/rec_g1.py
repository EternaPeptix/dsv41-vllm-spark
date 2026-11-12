#!/usr/bin/env python3
"""g130 fc1: drive the G1 prompts through a recorder instance and split the dump by type. Run on spark-one; it calls port 8000, so
only inside a window where the instance is the recorder instance (DSV41_MOE_FC1_ROUTE_REC=1). Gives no G1 number.
  python3 -I tools/rec_g1.py <outdir>
Writes <outdir>/routes.bin, splits.json and hist-{code,prose,math}.json for tools/eval_probe.py --hist."""
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path.home() / "dsv41-bench/goal"))
import goal_single as g  # noqa: E402

LOGS = Path.home() / "dsv41-logs"
TRIG = LOGS / "fc1-route-rec.trigger"
KIT = Path(__file__).resolve().parent.parent
base = "http://127.0.0.1:8000"


def dump():
    t0 = time.time()
    TRIG.touch()
    os.utime(TRIG, (t0, t0))
    for _ in range(60):
        time.sleep(2.0)
        js = [p for p in glob.glob(str(LOGS / "fc1-routes.*.json")) if os.path.getmtime(p) >= t0]
        if js:
            p = sorted(js, key=os.path.getmtime)[-1]
            return p, json.load(open(p))["records"]
    raise SystemExit("recorder did not dump: check ~/dsv41-logs/fc1-routes.*.err")


assert g.prompts_sha() == g.PROMPTS_SHA256
p, n = dump()
print("start records", n, flush=True)
splits = {}
for ty in g.TYPES:
    r = g.stream_chat(base, "dsv41", g.PROMPTS[ty], g.MAX_TOKENS)
    time.sleep(3.0)
    p, m = dump()
    splits[ty] = (n, m)
    print(ty, "records", n, m, "steps", (m - n) / 40, "sha", r.get("sha256", "")[:12], flush=True)
    n = m
binp = p[:-5] + ".bin"
shutil.copy(binp, out / "routes.bin")
(out / "splits.json").write_text(json.dumps(splits))
for ty, (a, b) in splits.items():
    subprocess.check_call([sys.executable, "-I", str(KIT / "tools/routes_to_hist.py"), str(out / "routes.bin"), "--from", str(a), "--to", str(b),
                           "--out", str(out / f"hist-{ty}.json")])
print("done", out)
