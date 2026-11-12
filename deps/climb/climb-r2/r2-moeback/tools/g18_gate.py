#!/usr/bin/env python3
"""G18 device gate (run by the lead on a STOPPED node, GPU free, from a copy of the kit that holds the vetted
build/athena_tp8_fc2_packed_graph_pdl_sm121.so):

    CUDA_VISIBLE_DEVICES=0 ~/venvs/vllm032/bin/python tools/g18_gate.py --kit . [--reps 200]

Runs, in order, and stops at the first failure:
  1. device_test.py --part g18 --reps >=200   (eager bitwise + >=200 CUDA-graph replays per case of the real
     SiLU-mul-quant with the early PDL trigger -> PDL FC2 with workspace2 aliasing, + timing)
  2. compute-sanitizer --tool racecheck --error-exitcode 99  device_test.py --part g18 --quick
  3. compute-sanitizer --tool synccheck --error-exitcode 99  device_test.py --part g18 --quick
and writes build/DEVICE-G18.json {pdl_so_sha256, pass, *_rc, reps, graph_cases_ok, host, ...} plus the three
logs (build/DEVICE-G18-*.log). install.py installs the PDL binary only when this stamp passed for the exact
sha recorded in KIT.json pdl_binary_sha256 (copy build/DEVICE-G18*.{json,log} back into the master kit).
Note: racecheck covers shared-memory hazards inside the kernels and synccheck barrier misuse; neither sees a
cross-kernel PDL race, which only the bitwise graph-replay check (step 1) covers.
Expected runtime: step 1 ~3-6 min, steps 2-3 ~10-30 min each (sanitizers are slow).
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import socket
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--kit", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--sanitizer", default="/usr/local/cuda/bin/compute-sanitizer")
a = ap.parse_args()
kit = os.path.abspath(a.kit)
so = os.path.join(kit, "build/athena_tp8_fc2_packed_graph_pdl_sm121.so")
stamp_p = os.path.join(kit, "build/DEVICE-G18.json")
want = json.load(open(os.path.join(kit, "KIT.json"))).get("pdl_binary_sha256")
got = hashlib.sha256(open(so, "rb").read()).hexdigest() if os.path.exists(so) else None
if not want or got != want:
    sys.exit(f"g18_gate: {so} sha {got} does not match KIT.json pdl_binary_sha256 {want} (run install.py vet-pdl "
             "first); refusing")
if os.path.exists(stamp_p):
    os.remove(stamp_p)                    # never leave an old stamp behind a new run
reps = max(200, a.reps)
py, dt = sys.executable, os.path.join(kit, "device_test.py")
steps = [
    ("device_test", [py, dt, "--kit", kit, "--part", "g18", "--reps", str(reps)]),
    ("racecheck", [a.sanitizer, "--tool", "racecheck", "--error-exitcode", "99", py, dt, "--kit", kit,
                   "--part", "g18", "--quick"]),
    ("synccheck", [a.sanitizer, "--tool", "synccheck", "--error-exitcode", "99", py, dt, "--kit", kit,
                   "--part", "g18", "--quick"]),
]
stamp = {"pdl_so_sha256": got, "reps": reps, "host": socket.gethostname(),
         "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "pass": False}
for name, cmd in steps:
    log = os.path.join(kit, f"build/DEVICE-G18-{name}.log")
    print(f"== {name}: {' '.join(cmd)}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(r.stdout)
    sys.stdout.write(r.stdout[-4000:])
    stamp[f"{name}_rc"] = r.returncode
    stamp[f"{name}_log_sha256"] = hashlib.sha256(open(log, "rb").read()).hexdigest()
    if name == "device_test":
        ok_graph = re.findall(r"^OK\s+g18 graph .*\((\d+) replays\)", r.stdout, re.M)
        stamp["graph_cases_ok"] = sum(int(n) >= 200 for n in ok_graph)
        stamp["fail_lines"] = len(re.findall(r"^FAIL ", r.stdout, re.M))
        if stamp["fail_lines"]:
            stamp[f"{name}_rc"] = stamp[f"{name}_rc"] or 1
    if stamp[f"{name}_rc"] != 0:
        break
stamp["pass"] = all(stamp.get(f"{n}_rc") == 0 for n, _ in steps) and stamp.get("graph_cases_ok", 0) >= 8
with open(stamp_p, "w") as f:
    json.dump(stamp, f, indent=1)
    f.write("\n")
print(json.dumps(stamp, indent=1))
sys.exit(0 if stamp["pass"] else 1)
