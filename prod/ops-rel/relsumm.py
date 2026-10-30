"""rel-2 startup summary of one rank log (runs on spark-one): phase timestamps and durations as JSON.

usage: python3 -I relsumm.py <rank0.log> [unit_start_epoch]
"""
import calendar
import hashlib
import json
import re
import sys
from datetime import datetime

path = sys.argv[1]
t_unit = float(sys.argv[2]) if len(sys.argv) > 2 else None
TS = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)")


def ts(line):
    m = TS.search(line)
    if not m:
        return None
    mo, d, h, mi, s = map(int, m.groups())
    return float(calendar.timegm((2026, mo, d, h, mi, s, 0, 0, 0)))  # rank logs are UTC


marks = {
    "first": None, "target_loaded": None, "draft_loaded": None, "model_loaded": None,
    "dspark_start": None, "greedy": None, "av_start": None, "av_end": None,
    "capture_done": None, "startup": None,
}
out = {"loads_s": [], "draft_shards_skipped": None, "graph_capture_s": None, "init_engine_s": None,
       "av_mode": None, "av_pin_reason": None, "calib_sha": None, "xstream": None, "fingerprint": None,
       "watchdog": [], "rdma_timeouts": 0, "wd_timeouts": 0, "cuda_errors": 0}
last_t = None
for line in open(path, errors="replace"):
    t = ts(line)
    if t is None:
        t = last_t  # e.g. uvicorn's untimed "Application startup complete."
    else:
        last_t = t
    if t is not None and marks["first"] is None:
        marks["first"] = t
    if "Loading weights took" in line:
        m = re.search(r"Loading weights took ([0-9.]+) seconds", line)
        if m:
            out["loads_s"].append(float(m.group(1)))
            key = "target_loaded" if len(out["loads_s"]) == 1 else "draft_loaded"
            marks[key] = marks[key] or t
    elif "safetensors shard(s) holding no weights" in line:
        m = re.search(r"Skipped (\d+)/(\d+)", line)
        out["draft_shards_skipped"] = m and f"{m.group(1)}/{m.group(2)}"
    elif "Model loading took" in line:
        marks["model_loaded"] = marks["model_loaded"] or t
    elif "Capturing model for DSpark" in line:
        marks["dspark_start"] = marks["dspark_start"] or t
    elif "capturing greedy draft graphs" in line:
        marks["greedy"] = marks["greedy"] or t
    elif "before_adaptive_calibration" in line:
        marks["av_start"] = marks["av_start"] or t
    elif "after_adaptive_calibration" in line:
        marks["av_end"] = marks["av_end"] or t
    elif "DSV41 AV profile skipped" in line:
        out["av_mode"] = "pinned-skip"
        out["av_pin_reason"] = line.split("pinned):", 1)[-1].strip()[:200]
    elif "DSV41 AV profile: pin miss" in line:
        out["av_mode"] = "pin-miss"
        out["av_pin_reason"] = line.split("pin miss", 1)[-1].strip()[:200]
    elif "Adaptive verification cost profile:" in line and out["av_mode"] is None:
        out["av_mode"] = "profiled"
    elif "DSV41 adaptive-verification calibration" in line:
        body = line.split("calibration", 1)[1]
        body = re.sub(r"^.*?\(json=", "(json=", body).strip()
        out["calib_sha"] = hashlib.sha256(body.encode()).hexdigest()[:12]
        m = re.search(r"json=(\S+)", body)
        out["calib_json"] = m and m.group(1)
    elif "Graph capturing finished in" in line:
        marks["capture_done"] = marks["capture_done"] or t
        m = re.search(r"finished in ([0-9.]+) secs", line)
        out["graph_capture_s"] = m and float(m.group(1))
    elif "init engine (profile, create kv cache, warmup model) took" in line:
        m = re.search(r"took ([0-9.]+) s", line)
        out["init_engine_s"] = m and float(m.group(1))
    elif "Application startup complete" in line:
        marks["startup"] = marks["startup"] or t
    elif "rdma_ar xstream (" in line:
        m = re.search(r"switches=(\d+) busy=(\d+) waits=(\d+) unknown=(\d+)", line)
        out["xstream"] = m and dict(zip(("switch", "busy", "wait", "unknown"), map(int, m.groups())))
    elif "rdma_ar fingerprint" in line:
        out["fingerprint"] = line.split("rdma_ar fingerprint", 1)[1].strip()[:80]
    elif "DSV41 capture watchdog: disarmed" in line:
        m = re.search(r"phase=(\S+) \(([0-9.]+) s\)", line)
        if m:
            out["watchdog"].append([m.group(1), float(m.group(2))])
    if "timeout waiting for" in line:
        out["rdma_timeouts"] += 1
    if re.search(r"Timeout \(\d+:\d\d:\d\d\)!", line):
        out["wd_timeouts"] += 1
    if "CUDA error" in line:
        out["cuda_errors"] += 1


def d(a, b):
    return None if marks[a] is None or marks[b] is None else round(marks[b] - marks[a], 1)


out["t_unit_to_first_log_s"] = None if (t_unit is None or marks["first"] is None) else round(marks["first"] - t_unit, 1)
out["t_unit_to_startup_s"] = None if (t_unit is None or marks["startup"] is None) else round(marks["startup"] - t_unit, 1)
out["first_to_startup_s"] = d("first", "startup")
out["draft_load_phase_s"] = d("target_loaded", "draft_loaded")
out["dspark_to_greedy_s"] = d("dspark_start", "greedy")
out["av_phase_s"] = d("av_start", "av_end")
out["startup_utc"] = marks["startup"] and datetime(1970, 1, 1).fromtimestamp(marks["startup"], tz=__import__("datetime").timezone.utc).strftime("%H:%M:%S")
print(json.dumps(out))
