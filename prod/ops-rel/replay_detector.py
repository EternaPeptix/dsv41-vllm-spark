"""Offline replay of saved rank logs through the dsv41-relaunch.sh detector (rel-2 test).

For each log: when the supervisor would flag it (first DSpark capture line + STALL s without
"capturing greedy draft graphs", or a failure marker), versus when the run actually died
(rdma_ar trap / EngineCore failure) or came up (Application startup complete).
Usage: python3 -I replay_detector.py STALL log...
"""
import re
import sys
from datetime import datetime

STALL = float(sys.argv[1])
TS = re.compile(r"(?:INFO|WARNING|ERROR|DEBUG) (\d\d-\d\d \d\d:\d\d:\d\d)")
MARK = {
    "dspark": "Capturing model for DSpark",
    "greedy": "capturing greedy draft graphs",
    "start": "Application startup complete",
    "f_rdma": "timeout waiting for",
    "f_eng": ("EngineCore failed to start", "Engine core initialization failed", "died unexpectedly"),
    "f_wd": re.compile(r"Timeout \(\d+:\d\d:\d\d\)!"),
}


def ts_of(s):
    return datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S")


def scan(path):
    txt = open(path, errors="replace").read()
    first = {}
    last_ts = None
    # split on newlines and carriage returns (tqdm) so markers inherit the latest timestamp seen
    for line in re.split(r"[\r\n]", txt):
        m = TS.search(line)
        if m:
            last_ts = ts_of(m.group(1))
        for k, pat in MARK.items():
            if k in first:
                continue
            hit = (pat.search(line) is not None) if hasattr(pat, "search") else (
                any(p in line for p in pat) if isinstance(pat, tuple) else pat in line)
            if hit:
                first[k] = last_ts
    return first


def verdict(first):
    """What dsv41-relaunch.sh would do, and when (seconds after the DSpark capture line)."""
    d = first.get("dspark")
    events = []
    for k in ("f_rdma", "f_wd", "f_eng"):
        if k in first and first[k] is not None:
            events.append((first[k], k))
    if d is not None:
        g = first.get("greedy")
        if g is None or (g - d).total_seconds() > STALL:
            events.append((d.__class__.fromtimestamp(d.timestamp() + STALL), "dspark_stall"))
    if not events:
        return "healthy" if "start" in first else "no-verdict", None
    t, k = min(events)
    return k, t


rows = []
for p in sys.argv[2:]:
    f = scan(p)
    v, t = verdict(f)
    died = [f[k] for k in ("f_rdma", "f_eng") if f.get(k)]
    died_t = min(died) if died else None
    d = f.get("dspark")
    save = (died_t - t).total_seconds() if (died_t and t) else None
    rows.append((p, v, d, t, died_t, save, f.get("start")))
    print(f"{p.split('scratchpad/')[-1]:70s} verdict={v:13s} dspark={d} flag_at={t} died_at={died_t} "
          f"saved_s={save} startup={f.get('start')}")
