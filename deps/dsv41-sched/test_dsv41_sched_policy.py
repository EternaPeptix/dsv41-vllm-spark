"""Offline tests for dsv41_sched_policy + a discrete-event sim of the engine loop
(async batch queue of 2 vs hold) under a step-time model T(n) = F + n/rate.
Run: python3 -I test_dsv41_sched_policy.py  (stdlib only)."""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_sched_policy import DEFAULTS, decide_hold, decide_threshold  # noqa: E402

P = dict(DEFAULTS)
fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    fails += not ok
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": got {got!r} want {want!r}"))


LONG = (True, 900_000)
DEC = (False, 0)
# --- decide_threshold
check("lone long prefill -> stock", decide_threshold([LONG], [], P), (0, ""))
check("long + decoder, no short, no dcap -> stock", decide_threshold([LONG, DEC], [], P), (0, ""))
check("long + 300-tok short -> cap long_cap", decide_threshold([LONG], [300], P), (2048, "short"))
check("short of 3000 -> thr rounded up to cover it", decide_threshold([LONG], [3000], P), (3008, "short"))
check("only long waiting (no short) -> stock", decide_threshold([LONG], [120_000], P), (0, ""))
check("long waiting ahead of short -> still short step", decide_threshold([LONG], [120_000, 300], P), (2048, "short"))
check("short beyond scan -> stock", decide_threshold([LONG], [99_999] * 32 + [300], P), (0, ""))
check("long nearly done (rem <= thr) -> stock", decide_threshold([(True, 1500)], [300], P), (0, ""))
check("no prefill running -> stock", decide_threshold([DEC, DEC], [300], P), (0, ""))
q = dict(P, decode_cap=8192)
check("dcap: long + decoder", decide_threshold([LONG, DEC], [], q), (8192, "dcap"))
check("dcap: lone long -> stock", decide_threshold([LONG], [], q), (0, ""))
check("dcap + short -> min", decide_threshold([LONG, DEC], [300], q), (2048, "short+dcap"))
q2 = dict(P, decode_cap=1024)
check("dcap below short size keeps short whole", decide_threshold([LONG, DEC], [3000], q2), (3008, "short+dcap"))
check("disabled -> stock", decide_threshold([LONG], [300], dict(P, enable=0)), (0, ""))
check("emu threshold", decide_threshold([LONG], [], dict(P, emu_threshold=8192)), (8192, "emu"))
check("long_cap 0 = skip mode", decide_threshold([LONG], [10], dict(P, long_cap=0)), (64, "short-skip"))
# --- decide_hold
check("hold: nothing in flight", decide_hold(P, [], [LONG], None, True), False)
check("hold: big step in flight", decide_hold(P, [16384], [LONG], None, True), True)
check("hold: small step in flight, long pending", decide_hold(P, [2348], [LONG], None, True), True)
check("hold: decode only", decide_hold(P, [12], [DEC, DEC], None, True), False)
check("hold: decode in flight, long waiting head", decide_hold(P, [12], [DEC], 200_000, True), True)
check("hold: long waiting but no room", decide_hold(P, [12], [DEC], 200_000, False), False)
check("hold: short waiting head", decide_hold(P, [12], [DEC], 300, True), False)
check("hold off", decide_hold(dict(P, hold=0), [16384], [LONG], None, True), False)
check("hold: emu mode never holds", decide_hold(dict(P, emu_threshold=8192), [16384], [LONG], None, True), False)


# --- discrete-event sim -------------------------------------------------------
def step_time(n_long, n_short):
    # rough fit to r4-e5 (16K chunk ~2.9 s at 128K; 4K chunk ~1.0 s): F=0.31 s, 16K/2.6 s
    n = n_long + n_short
    return 0.0 if n == 0 else 0.31 + n / 6300.0


def sim(L, arrivals, short_tok, mode, p, budget=16384):
    """mode: 'stock' (async lookahead 2), 'thr' (stock flag p['thr']), 'pol' (policy).
    Returns (long_ttft, [short ttft])."""
    t = 0.0
    long_rem = L
    shorts = [dict(arr=a, rem=short_tok, done=None) for a in arrivals]
    inflight = []  # (t_end, n_long, [short idx finishing])
    gpu_free = 0.0
    long_done = None
    while long_done is None or any(s["done"] is None for s in shorts):
        waiting = [i for i, s in enumerate(shorts) if s["arr"] <= t and s["rem"] > 0 and s.get("sched") is None]
        lookahead = 2 if mode != "pol" or not p.get("hold", 1) else 1
        if len(inflight) < lookahead and (long_rem > 0 or waiting):
            # schedule a step now
            thr = 0
            if mode == "thr" and (waiting or long_rem > 0) and len(waiting) + (long_rem > 0) + sum(
                1 for s in shorts if s.get("sched") is not None and s["done"] is None) > 1:
                thr = p["thr"]
            if mode == "pol" and waiting and long_rem > max(p["long_cap"], short_tok):
                thr = max(p["long_cap"], short_tok)
            nl = min(long_rem, thr or budget, budget)
            room = budget - nl
            fin = []
            ns = 0
            for i in waiting:
                if shorts[i]["rem"] <= room:
                    room -= shorts[i]["rem"]
                    ns += shorts[i]["rem"]
                    shorts[i]["sched"] = True
                    fin.append(i)
            long_rem -= nl
            start = max(t, gpu_free)
            end = start + step_time(nl, ns)
            gpu_free = end
            inflight.append((end, nl, fin, long_rem == 0 and nl > 0))
            continue
        if not inflight:
            # idle until next arrival
            t = min(s["arr"] for s in shorts if s["arr"] > t)
            continue
        end, nl, fin, ldone = inflight.pop(0)
        t = max(t, end) + (0.004 if mode == "pol" else 0.0)  # sync gap per step under hold
        for i in fin:
            shorts[i]["done"] = t
            shorts[i]["rem"] = 0
        if ldone:
            long_done = t
    return long_done, [s["done"] - s["arr"] for s in shorts]


L = 131072
arr = [2 + 3 * i for i in range(6)]
print("\nsim 128K long + 6 shorts (300 tok) arriving at", arr)
res = {}
for name, mode, p in [("stock", "stock", {}), ("thr8k", "thr", {"thr": 8192}), ("pol-noh cap2k", "pol", {"long_cap": 2048, "hold": 0}),
                      ("pol hold cap2k", "pol", {"long_cap": 2048, "hold": 1}), ("pol hold cap4k", "pol", {"long_cap": 4096, "hold": 1}),
                      ("pol hold cap8k", "pol", {"long_cap": 8192, "hold": 1})]:
    lt, st = sim(L, arr, 300, mode, p)
    res[name] = (lt, st)
    print(f"  {name:16s} long {lt:6.2f} s  short p50 {statistics.median(st):5.2f} max {max(st):5.2f}")
base_long = res["stock"][0]
check("sim: policy+hold short p50 < 3 s", statistics.median(res["pol hold cap2k"][1]) < 3.0, True)
check("sim: policy+hold long within +10% at 128K", res["pol hold cap2k"][0] < 1.10 * base_long, True)
check("sim: hold beats no-hold on p50", statistics.median(res["pol hold cap2k"][1]) < statistics.median(res["pol-noh cap2k"][1]), True)
print("\nALL PASS" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)


def test_v2_skip():
    from dsv41_sched_policy import DEFAULTS, decide_threshold
    p = dict(DEFAULTS, long_cap=0)
    thr, kind = decide_threshold([(True, 500000), (False, 0)], [300], p)
    assert (thr, kind) == (320, "short-skip"), (thr, kind)
    thr, kind = decide_threshold([(True, 500000)], [3000, 100], p)
    assert (thr, kind) == (3008, "short-skip"), (thr, kind)
    thr, kind = decide_threshold([(True, 200)], [300], p)  # long nearly done: stock
    assert thr == 0, thr
    thr, kind = decide_threshold([(True, 500000)], [9000], p)  # no short waiting
    assert thr == 0, thr
    print("v2 skip: PASS")


test_v2_skip()
