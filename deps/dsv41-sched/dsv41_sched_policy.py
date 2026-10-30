"""Pure policy for dsv41_sched (no vLLM imports, unit-testable anywhere).

Problem: in vLLM 0.32 a running long prefill takes the whole 16384-token budget
every step, and async scheduling pre-schedules step N+1 before step N finishes,
so a short request that arrives during a 1M prefill waits for the whole prefill
(~190 s).

Policy (all decisions read scheduler-local state only):
  * short step: if a long prefill is running and a waiting request (among the
    first `scan`) has an uncached remainder <= short_max, this step caps every
    request's new tokens at thr = max(long_cap, largest such remainder) (rounded
    up to the block size), so the short request(s) get their whole prompt in the
    same step and the long prefill gets only `long_cap` tokens.  Implemented by
    setting the stock long_prefill_token_threshold for that one step.
  * decode cap (optional, decode_cap > 0): while a long prefill runs together
    with a decoding request, cap the long chunk at decode_cap.
  * skip (long_cap=0): in a short step the running long prefill gets no tokens at
    all, so the short step costs only the shorts (a 2K long chunk at ~900K context
    still costs ~1 s).
  * hold: never pre-schedule (async lookahead) a step while a big step is in
    flight or while a long prefill still has >= hold_min tokens to go.  Long
    chunks are then scheduled synchronously, so a request that arrives during
    chunk N is in chunk N+1 instead of N+2.
  * lone long prefill: always the stock full chunk (no short waiting, no cap).
"""

DEFAULTS = {
    "enable": 1,         # 0 = exact stock AsyncScheduler behaviour
    "short_max": 4096,   # uncached remainder that counts as "short"
    "long_cap": 2048,    # long-prefill chunk in a short step; 0 = skip the long prefill in that step
    "scan": 32,          # waiting requests scanned for a short candidate
    "hold": 1,           # 1 = no async lookahead while a long prefill is pending;
                         # 2 = timed: hold only if a request arrives before the in-flight big step's
                         #     predicted end minus hold_margin_ms, else pre-schedule as stock
    "hold_margin_ms": 300,  # hold=2: pre-schedule this long before the predicted end of the big step
    "hold_min": 4096,    # "big step" / "long prefill" size for the hold
    "decode_cap": 0,     # >0: cap long chunk while a decoder is running
    "emu_threshold": 0,  # >0: emulate stock --long-prefill-token-threshold (policy off)
    "probe": 1,          # 1 = prefix-cache probe for waiting requests (uncached remainder)
}


def round_up(x, b):
    return ((int(x) + b - 1) // b) * b


def decide_threshold(running, waiting_fresh, p, block=64):
    """running: list of (prefilling: bool, prefill_remaining: int).
    waiting_fresh: uncached remainders of the first waiting requests (FCFS order).
    Returns (threshold, kind); threshold 0 = stock (no cap this step)."""
    if p.get("emu_threshold", 0) > 0:
        return int(p["emu_threshold"]), "emu"
    if not p.get("enable", 1):
        return 0, ""
    skip = int(p["long_cap"]) <= 0
    long_cap = block if skip else max(int(p["long_cap"]), block)
    long_rem = [rem for pf, rem in running if pf]
    decoding = any(not pf for pf, _ in running)
    shorts = [f for f in waiting_fresh[: int(p["scan"])] if 0 < f <= int(p["short_max"])]
    thr = 0
    kind = ""
    if shorts:
        t = round_up(max(long_cap, max(shorts)), block)
        if any(rem > t for rem in long_rem):
            # skip: running long prefills (rem > thr) get no tokens this step (the caller
            # removes them from `running` for the step); thr still caps waiting requests.
            thr, kind = t, "short-skip" if skip else "short"
    dc = int(p.get("decode_cap", 0))
    if dc > 0 and decoding:
        t = round_up(max(dc, block), block)
        if any(rem > t for rem in long_rem):
            if thr == 0:
                thr, kind = t, "dcap"
            else:
                thr = max(min(thr, t), round_up(max(shorts), block))
                kind = "short+dcap"
    return thr, kind


def decide_hold(p, inflight_tokens, running, waiting_head_fresh, room):
    """inflight_tokens: total_num_scheduled_tokens of each batch still in the engine's
    batch queue (empty list = nothing in flight -> never hold).
    running: list of (prefilling, prefill_remaining). waiting_head_fresh: fresh tokens
    of the first waiting request or None. room: True if another request can be admitted."""
    if not p.get("enable", 1) or not p.get("hold", 1) or p.get("emu_threshold", 0) > 0:
        return False
    if not inflight_tokens:
        return False
    hm = int(p["hold_min"])
    if any(t >= hm for t in inflight_tokens):
        return True
    if any(pf and rem >= hm for pf, rem in running):
        return True
    if room and waiting_head_fresh is not None and waiting_head_fresh >= hm:
        return True
    return False
