"""g130 fc1: which G each replay of the direct test runs. Pure Python, no torch, so tests/test_cpu.py can check it."""


def parse_gs(s):
    """'0-18' or '0,1,8,9' or a mix -> list of G, in order. '' -> []."""
    out = []
    for part in s.split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def check(gs, replays, top):
    """An explicit list must be run in full and must fit the tokens value: refuse rather than drop a G."""
    assert replays >= len(gs), "--replays %d cannot cover %d listed G" % (replays, len(gs))
    assert max(gs) <= top, "G %d is over top %d at these tokens" % (max(gs), top)


def g_for(r, top, gs, draw):
    """gs set: cycle over it. Otherwise the exact-run schedule: even r walks 0..top, odd r is draw(top + 1)."""
    if gs:
        return gs[r % len(gs)]
    return r % (top + 1) if r % 2 == 0 else draw(top + 1)
