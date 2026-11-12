#!/usr/bin/env python3
"""r5-glue CPU test: write sets of the new kernels == the live kernels' (every byte written once, same value class).

RQPAD   For every program (pid_c, r) of grid (N/256, M): the r2 loop `for p in range(M + r, PADDED_M, M)` and the kit's
        masked tile `p = M + r + j0 + jj*M (jj < PAD_J), mask p < PADDED_M, j0 in range(0, PADDED_M - M - r, PAD_J*M)`
        give the same padding-row set, hence the same SF byte offsets (all zero bytes); together with the real rows
        every SF byte of [PADDED_M * SF_COLS] is written exactly once. M = 1..PADDED_M for PADDED_M 128 / 256 / 384
        (sonar serves M <= 32, PADDED_M = 128; the rest guards the general formula), N = 1024 and 2048.
RPSPLIT Per-program store duties of the live kernel, hoist and split (grid (M*6, 5)), emulated from the kernel text
        (plan in numpy, same integer formulas): every (buffer, index) is written exactly once in each variant and the
        three write sets are identical, with the same producing (route, chunk) for every row byte / scale word.
        Routings: random distinct / same experts / hot experts / with dead (-1) ids, M = 1..10 (the RPSPLIT scope),
        both 32-aligned layouts and the strided dead-row fill (nprog = num_programs(0) in all three).
"""
import collections
import sys

import numpy as np

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------------------------------------- RQPAD
def sf_off(c, p, sf_cols):
    sf_col = (c % 4) + (c // 4) * 512
    return sf_col + (p % 32) * 16 + ((p % 128) // 32) * 4 + (p // 128) * (128 * sf_cols)


def rq_pad_rows_live(M, r, padded_m):
    return list(range(M + r, padded_m, M))


def rq_pad_rows_kit(M, r, padded_m, pad_j=128):
    rows = []
    for j0 in range(0, padded_m - M - r, pad_j * M):
        p = M + r + j0 + np.arange(pad_j) * M
        rows += [int(x) for x in p[p < padded_m]]
    return rows


def test_rq():
    n_cases = 0
    for n in (1024, 2048):
        sf_cols = n // 32
        for padded_m in (128, 256, 384):
            for M in range(max(1, padded_m - 127), padded_m + 1):
                if (M + 127) // 128 * 128 != padded_m:
                    continue
                writes = collections.Counter()
                for r in range(M):
                    live = rq_pad_rows_live(M, r, padded_m)
                    kit = rq_pad_rows_kit(M, r, padded_m)
                    if sorted(live) != sorted(kit) or len(set(kit)) != len(kit):
                        check(f"RQ pad rows N={n} PADDED_M={padded_m} M={M} r={r}", False, f"{live[:5]} vs {kit[:5]}")
                        return
                    for pid_c in range(n // 256):
                        c = pid_c * 8 + np.arange(8)
                        for p in [r] + kit:
                            for o in sf_off(c, p, sf_cols):
                                writes[int(o)] += 1
                    n_cases += 1
                full = set(range(padded_m * sf_cols))
                if set(writes) != full or set(writes.values()) != {1}:
                    check(f"RQ SF coverage N={n} PADDED_M={padded_m} M={M}", False,
                          f"missing {len(full - set(writes))}, extra {len(set(writes) - full)}, "
                          f"multi {sum(1 for v in writes.values() if v > 1)}")
                    return
    check(f"RQ padding rows live == kit and every SF byte written exactly once ({n_cases} (N, PADDED_M, M, r) "
          "programs, M from 1 to 384)", True)
    # the sonar range explicitly, and the loop trip count of the kit kernel (1 for every served M)
    trips = {M: len(range(0, 128 - M - 0, 128 * M)) for M in range(1, 33)}
    check("RQ kit: one masked store per program for every sonar M 1..32 (r = 0 worst case)",
          all(t == 1 for M, t in trips.items() if M < 128), str(trips))
    live_trips = {M: len(range(M, 128, M)) for M in (1, 2, 4, 5, 6)}
    print(f"INFO r2 loop trips (r = 0) at M 1/2/4/5/6: {live_trips}")


# -------------------------------------------------------------------------------------------------------- RPSPLIT
E, TOPK, ALIGN, H, GROUP, GB, FILL = 384, 6, 32, 5120, 128, 8, 64
PACKS = H // GROUP // 4
NCH = H // (GB * GROUP)


def m_sum(M):
    act = min(M * TOPK, E)
    ms = M * TOPK + act * (ALIGN - 1)
    return (ms + ALIGN - 1) // ALIGN * ALIGN


def plan(ids):
    flat = ids.reshape(-1)
    valid = (flat >= 0) & (flat < E)
    cnt = np.bincount(flat[valid], minlength=E)[:E]
    aligned = (cnt + ALIGN - 1) // ALIGN * ALIGN
    start = np.cumsum(aligned) - aligned
    return flat, valid, cnt, aligned, start, int(aligned.sum())


def duties(ids, variant):
    """[(buffer, index, producer)] for every store of every program; producer names the value's source."""
    M = ids.shape[0]
    Ms = m_sum(M)
    flat, valid, cnt, aligned, start, total = plan(ids)
    nprog = M * TOPK
    ys = range(NCH) if variant == "split" else [0]
    out = []
    for pid in range(nprog):
        for y in ys:
            is0 = y == 0
            t, k = divmod(pid, TOPK)
            e = int(flat[pid])
            live = 0 <= e < E
            if live:
                rank = int(np.sum(valid[:pid] & (flat[:pid] == e)))
                dest = int(start[e]) + rank
                first = rank == 0
                if is0:
                    out.append(("inv", (t, k), dest))
                    out.append(("layout", dest, e))
                chunks = range(NCH) if variant != "split" else [y]
                for ch in chunks:
                    # bytes [ch*1024, (ch+1)*1024) of slot row dest, from token row t's same columns
                    out.append(("row-chunk", (dest, ch), (t, ch)))
                    for p in range(ch * (GB // 4), (ch + 1) * (GB // 4)):
                        out.append(("sfp", (p, dest), (t, p)))
            else:
                first = False
            if first and is0:
                for r in range(int(start[e] + cnt[e]), int(start[e] + aligned[e])):
                    out.append(("layout", r, -1))
                    for pk in range(PACKS):
                        out.append(("sfp", (pk, r), 0))
            if is0:
                for r0 in range(total + pid * FILL, Ms, nprog * FILL):
                    for r in range(r0, min(r0 + FILL, Ms)):
                        out.append(("layout", r, -1))
                        for pk in range(PACKS):
                            out.append(("sfp", (pk, r), 0))
            if pid == 0 and is0:
                for b in range(E):
                    out.append(("ends", b, int(start[b] + cnt[b])))
    return out


def routings(M, rng):
    yield "distinct", np.stack([rng.choice(E, TOPK, replace=False) for _ in range(M)])
    yield "same", np.tile(rng.choice(E, TOPK, replace=False), (M, 1))
    hot = np.r_[np.arange(8)]
    yield "hot", np.stack([rng.choice(np.r_[hot, rng.choice(np.arange(8, E), TOPK, replace=False)], TOPK,
                                      replace=False) for _ in range(M)])
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(M)])
    ids[rng.random(ids.shape) < 0.3] = -1
    yield "dead", ids


def test_rp():
    rng = np.random.default_rng(20261109)
    n = 0
    for M in range(1, 11):
        for name, ids in routings(M, rng):
            ref = duties(ids, "live")
            keys = [(b, i) for b, i, _ in ref]
            if len(keys) != len(set(keys)):
                check(f"RP live write set has duplicates (emulator) M={M} {name}", False)
                return
            for v in ("hoist", "split"):
                got = duties(ids, v)
                gk = [(b, i) for b, i, _ in got]
                if len(gk) != len(set(gk)):
                    dup = [x for x, c in collections.Counter(gk).items() if c > 1][:3]
                    check(f"RP {v} M={M} {name}: every element written once", False, str(dup))
                    return
                if sorted(map(repr, got)) != sorted(map(repr, ref)):
                    check(f"RP {v} M={M} {name}: same (buffer, index, producer) set as live", False)
                    return
                n += 1
    check(f"RP hoist and split: identical write sets to the live kernel, each element once ({n} cases, M 1..10, "
          "4 routings)", True)


if __name__ == "__main__":
    test_rq()
    test_rp()
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
