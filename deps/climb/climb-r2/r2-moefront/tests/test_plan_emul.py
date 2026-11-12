"""CPU emulation of the route plan + scatter: live mfront (plan kernel + atomic scatter) vs the fused kernel's
per-program plan (every program recomputes the histogram; slot = start[e] + prefix rank; dead rows written only by
their owner). Integer/copy work only, so the check is exact array equality.

For M = 1..96 (+ the 170-token cap) with random routes, ROUTE_LIVE dead rows (-1), and hostile routings
(all tokens on the same 6 experts, all-distinct experts, all dead, one live row, the 384-expert saturation case):
  * grouped_layout [M_sum] identical (expert id on live rows, -1 everywhere else)
  * expert_ends [E] identical (start + count; LIVE_ACT reads it)
  * every live route's FP8 row bytes and packed scale words identical through inv_perm (row order inside an
    expert segment is the only difference: atomic-arrival order today, (token, k) order fused)
  * dead-row packed scale words 0 in both
  * the fused kernel writes every row of [0, M_sum) exactly once (no write-write race between programs), 40 random
    atomic orders of the live scatter all give the same contract
  * a row-independent FC1/FC2 stand-in followed by the fixed top-6 gather gives identical MoE outputs
Run: python3 tests/test_plan_emul.py
"""
import sys
import time

import numpy as np

E, TOPK, ALIGN, H_WORDS, PACKS, FILL_ROWS = 384, 6, 32, 1280, 10, 64
RNG = np.random.default_rng(7)
FAIL = []


def check(name, ok, detail=""):
    if not ok or VERBOSE:
        print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAIL.append(name)


VERBOSE = False


def m_sum(M):
    """compute_aligned_M_and_alignment (expert_tokens_meta None, alignment 32)."""
    s = M * TOPK + min(M * TOPK, E) * (ALIGN - 1)
    return (s + ALIGN - 1) // ALIGN * ALIGN


def plan(ids):
    valid = (ids >= 0) & (ids < E)
    cnt = np.bincount(ids[valid], minlength=E).astype(np.int64)
    aligned = (cnt + ALIGN - 1) // ALIGN * ALIGN
    start = np.cumsum(aligned) - aligned
    return cnt, aligned, start, int(aligned.sum())


def live_mfront(ids2d, rows, scales, Ms, order):
    """_dsv41_fp_plan_kernel + _dsv41_fp_scatter_kernel with the scatter programs arriving in `order`."""
    M = ids2d.shape[0]
    ids = ids2d.reshape(-1)
    layout = np.full(Ms, -1, np.int32)
    sfp = np.zeros((PACKS, Ms), np.int64)
    out = np.full((Ms, H_WORDS), 0x5A5A5A5A, np.int64)  # workspace garbage, never written on dead rows
    inv = np.full((M, TOPK), -7, np.int32)  # torch.empty: dead routes keep garbage
    _, _, start, _ = plan(ids)
    cursor = start.copy()
    for slot in order:
        t, k = divmod(int(slot), TOPK)
        e = ids[slot]
        if 0 <= e < E:
            dest = cursor[e]
            cursor[e] += 1
            inv[t, k] = dest
            layout[dest] = e
            out[dest] = rows[t]
            sfp[:, dest] = scales[t]
    return layout, sfp, out, inv, cursor


def fused(ids2d, rows, scales, Ms):
    """_dsv41_rp_kernel: program pid = route (t, k); writes recorded per row to prove single ownership."""
    M = ids2d.shape[0]
    ids = ids2d.reshape(-1)
    nprog = M * TOPK
    layout = np.full(Ms, 0x7777, np.int32)  # poisoned: every row must be written
    sfp = np.full((PACKS, Ms), 0x6666, np.int64)
    out = np.full((Ms, H_WORDS), 0x5A5A5A5A, np.int64)
    inv = np.full((M, TOPK), -7, np.int32)
    writes = np.zeros(Ms, np.int64)
    ends = np.full(E, -9, np.int64)
    for pid in range(nprog):
        cnt, aligned, start, total = plan(ids)  # recomputed by every program
        t, k = divmod(pid, TOPK)
        e = ids[pid]
        live = 0 <= e < E
        s = np.arange(nprog)
        rank = int(((s < pid) & (ids == e) & (ids >= 0) & (ids < E)).sum())
        if live:
            dest = start[e] + rank
            inv[t, k] = dest
            layout[dest] = e
            out[dest] = rows[t]
            sfp[:, dest] = scales[t]
            writes[dest] += 1
            if rank == 0:
                r = np.arange(start[e] + cnt[e], start[e] + aligned[e])
                layout[r] = -1
                sfp[:, r] = 0
                writes[r] += 1
        for r0 in range(total + pid * FILL_ROWS, Ms, nprog * FILL_ROWS):
            r = np.arange(r0, min(r0 + FILL_ROWS, Ms))
            layout[r] = -1
            sfp[:, r] = 0
            writes[r] += 1
        if pid == 0:
            ends[:] = start + cnt
    return layout, sfp, out, inv, ends, writes


def moe_standin(layout, out, inv, ids2d, w):
    """Row-independent FC1/FC2 stand-in (each output row a function of its own A row and its expert only) and the
    fixed top-6 order gather that skips id -1 (ep_gather / Athena gated gather semantics)."""
    act = np.zeros(out.shape[0], np.int64)
    liv = layout >= 0
    act[liv] = (out[liv] * (layout[liv, None] + 3)).sum(axis=1) % 1000003
    res = np.zeros(ids2d.shape[0], np.int64)
    for t in range(ids2d.shape[0]):
        for k in range(TOPK):
            if ids2d[t, k] >= 0:
                res[t] = (res[t] * 131 + act[inv[t, k]] * w[t, k]) % 1000000007
    return res


def routes(M, kind):
    if kind == "random":
        return np.stack([RNG.choice(E, TOPK, replace=False) for _ in range(M)]).astype(np.int64)
    if kind == "dead":  # ROUTE_LIVE=skip padding rows
        ids = routes(M, "random")
        ids[RNG.random(M) < 0.4] = -1
        return ids
    if kind == "same":
        return np.tile(RNG.choice(E, TOPK, replace=False), (M, 1)).astype(np.int64)
    if kind == "distinct":
        n = M * TOPK
        p = RNG.permutation(E) if n <= E else np.concatenate([RNG.permutation(E) for _ in range(n // E + 1)])
        return p[:n].reshape(M, TOPK).astype(np.int64)
    if kind == "alldead":
        return np.full((M, TOPK), -1, np.int64)
    if kind == "onelive":
        ids = np.full((M, TOPK), -1, np.int64)
        ids[RNG.integers(M)] = RNG.choice(E, TOPK, replace=False)
        return ids
    if kind == "hot":  # skewed: a few experts take most routes
        return np.stack([RNG.choice(np.r_[np.arange(8), RNG.choice(E, TOPK, replace=False)], TOPK, replace=False)
                         for _ in range(M)]).astype(np.int64)
    raise KeyError(kind)


def one(M, kind, orders=3):
    ids2d = routes(M, kind)
    Ms = m_sum(M)
    rows = RNG.integers(-2 ** 31, 2 ** 31, size=(M, H_WORDS))
    scales = RNG.integers(0, 2 ** 31, size=(M, PACKS))
    w = RNG.integers(1, 1000, size=(M, TOPK))
    fl, fs, fo, fi, fe, writes = fused(ids2d, rows, scales, Ms)
    tag = f"M={M} {kind}"
    check(f"{tag}: every row written exactly once", bool((writes == 1).all()),
          f"min {writes.min()} max {writes.max()}")
    ids = ids2d.reshape(-1)
    live = ids >= 0
    ref_out = None
    for o in range(orders):
        order = RNG.permutation(M * TOPK)
        ll, ls, lo, li, lc = live_mfront(ids2d, rows, scales, Ms, order)
        check(f"{tag} order {o}: grouped_layout", np.array_equal(ll, fl))
        check(f"{tag} order {o}: expert_ends", np.array_equal(lc, fe))
        dead_rows = ll < 0
        check(f"{tag} order {o}: dead-row scale words 0", bool((ls[:, dead_rows] == 0).all() and
                                                           (fs[:, dead_rows] == 0).all()))
        t_idx, k_idx = np.nonzero(ids2d >= 0)
        dl, df = li[t_idx, k_idx], fi[t_idx, k_idx]
        check(f"{tag} order {o}: live rows through inv_perm", np.array_equal(lo[dl], fo[df]) and
              np.array_equal(ls[:, dl], fs[:, df]))
        check(f"{tag} order {o}: inv_perm lands in the right segment", bool((fl[df] == ids[live]).all()))
        a = moe_standin(ll, lo, li, ids2d, w)
        b = moe_standin(fl, fo, fi, ids2d, w)
        check(f"{tag} order {o}: MoE stand-in output", np.array_equal(a, b))
        ref_out = a if ref_out is None else ref_out
        check(f"{tag} order {o}: live path itself order-independent", np.array_equal(a, ref_out))


if __name__ == "__main__":
    VERBOSE = "-v" in sys.argv
    t0 = time.time()
    n = 0
    for M in list(range(1, 97)) + [128, 170]:
        for kind in ("random", "dead", "same", "distinct", "alldead", "onelive", "hot"):
            one(M, kind, orders=3 if M <= 16 else 1)
            n += 1
    for M in (1, 4, 6, 36):  # many atomic orders on the serving shapes
        one(M, "random", orders=40)
        n += 1
    print(f"{n} cases (M 1..96, 128, 170 x 7 routings + 4 x 40 atomic orders): "
          f"{'ALL PASS' if not FAIL else 'FAILED ' + str(len(FAIL)) + ': ' + ', '.join(FAIL[:10])} "
          f"({time.time() - t0:.1f}s)")
    sys.exit(1 if FAIL else 0)
