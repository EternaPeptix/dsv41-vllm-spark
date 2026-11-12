#!/usr/bin/env python3
"""r6-head refine: CPU emulation of the r6 candidate/list/persistent-refine logic against the live certificate
(torch CPU; run on 512S1: ~/exo-venv313/bin/python3 tests/test_refine_logic.py).

The live module (base copy) supplies the certificate references (build_cert_tables, reference_bounds,
reference_needed_tiles, reference_masked). The "exact" logit stand-in is float64 x.W^T rounded once to bf16 (the
device test compares against the real kernels; here only the selection / list / assembly logic is under test, and
every r6 kernel computes a listed row with the live op sequence, so the stand-in is the same function for both).

Emulated kernels (statement for statement with dsv41_head_r6.py):
  cand    per CTA of CB rows: hrow = any_m(h >= T[m]) with masked loads (-inf, +inf) as in the kernel; htile = OR
          over TB rows; tile: sel = htile broadcast; row: sel = hrow; sel |= bad; sel &= j < V; list append in CTA
          order (and in a random CTA order: the output must not depend on it); -inf where not sel.
  refine  persistent grid NPROG, rpc = clamp(ceil(n / NPROG), 1, 16) (spread) or 16; chunk c -> rows c*rpc + rr.
Checks (every case, M in {1, 2, 7, 16, 17, 33, 96}):
  tile:  out == reference_masked(exact, reference_needed_tiles(...)) bit for bit (== the live refine's output)
  row:   out == where(row predicate, exact, -inf); row set subset of the tile set; list length == predicate count
  both:  leftmost argmax == stock (float64 -> bf16) leftmost argmax, per row; sharded leftmost-max exchange
         (sf_local_argmax + first-rank-wins onehot, emulated) == global stock argmax
  list:  every listed row computed exactly once for n in {0..16160} x NPROG x spread
  certificate sanity: lo <= exact <= hi for the stand-in (else the case is invalid)
Hostile cases: forced ties (duplicate rows, across tiles / inside one tile / across shards), all rows identical,
zero x rows, a non-finite x row (bad -> every row), padding rows (vreal < V), V not a multiple of 16.
Must-fail mutants (the harness must catch each in every listed case family): row mode dropping the first selected row
of each CTA; tile mode with tiles shifted by one row; the r6 self-check compare() fed a corrupted output.
"""
import importlib.util
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(0)
random.seed(0)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


r4 = load("vllm.models.deepseek_v41.nvidia.dsv41_r4_small",
          os.path.join(HERE, "base/vllm/models/deepseek_v41/nvidia/dsv41_r4_small.py"))
r6 = load("dsv41_head_r6", os.path.join(HERE, "files/vllm/models/deepseek_v41/nvidia/dsv41_head_r6.py"))
FAILS, N = [], [0]
NINF = float("-inf")


def check(name, ok, detail=""):
    N[0] += 1
    if not ok or os.environ.get("VERBOSE"):
        print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def bits(t):
    return t.contiguous().view(torch.int16)


# ------------------------------------------------------------------ emulated r6 kernels
def cand_emul(hi, T, bad, V, CB, TB, tile, order=None, mutant=None):
    """hi [M, V] fp32 (bf16 values), T [M], bad [M] int -> (sel [V] bool, idx list, ntile)."""
    M = hi.shape[0]
    BMX = max(16, r4.next_pow2(M))
    nprog = (V + CB - 1) // CB
    pids = list(range(nprog)) if order is None else order
    idx, ntile_total = [], 0
    sel_all = torch.zeros(V, dtype=torch.bool)
    anybad = bool((bad[:M] != 0).any())
    for pid in pids:
        j = pid * CB + torch.arange(CB)
        ok = j < V
        h = torch.full((BMX, CB), NINF)
        h[:M][:, ok] = hi[:, j[ok]]
        Tm = torch.full((BMX,), float("inf"))
        Tm[:M] = T
        hrow = (h >= Tm[:, None]).any(dim=0)
        if mutant == "tile_shift":
            hrow = torch.roll(hrow, 8)
        htile = hrow.view(CB // TB, TB).any(dim=1)
        tneed = htile | anybad
        tvalid = (pid * CB + torch.arange(CB // TB) * TB) < V
        ntile_total += int((tneed & tvalid).sum())
        sel = tneed.repeat_interleave(TB) if tile else (hrow | anybad)
        sel = sel & ok
        rows = j[sel].tolist()
        if mutant == "drop_first" and rows:
            rows = rows[1:]
            sel = sel.clone()
            sel[(j == j[sel][0]).nonzero()[0]] = False
        idx += rows
        sel_all[j[ok]] = sel[ok]
    return sel_all, idx, ntile_total


def chunks_emul(n, nprog, spread, BV=16):
    rpc = min(max((n + nprog - 1) // nprog, 1), BV) if spread else BV
    nch = (n + rpc - 1) // rpc if n else 0
    seen = []
    for pid in range(nprog):
        for c in range(pid, nch, nprog):
            for rr in range(BV):
                r = c * rpc + rr
                if rr < rpc and r < n:
                    seen.append(r)
    return seen


def assemble(exact, sel, idx, V):
    """out = -inf where not selected (cand kernel); listed rows = exact (refine kernel)."""
    out = torch.full_like(exact, NINF)
    if idx:
        ii = torch.tensor(idx)
        out[:, ii] = exact[:, ii]
    return out


def leftmost_argmax(t):
    return t.float().argmax(dim=-1)  # torch: first maximal index


def sharded_argmax(out, tp):
    """sf_local_argmax (leftmost max per shard, NaN -> -inf) + _sf_onehot (first rank holding the max)."""
    M, V = out.shape
    w = V // tp
    best_v, best_t = None, None
    for r in range(tp):
        sh = out[:, r * w:(r + 1) * w].float()
        sh = torch.where(torch.isnan(sh), torch.full_like(sh, NINF), sh)
        v, i = sh.max(dim=-1)
        i = sh.argmax(dim=-1)
        t = i + r * w
        if best_v is None:
            best_v, best_t = v, t
        else:
            upd = v > best_v
            best_t = torch.where(upd, t, best_t)
            best_v = torch.where(upd, v, best_v)
    return best_t


# ------------------------------------------------------------------ cases
def make_case(kind, M, V=1000, K=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(V, K, generator=g) * 0.02).to(torch.bfloat16)
    x = (torch.randn(M, K, generator=g) * 1.0).to(torch.bfloat16)
    vreal = None
    ties = []
    if kind == "tie_across_tiles":
        a, b = 37, 37 + 16 * 20 + 3
        W[b] = W[a]
        x = (W[a].float() * 400.0).repeat(M, 1).to(torch.bfloat16) + (torch.randn(M, K, generator=g) * 0.01).to(
            torch.bfloat16)
        ties = [a, b]
    elif kind == "tie_in_tile":
        a, b = 32, 41
        W[b] = W[a]
        x = (W[a].float() * 400.0).repeat(M, 1).to(torch.bfloat16)
        ties = [a, b]
    elif kind == "tie_across_shards":
        a, b = 100, 600  # shards of 250 rows at tp=4: shard 0 and shard 2
        W[b] = W[a]
        x = (W[a].float() * 400.0).repeat(M, 1).to(torch.bfloat16)
        ties = [a, b]
    elif kind == "all_identical":
        W[:] = W[0]
    elif kind == "zero_x":
        x[0] = 0
    elif kind == "bad_row":
        x[M - 1, 5] = float("inf")
    elif kind == "padding":
        vreal = V - 40
    elif kind == "near_ties":
        # many rows close to the max: a cluster of perturbed copies spread over the vocab
        base = W[7].clone()
        for j in range(50, V, 37):
            W[j] = (base.float() + torch.randn(K, generator=g) * 0.002).to(torch.bfloat16)
        x = (base.float() * 300.0).repeat(M, 1).to(torch.bfloat16)
    return W, x, vreal, ties


def run_case(kind, M, V=1000, K=512, seed=0, mutant=None, quiet=False):
    """Returns the list of failed property names for this case (the caller decides PASS/expected-FAIL)."""
    W, x, vreal, ties = make_case(kind, M, V, K, seed)
    st = r4.build_cert_tables(W, vreal=vreal)
    lo, hi, T, badm = r4.reference_bounds(st, x)
    bad = badm.to(torch.int32)
    exact = (x.double() @ W.double().t()).to(torch.bfloat16)
    if st.vreal < V:
        exact[:, st.vreal:] = NINF  # padding rows: never needed; value irrelevant
    errs = []
    fin = torch.isfinite(exact.float())
    if not bool(badm.any()):
        inb = ((lo <= exact.float()) | ~fin) & ((exact.float() <= hi) | ~fin)
        if not bool(inb[:, :st.vreal].all()):
            errs.append("certificate (stand-in outside [lo, hi])")
    needed = r4.reference_needed_tiles(hi, T, badm, bv=16)
    live_out = r4.reference_masked(exact, needed, bv=16)
    BMX = max(16, r4.next_pow2(M))
    CB = r6.cand_rows(BMX, 16)
    rowpred = r6.row_select(hi.to(torch.bfloat16), T, bad)
    for tile in (True, False):
        mode = "tile" if tile else "row"
        if mutant == "tile_shift" and not tile:
            continue
        if mutant == "drop_first" and tile:
            continue
        nprog = (V + CB - 1) // CB
        order = list(range(nprog))
        random.Random(seed + M).shuffle(order)
        sel, idx, ntile = cand_emul(hi, T, bad, V, CB, 16, tile, mutant=mutant)
        sel2, idx2, _ = cand_emul(hi, T, bad, V, CB, 16, tile, order=order, mutant=mutant)
        out = assemble(exact, sel, idx, V)
        out2 = assemble(exact, sel2, idx2, V)
        if not torch.equal(bits(out), bits(out2)):
            errs.append(f"{mode}: output depends on CTA order")
        if ntile != int(needed.sum()):
            errs.append(f"{mode}: live-equivalent tile count {ntile} != live {int(needed.sum())}")
        if len(set(idx)) != len(idx):
            errs.append(f"{mode}: duplicate list entries")
        if tile:
            if not torch.equal(bits(out), bits(live_out)):
                errs.append("tile: output != live refine output (bitwise)")
        else:
            exp = r6.expected_row_out(live_out, rowpred)
            if not torch.equal(bits(out), bits(exp)):
                errs.append("row: output != where(row predicate, live, -inf)")
            if len(idx) != int(rowpred.sum()):
                errs.append("row: list length != predicate count")
            tiles_of_rows = torch.zeros_like(needed)
            if idx:
                tiles_of_rows[torch.tensor(idx) // 16] = True
            if bool((tiles_of_rows & ~needed).any()):
                errs.append("row: a computed row outside the live tile set")
        ref = leftmost_argmax(exact[:, :st.vreal])
        got = leftmost_argmax(out)
        if not bool(badm.any()) and not torch.equal(ref, got):
            errs.append(f"{mode}: argmax != stock argmax")
        if ties and not bool(badm.any()) and not bool((got == min(ties)).all()):
            errs.append(f"{mode}: forced tie not resolved to the leftmost tied row {min(ties)}")
        if not bool(badm.any()) and V % 4 == 0 and st.vreal == V:
            if not torch.equal(sharded_argmax(out, 4), ref):
                errs.append(f"{mode}: sharded leftmost-max exchange != stock argmax")
        # the in-process self-check must accept the correct r6 result
        live_d = dict(out=live_out, hi=hi.to(torch.bfloat16), thr=T, bad=bad)
        new_d = dict(out=out, hi=hi.to(torch.bfloat16), thr=T, bad=bad, cnt=torch.tensor([len(idx)]))
        err = r6.compare(live_d, new_d, mode, False)
        if err:
            errs.append(f"{mode}: compare() rejected a correct result ({err})")
    return errs


def main():
    kinds = ["random", "tie_across_tiles", "tie_in_tile", "tie_across_shards", "all_identical", "zero_x", "bad_row",
             "padding", "near_ties"]
    Ms = [1, 2, 7, 16, 17, 33, 96]
    for kind in kinds:
        for M in Ms:
            for seed in (0, 1):
                errs = run_case(kind, M, seed=seed)
                check(f"{kind} M={M} seed={seed}", not errs, "; ".join(errs))
    # V not a multiple of 16 nor of CB, larger K
    for M in (2, 7):
        errs = run_case("random", M, V=1013, K=1024, seed=3)
        check(f"random V=1013 K=1024 M={M}", not errs, "; ".join(errs))
        errs = run_case("tie_across_tiles", M, V=1013, K=1024, seed=4)
        check(f"tie_across_tiles V=1013 K=1024 M={M}", not errs, "; ".join(errs))
    # list coverage
    for n in (0, 1, 2, 15, 16, 17, 100, 1000, 3000, 16160):
        for nprog in (1, 7, 192):
            for spread in (0, 1):
                seen = chunks_emul(n, nprog, spread)
                check(f"list coverage n={n} nprog={nprog} spread={spread}", sorted(seen) == list(range(n)))
    # cand rows per BMX
    for BMX, want in ((16, 256), (32, 128), (64, 64), (128, 32)):
        check(f"cand_rows(BMX={BMX}) == {want}", r6.cand_rows(BMX, 16) == want, str(r6.cand_rows(BMX, 16)))
    # must-fail mutants: each must be caught in every listed family
    for mut, fams in (("drop_first", ["tie_across_tiles", "tie_in_tile", "tie_across_shards"]),
                      ("tile_shift", ["random", "tie_across_tiles", "near_ties"])):
        for kind in fams:
            for M in (2, 7, 33):
                errs = run_case(kind, M, seed=0, mutant=mut)
                check(f"MUTANT {mut} caught on {kind} M={M}", bool(errs), "; ".join(errs)[:120] or "NOT CAUGHT")
    # compare() must reject corrupted outputs
    W, x, vreal, ties = make_case("random", 7)
    st = r4.build_cert_tables(W)
    lo, hi, T, badm = r4.reference_bounds(st, x)
    exact = (x.double() @ W.double().t()).to(torch.bfloat16)
    needed = r4.reference_needed_tiles(hi, T, badm, bv=16)
    live_out = r4.reference_masked(exact, needed, bv=16)
    bad = badm.to(torch.int32)
    live_d = dict(out=live_out, hi=hi.to(torch.bfloat16), thr=T, bad=bad)
    cor = live_out.clone()
    j = int(leftmost_argmax(exact)[0])
    cor[0, j] = (cor[0, j].float() + 2 ** -7 * abs(cor[0, j].float()) + 1e-3).to(torch.bfloat16)
    check("compare() rejects a one-ulp-ish change (tile)", r6.compare(live_d, dict(live_d, out=cor), "tile", False)
          != "")
    sel = r6.row_select(hi.to(torch.bfloat16), T, bad)
    rowout = r6.expected_row_out(live_out, sel)
    check("compare() accepts the exact row output", r6.compare(
        live_d, dict(live_d, out=rowout, cnt=torch.tensor([int(sel.sum())])), "row", False) == "")
    check("compare() rejects a wrong list length (row)", r6.compare(
        live_d, dict(live_d, out=rowout, cnt=torch.tensor([int(sel.sum()) + 1])), "row", False) != "")
    hi2 = hi.to(torch.bfloat16).clone()
    hi2[0, 3] = (hi2[0, 3].float() * 1.01).to(torch.bfloat16)
    check("compare() rejects a changed HI (screen)", r6.compare(live_d, dict(live_d, hi=hi2), "0", True) != "")
    print(f"\n{N[0] - len(FAILS)}/{N[0]} PASS" + (f"; FAILED: {FAILS}" if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
