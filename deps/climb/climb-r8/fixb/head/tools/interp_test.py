#!/usr/bin/env python3
"""r6-head: the REAL Triton kernel sources (live dsv41_r4_small base copy and dsv41_head_r6) run side by side in the
Triton interpreter on CPU (TRITON_INTERPRET=1, CUDA_VISIBLE_DEVICES=), small shapes. Logic check only: tl.dot is
numpy float32 matmul here, not the GPU MMA, so this checks indexing, masks, the list, the persistent loop, the
-inf fill, the self-check and the argmax path, with the dot as one fixed function shared by both sides.

    cd ~/climb-scratch/r6-head && CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 ~/venvs/vllm032/bin/python \
        tools/interp_test.py

Cases: V in {1000, 1013} (partial tile / partial CTA), K = 512, M in {2, 7, 16, 17, 33}; random, forced ties across
tiles / inside a tile, all rows identical, a bad row, padding rows (vreal < V). For each:
  screen DQ=1 and DQ=2: HI and THR bit-identical to the live screen;
  refine tile: output bit-identical to the live refine; row: == where(row predicate, live, -inf), list length;
  leftmost argmax through the live _sf_local_argmax_kernel == stock leftmost argmax (forced ties: the leftmost row);
  the module's own certified_logits (self-check path, then the steady path) returns the r6 result and logs a pass.
Numerics: Triton 3.7.1's interpreter evaluates a bf16 tl.dot on the raw bf16 bit patterns (the live screen and
refine both return ~5e11 for |logit| < 2: measured, results/interp.log + README), so NO check here depends on numeric
values: every check is a live-vs-r6 consistency check on the same interpreter (HI / THR bits of the DQ screens, the
tile-mode output bits, the row-mode output == where(row predicate, live, -inf), the list length, the self-check and
public-entry path). The numeric-gate INFO line per case records that the interpreter's live screen is far from the
torch reference; stock-argmax checks are skipped in such cases (all of them, with this Triton). The certificate,
tie and argmax logic with valid numerics is tests/test_refine_logic.py (torch CPU); real numerics: device_test.py.
Must-fail mutants (each must be caught at every M; logic mutants only, numeric ones belong to device_test.py):
refine skips the last K chunk; cand predicate `>` instead of `>=`
(caught by an exact-tie case where hi == T); screen exponent of the next 32-group; the first list entry of each CTA replaced by its left neighbour.
"""
import importlib.util
import os
import re
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["DSV41_R6_HEAD_REFINE"] = "row"
os.environ["DSV41_R6_HEAD_SCREEN"] = "1"
os.environ["DSV41_R6_HEAD_REFINE_CFG"] = "0,0,0,7,1"  # grid 7 (no device properties on CPU)
os.environ["DSV41_R6_HEAD_STATS_EVERY"] = "3"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

FAILS, N = [], [0]


def check(name, ok, detail=""):
    N[0] += 1
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R4P = os.path.join(HERE, "base/vllm/models/deepseek_v41/nvidia/dsv41_r4_small.py")
R6P = os.path.join(HERE, "files/vllm/models/deepseek_v41/nvidia/dsv41_head_r6.py")
r4 = load("vllm.models.deepseek_v41.nvidia.dsv41_r4_small", R4P)
r6 = load("vllm.models.deepseek_v41.nvidia.dsv41_head_r6", R6P)
r6._S6["nsm"] = 2


def mutant(tag, old, new):
    """A copy of dsv41_head_r6.py with one textual mutation (asserted to apply exactly once), as a fresh module."""
    src = open(R6P).read()
    assert src.count(old) == 1, (tag, src.count(old))
    d = tempfile.mkdtemp(prefix="r6mut-")
    p = os.path.join(d, f"dsv41_head_r6_{tag}.py")
    open(p, "w").write(src.replace(old, new))
    m = load(f"r6mut_{tag}", p)
    m._S6["nsm"] = 2
    return m


MUTANTS = {
    "kshort": ("            for k0 in range(0, K, BK):\n                k = k0 + tl.arange(0, BK)\n"
               "                w = tl.load(W + j64",
               "            for k0 in range(0, K - BK, BK):\n                k = k0 + tl.arange(0, BK)\n"
               "                w = tl.load(W + j64"),
    "strict": ("hrow = tl.max((h >= T[:, None])", "hrow = tl.max((h > T[:, None])"),
    "egroup": ("(k0 // 32 + tl.arange(0, G32))[:, None]", "((k0 // 32 + 1 + tl.arange(0, G32)) % (K // 32))[:, None]"),
    "dropfirst": ("tl.store(IDX + pos, j, mask=sel)",
                  "tl.store(IDX + pos, tl.where((pos == base) & (j > 0), j - 1, j), mask=sel)"),
}


def bits(t):
    return t.contiguous().view(torch.int16) if t.element_size() == 2 else t.contiguous().view(torch.int32)


def case(kind, M, V, K=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(V, K, generator=g) * 0.02).to(torch.bfloat16)
    x = torch.randn(M, K, generator=g).to(torch.bfloat16)
    vreal, ties = None, []
    if kind in ("tie_tiles", "tie_tile"):
        a, b = (37, 37 + 16 * 20 + 3) if kind == "tie_tiles" else (32, 41)
        W[b] = W[a]
        x = (W[a].float() * 400.0).repeat(M, 1).to(torch.bfloat16)
        ties = [a, b]
    elif kind == "identical":
        W[:] = W[0]
    elif kind == "bad":
        x[M - 1, 3] = float("inf")
    elif kind == "padding":
        vreal = V - 40
    return W, x, vreal, ties


def run_case(mod, kind, M, V, refine, screen, dq=1):
    W, x, vreal, ties = case(kind, M, V)
    st = r4.build_cert_tables(W, vreal=vreal)
    mod.SCREEN_DQ = dq
    mod.SCREEN = screen
    live = mod._run(st, x, False, "0")
    s6 = mod._state6(st, x.device)
    new = mod._run(st, x, screen and s6["dq_ok"], refine)
    return st, x, live, new, ties


def numerics_ok(st, x, live):
    """The interpreter's live screen agrees with the torch reference up to accumulation-order noise (the reference
    sums in a different fp32 order, so bit equality is not expected): |diff| <= 2^-5 |ref| + 2^-5 on finite HI/THR."""
    lo, hi, T, bad = r4.reference_bounds(st, x)
    a, b = live["hi"].float(), hi
    f = torch.isfinite(a) & torch.isfinite(b)
    ok = bool(torch.equal(torch.isfinite(a), torch.isfinite(b)))
    ok = ok and bool(((a - b).abs()[f] <= (b.abs()[f] * 2 ** -5 + 2 ** -5)).all())
    ft = torch.isfinite(T) & torch.isfinite(live["thr"])
    return ok and bool(((live["thr"] - T).abs()[ft] <= (T.abs()[ft] * 2 ** -5 + 2 ** -5)).all())


def stock_argmax(x, st):
    ex = (x.double() @ st.W.double().t()).to(torch.bfloat16).float()
    ex[:, st.vreal:] = float("-inf")
    return ex.argmax(dim=-1)


def sf_argmax(out, st):
    pair = r4.sf_local_pairs(out, st.vreal, 0)
    return pair[:, 1].long()


def main():
    kinds = ["random", "tie_tiles", "tie_tile", "identical", "bad", "padding"]
    for V in (1000, 1013):
        for M in (2, 7, 16, 17, 33):
            for kind in kinds:
                for refine in ("tile", "row"):
                    st, x, live, new, ties = run_case(r6, kind, M, V, refine, True, dq=1 if refine == "tile" else 2)
                    err = r6.compare(live, new, refine, True)
                    valid = numerics_ok(st, x, live)
                    if refine == "tile":
                        print(f"INFO V={V} M={M} {kind}: interpreter numerics {'valid' if valid else 'INVALID (live screen far from the torch reference): stock-argmax checks skipped'}")
                    if not valid:
                        # argmax may legitimately differ in row mode when the (interpreter) certificate is broken
                        err = r6.compare(live, dict(new, out=live["out"]) if refine == "row" else new, "tile" if refine == "tile" else "0", True) if err.startswith("argmax") else err
                    check(f"V={V} M={M} {kind} refine={refine} screen DQ{1 if refine == 'tile' else 2}", err == "", err)
                    if kind != "bad" and valid:
                        got = sf_argmax(new["out"], st)
                        ref = sf_argmax(live["out"], st)
                        ok = torch.equal(got, ref)
                        if ties:
                            ok = ok and bool((got == min(ties)).all())
                        check(f"V={V} M={M} {kind} refine={refine}: sf leftmost argmax == live"
                              + (f" == tied row {min(ties)}" if ties else ""), ok, f"{got.tolist()} vs {ref.tolist()}")
    # steady + self-check path through the public entry
    W, x, vreal, ties = case("tie_tiles", 7, 1000)
    st = r4.build_cert_tables(W)
    r6._S6.update(dead=False, checked=set(), calls=0, logged=False)
    r6.SCREEN, r6.SCREEN_DQ, r6.REFINE = True, 2, "row"
    o1 = r6.certified_logits(st, x)
    o2 = r6.certified_logits(st, x)
    check("certified_logits: self-check passed at M=7 and the r6 path stays on", 7 in r6._S6["checked"]
          and not r6._S6["dead"] and torch.equal(bits(o1), bits(o2)))
    check("certified_logits: forced tie -> leftmost tied row", bool((sf_argmax(o2, st) == min(ties)).all()))
    check("certified_logits: == the live certified_logits where the row predicate holds",
          r6.compare(r6._run(st, x, False, "0"), dict(r6._run(st, x, True, "row"), out=o2), "row", True) == "")
    # mutants: each must be caught at every M
    for tag, (old, new_) in MUTANTS.items():
        m = mutant(tag, old, new_)
        for M in (2, 7, 16, 17, 33):
            caught = []
            for kind in (["tie_exact"] if tag == "strict" else ["random", "tie_tiles"]):
                if tag == "strict":
                    # exact tie at T: a row whose hi equals T exactly is the argmax row's twin; make x hit a row with
                    # hi == T by forcing T := hi of a chosen row (same HI both sides), then the predicate differs.
                    W, x, vreal, ties = case("random", M, 1000)
                    st = r4.build_cert_tables(W)
                    dbg = {}
                    live = r6._run(st, x, False, "0")
                    hi, thr = live["hi"], live["thr"]
                    # choose the per-row max hi column and set T to it exactly: >= keeps it, > drops it
                    thr_new = hi.float().max(dim=1).values.contiguous()
                    a = run_cand(r6, hi, thr_new, live["bad"], st, x)
                    b = run_cand(m, hi, thr_new, live["bad"], st, x)
                    caught.append(not torch.equal(bits(a), bits(b)))
                    continue
                st, x, live, new, ties = run_case(m, kind, M, 1000, "row" if tag == "dropfirst" else "tile", True,
                                                  dq=1)
                err = r6.compare(live, new, "row" if tag == "dropfirst" else "tile", True)
                caught.append(err != "")
            check(f"MUTANT {tag} caught at M={M}", all(caught), str(caught))
    print(f"\n{N[0] - len(FAILS)}/{N[0]} PASS" + (f"; FAILED: {FAILS}" if FAILS else ""))
    return 1 if FAILS else 0


def run_cand(mod, hi, thr, bad, st, x):
    """cand + refine of `mod` on given HI/THR (row mode), output tensor."""
    k6 = mod._kernels6()
    M, K = x.shape
    V = st.V
    BMX = max(16, r4.next_pow2(M))
    out = torch.empty((M, V), dtype=torch.bfloat16)
    idx = torch.empty((V,), dtype=torch.int32)
    cnt = torch.zeros((1,), dtype=torch.int32)
    need = torch.zeros((1,), dtype=torch.int32)
    stats = torch.zeros((2,), dtype=torch.int64)
    cb = mod.cand_rows(BMX, 16)
    k6["cand"][((V + cb - 1) // cb,)](hi, thr, bad, out, out.stride(0), idx, cnt, need, stats, M, V,
                                      BMX=BMX, CB=cb, TB=16, TILE=False, num_warps=4)
    k6["refine"][(7,)](x, x.stride(0), st.W, out, out.stride(0), idx, cnt, M, 7, K=K, BV=16, BMX=BMX, BK=128,
                       SPREAD=True, num_warps=4, num_stages=3)
    return out


if __name__ == "__main__":
    sys.exit(main())
