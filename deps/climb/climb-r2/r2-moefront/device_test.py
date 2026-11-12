#!/usr/bin/env python3
"""r2-moefront device harness: live route-prep chain vs the fused kernel, bitwise, plus timings.

Run by the lead on a STOPPED node (no serving process on its GPU), e.g. spark-eight, in the serving venv:

    cd ~/climb-r2/r2-moefront    # this kit folder
    source ~/venvs/vllm032/bin/activate
    VLLM_USE_DEEP_GEMM=1 DSV41_MOE_FUSED_PERMUTE=1 DSV41_ROUTE_LIVE=skip \
        DSV41_SM121_TARGET_M32=1 DSV41_TARGET_M32_NATIVE_DIR=$HOME/dsv41-stage/m32/target-m32-native-v1 \
        DSV41_MOE_FC1_DEAL=deal DSV41_MOE_M32_G1=all DSV41_MOE_COMPACT=1 \
        python3 device_test.py [--quick] [--no-timing] [--no-fc1]

Expected runtime: ~4-8 min (Triton JIT of 5 SLOTS buckets x 3 front modes, ~25k launches, F: 3 fronts x (8 eager + 4
graph x 50 replays), 200-replay timings).
The kit module is loaded from the installed site file if present, else from files/ in this folder; the live
kernels always come from the installed vLLM (per_token_group_fp8_quant, dsv41_fused_permute = mfront plan+scatter,
dsv4_topk). Exit code 0 only when every comparison is bit-identical.

Cases (all compared on the raw bytes, first mismatching index reported):
  Q1  every non-NaN bf16 x (all 65,536 patterns minus NaN/inf) at every reachable UE8M0 scale (|x| <= group absmax)
  Q2  every finite bf16 group absmax (both signs) -> packed scale words
  P   routes from the live dsv4_topk (fp32 logits incl. ties, +-20 boundary, large) and synthetic hostile routings
      (same 6 experts for all tokens, all distinct, ROUTE_LIVE dead rows, all dead, one live, hot experts) for
      M = 1..96, 128, 170 with random and hostile x rows: grouped_layout [M_sum], expert_ends [384], dead-row scale
      words, and every live route's FP8 row + packed scale words through inv_perm (row order inside an expert
      segment is the only allowed difference)
  G   CUDA graph [dsv4_topk (PDL) -> fused kernel] for FC1_FRONT = 0 / pdl / pf2, 200 replays with poisoned outputs,
      each replay == eager reference
  W   the one-time precompile (_ready -> JITFunction.warmup of all 5 SLOTS buckets) succeeds on this GPU for
      FC1_FRONT = 0 / pdl / pf2 with the fake FC1 geometry (else the arm would log 'OFF (precompile failed ...)')
  F   FC1 = the production target_m32 deal kernel (PDL-launched: DeepGEMM set_pdl(True), checked) on random FP4
      weights, launched DIRECTLY after the fused kernel on the same stream (its PDL successor, as in serving), for
      FC1_FRONT = 0 / pdl / pf2 (pdl / pf2 = early griddepcontrol.launch_dependents, + L2 prefetch): eager, and as a
      CUDA graph [dsv4_topk -> fused -> FC1] with 50 poisoned replays; every live route's FC1 output row through
      inv_perm must equal the live chain's row bitwise (row independence of FC1 under the new in-segment order, and
      no early read of our outputs by FC1). A missing FC1 kernel is a FAIL, not a skip (unless --no-fc1, which then
      does NOT clear the rp-pdl / rp-pf gate).
  T   timings (median of 200, warm and cold-L2) for M in the serving set: live chain vs fused, with the top-k kernel
      in the graph (PDL chain) and without
"""
import argparse
import importlib.util
import os
import statistics
import sys
import time

os.environ.setdefault("DSV41_MOE_FUSED_PERMUTE", "1")
os.environ.setdefault("DSV41_ROUTE_LIVE", "skip")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E, TOPK, H, G = 384, 6, 5120, 128
FAILS = []
RNG = np.random.default_rng(20261108)


def say(*a):
    print(*a, flush=True)


def check(name, ok, detail=""):
    say(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


def first_diff(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    d = torch.nonzero(a != b)
    if d.numel() == 0:
        return ""
    i = int(d[0])
    return f"first mismatch at flat index {i}: {a[i].item()} vs {b[i].item()} ({d.numel()} total)"


# ------------------------------------------------------------------------------------------------ module loading
def load_rp(front):
    """A fresh dsv41_routeprep module object with DSV41_MOE_FC1_FRONT=front (module-level flags)."""
    os.environ["DSV41_MOE_ROUTEPREP"] = "1"
    os.environ["DSV41_MOE_FC1_FRONT"] = front
    import vllm.model_executor.layers.fused_moe as fm
    site = os.path.join(os.path.dirname(fm.__file__), "dsv41_routeprep.py")
    path = site if os.path.exists(site) else os.path.join(HERE, "files/vllm/model_executor/layers/fused_moe",
                                                          "dsv41_routeprep.py")
    spec = importlib.util.spec_from_file_location(f"dsv41_routeprep_{front}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, path


def m_sum(M):
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M_and_alignment
    ms, al = compute_aligned_M_and_alignment(M, TOPK, E, 32, None)
    assert al == 32, al
    return ms


# --------------------------------------------------------------------------------------------------- the chains
def live_chain(x, ids, Ms, aq_out):
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import dsv41_fused_permute
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    xq, xs = per_token_group_quant_fp8(x, G)  # == prepare(): _fp8_quantize -> per_token_group_quant_fp8(A, 128)
    out, sfp, layout, inv, cursor = dsv41_fused_permute(xq, xs, ids, E, aq_out, Ms, 32)
    return dict(out=out, sfp=sfp, layout=layout, inv=inv, ends=cursor, xq=xq, xs=xs)


def fused_chain(rp, x, ids, Ms, aq_out, w1=None, sf=None):
    out, sfp, layout, inv, al, ends = rp.fused_permute(x, rp.Deferred(w1, sf), ids, E, None, None, aq_out, Ms, 32,
                                                       G, False, True)
    return dict(out=out, sfp=sfp, layout=layout, inv=inv, ends=ends)


def compare(tag, ids, a, b, quiet_pass=True):
    ok = True
    det = []

    def eq(name, x, y):
        nonlocal ok
        d = first_diff(x, y)
        if d:
            ok = False
            det.append(f"{name}: {d}")

    eq("grouped_layout", a["layout"], b["layout"])
    eq("expert_ends", a["ends"], b["ends"])
    dead = a["layout"] < 0
    for lab, r in (("live", a), ("fused", b)):
        z = r["sfp"][dead]
        if z.numel() and bool((z != 0).any()):
            ok = False
            det.append(f"dead-row scale words not 0 ({lab})")
    t, k = torch.nonzero(ids >= 0, as_tuple=True)
    if t.numel():
        da, db = a["inv"][t, k].long(), b["inv"][t, k].long()
        eq("FP8 rows via inv_perm", a["out"].view(torch.uint8)[da], b["out"].view(torch.uint8)[db])
        eq("packed scale words via inv_perm", a["sfp"][da], b["sfp"][db])
        eq("segment of each route", b["layout"][db], ids[t, k].to(torch.int32))
        if "xq" in a:
            eq("fused rows == live quant rows", b["out"].view(torch.uint8)[db], a["xq"].view(torch.uint8)[t])
    if not ok or not quiet_pass:
        check(tag, ok, "; ".join(det))
    return ok


# ----------------------------------------------------------------------------------------------- input builders
def bf16_from_bits(bits_np):
    return torch.from_numpy(bits_np.astype(np.int16)).view(torch.bfloat16)


def all_finite_bf16_bits():
    b = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    e = (b >> 7) & 0xFF
    return b[e != 0xFF]


def scale_bits_np(absmax_f32):
    y = (np.maximum(absmax_f32.astype(np.float32), np.float32(1e-10)) / np.float32(448.0)).astype(np.float32)
    y = np.maximum(y, np.float32(1e-10))
    yb = y.view(np.uint32)
    e = yb & np.uint32(0x7F800000)
    return np.where((yb & np.uint32(0x7FFFFF)) != 0, e + np.uint32(0x800000), e).astype(np.uint32)


def q1_rows():
    """Groups: lane 0 = absmax A (the largest finite bf16 with scale s), lanes 1..127 = a sweep of every finite bf16
    x with |x| <= A. Returns bf16 bit rows [R, 5120]."""
    fin = all_finite_bf16_bits()
    fv = (fin.astype(np.uint32) << 16).view(np.float32)
    pos = fin[(fv > 0) | ((fin & 0x8000) == 0) & (fv == 0)]
    pv = (pos.astype(np.uint32) << 16).view(np.float32)
    sc = scale_bits_np(pv)
    groups = []
    for s in np.unique(sc):
        A = pos[sc == s].max()  # the largest finite bf16 absmax that maps to scale s
        Av = (np.array([A], dtype=np.uint32) << 16).view(np.float32)[0]
        xs = fin[np.abs(fv) <= Av]
        n = -(-xs.size // 127)
        pad = np.resize(xs, n * 127).reshape(n, 127)
        groups.append(np.concatenate([np.full((n, 1), A, dtype=np.uint16), pad], axis=1))
    g = np.concatenate(groups, axis=0)
    r = -(-g.shape[0] // 40)
    g = np.concatenate([g, np.zeros((r * 40 - g.shape[0], 128), dtype=np.uint16)], axis=0)
    return g.reshape(r, 5120), int(np.unique(sc).size)


def q2_rows():
    fin = all_finite_bf16_bits()
    g = np.zeros((fin.size, 128), dtype=np.uint16)
    g[:, 5] = fin
    r = -(-g.shape[0] // 40)
    g = np.concatenate([g, np.zeros((r * 40 - g.shape[0], 128), dtype=np.uint16)], axis=0)
    return g.reshape(r, 5120)


def hostile_x(M):
    kinds = []
    r = RNG.standard_normal((M, H)).astype(np.float32)
    kinds.append(r)
    kinds.append((r * np.exp2(RNG.integers(-60, 60, size=(M, 1)))).astype(np.float32))
    kinds.append((r * np.exp2(RNG.integers(-130, 120, size=(M, H)))).astype(np.float32))
    z = np.zeros((M, H), np.float32)
    z[:, ::2] = -0.0
    kinds.append(z)
    kinds.append(np.repeat(448.0 * np.exp2(RNG.integers(-30, 30, size=(M, 40))), 128, axis=1).astype(np.float32))
    kinds.append(np.full((M, H), 3.3895e38, np.float32) * RNG.choice([-1, 1], size=(M, H)).astype(np.float32))
    kinds.append((RNG.integers(1, 128, size=(M, H)) * 2.0 ** -133).astype(np.float32))
    k = kinds[RNG.integers(len(kinds))]
    with np.errstate(all="ignore"):
        k = np.where(np.isfinite(k), k, 0).astype(np.float32)
    return torch.from_numpy(k).to(torch.bfloat16)


def synth_ids(M, kind):
    if kind == "same":
        return np.tile(RNG.choice(E, TOPK, replace=False), (M, 1))
    if kind == "distinct":
        n = M * TOPK
        p = np.concatenate([RNG.permutation(E) for _ in range(n // E + 1)])
        return p[:n].reshape(M, TOPK)
    if kind == "alldead":
        return np.full((M, TOPK), -1)
    if kind == "onelive":
        ids = np.full((M, TOPK), -1)
        ids[RNG.integers(M)] = RNG.choice(E, TOPK, replace=False)
        return ids
    if kind == "hot":
        return np.stack([RNG.choice(np.r_[np.arange(8), RNG.choice(np.arange(8, E), TOPK, replace=False)], TOPK,
                                    replace=False) for _ in range(M)])
    raise KeyError(kind)


def topk_ids(M, dead_frac, dtype=torch.int32, hostile=False):
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
    logits = torch.randn(M, E, device="cuda", dtype=torch.float32) * 3
    if hostile:
        logits[:, ::7] = 20.0  # softplus branch boundary + ties
        logits[: M // 2, :] = logits[0]  # identical rows
        logits[:, 100:110] = 1e4
    bias = torch.randn(E, device="cuda", dtype=torch.float32) * 0.01
    pad = (torch.rand(M, device="cuda") < dead_frac)
    w, ids = dsv4_topk(logits, bias, dtype, 2.5, is_padding=pad)
    return ids, (logits, bias, pad)


# ------------------------------------------------------------------------------------------------------- cases
def run_rows(rp, rows_bits, name):
    """Quantize bf16 rows through both chains, M <= 96 per launch, 6 distinct experts per token."""
    x_all = bf16_from_bits(rows_bits).cuda()
    ok = True
    for i in range(0, x_all.shape[0], 96):
        x = x_all[i:i + 96].contiguous()
        M = x.shape[0]
        ids = torch.from_numpy(synth_ids(M, "distinct")).to(torch.int32).cuda()
        Ms = m_sum(M)
        a = live_chain(x, ids, Ms, torch.empty((Ms, H), device="cuda", dtype=torch.float8_e4m3fn))
        b = fused_chain(rp, x, ids, Ms, torch.empty((Ms, H), device="cuda", dtype=torch.float8_e4m3fn))
        ok &= compare(f"{name} rows {i}..{i + M - 1}", ids, a, b)
    return ok


def case_quant(rp):
    rows, nsc = q1_rows()
    check(f"Q1 every finite bf16 x at each of {nsc} reachable scales ({rows.shape[0]} rows)",
          run_rows(rp, rows, "Q1"))
    rows = q2_rows()
    check(f"Q2 every finite bf16 group absmax -> scale words ({rows.shape[0]} rows)", run_rows(rp, rows, "Q2"))


def case_plan(rp, quick):
    Ms_list = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 36, 48, 64, 96, 170] if quick else \
        list(range(1, 97)) + [128, 170]
    n_ok = n = 0
    for M in Ms_list:
        Ms = m_sum(M)
        for kind in ("topk", "topk-dead", "topk-hostile", "topk-i64", "same", "distinct", "alldead", "onelive",
                     "hot"):
            if kind.startswith("topk"):
                ids, _ = topk_ids(M, 0.4 if kind == "topk-dead" else 0.0,
                                  torch.int64 if kind == "topk-i64" else torch.int32, kind == "topk-hostile")
            else:
                ids = torch.from_numpy(synth_ids(M, kind)).to(torch.int32).cuda()
            x = hostile_x(M).cuda() if RNG.random() < 0.5 else torch.randn(M, H, device="cuda").to(torch.bfloat16)
            a = live_chain(x, ids, Ms, torch.empty((Ms, H), device="cuda", dtype=torch.float8_e4m3fn))
            b = fused_chain(rp, x, ids, Ms, torch.empty((Ms, H), device="cuda", dtype=torch.float8_e4m3fn))
            n += 1
            n_ok += compare(f"P M={M} {kind}", ids, a, b)
    check(f"P route plan/scatter, {n} cases", n_ok == n, f"{n - n_ok} failed")


def fake_fc1_weights():
    w1 = torch.randint(-128, 128, (E, 576, H // 2), device="cuda", dtype=torch.int8)
    sf = torch.empty_strided((E, 576, H // 128), (576 * (H // 128), 1, 576), device="cuda", dtype=torch.int32)
    sf.fill_(0x7F7F7F7F)  # UE8M0 2^0 in every byte
    return w1, sf


def case_graph(fronts, w1, sf):
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
    for front in fronts:
        rp, _ = load_rp(front)
        for M in (1, 4, 6, 16, 96):
            Ms = m_sum(M)
            logits = torch.randn(M, E, device="cuda") * 3
            bias = torch.randn(E, device="cuda") * 0.01
            pad = torch.zeros(M, dtype=torch.bool, device="cuda")
            pad[M // 3] = M > 2
            x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
            aq_out = torch.empty((Ms, H), device="cuda", dtype=torch.float8_e4m3fn)
            # eager reference (live chain on the same routes)
            _, ids_e = dsv4_topk(logits, bias, torch.int32, 2.5, is_padding=pad)
            ref = live_chain(x, ids_e, Ms, torch.empty_like(aq_out))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):  # warm the JIT outside capture
                    _, ids_w = dsv4_topk(logits, bias, torch.int32, 2.5, is_padding=pad)
                    fused_chain(rp, x, ids_w, Ms, aq_out, w1, sf)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _, ids_g = dsv4_topk(logits, bias, torch.int32, 2.5, is_padding=pad)
                out = fused_chain(rp, x, ids_g, Ms, aq_out, w1, sf)
            ok = True
            for rep in range(200):
                out["layout"].fill_(0x7777)
                out["sfp"].fill_(0x6666)
                out["inv"].fill_(-7)
                out["ends"].fill_(-9)
                aq_out.view(torch.uint8).fill_(0x55)
                ids_g.fill_(0)
                g.replay()
                torch.cuda.synchronize()
                if not torch.equal(ids_g, ids_e):
                    ok = False
                    say(f"   replay {rep}: top-k ids differ from eager")
                    break
                if not compare(f"G front={front} M={M} replay {rep}", ids_e, ref, out):
                    ok = False
                    break
            check(f"G graph [dsv4_topk(PDL) -> fused] front={front} M={M}: 200 poisoned replays bitwise", ok)
            del g


def case_precompile(w1, sf):
    for front in ("0", "pdl", "pf2"):
        rp, _ = load_rp(front)
        ok = rp._ready(w1, sf, torch.int32, torch.device("cuda"))
        pf_kb = rp._pf_args(w1, sf, E, None)[0]
        check(f"W precompile of all SLOTS buckets on this GPU (FC1_FRONT={front}, prefetch k-blocks {pf_kb})",
              ok is True and rp._READY is True and pf_kb == rp.PF_KB)


def fc1_kernel():
    """The production FC1 (target_m32 deal tree) and whether DeepGEMM launches it with PDL."""
    from vllm.model_executor.layers.fused_moe.experts.sm121_target_m32 import get_target_m32_kernel
    k = get_target_m32_kernel()
    pdl = bool(k.native.get_pdl()) if hasattr(k.native, "get_pdl") else None
    return k.gemm, k.alignment_scope, pdl


def fc1(gemm, scope, r, w1, sf, Ms, mm1=None):
    if mm1 is None:
        mm1 = torch.full((Ms, 576), float("nan"), device="cuda", dtype=torch.bfloat16)
    with scope():
        gemm((r["out"], r["sfp"]), (w1, sf), mm1, r["layout"], recipe_a=(1, 128), recipe_b=(1, 32))
    return mm1


def fc1_rows(mm1, r, ids):
    t, kk = torch.nonzero(ids >= 0, as_tuple=True)
    return mm1[r["inv"][t, kk].long()].view(torch.int16)


def case_fc1(w1, sf):
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
    try:
        gemm, scope, pdl = fc1_kernel()
    except Exception as exc:  # noqa: BLE001
        check("F production FC1 (target_m32) importable", False, repr(exc)[:300])
        return
    check("F production FC1 is PDL-launched (DeepGEMM get_pdl)", pdl is True, f"get_pdl={pdl}")
    for front in ("0", "pdl", "pf2"):
        rp, _ = load_rp(front)
        n = ok = 0
        for M in (1, 2, 4, 6, 8, 16, 36, 96):
            Ms = m_sum(M)
            ids, _ = topk_ids(M, 0.25)
            x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
            a = live_chain(x, ids, Ms, torch.empty((Ms, H), device="cuda", dtype=torch.float8_e4m3fn))
            ref = fc1_rows(fc1(gemm, scope, a, w1, sf, Ms), a, ids)
            torch.cuda.synchronize()
            # eager: FC1 launched right behind the fused kernel (no sync, no other launch in between)
            b = fused_chain(rp, x, ids, Ms, torch.empty((Ms, H), device="cuda", dtype=torch.float8_e4m3fn), w1, sf)
            mm1 = fc1(gemm, scope, b, w1, sf, Ms)
            torch.cuda.synchronize()
            n += 1
            d = first_diff(ref, fc1_rows(mm1, b, ids))
            ok += not d
            if d:
                check(f"F eager FC1 rows via inv_perm front={front} M={M}", False, d)
        check(f"F eager [fused(front={front}) -> FC1] rows bitwise through inv_perm ({n} M values)", ok == n)
        # graph: [dsv4_topk (PDL) -> fused (PDL; early trigger for pdl/pf2) -> FC1 (PDL)], poisoned replays
        for M in (1, 6, 16, 96):
            Ms = m_sum(M)
            logits = torch.randn(M, E, device="cuda") * 3
            bias = torch.randn(E, device="cuda") * 0.01
            pad = torch.zeros(M, dtype=torch.bool, device="cuda")
            pad[M // 3] = M > 2
            x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
            _, ids_e = dsv4_topk(logits, bias, torch.int32, 2.5, is_padding=pad)
            a = live_chain(x, ids_e, Ms, torch.empty((Ms, H), device="cuda", dtype=torch.float8_e4m3fn))
            ref = fc1_rows(fc1(gemm, scope, a, w1, sf, Ms), a, ids_e)
            aq_out = torch.empty((Ms, H), device="cuda", dtype=torch.float8_e4m3fn)
            mm1 = torch.empty((Ms, 576), device="cuda", dtype=torch.bfloat16)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    _, ids_w = dsv4_topk(logits, bias, torch.int32, 2.5, is_padding=pad)
                    fc1(gemm, scope, fused_chain(rp, x, ids_w, Ms, aq_out, w1, sf), w1, sf, Ms, mm1)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _, ids_g = dsv4_topk(logits, bias, torch.int32, 2.5, is_padding=pad)
                out = fused_chain(rp, x, ids_g, Ms, aq_out, w1, sf)
                fc1(gemm, scope, out, w1, sf, Ms, mm1)
            good = True
            for rep in range(50):
                mm1.fill_(float("nan"))
                out["layout"].fill_(0x7777)
                out["sfp"].fill_(0x6666)
                out["inv"].fill_(-7)
                aq_out.view(torch.uint8).fill_(0x55)
                ids_g.fill_(0)
                g.replay()
                torch.cuda.synchronize()
                d = "top-k ids differ from eager" if not torch.equal(ids_g, ids_e) else \
                    first_diff(ref, fc1_rows(mm1, out, ids_e))
                if d:
                    good = False
                    say(f"   F graph front={front} M={M} replay {rep}: {d}")
                    break
            check(f"F graph [dsv4_topk -> fused(front={front}) -> FC1] M={M}: 50 poisoned replays bitwise", good)
            del g


def bench(fn, flush_buf, n=200, cold=False):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    t = []
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(n):
        if cold:
            flush_buf.add_(1)  # 64 MiB read+write: evicts the 24 MiB L2
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        t.append(st.elapsed_time(en) * 1000.0)
    del g
    return statistics.median(t)


def case_timing(w1, sf):
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
    flush = torch.zeros(16 * 1024 * 1024, device="cuda", dtype=torch.float32)
    rps = {f: load_rp(f)[0] for f in ("0", "pdl", "pf2")}
    say("T  median us over 200 graph replays (one chain per replay)")
    say("T  M   | live q+plan+scatter | fused | fused-pdl | fused-pf2 || with top-k: live | fused | pdl | pf2  [warm / cold-L2]")
    for M in (1, 2, 4, 6, 8, 12, 16, 24, 36, 48, 64, 96):
        Ms = m_sum(M)
        logits = torch.randn(M, E, device="cuda") * 3
        bias = torch.randn(E, device="cuda") * 0.01
        pad = torch.zeros(M, dtype=torch.bool, device="cuda")
        x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
        aq_out = torch.empty((Ms, H), device="cuda", dtype=torch.float8_e4m3fn)
        _, ids = dsv4_topk(logits, bias, torch.int32, 2.5, is_padding=pad)
        row = []
        for cold in (False, True):
            cells = [bench(lambda: live_chain(x, ids, Ms, aq_out), flush, cold=cold)]
            for f, rp in rps.items():
                cells.append(bench(lambda rp=rp: fused_chain(rp, x, ids, Ms, aq_out, w1, sf), flush, cold=cold))

            def tk_live():
                _, i2 = dsv4_topk(logits, bias, torch.int32, 2.5, is_padding=pad)
                live_chain(x, i2, Ms, aq_out)
            cells.append(bench(tk_live, flush, cold=cold))
            for f, rp in rps.items():
                def tk_f(rp=rp):
                    _, i2 = dsv4_topk(logits, bias, torch.int32, 2.5, is_padding=pad)
                    fused_chain(rp, x, i2, Ms, aq_out, w1, sf)
                cells.append(bench(tk_f, flush, cold=cold))
            row.append(" ".join(f"{c:7.2f}" for c in cells))
        say(f"T  {M:3d} | warm {row[0]} | cold {row[1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="P on 17 M values instead of 98")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-fc1", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    assert torch.cuda.is_available(), "needs a GPU (stopped node)"
    say(f"device {torch.cuda.get_device_name()} cc {torch.cuda.get_device_capability()}; "
        f"DSV41_MOE_FUSED_PERMUTE={os.environ.get('DSV41_MOE_FUSED_PERMUTE')} "
        f"DSV41_ROUTE_LIVE={os.environ.get('DSV41_ROUTE_LIVE')}")
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
    check("prepare() quant runs with UE8M0 scales (is_deep_gemm_e8m0_used)", bool(is_deep_gemm_e8m0_used()))
    rp, path = load_rp("0")
    say(f"kit module: {path}")
    check("module static check (_static_ok: UE8M0, e4m3fn +-448, mfront=1)", rp._static_ok())
    w1, sf = fake_fc1_weights()
    case_precompile(w1, sf)
    case_quant(rp)
    case_plan(rp, args.quick)
    case_graph(("0", "pdl", "pf2"), w1, sf)
    if not args.no_fc1:
        case_fc1(w1, sf)
    else:
        say("NOTE --no-fc1: case F not run; this run does not clear the rp-pdl / rp-pf gate")
    if not args.no_timing:
        case_timing(w1, sf)
    say(f"{'ALL BIT-IDENTICAL' if not FAILS else 'FAILED: ' + ', '.join(FAILS[:20])} ({time.time() - t0:.0f}s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
