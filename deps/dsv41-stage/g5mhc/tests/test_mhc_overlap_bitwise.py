#!/usr/bin/env python3
"""Bitwise gate for DSV41_MHC_OVERLAP on ONE Spark (GB10, sm_121), in the live vLLM venv, no server needed.

Every comparison is bit-for-bit (int view of each tensor, so -0.0 / NaN payloads count); there is no ulp
tolerance anywhere (vLLM-Moet's own test_mhc_post_norm_sm120.py tolerates 1 ulp on layer_input / aux).

  pieces  S1 mhc_post_tilelang bf16 == mhc_fused_tilelang bf16 residual            (M <= 32)
          S2 pre_norm "input" layer_input / aux == "fused" layer_input / aux
          S3 pre_norm "stats" post / comb / next_pre == "fused" ones
          S4 narrow projection (tile_n=24) partials == served partials               (M <= 32)
          S5 Moet mhc_proj partials == served partials                               (--moet, M <= 32)
          S6 Moet mhc_post_norm, contraction modes 0/1/2, vs S1/S2 references        (--moet)
  boundary  served mhc_shifted_post_pre(stream=None) vs the overlap dispatcher (stream=side, joined),
            eager AND captured in a CUDA graph replayed with two fresh input sets, every output:
            residual, post, comb, layer_input, next_pre, aux; capture_aux False/True
  chain     five boundaries threaded like the decoder (boundary -> stand-in sublayer -> join), eager + graph
  pre       (--pre) vLLM mhc_pre_delayed_overlap vs mhc_pre_delayed_tilelang: first layer (broadcast) and Engram
  inputs    synthetic (Moet-style scales) and, with --dump DIR, recorded prefill boundaries dumped by a
            DSV41_MHC_OVERLAP_DUMP=DIR launch (rows sliced to each M; plus a strided row subset)
  M         1 2 4 5 6 8 12 16 20 24 32 (fused range) and 33 48 64 (post + tf32 range)

usage:
  python3 test_mhc_overlap_bitwise.py [--helper PATH/dsv41_mhc_overlap.py] [--modes stock[,moet]]
         [--proj fused,narrow[,moet]] [--moet-dir DIR] [--dump DIR] [--pre] [--bench] [--cold]
exit 0 = every check bitwise; 1 = a mismatch (printed with the count and max ulp); 2 = setup error.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import statistics
import sys

import torch

H, HC, MIX = 5120, 4, 24
EPS = (1e-20, 1e-6, 1e-6, 2.0, 20)  # rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat
NORM_EPS = 1e-20
SIZES = (1, 2, 4, 5, 6, 8, 12, 16, 20, 24, 32, 33, 48, 64)
FAILS = []


# ------------------------------------------------------------------------------------------ utils
def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    return t


def same(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, f"shape/dtype {tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}"
    if a.numel() == 0:
        return True, "empty"
    da, db = bits(a), bits(b)
    if torch.equal(da, db):
        return True, "="
    diff = (da.long() - db.long()).abs()
    return False, f"{int((diff != 0).sum())} differ, max {int(diff.max())} ulp"


def check(label: str, pairs) -> bool:
    ok, rep = True, []
    for name, a, b in pairs:
        s, msg = same(a, b)
        ok &= s
        rep.append(f"{name} {msg}")
    if not ok:
        FAILS.append(label)
    print(f"[{'ok ' if ok else 'BAD'}] {label}: " + ", ".join(rep))
    return ok


def synthetic(n: int, dev, seed: int):
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=dev, generator=g)  # noqa: E731
    u = lambda *s: torch.rand(*s, device=dev, generator=g)  # noqa: E731
    return dict(
        x=(r(n, H) * 0.5).to(torch.bfloat16),
        residual=(r(n, HC, H) * 0.8).to(torch.bfloat16),
        post_layer_mix=(u(n, HC, 1) * 2).float(),
        comb_res_mix=torch.softmax(r(n, HC, HC), dim=-1).float().contiguous(),
        pre_mix=(u(n, HC) * 0.9 + 0.05).float(),
        fn=(r(MIX, HC * H) * 0.01).float(),
        hc_scale=torch.tensor([0.02, 0.02, 0.05], device=dev),
        hc_base=(r(MIX) * 0.5).float(),
        norm_weight=(u(H) + 0.5).to(torch.bfloat16),
        norm_eps=NORM_EPS,
        eps=EPS,
    )


def recorded(dump_dir: str, dev):
    for path in sorted(glob.glob(os.path.join(dump_dir, "rank*", "b*.pt"))):
        blob = torch.load(path, map_location="cpu")
        yield os.path.basename(path), blob


def slice_rows(blob, rows: torch.Tensor, dev):
    out = dict(blob)
    for k in ("x", "residual", "post_layer_mix", "comb_res_mix", "pre_mix"):
        out[k] = blob[k][rows].contiguous().to(dev)
    for k in ("fn", "hc_scale", "hc_base", "norm_weight"):
        out[k] = blob[k].to(dev)
    return out


def served(ops, t, capture_aux):
    e = t["eps"]
    return ops.mhc_shifted_post_pre(
        t["x"], t["residual"], t["post_layer_mix"], t["comb_res_mix"], t["fn"], t["hc_scale"], t["hc_base"],
        e[0], e[1], e[2], e[3], e[4], pre_mix=t["pre_mix"], norm_weight=t["norm_weight"],
        norm_eps=t["norm_eps"], capture_aux=capture_aux, stream=None,
    )


def overlapped(ovl, t, capture_aux, side):
    e = t["eps"]
    outs = ovl.shifted_post_pre_overlap(
        t["x"], t["residual"], t["post_layer_mix"], t["comb_res_mix"], t["fn"], t["hc_scale"], t["hc_base"],
        e[0], e[1], e[2], e[3], e[4], t["pre_mix"], t["norm_weight"], t["norm_eps"], capture_aux, side,
    )
    torch.cuda.current_stream().wait_stream(side)  # the decoder layer's join
    return outs


NAMES = ("residual", "post", "comb", "layer_input", "next_pre", "aux")


# ----------------------------------------------------------------------------------------- pieces
def pieces(ovl, t, n, capture_aux, moet):
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_FUSED_TILELANG_KERNEL,
        _MHC_POST_TILELANG_KERNEL,
        mhc_fused_post_pre_split_config,
    )
    from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL, compute_mhc_pre_num_splits

    e = t["eps"]
    dev = t["x"].device
    post2d = t["post_layer_mix"].view(n, HC)
    cfg = mhc_fused_post_pre_split_config(n, H, HC)
    res_post = torch.empty_like(t["residual"])
    _MHC_POST_TILELANG_KERNEL(t["comb_res_mix"], t["residual"], post2d, t["x"], res_post, HC, H)
    if cfg is not None:
        mixes, sqrsum, res_fused = _MHC_FUSED_TILELANG_KERNEL(
            t["comb_res_mix"], t["residual"], post2d, t["x"], t["fn"].view(MIX, HC, H), HC, H, MIX)
        check(f"S1 M={n:2d} post kernel vs fused kernel bf16 streams", [("residual", res_fused, res_post)])
        ym, ys, _ = ovl._proj_narrow(t["x"], t["residual"], t["post_layer_mix"], t["comb_res_mix"],
                                     t["fn"], HC, H, MIX)
        check(f"S4 M={n:2d} narrow tile_n={ovl.NARROW_TILE_N} partials vs served", [("mixes", mixes, ym),
                                                                                 ("sqrsum", sqrsum, ys)])
        if moet is not None and moet.mhc_proj_applicable(t["x"], t["residual"]):
            pm, ps = moet.mhc_proj(t["x"], t["residual"], t["post_layer_mix"], t["comb_res_mix"], t["fn"])
            check(f"S5 M={n:2d} Moet mhc_proj partials vs served", [("mixes", mixes, pm), ("sqrsum", sqrsum, ps)])
        ref_res = res_fused
    else:
        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

        ns = compute_mhc_pre_num_splits(HC * H, n)
        mixes = torch.empty(ns, n, MIX, device=dev)
        sqrsum = torch.empty(ns, n, device=dev)
        tf32_hc_prenorm_gemm(res_post.view(n, HC * H), t["fn"], mixes, sqrsum, ns)
        ref_res = res_post
    epi = dict(hidden_size=H, rms_eps=e[0], hc_pre_eps=e[1], hc_sinkhorn_eps=e[2], hc_post_mult_value=e[3],
               sinkhorn_repeat=e[4], norm_eps=t["norm_eps"], hc_mult=HC, use_pre_mix_in=True, save_pre_mix=True,
               rms_numel=HC * H)

    def epilogue(mode, write_aux):
        post = torch.empty(n, HC, device=dev)
        comb = torch.empty(n, HC * HC, device=dev)
        li = torch.empty(n, H, dtype=torch.bfloat16, device=dev)
        nxt = torch.empty(n, HC, device=dev)
        aux = torch.empty(n if write_aux else 0, H, dtype=torch.bfloat16, device=dev)
        MHC_PRE_NORM_KERNEL(mixes, sqrsum, t["hc_scale"], t["hc_base"], ref_res, post, comb, li, t["norm_weight"],
                            t["pre_mix"], nxt, aux if write_aux else li, split_mode=mode, write_aux=write_aux, **epi)
        return post, comb, li, nxt, aux

    f = epilogue("fused", capture_aux)
    i = epilogue("input", capture_aux)
    s = epilogue("stats", False)
    check(f"S2 M={n:2d} aux={int(capture_aux)} input vs fused", [("layer_input", f[2], i[2]), ("aux", f[4], i[4])])
    check(f"S3 M={n:2d} stats vs fused", [("post", f[0], s[0]), ("comb", f[1], s[1]), ("next_pre", f[3], s[3])])
    if moet is not None:
        for mode in (0, 1, 2):
            ro, li, aux = moet.mhc_post_norm(t["x"], t["residual"], t["post_layer_mix"], t["comb_res_mix"],
                                             t["pre_mix"], t["norm_weight"], t["norm_eps"],
                                             capture_aux=capture_aux, mode=mode)
            ok = all(same(a, b)[0] for a, b in ((ref_res, ro), (f[2], li), (f[4], aux)))
            tag = "S6" if mode == ovl.MOET_MODE else "s6"
            print(f"[{'ok ' if ok else '-- '}] {tag} M={n:2d} aux={int(capture_aux)} Moet mhc_post_norm mode {mode}: "
                  + ", ".join(f"{nm} {same(a, b)[1]}" for nm, a, b in (("residual", ref_res, ro),
                                                                       ("layer_input", f[2], li), ("aux", f[4], aux))))
            if tag == "S6" and not ok:
                FAILS.append(f"S6 M={n} mode {mode}")


# --------------------------------------------------------------------------------------- boundary
def graphed(build, statics, fresh_sets):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        build()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs = build()
    torch.cuda.current_stream().wait_stream(s)
    res = []
    for fresh in fresh_sets:
        for k, v in fresh.items():
            if isinstance(v, torch.Tensor) and k in statics:
                statics[k].copy_(v)
        g.replay()
        torch.cuda.synchronize()
        res.append(tuple(o.clone() for o in outs))
    return res


def boundary(ops, ovl, t, n, capture_aux, side, label, fresh_sets):
    ref = served(ops, t, capture_aux)
    eag = overlapped(ovl, t, capture_aux, side)
    torch.cuda.synchronize()
    check(f"{label} M={n:2d} aux={int(capture_aux)} eager", list(zip(NAMES, ref, eag)))
    statics = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
    reps = graphed(lambda: overlapped(ovl, statics, capture_aux, side), statics, fresh_sets)
    for i, (fresh, got) in enumerate(zip(fresh_sets, reps)):
        tt = dict(t)
        tt.update(fresh)
        r = served(ops, tt, capture_aux)
        torch.cuda.synchronize()
        check(f"{label} M={n:2d} aux={int(capture_aux)} graph replay {i}", list(zip(NAMES, r, got)))


def chain(ops, ovl, t, n, side, sub_w, graph: bool):
    """Five boundaries threaded like the decoder: boundary -> sublayer (a GEMM on layer_input) -> join."""
    e = t["eps"]

    def run(use_side):
        x, res, post, comb, pre = t["x"], t["residual"], t["post_layer_mix"], t["comb_res_mix"], t["pre_mix"]
        outs = []
        for k in range(5):
            fn = t["fn"] if k % 2 == 0 else t["fn2"]
            if use_side:
                o = ovl.shifted_post_pre_overlap(x, res, post, comb, fn, t["hc_scale"], t["hc_base"], e[0], e[1],
                                                 e[2], e[3], e[4], pre, t["norm_weight"], t["norm_eps"], k == 3, side)
            else:
                o = ops.mhc_shifted_post_pre(x, res, post, comb, fn, t["hc_scale"], t["hc_base"], e[0], e[1], e[2],
                                             e[3], e[4], pre_mix=pre, norm_weight=t["norm_weight"],
                                             norm_eps=t["norm_eps"], capture_aux=k == 3, stream=None)
            res, post, comb, li, pre, aux = o
            x = (li @ sub_w).to(torch.bfloat16)  # the sublayer, on the caller stream, while the side stream runs
            if use_side:
                torch.cuda.current_stream().wait_stream(side)  # the decoder's join after the sublayer
            outs += [res, post, comb, li, pre, aux]
        return tuple(outs)

    ref = run(False)
    got = run(True)
    torch.cuda.synchronize()
    names = [f"b{k}.{nm}" for k in range(5) for nm in NAMES]
    check(f"chain M={n:2d} eager (5 boundaries)", list(zip(names, ref, got)))
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run(True)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                gout = run(True)
        torch.cuda.current_stream().wait_stream(s)
        g.replay()
        torch.cuda.synchronize()
        check(f"chain M={n:2d} graph replay (5 boundaries)", list(zip(names, ref, gout)))


def pre_boundaries(t, n, side):
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang
    from vllm.models.deepseek_v41.nvidia.ops.mhc import mhc_pre_delayed_overlap

    e = t["eps"]
    names = ("post", "comb", "layer_input", "next_pre")
    dev = t["x"].device
    fn_b = (torch.randn(MIX, H, device=dev) * 0.02).float()
    for label in ("first layer (broadcast)", "Engram (hc stream, carried pre-mix)"):
        if label.startswith("first"):
            res = t["x"].unsqueeze(1).expand(-1, HC, -1).contiguous()
            args, kw = (res, fn_b), dict(x=t["x"], norm_weight=t["norm_weight"], norm_eps=t["norm_eps"])
        else:
            args, kw = (t["residual"], t["fn"]), dict(pre_mix=t["pre_mix"], norm_weight=t["norm_weight"],
                                                       norm_eps=t["norm_eps"])
        rest = (t["hc_scale"], t["hc_base"], e[0], e[1], e[2], e[3], e[4])
        ref = mhc_pre_delayed_tilelang(*args, *rest, **kw)
        got = mhc_pre_delayed_overlap(*args, *rest, **kw, stream=side)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        check(f"pre M={n:2d} {label} eager", list(zip(names, ref, got)))


# ------------------------------------------------------------------------------------------ bench
def bench(ops, ovl, dev, cold: bool):
    """Per-boundary cost inside a CUDA graph: served vs overlap, alone and followed by a stand-in sublayer
    (two weight-streaming GEMMs at M rows, ~7.9 MB each) with the join after it, as in the decoder layer."""
    w1 = (torch.randn(H, 768, device=dev) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(768, H, device=dev) * 0.02).to(torch.bfloat16)
    flush = torch.empty(48 << 20, dtype=torch.uint8, device=dev)
    side = torch.cuda.Stream()

    def timed(fn, reps=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
            s.synchronize()
            with torch.cuda.graph(g, stream=s):
                for _ in range(reps):
                    if cold:
                        flush.zero_()
                    fn()
        torch.cuda.current_stream().wait_stream(s)
        ts = []
        for _ in range(10):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000 / reps)
        return statistics.median(ts)

    t_flush = timed(lambda: None) if cold else 0.0
    print(f"bench ({'cold L2: 48 MB flush per rep, subtracted' if cold else 'warm L2'}); us per boundary")
    for n in (1, 4, 6, 12, 16, 20, 24, 48):
        t = synthetic(n, dev, 100 + n)
        e = t["eps"]

        def work(li):
            return ((li @ w1) @ w2).to(torch.bfloat16)

        def srv():
            o = served(ops, t, False)
            work(o[3])

        def ovl_run():
            o = ovl.shifted_post_pre_overlap(t["x"], t["residual"], t["post_layer_mix"], t["comb_res_mix"], t["fn"],
                                             t["hc_scale"], t["hc_base"], e[0], e[1], e[2], e[3], e[4], t["pre_mix"],
                                             t["norm_weight"], t["norm_eps"], False, side)
            work(o[3])
            torch.cuda.current_stream().wait_stream(side)

        t_work = timed(lambda: work(t["x"])) - t_flush
        t_srv = timed(lambda: served(ops, t, False)) - t_flush
        line = (f"  M={n:2d}: served boundary {t_srv:6.2f} | sublayer stand-in {t_work:6.2f} | served+sub "
                f"{timed(srv) - t_flush:6.2f} | overlap+sub:")
        for mode, proj in (("stock", "fused"), ("stock", "narrow")) + ((("moet", "fused"),) if ovl._MOET else ()):
            ovl.configure(MODE=mode, PROJ=proj)
            line += f" {mode}/{proj} {timed(ovl_run) - t_flush:6.2f}"
        print(line)
    ovl.configure(MODE="stock", PROJ="fused")


# ------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--helper", default="", help="path to dsv41_mhc_overlap.py (default: the installed module)")
    ap.add_argument("--modes", default="stock")
    ap.add_argument("--proj", default="fused,narrow")
    ap.add_argument("--moet-dir", default=os.path.expanduser(
        "~/dsv41-stage/moet/src/vllm-moet-b0dc737/tools/dsv41_sm120/mhc_overlap"))
    ap.add_argument("--moet-mode", type=int, default=1)
    ap.add_argument("--dump", default="")
    ap.add_argument("--pre", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--bench-only", action="store_true", help="g5mhc: skip every check, run --bench only")
    ap.add_argument("--recorded-only", action="store_true", help="g5mhc: --dump sources only (skip synthetic)")
    ap.add_argument("--sizes", default="", help="g5mhc: comma list overriding the M sweep")
    a = ap.parse_args()
    global SIZES
    if a.sizes:
        SIZES = tuple(int(v) for v in a.sizes.split(",") if v)

    os.environ.setdefault("DSV41_MHC_OVERLAP", "stock")
    os.environ["DSV41_MHC_OVERLAP_MOET_DIR"] = a.moet_dir
    if a.helper:
        spec = importlib.util.spec_from_file_location("vllm.models.deepseek_v41.nvidia.dsv41_mhc_overlap", a.helper)
        ovl = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ovl
        spec.loader.exec_module(ovl)
    else:
        from vllm.models.deepseek_v41.nvidia import dsv41_mhc_overlap as ovl
    from vllm.models.deepseek_v41.nvidia.ops import mhc as ops

    dev = torch.device("cuda")
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}, cc {torch.cuda.get_device_capability()}")
    modes = [m for m in a.modes.split(",") if m]
    projs = [p for p in a.proj.split(",") if p]
    moet = None
    if "moet" in modes or "moet" in projs:
        ovl.configure(MOET_DIR=a.moet_dir, MOET_MODE=a.moet_mode)
        try:
            moet = ovl._moet()
        except Exception as exc:  # noqa: BLE001
            print(f"Moet kernels unavailable: {exc!r}")
            return 2
    side = torch.cuda.Stream()
    if a.bench_only:
        ovl.configure(MODE="stock", PROJ="fused", MAX_TOKENS=64)
        bench(ops, ovl, dev, a.cold)
        print("BENCH DONE")
        return 0

    # 1. pieces on synthetic inputs (and recorded, below)
    for n in (() if a.recorded_only else SIZES):
        t = synthetic(n, dev, 1000 + n)
        for capture_aux in (False, True):
            pieces(ovl, t, n, capture_aux, moet)

    # 2. boundary: every (mode, proj) on synthetic + recorded inputs, eager + graph
    sources = ([] if a.recorded_only else [("synthetic", None)]) + (
        [(name, blob) for name, blob in recorded(a.dump, dev)] if a.dump else [])
    print(f"sources: {[s for s, _ in sources]}")
    for mode in modes:
        for proj in projs:
            if proj == "moet" and moet is None:
                continue
            ovl.configure(MODE=mode, PROJ=proj, MAX_TOKENS=max(SIZES))
            for src, blob in sources:
                for n in SIZES:
                    for variant in ("head", "strided"):
                        if blob is None:
                            if variant == "strided":
                                continue
                            t = synthetic(n, dev, 2000 + n)
                            fresh = [{k: v for k, v in synthetic(n, dev, 3000 + 7 * n + i).items()
                                      if k in ("x", "residual", "post_layer_mix", "comb_res_mix", "pre_mix")}
                                     for i in range(2)]
                        else:
                            rows_all = blob["x"].shape[0]
                            if rows_all < 2 * n:
                                continue
                            rows = (torch.arange(n) if variant == "head"
                                    else torch.arange(0, rows_all, max(1, rows_all // n))[:n])
                            t = slice_rows(blob, rows, dev)
                            t["eps"] = blob["eps"]
                            other = slice_rows(blob, torch.arange(rows_all - n, rows_all), dev)
                            fresh = [{k: other[k] for k in ("x", "residual", "post_layer_mix", "comb_res_mix",
                                                            "pre_mix")}]
                            if variant == "head":
                                for aux_ in (False, True):  # g5mhc: aux both ways on recorded rows too
                                    pieces(ovl, t, n, aux_, moet)
                        for capture_aux in (False, True):
                            boundary(ops, ovl, t, n, capture_aux, side, f"{mode}/{proj} {src}:{variant}", fresh)

    # 3. chain, synthetic, eager + graph
    for mode in (() if a.recorded_only else modes):
        ovl.configure(MODE=mode, PROJ=projs[0])
        for n in (1, 6, 12, 24, 48):
            t = synthetic(n, dev, 4000 + n)
            t["fn2"] = synthetic(n, dev, 5000 + n)["fn"]
            sub_w = (torch.randn(H, H, device=dev) * 0.01).to(torch.bfloat16)
            chain(ops, ovl, t, n, side, sub_w, graph=True)

    # 4. first-layer / Engram mhc_pre (DSV41_MHC_OVERLAP_PRE=1 path)
    if a.pre:
        for n in (1, 6, 12, 24, 48, 64):
            pre_boundaries(synthetic(n, dev, 6000 + n), n, side)

    if a.bench:
        ovl.configure(MODE="stock", PROJ="fused", MAX_TOKENS=64)
        bench(ops, ovl, dev, a.cold)

    print("ALL BITWISE" if not FAILS else f"{len(FAILS)} FAILURES: " + "; ".join(FAILS[:20]))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
