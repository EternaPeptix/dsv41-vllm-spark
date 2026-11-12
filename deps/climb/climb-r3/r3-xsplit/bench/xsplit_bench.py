#!/usr/bin/env python3
"""r3-xsplit microbenchmark (deliverable a): S0, c, barrier and write-back cost of the REAL fused
XSPLIT kernel against the live sparse_mla_prefill_mg_dual kernel, on the production shapes.

Run on ONE STOPPED node (same node/protocol as device_test.py), serving venv:

    cd ~/climb-r3/r3-xsplit && timeout 1800 ~/venvs/vllm032/bin/python3 -I bench/xsplit_bench.py \
        --lib-dir build --out out/bench.jsonl          # ~6-10 min

Arms (all compiled with the live JIT flags from the same source, src/build.sh):
  live      sparse_mla_prefill_mg_dual_kernel<DSV4,BF16,8,64,PBSX,1> from the live headers (grid T)
  a0r1      XSPLIT exact, rope-B written back by phase A          <- the shipped variant
  a0r0      XSPLIT exact, phase B re-reads rope-B from the KV cache (correction 3 alternative)
  a1r1      timing only: no prefix-max wait (wrong output)       -> barrier cost = a0r1 - a1r1
  a2r1      timing only: B jobs skip replay + epilogue            -> B cost      = a0r1 - a2r1
  a3r1      timing only: A skips the row stores                   -> write-back  = a0r1 - a3r1
  a0r1p0    XSPLIT exact without the B-replay row prefetch            -> prefetch gain

Two timing modes per (arm, variant, T, n):
  solo   a CUDA graph of 20 back-to-back calls, replayed 30x; us/call = graph time / 20
         (in-graph launch gaps, like the serving graphs)
  woa    eager single calls with the WOA l2pf stand-in launched on a side stream just before each
         call (grid 2 x 64, 5 segments, 4.32 MB, 16 KB chunks); median of 200 (event-timed)
Fits per (arm, variant, T): us = S0 + c * n (least squares over the n grid). For live this is
F + P * n. The step model at the end uses the trace census (20 dual64 calls at n = 10; 18 dual32
calls at n = 13 prose / 16 code / 18 maths; T = 2/4 prose, 4/6 code and maths).
"""
import argparse
import ctypes
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import device_test as dt  # noqa: E402

ARMS = ["a0r1", "a0r0", "a1r1", "a2r1", "a3r1", "a0r1p0"]
STEP_MS = {"code": 24.26, "prose": 21.33, "maths": 24.04}
AL = {"code": 2.926, "prose": 1.843, "maths": 3.613}
N32 = {"code": 16, "prose": 13, "maths": 18}
TMIX = {"code": (4, 6), "prose": (2, 4), "maths": (4, 6)}


def time_graph(fn, reps=30, per=20):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(per):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / per)
    del g
    return float(np.median(ts))


def time_woa(fn, l2pf, n=200):
    ts = []
    for i in range(n + 20):
        l2pf()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        if i >= 20:
            ts.append(a.elapsed_time(b) * 1000.0)
    return float(np.median(ts))


def fit(ns, us):
    A = np.vstack([np.ones(len(ns)), np.array(ns, dtype=np.float64)]).T
    (s0, c), *_ = np.linalg.lstsq(A, np.array(us, dtype=np.float64), rcond=None)
    return float(s0), float(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib-dir", default="build")
    ap.add_argument("--out", default="out/bench.jsonl")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fo = open(args.out, "w")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    libs = {a: dt.Lib(os.path.join(args.lib_dir, f"libdsv41_xsplit_bench_{a}.so"))
            for a in args.arms.split(",")}
    base = libs.get("a0r1") or next(iter(libs.values()))
    ws = {a: l.ws_new(dev) for a, l in libs.items()}
    side = torch.cuda.Stream()
    seg_buf = torch.empty(5 * 865 * 1024, dtype=torch.uint8, device=dev)
    segs = torch.tensor([[seg_buf.data_ptr() + i * 865 * 1024, 865 * 1024] for i in range(5)],
                        dtype=torch.int64, device=dev)

    def l2pf():
        side.wait_stream(torch.cuda.current_stream())
        rc = base.l.dsv41_bench_l2pf(ctypes.c_void_p(segs.data_ptr()), 5, 16384, 2,
                                     ctypes.c_void_p(side.cuda_stream))
        assert rc == 0

    variants = [(64, 512, list(range(2, 11))), (32, 1024, [2, 4, 6, 8, 10, 12, 13, 14, 16, 18])]
    Ts = [1, 2, 4, 6]
    if args.quick:
        variants = [(64, 512, [2, 6, 10]), (32, 1024, [2, 10, 18])]
        Ts = [2, 6]
    table = {}
    for pbsx, topkx, ns in variants:
        for T in Ts:
            for n in ns:
                d = dt.to_dev(dt.gen_inputs(11, pbsx, topkx, T, n, 0), dev)
                o, l = dt.new_out(T, dev)
                fns = {"live": lambda: base.live_bench(d, o, l)}
                for a, lib in libs.items():
                    fns[a] = (lambda lib=lib, w=ws[a]: lib.xsplit(d, o, l, w))
                for name, fn in fns.items():
                    solo = time_graph(fn)
                    woa = time_woa(fn, l2pf, n=60 if args.quick else 200)
                    table[(name, pbsx, T, n)] = (solo, woa)
                    fo.write(json.dumps(dict(arm=name, pbsx=pbsx, topkx=topkx, T=T, n=n,
                                             solo_us=round(solo, 3), woa_us=round(woa, 3))) + "\n")
                    fo.flush()
                print(f"pbsx={pbsx} T={T} n={n} " + " ".join(
                    f"{k}={table[(k, pbsx, T, n)][0]:.1f}/{table[(k, pbsx, T, n)][1]:.1f}"
                    for k in fns), flush=True)

    print("\n== fits: us/call = S0 + c*n  (solo | beside WOA)")
    fits = {}
    for pbsx, topkx, ns in variants:
        for T in Ts:
            for name in ["live"] + list(libs):
                so = fit(ns, [table[(name, pbsx, T, n)][0] for n in ns])
                wo = fit(ns, [table[(name, pbsx, T, n)][1] for n in ns])
                fits[(name, pbsx, T)] = (so, wo)
                print(f"  pbsx={pbsx:2d} T={T} {name:5s}  S0 {so[0]:6.2f} c {so[1]:5.2f} | "
                      f"S0 {wo[0]:6.2f} c {wo[1]:5.2f}")
                fo.write(json.dumps(dict(fit=name, pbsx=pbsx, T=T, s0_solo=so[0], c_solo=so[1],
                                         s0_woa=wo[0], c_woa=wo[1])) + "\n")

    if "a0r1" in libs:
        print("\n== component costs (mean over the n grid, beside WOA, us/call)")
        for pbsx, topkx, ns in variants:
            for T in Ts:
                def mean_d(x, y):
                    return float(np.mean([table[(x, pbsx, T, n)][1] - table[(y, pbsx, T, n)][1]
                                          for n in ns]))
                parts = {}
                for arm, label in (("a1r1", "barrier"), ("a2r1", "B_replay+epilogue"),
                                   ("a3r1", "writeback"), ("a0r0", "rope_ws_vs_reread"),
                                   ("a0r1p0", "b_prefetch_gain")):
                    if arm in libs:
                        parts[label] = round(mean_d("a0r1", arm), 2)
                parts["saving_vs_live"] = round(mean_d("live", "a0r1"), 2)
                print(f"  pbsx={pbsx} T={T} {parts}")
                fo.write(json.dumps(dict(components=parts, pbsx=pbsx, T=T)) + "\n")

        print("\n== step model (trace census; beside-WOA times for consumer calls)")
        for wl in ("prose", "code", "maths"):
            out = []
            for T in TMIX[wl]:
                if T not in Ts:
                    continue

                def at(name, pbsx, n):
                    (s0, c) = fits[(name, pbsx, T)][1]
                    return s0 + c * n
                d64 = at("live", 64, 10) - at("a0r1", 64, 10)
                d32 = at("live", 32, N32[wl]) - at("a0r1", 32, N32[wl])
                ms = (20 * d64 + 18 * d32) / 1000.0
                g1 = (STEP_MS[wl] / (STEP_MS[wl] - ms) - 1.0) * 100.0
                out.append(f"T={T}: {ms:.3f} ms/step, {g1:+.2f}% tok/s")
                fo.write(json.dumps(dict(step_model=wl, T=T, ms_per_step=ms, pct=g1)) + "\n")
            print(f"  {wl:5s} " + "; ".join(out))
        print("  GO bar (push-B / kit): S0 <= 30 us and c <= 0.8 us/tile beside WOA, and the step "
              "saving >= 0.119 ms (+0.5% G1)")
    fo.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
