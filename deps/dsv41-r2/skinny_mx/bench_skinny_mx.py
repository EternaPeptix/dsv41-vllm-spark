#!/usr/bin/env python3
"""Offline correctness + microbench for skinny_mx vs the live kernels. NEEDS A GPU: run only when the
Spark is not serving (or with explicit OK), e.g.
    cd ~/dsv41-r2/skinny_mx && ~/venvs/vllm032/bin/python bench_skinny_mx.py [--quick]

Shapes are the per-rank TP8 decode GEMMs of DSv4.1 (see README). For each (shape, M):
  * A8 (MXFP8 activations, same operands as FlashInfer mm_mxfp8): max error vs an fp64 reference of
    the dequantised operands, bf16-ulp diff vs mm_mxfp8, run-to-run bit determinism;
  * A16 (wo_a): vs the live BF16 bmm over the dequantised weight;
  * cache-cold timing: weights rotated over >= 512 MB of copies so every call streams from DRAM
    (GB10 L2 is 24 MB); reports us/call and GB/s of weight+scale bytes for both kernels.
"""
import argparse
import glob
import importlib.util
import os
import sys

import torch

here = os.path.dirname(os.path.abspath(__file__))
so = sorted(glob.glob(os.path.join(here, "build", "skinny_mx_ext*.so")))[0]
spec = importlib.util.spec_from_file_location("skinny_mx_ext", so)
sk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sk)

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.utils import flashinfer as vfi  # noqa: E402

dev = torch.device("cuda", 0)
torch.manual_seed(0)

A8_SHAPES = [  # name, N, K, calls per target step at 1 user (from the 2026-10-21 trace)
    ("fused_wqa_wkv", 1792, 5120, 43),
    ("wq_b(+idx)", 4096, 1280, 48),
    ("wo_b", 5120, 1024, 40),
    ("shared_w13", 576, 5120, 40),
    ("shared_w2", 5120, 288, 40),
]
A16_SHAPES = [("wo_a", 1024, 4096, 40)]


def rand_w(N, K):
    w = (torch.randn(N, K, device=dev) * 2).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.randint(118, 124, (N, K // 32), device=dev, dtype=torch.uint8)
    return w, s


def timeit(fn, n):
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(3):
        fn(i)
    torch.cuda.synchronize()
    st.record()
    for i in range(n):
        fn(i)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1e3 / n  # us


def ulp_diff(a, b):
    ai = a.contiguous().view(torch.int16).int()
    bi = b.contiguous().view(torch.int16).int()
    return (ai - bi).abs().max().item(), (ai != bi).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    Ms = [1, 4, 6, 8] if args.quick else list(range(1, 9))
    print(f"GPU {torch.cuda.get_device_name()} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    total_live = total_new = 0.0
    for name, N, K, calls in A8_SHAPES:
        wbytes = N * K + N * K // 32
        copies = max(2, (512 << 20) // wbytes)
        Ws = [rand_w(N, K) for _ in range(copies)]
        swz = [swizzle_mxfp8_scale(s, M=N, K=K) for _, s in Ws]
        print(f"\n{name} N={N} K={K} split={sk.chosen_split(N, K)} copies={copies}")
        for M in Ms:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            w0, s0 = Ws[0]
            y = sk.skinny_mx_a8(xq.view(torch.uint8), xs.view(torch.uint8), w0.view(torch.uint8), s0)
            y2 = sk.skinny_mx_a8(xq.view(torch.uint8), xs.view(torch.uint8), w0.view(torch.uint8), s0)
            det = torch.equal(y.view(torch.int16), y2.view(torch.int16))
            xq_nsw, xs_nsw = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
            assert torch.equal(xq_nsw.view(torch.uint8), xq.view(torch.uint8))
            xd = dequant_mxfp8_to_bf16(xq_nsw, xs_nsw).double()
            wd = dequant_mxfp8_to_bf16(w0, s0).double()
            ref = xd @ wd.T
            fi = vfi.mm_mxfp8(xq, w0.t(), xs, swz[0], out_dtype=torch.bfloat16, backend="cutlass")
            rel = ((y.double() - ref).abs().max() / ref.abs().max()).item()
            rel_fi = ((fi.double() - ref).abs().max() / ref.abs().max()).item()
            mu, frac = ulp_diff(y, fi)
            t_new = timeit(lambda i: sk.skinny_mx_a8(xq.view(torch.uint8), xs.view(torch.uint8),
                                                     Ws[i % copies][0].view(torch.uint8), Ws[i % copies][1]),
                           args.iters)
            t_old = timeit(lambda i: vfi.mm_mxfp8(xq, Ws[i % copies][0].t(), xs, swz[i % copies],
                                                  out_dtype=torch.bfloat16, backend="cutlass"), args.iters)
            if M in (4, 6):
                total_live += calls * t_old / 2
                total_new += calls * t_new / 2
            print(f"  M={M}: skinny {t_new:6.1f}us {wbytes / t_new / 1e3:5.0f}GB/s | flashinfer {t_old:6.1f}us "
                  f"{wbytes / t_old / 1e3:5.0f}GB/s | x{t_old / t_new:4.2f} | relerr {rel:.1e} (fi {rel_fi:.1e}) "
                  f"ulp-vs-fi max {mu} frac {frac:.3f} det={det}", flush=True)
            assert det and rel < 1e-2, "correctness gate failed"
        del Ws, swz
        torch.cuda.empty_cache()
    for name, N, K, calls in A16_SHAPES:
        wbytes = N * K + N * K // 32
        copies = max(2, (512 << 20) // (N * K * 2))
        Ws = [rand_w(N, K) for _ in range(copies)]
        Wb = [dequant_mxfp8_to_bf16(w, s) for w, s in Ws]
        print(f"\n{name} (A16 vs live BF16 bmm) N={N} K={K} split={sk.chosen_split(N, K)} copies={copies}")
        for M in Ms:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            y = sk.skinny_mx_a16(x, Ws[0][0].view(torch.uint8), Ws[0][1])
            ref = x.double() @ Wb[0].double().T
            z = torch.empty((M, 1, N), dtype=torch.bfloat16, device=dev)
            xb = x.view(M, 1, K)

            def live(i):
                torch.bmm(xb.transpose(0, 1), Wb[i % copies].view(1, N, K).transpose(1, 2), out=z.transpose(0, 1))
            live(0)
            rel = ((y.double() - ref).abs().max() / ref.abs().max()).item()
            rel_l = ((z.view(M, N).double() - ref).abs().max() / ref.abs().max()).item()
            mu, frac = ulp_diff(y, z.view(M, N))
            t_new = timeit(lambda i: sk.skinny_mx_a16(x, Ws[i % copies][0].view(torch.uint8), Ws[i % copies][1]),
                           args.iters)
            t_old = timeit(live, args.iters)
            if M in (4, 6):
                total_live += calls * t_old / 2
                total_new += calls * t_new / 2
            print(f"  M={M}: skinny {t_new:6.1f}us {wbytes / t_new / 1e3:5.0f}GB/s | bf16 bmm {t_old:6.1f}us "
                  f"{N * K * 2 / t_old / 1e3:5.0f}GB/s | x{t_old / t_new:4.2f} | relerr {rel:.1e} (live {rel_l:.1e}) "
                  f"ulp-vs-live max {mu} frac {frac:.3f}", flush=True)
            assert rel < 1e-2
    print(f"\nper target step at M=4/6 avg (isolated, cache-cold): live {total_live / 1e3:.2f} ms -> skinny "
          f"{total_new / 1e3:.2f} ms, saving {(total_live - total_new) / 1e3:.2f} ms")


if __name__ == "__main__":
    sys.exit(main())
