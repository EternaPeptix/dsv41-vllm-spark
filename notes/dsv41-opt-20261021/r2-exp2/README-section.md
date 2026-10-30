
## Round 2, exp2: decode-sized MXFP8 GEMM (skinny_mx, "Patch B"). NOT PROMOTED (kernel gate failed)

**What was tried:** `$S/r2/dense/install-r2dense.sh skinny` staged the flag-gated skinny GEMM (`DSV41_SKINNY_MX=1`, `DSV41_SKINNY_MX_WOA=1`) with the service stopped. All 8 ranks matched the expected shas: flashinfer 07c7085d, emulation ddb4b951, o_proj 9df24b47, sm121_skinny_mx.py eedc1f49, .so c054de3a. The GPU kernel gate then ran on spark-two with the service stopped. Patch A (topk) was **not** installed.

**Gate result: FAIL.** Correctness passed: relerr ≤ 3.5e-3 on every shape, matching CUTLASS's own error; within 1 bf16 ulp of CUTLASS on almost every element; det=True for both A8 and A16. Speed failed. The shipped `bench_skinny_mx.py` times a host-side Python loop, so its FlashInfer baseline is launch-bound (~42-160 µs) and its 2-3x "speedups" are wrong. Re-timed inside CUDA graphs (64 DRAM-cold weight copies per graph, CUTLASS autotuned, as in the server); script `r2-exp2/e2_graph.py`:

| GEMM (per rank) | M=1 | M=4 | M=6 | M=8 | per step at M=4-6 |
|---|---|---|---|---|---|
| fused_wqa_wkv 1792x5120 | CUTLASS 45.1 µs / 210 GB/s vs skinny 52.6 / 180 (x0.86) | 45.5 vs 60.0 (x0.76) | 45.5 vs 68.2 (x0.67) | 46.3 vs 74.2 (x0.62) | -0.6 to -1.0 ms |
| wq_b 4096x1280 | 26.9 vs 29.4 (x0.92) | 26.7 vs 32.6 (x0.82) | 27.7 vs 35.9 (x0.77) | 27.4 vs 40.0 (x0.68) | -0.3 to -0.4 ms |
| wo_b 5120x1024 | 26.7 vs 31.7 (x0.84) | 27.7 vs 38.1 (x0.73) | 26.6 vs 41.9 (x0.63) | 26.7 vs 48.4 (x0.55) | -0.4 to -0.6 ms |
| shared w13 576x5120 | 21.2 vs 15.7 (x1.35) | 22.0 vs 18.0 (x1.22) | 21.2 vs 19.8 (x1.07) | 22.0 vs 22.6 (x0.97) | +0.06 to +0.16 ms (side stream) |
| shared w2 5120x288 | 11.2 vs 13.3 (x0.84) | 11.1 vs 16.0 (x0.70) | 11.1 vs 18.0 (x0.62) | 11.1 vs 20.7 (x0.54) | -0.2 to -0.3 ms (side stream) |
| WO-A 1024x4096 (A16) | bf16 bmm 53.9 vs skinny 23.3 (x2.32) | 41.2 vs 26.5 (x1.55) | 48.4 vs 28.7 (x1.69) | 47.1 vs 33.3 (x1.41) | **+0.6 to +0.8 ms** |

Isolated, autotuned CUTLASS MXFP8 already streams about 200 GB/s on wqa_wkv, wq_b and wo_b, about 85% of the 238 GB/s stream bandwidth. The skinny kernel reaches 130-185 GB/s, and its efficiency drops as M grows, because each extra row costs more in the activation-in-registers design. The planner's trace numbers (wqa_wkv 65 µs, wq_b 36.7 µs) were in-server times; the gap above the isolated 45/27 µs is contention or overlap, not kernel inefficiency. A8 overall loses about 1.5-2.5 ms per step. WO-A A16 is the only real win, about 0.6-0.8 ms per step (about 2-2.5% single-stream), which misses the 2x gate and the +4% promotion bar. No serving A/B was run, as the protocol requires after a gate failure.

**Reverted:** `uninstall-r2dense.sh` restored all 8 ranks to 97bcfaa6 480149ff d52c5a20 a5b93da2 f2893ee6. sm121_skinny_mx.py has been removed, no .bak-r2dense files are left, and the launcher is unchanged at 769a0564. The build and bench copies remain in `~/dsv41-r2/skinny_mx/` on each rank and are inert. Live config = exp1 (round-1 flags + DSV41_MOE_COMPACT=1), relaunched with `dsv41-launch-all.sh`.

**Possible follow-up:** use the A16 WO-A path only (`DSV41_SKINNY_MX_WOA=1` without `DSV41_SKINNY_MX`; the code paths are independent). Its expected gain is about +2-2.5% at 1 user and 0 at conc≥2, so it needs ≥3 ABAB pairs to resolve. Codex's WO-A Marlin candidate targets the same GEMM; compare the two before choosing. Another option is to keep the A8 path for shared w13 only at M ≤ 4, but that gain is marginal (~0.15 ms on a side stream).

Raw results: `~/dsv41-opt-20261021/r2-exp2/` (bench_cudagraph.txt = authoritative; bench_quick_hostloop.txt / bench_fi_autotuned_hostloop.txt = launch-bound host-loop timings; install/uninstall sha logs).
