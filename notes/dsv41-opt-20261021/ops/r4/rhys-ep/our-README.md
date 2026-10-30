# DSv4.1-Flash TP8 optimisation, 2026-10-21

Deployment: vLLM 0.32.0, TP8 over spark-one..spark-eight, systemd --user unit `dsv41-rank`, launcher `~/dsv41-tp8.sh <rank>`, API spark-one :8000 model `dsv41`.
Raw results per experiment are in this directory (`spec-len/`, `m32/`, `rdma-ar/`, `conc/`, `final/`), each with `summary.txt`.

## Final numbers (live config, final/ run)

Measured 18:05-18:09 UTC on the instance started 17:53:53 UTC (launcher 3045aa83). Single = greedy, 1024 tokens, 2 repeats per type, averaged. Conc = 512 tokens per request.

| metric | baseline 14:08 | final | change |
|---|---|---|---|
| single code | 56 | 71.4 (70.5 / 72.3) | +27% |
| single prose | 42 | 53.9 (52.2 / 55.7) | +28% |
| single math | 73 | 104.4 (102.8 / 106.0) | +43% |
| single mean | 57 | 76.6 | +34% |
| conc4 aggregate | 102 | 134.7 | +32% |
| conc8 aggregate | 147 | 192.6 | +31% |
| conc16 aggregate | about 147 (ms8 queues; about 190 at ms8 with RDMA) | 277.3 | about +88% vs baseline |
| GSM8K-50 gate | 50/50 | 50/50, 0 truncated (200 tok/s at c8) | pass |
| exact.py combined | n/a | fd1603b4e20b, which equals the conc B2 hash on the same instance, so the output reproduces within an instance | |

The server log shows no errors (only the benign transformers model-type notice). All 8 units are active, with no extra environment set.

## Changelog (in order applied)

| step | change | result | launcher sha256 after |
|---|---|---|---|
| 0 | baseline (14:08 UTC) | single 57, conc4 102, conc8 147 | 58ab5e61 |
| 1 spec-len | added env-gated `DSV41_SPEC_JSON` override (inert when unset). SPEC=3/4 and adaptive-off tested. | not promoted (SPEC=4 -3.9% single, SPEC=3 -7.4%, adaptive-off -11%/-22% at conc4/8) | f2fd3f08 |
| 2 m32 | M32 small-tile MoE path for M<=48 (`DSV41_SM121_TARGET_M32=1`, native lib `~/dsv41-stage/m32/target-m32-native-v1`) | promoted, about +5% at all loads | f4ba806e |
| 3 rdma-ar | custom RDMA all-reduce over CX7 (`DSV41_RDMA_AR=1`), 3-8x faster than PyNccl for 1-48 rows | promoted, +23% single, +17% conc4, +20% conc8 | 753f0a56 |
| 4 conc | `--max-num-seqs ${DSV41_MAX_NUM_SEQS:-16}` + spec-kernel warmup patch (`DSV41_WARM_SPEC_KERNELS=1`); inert M32/M64 max-M knob (default 48) | promoted, conc12 +30%, conc16 +45%, flat at <=8 | 3045aa83 |

Site-packages changes (all under `~/venvs/vllm032/lib/python3.12/site-packages/vllm/`, sha256 identical on all 8 ranks):

| file | live sha | backup |
|---|---|---|
| model_executor/layers/fused_moe/experts/deep_gemm_moe.py | 84ecc6ef | `.bak-m32` = fe39558e (pristine), `.bak-conc` = b8bb0327 (M32 before max-M knob) |
| model_executor/layers/fused_moe/deep_gemm_utils.py | b7e8de4d | `.bak-m32` = 5eb623a6 (pristine) |
| model_executor/layers/fused_moe/experts/sm121_target_m32.py (new) | 723fdf8d | `.absent-m32` marker |
| distributed/device_communicators/cuda_communicator.py | 52fc8cf5 | `.bak-rdma` = 2f5368d6 (pristine) |
| distributed/device_communicators/rdma_all_reduce.py (new) | 3a128082 | none (new file; uninstall-rdma.sh removes it) |
| v1/worker/gpu/warmup.py | 19009d42 | `.bak-warmspec` = daeac571 (pristine) |

Launcher backups on every rank: `~/dsv41-tp8.sh.bak-spec-len` (58ab5e61, original), `.bak-m32` (f2fd3f08), `.bak-rdma-ar` (f4ba806e), `.bak-conc` (753f0a56).
Other staging: `~/rdma_ar_dsv41/` (rdma_ar_ext.so dfeeab00), `~/dsv41-stage/m32/` (native lib + JIT cache, 3 entries).

## Caveats

- Greedy output is not bit-identical across server restarts, even with an unchanged config: within one instance the hashes repeat exactly, across restarts 0-2 of 6 prompts match. The cause is likely the adaptive-verification cost profile, which is re-measured at each startup, plus the RDMA reduction order. `exact.py` can therefore compare arms only within one instance.
- GSM8K-50 scores ranged 49-50 on every arm, controls included. The one recurring miss (item 12, 12 vs 13) is a genuine wrong answer, not a truncation.
- Mean accept length dropped from about 2.46 to about 2.35 with RDMA AR, because the reduction order changed. Throughput more than covers it.
- The MemAvailable floor while serving is about 22-23 GiB per rank at ms16. Startup briefly dips to about 14 GiB on spark-six and spark-seven.
- spark-one cannot ssh to spark-six, seven or eight (publickey denied). Run fleet scripts from the Mac.

## Rollback (run from the Mac; S = /private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad)

Fast, env only (site files remain but are inert):

```
bash $S/conc/launch-conc.sh DSV41_MAX_NUM_SEQS=8 DSV41_WARM_SPEC_KERNELS=0   # undo step 4
bash $S/conc/launch-conc.sh DSV41_RDMA_AR=0                                  # undo step 3 (PyNccl)
bash $S/conc/launch-conc.sh DSV41_SM121_TARGET_M32=0                         # undo step 2 (M64 path)
bash $S/conc/launch-conc.sh DSV41_MAX_NUM_SEQS=8 DSV41_WARM_SPEC_KERNELS=0 DSV41_RDMA_AR=0 DSV41_SM121_TARGET_M32=0   # all three
```

Launcher only, for example back to the step-3 launcher:

```
for h in one two three four five six seven eight; do ssh spark-$h 'cp ~/dsv41-tp8.sh.bak-conc ~/dsv41-tp8.sh'; done
bash $S/conc/launch-conc.sh
```

Full return to the 14:08 baseline. Stop first: `for h in one two three four five six seven eight; do ssh spark-$h 'systemctl --user stop dsv41-rank'; done`

```
# warmup.py on each rank:
for h in one two three four five six seven eight; do ssh spark-$h 'cd ~/venvs/vllm032/lib/python3.12/site-packages/vllm/v1/worker/gpu && cp warmup.py.bak-warmspec warmup.py'; done
bash $S/rdma/scripts/uninstall-rdma.sh      # restores cuda_communicator.py from .bak-rdma, removes rdma_all_reduce.py
bash $S/m32/uninstall.sh                    # restores deep_gemm_moe.py / deep_gemm_utils.py from .bak-m32, removes sm121_target_m32.py
for h in one two three four five six seven eight; do ssh spark-$h 'cp ~/dsv41-tp8.sh.bak-spec-len ~/dsv41-tp8.sh'; done
bash /private/tmp/claude-501/-Users-jeweled/1b0ccff9-d7d4-4c14-901f-1d71aef60396/scratchpad/dsv41-launch-all.sh   # run with bash; the file is not executable
```

Afterwards verify that sha256 is identical on all 8 ranks. The server is ready when `~/dsv41-logs/rank0.log` shows "Application startup complete" (about 6.5 min).

## Recommended next steps

1. Make greedy output deterministic across restarts: pin or persist the adaptive-verification cost profile, for example by loading it from a file instead of re-measuring at startup. Then exact.py becomes a usable regression gate across restarts.
2. Run a larger GSM8K cohort (200 or more) on the live config to rule out a 1-point accuracy shift from RDMA AR. Item 12 flips 12/13 between runs.
3. Tune RDMA AR (`RDMA_AR_MAX_REDUCE_BYTES`, two-shot threshold) and profile the step again. All-reduce was 10.7 ms of the 40.8 ms step and is now much smaller, so FP4 MoE (about 14 ms) and the dense GEMMs (about 12 ms) dominate.
4. With RDMA AR live, the conc16/ms16 envelope has headroom (MemAvailable about 22 GiB). Try ms24 or ms32 if many concurrent users are expected.
5. Recalibrate adaptive verification (`DSV41_AV_VERIFY_SCALE` / `FIXED_MS`) at concurrency. It showed little upside at one user, but verify costs changed a lot after RDMA AR.
6. Fix spark-one to spark-six/seven/eight ssh keys so in-bench memory sampling covers all ranks.

## Round 2, exp1: moe-compact (unpadded target FC1), PROMOTED 2026-10-21 20:40 UTC

**What changed.** The routed FP4 MoE FC1 weights of the target model are stored without TP padding: per rank and expert, N = 576 rows `[gate 288 | up 288]` instead of 768 `[gate 288 | 0 x96 | up 288 | 0 x96]`. FC2 keeps K = 384. The SiLU-mul-quant kernel writes the 96 zero columns into the 384-wide FP8 buffer that FC2 consumes, so FC2 inputs and scale groups are byte-identical to the padded path. FC2 cannot be compacted at K = 288: DeepGEMM asserts `not is_mixed_fp4 or k % 128 == 0` (the probe confirmed this), so it would need a kernel change. The drafter (128 experts) is untouched. The change is behind `DSV41_MOE_COMPACT`, and with the flag unset all three files take the old code paths.

**Files** (under `~/venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/`, sha256 identical on all 8 ranks, backups `.bak-compact` = previous live):

| file | live sha now | `.bak-compact` |
|---|---|---|
| fused_moe/experts/deep_gemm_moe.py | 45418259 | 84ecc6ef |
| quantization/mxfp4.py | 67339029 | a7fc1bcd |
| quantization/utils/fp8_utils.py | 8dc7dab8 | 013df536 |

Launcher: `export DSV41_MOE_COMPACT=${DSV41_MOE_COMPACT:-1}` added after `DSV41_LOCK_ENGRAM_HOST`. `~/dsv41-tp8.sh` is now **769a0564** on all 8 ranks, and the backup `~/dsv41-tp8.sh.bak-compact` = 3045aa83. Payload, scripts and test are in `~/dsv41-stage/moe-compact/` on every rank; the Mac originals are in `$S/r2/moe-compact/`.

**Offline gate** (`run_offline_test.sh`, spark-one rank 0 + spark-eight rank 7, service stopped). Every case was BIT-EXACT against the live padded expert (bf16 ±0 counted equal), and the flag-on-padded-weights run was also identical:
- M32 path, random weights: M = 1..48 every value (144 cases) plus the default sweep up to 1024 (60 cases, 128-row path for M > 48), with valid, half-padding and all-padding routes, and CUDA-graph replay with poisoned outputs. All passed on both hosts.
- Real checkpoint layers 0 and 39, all 384 experts, TP ranks 0 and 7: 21 + 21 cases per host, plus M = 1..48 for L39 on rank 7 (144). All passed.
- M64 fallback (M32 off): 60 cases, all passed. CPU loader check: 432 byte-exact slices.

Per-layer routed MoE timing, 8 distinct layer copies (L2-defeating), spark-one:

| M | path | padded µs | compact µs | speedup | padded GB/s | compact GB/s |
|---|---|---:|---:|---:|---:|---:|
| 1 | M32 | 119.4 | 110.2 | 1.084 | 157 | 142 |
| 6 | M32 | 608.3 | 519.5 | 1.171 | 176 | 172 |
| 12 | M32 | 1149.8 | 984.6 | 1.168 | 180 | 175 |
| 18 | M32 | 1634.3 | 1398.1 | 1.169 | 180 | 175 |
| 24 | M32 | 2127.1 | 1818.7 | 1.170 | 180 | 175 |
| 36 | M32 | 2875.5 | 2456.1 | 1.171 | 179 | 175 |
| 48 | M32 | 3535.3 | 3025.3 | 1.169 | 179 | 174 |
| 96 | 128-row | 8063.4 | 7653.6 | 1.054 | 117 | 103 |
| 6 / 24 / 48 | M64 fallback | 725 / 2515 / 4292 | 673 / 2301 / 3886 | 1.08 / 1.09 / 1.10 | 150 | 135-138 |

The byte model predicts 1.20. On the M32 path (64-wide N tiles, 576 = 9×64) the compact layout gets 1.17. On the 64- and 128-row paths it gets 1.05-1.10, which is consistent with DeepGEMM choosing 128-wide N tiles there, so the last tile over-reads 64 rows of the next expert. spark-eight matches within 2%.

**Serving A/B** (C, A, C, A, a restart each; `~/dsv41-opt-20261021/r2-exp1/{C1,A1,C2,A2}`; single = greedy 1024 tok; conc = aggregate tok/s at 512 tok):

| instance | code | prose | math | single mean | conc4 | conc8 | conc12 | conc16 | accept len (single / c4 / c16) | GSM8K-50 | exact.py x2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| C1 control | 64.6 | 54.5 | 107.1 | 75.4 | 132.2 | 189.2 | 235.5 | 281.0 | 2.32 / 2.36 / 2.34 | 49/50 | self-consistent |
| A1 compact | 68.2 | 54.6 | 110.4 | 77.7 | 141.9 | 208.4 | 258.8 | 298.1 | 2.29 / 2.35 / 2.28 | 50/50 | self-consistent |
| C2 control | 65.1 | 55.3 | 108.2 | 76.2 | 130.7 | 192.0 | 238.7 | 273.7 | 2.34 / 2.26 / 2.31 | 50/50 | self-consistent |
| A2 compact | 75.7 | 57.9 | 112.6 | 82.1 | 140.5 | 210.0 | 262.4 | 300.4 | 2.49 / 2.31 / 2.32 | 50/50 | self-consistent |
| **mean C → A** | | | | **75.8 → 79.9 (+5.4%)** | **131.5 → 141.2 (+7.4%)** | **190.6 → 209.2 (+9.8%)** | **237.1 → 260.6 (+9.9%)** | **277.3 → 299.2 (+7.9%)** | Δ within ±0.07 | | |

Every arm instance beats every control instance at conc4, 8, 12 and 16. Single-stream gains are noisier (A1 +3%, A2 +8%). Part of A2's single-stream gain comes from a higher accept length in that instance (2.49).

**Memory.** Model load is 49.71 → 42.21 GiB per rank (−7.50 GiB). MemAvailable during conc16 is about 23 → about 30.5 GiB per rank. The KV cache is unchanged (`--kv-cache-memory-bytes 8G` = 1,628,987 tokens, 6.21× full 262K contexts) because it is set explicitly. That leaves about 7.5 GiB of new headroom for a bigger KV cache (for example 12-14G) and/or ms24-32. This is not tested yet.

**Final live config:** the round-1 flags plus `DSV41_MOE_COMPACT=1` (launcher default). Restarted with `dsv41-launch-all.sh`: /health 200, rdma_ar enabled, compact log line on ranks 0, 4 and 6, 17×23 → "391".

**Rollback** (from the Mac, S = /private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad):
```
bash $S/conc/launch-conc.sh DSV41_MOE_COMPACT=0             # env only: padded layout, files inert
for h in one two three four five six seven eight; do ssh spark-$h 'cp ~/dsv41-tp8.sh.bak-compact ~/dsv41-tp8.sh'; done   # launcher -> 3045aa83
# full file revert (service stopped first):
for h in one two three four five six seven eight; do ssh spark-$h 'systemctl --user stop dsv41-rank'; done
bash $S/r2/moe-compact/uninstall.sh                          # restores 84ecc6ef / a7fc1bcd / 013df536 from .bak-compact
bash $S/dsv41-launch-all.sh
```
Raw results: `~/dsv41-opt-20261021/r2-exp1/` (instances, `inst.sh`, `summ.py`, `offline/` with both hosts' jsonl).

**Next:** (1) FC2 compact via a split main [5120,256] + repacked tail tile in `sm120_fp8_fp4_gemm_1d1d.cuh` (another ~8% of routed bytes). (2) Make the 128-row path pick 64-wide N tiles for N = 576, or extend M32-style tiles above M = 48: conc16 (M = 96) gets only 1.05× in the kernel. (3) Spend the freed 7.5 GiB on KV and/or max-num-seqs.

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

## Round 2 / exp3-rdma-thresholds (2026-10-21, 20:30-23:10 UTC) -- PROMOTED (env-only)

**What changed:** launcher `~/dsv41-tp8.sh` gains one line (after DSV41_MOE_COMPACT):
`export RDMA_AR_MAX_REDUCE_BYTES=${RDMA_AR_MAX_REDUCE_BYTES:-1048576}` -- the custom RDMA two-shot all-reduce now
handles up to 1 MiB (102 rows x 10,240 B) instead of 512 KiB (51 rows). Effect: the 7 drafter all-reduces at 11-16
users (55-80 rows) and target verify all-reduces at 52-102 tokens no longer fall back to NCCL. No site-file change
(rdma_all_reduce.py still 3a128082; the miss-audit patch was NOT installed). Launcher 769a0564 -> d0749eda on all 8
(backup `~/dsv41-tp8.sh.bak-rdmaext` = 769a0564 on all 8).

**Offline bench (service stopped, 8 ranks, CUDA-graph, us/op, bit-exact on all ranks, 0 failures E1 and E2):**

| rows | bytes | NCCL | 1-shot | 2-shot | 2-shot speedup |
|---|---|---|---|---|---|
| 48 | 480 KiB | 627 | 223 | 80 | 7.9x |
| 56 | 560 KiB | 752 | 256 | 87 | 8.6x |
| 64 | 640 KiB | 838 | 285 | 96 | 8.8x |
| 80 | 800 KiB | 924 | 348 | 112 | 8.2x |
| 96 | 960 KiB | 431 | 413 | 138 | 3.1x |
| 128 | 1.25 MiB | 671 | 543 | 173 | 3.9x |
| 192 | 1.9 MiB | 915 | 839 | 244 | 3.7x |

Logits all-gather (x,16160) bf16: RDMA wins at 5-30 rows (2.9-4.3x) but LOSES at 48-96 rows (0.66-0.89x; e.g. 64 rows
748 vs 982 us). Engram all-gathers 3.5-8.7x. => raising RDMA_AR_MAX_BYTES above 2 MiB is not useful (Arm C dropped).
spark-four rx_discards_phy 391/18 (f0/f1) unchanged through every bench and serving run; all other hosts 0.

**Cost profile (rank0 adaptive verification):** draft@12/16 req 16.5/19.1 -> 11.1/12.5 ms; verify@56 tok 243 -> 188 ms
(-> 119 ms with DSV41_TARGET_M32_MAX_M=96).

**Serving A/B (13 instances, alternating; runarm_conc single 1024 tok + conc4/8/12/16; GSM8K-50 @ CONC 16; exact.py x2
self-consistent in every instance):**

| arm | n | single mean | conc4 | conc8 | conc12 | conc16 | GSM8K |
|---|---|---|---|---|---|---|---|
| control (C1-C4) | 4 | 79.55 | 143.2 | 207.8 | 259.0 | 299.4 | 50,50,49,50 |
| A: MAX_REDUCE 1 MiB (A1,A2) | 2 | 78.2 (-1.7%) | 142.4 (-0.6%) | 205.3 (-1.2%) | 261.9 (+1.1%) | **315.7 (+5.4%)** | 50,50 |
| B: A + TARGET_M32_MAX_M=96 | 2 | 78.1 (-1.8%) | 140.3 (-2.1%) | 209.8 (+1.0%) | 267.7 (+3.4%) | 316.8 (+5.8%) | 50,50 |
| AP: A + MAX_BYTES 1 MiB | 2 | 78.4 (-1.4%) | 142.8 (-0.3%) | 205.2 (-1.3%) | 266.3 (+2.8%) | 316.2 (+5.6%) | 49,50 |
| BP: B + MAX_BYTES 1 MiB | 2 | 79.5 (0%) | 138.5 (-3.3%) | 205.9 (-0.9%) | 263.0 (+1.6%) | 312.2 (+4.3%) | 50,49 |

Per-instance conc16: C 301.9/297.9/298.6/299.3; A 312.9/318.5; B 318.9/314.6; AP 312.3/320.2; BP 310.4/313.9.
Single/conc4/conc8 do not exercise any changed path (<=50 rows); their deltas are noise. Arm A promoted as the
minimal change meeting every gate. B/AP are within noise of A at conc16 (B has the best conc12, +3.4%, but conc4
-2.1% noise). B only matters if max-num-seqs >16. Accept length unchanged (~2.28-2.36 at conc16).

**Final state:** restarted via dsv41-launch-all.sh; /health 200; `max_reduce=1048576` in all 8 rank logs;
draft@12/16 = 11.07/12.54 ms; sanity 17*23 -> 391.

**Rollback:** `for h in spark-{one..eight}; do ssh $h 'cp ~/dsv41-tp8.sh.bak-rdmaext ~/dsv41-tp8.sh'; done` (sha 769a0564),
or just launch with `RDMA_AR_MAX_REDUCE_BYTES=524288` via conc/launch-conc.sh. Raw results: `~/dsv41-opt-20261021/r2-exp3/`.

**Follow-ups:** (1) logits all-gather 48-64 rows is slower on RDMA than NCCL: a gather-specific limit (~1 MiB) is a
small free win once a separate env knob exists (RDMA_AR_MAX_BYTES also caps reduce). (2) DSV41_TARGET_M32_MAX_M=96 +
max-num-seqs 24/32 now that both verify cliffs (AR + MoE) are gone at 56-96 tokens.

## Round 2 / exp4: vocab-sharded Markov drafter (DSV41_DSPARK_SHARDED_MARKOV): NOT promoted (2026-10-21/22 UTC)

**What it is:** a patch to `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (sha 97bcfaa6 -> 774b34c2). When the env flag is set, each rank adds the Markov bias only to its own 16,160-token vocab slice. The ranks then exchange one 16 B (max, id) record per request per position (5 small fp32 gathers per step) instead of each reading the full 129280x256 Markov head 5 times and all-gathering full-vocab logits.
Sources: Mac scratchpad `r2/drafter/` (install.sh, markov-shard/). Install: `bash install.sh markov`. Revert: `bash install.sh markov rollback`.

**Correctness (all checks passed):**
- rank0.log has `DSV41_DSPARK_SHARDED_MARKOV: vocab shard [0, 16160) pad 0, markov rows/rank 16160 of 129280, tp 8`, with no "ignored" warning.
- RDMA_AR_AUDIT confirms the new `all_gather (n,4) float32` calls go through rdma_ar, not NCCL.
- exact.py was self-consistent in every instance. GSM8K-50 at conc 8 scored 50/50 in all 4 A/B instances.
- AV cost profile draft_ms(1): 5.10 / 5.20 (control) -> 3.72 / 3.76 (arm), i.e. -1.4 ms. draft_ms(16): 12.5 -> 9.95.

**A/B** (C = live config, A = +flag; same launcher d0749eda; order C1 A1 C2 A2). AL = mean accept length.

| inst | code | prose | math | mean | c4 | c8 | c12 | c16 | AL code/prose/math | GSM8K |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 76.29 | 58.17 | 110.38 | 81.61 | 140.78 | 212.26 | 267.22 | 316.45 | 2.59/1.73/3.78 | 50/50 |
| A1 | 72.51 | 60.44 | 113.84 | 82.26 | 144.64 | 216.86 | 274.68 | 322.42 | 2.27/1.69/3.72 | 50/50 |
| C2 | 72.61 | 57.14 | 107.65 | 79.13 | 140.70 | 206.21 | 265.92 | 311.93 | 2.58/1.71/3.69 | 50/50 |
| A2 | 71.31 | 58.04 | 111.14 | 80.16 | 144.41 | 212.76 | 278.47 | 320.44 | 2.36/1.64/3.64 | 50/50 |

Means, control -> arm:
- single 80.37 -> 81.21 (+1.0%; gate needs +2.5%)
- conc4 140.7 -> 144.5 (+2.7%)
- conc8 209.2 -> 214.8 (+2.7%)
- conc12 266.6 -> 276.6 (+3.8%)
- conc16 314.2 -> 321.4 (+2.3%)
- code AL 2.585 -> 2.31 (-0.27, outside the +/-0.08 gate). Prose -0.06, math -0.06.

**Why code AL dropped: the drafts did not get worse.** An AV-off probe was run (`DSV41_SPEC_JSON={"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":false}`, DC1 vs DA1, single x2 reps):
- Untrimmed acceptance matches: code 2.80 vs 2.78, prose 2.13 vs 2.10, math 3.59 vs 3.80, conc4 2.78 vs 2.82.
- exact.py hashes were identical across the two instances (47098a31f5e4).
- The cause is the adaptive-verification trimmer. It picks the draft budget k = argmax (1+sum s_i)/(draft_ms + verify_ms(k)). A cheaper draft_ms lowers the fixed cost, so the profiled verify slope (already ~3x too steep, see the drafter prep notes) trims harder. Code loses positions 2-5 and about 3% tok/s, which cancels most of the drafter saving at 1 user.
- With AV off, the flag gives +6.1% single (77.6 -> 82.3). At concurrency, the trimmer's decisions are dominated by verify cost, so the gain shows through.

**Result:** not promoted under the exp4 gate (single +1.0% < +2.5%; code AL outside +/-0.08). The patch was rolled back (speculator.py = 97bcfaa6 on all 8). The launcher is unchanged (d0749eda). Live = same config as before exp4 (launch-all, no extra env), /health 200, sanity 391.

**Follow-up:** re-test this flag together with the AV cost calibration (drafter patch B: `DSV41_AV_VERIFY_SCALE` / `DSV41_AV_COST_JSON`). Once the verify slope is calibrated, the cheaper drafter should no longer cause over-trimming at 1 user, and the +2.3-3.8% conc gain plus about +6% untrimmed single gain should both be available.
Raw results: `~/dsv41-opt-20261021/r2-exp4/` (C1 A1 C2 A2 DC1 DA1, summ2.py, dsumm.py, inst.sh, diag.sh).

## Round 2 / exp5 — adaptive-verification cost calibration (PROMOTED, 2026-10-22 01:31 UTC)

**What changed**
- `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` was patched (av-calib) on all 8 ranks: sha b0526416 -> ec240369. Backup is `.bak-avcal`. Installed with `bash $D/install.sh av`, where D = Mac scratchpad r2/drafter.
  - The patch adds the env knobs DSV41_AV_VERIFY_SCALE / SCALE_N0 / SCALE_N1 / FIXED_MS / COST_JSON / DUMP_JSON / LOG_BUDGET.
  - It recalibrates the startup cost profile on rank 0 before the profile is broadcast to the other ranks.
- `~/dsv41-tp8.sh` changed from d0749eda to e270215f on all 8 ranks; backup is `~/dsv41-tp8.sh.bak-avcal`. It now exports these defaults:
  `DSV41_AV_VERIFY_SCALE=0.6 DSV41_AV_FIXED_MS=2.0 DSV41_AV_SCALE_N0=48 DSV41_AV_SCALE_N1=56`.
  - Every verify width up to 48 is priced as V(1) + 0.6*(V(n)-V(1)). The price tapers back to the profiled cost by width 56; 48->56 is the cliff where the M32 MoE path ends.
  - Draft cost gets +2.0 ms for the time spent outside the graphs.
- Rank0.log shows a line 'DSV41 adaptive-verification calibration (... scale=0.6 n0=48.0 n1=56.0 fixed_ms=2.0)'.

**Why:** the startup profile uses random tokens. Its verify slope is about 3.5 ms per extra draft at 1 user, but the measured slope is 2.39 ms per draft: (math - prose step ms) / (math - prose drafts verified) from the C1 budget histograms. That puts the refit scale at a = 0.68. The time outside the graphs refits to 2.1 ms (prose and math both).
- The planner's default taper (N0=8, N1=48) made the cost slope at widths 24-48 *steeper* than the profile. At conc12/16 the trimmer then cut verified tokens from about 30 to 15-22, and conc12/16 regressed by 2-6% (arms A1 a=.4/f2.5 and A2 a=.6/f2).
- N1=60 (A4) still regressed by 1.4%. A flat scale up to 48 (A5) fixed it.

**A/B (same window, C/A alternated, every instance got GSM8K-50 at conc 8 and an exact.py x2 self-consistency check):**

| inst | code | prose | math | single mean | c4 | c8 | c12 | c16 | GSM | accept per pos (math) |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 68.5 | 55.5 | 106.8 | 76.95 | 135.8 | 205.4 | 264.8 | 317.2 | 50 | .84/.64/.56/.30/.26 |
| A2-1 (a.6 f2 N1 48) | 78.0 | 58.8 | 109.7 | 82.15 | 145.1 | 204.8 | 255.3 | 309.8 | 50 | .87/.72/.59/.36/.30 |
| A1-1 (a.4 f2.5 N1 48) | 76.2 | 58.9 | 103.1 | 79.40 | 142.3 | 208.8 | 253.8 | 293.4 | 49 | .82/.64/.54/.33/.27 |
| A4-1 (a.6 f2 N1 60) | 78.8 | 59.0 | 111.1 | 82.96 | 144.9 | 207.1 | 267.3 | 310.4 | 50 | .86/.72/.61/.38/.31 |
| C2 | 70.5 | 58.0 | 112.5 | 80.32 | 144.8 | 207.4 | 269.5 | 312.3 | 50 | .88/.71/.61/.36/.29 |
| A5-1 (a.6 f2 N0 48 N1 56) | 77.0 | 58.9 | 115.4 | 83.76 | 143.2 | 210.5 | 268.4 | 319.6 | 50 | .87/.75/.65/.43/.36 |
| C3 | 72.3 | 56.8 | 112.4 | 80.52 | 140.1 | 206.6 | 266.6 | 311.1 | 50 | .85/.71/.60/.38/.33 |
| A5-2 (same) | 79.9 | 60.0 | 113.3 | 84.38 | 144.4 | 212.0 | 271.8 | 313.2 | 50 | .87/.74/.61/.41/.33 |

Mean of the 3 controls vs A5 (2 instances):
- single: code 70.4 -> 78.5 (+11.4%), prose 56.8 -> 59.4 (+4.7%), math 110.6 -> 114.3 (+3.4%); mean **79.3 -> 84.1 (+6.1%)**
- conc4 140.2 -> 143.8 (+2.5%), conc8 206.5 -> 211.3 (+2.3%), conc12 267.0 -> 270.1 (+1.2%), conc16 313.5 -> 316.4 (+0.9%)
- Every instance was self-consistent on exact.py.

Drafts verified per step at 1 user, control -> A5:
- prose 1.61-2.09 -> 2.14-2.33 (verifying 1 draft: 68% -> 37-48%)
- code 2.70 -> 3.48
- math 3.22 -> 4.04 (verifying 5 drafts: 34% -> 56-59%)

Total tokens verified per step:
- r4: 7.4-7.7 -> 8.0-8.4
- r8: 21-22 -> 24
- r16: 30 -> 31.5

Code position-2 acceptance went from .36-.40 to .51-.55; prose position-2 acceptance went from .08-.10 to .16-.20.

**Cumulative since round 1** (76.6 / 134.7 / 192.6 / 277.3): single 84.1 (+9.8%), conc4 143.8 (+6.8%), conc8 211.3 (+9.7%), conc16 316.4 (+14.1%).

**Rollback**
- Launcher only (trimmer becomes stock): `cp ~/dsv41-tp8.sh.bak-avcal ~/dsv41-tp8.sh` on all 8 ranks. Alternatively launch with DSV41_AV_VERIFY_SCALE=1 DSV41_AV_FIXED_MS=0. Patch with scale 1 / fixed 0 = stock behaviour.
- Site file: `bash $D/install.sh av rollback`, then restart with `dsv41-launch-all.sh`.

**Raw data:** ./r2-exp5/<inst>/ (bench output, rank0.log copies, av-curves JSON) plus av5_summ.py and nrhist.py.

## Round 2 summary and final verification (2026-10-22 ~01:35-02:00 UTC)

**Verified live state (all 8 ranks):** unit dsv41-rank active since 01:25:47 UTC, unit env only `DSV41_EXTRA=` (empty), /health 200, 17*23 -> "391".
rank0.log shows `rdma_ar enabled ... max_reduce=1048576`, `target compact FC1 enabled ... FC1 rows 576`, and the adaptive-verification calibration line (scale=0.6 n0=48 n1=56 fixed_ms=2.0). No nsys or Codex processes.

sha256 (first 8 hex), identical on all 8 ranks:
- launcher ~/dsv41-tp8.sh e270215f (backups .bak-avcal .bak-rdmaext .bak-compact .bak-conc .bak-m32 .bak-rdma-ar present; .bak-spec-len exists only on some ranks, from round 1)
- site-packages (venvs/vllm032): deep_gemm_moe.py 45418259, mxfp4.py 67339029, fp8_utils.py 8dc7dab8, adaptive_verification.py ec240369, speculator.py 97bcfaa6 (stock), cuda_communicator.py 52fc8cf5, rdma_all_reduce.py 3a128082, deep_gemm_utils.py b7e8de4d, warmup.py 19009d42. 10 .bak-* files are present on every rank.

**Final measurement** (one instance, ./r2-final/, run.sh. Single is 1024 greedy tokens x 2 repeats per type; conc runs are 512 tokens each):

| metric | original baseline | round-1 final | **round-2 final** | vs R1 | vs baseline |
|---|---|---|---|---|---|
| single code | - | 71.4 | **72.5** | +1.5% | |
| single prose | - | 53.9 | **58.8** | +9.1% | |
| single math | - | 104.4 | **111.1** | +6.4% | |
| single mean | 57 | 76.6 | **80.8** | +5.5% | +41.8% |
| conc4 agg | 102 | 134.7 | **144.0** | +6.9% | +41.1% |
| conc8 agg | 147 | 192.6 | **210.1** | +9.1% | +43.0% |
| conc16 agg | - | 277.3 | **313.5** | +13.0% | |
| mean accept len single / c16 | | ~2.35 | 2.48 / 2.39 | | |
| GSM8K-50 (conc 8) | | 49-50/50 | **50/50** | | |
| GSM8K-200 (conc 16) | | (no reference) | **196/200 (98.0%)** | | |

- exact.py was run twice in this instance and matched itself (823ce1923c16).
- GSM8K-200 misses are items 93, 100, 119 (truncated, no answer) and 184. There is no earlier GSM8K-200 run to compare against, so this is the reference for future rounds.
- Single-stream varies between restarts by about ±4%. This instance's code number (72.5) is at the low end; across exp5's two A5 instances the single mean was 84.1. A fair estimate for round 2 is single ~81-84 tok/s.

**Round 2 outcome:**
- PROMOTED exp1 moe-compact (DSV41_MOE_COMPACT=1). Unpadded target FC1 (576 rows/expert/rank), bit-exact, -7.5 GiB/rank. Gains: single +5%, conc4-16 +7-10%.
- NOT promoted: exp2 skinny_mx dense MXFP8. It failed the CUDA-graph kernel gate, because autotuned CUTLASS already streams ~200 GB/s. Only WO-A A16 wins (1.4-2.3x), worth about 2% single-stream.
- PROMOTED exp3 rdma-thresholds (RDMA_AR_MAX_REDUCE_BYTES=1048576). It is env-only. Two-shot RDMA all-reduce now covers up to 102 rows, conc16 gains about 5%, and draft time at 16 requests drops 19.1 -> 12.5 ms.
- NOT promoted: exp4 sharded-Markov drafter. Drafts are identical and draft time drops 1.4 ms, but single was only +1.0%. The stock trimmer over-trims when drafting is cheaper.
- PROMOTED exp5 adaptive-verification cost calibration (scale 0.6, fixed 2 ms, flat to M=48, back to profile by 56). Single +6.1%, concurrency +1-2.5%.

**Full rollback to round-1 config** (run from the Mac; stop the service first for site files):
1. Launcher: `cp ~/dsv41-tp8.sh.bak-compact ~/dsv41-tp8.sh` on all 8. 3045aa83 is the round-1 launcher.
2. Site files: `bash $S/r2/moe-compact/uninstall.sh` (deep_gemm_moe/mxfp4/fp8_utils back to 84ecc6ef/a7fc1bcd/013df536) and `bash $D/install.sh av rollback` (adaptive_verification back to b0526416).
3. `bash $S/dsv41-launch-all.sh`.

Per experiment: see the exp1/exp3/exp5 sections above. Env-only partial rollbacks are DSV41_MOE_COMPACT=0, RDMA_AR_MAX_REDUCE_BYTES=524288, and DSV41_AV_VERIFY_SCALE=1 DSV41_AV_FIXED_MS=0.

**Scripts archived:** ./ops/r2/ holds a copy of the Mac scratchpad r2/ tree: moe-compact, dense/skinny_mx, rdma-ext, drafter, av-profile, exp1/exp4/exp5 helpers, and finalsumm.py.

**Next steps (round 3 candidates):**
1. Re-test exp4 sharded-Markov on top of the exp5 calibration. Cheaper drafting should no longer over-trim. With AV off it measured +6% single.
2. DSV41_TARGET_M32_MAX_M=96 together with max-num-seqs 24/32. The verify cliffs at 48/56 are gone (verify at 56 tokens took 119 ms with M32_MAX_M=96), and exp1 freed 7.5 GiB/rank. Also try a larger --kv-cache-memory-bytes (12-14G).
3. A separate RDMA all-gather size cap (~1 MiB), so 48-96-row logits gathers fall back to NCCL, which is faster there.
4. Compact FC2 (K=288). This needs a DeepGEMM mixed-FP4 kernel change, for about another 8% of routed bytes. Also try 64-wide N tiles for M>48, where the kernel gain is only 1.05x.
5. WO-A A16 skinny alone (DSV41_SKINNY_MX_WOA=1), for about +2% single. It needs at least 3 ABAB pairs, and should be compared against Codex's WO-A Marlin.


## Round 3 / e1: sharded Markov drafter + AV calibration (2026-10-22 UTC). PROMOTED (arm M2)

**Change:** `DSV41_DSPARK_SHARDED_MARKOV=1` (the round-2 exp4 patch to `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`, sha 97bcfaa6 -> 774b34c2) together with `DSV41_AV_FIXED_MS` 2.0 -> 3.4. The exp5 AV cost calibration stays live (VERIFY_SCALE 0.6, N0 48, N1 56).
Round-2 exp4 was not promoted only because the uncalibrated trimmer over-trimmed code. With calibration plus FIXED_MS 3.4, the trimmer keeps its previous choices and the drafter saves 1.5 ms per step: draft_ms(1) 5.2 -> 3.73, draft_ms(16) 12.6 -> 9.94.

**A/B** (C = round-2 live config; M1 = +flag; M2 = +flag with FIXED_MS 3.4). All instances were launched with `launch-conc.sh` in the order C1 M1 M2 C2 M1b M2b C3 M2c. Each ran runarm_conc (single code/prose/math, conc4/8/12/16, exact x2, GSM8K-50 at conc 8).

| inst | code | prose | math | mean | c4 | c8 | c12 | c16 | AL code/prose/math | GSM8K | draft_ms(1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 68.06 | 60.75 | 111.31 | 80.04 | 144.67 | 211.95 | 269.12 | 319.13 | 2.50/1.88/3.91 | 50/50 | 5.29 |
| M1 | 72.82 | 61.87 | 114.50 | 83.06 | 147.08 | 216.37 | 276.27 | 321.48 | 2.56/1.80/3.83 | 49/50 | 3.73 |
| M2 | 78.06 | 64.14 | 110.78 | 84.33 | 146.94 | 219.50 | 277.86 | 327.19 | 2.77/1.88/3.68 | 50/50 | 3.72 |
| C2 | 72.24 | 59.36 | 112.89 | 81.50 | 141.45 | 211.57 | 263.36 | 310.49 | 2.68/1.83/3.93 | 50/50 | 5.24 |
| M1b | 70.29 | 61.58 | 113.37 | 81.75 | 150.58 | 218.56 | 277.47 | 322.70 | 2.42/1.82/3.79 | 50/50 | 3.73 |
| M2b | 76.67 | 64.02 | 116.58 | 85.76 | 145.94 | 217.72 | 276.01 | 327.40 | 2.70/1.88/3.92 | 50/50 | 3.74 |
| C3 | 73.39 | 58.43 | 114.98 | 82.27 | 142.23 | 206.58 | 270.35 | 315.19 | 2.73/1.79/4.04 | 49/50 | 5.20 |
| M2c | 78.31 | 62.75 | 116.89 | 85.98 | 148.12 | 215.79 | 272.59 | 318.61 | 2.81/1.85/3.95 | 49/50 | 3.72 |

exact.py was self-consistent in all 8 instances.

**Pooled, control (3 instances) vs arm:**

| | single | c4 | c8 | c12 | c16 | AL code | AL prose | AL math |
|---|---|---|---|---|---|---|---|---|
| C | 81.27 | 142.78 | 210.03 | 267.61 | 314.94 | 2.63 | 1.83 | 3.96 |
| M2 | 85.36 (+5.0%) | 147.00 (+3.0%) | 217.67 (+3.6%) | 275.49 (+2.9%) | 324.40 (+3.0%) | 2.76 (+0.12) | 1.87 (+0.03) | 3.85 (-0.11) |
| M1 | 82.41 (+1.4%) | 148.83 (+4.2%) | 217.47 (+3.5%) | 276.87 (+3.5%) | 322.09 (+2.3%) | 2.49 (-0.15) | 1.81 (-0.02) | 3.81 (-0.15) |

- **M1** fails the gate (single +1.4%, code AL -0.15). With FIXED_MS 2.0 the trimmer still trims code a little too much.
- **M2** passes: single +5.0%, every concurrency level is +2.9% to +3.6%, and code AL is up. The math AL change of -0.11 comes from one instance (M2: 3.68). M2b and M2c (3.92/3.95) are inside the control range (3.91-4.04), and math single runs at drafts_per_step 5.0 in every instance, so it is not trimming.
- **GSM8K-200 at conc16 on M2c:** 197/200, misses idx 93, 119, 184 (reference 196: 93, 100, 119, 184).
- **Long-context check:** not needed. The change touches only the drafter and the AV cost model, not attention/indexer/KV/MoE numerics, and the target model still verifies every token.

**Promotion:**
- Launcher sha: e270215f -> abd49904, identical on all 8. The backup is `~/dsv41-tp8.sh.bak-r3markov` (e270215f) on every rank.
- New lines: `DSV41_AV_FIXED_MS` default 3.4, and `export DSV41_DSPARK_SHARDED_MARKOV=${DSV41_DSPARK_SHARDED_MARKOV:-1}`.
- speculator.py = 774b34c2 on all 8. The backup is `speculator.py.bak-dsmarkov` (97bcfaa6).
- After relaunching with `dsv41-launch-all.sh` (started 04:08:43 UTC): rank0.log shows `SHARDED_MARKOV: vocab shard [0, 16160)`, with no "ignored" warning and no "first call during capture" warning. draft_ms(1) is 3.75, and GSM8K-50 at conc 8 scored 50/50.

**Rollback:**
1. Restore the launcher on every rank: `cp -p ~/dsv41-tp8.sh.bak-r3markov ~/dsv41-tp8.sh`, and check that the sha is e270215f. For a temporary rollback instead, launch with `DSV41_DSPARK_SHARDED_MARKOV=0 DSV41_AV_FIXED_MS=2.0`.
2. Run `bash <Mac scratchpad>/r2/drafter/install.sh markov rollback`, which restores speculator.py to 97bcfaa6.
3. Relaunch.

**Note:** dspark/speculator.py is no longer stock, so the r3 spec-sampling install.sh, which expects sha 97bcfaa6, will refuse to run until it is rebased onto 774b34c2.

Raw results: `~/dsv41-opt-20261021/r3-e1/` (instance dirs, inst.sh, summ2.py, dsumm.py, M2c/gsm200.*, final-gsm50.*).

### Round 3 / e2: FC2 compaction (DSV41_MOE_COMPACT_FC2): NOT promoted, kernel faults, nothing installed

**Goal:** unpad routed-MoE FC2 (288->384) with a separately built DeepGEMM (`~/dsv41-stage/fc2c/native-fc2c-v1`, _C e7088a23, manifest 072fdc49) whose compact-B kernel remaps the B load. Expected about +2-5%, plus about 3.8 GiB per rank freed.

**Offline test** (`run_offline_test.sh`, spark-one, service stopped, 2026-10-22 ~04:15 UTC):
- **kw=288:** the first case (M=1, 32-row path) faults when `sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d_compact_b` launches with `CUDA_ERROR_ILLEGAL_INSTRUCTION (715)` (cuLaunchKernelEx, deep_jit driver.hpp:33). Confirmed with CUDA_LAUNCH_BLOCKING=1.
- **kw=320** (separate process): faults the same way on the same first case.
- **Healthy in the same process:** the padded kernel from the new native is bitwise identical to the live M32 library (native_padded_eq_live=true), and the padded baseline matches the fp32 reference (rel 1.66e-3). The sanitizer fixed 1,473,703 NaN dead scales.
- **Not the cause:** plain torch reductions work on this GPU.
- **Where the fault is:** in the compact-B patch codegen. Both variants JIT-compiled and are cached in `~/dsv41-stage/fc2c/fc2c-jit-cache-v1/cache/*compact_b.{45960e62,cf999d97}` with 27-line kernel.cu each. Untested idea: the TMA box/tensor-map or a PTX variant that sm_121 rejects.
- No 128-row case and no timing was reached.

**Decision:** stopped at offline step 5. `install.sh` was never run (site-packages unchanged) and there was no A/B.

**State after:**
- Service relaunched with `dsv41-launch-all.sh` (health 200); GSM8K-50 at conc8 scored 50/50 (221.8 tok/s).
- On all 8 ranks: launcher abd49904, deep_gemm_moe.py 45418259, mxfp4.py 67339029, sm121_fc2_compact.py absent.

**Rollback:** none needed. Staged files stay in `~/dsv41-stage/fc2c` for a kernel fix.

Raw results: `~/dsv41-opt-20261021/r3-e2/` (offline-test.log, final-gsm50.*).
