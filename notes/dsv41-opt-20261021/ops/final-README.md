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
