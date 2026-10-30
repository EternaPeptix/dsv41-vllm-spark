
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
