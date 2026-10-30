# rdma_ar offline microbench plan (for the experiment agent; unloaded window only)

Bundle: /private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad/rdma
Bench:  bench/bench_rdma_ar.py (staged to ~/rdma_ar_dsv41/bench_rdma_ar.py on every Spark)

## 0. Preconditions (no GPU work yet)
1. `scripts/stage-rdma.sh` (safe while serving: copies files into ~/rdma_ar_dsv41 only; prints STAGE_OK + identical sha256 on 8 hosts).
   The .so is already built on spark-four from kernel/rdma_ar.cu (sha256 05ed98d3...; .so dfeeab00...). `REBUILD=1` rebuilds it (CPU only, ~1-2 min, nice 10).
2. Window: stop dsv41 on all 8 ranks (`systemctl --user stop dsv41-rank` on each), and check nothing else is on the GPUs/fabric. run-microbench.sh refuses while dsv41-rank is active or any vllm/bench process exists.
3. Note spark-four `rx_discards_phy` and pause counters before and after (`ethtool -S enp1s0f0np0 | grep -E "discard|pause"`), as plan.md item 181 asks.

## 1. Runs (each ~3-6 min; BENCH_TIMEOUT 1200 s guard)
| tag | env | args | purpose |
|---|---|---|---|
| A0 | (defaults: 2-shot at >=32 KiB, VEC_PER_CTA 512, proxy unpinned) | (defaults) | correctness phases 1-3, latency rows 1,2,4,6,12,24,48 (bf16 x 5120), soak 2000x88 ops |
| A1 | `RDMA_AR_PROXY_CORE=19` | `--soak 0` | MiMo's pinned-proxy setting (X925 core) |
| A2 | `RDMA_AR_TWOSHOT_MIN_BYTES=20480 RDMA_AR_VEC_PER_CTA=2048` | `--soak 0` | rdma_ar5 tuning (the other MiMo variant) |
| A3 | best of A0-A2 | `--soak 0 --skew-us 20` | how both backends absorb ~0-20 us per-rank skew (peer-wait proxy) |
| A4 | best | `--rows 1,6,24,48 --trials 5 --soak 0` | repeat for noise (ABBA, the second window) |
| A5 | best + `RDMA_AR_TIMING=1` | `--rows 6,48 --soak 0` | stage / net / tail split (`rdma_stats` in json: n, stage, pub->flags, tail, total, proxy handoff, post, max post, ns) |

Command (from the Mac): `RDMA_AR_PROXY_CORE=19 scripts/run-microbench.sh A1 --soak 0`
Per rank output: `~/rdma_ar_dsv41/results/<tag>.rank<R>.{json,log}`; rank 0 prints the table.

## 2. Pass criteria (all needed before any serving trial)
- Phase 1: every bf16 case (rows 1..48, one-shot and two-shot) and fp32 case (rows 1,6,24) is **bitwise equal** to the fixed-order fp32 reference and has the same hash on all 8 ranks. max_abs_vs_nccl is reported only (NCCL ring order differs, so bf16 can differ by about 1 ulp).
- Phase 2: every routed all-gather is bitwise equal to the PyNccl all-gather, for (6,2) fp32 top-k pairs, the (6,4,512) engram-like shape and (6,16160) fp32 logits shards.
- Phase 3: 50 graph replays mixing rdma one-shot and two-shot, an rdma all-gather and an 8-byte PyNccl AR all succeed with 0 failures.
- Phase 6 soak: 176k ops, no timeout trap, final check OK. No `rdma_ar ... completion error` or `timeout` in any rank log.
- `failures_all_ranks` is all zero. `arena_info[2] == 1` (THP + cudaHostRegister path) on all ranks; `thp_bytes` should be about equal to arena bytes.
- Latency, which is the go/no-go for a serving trial: rdma_auto at 6 rows (60 KiB, the 1-user verify size) at most about 0.4x PyNccl, and at 48 rows (480 KiB) no worse than PyNccl. Expected, from MiMo: 20-40 us at 1-6 rows and about 50-100 us at 24-48 rows, against PyNccl at 160-250 us.
  - If 24/48 rows lose, lower `RDMA_AR_MAX_REDUCE_BYTES` to the crossover for serving (it must be a multiple of 16).

## 3. Projection to a serving step
88 ARs per target step + about 1.9 ms of NCCL in the drafter. Saving ≈ 88 × (pynccl_us − rdma_us) at 6 rows, minus peer wait. Use A3 (skew) to discount: if the rdma advantage collapses at 20 us skew, the gain is bounded by load imbalance, not transport.

## 4. Afterwards
Restart dsv41 with the launch-all helper and check rank0.log "Application startup complete". Run the standard bench once to confirm the baseline came back. Do not leave the bench processes running: the script prints any leftovers, and each process exits via os._exit with a timeout of 1200 s.

## 5. Serving trial (later, separate window; needs install-rdma.sh)
`scripts/install-rdma.sh` (backs up to .bak-rdma, refuses on unknown hashes, verifies identical sha on all 8), then relaunch all ranks with
`DSV41_EXTRA` unchanged and the env `DSV41_RDMA_AR=1 RDMA_AR_AUDIT=400` (+ best knobs from section 1) exported into the unit. That means adding `--setenv=DSV41_RDMA_AR=1 --setenv=RDMA_AR_AUDIT=400 ...` to the systemd-run line, or exporting them in ~/dsv41-tp8.sh behind a flag. The launcher `unset`s stale NCCL_*/GLOO_* but not RDMA_AR_*/DSV41_RDMA_AR, so check that /proc/<pid>/environ carries them.
Check in rank logs: `rdma_ar enabled: ... arena=[..., ..., 1]`; `Using ['RDMA_AR', 'PYNCCL'] all-reduce backends`; the audit lines (ops, shapes, dtypes, `capturing=True` during capture, and the set of streams; any stream other than the eager main stream and the capture stream is a red flag).
Gates: gsm8k_gate >= 49/50; exact.py probe run twice, where the rdma arm must be self-consistent (hashes are expected to differ from the NCCL baseline because the reduction order changes); bench single/conc4/conc8 against the baseline (57 / 102 / 147 tok/s).
Rollback: `scripts/uninstall-rdma.sh` and relaunch, or simply relaunch without DSV41_RDMA_AR (the installed patch is inert with the flag unset).
