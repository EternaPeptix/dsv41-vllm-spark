
## Round 4 / e5: TTFT + step-overhead diagnoses, prefill trio. PROMOTED idxshard only (2026-10-23 ~04:45 UTC)

**Promoted:** prefill-idxshard (env-only on top of installed `mla/indexer.py` 4753707e). Launcher `~/dsv41-tp8.sh` 22e83204 -> **5ccc4b79** on 8/8. It adds `DSV41_IDX_ROW_SHARD_MIN=${..:-1024}` and `DSV41_IDX_ROW_SHARD_MIN_CTX=${..:-32768}` after the VERIFY_CAP line. Backup: `~/dsv41-tp8.sh.bak-r4e5`.
**Not promoted:** prefill-sched (uninstalled, scheduler.py back to d9a39870 on 8/8). prefill-sp: STOP at the debug gate. model-merged stays installed (E1 CERT/MAINPROJ).

### Part A1: TTFT noise (live config, no patch)
- Most of the "16K" spread is how the prompt is built. bench.py's "16K" prompt is really about 23.9K tokens (2 chunks) and its "32K" prompt about 48.7K tokens (3 chunks). TTFT is a step-linear function of the token count (about 5.4-6.1K tok/s plus about 0.1-0.3 s per extra chunk). prefill_bench's tok/word calibration starts low, so its first "16K" rep is 25K tokens.
- **Warm server, sequential cold prompts** (10 x 16K + 5 x 32K, unique salt at the start; A1/ttft.jsonl): every TTFT is within ±4% of the linear fit (5.3-6.1K tok/s). No outliers.
- **The remaining noise is the first long prompt after a restart.** For the same 25,202-token prompt on 6 fresh instances, TTFT was 5.61 / 5.72 s on base/base2 (pf ran first), 4.88 / 4.35 s on thr8k/byp (a 128K starve ran first), and 4.48 / 4.77 s on idx/all. The second prompt was always 3.45-3.50 s (19.7K tokens). The historical K-runs show the same thing: 4.26-7.25 s at a constant 23.9K tokens, measured once per instance.
- **What happens on the first request** (1 Hz sampler on 8/8: vmstat, PSI, worker /proc stat, nvidia-smi):
  - MemAvailable drops about 6 GB on every rank (22.4 -> 16.2 GB), which is the lazy allocation of the prefill workspace.
  - Triton JIT warnings appear in every rank log: `_dsv41_live_cap_kernel` (e4 verify-cap) and `_screen_kernel` / `_refine_kernel` (e1 certified head, `dsv41_r4_small.py`).
  - The worker gets 400-1200 majflt and 600-4800 swap-in pages (at most about 19 MB). Fast and slow instances look the same here.
- **Hypotheses ruled out:**
  - (1) kswapd / reclaim: no direct scan, PSI memory is about 0 during prefills, and the swap-in volume is trivial.
  - (2) Engram: the two 12 GB engram tables are **mlocked and THP, with 0 swap**. The ~0.9 GB that is swapped out is python heap and NCCL /dev/shm.
  - (3) Chunk interleaving: requests ran one at a time.
  - (4) Thermal: SM clock was 1969-1989 MHz on all 8 ranks, temperature ≤ 60 C, and the throttle reasons were 0x0.
- **Conclusion:** the remaining noise is hypothesis (5), first use: lazy workspace allocation plus first-call JIT/autotune. Its cost varies from instance to instance (+0 to +1.3 s).
- **Fix candidate (not applied):** after /health, send one ≥ 24K-token warmup prompt (2 chunks) plus one short greedy request from launch-all or warmup.py. A better variant would add the CERT and live-cap Triton kernels to the engine warmup. This costs nothing for decode. Also, single-sample TTFT should never be gated on the first long prompt.

### Part A2: AL-dependent step cost
Tool: `tools/a2.py` + `an_a2.py`. C1, workloads coding / prose / copy / count, 3-5 reps each.
- **Not API-side.** stream=false vs stream=true wall time is identical (coding 1.4602 vs 1.4567 s; prose 2.801 vs 2.800 s). The API server uses 3.8% CPU. EngineCore sits at about 101%, which is the normal busy-poll, not saturation. The worker uses about 118%.
- **Engine-side, but mostly not per accepted token:**
  - Within a workload, the slope is only **0.6-0.7 ms per accepted token** (coding gap = 30.9 + 0.60·k, prose 27.2 + 0.62·k).
  - Most of the 1.4 ms/token cross-workload slope is a per-workload intercept. copy (AL 2.47) runs at **26.3 ms/step**, but prose at the same 2-3 tokens per step runs at 28.5 ms. count (AL 6.0) runs at **32.9 ms**, below coding with all 6 accepted (34.6 ms).
  - The within-workload jump is concentrated at 5-6 accepted tokens (coding: k=4 31.5, k=6 34.6; prose: k=4 29.2, k=6 35.0).
- With verify-cap and route-live off (R1), OLS gives 26.7 + 1.07·k. So the cap/dead-row skip accounts for part of the slope (rows it kills are the cheap low-AL steps).
- **Interpretation:** step cost follows the content of the 6 verify rows, i.e. the union of experts routed per layer, which sets MoE weight traffic at small M. Repetitive text (copy) touches few experts. Full-acceptance steps also feed back into the next step's drafts and verify width (AV/cap). The truly per-token work (drafter context update over accepted rows, KV slot writes, sampler epilogue, host copy) is at most about 0.6 ms/token.
- **Candidate next step (no env-only fix exists):** log the number of distinct experts per layer per verify step (DSV41 debug counter in dsv4_topk) and regress the gap on it. If it holds, expert-union-aware work (grouping or reordering drafts, or making the cap expert-cost-aware) is the lever, not the epilogue.

### Part B: prefill trio
**B0.** CPU tests PASS (test_sched_cpu, test_idxshard_cpu). Torch-profiler CUDA trace of one cold 16K chunk on rank 0 (B0prof/):
- NCCL AllReduce RING_LL: 18.0%
- sparse_mla_prefill: 11.5%
- mhc_post: 10.5%
- mhc_pre_big_fuse_with_norm: 7.0%
- hc_prenorm gemm: 4.7%
- deep_gemm fp8/fp4 MoE: 13.1%

AR + mHC + norms is about 41%, well above 10%, so B0 allowed prefill-sp.

**Starvation and lone prefill** (st = 128K long + 6 x 300-token shorts + a live decoder; 2 runs each; pf = lone prompt):

| arm | short TTFT p50 s | long TTFT s | decoder ITL inside p50 ms | pf 64K / 128K / 200K tok/s |
|---|---|---|---|---|
| base (x2 inst) | 21.85 / 21.64 / 21.84 / 21.67 | 29.85 / 29.63 / 29.85 / 29.67 | 31-2746 | 5735 / 5572 / 5437 |
| thr8k (stock `--long-prefill-token-threshold 8192`) | 4.33 / 4.20 | 33.46 / 33.31 (+12%) | 1627 / 1647 | 5765 / 5601 / 5463 |
| byp (BYPASS=2048) | 4.53 / 5.05 | 31.65 / 30.85 (+4.4%) | 1691 / 1423 | 5736 / 5580 / 5426 |
| bypdc (+DECODE_CAP=4096) | 2.99 / 2.98 | 39.36 / 38.38 (+31%) | 1004 / 1007 | 5752 / 5590 / (run cut) |

- **Sched gate FAIL.**
  - byp is 4.5x better than base and +4.4% on the long prompt, but its p50 (4.79 s) is worse than thr8k (4.27 s).
  - bypdc beats thr8k (2.98 s) but costs +31% on the long prompt.
  - Root cause: async scheduling has already scheduled the next full 16K step when a short arrives, so a short waits about 1 residual step plus 1 full step plus the bypass step. A 1.5-2 s TTFT needs a smaller steady chunk whenever another request is present, and that is exactly what costs long-prompt throughput.
  - Lone-prompt pf is unchanged in every sched arm.

**idx** (`DSV41_IDX_ROW_SHARD_MIN=1024 DSV41_IDX_ROW_SHARD_MIN_CTX=32768`). Lone prefill tok/s, 3 control instances (base, base2, c3) vs 3 idx instances (idx, all, promo):

| | 64K | 128K | 200K |
|---|---|---|---|
| control | 5737 | 5574 | 5434 |
| idx | 5822 (+1.5%) | **5807 (+4.2%)** | **5828 (+7.3%)** |

The 200K TTFT drops 37.8 -> 35.3 s.
- longctx 4K-200K PASS on both idx instances (needles 25/25, copy 1.000).
- lp compare (prompts ≤ 24.6K, which are below MIN_CTX, so the path is inert): all vs base mean |dlp| 0.040 and all vs base2 0.028, against base-vs-base 0.032.
- Long lp (40K / 70K / 110K, `tools/lplong.py`):
  - on(promo) vs off(c3): first token 3/3, diverge ≥ 24 tokens, mean |dlp| 0.028.
  - on(all) vs on(promo): diverge at 8 tokens.
  - So the idx effect is inside cross-restart prefill nondeterminism. promo and c3 emit identical text.

**Class: exact on decode and short prompts** (exact.py a8c3394af9d2 on every instance). **Greedy-equivalent at ≥ 32K context** (not bitwise-provable, because prefill is not restart-deterministic even base-vs-base).

**sp: STOP.**
- `DSV41_PREFILL_SP compare` (rows 16384, identical on all 8 ranks): hidden rel_l2 **0.664**, residual **0.599**, post_mix 0.19, pre_mix 0.22. That is about 1, which means a missed or doubled reduction.
- The engine then died during init with `CUDA error: unspecified launch failure` on all ranks (log: spdbg-rank0-initfail.log). Not run further.

**Decode non-regression** (runarm_conc + matched C1). C = base2, c3; I = all, promo:

| | single | c4 | c8 | c12 | c16 | matched coding / prose ms | exact |
|---|---|---|---|---|---|---|---|
| base2 (C) | 88.7 | 153.0 | 221.2 | 276.3 | 331.3 | 34.02 / 28.88 | a8c3394af9d2 |
| all (I) | 86.8 | 152.3 | 220.0 | 280.7 | 326.4 | 34.05 / 28.99 | a8c3394af9d2 |
| c3 (C) | 90.4 | 156.4 | 218.7 | 275.4 | 325.6 | 33.94 / 29.21 | a8c3394af9d2 |
| promo (I) | 90.5 | 149.6 | 219.5 | 281.4 | 325.7 | 34.19 / 29.54 | a8c3394af9d2 |
| pooled | 89.55 -> 88.65 (-1.0%) | 154.7 -> 150.9 (-2.4%) | 219.9 -> 219.8 | 275.9 -> 281.1 (+1.9%) | 328.4 -> 326.1 (-0.7%) | 33.98 -> 34.12, 29.05 -> 29.27 | |

- The idx code cannot engage in decode: it needs at least 1024 rows per rank, i.e. a step of at least 8192 rows, and context ≥ 32K. The decode hash is bit-identical, so the c4 -2.4% is the known c4 restart spread (e4 C instances ranged 148.5-157.4).
- **Correctness:** GSM8K-200 conc16 on 'all' was **195/200**, misses {12, 93, 100, 119, 184} (the reference set plus the known item-12 flip), at 302.7 tok/s. GSM8K-50 on the promoted instance: 50/50 at 239 tok/s. Lowest MemAvailable 15.2 GB, against the 4 GB gate.

**Promoted relaunch:** launch-all from the Mac with launcher 5ccc4b79 on 8/8. Site shas identical 8/8: indexer 4753707e, scheduler d9a39870, model.py e7264781. Health 200, exact a8c3394af9d2, GSM8K-50 50/50.

**Incident.** My wait loop timed out and I started the idx arm while bypdc's pf was still running, so bypdc lost its 16K/200K pf rows. The st results are complete and the remaining pf rows match base.

**Rollback** (from the Mac):
- Env only: `bash $S/conc/launch-conc.sh DSV41_IDX_ROW_SHARD_MIN=16384 DSV41_IDX_ROW_SHARD_MIN_CTX=0` (stock threshold).
- Launcher: `cp -p ~/dsv41-tp8.sh.bak-r4e5 ~/dsv41-tp8.sh` on all 8 (back to 22e83204), then relaunch.
- Site file: stop the service, run `bash $S/r4/prefill-idxshard/uninstall.sh` (indexer.py back to 43d4f705; do the launcher rollback first, though the env is harmless with the stock file).

**Proposals (not run):**
1. Post-health warmup prompt, from A1.
2. Expert-union instrumentation, from A2.
3. NCCL protocol for prefill all-reduce. It is 18% of a 16K chunk and NCCL picks RING_LL. Try `NCCL_PROTO=Simple` scoped to messages larger than the RDMA AR cap, measured on pf and decode.
4. Fix prefill-sp's reduction before any rerun.
5. Sched: a smaller steady chunk (8K) only while a short request is waiting, combined with BYPASS, may reach about 3 s at ≤ +5% long.

Raw: `./r4/e5/{A1,A2,R1,B0prof,base,base2,thr8k,byp,bypdc,idx,all,c3,promo}/`, `spdbg-rank0-initfail.log`, `lplong-all-idxon.json`, and `local-data.tgz` (monitor CSVs, profile, tools, launcher).
