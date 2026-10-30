# g1 / stepovh: exact step-overhead levers (knapcio port). Staged 2026-10-26, NOT installed

Offline prep only: no restart, no GPU job, no live site-packages or launcher edit. Staged on 8/8 under
`~/dsv41-stage/g1/stepovh/`. Base = live 04b6738e site files (sha checked by the install script).

| part | flag (default off) | what | class | expected ms/step C1 / C4 / C16 | tests |
|---|---|---|---|---|---|
| A | `DSV41_L2_PREFETCH_WOA=1` (`_WOA_MB=6`, `_WOB_MB=0`) | L2 window during attention: after the last wq_b of each layer, prefetch the wo_a sonar fp8 twin (4.13 MiB) on a 2nd side stream | exact (cache hint) | -0.20..-0.45 (central -0.3) / -0.2..-0.4 / -0.15..-0.4 | CPU planner PASS; GPU test + microbench written, not run |
| B | `DSV41_SYNC_FREE=argmax` (`audit` = check only) | CERT-armed (all-greedy) steps: 16 B/row (max, id) all-gather instead of the [M, 16160] bf16 logits all-gather; one-hot logits at the stock greedy token | exact (NaN rows aside, see B) | -0.05..-0.09 / -0.15..-0.18 / -0.5..-0.85 (greedy steps only) | CPU (Triton interpreter) PASS: 344 rows, 96 rejection_sample comparisons; GPU run not done |
| C | `DSV41_EAGER_GLUE=all,swam` (`swam` is NOT implied by `all`) | the 14 per-KV-group decode SWA metadata launches of a pure-decode step as ONE launch | exact (same stores) | -0.04..-0.05 at every concurrency | CPU (interpreter) PASS: 21 checks; GPU timing not done |
| sum | | | | **-0.3..-0.6 / -0.4..-0.6 / up to -1.3** | |

C1 coding step is 32.1 ms, so the sum is about -0.9 to -1.8% (single tok/s about +1 to +2%). Part A carries most of
it; B matters at concurrency; C is small.

## 0. Where the outside-graph time goes (measured, not guessed)

Codex nsys profile of 2027f4d8 (pinned AV), rank 0, 30 steps per case, `/private/.../scratchpad/g1/prof/` on the Mac
(tools `g1/tools/tl.py`, `tl_dist.py`). DT = drafter graph end -> target graph start, TD = target end -> drafter start.

| case | DT gap | DT busy / launch gaps / host-bound | TD gap | TD busy / gaps / host-bound | eager ops DT + TD |
|---|---|---|---|---|---|
| C1 | 297 µs | 140 / 158 / **0** | 856 µs | 782 / 80 / **0** | 63 + 31 |
| C4 | 329 µs | 169 / 160 / **0** | 1119 µs | 1042 / 82 / **0** | 63 + 31 |

So the outside-graph cost is 1.15 ms at C1 and 1.45 ms at C4 (the k1 "1.5-1.8 ms" was the older 1M profile).
- **TD busy at C1:** CERT screen 494 µs, logits all-gather 79-113 µs (Part B), main_proj split GEMM 62 µs and its
  all-gather 13-30 µs, refine 23 µs, the rejection-sampler chain ~30 µs, three `_prepare_dflash_inputs` launches ~16 µs.
- **DT busy at C1:** 14 `_dsv41_fused_decode_swa` launches (2.0 µs each plus ~2.1 µs gap each, Part C), the AV draft
  budget (`torch.compile`'d topk, 7 ops), the v2-runner input prep (~10 Triton kernels), 2x indexer metadata, block-table
  writes and small H2D memcpys.
- **Host:** the CPU issues the next target graph 24 ms (C1) and 48-88 ms (C4) before the GPU reaches it. Host-bound idle
  is 0 in 62 of 63 C1 intervals and in all 63 C4 intervals; the one exception (158 ms) is a request boundary.

## 1. Lever 1 (SPEC_SYNC_FREE equivalent): what rank 0 broadcasts per decode step

- **Host.** Two pickled RPCs per step go through vLLM's `MessageQueue` (`v1/executor/multiproc_executor.py`
  `collective_rpc`, `distributed/device_communicators/shm_broadcast.py` `enqueue`):
  - `execute_model(SchedulerOutput)`;
  - `sample_tokens(GrammarOutput | None)`.

  Each RPC is one pickle, written to the shm ring for the one local reader (rank 0's worker on spark-one) and sent once
  over a ZMQ XPUB socket to the 7 headless nodes. Only rank 0 replies: there is no KV/EC connector, so no aggregator.
  `take_draft_token_ids` is not called under async scheduling; it runs only on the structured-output deferred path.
- **GPU.** No rank-0 broadcast at all, so knapcio's 9 per-step NCCL broadcasts have no counterpart here.
  - PP=1, so `pp_handler` is None. The r4/spec-sync-free trace audit counted 0 broadcast kernels per step.
  - Every rank samples from the all-gathered logits.
  - Draft tokens come from the sharded-Markov (value, id) all-gather on greedy steps, or from counter-keyed Gumbel
    noise (`DSV41_DRAFT_FRESH_NOISE=1`) on sampled steps.
  - Block-verify uniforms are counter-keyed too (`DSV41_BLOCK_VERIFY_FRESH_U=1`).
- **What could be dropped:**
  - Host RPCs: nothing. The headless ranks have no scheduler, and the host is at least 24 ms ahead (section 0), so
    dropping them gains 0 ms.
  - The one real analogue is a collective whose payload every rank uses only to compute the same tiny result: the
    `[M, V/8]` bf16 logits all-gather on CERT-armed, all-greedy steps. That is Part B.

## 2. Part A: `DSV41_L2_PREFETCH_WOA` (takuz0 #5, knapcio `DSV41_L2_PREFETCH_WOA`)

File: `vllm/distributed/device_communicators/dsv41_l2_prefetch.py` 514dcb23 -> 998af3d7. It needs the main gate
`DSV41_L2_PREFETCH=1`, which is the launcher default.
- **Mechanism.** During each learn forward (the eager warm-up before each capture):
  - every `*.wq_b` module (attention and indexer) gets a forward post-hook that records an ordinal (`L`);
  - a wrapper on sonar's `try_project` records the wo_a bytes the o_proj reads (`woa`): the fp8 twin, scales first
    then codes, or the bf16 grouped weight when sonar declines, i.e. above 32 rows on the torch.bmm path.

  The WOA window is keyed by the last `L` before a `woa` in the same collective window. That is the main wq_b, or the
  indexer's wq_b in indexer layers. At capture, the post-hook of that ordinal forks
  `side2.wait_stream(main)` -> `dsv41_l2pf` with the window's segments (4.13 MiB). It is joined only at the end of the
  forward, so no collective waits on it. A count mismatch at capture skips the remaining WOA windows and logs
  MISMATCH.
- **Exactness.** The prefetch is a cache hint, so no math changes. Collective windows are byte-identical with WOA on
  and off (the `L` and `woa` events carry no items); the CPU test checks this. With the flag off, nothing is hooked or
  wrapped, and the planner equals live 514dcb23 (CPU test).
- **Expected gain.** In k1, sonar wo_a ran from DRAM at 27.3 µs/layer (158 GB/s). L2-served GEMMs in this stack run at
  about 450 GB/s (q_b: 12.0 µs for 5.41 MB), so wo_a should take about 10-12 µs. That is a ceiling of about -15 µs x 40
  layers = -0.6 ms. Applying the usual serving discount (knapcio measured -0.4 ms/step for WOA on TP4) gives -0.2 to
  -0.45 ms at C1.
  - Above 32 rows (C16), 6 of the 8.4 MB bf16 weight is prefetched.
  - `_WOB_MB` (also prefetch wo_b) is measured-flat at knapcio, so it stays 0.
- **Logs:** `DSV41_L2_PREFETCH_WOA plan tokens=N wq_b=48 woa_windows=40`,
  `DSV41_L2_PREFETCH_WOA captured ... forks=40`, and no MISMATCH.
- **Tests.**
  - `tests/test_woa_plan_cpu.py`: PASS, 12 checks.
  - `tests/test_woa_gpu.py`: a 40-layer toy with the real sonar kernel and twin. It checks arms none / coll / woa
    bitwise (eager and graph), forks2=40 and window bytes, and times the replay plus the isolated wo_a L2-cold vs
    prefetched. **Not run** (stopped window).

## 3. Part B: `DSV41_SYNC_FREE=argmax|audit`

File: `vllm/models/deepseek_v41/nvidia/dsv41_r4_small.py` 02f94f29 -> c46e66bd. It needs `DSV41_R4_CERT_HEAD=1`, the
launcher default.
- **Mechanism.** On a CERT-armed step (`step_armed`: every row greedy, no logprobs, no processors, no grammar, no
  watermark, no NaN counting), the only consumer of the target logits is the greedy argmax:
  - the rejection kernel's target argmax;
  - the bonus resample (greedy, so a plain argmax).

  Each rank takes the leftmost max of its certified shard (NaN read as -inf) and all-gathers (value, global id, 0, 0)
  as fp32, the same 16 B row the sharded-Markov drafter gathers. Every rank picks the first rank holding the row's max.
  Ranks hold ascending vocab ranges, so this is the global leftmost max, which is the stock greedy token, ties
  included. The step then returns `[M, org_vocab]` bf16 logits that are -inf except 0 at that token.

  This is the same collective count (1 all-gather), but 16 B/row instead of 32 KB/row.
- **Exactness.** The rejection sampler on the one-hot gives sampled tokens and counts identical to stock. The CPU test
  compares the real `rejection_sample` on full vs one-hot logits over 344 adversarial rows: ties within and across
  shards, -inf shards, block boundaries, all-zero rows, dead drafts.
  - The one exception is NaN rows. Stock maps a NaN block max to -inf at block level, while this path maps NaN
    elements; such a row is a broken forward either way.
  - The collective size depends on arming, so arming must be identical on all ranks. It is: it reads only replicated
    scheduler and sampling state, like the greedy-sharded drafter's graph choice.
  - A one-time TP CPU-group MIN at CERT build disables SYNC_FREE everywhere if any rank lacks CERT tables or has a
    padded or extended vocab shard (fail closed).
  - `audit` keeps the stock gather, adds the compact one, and logs `winner mismatches` every 2000 calls. Run it as the
    first serving screen; it should log 0 on 8/8.
- **Expected gain.** The all-gather costs 79-113 µs at 6 rows, about 205 µs at 24 and 0.6-1.0 ms at 48-64. The compact
  path costs about 15-25 µs, plus about 3 µs for the local argmax and about 5 / 25 / 60 µs for the one-hot fill at
  M = 6 / 24 / 48. Sampled steps are unchanged.
- **Logs:** `DSV41_SYNC_FREE=argmax agreed on every rank`, and `first armed step (M=6): 16 B/row all-gather replaces ...`.
- **Tests.** `tests/test_syncfree.py`: CPU PASS; `--gpu` written, not run.

## 4. Part C: `DSV41_EAGER_GLUE=all,swam`

Files:
- `vllm/v1/attention/backends/mla/sparse_swa.py` 38b16c51 -> 854c3e5f: a new multi-group kernel, `dsv41_swa_batch`,
  and a `defer_ok` argument.
- `vllm/models/deepseek_v41/nvidia/model_state.py` 53e4fe39 -> f21ee2e6: `prepare_attn` runs inside
  `dsv41_swa_batch()`.

- **Mechanism.** On a pure-decode build (`num_prefill_tokens == 0`, the only case where nothing later in the same
  build reads the outputs), `dsv41_fused_decode_swa` queues its arguments. At the end of prepare_attn, ONE launch with
  grid (programs, groups) runs, for each program, the exact body of the per-group kernel on group g's arguments, read
  from a cached int64 table. The table is rebuilt only when the buffer layout changes. Rows with other
  dtypes / layouts, a single group, mixed batches, and graph captures fall back to per-group launches.
- **Exactness.** Same loads, same int32 arithmetic and same stores into the same buffers. The test compares all
  outputs bitwise against per-group launches on poisoned buffers: 1-16 requests, invalid rows, replay_start, short
  sequences, 3 and 14 groups.
- **Expected gain.** 13 fewer launches x about 4 µs = about -0.05 ms/step at every concurrency.
- **Log:** `DSV41_EAGER_GLUE swam: 14 decode SWA groups built in one launch`.
- **Tests.** `tests/test_swam.py`: CPU PASS; `--gpu` (with per-group vs batched timing) written, not run.

## 5. Not built (ranked next candidates)

1. **main_proj early on a side stream.** The DSpark `combine_hidden_states` (cat + MXFP8 quant + 62 µs split GEMM)
   needs only the target aux hidden states, so it could overlap the 494 µs CERT screen. Its all-gather stays on main,
   in order.
   - Estimate: -0.03 to -0.08 ms. The screen and the GEMM are both DRAM-bound, and the GEMM uses only 5 CTAs.
   - Needs edits to model_runner, dflash propose and the r4 ColSplit.
2. **CERT screen at 72% of DRAM bandwidth** (85 MB in 494 µs). A retune to about 85% gives about -0.07 ms at C1. It is
   exact by construction: the certificate makes the argmax independent of screen rounding.
3. **`_prepare_dflash_inputs` x3 -> 1 launch:** about -0.015 ms.
4. **AV draft-budget topk** (`torch.compile`, 7 ops, about 30 µs): one kernel would need inductor-identical cumprod
   rounding and topk tie order. Skipped.
5. **Lever 4 (host gaps):** 0 in steady state (section 0); nothing to port.

## 6. Install, tests, A/B, rollback

```
# stopped window, all 8 (refuses on an active node; G1_ALLOW_ACTIVE=1 stages for the next start):
ssh spark-one cat ~/dsv41-stage/g1/stepovh/ops/g1-all.sh | bash -s install stepovh     # .bak-g1-stepovh backups
ssh spark-one cat ~/dsv41-stage/g1/stepovh/ops/g1-all.sh | bash -s status stepovh
# GPU unit tests + microbench on one stopped node (no install needed; loads the staged files by path):
ssh spark-two 'bash ~/dsv41-stage/g1/stepovh/ops/run-tests.sh gpu'   # results/<host>-<stamp>/summary.txt
# arms (pinned AV, one fixed MLABEL for matched C1; supervisor relaunch):
P=DSV41_AV_COST_JSON=/home/jeweled/dsv41-stage/pdab/av-pin-pdfinal.json
ssh spark-one cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh | bash -s -- --label g1-C1 $P
ssh spark-one cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh | bash -s -- --label g1-A1 $P DSV41_L2_PREFETCH_WOA=1
ssh spark-one cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh | bash -s -- --label g1-Bau $P DSV41_SYNC_FREE=audit
ssh spark-one cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh | bash -s -- --label g1-B1 $P DSV41_SYNC_FREE=argmax
ssh spark-one cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh | bash -s -- --label g1-G1 $P DSV41_EAGER_GLUE=all,swam
# rollback: unset the flag (inert); files: ... g1-all.sh | bash -s uninstall stepovh  (service stopped)
```

- **Order.**
  1. C first, with the bundle installed and no flags. It must give exact.py a8c3394af9d2 with the pinned counters,
     which proves the files are inert with the flags off.
  2. Bau: check `winner mismatches 0` on 8/8.
  3. ABAB A vs C, then B vs C and G vs C. If the window is short, run one combined ABG vs C ABAB.
- **Gates (task rules).** exact.py identical with identical spec counters, and G1 mean or G4 per-session mean +1.5%
  with nothing worse beyond noise.
- **Combined flag string for promotion:** `DSV41_L2_PREFETCH_WOA=1 DSV41_SYNC_FREE=argmax`, plus
  `DSV41_EAGER_GLUE=all,swam`, which replaces the launcher's `all`.
- **The rdma_ar fingerprint hashes the l2 prefetch module**, so install on 8/8 only. g1-all.sh enforces identical
  digests.
