# ROUND-4 EXPERIMENT QUEUE (DSv4.1-Flash TP8, vLLM 0.32)

S = /private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad.
Written 2026-10-22 ~05:40 UTC, while r3 is still running. **Start r4 only after r3 has finished** (the user said "let r3 finish").

## 0. State when this was written (read-only checks on all 8 ranks)

**Live state.** The service has been up since 05:27:56 UTC, which is an r3-e3 arm. The launcher is abd49904 (e1 promoted). spark-one has 13G free and spark-two 16G.

**All 20 target site files have the same sha on all 8 ranks, and they equal the "original" sha each r4 installer expects:**

| file | sha |
|---|---|
| utils/flashinfer.py | c0db917f |
| fused_moe/oracle/mxfp4.py | 10f0ebef |
| utils/b12x.py | 99bc2e6d |
| fused_moe/b12x.py | f60c0622 |
| dsv4_topk.py | 14f7c848 |
| fused_topk_bias_router.py | 8cf50cf2 |
| adaptive_verification.py | ec240369 |
| model_runner.py | fc6d5d6d |
| nvidia/model.py | 5fe8b4a7 |
| rdma_all_reduce.py | 3a128082 |
| sparse_swa.py | 2afd3136 |
| dflash/speculator.py | b731b09c |
| scheduler.py | d9a39870 |
| mla/indexer.py | 43d4f705 |
| r3-e4 targets: attention.py | c39e06c1 |
| r3-e4 targets: logits_processor.py | 35cf8235 |
| r3-e4 targets: dspark.py | 480149ff |
| r3-e4 targets: o_proj.py | f2893ee6 |
| dspark/speculator.py | 774b34c2 (Markov, live) |
| rejection_sampler_utils.py | 99556e84 |

Raw output: `S/r4/qwork/sha-*.txt`.

**r3 outcomes so far:**
- **e1 PROMOTED.** `DSV41_DSPARK_SHARDED_MARKOV=1` + `DSV41_AV_FIXED_MS=3.4`.
  - Single 81.3 → 85.4 (+5.0%), conc +3–3.6%.
  - draft_ms(1) went 5.2 → 3.7, so the 1-user step is now ≈ 32.0–32.3 ms, not 33.5.
- **e2 NOT promoted.** The compact-FC2 kernel faults with CUDA 715 at both kw=288 and kw=320. Nothing was installed.
- **e3 RUNNING** (conc/KV). Preliminary summary:

  | arm | conc24 | conc32 | GSM-200 |
  |---|---|---|---|
  | K0 | 300 | 304 | |
  | K1 (32 seqs + KV 16G-class) | 385 | 410 | 196 |
  | K2 (+ M32_MAX_M=96) | 404 | 461 | 195 |
  | K3 (+ RDMA 2 MiB) | 412 | 476 | 195 |

  K0b is in progress, and promotion has not happened yet.
- **e4 (dense-fix + sonar o_proj) and e5 (sampled spec arms): still queued in r3.**

**Precondition R4-P (run at the start of r4):**
1. Re-run `S/r4/qwork/shas.sh` on all 8 ranks and record the launcher sha.
2. e4 and e5 change attention.py, logits_processor.py, dspark.py, o_proj.py and rejection_sampler_utils.py. **None of these are r4 files**, so no r4 installer is affected.
3. If e3 promoted 32 seqs / M32_MAX_M=96, apply the "e3-promoted" notes in each experiment below. They extend the M ranges from 96 to 192 rows.

## 1. Promotability classes

- **X — exact / greedy-equivalent (bitwise, or a proof plus a runtime self-check). Promotable.**
  l2-prefetch, eager-glue, MAINPROJ_SPLIT, CERT_HEAD, verify-router (ROUTE_LIVE / VERIFY_CAP / INTRA_BUCKET), prefill-sched, prefill-idxshard.
- **R — same quantised operands, only the fp32 accumulation or reduction order differs (no new quantisation). Conditionally promotable.**
  b12x-smallm, prefill-sp.
  - This is the same class as r3-e4's skinny qkv_a and sonar o_proj.
  - Promote only if the strict quality gate passes (GSM8K-200, longctx, AL, and the logprob compare where available).
  - Promote only if r3-e4's outcome or the orchestrator accepts the reorder class. Otherwise, report and do not promote.
- **L — lossy (different quantisation or approximation). Optional arms only; never promoted without the user.**
  b12x-moe (MXFP8 k32 activations vs FP8 k128), the EP falsification, and the r3 FP8 *target* head.

## 2. Dedupe against r3

| r4 candidate | r3 status | r4 action |
|---|---|---|
| WO_A_W8 (knapcio) | r3-e4 D3 `DSV41_SONAR_OPROJ=fp8` | **not in r4** (r3 owns it) |
| DRAFT_HEAD_FP8 | r3-e4 D2 `DSV41_FP8_LMHEAD=draft` | **not in r4** |
| REPLICATED_SPLIT wqkv_a | r3-e4 D1 `DSV41_QKV_A_SHARD` | **not in r4**. Suggestion for r3 owners: reuse MAINPROJ_SPLIT's per-M bitwise self-check. |
| REPLICATED_SPLIT engram.wkv | already column-parallel | dropped |
| DRAFT_MAIN_PROJ_SPLIT | new | **E1** |
| CERT_HEAD | new | **E1** |
| b12x MoE (`--moe-backend b12x`) | r3 planner dropped it (not ready; conflicted with FC2) | r4 b12x-moe makes it ready, but it is class L → **O1, offline bench only** |
| `--linear-backend b12x` (sonar) | dropped by r3 | replaced by the overlay-free, M-gated **b12x-smallm (E3)** |
| FC2 compaction | r3-e2 FAULTED | not in r4. Kernel fix carried over; b12x has no exact N%128==32 path either. |
| Markov drafter | r3-e1 PROMOTED | live baseline |
| conc/KV/RDMA 2 MiB reduce | r3-e3 running | E4 covers only the separate *gather* var (`RDMA_AR_MAX_BYTES`) |
| probabilistic draft / block verify | r3-e5 queued | verify-router + block verify only together with e5's FRESH_U/FRESH_NOISE flags (E2 note) |
| longctx PR #1788 | r3 audit: no bug, PASS to 250K | none (padded-decode weight packing stays a hardening item; unreachable with VARLEN_INDEXER=1) |
| EP4/EP8 (rhys) | — | **dropped** (modelled −15..−45% MoE at 1 user; loses M32 and compact). Falsification env kept, not queued. |
| RoCEnante, resident engram, SPEC_SYNC_FREE, SHARED_PAD_K, HC_FUSED, chunked indexer | — | dropped (≈0 at TP8, already have it, or not applicable) |

## 3. File-touch matrix and conflict resolution

| idea | site files (under site-packages/vllm) | installer |
|---|---|---|
| b12x-smallm | utils/flashinfer.py | `S/r4/rhys-ep/b12x-smallm/install.sh apply` |
| b12x-moe (O1) | fused_moe/b12x.py, fused_moe/oracle/mxfp4.py, utils/b12x.py | `S/r4/b12x-moe/install.sh` |
| verify-router | fused_moe/router/dsv4_topk.py, router/fused_topk_bias_router.py, v1/worker/gpu/spec_decode/adaptive_verification.py, v1/worker/gpu/model_runner.py | `S/r4/verify-router/install.sh` |
| **model-merged** (small-kernels + prefill-sp) | models/deepseek_v41/nvidia/model.py, nvidia/dsv41_r4_small.py (new) | **`S/r4/model-merged/install.sh`** |
| l2-prefetch | distributed/device_communicators/rdma_all_reduce.py, dsv41_l2_prefetch.py (new) | `S/r4/l2-prefetch/install.sh` |
| eager-glue | v1/attention/backends/mla/sparse_swa.py, v1/worker/gpu/spec_decode/dflash/speculator.py | `S/r4/eager-glue/install.sh` |
| prefill-sched | v1/core/sched/scheduler.py | `S/r4/prefill-sched/install.sh` |
| prefill-idxshard | v1/attention/backends/mla/indexer.py | `S/r4/prefill-idxshard/install.sh` |

**File-level conflict (the only one), now resolved.** small-kernels (b403734f) and prefill-sp (67ea8ef5) both patch `nvidia/model.py` from 5fe8b4a7.
- `git merge-file` gave one conflict: both insert a flag block after `logger = ...`. I resolved it by keeping both blocks.
- All 19 + 247 added lines are present in the merge, and it passes ast.parse.
- Result: `S/r4/model-merged/` with `patched/model.py` (e7264781), `dsv41_r4_small.py` (02f94f29), `model-merged.diff`, `MANIFEST`, and `install.sh`/`uninstall.sh` (through `common/r4install.sh`: preflight 8/8, `.bak-r4-model-merged`, verify 8/8).
- **Do not run** `small-kernels/install.sh` or `prefill-sp/install.sh`. They would refuse after the merge anyway.
- Not yet staged on the Sparks; install.sh stages it.
- Run `py_compile` on one Spark CPU after install.

With that, no site file is shared between any two r4 bundles, or between r4 and r3-e4/e5. All r4 bundles can be installed together, and every one is inert with its flags unset.

**Runtime interactions (not file conflicts). Handle them in the arms:**
1. **MAINPROJ_SPLIT × b12x-smallm.** The split calls `vllm_flashinfer.mm_mxfp8(..., backend="cutlass")`, which the b12x router also intercepts. With both on, the per-M bitwise self-check compares b12x(slice) with b12x(full).
   - If you see "not bit-identical" lines, that M stays on stock, which is safe but loses MAINPROJ's gain.
   - The two also chase the same 0.3 ms: b12x-smallm also speeds stock main_proj.
   - In E3, read the MAINPROJ log lines and count the gain once.
2. **verify-router ROUTE_LIVE=skip × b12x-moe.** b12x has never been fed -1 expert ids. Never combine them (O1 stays separate).
3. **CERT_HEAD × r3 `DSV41_FP8_LMHEAD=target|both`** (lossy, r3 optional). Incompatible. `=draft` is fine.
4. **l2-prefetch × r3-e4 QKV_A_SHARD.** QKV_A_SHARD adds about 35 in-graph all-gathers, so the plan's collective count rises above 85. The plan is learned at warm-up, so that is fine, but check the "plan tokens=6 ... collectives=N" line and that there is no MISMATCH.
5. **verify-router × r3-e5.** A VERIFY_CAP arm at temperature > 0 with block verify is valid only with `DSV41_BLOCK_VERIFY_FRESH_U=1 DSV41_DRAFT_FRESH_NOISE=1`.
6. **prefill-sp / prefill-idxshard decide collectives**, so the env must be identical on all 8. The launchers already guarantee this.
7. **If e3 promoted 32 seqs:** verify rows reach about 192.
   - CERT_HEAD_M, b12x-smallm (≤128), MAINPROJ (≤128) and l2-prefetch (≤128 tokens) fall back to stock above their caps. That is safe.
   - The b12x-moe bench should add M=192.

## 4. Queue (in order). Wall-time assumptions are from r3:
- One serving instance with `runarm_conc.sh … 8` takes about 10.5 min, startup included (r3-e1 .chk timestamps).
- An instance with GSM8K-200 + longctx takes about 20 min.
- Restart-to-restart spread is ±4%, so every promotable decode arm needs at least 3 alternating control/candidate pairs, unless the stack effect is ≥ 5%.
- **Low-noise secondary metric for target-graph changes:** the AV cost profile in rank0.log (`verify_ms=[(1, …), …]`, as in r3-e3 profile.txt). Record verify_ms(1/2/6/48) and draft_ms(1/16) for every instance.

Common gates (G-common, every instance):
- GSM8K-50 ≥ 49/50.
- exact.py self-consistent.
- AL per type within ±0.08 of pooled control.
- No "first call during capture" warning.
- No new JIT messages after the first request.
- MemAvailable ≥ 4 GB on all ranks.

Promotion-grade gate (G-promote, on the best candidate instance):
- `python3 ~/gsm8k_gate.py <dir>/gsm200.jsonl 200 16` ≥ 195 (reference 196/197; misses 93/100/119/184).
- `python3 ~/dsv41-bench/longctx.py --out <dir>/longctx.jsonl` ends in LONGCTX PASS. Required for anything touching target numerics, attention or prefill.

Launch: `bash S/conc/launch-conc.sh VAR=val …` (identical env on all 8). Bench: `ssh spark-one 'nohup bash ~/dsv41-bench/runarm_conc.sh ~/dsv41-opt-20261021/r4-<tag>/<INST> <INST> 8 >/dev/null 2>&1 &'`, then poll `<dir>/DONE`. (Use 16 and add conc24/32 if e3 promoted 32 seqs.)

---

### R4-0: post-r3 baseline + matched benchmark (HTTP only, no restart). About 25 min.
- **Tag:** r4-0-baseline. Flags: none. Class: measurement.
- **Cmd:** `ssh spark-one 'cd ~/dsv41-stage/r4/rhys-ep && ~/venvs/vllm032/bin/python matched_bench.py --label post-r3 --out ~/dsv41-opt-20261021/r4-rhys-ep/matched-post-r3.json'` (about 10 min).
  - Then `runarm_conc.sh ~/dsv41-opt-20261021/r4-0/C0 C0 8`.
  - Also save the exact.py output hash; it is the reference for the inertness check in W1.
- **Purpose:**
  1. Establish the real engine gap against rhys101's 132 / 509 tok/s on their own prompts (merge_intervals: 47 tokens in, 200 out, thinking off). Clone-batch C8 is measured against `coding_div` C8.
  2. Re-anchor the step breakdown after e1, e3 and e4.
- **Decision use:** if coding C1 ≥ ~110 tok/s, the headline gap is mostly benchmark. Report it and do not chase EP. If it is < 100, compare the implied ms/step with their ~22–26 ms.

### W1: maintenance window. Install all r4 bundles (inert) + offline GPU unit tests in parallel. About 60–75 min.
**Step 1.** Stop dsv41-rank on all 8: `for h in one two three four five six seven eight; do ssh spark-$h 'systemctl --user stop dsv41-rank'; done`.

**Step 2.** Install everything from the Mac with the service stopped. Each installer preflights 8/8 original shas, backs up as `.bak-r4-<idea>` and verifies 8/8:
```
bash S/r4/rhys-ep/b12x-smallm/install.sh stage && bash S/r4/rhys-ep/b12x-smallm/install.sh apply
bash S/r4/verify-router/install.sh
bash S/r4/model-merged/install.sh      # instead of small-kernels + prefill-sp
bash S/r4/l2-prefetch/install.sh
bash S/r4/eager-glue/install.sh
bash S/r4/prefill-sched/install.sh
bash S/r4/prefill-idxshard/install.sh
# b12x-moe (class L) is NOT installed here; its bench runs from the stage dir.
ssh spark-one '~/venvs/vllm032/bin/python -m py_compile ~/venvs/vllm032/lib/python3.12/site-packages/vllm/models/deepseek_v41/nvidia/model.py'
```

**Step 3.** Run the GPU tests concurrently, one per host. Each uses one GPU and ≤ ~10 GB.

| host | test | pass criteria | approx. time |
|---|---|---|---|
| spark-one | `cd ~/dsv41-stage/r4/small-kernels && ~/venvs/vllm032/bin/python gpu_test_r4.py --out gpu_test_r4.json` | all bound / threshold / argmax / computed-logit violation counters = 0, accumulation margin ≪ 1; main_proj slices bit-identical; **record the printed `DSV41_R4_CERT_HEAD_M=<list>`** | 15 min |
| spark-two | `cd ~/dsv41-stage/r4/b12x-smallm && ~/venvs/vllm032/bin/python test_b12x_smallm.py --quick`, then without `--quick` | routing correct; rel err vs CUTLASS ≤ 0.01 and vs bf16 ≤ 0.05; graph replay OK; record µs and GB/s per N at M=6/48/96 and any N that regresses (→ `SKIP_N`) | 10 min |
| spark-three | `source ~/venvs/vllm032/bin/activate; cd ~/dsv41-stage/r4/verify-router && python3 test_verify_router.py all && DSV41_SM121_TARGET_M32=1 DSV41_MOE_COMPACT=1 VLLM_USE_DEEP_GEMM=1 DSV41_TARGET_M32_NATIVE_DIR=$HOME/dsv41-stage/m32/target-m32-native-v1 python3 test_verify_router.py moe` | exit 0; live rows bit-identical in skip and anchor; **record ms saved per dead row × 43 layers** (this sets E2's expected gain) | 10 min |
| spark-four | (O1, class L, informational) `bash ~/dsv41-stage/r4/b12x-moe/run_bench.sh` (add M=192 if e3 promoted 32 seqs) | O1 gate below; first JIT takes several minutes | 30–40 min |
| spark-five | `source ~/venvs/vllm032/bin/activate; cd ~/dsv41-stage/r4/l2-prefetch; DSV41_L2_PREFETCH=1 python3 test_l2pf_gpu.py` | graph outputs bitwise equal at M=1/6/24/96; record µs saved per window | 5 min |
| spark-six | `source ~/venvs/vllm032/bin/activate; python3 ~/dsv41-stage/r4/eager-glue/test_eager_glue_gpu.py` (needs the install above) | bit-exact vs stock; GPU-side µs | 5 min |

The handoffs list the small-kernels and verify-router tests on spark-one/three; any host works, since all ranks have the weights and the M32 native dir.

**Step 4.** Drop any idea whose unit test fails: relaunch without its flag, and uninstall it at the end of r4.

**Step 5.** Inertness control C-W1: `bash S/conc/launch-conc.sh` with no flags.
- `rank0.log` must show none of the r4 "active" lines.
- Run `runarm_conc.sh ~/dsv41-opt-20261021/r4-w1/CW1 CW1 8`.
- Numbers must be within ±4% of R4-0.
- exact.py hash must equal R4-0's, or match the known cross-restart behaviour.

If C-W1 fails, bisect by uninstalling bundles (most recent first) before going further.

---

### E1 `r4-e1-bitstack`: exact small-kernel stack. Class X. Expected +3.5–5.5% single, +1–2% conc16.
**Components and expected gain** (1-user step ≈ 32 ms after e1):

| component | flag | ms/step at 1 user | other loads |
|---|---|---|---|
| L2 prefetch during the 85 RDMA collectives | `DSV41_L2_PREFETCH=1` | −0.4..−0.9 | −0.3..−0.8 at c16 |
| DSpark main_proj column split, bitwise self-checked | `DSV41_R4_MAINPROJ_SPLIT=1` | −0.28..−0.32 | all loads |
| Screened target lm-head, argmax-identical | `DSV41_R4_CERT_HEAD=1 DSV41_R4_CERT_HEAD_M=<W1 list>` | −0.28..−0.33 | −0.1..−0.3 at c16 |
| Fused SWA metadata + 1024-wide dflash pad loops | `DSV41_EAGER_GLUE=all` | −0.15 | −0.14 at c16 |
| **Sum** | | **−1.1..−1.7 ms (+3.5–5.5% single)** | **≈ −0.7..−1.4 ms (+0.6–1.3%) at c16** |

Why one stack: each component alone is below the ±4% restart noise. All four are bitwise or greedy-equivalent, so the stack's greedy output should match control (a strong gate).

**Arms** (FLAGS = the four flags above):
1. **E1-chk (one instance, about 15 min, not timed):** FLAGS with `DSV41_R4_CERT_HEAD=check DSV41_R4_CERT_HEAD_LOG_EVERY=500`. Run exact.py and GSM8K-50 only. Required in the logs:
   - `grep 'DSV41_R4_CERT_HEAD rank' ~/dsv41-logs/rank*.log` shows argmax, value and computed-logit mismatches = 0 and armed calls > 0. Record the tiles/call (full is 1010).
   - "DSV41_R4_MAINPROJ_SPLIT active" and no "not bit-identical".
   - l2 "plan tokens=6 … collectives=85" (more if r3 QKV_A_SHARD is live) and no MISMATCH.
   - EAGER_GLUE active.
2. **Timing, alternating:** C, S, C, S, C, S (S = FLAGS with CERT_HEAD=1), each `runarm_conc.sh … 8`.
3. **Ablation only if S is flat or negative:** L2-only, then S minus L2 (one instance each), to find the culprit. L2 knobs if L2 is suspect: `DSV41_L2_PREFETCH_MB=4|8`, `DSV41_L2_PREFETCH_SKIP_N=512`.

**Promotion gate:**
- G-common.
- exact.py output of S identical to C, or differences no larger than C-vs-C across restarts.
- G-promote on the best S.
- Pooled single ≥ +2.5% and c4/c8/c16 not worse than −1%.
- Also: verify_ms(1) from the AV profile is lower on S than on C (catches L2 and eager-glue in-graph effects).

**Promote:** add `export` defaults for the four flags (with the qualified CERT_HEAD_M) to the launcher. Back up as `.bak-r4e1`, verify sha on 8/8, relaunch, run sanity GSM8K-50, and add a README "Round 4 / e1" section.

**Rollback:** relaunch without the flags; uninstall model-merged, l2-prefetch and eager-glue at the end of r4 if not promoted. Keep model-merged if E5 prefill-sp still needs it.

**Wall time:** about 15 + 6×10.5 + 20 (G-promote) ≈ 1.7 h, or +25 min with ablation.

### E2 `r4-e2-verifycap`: fresh-confidence verify cap + dead/padding-row expert skip. Class X. Expected +0.5–2.5% single (prose highest), +1–4% c4–c8, 0–2% c16.
- **Flags:** `DSV41_ROUTE_LIVE=skip DSV41_VERIFY_CAP=conf:0.1 | auto`, plus `DSV41_VERIFY_CAP_LOG=500`. Optional `DSV41_AV_INTRA_BUCKET=0.7`.
- **Base:** the live config, including E1 if promoted.
- **Unit-test dependency:** W1 spark-three PASS. Scale the expected gain by the measured ms per dead row.
- **Arms** (`bash S/r4/verify-router/launch-arm.sh <arm> [E1 flags if promoted]`): ctl, rl, vc10, ctl, vc10, vcauto, ctl, best(vc10|vcauto).
  - Then, only if vc10 or vcauto wins: vc10ib, plus vc05/vc20 for a single instance each.
  - `rl` is a one-instance exactness/neutrality guard (expected ≈0). Its exact.py output should equal ctl's.
- **Per-instance check:** `grep 'DSV41 verify-cap' ~/dsv41-logs/rank0.log` for dead-draft % and mean T. Dead% must be > 0, or the arm is a no-op.
- **Promotion gate:**
  - G-common and G-promote; also `gsm8k_gate.py` with the 200-item misses inside the reference set.
  - Pooled single ≥ +1.5% with conc ≥ −1%, or conc4/8 ≥ +2.5% with single ≥ −1%.
  - AL may *drop* by design: dead rows cap acceptance. Gate on tok/s and GSM8K, not on AL ±0.08. Report AL anyway.
  - If e5 promoted sampled arms, also run r3's `gsm8k_sampled_gate.py` on the winner.
- **Promote:** launcher exports `DSV41_ROUTE_LIVE=skip` and `DSV41_VERIFY_CAP=<winner>`; `.bak-r4e2`.
- **Risk to watch:** any cross-rank confidence divergence shows up as an AL collapse or garbage. If seen, rerun with `DSV41_VERIFY_CAP_TORCH=1` and stop.
- **Wall time:** 8 × 10.5 + 20 ≈ 1.75 h, plus up to 35 min for the extras.

### E3 `r4-e3-b12x-smallm`: small-M dense MXFP8 → FlashInfer built-in b12x. Class R. Expected −1.0..−2.2 ms at all loads (+3–6% single, +1–2% c16).
**Overlap with r3-e4:**
- If r3-e4 promoted QKV_A_SHARD, the qkv_a share (2.22 ms) is gone. Expect about −0.7..−1.5 ms.
- If r3-e4 also promoted sonar o_proj, that is a bmm, not mm_mxfp8, so there is no overlap.

**Flags:** `DSV41_MXFP8_B12X_MAX_M=128`. Optionally `DSV41_MXFP8_B12X_SKIP_N=<N that regressed in W1>`. Control: the same launch with `=0`.

**Unit-test dependency:** W1 spark-two PASS. If no N shape beats CUTLASS at M=6, skip E3 entirely.

**Arms:** C, B, C, B, C, B, where B = the live config (E1/E2 winners) + `DSV41_MXFP8_B12X_MAX_M=128`. On each B instance, check rank0.log for:
- the `DSV41_MXFP8_B12X:` line;
- that startup did not JIT inside capture (startup time compared with C);
- whether MAINPROJ_SPLIT self-check lines changed (interaction 1).

**Promotion gate:**
- G-common and G-promote (longctx mandatory).
- AL within ±0.08.
- Pooled single ≥ +2.5% with conc not worse than −1%; verify_ms(1) lower on B.
- Class R: promote only if the reorder class is accepted (section 1). Otherwise, report.

**Promote:** export `DSV41_MXFP8_B12X_MAX_M=128` (+ SKIP_N); `.bak-r4e3`.

**Rollback:** `=0`, plus `S/r4/rhys-ep/b12x-smallm/uninstall.sh` at the end of r4.

**Wall time:** 6 × 10.5 + 20 ≈ 1.4 h.

### E4 `r4-e4-draftgather-rdma`: env-only, move the c16 drafter NCCL all-gather onto RDMA. Class X. Expected about −0.6 ms/step at c16 (+0.5% aggregate), 0 at c1–c8.
- **Step 1:** audit instance `bash S/conc/launch-conc.sh RDMA_AR_AUDIT=400`, run conc16 only, and read the gather bytes in the rank logs.
  - Re-audit is needed because r3-e1 changed the drafter, and r3-e3 K3 may have changed `RDMA_AR_MAX_REDUCE_BYTES` already.
  - Proceed only if the drafter gather is still on NCCL and its bytes are ≤ 4 MiB.
- **Step 2:** arms C / G / C / G at conc16 (and conc32 if promoted), with G = `RDMA_AR_MAX_BYTES=4194304`.
- **Gate:** G-common; conc16 aggregate ≥ +1.5% pooled; no rdma errors; GSM8K-200 at conc16 ≥ 195.
- **Wall time:** about 1 h. Lowest priority among the decode items. Skip it if the audit shows nothing on NCCL.

### E5 `r4-e5-prefill`: prefill scheduling, indexer row shard and prefill SP. A separate metric (TTFT / prefill tok/s, starvation); decode must not regress.
Tooling: `S/r4/prefill/run_ab.sh <label> "<ENV>" [vllm args]` (restart + suite). Results go to `spark-one:~/dsv41-bench/r4prefill/<label>/`. Read `S/r4/prefill/README.md` first.

Patches are installed in W1 (model-merged carries prefill-sp). Expected gains and classes:
- **prefill-sched** (class X): short-request TTFT during a 128K prefill falls from about 5–20 s to about 1.5–2 s. Decoder ITL during a long prefill: 3 s → 0.8 s with DECODE_CAP.
- **prefill-idxshard** (class X): prefill +3–5% at 128K, +5–9% at 200K+.
- **prefill-sp** (class R): prefill +10–25%.

Steps, in order:
1. Step-0 nsys profile of one cold 16K chunk (about 10 min, per the README).
2. `base` (run twice, for `lp` noise).
3. `thr8k`: the stock `--long-prefill-token-threshold 8192` through DSV41_EXTRA. No patch needed; this is the control for sched.
4. `byp "DSV41_PREFILL_BYPASS=2048"` and `bypdc "DSV41_PREFILL_BYPASS=2048 DSV41_PREFILL_DECODE_CAP=4096"`, suite `st pf`.
5. `idx "DSV41_IDX_ROW_SHARD_MIN=1024 DSV41_IDX_ROW_SHARD_MIN_CTX=32768"`, suite `pf lc lp`, then `prefill_bench.py compare base/lp.json idx/lp.json`.
6. `spdbg "DSV41_PREFILL_SP=1 DSV41_PREFILL_SP_DEBUG=3"`, suite `pf`. Gate: `grep 'DSV41_PREFILL_SP compare' ~/dsv41-logs/rank*.log` on all 8 shows rel_l2(hidden, residual) ≤ 2e-2. A value near 1 means a missing or doubled reduction: stop.
7. `sp "DSV41_PREFILL_SP=1"`, suite `pf lp lc gs`.
8. `all`: the winners combined, suite `pf st lp lc gs`, plus one `runarm_conc.sh` decode non-regression instance.

**Gates:**
- **sched:** short-request p50 TTFT ≥ 3× better than base and not worse than thr8k; long-prompt TTFT ≤ +5%.
- **idx:** pf ≥ +3% at 128K, logprob compare within base-vs-base noise, longctx PASS.
- **sp:** pf ≥ +8% at 64K/128K, logprob compare within 2× base-vs-base noise, longctx PASS, GSM8K-50 ≥ 49, MemAvailable ≥ 4 GB at 200K (spark-one and spark-two are tight). Class R promotion rule applies.
- **all:** decode single and conc within ±2% of the current control.

**Promote:** export the winning envs; `.bak-r4e5`.

**Wall time:** about 9 restarts × (7 min startup + 15–25 min suite) ≈ 4–4.5 h. It can be deferred or run as its own block, because it does not interact with the decode experiments (prefill-sp and idxshard only engage on ≥ 2048-row eager forwards; l2-prefetch only on ≤ 128 tokens).

---

### Optional, non-promotable (class L), clearly labelled

**O1 `r4-o1-b12x-moe` (offline bench result from W1).**
- **What it is:** fused W4A8 b12x routed MoE, 288 → 320 pad onto the compact-N64 path, deterministic.
- **Expected:** 0 to −1 ms at 1 user; ±5% of MoE at c16. Bytes are identical to today; any gain comes only from fusion.
- **Bench gate:** `b12x_n64_pin_det` beats `dg` per layer at M=6 and M=12 and is not worse at 48/96 (and 192), and its error is ≤ max(1.25× dg, dg + 2e-3).
- **If and only if the bench gate passes,** it may be run as an *optional* serving arm. It is never promoted without the user, because it changes activation quantisation:
  1. `bash S/r4/b12x-moe/install.sh` (service stopped).
  2. `bash S/conc/launch-conc.sh DSV41_MOE_B12X=1 B12X_COMPILE_CACHE_DIR=/home/jeweled/dsv41-stage/r4/b12x-moe/b12x-compile-cache` vs control, 2 pairs.
  3. Never with `DSV41_ROUTE_LIVE`.
  4. `uninstall.sh` afterwards.
- **Known risk:** the overlay expects cutlass-dsl 4.7.1, but the venv has 4.8.0. If the JIT fails, the fix is a side install of about 200 MB per rank into the overlay. Do not install it on spark-one or spark-two before the disk cleanup.
- **Wall time:** bench is in W1; optional serving about 50 min.

**O2 EP falsification** (`S/r4/rhys-ep/ep-smoke.env`). Not queued; the prediction is a regression. Run only on explicit request.

**O3 lossy FP8 target head** (r3 dense `=both`). Belongs to r3; not in r4.

## 5. Order, dependencies and total time

| # | tag | class | expected (1 user / c16) | readiness | depends on | wall |
|---|---|---|---|---|---|---|
| P | precondition sha re-check | — | — | ready | r3 done | 5 min |
| 0 | r4-0-baseline + matched bench | meas. | — | ready (staged) | P | 25 min |
| W1 | install all + GPU unit tests (+O1 bench) | — | — | ready | 0 | 60–75 min |
| E1 | bitstack (L2 + MAINPROJ + CERT + EAGER) | X | −1.1..−1.7 ms (+3.5–5.5%) / +0.6–1.3% | ready-for-gpu-test | W1 tests | 1.7–2.1 h |
| E2 | verify-cap + route-live | X | +0.5–2.5% / c4–8 +1–4% | ready-for-gpu-test | W1, E1 decision | 1.75–2.3 h |
| E3 | b12x-smallm | R | −0.7..−2.2 ms (+2–6%) / +1–2% | ready-for-gpu-test | W1, r3-e4 outcome | 1.4 h |
| E4 | drafter gather on RDMA (env) | X | 0 / +0.5% | needs audit | e3 K3 outcome | ≤1 h |
| E5 | prefill trio | X/X/R | TTFT ×3–10 under load; prefill +3–25% | ready-for-gpu-test | W1 | 4–4.5 h |
| O1 | b12x MoE | L | 0..+3% / ±5% | ready-for-gpu-test (bench) | W1 bench | (in W1) + 50 min optional |

- **Decode block (P → E4):** about 7–8.5 h.
- **With E5:** about 11.5–13 h.
- **If time-boxed:** run P, 0, W1, E1, E2 and E3. They are about 85% of the expected decode gain.
- **Reordering rule:** if r3-e4 did *not* promote QKV_A_SHARD, move E3 ahead of E2. b12x-smallm then has its full 7.2 ms eligible pool and the largest single expected gain.

**End of r4:**
- Run `uninstall.sh` for every non-promoted bundle (no two bundles share a file, so the order does not matter).
- Check that every site file is either at its original sha or at the promoted patched sha on 8/8.
- Re-run the matched bench (`--label post-r4`).
- Write the README "Round 4" section.

## 6. Disk proposal (summary of `S/r4/disk-inventory.md`; nothing deleted; every item needs the user's explicit approval)

**Free space, 2026-10-22 04:47 UTC:**

| host | free |
|---|---|
| spark-one | 13G (99%) |
| spark-two | 16G (99%) |
| spark-three | 132G |
| spark-four | 181G |
| spark-five | 162G |
| spark-six | 59G |
| spark-seven | 118G |
| spark-eight | 132G |

One model copy per host (476G). No duplicates.

**Never delete:**
- `~/dsv41-stage/m32/target-m32-native-v1`
- `~/rdma_ar_dsv41/build/rdma_ar_ext.so`
- `~/venvs/vllm032`
- `~/balloon.py`
- `~/dsv41-logs`
- `~/.cache/vllm`
- the dsv41-opt dirs and `~/dsv41-stage`
- cisco8101-backup/ztp (spark-seven), sparkos-prebuilt, spbm-*

**Ranked proposal:**
1. **Dangling untagged MiMo SGLang docker images** on spark-one, six, seven and eight: about 33.7G each, ~135G total. Command: `docker image prune` (dangling only). This is the safest big win; spark-one goes from 13G to about 47G. **Recommended before r4**, because nsys or extra JIT caches on spark-one are otherwise risky.
2. Empty `~/queen-park` skeletons on spark-four and spark-five.
3. `~/exo-venv-9b23.tar.zst` on spark-one through four (1.6G each).
4. MiMo/GLM leftovers: `rdma_ar..rdma_ar5` (not `rdma_ar_dsv41`), `mimo-*`, `devimg`, glm52, exl3. About 8G total together with items 2–3.
5. uv/pip/.nv cache prune (a few GB; leave `.cache/vllm`).
6. pawn-9 s5000/s7500 worker copies (about 30G).
7. **spark-two BFS project** (`~/bfs-spark` + images): about 105G, the only real relief for spark-two. User project decision.
8. spark-one user projects (ComfyUI, imprint/clipfarmer, altfreezing): up to about 128G. User decision.
9. Queen round-9 per-rank FSDP shards: about 217G. This loses the round-9 optimizer resume (the archive holds rank 0 only); round 10 does not need them.
10. spark-three Qwen3.8-27B + FP8 (81G): depends on the pending Queen teacher choice.
11–13. voiceform (leave), other user projects, ECDSA scratch.

**Notes:**
- r4 itself needs under 50 MB per rank. The exception is O1's optional cutlass-dsl side install (about 200 MB per rank).
- spark-two's docker has a broken container (rw layer snapshot missing) that needs a separate look.
- Security observation for the user: 512S2 `authorized_keys` has 7 keys, with comments that include kevinvegda@gmail.com, sufithaire@gmail.com and ultor@olympusmons. The temporary queenmove key is gone. These should be recognised by the user.
- The request's "sure" is not item-specific, so no deletion has been performed or is implied. Each item needs an explicit go.

## Critic corrections

Read-only review against the r2 profile, the r3 handoffs (e1, spec, dense, sonar) and the r4 sources. All gains are against the post-e1 1-user step of about 32 ms.

**C1. E3 b12x-smallm is overstated. Corrected: −0.6 to −1.0 ms (+2–3%) single, +0.5–0.9% at c16.**
- The "7.2 ms eligible" includes shared w1/w3 (1.29 ms). That runs on the side stream (r3 crit_dense: 128 µs vs 270 µs routed), so it saves 0 ms on the critical path.
- Critical-path eligible time is about 5.2 ms: qkv_a 2.22, q_b 1.63, o_b about 1.0, main_proj 0.39.
- These already run at 171, 160, about 135 and 203 GB/s. At 200 GB/s the saving is about 0.95 ms. The 238 GB/s roofline caps it at about 1.6 ms, so the claimed −2.2 ms is impossible.
- If r3-e4 promotes QKV_A_SHARD, about −0.35 to −0.65 ms remains.
- Fix the reorder rule's "full 7.2 ms" to read "about 5.2 ms".
- E3 does not add to L2 prefetch. Both cut DRAM stall on the same GEMMs (qkv_a is the main prefetch target), so expect less than the sum.

**C2. MAINPROJ_SPLIT: about −0.2 to −0.3 ms.**
- The gather was costed at 15–30 µs, but our collectives are about 42 µs per call.
- It is eager-only (stock during capture). Confirm "MAINPROJ_SPLIT active" appears at decode M (around 6), not only at prefill M.

**C3. E1 stack: −0.9 to −1.5 ms (+3–5%).**
- **CERT_HEAD gains 0 at temperature > 0.** It arms only on all-greedy steps, so say so wherever its number is quoted.

**C4. Measurement resolution.**
- e1 controls spread 80.0–82.3 on mean-single, and code alone spreads ±4%. With 3 pairs, effects of ≤1% cannot be resolved: E2 single, CERT, main_proj and E4.
- Attribute E1's components from in-instance metrics:
  - target-graph items: verify_ms
  - main_proj: draft_ms(1/16)
  - CERT: tiles/call logs plus nsys
- verify_ms does not see CERT or eager-glue, because both run eager.
- Judge E2 on c4–c8, prose single and the logged dead-draft %.

**C5. Exactness labels.**
- **E1 gate:** "same exact.py output as control" holds only if control matches control across restarts. Run C vs C first. If they differ, gate on self-consistency plus the GSM8K-200 miss set.
- **prefill-sched is not bit-identical under load.** Its chunk boundaries and batch composition change GEMM M. Label it X(sched) and keep the gs/lc gates. A lone prompt should be identical; check that.
- **prefill-idxshard** is X only if `lp compare` shows zero diff; otherwise it is R.
- **ROUTE_LIVE/VERIFY_CAP** are X only if T2 (`test_verify_router.py moe`) shows live rows bit-identical on the real M32+compact path. T2 becomes a hard prerequisite for E2. It is API-fragile, so budget time to fix it.
- **E4 is exact:** `max_reduce_bytes = min(max_bytes, 512 KiB)`, so only gathers move. It doubles the RDMA slot arena, so check MemAvailable on spark-one/two.

**C6. E5 gains are unsupported. Make E5 conditional.**
- No prefill profile exists. "+3–25%" combines idxshard (0 at ≤32K) with an unprofiled prefill-sp guess.
- Run the 16K-chunk nsys profile first. Skip prefill-sp if all-reduce plus mHC is under about 10% of the chunk.
- The TTFT gain (about 6–8×) is credible for prefill-sched only.

**C7. W1 logistics.**
- Run the b12x-moe bench on spark-four/three only. The optional cutlass-dsl 4.7.1 (about 200 MB) and the b12x compile cache stay off spark-one/two unless the prune is approved.
- After W1, the "all installed, flags off" state is new code. Add an inert control C0′ and compare it with the step-0 baseline (runarm_conc plus verify_ms). Later controls are C0′-type.

**C8. Prerequisites.**
- **VERIFY_CAP `auto`:** tau comes from the AV calibration (VERIFY_SCALE 0.6, N0 48, N1 56, FIXED_MS 3.4), which was tuned on pre-E1/E3 costs. If e3 promotes 32 seqs, re-check the calibration log too.
- **temp>0 VERIFY_CAP:** this needs r3-e5, and e5's dspark/speculator.py must first be rebased onto 774b34c2 (r3-e1 note).
- **Sonar fp8 o_proj:** it uses DeepGEMM `fp8_einsum`, not `mm_mxfp8`, so it has no interaction with b12x-smallm. Confirmed.

**C9. No r3 duplication.**
- WO_A fp8, the FP8 draft head and the qkv_a shard are rightly left to r3-e4.
- FC2 is rightly dropped.
- EP is rightly not queued: M32 and compact are both disabled under EP.

**Corrected order:**
1. P
2. 0 (+matched bench)
3. W1 (b12x-moe on spark-four)
4. C0′
5. E1
6. E3
7. E2 (after T2 passes and the AV calibration is re-checked)
8. E4
9. O1 (optional)
10. E5 (after the profile)

E3 now comes before E2 regardless of r3-e4.

**Expected decode total:** about −1.5 to −2.5 ms at 1 user (+5–8%) and +1.5–3% at c16, plus E2's c4–c8 +1–4%.
