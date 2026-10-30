# vLLM-Moet drafter for dsv41: offline prep and exact A/B plan (2026-10-26)

**Scope.** The user asked: "can we try their drafter" (https://github.com/kacper-daftcode/vLLM-Moet#fraser-bench-single-stream-code-decode, Apache-2.0, commit b0dc737).

**Constraints this prep ran under.** Goal window 3 (wf_6355e766-617) owns the cluster, so this prep did:

- no restart
- no GPU job
- no live-file edit
- no site-packages install

Everything is in `~/dsv41-stage/moet/` on 8/8; `SHA256SUMS` covers it. The Fraser client is in spark-one `~/dsv41-bench/fraser/`.

## TL;DR

- **Their drafter is our drafter.** It is the same DSpark drafter: the checkpoint's `mtp.*` weights and vLLM's `DSparkSpeculator`. Moet differs in three ways:
  - **Configuration.** k=7, `draft_sample_method=probabilistic`, `rejection_sample_method=block`, adaptive verification OFF. The two sampling settings are the same as ours; k and AV differ.
  - **Drafter lm_head.** Theirs runs on an MXFP8 GEMV.
  - **Three launch-fusion kernels.** These cover Markov sampling, the dflash input prep and the drafter SWA metadata.
- **We already have the byte savings.**
  - FP8 draft head (r3-e4).
  - Vocab-sharded greedy Markov (r3-e1), with greedy-sharded drafter graphs (r4-e2).
  - Their kernels would save at most 0.02-0.06 ms each on our 27-31 ms C1 step (< 0.2%), so none is staged into site-packages.
- **The thing worth testing is k.** Arms:
  - k=6 and k=7, each with OUR adaptive verification on and a fresh AV pin per k.
  - Moet's exact config as a reference arm (`k7off`).
- **k=6/7 needs no code change.**
  - `DSV41_SPEC=6|7` already flows through the launcher.
  - vLLM 0.32 sizes every graph set from k by itself, so the "192-row cap" is not a hard limit. At k=7 it captures FULL decode graphs up to 256 rows (`tools/capsizes.py`, validated against the live k=5 log).
  - The open question is memory (section 2.3).
- **Exactness.**
  - We expect `exact.py` to keep the anchor `a8c3394af9d2`. The r3-e5 precedent: a drafter and trajectory change kept a8 on every instance.
  - The G1 output digest is trajectory-sensitive. It already differs between AV profiles (FU-2 939bb576 vs pinned bc1e8386). k=6/7 changes the trajectory, so **E1 (bitwise G1 vs k=5) may fail**.
  - If it fails, the plan stops at a user decision (section 4.5). It never auto-promotes.
- **The most valuable exact item in their repo is not on the drafter side.** It is their mHC boundary overlap (Moet step 10 plus the `mhc_post_norm` kernel, +3-5% steps/s for them):
  - Our r2 nsys shows a strictly serial `mhc_fused` (20-21 us) + `pre_big` (5.8 us) at each of the 80 sublayer boundaries, about 2.1 ms/step at C1. That is an upper bound: the L2 prefetch of `mhc_fn` added since may have shortened it.
  - This matches kernel-ideas #8.
  - The sources are in `src/` for a separate window.

## 1. Moet's DSpark-related changes mapped to our stack

The C1 step is 27-31 ms (gw2: coding 31.2, prose 27.0). "Exact" means the target's argmax decides every token; changes on the draft side can only move the trajectory, and the G1-digest gate catches that.

| # | Moet change (their switch) | what it does there | ours today | exact for greedy output | expected on our step | verdict |
|---|---|---|---|---|---|---|
| 1 | step 18: drafter lm_head on the MXFP8 GEMV (`VLLM_MOET_DRAFTER_LM_HEAD_MXFP8`) | BF16 331 MB/rank to an MXFP8 copy plus their GEMV, <= 16 rows, about -106 us/step | **We have the bytes half.** `DSV41_FP8_LMHEAD=draft`: MXFP8 weights and MXFP8 activations (`DSV41_FP8_LMHEAD_ACT=a8`) through CUTLASS `mm_mxfp8`, 16160x5120 = 82.7 MB/rank | Draft-only numerics, same class as ours | Kernel swap only: <= 0.05 ms. CUTLASS runs at about 200 GB/s and our best skinny kernels at about 230 GB/s on GB10. The skinny MXFP8 and b12x small-M arms already measured 0%. | skip |
| 2 | step 20: Markov sampling in one launch per draft step (upstream gfx942 `amd/ops/fused_markov.py` Triton on CUDA, `VLLM_MOET_FUSED_MARKOV`) | Bias GEMV block + base logits + Gumbel-max in one launch. -60 us of drafter graph per step at k=7. **Falls back to vLLM's loop for greedy drafts.** | **Greedy steps:** vocab-sharded Markov (8.3 MB/rank per position instead of 66 MB) in a second graph set. **Sampled steps:** the stock full-vocab loop, 66 MB x k per rank, about 1.5 ms at k=5 and 2.1 ms at k=7. `fused_markov.py` does not exist in vLLM 0.32.0. | Draft-only (the bias's last bf16 bit) | Greedy G1/G4: 0. Sampled: -0.03 to -0.08 ms (launches only; bytes unchanged). | skip. Backlog: a vocab-sharded *probabilistic* Markov step (per-shard Gumbel-max + (max, id) gather). The block verifier still needs q for the drafted tokens and the residual. Worth about -1.2 to -1.8 ms per sampled step at k=5-7; production traffic is sampled. |
| 3 | step 30: drafter input prep for all draft KV groups in one launch (`VLLM_MOET_DFLASH_PREP_GROUPS`) | Three eager `prepare_dflash_inputs` launches become one (3 x 9.4 us on their card) | Stock three launches. r4 `dpad` already cut about 15 us/call from the padding loops. | Bitwise (same stores) | -0.01 to -0.02 ms | skip (< 0.1%). Possible member of a future glue bundle. |
| 4 | step 36: drafter SWA metadata update in one launch (`VLLM_MOET_SWA_DRAFT_BUILD`) | In the drafter graph, the 3 ops x 3 builders of `update_draft_decode_metadata` become 3 launches | Stock. Our `swam` fuses the *target's* 14 eager decode SWA builds into 1, which is more than their step 32. | Bitwise | -0.01 to -0.02 ms | skip (< 0.1%) |
| 5 | drafter mHC side stream (`patch_vllm_dspark_mhc_stream_sm120.py`) | The 3 DSpark layers use the target's `mhc_stream` | No stream at all: `supports_mhc_overlap()` is gated to capability family 100, so the target has none either | Bitwise (same kernels, other stream) | At most -0.1 to -0.15 ms (6 boundaries), and only together with #6 | with #6 |
| 6 | **(target, not drafter)** step 10: mHC boundary off the critical path (`VLLM_MOET_MHC_OVERLAP`, `mhc_overlap/mhc_post_norm_sm120.cu`) | One CUDA kernel on the path (post-mix, collapse, RMSNorm: 3.3 us vs 7.4 on a 5090). Projection and Sinkhorn move to the side stream. Bit-identical per their `test_mhc_post_norm_sm120.py`. | Not done. It is queued as kernel-ideas #8. r2 nsys: `mhc_fused` 20-21 us + `pre_big` 5.8 us per boundary at M=5-6, x80, serial. | Bitwise on sm_120 per their test. Must be re-proven on sm_121 with our vLLM 0.32 TileLang kernels and our rdma_ar/L2-prefetch side streams. At TP8 `fuse_mhc_all_reduce` stays False. | **-0.8 to -1.6 ms at C1 (+3 to +5%)**, more at C4 | **Recommended next exact lever** (its own window). Kernel and patcher in `src/`. |
| 7 | served config: k=7, probabilistic drafts, block verify, AV off | Wider draft. Their indexer path cannot run AV. | k=5, probabilistic + block with our lossless FRESH_U and FRESH_NOISE, AV on + verify-cap auto + greedy-sharded graphs | A k change is a trajectory change (gates E0 and E1, section 4.5) | Theirs: steps/s -12%, tokens/step 4.54 to 5.38 (+18.5%), tok/s +4% (Fraser code) | **A/B: k6, k7 with AV on, plus k7off** |
| 8 | drafter MoE on b12x (`"moe_backend":"b12x"`) | Their MoE backend | b12x measured 0% at small M on GB10, and `DSV41_ROUTE_LIVE` must never run with b12x | n/a | n/a | skip |
| 9 | Fraser bench (`tools/sm120_perf/fraser_bench.py` + `fraser_prompt.txt`) | single-prompt code decode, median | Ported to `~/dsv41-bench/fraser/` (section 3) | n/a | n/a | done |

Other Moet items have equivalents or did not help on GB10, and are not drafter-related:

- Dense GEMV.
- P2P all-reduce: ours is rdma_ar.
- Deterministic top-k: ours is TOPK_CANON.
- Fused MoE.

## 2. k=6 / k=7 readiness

### 2.1 Graph and AV shapes

These come from `tools/capsizes.py`, a pure model of `VllmConfig._set_cudagraph_sizes`, `CudaGraphManager._init_candidates` and `dsv41_rel` tails.

The model was checked against the live k=5 instance (rank0.log 10-26 17:45) and agreed on every count:

- PIECEWISE 45, FULL 29, dspark 22 x 2
- the 8 tails
- 185 profile runs
- pin shape_key == `av-pin-pdfinal.json`

| arm | rows/req | max decode rows | max capture | PIECEWISE | target FULL | dspark (x2 sets) | AV tails | AV profile runs | AV draft sizes |
|---|---|---|---|---|---|---|---|---|---|
| k5 (live) | 6 | 192 | 384 | 45 | 29 (to 192) | 22 (to 160 tok) | 288..16384 (8) | 185 | 1,2,4,6,8,12,16,24,32 |
| k6 | 7 | 224 | 448 | 50 | 34 (to 224) | 26 (to 192) | 336..16384 (8) | 210 | 1,2,4,7,8,14,16,24,28,32 |
| k7 | 8 | 256 | 512 | 51 | 35 (to 256) | 29 (to 224) | 384..16384 (7) | 210 | 1,2,4,8,16,24,32 |
| k7off (Moet) | 8 | 256 | 512 | 51 | 32 uniform (to 256) | 29 | no AV | - | - |
| k7 @ max-num-seqs 24 (fallback) | 8 | 192 | 384 | 43 | 27 (to 192) | 22 | 288..16384 | 175 | 1,2,4,8,16,24 |

Nothing has to be extended. `max_cudagraph_capture_size = min(seqs*(1+k)*2, 512)`, and the varlen FULL decode graphs (AV on) cover every capture size up to `seqs*(1+k)`.

### 2.2 Code paths checked (live site files, read-only)

- **`dspark/speculator.py`** is generic in k:
  - The sharded Markov runs k positions, each a local 16160x256 GEMV plus a 16 B/row rdma all-gather.
  - The greedy-sharded second graph set is generic.
  - `DSV41_DRAFT_FRESH_NOISE` has `assert n_spec <= 8` (key = 2^29 + 8*anchor + step), so **k <= 8**.
- **`rejection_sampler_utils.py` FRESH_U** is keyed by (seed, pos, start_pos): generic. Greedy rows are untouched.
- **`adaptive_verification.py`:**
  - The verify-cap kernel uses BLOCK_S = next_pow2(k) = 8 and BLOCK_J = next_pow2(k+1) = 8 for k = 5..7. k=8 would give a new BLOCK_J=16 variant.
  - The calibration knobs (SCALE 0.6, N0 96 / N1 104, FIXED_MS 3.4) are per row or per width. Keep them. FIXED_MS covers outside-graph glue, which does not grow with k.
  - With AV off the manager is None, so the verify-cap, the AV profile and the pin logic are all skipped (`model_runner.py` 1138-1160).
- **Dense regimes for a single stream:**
  - The verify M is at most k+1, which is 8 at k=7.
  - Every single-stream verify step therefore stays in the same `DSV41_QKV_A_SHARD` (M <= 8) and L2-plan "collectives=125" regime as k=5.
  - k=8 (M=9) would cross into the replicated qkv_a GEMM, which is another exactness variable. **k=7 is the widest safe width.**
  - CERT head (M 2-96), SYNC_FREE, sonar o_proj (<= 32), M32 MoE (<= 96): unchanged at 1-4 sessions.
  - At k=7 with 26 or more sequences (>= 208 rows), the all-reduce exceeds the 2 MiB rdma_ar slot and goes to PyNccl inside the graph. That is the same as today's 288/384-token piecewise graphs.
- **DSpark config:**
  - k != `dspark_block_size` (5) is accepted for DeepSeek DSpark (the equality check is Qwen3-Omni-only), so the drafter extrapolates past its trained block. Moet's AL went from 4.54 to 5.38 at k=7.
  - `num_speculative_tokens_per_batch_size` (dynamic k) exists only in the capture code. The v2 GPU runner and DSpark do not implement it, so AV stays the per-step width control.
- **JIT:**
  - New Triton variants (constexpr k) compile on the first k=6/7 launch; the profile arms absorb that.
  - DeepGEMM keys do not include M (M32 contiguous layout), and the indexer decode flattens next_n.
  - The gate is "no NEW JIT names after health vs the k5 control" (`logfacts.py jit_after_health`). The live k5 already shows 7 known first-use names.
  - Snapshot file counts of `~/dsv41-stage/g1/m32-bw/jit` and `~/dsv41-stage/g2/fc2c-on-m32all/jit-cache` (serving dependencies) before and after. New entries are additive; record them.

### 2.3 Memory (1M MemAvailable floor 6 GB)

**Estimate at k=7: +0.4 to 0.8 GB/rank**, from:

- FULL graphs to 256 rows instead of 192
- PIECEWISE to 512 tokens instead of 384
- draft-logits and rejection buffers of 32x7x129280
- 7 more drafter graphs per set

Today's 1M minimum is about 7.8 GB on spark-one (gw2), so k=7 would leave about 7.0-7.4 GB. **This must be measured.**

- Every instance records `memory.after_startup_cache_trim` and `graph_gib` (live k5: 16.7 GB / 5.90 GiB).
- A promotion candidate also runs the 1M probe with memwatch on 8/8 (fin.sh prod-mode style).
- **Fallback:** `DSV41_MAX_NUM_SEQS=24` at k=7 restores the k5 ceilings (192 rows / 384 tokens). The cost is losing 25-32-way batching, which is a production trade-off for the user.

### 2.4 A fresh AV pin per k

A k5 pin can never serve k6/k7: `check_pin` reports a shape_key mismatch, the launch profiles automatically, and the reverse holds too.

1. **Profile launch** (`ab/moetab.sh <label> k<K>p`):
   - Flags: `DSV41_SPEC=K DSV41_AV_PROFILE=force DSV41_AV_COST_JSON= DSV41_CAPTURE_WATCHDOG_S=300` and supervisor `--stall 600 --budget 1800` (margin for first-launch JIT).
   - Then gstart (exact.py first) and `logfacts.py --expect-k K` (graph counts vs capsizes).
   - Then `tools/mkpin_k.py make --expect-k K`:
     - It parses the raw "Adaptive verification cost profile:" line.
     - It adds the shape_key and cross-checks the profiled sizes against capsizes.
     - It validates with a verbatim copy of `dsv41_rel.check_pin` (bodies AST-identical; `_tail_sizes` differs only by a type annotation).
     - It writes the immutable `~/dsv41-stage/moet/pins/av-pin-k<K>-<label>.json` on spark-one; only rank 0 reads pins.
   - The pin is recorded as `PIN_K<K>=` in the Mac `ab/pins.env`.
2. **Measured launches** (`moetab.sh <label> k<K>`):
   - `DSV41_AV_COST_JSON=$PIN_K<K>`; the launcher default is `DSV41_AV_PROFILE=pinned`.
   - `mkpin_k.py check` requires the log line "pin hit" with that file, else the run aborts.
3. **Profile all three k (k5p, k6p, k7p) in the same window on the same stack.** The k5 control then uses a same-day pin rather than an older one.

## 3. Fraser bench (spark-one `~/dsv41-bench/fraser/`)

The prompt is verbatim (sha256 5a6b7bc1...). The settings follow their method:

- temperature 0, 2048 tokens, thinking OFF, one stream, a fresh prefill per rep
- the median of `completion_tokens/(last-first)`
- AL and steps/s from `/metrics` deltas, plus pooled per-position acceptance

Our defaults:

- Fixed per-rep nonces, plus a random `cache_salt` per request, so outputs are bitwise comparable across arms.
- One warm-up rep and 10 reps.
- A busy check before each rep.

`--nonce-mode random` reproduces their exact behaviour. The client was tested against `tests/mock_vllm.py` only. Details are in that directory's README.md.

## 4. Exact A/B plan (next window, after goal window 3 has finished)

### 4.1 Preconditions

- The cluster is free: goal window 3 is done, and the user has paused the Codex app (its read-only rounds added about 0.4% C1 noise before).
- Telemetry is off.
- SM clock is 2000 on 8/8. Never raise it; `clk8.sh` runs before every phase.
- Fix the launcher once: `export EXP=$(ssh spark-one 'sha256sum ~/dsv41-tp8.sh | cut -c1-8')` (gw2 left 592793fd), so every arm runs the same launcher.
- Record the live default config and the pin goal window 3 leaves, for `restore`.

### 4.2 Arms

Every arm runs the live stack with clock 2000:

| arm | how | AV |
|---|---|---|
| **k5** (control) | `DSV41_SPEC=5` + pin k5 | on |
| **k6** | `DSV41_SPEC=6` + pin k6 | on |
| **k7** | `DSV41_SPEC=7` + pin k7 | on |
| **k7off** (Moet reference) | `DSV41_SPEC_JSON={"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}` | off, so no pin and no verify-cap |

### 4.3 Order and time (about 3.5-4 h of GPU)

1. **P (pins, about 25 min):** `k5p`, `k6p`, `k7p`.
2. **M (measure, about 2.5 h):** `k5 k7 k6 k7off | k7off k6 k7 k5`. That is ABBA; each instance takes about 5 min to healthy plus about 12 min of core suite.
3. **F (only if a candidate passes section 4.5):** 1 full-suite instance of it (conc8/16/32, GSM8K-50, longctx), plus a cold 1M probe with memwatch on 8/8.
4. **R:** `moetab.sh <label> restore`. This relaunches the live default and checks the exact.py anchor.

Driver commands, from the Mac:

    export EXP=<sha8>          # AB defaults to the directory of moetab.sh; ops/ must sit next to ab/
    bash ab/moetab.sh moet-p5 k5p
    ... moet-p6 k6p ; moet-p7 k7p
    bash ab/moetab.sh moet-c1 k5 ; moet-a7 k7 ; moet-a6 k6 ; moet-o7 k7off ; moet-o7b k7off ; moet-a6b k6 ; moet-a7b k7 ; moet-c2 k5
    bash ab/moetab.sh moet-R restore
    python3 -I ab/moetsumm.py ab/runs > ab/summary.md

### 4.4 Per-instance suite (`moetab.sh` core = `ops/moetphase.sh` phases)

1. **gstart:** exact.py FIRST with spec-counter deltas, then `logfacts --expect-k`, then the pin check.
2. matched C1 coding/prose x5. Use ONE fixed label `pdab` in every arm: the label is written into the prompts.
3. G1 (`goal_single.py --repeats 3`).
4. `g1text` (G1 texts, for a divergence diff).
5. G4 (`goal_conc4.py`).
6. **Fraser x10.**
7. `t1` sampled single. This is production-relevant: k widens the full-vocab sampled Markov loop.
8. exact.py #2.
9. facts.

Steps 2-7 each have a spec-counter snapshot before and after, with per-position acceptance. `moetsumm.py` tabulates every instance, the arm means, the deltas against k5, and the verdicts.

### 4.5 Gates and pre-registered decision rules

- **E0 (hard, every arm):** exact.py = `a8c3394af9d2` at the start AND the end of every instance. Also:
  - 0 ERROR/Traceback
  - no "first call during capture"
  - graph counts == capsizes for k
  - no NEW JIT names after health vs k5
  - pin hit (k5/6/7)
- **E1 (bitwise, the standing goal):** the G1 digest AND the Fraser digest are identical to the k5 control's. The k5 control must reproduce its own digests on both of its instances.
  - Where a digest differs, run `tools/g1text.py diff` on the two instances' `g1text.json` and report: type, character position, shared context, both continuations.
  - If E0 passes and E1 fails, the arm is **greedy-equivalent but not bitwise**, the same class as r4-e4's verify-cap. That is a USER DECISION, made with the measured gains. **No promotion without it.**
- **Performance (pinned, both ABBA pairs agree):**
  - G1 mean at least +1.5% vs k5 AND G4 prose-min not below -0.5%; OR G4 prose-min at least +1.5% AND G1 not below -0.5%.
  - Also t1 >= -2%, and in the full-suite instance conc32 >= -3%.
  - GSM8K-50 >= 49/50, longctx PASS, 1M MemAvailable >= 6 GB on 8/8.
  - Fraser and matched C1 are reported but are not gates.
- **k7off** is a reference only. AV off lost 11-22% at conc4/8 in round 1, so it is not promoted unless the user explicitly asks.
- **Promotion** would be one launcher line, `SPEC=${DSV41_SPEC:-K}`, plus making that k's pin the default. Both are user decisions. **Rollback** = `DSV41_SPEC=5` or the launcher backup.

### 4.6 Predictions (written before any measurement)

Per-step model:

- C1 step ≈ 27-31 ms + 0.6-1.07 ms per extra verified row (A2 regression).
- k=7 adds about +0.3-0.5 ms of drafter time on every step: 2 sharded Markov positions plus 2 more drafter query tokens per request.
- Extra tokens come only from steps that accept all 5 drafts. exact.py's per-position counts give 68/471 = 14% of steps; conditional acceptance at depth is 0.6-0.75.

| metric | k7 vs k5 | k6 vs k5 | k7off vs k5 |
|---|---|---|---|
| G1 | **-1 to +4%** (math and code up, prose flat or down) | about half of k7 | code/math about k7, prose and G4 worse |
| G4 prose-min | **-1 to -3%** (drafter cost on every step, 4 x 7 query tokens) | -0.5 to -1.5% | -10 to -20% |
| matched C1 coding (AL 5.22 of 6, saturated) | +6 to +12% tok/s | +3 to +6% | similar to k7 |
| Fraser (code, thinking off) | +3 to +10% | +2 to +5% | similar to k7 |
| t1 sampled | -1 to -3% (full-vocab Markov grows 1.5 to 2.1 ms/step) | -0.5 to -1.5% | n/a |

Our reading: on its own, k is unlikely to close the G1 gap (+19.5% needed) or the G4 gap (+4-8% needed). It may still be worth a few percent on G1 or on code-heavy single-stream traffic.

## 5. Files (`~/dsv41-stage/moet/`, identical on 8/8)

| path | what |
|---|---|
| `tools/capsizes.py` | graph and AV shape model per (k, max_num_seqs); `--json` |
| `tools/mkpin_k.py` | `make` / `check` / `show` AV pins with shape_key for any k (stdlib) |
| `tools/logfacts.py` | startup facts of a launch: k/AV, graph counts vs model, pin hit/miss, capture s / GiB, memory per stage, JIT names after health, errors |
| `tools/g1text.py` | `save` G1 texts (same requests as goal_single), `diff` two instances |
| `ops/moetphase.sh` | spark-one phase runner: delegates to gw2 `finphase.sh` with spec snapshots; adds `fraser`, `t1`, `g1text`, `facts` |
| `ab/moetab.sh`, `ab/clk8.sh`, `ab/moetsumm.py` | Mac driver (bash 3.2): supervised relaunch, gstart, pins, suite with clock checks, fetch, summary |
| `fraser/` | copy of the Fraser client (live copy: spark-one `~/dsv41-bench/fraser/`) |
| `tests/mock_vllm.py` | offline SSE + /metrics mock used to test fraser_bench.py |
| `src/vllm-moet-b0dc737/` | read-only reference: their DSpark patchers and tests, sm120_gemv, mhc_overlap and mhc_pre_norm kernels and patcher, the Fraser bench, LICENSE, NOTICE |
| `pins/` | empty; profile arms write `av-pin-k<K>-<label>.json` here on spark-one |

Nothing here is imported by the live server, and no launcher or site file references it. Removing the directory has no effect on serving.
