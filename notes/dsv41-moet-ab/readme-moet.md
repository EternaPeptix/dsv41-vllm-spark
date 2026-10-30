
## Moet drafter A/B (2026-10-26 23:57 - 2026-10-27 02:22 UTC): DSpark k=6 / k=7 (vLLM-Moet's width) against k=5, plus Moet's AV-off config. All exact (exact.py a8 on every instance) but NOT promoted: k=7 loses G1 and G4, k=6 is flat on G1 and loses on the G4 prose session; live config unchanged (k=5, production)

Scope: the user asked "can we try their drafter" (https://github.com/kacper-daftcode/vLLM-Moet#fraser-bench-single-stream-code-decode, Apache-2.0, commit b0dc737). Moet's drafter is **our drafter**: the same DSpark drafter (the checkpoint's `mtp.*` weights, vLLM's `DSparkSpeculator`). What they change is the configuration (k = `num_speculative_tokens` 7 instead of 5, adaptive verification OFF), an MXFP8 GEMV for the drafter lm_head and three launch-fusion kernels. The offline prep (`~/dsv41-stage/moet/README.md`, on 8/8) mapped every Moet change to our stack and found the kernels worth <= 0.2% each on our 27-31 ms single-user step (we already have the byte savings: FP8 draft head, vocab-sharded greedy Markov), so this window measured **the draft width k** on the live stack (launcher b0cf94d0, 2000 MHz, telemetry off). Mac drivers and raw results: scratchpad `mab/` (`scripts/mab.sh` one instance, `mabblk.sh` serial block, `mabphase.sh` per-phase runner on spark-one, `clk8.sh`, `mabline.py`, `mabagg.py`, `pincmp.py`; `ab.log`, `ab.jsonl`, `tables.md`, `ab/<label>/`, `runs/<label>/`). On spark-one: `~/dsv41-moet-ab/runs/`.

### 1. Method
- **Arms** (every launch: `--expect-launcher b0cf94d0`, `DSV41_GPU_CLOCK_MHZ=2000`, the promoted window-3 stack unchanged; only these variables differ):
  - **k5** (control): `DSV41_SPEC=5 DSV41_AV_COST_JSON=~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json`.
  - **k6**: `DSV41_SPEC=6 DSV41_AV_COST_JSON=~/dsv41-stage/moet/pins/av-pin-k6-mab-p6.json`.
  - **k7**: `DSV41_SPEC=7 DSV41_AV_COST_JSON=~/dsv41-stage/moet/pins/av-pin-k7-mab-p7.json`.
  - **k7off** (Moet's served config, reference only): `DSV41_SPEC_JSON={"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}` (their `moe_backend: b12x` left out: 0% at small M on GB10 and not allowed with `DSV41_ROUTE_LIVE`). AV off means no pin, no verify-cap and no acceptance estimator.
- **One AV pin per k, each made from that k's own startup profile on today's stack** (`moet/tools/mkpin_k.py make`, validated with a verbatim copy of `dsv41_rel.check_pin`; files on 8/8 in `~/dsv41-stage/moet/pins/`, the source logs in `pins/src/` and `runs/mab-p{6,7}-start/`):
  - k5: the rank-0 startup profile of **g3-FU** (the production launch on b0cf94d0 that goal window 3 left live, unpinned, healthy 23:22:34 UTC) -> `av-pin-k5-g3FU.json` **eeac4f97**. Not `av-pin-g3.json`: that one was profiled on 592793fd, before dhtac, and prices a 4-request draft at 7.36 ms against 6.49 ms on today's stack, while the k6/k7 profiles were measured with dhtac; a stale k5 pin would have biased the 4-session comparison.
  - k6 / k7: dedicated profile launches **mab-p6 / mab-p7** (`DSV41_AV_PROFILE=force`, no pin, `DSV41_CAPTURE_WATCHDOG_S=300`, supervisor `--stall 600 --budget 1800` for first-launch JIT; healthy +426 / +420 s) -> `av-pin-k6-mab-p6.json` **b166aacd** (34 capture sizes to 224, 10 draft sizes) and `av-pin-k7-mab-p7.json` **5886e752** (35 capture sizes to 256, 7 draft sizes).
  - Every measured launch logged "pin hit" and passed `mkpin_k.py check`.
- **Order:** profile launches p6, p7; then **C1 A7 A6 C2 B6 B7 O7 C3** (C = k5, A/B = k7 / k6, O = k7off). Pairs: each candidate against the mean of its bracketing controls (A: C1+C2, B and O: C2+C3).
- **Per instance** (`mab.sh`): supervised launch -> SM clock on 8/8 -> `gstart.sh` (exact.py FIRST, spec-counter deltas) -> `logfacts.py --expect-k` (graph counts against `capsizes.py`, memory, JIT) -> pin check -> suite phase by phase, telemetry OFF, one `nvidia-smi` read on 8/8 before every measurement phase, 2 s `/metrics` sampler for foreign requests: matched C1 coding / prose x5 (label pdab), G1 x2 (`goal_single.py`), G1 texts (`g1text.py`), G4 warm-up + 6 runs (`goal_conc4.py`), **Fraser bench** x10 + 1 warm-up (Moet's prompt and method, `~/dsv41-bench/fraser/`), conc8, conc16, conc8, conc16, t1 sampled (temp 1.0 / top-p 0.95, seed 20261027, 2 reps x 3 types x 512 tokens), exact.py #2, facts. Each measurement phase is bracketed by spec-counter snapshots (accepted tokens per draft position).
- **Clocks:** one `nvidia-smi` read on 8/8 before every measurement phase and after every launch: 101 good checks at **1969-1989 MHz**, event reasons 0x0, at most 64 C (one check got no reading from one node, see the incident below). No clock or power setting was changed.
- **Foreign traffic:** none in any timed phase (sampler maximum = the phase's own concurrency; Fraser's own busy check saw 0 before every rep).
- **Incident:** the first C2 attempt aborted before its G4 phase at 01:03 UTC because the clock check got no reading from spark-five (a transient tailnet ssh failure; spark-five answered 1976 MHz seconds later). `clk8.sh` now re-reads a node that returns nothing, twice, before it counts as BAD. The partial instance is kept as `mab-C2x` (its G1 101.86 with digest 9a18f802d930f6e7 and C1 30.27 / 26.29 ms match C1/C2) and C2 was relaunched from scratch.

### 2. Drafter cost and startup facts per k
| | k5 | k6 | k7 | k7off |
|---|---|---|---|---|
| drafter ms per step at 1 / 2 / 4 requests (startup AV profile, raw `draft_ms`) | 4.28 / 4.74 / 6.49 | 4.73 / 5.34 / 7.20 | 5.14 / 6.13 / 7.86 | - (no AV) |
| AV profile verify ms, 1 request x (k+1) rows | 30.23 (6 rows) | 31.68 (7) | 33.58 (8) | - |
| graphs PIECEWISE / FULL / DSpark (x2 sets) / decoder replay (all = `capsizes.py`) | 45 / 29 / 22 / 3 | 50 / 34 / 26 / 4 | 51 / 35 / 29 / 4 | 51 / 32 / 29 / 4 |
| pinned launch to healthy | 276-284 s | 293 s | 284-289 s | 320 s |
| rank-0 MemAvailable after the startup trim | 16.14-16.52 GB | 15.84-15.86 GB | 15.13-15.29 GB | 15.47 GB |
| MemAvailable at the end of the suite (min over 8 nodes) | 14.4 GB | 13.7 GB | 13.2-13.3 GB | 13.4 GB |

- One more draft position costs **+0.45 ms per single-user step** (k6) and two cost **+0.86 ms** (k7); at 4 requests +0.71 / +1.37 ms. Each position is one more drafter query token per request plus one more sequential Markov step. The k7 profile also shows 25.4 ms at 32 requests (k5 15.6 ms).
- Memory: k6 costs about 0.5 GB and k7 about 1.1 GB per node against k5. The 1M probe was not run because no arm qualified; scaled from today's 1M minimum (8.3 GB, g3-FU), k7 would be near 7.2 GB, above the 6 GB floor (estimate only).

### 3. Exactness
- **E0: exact.py a8c3394af9d2 on all 12 launches**: the 2 profile launches, the 9 measured launches (including the aborted C2x; the 8 complete instances again after their suite) and the restored production instance. Counters (accepted / drafts / draft tokens / generated; accepted per position):
  - k5 under its pin: 909 / 467 / 2335 / 1373; 348/233/171/90/67, identical to g3-FU (the pin reproduces its source launch);
  - k6: 925 / 453 / 2718 / 1373; 339/226/166/82/63/49 (profile and pinned launches identical);
  - k7: 922 / 456 / 3192 / 1373; 340/224/169/67/55/38/29 (same);
  - k7off: 979 / 401 / 2807 / 1373; 294/213/160/118/89/64/41.
  - Also on every launch: 0 ERROR/Traceback on 8/8, graph counts equal to `capsizes.py`, no "first call during capture", and no new JIT names after health (k6/k7 show the same first-use Triton names as k5).
- **G1 outputs.** Each arm reproduced its own digest on every instance with identical repetitions: k5 **9a18f802d930f6e7** on all 3 (and the partial C2x), k6 and k7 **69ad872124100141** on both, k7off the same. **Code and math are bitwise identical in all four arms** (596b478a / 92e5d8cf). **Prose differs at one near tie**: character 3885 of 4927 in the reasoning, "... Differences: print" followed by " capital-intensive" (k5) or "ing capital-intensive" (k6 / k7 / k7off).
  - The k6/k7 prose (e8a524f3) is the k5 prose under `av-pin-g3.json`, so their digest 69ad872124100141 equals goal window 3's pinned A/B reference. The k5 control here (pin from g3-FU) gives g3-FU's prose (923a46be). k5 itself flips this tie between AV profiles; the k change moves it the same way an AV profile does.
- **E1 verdict.** Against this window's k5 control, k6/k7 are **greedy-equivalent but not bitwise**: the same class as r4-e4's verify-cap and every AV-profile change. Against window 3's reference digest they would pass. Either way that is a user decision, and the measured performance makes it moot (section 6).
- **Fraser outputs are not a usable bitwise gate in this suite order.** Rep 5 of 10 flipped between k5 instances under the same pin (C1 19d96c52 against C2 and C3 458268ad); the other 9 reps are identical on every instance of every arm, and all k6/k7/k7off instances give 458268ad. Most likely cause: Fraser runs after the batched G4 phase, and AV's online acceptance estimator (global state, refit from the traffic served) then starts from a timing-dependent state. Only phases that run before any batched phase (exact.py, matched C1, G1) are bitwise-reproducible gates.

### 4. Results (8 pinned instances, 2000 MHz, telemetry off, no foreign traffic)
| # | instance | arm | exact.py start / #2 | G1 digest | G1 (code / prose / math) | G4 sess. mean / prose MIN | G4 agg | matched C1 coding / prose tok/s (ms/step) | Fraser tok/s (AL, steps/s) | conc8 / conc16 | t1 sampled | MemAvail end GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mab-C1 | k5 | a8 / a8 | 9a18f802d930f6e7 | 101.66 (95.95 / 76.60 / 132.43) | 60.85 / 48.59 | 182.9 | 171.18 / 103.33 (30.33 / 26.36) | 149.72 (4.50, 33.37) | 260.8 / 398.9 | 98.99 | 14.4 |
| 2 | mab-A7 | k7 | a8 / a8 | 69ad872124100141 | 98.94 (93.58 / 74.75 / 128.48) | 61.53 / 47.74 | 180.7 | 188.27 / 102.62 (32.43 / 26.40) | 156.09 (4.97, 31.55) | 261.2 / 392.8 | 97.09 | 13.3 |
| 3 | mab-A6 | k6 | a8 / a8 | 69ad872124100141 | 101.77 (94.62 / 76.42 / 134.27) | 62.53 / 47.54 | 182.3 | 187.84 / 104.12 (31.55 / 26.49) | 156.66 (4.86, 32.28) | 263.3 / 400.9 | 99.63 | 13.7 |
| 4 | mab-C2 | k5 | a8 / a8 | 9a18f802d930f6e7 | 101.81 (96.13 / 76.67 / 132.62) | 61.67 / 48.24 | 182.4 | 171.32 / 103.63 (30.30 / 26.28) | 150.07 (4.50, 33.42) | 265.3 / 400.4 | 99.25 | 14.4 |
| 5 | mab-B6 | k6 | a8 / a8 | 69ad872124100141 | 101.56 (94.55 / 76.00 / 134.13) | 62.11 / 48.25 | 180.4 | 187.50 / 103.89 (31.61 / 26.55) | 156.53 (4.86, 32.26) | 258.5 / 393.7 | 99.48 | 13.7 |
| 6 | mab-B7 | k7 | a8 / a8 | 69ad872124100141 | 98.93 (93.65 / 74.69 / 128.44) | 60.52 / 47.63 | 178.4 | 188.22 / 102.48 (32.44 / 26.44) | 155.85 (4.97, 31.54) | 255.7 / 396.3 | 97.06 | 13.2 |
| 7 | mab-O7 | k7off | a8 / a8 | 69ad872124100141 | 93.65 (87.51 / 64.48 / 128.97) | 51.09 / 40.27 | 153.2 | 195.63 / 98.53 (33.04 / 32.39) | 157.39 (5.35, 29.50) | 216.9 / 217.7 | 92.58 | 13.4 |
| 8 | mab-C3 | k5 | a8 / a8 | 9a18f802d930f6e7 | 101.81 (96.07 / 76.62 / 132.73) | 61.79 / 48.56 | 182.3 | 171.55 / 103.65 (30.26 / 26.28) | 149.90 (4.50, 33.40) | 258.8 / 396.4 | 99.16 | 14.4 |

**Per arm (means):**

| arm | n | G1 (code / prose / math) | G4 sess. mean / prose MIN | G4 agg | conc8 / conc16 | Fraser tok/s (steps/s) | t1 | matched C1 coding / prose tok/s (ms/step) |
|---|---|---|---|---|---|---|---|---|
| k5 | 3 | **101.76** (96.05 / 76.63 / 132.60) | 61.43 / **48.47** | 182.6 | 261.6 / 398.6 | 149.90 (33.40) | 99.13 | 171.35 / 103.54 (30.30 / 26.31) |
| k6 | 2 | **101.67** (94.59 / 76.21 / 134.20) | 62.32 / **47.89** | 181.3 | 260.9 / 397.3 | 156.60 (32.27) | 99.56 | 187.67 / 104.00 (31.58 / 26.52) |
| k7 | 2 | **98.93** (93.61 / 74.72 / 128.46) | 61.03 / **47.69** | 179.5 | 258.5 / 394.6 | 155.97 (31.54) | 97.08 | 188.25 / 102.55 (32.44 / 26.42) |
| k7off | 1 | **93.65** (87.51 / 64.48 / 128.97) | 51.09 / **40.27** | 153.2 | 216.9 / 217.7 | 157.39 (29.50) | 92.58 | 195.63 / 98.53 (33.04 / 32.39) |

Control spread (k5, n=3): G1 101.66-101.81 (sd 0.09%), G4 prose MIN 48.24-48.59 (sd 0.4%), Fraser 149.72-150.07, t1 98.99-99.25.

**Pinned pairs** (each candidate against the mean of its bracketing controls; + = better, %):

| inst | controls | G1 | G1 code / prose / math | G4 prose MIN | G4 sess. mean | G4 agg | conc8 | conc16 | Fraser | t1 | C1 coding tok/s | C1 prose tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A7 (k7) | C1+C2 | **-2.75** | -2.56 / -2.46 / -3.06 | **-1.41** | +0.45 | -1.09 | -0.70 | -1.73 | +4.13 | -2.05 | +9.94 | -0.83 |
| B7 (k7) | C2+C3 | **-2.83** | -2.55 / -2.55 / -3.19 | **-1.59** | -1.95 | -2.17 | -2.42 | -0.52 | +3.91 | -2.16 | +9.79 | -1.12 |
| A6 (k6) | C1+C2 | **+0.04** | -1.48 / -0.28 / +1.31 | **-1.81** | +2.07 | -0.21 | +0.08 | +0.30 | +4.51 | +0.51 | +9.69 | +0.62 |
| B6 (k6) | C2+C3 | **-0.24** | -1.61 / -0.84 / +1.10 | **-0.32** | +0.61 | -1.11 | -1.37 | -1.18 | +4.37 | +0.28 | +9.37 | +0.24 |
| O7 (k7off) | C2+C3 | **-8.01** | -8.94 / -15.87 / -2.79 | **-16.81** | -17.24 | -16.01 | -17.22 | -45.36 | +4.94 | -6.68 | +14.11 | -4.93 |

**Acceptance (AL = 1 + accepted / drafts) per workload** (deterministic phases are identical across an arm's instances):

| arm | G1 code / prose / math | matched C1 coding / prose | Fraser (median) | G4 batch | conc8 / conc16 | t1 sampled | G1 accepted per position (share of steps) |
|---|---|---|---|---|---|---|---|
| k5 | 2.590 / 1.847 / 3.594 | 5.191 / 2.725 | 4.50 | 2.50-2.57 | 2.29-2.36 / 2.42 | 2.608 | .695 / .402 / .255 / .079 / .058 |
| k6 | 2.564 / 1.852 / 3.701 | 5.926 / 2.759 | 4.86 | 2.61-2.62 | 2.33-2.34 / 2.44-2.48 | 2.629 | .702 / .396 / .235 / .078 / .056 / .033 |
| k7 | 2.538 / 1.807 / 3.561 | 6.107 / 2.709 | 4.97 | 2.52-2.55 | 2.30-2.35 / 2.45-2.47 | 2.607 | .686 / .379 / .228 / .064 / .045 / .027 / .015 |
| k7off | 2.834 / 2.064 / 3.958 | 6.463 / 3.191 | 5.35 | 2.92 | 2.72 / 2.85 | 2.922 | .689 / .434 / .263 / .172 / .104 / .062 / .030 |

- With AV on, positions 6 and 7 are accepted on only 3.3% (k6) and 2.7% / 1.5% (k7) of G1 steps, while every step pays for drafting them. At k7 the acceptance of positions 1-5 also falls: G1 AL code 2.590 -> 2.538 and math 3.594 -> 3.561 **with bitwise-identical outputs**, so k7 needs more steps for the same text. The DSpark drafter was trained with block size 5; a 7-token block changes what its early positions predict (Fraser position 1-5 rates drop the same way).
- Matched C1 coding is near-deterministic code continuation (AL 5.19 of 6 at k5), the one place a wider draft pays: +9.4..9.9% tok/s.

### 5. Fraser bench against vLLM-Moet (single-stream code, thinking off, their prompt and method, median of 10)
| | k5 (AV on) | k6 | k7 | k7off (Moet's config) | Moet, 4x RTX PRO 6000 TP4 |
|---|---|---|---|---|---|
| median tok/s (n / (last - first)) | 149.9 | 156.6 | 156.0 | 157.4 | 680 (k=7) |
| AL | 4.50 | 4.86 | 4.97 | 5.35 | 4.54 (k=5), 5.34-5.38 (k=7) |
| steps/s | 33.40 | 32.27 | 31.54 | 29.50 | 127.8 (k=7) |
| tok/s vs k5 | - | +4.4..4.5% | +3.9..4.1% | +4.9% | +4% (k=7 against k=5: tokens/step +18.5%, steps/s -12%) |

Our k7off reproduces their relative numbers almost exactly (AL +18.8%, steps/s -11.7%, tok/s +4.9%): the same drafter with the same acceptance on the same prompt. The 4.5x absolute gap is the hardware (GB10 x8 at TP8 over CX7: about 30 ms per step against about 8 ms).

### 6. Decisions (rule for this window: exact, and G1 mean or G4 prose MIN >= +0.5% in >= 2 pinned pairs without lowering the other beyond noise)
- **k7: NOT promoted.** G1 **-2.75 / -2.83%** (every prompt -2.5..-3.2%), G4 prose MIN **-1.41 / -1.59%**, conc8 -0.7 / -2.4%, conc16 -1.7 / -0.5%, t1 sampled -2.1 / -2.2%. It wins only on high-acceptance code without thinking: Fraser +3.9..4.1%, matched C1 coding +9.8..9.9%. Cause: +0.86 ms of drafting on every single-user step, little acceptance at positions 6-7, and lower acceptance at positions 1-5 (section 4).
- **k6: NOT promoted.** G1 **+0.04 / -0.24%** (code -1.5 / -1.6%, math +1.3 / +1.1%, prose -0.3 / -0.8%), G4 prose MIN **-1.81 / -0.32%**, conc8 +0.1 / -1.4%, conc16 +0.3 / -1.2%, t1 +0.5 / +0.3%. No pair reaches +0.5% on G1 or on G4 prose MIN, and G4 prose falls in both. Fraser +4.4..4.5%, matched C1 coding +9.4..9.7%.
- **k7off (Moet's served config): reference only.** G1 **-8.0%** (prose -15.9%), G4 prose MIN **-16.8%**, G4 aggregate -16%, conc8 -17%, conc16 **-45%**, t1 -6.7%. With AV off every step verifies all 8 rows, so this is round 1's AV-off result again (-11..-22% at 4-8 users), worse at k=7.
- **A per-workload k would be needed and is not available.** k=6/7 pays only for code-like, high-acceptance traffic without thinking (Fraser, matched coding) and costs on prose and on the goal mix. vLLM 0.32's v2 GPU runner and DSpark have no per-request or per-batch draft width: `num_speculative_tokens` is server-wide, and `num_speculative_tokens_per_batch_size` exists only in the capture code. AV caps how many drafted tokens are verified, but the drafter still pays for all k. Even a perfect per-type choice would add at most about +0.4% to G1 (k6 for math only) and nothing to G4, whose binding prose session is best at k5.
- **Moet's one-launch Markov sampling: NOT ported**, because it is neither exact nor reachable in our config:
  - It would never run here. Their `fused_markov.sample_sequential` falls back to vLLM's loop for greedy drafts and whenever the acceptance estimator is on, and our estimator is on whenever AV is on (`speculator.py`: `use_acceptance_estimator = enable_adaptive_verification`). G1/G4 are greedy in any case, and our greedy drafts already use the vocab-sharded Markov path (8.3 MB instead of 66 MB per position per rank).
  - It is not bitwise. Their own patcher notes that the fused bias GEMV can differ in the last bf16 bit and then change a draft token.
  - It is not cheap. Its source (`models/deepseek_v41/amd/ops/fused_markov.py`) is not in our vLLM 0.32.0, and our sampled path keys its Gumbel noise with FRESH_NOISE, which a port would have to reproduce.
  - The only gain would be on sampled steps with AV off: launch overhead, about 0.03-0.08 ms per step.
- **Moet's other drafter kernels** (MXFP8 GEMV head, dflash input prep, SWA draft metadata) stay unported: at most 0.05 ms each (prep section 1). The most useful exact item in their repo is still the **mHC boundary overlap**: target side, kernel-ideas #8, about -0.8..-1.6 ms per single-user step if it proves bitwise on sm_121. Its sources are staged in `~/dsv41-stage/moet/src/`.
- **Goal status: unchanged.** The k5 control gives G1 101.66-101.81 and G4 prose MIN 48.24-48.59 (pinned, 3 instances), so G1 needs **+17.9%** and G4 prose **+3.2%**. A wider draft is not the acceptance lever G1 needs; a better-trained drafter, or a verify path that makes deep positions cheap, would be.

### 7. Live state and rollback
- **Live: instance mab-R**, the production config restored at the end: launcher defaults (k=5, AV unpinned = own startup profile, "pin miss"), launcher **b0cf94d0** on 8/8, supervisor 7bea1cf3, site manifest b1150ecfcf49, clock 0-2000 MHz.
  - Healthy since **02:21:45 UTC** (+355 s, first try) with 0 restarts and 0 ERROR/Traceback on 8/8.
  - exact.py **a8c3394af9d2** (917 / 459 / 2295 / 1373; 344/236/176/93/68).
  - `/health` returns 200 locally and on the LAN address; `dsv41-lanproxy` is active.
- **Nothing was installed or changed** in the launcher, site-packages, supervisor or services. New files only:
  - the pins and their source logs in `~/dsv41-stage/moet/pins/` on 8/8: `av-pin-k5-g3FU.json` eeac4f97, `av-pin-k6-mab-p6.json` b166aacd, `av-pin-k7-mab-p7.json` 5886e752;
  - results in `~/dsv41-moet-ab/` on spark-one.
- No telemetry, sampler or bench process is left on 8/8.
- **To serve k=6/7 anyway** (for example a code-only deployment): pass `DSV41_SPEC=6|7 DSV41_AV_COST_JSON=~/dsv41-stage/moet/pins/av-pin-k<K>-mab-p<K>.json` as supervisor arguments. exact.py a8 is verified for both. Graph memory is about 0.5 / 1.1 GB more per node, and the 1M probe was not run at k6/k7. Rollback: relaunch without the two variables.

### 8. Files
- **Mac scratchpad `mab/`:**
  - scripts: `mab.sh`, `mabblk.sh`, `mabphase.sh`, `clk8.sh` (with the re-read), `finsampler.sh`, `mabline.py`, `mabagg.py`, `pincmp.py`;
  - logs and results: `ab.log`, `ab.jsonl`, `tables.md` (full tables, including per-position acceptance and startup facts), `launches.tsv`, `pins.env`;
  - per instance: `ab/<label>/` (relaunch log, clock reads, pin make/check) and `runs/<label>/`, `runs/<label>-start/`.
- **spark-one `~/dsv41-moet-ab/`:** `runs/<label>/` (per-phase outputs, including `g1text.json` for divergence diffs: `python3 -I ~/dsv41-stage/moet/tools/g1text.py diff A B`), `runs/<label>-start/` (gstart, `rank0.startup.log`, `facts.json`), `mabphase.sh`, `finsampler.sh`.
