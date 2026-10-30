
## Long-prompt determinism (goal window 5, Determinism phase, 2026-10-27 16:14-18:07 UTC): sparse-indexer top-k boundary ties; `DSV41_TOPK_CANON_TIES=all` PROMOTED (launcher 95f08eff -> **f214c2c9**)

**Result.**
- **Cause.** Some rows of the sparse-indexer top-k have a boundary tie: two or more candidates have exactly the same fp32 index score as the row's 512th value. vLLM's top-k kernels then keep an arbitrary one of them, picked by an atomic race:
  - prefill: `top_k_per_row_prefill`;
  - decode: `persistent_topk` / `top_k_per_row_decode`.
- **Why the earlier fix missed it.** `DSV41_TOPK_CANON` (2026-10-24) makes the ORDER of the selected indices canonical, but not the SET.
- **One tie is enough.** It changes which KV entries one query attends to. From there, the next index layer's inputs differ in hundreds of rows, and the greedy output diverges.
- **Where the threshold is.** It is not the ~2K of the hypothesis. It is where the indexer really starts selecting:
  - in prefill, above 1,024 prompt tokens: layers 2/8/14 are ratio-2, so `index_topk` 512 = 1,024 tokens;
  - in decode, above 512 context on the ratio-1 layers (`index_topk` 512).
- **Why the drift looks sporadic.** Ties depend on content and are rare: 1-2 tied rows among ~9,600 selecting rows of a 4K prompt. No probe drifted at 1,300 tokens or fewer; the first drift was at 1,536.
- **Fix: `DSV41_TOPK_CANON_TIES=all`.**
  - Runs after the stock top-k on prefill and decode rows.
  - A tied row keeps every score above the 512th value plus the lowest column indices among the scores equal to it.
  - Rows without a tie are untouched (bitwise stock).
- **Verification.**
  - Cold probes from 2.5K to 128K tokens and three recorded 6K tool prompts give 4 identical runs out of 4, for both tokens and logprobs. The flag-off control gave 2-4 distinct token outputs out of 4 on 7 of the 9 probes, and drifting logprobs on an eighth.
  - exact.py a8c3394af9d2 with the pin's counters, and the G1 reference digest, are unchanged.
  - GSM8K 50/50; longctx PASS.
  - TTFT: 16K +0.0%, 128K +0.35% (same-instance A/B). 1M 168.4 s, good class.
  - Decode within noise: G1 -0.16% in the pinned pair.
- **Not fixed: warm (prefix-cache hit) vs cold.**
  - They still differ from the first generated token. This is row-count-dependent numerics, not ties (section 5).
  - Warm runs now agree with each other.

### 1. Reproduction on the live production instance (g4v-FU, launcher 95f08eff, AV unpinned, idle; 16:16-16:45 UTC)
Setup (`~/dsv41-g5det/scripts/detprobe.py`):
- Token-id prompts on `/v1/completions`: [BOS] + the first L-1 ids of a fixed text.
  - prose = GPL-3 + Apache-2.0 + GFDL-1.3;
  - code = an old copy of `sparse_attn_indexer.py`.
- Greedy, 128 tokens, `ignore_eos`, one request at a time.
- Cold = a fresh `cache_salt` per request; 4 runs per length.
- Pass A records tokens only, with cold and warm runs (warm = one salt, so runs 2-4 hit the prefix cache).
- Pass B is cold with the sampled token's logprob per step, used as a bitwise fingerprint of every step. "Step 0" is the prefill's own logits.

| prompt tokens | prose, cold: distinct tokens / logprob seqs (first differing step) | code, cold: same | warm vs cold, first differing token (prose / code) |
|---|---|---|---|
| 512, 768, 1,100, 1,300 | 1 / 1 | 1 / 1 | not run |
| 1,024 | 1 / 1 | 1 / 1 | same / 80 |
| 1,536 | 1 / **3 (step 0)** | 1 / 1 | same / 0 |
| 1,900 | 1 / 1 | 1 / 1 | 94 / 122 |
| 2,000 | 1 / 1 | 1 / 1 | same / 3 (the 3 warm runs also differ from each other: 4 distinct) |
| 2,048 | 1 / **3 (step 17)** | 1 / 1 | same / 44 |
| 2,100, 3,000, 4,200 | 1 / 1 | 1 / 1 | same / 36, 29, 39 |
| 2,500 | 1 / 1 | **3 / 4 (steps 0-1)** | same / 82 |
| 4,000 | 1 / 1 | **4 / 4 (step 0)** | same / 12 |
| 6,000 | 1 / **4 (steps 18-19)** | 1 / 1 | same / 13 |
| 8,000 | 1 / **4 (step 1)** | 1 / 1 | same / 0 (warm runs: 3 distinct) |

- Drift starts at 1,536 tokens, but only on some prompts; 2,100-4,200 prose is clean.
- The step at which it starts is either 0 (the prefill differs) or a later decode step.
- Prose often keeps the same tokens while its logprobs drift. Code flips tokens more readily.

### 2. Localization
**Kernel test** (`tk_tie_test.py`, `tk_ties_unit.py`; spark-eight; no model; 20 repeats per case):
- Continuous random scores: the selected SET is identical in 64/64 rows, and the ORDER differs in 64/64 (the part `DSV41_TOPK_CANON` already fixes).
- Scores with boundary ties (values on a 0.25 grid, or all equal): the SET differs run to run in 61-64 of 64 rows, for all three stock kernels. None of them keeps the lowest index consistently.
- Rows with fewer than 512 finite scores: the stock kernels pick an arbitrary subset of the -inf columns (31-32 of 32 rows vary). `top_k_per_row_decode` on 300K-wide rows returns only part of 512 (for example 304 valid indices). This case cannot occur here, because the candidate mask keeps ≥ 16,384 positions per row.

**In-forward dump.** Instance D1 (pinned; 16:52) ran with the new debug flag `DSV41_DET_DUMP` and the tie fix off. For every indexer call, rank 0 saved:
- the stock top-k rows;
- per-row boundary statistics;
- per-row fingerprints of the attention input, index queries, head weights, gathered index keys and scores.

Result for the code 4,000-token prompt, two cold runs:

| index call (layer, rows x compressed entries) | inputs (attn input / q / weights / keys) | scores | stock top-k SET | ORDER |
|---|---|---|---|---|
| layer 2, 4,000 x 2,000 | identical | identical | **differs in 1 row** | differs in 2,975 rows (CANON sorts it) |
| layer 8, 4,000 x 2,000 | differ in 636 rows; 319 / 2,000 keys | 2,511 rows | 2,323 rows | - |
| layer 14, 4,000 x 2,000 | differ in 2,511 rows; 1,256 / 2,000 keys | 2,511 rows | 2,510 rows | - |
| layers 20/24/28/32/36, 128-row decoder-replay batch x 4,000 | all 128 rows | all | all | - |

- **The tied row.** Row 1489 has 745 entries. Its 512th score, -2.0917205810546875, is shared by columns 222 and 306: 511 scores are greater and 2 are equal. The two keys have different bytes and the same block scale. Each run keeps one of the two at random.
- **Other prompts.**
  - Prose 1,536 and code 2,500 each had one tied row in layer 2. In this pair both runs happened to keep the same column; in production these lengths drifted in 2-3 of 4 runs.
  - The 4,000-token prompt also has a tie in layer 8 on one of its trajectories.
- **With the fix on (same instance).** `TIES=prefill` was switched on 8/8 through the debug file. The stock SET in row 1489 still differed run to run, but the fixed output was identical. Every later layer was bitwise identical, and 3/3 runs gave identical tokens and logprobs.
- **Ruled out.** Inputs are bitwise identical up to the first tied top-k, so none of these is the cause:
  - layers 0-1: SWA attention, MoE (the ep_scatter atomics), Engram, mHC;
  - SP prefill and E7;
  - all-reduces;
  - async scheduling;
  - chunking (single chunk);
  - the index-score kernel.
- **Prefill index calls run as follows:** layers 2/8/14 on every row; layers 20-36 only on the 128-row replay batch.

**Decode.**
- The decode indexer runs inside the FULL CUDA graphs, so Python cannot dump it at replay. The evidence is an A/B instead.
- With prefill rows fixed but decode stock (D1 `TIES=prefill`), prose 6,000 still drifted from decode step 18: 4 distinct logprob sequences, same tokens.
- In the same pass, prose 8,000 / 2,048, code 2,500 / 4,000, prose 1,536 and the three tool prompts became identical, so their drift had come from prefill ties.
- With decode rows fixed as well (D2, 17:09: debug file set to `all` before startup, so the graphs capture the fix), prose 6,000 was 4/4 identical. So were all 13 cold cases there, including big 2.5K / 8K / 32K / 128K.

**Layer-20 candidate blocks** (`scores.topk(2048)`, used only for contexts > 16K):
- In a 20-repeat test, `torch.topk` gave the same result run to run on tied block scores in every shape, although it does not always keep the lowest index.
- No change was made. Cold 32K and 128K probes are deterministic with the fix.

### 3. Fix: `DSV41_TOPK_CANON_TIES` (new site module `layers/dsv41_topk_ties.py` 943961bc, hooks in `sparse_attn_indexer.py` f2ae649b -> 005785f9)
- **Where it runs.** After the stock top-k and before CANON:
  - every prefill chunk (`top_k_per_row_prefill`; indices relative to the row start);
  - with `all`, every decode call (absolute indices).
- **How it works.** One Triton program per row:
  1. reads the 512 selected scores, takes their minimum v_k;
  2. counts the row's scores > v_k and == v_k;
  3. only if (#> + #==) > 512, rewrites the row ascending: every score > v_k, then the lowest-index scores == v_k until 512.
- **Edge cases.**
  - A -inf boundary is covered too.
  - These are left to stock: rows of ≤ 512, rows with a NaN or +inf k-th value, and unsupported layouts (a one-time warning, no crash).
- **Decode split.** Decode rows wider than 16,384 (the logits width is `max_model_len`/ratio) split the count over up to 64 programs per row (partials, no atomics). One program per row then decides, and rewrites the row if tied.
- **Unit test** (`tk_ties_unit.py`, prefill + both decode kernels; continuous / tied / all-equal / masked; widths 700 to 300K):
  - untied rows bitwise equal to stock;
  - tied rows equal to a float64 reference;
  - 0 rows varying over 20 runs.
- **Cost, prefill.** The check runs about as long as the stock top-k (+94% at 16K x 8K, +127% at 16K x 16K, kernel only). In serving, the top-k is a small part of TTFT: same-instance A/B +0.0% at 16K, +0.35% at 128K (below).
- **Cost, decode.** Measured inside a CUDA graph on spark-eight:
  - 4.3-4.6 µs per index call at G1 contexts (6 rows, ≤ 1.1K); 7.3 µs for 24 rows at 2K; 20.2 µs for 96 rows;
  - 12.3 µs at 131K and 24 µs at 1M context (6 rows);
  - in total, about 35 µs per decode step at G1 contexts (8 index layers), ~0.1% of a 30 ms step.
- **Modes.** `all` = prefill and decode (live default); `prefill` = prefill rows only; `0` = stock.
- **Debug only:** `DSV41_DET_DUMP=<dir>`.
  - Rank 0 writes per-call dumps while `<dir>/ON` exists.
  - `<dir>/TIES` overrides the mode for prefill at run time and must be written on 8/8.
  - Decode graphs capture the mode at startup.

### 4. Verification
**Same-instance TTFT A/B.** Instance D1, AV pin, cold `sbench solo`. The debug file was switched on 8/8 between blocks, in the order 0, prefill, 0, prefill:

| TTFT | `TIES=0` | `TIES=prefill` | delta |
|---|---|---|---|
| 16K (3 prompts x 2) | 2.502 s | 2.503 s | +0.0% |
| 128K (2 prompts x 2) | 19.641 s | 19.709 s | +0.35% |

**Pinned pair.** Launcher 95f08eff on both, with the k5-g3FU pin (eeac4f97). C0 has the flag off; C1 has `DSV41_TOPK_CANON_TIES=all` from the supervisor. Both ran the trimmed decode suite plus the cold/warm determinism probes; C1 also ran the prefill/long-context checks.

| metric | C0, flag off (17:21) | C1, `all` (17:32) | delta |
|---|---|---|---|
| exact.py (both runs) | a8c3394af9d2 909/467/2335/1373, 348/233/171/90/67 x2 | identical x2 | identical |
| G1 (code / prose / math) | 101.654 (96.00 / 76.54 / 132.43) | 101.495 (95.82 / 76.38 / 132.29) | -0.16% |
| G1 digest (3 reps / first 2) | bc1e8386878bd2b7 / 9a18f802d930f6e7 | identical | identical |
| matched C1 coding / prose (ms per step) | 30.23 / 26.33 | 30.25 / 26.41 | +0.07% / +0.30% |
| G4 mean / prose MIN / G4min_hat | 60.79 / 48.56 / 48.03 | 62.39 / 49.34 / 48.13 | noise (G4 outputs differ run to run in both) |
| conc8 / conc16 (aggregate tok/s) | 262.0 / 407.1 | 263.2 / 404.1 | +0.5% / -0.7% |
| TTFT 16K / 128K | - | 2.498 / 19.710 s | see the A/B above |
| GSM8K-50 / longctx | - | 50/50 / PASS (needle 60/60, copy 1.000) | - |

Cold determinism probes, 4 runs each. Cells give distinct tokens / logprob sequences (the first differing step); "1/1" = all 4 runs identical.

| probe | C0, flag off | C1, `all` | g5-PU2, production, `all` |
|---|---|---|---|
| big 2,500 | 1/1 | 1/1 | 1/1 |
| big 8,000 | **3/4 (steps 0-3)** | 1/1 | 1/1 |
| big 32,000 | **2/4 (step 0)** | 1/1 | 1/1 |
| big 128,000 | **4/4 (step 0)** | 1/1 | 1/1 |
| code 4,000 | **4/4 (step 0)** | 1/1 | 1/1 |
| prose 6,000 | **1/4 (steps 18-19, decode)** | 1/1 | 1/1 |
| tool B:T16 (5,903) | **3/4 (step 0)** | 1/1 | 1/1 |
| tool B:T02 (5,891) | **4/4 (step 0)** | 1/1 | 1/1 |
| tool A:T01 (6,107) | **3/3 (step 0)** | 1/1 | 1/1 |
| warm code 4,000 (1 cold + 3 warm) | 2/2: warm runs identical, differ from cold at token 12 | same | same |
| warm big 32,000 | tokens 1, logprobs 2 | 2/2 (token 82) | 2/2 (token 82) |

"big" = token-id prompts from a fixed 1.43 MB code corpus: the 57 `.py` files under vllm `v1/core`, `engine` and `config`, sha f3651c43077e.

**Production (g5-PU2).**
- Launched 17:53 by the supervisor with launcher f214c2c9 defaults: AV unpinned, no extra variables.
- Healthy at +205 s.
- exact.py **a8c3394af9d2** x2, own profile 910/466/2330/1373.
- G1 x3 **101.52** (95.61 / 76.32 / 132.64). The digest is bc1e8386878bd2b7 (first 2: 9a18f802d930f6e7), the reference outcome.
- TTFT: 16K 2.505 s (g4v-FU 2.494); 128K 19.668 s (19.617).
- ONE cold 1M: 168.4 s, good class (late/early 1.20, chunks 2.4 -> 2.9 s). g4v-FU was 166.2 s; good-class launches range 166-170 s. Lowest MemAvailable on spark-one: 8.73 GB.
- Determinism: 9/9 cold probes identical; warm runs identical to each other.
- 0 errors, 0 restarts; SM 1976-1989 MHz at every check.

**First production launch (g5-PU, 17:48).**
- It gave the AV-profile alternate aeef687377ef (942/437/2185/1373).
- Its unpinned startup profile ran while the site tree was being hashed on 8/8: verify_ms(2) was 22.65, against 18.1-19.1 in 10 earlier launches.
- It was relaunched with nothing else running (g5-PU2), which gave a8.
- This was not a code effect: the pinned C1 and D2 reproduced the pin's counters exactly.

### 5. Warm (prefix-cache hit) vs cold: not fixed (row-count dependence)
- **Status.** After the fix, warm runs (one `cache_salt`, runs 2-4 cached) are identical to each other in every case tested: D1, D2, C1 and PU2 (6 cases in D2). They still differ from the cold run:
  - code 2,000 / 4,000 / 8,000 at token 3 / 12 / 0;
  - big 32K at token 82;
  - tool B:T16 at token 231;
  - prose 2,048: tokens equal, logprobs differ from step 0.
- **Evidence** (D2 dump, code 4,000, cold vs warm). The warm step recomputes 160 rows (32 uncached + 128 SWA replay) through layers 2/8/14 and the replay layers.
  - Already at layer 2, the attention inputs of those 160 rows differ from the cold prefill in 160 of 160 rows, with no tie involved. The same tokens went through layers 0-1 in a 160-row batch instead of a 4,000-row one, and the FP8 linears, the MoE grouped GEMMs, the mHC split-K and attention choose tilings and reduction splits by row count.
  - 16 of 2,000 index keys differ: the re-written tail.
- **What a fix would take.** Batch-invariant kernels, with one tiling and reduction order for every row count, in:
  - the DeepGEMM FP8 linears;
  - the M32 / M64 / contiguous FP4 MoE paths;
  - mHC pre/post;
  - sparse MLA prefill and decode;
  - the indexer;
  - the all-reduce paths.

  That is a large rewrite, and batch-invariant GEMMs usually cost 1.2-2x at small row counts. It was not started.
- **Practical rule.** Compare long prompts cold with cold (a fresh `cache_salt`) or warm with warm; a warm result will not equal a cold one.

### 6. Live state, files, rollback
- **Launcher** `~/dsv41-tp8.sh`: 95f08eff -> **f214c2c9** on 8/8. Backup `~/dsv41-tp8.sh.bak-g5-det` = 95f08eff. One added line, after `DSV41_TOOL_STREAM_NEWLINE`:
  `export DSV41_TOPK_CANON_TIES=${DSV41_TOPK_CANON_TIES:-all}  # g5det 2026-10-27 ...`
- **Supervisor.** 7bea1cf3, unchanged. Its preflight manifest hashes the launcher, so it changed 40eb7abf08c7 -> **d773ed60f27b**.
- **Site.**
  - `model_executor/layers/sparse_attn_indexer.py` f2ae649b -> **005785f9** (backup `.bak-g5-det`);
  - new `model_executor/layers/dsv41_topk_ties.py` **943961bc**;
  - vllm site tree 38062de05f0a -> **4baa4e458a1a** (5,453 files), identical on 8/8;
  - installer `~/dsv41-stage/g5det/g5det_install.py` (850e9c9f) `status|install|uninstall`, staged on 8/8.
- **Live instance** g5-PU2: env `DSV41_TOPK_CANON_TIES=all` from the launcher; no `DSV41_DET_DUMP`, no `DSV41_AV_COST_JSON`.
- **The AV pin** `~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json` (eeac4f97) stays valid: exact.py counters and the G1 digest are identical with the fix.
- **Rollback.**
  - Quickest: `DSV41_TOPK_CANON_TIES=0` as a supervisor argument (no file change).
  - Or restore the launcher on 8/8: `cp ~/dsv41-tp8.sh.bak-g5-det ~/dsv41-tp8.sh`.
  - Full removal: stop the service, run `python3 ~/dsv41-stage/g5det/g5det_install.py uninstall` on 8/8 (restores f2ae649b and deletes the module), then restore the launcher.
- **New check for every prefill / long-context candidate (~4 min):**
  `cd ~/dsv41-g5det && python3 -I scripts/dbgrun.py - <out.jsonl> <label> big:2500,big:8000,big:32000,big:128000,code:4000,prose:6000,l3:B:T16-spam-reply:1,l3:B:T02-classify:1,l3:A:T01-reviews:1 4 0 cold`
  Every case must print `distinct_tok=1 distinct_lp=1`.
- **Raw results.**
  - spark-one `~/dsv41-g5det/`: `runs/` (A/B production passes, D1/D2, g5-C0/C1/PU2 suites, TTFT A/B), `dumps/D1/` (~0.3 GB of debug dumps; safe to delete), `scripts/`.
  - Mac scratchpad `g5det/`: `ab.log`, `launches.tsv`, `runs/<label>/vsumm.md`, drivers `scripts/g5inst.sh`, `ttftab.sh`, unit tests.
- **Goal status: unchanged.** Production G1 101.5 of 120 tok/s. The G4 prose session is 48.6-49.3 of 50 in the pinned pair. This was an exactness item, not a speed lever.
