
## Goal window 4, client step (2026-10-27 13:22-14:05 UTC): Codex 07 (multi-turn prefix reuse) is a template default, not a server bug; it becomes a CLIENT RULE, and the server patch is NOT installed because it is not exact. Codex 03 (prose oracle): dump done on a pinned serving instance (exact); the replay finds NO calibration or tree headroom; the one lever is a better drafter head (ceiling +7% top-2 / +12% top-4 / +18% top-16 on the G4 prose session). Live config unchanged (launcher 31d04d8f)

Scope: the client step of goal window 4 (light; restarts only where needed): (a) Codex item 07, its `--live` multi-turn check on an idle server (`cached_tokens` per turn from `prompt_tokens_details`, thinking on and off); (b) Codex item 03, its prose-oracle dump during a serving period, then the offline replay. Trim rule: no long-prompt probes, no GSM8K-50 / longctx (nothing promoted). Every launch used `DSV41_GPU_CLOCK_MHZ=2000`; SM clock 1969-1989 MHz on 8/8 at every check; no foreign request in any timed phase.

Files: spark-one `~/dsv41-g4c/` (`scripts/`, `runs/`, `orc/g4c-O1/` = the dump, 314 MB, rank0-7), Mac scratchpad `g4c/` (`scripts/` = `orcab.sh` instance driver, `sw.sh` stopped window, `phase.sh`, `collect.sh`, `orcan.py`, `orcproj.py`; `orc/` = the bundle; `ab.log`, `ab.jsonl`, `runs/`). Bundle staged on 8/8: `~/dsv41-stage/g4/orc/` (NOT installed).

### 1. State check (13:22 UTC)
- Live: g4d2-FINAL, launcher **31d04d8f** on 8/8, healthy since 13:13:39, 0 requests in flight. Site status matched the README on 8/8: g4d f99d02f68320 (flags off), g4p aef7c824bb9a, fload ca05f12513a1.
- The task text still named launcher fd1e809e as "production"; that is superseded by the Loader phase (31d04d8f, `DSV41_FAST_LOAD`), and the decode+ops step had already left production defaults live. Nothing to restore.

### 2. Item 07: multi-turn prefix reuse, live (13:25-13:33 UTC, idle production instance g4d2-FINAL)
- Scripts (spark-one `~/dsv41-g4c/scripts/`): `mt_probe.py` (532cb8da; Codex's 4-turn `--live` conversation: ~5.3K-token code prefix, short questions; non-streaming, temperature 0, `reasoning_effort` low, `return_token_ids`; waits for `num_requests_running+waiting = 0` before every request) and `mt_ttft.py` (c6816244; streaming TTFT, long answers of 0.9-1.4K tokens per turn). Each variant has its own nonce, so variants never hit each other's cache. Results: `runs/mt_probe_1.json`, `runs/mt_ttft_1.json`.
- "Best possible" = full 64-token blocks of the common prefix of turn t's prompt ids and turn t-1's prompt + output ids (from `return_token_ids`); `cached_tokens` includes the 128-token SWA replay.

| variant (`chat_template_kwargs`) | turn 2 / 3 / 4: cached of prompt | = best possible? | where turn t leaves turn t-1's prompt + output |
|---|---|---|---|
| think, stock (default `drop_thinking`) | 5248/5371, 5312/5436, 5376/5504 (97.7-97.9%) | yes | at the LAST prompt token (`<think>` becomes `</think>`): the previous answer is always re-prefilled |
| think, `drop_thinking: false` + reasoning sent back | 5376/5431, 5504/5532, 5632/5656 (99.0-99.6%) | yes | after the whole previous output |
| think, `drop_thinking: false`, reasoning NOT sent back | 5248/5384, 5376/5462, 5440/5543 | yes | at the FIRST output token (empty `<think></think>`) |
| think, `tools` attached, reasoning sent back | 5632/5667, 5696/5761, 5824/5852 (98.9-99.5%) | yes | after the whole previous output (tools force `drop_thinking` off) |
| thinking off (`thinking: false`) | 5248/5327, 5312/5372, 5376/5420 (98.5-99.2%) | yes | after the whole previous output |

| long answers (streaming) | turn 2 / 3 / 4 cached | TTFT turn 2 / 3 / 4 | prompt at turn 4 |
|---|---|---|---|
| think, stock | 92.6 / 92.1 / 92.3% | 0.425 / 0.422 / 0.443 s | 6,586 tokens |
| think, `drop_thinking: false` + reasoning sent back | 98.7 / 99.5 / 99.7% | 0.396 / 0.280 / 0.275 s | 9,117 tokens |

- **The server side is already optimal:** in every case the cache hit equals the best possible block hit, so the prefix cache (MLA group + bounded SWA replay) loses nothing beyond block granularity. The only loss is the V4.1 template's default: it drops earlier reasoning and closes the old turn with `</think>`, so the previous answer (7-8% of the prompt with ~1K-token answers here, more when answers are long relative to the prompt) is prefilled again on the next turn. Codex's offline 82% -> 99% is the same effect on a conversation with longer answers.
- **Codex's server patch (`DSV41_PREFIX_STABLE_THINKING`): NOT installed, not staged.** It is server-side but **not exact**. When on, it rewrites every multi-turn thinking prompt so that the earlier reasoning stays in context, and its LRU "recovery" puts reasoning back even for clients that deliberately left it out. The model then sees a different prompt from turn 2 on, so outputs change against the stock template. exact.py and G1 are single-turn and would not notice, but the outputs of real chats would change. The same rendering is available per request with a stock kwarg (`deepseek_v41.py` passes `chat_template_kwargs.drop_thinking` to `encode_messages`), so no server change is needed.
- An exact server-side alternative (prefill the stock-rendered previous answer in the background after each thinking turn, so the next turn hits) would save at most ~0.15 s TTFT per turn here and would compete with other sessions' decode; not pursued.

**Client rule (multi-turn chats against this server):**
1. Thinking OFF: nothing to do; the whole conversation is reused.
2. Thinking ON, default: each new turn prefills the previous answer again (TTFT ~0.42 s at 5-7K prompts here). This is the model's documented default and the only exact-to-stock choice.
3. Thinking ON, to reuse everything: send each assistant turn back with `reasoning_content` (or `reasoning`) exactly as returned, together with its `content`, and set `"chat_template_kwargs": {"drop_thinking": false}` on EVERY turn (turn 1 renders identically either way). This gives 98.7-99.7% cached and TTFT 0.28 s on turns 3-4. It changes what the model sees (its earlier reasoning stays in context, as in DeepSeek's tool-call mode), so outputs differ from the default, and the prompt grows by the reasoning (+38% by turn 4 here). It is the client's choice per conversation.
4. Never set `drop_thinking: false` without sending the reasoning back: the template then renders an empty `<think></think>`, the prompt changes and nothing extra is reused.
5. With `tools`, the template already keeps reasoning (`drop_thinking` is forced off): tool loops must send `reasoning_content` back on every assistant message, or the cache breaks at the first output token of each step.
6. Keep the system message (and `tools`) at index 0; a mid-conversation system message moves the tools block and invalidates the cache from block 0 (Codex hazard C).

### 3. Item 03: prose oracle dump (instance g4c-O1, 13:45-13:52 UTC) and offline replay
**Bundle `orc`** (Codex shipped only a sketch; this is a full implementation, flag `DSV41_PROSE_ORACLE_DUMP=<dir>`, unset = off: no buffer, no extra kernel, stock graphs):

| file (site-packages) | before -> orc | what |
|---|---|---|
| vllm/v1/worker/gpu/spec_decode/dspark/speculator.py | 439b390e -> b7616890 | armed: inside the captured greedy-sharded draft graph, per draft position, the rank's vocab-shard top-64 (`DSV41_PROSE_ORACLE_K`) of the total logit (base + Markov bias), the bias of those candidates and the shard logsumexp go to persistent buffers. Read-only: draft tokens, confidences and the all-gather are unchanged |
| vllm/v1/worker/gpu/model_runner.py | 6a4f2433 -> 33e3472f | after `propose()` + `record_confidences()`: `record_step()` if the speculator owns the buffers (one getattr otherwise) |
| vllm/v1/worker/gpu/dsv41_oracle.py | ABSENT -> 091f1566 | writer: every rank pickles its shard top-64 per step to `<dir>/rank<R>/shard-*.pkl`; rank 0 adds request ids, sampled tokens / counts, confidences, AV-admitted and verify-cap-live draft counts, prompt token ids |

- No collective was added: the 8 shard files are merged offline. That merge is exact, because every global top-64 entry is in its shard's top-64.
- Checks before serving: CPU test (`tests/test_orc_cpu.py`: 40 random 8-shard merges equal to the global top-k; the writer fields) PASS; **GPU test (spark-two, service down): the capture inside a CUDA graph, 4 batch sizes x 5 replays, the graph's argmax equal to eager and the buffers equal to an eager topk / gather / logsumexp: PASS**; import check PASS.
- **Stopped window** 13:44:40-13:45:18 (`sw.sh`): stopped with 0 requests in flight, pre-status as in section 1, GPU test, `g4-all.sh install orc` (digest **a6223fa42311** on 8/8, backups `<file>.bak-g4-orc`, absent marker for the new module), import check.
- **Instance g4c-O1** (`orcab.sh`): pinned (`av-pin-k5-g3FU.json` eeac4f97, "pin hit") + `DSV41_PROSE_ORACLE_DUMP=~/dsv41-g4c/orc/g4c-O1`; healthy +126 s (13:47:42). Suite: exact.py, matched C1 (coding/prose x5, thinking off), G1 x2, G4 (warm-up + 6), exact.py #2. 0 ERROR/Traceback, no foreign request.
  - **Exact:** exact.py **a8c3394af9d2** at start and end, with the pin's counters 909/467/2335/1373 and 348/233/171/90/67. G1 digest **9a18f802d930f6e7**, and every G1 output (code/prose/math x 2 reps) is identical to the pinned control g4d2-C1. The dump changes no output.
  - **4-user outputs are not a bitwise gate in any instance:** the G4 outputs differ run to run in the control itself (g4d2-C1: 6 distinct output sets in 7 runs; g4d2-F1 7/7; g4c-O1 4/7; one set is common to all three).
  - **Cost while armed** (one device sync per step): C1 31.83 / 27.97 ms (control 30.25 / 26.34), G1 96.2 (101.6), G4 prose MIN 46.0 (48.1). It is for measurement windows only.
- **Merge** (`orcan.py`, on spark-one, ~10 s): 6,337 steps on 8 ranks aligned with 0 missing / count / validity / draft mismatches. The merged global top-1 equals the drafted token at all 49,810 positions. The replayed acceptance equals the verifier's own count in all 9,469 records after startup (20 of 493 records of the startup exact.py batch do not cross-check; not used). Workloads were labelled from the prompt tokens and the phase.

**Headroom per workload** (draft steps n; AL = 1 + accepted drafts per step, live; recall = the target's rank in the drafter's top-K at the first missed position, i.e. conditioned on a correct prefix):

| workload | n | AL live | drafts live / admitted per step | steps missing at pos 0 | recall at first miss top-2 / 4 / 8 / 16 / 64 | Markov-weight calibration (held-out) AL | cap off AL | ceiling: target in top-2 / 4 / 16 counts as hit, AL |
|---|---|---|---|---|---|---|---|---|
| **prose, G4 session (the slow one)** | 1,964 | **1.810** | 1.31 / 1.65 | 37% | **19.7 / 36.9 / 48.6 / 59.3 / 77.2%** | 1.817 (+0.4%) | 1.840 | 1.934 / 2.032 / 2.133 |
| prose, G1 (1 user) | 1,104 | 1.846 | 1.55 / 2.16 | 41% | 23.0 / 41.0 / 54.0 / 65.3 / 79.6% | 1.844 (-0.1%) | 1.889 | 2.011 / 2.145 / 2.312 |
| prose, matched C1 (thinking off) | 558 | 2.742 | 2.82 / 3.48 | 20% | 40.0 / 63.2 / 80.8 / 87.3 / 95.8% | 2.735 (-0.3%) | 2.821 | 3.052 / 3.251 / 3.523 |
| code / math / mixed, G4 | 1,475 / 1,017 / 1,253 | 2.420 / 3.557 / 2.884 | 2.21 / 3.24 / 2.62 live | 24 / 15 / 21% | top-2 39.5 / 48.0 / 44.9%, top-16 88.0 / 91.5 / 88.4% | +-0.3% | | |
| code, matched C1 | 271 | 5.214 | 4.74 / 4.89 | 2% | 49.2 / 76.9 / 84.6 / 92.3 / 93.8% | -0.8% | 5.262 | |

- The drafter's per-position Markov weight is already right: the held-out optimum is alpha 0.8-1.1 at every position, and the AL change is within +-0.4%. The DSpark confidence head is calibrated within a few points in every bucket and position. AV and the verify cap therefore already size the verify window well.
- For thinking prose the drafter is diffuse, not miscalibrated. When its top-1 misses, the target is its second choice only 20-23% of the time, and outside its top-64 21-23% of the time.

**Cost-adjusted** (`orcproj.py`, first order). Baselines: the pinned control g4d2-C1, i.e. G1 prose 76.52 tok/s (24.12 ms/step), C1 prose 26.34 ms, G4 4-session step 44.56 ms. Cost per extra verified row: 2.04 ms at 1 user (matched C1 coding vs prose), or 2.84 ms on the pin's curve; 1.79 ms at 4 users (pin curve, 12-16 rows). Server-wide = every G4 request pays its own extra rows.

| change | G1 prose tok/s | C1 prose tok/s | G4 prose session, server-wide (prose request only) | goal: +3.4% (prose MIN 48.36 -> 50) |
|---|---|---|---|---|
| Markov-weight calibration (Codex's "calibration only") | -0.1% | -0.3% | +0.4% | no |
| verify cap off (all admitted drafts live) | -2.6% | -2.2% | -3.7% (+0.3%) | no |
| tree K=2, confidence-gated (best realizable: conf < 0.3 / 0.5 / 0.7) | -0.8 / -1.4 / -0.5% | 0.0 / -1.8 / -1.7% | 0.0 / -1.5 / -6.1% (+0.1 / +0.6 / +1.7%) | no |
| tree K=4, gated | -4.3..-7.3% | -0.2..-10.4% | -0.2..-18.5% | no |
| tree K=2 at the first miss (oracle placement, not realizable) | +0.3% | +2.6% | -7.9% (+2.7%) | no |
| better drafter head, ceiling at the same verify width: target in top-2 / top-4 / top-16 promoted | +8.9 / +16.2 / +25.2% | +11.3 / +18.6 / +28.5% | **+6.9 / +12.2 / +17.8%** | yes if it captures about half of the top-2 ceiling or a quarter of the top-16 ceiling |

- **Codex's own analyzer** (`analyze_prose_oracle.py`, unchanged, run on the same records converted to its npz input with the current pin; `runs/codex-analyzer-O1.{txt,json}`) reproduces the recall tables above. It also reports "calibration" +10.4% for G1 prose and a G4 prose session of 52.1 tok/s. **That number is a replay error:** its calibration (and tree and ceiling) replays walk all 5 draft positions and ignore the live window that AV and the verify cap chose, while charging +0.25 verify rows. Its own `uncapped_5drafts` row (verify all 5) gives the same AL for the G4 prose session (2.119 against 2.128) at +3.69 rows and 41.9 tok/s. The α refit itself is worth +0.009 AL. The self-test's synthetic prose had a built-in Markov mismatch; the real drafter does not.
- **What the oracle recommends:** no calibration change and no tree (static or gated) for prose. The only lever with enough headroom is a better DSpark draft head for thinking prose, for example a fine-tune or self-distillation of the Markov/LM head, or a top-16 reranker, trained on this model's own greedy thinking-prose outputs. The dump format is a ready training and evaluation set: per position the top-64 with base and Markov parts and the target. This fits goal window 5's drafter-capture item. Whether a trained head reaches a useful share of the ceiling is open, and only training can tell.
- **Could it be exact?** Any drafter change (head, calibration, tree) changes the drafts, so it changes acceptance, AV's budget and the verify layout. The greedy target tokens are the same in exact arithmetic, but bitwise near-ties can flip with the verify layout. This is the same class as the Moet k6/k7 A/B ("greedy-equivalent but not bitwise"): exact.py a8 held there, code and math G1 stayed bitwise, and the known G1 prose tie (character 3885, " capital-intensive" / "ing capital-intensive") flipped between 9a18f802 and 69ad8721, exactly as an AV-profile change flips it.
  - **Gate it would need:** a pinned A/B, with the arm's own AV pin if the drafter cost changes. Then:
    - exact.py combined hash a8c3394af9d2 on every launch (the acceptance counters change by design);
    - G1 code / math bitwise (596b478a / 92e5d8cf);
    - G1 prose = one of the two known tie outcomes (923a46be / e8a524f3), with any other first divergence counting as a fail;
    - G4 / conc: no equality gate is possible (the control is not reproducible run to run), so a first-divergence census instead: each session's first divergence from its single-user greedy output must be a near-tie, at the control's rate.
  - Under the strict rule (identical G1 digest under the current pin), any arm that flips the prose tie fails. Promoting one is a user decision, as for k6/k7.

### 4. Production restore (13:57-14:03 UTC)
- Stopped (0 requests in flight) and `g4-all.sh uninstall orc`: digest 756484cd51b4 = baseline. Then on 8/8: speculator 439b390e, model_runner 6a4f2433, no `dsv41_oracle.py`; g4d f99d02f68320 / g4p aef7c824bb9a / fload ca05f12513a1 unchanged; launcher 31d04d8f. Supervisor preflight manifest back to **18b3d8175e12**.
- **g4c-FINAL:** supervisor launch with defaults (unpinned, production), healthy at 14:01:06 UTC (+198 s).
  - exact.py **a8c3394af9d2** at start and after the suite (own profile 908/468/2340/1373, "pin miss").
  - G1 x2 **101.67** tok/s, digest 69ad872124100141 (the other outcome of the known prose tie; unpinned production lands on either; g4d2-FINAL 102.03).
  - 0 ERROR/Traceback on 8/8, no foreign request, MemAvailable min 13.5 GB, SM 1969-1989 MHz.

### 5. Live state and rollback
- **Live:** g4c-FINAL, launcher **31d04d8f** defaults on 8/8, supervisor 7bea1cf3 (manifest 18b3d8175e12), AV unpinned, clock lock 0-2000 MHz, LAN proxy unchanged.
- **Site:** g4d installed with every flag off (f99d02f68320), g4p (aef7c824bb9a), fload (ca05f12513a1). orc NOT installed.
- **Nothing to roll back.**
- **Re-run the dump** (stopped window): `ssh spark-one cat ~/dsv41-stage/g4/common/g4-all.sh | bash -s install orc` (expects g4d's model_runner 6a4f2433 and g4p's speculator 439b390e). Launch pinned with `DSV41_PROSE_ORACLE_DUMP=<dir>`, run the prose traffic, then `collect.sh <label>` and `orcan.py`. Uninstall afterwards with `... | bash -s uninstall orc`.

### 6. Notes for the next steps
- 07 is closed: a client rule, no server work. Clients that want full reuse in thinking chats opt in per conversation (rule 3). Tool clients must send `reasoning_content` back (rule 5).
- 03: the prose goal (+3.4% on the G4 prose session) is not reachable by calibration, the verify cap or trees. A trained drafter head is the only lever with that headroom. If one is trained, judge it on this dump's held-out prose records first (it must beat the top-1 hit rate at the first miss without breaking the 53-59% of positions where top-1 is already right), then run the E1 gate above.
