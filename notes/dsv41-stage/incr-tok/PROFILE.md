# Incremental tokenization: request path trace and stage profile (DSv4.1 API server)

Date: 2026-11-03. Host for all measurements: spark-two, CPU only (`CUDA_VISIBLE_DEVICES=`, `nice -n 19`,
venv `~/venvs/vllm032`). spark-two has the same CPU (GB10: 10x X925 at 3.9 GHz and 10x A725 at 2.8 GHz) and
byte-identical code as spark-one (sha256 below). Data: synthetic only (`synth.py`). Nothing was called on
port 8000, nothing was restarted or installed, and no site-packages file was touched.

## Bottom line

1. **Tokenizing dominates; rendering does not.** At about 300K tokens the API server spends about 950 ms
   turning one request into prompt ids. Of that, **907 ms (95.6%) is the HF tokenizer call**
   (`renderers/base.py:638`). Rendering the chat template is 10.7 ms and JSON + pydantic is 7.9 ms. Every
   other step is 1 to 17 ms.
2. **Tokenize cost grows about 0.28 s per 100K tokens** (2.8 µs/token, 0.88 µs/char). That is roughly
   80% of the 0.35 s/100K first-token growth found in the audit. The audit's estimate of 0.15 s/100K for
   render + tokenize was about 2x too low.
3. **The incremental prototype gives identical ids on every test.** It cuts at the last added-token
   literal inside the common prefix, then tokenizes only the new suffix. It ran on 12/12 profiled
   requests (50K to 506K) and 273/273 end-to-end steps across 7 chat modes, with 0 mismatches. One hit
   costs **10 to 19 ms** where a full tokenize costs 178 to 1428 ms. Saving per agent step: **about
   0.17 s at 55K, 0.49 s at 158K, 0.89 s at 304K and 1.41 s at 506K.**
4. **Do not cache rendering.** It is 1.2% of the path at 506K (18 ms), and its output depends on global
   message state (last user index, drop_thinking, tools). It can be revisited later.
5. **Safe cut points** are the start and end of any literal occurrence of an added-token string,
   **except `｜DSML｜`**, which overlaps itself. The proof is in section 4. The cuts the prototype uses are
   `<｜end▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`, `</think>` and `<think>`, and the V4.1 encoder
   writes one of them at every message boundary. Newline, blank-line and `<tool_result>` cuts are **not**
   safe: they failed 29%, 4.6% and 20% of the time.

## 1. Code path: `/v1/chat/completions` to prompt token ids

Paths are relative to `~/venvs/vllm032/lib/python3.12/site-packages/vllm/`. The files are byte-identical
on spark-one and spark-two (sha256 prefixes in section 7). Launcher `~/dsv41-tp8.sh` sha `66eaa1d9`.

**Startup (once)**
- `config/model.py:750-751`: arch `DeepseekV41ForCausalLM` sets `tokenizer_mode = "deepseek_v41"`
  (there is no CLI flag for this).
- `tokenizers/registry.py:50`, then `tokenizers/deepseek_v41.py:117-121`: `TokenizersBackend.from_pretrained`
  is wrapped by `get_deepseek_v41_tokenizer` (`:48`) and `get_cached_tokenizer` (`tokenizers/hf.py:106`).
  `VLLM_USE_FASTOKENS` is unset in the live process environment, so the stock Rust `tokenizers` 0.23.3
  backend is used (transformers 5.19.0).
- `renderers/registry.py:26`: `deepseek_v41` maps to `DeepseekV4Renderer` (`renderers/deepseek_v4.py:24`).
- `renderers/base.py:143`: one `_SwappableExecutor(max_workers=renderer_num_workers)`. The default is
  **1** (`config/model.py:408`). Both the chat-template render (`deepseek_v4.py:32`) and the tokenize call
  (`base.py:155`) run on this executor.

**Per request** (stage names match the timing table)

| # | Stage | Where |
|---|---|---|
| A | `http_json_pydantic` | `entrypoints/openai/chat_completion/api_router.py:41-62`. FastAPI parses the JSON body into `ChatCompletionRequest` (pydantic) before the handler runs. |
| B | `params_parser_tools` | `chat_completion/serving.py:281` → `:296 _create_chat_completion`. Parser instance (`:307`), `_effective_chat_template_kwargs` (`:192`). `render_chat_request` (`:258`) → `online_renderer.py:224 render_chat`. `tool_dicts = [t.model_dump() ...]` (`:287`). `preprocess_chat` (`:752`) builds `tokenize=False` kwargs (`:773-781`), `tok_params` (`:784`), `chat_params` (`:785`) and checks decode-side reuse ids (`:793`, not used here). |
| C | `parse_chat_messages` | `online_renderer.py:804` → `base.py:1327 render_chat_async` → `:1348` → `deepseek_v4.py:69 render_messages_async` → `:74 parse_chat_messages_async` (`renderers/chat_utils.py:2230`). The image items are collected here. |
| D | `render_apply_chat_template` | `deepseek_v4.py:82` (on the executor) → `tokenizers/deepseek_v41.py:54 apply_chat_template`. Steps: `_normalize_messages` (`:20`, one `copy.deepcopy` of all messages, `:23`), tools attached to the system message (`:63-68`), thinking/effort resolved (`:70-86`), `encode_messages` (`tokenizers/deepseek_v41_encoding.py:533`). Inside that: `merge_tool_messages` (`:73`, another deepcopy per message at `:83`), `sort_tool_results_by_call_order` (`:129`), drop_thinking forced off when any message has tools (`:556-558`), and the per-message `render_message` loop (`:568-575`, `:309`). Returns a `str` because `tokenize=False` (`deepseek_v41.py:94,101`). `parse_dec_only_prompt` is at `deepseek_v4.py:88`. |
| E | `tokenize` | `base.py:1352 tokenize_prompts_async` → `:795` → `:712 _tokenize_singleton_prompt_async`. `apply_pre_tokenization` (`:725`, `params.py:445`, a length check on chars) → `_tokenize_prompt_async` (`:727`, executor) → `_tokenize_prompt` (`:628`) → **`tokenizer(prompt, truncation=True, max_length=max_input_tokens+1, add_special_tokens=False)`** (`:638`; kwargs from `params.py:351-376`; `add_special_tokens` defaults to False for chat, `chat_completion/protocol.py:337`). HF `__call__` → `transformers/tokenization_utils_tokenizers.py:961 _encode_plus` → `:1063 self._tokenizer.encode_batch([...])` (Rust). `_build_tokens_prompt` copies with `list(token_ids)` (`base.py:604`). |
| F | `post_tok_tokens_input` | `params.py:554 apply_post_tokenization` (truncate / pad / length check) → `base.py:1358 process_for_engine_async` → `:1002 _process_tokens_async` → `tokens_input` (`:1021`). Image requests branch to `_process_multimodal_async` here (`:1012`). |
| G | `adjust_request` | `online_renderer.py:823-853`. A second parser instance runs `.adjust_request(request)`. |
| H | `reasoning_end_validate_ids` | `serving.py:334` takes the ids. `_engine_reasoning_kwargs` (`:236`) → `resolve_reasoning_ended` (`chat_completion/protocol.py:602`, reverse scan from the end). `engine_client.generate` (`serving.py:389`) → `v1/engine/async_llm.py:698` → `:474 input_processor.process_inputs` (`v1/engine/input_processor.py:373`) → `_validate_model_input` (`:604`) → `renderer.validate_token_ids` (`base.py:420`, `max()`/`min()` over all ids). |
| X | `x_*` (outside the renderer) | `async_llm.py:545 output_processor.add_request` → `v1/engine/detokenizer.py:195 DecodeStream(ids=prompt_token_ids)` (API server). The msgpack encode of `prompt_token_ids` goes to EngineCore. The EngineCore side decodes it and computes sha256 block hashes (`v1/core/kv_cache_utils.py:695,941`; `prefix_caching_hash_algo` defaults to `sha256`, `config/cache.py:143`). |

The rendered prompt text also stays on the engine input (`base.py:1024`, `engine_input["prompt"]`).

## 2. Stage timings (measured, spark-two CPU, ms per request, median of 5 reps over 3 consecutive requests)

Conversation shape: one system message with 12 tool schemas, one user task, then agent steps. Each step is
an assistant message (reasoning, sometimes content, 1-3 tool calls) followed by its tool result(s): code,
logs, JSON, prose, CJK/emoji text, and occasional literal special-token strings. Each request adds one step.
Thinking is on, effort is high, and tools are present (so drop_thinking is off), which matches the agent
setup. The column headers are the actual prompt lengths at the last measured request: 55,457 / 158,582 /
304,181 / 506,047 tokens, with 60 / 185 / 347 / 578 messages and 0.20 / 0.53 / 1.01 / 1.67 MB of JSON body.

| stage | ~50K | ~150K | ~300K | ~500K |
|---|---:|---:|---:|---:|
| A http_json_pydantic | 1.82 | 4.45 | 7.93 | 13.56 |
| B params_parser_tools | 0.57 | 0.59 | 0.62 | 0.62 |
| C parse_chat_messages | 0.59 | 0.82 | 1.39 | 2.48 |
| D render_apply_chat_template (total) | 1.81 | 5.12 | 10.68 | 18.34 |
| D.1 normalize (deepcopy) | 1.11 | 2.73 | 7.01 | 13.26 |
| D.2 merge_tool_messages (deepcopy) | 0.58 | 2.57 | 2.43 | 7.23 |
| D.3 sort_tool_results | 0.03 | 0.05 | 0.11 | 0.18 |
| D.4 render_message loop | 1.14 | 2.78 | 6.14 | 11.52 |
| **E tokenize (HF `__call__`, base.py:638)** | **178.36** | **497.48** | **906.76** | **1427.71** |
| E' raw `backend.encode` (same string) | 173.88 | 507.74 | 930.13 | 1471.19 |
| F post_tok + tokens_input | 0.01 | 0.02 | 0.02 | 0.03 |
| G adjust_request | 0.42 | 0.45 | 4.87 | 5.08 |
| H reasoning_end + validate_token_ids | 0.93 | 3.26 | 10.02 | 16.78 |
| X DecodeStream prime (API server) | 0.80 | 2.47 | 4.79 | 8.18 |
| X msgpack encode ids (API server) | 0.36 | 0.88 | 1.90 | 3.44 |
| **API-server total (A-H + X API)** | **185.7** | **515.5** | **949.0** | **1496.2** |
| tokenize share of total | 96.1% | 96.5% | 95.6% | 95.4% |
| X msgpack decode (EngineCore) | 1.19 | 4.48 | 9.61 | 16.23 |
| X sha256 block hashes (EngineCore) | 1.11 | 3.60 | 7.39 | 13.06 |
| **Incremental prototype (replaces E)** | **10.00** | **11.36** | **17.71** | **18.90** |
| API-server total with the prototype | 17.3 | 29.4 | 59.9 | 87.4 |
| **Saving per step** | **168** | **486** | **889** | **1409** |

Notes:
- D.1 to D.4 come from a separate decomposition run on the same inputs (1 sample per request), so they
  can add up to more than D.
- The minimum tokenize times were 158 / 471 / 888 / 1416 ms. A first run with slightly larger
  conversations (60K/172K/342K/568K) measured 198 / 539 / 1017 / 1660 ms (`results/profile_run1.json`).
- A prototype hit adds 1K to 4.5K new tokens. Its cost is mostly the suffix tokenize (about 2.8 µs/token)
  plus copying `prev_ids[:k]` (about 3 ms at 500K).

**Core type and content** (`tok_bench.py`, the 1.09 MB / 335,520-token synthetic prompt)

| | median | min | µs/token |
|---|---:|---:|---:|
| unpinned | 973 ms | 968 ms | 2.90 |
| X925 big core (pinned) | 936 ms | 935 ms | 2.79 |
| A725 little core (pinned) | 1161 ms | 1155 ms | 3.46 |

- The live API server is not pinned (affinity 0-19), so either core type can apply.
- `pre_tokenize_str` alone takes about 900 ms. The Split-regex pre-tokenizer, not BPE, is the cost.
- By content type on the X925 core, cost per token is flat for Latin-script text:

| content | µs/token | chars/token |
|---|---:|---:|
| prose | 2.73 | 4.9 |
| code | 2.52 | 3.5 |
| logs | 2.77 | 2.4 |
| JSON | 2.77 | 3.2 |
| tool-result mix | 2.62 | 2.9 |
| CJK | 0.89 | 1.05 |

  The synthetic numbers are therefore unlikely to overstate real agent text by much.

**GIL and concurrency.**
- The server path (HF `__call__`, which uses `encode_batch`) releases the GIL. At 335K tokens the asyncio
  thread stalled for at most 36 ms, which is the Python list build.
- `Tokenizer.encode` (single) holds the GIL for the whole 891 ms. **The overlay must use the `__call__` /
  `encode_batch` path, not `.encode`.**
- The executor has a single worker, so concurrent long requests queue behind each other's ~1 s
  tokenize. A 4-session burst at 300K waits up to about 3 s. The cache removes most of that.

## 3. Where the time goes, compared with the audit

- **Slopes.** Tokenize grows about 277 ms per 100K tokens (55K to 506K). The whole API-server path grows
  about 290 ms/100K. The audit measured 0.35 s/100K first-token growth, so about 80% of it is this one
  call. The remaining ~0.06 s/100K fits the EngineCore-side msgpack decode and sha256 hashing (about
  6 ms/100K together), plus scheduling and the GPU prefix-cache lookup.
- **Rendering is cheap.** Chat template plus the two deepcopies is about 3.6 ms/100K, and JSON + pydantic
  about 2.7 ms/100K. Rendering is not worth caching now. If wanted later, the cheapest real win is dropping
  the double deepcopy (`deepseek_v41.py:23` plus `deepseek_v41_encoding.py:83`, about 20 ms at 500K).
  That is a different change with its own equivalence argument.
- **Estimated daily saving.** The audit's 85-100 s/day at 0.35-0.39 s/step implies about 250 steps/day
  near 300K. At the measured 0.89 s/step that is about **220 s/day**, more at 500K. This rests on the
  audit's step mix and has not been measured live, because port 8000 was off limits. The overlay should log
  per-request `t_full_ms` and `t_incr_ms` so this can be confirmed.

## 4. Which boundaries are provably safe

**Tokenizer facts** (`inspect_tok.py`, `~/models/dsv41/uncensored-fp8/tokenizer.json` sha `c90dfa01249db1be`,
identical on spark-one):
- normalizer: `Sequence([])`, the identity.
- pre_tokenizer: `Sequence[Split(\p{N}{1,3}), Split(CJK), Split(big regex), ByteLevel(add_prefix_space=false, use_regex=false)]`.
  All Isolated, none position-dependent.
- model: BPE, 128,000 vocab, 127,741 merges, no byte_fallback, no dropout.
- post_processor: `ByteLevel(trim_offsets=false)`, which does not change ids. There is no template, and
  chat passes `add_special_tokens=False`.
- 1,283 added tokens, all with `lstrip = rstrip = single_word = false`:
  - 1,229 special, not normalized (BOS id 0, EOS id 1, ...);
  - 47 normalized (`<think>` 128821, `</think>` 128822, `｜DSML｜` 128825, `<｜User｜>` 128803,
    `<｜Assistant｜>` 128804, `<｜System｜>` 128799, `<｜latest_reminder｜>` 128828, task tokens);
  - 7 other non-normalized (`<｜deepseek_image｜>` 129264, ...).
- The longest added token is 36 characters. `<tool_result>` and `</tool_result>` are **not** added tokens.

**Structural property, checked exhaustively** (`boundary_check.py` part 1):
- Over all 1,283 added-token strings, no non-empty proper suffix of one equals a prefix of another, and
  none contains another.
- The single exception is `｜DSML｜`, which overlaps itself: in `｜DSML｜DSML｜` the trailing `｜` of the
  first occurrence starts a second occurrence.

**Claim.** Let S be any string and T any added-token string other than `｜DSML｜`. Let [s, e) be any literal
occurrence of T in S. Then `enc(S) == enc(S[:s]) + enc(S[s:])` and `enc(S) == enc(S[:e]) + enc(S[e:])`,
where `enc` is `add_special_tokens=False` without hitting the truncation length.

**Proof sketch.**
1. **The occurrence is always matched.** The AddedVocabulary splits the raw string on non-normalized
   tokens, then normalizes the remaining pieces (identity here) and splits them on normalized tokens. Both
   passes use leftmost-longest, non-overlapping Aho-Corasick. An occurrence of T can be skipped only if a
   match starting earlier overlaps it, or a match contains it. Either needs a suffix/prefix overlap or a
   containment between T and another added token, and the check shows none exists. A pass-1 match cannot
   cut into a pass-2 token for the same reason. So [s, e) is always a whole added-token piece.
2. **The prefix splits the same way on its own.** A leftmost-longest scan of S[:e] reports the same matches
   as the scan of S up to e. At each step, S's choice ends at or before e, so it is still available in the
   smaller candidate set. No earlier-starting candidate can exist there, and none can be longer.
3. **The suffix splits the same way on its own.** The non-overlapping scan restarts in its start state at
   each match end, so S[e:] gives the same matches as S does after e.
4. **The rest is per piece.** The normalizer (identity), the pre-tokenizers (regex Splits plus ByteLevel
   without a prefix space) and BPE all work on each piece or word on its own. The post-processor does not
   change ids. So the id sequence is the concatenation over pieces. The same argument holds at s.

**Empirical check** (`boundary_check.py`, `results/boundary_check.json`):
- 4,000 adversarial random strings: special tokens, partial tokens such as `<｜Use` + `r｜>`, `<thi` +
  `nk>`, `｜>`, `</`, whitespace runs, digits, CJK, emoji and `<tool_result>`.
- **62,445 cuts at non-DSML added-token boundaries: 0 failures.**
- Cuts that are only `｜DSML｜` boundaries: 4,151 cuts, **3 failures**, all inside `｜DSML｜DSML｜` chains.
  This is why `｜DSML｜` is excluded.
- Negative controls (expected to fail):

| cut point | failure rate |
|---|---:|
| after a space | 74% (8,022/10,810) |
| after `\n` | 29% (2,694/9,312) |
| after `\n\n` | 4.6% (196/4,270) |
| before `<tool_result>` | 20% (413/2,034) |
| random position | 79% (3,166/4,000) |

- HF `__call__(truncation=True, max_length=1048577, add_special_tokens=False)` matched the raw backend on
  every sample (0 mismatches).

**Message serialisation** (`deepseek_v41_encoding.py`). Every message boundary sits next to a safe literal:
- user and tool-result messages start with `<｜User｜>` (`:381`);
- after a user, developer or mid-conversation system message the encoder appends `<｜Assistant｜>`
  followed by `<think>` or `</think>` (`:495-506`);
- the assistant thinking part ends with `</think>` (`:447`);
- assistant messages end with `<｜end▁of▁sentence｜>` (`:38`), except `wo_eos` (prefill continuation);
- tool calls use `｜DSML｜`, which is not used as a cut.

So the last safe cut before the new suffix is at most one message back. The message boundaries themselves
(the `\n\n` joins at `:404`, and `<tool_result>` at `:46`) are **not** safe.

**When the new prompt is a plain extension of the old one**
- **Tools present, which is the agent case.** drop_thinking is forced off (`:556-558`). Each message's
  rendering depends only on itself and on whether a message follows. The old prompt ends with
  `<｜Assistant｜><think>`, and the new one continues it. Extension in 39/39 steps.
- **No tools, thinking on, drop_thinking on** (the default). This is **not** an extension (0/39). The old
  last user message's `<think>` becomes `</think>`, and earlier assistants lose their reasoning once a later
  user turn exists (`:445-450`, `:503-506`, `:511-530`). The longest-common-prefix cut still reuses about
  93% of the ids, with 39/39 identical.
- **History edits** (compaction): 39/39 identical, about 88% reuse.

**End-to-end results.**
- 7 chat modes, 40 steps each, with tricky strings at message edges: 273/273 incremental results identical.
  Modes: tools+thinking, tools+effort none, no tools with drop, no tools with keep, mid-conversation
  system messages, plain chat, and history edits.
- The 12 profiled requests at 50K to 506K were also identical.

## 5. What this means for the overlay (next step)

- **Hook.** Override `DeepseekV4Renderer._tokenize_prompt` in an overlay copy of `renderers/deepseek_v4.py`.
  `base.py:155` binds `_tokenize_prompt_async = make_async(self._tokenize_prompt)` when the instance is
  built, so a method defined on the class is picked up and `base.py` stays untouched. Base shas: `deepseek_v4.py`
  `f7175f99...`, `base.py` `ec2fe845...`. The installer should refuse on any other sha.
- **Take the cache path only when all of these hold** (otherwise use the stock call):
  - the env flag is on (default off);
  - `prompt["prompt"]` is a `str`;
  - there is no `multi_modal_data` or `multi_modal_uuids`, so image requests fall back;
  - offsets are not wanted (`_wants_offsets` is False);
  - `params.truncate_prompt_tokens is None`;
  - the composed length is `< max_length` (= `max_input_tokens + 1`) from `get_encode_kwargs()`. If it is
    `>= max_length`, fall back so truncation and length errors behave exactly as stock.
- **Algorithm (as prototyped in `profile_stages.py:incremental`)**
  1. Pick the cached entry P with the longest common prefix L.
  2. `c = max(S.rfind(lit, 0, L) + len(lit))` over the five literals.
  3. `tail = enc(P[c:])`, `k = len(ids_P) - len(tail)`, then self-check `ids_P[k:] == tail`.
  4. `ids = ids_P[:k] + enc(S[c:])`, with `enc` being the **same HF `__call__` and kwargs** the server
     uses, so the GIL is released and the flags match.
  5. Fall back to the full tokenize if c is 0 or less, or the self-check fails.
- **Guard.** Fully tokenize and compare on the first N hits (e.g. 32) and then on 1 in K (e.g. 64) at
  random. On a mismatch, log it, disable the cache for the process, and serve the full result. A verified
  hit costs about full + 2%.
- **Memory.** Per conversation: the prompt string at 2 B/char (UCS-2, because of the `｜`/`▁`/CJK
  characters; 4 B/char if there is emoji) plus about 40 B/token for the list of int objects. That is
  roughly 3.3-6.7 MB + 20 MB at 500K. Successive entries share int objects, and a hit replaces its
  predecessor. Use an LRU bounded by bytes (e.g. 256 MB). spark-one had 19 GB MemAvailable at 07:31.
- **Thread safety.** Use a lock around the entry map only, and tokenize outside the lock. The executor
  has 1 worker today, but `renderer_num_workers` is configurable.
- **Render cache: not recommended** in the first version (section 3).

## 6. Risks and caveats

- **Synthetic, not live.** The cost per token is flat across Latin content types, but real sessions with
  much CJK text would cost less per token (about 0.9 µs). Confirm with the overlay's timing log.
- **spark-two is not idle.** Its TP-rank worker had 2 busy threads, on cores 5 and 15. Runs were
  unpinned at nice 19; pinned runs bracket the result at 0.94-1.16 s for 335K tokens.
- **Tokenizer upgrades.** The proof depends on the added vocabulary and the pipeline: identity normalizer,
  no lstrip/rstrip, Split pre-tokenizers, no BOS template. A `tokenizer.json` or `tokenizers` change should
  disable the cache at startup. Have the overlay re-run the overlap/containment check (under 1 s) at load
  and compare `tokenizer.json` against sha `c90dfa01...`.
- **The `｜DSML｜` self-overlap is real.** It must never be used as a cut. The other 1,282 added tokens are
  safe.
- **`wo_eos` / `continue_final_message`.** The last assistant message has no EOS, but the LCP cut still
  finds the previous safe literal, and the prototype handles it like any other divergence.
- **Behaviour on mismatch.** `prompt_token_ids` identical to stock means engine and model behaviour cannot
  change. Only latency and API-server memory change.
- **Service restart (not caused by this work).** The live service was restarted at about 07:31 spark time,
  just before this work began, and the API server PID changed again within the first minutes. Only file reads and
  `/proc` reads were made on spark-one.

## 7. Reproduce, files, base shas

On the Mac under `~/Documents/dsv41-kits/incr-tok/`, mirrored to spark-two `~/dsv41-stage/incr-tok/`:
- `synth.py`: synthetic conversation generator.
- `profile_stages.py`: stage replay plus the incremental prototype. Env: `SIZES`, `REPS`, `STEPS`,
  `DEADLINE_S`, `DUMP_PROMPT`.
- `boundary_check.py`: structure check, random split test, negative controls and 7-mode end-to-end test.
- `tok_bench.py`: core pinning, content mix (`CONTENT_MIX=1`) and GIL test (`GIL_TEST=1`).
- `inspect_tok.py`: tokenizer config and added-token flags.
- `results/`: `profile_run2.json` (the table above), `profile_run1.json`, `boundary_check.json` and logs.

Command used (each run finished in under 90 s; the guards abort if MemAvailable < 8,000,000 kB or the
deadline passes):
`cd ~/dsv41-stage/incr-tok && CUDA_VISIBLE_DEVICES= nice -n 19 ~/venvs/vllm032/bin/python profile_stages.py out.json`

Base sha256 (identical on spark-one and spark-two):
```
f7175f9980550e5ccfcc94fcebfdeee113eee66efc7cd1ec539d5b2af1f0d8cd  renderers/deepseek_v4.py
ec2fe84538bb30ad7ed1e10833081c8c389e6a2f8fda1dcef50005b9db75a1a7  renderers/base.py
a8f90b4d1d41b2194557992a12105f65bf720d743384e18ee01c2870a75a7c72  renderers/online_renderer.py
5113461cd308266727d9ced82900cbe65ae50245a2f3cb16224f6689574570a9  renderers/params.py
fff82623f5676021bd93766e8ec7106ff0384181b83ab4f2aa3a492ffa0b3f1f  tokenizers/deepseek_v41.py
477f22ac250b43c12bd19bb3c93f934d6637b06410b8cb7ccba0862d1049dee8  tokenizers/deepseek_v41_encoding.py
4c2712de92e8fec91b9d7f3b5b162bd1f0683e7fc21c16b20188a505fc6c6fea  tokenizers/hf.py
5e619c0a6866758a247c76e7b8c472f6e008ddbc720f7efe47cc6776cb986d4f  entrypoints/openai/chat_completion/serving.py
b9e23b5714b80fbccbff3149541f690bc3ff1d41b064f27bfa7fe85b2bc15476  v1/engine/detokenizer.py
97f66659814177d3f7b72b959f09201cbdd1794de21ec1542fe8970f18a1a2b6  v1/engine/input_processor.py
```
