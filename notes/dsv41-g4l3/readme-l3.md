
## sPTC layer 3 (goal window 4, Layer3 step, 2026-10-27 14:08-15:30 UTC): the DeepSeek-V4.1 tool-parser patch is installed on 8/8 and `DSV41_TOOL_STREAM_NEWLINE=1` is PROMOTED as a launcher default (launcher 31d04d8f -> **95f08eff**). It is output-neutral: exact.py a8 and the pinned G1 digest are unchanged, and streamed tool arguments equal the non-streamed ones for the same tokens in 28/28 cases. Tool-call programs now stream line by line: the largest single argument delta drops from 78% to 17% of the program (median). Layer-3 A/B (DSH plugin c981260, 20 code-mode tasks x 2 rounds per arm, thinking off, JIT off): 238/238 speculative sub-calls claimed, 0 wasted, no extra server tokens. run_code time drops 26%, but the agent's own stream is ~4% slower while sub-calls overlap it. Turn p50 9.00 -> 8.48 s, p90 17.63 -> 17.07 s, paired median -0.14 s (on faster in 25/40), mean flat. Recommendation for the Mac app: layer 3 ON as a small win, at its next plugin rebuild

Scope: the user-approved (2026-10-26) layer 3 of sPTC for DeepSeek Harness. The offline build was ready: `512S2:~/dsh-sptc/LAYER3_READY`, plugin `~/dsh-sptc/sptc-plugin` main **c981260**, server patch staged in `~/dsv41-stage/layer3/` on 8/8. Steps:
1. install the parser patch with the flag OFF, relaunch, exact.py;
2. relaunch with the flag ON, check stream integrity, exact.py and G1;
3. run the layer-3 A/B from 512S2;
4. keep the flag as a launcher default if integrity and exactness pass;
5. write this section.

Trim rule (user, 08:35 UTC): the patch only touches the API server's parser, not the engine. So there were no decode or long-prompt suites. Each instance ran exact.py first and last, the integrity case set, and G1 x2 where the step needed it. There was no GSM8K, longctx or 1M run, because the model's computation does not change. Every launch used `DSV41_GPU_CLOCK_MHZ=2000`, and the SM clock was 1969-1989 MHz on 8/8 at every check. Every integrity request was the only request on the server: each one ran on an idle server and added exactly 1 to `request_success_total`.

Files: spark-one `~/dsv41-g4l3/` (`scripts/`, `cases/l3cases.jsonl`, `runs/<label>/`). Mac scratchpad `g4l3/`:
- `scripts/`: `l3sw.sh` (stopped window), `l3ab.sh` (instance driver), `phase.sh`, `l3phase.sh`, `l3cases.py`, `l3integrity.py`, `l3ref.py`, `l3cmp.py`, `l3repeat.py`, `promote_l3.sh` and `lines-L3.txt`;
- results: `ab.log`, `sw.log`, `runs/` and `ab/<label>/`.

A/B outputs on 512S2: `~/dsh-sptc/ab/g4l3-20261027/` (`report.md`, `report.json`, `runs.jsonl`, per-run captures) and `~/dsh-sptc/ab/g4l3-20261027.log`.

### 1. State check (14:08-14:31 UTC)
- **Live:** g4c-FINAL, launcher **31d04d8f** on 8/8, healthy since 14:01:06, 0 requests in flight.
- **Site:** g4d f99d02f68320 (flags off), g4p aef7c824bb9a, fload ca05f12513a1. The parser files were the deployed ones on 8/8 (`deepseek_v4.py` 709f328f, `parser_engine.py` 62b741a0), with no `.bak-layer3`. The stage was identical on 8/8 (digest bb8f901117da).
- **Nothing to restore:** the task text again named fd1e809e as production, but the Loader phase made 31d04d8f production, and no candidate was installed or running.
- **The READY marker exists** (written 2026-10-26). Its orchestrator update says to use plugin main c981260, the commit with the review fixes.

### 2. The patch and how it was checked
**Patch** (staged `~/dsv41-stage/layer3`, MANIFEST for vLLM 0.32.0; site-packages `vllm/`):

| file | deployed -> patched (sha256) | what the flag changes |
|---|---|---|
| `parser/deepseek_v4.py` | 709f328fe3b0711c98cc76f572bee14c6342efaaf2160f010517d3fd54794cee -> **3334c1fa88c30f1b1c8fadb445332f264d8415641c44afc8f6478ee7bd4f83db** | argument deltas flush at `>` **and at newline** (`arg_structural_chars`) |
| `parser/engine/parser_engine.py` | 62b741a0557beff5e2d7ed630e907c0203102ab8fda26b06190ded0563f0e120 -> **6e3eeaba2a38b3ec9d9e42437d43246f77ae9d21c7b9d2540a7c08a5b461171e** | an open string parameter streams as incrementally escaped JSON. It falls back to the stock full conversion at a DSML mark, or past 8,192 characters outside an open string (`DSV41_TOOL_STREAM_NEWLINE_MAX_CHARS`). |

- The flag is read once at import: `DSV41_TOOL_STREAM_NEWLINE` must be exactly `1` in the API server's environment. With it unset, the patched files take the stock paths.
- Non-streaming responses use `parse()`, which the flag does not touch.

**Integrity probe** (`l3integrity.py` 66841a63 -> token-id version; `l3cases.py`). The case set has **32 requests**:
- **27 recorded:** main-agent requests from the 2026-10-26 DSH check runs (`512S2:~/dsh-sptc/check`), first turns and later turns with tool results:
  - A thinking 11, B no-thinking 8, D/E stress 4 (all `/v1/chat/completions`);
  - C 4 on the Messages route `/v1/messages`;
  - the OpenAI-route prompts are 5.8-8.8K tokens.
- **5 new:**
  - an HTML table program with `<`/`>` on every line, unicode (Reykjavík, 東京) and escapes;
  - two parallel `get_weather` calls;
  - `write_file` with an integer parameter and thinking on;
  - tools attached but no call;
  - a plain chat without tools.

How each case runs:
- A non-streaming request, then a streaming one, each on an idle server, each with its own `cache_salt` (cold prefill), and with `return_token_ids` on the OpenAI route.
- Three verdicts:
  - **direct:** live non-stream vs live stream: content, reasoning, tool names, arguments (string-exact), finish reason and completion tokens. It is valid only when both requests generated identical token ids.
  - **ref** (`l3ref.py`, CPU, the installed vLLM): the stream's OWN token ids, decoded with the served tokenizer, go through the installed parser's non-streaming path exactly as `chat_completion_full_generator` calls it (`set_prompt_token_ids` + `parse(text, request, enable_auto_tools=True)`). The result is compared with what the stream delivered. The same reproduction applied to the live non-stream token ids gave back the live non-stream response in every case (28/28 in each instance), so the reference is faithful.
  - **cross-instance** (`l3cmp.py`): flag off vs flag on, for cases with identical token ids in both instances.
- Why the ref verdict is needed: the two live requests of one case often generate different tokens (section 6). Without it, only the cases with matching tokens could be checked.

### 3. Install (stopped window 14:31:14-14:31:40 UTC, `l3sw.sh`)
- Stopped with 0 requests in flight. The pre-state was as in section 1 on 8/8.
- `bash ~/dsv41-stage/layer3/install.sh` on 8/8: rc 0 everywhere. It is sha-guarded and keeps backups `<file>.bak-layer3`.
- **After the install, on 8/8:** `deepseek_v4.py` 3334c1fa, `parser_engine.py` 6e3eeaba, backups 709f328f / 62b741a0. `status.sh` reports patched.
- **Import check** (spark-one, CPU):
  - flag unset: `arg_structural_chars = {'>'}`, `_L3_NEWLINE False`;
  - `DSV41_TOOL_STREAM_NEWLINE=1`: `{'\n', '>'}`, `_L3_NEWLINE True`, `_L3_INCREMENTAL True`.
- The supervisor's preflight manifest does not hash the parser files, so it stayed 18b3d8175e12.

### 4. Instance g4l3-F0: patch installed, flag unset (pinned `av-pin-k5-g3FU.json` eeac4f97; 14:31:48-14:41:30)
- Healthy at +131 s (fast loader).
- **exact.py a8c3394af9d2** with the pin's counters 909/467/2335/1373 and 348/233/171/90/67 (pin hit), and again after the suite (#2). 0 ERROR/Traceback on 8/8.
- **Cases:** all 64 requests answered normally.
  - direct PASS for **17/17** OpenAI cases with identical tokens and 2/2 Messages cases with identical outputs; 0 FAIL;
  - ref(stream) **28/28** PASS, ref(non-stream) 28/28.
- **Same as stock with the flag unset.** B0 is the same 32 cases on the stock parser (live g4c-FINAL, 14:22-14:30, before the install; first probe version without token ids).
  - 17 cases have identical stream text in B0 and F0, and 13 of them contain tool calls. In 12 of those 13, the argument chunking (the sequence of delta lengths) is identical. The odd one has 6 pieces against 5, i.e. two `>` flushes fell into one engine step.
  - Content and reasoning chunking differs between B0 and F0 even in a no-tools chat. B0 is unpinned and F0 pinned, so speculative decoding groups tokens into steps differently. That is why chunk-level comparisons are only made between F0 and F1 (same pin).

### 5. Instance g4l3-F1: `DSV41_TOOL_STREAM_NEWLINE=1` (pinned; 14:41:55-14:52:40)
- Healthy at +127 s. The API server's environment carried the flag.
- **exact.py a8c3394af9d2** with the pin's counters (pin hit), and again after the suite (#2).
- **G1 x2 101.66 tok/s** (95.95 / 76.50 / 132.53), **digest 9a18f802d930f6e7**, the pin's reference (as g4c-O1 and g4d2-C1). Both repetitions were identical, with 6 completions for 6 sent.
- 0 ERROR/Traceback on 8/8.
- **Stream integrity:**
  - ref(stream) **28/28 PASS**: for the same tokens, the streamed tool arguments concatenate to exactly the non-streamed arguments, and the streamed content and reasoning equal the non-streamed ones. This includes the 13 cases whose two live requests generated different tokens.
  - ref(non-stream) 28/28. direct PASS **15/15** OpenAI cases + **2/2** Messages; 0 FAIL.
  - Every streamed argument concatenation parses as JSON: 25 tool calls on both routes.
  - On the Messages route (no token ids), the `input_json_delta` concatenation equals the non-streamed `tool_use.input` in both cases whose two requests produced the same output.
- **F0 vs F1** (13 OpenAI cases with identical token ids in both instances, non-stream and stream, plus 2 Messages cases with identical outputs):
  - non-streaming responses identical **15/15**;
  - content and reasoning delta sequences identical **chunk by chunk, 15/15**;
  - argument concatenations identical 15/15.
  - Only the chunking of tool arguments differs. Arguments without a newline (the `get_weather` calls) chunk exactly as before.
- **How the arguments now arrive** (25 streamed tool calls; same-token subset in brackets; times are seconds before the last argument delta):

  | median per tool call | F0 (flag unset = stock) | F1 (flag) |
  |---|---|---|
  | argument deltas | 6 [6] | 22 [19] |
  | largest single delta, share of the program | 0.78 [0.70] | **0.17** [0.17] |
  | mean lead of an argument character | 0.48 s [0.47] | **1.58 s** [1.46] |
  | lead of the 50% mark | 0.49 s [0.49] | **1.68 s** [1.37] |

  Example (A:T05, same tokens): 4 deltas with 95% of the program in one burst, against 27 deltas and 17%.

### 6. Finding (pre-existing, not caused by this step): long-prompt greedy requests are not reproducible run to run
- **Back-to-back pairs differ.** Two identical greedy requests sent back to back to the same live instance (cold prefill each) generated different tokens:
  - 12/28 OpenAI cases on the **stock** instance B0, before anything was installed;
  - 11/28 in F0 and 13/28 in F1;
  - the Messages route 1-2/4.
  - Which cases differ changes between F0 and F1, although both had the same pin and the same request history.
  - Every case that differs has a 5.8-8.8K-token prompt. The new short-prompt cases (20-420 tokens) never differed.
- **Repeat probe** (`l3repeat.py`, on F1 after its suite; 4 identical non-streaming requests per mode):

  | case (prompt) | cold: fresh `cache_salt` each time, distinct outputs / first divergence | warm: one salt, prefix-cache hits |
  |---|---|---|
  | B:T16 (5.9K) | 3 of 4 / token 186 | 4 of 4 / token 155 (cached 5,888) |
  | B:T02 (5.9K) | 3 of 4 / tokens 218-225 | 2 of 4 / token 162 |
  | A:T01 (6.1K, thinking) | 2 of 4 / token 429 | 2 of 4: every cached run differs from the cold one at **token 0** (cached 6,080) |
  | N:html-table (416) | **1 of 4** (identical) | 2 of 4: cached runs differ from the cold one at token 41 (cached 384) |

- **What this means:**
  - Cold short prompts are reproducible.
  - Cold prompts above ~2K tokens are not, even with nothing else on the server.
  - A prefix-cache hit can change the output against a cold prefill at any length.
  - exact.py (268 prompt tokens) and G1 (103-134-token prompts + 1,024 output tokens) never reach 2K tokens of context. They held on every launch here, so the exactness gate as defined cannot see this.
- **Suspects, not investigated in this step:**
  - the long-context path above the sparse-attention top-k size of 2,048 (indexer top-k selection or tie-breaking that depends on physical KV placement);
  - cached-prefix vs cold-prefill numerics.
- **Effect on this step:** none on the verdicts. The integrity check compares the stream with the non-streaming parse of the same tokens, so it does not depend on reproducibility. This is an open item for the exactness program: an exactness gate for contexts above 2K tokens is missing.

### 7. Promotion (14:56 UTC)
- **Integrity and exactness passed**, so the flag became a launcher default (`scripts/promote_l3.sh l3 scripts/lines-L3.txt`): `~/dsv41-tp8.sh` 31d04d8f -> **95f08eff** on 8/8. The backup `~/dsv41-tp8.sh.bak-g4-l3` is 31d04d8f on 8/8. One line, after `DSV41_FAST_LOAD`:
  ```sh
  export DSV41_TOOL_STREAM_NEWLINE=${DSV41_TOOL_STREAM_NEWLINE:-1}  # g4l3 sPTC layer 3 2026-10-27: ... 0 = stock flushing at ">"
  ```
- The supervisor is unchanged (7bea1cf3). Its preflight manifest, which hashes the launcher, is now **40eb7abf08c7**.

### 8. Production instance g4l3-PROD (launcher 95f08eff defaults, AV unpinned; 14:58-15:21 UTC)
- Supervisor launch with no extra variables (`--expect-launcher 95f08eff`; preflight manifest 40eb7abf08c7 and launcher identical on 8/8). Healthy at 15:01:43 UTC on the first try (+200 s).
- The API server's environment has `DSV41_TOOL_STREAM_NEWLINE=1` from the launcher (no supervisor argument) and `DSV41_FAST_LOAD=1`.
- **exact.py a8c3394af9d2** first (own profile 909/467/2335/1373, per position 348/231/171/91/68, "pin miss" = anchor OK) and again at 15:21 after the A/B (#2).
- **G1 x2 101.37 tok/s** (95.70 / 75.80 / 132.61), digest 69ad872124100141 = the other outcome of the known prose tie (unpinned production lands on either; g4c-FINAL had the same digest, 101.67). Both repetitions identical; 6 completions for 6 sent.
- **Spot integrity check** with the flag coming from the launcher (5 cases: A:T05, B:T06, E:S22, N:html-table, C:T01 on the Messages route): direct PASS 5/5, ref 4/4 (OpenAI route); 17-32 argument deltas per call.
- 0 ERROR/Traceback on 8/8 at start and end. MemAvailable after start 13.2 GB (min of 8), 10.6 GB on spark-one after the A/B. SM clock 1969-1989 MHz on 8/8 at every check.

### 9. Layer-3 A/B (512S2, 15:04:29-15:19:28 UTC, on g4l3-PROD; newline flag on in both arms)
Command (plugin main c981260 through profile `sptc`, which links the main checkout):
```sh
ssh 512S2; source ~/dsh-sptc/env.sh
SPTC_ALLOW_SERVER=1 node ~/dsh-sptc/sptc-plugin/scripts/l3-ab.mjs --rounds 2 --jit 0 --wait-idle-sec 600 --out ~/dsh-sptc/ab/g4l3-20261027
```
- 20 code-mode tasks with `llm_query` sub-calls (T01-T16, T18, T19, S21, S26) x {off, on} x 2 rounds = 80 runs, one at a time, with alternating arm order. Route oai (`/v1/chat/completions` through the LAN API), thinking off, agent temperature 0, layer 3 literal + statements, speculative concurrency 2.
- **JIT pre-launch off in both arms.** That is the Mac app's setting; the harness default is JIT on.
- The server was idle before every run (load 0+0 in 80/80) and after it. 0 failed runs, 0 timeouts.

| arm | runs (failed) | turn p50 s | turn p90 s | turn mean s | turns summed s | run_code summed s | programs | llm_query calls | argument deltas per run_code call (median) |
|---|---|---|---|---|---|---|---|---|---|
| off | 40 (0) | 9.00 | 17.63 | 10.16 | 406.4 | 120.9 | 51 | 248 | 17 |
| on | 40 (0) | **8.48** | **17.07** | 10.32 | 413.0 | **90.0** | 57 | 249 | 17 |

- **Speculation (on):** 238 streaming launches, **238 claimed, 0 wasted**, 0 scopes dropped, 0 wasted tokens. Summed lead before dispatch was 253.6 s (1.07 s per claimed call).
- **Server tokens:** `llm_query` was the same in both arms: 248 / 249 requests, prompt 24,935 / 24,595, completion 12,367 / 13,096, 0 cancelled. In total, off used 657,292 tokens and on used 701,201. All of the difference is agent requests (91 / 97), because the on arm wrote more programs in four runs (below).
- **Paired, all 40 pairs:** on - off turn time has a median of -0.14 s and a mean of +0.16 s; on was faster in 25/40.
  - In 4 pairs the on arm's agent wrote more programs: T05 r1 1 -> 2, T07 r2 1 -> 2, T13 r2 1 -> 2, T19 r2 2 -> 5. They cost +0.8 to +5.2 s.
  - The final answers differ in all 40 pairs. This happens in both arms: long prompts are not reproducible (section 6), and the sub-call outputs vary too. The extra programs are therefore model variance, not something the plugin causes (it claims only byte-identical requests).
- **Paired, the 36 pairs with the same number of programs and steps:**
  - turns summed 357.2 -> 353.9 s (**-0.9%**);
  - paired median **-0.20 s (-2.7%)**, on faster in 25/36;
  - run_code summed 101.7 -> 78.0 s (**-23%**);
  - model/other time per run 7.10 -> 7.66 s (+0.56 s);
  - agent completion tokens per model-second 107.7 -> 102.9 (paired median ratio 0.957).
- **What this means:** layer 3 moves about a quarter of the run_code time into the agent's stream, because the sub-calls start while the program is still being written. The agent's own stream then gets about 4% slower, because the server decodes it together with the speculative sub-calls. The net is a small but consistent gain at the median (about 0.2 s per turn, 2-3%; p50 -5.8%, p90 -3.2%) and a flat mean. The offline estimate was +5% of turn time with the newline patch. The measured gain is about half of that at the median and about zero on the mean.

### 10. Recommendation for the Mac app (DeepSeek Harness desktop, profile `desktop`)
- **Layer 3: ON, as a small win.** Expect about 0.2 s faster typical turns, with no wasted sub-calls and no extra server tokens. Do not expect a lower mean. Keep JIT off.
- **What enabling needs:**
  - The installed Mac plugin is the layer-1 build (cfa2640 + the JIT fix) and does not contain layer 3.
  - Rebuild it from 512S2 main **c981260** with the declared DSH range widened to `>=0.2.1-rc.2`, as the layer-1 desktop packaging did (scratchpad `macinst/scripts/build-desktop.sh`).
  - Set `layer3: true` in the plugin config (or `SPTC_L3=1`), then restart the app with ⌘Q.
  - None of this was done in this step: the Mac app is unchanged.
- The Mac profile runs reasoning `max` (thinking on), while this A/B ran with thinking off. Reasoning adds time to every turn, so the relative gain there will be smaller still.
- **The server side is done either way.** `DSV41_TOOL_STREAM_NEWLINE=1` is the launcher default. Layer 3 needs it, and any streaming client now receives tool programs line by line.

### 11. Live state and rollback
**Live:** instance **g4l3-PROD**, healthy since 15:01:43 UTC.
- Launcher **95f08eff** on 8/8 (= 31d04d8f + the `DSV41_TOOL_STREAM_NEWLINE` line); backup `~/dsv41-tp8.sh.bak-g4-l3` = 31d04d8f.
- Supervisor 7bea1cf3, preflight manifest **40eb7abf08c7**; AV unpinned; clock lock 0-2000 MHz; LAN/tailnet proxy `dsv41-lanproxy.service` active and unchanged.
- **Site files on 8/8:** the parser patch (`vllm/parser/deepseek_v4.py` **3334c1fa**, `vllm/parser/engine/parser_engine.py` **6e3eeaba**; backups `.bak-layer3` = 709f328f / 62b741a0). g4d f99d02f68320 (flags off), g4p aef7c824bb9a and fload ca05f12513a1 are unchanged.
- 8/8 active, 0 ERROR/Traceback, no test or sampler process left. The 512S2 harness and its capture proxies have exited.

**Rollback:**
- **One launch:** `DSV41_TOOL_STREAM_NEWLINE=0` as a supervisor argument restores the stock flushing at `>`. Outputs are the same either way.
- **Permanent:** `cp -p ~/dsv41-tp8.sh.bak-g4-l3 ~/dsv41-tp8.sh` on 8/8 (31d04d8f), then relaunch with `--expect-launcher 31d04d8f`.
- **Site files** (service stopped, after the launcher rollback): `bash ~/dsv41-stage/layer3/uninstall.sh` on 8/8. It restores the `.bak-layer3` files and is sha-checked. Check with `bash ~/dsv41-stage/layer3/status.sh`. With the flag unset, the installed files take the stock paths anyway (section 4).

### 12. Notes for the next steps
- **Open exactness item (section 6):** greedy requests with prompts above ~2K tokens are not reproducible run to run on one instance, and a prefix-cache hit can change an output at any length. exact.py and G1 cannot see this.
  - Suggested gate: 4 cold repeats of three recorded 6K-token tool prompts (`~/dsv41-g4l3/cases/l3cases.jsonl`; `l3repeat.py`) must be token-identical.
  - Then bisect the long-context path: the sparse-attention top-k (2,048) tie-breaking against physical KV placement, and cached vs cold prefill.
- **Layer 3's ceiling** is set by the agent-stream slowdown while sub-calls overlap it.
  - Only speculative concurrency 2 was measured. `SPTC_L3_CONCURRENCY` takes 1-4, and 1 might keep most of the overlap at a smaller slowdown.
  - A server-side priority for the agent request over its own speculative sub-calls would remove the slowdown.
- **Tool clients must keep Codex 07 rule 5:** send `reasoning_content` back on every assistant message in tool loops.
