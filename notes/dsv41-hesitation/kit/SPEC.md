# DSv4.1-Flash-1017: removing unproductive hesitation from thinking. Method spec

Date: 2026-11-03. Author: Claude (workflow step, spec only). Nothing in this file has been run yet.
Revised 2026-11-03 (second revision, before P0/P1): one answer scorer for the whole kit (`tools/answer_score.py`, 0.5); production launcher is now `416e3ace`; collection and evals use top-level `reasoning_effort` `"max"` (what the user runs) with `"high"` as the comparison, verified by `/detokenize` before any run (1.3); AIME 2024 added to D hard; concurrency capped at 6; P0/P1 run as a durable unit (`ops/run_p0_p1.sh`, 6.1).
Revised 2026-11-03 after review (fixes B1-B5): one trace schema (1.5), switch episodes out of every contrast (2.1), one labeler (2.2), the main contrast is now a matched decision contrast with a two-sided sign screen (3.1, 3.2, 3.5, 4, 4.1), and an unmeasured gate fails the verdict (5.3). All of it is tested only offline, on synthetic traces and mocks.
Inputs: `research/literature.md` (papers), `research/stack.md` (serving stack, edit targets, capture limits), `research/requant_sim.{py,out}`, plus read-only checks on spark-one made for this spec (section 0).

Evidence tags:
- **[V]** I checked it this session, in code, a log, a file or a measurement.
- **[Vp]** The number was checked against the paper's full text in `research/literature.md`. I did not re-read the papers for this spec.
- **[A]** It comes from an abstract or a dataset card only.
- **[I]** It is my inference or a design choice.
- **[?]** Unknown. It must be tested before anything relies on it.

Goal, from the user: make DeepSeek-V4.1-Flash-1017 hesitate less in its thinking ("Wait…", "Hmm, maybe…", re-deriving an answer it already has). The checking it does as an agent and on hard maths must stay.
The plan has two stages. First, prototype the change as an opt-in runtime steer. Then bake the winner as a flag-gated rank-1 weight edit at load time. Rollback is a relaunch, and the checkpoint on disk is never touched.

---

## 0. What I found while writing this spec (read before running anything)

1. **The MATH-500 "steered" arms from this morning's window ran unsteered.** [V]
   - `~/dsv41-steer/vectors/` is empty, so the `analytic` extraction never published (`window.log`: `extract.py` died on `ModuleNotFoundError: numpy`).
   - `rank0.log` holds 660 `DSV41_STEER ... unknown vector 'analytic' ignored` warnings.
   - So the arms `a4`, `a8`, `a12` and `neg8` in `~/dsv41-steer/eval/nothink` are **five unsteered greedy runs of the same 120 problems**. Do not read them as a steering result.
   - The kit drops unknown vector names with only a warning. Every eval in this plan therefore starts with a **load canary** (section 6, P0).
2. **That accident measured the run-to-run noise floor.** Five runs, identical config, temperature 0, `thinking=False`, MATH-500 levels 3-5, n=120 [V]:
   - Accuracy ranged from 104 to 108 out of 120 (86.7-90.0%).
   - Two runs disagreed on correctness for **3 to 8 of the 120 problems**. 12/120 problems flipped at least once across the five runs.
   - The completion token count differed on 97-108 of 120 problems. Mean tokens per run ranged from 616 to 645, a 4.5% spread.
   - **Greedy decoding is not deterministic on this stack**, probably from batch-dependent kernels [I]. Every comparison in this plan is therefore paired by problem, uses repeated samples, and has a null arm to compare against. A single base-vs-steer run at n=120 cannot resolve differences under about 4 points.
   - The `rnd8` arm (a random direction at strength 8%, actually steered) scored 102/120 with mean 702 tokens and 7 truncations. That is just outside the null range, and it is the only real steering data point so far [V numbers; I reading].
3. **The thinking-mode eval looks wrong, and must be explained before collecting traces.** `eval/think/base.jsonl` holds 60 MATH-500 L3-5 problems, thinking on, default effort [V]:
   - It got 46/60 right. Non-thinking got 53/60 on the same 60 problems.
   - Median completion was 301 tokens, the mean 794, and 4 completions were under 100 tokens.
   - The model card asks for `max_tokens ≥ 256K` for a "heavy reasoner", so these traces are far shorter than they should be.
   - Possible causes [?]: the `{"thinking": true}` kwarg not switching thinking on, the scorer reading only `content`, or effort not being rendered.
   - P0 settles this by checking `prompt_token_ids` for the `<think>` opener and the `Reasoning Effort:` prefix, and by checking that `reasoning_content` is non-empty.
4. **What I checked in code and files** [V]:
   - `DEFAULT_REASONING_EFFORT = "high"`, and `high` maps to 75. `reasoning_effort` accepts an int from 1 to 100, `"high"` or `"max"`.
   - **Effort plumbing, settled 2026-11-03 [V]:** DSH sends a **top-level** `reasoning_effort: "max"`, and the chat path renders `Reasoning Effort: 100`. With no effort the prompt says 75. `/tokenize` ignores the top-level field, so effort is only ever verified from a chat request's `prompt_token_ids` (`return_token_ids`) passed through `/detokenize`.
   - With the scorer fixed, the 60-problem MATH L3-5 rerun scored 55/60 without thinking, 59/60 at effort 75 and 57/60 at max (mean 606 / 771 / 1,356 tokens). MATH L3-5 sits at the model's ceiling, so hard maths means AIME/HMMT (1.1).
   - The model card recommends `temperature=1.0`, `top_p=1.0`, `max_tokens ≥ 256K`.
   - Prod runs with `--max-num-seqs 16` and `--max-num-batched-tokens 16384`.
   - MATH-500 level counts: L1 43, L2 90, L3 105, L4 128, L5 134.
   - spark-one has 178 GB free on /home and 61 GB on /dev/shm.
   - A relaunch costs **161-167 s of downtime** (`window.log`).
   - `/usr/bin/python3` has no numpy. Kit scripts must run with `~/venvs/vllm032/bin/python` (numpy 2.3.5).
5. **The answer scorer was wrong for compact LaTeX** [V]. The `steer_eval.py` normaliser that four kit files copied scored `16\sqrt3` ≠ `16 \sqrt{3}`, `\frac14` ≠ `\frac{1}{4}`, `4210_5` ≠ `4210_{5}`, a pmatrix with `\ ` spacing ≠ the same pmatrix without, and `36\text{ seconds}` ≠ `36`. Thinking mode writes these forms, so thinking accuracy was understated.
   - Now there is **one scorer**, `tools/answer_score.py`, ported from the fixed `steer-analytic/steer_eval.py` (sha256 `8c99e562…`), plus thousands-separator removal and "empty never matches". `common.py` and `hes_common.py` re-export it; no other file defines a scorer (`tools/test_scorer.py` checks this by AST).
   - **Still unmatched, by design** (`answer_score.KNOWN_UNMATCHED`, tested): tuple vs pmatrix, inequality vs interval, decimal vs fraction, `1/4` vs `\frac14`, `\frac\pi2` (a control word as an unbraced argument), reordered sums. These count as misses. So every eval report lists **every miss with `pred` and `ref`** (`report.misses`, `misses.txt`; collector `summary.json: misses`) for a human audit.

---

## 1. Trace collection

### 1.1 Prompt pools (public data only)

The user's DSH sessions (`~/.dsh/sessions` on this Mac) are private. They must **never** be used as trace, capture, judge or eval text.
- Enforcement: every collection script reads prompts only from `~/dsv41-hesitation/data/*.jsonl` on spark-one. It refuses any path containing `.dsh`. It records the dataset name and SHA-256 of every input file in `runs/<run>/inputs.json`.
- Nothing is read from the Mac's working directory.

Two disjoint splits, fixed before any trace is collected (seed `20261103`):
- **D (direction/dev):** used for traces, labels, captures and the strength sweep.
- **E (eval):** touched only by the final confirmation run. Problems used in this morning's runs (seed 20261103, L3-5, first 120) go to D, never E.

| Bucket | D (direction + dev sweep) | E (held-out confirmation) | Source |
|---|---|---|---|
| Easy | GSM8K **train**, 300 | GSM8K **test**, 300 | `openai/gsm8k` |
| Medium | MATH **train** L2-4, 300 | MATH-500 L1-4, 300 held out of the 366 (L1 43, L2 90, L3 105, L4 128) | `hendrycks/competition_math` (train); MATH-500 already at `~/dsv41-steer/eval/math500.jsonl` [V] |
| Hard | MATH train L5, 200; AIME 1983-2023, 150; AIME 2024, 30 | MATH-500 L5 (134); AIME 2025 (30); AIME 2026 (30); HMMT Feb 2025 (30); HMMT Feb 2026 (~30) = **~254** | MathArena HF datasets `MathArena/aime_2025`, [`MathArena/aime_2026`](https://huggingface.co/datasets/MathArena/aime_2026/blob/main/README.md), `MathArena/hmmt_feb_2025`, [`MathArena/hmmt_feb_2026`](https://huggingface.co/datasets/MathArena/hmmt_feb_2026/blob/main/README.md) [A: cards exist; row counts not opened] |
| Agent | see 1.2: 120 episodes | see 1.2: 200+ episodes | public benchmarks + our fault-injection suite |

Notes on the hard bucket:
- The hard E set uses 2025-2026 competitions because pre-2025 problems may be in DSv4.1's training data [I; the training cutoff is unknown].
- Whether the 2025 sets are contaminated is also unknown [?]. The 2026 sets are the cleanest available [I].

### 1.2 Agent tasks

The literature has no agentic evaluation of reflection suppression (literature.md, gap 2). So we need our own, and it must contain checks with known ground truth.

1. **Fault-injection tool suite (ours, primary for "productive checks kept").** The tools are simulated, deterministic Python functions: a calculator, a key-value file store, a unit-conversion API, a "run tests" stub that returns canned results, and a lookup table. **Model-written code is never executed** [I: safety].
   - A task takes 3-8 tool calls, for example "compute X using the tools, store the result, then report".
   - In 50% of episodes the harness injects one fault: a wrong numeric result, an error string, stale state, or a silently truncated list. The injection is logged, which gives a ground-truth "a check was needed here" event.
   - About 60 task templates × parameters. D gets 120 episodes, E gets 240 episodes, so ~120 injected faults in E.
2. **BFCL v3 multi-turn** ([Gorilla BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html); state-based grading on simulated APIs; the multi-turn base set is reported as 200 tasks [A]). Use E only. It measures task success.
3. **τ²-bench** ([arXiv:2506.07982](https://arxiv.org/abs/2506.07982); airline, retail and telecom domains with a simulated user [A]). The simulated user is our own model with thinking off and no steer. Use E only, one domain first (retail).
4. SWE-bench Verified is out of scope for now: it needs docker and long rollouts. The one agentic overthinking study (Cuadron et al., [arXiv:2502.08235](https://arxiv.org/abs/2502.08235)) used it and gave the DeepSeek-R1 family the highest overthinking score, 6.70 vs 3.51 average [Vp].

How agent traces are formed [V, stack.md §2]:
- Agent turns go through the chat template with tools.
- `drop_thinking` keeps the reasoning of assistant turns after the last user message, so the model re-reads its own earlier thinking inside a tool loop.
- Every turn is stored with `prompt_token_ids` and the completion `token_ids` (`return_token_ids: true`, as `extract.py` already uses [V]). Spans can then be located by exact token position later.

### 1.3 Sampling settings

- Thinking on: `chat_template_kwargs: {"thinking": true, "enable_thinking": true}` plus a **top-level** `reasoning_effort` E ∈ {`"max"` (100, what the user runs), `"high"` (75, the comparison)}, the same fields DSH sends. Integer efforts (I1, `d_effort`) go in `chat_template_kwargs.reasoning_effort`, since the top-level field takes names only [I].
- **Render check before every run** (`collect_traces.py` preflight, `eval_hesitation.py` canary): for each effort, the exact request body with `max_tokens 1`; `prompt_token_ids` → `/detokenize` must contain `Reasoning Effort: 100` (max) or `75` (high) and end with the `<think>` id. Any failure aborts the run before its first real request. `/tokenize` is never used for this (section 0.4).
- **Trace collection (D):**
  - `temperature 1.0, top_p 1.0`, as the model card recommends [V].
  - k = 2 samples per easy/medium problem and k = 4 per hard problem. Several samples give both correct and incorrect runs per problem, and productive corrections are rare (First Try Matters: wrong→right corrections are mostly <2% of reflections [Vp], [arXiv:2510.08308](https://arxiv.org/abs/2510.08308)).
  - Effort split: 75% at `max`, 25% at `high` (cycle `max,max,max,high`). Hard problems (k = 4) get both.
- `max_tokens` = 65,536 for collection; count every truncation. P0 measures the real thinking lengths and adjusts.
- Pass a `seed` per request for bookkeeping. It does not make runs reproducible on this stack (section 0.2).
- **No steer, no logit bias**, prod server, unsteered requests. Unsteered rows are bit-identical to the unpatched stack [V, kit docstring].
- Concurrency **6 at most** (`MAX_CONC` in the collector and the eval), so a single interactive user keeps reasonable latency under max-num-seqs 16.

### 1.4 Volume and time

- Targets after labeling (section 2): **≥ 600 U spans that have a matched control** (3.1) and **≥ 300 P spans** in D, from **≥ 150 distinct problems per class**. Each class is capped at 5 spans (or pairs) per problem so a few long traces do not dominate. P spans now serve the G5 base rates and the selectivity diagnostic, not the main contrast.
- Rough count: easy 600 traces, medium 600, hard 1,520 (MATH L5 800, AIME 1983-2023 600, AIME 2024 120), agent 120 episodes, so ≈ 2,840 traces.
- Token budget [I]:
  - The planning throughput comes from this morning's non-thinking run: 120 × 645 tokens in 171 s ≈ **450 tok/s aggregate at concurrency 8** [V numbers, I extrapolation].
  - The thinking run reached about 250 tok/s because its tail was one 16K truncation.
  - If thinking traces average 3K (easy/medium) and 15K (hard), that is about 25M tokens, or about **15 h at 450 tok/s**.
  - P0 replaces these guesses with measured lengths. If the budget is too large, cut hard k from 4 to 2 before cutting problems.
- If productive spans fall short of 300, add fault-injected agent episodes and more hard samples. Do not lower the label bar.

### 1.5 Storage (spark-one only)

- Traces go to `~/dsv41-hesitation/traces/<bucket>/<run>.jsonl` on spark-one (dir mode 700). Each record holds problem id, dataset, sample index, effort, prompt_token_ids, completion token_ids, reasoning_content, content, finish_reason, usage and wall time.
- **One record schema for every tool**: `tools/trace_schema.py` (`hes-trace/1`). Canonical fields: `trace_id` (`<run>:<uid>:s<sample>` for collector traces, `<run>:<arm>:<uid>:s<sample>` for eval traces), `problem_id` (the data file's `uid`), `problem` (the text sent, without the boxed suffix), `ref` (the reference answer, which P+/P− need), `prompt_token_ids`, `token_ids`, `reasoning_content`, `content`, `finish_reason`. Both writers (`collect_traces.py`, `eval_hesitation.py --keep-traces`) build records with it. Every reader (segment, label, judge, audit, capture, eval relabel) goes through `normalize()`, which also accepts the old names (`uid`, `answer`, `reasoning`, `finish`). A record without a problem id, or two records with the same trace id, is an error, never a merge. [V: `test_collect.py::t_schema_end_to_end` runs collector → label.py → judge → audit on mock output]
- Labels, spans and captures sit beside them in `~/dsv41-hesitation/{labels,spans,capture,vectors,eval}`.
- **No trace text is copied to this Mac or anywhere else.** Only aggregates leave spark-one: counts, metrics, direction vectors and plots.
- Traces of public problems are not private, but the user asked for local-only storage, and it also keeps any later mix-up with DSH data impossible.

---

## 2. Segmentation and labeling

### 2.1 Segmentation (deterministic, on token ids)

1. Decode the completion `token_ids` incrementally to map each token to its character offset. Reasoning is everything before the `</think>` token.
2. **Step** = text between `\n\n` boundaries, the SEAL/ReflCtrl convention ([arXiv:2504.07986](https://arxiv.org/abs/2504.07986), [arXiv:2512.13979](https://arxiv.org/abs/2512.13979) [Vp]). A step's **boundary token** is the token holding the `\n\n` that opens it.
3. **Hesitation step** = a step whose first 8 words match the hesitation lexicon, case-insensitive, ignoring leading `*`, `-` and quotes:
   - Base list: `wait`, `hmm`, `hold on`, `actually,`, `but wait`, `let me (re)?(check|verify|double-check|recheck|reconsider|recompute|re-?derive)`, `let's (double-)?check`, `is that (right|correct)`, `on second thought`, `alternatively`, `maybe i (should|made)`, `i think i made`, `no,? wait`.
   - DSv4.1-specific openers are mined from D traces: rank step-initial 1-3-grams by frequency. Any opener over 0.5% of steps that a human reviewer marks as hesitation is added. The list is frozen before E runs.
   - `Alternatively` is tagged separately (`switch`). TIP ([arXiv:2501.18585](https://arxiv.org/abs/2501.18585)) shows that penalising thought switches *early* in a thought improved hard tasks: QwQ-32B-Preview AIME24 38.3→44.1 [Vp]. So `switch` steps are not U by default.
   - Switch episodes get rule labels like any other (for reporting), but they enter **no contrast class and no U/P count** unless a tool is run with `--include-switch`: `label.py combine`, `capture_spans.py spans/capture/direction` and the eval counts (`label.trace_counts`, which counts them only as `switch`).
4. **Episode** = a hesitation step plus the following steps, up to (not including) whichever comes first:
   - the next hesitation step,
   - a step that starts a new subgoal (detected by the judge, 2.3, or by `Now`/`Next`/`Then`/`So the answer`/`Therefore` openers),
   - `</think>`,
   - or 8 steps / 1,024 tokens.
5. **Agent unit** = the reasoning between consecutive tool results, tagged with whether the latest tool result had an injected fault.

### 2.2 Rule labels (primary, testable)

There is **one implementation** of these rules: `tools/label.py` with `tools/segment.py`. The direction spans (`capture_spans.py spans`) and the per-trial counts behind gates G4 and G5 (`eval_hesitation.py`) call it (`label_record`, `trace_counts`, `fine_label`), so the labels the human audit validates (2.5) are the labels the gates count. The earlier second labeler in `hes_common.py` was removed.

Value tracking, per trace:
- Extract **candidate values** from each step: `\boxed{…}`; `answer is X`; `= X` at the end of a line or sentence; `so X`; `therefore X`; `x = X` for the asked variable. Normalise them with the `steer_eval.py` normaliser [V] (the same normaliser everywhere, so it is consistent even where it is imperfect).
- `A_before(e)` = the last candidate final answer stated before episode e, if any. `V_e` = the values stated inside e. `F` = the final boxed answer. `ref` = the reference answer.

Maths labels:
- **U (unproductive)** when all of these hold:
  1. `A_before(e)` exists.
  2. Every value in `V_e` equals `A_before(e)`, or `V_e` is empty.
  3. `F == A_before(e)`.
  4. The episode contains no error marker (`mistake|error|incorrect|wrong|should be|I made|typo|miscalculat|sign error`) followed by a changed value.
  - **U-strong** adds a re-derivation test: ROUGE-L between e and the reasoning before e is ≥ 0.4. This is the "re-derives an answer it already had" case.
- **P (productive)** when either holds:
  - (a) **Answer change:** `A_before(e)` exists, `V_e` holds a value ≠ `A_before(e)`, and `F` equals that new value. Split by `ref`: **P+** if `A_before ≠ ref` and `F == ref` (wrong→right); **P−** if right→wrong. P− is harmful. It is kept out of both classes and reported.
  - (b) **Intermediate correction:** an error marker appears in e, followed by a value ≠ the value the episode questions, and the corrected value is used later.
- **A (ambiguous)** = everything else, including hesitation before any candidate answer exists with no value change. A spans are excluded from the contrast.

Agent labels:
- **P-agent** when all hold: the step comes after a tool result; it quotes or uses a token from that result (a number, an error code or a key); and either the next action differs from the action planned before the result (retry, changed arguments, a different tool, or telling the user) or the step explicitly flags the result as wrong. With an injected fault, this is **ground truth P**.
- **U-agent**: a hesitation step with **no new tool result** since the previous decision, whose next action equals the action already planned (same tool and arguments up to normalisation).

### 2.3 Optional judge pass (our own model)

- Same API, `thinking: false`, temperature 0, no steer, JSON output. Thinking is off so the judge does not hesitate itself.
- Input:
  - the problem and `ref`;
  - the candidate values found so far, with their step numbers;
  - the last 1,500 tokens of reasoning before e;
  - e itself;
  - the next 300 tokens.
- Output: `{"label": "U|P|A", "changed_value": bool, "error_found": bool, "evidence_quote": "<exact substring of e>"}`. Reject the output if the quote is not a verbatim substring.
- Bias risk: the model is judging its own traces [I]. Mitigations: rules stay primary; **only spans where the rule label and the judge agree** enter the direction set.

### 2.4 Counterfactual label check (the decisive test)

Keyword labels are unreliable: Zhuang et al. found 93.3% of keyword-detected boundaries did not reproduce the behaviour on regeneration [Vp, literature.md]. So each candidate span also gets a forced-exit test [I; design follows DEER/CGRS confidence probes, [arXiv:2504.15895](https://arxiv.org/abs/2504.15895), [arXiv:2508.05337](https://arxiv.org/abs/2508.05337)]:

- **Exit-before test:** send the prompt plus reasoning up to the boundary token of e, then append `\n</think>\n\n`, and sample 4 short answers (thinking closed, `max_tokens 256`, temperature 1.0).
  - Confirmed U: ≥ 3 of 4 give `F`, so the episode was not needed.
  - Confirmed P: ≥ 3 of 4 give `A_before`, or a wrong value, while `F` is right.
- Prefix caching makes these cheap: the trace prefix is shared, so each probe prefills only a few new tokens [I; prefix caching is on, V].
- Optional, if `prompt_logprobs` works on this stack [?]: score `p(F | prefix + "</think>\n\n\\boxed{")` at each boundary. This yields an "answer already known" curve (Halt Vector, [arXiv:2608.28859](https://arxiv.org/abs/2608.28859) [Vp]; probe early exit, [arXiv:2504.05419](https://arxiv.org/abs/2504.05419) [Vp]).

**Final classes:**
- **U\*** = rule U, judge U, and confirmed U by the exit test. U\* episodes are the positive side of the decision contrast (3.1).
- **P\*** = rule P (P+ or intermediate correction, or P-agent), judge P, and confirmed P (maths) or ground-truth fault (agent). P\* is used for the G5 base rates and as a selectivity diagnostic. It is not the negative class of the main direction (3.1).

### 2.5 Label audit

- **Human audit:** a stratified random sample of 120 spans (40 U\*, 40 P\*, 40 A). The person reads it blind to the label, on spark-one. Targets: precision **≥ 90% for U\*** and **≥ 80% for P\***. If either is missed, fix the rules and re-label. Report the confusion matrix.
- **Agreement:** report Cohen's κ for rule vs judge on all spans. The target is ≥ 0.6. Below 0.4, the judge is dropped and the exit test becomes the second annotator.
- **Exit-test agreement:** the fraction of rule-U spans that the exit test confirms, and the same for rule-P. These two numbers are the closest we have to a check of the Zhuang result on our own model.
- **Stability:** re-run the judge on 100 spans with a different prompt order. Label flips should stay ≤ 10%.
- **Base rates** (reported, never tuned): U\* and P\* episodes per 100 problems by bucket; the wrong→right correction rate after the first candidate answer (the First Try Matters metric).

---

## 3. Contrast for the direction

### 3.1 What is contrasted

**Primary: the decision contrast** [I, design; literature reasons in `research/literature.md` §6.1]
- `d = mean over problems [ mean over matched pairs ( x(b_U) − x(b_C) ) ]`, at most 5 pairs per problem.
- `b_U` is the boundary token of a step that opens a **U\*** episode: the point where the model goes on to an unproductive re-check.
- `b_C` is the boundary token of a **matched control** step where the model does not hesitate and proceeds. The control must:
  1. be a plain or subgoal step that opens no episode and lies inside none;
  2. hold a candidate answer (2.2's `A_before` rule, applied at the step's start) **equal to the U episode's `A_before`, with the same correctness against `ref`**;
  3. sit within ±0.15 relative position of `b_U`;
  4. come from the same trace if one qualifies, otherwise from another sample of the same problem.
- Each control is used once. U spans with no control are dropped and counted (`spans.stats.json: decision_pairs`).
- Both reads are `\n\n`-type boundary tokens before the step's first word. Held answer, its correctness, problem and position are matched by construction, so what differs is the decision to re-check.
- Implemented by `capture_spans.py spans` (pairing) and `capture_spans.py direction --contrast decision` (the default).

Why not `mean(U*) − mean(P*)`, the contrast this section used before:
- Both classes open with a hesitation step, so at `b` both are "about to hesitate". The contrast does not isolate the decision to hesitate.
- The classes differ in held-answer correctness by construction. U needs `F == A_before`, and on D that answer is mostly right. P+ needs `A_before` wrong.
- Hidden states encode the correctness of an intermediate answer (Zhang et al., AUROC > 0.7, > 0.9 on a 32B model [Vp]). A mean difference across that split therefore reads mostly "my answer is right", and steering along it pushes a belief about correctness, not "do not re-check" [I].
- Its sign was also assumed (negative = less hesitation). The decision contrast drops that assumption (4, 4.1).
- The old contrast survives only as a demonstration of the confound: `--contrast labels --pos U --neg P+,Pi` flags itself in its report. A synthetic test plants a pure correctness signal and shows that this contrast picks it up at held-out AUROC > 0.9 and fails the leak test. On the same data the decision contrast stays below 0.7 [V, `test_capture_eval.py::decision_contrast_ignores_held_answer_correctness`, mock data only].
- P\* spans keep two jobs. They are base rates for G5. They also give a selectivity diagnostic: AUROC(U\* vs P\*) on `d` is reported but not gated.

Secondary directions, captured as well and compared on held-out separation, not deployed by default:
- `d_refl` = mean(all hesitation steps) − mean(non-hesitation steps). This is SEAL-style, for comparison.
- `d_known` = mean(boundaries after the exit test says the answer is already known) − mean(boundaries before). This is the Halt Vector style contrast.
- `d_effort` = mean(same prompts at effort 100) − mean(at effort 50), at the same boundary positions. It tests whether the effort knob already encodes this axis.

### 3.2 Positions

Main reading: the **boundary token**, the `\n\n` token opening the step, for both the U\* step and its control.
- That is the position where the model decides what the next step is, before the hesitation word is emitted. SEAL reads at the `\n\n` of each thought, and ReflCtrl at the step's first token [Vp].
- Capture window `[b, b+1)`.

Sensitivity reading: the first 3 tokens of the step, `[b, b+3)`, and the whole episode mean (one step for a control).

Balancing:
- Pairs are matched within a trace (or a problem), so problem identity, held answer, correctness and position are balanced by construction. Pair differences are averaged within each problem first, then across problems.
- Report the pairs' mean |Δ relative position| and mean |Δ log₂ tokens-so-far| (`decision_pairs` in the spans stats). Cross-trace pairs match relative position but not absolute length.
- The secondary class-mean contrasts (`--contrast labels`) keep the earlier balancing: bin by relative-position decile and log tokens-so-far, and re-weight the negative class to the positive class's histogram.
- Report `d` both raw and with the mean of a position-matched non-hesitation set projected out.

### 3.3 Layers

- **Now: layer 14 only** (Engram layer, the existing steer/capture point). Layer 1 is the only alternative: it needs a relaunch with `DSV41_STEER_LAYER=1` [V, stack.md §8]. Layer 14 sits below 20, so decoder bounded replay does not apply and every prompt position computes at L14 [V].
- **Multi-layer (phase 2, code):** hooks after `mhc_shifted_post_pre` in each layer, or `capture_aux=True`, with graph-safe per-request `index_add_` accumulators (~20 MB/layer) [I, design in stack.md §8].
  - Candidate layers: 6, 10, 14, 17, 20, 24, 28, 32 (KV sources and indexer sources).
  - For L ≥ 20 the probe must **end at the token of interest**, because bounded replay computes only each request's last 128 prompt tokens. The capture recipe below already does this.

### 3.4 Capture recipe (feeding trace text as prompt)

This needs the capture server (`DSV41_STEER_CAPTURE=1`, a relaunch). Per trace:

1. Rebuild the exact token sequence: `prompt_token_ids + completion_token_ids[:k]`.
2. For the spans of this trace in increasing position order (b₁ < b₂ < …): send `/v1/completions` with `prompt = tokens[: b_i + w]` (w = 1 or 3), `max_tokens 1`, temperature 0, `cache_salt = <trace id>`, and `vllm_xargs {"steer_capture": "<span id>:<b_i>:<b_i+w>"}`.
   - All spans of one trace share one salt. Earlier positions are therefore served from the prefix cache, and each probe recomputes only the new tokens since the previous probe, so one trace costs about one full prefill.
   - The captured window is always in the freshly computed tail. Cached positions are never captured [V].
3. Check `count == w` in the npz, as `extract.py` does [V]. On mismatch (for example block-aligned cache reuse reaching into the window), retry with a fresh salt.
4. One npz per span, `sum/count` = the 4-stream-mean residual at L14, **before** any steer add [V]. Move the npz files from `/dev/shm` to `~/dsv41-hesitation/capture/` at the end.
- Never capture decode-time rows. DSpark verify forwards include drafts that may be rejected [V/I, stack.md §8].

Cost [I]: about 2,700 traces × one prefill at ~16K-token chunks, plus about 2,000 short probes. A few hours on the capture server, which serves prod traffic with ~356 MB/rank extra and a host sync per captured step.

### 3.5 Direction quality checks (all before any steering)

- **Split-half cosine** of `d`, split by problem: ≥ 0.8. The kit already reports `split_half_cos`.
- **Held-out separation:** project the per-span means of the held-out 20% of problems on `d`. Requirements:
  - AUROC(U\* boundary vs its matched control) ≥ 0.70.
  - AUROC(U\* vs non-hesitation) and AUROC(U\* vs P\*) are also reported (diagnostics, not gated).
  - Compare against 100 random unit directions; their |cos| is about 1/√5120 ≈ 0.014. `d` must beat the random 99th percentile.
  - Literature calibration: ReflCtrl's reflection direction predicted correctness at AUROC 0.85 on a 14B model but only 0.56 on QwQ-32B [Vp]. So a weak separation on a strong model is plausible, and if AUROC is < 0.6 the project stops here.
- **Correctness leak:** take the non-hesitation control boundaries that did *not* build `d`. On them, the projection on `d` must not separate "held answer right" from "held answer wrong". It fails if |AUROC − 0.5| > 0.10 and |z| > 2.58. Fewer than 5 boundaries in either group means the test is unmeasured, and an unmeasured test is not a pass (`checks.no_correctness_leak` must be true for `checks.pass`).
- Cosines between `d`, `d_refl`, `d_known` and `d_effort`. If cos(`d`, `d_effort`) > 0.7, the effort knob already does most of the job (see I1).
- Norm: steering strength is in % of the L14 mean residual norm, taken from capture `norm_sum` (kit convention [V]).
- Optional low-rank clean-up, as in Manifold Steering ([arXiv:2505.22411](https://arxiv.org/abs/2505.22411) [Vp]): project `d` onto the top-k principal subspace of the captured span means. Here k ≤ 10 is chosen by held-out AUROC. This cannot be done with the current kit: capture keeps no covariance, so per-span means are the data points.

---

## 4. Interventions to compare

All arms run on D first (screening), then the survivors run on E (confirmation, section 5).

**Sign: not assumed.** `d` points from the matched proceed state to the re-check state, but which kit strength sign reduces unproductive hesitation is an empirical question. The literature shows the same kind of intervention changing sign or reversing (literature.md §6.1) [Vp/I]. Every steer screen on D therefore runs **both signs** at each magnitude, and the sign is read from the screen. The E confirmation uses the sign the screen picked, frozen in `FROZEN.json`. `eval_hesitation.py` refuses to start a screen (split-D data or `--phase screen`) in which any named steer vector appears at one sign only. Its `report.sign_screen` lists, per vector and sign, the U ratio, token ratio and accuracy change. If both signs reduce U, that points to generic perturbation, and G7 decides. The ablation arms (I5, I7) remove the component along ±`d` and have no sign.

| ID | Intervention | Needs | Literature anchor |
|---|---|---|---|
| I0 | Base. Run **twice** as two separate arms (null control) | nothing | section 0.2 |
| I1 | **Effort knob**: `reasoning_effort` 75 (`high`), 60 and 50 instead of 100 (`max`, the base) | nothing (prompt) | [V] encoder accepts 1-100 |
| I2a | **NoWait logit ban**: `logit_bias` −100 on every vocabulary token containing the NoWait 17 keywords, including `check` and `verify` | API `logit_bias` [? verify in P0] | NoWait, [arXiv:2506.08343](https://arxiv.org/abs/2506.08343): −13-31% tokens; ±2 on AIME for RL-trained 32B; −8.7 to −16.7 on AIME25 for distilled models [Vp] |
| I2b | **Hesitation-only bias**: −5 and −100 on `Wait`/`Hmm`/`Hold`/` wait`/` hmm` variants, **not** `check`/`verify` | same | CGRS: Qwen3-32B GPQA-D −7.4 [Vp], so expect science/hard losses |
| I3 | **Runtime additive steer**, L14, every generated token (existing kit): `d` at ±2, ±4, ±6, ±8, ±12 % (both signs in the screen; one frozen sign on E) | vectors published (hot reload) | ReflCtrl: every-token steering "degrades performance significantly" beyond \|0.2-0.3\| [Vp] |
| I4 | **Boundary-gated steer**: the same add only on rows whose input token is a `\n\n`-type token, and only inside `<think>` (stop after `</think>` or a tool-call token) | kit change: row mask by input token id, plus a phase flag per request. Graph-safe, because the coefficient tensor is already per row [I] | SEAL and ReflCtrl both steer only at step boundaries [Vp] |
| I5 | **Runtime directional ablation**: `r ← r − a (r·d) d` at L14, a ∈ {0.5, 1.0}, every generated row; variant I5g gated like I4 | kit change (new op) | Arditi et al., [arXiv:2406.11717](https://arxiv.org/abs/2406.11717): ablation equals weight orthogonalisation [Vp] |
| I6 | **Multi-layer** I4/I5 at a layer band | multi-layer hook (phase 2 code) | ReflCtrl uses all layers except the first and last six [Vp] |
| I7 | **Load-time rank-1 weight edit** (the bake), see 4.2 | loader patch + relaunch | ThinkEdit, [arXiv:2503.22048](https://arxiv.org/abs/2503.22048) (heads, opposite sign) [Vp] |
| C1 | **Random-direction control**: same norm and strength as the winning arm, 3 seeds (`random:<seed>` slots in the kit [V]) | nothing | — |

DEER and CGRS-style confidence-gated exits need a probe at each step and cannot be baked. They stay out of scope as interventions. Their idea is reused in the 2.4 label test.

### 4.1 How strength is chosen

1. **Ceiling from stability:** for each sign separately, the largest |strength| at which, on 40 D problems, the loop rate (`steer_eval.loopy`), `finish=length` rate and missing-`</think>` rate do not exceed the I0 null range. The Halt Vector naive bake broke `</think>` and made outputs 43% longer [Vp].
2. **KL budget, ASC-style** ([arXiv:2507.04742](https://arxiv.org/abs/2507.04742) [Vp]):
   - Teacher-force 50 base traces with and without the steer, and measure next-token KL on **non-hesitation positions**: answer tokens, tool-call tokens, and non-hesitation thinking steps. Request `logprobs: 20`.
   - Keep strengths with mean KL ≤ 0.01 nats on those positions. This makes the "edits elsewhere" check explicit.
   - If `prompt_logprobs` is unavailable [?], use generated-token top-20 logprobs from steered vs unsteered runs on matched prefixes instead.
3. **Dev sweep on D:** within the allowed range, run every magnitude at **both signs** (`eval_hesitation.py` enforces this on D). Pick the sign and strength that maximise reasoning-token savings on easy+medium D, **subject to** the hard-D and agent-D screens (section 5 gates, applied with D numbers). If only the sign the earlier spec assumed (negative) works, record that. If the positive sign works, use it. If neither sign reduces U\* more than the random arms do, stop.
4. **Freeze** the vector, strength, gating and lexicon in `~/dsv41-hesitation/FROZEN.json`, with its hash, before E runs. E never feeds back into the choice.

### 4.2 The bake (I7)

Constraints from the stack [V, stack.md]:
- There are **no bias parameters** anywhere, so a constant additive `+c d` cannot be expressed as an edit of an existing tensor. An additive steer (I3/I4) has **no exact weight twin**. Only ablation (I5) does.
- The task brief's `W' = W − α (W d) dᵀ` is the *input-side* form. For writers into the residual, the right form is the **output-side** `W' = (I − α d dᵀ) W`. That is rank-local and exact under the TP all-reduce for the RowParallel `wo_b` and shared `down_proj`.
- MXFP8 requant with round-to-nearest loses 40-80% of a weak rank-1 edit (measured on real L14 slices: random d keeps 0.427 of the write at α=1, and 0.821 vs the 0.50 target at α=0.5) [V].

Recipe [I]:
1. **Targets:** `attn.wo_b` and `ffn.shared_experts.down_proj` in a layer band ending at the steering layer, plus the Engram value rows at L1/L14 with an offline `u = dᵀV`.
   - The band is chosen from `‖dᵀW_l‖/‖W_l‖_F`, computed offline on spark-one CPU from the checkpoint (numpy, one pass, minutes). Start with the 6-8 layers where `d` is written most, all at or below 14.
   - Routed experts (MXFP4) are not edited, so the ablation is **partial**. This is a known gap.
2. **Precision:** route the edited prefixes to `EmulationMxfp8LinearKernel` (dequantised to BF16 at load) and apply the edit to the BF16 weight. This avoids requant loss. Cost: about +10.5 MB/rank and about 0.02 ms per decode step per edited `wo_b` [I, unmeasured].
   - Open question [?]: whether `wo_b` ever receives a `QuantizedActivation` on SM12x. This is checked in the first edited launch: the log must show the Emulation kernel for those prefixes and the exact check must pass.
3. **Mechanics:** a flag `DSV41_WEIGHT_EDIT=/path/edit.json`.
   - The file holds `d`, α, targets and a SHA-256.
   - The edit is applied inside `DeepseekV41LLMForCausalLM.load_weights`, after `AutoWeightsLoader`. For the BF16 route it is applied after quant processing.
   - All ranks must agree on the hash (the `_dsv41_sp_agree_min` pattern) or startup aborts.
   - Each tensor's checksum is logged before and after the edit.
   - CUDA graphs are captured after loading, so they pick up the edit with no shape change [V/I]. Rollback = relaunch without the flag.
4. **Equivalence check:** run the edited server with capture on. At U\* boundaries, the projection of the L14 residual on `d` must fall by the fraction I5 predicts, within ±20%. If it does not, the partial ablation does not reproduce the runtime result, and the bake is judged on its own E numbers only.
5. **Global scope:** a bake acts on prompt positions, KV and agent turns too, and it cannot be turned off per request. Only I5 (ungated, every row) predicts it. **I4/I5g results do not transfer to I7.** If only a gated arm passes the gates, the honest recommendation is a flag-gated runtime default, not a weight edit (section 7).

Training (SFT/DPO/LoRA) is out of scope. vLLM is inference-only. There is no training stack on the Sparks, and an MXFP4/MXFP8 665B MoE across 8 GB10 nodes has no tested training path [I]. I have not shown it is feasible.

---

## 5. Evaluation sets and gates

### 5.1 Design

- **Paired by problem.** Every arm sees the same problems. E maths uses k samples per problem at temperature 1.0, top_p 1.0, so effects are measured as per-problem pass rates.
- Greedy decoding is not reproducible here (section 0.2), so sampling gives honest variance.
- Each confirmation includes **two base arms** (I0, I0′). Their difference is the null, and every gate must clear it as well as the stated margin.
- Statistics:
  - Paired bootstrap over problems (10,000 resamples) for accuracy differences and token ratios.
  - One-sided 95% bounds for non-inferiority. Two-sided 95% CIs are reported.
  - Exact sign test on discordant problems as a cross-check.
- Every eval starts with the **load canary** (P0): the steered arm must change output on a fixed prompt, and `rank0.log` must show 0 `unknown vector` warnings for the run id.

### 5.2 Sample sizes (what confidence the numbers buy) [I]

Non-inferiority with a paired design: n ≈ (z₀.₉₅ + z₀.₈)² · p_disc / m² = 6.19 · p_disc / m². Here p_disc is the fraction of trials where arms disagree on correctness, and m is the margin.

The null discordance measured today was 2.5-6.7% at greedy on MATH L3-5. Steering and T=1 sampling raise it, and 10-20% is assumed for hard sets [I].

| margin m | p_disc 0.10 | p_disc 0.20 |
|---|---|---|
| 5 points | 248 trials | 495 |
| 3 points | 688 | 1,376 |
| 2 points | 1,548 | 3,096 |

Resulting choices:
- **Hard E:** ~254 problems × 4 samples = **~1,016 trials per arm**. That resolves about a 3-point margin at p_disc 0.10, or about 4 points at 0.20.
- "**No loss on hard maths**" cannot be proven with any finite sample. The gate below is non-inferiority at −2.5 points (one-sided 95%) **and** a point estimate ≥ −0.5. If that is unaffordable, say so and report the bound reached.
- **Easy/medium E:** 300 + 300 problems × 2 samples = 1,200 trials, with a −1.5-point margin. Token savings: the null spread of mean tokens was 4.5% at n=120, so a 15% saving is clearly resolvable at n=600 [I].
- **Productive checks:** these are rare events, so a ratio CI needs about 100+ events per arm.
  - Agent fault injection gives ~120 injected faults in E per arm. P\* maths corrections are counted on hard E.
  - With 100 events, a "keeps ≥ 80%" bound is resolvable; 90% is not [I]. The gate uses 80% for the bound and 90% for the point estimate.
- Time [I, rough]: about 9M tokens per arm on hard E (AIME/HMMT ≈ 15K tokens, MATH L5 ≈ 3K, ×4 samples) ≈ 5-6 h per arm at 450 tok/s. With 5 confirmation arms (I0, I0′, winner, C1, I7) that is about **30 h of cluster time**. P0 measures the real lengths, and this is re-planned before P6.

### 5.3 Gates (pre-registered; all must pass on E)

| Gate | Set | Metric | Pass |
|---|---|---|---|
| G1 savings | Easy + medium E | median paired ratio of reasoning tokens (winner/base) | ≤ 0.85, CI upper bound < 0.95, and outside the I0/I0′ null |
| G2 easy accuracy | Easy + medium E | paired Δ accuracy | lower 95% bound ≥ −1.5 points |
| G3 hard maths | Hard E (MATH L5, AIME 25/26, HMMT 25/26) | paired Δ pass rate | lower bound ≥ −2.5 points **and** point estimate ≥ −0.5; truncation and loop rates ≤ base + null spread; `</think>` emitted in ≥ 99.5% of runs. Report each competition separately (no averaging away a loss on AIME 2026) |
| G4 hesitation actually removed | Hard + medium E, re-labelled with the frozen section 2 rules | U\* episodes per problem | ≥ 30% fewer than base (CI excludes 0) |
| G5 useful checks kept | Hard E maths + agent E | P\*/problem and the wrong→right correction rate after the first answer; **fault detection rate** on injected faults | point estimate ≥ 90% of base, lower bound ≥ 80% of base; fault detection lower bound ≥ base − 5 points |
| G6 agent success | BFCL v3 multi-turn (E) + τ²-bench retail + fault suite | task success | lower bound ≥ −3 points per benchmark; tool-call parse error rate ≤ base + null spread |
| G7 direction-specific | Same sets as G1/G3, random arms C1 (3 seeds, same strength) | savings and accuracy | the winner's savings exceed every random seed's, with the CI of the difference excluding 0. If random directions give similar savings, the effect is generic perturbation (as `rnd8` hinted, section 0.2), not hesitation removal. Reject |
| G8 bake fidelity (I7 only) | 50 D prompts + section 4.2 check | projection drop vs I5; tok/s and DSpark acceptance | projection drop within ±20% of I5's; decode tok/s ≥ base − 3% |
| G9 no side damage | MATH-500 non-thinking (120), 50 general chat prompts | accuracy, loops | within the null range (section 0.2) |

**Verdict rule.** An arm passes only if every gate required for its role is **measured and passes**:
- Candidate (steer) arms need G1, G2, G3, G4, both halves of G5 (maths and agent fault detection, recorded separately), G6, G7 and G9.
- Baseline arms (I1, I2) need the same list without G7.
- A missing or unmeasured gate counts as a **FAIL** and is listed under `verdict.unmeasured`; it is never skipped. Examples:
  - no null arm: G1's "outside the null", and G3's and G9's null spreads, cannot be measured;
  - no random arms: G7 cannot be measured;
  - the agent gates (G5-agent, G6) have no result from the agent harness;
  - a bucket is missing from the data;
  - the base arm has no U or P events, so the ratio is undefined;
  - a hard competition with no paired error-free trials (G3 reports each one; it is unmeasured, not skipped).
- Errored or missing trials drop problems from the paired comparisons, which can only flatter an arm. So the verdict also FAILs when the arm, or an arm its gates compare against (base, null, and the random arms for a candidate), has an unresolved error or a planned trial with no row (`report.completeness`, listed under `verdict.incomplete`). An error that a resume later retries successfully is resolved.
- An arm halted by the stop rule fails (`meta.stopped`, listed under `verdict.stopped`), and a resume keeps it halted.
- `eval_hesitation.py --gates` implements this (`gates()` → `verdict`). The agent harness's G5-agent and G6 results can only enter through `--external-gates FILE`. A file cannot override a gate the tool computes.
- G8 is judged in the I7 window, outside this tool.

Stop rules:
- Stop at once if any **hard competition** set shows a point estimate ≤ −5 at the first checkpoint (half the samples).
- Stop if the direction fails 3.5 (AUROC < 0.6).
- If I1 (effort 60) or I2b reaches G1-G6 by itself, report that a weight edit is not needed and stop before I7.

---

## 6. What runs on the cluster, in order

General rules:
- Each phase takes `~/dsv41-opt-20261021/CLUSTER_LOCK`, waits for idle (`idlewait.py`) and logs to `AGENTS_LOG.md`, following the `window.sh` pattern [V].
- Relaunches use `dsv41-relaunch.sh --expect-launcher 416e3ace` [V: `sha256sum ~/dsv41-tp8.sh` on spark-one, 2026-11-03]. P0/P1 check the same 8 characters on all 8 nodes before staging.
- After every relaunch, run the `exact.py` bit-exactness check against the stored hash.
- Measured relaunch downtime is **161-167 s** [V]. Steps P0-P2 and P4 use the prod API with unsteered or opt-in requests and cause **no downtime**.
- Every phase needs the user's go-ahead. This spec authorises nothing.

| # | Phase | Server state | Downtime | Duration (est.) |
|---|---|---|---|---|
| P0 | **Preflight.** (a) Thinking: the render check (1.3) for top-level `"max"` and `"high"`, then the P0c runs; check that `prompt_token_ids` contain `<think>` and `Reasoning Effort: 75`/`100`, and that `reasoning_content` is non-empty. This resolves the section 0.3 anomaly. (b) Check that `logit_bias` and `prompt_logprobs` are accepted. (c) Measure thinking lengths on 20 problems per bucket. (d) Fix the kit's interpreter (`~/venvs/vllm032/bin/python`). (e) Build the data files and the D/E split with hashes. | prod | 0 | 1 h |
| P1 | **Trace collection** (section 1), D only, unsteered | prod | 0 | ~15 h (re-planned after P0c) |
| P2 | **Labeling**: rules (CPU on spark-one, light), judge requests, exit tests (short requests, prefix-cached), human audit | prod | 0 | 4-6 h + audit |
| P3 | **Capture.** Relaunch with `DSV41_STEER_CAPTURE=1`, run the section 3.4 probes, relaunch prod | capture server (prod traffic still served, with overhead) | **2 × ~165 s** | 3-5 h |
| P4 | **Vectors + screening.** Compute `d` etc. on spark-one CPU; 3.5 checks; publish to all 8 nodes (hot reload, no relaunch [V]); load canary; KL budget; dev sweeps of I1, I2a/b, I3 (both signs), C1 on D | prod, opt-in arms only | 0 | 6-10 h (×2 steer arms for the two signs) |
| P5 | **Kit v2** (I4 boundary/phase gating, I5 ablate op). Install on all 8 nodes and relaunch; `exact.py` must show unsteered requests bit-identical; CPU tests as in `test_v2_cpu.py`; dev sweep of I4/I5 on D | prod with new kit | **~165 s** | 6-8 h |
| P6 | **Freeze + confirmation on E**: I0, I0′, winning runtime arm, C1 (3 seeds folded into one arm by seed rotation), and I1/I2b if they were competitive. Gates G1-G7, G9 | prod, opt-in arms | 0 | ~30 h (5.2) |
| P7 | **Bake (I7)**, only if an *ungated* I5 arm passed P6. Offline: band selection and edit file (CPU). Relaunch with `DSV41_WEIGHT_EDIT` (+ capture for G8); fidelity check; E confirmation of I7 vs the P6 base arms; **all traffic on this server gets the edit**, so schedule it as a window; relaunch prod without the flag | edited server | **2-3 × ~165 s** (+ load-time edit, est. < 1 min) | 6-30 h depending on E scope |
| P8 | **Decision**: if I7 passed, document `DSV41_WEIGHT_EDIT` as an opt-in launch flag. The default launcher is unchanged until the user decides | prod | 0 | — |

Total planned downtime: **5-6 relaunches ≈ 14-17 min**, plus the P3 capture window (degraded) and the P7 window (modified model for all users).
Multi-layer (I6) adds one more kit relaunch plus a capture relaunch pair, and runs only if P4/P5 show a single-layer effect that is real but too weak.

### 6.1 P0 + P1 runner (`ops/`)

- `ops/run_p0_p1.sh` (from the Mac): refuses if unit `hes-p1` is active on spark-one or `CLUSTER_LOCK` is held; checks the launcher (`416e3ace`) on 8/8 nodes; stages the kit and `data/splits` to spark-one `~/dsv41-hesitation/{kit,data}` with a `SHA256SUMS` check (a split file that changed under an existing run stops staging); takes the lock atomically (`set -C`, with a token) and logs to `AGENTS_LOG.md`; starts `systemd-run --user --unit hes-p1 --collect -p KillMode=mixed -p ExecStopPost=release_lock.sh`, so the run survives the Mac disconnecting. `--stage-only` stages and verifies only. `--reclaim-stale` takes over a lock left by a dead `hes-p1` unit.
- The unit (`ops/hes_unit.sh` → `ops/p0p1.py`): **P0** blocking checks are interpreter, kit checksums, scorer tests, split hashes, launcher + `dsv41-rank` active, `/v1/models`, free disk, the render check (1.3) for `max` and `high`, and the P0c length runs (20 D problems per source at P1 settings: no empty reasoning, no prompt without `<think>`, every unclosed `</think>` explained by truncation, AIME median reasoning ≥ 1,000 tokens). Any failure = exit 1 and **no P1** (fail closed). P0b (`logit_bias`, `prompt_logprobs`) is recorded, not blocking. P0 writes `plan.json`: tokens and hours from the P0c lengths and tok/s. Over 36 h the hard k drops to 2 (1.4); still over = fail closed. **P1** runs `collect_traces.py` per D file (easy, medium, AIME 2024, AIME 1983-2023, MATH L5) at conc 6 with the plan frozen.
- Resumable: rerunning `run_p0_p1.sh` re-checks P0 (P0c runs resume, so it is quick) and P1 continues each run where it stopped. The lock is released by the unit's EXIT trap and again by `ExecStopPost`; both remove it only if the token matches.
- Progress: `ops/status.sh` (unit state, phase, per-run done/expected, tok/s, ETA, log tail). Files: `~/dsv41-hesitation/runs/p0p1/{progress,p0,p0c,plan,summary}.json`, `summary.txt`, `unit.log`.

---

## 7. Risks and honest expectations

- **No precedent at this scale.** All published activation-steering and length-direction results are on dense models ≤ 32B. None covers a 600B+ MoE or a hyper-connection residual (literature.md, gap 1). The literature pattern is savings of 40-70% on easy tasks, 10-25% on AIME-level ones, and losses concentrated on hard sets [Vp].
- **The bake may not be possible as stated.** If the winner is an additive or gated steer, there is no exact weight twin (no biases; global scope). A partial writer ablation also leaves routed experts writing `d`. ThinkEdit reports that steering toward shorter thinking "consistently degrades accuracy" [Vp]. The likely end state may be a flag-gated runtime default instead of a weight edit. That is decided by data in P6/P7, not assumed.
- **The model may already be efficient.** DSv4.1-1017 is described as already concise. Large RL-trained models were the most robust to NoWait (±2 on AIME) [Vp], which also means less slack to remove. The effort knob (I1) may already capture most of it.
- **Label noise** (Zhuang: 93.3% of keyword boundaries did not reproduce [Vp]) is handled with three-way agreement and the exit test. The cost is fewer P\* spans.
- **Decision pairs may be scarce.** A U\* span without a matched control (same held answer and correctness, within ±0.15 relative position, same trace or problem) is dropped. If fewer than 600 pairs remain, first widen `--max-rel-gap` or add samples per problem. Never drop the correctness match: that is what keeps `d` from encoding correctness. On real traces the yield is unknown [?]. The mock traces paired 91 of 96 U spans, but they were built to have controls.
- **Run-to-run nondeterminism** (section 0.2) means small effects cannot be resolved without the sample sizes in 5.2.
- **Agent coverage is synthetic**: simulated tools, BFCL and τ²-bench. Real DSH agent sessions are deliberately not used, so effects on the user's own agent work are inferred, not measured.
- **Unknowns to settle early:**
  - thinking/effort plumbing (P0a);
  - `logit_bias` and `prompt_logprobs` support (P0b);
  - whether `wo_b` stays on the BF16 path after rerouting;
  - whether the DSpark drafter shares `embed`;
  - the exact row counts of the MathArena sets;
  - the contamination of 2025 competitions.
