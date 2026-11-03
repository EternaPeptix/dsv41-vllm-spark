# incr-tok: window runbook (server-side incremental tokenization)

The API server re-tokenizes the whole conversation on every agent step. This kit caches (prompt string, prompt ids) for
recent prompts in the API server and tokenizes only the new suffix, cut at a token boundary that is proven restartable.
It is behind `DSV41_INCR_TOK` (default off). Prompt ids are identical to a full tokenize, so model output cannot change.

Nothing here has run in the live service. The CPU tests and the bench ran on spark-two (results below). The ops scripts
passed against 8 stub hosts on the Mac (`tests/test_ops_stub.py`), but they have not run against the real 8, so expect a
first-run fix or two, and a failure there should be loud.

Launcher `66eaa1d9`. Base file `renderers/deepseek_v4.py` `f7175f99` (identical on spark-one and spark-two). The kit does
not touch the supervisor manifest files.

## What it changes

| File | Change |
|---|---|
| `renderers/deepseek_v4.py` | Anchored edit adding three things: `import os`, one env read (`_DSV41_INCR_TOK`), and at the end of `DeepseekV4Renderer.__init__`: `if _DSV41_INCR_TOK: from .dsv41_incr_tok import attach; attach(self)`. Backup: `deepseek_v4.py.bak-incrtok`. |
| `renderers/dsv41_incr_tok.py` | New file. The cache, the load checks and the guard. Never imported while the flag is unset. |

`attach()` swaps the renderer instance's `_tokenize_prompt` and `_tokenize_prompt_async` (same executor) for the cache
wrapper. `base.py` is untouched.

**Flag values**
- `DSV41_INCR_TOK` unset or `0`: stock behaviour.
- `DSV41_INCR_TOK=1`: serve cache hits, with the guard.
- `DSV41_INCR_TOK=audit`: always serve the full tokenize. The incremental result is computed too and compared (shadow
  mode).

**Tunables** (defaults in parentheses):
- `DSV41_INCR_TOK_VERIFY_FIRST` (32)
- `DSV41_INCR_TOK_VERIFY_EVERY` (64)
- `DSV41_INCR_TOK_MAX_MB` (256)
- `DSV41_INCR_TOK_MAX_ENTRIES` (32)
- `DSV41_INCR_TOK_MIN_CHARS` (32768): shorter prompts take the stock path.
- `DSV41_INCR_TOK_MIN_REUSE` (0.5)
- `DSV41_INCR_TOK_STATS_EVERY` (200)
- `DSV41_INCR_TOK_LOG`: path for a per-request JSON line with timings. Prompt text is never logged.

**Algorithm**
1. Find the cached prompt P with the longest common prefix L with the new prompt S.
2. Set c = the end of the last occurrence, inside S[:L], of one of `<｜end▁of▁sentence｜>`, `<｜User｜>`,
   `<｜Assistant｜>`, `</think>` or `<think>`.
3. Check that ids(P) ends with enc(P[c:]).
4. Return ids(P)[:k] + enc(S[c:]). `enc` is the same HF `__call__` (`encode_batch`, releases the GIL) with the same
   kwargs as stock.

**Why it is exact.** PROFILE.md section 4 has the proof. In short, for this tokenizer, a literal occurrence of an added
token that overlaps no other added token is always matched as that token. Everything after the added-token split
works on each piece separately. `｜DSML｜` overlaps itself and is never a cut.

**Load checks.** If any of these fails, the overlay logs `DISABLED` and the renderer stays stock:
- `tokenizer.json` sha256 is `c90dfa01...`;
- the loaded pipeline's structure digest is `fbfe19de...`;
- normalizer, pre-tokenizer, BPE and added-token flags are as the proof needs;
- the 5 cut tokens have their ids and overlap no added token;
- a known-answer split test passes.

They take about 0.5 s at API-server start.

**Falls back to the stock call:**
- prompts that are not text, or are shorter than MIN_CHARS;
- images (`multi_modal_data` / `multi_modal_uuids`);
- offsets, `truncate_prompt_tokens`, `add_special_tokens=True` and any unexpected encode kwarg;
- any result whose length reaches the truncation `max_length`, so length errors and truncation stay stock;
- reuse under 50%;
- a failed tail check;
- any internal exception.

**Guard.** The first 32 hits of a process are also tokenized in full and compared. After that, 1 hit in 64 is compared,
at random. On any difference the overlay does three things:
- logs `DSV41_INCR_TOK MISMATCH` at ERROR;
- serves the full result;
- clears and disables the cache for the life of the process.

A wrong id that slipped past the sampling would be reused by the next hits of that conversation, so the next sampled
check of that conversation would catch it (the guard test's control case shows this).

**Memory.** Each cache entry holds the prompt string plus 4 B per token: 3.2 / 5.9 / 9.8 MB at 161K / 306K / 507K
tokens. Entries are evicted LRU, by bytes (256 MB) and by count (32). A request that extends a cached prompt replaces it.

## Results so far (spark-two CPU, synthetic data only)

**Tests** (`tests/test_incr_tok.py`, 147 s, VERDICT PASS, `results/test_incr_tok.json`). Each request goes through
pydantic and the real `DeepseekV4Renderer`, with verification off so that hits are really served. The whole
`TokensPrompt` is compared with the stock `_tokenize_prompt`.

- **Equality fuzz.** 2,000 conversations, 11,909 requests and 58.1M tokens, all identical:
  - 9,274 hits; 66% of all tokens reused;
  - all 7 chat modes;
  - 4 conversations interleaved, sharing one system prompt and tool list;
  - ops: grow, middle edits, system edits, rollbacks, resends, compaction and mode switches;
  - 566 edit requests, each reusing nothing past the edit; the 399 with the edit before the reuse floor all missed.
- **Concurrency.** 8 conversations on 4 threads: 112/112 identical.
- **Eviction.** 1 MB / 4-entry cache: 120/120 identical, and the bounds held after every request.
- **String-level fuzz.** 6,000/6,000 identical. This covers random edits, truncations and special tokens split on purpose.
- **Fallbacks.** Images, `truncate_prompt_tokens`, `add_special_tokens`, short prompts, and n = max_length - 1 / = /
  > max_length all behaved as specified.
- **Guard.** A forced wrong id is caught by the first-N check, by the 1-in-K check and by audit mode. Each time the full
  result is served, the cache disables itself and later requests run stock.
- **Installer.** dry-run, install, status and uninstall round trip. It refuses a moved base sha. The flag-off AST
  difference is exactly the three additions.

**Bench** (`tests/bench_incr_tok.py`, `results/bench_incr_tok.json`). The real renderer, synth.py agent conversations,
one step per request, ms per request, idle cluster:

| prompt tokens | stock tokenize | cache hit | saved | verified hit | miss (new conversation) |
|---:|---:|---:|---:|---:|---:|
| 161K | 224.9 | 8.0 | **216.9** | 223.6 | 226.2 |
| 306K | 432.0 | 10.6 | **421.4** | 424.1 | 418.7 |
| 507K | 672.6 | 13.6 | **659.0** | 670.9 | 661.3 |

**The saving depends on cluster load.** The same prompts tokenized 2x slower an hour earlier, while the cluster was
serving 8 concurrent requests at about 400 tok/s (profile: 497 / 907 / 1428 ms). A probe on the same 335K-token string
took 899 ms then and 490 ms now. Expect about 0.42 s per step at 300K on an idle cluster and up to about 0.9 s under
load. Confirm with `ops/log_summary.py` on live traffic (section 8).

## 0. Setup, from the Mac

    export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
    which ssh    # must print the shim
    K=$HOME/Documents/dsv41-kits/incr-tok
    (cd $K && shasum -a 256 -c --quiet SHA256SUMS) && python3 -I $K/tests/test_ops_stub.py | tail -1    # VERDICT PASS, no host contacted

Take CLUSTER_LOCK and log to AGENTS_LOG.md as usual before the stop. The A/B needs a quiet cluster: `ttft_ab.py` waits
for an idle server before each step and records foreign traffic. Release the lock after section 7 or 8.

## 1. Stage to all 8 (disk only, safe while serving)

    bash $K/ops/stage8.sh

It must end `stage8: 8/8 staged OK <digest>` and exit 0. Every host also dry-runs the installer, which fails unless the
base is `f7175f99` and the side file is absent.

Optional: re-run the CPU tests from the staged copy on spark-two (CPU only, about 2.5 min plus about 3 min):

    ssh -n spark-two 'cd ~/dsv41-stage/incr-tok && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 19 timeout 600 ~/venvs/vllm032/bin/python tests/test_incr_tok.py 2>&1 | tail -1'
    ssh -n spark-two 'cd ~/dsv41-stage/incr-tok && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 19 timeout 600 ~/venvs/vllm032/bin/python tests/bench_incr_tok.py 2>&1 | tail -1'

**A failure in section 0 or 1 changed nothing.** Do not stop the service. Fix the cause and rerun section 1, or end the
window.

## 2. Stop the service (your call)

From this stop on, any failure goes to section 7.

    for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do ssh -n $h 'timeout 30 systemctl --user stop dsv41-rank; systemctl --user is-active dsv41-rank' & done; wait

## 3. Install on 8/8, flag off

    bash $K/ops/site8.sh install
    bash $K/ops/site8.sh status patched

`install` first runs a precheck on all 8: base sha, anchors, and the service stopped. Nothing is written anywhere unless
all 8 pass. Then it installs. It must end `site8 install: 8/8 exit 0, identical, state patched`.

Only spark-one's API server uses the renderer, but all 8 are installed so the sites stay identical. The supervisor's
preflight manifest does not cover `renderers/`, so `site8.sh status` is the check for that.

## 4. Flag-off arm on the patched site

    bash $K/ops/arm.sh off

`arm.sh` checks the stage digest and `patched` on 8/8. It then relaunches through
`~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh --expect-launcher 66eaa1d9 DSV41_GPU_CLOCK_MHZ=2000` and runs these on
spark-one:

1. `exact.py`: anchor `a8c3394af9d28249`.
2. `goal_single.py`: G1 digest `9a18f802d930f6e7`.
3. The 9 long-prompt cases x4 cold through `~/dsv41-g5det/scripts/dbgrun.py`. Every run must hit the reference token
   hashes: big:2500 3f4a9dd1, big:8000 d3e0f849, big:32000 7ba3d1a9, big:128000 89bd9111, code:4000 43c4b492,
   prose:6000 05e29037, T16 37dd0438, T02 6b7032f2, T01 6d8cf8a2. These are the same as final3/4/5.
4. `ops/ttft_ab.py`: a synthetic growing agent conversation of 56 steps, about 230K to 330K tokens, at temperature 0
   with `return_token_ids`.
5. The `DSV41_INCR_TOK` lines and the ERROR/Traceback count of `rank0.log`.

`ops/arm_check.py` decides. With the flag off there must be no `DSV41_INCR_TOK` line at all. The arm must end
`ARM off: PASS` (exit 0). Results are in `spark-one:~/dsv41-incrtok/off/`. The service stays up.

Expected time: the relaunch, then about 1 to 2 min for exact, about 2 min for G1, about 4 to 5 min for the long prompts,
and about 3 min for the A/B. Step 0 of the A/B is a cold 230K prefill, about 40 to 60 s. None of these durations are
measured.

Note that exact.py, G1 and the long prompts never reach the cache:
- exact.py and G1 prompts are short;
- dbgrun's big/code/prose cases are token-id prompts;
- the l3 chat cases are short.

They prove that nothing else moved. The A/B's prompt-id hashes are the cache's exactness evidence on the live server.

## 5. Flag-on arm, then the A/B

    bash $K/ops/arm.sh on DSV41_INCR_TOK=1

The same checks run. `rank0.log` must also show `DSV41_INCR_TOK=on enabled (incr-tok v1): tokenizer.json c90dfa01,
structure fbfe19de, ...` and no `DISABLED`, `MISMATCH`, `attach failed` or `internal error`. arm.sh adds
`DSV41_INCR_TOK_LOG=/home/jeweled/dsv41-logs/incr-tok.jsonl`. Then compare the two arms:

    ssh -n spark-one 'python3 -I ~/dsv41-stage/incr-tok/ops/ab_compare.py ~/dsv41-incrtok/off/ab.jsonl ~/dsv41-incrtok/on/ab.jsonl ~/dsv41-incrtok/on/incr-tok.jsonl'

**Gates** (exit 1 on any):
- every step has the same prompt_tokens and the same sha of the server's `prompt_token_ids` in both arms. This is the
  cache's exactness on the live server, and none may differ;
- the overlay log has no mismatch record;
- at least 10 steps were real cache hits. With the defaults, the first 32 hits of the process are verified hits, so
  steps 1-32 show no saving and steps 33-55 do.

**Reported, not gated:**
- greedy output agreement (`out_sha`). A difference while the ids are equal points at decode nondeterminism, not at
  this kit. Stop and review it before promoting;
- TTFT medians over all steps, over hit steps and over verified steps, with the per-step saving. Expect about 0.4 s
  less TTFT on hit steps at about 300K if the cluster is idle.

Optional confirmation pair: `bash $K/ops/arm.sh on2 DSV41_INCR_TOK=1`, then `bash $K/ops/arm.sh off2`, then compare off2
against on2 the same way.

**Promotion:**
- `ARM off: PASS`;
- `ARM on: PASS`;
- `AB_COMPARE PASS`;
- greedy outputs identical;
- a TTFT saving on hit steps.

**Alternative: audit first.** `bash $K/ops/arm.sh audit DSV41_INCR_TOK=audit` serves stock on every request. It compares
the incremental result on every eligible request of live traffic, and its log gives `ms_full` and `ms_incr` for each
one. That is a zero-risk way to collect a day of live mismatch=0 evidence and the live saving estimate before serving
from the cache.

## 6. Keep it on (only after section 5 passes)

`arm.sh on` leaves the service up with the flag on. The launcher does not set the flag, so **any later relaunch without
`DSV41_INCR_TOK=1` runs stock**, with the site still patched and the overlay never imported. To keep it across relaunches
there are two options:
- pass `DSV41_INCR_TOK=1` (and optionally `DSV41_INCR_TOK_LOG=...`) to every `dsv41-relaunch.sh`;
- or add `export DSV41_INCR_TOK=${DSV41_INCR_TOK:-1}` to `~/dsv41-tp8.sh` on all 8. That changes the launcher sha
  (`66eaa1d9`) that other kits' `--expect-launcher` pins, so it is your decision.

Release CLUSTER_LOCK.

## 7. Rollback / back to production

**Fast** (the site stays patched, the flag is off, the overlay is never imported):

    ssh -n spark-one 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > /tmp/rel.sh && bash /tmp/rel.sh --label incrtok-flagoff --expect-launcher 66eaa1d9 DSV41_GPU_CLOCK_MHZ=2000; echo rc=$?

**Full**, which is the path for any failure after section 2:

    for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do ssh -n $h 'timeout 30 systemctl --user stop dsv41-rank; systemctl --user is-active dsv41-rank' & done; wait
    bash $K/ops/site8.sh uninstall && bash $K/ops/site8.sh status base && ssh -n spark-one 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > /tmp/rel.sh && bash /tmp/rel.sh --label incrtok-prod --expect-launcher 66eaa1d9 DSV41_GPU_CLOCK_MHZ=2000; echo prod_rc=$?
    ssh -n spark-one 'cd ~/dsv41-bench && timeout 900 python3 exact.py /tmp/incrtok-prod-exact.json incrtok-prod > /dev/null 2>&1; python3 -I -c "import json;c=json.load(open(\"/tmp/incrtok-prod-exact.json\"))[\"combined\"];print(c[:16],\"OK\" if c.startswith(\"a8c3394af9d28249\") else \"MISMATCH\")"'

You need `prod_rc=0` and `a8c3394af9d28249 OK`. `uninstall` refuses while the service runs. It restores
`deepseek_v4.py` from `.bak-incrtok`, deletes the side file and its pyc, and must end `state base` on 8/8. If it fails
on a host, read that host's output and fix it there. Production must not come back on a half-patched site.

## 8. Watching it live

    ssh -n spark-one 'grep -a "DSV41_INCR_TOK" ~/dsv41-logs/rank0.log | tail -3 | cut -c1-400'    # enabled line + stats every 200 requests
    ssh -n spark-one 'python3 -I ~/dsv41-stage/incr-tok/ops/log_summary.py ~/dsv41-logs/incr-tok.jsonl'  # needs DSV41_INCR_TOK_LOG

`log_summary.py` reports, from the per-request log:
- the counts per kind;
- the live full-tokenize cost in us/token (from the verified and miss requests);
- the hit cost;
- an estimated saved seconds per day.

That figure is the live confirmation of the ~85-220 s/day estimate, which has not been measured. A `MISMATCH` line in
rank0.log means the cache already turned itself off for that process. Keep that log and the incr-tok.jsonl record, and
roll back with section 7.

## Hazards and limits

- **What is not exercised by the anchors.** exact.py, G1 and the long prompts never reach the cache path, as explained
  in section 4. The live-server exactness evidence for the cache is the A/B's per-step prompt-id hashes and the
  overlay's own first-32 verification. Audit mode is the stronger option if you want it on real traffic.
- **Concurrency.** `renderer_num_workers` is 1 today, so the tokenizer calls run on one thread, as in stock. The cache
  is lock-protected and was tested with 4 threads. With more workers, the HF backend's truncation state is shared
  between threads in stock code too, and this kit does not change that.
- **Tokenizer change.** The overlay disables itself at load (logs `DISABLED`) if the tokenizer changes, for example
  through a new tokenizer.json, a new tokenizers or transformers version, or new added tokens. To re-pin it, re-run
  `PROFILE.md` section 4 and the tests.
- **Requests the cache does not serve.** Images, `/v1/completions` with `add_special_tokens` (the default),
  `truncate_prompt_tokens` and prompts under 32,768 chars all take the stock path unchanged.
- **GIL.** A hit holds the GIL for about 10 to 15 ms at 500K: the slice and `tolist()`, plus the array copy on store.
  Stock already stalls the event loop for about 36 ms building its id list.
- **Ops scripts.** None has run against the real hosts. `arm.sh` is new; the relaunch, exact, G1 and dbgrun calls follow
  the g130 kits' working pattern.
