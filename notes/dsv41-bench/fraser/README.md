# Fraser bench for dsv41 (single-stream code decode, vLLM-Moet method)

`fraser_bench.py` (stdlib only) ports the "Fraser bench" from kacper-daftcode/vLLM-Moet
(`tools/sm120_perf/fraser_bench.py`, commit b0dc737, Apache-2.0; see `NOTICE`). The goal is a number we can put next
to theirs. On 4x RTX PRO 6000 TP4 at k=7 they report a median of 680 tok/s, AL 5.34 and 127.8 steps/s.

## Method

The method is unchanged from theirs:

- One fixed prompt, `fraser_prompt.txt`, a byte-for-byte copy of theirs. It is 2,678 tokens, and its sha256 `5a6b7bc1cbbc73a7...` is pinned in the script.
- The first line of each request is a cache-bust nonce: `[LocalMaxxing cache-bust nonce: <hex>]`.
- Temperature 0, `max_tokens` 2048, thinking OFF (`chat_template_kwargs {"thinking": false}`), one stream, a fresh prefill for every rep.
- Their headline number is `tok_s_out_lm` = completion_tokens / (last token - first token). We also report `tok_s_out` = (n-1) / (last - first), which is the convention our G1/G4 scripts use.
- Spec-decode numbers come from `/metrics` deltas taken around each request:
  - AL = 1 + accepted/drafts.
  - acceptance = accepted / draft_tokens.
  - steps/s = drafts / (last - first).
  - accepted per position. The pooled per-position rates show what k=6/7 adds at positions 5-6.
- Report the MEDIAN over the reps, not the best.

## Our defaults (each one can be switched off)

| ours (default) | theirs |
|---|---|
| `--nonce-mode fixed`: rep i uses nonce sha256("fraser-i")[:32], and every request carries a random `cache_salt`. Every arm therefore sees the same prompts with a fresh prefill each time, and `outputs_digest` is a bitwise check across arms. | `--nonce-mode random`: uuid4 nonce, no salt |
| `--warmup 1`: one rep with nonce "warmup" runs first and is excluded | none |
| Refuses to start a rep while the server has running or waiting requests (`--allow-busy` turns this off) | none |
| `--reps 10` | `--reps 3` (their published row used 40) |

## Usage (spark-one, server healthy, no foreign traffic, telemetry off)

    python3 -I ~/dsv41-bench/fraser/fraser_bench.py --label <arm> --out <dir>/fraser.jsonl [--reps 10]

The script writes one JSON line per rep (`metric: FRASER`) and a `FRASER_summary` line. The summary holds:

- median, best and mean of `tok_s_out_lm`
- median `tok_s_out`, steps/s, AL and acceptance
- the pooled per-position accept rate
- `outputs_digest`

In the k A/B (`~/dsv41-stage/moet/ab/moetab.sh`) this is the `fraser` phase (`ops/moetphase.sh`). That phase also takes a spec-counter snapshot before and after.

## Comparability

Their prompt and decoding settings are kept. The hardware and stack are different: 8 GB10 nodes at TP8 over CX7 against 4 RTX PRO 6000 at TP4 over PCIe. Read the absolute tok/s as context only. What matters is the same-prompt comparison between our k=5, 6 and 7 arms.

The script was tested only offline, against a mock SSE server (`~/dsv41-stage/moet/tests/mock_vllm.py`). It has not yet sent a request to the live server.
