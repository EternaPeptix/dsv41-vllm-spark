# r4 prefill-sched: short requests no longer wait behind a long chunked prefill

This is the vLLM 0.32 port of knapcio's `DSV41_PREFILL_BYPASS` (SGLang `adapter/prefill_bypass.py`) and of their `--prefill-decode-interval`.

## Problem (from reading vLLM 0.32 `v1/core/sched/scheduler.py`)

- `schedule()` serves RUNNING requests first. A request in the middle of a long prefill takes `min(remaining, token_budget)`, which is the whole 16384-token budget. The WAITING loop then starts with `token_budget == 0`.
- As a result, a short request that arrives during a 128K prefill gets its first token only after the whole prefill finishes. At about 5.4K tok/s, a 128K prefill has up to about 24 s left.
- A request that was already decoding still gets its 1+5 tokens each step, but every step is now a whole 16K chunk of about 3 s. So its inter-token latency is about 3 s.
- The stock knob `--long-prefill-token-threshold T` caps every prefill at T tokens whenever more than one request exists (the num_eligible>1 check at line 645). It fixes starvation, but it costs prefill throughput whenever anything else is running. The adaptive variant has the same problem.

## Patch (flag-gated, default OFF): `vllm/v1/core/sched/scheduler.py` only

Only rank 0's EngineCore runs the scheduler. The file is installed on all 8 ranks anyway, so the hashes stay uniform.

| env | meaning |
|---|---|
| `DSV41_PREFILL_BYPASS=<N>` | When a waiting request with ≤ N fresh tokens is among the first `_SCAN` waiting requests, and a prefill longer than `_CHUNK` is either running or queued ahead of it: that step caps long prefills at `DSV41_PREFILL_BYPASS_CHUNK` tokens and defers long waiting requests by one step. The short request is admitted in the same step. A lone long prefill still gets the full 16384 chunk. |
| `DSV41_PREFILL_BYPASS_CHUNK` (default 8192) | Cap on the long prefill during a bypass step. Rounded down to the 64-token block size. |
| `DSV41_PREFILL_BYPASS_SCAN` (default 32) | How many waiting requests are scanned for a short candidate. |
| `DSV41_PREFILL_DECODE_CAP=<T>` (default 0 = off) | Caps a long prefill at T tokens per step while other requests are decoding. This bounds their ITL to about T/5.4K s, at some cost to prefill throughput. It plays the role of `--prefill-decode-interval`. |

- Both knobs stand down when the stock `--long-prefill-token-threshold` is set.
- The decision reads only scheduler state, so it is deterministic and makes no collectives.
- Diff: `prefill-sched.diff` (+~90 lines). Full file: `patched/vllm/v1/core/sched/scheduler.py`.

### Zero-code alternative to A/B first

Pass `DSV41_EXTRA="--long-prefill-token-threshold 8192"`. This is stock, works through the launcher's `DSV41_EXTRA`, and needs no install. Expect it to fix starvation as well, but to cut prefill throughput whenever any other request is running.

## Tests

- `test_sched_cpu.py`: drives the decision logic through ast extraction, with no vLLM needed. **PASS** on the Mac and on spark-one, 12 scenarios.
- GPU/E2E: `../prefill/run_ab.sh`, suite `st` (starvation) plus `pf` (lone-prompt prefill must not regress). See `../prefill/README.md`.

## Expected effect

- Lone long prompt: no change. A single request means the stock path is unchanged.
- Short request (300 tokens) arriving during a 128K cold prefill:
  - TTFT goes from about 5–20 s (the rest of the long prefill; median about 12 s) to about one capped step plus its own prefill, roughly 1.5–2 s.
  - knapcio measured 80 s → 4.6 s at 400K on 4 nodes.
- Long prompt TTFT: each bypass step processes 8192 instead of 16384 long tokens. Expect roughly +0.1–0.3 s per admitted short request.
- With `DSV41_PREFILL_DECODE_CAP=4096`, a decoder's ITL during a long prefill should drop from about 3 s to about 0.8 s. The long prefill gets roughly 5–10% slower while decoders are active, because of smaller-chunk efficiency. Measure that.

## Files touched

`vllm/v1/core/sched/scheduler.py`. Live sha256 `d9a39870286c424c…`, identical on all 8 ranks. Patched sha256 is in `FILES.txt`.

## Install

1. `./stage.sh`. Already done: staged at `~/dsv41-stage/r4/prefill-sched/vllm/...` on all 8 ranks.
2. `./install.sh`. It backs up to `.bak-r4-prefill-sched`, refuses on drift, and sha-verifies all 8 ranks.
3. Restart with the env, e.g. `../prefill/launch-env.sh "DSV41_PREFILL_BYPASS=2048"`.
4. `./uninstall.sh` restores the backup and verifies the original sha.
