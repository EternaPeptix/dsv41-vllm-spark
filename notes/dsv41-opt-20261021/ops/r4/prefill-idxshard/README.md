# r4 prefill-idxshard: turn on vLLM's built-in TP row split of the dense prefill indexer at TP8

This is the vLLM equivalent of rhys101's SG18 / knapcio's `SPARK_PREFILL_TP_SPLIT`. On TP4, knapcio measured prefill going from 3499 to 4364 tok/s at 128K and from 2701 to 3893 at 262K.

## Finding

vLLM 0.32 already contains this feature. In `sparse_attn_indexer.py`, `PrefillRowShard` gives each rank a contiguous, cost-balanced slice of the prefill query rows. The ranks then all_gatherv the int32 top-k, and `balanced_prefill_row_shard` in `v1/attention/backends/mla/indexer.py` computes the slices.

It is gated by `MIN_TP_SHARD_ROWS_PER_RANK = 16384`. At TP8, a step needs ≥ 131072 prefill rows to engage, but with `--max-num-batched-tokens 16384` it can have at most 16384. So the split **never runs** on our deployment.

Which layers it covers:
- Only the index-source layers before the candidate source run the dense indexer on all rows: layers 2, 8 and 14 (ratio-2 compressed keys).
- Layer 20 is the candidate source. It only writes K in `forward_kv` during the eager replay, and candidate layers stay replicated by design.
- Layers 21–39 run on 128 rows per request.

So this covers all of the full-row indexer work.

Other prerequisites, all already satisfied:
- `tp_prefill_row_sharding_supported`: CUDA, no DCP/PCP, PyNCCL on, no NCCL symm-mem, not batch-invariant, and cudagraph mixed mode = PIECEWISE.

## Patch (default = stock): `vllm/v1/attention/backends/mla/indexer.py`

| env | meaning |
|---|---|
| `DSV41_IDX_ROW_SHARD_MIN=<rows/rank>` | Replaces the 16384 floor. Recommended: 1024, which engages at ≥ 8192 prefill rows in a step. |
| `DSV41_IDX_ROW_SHARD_MIN_CTX=<tokens>` | Also requires the longest prefill in the step to reach this context, so short prompts skip the exchange. Recommended: 32768. |

- **Every rank must have the same env.** The decision gates a collective. It depends only on the rank-identical `seq_lens_cpu` / `query_lens_cpu`.
- Diff: `prefill-idxshard.diff` (+19 lines).

## Tests

- `test_idxshard_cpu.py` (torch CPU, extracts the functions via ast). **PASS** on spark-three CPU, run both stock and with MIN=1024 / MIN_CTX=32768. Output is in `test_idxshard_cpu.out`.
  - For chunk 3 of a 128K prompt, the slices are cost-balanced: [2467, 2305, 2172, 2059, 1962, 1878, 1803, 1738].
- E2E: `run_ab.sh` with suites `pf` (prefill tok/s at 16k/64k/128k/200k) and `lc` (longctx needle/copy gate; this is the correctness gate for top-k exchange bugs), plus `lp` against baseline.

## Expected gain

Grounded in our flat-but-declining prefill: 5.6K tok/s at 64K → 5.3K at 200K.

- At 128K the dense indexer costs roughly 120–180 ms per 3 s chunk at mid-prompt. That breaks down as about 4.4 TFLOP of FP8 logits per layer × 3 layers, plus about 2 GB of fp32 logits written and read per layer.
- Sharding removes 7/8 of that. The added exchange is about 32 MB of int32 top-k per layer.
- Prefill: **+3–5% at 128K, +5–9% at 200K–256K, 0 at ≤ 32K**. TTFT for a 200K prompt drops about 2–3 s.
- Decode is unaffected: the change is prefill metadata only.

## Files touched

`vllm/v1/attention/backends/mla/indexer.py`. Live `43d4f705d0fcecf5…`, identical on all 8 ranks. This is the file that already carries `DSV41_SM121_VARLEN_INDEXER`.

## Install

`./install.sh` / `./uninstall.sh` (staged). Then `../prefill/launch-env.sh "DSV41_IDX_ROW_SHARD_MIN=1024 DSV41_IDX_ROW_SHARD_MIN_CTX=32768"`.
