# r4 spec-sync-free: nothing to port (no per-step rank-0 broadcast on our GPU path)

## knapcio's change (SGLang)
`DSV41_SPEC_SYNC_FREE=all` drops 9 NCCL broadcasts from rank 0 per decode step. They gained about −0.14 ms on the critical path; in-boot A/B gave −0.53 ms and the reboot suite about −0.2 ms.
- 5 in the draft graph: one per Markov step, `DSPARK_GRAPH_SAMPLE`.
- 3 in the verify epilogue: correct_len, bonus and cap_trim_lens.
- 1 for verify_cap's live length.

Those broadcasts are SGLang `SpecTpSync` sites. They exist because its DSpark sampler draws rank-local torch-generator noise.

## Our stack (vLLM 0.32 model-runner v2, MRv2)
### Code read
spark-one site-packages, identical on all 8 ranks.
- `v1/worker/gpu/spec_decode/{dspark,dflash}`, `rejection_sampler*`, `sample/*`, `model_runner.py` and `async_utils.py` contain **no per-step broadcast**.
- The only `broadcast_object` is in `adaptive_verification.py:342`. It runs once, at cost calibration.
- `pp_handler.broadcast*` is PP-only, and we run PP=1.
- Every rank samples and verifies identically:
  - The logits are all-gathered.
  - Per-request seeds are identical on every rank (see the r3 spec-sampling handoff).
  - Draft tokens come from the DSpark sharded-Markov all-gather of (score, token) pairs, which is computation, not a broadcast.

### Traces
`tools/sync_audit.py` over the r2 nsys traces; output in `sync_audit_r2.txt`.

| case | broadcast kernels/step | NCCL kernels/step | eager collectives/step |
|---|---:|---:|---|
| c1 code | 0 | 0 | 1 `ar_kernel`, 66 µs: the target logits all-gather |
| c4 | 0 | 0 | 1, 170 µs |
| c16 | 0 | 7 AR + 1 AG (drafter size fallback) | 1, 624 µs |
| c1 prose | 0 | 0 | 1, 41 µs |

### Host side
- The per-step rank-0 traffic is the scheduler output, sent through vLLM's `MessageQueue` broadcast (ZMQ to the 7 remote workers).
- It is off the critical path. Every rank's CPU runs at least one step ahead of its GPU:
  - rank 0: the next target graph is launched 29.5 ms before the drafter ends.
  - rank 7, re-measured for r4: the drafter is launched 54 ms before it starts, and the next target starts 0.42 ms after the drafter ends.
- Those 0.42 ms are exactly the eager glue that `r4/eager-glue` attacks.
- The 2 `cudaEventSynchronize` calls per step (`record_confidences`, `AsyncOutput.get_output`) block the host on the GPU, not the reverse.

**Estimated gain from porting: 0 ms.** There is nothing to drop.

## Related leads that are real collectives, not redundant broadcasts
Not implemented here.
1. **c16 drafter NCCL all-gather (845 µs/step at c16 in r2; RING_LL, inside the drafter graph).**
   - exp3's `RDMA_AR_MAX_REDUCE_BYTES=1 MiB` already moved the 7 drafter all-reduces to RDMA. 80 rows × 5120 × 2 B = 0.8 MB.
   - The gather is still above `RDMA_AR_MAX_BYTES` (default 2 MiB).
   - Env-only check:
     1. Run one boot with `RDMA_AR_AUDIT=400` to log the gather's bytes.
     2. If it fits the arena budget, try `RDMA_AR_MAX_BYTES=4194304`. The arena grows by about 80 MB per rank.
   - Expected about −0.6 ms at c16 (+0.5% aggregate), nothing at c1–c8.
   - This may already be different after r3-e1's sharded Markov drafter, so re-audit first.
2. **Target logits all-gather (eager, 37–66 µs at c1, 624–734 µs at c16, plus an 87 µs fp32 copy of the 96×129280 logits).**
   - For all-greedy batches, a vocab-parallel argmax would remove it. Each rank reduces its shard; then an (value, index) pair all-gather. `LogitsProcessor.get_top_tokens` already exists.
   - It needs the rejection sampler to take target argmax ids instead of logits, plus a per-batch greedy check.
   - It touches the same files as r3 spec-sampling (`rejection_sampler_utils.py`), so it is postponed until r3 lands.
   - Expected −0.7 ms at c16 when every request is greedy, about 0 at c1. Not pursued in r4.
   - vLLM's own `--enable-batch-sharded-sampling` is **not** usable as is:
     - Its shard plan is built from the CPU `cu_num_logits_np`, while adaptive verification reallocates drafts on the GPU only.
     - It uses an eager NCCL `all_to_all_single`.
