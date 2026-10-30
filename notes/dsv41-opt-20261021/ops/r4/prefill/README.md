# r4 prefill cluster: TTFT and long-prompt ports (prep only; nothing was run on GPU)

Baseline:
- Prefill is about 5.4–5.6K tok/s from 16K to 64K and 5.3K at 200K (longctx-r3 timings).
- TTFT is 23.6 s at 128K.
- `--max-num-batched-tokens 16384` with chunked prefill, so one 16K chunk takes about 3.0 s.
- There is **no prefill profile yet.** All gains below are estimates from bytes/FLOP models and the reference repos.

## Mapping of the knapcio / rhys prefill features to vLLM 0.32

| reference feature | vLLM 0.32 status | r4 deliverable |
|---|---|---|
| `DSV41_PREFILL_BYPASS` + `--prefill-decode-interval` (short request admitted beside an in-flight chunk) | Starves: RUNNING prefills take the whole budget first (scheduler.py:657–857). Stock `--long-prefill-token-threshold` fixes it but caps prefill whenever >1 request exists. | **prefill-sched**: `DSV41_PREFILL_BYPASS`, `_CHUNK`, `DSV41_PREFILL_DECODE_CAP` (scheduler.py) |
| `SPARK_PREFILL_TP_SPLIT` (SG18 indexer row split) | Already in vLLM (`PrefillRowShard`) but gated at 16384 rows/rank, so it never engages at TP8 with a 16K budget. | **prefill-idxshard**: `DSV41_IDX_ROW_SHARD_MIN`, `_MIN_CTX` (indexer.py) |
| `DSV41_PREFILL_SP` (RS+AG, per-row work on 1/TP rows) | vLLM SP is static and EP/MegaMoE-only, so it is off for TP8. | **prefill-sp**: dynamic per-forward SP, `DSV41_PREFILL_SP`, `_MIN_ROWS`, `_DEBUG` (nvidia/model.py) |
| `DSV41_HC_FUSED` (one-pass hc stats, SGLang Triton split-K kernel) | Not portable as is. vLLM uses tilelang `mhc_pre_delayed_tilelang` / `mhc_shifted_post_pre` (the DeepGEMM `tf32_hc_prenorm_gemm` overlap path is SM100-only). SP already cuts these rows 8×, which makes it mostly moot. | none: measure the mHC share in the prefill profile first |
| `DSV41_INDEXER_CHUNKED` (sglang#39187, bounded logits transient) | Native: `sparse_attn_indexer` already iterates `prefill_metadata.chunks` bounded by the workspace. | none |
| `CHUNKED_PREFILL_SIZE` 8192 vs 4096 (+17–28% at 400K–1M) | We already run 16384. knapcio warns that 16384 at 1M hung a node (memory); our max_model_len is 262144. | config sweep only (optional): `--max-num-batched-tokens 24576` / `32768`, measured with `pf` and MemAvailable |
| `PREFILL_SP_FP8` (MXFP8 gathers) | Follow-up on top of prefill-sp. | not done |

All three patches touch different files and compose. Each one is inert unless its env is set, every rank must get the same env, and `launch-env.sh` passes it to all 8 ranks.

## Files

- `/…/r4/prefill-sched/`: patched `vllm/v1/core/sched/scheduler.py`, `prefill-sched.diff`, `install.sh`, `uninstall.sh`, `stage.sh`, `FILES.txt`, `test_sched_cpu.py` (PASS), `README.md`
- `/…/r4/prefill-idxshard/`: patched `vllm/v1/attention/backends/mla/indexer.py`, `prefill-idxshard.diff`, the same scripts, `test_idxshard_cpu.py` (PASS on spark-three CPU), `README.md`
- `/…/r4/prefill-sp/`: patched `vllm/models/deepseek_v41/nvidia/model.py`, `prefill-sp.diff`, the same scripts, `README.md` (the GPU test is the in-server DEBUG compare plus A/B)
- `/…/r4/prefill/`:
  - `bench/prefill_bench.py`: stdlib bench with modes `prefill`, `starve`, `logprobs`, `compare`. Validated against a local mock SSE server; not yet run against the live server, which belongs to r3.
  - `run_ab.sh`: one arm = restart, then the suite.
  - `launch-env.sh`: restart all 8 ranks with `systemd-run -E` env, leaving the launcher untouched.
  - `tools/mk_*.py`: patch generators with sha-pinned anchored edits; re-run to rebase.
  - `live/`: snapshot of the live sources the patches are based on.
- Staged on the Sparks:
  - `~/dsv41-stage/r4/{prefill-sched,prefill-idxshard,prefill-sp}/vllm/...` on all 8 ranks, sha-verified.
  - On spark-one: `~/dsv41-stage/r4/prefill-bench/prefill_bench.py` and the CPU tests.

## TTFT / starvation benchmark plan (experiment agent, after r3 finishes)

Run everything from the Mac. Results land in `spark-one:~/dsv41-bench/r4prefill/<label>/`. Install the patch files once at the start (`*/install.sh`); with no env set they are inert, so the baseline arm is still stock behaviour. Check that with the step 0 smoke.

0. **Prefill profile (recommended first, ~10 min).**
   - Run nsys on rank 0 for one cold 16K-token chunk, the same way as r2-profile.md.
   - Bucket the time: MoE FP4, dense GEMM, NCCL AR, mHC (tilelang), indexer logits + top-k, attention, Engram.
   - This sets the real ceiling for prefill-sp (mHC + Engram + norm share × 7/8) and prefill-idxshard (indexer share × 7/8 at 64K–128K context).
1. **Baseline arm:** `run_ab.sh base ""`, suite `pf st lp lc`.
   - Repeat `lp` with a second salt to get the baseline-vs-baseline logprob noise.
2. **prefill-sched arms** (suite `st pf`):
   - `sched-stock`: `run_ab.sh thr8k "" --long-prefill-token-threshold 8192`. Stock knob, for reference.
   - `sched-bypass`: `run_ab.sh byp "DSV41_PREFILL_BYPASS=2048"`.
   - `sched-bypass+cap`: `run_ab.sh bypcap "DSV41_PREFILL_BYPASS=2048 DSV41_PREFILL_DECODE_CAP=4096"`.
   - Metrics from `st` (128K long + 6 × 300-token shorts every 2 s from t=3 s, plus a decoder started 3 s before):
     - `short_ttft_p50/max`
     - `long_ttft_s`
     - `decoder_itl_inside_ms_p50/max`
   - Metric from `pf`: lone-prompt tok/s, which must equal baseline ±2%.
   - **Win criteria:** short TTFT p50 ≤ 3 s (baseline is expected to be 5–20 s); long TTFT +≤ 5%; `pf` unchanged.
3. **prefill-idxshard arm:** `run_ab.sh idx "DSV41_IDX_ROW_SHARD_MIN=1024 DSV41_IDX_ROW_SHARD_MIN_CTX=32768"`, suite `pf lc lp`.
   - Expected +3–5% at 128K and +5–9% at 200K; flat at 16K.
   - longctx must PASS (needle 100%, copy ≥ 0.9).
4. **prefill-sp arm:**
   - Debug smoke first: `run_ab.sh spdbg "DSV41_PREFILL_SP=1 DSV41_PREFILL_SP_DEBUG=3"` with `SUITE=pf` and short lens. Gate on the compare lines (see prefill-sp/README).
   - Then `run_ab.sh sp "DSV41_PREFILL_SP=1"`, suite `pf lp lc gs`.
   - Expected +10–25% prefill tok/s at all lengths ≥ 2K.
5. **Combined arm:**
   - `run_ab.sh all "DSV41_PREFILL_SP=1 DSV41_IDX_ROW_SHARD_MIN=1024 DSV41_IDX_ROW_SHARD_MIN_CTX=32768 DSV41_PREFILL_BYPASS=2048"`, suite `pf st lp lc gs`.
   - Plus the usual 1-user and conc4/8/16 decode bench, which must be unchanged: all three are prefill-only paths.
6. Optional: chunk-size sweep (`--max-num-batched-tokens 24576`, then `32768`) on the best arm.
   - Watch spark-one/two MemAvailable during a 200K `pf`; abort if it goes below 4 GiB.

Order by risk: sched (scheduler only, rank-0 logic), then idxshard (stock vLLM code path, threshold only), then sp (model forward). Uninstall with each `uninstall.sh`, which verifies the original sha on all ranks.
