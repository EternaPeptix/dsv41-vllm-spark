# r4 prefill-sp: dynamic prefill sequence parallel for the TP8 MoE path

This is the vLLM 0.32 port of knapcio's `DSV41_PREFILL_SP` (SGLang `adapter/prefill_sp.py`, +20% prefill from 16k to 262k on TP4).

## Why it should help (more on TP8 than on TP4)

- In prefill steps every rank repeats the per-row hyper-connection work on all rows. For each layer that is two `mhc_shifted_post_pre` calls (or a pre and a post), the norms, and on layers 1 and 14 the Engram gate.
- The stream is `hc_mult=4 × 5120` bf16 = 40 KB per row. One 16K chunk (16384 rows, about 3.0 s) therefore streams about 70 GB of hc traffic across 20 layers, or ≥ 0.3 s at roofline on 238 GB/s. That is about 10–20% of the chunk, and all of it is redundant 8×.
- SP keeps the residual stream sharded by rows. Attention and the MoE still see all rows (all-gather before them), but they return reduce-scattered partial sums instead of all-reduced ones. Collective bytes are the same: RS + AG equals a ring AR.
- vLLM's own SP (`_use_sequence_parallel`) is static and needs EP/MegaMoE, so it is off for us. This port is **dynamic**, per forward:
  - On only for eager forwards with ≥ `DSV41_PREFILL_SP_MIN_ROWS` rows (default 2048), outside CUDA-graph capture, with no PP and no static SP.
  - Decode, verify, the drafter and every captured graph (≤ 768 rows) never take it.

## Patch (default OFF): `vllm/models/deepseek_v41/nvidia/model.py` only

| Location | Change |
|---|---|
| `DeepseekV4Model.forward` | Decides `dsv41_sp` from the row count only, so it is rank-identical. It shards the embedding rows (`sp_shard`), runs layers `[start, decoder_replay_start)` (0–19) with `dsv41_sp=True`, then all-gathers `hidden/residual/post_mix/res_mix/pre_mix` once before the decoder-replay layers. Layer 20 `write_kv` and the replay of layers 20–39 are unchanged. |
| `DeepseekV4DecoderLayer.forward(dsv41_sp=...)` | mHC pre/post run on this rank's rows. Attention: `all_gather → attn` with `wo_b.reduce_results=False` toggled for the call, then `sp_reduce_scatter`. FFN: `all_gather → MoE` with `moe_config.skip_final_all_reduce=True` toggled, then `sp_reduce_scatter`. If the MoE kernel reports `output_is_reduced`, it takes the reduced output's local rows instead. |
| Engram layers (1, 14) | `_dsv41_sp_engram`: `wkv` runs as in stock (TP column-parallel + gather, all rows), and the gate/injection kernel runs on this rank's rows (kv/mask offset by the shard start). |
| `_run_layers` | Gathers `previous_aux` (DSpark aux taps) when SP is on. |
| `DSV41_PREFILL_SP_DEBUG=<n>` | For the first n SP forwards, runs the stock all-reduce path on the same inputs first, then logs max_abs / rel_l2 of all 5 gathered tensors per rank: `DSV41_PREFILL_SP compare ...` in rankN.log. KV/indexer writes are overwritten by the SP pass. Debug only; it doubles those forwards. |

- Diff: `prefill-sp.diff` (~300 lines, mostly helpers).
- Not bit-identical to stock, because the reduce-scatter sum order differs from the all-reduce. knapcio has an EXACT mode that keeps the AR; this port does not.

## Tests (run by the experiment agent; nothing here was run on GPU)

1. Install, then `launch-env.sh "DSV41_PREFILL_SP=1 DSV41_PREFILL_SP_DEBUG=3"`, then send 3 prompts of ≥ 4K tokens (`prefill_bench.py prefill --lens 4k,8k,16k --reps 1`), then grep `DSV41_PREFILL_SP compare` in every `~/dsv41-logs/rankN.log`.
   - **Gate:** `hidden` and `residual` rel_l2 ≤ 2e-2 and no NaN/inf. Expected is bf16 sum-order noise of about 1e-3–1e-2 after 20 layers.
   - A rel_l2 of about 1 means a missed or extra reduction (for example, the MoE reduced internally): **stop**.
2. `run_ab.sh sp1 "DSV41_PREFILL_SP=1"`, suites `pf lp lc`, plus `gs` (gsm8k 50). Then `prefill_bench.py compare base/lp.json sp1/lp.json`.
   - **Gate:** first token equal ≥ 5/6, pre-divergence mean |Δlogprob| within 2× a baseline-vs-baseline repeat; longctx PASS; gsm8k within ±1 of baseline.
3. Decode non-regression: the 1-user code/prose bench must be unchanged, since the path is not taken below 2048 rows.

## Expected gain

- Prefill tok/s **+10–25% (median estimate +15%)** for chunks ≥ 2048 rows, i.e. all prompts ≥ 2K tokens.
  - Grounding: the 7/8 removal of the per-row hc work estimated above (≥ 10–20% of chunk time), minus one extra state gather (about 820 MB at 16K rows, about 20–30 ms per chunk), cross-checked against knapcio's +20% on TP4, where only 3/4 of the work was redundant.
  - TTFT at 128K: about 24 s → about 20–21 s.
- Composes with prefill-idxshard (different file, independent).
- 0 effect on decode.

## Risks / open items

- `skip_final_all_reduce` and `reduce_results` are toggled at runtime. This is safe because the compilation mode is NONE (no torch.compile), the path is eager, and the toggles sit in try/finally. If r2/r3 MoE kernels (M32 / compact FC1 / fc2) reduce elsewhere, the DEBUG compare will show it.
- Extra transient memory: one gathered state of N×(4+1)×5120 bf16 (about 820 MB at 16K rows). Watch MemAvailable on spark-one/two during a 200K prefill.
- Follow-ups not ported:
  - knapcio's `PREFILL_SP_FP8`: MXFP8 attention-input gathers, +2–3%.
  - Row-splitting the replicated attention-front GEMMs (wqkv_a / compressor / indexer q) under SP: they still run on all rows on every rank. This is the next prefill lever after SP if the profile shows them large.

## Files touched

`vllm/models/deepseek_v41/nvidia/model.py`. Live `5fe8b4a758ab586d…` on all 8 ranks. Any other r4 idea that edits this file (EP / MoE ports) conflicts; `install.sh` refuses on drift. To rebase, re-run `../prefill/tools/mk_sp.py`, which uses anchored replacements.
