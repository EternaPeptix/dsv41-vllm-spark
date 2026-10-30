# r4 eager-glue: fewer eager kernels between the target and drafter graphs

Status: staged, not installed. Nothing was restarted and no GPU jobs were run.
- Both new kernels are **bit-exact against the stock code** in Triton interpreter mode on spark-one's CPU. Run `test_eager_glue_interp_cpu.py`; output is in `tests/interp_cpu.out`.
- SWA: 4 cases, covering padding rows, invalid decode rows, replay_start and short sequences.
- DPAD: 1, 5 and 16 requests.
- The GPU test (`test_eager_glue_gpu.py`) is still to be run by the experiment agent.

Flag: `DSV41_EAGER_GLUE=all`. The parts can also be set separately: `swa` and `dpad`. Default is off; when off, both files take the stock code path.

## What the r2 trace shows (rank 0, c1 code)
Inventory: `r4/tools/eagerdump.py`, output in `r4/tools/ed.r0.1.txt` and `ed.r0.3.txt`.
- **Totals.** Per step there are 102 eager kernels and 13 memcpys, covering 1.48 ms GPU-busy outside the graphs.
- **Large kernels.** Most of that time is three big kernels that a graph would not speed up:
  - lm-head 748 µs
  - main_proj 380 µs
  - logits all-gather 37 µs
- **Drafter end → next target: 0.42 ms window, 92 ops, only 0.17 ms busy.** The rest is the GPU front-end gap, about 2–3 µs per launch. The CPU runs about 29 ms ahead, so this is not host latency.
  - **15× the same triple from `DeepseekSparseSWAMetadataBuilder.build`, one per SWA KV-cache group**, about 12.3 µs each, 185 µs in total:
    1. a 1-block elementwise op, `torch.ge(slot_mapping, 0, out=is_valid_token)`
    2. a 16-block fill, `decode_swa_lens[n:] = 0`
    3. `_compute_swa_indices_and_lens_kernel`
  - Adaptive-verification `reallocate_drafts`: about 16 small kernels (cumprod, gatherTopK, 2× cub scan), about 70 µs.
  - prepare_inputs / prepare_attn / indexer metadata: about 25 ops, about 90 µs.
- **Target end → drafter start: 23 ops.**
  - **3 × `_prepare_dflash_inputs_kernel` at 15 µs each, grid (1,1,1).** The cost is the last request's padding loop `for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE)`.
    - `max_num_tokens` = `max_num_batched_tokens` = 16384, and `BLOCK_SIZE` = 16.
    - That makes about 1000 serial 16-wide masked stores in one program.

## Parts
- **swa** (`v1/attention/backends/mla/sparse_swa.py`):
  - For the causal decode path, `build()` launches one new Triton kernel, `_dsv41_fused_decode_swa_kernel`, instead of the three launches.
    - Decode programs compute `is_valid = slot_mapping[t] >= 0` themselves and store it, then run the stock body.
    - Tail programs (1024 wide) write `is_valid_token` for the remaining rows and zero `decode_swa_lens[n:]`.
  - Unchanged:
    - non-causal (DSpark draft) builds
    - prefill-only builds
    - prefill / image paths, which still read `is_valid_token`, now written by the fused kernel earlier on the same stream
  - Saves 30 launches per step (15 groups × 2).
- **dpad** (`v1/worker/gpu/spec_decode/dflash/speculator.py`):
  - `_prepare_dflash_inputs_kernel` gets a `PAD_BLOCK` constexpr, used only by its four padding loops: 1024 when the flag is on, `BLOCK_SIZE` when off (stock).
  - The stores are the same.
- **Not done** (documented leads):
  - Fusing AV `reallocate_drafts` into one kernel: about −50 µs. It must reproduce the gatherTopK tie-break, so bit-exactness needs care.
  - Capturing the whole D→T glue in a graph: knapcio's tvglue/dglue failed on stale per-step tensor addresses.
  - Batch-sharded sampling: incompatible with AV's GPU-side draft reallocation, and uses an NCCL all_to_all.

## Expected gain
| | c1 (33.5 ms step) | c16 (111 ms step) |
|---|---:|---:|
| swa: 15 × (fill 1.3 + ge 1.1 + about 5.8 µs of front-end gaps), minus the fused kernel's extra cost | −0.10 to −0.12 ms | −0.10 ms |
| dpad: 3 × (15 → about 1.5 µs) | −0.04 ms | −0.04 ms |
| **total** | **−0.15 ms (+0.4–0.5%)** | **−0.14 ms (+0.13%)** |

Small but free, and bit-identical. The GPU test reports both numbers directly. It measures the GPU-side time behind a sleep kernel, which is how serving runs (CPU ahead).

## Files touched (conflict detection)
| file | live sha → patched | notes |
|---|---|---|
| `site-packages/vllm/v1/attention/backends/mla/sparse_swa.py` | 2afd3136 → 38b16c51 | diff `sparse_swa.diff`, +170 lines: new kernel and wrapper before `ComputeSWAIndicesAndLensKernel`, `import os`, `build()` branch |
| `site-packages/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | b731b09c → see `SHA256SUMS` | diff `dflash_speculator.diff`. DSpark subclasses it; r3 spec patches `dspark/speculator.py`, not this file, so there is no overlap |

## Install / test / A-B
```
S=/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad
bash $S/r4/eager-glue/install.sh        # preflight 8/8, .bak-r4-eager-glue, verify 8/8 (inert until flag)
ssh spark-one 'source ~/venvs/vllm032/bin/activate; python3 ~/dsv41-stage/r4/eager-glue/test_eager_glue_gpu.py'
#   expect: all SWA cases "bit-exact", DPAD "bit-exact", PASS; prints us saved per step (15 groups / 3 calls)
bash $S/conc/launch-conc.sh DSV41_EAGER_GLUE=all
#   exact.py within instance must match the flag-off control hash on the SAME instance class
#   (bit-identical metadata => identical greedy output); bench single/conc.
bash $S/r4/eager-glue/uninstall.sh
```
