# r1-attn adversarial review (review-attn, 2026-11-08)

**Verdict: SHIP** (G20 / DSV41_IDXPAR installable as is; G11(b) skip accepted). No blocking items.
Four non-blocking hardening items, listed at the end. Nothing was run on a GPU, by me or by the builder.

## What I verified myself

| check | result |
|---|---|
| live `models/deepseek_v41/attention.py` sha256, spark-one..eight | `f2f48198de81ebc7…` on 8/8, matching `base/` and `KIT.json`. No `attention.py.bak-r1attn` and no `nvidia/dsv41_idxpar.py` on any node (8/8 absent). |
| kit hashes | base `f2f48198…a801b857ca`, patched `0e4b6e91bc54d88f…83514fd`, helper `b682d81332df0198…25b79da`: all match KIT.json. |
| other base copies vs live spark-one | dsv41_l2_prefetch.py `63737f9c`, nvidia/model.py `860b1dfd`, compressor.py `5807de8c`, multi_stream_utils.py `015ff402`, breakable_cudagraph.py `18702fc5`, cudagraph_utils.py `ef2aae2e`, launcher `fda6d05a`: all identical. |
| tests | Reran `tests/run_all.sh ~/exo-venv313/bin/python3` on 512S1 (Python 3.13.13) from a fresh copy: 9 + 8 + 3 = 20/20 OK, rc=0. TEST-RESULTS.txt is genuine. |
| installer on a temp tree (`R1ATTN_VLLM`) | install f2f48198 → 0e4b6e91 with the backup at f2f48198; a second install is a no-op; status and dry print the 10 tagged lines (87-88, 615-617, 864-868); uninstall restores f2f48198 and removes the helper; uninstall on a clean tree is a no-op with rc=0. |
| torch instantiate flags | live libtorch_cuda.so (torch 2.13.0+cu130, git cf30153c): `CUDAGraph::instantiate` at 0xf3a8bc does `mov x2,#0x9` → `bl cudaGraphInstantiateWithFlags`. 9 = AutoFreeOnLaunch \| UseNodePriority, so the builder's priority claim holds. |
| cuBLAS workspace at capture | `CUDAGraph::capture_begin` (0xf40fe0) never calls `clearCublasWorkspaces`, and `torch/cuda/graphs.py __enter__` only does synchronize/empty_cache. Workspaces held by allocated blocks survive, so `_warm_blas` in the first eager forward really pre-allocates the medium stream's workspace. |
| MXFP8 GEMM scratch | flashinfer `mxfp8_gemm_template_sm120.h:140-142` uses `GemmUniversal<…, cutlass::gemm::PersistentScheduler>`, which is not stream-K, so the workspace is 0. The two concurrent wq_b GEMMs never touch the shared `mm_mxfp8_workspace` (gemm_base.py:6225). `mm_mxfp8(out=None)` gives fresh outputs. |
| production capture mode | rank0.prev.log shows `FULL_AND_PIECEWISE` with breakable CG auto-enabled: 31 FULL captures, which go through plain `torch.cuda.graph` (cudagraph_utils.py capture loop), and 47 PIECEWISE ones. Decode runs in FULL graphs, where IDXPAR is active; breakable/PIECEWISE captures set `BreakableCUDAGraphCapture._tls.active` (breakable_cudagraph.py:166), so `_breakable_active()` sends them to stock. |

## Flag-off identity: holds

`diff base patched` gives exactly 3 hunks and 10 lines, each carrying `# r1-attn`:
- the module-level bool (attention.py:87-88);
- `if _DSV41_IDXPAR:` before `enable_jit_warmup` in `__init__` (615-617);
- `if _DSV41_IDXPAR and self._dsv41_idxpar is not None:` at the top of `_prepare_and_attn` (864-868).

With the flag unset, both guards are false and short-circuit: `_dsv41_idxpar` is never read and the helper is never imported. No stream, event, allocation or kernel is added, so launch order and graph contents are unchanged. The AST-equality test confirms this, and I re-derived it from the diff.

## Exactness when on: holds by construction (no device run)

In capture, `DeepseekV4Indexer.forward` is exactly `_produce_k` (if owns_k) followed by `forward_q`, because the short-context branch requires `not is_current_stream_capturing()` (attention.py:1492-1525). The helper calls the same two methods. The only module whose `__call__` is skipped is the indexer itself, and nothing hooks it: the L2 hooks are on weight-holding leaves and wq_b (dsv41_l2_prefetch.py:487-507), and the model.py `m.forward=` swaps sit in prefill-SP debug paths. The compressor is still called through `__call__`.

Every kernel runs with the same arguments; only its stream and priority change. Nothing in the reorder adds or moves a floating-point reduction:
- the compressor chain is now serial (compressor → wk → kstore → insert_cache);
- `insert_cache` (rope_quant_insert, compressor.py:297-327) reads the latent but does not write it.

The only PDL kernel in the region is `fused_q_kv_rmsnorm_quant`, which sits before E0 and keeps its predecessor. Grepping deepseek_v41/ and v1/attention found no other decode-path vLLM kernel with `launch_pdl`. The DSpark drafter layers are built without `aux_stream_list` (dspark.py:130-137), so `LayerPlan.create` returns None and they run stock.

## Races: none found

**Inputs and joins.** Every branch input (qr, kv, kv_score, indexer_weights, positions) is produced on main, or joined into main by `_run_parallel_input_projections`, before E0. Each branch records its event after its last kernel, and main waits on all of them before `_sparse_indexer_and_attn`.

**Allocator.** Branch-allocated outputs (q on aux1, index_q/scale/weights on hi, latent and k_pre on med) are freed back to their own stream's pool. The next allocation on those streams waits on an event recorded on main after the consumer: E0 of the next index layer, or `ln_events[0]` for the aux1 `weights_proj`. Caller-owned inputs live until `forward()` returns, after the join.

**Write concurrency.** The new concurrent writes are SWA insert ∥ compressed insert ∥ indexer-K store. They target distinct cache layers and slot mappings, so their bytes are disjoint.

**L2 prefetch.** The WOA/attn forks (`_fork2` and `_fork_part`) fork from the current stream, which is now hi, right after the indexer wq_b, and are joined into main by `_join2`/`_join3` at the end of the forward. They only issue cache hints.

**Events.** Within each `_schedule` call, every event wait follows a record of that event in the same capture, so there is no cross-capture event use. Events and streams are created outside capture.

## Non-blocking items (recommended, do not gate install)

1. **uninstall is blind** (install.py:103-108): `os.replace(BAK, ATTN)` runs without checking that the current file is the kit's patched sha or that the backup is the base sha. I demonstrated both failures on a temp tree:
   - a line appended after install (status reported `base=MISMATCH`) was silently discarded;
   - a stale `.bak-r1attn` containing foreign bytes was restored as is (sha 376fd8b6 instead of f2f48198).

   Today no other kit edits attention.py: r1-pf only wraps `_run_parallel_input_projections` at runtime, and there is no stale backup on 8/8. The normal flow is therefore exact. Suggested fix: refuse unless sha(ATTN) == patched_sha256 and sha(BAK) == base_sha256, and fall back to `unpatched()` when only the first holds.
2. **Flag parsing differs between the two files.** attention.py treats any value other than ""/"0" as on. The helper accepts only 1/on/prio/flat and raises `ValueError` during model init for anything else, so `DSV41_IDXPAR=off` or `=true` would crash startup. This fails closed and is visible, but the parsers should match.
3. **The installer does not check the helper's sha** against `KIT.json new_sha256` before `copy2`, and it overwrites any existing `dsv41_idxpar.py` without checking.
4. **Cosmetic:** the install message says "11 tagged lines" for 10 lines, because line 87 carries the TAG twice. The builder's claim that "the PDL overlap is lost" may be pessimistic: with programmatic stream serialization, the cross-stream edge from q_kv_norm may stay programmatic. That can only help.

## Device checks to insist on in the window (already listed in KIT.json)

- The boot log must print distinct priorities -2/-1. If they print the same value, the arm degrades towards `flat`.
- The WOA `wq_b=48 (learned 48)` lines must match control, with no MISMATCH.
- exact.py must give ec2f4437f8e1ed09, the G1 digest 2f7fdccbd388225a, and the 9 long-prompt hashes must match. The kv-source caches are now written from new streams.
- An nsys trace read with `scripts/idxpar_trace.py` must show ovl=1.
- The boot memory lines must be compared against control (graph-pool growth plus one more cuBLAS workspace).
- Test IDXPAR on its own before stacking it with r1-pf. G01/G11a of r1-pf move fork points that IDXPAR now issues from the hi stream.
