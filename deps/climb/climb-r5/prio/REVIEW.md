# r5-prio: adversarial review

Reviewer: prio:review, climb round 5. Date: 2026-11-08 (America/Phoenix). Base: launcher 166d7aac.

**Verdict: FIX.** There is one small blocking item, in the device test. I found no correctness defect in the levers, the patches or the installer.

The estimates are a separate problem. The CMPPRIO model is optimistic for both variants, and the `serial` arm, which the kit ranks as priority 1 and says needs no gate, is probably worth about 0. Re-rank the arms before you spend a window on them (section 4).

## 0. What I ran and what I did not run

| check | where | result |
|---|---|---|
| sha256 of the 5 patched files on the live nodes (read-only) | spark-two to spark-eight | all 7 match KIT `base_sha256`: `0e4b6e91 1ad5c4e8 2f0bca4b aa8aad1b dd46a65e`. No `dsv41_prio.py` installed. No `CLUSTER_LOCK`. |
| same, spark-one | | **not verified**: the ssh connection to 192.168.0.238 timed out (directly and through spark-eight). The kit's 8/8 claim for spark-one rests on the author's earlier read. |
| base/ = live/ = KIT base sha; patched/ = KIT patched sha; `install.patched(base)` regenerated = KIT patched sha; `dsv41_prio.py` = `kit_sha256` | this Mac | all true; 30 tagged lines (6+12+4+4+4) |
| `tests/test_prio.py` | 512S1, exo-venv313, in a temporary directory that I deleted afterwards | **60 passed**; `device_test.py --cpu-selftest`: OK |
| composition with r5-ffnfront on `model.py` (the only file both kits patch) | this Mac, `ffnfront/edits.py` and `prio/install.py` applied in both orders | the anchors apply in both orders and the two results are byte-identical. With ffnfront installed first, the sha to review and add to `compose_sha256` is `6960efb2acd04421…` |
| t2 trace probes | 512S1 `~/climb-traces/t2/prod` (q0.2, q4.1, q0.3) | see sections 3 and 4 |
| device_test.py T0-T4 on a GPU | | **NOT RUN.** The cluster is read-only and I had no GPU. |
| any serving arm | | **NOT RUN** |

## 1. Flag-off identity: PASS

- Every inserted statement is guarded by a module-level boolean read once at import, and `dsv41_prio` is imported only under one of those guards.
  - In `dsv41_r3_dense.py`, the flag-off path only adds `_dsv41_pt = None` and two `False and ...` tests. These are host-only and emit nothing on the GPU.
  - In `engram.py`, the conditional expression evaluates the original statement first whenever the flag is off.
- The AST-identity test and the dynamic kernel/stream/event-sequence tests pass (I re-ran them). The shas match the live files on 7/8 nodes; spark-one could not be reached.
- An odd value such as `DSV41_PRIO_CMP=yes` turns on the file-level flags, but the module then sets `CMP_MODE=""`. `cmp_enter` returns None and the live `execute_in_parallel` runs, so the result is still inert.

## 2. Exactness when on: PASS (by argument and the CPU race checker; no device evidence yet)

What I checked beyond what the kit states:

- **All-gather outputs are fresh tensors, not views of a persistent rdma buffer.**
  - `rdma_all_reduce.py` `all_gather_stacked` and `all_gather_cols` both `torch.empty` their output.
  - So the hoisted wkv input `x` (a view of the rows all-gather output, kept alive by `rec.x`) cannot be overwritten by layer 0's own collectives while the side stream reads it.
  - This was the main race I went looking for, and it is not there.
- **Collectives stay on one stream.** rdma_ar's contract is "every rank issues the same rdma ops in the same order on ONE stream". All three levers keep every collective on main:
  - the CMPPRIO all-gather;
  - the hoist's rows all-gather (inside `embed`, on main);
  - the kv all-gather in `engram_take`.

  The hoist decision depends only on rank-symmetric inputs: the token count, the SP flag, the breakable-capture state and `EH.on`.
- **The FlashInfer workspace is safe.** The sm120 MXFP8 template (`mxfp8_gemm_template_sm120.h:142`) uses `cutlass::gemm::PersistentScheduler`, so there is no split-K or stream-K and the workspace is not written. Production already runs shared-expert `mm_mxfp8` on `aux_stream` at the same time as main-stream `mm_mxfp8`, using the same `mm_mxfp8_workspace`. The extra concurrency the hoist and SHPRIO add is therefore a pattern serving already exercises. `DSV41_DENSE_TACTIC`, which has its own global `_ws`, is not set in the launcher.
- **No fused-activation registry.** `as_quantized_activation` is type-based (`isinstance QuantizedActivation`), not a registry keyed on data_ptr. Issuing `wkv.quant_method.apply` early cannot pick up a stale fused activation.
- **The hoist path is valid in production.** `EngramBatch.create` returns None when `tp_size > 1`, so the production engrams (TP8) take the plain `embed` → `wkv` path that the hoist mirrors.
  - `ColumnParallelLinear.forward` in `linear.py` is stock: `quant_method.apply(self, x, bias)` followed by `tensor_model_parallel_all_gather(out)`. The wkv is built with `bias=False, gather_output=True, return_bias=False`.
  - The decoder layer gets `engram_hashes` as positional parameter 7 and `dsv41_sp` as a keyword, which matches the pre-hook.
  - No forward hooks change numerics on wkv or the engram. The L2-prefetch pre-hooks only record weights in learn mode; dsv41_oblit uses explicit `project()` calls in the model code.
- **The capture path is valid in production.** Serving uses full CUDA graphs: `VLLM_USE_BREAKABLE_CUDAGRAPH` is unset on the live process and the model is not torch.compiled. DTGRAPH captures only the drafter→target glue. Every fork and join therefore sits inside one capture.
  - Streams and events are created at construction or first eager use.
  - Side-stream allocations are reused only after the next `wait_stream(main)`.
- **The prefetch-wait shift costs little.** Hoisting `embed` makes layer 0's start wait on engram 1's host-memory prefetch, which the model forward started just before. The t2 `_engram_lookup_kernel` takes 3-14 µs (median 3.7-5.5 µs) and ends a few µs after layer 0 begins. The cost is at most a few µs per step and is not modelled.

## 3. Device test realism

- **BLOCKING (B1).** T2 does not assert the serving qkv_a GEMM.
  - Production runs `skinny_v2_kernel` (t2 trace: `g48x1x1 b256`, `DSV41_SKINNY_V2=1`).
  - T2 only checks `QKV_GEMM == "skinny"`. If `dsv41_r2_skinny.build` fails it logs a "note" and tests the v1 kernel.
  - Fix: when `r3.R2_V2` is set, a failed v2 build (`st_._r2 is None`) must be a T0/T2 FAIL (exit 1), not a note.
- T4 builds its own `QM.apply` from `mx_mm` and a mock `embed`, instead of running the serving `wkv.quant_method.apply` and `Engram.embed`. Exactness rests on "same call, same args", and that holds by construction. T4 really only tests the synchronization. Recommended: build the wkv through the serving MXFP8 linear method (K6144 → N3200 per rank), or at least assert `type(wkv.quant_method).apply` is the serving class.
- The T2 mutants (prio without the join; compressor wait dropped, serial and prio) run at M = 1..8, which is every served M (`QKV_MAX_M=8`). The T3 and T4 mutants run at M = 1, 2, 3, 4, 6, 8. That is adequate.
- T1 design is sound. It gates on `torch.cuda.graph` + `replay()`, the serving mechanism, and treats same/swap as must-fail mutants, with INCONCLUSIVE when the same arm looks honoured. **Expect FAIL.** torch's `CUDAGraph` instantiates without `cudaGraphInstantiateFlagUseNodePriority`, and without that flag a graph runs every node at the launch stream's priority. Plan as if `cmp-prio` and `shared` are inert in serving (exit 4). This is an expectation; I have not tested it.
- `t2_timing` cannot show the effect in section 4:
  - It measures "start → every branch joined", not the moment the main-side input of `_fused_save_compress_norm` (FSCN) is ready.
  - Its all-gather stub is `torch.cat`, which bears no resemblance to the rdma kernel (`ar_kernel_pl`: 1 CTA × 512 threads, 96 regs/thread).

## 4. Estimates against the t2 traces (non-blocking, but they change the arm order)

### CMPPRIO: the backlog moves to the next main kernel

One KV-source layer (q0.2, graph replay; times in µs from the compressor start):

```
  0.0- 46.9  Kernel2 8x8x16 (compressor; 32 thr, 124 regs, 16.9 KB smem)
 37.9- 51.3  skinny_v2 g48             <- waited ~37 us for the compressor backlog
 51.9- 74.2  ar_kernel_pl g1 b512 r96  (rdma qkv_a all-gather, main)
 74.5- 78.9  _q_kv_norm_quant g128x2
 88.9-118.1  _fused_save_compress_norm
```

- **Why the backlog moves.** In live, the all-gather kernel starts after the compressor has finished. With five resident compressor CTAs per SM (smem-limited: about 20 K registers), the all-gather's 49 K registers do not fit next to them. So the all-gather waits for the backlog to drain, exactly as skinny does today.
- **What that does to each variant.**
  - **serial** starts the compressor at skinny's end, i.e. right when `ar_kernel_pl` is issued on main, so the ~37 µs wait moves from skinny to the all-gather.
  - **prio** lifts only quant + skinny. The all-gather has to stay on main because of the rdma contract, so it faces the backlog too.
- **The model misses this.** `prio_t2.py` moves `Mx` (the `_q_kv_norm` end) earlier by the whole skinny wait `g`, which assumes the main kernels after skinny keep their live, post-backlog timing.
- **It also hides losses.** `gp`/`gs = min(g, max(0, S_F − F'))` clamps every instance at ≥ 0, so layers where the delayed compressor makes FSCN later never count as negative.
- **Revised view, an estimate only:**

  | variant | revised gain | why |
  |---|---|---|
  | serial | about 0 (−0.1 … +0.05 %) | the stall moves to the all-gather |
  | prio | ≤ half the model, only if T1 PASSes | the all-gather still meets the backlog |

- **Rank `cmp-serial` below `engram-hoist`.** It is exact and cheap to arm, but a flat result is the expected outcome, not a surprise.
- **Optional device check before a window.** Extend `t2_timing` with a resource-matched all-gather stand-in (1 CTA, 512 threads, ≥ 96 regs/thread, spinning ~20 µs) and a 128×2 norm after skinny. Report the main-side end (FSCN input), not the all-branch join.

### ENGRAM-HOIST

- **DRAM contention.** The wkv GEMM runs at ~234 GB/s, about 85% of GB10 DRAM peak, so it takes bandwidth from layer 0/13 almost one-for-one. The live L2 prefetch already prefetched about 6 MB of wkv in the `ag4` window; under the hoist that is lost, so the off-CP GEMM gets slower and the window goes to other weights.
- **Revised estimate.** I would put central near 0 to +0.1%, low −0.2% (the kit's low is fair). It is still the best of the three to arm, because it is the only lever whose CP removal (167-170 µs/step) does not depend on scheduler priority.

### SHPRIO

The kit concludes the shared-expert starvation is occupancy behind the one-wave FC1 and sits off the critical path; I agree. Standalone ≈ 0. Arm it only with GATE_FIRST, and only if T1 PASSes, which I expect it will not.

## 5. Installer and composition

- **Installer behaviour.**
  - Every file's sha and anchors are checked before any write, so a refused install leaves the node untouched.
  - Writes go through tmp + `os.replace`, with the mode copied from the backup.
  - `status` distinguishes CLEAN, INSTALLED and INCONSISTENT.
  - `uninstall` refuses edited files unless `--force`, which would also drop another kit's later edits; do not use `--force` on a composed node.
- **Minor (non-blocking).** If a write fails part-way, or the post-write `patched_sha256` check trips, the earlier files stay patched. Writes are not transactional. `status` reports this as INCONSISTENT and every edit is inert while the flags are off, so the risk is low.
- **Composition with r5-ffnfront.**
  - `model.py` composes in either order with an identical result (see section 0).
  - ffnfront does not patch `shared_experts.py`, and its GATE_FIRST path drives `se._stream` / `se._input_ready_event`, so the SHPRIO stream swap applies under GATE_FIRST.
  - Both kits refuse the other's patched sha until `compose_sha256` is reviewed and added. That is the right behaviour.
- **Composition with r1/r2/r3/r4.** The base shas are the live files with r1-attn, r2-skinny, r3-dense and ds41f-we2 installed. Uninstall in LIFO order: an older kit's uninstall will refuse a file that carries r5-prio lines.
- **Residual SHPRIO risk.** Moving the shared MLP off vLLM's global `aux_stream` removes rdma_ar's "refuse collectives on aux_stream" safety net.
  - A collective issued from the new stream would reach rdma on a second stream, and rdma_ar's own notes say that means a sequence desync and a 600 s hang.
  - `_reduces()` covers down_proj/w2 `reduce_results`.
  - Recommended: arm every r5-prio arm with `RDMA_AR_AUDIT=200`, and treat any `NEW_STREAM` line in rank0.log as a failure. That cheaply proves the "collectives stay on main" claim for all three levers.

## 6. Summary for the lead

1. **Fix B1.** Make the T2 serving-op check require skinny v2 when `DSV41_SKINNY_V2=1`. Then run `device_test.py` on a stopped spark-eight with the mandatory env. Expect exit 4 (gate FAIL).
2. **Arm order.**
   1. `engram-hoist`, with `RDMA_AR_AUDIT`.
   2. `cmp-serial`, expected flat.
   3. `cmp-prio` and `shared+GATE_FIRST`, only if T1 PASSes.
3. **Revise the CMPPRIO estimates in KIT.json and the README.** The model leaves out the all-gather backlog and clamps losses. Serial central should be about 0.

## Fixes (prio:fix, 2026-11-09)

### B1 (blocking): T2 now asserts the serving qkv_a GEMM. FIXED.
- **The old note is gone.** `device_test.py` no longer logs "note: r2 skinny v2 build failed ... the v1 skinny GEMM is under test" and continues.
- **New pure helper `skinny_serving_checks()`.** T2 records its results through `check()`, so any failure goes into `FAILS` and gives exit 1. The checks:
  - `QKV_GEMM == "skinny"` and the skinny ext loads (as before).
  - **New:** `DSV41_SKINNY_V2` is on (`r3.R2_V2`), as in launcher 166d7aac. The test still `setdefault`s it to 1.
  - **New:** with `R2_V2` set, the r2 build must return an object, and its `ok_a8` must cover every served M (1..`QKV_MAX_M`). If the build raises, the error goes into the check detail. If it returns None or covers only some M, that is a FAIL, because in serving those cases silently fall back to v1.
- **New dynamic check.** A spy wraps `st_._r2.e.skinny_v2_a8`, and `QkvAShard.__call__` runs eagerly at M = 1..`QKV_MAX_M`. Each call must launch `skinny_v2_a8` exactly once. This catches a `gemm_a8` that returns None at run time. The check uses its own CPU generator, so T2's weights and inputs are unchanged.
- **Tests.**
  - `cpu_selftest()` asserts the good case passes and seven degraded cases fail: build None, build raised, partial M, empty M, v2 off, not skinny, ext missing.
  - New CPU tests: `test_device_test_b1_*` (7 cases). Each degraded case drives `check()` and `exit_code()`, which must give 1 whether T1 says PASS or FAIL.
  - **Results, run on 512S1:** 67/67 passed and `--cpu-selftest` OK.
  - **Harness mutant:** with the pre-fix `device_test.py` swapped back in, the B1 tests give 7 failed and 1 passed (`TEST-RESULTS.txt`).
- **Not run:** the new T2 device checks. They run only on a GPU. T0-T4 are still NOT RUN.

### Non-blocking items handled
- **Estimates and arm order (section 4).** Both are revised in `KIT.json` (`expected_gain_pct`, `window_arms`) and in `README.md` (table, 1(A), 1(C), section 7, section 8). The old values are given in brackets.
  - **ENGRAM-HOIST** −0.2 / 0.05 / 0.6 (central was 0.15).
  - **cmp-serial** −0.1 / 0.0 / 0.05 (was 0.03 / 0.15 / 0.3). Expected flat, because the backlog moves to `ar_kernel_pl`.
  - **cmp-prio** −0.1 / 0.1 / 0.2, and only if T1 PASSes; otherwise 0 (was 0.05 / 0.2 / 0.45).
  - **SHPRIO** is unchanged.
  - **Combined central** 0.05 (was 0.35).
  - **New arm order:** 1 engram-hoist, 2 cmp-serial, 3 cmp-prio and shared+GATE_FIRST (both only on T1 PASS), 4 cmp+hoist. `prio_t2.py` is left as it was. Its two flaws (the full-g Mx shift and the clamp at ≥ 0) are written up in README 1(A), and the model is kept only for the record.
- **RDMA_AR_AUDIT=200 in every arm, control included.** This keeps the pairs matched; the audit is host-side only. The pass criterion goes in KIT `device_checks` and README section 7:
  - the arm's `NEW_STREAM` lines in rank0.log must match control's;
  - control has some too (main and capture streams), so the test is "no extra stream", not "no NEW_STREAM line at all";
  - an extra one is a failure, and the arm is stopped.
- **Installer.** README and KIT now say: uninstall in LIFO order; never `--force` on a composed node; writes are non-transactional and leave INCONSISTENT on a mid-way failure. Code is unchanged.
- **Composition.** The review's compose sha prefix `6960efb2acd04421…` (model.py, ffnfront first) is noted in KIT `files[model.py].base_note`. `compose_sha256` stays empty until compose-check prints the full sha against ffnfront's final kit.
- **T4 mock linear and t2_timing realism.** Both are recorded as open items in README section 8. They are not built: a serving-class MXFP8 wkv in T4, and a resource-matched all-gather stand-in plus FSCN-main-ready timing in t2_timing.
- **spark-one.** The read-only sha re-check was tried again in this pass and failed the same way: ssh to 192.168.0.238 timed out directly and "No route to host" via spark-eight. The README now says 8/8 by the author, 7/8 re-verified. Run `install.py status` on spark-one before installing.

### State after the fix
- **Kit levers and patches are unchanged.** The patched/ and files/ shas equal KIT.json. Changed files: `device_test.py` (da79961d55cdf4eb), `tests/test_prio.py`, `KIT.json`, `README.md`, `TEST-RESULTS.txt`.
- **Next step for the lead:**
  1. Run `device_test.py` on a stopped spark-eight with the mandatory env. Expect exit 4 (T1 FAIL); exit 1 now also covers a missing or partial skinny v2.
  2. Then run the arms in the new order with `RDMA_AR_AUDIT=200`.
