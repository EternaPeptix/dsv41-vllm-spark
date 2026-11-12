# r1-eager review (review-eager, climb round 1, 2026-11-08)

**Verdict: SHIP.** The kit is installable as a default-off lever for the window. No blocking items.
- G08 dtgraph: ship. Exactness when on is argued by construction but **unproven on device**. The window must prove it with exact.py ec2f4437f8e1ed09, G1 digest 2f7fdccbd388225a and the 9 long-prompt hashes.
- G22: not built (drop for this round).
- The central gain (+0.15%) is below the +0.3% promotion bar, so expect it not to promote.

## What I verified myself

| Check | Result |
|---|---|
| Live base on spark-one | `v1/worker/gpu/model_runner.py` sha256 `efb031733472a476...` = kit `base/` copy = KIT.json base. Also identical: sparse_swa.py `854c3e5f`, block_table.py `822b65f3`, buffer_utils.py `1f387c35`. |
| Patched result | `aa3e9d9e57462c40...` (matches KIT.json). Module `dsv41_dtgraph.py` `5afb75d429763256...` (top-level copy = patched copy). |
| Installer, scratch tree on 512S1 | dry prints 10 tagged lines, result `aa3e9d9e`. install `efb03173 -> aa3e9d9e`. Second install says "already patched". status shows PATCHED, module "(= kit)". After appending a foreign line, install **refuses** (rc=1, "untagged text is not the KIT.json base"). uninstall restores `efb031733472...` byte-exact and removes the module. |
| Tests (re-run on 512S1, torch 2.11 CPU) | `50 passed, 2 warnings in 0.97s`. The 2 warnings say the GroupCoordinator and Triton hooks are not installed there: no vllm or triton on that host. |
| Runtime transforms | The only sitecustomize is `~/dsv41-fc2/site/sitecustomize.py`, gated by `ATHENA_FC2_SERVING_TRIAL`. It does not touch model_runner.py. Nothing else monkeypatches `GPUModelRunner.execute_model`: r1-drafter wraps `sample_tokens`, so they compose. No other r1 kit edits model_runner.py. |

## Flag-off identity (diff, line by line)

There are 4 insertions, all tagged `# dsv41-r1eager`:
- **E1:** a module-level bool from `os.environ`, then `if False:` (the import and `_dsv41_dtg.ON` sit inside it).
- **E2:** `if _DSV41_DTGRAPH:` before `prepare_inputs`. It is in the non-dummy branch only.
- **E3:** `if _DSV41_DTGRAPH:` before `input_ids = input_batch.input_ids`.
- **E4:** at end of file, `if _DSV41_DTGRAPH:`, then `wrap_execute_model`.

With the flag unset, nothing is imported, nothing is wrapped and no hook is installed. The only runtime cost is two global-bool tests per execute_model.

There are no CUDA calls, allocations, streams, events or graph changes. The AST test (guarded blocks stripped = base AST) passed in my rerun. A value of `td` alone, or `0`, `off` or `no`, still evaluates to off. An unknown token raises at import, so boot fails closed.

## Exactness when on (analysis; unproven on device)

**What is captured.** The region holds this step's own Python glue. It is captured on the current non-default stream (thread_local mode, private pool) and launched once, at the next hazard or at the region end. Kernels, scalar arguments and order are the eager ones, and the graph is never replayed later. The glue is integer/copy work, plus the fp32 elementwise steer `mul` when steer is active, which is per-element and identical under capture.

**Capture-dependent branches in region callees.** I grepped the live tree. Every `is_current_stream_capturing()` site is in model forward or TD code: attention.py:1506 forward, flashinfer_sparse.py:107/955 decode remap, dsv41_steer.py:384 apply, sparse_attn_indexer.py:678/804, model.py, dsv41_r4_small.py, dsv41_l2_prefetch.py:560. None is in the DT glue except `dsv41_swa_batch.__enter__` (sparse_swa.py:1484).
- The kit arms that batch before capture.
- The inner `with dsv41_swa_batch()` in model_state.py:368 then sees `pending is not None` and becomes a no-op.
- The single flush therefore moves from the inner exit to `dt_end`, which immediately follows `model_state.prepare_attn`. Nothing in between reads the SWA outputs.

**torch.compile path.** `_assign_draft_token_budget_compiled` (adaptive_verification.py:509) would run eager (not inductor) under a dispatch mode. It is unreachable on captured steps: eligibility requires `dsv41_fused_av_on` (`_dsv41_fav == 1`, adaptive_verification.py:670), and that path returns before the compiled call (adaptive_verification.py:1083-1091). No other torch.compile is in the region callees.

**Host-to-device sources.**
- The decode path's CPU-sourced copies are all fresh `async_tensor_h2d` temporaries: model_runner.py:1442/1485/1514 and the fused-AV `packed`.
- `UvaBufferPool.copy_to_gpu` is a CUDA-side copy (model_state.py:412) and passes through.
- So the arena's snapshot-at-dispatch semantics equal the eager semantics.
- torch 2.13 enforces pinned sources under capture ("Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned", in libtorch_cuda.so). The arena satisfies this.

**Syncs and hazards.**
- `.item()` and `.tolist()` (live cap, fav check), `torch.cuda.synchronize` (adaptive_verification.py:991) and the SWAM table build (`torch.tensor(key).to(dev)`, sparse_swa.py:1458, a blocking H2D) are all classified as hazards. The segment is closed and launched before the sync runs.
- vLLM's own `set_stream` patch (torch_utils.py:810) is chained, not bypassed.
- Triton 3.7.1 hooks fire only on a compile miss (jit.py:862) and before a module load (compiler.py:465). Returning None does not suppress compilation.

## Races

- **Streams.** Everything stays on one stream, in program order. Any event, stream or graph-replay call latches the region off after closing the segment first, so a cross-stream consumer still waits on an event recorded after its producer.
- **Arena slots.** A slot is reused after 4 regions, only after an unhooked `Event.record` (fence) behind the region's last launch has completed. `wait_slot` runs before `capture_begin`.
- **Private-pool blocks.** They are reused only by later captures on the same stream, and each graph is launched before the next capture. torch defers event-tracked frees while a capture is underway.
- **Thread-local capture.** It lets the NCCL watchdog and the copy and rdma threads keep making their own calls. This build of torch has no "pending NCCL work" busy-wait in `capture_begin`: the string is absent from every torch .so.

## Non-blocking items (fix if cheap, or watch in the window)

1. **Gain.** The central estimate is +0.15% (0.035 ms/step), below the bar. Per-step instantiate and first-launch upload may eat it, which is why the `update` arm exists. The flag adds about 0.5-2 ms of host time per step (Python dispatch over every aten op in the glue, including CPU ops in the builders). That is fine at a CPU lead of 8.8 ms, but check G4 before any promotion.
2. **Arena aliasing guard is missing.** Any `torch.empty/empty_strided(..., pin_memory=True)` made inside a captured segment is served from the arena and silently recycled 4 regions later. A persistent pinned buffer lazily created there would be corrupted. None exists on the decode path today:
   - the only `empty(pin_memory)` sites are torch_utils.py:730 (an async_tensor_h2d temporary) and indexer.py:475 (PCP only);
   - UvaBuffer and CpuGpuBuffer use `torch.zeros(pin_memory=True)` (buffer_utils.py:33), which the kit does not intercept.

   A cheap hardening: serve PIN only while `async_tensor_h2d` is on the stack, or hazard instead.
3. **Pinned or device allocations that bypass the dispatch mode.**
   - `torch.zeros(pin_memory=True)` and `torch.tensor(list, pin_memory=True)` allocate real pinned blocks in capture, and those blocks are leaked by design (CachingHostAllocator.h:394-400). This only happens on PCP and prefill mm paths, which are excluded.
   - `torch.tensor(list, device=cuda)` does a blocking copy that the mode cannot see. It would raise under capture and abort the engine (loud). Its only region site is the lazy `_replay` init (model_state.py:387), which runs on the first, warm, eager call.
4. **Engine abort.** Any capture-illegal CUDA call that is not hooked aborts the step. The engine dies: the failure is loud, not silent. This can only happen in the window.
5. **Installer.**
   - `uninstall` restores `.bak-r1-eager` without checking the current file. That would drop a later kit's edits.
   - A stale `.bak` is never checked against the base sha.

   Both behaviours match the reference `install_dtac.py`, and no other r1 kit touches model_runner.py.
6. **Manifest.** `dsv41_dtgraph.py` is not in the supervisor manifest. Verify sha `5afb75d4` on 8/8 before arming. If it is missing, the flag-on boot fails with an ImportError (loud).
7. **Process-wide hooks when on.** Event.record/wait/query/synchronize, the Stream methods, `CUDAGraph.replay`, `set_stream`, the GroupCoordinator and rdma collectives and the Triton hook are wrapped globally. Each production call pays one thread-local lookup. An `Event.record`/`query` inside the glue would latch the region off every step and erase the gain: for example the step timers in async_utils.py:97-110, if enabled. Read the `log=500` hazard histogram.
8. **Memory.** Region temporaries now come from a separate graph pool: small, but extra 2 MiB segments.

## Window checklist (beyond KIT.json device_checks)

- Flags-off relaunch first, with both anchors.
- Each arm: anchors, plus a probe with `log=500`. Expect segments/region ≈ 1 and no steady `h2d_blocking`; a steady `h2d_blocking` would mean SWAM table-key churn under the private pool.
- No "DSV41_DTGRAPH disabled".
- The pinned-memory RSS of the worker stays flat over 1k steps. This confirms that no in-capture pinned allocations leak.
