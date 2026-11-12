# r1-eager: drafter->target glue as per-step CUDA graphs (G08 "dtgraph", `DSV41_DTGRAPH`)

Climb round 1, kit `r1-eager`, 2026-11-08. One flag, `DSV41_DTGRAPH`, default off. Only part of G08 is built:
the drafter->target region (DT, the README "dtgraph"). The target->drafter region (TD) is refused, and G22
(spec:C11) is skipped. Both are explained under "Not built" at the end.

| file | action | base sha256 | result sha256 |
|---|---|---|---|
| `vllm/v1/worker/gpu/model_runner.py` | patch (10 lines, tag `# dsv41-r1eager`, backup `.bak-r1-eager`) | `efb031733472a4769038668e4e04742f2156f90a582bbaaf474618f2ae3148cc` (8/8 identical on 2026-11-08) | `aa3e9d9e57462c40f8e42022ce291752c0fd673106ee20a1a134c4de56421a0a` |
| `vllm/v1/worker/gpu/dsv41_dtgraph.py` | new | - | `5afb75d429763256ee1a119af574718460d833f8207fa325dd2be1dbf0f6f566` |

`model_runner.py` is a supervisor manifest file, so install on 8/8: `python3 install.py install` (also
`uninstall | status | dry`). The installer refuses any `model_runner.py` whose sha256 is not the KIT.json base,
unless the file is already patched by this kit. `patched/model_runner.diff` shows the whole patch.

The brief named "eager_ops.py, segments.py" as glue modules. Neither exists in site-packages: they are the audit's
trace scripts (`work/audit-host/scripts/`). The real glue lives in `model_runner.py`, plus `buffer_utils.py`,
`input_batch.py`, `block_table.py`, `adaptive_verification.py`, `model_states/default.py`,
`models/deepseek_v41/nvidia/model_state.py` and the attention metadata builders. Only `model_runner.py` is
edited. `buffer_utils.py` (sha256 `1f387c35...`) is unchanged, and `dsv41_rel.py` is not touched.

## Mechanism

Today, after the drafter graph, each decode step issues ~34 eager ops on the main stream (s16) before the target
graph starts. The CPU issues them at least 8.8 ms ahead (audit-host 2a), so the gaps between them are GPU
front-end launch latency: 2.1-3.4 us per op, against ~0.54 us between nodes inside a graph (verify-G08 NOTES).

With `DSV41_DTGRAPH=1`, `execute_model` opens a capture region:
- **Start:** right before `self.prepare_inputs(...)` (`model_runner.py:1914` base, 1919 patched).
- **End:** right after `self.model_state.prepare_attn(...)`, before the inputs_embeds block (`:2034` base, 2040
  patched).
- **What it covers:** about 25 of the 34 ops, all on s16:
  - H2D idx_mapping and the packed fused-AV H2D;
  - `_dsv41_fused_av_kernel`, `_expand_idx_mapping`, `_prepare_pos_seq_lens`,
    `_combine_sampled_and_draft_tokens`, `_dsv41_live_cap`;
  - `_gather_block_tables`, `_compute_slot_mappings`;
  - the replay-start UVA copy, `_token_request_mapping`, the elementwise metadata ops;
  - 2 x (indexer decode metadata, paged-MQA metadata, D2D);
  - `_ring_slot_mapping`, and the single swam `_dsv41_multi_decode_swa` launch.
- **What stays eager:** the `sample_tokens` tail (draft_tokens index_put, AV confidence D2D, D2H on s68), the
  embedding with its `ar_kernel_pl` all-reduce and D2D, and `_gather_lookback`.

### Capture then launch once, every step

`dsv41_dtgraph.Region` starts a `torch.cuda.CUDAGraph` capture on s16 (`capture_error_mode="thread_local"`, one
private pool shared by all of the lever's captures). The unchanged Python glue then runs. At the region end the
graph is closed and launched once, on s16, at the point where the last eager op would have been queued.

The graph is built from this step's own Python code, so it holds exactly this step's kernels, grids and scalar
arguments (draft budget, live-cap tau, `num_tokens`, host branches such as `replay_start_np.any()`) in program
order. Nothing is replayed from an older step, so there is no stale-capture or variant-explosion problem and no
per-key graph cache. The cost moves to the host: capture plus instantiate plus Python dispatch per step, which
the ≥ 8.8 ms CPU lead absorbs.

`DSV41_DTGRAPH=update` keeps one executable graph per (segment, arena slot) and calls `cudaGraphExecUpdate`
(cuda.bindings 13.4.4) with the new capture instead of instantiating it. This avoids a fresh exec, and its
first-launch upload, every step. If the topology changed, the exec cannot be updated: the new graph is
instantiated and becomes the owner. Any binding error switches the update path off for good.

### Safety rails (host-side)

1. **Hazard exits.**
   - A `TorchDispatchMode` (active only inside the region) classifies every aten call. The following run
     eagerly and are never captured:
     - host syncs: `_local_scalar_dense`, `nonzero`, 1-arg `where`, `unique`, `repeat_interleave`, ...;
     - D2H copies;
     - blocking or dtype-casting H2D copies;
     - boolean-mask indexing;
     - CPU tensor arguments of device ops;
     - torch RNG on the device;
     - `record_stream`;
     - collectives (`c10d`, `vllm::*all_reduce|gather|...`).
   - Python-level hooks (flag-on only, inert outside a region: one thread-local lookup) cover:
     - `torch.cuda.Event.record/wait/synchronize/query`;
     - `Stream.wait_stream/wait_event/record_event/synchronize/query`;
     - `torch.cuda.synchronize` and `torch.cuda.set_stream` (which also covers `torch.cuda.stream()` and vLLM's
       `stream()`);
     - `CUDAGraph.replay`;
     - every `GroupCoordinator` collective and `RdmaAllReduce.all_reduce/all_gather_*`;
     - Triton `jit_cache_hook` (compile) and `kernel_load_start_hook` (module load).
   - On a hazard, the capture so far is closed and launched first. The hazard then runs eagerly, and capture
     resumes at the next aten op on s16.
   - Stream, event and graph hazards latch the region off: everything after them stays eager.
   - Program order on s16 is therefore always the eager order. A hazard only costs some of the saving.
2. **Pinned staging arena** (4 slots x 64 KiB, pinned once at arming).
   - Inside a capture, CPU-source H2D copies (`copy_` / `_to_copy`, non_blocking, contiguous, same dtype) are
     copied into the current slot and captured from there.
   - Pinned host allocations (`empty(pin_memory=True)`, `_pin_memory`, the `async_tensor_h2d` temporaries) are
     served from the slot, so no `cudaHostAlloc` runs during a capture.
   - Without the arena, torch 2.13 never recycles pinned blocks used inside a capture
     (`ATen/core/CachingHostAllocator.h:383-405`: "Leaking memory like this is intentional"). That would leak
     several pages per step.
   - A slot is reused only after the event recorded behind that region's last launch has completed. Four regions
     of slack against a CPU lead of 1-3 steps means this never blocks in practice; the stats count `slot_wait`.
3. **Eligibility.**
   - Captured: FULL-cudagraph decode steps without prefill, ubatching, PCP, DCP, LoRA, encoder inputs or
     encoder-decoder, with a single PP stage, and with `DSV41_FUSED_AV` on or no AV. The stock AV path calls a
     `torch.compile`d function.
   - A step shape `(num_reqs, num_tokens, graph tokens)` must first run eagerly `warm` times (default 2).
   - Anything else runs today's eager path untouched.
4. **SWA batch.**
   - `dsv41_swa_batch` (`sparse_swa.py:1483`) switches itself off while a stream is capturing.
   - So `dt_begin` enters the batch *before* `capture_begin`. The inner `with dsv41_swa_batch()` in
     `model_state.prepare_attn` then nests (no-op), the builders queue into it exactly as today, and `dt_end`
     exits it while still capturing. The single fused SWA launch is issued as the last node of the graph, at the
     same place the inner block issues it today.
   - The new-layout table upload inside the flush (`torch.tensor(key).to(dev)`, a blocking H2D) is a hazard and
     runs eagerly.
5. **Exceptions.**
   - `wrap_execute_model` (flag-on only) wraps `GPUModelRunner.execute_model`. If anything raises while a region
     is open, `dt_abort` drops the open capture, pops the dispatch mode, clears a pending SWA batch, disables the
     lever and re-raises: the step fails loudly. Later CUDA work is never silently captured.
   - Any failure before the first `capture_begin` disables the lever and keeps the step eager.
   - An unknown flag token fails the import, so a typo fails the boot closed.

## Why flag-off is identical

- **The patch adds 10 lines** (see `patched/model_runner.diff`):
  - `_DSV41_DTGRAPH = <env test>`: one host string comparison at import;
  - `if _DSV41_DTGRAPH:` around the module import;
  - the two calls in `execute_model`;
  - the class wrapper at module end.
- **With the flag unset:** no new import, no hook, no mode, no stream, event, allocation or launch; the region
  calls are `if False:`. Kernels, launch order, streams, events, allocations and graph contents are unchanged.
- **Tests:**
  - `test_flag_off_ast_identical` removes the kit's flag assignment and every `if _DSV41_DTGRAPH:` block from the
    patched AST and gets a dump identical to the live base file;
  - `test_inserted_statements_are_all_guarded` checks that every tagged line sits under that guard;
  - `test_flag_expression` checks that unset, "", "0", "off", "no" and "false" all evaluate False.
- **Runtime transforms:**
  - The only runtime source transform in the serving stack is `~/dsv41-fc2/athena_fc2_forty_layer_trial.py`
    `transform_apply`, loaded via `~/dsv41-fc2/site/sitecustomize.py` (fail-closed `os._exit(97)`). It rewrites
    `DeepGemmFP4Experts.apply` only, so none of its anchors touch `model_runner.py`.
  - Nothing else monkeypatches `GPUModelRunner` methods (grep on spark-one).

## Why flag-on is bit-exact, op by op

The region launches the same kernels, with the same arguments, in the same order, on the same stream, as the
eager path would. Only the launch mechanism differs (graph nodes, plus memcpy nodes for copies). Op by op:

- **Kernel ops** (all integer, index or copy work: fused AV, expand_idx, pos/seq_lens, combine, live cap, block
  tables, slot mappings, token->request map, indexer and paged-MQA metadata, ring slot, SWA):
  - captured from the identical Python call, with the same grid, block and scalar arguments;
  - device pointers are the same buffers, or private-pool temporaries whose contents are produced by the same
    kernels;
  - the same arithmetic runs in the same order, so every output element gets the same op sequence.
  - No GEMM, reduction or float op is in the region except `_dsv41_live_cap` (float compares on confidences,
    unchanged).
- **H2D copies** (idx_mapping, packed AV, replay-start UVA):
  - the destination receives the same bytes;
  - the staged source is a byte copy taken at capture time;
  - the eager source is a pinned temporary written once and never touched again, or a UVA pool slot, which is
    read in place in both modes.
- **D2D copies** (indexer metadata): memcpy nodes, same bytes.
- **Hazard ops:** run eagerly, after everything captured before them has been launched, so they are ordered
  exactly as in eager mode.
- **Not changed:** no verify width, tile, tactic, reduction order or draft change. RNG ops are hazards, so torch
  Philox offsets advance exactly as in eager mode.
- **Code paths that change under capture.** Every `is_current_stream_capturing()` in code the region reaches was
  checked:
  - `sparse_swa.py:1484` is handled by entering the batch before the capture starts;
  - `dsv41_steer.py:384`, `attention.py:1506`, `flashinfer_sparse.py:107/955` and `model.py:360/405/1040/2037`
    run in the forward (inside the target graph), not in the region;
  - `prepare_v2` (steer) has no capture branch: its pageable H2D copies are hazards.

## Race analysis

- **Single stream.** All captured work is on s16, and so is everything that reads it (target graph, sample,
  drafter graph). Each graph is launched at the point where its first op would have been queued, so the stream
  order matches eager mode exactly. No new cross-stream consumer is created. Copies to s20/s68 are outside the
  region; a stream switch inside the region latches capture off.
- **Private pool reuse.** Blocks of the lever's private pool freed after a capture are reused only by later
  captures of the same pool. Those launch later on s16, after every consumer of the old tensor, which also runs
  on s16. Cross-stream frees go through the allocator's events, as today.
- **Arena slots.** Reuse is fenced by a real event recorded eagerly behind each region's last launch.
- **UVA pool slots.** These are read by the graph at the same stream position as by the eager copy. The existing
  `UvaBufferPool` reuse rule (2 slots, retire readers before reuse) therefore holds unchanged.
- **Graph lifetime.** Graph objects (and their execs) are kept in an 8-deep ring.
  `cudaGraphExecDestroy` on an in-flight exec frees it asynchronously on completion (CUDA driver API docs).
- **update arm.** It updates the exec of slot k only after the slot-k event has completed, so the exec's previous
  launch has finished. It does not rely on ExecUpdate semantics for in-flight launches.
- **Other threads.** Thread-local capture mode leaves the output thread's event syncs unaffected. The Python
  hooks act only when the calling thread owns an open region.

## Expected saving

- **DT gaps today** (verify-G08, 6 d200 windows, ranks 0 and 4): 88-91 us of inter-op gaps plus a 10.5-10.8 us
  graph-start gap per step.
- **This kit removes the gaps between ~25 consecutive ops.**
  - Eager: ~25 x 2.7 = ~67 us.
  - Graph: one extra graph-start gap (~3.7 us for a small graph, as measured at the drafter-graph start) plus
    24 x 0.54 us, plus ~5-10 us extra for the 5 memcpy nodes, so ~25 us.
  - Gross saving: ~0.04 ms/step.
- **Deductions:**
  - instantiate mode pays an unknown first-launch upload of a fresh exec on the stream, maybe 5-20 us;
  - nsys inflates eager gaps by 0-25 %.
- **d:** 0.00 / 0.035 / 0.06 ms, which gives **G1 +0.0 / +0.15 / +0.26 %**. The low end includes "upload eats
  the gain" for the instantiate arm; the update arm should sit nearer the central or high value.
- **Against the promotion bar:** the central value is *below* the window's +0.3 % bar. The other ~45 % of the
  audited G08 saving is the TD region, which is not exact under capture (see below).
- **CPU cost:** dispatch-mode Python for ~100-200 aten calls, plus capture and instantiate, is an estimated
  0.5-2 ms of host time per step. It is hidden at C1 (lead ≥ 8.8 ms), but the lead must be re-checked under
  nsys.

## Reading the effect in an nsys trace

- **Graph launches per step:**
  - flag off: 2 `cudaGraphLaunch` per step (target 1914 nodes, drafter 185 nodes);
  - flag on: 3 per step; the third is a ~25-node graph launched just before the target graph.
  - Group by correlationId as in `verify-G08-combined/seg.py`.
- **DT segment** (drafter graph end -> target graph start):
  - eager ops on s16 fall from ~34 to ~9: the sample tail, embedding, `ar_kernel_pl`, D2D, `_gather_lookback`;
  - inter-op gaps inside the small graph are ~0.5 us;
  - DT idle should fall from ~100 us to ~50-60 us per step.
- **Host:**
  - capture shows up as `cudaStreamBeginCapture` / `EndCapture` / `cudaGraphInstantiate` (or `cudaGraphExecUpdate`
    in the update arm) on the worker thread;
  - check that the CPU lead (end of the target `cudaGraphLaunch` API call -> first target kernel) stays ≥ 5 ms.
- **A fresh exec's upload**, if visible, appears as a small gap or "graph upload" activity right before the small
  graph's first node. Compare it between the 1 and update arms.
- **Probe arm** `DSV41_DTGRAPH=1,log=500`: rank 0 logs `segments/region` (should be ~1.0), `staged`, a hazards
  histogram and skips. A steady non-zero hazard means some glue op cannot be held. Report the reason key.

## Window arms

| arm | env | notes |
|---|---|---|
| dtg | `DSV41_DTGRAPH=1` | instantiate-per-step |
| dtg-update | `DSV41_DTGRAPH=update` | cudaGraphExecUpdate into per-(segment, slot) execs; falls back to instantiate |

Exactness per arm: exact.py `ec2f4437f8e1ed09`, G1 digest `2f7fdccbd388225a` on 2 reps, and the 9 long-prompt
hashes. The flags-off relaunch must reproduce both values first.

## Risks

- **Unproven on device.** Exactness holds by construction, but no GPU was available; the window's digest and
  exact checks are the proof. A CUDA error during a capture (an unsafe API call that none of the hooks catch, for
  example a C++ extension that synchronises internally) aborts the step and stops the engine, loudly. It never
  produces wrong bits silently.
- **Out-of-dispatch pinned allocations.** `torch.tensor(list, pin_memory=True)` pins outside the dispatcher (seen
  on torch 2.13), so a list-sourced `async_tensor_h2d` inside a capture would still allocate pinned memory there.
  Its copy is staged through the arena, so the bytes are right. The only such sites are PCP / mm-prefix paths
  (`indexer.py:491/498`, `attention/backends/utils.py:100`), which are not on the decode path.
- **Gain may be small or negative.** In the instantiate arm, per-step instantiate plus first-launch upload could
  cancel the saving. That is why there are two arms.
- **Host time.** Under heavy concurrency the CPU lead shrinks. The added host time could then reach the critical
  path (G4 should be checked in the window).
- **Stacking.** Stacking with a future lever that moves step-N+1 prep to a side stream (G22, H4b) is safe:
  stream switches latch off. The two savings do not add up.

## Not built

- **G08 target->drafter region (`td`, refused by the parser).**
  - The TD glue takes different code paths while a stream is capturing:
    - `dsv41_r4_small.target_logits` returns None under capture (`dsv41_r4_small.py:835`, also `:584`, `:957`,
      `:976`), so the certified head would be replaced by the plain head;
    - `dsv41_oblit.py:83` and the DSpark main_proj folds (`dsv41_boundary.py:269`, `:498`) switch path as well.
  - Capturing TD would therefore change arithmetic. It also holds 3 collectives (argmax all-gather, main_proj AR)
    and the s20 AsyncOutput copy, which would split it into 4+ segments.
  - Building TD needs "capture but behave as eager" switches in those model-owned modules (and a device
    replay-compare); they are outside this kit's files.
- **G22 / spec:C11 (skipped).**
  - The drafter shares `BlockTables` (`spec_decode/speculator.py:298`), and `dflash/speculator.py:429-436`
    writes the draft query slot mapping into `block_tables.slot_mappings[gid]`, which the DSpark graphs read at
    replay.
  - Running the next step's expand_idx..SWA prep on a side stream during the draft graph would therefore race the
    drafter's KV writes, unless the draft KV group gets private `slot_mappings` / `input_block_tables` and the
    DSpark graphs are recaptured on them.
  - Both changes are in drafter-owned files (`speculator.py`, `dflash/speculator.py`, `dspark.py`), which this
    kit may not edit.
  - What it needs:
    1. private draft slot-mapping / block-table buffers allocated by the speculator, plus DSpark graph recapture
       on their addresses;
    2. a CPU formula for qsl / cu_num_logits when `num_reqs == 1` (`cap = min(budget, sched)`,
       `adaptive_verification.py:268-290`);
    3. a side stream that waits on an event after `prepare_dflash`, with combine, live cap, embedding, AR and
       lookback-copy joined after the draft graph;
    4. a full audit of the buffers the draft graph reads, plus per-step buffer dumps.
  - Its marginal gain on top of this kit would also shrink, since the prep it overlaps is now cheaper.

## Tests

- `tests/test_install.py`:
  - anchors apply with exact counts to the live base;
  - dry / install / idempotent re-install / status / uninstall (byte-identical restore) in a scratch tree;
  - refusals on a drifted base;
  - flag-off AST identity, guard coverage and flag-expression truth table;
  - region placement around the glue.
- `tests/test_dtgraph.py`:
  - flag parser;
  - op classifier table (meta tensors stand in for device tensors);
  - arena slots, alignment and fullness;
  - pinned-alloc service;
  - segments and ordering with a fake graph backend: hazard -> launch -> eager -> resume; latch; other-stream
    no-resume; max segments; slot cycling and waits; re-entry refusal;
  - real torch ops through the dispatch mode: H2D staging and the `async_tensor_h2d` pattern;
  - eligibility table;
  - `dt_begin`/`dt_end` with a fake runner, checking that the SWA batch is armed before the capture and flushed
    inside it;
  - `wrap_execute_model` abort and re-raise;
  - hook install idempotence and inertness outside a region.
- Run on 512S1 (torch 2.11 CPU): **50 passed**. A read-only API probe against the production torch 2.13 on
  spark-one (`CUDA_VISIBLE_DEVICES=` so no GPU is touched, script piped over stdin, no files written) confirmed:
  the CUDAGraph API (`keep_graph`, `capture_begin(pool, capture_error_mode)`, `raw_cuda_graph(_exec)`,
  `instantiate`), the dispatch classification of the `async_tensor_h2d` pattern, the `cudaGraphExecUpdate(exec,
  graph)` binding and the Triton hook names. Details are in `TEST-RESULTS.txt`.
