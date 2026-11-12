# r5-prio: graph stream-priority gate and three scheduling-only levers

Climb round 5, kit `r5-prio`. Base: production launcher **166d7aac**. All five patched files were confirmed identical on 8/8 nodes by the author (2026-11-08, read-only). The review re-verified spark-two to spark-eight; spark-one was unreachable both to the review and to the fix pass (ssh to 192.168.0.238 timed out, directly and via spark-eight), so re-check it with `install.py status` before installing.

The kit has two parts:
- A device microtest that answers one question: **does a CUDA graph captured by torch honour stream priority?** The answer gates the two priority levers.
- Three default-off levers. Each one changes only which stream a kernel runs on, that stream's priority, or one event edge.

| lever (in arm order) | flag | needs the T1 gate? | t2-based gain (G1 %, low / central / high), revised after review |
|---|---|---|---|
| 1. Engram wkv GEMM hoist | `DSV41_PRIO_ENGRAM_HOIST=1` | no | −0.2 / **0.05** / 0.6 (was 0.15) |
| 2. CMPPRIO, ordering variant | `DSV41_PRIO_CMP=serial` | no | −0.1 / **0.0** / 0.05 (was 0.15; expected flat) |
| 3. CMPPRIO, priority variant | `DSV41_PRIO_CMP=prio` | yes | −0.1 / **0.1** / 0.2 if T1 PASSes, else 0 (was 0.2) |
| 3. Shared experts on a high-priority stream | `DSV41_PRIO_SHARED=1` | yes | −0.1 / **0.0** / 0.15 (keep only as GATE_FIRST insurance) |

Combined central: about **0.05%**, because T1 is expected to FAIL (see section 8), which leaves `prio` and `shared` inert in serving. The revisions follow the adversarial review (`REVIEW.md` section 4 and Fixes). The model in section 1(A) is kept for the record, but it overstates both CMPPRIO variants.

Scale: 1% G1 ≈ 0.21 ms per step. Every estimate is re-based on the t2 traces of today's production (`512S1:~/climb-traces/t2/prod`, 6 traces: code, prose and maths on ranks 0 and 4). The script is `scripts/prio_t2.py` and its output is `out/prio_t2.txt`. It reuses the round-4/5 `cp4.py` critical-path walker and the `cls4.py` classifier unchanged.

## 1. Re-base on t2 (what changed against round 4)

### (A) qkv_a vs the compressor (CMPPRIO)
- **Layers affected.** The KV-source layers 2/8/14/20 run the replicated BF16 compressor `torch.mm(hidden, fused_wkv_wgate.T, fp32)` on aux stream 0. It is a cuBLAS `Kernel2` with grid 8x8x16 (8x4x16 in one layer): about 1024 CTAs and about 41-42 µs. The qkv_a quant and the 48-CTA skinny shard GEMM are issued on main from the same fork. The compressor's CTAs are dispatched first, so the skinny GEMM waits behind the whole backlog.
- **The stall.** With a compressor running, skinny waits **127-135 µs/step** (4.0 waits/step). Of that, **83-94 µs/step** is a critical-path (CP) gap. Waits without a compressor running total 27-31 µs/step.
- **Where the two branches merge.** `_fused_save_compress_norm` (FSCN) consumes both branches. It sits on the CP in 55-70% of instances, for 50-70 µs/step. On average the compressor branch is ready 23-27 µs *before* main, so main is the late input.
- **Model.** The model assumes skinny waits 2.5 µs, and that the compressor is delayed by the skinny duration (prio) or by quant→skinny end (serial).

  | variant | FSCN-ready earlier (µs/step) | counting only where skinny or FSCN is on the CP |
  |---|---|---|
  | prio | 46.9-52.7 | 43.3-51.7 |
  | serial | 35.0-40.5 | 32.7-39.4 |

- **Round-4 comparison.** Round 4 quoted about 125 µs/step. That was the raw wait, not CP time.
- **Why the model overstates the gain (review, t2 q0.2).** In live, the qkv_a rdma all-gather (`ar_kernel_pl`: 1 CTA × 512 threads, 96 regs/thread) starts only after the compressor has drained. With about 5 resident compressor CTAs per SM, it does not fit next to them. The all-gather must stay on main because of rdma_ar's one-stream contract.
  - **serial** starts the compressor at skinny's end, which is when the all-gather is issued, so the ~37 µs backlog moves from skinny to the all-gather.
  - **prio** lifts only quant + skinny, so the all-gather still meets the backlog.
  - `prio_t2.py` also shifts the `_q_kv_norm` end earlier by the whole skinny wait and clamps per-layer losses at ≥ 0.
  - **Revised:** serial about 0; prio at most half the model, and only if T1 PASSes.

### (B) Shared-expert act_and_mul "starvation" behind FC1 (SHPRIO)
- **The starvation is real.** In 19-22 of 40 MoE layers per step, `act_and_mul` starts more than 10 µs after its gate_up GEMM (N640) ends. The mean wait is 39-91 µs.
- **Cause: occupancy, not queue order.** FC1 has grid 48x1x1, i.e. one CTA per SM, running 85-140 µs. N640 ends 12-18 µs after FC1 started, and act_and_mul starts about 32 µs before FC1 ends, i.e. at the first FC1 CTA retirement. Stream priority cannot preempt a running CTA. T1 includes a one-wave persistent-flood diagnostic that shows this on the device.
- **It does not delay anything that depends on the shared output.** The only consumer is the FC2 gather-add (`take()` waits on `_output_ready_event`).
  - In every starved layer, the shared down GEMM ends **31-38 µs before FC2 ends at the closest**, and 46-81 µs before on average.
  - The total gather-add join wait is 22 µs/step over 40 layers, which is launch latency. Only 0-0.02 layers per step wait more than 5 µs.
  - The cp4 edge "gemm_mxfp8_N640->moe_act" (44-92 µs/step, the 31-96 µs/step quoted in the task) comes from the walker's latest-consistent-predecessor heuristic. It is not a true data dependency.
- **The ordering alternative is ruled out.** Making FC1 wait for the shared chain costs 2.4-3.8 ms/step. Waiting for act+quant only still costs 1.4-1.7 ms/step. The shared chain is slow (N640 runs on 5 CTAs) and FC1 is normally ready 40-125 µs earlier. Neither ordering variant was built.
- **What is left.** The lever is a dedicated high-priority stream for the shared experts. Standalone it should be about 0. Its job is insurance for r5-ffnfront **GATE_FIRST**: GATE_FIRST issues the router before the shared branch, so FC1 starts earlier and the shared chain may become late at the gather-add. Test SHPRIO only together with GATE_FIRST, against a GATE_FIRST-only control.

### (C) Engram wkv GEMM (ENGRAM-HOIST)
- **Cost on the CP.** Layers 1 and 14 run the wkv local GEMM (MXFP8, K 6144 → N 3200 per rank; device_kernel grid 1x25). It takes **83.6-85.2 µs each, 167.5-170.3 µs/step, all on the CP**, plus its activation quant at 4.5-5.9 µs.
- **The all-gathers around it.** The rows all-gather costs 41.9-46.2 µs/step and the kv all-gather 48.9-69.5 µs/step.
- **Why the realisable gain is small.** The GEMM reads about 19.7 MB of fp8 weight at about 234 GB/s, so it is DRAM-bound. Overlapping it with DRAM-bound decoder work saves little. Only the parts of the host layer that do not use DRAM fully gain: collective spins, which the paced L2 prefetch already partly uses, and latency-bound small kernels.
- **DRAM contention and the lost prefetch (review).** At ~234 GB/s the GEMM uses about 85% of GB10 DRAM peak, so it takes bandwidth from layer 0/13 almost one-for-one. The live paced L2 prefetch already pulls about 6 MB of wkv in the `ag4` window; under the hoist that is lost. Revised central: 0 to +0.1%. It is still the arm to run first: it is the only lever whose CP removal (167-170 µs/step) does not depend on scheduler priority. Hoisting `embed` also makes layer 0 wait on engram 1's host prefetch (t2 lookup 3-14 µs), a few µs at most.
- **Risk.** The GEMM's 25 CTAs hold 25 SMs for about 84 µs. A side stream cannot be given a lower priority than main, because main already runs at the least priority (0). If the host layer's FC1 (48 CTAs, one wave) launches while the GEMM occupies 25 SMs, FC1 waits. That is why the low estimate is negative.

## 2. Mechanisms

### CMPPRIO (`attention.py` + `dsv41_r3_dense.py` + `dsv41_prio.py`)
`DeepseekV4Attention._run_parallel_input_projections` calls `dsv41_prio.cmp_parallel_input_projections` only for layers with a compressor. That function keeps the live enqueue order (aux first) and hands the r3 shard to the two hooks in `QkvAShard.__call__`:

- **prio:**
  1. `cmp_enter`: the high stream waits on main (`wait_stream`), then the quant and the skinny GEMM (or the r2 QFUSE fused kernel) run on the high stream.
  2. `cmp_leave`: an event is recorded on the high stream, the context returns to main, and main waits on that event.
  3. The rdma all-gather runs on **main**, exactly as before.
- **serial:** all kernels stay on their live streams. `cmp_leave` records an event on main right after the shard GEMM, before the all-gather. The compressor is enqueued after `default_fn` on aux stream 0 and waits on that event.
- **Fallback.** The live `execute_in_parallel` call runs (identical behaviour) whenever the shard path does not apply:
  - M > `DSV41_QKV_A_SHARD_MAX_M`;
  - the stock GEMM path;
  - first use inside a capture, where no event or stream can be created;
  - an unknown mode value.

### SHPRIO (`shared_experts.py`)
`SharedExperts.__init__` replaces `self._stream = aux_stream()` with `dsv41_prio.shared_stream(...)`. That returns a dedicated greatest-priority stream, separate from the CMPPRIO stream so the two levers never add a stream-order edge between each other.
- Everything else is unchanged: the same events, the same `maybe_forward_async`/`wait`, the same fold-glue path.
- It refuses, keeping aux_stream, if the shared MLP reduces its own output. The reason: rdma_ar refuses collectives only on vLLM's global aux_stream (`rdma_all_reduce.py:282`).
- It does not touch `moe_runner.py`, so it composes with any launch order ffnfront chooses.

### ENGRAM-HOIST (`engram.py` + `model.py` + `dsv41_prio.py`)
`DeepseekV4Model.__init__` calls `engram_install`, which:
- validates each engram:
  - non-batch path;
  - not SP;
  - wkv is a plain `ColumnParallelLinear` with `gather_output` and no bias return;
  - rows are TP-gathered;
- validates the decoder-layer signature (`engram_hashes` must be positional parameter 7);
- registers a forward pre-hook on each host layer (default `1:0,14:13`).

At each host layer's start, for forwards of at most `DSV41_PRIO_ENGRAM_MAX_TOKENS` (128) tokens, outside SP and outside a breakable capture:
1. On main: `Engram.embed(hash_ids)`, i.e. the prefetch wait plus the **rows all-gather, same op and stream**.
2. A least-priority side stream waits on main and runs `wkv.quant_method.apply(wkv, x, None)`. This is the exact matrix multiply that `ColumnParallelLinear.forward` runs.
3. A done event is recorded.

`Engram.forward` replaces `self.wkv(self.embed(hash_ids).flatten(-2))` with `self._dsv41_prio_take(hash_ids)`:
- If something was hoisted: main waits on the done event, then runs `tensor_model_parallel_all_gather(out)` on **main, where `ColumnParallelLinear.forward` issued it**.
- If nothing was hoisted: it runs exactly the live statement.

A model forward hook joins any hoist nobody consumed (keeping a capture legal), logs an error, and switches the lever off for good.

## 3. Why flag-off is identical
- Every inserted statement sits under one of three module-level booleans, read once at import (`_DSV41_PRIO_CMP`, `_DSV41_PRIO_SHARED`, `_DSV41_PRIO_EH`). With the env vars unset, `dsv41_prio` is never imported.
- `tests/test_prio.py::test_flag_off_is_ast_identical` constant-folds these flags to False and drops the kit's flag definitions and helper defs. Each of the five patched files is then **AST-identical to the live file**.
- The dynamic tests run the live and the patched functions (flag off) side by side under the fake CUDA layer. They emit the **same kernel / stream / event sequence**. This holds for attention+r3 at M = 1, 4, 8, 9 with and without QFUSE, for shared_experts, and for engram.

## 4. Why flag-on is bit-exact, op by op
- **CMPPRIO.** The same kernels run on the same operands: the FlashInfer MXFP8 quant, skinny v1/v2 or QFUSE, `torch.mm` BF16→fp32 for the compressor, the indexer `weights_proj`, and the rdma all-gather. No tile, tactic, reduction order or rounding point changes; only the stream or the event edges do. The serial variant only adds a wait.
- **SHPRIO.** Identical kernels; only the `cudaStream` object differs.
- **ENGRAM-HOIST.** `embed` is the same function. `quant_method.apply` is the call `ColumnParallelLinear.forward` makes, with the same input tensor and `bias=None` (`skip_bias_add` is False and bias is None). The all-gather is the same function with the same default `dim=-1` on the same tensor. The kv tensor reaching `_fused_engram_post_wkv_kernel` is therefore the same bytes.
- **Two side streams on the FlashInfer workspace.** The process-wide `mm_mxfp8_workspace` buffer is shared. The sm120 template uses `cutlass::gemm::PersistentScheduler`, which has no split-K and no stream-K, so the workspace is not written. Device tests T3 and T4 also run concurrent MXFP8 GEMMs on two streams and compare bitwise.

## 5. Race analysis
Each cross-stream consumer waits on an event recorded after its producer:
- **CMPPRIO prio:** y → the all-gather. Main waits on done after the GEMM.
- **CMPPRIO serial:** the compressor depends only on hidden_states, which it already waited for via start_event. The event adds ordering, not a data dependency.
- **SHPRIO:** the live events, unchanged.
- **ENGRAM-HOIST:** x is read on the side stream after `side.wait_stream(main)`, and `out` is read on main after `wait_event(done)`.

Allocator safety. No `record_stream` is used (it is awkward under capture). The argument per lever:
- Blocks allocated on a side stream (y and the quant buffers; the hoisted `out`) are only reused by that same stream.
- The next use of that stream always starts with `wait_stream(main)`: `cmp_enter`, or the next hoist. That point is after every main-stream reader.
- The hoisted `x` (main pool) is kept alive by the hoist record until the join.

All branches are joined before `cmp_parallel_input_projections` returns, at the gather-add (shared), or at the take or the forward-end hook (hoist). A capture never ends with an open branch.

The CPU vector-clock checker found no race. It catches the missing-join mutant of each lever, and four kit mutations: no `side.wait_stream`, no serial event record, no `hi.wait_stream`, no take join (see TEST-RESULTS.txt).

CUDA-graph safety:
- Streams and events are created at model construction or on first eager use, never inside a capture. A first use inside a capture falls back to the live path.
- There are no host syncs.
- Allocations inside capture are the same tensors as today, in different stream pools.

## 6. Device test (`device_test.py`, run by the lead on a stopped spark-eight)

```
ssh spark-eight 'source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda; \
  cd ~/climb-scratch/r5-prio && python3 device_test.py --json t.json'
```

Copy the kit folder to `~/climb-scratch/r5-prio` first, or use `--source installed` after `install.py install`. No compile step is needed: Triton JIT-compiles the spin kernels and the skinny `.so` is the live one. Expected runtime is about 2-4 minutes.

T0 refuses to run unless all of the following hold:
- `CUDA_HOME=/usr/local/cuda`, `nvcc` comes from `/usr/local/cuda/bin`, and the interpreter is venv vllm032;
- the device is GB10 sm_121 with a stream priority range of more than one level;
- `has_flashinfer()` is True, i.e. the serving MXFP8 quant, not the torch fallback;
- the modules under test carry the KIT.json sha256;
- (T2, review B1) the qkv_a GEMM under test is the serving one: `QKV_GEMM == "skinny"`, `DSV41_SKINNY_V2` on as in launcher 166d7aac, the r2 v2 build returns an object whose `ok_a8` covers every served M (1..`QKV_MAX_M`), and a call through `QkvAShard` launches `skinny_v2_a8` exactly once at every served M. A v2 build that raises or returns None is a FAIL (exit 1), not a note: the serving code would silently keep v1, and the test would then check the wrong kernel.

**T1 (the gate).** A Triton spin kernel stamps `%globaltimer` and `%smid` per CTA. The flood is about 8 waves of 512-thread CTAs (3 per SM). The probe is 48 CTAs of 256 threads. P is the fraction of flood CTAs that start after the probe's median start: about `1-1/waves` if priority is honoured, about 0 if not.

| arms | role |
|---|---|
| eager hi / same / swap | gated |
| `torch.cuda.graph` + `replay()`, flood captured first: hi / same / swap | gated |
| graph with the probe captured first | diagnostic |
| captured kernel-node priority attributes and a re-instantiation with `cudaGraphInstantiateFlagUseNodePriority`, both through `cuda.bindings` | diagnostic: tells whether torch's instantiation is what drops priority |
| one-wave persistent flood | diagnostic: confirms there is no preemption |

The same and swap arms are the must-fail mutants.

**T2-T4.** Bitwise checks of every lever against the live path, using the real r3 `QkvAShard` with skinny v2 (required, see above), a real `SharedExperts`, and FlashInfer cutlass MXFP8 GEMMs. Each lever runs eager and in CUDA-graph replays with a **new input every replay**, at M = 1..8 (CMPPRIO) or M = 1, 2, 3, 4, 6, 8. The inputs include hostile data: wide per-block exponent range, zeros, ties, and 448 saturation.

Must-fail mutants, each with a `_sleep` injected on the producer stream:
- prio without the join;
- the compressor done-event wait dropped, in serial and in prio;
- shared without `wait()`;
- engram take without the join.

Each must mismatch at every M; if one does not, the test fails because the harness is too weak.

T2 also prints the informational main-chain latency per mode (graph, M=4).

**Exit codes.**

| code | meaning | arms that are useful |
|---|---|---|
| 0 | all pass, gate PASS | all |
| 4 | T2-T4 pass, gate FAIL (prio and shared are inert in graphs) | only `cmp-serial` and `engram-hoist` |
| 3 | gate INCONCLUSIVE | |
| 1 | environment wrong, a mismatch, or a mutant not caught | |

`--cpu-selftest` checks the verdict logic only. It ran on this Mac and on 512S1. The GPU parts have **not** been run.

## 7. Install, arms and how to read the effect
- **Install.** `python3 install.py install` on 8/8. Five anchor patches (30 tagged lines), `.bak-r5-prio` backups, and `dsv41_prio.py`. The install is all-or-nothing: it refuses before writing anything if a base sha256 is not in KIT.json.
- **Other commands.** `status` reports CLEAN, INSTALLED or INCONSISTENT. `uninstall` restores a backup only when the file is exactly patch(backup). `dry` lists the patch without writing.
- **Composition.** r5-ffnfront probably patches `shared_experts.py` and/or `model.py`. Install it first, run `install.py compose-check`, review, and add the printed sha to `files[].compose_sha256`.
- **Uninstall order.** Uninstall in LIFO order: r5-prio before any older kit (an older kit's uninstall refuses a file that carries r5-prio lines). Never use `uninstall --force` on a composed node; it drops later kits' edits. Writes are not transactional across files: if a write fails part-way, `status` reports INCONSISTENT (every edit is inert while the flags are off); re-run install or uninstall.
- **Window arms.** All of them run on control = kit installed with flags off. **Every arm, control included, sets `RDMA_AR_AUDIT=200`** (a host-side log of the first 200 routed calls and every new stream, so the pairs stay matched):
  1. `DSV41_PRIO_ENGRAM_HOIST=1` (run first);
  2. `DSV41_PRIO_CMP=serial` (no gate needed; expected flat);
  3. `DSV41_PRIO_CMP=prio` (only if T1 PASS; expected not to);
  3. `DSV41_PRIO_SHARED=1` (only together with GATE_FIRST, against a GATE_FIRST control, and only if T1 PASS);
  4. then `cmp+hoist` (serial instead of prio if T1 fails).

  Promotion follows the usual bar: ≥ +0.3% in both pairs, exact.py ec2f4437f8e1ed09, digest 2f7fdccbd388225a, and 9/9 long prompts.
- **rank0.log lines to check:**
  - `DSV41_PRIO r5-prio: cmp=... shared=... engram_hoist=...`;
  - `DSV41_PRIO_ENGRAM_HOIST: engram 0@layer 0, engram 1@layer 13 (side stream priority 0)`;
  - no `unconsumed hoist` error;
  - `RDMA_AR_AUDIT`: the `NEW_STREAM` lines must match control's. An extra one means a collective reached rdma_ar on a side stream, which is a sequence desync and a 600 s hang. SHPRIO removes rdma_ar's "refuse collectives on aux_stream" net, so this matters most there. Stop the arm.
  - l2pf `g2 captured ... planned` equal, with no MISMATCH. With the hoist, the collective order changes identically on every rank, and the l2pf plan is re-learned at each warm-up. The wkv weight is then no longer in a collective prefetch window (it was 6 MB in the `ag4` window).
- **nsys checks:**
  - CMPPRIO: the skinny GEMM in layers 2/8/14/20 starts within about 2-10 µs of its quant (it was about 30 µs behind the `Kernel2` 8x8x16).
  - serial: `Kernel2` starts after the skinny GEMM ends.
  - hoist: `device_kernel (1,25,1,384)` appears at the start of layers 0 and 13 on a non-main stream, and the kv all-gather in layer 1/14 follows `mhc_post` without the 85 µs GEMM.

## 8. Risks
- **The T1 gate may fail.** torch may instantiate graphs without `cudaGraphInstantiateFlagUseNodePriority`, so every node runs at the launch stream's priority. The prio and shared levers would then do nothing in serving. The serial and hoist arms do not depend on it. The T1 diagnostic shows whether re-instantiating with the flag would help; that would be a follow-on change to graph instantiation and is not part of this kit.
- **Delayed compressor.** prio delays the compressor by the skinny duration and serial by quant + skinny. The model says the main side is the late input at FSCN by 23-27 µs on average, but layers where the compressor is late lose a little (the model clamped those losses at 0). The qkv_a all-gather then meets the compressor backlog that skinny meets today, so serial is expected flat.
- **What T2 timing cannot show.** `t2_timing` measures from the start until every branch has joined, and its all-gather stub is `torch.cat`. It does not reproduce the backlog above. A resource-matched all-gather stand-in (1 CTA, 512 threads, ≥ 96 regs, spinning ~20 µs), the 128×2 norm after skinny, and the time the FSCN main-side input is ready are open items, not built.
- **T4 uses a mock MXFP8 linear.** It calls `mx_mm` through its own `QM.apply` and a mock `embed`, not the serving `wkv.quant_method.apply` and `Engram.embed`. Exactness holds by construction (same call, same arguments), so T4 really tests the synchronization. Building the wkv through the serving MXFP8 linear method is an open item.
- **The hoist may delay FC1.** Its 25 resident CTAs can delay the host layer's one-wave FC1. Try a different host (`DSV41_PRIO_ENGRAM_HOIST_AT=1:0,14:12`) if the arm is flat or negative.
- **SHPRIO on top of GATE_FIRST.** The high-priority shared kernels take SMs ahead of FC2's pending CTAs, which is the CP.
- **Calibration.** Round-4 bench and model estimates were 3-4× off in both directions. Only serving C-T-C-T decides.

## 9. Files
- `install.py`, `KIT.json` (shas, flags, arms)
- `files/vllm/models/deepseek_v41/nvidia/dsv41_prio.py` (new module)
- `base/` (live copies with their shas) and `patched/` (review copies; regenerate them with `scripts/refresh_kit.py`)
- `tests/test_prio.py` and `tests/fakecuda.py` (CPU tests, vector-clock race checker)
- `device_test.py`
- `scripts/prio_t2.py`, `scripts/dbg_act.py` (t2 analysis, run on 512S1 under `~/climb-traces/r5-prio`) and `out/prio_t2.txt`
- `TEST-RESULTS.txt`
