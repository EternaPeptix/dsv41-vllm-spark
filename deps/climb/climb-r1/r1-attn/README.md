# r1-attn — climb round 1, attention-side scheduling kit

One kit, one installer, one flag:

| lever | flag | status |
|---|---|---|
| **G20 (attn:L-IP) IDXPAR**: overlap the indexer q chain with the main q chain and the wk/kstore chain in the 8 index-source layers | `DSV41_IDXPAR=1` (`flat` = diagnostic arm) | **built** |
| **G11 part (b)**: stop the kv-source compressor bf16 GEMM holding the SMs ahead of the qkv_a skinny (node priority / join moved) | — | **skipped** (reasons in §2) |

Files (paths relative to `site-packages/vllm`):

| file | action | sha256 |
|---|---|---|
| `models/deepseek_v41/attention.py` | patch (3 anchors, 10 tagged lines `# r1-attn`, backup `.bak-r1attn`) | base `f2f48198de81ebc7…` (live, 8/8 checked 2026-11-08) → patched `0e4b6e91bc54d88f…` |
| `models/deepseek_v41/nvidia/dsv41_idxpar.py` | new | `b682d81332df0198…` |

`python3 install.py install | uninstall | status | dry` (refuses unless attention.py is the KIT.json base or this
kit's patched result; idempotent). Install is 8/8 like every kit. No other file is touched; nothing under the
supervisor's preflight manifest changes.

---

## 1. G20 IDXPAR (`DSV41_IDXPAR`)

### 1.1 What runs today (live code, base f2f48198)

`DeepseekV4Attention._prepare_and_attn` (attention.py:840-922), after `_q_kv_norm_quant_kernel` on the capture
stream ("main"):

* kv-source layers 2/8/14/20 (compressor + indexer with `owns_k`):
  phase 1 `maybe_execute_in_parallel(main wq_b → fused qnorm/rope/SWA insert, aux0: compressor → latent)`
  (l.876-885); phase 2 `maybe_execute_in_parallel(indexer(...) = wk → indexer-K store → indexer wq_b → fused
  q rope/quant, aux0: compressed-cache insert)` (l.901-908; `DeepseekV4Indexer.forward` l.1492-1533).
* index-only layers 24/28/32/36: main wq_b → SWA insert → indexer wq_b → q rope/quant, all serial on main.
* then `_sparse_indexer_and_attn` (logits, top-k, sparse MLA) on main (l.913).

The indexer chain needs only `qr`, `positions`, `indexer_weights` (`forward_q`, l.1461-1490), yet it waits for
the main wq_b + insert (and, in kv-source layers, wk + kstore). Production traces (512S1
`~/audit-20261106/traces`, all six d200 windows, both ranks; `scripts/idxpar_trace.py`, output in
TEST-RESULTS.txt): q_kv_norm end → logits start = **52.8-54.6 µs** (kv-source) / **41.9-44.7 µs** (index-only);
the indexer wq_b (23.2-25.4 µs, DRAM-bound) starts **22.9-24.8 / 14.1-14.8 µs** late; nothing else runs on any
stream in that window; **sum of the 8 prologues = 383-390 µs per forward**.

### 1.2 What runs with `DSV41_IDXPAR=1` (captured FULL graphs only)

`dsv41_idxpar.LayerPlan._schedule`:

```
main : record E0                                     (all inputs were produced on / joined into main)
aux1 = aux_stream_list[1], priority 0   : wait E0; main wq_b -> fused qnorm/rope/SWA insert; record El
med  = new stream, priority -1 (kv-src) : wait E0; compressor -> wk -> indexer-K store -> compressed insert; record Em
high = new stream, priority -2          : wait E0; indexer wq_b -> fused q rope/quant; record Eh
main : wait El, (Em), Eh ; _sparse_indexer_and_attn(...)   (unchanged)
```

This is the skeptic's refined variant (verdicts-r2 `verify:G20:combined`): indexer chain highest priority,
main wq_b + insert on a low-priority aux1, compressor → (insert ‖ wk → kstore) on aux0 at medium priority
(here serial on the medium stream: wk → kstore first because the logits read it first), and **host launch
order aux1 first**, so the WOA fork stays behind the indexer wq_b (§1.6). Upstream template:
`models/deepseek_v4/amd/rocm.py` `_forward_csa2_reindex` (~l.763-800) does the same split on ROCm.

Where IDXPAR does **not** engage (stock path, byte-for-byte the base code): every eager forward (prefill,
uncaptured sizes, the profile run, the warm-up forward vLLM runs before each capture), breakable/piecewise
captures (`BreakableCUDAGraphCapture.is_active()`, where `maybe_execute_in_parallel` serialises anyway),
profile metadata (`attn_metadata` not a dict), layers without an indexer (incl. all DSpark drafter layers),
platforms without aux streams. A layer captured before any eager forward also falls back (logged).

### 1.3 Why flag-off is identical

* The patch adds 10 tagged lines: the flag (`_DSV41_IDXPAR`, read once at import), one guarded block at the end
  of `__init__` (after the compressor exists) and one guarded early-return at the top of `_prepare_and_attn`.
  With the flag unset/0 the executed code is the base code plus two false bool tests; `dsv41_idxpar` is never
  imported, no stream/event/allocation is created, no kernel/launch order/graph content changes.
* Tests: removing the tagged lines restores the base byte for byte; the patched AST minus the flag-guarded
  statements equals the base AST; every tagged statement is inside an `if _DSV41_IDXPAR` guard; the helper
  import is only inside the guard (`tests/test_install.py`). The real base and patched `_prepare_and_attn`
  sources, run against a logging fake CUDA layer over a 9-layer sequence, produce identical logs (kernels,
  streams, events, module calls) with the flag off (`tests/test_plan.py::test_flag_off_identical`).

### 1.4 Why flag-on is bit-exact (op by op)

IDXPAR only changes the stream a kernel is issued on, its node priority and the event edges. Each kernel is
the same call with the same arguments as the stock path:

| op | stock stream | IDXPAR stream | inputs (all read-only here) | writes (private to the branch) |
|---|---|---|---|---|
| main wq_b (`self._wq_b_proj`, MXFP8 CUTLASS SM120, grid 1×32, `PersistentScheduler`, no split-K) | main | aux1 | qr (quantised) | q_raw |
| fused qnorm/rope/SWA insert (`_fused_qnorm_rope_kv_insert`) | main | aux1 | q_raw, kv, positions, cos/sin | q (padded), SWA cache rows of this step |
| compressor (`_fused_save_compress_norm`) | aux0 | med | kv_score, positions, state cache | latent, compressor state cache |
| wk (`ReplicatedLinear` bf16, cuBLAS Kernel2, no splitKreduce) | main | med | latent | k_pre |
| indexer-K store (`indexer_k_norm_rope_store`) | main | med | k_pre, positions, k_norm | indexer K cache rows of this step |
| compressed insert (`rope_quant_insert`) | aux0 | med | latent, positions | compressed KV cache rows |
| indexer wq_b (same MXFP8 GEMM) | main | high | qr | iq_raw |
| fused indexer q rope/quant | main | high | iq_raw, positions, indexer_weights | index_q, scale, weights (fresh tensors) |

* No reduction crosses CTAs whose order could depend on concurrency: the MXFP8 GEMM uses CUTLASS
  `PersistentTileSchedulerSm90` (flashinfer `mxfp8_gemm_template_sm120.h` l.139-141; static tile→CTA map,
  `get_workspace_size` returns 0, `sm90_tile_scheduler.hpp` l.140-142), so the shared flashinfer workspace
  buffer is never touched and two concurrent wq_b GEMMs cannot interfere. wk: cuBLAS chooses its algorithm
  from shape, math mode and workspace size; the workspace size is global, so the medium stream gets the
  same algorithm (its workspace is allocated eagerly, §1.5). The others are elementwise/fused per-row kernels
  allocating their outputs.
* The kernels launch on the stream the caller selects: flashinfer's TVM-FFI binding takes
  `torch._C._cuda_getCurrentRawStream` at call time (`tvm_ffi/cython/function.pxi` l.172-212, 283-289),
  Triton/torch ops use the current stream; the stock aux paths rely on the same.
* No dataflow change: the CPU plan test checks that every op reads the same producer's output as in the stock
  run (`producers(...)` identical), the kernel multiset (name, operands) is identical, and the module-call order
  (L2 recorder events, wq_b ordinals) is identical.

### 1.5 Race analysis

* **Fork:** each branch's first action is `wait E0`, recorded on main after every input was produced on or
  joined into main (`_run_parallel_input_projections` joins aux0/aux1 before returning; `_split_qkv_and_norm`
  runs on main).
* **Join:** main waits El, Em, Eh before `_sparse_indexer_and_attn`: logits read the indexer-K cache (written by
  layer 20's medium branch for layers 24-36: Em joined in layer 20, main-ordered afterwards) and index_q;
  sparse MLA reads q, the SWA cache and the compressed cache. No branch outlives the function, so the capture
  never ends with an open branch and no work leaks into the next layer or replay.
* **Writes are disjoint** across concurrent branches (table above); reads of shared inputs are read-only.
* **Caching allocator under capture:** tensors allocated on aux1/high/medium (q_raw, q, latent, k_pre, index_q,
  …) are consumed on main after the joins and freed on the host only after `_sparse_indexer_and_attn` was
  enqueued; the next allocation on the same stream that could reuse such a block happens in a later layer
  after that layer's `wait E0`, which is recorded on main after the consumer was enqueued. Main-allocated
  inputs (qr, kv, positions, kv_score, indexer_weights) stay referenced until the function returns, i.e. after
  the joins. Same argument as the stock aux-stream code (latent on aux0, indexer_weights on aux1).
* **Nothing is created inside a capture:** the two streams are created at model construction; the first
  eager forward of each indexer layer materialises its four events and (once per device) runs a tiny
  `F.linear`/`mm` on each new stream so cuBLAS's per-(handle, stream) workspace is allocated outside capture.
  If a layer is ever captured before it ran eagerly, it uses the stock path in that graph (logged).
* **Mutation tests** (`tests/test_mutations.py`) confirm the checker catches: a missing medium/high join, a
  missing fork wait on high/aux1, an event recorded before its producer (high, aux1), a dropped kernel, and the
  wrong host order (WOA fork after the main wq_b).
* **PDL:** kernels keep their own launch attributes. The main wq_b loses its same-stream PDL predecessor
  (q_kv_norm) and now depends on it through an event edge (it can no longer start ~1 µs before q_kv_norm ends,
  `mainS` = −0.9 µs today); if capture turns the event edge into a programmatic edge, the GEMM's own
  `griddepcontrol.wait` still waits for q_kv_norm completion, exactly as today on one stream.

### 1.6 Stream priorities inside the captured graph (proof)

* vLLM captures FULL graphs with `torch.cuda.CUDAGraph()` + `torch.cuda.graph(...)`
  (`v1/worker/gpu/cudagraph_utils.py` l.484-500; `keep_graph=False`, so `capture_end` instantiates).
* torch 2.13.0+cu130 (git `cf30153c`) `CUDAGraph::instantiate()` calls `cudaGraphInstantiateWithFlags(...,
  cudaGraphInstantiateFlagAutoFreeOnLaunch | cudaGraphInstantiateFlagUseNodePriority)`
  (`aten/src/ATen/cuda/CUDAGraph.cpp` l.243-246 at that commit). The live binary agrees:
  `libtorch_cuda.so` `at::cuda::CUDAGraph::instantiate()` @0xf3a860: `mov x2, #0x9` then
  `bl cudaGraphInstantiateWithFlags@plt` (flags 9 = 1 | 8).
* CUDA 13.0 `cuda.h` l.20366-20369 (spark-one): with USE_NODE_PRIORITY the graph "uses the priorities from the
  per-node attributes rather than the priority of the launch stream … priorities are only available on kernel
  nodes, and are copied from stream priority during stream capture."
* torch pool streams are created with `cudaStreamCreateWithPriority(..., -p)`, p ∈ [0, min(4, device range))
  (`c10/cuda/CUDAStream.cpp` l.182-204, 344-361); `torch.cuda.Stream(priority=-2/-1)` therefore gives real
  CUDA priorities −2/−1 if the device reports ≥3 levels (recent NVIDIA GPUs typically report 0..−5; unverified on GB10 offline). vLLM's capture stream and all
  other streams are priority 0, the least. The helper logs the actual values at boot (device check 1); if
  the device had only one level above 0, high and medium share it (warning logged) and the lever degrades
  towards `flat`.
* Since flags = 9 already today, every existing node already runs with its capture stream's priority (all 0):
  IDXPAR's high/medium nodes are the only non-default ones in the graph.

### 1.7 Interaction with the L2 prefetcher (WOA) — the skeptic's pitfall

`dsv41_l2_prefetch._wq_b_post_hook` (l.513-532) forks the attention window at the **last** wq_b post-hook of the
block with `side2.wait_stream(torch.cuda.current_stream())` (`_fork_part` l.1056-1081 under G2/pacing, `_fork2`
l.1099-1109 otherwise). IDXPAR launches aux1 (main wq_b) first and the indexer wq_b last on the host, so the
last post-hook is the indexer wq_b, issued on the high stream: the WOA prefetch (4.3 MB wo_a, burst — `attn`
is not in `PACE_KINDS=ar`) starts right after the indexer wq_b, as today, and never competes with it for DRAM.
The module-call order (main wq_b, compressor, wk, indexer wq_b) equals the stock order, so the recorder learns
the same events and ordinals (`WOA captured … wq_b=48 (learned 48)` must not change). The indexer module's own
`__call__` is bypassed (`_produce_k`/`forward_q` are called directly, like the ROCm template); it holds no direct
parameter, so the recorder registers no hook on it (`_module_tensors` returns None).
Test: `test_plan.py::test_woa_fork_point`, and the inverse in `test_mutations.py::test_wrong_host_order_moves_woa_fork`.

### 1.8 Expected saving

Ideal overlap saves 152-155 µs per forward on every d200 window and both ranks (verify-G20 `prolog.py`);
derated by the skeptic for (1) SM oversubscription — two 32-CTA, 1-CTA/SM GEMMs on 48 SMs; with priorities the
indexer wq_b is dispatched first, without them saving falls to ~60 µs — (2) the WOA fork placement (handled,
§1.7) and (3) ~1-2 µs per layer lost when the event edge replaces same-stream PDL. Kit estimate
d = 0.035 / **0.105** / 0.145 ms per step → G1′ = mean 1000·AL/(T−d) with T = 24.26/21.33/24.04 ms,
AL = 2.926/1.843/3.613: **+0.15 / +0.45 / +0.62 %** (skeptic +0.26 / +0.52 / +0.62 %).

### 1.9 Risks

* Priority effect on the CTA dispatcher is not observable offline: if the main wq_b still grabs 32 SMs first,
  the gain falls to ~+0.26 %; the `flat` arm and the trace readout separate the two effects.
* Graph pool memory: per-stream free lists reduce block reuse inside each graph slightly (q/latent/index_q are
  ≤ 0.5 MB each at ≤ 64 tokens); plus up to two cuBLAS workspaces for the new streams (allocated in the first
  eager forward, i.e. inside vLLM's memory profiling). Compare the boot memory/KV lines with control.
* The DRAM-bound indexer wq_b now overlaps the main wq_b (L2-fed by the AG-window prefetch) and the small
  compressor-chain kernels; both are in the skeptic's derating.
* Kits that edit the WOA fork (`r1-pf` copies `dsv41_l2_prefetch.py`) or the q_kv_norm/PDL behaviour (`r1-pdl`
  copies `common/ops/query_quant.py`) do not touch attention.py, but A/B IDXPAR alone first; if r1-pf moves the
  attention-window fork off `torch.cuda.current_stream()` at the wq_b post-hook, re-check §1.7.

### 1.10 Reading the effect in an nsys trace

`python3 scripts/idxpar_trace.py <q*.sqlite>` (60-step window, `--cuda-graph-trace=node`; baseline output for all
six production d200 windows is in TEST-RESULTS.txt). Per index-source layer:

* `prolog` (q_kv_norm end → `sm120_fp8_paged_mqa_logits` start): today 52.8-54.6 / 41.9-44.7 µs; expect ~30-37 µs
  in all 8 layers; `sum of prologues per forward` today 383-390 µs, expect ~240-270 µs.
* `idxS` (indexer wq_b start): today 23-25 / 14-15 µs; expect ~0-2 µs; `ovl` = 1 (the two 1×32 `device_kernel`
  GEMMs overlap); `mainS` moves from −0.9 µs to ≥ 0.
* `woaS` (first `l2pf_kernel`): must stay ≥ `idxS + idxD` (fork after the indexer wq_b), never right after the
  main wq_b.
* `streams`: 3 → 4 (kv-source) and 2 → 3 (index-only); the kernel set and grids per layer unchanged.

### 1.11 Window arms and device checks

Arms (C T C T, ≥ +0.3 % G1 in both pairs, every exactness check identical):
`control` = {} ; `idxpar` = {`DSV41_IDXPAR=1`}. Optional diagnostic if idxpar < +0.3 %:
`idxpar-flat` = {`DSV41_IDXPAR=flat`} (same branches, no priorities).

Device checks (KIT.json `device_checks`): boot log priorities 8/8 and no fallback warnings; L2 WOA capture lines
unchanged; exact.py `ec2f4437f8e1ed09`, G1 digest `2f7fdccbd388225a`, 9 long-prompt hashes (kv-source layers
write the compressed KV, compressor state and indexer-K caches); trace readout; memory lines vs control.

---

## 2. G11 part (b) — skipped

Goal: in kv-source layers 2/8/14/20 the side-stream compressor `torch.mm` (cuBLAS Kernel2, 1024 CTAs × 16.9 KB
smem, enqueued first by `execute_in_parallel(default_first=False)`, attention.py l.995-1003) fills every SM
before the critical qkv_a skinny GEMM (48 CTAs × 61.6 KB smem) can start; the skinny waits 46-51 / 42-45 / 25-26 µs.

Why it is not built:

1. **Priority cannot be applied from the attention side without moving the critical chain off the capture
   stream.** CUDA priority 0 is the least priority and the capture stream and the compressor's aux0 are both 0,
   so the compressor cannot be demoted below the qkv_a chain; the qkv_a chain would have to be promoted, i.e.
   `_fused_wqa_wkv_gemm` → `dsv41_r3_dense.qkv_a_sharded` (shard GEMM **plus the RDMA all-gather** and the l2pf
   AG-window fork/join in `before_collective`) issued on a high-priority non-origin stream inside the capture.
   The RDMA collective (`rdma_ar_ext.so` 14d049dd, split post / proxy hygiene) has never been captured off the
   origin stream, its protocol is not inspectable here, and the skinny/all-gather live in files this kit does
   not own (dsv41_r3_dense.py, rdma_all_reduce.py). Promoting only the skinny GEMM would need an edit inside
   `qkv_a_sharded`.
2. **The gain depends on undocumented dispatcher behaviour.** Even with node priority, the skinny's 61.6 KB CTA
   fits on an SM only after 4 of its 6 compressor CTAs retire; whether the CTA scheduler holds freed SMs for the
   pending high-priority CTA instead of back-filling them with low-priority compressor CTAs is undocumented
   (the skeptic: "needs a device test"). The two cheap variants are measured/modelled at ~0: the simple
   reorder (fork after the skinny) and moving the join to the consumer (wq_b's 96 KB CTAs cannot co-reside with
   compressor CTAs either) — verdicts-r2 `verify:G11:combined`.
3. Most of G11's value is part (a) (prefetching `fused_wkv_wgate` in the post-MoE AR window), which belongs to
   the prefetch kit, not here.

A later round could revisit (b) only with a device micro-test of priority back-fill behaviour (skinny_mx +
Kernel2 co-run on one stopped Spark) and with the r3/rdma owners.

---

## 3. Tests (CPU, 512S1, torch 2.11) — see TEST-RESULTS.txt

`tests/run_all.sh ~/exo-venv313/bin/python3` → 20 tests, all OK:

* `test_install.py` (9): base sha = KIT.json = live 8/8; anchor counts; patched compiles; tag-strip = base;
  flag-off AST = base AST; every tagged statement guarded; flag parsing; installer install/idempotent/status/
  dry/uninstall on a temp tree; refusal on a drifted base.
* `test_plan.py` (8): real base/patched `_prepare_and_attn` + real `maybe_execute_in_parallel` + real helper on a
  fake CUDA layer over layers 1,2,3,8,20,21,24,28,36: flag-off log identical; eager/unprimed/breakable/profile
  → stock; captured: same kernels, same dataflow producers, same module-call order, every read after its
  producer, no unordered write/write or read/write, all branches joined, placement/priorities as designed;
  WOA fork after the indexer wq_b on the high stream; flat/invalid modes.
* `test_mutations.py` (3): the checker rejects 7 broken variants and the wrong host order.

Not testable offline (device checks): actual stream priority levels, CTA dispatch order under priority,
timing.
