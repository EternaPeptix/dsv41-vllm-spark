# r6-step: step-start collectives (C3) and the E1 main_proj side stream (C4)

Climb round 6. Production launcher 166d7aac, G1 130.23. 1% G1 is about 0.21 ms/step. Every number below is
re-based on the t2 traces (512S1 `~/climb-traces/t2/prod/q{0,4}.{1,2,3}.sqlite`). The scripts are in `scripts/`
and their outputs are `scripts/*_t2.txt`.

**What has not run.** No GPU device test and no serving arm has run. The GPU parts of `device_test.py` need a
stopped node. The CPU tests, the harness CPU self-test, a no-launch sm_121 compile check and a CPU-only check of
the device test's serving-layer builder against the live vllm032 package and the real checkpoint did run (see
`TEST-RESULTS.txt`). The r6-review blockers and what changed for them are in `REVIEW.md`, section "Fixes".

## 0. Summary

| lever | flag | what leaves the critical path | ms/step low / central / high | % G1 |
|---|---|---|---|---|
| C3 COLEMB | `DSV41_STEP_COLEMB=1` | target embedding AR (eager) and DSpark embedding AR (drafter graph) become column all-gathers | 0.016 / 0.024 / 0.034 | 0.08 / 0.11 / 0.16 |
| C3 PACK look | `DSV41_STEP_ENGRAM_PACK=look` | one of the two Engram rows all-gathers | −0.003 / 0.008 / 0.016 | −0.01 / 0.04 / 0.08 |
| C3 PACK all (on top of look) | `DSV41_STEP_ENGRAM_PACK=all` | one of the two Engram wkv all-gathers | 0.003 / 0.013 / 0.025 | 0.01 / 0.06 / 0.12 |
| C4 | `DSV41_STEP_C4=1` | DSpark main_proj MXFP8 quant + CUTLASS GEMM (E1) | −0.06 / 0.056 / 0.065 | −0.3 / 0.27 / 0.31 |
| all four | | | about −0.04 / 0.10 / 0.14 | about −0.2 / 0.48 / 0.67 |

Collective count per step:
- Today there are 5 step-start collectives: the eager embedding AR, two Engram rows AGs and two Engram wkv AGs.
- With `PACK=all` there are 3: an eager embedding AG, rows AG#1 and wkv AG#2. Two of those are merged and one is
  cheaper.
- The drafter embedding changes from AR to AG.

The analyst's full 5 → 2 design would also fold the embedding into AG#1. I did not build that step (§6) because
on t2 it would add about 10–20 µs more and needs the embedding moved into the target graph with an image-row mask.

**What t2 changed in the C3 estimate.** The analyst counted C3 at 0.05 / 0.09 / 0.13 ms. On t2, both Engram
rows-AG windows carry 18–22 µs of paced L2 prefetch, which is all of each window. The wkv-AG windows carry 10–13 µs.
So removing those collectives also removes DRAM-hiding time that the l2pf planner was using
(`scripts/engpf_t2.txt`). The embedding ARs carry no prefetch, so their saving is clean.

**The G12 prior for C4.** G12 `DSV41_MAINPROJ_HOIST=chain` (r1-drafter, live and off) forked the same main_proj
chain at the same point (after the screen). In w2 it measured **−0.95% G1** (122.42 against controls
123.65 / 123.57), and nobody found the cause. C4 differs from G12 in the three places where I can see a plausible
cause:
1. The all-gather stays on the main stream, so there is no rdma xstream hand-off.
2. propose's arguments are not rewritten: combine_hidden_states runs as usual and only the local GEMM is swapped.
3. There is no `record_stream`.

G12 "gemm" mode, the nearest variant, was never measured. The low end of my C4 range is therefore negative. Only
C-T-C-T decides.

## 1. C3: step-start collectives

### 1.1 Today (t2, rank 0 and rank 4, medians)

- **Eager pre-target region** (drafter graph end → target graph start: 51.7–57.7 µs):
  - gap 2.6 µs
  - `vocab_parallel_embedding_kernel` 1.8–4.0 µs
  - gap 2.1–2.3 µs
  - **`ar_kernel_pl` AR 24.5–31.5 µs**
  - gap 7.0–7.4 µs
  - `_gather_lookback_kernel` 1.5 µs
  - 10 µs to the graph launch
- **Target graph, Engram layers 1 and 14**, each:
  - rows AG `[T,3,256]` p50 18.0–22.0 µs
  - quant 2 µs
  - wkv GEMM 83.4–86.4 µs (DRAM-bound)
  - wkv AG `[T,3200]` p50 22.7–33.6 µs
  - `_fused_engram_post_wkv_kernel`

  The rows AG's predecessor on the stream is `mhc_post_tilelang_kernel`, but that is program order only: the AG
  depends only on the lookup.
- **Drafter graph:** the first collective is the DSpark embedding AR (grid 7, two-shot), p50 26.6–32.1 µs, with
  0.0 µs of l2pf overlap.

### 1.2 COLEMB (`DSV41_STEP_COLEMB=1`)

`VocabParallelEmbedding.forward` is wrapped at class level. The wrapper acts only on plain
`VocabParallelEmbedding` instances that meet all of these conditions (`colemb_refusal`):
- unquantized bf16 weight;
- tp > 1, reduce on;
- no added vocabulary;
- D/tp is 16-byte aligned.

The DSpark embedding aliases the target's; tables are keyed by the weight pointer.

**Build** (eager, once, at the first call, which is the profile or dummy run; never inside a capture):
- Every id goes through the **live** forward in chunks of up to 128 rows: the fused kernel plus the rdma AR.
- Each rank keeps its 640 columns.
- At the same time each rank compares every row it owns, all 5120 columns, against `canon(W_local)`.
- Each rank also accumulates a checksum of every AR output, and the ranks compare the checksums.
- Any mismatch on any rank turns the lever off on every rank (`_tp_agree_min`). The log line is
  `DSV41_STEP_COLEMB rank r: full-table check 129280 x 5120 ... PASS|FAIL`, with counts of −0, NaN (canonical or
  not), denormal and inf values.
- This is the full-table bitwise proof on the real 8-rank AR, rerun at every startup.

**Serve** (0 < T ≤ min(`DSV41_STEP_COLEMB_MAX_T`=128, 2 MiB / (2·5120) = 204 rows)): a single Triton kernel does
`out[t] = table[id]` when 0 ≤ id < V, and +0 otherwise. It is a raw int16 copy (`tools/compile_check.py`: no FP
opcode in its SASS). Then `tensor_model_parallel_all_gather(dim=-1)` runs the rdma column gather. Above the cap,
and for every other embedding, the stock path runs unchanged.

**Exactness, op by op.** Stock output element (t, c) is `bf16_rn(((0.f + x_0) + x_1) + … + x_7)`, where:
- `x_r` is rank r's fused-kernel partial: the owner rank's partial is W[id, c], and every other partial is ±0;
- the reduction is FADD in rank order from a +0.0 seed, then one F2FP.

The live .so is 14d049dd. I checked its SASS read-only on spark-eight: it has 1384 FADD, 60
F2FP.BF16.F32.PACK_AB, and **no FADD.FTZ**. With exactly one non-zero contributor, the result is
`canon(v)`:
- −0 → +0;
- any NaN → 0x7FFF (FADD returns the canonical fp32 NaN, F2FP packs it to 0x7FFF);
- every other value, including denormals and ±inf, is unchanged.

`canon_bits` implements exactly this map. `tests/test_step.py` checks it against an instruction-level emulation
for all 65536 bf16 patterns × owner rank {0, 3, 7} × masked partial {+0, −0}.

The served value is `table[id]`, which is the stock AR output for that id. The gather is a byte copy into the
same columns (`cuda_communicator.all_gather` → `all_gather_cols`). Ids outside [0, V) give +0 on both paths.

**Composition.** It replaces one collective with another at the same program point on every rank, so the rdma
op sequence stays rank-consistent. The l2pf planner re-learns its windows at each capture's eager warm-up; the
"ag" window kind replaces "ar".

**Cost.** +165 MB per rank (129280 × 640 bf16). The vocab-parallel table stays, because the stock path still
serves T above the cap.

### 1.3 PACK (`DSV41_STEP_ENGRAM_PACK=look|all`)

`Engram.__init__` is wrapped to record the two instances in layer order. `Engram.embed` is wrapped. The plan
refuses (`engram_refusal`) on any of:
- EngramBatch;
- sequence parallelism;
- tp 1;
- Engram DP > 1;
- shards that do not tile `n_hash_cols`;
- a wkv that is not a plain gathering ColumnParallelLinear.

The plan also requires exactly 2 Engram layers with identical shapes.

**look.** At layer 1's `embed` (or at layer 0 under r5-prio's hoist):
1. `a._ready_rows(T)` and `b._ready_rows(T)`. Main waits on both lookups' events. These are the stock waits;
   layer 14's lookup was started at the forward's start on the same prefetch stream.
2. One cat → `[T,6,256]`.
3. **One** column AG → `[T,48,256]`, laid out by rank r as [a heads | b heads].
4. One permute copy → both stock layouts `[T,24,256]`. Layer a's result is returned. Layer b's is stored with the
   forward's sequence number and T, and returned by b's `embed`, which runs no kernel and no collective.

`tests/test_step.py` and `device_test.py` compare the layout with the stock `all_gather(rows, dim=1)` on hostile
bit patterns (NaN payloads, −0, denormals, inf). The mutants are the two halves swapped and the rank order
reversed.

**all** (look, plus):
1. Layer a's wkv forward is wrapped per instance. It runs `wkv_a.quant_method.apply(x_a)`, then
   `wkv_b.quant_method.apply(x_b)`, where x_b is layer b's stored rows: the same bytes as b's operand.
2. One cat `[T,6400]`.
3. **One** column AG.
4. One copy that splits the result into the two stock `[T,25600]` gathers.
5. Layer b's wkv forward returns the stored kv.

The GEMM is the same kernel with the same shape, weights, input bytes and tactic; only its time changes. The
device test runs the kit's own wkv wrappers on the real layer-1 and layer-14 wkv weights. It builds them through
the live quant-method lifecycle, so they have the serving-layout scale and use the serving kernel
(`FlashInferCutlassMxfp8LinearKernel`). For every rank slice and every served T, the interleaved order must equal
the stock order bitwise: layer 1's GEMM, the layer 2–13 stand-in GEMMs, then layer 14's GEMM. The mutants feed
layer 14 with layer 1's rows, or run layer 14's GEMM with layer 1's weights. Both GEMMs stay on main, so the
critical-path GEMM total is unchanged.

`all` falls back to `look` while `DSV41_PRIO_ENGRAM_HOIST` is on, because the hoist then owns both GEMMs. The
choice is made from the env, so it is the same on every rank.

**Eligibility.** 0 < T ≤ 128 (`DSV41_STEP_ENGRAM_PACK_MAX_T`), and not in a breakable capture. The pairing
(sequence number, T) means a forward in which layer a takes the stock path makes layer b take the stock path too.

### 1.4 C3 and r5-prio's Engram hoist (both install states)

- **r5 not installed:** the r5-prio lines are absent, so look and all run as described. Installer base is
  `aa8aad1b`.
- **r5 installed, hoist flag off:** r5-prio's `Engram.forward` line calls the live statement, which reaches my
  wrapped `embed` / wkv. Installer base is `c8170bd9`. `tests/test_step.py::test_compose_with_r5_prio_both_orders`
  runs the real r5-prio installer on copies.
- **r5 installed, hoist on, look:** the hoist calls `engram.embed` at layers 0 and 13. AG#1 runs at layer 0 and
  layer 13 takes the stored rows (`test_engram_pack_under_r5_hoist_order`). The hoist's side-stream GEMMs and their
  AGs are unchanged.
- **r5 installed, hoist on, all:** falls back to look and logs it.

**Which one should replace the other.** For the wkv part, `all` and the hoist are mutually exclusive:
- `all` removes one AG, a deterministic critical-path removal of about 0.013 ms central.
- The hoist takes both 84 µs GEMMs off the critical path, but they are DRAM-bound against DRAM-bound layers (r5
  central +0.05%, range −0.2..+0.6).

Run arms `pack-all` and r5 `engram-hoist` (+`pack-look`):
- Keep **all**, which replaces the hoist, unless hoist+look beats all by ≥ 0.1% G1 in both pairs.
- If the hoist wins, keep the hoist with `look`.

A third variant would put both GEMMs on the hoist side stream at layer 0 and join before AG#2. It is not built,
because 168 µs of DRAM-bound GEMM under layer 0 is very likely zero-sum.

### 1.5 C3 estimate (t2)

**COLEMB:**
- Target, eager: the AR is 24.5–31.5 µs. An eager AG of [T,640] should cost what E1's main_proj AG [M,640] costs:
  16–22 µs. The lookup kernel costs the same as the fused kernel. Saving 3 / 9 / 15 µs.
- Drafter, in graph: the AR is 26.6–32.1 µs and its window has no prefetch. An in-graph AG costs what the drafter
  AGs cost: p50 12.0–12.3 µs. Saving 13 / 15 / 19 µs.

**look:**
- Gross saving: the layer-14 rows AG, 19.8–21.6 µs, plus 1.05 µs of gaps.
- Costs:
  - the extra cat and split copy plus the larger AG: 5–7 µs;
  - up to about 17 µs of lost prefetch hiding, since the window carried 18–22 µs at the paced rate;
  - the packed gather at layer 1 now also waits on layer 14's lookup event. Both lookups run back to back on the
    prefetch stream at the forward's start. `scripts/engwait.py` on all 6 t2 traces (360 forwards): layer 14's
    lookup ends 379 µs (min) / 406 µs (p10) / 510 µs (p50) before layer 1's rows AG starts, never after it
    (`scripts/engwait_t2.txt`). So this wait costs 0 µs on t2, and the low bound is unchanged. It would cost
    something only if a forward's lookups ran over 380 µs later than in t2.
- Net −3 / 8 / 16 µs.

**all, extra:**
- Gross saving: the wkv AG, 24.8–33.6 µs, plus 1.24 µs of gaps.
- Costs:
  - the cat, the copy and twice the payload: 8–12 µs;
  - 0–11 µs of lost hiding.
- Net 3 / 13 / 25 µs.

## 2. C4: the E1 main_proj GEMM under refine and sampling (`DSV41_STEP_C4=1`)

### 2.1 Today (t2, `scripts/e1host_t2.txt`, 6 traces, medians, µs from the target graph's end)

| time from target-graph end | event |
|---|---|
| 25–29 | screen starts (459–477 µs) |
| 486–509 | refine (22–49 µs) |
| | sf_local_argmax |
| 521–578 | argmax AG (17–107 µs) |
| | sampling glue |
| 627–726 | CatArray (2.3–2.6) |
| | quant (3.7–4.3) |
| | **GEMM `device_kernel` (61.4–64.8)** |
| | main_proj AG |
| | rms_norm |
| | prepare_dflash |

- The window from the screen's end to CatArray is 126–215 µs (p10 114–155).
- The host leads the GPU by 25–37 ms, so all of this is GPU time.

### 2.2 Mechanism

The patch to `dsv41_r4_small.py` has four tagged blocks:

1. **Install.** At `maybe_install()`, `install_c4()` wraps `GPUModelRunner.sample_tokens`, which captures
   (`aux_hidden_states` list, speculator) as G12 does. It refuses when:
   - G12 is on;
   - PCP or a batch sharder is active;
   - the speculator is not DSpark.
2. **Fork.** In `target_logits`, on a CERT-armed step, `c4_target_hook()` plans the fork. The conditions are:
   - not capturing;
   - the r4 split exists;
   - all taps have the same row count M, with `checked[M] is True`.

   It then passes `after_screen` to `certified_logits`, which calls it right after the screen launch:
   - `ev_screen.record(main)`, then `S.wait_event`;
   - on S: `torch.cat(aux)` and `ColSplit.local` (the live MXFP8 quant and CUTLASS `mm_mxfp8`);
   - `ev_done.record(S)`.
3. **Take.** In `mainproj_split`, before the verdict lookup, `c4_take(st, x, M)` runs. It checks that the split
   object, M and the operand shape are the same, then `main.wait_event(ev_done)`, then
   `st.gather(y)` (the rdma AG, on main, at the stock point), and `main_norm` follows as today. On a mismatch it
   joins and the stock path runs.
4. **Drop.** At the end of `sample_tokens`, any launch nobody consumed is joined (`main.wait_event(ev_done)`) and
   counted as dropped.

### 2.3 Why the output is bit-exact

The kernels, their inputs and shapes are the same. The GEMM has the same M, so the same autotuned tactic
(`DSV41_DENSE_TACTIC` is unset in the launcher). The live combine path computes `local(cat(aux))` then
`gather`. C4 computes the same `local(cat(aux))` earlier on S, from the same tensors, then the same `gather` on
main.

**Nothing writes the aux taps between the target graph and propose.**
- The taps are the target CUDA graph's static outputs (`run_fullgraph` replays; `execute_model` unpacks
  `hidden_states, aux_hidden_states = model_output`, model_runner.py:2180).
- From there to propose, the code (model_runner.py:2217-2380) only reads them, which I checked line by line:
  - `execute_model_state` → `sample_tokens`;
  - `compute_logits` / `target_logits`, which reads `hidden_states`;
  - the sampler;
  - the rejection sampler;
  - `_post_update`;
  - `pcp_manager.restore_for_sampling`, which C4 refuses when PCP is present;
  - `propose(… aux_hidden_states …)`;
  - `DFlashSpeculator.propose` → `combine_hidden_states(torch.cat(aux))`, dflash/speculator.py:394-397.
- The only writer is the next target replay. It runs on main after this step's `wait_event(ev_done)`, so after
  S's last read.
- No other graph aliases the taps: the graph manager keeps them alive as outputs, so later captures cannot reuse
  their memory.

This argument is backed three ways:
- `DSV41_STEP_C4_AUDIT=N` checks it in serving: at combine time, on the real inputs at every served M, it
  recomputes the stock GEMM on main and compares bitwise, so a write after the fork would show as a mismatch.
  During those N steps the stock result is the one gathered and served, so the audit arm is exact whether or not
  C4 is (`tests/test_step.py::test_c4_audit_serves_stock_y`). The window's audit arm uses N = 2000.
- `tests/test_step.py::test_c4_happens_before[aux_write]` (vector-clock model) catches a write on main between
  the fork and the take.
- The device test's mutant 3 does the same on a GPU.

### 2.4 Race analysis

- **S reads the taps after the target graph.** S waits on `ev_screen`, which is recorded on main after the screen,
  which is after the graph.
- **main reads y after S writes it.** The `wait_event(ev_done)` comes before the gather.
- **The next target replay writes the taps after S's reads.** That follows from the join above. If propose is
  skipped, the `sample_tokens` drop path joins instead.
- **S's next allocations come after main's consumer.** The next `after_screen` starts with S waiting on main.
  Blocks allocated on S return only to S's pool, so `record_stream` is not needed. The last launch's tensors are
  also held in `C4S.keep` until the next fork.
- **Shared mm_mxfp8 workspace.** FlashInfer's `mm_mxfp8` has a process-wide workspace. Main runs no `mm_mxfp8`
  between the fork and the take: refine and argmax are Triton kernels, then the AG and the torch sampling glue.
- **No collective on S.** The rdma ext's order contract and the xstream guard are therefore untouched;
  `RDMA_AR_AUDIT` NEW_STREAM lines must equal the control's.
- The vector-clock tests check all of this, with mutants for a dropped join, a missing screen wait and a tap write
  (each one is caught).

### 2.5 Graph-capture safety

E1 is eager. `c4_target_hook` returns None while capturing. Streams and events are created lazily on the first
eager fork. Nothing in a captured graph changes.

### 2.6 C4 estimate

- **Gross:** quant 3.7–4.3 + GEMM 61.4–64.8 + about 1.2 µs of gaps leave main: 66–70 µs. The event wait adds
  back 1–2 µs.
- **Hidden?** The side chain (cat + quant + GEMM, about 70 µs) starts at the screen's end. Main reaches combine
  126–215 µs later (p10 114–155), so it is hidden at p10 on every workload and rank.
- **Contention:**
  - the GEMM holds 5 SMs (a CUTLASS CTA needs a whole SM) for about 61 µs while refine's 1010 CTAs run;
  - it reads about 10 MB of weights during refine's DRAM reads;
  - estimated 3–15 µs (more on maths rank 4, where refine is 49–77 µs).
- **Net:** 50 / 56 / 62 µs. I use 0.056 central, 0.065 high, and −0.06 low for the unexplained G12 mode.

## 3. Flag off is identical

Both patched files only gain:
- a module-level bool (`_DSV41_STEP_C3` in engram.py, `_DSV41_STEP_C4` in dsv41_r4_small.py);
- `if <bool>:` blocks.

With the flags unset, `dsv41_step` is never imported, no class or function is wrapped, and no stream, event or
allocation is added. `tests/test_step.py::test_flag_off_is_ast_identical` folds the bools to False and checks that
the AST equals the base AST, for the live engram.py, the r5-composed engram.py and dsv41_r4_small.py.

## 4. Install, composition, files

`python3 install.py install | uninstall | status | dry | compose-check`, modelled on r5-prio:
- anchor edits with exact counts;
- every inserted line tagged `# r6-step`;
- one `.bak-r6-step` per file;
- all-or-nothing;
- refuses unless each file's sha256 is a KIT.json base or compose sha;
- uninstalls only when the file is exactly patch(backup).

| file | base(s) | patched |
|---|---|---|
| `models/deepseek_v41/common/engram.py` (an end-of-file install block) | live `aa8aad1b`; r5-prio `c8170bd9` | `34cf684e`; `e99f623d` |
| `models/deepseek_v41/nvidia/dsv41_r4_small.py` (4 blocks) | live `3364d8c8` (no r5 kit touches it) | `d9bf5d59` |
| `models/deepseek_v41/nvidia/dsv41_step.py` | new | see KIT.json |

All bases are identical 8/8 (census 2026-11-09, read-only). No runtime source transform matches these files: there
is no sitecustomize, and the .pth files do not reference them.

**Install order.**
- Install after `climb-r5/install-all.sh`, or on live.
- r5-prio pins engram.py, so it refuses to install or uninstall while r6-step is installed. Uninstall r6-step
  first, LIFO; this is tested.
- dsv41_r4_small.py is not in the supervisor manifest list in the r1 brief. Install 8/8 anyway.

**Window plan (explicit).**
- `install.py` has no fleet driver and does not check `CLUSTER_LOCK`. Run it on each of the 8 Sparks **only**
  from the window's `KIT_INSTALL` step, while `climb-window.sh` holds the lock. After it runs, check
  `python3 install.py status` 8/8: both patched files must show their KIT.json `patched_sha256`, and dsv41_step.py
  must show INSTALLED.
- Order when r5 is also in the window: `climb-r5/install-all.sh install`, then r6-step `install.py install` 8/8.
- **LIFO, every time.** While r6-step sits on top of r5-prio, `climb-r5/install-all.sh status`, `uninstall`,
  `preflight` and its device-plan report engram.py as FOREIGN (34cf684e / e99f623d is not an r5 state) and refuse.
  Run `install.py uninstall` 8/8 first, then any r5 install-all action. Re-install r6-step afterwards if needed.

## 5. Device test (`device_test.py`, a stopped node, about 4–8 min)

```
source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/climb-scratch/r6-step && python3 device_test.py --source kit      # or --source installed after install
```

**E0** asserts the environment and that the serving ops are the ones under test:
- venv vllm032, CUDA_HOME, nvcc on PATH, sm_12x;
- vllm loaded from the venv;
- `torch.ops._C.vocab_parallel_embedding` is the fused CUDA kernel;
- FlashInfer `mm_mxfp8`;
- the live `dsv41_r4_small.ColSplit`;
- `DSV41_DENSE_TACTIC` and `VLLM_DISABLED_KERNELS` unset.

**E0, serving linears.** `mtp.0.main_proj` and `layers.{1,14}.engram.wkv` (all 8 column rank slices) are built
from the checkpoint through the live quant-method lifecycle:
- `DeepseekV4FP8Config(checkpoint quantization_config).get_quant_method` gives `ModelOptLinearMethod(kMxfp8Static,
  CkptCtx(scale_block_size=(32, 32)))`.
- The live MXFP8 scale loader turns the e8m0 32×32 block scale ([160, 480] for main_proj, [800, 192] for wkv)
  into uint8 `repeat_interleave(32, 0)` ([5120, 480] / [25600, 192]).
- ColumnParallelLinear's load narrow takes a rank slice of both params.
- The live `select_linear_kernel` runs, then the live `process_weights_after_loading`, which leaves the swizzled
  1-D serving scale.

E0 gates on that build:
- method `ModelOptLinearMethod`, default `FormatScheme`, kernel `FlashInferCutlassMxfp8LinearKernel` (the
  serving choice on sm_12x), swizzled 1-D scale;
- `ColumnParallelLinear.forward` and `ReplicatedLinear.forward` contain exactly `quant_method.apply(self, …, bias)`,
  so `layer.quant_method.apply(layer, x, None)` is the serving GEMM;
- `concat_r ColSplit(main_proj, r, 8).local(x)` == the stock main_proj GEMM, bitwise, at M = 1, 7, 32, with a
  rank-order mutant. This proves C4's GEMM is the serving GEMM.

The old test handed the raw checkpoint scale to ColSplit, which crashed. `tools/serving_layers_cpu_check.py` ran
the same builder on CPU on spark-eight (live vllm032, real checkpoint, no GPU): the live kernel selection picked
`FlashInferCutlassMxfp8LinearKernel`, and the live ColSplit builds for all 8 ranks with an exact slice and scale.
The raw scale is still rejected.

If any E0 gate fails, the test exits 2 and tests nothing else.

**C3a: the full table, 129280 × 5120, real and hostile.**
- The hostile variants are planted rows: −0, NaN payloads, denormal, inf, and the maximum id.
- Live fused-kernel partials for all 8 shard ranges:
  - owner rows must equal W;
  - all other rows must be ±0.
- A Triton AR emulator uses the live kernel's instruction kinds: 16-bit widening, FADD without .FTZ from +0.0 in
  rank order, F2FP. `tools/compile_check.py` asserts these in its sm_121 SASS.
- Checks against the emulated AR:
  - `canon(W)` == the emulated AR;
  - the kit's lookup plus column concatenation == the emulated AR;
  - replays at every served T (1..16, 20, 24, 25, 30, 32, 40, 48, 64, 96, 128) with random, hostile and invalid
    ids.
- Mutants, at **every** served T including T = 1. Each mutant gets its own id vector whose index 0 is the row it
  must change, and that vector also has to pass the real gate:
  - a table without the map: the −0 row at index 0, then the NaN row;
  - the column shard shifted: an ordinary row at index 0;
  - invalid ids not zeroed: −1 at index 0, then V+7.

  The CPU self-test checks that each id vector catches its mutant at all 26 served T.

**C3b.**
- Gates: rows pack and wkv pack equal the stock layouts at every T, on hostile bytes.
- PACK=all GEMM: the kit's `_wrap_wkv` / `wkv_first` / `wkv_second` run on the E0 serving wkv layers. Interleaved
  must equal stock order, bitwise, for both layers' gathered kv, for every rank and every served T (§1.3).
- Mutants: halves swapped and ranks reversed at every T. Also at every T, layer 14 fed layer 1's rows, and layer 14's
  GEMM run with layer 1's weights. The CPU self-test runs this harness with stand-in GEMMs: 60 gates pass and 20
  mutants are caught.

**C4.**
- Setup:
  - the E0 serving `mtp.0.main_proj` (serving-layout scale);
  - the live `ColSplit` for all 8 rank slices;
  - the kit's `_Launch` / `c4_take` with a real side stream;
  - an 85 MB "screen" copy and a "refine" copy on main.
- Gate: hoisted == the stock local GEMM, bitwise, at every served M, 3 iterations each.
- Mutants, at every served M and every rank:
  - a dropped join (S delayed with `_sleep`);
  - S not waiting for the taps' producer;
  - a tap written after S read it.
- Also prints a median main-stream span, stock against hoisted.

**Exit codes.** 0 only if every gate passes and every mutant is caught.

**What one node cannot test.** One node cannot run the 8-rank rdma AR, which is why the C3a reference is
emulated. The serving-time builder covers it: it re-proves owner rows × 5120 columns on the real AR at every
startup and refuses on any mismatch.

## 6. Not built, and where I see more

- **The 5 → 2 completion: embedding in the target graph, riding AG#1.**
  - What it needs:
    - text-only steps skip the eager embedding;
    - the graph does the column lookup at the forward's start;
    - a persistent per-row mask selects the eager image rows on multimodal steps;
    - AG#1 moves to the forward's start, which puts the Engram host-memory lookup wait (3–14 µs) on the critical
      path.
  - Net is about +10–20 µs over COLEMB+PACK.
  - It touches `model_state.prepare_inputs_embeds` and the capture contract for padded rows. Effort is about 2–3
    days.
- **The prefetch-window coupling that §1.5 found applies to every collective-removal lever.**
  - The rows-AG windows are 100% used by paced prefetch, so the net value of removing a collective depends on
    whether the l2pf planner re-places that DRAM work.
  - A replan of the r1/xsr planner that moves the prefetch into the remaining windows would recover the 0–17 µs
    lost here.
  - It would also make C1/C2 estimates more reliable.
- **C4's sibling for the head (max,id) gather.** The analyst's merge with the main_proj AG is zero-sum (§4 C4
  of the analyst's report). Not pursued.

## 7. How to read it in nsys

- **COLEMB, eager pre-target region:** `vocab_parallel_embedding_kernel` + `ar_kernel_pl` (AR) becomes
  `_colemb_kernel` + `ar_kernel_pl` (gather). Expect span −3..−15 µs.
- **COLEMB, drafter graph:** the first `ar_kernel_pl` changes from grid 7 to the gather grid and gets shorter,
  and `vocab_parallel_embedding_kernel` becomes `_colemb_kernel`.
- **PACK:**
  - layer 1 shows CatArray, then the AG, then one `elementwise` copy;
  - layer 14 has no rows AG;
  - with `all`, two wkv `device_kernel`s run back to back in layer 1, and layer 14 has no quant, GEMM or wkv AG
    before `_fused_engram_post_wkv_kernel`.
- **C4:**
  - CatArray, MXFP8 quant and the main_proj GEMM appear on a new stream starting at the screen's end;
  - on main, propose's CatArray is followed directly by the main_proj `ar_kernel_pl`;
  - `scripts/e1host.py` milestones: CatArray → rms_norm should fall from about 95 µs to about 30 µs.

## 8. Risks

- **C4 can repeat G12's regression.** The cause is unknown, so a G12 trace or a C4 trace is the first diagnostic
  if the arm loses.
- **PACK can come out negative** if the l2pf planner cannot re-place the lost windows' prefetch.
- **COLEMB adds 165 MB per rank.** If startup memory accounting is tight, relaunch fails closed (a startup
  failure, not an exactness risk).
- **The COLEMB build adds about 1010 eager ARs at the first forward,** about 0.1–0.3 s once. With
  `RDMA_AR_AUDIT=200`, those calls use up the audit's 200 logged calls. Compare NEW_STREAM lines between arms, not
  the routed-call list.
- **The C4 audit arm is not a timing arm.**
