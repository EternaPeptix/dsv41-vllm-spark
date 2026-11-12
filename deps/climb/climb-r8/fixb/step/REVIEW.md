# r6-step review (climb round 6, step:review, 2026-11-09)

**Verdict: FIX.** The kit code looks inert when its flags are off and exact when they are on, as far as I can tell
without a GPU. The device-test deliverable is broken for C4. Its PACK-all GEMM gate does not exercise or assert the
serving op. Its mutant coverage is narrower than the brief and the kit summary claim. Fix those three things before
any device window. Nothing in the serving path needs to change before it can be installed.

## What I checked and ran

- **Live sha census (read-only, 8/8, today).** spark-one was reached over Tailscale because its WiFi is down.
  - engram.py `aa8aad1b` and dsv41_r4_small.py `3364d8c8` match the kit's bases.
  - vocab_parallel_embedding.py `10e17024`, cuda_communicator.py `87bf77e3`, rdma_all_reduce.py `26a5d834`,
    dspark.py `83acc8cc`, model_runner.py `aa3e9d9e` and dflash/speculator.py `17ddccf1` all match the kit's
    `live/` copies.
  - `dsv41_step.py` is absent on every node, and there is no CLUSTER_LOCK.
- **r5 composed base.** engram.py `c8170bd9` equals the FULL pin in `climb-r5/install-all.sh`. No r5 kit touches
  dsv41_r4_small.py.
- **Patched shas.** I recomputed them locally from the kit's EDITS: `d9bf5d59`, `34cf684e` and `e99f623d`. All three
  match KIT.json.
- **Reran on 512S1**, in a temp copy that I deleted afterwards (exo-venv313, torch 2.11 CPU):
  - `tests/test_step.py`: **28 passed**.
  - `STEP_SLOW=1 -k exhaustive`: **1 passed**.
  - `device_test.py --cpu-selftest`: **5 checks, 0 failed**.
  - I did not rerun `tools/compile_check.py` (spark-eight). No GPU test and no serving test ran.
- **Reproduced a device-test crash on CPU** (512S1, live dsv41_r4_small.py helpers). It is blocking item 1.

## Blocking

1. **`device_test.py --part c4` crashes before any C4 gate or mutant runs.**
   - `part_c4` passes the raw checkpoint scale `mtp.0.main_proj.scale` to the live `ColSplit`. That tensor is
     F8_E8M0 `[160, 480]`, a 32×32 block scale; I read the safetensors header read-only.
   - `rowmajor_scale()` accepts only the loaded per-row layout (`[>=N, >=K/32]` or swizzled 1-D), so it returns None.
   - `ColSplit.__init__` then raises `RuntimeError('unrecognised MXFP8 weight/scale layout')`. I reproduced this.
   - **Fix:** give `ColSplit` the serving-layout scale. Best is to build it exactly as the live loader's
     process-weights step does; for 32×32 blocks that should be `repeat_interleave(32, 0)` to `[5120, 480]`, but
     verify it against the loader. Then add an E0 gate: on the real weights, `ColSplit(...).local` + concat must equal
     the stock `ReplicatedLinear` forward bitwise for one M. That proves the test GEMM is the serving GEMM.
2. **The C3b "wkv GEMM run earlier == later" gate does not test or assert the serving op.**
   - The kit summary says the gate runs "both wkv GEMMs via the live FlashInfer mm_mxfp8 quant_method path". The code
     does not: it calls a hand-written `mxfp8_e4m3_quantize` + `mm_mxfp8(backend="cutlass")` with random weights and
     random swizzled scales.
   - E0 never checks what `engram.wkv.quant_method.apply` dispatches to. The checkpoint is `fp8` with block [32, 32]
     ue8m0, and the wkv is `[25600, 6144]` with a `[800, 192]` scale.
   - **Fix:** load `layers.{1,14}.engram.wkv` (weight and serving-layout scale), and call the live quant method's
     apply path or assert that it is exactly this mm_mxfp8 call. Then run both layers' GEMMs interleaved against stock
     order at every T, with the wrong-operand mutant. Without this fix, PACK=all has no device gate.
3. **Mutants do not run at every gated shape.** The brief requires that, and README/KIT claim it for C4.
   - `_c4_mutants` runs only for M ∈ {1, 4, 6, 16}, out of 26 served M.
   - The C3a mutants run only for the hostile table at T ≥ 5.
   - **Fix:** run the C4 mutants at every served M; they are cheap with `_sleep`. Run the C3a map and shard mutants at
     every T; plant the −0 id at index 0 for T < 5. Otherwise, correct the claims.

## Flag-off identity: OK

- Both patched files only gain one module-level bool and `if <bool>:` blocks. With the flags unset, `dsv41_step` is
  never imported.
- The AST fold test passes for live engram, r5 engram and r4_small. I reran it.
- The target_logits insertion short-circuits on `_DSV41_STEP_C4` before it reads anything. maybe_install and
  mainproj_split gain only a guarded block.
- The engram anchor is the file's last statement in both bases. The install block is module-level code at EOF, not
  inside the class.

## Exactness when on

### COLEMB: sound

- **Build.** The table is copied from the live forward: fused kernel + rdma AR, `[≤128, 5120]` chunks, two-shot.
- **Proof.** Every row is checked against `canon(W)` on its owner rank, across all 5120 columns. The cross-rank AR
  checksum must be equal, and `_tp_agree_min` keeps the lever off everywhere on any failure.
- **Serving.** The served value is the table row followed by a byte-copy column gather. Out-of-range and padded ids
  give +0 on both paths. `use_fused_embedding` is True in serving (unquantized, tp 8), so E0's op assertion is the
  right one.
- **Gap (nonblocking).** The startup proof exercises only the two-shot AR path. Serving T = 1..3 (≤ 30 KiB, below
  `RDMA_AR_TWOSHOT_MIN_BYTES`) would have used one-shot. The map should be identical, since both accumulate fp32 FADD
  and there is exactly one non-zero contributor, but nothing proves it.
  - Re-prove the special-valued rows (−0, NaN, denormal), plus a sample, at T = 1 during the build.
  - Better: add a rank-agreed first-use check per T, as r4 does with `checked[M]`. Compare the colemb output with
    `orig(self, ids)` bitwise, eagerly, before any capture. That also covers the real rdma `all_gather_cols` layout,
    which the device test only emulates.

### PACK look/all: sound

- **Layout.** `[T, tp, 2, H·hd]` and `[T, tp, 2, n]` views of a `dim=1`/`dim=-1` concat-gather equal the stock
  per-layer gathers. Byte copies only.
- **Pairing.** Layer b takes the stored rows and kv only when the sequence number and T match, so a skipped layer a
  makes b take the stock path.
- **Same GEMM.** Layer b's GEMM in `all` gets the same operand bytes, weights and shape through the same
  `quant_method.apply` call that `ColumnParallelLinear.forward` makes (bias None, `return_bias=False`; I checked live
  linear.py:603-621).
- **Under r5-prio.** With the hoist on, the hoist's own `embed` calls carry the packing: layer 0 packs and layer 13
  takes. With the hoist off, r5's Engram.forward line reaches the wrapped `embed`/`wkv`.
- **Prefill SP and large T.** `_dsv41_sp_engram` and large-T paths fall back to stock, or stay exact because `embed`
  returns stock-identical rows.
- **Gap (nonblocking).** The device test emulates the gathers with `torch.cat` and never runs the real rdma column
  gather on the new shapes ([T,6,256] and [T,6400]). A first-use rank-agreed check per T would close this, as for
  COLEMB.

### C4: sound

- **Ordering.**
  - S waits on an event recorded on main after the screen, which comes after the target graph.
  - Main waits on `ev_done` before the gather.
  - The next target replay is after that join, or after the `sample_tokens` drop-join.
  - No collective runs on S.
  - S's pool blocks are reused only after S's next `wait_event`.
- **The taps are written only by FULL replays.** They are the persistent `aux_hidden_states` buffers in
  cudagraph_utils.py:636-653, written only by FULL target graph replays.
- **The operand is the same.** `combine_hidden_states` is `main_norm(main_proj(cat(aux)))` (dspark.py:178), the r4
  hook calls `mainproj_split` on the same `cat`, and `c4_take` sits after `if not st` and before the verdict lookup.
- **The fork is never refused for PCP or the sharder in production.** `pcp_manager` is None (prefill-context size 1)
  and `batch_sharder` is None.
- **E7X skipped steps** end in a drop-join. That is harmless.
- **Audit (nonblocking).** While `DSV41_STEP_C4_AUDIT` is active, the hoisted `p.y` is served, not `y_ref`, so an
  audit arm is exact only if C4 already is. Serve `y_ref` during the audit window and compare.
- **Doc fix (nonblocking).** KIT.json's c4-audit arm uses N=500, but README and the summary say 2000.

## Device-test realism (beyond the blockers)

- **E0** is good: venv, CUDA_HOME, nvcc, sm_12x, vllm from the venv, the fused vocab_parallel_embedding op, the live
  r4 module, and `DSV41_DENSE_TACTIC` unset. Two additions are needed: blocker 2's wkv-op assertion, and blocker 1's
  ColSplit == stock gate.
- **C3a** uses the real checkpoint table. Its emulator's SASS kinds were checked by compile_check (not rerun by me).
  The real 8-rank map is covered only by the startup proof.

## Installer and composition

- **install.py.** Exact anchor counts, a TAG on every line, one backup per file, all-or-nothing planning before
  writes, uninstall only when the file equals patch(backup), module-file FOREIGN refusal, and idempotent. Tested in
  both base states and against the real r5-prio installer in both orders; I reran these tests.
- **No fleet driver (nonblocking).** `install.py` neither checks CLUSTER_LOCK nor stages or verifies 8/8, and there
  is no rollback. Run it only from the window's KIT_INSTALL step, or add an install-all like r5's.
- **r5 tooling reports FOREIGN while r6-step sits on top of r5 (nonblocking).** `climb-r5/install-all.sh`
  status, uninstall and preflight report FOREIGN on engram.py (e99f623d/34cf684e is in no r5 known state). So
  r5 uninstall and device-plan refuse until r6-step is uninstalled. The LIFO rule is documented; put it in the window
  plan explicitly.
- **Other kits.** No other kit wraps `VocabParallelEmbedding.forward`. OBLIT projects hidden states inside model.py
  and does not touch the embedding.
- **Composes with r1–r4.**
  - G12 is refused.
  - DTGRAPH is on in production (launcher line 117). Its DT region exits capture at collectives, and the colemb
    lookup is capturable there.
  - CERT M = 2–96.

## Estimates (t2)

The t2 numbers in `scripts/*_t2.txt` back the kit's ranges: E-pre AR 24.5–31.5 µs, rows AG 18–22 µs, wkv AG
22.7–32 µs, drafter embedding AR ~30.8 µs, and screen-end → CatArray 145–215 µs against ~70 µs of side chain.

One cost is not counted (nonblocking). Under `look`/`all`, layer b's Engram lookup event is now waited on at layer 1
instead of layer 14. Both lookups run in order on one prefetch stream, so any part of b's lookup still running at
layer 1 lands on the critical path. Layer 0 probably hides it, but measure it from t2 (b-lookup end vs layer-1 rows
AG start) and fold it into PACK's low bound.

The overall central of about +0.48% G1 (~0.1 ms/step) is plausible. C4's low bound of −0.3% (from the G12 prior)
is honest. Only C-T-C-T decides.

## Not run by me

- Any GPU test: device_test GPU parts, compile_check rerun, the serving-time COLEMB proof, the C4 audit.
- Any install on a Spark, any arm, exact.py, the G1 digest, det.

## Fixes (step:fix, 2026-11-09)

All three blockers are fixed in `device_test.py`. Five of the eight nonblocking items are fixed or measured, and
three stay open. The kit module changed in one place (the C4 audit serves the stock y). That changes its sha from
`66d697b9` to `ebe2f7bb`. The two patched live files and their shas (`d9bf5d59`, `34cf684e`, `e99f623d`) are
unchanged. **No GPU test and no serving arm ran in this pass either.** The device test still has to run on a
stopped node. Pre-fix copies are in `prev/`.

### Blocking 1: C4 gave ColSplit the raw checkpoint scale. Fixed.

- **New E0 builder.** `serving_linear()` builds `mtp.0.main_proj` through the live quant-method lifecycle, as
  model loading does:
  1. `DeepseekV4FP8Config(checkpoint quantization_config).get_quant_method` returns
     `ModelOptLinearMethod(kMxfp8Static, CkptCtx(scale_block_size=(32, 32)))`.
  2. The live scale loader, `KMxfp8Static.get_scale_weight_loader`, turns the e8m0 [160, 480] block scale into
     uint8 `repeat_interleave(32, 0)` = [5120, 480].
  3. The live `select_linear_kernel` runs.
  4. The live `ModelOptLinearMethod.process_weights_after_loading` runs, leaving the swizzled 1-D serving scale.
- **C4 uses that layer.** `part_c4` now uses this layer. The raw tensor never reaches ColSplit.
- **New E0 gates.**
  - The method is `ModelOptLinearMethod` with the default `FormatScheme`.
  - The kernel is `FlashInferCutlassMxfp8LinearKernel`, and its `apply_weights` comes from the live module.
  - The scale is 1-D.
  - `ReplicatedLinear.forward` and `ColumnParallelLinear.forward` contain exactly
    `self.quant_method.apply(self, x|input_, bias)`.
  - **concat_r `ColSplit(main_proj, r, 8).local(x)` == `main_proj.quant_method.apply(main_proj, x, None)`
    bitwise** at M = 1, 7, 32, with a rank-order-reversed mutant at each M. Any E0 failure exits 2.
- **Verified on CPU** (spark-eight, live vllm032, real checkpoint, no GPU; `tools/serving_layers_cpu_check.py`,
  24 checks, 0 failed):
  - the live kernel selection returns `FlashInferCutlassMxfp8LinearKernel`;
  - the processed scale == `swizzle_mxfp8_scale(repeat_interleave(raw, 32, 0))`;
  - `rowmajor_scale` recovers [5120, 480];
  - the live `ColSplit` builds for all 8 ranks, with an exact weight slice and swizzled scale;
  - the raw scale still raises `unrecognised MXFP8 weight/scale layout`, which reproduces the review's crash as a
    caught mutant.
- **Not run.** The GEMM-level gate (ColSplit == stock) needs a GPU.

### Blocking 2: the C3b PACK=all GEMM gate did not use the serving op. Fixed.

- **Removed.** The hand-written `mxfp8_e4m3_quantize` + `mm_mxfp8` call on random weights is gone.
- **E0 builds the real wkv layers.** It builds `layers.1.engram.wkv` and `layers.14.engram.wkv` for all 8
  ColumnParallel rank slices with the same builder. The scale goes [800, 192] → [25600, 192], then the rank's
  3200 rows, narrowed on output_dim 0 as `load_column_parallel_weight` does. E0 gates the method and kernel as for
  main_proj.
- **New `part_pack_gemm`.** It runs the kit's own `_wrap_wkv` / `wkv_first` / `wkv_second` on those layers.
  - The call is `wa.quant_method.apply(wa, x, None)`, the same call `ColumnParallelLinear.forward` makes. E0
    asserts that.
  - It runs for every rank and every served T (1..16, 20, 24, 25, 30, 32, 40, 48, 64, 96, 128).
  - **Kit order:** layer 1's wkv forward runs both GEMMs and the packed gather. The emulated column gather uses
    the other ranks' real slices. Then the layer 2–13 stand-in GEMMs run (main_proj and other wkv slices, same
    kernel). Then layer 14 takes the stored kv.
  - **Stock order:** layer 1's GEMM for all slices, then the stand-ins, then layer 14's GEMM.
  - Gate: both layers' gathered kv are bitwise equal.
  - Mutants at every T: layer 14 fed layer 1's rows, and layer 14's GEMM run with layer 1's weights.
- **CPU check of the harness.** `--cpu-selftest` runs it with stand-in matmul GEMMs: 60 gates pass and 20/20
  mutants are caught.
- **Not run.** The real GEMMs need a GPU.

### Blocking 3: mutants did not run at every gated shape. Fixed.

- **C4.** `_c4_mutants` now runs at every served M (26) for every rank. The `M in {1,4,6,16}` condition is gone.
- **C3a.** The map, shard and invalid mutants now run at every served T, T = 1 included. Each mutant gets its own
  id vector from `c3a_mutant_ids()`, with the row it must change at index 0:
  - map: the −0 row, then the NaN row;
  - shard: an ordinary row (12345);
  - invalid: −1, then V+7.

  The same vector must also pass the real gate. The old fixed vector would have missed the shard mutant at T = 1,
  because row 7 is all ±0 and identical in every shard.
- **CPU check.** `--cpu-selftest` checks on CPU that each vector catches its mutant at all 26 T.
- **C3b.** The layout mutants already ran at every T. The new GEMM mutants run at every T too.

### Nonblocking

| item | status |
|---|---|
| COLEMB proof covers only two-shot AR (T=1..3 serve one-shot); per-T rank-agreed first-use check | **open.** Not built. The DSpark lookup's first use is inside the drafter-graph capture, where an eager check cannot run, so a per-T check would need a pre-capture warm call per T. The startup proof is still two-shot only. |
| PACK: real rdma column gather on [T,6,256] / [T,6400] never run; first-use check | **open.** Not built. Still emulated with `torch.cat` in the device test. The serving-time protection is exact.py, the G1 digest and det in the arm. |
| C4 audit serves the hoisted `p.y` | **fixed.** `c4_take` gathers `_audit()`'s stock `y_ref` during the N audited steps, so the audit arm is exact whether or not C4 is. New `tests/test_step.py::test_c4_audit_serves_stock_y[False/True]`: the stock y is served in the window, a wrong hoisted y is counted and disables C4, and after the window the hoisted y is served. |
| KIT.json c4-audit N=500 vs 2000 | **fixed.** It is 2000 in KIT.json (arm env and note) and in README §2.3. |
| install.py: no CLUSTER_LOCK check, no fleet driver | **documented, not built.** README §4 now has an explicit window plan: KIT_INSTALL only, 8/8 `status` against the KIT.json shas, and the order relative to r5. There is still no 8/8 driver with staging, sha verification or rollback. |
| r5 install-all reports engram.py FOREIGN while r6-step is on top | **documented.** The LIFO rule is in README §4 "Window plan" and in KIT.json `uninstall`: run `install.py uninstall` 8/8 before any r5 install-all status, uninstall, preflight or device-plan. |
| PACK: the new wait on layer 14's lookup at layer 1 | **measured.** `scripts/engwait.py` on all 6 t2 traces, 360 forwards: layer 14's lookup ends at least 379.1 µs before layer 1's rows AG starts (p10 405.9, p50 509.5) and is never late. The wait costs 0 on t2, so PACK's low bound is unchanged. README §1.5 and the KIT.json basis now say so. |
| "Verified OK" record | No action. |

### Tests rerun in this pass (all CPU; details in TEST-RESULTS.txt)

- `tests/test_step.py` on 512S1: **30 passed** (28 + 2 new).
- Exhaustive AR-map test: **1 passed**.
- `device_test.py --cpu-selftest`: **11 checks (6 mutants), 0 failed**.
- `tools/serving_layers_cpu_check.py` on spark-eight: **24 checks, 0 failed**.
- `tools/compile_check.py` on spark-eight: **ALL PASS**.
- Both spark-eight runs used `CUDA_VISIBLE_DEVICES=`, nice 19, and no bytecode or Triton cache outside
  `~/climb-scratch/r6-step`. CLUSTER_LOCK was absent.

### Still not run

- `device_test.py` on a GPU: every E0 GEMM gate, C3a, C3b including the PACK=all serving-wkv test, and C4 with its
  mutants.
- The COLEMB startup proof on the real AR.
- The C4 audit.
- Any install, arm, exact.py, G1 digest or det.
