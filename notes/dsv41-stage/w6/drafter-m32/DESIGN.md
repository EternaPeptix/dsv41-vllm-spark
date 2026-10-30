# w6 lane drafter-m32: DSpark drafter routed MoE on the M32 native (`DSV41_DRAFT_MOE_M32`)

**Verdict: exact by construction, ready pending the stopped-window GPU check.** The drafter's routed MoE can run on
the M32 native with bitwise-identical outputs at every row count. Both kernels instantiate the same
`sm120_fp8_fp4_gemm_1d1d.cuh` template. The source is byte-identical between the installed DeepGEMM and the M32 trees,
and the arithmetic per output element is the same. The only differences are tiling (BLOCK_M 128→32, BLOCK_N 128→64),
pipeline depth and which bytes get loaded or stored. The change is Python-only (one patched file plus one new module),
uses no new native and no new tree, and is off by default.

Expected (corrected after review, 2026-10-28): **-0.30..-0.60 ms per 4-session step** (central -0.45..-0.50, i.e.
G4min_hat **+0.7..+1.35%**, central about +1.05%) and **-0.10..-0.16 ms per 1-user step** (G1 **+0.4..+0.65%**). Larger
savings at conc8/16 are likely but unmeasured. The bench gate below runs before any A/B.

**Window 6 bundle (2026-10-28).** This lane's `deep_gemm_moe.py` hunks are merged with fc1fc2-pipe's
(`DSV41_MOE_FC2_HEADPF`) into ONE file, installed by ONE installer: `.../w6/bundle/` (`deep_gemm_moe.py.all` 34e8b402,
merged; `deep_gemm_moe.py.drm32` = this lane's 9364f121, byte-identical, the fallback if H is NO-GO; `w6-bundle.sh`
replaces `ops/drm32-all.sh install|uninstall`). `bundle/tests/test_bundle_cpu.py` proves that for the E=128 drafter
instances the merged file is AST-identical to 9364f121 (the H code is gated to E=384), so this lane's qualification
carries over; step A2 can also run the merged file directly (`DRM32_CAND`, section 7).

## 1. What runs today and with the flag

DSpark drafter MoE: mtp.0-2, 128 routed experts, top-3, hidden 5120. Per TP8 rank, FC1 is padded to
N=768 (`[gate 288+96 | up 288+96]`; `DSV41_MOE_COMPACT` only applies to E=384) and FC2 has K=384, packed as FP4 bytes 192.
Inputs are FP8 e4m3 activations with UE8M0 scales per 128 and MXFP4 weights with UE8M0 scales per 32.

| | today (flag unset) | `DSV41_DRAFT_MOE_M32=1`, 1 <= M <= 160 |
|---|---|---|
| expert alignment / M_sum | 128: `3M + min(3M,128)*127` (M=5: 1920, M=20: 7680) | 32: `3M + min(3M,128)*31` (M=5: 480, M=20: 1920), 4x fewer rows |
| GEMM | installed DeepGEMM (`vllm/third_party/deep_gemm`, `_C_extension` 34f79631), JIT entries in use: FC1 `5559d059` (G128 BM128 BN128 BK128, 2 stages, EpiSubM 128) and, at some M, `2201c753` (BM128 BN64 BK128, 3 stages), FC2 `09f26aa9` (G128 BM128 BN128 BK128) | the M32 native the target already uses (`get_target_m32_kernel()`, `DSV41_MOE_M32_G1=all` tree `all-base`, `_C` b1c4eff8): G128 BM32 BN64 BK128, 7 stages, EpiSubM 32 (2 new JIT entries) |
| SiLU-mul-FP8-quant | `_silu_mul_quant_fp8_packed_kernel` variant chosen from M_sum (M>=3: live rows, EXPERT_ALIGNMENT 128) | same kernel, variant chosen from the smaller M_sum (M<=10 dense; M>=11 live rows, EXPERT_ALIGNMENT 32) |
| permute / gather | ep_scatter at align 128; ep_gather | ep_scatter at align 32; the same ep_gather |

M > 160 (prefill chunks and mixed steps) stays on the 128-row path. Every other drafter kernel is unchanged: dense, head,
Markov, mHC, rdma_ar and the shared expert. The patch also leaves the target path, AV, the scheduler and `dspark.py` alone.

## 2. Why it is bitwise

1. **Same template, same per-element arithmetic.**
   - `impls/sm120_fp8_fp4_gemm_1d1d.cuh` 9969531b, plus `mma/sm120.cuh`, `common/sm120_utils.cuh`, `scheduler/sm120_gemm.cuh`
     and `tma_copy.cuh`, are byte-identical between the installed DeepGEMM and the M32 v1 tree, and so are all cute/cutlass headers.
   - Correction (review 1a): of the 16 headers in the kernel's transitive include closure, 13 are byte-identical in all three
     trees (installed, M32 v1, all-base). Two differ between the installed DeepGEMM and M32/all-base, additively and in
     helpers the 9969531b kernel never uses: `common/utils.cuh` (6791c5cc installed vs 9fd0fdb8: the installed copy adds a
     `shfl_sync` template and `#include <type_traits>`) and `ptx/tma.cuh` (519bf2f4 vs 2fa03736: M32 adds
     `tensor_map_replace_global_dim_in_smem`). An unused helper does not change codegen; the GPU test decides.
   - The 'all-base' tree header (faafb3f6) adds only `kG1*` features. Their gate requires `kNumGroups == 384`, so for E=128
     every `if constexpr (kG1…)` block is compiled out.
   - The one unguarded edit is an explicit `kG1BHint` cache-hint argument. With kG1 off it equals EVICT_NORMAL, which is
     `tma::copy`'s default. In all cases it is an L2 policy, not arithmetic.
   - Our config is FP8 A × FP4 B with kGranKA=128, kGranKB=32 and BLOCK_K=128. It takes the `!kUseSFMajorLoop`,
     `!kUsePerNTileX4` branch in both kernels. Per output element (row r, column c) the arithmetic is:
     `accum = 0; for each k32 chunk j in K order: accum = mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.f32.e4m3.e2m1.f32.ue8m0(a[r, j], b[c, j], sfa[r, j/4], sfb[c, j], accum)`,
     then `bf16 = RN(accum)` (`cd_dtype_t(v)`) in the epilogue.
   - BLOCK_M, BLOCK_N, stages, warp layout (2x4 vs 4x2) and EpiSubM only decide which thread holds which accumulator and
     how bytes are staged. Each accumulator still receives exactly one MMA per k32 chunk, in K order.
2. **No split-K, same K tile.**
   - The SM120 heuristic (`heuristics/sm120.hpp` in the M32 native source) gives split-K only to `GemmType::Normal`.
     Grouped-contiguous calls are always `split_k_factor = 1`.
   - BLOCK_K stays 128 because block-K 64 is excluded for mixed dtypes and for alignment 32.
   - All 9 installed G128/G384 entries and the 3 M32 entries read from the JIT caches have split-K 1 and BK 128.
   - The GPU test re-audits every G128 instantiation it creates.
3. **Same nvcc flags except the arch tag.** Both use `-O3`, `use_fast_math=false` and register-usage-level 10. The
   installed build uses `sm_121f`; the M32 natives use `sm_120f`. (`live/dg/jit/meta-compare.txt` missed the FC1 entry
   because of a path typo, `559d0591...` for `5559d059...`; re-read read-only on spark-one 2026-10-28 and appended there:
   5559d059 and 2201c753 are `sm_121f` with flags identical to 09f26aa9.) The same GB10 runs both, and MMA, cvt.rn and
   FMA-free accumulation semantics do not depend on the arch tag.
4. **Same operand bytes.**
   - Weights and weight scales are the same tensors (`deepgemm_post_process_weight_scale_block` at load).
   - Activation scales are power-of-two fp32. Both natives' `transpose_and_pack_fp32_into_ue8m0` extract the exponent
     byte exactly (`values >> 23`); the vendored one asserts a zero mantissa.
   - ep_scatter copies rows. Row order inside an expert segment already depends on atomics today, and outputs are
     row-position independent. The g2 test showed this for the M32 kernel: positional mismatches, yet bitwise-equal final
     outputs.
5. **SiLU-quant is per row.** clamp → `gate/(1+exp(-gate))*up` → bf16 round-trip → exact max over each 128-group →
   power-of-two scale → `clamp(y/scale)` → FP8. BM, PACKS_PER_CTA and EXPERT_ALIGNMENT change tiling and which rows are
   written, not the per-element code. The 10-24 live-act test showed dense vs live variants bitwise on live rows for this
   drafter geometry. Padding rows are never consumed: FC2 rows depend only on their own A row, and ep_gather reads live
   rows only. `ensure_zero_padding=False` on the M32 path is the target's production contract since 10-21.
6. **Gather is layout-independent.** `_fwd_kernel_ep_gather` accumulates fp32 `out[t] += fc2[inv_perm[t,k]] * w[t,k]` for
   k = 0..2 in order. Its grid depends on tokens, not M_sum.
7. **Drafts and AV.** The MoE output is the only changed tensor, so DSpark drafts and confidences are bitwise unchanged.
   Under the AV pin, verify budgets are identical, and so are target numerics.
   - Unpinned production re-profiles AV costs at boot. That holds for every speed lever, and exactness is judged under the
     pin as in every window.
   - G4 outputs are not reproducible across launches anyway because of the batching lottery. Real-traffic drafter
     bitwiseness at multi-user row counts is therefore proven by the V (verify) instance below, not by G4 digests.

## 3. Files (live shas diffed against, identical on 8/8 at 2026-10-28 ~18:50 UTC)

Patched (site `vllm/model_executor/layers/fused_moe/experts/`):
- `deep_gemm_moe.py` **5162dc0f → 9364f121** (`patch/deep_gemm_moe.py.patch`, generated by `ops/make_patch.py` with exact unique replacements). Three methods of `DeepGemmFP4Experts` change:
  - `__init__`: one guarded block. Drafter-geometry instances share the target's `TargetM32Kernel`, and get the verifier in verify mode.
  - `_workspace_m_alignment`: one gate that returns 32 for `(768, 5120, 3, 128)`, `1 <= M <= DSV41_DRAFT_M32_MAX_M`, with `DSV41_SM121_TARGET_M32=1` and SM121.
  - `apply`: records the original args only in verify mode; `block_m = 32` only while the verify shadow runs; calls the shadow after the served gather.
- `dsv41_draft_m32.py` **new, a4a83834**: flag parsing, eligibility, announce line, `DraftM32Verifier` (device counters, in-graph, reporter thread as in the D05/FC2C verifiers).

Unchanged dependencies (checked): `sm121_target_m32.py` ea4a8814, `deep_gemm_utils.py` a4a0486a, `fp8_utils.py` b35c0561, `routed_experts.py` 1a2e7324, `nvidia/dspark.py` 36811ca9, installed DeepGEMM `_C_extension` 34f79631 / header 9969531b, `~/dsv41-stage/g1/m32-bw/trees/all-base` header faafb3f6 (manifest 74d0d4f2), launcher **afa97ae6**. The supervisor manifest (0c96a727fb81) does not list `deep_gemm_moe.py`, so installing changes it only at promotion, through the launcher line. The site tree gains 1 file (5,454 → 5,455).

## 4. Flag

- `DSV41_DRAFT_MOE_M32=0` / unset: original behaviour. With the flag off, `apply()` is AST-identical to the live one (CPU test).
- `=1`: drafter calls with 1 <= M <= `DSV41_DRAFT_M32_MAX_M` (default 160 = 32 seqs × block 5) use the M32 native.
- `=verify`: serves the original 128-row bytes, re-runs the M32 path on the same inputs into private buffers and counts
  mismatching int16 words per drafter layer on the device. This works eager and inside the drafter CUDA graphs.
  - Allocation is about 70 MB at the first eager call, before capture.
  - Report: `touch ~/dsv41-logs/draft-m32-verify.trigger`, at least 900 s after worker start →
    `draft-m32-verify.<host>.<pid>.json`.
- The flag is inert without `DSV41_SM121_TARGET_M32=1` (it warns once).

## 5. Expected gain (arithmetic)

Live drafter MoE GEMMs from the nsys traces of afa97ae6 (rank 0; `prof/q0.*.sqlite`, 60 drafter graphs each):

| step | FC1 L0/L1/L2 (us) | FC2 L0/L1/L2 (us) | GEMM sum per drafter step |
|---|---|---|---|
| 4 sessions (q0.1, 20 rows) | 412.7 / 299.5 / 290.6 | 280.4 / 187.4 / 178.7 | **1,649 us** (q0.3: 1,476) |
| 1 user (q0.2, 5 rows) | 162.7 / 92.4 / 67.7 | 79.4 / 47.3 / 37.6 | **487 us** |

Bytes per live expert and layer (128-row tile vs 32-row tile):
- FC1:
  - weights 2.09 MB in both;
  - A box 0.66 MB (128 rows × 5120) vs 0.16 MB;
  - D store 0.20 MB vs 0.05 MB.
- FC2:
  - weights 1.04 MB in both;
  - D store **1.31 MB** (128 rows × 5120 × bf16, almost all padding rows) vs 0.33 MB;
  - A 0.05 MB vs 0.01 MB.
- Total: **5.35 → 3.68 MB (-31%)**. If the GEMMs stream all bytes at the same rate, the saving is 0.31 × 1.65 = **-0.51 ms** at 4 sessions and 0.31 × 0.49 = **-0.15 ms** at 1 user.
- Lower bound, if the A re-reads all hit L2: 21% → -0.35 ms / -0.10 ms.
- Upper reference, OPTIMISTIC (review 3): the 10-24 measurement in one harness put the target's 128-row path at about 1.66x the M32 path per active expert (M=128: 8,649 us vs M=96: 4,584 us; ~330 vs ~300 active experts). That would be -0.65 ms / -0.19 ms, but it is a warm-L2 full MoE apply whose 128-row side is inflated by glue that grows with M_sum 49,536 (zero-fill of 7.9 MB of fp32 scales, the dense SiLU over padding rows); at the drafter's M_sum of 7,680 those costs are much smaller. Treat it as a padding/glue-inflated bound, not a central value.
- Corrected range (review 3): 4 sessions **-0.30..-0.60 ms per step** (central about -0.45..-0.50); 1 user **-0.10..-0.16 ms**.
- Step shares:
  - 4-session step ~44 ms → **-0.7..-1.35%** (central about -1.05%): G4min_hat 49.25 → ~49.75 production, 49.42 → ~49.9..50.0 pinned. This helps but probably does not close the 1.2-1.5% gap alone at the central value (review 3).
  - C1 25.6-29.6 ms → G1 **+0.4..+0.65%** (review 3).
  - conc8/16 (drafter M=40/80, M_sum 15,360/16,512 → 3,840/4,224): larger, est. -0.6..-1.5 ms per step.

## 6. CPU tests (done on this Mac, `tests/test_cpu.py`, 1.6 s, 40/40 PASS, `tests/cpu-test.txt`)
- base = live 5162dc0f. The patch reproduces the site file (9364f121), and both files compile.
- **Flag-off AST equivalence:** substitute flag state, fold the constants and propagate `dsv41_call = None`. The result is `ast.dump`-identical to the live `apply`. `__init__` and `_workspace_m_alignment` differ only by inserted guarded blocks. The other 20 top-level nodes are identical.
- `_workspace_m_alignment` sweep: 417,312 base/patched comparisons over 7 geometries, M 0..16384 and 216 env combinations (M32/M64/COMPACT/caps/flag values 0, off, verify, 1, ON). Only the drafter geometry with the flag on, target M32 on and M within the cap changes (128 → 32); 3,200 expected changes, 0 unexpected.
- Module checks: mode parsing, cap bounds, the M_sum formula, and verifier buffers sized for M=160 (4,448 rows: 22.8 + 45.5 MB).
- `tests/cpu-table.json`: M_sum and the SiLU variant pair per M. The GPU M list covers all 6 variant pairs.
- A numpy model of the layout argument: scatter in random atomics order at two alignments → bitwise-equal outputs. Two negative controls show the check can fail: a K-order change flips accumulator bits, and an injected layout bug is caught. This is a model only; real-kernel equality is the GPU test's job.

## 7. GPU bitwise test (stopped window; `tests/test_draft_m32_gpu.py`, `ops/run-gpu-tests.sh`, `ops/run-gpu-tests-all.sh`)

**Runs on 8/8 in parallel**, each on its own TP slice of the real mtp.0-2 checkpoint experts (through the serving loader method). It mirrors the launcher's MoE env and uses **the serving `TargetM32Kernel` object** (all-base + D05 + FC2C), with every tree and JIT cache, the vLLM DeepGEMM cache and the Triton cache redirected to private byte copies. Serving caches and site files are snapshotted before and after and must be unchanged.

- **A, eager:**
  - Arms: base (live file) vs off / on / verify-served, compared bitwise on int16 views.
  - Scope: 3 layers × 28 M values (1, 2, 3, 4, 5, 6, 10, 11, 12, 15, 16, 20, 24, 25, 30, 35, 40, 48, 60, 80, 96, 100, 120, 128, 155, 160, 161, 200; this includes 1/6/12/24/48 and both sides of the cap) × 7 route kinds (valid, DSpark-clustered, skewed hot expert with >32-row segments, edge experts 0/127, half padding, 30% holes, all padding) × 4 hidden-state distributions. That is 588 cases per rank.
  - Every run uses NaN-poisoned workspaces and allocator pool.
  - Path spies check that base/off take the vendored path with alignment 128, that on takes the M32 native with alignment 32 for M <= 160 and the vendored path above, and that verify takes the vendored path plus the M32 shadow.
  - Verify-shadow counters must show 0 mismatches.
- **B:** on any failure, per-expert multisets of live FC1/FC2 rows to localise the stage.
- **C, graphs:**
  - Per (layer, M), capture on, base and verify, then **200 replays of on** (50 each of base and verify), cycling 16 fresh input sets into the static inputs.
  - Each replay is compared on the device against the eager base reference.
  - A second pass on mtp.0 adds a **co-runner** (large copies on a side stream). This is insurance only: the change adds no concurrency.
  - Per rank this is about 25,000 replays.
- **D, JIT audit:** every kNumGroups=128 instantiation in both caches must have BK128, split-K 1, kGranKA 128 / kGranKB 32, B FP4 K-major, no accumulation, bf16 out and EpilogueIdentity. Both drafter M32 shapes (FC1 N768/K5120, FC2 N5120/K384) must be present.
  - **JIT-count guard (review fix 2):** after the M sweep the private `g1root/jit/all-base/cache` holds EXACTLY two kNumGroups=128 entries, both BLOCK_M 32 (FC1 N768/K5120 and FC2 N5120/K384), and every entry that appeared during the run is one of them (`event: jit_count_guard`). These are the same two `ops/prejit-node.sh` builds at M = 5 and 20, so serving cannot JIT a third drafter kernel at an M prejit did not visit. Decisive in `full` (28 M values, 1..200); `quick` checks M 1/5/20/161 only.
  - **Candidate (bundle):** `--cand PATH` (or `DRM32_CAND=PATH` for `ops/run-gpu-tests.sh` / `-all.sh`) tests another candidate file; only 9364f121 (this lane) and 34e8b402 (the w6 bundle's merged `deep_gemm_moe.py.all`) are accepted.
- **T, bench (ranks 0 and 7):** one graph per arm containing the 3-layer drafter step (each layer's live weights exceed the 24 MB L2), base / M32 / base (A/B/A), at M = 1, 5, 10, 15, 20, 25, 40, 80, 160 with clustered routes.
- **Gates:**
  - all 8 nodes `passed`, `graph_mismatch_words` 0, `jit_audit_ok`, serving unchanged;
  - **bench: saved ≥ 250 us per drafter step at M=20 (4 requests) on both ranks**, otherwise STOP (not worth it).
- **Time:** about 10-15 min (JIT for 2 kernels about 1-2 min, Triton recompiles into the private cache, 3 checkpoint shards). Start with `quick` (about 3 min) as a smoke run.

## 8. Pinned A/B plan (short suite, foreign-traffic rule; `ab/inst.sh` = g5host `inst.sh` fb83b234 + arms X/V, `ab/inst.sh.patch`)

Pin `~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json` (eeac4f97); every pinned launch must log "pin hit". Clock 2000 MHz.

0. **Stopped window, from the Mac:**
   - `bash ops/drm32-all.sh stage` → `bash ops/run-gpu-tests-all.sh quick`, then `full`; the gates of section 7 must hold.
   - `bash ops/drm32-all.sh install`: preflight 8/8, backups `.bak-w6-drm32`, inert.
   - `bash ops/drm32-all.sh prejit`: 2 new entries per node in `g1/m32-bw/jit/all-base/cache`, recorded in `jit-added.<host>.txt`. Nothing existing changes.
   - `bash ops/drm32-all.sh status`.
1. **C1** (pin, flag unset). This proves the install is inert:
   - exact.py **a8c3394af9d2**;
   - counters **acc=909 dtok=2335 drafts=467 gen=1373**, per-position **348/233/171/90/67**;
   - G1 outputs equal to the pinned reference digest **bc1e8386878bd2b7**.
2. **V1** (pin + `DSV41_DRAFT_MOE_M32=verify`; suite `init matched g1 g4 conc8 conc16 exact2 end`):
   - the same exact.py and G1 checks as C1 (served path = original);
   - then `ab/collect-verify.sh V1` must give **8/8 OK, 0 mismatching words** on all 3 drafter layers, `capture_misses` 0 and calls > 0. This covers the real-traffic drafter row counts 5-80 (G1 / G4 / conc8 / conc16), eager and in graphs.
   - V's timings are not used.
3. **X1** (pin + `DSV41_DRAFT_MOE_M32=1`), **C2**, **X2** (ABAB with C1). Add C3/X3 only if a pair straddles the bar.
   Every X launch must have:
   - the announce line on rank 0 (the `drm32_lines` guard in `ab/inst.sh`) and no "ignored" warning;
   - exact.py a8c3394af9d2 with the pin counters and per-position values;
   - the G1 reference digest;
   - matched C1 AL equal to the adjacent C's (drafts are identical, so per-trial AL is identical).
4. **Decision rule** (window rule):
   - exact on every launch, and V 8/8 OK;
   - **G4min_hat ≥ +0.5% in both pinned pairs**, or G1 ≥ +0.5% in both;
   - conc8/16 not below the control range;
   - early stop if the first pair is negative (trim rule).
   - Expected (review-corrected): G4min_hat +0.7..+1.35%, G1 +0.4..+0.65%, matched C1 -0.10..-0.16 ms, conc8/16 larger (unmeasured).
5. **Promotion:**
   - Launcher line `export DSV41_DRAFT_MOE_M32=${DSV41_DRAFT_MOE_M32:-1}` after the `DSV41_MOE_FC2C` line. This changes the launcher sha and the supervisor manifest.
   - Production instance plus the regression gate (kit 08) in the runbook's final check.
   - After promotion, the pin's draft-cost table is slightly pessimistic. Re-pinning is a new-anchor user decision, as for mHC.
6. **Rollback:**
   - one launch: `DSV41_DRAFT_MOE_M32=0`;
   - permanent: the launcher backup;
   - files: with the service stopped, `bash ops/drm32-all.sh uninstall` (restores 5162dc0f, removes the module). The 2 pre-JIT entries are inert and may stay.

## 9. Risks
- **Hardware assumption:** the MMA result for an element does not depend on its position in the 16x8 tile or on BLOCK_N. It is the same instruction with the same operands, and position independence is already shown empirically for the target M32 rows. The GPU test on real weights is the proof.
- **Triton SiLU variants** at the new M_sum values: (BM, PPC, EXPERT_ALIGNMENT) = (1, 2, 0) at M ≤ 5, (4, 1, 0) at M 6..10, (4, 1, 32) at M ≥ 11. The GPU M list hits every variant pair.
- **Serving-only paths:**
  - `TargetM32Kernel.gemm`'s `g1_la` assert (A contiguous and 16-B aligned) now also runs for drafter calls. It would fail loudly at the first drafter call, never silently.
  - verify mode adds about 70 MB and about 2x drafter MoE time (diagnostic only).
  - The first-call JIT is pre-built by `prejit`.
- **Interaction with other w6 lanes:** fc1fc2-pipe, fc2c-contig and qb-l2pf change M32 trees or kernels. With today's `kNumGroups == 384` gate the drafter keeps the plain tiling. A lane that widens a gate, or changes the shared base native or tree, must run this harness (the drafter G128 shapes) too. pdl-glue edits to the SiLU kernel apply identically to both paths.
- **Gain uncertainty** (-0.30..-0.60 ms after review): handled by the bench gate before any A/B.
- **Not covered by the test:** the rollback combinations `DSV41_MOE_M32_G1=0` or the M32 v1 native. They rest on the same argument but are untested; run `ops/run-gpu-tests.sh` with that env before relying on such a rollback with the flag on.

## 10. Follow-ups (not in this patch; each exact by the same argument, each needs its own bitwise test)
- **Tier 2: the 'all' features for E=128.** A new m32-bw tree with the kG1 gate widened to `kNumGroups ∈ {128, 384}`: FC1 pf + la + ds + D04 + EF; FC2 ds + D04 + EF. Est. a further -0.1..-0.2 ms at 4 sessions (the FC2 D store goes to live rows only). It needs a new pinned tree, a separate native instance (a .so copy) or target re-qualification, and pre-JIT. About 1-2 days.
- **Tier 3: compact drafter FC1** (N 768 → 576, -25% FC1 weight bytes, est. -0.15..-0.25 ms at 4 sessions). Widen the `mxfp4.py` create_weights gate, the fast-loader expert-row layout and `_dsv41_compact_fc1`. The FC2 input bytes stay identical, as in the target's 10-21 qualification.
- **FC2C for the drafter:** the native host code hardcodes `num_groups == 384`, so this needs a rebuild.

## 11. Bundle (`/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad/w6/drafter-m32/`)
- `base/…/deep_gemm_moe.py` (live 5162dc0f), `site/…/deep_gemm_moe.py` (9364f121), `site/…/dsv41_draft_m32.py` (a4a83834), `patch/*.patch`, `ops/make_patch.py`.
- `tests/test_cpu.py` (+ `cpu-test.txt`, `cpu-table.json`), `tests/test_draft_m32_gpu.py`.
- `ops/run-gpu-tests.sh` (node), `ops/run-gpu-tests-all.sh` (Mac, 8/8), `ops/drm32-all.sh` (stage | check | status | install | uninstall | prejit), `ops/drm32-node.sh`, `ops/prejit-node.sh`.
- `ab/inst.sh` (+ `.patch` vs g5host fb83b234), `ab/collect-verify.sh`.
- `MANIFEST.sha256` (the stage set); `live/` (live copies and shas used for the diff, nsys-derived kernel tables in `prof/`).
