# w6 lane fc1fc2-pipe: MoE FC1 -> FC2 per-expert pipelining

Offline prep, 2026-10-28, Mac only. Read-only ssh (`nice -n 19`) fetched the live files; no restart, no GPU job, no write on any node.
Lane dir: `/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad/w6/fc1fc2-pipe/`.

## 0. Verdict

- **Per-expert FC1 -> FC2 pipelining as specified is NOT worth building.** Starting an expert's FC2 as soon as its 9 FC1 tiles and its SwiGLU rows finish can be made bitwise. On the live config, though, it can only remove the FC1 -> FC2 boundary bubble, and that bubble is small:
  - measured at **4.2 us per layer at 4 sessions** (0.17 ms per step) and 5.7 us at 1 user;
  - **ceiling about 0.17-0.34 ms per 4-session step** including FC1's drain and FC2's ramp, not the -0.4..-1.0 ms on the list;
  - a persistent pipeline would also need de-aliased workspaces, an in-kernel bit-exact copy of the Triton SiLU, and would push the aux-stream shared expert onto the critical path in about 40% of the M=16 layers;
  - effort XL (2-3 weeks). Section 2.
- **Delivered instead: an exact lever on the same bubble, `DSV41_MOE_FC2_HEADPF=silu`** (FC2-head L2 prefetch). The DRAM is idle during the boundary, and FC2's binding resource, its weight stream, does not depend on FC1. Only its 12 KB/expert A operand does. So the SiLU-mul-quant kernel between FC1 and FC2 prefetches the weight tiles FC2's 48 persistent CTAs load first into L2.
  - It is a cache hint only. FC1, FC2, their JIT keys, the SiLU arithmetic and the combine are unchanged, so outputs are bitwise by construction.
  - The patch touches 3 site files, all against the live shas. Every flag-off code path is unchanged.
  - CPU tests: 178/178 PASS after the window-6 bundle rebuild (163 before).
  - **Window-6 bundle rebuild (2026-10-28, review fixes).** P1: a CTA without prefetch work (r0 >= r1) does no grouped_layout load. P2: an expert id must satisfy 0 <= e < 384 before it forms an address. P3: the FC2 layout is a runtime kernel argument Triton does not specialize on (`do_not_specialize`), so the eager warm-up (no FC2C tail yet: layout 2) and the capture of the same size (layout 1) share one compiled kernel and no hpf specialization is first compiled inside a CUDA-graph capture. fp8_utils.py is now append-only (every live line, kernel and Triton cache key unchanged; the public wrapper is re-bound at the end to a dispatcher whose `dsv41_head_pf=None` branch calls the live function with the same arguments). deep_gemm_moe.py builds `_dsv41_fc2_headpf` only for E=384, so the E=128 drafter (which shares the M32 kernel object under `DSV41_DRAFT_MOE_M32=1`) never builds a spec. This lane's hunks are merged with drafter-m32's into `.../w6/bundle/` (one file, one installer `w6-bundle.sh`); `patched/` here is the hpf-only variant.
  - **Verdict: ready-needs-gpu-check, bundle item only.** Review-corrected estimate for H alone: **-0.06..-0.11 ms per 4-session step (G4min_hat +0.14..0.25%)** and **G1 ~0..+0.25%** (at 1 user the SiLU is the dense, DRAM-bound kernel unless `DSV41_MOE_LIVE_ACT_MIN_ROWS=192` (L) moves it to the live path; S = H + L, G1 ~+0.5..0.8%, estimate). The lane's own model said -0.10..-0.17 ms (+0.25..0.4%) and G1 +0.4..0.7%. It is a bundle item: alone it is below the 0.5% G4 rule. The bench gate (section 7) decides.
- Upgrade path (planned, not built): Tier B issues the same prefetch from FC1's TMA producer after its last load. It also covers FC1's drain, for about -0.24..-0.30 ms per 4-session step. It needs a new FC1 host opt-in (native rebuild) and takes 4-6 days (section 3.3).

## 1. Evidence: what the live FC1 -> FC2 boundary costs

**Source.** nsys `--cuda-graph-trace=node` of the window-5 Host step (instance g5host-T2):
- launcher afa97ae6 plus the hosttrace wrapper, pinned av-pin-k5-g3FU, mHC moet/moet on, 2000 MHz;
- rank 0; 60 steps each of G4 #1 (`q0.1`), G4 #2 (`q0.3`) and C1 (`q0.2`);
- sqlite copies in `nsys/`; analysis scripts `tools/moe_gaps2.py`, `tools/moe_timeline.py`, `tools/shared_vs_fc1.py`, outputs `nsys/*.gaps2.txt`, `nsys/q0.1.timeline{12,16}.txt`, `nsys/shared_vs_fc1.txt`.

Target layers only (scatter_1 over 384 experts); us, means over layers:

| load | M (rows) | layers | FC1 | FC1 end -> SiLU | SiLU | SiLU -> FC2 | FC2 | boundary (g1+SiLU+g2) |
|---|---|---|---|---|---|---|---|---|
| G4 | 8 | 200 | 256-263 | 0.57 | 2.6-2.8 | 0.37 | 144-148 | 3.7 |
| G4 | 12 | 2,040 | 309-316 | 0.57 | 3.0 | 0.37 | 175-179 | 3.96 |
| G4 | 16 | 2,560 | 358-362 | 0.57 | 3.4-3.6 | 0.35 | 201-205 | 4.45 |
| C1 | 2 | 1,080 | 89 | 0.56 | 3.74 (dense SiLU, 384 rows) | 0.34 | 47 | 4.64 |
| C1 | 4 | 1,280 | 107 | 0.58 | 5.81 (dense SiLU, 768 rows) | 0.32 | 59 | 6.71 |

- **Weighted boundary: G4 4.21 us per layer = 0.168 ms per 4-session step** (40 routed layers). **C1: 5.72 us per layer = 0.229 ms per step.**
- No other stream runs anything during the boundary: overlap 0.00-0.06 us per layer. The DRAM is idle apart from the SiLU's L2-resident traffic.
- FC2's fixed cost is small. The balanced fits (README window 4, item 02) are FC1 = 6.0 + 6.93·G us and FC2 = 1.4 + 4.66·G us. FC2's ramp is therefore at most 1.4 us, and FC1's drain is part of its 6.0 us intercept.
- So the DRAM-idle window per layer is boundary + FC1 drain + FC2 ramp, about **4.2-8.6 us at G4 (<= 0.34 ms per step) and 5.7-10 us at C1 (<= 0.40 ms)**.
- The README's "10-25 us/layer bubble" (window 3, before LIVE_ACT, mHC and the 'all' trees) is not what the live config has.
- **The FC1 residue tail is a separate cost.** The static 8-m-block swizzle pins the tiles of a partial last group to the CTAs of its m residue (`model/headpf_model.py` replicates the live scheduler). Example: E=34 gives FC1 per-CTA tiles {6: 36, 7: 6, 8: 6} and FC2 max 67 against a mean of 56.7.
  - Item 02 measured the best possible scheduler at 1.020x (M=12) and 1.015x (M=16): 0.29-0.35 ms per step for FC1 plus FC2.
  - **The aux stream already uses FC1's tail.** In 40% of the M=16 layers (638/1600 and 369/960), the shared expert's act_and_mul -> quant -> w2 (25-35 us) waits for the first FC1 CTA to exit and runs in FC1's tail (lower bound 57 us). Its w2 already ends after FC1 in 48/1600 and 29/960 of those layers. At M=12 this happens in only 3/760 and 12/1280 layers.
- **The closest measured data point** is the pd window's FC2 early-B PDL prologue (FC2 weight loads issued during the SiLU, older tree). It gave C1 -0.33% / -0.16%, about -0.1 ms per step or ~2.6 us per layer, and was put on HOLD. That is consistent with this bubble size.

## 2. The lever as specified: per-expert FC1 -> SwiGLU -> FC2 pipelining (design, exactness, why not)

### 2.1 Schedule (precise plan; not built)

**One persistent 48-CTA kernel `dsv41_fc1fc2_v1`** (or FC1 and FC2 as two PDL-chained grids that synchronize through flags).
- **Work list.** Each CTA runs its live FC1 tiles in today's static order (sched_sm120: block b -> CTA b % 48, 8-m-block groups, g1-dw stop), then its FC2 tiles in today's FC2 order.
- **FC1 tile epilogue.** After the 'ds' live-row stores of tile (m, n), run `__threadfence()` and then one `atomicAdd(&fc1_done[m], 1)` (release).
- **SwiGLU per expert.** The CTA whose increment brings `fc1_done[m]` to 9 (all 576 columns of m-block m) runs the SiLU-mul-quant for m's 32 rows. It reads mm1_out through L2 after an acquire, writes FP8 and the packed UE8M0 scales, then does `fence.proxy.async` (FC2 reads A with TMA, which is the async proxy) and `st.release.gpu act_ready[m] = 1`.
- **FC2 producer.** Before the A/SFA TMA loads of m-block m it spins on `ld.acquire.gpu act_ready[m]`. B (weights) and SFB loads are issued unconditionally, so FC2 weight streaming can start in FC1's tail.
- **Deadlock-free by construction:** a CTA finishes all of its FC1 tiles before it reaches any FC2 tile, so every FC1 tile is either running or done before anyone waits on it.

### 2.2 Why it would be bitwise

- **Same tiles, same arithmetic.** Every FC1/FC2 output element is computed by the same MMA chain over the same K order, with the same scales and the same epilogue conversion as today. No split-K and no tile shape change. Which CTA computes a tile never changes its bits; the m32-bw README states the exactness rule.
- **Same FC2 input bytes.** FC2's A for expert m is exactly the SwiGLU output of m's rows. Rows are independent: one 128-column group's amax uses only that row's 128 values. So computing it per expert instead of per grid gives the same bytes, provided the SwiGLU arithmetic is the same instruction sequence. The Triton `_silu_mul_quant_fp8_packed_kernel` lowering (exp2 path, bf16 round trip, `ceil(log2)` scale) would have to be re-implemented in CUDA bit for bit. Qualify it with an exhaustive 65,536-pattern gate table and the recorded mm1_out.
- **Same combine.** ep_gather runs after the kernel, unchanged: fp32 from +0 in slot order, bf16 store, then the separate bf16 shared add.

### 2.3 Why not

- **Small ceiling.** The fused kernel's DRAM-continuity benefit is bounded by section 1's window (<= 0.34 ms per 4-session step) plus at most about half of item 02's tail ceiling (~0.15 ms). After costs, realistic net is 0.2-0.4 ms:
  - the last expert's SwiGLU stays on the critical path (1-2 us per layer);
  - release/acquire polling and atomics.
- **Workspace aliasing.**
  - In `DeepGemmFP4Experts.apply`, `mm1_out` and `mm2_out` are both `_resize_cache(workspace2, ...)`, with 1,152-B and 10,240-B rows.
  - `a1q_perm` and `quant_out` are both views of workspace13 (5,120-B and 384-B rows).
  - Concurrent FC1/FC2 on different experts would overwrite live FC1 inputs and outputs. It needs de-aliased buffers (+~19 MB per rank) and a modular-kernel workspace contract change.
- **Shared-expert displacement.** In ~40% of the M=16 layers the shared expert's second half runs on SMs freed by FC1's tail. A kernel that keeps all 48 SMs busy until its end pushes those 25-35 us after it, onto the critical path. The shared expert would have to move (stream priority, or folding it into the kernel).
- **Race class.** TMA reading generic-proxy writes of other CTAs needs `fence.proxy.async`, which is the failure mode of the spark-ds41f proxy audit. The full co-residency and replay gate would apply.
- **Effort XL, 2-3 weeks:** new fused JIT tree carrying the FC2C, D05 and 'all' features, the SwiGLU port and its exhaustive test, the workspace contract, scheduler changes, race gate. The spark-ds41f critic reached the same conclusion (merged C09: NO-GO at TP8, 0..-0.4 ms at 4 users).

## 3. Exact replacement: FC2-head L2 prefetch (`DSV41_MOE_FC2_HEADPF`)

### 3.1 Mechanism

**What FC2 loads first.** FC2 is the persistent 48-CTA `sm120_fp8_fp4_gemm_1d1d_impl` (g2 FC2C tree), N 5120, BLOCK_N 64, so 80 n-blocks.
- Its first group holds G0 = min(8, M_sum/32) m-blocks. Inside it, block b -> CTA b % 48 with m = b % G0 and n = b // G0.
- So **the first W tiles of every CTA are FC2 output rows [0, R) of the experts in m-blocks [0, G0)**, with R = 64·ceil(48W/G0): 384W at G0=8, 512W at G0=6 (1 token).
- `model/headpf_model.py` checks this on the scheduler replica. The CPU test covers 870 (tokens, live experts, waves) cases.

**Prefetch layout and volume.** Bytes are prefetched through the layout this call's FC2 actually uses:
- FC2C: w2 bytes [0,128) of each padded 192-B row plus the 32-B TAIL row;
- D05/base (1 token, or a tail first seen during capture): the 192-B row;
- plus the 3 int32 UE8M0 scale planes of those rows.
- FC2C at G0=8 is 0.53 MB per wave; the default of 2 waves is 1.06 MB.

**Who issues it (Tier A).** A copy of the SiLU kernel, `_silu_mul_quant_fp8_packed_kernel_hpf`, is used only when the flag is on.
- CTA q of the SiLU grid takes head m-block q % G0 and rows [(q // G0)·rb, +rb), with rb a multiple of 4. The CTAs dispatched first therefore cover wave 0 of every head expert.
- Dead-expert CTAs (~90% of the live-path grid, no rows) issue their share at once. CTAs with rows issue theirs after their unchanged SiLU rows, so no SiLU load or store queues behind a prefetch.
- Instructions:
  - `cp.async.bulk.prefetch.L2.global [addr], 16|32|128|192`, predicated (exact bytes, no masked-lane duplicates);
  - one int32 load of `grouped_layout[32m]` per CTA that has prefetch work (P1: CTAs with r0 >= r1 load nothing), range-checked 0 <= e < 384 (P2);
  - integer index math.

### 3.2 Why outputs stay bitwise

1. **Flag off.**
   - `_silu_mul_quant_fp8_packed_kernel` is source-identical (CPU test 3), so it has the same Triton cache key and binary.
   - The wrapper only gains a `dsv41_head_pf=None` keyword and one `if ...: return` branch. Without the keyword the original launch runs with the original arguments (test 3: wrapper minus the branch == live).
   - deep_gemm_moe.py has additions only, each guarded (test 3). sm121_target_m32.py has an appended block only.
2. **Flag on, FC1 / FC2 / gather.** Nothing in their launch, arguments, JIT keys or binaries changes; the spec builder is host-only. FC2 reads the same bytes: L2 is coherent and a prefetch writes no memory. The combine order is unchanged.
3. **Flag on, the SiLU copy.** The SwiGLU-quant code is a verbatim copy (test 4: hpf body minus the 3 prefetch-site lines == original body). The added code writes no register, shared or global location the SiLU math reads:
   - a predicated bulk L2 prefetch has no destination;
   - the grouped_layout load feeds only address math.
   The GPU test proves the result bitwise on every written output byte and scale word, including an exhaustive 65,536-pattern gate table, NaN/inf/clamp edges and real checkpoint layers under 7 route kinds. The bench repeats the check on the last layer of every recorded-route set. It also checks the PTX for no new FP instruction kinds and no new store or atomic.
4. **No new concurrency.**
   - No new kernel, stream, event or SM-occupancy change. FC2 still starts after the SiLU grid completes (same-stream order); the SiLU kernel can only get longer.
   - Bulk L2 prefetch has no shared-memory destination and no mbarrier, so the proxy-fence race class of the window-5 co-residency gate does not apply.
   - The shared expert's placement is unchanged; a longer SiLU only gives its w2 more time to finish before FC2 launches.

### 3.3 Tier B (planned): issue from FC1's TMA producer

- **What it adds.** The window then also covers FC1's drain and g1: the producer finishes its last FC1 load ~7 stages (~6 us) before the CTA's MMA ends.
- **Kernel** (all-base tree header `sm120_fp8_fp4_gemm_1d1d.cuh` faafb3f6). New template parameter `kFC2HeadPF` (waves; 0 = absent), compile-gated to kG1FC1. After the TMA leader's tile loop, the producer warp replays the FC2 scheduler for FC2-CTA `blockIdx.x` (N 5120, 80 n-blocks) and issues the same bulk prefetches from its 32 lanes.
- **Host** (`_C` of base b1c4eff8 + patch, like g1 fc2c-d05's 33d4eb19). A private opt-in argument `dsv41_fc1_fc2hpf` (a 64-B device descriptor {w2, tail, sf, strides, layout}, built once per layer outside capture like `FC2Compact.descs`). It is passed as `.gmem_workspace`, which is unused by FC1 at split-K 1, under a new JIT name.
- **No-rebuild alternative:** pass the descriptor pointer in a 4-int trailer of grouped_layout, written by ep_scatter_1's CTA 0 from scalar arguments (no extra launch). It edits deep_gemm_utils.py, which the R07/R08 owners also edit.
- **Risk:** producer stalls on TMA backpressure delay CTA exit. In the 40% class-B M=16 layers the shared act/w2 then get SMs later. Mitigation: only CTAs with the maximum tile count issue, budget <= 2 waves.
- **Effort and size:** 4-6 days. Estimate -0.24..-0.30 ms per 4-session step, -0.25..-0.32 ms per 1-user step.

## 4. Expected gain (Tier A, the patch)

**Model.**
- Bulk prefetch issue is backpressured: the live `l2pf_kernel` (5 CTAs, 16 KB per instruction) lasts about as long as its transfer at ~250 GB/s. So the SiLU grid lives about max(T_silu, X/BW_pf).
- FC2's DRAM stream shrinks by X/BW_fc2, and its first loads hit L2 (about 0.5 us off FC2's ramp).
- Gain per layer is about X/BW_fc2 - max(0, X/BW_pf - T_silu) + 0.5 us, which saturates at about T_silu + 1 us once X/BW_pf > T_silu.

**With X = 1.06 MB (2 waves), BW_pf ~200 GB/s (5.3 us), BW_fc2 ~185-215 GB/s:**

| load | M | T_silu (us) | X/BW_fc2 (us) | SiLU extension (us) | gain per layer (us) |
|---|---|---|---|---|---|
| G4 | 12 | 3.0 | 5.7 | 2.3 | ~3.9 |
| G4 | 16 | 3.5 | 5.7 | 1.8 | ~4.4 |
| C1 | 4 | 5.8 | 4.9 | 0 | ~5.4 |
| C1 | 2 | 3.7 | 4.9 | 1.6 | ~3.8 |

**Totals over 40 layers** (allowing ~40% inefficiency):
- **G4 about 4.2 us per layer, -0.10..-0.17 ms per 4-session step** of ~44 ms: +0.25..0.4% G4min_hat. The G4 gap today is ~1.2-1.5% (0.53-0.66 ms), so Tier A is about a quarter of it.
- **C1 about 4.6 us per layer, -0.12..-0.18 ms per 1-user step** of 25.6-29.6 ms: G1 +0.4..0.7%.
- Waves 1/2/3 are all measured; the gain saturates at 2-3.
- **Wave 1 (review correction: NOT a no-extension floor).** 0.53 MB takes ~2.6 us, about as long as the G4 SiLU; but the SiLU grid is CTA-dispatch-bound and an hpf CTA lives ~0.5 us longer (dependent load, address math, issue), so expect +0.5..1 us of SiLU time even at W=1 (review section 4). Gain is then ~X/BW_fc2 + 0.5, about 3.0-3.4 us per layer.
- **Net-loss case.** If the 128-B-row bulk prefetch streams much worse than FC2's own TMA boxes over the same bytes (for example ~100 GB/s), waves 2-3 extend the SiLU by more than FC2 saves. The bench's wave sweep exposes this, and the gate then picks wave 1 or stops.

## 5. Files and live shas

All three base files are sha256-identical on 8/8 (2026-10-28; `live/shas_8nodes.txt`).

| file (site-packages vllm/...) | live sha256 (patch base) | patched sha256 |
|---|---|---|
| model_executor/layers/quantization/utils/fp8_utils.py | b35c056169aae67a2beff93fe5632dc34cd067d1ef7d646ab74040a83f90ad2d | 6b466c31350d769155a9945cdd20a49c45fb2b0b2beb36a9ca175071930b9837 (bundle rebuild; pre-fix cd7317b5) |
| model_executor/layers/fused_moe/experts/deep_gemm_moe.py | 5162dc0fdb59ea83d413411e861623d0c3a68f3fe2e87c2b20bc1ad0d937b93f | 538da3be5ed500b4a3beb0a0cd803d9161913a537e0fb6dae6c8618e59b6911b (hpf-only, E=384 gate; pre-fix e3c697a4; merged with drafter-m32: 34e8b402, `.../w6/bundle/node/deep_gemm_moe.py.all`) |
| model_executor/layers/fused_moe/experts/sm121_target_m32.py | ea4a8814986c3618d26d7acb748561d75e50e839cbea225d9f06154febb290e5 | ae306c6b9bacaea9c5548649539144493954622e2b86f801ec5656e80da240e0 |

Context, unchanged and read only:
- launcher afa97ae6dfc6...;
- deep_gemm_utils.py a4a0486a...;
- trees: all-base manifest 74d0d4f2..., g2 FC2C manifest 5a79c849...;
- headers: FC1 all-base faafb3f6, FC2 g2 1900579d, sched_sm120 3d186b16.

Lane files:
- `patches/make_patch.py` rebuilds `patched/` from `live/` (refuses on any sha mismatch). Also in `patches/`: the three `*.diff` files and `SHAS.txt`.
- `patched/vllm/...`: the 3 patched files (staged as `site/` on the nodes).
- `model/headpf_model.py`: scheduler replica, head geometry and SiLU-grid mirror. `model/sched_model.py`: tile histograms.
- `tests/test_headpf_cpu.py` with `tests/cpu_test_full.txt` (178 PASS, 0 FAIL after the bundle rebuild; the pre-fix generator, outputs and test are kept in `superseded-prefix/`).
- `gpu/test_headpf_gpu.py`: stopped-window bitwise test. `gpu/bench_headpf.py`: microbench.
- `ops/stage.sh`, `ops/prep-node.sh`, `ops/run-gpu-tests.sh`, `ops/install.sh`.
- `tools/*.py`: nsys analysis; outputs in `nsys/`.
- `ab/arms.env`: the A/B arm settings.
- `live/`: base copies plus `readme_snippets/` (the README sections read for this lane).

**Flags.** All default off; any value other than 0/off/silu raises.
- `DSV41_MOE_FC2_HEADPF=silu`;
- `DSV41_MOE_FC2_HEADPF_WAVES` (2; 1..13);
- `DSV41_MOE_FC2_HEADPF_MIN_TOKENS` / `_MAX_TOKENS` (1/96).

**Scope.** Only the target M32 E384 path (`target_m32` set; the spec builder exists only on E=384 instances), the fused UE8M0 SiLU launches, and M <= 96. The drafter (E128) never builds a spec. Correction (review): prefill chunks of up to 96 tokens on the M32 path also run the hpf kernel (exact, but not "untouched"); longer prefill keeps the original kernel.

## 6. CPU tests (done, Mac, pure Python)

`python3 -I tests/test_headpf_cpu.py`: **178 PASS / 0 FAIL** (bundle rebuild; 163 before). New: append-only layout, the dispatcher's pass-through, P1 / P2 / P3 by AST, coverage at `DSV41_MOE_LIVE_ACT_MIN_ROWS=192` grids (t 1..5) and a P2 negative control (corrupt expert id -> no prefetch, all in bounds).

1. **Live copies:** sha == pin, identical on 8/8.
2. **Patch reproducible** from the live copies.
3. **fp8_utils, flag off:**
   - the original kernel source is byte-identical;
   - only the wrapper changed among live defs, and module-level statements are unchanged;
   - the wrapper minus its single guarded branch == live.
4. **hpf kernel:**
   - the 11 new params sit between EXPERT_ALIGNMENT and PDL_TRIGGER;
   - its body minus the 3 prefetch-site lines == the original;
   - the prefetch helpers have no load or store;
   - the head function has no store or atomic and one load.
5. **deep_gemm_moe / sm121:**
   - additions only, all guarded; DeepGemmExperts unchanged;
   - FC2HeadPrefetch with a fake torch: route mirror of `TargetM32Kernel.gemm` (FC2C entry / capture miss / no tail yet / D05 verify / no FC2C), geometry, token gate, skip on unexpected scale layout, flag parsing.
6. **Scheduler replica** reproduces the README facts: G=33 gives a busiest FC2 CTA of 67 tiles; M=6 gives FC1 5 / 4.1.
7. **Coverage, 870 cases** (t 1..96, live experts, waves 1-3): every byte of every CTA's first-W FC2 B tiles is prefetched, in bounds, 16-B aligned, with no duplicate per CTA and 0.0% over-fetch. 1.06 MB per call, max 22-44 prefetches per CTA.

## 7. GPU test plan (stopped window; nothing installed by the tests)

1. **Stage.** Run `ops/stage.sh` from the Mac in a step allowed to write `~/dsv41-stage`; it runs `prep-node.sh build` on 8/8, which checks the node's live shas and runs the CPU test.
2. **Run.** In the stopped window: `ssh spark-one cat ~/dsv41-stage/w6/fc2hpf/ops/run-gpu-tests.sh | bash` (~15-25 min).
   - **Private caches:** `pre` copies the serving caches into `jit-test/`; `post` checks that the serving caches are unchanged.
   - **Bitwise test on 8 ranks in parallel**, 5 real checkpoint layers each = all 40 target layers. Arms live, off, on1, on2, on3; tokens 1,2,4,6,8,12,16,24,48,96. Every comparison is int-view bitwise with NaN-poisoned outputs:
     - (a) SiLU wrapper: live vs off vs on{1,2,3} in layouts 0/1/2, live path and dense path, real and edge-value inputs;
     - (b) exhaustive gate table: 65,536 bf16 patterns x up values, dense and live;
     - (c) whole `DeepGemmFP4Experts.apply` (served COMPACT + LIVE_ACT + G1=all + D05 + FC2C) on routes valid / skewed / spec / half / holes / one / all-dead;
     - (d) CUDA graphs: **200 replays at M = 1, 6, 12, 24, 48** (20 elsewhere), each replay compared;
     - (e) capture-miss: FC2C tail absent during capture -> layout 0 still bitwise;
     - (f) PTX: the original kernel's PTX is identical between live and off; the hpf PTX contains the bulk prefetch, has no new FP instruction kinds and no new st.global/atom/red;
     - (g) engaged: spec calls > 0, skips == 0; no new native JIT entries.
     - PASS = 0 failures on 8/8.
   - **compute-sanitizer** racecheck, synccheck and memcheck once on spark-one.
   - **No co-runner stress.** The lever adds no concurrency (section 3.2). If the reviewer still wants it, run the window-5 stress harness's routed-MoE co-run arm once with the flag on.
   - **Bench** on rank 0 and rank 7, `gpu/bench_headpf.py`:
     - real layers 0/13/26/39, 40-layer graphs (L2-cold weights);
     - recorded C1 windows at M = 1, 2, 4, 6 and c4rec at M = 12, 24 (plus 48);
     - arms interleaved, 25 trials, median, last-layer bitwise check per set.
3. **Bench gate for the serving A/B:**
   - the best on-arm (default on2) is >= 2.0 us/layer faster than off at c4rec M=12 and at M=4;
   - no M is slower than off by > 0.5 us/layer;
   - A/A |live - off| <= 1%.
   - If it fails, stop and record the numbers (expected failure mode: SiLU extension >= the FC2 saving, i.e. no idle DRAM to fill).
   - Optional `--nsys` pass: per-layer SiLU and FC2 durations through `tools/moe_gaps2.py`. The SiLU may get longer; FC2 must get shorter by more.

## 8. Pinned A/B plan (short suite; only after the gate)

- **Install** in the stopped window: `ssh spark-one cat ~/dsv41-stage/w6/fc2hpf/ops/install.sh | bash -s install`. It refuses while serving, backs up to `.bak-w6-fc2hpf`, and verifies the patched shas on 8/8. The launcher is NOT changed.
- **A/A first:** one launch with the flag unset. exact.py a8c3394af9d2 with the pin's counters 909/467/2335/1373 (per-position 348/233/171/90/67) and "pin hit".
- **Arms:** supervisor relaunches on launcher afa97ae6 with `DSV41_AV_COST_JSON=~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json` (eeac4f97) and `DSV41_GPU_CLOCK_MHZ=2000`:
  - C = flag unset;
  - H = `DSV41_MOE_FC2_HEADPF=silu`, `DSV41_MOE_FC2_HEADPF_WAVES` = the gate's best (default 2).
  - Order C1 H1 C2 H2 (ABAB, >= 2 pairs); a 3rd pair if a pair disagrees.
- **Per-instance suite** (window-5 `inst2.sh` / `vphase.sh` pattern, telemetry off, SM-clock read on 8/8 before each phase, success-counter and 2 s sampler foreign-traffic guard; an overlapped phase is NOT PROVEN and is rerun):
  1. exact.py as the first request: a8 plus pinned counters plus pin hit;
  2. matched C1 (label pdab, coding 1x5 + prose 1x5), ms/step and AL;
  3. G1 x3 (1024 tokens, code/prose/math): digests bc1e8386878bd2b7 (3 reps) and 9a18f802d930f6e7 (first 2);
  4. G4 warm-up + 3 timed runs: per-session mean, prose MIN, G4min_hat, 4-session step ms;
  5. bench conc 8 and 16 (512 tokens);
  6. exact.py #2 after all load: a8 plus the same counters.
- **Exactness** (every H instance): a8 plus counters plus both G1 digests identical to C; any difference means STOP and roll back. The lever cannot legitimately change a bit.
- **Decision** (window rule): exact, and G1 or G4min_hat >= +0.5% in every pinned pair (>= 2 pairs), nothing worse beyond noise. Expected outcome: G1 may clear the rule alone, G4 most likely not.
  - Pass: promote `export DSV41_MOE_FC2_HEADPF=${DSV41_MOE_FC2_HEADPF:-silu}` in the launcher (manifest update).
  - Otherwise: keep it installed and inert, as a bundle candidate with R08 / the L2 plan / Tier B.
- **Rollback:**
  - one launch: relaunch without the flag;
  - files: `install.sh uninstall`, service stopped, restores the base shas from the backups on 8/8.

## 9. Risks

- **Gain below the estimate.** Bulk-prefetch issue or CTA-exit stalls may make the SiLU kernel last as long as the full prefetch. The gain then shrinks toward T_silu and the FC2 strided-row efficiency. The bench measures it; waves 1/2/3 bracket it.
- **Triton specifics:**
  - `inline_asm_elementwise` with an unused result and is_pure=False must not be dropped. The PTX check (f) catches this; fallback is a data-dependent keep-alive store under an impossible runtime condition.
  - `cp.async.bulk.prefetch.L2.global` with immediate sizes must assemble for sm_121a. The live l2pf.cu uses the same instruction with a register size.
  - The extra int64 arguments add specializations: a handful of compiles at warm-up, none during capture after warm-up.
- **Scale layout assumption** (int32 [384, 5120, >=3], MN-major). Anything else is detected per call and the spec is skipped (unchanged path, `skips` counter; GPU check (g) requires skips == 0).
- **L2 pressure:** 1.06 MB per layer, consumed within ~10 us, against 24 MiB L2 and the 6 MB l2pf windows that are issued after the gather. Low risk.
- **Interactions with other w6 lanes:**
  - fc2c-contig changes the FC2 weight layout and needs one more PF_LAYOUT mode: one contiguous 10 KB range per tile, simpler;
  - pdl-glue (R08) does not list this SiLU kernel. If it adds `launch_pdl`/`gdc_wait` there, apply it to both kernel copies;
  - qb-l2pf: no overlap in time.
- **Host overhead:** one small Python spec per routed layer per eager or captured call; graph replays are unaffected.

## 10. Related exact items found on the way (not in this patch)

- **FC1-head prefetch during the router/permute glue.** From the end of topk to FC1's start there are ~20-24 us of small glue kernels, with DRAM partly idle (shared w13 streaming at ~60-100 GB/s). FC1's first wave is m-blocks 0..7 x n-blocks 0..5.
  - It has to be issued off the critical path. In-stream issue extends the glue; side-stream issue risks displacing static-schedule FC1 CTAs.
  - Potentially 5-10 us per layer. It needs its own analysis and has overlap with R07 / mfront.
- **`DSV41_MOE_LIVE_ACT_MIN_ROWS=192` (env only).** It moves C1's M=2/4 SiLU from the dense path (3.7 / 5.8 us) to the live path (~2.5-3 us). Exact: padding rows are never consumed.
  - It shortens the same boundary that Tier A fills, so the two do not add. Choose one by measurement.
