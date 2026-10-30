# r4 b12x-moe: the routed MoE on b12x behind `DSV41_MOE_B12X=1` (default OFF)

## What it is
- vLLM 0.32 already ships a b12x MoE backend: `B12xExperts` in `fused_moe/b12x.py`, selected with `--moe-backend b12x`.
- What our venv lacks is the `b12x` package itself. Round 3 (the sonar step) built an overlay from flashinfer-python 0.7.2's `experimental/b12x`, which is b12x at 6627d342 plus a `b12x` shim. Its import test passed on spark-three.
- This port reuses that overlay and adds three flag-gated changes:
  1. **`oracle/mxfp4.py`**: with `DSV41_MOE_B12X=1`, `select_deepseek_v4_mxfp4_moe_backend` forces `b12x` for the 384-expert target MoE.
     - `DSV41_MOE_B12X_SCOPE=all` also moves the 128-expert DSpark drafter.
     - No launcher or CLI change is needed.
  2. **`utils/b12x.py`**: with `DSV41_MOE_B12X=1`, it puts `DSV41_B12X_OVERLAY` (default `~/dsv41-stage/r4/b12x-moe/overlay`) on `sys.path`, so `b12x` imports.
     - Without the flag, `b12x` stays absent, exactly as today.
  3. **`fused_moe/b12x.py`** (`B12xExperts`) gets three changes:
     - **Width padding, `DSV41_MOE_B12X_PAD=auto`.** At TP8 the per-rank width is 288, and 288 % 128 = 32.
       - Stock b12x then takes its legacy N128 repack: zero tails to 384, which reads 3.13 MB per expert slice. That is +20% bytes against our current compact FC1 path.
       - `auto` zero-pads each half 288 → 320, so 320 % 128 = 64. That selects b12x's exact **compact-N64 W4A8** path.
       - Bytes per slice: FC1 640×2560 + FC2 5120×160 + scales = 2,611,200 B. That is exactly the same as our current DeepGEMM compact path (FC1 576 rows + FC2 K 384).
       - Pad weights are FP4 zero and pad scales are UE8M0 127 (1.0), so the output is mathematically unchanged; the CPU test checks this.
       - After the padded copy is prepared, the checkpoint-width tensors are released, so memory does not double.
     - **Small-M plan pin, `DSV41_MOE_B12X_SMALL_PLAN=triton:48:16`, at ≤ `DSV41_MOE_B12X_SMALL_MAX=8` rows.** Ported from knapcio's adapter.
       - b12x's heuristic picks `micro` here, which streams one expert slice per (token, top-k slot).
       - The pin uses `dynamic`/grouped/M16 with 48 clusters (the GB10 SM count). It streams each distinct expert once.
       - knapcio measured 523 → 349 µs per call at M=6, N=576.
     - **`DSV41_MOE_B12X_DETERMINISTIC=1` (default when on).** b12x's fixed-order top-k reduction replaces atomics, so greedy output is bit-stable. Under it the pinned planner becomes `internal`.
     - It also pre-creates every serving plan in `warmup_launches`, so no plan is created during CUDA graph capture.

## Why the gain is uncertain
**Bytes (the limiter).**
- The r2 profile shows the routed MoE is bandwidth-bound: 9.5 / 29.5 / 64.5 ms at 1 / 4 / 16 users.
- b12x compact-N64 reads the same bytes per expert as our compact path, so there is no byte win.
- Of the 288→384 padding, 96 FC2-K columns are still read. b12x has no path that avoids reading the padding for N % 128 = 32.

**What b12x can win:**
- About 6 launches per layer become 1 fused kernel (input MXFP8 quant, routing, FC1, SwiGLU, FC2 and weighted reduce). Ours is quant → permute → FC1 → act-quant → FC2 → unpermute/reduce.
- That cuts in-graph launch tails and gaps by roughly 5–25 µs per layer, about 0.2–1.0 ms per step.
- The `dynamic` scheduler streams each distinct expert once, with no M32 row padding.

**Against it:**
- knapcio measured b12x at about 214 GB/s for 576-wide slices. Our slices are narrower (320).
- DeepGEMM M32 already runs near roofline: about 238 GB/s implied in the r2 profile.

**Grounded estimate:**
- 1 user: −0.0 to −1.0 ms per step out of 33.5 ms, which is +0 to +3%.
- 16 users: ±5% of the 64.5 ms MoE. Unknown; the bench decides.
- Knock-on upside: `DSV41_MOE_COMPACT_FC2` faulted in r3 (DeepGEMM compact-B illegal instruction). This path is a working alternative.

**Accuracy:**
- b12x quantizes activations as MXFP8 k32. DeepGEMM uses FP8 k128. Both use UE8M0 scales.
- So b12x should be equal or slightly *more* accurate. It is **not** bit-exact against the current path.
- The gate is rel-L2 against an fp32 reference.

## Files touched (conflict check)
| file | sha before | sha after |
|---|---|---|
| `vllm/model_executor/layers/fused_moe/b12x.py` | f60c0622 | 4d75406b |
| `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` | 10f0ebef | 0d25ce80 |
| `vllm/utils/b12x.py` | 99bc2e6d | 3ac51888 |

- None of these is touched by the r1–r3 patches: `deep_gemm_moe.py`, `quantization/mxfp4.py` and `sm121_*` are NOT modified.
- Other r4 ideas that edit `oracle/mxfp4.py` must merge with this one. The hunk is small: one helper function and three lines in `select_deepseek_v4_mxfp4_moe_backend`.

## Steps for the experiment agent
1. **Stage.** Already done by the porting agent, on all 8 ranks, nothing live. Re-run it with `bash r4/b12x-moe/stage.sh`.
2. **Offline unit test and microbench** (one host, service stopped on it, e.g. spark-four):
   ```
   ssh spark-four 'bash ~/dsv41-stage/r4/b12x-moe/run_bench.sh'          # full: M=1,6,12,24,48,96, 4 layers
   ssh spark-four 'bash ~/dsv41-stage/r4/b12x-moe/run_bench.sh --quick'  # M=1,6,48, 1 layer
   ```
   - It runs the CPU pad test, then `test/bench_b12x_moe.py`.
   - Variants: `dg` (live path), `b12x_n128`, `b12x_n64_heur`, `b12x_n64_pin`, `b12x_n64_pin_det`.
   - It reports correctness against the fp32 reference and against dg, repeatability, and CUDA-graph µs per layer (chained over 4 layers so L2 is cold, plus hot).
   - It also reports distinct experts, GB/s and estimated ms per step (×40 layers).
   - Results go to `~/dsv41-stage/r4/b12x-moe/results/`.
   - The first b12x run JIT-compiles CuTe DSL kernels, which takes minutes. The cache is kept in `b12x-compile-cache/`.
3. **Promote only if** `b12x_n64_pin_det` chain µs per layer beats `dg` at M=6 and M=12 (1 user verify) and is not worse at 48 and 96.
4. **Install** (all 8; preflight aborts on any conflicting file): `bash r4/b12x-moe/install.sh`
5. **A/B arm** (the env reaches every rank through the existing tool):
   ```
   bash $S/conc/launch-conc.sh DSV41_MOE_B12X=1 B12X_COMPILE_CACHE_DIR=/home/jeweled/dsv41-stage/r4/b12x-moe/b12x-compile-cache
   ```
   - The control arm is the same command without `DSV41_MOE_B12X`. Once installed and unset, the patched files behave identically to the originals.
   - Optional knobs: `DSV41_MOE_B12X_SMALL_PLAN=off`, `DSV41_MOE_B12X_DETERMINISTIC=0`, `DSV41_MOE_B12X_PAD=0`.
   - `DSV41_MOE_COMPACT` / `DSV41_SM121_TARGET_M32` can stay set: they only apply to the DeepGEMM backend.
   - Check the logs for `DSV41_MOE_B12X: E=384 routed MoE -> b12x`, `routed experts padded 288 -> 320` and `Warmed up ... b12x MoE`.
   - Then run the usual single, conc4/8/16 and GSM8K-50 checks.
6. **Rollback.** `bash r4/b12x-moe/uninstall.sh`, which sha-verifies that 24/24 files are original, then relaunch without the flag.

## Already verified (CPU only, spark-four, nice 19)
- **`test/test_pad_cpu.py`: PASS.**
  - Width selection is correct.
  - The pad layout is correct.
  - Padded vs unpadded expert output on a dequantised fp64 SwiGLU-clamp MoE has max diff 0.
- **`test/probe_plan_cpu.py`:** the overlay's legacy `plan_weights` accepts E=384, K=5120 at N=288 and at N=320.
  - For both it reports W4A8 `qmma_repacked`, `reuses_source_storage=True` (repacked in place into our padded copy) and `discards_source_parameters=True`.
  - The pinned `MoeDecodeConfig` constructs.
  - Log: `probe-cpu.log`.
- **Patched files and the bench:** `py_compile` OK under the venv's python 3.12.
- **Not verified:** no GPU code has run. Kernel JIT, numerics and timing are all pending step 2.

## Risks
- **Version skew.** The overlay's b12x pins `nvidia-cutlass-dsl==4.7.1`; the venv has 4.8.0.
  - Imports work (r3 checked), but kernel JIT compiles are untested on 4.8.0.
  - Step 2 is the test. If it fails, the fix is a side venv or a `--target` install of cutlass-dsl 4.7.1 into the overlay dir (about 200 MB; spark-one/two have 13–16 GB free).
- **Untested API path.** vLLM 0.32's `B12xExperts` uses b12x's legacy flat API (`plan_weights(quant_modes=...)`, `prepare_weights(w1_fp4=...)`, `fused_moe.plan(Caps(...))`).
  - The overlay keeps it through `_compat.py`.
  - The compact-N64 path has only been exercised by knapcio through the canonical API, at N=576.
- **N=320 is untested.**
  - N=320 is the first compact-N64 width with n % 128 = 64 that has **two** 128-tiles plus a 64-tail. knapcio's 576 has four plus a tail.
  - The `n64_tail` code exists for exactly this case, but nobody has validated N=320.
- **Memory.** Load-time transient is about +1 GB per layer while one layer is padded (it is released right after). Steady state is equal to today.
- **Engine-level gaps not exercised by the bench.** The bench calls the same `FusedMoEKernel` the engine uses, but not the engine's CUDA-graph capture or warmup path.
