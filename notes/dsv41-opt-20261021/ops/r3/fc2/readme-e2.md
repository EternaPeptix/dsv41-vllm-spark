
### Round 3 / e2: FC2 compaction (DSV41_MOE_COMPACT_FC2): NOT promoted, kernel faults, nothing installed

**Goal:** unpad routed-MoE FC2 (288->384) with a separately built DeepGEMM (`~/dsv41-stage/fc2c/native-fc2c-v1`, _C e7088a23, manifest 072fdc49) whose compact-B kernel remaps the B load. Expected about +2-5%, plus about 3.8 GiB per rank freed.

**Offline test** (`run_offline_test.sh`, spark-one, service stopped, 2026-10-22 ~04:15 UTC):
- **kw=288:** the first case (M=1, 32-row path) faults when `sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d_compact_b` launches with `CUDA_ERROR_ILLEGAL_INSTRUCTION (715)` (cuLaunchKernelEx, deep_jit driver.hpp:33). Confirmed with CUDA_LAUNCH_BLOCKING=1.
- **kw=320** (separate process): faults the same way on the same first case.
- **Healthy in the same process:** the padded kernel from the new native is bitwise identical to the live M32 library (native_padded_eq_live=true), and the padded baseline matches the fp32 reference (rel 1.66e-3). The sanitizer fixed 1,473,703 NaN dead scales.
- **Not the cause:** plain torch reductions work on this GPU.
- **Where the fault is:** in the compact-B patch codegen. Both variants JIT-compiled and are cached in `~/dsv41-stage/fc2c/fc2c-jit-cache-v1/cache/*compact_b.{45960e62,cf999d97}` with 27-line kernel.cu each. Untested idea: the TMA box/tensor-map or a PTX variant that sm_121 rejects.
- No 128-row case and no timing was reached.

**Decision:** stopped at offline step 5. `install.sh` was never run (site-packages unchanged) and there was no A/B.

**State after:**
- Service relaunched with `dsv41-launch-all.sh` (health 200); GSM8K-50 at conc8 scored 50/50 (221.8 tok/s).
- On all 8 ranks: launcher abd49904, deep_gemm_moe.py 45418259, mxfp4.py 67339029, sm121_fc2_compact.py absent.

**Rollback:** none needed. Staged files stay in `~/dsv41-stage/fc2c` for a kernel fix.

Raw results: `~/dsv41-opt-20261021/r3-e2/` (offline-test.log, final-gsm50.*).
