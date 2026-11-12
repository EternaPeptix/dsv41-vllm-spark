# r2-skinny: skinny_mx v2 for the qkv_a shard (G10) + fused qkv_a MXFP8 quant (G09 part D2)

Climb round 2, kernel levers. Default off and inert when off. Bit-exact when on, by op-order construction (see
`EXACTNESS.md`). The CPU emulation proves the index mapping.

**Status (2026-11-08 15:20Z):** compiled on spark-eight (15:15Z, lock free). The kit ships
`prebuilt/skinny_mx_v2_ext.so`, sha256 `b3af28da9120e5efe1493814768fbf42a5d7194e343a5b9369e3214d9f1f9a92`
(`build.log`: ptxas -v, 0 spill stores/loads and 0 stack for every kernel). A CPU-only import of that .so with the serving
venv works. **The device bitwise test has not run yet.** It needs a stopped node, which only the lead can provide
in a window (brief rule 3; hard rule 1). It is the gate before any window arm:
`bash scripts/run_device_test.sh spark-eight 7` must exit 0.

| lever | flag | what changes on the GPU |
|---|---|---|
| G10 (dense:D3) | `DSV41_SKINNY_V2=1` | `skinny_mx_kernel<M,4,8>` (live .so) → `skinny_v2_kernel<M,5,false>` (new .so), same operands, same per-output fp32 op sequence |
| G10, PDL arm | `DSV41_SKINNY_V2=pdl` | same kernel, launched with programmatic stream serialization: its weight loads start before the producer kernel (the quant) finishes |
| G09 part D2 (dense:D2) | `DSV41_SKINNY_QFUSE=1` (needs V2) | the qkv_a-input `MXFP8QuantizeSwizzled` launch (grid 128, block 640; 43 per step) disappears; `skinny_v2_kernel<M,5,true>` quantizes bf16 x itself with FlashInfer's PTX sequence |
| woar epilogue (D2 ii) | — | **skipped**, see below |

## Files

- `kernel/skinny_mx_v2.cu`, `kernel/build.py`: the new extension, which builds `skinny_mx_v2_ext.so`. It is
  installed to `~/dsv41-r2/skinny_mx_v2/` on each node. The live `~/dsv41-r2/skinny_mx/` is not touched.
- `new/dsv41_r2_skinny.py`: loader, eager bitwise self-check, and the serving entry points. It is installed next to
  `dsv41_r3_dense.py`.
- `install.py`: 4 anchor edits to `models/deepseek_v41/nvidia/dsv41_r3_dense.py`, 21 inserted lines, each tagged
  `# r2-skinny`. Also `.bak-r2-skinny`, sub-commands `install | uninstall | status | dry`, and a refusal unless the
  base sha256 is recorded in `KIT.json`.
- `prebuilt/skinny_mx_v2_ext.so` (the binary `install.py` ships; sha256 in `KIT.json`), `build.log` (its compile
  log with ptxas -v), `scripts/compile_spark8.sh` + `scripts/drive_compile.sh` (how it was built, lock-guarded),
  `scripts/run_device_test.sh` (the lead's pre-window device test, refuses on a node with a GPU process).
- `EXACTNESS.md` (op-order map), `device_test.py` (bitwise + timing on a stopped node), `tests/` (CPU tests),
  `TEST-RESULTS.txt`, `base/` (live copies), `ref/` (FlashInfer quant source, W-E2 fold for reference).

## Mechanism

**Today** (`skinny_mx.cu`, trace numbers from audit `verify-G10-combined`):
- 48 persistent blocks. Each walks 4-5 rows one after another.
- Per row, every lane issues one 16-byte weight load and waits for it. Lanes 0-7 then issue a second load and wait
  again (the 40-vector K slice never enters the UNROLL loop).
- Then a 5-step shuffle tree per m and two `__syncthreads`.
- Before any of this, a block-wide prologue dequantizes all M×5120 activations, followed by another
  `__syncthreads`.
- Result: about 10 serial memory latencies plus about 10 barriers per call. Calls take 9.7 / 9.8 / 12.6 / 16.3 /
  20.3 µs at M = 1 / 2 / 4 / 5 / 6 for 1.15 MB of mostly L2-resident weights (5.2 µs DRAM floor).

**v2** keeps every output's arithmetic and changes the schedule:
1. Each lane issues all of its weight loads for all of its rows (5 first vectors and 2 second-vector loads) before
   anything waits, so there is one memory latency instead of about 10.
2. Each warp builds only its own K slice of the activations, behind `__syncwarp`. There is no block-wide prologue
   barrier, and all prologue loads are issued up front.
3. The fp8→float weight conversion runs once per vector instead of M times.
4. Each row's 8 second-vector dots (v1: lanes 0-7 only, a second serial step) are computed by all 32 lanes, 4 rows
   per pass. Each dot is handed to its owning lane with one `__shfl_sync`, and the owning lane applies v1's fmaf.
   That is 7 vector steps per lane instead of 10.
5. All rows' part sums are stored, then one `__syncthreads`, then all (row, m) combines run in parallel.
6. `pdl`: the weight loads sit before `griddepcontrol.wait`, so with PDL they overlap the quant kernel's tail and
   the launch gap. The quant already triggers `launch_dependents`.

**QFUSE**: the warp quantizes its own 640-element slice of each row from bf16. It copies the FlashInfer CuTe-DSL
4-threads-per-block sequence as PTX: bf16x2 abs/max tree, xor-shuffle max over 4 lanes, ue8m0 bit math,
x·inv_scale, clamp, `cvt.rn.satfinite.e4m3x2`. It then dequantizes exactly as v1's A8 prologue does. The separate
quant kernel and its output buffers disappear.

## Why flag-off is identical

- Every inserted branch in `dsv41_r3_dense.py` is behind the module-level bools `R2_V2` / `R2_QFUSE`. Both are
  read once at import and are False when the variables are unset or set to `0`/`off`.
- With both False, the executed code is byte-for-byte the old code. `tests/test_install.py` checks:
  - the AST of the patched file minus tagged lines == the AST of the base;
  - every outermost tagged `if` tests `R2_V2` / `R2_QFUSE`;
  - `dsv41_r2_skinny` is imported only inside `if R2_V2:`, and the new .so is never loaded.
- `tests/test_routing.py` runs the patched module against stub kernels:
  - with flags unset: v1 + separate quant on every call, no `_r2` attribute, r2 module never imported;
  - `QFUSE=1` without `V2`: identical to off, plus one warning.
- Invalid values (`DSV41_SKINNY_V2=2`) raise at import. They do not fall back silently: a fallback that disabled
  the shard would switch to the stock replicated GEMM, which is NOT bitwise equal to the skinny path.
- If the r2 module fails to import or build, the live v1 shard path is kept (tested).

## Why flag-on is bit-exact

Per output element (row, m), v2 performs exactly v1's operations in v1's order:
- same activation operand (v1 dequant formula of the same bytes);
- same `dot16` fmaf chain;
- lane accumulator `fmaf(s1, d1, fmaf(s0, d0, 0))` on lanes < 8 and `fmaf(s0, d0, 0)` on the others;
- the same xor tree read at lane 0;
- `t = 0; t += part[q]` for q = 0..7, then bf16 RN.

Only where and when they run changes: which lane computes a dot16, which thread combines, and when loads are
issued. Data moves between lanes only through `__shfl_sync`, which is exact.

QFUSE produces the same (e4m3, ue8m0) bytes as the live quant, by the same PTX instructions. The one
source-ambiguous instruction is the FTZ mode of `amax*(1/448)`, which differs only for amax ≈ 2^-118. It is pinned
at runtime by comparison against the live kernel and verified exhaustively by `device_test.py`.

Enforced at serving time as well as by the tests: at the first eager build of each layer's shard,
`dsv41_r2_skinny.SkinnyV2._self_check` runs:
- every M in 1..`DSV41_QKV_A_SHARD_MAX_M` (8);
- on the layer's **real shard weights**;
- with 3 activation sets (post-norm-like, hostile, random);
- through live quant + live v1, and through v2 A8 and v2 fused, pdl off and on.

An M is served by v2 only if every output is bitwise equal. Otherwise it stays on v1 and the log says
`kept live`. Full table: `EXACTNESS.md`.

## Race analysis

- smem activations are warp-private, written and read by the same warp behind `__syncwarp`.
- `part_sums` is written by lane 0 of each warp and read after `__syncthreads`.
- There are no atomics and no global scratch. y is a fresh `torch.empty` per call, as in v1.
- PDL: only static weight tensors are read before `griddepcontrol.wait`. x/xq/xs reads and the y write come after
  it.
- Graph safety:
  - streams and events: none new;
  - allocations: the same single `torch.empty` as v1;
  - smem attributes: set eagerly by `prepare()` at load, never during capture;
  - self-check: eager only (it asserts that no capture is in progress), on the first eager call, which the shard
    build already requires.
- The L2 prefetch recorder (`dsv41_l2_prefetch._wrap_r3_dense`) still records `st.s_rm` and `st.w`. v2 reads the
  same tensors (a `uint8` view of the same storage), so the prefetch plan is unchanged.

## Expected saving (trace evidence: audit `verify-G10-combined`, `verify-G09-combined`, `est-critpath`)

Critical path:
- Skinny is followed after about 0.6 µs by the all-gather `ar_kernel_pl`, and then by `_q_kv_norm_quant`.
- The side branches (mhc_pre_big_fuse, compressor, weights_proj) finish 8-24 µs (p5) before the all-gather ends.
- So every µs cut from skinny, and every µs of quant removed, comes off the step. That holds on 39 of the 43 calls
  per step. The other 4 are stalled about 41 µs behind the 8x8x16 side GEMM, and no credit is given for them.

| lever | per-call saving assumed (low / central / high) | ms per step (code / prose / maths) | G1 |
|---|---|---|---|
| G10 v2 | M1-2: 1 / 3 / 5 µs; M4: 2 / 4.5 / 7.5; M5 (drafter): 3 / 6 / 10; M6: 3.5 / 7 / 12 | central 0.22 / 0.15 / 0.23 | +0.2 / **+0.7** / +1.3 % (my model gives 0.41 / 0.88 / 1.49 %, discounted to the skeptic's range) |
| G10 pdl increment | 0.3 / 0.7 / 1.1 µs × 39 | 0.012 / 0.027 / 0.043 | +0.05 / **+0.12** / +0.18 % |
| G09 D2 (QFUSE) | quant 2.46-2.97 µs + 0.8 µs gap, minus 0.5-1.5 µs of added prologue: 1.8 / 2.6 / 3.2 µs × 39 | 0.07 / 0.10 / 0.125 | +0.30 / **+0.44** / +0.54 % |
| kit, all on | | | about +0.5 / **+1.2** / +1.9 % |

The model for v2, M = 6, central:
- launch and ramp 1.5 µs;
- weight latency of about 1 µs, overlapped with the prologue (about 0.8 µs);
- 7 vector steps of compute, about 1.1 µs;
- shuffles about 0.6 µs;
- combine about 0.3 µs.

That is about 5 µs, against 16.1 µs in the bench and 20.3 µs in the trace.

Not counted: a lighter v2 might start earlier on the 4 stalled calls per step. It uses the same smem as v1 but has
no block-wide prologue barrier.

## How to read it in an nsys trace (`anat3.py` buckets: rename the skinny regex to `skinny_(mx|v2)_kernel`)

- **G10**: `skinny_mx_kernel<…>` (43 per step, grid 48, block 256) is replaced by `skinny_v2_kernel<M,5,false>`,
  same grid and block.
  - Expected durations at M2 / M4 / M5 / M6: about 4-6 / 5-8 / 6-9 / 7-10 µs, against 9.8 / 12.6 / 16.3 / 20.3.
  - The following `ar_kernel_pl` starts correspondingly earlier.
  - Registers / smem: read `registersPerThread` and `dynamicSharedMemory` (M×10240 B, as v1).
- **pdl**: the `skinny_v2_kernel` start moves to before the end of the preceding `MXFP8QuantizeSwizzled` (grid
  128). The quant→skinny gap (median 0.55-0.86 µs today) goes to ≤ 0 µs.
- **QFUSE**: the `kernel_cutlass_…MXFP8QuantizeSwizzledKernel…` launches with grid 128 / block 640 drop from 43 per
  step to 0. The grid-16 / block-1024 ones (wo_b input) remain.
  - `skinny_v2_kernel<M,5,true>` follows `_proj_rows` (38 sites) or `mhc_pre_big_fuse_with_norm` (5 sites)
    directly.
  - It is about 0.3-1 µs longer than the A8 variant (prologue).
- Per-step check: the sum over skinny kernels plus grid-128 quants per step drops by the table's ms. Step time
  drops by about the same, because the chain is on the critical path. Check that `_q_kv_norm_quant` starts earlier
  relative to the layer start.

## Window arms (env per arm; every rank the same)

| arm | env |
|---|---|
| control | (none: `DSV41_SKINNY_V2` unset) |
| T-v2 | `DSV41_SKINNY_V2=1` |
| T-v2pdl | `DSV41_SKINNY_V2=pdl` |
| T-qfuse | `DSV41_SKINNY_V2=pdl DSV41_SKINNY_QFUSE=1` (if pdl is not promoted: `DSV41_SKINNY_V2=1 DSV41_SKINNY_QFUSE=1`) |

Promotion protocol as in the round-1 brief. QFUSE is measured against a control with the same V2 value.

**Before the first arm**:
- On a stopped node: `bash scripts/run_device_test.sh spark-eight 7` (from the Mac kit folder). It must exit 0,
  with the `prebuilt/` .so (the test checks its sha256 against `KIT.json`) and real shards of layers 0/13/39.
- Re-run `python3 install.py dry` on a node right before installing (production moved to launcher `08a249da`;
  the base file `1f1c0456…` was unchanged on 8/8 at 15:18Z).
- After the boot, check rank logs for `DSV41_SKINNY_V2=… active: N=224 K=5120, a8 M=[1..8], qfuse M=[1..8]
  (ftz=X)` on all 8 ranks. Absence of `kept live` / `disabled` warnings is required.
- Then exact.py == ec2f4437f8e1ed09 and G1 digest == 2f7fdccbd388225a, as the brief requires.

## Build / install

1. **Done** (2026-11-08 15:15Z, spark-eight, CLUSTER_LOCK free): `scripts/drive_compile.sh` (from the Mac) checks
   the lock, copies `kernel/{skinny_mx_v2.cu,build.py}` to `~/climb-scratch/r2-skinny/kernel`, runs
   `scripts/compile_spark8.sh` there (`PATH=$venv/bin:$PATH MAX_JOBS=1 nice -n 19 $venv/bin/python build.py`; ninja
   comes from the venv), polls the lock every 10 s and kills the build if a lock appears. It then fetches
   `build.log` and the `.so` and removes the remote build dir. `build.py` sets `CUDA_VISIBLE_DEVICES=""`, as the
   live build.py does. Toolchain: nvcc 13.0.88, torch 2.13.0+cu130, `-gencode arch=compute_121,code=sm_121
   -O3 -lineinfo -Xptxas=-v -std=c++20`, no fast-math. Result: `prebuilt/skinny_mx_v2_ext.so`, sha256 `b3af28da…`.
2. Device test (lead, stopped node, before any arm): `bash scripts/run_device_test.sh spark-eight 7`. It copies
   `device_test.py`, `KIT.json`, `new/` and `prebuilt/` to `~/climb-scratch/r2-skinny` and refuses if any vLLM or
   GPU compute process is running. It then runs
   `device_test.py --model ~/models/dsv41/official-fp8 --layers 0,13,39 --rank 7 --json …` and fetches the log and
   json into `tests/device-<node>-<UTC>.{log,json}`. Expect about 3-5 min. Exit 0 means every case was bit-identical
   and the .so was the KIT.json binary.
3. In the window, on all 8 nodes: `python3 install.py install`. This patches the file, copies the module, and
   copies `prebuilt/skinny_mx_v2_ext.so` after checking its sha256. Install **refuses**:
   - a null or mismatching KIT.json sha256;
   - a destination .so that is not that binary;
   - a stale patch from an older kit revision. An already-patched file must equal `patched(unpatched(file))`.

   `install --build` (an untested per-node build) remains only as an escape hatch, and `status` then reports the
   .so as `MISMATCH`. Check with `python3 install.py status`, which prints .so `present (== KIT.json)` and
   `patch matches this kit revision`. Undo with `python3 install.py uninstall`.

## Risks

- **Gain**: the per-call savings are modelled, not measured, and the device test's timing section sizes them
  first. Register pressure (`build.log`, measured): 0 spills everywhere. Registers per thread for
  `skinny_v2_kernel<M,5,·>` with M = 1..6:
  - A8: 95 / 117 / 126 / 151 / 153 / 156
  - QF: 95 / 119 / 127 / 128 / 136 / 143
  - live v1 `skinny_mx_kernel<M,4,8>`: 64-96

  At grid 48 on 48 SMs this is still one block per SM, as in v1. M=8 A8 (MAXR 5) reaches 255 registers, still with 0
  spills. M=7-8 are within `DSV41_QKV_A_SHARD_MAX_M=8` and are self-checked like the other M. Verify widths in
  the traces are 1..6.
- **PDL residency (performance only)**: with `pdl`, a parked v2 block holds about 256×(95-156) registers, which is
  24-40k of the SM's 64k, plus up to 80 KB of smem. It waits at `griddepcontrol.wait` from the quant's
  end-of-kernel trigger onward and can delay aux-stream compressor/weights_proj blocks on that SM. The T-v2pdl arm
  measures it against T-v2.
- **QFUSE prologue cost**: each of the 48 blocks re-quantizes all M×5120 elements (in parallel over its 8 warps).
  If that costs more than about 2 µs, the D2 arm loses its gain; the timing section shows this.
- **PDL inside graphs on GB10**: section C of the device test captures and replays it. The live quant already uses
  PDL inside graphs.
- **Real weights**: the checkpoint's 32×32-block e8m0 scales are expanded to [N, K/32] in `device_test.py`. The
  serving self-check uses the shard as built by `QkvAShard`, so it does not depend on that.

## Skipped: D2(ii), the woar (`_sm120_wo_a_reduce`) quant epilogue

This is a Triton kernel in `models/deepseek_v4/nvidia/ops/sonar_bf16_bmm.py:103`, not this kit's kernel. It is not
trivially exact:
- the epilogue must quantize the bf16-rounded value that `tl.store` produces today;
- each program owns 256 flat positions of an [M, G, N] layout, so 32-wide blocks along wo_b's K (= G·N) must be
  shown to align with program boundaries;
- it needs inline-asm copies of the PTX, because Triton's own fp32→e4m3 lowering on sm_121 is a known trap
  (AGENTS_LOG W-E2);
- wo_b's `mm_mxfp8` must receive a `QuantizedActivation` through `as_quantized_activation`, which is a different
  module chain.

It belongs to a glue:B / B2 kit; the skeptic's best variant for G09 lists it separately.
