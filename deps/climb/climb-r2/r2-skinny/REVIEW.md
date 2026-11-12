# r2-skinny: adversarial review (review-skinny, climb round 2, 2026-11-08 ~14:15Z)

**Verdict: FIX.** The design is sound, and I found no exactness, race or flag-off defect in the source. But the
kernel has never been through nvcc/ptxas, no binary exists (`prebuilt/` is absent and the KIT.json `.so` sha256 is
`null`), and `device_test.py` has never run. The brief makes the device bitwise test and a compile log deliverables
before any window, so the kit cannot be installed as it stands.

## What I verified myself

| item | result |
|---|---|
| live `dsv41_r3_dense.py` on spark-one..eight | sha256 `1f1c0456…a9193` on 8/8 (rechecked at 14:14Z during window w1c) == `base/` == KIT.json |
| live `skinny_mx.cu` / `skinny_mx_ext.so` (spark-one) | `0cf2d309…` / `c054de3a…`; the .so is identical on 8/8 |
| `ref/flashinfer/*.py` vs live flashinfer 0.7.0.post1 | sha256 identical (mxfp8_quantize.py `271ec175…`, quantization_cute_dsl_utils.py `3689f299…`, fp4_common.py `a430b317…`) |
| kit file hashes vs KIT.json | `dsv41_r2_skinny.py` `5010c4e9…`, `skinny_mx_v2.cu` `48a18b92…`, `build.py` `d12de6ff…`: all match |
| `tests/test_install.py` (this Mac) | rerun: ALL PASS (21 tagged lines, AST(patched minus tagged) == AST(base), 5 guarded outer branches, uninstall restores `1f1c0456…`, refusal works) |
| `tests/test_emulation.py` (512S1, full) | rerun: ALL PASS, 79 PASS lines, 38 s, exit 0, same as the claimed log |
| `tests/test_routing.py` (512S1) | rerun: ALL PASS (8 scenarios) |
| `tests/test_device_harness_cpu.py` (512S1) | rerun: ALL PASS (ftz=0 matches, ftz=1 differs in 8456 scale bytes) |
| compile | **not possible.** CLUSTER_LOCK is held again: window w1c, token `claude-w1c-20261108T134808Z`, eta 17:48Z, on production `08a249da`. There is no nvcc off-cluster. |

## Flag-off identity: holds

- Edit 1 (module level) only reads env vars. `R2_QFUSE = R2_V2 and …` short-circuits.
- Edit 2: `if R2_V2:` is False, so no import, no `.so` load and no `_r2` attribute.
- Edit 3 `if R2_V2 and self._r2…` and edit 4 `if R2_QFUSE and self._r2…` short-circuit on the bool, so `_r2` is never
  touched.
- Kernels, launch order, streams, events, allocations and graph contents are unchanged. The AST check confirms
  this, and so does the routing test (no `_r2`, r2 module never imported).
- No other kit (r1-attn/drafter/eager/pdl/pf, r2-attnout/draftmoe/moeback/moefront/oblit) patches
  `dsv41_r3_dense.py` or `dsv41_r2_skinny.py`.
- There are no sitecustomize/transform anchors on this file.
- `dsv41_l2_prefetch._wrap_r3_dense` wraps `qkv_a_sharded` only and records `st.s_rm` / `st.w`. v2 reads the same
  storage.

## Exactness when on: correct by construction, but not proven on hardware

I checked `skinny_mx_v2.cu` against `skinny_mx.cu` line by line.

- v1 at (N=224, K=5120, grid 48) gives `pick_split` = 8 (eff 0.933 vs 0.778 for 4), so ROWS_PER_IT = 1 and kv_per = 40.
  The UNROLL loop never runs, because `kb+L+96 < kb+40` is false. Each lane therefore computes
  `fmaf(s0,dot(kb+L),0)`, then `fmaf(s1,dot(kb+32+L),acc)` for L < 8. v2:337-369 reproduces this.
- The second-vector shuffle source `(i%4)*8 + L%8` from pass `i/4` holds row `4*(i/4) + src/8 = i` and vector
  `kb+32+L`, which is correct. The shuffle runs outside the `lane<E` guard, as it must for FULL-mask shuffles.
- Activation smem index `m*kv8 + 2k` (A8) and `m*kv8 + elem>>3` (QF) equal v1's `2*(m*kv16+v)` layout.
- K slices are 40 vectors = 20 MX blocks, so each warp's slice is block-aligned and warp-private. `__syncwarp`
  suffices.
- `deq8` and `dot16_cvt` are v1's arithmetic. Weight conversion hoisting is exact.
- The butterfly order and the `t=0; t+=part[q]` combine order are unchanged. Build flags are identical to the live
  `build.py` (`-O3 -lineinfo -Xptxas=-v`, no fast-math).
- QF PTX is byte-for-byte the FlashInfer inline asm:
  - `bfloat2_habs2`, `hmax2`, `hmax_reduce_to_f32`, `fmax_f32`, `float_to_ue8m0_fast`, `ue8m0_to_inv_scale_fast` and
    `bfloat2_to_fp8x2_scaled` (`mul.f32`, clamp, `cvt.rn.satfinite.e4m3x2.f32 d, f1, f0`);
  - the packing order matches;
  - the fp32 constant 1/448 = `0x3b124925` is correct.
- Max is exact, so the 2T and 4T reduction trees cannot differ.
- vllm `mxfp8_e4m3_quantize` → flashinfer `backend="cute-dsl"` (live mxfp8_utils.py:196-206) is confirmed. M·160 <
  65536 gives the 4T variant.
- `as_quantized_activation` returns None for plain tensors (live quant_activation.py:82), so skipping it in the QF
  path changes nothing. At the only caller (`attention.py` `_fused_wqa_wkv_gemm`), `hidden_states` is a plain bf16
  tensor that the compressor `torch.mm` also reads.
- The remaining gap is that the CPU emulation checks a Python model of the mapping (`tests/emu.py`), not the
  compiled kernel. Hence "unproven" until G/C/X pass on device.

## Races, PDL and capture: no defect found

- Before `griddepcontrol.wait` only static weights and scales are read (v2:245-268). x/xq/xs reads, smem writes and the
  y store all come after it.
- v2 never triggers `launch_dependents`, so the following `ar_kernel_pl` (which never triggers either) sees the same
  ordering as after v1.
- No new streams, events or allocations: one `torch::empty` per call, as v1.
- smem attributes are set eagerly by `prepare()`. The per-instantiation static `cudaFuncSetAttribute` in `launch()`
  also fires on the first (eager, self-check) launch.
- The self-check asserts it is not capturing and runs only in `QkvAShard.__init__`, which the live code already
  restricts to non-capture calls.
- No collectives run in the self-check, so per-rank pick differences cannot deadlock. Every served kernel is still
  bitwise-checked against v1.
- The live quant triggers `launch_dependents` at the very end of its kernel (mxfp8_quantize.py:731-733), so pdl
  overlap is launch-tail only.
- PDL secondaries launch only after every primary block has triggered or exited, so there is no co-residency
  deadlock.
- Device test section C covers a PDL launch whose capture predecessor is a `copy_` (cudaMemcpyAsync node), which is
  the riskiest capture shape for QF+pdl.

## Installer: mechanism OK

- Anchors apply to the live base with the exact counts. Every inserted line is tagged, and there is a `.bak-r2-skinny`
  backup.
- Install refuses an unrecorded sha and is idempotent. Uninstall strips tags, which restores `1f1c0456…` exactly (test
  rerun), and removes the module and `~/dsv41-r2/skinny_mx_v2`.
- The live `~/dsv41-r2/skinny_mx/` is untouched.

## BLOCKING

1. **No binary, never compiled.**
   - Build on spark-eight once CLUSTER_LOCK is gone, per `COMPILE-LOG.txt`.
   - Commit `build.log` with `ptxas -v` for `skinny_v2_kernel<M,5,false|true>`, M ≤ 6: 0 spill stores/loads, and
     registers ≤ 255.
   - Put the `.so` in `prebuilt/` and its sha256 in KIT.json `files[]`.
   - Until then, `install.py install` cannot run without `--build`. With `--build` it compiles 8 independent binaries
     in the window, which no device test has covered.
2. **`device_test.py` never run.**
   - It needs exit 0 on a stopped node with that exact `.so`.
   - It must include `--model ~/models/dsv41/official-fp8` (real shards) and sections Q/G/C/X.
   - Brief rule 3 requires this before any window arm.

## Non-blocking (should fix)

- **Q section runs the 2T kernel.** The exhaustive quant check feeds ~2.7k rows (R·160 ≥ 65536), so the live
  reference runs FlashInfer's **2T/SF** compiled variant. Serving (M ≤ 8) and `_pick_ftz` (256 rows = 40960 blocks) use
  the **4T/SF** variant. The two are equal by the max-exactness argument, but the check should hit the serving
  kernel: run the exhaustive x in chunks of ≤ 409 rows.
- **Installer accepts any `.so`.** `install.py` accepts any existing `SO_DST` when KIT.json's sha256 is null, and
  `status` cannot flag a mismatch then.
- **Installer accepts a stale patch.** The "already patched" path checks only the tag-stripped base sha, not that the
  tagged lines equal the current `EDITS`. Compare `patched(unpatched(src)) == src`.
- **No recorded activations.** `device_test.py` has no recorded production activations (`--acts`), which the brief
  asks for.
- **PDL perf risk (not exactness).** With pdl, v2 blocks (`__launch_bounds__(256,1)`, up to 80 KB smem) become
  resident at the quant's trigger and park at the wait. If registers exceed 128 per thread, a parked block owns the
  SM's register file, and aux-stream compressor/weights_proj blocks can be delayed. The window A/B will show this.
- **Gains are modelled only** (+0.5 / +1.2 / +1.9 % G1 all-on). Section T of the device test must size them before
  the window is spent.
- **Production drift.** Production was on launcher `08a249da` during this review, not `fda6d05a`. The base file is
  unchanged (rechecked), but re-run `install.py dry` right before installing.

## Fixes (fix-skinny, 2026-11-08 ~15:25Z)

### Blocking 1: no binary, never compiled → FIXED

- **When and where.** Compiled on spark-eight at 15:15:25-15:15:58Z, after window w1c released CLUSTER_LOCK at
  about 15:14Z.
- **Lock discipline.** `scripts/drive_compile.sh` checked the lock first and polled it every 10 s during the build.
  It would have killed the build if a lock appeared, and it removed the remote build dir afterwards.
- **Build settings.** `MAX_JOBS=1 nice -n 19`, `CUDA_VISIBLE_DEVICES=""`, venv toolchain: nvcc 13.0.88, torch
  2.13.0+cu130. Flags are `-gencode arch=compute_121,code=sm_121 -O3 -lineinfo -Xptxas=-v`, no fast-math, the same
  as the live build.py.
- **First attempt.** It failed immediately (`Ninja is required`, ninja is in the venv bin). Fixed by putting the
  venv bin on PATH.
- **`build.log`** (kit root, sha256 `93fcca07…`) covers all 33 entry points. Every one shows 0 bytes stack, 0 spill
  stores and 0 spill loads.
- **Registers, serving instantiation `skinny_v2_kernel<M,5,·>`:**

  | M | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | A8 | 95 | 117 | 126 | 151 | 153 | 156 |
  | QF | 95 | 119 | 127 | 128 | 136 | 143 |

  The full table is in `COMPILE-LOG.txt`. All values are ≤ 255.
- **Shipped binary.** `prebuilt/skinny_mx_v2_ext.so`, sha256
  `b3af28da9120e5efe1493814768fbf42a5d7194e343a5b9369e3214d9f1f9a92`, recorded in KIT.json `files[]`, plus a
  `compiled{}` record. It contains a single `sm_121` cubin.
- **Load check.** A CPU-only import with the serving venv on spark-eight returned IMPORT_OK with the 6 bindings. No
  GPU context was created, and the copy was deleted.
- **Installer.** `install.py install` now copies exactly this binary. It refuses:
  - a null KIT.json sha256;
  - a prebuilt whose sha256 differs from KIT.json;
  - a destination `.so` that differs when the kit has no prebuilt.

  `--build` remains only as an escape hatch, and `status` then reports `MISMATCH`.

### Blocking 2: device_test.py never run → NOT RUN, still the gate before any arm

- **Why I could not run it.** The test needs a stopped GB10 node. Under the round-1 hard rule 1 I may not stop or
  restart anything. Brief rule 3 assigns this run to the lead, in a window. Rule 4 says never to load the result
  into a GPU process. Production was serving on all 8 nodes (launcher `08a249da`).
- **What is ready.**
  - **`scripts/run_device_test.sh [node] [rank]`** (run from the Mac):
    - refuses if any vLLM or GPU compute process runs on the node;
    - copies `device_test.py`, `KIT.json`, `new/` and `prebuilt/`;
    - runs `device_test.py --model ~/models/dsv41/official-fp8 --layers 0,13,39 --rank 7`;
    - fetches the log and json into `tests/`;
    - exits with the test's exit code.
  - **`device_test.py` checks the binaries.** `--v2-lib` now defaults to `prebuilt/`. The test checks that the v2
    `.so` sha256 equals KIT.json (a FAIL otherwise, unless `--allow-any-so`). It checks that the v1 `.so` is the
    live `c054de3a…`.
  - **Sections.** Q, G (including real shards, with QF at pdl off and on), C (graph replay with PDL after a memcpy
    node, for A8 and QF), X and T are unchanged in content.
- **Lever status.** G10 and G09 stay "partial". Their exactness on hardware is unproven until this test exits 0.

### Non-blocking items

- **Q ran the 2T/SF variant → FIXED.** The live reference quant now runs in chunks of ≤ 384 rows (384·160 = 61440 <
  65536), which is the 4T/SF serving variant. `tests/test_device_harness_cpu.py` asserts every live-quant call has
  ≤ 409 rows; it saw [103, 256, 384].
- **Installer accepted any `.so` when the sha256 was null → FIXED** (see Blocking 1). `status` prints
  `UNVERIFIED (KIT.json sha256 null)` or `MISMATCH (!= KIT.json)`. There are 4 new installer tests.
- **Stale patch passed the "already patched" path → FIXED.** `install` requires `patched(unpatched(src)) == src`
  and otherwise refuses without touching the file. `status` prints "patch matches this kit revision" or
  "WARNING … stale patch". This is tested.
- **No recorded activations → PARTLY FIXED.** With `--model`, `device_test.py` adds a `ckpt-L<l>` activation set
  per layer: `rmsnorm(embed[tok]) * attn_norm.weight` from the real checkpoint (64 token ids), with a different row
  window per (M, weight set). It is production-shaped, not a serving record. `--acts file.pt` still accepts real
  records. Capturing them would need a hook in the serving process, which this kit must not add. The CPU harness
  checks `ckpt_acts` against a hand computation on a fake checkpoint.
- **PDL residency (performance only) → DOCUMENTED with measured numbers.**
  - A parked block takes 256 × 95-156 registers, which is 24-40k of the SM's 64k, plus up to 80 KB of smem.
  - The window's T-v2pdl vs T-v2 pair measures the effect. README "Risks" has the numbers.
- **Gains modelled only → UNCHANGED.** Section T of the device test (warm and cold-L2 medians of 200 graph
  replays) sizes them in the same lead run, before any arm.
- **Production drift → CHECKED.** At 15:18Z, all 8 nodes were on launcher `08a249da`, `dsv41_r3_dense.py` was
  `1f1c0456…`, `skinny_mx_ext.so` was `c054de3a…`, and `dsv41_r2_skinny.py` was absent. README tells the lead to
  re-run `install.py dry` right before installing.

### Tests rerun after the fixes (all exit 0)

| test | where | result |
|---|---|---|
| `tests/test_install.py` | this Mac | 24 PASS, including the new binary and stale-patch refusals, run against the real prebuilt |
| `tests/test_emulation.py` (full) | 512S1 | 79 PASS, 40 s |
| `tests/test_routing.py` | 512S1 | ALL PASS |
| `tests/test_device_harness_cpu.py` | 512S1 | ALL PASS |

Logs are in `TEST-RESULTS.txt`.
