# r3-xsplit — adversarial review (review-xsplit, 2026-11-08)

**Verdict: ship (install).** With the flag off, the patched file is byte-identical to the live one apart from the
tagged lines, the installer is sound, and the claimed tests reproduce. **Flag-on is NOT proven.** Exactness and race
freedom hold on paper (op-by-op and memory-model review below), but no GPU run exists. Turn on `DSV41_ATTN_XSPLIT=1`
for serving only after the kit's own stage 1 on a stopped node: `device_test.py` must print `DEVICE TEST PASS` (exit 0)
and `bench/xsplit_bench.py` must return GO. No blocking defect was found. Six non-blocking fixes are listed at the end
(N1 and N2 first).

## What I verified myself

| check | result |
|---|---|
| live base `flashinfer/mla/_sparse_mla_sm120.py` sha256, 8/8 nodes (re-read 2026-11-08, read-only) | `0b93e36914f2614d…` on spark-one..eight = KIT.json `base_sha256`; no `.bak-*`, no `dsv41_xsplit.py`, no `~/dsv41-r3/xsplit` on any node |
| kit `build/*.so` sha256 vs `build/SHA256SUMS`, KIT.json, sidecar | all 7 match; serving `libdsv41_xsplit.so` = `d5250d7949fa7cae…` (also on spark-eight scratch) |
| `.so` deps | `libcudart.so.13` (soname, resolves to torch's already-loaded cudart), libstdc++, libgcc_s, libc |
| `tests/test_install.py` (512S1, exo-venv313) | rc=0, 23 PASS, `ALL PASS` |
| `tests/test_gate.py` | rc=0, 31 PASS, `ALL PASS` |
| `tests/test_devtest_cpu.py` | rc=0, `input sets checked: 924`, `ALL PASS` |
| `tests/emu_xsplit.py` | rc=0, 16 s; `runs=401 mismatches=0`, negative controls l_ffma 7 / rope_split 7 / stale_epoch 16 / no_alpha1 1, `EMULATION PASS`; identical to `tests/emu_xsplit.out` |
| `tools/sass_check.py` on the kit's SASS dumps | rc=0, C0 dual64/dual32 2429 = 2429 IDENTICAL, FP32 without .FTZ = 0, `SASS CHECK PASS` |
| sandbox install → diff → reinstall → uninstall (copy of live base) | 8 tagged lines (line 81, lines 698–704), `already patched` on re-run, uninstall restores `0b93e369…`, module + lib + sidecar removed |
| `device_test.py`, `bench/xsplit_bench.py` | **not run** (need a stopped node). The builder makes no claim about them. |

## Flag-off identity

The diff against the live file is exactly:
`+ _DSV41_ATTN_XSPLIT = __import__('os').environ.get('DSV41_ATTN_XSPLIT','0') == '1'` after `import torch`, and a
7-line `if _DSV41_ATTN_XSPLIT and planned.variant is KernelVariant.PREFILL_MG_DUAL:` block before the
`DECODE_SPLITK` dispatch inside the registered custom op `_paged_attention`.

With the flag unset:
- Each Python call does one global lookup and one bool test (eager and capture only; graph replays run no Python).
- Nothing is imported, loaded or allocated.
- Kernels, launch order, streams, events and graph contents are today's.
- The live JIT module, its sources, headers and cache are untouched.

No runtime transform (`*.pth`, sitecustomize, launcher, supervisor manifest) touches this file. No round-1 or round-2
kit patches it either, so the installer composes trivially.

## Exactness when on (paper review; the device test is the proof)

I compared `src/dsv41_xsplit_kernel.cuh` with the live `prefill_mg_impl` (base `prefill_mg_kernel.cuh`, sha
`0ec1603d…`, lines 1300–2127) and `xv_rope_mma_mg` (`common/xv_rope_mma.cuh` 126–196).

**A unit = the live per-tile body verbatim.**
- Same parts, in live order: QK nope (BF16 path, `MG_N_HG == 1`), `compute_qk_rope`, the masking (dual,
  `!ASSUME_FULL_TILES`), `s = qk*sm_scale_log2e`, warp and cross-warp max, the `w_head_sc` atomicMax path,
  `/FP8_MAX`, the UE8M0 W quant and the XV-nope MMAs.
- The running max is folded sequentially from -1e30 over the published tile maxima: `fmaxf(fmaxf(-1e30,tm0),tm1)…`.
  This is the same chain as the live `m_smem` update, and fmaxf is exact.
- `alpha = exp2f(old_m-nm)` and `w = exp2f(s-nm)` use the same expression text and the same flags.

**B replay.**
- The live guard (`alpha0<1 || alpha1<1`, with the dead-row alpha1 carried) is followed by FMULs.
- `l` is replayed as a separate FMUL and then an FADD (F2).
- The acc FFMA is one FFMA per element.
- The rope MMA uses A = bf16(w) for the live rows. Dead rows 8–15 get 0 instead of bf16(w2/w3), which only feeds
  dead output rows because HMMA rows are independent (F3).
- The rope-B operands come from the same `load_rope_v` loads (idx<0 → 0, pbs per phase).
- The epilogue is pinned to the live SASS forms: sink FFMA, `rcp.approx.ftz`, the LSE FMUL/FADD chain.
- The final `m` is the last tile's `nm`; `n == 0` gives -1e30 and l = 0, as in the live kernel.

**SASS cross-checks** (my reading of the histogram):
- FFMA: xsplit 15 = 14 acc + 1 sink; live 29 = 28 acc + 1 sink. Neither build contracts `l`, `s-nm` or the
  rescale.
- HMMA 36 vs 40 is **not** a hidden op difference. The live kernel instantiates `xv_rope_mma_mg` twice (the
  `is_main` page-size-64 and extra page-size-`PBSX` template instances, live SASS lines 1996–2016 and 2218–2238).
  XSPLIT has one B call site because the rope-B operands come from the workspace.
- `FADD(0, x)` at the start of the l and ts sums cannot differ from the live constant-folded form: warp l partials
  are 0 or normal under FTZ, never denormal.

**Remaining exactness risk:**
- Codegen-level: e.g. a pinned `__fmul_rn` form against a live FMUL that the compiler moved under a different
  predicate. Static checks cannot rule this out; only `device_test.py` can.
- `device_test.py` uses synthetic random and hostile inputs, not *recorded* production inputs (the brief asked
  for both). This is mitigated by the window protocol: flag-on `exact.py` / G1 digest / 9 long prompts are
  end-to-end bitwise.

## Races, graphs, deadlock

**Deadlock.** The claim order is global and monotone, and each CTA claims at most one item ahead. Every wait (prefix
max, done flag, B prefetch `wait_row(ti+1)`) targets a smaller index. A unit publishes its tile max *before* its own
look-back. So the smallest unfinished item is always being executed by a resident CTA. The kernel progresses for any
grid, any residency and any co-runner, because a CTA that is not resident holds no claims.

**Memory ordering.**
- Rows are written with `st.global.cg`, then a math `bar.sync`, then thread 0 does `__threadfence` + `st.release.gpu`
  on `done`.
- The consumer does `ld.acquire.gpu`, then `bar.sync`, then `ld.global.cg`.
- The tile maxima are aligned 64-bit release/acquire words.
- In the SASS, every workspace load is `LDG.E(.64).STRONG.GPU` (74 + 2). The weak `LDG.E/.U16` loads are read-only
  inputs: indices and the rope loads from the KV cache.

**Epoch and reset.**
- Every CTA reads `GEN` at start.
- The last CTA, decided by a `FINISHED` atomic that every CTA increments after its END claim, resets
  `CLAIM`/`FINISHED` and publishes `GEN`. By then no claim can still be in flight.
- At the wrap, the last CTA clears the flags and sets `GEN=0`, so E restarts at 1. A zeroed workspace starts at E=1.

**In-CTA.**
- This is the live mbarrier + `bar_sync_alt/arrive_alt` protocol. Counts match: IO syncs once per non-END item, math
  arrives once per non-END item.
- The slot handoff uses the mbarrier release/acquire.
- The B staging buffer is the item's own KV buffer.
- The rope A operand is double-buffered by `ti&1`. The barriers ordering read(ti) → write(ti+2) are present.
- The new-token Q prologue writes the q_rope scratch, which aliases W_FP8, only after the previous A unit's
  end-of-unit barrier. A never follows B in claim order.

**Graphs.**
- No host sync, allocation or attribute set inside capture: the pool is allocated and `cudaFuncSetAttribute` runs at
  the first eager call or self-check.
- The grid and NA are fixed per captured shape; per-token lengths are read on device.
- No PDL attribute is set and nothing triggers dependents.

**Streams (production startup census, from rank0.prev.log and the V2 runner).**
- `--kv-cache-memory-bytes` is set, so memory and CUDA-graph profiling are skipped ("skipped memory profiling").
- `warmup_kernels` runs eager decode steps (tokens 7/6/1) on the **default** stream, which takes slot 0 and checks
  those keys.
- `CudaGraphManager.capture` runs each desc's eager warmup and then captures on a fresh `graph_capture()` stream,
  which takes slot 1. Its warmups self-check every T ≤ 64 key before capture.
- Replay and graph-capture (DSpark) captures that come later find no slot and keep the live kernel. That is safe.
- So under today's config the decode graphs should route. The slot design is fragile, though (see N1).

## Installer

- Anchors are counted against the live base (1 and 1). The TAG is on all 8 lines.
- It refuses when the base sha256 differs and refuses a tagged file that is not this patch. It is idempotent.
- The tmp-file + `os.replace` write keeps the file mode.
- The library sha256 is checked against KIT.json before and after the copy, and the gate re-checks the sidecar
  sha256 at load.
- Uninstall restores the backup's bytes; I confirmed this in the sandbox.

**Demonstrated defect (non-blocking: no backup exists on any node today).** When a `.bak-r3-xsplit` already exists
and the target is clean, `install` keeps the existing backup without checking it. `uninstall` then writes that stale
backup over the live file. Repro on 512S1: a stale backup containing `# stale`, then install, then uninstall, left a
file with sha `a7954016…` and content `# stale`.

## Non-blocking fixes (recommended before the stage-2 serving A/B)

**N1 (most important for a valid A/B).** `fallback_no_ws` and `fallback_cap` are counted but never logged, and
`stats()` is never printed. Slots are owned per stream and never released, and each `graph_capture()` creates a new
stream. If one more in-envelope stream appears before the real capture (CUDA-graph memory profiling enabled, an
extra warmup on a side stream), every production graph silently keeps the live kernel. The stage-2 checks
(`loaded … sha256=`, no `DISABLED`) would still pass, and the window would measure a no-op.

Fix:
- Log once per (stream, reason) for each no-workspace fallback.
- Log a one-line summary of the routed and fallback counts after capture.
- Add to the runbook: a flag-on nsys trace must show `dsv41_xsplit_mg_dual_kernel` and **zero**
  `sparse_mla_prefill_mg_dual_kernel` in decode steps.
- Optionally default `NWS` to 4.

**N2.** `install()` should check `sha(BAK) == BASE_SHA` when a backup already exists, or rewrite it from the
verified target. It should also verify the library's sha256 **before** patching, so a library mismatch cannot leave a
half-installed state (patched file and module, no library). That state is safe but untidy.

**N3.** KIT.json records no sha256 for the new gate module `files/flashinfer/mla/dsv41_xsplit.py` (sha `932f1b94…`
as copied), and the installer does not verify it.

**N4.** With `ON_MISMATCH=disable`, graphs captured before a later mismatch keep running XSPLIT. The default `raise`
is fail-closed and fine; document this or remove the option.

**N5.** The envelope accepts `model_type == -1` (AUTO). It is harmless, because the 584-byte row check and the
planner exclude other families, but it should be DSV4 only.

**N6.** README and KIT.json say the 2 × 29.3 MiB workspace is allocated in the profiling run and the KV cache shrinks
by that much. Production sizes the KV cache with `--kv-cache-memory-bytes` and skips profiling, so the workspace is
just 58.6 MiB of extra free memory, taken at the first eager MG_DUAL call (`kernel_warmup`). Only the doc is wrong.

## Notes

- The 2 s `__trap` measures wall time (globaltimer). If the GPU is time-sliced with another process for more than
  2 s, it would kill the rank even though nothing is wrong. That does not apply on a dedicated serving node or the
  stopped test node.
- After a flag-on boot, check that the FlashInfer SM120 sparse-MLA autotune and crossover cache file is unchanged.
  A re-calibration with XSPLIT routed could move the MG_DUAL / DECODE_SPLITK crossover, and those are different
  kernels. Today the cache is loaded from file, so the calibration does not run.
- The gain (+0.6 / +1.7 / +3.1 %) is modelled only. The bench kill bars (S0 ≤ 30 µs, c ≤ 0.8 µs/tile beside the
  l2pf, step saving ≥ 0.119 ms) are the go/no-go.
