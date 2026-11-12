# r3-xsplit — lever G02 (attn:L-XS): fused split-tile sparse MLA decode with exact in-order look-back

One flag, `DSV41_ATTN_XSPLIT` (default 0). One installer, `install.py`. The live FlashInfer kernel, its headers,
its JIT module and its cache are never touched; the new kernel lives in a separate prebuilt library
(`libdsv41_xsplit.so`, loaded with ctypes) and is reached through one gated branch in
`flashinfer/mla/_sparse_mla_sm120.py`.

**Status.** (a) timing build + microbenchmark: **built** (7 libraries on spark-eight, no spills, REG 168 like the
live kernel) and `bench/xsplit_bench.py` written — **not run** (needs a stopped node); (b) exact kernel +
`device_test.py`: built, CPU emulation of the op order passes (401/401, negative controls detected), static SASS
checks pass (C0 of the reference kernel IDENTICAL with operands, FTZ on every FP32 op) — **no GPU run yet**, so no
hardware exactness or timing claim; (c) install kit: done, CPU-tested end to end, `KIT.json` carries the serving
library's sha256. The first window is a **go/no-go on a stopped node** (device test + bench), not a serving A/B.
Nothing outside `~/climb-scratch/r3-xsplit` on spark-eight was written on any Spark.

## Mechanism

Today every decode attention call of the 38 compressed layers runs `sparse_mla_prefill_mg_dual_kernel<DSV4,BF16,8,
64,PBSX,1>` with grid = T tokens (1–6 in decode). Each CTA walks its token's n tiles (n = 10 at d200; 64 KV rows per
tile) serially at ~4.5 µs/tile, so a call costs ~45–65 µs on 1–6 of the 48 SMs (trace q0.4: dual64 61.0 µs ×20,
dual32 44.6 µs ×18 per step, 2.13 ms/step of MLA, on the main stream, overlapping only the WOA l2pf).

XSPLIT keeps the per-tile work, but spreads it:

- **One launch**, persistent grid `min(46, T·nmax + T)` CTAs (46 = 48 SMs minus the 2 the l2pf co-runner holds;
  the MLA CTA owns the whole register file), 384 threads, the live smem layout + 16 B.
- Work items are **claimed from a device counter in index order**: A units (token i, tile t) token-major and
  tile-ascending, then one B job per token, then END. IO warps claim one item ahead and gather it into the parity
  buffer like the live kernel does (so a CTA pipelines its next tile's gather behind the current tile's math).
- **A unit** = the live per-tile body verbatim (gather, QK nope+rope, masks, warp max, cross-warp max), then a
  per-token **prefix-max look-back**: the unit publishes its tile max (one 64-bit epoch-tagged word per head) and
  folds the maxima of tiles 0..t−1 of its token sequentially from −1e30 exactly as the live kernel updates m —
  then continues with w = exp2(s − nm), the warp sums, the V-scale max, the FP8 W quant and the XV-nope MMAs, and
  writes back the operands the cross-tile ops need: alpha for all 16 rows, nm, sc, ls, w, xv, the rope-V B
  operands (25,472 B per tile-row), then a done flag (release).
- **B job** (one CTA per token) replays tiles 0..n−1 in order as their done flags appear (loading the next
  tile's row while replaying the current one; the rope A operand is double-buffered in smem): the guarded FMUL
  (`alpha0 < 1 || alpha1 < 1`, dead-row alpha1 carried), the FFMA `acc += xv·sc`, the separate FADD `l += ls`,
  the 4 chained bf16 MMAs on `acc_rope` with A = bf16(w) staged as the live helper does, then the live epilogue.

Nothing is reduced in a new order: the running max is the only cross-tile reduction in A, and it is folded
sequentially. EXACTNESS.md has the op-by-op map and how the four mandatory corrections are met.

## Why flag-off is identical

- `install.py` makes two anchored edits to `_sparse_mla_sm120.py` (8 lines, all tagged `# dsv41-r3-xsplit`): one
  module-level bool `_DSV41_ATTN_XSPLIT` read from the environment at import, and an `if _DSV41_ATTN_XSPLIT and
  planned.variant is KernelVariant.PREFILL_MG_DUAL:` branch placed before the existing dispatch. With the flag
  unset the branch is a single false bool test: the gate module is never imported, the library never loaded, no
  workspace allocated; the planner, the binding call, the kernels, streams and CUDA graphs are today's.
- `tests/test_install.py` proves the AST of the patched file minus the tagged lines equals the base AST, and that
  the anchors match exactly once on the live base (sha256 `0b93e369…`, identical 8/8, untouched by every round-1
  and round-2 kit).
- The live JIT module (`sparse_mla_sm120.so`, its sources and headers) is not modified, so it is not rebuilt.

## Why flag-on is bit-exact (summary; details in EXACTNESS.md)

Every output element sees the live op sequence on bit-identical operands: A runs the live per-tile code text with
the live compile flags; B runs the four cross-tile recurrences (guarded rescale, FFMA, FADD, rope MMA chain) and the
epilogue in tile order with the ops pinned to the live SASS forms (`__fmul_rn`, `__fadd_rn`, `__fmaf_rn`,
`rcp.approx.ftz`). Dead rows 8–15 never reach the output; their alpha1 still drives the guard. The rope B operands
are copied with the live loads (idx < 0 → 0). Evidence so far: CPU emulation 401/401 bit-identical with negative
controls detected; the device test is the proof.

Two run-time guards in the gate: the first eager call of every shape is checked bit for bit against the live
kernel on the live inputs (a shape is never routed inside a CUDA-graph capture before it passed), and a mismatch
raises (fail closed) unless `DSV41_ATTN_XSPLIT_ON_MISMATCH=disable`. Warmup inputs are dummies, so this check is a
guard against a wrong library or ABI, not the exactness proof.

## Race analysis

- **Cross-CTA visibility**: tile maxima are single-copy 64-bit words `(epoch << 32) | bits` (`st.release.gpu` /
  `ld.acquire.gpu`); rows are written with `st.global.cg`, math barrier, `__threadfence` + `st.release.gpu` of the
  done flag, read after `ld.acquire.gpu` + math barrier with `ld.global.cg`. No `__ldg`/`ld.global.nc` and no
  `const __restrict__` on workspace pointers (`tools/sass_check.py` lists the LDG flavours of the build).
- **Stale state across the 38 calls per step and graph replays**: the epoch E = GEN + 1 is read once per call; the
  last CTA to finish resets claim/finished and publishes the next GEN (clears every flag before a 32-bit wrap).
  A flag from an earlier call or replay never equals E. The device test poisons rows with NaN bytes and flags with
  the previous epoch / random tags between calls and graph replays.
- **Deadlock**: every wait targets a smaller claim index whose CTA is running and publishes it without waiting on
  a larger one, so the kernel completes for any grid, any residency, any co-runner. The device test runs it beside
  46/47/48 SM-hogging CTAs and beside the l2pf stand-in; a 2 s spin bound traps instead of hanging.
- **In-CTA**: IO/math handoff = the live mbarrier + `bar_sync_alt/bar_arrive_alt` protocol, one phase per item; a
  claimed item id travels through a per-parity smem slot written before the mbarrier arrive. B stages its output in
  the current item's own KV buffer, which the IO warps cannot refill before the item's arrive.
- **Streams**: one workspace slot per calling stream (eager stream, capture stream), allocated once at the first
  eager MG_DUAL call (in vLLM's profiling run, so the 2 × 29.3 MiB are accounted before the KV cache is sized);
  a graph keeps the slot of its capture stream. Assumption (as today): routed calls are never concurrent on two
  streams sharing a capture stream — true for the target forward; the drafter's single-cache kernel is not routed.
- **PDL**: launched without the programmatic attribute and never triggers dependents, so stream order around it
  is today's even if neighbouring kernels use PDL.

## Expected saving (models; nothing measured on a real XSPLIT kernel yet)

- Gain skeptic (model2/3, request-averaged): **+0.6 / +2.35 / +3.1 % G1** (low/central/high); exactness skeptic
  +0.43 / +1.74 / +3.54 %; push-B (fused, write-back priced) +0.25..+0.62 % at its central. Per call (central):
  dual64 T=2/4 63 → 34 µs, T=6 61 → 39 µs; dual32 T=2 44.6 → 35.9 µs. KIT.json carries 0.6 / 1.7 / 3.1.
- The kill line: real S0 > 30 µs or c > 0.8 µs/tile beside WOA, or the step model under +0.119 ms (+0.5 % G1).
  `bench/xsplit_bench.py` prints exactly these.
- Haircuts already known: the WOA wo_a prefetch gets a shorter attention window to hide in (0–0.2 ms/step), and the
  diet/xv attention levers are not additive with this one.
- Write-back: 25.5 KB per tile-row, ≤ 2.7 MB per call, L2-resident; the bench's `a3r1` arm (A without row stores)
  prices it directly, `a1r1` prices the prefix barrier, `a2r1` the B replay + epilogue, `a0r0` the rope-B copy.

## How to read it in an nsys trace

- Control: per step 20 × `sparse_mla_prefill_mg_dual_kernel<…,64,64,1>` (~61 µs, grid T) and 18 ×
  `<…,64,32,1>` (~45–57 µs), plus 5 × `sparse_mla_prefill_mg_kernel` (single, unchanged).
- Treatment: the 38 dual calls are replaced by `dsv41_xsplit_mg_dual_kernel<64>` / `<32>` (grid min(46, T·nmax+T),
  384 threads, 96,928 B dynamic smem); the 5 single calls stay. No other new kernel, no memset, no extra launch per
  call. The attention-bucket sum per step should drop by the bench's step-model figure; the l2pf (`l2pf_kernel`,
  grid 2) should still end inside the attention window; the kernel after each attention call should start ~0.35 µs
  after it as today. Update `anat3.py` to put `dsv41_xsplit_mg_dual_kernel` in the attention bucket.
- If any dual call still appears in a flag-on trace: that shape fell back (T > 64, capacity, unchecked at capture);
  `DSV41_ATTN_XSPLIT_VERBOSE=1` logs routed shapes, `rank0.log` carries the gate's `[dsv41_xsplit]` lines.

## Build (CPU only, spark-eight, never while CLUSTER_LOCK exists) — already done, `build/`

    bash ops/build_guarded.sh serving a0r1 a0r0 a1r1 a2r1 a3r1 a0r1p0   # refuses / kills on a lock
    # copy ~/climb-scratch/r3-xsplit/build back into ./build; serving sha256 goes to KIT.json files[2].sha256

Built 2026-11-08 ~15:16–15:45Z: `build/SHA256SUMS`; serving `libdsv41_xsplit.so` = `d5250d79…`; its XSPLIT
kernels are SASS-identical to the bench library's (`TEST-RESULTS.txt`).

`tools/sass_check.py build/libdsv41_xsplit.sass build/libdsv41_xsplit_bench_a0r1.sass base/sass/dual64.full
base/sass/dual32.full` must print `SASS CHECK PASS`.

## Window runbook (lead)

Stage 1 — go/no-go, one stopped node (say spark-five), ~45 min including stop/relaunch. `bash ops/stage.sh
spark-five` copies the kit (build/ included, sha256-checked) to `~/climb-r3/r3-xsplit`; stop the service on all
ranks as usual, then:

    cd ~/climb-r3/r3-xsplit
    timeout 2400 ~/venvs/vllm032/bin/python3 -I device_test.py --lib-dir build --out out/devtest.jsonl > out/devtest.log 2>&1; echo rc=$?
    timeout 1800 ~/venvs/vllm032/bin/python3 -I bench/xsplit_bench.py --lib-dir build --out out/bench.jsonl > out/bench.log 2>&1; echo rc=$?
    pgrep -af "device_test|xsplit_bench" || echo GONE

`device_test.py` exit 0 and its last line `DEVICE TEST PASS` are required; any mismatch line names phase, class,
shape and first differing index. The bench prints S0/c fits (solo and beside the l2pf stand-in), the barrier /
B / write-back / rope components and the step model. GO = 0 mismatches and S0 ≤ 30 µs, c ≤ 0.8 µs/tile and
step saving ≥ 0.119 ms; NO-GO otherwise (the family closes if the real kernel lands at the push-B kill line).
Relaunch production unchanged afterwards.

Stage 2 — serving, only after stage 1 GO: `python3 install.py install` on all 8 nodes (copies the library to
`~/dsv41-r3/xsplit/build`, checks its sha256), flags-off relaunch → `exact.py` == ec2f4437f8e1ed09 and G1 digest
2f7fdccbd388225a, then C T C T with

| arm | env |
|---|---|
| control | `DSV41_ATTN_XSPLIT=0` (or unset) |
| xsplit  | `DSV41_ATTN_XSPLIT=1` |

with the AV pin kept (a re-profiled AV table could change verify widths, F8), the flag verified in the engine
environ, `grep dsv41_xsplit ~/dsv41-logs/rank0.log` showing `loaded … sha256=` and no `DISABLED`/`self-check`
line, and every exactness check identical (exact.py, G1 digest, 9 long prompts). `python3 install.py uninstall`
restores the base bytes and removes the module and library.

## Risks

- The real kernel may land far from the proxy (first build of a persistent kernel): the bench is the gate.
- Register pressure: the IO warps run claim/geometry code after `setmaxnreg.dec 32`; the build log
  (`*.resource.txt`, ptxas output) must show no spill growth that hurts; the math warps keep 232.
- Workspace 2 × 29.3 MiB per rank (env-tunable), allocated in the profiling run: the KV cache shrinks by that much.
- Shapes outside the envelope (T > 64, other head counts, other page sizes, missing lengths) run the live kernel;
  long-context topk_extra beyond 1152/T tiles too (production index_topk = 512 → nmax 10, so T ≤ 64 always fits).
- `__trap()` on a 2 s spin is fail-loud: it kills the process (CUDA error) instead of hanging the step.
- Not additive with the g130-attn diet/xv arms; a shorter attention window gives the WOA prefetch less cover.

## Files

- `src/dsv41_xsplit_kernel.cuh` the kernel (includes the live `prefill_mg_kernel.cuh`); `src/dsv41_xsplit.cu` the
  C ABI (serving) + bench exports (live dual kernel instantiation, l2pf stand-in, SM spinner); `src/build.sh`.
- `files/flashinfer/mla/dsv41_xsplit.py` the gate; `install.py`; `KIT.json`; `EXACTNESS.md`.
- `device_test.py` (deliverable b), `bench/xsplit_bench.py` (deliverable a), `tools/sass_check.py`.
- `tests/test_install.py`, `tests/test_gate.py`, `tests/emu_xsplit.py` (+ `.out`); `TEST-RESULTS.txt`.
- `build/` the 7 prebuilt libraries + `SHA256SUMS`, resource usage, build logs, `sass_check.txt`.
- `ops/build_guarded.sh` (lock-guarded build on spark-eight), `ops/stage.sh` (copy the kit to a test node).
- `base/` live copies (flashinfer mla/, csrc/, include subtree, jit/, live `build.ninja`, live SASS dumps).

Launcher note: the brief names fda6d05a; the lead's 14:35Z lock says production 08a249da. This kit touches no
launcher line and its base file (`_sparse_mla_sm120.py`, `0b93e369…`) was re-verified identical on 8/8 nodes.
