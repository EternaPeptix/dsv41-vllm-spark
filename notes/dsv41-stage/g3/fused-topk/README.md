# Fused CANON: isolated next-window GPU qualification

**Ready to run; GPU execution has not been performed.** This folder is self-contained. Local checks passed without importing Torch, Triton or vLLM. It tests the R17 fused **canonicalization** kernel, not streaming MQA/top-512 selection. It does not install or patch serving code, start services, or run inference.

The default plan has **476 qualification cases and 41 timed cases**. Rows 1–192 are swept individually, plus 2048 and 16384. Every row count gets a compact mixed-value case. Stress counts get all four layouts and seven value patterns; twelve additional padded-decode cases check compaction, including zero live rows. `--matrix full` expands to the complete Cartesian product.

## Commands

CPU-only checks can run now, from this directory:

```sh
python3 check_cpu.py
python3 run.py --mode plan --output plan.json
```

For a **future authorized, exclusive GPU test window**, place this whole directory in a dedicated qualification directory on one Spark. Use the existing Torch/Triton environment; no package installation is needed. These commands have **not** been run on the cluster:

```sh
cd /home/jeweled/qualification/fused-topk
PYTHON_BIN=/home/jeweled/venvs/vllm032/bin/python

# Short test, including side-stream execution and graph replay.
timeout 1800s "$PYTHON_BIN" run.py --mode all --execute-gpu \
  --rows 1,7,192 --layouts compact,offset,rowpad \
  --patterns mixed,bounds,max,negative --matrix full \
  --warmup 2 --iterations 10 --samples 2 --output runs/smoke.json

# Full requested row sweep, adversarial stress layouts, graph replay and timing.
timeout 3600s "$PYTHON_BIN" run.py --mode all --execute-gpu \
  --output runs/full.json

# Memory-safety run; its timings are deliberately disabled.
compute-sanitizer --tool memcheck --error-exitcode 86 \
  "$PYTHON_BIN" run.py --mode qualify --execute-gpu \
  --rows 1,7,192,2048 --layouts compact,offset,rowpad,wide \
  --patterns mixed,bounds,max,negative --matrix full \
  --warmup 1 --no-graphs --output runs/memcheck.json
```

If `compute-sanitizer` is not on PATH, use its existing installation path. Do not install it as part of this harness. A timeout, signal, missing final result, nonzero exit, or any final status other than `PASS` is **not a qualified run**. The CPU planner's `PLANNED` status is never GPU evidence. The deliberate execution-gate rejection printed by `check_cpu.py` is an expected negative control.

Default target is logical CUDA device 0 with capability **12.1 (SM121)**. `--expect-sm 12.0` is an explicit alternative for separate SM120 qualification; results record the actual device. Default maximum individual backing is 67,106,968 bytes (just under 64 MiB); simultaneous input/output/graph/temporary allocations require more memory. The harness does not reserve or inspect other workloads, so run it only in the agreed window.

## What is executed

`kernels.py` is extracted from pinned sources bundled in `source/`:

- the complete R17 `CanonTopK512Kernel.kernel`, including its original Triton decorator options and `num_warps=4` launch;
- the complete original installed `_canon_topk_` Torch baseline;
- the actual integer `UnpackSeqTritonKernel.kernel`, launched with the caller's 64×64 blocks, four warps and two stages.

The only extraction changes are removing the Python `staticmethod` descriptor, renaming the unpack function to avoid a name collision, and formatting. Full body/decorator ASTs and hashes are checked before any GPU import. Runtime pins require the recorded Torch Git commit and exact installed Triton JIT/sorting-network source hashes. `build_sources.py` is a local maintainer extraction tool; the run does not need it to regenerate anything or access its historical origin paths.

Inputs cover arbitrary negatives including `INT32_MIN`, positive `INT32_MAX`, duplicate valid indices, all-invalid rows, full signed random data, canonical sorted data and reversed data. Mapping is precisely: negatives become `INT32_MAX`, sort ascending, then every `INT32_MAX` becomes `-1`. Duplicates remain duplicates.

Layouts include compact rows, a nonzero storage offset, 527-element padded row strides and 1024-element wide strides. The complete backing storage is compared byte-for-byte against both the original Torch operation and an independent CPU integer oracle. Prefix, suffix and inter-row canaries must remain unchanged. Padded cases canonicalize all padded rows, unpack to distinct output storage, and copy the compact prefix back. The entire main and unpack backing is checked, including the canonicalized tail; no tail is silently discarded.

Current- and side-stream tests synchronize their producers before use and keep all input/output/length/graph objects alive through completion. Graph replay changes the input pattern while preserving capture addresses, and compares every replay. For padded cases the graph covers a preallocated canonical→unpack→copy kernel sequence. It does **not** claim the original host `.item()`/allocation wrapper is graph-capturable.

## Warmup, safety and timing

GPU execution requires both a non-plan mode and `--execute-gpu`. `TRITON_INTERPRET`, always-compile and override/dump modes are rejected before importing Triton. Torch override modes, Triton interception/stage/launch/pre-run hooks, instrumentation and external async compilation are rejected. These checks are not a seal against arbitrary modified libraries or concurrent global-state mutation; use a fresh isolated process.

All planned shapes/alignment cases execute in the explicit warmup phase. The harness then seals each **isolated kernel object's** `_do_compile` entry and snapshots cache keys/objects. Any subsequent compile attempt or cache change fails qualification, graph capture or timing. This establishes no post-warmup **Triton compile entry** in a successful run; it does not attest every CUDA-driver internal JIT/cache activity. A warmup or later failure records `gpu_executed: true` once CUDA warmup may have started, and retains the compile audit.

Compiled artifacts go into an output-specific marked cache directory, normally `runs/full.triton-cache`. Existing unmarked nonempty directories, `~/.triton` descendants and installed-package locations are rejected. Nothing is written to the installed serving modules or their normal Triton cache. The source objects and their compile-entry traps exist only in this new process.

Timings are reported separately for direct launch and graph replay, with alternating baseline/fused order across samples. Every iteration restores the same unsorted input **outside** its CUDA-event interval and host-operation timer. JSON contains CUDA-event mean/median/min/p95, host submission timings and total batch wall time including resets/events/synchronization. Final timed output is rechecked. Resets warm input caches and event/host submission overhead can affect these short operations; these are isolated-operation measurements, not serving throughput.

The fused arm directly launches the extracted kernel. It **excludes installed eligibility checks and warmup-owner wrapper overhead**. Padded timing includes the identical unpack/copy tail in both arms and is labeled separately by case. A faster isolated result does not establish an end-to-end TP8 or deployment gain.

## Files and current validation

- `run.py`: gated runner and JSON output.
- `plan.py`: standard-library-only cases, data, CPU oracle and canaries.
- `kernels.py`, `source/`, `provenance.json`: exact extracted implementation and source pins.
- `check_cpu.py`, `cpu-results.json`: local self-checks; `main(write=False)` is available for independent review.
- `plan.json`: default CPU-only execution plan.

Local checks passed: 152 oracle fixtures / 719,872 values, full-backing and canary corruption controls, 17 CLI/cache/interpreter/JIT rejection checks, source AST/hash checks, reset/event timing order, and truthful failure metadata. No GPU imports, compilation, CUDA execution, graph capture, sanitizer run or GPU timings have occurred. Native exactness and performance remain for the authorized execution window.
