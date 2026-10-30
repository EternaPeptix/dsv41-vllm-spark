# D05 component candidate v1

October 24, 2026. **Source candidate ready for root's native build and GPU gate.**
Only CPU source/dispatch/layout checks have run here. No native compilation,
remote mutation, installation, GPU test or performance result is claimed.
This supersedes the earlier investigation-only status in README.md; its live-row
contract and gain caveats still apply.

## Recovered baseline provenance

The original native source was recovered read-only from spark-one:

    /home/jeweled/dsv41-opt-20261020/deepgemm-m32-hidden-v2/source

Its sibling original build artifact has exactly the installed binary SHA256
b1c4eff850fc243ebaad12bd04bae00a7910f310dde20727649517ecf98105a9.
The recorded manifest pins source commit 4c0aac24b7fc860f1f6c9bf1428249ee94e12718,
archive eb3d886984aab798e0f6be6054f201c84a58e1b80c647ee123a617cccd83460f,
and the hidden-symbol setup.py hash bd2420015fa68713eefb0620df44cffdee8f5d8c04f306989d2733e080862ff0.
The current device header and historical host-header snapshot match that source.
The 6,943 copied source-file hashes are saved in baseline-provenance/source-files.json;
the recorded build result, manifest, build script and visibility patch are beside it.
No fc2c-v2 or held D04 code is included.

The immutable baseline native tree remains
/home/jeweled/dsv41-stage/m32/target-m32-native-v1. Preparation validates its
entire manifest, header and installed wrapper before making a new copy.

## Exact change and default-off contract

The patch touches three original native source files, listed with full baseline
and candidate hashes in [candidate-manifest.json](candidate-manifest.json).
[source.patch](source.patch) is reviewable and reverse-applies cleanly.

- The grouped native API adds an optional dsv41_fc2_live_store=False argument.
  Opt-in asserts SM121, alignment32, normal layout and ensure_zero_padding=False.
  The caller explicitly promises the private scatter32 prefix contract.
- The host uses an independent descriptor height8 only for E384, N5120,
  logical K384, FP8×FP4, BF16 contiguous output, BM32/BN64/BK128, staging32,
  swizzle128, 48 SMs, 256 math threads, splitK1 and compiled N/K dimensions.
  Unsupported configurations retain the original full-store path.
- The device appends a default-false template flag, asserts the pinned geometry,
  and changes only the elected lane's output-copy loop. Every live-prefix group
  of eight rows is written; fully dead groups are omitted. Full32 staging,
  BF16 conversions, MMA order, reductions and scheduler are unchanged.
- The generated private wrapper reads DSV41_FC2_LIVE_STORE with default0 once
  when initialized. Only FC2 shape (384,5120,192 packed) opts into the new API.
  The on arm uses a separately hashed native module and JIT cache. The off arm
  loads the original native module/cache. The wrapper is generated only after
  a successful isolated build, with the actual candidate binary hash embedded.

The new JIT name dsv41_fc2_live_store_v1 and template flag identify the actual
on path. The GPU harness requires that kernel to appear in the candidate cache;
a completely guarded-out candidate cannot pass merely by running the fallback.

## Synchronization reasoning

| Mechanism | Original | Candidate |
|---|---|---|
| Shared output staging | All32 rows | Same32 rows, same addresses and conversions |
| TMA output copies per active 64-column tile | One32-row copy | One to four8-row copies, each in-bounds/live at its first row |
| Output group commit | One tma_store_arrive after copies | Same single commit after all selected copies |
| Wait before staging-buffer reuse | Elected lane wait<0>, then all-math named barrier | Unchanged |
| Fence before reading shared output via TMA | Store fence, then all-math named barrier | Unchanged |
| Final output drain | Elected lane wait<0> | Unchanged |
| Input TMA transaction-byte counts and producer/consumer barriers | Original A/B/scales pipeline | Byte-for-byte unchanged |

All math threads stage the full tile and execute the same barriers regardless
of occupancy. Only the already-elected math lane branches on layout values.
Its copies belong to the same committed bulk group. The original waits protect
the entire group's shared-source reuse. Output stores are not charged to the
input-load transaction barriers, so skipping them does not change expected_tx.
The final wait and stream ordering retain the original output completion
contract. The source retains the exact existing wait semantics, including its
read-completion variant; this is not a claim of new stronger ordering.

The scatter32 contract ensures that an active tile has a live first eight-row
group; a completely dead tile is already skipped by both original scheduler
walks. There is no new scheduler early exit. Because swizzle128's row phase
repeats every eight rows, offsets0/1024/2048/3072 bytes preserve the staged layout.
The 8-row descriptor and 32-row staging memory size are intentionally independent.
This reasoning still needs SM121 execution and sanitizer validation.

## CPU validation performed

Run locally:

    python3 make_candidate.py
    python3 check_layout.py
    python3 check_candidate.py

The checks pass 2,130 prefix layouts, including fully invalid reservation tiles,
and 4,096 swizzled byte addresses. Source checks verify the entire scheduler,
numeric body and staging body remain byte-for-byte identical, all existing
synchronization calls/counts are preserved, the wrapper defaults off, FC1/psum
do not request the feature, and the patched Python files parse. Reverse-patch
dry-run and Python syntax compilation also pass. They do not validate native
ABI compilation, CUDA resources, TMA legality, race freedom or bitwise GPU output.

## Root-operated preparation and build

These commands are provided for root to schedule; this agent has not run them
remotely. Use the existing vllm032 Python. On spark-one the recovered original
source and original development headers already exist; for another host copy
the source tree and use its existing equivalent headers. No package install is
required. Replace the example output with a fresh directory for every attempt.

    ~/venvs/vllm032/bin/python prepare_candidate.py \
      --source ~/dsv41-opt-20261020/deepgemm-m32-hidden-v2/source \
      --native ~/dsv41-stage/m32/target-m32-native-v1 \
      --wrapper ~/venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/sm121_target_m32.py \
      --out ~/dsv41-goal120-20261024/deadstores/build-v1

    ~/venvs/vllm032/bin/python build_candidate.py \
      --prepared ~/dsv41-goal120-20261024/deadstores/build-v1 \
      --system-headers ~/dsv41-opt-20261020/deepgemm-m32-build-v1/dependencies/root/usr/include

The build hides GPUs, prepends the interpreter's directory to PATH for ninja,
keeps hidden native symbols, limits MAX_JOBS=1 and writes a build log/result.
It does not import the produced extension or edit installed files. It copies the
exact original runtime include tree, replacing only the candidate header and
new binary, then writes a new manifest and generated wrapper. CUDA JIT remains
part of root's later GPU harness.

## Root-operated component gate

The harness refuses an active dsv41-rank unit or any existing GPU compute process.
It checks installed MoE/util source hashes and the installed native binary hash.
It imports the installed current MoE for all three arms:

1. Installed original native control.
2. New native binary with the new flag false, isolating rebuild behavior.
3. New native binary with D05 enabled via its private wrapper.

Example after root reserves an idle GPU window:

    ~/venvs/vllm032/bin/python gpu_exactness.py \
      --candidate ~/dsv41-goal120-20261024/deadstores/build-v1 \
      --tokens 1,2,4,6,8,12,24,48,64,96,97 \
      --checkpoint-layer 0 --tp-rank 3 --timing > gpu-layer0-r3.jsonl

Without checkpoint-layer the harness uses synthetic packed weights. A complete
gate repeats real layers/ranks, including tail-heavy cases; one rank/layer is
not all-rank qualification. The loader reads expert tensors only and changes
no checkpoint file or Engram mapping.

Direct FC2 tests compare live outputs bitwise and require skipped eight-row
output groups to retain alternating byte sentinels. They cover every important
occupancy boundary0/1/7/8/9/15/16/17/23/24/25/31/32/33/63/64, multiple experts,
all-invalid rows and repeatedly changing captured layouts. Dead FP8 inputs and
scales are poisoned with NaNs. The installed full MoE then runs valid, skewed,
half-dead, random-hole and all-dead routes with poisoned workspaces, eager and
two changing-route graph-replay cycles, plus the M97 fallback.

Optional timing reports warm repeated-layer latency and FC2-only latency on
those actual MoE-produced quantized inputs. Routes are synthetic, even with
real weights. These are screening results, not DRAM-paid or serving results.
Retain the original D05 ≥10%-lower-FC2-time C4 gate; actual route histograms,
cache-rotating weights, sanitizer memcheck/racecheck/synccheck, resource/SASS
inspection, matched pinned-AV serving tests and 1M behavior remain separate
root-controlled gates. Kill on any live mismatch, touched skipped group,
fallback regression, CUDA error, sanitizer issue or unresolved timing result.

## Rollback

No installed state has been changed by preparation or build. For a later
root-controlled trial, DSV41_FC2_LIVE_STORE=0 selects the original module/cache;
the original wrapper is saved as wrapper-baseline.py. The original native tree
and an immutable native-baseline copy are preserved. Never combine this patch
with D04 or another native change before its independent qualification.
