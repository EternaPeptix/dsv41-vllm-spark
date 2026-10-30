# D05: skip fully dead FC2 output-store groups

October 24, 2026. **Investigation and prototype specification only.** No native
candidate has been built, no GPU experiment has been run for D05, and no remote
file has been changed. This directory also contains a separate drafter follow-up.

The strongest defensible conclusion is that D05 has a narrow exact implementation
path, but its current serving gain is unknown and is unlikely to close the
120-token/s gap alone. It removes global writes of padding; it does not remove
MMA, activation loads, weight loads, shared-memory staging or most synchronization.

## Source identity and live-row contract

The source snapshots in src/ were read from spark-one. Full SHA256 values and
remote paths are in [source-manifest.json](source-manifest.json). This is one-rank
source evidence, not an assertion that a new candidate is qualified on all ranks.

1. In [deep_gemm_utils.py:114](src/deep_gemm_utils.py#L114), scatter phase 1
   rounds each expert's count to the current alignment and assigns its segment.
   The normal contiguous layout writes an expert ID only for real rows.
   [Lines 467–577](src/deep_gemm_utils.py#L467) initialize the layout to -1.
2. [Scatter phase 2, lines 163–275](src/deep_gemm_utils.py#L163) allocates only
   valid routes with an atomic counter. Therefore each expert segment has a
   contiguous live prefix followed by -1 padding. With alignment 32, every tile
   belongs to one expert and only its final tile can contain a partial prefix.
3. [Gather, lines 363–425](src/deep_gemm_utils.py#L363) reads the inverse
   permutation and output only for expert_id >= 0. Padding output is unobserved
   by this private MoE path. Keep gather's top-k accumulation order unchanged.
4. [sm121_target_m32.py:33](src/sm121_target_m32.py#L33) deliberately invokes
   the private native call with ensure_zero_padding=False. This is not permission
   to change the public grouped-GEMM output contract or a psum layout.
5. [deep_gemm_moe.py:690](src/deep_gemm_moe.py#L690) uses the same grouped
   layout for FC1 and FC2, then gathers FC2. Current target FC2 has 384 experts,
   N=5120, logical K=384, top-k=6. Its input padding can contain stale NaN/Inf
   because live-row activation no longer clears the full workspace. Do not infer
   zero padding, sanitize dead rows or feed them into a live reduction.

Under these conditions an eight-row group is entirely dead iff its first row is
-1. A partially live group must still be written in full. A malformed arbitrary
grouped layout with holes does not meet this contract and must never select this
candidate. No rank, checkpoint or Engram change is part of D05.

## Small isolated prototype

Select only target FC2 with normal MGroupedContiguous, E=384, N=5120,
logical K=384, BF16 output, BM32/BN64/BK128, output swizzle 128, no accumulation,
no split-K, contiguous output and the private no-zero-padding contract. Keep
FC1, larger-M/align128, psum layouts, public APIs and every fallback unchanged.
Use a dedicated candidate native module/cache and an explicit FC2 selector.
The current manifest-checked adapter cannot be silently edited in place.

The current [TMA epilogue, lines 1386–1481](src/sm120_fp8_fp4_gemm_1d1d.cuh#L1386)
stages 32 rows and emits one 32-row TMA store per 64-column tile. The proposal is:

1. Keep kEpiSubM=32 and all 32 rows of shared-memory staging, conversions and
   arithmetic unchanged. **Simply setting kEpiSubM=8 is unsafe:** the staging
   body also writes sub_row0+8, outside an eight-row allocation.
2. Introduce a distinct store-descriptor height of 8, separate from the staging
   height. The historical host implementation creates the descriptor at
   [lines 516–523](src/historical-sm120_fp8_fp4_gemm_1d1d.hpp#L516). Its original
   build provenance for the live native library is not yet verified; recover and
   hash that build tree before implementation. A device-header-only edit is
   insufficient because the descriptor must match the emitted store geometry.
3. After the unchanged store fence and named barrier, the existing elected
   lane checks g=0,8,16,24. When m_base+g is in range and its layout value is
   nonnegative, issue the 8-row TMA store from smem_src+g*128 to row m_base+g.
   Keep the descriptor outside graph capture; it is immutable for this geometry.
4. Keep the existing arrive, pre-reuse wait, named barriers and final drain.
   No synchronization can become conditional on a per-thread live predicate.
   The elected lane may issue 1–4 stores per tile; every math thread still stages
   the same bytes and participates in the same barriers.
5. Add an explicit compile-time candidate bool to generated code and cache key.
   Add the independent descriptor height only to this qualified branch. Preserve
   both the old staging-memory size calculation and every other template path.

At 128-byte swizzle the row phase repeats every eight rows, and each subview's
base is shifted by 1024 bytes. [check_layout.py](check_layout.py) exhausts the
32-row address mapping and verifies the prefix predicate on 2,130 layouts,
including trailing all-invalid reservation tiles.
This is a CPU model, not proof of TMA legality, CUDA race freedom or bitwise
numerical behavior. SM121 execution and sanitizer checks remain mandatory.

The existing [non-TMA epilogue, lines 1492–1552](src/sm120_fp8_fp4_gemm_1d1d.cuh#L1492)
already masks individual invalid rows. It is a useful separate timing control,
but changing to it also changes the store mechanism and is not evidence for
the performance of the proposed eight-row TMA path.

## Gain bounds and stop conditions

For each active BM32 tile across N=5120, the old BF16 output is 327,680 bytes.
The candidate writes 81,920, 163,840, 245,760 or 327,680 bytes for 1–8, 9–16,
17–24 or 25–32 live rows respectively. Thus the largest reduction is 75% of
these output bytes. Four stores instead of one can lose when all groups are
live; the first C1/C2/C4 route histogram determines whether this is worthwhile.

The historical model estimated junk output at about 23% of align32 FC2 traffic.
If that fraction still applied, and every removed byte were paid on the critical
DRAM path, even the favorable 75% removal gives only a 17.25% total-byte
reduction: an ideal 1.208x FC2 bandwidth speedup before added issue overhead.
This is a conditional byte ceiling, not a measured kernel or serving gain.
The previous observed C1 FC2/FC1 ratio was already consistent with weight
streaming, suggesting the extra writes were largely absorbed by cache.

The historical 0–0.3 ms C1 and 0.3–0.9 ms C4 projections remain hypotheses;
they predate current live activation, indexer and RDMA improvements. Do not
promote their upper ends to forecasts or add D05 to other memory-traffic gains.
Source: [kernel-ideas.md, lines 269–318](../../../outputs/experiment-roadmap-20261024/sources/claude/kernel-ideas.md#L269).
Fresh route occupancy, DRAM/L2 writes, FC2 duration and decode critical-span
measurements are missing. Kill the experiment if component results show no
resolved C1/C2/C4 saving or if store issue cost consumes the byte saving.

## Root-controlled validation sequence

1. Recover the actual native host build source, prove baseline rebuild identity
   behavior, and pre-JIT the candidate in a new cache. Record flags, hashes,
   ptxas resources and shared-memory requirements. Never alter the serving cache.
2. Compare live FC2 outputs bitwise with the installed native binary on actual
   processed weights/routes and all live-prefix lengths 0–64, including multi-
   tile experts, all-invalid and partially invalid routes. Compare full gathered
   MoE outputs too. Poison unused activation/scales and destination rows with
   NaN/Inf and changing sentinels. Require untouched skipped groups and no live
   row lost. Whole padding tensors need not match the baseline.
3. Run eager, graph capture and mutated replay, sanitizer race/sync/memory
   checks. Exercise excluded dispatches and ensure their original paths remain.
4. Pair baseline/candidate ABBA under warm and cache-rotating weights for actual
   C1/C2/C4 route mixtures. Report FC2 time, layer time, bytes and variance.
   Retain D05's at-least-10%-lower-FC2-time gate at representative C4 row buckets,
   then require a useful serving gain against root's fresh profile. Preserve a
   full-store candidate control if isolating store
   grouping overhead is necessary.
5. Only a useful exact component result advances to all-rank pinned-AV serving
   tests and unchanged 1M behavior. Root alone schedules/executes these.

Rollback is the existing native adapter/cache and disabled FC2 selector. The
baseline source copies and SHA manifest remain immutable evidence. There is no
candidate or installation to roll back from this investigation itself.
