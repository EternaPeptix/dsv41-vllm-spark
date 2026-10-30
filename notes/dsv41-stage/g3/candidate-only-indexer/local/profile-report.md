# Profile targets: evidence and exactness boundary

The requested first target is **Fused CANON**, the int32 multiset canonicalization used after indexer top-k, as confirmed by the parent. It is separate from streaming FP8 MQA-logits/top-512 fusion. The second target is candidate-only scoring in the four late consumer indexers, layers 24/28/32/36. The saved profile supports investigating that work, but the quoted ~87 ms includes the layer-20 producer; the four consumers account for **72.1617 ms** of logits/top-k/mask or **77.4139 ms** including their flags and K gather in the cited late chunk. No candidate speedup has been measured by this investigation.

All evidence below was copied from existing local files. No cluster call, GPU import, compiler, benchmark, inference or service change occurred. `source-manifest.json` records 15 full SHA-256 identities and original paths; `check_profile.py` rechecks them and recomputes `profile-results.json` with ordinary Python.

## Authoritative saved pointers

- `claude/codex-research-review.md:15`: adoption item 1 names R5 patch `7fc2d1ab` / `DSV41_TOPK_CANON_FUSED`, int32 multiset sort. It estimates 0.12–0.15 ms/step at C1 and requests real GPU bitwise tests for rows 1–192, 2K and 16K, signed/sentinel extremes, strided views and replay. This is the harness target; its gain remains an estimate until measured.
- `claude/codex-research-review.md:56`: the later prefill-specific lead calls consumers 24–36 ~87 ms/chunk at 942K and projects ~2 seconds per 1M prompt, with tie-break and exact-E7 overlap checks still required.
- `claude/next-experiments.md:319`: item 8A is the original compact-scratch candidate scoring proposal; 8B is a separate, much larger streaming score/top-k kernel.
- `claude/kernel-ideas.md:533`: the earlier compact-scratch proposal is explicitly **decode**, originally supported by a byte model. It calls for ascending candidate K order, the same SM120 score kernel, compact selection and global-index remapping. Its exactness is explicitly conditional on per-position reduction invariance under changed tiling/width. It is not evidence of a successful prefill implementation.
- `claude/pd/xray.md:53`: historical good-launch B profile, reported chunk labels 2/12/35/58 and context labels 24K/188K/565K/942K. It attributes 99 ms of late-chunk work to the whole tail indexer, including layer 20.
- `claude/pd/xray/an/r0.4.json`, `idx_kernels`: the actual exported event records supporting the calculation below; `claude/pd/xray/an/idx_by_seg.txt` is the human-readable aggregation. These are exported analysis records, not a new trace or a current process observation.

## Corrected attribution and representative shapes

The tail segment contains five ordered gather/logits/top-k sequences. The first includes candidate publication and no candidate mask; the remaining four include a mask. Captured source/config identify the first with source 20 and the four consumers with 24/28/32/36. These layer names are source-order attribution, not explicit per-kernel layer annotations in the trace.

| Reported context label | Consumer logits | Consumer top-k | Consumer mask | Sum of those three | Including consumer flags + gather |
|---|---:|---:|---:|---:|---:|
| 24K | 1.4963 ms | 0.1492 ms | 0.2615 ms | 1.9070 ms | 2.1351 ms |
| 188K | 10.1027 ms | 3.8256 ms | 2.2871 ms | 16.2154 ms | 17.3892 ms |
| 565K | 27.9153 ms | 10.5135 ms | 6.4888 ms | 44.9176 ms | 48.3467 ms |
| 942K | 44.6160 ms | 16.9348 ms | 10.6109 ms | 72.1617 ms | 77.4139 ms |

The published ~87 ms is reproduced by 55.5738 ms of **five** logits calls + 21.0900 ms of **five** top-k calls + 10.6109 ms of four masks = 87.2747 ms. Source-20 logits/top-k contribute 15.1130 ms and must remain. The checker explicitly rejects attributing those source-20 calls to the four consumer savings.

Every tail top-k grid is `[128,1,1]`, not 16K query rows. The mask grids are `[128,32/208/576/944,1]`; their Y extent reflects the full-width walk. The historical request has 1,024,391 prompt tokens, nominal 16,384-token chunks, and 63 expected chunks (the saved chunk analyzer reports 64 seen). Its c58 timeline row spans context 933,888–950,272, midpoint 942,080. Do not treat that timeline label as an exact per-kernel width: the exported masks and K-gather grids have their own extent, and the cached/padded kernel input dimensions must be recorded by any real harness.

Saved model configuration (`claude/pd/xray/config.txt:49–58`) specifies index heads 32, head dim 128, top-k 512, candidate source 20, 2,048 blocks of eight positions, and index sources `[2,8,14,20,24,28,32,36]`. A useful candidate-only test panel therefore includes 128-row tails, context widths near 32K/208K/576K/944K/1M, ragged causal starts/ends and partial candidate blocks. Include small/mixed rows as fallback/admission tests. The 2,048 candidate blocks represent at most 16,384 unique positions; padding or duplicate candidates must not be counted as extra positions.

The ~2-second whole-prompt figure is the saved author's projection, not an A/B measurement or a sum independently established here. None of the table is automatically removable: compact gather, score, selection, remap and sorting still cost time. GPU stage sums also do not establish critical-path savings across ranks or account for changed allocation/launch overhead. The profile's separate rank-5 paged-KV straggler is not addressed by this proposal.

## Source-visible flow and exactness gates

Pinned `source/sparse_attn_indexer.py` SHA `7112c01d…` retains prefill dense `fp8_fp4_mqa_logits(...clean_logits=False)` at lines 697–705, then candidate source publication or consumer mask at 708–729, then native `top_k_per_row_prefill` at 730–738. This confirms the source-visible opportunity. It does not attest the currently running process, effective feature flags, graph selection or a live candidate hit. Current enabled status is **unknown**; the parent deliberately avoids further live reads during others' profiling windows.

A compact-scoring candidate needs these specific checks:

1. Keep layer-20 scoring/selection and all existing per-chunk index-K/SWA cache writes. Candidate-only computation should not silently become E7 tail omission.
2. Preserve selected-candidate **set** semantics when block IDs repeat or contain `-1`, and the exact per-row causal start/end mask. Compact duplicates must not displace distinct selected tokens.
3. Establish bitwise score equality at candidate positions under the same FP8 format/scales and reduction ordering. Calling the same named kernel at a different width/layout does not prove identical lowering or accumulation order.
4. Preserve native top-512 threshold/tie behavior after global-index remapping, including signed zeros, NaNs where allowed, `-inf`, fewer than 512 valid positions and masked padding. Sorted final indices cannot repair a different selected set.
5. Keep output/scratch allocation, stream/event ownership, graph replay and request/chunk lifetime valid; any new scratch must not overwrite input cache or concurrent readers.

E7 is a concrete hazard, not merely hypothetical: `claude/pf/readme_pf_final.md:51–68` records apparent ~11.8-second 1M improvement but deterministic E7X/E7XP comparison failure. The cause was omission of persistent K-cache writes for replay rows in layers 20/24/28/32/36. `claude/pf/STATE.final.md:23–25` records the same rejection. Its timing must not be added to an exact candidate-only estimate.

## Harness review scope

The separate Fused CANON GPU harness will be reviewed for direct importability, dependency failures, pinned baseline/candidate source equivalence, actual GPU rather than interpreter execution, exact byte comparisons, canaries/input immutability, representative strided/offset views, real graph replay and nonzero failure exits. It must not install site files, change services or issue model requests. A harness that is ready to run is not a passing GPU result.

One saved concrete pitfall is `claude/pf/readme_pf_final.md:49`: a prior GPU test imported an interpreter test that set `TRITON_INTERPRET=1`, generating false mismatches. Reject interpreter mode before GPU imports and avoid test-module imports that change its environment. This evidence collection runs neither harness nor GPU.
