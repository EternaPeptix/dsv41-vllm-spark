# DeepSeek V4.1 on eight DGX Sparks — October 24

**Verified at 17:35:49 UTC (10:35:49 AM Phoenix): the server is up on all eight Sparks with the new default. The API and fresh generation pass; the timing pin is absent.**

## Best qualified starting point from Claude's logs

The latest fully validated fast bundle was **k1/2-silu-liverows**, on top of k1/1's length-aware indexer, the round-4 optimizations, the short-request scheduler, and **1,048,576-token context**. Its launcher was `1a2f26f4`. The checkpoint remains `/home/jeweled/models/dsv41/uncensored-fp8`, served as `dsv41`, with TP8 and the existing fixed node/rank mapping.

Claude's final qualified run at 15:09 UTC recorded 86.1 / 69.1 / 118.0 tokens/s for its thinking-enabled code / prose / math prompts, exact-output hash `a8c3394af9d2` twice, GSM8K 50/50, long-context retrieval 60/60, and seven of seven needles in a 999,411-token prompt. These are historical measurements, not new runs made by this pass. Different prompts and thinking settings produce materially different speeds; there is no workload-independent fastest checkpoint number.

The older October 21 M64 deployment artifacts in this workspace were superseded by Claude's later work and were not restored.

## Recovery and change

Fresh inspection found all eight hosts reachable, all `dsv41-rank` units stopped, no GPU compute processes, and no API listener. Claude's logs show a completed RDMA candidate run followed by the start of the next control restart at 16:25 UTC. The services stopped, but replacements were never launched. No experiment driver was still running. This establishes an interrupted benchmark transition; the reason the controller ended is unknown.

The qualified bundle was restarted and verified with generation. The interrupted **k1/3-rdma-loads** experiment was then continued in separate artifact directories.

The change enables Claude's already-built parallel-load RDMA kernel. It issues the eight peer loads before consuming their values, retaining the same ordered FP32 accumulation. It targets communication latency. The existing model weights, quantization settings, scheduler, context limit and rank assignment are preserved. The source and assembly checks, plus numerical component tests, came from Claude's prior work; this pass verified the installed identities and performed the additional serving comparison and deployment.

Historical component evidence for the exact installed binary includes **6,384 checks per rank, zero failures on all eight ranks**, plus multi-node collective and graph-replay checks with zero recorded failures. These tests cover the tested cases, not every possible execution.

## Fresh controlled comparison

Control `c0-recovered` and candidate `a1-pinned` used identical effective adaptive-verification cost arrays, confirmed from the live logs. Coding and prose prompt tags were identical across these new arms. Both produced exact-output hash `a8c3394af9d2` twice, and their thinking-enabled greedy single-stream acceptance counters matched exactly. All measured requests completed without request errors.

| Single-user metric | Control | Candidate | Change |
|---|---:|---:|---:|
| Matched coding, implied ms/step, 8 trials | 33.804 | 33.449 | 1.05% lower |
| Matched prose, implied ms/step, 8 trials | 29.040 | 28.591 | 1.55% lower |
| Thinking-enabled code, tokens/s, 2 repeats | 86.38 | 87.02 | +0.74% |
| Thinking-enabled prose, tokens/s, 2 repeats | 68.39 | 68.72 | +0.48% |
| Thinking-enabled math, tokens/s, 2 repeats | 118.39 | 119.66 | +1.08% |

All 16 matched single-user trials had lower implied step times, with identical acceptance counters. Claude's earlier unpinned pair showed the same direction (coding 33.922 → 33.547 ms; prose 29.233 → 28.685 ms). That earlier pair had different startup calibration, so its larger throughput gains are not used as the controlled estimate.

The matched prompts have thinking disabled; the long single-stream prompts have thinking enabled. Their rates are different measurements and should not be combined into one speed number.

| Mixed bursts, 3 repeats per case | Aggregate tokens/s, control → candidate | Mean per-request decode tokens/s, control → candidate |
|---|---:|---:|
| 2 users, greedy | 103.23 → 101.80 | 64.90 → 65.63 |
| 4 users, greedy | 153.57 → 155.24 | 50.80 → 50.67 |
| 2 users, sampled | 98.33 → 101.49 | 58.90 → 62.65 |
| 4 users, sampled | 149.03 → 150.21 | 49.48 → 48.56 |

Median first-streamed-token times were about 0.35 seconds at two users and 0.71–0.73 seconds at four users. The largest control/candidate difference in those medians was 17.4 ms. Chunk-gap results were also mixed at four users. These bursts vary with scheduling and generated text; they do not establish a clear multi-user gain or a statistically resolved regression.

Additional matched C2/C4 trials favored the candidate in all four mean step-time comparisons. The apparent 7% C4 coding gain is particularly noisy (control trial standard deviation 4.62 ms) and is not treated as a demonstrated kernel gain. The strongest conclusion is a **modest, approximately 1% single-user improvement**, with multi-user performance broadly similar in this sample.

GSM8K-50 was 49/50 for the control (known variable item 12) and 50/50 for the candidate, with no truncations. That one-answer difference is not claimed as an accuracy improvement.

## Final default verification

- All eight ranks active with the promoted kernel; zero service restarts and no ERROR/Traceback lines in current rank logs.
- API health HTTP 200; `dsv41` reports context limit 1,048,576. A fresh request after the long-context test returned `391` for 17 × 23. No active or queued requests remained.
- Default-startup single-stream decode: **87.39 / 70.13 / 120.57 tokens/s** for code / prose / math (two repetitions each). One four-user burst: **151.4 aggregate tokens/s**, **50.33 mean per-request tokens/s**. These are end-state observations; the fixed-calibration comparison above is the causal estimate.
- GSM8K: **50/50**, zero truncations.
- Long-context suite: **60/60** facts retrieved, all 12 copied passages exact, through about 200K tokens.
- Near-1M probe: **999,411 prompt tokens**, **7/7** facts retrieved, exact copying. First token took **211.81 seconds**. This is a functionality check, not evidence of a prefill speedup.
- Final exact-output hash: `9330f03c4f97f247d0c4a9c7d83934b6cfea2b95c0532fbf2f8a46cb1d2f6371`, identical in two runs. This differs from the pinned comparison but exactly matches three saved Claude `A8-262k-canon-nopc` runs documented on October 23, before the RDMA candidate. Exact equivalence between the kernels was demonstrated with common calibration; cross-restart equivalence under arbitrary fresh calibration is not claimed.
- Lowest end-state MemAvailable: approximately **9.4 GiB**, on spark-one.

The first near-1M test invocation used unsupported shorthand `1m` and exited before making a request. It was corrected to `1000k`; only that probe was rerun successfully. The original failed log and exit code are preserved separately from the successful retry.

## Deployment identity and rollback

- New launcher SHA256: `2027f4d8548856bed458ab7c3cf0a73170e931f51b69692d28d8aa5c81c98fb4`.
- Previous launcher / backup SHA256: `1a2f26f46d30924f110ef7c2b0a77296646e4dd6ded4b0f99cc3f122b5dfcf37`.
- Backup on each Spark: `~/dsv41-tp8.sh.bak-k1-3-rdma-loads`.
- Candidate binary: `~/rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so`, SHA256 `62daad6ccc71026b26cb73e28c2e11bb46b017e86af7b0a0e811f94a838b51f1`.
- Legacy binary: `~/rdma_ar_dsv41/build/rdma_ar_ext.so`, SHA256 `dfeeab009e3a03018ddc48157401c668d08dc68c02ef7d8380322afc67960885`.
- RDMA selector binding SHA256: `acbeeef765eabef9656a4cc23f81a1db3fe777136be29cf21fb5d4c094e0f00b`.

The one-line launcher change defaults `DSV41_RDMA_PARLOAD` to `1`. The old binary remains installed. An explicit `DSV41_RDMA_PARLOAD=0` restart selects the legacy path. For a permanent rollback, restore only the verified launcher backup on all eight hosts and restart with the existing guarded launch script. Do not use the older M64 restore bundle for this rollback.

Seven hosts have rank-specific Engram shard holes; spark-four is the pristine restore source. Preserve the existing Engram checks, TP8 and rank mapping. No checkpoint tensor was edited in this pass.

## Scope and provenance

Timings are measured at the server, excluding a user's network path. Decode rates include reasoning tokens. First-streamed-token time can be the first reasoning token, not the first answer-content token or time to a completed answer. Streamed chunks can contain several tokens under speculative decoding. The 512/1,024-token timing prompts intentionally reach their requested token limits; exact-output and arithmetic tests are separate correctness probes.

The comparison is a limited serving sample, not proof of full memory-bandwidth utilization. No updated Nsight trace or bandwidth-saturation measurement was made. The existing profile suggests prefetch/collective overlap is a useful next target; it should be remeasured on the new default before another kernel change.

Raw new runs: `~/dsv41-opt-20261024-codex-rdma/` on spark-one, with local copies in `work/claude-handoff-20261024/instances/`. Historical source: Claude session `64b45536-e9c4-4d44-9db6-63d2a8374326`, its `k1/3-rdma-loads` scratchpad, and the saved `~/dsv41-opt-20261021/README.md`. Copies of the historical raw runs and component evidence are retained under `work/claude-handoff-20261024/claude-evidence/` with SHA256 identities.

See `benchmark-summary.json`, `benchmark-comparison.csv`, `measurement-protocol.md`, and `effective-profile-match.json` in this directory for the recorded measurements and methods.
