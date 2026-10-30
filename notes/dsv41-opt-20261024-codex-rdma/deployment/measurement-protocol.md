# October 24 recovery and RDMA qualification

User objective: fast responses for 1–4 active users on the existing eight-Spark DeepSeek V4.1 deployment.

## Recovered baseline

- Model: `/home/jeweled/models/dsv41/uncensored-fp8`, served as `dsv41`, TP8, fixed rank mapping one..eight = 0..7.
- Context limit: 1,048,576; preserve current short-request scheduler and all previously promoted optimizations.
- Launcher SHA256: `1a2f26f46d30924f110ef7c2b0a77296646e4dd6ded4b0f99cc3f122b5dfcf37` on all eight hosts.
- Legacy RDMA binary: `dfeeab009e3a03018ddc48157401c668d08dc68c02ef7d8380322afc67960885`.
- Inactive candidate binary, already installed by Claude: `62daad6ccc71026b26cb73e28c2e11bb46b017e86af7b0a0e811f94a838b51f1`.
- Selector binding: `acbeeef765eabef9656a4cc23f81a1db3fe777136be29cf21fb5d4c094e0f00b`.

## Evidence and interruption

Claude session `64b45536-e9c4-4d44-9db6-63d2a8374326` and the saved remote deployment README identify k1/2-silu-liverows as the last fully qualified promotion. k1/3-rdma-loads i1-C and i2-A completed, but the i3-C transition stopped the services at 16:25 UTC and never started replacements. Fresh reads found no running experiment driver, no GPU processes and inactive units on all eight nodes. This establishes an interrupted transition; it does not establish why the controller ended.

Historical evidence is copied under `claude-evidence/` with SHA256 identities. These measurements are not relabeled as new runs.

## Controlled comparison

The current control is measured after recovery, followed by the candidate with exactly the control's adaptive-verification cost profile. That profile is archived in `av-pin-c0.json`, SHA256 `272cdc200b4f1d4742c392541134bba0f0a6452535f43c71a20b3a042f18bde4`, and copied to all eight hosts. An additional reversed pair is warranted if the observed gain is marginal or noisy.

Each new arm runs the same suite, in the same order:

1. One short sampled warmup, excluded.
2. Eight coding and eight prose C1 trials, identical prompt tags across new arms. These have thinking disabled; their implied step time is not the same metric as the long thinking-enabled benchmark below.
3. Thinking-enabled greedy single-stream code/prose/math, 1,024 output tokens, two repetitions each.
4. Two- and four-user bursts, 512 output tokens each, three repetitions for greedy and sampled (temperature 1, top-p .95, fixed seeds).
5. Sampled single-stream code/prose/math, 512 tokens, two repetitions.
6. One burst each at 8, 16 and 32 users as a regression check, not a precision throughput estimate.
7. Six fixed greedy exact-output prompts, twice, and GSM8K-50 at concurrency eight.
8. Additional matched coding/prose at C2 and C4, six trials of each with identical prompt tags across arms. Added after baseline burst measurements showed substantial scheduling variation; run after the same base suite on both arms.

Record every request's time to first output, total decode rate, and streamed chunk gap. Chunk gaps are not token gaps: speculative decoding returns multiple tokens in a chunk. These are server-local measurements, excluding a user's network path. A 512/1,024-token benchmark ending at its requested length is expected, not a completed-answer quality test.

## Decision

Prefer repeatable lower step time and better 1–4-user latency over a single peak throughput number. Require matching exact-output checks and greedy acceptance counters with the common adaptive profile, no failed requests, and no material regression in the small-concurrency measurements. Treat gains of only a few percent as provisional until replicated. Before leaving a candidate as the default, require GSM8K, long-context needle/copy checks, an unpinned default restart, all-rank kernel confirmation and fresh generation.

No checkpoint tensor is modified. Never bypass the Engram row checks, change the TP size or rank mapping, or copy punched shards from another rank. Spark-four remains the pristine restore source. The old October 21 M64 recovery bundle is historical and is not used for this recovery.
