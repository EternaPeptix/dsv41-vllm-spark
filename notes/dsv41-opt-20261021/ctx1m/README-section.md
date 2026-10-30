
## Context 1M (2026-10-23 05:50-07:05 UTC). Live: max_model_len 1,048,576

**What changed:** one launcher line on all 8 ranks: `--max-model-len 262144` became `--max-model-len ${DSV41_MAX_MODEL_LEN:-1048576}`. Nothing else changed: KV bytes are still 16G/rank, max-num-batched-tokens is still 16384, max-num-seqs is still 32, and no site files were touched. Launcher sha256 is **85189840** on 8/8. The backup `~/dsv41-tp8.sh.bak-1m` is 5ccc4b79 (the round-4 launcher) on 8/8. No mitigations were needed. Raw results are in ./ctx1m/ (lc500k, lc1m, conc1m, regress/, ab262k/, final-instance/, mem/ = 5 s MemAvailable logs for all 8 ranks; scripts longctx_ttft.py (longctx.py + streaming TTFT), conc1m.py, regress.sh, run.sh).

**KV pool:** still 16 GiB/rank (`--kv-cache-memory-bytes 16G`). vLLM now reports **6,390,007 tokens, max concurrency 6.09x at 1,048,576** (at 262144 it reported 3,257,974 tokens, 12.43x). The bytes are the same. The token figure changes because the fixed per-request sliding-window/compressed-group blocks are spread over a longer max_model_len. **6 full-length 1M requests fit at once.** /v1/models shows max_model_len 1048576.

**Long-context validation** (thinking off, greedy, 7 depths 1/10/25/50/75/90/99 %, 2 seeds per length):

| length | prompt tok | needles | copy | TTFT | prefill tok/s |
|---|---|---|---|---|---|
| 500K | 497,263 / 500,186 | 14/14 | 2/2 exact (sim 1.000) | 90.7 / 90.4 s | 5,485 / 5,534 |
| 1M | 999,411 / 999,021 | 14/14 | 2/2 exact (sim 1.000) | 209.8 / 209.4 s | 4,763 / 4,770 |
| 1M + 4 short chats | 998,495 | 7/7 | - | 209.2 s | 4,772 |

The copy question reuses the prefix cache, which takes 1.8-3.1 s.

**1M + 4 concurrent short chats:** all 4 completed (256 tokens, finish=length). **However, they started 20 s into the 1M prefill and did not get a first token until the prefill finished (TTFT 189 s each, e2e 191-194 s).** The scheduler gives the running long prefill the whole 16384-token budget every step, so new requests wait in the queue. This behaviour existed before (a 200K prompt blocks new requests for about 35 s) but is much longer at 1M. If it matters, the candidate fix is `--long-prefill-token-threshold` (supported in this vLLM; for example 8192 leaves half of each step for new requests). It is untested, and it will slow long prefills.

**Memory** (5 s samples on all 8 ranks): after startup and idle, MemAvailable is about 18 GB per rank. The first long prefill allocates workspace, which settles at about 12-13 GB and then stays flat (the KV pool is preallocated). Minimum during the tests: spark-one 11.5, two 12.9, three 13.0, four 12.8, five 12.8, six 12.7, seven 12.9, eight 13.3 GB. This is above the 4 GB gate everywhere and matches round 4 (11.2 GB low). There were no new kernel NVRM/Xid/OOM messages during the tests. The one `NV_ERR_NO_MEMORY` mem_desc.c message per launch appears at every startup, round 4 included. spark-four's kernel journal cannot be read without sudo. The rank logs have no errors.

**Regression** (same day, 1M config vs round-4 reference):

| metric | round-4 ref | 1M | A/B same day at 262144 (`--max-model-len 262144` extra) |
|---|---|---|---|
| single code/prose/math | 86.3/67.7/117.8 | 83.5/67.4/123.6 | - |
| single mean | 90.6 | **91.5** | - |
| conc8 agg | 221.9 | 212.7 / 215.8 / 218.7 (mean 215.7, -2.8%) | 220.5 / 222.9 / 222.6 |
| conc16 agg | 330.9 | 311.8 (-5.8%) | 330.7 |
| conc32 agg | 479.0 | 456.9 / 457.1 / 458.7, final instance 448.7 (-4.5..-6%) | 476.8 / 475.7 / 482.5 |
| GSM8K-50 conc8 | 49/50 | 49/50 (231.8 tok/s) | - |
| longctx 4K-200K | 60/60, copy 1.000 | 60/60, copy 1.000, PASS | - |
| exact.py | a8c3394af9d2 | 7f54b0fea5ae (cross-restart hash; not comparable, see caveats) | - |

Accept lengths match (conc32 AL 2.31-2.34 at both lengths). **Single and correctness are unchanged. Concurrent throughput is about 3% lower at conc8 and 4.5-6% lower at conc16/32. The same-day A/B relaunch at 262144 restored the round-4 numbers, so the slowdown comes from max_model_len itself.** The likely cause is per-step cost that scales with block-table width: the block table is [max_num_seqs, max_model_len/block] for each of 17 KV groups (model_runner.py ~L621), and all_token_ids is [32, max_model_len]. Not profiled.

**Rollback** (from the Mac, which reaches all 8 ranks):
- Env/arg only, keeps the launcher: `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh' | bash -s -- --max-model-len 262144` (or 524288 as a middle ground; untested).
- Launcher: `for h in one two three four five six seven eight; do ssh spark-$h 'cp ~/dsv41-tp8.sh.bak-1m ~/dsv41-tp8.sh'; done; ssh spark-one 'cat ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh' | bash`, then check that sha256 is 5ccc4b79 on 8/8.
- Ops note: the first relaunch hung at the first DSpark graph capture (the known rdma_ar transient, all ranks stuck at the `rdma_ar audit ... (160, 5120)` line for more than 6 min). Stopping all ranks and relaunching fixed it. The next 3 launches were clean (about 6.5-7 min each).
