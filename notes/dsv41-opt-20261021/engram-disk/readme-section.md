
## Engram on disk: A/B measurement (2026-10-23). Measured only; NOT enabled

Question: how much RAM would we get back by serving the Engram tables (layers 1/14, about 23.6 GiB per rank, mlocked THP) from NVMe, and what would it cost? The patch is behind `DSV41_ENGRAM_DISK=1`. It reads each rank's own rows with O_DIRECT from the checkpoint shards, so it works on punched nodes. Raw data is in `engram-disk/<arm>/`, offline tests in `engram-disk/offline/`, the NVMe microbench in `engram-disk/nvme/`, and the tables in `engram-disk/tables.md` and `breakdown.txt`.
**State after the run:** the patch was uninstalled on 8/8 (engram.py aa8aad1b, model_state.py 53e4fe39, engram_disk.py absent). Live 1M was relaunched with launch-all (the punch check passed). Launcher 0f1ab42f was unchanged throughout. After the restore: /health 200, GSM8K-50 50/50, exact 7e7937d4a011 (1M is non-deterministic, so this is informational).

**Exactness (passed):**
- GPU staged lookup == resident lookup, bitwise (bf16 and packed, ranks 0/3/7).
- O_DIRECT == buffered on real rows on 8/8, using exact per-rank ranges.
- Verify arm (disk4pf, every stage double-read, deterministic config VERIFY_CAP=0 ROUTE_LIVE=0 MAX_MODEL_LEN=262144): exact.py **a8c3394af9d2 2/2**, the same as the reference. verify_failures=0 on 8/8, verify_reads 1623 and verify_fwd 56 on every rank. GSM8K 50/50.

**Arms** ran on the live 1M config. Comparisons are against the mean of control and control2. Single t1 restart spread is about 1 %.

| arm | RAM freed/rank (min) | single t1 | conc8 t1 | conc32 t1 | added ms/step single / conc32 (max rank) | TTFT 128K real |
|---|---|---|---|---|---|---|
| control / control2 | 0 | 85.6 / 84.9 | 206.6 / 199.3 | 426.3 / 421.6 | – | 22.7 / 22.8 s |
| disk0 | +22.4 GiB | −6.0 % | −2.6 % | −3.9 % | 1.44 / 4.24 | +9.3 % |
| disk0pf | +22.4 | −4.0 % | −3.0 % | −2.5 % | 1.44 / 3.39 | −0.9 % |
| disk4 | +18.6 | −3.2 % | −1.9 % | −4.3 % | 1.44 / 3.55 | +5.6 % |
| disk4pf | +18.7 | −6.3 % | −0.1 % | −4.6 % | 1.46 / 4.17 | −0.6 % |
| disk8 | +14.4 | −2.1 % | −1.7 % | −3.7 % | 1.41 / 3.04 | +5.5 % |
| disk4pf, 127 threads | +18.5 | −3.5 % | 0.0 % | −2.9 % | 1.40 / 2.74 | −1.5 % |
| disk0, 127 threads | +22.6 | −5.0 % | −0.4 % | −4.3 % | 1.42 / 3.48 | +4.7 % |

**Verdict.** Disk mode gives back about 22.5 GiB per rank with no cache, or 18.7 GiB with a 4 GiB cache. Decode always costs about **1.4 ms per step**, whatever the cache size or queue depth. That is about −5 % single-stream and −3 to −4 % at conc32. Each step needs 17-25 misses even at a 50-85 % hit rate, and one NVMe wave at the worst of 8 ranks takes about 1 ms. Prefill is free only with prefetch (128K TTFT within ±1.5 %). Without prefetch it costs +5 to +9 %.
Not worth it as a default while RAM is not the constraint. If RAM becomes the constraint, disk4pf with 127 threads is the arm to use: about −3.5 % single, −2 to −3 % conc32, prefill unchanged.

**Notes.**
- NVMe QD33 caps at about 170K IOPS per rank (QD64 about 348K, QD128 about 571K). The default 32 threads is too few for cold prefill.
- In both control runs, the single-chunk 16K real prefill TTFT is about 1.4 s slower than in every disk arm (4.3/4.0 s vs 2.8 s). This is unexplained. Do not read it as a disk gain.
- Graph-capture rdma_ar hang: verify attempt 1 and disk0_q128 attempt 1 each needed one relaunch.
- nvme_bench.py now uses exact per-rank row ranges (the old loose bounds read a neighbour's punched rows).

**Production-worthy would need:**
1. Hiding the per-step wave: issue the reads for the bonus and draft tokens' n-grams speculatively, during the previous step's verify, for all candidate continuations; or keep the hot rows of the 4-gram ranks (r5-r7) resident.
2. io_uring or a higher QD by default.
3. Per-read timeouts, error handling and a fallback, because one stuck pread stalls all 8 ranks.
4. Restart-to-restart tests at 1M.
5. A watchdog on NVMe health and temperature.
