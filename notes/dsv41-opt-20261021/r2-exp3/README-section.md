
## Round 2 / exp3-rdma-thresholds (2026-10-21, 20:30-23:10 UTC) -- PROMOTED (env-only)

**What changed:** launcher `~/dsv41-tp8.sh` gains one line (after DSV41_MOE_COMPACT):
`export RDMA_AR_MAX_REDUCE_BYTES=${RDMA_AR_MAX_REDUCE_BYTES:-1048576}` -- the custom RDMA two-shot all-reduce now
handles up to 1 MiB (102 rows x 10,240 B) instead of 512 KiB (51 rows). Effect: the 7 drafter all-reduces at 11-16
users (55-80 rows) and target verify all-reduces at 52-102 tokens no longer fall back to NCCL. No site-file change
(rdma_all_reduce.py still 3a128082; the miss-audit patch was NOT installed). Launcher 769a0564 -> d0749eda on all 8
(backup `~/dsv41-tp8.sh.bak-rdmaext` = 769a0564 on all 8).

**Offline bench (service stopped, 8 ranks, CUDA-graph, us/op, bit-exact on all ranks, 0 failures E1 and E2):**

| rows | bytes | NCCL | 1-shot | 2-shot | 2-shot speedup |
|---|---|---|---|---|---|
| 48 | 480 KiB | 627 | 223 | 80 | 7.9x |
| 56 | 560 KiB | 752 | 256 | 87 | 8.6x |
| 64 | 640 KiB | 838 | 285 | 96 | 8.8x |
| 80 | 800 KiB | 924 | 348 | 112 | 8.2x |
| 96 | 960 KiB | 431 | 413 | 138 | 3.1x |
| 128 | 1.25 MiB | 671 | 543 | 173 | 3.9x |
| 192 | 1.9 MiB | 915 | 839 | 244 | 3.7x |

Logits all-gather (x,16160) bf16: RDMA wins at 5-30 rows (2.9-4.3x) but LOSES at 48-96 rows (0.66-0.89x; e.g. 64 rows
748 vs 982 us). Engram all-gathers 3.5-8.7x. => raising RDMA_AR_MAX_BYTES above 2 MiB is not useful (Arm C dropped).
spark-four rx_discards_phy 391/18 (f0/f1) unchanged through every bench and serving run; all other hosts 0.

**Cost profile (rank0 adaptive verification):** draft@12/16 req 16.5/19.1 -> 11.1/12.5 ms; verify@56 tok 243 -> 188 ms
(-> 119 ms with DSV41_TARGET_M32_MAX_M=96).

**Serving A/B (13 instances, alternating; runarm_conc single 1024 tok + conc4/8/12/16; GSM8K-50 @ CONC 16; exact.py x2
self-consistent in every instance):**

| arm | n | single mean | conc4 | conc8 | conc12 | conc16 | GSM8K |
|---|---|---|---|---|---|---|---|
| control (C1-C4) | 4 | 79.55 | 143.2 | 207.8 | 259.0 | 299.4 | 50,50,49,50 |
| A: MAX_REDUCE 1 MiB (A1,A2) | 2 | 78.2 (-1.7%) | 142.4 (-0.6%) | 205.3 (-1.2%) | 261.9 (+1.1%) | **315.7 (+5.4%)** | 50,50 |
| B: A + TARGET_M32_MAX_M=96 | 2 | 78.1 (-1.8%) | 140.3 (-2.1%) | 209.8 (+1.0%) | 267.7 (+3.4%) | 316.8 (+5.8%) | 50,50 |
| AP: A + MAX_BYTES 1 MiB | 2 | 78.4 (-1.4%) | 142.8 (-0.3%) | 205.2 (-1.3%) | 266.3 (+2.8%) | 316.2 (+5.6%) | 49,50 |
| BP: B + MAX_BYTES 1 MiB | 2 | 79.5 (0%) | 138.5 (-3.3%) | 205.9 (-0.9%) | 263.0 (+1.6%) | 312.2 (+4.3%) | 50,49 |

Per-instance conc16: C 301.9/297.9/298.6/299.3; A 312.9/318.5; B 318.9/314.6; AP 312.3/320.2; BP 310.4/313.9.
Single/conc4/conc8 do not exercise any changed path (<=50 rows); their deltas are noise. Arm A promoted as the
minimal change meeting every gate. B/AP are within noise of A at conc16 (B has the best conc12, +3.4%, but conc4
-2.1% noise). B only matters if max-num-seqs >16. Accept length unchanged (~2.28-2.36 at conc16).

**Final state:** restarted via dsv41-launch-all.sh; /health 200; `max_reduce=1048576` in all 8 rank logs;
draft@12/16 = 11.07/12.54 ms; sanity 17*23 -> 391.

**Rollback:** `for h in spark-{one..eight}; do ssh $h 'cp ~/dsv41-tp8.sh.bak-rdmaext ~/dsv41-tp8.sh'; done` (sha 769a0564),
or just launch with `RDMA_AR_MAX_REDUCE_BYTES=524288` via conc/launch-conc.sh. Raw results: `~/dsv41-opt-20261021/r2-exp3/`.

**Follow-ups:** (1) logits all-gather 48-64 rows is slower on RDMA than NCCL: a gather-specific limit (~1 MiB) is a
small free win once a separate env knob exists (RDMA_AR_MAX_BYTES also caps reduce). (2) DSV41_TARGET_M32_MAX_M=96 +
max-num-seqs 24/32 now that both verify cliffs (AR + MoE) are gone at 56-96 tokens.
