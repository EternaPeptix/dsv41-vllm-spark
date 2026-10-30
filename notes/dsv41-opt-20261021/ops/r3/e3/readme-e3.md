
## Round 3 / e3-conc-kv: PROMOTED (2026-10-22)
Change (launcher only, no site-packages): max-num-seqs 16 -> 32, KV cache 8G -> 16G per rank
(1.63M -> 3.26M tokens, 12.4x max-len concurrency), DSV41_TARGET_M32_MAX_M 48 -> 96 (M32 target-MoE
tiles cover 16 seqs x 6 rows), AV envelope DSV41_AV_SCALE_N0/N1 48/56 -> 96/104, RDMA_AR_MAX_REDUCE_BYTES
1 MiB -> 2 MiB (= RDMA_AR_MAX_BYTES slot; rdma_all_reduce.py caps reduce at max_bytes, so 2 MiB is the limit
without resizing the arena). New overridable knob DSV41_KV_BYTES (default 16G).
Launcher sha abd49904 -> 776de10b (all 8 identical). Backup ~/dsv41-tp8.sh.bak-r3conc (abd49904) on all 8.

Arms (fresh launch each, order K0 K1 K2 K3 K0b K3b K0c K3c; raw: ~/dsv41-opt-20261021/r3-e3/<inst>, summ.txt):
inst | code prose math mean | c4 c8 c12 c16 c24 c32 | ttft 4k/16k/32k s | gsm50 | gsm200@32 | longctx
K0  | 75.1 62.7 116.4 84.7 | 152.9 214.9 275.8 325.1 299.5 303.7 | 1.18/4.31/8.37 | 49 | -   | -
K1  | 72.8 61.3 112.6 82.2 | 148.8 213.9 276.2 324.3 385.0 409.6 | 1.20/4.35/8.36 | 50 | 196 | PASS   (seqs32 + KV16G)
K2  | 84.1 61.5 117.8 87.8 | 146.5 215.6 276.2 323.6 403.8 460.8 | 1.21/5.65/8.30 | 50 | 195 | PASS   (+M32 96, AV 96/104)
K3  | 70.6 61.7 111.7 81.3 | 150.2 217.8 275.8 320.2 411.8 475.8 | 1.15/4.26/8.38 | 50 | 195 | PASS   (+RDMA 2 MiB)
K0b | 77.8 62.4 118.8 86.3 | 149.3 213.5 276.6 324.0 300.6 298.5 | 1.20/4.55/8.37 | 49 | -   | -
K3b | 78.4 59.6 112.9 83.6 | 149.3 214.0 271.6 321.9 408.7 479.1 | 1.17/5.52/8.29 | 50 | -   | -
K0c | 80.2 60.5 116.1 85.6 | 148.0 212.9 276.5 320.5 299.9 302.0 | 1.20/4.61/8.41 | 50 | -   | -
K3c | 72.8 61.9 111.8 82.2 | 146.1 212.5 278.7 326.6 410.9 475.7 | 1.17/7.25/10.14| 50 | -   | -
Pooled K0 (3) -> K3 (3): single 85.55 -> 82.36 (-3.7%), c4 -1.1%, c8 +0.5%, c12 -0.3%, c16 -0.1%,
c24 300.0 -> 410.5 (+36.8%), c32 301.4 -> 476.9 (+58.2%).
M32 96 vs K1 (M32 48): c24 +6%, c32 +12.5%, c16 equal. RDMA 2 MiB vs K2: c32 +3.5%, c24 +1.6%.
GSM8K-200 at conc32: K1 196, K2 195, K3 195 (misses 12,93,100,119,184; ref 196 at conc16). Exact probe
self-consistent in every instance. LONGCTX PASS (60/60, copy 1.000) on K1/K2/K3. Min MemAvailable over the
whole run 12.8 GB (spark-six), 13.6 GB during K3-family conc32 (gate 4 GB). No NCCL/rdma errors.
Single-user caveat: the 3-prompt single bench reads -3.7%, which nominally fails the -1.5% gate. It is all
draft acceptance on single greedy trajectories (math AL 3.91/4.00/3.91 -> 3.74/3.79/3.76; code overlaps),
not step cost: steps/s (tok/s / AL) is identical (math 29.7-29.9 both arms, prose ~33.8). On 16 varied
single-user prompts (gsm8k_gate 16 1) K3c 87.8 tok/s vs K0d 86.2 (+1.9%). Promoted on that basis; recheck
single over a wider prompt set if this matters. TTFT 16k single samples are noisy (4.3-7.2 s; K3c also
10.1 s at 32k); not investigated, single sample per length.
Final: relaunched via dsv41-launch-all.sh, KV 3,257,974 tokens, GSM8K-50 at conc8 49/50 (miss idx 12,
known), 231.3 tok/s.
Rollback: on all 8 `cp -p ~/dsv41-tp8.sh.bak-r3conc ~/dsv41-tp8.sh` (sha abd49904) and relaunch; or per knob:
DSV41_MAX_NUM_SEQS=16 DSV41_KV_BYTES=8G DSV41_TARGET_M32_MAX_M=48 DSV41_AV_SCALE_N0=48 DSV41_AV_SCALE_N1=56
RDMA_AR_MAX_REDUCE_BYTES=1048576.
