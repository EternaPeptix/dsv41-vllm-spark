
## Takeover audit after Codex goal-120 (2026-10-24 21:15-21:45 UTC). No config change, no restart

Raw: `~/dsv41-takeover-20261024/` on spark-one (`anchor/live-2055/`, `baseline/live-2055/`, `scripts/tsuite.sh`, `scripts/tsumm.py`, `avtau.py`). Full audit with the experiment list and lead map: Mac scratchpad `takeover-audit.md`.

**Who drives.** Codex (ChatGPT app, goal "reach 120 tok/s while still remaining exact") ran the cluster from 16:42 to 21:13 UTC. Its goal is **paused, not cancelled** (21:13:10). If someone resumes it, Codex will drive the cluster again. At 21:18 UTC there was no Codex process on the Mac and no Codex automation or cron entry. The Sparks ran only the 8 `dsv41-rank` units and the system daemons: no tmux, screen, cron or user timers. The API queue was 0/0 and the request counter stayed at 60 until our runs. PID 78044, Codex's last live handle, does not exist. Codex's final action, the c1/c4/c8/c16 load screen, completed all 8 cells, but Codex never audited them.

**Live instance.** Codex's D05 lifecycle script started it at 20:55:26 UTC, which is why the unit description reads `d05-lifecycle pooled-shadow-v2a normal rankN`. It is the normal launcher, and the next launch-all resets the name.
- `~/dsv41-tp8.sh` is **2027f4d8** on 8/8. The env is the launcher defaults: the k1/2 set plus `DSV41_RDMA_PARLOAD=1`. There is no AV pin.
- No goal-120 code is loaded. Worker maps on 8/8 show only skinny_mx, the m32 native lib, libdsv41_l2pf and `rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so`.
- 0 restarts, 0 ERROR/Traceback lines and Engram check OK on 8/8.

**Site files that differ from the last documented live set (k1/2, 15:10 UTC).** sha256 is identical on 8/8 for every file.

| file (vllm/...) | documented | live | installed by | gating |
|---|---|---|---|---|
| v1/worker/gpu/model_runner.py | 9fecbda4 | **6f9d997f** | Codex C02, 18:09:37 | **none**: an unconditional `torch.accelerator.synchronize()` between the decoder graph capture and `speculator.capture()` (startup only) |
| distributed/device_communicators/dsv41_l2_prefetch.py | 7d2acdc8 | **4da7745b** | Codex D01, 18:09:37 | `DSV41_L2_PREFETCH_PRUNE_ROUTER/_SHARED/_STALE_QKV`, `_PLAN_AUDIT`, all default 0. With them off it builds the same plan |
| distributed/device_communicators/rdma_all_reduce.py | 99102ff1 | **acbeeef7** | Codex k1/3 promotion, 15:56:56 | `DSV41_RDMA_PARLOAD` (launcher default 1) loads `~/rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so` 62daad6c. The legacy `build/` dfeeab00 is still there |
| launcher `~/dsv41-tp8.sh` | 1a2f26f4 | **2027f4d8** | Codex k1/3, 17:18:40 | `.bak-k1-3-rdma-loads` = 1a2f26f4 |

- The originals of the first two files are in `~/dsv41-goal120-20261024/backups/{model_runner.py.9fecbda4,dsv41_l2_prefetch.py.7d2acdc8}` on 8/8, not as `.bak-*` next to the files. `rdma_all_reduce.py.bak-k1-3-rdma-loads` = 99102ff1 on 8/8.
- Every other site file matches the README. 33 modified or new files plus 43 backups, all identical on 8/8.

**Staged, not live:**
- The D05 native package `~/dsv41-goal120-20261024/d05-shadow-v1/native` (889 files, file-list hash 2a8a84f7e765) and the serving-v1/v2 adapters (c807f9d1eaf0 / b5a08273dfa5) are identical on 8/8. They are loaded only by Codex's shadow lifecycle launcher.
- D02 `qkv-pipeline`, D04 `m32-deadwalk` (+ `-native-v1`) and P09 `dualquant` are on spark-four only, for component tests.
- The D01 post-budget arm, the fixed-cohort schedule gate and the proposal-capture prototype exist only on the Mac, in Codex's workspace `~/Documents/Codex/2026-10-20/let-s-improve-our-deployment-of-2/work/goal120-20261024/`.

**Exactness anchor.**
- Both hashes come from the same `~/dsv41-bench/exact.py` (e50b924d). Codex ran it through the k1 `ksuite.sh`, with the live config and prefix caching on.
- The live instance gives **a8c3394af9d2 x3**, plus x1 after the baseline load. Per prompt: 8c6a41c0 e3aa05bc 1b2d3bb7 cd93c706 9ba77590 bb1e4093.
- Codex's 9330f03c4f97 differs only in p2 (2228532c) and p4 (8c5e765e). The other known alternate, b7e868c9d700 (k1/2 i2-A), is p2 2228532c + p4 b246b2a0.

Cause: the startup AV cost profile under `DSV41_VERIFY_CAP=auto`. The profile came from Codex's 17:25 instance (`final-default`), the same instance that later served its 18:02 `control` run.
1. Codex's `c0-recovered` (PARLOAD off) and `a1-pinned` (PARLOAD on, with c0's profile pinned) both gave a8 x2, with identical spec counters.
2. `profile-control-screen` and `profile-shared-pruned-screen` were pinned to the 17:25 profile (`staging/av-pin-initial.json`). They reproduce 9330 x2 even after C02 and the l2pf file were installed.
3. Unpinned normal instances with the same code gave a8 at 19:05 and at 20:55.

The nondet arm A8 (VERIFY_CAP=0, 262k, prefix caching **off**) reaches the same 9330 branch through prefill numerics. Codex's instances had prefix caching on, so prefix caching is not the cause there. No code change altered greedy output: PARLOAD, C02 and l2pf with flags off all give identical outputs and counters under a common AV profile. `avtau.py` lists the effective profile against the hash for every saved instance; the flip boundary is not one-dimensional.

**Baseline** (this instance, `tsuite.sh`, 21:33-21:38 UTC):

| metric | live (20:55 instance) | k1/2 final live (15:09) |
|---|---|---|
| matched C1 coding ms/step (x5) | **33.574** (sd 0.31, AL 5.13) | 34.08 |
| matched C1 prose ms/step (x5) | **28.620** (sd 0.40, AL 2.70) | 29.29 |
| bench single code / prose / math tok/s (x2) | **87.9 / 69.7 / 120.9** (AL 2.573 / 1.816 / 3.543; counters identical across reps) | 86.1 / 69.1 / 118.0 |
| conc4 agg (per-req mean / min) | 153.9 (50.2 / 40.6) | - |
| conc8 | 228.9, 228.7 (37.2 / 30.0) | 224.5, 227.4 |
| conc16 | 332.2, 339.8 (28.5 / 21.9) | 332.8-334.6 |
| conc32 | 497.8, 496.6 (20.9 / 15.6) | 489.9-493.5 |
| exact.py | a8c3394af9d2 x4 | a8c3394af9d2 x2 |

Codex's load screen (21:11-21:13; its own prompts, thinking on, 512-token cap; per-user min / aggregate tok/s):

| type | c1 | c4 | c8 | c16 |
|---|---|---|---|---|
| code | 76.4 | 47.1 / 179 | 33.9 / 263 | 24.5 / 389 |
| prose | 71.6 | 40.6 / 157 | 29.9 / 230 | 22.1 / 341 |

**Codex goal-120 experiments.**
- No new candidate was promoted after k1/3.
- C02 and the D01 flags were installed without a README entry.
- D05 is the strongest pending decode result, but it has had no serving A/B.
- The profile (nsys C1/C2/C4 on 2027f4d8, pinned AV) is reusable: `~/dsv41-goal120-20261024/profile-control/r{0..7}.{1,2,3}.nsys-rep`. C1 step is 28.9 ms (target 25.1, drafter 2.6). FC1/FC2 kernel sums are 5.34/3.11 ms at C1 and 14.44/9.70 ms at C4.
- Full list and outcomes: `takeover-audit.md`.

**Rollback of Codex's undocumented installs** (stop the service first, then run on each rank):
```
cp -p ~/dsv41-goal120-20261024/backups/model_runner.py.9fecbda4 ~/venvs/vllm032/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py
cp -p ~/dsv41-goal120-20261024/backups/dsv41_l2_prefetch.py.7d2acdc8 ~/venvs/vllm032/lib/python3.12/site-packages/vllm/distributed/device_communicators/dsv41_l2_prefetch.py
```
Then check sha 9fecbda4 / 7d2acdc8 on 8/8. The k1/3 rollback is Codex's: `DSV41_RDMA_PARLOAD=0` via launch-conc.sh, or restore `~/dsv41-tp8.sh.bak-k1-3-rdma-loads`.
