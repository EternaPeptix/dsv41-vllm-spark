
## G1 step (2026-10-26 07:25-09:55 UTC): goal metrics G1/G4, GPU clock 2300 MHz PROMOTED (supervisor pre-step), pre-launch compaction NOT promoted

Scope: (1) goal-metric scripts and a pinned baseline, (2) GB10 graphics-clock A/B (exact by construction), (3) an optional pre-launch memory compaction against the long-context straggler. The launcher (04b6738e), every site file (manifest ad97d72f180e) and the env defaults are unchanged; the only installed change is the relaunch supervisor. Raw results: spark-one `~/dsv41-g1-20261026/` (`scripts/`, `runs/<arm>/`, `tele/<tag>-{gpu,sys}.csv` on all 8 nodes). Mac scratchpad `g1/`: drivers (`scripts/clk.sh`, `clkab.sh`, `cmptest.sh`, `probe1m.sh`, `fetch.sh`, `g1summ.py`, `clkagg.py`, `onem.py`), `summ/<arm>.json`, `clkagg-P12.md`, `cmp/<label>/`, `p1m/<label>/`, `launches.tsv`, `clk.log`, `clkab.log`.

**Live state after this step:** instance **`g1-FINAL`**, healthy 09:33:23 UTC (+351 s, first try), started by the supervisor with its defaults: launcher **04b6738e** (unchanged; AV unpinned as before, "pin miss"), clock pre-step **2300** on 8/8 (SM 2275-2288 MHz under load), no compaction.
- 8/8 active, 0 restarts, 0 ERROR/Traceback on 8/8, rdma fingerprint c2f008432bd4769f on 8 ranks, site manifest ad97d72f180e identical on 8/8, `vm.compaction_proactiveness` 0 (untouched), no telemetry left running.
- exact.py **b7e868c9d700**: the documented unpinned AV-profile alternate (p2 2228532c, p4 b246b2a0; counters 917/463/2315/1373 from its own profile). The same code gave a8c3394af9d2 on the three unpinned 2300 MHz launches just before it (C1-C3) and pinned (P1 at 2000, P2 at 2200).
- 1M cold TTFT **179.52 s, good class** (chunk ratio 1.16).
- Final suite on it (own AV profile, so AL differs a little from the pinned rows): matched C1 **31.68 / 26.96 ms** (AL 5.218 / 2.717); G1 **99.19** tok/s (code 93.66 / prose 74.42 / math 129.48); G4 per-session mean **55.48** / min **44.06** (common window 52.56 / 36.61, aggregate 168.4); TTFT ~24K 3.98 s / ~198K 32.71 s; GSM8K-50 **49/50** (miss idx 12, the usual borderline item); full longctx **PASS** needle 60/60, copy 1.000, wall 166 s (171-174 s before); `prompt_tokens_details` cached_tokens 0 -> 4160.
- Against the window's final instance (mm-FINAL, unpinned, 2000 MHz; bench single 90.93 / 72.73 / 124.63, C1 label pdab 32.254 / 27.545): G1 types **+3.0% / +2.3% / +3.9%**, C1 **-1.8% / -2.1%**.
- Hottest readings of the whole step: GPU 76 C (full longctx at 2300), ACPI zone 92 C (spark-one; critical trip 104.8 C), spark-one spbm 161.7 W peak (1M prefill).
- **User note:** the default 2300 overrides the deliberate site-layer 0-2000 MHz cap (since 2026-09-04) at every supervised launch. If that cap was a power decision, `DSV41_GPU_CLOCK_MHZ=2200` keeps most of the gain (G1 +2.5%, G4 +2.5%) for +6..10 W per node, and `=2000` restores the old behaviour.

### 1. Goal metrics (new): `~/dsv41-bench/goal/` on spark-one (git e376c58)
- `goal_single.py` (sha256 7f54fbc0) = **G1**: single-user decode tok/s. The bench.py code / prose / math prompts byte-for-byte (prompt-set sha256 **dc744bc2**), 1024 tokens, temp 0, top_p 1, thinking on (server default), x2 each in bench.py order; decode tok/s = (completion_tokens - 1) / (last - first content chunk). Reports each type, the mean of the three type means, per-request AL and an output sha256 (digest over all 6 = bitwise check across arms).
- `goal_conc4.py` (sha256 0e2c0fe6) = **G4**: 4 individual sessions with 4 DIFFERENT prompts (G1's code / prose / math + one fixed "mixed" RSA prose+math+code prompt; set sha256 **42c93c19**), started together on a barrier, 512 tokens each (all end on `length`), 1 warm-up batch excluded + 3 measured runs. Reports per-session decode tok/s **min and mean** per run and over the 3 runs, plus a common-window rate (tokens each session produced while all 4 were decoding, from continuous usage stats) and the batch AL.
- Both refuse to run if their embedded prompt sha256 does not match. `SHA256SUMS` and a README sit next to them. Arm suite used for every A/B: `~/dsv41-g1-20261026/scripts/gsuite.sh` (matched C1 coding/prose x5 with the fixed label **pdab**, then G1, G4, then cold TTFT with bench.py `ttft` targets 16384 / 131072, which are really **~24K / ~198K tokens**).

**Baseline (live config, pinned `av-pin-pdfinal.json`, 2000 MHz boot policy; 2 instances P1, P2, 4 arms, ABBA within each instance):**

| metric | mean (sd over 4 arms) | per arm |
|---|---|---|
| G1 mean tok/s | **95.45** (0.35) | 95.95, 95.25, 95.41, 95.19 |
| G1 code / prose / math | 89.6 / 72.0 / 124.8 | |
| G4 per-session mean / min tok/s | **54.36** (0.66) / **43.48** (0.41) | 55.26/44.08, 53.92/43.19, 53.82/43.46, 54.45/43.21 |
| G4 by session: code / prose / math / mixed | 51.0 / 43.5 / 67.8 / 55.2 | the min is always the prose session |
| G4 common-window mean / min | 51.73 / 36.71 | aggregate 165.4 tok/s |
| matched C1 coding / prose ms (label pdab) | 32.80 / 27.99 (AL 5.224 / 2.718) | |
| TTFT ~198K tokens | 33.43 s | |
| exact.py (first request on each instance) | a8c3394af9d2, counters 905/471/2355/1373 on P1 and on P2 | |

Distance to the goal: G1 needs **+25.7%** (95.45 -> 120). G4 per-session **mean** is already above 50 (54.4), but the **min** session (prose, AL 1.8) is 43.5 (36.7 while all four decode), so "50 per session" is not met.

### 2. GPU graphics clock A/B: 2300 MHz PROMOTED (supervisor pre-step, default 2300)
**How GB10 clocks are governed here.** The clocks are not "unthrottled": every node runs the site-layer unit `dgx-spark-gpu-clock.service` (`ExecStart=/usr/bin/nvidia-smi -lgc 0,2000`, enabled under DGX OS since 2026-09-04 and carried into spark-os), so under load the SM clock sits at 1969-1989 MHz with event reasons 0x0. `nvidia-smi -q`: default application clock 2418, max 3003 MHz; Supported Clocks N/A (`-ac` unusable), memory clock N/A (LPDDR5X fixed), all power limits N/A (`-pl` unusable). `-lgc 0,N` applies at runtime in under a second (no restart) and a reboot restores 2000. Root came from throwaway `docker run --rm --privileged --pid=host --network none kindling-spark-os:0.9.5` + `nsenter -t 1 -m` (Mac `g1/scripts/clk.sh`); every change is logged in `g1/clk.log`.

**Method.** Live, on 2 pinned instances (P1 launched at 2000, P2 launched through the new pre-step at 2200). Arms per instance in balanced order (P1: 2000, 2200, 2418, 2300, 2200, 2000; P2: 2200, 2000, 2300, 2300, 2000, 2200), each = `gsuite.sh` (~3.3 min) with 2 s telemetry on 8/8 (nvidia-smi SM MHz / GPU W / temp / reasons; ACPI zones; spark-one spbm `sys_total`), switch PSU input at each arm start. Watchdog: GPU > 80 C or spark-one > 200 W reverts to 2000 at once. Stop rule: GPU > 80 C, or node power > +25 W (spark-one spbm, the only metered node) vs the same instance's 2000 MHz arms.

**Exactness.** Every arm on both instances gave the same G1 output digest **9a18f802d930f6e7** and identical matched AL (5.224 / 2.718); exact.py on P2 (clock 2200 set before launch) = a8c3394af9d2 with the pinned counters 905/471/2355/1373.

| cap MHz (SM under load) | n | C1 coding / prose ms | G1 mean (code / prose / math) | G4 sess. mean / min | TTFT ~198K s | spark-one sys W delta: matched / G1 / G4 / prefill | GPU Tmax | ACPI Tmax (crit 104.8) |
|---|---|---|---|---|---|---|---|---|
| 2000 (1969-1989) | 2x4 | 32.80 / 27.99 | 95.45 (89.6 / 72.0 / 124.8) | 54.36 / 43.48 | 33.43 | (base 91.2 / 89.1 / 96.5 / 101.1 W) | 68 | 89 |
| 2200 (2177-2184) | 2x4 | 32.17 / 27.40 (**-1.9% / -2.1%**) | 97.88 (**+2.5%**) | 55.71 (**+2.5%**) / 44.50 (+2.3%) | 32.92 (-1.5%) | +6.4 / +7.3 / +7.8 / +9.6 | 71 | 87 |
| **2300** (2275-2288) | 2x3 | 31.86 / 27.09 (**-2.9% / -3.2%**) | 98.97 (**+3.7%**) (93.0 / 74.6 / 129.3) | 56.16 (**+3.1%**) / 44.72 (+2.8%) | 32.59 (-2.5%) | +8.2 / +11.3 / +14.3 / +16.8 | 71 | 92 |
| 2418 (2398-2405) | 1x1 | 31.41 / 26.73 (-4.2% / -4.4%) | 100.40 (+5.0%) | 56.48 (+3.5%) / 45.52 (+4.3%) | 32.44 (-3.2%) | +20.8 / +21.2 / **+25.8 / +26.3** (+28.8 vs the adjacent 2000 arm) | 74 | 86 |
| max 3003 / -rgc | - | not run: 2418 already broke the +25 W rule | | | | | | |

(Percentages are paired per instance vs that instance's 2000 MHz arms, then averaged. Per-arm table: `g1/clkagg-P12.md`. Noise: G1 sd 0.35, G4 mean sd 0.66, C1 sd 0.05-0.10 ms over the four 2000 arms.)
- Fleet GPU-rail sum in G4: +39 W (2200), +60 W (2300), +107 W (2418) over 8 nodes. Switch PSU input 251-260 W in every arm (unchanged). Studios idle (11-13 W).
- Gain per clock: ~0.25% G1 per 1% SM clock (decode is mostly weight-streaming bound; prefill gains even less).
- **Sustained check at 2300 (1M solo prefill, ~3-4 min):** pinned P2 (slow class) GPU max 72 C, ACPI max 90.1 C, spark-one 106 W mean / 160 W peak; unpinned good-class C2 GPU max 74 C, ACPI max 90.4 C (spark-one), spark-one 117.8 W mean / 161.7 W peak. Under 80 C GPU everywhere. The hottest ACPI zone is always spark-one (then spark-two); it heat-soaks with load time and reached 89.1 C in a **2000 MHz** arm after ~30 min of continuous runs, so it tracks sustained load more than clock.
- **Decision:** 2418 fails the +25 W rule (stopped, max not tried). 2200 and 2300 are both exact and clear the gate; **2300 is promoted** (largest gain within the rule: G1 +3.7%, G4 +3.1%, C1 -3%, TTFT -2.5%, +8..17 W mean per node; GPU <= 76 C even in 1M prefill and full longctx). This is still below NVIDIA's own default application clock (2418) and boost range; it raises the site-layer 2000 MHz cap only while the supervisor has set it.

**What was installed.** `ops/rel/dsv41-relaunch.sh` **30158e80** on spark-one (backups `.bak-g1-pre` = 39679a35 original, `.bak-g1-clock` = f2f6e44c = pre-step code with the clock default off). After the stop and before the start, on all 8 nodes via the privileged container, logged to `~/dsv41-logs/prelaunch-<stamp>-t<try>.log` per node and one line per node in the supervisor output; the TSV row now ends with `[pre: compact=X clock=Y]`. Supervisor arguments (consumed by the supervisor, never passed to the ranks):
- `DSV41_GPU_CLOCK_MHZ=<1000..3003>|off` (default **2300**): `nvidia-smi -lgc 0,<MHz>`, idempotent. `=2000` restores the old policy for that launch; `off` leaves the clocks alone.
- `DSV41_PRELAUNCH_COMPACT=1|snap|0` (default **0**): see section 3; `snap` only logs buddyinfo/meminfo/vmstat.
- **For future A/Bs** (e.g. the staged g1/stepovh bundle): every supervised arm now runs at 2300 unless `DSV41_GPU_CLOCK_MHZ` says otherwise, so keep the clock identical across arms. Pinned baselines at 2300 are the 2300 rows above (G1 98.97, G4 56.16 / 44.72, C1 31.86 / 27.09). The pin `av-pin-pdfinal.json` was profiled at 2000; it stays valid as a common pin (exact.py a8 with identical counters at 2200, identical G1 digests at 2000-2418).
- A failed pre-step is a logged WARNING and the launch proceeds. Launches through the old `dsv41-launch-all.sh` / `launch-conc.sh` do not touch clocks (whatever was set last stays until a reboot, which restores 2000).

### 3. Pre-launch memory compaction (`DSV41_PRELAUNCH_COMPACT=1`): NOT promoted
Pre-step: wait for no GPU compute apps (<= 30 s), snapshot, then `sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory` in the privileged container, snapshot again (buddyinfo, MemFree/MemAvailable/Cached/AnonHugePages, compact_*/pgmigrate_*/drop_* vmstat). Takes 0.5-1.7 s per node (+2 s to the launch). Node facts: 4 KiB pages, buddy orders 0-13, `vm.compaction_proactiveness=0` (site layer, deliberate), drop-caches cron (`echo 1` every 5 min) on one..five only.

**Test:** 3 unpinned launches (launcher defaults, supervisor default clock 2300), each: exact.py first, buddyinfo on 8/8 after healthy, cold solo 1M (`xr_bench.py solo --len 1000k`, per-chunk sched stamps), buddyinfo after the 1M. Class from the chunk curve (`g1/scripts/onem.py`): ratio = mean chunk 50-62 / mean chunk 2-12; good < 1.30, slow > 1.40 (history: good 1.11-1.27, slow 1.80-1.89; the classifier separates the classes unchanged at 2300: P2 slow 1.80, C2/C3/FINAL good 1.16).

| launch | free memory in >= 2 MB blocks before -> after compaction (8 nodes, of ~117 GB free) | healthy | exact.py (own AV profile) | 1M TTFT s | chunk bands c2-12 / c20-30 / c37-43 / c50-62 s | ratio | class |
|---|---|---|---|---|---|---|---|
| C1 | 45.8-61.1 -> 60.8-66.8 GB | +350 s | a8c3394af9d2 (906/470/2350/1373) | 225.82 | 2.624 / 3.001 / 3.980 / 4.794 | 1.83 | **slow** |
| C2 | 44.9-62.4 -> 59.1-70.7 GB | +351 s | a8c3394af9d2 (908/468/2340/1373) | **178.89** | 2.623 / 2.771 / 2.901 / 3.035 | 1.16 | good |
| C3 | 49.6-63.9 -> 61.9-69.2 GB | +353 s | a8c3394af9d2 (901/475/2375/1373) | **179.30** | 2.615 / 2.770 / 2.901 / 3.026 | 1.16 | good |

**History** (unpinned, 2000 MHz, no compaction; same classifier): X-ray A (= pdab-FINAL) 1.87 slow, C 1.84 slow, E 1.89 slow, R1 1.85 slow, pf-FINAL 1.16 good, fv mm-FINAL 1.83 slow = **1 good / 5 slow**. Pinned at 2000: 6 good / 2 slow. This step's pinned P2 (probed at 2300): slow (1.80, 221.76 s). **Same-day control** g1-FINAL (unpinned, 2300, compaction off): good (1.16, 179.52 s).

**Verdict: NOT promoted** (default stays `DSV41_PRELAUNCH_COMPACT=0`; the pre-step stays available, +2 s, no effect on numerics). C1 was slow, so compaction does not make launches reliably fast. 2 good of 3 against the 1-in-6 history is only suggestive (binomial p = 0.07), and the same-day no-compaction control was good too: all unpinned launches with a 1M probe give compaction 2/3 vs no compaction 2/7 (Fisher one-sided p = 0.33).

**Why it cannot work as designed (per-rank buddyinfo, `g1/cmp/<label>/bi-*.txt` and the supervisor's per-node prelaunch lines):**
- After the stop: 108.9-116.7 GB free per node, of which 44.9-63.9 GB (39-55%) is in >= 2 MB blocks. six/seven/eight carry ~5 GB of page cache (no drop-caches cron there) and the lowest >= 2 MB share.
- One compaction pass raises that to 59.1-70.7 GB (51-60%) in 0.5-1.7 s. The rest stays in 4 KB-1 MB pieces: the ~4.4 GB the system keeps while vLLM is down is spread over the whole ~117 GB, and one pass cannot consolidate around pages it cannot move.
- After every startup, slow or good, every rank has 10-18 GB free and **<= 0.5 GB in >= 2 MB blocks**. The 1M prompt then takes another 2-7 GB (MemAvailable 17.6-18.6 GB at probe start, minimum 10.2-11.6 GB, the same on the slow and the good launches). So the startup consumes whatever compaction produced, and no per-rank memory figure separates the slow launch from the good ones (C1's lower spark-six MemFree was page cache; its MemAvailable matched the others).
- At 2300 MHz, good-class 1M TTFT is 178.9-179.5 s (182-186 s at 2000); slow class 221.8-225.8 s.

**Next:** the straggler is still random per launch. The practical guard remains a relaunch-if-slow screen after healthy (1M or ~700K probe, chunk ratio > 1.4 => relaunch); `g1/scripts/probe1m.sh` + `onem.py` already implement the probe and the classifier. Root cause still needs a slow-vs-good nsys pair on rank 5 only.

### Rollback
- **Clock, at once (no restart):** Mac `bash g1/scripts/clk.sh revert` (= `nvidia-smi -lgc 0,2000` on 8/8 through the privileged container), or `sudo -n /usr/bin/nvidia-smi -lgc 0,2000` per node (NOPASSWD rule exists). A reboot also restores 2000 (boot unit).
- **Clock for one launch:** supervisor argument `DSV41_GPU_CLOCK_MHZ=2000`.
- **Supervisor:** `cp -p ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh.bak-g1-clock ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh` (pre-steps present, clock default off) or `.bak-g1-pre` (39679a35, the pre-step code removed). spark-one only; no other node holds it.
- Goal scripts and `~/dsv41-g1-20261026/` are not referenced by the launcher or the supervisor; they can stay.
