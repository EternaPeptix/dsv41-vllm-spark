
## Goal window 5, ops step (2026-10-27 18:59-20:20 UTC): Codex ops kits 04 / 08 / 10 / 03 / 06. Kit 08 regression gate INSTALLED (first run PASS); kit 10 black box + panic sysctls on spark-three through the site layer (image 0.9.5-1019-bb1, reboot-tested, promoted on spark-three only); kit 03 NOT promoted; kit 06 profiled, nothing installed. Launcher f214c2c9 unchanged; live g5ops-PROD

Exactness: nothing here touched the serving path. The vllm site tree is still 4baa4e458a1a (5,453 files) on 8/8, the
supervisor manifest is still d773ed60f27b, and exact.py gave a8c3394af9d28249 on both instances of this step (K03 and
PROD). Clocks: SM 1969-1989 MHz at every read (supervisor pre-step 2000 MHz; nothing raised).

### 1. Results

| Kit | Result | Evidence |
|---|---|---|
| 04 fast loader | Already live since window 4; confirmed | `DSV41_FAST_LOAD=1` in the env on 8/8. Rank 0: target weights 35.9 s, model load 46.7 s. Launch to healthy: production (AV unpinned) +200 s (g5dr-PROD 18:49 and g5ops-PROD 19:47 UTC); pinned +138 s (g5ops-K03). Window 4 measured 355 -> ~200 s production and 286 -> 126-132 s pinned |
| 08 regression gate | INSTALLED: `bash ~/dsv41-bench/gate/gate.sh --live` | First run 19:51 UTC on g5ops-PROD: PASS in 570 s (budget 900 s). Results in `~/dsv41-bench/gate/runs/f214c2c9/` |
| 10 black box | INSTALLED on spark-three only (site layer, image 0.9.5-1019-bb1) | Two reboots of spark-three; both verifies 0 problems; panic=10 / oops / soft+hard lockup sysctls live on the promoted boot; log growing; earlier boot classified `clean_stop` |
| 03 proxy idle | NOT promoted | Adds +15.7 ms (median; max +20.9) to the first request after 3 s of idle. Idle power not measurably lower. The kit's premise was ~75x too high |
| 06 host path | Profile only; nothing installed | API server 2% of a core; EngineCore busy-waits; the rank-0 worker's pre-launch Python (~1.6 ms per G4 step) is the only host lever with room |

### 2. Kit 08: the regression gate
- One command on spark-one: `bash ~/dsv41-bench/gate/gate.sh --live [--label L] [--no-1m]`. `--plan` prints the steps,
  `--last` the newest result. Exit 0 PASS, 1 FAIL (the summary names the step), 3 REFUSED (busy; nothing is run).
- Steps (each stops the gate on a hard failure): refuse_if_busy, exact.py (anchor a8c3394af9d2), G1 one pass
  (tok/s and digest vs the previous PASS under the same launcher; informational), ONE cold 1M probe (TTFT <= 220 s,
  MemAvailable minimum), GSM8K-200 (>= 194), longctx 4K-200K (LONGCTX PASS), kl_panel verify / smoke /
  capture --sets tools,math (81,840 positions; compare --tier equiv against the previous gate capture).
- Results: `runs/<launcher sha8>/<UTC>.json` plus a raw directory. A 2 s /metrics sampler and per-step
  sent/finished counts flag foreign traffic.
- **First run, 19:51 UTC, f214c2c9 / g5ops-PROD (production, AV unpinned): PASS, 570 s.**

  | Step | Result |
  |---|---|
  | exact.py | a8c3394af9d28249 |
  | G1 one pass | 101.81 tok/s; digest abcdf1eb96654d71 (code 596b478a, math 92e5d8cf, prose e8a524f3) |
  | GSM8K-200 | 195/200 (misses 93, 100, 119, 182, 184; reference 196) |
  | longctx | 60/60 needles, copy 1.000 |
  | 1M probe | TTFT 168.26 s for 1,024,611 tokens, good class; MemAvailable min 5.98 GB on spark-one |
  | KL panel | verify OK; smoke OK (first prompt_logprobs requests on this server, 1,569 pos/s); capture tools+math 80 seqs in 38 s = baseline `gate-f214c2c9-20261027T195149Z` |

- **Two fixes after the first run (script v2, sha 1889ae4a5ce3; v1 kept as `gate.sh.v1`):**
  - The 1M probe now runs before GSM8K and longctx. In the kit's order it ran right after the 200K longctx requests
    and bottomed at MemAvailable 5.98 GB on spark-one, against the 6 GB floor. Earlier suites ran it first and saw
    8.6-8.7 GB. All 8 nodes still read 6.2-7.4 GB at idle after it (PyTorch caching allocator).
  - Requests-sent counts: longctx sends 2 requests per row, and xrb4 solo sends 2 settle requests plus the 1M one.
    The first run's "foreign" flags on those two steps were this miscount (erratum added to its JSON); the sampler
    shows no foreign request.
- The G1 prose sha differs from g5dr-PROD's capture (e8a524f3 vs a22e1d39) on the same launcher. That is the known
  prose near-tie on unpinned launches; code and math are identical. That is why the digest check is informational.

### 3. Kit 10: black box and panic sysctls on spark-three, and the reboot test
- **Site layer** (spark-three `~/kindling-site`; pre-change tree `~/kindling-site.bak-g5-ops.tgz`; note and removal
  commands in `~/kindling-site/README-g5ops.md`):

  | File | What |
  |---|---|
  | `root/opt/spark-site/blackbox.py` (f5014d821794) | 1 Hz /proc logger: meminfo, memory PSI, loadavg, uptime. One fsync'd line per second to `/var/log/spark-blackbox/blackbox.log`, which is bound from the DGX OS disk. Writes BOOT / RESTART / STOP markers, copies pstore records next to the log, and rotates at 64 MB x 3. `classify` names each boot's ending: power_cut / panic / clean_stop / unknown |
  | `root/opt/spark-site/90-spark-panic.conf` (4905c9f15c72) | panic=10, panic_on_oops=1, softlockup_panic=1, hardlockup_panic=1 |
  | `spark-panic-sysctl.service` | Applies that file at sysinit, overriding the promoted cmdline's panic=30 |
  | `spark-blackbox.service` | The logger: nice 19, idle I/O class, Restart=always |
  | `customize.d/20-spark-three-blackbox.sh` | Enables both units |

  - Both units carry `ConditionHost=spark-three`, so a later fleet rebuild with this layer stays inert on the other seven.
  - Not set: `vm.min_free_kbytes` (the recipe's 2 GiB would move our KV pool) and `hung_task_panic` (a 1M prefill
    blocks on purpose).
  - The kit's systemd watchdog drop-in (`RuntimeWatchdogSec=30`) is NOT installed. spark-os's own
    `spark-watchdog.service` already holds the single-open SBSA `/dev/watchdog`, with the site's 310 s / 12-probe
    setting.
- **Image 0.9.5-1019-bb1.**
  - Built on spark-three 19:20-19:25 UTC with `~/sparkos-build-bb1.sh`.
  - `~/sparkos-deploy/image_check-bb1.sh` gave IMAGE-OK, including a package diff against 0.9.5-1019.
  - Kernel 7.0.0-1019-nvidia (vmlinuz ddd5a25c2326) and driver 580.178.04 are identical. The build pulled 33 current
    Ubuntu updates: systemd 8.17->8.18, openssh 13.19->13.20, curl, bind9, ca-certificates, glib, xz, zlib, git,
    tzdata, tmux, apparmor, brotli, and nvidia-modprobe 615.71.09->615.71.10.
  - Installed from spark-os as a one-time trial (`~/sparkos-deploy/install-bb1.sh`, which uses
    `-v /run/sparkos/host:/host`). GRUB saved_entry stayed sparkos-0.9.5-1019 until the promote.
- **Reboot test (spark-three = rank 2; the other seven stayed up).** Quiet time: no requests running or waiting. The
  only traffic was a LAN client (192.168.0.86) polling GET /v1/models every ~2.5 s.

  | UTC | Event |
  |---|---|
  | 19:29:12 | dsv41 stopped on 8/8 |
  | 19:29:51-19:31:52 | Stopped-state idle measurement (kit 03) |
  | 19:32:45 | Reboot #1, back at 19:33:23 on bb1 (trial) |
  | 19:33:4x | `~/sparkos-deploy/verify-bb1.sh`: 0 problems. That is the fleet verify (fabric 4/4, RDMA 4 ports, GID 3, mentat 7 peers, tailscale, docker GPU, torch GPU) plus the unit, sysctl, log-growth and watchdog-owner checks |
  | 19:34:13 | Promoted (`verify-bb1.sh --promote`); saved_entry is now sparkos-0.9.5-1019-bb1 |
  | 19:34:21 | Reboot #2, back at 19:35:03 on the promoted boot (cmdline panic=30, live kernel.panic=10); verify 0 problems |
  | 19:36:23 | Relaunch g5ops-K03 (kit 03 arm), healthy +138 s, exact a8c3394af9d28249 |
  | 19:43:31 | Relaunch g5ops-PROD (launcher defaults), healthy 19:47:07 (+200 s) |

  - Service fully down 19:29:12-19:38:51 and 19:43:31-19:47:07. Off the production config 19:29-19:47 (~18 min).
  - The black box classified the trial boot's end as `clean_stop`: STOP marker, signal 15.
  - At 20:1x UTC the log was still growing, one line per second.
- **Check:** `ssh spark-three 'python3 -I /opt/spark-site/blackbox.py classify'`, or `tail /var/log/spark-blackbox/blackbox.log`.
- **Remove** (in order):
  1. Stop dsv41 on 8/8.
  2. `ssh spark-three 'sudo -n /usr/local/sbin/sparkos-rollback 0.9.5-1019'` (fleet image as the default again, then a reboot).
  3. `ssh spark-three 'docker run --rm --privileged -v /run/sparkos/host:/host kindling-spark-os:0.9.5 remove 0.9.5-1019-bb1'`.
  4. `ssh spark-three 'cd ~ && rm -rf kindling-site && tar xzf kindling-site.bak-g5-ops.tgz'`.
  5. Relaunch with the supervisor.

### 4. Kit 03: the RDMA proxy at idle. NOT promoted
The kit models the proxy at about 2/3 of a core when idle (~2.4 W per node, ~19 W fleet). The measurement (120 s per
arm, 8/8 nodes, per-thread /proc, spbm on spark-one) puts it at 0.9% of a core:

| Arm (all idle) | Box CPU busy | dsv41 CPU | Proxy thread | Proxy wakes | Box ctxt/s | spark-one sys_total | spark-one cpu_p+cpu_e | spark-one GPU rail |
|---|---|---|---|---|---|---|---|---|
| Production #1, 19:04 (max sleep 1 ms) | 0.35% | 3.26% of a core | 0.88% | (not taken) | - | 33.63 W | 1.716 W | 8.82 W |
| Service stopped, 19:30 | 0.24% | 0 | - | - | 1,336* | 27.84 W | 0.993 W | 5.11 W |
| K03, 19:40 (`RDMA_AR_MAX_SLEEP_US=20000`) | 0.32% | 3.44% | 0.25% | 49/s | 3,226 | 34.60 W | 1.729 W | 8.91 W |
| Production #2, 19:47 (1 ms) | 0.34% | 3.30% | 0.89% | 943/s | 5,088 | 35.74 W | 1.904 W | 9.23 W |

(* spark-one, from a 4 s sample. Other columns are 8-node means.)

- **Power.** The 20 ms sleep cuts the proxy from ~943 wakes/s to ~49/s. On spark-one, the only node with a box meter,
  the CPU rails were 0.08 W below the mean of the two production measurements. The production measurements alone
  differed by 0.19 W. sys_total moved with GPU temperature after the relaunches: 33.6 -> 34.6 -> 35.7 W. So there is no
  measurable saving; an upper bound is ~0.1 W per node.
- **The whole service at idle** costs spark-one ~6.9 W against stopped: GPU +3.7-4.1 W, CPU +0.7-0.9 W, the rest memory
  and SoC. Most of the remaining service wakes are 8 gloo_tcp_loop threads at ~95/s each.
- **Latency.** `~/dsv41-g5ops/scripts/ttft_idle.py`, 12 pairs: a tiny request after 3 s of idle, then one right after
  it. K03 adds +15.7 ms median (max +20.9 ms). Production at 1 ms adds +1.8 ms (max +4.6).
- The rule requires no added latency, and there is no power to gain, so it is not promoted. G1 blocks were not run:
  the trim rule says stop at a clearly negative first result.
- A no-latency version would need the worker to wake the proxy at step launch (.so plus Python change). For
  <= 0.1 W per node it is not worth building.

### 5. Kit 06: py-spy of rank 0 during G4
- **Method.** py-spy 0.4.2 (`~/.local/bin/py-spy`), root through the privileged spark-os container, `--nonblocking`,
  50 Hz, 20 s, during batches 2-3 of a 6-batch G4. Two runs: API + EngineCore, then the rank-0 worker main thread.
- **Impact.** The profiled batches fall inside the unprofiled ones' spread. G4 min/mean per run was 48.04/61.32 and
  48.07/60.91, against 48.41/61.05 in window 4 production.
- **Ranking:**
  1. **Rank-0 worker main thread** is busy 11.5% of wall: ~115 ms CPU/s, ~5 ms per ~45 ms G4 step. Of that:
     - torch.cuda synchronize in AV `record_confidences`: 35 ms/s (~1.6 ms/step, host blocked on the GPU);
     - Triton launch overhead in `_dsv41_prepare_attn`: 16 ms/s;
     - `build_attn_metadata`: 13 ms/s;
     - drafter `propose`: 14 ms/s;
     - `sample`: 9 ms/s;
     - rejection `_verify`: 8 ms/s;
     - `prepare_inputs`: 7 ms/s.

     The pre-launch part (`prepare_inputs` + `prepare_attn` + `build_attn_metadata`) is ~1.6 ms/step. Where the GPU sits
     idle during it, overlapping or shrinking it is worth up to ~3.5% of the G4 step (the prose session needs +3.3%).
     Exact options: pre-bound Triton launchers (skip the per-launch cache-key and binding work), CUDA-graph the
     metadata kernels, or async-scheduling overlap. **First get an nsys timeline of the GPU-idle share.**
  2. **EngineCore** is active 86% of wall, and ~88% of that is the shm_broadcast busy-wait for worker output
     (sched_yield, wait, acquire_read). It is not on the critical path. It burns ~0.9 core while serving, a power item.
     The kit's "scheduler skips the empty waiting queue" has nothing to gain.
  3. **API server** is active 2.2% of wall (19 ms CPU/s), with no detokenize frame in the top. vLLM v1 already
     detokenizes incrementally, and the kit's 96 ms/step model does not apply at 4 sessions. Incremental detokenize
     and batched SSE: no gain.
  - Two RPCs per step to the workers (`execute_model`, `sample_tokens`), as the kit said.

### 6. Live state, files, rollback
- **Live state (8/8, 20:0x UTC).**
  - Launcher f214c2c9 (unchanged); supervisor 7bea1cf3; manifest d773ed60f27b; site tree 4baa4e458a1a.
  - dsv41-rank active as g5ops-PROD, 0 restarts, /health 200, rank0 log 0 errors.
  - Env: `DSV41_TOPK_CANON_TIES=all`, `DSV41_FAST_LOAD=1`; no `DSV41_AV_COST_JSON`, so AV is unpinned.
  - SM 1969-1989 MHz.
  - spark-os: spark-three 0.9.5-1019-bb1 (promoted, black box active, panic=10/1/1/1); the other seven 0.9.5-1019
    (panic=30/0, unchanged).
- **New files on spark-one:**
  - `~/dsv41-bench/gate/` (gate.sh v2, gate.sh.v1, README.md, runs/);
  - `~/dsv41-g5ops/` (scripts idlemeas.py, ttft_idle.py, g5prof.sh; runs: idle-*.json on every node, ttft-idle-*.json,
    prof-fe, prof-wk);
  - `~/.local/bin/py-spy` and `uv` (tools only).
- **New files on spark-three:**
  - the site-layer files above, `~/kindling-site/README-g5ops.md`, `~/kindling-site.bak-g5-ops.tgz`;
  - `~/sparkos-build-bb1.sh`, `~/sparkos-prebuilt/0.9.5-1019-bb1/` (+ .sha256);
  - `~/sparkos-deploy/{image_check,install,verify}-bb1.sh`;
  - `/images/spark-os-0.9.5-1019-bb1.erofs` and `/boot/sparkos/0.9.5-1019-bb1/` on the DGX OS disk.
- **Leftover /dev/shm segments.** spark-one has 26 unreferenced `psm_*` files in /dev/shm, left by SIGKILLed instances
  since Oct 22. Only ~1 MB is resident, so they are harmless clutter; delete them with the service stopped.
- **Open items:**
  - 1M MemAvailable headroom depends on request history: 5.98 GB after 200K requests.
  - The LAN/tailnet API still has no auth; 192.168.0.86 polls /v1/models every ~2.5 s.
  - Kit 06 lever 1 needs an nsys timeline.
- **Goal: unchanged.** G1 101.81 in the gate's one pass (production 101.5-102); the G4 prose session 48.04 / 48.07
  (per-session min) in this step's two profiled 6-batch G4 runs.
