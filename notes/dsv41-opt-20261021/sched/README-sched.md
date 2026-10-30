
## Short-request scheduling -- PROMOTED (2026-10-24, arm A2 "sk")

**Problem.** When a short chat arrives during a long cold prefill, it waits for the whole prefill: about 14 s p50 at 128K and about 130 s p50 (220 s max) at 1M. There are two causes in vLLM 0.32:
- A running long prefill takes the whole 16384-token budget every step.
- Async scheduling (batch queue of 2) has already scheduled step N+1, a full 16K chunk, before step N finishes. So even a perfect cap leaves a newcomer waiting at least one full step.

**Fix.** `~/dsv41-sched/dsv41_sched.py` (`DSV41Scheduler`, an `AsyncScheduler` subclass) plus `dsv41_sched_policy.py`, which holds the pure policy and is unit-tested. It is loaded through `PYTHONPATH` and `--scheduler-cls`. Nothing in site-packages is edited.
- Only rank 0's EngineCore runs it. On ranks 1-7, `ps` shows `vllm serve --headless` and a Worker but no EngineCore. The files are copied to all 8 ranks anyway, so the layout stays uniform.
- **Short step:** a short request is a waiting request (among the first 32) whose uncached remainder is at most 4096 tokens, measured with a prefix-cache probe. When one is waiting and a long prefill is running, that one step takes the running long prefill out of `running`, admits the short request(s), and caps any waiting request at the short size. The long prefill is put back afterwards and loses only that step.
- **Hold:** while a step of 4096 or more tokens is in flight, or a long prefill is still pending, `has_requests()` reports False. The engine then does not pre-schedule step N+1. A request that arrives during chunk N lands in the step right after N, not after N+1.
- **Lone long prefill:** stock full 16K chunks; the only difference is that they are scheduled synchronously. Decode steps (under 4096 tokens) are never held.
- **Knobs** (env `DSV41_SCHED_<KEY>`, read at start):
  - `LONG_CAP`: launcher 0 = skip the long prefill in a short step; N = give it N tokens instead.
  - `SHORT_MAX`: 4096.
  - `HOLD`: 1; 2 = timed pre-schedule.
  - `HOLD_MIN`: 4096.
  - `DECODE_CAP`: 0.
  - `DSV41_SCHED=0` = stock `AsyncScheduler`.
- **Testing hook:** `DSV41_SCHED_CTL=<json file>` re-reads the params live. It is used only for the in-instance A/B and is not set in the live config. Counters are written to `/dev/shm/dsv41-sched-stats.json`.

**Bench.** `~/dsv41-sched/bench/sbench.py` reuses the r4-e5 `prefill_bench.py` helpers. Each run is one cold long prompt plus 6 x 300-token chats plus one live decoder.
- Chat offsets are 2..17 s at 128K and 10..185 s at 1M. The 1M prompt is 998,748 tokens.
- `solo` = the same long prompt with nothing else running.
- In-instance arm switching is done by `abrun.py`. Summaries: `ssumm.py` (TTFT) and `dsumm.py` (decode).

### Results (live 1M config; 4 instances: i1 sched v1, i2 sched v2, i3 stock flag A1, i4 promoted launcher)

| arm | 128K short p50 / p95 s | 128K long s | 1M short p50 / p95 / max s | 1M long s | decoder max gap 128K / 1M |
|---|---|---|---|---|---|
| stock (off; i1, i2, i4, base0) | 14.0 / 21.5 | 23.1 | 129-133 / 208-212 / 220 | 226.3, 230.2 | 2.8-3.0 / 5.0-5.2 s |
| A1 `--long-prefill-token-threshold 8192` (i3, real flag) | 4.12 / 4.79 | 26.26 (+13.5%) | 5.03 / 6.48 / 6.73 | 262.9 (+15%) | 1.8 / 4.3 s |
| A1 emu 8192 (i1, same code path via ctl) | 4.04 / 4.74 | 26.16 | 4.97 / 5.6 | 262.4 | 1.7 / 4.2 s |
| A1 emu 4096 (i1) | 2.83 / 2.91 | 32.06 (+39%) | - | - | 1.1 s |
| A2 cap 2048 + hold (i1, i2) | 1.94-1.96 / 3.2 | 25.2 (+9%) | 3.87 / 5.6 | 235.0 | 2.8 / 5.3 s |
| A2 cap 4096 + hold (i1) | 2.73 / 3.67 | 25.0 | 4.10 / 5.3 | 239.4 | 2.8 / 5.3 s |
| A2 cap 8192 + hold (i1) | 3.40 / 4.26 | 24.3 | - | - | 2.8 s |
| A2 cap 2048, no hold (i1) | 3.84 / 5.51 | 25.3 | - | - | 2.8 s |
| A2 cap 2048 + decode cap 8192 (i1) | 1.50 / 1.97 | 27.2 (+18%) | - | - | 2.7 s |
| **A2 skip + hold = "sk" (i2, i4; promoted)** | **1.73-1.80 / 2.2** | **24.9 (+8%)** | **2.32 / ~4.0 / 5.19** (12 chats, 2 instances) | **231.2, 232.5 (+1.6%)** | 2.8 / 5.1 s |
| A2 skip + timed hold (hold=2; i2) | 1.70 / 2.17 | 24.8 | 2.65 / 4.06 / 4.27 | 229.9 | 2.8 / 5.0 s |

Long-prompt deltas compare against stock starve runs with a decoder present: 23.1 s at 128K, and 228.3 s at 1M (mean of i1 and base0).

**Solo long prompt** (nothing else running; ABBA order within an instance):

| | 128K s | 500K s | 1M s |
|---|---|---|---|
| stock | 22.36 (i1 x5), 22.37 (i2 x4), 22.25 (i4 x2), 22.27 (A1 i3 x3) | 93.2 (x4) | 226.5 (i1); 218.0, 230.9 (i2); 224.2, 221.3 (i4): mean 224.2 |
| sk (hold=1) | 21.74 (i1 x5, cap 2K), 21.76 (i2 x4), 21.72 (i4 x2): **-2.7%** | 90.4 (x4): **-3.0%** | 233.2 (i1, cap 2K); 223.9, 235.1 (i2); 224.6, 218.5 (i4): mean 227.0, **+1.3%** |
| hold=2 | 21.86 (-2.3%) | 93.4 (+0.2%) | 220.3, 238.2 |

- The 1M solo spread is ±4% even within one arm, because a thermal and run-order drift adds up over the 3.7-minute run. Pooled over 5 runs per arm it is +1.3%, inside the +2% gate. At 128K and 500K, synchronous long chunks are 2.4-3% faster than async pre-scheduling, measured 4-5 times per arm.
- hold=2 (pre-schedule 300 ms before the predicted end) works: its counters show it pre-scheduled 34 of 44 steps in a solo run. It behaves like stock on solo and is no better on shorts, so it was not chosen.

**What it costs and what it doesn't fix.**
- Each short step costs the long prompt about 0.3 s, because one extra eager step runs. At 128K, with 6 chats in 23 s, that adds up to +8%. At 1M it is +1.6%.
- Chats that arrive late in a 1M prefill still wait out the current chunk, which takes about 5 s near 1M context. The TTFT is about (remaining time of the current chunk) + 0.1-0.6 s, which is where the 5.2 s max comes from.
- The decoder's inter-token gap during a long prefill is unchanged (one chunk, 2.8 s at 128K, 5 s at 1M). `DECODE_CAP` exists but costs +18% on the long prompt at 8192, so it is off.

**Decode non-regression** (i4, sk vs off ABAB in one instance, matched C1 + bench.py):

| | coding ms/step | prose ms/step | single code/prose/math tok/s | conc8 tok/s | conc32 tok/s |
|---|---|---|---|---|---|
| off (2 suites) | 34.40 / 34.73 | 29.89 / 30.18 | 84.6/67.9/115.9-116.0 | 216.2, 218.5 + 218.5, 218.6, 218.1 (mean 218.0) | 453.0 / 452.3 |
| sk (2 suites) | 34.60 / 34.58 | 29.25 / 29.71 | 84.7/67.3/116.3, 84.6/68.0/115.8 | 214.1, 214.0 + 215.1, 219.3, 216.0 (mean 215.7, **-1.1%**) | 453.5 / 454.8 |
| A1 i3 (thr8k flag, 1 suite) | 34.53 | 28.79 | 84.3/67.2/116.3 | 214.5 | 452.4 |

- Every metric is within the run-to-run noise: C1 is ±0.8%, and conc8's A/A interval in the wake study was ±2.1%. Decode steps are never held or capped.

**Correctness** (i4, sk):
- exact.py a8c3394af9d2.
- GSM8K-50 conc8: 50/50 (237 tok/s).
- longctx default gate: PASS (needles 60/60, copy 1.000).

**Promotion.**
- Launcher `~/dsv41-tp8.sh` 80c749ca -> **f4e4e214** on 8/8. Backup: `~/dsv41-tp8.sh.bak-sched`.
- The launcher adds `DSV41_SCHED=${..:-1}`, `DSV41_SCHED_LONG_CAP=${..:-0}`, `DSV41_SCHED_HOLD=${..:-1}`, prepends `$HOME/dsv41-sched` to `PYTHONPATH`, and appends `--scheduler-cls dsv41_sched.DSV41Scheduler` when `DSV41_SCHED=1`.
- Files on 8/8: `~/dsv41-sched/dsv41_sched.py` 6074389f and `dsv41_sched_policy.py` 1e445cd9. Also there: `test_dsv41_sched_policy.py`, the offline policy tests plus an engine-loop simulation, ALL PASS.
- Not promoted: A1 (stock flag). It is 4.1 s / 5.0 s p50 and costs +13.5% / +15% on the long prompt.

**Rollback** (from the Mac):
- Env only: `bash ~/dsv41-opt-20261021/ops/conc/launch-conc.sh DSV41_SCHED=0`. This gives the stock AsyncScheduler with no module loaded.
- Launcher: `cp -p ~/dsv41-tp8.sh.bak-sched ~/dsv41-tp8.sh` on all 8 (back to 80c749ca), then relaunch. `~/dsv41-sched/` can stay; it is inert without the flag.

**Final live relaunch** (2026-10-24 07:32 UTC, launcher f4e4e214, no ctl hook): health 200, max_model_len 1,048,576, `dsv41_sched armed` with long_cap 0 / hold 1, exact.py a8c3394af9d2, starve128 short p50 1.99 s (max 2.44) with the long prompt at 25.4 s.

Raw results: `./sched/runs/{base0,i1,i2,i3_thr8k,i4,final}/`. Each holds starve*/solo* jsonl and `log.txt`; `stats.jsonl` has per-step scheduler counters and temperatures; `d_*` and `c_sk` are the dsuite outputs. Tools are in `~/dsv41-sched/bench/`.
