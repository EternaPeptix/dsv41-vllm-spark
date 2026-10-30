
## pd base step (2026-10-25 18:40-21:00 UTC): prompt-tokens-details ON, startup hang qualified, relaunch diet. PROMOTED

User request (2026-10-25): keep `--enable-prompt-tokens-details` on going forward, then improve prefill and decode. This base step did three things: (1) turned the flag on permanently, (2) qualified Codex's C02 hang fix and the rel-2 graph-safety patch, (3) shipped the relaunch diet. Raw results: spark-one `~/dsv41-pd-20261025/` (runs/<label>/ per launch: startup.json, exact.json, exact_spec.txt, matched.json, usage.txt, gsm8k50.jsonl, longctx.jsonl). Supervisor logs and the launch table: Mac scratchpad `pd/` (`launches.tsv`, `table_all.md`, `runs/<label>/{sup.log,ranks.txt,result.txt,mem.txt}`).

**Live state after this step** (sha256 identical on 8/8):
- Launcher `~/dsv41-tp8.sh` **b1e75497**. Backups: `.bak-pd-rel` = 2fd26e53 (flag on, rel-2 defaults off), `.bak-pd-ptd` = 2027f4d8 (pre-step).
- Site files (vllm/...): `v1/worker/gpu/model_runner.py` **316fca44** (was 6f9d997f = Codex C02), new `v1/worker/gpu/dsv41_rel.py` **f373f911**, `distributed/device_communicators/rdma_all_reduce.py` **9056dcdf** (was acbeeef7), `models/deepseek_v41/nvidia/dspark.py` **36811ca9** (was fa56d07d). Backups `<file>.bak-pd-rel2` hold the old versions. Staged copies and tests: `~/dsv41-stage/rel2/` on 8/8.
- spark-one ops: `dsv41-launch-all.sh` e779b77e and `conc/launch-conc.sh` 698e7943 now pass `-p TimeoutStopSec=20` (`.bak-pd-rel` = bbdacd0b / 6842bdcc). New supervisor `ops/rel/dsv41-relaunch.sh` 39679a35 (+ relstatus.sh 023e5a8d, relsumm.py 46eac2fe, replay_detector.py, shardfilter_check.py, test_dsv41_rel.py).
- AV profile stays **unpinned** by default (startup-profiled, as before). See the pin-policy proposal below.

### 1. `--enable-prompt-tokens-details` (always on)
Added to the common `vllm serve` ARGS (frontend-only flag; headless ranks accept it; no engine or compile-cache change). Checked on the first instance (pd-L0):
- before (flag off): `"prompt_tokens_details": null`.
- chat, first call of a 4,205-token prompt: `{"cached_tokens": 0, "created_cache_tokens": 4160}`; same prompt again: `{"cached_tokens": 4160, "created_cache_tokens": 0}`. Streaming usage and /v1/completions carry it too (completions repeat: cached 4160).
- exact.py on that instance (pinned live-2055 AV profile): **a8c3394af9d2**, spec counters identical to every later launch.

### 2. Startup hang (rdma_ar graph-capture race): root cause confirmed, C02 + rel-2 graph safety qualified
rel-2 (all flags are env; defaults now set in the launcher):
- `DSV41_CAPTURE_SYNC` (code default 1) keeps Codex C02's device sync between the decoder-replay capture and the DSpark capture; 0 restores the pre-C02 order (test only).
- `RDMA_AR_XSTREAM=wait` (launcher default): an eager rdma_ar call that arrives on a new stream while the previous eager stream is still busy makes the new stream wait for it (GPU-side `wait_stream`, never inside a capture). `detect` only counts/logs.
- `RDMA_AR_FINGERPRINT=1`: ranks compare a hash of the rdma env, rdma_all_reduce.py, the .so and the l2 prefetch module at init and refuse to start on a mismatch.
- `DSV41_CAPTURE_WATCHDOG_S=90`: if the decoder-replay or DSpark capture phase exceeds 90 s, faulthandler dumps all stacks to the rank log ("Timeout (0:01:30)!") and the worker exits (AV phase budget 420 s). Replaces the silent 600 s rdma_ar trap at startup.
- Supervisor `ops/rel/dsv41-relaunch.sh` (run from the Mac): pre-flight sha/manifest + Engram check on 8/8, stop with a 30 s cap then SIGKILL, start with TimeoutStopSec=20, watch rank0 every 5 s, fail a try on rdma timeout / watchdog / EngineCore failure / fingerprint mismatch / DSpark stall 120 s / budget, save all rank logs as `~/dsv41-logs/relfail-<stamp>-t<try>-rank<i>.log`, SIGKILL and retry (default 2 tries).

**Forced-skew reproduction** (rank 3 sleeps 300 ms before each of the 6 decoder-replay graphs; pinned AV; diet on):

| arm | C02 sync | xstream | result | xstream counters |
|---|---|---|---|---|
| S0d | off | detect | **hang on 8/8 ranks**, watchdog fired 90 s after DSpark start (supervisor fail at +342 s) | first DSpark eager all_reduce (160x5120) issued while the replay stream was still busy |
| S0w | off | wait | healthy +283 s, exact a8 | busy=1 waits=1 on the 7 fast ranks, 0 on the delayed rank 3 |
| S1 | on | detect | healthy +286 s, exact a8 | busy=0 on 8/8 |

So the hang is the decoder-replay stream overlapping DSpark's first eager rdma_ar, exactly as diagnosed in round-2 #2. C02 alone removes it, and the xstream guard alone also removes it.

**Fail-fast end-to-end** (pd-H-GD, `DSV41_TEST_CAPTURE_HANG=3` on try 1 only): rank 3 blocked at the DSpark capture; all ranks' watchdogs fired at 90 s; the supervisor flagged it at +341 s (88 s after DSpark start), saved logs, killed, relaunched; try 2 healthy at +281 s, exact a8. A hang now costs ~12 min to healthy with no operator (was ~23 min: 600 s trap + 90 s stop timeout + manual relaunch).

**Campaign:** 14 consecutive production-config launches (no fault hooks), all healthy on their first try, **0 hangs**. The 13 pinned ones all gave exact.py **a8c3394af9d2** with identical spec counters (accepted 901 / drafts 473 / draft tokens 2365 / generated 1373); the unpinned FINAL also gave a8 (different counters, as expected from its own AV profile). Every rel-2 launch reported xstream switches=4 busy=0 waits=0: C02 prevented every overlap and the guard never had to act. With Codex's 8 clean post-C02 startups that is 23 consecutive clean launches with C02 on (14 here + S1 + Codex's 8) against the old ~12% hang rate (P = 0.88^23 = 0.05), on top of the decisive skew test.

| # | label | arms | try | healthy s | downtime s | AV phase s | exact | C1 coding / prose ms |
|---|---|---|---|---|---|---|---|---|
| 1 | pd-L0 | flag only, pin | 1/2 | 396 | 413 | 98 | a8 | 33.12 / 28.71 |
| 2 | pd-L1-G | + graph safety | 1/2 | 410 | 426 | 103 | a8 | 33.46 / 28.72 |
| 3 | pd-L2-GD | + diet | 1/2 | 281 | 298 | 22 | a8 | 33.62 / 28.25 |
| - | pd-S0d | skew, sync off, detect | 1/1 | fail (watchdog) | - | - | - | - |
| - | pd-S0w | skew, sync off, wait | 1/1 | 283 | 292 | 20 | a8 | - |
| - | pd-S1 | skew, sync on, detect | 1/1 | 286 | 303 | 21 | a8 | - |
| - | pd-H-GD try 1 | hang hook | 1/2 | fail (watchdog) | - | - | - | - |
| 4 | pd-H-GD try 2 | G+D | 2/2 | 281 | 289 | 20 | a8 | - |
| 5 | pd-L3-G | G | 1/2 | 386 | 403 | 98 | a8 | 33.30 / 28.61 |
| 6 | pd-L4-GD | G+D | 1/2 | 288 | 304 | 21 | a8 | 33.35 / 28.66 |
| 7 | pd-L5-G | G | 1/2 | 386 | 402 | 98 | a8 | 33.27 / 28.63 |
| 8-13 | pd-L6..L11-GD | G+D | 1/2 each | 276, 281, 284, 285, 287, 287 | 292-304 | 19-24 | a8 x6 | 33.26-33.74 / 28.32-28.73 |
| 14 | pd-FINAL | launcher b1e75497 defaults, unpinned (live now) | 1/2 | 356 | 373 | 98 (pin miss, profiled) | a8 (own profile; counters 911/465) | 33.38 / 28.93 |

### 3. Relaunch diet (round-2 #4): PROMOTED
- **Arm S, `DSV41_DSPARK_SHARD_FILTER=1`:** DSpark's model class names every non-`mtp.N.*` checkpoint weight unused, so vLLM's default loader skips the 45/48 shards with no draft tensor (header scan). Draft load 31-36 s -> **4.0-4.2 s**. The same mtp tensors arrive in the same order. Applies to every launch.
- **Arm P, `DSV41_AV_PROFILE=pinned`:** when `DSV41_AV_COST_JSON` is set and its raw curves cover this shape set (shape_key: capture sizes, max_num_seqs 32, max_num_batched_tokens 16384, num_spec 5, TP 8), rank 0 broadcasts a hit; every rank runs one untimed replay of each profiled size (37 runs instead of 185: JIT + 16K workspace growth before lock_workspace) and the unchanged set_cost_curves path prices the pin. AV phase 98-103 s -> **19-24 s**. Without a pin (the default today) it logs "pin miss" and profiles as before.
- **Time to healthy (systemd-run -> /health 200), pinned launches:** diet off 386-410 s (mean 394.5, n=4) vs diet on 276-288 s (mean **283.5**, n=11): **-111 s (-28%)**; downtime 402-426 s -> 289-304 s. Graph-capture phase 113-121 s -> 39-40 s. Unpinned default launches get arm S only: pd-FINAL 356 s vs 386-410 s (-10%).
- **Exactness:** diet on and off give a8 with identical spec counters (table above). Startup memory observations after AV are identical (reserved 80,304,144,384 B before trim, 72,347,549,696 B after), MemAvailable 17-18 GiB healthy on both arms and 11-12 GiB after the longctx runs on both arms. Quick longctx 16K/32K PASS on a diet-on (L2) and a diet-off (L3) instance.
- **Perf:** startup-only; matched C1 means diet off 33.29 / 28.67 ms, diet on 33.48 / 28.52 ms (coding/prose), opposite signs and inside the per-launch spread; baseline 33.57 / 28.62.

**Gates** (promoted config + pin, pd-L11-GD): GSM8K-50 **49/50** (miss idx 12, 13 vs 12; conc-8 GSM8K is not bit-reproducible across instances), longctx full **PASS** needle 60/60, copy 1.000, wall 174 s. Final unpinned instance (live): GSM8K-50 **50/50**, exact.py **a8c3394af9d2** (its fresh profile happened to land on the anchor), usage carries prompt_tokens_details, 0 ERROR/Traceback lines, fingerprint identical on 8 ranks, xstream busy 0.

### Proposal (not applied): default AV pin
Setting `export DSV41_AV_COST_JSON=${DSV41_AV_COST_JSON-$HOME/dsv41-pins/av-pin-live2055.json}` in the launcher (copy of `~/dsv41-stage/rel2/av-pin-live2055.json`, the raw curves of the 10-24 20:55 instance) would make every launch (a) skip the AV profile (~395 -> ~285 s), (b) reproduce the anchor a8c3394af9d2 on every restart (no 1-in-9 flips to 9330f03c4f97 / b7e868c9d700), and (c) drop the ~3% boot-to-boot spread from re-profiled curves. Costs: a stale pin after a kernel change mis-prices verify widths (throughput/AL risk, not correctness); A/Bs sharing one pin are one profile draw. Refresh rule: after any promoted change that alters step cost, launch once with `DSV41_AV_COST_JSON=` (empty = profile), check exact + matched, then `mkpin.py` + add shape_key. A changed capture/batch/spec/TP shape misses automatically and profiles. **User decision.**

### Launching now
- Supervised (recommended): `ssh spark-one cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh | bash -s -- --label <x> [--expect-launcher b1e75497] [VAR=VAL ...]` (exit 0 healthy, 1 failed after retries, 2 pre-flight refused).
- Pinned A/B arm: add `DSV41_AV_COST_JSON=/home/jeweled/dsv41-stage/rel2/av-pin-live2055.json` (absolute path; diet hit is automatic).
- The old `ops/dsv41-launch-all.sh` / `ops/conc/launch-conc.sh` still work (now with TimeoutStopSec=20) but do not retry.

### Rollback
- rel-2 defaults off, env only: `launch-conc.sh DSV41_CAPTURE_WATCHDOG_S=0 RDMA_AR_XSTREAM=off RDMA_AR_FINGERPRINT=0 DSV41_AV_PROFILE=off DSV41_DSPARK_SHARD_FILTER=0` (code paths then equal Codex C02 live).
- Launcher: `cp -p ~/dsv41-tp8.sh.bak-pd-rel ~/dsv41-tp8.sh` (flag kept, rel-2 defaults off) or `.bak-pd-ptd` (pre-step) on 8/8.
- Site files (service stopped): for f in model_runner.py rdma_all_reduce.py dspark.py, `cp -p <f>.bak-pd-rel2 <f>`; `rm v1/worker/gpu/dsv41_rel.py`; expect 6f9d997f / acbeeef7 / fa56d07d on 8/8.
- Ops scripts: `cp -p <script>.bak-pd-rel <script>` on spark-one.
