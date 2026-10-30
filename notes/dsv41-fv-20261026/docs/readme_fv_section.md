

## Prefill+decode window (2026-10-25 18:40 - 2026-10-26 07:00 UTC): final verification and rollbacks

The window's steps, each documented in its own section above, were: pd base step, pd decode A/B step, pd long-context X-ray, pf prefill A/B step and pd Marlin MoE measurement. This final verification ran on the live instance with no restart: mm-FINAL, started 06:15:01 UTC and healthy 06:20:52. The suite ran from 06:37:36 to 06:53:20 UTC. A 2 s metrics sampler showed no outside requests during the suite: the running count never exceeded the suite's own concurrency.

**Result.** The server runs launcher **04b6738e** on 8/8 with its defaults and an unpinned AV profile. `--enable-prompt-tokens-details` is active. exact.py gives **a8c3394af9d2** in all three runs of the suite (two at the start, one after the full load), each with spec counters 902/474/2370/1373; the Marlin step's two runs on this instance also gave a8. Compared with the window's baseline (live 2027f4d8, takeover audit 2026-10-24):
- Decode: the 1-user step is **4.3% faster for coding and 3.7% for prose** (same prompts as the baseline). Single-stream tok/s is **+3.1 to +4.4%**, and aggregate tok/s at 4-32 users is **+3.4 to +5.8%**. This matches the pinned A/B of the decode step (D05 + L2 prune-shared: -4.3% / -4.0%, conc16/32 +5.1/+5.4%).
- Prefill: **unchanged.** Every prefill candidate changed output, was not yet bitwise, or crashed, so none was promoted. 16K and 128K TTFT, starvation, and the 1M chunk times up to ~500K tokens match the baseline.
- 1M on this instance: **229.1 s**. This launch is in the slow class: chunks 37-62 run at 4.0-4.9 s instead of ~3.1-3.4 s. The X-ray step traced that class to a rank-5 (spark-six) straggler. Good-class launches give ~182 s.
- Startup: a capture hang is now detected and relaunched automatically. Unpinned launches take 351-373 s to healthy, down from 386-410 s.

### Final measurement vs BASELINE (one instance, mm-FINAL, unpinned AV)

| metric | BASELINE (live 2027f4d8) | FINAL (04b6738e) | change |
|---|---|---|---|
| matched C1 coding ms/step (label live2055 = the baseline's prompts) | 33.574 (n=5, AL 5.13) | **32.116** (n=10, sd 0.13, AL 5.226) | **-4.3%** |
| matched C1 prose ms/step (label live2055) | 28.620 (AL 2.70) | **27.567** (sd 0.34, AL 2.729) | **-3.7%** |
| matched C1 coding / prose, label pdab (pd-era prompts) | 33.555 / 28.680 (pd pre-step, pinned) | 32.254 / 27.545 (AL 5.165 / 2.700) | -3.9% / -4.0% |
| single code tok/s (1024 tok x2) | 87.90 | **90.93** (91.06, 90.80; AL 2.554) | **+3.4%** |
| single prose tok/s | 69.66 | **72.73** (72.77, 72.70; AL 1.809) | **+4.4%** |
| single math tok/s | 120.85 | **124.63** (124.57, 124.70; AL 3.531) | **+3.1%** |
| conc4 aggregate tok/s (per-request mean / min) | 153.9 (50.2 / 40.6) | **162.8** (163.3, 162.4; 54.5 / 41.2) | **+5.8%** |
| conc8 | 228.8 (37.2 / 30.0) | **238.9** (239.5, 238.4; 39.9 / 30.2) | **+4.4%** |
| conc16 | 336.0 (28.5 / 21.9) | **354.2** (353.1, 355.2; 29.9 / 22.7) | **+5.4%** |
| conc32 | 497.2 (20.9 / 15.6) | **514.1** (515.6, 512.5; 21.9 / 16.9) | **+3.4%** |
| cold solo TTFT 16K s | 16K chunk 2.71-2.81 | 2.709 (2.758, 2.629, 2.739) | flat |
| cold solo TTFT 128K s | ~22 (sk solo 21.72-21.76) | 21.785 (21.820, 21.749) | flat (+0.2%) |
| cold solo TTFT 1M s (1,024,611 tokens) | ~205-228 (sk mean 227.0) | 229.1, **slow-class launch** | flat (+0.9%); good-class launches give ~182 |
| 1M chunk s: c2-12 / c20-30 / c37-43 / c50-62 | good ~2.7 / ~3.0 / ~3.1 / ~3.4; slow up to 5.0-5.2 | 2.687 / 2.982 / 4.044 / 4.907 | slow class |
| 128K starvation: long TTFT / short p50 / p95 / max s | 24.9 / 1.73-1.80 / 2.2 (sk) | 24.936 / 1.733 / 2.051 / 2.087; decoder gap p50 420 ms, max 2.80 s | flat |
| exact.py | a8c3394af9d2 | **a8c3394af9d2 x3**, counters 902/474/2370/1373 each time | exact |
| GSM8K-50 (conc 8) | gate >= 49/50 | **49/50**; the miss is idx 12 (13 vs 12), the same borderline item that missed on pd-L11-GD | pass |
| longctx (full) | PASS | **PASS**, needle 60/60, copy 1.000 | pass |
| usage `prompt_tokens_details` | null (flag off) | 4,205-token chat: cached_tokens 0 (created 4160), then 4160 on the repeat; streaming chat 4160; /v1/completions 0 -> 4160 | on |

Notes:
- **AV profile.** BASELINE and FINAL are both unpinned instances, each with its own AV profile, so acceptance length differs a little (coding AL 5.13 vs 5.226). The pinned same-AL decode A/B is in "pd decode A/B step".
- **Instance health.** errors 0 at start and end. Both rank 0 and the API stayed healthy throughout. MemAvailable is 12-13 GiB per node after longctx (base step: 11-12 GiB after longctx).
- **Raw data.** spark-one `~/dsv41-fv-20261026/`: `runs/FINAL/` (all outputs, `sampler.txt`, `summ.json`, `table.md`), `scripts/fsuite.sh`, `fsumm.py`, `vcheck.sh`, and `pre/` / `post/` (static checks before and after the suite). The Mac scratchpad `fv/` holds pre/post checks and drafts.

### Live config, verified 2026-10-26 06:30-06:37 UTC and again after the suite (no restart; instance mm-FINAL, healthy since 06:20:52)
- **Launcher** `~/dsv41-tp8.sh` **04b6738e** on 8/8, with `--enable-prompt-tokens-details` in the common ARGS and in the running `vllm serve` cmdline on 8/8.
- **Env on 8/8** (launcher defaults, no test variables, AV unpinned): `DSV41_MOE_FC2_D05=1 DSV41_L2_PREFETCH_PRUNE_SHARED=1 DSV41_CAPTURE_WATCHDOG_S=90 RDMA_AR_XSTREAM=wait RDMA_AR_FINGERPRINT=1 DSV41_AV_PROFILE=pinned DSV41_DSPARK_SHARD_FILTER=1 DSV41_RDMA_PARLOAD=1 DSV41_TOPK_CANON=1 DSV41_SCHED=1`; `DSV41_AV_COST_JSON` unset (startup profile, logged as "pin miss").
- **Site files, identical on 8/8:** model.py e7264781, model_state.py 53e4fe39, decoder_replay_layers.py 2fedc0a6, dspark/speculator.py bbcee9e3, moe_runner.py 20b69fe5, deep_gemm_utils.py a4a0486a, sm121_target_m32.py 5acc5fde, sm121_d05_verify.py c04c363d, dsv41_l2_prefetch.py 514dcb23, fp8_utils.py b35c0561 (pdl, inert), model_runner.py 316fca44, dsv41_rel.py f373f911, rdma_all_reduce.py 9056dcdf, dspark.py 36811ca9, gate_linear.py 41adaba8 (pristine); no dsv41_rgate.py; scheduler `~/dsv41-sched/dsv41_sched.py` 6074389f, no dsv41_pd_policy.py. Relaunch manifest ad97d72f180e on 8/8.
- **Backups, identical on 8/8** (18 files, listing hash ae661a994ec7): launcher `.bak-pd-ptd` 2027f4d8, `.bak-pd-rel` 2fd26e53, `.bak-pd-d05` b1e75497, `.bak-pd-l2pfS` 92dbac8e; `model_runner.py.bak-pd-rel2` 6f9d997f, `rdma_all_reduce.py.bak-pd-rel2` acbeeef7, `dspark.py.bak-pd-rel2` fa56d07d, `sm121_target_m32.py.bak-pd-d05` 723fdf8d, `dsv41_l2_prefetch.py.bak-pd-l2pf` 4da7745b, `fp8_utils.py.bak-pd-pdl` 8dc7dab8, `gate_linear.py.bak-pd-rgate` 41adaba8, `model.py.bak-pd-spg` fe7a7d12, and `.bak-pd-pf` copies of model.py / model_state.py / decoder_replay_layers.py / speculator.py / moe_runner.py / deep_gemm_utils.py (each equal to the live file). spark-one ops: `dsv41-launch-all.sh` e779b77e (`.bak-pd-rel` bbdacd0b), `conc/launch-conc.sh` 698e7943 (`.bak-pd-rel` 6842bdcc), `rel/dsv41-relaunch.sh` 39679a35, `rel/relstatus.sh` 023e5a8d, `rel/relsumm.py` 46eac2fe.
- **Health:** units active on 8/8, 0 restarts, 0 ERROR/Traceback lines, rdma_ar fingerprint c2f008432bd4769f on 8 ranks, Engram check OK 8/8 (spark-four whole, the other 7 punched), D05 native + libdsv41_l2pf + target-m32 native mapped on 8/8, max_model_len 1,048,576.

### What the window changed (all exact: identical exact.py output and spec counters under a pinned AV profile)

| change | switch (launcher default) | measured in its step |
|---|---|---|
| `--enable-prompt-tokens-details` | common `vllm serve` ARGS, always on (user request) | usage carries `prompt_tokens_details` (`cached_tokens`, `created_cache_tokens`); frontend only |
| rel-2 graph safety: Codex C02 sync (in code, `DSV41_CAPTURE_SYNC=1`), capture watchdog, xstream guard, rank fingerprint, `TimeoutStopSec=20`, supervisor `ops/rel/dsv41-relaunch.sh` | `DSV41_CAPTURE_WATCHDOG_S=90`, `RDMA_AR_XSTREAM=wait`, `RDMA_AR_FINGERPRINT=1` | startup-hang root cause confirmed by a forced-skew test; a hang is now detected in ~90 s and relaunched automatically. 45 supervised starts in the window: 41 healthy, 0 unintended capture hangs (the 4 failures were 2 deliberate fault injections and 2 gather-add startup crashes) |
| relaunch diet | `DSV41_DSPARK_SHARD_FILTER=1`, `DSV41_AV_PROFILE=pinned` (the AV skip only engages with `DSV41_AV_COST_JSON`) | draft load 31-36 s -> 4 s; time to healthy 386-410 s -> 351-373 s without a pin, 276-288 s with one |
| D05: skip fully dead 8-row FC2 output-store groups (Codex D05 native) | `DSV41_MOE_FC2_D05=1` | pinned pairs: C1 coding -2.50% / -2.61%, prose -1.72% / -2.25%; conc16/32 +3.09/+4.51% and +5.40/+5.96% |
| L2 prefetch arm S: drop shared-expert hints, router hints backfill | `DSV41_L2_PREFETCH_PRUNE_SHARED=1` | pinned pairs on top of D05: C1 coding -2.05% / -1.71%, prose -2.11% / -2.06% |
| pdl (`fp8_utils.py` b35c0561) | installed, inactive with its flags off | HOLD: -0.33% / -0.16% C1, below the gates |

Not promoted: rgate (killed, slower than cuBLAS), l2pf PB / SQ, E7 tail skip (changes output: replay-layer indexer-K writes), SP v2 exact (not bitwise at 12.7-14.2K rows yet), gather-add fold (startup crash as staged), NCCL arms (closed), P09 and lone-24K chunks (not run), Marlin MoE (measurement only; non-exact).

### Rollbacks (newest first; launcher and site changes need all 8 nodes, site files only with the service stopped)
Env-only rollbacks are supervisor `VAR=VAL` arguments and last until the next default launch. Launcher backups are cumulative: each `.bak-pd-<tag>` is the launcher from before that step, so restoring an older one also drops every later default.
- **L2 prune-shared:** env `DSV41_L2_PREFETCH_PRUNE_SHARED=0`; launcher `cp -p ~/dsv41-tp8.sh.bak-pd-l2pfS ~/dsv41-tp8.sh` (92dbac8e = D05 only); file `ssh spark-one cat ~/dsv41-stage/pd/common/pd-all.sh | bash -s uninstall l2pf` (back to 4da7745b).
- **D05:** env `DSV41_MOE_FC2_D05=0`; launcher `.bak-pd-d05` (b1e75497 = pre-decode-step) first, then `pd-all.sh ... uninstall d05` (sm121_target_m32.py back to 723fdf8d, sm121_d05_verify.py removed).
- **pdl (inert):** `pd-all.sh ... uninstall pdl` (fp8_utils.py back to 8dc7dab8).
- **Relaunch diet and rel-2 guards, env only:** `DSV41_DSPARK_SHARD_FILTER=0 DSV41_AV_PROFILE=off DSV41_CAPTURE_WATCHDOG_S=0 RDMA_AR_XSTREAM=off RDMA_AR_FINGERPRINT=0` (code paths then equal Codex C02 live). Files: `model_runner.py`, `rdma_all_reduce.py`, `dspark.py` from `<file>.bak-pd-rel2` (6f9d997f / acbeeef7 / fa56d07d), `rm v1/worker/gpu/dsv41_rel.py`; launcher `.bak-pd-rel` (2fd26e53); spark-one ops `dsv41-launch-all.sh` / `conc/launch-conc.sh` from `.bak-pd-rel` (bbdacd0b / 6842bdcc).
- **prompt-tokens-details:** delete the `--enable-prompt-tokens-details)` line from ARGS in `~/dsv41-tp8.sh` (move the closing parenthesis to the line above) on 8/8 and relaunch; frontend only, no exactness effect. Full pre-window launcher: `.bak-pd-ptd` (2027f4d8).
- Every relaunch: `ssh spark-one cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh | bash -s -- --label X [VAR=VAL ...]` from the Mac (waits for healthy, auto-retries one hang).

### Open items and user decisions (nothing below is applied)
- **Default AV pin** (proposed in the base step, unchanged): `export DSV41_AV_COST_JSON=${DSV41_AV_COST_JSON-$HOME/dsv41-stage/pdab/av-pin-pdfinal.json}` in the launcher. It gives a deterministic a8 anchor on every restart, ~285 s launches instead of ~350-370 s, and better odds against the rank-5 long-context straggler (X-ray: slow on 5/5 unpinned and 0/2 pinned launches; pf step: pinned 6 good / 2 slow, unpinned 1 good / 4 slow). Cost: the pin must be refreshed after kernel changes (launch once with the variable empty, then `mkpin`).
- **Unpadded Marlin MoE (munpad):** about -3.2 ms/step at 1 user and -14 to -17 ms/step at 16 users, but not bitwise (MoE output 3.2% rel L2 vs live, about 7x closer to FP32) and up to 6% slower 16K-chunk prefill. Needs a new anchor and full quality gates if wanted. Stock `--moe-backend marlin` is a NO-GO.
- **Exact leads, in expected-value order:** (1) M32 FC1/FC2 weight pipelining: the live kernels reach ~170-194 GB/s vs Marlin's 211-234 GB/s on the same bytes (ma64 1.18-1.26x); (2) SP v2: find the second row-count-dependent kernel (bitwise from 15,022 rows, not at 12,760-14,241) with a layer bisect, worth ~6% TTFT 16K-1M; (3) an E7 variant that keeps the replay indexer-K writes; (4) gather-add fold: peek `_output[_output_idx]` instead of the consuming `output` property (kernel 0.78x at 16K rows); (5) a relaunch-if-slow 700K screen against the rank-5 straggler.
- **Known, pre-existing:** long prompts (above ~4.1K tokens) are not run-to-run deterministic within one instance (stock too), and exact.py can depend on request history; run exact.py first on a fresh instance when judging a change.
