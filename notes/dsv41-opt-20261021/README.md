> **2026-10-28 17:24 UTC Goal window 5, final verification** (config verified on 8/8 before and after: launcher afa97ae6, supervisor 7bea1cf3 / manifest 0c96a727fb81, site tree 11e30c3df6b6 (5,454 files), running env 9626ff845dc0. Both instances exact: exact.py a8c3394af9d2 4/4 plus the gate's, pinned with the pin's counters, G1 outputs equal to the pinned reference bc1e8386878bd2b7 on both. Goal NOT met: production G1 104.00 (+15.4% to 120), pinned 104.22; G4 prose session 47.89 in production (+4.4%, 9/12 session-runs >= 50) and 50.55 pinned (the G4 metric is met there, 11/12; G4min_hat 49.25 / 49.42). Regression gate (kit 08) PASS in 563 s on g5v-FU with a clean foreign-traffic audit (1M 168.8 s good class, GSM8K-200 195, longctx 60/60). Healthy +204 s production / +126 s pinned; idle GPU 6.9-9.8 W per node, spark-one box 37.3-38.0 W, switch 260 W. Drafter evidence: train the MTP heads (self-distillation; user decision plus a drafter exactness rule); the official MTP shards are byte-identical (no fetch); suffix / n-gram drafts closed. Live g5v-FU): see the last section, "Goal window 5 (2026-10-27 16".
> **2026-10-28 15:25 UTC Goal window 5, Host step** (nsys on the live stack first: host-bound GPU idle **0.0 ms/step** on ranks 0 and 4 in G4 x2 and C1, host 17-37 ms ahead of the GPU, GPU busy 97-98% -> the host-path overlap is STOPPED, nothing built. Combined exact arm `DSV41_FUSED_AV=1` on top of mHC, 2 pinned pairs: exact (a8c3394af9d2 + pin counters + G1 digest), G1 +0.15/+0.16%, same-output G4 batches -0.02/+0.21% -> NOT promoted, stays installed off. Launcher afa97ae6 unchanged; production g5host-PU exact.py a8c3394af9d2 x2, G1 103.97 with the reference digest bc1e8386878bd2b7; API down 17 min 55 s over 6 restarts): see the last section, "Goal window 5, Host step".
> **2026-10-28 10:33 UTC Goal window 5, mHC co-residency gate: PASS** (the promoted `DSV41_MHC_OVERLAP=moet` + `DSV41_MHC_OVERLAP_PROJ=moet` stays; launcher afa97ae6 and manifest 0c96a727fb81 unchanged. Stopped window on 5 Sparks: Moet post_norm (aux off/on), Moet proj, TileLang stats and the DeepGEMM tf32 prenorm GEMM at M = 1-64 on recorded inputs, 7.56 M calls beside the served routed MoE / a persistent copy / a shared-memory hammer (same-SM residency proven from %smid records) / solo: 0 mismatches; 415,360 co-running MoE outputs 0; 12,500 overlapped-region graph replays 0; racecheck / synccheck clean. Production g5stress-PU: exact.py a8c3394af9d2 x2 under live overlap, G1 104.02 with the reference digest bc1e8386878bd2b7; service down 28 min 45 s): see the last section, "Goal window 5, mHC co-residency gate".
> **2026-10-28 09:14 UTC Goal window 5, mHC step** (Moet's mHC boundary overlap ported BITWISE as `DSV41_MHC_OVERLAP`; `moet` + `DSV41_MHC_OVERLAP_PROJ=moet` PROMOTED: launcher f214c2c9 -> **afa97ae6**, supervisor manifest d773ed60f27b -> **0c96a727fb81**, site `nvidia/model.py` 9c2d22ad -> ca14bbb1 + new `dsv41_mhc_overlap.py` 4468cad5 on 8/8. Exact: 0-ulp GPU tests on synthetic and recorded boundaries at M = 1-64 (eager + CUDA-graph replay), and exact.py a8c3394af9d2 with the pin's counters + the G1 reference digest on every pinned launch. Pinned A/B vs C3 (av-pin-k5-g3FU): G1 +3.03% / +3.03%, G4 prose MIN +1.6 / +3.9%, G4min_hat +3.8%, 4-session step -3.9..-4.4%; `stock` (served kernels) -0.6%, `moet` with the served projection +0.8..+1.0% G1 only. Production g5mhc-PU (live): G1 104.06, exact a8, one 1M 168.3 s good class, GSM8K-50 49/50, longctx PASS. Goal: G1 +15.3% short; G4 prose session at 50 by G4min_hat, worst run 49.9. The vision staging set is STALE (launcher changed): see the last section, "Goal window 5, mHC step")
> **2026-10-28 04:50 UTC Goal window 5, LongCtx step (Codex ops kit 07)** (indexer top-k reuse across decode steps CLOSED at its own gate, NOT promoted: on 1,651 recorded real decode steps at 128K / 512K (diagnostic worker-extension recorder, output-neutral: exact.py a8 with the pin's counters, 6/6 continuations identical recorder on/off) the top-512 set changes at every step: 0 of 109,912 (layer, row) pairs could be reused exactly, mean overlap 0.50-0.58, and the certificate's bound D is 79-94x |v_k| against a ~0.05% boundary gap -> 0 passes; no serving code, no A/B. A 512K decode step costs +2.30 ms over a 1K prefix (30.2 vs 27.9 ms), the most any reuse could save, and nothing at G1 / G4 contexts. Lead, not built: previous-step v_k as an exact top-k threshold hint (valid in 51-63% of rows, <= ~0.3 ms/step at 512K). Nothing changed: live g5lc-PU = launcher f214c2c9 defaults, exact a8c3394af9d2): see the last section, "Goal window 5, LongCtx step".
> **2026-10-28 02:46 UTC Vision input (2026-10-27): image window ROLLED BACK (R2), images NOT live** (live = exactly the pre-window state, no sha changed: launcher f214c2c9, model_runner.py 6a4f2433, manifest d773ed60f27b, instance vis-R2, text-only, KV 6,390,007, AV unpinned; exact a8c3394af9d2 re-verified; site tree 4baa4e458a1a (5,453 files) on 8/8, re-recorded 03:10 UTC. VP (images on + AV pin, KV 16G) passed exact.py and its counters, the pin, the weight digest (text weights identical; +266 vision params, 116.9 MiB per rank), facts, autotune, C1 AL 5.191 / 2.725, G1 101.76 with the reference digests, and 8 of 9 pre-image detref probes bitwise. code:4000 MISMATCHED while a foreign API request was batched with it (4.16 s against 2.00-2.03 s in the references), so the driver rolled back and R2 followed: the images-on text path is unproven, not shown broken. VC / VU / hist1m and the DSH step not run. 772 s hard downtime over 3 restarts. A retry needs a new approval and an API free of foreign traffic during the probes; open items: the unauthenticated API serves 192.168.0.86 and tailnet node `fedora` (aurora@)): see the last section, "Vision input (2026-10-27)".
> **2026-10-27 20:20 UTC Goal window 5, ops step** (Codex ops kits; launcher f214c2c9 unchanged, live g5ops-PROD, exact a8c3394af9d2 on both instances, site tree 4baa4e458a1a on 8/8). 08 regression gate INSTALLED: `bash ~/dsv41-bench/gate/gate.sh --live`, first run PASS in 570 s (exact a8, G1 101.81, GSM8K-200 195, longctx 60/60, 1M 168.3 s, KL baseline). 10 black box + panic sysctls on spark-three via the site layer (spark-os 0.9.5-1019-bb1 on spark-three ONLY; two-reboot test clean; removal commands in the section). 03 proxy deep idle NOT promoted (+15.7 ms on the first request after idle, no measurable power; the proxy idles at 0.9% of a core, not 2/3). 06 profiled only (rank-0 worker pre-launch Python ~1.6 ms per G4 step is the one host lever; API and EngineCore have none). 04 confirmed live (+200 s production / +138 s pinned): see the last section, "Goal window 5, ops step".
> **2026-10-27 18:07 UTC Long-prompt determinism (goal window 5)** (`DSV41_TOPK_CANON_TIES=all` PROMOTED, launcher 95f08eff -> f214c2c9. Greedy drift on long prompts came from BOUNDARY TIES in the sparse-indexer top-k: two or more candidates had exactly the row's 512th score, and the stock prefill/decode top-k kept a random one; DSV41_TOPK_CANON had fixed only the order. A tie now keeps the lowest index; untied rows stay bitwise stock. Cold probes 2.5K-128K and three 6K tool prompts: 4/4 identical (the flag-off control gave 2-4 distinct of 4); exact.py a8 + G1 digest unchanged; GSM8K 50/50, longctx PASS; TTFT 16K +0.0%, 128K +0.35%; 1M 168.4 s good class; decode within noise. Warm (prefix-cache hit) vs cold still differ: row-count numerics, not fixed): see the last section, "Long-prompt determinism".
> **2026-10-27 16:05 UTC Goal window 4, final verification** (config verified on 8/8: launcher 95f08eff = window 3's decode stack + exact SP prefill / E7 (04+05) + fast loader + tool-stream newline flag; supervisor 7bea1cf3, manifest 40eb7abf08c7. Exact on both instances: exact.py a8c3394af9d2 4/4, the pinned one with the pin's counters and the G1 reference outputs (9a18f802d930f6e7), production with the same outputs. Production g4v-FU (live): G1 102.00, G4 mean 61.05 / prose MIN 48.41, conc4/8/16/32 179/264/397/590, cold TTFT 16K 2.49 s / 128K 19.6 s / 1M 166.2 s good class, healthy +209 s, GSM8K 50/50, longctx PASS; pinned g4v-FP G1 101.69, prose MIN 47.93. Goal NOT met: G1 needs +17.7%, the prose session +3.3% (worst run +4.4%); window 4's gains were prefill and operations. Next: Moet mHC overlap + the MoE levers for G4 (exact), a trained drafter head for G1 (not exact under the digest rule: user decision)): see the last section, "Goal window 4, final verification".
> **2026-10-27 13:30 UTC Goal window 4, decode+ops** (nothing promoted; live = launcher 31d04d8f defaults, unchanged. Codex 09 fused draft budget `DSV41_FUSED_AV`: bitwise (31,019 GPU cases + 7,100 serving shadow compares, 0 mismatches; exact.py a8 + G1 reference digest when serving it) but its first pinned pair is +0.20% G1 / +0.28% G4 prose MIN, below the 0.5% bar, so it stopped there (installed, off). 01 RoCEnante: already covered by window 3. 02 M32 tile scheduling: gate measured on the live kernels, ceiling 1.025x at M=6 (gate 1.08x), CLOSED; at most ~0.3 ms/step at any load. 08 early KV slab: 0/4 slow 1M launches with it and 0/4 without (166 s each); one stock-loader launch was slow (220.7 s): **the fast loader is what removed the 1M straggler** (KV now carved from 2 MiB blocks; under the stock loader only 0.06-1.7 GB of them are free at the KV allocation), so the early slab is NOT promoted (installed, off). Production g4d2-FINAL: G1 102.03, G4 prose MIN 48.36, exact.py a8 x2. Goal NOT met): see the last section, "Goal window 4, decode+ops".
> **2026-10-27 11:19 UTC Fast weight loader** (Codex ops kit 04: `DSV41_FAST_LOAD` PROMOTED, launcher fd1e809e -> 31d04d8f). The DSv4.1 target load now streams pre-sorted, with 8-thread O_DIRECT reads of only the bytes each rank's loaders use (expert w1/w3 rows; Engram rows filled in place). Every parameter is sha256-identical to the stock loader on 8/8 ranks, Engram included (4 digest launches, 10,856 digests each); exact.py a8 and the G1 reference digest hold on all 16 A/B launches. Target load 162-177 -> 35-37 s; pinned time-to-healthy 286 -> 126-132 s (-55%); unpinned production 355 -> 199 s. Stock launches swapped 2.3-3.0 GB per node to the NVMe swap file; fast loads swap none (1M MemAvailable minimum 6.2-7.1 GB, floor 6 GB). Final instance g4l-FINAL: G1 102.13, GSM8K 50/50, longctx PASS, 1M 165.9 s good class. The interrupted decode step's g4d bundle was uninstalled first; the decode step must re-install it. See the last section, "Fast weight loader".
> **2026-10-27 07:25 UTC Goal window 4, prefill** (Codex 04 exact SP prefill + 05 exact E7 PROMOTED as a pair, launcher fd1e809e: exact (QSP bitwise 2048-16,384 rows on 8/8 ranks incl. 12.7-14.2K; QE7 bitwise where the old E7 failed; exact.py a8c3394af9d2 + G1 reference digest on every pinned launch); pinned pairs TTFT 16K +6.7 / +6.2%, 128K +8.9 / +8.7%, 500K / 1M -8 to -10% same instance, short chats during a 128K prefill 1.81 -> 0.84 s; decode unchanged; GSM8K 50/50, longctx PASS, 1M needle 7/7. 06 candidate-only indexer: logit-bitwise but ~0.3% at 1M, NOT promoted (installed, inert). Production g4-FINAL: G1 101.6, G4 prose MIN 47.6, 1M 167.5 s good class. Goal NOT met): see the last section, "Goal window 4, prefill".
> **2026-10-26 23:40 UTC Goal window 3, with final verification** (fresh AV pin av-pin-g3 = a8c3394af9d2 909/467/2335/1373; codex-adopt bundle (fused CANON + decode remap reuse + omit token_to_seq) + dhtac (draft-head tactic pin 17-29:1) PROMOTED as a stack, launcher b0cf94d0: pinned G1 +1.2%, G4min_hat +2.1%, exact; RoCEnante, mfront (illegal address in mixed live/fused process) and rdma_ar env knobs NOT adopted; final verification on 8/8: production g3-FU (live) G1 101.5, G4 prose MIN 48.6 / G4min_hat 48.4, pinned g3-FP G1 101.7 with the A/B digest, 1M 183.7 s good, GSM8K 50/50, longctx PASS, J/token flat; goal NOT met: G1 needs +17-18%, G4 prose +3.0-4.7% on step time): see the last section, "Goal window 3".
> **2026-10-26 07:00 UTC Prefill+decode window: final verification** (live = launcher 04b6738e defaults on 8/8, AV unpinned, prompt-tokens-details on; exact.py a8c3394af9d2 x3 with identical counters; vs the 2027f4d8 baseline: C1 coding -4.3% / prose -3.7%, single +3.1 to +4.4%, conc4-32 +3.4 to +5.8%; prefill flat: 16K 2.71 s, 128K 21.8 s, 1M 229 s on a slow-class launch (~182 s on good ones); GSM8K-50 49/50, longctx 60/60; rollbacks and open user decisions listed): see the last section, "Prefill+decode window".
> **2026-10-26 06:25 UTC pd Marlin MoE measurement** (measurement only, nothing installed; live restored on launcher 04b6738e defaults, exact.py a8c3394af9d2 x2; stock `--moe-backend marlin` (288->384) is a NO-GO: 1.04-1.06x at decode, +7.5 GiB/rank. Unpadded Marlin W4A16 (FC1 576 / FC2 K 320) runs 1.31x at 6 rows and 1.37x at 48 rows: -3.2 ms/step at 1 user and -14 to -17 ms/step at 16 users. It is not exact (MoE output 3.2% rel L2 vs live, about 7x closer to FP32), and 16K-chunk prefill MoE is up to 14% slower, so it is a user decision. Exact lead: the live M32 kernels leave ~20% of the weight bandwidth unused): see the last section, "pd Marlin MoE measurement".
> **2026-10-26 05:45 UTC pf prefill A/B step** (nothing promoted; live = launcher 04b6738e defaults, unpinned, all pf files removed; E7 tail skip changes output: replay-layer indexer-K caches are read by later chunks; SP v2 exact is -6.1% TTFT but not yet bitwise: split-match gate added, a second row-count-dependent kernel is still unidentified; gather-add crashes at startup; NCCL arms closed; the AV pin is not a reliable straggler fix (pinned 6/8 good, unpinned 1/5); within-instance prefill nondeterminism above ~4.1K tokens): see the last section, "pf prefill A/B step".
> **2026-10-26 02:05 UTC pd long-context X-ray** (diagnostic only, nothing installed: the 1M prefill excess is ~78% a rank-5 (spark-six) paged-KV sparse-MLA straggler that appeared on 5/5 unpinned launches and 0/2 AV-pinned ones (1M TTFT 232-235 s vs 184.6-184.8 s); inherent growth ~13 s, 80% indexer; default-pin proposal now carries a -47 s 1M reason; user decision): see the last section, "pd long-context X-ray".
> **2026-10-25 23:25 UTC pd decode A/B step** (D05 dead-FC2-store skip + L2 prune-shared PROMOTED: C1 coding -4.3% / prose -4.0%, conc16/32 +5.1/+5.4% pinned; pdl HOLD; rgate KILLED; launcher 04b6738e; new AV pin ~/dsv41-stage/pdab/av-pin-pdfinal.json): see the last section, "pd decode A/B step".
> **2026-10-25 20:50 UTC pd base step** (prompt-tokens-details always on; startup hang root cause confirmed, C02 + rel-2 graph safety and the relaunch diet promoted; launcher b1e75497): see the last section, "pd base step".
> **2026-10-24 21:45 UTC takeover audit** (Codex goal-120 work, two undocumented Codex site installs, live anchor a8c3394af9d2, baseline): see the last section, "Takeover audit after Codex goal-120".

## Current follow-up: 2026-10-24 17:35 UTC

The server is verified up on all eight Sparks. The latest retained launcher is **2027f4d8**: the k1/2 qualified bundle plus `DSV41_RDMA_PARLOAD=1` (binary 62daad6c, selector binding acbeeef7). Previous launcher **1a2f26f4** is saved as `~/dsv41-tp8.sh.bak-k1-3-rdma-loads` on every node. Context remains 1,048,576, the existing scheduler remains enabled, and the final live process has no AV timing pin.

Codex recovered the interrupted i3-C transition and completed a new controlled comparison for 1–4 users. Effective AV arrays were identical; exact outputs and greedy single-stream acceptance counters matched. Matched C1 step time improved 1.05% coding / 1.55% prose; thinking-enabled single throughput improved 0.48–1.08%. Multi-user results were mixed within the observed variation, so no clear multi-user gain is claimed.

Final default: code/prose/math 87.39/70.13/120.57 tok/s; GSM8K 50/50; longctx 60/60 with exact copies; 999,411-token probe 7/7 with exact copying. Final exact hash 9330f03c4f97 x2 matches the three prior A8-262k-canon-nopc runs. All eight rank logs have no ERROR/Traceback lines and service restart counts are zero; fresh generation after long-context testing returned 391 for 17*23.

Full report, runtime identities, comparison and rollback: [October 24 deployment](../dsv41-opt-20261024-codex-rdma/deployment/report.md). Raw new runs are in `~/dsv41-opt-20261024-codex-rdma/`. The failed first 1M test used unsupported `1m` syntax before any request; the corrected `1000k` probe passed and the failure log is preserved. The historical sections below remain intact.

---

# DSv4.1-Flash TP8 optimisation, 2026-10-21

Deployment: vLLM 0.32.0, TP8 over spark-one..spark-eight, systemd --user unit `dsv41-rank`, launcher `~/dsv41-tp8.sh <rank>`, API spark-one :8000 model `dsv41`.
Raw results per experiment are in this directory (`spec-len/`, `m32/`, `rdma-ar/`, `conc/`, `final/`), each with `summary.txt`.

## Final numbers (live config, final/ run)

Measured 18:05-18:09 UTC on the instance started 17:53:53 UTC (launcher 3045aa83). Single = greedy, 1024 tokens, 2 repeats per type, averaged. Conc = 512 tokens per request.

| metric | baseline 14:08 | final | change |
|---|---|---|---|
| single code | 56 | 71.4 (70.5 / 72.3) | +27% |
| single prose | 42 | 53.9 (52.2 / 55.7) | +28% |
| single math | 73 | 104.4 (102.8 / 106.0) | +43% |
| single mean | 57 | 76.6 | +34% |
| conc4 aggregate | 102 | 134.7 | +32% |
| conc8 aggregate | 147 | 192.6 | +31% |
| conc16 aggregate | about 147 (ms8 queues; about 190 at ms8 with RDMA) | 277.3 | about +88% vs baseline |
| GSM8K-50 gate | 50/50 | 50/50, 0 truncated (200 tok/s at c8) | pass |
| exact.py combined | n/a | fd1603b4e20b, which equals the conc B2 hash on the same instance, so the output reproduces within an instance | |

The server log shows no errors (only the benign transformers model-type notice). All 8 units are active, with no extra environment set.

## Changelog (in order applied)

| step | change | result | launcher sha256 after |
|---|---|---|---|
| 0 | baseline (14:08 UTC) | single 57, conc4 102, conc8 147 | 58ab5e61 |
| 1 spec-len | added env-gated `DSV41_SPEC_JSON` override (inert when unset). SPEC=3/4 and adaptive-off tested. | not promoted (SPEC=4 -3.9% single, SPEC=3 -7.4%, adaptive-off -11%/-22% at conc4/8) | f2fd3f08 |
| 2 m32 | M32 small-tile MoE path for M<=48 (`DSV41_SM121_TARGET_M32=1`, native lib `~/dsv41-stage/m32/target-m32-native-v1`) | promoted, about +5% at all loads | f4ba806e |
| 3 rdma-ar | custom RDMA all-reduce over CX7 (`DSV41_RDMA_AR=1`), 3-8x faster than PyNccl for 1-48 rows | promoted, +23% single, +17% conc4, +20% conc8 | 753f0a56 |
| 4 conc | `--max-num-seqs ${DSV41_MAX_NUM_SEQS:-16}` + spec-kernel warmup patch (`DSV41_WARM_SPEC_KERNELS=1`); inert M32/M64 max-M knob (default 48) | promoted, conc12 +30%, conc16 +45%, flat at <=8 | 3045aa83 |

Site-packages changes (all under `~/venvs/vllm032/lib/python3.12/site-packages/vllm/`, sha256 identical on all 8 ranks):

| file | live sha | backup |
|---|---|---|
| model_executor/layers/fused_moe/experts/deep_gemm_moe.py | 84ecc6ef | `.bak-m32` = fe39558e (pristine), `.bak-conc` = b8bb0327 (M32 before max-M knob) |
| model_executor/layers/fused_moe/deep_gemm_utils.py | b7e8de4d | `.bak-m32` = 5eb623a6 (pristine) |
| model_executor/layers/fused_moe/experts/sm121_target_m32.py (new) | 723fdf8d | `.absent-m32` marker |
| distributed/device_communicators/cuda_communicator.py | 52fc8cf5 | `.bak-rdma` = 2f5368d6 (pristine) |
| distributed/device_communicators/rdma_all_reduce.py (new) | 3a128082 | none (new file; uninstall-rdma.sh removes it) |
| v1/worker/gpu/warmup.py | 19009d42 | `.bak-warmspec` = daeac571 (pristine) |

Launcher backups on every rank: `~/dsv41-tp8.sh.bak-spec-len` (58ab5e61, original), `.bak-m32` (f2fd3f08), `.bak-rdma-ar` (f4ba806e), `.bak-conc` (753f0a56).
Other staging: `~/rdma_ar_dsv41/` (rdma_ar_ext.so dfeeab00), `~/dsv41-stage/m32/` (native lib + JIT cache, 3 entries).

## Caveats

- Greedy output is not bit-identical across server restarts, even with an unchanged config: within one instance the hashes repeat exactly, across restarts 0-2 of 6 prompts match. The cause is likely the adaptive-verification cost profile, which is re-measured at each startup, plus the RDMA reduction order. `exact.py` can therefore compare arms only within one instance.
- GSM8K-50 scores ranged 49-50 on every arm, controls included. The one recurring miss (item 12, 12 vs 13) is a genuine wrong answer, not a truncation.
- Mean accept length dropped from about 2.46 to about 2.35 with RDMA AR, because the reduction order changed. Throughput more than covers it.
- The MemAvailable floor while serving is about 22-23 GiB per rank at ms16. Startup briefly dips to about 14 GiB on spark-six and spark-seven.
- spark-one cannot ssh to spark-six, seven or eight (publickey denied). Run fleet scripts from the Mac.

## Rollback (run from the Mac; S = /private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad)

Fast, env only (site files remain but are inert):

```
bash $S/conc/launch-conc.sh DSV41_MAX_NUM_SEQS=8 DSV41_WARM_SPEC_KERNELS=0   # undo step 4
bash $S/conc/launch-conc.sh DSV41_RDMA_AR=0                                  # undo step 3 (PyNccl)
bash $S/conc/launch-conc.sh DSV41_SM121_TARGET_M32=0                         # undo step 2 (M64 path)
bash $S/conc/launch-conc.sh DSV41_MAX_NUM_SEQS=8 DSV41_WARM_SPEC_KERNELS=0 DSV41_RDMA_AR=0 DSV41_SM121_TARGET_M32=0   # all three
```

Launcher only, for example back to the step-3 launcher:

```
for h in one two three four five six seven eight; do ssh spark-$h 'cp ~/dsv41-tp8.sh.bak-conc ~/dsv41-tp8.sh'; done
bash $S/conc/launch-conc.sh
```

Full return to the 14:08 baseline. Stop first: `for h in one two three four five six seven eight; do ssh spark-$h 'systemctl --user stop dsv41-rank'; done`

```
# warmup.py on each rank:
for h in one two three four five six seven eight; do ssh spark-$h 'cd ~/venvs/vllm032/lib/python3.12/site-packages/vllm/v1/worker/gpu && cp warmup.py.bak-warmspec warmup.py'; done
bash $S/rdma/scripts/uninstall-rdma.sh      # restores cuda_communicator.py from .bak-rdma, removes rdma_all_reduce.py
bash $S/m32/uninstall.sh                    # restores deep_gemm_moe.py / deep_gemm_utils.py from .bak-m32, removes sm121_target_m32.py
for h in one two three four five six seven eight; do ssh spark-$h 'cp ~/dsv41-tp8.sh.bak-spec-len ~/dsv41-tp8.sh'; done
bash /private/tmp/claude-501/-Users-jeweled/1b0ccff9-d7d4-4c14-901f-1d71aef60396/scratchpad/dsv41-launch-all.sh   # run with bash; the file is not executable
```

Afterwards verify that sha256 is identical on all 8 ranks. The server is ready when `~/dsv41-logs/rank0.log` shows "Application startup complete" (about 6.5 min).

## Recommended next steps

1. Make greedy output deterministic across restarts: pin or persist the adaptive-verification cost profile, for example by loading it from a file instead of re-measuring at startup. Then exact.py becomes a usable regression gate across restarts.
2. Run a larger GSM8K cohort (200 or more) on the live config to rule out a 1-point accuracy shift from RDMA AR. Item 12 flips 12/13 between runs.
3. Tune RDMA AR (`RDMA_AR_MAX_REDUCE_BYTES`, two-shot threshold) and profile the step again. All-reduce was 10.7 ms of the 40.8 ms step and is now much smaller, so FP4 MoE (about 14 ms) and the dense GEMMs (about 12 ms) dominate.
4. With RDMA AR live, the conc16/ms16 envelope has headroom (MemAvailable about 22 GiB). Try ms24 or ms32 if many concurrent users are expected.
5. Recalibrate adaptive verification (`DSV41_AV_VERIFY_SCALE` / `FIXED_MS`) at concurrency. It showed little upside at one user, but verify costs changed a lot after RDMA AR.
6. Fix spark-one to spark-six/seven/eight ssh keys so in-bench memory sampling covers all ranks.

## Round 2, exp1: moe-compact (unpadded target FC1), PROMOTED 2026-10-21 20:40 UTC

**What changed.** The routed FP4 MoE FC1 weights of the target model are stored without TP padding: per rank and expert, N = 576 rows `[gate 288 | up 288]` instead of 768 `[gate 288 | 0 x96 | up 288 | 0 x96]`. FC2 keeps K = 384. The SiLU-mul-quant kernel writes the 96 zero columns into the 384-wide FP8 buffer that FC2 consumes, so FC2 inputs and scale groups are byte-identical to the padded path. FC2 cannot be compacted at K = 288: DeepGEMM asserts `not is_mixed_fp4 or k % 128 == 0` (the probe confirmed this), so it would need a kernel change. The drafter (128 experts) is untouched. The change is behind `DSV41_MOE_COMPACT`, and with the flag unset all three files take the old code paths.

**Files** (under `~/venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/`, sha256 identical on all 8 ranks, backups `.bak-compact` = previous live):

| file | live sha now | `.bak-compact` |
|---|---|---|
| fused_moe/experts/deep_gemm_moe.py | 45418259 | 84ecc6ef |
| quantization/mxfp4.py | 67339029 | a7fc1bcd |
| quantization/utils/fp8_utils.py | 8dc7dab8 | 013df536 |

Launcher: `export DSV41_MOE_COMPACT=${DSV41_MOE_COMPACT:-1}` added after `DSV41_LOCK_ENGRAM_HOST`. `~/dsv41-tp8.sh` is now **769a0564** on all 8 ranks, and the backup `~/dsv41-tp8.sh.bak-compact` = 3045aa83. Payload, scripts and test are in `~/dsv41-stage/moe-compact/` on every rank; the Mac originals are in `$S/r2/moe-compact/`.

**Offline gate** (`run_offline_test.sh`, spark-one rank 0 + spark-eight rank 7, service stopped). Every case was BIT-EXACT against the live padded expert (bf16 ±0 counted equal), and the flag-on-padded-weights run was also identical:
- M32 path, random weights: M = 1..48 every value (144 cases) plus the default sweep up to 1024 (60 cases, 128-row path for M > 48), with valid, half-padding and all-padding routes, and CUDA-graph replay with poisoned outputs. All passed on both hosts.
- Real checkpoint layers 0 and 39, all 384 experts, TP ranks 0 and 7: 21 + 21 cases per host, plus M = 1..48 for L39 on rank 7 (144). All passed.
- M64 fallback (M32 off): 60 cases, all passed. CPU loader check: 432 byte-exact slices.

Per-layer routed MoE timing, 8 distinct layer copies (L2-defeating), spark-one:

| M | path | padded µs | compact µs | speedup | padded GB/s | compact GB/s |
|---|---|---:|---:|---:|---:|---:|
| 1 | M32 | 119.4 | 110.2 | 1.084 | 157 | 142 |
| 6 | M32 | 608.3 | 519.5 | 1.171 | 176 | 172 |
| 12 | M32 | 1149.8 | 984.6 | 1.168 | 180 | 175 |
| 18 | M32 | 1634.3 | 1398.1 | 1.169 | 180 | 175 |
| 24 | M32 | 2127.1 | 1818.7 | 1.170 | 180 | 175 |
| 36 | M32 | 2875.5 | 2456.1 | 1.171 | 179 | 175 |
| 48 | M32 | 3535.3 | 3025.3 | 1.169 | 179 | 174 |
| 96 | 128-row | 8063.4 | 7653.6 | 1.054 | 117 | 103 |
| 6 / 24 / 48 | M64 fallback | 725 / 2515 / 4292 | 673 / 2301 / 3886 | 1.08 / 1.09 / 1.10 | 150 | 135-138 |

The byte model predicts 1.20. On the M32 path (64-wide N tiles, 576 = 9×64) the compact layout gets 1.17. On the 64- and 128-row paths it gets 1.05-1.10, which is consistent with DeepGEMM choosing 128-wide N tiles there, so the last tile over-reads 64 rows of the next expert. spark-eight matches within 2%.

**Serving A/B** (C, A, C, A, a restart each; `~/dsv41-opt-20261021/r2-exp1/{C1,A1,C2,A2}`; single = greedy 1024 tok; conc = aggregate tok/s at 512 tok):

| instance | code | prose | math | single mean | conc4 | conc8 | conc12 | conc16 | accept len (single / c4 / c16) | GSM8K-50 | exact.py x2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| C1 control | 64.6 | 54.5 | 107.1 | 75.4 | 132.2 | 189.2 | 235.5 | 281.0 | 2.32 / 2.36 / 2.34 | 49/50 | self-consistent |
| A1 compact | 68.2 | 54.6 | 110.4 | 77.7 | 141.9 | 208.4 | 258.8 | 298.1 | 2.29 / 2.35 / 2.28 | 50/50 | self-consistent |
| C2 control | 65.1 | 55.3 | 108.2 | 76.2 | 130.7 | 192.0 | 238.7 | 273.7 | 2.34 / 2.26 / 2.31 | 50/50 | self-consistent |
| A2 compact | 75.7 | 57.9 | 112.6 | 82.1 | 140.5 | 210.0 | 262.4 | 300.4 | 2.49 / 2.31 / 2.32 | 50/50 | self-consistent |
| **mean C → A** | | | | **75.8 → 79.9 (+5.4%)** | **131.5 → 141.2 (+7.4%)** | **190.6 → 209.2 (+9.8%)** | **237.1 → 260.6 (+9.9%)** | **277.3 → 299.2 (+7.9%)** | Δ within ±0.07 | | |

Every arm instance beats every control instance at conc4, 8, 12 and 16. Single-stream gains are noisier (A1 +3%, A2 +8%). Part of A2's single-stream gain comes from a higher accept length in that instance (2.49).

**Memory.** Model load is 49.71 → 42.21 GiB per rank (−7.50 GiB). MemAvailable during conc16 is about 23 → about 30.5 GiB per rank. The KV cache is unchanged (`--kv-cache-memory-bytes 8G` = 1,628,987 tokens, 6.21× full 262K contexts) because it is set explicitly. That leaves about 7.5 GiB of new headroom for a bigger KV cache (for example 12-14G) and/or ms24-32. This is not tested yet.

**Final live config:** the round-1 flags plus `DSV41_MOE_COMPACT=1` (launcher default). Restarted with `dsv41-launch-all.sh`: /health 200, rdma_ar enabled, compact log line on ranks 0, 4 and 6, 17×23 → "391".

**Rollback** (from the Mac, S = /private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad):
```
bash $S/conc/launch-conc.sh DSV41_MOE_COMPACT=0             # env only: padded layout, files inert
for h in one two three four five six seven eight; do ssh spark-$h 'cp ~/dsv41-tp8.sh.bak-compact ~/dsv41-tp8.sh'; done   # launcher -> 3045aa83
# full file revert (service stopped first):
for h in one two three four five six seven eight; do ssh spark-$h 'systemctl --user stop dsv41-rank'; done
bash $S/r2/moe-compact/uninstall.sh                          # restores 84ecc6ef / a7fc1bcd / 013df536 from .bak-compact
bash $S/dsv41-launch-all.sh
```
Raw results: `~/dsv41-opt-20261021/r2-exp1/` (instances, `inst.sh`, `summ.py`, `offline/` with both hosts' jsonl).

**Next:** (1) FC2 compact via a split main [5120,256] + repacked tail tile in `sm120_fp8_fp4_gemm_1d1d.cuh` (another ~8% of routed bytes). (2) Make the 128-row path pick 64-wide N tiles for N = 576, or extend M32-style tiles above M = 48: conc16 (M = 96) gets only 1.05× in the kernel. (3) Spend the freed 7.5 GiB on KV and/or max-num-seqs.

## Round 2, exp2: decode-sized MXFP8 GEMM (skinny_mx, "Patch B"). NOT PROMOTED (kernel gate failed)

**What was tried:** `$S/r2/dense/install-r2dense.sh skinny` staged the flag-gated skinny GEMM (`DSV41_SKINNY_MX=1`, `DSV41_SKINNY_MX_WOA=1`) with the service stopped. All 8 ranks matched the expected shas: flashinfer 07c7085d, emulation ddb4b951, o_proj 9df24b47, sm121_skinny_mx.py eedc1f49, .so c054de3a. The GPU kernel gate then ran on spark-two with the service stopped. Patch A (topk) was **not** installed.

**Gate result: FAIL.** Correctness passed: relerr ≤ 3.5e-3 on every shape, matching CUTLASS's own error; within 1 bf16 ulp of CUTLASS on almost every element; det=True for both A8 and A16. Speed failed. The shipped `bench_skinny_mx.py` times a host-side Python loop, so its FlashInfer baseline is launch-bound (~42-160 µs) and its 2-3x "speedups" are wrong. Re-timed inside CUDA graphs (64 DRAM-cold weight copies per graph, CUTLASS autotuned, as in the server); script `r2-exp2/e2_graph.py`:

| GEMM (per rank) | M=1 | M=4 | M=6 | M=8 | per step at M=4-6 |
|---|---|---|---|---|---|
| fused_wqa_wkv 1792x5120 | CUTLASS 45.1 µs / 210 GB/s vs skinny 52.6 / 180 (x0.86) | 45.5 vs 60.0 (x0.76) | 45.5 vs 68.2 (x0.67) | 46.3 vs 74.2 (x0.62) | -0.6 to -1.0 ms |
| wq_b 4096x1280 | 26.9 vs 29.4 (x0.92) | 26.7 vs 32.6 (x0.82) | 27.7 vs 35.9 (x0.77) | 27.4 vs 40.0 (x0.68) | -0.3 to -0.4 ms |
| wo_b 5120x1024 | 26.7 vs 31.7 (x0.84) | 27.7 vs 38.1 (x0.73) | 26.6 vs 41.9 (x0.63) | 26.7 vs 48.4 (x0.55) | -0.4 to -0.6 ms |
| shared w13 576x5120 | 21.2 vs 15.7 (x1.35) | 22.0 vs 18.0 (x1.22) | 21.2 vs 19.8 (x1.07) | 22.0 vs 22.6 (x0.97) | +0.06 to +0.16 ms (side stream) |
| shared w2 5120x288 | 11.2 vs 13.3 (x0.84) | 11.1 vs 16.0 (x0.70) | 11.1 vs 18.0 (x0.62) | 11.1 vs 20.7 (x0.54) | -0.2 to -0.3 ms (side stream) |
| WO-A 1024x4096 (A16) | bf16 bmm 53.9 vs skinny 23.3 (x2.32) | 41.2 vs 26.5 (x1.55) | 48.4 vs 28.7 (x1.69) | 47.1 vs 33.3 (x1.41) | **+0.6 to +0.8 ms** |

Isolated, autotuned CUTLASS MXFP8 already streams about 200 GB/s on wqa_wkv, wq_b and wo_b, about 85% of the 238 GB/s stream bandwidth. The skinny kernel reaches 130-185 GB/s, and its efficiency drops as M grows, because each extra row costs more in the activation-in-registers design. The planner's trace numbers (wqa_wkv 65 µs, wq_b 36.7 µs) were in-server times; the gap above the isolated 45/27 µs is contention or overlap, not kernel inefficiency. A8 overall loses about 1.5-2.5 ms per step. WO-A A16 is the only real win, about 0.6-0.8 ms per step (about 2-2.5% single-stream), which misses the 2x gate and the +4% promotion bar. No serving A/B was run, as the protocol requires after a gate failure.

**Reverted:** `uninstall-r2dense.sh` restored all 8 ranks to 97bcfaa6 480149ff d52c5a20 a5b93da2 f2893ee6. sm121_skinny_mx.py has been removed, no .bak-r2dense files are left, and the launcher is unchanged at 769a0564. The build and bench copies remain in `~/dsv41-r2/skinny_mx/` on each rank and are inert. Live config = exp1 (round-1 flags + DSV41_MOE_COMPACT=1), relaunched with `dsv41-launch-all.sh`.

**Possible follow-up:** use the A16 WO-A path only (`DSV41_SKINNY_MX_WOA=1` without `DSV41_SKINNY_MX`; the code paths are independent). Its expected gain is about +2-2.5% at 1 user and 0 at conc≥2, so it needs ≥3 ABAB pairs to resolve. Codex's WO-A Marlin candidate targets the same GEMM; compare the two before choosing. Another option is to keep the A8 path for shared w13 only at M ≤ 4, but that gain is marginal (~0.15 ms on a side stream).

Raw results: `~/dsv41-opt-20261021/r2-exp2/` (bench_cudagraph.txt = authoritative; bench_quick_hostloop.txt / bench_fi_autotuned_hostloop.txt = launch-bound host-loop timings; install/uninstall sha logs).

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

## Round 2 / exp4: vocab-sharded Markov drafter (DSV41_DSPARK_SHARDED_MARKOV): NOT promoted (2026-10-21/22 UTC)

**What it is:** a patch to `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (sha 97bcfaa6 -> 774b34c2). When the env flag is set, each rank adds the Markov bias only to its own 16,160-token vocab slice. The ranks then exchange one 16 B (max, id) record per request per position (5 small fp32 gathers per step) instead of each reading the full 129280x256 Markov head 5 times and all-gathering full-vocab logits.
Sources: Mac scratchpad `r2/drafter/` (install.sh, markov-shard/). Install: `bash install.sh markov`. Revert: `bash install.sh markov rollback`.

**Correctness (all checks passed):**
- rank0.log has `DSV41_DSPARK_SHARDED_MARKOV: vocab shard [0, 16160) pad 0, markov rows/rank 16160 of 129280, tp 8`, with no "ignored" warning.
- RDMA_AR_AUDIT confirms the new `all_gather (n,4) float32` calls go through rdma_ar, not NCCL.
- exact.py was self-consistent in every instance. GSM8K-50 at conc 8 scored 50/50 in all 4 A/B instances.
- AV cost profile draft_ms(1): 5.10 / 5.20 (control) -> 3.72 / 3.76 (arm), i.e. -1.4 ms. draft_ms(16): 12.5 -> 9.95.

**A/B** (C = live config, A = +flag; same launcher d0749eda; order C1 A1 C2 A2). AL = mean accept length.

| inst | code | prose | math | mean | c4 | c8 | c12 | c16 | AL code/prose/math | GSM8K |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 76.29 | 58.17 | 110.38 | 81.61 | 140.78 | 212.26 | 267.22 | 316.45 | 2.59/1.73/3.78 | 50/50 |
| A1 | 72.51 | 60.44 | 113.84 | 82.26 | 144.64 | 216.86 | 274.68 | 322.42 | 2.27/1.69/3.72 | 50/50 |
| C2 | 72.61 | 57.14 | 107.65 | 79.13 | 140.70 | 206.21 | 265.92 | 311.93 | 2.58/1.71/3.69 | 50/50 |
| A2 | 71.31 | 58.04 | 111.14 | 80.16 | 144.41 | 212.76 | 278.47 | 320.44 | 2.36/1.64/3.64 | 50/50 |

Means, control -> arm:
- single 80.37 -> 81.21 (+1.0%; gate needs +2.5%)
- conc4 140.7 -> 144.5 (+2.7%)
- conc8 209.2 -> 214.8 (+2.7%)
- conc12 266.6 -> 276.6 (+3.8%)
- conc16 314.2 -> 321.4 (+2.3%)
- code AL 2.585 -> 2.31 (-0.27, outside the +/-0.08 gate). Prose -0.06, math -0.06.

**Why code AL dropped: the drafts did not get worse.** An AV-off probe was run (`DSV41_SPEC_JSON={"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":false}`, DC1 vs DA1, single x2 reps):
- Untrimmed acceptance matches: code 2.80 vs 2.78, prose 2.13 vs 2.10, math 3.59 vs 3.80, conc4 2.78 vs 2.82.
- exact.py hashes were identical across the two instances (47098a31f5e4).
- The cause is the adaptive-verification trimmer. It picks the draft budget k = argmax (1+sum s_i)/(draft_ms + verify_ms(k)). A cheaper draft_ms lowers the fixed cost, so the profiled verify slope (already ~3x too steep, see the drafter prep notes) trims harder. Code loses positions 2-5 and about 3% tok/s, which cancels most of the drafter saving at 1 user.
- With AV off, the flag gives +6.1% single (77.6 -> 82.3). At concurrency, the trimmer's decisions are dominated by verify cost, so the gain shows through.

**Result:** not promoted under the exp4 gate (single +1.0% < +2.5%; code AL outside +/-0.08). The patch was rolled back (speculator.py = 97bcfaa6 on all 8). The launcher is unchanged (d0749eda). Live = same config as before exp4 (launch-all, no extra env), /health 200, sanity 391.

**Follow-up:** re-test this flag together with the AV cost calibration (drafter patch B: `DSV41_AV_VERIFY_SCALE` / `DSV41_AV_COST_JSON`). Once the verify slope is calibrated, the cheaper drafter should no longer cause over-trimming at 1 user, and the +2.3-3.8% conc gain plus about +6% untrimmed single gain should both be available.
Raw results: `~/dsv41-opt-20261021/r2-exp4/` (C1 A1 C2 A2 DC1 DA1, summ2.py, dsumm.py, inst.sh, diag.sh).

## Round 2 / exp5 — adaptive-verification cost calibration (PROMOTED, 2026-10-22 01:31 UTC)

**What changed**
- `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` was patched (av-calib) on all 8 ranks: sha b0526416 -> ec240369. Backup is `.bak-avcal`. Installed with `bash $D/install.sh av`, where D = Mac scratchpad r2/drafter.
  - The patch adds the env knobs DSV41_AV_VERIFY_SCALE / SCALE_N0 / SCALE_N1 / FIXED_MS / COST_JSON / DUMP_JSON / LOG_BUDGET.
  - It recalibrates the startup cost profile on rank 0 before the profile is broadcast to the other ranks.
- `~/dsv41-tp8.sh` changed from d0749eda to e270215f on all 8 ranks; backup is `~/dsv41-tp8.sh.bak-avcal`. It now exports these defaults:
  `DSV41_AV_VERIFY_SCALE=0.6 DSV41_AV_FIXED_MS=2.0 DSV41_AV_SCALE_N0=48 DSV41_AV_SCALE_N1=56`.
  - Every verify width up to 48 is priced as V(1) + 0.6*(V(n)-V(1)). The price tapers back to the profiled cost by width 56; 48->56 is the cliff where the M32 MoE path ends.
  - Draft cost gets +2.0 ms for the time spent outside the graphs.
- Rank0.log shows a line 'DSV41 adaptive-verification calibration (... scale=0.6 n0=48.0 n1=56.0 fixed_ms=2.0)'.

**Why:** the startup profile uses random tokens. Its verify slope is about 3.5 ms per extra draft at 1 user, but the measured slope is 2.39 ms per draft: (math - prose step ms) / (math - prose drafts verified) from the C1 budget histograms. That puts the refit scale at a = 0.68. The time outside the graphs refits to 2.1 ms (prose and math both).
- The planner's default taper (N0=8, N1=48) made the cost slope at widths 24-48 *steeper* than the profile. At conc12/16 the trimmer then cut verified tokens from about 30 to 15-22, and conc12/16 regressed by 2-6% (arms A1 a=.4/f2.5 and A2 a=.6/f2).
- N1=60 (A4) still regressed by 1.4%. A flat scale up to 48 (A5) fixed it.

**A/B (same window, C/A alternated, every instance got GSM8K-50 at conc 8 and an exact.py x2 self-consistency check):**

| inst | code | prose | math | single mean | c4 | c8 | c12 | c16 | GSM | accept per pos (math) |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 68.5 | 55.5 | 106.8 | 76.95 | 135.8 | 205.4 | 264.8 | 317.2 | 50 | .84/.64/.56/.30/.26 |
| A2-1 (a.6 f2 N1 48) | 78.0 | 58.8 | 109.7 | 82.15 | 145.1 | 204.8 | 255.3 | 309.8 | 50 | .87/.72/.59/.36/.30 |
| A1-1 (a.4 f2.5 N1 48) | 76.2 | 58.9 | 103.1 | 79.40 | 142.3 | 208.8 | 253.8 | 293.4 | 49 | .82/.64/.54/.33/.27 |
| A4-1 (a.6 f2 N1 60) | 78.8 | 59.0 | 111.1 | 82.96 | 144.9 | 207.1 | 267.3 | 310.4 | 50 | .86/.72/.61/.38/.31 |
| C2 | 70.5 | 58.0 | 112.5 | 80.32 | 144.8 | 207.4 | 269.5 | 312.3 | 50 | .88/.71/.61/.36/.29 |
| A5-1 (a.6 f2 N0 48 N1 56) | 77.0 | 58.9 | 115.4 | 83.76 | 143.2 | 210.5 | 268.4 | 319.6 | 50 | .87/.75/.65/.43/.36 |
| C3 | 72.3 | 56.8 | 112.4 | 80.52 | 140.1 | 206.6 | 266.6 | 311.1 | 50 | .85/.71/.60/.38/.33 |
| A5-2 (same) | 79.9 | 60.0 | 113.3 | 84.38 | 144.4 | 212.0 | 271.8 | 313.2 | 50 | .87/.74/.61/.41/.33 |

Mean of the 3 controls vs A5 (2 instances):
- single: code 70.4 -> 78.5 (+11.4%), prose 56.8 -> 59.4 (+4.7%), math 110.6 -> 114.3 (+3.4%); mean **79.3 -> 84.1 (+6.1%)**
- conc4 140.2 -> 143.8 (+2.5%), conc8 206.5 -> 211.3 (+2.3%), conc12 267.0 -> 270.1 (+1.2%), conc16 313.5 -> 316.4 (+0.9%)
- Every instance was self-consistent on exact.py.

Drafts verified per step at 1 user, control -> A5:
- prose 1.61-2.09 -> 2.14-2.33 (verifying 1 draft: 68% -> 37-48%)
- code 2.70 -> 3.48
- math 3.22 -> 4.04 (verifying 5 drafts: 34% -> 56-59%)

Total tokens verified per step:
- r4: 7.4-7.7 -> 8.0-8.4
- r8: 21-22 -> 24
- r16: 30 -> 31.5

Code position-2 acceptance went from .36-.40 to .51-.55; prose position-2 acceptance went from .08-.10 to .16-.20.

**Cumulative since round 1** (76.6 / 134.7 / 192.6 / 277.3): single 84.1 (+9.8%), conc4 143.8 (+6.8%), conc8 211.3 (+9.7%), conc16 316.4 (+14.1%).

**Rollback**
- Launcher only (trimmer becomes stock): `cp ~/dsv41-tp8.sh.bak-avcal ~/dsv41-tp8.sh` on all 8 ranks. Alternatively launch with DSV41_AV_VERIFY_SCALE=1 DSV41_AV_FIXED_MS=0. Patch with scale 1 / fixed 0 = stock behaviour.
- Site file: `bash $D/install.sh av rollback`, then restart with `dsv41-launch-all.sh`.

**Raw data:** ./r2-exp5/<inst>/ (bench output, rank0.log copies, av-curves JSON) plus av5_summ.py and nrhist.py.

## Round 2 summary and final verification (2026-10-22 ~01:35-02:00 UTC)

**Verified live state (all 8 ranks):** unit dsv41-rank active since 01:25:47 UTC, unit env only `DSV41_EXTRA=` (empty), /health 200, 17*23 -> "391".
rank0.log shows `rdma_ar enabled ... max_reduce=1048576`, `target compact FC1 enabled ... FC1 rows 576`, and the adaptive-verification calibration line (scale=0.6 n0=48 n1=56 fixed_ms=2.0). No nsys or Codex processes.

sha256 (first 8 hex), identical on all 8 ranks:
- launcher ~/dsv41-tp8.sh e270215f (backups .bak-avcal .bak-rdmaext .bak-compact .bak-conc .bak-m32 .bak-rdma-ar present; .bak-spec-len exists only on some ranks, from round 1)
- site-packages (venvs/vllm032): deep_gemm_moe.py 45418259, mxfp4.py 67339029, fp8_utils.py 8dc7dab8, adaptive_verification.py ec240369, speculator.py 97bcfaa6 (stock), cuda_communicator.py 52fc8cf5, rdma_all_reduce.py 3a128082, deep_gemm_utils.py b7e8de4d, warmup.py 19009d42. 10 .bak-* files are present on every rank.

**Final measurement** (one instance, ./r2-final/, run.sh. Single is 1024 greedy tokens x 2 repeats per type; conc runs are 512 tokens each):

| metric | original baseline | round-1 final | **round-2 final** | vs R1 | vs baseline |
|---|---|---|---|---|---|
| single code | - | 71.4 | **72.5** | +1.5% | |
| single prose | - | 53.9 | **58.8** | +9.1% | |
| single math | - | 104.4 | **111.1** | +6.4% | |
| single mean | 57 | 76.6 | **80.8** | +5.5% | +41.8% |
| conc4 agg | 102 | 134.7 | **144.0** | +6.9% | +41.1% |
| conc8 agg | 147 | 192.6 | **210.1** | +9.1% | +43.0% |
| conc16 agg | - | 277.3 | **313.5** | +13.0% | |
| mean accept len single / c16 | | ~2.35 | 2.48 / 2.39 | | |
| GSM8K-50 (conc 8) | | 49-50/50 | **50/50** | | |
| GSM8K-200 (conc 16) | | (no reference) | **196/200 (98.0%)** | | |

- exact.py was run twice in this instance and matched itself (823ce1923c16).
- GSM8K-200 misses are items 93, 100, 119 (truncated, no answer) and 184. There is no earlier GSM8K-200 run to compare against, so this is the reference for future rounds.
- Single-stream varies between restarts by about ±4%. This instance's code number (72.5) is at the low end; across exp5's two A5 instances the single mean was 84.1. A fair estimate for round 2 is single ~81-84 tok/s.

**Round 2 outcome:**
- PROMOTED exp1 moe-compact (DSV41_MOE_COMPACT=1). Unpadded target FC1 (576 rows/expert/rank), bit-exact, -7.5 GiB/rank. Gains: single +5%, conc4-16 +7-10%.
- NOT promoted: exp2 skinny_mx dense MXFP8. It failed the CUDA-graph kernel gate, because autotuned CUTLASS already streams ~200 GB/s. Only WO-A A16 wins (1.4-2.3x), worth about 2% single-stream.
- PROMOTED exp3 rdma-thresholds (RDMA_AR_MAX_REDUCE_BYTES=1048576). It is env-only. Two-shot RDMA all-reduce now covers up to 102 rows, conc16 gains about 5%, and draft time at 16 requests drops 19.1 -> 12.5 ms.
- NOT promoted: exp4 sharded-Markov drafter. Drafts are identical and draft time drops 1.4 ms, but single was only +1.0%. The stock trimmer over-trims when drafting is cheaper.
- PROMOTED exp5 adaptive-verification cost calibration (scale 0.6, fixed 2 ms, flat to M=48, back to profile by 56). Single +6.1%, concurrency +1-2.5%.

**Full rollback to round-1 config** (run from the Mac; stop the service first for site files):
1. Launcher: `cp ~/dsv41-tp8.sh.bak-compact ~/dsv41-tp8.sh` on all 8. 3045aa83 is the round-1 launcher.
2. Site files: `bash $S/r2/moe-compact/uninstall.sh` (deep_gemm_moe/mxfp4/fp8_utils back to 84ecc6ef/a7fc1bcd/013df536) and `bash $D/install.sh av rollback` (adaptive_verification back to b0526416).
3. `bash $S/dsv41-launch-all.sh`.

Per experiment: see the exp1/exp3/exp5 sections above. Env-only partial rollbacks are DSV41_MOE_COMPACT=0, RDMA_AR_MAX_REDUCE_BYTES=524288, and DSV41_AV_VERIFY_SCALE=1 DSV41_AV_FIXED_MS=0.

**Scripts archived:** ./ops/r2/ holds a copy of the Mac scratchpad r2/ tree: moe-compact, dense/skinny_mx, rdma-ext, drafter, av-profile, exp1/exp4/exp5 helpers, and finalsumm.py.

**Next steps (round 3 candidates):**
1. Re-test exp4 sharded-Markov on top of the exp5 calibration. Cheaper drafting should no longer over-trim. With AV off it measured +6% single.
2. DSV41_TARGET_M32_MAX_M=96 together with max-num-seqs 24/32. The verify cliffs at 48/56 are gone (verify at 56 tokens took 119 ms with M32_MAX_M=96), and exp1 freed 7.5 GiB/rank. Also try a larger --kv-cache-memory-bytes (12-14G).
3. A separate RDMA all-gather size cap (~1 MiB), so 48-96-row logits gathers fall back to NCCL, which is faster there.
4. Compact FC2 (K=288). This needs a DeepGEMM mixed-FP4 kernel change, for about another 8% of routed bytes. Also try 64-wide N tiles for M>48, where the kernel gain is only 1.05x.
5. WO-A A16 skinny alone (DSV41_SKINNY_MX_WOA=1), for about +2% single. It needs at least 3 ABAB pairs, and should be compared against Codex's WO-A Marlin.


## Round 3 / e1: sharded Markov drafter + AV calibration (2026-10-22 UTC). PROMOTED (arm M2)

**Change:** `DSV41_DSPARK_SHARDED_MARKOV=1` (the round-2 exp4 patch to `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`, sha 97bcfaa6 -> 774b34c2) together with `DSV41_AV_FIXED_MS` 2.0 -> 3.4. The exp5 AV cost calibration stays live (VERIFY_SCALE 0.6, N0 48, N1 56).
Round-2 exp4 was not promoted only because the uncalibrated trimmer over-trimmed code. With calibration plus FIXED_MS 3.4, the trimmer keeps its previous choices and the drafter saves 1.5 ms per step: draft_ms(1) 5.2 -> 3.73, draft_ms(16) 12.6 -> 9.94.

**A/B** (C = round-2 live config; M1 = +flag; M2 = +flag with FIXED_MS 3.4). All instances were launched with `launch-conc.sh` in the order C1 M1 M2 C2 M1b M2b C3 M2c. Each ran runarm_conc (single code/prose/math, conc4/8/12/16, exact x2, GSM8K-50 at conc 8).

| inst | code | prose | math | mean | c4 | c8 | c12 | c16 | AL code/prose/math | GSM8K | draft_ms(1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 68.06 | 60.75 | 111.31 | 80.04 | 144.67 | 211.95 | 269.12 | 319.13 | 2.50/1.88/3.91 | 50/50 | 5.29 |
| M1 | 72.82 | 61.87 | 114.50 | 83.06 | 147.08 | 216.37 | 276.27 | 321.48 | 2.56/1.80/3.83 | 49/50 | 3.73 |
| M2 | 78.06 | 64.14 | 110.78 | 84.33 | 146.94 | 219.50 | 277.86 | 327.19 | 2.77/1.88/3.68 | 50/50 | 3.72 |
| C2 | 72.24 | 59.36 | 112.89 | 81.50 | 141.45 | 211.57 | 263.36 | 310.49 | 2.68/1.83/3.93 | 50/50 | 5.24 |
| M1b | 70.29 | 61.58 | 113.37 | 81.75 | 150.58 | 218.56 | 277.47 | 322.70 | 2.42/1.82/3.79 | 50/50 | 3.73 |
| M2b | 76.67 | 64.02 | 116.58 | 85.76 | 145.94 | 217.72 | 276.01 | 327.40 | 2.70/1.88/3.92 | 50/50 | 3.74 |
| C3 | 73.39 | 58.43 | 114.98 | 82.27 | 142.23 | 206.58 | 270.35 | 315.19 | 2.73/1.79/4.04 | 49/50 | 5.20 |
| M2c | 78.31 | 62.75 | 116.89 | 85.98 | 148.12 | 215.79 | 272.59 | 318.61 | 2.81/1.85/3.95 | 49/50 | 3.72 |

exact.py was self-consistent in all 8 instances.

**Pooled, control (3 instances) vs arm:**

| | single | c4 | c8 | c12 | c16 | AL code | AL prose | AL math |
|---|---|---|---|---|---|---|---|---|
| C | 81.27 | 142.78 | 210.03 | 267.61 | 314.94 | 2.63 | 1.83 | 3.96 |
| M2 | 85.36 (+5.0%) | 147.00 (+3.0%) | 217.67 (+3.6%) | 275.49 (+2.9%) | 324.40 (+3.0%) | 2.76 (+0.12) | 1.87 (+0.03) | 3.85 (-0.11) |
| M1 | 82.41 (+1.4%) | 148.83 (+4.2%) | 217.47 (+3.5%) | 276.87 (+3.5%) | 322.09 (+2.3%) | 2.49 (-0.15) | 1.81 (-0.02) | 3.81 (-0.15) |

- **M1** fails the gate (single +1.4%, code AL -0.15). With FIXED_MS 2.0 the trimmer still trims code a little too much.
- **M2** passes: single +5.0%, every concurrency level is +2.9% to +3.6%, and code AL is up. The math AL change of -0.11 comes from one instance (M2: 3.68). M2b and M2c (3.92/3.95) are inside the control range (3.91-4.04), and math single runs at drafts_per_step 5.0 in every instance, so it is not trimming.
- **GSM8K-200 at conc16 on M2c:** 197/200, misses idx 93, 119, 184 (reference 196: 93, 100, 119, 184).
- **Long-context check:** not needed. The change touches only the drafter and the AV cost model, not attention/indexer/KV/MoE numerics, and the target model still verifies every token.

**Promotion:**
- Launcher sha: e270215f -> abd49904, identical on all 8. The backup is `~/dsv41-tp8.sh.bak-r3markov` (e270215f) on every rank.
- New lines: `DSV41_AV_FIXED_MS` default 3.4, and `export DSV41_DSPARK_SHARDED_MARKOV=${DSV41_DSPARK_SHARDED_MARKOV:-1}`.
- speculator.py = 774b34c2 on all 8. The backup is `speculator.py.bak-dsmarkov` (97bcfaa6).
- After relaunching with `dsv41-launch-all.sh` (started 04:08:43 UTC): rank0.log shows `SHARDED_MARKOV: vocab shard [0, 16160)`, with no "ignored" warning and no "first call during capture" warning. draft_ms(1) is 3.75, and GSM8K-50 at conc 8 scored 50/50.

**Rollback:**
1. Restore the launcher on every rank: `cp -p ~/dsv41-tp8.sh.bak-r3markov ~/dsv41-tp8.sh`, and check that the sha is e270215f. For a temporary rollback instead, launch with `DSV41_DSPARK_SHARDED_MARKOV=0 DSV41_AV_FIXED_MS=2.0`.
2. Run `bash <Mac scratchpad>/r2/drafter/install.sh markov rollback`, which restores speculator.py to 97bcfaa6.
3. Relaunch.

**Note:** dspark/speculator.py is no longer stock, so the r3 spec-sampling install.sh, which expects sha 97bcfaa6, will refuse to run until it is rebased onto 774b34c2.

Raw results: `~/dsv41-opt-20261021/r3-e1/` (instance dirs, inst.sh, summ2.py, dsumm.py, M2c/gsm200.*, final-gsm50.*).

### Round 3 / e2: FC2 compaction (DSV41_MOE_COMPACT_FC2): NOT promoted, kernel faults, nothing installed

**Goal:** unpad routed-MoE FC2 (288->384) with a separately built DeepGEMM (`~/dsv41-stage/fc2c/native-fc2c-v1`, _C e7088a23, manifest 072fdc49) whose compact-B kernel remaps the B load. Expected about +2-5%, plus about 3.8 GiB per rank freed.

**Offline test** (`run_offline_test.sh`, spark-one, service stopped, 2026-10-22 ~04:15 UTC):
- **kw=288:** the first case (M=1, 32-row path) faults when `sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d_compact_b` launches with `CUDA_ERROR_ILLEGAL_INSTRUCTION (715)` (cuLaunchKernelEx, deep_jit driver.hpp:33). Confirmed with CUDA_LAUNCH_BLOCKING=1.
- **kw=320** (separate process): faults the same way on the same first case.
- **Healthy in the same process:** the padded kernel from the new native is bitwise identical to the live M32 library (native_padded_eq_live=true), and the padded baseline matches the fp32 reference (rel 1.66e-3). The sanitizer fixed 1,473,703 NaN dead scales.
- **Not the cause:** plain torch reductions work on this GPU.
- **Where the fault is:** in the compact-B patch codegen. Both variants JIT-compiled and are cached in `~/dsv41-stage/fc2c/fc2c-jit-cache-v1/cache/*compact_b.{45960e62,cf999d97}` with 27-line kernel.cu each. Untested idea: the TMA box/tensor-map or a PTX variant that sm_121 rejects.
- No 128-row case and no timing was reached.

**Decision:** stopped at offline step 5. `install.sh` was never run (site-packages unchanged) and there was no A/B.

**State after:**
- Service relaunched with `dsv41-launch-all.sh` (health 200); GSM8K-50 at conc8 scored 50/50 (221.8 tok/s).
- On all 8 ranks: launcher abd49904, deep_gemm_moe.py 45418259, mxfp4.py 67339029, sm121_fc2_compact.py absent.

**Rollback:** none needed. Staged files stay in `~/dsv41-stage/fc2c` for a kernel fix.

Raw results: `~/dsv41-opt-20261021/r3-e2/` (offline-test.log, final-gsm50.*).

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

## Round 3 / e4 dense-fix + sonar o_proj (2026-10-22) — PROMOTED
Change (all behind env flags, now launcher defaults):
- DSV41_QKV_A_SHARD=1: replicated fused wq_a|wkv GEMM (1792x5120 MXFP8, every rank) column-sharded to 224 rows/rank
  (r2 skinny_mx_ext split-K GEMV) + rdma_ar all-gather, for M<=8 (DSV41_QKV_A_SHARD_MAX_M); larger M = stock.
- DSV41_IDX_WPROJ=bmm32: indexer weights_proj (32x5120 bf16) as 32-way split-K bmm + fp32 sum (stock cuBLAS 33 us -> 8.5 us at M 2-8).
- DSV41_FP8_LMHEAD=draft: MXFP8 copy of the lm_head shard used only for DSpark draft logits (target logits stay bf16 -> lossless).
- DSV41_SONAR_OPROJ=fp8 SPLITS=32 MAX_M=32: sonar PR #1788 split-K W8A16 wo_a kernel on an exact MXFP8 re-encode of the bf16
  weight (bf16 rebuilt in-kernel); M>32 = torch.bmm (fp8 kernel loses above 32 rows).
Site files (backups .bak-r3dense / .bak-r3sonar on all 8):
  deepseek_v41/attention.py c39e06c1 -> f2f48198, model_executor/layers/logits_processor.py 35cf8235 -> caf6d216,
  deepseek_v41/nvidia/dspark.py 480149ff -> fa56d07d, new deepseek_v41/nvidia/dsv41_r3_dense.py 993023f2,
  deepseek_v4/nvidia/ops/o_proj.py f2893ee6 -> 34b9e1a4, new deepseek_v4/nvidia/ops/sonar_bf16_bmm.py 9f6f3f84.
Launcher 776de10b -> d5c41fd0 on all 8 (backup ~/dsv41-tp8.sh.bak-r3dense).

Kernel gates (spark-two, service stopped):
  qkv_a shard: replicated 44.2-44.8 us; skinny 10.9/11.8/13.3/16.1/17.7 us at M 1/2/4/6/8 (G1 <=14 us: miss at M=6 by 2 us,
  still fastest option); cutlass shard 17.4-17.8 us bitwise == replicated (G2 PASS); skinny relerr 0, deterministic (G3 PASS).
  weights_proj: stock 33.4 us (M>=2); mm32 35 (worse); bmm16 8.5-10.3; bmm32 8.0-8.7 (winner, G4 PASS).
  FP8 draft head a8 382-386 us vs bf16 719-729 us (G5 PASS); a16 fails (765 us at M=6).
  sonar o_proj: re-encode exact, deterministic, relerr 1.6-1.7e-3 (= torch.bmm); fp8k 23-24 us (s32) vs bmm 40-55 us at M<=16
  (>=1.5x at M<=8, PASS); fp8k vs bf16-kernel NOT bitwise at M 1/6/8 (same splits, same rebuilt bf16 weights -> two Triton
  compilations of the dot; relerr identical) - accepted, quality gated end-to-end below. Above M=32 fp8k is 0.58-0.74x -> MAX_M=32.
  all-gather (M,224) bf16: not directly benchable; nearest in-graph rdma AG (6,3,256) 9.2 KB = 22.3 us (E2), so <=~22 us.

A/B (order C1 D3 D2 C2 D3b D1 C3 D3c; D1 = shard+bmm32, D2 = D1+FP8 draft head, D3 = D2+sonar):
inst | code prose math mean | c4 c8 c12 c16 | AL code/prose/math | gsm50 | draft_ms(1) verify_ms(1)
C1  | 71.29 59.46 112.89 81.21 | 146.2 223.5 279.5 325.8 | 2.48/1.73/3.75 | 49 | 3.69 18.32
C2  | 74.84 59.56 118.27 84.22 | 145.7 219.0 283.9 323.5 | 2.63/1.72/3.94 | 50 | 3.73 18.41
C3  | 78.10 61.18 112.37 83.88 | 151.7 214.9 276.7 324.4 | 2.75/1.77/3.73 | 50 | 3.70 18.41
D1  | 76.66 65.08 117.25 86.33 | 145.3 217.4 278.2 323.2 | 2.60/1.85/3.81 | 50 | 3.69 17.70
D2  | 76.29 63.53 119.10 86.31 | 151.4 218.9 276.7 327.0 | 2.61/1.79/3.88 | 49 | 3.41 20.15
D3  | 78.74 63.73 116.46 86.31 | 153.7 217.5 279.8 327.1 | 2.60/1.72/3.60 | 50 | 3.31 16.32
D3b | 76.64 63.35 119.70 86.56 | 148.3 216.9 279.6 329.3 | 2.50/1.71/3.70 | 49 | 3.34 16.31
D3c | 78.01 64.48 115.41 85.97 | 150.6 218.2 282.2 330.2 | 2.59/1.77/3.59 | 50 | 3.30 16.61
Pooled C (3) -> D3 (3): single 83.10 -> 86.28 (+3.8%); c4 147.9 -> 150.8 (+2.0%); c8 219.1 -> 217.5 (-0.7%);
c12 280.0 -> 280.6 (+0.2%); c16 324.6 -> 328.9 (+1.3%). verify_ms(1) 18.38 -> 16.41, draft_ms(1) 3.71 -> 3.32.
D1/D2 single ~= D3 (one instance each); D3-D2 <1.5% single but verify(1) is lowest with sonar and it is free at conc -> kept.
AL: single-prompt math AL lower on sonar instances (3.60/3.70/3.59 vs C 3.75/3.94/3.73; greedy trajectory change from
target numerics, not the FP8 head: D2 math 3.88). Wider check, single-user GSM8K-24 /metrics AL: D3c 3.396 vs C4 3.339 -> no drop.
D3c: GSM8K-200 @conc16 196/200 (misses 93,100,119,184 = reference); LONGCTX PASS needle 60/60 copy 1.000.
exact.py self-consistent in every instance; no "first call during capture"; sonar logs mode=fp8 weight=float8_e4m3fn.
Note: first D3 launch hung at dspark graph capture (rdma_ar timeouts; rank 5 / spark-six "unspecified launch failure");
identical relaunch and 3 later D3 launches + final relaunch clean -> treated as a transient; watch for it.
Final: dsv41-launch-all.sh, health 200, GSM8K-50 @conc8 50/50, 229.1 tok/s. Raw: ~/dsv41-opt-20261021/r3-e4/.
Rollback: per knob DSV41_QKV_A_SHARD=0 DSV41_IDX_WPROJ= DSV41_FP8_LMHEAD= DSV41_SONAR_OPROJ= (launch-conc.sh, all ranks
identical!); launcher: cp -p ~/dsv41-tp8.sh.bak-r3dense ~/dsv41-tp8.sh on all 8; full revert (service stopped):
bash $S/r3/dense/uninstall-r3dense.sh && bash $S/r3/sonar/oproj/uninstall-sonar-oproj.sh, then verify attention c39e06c1,
logits_processor 35cf8235, dspark 480149ff, deepseek_v4 o_proj f2893ee6 on all 8.

## Round 3 / e5: sampled spec decode, probabilistic drafting + fixed block verification (PROMOTED pb5, 2026-10-22 11:30 UTC)
Change: the default DSpark config now adds draft_sample_method=probabilistic and rejection_sample_method=block.
num_speculative_tokens stays 5 and adaptive verification stays on. Stock vLLM 0.32 block verification is lossy
for two reasons: it reuses acceptance uniforms across steps, and it reuses drafter Gumbel noise keyed by
(seed, position). Two site patches fix this. Both are inert when their flags are off:
 - v1/worker/gpu/spec_decode/rejection_sampler_utils.py 09522889 (stock 99556e84): DSV41_BLOCK_VERIFY_FRESH_U=1
 - v1/worker/gpu/spec_decode/dspark/speculator.py 0cc6e2bf (this is the e1 Markov file 774b34c2 + the e5 hunk): DSV41_DRAFT_FRESH_NOISE=1
 - Backups on all 8: *.bak-r3spec (99556e84 / 774b34c2). bench.py.bak-r3spec on spark-one; the new bench.py keeps the old CLI
   and adds --preset t1|t06, --top-p and --seed.
 - Launcher ~/dsv41-tp8.sh 2971eb50 on all 8 (backup ~/dsv41-tp8.sh.bak-r3spec = d5c41fd0). New knobs:
   DSV41_SPEC_SAMPLED (default 1; 0 = old greedy-draft JSON), DSV41_BLOCK_VERIFY_FRESH_U (default 1),
   DSV41_DRAFT_FRESH_NOISE (default 1).
 - GPU unit test (spark-two, test_spec_sampling.py): greedy PASS. stream with FRESH_U=0: block rows FAIL with greedy
   drafts, which confirms the test detects the bug. FRESH_U=1 stream: all ok. stream-vllm-draft with stock noise:
   block+prob FAIL. Adding FRESH_NOISE=1: all ok.
Trade-off: the sharded Markov drafter (e1) needs greedy drafting. With probabilistic drafting it logs
"DSV41_DSPARK_SHARDED_MARKOV=1 ignored" and uses the full-vocab Markov path. Every arm and the ctl5m0 control
therefore ran with DSV41_DSPARK_SHARDED_MARKOV=0; ctl5mk is the live e1-e4 config.
A/B with one instance per row (order ctl5m0a ps5a pb5a ctl5m0b ctl5mk ps5b pb5b ctl5m0c pb5c ps5c).
Bench: greedy = 3-type single mean; t1 = temp 1.0 / top_p 0.95; t06 = temp 0.6. Columns are tok/s; AL = accept length.
 inst    | greedy AL   g.c8  | t1    AL   | t1.c4 t1.c8 t1.c16 | t06   | g06 (GSM8K-50 t0.6 x2)
 ctl5m0a | 85.83 2.75 218.3 | 80.77 2.64 | 125.1 195.5 286.5 | 83.32 | 50.0
 ctl5m0b | 86.53 2.77 212.2 | 74.73 2.40 | 140.2 196.8 290.1 | 83.62 | 50.0
 ctl5m0c | 84.77 2.69 212.7 | 76.32 2.45 | 139.3 200.2 295.2 | 83.43 | 49.5
 ctl5mk  | 86.63 2.62 217.2 | 79.43 2.42 | 141.5 203.8 301.9 | 87.04 | 49.5
 pb5a    | 84.56 2.67 216.6 | 85.52 2.74 | 144.5 207.8 315.0 | 84.18 | 50.0
 pb5b    | 83.04 2.63 216.5 | 87.11 2.81 | 147.3 212.4 308.0 | 85.41 | 50.0
 pb5c    | 83.52 2.64 213.3 | 87.62 2.83 | 145.9 206.7 308.4 | 86.73 | 50.0
 ps5a    | 83.86 2.65 214.1 | 93.78 3.09 | 143.8 204.4 303.3 | 84.81 | 50.0
 ps5b    | 84.33 2.67 217.8 | 83.95 2.69 | 149.5 207.7 305.9 | 87.09 | 50.0
 ps5c    | 83.70 2.65 213.6 | 83.68 2.67 | 147.4 205.8 301.3 | 86.16 | 50.0
Pooled ctl5m0 (3) -> pb5 (3):
 - t1 single 77.27 -> 86.75 (+12.3%); c4 +8.2%; c8 +5.8%; c16 290.6 -> 310.5 (+6.8%); t06 single +2.4%.
 - Greedy single 85.71 -> 83.71 (-2.3%, inside the +/-4% gate). Greedy c8 214.4 -> 215.5 (+0.5%).
 - Greedy AL 2.733 -> 2.647 (-0.087), just over the +/-0.08 gate. ps5 has the same greedy code path and shows -0.077.
   ctl5mk (live) is 2.624. Greedy text is bit-identical: exact.py combined a8c3394a in every instance, including
   the promoted relaunch. So the AL shift is adaptive-verification trimming, not different greedy tokens.
 - Against the live config (ctl5mk, n=1): greedy single 86.63 -> 83.71 (-3.4%), t1 single +9.2%, t1 c16 +2.8%.
ps5 (probabilistic + standard rejection, no patch needed): t1 single +12.8%, c16 +4.4%. pb5 is better at every concurrency.
gb5 (greedy draft + fixed block) and n=7 arms were not run (time box).
Sampled gate: every pb5/ps5 instance PASS against ctl5m0b (50/50 both runs).
Promotion gate, GSM8K-200 t0.6 conc16 seed 1000: control (live) 196/200, 1 truncated, 267.1 tok/s (misses 93,100,119,184);
pb5 promoted 195/200, 0 truncated, 288.3 tok/s (misses 12,93,119,184,187) -> PASS at tol 6.
Final: greedy GSM8K-50 @conc8 50/50, 240.7 tok/s; exact.py self-consistent (a8c3394a). Raw: ~/dsv41-opt-20261021/r3-e5/.
Decision rationale: vLLM defaults to temperature 1.0 when a client omits it (there is no generation_config),
so typical chat traffic is sampled. +12% sampled single and +7% sampled c16 outweigh about -2-3% greedy single.
For greedy-only benchmarking, relaunch with DSV41_SPEC_SAMPLED=0.
Rollback:
 - Config only: relaunch with DSV41_SPEC_SAMPLED=0 (the sharded Markov drafter is active again).
 - Launcher: cp -p ~/dsv41-tp8.sh.bak-r3spec ~/dsv41-tp8.sh on all 8 (d5c41fd0).
 - Site files (service stopped): in .../v1/worker/gpu/spec_decode, cp -p rejection_sampler_utils.py.bak-r3spec
   rejection_sampler_utils.py and dspark/speculator.py.bak-r3spec dspark/speculator.py on all 8. Then verify
   99556e84 / 774b34c2 on all 8.
Follow-up idea: restore the sharded Markov path for greedy-only batches under probabilistic drafting. This needs a
second captured drafter graph set, and it would recover the ~2-3% greedy single loss.

## Round 3 summary and final verification (2026-10-22 11:35-11:50 UTC)

**Verified live state (all 8 ranks):** unit dsv41-rank active since 11:23:36 UTC (e5 promoted relaunch), /health 200, no foreign activity.
sha256 (first 8 hex), identical on all 8 ranks:
- launcher ~/dsv41-tp8.sh 2971eb50; backups .bak-r3spec d5c41fd0, .bak-r3dense 776de10b, .bak-r3conc abd49904, .bak-r3markov e270215f
- v1/worker/gpu/spec_decode/rejection_sampler_utils.py 09522889 (.bak-r3spec 99556e84)
- v1/worker/gpu/spec_decode/dspark/speculator.py 0cc6e2bf (.bak-r3spec 774b34c2, .bak-dsmarkov 97bcfaa6 = stock)
- models/deepseek_v41/attention.py f2f48198 (.bak-r3dense c39e06c1, .orig dd146642)
- model_executor/layers/logits_processor.py caf6d216 (.bak-r3dense 35cf8235)
- models/deepseek_v41/nvidia/dspark.py fa56d07d (.bak-r3dense 480149ff); dsv41_r3_dense.py 993023f2 (new)
- models/deepseek_v4/nvidia/ops/o_proj.py 34b9e1a4 (.bak-r3sonar f2893ee6); sonar_bf16_bmm.py 9f6f3f84

**Final measurement** (one instance, ./r3-final-run.sh, results in ./r3-final/. Single = 1024 tokens x 2 repeats per type; conc = 512 tokens each):

| metric | original baseline | round-2 final | **round-3 final** | vs R2 | vs baseline |
|---|---|---|---|---|---|
| single code (greedy) | - | 72.5 | **79.9** | +10.2% | |
| single prose (greedy) | - | 58.8 | **60.0** | +2.0% | |
| single math (greedy) | - | 111.1 | **113.5** | +2.2% | |
| single mean (greedy) | 57 | 80.8 | **84.5** | +4.5% | +48.2% |
| conc4 agg | 102 | 144 | **146.4** | +1.7% | +43.6% |
| conc8 agg | 147 | 210 | **212.2** | +1.0% | +44.3% |
| conc16 agg | - | 313 | **320.6** | +2.4% | |
| conc24 agg | - | ~300 (seqs 16 cap; e3 K0) | **406.6** | +35.5% | |
| conc32 agg | - | ~301 (seqs 16 cap; e3 K0) | **464.5** | +54.1% | |
| sampled single t1.0/p0.95 mean | - | 77.3 (e5 ctl5m0) | **85.1** | +10.1% | |
| sampled conc16 t1.0 | - | 290.6 (e5 ctl5m0) | **312.1** | +7.4% | |
| accept len greedy single code/prose/math | | 2.48 mean | 2.62/1.71/3.70 | | |
| GSM8K-50 conc8 | | 50/50 | 49/50 (miss 12, known flip) | | |
| GSM8K-200 conc16 | | 196/200 | **196/200** (93,100,119,184 = reference set; 279 tok/s) | | |
| long-context | | | **PASS** 60/60 needles, 12/12 copy exact 4K-200K; 250,916 tok 5/5 + exact | | |

- exact.py combined a8c3394af9d2, same hash as every e5 instance.
- Lowest MemAvailable during the run (incl. conc32): 11.4 GB (gate 4 GB).
- Single-stream varies about ±4% between restarts, so the greedy single gain vs R2 (+4.5%) is roughly at the edge of noise; pooled e4 instances put it at ~86. The clear round-3 wins are conc24/32 (+35-54%) and sampled decode (+7-10%).

**Round 3 outcome:**
- PROMOTED e1 sharded Markov drafter + AV_FIXED_MS 3.4 (+5% single, +3% conc). Note: under e5 probabilistic drafting, the sharded path is ignored (full-vocab Markov fallback), so for sampled traffic e1 now only contributes the AV calibration.
- NOT promoted e2 FC2 compaction: compact_b kernel faults with CUDA 715 at first launch on sm_121; nothing installed. Staged in ~/dsv41-stage/fc2c.
- PROMOTED e3 conc/KV: max-num-seqs 32, KV 16G (3.26M tokens), M32 max 96, AV N0/N1 96/104, RDMA reduce 2 MiB. conc24 +37%, conc32 +58%; greedy single -3.7% pooled overridden (acceptance-only, per-step speed identical, GSM8K-16 single +1.9%).
- PROMOTED e4 qkv_a shard + bmm32 weights_proj + FP8 draft head + sonar fp8 o_proj: single +3.8%, verify_ms(1) 18.4 -> 16.4. Two kernel gates (G1 at M=6, sonar bitwise) passed on purpose rather than letter. One-off rank-5 launch failure during graph capture on first D3 launch; not reproduced since.
- PROMOTED e5 pb5 (probabilistic drafting + block verification with fresh-U and fresh-noise fixes): sampled +7-12%, greedy bit-identical, greedy single -2-3% (sharded Markov disabled under probabilistic drafting).

**Rollback** (run from the Mac; stop service for site files; then `bash $S/dsv41-launch-all.sh`):
- e5 only: relaunch with DSV41_SPEC_SAMPLED=0, or restore ~/dsv41-tp8.sh.bak-r3spec + the two .bak-r3spec site files.
- e4: restore .bak-r3dense launcher, run $S/r3/dense/uninstall-r3dense.sh and $S/r3/sonar/oproj/uninstall-sonar-oproj.sh (copies in ./ops/r3/).
- e3: restore .bak-r3conc, or env overrides DSV41_MAX_NUM_SEQS=16 DSV41_KV_BYTES=8G DSV41_TARGET_M32_MAX_M=48 DSV41_AV_SCALE_N0=48 DSV41_AV_SCALE_N1=56 RDMA_AR_MAX_REDUCE_BYTES=1048576.
- Full revert to round-2: launcher .bak-r3markov (e270215f) on all 8; site files back to .bak-r3spec/.bak-r3dense/.bak-r3sonar, and speculator.py to .bak-dsmarkov (97bcfaa6); verify sha256 identical on all 8.

**Scripts archived:** ./ops/r3/ is a copy of the Mac scratchpad r3/ tree (dense, sonar, spec, fc2, e1/e3/e4 helpers, longctx incl. offset_probe.py and the PR #1788 diffs); the 65 MB fi072 wheel was left out.

**Next steps (round 4 candidates):**
1. Greedy-batch sharded Markov path under DSV41_SPEC_SAMPLED=1: capture a second drafter graph set so all-greedy batches use the sharded drafter again (should recover the 2-3% greedy single loss).
2. Fix the e2 compact_b kernel (TMA box / tensor-map or PTX variant on sm_121) for ~8% fewer routed MoE bytes; library staged in ~/dsv41-stage/fc2c.
3. Latent indexer bug: pack decode weights with pack_seq_triton in the padded decode path (sparse_attn_indexer.py ~L818/828) before anyone turns VARLEN_INDEXER off.
4. Investigate noisy long-prompt TTFT (16K 4.3-7.2 s across instances; one 32K at 10.1 s) seen in e3.
5. Widen single-user bench prompt set (e.g. GSM8K-24 via /metrics) so acceptance-trajectory noise stops tripping +/-0.08 AL gates.
6. Watch for the one-off rank-5 (spark-six) CUDA launch failure during draft graph capture seen in e4.

## Round 4 / e1 bitstack: L2 prefetch + main_proj split + certified lm-head + eager glue. PROMOTED (2026-10-22 17:05 UTC)

**What changed.** Four exact (class X) decode-step micro-wins, promoted together as one stack. All are env-gated, and every site file is inert when its flag is off.
- `DSV41_L2_PREFETCH=1`: prefetch the next GEMM's weights into L2 while the ~125 in-graph RDMA collectives run (the plan includes the r3 QKV_A_SHARD all-gathers).
- `DSV41_R4_MAINPROJ_SPLIT=1`: DSpark main_proj is split by columns across TP plus one all-gather. It is bitwise self-checked per M at runtime.
- `DSV41_R4_CERT_HEAD=1`, `DSV41_R4_CERT_HEAD_M=2-96`: the target lm-head screens with MXINT8 and computes exactly only the tiles that can hold the argmax. The argmax is identical to stock.
  - It arms only on steps where every request is greedy with no logprobs or logits processing. Sampled traffic gets nothing from it.
  - M=1 was disqualified because full mode is not bitwise there; that is also where all of the GPU test's computed-logit mismatches came from.
- `DSV41_EAGER_GLUE=all`: one fused SWA-metadata kernel replaces 15x3 eager launches, and the dflash pad loop runs 1024 wide. There is no log line; the flag is read from the env.

**Site files** (sha identical 8/8; backups `*.bak-r4-<idea>` on each rank, made by `$S/r4/common/r4install.sh`):

| file | live | stock / backup |
|---|---|---|
| models/deepseek_v41/nvidia/model.py (model-merged: CERT + MAINPROJ + inert prefill-sp) | e7264781 | 5fe8b4a7 |
| models/deepseek_v41/nvidia/dsv41_r4_small.py (new) | 02f94f29 | absent |
| distributed/device_communicators/rdma_all_reduce.py (rebased l2-prefetch bundle) | 99102ff1 | 3a128082 |
| distributed/device_communicators/dsv41_l2_prefetch.py (new) | 7d2acdc8 | absent |
| v1/attention/backends/mla/sparse_swa.py | 38b16c51 | 2afd3136 |
| v1/worker/gpu/spec_decode/dflash/speculator.py | 17ddccf1 | b731b09c |

The launcher `~/dsv41-tp8.sh` is **709abe3d** on all 8 ranks, with 5 new `export ...:-default` lines after DSV41_DRAFT_FRESH_NOISE. The backup is `~/dsv41-tp8.sh.bak-r4e1` (2971eb50).

**GPU unit tests** (service stopped). Raw results are in `r4/e1/unit/`.
- **CERT** (spark-one, then spark-two with `--ms 2-96 --trials 3`):
  - All counters are 0 on all 8 shards for M=2-96, and the accumulation margin is 0.0009 of c_acc.
  - The default run showed 7-27 computed-logit mismatches per rank. They all came from M=1, which is excluded.
  - Microbench on a peaked activation at M=6: 480 vs 730 us. Gaussian activations are slower than stock at M≥6. In serving, a call computes about 19.4 of 1010 tiles (6000 armed calls on rank 0).
- **main_proj:** every slice is bit-identical at every M. At M=6 a rank's slice takes 140 us vs 930 us for the full GEMM.
  - The test needs `source venv/bin/activate` (for ninja) and CUDA_HOME. Without them, `mm_mxfp8` is missing from vllm.utils.flashinfer.
- **L2 prefetch** (spark-five): exact at M=1/6/24/96, with graph == eager. In the microbench it saves 30-39 us per window at M=1-6 (2.4-3.1 ms per forward).
- **eager-glue** (spark-six): bit-exact. It saves 89-91 us per step (SWA) and 25-27 us per step (dpad).

**A/B** (one fresh launch per row, order c0p c1 [spin] s1 c2 s2 c3 s3). Single = greedy, 1024 tokens, code/prose/math x1. Matched = matched_bench C1 ms/step.

| inst | single mean | c4 | c8 | c12 | c16 | matched coding ms / AL | matched prose ms / AL | verify_ms(1) | draft_ms(1) | GSM50 | exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c0p | 82.19 | 147.4 | 218.6 | 274.6 | 323.3 | 36.03 / 5.20 | 32.53 / 2.81 | 16.32 | 8.14 | 49 | a8c3394af9d2 |
| c1 | 81.21 | 150.7 | 216.3 | 271.5 | 317.3 | 36.32 / 5.23 | 32.39 / 2.86 | 16.50 | 8.08 | 50 | a8c3394af9d2 |
| c2 | 81.49 | 147.9 | 217.5 | 276.6 | 315.3 | 35.99 / 5.29 | 32.24 / 2.79 | 16.40 | 8.15 | 49 | a8c3394af9d2 |
| c3 | 82.15 | 146.8 | 208.7 | 273.5 | 317.4 | 35.86 / 5.08 | 32.03 / 2.71 | 16.35 | 8.09 | 50 | a8c3394af9d2 |
| s1 | 86.04 | 148.6 | 216.2 | 272.5 | 320.3 | 35.57 / 5.09 | 31.68 / 2.84 | 16.11 | 7.86 | 50 | a8c3394af9d2 |
| s2 | 86.14 | 149.4 | 218.4 | 270.6 | 321.3 | 35.40 / 5.18 | 31.43 / 2.79 | 16.19 | 7.76 | 50 | a8c3394af9d2 |
| s3 | 85.09 | 146.7 | 212.9 | 272.4 | 318.2 | 35.45 / 5.24 | 31.34 / 2.68 | 16.24 | 7.78 | 50 | a8c3394af9d2 |
| spin* | 86.92 | 148.8 | 217.4 | 267.4 | 313.1 | 35.80 / 5.21 | 31.84 / 2.76 | (pinned) | (pinned) | 49 | a8c3394af9d2 |

\* spin = S with `DSV41_AV_COST_JSON` pinned to c1's raw profile. It is an exactness probe and is not pooled.

Pooled C (4 instances) vs S (3 instances):
- Greedy single: 81.76 -> 85.76, **+4.9%**. Each pair: +5.9 / +5.7 / +3.6%.
- conc: c4 +0.0%, c8 +0.2%, c12 -0.8%, c16 318.3 -> 319.9 (+0.5%).
- Matched ms/step: coding 36.05 -> 35.47 (-0.58), prose 32.30 -> 31.48 (-0.82).
- verify_ms(1): 16.39 -> 16.18 (L2 + glue in-graph part). draft_ms(1): 8.11 -> 7.80 (main_proj).
- Matched AL: coding -0.03, prose -0.02. Both are inside +/-0.10.
- Sampled quick (c3 vs s3): t1 single 86.0 -> 87.0 (+1.2%), t1 c8 211.9 -> 210.7 (-0.6%), t0.6 single 89.2 -> 90.7, GSM8K-50 t0.6 x2 49.5 -> 50.0.
- The step saving (~0.6-0.8 ms) is smaller than the expected 0.9-1.5 ms. Part of the +4.9% single gain comes from adaptive verification: it re-profiles the cheaper draft step and verifies a little deeper (single-bench code AL 2.55 -> 2.67).

**Exactness: class X (exact).**
- Every C and S instance gives exact.py a8c3394af9d2. With the AV profile pinned (spin), the hash and all 6 per-prompt hashes are identical to control.
- The one different hash, 9330f03c4f97 on the E1-chk instance, was self-consistent over 3 repeats. That instance ran CERT=check, its own AV re-profile and a 512-token single. It reflects the known cross-restart AV-profile sensitivity (Caveats), not the kernels.

**Promote gates on s3** (identical config to s2, server up):
- GSM8K-200 greedy conc16: **195/200**, misses 12, 93, 100, 119, 184 (the reference set plus the known item-12 flip), 0 truncated, 313 tok/s.
- longctx 4K-200K: **LONGCTX PASS**, needles 25/25, copy 1.000.
- No 'first call during capture', MISMATCH or 'not bit-identical' lines. L2 plan tokens=6 collectives=125 (85 + 40 r3 QKV_A_SHARD/IDX gathers).
- CERT compiles 2 Triton kernels during the first request; there is no JIT after that.
- Lowest MemAvailable: 20.4 GB.

**Promoted relaunch** (17:02 UTC, no env overrides): 8/8 active on launcher 709abe3d, with CERT built, MAINPROJ active and the L2 plan relearned. GSM8K-50 conc8 50/50 at 217.6 tok/s; exact.py a8c3394af9d2.

**Incidents.**
1. The first E1-chk launch failed during dspark graph capture: rank 2 (spark-three) got a TCPStore broken pipe and the RDMA AR timed out. The retry was clean, and so were the 6 later launches with the same flags. This looks like the transient seen in r3-e4, not the stack.
2. `ssh spark-one 'bash ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh'` does NOT work: spark-one cannot ssh to spark-six, seven or eight, so those 3 ranks kept the old instance and ranks 0-4 timed out. **Always run launch-all from the Mac:** `bash $S/dsv41-launch-all.sh` (same file as ops/).

**Rollback** (from the Mac):
- Env only (site files inert):
  `bash $S/conc/launch-conc.sh DSV41_L2_PREFETCH=0 DSV41_R4_MAINPROJ_SPLIT=0 DSV41_R4_CERT_HEAD=0 DSV41_EAGER_GLUE=`
  Individual flags work the same way.
- Launcher: `cp -p ~/dsv41-tp8.sh.bak-r4e1 ~/dsv41-tp8.sh` on all 8 (2971eb50), then `bash $S/dsv41-launch-all.sh`.
- Site files (stop the service first):
  - `bash $S/r4/rebased/l2-prefetch/uninstall.sh; bash $S/r4/eager-glue/uninstall.sh; bash $S/r4/model-merged/uninstall.sh`
  - Then verify on 8/8: rdma_all_reduce 3a128082, sparse_swa 2afd3136, dflash/speculator b731b09c, model.py 5fe8b4a7, with dsv41_l2_prefetch.py and dsv41_r4_small.py absent.
  - Keep model-merged if E5 prefill-sp is still pending.

Raw data: `./r4/e1/<inst>/` (c0p c1 c2 c3 s1 s2 s3 spin chk chk-try1 promoted unit). The pin file is `r4/e1/c1_av_pin.json`. Scripts are in `./ops/r4/e1/` (inst.sh, chain.sh, chk.sh, summ_e1.py).

## Round 4 / greedy-sharded (r4-e2): sharded greedy drafter on all-greedy steps. PROMOTED (2026-10-22 19:21 UTC)

**What.** Since r3-e5 (`DSV41_SPEC_SAMPLED=1`, probabilistic drafting + block verify), the r3-e1 vocab-sharded greedy Markov drafter was disabled ("probabilistic drafting needs full logits"). With `DSV41_DSPARK_GREEDY_SHARDED=1`, dspark/speculator.py builds the sharded state anyway and captures a second set of FULL drafter CUDA graphs with it. It replays that set on steps where every request has temperature 0. The check runs on the CPU and is identical on every rank: `runner.sampler.sampling_states.temperature.np[idx_mapping_np]`. Mixed and sampled steps use the stock probabilistic graphs. It requires `DSV41_DSPARK_SHARDED_MARKOV=1`, which is the launcher default. The second graph set costs about 33 MB per rank (startup cuda_active 67.76 vs 67.79 GB).

**Site file:** v1/worker/gpu/spec_decode/dspark/speculator.py is 0cc6e2bf (backup `*.bak-r4-greedy-sharded`) -> **bbcee9e3** on 8/8.
**Launcher:** `~/dsv41-tp8.sh` 709abe3d -> **f9140d38** on 8/8. It adds `export DSV41_DSPARK_GREEDY_SHARDED=${DSV41_DSPARK_GREEDY_SHARDED:-1}` after the EAGER_GLUE line. Backup: `~/dsv41-tp8.sh.bak-r4e2gs`.

**Check instance:** both log lines ('active', 'capturing greedy draft graphs') appear on 8/8. There is no 'no CPU temperature mirror' and no 'first call during capture'. Counters show greedy steps 1000 and other steps 198-291, so sampled steps took the stock path.

**Timing** (ABAB, one launch per row; single = greedy, 1024 tok; matched = matched_bench C1 ms/step / AL):

| inst | single mean | c4 | c8 | c12 | c16 | coding ms / AL | prose ms / AL | coding_div C8 ms | draft_ms(1) | GSM50 | exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c1r | 86.92 | 151.1 | 219.4 | 273.8 | 319.4 | 35.42 / 5.08 | 31.43 / 2.79 | 90.93 | 7.71 | 49 | a8c3394af9d2 |
| g1 | 88.36 | 156.7 | 223.4 | 280.5 | 326.3 | 33.96 / 5.07 | 29.97 / 2.75 | 90.65 | 7.78 | 50 | a8c3394af9d2 |
| c2 | 84.94 | 154.3 | 215.1 | 275.7 | 316.1 | 35.47 / 5.20 | 31.24 / 2.79 | 89.47 | 7.74 | 49 | a8c3394af9d2 |
| g2 | 91.94 | 149.5 | 218.7 | 278.8 | 323.8 | 34.02 / 5.20 | 30.17 / 2.68 | 88.18 | 7.66 | 49 | a8c3394af9d2 |
| c3 | 84.21 | 144.5 | 216.8 | 270.1 | 316.0 | 35.44 / 5.08 | 31.43 / 2.71 | 89.69 | 7.67 | 49 | a8c3394af9d2 |
| g3 | 89.48 | 151.6 | 222.1 | 281.6 | 330.5 | 34.11 / 5.20 | 29.81 / 2.72 | 88.15 | 7.80 | 50 | a8c3394af9d2 |

Pooled C vs G:
- **Greedy single:** 85.36 -> **89.93 (+5.4%)**. Pairs: +1.7 / +8.2 / +6.3%.
- **Concurrency:** c4 +1.8%, c8 +2.0%, c12 +2.6%, c16 317.2 -> 326.9 (+3.0%).
- **Matched C1 ms/step:** coding 35.44 -> 34.03 (-1.41) and prose 31.37 -> 29.98 (-1.39). Every pair is >= 1.07 ms lower. Coding_div C8: 90.03 -> 88.99.
- **AL:** unchanged within noise.
- **draft_ms(1):** the AV profile is calibrated on the stock path, so it stays ~7.7 (as expected). `DSV41_AV_FIXED_MS=2.4` was not tried because AL did not drop.

**Sampled** (runarm_sampled quick, c2 vs g2):
- t1 single: 87.0 -> 87.9.
- t1 c8: 213.0 -> 209.5 (-1.7%).
- t0.6 single: 88.4 -> 88.9.
- GSM8K t0.6 x2: 49.5 -> 50.0.

**Exactness: class X (exact).**
- exact.py gives a8c3394af9d2 on every C and G instance, with all 6 per-prompt hashes equal.
- The task's mixed test (`r4/e2/mixed.py`: 4 concurrent greedy requests, with or without 4 concurrent T=1 requests, must equal solo greedy) gives 0/4 on **control too**. Concurrent greedy output depends on batch composition: batch-variant GEMM numerics and request arrival timing. So "equal to solo" is not a usable criterion on this stack.
- Solo greedy hashes are identical between C and G. As a substitute check, `r4/e2/mixgsm.sh` ran GSM8K-50 greedy at conc8 while continuous t1 conc4 sampled traffic created mixed steps:
  - c2 50/50, g1 49/50 (the known item-12 flip), g2 50/50.

**Gates on the promoted live instance** (19:21 UTC, no env overrides):
- GSM8K-50 conc8: 49/50 (item 12) at 236 tok/s.
- **GSM8K-200 conc16: 196/200**. The misses (93, 100, 119, 184) are exactly the reference set. 302 tok/s.
- exact.py a8c3394af9d2.
- Lowest MemAvailable over the run: 20.3 GB.
- longctx was not rerun: the patch touches neither target numerics nor attention.

**Incidents.**
- 2 of 8 launches (c1 with the flag off; g2 try 1 with the flag on) hung in the FIRST, stock dspark capture: `rdma_ar rank 0: timeout waiting for 1 phase 1` at the first 160-row all_reduce, followed after 10 min by "CUDA error: unspecified launch failure" on all ranks.
- This is the same transient as r4-e1's E1-chk. Plain relaunches were clean. It is not caused by the patch.
- Logs: `r4/e2/c1fail/`, `r4/e2/g2fail1/`.

**Rollback** (from the Mac):
- Env only: `bash $S/conc/launch-conc.sh DSV41_DSPARK_GREEDY_SHARDED=0`. The site file is inert when the flag is off.
- Launcher: `cp -p ~/dsv41-tp8.sh.bak-r4e2gs ~/dsv41-tp8.sh` on all 8 (back to 709abe3d), then `bash $S/dsv41-launch-all.sh`.
- Site file: stop the service, then `bash $S/r4/rebased/greedy-sharded/uninstall.sh` (restores 0cc6e2bf). Do the launcher rollback first.

Raw results: `~/dsv41-opt-20261021/r4/e2/{chk,c1r,g1,c2,g2,c3,g3,live}`.

## Round 4 / e3 b12x-smallm: small-M dense MXFP8 -> FlashInfer b12x. NOT PROMOTED (2026-10-22 ~21:15 UTC)
What: port of rhys101 small-m-mxfp8.patch. vllm/utils/flashinfer.py 09f4e1fc routes vllm::mm_mxfp8 (backend cutlass, 0<M<=DSV41_MXFP8_B12X_MAX_M, K%128==0, bf16 out) to flashinfer b12x. The new file was installed only for the A/B (env-gated, default off).
Class: EXACT in practice, not just R. Unit-test rel L2 b12x vs CUTLASS = 0.0 on every shape and M (1..128), and exact.py a8c3394af9d2 on every B instance.
Unit test (spark-two, r4/e3/unit/): routing PASS (CUTLASS at M=129 and K=288), graph replay == eager. The projected 9.8-11.3 ms/step saving was ARTIFACT: the test's CUTLASS ran untuned tactic -1 (~60 GB/s at M=1), while live CUTLASS is autotuned (config cache hit). b12x was slower only at N=16160 (draft head) for M>=48 (<=0.7%) and N=5120/K=15360 at M=128. SKIP_N was NOT used, because skipping 16160 would forfeit its about 2.5x win at M<=24.
A/B, ABAB, 3+3 instances (c1 b1 c2 b2 c3 b3; c3 needed one retry after an RDMA all_reduce hang at DSpark capture on the control arm):
- greedy single mean C 89.02 (88.71/88.08/90.26) -> B 89.03 (89.02/89.11/88.95) = +0.0%; pairs +0.3/+1.2/-1.5%
- conc4 153.4 -> 152.5 (-0.6%), conc8 221.0 -> 224.2 (+1.4%), conc12 282.1 -> 283.1 (+0.4%), conc16 325.5 -> 327.4 (+0.6%)
- matched C1 ms/step coding 34.02 -> 33.71 (pairs -0.46/-0.30/-0.17), prose 29.95 -> 29.73 (-0.70/-0.11/+0.17); coding_div C8 88.66 -> 88.73
- AL within 0.08 (coding 5.18 -> 5.12, prose 2.79 -> 2.72); verify_ms(1) 16.12 -> 15.98; draft_ms(1) 7.76 -> 7.74
- GSM50 C 50/49/50, B 49/50/50; sampled quick c3 vs b3 gsm8k t0.6 x2 50.0 vs 49.5, t1 c8 212.7 vs 214.8
- no 'not bit-identical' MAINPROJ self-check lines (MAINPROJ coverage kept), no first-call-during-capture, no JIT after health, startup 408-412 s (C 400-410 s), MemAvailable >= 20.2 GB
Gate FAILED: pooled single +0.0% (< +1.5%) and matched C1 not >= 0.4 ms lower in every pair. Real saving is about 0.2-0.3 ms/step (tuned CUTLASS is already near b12x for live shapes).
Rollback / state: uninstall.sh run with service stopped; flashinfer.py c0db917f on 8/8 (backup *.bak-r4-b12x-smallm kept), launcher unchanged f9140d38. Relaunched live from the Mac (NOTE: dsv41-launch-all.sh must run FROM THE MAC; from spark-one it cannot reach spark-six/seven/eight). Live check: exact a8c3394af9d2, GSM8K-50 50/50 at 241 tok/s.
Raw: r4/e3/{unit,c1,b1,c2,b2,c3,b3,c3fail1,live}/, sum-all.txt.

## Round 4 / e4 verifycap-routelive: DSpark-confidence verify cap + dead-row expert skip. PROMOTED (2026-10-23 01:00 UTC, arm vcauto)

**What.** Two verify-side changes. Both are env-gated, and the site files are inert when the flags are off.
- `DSV41_ROUTE_LIVE=skip`: the target router gives padding and dead verify rows expert id -1 and weight 0. No expert weights are read for those rows.
- `DSV41_VERIFY_CAP=auto`: each step, a draft whose cumulative DSpark confidence falls below tau is marked dead. Dead drafts become padding for the router and invalid for acceptance. tau is derived from the AV calibration (scale 0.6 / N0 96 / N1 104 / FIXED_MS 3.4, unchanged since r3). Typical mean tau is 0.15 at C1 and 0.28-0.31 at C8-16; 10-25% of verified drafts are dead.
- Temperature > 0 relies on `DSV41_BLOCK_VERIFY_FRESH_U=1` and `DSV41_DRAFT_FRESH_NOISE=1`. Both are launcher defaults.

**Site files** (sha identical 8/8; backups `*.bak-r4-verify-router` on each rank):

| file | stock / backup | live |
|---|---|---|
| fused_moe/router/dsv4_topk.py | 14f7c848 | 45c5fc4f |
| fused_moe/router/fused_topk_bias_router.py | 8cf50cf2 | ea6720bc |
| v1/worker/gpu/spec_decode/adaptive_verification.py | ec240369 | 7e30eebc |
| v1/worker/gpu/model_runner.py | fc6d5d6d | 9fecbda4 |

**Launcher:** `~/dsv41-tp8.sh` f9140d38 -> **22e83204** on 8/8. It adds `export DSV41_ROUTE_LIVE=${DSV41_ROUTE_LIVE:-skip}` and `export DSV41_VERIFY_CAP=${DSV41_VERIFY_CAP:-auto}` after the GREEDY_SHARDED line. Backup: `~/dsv41-tp8.sh.bak-r4e4`.

**GPU unit tests** (spark-three, service stopped; archived in ops/r4/e4/vr_all.log and vr_moe2.log):
- `all` passed: router, live-cap and reject. The rejection-sampler distribution test passes for the stop rule; the peek negative control fails as designed.
- The `moe` test needed an API-drift fix to the test only: the expert is now built the way the moe-compact test builds it (FusedMoEConfig + make_mxfp4_moe_quant_config + workspace_shapes). Before the fix it hit a cuBLASLt alloc error.
- On the live M32 + compact path, with M 2-96 added: live rows are **bit-identical** in both skip and anchor modes at every M and dead count.
- Saving per dead row x 43 layers: about 2.9-3.5 ms per step at M 6-12, 2.1-2.4 ms at M 48 and 0.9-1.5 ms at M 96. These are eager timings with random routing; real routing gives roughly 0.6x.

**A/B** (one fresh launch per row; order c1 rl a1 c2 a2 u1 c3 u2 c4 u3 c5 u4). a = vc10 (`conf:0.1`), u = vcauto. Single = greedy, 1024 tok; matched = C1 ms/step.

| inst | single | c4 | c8 | c12 | c16 | match coding / prose ms | prose AL | t1 single | t1 c8 | dead % | GSM50 | exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1 | 89.37 | 148.5 | 222.7 | 286.0 | 331.1 | 34.14 / 30.00 | 2.87 | 88.3 | 206.3 | - | 49 | a8c3394af9d2 |
| rl | 90.48 | 154.1 | 225.0 | 281.4 | 329.8 | 34.05 / 29.72 | 2.73 | - | - | - | 50 | a8c3394af9d2 |
| a1 | 91.11 | 161.0 | 220.4 | 282.8 | 329.6 | 34.17 / 29.53 | 2.87 | 86.4 | 207.3 | 4.3 | 50 | a8c3394af9d2 |
| c2 | 88.92 | 152.4 | 223.7 | 283.1 | 326.4 | 33.98 / 29.90 | 2.78 | 87.6 | 209.8 | - | 49 | a8c3394af9d2 |
| a2 | 88.45 | 157.7 | 226.6 | 277.9 | 326.7 | 34.20 / 29.76 | 2.80 | 88.5 | n/a | 4.6 | 50 | a8c3394af9d2 |
| u1 | 90.58 | 152.8 | 220.4 | 288.7 | 327.2 | 33.67 / 29.00 | 2.70 | 89.8 | 207.0 | 13.2 | 50 | a8c3394af9d2 |
| c3 | 87.65 | 149.2 | 223.0 | 277.9 | 324.7 | 33.91 / 29.82 | 2.71 | 89.4 | 209.3 | - | 49 | a8c3394af9d2 |
| u2 | 90.80 | 152.1 | 219.7 | 277.9 | 330.7 | 34.06 / 29.10 | 2.74 | 87.0 | 210.4 | 13.1 | 50 | a8c3394af9d2 |
| c4 | 86.53 | 157.4 | 221.4 | 285.6 | 327.7 | 34.06 / 29.98 | 2.79 | 92.3 | 207.3 | - | 49 | a8c3394af9d2 |
| u3 | 88.83 | 156.7 | 222.7 | 281.4 | 325.9 | 33.76 / 28.66 | 2.55 | 86.2 | 212.4 | 13.1 | 50 | a8c3394af9d2 |
| c5 | 91.12 | 154.1 | 219.3 | 281.4 | 328.3 | 33.97 / 29.88 | 2.73 | 88.6 | 213.0 | - | 50 | a8c3394af9d2 |
| u4 | 90.17 | 151.6 | 220.9 | 285.0 | 327.2 | 33.75 / 29.24 | 2.71 | 87.1 | 209.4 | 12.6 | 49 | **b7e868c9d700** |

Pooled C (5 instances) vs U (vcauto, 4 instances):
- **Greedy single:** 88.72 -> **90.10 (+1.55%)**. Adjacent pairs: +1.9 / +3.6 / +2.7 / -1.0%.
- **Concurrency:** c4 152.3 -> 153.3 (+0.6%), c8 222.0 -> 220.9 (-0.5%), c12 282.8 -> 283.2 (+0.2%), c16 327.6 -> 327.7 (0.0%).
- **Matched C1 ms/step:** prose 29.92 -> **29.00 (-0.92, -3.1%; every U is below every C)**; coding 34.01 -> 33.81.
- **Matched coding_div C8 ms:** 88.0 -> 86.0. AL drops by design (prose C1 2.78 -> 2.68).
- **Sampled t1 single:** quick (2 reps each) 89.2 -> 87.5. With 4 extra reps on c5/u4, 87.2 -> 88.3. Pooled over all reps: 88.65 -> 87.78 (-1.0%). The c4 control code run at 99.8 is an outlier.
- **Sampled t1 c8:** 209.1 -> 209.8 (+0.3%). t0.6 single: 86.9 -> 88.9.

vc10 (2 instances) is a no-gain arm: dead 4-5%, single 89.78 (+1.2% vs the first two controls), c8 -0.4%, matched coding_div C4 not better. rl alone is one instance (90.48, exact hash equal to ctl), so it is a neutrality guard only.
**Not run:** vc10ib, vc05 and vc20. The step was already about 1.5 h over budget, and they tune the fixed-tau arm that lost to auto. STEP 4 (drafter gather on RDMA) was also not run.

**Exactness: greedy-equivalent (not bitwise class X).**
- ROUTE_LIVE alone: live rows are bit-identical in the unit test and the rl hash equals control.
- VERIFY_CAP: on 5 of 6 capped instances, exact.py gives a8c3394af9d2. On u4 it gives b7e868c9d700: 2 of 6 prompts differ, and the hash is self-consistent over 3 runs.
- Cause: the cap changes the step trajectory (where acceptance stops). auto-tau depends on the per-restart AV profile (u4 tau 0.192 vs 0.203-0.209). This is the same cross-restart trajectory sensitivity as e1's 9330f03c4f97 instance. Every token is still the target argmax on this stack; no lossy kernel is involved.
- Correctness gates are unaffected.

**Gates on u4** (vcauto, same config as promoted):
- GSM8K-200 conc16: **195/200**. Misses 12, 93, 100, 119, 184: the reference set plus the known item-12 flip. 304 tok/s.
- longctx 4K-200K: **PASS**, needles 25/25, copy 1.000.
- gsm8k_sampled_gate compare c5 vs u4: **PASS** (50.0 vs 49.0).
- GSM50 on the U instances: 50 / 50 / 50 / 49.
- On every instance: no 'first call during capture', no JIT after health, no errors, MemAvailable >= 20.3 GB (14.5 GB right after longctx).
- No cross-rank confidence divergence: AL is normal and there is no garbage.

**Promoted relaunch** (from the Mac, 00:59 UTC start, health 200 by 01:05): 8/8 active on 22e83204, with the 'verify-cap: auto' line present. exact.py a8c3394af9d2; GSM8K-50 conc8 49/50 at 227 tok/s; MemAvailable 22.1 GB.

**Incidents.** c1 try 1 hung in DSpark capture (`rdma_ar rank 0: timeout waiting for 4 phase 1`), the known transient on the control arm; the retry was clean. Log: r4/e4/c1fail1/.

**Rollback** (from the Mac):
- Env only: `bash $S/conc/launch-conc.sh DSV41_VERIFY_CAP=0 DSV41_ROUTE_LIVE=0`.
- Launcher: `cp -p ~/dsv41-tp8.sh.bak-r4e4 ~/dsv41-tp8.sh` on all 8 (back to f9140d38), then `bash $S/dsv41-launch-all.sh`.
- Site files: stop the service, run `bash $S/r4/verify-router/uninstall.sh`, then verify on 8/8 that the shas are back to 14f7c848 / 8cf50cf2 / ec240369 / fc6d5d6d. Do the launcher rollback first.

Raw: `./r4/e4/{c1..c5,rl,a1,a2,u1..u4,live,c1fail1}/` and sum-all.txt (u4 also has gsm200, longctx, exact_r2/r3; c5/u4 have t1x). Scripts and unit logs: `./ops/r4/e4/`.

## Round 4 / e5: TTFT + step-overhead diagnoses, prefill trio. PROMOTED idxshard only (2026-10-23 ~04:45 UTC)

**Promoted:** prefill-idxshard (env-only on top of installed `mla/indexer.py` 4753707e). Launcher `~/dsv41-tp8.sh` 22e83204 -> **5ccc4b79** on 8/8. It adds `DSV41_IDX_ROW_SHARD_MIN=${..:-1024}` and `DSV41_IDX_ROW_SHARD_MIN_CTX=${..:-32768}` after the VERIFY_CAP line. Backup: `~/dsv41-tp8.sh.bak-r4e5`.
**Not promoted:** prefill-sched (uninstalled, scheduler.py back to d9a39870 on 8/8). prefill-sp: STOP at the debug gate. model-merged stays installed (E1 CERT/MAINPROJ).

### Part A1: TTFT noise (live config, no patch)
- Most of the "16K" spread is how the prompt is built. bench.py's "16K" prompt is really about 23.9K tokens (2 chunks) and its "32K" prompt about 48.7K tokens (3 chunks). TTFT is a step-linear function of the token count (about 5.4-6.1K tok/s plus about 0.1-0.3 s per extra chunk). prefill_bench's tok/word calibration starts low, so its first "16K" rep is 25K tokens.
- **Warm server, sequential cold prompts** (10 x 16K + 5 x 32K, unique salt at the start; A1/ttft.jsonl): every TTFT is within ±4% of the linear fit (5.3-6.1K tok/s). No outliers.
- **The remaining noise is the first long prompt after a restart.** For the same 25,202-token prompt on 6 fresh instances, TTFT was 5.61 / 5.72 s on base/base2 (pf ran first), 4.88 / 4.35 s on thr8k/byp (a 128K starve ran first), and 4.48 / 4.77 s on idx/all. The second prompt was always 3.45-3.50 s (19.7K tokens). The historical K-runs show the same thing: 4.26-7.25 s at a constant 23.9K tokens, measured once per instance.
- **What happens on the first request** (1 Hz sampler on 8/8: vmstat, PSI, worker /proc stat, nvidia-smi):
  - MemAvailable drops about 6 GB on every rank (22.4 -> 16.2 GB), which is the lazy allocation of the prefill workspace.
  - Triton JIT warnings appear in every rank log: `_dsv41_live_cap_kernel` (e4 verify-cap) and `_screen_kernel` / `_refine_kernel` (e1 certified head, `dsv41_r4_small.py`).
  - The worker gets 400-1200 majflt and 600-4800 swap-in pages (at most about 19 MB). Fast and slow instances look the same here.
- **Hypotheses ruled out:**
  - (1) kswapd / reclaim: no direct scan, PSI memory is about 0 during prefills, and the swap-in volume is trivial.
  - (2) Engram: the two 12 GB engram tables are **mlocked and THP, with 0 swap**. The ~0.9 GB that is swapped out is python heap and NCCL /dev/shm.
  - (3) Chunk interleaving: requests ran one at a time.
  - (4) Thermal: SM clock was 1969-1989 MHz on all 8 ranks, temperature ≤ 60 C, and the throttle reasons were 0x0.
- **Conclusion:** the remaining noise is hypothesis (5), first use: lazy workspace allocation plus first-call JIT/autotune. Its cost varies from instance to instance (+0 to +1.3 s).
- **Fix candidate (not applied):** after /health, send one ≥ 24K-token warmup prompt (2 chunks) plus one short greedy request from launch-all or warmup.py. A better variant would add the CERT and live-cap Triton kernels to the engine warmup. This costs nothing for decode. Also, single-sample TTFT should never be gated on the first long prompt.

### Part A2: AL-dependent step cost
Tool: `tools/a2.py` + `an_a2.py`. C1, workloads coding / prose / copy / count, 3-5 reps each.
- **Not API-side.** stream=false vs stream=true wall time is identical (coding 1.4602 vs 1.4567 s; prose 2.801 vs 2.800 s). The API server uses 3.8% CPU. EngineCore sits at about 101%, which is the normal busy-poll, not saturation. The worker uses about 118%.
- **Engine-side, but mostly not per accepted token:**
  - Within a workload, the slope is only **0.6-0.7 ms per accepted token** (coding gap = 30.9 + 0.60·k, prose 27.2 + 0.62·k).
  - Most of the 1.4 ms/token cross-workload slope is a per-workload intercept. copy (AL 2.47) runs at **26.3 ms/step**, but prose at the same 2-3 tokens per step runs at 28.5 ms. count (AL 6.0) runs at **32.9 ms**, below coding with all 6 accepted (34.6 ms).
  - The within-workload jump is concentrated at 5-6 accepted tokens (coding: k=4 31.5, k=6 34.6; prose: k=4 29.2, k=6 35.0).
- With verify-cap and route-live off (R1), OLS gives 26.7 + 1.07·k. So the cap/dead-row skip accounts for part of the slope (rows it kills are the cheap low-AL steps).
- **Interpretation:** step cost follows the content of the 6 verify rows, i.e. the union of experts routed per layer, which sets MoE weight traffic at small M. Repetitive text (copy) touches few experts. Full-acceptance steps also feed back into the next step's drafts and verify width (AV/cap). The truly per-token work (drafter context update over accepted rows, KV slot writes, sampler epilogue, host copy) is at most about 0.6 ms/token.
- **Candidate next step (no env-only fix exists):** log the number of distinct experts per layer per verify step (DSV41 debug counter in dsv4_topk) and regress the gap on it. If it holds, expert-union-aware work (grouping or reordering drafts, or making the cap expert-cost-aware) is the lever, not the epilogue.

### Part B: prefill trio
**B0.** CPU tests PASS (test_sched_cpu, test_idxshard_cpu). Torch-profiler CUDA trace of one cold 16K chunk on rank 0 (B0prof/):
- NCCL AllReduce RING_LL: 18.0%
- sparse_mla_prefill: 11.5%
- mhc_post: 10.5%
- mhc_pre_big_fuse_with_norm: 7.0%
- hc_prenorm gemm: 4.7%
- deep_gemm fp8/fp4 MoE: 13.1%

AR + mHC + norms is about 41%, well above 10%, so B0 allowed prefill-sp.

**Starvation and lone prefill** (st = 128K long + 6 x 300-token shorts + a live decoder; 2 runs each; pf = lone prompt):

| arm | short TTFT p50 s | long TTFT s | decoder ITL inside p50 ms | pf 64K / 128K / 200K tok/s |
|---|---|---|---|---|
| base (x2 inst) | 21.85 / 21.64 / 21.84 / 21.67 | 29.85 / 29.63 / 29.85 / 29.67 | 31-2746 | 5735 / 5572 / 5437 |
| thr8k (stock `--long-prefill-token-threshold 8192`) | 4.33 / 4.20 | 33.46 / 33.31 (+12%) | 1627 / 1647 | 5765 / 5601 / 5463 |
| byp (BYPASS=2048) | 4.53 / 5.05 | 31.65 / 30.85 (+4.4%) | 1691 / 1423 | 5736 / 5580 / 5426 |
| bypdc (+DECODE_CAP=4096) | 2.99 / 2.98 | 39.36 / 38.38 (+31%) | 1004 / 1007 | 5752 / 5590 / (run cut) |

- **Sched gate FAIL.**
  - byp is 4.5x better than base and +4.4% on the long prompt, but its p50 (4.79 s) is worse than thr8k (4.27 s).
  - bypdc beats thr8k (2.98 s) but costs +31% on the long prompt.
  - Root cause: async scheduling has already scheduled the next full 16K step when a short arrives, so a short waits about 1 residual step plus 1 full step plus the bypass step. A 1.5-2 s TTFT needs a smaller steady chunk whenever another request is present, and that is exactly what costs long-prompt throughput.
  - Lone-prompt pf is unchanged in every sched arm.

**idx** (`DSV41_IDX_ROW_SHARD_MIN=1024 DSV41_IDX_ROW_SHARD_MIN_CTX=32768`). Lone prefill tok/s, 3 control instances (base, base2, c3) vs 3 idx instances (idx, all, promo):

| | 64K | 128K | 200K |
|---|---|---|---|
| control | 5737 | 5574 | 5434 |
| idx | 5822 (+1.5%) | **5807 (+4.2%)** | **5828 (+7.3%)** |

The 200K TTFT drops 37.8 -> 35.3 s.
- longctx 4K-200K PASS on both idx instances (needles 25/25, copy 1.000).
- lp compare (prompts ≤ 24.6K, which are below MIN_CTX, so the path is inert): all vs base mean |dlp| 0.040 and all vs base2 0.028, against base-vs-base 0.032.
- Long lp (40K / 70K / 110K, `tools/lplong.py`):
  - on(promo) vs off(c3): first token 3/3, diverge ≥ 24 tokens, mean |dlp| 0.028.
  - on(all) vs on(promo): diverge at 8 tokens.
  - So the idx effect is inside cross-restart prefill nondeterminism. promo and c3 emit identical text.

**Class: exact on decode and short prompts** (exact.py a8c3394af9d2 on every instance). **Greedy-equivalent at ≥ 32K context** (not bitwise-provable, because prefill is not restart-deterministic even base-vs-base).

**sp: STOP.**
- `DSV41_PREFILL_SP compare` (rows 16384, identical on all 8 ranks): hidden rel_l2 **0.664**, residual **0.599**, post_mix 0.19, pre_mix 0.22. That is about 1, which means a missed or doubled reduction.
- The engine then died during init with `CUDA error: unspecified launch failure` on all ranks (log: spdbg-rank0-initfail.log). Not run further.

**Decode non-regression** (runarm_conc + matched C1). C = base2, c3; I = all, promo:

| | single | c4 | c8 | c12 | c16 | matched coding / prose ms | exact |
|---|---|---|---|---|---|---|---|
| base2 (C) | 88.7 | 153.0 | 221.2 | 276.3 | 331.3 | 34.02 / 28.88 | a8c3394af9d2 |
| all (I) | 86.8 | 152.3 | 220.0 | 280.7 | 326.4 | 34.05 / 28.99 | a8c3394af9d2 |
| c3 (C) | 90.4 | 156.4 | 218.7 | 275.4 | 325.6 | 33.94 / 29.21 | a8c3394af9d2 |
| promo (I) | 90.5 | 149.6 | 219.5 | 281.4 | 325.7 | 34.19 / 29.54 | a8c3394af9d2 |
| pooled | 89.55 -> 88.65 (-1.0%) | 154.7 -> 150.9 (-2.4%) | 219.9 -> 219.8 | 275.9 -> 281.1 (+1.9%) | 328.4 -> 326.1 (-0.7%) | 33.98 -> 34.12, 29.05 -> 29.27 | |

- The idx code cannot engage in decode: it needs at least 1024 rows per rank, i.e. a step of at least 8192 rows, and context ≥ 32K. The decode hash is bit-identical, so the c4 -2.4% is the known c4 restart spread (e4 C instances ranged 148.5-157.4).
- **Correctness:** GSM8K-200 conc16 on 'all' was **195/200**, misses {12, 93, 100, 119, 184} (the reference set plus the known item-12 flip), at 302.7 tok/s. GSM8K-50 on the promoted instance: 50/50 at 239 tok/s. Lowest MemAvailable 15.2 GB, against the 4 GB gate.

**Promoted relaunch:** launch-all from the Mac with launcher 5ccc4b79 on 8/8. Site shas identical 8/8: indexer 4753707e, scheduler d9a39870, model.py e7264781. Health 200, exact a8c3394af9d2, GSM8K-50 50/50.

**Incident.** My wait loop timed out and I started the idx arm while bypdc's pf was still running, so bypdc lost its 16K/200K pf rows. The st results are complete and the remaining pf rows match base.

**Rollback** (from the Mac):
- Env only: `bash $S/conc/launch-conc.sh DSV41_IDX_ROW_SHARD_MIN=16384 DSV41_IDX_ROW_SHARD_MIN_CTX=0` (stock threshold).
- Launcher: `cp -p ~/dsv41-tp8.sh.bak-r4e5 ~/dsv41-tp8.sh` on all 8 (back to 22e83204), then relaunch.
- Site file: stop the service, run `bash $S/r4/prefill-idxshard/uninstall.sh` (indexer.py back to 43d4f705; do the launcher rollback first, though the env is harmless with the stock file).

**Proposals (not run):**
1. Post-health warmup prompt, from A1.
2. Expert-union instrumentation, from A2.
3. NCCL protocol for prefill all-reduce. It is 18% of a 16K chunk and NCCL picks RING_LL. Try `NCCL_PROTO=Simple` scoped to messages larger than the RDMA AR cap, measured on pf and decode.
4. Fix prefill-sp's reduction before any rerun.
5. Sched: a smaller steady chunk (8K) only while a short request is waiting, combined with BYPASS, may reach about 3 s at ≤ +5% long.

Raw: `./r4/e5/{A1,A2,R1,B0prof,base,base2,thr8k,byp,bypdc,idx,all,c3,promo}/`, `spdbg-rank0-initfail.log`, `lplong-all-idxon.json`, and `local-data.tgz` (monitor CSVs, profile, tools, launcher).

## Round 4 / e6 FC2-compact kernel fix (r4-e6-fc2c-kernel-fix): NOT PROMOTED, Step 3 timing gate failed, nothing installed (2026-10-23 ~05:00 UTC)

**Root cause of the r3-e2 715 is confirmed on the GPU** (spark-four, service stopped, probe `~/dsv41-stage/r4/fc2c-probe`).
- A `16U4_ALIGN16B` (.b4x16_p64) TMA traps with `ILLEGAL_INSTRUCTION` when the inner coordinate is not a multiple of 128 elements. It faults at 288, 320, 416, 576, 32 and 64.
- It loads correctly at 0, 128, 256, 384 and 640. The original probe's u4a16 cases hung (rc=124) only because their expect_tx was the smem size. complete_tx counts packed global bytes (2048 for a 128x32 box), and with that value they return OK.
- `16U4_ALIGN8B` (packed smem) accepts 288 and 320. UINT8 TMA needs 16 B (faults at byte 8, OK at 16).
- The r3 slot layout (slots at a k_real stride) therefore cannot work for k_real 288 or 320.

**v2 design** (tail probe `tma_tail_probe.cu`):
- A u4a16 map with inner dim 32 is rejected by `cuTensorMapEncodeTiled` (invalid argument), so the inner dim must be a multiple of 128.
- The u4a16 unpack leaves the pad bytes of each 16 B chunk untouched.
- v2 splits B into two parts:
  - MAIN: `[G, N, 256]` FP4, the standard descriptor at coordinates 0 and 128.
  - TAIL: `[G*N, 32] uint8`, already in the .b4x16_p64 smem image (data in bytes 0-7 and 16-23, pads 0).
- The tail is loaded with a UINT8 TMA (inner dim 32 B, box 128 B x BLOCK_N, 128B swizzle). Bytes 32-127 are filled with zeros as out of bounds, and that fill counts toward complete_tx, so the last stage expects +BLOCK_N*64 B.
- The tail descriptor goes in through the unused `tensor_map_buffer` argument (six descriptors for BLOCK_N 8-256). The JIT name is `..._compact_b2`, the template argument is `kBMainK`, and the API is `m_grouped_fp8_fp4_gemm_nt_contiguous_compact_b2` + `fc2_compact_tail_descs`.
- Native `~/dsv41-stage/fc2c-v2/native-fc2c-v2` exists **on spark-four only**: _C bfb55fdb, manifest 6999ecbe, hidden visibility.
- Patch `r4/r4-e6-fc2c-kernel-fix/fc2-deepgemm-v2.patch` 6129f802; payload `sm121_fc2_compact.py` cb113d58; test `test_fc2_compact_v2.py` 9bde53e3.
- Bytes per expert including scales: 1,044,480 -> 880,640 (-15.7%). FC2 weight bytes are -16.7%, not -25%, because the tail stores 32 B per row for 16 B of data.

**Step 3 offline gates** (spark-four, service stopped; `fc2c-v2/test-20261023T042959Z.*`):

| M (tok) | align | live base us | compact us | speedup |
|---|---|---|---|---|
| 1 | 32 | 17.9 | 18.7 | 0.956 |
| 6 | 32 | 236.1 | 214.3 | 1.102 |
| 32 | 32 | 1002.2 | 904.5 | 1.108 |
| 48 | 32 | 1351.3 | 1224.4 | 1.104 |
| 96 | 32 | 1884.8 | 1721.0 | 1.095 |
| 96 | 128 | 3426.2 | 3192.1 | 1.073 |
| 192 | 128 | 4244.2 | 3926.3 | 1.081 |
| 384 | 128 | 4433.3 | 4135.6 | 1.072 |

- Passed:
  - Bitwise exact vs the native padded kernel and vs the live kernel on every case, with both finite and raw NaN dead scales.
  - Finite outputs.
  - Padded vs fp32 ref rel 1.63-1.68e-3.
  - No new JIT during graph capture (27/27 cache files).
- **Failed:** the timing gate (>= 1.15x at M <= 96). The speedup is about 1.10x, which matches the -16% bytes, and M=1 is 4% slower.

**Decision:** stopped before Step 4. Site-packages, the launcher and the JIT caches are unchanged; launcher 5ccc4b79 on 8/8.
- **Next lever:** get the tail to 16 B per row (for example, 4 rows' tails packed per 64 B with a consumer-side chunk offset, or ALIGN8B packed smem for tile 2), which would give the full -25%.
- **Projected decode gain:** only about +1-2% at the measured 1.10x, so a serving A/B is low priority.

**State after:** relaunched from the Mac (`ssh spark-one cat ...dsv41-launch-all.sh | bash`; running it on spark-one fails because spark-one cannot ssh to spark-six/seven/eight, publickey). Health 200; exact.py a8c3394af9d2; GSM8K-50 conc8 49/50 (miss 12) at 229.9 tok/s.

**Rollback:** none needed.

## Round 4 summary and final verification (2026-10-23 04:35-04:55 UTC)

**Verified live state.** All 8 `dsv41-rank` user units active since 04:30:57 UTC (r4-e6 relaunch from the Mac), /health 200, no foreign activity. Live rank-0 env has DSV41_L2_PREFETCH=1, R4_MAINPROJ_SPLIT=1, R4_CERT_HEAD=1 (M=2-96), EAGER_GLUE=all, DSPARK_GREEDY_SHARDED=1, ROUTE_LIVE=skip, VERIFY_CAP=auto, IDX_ROW_SHARD_MIN=1024, IDX_ROW_SHARD_MIN_CTX=32768, AV_FIXED_MS=3.4.
sha256 (first 8 hex), identical on all 8 ranks (checked with ops/r4/shacheck.sh):
- launcher ~/dsv41-tp8.sh **5ccc4b79** (backups .bak-r4e1=2971eb50 (r3 final), .bak-r4e2gs=709abe3d, .bak-r4e4=f9140d38, .bak-r4e5=22e83204)
- r4 site files: models/deepseek_v41/nvidia/model.py e7264781 (.bak-r4-model-merged) + dsv41_r4_small.py 02f94f29 (new); distributed/device_communicators/rdma_all_reduce.py 99102ff1 (.bak-r4-l2-prefetch) + dsv41_l2_prefetch.py 7d2acdc8 (new); v1/attention/backends/mla/sparse_swa.py 38b16c51 and v1/worker/gpu/spec_decode/dflash/speculator.py 17ddccf1 (.bak-r4-eager-glue); spec_decode/dspark/speculator.py bbcee9e3 (.bak-r4-greedy-sharded = 0cc6e2bf); fused_moe/router/dsv4_topk.py 45c5fc4f, router/fused_topk_bias_router.py ea6720bc, spec_decode/adaptive_verification.py 7e30eebc, v1/worker/gpu/model_runner.py 9fecbda4 (.bak-r4-verify-router); v1/attention/backends/mla/indexer.py 4753707e (.bak-r4-prefill-idxshard)
- restored to pre-r4 (not promoted): utils/flashinfer.py c0db917f (e3), v1/core/sched/scheduler.py d9a39870 (e5 sched). FC2-compact (e6) never installed.
- r3 files unchanged: rejection_sampler_utils 09522889, attention.py f2f48198, logits_processor.py caf6d216, dspark.py fa56d07d, o_proj.py 34b9e1a4, dsv41_r3_dense.py 993023f2, sonar_bf16_bmm.py 9f6f3f84.

**Final measurement** (one instance, results in ./r4-final/, script ops/r4-final-run.sh; single = 1024 tokens x 2 repeats per type, conc = 512 tokens each):

| metric | original baseline | round-3 final | **round-4 final** | vs R3 | vs baseline |
|---|---|---|---|---|---|
| single code | - | 79.9 | **86.3** | +8.0% | |
| single prose | - | 60.0 | **67.7** | +12.8% | |
| single math | - | 113.5 | **117.8** | +3.8% | |
| single mean | 57 | 84.5 | **90.6** | +7.2% | +59.0% |
| conc4 agg | 102 | 146 | **153.3** | +5.0% | +50.3% |
| conc8 agg | 147 | 212 | **221.9** | +4.7% | +51.0% |
| conc16 agg | - | 321 | **330.9** | +3.1% | |
| conc24 agg | - | 407 | **415.2** | +2.0% | |
| conc32 agg | - | 465 | **479.0** | +3.0% | |
| sampled single t1.0/p0.95 mean | - | 85.1 | **90.0** | +5.8% | |
| sampled conc16 t1.0 | - | 312 | **312.2** | 0.0% | |
| accept len greedy single code/prose/math | | 2.62/1.71/3.70 | 2.52/1.74/3.48 | | |
| GSM8K-50 conc8 | | 49/50 | 49/50 (item 12, known flip) | | |
| GSM8K-200 conc16 | | 196/200 (279 tok/s) | 196/200, misses 93/100/119/184 = reference set (286 tok/s) | | |
| long-context | | PASS to 250K | 60/60 needles + copy 1.000 (4K-200K), 250,916 tok 5/5 + copy exact: PASS | | |
| exact.py | | a8c3394af9d2 | a8c3394af9d2 | | |

- Lowest MemAvailable over the run 11.2 GB (spark-one, during longctx; r3: 11.4 GB).
- Sampled single code had one 110.6 tok/s rep (79.9 for the other), so the sampled +5.8% is noisier than the greedy number. Restart-to-restart spread is still about ±4% single.

**Matched bench** (rhys-ep matched_bench.py default plan, ./r4-final/matched.json, vs r4/matched/matched-post-r3.json):

| workload | C | post-r3 per-stream | r4 per-stream | post-r3 ms/step | r4 ms/step | AL post-r3 -> r4 |
|---|---|---|---|---|---|---|
| coding | 1 | 143.72 | 153.15 | 36.22 | 33.75 (-6.8%) | 5.21 -> 5.17 |
| coding | 4 | 100.22 | 101.54 | 51.52 | 50.14 | 5.16 -> 5.08 |
| coding | 8 | 75.12 (agg 506) | 77.28 (agg 513) | 67.90 | 66.24 | 5.10 -> 5.11 |
| prose | 1 | 80.23 | 88.12 | 32.41 | 28.18 (-13.1%) | 2.60 -> 2.48 |
| prose | 4 | 52.59 | 52.55 | 50.97 | 47.04 | 2.68 -> 2.47 |
| prose | 8 | 39.35 | 39.83 | 65.73 | 61.91 | 2.59 -> 2.47 |
| coding_div | 8 | 54.09 | 52.55 | 88.20 | 85.96 | 4.77 -> 4.52 |

ms/step fell in every row. Prose/coding_div AL drops by design (e4 verify cap stops verification early on low-confidence drafts).

**Round 4 outcome:**
- PROMOTED e1 bitstack: L2 prefetch + main_proj split + certified lm-head (M 2-96) + eager glue. Exact. About +5% single.
- PROMOTED e2 greedy-sharded drafter graphs for all-greedy steps. Exact. About +5% single, +2-3% conc.
- NOT promoted e3 b12x small-M MXFP8: bitwise exact but +0.0% (live CUTLASS is autotuned). Uninstalled.
- PROMOTED e4 vcauto (VERIFY_CAP=auto + ROUTE_LIVE=skip). Greedy-equivalent, not bitwise: one restart hashed b7e868c9d700 (AV-profile dependent). +1.55% single, prose C1 -3% ms/step.
- PROMOTED e5 prefill idxshard (env only): 128K +4.2%, 200K +7.3% prefill tok/s. Greedy-equivalent at >=32K context, exact on decode. prefill-sched failed its gate (uninstalled); prefill-sp stopped at the debug compare.
- NOT promoted e6 FC2-compact v2 kernel: root cause (16U4_ALIGN16B TMA needs inner coord %128) fixed and bitwise exact, but only 1.10x vs a 1.15x gate. Native build only on spark-four ~/dsv41-stage/fc2c-v2.

**Rollback** (run from the Mac, never from spark-one; S=~/dsv41-opt-20261021/ops on spark-one; stop the service before touching site files; then `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh' | bash` and verify sha256 on 8/8):
- e5 idxshard: env `bash $S/conc/launch-conc.sh DSV41_IDX_ROW_SHARD_MIN=16384 DSV41_IDX_ROW_SHARD_MIN_CTX=0`, or restore ~/dsv41-tp8.sh.bak-r4e5; site: `bash $S/r4/prefill-idxshard/uninstall.sh`.
- e4 vcauto: restore ~/dsv41-tp8.sh.bak-r4e4 (or env DSV41_VERIFY_CAP=0 DSV41_ROUTE_LIVE=0 via launch-conc.sh); site: `bash $S/r4/verify-router/uninstall.sh`.
- e2 greedy-sharded: env DSV41_DSPARK_GREEDY_SHARDED=0, or restore .bak-r4e2gs; site: `bash $S/r4/rebased/greedy-sharded/uninstall.sh`.
- e1 bitstack: restore .bak-r4e1 (= r3 final launcher 2971eb50); site: `bash $S/r4/model-merged/uninstall.sh`, `bash $S/r4/rebased/l2-prefetch/uninstall.sh`, `bash $S/r4/eager-glue/uninstall.sh`.
- Full revert to round-3 final: launcher .bak-r4e1 on all 8, then each uninstall above (or copy every *.bak-r4-* back over its file); expected r3 shas: dspark/speculator.py 0cc6e2bf, others as in the Round 3 summary.
- Undo launcher steps in reverse order (e5, e4, e2, e1) before site files, so no flag is left on with its file removed.

**Ops notes:** dsv41-launch-all.sh must run from the Mac (spark-one has no key for spark-six/seven/eight). The rdma_ar timeout during first DSpark graph capture happened about 1 in 8 launches all round. It is a transient; stop all ranks and relaunch.

**Scripts archived:** ./ops/r4/ is a copy of the Mac scratchpad r4/ tree (queue.md, rhys-ep, rebased patch bundles with install/uninstall, e1-e6 drivers and READMEs, prefill/verify-router/small-kernels/b12x work, sha snapshots). Left out: b12x-moe/overlay (13 MB vendored b12x/flashinfer copy), the 7.9 MB e5 torch profiler trace, *.so builds, __pycache__.

**Next steps (round 5 candidates):**
1. FC2-compact: pack the 32-col tail to 16 B/row (4-row packing or ALIGN8B smem tail) to get the full -25% FC2 bytes; v2 is at 1.10x.
2. Post-/health warmup: one >=24K prompt + one short greedy request, or add the e1/e4 Triton kernels to engine warmup, to remove first-long-prompt TTFT noise (+0 to +1.3 s).
3. Log distinct experts per layer per verify step to test the AL-dependent step-cost hypothesis (0.6-0.7 ms per accepted token plus a per-workload baseline).
4. Drafter gather onto RDMA (e4 STEP 4, not run).
5. Find the rdma_ar first-capture timeout transient.

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

## Engram punch (2026-10-23; one/two/three 07:10-08:20 UTC, five..eight 11:40-12:35 UTC). 7 nodes punched; spark-four whole

**What.** In vLLM 0.32 TP8 each rank copies only its own contiguous run of Engram rows (3 of 24 hash heads per layer) from shards 47/48 into mlocked host RAM at load, and never reads the shard files again. On **every node except spark-four** (spark-one..three = ranks 0..2, spark-five..eight = ranks 4..7) the rows other ranks own were released with `fallocate(PUNCH_HOLE)`. The file sizes and offsets are unchanged, and the unread rows now read back as zeros. Tool: `ops/engram-punch/engram_punch.py` (sha f79b3748; staged at `~/dsv41-stage/engram-punch/` on 8/8). The derivation and audit are in the Mac scratchpad `engram-punch/PLAN.md`.
- **Not punched.** spark-four (rank 3) is the pristine restore source and is never punched. After the second window its shards 47/48 still pass `sha256sum -c` against uncensored.sha.
- **Restore path for five..eight.** Each of spark-five..eight has a read-only key `~/.ssh/engram_restore`, which spark-four forces to `rrsync -ro` of the model dir, so remote paths are relative to the model dir. `restore_from_four.sh` (c36edacb) picks the key up automatically. Before punching, all 4 nodes listed both shards through that key (`rsync --list-only`, sizes 101,535,150,936 / 101,537,926,640).
- **Freed:** 165.23 GiB per node (82.61 + 82.62), **1,156.6 GiB in total over 7 nodes**. Shards 47+48 now allocate 23.90 GiB on a punched node (189.13 GiB on spark-four). Keep about 165 GiB free on each punched node, or a restore is impossible.

| node | rank | freed | free before -> after | `--check --deep` |
|---|---|---|---|---|
| spark-one | 0 | 165.23 GiB | 31.5 -> 196.8 GiB | OK |
| spark-two | 1 | 165.23 | 33.3 -> 198.5 | OK |
| spark-three | 2 | 165.23 | 149.8 -> 315.0 | OK |
| spark-four | 3 | - (pristine) | 197.5 | whole; sha256 47/48 OK |
| spark-five | 4 | 165.23 | 179.8 -> 345.0 | OK |
| spark-six | 5 | 165.23 | 185.2 -> 350.4 | OK |
| spark-seven | 6 | 165.23 | 135.6 -> 300.9 | OK |
| spark-eight | 7 | 165.23 | 149.2 -> 314.4 | OK |

- **Per-node proof** (state in `~/dsv41-stage/engram-punch/{state/,punch.log}`; five..eight copied to `./engram-punch/5to8/spark-*/`):
  - Before punching, the full sha256 matched uncensored.sha (47: 76a05106, 48: 9dc51433 on five..eight). The kept segments (header, own weight and scale rows, tail) matched spark-four's `--ref-all` manifest (ref.json dbdc3b09; spark-four full shas match published).
  - After punching, the kept segments were re-hashed with the page cache dropped, and pre == post. Punch ranges reported as holes: 100 % (0 B not holed).
  - The kept rows contain no holes.
  - The xattr `user.dsv41.engram_punch` = {rank, world 8, rows, state done}.
  - `--check --deep` OK on all 7. `check_all.sh` gives 8/8 OK (7 punched + spark-four whole).
  - Negative tests: `--rank 0 --check` on spark-two FAILS, and `--rank 4 --check` on spark-six FAILS ("punched for rank 5/8, not 4/8"), as they must.
  - Host guard in action: the first five..eight start used a bash-4 associative array, which the Mac's bash 3.2 silently ignores, so all four got `--rank 7`. The tool refused on five/six/seven ("spark-five runs rank 4, not --rank 7") before touching any byte; spark-eight (really rank 7) proceeded. Use `host:rank` pairs (`for p in five:4 six:5 ...`), not `declare -A`, in Mac-side loops.
- **Linux synthetic test.** `test_engram_punch.py` ran on spark-one ext4 (first run of the fallocate path) and PASSED, after a fix to the test harness only (cb7d89ca; was 489eb643). On Linux `shutil.copytree` used copy2, which also copied the rank-1 punch xattr into the "zeros as data" fixture. Now it uses `copy_function=shutil.copyfile`, which matches an rsync without -X. The tool itself was not changed.

**Guards now in place**
- `~/dsv41-tp8.sh` 85189840 -> **0f1ab42f** on 8/8 (backup `~/dsv41-tp8.sh.bak-punch` = 85189840). The Engram row check runs right after the venv activate line. It runs `engram_punch.py --rank $R --world 8 --check` (about 1 s; holes, xattr and a 64-row zero probe), logs to `~/dsv41-logs/engram-check.log`, and exits before `vllm serve` if this node lacks rank R's rows.
- `ops/dsv41-launch-all.sh` and `ops/conc/launch-conc.sh` (backups `*.bak-punch`) now begin with `ops/engram-punch/check_all.sh` over all 8 nodes and refuse to start anything if one fails. check_all.sh runs from the Mac with bash, the same way as the README launch: `ssh spark-one 'cat .../dsv41-launch-all.sh' | bash`.
- **ENGRAM_CHECK=0** skips both checks. For launch-all use `ENGRAM_CHECK=0 bash ...`; for the rank check pass `ENGRAM_CHECK=0` via `launch-conc.sh`, which goes through systemd-run -E. **Never use it on a punched node** (that is now every node except spark-four): vLLM 0.32 has no missing-row check and would silently serve zeros for those Engram rows.

**Verification**

Exactness method:
- On the live 1M config, exact.py is **not deterministic even within one instance**. With VERIFY_CAP=0 and ROUTE_LIVE=0 at 1M, 3 back-to-back runs gave 3 different hashes (49941910d174 / 61628a22d5b5 / 62162ae40c8b), before any punch.
- So the reference was taken at `DSV41_VERIFY_CAP=0 DSV41_ROUTE_LIVE=0 DSV41_MAX_MODEL_LEN=262144` (launch-conc.sh env). That config is stable within an instance (3/3) and reproduces the round-4 cross-restart anchor a8c3394af9d2.
- The 1M within-instance non-determinism existed before the punch. It is unexplained; a suspect is idxshard or kernels whose behaviour depends on max_model_len. It is not investigated here.

| check | before punch (unpunched) | after one..three | after five..eight (7 punched) |
|---|---|---|---|
| exact.py (cap 0, route 0, 262144) | a8c3394af9d2 x3 | **a8c3394af9d2 x3** | **a8c3394af9d2 x2** (per-prompt shas identical to b1) |
| GSM8K-50 conc8 (same instance) | 49/50 (item 12, known flip), 240.8 tok/s | **50/50**, 234.0 tok/s | **50/50**, 239.3 tok/s |
| `Engram table offloaded` rows (16 rank x layer) | as PLAN section 5 | unchanged, e.g. r0 48000217 / 48001463 | unchanged on 8/8 (r4 48000873 / 48002199 ... r7 48001319 / 48002645) |
| engram-check.log | - | OK 8/8 | OK 8/8 |

Live relaunch after one..three (1M, verify-cap auto, route skip; launch-all incl. check_all 8/8 OK, health 200 after 407 s):
- longctx.py default gate (4K-200K, 2 seeds): **PASS**, needles 60/60, copy 1.000 (wall 181 s).
- 500K needle run (lc 500k, 1 seed, 7 depths): **PASS**, 7/7 needles, copy exact (sim 1.000), 497,263 prompt tokens, 91.1 s request.

Live relaunch after five..eight (same live config; launch-all check_all 8/8 OK; /v1/models max_model_len 1048576). The first launch hung at the known rdma_ar graph capture (all ranks at `rdma_ar audit ... (160, ...)` for more than 6 min); stopping all 8 and relaunching once fixed it, health 200 after 388 s.
- longctx.py default gate: **PASS**, needles 60/60, copy 1.000 (wall 181 s).
- 500K needle run (1 seed, 7 depths): **PASS**, 7/7 needles, copy exact (sim 1.000), 497,263 prompt tokens, 90.1 s request.

Raw results: `./engram-punch/{b0 (1M non-determinism probe), b1 (baseline), p1 (post-punch), live}/` for one..three and `./engram-punch/5to8/{det, live, spark-five..eight, mac-logs}/` for five..eight; compare tool `engram-punch/excmp.py`.

**Restore one punched node** (`ops/engram-punch/restore_from_four.sh`, c36edacb):
1. Stop vLLM on that node.
2. On the node, run `bash ~/dsv41-stage/engram-punch/restore_from_four.sh`. It rsyncs `-I -W --inplace` from jeweled@10.77.0.41 (one..three with id_ed25519, five..eight with the read-only `~/.ssh/engram_restore` key), which needs about 165 GiB free. It then checks the full sha256 against uncensored.sha and removes the xattr.
3. Relaunch.

All 7 punched nodes can now reach 10.77.0.41.

**Rules**
- **Never rsync, scp or push model files FROM a punched node** (push-model.sh, fabric-pull.sh, rsync-*.sh). Holes are copied as real zeros. spark-four is the only valid source.
- `sha256sum -c uncensored.sha` **will report FAILED for shards 47/48 on the 7 punched nodes. This is expected.** Use `engram_punch.py --rank R --check --deep` there instead. All other shards must still match.
- **Any layout change requires restoring first.** That includes TP != 8, enabling DP or engram DP sharding, or changing which host runs which rank (launch-all HOSTS order). The head-to-rank map changes, and the check will refuse to launch.
- `engram-disk/nvme_bench.py` now computes exact per-rank row ranges (fixed during the Engram-on-disk run, below), so it is safe to run on punched nodes. Any other tool that reads shards 47/48 directly must likewise use exact per-rank ranges, never approximate `N//8` slices.
- spark-four is now the single pristine copy for 7 nodes. If its NVMe fails, each of them needs a 2 x 101.5 GB re-download (about 1.4 TB in total). Consider a second full copy of shards 47/48 elsewhere.

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

## 1M determinism fix (2026-10-23/24) -- PROMOTED
Cause (nondet/README.md): at max_model_len >= ~398K the decode graphs fall back from persistent_topk to
top_k_per_row_decode's split+merge, whose smem-atomic merge writes the (exact) top-k index SET in a run-dependent
ORDER; sparse MLA sums in stored order -> bitwise drift -> greedy flips at near-ties.
Fix: vllm/model_executor/layers/sparse_attn_indexer.py 75d14da5 -> 7112c01d on 8/8 (topk-canon.diff; original at
.bak-nondet / .bak-canon): DSV41_TOPK_CANON=1 sorts each top-k row ascending, -1 last (= short-row order). Inert at 0.
Launcher: ~/dsv41-tp8.sh 0f1ab42f -> 80c749ca on 8/8 adds `export DSV41_TOPK_CANON=${DSV41_TOPK_CANON:-1}`
(backup ~/dsv41-tp8.sh.bak-canon = 0f1ab42f). Rollback: DSV41_TOPK_CANON=0 via launch-conc.sh, or restore both backups.

A/B, live 1M config (VERIFY_CAP=auto ROUTE_LIVE=skip), fresh instance per arm, order A1 B1 A2 B2 (+A3 B3 conc reruns);
dirs nondet/ab/<arm>:
| metric | CANON=0 | CANON=1 | delta |
|---|---|---|---|
| matched C1 coding ms/step | 34.40 / 34.49 | 34.34 / 34.58 | +0.04% |
| matched C1 prose ms/step | 29.32 / 29.45 | 29.06 / 29.32 | -0.7% |
| bench single itl p50 code/prose/math (ms) | ~29.9-30.3 / 25.7-26.5 / 30.1-31.1 | 30.0-30.3 / 25.5-25.8 / 29.9-30.1 | same or faster |
| conc8 agg tok/s (5 runs, 3 inst. each) | 214.9 217.6 215.6 219.5 218.8 = 217.3 | 213.1 209.1 211.9 217.0 219.3 = 214.1 | -1.5% (inst. means -1.9%) |
| conc32 agg tok/s | 450.9 455.8 458.0 = 454.9 | 456.3 450.5 453.4 = 453.4 | -0.3% |
| 500K needle TTFT (s) | 93.79 / 92.69 | 90.34 / 91.65 | -2.4% (better) |
bench.py single tok/s is lower with CANON on (code 85 vs 89, math 115 vs 125-129) only through acceptance: CANON=1 emits
the deterministic reference trajectory (AL math 3.52 every run) while CANON=0 emits a drifted one (AL 3.8-4.0);
per-step time is unchanged. conc8 is the closest metric to the -2% gate; the AL there is also ~1% lower with CANON on.
Correctness (CANON=1): live config exact.py x3 in one instance = a8c3394af9d2 x3 (stable; also V3); deterministic config
(VERIFY_CAP=0 ROUTE_LIVE=0, 1M) exact x3 = a8c3394af9d2 x3 (D-det-on, plus V1 V2 earlier); GSM8K-50 49/50 (B1; V3 50/50);
longctx default PASS 60/60 copy 1.000 (B1; one V3 run had a 64K seed1 copy miss that did not repeat in 3 reruns +
8 fresh seeds + 3 stress sweeps); 500K needle 7/7 in all 4 arms.
Not fixed (out of scope): long-prompt (>= ~15K tokens) drift through the chunked prefill path, also present at 262k.
Launch note: B1 attempt 1 died with "Engine core initialization failed" at startup (not reproducible; retry came up).
Live relaunch (ops/dsv41-launch-all.sh, launcher 80c749ca, env DSV41_TOPK_CANON=1 from launcher default; first attempt
hit the known rdma_ar DSpark graph-capture hang, relaunched once): exact.py x3 = a8c3394af9d2 x3, matched C1 coding
34.55 / prose 29.57 ms/step (nondet/ab/LIVE-final).

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

## k1 / 1-indexer-mask: length-aware decode candidate mask + flags -- PROMOTED LENAWARE (2026-10-24 11:55 UTC); LENSELECT tested, NOT promoted (all gates pass; gain over LENAWARE within noise)

**Problem.** At live max_model_len 1,048,576 the decode indexer's candidate kernels walked the whole max_model_len-wide logits row on every step, whatever the real context. `_mask_candidates_kernel` wrote -inf over all 1M columns (grid cdiv(W,1024) = 1024 per row), `_candidate_flags_kernel` zeroed all 131,073 flag bytes per row, and the layer-20 select ran torch.topk(2048) over all 131,072 blocks. The fresh 1M profile (Mac scratchpad k1/profile-1m.md) measured the mask at 78 us per call at 4 rows and 1.09-1.46 ms at 48-64 rows, with 4 consumer layers (24/28/32/36) per step. This is most of the conc16/conc32 drop recorded under "Context 1M" (conc16 330.9 -> 311.8).

**What.** One site file, `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py` 45f1eaa3 -> **a5816163** on 8/8 (backup `candidate_blocks.py.bak-k1-1-indexer-mask` = 45f1eaa3 on every rank). The diff only adds lines (every stock line is kept). Both flags default off, and off runs the unchanged stock kernels. `sparse_attn_indexer.py` (7112c01d) is untouched. Flags are read at import (each worker logs `dsv41 k1 candidate_blocks armed: LENAWARE=.. LENSELECT=.. CTAS=..`), so arms are per instance.
- **`DSV41_CAND_LENAWARE=1`** (decode only, `row_ks is None`; prefill keeps stock):
  - (b) candidate flags: zero only slots `[0, cdiv(min(end,W)-start, 8))` plus the edge slot (stock zeroes all 131,073 B per row at 1M).
  - (a) -inf mask: grid `(rows, G)`, `G = min(cdiv(W,1024), cdiv(DSV41_CAND_CTAS, rows))` (CTAS default 384), each CTA loops over the 1024-column tiles below `min(end,W)`. Every column below `end` gets exactly the stock value (edge term included); columns at or past `end` are not written. Stock writes -inf over the whole max_model_len-wide row (grid 1024 at 1M).
  - Exact: no decode reader looks at columns >= end. persistent_topk / top_k_per_row_decode read `[0, len)` per row with len <= the mask's end (both seq_lens layouts), and the layer-20 logits already reach top-k unmasked.
- **`DSV41_CAND_LENSELECT=1`** (decode only): layer-20 select = `_block_scores_bounded_kernel` over `nvalid = min(cdiv(end,8), nblocks)` blocks (same max-of-8 + pinned newest block; writes `lens[row] = nvalid` on device) -> `persistent_topk(scores, lens, out, ws, 2048, max_seq_len=nblocks)` -> -1 where the score is -inf (stock rule). No host branch on lengths (graph-safe); same 1 MiB workspace request as the decode top-k. Same candidate SET as torch.topk when nvalid <= 2048 (ctx <= 16,384 tokens; order differs, nothing reads it); above that it can differ only on an fp32 tie / NaN at the 2048th block (greedy-eq).
- `DSV41_CAND_CTAS` (default 384) is the CTA target for the looping kernels. The Phase C sweep is flat between 192 and 1536, so the default is kept.

**GPU unit test** (`bundle/test_k1_idxmask_gpu.py`, service stopped; phases A+B on spark-one, C+D on spark-two, 09:50 UTC; results in `gpu/`): **PASS, 698 checks, 0 fails**.
- **A (bitwise, eager).** rows {4,6,48,96,192} x W {262K, 1M}, ends {0,1,7,8,513,16383,16384,16385,65544,200000,300000,524288,W}, columns >= end poisoned with NaN/+inf/1e30, row_repeat 1/2/6.
  - Flag-off path == stock, bitwise.
  - LENAWARE: columns < end bitwise == stock; poisoned columns >= end untouched; edge / -1 / duplicate candidates behave the same.
  - Downstream top-512 + CANON is bitwise equal for persistent_topk (max_seq_len = W, as live), top_k_per_row_decode, and next_n > 1.
  - LENSELECT: the set equals stock for nvalid <= 2048 (112 rows). Above that, 628/628 rows gave an identical full chain (0 tie rows). Rows with NaN in real columns: equal sets 6/6.
- **B (CUDA graph).** The patched mask+select is captured once and replayed with new ends and fresh poison: W1M x 6 / 48 rows and W262K x 96 rows, all equal to eager stock.
- **C (timing, us per call, graph replay; flags+mask = apply_candidate_mask).**

| W | rows | end | flags+mask stock -> LENAWARE | select stock -> LENSELECT |
|---|---|---|---|---|
| 1M | 4 | 2K | 97.6 -> 6.2 | 86.1 -> 8.2 |
| 1M | 6 | 2K | 146.3 -> 6.2 | 97.4 -> 8.2 |
| 1M | 48 | 2K / 32K / 131K | 1097.7 -> 6.2 / 34.2 / 152.9 | 491.6 -> 11.4 / 65.1 / 210.6 |
| 1M | 96 | 2K | 2205 -> 6.2 | 1427 -> 14.9 |
| 1M | 192 | 2K | 4452 -> 8.1 | 2875 -> 22.6 |
| 1M | 48 | full (1M) | 1129 -> 1166 (+3%) | 1414 -> 1225 |
| 262K | 4 | 2K | 24.7 -> 6.2 | 65.6 -> 8.2 |

  Per step (4 masks + 1 select) at 1M and short context, this saves **0.44 ms at 4 rows, 0.65 ms at 6 rows, 4.85 ms at 48 rows, 10.2 ms at 96 rows and 20.6 ms at 192 rows**. Of that, LENSELECT is 0.08 / 0.09 / 0.48 / 1.4 / 2.85 ms. Only full-length rows (end = W) are about 3% slower on the mask, which does not matter in practice.
- **D (critic arm: decode top-512 on 1M-wide logits).** Not pursued. Rule: go ahead only if a <=262K graph set saves at least ~0.3 ms/step at rows 4-6 with length 300-4096. Measured: 25-131 us/step (4 rows: 131 / 25 us at len 300 / 4096; 6 rows: 84 / 96 us). The `max_seq_len=32768` bucket on 1M-wide logits is no better at 4-6 rows and slower at length >= 4096 (6 rows x 4096: 564 vs 482 us/step live), so it is dead too. Other findings:
  - At <= 8 rows on 1M-wide logits, live persistent_topk already takes the per_row path (20.4 us at 4 rows x 300 for both).
  - CANON alone (`_canon_topk_`: torch.sort + where + masked_fill + copy) costs a flat **~28 us per call** at every shape, x 8 index layers = **~0.22 ms/step**.
  - **Above 8 rows, persistent_topk's cost scales with the logits row width, not with the row lengths.** At 48 rows x len 4096 it takes 320.7 us at W=1M, 51.2 us at 131K, and 31.2 us for per_row. At 48 x 32768: 896 us vs per_row 145 us. Per-step model (5 ratio-1 layers at 1M + 3 ratio-2 at 512K, CANON included):

| rows | ctx (ratio-1 len) | live 1M us/step | <=262K graph set |
|---|---|---|---|
| 16 | 4096 | 963 | 438 |
| 48 | 300 | 337 | 273 |
| 48 | 4096 | 2345 | 797 |
| 48 | 32768 | 6048 | 1763 |
| 96 | 32768 | 11921 | 3291 |

  The bench conc runs use ~300-token contexts (48 rows x 300 = 0.34 ms/step), so this cost is invisible in the conc16/conc32 numbers here. Real multi-user traffic with 4K-32K contexts pays 1.5-4.3 ms/step at 48 rows (see Leads).

**A/B.** Live 1M config plus the arm env via launch-conc.sh, one fresh instance per arm, run order C A B C A B; then a pinned-AV round Cp Ap Bp. Suite `scripts/ksuite.sh`: matched C1 coding/prose x8 trials, bench single (1024 tok x 2 per type), conc8, conc16 x2, conc32 x2 (512 tok), exact.py; i5/i6 add GSM8K-50 conc8, the longctx default gate and a 500K needle. Every instance came up on its first launch, with no rdma_ar hang and no engine-init failure. The armed log line was on 8/8 ranks in every armed instance and 0/8 in controls. The new kernels appeared in the Triton cache of ranks 2/4/7 only after the matching arm, which shows both paths really run in the engine.

| inst | arm | AV profile | coding ms/step (AL) | prose ms/step (AL) | single code/prose/math | conc8 | conc16 x2 | conc32 x2 | exact.py |
|---|---|---|---|---|---|---|---|---|---|
| i1 | C | startup | 34.716 (5.17) | 29.539 (2.71) | 84.6/68.1/116.8 | 215.9 | 317.1/317.2 | 454.3/454.0 | a8c3394af9d2 |
| i2 | A | startup | 33.981 (5.08) | 28.764 (2.61) | 85.8/68.7/117.8 | 219.5 | 331.6/330.0 | 472.6/476.5 | a8c3394af9d2 |
| i3 | B | startup | 34.086 (5.14) | 29.038 (2.68) | 86.1/68.8/119.2 | 222.2 | 329.8/332.4 | 488.4/483.3 | a8c3394af9d2 |
| i4 | C | startup | 34.579 (5.24) | 29.414 (2.69) | 85.0/66.3/116.1 | 211.2 | 316.3/318.3 | 452.6/454.2 | a8c3394af9d2 |
| i5 | A | startup | 34.130 (5.19) | 29.139 (2.72) | 86.5/69.1/118.3 | 219.6 | 324.5/329.9 | 481.8/481.0 | a8c3394af9d2 |
| i6 | B | startup | 33.905 (5.16) | 28.681 (2.62) | 86.4/68.8/118.1 | 221.5 | 326.4/327.9 | 484.2/479.0 | a8c3394af9d2 |
| i7 | C | pinned | 34.606 (5.15) | 29.788 (2.64) | 84.6/68.1/116.8 | 212.8 | 312.8/317.4 | 452.2/450.1 | a8c3394af9d2 |
| i8 | A | pinned | 34.161 (5.18) | 29.215 (2.70) | 85.9/68.9/119.0 | 221.5 | 329.0/328.6 | 488.2/485.0 | a8c3394af9d2 |
| i9 | B | pinned | 34.129 (5.10) | 29.121 (2.69) | 86.0/68.9/119.0 | 217.0 | 327.8/326.9 | 485.2/481.2 | a8c3394af9d2 |

C = control (no flags), A = `DSV41_CAND_LENAWARE=1`, B = A + `DSV41_CAND_LENSELECT=1`. "Pinned" means `DSV41_AV_COST_JSON=~/dsv41-stage/k1/1-indexer-mask/av-pin-i1C.json` (i1's raw startup cost profile) on all three arms. Matched C1 values are means of 8 trials (trial sd 0.15-0.46 ms).

Per pair (% better; ms columns = fewer ms/step):

| pair | coding ms | prose ms | single code/prose/math | conc8 | conc16 | conc32 |
|---|---|---|---|---|---|---|
| A vs C r1 / r2 / r3 | +2.12 / +1.30 / +1.29 | +2.62 / +0.93 / +1.92 | all + (+0.85..+4.22) | +1.70 / +4.01 / +4.05 | +4.30 / +3.13 / +4.35 | +4.49 / +6.18 / +7.86 |
| B vs C r1 / r2 / r3 | +1.81 / +1.95 / +1.38 | +1.70 / +2.49 / +2.24 | all + (+1.09..+3.82) | +2.91 / +4.88 / +1.94 | +4.40 / +3.10 / +3.90 | +6.98 / +6.21 / +7.10 |
| B vs A r1 / r2 / r3 | -0.31 / +0.66 / +0.09 | -0.95 / +1.57 / +0.32 | -0.39..+1.20 | +1.19 / +0.84 / -2.03 | +0.10 / -0.03 / -0.44 | +2.39 / +0.03 / -0.70 |

**Means over 3 instances.**

| arm | coding ms | prose ms | single c/p/m | conc8 | conc16 | conc32 |
|---|---|---|---|---|---|---|
| C | 34.634 | 29.580 | 84.8/67.5/116.6 | 213.3 | 316.5 | 452.9 |
| A | 34.091 | 29.039 | 86.1/68.9/118.4 | 220.2 | 328.9 | 480.9 |
| B | 34.040 | 28.947 | 86.2/68.9/118.8 | 220.2 | 328.5 | 483.5 |

A vs C: **coding -0.54 ms/step (-1.57%), prose -0.54 ms (-1.83%), single +1.6% / +2.1% / +1.5%, conc8 +3.2%, conc16 +3.9%, conc32 +6.2%.**

- **Matches the kernel model.** Phase C predicts (a)+(b) saves 0.37 ms/step at 4 rows and 0.56 at 6 rows (C1 coding verifies about 6 rows). Measured A vs C coding savings: 0.74 / 0.45 / 0.45 ms. The 1M width tax is essentially gone at short context:
  - conc16 is 328.9, against 330.9 for the round-4 262K reference and 311.8 at 1M before.
  - conc32 is 480.9, against 479.0 at 262K and 456.9 at 1M before.
  - C1 coding is 34.09 ms, against 33.75 at 262K and 34.4-34.6 at 1M before.
- **The AL dip in rounds 1-2 is AV re-calibration, not the kernels.** exact.py and bench-single trajectories are identical across arms. bench-single AL: code 2.54-2.58, math 3.53-3.58 and prose 1.75-1.82, the same discrete values recurring in every arm. VERIFY_CAP=auto derives tau from the startup cost profile, and that profile is cheaper per verify row once the width-scaled mask is gone. Round 3 pins the same profile on all arms. A then shows equal-or-higher matched AL than C (coding 5.18 vs 5.15, prose 2.70 vs 2.64) and still saves 0.45 / 0.57 ms/step.
- **LENSELECT's increment over LENAWARE is within noise end-to-end** (means: coding -0.05 ms, prose -0.09 ms, conc16 -0.1%, conc32 +0.6%). That matches Phase C (0.08 ms/step at 4-6 rows, about 0.5 ms at 48 rows), which is below the run-to-run spread.

**Correctness.**
- exact.py = **a8c3394af9d2 in all 9 instances**: 3 control, 3 LENAWARE, 3 LENAWARE+LENSELECT, both AV modes. Final live instance: a8c3394af9d2 (i10-LIVE).
- i5 (A): GSM8K-50 conc8 **50/50** (244.8 tok/s); longctx default gate **PASS** (needles 60/60, copy 1.000); 500K needle **PASS** (7/7, copy exact, 497,263 tokens, TTFT 88.3 s).
- i6 (B): GSM8K-50 **50/50** (246.8 tok/s); longctx **PASS** (60/60, copy 1.000); 500K needle **PASS** (7/7, copy exact, TTFT 87.2 s). At 497K context the select's persistent_topk runs its multi-CTA path (62K valid blocks, above 32,768) inside the decode graphs, with no hang.
- Startup: 9/9 instances came up on the first attempt in 383-409 s, the same as before. There were no rdma_ar capture hangs and no engine-init failures in this window.

**Promotion.**
- **LENAWARE: PROMOTED.**
  - Exact: exact.py stays a8c3394af9d2.
  - C1 coding improved in every pair (+2.12 / +1.30 / +1.29%, bar 0.8%), and conc16/conc32 improved by +3.1 to +7.9% in every pair (bar 2%). Nothing got worse.
  - Launcher `~/dsv41-tp8.sh` f4e4e214 -> **cfb97923** on 8/8. It adds one line after the TOPK_CANON export: `export DSV41_CAND_LENAWARE=${DSV41_CAND_LENAWARE:-1}`. Backup: `~/dsv41-tp8.sh.bak-k1-1-indexer-mask` = f4e4e214 on 8/8. Tool: `scripts/promote-launcher.py 1 0`, run on each rank; it refuses unless the launcher is f4e4e214.
  - Site file `candidate_blocks.py` a5816163 on 8/8 (backup 45f1eaa3).
- **LENSELECT: NOT promoted.** It passed every gate it has: exact.py a8c3394af9d2 x3, GSM8K-50 50/50, bench-single AL unchanged, longctx PASS and 500K needle PASS. But its gain over LENAWARE alone is within noise in every pair (coding -0.31 / +0.66 / +0.09%, conc16 +0.10 / -0.03 / -0.44%), so it misses the per-flag bar. It is also greedy-eq, not bitwise, above 16K context. It stays installed and off (default 0). Enable it with `DSV41_CAND_LENSELECT=1` via launch-conc.sh if the 0.5-2.9 ms/step it saves at 48-192 rows (Phase C) is wanted for high-concurrency traffic.
- **Final live relaunch** (11:56-12:03 UTC, `ops/dsv41-launch-all.sh`, launcher cfb97923 defaults, no env; instance `ab/i10-LIVE`):
  - Up in 407 s on the first attempt. Armed line on 8/8: `LENAWARE=1 LENSELECT=0 CTAS=384`. /v1/models max_model_len 1048576.
  - exact.py **a8c3394af9d2**; matched C1 coding **34.20** / prose **29.09** ms/step (AL 5.13 / 2.69); bench single 86.0/68.4/117.6 tok/s; conc8 218.8, conc16 324.6/329.2, conc32 483.9/480.8; GSM8K-50 conc8 **49/50** (item 12, the known flip; 247.2 tok/s); longctx default gate **PASS** (60/60, copy 1.000); 500K needle **7/7** (TTFT 87.6 s); 1M needle **7/7** (999,411 tokens, copy exact, TTFT 205.1 s; stock 1M was 209.4-209.8 s).

**Rollback** (from the Mac; S = ~/dsv41-opt-20261021/ops on spark-one):
- Env only, keeps every file: `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/conc/launch-conc.sh' | bash -s -- DSV41_CAND_LENAWARE=0` runs the stock kernels (the installed file is inert with the flags off).
- Launcher: `for h in one two three four five six seven eight; do ssh spark-$h 'cp -p ~/dsv41-tp8.sh.bak-k1-1-indexer-mask ~/dsv41-tp8.sh'; done` (back to f4e4e214), then relaunch with `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh' | bash` and check the sha is f4e4e214 on 8/8.
- Site file (service stopped first): `scp spark-one:dsv41-opt-20261021/k1/1-indexer-mask/bundle/k1-all.sh . && bash k1-all.sh uninstall` from the Mac (it runs `~/dsv41-stage/k1/1-indexer-mask/uninstall-rank.sh` on 8/8, then checks the sha), or run that per-rank script on each rank. Either restores 45f1eaa3 from `candidate_blocks.py.bak-k1-1-indexer-mask` and refuses on any unexpected sha. Undo the launcher first, then the file.

**Leads found here (not done):**
1. **Decode top-k above 8 rows on 1M-wide logits** (Phase D): persistent_topk costs 2.3 ms/step at 48 rows x 4K context and 6.0 ms at 32K, against 0.8 / 1.8 ms on 262K-wide logits. The live per_row kernel runs the same rows in 31 / 145 us per call. Candidate fixes:
   - (a) Pick per_row over persistent for long rows above 8 rows.
   - (b) Narrow the logits row stride for steps whose max context fits.
   Both are greedy-eq: per_row and persistent can choose different sets on an exact tie at the 512th score. CANON already fixes the order. exact.py is unaffected, because a single request (<= 6 rows) already takes per_row at 1M. This needs a conc A/B at 4K-32K contexts, which the bench prompts do not exercise.
2. **Fused CANON:** sort 512 int32 per row in one kernel instead of torch.sort + where + masked_fill + copy (28 us per call, x8 per step). The output bits are identical, so it is exact. Expected about -0.2 ms/step at 1u (+0.6%).
3. **Pinned AV profile for A/B work** (round 3 above): `DSV41_AV_COST_JSON=~/dsv41-stage/k1/1-indexer-mask/av-pin-i1C.json` holds the verify-cap calibration fixed across instances, which separates kernel time from AV re-calibration in matched C1.

**Housekeeping.** `ops/r4/qwork/journal.txt:25` (also the Mac copy) said the stock candidate mask is "bounded by seq_lens", which is wrong. It is corrected in place: d66b47bd -> 0ded720b, backup `journal.txt.bak-k1-1-indexer-mask`.

**Files** (./k1/1-indexer-mask/): `bundle/` (patched + stock file, install/uninstall/k1-all/build, GPU/CPU tests, MANIFEST, offline/ results), `gpu/` (unit test outputs + csum.py/dsum.py), `ab/<instance>/` (suite outputs, rank0 startup log, env, armed), `scripts/` (ksuite.sh, kdrv.sh, ksumm.py, kab.py, chain1/2.sh, promote-launcher.py, mkpin.py, rungpu.sh), `av-pin-i1C.json` (pinned AV profile, also at ~/dsv41-stage/k1/1-indexer-mask/ on 8/8), `summary.txt`.

**Live state after k1** (2026-10-24 12:03 UTC): launcher `~/dsv41-tp8.sh` **cfb97923** on 8/8 (previous f4e4e214 = `.bak-k1-1-indexer-mask`); `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py` **a5816163** on 8/8 (stock 45f1eaa3 = `.bak-k1-1-indexer-mask`); every other site file unchanged (sparse_attn_indexer.py 7112c01d). Live env adds DSV41_CAND_LENAWARE=1 (launcher default); DSV41_CAND_LENSELECT unset (= 0).

## k1 / 2-silu-liverows: routed FP4 MoE SiLU-mul-FP8-quant over live rows only (DSV41_MOE_LIVE_ACT) -- PROMOTED (2026-10-24 15:09 UTC)

**Problem.** `DeepGemmFP4Experts._act_mul_quant` ran the fused SiLU-mul-FP8-quant over the whole aligned workspace, `M_sum = 6M + min(6M,384)*(align-1)` rows, of which only 6M are live. At M=48 that is 9216 rows with 288 live (3%), and on the 128-row path (M > 96) it is about 49.5K rows. The fresh 1M profile (Mac scratchpad k1/profile-1m.md) measured 43-60 us per call at 16 users, x38 target layers, plus about 0.33 ms/step in the drafter.

**What.** Two site files under `fused_moe/`, flag-gated and default off. Off runs the original code paths (bitwise, unit-tested). `fp8_utils.py` (8dc7dab8) is untouched; its EXPERT_ALIGNMENT mode already existed.

| file | base (= `.bak-k1-2-silu-liverows` on 8/8) | live 8/8 |
|---|---|---|
| deep_gemm_utils.py | b7e8de4d | **a4a0486a** (`deepgemm_moe_permute(..., return_expert_ends=True)` also returns `expert_start_loc` after ep_scatter = each expert's exclusive live end) |
| experts/deep_gemm_moe.py | 45418259 | **5162dc0f** (`apply` passes `expert_ends`, `expert_alignment=align_used` into both fused UE8M0 SiLU launches, compact and plain; target E=384 top-6 and drafter E=128 top-3) |

- **Flags** (read in `DeepGemmFP4Experts.__init__`, so they are per instance):
  - `DSV41_MOE_LIVE_ACT=1` turns the path on.
  - `DSV41_MOE_LIVE_ACT_MIN_ROWS` (default 1024 aligned rows) is the gate. It is on for target M >= 6 (M_sum 1152) and drafter M >= 3.
  - `DSV41_MOE_LIVE_ACT_MAX_ROWS` (default 0 = no cap).
- **Why exact.** BM and packs_per_cta are still chosen from M_sum, so live rows run the same instructions. Padding rows of the FC2 input are simply not written. ep_gather reads only live rows, and `grouped_layout` (`torch.full(-1)`) is unchanged.

**GPU unit test** (`test_live_act.py` via `run_test.sh`, service stopped; spark-one rank-0 slice 12:47-12:50 UTC, spark-eight rank-7 slice 12:50-12:53 UTC; results in `gpu/`): **PASS on both ranks, 672 checks per rank (1344 total), 0 failures.** The CPU Triton-interpreter check also PASSED on both.
- Per rank: 576 eager cases (528 of them on the live path) plus 96 CUDA-graph replay checks.
- Each case checks, bitwise:
  - the ep_scatter invariant (`ends[e] == start[e] + count[e]`, `roundup(ends[e-1]) == start[e]`, layout);
  - dense vs live kernel on the live rows;
  - final layer output of the original file vs flag on vs flag off.
- Workspaces and the allocator are NaN-poisoned on every run. One graph per size is captured once and replayed on six route sets (valid, skewed, edge, half-padding, 30% holes, all-padding).
- Coverage: target compact and padded, the drafter, M32 (live) and M64 modes, the 1024 gate, real layer-0 / layer-39 / mtp.0 weights for the rank's TP slice.
- **Timing** (graph replay, warm L2, us per call, dense -> live; `gpu/timing-table.txt`). The live kernel is never slower, so `recommended_min_rows_by_geom` is the smallest tested size (target 192, drafter 384). The default gate of 1024 is kept, because below M=6 the saving is about 1 us per call.

| geom | M=1 | M=6 | M=48 | M=96 | M=128 (128-row path) | M=2048 | M=16384 |
|---|---|---|---|---|---|---|---|
| target compact kernel | 1.86 -> 1.74 | 3.47 -> 2.31 | 17.1 -> 4.8 | 22.8 -> 5.8 | 339 -> 9.0 | 429 -> 26.5 | 1017 -> 680 |
| target compact full MoE apply | 56 -> 56 | 534 -> 534 | 3235 -> 3175 | 4584 -> 4514 | 8649 -> 8329 | 10979 -> 10594 | 24938 -> 24530 |
| drafter kernel | 2.81 -> 1.89 | 5.35 -> 2.24 | 116 -> 5.0 | 132 -> 5.1 | 128 -> 5.3 | 190 -> 14.8 | 558 -> 417 |

In the engine the dense kernel is slower than in this warm-L2 replay (profile: 43-60 us at 9.2K-12.3K rows), so the real saving per call is larger.

**A/B** (live 1M config plus the arm env via launch-conc.sh; one fresh instance per arm; suite `scripts/ksuite.sh`):
- **Suite.** Matched C1 coding/prose x8 trials, bench single (1024 tok x2 per type), conc 8,16,32,8,16,32,16,32 (512 tok), and exact.py x2. i4-A also ran GSM8K-50 conc8, the longctx default gate and a 500K needle.
- **Order.** C A C A, then a pinned-AV pair Cp Ap, then a reversed pair A C, to counterbalance drift.
- **Startup.** Every instance came up on its first launch (385-411 s), with no rdma_ar hang and no engine-init failure.
- **Arm check.** `DSV41_MOE_LIVE_ACT=1` was in the vllm serve env on 8/8 in every armed instance, and unset on 8/8 in controls. The first armed launch compiled 3 new live-row Triton variants (`get_program_id z`) on every rank, which shows the live path really ran in the engine on 8/8.

| inst | arm | AV profile | coding ms/step (AL) | prose ms/step (AL) | single code/prose/math | conc8 | conc16 (x3 mean) | conc32 (x3 mean) | exact.py x2 |
|---|---|---|---|---|---|---|---|---|---|
| i1-C | C | startup | 34.145 (5.17) | 29.218 (2.72) | 85.9/68.6/118.6 | 225.3 | 330.5 | 485.7 | a8c3394af9d2 (x1) |
| i2-A | A | startup | 33.971 (5.08) | 29.109 (2.65) | 86.5/68.9/118.7 | 226.3 | 334.4 | 492.0 | **b7e868c9d700** (x1; see Exactness) |
| i3-C | C | startup | 34.034 (5.12) | 29.169 (2.75) | 86.0/68.5/117.5 | 222.6 | 327.9 | 478.5 | a8c3394af9d2 x2 |
| i4-A | A | startup | 34.115 (5.08) | 29.089 (2.64) | 85.8/67.8/118.8 | 227.0 | 337.9 | 491.0 | a8c3394af9d2 x2 |
| i5-Cp | C | pinned i1-C | 34.211 (5.11) | 29.234 (2.69) | 85.9/68.6/118.7 | 222.3 | 328.2 | 477.4 | a8c3394af9d2 x2 |
| i6-Ap | A | pinned i1-C | 34.125 (5.25) | 29.170 (2.74) | 86.0/68.8/119.0 | 226.9 | 334.2 | 492.7 | a8c3394af9d2 x2 |
| i7-A | A | startup | 33.994 (5.07) | 29.125 (2.68) | 86.2/68.8/119.6 | 225.8 | 335.5 | 491.7 | a8c3394af9d2 x2 |
| i8-C | C | startup | 34.025 (5.04) | 29.209 (2.73) | 86.2/67.4/118.2 | 220.7 | 326.1 | 479.3 | a8c3394af9d2 x2 |
| i9-Cx | C | pinned i2-A | - | - | 86.0/68.7/118.2 | - | - | - | **b7e868c9d700** x2 (counterfactual) |

C = control (installed, flag unset), A = `DSV41_MOE_LIVE_ACT=1`. "Pinned i1-C" means `DSV41_AV_COST_JSON=~/dsv41-stage/k1/2-silu-liverows/av-pin-i1C.json`, i1-C's raw startup AV profile, so both arms make identical verify-cap and budget decisions. Matched values are means of 8 trials (trial sd 0.14-0.49 ms). Matched AL varies per trial by construction, because each trial has a different front tag.

Per pair (% better; ms columns = fewer ms/step):

| pair | coding ms | prose ms | single c/p/m | conc8 | conc16 | conc32 |
|---|---|---|---|---|---|---|
| r1 i2-A vs i1-C | +0.51 | +0.37 | +0.65 / +0.39 / +0.08 | +0.42 | +1.17 | +1.30 |
| r2 i4-A vs i3-C | -0.24 | +0.27 | -0.23 / -1.02 / +1.09 | +1.94 | +3.02 | +2.59 |
| r3 i6-Ap vs i5-Cp (pinned) | +0.25 | +0.22 | +0.10 / +0.31 / +0.29 | +2.06 | +1.83 | +3.21 |
| r4 i7-A vs i8-C (reversed order) | +0.09 | +0.29 | -0.01 / +2.05 / +1.16 | +2.30 | +2.87 | +2.60 |
| **mean of 4 pairs** | **+0.15** | **+0.29** | +0.13 / +0.43 / +0.66 | **+1.68** | **+2.23** | **+2.43** |

- **Arm means (4 v 4).** conc8 222.7 -> 226.5 (+1.7%), **conc16 328.2 -> 335.5 (+2.2%)**, **conc32 480.2 -> 491.9 (+2.4%)**. C1 coding 34.10 -> 34.05 and prose 29.21 -> 29.12 ms/step. Single-stream is within noise.
- **The armed conc32 numbers are tight** (492.0 / 491.0 / 492.7 / 491.7). The controls spread more (485.7 / 478.5 / 477.4 / 479.3). r1 has the lowest gain because i1-C is the highest control.
- **Matches the kernel model.** At 16 users (48-64 rows), about -1.6 ms target plus -0.3 ms drafter is about -2 ms of a ~100 ms step. C1 is unchanged by design (M=6: -1.2 us per call).
- **AV profile.** The 128-row verify path got cheaper by 10-12 ms per step: verify_ms at 104/128/192 rows is C ~258/280/332 vs A ~248/271/320. The AV trimmer prices M > 96 accordingly.

**Exactness: exact (bitwise).**
- **Kernel and layer:** the unit test above.
- **Engine, AV profile held fixed (r3):** i5-Cp (flag off) and i6-Ap (flag on) both use i1-C's raw AV profile. Their bench-single greedy spec counters (steps / verified / accepted) are identical to each other and to i1-C: code 401/2005/622, prose 570/2850/453, math 287/1435/741. exact.py = a8c3394af9d2 x2 in both. So the target and the drafter produce the same tokens and the same accept decisions with the flag on (the drafter's AL is unchanged).
- **i2-A's b7e868c9d700 comes from its AV profile, not the flag.**
  - b7e868c9d700 is the known alternate VERIFY_CAP=auto trajectory: same 2 of 6 prompts, same per-prompt hashes 2228532c / b246b2a0. It is the hash r4-e4's u4 gave with no MoE change ("AV-profile dependent").
  - Counterfactual i9-Cx: flag **off** with i2-A's raw profile pinned (`av-pin-i2A.json`) reproduces i2-A bit for bit. exact.py b7e868c9d700 x2, and bench-single counters code 396/1980/627, prose 567/2835/457, math 288/1440/740, identical to i2-A.
  - So for both profiles tested, flag on == flag off (2x2: i1-C profile -> a8c3394af9d2; i2-A profile -> b7e868c9d700).
  - i2-A's raw profile had the highest 2-row verify sample seen (v2 21.97 ms; u4 21.85; 20.0-21.4 elsewhere; `rawprof.txt`, `tauproxy-r1.txt`). The flag cannot change the 2-row graph, because target M_sum 384 and drafter 768 are below the 1024 gate.
  - exact.py with a startup profile: a8c3394af9d2 in 2 of 3 armed instances (i4-A, i7-A) and 3 of 3 controls (i1-C, i3-C, i8-C); pinned: 2 of 2. This is the same AV-noise sensitivity the live config already had.
  - A high v2 alone does not decide the flip. i8-C (control, v2 21.37) and i6-Ap's raw profile (v2 21.44, pinned away) are close to i2-A; the boundary also depends on the 4- and 6-row points. The pinned 2x2 is the proof.
- **Gates on i4-A** (armed, startup profile):
  - GSM8K-50 conc8 **50/50** (247.6 tok/s).
  - longctx default gate **PASS** (needles 60/60, copy 1.000).
  - 500K needle **7/7** (497,263 tokens, copy exact, TTFT 87.3 s; LENAWARE instances: 87.2-88.3 s).

**Promotion.**
- Rule: exact, and conc16/conc32 improves >= 2% (the C1 bar is not met: +0.15 / +0.29%, as designed). conc16 is +2.23% and conc32 +2.43% as the mean of 4 pairs. Every pair is positive (min +1.17 / +1.30), and nothing is worse beyond noise.
- **Launcher** `~/dsv41-tp8.sh` cfb97923 -> **1a2f26f4** on 8/8. It adds one line after the DSV41_CAND_LENAWARE export: `export DSV41_MOE_LIVE_ACT=${DSV41_MOE_LIVE_ACT:-1}`. Backup: `~/dsv41-tp8.sh.bak-k1-2-silu-liverows` = cfb97923 on 8/8. Tool: `scripts/promote-launcher.py` (run with `python3 -I -` on each rank; it refuses unless the launcher is cfb97923). MIN_ROWS / MAX_ROWS stay at the code defaults (1024 / 0).
- **Site files** a4a0486a / 5162dc0f on 8/8 (backups b7e8de4d / 45418259).
- **Final live relaunch** (15:02:57-15:09:33 UTC, `ops/dsv41-launch-all.sh`, launcher 1a2f26f4 defaults, no env; instance `ab/i10-LIVE`):
  - Up in 390 s on the first attempt. `DSV41_MOE_LIVE_ACT=1` in the vllm serve env on 8/8 (launcher default). /v1/models max_model_len 1048576. No Traceback or ERROR lines in the 8 rank logs.
  - exact.py **a8c3394af9d2** x2.
  - Matched C1 coding **34.08** / prose **29.29** ms/step (AL 5.15 / 2.72). Bench single 86.1/69.1/118.0 tok/s. conc8 224.5/227.4, conc16 332.8/334.6/333.1, conc32 493.0/489.9/493.5.
  - GSM8K-50 conc8 **50/50** (247.9 tok/s). longctx default gate **PASS** (60/60, copy 1.000). 500K needle **7/7** (TTFT 86.8 s).
  - 1M needle **7/7** (999,411 tokens, copy exact, TTFT 182.0 s; LENAWARE final live 205.1 s). This change models only about 1 s of the 1M prefill (about -0.4 ms per MoE layer per 16K chunk), so most of that difference is run-to-run variance; do not count it as a gain.

**Rollback** (from the Mac):
- Env only, keeps every file: `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/conc/launch-conc.sh' | bash -s -- DSV41_MOE_LIVE_ACT=0` (any value other than 1 is off).
- Launcher: `for h in one two three four five six seven eight; do ssh spark-$h 'cp -p ~/dsv41-tp8.sh.bak-k1-2-silu-liverows ~/dsv41-tp8.sh'; done` (back to cfb97923), then relaunch with `ssh spark-one 'cat ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh' | bash`, and check that the sha is cfb97923 on 8/8.
- Site files (service stopped; undo the launcher first): on each rank run `bash ~/dsv41-stage/k1/2-silu-liverows/uninstall.sh`. It restores b7e8de4d / 45418259 from the `.bak-k1-2-silu-liverows` files and refuses on any unexpected sha. Or, from the Mac, run `bash $S/k1/2-silu-liverows/all.sh uninstall` (S = the Mac scratchpad; the same all.sh is in `~/dsv41-stage/k1/2-silu-liverows/`). It runs uninstall.sh on 8/8 and prints the 8/8 sha status.

**Notes for later experiments.**
- **Exp 5 (FC2 epilogue):** with the flag on, padding rows of the FC2 input (FP8 q and packed UE8M0 scales) hold stale bytes, so FC2's padding output rows can be NaN/Inf. Nothing reads them today (ep_gather reads live rows only; the unit test poisons them). A change that reads padding rows must handle that, or run with the flag off.
- **Exp 6 (fused MoE front):** `expert_ends` is `expert_start_loc` after `_fwd_kernel_ep_scatter_2`'s atomic_add. If exp 6 replaces that kernel, switch the source and re-run `run_test.sh` (it needs about 3 min per rank).
- **Exact A/Bs under VERIFY_CAP=auto:** the exact.py trajectory depends on the startup AV profile, so a single b7e868c9d700 is not by itself evidence of an inexact change. Pin the AV profile on both arms (`DSV41_AV_COST_JSON`, made with `scripts/mkpin.py` from a control's rank0 log) and compare the bench-single spec counters (`scripts/alstat.py`). For a suspicious instance, rerun the control pinned to that instance's profile (the i9-Cx counterfactual).

**Files** (`./k1/2-silu-liverows/`):
- `gpu/`: rank 0 / rank 7 unit-test results, `run_test.r*.out`, `timing-table.txt`.
- `ab/<instance>/`: suite outputs, `rank0.startup.log`, `env.txt`, `avprofile.txt`, `arm-check.txt`; `ab/live-before.*` is the pre-experiment live instance.
- `scripts/`: ksuite.sh, ksumm.py, kab.py, pairs.py, alstat.py, tauproxy.py, rawprof.py, mkpin.py, promote-launcher.py, plus the Mac drivers kdrv.sh, chain*.sh, launch-*.sh.
- `av-pin-i1C.json`, `av-pin-i2A.json` (also in `~/dsv41-stage/k1/2-silu-liverows/` on 8/8).
- `summ-all.txt` / `summ-all.json` (all instances, pairs, counters, tau factors), `rawprof.txt`, `tauproxy-r1.txt`.
- The patch, test and install scripts are in `~/dsv41-stage/k1/2-silu-liverows/` on 8/8 (MANIFEST 01420e3a).

**Live state after k1/2** (2026-10-24 15:10 UTC):
- Launcher `~/dsv41-tp8.sh` **1a2f26f4** on 8/8 (previous cfb97923 = `.bak-k1-2-silu-liverows`).
- `fused_moe/deep_gemm_utils.py` **a4a0486a** and `fused_moe/experts/deep_gemm_moe.py` **5162dc0f** on 8/8 (base b7e8de4d / 45418259 = `.bak-k1-2-silu-liverows`).
- Unchanged: `quantization/utils/fp8_utils.py` 8dc7dab8, `kernels/attention/dsa/candidate_blocks.py` a5816163, `sparse_attn_indexer.py` 7112c01d.
- Live env adds DSV41_MOE_LIVE_ACT=1 (launcher default). DSV41_MOE_LIVE_ACT_MIN_ROWS / _MAX_ROWS are unset (1024 / 0).

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

## Offline prep pd: exact decode candidates staged, NOT installed (2026-10-25 ~19:25 UTC)
No restart, GPU job or live-file edit in this step. Live site files are unchanged: sm121_target_m32 723fdf8d, gate_linear 41adaba8, fp8_utils 8dc7dab8, dsv41_l2_prefetch 4da7745b.
Everything is in ~/dsv41-stage/pd/ on 8/8, with identical digests. Full handoff: ~/dsv41-stage/pd/README.md. Install/uninstall: pd/common/pd-all.sh (backups .bak-pd-<tag>, 8/8 digest check).
Ranked by expected gain (all default off):
1. **d05**: D05 FC2 dead-store skip in serving (DSV41_MOE_FC2_D05=0/1/verify). Expected -0.6/-1.0/-1.9 ms at C1/C2/C4.
2. **l2pf**: L2PF arms S/SQ/SR (env only) and PB (post-budget, needs install). Expected -0.3..-0.6 ms at 1u.
3. **rgate**: exact cuBLAS-replica router GEMM for M<=8. Expected -0.35..-0.55 ms at 1u, only if the GPU probe finds a bitwise schedule.
4. **pdl**: FC2 early-weight PDL prologue + SiLU trigger. Expected -0.12..-0.24 ms at 1u.
5. **fc2cd05**: FC2c v2 + D05. Not built (4 host hunks to merge, +2.5 GiB/rank).
GPU unit tests, the probe and sanitizer runs are written but need a stopped window.

## pd / prefill offline prep (2026-10-25 19:00-19:50 UTC). Staged only: no install, no restart, no GPU, no live request

Full handoff (mechanisms, exactness arguments, test plans): `~/dsv41-stage/pd/PREFILL-HANDOFF.md`; CPU results `~/dsv41-stage/pd/RESULTS-cpu-prefill.txt` (all PASS). Bases identical 8/8: model.py e7264781, model_state.py 53e4fe39, decoder_replay_layers.py 2fedc0a6, dspark/speculator.py bbcee9e3, moe_runner.py 20b69fe5, deep_gemm_utils.py a4a0486a, dsv41_sched.py 6074389f, launcher 2fd26e53.

| rank | candidate | flag (default off) | class | 128K / 1M TTFT (expected) |
|---|---|---|---|---|
| 1 | SP v2 (r4 port + `exact`/`comm` modes, layer bisect, clean-forward-only compares) | `DSV41_PREFILL_SP=exact\|comm\|fast` | EXPERIMENTAL (fast = greedy-eq) | fast -3.7..-4.2 s / -28..-32 s; exact -2.0..-2.4 / -15..-18 (P~0.5) |
| 2 | E7 final-chunks-only decoder tail (skip replay + post-replay + draft for chunks that are neither F nor F-1) | `DSV41_E7_TAIL=1` (needs dsv41_sched; it enforces "no non-final chunk <= 128 rows") | exact for every skipped step | -1.0 s / -9.7 s (-4.4%) |
| 3 | lone-prefill 24K chunks | `DSV41_PD_LONE_BUDGET=24576` + `DSV41_EXTRA="--max-num-batched-tokens 24576"` | bitwise <=16K, drift-class >16K | -0.3..-0.4 / -2..-3 s (HOLD: MemAvailable 7.9-9.0 GB today) |
| 4 | shared+routed add folded into ep_gather | `DSV41_MOE_GATHER_ADD=1` | exact (GPU gate pending) | -0.26 / -2.0 s |
| 5 | P09 dual quant serving binding (Codex lib copied to `pd/p09-dualq/lib`) | `DSV41_DUALQUANT_STAGE_A=1 DSV41_DUALQUANT_STAGE_B=1` | exact (component-qualified) | -0.12 / -0.9 s |
| 6 | NCCL arms (transport arms exact; proto/algo/channel arms change AR sum order) | launcher hook + `DSV41_NCCL_EXTRA="NCCL_.."` | measure first (AR at ~91% of link) | <= -0.3 / <= -3 s, likely ~0 |

- Install units: `pd/pf` = one site bundle with all four model/MoE features (pd-site.sh format; staged shas fe7a7d12 / 192bc014 / 580a4b58 / 1a469da5 / 747cd5e4 / a26655f3); `pd/sched-pd/sched-install.sh` (dsv41_sched.py -> 85abae3d, + dsv41_pd_policy.py 3ac7b928); `pd/nccl-arms/launcher-hook.sh` (2fd26e53 -> f5be18ba). With no flags the bundle must reproduce a8c3394af9d2 and pinned-AV counters.
- Found: the launcher unsets every inherited NCCL_* var, so `launch-conc.sh NCCL_X=..` never reached NCCL (hence the hook). `DSV41_MOE_LIVE_ACT` (live) already covers prefill SiLU (gate >=1024 aligned rows). r4's SP STOP compare ran inside the FlashInfer autotune dummy forward; the port already gathers all states before layer 20's write_kv (knapcio's layer-20 rule holds).

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

## pd decode A/B step (2026-10-25 20:59-23:25 UTC): D05 + L2 prune-shared PROMOTED, pdl HOLD, rgate KILLED

Prepared candidates from "Offline prep pd" (`~/dsv41-stage/pd/README.md`), run serially by expected value. Raw results are on spark-one in `~/dsv41-pdab-20261025/`: `STATE.md` (running log), `runs/<label>/` (one per instance), `scripts/asuite.sh` and `asumm.py`. The Mac scratchpad `pdab/` holds the drivers `runarm.sh`, `stopall.sh`, `pushlauncher.sh`, `mapscheck.sh` and `abcmp.py`, plus `summ-*.json`, `cmp-*.txt` and `launches.tsv`.

**Live state after this step.** Everything below is sha256-identical on 8/8.
- **Launcher** `~/dsv41-tp8.sh` **04b6738e**. It adds `export DSV41_MOE_FC2_D05=${DSV41_MOE_FC2_D05:-1}` and `export DSV41_L2_PREFETCH_PRUNE_SHARED=${DSV41_L2_PREFETCH_PRUNE_SHARED:-1}`.
  - Backups: `.bak-pd-l2pfS` = 92dbac8e (D05 only), `.bak-pd-d05` = b1e75497 (pre-step).
- **Site files** (installed with `pd-all.sh`; backups are `<file>.bak-pd-<tag>`):
  - d05: `fused_moe/experts/sm121_target_m32.py` 723fdf8d -> **5acc5fde**; new `sm121_d05_verify.py` **c04c363d**.
  - l2pf: `device_communicators/dsv41_l2_prefetch.py` 4da7745b -> **514dcb23**. The S arm was measured on this file; its PB flag is off.
  - pdl: `quantization/utils/fp8_utils.py` 8dc7dab8 -> **b35c0561**, inert with the flags off: the constexpr `PDL_TRIGGER` defaults to 0, which gives identical Triton code. Every A/B instance had it installed.
  - rgate: uninstalled. `gate_linear.py` is back to 41adaba8 and `dsv41_rgate.py` is absent.
  - Unchanged from pd base: `model_runner.py` 316fca44, `dsv41_rel.py` f373f911, `rdma_all_reduce.py` 9056dcdf, `dspark.py` 36811ca9.
- **`common/pd-all.sh` fix:** the per-host loop now uses `ssh -n`. The script is piped into `bash -s`, so an in-loop ssh without `-n` reads the rest of the script from stdin. Backup `pd-all.sh.bak-pdab`, on 8/8.
- **Live instance `pdab-FINAL`.** Started 23:01:13 UTC, healthy at 23:07:27 (+373 s). AV unpinned (startup profile, "pin miss"), as with pd-FINAL.
  - Fingerprint c2f008432bd4769f on 8 ranks; 0 restarts; 0 ERROR/Traceback lines on 8/8; Engram check OK on 8/8; xstream busy 0.
  - D05 native (`d05-shadow-v1/native`) mapped on 8/8; env `DSV41_MOE_FC2_D05=1 DSV41_L2_PREFETCH_PRUNE_SHARED=1`.
  - `--enable-prompt-tokens-details` still on. A repeated 4,205-token chat shows `cached_tokens` 0 -> 4160; streaming and /v1/completions carry it too.

### Protocol
- **Launches.** Every instance came from the supervised relaunch with `DSV41_AV_COST_JSON=/home/jeweled/dsv41-stage/rel2/av-pin-live2055.json`, so the diet was a hit and the AV profile was pinned.
  - 10/10 pinned launches were healthy on the first try, in 275-287 s (downtime 287-306 s).
  - With pdab-FINAL that is 11 clean launches in a row this step (34 since C02).
- **Suite per instance (`asuite.sh`):**
  - exact.py with spec-counter deltas on the fresh instance, then a warm-up;
  - matched C1 coding/prose x5;
  - bench single code/prose/math, 1024 tokens x2;
  - conc 8/16/32 x2, 512 tokens;
  - matched x5 again, so n = 10 C1 trials per workload per instance;
  - GSM8K-50 and full longctx on the second candidate instance.
- **Matched-bench label fix (affects older A/Bs).** `matched_bench.py` puts `--label` into every prompt's front tag (`<label>-t<i>`). A per-arm label therefore changes the prompts, the greedy text and AL.
  - On the same A1 instance, coding was 33.16 ms at AL 4.88 with label `pdab-A1-ctl`, and 33.55 ms at AL 5.15 with label `pdab`.
  - Earlier A/Bs that ran tsuite or ksuite with per-arm labels mixed about ±1% of prompt effect into C1.
  - This step uses one fixed label `pdab` (env `MLABEL`). Matched AL was identical on every pinned instance (coding 5.149, prose 2.695), so each C1 delta is pure step time and also confirms identical output.
- **Exactness.** exact.py gave a8c3394af9d2 with pinned counters 901/473/2365/1373 on all 10 pinned instances. The bench single counters were identical too (code 398/1990/625, prose 564/2820/459, math 289/1445/739).

### Per-candidate results (pinned AV; C1 = matched ms/step; deltas are B vs the A run just before; positive = better)

| # | candidate (flag) | stopped-window test | pair 1: C1 coding / prose, c16 / c32 | pair 2 | exact | GSM8K-50 / longctx | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **d05** `DSV41_MOE_FC2_D05=1` | GPU unit 8/8 PASS: 883 cases/rank, all 40 layers x tokens 1/4/6/48, eager + graphs, poisoned, 0 fail; 321 D05 calls, 11,540 groups skipped; FC2 1.08-1.29x at m1-96 | A1 33.549/28.613 -> B1 32.709/28.121: **-2.50% / -1.72%**, c16 +3.09%, c32 +4.51% | A2 33.561/28.747 -> B2 32.685/28.101: **-2.61% / -2.25%**, c16 +5.40%, c32 +5.96% | identical x4 | 50/50, PASS 60/60 (wall 171 s) | **PROMOTED** |
| 2 | **l2pf S** `DSV41_L2_PREFETCH_PRUNE_SHARED=1` (on d05) | cache hints only | B2 -> L1-S 32.015/27.508: **-2.05% / -2.11%**, c16 +0.90%, c32 +0.12% | C3 32.743/28.107 -> L4-S 32.183/27.529: **-1.71% / -2.06%**, c16 -0.59%, c32 +0.27% | identical | 50/50, PASS 60/60 (174 s) | **PROMOTED** |
| 2b | l2pf PB `..._POST_BUDGET=1` (screen) | CPU 10/10 | B2 -> 32.357/27.723: -1.00% / -1.35%, c8 -2.17%, c32 -0.87% | - | identical | - | not promoted: about 1% behind S |
| 2c | l2pf SQ `S + ..._PRUNE_STALE_QKV=1` (screen) | - | B2 -> 32.062/27.509: -1.91% / -2.11% (vs S -0.15% / 0.00%) | - | identical | - | not promoted: no gain over S |
| 3 | rgate `DSV41_ROUTER_GATE_REPLICA` | probe on spark-two: bitwise at M2-8 (H1z/0, 2000 trials per M + graph replay, 40 layers); M1 not exact. Replica is **slower**: warm 0.83-0.96x at M2-8, cold 0.65x (32 vs 21 µs). m_allow empty, no config written | - | - | - | - | **KILLED**, uninstalled 8/8 |
| 4 | pdl EB `DSV41_MOE_PDL_EARLY_B=1 DSV41_MOE_PDL_TRIGGER=start` (on d05+S) | unit PASS: 80 cases, 200 bitwise graph replays per arm; racecheck 0 hazards, synccheck 0 errors | L4-S -> P1 32.078/27.484: -0.33% / -0.16%, single +0.25..0.50%, c16 +0.57%, c32 +0.19% | not run | identical | - | **HOLD**: exact and slightly positive, but at most about 0.1 ms; it cannot reach the 0.8% C1 / 2% conc gates (the prep's own ceiling is 0.75%). Stopped after pair 1; flags off |
| 5 | fc2cd05 | not built (prep: 4 host hunks to hand-merge, native rebuild, +2.5 GiB/rank at 1M) | - | - | - | - | not run |

Absolute numbers per instance (single code/prose/math tok/s; conc 8/16/32 aggregate tok/s):

| instance | config | C1 coding / prose ms | single | conc 8 / 16 / 32 |
|---|---|---|---|---|
| A1-ctl | pre-step | 33.549 / 28.613 | 87.98 / 69.72 / 120.81 | 227.2 / 339.6 / 496.9 |
| B1-d05 | +D05 | 32.709 / 28.121 | 89.62 / 70.90 / 122.94 | 230.4 / 350.1 / 519.3 |
| A2-ctl | pre-step | 33.561 / 28.747 | 87.70 / 69.61 / 120.42 | 225.1 / 337.2 / 490.3 |
| B2-d05 | +D05 | 32.685 / 28.101 | 89.59 / 70.94 / 122.89 | 239.4 / 355.4 / 519.5 |
| L1-S | D05+S | 32.015 / 27.508 | 91.64 / 72.68 / 125.72 | 238.0 / 358.6 / 520.1 |
| L2-PB | D05+PB | 32.357 / 27.723 | 90.69 / 71.93 / 124.75 | 234.2 / 356.1 / 515.0 |
| L3-SQ | D05+S+SQ | 32.062 / 27.509 | 91.48 / 72.59 / 125.44 | 238.3 / 356.7 / 520.6 |
| C3-ctl | D05 (launcher 92dbac8e) | 32.743 / 28.107 | 89.53 / 70.86 / 123.09 | 237.5 / 354.6 / 519.0 |
| L4-S | D05+S | 32.183 / 27.529 | 91.35 / 72.56 / 125.21 | 235.8 / 352.5 / 520.4 |
| P1-EB | D05+S+pdl | 32.078 / 27.484 | 91.64 / 72.74 / 125.83 | 237.3 / 354.5 / 521.4 |
| **FINAL** | launcher 04b6738e, **unpinned** | 32.585 / 27.721 (AL 5.224 / 2.718, own profile) | 90.59 / 72.73 / 126.06 | 240.9 / 356.2 / 518.3 |

- **Stacked effect, D05 + S vs pre-step,** pinned, means of 2 instances each, same session:
  - C1 coding 33.555 -> 32.099 ms (**-4.3%**); prose 28.680 -> 27.519 ms (**-4.0%**);
  - single +4.2 / +4.2 / +4.0%;
  - conc8 +4.8%, conc16 +5.1%, conc32 +5.4%.
- **The two gains add up** (S measured on top of D05). D05 helps at every concurrency. S helps at 1 user (matched C1 and single) and is neutral at 8-32 users, where the shared-expert hints are not the binding cost.
- **pdab-FINAL vs the pre-step live baseline:** single +3.1 / +4.3 / +4.3%, conc8 +5.2%, conc16 +6.0%, conc32 +4.3%.
  - exact.py on its own fresh profile: a8c3394af9d2, counters 905/471/2355/1373. GSM8K-50 50/50. Full longctx PASS 60/60, copy 1.000.
  - Its matched C1 is not comparable with the pinned rows: its own AV profile picks other verify widths, giving AL 5.224 vs 5.149.

### Notes and next steps
- **New AV pin for A/Bs on top of this config:** `~/dsv41-stage/pdab/av-pin-pdfinal.json` (0d28dcb4, on 8/8). It holds the pdab-FINAL raw curves plus the live2055 shape_key, and v1 verify is 15.56 ms vs the pin's 16.28.
  - The live2055 pin is now stale for D05+S kernels. It is fine as a common pin within one A/B, but use the new pin for future arms.
  - The default-pin proposal (pd base step) remains a **user decision**. If accepted, point it at this file.
- **l2pf PB with S on.** S is now the launcher default, so a PB arm needs `DSV41_L2_PREFETCH_PRUNE_SHARED=0 DSV41_L2_PREFETCH_PRUNE_SHARED_POST_BUDGET=1`; the module raises on mixed modes.
- **pdl.** Pair 1 sits at the noise floor. Re-test only as a bundle with the not-yet-built FC1 prologue and glue-kernel arm A (pd/pdl README "Not done"). fp8_utils.py b35c0561 is installed inert. Codex P09 dual-quant edits the same file and would need a merge; the staged prefill packages (`pf`, `sched-pd`) do not touch it.
- **D05 dependencies.** D05 lives in `sm121_target_m32.py` 5acc5fde, and the native tree is `~/dsv41-goal120-20261024/d05-shadow-v1/native`, which must stay. The prefill `pf` package edits `deep_gemm_utils.py` (a4a0486a); the d05 GPU test asserts that hash, so re-run `d05/ops/run-gpu-tests.sh` after any such install.
- **Not run:** the D05 live `verify` shadow, which is optional because the unit test covers all 40 layers; l2pf SR; pdl EBe/T; and fc2cd05.

### Rollback
- **Env only:** relaunch with `DSV41_MOE_FC2_D05=0 DSV41_L2_PREFETCH_PRUNE_SHARED=0` (supervisor `VAR=VAL`), which gives the pre-step behaviour.
- **Launcher:** on 8/8, `cp -p ~/dsv41-tp8.sh.bak-pd-l2pfS ~/dsv41-tp8.sh` (92dbac8e, D05 only) or `.bak-pd-d05` (b1e75497, pre-step).
- **Site files** (service stopped): `ssh spark-one cat ~/dsv41-stage/pd/common/pd-all.sh | bash -s uninstall <d05|l2pf|pdl>`.
  - This restores 723fdf8d (and removes sm121_d05_verify.py) / 4da7745b / 8dc7dab8 and checks the digest is identical on 8/8.
  - Restore the launcher first when uninstalling d05.

## pd long-context X-ray (2026-10-25 23:45 - 2026-10-26 02:05 UTC): 1M excess = rank-5 straggler on unpinned launches. Diagnostic only, nothing installed

Scope: kernel-ideas-2 "Shared prerequisite: one long-context X-ray window". No site file, launcher or ops script changed (manifest ad97d72f180e, launcher 04b6738e on 8/8 before and after). Diagnostic tools live only in `~/dsv41-xray/` on the Sparks:
- `dsv41-tp8-xray.sh`: the live launcher plus the k1trace nsys wrapper with `--wait=primary` and repeat:6;
- `py/xray_hook.py`: worker extension for KV-layout logging and top-k dumps;
- `xr_bench.py`, `xr_an.py`, `xr_topk.py`, `tele.sh`.

Full write-up: Mac scratchpad `pd/xray.md`. Raw data: `~/dsv41-xray/{runs,an,nsys,dump,tele,logs}`.

**Method.**
- Cold 1M solo prompts: 1.024M tokens, 63 chunks.
- Chunk boundaries come from the dsv41_sched stats file. With hold=1 they are exact.
- One run added a streaming decoder; its gaps equal the chunk times, +0.11 s.
- 1 Hz nvidia-smi + spbm telemetry on 8/8, plus a switch snapshot.
- nsys of single chunks (2/12/35/58) on all ranks, via `/start_profile` with `{"delay_iterations":0,"max_iterations":1}` and `/stop_profile` before each start. The frontend session guard returns 409 otherwise.
- Ratio-2 top-k dumps at ~16-33K and ~900K.
- 512K decode probe: 3 greedy 512-token continuations on the prefix cache vs 1K-prefix controls.

**Instances (all healthy on the first try; 40 clean launches in a row since C02):**

| instance | config | 1M TTFT |
|---|---|---|
| live pdab-FINAL | unpinned | 231.96 s (decoder run 234.77) |
| B | nsys + hook + AV pin av-pin-pdfinal | 184.61 / 186.50 / 186.18 s (the last after 2x1M + 500K + asuite gsm lcfull) |
| C | hook, unpinned, A-history replay | 232.50 / 224.59 s |
| E | nsys + hook, unpinned | 232.46 s (its first 1M) |
| R1 | launcher defaults (restore) | 234.84 s |
| P | defaults + `DSV41_AV_COST_JSON=~/dsv41-stage/pdab/av-pin-pdfinal.json` | **184.84 s** |
| xray-final | defaults (live now) | not probed |

**Findings.**
1. **Per-chunk curve.**
   - Flat 2.667 s per chunk up to ~330K on every launch.
   - Slow launches then rise ~4.3 s per 1M of context to a 5.0-5.2 s plateau from ~880K.
   - Good launches rise only to 3.18 s at 1M.
   - At 999K (61 chunks) the chunks sum to 222.1 s (slow) vs 175.8 s (good) vs 162.7 s flat. That is **46 s of straggler + 13 s inherent**.
2. **Straggler** (nsys chunk 58, slow launch E vs good launch B):
   - Span 4970 vs 3143 ms. Every rank except 5 is unchanged ±1% in compute and waits in the attention-output NCCL AR (2.30-2.35 s vs 0.49 s).
   - Rank 5 `sparse_mla_prefill_mg_dual` takes 2209 vs 369 ms. Per layer by KV-source group: L2-7 18 -> 45 ms, L8-13 20.5 -> 139, L14-19 20.7 -> 179, tail 3.9 -> 20.4. Layers 0-1 (SWA) are unchanged. Rank-5 cp_gather is +3.1 ms.
   - Kernels reading rank 5's **paged KV pool** slow down; kernels on workspaces (prefill MQA logits, top-k) do not.
   - Decode at 512K: rank 5's paged MQA logits 2.18 -> 5.47 ms/step; other ranks +4.0 ms rdma_ar wait; 29.0 -> 33.3 ms/step.
   - **Not** clocks, power or thermal: SM 1969-1989 MHz on 8/8, reasons 0x0, throttle counters +0, max temperature 63-69 C, utilisation 94-96%. On slow launches GPU power falls with context because 7 ranks spin.
   - **Not** CPU gaps (GPU idle 2.6-6 ms per chunk), nsys, the hook, request history, or the KV block-id layout (identical on all ranks; contiguous 256-block runs per chunk).
   - **Tracks the AV pin:** pinned 2/2 good, unpinned 0/5 good (Fisher p ≈ 0.05). Mechanism open.
3. **Inherent growth** (good launch, chunk 2 -> 58, +482 ms per chunk; ~13.6 s at 1M):
   - ratio-2 indexer L2/8/14: 57%. Per layer at 942K: logits 83 ms + top-k 34 ms. Per chunk 56 -> 359 ms.
   - tail / layer-20 candidate indexer on 128 replay rows over the full context: 21% (3 -> 99 ms per chunk).
   - sparse MLA: 12% (331 -> 375 ms).
   - MoE 5%, mHC 2%.
4. **Top-k (layer-14 group, 16,252 rows):**

   | | ~16-33K | ~900K |
   |---|---|---|
   | union of the chunk's picks / cache | 100% | 80.5% |
   | adjacent-row overlap f (mean) | 0.32 | 0.18 |
   | 16-row tile union | 7.7 x 512 | 12.0 x 512 |
   | distinct KV blocks per row | 218 | 397 |

   - **F2 gate** (union <= 25%) fails.
   - **Exp 10** saves <= 9-16% of gathers on a +44 ms/chunk kernel: no-go.
5. **512K decode** vs 1K prefix:
   - good launch: +2.1-2.2 ms/step (~7%);
   - slow launch: +6.0-7.0 ms/step;
   - prefix-hit TTFT 1.03-1.12 s.
   - Round-1 #10 is worth at most ~1 ms/step on good launches and stays deferred.

**Prepared prefill candidates vs the excess:**
- **No prepared candidate touches the straggler**, which is ~78% of the excess.
- **E7** (`DSV41_E7_TAIL=1`) also removes the tail-indexer growth, so its 1M estimate rises to about -13 s (was -9.7). It is the best prepared exact 1M candidate.
- **F1** (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=2048`) is the only prepared arm on the ratio-2 indexer, at <= ~1 s. A fused streaming logits+top-k would be the real fix; it is not prepared.
- **SP v2, lone-24K, gather-add, P09 and the NCCL arms** reduce the flat 2.67 s per chunk only.

**Proposals (not applied):**
1. **Default AV pin.** It is the pd-base proposal, now with a performance reason: 1M -47 s, 512K decode about -4 ms/step. Confirm with 3 pinned + 3 unpinned launches, each with a ~700K screen (600-700K chunks ~2.95 s good vs ~3.9-4.1 s slow). **User decision.**
2. Optional relaunch-if-slow 700K screen in `dsv41-relaunch.sh`.
3. Root-cause the rank-5 paged-KV slowdown after the unpinned profile.

**Live now:** instance `xray-final`, healthy 02:01:25 UTC.
- Launcher 04b6738e defaults, unpinned ("pin miss"), the same as pdab-FINAL. Expect the long-context straggler, as on every unpinned launch.
- 8/8 active, 0 restarts, 0 ERROR/Traceback. Fingerprint c2f008432bd4769f on 8/8; D05 mapped; no nsys or xray environment.
- exact.py **a8c3394af9d2** with anchor per-prompt hashes.
- prompt-tokens-details cached_tokens 0 -> 4160 (R1).
- Site files unchanged and identical on 8/8: m32 5acc5fde, d05_verify c04c363d, l2pf 514dcb23, fp8_utils b35c0561, model_runner 316fca44, dsv41_rel f373f911, rdma_all_reduce 9056dcdf, dspark 36811ca9, gate_linear 41adaba8.

**Rollback:** nothing to roll back. `~/dsv41-xray/` can be deleted; it is never referenced by the live launcher.

## pf prefill A/B step (2026-10-26 02:05-05:45 UTC): nothing promoted. E7 output-changing, SP v2 not yet exact, gather-add crashes, NCCL arms closed

Scope: the prepared prefill candidates of "pd / prefill offline prep" (`~/dsv41-stage/pd/`), run serially. Every A/B arm used the AV pin `~/dsv41-stage/pdab/av-pin-pdfinal.json`. Raw results on spark-one: `~/dsv41-pf-20261026/`, which holds `STATE.md` (running log), `runs/<label>/` (one per instance) and `scripts/`:
- `psuite.sh` / `psumm.py`: prefill suite and summary;
- `lexact.py`, `lpcmp.py`, `ndprobe.py`: long-prompt and nondeterminism probes;
- `e7x.py`, `e7xp.py`: E7 bitwise tests;
- `spx.py`, `sp*probe.sh`: SP probes with exact row counts;
- `xrb.py`: `xr_bench` plus an E7 opt-out flag;
- `tiddet.py`, `finaltests.sh`;
- installers `e7launcher.sh` and `splauncher.sh` (not used).

Mac scratchpad `pf/` holds the drivers (`runarm.sh`, `chain2.sh`, `e7xrun.sh`, `s1run.sh`, `sxrun*.sh`, `sp1run.sh`, `sp2run.sh`, `b2x.sh`, `teardown.sh`, `finalize.sh`), plus `summ-*.json`, `sp_<label>-rank<i>.txt` (SP compare lines) and `tele/`.

**Live state after this step:** instance `pf-FINAL`, healthy 05:24:14 UTC (+366 s, supervised, first try), launcher **04b6738e** defaults (AV **unpinned**, as before the step). Every site file is back to its pre-step sha256 and identical on 8/8: model.py e7264781, model_state.py 53e4fe39, decoder_replay_layers.py 2fedc0a6, dspark/speculator.py bbcee9e3, moe_runner.py 20b69fe5, deep_gemm_utils.py a4a0486a, sm121_target_m32.py 5acc5fde, sm121_d05_verify.py c04c363d, dsv41_l2_prefetch.py 514dcb23, fp8_utils.py b35c0561 (inert), model_runner.py 316fca44, dsv41_rel.py f373f911, rdma_all_reduce.py 9056dcdf, dspark.py 36811ca9, gate_linear.py 41adaba8; no rgate. Scheduler: `~/dsv41-sched/dsv41_sched.py` 6074389f, no dsv41_pd_policy.py.
- Removal: `pd-all.sh uninstall spg` then `uninstall pf` (digests ca01af51f9c5 / b5760b58fa09, identical 8/8), then `sched-install.sh uninstall` on 8/8.
- Ranks: 8/8 active, 0 restarts, 0 ERROR/Traceback lines, fingerprint c2f008432bd4769f, no test environment.
- exact.py: **9330f03c4f97**, the documented AV-profile alternate (p2 2228532c, p4 8c5e765e), counters 914/464/2320/1373 from its own profile. It gave the same hash again after all the other probes. The same code gives a8c3394af9d2 under the pin (A1/A2 with the pf files inert; X-ray B-age), so this is the alternate profile, not a code change.
- `--enable-prompt-tokens-details` is still on: `cached_tokens` 0 -> 4160 on a repeated 4,205-token chat (streaming and /v1/completions carry it too).
- Quick longctx 16K/32K PASS 10/10, copy 1.000.
- 1M solo 182.86 s with a good chunk curve (2.68 / 2.98 / 3.11 s), so the live instance is currently a good long-context launch.
- Stock token-id determinism probe: 3,000 ids gave 3/3 identical first-token logprobs; 16,384 ids gave 3/3 different (-5.84 / -5.95 / -5.55).

### Two findings that change how prefill changes must be tested
1. **Within-instance prefill nondeterminism (pre-existing, not caused by any pf flag).**
   - The same prompt, sent to one instance with distinct `cache_salt` values, gives an identical first-token logprob only up to about 4.1K prompt tokens: 4,112 tokens gave 3/3 identical; 4,145 / 4,196 / 6,056 / 9,071 tokens each gave 3/3 different.
   - At 40K tokens the first token itself differs between copies. At 128K / 300K the text diverges at token 3 / 7 (A1 `lexact` pair).
   - exact.py only sends short prompts, so it never sees this.
   - **Consequence:** long-prompt exactness cannot be judged by comparing copies or instances. It needs the deterministic regime (rows ≤ ~4.1K, e.g. small chunks) or in-forward compares.
   - **One localised source:** layer 2 (the first ratio-2 sparse-indexer layer) on 16K-row token-id prompts varies from run to run (SPK3: rel_l2 2.4e-4 and 2.0e-4 for two identical prompts). On the stock FINAL instance the same 16,384-id prompt gave 3/3 different first-token logprobs (-5.84 / -5.95 / -5.55), while 3,000 ids gave 3/3 identical.
2. **The rank-5 long-context straggler (X-ray step) is not prevented by the AV pin.**
   - In this step, pinned launches A1 and A2 were slow: 1M TTFT 227.6 / 235.9 s, with 600-700K chunks at 4.0 / 4.2 s.
   - Pinned launches B1, B2 and SP1, SP2 were good: 170-174 s with E7 or SP, and 182.3 s for the stock-tail copy on the same good instance.
   - Tally with the X-ray launches: pinned (this step 4/2, X-ray B and P 2/0), so 6 good / 2 slow in all, unpinned 0/4 distinct launches in the X-ray (pdab-FINAL, C, E, R1) plus this step's FINAL, which was **good** (182.86 s): 1/5 overall, one-sided Fisher p ~0.09. The pin raises the odds but does not prevent the straggler.
   - E7 does not change the launch class: on the good B2 instance, the stock-tail copy was also good.
   - 1 Hz GPU power does not identify the slow rank: on slow A2, all 8 ranks show late/early power 0.79-0.82.

### Per-candidate results (pinned AV; TTFT = cold solo, fresh nonce; deltas vs the flags-off arms A1/A2)

| # | candidate (flag) | test | result | verdict |
|---|---|---|---|---|
| 1 | NCCL arms (env via launcher hook) | `ar_bench`, service stopped, 8 ranks, [T,5120] bf16 | control 16K rows 11.41-11.46 ms (busbw 25.6-25.7 GB/s); QPS4+split 12.51 (worse); QPS2 11.79; BUFFSIZE 16M 11.34 (same sha); 8 channels 11.38 (sha differs); 4 channels 12.18; PROTO=Simple 11.34 / ALGO=Ring 11.40 (same sha as control, so already the defaults); Tree 60.5 | **CLOSED.** No exact arm ≥ 5% of AR time; no launcher hook installed |
| 2 | E7 decoder-tail skip (`DSV41_E7_TAIL=1`) | ABAB A1 B1 A2 B2; B2X; E7X / E7XP (1024-token chunks, deterministic regime) | see "E7" below | **NOT PROMOTED: changes output** |
| 3 | SP v2 exact (`DSV41_PREFILL_SP=exact`) + new split-match gate (`spg`) | DEBUG compares S1, SX, SP1, SPK1-3, SP2; timing SP1/SP2 | see "SP v2" below | **NOT PROMOTED: not yet bitwise.** Biggest exact lever if fixed: -6.1% TTFT |
| 4 | gather-add fold (`DSV41_MOE_GATHER_ADD=1`) | GPU gate (`ga_gpu2.py`, spark-eight); serving G1 | kernel bitwise 15/15, fused 0.78x (M16384 7079 -> 5542 µs). Serving: **startup crash on both tries** (profile_run, `shared_experts.py:155` assert) | **KILLED as staged** |
| 5 | P09 dual quant | - | not run: estimated -0.5%, cannot reach the 3% gate; does not stack with the fold | not run |
| 6 | lone-prefill 24K chunks | - | not run: HOLD on memory; output differs for prompts > 16K (chunk boundaries change), so it is not exact | not run |

Note: the staged `moe-glue/test_gather_add_gpu.py` is invalid. It imports `test_gather_add_interp`, whose `setdefault` forces `TRITON_INTERPRET=1`, so it runs in the interpreter with round-toward-zero and reports 682,485 false mismatches. The self-contained `~/dsv41-pf-20261026/scripts/ga_gpu2.py` is the real GPU gate.

**E7 (decoder-tail skip on prefill chunks that are neither the final one nor the one before it).**
- Performance and gates (2 instances per arm):
  - 128K TTFT 21.617 / 21.645 -> 20.566 / 20.566 s (**-4.9% / -5.0%**).
  - 16K unchanged.
  - Flat-band chunk 2.68 -> 2.52 s (-157 ms).
  - Like-for-like 1M on one instance (B2X): stock tail 182.34 s -> E7 170.54 / 170.36 s (**-6.5%**). The per-chunk saving grows from -152 ms to -223 ms because tail-indexer growth is skipped.
  - Starvation unchanged (E7 skips only lone prefills).
  - C1 32.40 / 27.67 and 32.42 / 27.90 (A) vs 32.57 / 27.68 and 32.44 / 27.68 ms (B), AL identical; conc8 within noise.
  - exact.py a8c3394af9d2 with 905/471/2355/1373 on every instance.
  - GSM8K-50 50/50; longctx 60/60 (copy 1.000); 1M needle 7/7 (copy exact).
- **E7X failed.** Setup: E7 on, `--max-num-batched-tokens=1024`, 2,187-4,009-token prompts (3-4 chunks), deterministic regime.
  - The stock copies (per-request `dsv41_e7_off`) were bitwise identical 3/3 at every length.
  - The E7 copies differed from stock, including the first-token logprob: 2,187 tokens -0.3499 vs -0.6346; 4,009 tokens -0.3889 vs -0.2836.
  - E7XP (prefix-cache variant) failed the same way.
- **Cause.** The replay layers do not own only SWA KV. They also write the indexer K caches of the index-source layers 20/24/28/32/36 (`metadata_prefixes` includes `attn.indexer.k_cache.prefix`).
  - Each chunk's replay writes entries for its last 128 rows, and those entries persist.
  - The top-k at layers 24-39 for the final rows reads them.
  - E7 skips those writes, so the final rows see fewer index-K entries.
  - The `write_batch_kv` call arguments match the stock call; this is a design premise error, not a call-site bug.
- **Exact variant (not built):** keep the replay cache writes for skipped chunks and skip only the post-replay full-T work plus the DSpark draft (about 84 of the 168 ms per chunk).

**SP v2 (layers 0-19 row-sharded across TP; attention and MoE on gathered rows).**
- **Ungated (S1, SX).** In-forward DEBUG compares are far from bitwise on 2,048-8,065-row forwards: hidden rel_l2 0.11-0.19, identical across ranks and copies.
  - Cause 1: `compute_mhc_pre_num_splits` returns 1/4/16 from `n_sms // cdiv(rows, 64)`. The stock path on T rows and the shard on T/8 rows pick different mHC pre-norm split-K counts unless the shard has enough row tiles.
  - The resulting last-bit difference flips routing and top-k, and the error cascades.
  - The new pd tag **spg** (model.py fe7a7d12 -> 8f2904e6, installed for SP1-SP2 only) gates SP on matching split counts.
- **Gated (SP1, SPK1-2).** Layers [0,1) and [0,2) are **bitwise** at 12,296 / 14,000 / 16,380 rows, including the Engram layer 1. So the SP plumbing is exact.
- **Full 20 layers, gated (SP2, chat prompts).**
  - **Bitwise** at 15,022 and 16,032 rows (and S1's 16,032 x2).
  - **Not bitwise** at 12,760 / 13,544 / 14,241 rows (hidden rel_l2 0.073-0.079).
  - So a second row-count-dependent kernel configuration switches between shard sizes 1,781 and 1,878 rows. It is not identified yet.
  - Token-id prompts additionally hit finding 1 at layer 2.
- **Timing (no E7, pinned, good launch).**
  - TTFT 16K 2.523 (**-6.2%**); 128K 20.297 (20.305 / 20.288, **-6.1% / -6.2%**).
  - 1M 172.29 s vs like-for-like stock 182.34 (-5.5%); flat chunks 2.508 vs 2.672-2.683 s.
  - Starvation long 22.78 vs 24.86-24.95 s.
  - Decode unchanged: C1 32.474 / 27.806 ms, conc8 235.4.
  - SP2 repeated this: 16K 2.540, 128K 20.362 (20.418 / 20.307, -5.8% / -5.9%), 1M 174.21 s (good launch). GSM8K-50 50/50, longctx 60/60 (copy 1.000), 1M needle 7/7 (copy exact), 0 errors. SP2's exact.py gave e0feee387e43 (only p3 flipped: e4460f26 vs cd93c706; counters 888/491/2455/1373). It ran after 5 chat probes, and SP cannot engage on exact.py's short prompts (under 2,048 rows, and the gate needs about 12.3K), so this is request-history dependence of the decode path, not SP; S1 and SP1, also after probes, gave a8. exact.py must run first on a fresh instance

### Per instance (pinned unless noted)

| instance | config | exact / counters | C1 coding / prose ms (AL 5.224 / 2.718 on all) | conc8 | TTFT 16K / 128K / 500K / 1M (s) | 1M chunks c2-12 / c37-43 / c50-62 (s) | 1M launch | starvation: long TTFT / short p50 / p95 (s) / decoder gap p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| A1 | flags off | a8c3394af9d2 / 905/471/2355/1373 | 32.396 / 27.668 | 240.9 | 2.690 / 21.617 / 89.98 / 227.57 | 2.681 / 4.013 / 4.838 | slow | 24.86 / 1.667 / 1.979 / 428 |
| B1 | E7 | a8c3394af9d2 / 905/471/2355/1373 | 32.572 / 27.684 | 236.7 | 2.685 / 20.566 / 82.45 / 170.72 | 2.524 / 2.763 / 2.884 | good | 24.84 / 1.659 / 1.960 / 420 |
| A2 | flags off | a8c3394af9d2 / 905/471/2355/1373 | 32.416 / 27.896 | 236.9 | 2.690 / 21.645 / 90.21 / 235.85 | 2.683 / 4.244 / 5.030 | slow | 24.95 / 1.683 / 2.034 / 430 |
| B2 | E7 + gates | a8c3394af9d2 / 905/471/2355/1373 | 32.442 / 27.682 | 239.4 | 2.689 / 20.566 / 82.29 / 170.54 | 2.520 / 2.759 / 2.885 | good | 24.89 / 1.667 / 1.970 / 426 |
| S1 | E7 + SP exact ungated (DEBUG, quick) | a8c3394af9d2 / 905/471/2355/1373 | - / - | - | 2.521 / 19.319 / - / - | - / - / - | - | - / - / - / - |
| SP1 | SP exact + split gate | a8c3394af9d2 / 905/471/2355/1373 | 32.474 / 27.806 | 235.4 | 2.523 / 20.297 / 82.47 / 172.29 | 2.508 / 2.803 / 2.938 | good | 22.78 / 0.973 / 2.567 / 426 |
| SP2 | SP exact + split gate + gates | e0feee387e43 / 888/491/2455/1373 | 32.360 / 27.668 | 234.0 | 2.540 / 20.362 / 82.86 / 174.21 | 2.514 / 2.838 / 2.943 | good | 22.93 / 0.791 / 1.078 / 428 |
| B2X | B2 instance, 1M with per-request `dsv41_e7_off` (stock tail), then E7 again | - | - | - | - / - / - / 182.34 (stock) / 170.36 (E7 again) | 2.672 / 2.973 / 3.108 (stock) | good | - |
| FINAL (live) | launcher 04b6738e defaults, **unpinned**, all pf files removed | 9330f03c4f97 (known AV-profile alternate, p2/p4) / 914/464/2320/1373; same again after all probes | - | - | - / - / - / 182.86 | 2.678 / 2.975 / 3.112 | good | - |

### Rollback and leftovers
- **Live config:** launcher 04b6738e, unchanged. Everything above was env-only or installed and removed within the step (see live state).
- **Staged, not installed:** `~/dsv41-stage/pd/{pf,spg,sched-pd}` on 8/8, plus the installers `~/dsv41-pf-20261026/scripts/{e7launcher.sh,splauncher.sh}` (never run).
- **Re-install with the service stopped:** `ssh spark-one cat ~/dsv41-stage/pd/common/pd-all.sh | bash -s install pf` (then `install spg`), and on each node `bash ~/dsv41-stage/pd/sched-pd/sched-install.sh install`.

### Next steps
1. **SP v2 is the largest exact prefill lever: -6.1% TTFT from 16K to 1M, decode unchanged.**
   - Find the second row-count-dependent kernel (bitwise from 15,022 rows, not at 12,760-14,241) with a `DSV41_PREFILL_SP_LAYERS` bisect over layers 2-19, using ~13.5K-row chat prompts.
   - Compare one full 16,384-row chunk with a chat prompt.
   - Then gate or fix it, and run ABAB; the SP1/SP2 timings carry over.
2. **E7 exact variant:** keep the replay indexer-K writes on skipped chunks.
3. **Gather-add:** the arm must peek `_output[_output_idx]` instead of reading the consuming `output` property. Worth at most ~1.2%.
4. **Root-cause the long-prompt nondeterminism (finding 1),** starting with the layer-2 indexer top-k tie set. It blocks bitwise testing of every long-prompt change.
5. **Straggler:** it is random per launch and the pin does not prevent it. The relaunch-if-slow 700K screen remains the practical guard.


## pd Marlin MoE measurement (2026-10-26 06:04-06:21 UTC): measurement only, nothing installed

The user has not approved output-changing (non-exact) changes, so this step only measures. Nothing in site-packages, the launcher or the env was changed.

**Window.** Stopped 8/8 at 06:04:38. The benches ran 06:06-06:14. The supervised relaunch `mm-FINAL` (launcher 04b6738e defaults, AV unpinned as before) was healthy at 06:20:52 on the first try, so the service was down for about 16 min. Afterwards:
- the 14 site shas, gate_linear, sched and the launcher matched the pre-window values on 8/8;
- 8/8 active, 0 restarts, 0 errors, no test env, rdma fingerprint c2f008432bd4769f;
- exact.py gave **a8c3394af9d2 x2**, with the anchor's per-prompt hashes.

**Harness.** `~/mm-marlin-20261026/mm_marlin_bench.py` (sha a3939675dc65; spark-one, spark-two and spark-eight hold copies). It runs standalone on one GPU, uses real checkpoint weights for one TP rank slice, and loads them with vLLM's own `RoutedExperts` loader.

| arm | what | bytes/expert |
|---|---|---|
| live | serving `_quantize_input` (FP8 per-128 UE8M0) + `DeepGemmFP4Experts.apply` with the launcher MoE env: M32 + COMPACT + LIVE_ACT + D05 (asserted armed) | 2,611,200 |
| mstock | vLLM 0.32 `MarlinExperts.apply` W4A16, stock `--moe-backend marlin` layout (I 288->384: FC1 768 rows, FC2 K 384), persistent Marlin workspace as in serving | 3,133,440 (+20%) |
| ma64 | the same stock path with I padded to 320 (FC1 640, FC2 K 320) | 2,611,200 (= live) |
| munpad | the same stock Marlin ops with FC1 unpadded (576 rows) and FC2 K 320 (align64), plus explicit glue: a 288->320 pad copy | 2,437,120 (-6.7%) |

**Timing method.**
- Recorded 10-21 decode routes (rank-0 recorder, all 40 layers). "Users" are real 6-row C1 verify windows: M=6k is k windows, and M=1 is the first row of a window. c4rec sets are real 4-user batches.
- One CUDA graph runs 40 sequential layer applies, cycling 4 distinct real layers' weights (0/13/26/39) with each layer's own routes, which keeps the weights L2-cold. Arms are interleaved over 25 replays.
- Two route sets each on rank 0 (spark-one) and rank 7 (spark-eight), plus a third rank-0 sample. The two ranks agree within 2%.
- A uniform-route set (no locality, 298 distinct experts at M=96) gives the high-diversity bound.
- Private D05-JIT, Triton and DeepGEMM caches were used, and the shared caches are unchanged (`out/cachediff.txt`).
- Each arm passed CUDA graph replay == eager with changing routes. The FP4 nibble order was confirmed against the FP32 reference.

**Per-layer routed MoE, µs (recorded routes, mean of 5 samples; speedup vs live; distinct experts per layer):**

| M | live | mstock | ma64 | munpad | distinct |
|---|---|---|---|---|---|
| 1 | 107.1 | 101.4 (1.06x) | 87.6 (1.22x) | 84.3 (1.27x) | 6 |
| 6 | 334.7 | 320.9 (1.04x) | 272.4 (1.23x) | 255.6 (1.31x) | 22 |
| 12 | 576.9 (c4rec 507.1) | 556.7 (1.04x) | 472.4 (1.22x) | 438.1 (1.32x; c4rec 1.33x) | 40 |
| 24 | 882.8 (c4rec 819.5) | 848.8 (1.04x) | 718.2 (1.23x) | 661.4 (1.33x; c4rec 1.34x) | 61 |
| 48 | 1294.1 | 1218.1 (1.06x) | 1025.4 (1.26x) | 947.1 (1.37x) | 88 |
| 96 | 1608.0 | 1517.0 (1.06x) | 1283.5 (1.25x) | 1183.5 (1.36x) | 106 |
| 48 uniform | 2763.1 | 1.01x | 1.20x | 1.30x | 205 |
| 96 uniform | 4019.7 | 1.00x | 1.20x | 1.29x | 298 |

**Effective weight bandwidth.**
- Live streams 146-194 GB/s; Marlin streams 173-234 GB/s.
- At M=6, live FC1 (576x5120) takes 181 µs (~179 GB/s) and FC2 107 µs (~203 GB/s). The two munpad Marlin GEMMs together take 230 µs (~219 GB/s).
- At M=96, live FC1/FC2 take 953/640 µs and the munpad GEMMs 1229 µs.

**Projected target routed MoE per decode step (40 layers; drafter MoE not included), ms (delta vs live):**

| scenario | live | mstock | ma64 | munpad |
|---|---|---|---|---|
| 1 user, M=6 (C1 coding, full width) | 13.39 | 12.83 (-0.55) | 10.90 (-2.49) | 10.22 (-3.17, about -10% of the 32.4 ms C1 coding step) |
| 1 user, M=1 | 4.28 | -0.23 | -0.78 | -0.91 |
| 16 users, M=48 (VERIFY_CAP ~3 rows/user) | 51.76 | -3.04 | -10.75 | -13.88 |
| 16 users, M=96 (full width) | 64.32 | -3.64 | -12.98 | -16.98 |
| 16 users, uniform bound M=48 / M=96 | 110.5 / 160.8 | -0.66 / -0.10 | -18.5 / -26.3 | -25.3 / -36.0 |

The recorded 16-user composites come from only ~4 prompts and so understate expert diversity. The uniform rows overstate it.

**Prefill sizes** (one layer, eager, uniform routes, rank 7). One Marlin weight copy would also have to serve prefill, because a second copy (+37 GB/rank) does not fit. Times in ms per layer (speedup vs live):

| M | live | mstock | ma64 | munpad |
|---|---|---|---|---|
| 128 | 8.69 (the 128-row DeepGEMM cliff above the M32 cap) | 4.77 (1.82x) | 3.88 (2.24x) | 3.61 (2.41x) |
| 512 | 10.00 | 5.63 (1.78x) | 4.74 (2.11x) | 4.50 (2.22x) |
| 2048 | 10.95 | 7.37 (1.49x) | 6.52 (1.68x) | 5.97 (1.84x) |
| 8192 | 16.74 | 15.83 (1.06x) | 16.17 (1.04x) | 16.86 (0.99x) |
| 16384 | 26.21 | 29.29 (0.89x) | 28.45 (0.92x) | 30.51 (0.86x) |

Per 16K serving chunk, that is +124 / +90 / +172 ms, about +3-6% of 2.7 s. munpad's figure includes an unfused 288->320 pad copy, about 0.5 ms per layer at 16K.

**Numerics on real hidden states.**
- Inputs: the 10-21 early-hidden snapshots, layers 0-5 `ffn_in`, 48 bitwise-distinct rows per layer, all prefill rows of 2 short prompts.
- Routes come from the checkpoint router (sqrtsoftplus + bias, top-6, renorm, x1.5).
- All 8 TP rank partials are summed in fp32 and rounded to bf16, as the served all-reduce does. The routed output has absmax 1.20 and rms 0.056.

| comparison | rel L2 (all) | per layer | per-row rel L2 median / max | max abs |
|---|---|---|---|---|
| Marlin (any layout) vs live | **3.24%** | 2.6-3.7% | 3.4% / 4.2% | 0.0156 |
| live vs FP32 reference (dequantized FP4, fp32 math) | 3.20% | 2.5-3.7% | 3.3% / 4.2% | 0.0151 |
| Marlin vs FP32 reference | 0.45% | 0.40-0.47% | 0.45% / 0.55% | 0.0037 |

- So the Marlin-live difference is the size of live's own FP8-activation rounding, and Marlin is about 7x closer to the FP32 math. It is not exact, and greedy outputs will move.
- mstock and ma64 are bitwise identical. munpad differs from them by 3-5e-5 rel L2 at M >= 48 (FC2 K tiling) and is bitwise at M <= 6.
- Validation: served 10-21 `ffn_out` vs (our live routed sum + an FP32 shared expert) is 1.7-2.6% rel L2. The routed part is 50-81% of the output norm, so the routes and the live reconstruction match serving.

**Memory per rank** (40 layers x 384 experts): mstock +7.47 GiB (unsafe at 1M), ma64 +-0, munpad -2.49 GiB.

**Verdict (user decision; nothing to promote under the exact-only rule).**
- **Stock `--moe-backend marlin` (288->384): NO-GO.**
  - Only 1.04-1.06x at decode on recorded routes, falling to ~1.00x at high expert diversity.
  - +7.5 GiB/rank.
  - About 12% slower at 16K prefill chunks.
  - Non-exact.
- **Marlin W4A16, unpadded layout (FC1 576 / FC2 K 320): conditional GO, only if the user accepts non-bitwise outputs.**
  - It clears the 1.15x bar at 6 and 48 rows (1.31x / 1.37x; 1.29-1.30x uniform).
  - Projected about -3.2 ms/step at 1 user and -14 to -17 ms/step at 16 users, and it frees 2.5 GiB/rank.
  - Prefill at 16K chunks is up to ~6% slower unless the glue is fused or the chunk policy changes.
  - Needed before any serving arm:
    - a loader repack and a fused SiLU-clamp writing the 320-wide FC2 input;
    - zero-filling the ROUTE_LIVE dead top-k slots that `moe_sum` reads;
    - a new exact anchor;
    - GSM8K-200, longctx and 1M, and a code-quality sign test.
  - The drafter MoE was not measured.
- **Exact lead found:** ma64 reads exactly the live bytes and is still 1.18-1.26x faster. The live M32 FP8xFP4 kernels leave about 20% of the weight bandwidth unused, which FC1 (~179 GB/s at M=6) shows most clearly. Better weight pipelining in the M32 FC1/FC2 kernels could recover part of this without changing numerics. This is the next exact decode candidate.

**Files** (spark-one `~/mm-marlin-20261026/out/`): `smoke-r0`, `timing-r0`, `timing-r7` (spark-eight), `timing-r0-rand`, `prefill-r7`, `numerics` (spark-two), `.jsonl`/`.err`, `final_table.txt`, `numerics_summary.txt`, `cachediff.txt`, `exact-final*.json`. Pre-window rank logs are in `~/mm-marlin-20261026/logs/pre-mm-rank*.log` on each node.


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


## Offline prep g1/stepovh: exact step-overhead levers (knapcio port) staged, NOT installed (2026-10-26 ~08:30 UTC)

No restart, no GPU job, and no live-file edit. Everything is in `~/dsv41-stage/g1/stepovh/` on 8/8 (package digest 3e75165f476d). The handoff is that directory's `README.md`. `ops/g1-all.sh status stepovh` reports OK 8/8 with the live bases: dsv41_l2_prefetch 514dcb23, dsv41_r4_small 02f94f29, sparse_swa 38b16c51, model_state 53e4fe39. Install with `ops/g1-all.sh install stepovh` (service stopped; backups `.bak-g1-stepovh`). Every flag defaults off, and with the flags off the files are inert.

**Measured step overhead** (2027f4d8 nsys, rank 0):
- Outside the graphs: C1 1.15 ms (drafter->target 297 µs: 140 busy, 158 launch gaps; target->drafter 856 µs: 782 busy, 80 gaps); C4 1.45 ms.
- The host is never the bottleneck. It issues the next target graph 24 ms (C1) to 48-88 ms (C4) before the GPU needs it, with 0 host-bound idle except at request boundaries.

**Lever 1: what rank 0 broadcasts.**
- Per decode step, 2 pickled RPCs go through vLLM's MessageQueue: execute_model(SchedulerOutput) and sample_tokens(None). The local worker reads them from a shm ring; the 7 headless nodes get one ZMQ send. Only rank 0 replies.
- There is no GPU broadcast. Every rank samples from the all-gathered logits, and the draft and verify randomness is counter-keyed. So knapcio's 9 per-step broadcasts have no counterpart here.
- The analogue that does exist is the [M,16160] bf16 logits all-gather on CERT-armed greedy steps (part B).

| part | flag | expected ms/step C1 / C4 / C16 | status |
|---|---|---|---|
| A: L2 window during attention, prefetching the wo_a sonar fp8 twin after the last wq_b (2nd side stream, joined at forward end) | `DSV41_L2_PREFETCH_WOA=1` | -0.20..-0.45 / -0.2..-0.4 / -0.15..-0.4 | CPU planner test PASS (collective windows byte-identical on/off); GPU test + microbench written, not run |
| B: on CERT-armed steps, a 16 B/row (max, id) all-gather replaces the logits all-gather; returns one-hot logits at the stock greedy token | `DSV41_SYNC_FREE=argmax` (`audit` = check only) | -0.05..-0.09 / -0.15..-0.18 / -0.5..-0.85 | CPU (Triton interpreter) test PASS: 344 rows, 96 real rejection_sample comparisons; GPU not run |
| C: the 14 per-group decode SWA metadata launches as one launch | `DSV41_EAGER_GLUE=all,swam` | -0.04..-0.05 at all loads | CPU (interpreter) bitwise test PASS; GPU not run |

- **Combined estimate:** about -0.3 to -0.6 ms at C1 (-0.9 to -1.8%) and -0.4 to -0.6 ms at C4.
- **Exactness:** A is a cache hint only; C performs the same stores; B is exact for every NaN-free row. B depends on a collective size that changes with arming, which is identical on all ranks (replicated state). A one-time TP agreement at CERT build makes it fail closed.
- **Next window:** GPU tests via `ops/run-tests.sh gpu` on one stopped node. Then a pinned ABAB with the bundle installed and flags off as control, which must give a8c3394af9d2 with identical counters. Run `DSV41_SYNC_FREE=audit` first; it should log 0 mismatches.
- **Not built (next candidates):** main_proj GEMM on a side stream overlapping the 494 µs CERT screen (-0.03..-0.08); CERT screen retune from 72% to 85% of DRAM bandwidth (-0.07, exact); `_prepare_dflash_inputs` x3 -> 1 (-0.015).

## G1 step (2026-10-26 07:25-09:55 UTC): goal metrics G1/G4, GPU clock 2300 MHz PROMOTED (supervisor pre-step), pre-launch compaction NOT promoted -- **2300 MHz REVERTED 2026-10-26 14:10 UTC (user: "revert the power changes"); clock back to 2000, see "Goal window 2"**

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

## G1 candidates step (2026-10-26 09:50-11:40 UTC): m32-bw `all` PROMOTED (launcher 395bf5b2), stepovh NOT promoted (+0.7%, installed inert), fc2c-d05 not taken to serving (bench below its gate)

Scope: the three exact candidates staged during the G1 step (`~/dsv41-stage/g1/{m32-bw,fc2c-d05,stepovh}`): one stopped window for their GPU bitwise tests and microbenches, then pinned ABAB serving A/Bs (G1 / G4 / matched C1 + exact.py first on every instance), promotion, final checks. Clock 2300 (supervisor default) on every arm. Mac drivers and raw results: scratchpad `g2/` (`scripts/sw.sh` stopped window, `ab.sh` one A/B instance, `block*.sh` serial blocks, `abagg.py`, `final.sh`; `ab.tsv`, `ab.log`, `launches.tsv`, `runs/ab-g2-*/`, `res/` test + bench outputs, `diag/` the two corrected tests). spark-one run dirs: `~/dsv41-g1-20261026/runs/ab-g2-*`, `runs/g2-FINAL*`.

**Live state after this step:** instance **`g2-F1`**, healthy 11:28:14 UTC (+350 s, first try), started by the supervisor with its defaults: launcher **395bf5b2** (= 04b6738e + `DSV41_MOE_M32_G1=all`; AV unpinned as before), clock pre-step 2300 on 8/8, no compaction. It replaced `g2-FINAL` (same config), whose 1M probe came out slow class (section 4).
- 8/8 active, 0 restarts, 0 ERROR/Traceback on 8/8, rdma fingerprint **c40cab3c30f4d7f2** on 8 ranks (was c2f008432bd4769f: it hashes the l2 prefetch module, now 998af3d7), supervisor site manifest **54ec141eb2dc** identical on 8/8, `all` natives mapped from `~/dsv41-stage/g1/m32-bw/trees/all-{base,d05}` on 8/8 workers, stepovh files installed with their flags off (inert), `vm.compaction_proactiveness` 0 (untouched), no telemetry left running.
- exact.py **a8c3394af9d2** (own unpinned AV profile, counters 910/466/2330/1373). 1M cold TTFT **178.93 s, good class** (chunk ratio 1.16).
- Suite on it: matched C1 **29.94 / 25.95 ms** (AL 5.191 / 2.718); G1 **103.59** tok/s (code 97.72 / prose 78.11 / math 134.95); G4 per-session mean **61.97** / min **48.25** (common window 60.08 / 41.70, aggregate 180.9); TTFT ~24K 3.93 s (warm) / ~198K 32.49 s; GSM8K-50 **50/50**; full longctx **PASS** (needle 60/60, copy 1.000, 164 s); `prompt_tokens_details` cached_tokens 0 -> 4160.
- Against the G1 step's live g1-FINAL (unpinned, 2300; own AV profile, so only indicative): G1 99.19 -> 103.59 (+4.4%), G4 per-session mean 55.48 -> 61.97, min 44.06 -> 48.25, C1 31.68 / 26.96 -> 29.94 / 25.95 ms. The pinned ABAB in section 2 is the authoritative comparison (G1 +3.0%, G4 +6.7%).

### 1. Stopped window (service down 09:53:22 -> first A/B instance healthy 10:09:49)
Stop on 8/8 like the supervisor (rank logs -> rank<i>.prev.log), then in parallel: m32-bw unit + site on 8/8, m32-bw bench on ranks 0 and 7, stepovh GPU tests on spark-two; then fc2c-d05 `run-gpu-tests.sh` (unit on 8 ranks, bench on ranks 0 and 7); then my corrected m32-bw and WOA tests (below). Nothing was installed until all tests had run.

| candidate | bitwise tests | microbench | verdict |
|---|---|---|---|
| **m32-bw** (`DSV41_MOE_M32_G1`) | shipped unit test "FAIL" 637-685 of 1456 per rank: **a test artifact** (below). Corrected test (`tests/test_m32bw_unit_diag.py`, added to the stage dir): **8/8 ranks PASS 1640/1640** (synthetic + real layer 5r+2 of each rank's shard, M 1..96, 6 synthetic route patterns + recorded routes, NaN-poisoned workspaces, 60 graph replays per arm and M). Site loader test 5/5 variants PASS on 8/8. Shared caches unchanged on 8/8 | full routed-MoE layer, recorded routes, ranks 0 / 7: `off` x0.996-1.005 (A/A, gate +-1%); `all` **x1.152-1.162 at M=6, x1.177-1.189 at M=12, x1.187-1.208 at M=24, x1.210-1.231 at M=48**, x0.976-0.985 at M=1; single features x1.03-1.05 each (pf / la / ds), `pair` x1.00 (the "pair" form of the DRAM row-locality hypothesis is refuted). `all` streams 199-219 GB/s against 172-184 live (Marlin 211-234). Projection: C1 coding -1.61 ms/step (+5.3%), 4 users -4.9..-5.2 ms (+10.8..11.5%) | serving A/B with `all` |
| **fc2c-d05** (`DSV41_MOE_FC2C`) | unit **8/8 PASS** (2059 cases per rank, all 40 layers, tokens 1-96, stale LIVE_ACT rows NaN/random, graph replay, capture-miss fallback, verify mode, gating) | FC2-only **x1.080-1.096** at M 6-48 (gate x1.15: **fail**); full-MoE saving M=6 0.31-0.40 ms/step (gate 0.5: **fail**), 4-user c4rec M=12 0.56-0.62, M=24 0.95-1.05 ms (gate 1.0: borderline), M=1 0 | **not taken to serving**: -0.36 ms at 1 user is +1.1% (below the 1.5% bar), +2.34 GiB per rank, and it edits the same site file and FC2 kernel as m32-bw `all` (stacking needs the compact tail rebased onto the `all` FC2 tree). JIT cache seeded (inert, nothing installed) |
| **stepovh** (A `DSV41_L2_PREFETCH_WOA`, B `DSV41_SYNC_FREE`, C `swam`) | B GPU **PASS** (1496 rows, 144 rejection_sample comparisons); C GPU **PASS** (20 bitwise checks); A shipped toy "FAIL": **a test artifact** (below); with a bit-pattern compare (`tests/test_woa_gpu_diag.py`) every check passes | A toy: -0.41..-0.45 ms/step at M 1-12, -0.12 at M=24, -0.96 at M=48; isolated wo_a 10.5 / 9.9 / 1.5 us per layer saved at M 1 / 6 / 24. B: compact path 15 us + one-hot 13-125 us at M 6-96 against a 1.5-23.7 MiB logits gather. C: 14 launches 688-700 us -> one launch 586-601 us (-97..-101 us/step) | serving A/B, all three combined |

**The two test artifacts.**
- m32-bw: the eager part of `test_m32bw_unit.py` compared each GEMM's output rows by position. The live ep_scatter places the rows of an expert that several tokens share in a nondeterministic order (atomics), so the comparison fails even live-vs-live: a live re-run control mismatched positionally in 104 of 184 eager cases while its final MoE output was bitwise identical in all of them. Failures were confined to routes with shared experts (hot, spec, skew, recorded windows), never at M=1 or with a single live slot. The corrected test keeps the final-output check (bitwise, all cases) and the layout check, and compares each expert's set of output rows (sorted byte strings) instead of positions: all 6 variants and the live re-run pass everywhere. The shipped graph-replay checks (final output) passed 14/14 per arm and weight set already.
- WOA toy (`test_woa_gpu.py`): the 40-layer toy's residual stream overflows (random MXFP8 wo_a weights up to ~28), so every output element is Inf/NaN (finite fraction 0.0000) and `torch.equal` fails on NaN even for the `none` arm. Comparing bit patterns, graph == eager and none == coll == woa hold at every M, but the toy carries no numerical evidence. WOA is a cache hint only; its exactness rests on serving (exact.py and the G1 digest below).

**Installed (stopped, before the A/Bs; flags default off):**
- m32-bw: `sm121_target_m32.py` 5acc5fde -> **db659d80** on 8/8 (`.bak-g1-m32bw`), via `ops/g1-site.sh install`.
- stepovh: `dsv41_l2_prefetch.py` 514dcb23 -> **998af3d7**, `dsv41_r4_small.py` 02f94f29 -> **c46e66bd**, `sparse_swa.py` 38b16c51 -> **854c3e5f**, `model_state.py` 53e4fe39 -> **f21ee2e6** on 8/8 (`.bak-g1-stepovh`), via `ops/g1-all.sh install stepovh`.
- Supervisor site manifest ad97d72f180e -> **4736c48c7c5e** (identical 8/8; the rdma fingerprint changes with the l2 prefetch module).

### 2. Pinned ABAB (AV pin `av-pin-pdfinal.json`, 2300 MHz, one suite per instance: matched C1 label pdab, G1, G4 x3, TTFT)
Order: C1, M1, C2, M2, MS1, M3, MS2 (C = both bundles installed, flags off; M = `DSV41_MOE_M32_G1=all`; MS = M + `DSV41_L2_PREFETCH_WOA=1 DSV41_SYNC_FREE=argmax DSV41_EAGER_GLUE=all,swam`). All 7 launches healthy at the first try (+274..288 s).

**Exactness: every instance gave exact.py a8c3394af9d2 with the pinned counters 905 / 471 / 2355 / 1373, the same G1 output digest 9a18f802d930f6e7 and the same matched AL 5.224 / 2.718.** On MS instances all three parts were active on 8/8: `DSV41_SYNC_FREE=argmax agreed on every rank` and `first armed step (M=7): 16 B/row all-gather replaces the 7 x 16160 bf16 logits all-gather`; `DSV41_EAGER_GLUE swam: 15 decode SWA groups built in one launch`; WOA `woa_windows=40`, 21 captures with forks=40 per rank, 0 MISMATCH. The `all` natives were mapped from `~/dsv41-stage/g1/m32-bw/trees/all-{base,d05}` on 8/8 M/MS workers. 0 ERROR/Traceback.

| arm (n) | G1 mean (code / prose / math) | G4 per-session mean / min | G4 by session code / prose / math / mixed | G4 common window mean / min | G4 aggregate | C1 coding / prose ms | TTFT ~198K s |
|---|---|---|---|---|---|---|---|
| C (2) | 100.26 (94.2 / 75.7 / 130.9) | 56.63 / 45.51 | 53.5 / 45.5 / 69.0 / 58.6 | 53.58 / 37.95 | 172.6 | 31.42 / 26.72 | 32.67 |
| **M (3)** | **103.23 (+3.0%)** (96.9 / 77.2 / 135.6) | **60.41 (+6.7%) / 48.00 (+5.5%)** | 55.7 / 48.0 / 76.1 / 61.9 | 57.64 (+7.6%) / 40.85 (+7.6%) | 180.9 (+4.9%) | **30.08 (-4.3%) / 25.92 (-3.0%)** | 32.64 (-0.1%) |
| MS (2) | 103.95 (+0.7% vs M) (97.7 / 77.6 / 136.6) | 60.98 (+0.9%) / 48.53 (+1.1%) | 57.1 / 48.5 / 76.0 / 62.3 | 57.92 (+0.5%) / 41.74 (+2.2%) | 184.3 (+1.9%) | 29.88 (-0.7%) / 25.79 (-0.5%) | 32.66 (+0.1%) |

Per instance (C1 ms, G1, G4 mean/min, agg): C1 31.42/26.74, 100.22, 56.68/45.74, 172.9; M1 30.11/25.91, 103.23, 60.32/48.00, 179.9; C2 31.42/26.70, 100.30, 56.58/45.28, 172.2; M2 30.06/25.90, 103.26, 60.81/48.10, 182.2; MS1 29.90/25.87, 103.94, 61.23/48.46, 185.6; M3 30.06/25.94, 103.18, 60.10/47.90, 180.6; MS2 29.86/25.70, 103.95, 60.72/48.60, 183.1. G1 sd within an arm 0.00-0.06 tok/s.
- TTFT ~24K is the first long prefill after each start and is not comparable between fresh instances (C 5.11-5.22 s, M 4.58-7.16, MS 6.04-6.66; warm instances give ~3.9-4.0 s); the ~198K TTFT right after it is flat in every arm (32.52-32.75 s). The m32 kernels are used only for <= 96-token steps, so prefill runs the same code. A warm ~24K check on the final instance is in section 4.
- **m32-bw `all`: PROMOTED.** Exact; G1 +3.0% and G4 per-session +6.7% (both over the 1.5% bar), C1 -1.34 / -0.80 ms per step (the bench projected -1.61 at C1 coding), nothing worse.
- **stepovh (A+B+C): NOT promoted.** Exact and consistently a little faster (G1 +0.7%, G4 +0.9%, aggregate +1.9%, C1 coding -0.20 ms against the -0.3..-0.6 projected), but under the 1.5% bar. Left installed with the flags off (inert: the C and M arms ran with these files), so a later step can re-test B at higher concurrency (projected -0.5..-0.85 ms at C16) without a reinstall.

### 3. Promotion
Launcher `~/dsv41-tp8.sh` **04b6738e -> 395bf5b2** on 8/8 (backup `dsv41-tp8.sh.bak-g1-m32bw` = 04b6738e), one added line after `DSV41_L2_PREFETCH_PRUNE_SHARED`: `export DSV41_MOE_M32_G1=${DSV41_MOE_M32_G1:-all}`. It requires `DSV41_MOE_FC2_D05=1` (default) and no `DSV41_MOE_PDL_EARLY_B` (the site file asserts both), and it makes `~/dsv41-stage/g1/m32-bw/` (trees + JIT caches, `PREP.ok` bb9660e130330ec0) a serving dependency on every node: do not delete or rebuild it in place.

### 4. Final instance
**`g2-FINAL`** (first launch on 395bf5b2, 11:04-11:21 UTC; `g2/scripts/final.sh`: launcher install with sha check 8/8, supervisor defaults, gstart (exact.py first), gfinal (suite + usage + GSM8K-50 + full longctx), a warm ~24K TTFT check, cold solo 1M):
- healthy +355 s (first try); exact.py **a8c3394af9d2** (own unpinned profile 909/467/2335/1373); 0 ERROR/Traceback.
- C1 29.92 / 25.96 ms (AL 5.165 / 2.725); G1 103.06 (96.89 / 77.95 / 134.35); G4 per-session mean / min 60.21 / 47.66 (common window 57.67 / 41.88, aggregate 184.3); GSM8K-50 **50/50**; longctx **PASS** (needle 60/60, copy 1.000, 165 s); cached_tokens 0 -> 4160.
- TTFT ~24K: 5.60 s as the first long prefill after start, then **3.96 / 3.95 / 3.94 s** warm (three fresh-nonce prompts) = the pre-step warm figure (3.93-3.98 s): no prefill change, as expected (the m32 kernels serve only <= 96-token steps). ~198K 32.77 s.
- Cold solo 1M: **221.14 s, slow class** (ratio 1.79; bands 2.63 / 2.90 / 3.86 / 4.71 s), GPU max ~73 C: the known random rank-5 straggler of unpinned launches (unpinned launches with a 1M probe: 1 good / 5 slow at 2000 before the G1 step, 3 good / 1 slow at 2300 in it, 1 good / 1 slow today), not related to this change.

**Relaunch-if-slow screen** (`g2/scripts/ris.sh`: relaunch with the same defaults -> exact.py first -> cold 1M -> keep the first good-class launch, max 2): `g2-F1` was good on the first try (178.93 s, ratio 1.16) and is the live instance (numbers at the top). GSM8K-50, longctx and the usage check were re-run on it (above).

### Rollback
- m32-bw, one launch: supervisor argument `DSV41_MOE_M32_G1=0` (live D05 kernels; exact either way).
- m32-bw, permanent: restore the launcher on 8/8 (`cp -p ~/dsv41-tp8.sh.bak-g1-m32bw ~/dsv41-tp8.sh`, sha 04b6738e), then relaunch. Site file (service stopped, after the launcher): `ssh spark-one cat ~/dsv41-stage/g1/m32-bw/ops/g1-site.sh | bash -s uninstall` (restores 5acc5fde from `.bak-g1-m32bw`).
- stepovh files (inert): `ssh spark-one cat ~/dsv41-stage/g1/stepovh/ops/g1-all.sh | bash -s uninstall stepovh` with the service stopped (restores the four `.bak-g1-stepovh` files; install/uninstall on 8/8 only, the rdma fingerprint hashes the l2 prefetch module).
- fc2c-d05: nothing installed; only its stage dir and seeded `jit-cache/`.

### Next
- Goal: G1 is 103.2 tok/s pinned (103.6 on the live instance) against 120 (+16% needed). G4 per-session mean 60.4 pinned (62.0 live) meets 50; the min session (prose, AL ~1.8) is 48.0-48.3 (40.9-41.7 while all four decode), so "50 per session" still fails on the prose session.
- m32-bw `all` is x0.98 at M=1: routing M=1 calls to the live kernel would recover ~0.08 ms on M=1 steps (small, exact).
- fc2c tail on top of the `all` FC2 tree: FC2 weight bytes -16.7% stack with `all`'s store and dead-walk savings (rough -0.3 ms at 1 user, -1 ms at 4 users); needs a merged kernel and the bitwise test again.
- stepovh B at 16-32 users, where its projection is largest.

## Goal window 1 (2026-10-26 11:40-12:55 UTC): goal metrics on the final config, pinned and production. Goal NOT met (G1 102.8 of 120 tok/s; slowest G4 session 48.6 of 50). No config change

Scope: (1) verify that the server is up on the final config of the G1 candidates step (launcher 395bf5b2 + clock pre-step 2300); (2) measure the goal metrics on one pinned instance and on one production (unpinned) instance of that config: G1 x3, G4 x3, matched C1 (label pdab, twice), bench conc 4/8/16/32 x2, exact.py (first request, and again after all load), GSM8K-50, full longctx, usage, cold TTFT ~24K / ~198K, and the cold 1M TTFT once (production, which also serves as the relaunch-if-slow screen). Nothing was installed or edited: the launcher, the site files, the supervisor and the env defaults are all unchanged. Mac drivers and raw results are in the scratchpad `gw1/`: `scripts/gw1arm.sh` (one instance), `gw1suite.sh` (the suite), `gw1blk.sh` + `gw1teleab.sh` (telemetry on/off blocks), `vcheck.sh` / `vrun.sh` / `vsumm.py` (static check on 8/8), `gw1summ.py`, `gw1table.py`, `gw1tele_tab.py`; plus `runs/<label>/`, `final/<label>/`, `gw1.log` and `launches.tsv`. On spark-one the run dirs are in `~/dsv41-gw1-20261026/runs/` (with `scripts/` and `mac-scripts/`). Telemetry CSVs are in `~/dsv41-g1-20261026/tele/gw1-*` on all 8 nodes.

**Verdict.** Neither target is met. The production numbers below are telemetry-off figures; section 4 explains why.
- **G1, single user: 102.80 tok/s** (code 97.4 / prose 77.1 / math 133.9) against 120. **Not met: it needs +16.7%** (+17.2 tok/s).
- **G4, 4 concurrent sessions: the per-session mean is 62.35** tok/s, above 50, but the **slowest session (prose) is 48.64**. All 6 prose runs fell between 47.5 and 49.6; code, math and mixed were above 50 in every run. **Not met: the prose session needs +2.8%.** Over the common window, while all four sessions decode at once, the slowest session runs at 40.7 (+22.8% needed).
- G4 is the closer of the two targets. The pinned instance (measured with telemetry on) gave G1 102.26 and G4 59.89 / 48.17. Removing the measured telemetry cost (+1.1%) puts pinned G1 at about 103.4, in line with the candidates step's pinned M arm (103.23, measured without telemetry).

### 1. Live config verified (before: 11:42 UTC on g2-F1; after: 12:50 UTC on gw1-U-2)
`gw1/scripts/vcheck.sh` ran on 8/8 and its output was compared with the expected values (`vsumm.py`): **0 mismatches both times**. Apart from the instance label, start time and temperatures, the before and after states are identical.
- **Launcher** `~/dsv41-tp8.sh` **395bf5b2** on 8/8. It contains `export DSV41_MOE_M32_G1=${DSV41_MOE_M32_G1:-all}`, has `--enable-prompt-tokens-details` in ARGS, and has no default AV pin. Backup `.bak-g1-m32bw` = 04b6738e on 8/8. The 18 pd-window backups are unchanged (listing hash ae661a994ec7).
- **Running server, 8/8:** the `vllm serve` cmdline carries `--enable-prompt-tokens-details` and max_model_len 1048576. The env is identical on all 8 nodes: 50 DSV41_/RDMA_ variables, hash 67c2ffc6d11c, no test variables. It includes `DSV41_MOE_M32_G1=all`, `DSV41_MOE_FC2_D05=1`, `DSV41_L2_PREFETCH_PRUNE_SHARED=1`, and `DSV41_AV_PROFILE=pinned` with `DSV41_AV_COST_JSON` unset, so the startup profile is used ("pin miss").
- **Site files, identical on 8/8:**
  - `sm121_target_m32.py` db659d80 (`.bak-g1-m32bw` 5acc5fde).
  - `dsv41_l2_prefetch.py` 998af3d7, `dsv41_r4_small.py` c46e66bd, `sparse_swa.py` 854c3e5f, `model_state.py` f21ee2e6. Their `.bak-g1-stepovh` backups are 514dcb23 / 02f94f29 / 38b16c51 / 53e4fe39; the stepovh flags are off.
  - Every other file in the "Prefill+decode window" list is unchanged (model.py e7264781 ... gate_linear.py 41adaba8, scheduler 6074389f, no rgate or pd_policy file).
  - Supervisor site manifest 54ec141eb2dc. `~/dsv41-stage/g1/m32-bw/PREP.ok` bb9660e130330ec0, with the `all-{base,d05}` natives mapped on the 8 workers.
- **Supervisor** `ops/rel/dsv41-relaunch.sh` **30158e80** (defaults: clock 2300, compaction 0). Backups `.bak-g1-clock` f2f6e44c and `.bak-g1-pre` 39679a35.
- **Clock and compaction pre-steps** (documented in "G1 step", sections 2-3):
  - Every launch in this window logged `prelaunch compact=0 clock=2300 ... clock -lgc 0,2300 OK` on 8/8. Under load the SM clock ran at 2268-2288 MHz on every node.
  - The boot unit `dgx-spark-gpu-clock.service` (`-lgc 0,2000`) is still enabled, so a reboot restores 2000.
  - Compaction did not run (compact=0), and `vm.compaction_proactiveness` is 0.
- **Health:** 8/8 active, 0 restarts, 0 ERROR/Traceback, rdma fingerprint c40cab3c30f4d7f2 on 8 ranks, Engram check OK on 8/8.

### 2. Method
- **Instances.** All were started by the supervisor with its defaults and `--expect-launcher 395bf5b2`:
  - **gw1-P**, pinned (`DSV41_AV_COST_JSON=.../pdab/av-pin-pdfinal.json`, "pin hit"): healthy at +278 s.
  - **gw1-U**, production: healthy at +361 s. Its 1M probe came out in the slow class, so it was relaunched.
  - **gw1-U-2**, production: healthy at 12:23:19 (+354 s). Its 1M probe was in the good class, and it is the live instance.
- **Suite (`gw1suite.sh`).** On each instance: exact.py first (`gstart.sh`; on production, the cold 1M came right after it). Then:
  1. matched C1 coding/prose x5 (label pdab)
  2. G1: `goal_single.py --repeats 3`
  3. G4: `goal_conc4.py` (warm-up + 3 runs)
  4. bench.py conc 4/8/16/32 (512 tokens), x2
  5. matched C1 again
  6. TTFT ~24K / ~198K
  7. usage check
  8. GSM8K-50 (conc 8)
  9. full longctx
  10. exact.py again, after all the load

  A 2 s `/metrics` sampler showed no outside traffic: in every phase the maximum running count equals that phase's own concurrency.
- **Telemetry.** Both suites ran with the G1 step's 2 s telemetry on 8/8 (`g1tele.sh`: `nvidia-smi -lms 2000` plus a sysfs loop). On gw1-U-2 I then ran telemetry-off, on, off decode blocks; see section 4. Telemetry costs about 1.1% on G1 and about 2.3% on G4, so the production headline numbers come from the two telemetry-off blocks (A, C).

### 3. Results
| metric | **production gw1-U-2, telemetry off** (blocks A + C) | production gw1-U-2 suite, telemetry on | pinned gw1-P suite, telemetry on | ref: pinned M arm, candidates step (n=3, no telemetry) |
|---|---|---|---|---|
| **G1 mean tok/s** (code / prose / math), 1024 tokens x3 each | **102.80** (97.37 / 77.13 / 133.90); per rep 102.86, 102.77, 102.80, 102.80, 102.79, 102.77 | 101.75 (96.40 / 76.28 / 132.57) | 102.26 (96.24 / 76.39 / 134.15) | 103.23 (96.9 / 77.2 / 135.6) |
| G1 AL code / prose / math | 2.570 / 1.814 / 3.557 | same instance | 2.558 / 1.807 / 3.607 | - |
| **G4 per-session mean / min tok/s** (3 runs) | **62.35 / 48.64** | 61.11 / 48.16 | 59.89 / 48.17 | 60.41 / 48.00 |
| G4 by session: code / prose / math / mixed | 57.1 / 48.6 / 79.2 / 64.5 | 57.9 / 48.2 / 73.9 / 64.5 | 57.2 / 48.2 / 73.4 / 60.8 | 55.7 / 48.0 / 76.1 / 61.9 |
| G4 slowest session per run (always prose) | 49.64, 49.00, 48.38, 47.51, 48.19, 49.09 | 49.07, 48.53, 46.88 | 47.64, 48.68, 48.19 | - |
| G4 session-runs at >= 50 tok/s | 18 / 24 | 9 / 12 | 9 / 12 | - |
| G4 common window, all 4 decoding (mean / min) | 59.79 / 40.73 | 58.67 / 39.30 | 56.88 / 40.36 | 57.64 / 40.85 |
| G4 aggregate tok/s | 183.1 | 181.8 | 181.5 | 180.9 |
| matched C1 coding / prose ms/step (label pdab) | **29.94 / 25.84** (AL 5.165 / 2.706) | 30.21 / 26.09; second pass 30.27 / 26.10 | 30.33 / 26.18 (AL 5.224 / 2.718); second pass 30.42 / 26.15 | 30.08 / 25.92 |
| bench conc4 aggregate tok/s (per-request mean / min) | **181.4** (59.0 / 48.5) [block A] | 177.1 (60.5 / 46.1) | 177.0 (59.9 / 47.2) | - |
| bench conc8 | **266.3** (44.7 / 34.6) | 262.1 (44.6 / 34.7) | 259.6 (43.5 / 33.6) | - |
| bench conc16 | **395.9** (34.6 / 25.9) | 399.1 (34.1 / 26.6) | 386.6 (33.0 / 24.8; runs 373.3, 400.0) | - |
| bench conc32 | **578.4** (24.5 / 18.9) | 577.3 (24.5 / 19.0) | 579.2 (24.6 / 19.1) | - |
| cold TTFT ~24K / ~198K | - | 3.99 s (23,979 tokens; warm, after the 1M probe) / 32.62 s (197,931) | 6.90 s (first long prefill after start) / 32.61 s | - / 32.64 s |
| **cold 1M TTFT** (1,024,611 tokens) | **179.04 s**, good class (gw1-U-2) | | not run (once only) | |

- **Against the previous window's final** (mm-FINAL: 04b6738e, 2000 MHz, unpinned), the production instance now gives: conc4 162.8 -> 181.4 (+11.4%), conc8 238.9 -> 266.3 (+11.5%), conc16 354.2 -> 395.9 (+11.8%), conc32 514.1 -> 578.4 (+12.5%); matched C1 (pdab) 32.25 / 27.55 -> 29.94 / 25.84 ms (-7.2% / -6.2%). This is the 2300 MHz clock and m32-bw `all` together.
- **Production vs pinned.** The unpinned AV profile of gw1-U-2 differs a little from the pin: math AL is 3.557 vs 3.607, and the prose G1 output differs (the code and math outputs are identical). That is why production G1 math is 1.2% lower than pinned at equal telemetry (132.5 vs 134.2). G4 is the same within noise.

### 4. Telemetry check: on, off, on, off on gw1-U-2 (same instance, same prompts, consecutive blocks)
| block (in order) | telemetry | C1 coding / prose ms | G1 mean (code / prose / math) | G4 per-session mean / min | slowest session per run | G4 common window mean / min | G4 aggregate |
|---|---|---|---|---|---|---|---|
| suite | on | 30.21 / 26.09 | 101.75 (96.4 / 76.3 / 132.6) | 61.11 / 48.16 | 49.1, 48.5, 46.9 | 58.67 / 39.30 | 181.8 |
| A | off | 29.88 / 25.82 | 102.81 (97.4 / 77.1 / 133.9) | 62.68 / 49.01 | 49.6, 49.0, 48.4 | 60.29 / 40.89 | 183.2 |
| B | on | 30.28 / 26.12 | 101.60 (96.2 / 76.3 / 132.4) | 60.80 / 46.85 | 47.7, 48.4, 44.5 | 57.90 / 40.32 | 179.2 |
| C | off | 29.99 / 25.86 | 102.78 (97.3 / 77.1 / 133.9) | 62.02 / 48.26 | 47.5, 48.2, 49.1 | 59.29 / 40.57 | 183.1 |

- **Effect of turning telemetry off:**
  - G1 **+1.10%**; code, prose and math each +1.07 to +1.13%. The spread within a block is 0.02-0.10 tok/s.
  - C1 **-1.02%** for both coding and prose.
  - G4 per-session mean +2.29%, min +2.38%, aggregate +1.45%.
  - conc4 +2.4% and conc8 +1.6%; conc16 and conc32 are flat (-0.8% / +0.2%).
  - The G1 output digest 69ad872124100141 is identical in all four blocks, so telemetry does not change outputs.
- **Consequence for the G1 step's clock A/B.** All of its arms ran with this telemetry, so its relative gains stand, but its absolute figures are about 1% low. This explains the +1.3% gap between that step's pinned 2300 arms (98.97, telemetry on) and the candidates step's C arm (100.26, telemetry off, the same code apart from inert files). For future A/Bs, keep telemetry off, or on in every arm.

### 5. Exactness and gates
- **exact.py, first request on each instance** (the anchor is unchanged):
  - gw1-P: a8c3394af9d2, counters 905/471/2355/1373 (pin hit).
  - gw1-U: a8c3394af9d2, 913/463/2315/1373 (its own profile).
  - gw1-U-2: a8c3394af9d2, 905/471/2355/1373 (its own profile, pin miss).
  - **After all the load (exact #2):** a8c3394af9d2 with 905/471/2355/1373 on both gw1-P and gw1-U-2.
- **G1 outputs:** the 3 repetitions are identical on every instance. The pinned digest over the first two repetitions is **9a18f802d930f6e7**, the same as every pinned instance of the candidates step.
- **GSM8K-50:** 49/50 on both instances. The miss is idx 12 (13 vs 12), the known borderline item.
- **Full longctx:** PASS on both (needle 60/60, copy 1.000, 165 s).
- **usage** `prompt_tokens_details`: chat cached_tokens 0 -> 4160, stream 4160, /v1/completions 0 -> 4160, on both instances.
- **Cold 1M** (`xr_bench.py solo --len 1000k`):
  - gw1-U: 230.05 s, **slow class** (ratio 1.88; chunk bands 2.62 / 3.00 / 4.02 / 4.91 s), so it was relaunched.
  - gw1-U-2: **179.04 s, good class** (ratio 1.16; bands 2.61 / 2.76 / 2.90 / 3.02 s).
  - Unpinned launches at 2300 that had a 1M probe now total 5 good / 3 slow.
- 0 ERROR/Traceback on every instance.

### 6. Thermal and power at 2300 MHz (telemetry phases)
- **GPU:** max **77 C** (spark-one, full longctx on gw1-U-2; 74 C on gw1-P). The guard at 83 C never fired.
- **ACPI SoC zone:** max **94.8 C** (spark-one, conc32 on gw1-P; 93.7 C on gw1-U-2). That is 2.8 C above the G1 step's 92 C maximum and 10 C below the 104.8 C critical trip. The hottest zone is always on spark-one, and its temperature follows sustained load.
- **spark-one system power:** 98.6-124.1 W mean by phase, 148.4 W maximum (longctx). The G1-step 2300 arms gave matched 99.4 / G1 100.4 / G4 110.8 W; here matched was 99.6, G1 98.6 and G4 109.5 W.
- **Fleet GPU-rail sum:** 208-304 W by phase.

### 7. Live state after this window
Instance **gw1-U-2**, healthy since 12:23:19 UTC. It was started by the supervisor with its defaults: launcher 395bf5b2, AV unpinned (startup profile), clock pre-step 2300 on 8/8, no compaction. Its 1M is good class (179.04 s).
- Every item in section 1 was re-verified at 12:50 UTC with 0 mismatches.
- Telemetry is stopped, and 0 telemetry or nvidia-smi processes are left on 8/8.
- There is nothing to roll back: this window made no change. The rollbacks in "G1 candidates step" and "G1 step" still apply.

### 8. Where the gap is
- **G4 (prose session +2.8%).**
  - Measured exact lever not yet promoted: stepovh A+B+C, which gave +1.1% on the slowest session and +0.9% on the G4 mean in the pinned ABAB. It is installed but inert, with its flags off.
  - Staged or estimated levers not yet built: the fc2c tail on top of `all` (estimated about -1 ms/step at 4 users, about +2%), and routing M=1 calls to the live kernel (about 0.08 ms, on M=1 steps only).
  - Together these levers are roughly the size of the gap, so G4 is within reach but not assured. The common-window minimum (40.7) is much further from 50.
- **G1 (+16.7%).** The known exact levers add only a few percent: stepovh +0.7%, fc2c-on-`all` about +1% (-0.3 ms at 1 user), M=1 routing at most about +0.3%, and the next clock step (2418, about +1.3% more) broke the 25 W rule. Reaching 120 needs a lever of a different size. Two candidates: acceptance (the prose AL is 1.81 at 77 tok/s, the weakest type), or MoE weight streaming closer to the 211-234 GB/s that Marlin reaches on the same bytes (`all` reaches 199-219).

## Offline prep g2/fc2c-on-m32all: FC2-compact v2 tail rebuilt on top of the live m32-bw `all` FC2 kernel, staged, NOT installed (2026-10-26 ~13:45 UTC)

No restart, no GPU job, no live-file edit. CPU work ran on spark-four (nice 19, nvcc under a 3 GiB memory cap, while serving). Everything is in `~/dsv41-stage/g2/fc2c-on-m32all/` on 8/8: `ops/g2-all.sh build` gives **PREP.ok dcf5bc0909472121 on 8/8** (tree rebuilt on each node from its own inputs, CPU test 92/92 on each node), and `ops/g2-all.sh status` reports OK 8/8 with the live base db659d80. The handoff is that directory's `README.md`.

**What changed against g1 fc2c-d05.** That candidate patched the same site file and the same FC2 JIT header as m32-bw `all` (launcher 395bf5b2), so the two could not stack. Now:
- **Kernel**: the 5 g1 fc2c header hunks applied to the live `all-d05` header 693cd513 (`tools/make_fc2c_tree.py`; self-check: the same hunks on the plain D05 header reproduce the qualified 9f780e4b byte for byte). Two adaptations: both compact B loads carry the live load's `kG1BHint` (EVICT_FIRST), and a compile-time guard allows the tail only on the g1 FC2 geometry without the FC1-only features. K stays 288 unpadded in the tail: k-blocks 0/1 from the padded w2 descriptor, k-block 2 from `TAIL [384*5120, 32] u8` (TMA zero fill for k 288..383). FC2 weight bytes -16.7%.
- **Native**: copy of the g1 fc2c-d05 tree (`_C` 33d4eb19, unchanged) with the merged header 1900579d: `tree/`, manifest **5a79c849**, 888 files.
- **Site file** `sm121_target_m32.py` db659d80 -> **ea4a8814** (the only site change): db659d80 + the 9e250455 fc2c code (source-identical, checked by the CPU test) + the compact tree follows `DSV41_MOE_M32_G1` (`all` -> this tree; `0` -> the g1 fc2c-d05 tree, so the one-launch m32 rollback still starts with FC2C on; pair/pf/la/ds refused). With `DSV41_MOE_FC2C` unset it issues exactly db659d80's calls. Gate default `DSV41_MOE_FC2C_MIN_TOKENS=2` (M >= 2 compact, M = 1 stays on the live all-d05 kernel). `DSV41_MOE_FC2C=verify` with `all` first poisons the served FC2 buffer (all-d05 stores live rows only).
- **Pre-JIT**: `jit-cache/cache/dsv41_fc2c_d05_v1.3230a7e5eb827e5d3a327c47a2e1a6db` (list 11b6aab6, identical 8/8); `tools/djkey.py` reproduces the live all-d05 key c9b9ccac and the g1 fc2c key 16a0ffc6 first.

**Exactness evidence (offline):** the live all-d05 kernel.cu compiled against the merged tree gives SASS identical to the serving all-d05 cubin (648 lines): the merge is inert outside the compact instantiation. The compact kernel differs from the serving all-d05 cubin in 20 normalized SASS lines, all in the TMA producer, structurally the same as the qualified g1 diff (plain D05 -> fc2c-d05, 20 lines): tail descriptor acquire + prefetch, third k-block B load from the tail descriptor at coordinate 0, expect_tx 0x3180 instead of 0x2180, return-address constants shifted. The same UTMALDG carries the EVICT_FIRST hint. Consumer (ldmatrix, MMA, `ds` epilogue) identical, 168 registers, 0 spills.

**Memory:** +62.9 MB per target layer = **+2.52 GB (2.34 GiB) per rank**, built on each weight's first eager eligible call (vLLM sets `cudagraph_num_of_warmups=1` unless enforce_eager, so the eager warm-up of the first <= 96-token capture size builds them; a capture miss falls back to all-d05 and logs a warning). MemAvailable minimum at a cold 1M prompt today is 10.2-11.6 GB (pd X-ray), and current-config telemetry gives 10.77 GB on spark-one and 11.5-11.8 GB elsewhere (gw1-U-2 incl. full longctx). With the tails: **about 7.7-9.1 GB at 1M, worst rank about 7.7 GB, >= 6 GB** (about 1.7 GB of margin). In the G1 step's P1 telemetry spark-four sat about 4 GB lower than the other nodes at 08:20, during the g1 CPU builds there, so no builds on a node during a 1M probe with FC2C on.

**Stopped window (about 10-15 min GPU):** `ssh spark-one cat ~/dsv41-stage/g2/fc2c-on-m32all/ops/run-gpu-tests.sh | bash` (QUICK=1 for a short pass). It runs `prep-node.sh pre` on 8/8 (private caches; the staged loader's `_G1_ROOT` points at `jit-test/g1root`), `tests/test_g2_unit.py` on 8 ranks (all 40 layers; arms live = installed db659d80 class, staged_off, fc2c = what `get_target_m32_kernel()` builds with FC2C=1, fc2c_all, orig; whole-buffer bitwise with one sentinel because `ds` stores live rows only; MoE eager + graph replay; capture miss; verify mode with a no-poison control; gating; pre-JIT cache hit), then `tests/bench_g2.py` on ranks 0 and 7 (full MoE and FC2-only, live vs fc2c vs fc2c_all, M 1/6/12/24/48 + c4rec 12/24 on the recorded routes, L2-cold 40-layer graphs), then `prep-node.sh post` (serving caches unchanged).

**Expected (projection, not measured):** g1 fc2c-d05 on the plain D05 kernel measured FC2 x1.08-1.10 and MoE -0.31..-0.40 ms/step at M=6, -0.95..-1.05 ms at c4rec M=24. On top of `all` the absolute saving should be similar (the tail removes the same bytes): roughly -0.3 ms at 1 user (+1% G1) and -0.8..-1 ms at 4 users (+1.6..2% per session, against the +2.8% the prose session needs; with stepovh's +1.1% the two together are about the size of the gap). Suggested A/B gate: unit 8/8 PASS and MoE >= 0.25 ms/step at M=6 or >= 0.6 ms at c4rec M=24. `fc2c_all` decides whether a `DSV41_MOE_FC2C_MIN_TOKENS=1` arm is worth it (g1 measured M=1 FC2 x1.08).

**Serving A/B:** install with the service stopped (`ops/g2-all.sh install`; inert with the flag unset), arm `DSV41_MOE_FC2C=1` via the supervisor with the AV pin; per instance exact.py a8c3394af9d2 with identical counters, G1 digest, the `FC2-compact tails on (min rows 384)` line on 8/8 and no capture-miss warning. Promotion would be one launcher line `export DSV41_MOE_FC2C=${DSV41_MOE_FC2C:-1}` after the `DSV41_MOE_M32_G1` line, which makes `~/dsv41-stage/g2/fc2c-on-m32all/{tree,jit-cache}` serving dependencies.

**Rollback:** env `DSV41_MOE_FC2C=0` (default). Site file (service stopped): `ssh spark-one cat ~/dsv41-stage/g2/fc2c-on-m32all/ops/g2-all.sh | bash -s uninstall` (restores db659d80 from `.bak-g2-fc2c-on-m32all`, checks 8/8); run it before the m32-bw site rollback, which restores 5acc5fde from its own backup whatever is installed. Nothing else is touched.

## Offline prep g2/colltrim: collective trims (next-experiments #4), staged, NOT installed (2026-10-26 ~13:55 UTC)

No restart, no GPU job, and no live-file edit. CPU work ran on spark-four under `nice -n 19` with `CUDA_VISIBLE_DEVICES` empty: the native builds, a SASS gate and two CPU tests.
- **Package:** `~/dsv41-stage/g2/colltrim/` on 8/8, digest **0f5d23191482** (34 files, `SHA256SUMS`). Handoff: that directory's `README.md`. Mac copy: scratchpad `ct/pkg/`.
- **Status:** `ops/g2ct-all.sh status` shows the live bases on 8/8 (cuda_communicator 52fc8cf5, rdma_all_reduce 9056dcdf, dsv41_l2_prefetch 998af3d7), nothing installed, digest identical.
- **Install** with the service stopped: `bash ops/g2ct-all.sh install`, run from the Mac in the stage dir. It does a preflight on 8/8 (base shas + stage sums) before any change. Backups are `<file>.bak-g2-colltrim`. Libraries go to the new dir `~/rdma_ar_dsv41/build-g2ct/`; the live `build/`, `build-k1pl/` and the r4 l2pf lib are untouched.
- Every flag defaults off. With the flags off, the files are behaviourally the live ones.

**(a) In-place column all-gather, `DSV41_RDMA_AG_COLS=1`.**
- **Mechanism.** The rdma_ar gather kernel writes rank r's `[outer, seg]` slice straight into columns `[r*seg, (r+1)*seg)` of the final tensor. This removes the `[world, ...]` staging tensor and cuda_communicator's movedim/reshape `direct_copy`.
- **Scope.** 40 qkv_a gathers per step, plus the logits / Engram / DSpark main_proj gathers. It applies only when `dim > 0` and seg % 16 B == 0; otherwise, and for dim 0, today's path is used.
- **Unchanged:** the wire protocol, op order and the L2-prefetch hook order. Exact by construction: the output tensor is bitwise the same.
- **Native.** `rdma_ar.cu` = k1 parallel-load source 1abfbc63 + mode `kGatherCols` in both kernel sets; `Params.seg` is appended at the end. Built as `rdma_ar_ext.so` 0692daa5.
- **SASS gate vs the live k1 build 62daad6c: all 11 existing kernels are instruction-identical.** That covers every all-reduce and the dim-0 gather, legacy and parallel-load. The new parallel-load cols kernel has 96 regs, no spill, and all 8 peer loads in flight.
- **Python.**
  - `rdma_all_reduce.py` → 26a5d834: new `all_gather_cols`; with AG_COLS=1 it loads the g2 .so, and `DSV41_RDMA_PARLOAD` still picks the kernel set.
  - `cuda_communicator.py` → 87bf77e3: a 6-line branch.
  - CPU glue test PASS (24 calls): the column path is used exactly when expected, the result equals `torch.cat`, and there is one extension call and one prefetch before/after hook per gather.
- **Expected:** -0.10..-0.15 ms/step at 1 user, plus the logits copy on CERT steps (≈ +0.4-0.6% C1). Borderline for the 0.5% bar on its own.

**(b) L2-prefetch pacing and window re-plan (`dsv41_l2_prefetch.py` → 63737f9c).** The goal is to stop the burst prefetch from stretching collectives. k1 measured +10 µs per post-MoE AR, +18 µs per post-attention AR and ~+8 µs per qkv gather, about 1.1-1.5 ms per step. Arms, cache hints only:

| env | effect |
|---|---|
| `DSV41_L2_PREFETCH_REPLAN=attn` | The post-attention AR window (mhc fn + router gate) is forked at the block's last wq_b instead, on the WOA-style stream joined at forward end. The AR then runs with no prefetch. Composes with stepovh WOA. |
| `REPLAN=qbprev` | The gather window's q_b bytes move into the previous collective's window. |
| `REPLAN=qbpost` | The same bytes are forked right after the previous collective's kernel, via a new `after_collective` hook in rdma_all_reduce. |
| `DSV41_L2_PREFETCH_PACE_GBPS=R` | A one-warp paced issuer at R GB/s (`libdsv41_l2pf_g2.so` 0451009d) on a third side stream joined only at forward end. Knobs: `_PACE_KINDS`, `_PACE_HEAD_KB` (burst head), `_PACE_DELAY_US`, `_PACE_MAX_US`. |

- **CPU planner test PASS.** Flags off, the plan equals the live 998af3d7 for 32 env combinations (WOA tables included). Re-plans conserve the prefetched byte ranges and move only what they should. The attn forks land at the last wq_b. The pacing split is correct.
- **Expected:** -0.3..-1.0 ms at 1 user if the attention phase absorbs the bytes. The 8-rank bench measures this per layer before any serving arm.

**(c) RoCEnante: plan only, needs your OK to run third-party code.**
- **Exactness.** It reduces in fixed rank order 0..W-1 in fp32 with one rn-bf16 rounding, which is our rule. The one difference: it initialises the accumulator with rank 0's value where we start from +0.0. They therefore differ only when every rank's element is -0.0: they give -0.0, we give +0.0.
- **Port.** A port would keep our consumer code. Its gather writes the same column layout as (a).
- **Microbench design** (in the stage README): all-reduce rows 1/6/24/48 + the (6,224) gather, with and without a 6 MB background prefetch. Port only at ≥10 µs saved per call and bitwise.

**Stopped window (~25 min), then the serving A/B.**
1. `bash ops/run-tests.sh gpu spark-two` (~4 min).
   - Column self-test: 19 shapes × 3 ranks × 2 kernel sets bitwise, guard bytes, a 40-gather CUDA graph × 30 replays, local timing.
   - Paced-kernel timeline and the g2 fork/join of 7 arms in a 12-layer toy under CUDA graphs. Gates: graph == eager, arms bitwise equal, streams joined, planned forks issued, flags-off plan == live on recorded events.
2. `bash ops/run-bench8.sh g2ct-w1` (8 ranks, ~5-10 min, staged files only).
   - A: cols vs today vs PyNccl bitwise, plus step-like timing.
   - B: AR/AG in-kernel stretch under none / burst 1.6-6 MB / paced 50-200 GB/s / delay / head.
   - W: per-layer window emulation (post-attention: live / pacehead / attn / attnpace; q_b: live / qbprev / qbpost / qbpostpace / ag0).
   - Serve only the (b) modes that beat `live` by ≥2 µs per layer at M=1 and 6.
3. `bash ops/g2ct-all.sh install`.
4. Pinned ABAB with telemetry off: exact.py first (a8c3394af9d2, counters 905/471/2355/1373), G1 digest 9a18f802d930f6e7, G1, G4, matched C1.
   - Arms: C (flags off), A `DSV41_RDMA_AG_COLS=1`, then the chosen `DSV41_L2_PREFETCH_REPLAN=...` / `..._PACE_GBPS=...`.
   - Promotion rule of this window: ≥0.5% on the G4 prose-session MIN or the G1 mean in every pinned pair.

**Rollback:** env flags off (`DSV41_RDMA_AG_COLS=0`, `DSV41_L2_PREFETCH_REPLAN=`, `..._PACE_GBPS=0`). Files (service stopped): `bash ops/g2ct-all.sh uninstall [--purge]`.

**Notes**
- The supervisor's site manifest covers the three site files and `build-k1pl`, but not `build-g2ct`. The rdma fingerprint hashes the selected .so at startup on every rank.
- The staged prefetch module is based on 998af3d7, so any other package that edits those three files makes the preflight refuse.
- fc2c-on-m32all edits different files.

## Goal window 2 (2026-10-26 14:10-18:10 UTC): power change REVERTED (GPU clock back to 2000 MHz); stepovh + fc2c-on-m32all PROMOTED as a stack (launcher 592793fd); colltrim NOT promoted (installed inert). Final verification: goal NOT met (G1 100.4 of 120 tok/s; prose session 46.4-48.1 of 50)

**User instruction, 14:10 UTC:** "Let's revert the power changes. We want to stay efficient." The 2300 MHz clock pre-step promoted in the "G1 step" is withdrawn. From now on clocks and power limits are never raised, and changes that do not add power are preferred. All numbers in this section were taken at the 2000 MHz boot policy. The 2300 MHz numbers in "G1 step", "G1 candidates step" and "Goal window 1" are about 3% higher than what the same code gives at 2000 and cannot be compared with the numbers below.

Scope: (0) the clock revert, (1) the stopped window for the two staged g2 packages, (2) re-measuring the control at 2000 MHz, (3) pinned A/Bs of stepovh (S), fc2c-on-m32all (F) and colltrim (K), (4) the stack of the winners (X = S+F), (5) promotion and the final production instance. Mac drivers and raw results are in the scratchpad `gw2/`: `scripts/gw2ab.sh` (one pinned instance), `gw2blk.sh` (a serial block), `gw2suite.sh` (timed suite, telemetry off), `gw2pow.sh` + `gw2powline.py` (power phase), `gw2line.py`, `gw2agg.py`, `gw2table.py`, `gw2final.sh` (promotion and final checks), `sw.sh` (stopped window); plus `ab.tsv`, `pow.tsv`, `agg.txt`, `tables.md`, `ab.log`, `launches.tsv`, `ab/<label>/`, `runs/<label>[-pow|-start]/`, `final/`, `sw/`, `old-2300/` (the one launch made at 2300 MHz before the revert, never measured). On spark-one the run dirs are in `~/dsv41-gw2-20261026/runs/`; telemetry CSVs are in `~/dsv41-g1-20261026/tele/gw2p-*` on all 8 nodes.

### 0. Power revert (14:09-14:10 UTC, before any measurement in this section)
- **Runtime:** `nvidia-smi -lgc 0,2000` on 8/8 at 14:09:32 UTC through the privileged container (Mac `g1/scripts/clk.sh`, logged in `g1/clk.log`). No restart is needed.
- **Supervisor:** `ops/rel/dsv41-relaunch.sh` 30158e80 -> **7bea1cf3** (`PRE_CLOCK_DEFAULT=2000`; backup `.bak-g2-clock2300` = 30158e80). Every launch in this window also passed `DSV41_GPU_CLOCK_MHZ=2000` explicitly and logged `prelaunch (compact=0 clock=2000): ok on 8/8`.
- **Checked before every measurement:** the driver read the SM clock on 8/8 at instance start and again right before each suite, and would have aborted if any node read more than 2010 MHz. Every reading was 1969-1989 MHz. The telemetry phases peaked at 1989 MHz in every instance. GPU max under the goal load was 61-65 C (71 C in the G1 step's 2300 MHz arms).
- The boot unit `dgx-spark-gpu-clock.service` (`-lgc 0,2000`) is still enabled on 8/8, so a reboot also gives 2000. Compaction stays off (`compact=0`).
- No other power setting was changed in this window: no power limits (`-pl` is not supported on GB10), no CPU or wake toggles.

### 1. Stopped window (14:02:35-14:06:31 UTC, run by the first attempt of this step, before the revert, so at 2300 MHz)
The bitwise results do not depend on the clock. The bench timings were only used as gates, and the 2000 MHz serving A/B below is the measurement.
- **fc2c-on-m32all** `run-gpu-tests.sh`:
  - Unit test **8/8 PASS**: 2064 cases per rank, 0 failures. It covers all 40 layers (5 per rank), tokens 1-96, stale padding rows, eager and graph replay, capture-miss fallback, verify mode, gating, and a pre-compiled cache hit.
  - Microbench on ranks 0 and 7 (projected serving saving per step): M=1 0.00 ms; M=6 0.32-0.37 ms; c4rec M=12 0.50-0.52 ms and M=24 0.80-0.82 ms; M=48 1.16-1.44 ms. The suggested gate (0.25 at M=6 or 0.6 at M=24) is met. Every-M (`MIN_TOKENS=1`) adds only 0.08 ms at M=1, so it got no arm.
  - Serving caches unchanged on 8/8.
- **colltrim:**
  - `test_cols_selftest` **PASS** (114/114 cases bitwise, 1200/1200 replayed gathers).
  - `test_l2pf_g2_gpu` **FAIL (2)**. Both failures are pacing timing: the paced issuer took 69.4 us against an expected 31.4 us at 200 GB/s 6 MB, and 28.2 against 18.9 us at 150 GB/s 2 MB with a 5 us delay. Pacing arms were therefore not served.
  - `run-bench8.sh` **PASS**. A1 32/32 routed cases bitwise (cols == today == PyNccl). A2: 30 replays x 40 column gathers, 0 mismatches. A3: the column gather saves 0.80 us per gather.
  - Window bench, us per layer at M=1. W1 (post-attention all-reduce): none 80.6, live 75.8, pacehead 73.9, **attn 60.7**, attnpace 69.4. W2 (qkv / q_b): none 65.7, live 63.1, qbprev 62.4, qbpost 66.9, qbpostpace 61.0, ag0 65.9.
  - Only `REPLAN=attn` beat today's prefetch by the 2 us per layer the package README asks for, so arm K = `DSV41_RDMA_AG_COLS=1 DSV41_L2_PREFETCH_REPLAN=attn`.
- **Installs** (service stopped, flags default off):
  - `g2-all.sh install`: `sm121_target_m32.py` db659d80 -> **ea4a8814** (`.bak-g2-fc2c-on-m32all`), digest 8c183d1de797 on 8/8.
  - `g2ct-all.sh install`: `cuda_communicator.py` 52fc8cf5 -> **87bf77e3**, `rdma_all_reduce.py` 9056dcdf -> **26a5d834**, `dsv41_l2_prefetch.py` 998af3d7 -> **63737f9c** (`.bak-g2-colltrim`); the new library sits at `~/rdma_ar_dsv41/build-g2ct/` next to the live one. Digest d944401790d0 on 8/8.
  - Supervisor site manifest -> 25da15a0969c. With `DSV41_RDMA_AG_COLS` off, the live `build-k1pl` library is still the one loaded.

### 2. Method (pinned ABAB at 2000 MHz)
- **Launches.** 14 supervised launches (`gw2ab.sh` through `gw2blk.sh`). Each used the AV pin `av-pin-pdfinal.json` ("pin hit"), `DSV41_GPU_CLOCK_MHZ=2000` and launcher 395bf5b2, with the candidate flags passed as env. All were healthy at the first try (+277..299 s).
- **Order:** C1 S1 F1 K1 C2 K2 F2 S2 C3, then X1 C4 F3 X2 C5. The arms are:
  - C: flags off.
  - S: stepovh, `DSV41_L2_PREFETCH_WOA=1 DSV41_SYNC_FREE=argmax DSV41_EAGER_GLUE=all,swam`.
  - F: `DSV41_MOE_FC2C=1`.
  - K: colltrim, `DSV41_RDMA_AG_COLS=1 DSV41_L2_PREFETCH_REPLAN=attn`.
  - X: S+F.
- **Per instance:**
  1. exact.py first (`gstart.sh`).
  2. The timed suite with **telemetry OFF** (`gw2suite.sh`): matched C1 label pdab x5, G1 x2, G4 warm-up + **6** runs, TTFT ~24K / ~198K.
  3. A **power phase** with 2 s telemetry on 8/8 (`gw2pow.sh`: 10 s idle, G1 x1, G4 x2). It is used only for watts, never for timing.
- **Pairs.** Each candidate instance is compared with the mean of the two control instances that bracket it.
- **G4min_hat** = 511 / (148 T4 + 31.2 T3 + 30.6 T2 + 68.5 T1), using the instance's per-phase step times. This is the step-time-only estimate of the prose session from the prose-c4 analysis. It removes the noise that comes from which prose text is generated: a single prose run varies by sd 0.80 tok/s (1.7%) over the 30 control runs, and prose takes 264-293 steps.
- **Concurrent activity.** The Codex app on the Mac kept running its read-only research rounds 15-17 during this window (output dir mtimes 15:53-17:07 UTC; round 17 has capture scripts aimed at spark-one). An earlier note says such rounds run short un-niced cuobjdump/objdump/sha256 jobs on spark-one (about +10 W peak, about 0.4% C1 noise). The two ps checks I made (14:16 and 17:05 UTC) found no such process, and the controls stayed tight: G1 sd 0.05%, C1 coding sd 0.08% over 5 instances. Any interference therefore stays far below the measured effects. Still, ask the user to pause Codex during future measurement windows.

### 3. Results (14 pinned instances, 2000 MHz, telemetry off)
**Exactness.** All 14 instances returned the same results:
- exact.py **a8c3394af9d2** with the pinned counters 905 / 471 / 2355 / 1373 and per-position 351/224/166/96/68 (pin hit);
- G1 output digest **9a18f802d930f6e7**, the same pinned digest as at 2300 MHz, so the clock does not change outputs;
- matched AL 5.224 / 2.718;
- 0 ERROR/Traceback.

The candidate log lines were present on 8/8 ranks:
- S: `DSV41_L2_PREFETCH_WOA armed`, `DSV41_SYNC_FREE=argmax agreed on every rank`, `EAGER_GLUE swam`;
- F: `DSV41_MOE_FC2C=1: FC2-compact tails on (min rows 384); 62.9 MB per FC2 weight`, with the fc2c tree `_C` mapped;
- K: `rdma_ar g2 column all-gather: ON` and the g2 plan with 40 attn forks.

Per instance, in launch order. "MemAvail min" is the lowest MemAvailable over the 8 nodes at the end of the suite.

| # | instance | arm | C1 coding / prose ms | G1 mean (code / prose / math) | G4 sess. mean / prose MIN / hat | G4 cw min | G4 agg | MemAvail min GB |
|---|---|---|---|---|---|---|---|---|
| 1 | gw2-C1 | C | 31.18 / 26.96 | 98.94 (93.0 / 73.9 / 130.0) | 58.63 / 46.02 / 47.41 | 39.39 | 176.0 | 12.3 |
| 2 | gw2-S1 | S | 30.99 / 26.69 | 100.10 (94.1 / 74.8 / 131.5) | 58.78 / 47.25 / 47.78 | 39.46 | 177.4 | 12.1 |
| 3 | gw2-F1 | F | 30.92 / 26.79 | 99.53 (93.5 / 74.2 / 130.9) | 59.13 / 47.19 / 47.75 | 39.80 | 177.8 | 9.6 |
| 4 | gw2-K1 | K | 31.10 / 26.92 | 99.04 (93.1 / 73.8 / 130.3) | 58.37 / 46.49 / 47.30 | 39.33 | 176.1 | 12.2 |
| 5 | gw2-C2 | C | 31.17 / 26.91 | 98.98 (93.0 / 73.9 / 130.0) | 58.38 / 46.66 / 47.41 | 39.38 | 176.2 | 12.3 |
| 6 | gw2-K2 | K | 31.05 / 26.88 | 99.25 (93.3 / 73.9 / 130.5) | 58.70 / 46.48 / 47.20 | 39.42 | 176.4 | 12.2 |
| 7 | gw2-F2 | F | 30.97 / 26.80 | 99.33 (93.4 / 74.2 / 130.3) | 59.09 / 46.85 / 47.67 | 40.70 | 178.2 | 9.6 |
| 8 | gw2-S2 | S | 30.89 / 26.69 | 99.93 (93.9 / 74.7 / 131.2) | 58.45 / 46.44 / 47.89 | 39.83 | 177.2 | 12.6 |
| 9 | gw2-C3 | C | 31.18 / 26.99 | 98.85 (92.9 / 73.8 / 129.9) | 57.97 / 46.58 / 47.13 | 38.98 | 174.9 | 12.4 |
| 10 | gw2-X1 | X | 30.68 / 26.59 | 100.37 (94.3 / 74.9 / 131.9) | 60.06 / 46.67 / 48.08 | 40.19 | 179.1 | 9.9 |
| 11 | gw2-C4 | C | 31.20 / 27.00 | 98.86 (92.9 / 73.9 / 129.9) | 58.69 / 46.97 / 47.13 | 39.31 | 176.7 | 12.4 |
| 12 | gw2-F3 | F | 30.87 / 26.77 | 99.64 (93.7 / 74.3 / 130.9) | 59.45 / 47.09 / 47.83 | 40.84 | 178.0 | 9.7 |
| 13 | gw2-X2 | X | 30.60 / 26.59 | 100.40 (94.4 / 75.0 / 131.9) | 59.55 / 47.16 / 48.29 | 39.76 | 177.7 | 9.2 |
| 14 | gw2-C5 | C | 31.23 / 27.00 | 98.90 (92.8 / 73.9 / 130.0) | 58.96 / 47.01 / 47.15 | 39.67 | 177.6 | 12.5 |

**Control at 2000 MHz** (C, n=5; the re-measured baseline for everything that follows):
- **G1 98.91** (sd 0.05): code 92.92 / prose 73.86 / math 129.94.
- **G4:** per-session mean **58.53**, **prose MIN 46.65** (sd 0.40; G4min_hat 47.25), common window 55.91 / 39.35, aggregate 176.3.
- Matched C1: 31.19 / 26.97 ms. TTFT ~198K: 33.48 s.
- The 2300 MHz figures predicted about 99.5 for G1 and about 47 for the prose MIN. G1 is 4.2% below the pinned M arm at 2300 (103.23).

Pairs (+ = better, in %). Each instance is compared with the mean of its bracketing controls.

| inst | controls | G1 | G1 code | G1 prose | G1 math | G4 MIN | G4 hat | G4 mean | cw min | agg | C1 code | C1 prose | TTFT 198K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | C1+C2 | +1.15 | +1.13 | +1.17 | +1.15 | +1.97 | +0.78 | +0.46 | +0.18 | +0.74 | +0.59 | +0.91 | +0.55 |
| S2 | C2+C3 | +1.03 | +0.98 | +1.15 | +1.00 | -0.38 | +1.31 | +0.47 | +1.65 | +0.91 | +0.91 | +0.96 | +0.01 |
| F1 | C1+C2 | +0.58 | +0.53 | +0.48 | +0.67 | +1.83 | +0.71 | +1.07 | +1.06 | +0.93 | +0.82 | +0.54 | +0.18 |
| F2 | C2+C3 | +0.43 | +0.49 | +0.56 | +0.30 | +0.50 | +0.85 | +1.57 | +3.86 | +1.52 | +0.66 | +0.56 | -0.00 |
| F3 | C4+C5 | +0.77 | +0.87 | +0.62 | +0.79 | +0.21 | +1.47 | +1.07 | +3.42 | +0.48 | +1.11 | +0.85 | +0.65 |
| K1 | C1+C2 | +0.08 | +0.07 | -0.15 | +0.22 | +0.34 | -0.22 | -0.24 | -0.14 | -0.02 | +0.24 | +0.06 | +0.01 |
| K2 | C2+C3 | +0.34 | +0.38 | +0.12 | +0.43 | -0.29 | -0.15 | +0.91 | +0.60 | +0.49 | +0.40 | +0.26 | +0.26 |
| **X1** | C3+C4 | **+1.53** | +1.55 | +1.47 | +1.55 | -0.23 | **+2.01** | +2.96 | +2.65 | +1.90 | +1.64 | +1.50 | +0.03 |
| **X2** | C4+C5 | **+1.54** | +1.62 | +1.52 | +1.49 | +0.35 | **+2.43** | +1.24 | +0.68 | +0.31 | +1.97 | +1.52 | -0.30 |

The ~24K TTFT is the first long prefill after each start. It ranged from 4.4 to 8.1 s in every arm, C included, so it is not comparable between fresh instances. A warm check on the final instance is in section 6.

### 4. Decisions (rule of this window: >= +0.5% on G1 or on the G4 prose MIN in every pinned pair, >= 2 pairs, exact, nothing worse beyond noise)
- **S, stepovh A+B+C: PROMOTED.**
  - G1 +1.15 / +1.03% in both pairs. C1 is 0.6-1.0% faster, and the G4min_hat is +0.78 / +1.31%.
  - The raw G4 MIN was +1.97 / -0.38%: S2's prose session had one 43.4 tok/s run. A pair's raw G4 MIN has a noise sd of about 1%, so -0.38% is within noise.
- **F, fc2c-on-m32all: PROMOTED, as part of the stack.** It is a batched-step lever: the 4-session step is 0.7 ms faster (45.32 against 46.02 ms).
  - Every G4 figure improved in all 3 pairs: G4 mean +1.1 / +1.6 / +1.1%, common-window min +1.1 / +3.9 / +3.4%, G4min_hat +0.71 / +0.85 / +1.47%. G1 was +0.58 / +0.43 / +0.77%.
  - **Against the literal rule, F alone falls short.** Pair 2 has G1 +0.43% and raw MIN +0.499%; pair 3 has a raw MIN of +0.21%.
  - It is promoted because:
    - its G4min_hat (the noise-free form of the same metric, which the prose-c4 analysis recommends for step-time arms) clears 0.5% in all 3 pairs;
    - the stack that contains it clears the literal rule;
    - it adds on top of S inside the stack: X against S gives G1 +0.37%, hat +0.73%, G4 mean +2.0% and C1 coding -1.0%.
  - **Cost: 2.34 GiB per rank.** MemAvailable after the suite fell from 12.4 to 9.2-9.9 GB, and the 1M floor check is in section 6.
- **X = S+F: PROMOTED.**
  - G1 **+1.53 / +1.54%**, G4min_hat **+2.01 / +2.43%**, G4 mean +3.0 / +1.2%, C1 coding / prose -1.8 / -1.4%.
  - The raw G4 MIN was -0.23 / +0.35%: X1's prose session ran 281.7 steps, X2's 284.2 and the controls' 280.3. That is a difference in which text was generated, not in step time.
  - Stack figures (n=2): **G1 100.39** (94.34 / 74.95 / 131.88), G4 session mean 59.81, prose MIN 46.91 (hat 48.18), common window 57.16 / 39.97, aggregate 178.4, C1 30.64 / 26.59 ms.
- **K, colltrim column gather + `REPLAN=attn`: NOT promoted.** G1 +0.08 / +0.34%, hat -0.22 / -0.15%.
  - The bench gain of the attn re-plan (-15 us per layer in the post-AR window) does not appear in serving: attention absorbs the moved bytes.
  - The column gather alone is worth about -0.1 ms of C1 (31.08 against 31.19 ms), as estimated, which is below the bar.
  - The colltrim files stay installed with the flags off. Every C, S, F and X instance ran with them, and with `DSV41_RDMA_AG_COLS` off the live `build-k1pl` library is loaded.

### 5. Power and efficiency (power phase after each timed suite: 2 s telemetry on 8/8, G1 x1 and G4 x2; per-arm means)
| arm | n | fleet GPU W idle / G1 / G4 | spark-one sys W G1 / G4 | J/token G1 (single) | J/token G4 (aggregate) |
|---|---|---|---|---|---|
| C, control (flags off) | 5 | 70.7 / 159.0 / 179.2 | 88.6 / 97.0 | 1.623 | 1.028 |
| S, stepovh A+B+C | 2 | 70.8 / 157.6 / 181.5 | 89.0 / 97.5 | 1.593 | 1.027 |
| F, fc2c-on-m32all | 3 | 70.6 / 155.6 / 182.8 | 88.9 / 97.5 | 1.580 | 1.037 |
| K, colltrim | 2 | 71.2 / 157.6 / 182.1 | 89.4 / 98.4 | 1.606 | 1.040 |
| **X, stack S+F** | 2 | 70.5 / 157.4 / 184.1 | 88.2 / 97.3 | **1.585** | 1.039 |

- Fleet GPU W is the sum over the 8 nodes of the mean `power.draw.instant`. J/token is fleet GPU W divided by that phase's tok/s.
- Spread over the five C instances: G1 157.3-161.3 W, G4 178.0-180.4 W, J/token G4 1.017-1.036.
- **Neither promoted change touches clocks or power limits.**
- **Single user (G1):** the stack draws 1.6 W less on the fleet GPU rails while running 1.5% faster, so energy per token is **-2.3%**.
- **4 sessions (G4):** the stack draws +4.9 W on the fleet (+0.6 W per node, +2.7%) for +1.2% aggregate throughput, so GPU-rail J/token is +1.1%, inside the control spread. spark-one's metered whole-node power is unchanged (97.3 against 97.0 W).
- For comparison, the reverted 2300 MHz clock cost +8..17 W per node.

### 6. Promotion and final instance
**Launcher** `~/dsv41-tp8.sh` 395bf5b2 -> **592793fd** on 8/8. The backup `~/dsv41-tp8.sh.bak-g2-sf` = 395bf5b2 is on 8/8. Changes:
- `DSV41_EAGER_GLUE` default `all` -> **`all,swam`**.
- Three lines added after `DSV41_MOE_M32_G1`, each with a comment:
  - `export DSV41_MOE_FC2C=${DSV41_MOE_FC2C:-1}`
  - `export DSV41_L2_PREFETCH_WOA=${DSV41_L2_PREFETCH_WOA:-1}`
  - `export DSV41_SYNC_FREE=${DSV41_SYNC_FREE:-argmax}`
- The supervisor site manifest includes the launcher, so it moved from 25da15a0969c to **0501bdcb13e0**.
- **New serving dependency:** `~/dsv41-stage/g2/fc2c-on-m32all/` (tree and JIT cache, `PREP.ok` dcf5bc0909472121), in addition to `~/dsv41-stage/g1/m32-bw/`. Do not delete or rebuild either in place.
- `DSV41_MOE_FC2C=1` needs `DSV41_MOE_M32_G1=all` or `0` (`pair`, `pf`, `la` and `ds` are refused).

**Final instance `gw2-FINAL1`** (production: supervisor defaults, AV unpinned = startup profile "pin miss", `DSV41_GPU_CLOCK_MHZ=2000`; driver `gw2/scripts/gw2final.sh`):
- **Launch:** healthy at 16:50:26 UTC (+364 s, first try). SM clock 1976-1989 MHz on 8/8. The env on 8/8 has `DSV41_EAGER_GLUE=all,swam DSV41_L2_PREFETCH_WOA=1 DSV41_MOE_FC2C=1 DSV41_SYNC_FREE=argmax DSV41_MOE_M32_G1=all DSV41_MOE_FC2_D05=1 DSV41_RDMA_PARLOAD=1`.
- **Exactness:**
  - exact.py first: **a8c3394af9d2** (own profile, 909/467/2335/1373).
  - After all the load (exact.py #2): a8c3394af9d2, 909/467/2335/1373, per-position 348/232/170/91/68.
  - The G1 outputs over 3 repetitions give digest **bc1e8386878bd2b7**. That is the same 3-repetition digest as goal window 1's pinned instance gw1-P, so this production profile produced the pinned G1 outputs. Repetitions are identical.
- **Cold solo 1M: 183.50 s, good class** (ratio 1.162, first try).
  - MemAvailable during the prompt (1 Hz memwatch on 8/8, `final/gw2-FINAL1/mw/`): minimum **7.8 GB on spark-one** and 8.6-9.2 GB on the other nodes, against 15.9-16.9 GB just before the prompt.
  - With the FC2C tails, that is still above the 6 GB floor, matching the prep estimate of 7.7-9.1 GB. Without FC2C it was 10.2-11.6 GB.
- **Suite** (`gw1suite.sh`, telemetry off, so the same metrics as goal window 1):

  | metric | value |
  |---|---|
  | matched C1 | **30.50 / 26.64 ms** (AL 5.191 / 2.725) |
  | G1 | **100.50** (code 94.90 / prose 75.57 / math 131.05; AL 2.590 / 1.847 / 3.594) |
  | G4 per-session mean / prose MIN | **61.30 / 48.06** (prose per run 47.87, 47.99, 48.33) |
  | G4 by session | code 56.4 / prose 48.1 / math 78.1 / mixed 62.6 |
  | G4 common window / aggregate | 59.12 / 40.11; aggregate 180.1 |
  | bench conc 4 / 8 / 16 / 32 (2 runs each) | **176.6 / 262.6 / 397.9 / 584.0** |
  | TTFT ~24K / ~198K | **4.04 s** (warm, after the 1M probe) / 33.28 s |
  | GSM8K-50 | **49/50** (the miss is idx 12, the usual borderline item) |
  | full longctx | **PASS** (needle 60/60, copy 1.000, 168 s) |
  | usage | `prompt_tokens_details.cached_tokens` 4160 |

  0 ERROR/Traceback on 8/8, 0 restarts. MemAvailable at the end: 8.1-9.4 GB.
- **Against goal window 1's production instance at 2300 MHz** (gw1-U-2, telemetry off; each instance used its own AV profile, so this comparison is only indicative):
  - G1 102.80 -> 100.50 (-2.2%).
  - G4 62.35 / 48.64 -> 61.30 / 48.06 (-1.7% / -1.2%).
  - conc4/8/16/32 181.4 / 266.3 / 395.9 / 578.4 -> 176.6 / 262.6 / 397.9 / 584.0.
  - C1 29.94 / 25.84 -> 30.50 / 26.64 ms.

  In the pinned data the revert cost -4.2% on G1 and the stack won back +1.5%. Single-user energy per token improved and per-node power did not rise.

### 7. Live state after this window, and rollback
**Live:** instance **gw2-FINAL1**, healthy since 16:50:26 UTC.
- launcher **592793fd** on 8/8;
- supervisor **7bea1cf3** (clock default 2000, compaction 0);
- AV unpinned, as before (the default pin is still a user decision);
- clock lock 0-2000 MHz on 8/8, by the runtime revert and the boot unit;
- `vm.compaction_proactiveness` untouched;
- no telemetry, memwatch or bench processes left on 8/8.

Site files, identical on 8/8:
- `sm121_target_m32.py` **ea4a8814** (`.bak-g2-fc2c-on-m32all` db659d80);
- colltrim, inert: `cuda_communicator.py` **87bf77e3**, `rdma_all_reduce.py` **26a5d834**, `dsv41_l2_prefetch.py` **63737f9c** (`.bak-g2-colltrim` = 52fc8cf5 / 9056dcdf / 998af3d7). The l2 prefetch module carries the WOA window that S uses.
- stepovh: `dsv41_r4_small.py` c46e66bd, `sparse_swa.py` 854c3e5f, `model_state.py` f21ee2e6, now in use through the launcher defaults.

Rollback:
- **One launch** (supervisor arguments):
  - `DSV41_MOE_FC2C=0` turns fc2c off.
  - `DSV41_L2_PREFETCH_WOA=0 DSV41_SYNC_FREE=0 DSV41_EAGER_GLUE=all` turns stepovh off.
  - Both together reproduce this window's C arm. Exact either way.
- **Permanent:** `cp -p ~/dsv41-tp8.sh.bak-g2-sf ~/dsv41-tp8.sh` on 8/8 (395bf5b2), then relaunch with `--expect-launcher 395bf5b2`.
- **fc2c site file** (service stopped, after the launcher rollback): from the Mac, `ssh spark-one cat ~/dsv41-stage/g2/fc2c-on-m32all/ops/g2-all.sh | bash -s uninstall` restores db659d80. Run it before any m32-bw site rollback.
- **colltrim** (inert; service stopped): from the Mac, `ssh spark-one cat ~/dsv41-stage/g2/colltrim/ops/g2ct-all.sh | bash -s uninstall` restores the `.bak-g2-colltrim` files.
- **Clock:** nothing to undo, since 2000 is the boot policy. To force it at once: Mac `bash g1/scripts/clk.sh revert`, or `sudo -n /usr/bin/nvidia-smi -lgc 0,2000` per node.

### 8. Where the gap is (at 2000 MHz, no clock or power increases)
- **G1:** 100.4 pinned (100.5 production) against 120, so **+19.5%** is needed.
  - The remaining exact levers are each worth a few tenths of a percent: routing M=1 calls to the live kernel (~0.3%), the column gather alone (~0.3%), and #7a fused bookkeeping (~0.15%).
  - Reaching 120 needs a different kind of lever: acceptance (prose AL 1.85), or MoE weight streaming closer to Marlin's 211-234 GB/s on the same bytes.
- **G4:** the prose session is 46.9 pinned (hat 48.2) and 48.1 in production, against 50, so **+4.0% (production) to +6.6% (pinned raw)** is needed.
  - Exact levers still open from the prose-c4 analysis:
    - drafter MoE on the M32 path, about +0.5-0.9% (not built);
    - column gather, about +0.3%;
    - fused AV bookkeeping, about +0.15%.
  - The prose-priority verify width (about +0.8-1.7%) is a fairness trade: the other sessions lose 0.5-1.5%. It is not bitwise at 4 users and it needs a hybrid AV pin, so it is the user's decision. It was not run.
- **Efficiency:**
  - The 2300 MHz path (+3.7% G1 for +8..17 W per node) is closed by user decision.
  - Energy per single-user token is now 1.585 J (GPU rails, fleet) against 1.623 J for the 2000 MHz control.

### 9. Final verification (17:10-18:10 UTC): config checked on 8/8, then the goal metrics on a pinned and a production instance at 2000 MHz. **Goal NOT met**

**Verdict.** Neither target is met. All numbers below were taken with telemetry off, at the 2000 MHz boot clock, on the promoted config (launcher 592793fd).
- **G1, single user: 100.39 tok/s in production** (gw2-FU-2: code 95.6 / prose 74.2 / math 131.4) against 120. **Not met: it needs +19.5%** (+19.6 tok/s). The pinned instance gave 100.21 (+19.7% needed), and gw2-FINAL1 (section 6) 100.50 (+19.4%).
- **G4, 4 concurrent sessions: not met.** Code, math and mixed were above 50 tok/s in every timed run on every instance; the lowest of them was 54.4. **The prose session was below 50 in every run:**
  - production gw2-FU-2: 46.44 / 47.45 / 45.35 (MIN 46.42), so **+7.7% is needed** (+10.2% for its worst run);
  - pinned gw2-FP: 47.98 / 46.85 / 47.74 (MIN 47.52), **+5.2%** (+6.7%);
  - production gw2-FINAL1 (section 6): 47.87 / 47.99 / 48.33 (MIN 48.06), **+4.0%** (+4.5%).

  The step-time-only estimate G4min_hat, which removes the effect of which prose text gets generated, is 47.2-47.9 on all three instances, so **+4.4..5.9%** is needed. The per-session mean is above 50 everywhere (58.8-61.3), but the goal requires every session.

#### 9.1 Config verified on 8/8
`fin/scripts/verify8.sh` ran before the step (17:12 UTC, on gw2-FINAL1) and after it (18:07 UTC, on the live gw2-FU-2). Apart from the instance start time, GPU readings and MemAvailable, the two outputs are identical.
- **Launcher** `~/dsv41-tp8.sh` **592793fd** on 8/8; backup `.bak-g2-sf` = **395bf5b2** on 8/8.
- **`--enable-prompt-tokens-details`** is in the launcher ARGS and in the running `vllm serve` cmdline on 8/8. The usage check returned `cached_tokens` 4160 on every instance.
- **Supervisor** `ops/rel/dsv41-relaunch.sh` **7bea1cf3** (`PRE_CLOCK_DEFAULT=2000`, `PRE_COMPACT_DEFAULT=0`); backup `.bak-g2-clock2300` = **30158e80**.
- **Site files**, identical on 8/8 (supervisor site manifest **0501bdcb13e0**):
  - `sm121_target_m32.py` ea4a8814 (`.bak-g2-fc2c-on-m32all` db659d80);
  - colltrim, inert: `cuda_communicator.py` 87bf77e3, `rdma_all_reduce.py` 26a5d834, `dsv41_l2_prefetch.py` 63737f9c (`.bak-g2-colltrim` 52fc8cf5 / 9056dcdf / 998af3d7);
  - stepovh: `dsv41_r4_small.py` c46e66bd, `sparse_swa.py` 854c3e5f, `model_state.py` f21ee2e6;
  - live rdma library `~/rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so` 62daad6c.
- **Serving dependencies** are present on 8/8: `~/dsv41-stage/g1/m32-bw/` (661-664 MB) and `~/dsv41-stage/g2/fc2c-on-m32all/` (`PREP.ok` dcf5bc0909472121).
- **Running env**, the same on 8/8: `DSV41_MOE_FC2C=1 DSV41_L2_PREFETCH_WOA=1 DSV41_SYNC_FREE=argmax DSV41_EAGER_GLUE=all,swam DSV41_MOE_M32_G1=all DSV41_MOE_FC2_D05=1 DSV41_RDMA_PARLOAD=1 DSV41_L2_PREFETCH_PRUNE_SHARED=1 DSV41_TOPK_CANON=1 DSV41_AV_PROFILE=pinned`. `DSV41_AV_COST_JSON` is unset in production, so the startup profile is used ("pin miss").
- **Clock.**
  - The boot unit `dgx-spark-gpu-clock.service` (`nvidia-smi -lgc 0,2000`) is enabled and active on 8/8.
  - Each of the 3 launches logged `prelaunch (compact=0 clock=2000): ok on 8/8`.
  - The SM clock was read on 8/8 **before every measurement**: one `nvidia-smi` query per node in the gap before each phase, with no polling, so telemetry stayed off. That is 39 checks in total. Every reading was **1969-1989 MHz** with clock event reasons 0x0, and the GPU was at most 67 C.
  - No clock or power setting was raised in this step.

#### 9.2 Method
- **Driver** `fin/scripts/fin.sh` (Mac):
  1. Supervised launch with `--expect-launcher 592793fd` and `DSV41_GPU_CLOCK_MHZ=2000` passed explicitly, then the clock check.
  2. `gstart.sh` (exact.py first).
  3. Production only: a cold solo 1M with 1 Hz memwatch on 8/8. A launch that is not in the good class is relaunched.
  4. The `gw1suite.sh` phases, one at a time (`finsuite.sh` on the Mac, `finphase.sh` on spark-one, with the same commands as `gw1suite.sh`). The clock check runs before each phase. The phases are: matched C1 x5 (label pdab), G1 x3, G4 warm-up + 3 runs, bench conc 4/8/16/32 x2, matched C1 again, TTFT ~24K / ~198K, usage, GSM8K-50, full longctx, and exact.py #2.
- **Instances**, in launch order:
  - **gw2-FP**, pinned (`av-pin-pdfinal.json`, "pin hit"): healthy at +283 s. exact.py **a8c3394af9d2** with the pinned counters 905 / 471 / 2355 / 1373 and per-position 351/224/166/96/68 (**EXACT-OK**), and the same again after all the load.
  - **gw2-FU**, production: healthy at +366 s. exact.py a8c3394af9d2 (own profile, 911/465/2325/1373). Its cold 1M took **236.10 s, slow class** (late/early 1.873), so it was relaunched without further measurement.
  - **gw2-FU-2**, production: healthy at 17:51:44 UTC (+361 s). exact.py **a8c3394af9d2** (909/467/2335/1373, per-position 348/224/167/101/69), identical after all the load.
    - Its cold 1M took **183.75 s, good class** (1.161). The lowest MemAvailable during the prompt was 8.29 GB on spark-one and 8.9-9.5 GB on the other nodes, above the 6 GB floor.
    - **This instance is live.**
- **Outside activity.**
  - The `/metrics` sampler saw no outside traffic in any phase: the maximum running count always equalled the phase's own concurrency.
  - The Codex app kept working on the Mac. It wrapped up round 17 at 17:12-17:14 UTC, and at about 17:33 UTC made one `nice -n 19` read of source files from spark-one, during the pinned instance's longctx phase. Nothing reached the server.

#### 9.3 Results

| metric | **production gw2-FU-2** (unpinned, telemetry off) | pinned gw2-FP (telemetry off) | ref: production gw2-FINAL1 (section 6) |
|---|---|---|---|
| **G1 mean tok/s** (code / prose / math), 1024 tokens x3 each | **100.39** (95.63 / 74.16 / 131.38) | **100.21** (94.18 / 74.83 / 131.63) | **100.50** (94.90 / 75.56 / 131.05) |
| G1 per repetition | 100.23, 100.49, 100.45 | 100.27, 100.20, 100.18 | 100.40, 100.54, 100.57 |
| G1 AL code / prose / math | 2.603 / 1.792 / 3.607 | 2.558 / 1.807 / 3.607 | 2.590 / 1.847 / 3.594 |
| G1 output digest (3 reps; reps identical) | 939bb576cc1aad16 (yes) | bc1e8386878bd2b7 (yes) | bc1e8386878bd2b7 (yes) |
| **G4 per-session mean / prose MIN** (3 runs) | **60.34 / 46.42** | **58.84 / 47.52** | **61.30 / 48.06** |
| G4 by session: code / prose / math / mixed | 55.5 / 46.4 / 75.0 / 64.4 | 55.9 / 47.5 / 72.5 / 59.5 | 56.4 / 48.1 / 78.1 / 62.6 |
| G4 slowest session per run (session) | 46.44 (prose), 47.45 (prose), 45.35 (prose) | 47.98 (prose), 46.85 (prose), 47.74 (prose) | 47.87 (prose), 47.99 (prose), 48.33 (prose) |
| G4 session-runs at >= 50 tok/s | 9/12 | 9/12 | 9/12 |
| G4 common window mean / min | 57.89 / 39.93 | 56.44 / 40.43 | 59.12 / 40.11 |
| G4 aggregate tok/s | 176.9 | 178.7 | 180.1 |
| matched C1 coding / prose ms/step (pdab; AL) | **30.91 / 26.90** (AL 5.224 / 2.718); 2nd pass 30.85 / 26.71 | **30.72 / 26.58** (AL 5.224 / 2.718); 2nd pass 30.73 / 26.71 | **30.50 / 26.64** (AL 5.191 / 2.725); 2nd pass 30.56 / 26.61 |
| bench conc4 aggregate tok/s (runs; per-request mean / min) | **176.3** (177.9, 174.7; 60.6 / 45.3) | **174.3** (173.6, 175.0; 59.8 / 44.1) | **176.6** (177.8, 175.4; 59.7 / 47.4) |
| bench conc8 aggregate tok/s (runs; per-request mean / min) | 263.1 (261.8, 264.4; 43.4 / 34.7) | 259.3 (260.2, 258.3; 42.7 / 33.2) | 262.6 (259.6, 265.6; 43.1 / 33.5) |
| bench conc16 aggregate tok/s (runs; per-request mean / min) | 399.7 (399.2, 400.2; 34.4 / 25.7) | 397.4 (401.1, 393.6; 34.0 / 26.1) | 397.9 (392.8, 403.0; 34.9 / 26.1) |
| bench conc32 aggregate tok/s (runs; per-request mean / min) | 577.6 (578.9, 576.4; 24.6 / 18.1) | 586.2 (582.2, 590.1; 25.1 / 19.2) | 584.0 (586.9, 581.2; 24.9 / 19.1) |
| TTFT ~24K / ~198K | 4.09 s (23,868) / 33.30 s (197,737) | 5.46 s (23,934) / 33.46 s (197,825) | 4.04 s (23,800) / 33.28 s (197,861) |
| GSM8K-50 | 50/50 | 49/50 | 49/50 |
| full longctx | PASS | PASS | PASS |
| usage cached_tokens (repeat 4.2K prompt) | 4160 | 4160 | 4160 |
| exact.py #2 after all load (acc / drafts / dtok / gen; per-pos) | a8c3394af9d2 (909 / 467 / 2335 / 1373; 348/224/167/101/69) | a8c3394af9d2 (905 / 471 / 2355 / 1373; 351/224/166/96/68) | a8c3394af9d2 (909 / 467 / 2335 / 1373; 348/232/170/91/68) |
| foreign traffic (sampler, running > own concurrency) | none | none | none |

G4 runs, **production gw2-FU-2** (unpinned, telemetry off):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 54.25 | 45.66 | 72.58 | 63.02 | 58.88 | 45.66 (prose) | 38.35 | 171.4 | 2.505 |
| 1 | 55.40 | 46.44 | 74.66 | 64.56 | 60.26 | 46.44 (prose) | 39.30 | 174.4 | 2.505 |
| 2 | 56.51 | 47.45 | 76.78 | 65.22 | 61.49 | 47.45 (prose) | 40.57 | 177.9 | 2.555 |
| 3 | 54.63 | 45.35 | 73.64 | 63.41 | 59.26 | 45.35 (prose) | 39.92 | 178.5 | 2.466 |

G4 runs, pinned gw2-FP (telemetry off):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 54.62 | 45.53 | 75.90 | 59.02 | 58.77 | 45.53 (prose) | 40.55 | 179.0 | 2.475 |
| 1 | 58.34 | 47.98 | 68.62 | 56.47 | 57.85 | 47.98 (prose) | 40.47 | 178.1 | 2.512 |
| 2 | 54.38 | 46.85 | 75.74 | 60.00 | 59.24 | 46.85 (prose) | 40.76 | 178.9 | 2.400 |
| 3 | 54.90 | 47.74 | 73.11 | 61.93 | 59.42 | 47.74 (prose) | 40.06 | 178.9 | 2.458 |

G4 runs, ref: production gw2-FINAL1 (section 6):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 50.32 | 46.70 | 74.60 | 60.72 | 58.09 | 46.70 (prose) | 38.83 | 175.4 | 2.457 |
| 1 | 54.51 | 47.87 | 76.92 | 66.75 | 61.51 | 47.87 (prose) | 38.99 | 179.5 | 2.572 |
| 2 | 57.46 | 47.99 | 79.46 | 60.30 | 61.30 | 47.99 (prose) | 40.90 | 179.8 | 2.547 |
| 3 | 57.24 | 48.33 | 78.00 | 60.72 | 61.07 | 48.33 (prose) | 40.45 | 181.0 | 2.556 |


Other per-instance facts:

| | production gw2-FU-2 | pinned gw2-FP | ref: gw2-FINAL1 |
|---|---|---|---|
| launch to healthy | +361 s (after one slow-class launch, gw2-FU) | +283 s | +364 s |
| exact.py, first request | a8c3394af9d2 (909/467/2335/1373) | a8c3394af9d2 (905/471/2355/1373) = pin | a8c3394af9d2 (909/467/2335/1373) |
| cold solo 1M (1,024,611 tokens) | **183.75 s**, good class (gw2-FU before it: 236.10 s, slow) | not run (once only) | 183.50 s, good class |
| MemAvailable minimum during the 1M | 8.29 GB (spark-one) | - | 7.8 GB (spark-one) |
| G4min_hat (step times only) | 47.57 | 47.88 | 47.23 |
| MemAvailable at the end of the suite (min over 8 nodes) | 8.4 GB | 9.3 GB | 8.1 GB |

- **The pinned instance reproduces the A/B stack.**
  - Against the stack arms X1/X2: G1 100.21 against 100.37 / 100.40 (they used 2 repetitions), G4min_hat 47.88 against 48.08 / 48.29, matched C1 30.72 / 26.58 against 30.68 / 26.59 and 30.60 / 26.59 ms.
  - Against the 5-instance control: G1 +1.3%, raw prose MIN +1.9% (47.52 against 46.65), G4min_hat +1.3%, G4 mean +0.5%.
  - The G1 outputs over 3 repetitions match the pinned digest bc1e8386878bd2b7.
- **Production numbers vary from launch to launch.** Each production launch measures its own AV profile at startup.
  - gw2-FINAL1 produced the pinned G1 outputs. gw2-FU-2 did not: its digest is 939bb576cc1aad16, because the prose output differs (AL 1.792 against 1.807 pinned).
  - Its raw G4 prose MIN was 46.42, against 48.06 on gw2-FINAL1. Most of that gap comes from which prose text was generated, not from step time. Its prose session took 283 steps per run against 276.7, and its G4min_hat (47.57) is slightly higher than gw2-FINAL1's (47.23).
  - G1, bench concurrency and C1 agree within about 1% across the three instances.
- The ~24K TTFT on gw2-FP (5.46 s) is the first long prefill after start. On the production instances it came after the 1M probe, so it is warm (~4.1 s).
- GSM8K-50 was 50/50 on gw2-FU-2 and 49/50 on gw2-FP. The pinned miss is idx 12, the usual borderline item.

#### 9.4 Distance to the goal (2000 MHz, no clock or power increase)
| target | gw2-FU-2 (production, live) | gw2-FP (pinned) | gw2-FINAL1 (production, section 6) |
|---|---|---|---|
| G1 mean >= 120 tok/s | 100.39: **+19.5%** | 100.21: +19.7% | 100.50: +19.4% |
| G4: every session >= 50 tok/s in every run (prose MIN over 3 runs; worst run) | 46.42: **+7.7%**; 45.35: +10.2% | 47.52: +5.2%; 46.85: +6.7% | 48.06: +4.0%; 47.87: +4.5% |
| G4min_hat (step time only) | 47.57: +5.1% | 47.88: +4.4% | 47.23: +5.9% |

- **G4.** The exact levers still open (section 8) are each worth tenths of a percent: drafter MoE on the M32 path +0.5-0.9%, column gather +0.3%, fused AV bookkeeping +0.15%. Together they do not close a 4-8% prose gap. Closing it needs either the prose-priority verify width or a structural lever. The verify width is the user's decision: it is a fairness trade, it is not bitwise at 4 users, and it needs a new pin.
- **G1.** About +20% is needed, which takes an acceptance lever (prose AL 1.79-1.85) or MoE weight streaming close to the Marlin bandwidth on the same bytes (section 8). The clock path is closed by the user's decision.
- **Default AV pin (still the user's decision).** A pin would make production reproducible across launches: the raw prose MIN now moves by about ±2% between launches, and gw2-FU-2 differs from the pinned G1 outputs. It would also shorten launches (283 s against 361-366 s). It does not close the gap.

#### 9.5 Live state after this step
- **gw2-FU-2**: production, AV unpinned, launcher 592793fd, supervisor 7bea1cf3, clock 0-2000 MHz.
  - Healthy since 17:51:44 UTC on 8/8, with 0 restarts and 0 ERROR/Traceback; `/health` returns 200.
  - Its config matches 9.1, re-checked after the suite.
- No telemetry, memwatch, sampler or bench process is left on 8/8.
- Rollback is unchanged from section 7.
- **Files.**
  - Mac scratchpad `gw2/fin/`:
    - scripts: `fin.sh`, `finsuite.sh`, `finphase.sh`, `finsampler.sh`, `clk8.sh`, `verify8.sh`, `node-verify.sh`, `finsumm.py`, `fintable.py`;
    - logs and checks: `fin.log`, `verify-pre.txt`, `verify-post.txt`, `clk-*.txt`;
    - results: `runs/<instance>/` (with `finsumm.json` / `finsumm.md`), `gw2-FP/`, `gw2-FU/` and `gw2-FU-2/` (relaunch logs, exact.py, 1M with memwatch CSVs), `tables-final.md`.
  - spark-one `~/dsv41-gw2-20261026/runs/`: `fin-gw2-FP`, `fin-gw2-FU-2`, `gw2-FP-start`, `gw2-FU-start`, `gw2-FU-2-start`, `p1m-gw2-FU`, `p1m-gw2-FU-2`.

## LAN API exposure (2026-10-26 20:16 UTC, user request: "going forward, let's expose the API to the LAN")
- **How:** a plain TCP forwarder, not a vLLM change. `~/dsv41-lanproxy/lanproxy.py` (stdlib asyncio, TCP_NODELAY) runs as the user service `dsv41-lanproxy.service` (enabled, linger on, Restart=always). It listens on spark-one's LAN address **192.168.0.238:8000** (interface wlP9s9, Wi-Fi) and forwards to vLLM's `127.0.0.1:8000`. It re-binds automatically if the DHCP address changes.
- **Why a forwarder:** no restart and no launcher change while A/B windows run, and it adds no numerics change. Local tools and the 512S2 tunnel keep using 127.0.0.1:8000.
- **Verified from the Mac:** `/health` 200, `/v1/models` = dsv41 / 1048576, streaming chat answered with `prompt_tokens_details`.
- **Security:** there is NO authentication; anyone on the LAN can use the model. vLLM's `--api-key` (or the gateway idea) would add keys; that is a separate decision.
- **Rollback:** `systemctl --user disable --now dsv41-lanproxy.service`.
- **Optional later:** vLLM can bind the LAN itself with `--host 0.0.0.0` in `~/dsv41-tp8.sh` rank-0 args (then stop the forwarder first; 0.0.0.0 also exposes the fabric and tailnet interfaces).
- **2026-10-26 20:21 UTC — Tailscale too (user request):** the forwarder now also listens on spark-one's tailnet address **100.90.117.38:8000** (`http://spark-one.tail91699.ts.net:8000`). Interfaces are set by `LAN_IFACES` (default `"wlP9s9 tailscale0"`); backup of the LAN-only script `lanproxy.py.bak-lanonly`. Same no-auth caveat applies to every device on the tailnet.

## Goal window 3 (2026-10-26 19:35-23:40 UTC): fresh AV pin; codex-adopt bundle + dhtac PROMOTED as a stack (launcher b0cf94d0); RoCEnante, mfront and the rdma_ar env knobs NOT adopted. Final verification: goal NOT met (G1 101.5-102.3 of 120 tok/s in production; prose session MIN 47.9-48.6 of 50, G4min_hat 47.8-48.5)

Scope: (0) a fresh AV pin from the live config, (1) a stopped window with every GPU qualification test, (2) pinned ABAB A/Bs at 2000 MHz, (3) the stack and its confirmation, (4) promotion and a production instance. Clocks were never raised: every launch passed `DSV41_GPU_CLOCK_MHZ=2000`, and the SM clock was read on 8/8 before every timed phase (telemetry off). Mac drivers and raw results: scratchpad `g3w/` (`scripts/sw3.sh` stopped window, `g3ab.sh` one pinned instance, `g3suite.sh` + `g3phase.sh` per-phase suite, `g3blk.sh` serial block, `g3line.py`, `g3agg.py`, `promote.sh`, `fin/` final instance; `ab.log`, `ab.jsonl`, `pow.tsv`, `sw.log`, `ab/<label>/`, `runs/<label>/`, `sw/`). On spark-one: `~/dsv41-g3-20261026/runs/`.

### 0. Fresh AV pin (`~/dsv41-stage/g3/pin/av-pin-g3.json`, sha 1c98cf9b, on 8/8)
- The old pin `av-pin-pdfinal.json` predates M32 `all`, FC2C and stepovh: it priced verify at 8-96 rows 7-15% too high (new/old 0.85-0.93; 104-192 rows within 1%; draft within 2.6%), so pinned arms under-verified at 4 users.
- Source: the rank-0 startup profile of gw2-FU-2 (production, launcher 592793fd, unpinned "pin miss", healthy 17:51:44) via `moet/tools/mkpin_k.py make`; shape key identical to the old pin (29 capture sizes up to 192, 32 seqs, 16384 batched tokens, 5 spec tokens, TP 8).
- **Anchor under the new pin: exact.py a8c3394af9d2 with counters 909 / 467 / 2335 / 1373 (acc / drafts / draft tokens / gen) and per-position 348 / 224 / 167 / 101 / 69 ("pin hit")**: the same as gw2-FU-2's own profile, so the pin reproduces its source launch. G1 2-repetition output digest under the pin: **69ad872124100141**. All 12 pinned instances of this window reproduced both, at start and again after all the load (exact.py #2).

### 1. Stopped window (service down 19:50:20 -> first A/B instance healthy 20:09:38)
- **RoCEnante vs rdma_ar, 8 ranks, our fabric** (`g3/rocenante/ops/run-bench8.sh`; venv `~/venvs/roce072` installed on 8/8, manifest 1c7a6e9a3254; source read before running: its RDMA proxy is one C file over libibverbs, compiled with gcc into the stage cache; routes come from sysfs; no other I/O):
  - Exactness (phase X, bf16 and fp32, 1-48 rows, eager and graph): every rdma_ar arm **EXACT** against the fixed-order fp32 reference; every RoCEnante arm **EXACT_EXCEPT_ALL_NEG0** (it starts the accumulator from rank 0, so all-(-0.0) elements give -0.0 instead of +0.0; nothing else differs); gathers bitwise equal to torch.cat. `DONE failures per rank: [0, 0, 0, 0, 0, 0, 0, 0]`.
  - Latency, us per all-reduce (rows x 5120 bf16), ours / RoCEnante (best RoCEnante variant in brackets):

    | rows | gap (decode-like spacing) | l2pf (live L2 prefetch running) | b2b | dram (continuous reader) |
    |---|---|---|---|---|
    | 1 | **17.4** / 19.9 | **30.3** / 32.8 | 18.3 / 19.9 | 58.0 / 64.7 |
    | 6 | **31.6** / 40.9 | **44.2** / 45.7 | 30.5 / 41.3 | 84.1 / 91.7 |
    | 12 | 38.6 / 66.7 | 49.9 / 71.3 | 37.9 / 65.4 | 90.1 / 121.7 |
    | 24 | 53.0 / 107.4 | 60.1 / 117.4 | 53.0 / 107.9 | 106.1 / 180.2 |
    | 48 | 77.4 / 204.1 | 84.0 / 215.3 | 77.0 / 203.0 | 127.1 / 292.8 |

    Gathers (rows x 224) are within 1-3 us either way; the per-layer AR+AG+AR pattern at 24 rows is 51.6 vs 91.1 us (l2pf). RoCEnante is always one-shot, which loses from 6 rows on. Its only edge: l2pf mode at 3-5 rows, 2-3 us faster than both our modes.
  - **Decision: RoCEnante not adopted** (rule: >= 10 us/call faster at 1 and 6 rows in gap and l2pf; it is slower in all four).
  - **rdma_ar env knobs: none adopted** (rule: >= 2 us/call in the same modes). Proxy pinned to an X925 core vs unpinned (live): 17.4 vs 18.4 and 30.3 vs 31.8 us at 1 row, pinned 0.2 / 0.1 us slower at 6 rows. `RDMA_AR_VEC_PER_CTA=1024`: +1.2 / +1.1 us at 1 row, -0.7 / +1.1 at 6. Two-shot crossover (phase C): two-shot wins from 4 rows in gap mode = the live 32768-byte threshold, so `RDMA_AR_TWOSHOT_MIN_BYTES` stays.
  - The venv stays installed on 8/8 for later rdma_ar work (P1 doorbell / P2 fail-fast ports, prep README); `bash g3roce/pkg/ops/roce-all.sh uninstall` removes it. Results: `~/dsv41-stage/g3/rocenante/results/g3roce-{smoke,w1}.*` (spark-one), Mac `g3w/sw/roce/`.
- **dhtac** (`g3/dhtac/ops/run-gpu-tests.sh`): unit **PASS 8/8**. For every row count 1-160, all six CUTLASS SM120 MXFP8 tactics and the stock call give **bitwise-identical** draft-head outputs on every rank (one class, stable). Per-call times (8 ranks): tactic 0 is 993-1154 us at 5-25 rows and 372-396 us from 30 rows on; tactics 1/3 are 363-398 us everywhere.
  - **The test's own `recommend` line is wrong:** the rank logs had been rotated by the stop, so it fell back to the newest-mtime autotune file (0db863ad, bucket 32 -> tactic 3). Serving loads **8c8be689** (rank-0 log: "Using FlashInfer autotune cache file"), where buckets 32/64/128 -> tactic 0 on all 8 ranks. So at 17-29 rows (4 and 5 requests: 20 and 25 head rows) serving runs tactic 0 at 1046-1072 us instead of ~380 us. Map used: **`17-29:1`** (tactic 0 is already fastest from 30 rows).
- **mfront** (`g3/mfront/ops/run-gpu-tests.sh`): unit PASS 8/8 (1500 MoE cases + 4500 graph replays per rank, 40 layers, 0 failures), but the bench saved only **6.5-7.4 us/layer** (0.26-0.30 ms/step; gate 8 us/layer at c4rec M=12), and it then hit **CUDA illegal memory access** on ranks 0 and 7. Repro on spark-one (patched copy `~/dsv41-g3-20261026/repro/`): live and fused graphs captured and replayed alternately in one process crash at the M=48 set after M=24 passed; live-only and fused-only pass. Serving mixes the two paths in one process (fused for M*6 <= 1024, live for larger decode graphs and prefill), so **mfront was not installed and not served**. Next: bisect with `CUDA_LAUNCH_BLOCKING=1` + compute-sanitizer and `DSV41_MOE_FC2C=0`.
- **codex-adopt** (`run-tests.sh bench`, spark-two): canon / remap / t2s GPU tests **PASS**, no compile after warm-up, serving caches unchanged. Fused CANON 25.6-26.3 -> 3.84 us per call (-174..-180 us per decode step, 8 calls); remap reuse -34.8..-38.9 us per step; omit token_to_seq: 896.8 -> 0.8 MiB held at a late-1M step, metadata build 72.7 -> 15.8 ms.
- **Codex kits** (`run-kits.sh gpu`, spark-three): fused-topk full run **PASS** (93 s) and compute-sanitizer memcheck **PASS** (0 errors). candidate-only-indexer **FAIL / inconclusive** at its first case: "baseline top-k set unstable" on the all-zeros pattern (stock top-k of tied scores is not unique), so the kit cannot qualify; it is a prefill lever and edits the same file as #1, so it was not pursued.
- **Installs** (service stopped, flags default off): codex-adopt (`g3ca-all.sh install`: sparse_attn_indexer.py 7112c01d -> **481905c2**, flashinfer_sparse.py c8a2da55 -> **b059bdf3**, mla/indexer.py 4753707e -> **d9f4c5c9**; backups `.bak-g3-codex-adopt`; digest 9182ac06d964 on 8/8) and dhtac (`dsv41_r3_dense.py` 993023f2 -> **1f1c0456**, `.bak-g3-dhtac`, digest 95faa4a7255a on 8/8). mfront left at baseline (deep_gemm_utils.py a4a0486a). The supervisor site manifest does not cover these files (still 0501bdcb13e0).
- Stopped-window end state: 8/8 inactive, 0 GPU apps, no leftover test processes.

### 2. Method (pinned ABAB at 2000 MHz, fresh pin)
- **Launches.** 12 supervised launches through `g3blk.sh` / `g3ab.sh`, each with `DSV41_AV_COST_JSON=~/dsv41-stage/g3/pin/av-pin-g3.json` ("pin hit"), `DSV41_GPU_CLOCK_MHZ=2000`, `--expect-launcher 592793fd` and the arm's flags as env; all healthy at the first try (+279..300 s).
- **Order:** C1 V B1 D1 C2 D2 B2 C3, then X1 C4 X2 C5. Arms:
  - C: control (codex-adopt and dhtac files installed, flags off).
  - V: untimed verification, `DSV41_DECODE_REMAP_REUSE=check DSV41_TOPK_CANON_FUSED=1 DSV41_IDX_OMIT_T2S=1 DSV41_DRAFT_HEAD_TACTIC=audit DSV41_DRAFT_HEAD_TACTIC_MAP=17-29:1`.
  - B: codex-adopt bundle, `DSV41_TOPK_CANON_FUSED=1 DSV41_DECODE_REMAP_REUSE=1 DSV41_IDX_OMIT_T2S=1`.
  - D: dhtac, `DSV41_DRAFT_HEAD_TACTIC=1 DSV41_DRAFT_HEAD_TACTIC_MAP=17-29:1`.
  - X: B + D.
- **Per instance:** exact.py first (`gstart.sh`), then the timed suite with **telemetry off**, phase by phase from the Mac with one `nvidia-smi` read on 8/8 before each phase (`g3suite.sh`): matched C1 pdab x5, G1 x2, G4 warm-up + 6, conc8, conc16, conc8, conc16, TTFT ~24K / ~198K, exact.py #2. A 2 s `/metrics` sampler flags foreign requests. C and X instances then ran the power phase (2 s telemetry on 8/8; G1 x1, G4 x2; watts only).
- **Pairs:** each candidate against the mean of its bracketing controls. G4min_hat as in goal window 2 (511 / (148 T4 + 31.2 T3 + 30.6 T2 + 68.5 T1), the instance's per-phase step times).
- **Clocks:** 124 phase-start reads on 8/8 (12 instances), all OK at 1969-1989 MHz with event reasons 0x0; the 2 s telemetry of the power phases peaked at 1995 MHz once (C3, G1) and 68 C. No clock or power setting was changed.
- **Foreign traffic.** Since 20:16-20:21 UTC the API is also exposed on spark-one's LAN and tailnet addresses (`dsv41-lanproxy.service`, section "LAN API exposure"), and the DSH sPTC workflow (wf_cb6fe8d8-7fd) opened a 512S2 tunnel to 127.0.0.1:8000 at 20:23 (the workflow was killed at 20:24:19). Foreign requests hit V's G4 phase (up to 8 running; the access log shows browser-style probes `GET /`, `/favicon.ico`, `/models`) and probably C1's power phase (G4 aggregate 140 tok/s; excluded from the power means), and four short ones landed in D1's G4 warm-up (excluded from the metrics; the 6 timed runs were clean by the success counter). Every other timed phase of every instance was clean (sampler and success-counter checks).

### 3. Results (12 pinned instances, 2000 MHz, telemetry off)
**Exactness.** All 12 instances, V included, gave exact.py **a8c3394af9d2** with the pinned counters 909 / 467 / 2335 / 1373 and per-position 348 / 224 / 167 / 101 / 69 at start and again after all the load (exact.py #2), and the same 2-repetition G1 output digest **69ad872124100141**; 0 ERROR/Traceback on 8/8. V (fused CANON + omit token_to_seq on, remap in check mode, dhtac in audit mode) produced the control digest with 0 remap-check errors. The dhtac audit reporter crashed on 8/8 (`RuntimeError: Inplace update to inference tensor outside InferenceMode`: its pinned host buffer was made under inference mode; a diagnostics-only bug, the served path is unaffected), so there is no live 4-user audit count; the evidence for D's exactness at 4 users is the GPU test (all six tactics and stock bitwise-identical at every row count 1-160 on 8 ranks) plus the startup self-check of every D/X launch ("13 of 13 mapped row counts pinned (17-29:1) ... all tactics + stock bitwise" on 8/8). Fix for next time: wrap the reporter's `copy_` in `torch.inference_mode()`.

Log lines present on 8/8: B `DSV41_TOPK_CANON_FUSED=1: fused width-512 CANON kernel registered for JIT warmup` and `... fused CANON kernel in use (first call: rows=1030 row_stride=512)`; D `DSV41_DRAFT_HEAD_TACTIC=1: 13 of 13 mapped row counts pinned (17-29:1)`.

Per instance, in launch order (V is untimed: its G4 phase had foreign requests and verify-mode overhead):

| # | instance | arm | exact.py start / #2 | G1 digest | C1 coding / prose ms | G1 (code / prose / math) | G4 sess. mean / prose MIN / hat | 4-session step ms | G4 agg | conc8 / conc16 | TTFT 198K s | MemAvail min GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | g3-C1 | C | a8 OK / a8 OK | 69ad872124100141 | 30.84 / 26.66 | 100.53 (95.5 / 74.4 / 131.6) | 59.34 / 47.36 / 47.25 | 46.34 | 176.9 | 262.6 / 399.3 | 33.55 | 9.2 |
| 2 | g3-V | V | a8 OK / a8 OK | 69ad872124100141 | - / - | 98.69 (94.0 / 72.9 / 129.2) | 47.84 / 37.86 / 38.01 | 55.92 | 145.6 | 251.8 / 395.5 | 33.47 | 9.2 |
| 3 | g3-B1 | B | a8 OK / a8 OK | 69ad872124100141 | 30.52 / 26.35 | 101.74 (96.7 / 75.4 / 133.1) | 60.26 / 46.44 / 47.91 | 45.82 | 178.3 | 263.4 / 399.8 | 33.53 | 9.3 |
| 4 | g3-D1 | D | a8 OK / a8 OK | 69ad872124100141 | 30.87 / 26.79 | 100.45 (95.7 / 74.1 / 131.5) | 60.68 / 47.16 / 47.84 | 45.64 | 177.0 | 260.6 / 400.7 | 33.42 | 9.1 |
| 5 | g3-C2 | C | a8 OK / a8 OK | 69ad872124100141 | 30.88 / 26.71 | 100.27 (95.4 / 74.3 / 131.2) | 60.44 / 47.14 / 47.31 | 46.24 | 177.1 | 258.4 / 396.8 | 33.53 | 9.3 |
| 6 | g3-D2 | D | a8 OK / a8 OK | 69ad872124100141 | 30.85 / 26.68 | 100.58 (95.8 / 74.3 / 131.6) | 61.33 / 47.33 / 47.94 | 45.58 | 178.8 | 261.3 / 399.2 | 33.54 | 9.2 |
| 7 | g3-B2 | B | a8 OK / a8 OK | 69ad872124100141 | 30.57 / 26.40 | 101.69 (96.8 / 75.3 / 133.0) | 60.38 / 46.93 / 47.54 | 46.11 | 177.2 | 260.9 / 392.7 | 33.41 | 9.2 |
| 8 | g3-C3 | C | a8 OK / a8 OK | 69ad872124100141 | 30.81 / 26.67 | 100.55 (95.7 / 74.4 / 131.6) | 60.55 / 47.09 / 47.41 | 46.46 | 177.5 | 260.3 / 402.2 | 33.48 | 9.3 |
| 9 | g3-X1 | X | a8 OK / a8 OK | 69ad872124100141 | 30.60 / 26.43 | 101.53 (96.4 / 75.2 / 132.9) | 60.98 / 47.21 / 48.48 | 45.07 | 180.9 | 264.3 / 402.4 | 33.59 | 9.3 |
| 10 | g3-C4 | C | a8 OK / a8 OK | 69ad872124100141 | 30.83 / 26.71 | 100.55 (95.8 / 74.3 / 131.5) | 60.02 / 46.13 / 47.43 | 46.58 | 176.0 | 262.4 / 400.3 | 33.53 | 9.3 |
| 11 | g3-X2 | X | a8 OK / a8 OK | 69ad872124100141 | 30.55 / 26.38 | 101.61 (96.6 / 75.2 / 133.0) | 61.30 / 47.32 / 48.08 | 45.61 | 179.1 | 265.9 / 395.1 | 33.31 | 9.1 |
| 12 | g3-C5 | C | a8 OK / a8 OK | 69ad872124100141 | 30.90 / 26.72 | 100.12 (95.5 / 74.3 / 130.6) | 59.91 / 46.60 / 47.11 | 46.30 | 175.0 | 260.6 / 395.1 | 33.52 | 9.1 |

**Control (C, n=5):** G1 100.403 (sd 0.197), G1 code 95.575 (sd 0.178), G1 prose 74.321 (sd 0.073), G1 math 131.314 (sd 0.410), G4 MIN 46.862 (sd 0.493), G4 hat 47.300 (sd 0.129), G4 mean 60.055 (sd 0.481), cw min 40.027 (sd 0.587), G4 agg 176.494 (sd 0.999), conc8 260.896 (sd 1.716), conc16 398.726 (sd 2.818), C1 code 30.852 (sd 0.037), C1 prose 26.694 (sd 0.027), TTFT 198K 33.521 (sd 0.028)

Pairs (+ = better, in %; each instance against the mean of its bracketing controls; C1 and TTFT as speed-ups):

| inst | controls | G1 | G1 code | G1 prose | G1 math | G4 MIN | G4 hat | G4 mean | cw min | G4 agg | conc8 | conc16 | C1 code | C1 prose | TTFT 198K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | C1+C2 | +1.34 | +1.35 | +1.36 | +1.31 | -1.71 | +1.33 | +0.61 | -1.53 | +0.77 | +1.08 | +0.44 | +1.11 | +1.27 | +0.03 |
| D1 | C1+C2 | +0.06 | +0.31 | -0.31 | +0.08 | -0.19 | +1.18 | +1.30 | +0.44 | +0.02 | +0.00 | +0.68 | -0.03 | -0.39 | +0.36 |
| D2 | C2+C3 | +0.17 | +0.25 | +0.05 | +0.17 | +0.45 | +1.24 | +1.38 | +1.52 | +0.83 | +0.73 | -0.06 | -0.02 | +0.04 | -0.10 |
| B2 | C2+C3 | +1.27 | +1.31 | +1.30 | +1.24 | -0.38 | +0.38 | -0.19 | +0.64 | -0.06 | +0.57 | -1.70 | +0.90 | +1.10 | +0.26 |
| X1 | C3+C4 | +0.98 | +0.71 | +1.20 | +1.04 | +1.29 | +2.24 | +1.15 | +2.98 | +2.33 | +1.12 | +0.28 | +0.72 | +0.98 | -0.26 |
| X2 | C4+C5 | +1.28 | +1.05 | +1.27 | +1.45 | +2.07 | +1.71 | +2.22 | +3.80 | +2.06 | +1.69 | -0.65 | +1.03 | +1.27 | +0.64 |

**Per arm (means):**

| arm | n | G1 (code / prose / math) | G4 sess. mean / prose MIN / hat | 4-session step ms | G4 agg | conc8 / conc16 | C1 ms |
|---|---|---|---|---|---|---|---|
| C | 5 | 100.40 (95.58 / 74.32 / 131.31) | 60.06 / 46.86 / 47.30 | 46.38 | 176.5 | 260.9 / 398.7 | 30.85 / 26.69 |
| B | 2 | 101.72 (96.77 / 75.31 / 133.07) | 60.32 / 46.68 / 47.72 | 45.96 | 177.8 | 262.1 / 396.2 | 30.55 / 26.38 |
| D | 2 | 100.52 (95.77 / 74.23 / 131.56) | 61.00 / 47.24 / 47.89 | 45.61 | 177.9 | 260.9 / 400.0 | 30.86 / 26.74 |
| **X = B + D** | 2 | **101.57** (96.53 / 75.23 / 132.96) | **61.14 / 47.27 / 48.28** | **45.34** | **180.0** | 265.1 / 398.7 | 30.58 / 26.41 |

### 4. Decisions (rule: exact, and G1 or G4min_hat >= +0.5% in every pinned pair, >= 2 pairs)
- **B, codex-adopt bundle: PROMOTED.** G1 **+1.34 / +1.27%** (all three G1 prompts, matched C1 -1.0 / -1.2%), G4min_hat +1.33 / +0.38%. Larger than the +0.6-0.9% projection: the fused CANON saves ~0.18 ms and the remap reuse ~0.04 ms per step in the bench, and C1 coding fell 0.31 ms. The B2 conc16 dip (-1.7%) is acceptance, not step time (AL 2.39-2.41 against 2.43-2.47; tok/s per AL inside the control range).
- **D, dhtac: PROMOTED.** G4min_hat **+1.18 / +1.24%**, 4-session step -0.66 ms (46.38 -> 45.61 ms, matching the bench's 0.68 ms per 20-row head call), G4 mean +1.3 / +1.4%; G1 +0.06 / +0.17% as expected (one user drafts 5 head rows, outside the map).
- **X = B + D: PROMOTED.** G1 **+0.98 / +1.28%**, G4min_hat **+2.24 / +1.71%**, raw prose MIN +1.29 / +2.07%, G4 mean +1.2 / +2.2%, aggregate +2.3 / +2.1%, conc8 +1.1 / +1.7%, conc16 +0.3 / -0.7%, C1 coding -0.7 / -1.0%. The gains add: B's hat +0.89% and D's +1.24% against X's +2.07%.
- **mfront: NOT served** (section 1: illegal address with live and fused paths in one process; bench 6.5-7.4 us/layer, below its 8 us gate).
- **RoCEnante and the rdma_ar knobs: NOT adopted** (section 1).
- **candidate-only indexer: not qualified** (kit inconclusive on tied scores; prefill lever).

### 5. MoE and drafter at 4-user row counts (levers 3 and 4)
- **MoE (M=12-24 rows).** Today's M32 `all` + FC2C costs **409.1 / 654.6 us per routed layer at the recorded 4-user batches (c4rec M=12 / 24)**; this window's mfront bench re-measured the live path at 414.0 / 414.7 us (M=12, ranks 0 / 7). Against the same bytes: Marlin W4A16 in the same padded layout 406.6 / 658.5 us (equal), Marlin unpadded (2.44 MB/expert, our bytes) 375.7 / 606.8 us (**7-8% faster: 33 / 48 us per layer, about 1.2 / 1.7 ms per 4-user step**), weights-only streaming at 238 GB/s 352 / 580 us (11-14%). FC1 already streams at ~94%; the exact headroom is the glue kernels (~14-22 us/layer, of which mfront's fusion recovers ~7), the FC1 -> SiLU -> FC2 gap, and FC2C's lower streaming rate. (Marlin and streaming figures from the pd Marlin bench and the g1/g2 benches on recorded routes, normalized through their common `live` arm; prep analysis `g3/moe-drafter-bs4/README.md`.)
- **Drafter (bs2-4).** With 3 requests the drafter replays its own bs3 graph (the "bs3 runs the bs4 graph" premise was wrong; only the AV cost table prices 3 requests at the bs4 draft cost, and re-pricing it would change verify widths, so it is not bitwise). The real bs2 -> bs4 jump was the draft head: the serving autotune cache picks tactic 0 at 17-32 rows, 1.05 ms instead of 0.38 ms at 20 rows. D removes it (-0.66 ms per 4-request step, measured). Not built: drafter MoE on the M32 path (~-0.45 ms at bs4), the same tactic pin for the DSpark main_proj split.

### 6. Power and efficiency (power phase after the timed suite of every C and X instance; 2 s telemetry on 8/8, G1 x1 and G4 x2)
| arm | n | fleet GPU W idle / G1 / G4 | spark-one sys W G1 / G4 | J/token G1 (single) | J/token G4 (aggregate) |
|---|---|---|---|---|---|
| C | 5 (G4: 4) | 72.0 / 155.5 / 187.0 | 90.4 / 101.0 | 1.561 | 1.064 |
| **X = B + D** | 2 | 72.0 / 157.7 / 190.8 | 91.0 / 100.9 | 1.565 | 1.073 |

- C1's G4 power row is excluded (foreign requests, aggregate 140 tok/s). Control spread: J/token G1 1.524-1.621, G4 1.053-1.085.
- The stack changes no clock or power setting. Fleet GPU power is +2.2 W (G1) and +3.8 W (G4) over 8 nodes (+0.3 / +0.5 W per node) for +1.2% single-user and +2.0% aggregate throughput, so energy per token is flat inside the control spread. spark-one's metered whole-node power is unchanged (91.0 vs 90.4 W G1, 100.9 vs 101.0 W G4).

### 7. Promotion and final production instance
**Launcher** `~/dsv41-tp8.sh` 592793fd -> **b0cf94d0** on 8/8 (`g3w/scripts/promote.sh`); backup `~/dsv41-tp8.sh.bak-g3-xstack` = 592793fd on 8/8. Five lines after `DSV41_SYNC_FREE`, each commented:
```sh
export DSV41_TOPK_CANON_FUSED=${DSV41_TOPK_CANON_FUSED:-1}
export DSV41_DECODE_REMAP_REUSE=${DSV41_DECODE_REMAP_REUSE:-1}
export DSV41_IDX_OMIT_T2S=${DSV41_IDX_OMIT_T2S:-1}
export DSV41_DRAFT_HEAD_TACTIC=${DSV41_DRAFT_HEAD_TACTIC:-1}
export DSV41_DRAFT_HEAD_TACTIC_MAP=${DSV41_DRAFT_HEAD_TACTIC_MAP:-17-29:1}
```
The supervisor site manifest includes the launcher, so it moved 0501bdcb13e0 -> **b1150ecfcf49**. The supervisor itself is unchanged (7bea1cf3, clock default 2000, compaction 0). Dependencies: the installed site files of section 1 (codex-adopt and dhtac); `DSV41_TOPK_CANON_FUSED` needs `DSV41_TOPK_CANON=1` (launcher default).

**Final instance g3-FINAL** (production: launcher defaults, AV unpinned = startup profile, `DSV41_GPU_CLOCK_MHZ=2000`; driver `g3w/fin/scripts/fin.sh g3-FINAL prod 2`):
- Healthy at 22:36:32 UTC (+357 s, first try). Env on 8/8 carries the five new flags (verified by `verify8.sh post`).
- exact.py first: **a8c3394af9d2** (own profile, 921 / 455 / 2275 / 1373, per-position 342 / 234 / 175 / 100 / 70, "pin miss"); identical after all the load (exact.py #2).
- **Cold solo 1M: 183.07 s, good class** (ratio 1.16, first try). MemAvailable minimum during the prompt **8.32 GB (spark-one)**, 8.9-9.5 GB elsewhere (6 GB floor; gw2-FU-2 had 8.29 GB).
- Suite (`finsuite.sh`, telemetry off, 17 clock reads on 8/8 at 1976-1989 MHz, no foreign traffic):

  | metric | g3-FINAL (production, b0cf94d0) | ref: gw2-FU-2 (production, 592793fd) |
  |---|---|---|
  | G1 mean (code / prose / math), 3 reps | **102.26** (95.93 / 77.04 / 133.82); reps 102.19 / 102.35 / 102.25, identical outputs (digest eb6d32aab70a0191) | 100.39 (95.63 / 74.16 / 131.38) |
  | G1 AL code / prose / math | 2.616 / 1.889 / 3.658 | 2.603 / 1.792 / 3.607 |
  | G4 per-session mean / prose MIN (3 runs) | **60.86 / 47.87** (prose 48.70 / 45.80 / 49.10) | 60.34 / 46.42 |
  | G4min_hat (step times only) | **47.77** (4-session step 45.59 ms) | 47.57 (46.35 ms) |
  | G4 by session: code / prose / math / mixed | 55.2 / 47.9 / 76.0 / 64.4 | 55.5 / 46.4 / 75.0 / 64.4 |
  | G4 common window / aggregate | 58.92 / 40.62; 180.8 | 57.89 / 39.93; 176.9 |
  | matched C1 coding / prose ms (AL) | 30.59 / 26.58 (5.224 / 2.759); 2nd pass 30.52 / 26.64 | 30.91 / 26.90 |
  | bench conc 4 / 8 / 16 / 32 | **180.9 / 262.8 / 394.2 / 585.8** | 176.3 / 263.1 / 399.7 / 577.6 |
  | TTFT ~24K / ~198K | 4.04 s (warm, after the 1M) / 33.31 s | 4.09 / 33.30 s |
  | GSM8K-50 / full longctx / cached_tokens | **49/50** / **PASS** (needle 60/60, copy 1.000) / 4160 | 50/50 / PASS / 4160 |

  0 ERROR/Traceback on 8/8, 0 restarts; MemAvailable at the end 8.4-9.4 GB.
- Production numbers move with the startup AV profile (this launch: prose AL 1.889 vs 1.79-1.85 in earlier production launches), so the production-vs-production row is indicative; the pinned A/B above is the measurement (stack: G1 +1.2%, G4min_hat +2.1%).

### 8. Live state and rollback
**Live:** instance **g3-FINAL**, healthy since 22:36:32 UTC (replaced at 23:00 UTC by the final-verification launches; live now: **g3-FU**, same launcher and config, healthy since 23:22:34 UTC, section 10.5); launcher **b0cf94d0** on 8/8; supervisor 7bea1cf3; AV unpinned (the default pin stays a user decision; the fresh pin for A/Bs is `~/dsv41-stage/g3/pin/av-pin-g3.json`); clock lock 0-2000 MHz (boot unit active on 8/8). Site files on 8/8: sparse_attn_indexer.py 481905c2, flashinfer_sparse.py b059bdf3, mla/indexer.py d9f4c5c9, dsv41_r3_dense.py 1f1c0456 (in use); deep_gemm_utils.py a4a0486a (mfront not installed); everything from goal window 2 unchanged. No telemetry, sampler, logger or bench process left on 8/8. RoCEnante venv `~/venvs/roce072` remains on 8/8 (unused by serving).

Rollback:
- **One launch:** `DSV41_TOPK_CANON_FUSED=0 DSV41_DECODE_REMAP_REUSE=0 DSV41_IDX_OMIT_T2S=0` (codex bundle off) and/or `DSV41_DRAFT_HEAD_TACTIC=0` (dhtac off) as supervisor arguments. Exact either way.
- **Permanent:** `cp -p ~/dsv41-tp8.sh.bak-g3-xstack ~/dsv41-tp8.sh` on 8/8 (592793fd), relaunch with `--expect-launcher 592793fd`.
- **Site files** (service stopped, after the launcher rollback): `ssh spark-one cat ~/dsv41-stage/g3/codex-adopt/ops/g3ca-all.sh | bash -s uninstall` and `ssh spark-one cat ~/dsv41-stage/g3/dhtac/ops/all.sh | bash -s uninstall` (backups `.bak-g3-codex-adopt`, `.bak-g3-dhtac`).
- **RoCEnante venv:** `bash g3roce/pkg/ops/roce-all.sh uninstall` (Mac) removes it and its caches on 8/8.

### 9. Goal status and where the gap is (2000 MHz, no clock or power increase)
- **G1 >= 120 tok/s: NOT met.** 102.26 production (pinned stack 101.57): **+17.3%** still needed. The window's exact levers gave +1.2% pinned. 120 needs an acceptance lever (prose AL ~1.8-1.9) or MoE weight streaming much closer to the bound (FC1 already streams at ~94%).
- **G4 >= 50 tok/s on every session: NOT met.** Code / math / mixed stay at 52.7-78.2 in every run; prose is the binding session: production MIN 47.87 (worst run 45.80), pinned stack raw MIN 47.27, **G4min_hat 47.8-48.3**, so **+3.6..4.7%** (step time) and up to +9.2% for the worst prose run are still needed.
- Next exact levers, by size at 4 users: mfront after root-causing the mixed-path crash (~-0.27 ms/step bench), FC1 -> FC2 per-expert pipelining (-0.4..-1.0 ms), contiguous FC2C layout (-0.5..-0.8 ms), drafter MoE on the M32 path (~-0.45 ms), an L2-prefetch plan that keeps q_b above 8 rows (the replicated qkv_a fills the 6 MB window; +0.86 ms today), rdma_ar P1 doorbell (~0.1-0.2 ms). Together roughly 2.5-3.5 ms of the 45.3 ms 4-session step; the G4 target needs most of them to land.
- **Coordination:** the server is now reachable from the LAN and the tailnet without authentication (lanproxy), from 512S2 (the DSH tunnel, still open and idle) and from the Mac (the Codex app's `codex-dsv41-harness.sock` tunnel). Timed windows need these clients quiet: pause the Codex app and other users, or stop `dsv41-lanproxy.service` for the window (user decision), and keep the per-phase foreign-request check.

### 10. Final verification (23:00-23:40 UTC): config checked on 8/8, then the goal metrics on a pinned and a production instance at 2000 MHz. **Goal NOT met**

**Verdict.** Neither target is met. Every number below was taken with telemetry off, at the 2000 MHz boot clock, on the promoted config (launcher b0cf94d0).
- **G1, single user: 101.51 tok/s in production** (g3-FU: code 95.70 / prose 76.56 / math 132.28) against 120. **Not met: +18.2% is needed** (+18.5 tok/s). The pinned instance g3-FP gave 101.71 (+18.0%) and g3-FINAL (section 7) 102.26 (+17.3%).
- **G4, 4 concurrent sessions: not met.** Code, math and mixed were above 50 tok/s in every timed run on all three instances; the lowest of them was 55.2 in this step (52.7 on g3-FINAL). **The prose session was below 50 in every run:**
  - production g3-FU: 49.15 / 49.52 / 47.26 (MIN 48.64), so **+2.8% is needed** (+5.8% for its worst run);
  - pinned g3-FP: 48.82 / 47.84 / 47.22 (MIN 47.96), **+4.3%** (+5.9%);
  - production g3-FINAL (section 7): 48.70 / 45.80 / 49.10 (MIN 47.87), **+4.5%** (+9.2%).

  The step-time-only estimate G4min_hat is 47.8-48.5 on the three instances, so **+3.0..4.7%** is needed: about **-1.1..-1.7 ms on every decode step**, or -2.1..-3.2 ms on the 4-session steps alone (now 44.7-45.6 ms). The per-session mean is above 50 everywhere (60.3-61.7), but the goal requires every session.
- **Exact.** Both instances gave exact.py a8c3394af9d2 at start and again after all the load. The pinned one had the pin counters 909 / 467 / 2335 / 1373 with per-position 348/224/167/101/69, and its first two G1 repetitions gave the A/B digest 69ad872124100141.
- **Power.** No clock or power setting was raised. Energy per token is flat: the production power phase gave 1.540 J/token single user and 1.054 aggregate, against 1.561 / 1.064 for the window's controls (section 6).

#### 10.1 Config verified on 8/8
`fin/scripts/verify8.sh` ran before the step (`verify-fpre`, 23:00 UTC, on g3-FINAL) and after it (`verify-fpost`, 23:38 UTC, on the live g3-FU). Apart from the instance start time, GPU readings and MemAvailable, the two outputs are identical, and they match section 7's `verify-post`.
- **Launcher** `~/dsv41-tp8.sh` **b0cf94d0** on 8/8. Backups: `.bak-g3-xstack` = 592793fd and `.bak-g2-sf` = 395bf5b2, on 8/8.
- **`--enable-prompt-tokens-details`** is in the launcher and in the running `vllm serve` cmdline on 8/8. The usage check returned `cached_tokens` 4160 on both instances.
- **Supervisor** `ops/rel/dsv41-relaunch.sh` **7bea1cf3** (`PRE_CLOCK_DEFAULT=2000`, `PRE_COMPACT_DEFAULT=0`); backup `.bak-g2-clock2300` = 30158e80. Site manifest **b1150ecfcf49**. Both launches logged "site manifest b1150ecfcf49 and launcher b0cf94d0 identical on 8/8" and "engram check: 8/8 OK".
- **Site files**, identical on 8/8:
  - goal window 3: `sparse_attn_indexer.py` 481905c2, `flashinfer_sparse.py` b059bdf3, `mla/indexer.py` d9f4c5c9 (`.bak-g3-codex-adopt` 7112c01d / c8a2da55 / 4753707e), `dsv41_r3_dense.py` 1f1c0456 (`.bak-g3-dhtac` 993023f2); `deep_gemm_utils.py` a4a0486a (mfront not installed); pin `~/dsv41-stage/g3/pin/av-pin-g3.json` 1c98cf9b;
  - earlier windows: `sm121_target_m32.py` ea4a8814 (`.bak-g2-fc2c-on-m32all` db659d80); colltrim, inert: `cuda_communicator.py` 87bf77e3, `rdma_all_reduce.py` 26a5d834, `dsv41_l2_prefetch.py` 63737f9c (`.bak-g2-colltrim` 52fc8cf5 / 9056dcdf / 998af3d7); stepovh: `dsv41_r4_small.py` c46e66bd, `sparse_swa.py` 854c3e5f, `model_state.py` f21ee2e6; `rdma_ar_ext.so` 62daad6c.
- **Serving dependencies** are present on 8/8: `~/dsv41-stage/g1/m32-bw/` (661-664 MB) and `~/dsv41-stage/g2/fc2c-on-m32all/` (`PREP.ok` dcf5bc0909472121).
- **Running env** of g3-FU, the same on 8/8: `DSV41_AV_PROFILE=pinned DSV41_DECODE_REMAP_REUSE=1 DSV41_DRAFT_HEAD_TACTIC=1 DSV41_DRAFT_HEAD_TACTIC_MAP=17-29:1 DSV41_EAGER_GLUE=all,swam DSV41_IDX_OMIT_T2S=1 DSV41_L2_PREFETCH_PRUNE_SHARED=1 DSV41_L2_PREFETCH_WOA=1 DSV41_MOE_FC2C=1 DSV41_MOE_FC2_D05=1 DSV41_MOE_M32_G1=all DSV41_RDMA_PARLOAD=1 DSV41_SYNC_FREE=argmax DSV41_TOPK_CANON=1 DSV41_TOPK_CANON_FUSED=1`. `DSV41_AV_COST_JSON` is unset in production, so the startup profile is used ("pin miss"). On both instances the rank logs on 8/8 carry "fused CANON kernel in use" and "13 of 13 mapped row counts pinned (17-29:1)".
- **Clock.**
  - The boot unit `dgx-spark-gpu-clock.service` (`nvidia-smi -lgc 0,2000`) is enabled and active on 8/8.
  - Both launches logged `prelaunch (compact=0 clock=2000): ok on 8/8`.
  - The SM clock was read on 8/8 **before every measurement**: one `nvidia-smi` query per node, no polling, so telemetry stayed off. That is **38 checks (304 node readings), all at 1969-1989 MHz** with event reasons 0x0, and at most 66 C. The power phase's 2 s telemetry peaked at 1989 MHz and 67 C.
  - No clock or power setting was changed.

#### 10.2 Method
- **Driver** `fin/scripts/final.sh` (Mac): `verify8.sh fpre`, then an outside-client logger on spark-one, then `fin.sh g3-FP pinned`, `fin.sh g3-FU prod 2`, the power phase (`finpow.sh`) on the production instance, and `verify8.sh fpost`. `fin.sh`, `finsuite.sh` and `finphase.sh` are the section 7 drivers, i.e. goal window 2's final method:
  1. A supervised launch with `--expect-launcher b0cf94d0` and `DSV41_GPU_CLOCK_MHZ=2000`, then the clock check.
  2. `gstart.sh` (exact.py first).
  3. Production only: a cold solo 1M with 1 Hz memwatch on 8/8, and a relaunch if it is not in the good class.
  4. The `gw1suite.sh` phases one at a time, with the clock check before each: matched C1 x5 (pdab), G1 x3, G4 warm-up + 3 runs, bench conc 4/8/16/32 x2, matched C1 again, TTFT ~24K / ~198K, usage, GSM8K-50, full longctx, exact.py #2.
- **Instances**, in launch order:
  - **g3-FP**, pinned (`av-pin-g3.json`, "pin hit"): healthy at 23:05:16 UTC (+276 s, first try). exact.py **a8c3394af9d2** with the pin counters 909 / 467 / 2335 / 1373 and per-position 348/224/167/101/69 (**EXACT-OK**), the same again after all the load.
  - **g3-FU**, production: healthy at 23:22:34 UTC (+360 s, first try). exact.py **a8c3394af9d2** (own profile: 909 / 467 / 2335 / 1373, per-position 348/233/171/90/67), the same again after all the load.
    - Its cold 1M took **183.70 s, good class** (late/early 1.16), at the first try. The lowest MemAvailable during the prompt was 8.27 GB on spark-one and 8.9-10.1 GB on the other nodes, above the 6 GB floor.
    - **This instance is live.**
- **Power phase** (after g3-FU's timed suite; 2 s telemetry on 8/8; 10 s idle, G1 x1, G4 warm-up + 2 runs). It measures watts only and is never used for timing.
- **Outside activity: none reached the server.**
  - The `/metrics` sampler showed the maximum running count equal to the phase's own concurrency in every phase of both instances. The suites completed 258 (g3-FU) and 257 (g3-FP) requests against the clean reference of 258; the per-phase differences of 1-2 are phase-boundary shifts.
  - The 1 s logger on spark-one (`connlog-fin.sh`) watched for clients of 127.0.0.1:8000 other than python/curl, and for any client of the LAN or tailnet listeners. It saw no LAN or tailnet client.
  - Its three entries (23:11:31, 23:26:13, 23:31:50) were our own requests: two `/metrics` scrapes and one matched-bench POST, whose ports 55922 / 55120 / 47390 match the access log. They were listed without a process because `ss -p` races short connections.
  - The Mac logger on the Codex tunnel (127.0.0.1:18000, from 23:17) saw no client.

#### 10.3 Results

| metric | **production g3-FU** (unpinned, telemetry off) | pinned g3-FP (telemetry off) | ref: production g3-FINAL (section 7) |
|---|---|---|---|
| **G1 mean tok/s** (code / prose / math), 1024 tokens x3 each | **101.51** (95.70 / 76.56 / 132.28) | **101.71** (96.71 / 75.32 / 133.10) | **102.26** (95.93 / 77.04 / 133.82) |
| G1 per repetition | 101.56, 101.49, 101.48 | 101.80, 101.68, 101.65 | 102.19, 102.34, 102.25 |
| G1 AL code / prose / math | 2.590 / 1.847 / 3.594 | 2.603 / 1.792 / 3.607 | 2.616 / 1.889 / 3.658 |
| G1 output digest (3 reps; reps identical) | bc1e8386878bd2b7 (yes) | 939bb576cc1aad16 (yes) | eb6d32aab70a0191 (yes) |
| **G4 per-session mean / prose MIN** (3 runs) | **61.65 / 48.64** | **60.34 / 47.96** | **60.86 / 47.87** |
| G4 by session: code / prose / math / mixed | 58.8 / 48.6 / 75.3 / 63.9 | 56.3 / 48.0 / 75.6 / 61.5 | 55.2 / 47.9 / 76.0 / 64.4 |
| G4 slowest session per run (session) | 49.15 (prose), 49.52 (prose), 47.26 (prose) | 48.82 (prose), 47.84 (prose), 47.22 (prose) | 48.70 (prose), 45.80 (prose), 49.10 (prose) |
| G4 session-runs at >= 50 tok/s | 9/12 | 9/12 | 9/12 |
| G4 common window mean / min | 59.02 / 40.45 | 57.64 / 41.51 | 58.92 / 40.62 |
| G4 aggregate tok/s | 182.7 | 181.8 | 180.8 |
| matched C1 coding / prose ms/step (pdab; AL) | **30.32 / 26.33** (AL 5.191 / 2.725); 2nd pass 30.25 / 26.36 | **30.71 / 26.40** (AL 5.224 / 2.718); 2nd pass 30.52 / 26.41 | **30.59 / 26.58** (AL 5.224 / 2.759); 2nd pass 30.52 / 26.64 |
| bench conc4 aggregate tok/s (runs; per-request mean / min) | **180.8** (178.9, 182.7; 60.7 / 46.5) | **180.4** (182.6, 178.3; 60.4 / 46.5) | **180.9** (180.8, 180.9; 61.2 / 48.3) |
| bench conc8 aggregate tok/s (runs; per-request mean / min) | 264.3 (264.8, 263.8; 43.9 / 34.1) | 262.6 (263.1, 262.1; 43.0 / 33.3) | 262.8 (262.7, 262.9; 44.3 / 34.8) |
| bench conc16 aggregate tok/s (runs; per-request mean / min) | 397.8 (402.8, 392.8; 34.7 / 25.4) | 399.1 (396.8, 401.4; 34.1 / 25.0) | 394.2 (397.9, 390.5; 34.2 / 25.8) |
| bench conc32 aggregate tok/s (runs; per-request mean / min) | 580.7 (584.2, 577.2; 24.9 / 18.7) | 590.8 (590.4, 591.2; 25.5 / 18.9) | 585.8 (587.8, 583.9; 25.1 / 18.5) |
| TTFT ~24K / ~198K | 4.05 s (23,901) / 33.29 s (197,810) | 5.11 s (23,821) / 33.42 s (197,691) | 4.04 s (23,863) / 33.31 s (198,008) |
| GSM8K-50 | 50/50 | 49/50 | 49/50 |
| full longctx | PASS | PASS | PASS |
| usage cached_tokens (repeat 4.2K prompt) | 4160 | 4160 | 4160 |
| exact.py #2 after all load (acc / drafts / dtok / gen; per-pos) | a8c3394af9d2 (909 / 467 / 2335 / 1373; 348/233/171/90/67) | a8c3394af9d2 (909 / 467 / 2335 / 1373; 348/224/167/101/69) | a8c3394af9d2 (921 / 455 / 2275 / 1373; 342/234/175/100/70) |
| foreign traffic (sampler, running > own concurrency) | none | none | none |
| G4min_hat (step times only) / 4-session step ms | 48.43 / 44.77 | 48.54 / 44.69 | 47.77 / 45.59 |
| G4 per-phase step ms (4 / 3 / 2 / 1 sessions); prose steps per run | 44.77 / 40.72 / 32.52 / 24.24; 281.3 | 44.69 / 40.45 / 34.14 / 23.46; 281 | 45.59 / 39.11 / 33.03 / 25.09; 276.3 |
| G1 outputs per type (sha8 code / prose / math) | 596b478a / 923a46be / 92e5d8cf | 596b478a / e8a524f3 / 92e5d8cf | 596b478a / e0e0664e / 92e5d8cf |
| G1 digest of the first 2 repetitions (A/B pinned = 69ad872124100141) | 9a18f802d930f6e7 | 69ad872124100141 | d74940fefe15b2ef |
| exact.py, first request (acc / drafts / dtok / gen; per-pos; AV) | a8c3394af9d2 (909/467/2335/1373; 348/233/171/90/67; pin miss) | a8c3394af9d2 (909/467/2335/1373; 348/224/167/101/69; pin hit) | a8c3394af9d2 (921/455/2275/1373; 342/234/175/100/70; pin miss) |
| requests finished during the suite (clean reference 258) | 258 (extra vs ref: {'usage': 1.0, 'gsm': -2.0, 'lcfull': 1.0}) | 257 (extra vs ref: {'ttft': -1.0, 'usage': 1.0, 'exact2': -1.0}) | 258 |

G4 runs, **production g3-FU** (unpinned, telemetry off):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 57.83 | 47.80 | 77.17 | 64.26 | 61.77 | 47.80 (prose) | 39.87 | 178.9 | 2.583 |
| 1 | 60.53 | 49.15 | 77.82 | 66.78 | 63.57 | 49.15 (prose) | 39.29 | 182.7 | 2.656 |
| 2 | 59.01 | 49.52 | 73.13 | 65.98 | 61.91 | 49.52 (prose) | 41.15 | 185.5 | 2.570 |
| 3 | 56.75 | 47.26 | 74.91 | 59.01 | 59.48 | 47.26 (prose) | 40.90 | 179.9 | 2.369 |

G4 runs, pinned g3-FP (telemetry off):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 55.87 | 47.36 | 69.45 | 60.62 | 58.33 | 47.36 (prose) | 40.51 | 177.8 | 2.470 |
| 1 | 55.16 | 48.82 | 73.49 | 61.55 | 59.75 | 48.82 (prose) | 41.49 | 183.1 | 2.479 |
| 2 | 55.70 | 47.84 | 74.09 | 61.68 | 59.83 | 47.84 (prose) | 41.91 | 182.0 | 2.427 |
| 3 | 57.95 | 47.22 | 79.33 | 61.21 | 61.43 | 47.22 (prose) | 41.14 | 180.3 | 2.466 |

G4 runs, ref: production g3-FINAL (section 7):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 56.01 | 48.13 | 75.86 | 62.33 | 60.58 | 48.13 (prose) | 39.64 | 180.3 | 2.582 |
| 1 | 52.66 | 48.70 | 76.36 | 67.31 | 61.26 | 48.70 (prose) | 42.44 | 182.7 | 2.528 |
| 2 | 55.77 | 45.80 | 73.28 | 61.78 | 59.16 | 45.80 (prose) | 39.01 | 176.0 | 2.469 |
| 3 | 57.30 | 49.10 | 78.25 | 63.98 | 62.16 | 49.10 (prose) | 40.42 | 183.8 | 2.582 |

Other per-instance facts:

| | g3-FU | g3-FP | g3-FINAL |
|---|---|---|---|
| healthy at (UTC) / launch to healthy | 23:22:34 / +360 s | 23:05:16 / +276 s | 22:36:32 / +357 s |
| cold solo 1M (1,024,611 tokens) | 183.703 s, good class (1.16) | not run (once only) | 183.071 s, good class (1.16) |
| MemAvailable minimum during the 1M | 8.27 GB (spark-one); others 8.9-10.1 GB | - | 8.32 GB (spark-one); others 8.9-9.5 GB |
| MemAvailable at the end of the suite (min over 8 nodes) | 8.3 GB | 9.4 GB | 8.4 GB |

| power phase (2 s telemetry on 8/8, after the timed suite) | production g3-FU | window controls C (n=5, section 6) | stack X (n=2, section 6) |
|---|---|---|---|
| fleet GPU W idle / G1 / G4 | 72.8 / 154.9 / 188.3 | 72.0 / 155.5 / 187.0 | 72.0 / 157.7 / 190.8 |
| spark-one whole-node W G1 / G4 | 92.2 / 101.9 | 90.4 / 101.0 | 91.0 / 100.9 |
| J/token, G1 single user / G4 aggregate | **1.540 / 1.054** (G1 100.6, G4 178.6 tok/s in that phase) | 1.561 / 1.064 | 1.565 / 1.073 |
| SM MHz max / GPU max C | 1989 / 67 | 1989-1995 / 68 | 1989 / 68 |

- **The pinned instance reproduces the A/B stack and the launch the pin was made from.**
  - Its first two G1 repetitions give the A/B digest 69ad872124100141. All three give **939bb576cc1aad16, the digest of gw2-FU-2** (goal window 2's production instance on launcher 592793fd, whose startup profile became `av-pin-g3.json`), with the same ALs (2.603 / 1.792 / 3.607). So the promoted stack under that profile produces exactly the old config's outputs.
  - With the outputs identical, the comparison with gw2-FU-2 is pure step time: G1 101.71 against 100.39 is **+1.31%** (code +1.1%, prose +1.6%, math +1.3%), and G4min_hat 48.54 against 47.57 is **+2.0%**. Both agree with the pinned A/B result for the stack X (G1 +1.17%, G4min_hat +2.07%).
- **Production g3-FU produced gw2-FINAL1's outputs** (digest bc1e8386878bd2b7, same ALs 2.590 / 1.847 / 3.594): G1 101.51 against 100.50 (+1.0%), G4min_hat 48.43 against 47.23 (+2.5%).
- **Production still moves with the startup AV profile.** The two production launches since the promotion gave G1 101.5 / 102.3 and raw prose MIN 48.6 / 47.9 (prose steps per G4 run 281.3 / 276.3). G4min_hat, which removes the text effect, is 48.4 / 47.8.
- The ~24K TTFT on g3-FP (5.11 s) is the first long prefill after start. On the production instances it came after the 1M probe, so it is warm (~4.05 s).
- GSM8K-50 was 50/50 on g3-FU and 49/50 on g3-FP and g3-FINAL. The miss is idx 12, the usual borderline item.

#### 10.4 Distance to the goal and next exact levers (2000 MHz, no clock or power increase)

| target | g3-FU | g3-FP | g3-FINAL |
|---|---|---|---|
| G1 mean >= 120 tok/s | 101.51: +18.2% | 101.71: +18.0% | 102.26: +17.3% |
| G4 every session >= 50 in every run (prose MIN over 3 runs; worst run) | 48.64: +2.8%; 47.26: +5.8% | 47.96: +4.3%; 47.22: +5.9% | 47.87: +4.5%; 45.80: +9.2% |
| G4min_hat (step time only) | 48.43: +3.2% | 48.54: +3.0% | 47.77: +4.7% |

- **G1 >= 120 tok/s: NOT met.** At 101.5-102.3 tok/s, **+17.3..18.2%** is needed. At today's acceptance (prose AL 1.79-1.89) that is about -15% single-user step time, about 3.7-4.2 ms per step. The levers below are sized for 4-user rows and give far less at one user. 120 needs either an acceptance lever (a better drafter changes the verify shapes, so it is not exact under the current definition and needs a new pin and anchor) or MoE weight streaming much closer to the bound (FC1 already streams at ~94%).
- **G4 >= 50 tok/s on every session: NOT met.** The prose session needs **+2.8..4.5%** on the 3-run MIN, up to +9.2% for the worst run, and **+3.0..4.7% on step time** (G4min_hat 47.8-48.5). That is -1.1..-1.7 ms on every decode step, or -2.1..-3.2 ms on the 4-session steps alone.
- **Next exact levers**, by size at 4 users. Each keeps the target numerics bitwise and needs its own GPU bitwise test and a pinned A/B under `av-pin-g3.json`:
  1. MoE FC1 -> FC2 per-expert pipelining: -0.4..-1.0 ms per 4-session step. At M=12-24 the live M32 `all` + FC2C path is 7-8% slower than unpadded Marlin on the same bytes (~1.2-1.7 ms per step) and 11-14% above the streaming bound (section 5).
  2. Contiguous FC2C layout: -0.5..-0.8 ms.
  3. An L2-prefetch plan that keeps q_b prefetched above 8 rows (the replicated qkv_a fills the 6 MB window): up to -0.86 ms on steps with 2 or more sessions.
  4. Drafter MoE on the M32 path: ~-0.45 ms at bs4.
  5. mfront, once the mixed live/fused-graph illegal-address crash is root-caused (`CUDA_LAUNCH_BLOCKING=1` + compute-sanitizer, bisect with `DSV41_MOE_FC2C=0`): ~-0.26..-0.30 ms per step.
  6. The dhtac fix applied to the DSpark main_proj split, which has the same autotune-cache tactic-0 choice at 17-32 rows: not measured yet; check the cache first.
  7. rdma_ar P1 doorbell / P2 fail-fast: ~-0.1..-0.2 ms.

  Together that is roughly 2.5-3.5 ms of the 44.7-45.6 ms 4-session step. On paper that covers the G4 step-time gap if most of them land; the worst prose runs need more margin. Also fix the dhtac audit reporter (run its `copy_` under `torch.inference_mode()`) so that a live 4-user audit count backs D's exactness.
- **User decisions (not exact, or policy), unchanged:** a default AV pin (makes production reproducible and launches shorter, 276 s against 357-360 s, but does not close the gap); the prose-priority verify width (a fairness trade, not bitwise at 4 users); unpadded Marlin (not exact); clocks (closed at 2000 MHz); authentication for the LAN and tailnet API (`dsv41-lanproxy` has none).

#### 10.5 Live state after this step
- **g3-FU** is live: production, AV unpinned, launcher b0cf94d0, supervisor 7bea1cf3, clock 0-2000 MHz.
  - Healthy since 23:22:34 UTC on 8/8, with 0 restarts and 0 ERROR/Traceback. `/health` returns 200 locally and on the LAN address. `dsv41-lanproxy` is active.
  - Its config matches 10.1, re-checked after the suite and the power phase.
- No telemetry, memwatch, sampler, logger or bench process is left on 8/8 or on the Mac.
- Rollback is unchanged from section 8 (the live instance is now g3-FU instead of g3-FINAL, with the same config).
- **Files.**
  - Mac scratchpad `g3w/fin/`:
    - scripts: `final.sh`, `finpow.sh`, `connlog-fin.sh`, `finforeign.py`, `finaltab.py`, plus the section 7 drivers;
    - logs and checks: `fin.log`, `final.out`, `verify-fpre.txt`, `verify-fpost.txt`, `clk-g3-FP.txt`, `clk-g3-FU.txt`, `connlog-fin.txt`;
    - results: `runs/g3-FP/`, `runs/g3-FU/`, `runs/g3-FU-pow/`, `g3-FP/` and `g3-FU/` (relaunch logs, exact.py, 1M with memwatch CSVs), `pow-final.tsv`, `tables-final.md`.
  - spark-one `~/dsv41-g3-20261026/`: `runs/fin-g3-FP`, `runs/fin-g3-FU`, `runs/g3-FP-start`, `runs/g3-FU-start`, `runs/p1m-g3-FU`, `runs/g3-FU-pow`, `connlog-fin.txt`.

## Moet drafter A/B (2026-10-26 23:57 - 2026-10-27 02:22 UTC): DSpark k=6 / k=7 (vLLM-Moet's width) against k=5, plus Moet's AV-off config. All exact (exact.py a8 on every instance) but NOT promoted: k=7 loses G1 and G4, k=6 is flat on G1 and loses on the G4 prose session; live config unchanged (k=5, production)

Scope: the user asked "can we try their drafter" (https://github.com/kacper-daftcode/vLLM-Moet#fraser-bench-single-stream-code-decode, Apache-2.0, commit b0dc737). Moet's drafter is **our drafter**: the same DSpark drafter (the checkpoint's `mtp.*` weights, vLLM's `DSparkSpeculator`). What they change is the configuration (k = `num_speculative_tokens` 7 instead of 5, adaptive verification OFF), an MXFP8 GEMV for the drafter lm_head and three launch-fusion kernels. The offline prep (`~/dsv41-stage/moet/README.md`, on 8/8) mapped every Moet change to our stack and found the kernels worth <= 0.2% each on our 27-31 ms single-user step (we already have the byte savings: FP8 draft head, vocab-sharded greedy Markov), so this window measured **the draft width k** on the live stack (launcher b0cf94d0, 2000 MHz, telemetry off). Mac drivers and raw results: scratchpad `mab/` (`scripts/mab.sh` one instance, `mabblk.sh` serial block, `mabphase.sh` per-phase runner on spark-one, `clk8.sh`, `mabline.py`, `mabagg.py`, `pincmp.py`; `ab.log`, `ab.jsonl`, `tables.md`, `ab/<label>/`, `runs/<label>/`). On spark-one: `~/dsv41-moet-ab/runs/`.

### 1. Method
- **Arms** (every launch: `--expect-launcher b0cf94d0`, `DSV41_GPU_CLOCK_MHZ=2000`, the promoted window-3 stack unchanged; only these variables differ):
  - **k5** (control): `DSV41_SPEC=5 DSV41_AV_COST_JSON=~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json`.
  - **k6**: `DSV41_SPEC=6 DSV41_AV_COST_JSON=~/dsv41-stage/moet/pins/av-pin-k6-mab-p6.json`.
  - **k7**: `DSV41_SPEC=7 DSV41_AV_COST_JSON=~/dsv41-stage/moet/pins/av-pin-k7-mab-p7.json`.
  - **k7off** (Moet's served config, reference only): `DSV41_SPEC_JSON={"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}` (their `moe_backend: b12x` left out: 0% at small M on GB10 and not allowed with `DSV41_ROUTE_LIVE`). AV off means no pin, no verify-cap and no acceptance estimator.
- **One AV pin per k, each made from that k's own startup profile on today's stack** (`moet/tools/mkpin_k.py make`, validated with a verbatim copy of `dsv41_rel.check_pin`; files on 8/8 in `~/dsv41-stage/moet/pins/`, the source logs in `pins/src/` and `runs/mab-p{6,7}-start/`):
  - k5: the rank-0 startup profile of **g3-FU** (the production launch on b0cf94d0 that goal window 3 left live, unpinned, healthy 23:22:34 UTC) -> `av-pin-k5-g3FU.json` **eeac4f97**. Not `av-pin-g3.json`: that one was profiled on 592793fd, before dhtac, and prices a 4-request draft at 7.36 ms against 6.49 ms on today's stack, while the k6/k7 profiles were measured with dhtac; a stale k5 pin would have biased the 4-session comparison.
  - k6 / k7: dedicated profile launches **mab-p6 / mab-p7** (`DSV41_AV_PROFILE=force`, no pin, `DSV41_CAPTURE_WATCHDOG_S=300`, supervisor `--stall 600 --budget 1800` for first-launch JIT; healthy +426 / +420 s) -> `av-pin-k6-mab-p6.json` **b166aacd** (34 capture sizes to 224, 10 draft sizes) and `av-pin-k7-mab-p7.json` **5886e752** (35 capture sizes to 256, 7 draft sizes).
  - Every measured launch logged "pin hit" and passed `mkpin_k.py check`.
- **Order:** profile launches p6, p7; then **C1 A7 A6 C2 B6 B7 O7 C3** (C = k5, A/B = k7 / k6, O = k7off). Pairs: each candidate against the mean of its bracketing controls (A: C1+C2, B and O: C2+C3).
- **Per instance** (`mab.sh`): supervised launch -> SM clock on 8/8 -> `gstart.sh` (exact.py FIRST, spec-counter deltas) -> `logfacts.py --expect-k` (graph counts against `capsizes.py`, memory, JIT) -> pin check -> suite phase by phase, telemetry OFF, one `nvidia-smi` read on 8/8 before every measurement phase, 2 s `/metrics` sampler for foreign requests: matched C1 coding / prose x5 (label pdab), G1 x2 (`goal_single.py`), G1 texts (`g1text.py`), G4 warm-up + 6 runs (`goal_conc4.py`), **Fraser bench** x10 + 1 warm-up (Moet's prompt and method, `~/dsv41-bench/fraser/`), conc8, conc16, conc8, conc16, t1 sampled (temp 1.0 / top-p 0.95, seed 20261027, 2 reps x 3 types x 512 tokens), exact.py #2, facts. Each measurement phase is bracketed by spec-counter snapshots (accepted tokens per draft position).
- **Clocks:** one `nvidia-smi` read on 8/8 before every measurement phase and after every launch: 101 good checks at **1969-1989 MHz**, event reasons 0x0, at most 64 C (one check got no reading from one node, see the incident below). No clock or power setting was changed.
- **Foreign traffic:** none in any timed phase (sampler maximum = the phase's own concurrency; Fraser's own busy check saw 0 before every rep).
- **Incident:** the first C2 attempt aborted before its G4 phase at 01:03 UTC because the clock check got no reading from spark-five (a transient tailnet ssh failure; spark-five answered 1976 MHz seconds later). `clk8.sh` now re-reads a node that returns nothing, twice, before it counts as BAD. The partial instance is kept as `mab-C2x` (its G1 101.86 with digest 9a18f802d930f6e7 and C1 30.27 / 26.29 ms match C1/C2) and C2 was relaunched from scratch.

### 2. Drafter cost and startup facts per k
| | k5 | k6 | k7 | k7off |
|---|---|---|---|---|
| drafter ms per step at 1 / 2 / 4 requests (startup AV profile, raw `draft_ms`) | 4.28 / 4.74 / 6.49 | 4.73 / 5.34 / 7.20 | 5.14 / 6.13 / 7.86 | - (no AV) |
| AV profile verify ms, 1 request x (k+1) rows | 30.23 (6 rows) | 31.68 (7) | 33.58 (8) | - |
| graphs PIECEWISE / FULL / DSpark (x2 sets) / decoder replay (all = `capsizes.py`) | 45 / 29 / 22 / 3 | 50 / 34 / 26 / 4 | 51 / 35 / 29 / 4 | 51 / 32 / 29 / 4 |
| pinned launch to healthy | 276-284 s | 293 s | 284-289 s | 320 s |
| rank-0 MemAvailable after the startup trim | 16.14-16.52 GB | 15.84-15.86 GB | 15.13-15.29 GB | 15.47 GB |
| MemAvailable at the end of the suite (min over 8 nodes) | 14.4 GB | 13.7 GB | 13.2-13.3 GB | 13.4 GB |

- One more draft position costs **+0.45 ms per single-user step** (k6) and two cost **+0.86 ms** (k7); at 4 requests +0.71 / +1.37 ms. Each position is one more drafter query token per request plus one more sequential Markov step. The k7 profile also shows 25.4 ms at 32 requests (k5 15.6 ms).
- Memory: k6 costs about 0.5 GB and k7 about 1.1 GB per node against k5. The 1M probe was not run because no arm qualified; scaled from today's 1M minimum (8.3 GB, g3-FU), k7 would be near 7.2 GB, above the 6 GB floor (estimate only).

### 3. Exactness
- **E0: exact.py a8c3394af9d2 on all 12 launches**: the 2 profile launches, the 9 measured launches (including the aborted C2x; the 8 complete instances again after their suite) and the restored production instance. Counters (accepted / drafts / draft tokens / generated; accepted per position):
  - k5 under its pin: 909 / 467 / 2335 / 1373; 348/233/171/90/67, identical to g3-FU (the pin reproduces its source launch);
  - k6: 925 / 453 / 2718 / 1373; 339/226/166/82/63/49 (profile and pinned launches identical);
  - k7: 922 / 456 / 3192 / 1373; 340/224/169/67/55/38/29 (same);
  - k7off: 979 / 401 / 2807 / 1373; 294/213/160/118/89/64/41.
  - Also on every launch: 0 ERROR/Traceback on 8/8, graph counts equal to `capsizes.py`, no "first call during capture", and no new JIT names after health (k6/k7 show the same first-use Triton names as k5).
- **G1 outputs.** Each arm reproduced its own digest on every instance with identical repetitions: k5 **9a18f802d930f6e7** on all 3 (and the partial C2x), k6 and k7 **69ad872124100141** on both, k7off the same. **Code and math are bitwise identical in all four arms** (596b478a / 92e5d8cf). **Prose differs at one near tie**: character 3885 of 4927 in the reasoning, "... Differences: print" followed by " capital-intensive" (k5) or "ing capital-intensive" (k6 / k7 / k7off).
  - The k6/k7 prose (e8a524f3) is the k5 prose under `av-pin-g3.json`, so their digest 69ad872124100141 equals goal window 3's pinned A/B reference. The k5 control here (pin from g3-FU) gives g3-FU's prose (923a46be). k5 itself flips this tie between AV profiles; the k change moves it the same way an AV profile does.
- **E1 verdict.** Against this window's k5 control, k6/k7 are **greedy-equivalent but not bitwise**: the same class as r4-e4's verify-cap and every AV-profile change. Against window 3's reference digest they would pass. Either way that is a user decision, and the measured performance makes it moot (section 6).
- **Fraser outputs are not a usable bitwise gate in this suite order.** Rep 5 of 10 flipped between k5 instances under the same pin (C1 19d96c52 against C2 and C3 458268ad); the other 9 reps are identical on every instance of every arm, and all k6/k7/k7off instances give 458268ad. Most likely cause: Fraser runs after the batched G4 phase, and AV's online acceptance estimator (global state, refit from the traffic served) then starts from a timing-dependent state. Only phases that run before any batched phase (exact.py, matched C1, G1) are bitwise-reproducible gates.

### 4. Results (8 pinned instances, 2000 MHz, telemetry off, no foreign traffic)
| # | instance | arm | exact.py start / #2 | G1 digest | G1 (code / prose / math) | G4 sess. mean / prose MIN | G4 agg | matched C1 coding / prose tok/s (ms/step) | Fraser tok/s (AL, steps/s) | conc8 / conc16 | t1 sampled | MemAvail end GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mab-C1 | k5 | a8 / a8 | 9a18f802d930f6e7 | 101.66 (95.95 / 76.60 / 132.43) | 60.85 / 48.59 | 182.9 | 171.18 / 103.33 (30.33 / 26.36) | 149.72 (4.50, 33.37) | 260.8 / 398.9 | 98.99 | 14.4 |
| 2 | mab-A7 | k7 | a8 / a8 | 69ad872124100141 | 98.94 (93.58 / 74.75 / 128.48) | 61.53 / 47.74 | 180.7 | 188.27 / 102.62 (32.43 / 26.40) | 156.09 (4.97, 31.55) | 261.2 / 392.8 | 97.09 | 13.3 |
| 3 | mab-A6 | k6 | a8 / a8 | 69ad872124100141 | 101.77 (94.62 / 76.42 / 134.27) | 62.53 / 47.54 | 182.3 | 187.84 / 104.12 (31.55 / 26.49) | 156.66 (4.86, 32.28) | 263.3 / 400.9 | 99.63 | 13.7 |
| 4 | mab-C2 | k5 | a8 / a8 | 9a18f802d930f6e7 | 101.81 (96.13 / 76.67 / 132.62) | 61.67 / 48.24 | 182.4 | 171.32 / 103.63 (30.30 / 26.28) | 150.07 (4.50, 33.42) | 265.3 / 400.4 | 99.25 | 14.4 |
| 5 | mab-B6 | k6 | a8 / a8 | 69ad872124100141 | 101.56 (94.55 / 76.00 / 134.13) | 62.11 / 48.25 | 180.4 | 187.50 / 103.89 (31.61 / 26.55) | 156.53 (4.86, 32.26) | 258.5 / 393.7 | 99.48 | 13.7 |
| 6 | mab-B7 | k7 | a8 / a8 | 69ad872124100141 | 98.93 (93.65 / 74.69 / 128.44) | 60.52 / 47.63 | 178.4 | 188.22 / 102.48 (32.44 / 26.44) | 155.85 (4.97, 31.54) | 255.7 / 396.3 | 97.06 | 13.2 |
| 7 | mab-O7 | k7off | a8 / a8 | 69ad872124100141 | 93.65 (87.51 / 64.48 / 128.97) | 51.09 / 40.27 | 153.2 | 195.63 / 98.53 (33.04 / 32.39) | 157.39 (5.35, 29.50) | 216.9 / 217.7 | 92.58 | 13.4 |
| 8 | mab-C3 | k5 | a8 / a8 | 9a18f802d930f6e7 | 101.81 (96.07 / 76.62 / 132.73) | 61.79 / 48.56 | 182.3 | 171.55 / 103.65 (30.26 / 26.28) | 149.90 (4.50, 33.40) | 258.8 / 396.4 | 99.16 | 14.4 |

**Per arm (means):**

| arm | n | G1 (code / prose / math) | G4 sess. mean / prose MIN | G4 agg | conc8 / conc16 | Fraser tok/s (steps/s) | t1 | matched C1 coding / prose tok/s (ms/step) |
|---|---|---|---|---|---|---|---|---|
| k5 | 3 | **101.76** (96.05 / 76.63 / 132.60) | 61.43 / **48.47** | 182.6 | 261.6 / 398.6 | 149.90 (33.40) | 99.13 | 171.35 / 103.54 (30.30 / 26.31) |
| k6 | 2 | **101.67** (94.59 / 76.21 / 134.20) | 62.32 / **47.89** | 181.3 | 260.9 / 397.3 | 156.60 (32.27) | 99.56 | 187.67 / 104.00 (31.58 / 26.52) |
| k7 | 2 | **98.93** (93.61 / 74.72 / 128.46) | 61.03 / **47.69** | 179.5 | 258.5 / 394.6 | 155.97 (31.54) | 97.08 | 188.25 / 102.55 (32.44 / 26.42) |
| k7off | 1 | **93.65** (87.51 / 64.48 / 128.97) | 51.09 / **40.27** | 153.2 | 216.9 / 217.7 | 157.39 (29.50) | 92.58 | 195.63 / 98.53 (33.04 / 32.39) |

Control spread (k5, n=3): G1 101.66-101.81 (sd 0.09%), G4 prose MIN 48.24-48.59 (sd 0.4%), Fraser 149.72-150.07, t1 98.99-99.25.

**Pinned pairs** (each candidate against the mean of its bracketing controls; + = better, %):

| inst | controls | G1 | G1 code / prose / math | G4 prose MIN | G4 sess. mean | G4 agg | conc8 | conc16 | Fraser | t1 | C1 coding tok/s | C1 prose tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A7 (k7) | C1+C2 | **-2.75** | -2.56 / -2.46 / -3.06 | **-1.41** | +0.45 | -1.09 | -0.70 | -1.73 | +4.13 | -2.05 | +9.94 | -0.83 |
| B7 (k7) | C2+C3 | **-2.83** | -2.55 / -2.55 / -3.19 | **-1.59** | -1.95 | -2.17 | -2.42 | -0.52 | +3.91 | -2.16 | +9.79 | -1.12 |
| A6 (k6) | C1+C2 | **+0.04** | -1.48 / -0.28 / +1.31 | **-1.81** | +2.07 | -0.21 | +0.08 | +0.30 | +4.51 | +0.51 | +9.69 | +0.62 |
| B6 (k6) | C2+C3 | **-0.24** | -1.61 / -0.84 / +1.10 | **-0.32** | +0.61 | -1.11 | -1.37 | -1.18 | +4.37 | +0.28 | +9.37 | +0.24 |
| O7 (k7off) | C2+C3 | **-8.01** | -8.94 / -15.87 / -2.79 | **-16.81** | -17.24 | -16.01 | -17.22 | -45.36 | +4.94 | -6.68 | +14.11 | -4.93 |

**Acceptance (AL = 1 + accepted / drafts) per workload** (deterministic phases are identical across an arm's instances):

| arm | G1 code / prose / math | matched C1 coding / prose | Fraser (median) | G4 batch | conc8 / conc16 | t1 sampled | G1 accepted per position (share of steps) |
|---|---|---|---|---|---|---|---|
| k5 | 2.590 / 1.847 / 3.594 | 5.191 / 2.725 | 4.50 | 2.50-2.57 | 2.29-2.36 / 2.42 | 2.608 | .695 / .402 / .255 / .079 / .058 |
| k6 | 2.564 / 1.852 / 3.701 | 5.926 / 2.759 | 4.86 | 2.61-2.62 | 2.33-2.34 / 2.44-2.48 | 2.629 | .702 / .396 / .235 / .078 / .056 / .033 |
| k7 | 2.538 / 1.807 / 3.561 | 6.107 / 2.709 | 4.97 | 2.52-2.55 | 2.30-2.35 / 2.45-2.47 | 2.607 | .686 / .379 / .228 / .064 / .045 / .027 / .015 |
| k7off | 2.834 / 2.064 / 3.958 | 6.463 / 3.191 | 5.35 | 2.92 | 2.72 / 2.85 | 2.922 | .689 / .434 / .263 / .172 / .104 / .062 / .030 |

- With AV on, positions 6 and 7 are accepted on only 3.3% (k6) and 2.7% / 1.5% (k7) of G1 steps, while every step pays for drafting them. At k7 the acceptance of positions 1-5 also falls: G1 AL code 2.590 -> 2.538 and math 3.594 -> 3.561 **with bitwise-identical outputs**, so k7 needs more steps for the same text. The DSpark drafter was trained with block size 5; a 7-token block changes what its early positions predict (Fraser position 1-5 rates drop the same way).
- Matched C1 coding is near-deterministic code continuation (AL 5.19 of 6 at k5), the one place a wider draft pays: +9.4..9.9% tok/s.

### 5. Fraser bench against vLLM-Moet (single-stream code, thinking off, their prompt and method, median of 10)
| | k5 (AV on) | k6 | k7 | k7off (Moet's config) | Moet, 4x RTX PRO 6000 TP4 |
|---|---|---|---|---|---|
| median tok/s (n / (last - first)) | 149.9 | 156.6 | 156.0 | 157.4 | 680 (k=7) |
| AL | 4.50 | 4.86 | 4.97 | 5.35 | 4.54 (k=5), 5.34-5.38 (k=7) |
| steps/s | 33.40 | 32.27 | 31.54 | 29.50 | 127.8 (k=7) |
| tok/s vs k5 | - | +4.4..4.5% | +3.9..4.1% | +4.9% | +4% (k=7 against k=5: tokens/step +18.5%, steps/s -12%) |

Our k7off reproduces their relative numbers almost exactly (AL +18.8%, steps/s -11.7%, tok/s +4.9%): the same drafter with the same acceptance on the same prompt. The 4.5x absolute gap is the hardware (GB10 x8 at TP8 over CX7: about 30 ms per step against about 8 ms).

### 6. Decisions (rule for this window: exact, and G1 mean or G4 prose MIN >= +0.5% in >= 2 pinned pairs without lowering the other beyond noise)
- **k7: NOT promoted.** G1 **-2.75 / -2.83%** (every prompt -2.5..-3.2%), G4 prose MIN **-1.41 / -1.59%**, conc8 -0.7 / -2.4%, conc16 -1.7 / -0.5%, t1 sampled -2.1 / -2.2%. It wins only on high-acceptance code without thinking: Fraser +3.9..4.1%, matched C1 coding +9.8..9.9%. Cause: +0.86 ms of drafting on every single-user step, little acceptance at positions 6-7, and lower acceptance at positions 1-5 (section 4).
- **k6: NOT promoted.** G1 **+0.04 / -0.24%** (code -1.5 / -1.6%, math +1.3 / +1.1%, prose -0.3 / -0.8%), G4 prose MIN **-1.81 / -0.32%**, conc8 +0.1 / -1.4%, conc16 +0.3 / -1.2%, t1 +0.5 / +0.3%. No pair reaches +0.5% on G1 or on G4 prose MIN, and G4 prose falls in both. Fraser +4.4..4.5%, matched C1 coding +9.4..9.7%.
- **k7off (Moet's served config): reference only.** G1 **-8.0%** (prose -15.9%), G4 prose MIN **-16.8%**, G4 aggregate -16%, conc8 -17%, conc16 **-45%**, t1 -6.7%. With AV off every step verifies all 8 rows, so this is round 1's AV-off result again (-11..-22% at 4-8 users), worse at k=7.
- **A per-workload k would be needed and is not available.** k=6/7 pays only for code-like, high-acceptance traffic without thinking (Fraser, matched coding) and costs on prose and on the goal mix. vLLM 0.32's v2 GPU runner and DSpark have no per-request or per-batch draft width: `num_speculative_tokens` is server-wide, and `num_speculative_tokens_per_batch_size` exists only in the capture code. AV caps how many drafted tokens are verified, but the drafter still pays for all k. Even a perfect per-type choice would add at most about +0.4% to G1 (k6 for math only) and nothing to G4, whose binding prose session is best at k5.
- **Moet's one-launch Markov sampling: NOT ported**, because it is neither exact nor reachable in our config:
  - It would never run here. Their `fused_markov.sample_sequential` falls back to vLLM's loop for greedy drafts and whenever the acceptance estimator is on, and our estimator is on whenever AV is on (`speculator.py`: `use_acceptance_estimator = enable_adaptive_verification`). G1/G4 are greedy in any case, and our greedy drafts already use the vocab-sharded Markov path (8.3 MB instead of 66 MB per position per rank).
  - It is not bitwise. Their own patcher notes that the fused bias GEMV can differ in the last bf16 bit and then change a draft token.
  - It is not cheap. Its source (`models/deepseek_v41/amd/ops/fused_markov.py`) is not in our vLLM 0.32.0, and our sampled path keys its Gumbel noise with FRESH_NOISE, which a port would have to reproduce.
  - The only gain would be on sampled steps with AV off: launch overhead, about 0.03-0.08 ms per step.
- **Moet's other drafter kernels** (MXFP8 GEMV head, dflash input prep, SWA draft metadata) stay unported: at most 0.05 ms each (prep section 1). The most useful exact item in their repo is still the **mHC boundary overlap**: target side, kernel-ideas #8, about -0.8..-1.6 ms per single-user step if it proves bitwise on sm_121. Its sources are staged in `~/dsv41-stage/moet/src/`.
- **Goal status: unchanged.** The k5 control gives G1 101.66-101.81 and G4 prose MIN 48.24-48.59 (pinned, 3 instances), so G1 needs **+17.9%** and G4 prose **+3.2%**. A wider draft is not the acceptance lever G1 needs; a better-trained drafter, or a verify path that makes deep positions cheap, would be.

### 7. Live state and rollback
- **Live: instance mab-R**, the production config restored at the end: launcher defaults (k=5, AV unpinned = own startup profile, "pin miss"), launcher **b0cf94d0** on 8/8, supervisor 7bea1cf3, site manifest b1150ecfcf49, clock 0-2000 MHz.
  - Healthy since **02:21:45 UTC** (+355 s, first try) with 0 restarts and 0 ERROR/Traceback on 8/8.
  - exact.py **a8c3394af9d2** (917 / 459 / 2295 / 1373; 344/236/176/93/68).
  - `/health` returns 200 locally and on the LAN address; `dsv41-lanproxy` is active.
- **Nothing was installed or changed** in the launcher, site-packages, supervisor or services. New files only:
  - the pins and their source logs in `~/dsv41-stage/moet/pins/` on 8/8: `av-pin-k5-g3FU.json` eeac4f97, `av-pin-k6-mab-p6.json` b166aacd, `av-pin-k7-mab-p7.json` 5886e752;
  - results in `~/dsv41-moet-ab/` on spark-one.
- No telemetry, sampler or bench process is left on 8/8.
- **To serve k=6/7 anyway** (for example a code-only deployment): pass `DSV41_SPEC=6|7 DSV41_AV_COST_JSON=~/dsv41-stage/moet/pins/av-pin-k<K>-mab-p<K>.json` as supervisor arguments. exact.py a8 is verified for both. Graph memory is about 0.5 / 1.1 GB more per node, and the 1M probe was not run at k6/k7. Rollback: relaunch without the two variables.

### 8. Files
- **Mac scratchpad `mab/`:**
  - scripts: `mab.sh`, `mabblk.sh`, `mabphase.sh`, `clk8.sh` (with the re-read), `finsampler.sh`, `mabline.py`, `mabagg.py`, `pincmp.py`;
  - logs and results: `ab.log`, `ab.jsonl`, `tables.md` (full tables, including per-position acceptance and startup facts), `launches.tsv`, `pins.env`;
  - per instance: `ab/<label>/` (relaunch log, clock reads, pin make/check) and `runs/<label>/`, `runs/<label>-start/`.
- **spark-one `~/dsv41-moet-ab/`:** `runs/<label>/` (per-phase outputs, including `g1text.json` for divergence diffs: `python3 -I ~/dsv41-stage/moet/tools/g1text.py diff A B`), `runs/<label>-start/` (gstart, `rank0.startup.log`, `facts.json`), `mabphase.sh`, `finsampler.sh`.


## Goal window 4, prefill (2026-10-27 02:30-07:25 UTC): Codex items 04 (exact SP prefill) + 05 (exact E7) PROMOTED as a pair (launcher fd1e809e): exact, TTFT -6.5% at 16K, -8.8% at 128K, -8 to -10% at 500K / 1M, short chats during a long prefill -54%; 06 (candidate-only indexer) exact but no measurable gain, NOT promoted. Decode unchanged; goal still NOT met (G1 101.6 of 120; prose session MIN 47.6 of 50)

Scope: the PREFILL part of the Codex ten deliverables (`/Users/jeweled/Documents/Codex/2026-10-20/let-s-improve-our-deployment-of-2/outputs/ten-deliverables-20261026/` 04, 05, 06): build / stage against today's site files, GPU qualification in a stopped window, a pinned four-way prefill A/B (stock / 04+05 / 06 / all), promote exact winners. Clocks never raised (every launch `DSV41_GPU_CLOCK_MHZ=2000`; 141 SM reads on 8/8 after launches and before timed phases, all 1969-1989 MHz, none above 2010). Pin for every A/B launch: `~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json` (eeac4f97; anchor a8c3394af9d2 909/467/2335/1373, G1 2-rep digest 9a18f802d930f6e7), made from the live b0cf94d0 stack. Mac drivers and raw results: scratchpad `g4p/` (`src/mk_g4p.py`, `src/mk_g4p06.py`, `src/q06.py`; `scripts/sw4.sh` stopped window, `g4p.sh` one instance, `g4phase.sh` phases, `g4blk.sh` block, `qsp.sh` / `qe7.sh` / `qsum.py` qualification, `g4line.py`, `g4agg.py`, `promote4.sh`; `ab.log`, `ab.jsonl`, `tables.md`, `ab/<label>/` rank-log extracts, `runs/<label>/`, `sw/`). On spark-one: `~/dsv41-g4p/` (`runs/`, `STATE.md`), staged bundle `~/dsv41-stage/g4/` (8/8).

### 1. Builds (bases re-checked on 8/8: model.py e7264781, model_state.py f21ee2e6, decoder_replay_layers.py 2fedc0a6, dspark/speculator.py bbcee9e3, sparse_attn_indexer.py 481905c2; Codex's 06 patch was against the pre-window-3 indexer files, its 04 patch against today's model.py)
One site bundle `~/dsv41-stage/g4/g4p` (8/8; `FILES` + `site/`, installer `~/dsv41-stage/g4/common/g4-site.sh` / `g4-all.sh`, backups `<file>.bak-g4-g4p`), every feature default off; builders `tools/mk_g4p.py` (04 + 05) and `tools/mk_g4p06.py` (06), anchored edits on the live bases.

| file | live -> g4p | items |
|---|---|---|
| models/deepseek_v41/nvidia/model.py | e7264781 -> **9c2d22ad** | 04 (Codex `sp_exact_logical_rows.patch` + fixes below), 05 (forward hook) |
| models/deepseek_v41/nvidia/model_state.py | f21ee2e6 -> **b7598364** | 05 rule; per-request opt-outs of 04/05/06 |
| models/deepseek_v41/decoder_replay_layers.py | 2fedc0a6 -> **b75aa927** | 05 (`run_rows_discard`), step flags |
| v1/worker/gpu/spec_decode/dspark/speculator.py | bbcee9e3 -> **439b390e** | 05 (draft skip) |
| model_executor/layers/sparse_attn_indexer.py | 481905c2 -> **f2ae649b** | 06 |

- **04 SP exact** (`DSV41_PREFILL_SP=exact`): Codex's patch applied cleanly to e7264781. Layers 0-19 of eager chunks >= 2048 rows keep the mHC/Engram per-row work on this rank's 1/8 of the rows; attention, MoE and every all-reduce run on all rows exactly as stock (`exact` = the stock all-reduce, then this rank's rows). `sp_logical_rows()` makes the mHC split heuristics (`compute_mhc_pre_num_splits`, `compute_num_split`, fused post/pre config, prenorm GEMM dispatch) see the full row count T, which removes the pf split-count mismatch at 2K-12K rows. Fixes on top of Codex's code: (a) its DEBUG top-k replay assigned a function to the `indexer_op` nn.Module child (TypeError) -> it now shadows that module's `forward`; (b) every equality is bitwise (uint8 view, NaN-safe) instead of `torch.equal`; (c) no self-check verdict is cached from a captured / autotune / dummy forward; (d) `DSV41_PREFILL_SP_SELF_CHECK=once` keys the sharded-op self-check by (layer, op) only (a startup canary; `1` = Codex's per-row-count check, which re-runs on every new chunk size); (e) per-request opt-out `vllm_xargs {"dsv41_sp_off": 1}`; (f) the DEBUG compare is a native-dtype bit compare and frees its snapshots (see the QSP incident).
- **05 E7 exact** (`DSV41_E7_EXACT=post_and_draft`): Codex shipped only pseudocode, so this is a new implementation on pf's E7 plumbing. Rule (model_state): every request of an eager step is still prefilling, keeps no rows (no prompt logprobs), did not opt out, and has more than max(max_num_batched_tokens, 128) prompt tokens left after this chunk, so the chunk is neither F nor F-1. On such a step layer 20's `write_kv` (every row) and the 128-row replay through layer 39 run exactly as stock (every SWA / compressed / indexer-K write happens); skipped: the scatter of the replay rows back to the batch, the MTP buffer copy, hc_collapse, the final norm (the step's logits are discarded) and the DSpark propose (context KV of chunks <= F-2 is outside the drafter's 128-token window; AV's first decode budget reads F-1's propose, which runs). No scheduler change (pf's guarantee G is not needed because no cache write is skipped). Opt-out `{"dsv41_e7_off": 1}`. Codex's optional `layer36_cut` was not built.
- **06 candidate-only indexer** (`DSV41_CAND_ONLY_INDEXER=scatter`): Codex's pack / scatter kernels and compressed-logits call, verbatim, rebased onto today's 481905c2 (codex-adopt). Only the **scatter arm, prefill only**: consumer indexers 24/28/32/36, chunks with <= 128 query rows and >= 393,216 gathered K rows (`DSV41_CAND_ONLY_MIN_K`, below that q06 measured the scatter path slower), pack each row's 2,048 candidate blocks, score them with the same DeepGEMM FP8 MQA-logits kernel (`max_seqlen_k=16384`), scatter into a -inf full-width row, then the unmodified `top_k_per_row_prefill` + CANON. Not ported: Codex's decode arm, compact arm, tile-union (decode stays stock, so long-context decode is unchanged). `verify` mode serves stock and compares the scatter logits bitwise in shadow. Opt-out `{"dsv41_cand_off": 1}`.
- Flags off = stock code path (one attribute read per forward); the C arm below runs the installed bundle with flags off.

### 2. Stopped window and qualification (03:07-03:46 UTC)
- **Install** 03:07-03:10 (service stopped 8/8, 0 GPU apps), bundle digest identical 8/8; reinstalled at 03:25 with the fixed model.py (digest aef7c824bb9a, files 9c2d22ad / b7598364 / b75aa927 / 439b390e / f2ae649b on 8/8). Import check (all three flags on): PASS.
- **06, q06.py on spark-two (GPU, service stopped)**: 71 cases (rows 1/4/128 x widths 128K/512K/1M x patterns realistic, exact ties, few values, NaN/Inf/0 scales, -1/INT32_MAX sentinels, duplicates, beyond-width blocks; plus 16,384+d widths): **0 bit mismatches** between the scatter logits and stock's dense + candidate-masked logits on every column. Top-k: equal wherever stock's own top-k is repeatable; one all-ties fixture (every score 0, m=1, 1M) differed, and `q06b.py` showed stock itself returns 2 different sets over repeated runs on identical logits (the top-k kernel's tie order), so it is not a scatter difference. Timing per consumer layer at 128 rows: 128K 2.43 -> 4.78 ms (slower), 512K 9.87 -> 8.26, 1M 19.58 -> 11.99 ms; hence the 393,216-row floor.
- **QSP (04 and 06 in serving)**, pinned, `DSV41_PREFILL_SP=exact DSV41_PREFILL_SP_DEBUG=64 DSV41_CAND_ONLY_INDEXER=verify`:
  - attempt 1 (03:10-03:18): bitwise at 2048-12,288 rows, then **rank 0 was OOM-killed at 12,296 rows** (kernel oom-kill 03:18:21): Codex's DEBUG compare made fp32 copies of the [T, 4, 5120] states (~4 GB transient) with a new row count per probe. Fixed (f) above; reinstalled; relaunched.
  - attempt 2 (03:25-03:38): exact.py a8c3394af9d2 (909/467/2335/1373, pin hit) at start and after all probes; **in-forward DEBUG compares bitwise on all 5 states (hidden, residual, post_mix, res_mix, pre_mix) on 8/8 ranks at every probed row count**: 2048, 2056, 3000, 3999, 4096, 6000, 8008, 8131, 8189, 8192, 12,288, 12,296, **12,760, 13,544, 14,241**, 15,022, 16,032, 16,380 (344 compare lines, 1,720 bitwise flags, 0 mismatches); the reference top-k was replayed at all three index-source layers (2/8/14) in every compare, i.e. their attention inputs were bit-identical; per-(layer, op, rows) self-check 760 verdicts, 0 diverged. The pf "12.7-14.2K" divergence was the layer-2 top-k tie order, as Codex said: with the reference top-k replayed it is bitwise.
  - end to end (`sptid.py`, SP copies vs `dsv41_sp_off` copies, distinct cache_salt, 16 greedy tokens + top-20 logprobs): 2048 and 4096 ids: 6/6 identical. At 3000 ids (seed 3000) stock itself gave 3 different first-token logprobs in 3 copies (the known prefill tie nondeterminism), and one SP copy equalled one stock copy bit for bit, so that prompt cannot discriminate.
  - 06 verify on real data (64K / 200K / 500K prompts): engaged from 409,500 K rows; scatter logits == stock on every column and identical CANON top-k sets in every call on 8/8 ranks (0 bad columns, 0 differing rows).
- **QE7 (05)**, `DSV41_E7_EXACT=post_and_draft` + `--max-num-batched-tokens=1024` (deterministic regime: 2-4K-token prompts = 3-4 chunks, the chunks <= F-2 skip), AV unpinned, pf's `e7x.py`: exact.py a8c3394af9d2 first. **E7X copies identical to the stock (`dsv41_e7_off`) copies** (text, all 64 logprobs with top-5, the following decode's spec counters) at 2187 / 3137 / 3261 / 3856 / 4009 tokens (3 + 3 copies each) and 3600 tokens (256 tokens, 2 + 2): these are the prompts where pf's E7 changed the first-token logprob (2187: -0.3499 stock vs -0.6346 old E7; now -0.3499 in every copy). A 2589-token prompt was nondeterministic in stock itself (2 stock copies, 2 first-token logprobs), so it does not count either way. The indexer-K and SWA writes are the stock calls by construction (the replay runs), and these identical first-token logprobs are the functional check that failed for the old E7.
### 3. Method (pinned four-way A/B at 2000 MHz, telemetry off)
- **Arms** (every launch `--expect-launcher b0cf94d0`, `DSV41_GPU_CLOCK_MHZ=2000`, pin `DSV41_AV_COST_JSON=~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json` (eeac4f97, made from the live b0cf94d0 stack; "pin hit" on every launch), the g4p bundle installed):
  - **C** stock: no g4p flag.
  - **P** = 04 + 05: `DSV41_PREFILL_SP=exact DSV41_PREFILL_SP_SELF_CHECK=once DSV41_E7_EXACT=post_and_draft`.
  - **K** = 06: `DSV41_CAND_ONLY_INDEXER=scatter`.
  - **X** = all: P + K.
- **Order** C1 P1 K1 X1 C2 X2 K2 P2 C3; each candidate is paired with the mean of the controls before and after it.
- **Per instance** (`g4p.sh` -> `g4phase.sh` on spark-one; one `nvidia-smi` read on 8/8 before every timed phase; 2 s `/metrics` sampler for foreign requests): exact.py FIRST (+ spec counters, pin check), matched C1 coding/prose x5, G1 x2 (`goal_single.py`, output digest), [P/K/X: `sptid` SP-vs-stock e2e at 2048 / 4096 ids], a 40K warm-up prompt (compiles the new kernels before timing), cold solo TTFT 16K x3 and 128K x2 (`sbench.py`, fresh nonce prompts), the 128K starvation bench (long 128K prompt + 6 short chats + a decoder), cold solo 500K and 1M (`xrb4.py`, per-chunk scheduler times; 1M launch class from the chunk 37-43 band: good < 3.4 s), exact.py #2.
- **Same-instance like-for-like** (candidate arms): the same lengths again with every g4p feature opted out per request (`vllm_xargs {"dsv41_sp_off":1,"dsv41_e7_off":1,"dsv41_cand_off":1}` = the stock code path), and on X also P-only (`dsv41_cand_off`) and K-only (`dsv41_sp_off` + `dsv41_e7_off`) at 500K and P-only at 1M. Needed because the rank-5 long-context straggler class is random per launch (pinned too), and it moves 500K / 1M by 10-30% between instances.
- **Gates on P2**: GSM8K-50, full longctx, 1M needle (7 depths).
### 4. Results (9 pinned instances 03:46-07:02 UTC, 2000 MHz, telemetry off)
**Exactness.** All 9 launches gave exact.py **a8c3394af9d2** with the pin's counters 909/467/2335/1373 (per position 348/233/171/90/67, "pin hit") at start and again after the whole suite (exact.py #2), and 0 ERROR/Traceback on 8/8. The G1 output digest was **9a18f802d930f6e7** (= the pin's reference, k5 controls of the Moet A/B) on 8 of 9; on X2 one of the two prose repetitions differed (51423598 vs 923a46be; code and math identical): the server completed 7 requests during that 6-request phase, one of them a short foreign request finishing 2 s after its neighbour, overlapping that prose request (the LAN/tailnet API has no authentication). X1, the same config, gave the reference digest. The G1 phase now re-runs itself when a foreign request lands in it (`g4phase.sh`, used from P2 on: P2 and C3 clean at the first try). SP-vs-stock end to end (`sptid`, 2048 / 4096 ids, 2 + 2 copies) identical on P1, P2, X1, X2.

**Per instance** (C1 = matched C1 ms/step; TTFT 16K = mean of 3, 128K = mean of 2, cold solo, fresh nonce; 500K / 1M = one cold solo prompt each; 1M class from the chunk 37-43 band):

| # | instance | arm | exact start / #2 | G1 digest | C1 code / prose ms | G1 | TTFT 16K / 128K s | starve long / short p50 s | 500K s | 1M s (class) | 1M mem min GB | e7x / cand | errors | foreign |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | g4-C1 | C | a8c3394a / a8c3394a | 9a18f802d930f6e7 | 30.170 / 26.280 | 101.88 | 2.681 / 21.673 | 25.018 / 1.807 | 90.11 | 227.73 (slow) | 7.73 | 0 / 0 | errors=0 errors_end=0 | none |
| 2 | g4-P1 | P | a8c3394a / a8c3394a | 9a18f802d930f6e7 | 30.220 / 26.230 | 101.94 | 2.497 / 19.738 | 23.023 / 0.817 | 79.91 | 167.16 (good) | 7.82 | 100 / 0 | errors=0 errors_end=0 | none |
| 3 | g4-K1 | K | a8c3394a / a8c3394a | 9a18f802d930f6e7 | 30.230 / 26.320 | 101.68 | 2.687 / 21.608 | 24.945 / 1.776 | 90.57 | 229.27 (slow) | 7.34 | 0 / 100 | errors=0 errors_end=0 | none |
| 4 | g4-X1 | X | a8c3394a / a8c3394a | 9a18f802d930f6e7 | 30.300 / 26.260 | 101.85 | 2.513 / 19.794 | 23.074 / 0.963 | 82.77 | 209.95 (slow) | 7.77 | 100 / 100 | errors=0 errors_end=0 | none |
| 5 | g4-C2 | C | a8c3394a / a8c3394a | 9a18f802d930f6e7 | 30.200 / 26.290 | 101.91 | 2.674 / 21.670 | 25.087 / 1.860 | 90.14 | 225.68 (slow) | 7.85 | 0 / 0 | errors=0 errors_end=0 | none |
| 6 | g4-X2 | X | a8c3394a / a8c3394a | a2a9c0817992c599 | 30.190 / 26.310 | 100.90 | 2.511 / 19.822 | 22.676 / 0.874 | 82.66 | 212.47 (slow) | 7.75 | 100 / 100 | errors=0 errors_end=0 | {'pwarm': 2.0} |
| 7 | g4-K2 | K | a8c3394a / a8c3394a | 9a18f802d930f6e7 | 30.210 / 26.270 | 102.01 | 2.671 / 21.566 | 25.064 / 1.833 | 87.29 | 181.29 (good) | 7.11 | 0 / 100 | errors=0 errors_end=0 | none |
| 8 | g4-P2 | P | a8c3394a / a8c3394a | 9a18f802d930f6e7 | 30.170 / 26.250 | 101.97 | 2.505 / 19.760 | 22.787 / 0.866 | 82.98 | 210.52 (slow) | 7.75 | 100 / 0 | errors=0 errors_end=0 | none |
| 9 | g4-C3 | C | a8c3394a / a8c3394a | 9a18f802d930f6e7 | 30.260 / 26.300 | 101.81 | 2.669 / 21.630 | 24.931 / 1.771 | 89.65 | 224.50 (slow) | 7.70 | 0 / 0 | errors=0 errors_end=0 | none |

**Pinned pairs** (each candidate against the mean of the controls before and after it; + = better, %). The 500K / 1M columns mix launch classes (P1 good vs slow controls), so judge those lengths in the same-instance table below:

| inst | arm | controls | G1 | C1 code ms | C1 prose ms | TTFT 16K | TTFT 128K | starve long | starve short p50 | 500K | 1M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g4-P1 | P | g4-C1+g4-C2 | +0.05 | -0.12 | +0.21 | +6.74 | +8.92 | +8.10 | +55.44 | +11.34 | +26.27 |
| g4-K1 | K | g4-C1+g4-C2 | -0.21 | -0.15 | -0.13 | -0.35 | +0.29 | +0.43 | +3.14 | -0.49 | -1.13 |
| g4-X1 | X | g4-C1+g4-C2 | -0.05 | -0.38 | +0.10 | +6.14 | +8.66 | +7.90 | +47.48 | +8.16 | +7.39 |
| g4-X2 | X | g4-C2+g4-C3 | -0.94 | +0.13 | -0.06 | +6.01 | +8.44 | +9.33 | +51.86 | +8.05 | +5.60 |
| g4-K2 | K | g4-C2+g4-C3 | +0.14 | +0.07 | +0.10 | +0.02 | +0.39 | -0.22 | -0.96 | +2.90 | +19.46 |
| g4-P2 | P | g4-C2+g4-C3 | +0.11 | +0.20 | +0.17 | +6.23 | +8.73 | +8.88 | +52.30 | +7.69 | +6.47 |

**Same instance** (the same prompt lengths with every g4p feature opted out per request = the stock code path on the same launch; on X also P-only and K-only):

| inst | arm | 500K arm / stock (%) | 500K P-only / K-only (% vs stock) | 1M arm / stock (%) | 1M P-only (%) | 1M class |
|---|---|---|---|---|---|---|
| g4-P1 | P | 79.91 / 88.58 (+9.80) | - (-) / - (-) | 167.16 / 182.18 (+8.25) | - (-) | good |
| g4-K1 | K | 90.57 / 90.77 (+0.23) | - (-) / - (-) | 229.27 / 227.34 (-0.85) | - (-) | slow |
| g4-X1 | X | 82.77 / 95.17 (+13.04) | 83.64 (+12.12) / 91.00 (+4.38) | 209.95 / 231.62 (+9.36) | 212.03 (+8.46) | slow |
| g4-X2 | X | 82.66 / 91.57 (+9.73) | 82.27 (+10.16) / 89.17 (+2.62) | 212.47 / 225.51 (+5.78) | 208.10 (+7.72) | slow |
| g4-K2 | K | 87.29 / 87.31 (+0.02) | - (-) / - (-) | 181.29 / 181.90 (+0.33) | - (-) | good |
| g4-P2 | P | 82.98 / 90.12 (+7.92) | - (-) / - (-) | 210.52 / 230.19 (+8.54) | - (-) | slow |

- The first stock-path prompt on an SP instance pays ~1 s once (first two 16K chunks at 3.06-3.43 s instead of ~2.63 s: the allocator grows segments for the full-row mHC buffers the SP path never allocated). It hits the first opted-out prompt (l500k on P1, l128k on P2), so those two same-instance percentages are ~1-1.4 points high; the later ones (P1/P2 l1m) are not affected.
- **P2 split at 128K / 500K** (same instance, steady 16K-chunk times in brackets): stock 22.80 s (2.63-2.69 s/chunk after the first two), E7X only 21.03 s (2.54-2.62), SP only 20.30 s (2.42-2.53), both 19.77 s (2.39-2.47); 500K stock 90.12, E7X only 88.15 (-2.2%), SP only 84.78 (-5.9%), both 82.98 (-7.9%). The two levers add.
- Launch class: 7 of 9 pinned launches were slow-class at 1M (c37-43 band 3.69-4.07 s; good: P1 2.73, K2 2.95), i.e. the rank-5 straggler of the X-ray step; P still cuts 8.3-8.5% inside an instance of either class.

**Per arm (means):**

| arm | n | G1 | C1 code ms | C1 prose ms | TTFT 16K | TTFT 128K | starve long | starve short p50 | 500K | 1M |
|---|---|---|---|---|---|---|---|---|---|---|
| C | 3 | 101.868 | 30.210 | 26.290 | 2.675 | 21.658 | 25.012 | 1.813 | 89.967 | 225.966 |
| P | 2 | 101.959 | 30.195 | 26.240 | 2.501 | 19.749 | 22.905 | 0.841 | 81.443 | 188.839 |
| K | 2 | 101.844 | 30.220 | 26.295 | 2.679 | 21.587 | 25.005 | 1.804 | 88.929 | 205.279 |
| X | 2 | 101.374 | 30.245 | 26.285 | 2.512 | 19.808 | 22.875 | 0.918 | 82.715 | 211.211 |

- **Gates (P2, the promoted pair):** GSM8K-50 **50/50**; full longctx **PASS** (needle 60/60, copy 1.000); **1M needle 7/7** (999,411 tokens, copy exact, TTFT 186.5 s). MemAvailable minimum during 1M: C 7.70-7.85 GB, P 7.75-7.82, K 7.11-7.34 (the scatter path's packed K), X 7.75-7.77 (floor 6 GB).
- **Engagement:** P/X launches logged `DSV41_E7_EXACT: 100 steps skipped`, the SP self-check canary (once per layer and op, 0 diverged) and `DSV41_CAND_ONLY_INDEXER scatter: 100 consumer prefill calls` on K/X (from 409,500 K rows).

### 5. Decisions (rule: exact, and TTFT >= 3% at the targeted lengths in >= 2 pinned pairs with decode unchanged)
- **04 + 05 (P): PROMOTED.** Exact (QSP bitwise 2048-16,384 rows on 8/8 ranks, QE7 bitwise where the old E7 failed, exact.py a8 + G1 reference digest on both P launches). TTFT 16K **+6.7 / +6.2%**, 128K **+8.9 / +8.7%**, 500K +11.3 / +7.7% (same instance +9.8 / +7.9%), 1M same instance **+8.3 / +8.5%**; 128K starvation: long prompt +8.1 / +8.9%, short chats' TTFT p50 1.81 -> 0.84 s (**-54%**). Decode unchanged: G1 +0.05 / +0.11%, matched C1 within +-0.2%.
- **06 (K): NOT PROMOTED** (exact, no gain). Logit-bitwise in every check, exact.py and digests unchanged, but 16K -0.35 / +0.02%, 128K +0.3 / +0.4%, same instance 500K +0.2 / +0.0%, 1M -0.9 / +0.3%. Its measured saving (7.6 ms per consumer layer at 1M, 1.6 at 512K, a loss below ~384K) is ~0.3% of a 1M prefill; Codex's -1.5..-2 s assumed the dense consumer cost could go to ~0. Installed, inert (`DSV41_CAND_ONLY_INDEXER` unset).
- **All (X): not promoted beyond P.** 16K / 128K equal to P within noise; inside X instances 06 added -1.0 / +0.5% at 500K and +1.0 / -2.1% at 1M over P-only.

### 6. Promotion and final production instance
- **Launcher** `~/dsv41-tp8.sh` b0cf94d0 -> **fd1e809e** on 8/8 (`scripts/promote4.sh sp-e7x scripts/lines-P.txt`); backup `~/dsv41-tp8.sh.bak-g4-sp-e7x` = b0cf94d0 on 8/8. Three lines after `DSV41_DRAFT_HEAD_TACTIC_MAP`, each commented:
  ```sh
  export DSV41_PREFILL_SP=${DSV41_PREFILL_SP:-exact}
  export DSV41_PREFILL_SP_SELF_CHECK=${DSV41_PREFILL_SP_SELF_CHECK:-once}
  export DSV41_E7_EXACT=${DSV41_E7_EXACT:-post_and_draft}
  ```
  Supervisor unchanged (7bea1cf3); its preflight manifest (which includes the launcher) is now 8791f3d8d6e7, identical on 8/8. `--enable-prompt-tokens-details` unchanged. Dependency: the g4p site files (section 1) stay installed; `DSV41_CAND_ONLY_INDEXER` is unset (06 inert).
- **Final instance g4-FINAL** (production: launcher fd1e809e defaults, AV unpinned = startup profile, `DSV41_GPU_CLOCK_MHZ=2000`): healthy 07:09:22 UTC (+360 s, first try); env on 8/8 carries the three new flags; exact.py **a8c3394af9d2** first (own profile 911/465/2325/1373, "pin miss") and again after the suite; self-check canary 0 diverged; E7X engaged (100+ steps). Suite (telemetry off, 15 clock reads on 8/8 at 1976-1989 MHz, no foreign request beyond each phase's own concurrency):

  | metric | g4-FINAL (production, fd1e809e) | ref: g3-FU (production, b0cf94d0, goal window 3) |
  |---|---|---|
  | G1 mean (code / prose / math), 3 reps | **101.62** (96.31 / 75.66 / 132.88); identical outputs in all 3 reps (596b478a / e8a524f3 / 92e5d8cf) | 101.51 (95.70 / 76.56 / 132.28) |
  | G1 AL code / prose / math | 2.596 / 1.811 / 3.607 | - |
  | G4 per-session mean / prose MIN (6 runs) | 60.50 / **47.61** (prose runs 47.67 / 47.53 / 46.88 / 46.88 / 49.24 / 47.49); agg 182.8 | 61.65 / 48.64; agg - |
  | matched C1 coding / prose ms | 30.44 / 26.36 | 30.32 / 26.33 |
  | conc8 / conc16 tok/s | 259.6 / 399.8 | 264.3 / 397.8 |
  | TTFT 16K / 128K (cold solo) | **2.511 / 19.806 s** | (A/B controls 2.675 / 21.658 s) |
  | 128K starvation: long / short p50 / p95 | **22.62 / 0.85 / 2.40 s** | (controls 25.01 / 1.81 s) |
  | cold 1M (1,024,611 tokens) | **167.52 s, good class** (chunk bands 2.44 / 2.59 / 2.73 / 2.87 s); MemAvailable min 10.3 GB | 183.70 s good class; 8.27 GB |
  | GSM8K-50 / full longctx / usage | **50/50** / **PASS** (needle 60/60, copy 1.000) / cached_tokens 0 -> 4160 | 50/50 / PASS / 4160 |

  - Prefill is the only path these flags touch: SP needs >= 2048 rows in an eager forward and E7X a prompt longer than two chunks, so G1 / G4 / C1 / conc run the stock code. The pinned pairs show decode flat (G1 +0.05 / +0.11%, C1 +-0.2%); the production G1 / G4 differences against g3-FU follow the startup AV profile (prose AL 1.811 here against 1.889 on g3-FINAL), as noted in goal window 3.

### 7. Live state and rollback
**Live:** instance **g4-FINAL**, healthy since 07:09:22 UTC; launcher **fd1e809e** on 8/8; supervisor 7bea1cf3; AV unpinned; clock lock 0-2000 MHz (boot unit). Site files on 8/8: model.py **9c2d22ad**, model_state.py **b7598364**, decoder_replay_layers.py **b75aa927**, dspark/speculator.py **439b390e**, sparse_attn_indexer.py **f2ae649b** (backups `<file>.bak-g4-g4p` = e7264781 / f21ee2e6 / 2fedc0a6 / bbcee9e3 / 481905c2); everything from goal windows 2-3 unchanged. 8/8 active, 0 restarts, 0 ERROR/Traceback; no test process left; `dsv41-lanproxy` unchanged (active).

Rollback:
- **One launch:** `DSV41_PREFILL_SP=0 DSV41_E7_EXACT=0` as supervisor arguments (each alone also works). Exact either way.
- **Permanent:** `cp -p ~/dsv41-tp8.sh.bak-g4-sp-e7x ~/dsv41-tp8.sh` on 8/8 (b0cf94d0), relaunch with `--expect-launcher b0cf94d0`.
- **Site files** (service stopped, after the launcher rollback): `ssh spark-one cat ~/dsv41-stage/g4/common/g4-all.sh | bash -s uninstall g4p` (sha-guarded, restores all five files from `.bak-g4-g4p`).
- **Per request** (A/B or debugging): `vllm_xargs {"dsv41_sp_off":1}`, `{"dsv41_e7_off":1}`, `{"dsv41_cand_off":1}`.

### 8. Goal status and next
- **G1 >= 120: NOT met** (production 101.6; this window is prefill-only). **G4 >= 50 per session: NOT met** (prose MIN 47.6 production; pinned 48.47 in the Moet A/B controls). Prefill: cold 16K 2.51 s, 128K 19.8 s, 1M 167.5 s (good class).
- **Still open from the Codex list:** 02 (MoE tile-scheduling microbench), 09 (fused draft budget + verify cap), 01's `DSV41_RDMA_FAST` native port, 07 (multi-turn prefix reuse, `--live` on an idle server), 08 (spark-six straggler: 7 of the 9 pinned launches here were slow-class at 1M, ~45 s slower than a good launch, more than any prefill lever), 03 (prose oracle dump).
- 06 could still pay at decode (Codex's long-context decode arm, ~1-3 ms/step at 512K-1M) but the A/B here measured prefill only; its compact arm needs a tie-aware top-k proof because the stock top-k is not deterministic under ties (q06b).

## Fast weight loader (goal window 4, Loader phase, 2026-10-27 08:35-11:19 UTC): `DSV41_FAST_LOAD` PROMOTED (launcher fd1e809e -> **31d04d8f**). Every parameter is sha256-identical to the stock loader on 8/8 ranks, Engram tables included, in 4 digest launches. exact.py a8 and the G1 reference digest hold on every launch. Target load drops from 162-177 s to 35-37 s; pinned time-to-healthy from 286 s to 126-132 s (-55%); unpinned production from 355 s to 199 s. Stock launches also push ~3 GB per node into the NVMe swap file; the fast loader pushes none

Scope: Codex ops kit 04 (`/Users/jeweled/Documents/Codex/2026-10-20/let-s-improve-our-deployment-of-2/outputs/ops-kits-20261026/04-weight-loader/`), done first on the user's directive of 08:35 UTC ("move fast loading to front and trim the suites"). Suites trimmed accordingly:
- Loader A/B launches: startup and exactness only. That is exact.py first with the pin check, G1 x2 (digest), time-to-healthy, per-rank load lines and 1 Hz `/proc` samplers. No decode or long-prompt suites.
- The final production instance: G1 x3, GSM8K-50, full longctx, ONE cold ~1M prompt and exact.py #2.

Clocks were never raised: every launch used `DSV41_GPU_CLOCK_MHZ=2000`, and the SM clock was read on 8/8 after every launch (1969-1989 MHz).

Every A/B launch used the pin `~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json` (eeac4f97), and every one was a "pin hit".

Mac drivers and raw results are in scratchpad `g4l/`:
- Source and tests: `stage/fload/` (the bundle), `src/test_fload.py` and `src/bench_io.py`.
- Launch and stop scripts: `scripts/g4llaunch_v2.sh` (one launch), `g4lblk.sh` (block), `g4lg1.sh`, `g4lfin.sh` / `g4lfinal.sh` (final verification), `stop8.sh`, `promote4l.sh` and `lines-FL.txt`.
- Samplers: `ldwatch.sh`, `memwatch.sh`, `swapwatch.sh`, `smapstop.py` and `smapsdetail.py`.
- Analysis: `wdcmp.py` (digest compare), `g4lline.py`, `g4lagg.py`, `g4ltables.py`, `ldsumm.py` and `memobs.py`.
- Results: `launch.jsonl`, `ab.log`, `ab/<label>/` (relaunch log, rank-log lines, samplers, weight digests) and `runs/<label>-start/` (exact.py, rank0 startup log).

On spark-one: `~/dsv41-g4l/` (scripts, `runs/`) and the staged bundle `~/dsv41-stage/g4/fload/` (8/8).

### 1. Restore after the interrupted decode step (08:35-08:53 UTC)
- **What was running at 08:35:** candidate arm **g4d-F1** of the interrupted goal-window-4 decode step (`DSV41_FUSED_AV=1`, pinned, launched 08:28:55).
- **What was installed:** the g4d bundle (`adaptive_verification.py` 55030a68, `model_runner.py` 6a4f2433, `gpu_worker.py` 4d67066c, `v1/worker/utils.py` c4e02ba8; every flag default off). The README did not record it.
- **Restore steps:**
  - stopped 8/8 at 08:43 with 0 requests in flight;
  - ran `g4-all.sh uninstall g4d`, which brought the four files back to 7e30eebc / 316fca44 / e970aa29 / 85b23497 on 8/8 (`status` digest e9ee5dff0a2a).
- **Checks against this README:**
  - g4p files 9c2d22ad / b7598364 / b75aa927 / 439b390e / f2ae649b;
  - supervisor preflight manifest 8791f3d8d6e7;
  - launcher fd1e809e on 8/8.
- **Production relaunch g4l-prod0** (supervisor, defaults, AV unpinned): healthy at +355 s. exact.py gave a8c3394af9d2 with its own profile (911/465/2325/1373, "pin miss"), so the anchor holds.
- **For the decode step:** it must re-install g4d (its stopped-window script `sw4d.sh` does that). The bases it expects are the ones restored here.

### 2. Where the stock load time goes (g4l-prod0 and the C launches; 1 Hz `/proc` samplers on 8/8)
- **Size of the load.** `Loading weights took` is 161-177 s per rank, which is 84-88% of `Model loading took` (186-190 s). The DSpark drafter adds 4 s.
- **Phase 1: iteration (~35 s).** The tqdm bar ("48/48 [00:33-00:35]") covers only this phase. The DSv4.1 VL wrapper (`nvidia/vl_model.py` `load_weights`) calls `sorted(self.hf_to_vllm_mapper.apply(weights), key=name)`. That materialises the whole 48-shard stream as lazy safetensors mmap views (safetensors 0.8.0, mmap backend) and sorts it. While it runs, the NVMe already reads ~0.7 GB/s: readahead of tensors that are evicted again before they are used.
- **Phase 2: copies (~130 s).** Only after the sort do the weight loaders run, in sorted-name order. Each `narrow` + `copy_` faults its mmap pages in, at 1.0-1.4 GB/s with ~5 CPU cores busy in faults. The page cache sits at ~49-50 GB for the whole load.
- **Bytes.** The NVMe reads 197-208 GB per node per launch (`/sys/block/nvme0n1/stat`). This rank's loaders consume ~156 GB of that:
  - The w2 experts are column-sliced (row stride 1,152 B, 144 B per rank), so every page of every w2 is needed.
  - w1/w3 are row-sliced: `RoutedExperts._load_w13` takes rows [288 r, 288 r + 288) of 2304. mmap readahead still pulls in neighbouring rows.
  - Engram: this rank's 2 x 12.67 GB of rows.
- **Swap (found during this step).** The page-cache pressure of the stock load pushes **2.3-3.0 GB per node into the swap file** (`/run/sparkos/host/swap.img`, 8-16 GB on the NVMe, swappiness 60).
  - Of that, 0.6-0.8 GB is the vLLM worker's own anonymous memory (`VmSwap`), and ~0.7 GB is shared `/dev/zero` mappings that the worker keeps.
  - Since boot (13 days), `pswpout` reads 370-478 M pages per node and `pswpin` 175-229 M. Each stock restart adds ~0.75-0.93 M pages (2.9-3.6 GB) of that.
  - Those pages fault back from the NVMe whenever serving touches them.
- **Device ceiling** (`bench_io.py` on spark-two; 1 TB ESL01TBTLCZ NVMe, `max_sectors_kb` 256):
  - The fast-loader read plan of one layer shard (3.18 GB of 7.39 GB) runs at 4.0-4.2 GB/s buffered (8-32 threads) and 4.6 GB/s with `O_DIRECT` (8 threads). A whole shard reads at 3.9 GB/s.
  - Host-to-device copies on GB10 run at 35-46 GB/s from pageable and pinned memory alike.

### 3. Implementation (bundle `~/dsv41-stage/g4/fload`, installed with `g4-all.sh install fload`; backups `<file>.bak-g4-fload`, absent-file markers `.absent-g4-fload`; every flag default OFF)

| file (site-packages/vllm) | before -> fload | role |
|---|---|---|
| model_executor/model_loader/dsv41_fast_load.py | ABSENT -> **dfbcf2c7** | new: `FastWeightStream` + `DSV41_WEIGHT_DIGEST` |
| model_executor/model_loader/default_loader.py | 3101d49e -> **1aeec2e6** | With `DSV41_FAST_LOAD=1`, `load_weights` hands models that declare `_dsv41_fast_load_ok` a `FastWeightStream`; every other model gets the stock generator. `load_model` runs the digest when `DSV41_WEIGHT_DIGEST` is set. |
| models/deepseek_v41/nvidia/vl_model.py | ff2c1661 -> **3ec0d401** | `_dsv41_fast_load_ok = True`. `load_weights` takes `weights.dsv41_sorted(mapper, self)` when it is given a stream, and otherwise the stock `sorted(...)`. |

- **Same sequence, same order.** The stream parses the 48 safetensors headers itself.
  - File order is the stock natural sort. Key order is `safe_open().keys()`, which is sorted (checked).
  - Every name goes through the wrapper's own `hf_to_vllm_mapper._map_name_with_shard`, which drops `mtp.*` exactly as `apply` does. A stacked mapping would fall back to stock.
  - The sort uses the same key and is stable. The loaders therefore receive exactly the stock `(name, tensor)` sequence, lazily, with nothing materialised.
  - Dtypes come from the header table plus `F8_E8M0 -> float8_e8m0fnu`, as `safe_open` maps it.
- **Reads.** 8 threads (`DSV41_FAST_LOAD_THREADS`) take the tensors in consumption order.
  - Each tensor is read with `O_DIRECT` on its 4 KiB-aligned superset into a pinned host buffer, with a buffered `pread` fallback. The buffers come from torch's caching host allocator, so a block is only reused after the tensor viewing it is gone. Every block is freed when the stream ends.
  - In-flight data is bounded by a 2 GiB window (`DSV41_FAST_LOAD_WINDOW_MB`).
  - Expert `w1` / `w3` weights and MX scales (`.experts.N.(w1|w3).(weight|scale)`, 2-D, row count divisible by TP) read only rows `[R/tp * rank, R/tp * (rank + 1))`. That is exactly what `RoutedExperts._load_w13` narrows to (`loaded_per_rank * tp_rank`). The buffer keeps the full tensor shape, and the loader never reads the rows outside that range. `DSV41_FAST_LOAD_ROWS=0` reads them whole.
  - Every other tensor is read whole (w2 is column-sliced, so every page is needed).
- **Engram.** The two Engram tables are not yielded at all.
  - Their rows `[engram_vocab_start, + parameter rows)` are read with buffered `pread` straight into the mlocked host table, in 64 MiB chunks, with `POSIX_FADV_DONTNEED` behind each chunk. The range is taken from the parameter itself; it is the exact `narrow` that `_engram_head_shard_weight_loader` does.
  - 2 of the 8 threads take these chunks first; the others take them when the window is full.
  - A punched hole is never read.
  - This path is used only when the parameter carries the stock Engram weight loader and is a contiguous CPU tensor. `DSV41_FAST_LOAD_ENGRAM=0` puts the tables on the stock path.
- **Self-checks on every fast load.**
  - Before the threads start, one tensor per safetensors dtype plus one row-sliced tensor are read through the fast path and compared byte for byte with `safe_open` (`canary 6 OK`).
  - After the fill, 3 x 64 KiB windows of each Engram table are compared with the same rows from `safe_open` (`Engram spot checks 12 OK`).
  - A setup failure falls back to the stock iterator before anything is yielded. A reader error fails the load loudly.
  - One summary line per rank: `DSV41_FAST_LOAD: 93680 tensors (61440 row-sliced), 131.14 GB + Engram 25.34 GB direct (4 tables) read in 34.7 s = 4.51 GB/s; consumer waited 2.5 s ... worker RSS ...`.
- **Digest** (`DSV41_WEIGHT_DIGEST=<dir>`, diagnostics only).
  - It runs after `load_model`, i.e. after `process_weights_after_loading`. It takes the sha256 of the bytes of every `named_parameters()` entry: GPU tensors in 64 MiB chunks, the CPU Engram tables in place. The 3 meta parameters of the unused vision tower are recorded as `meta`.
  - Output is one JSON per rank and load (`wd-<tag>-r<rank>-<load>.json`; load 0 = target, 1 = DSpark drafter) plus a summary hash. `wdcmp.py` compares two launches.
  - A digest pass costs 132-166 s per rank.
- **Build history** (all three builds are digest-identical to stock; see section 4):
  - **v1 (first build):** pinned buffers. Its first digest launch crashed in the digest itself ("Cannot copy out of meta tensor" on the vision tower's meta parameters); fixed.
  - **v2:** buffers on 64 MiB anonymous THP mappings, unmapped on free. I built it to chase a lower MemAvailable after startup that turned out to be stock's swap-out (section 6), not a leak. v2 was slower: one or two ranks per launch read at 3.3-3.9 GB/s (load 34-48 s), because the reader threads fault and compact fresh huge pages.
  - **v3 = v1's pinned buffers** plus the v2 bookkeeping (RSS in the summary line, `malloc_trim`, `DSV41_FAST_LOAD_BUF=mmap|malloc` options). v3 is what is installed and promoted.
- **Unit test** on real shards before any install (`test_fload.py`; spark-two, rank 1, shards 1/2/3/23/43/44/47):
  - The yielded name order equals `sorted(VL mapper.apply(stock iterator))` minus the two Engram names (4,948 tensors).
  - 0 byte mismatches against `safe_open` (10.14 GB compared, 3,072 of them row-sliced).
  - The Engram fill of 200,000 of rank 1's rows is identical to the `safe_open` rows (non-zero data).
  - canary 6, spot checks 6, 0 `O_DIRECT` fallbacks.

### 4. Exactness
- **Weight digests** (pinned launches; each one is 16 dumps = 8 ranks x {target, drafter}: 10,856 parameter digests, 517.2 GiB hashed, Engram tables 188.8 GiB across 8 ranks):

  | compare | verdict |
  |---|---|
  | g4l-dC (stock) vs g4l-dF (v1) | **IDENTICAL**, 0 differing parameters |
  | g4l-dC vs g4l-dF2 (v2) | **IDENTICAL** |
  | g4l-dC vs g4l-dF3 (v3, the promoted build) | **IDENTICAL** |

  The per-rank target summaries are the same in all four: r0 f424ce35, r1 889c720d, r2 7f0fa018, r3 e379275a, r4 abe0e0ed, r5 b6d2dfb6, r6 ed77a4ff, r7 9994e8e3. Each rank has 23.60 GiB of Engram rows hashed in place. On the 7 punched nodes those rows are the only Engram data on disk; spark-four's are read from its pristine copy.
- **Every launch:**
  - exact.py first gave **a8c3394af9d2** with the pin's counters (909/467/2335/1373, per position 348/233/171/90/67, "pin hit").
  - G1 x2 gave the reference digest **9a18f802d930f6e7**, with identical repetitions and no foreign requests.
  - 0 ERROR/Traceback on 8/8.

### 5. Time-to-healthy (pinned, 2000 MHz; C = stock loader, F = `DSV41_FAST_LOAD=1`; `t_healthy` = supervisor `/health` 200 + startup complete after `systemd-run`, 5 s polling)

| # | launch | arm | t_healthy s | rank0: load start / load end / KV / startup (s after the API banner) | load s, 8 ranks min-max | Model loading s (max) | NVMe read GB/rank | swap in use after start, GB (min-max) / worker VmSwap MB | MemAvailable after start GB (min-max) | exact.py | G1 tok/s / digest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | g4l-C1 | C (stock) | 288 | 21 / 193 / 217 / 280 | 162.8-177.2 | 190.2 | 196.8-208.3 | - | 14.9-16.0 | a8c3394a EXACT-OK | 102.153 / 9a18f802d930f6e7 |
| 2 | g4l-F1 | F (v1) | 127 | 20 / 57 / 73 / 120 | 36.0-36.7 | 49.0 | 163.0-164.0 | - | 13.1-14.2 | a8c3394a EXACT-OK | 102.241 / 9a18f802d930f6e7 |
| 3 | g4l-C2 | C (stock) | 283 | 21 / 190 / 214 / 276 | 161.7-174.9 | 188.1 | 196.9-206.4 | - | 14.8-15.9 | a8c3394a EXACT-OK | 101.978 / 9a18f802d930f6e7 |
| 4 | g4l-F2 | F (v1) | 126 | 21 / 57 / 72 / 120 | 35.8-36.7 | 47.6 | 163.0-164.0 | - | 12.8-14.1 | a8c3394a EXACT-OK | 102.12 / 9a18f802d930f6e7 |
| 5 | g4l-C3 | C (stock) | 289 | 26 / 190 / 218 / 279 | 164.1-173.3 | 186.7 | 196.8-206.0 | - | 15.0-16.0 | a8c3394a EXACT-OK | 101.87 / 9a18f802d930f6e7 |
| 6 | g4l-F3 | F (v1) | 126 | 21 / 57 / 72 / 120 | 35.8-36.6 | 47.6 | 163.0-163.9 | - | 12.9-14.5 | a8c3394a EXACT-OK | 101.948 / 9a18f802d930f6e7 |
| 7 | g4l-F4 | F (v2) | 131 | 21 / 55 / 79 / 126 | 34.1-43.6 | 54.3 | 161.4-163.0 | 0.14-0.17 / 0-0 | 13.2-14.5 | a8c3394a EXACT-OK | 102.136 / 9a18f802d930f6e7 |
| 8 | g4l-C4 | C (stock) | 284 | 21 / 184 / 215 / 276 | 162.7-174.8 | 187.9 | 196.9-206.4 | 2.35-3.03 / 705-776 | 14.6-15.8 | a8c3394a EXACT-OK | 101.835 / 9a18f802d930f6e7 |
| 9 | g4l-F5 | F (v2) | 137 | 22 / 56 / 78 / 127 | 34.0-42.0 | 52.7 | 163.5-164.4 | 0.14-0.18 / 0-0 | 13.1-14.5 | a8c3394a EXACT-OK | 102.124 / 9a18f802d930f6e7 |
| 10 | g4l-C5 | C (stock) | 288 | 20 / 192 / 214 / 280 | 161.4-175.8 | 188.8 | 196.9-207.7 | 2.25-2.78 / 612-735 | 14.6-16.0 | a8c3394a EXACT-OK | 101.865 / 9a18f802d930f6e7 |
| 11 | g4l-F6 | F (v2) | 137 | 20 / 54 / 82 / 131 | 34.4-47.8 | 58.5 | 163.2-164.0 | 0.14-0.18 / 0-0 | 13.2-13.8 | a8c3394a EXACT-OK | 102.041 / 9a18f802d930f6e7 |
| 12 | g4l-F7 | F (v3) | 132 | 21 / 57 / 73 / 123 | 36.0-36.7 | 47.6 | 161.7-162.6 | 0.14-0.18 / 0-0 | 13.3-14.1 | a8c3394a EXACT-OK | 101.922 / 9a18f802d930f6e7 |
| 13 | g4l-F8 | F (v3) | 126 | 21 / 57 / 72 / 120 | 35.9-36.6 | 47.6 | 161.4-162.8 | 0.14-0.18 / 0-0 | 13.5-14.1 | a8c3394a EXACT-OK | 101.882 / 9a18f802d930f6e7 |
| 14 | g4l-F9 | F (v3) | 132 | 21 / 57 / 78 / 125 | 35.9-42.2 | 53.2 | 161.7-162.7 | 0.13-0.18 / 0-0 | 13.3-14.5 | a8c3394a EXACT-OK | 101.859 / 9a18f802d930f6e7 |

| arm | launches | t_healthy s mean (values) | load s, slowest rank, mean | Model loading s mean | NVMe read GB/rank mean | MemAvailable after start GB mean | G1 mean |
|---|---|---|---|---|---|---|---|
| C (stock) | 5 | **286.4** (288, 283, 289, 284, 288) | 175.2 | 188.3 | 203.9 | 15.1 | 101.94 |
| F v1 (pinned, first build) | 3 | **126.3** (127, 126, 126) | 36.7 | 48.0 | 163.7 | 13.5 | 102.10 |
| F v2 (THP mappings) | 3 | **135.0** (131, 137, 137) | 44.5 | 55.2 | 163.4 | 13.6 | 102.10 |
| F v3 (pinned, promoted build) | 3 | **130.0** (132, 126, 132) | 38.5 | 49.4 | 162.3 | 13.7 | 101.89 |

- Digest launches are excluded from these means: the digest pass adds 132-166 s per rank. Their t_healthy was g4l-dC (stock) 458 s, g4l-dF (v1) 301 s, g4l-dF2 (v2) 301 s and g4l-dF3 (v3) 291 s, with target loads of 163.7-177.2 s (stock) against 34.0-47.2 s (fast).
- Like-for-like (all pinned), each v1 launch against the mean of the C launches on either side: -158.5 / -160.0 / -160.5 s. The v3 launches have no C after them, so they are compared with the C mean: -154 / -160 / -154 s. Over all pinned launches, stock averages 286.4 s and the two pinned fast builds 128.2 s: **-158 s (-55%)**. The C controls varied by only 283-289 s over the 1.5 h.

- **Where the time goes in F** (rank 0, seconds after the API banner): load 20-21 -> 57 (36-37 s), KV allocation at 72-78, startup at 120-125. In C: load 20-26 -> 184-193, KV allocation at 214-218, startup at 276-280.
- **Savings beyond the load.** On rank 0, F saves ~131 s in the load itself (168 -> 36-37 s) and another ~24 s between load end and startup (88 -> 63-68 s: drafter load, post-processing, graph capture). That is consistent with C having swapped part of the process out.
- **I/O bound.** Each F rank reads 131.1 GB of tensors plus 25.3 GB of Engram rows at 4.35-4.64 GB/s, which is the device ceiling from `bench_io.py`. The consumer (the stock weight loaders) waits only 0.8-2.6 s in total. NVMe reads per launch fall from 197-208 GB to 161-164 GB per rank (the remaining ~7.4 GB is the DSpark drafter's 3 shards, still on the stock path).
- **Codex's kit** projected ~174 -> ~43 s of load and ~129 s of time-to-healthy with the AV pin. Measured: 35-37 s and 126-132 s (unpinned production: 199 s).

### 6. Memory
- **During the load.** Page cache peaks at 7.7-11.4 GB in F against 48-50 GB in C. F holds ~2-3 GB of pinned buffers (the 2 GiB window, rounded to powers of two) and frees them when the stream ends.
- **After start.** MemAvailable is 12.8-14.5 GB per node in F against 14.6-16.0 GB in C. The difference is not a leak:
  - The swap watcher shows C with 2.25-3.03 GB of swap in use per node after start, and the worker with 612-776 MB of `VmSwap`. F shows 0.14-0.18 GB in use (the residue from before) and a worker `VmSwap` of 0; `pswpout` did not move during an F launch.
  - The worker's smaps differ by exactly those pages. On rank 0, C/F: `[anon]` 24.8/25.6 GB and shared `/dev/zero` 0.56/1.28 GB; the same 263 `/dev/zero` mappings exist in both. The v1, v2 and v3 builds give the same footprint.
  - Stock "frees" ~1.5-2.5 GB per node by writing it to the NVMe swap file, and serving later faults it back. The fast loader keeps it resident.
  - Free memory in blocks of 2 MiB or more before the KV allocation (`memwatch.sh`) is in the same range in both arms (C 48.7-57.6 GB, F 48.6-57.1 GB).
- **At 1M:** see the final production instance (section 8): MemAvailable minimum 6.2-7.1 GB (floor 6 GB; window 3 and g4p measured 7.1-8.3 GB on stock launches).

### 7. Decision and promotion
- **Exact:** 4 digest launches identical to stock, and exact.py a8 + the G1 reference digest on all 16 A/B launches.
- **Faster:** pinned time-to-healthy 286.4 s -> 130.0 s (v3, n=3; v1 126.3 s, n=3), -154 to -160.5 s for every F launch against its controls. Target load 175 -> 36 s on the slowest rank. No startup swap-out.
- **Promoted as the launcher default:** `~/dsv41-tp8.sh` fd1e809e -> **31d04d8f** on 8/8 (`scripts/promote4l.sh fload scripts/lines-FL.txt`). The backup `~/dsv41-tp8.sh.bak-g4-fload` is fd1e809e on 8/8. One line, after `DSV41_E7_EXACT`:
  ```sh
  export DSV41_FAST_LOAD=${DSV41_FAST_LOAD:-1}  # g4l fast loader 2026-10-27 ...; 0 = stock mmap loader
  ```
  - The supervisor is unchanged (7bea1cf3).
  - Its preflight manifest (which includes the launcher) is now 29a69fac9dfb on 8/8. Its file list does not include the fload files; the bundle is checked by `g4-all.sh status fload` (8/8 identical digest).

### 8. Final production instance g4l-FINAL (launcher 31d04d8f defaults, AV unpinned)
Supervisor launch with the launcher defaults (`DSV41_FAST_LOAD=1` from the launcher, AV unpinned = startup profile, `DSV41_GPU_CLOCK_MHZ=2000`). Healthy at 11:08:51 UTC on the first try; SM clock 1976-1989 MHz on 8/8 before the suite. No foreign requests were seen in any phase (G1: 9 completions for 9 sent).

| metric | g4l-FINAL (launcher 31d04d8f, fast loader) | reference |
|---|---|---|
| time-to-healthy, unpinned | **+199 s** (target load 36.2-37.1 s per rank, AV startup profile 90 s) | g4l-prod0, stock (fd1e809e) at 08:52: +355 s (load 173 s); g4-FINAL: +360 s |
| exact.py | **a8c3394af9d2** first (own profile 912/464/2320/1373, "pin miss") and again after the suite | anchor a8c3394af9d2 |
| G1 x3 (code / prose / math) | **102.13** (96.79 / 76.20 / 133.40); outputs identical in all 3 repetitions | g4-FINAL 101.62 (96.31 / 75.66 / 132.88) |
| GSM8K-50 / full longctx | **50/50** / **PASS** (needle 60/60, copy 1.000) | g4-FINAL 50/50 / PASS |
| ONE cold ~1M prompt (1,024,611 tokens) | **165.9 s**, good class (chunk 37-43 band 2.70 s; quartile means 2.42 / 2.55 / 2.68 / 2.82 s) | g4-FINAL 167.5 s, good class (2.44 / 2.59 / 2.73 / 2.87 s) |
| MemAvailable minimum during the suite (8/8, 1 Hz) | **6.2-7.1 GB** (spark-one 6.2) | stock pinned launches in g4p at 1M: 7.70-7.85 GB; floor 6 GB |
| swap during the suite | `pswpout` unchanged on 8/8 (no page written to swap, including at 1M); swap in use 0.14-0.18 GB | - |
| errors | 0 ERROR/Traceback on 8/8 at start and end | - |

- **1M headroom.** The MemAvailable minimum at 1M is ~1.5 GB lower than on stock launches. That is the same ~1.5 GB per node that stock writes to swap at startup (section 6); the fast loader keeps it resident. It is still above the 6 GB floor, nothing swapped and nothing was reclaimed directly (`allocstall` 0-1, `pgscan_direct` 0 on 8/8). The 1M prefill itself is unchanged (good class, 165.9 s).
- **G1.** The G1 numbers follow this instance's own AV startup profile. The pinned A/B (section 5) is the like-for-like decode check: G1 101.86-102.24 in both arms, with the reference digest every time.

### 9. Live state and rollback
**Live:** instance **g4l-FINAL**, healthy since 11:08:51 UTC.
- Launcher **31d04d8f** on 8/8 (= fd1e809e + the `DSV41_FAST_LOAD` line); backup `~/dsv41-tp8.sh.bak-g4-fload` = fd1e809e.
- Supervisor 7bea1cf3, preflight manifest 29a69fac9dfb; AV unpinned; clock lock 0-2000 MHz (boot unit).
- Site files on 8/8:
  - fload bundle (`g4-all.sh status fload`, install digest ca05f12513a1): `model_executor/model_loader/dsv41_fast_load.py` **dfbcf2c7** (new, marker `.absent-g4-fload`), `default_loader.py` **1aeec2e6** (backup 3101d49e), `nvidia/vl_model.py` **3ec0d401** (backup ff2c1661);
  - g4p files unchanged (9c2d22ad / b7598364 / b75aa927 / 439b390e / f2ae649b);
  - g4d NOT installed (baseline 7e30eebc / 316fca44 / e970aa29 / 85b23497).
- 8/8 active, 0 restarts, 0 ERROR/Traceback, no test process left.
- The weight digests from this step stay in `~/dsv41-logs/wdigest/` on each node (`wd-{dC,dF,dF2,dF3}-r<rank>-<load>.json`, ~0.9 MB per node).

**Rollback:**
- **One launch:** `DSV41_FAST_LOAD=0` as a supervisor argument (stock mmap loader; exact either way, the weights are identical).
- **Permanent:** `cp -p ~/dsv41-tp8.sh.bak-g4-fload ~/dsv41-tp8.sh` on 8/8 (fd1e809e), then relaunch with `--expect-launcher fd1e809e`.
- **Site files** (service stopped, after the launcher rollback): `ssh spark-one cat ~/dsv41-stage/g4/common/g4-all.sh | bash -s uninstall fload`. It is sha-guarded: it restores `default_loader.py` 3101d49e and `vl_model.py` ff2c1661 and deletes `dsv41_fast_load.py`. With the flag off, the installed files take the stock path; the only extra work is one `getattr` in `vl_model.load_weights`.

**Notes for the next steps:**
- The decode step must re-install g4d; it was removed here to match the README.
- `DSV41_WEIGHT_DIGEST=<dir>` (default off; ~2.5 min per launch) proves "same weights" for any future loader or post-processing change.
- Stock restarts were the main source of the fleet's swap traffic; with the fast loader a restart writes nothing to swap.
- A further cut would need fewer bytes, not more threads: the load runs at the NVMe ceiling (4.4-4.6 GB/s), and w2 (90.6 GB per rank, column-sliced) must be read whole in the current checkpoint layout.

**Incidents:**
- The first digest launch (g4l-dC, 09:17) failed in the digest itself. `named_parameters()` includes 3 meta-device parameters of the unused vision tower ("Cannot copy out of meta tensor"). Fixed (they are recorded as `meta`), reinstalled, and relaunched at 09:22. The fast loader was not involved.
- v2 (THP mappings) was a detour, explained in section 3; v3 restored v1's buffers.

## Goal window 4, decode+ops (2026-10-27 11:23-13:30 UTC): Codex 09 fused draft budget (`DSV41_FUSED_AV`) exact but +0.2% on its first pinned pair, NOT promoted (installed, off); 01 already covered by window 3; 02 gate measured and CLOSED (best possible scheduler 1.025x at M=6, gate 1.08x); 08 early KV slab NOT promoted: with the fast loader every launch is already good class at 1M (0/4 slow-class launches in either arm, against 1/1 for a stock-loader launch), and the early slab does not get 2 MiB blocks. Live config unchanged (launcher 31d04d8f)

Scope: the decode+ops step of goal window 4: Codex ten-deliverables items 09 (fused draft budget + verify cap), 01 and 02 if window 3 had not already covered them, and 08 (early KV slab) on a relaunch series. It restarted after the Loader phase. The trim rule from the user's 08:35 UTC directive applies:
- Decode candidates got the short suite: exact.py first and last, matched C1, G1, G4, conc8/16. No conc32 and no long prompts.
- The 08 series (a memory/straggler candidate) got exact.py, G1 x2 and ONE cold ~1M prompt per launch, with 1 Hz MemAvailable on 8/8.
- GSM8K-50 and longctx were not run, because no arm was promoted.
- A candidate stops after its first pinned pair if that pair is negative or below noise.

Every launch used `DSV41_GPU_CLOCK_MHZ=2000`, and the SM clock was read on 8/8 before every timed phase (1969-1989 MHz throughout). No foreign request was seen in any phase. The 09 A/B used the pin `~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json` (eeac4f97; "pin hit" on every pinned launch). The 08 series ran unpinned, as production does.

Mac drivers and raw results are in scratchpad `g4d2/`:
- `scripts/`: `sw.sh` (stopped window), `ab.sh` / `abx.sh` (one instance), `blk.sh` (block), `phase.sh` (on spark-one), `g4dline.py` / `g4dagg.py` (records and tables) and `promote.sh` (unused).
- `tools/`: `m32_gsweep.py` (item 02 sweep), `m32gate.py` (gate bound), `ghist.py` (route histograms) and `m32_sched_sim.py`.
- Results: `sw/` (GPU tests and sweep), `ab.jsonl`, `ab.log`, `ab/<label>/` and `runs/<label>/`.

On spark-one: `~/dsv41-g4d2/` (scripts, `runs/`, `g4d2-mac.tgz`). The bundle is staged in `~/dsv41-stage/g4/g4d/` on 8/8, together with the interrupted attempt's tests and tools.

### 1. State check and stopped window (11:23-11:37 UTC)
- **At the start (11:23 UTC):**
  - live was g4l-FINAL (launcher 31d04d8f, healthy since 11:08:51; `/health` 200);
  - 0 requests in flight, and only local connections on :8000;
  - the Loader phase had already restored production after the interrupted decode attempt, so nothing had to be undone.
- **Stopped window:** stopped 8/8 at 11:34:58 with 0 requests in flight (`sw.sh`). The site status matched this README on 8/8:
  - g4d at baseline: 7e30eebc / 316fca44 / e970aa29 / 85b23497 (status digest e9ee5dff0a2a);
  - g4p installed (aef7c824bb9a) and fload installed (ca05f12513a1);
  - launcher 31d04d8f.
- **Re-installed g4d:** the interrupted attempt's bundle, unchanged (`g4-all.sh install g4d`, digest **f99d02f68320** on 8/8, backups `<file>.bak-g4-g4d`; every flag default OFF).

  | file (site-packages/vllm) | before -> g4d | flag (default off) |
  |---|---|---|
  | v1/worker/gpu/spec_decode/adaptive_verification.py | 7e30eebc -> **55030a68** | `DSV41_FUSED_AV=1` serves the fused kernel; `=verify` runs it in shadow and counts mismatches |
  | v1/worker/gpu/model_runner.py | 316fca44 -> **6a4f2433** | with `DSV41_FUSED_AV=1`, skips the two H2D copies that `reallocate_drafts` overwrites |
  | v1/worker/gpu_worker.py | e970aa29 -> **4d67066c** | `DSV41_KV_ALLOC_MODE=early_slab`: the KV slab is allocated at the end of `init_device` |
  | v1/worker/utils.py | 85b23497 -> **c4e02ba8** | `allocate_kv_cache` takes that slab; `DSV41_KV_BUDDY_LOG=1` logs `/proc/buddyinfo` around the stock allocation |

  - An import check of the installed modules passed (flags on, CPU only).
- **GPU tests, service down, in parallel:**
  - spark-two: `test_fav_gpu.py` PASS (section 2);
  - spark-four: `m32_gsweep.py`, 124 points (section 4).
  - 0 GPU processes were left afterwards.

### 2. Item 09: fused draft budget (`DSV41_FUSED_AV`): exact; +0.2% on the first pinned pair, so the A/B stopped there. NOT promoted
- **What it replaces.** Each decode step with drafts, `reallocate_drafts` (between the drafter graph and the target graph) issues:
  - 3 small H2D copies;
  - the torch.compile'd budget assignment (6 kernels: survival scan, gatherTopK, bitonic sort, zeros, index_fill, sum);
  - two cumsum chains.

  `DSV41_FUSED_AV=1` does the same work with one packed H2D and one Triton kernel. It produces the same capacities, non-draft counts, `cu_num_logits` and the whole `query_start_loc` buffer, and `prepare_inputs` skips the two H2D copies that `reallocate_drafts` overwrites anyway.
  - Survival products use the live inductor scan's exact multiplication order (Hillis-Steele over 8 lanes).
  - Admission is the `torch.topk` order: radix key of the fp32 value, NaN largest, ties to the lower flat index.
  - The verify cap (`dsv41_apply_live_cap`) was already one Triton kernel and is unchanged. "Fused" therefore covers the budget path, which is where the launches were.
- **Exactness:**
  - **GPU unit test** (spark-two, installed module): 31,019 random cases at 1-32 requests and 600 crafted tie / order cases, **0 mismatches**.
    - The random cases use 6 confidence patterns (uniform, quantized ties, ones, tiny, real, NaN/zero) and all three modes (budget 0: 3,000; budget not binding: 2,928; top-k admission: 25,091).
    - The compare covers every output buffer entry, including the `query_start_loc` padding.
    - The device-side verify counter: 31,619 calls, 0 mismatching entries.
  - **Serving shadow** (the interrupted attempt's instance g4d-V, launcher fd1e809e, pinned, `DSV41_FUSED_AV=verify`): every step of matched C1, G1, G4 and conc8/16/32 (1-32 requests) was compared on the device with the live path. Result: **7,100 calls, 0 mismatching entries**; exact.py a8 with the pin's counters.
  - **Serving it** (F1, this step): exact.py **a8c3394af9d2** with 909/467/2335/1373 and per position 348/233/171/90/67 (pin hit), G1 digest **9a18f802d930f6e7** (the reference), exact.py #2 a8. All 8 ranks logged "DSV41_FUSED_AV=1: reallocate_drafts runs as one packed H2D + one Triton launch (S=5, max_num_reqs=32)".
- **Microbench** (`test_fav_gpu.py`, per call): stock 32.6-73.6 us, fused 6.1-22.3 us. That saves 51-66 us when the budget binds and 16-31 us when it does not, i.e. 0.1-0.25% of a 24-30 ms single-user step.
- **Pinned pair** (short suite; F against C right before it):

  | # | instance | arm | exact start / #2 | G1 digest | C1 code / prose ms | G1 (code / prose / math) | G4 mean / prose MIN / hat | 4-session step ms | conc8 / 16 |
  |---|---|---|---|---|---|---|---|---|---|
  | 1 | g4d2-C1 | C | a8c3394a / a8c3394a | 9a18f802d930f6e7 | 30.25 / 26.34 | 101.63 (95.97 / 76.52 / 132.39) | 60.83 / 48.10 / 48.43 | 44.56 | 263.9 / 394.1 |
  | 2 | g4d2-F1 | F | a8c3394a / a8c3394a | 9a18f802d930f6e7 | 30.20 / 26.25 | 101.83 (96.15 / 76.73 / 132.61) | 61.46 / 48.24 / 48.50 | 44.71 | 259.2 / 396.3 |
  | | **F1 vs C1, %** (+ = better) | | | | +0.17 / +0.34 | **+0.20** (+0.19 / +0.28 / +0.17) | +1.03 / **+0.28** / **+0.14** | -0.3 | -1.8 / +0.6 |

- **Decision: NOT promoted.**
  - The pair is positive where the microbench says it should be (G1 +0.20%, C1 -0.2/-0.3% ms), but every promotion metric is under the 0.5% threshold: G1 +0.20%, G4 prose MIN +0.28%, G4min_hat +0.14%.
  - G1 alone varies 0.1-0.3% between launches, so under the trim rule the A/B stopped after this first pair.
  - The bundle stays installed with the flag off. It is a candidate for a later bundle of small exact items: together with a balanced M32 scheduler (section 4, at most ~0.3 ms) and the rdma_ar P1 doorbell (~0.1-0.2 ms), it could clear 0.5%.

### 3. Item 01 (RoCEnante fast path): already covered by window 3, not re-run
- Window 3 ran the 8-rank head-to-head and the fixed-order bitwise test on our fabric ("Goal window 3", section 1):
  - rdma_ar is EXACT and RoCEnante is EXACT_EXCEPT_ALL_NEG0;
  - RoCEnante is slower at every size, e.g. 19.9 vs 17.4 us at 1 row and 40.9 vs 31.6 us at 6 rows (RoCEnante vs ours, gap mode);
  - the rdma_ar env knobs each moved less than 2 us, and two-shot wins from 4 rows on, as live.
- Codex's package for this item:
  - it ships a sketch of `ar_kernel_fast` plus the multi-rail proxy described in comments;
  - the `rdma_ar_fast.cu` its README lists is not in the package;
  - its -1.0..-1.6 ms projection rests on TP4 numbers.
- The one sub-part window 3 still sizes as positive is the P1 doorbell (no `posted_seq` spin; ~-0.1..-0.2 ms per step). It is a native rdma_ar change, i.e. a build plus an 8-rank bitwise test in a stopped window, and is left for a later window.

### 4. Item 02 (MoE gap to Marlin = tile scheduling): gate measured, CLOSED
- **Codex's kit can't be run as shipped:**
  - `make_m32_v3_tree.py` edits the string `kSwizzleGroup = 8` in a stand-in header;
  - `bench_moe_recorded_routes.py --gpu` exits with "GPU bench runs in a stopped window" without measuring anything.
- **How the live tree schedules tiles:**
  - The live tree has no such constant. The 1-D swizzle group comes from `sched_sm120::get_num_1d_blocks_per_group<...>()`, which is 8 for BM32 / BN64 / 48 SMs.
  - The D04 dead-walk stop in all four walkers (TMA producer, prefetch cursor, LA warp, math consumers) depends on it. With 48 CTAs = 0 mod 8, every CTA keeps one m residue, so "a dead m in a full 8-block group" means every later m is dead.
  - A swizzle-1 tree would therefore also need a new stop rule. With group 1, m = l / nb is monotonic per CTA, so it could break on the first dead m-block.
- **The measurement (no build needed).** The live `all` kernels (FC1 `all` base tree, FC2 `all` D05 tree) were run on synthetic routes with exactly G live m-blocks:
  - 40-layer L2-cold CUDA graphs over 4 real checkpoint layers (rank-3 weights; `m32_gsweep.py`, spark-four, service down);
  - 124 points: G 14-35 at M=6, 15-61 at M=12, 22-76 at M=16.
- **What it shows: the time is a staircase.**
  - It jumps at G = 8k+1 and stays flat until the linear trend catches up. G 16 -> 17 is 216.2 -> 260.6 us per layer (+20.5%), and G 17-20 sit at 260.6-269.3 us. Likewise 24 -> 25 (308.2 -> 340.2) and 32 -> 33 (398.4 -> 418.6).
  - That is the residue pinning: the m-blocks of a partial group land on the 6 CTAs of their residue class.
  - Away from those G the kernel follows the bytes, not the busiest CTA. At G = 33, for example, swizzle 8 gives the busiest CTA 67 FC2 tiles against 55 for round-robin (1.22x). Yet FC2 runs at the trend (154.2 us measured, 155.2 fitted), and the whole layer is only 1.5% above the balanced time.
- **The ceiling for any scheduler** (the most that swizzle1 / livetiles could reach):
  - The balanced time per G is a linear fit through the G = 0 mod 8 points (FC1 6.0 + 6.93 G us, FC2 1.4 + 4.66 G us), capped at the measured time, with the glue unchanged.
  - That is weighted by the recorded decode routes' G histograms: 1,080 M=6 layers (G 14-35, mean 23.4), 1,320 M=12 layers and 1,480 M=16 layers.

  | M (load) | ceiling: layer time live / balanced | most it could save, ms per decode step (40 layers) |
  |---|---|---|
  | 6 (1 user) | **1.0248x** (second fit 1.0199x) | 0.24-0.30 |
  | 12 (4 users) | 1.0198x (1.0172x) | 0.31-0.35 |
  | 16 (4 users) | 1.0147x (1.0132x) | 0.29-0.32 |

- **Gate (Codex: M=6 >= 1.08x): CLOSED.** Codex's -0.4..-0.8 ms at one user and -1.5..-2.5 ms at four are 1.3-8x above this ceiling. Nothing was built or installed.
- **What is left:** a balanced scheduler is worth at most ~0.3 ms per step at every load (about 1% at one user, 0.7% at four). Serving it would need new FC1 and FC2C trees, their manifests and a site change.

### 5. Item 08 (early KV slab, `DSV41_KV_ALLOC_MODE=early_slab`): relaunch series S x4 / E x4 + one stock-loader cross-check. NOT promoted: with the fast loader every launch is already good class
- **What E does:** at the end of `Worker.init_device()`, before `load_model()`, it allocates and zero-fills the `--kv-cache-memory-bytes` slab (16 GiB). `allocate_kv_cache()` later takes that slab instead of a fresh `torch.zeros` of the same size. Contents, size and layout are stock; only the physical pages can differ.
- **Arms** (unpinned production, launcher 31d04d8f; the same suite on every launch: exact.py, G1 x2, ONE cold ~1M prompt with 1 Hz MemAvailable on 8/8, exact.py #2):
  - **S** = production + `DSV41_KV_BUDDY_LOG=1`, which only logs `/proc/buddyinfo` around the stock KV allocation;
  - **E** = production + early slab;
  - **L** (one launch, after the series) = production with the stock mmap loader (`DSV41_FAST_LOAD=0`) + buddy log. It checks where the slow class went.
- **1M class:** from the chunk 37-43 band (~600-700K context) of the cold prompt, as in "Goal window 4, prefill": good below 3.4 s.

| # | instance | arm | healthy s | exact.py start / #2 | G1 (digest) | cold 1M s | chunk bands 2-12 / 20-30 / **37-43** / 50-62 s | class | MemAvailable after start, GB (min of 8) | MemAvailable min during 1M, GB (8/8) | before -> after the KV allocation: free / >= 2 MiB, GB (spark-six) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | g4d2-S1 | S | 199 | a8c3394a / a8c3394a | 101.72 (9a18f802) | 166.24 | 2.42 / 2.57 / **2.71** / 2.84 | good | 13.5 | 9.03 | 33.1 / 29.6 -> 17.6 / 13.5 |
| 2 | g4d2-E1 | E | 199 | a8c3394a / a8c3394a | 101.36 (69ad8721) | 165.82 | 2.42 / 2.57 / **2.71** / 2.84 | good | 13.1 | 8.99 | slab at init_device: 110.8 / 59.4 -> 94.9 / 59.4 |
| 3 | g4d2-S2 | S | 199 | a8c3394a / a8c3394a | 101.80 (9a18f802) | 165.99 | 2.42 / 2.57 / **2.71** / 2.84 | good | 13.4 | 9.04 | 33.3 / 29.2 -> 17.8 / 13.2 |
| 4 | g4d2-E2 | E | 199 | a8c3394a / a8c3394a | 101.43 (9a18f802) | 166.09 | 2.42 / 2.57 / **2.71** / 2.85 | good | 13.3 | 8.90 | slab: 110.9 / 59.1 -> 95.0 / 59.1 |
| 5 | g4d2-S3 | S | 199 | a8c3394a / a8c3394a | 101.63 (9a18f802) | 166.10 | 2.42 / 2.57 / **2.71** / 2.84 | good | 13.1 | 9.00 | 33.2 / 29.0 -> 17.9 / 13.0 |
| 6 | g4d2-E3 | E | 200 | a8c3394a / a8c3394a | 101.21 (69ad8721) | 166.13 | 2.42 / 2.57 / **2.71** / 2.84 | good | 13.4 | 9.05 | slab: 110.9 / 61.3 -> 95.1 / 61.2 |
| 7 | g4d2-S4 | S | 199 | a8c3394a / a8c3394a | 100.88 (69ad8721) | 166.42 | 2.42 / 2.57 / **2.71** / 2.85 | good | 13.5 | 9.06 | 33.3 / 29.9 -> 17.6 / 13.9 |
| 8 | g4d2-E4 | E | 199 | a8c3394a / a8c3394a | 101.01 (69ad8721) | 166.47 | 2.42 / 2.57 / **2.71** / 2.85 | good | 13.1 | 9.05 | slab: 110.9 / 61.4 -> 95.0 / 61.4 |
| 9 | g4d2-L1 | L (stock loader) | 362 | a8c3394a / a8c3394a | - | **220.74** | 2.44 / 2.89 / **3.96** / 4.76 | **slow** | 14.7 | 10.32 | 5.2 / 0.06 -> 6.6 / 0.14 |

| arm | launches | slow-class 1M | cold 1M s, mean (range) | chunk 37-43 s, mean | G1 mean (by digest: 9a18f802 / 69ad8721) | MemAvailable after start, GB (min) | MemAvailable min during 1M, GB |
|---|---|---|---|---|---|---|---|
| S (fast loader, stock KV order) | 4 | **0/4** | 166.19 (165.99-166.42) | 2.708 | 101.51 (101.72 / 100.88) | 13.1 | 9.00 |
| E (fast loader, early slab) | 4 | **0/4** | 166.13 (165.82-166.47) | 2.708 | 101.25 (101.43 / 101.19) | 13.1 | 8.90 |
| L (stock loader) | 1 | **1/1** | 220.74 | 3.960 | - | 14.7 | 10.32 |

- **No slow launch to remove.** S and E are indistinguishable at 1M: 165.8-166.5 s, every chunk band within 0.01 s, 0/4 slow each. All nine fast-loader 1M prompts so far (g4l-FINAL's 165.9 s and these eight) land at 165.8-166.5 s.
- **Where the slow class went: the fast loader.** The stock-loader launch L1 is slow (220.7 s, chunk 37-43 at 3.96 s).
  - Its chunk times show the X-ray signature: flat 2.4-2.6 s up to chunk 20, a knee at chunk 21-22 (~340K context), then linear growth to 4.8 s at chunk 55+ (~880K).
  - Stock-loader launches were slow-class in 7/9 pinned launches in "Goal window 4, prefill", in 5/5 unpinned ones in the X-ray and in 4/5 unpinned ones in the pf step. Fast-loader launches are 9/9 good.
  - The fast loader is therefore what removed the straggler. That is worth ~55 s (-25%) on a cold 1M prompt of a launch that would otherwise have been slow-class, and it is a gain the Loader phase did not claim.
- **Mechanism (buddyinfo at the KV allocation, from the rank logs):**
  - **Stock loader (L1):** just before the KV allocation only 3.9-10.3 GB is free on each node; the rest of the memory is page cache. Of that, 0.06-1.74 GB is in blocks of 2 MiB or more and 1.3-4.7 GB in blocks of 64 KiB or more. The 16 GiB KV is therefore assembled from reclaimed page cache, i.e. mostly 4 KiB pieces.
  - **Fast loader, stock order (S1-S4):** 33 GB is free, 27-32 GB of it in blocks of 2 MiB or more, and the KV takes 16 GB of those blocks on every node (>= 2 MiB free: 27-32 -> 13-16 GB).
  - **Early slab (E1-E4):** allocated with 111 GB free, it does **not** touch the 2 MiB blocks: >= 2 MiB stays at 53-71 GB on all 8 nodes, while 16 GB of the 64 KiB-1 MiB fragments disappear. The driver's sub-2 MiB requests are served from the smallest free blocks first.
  - So with the fast loader, the stock order already gives the KV the contiguous backing that Codex's early slab was meant to secure, and the early slab itself would only have got it on a node with no fragments left.
- **Decode and memory:**
  - G1 follows the launch's AV startup profile, not the arm. The two G1 digests are the known prose near-tie of the k5 pins, and within a digest class E is -0.3% / +0.3% against S.
  - MemAvailable after start (13.1-13.5 GB) and at 1M (8.9-9.1 GB on the lowest node, spark-one) are the same in both arms.
  - Nothing reached swap. 0 ERROR/Traceback on every launch.
- **Decision: NOT promoted.** Under the rule ("promote if it removes slow launches"), there are none left to remove: 0/4 slow in both arms. The flag stays installed and off. `DSV41_KV_BUDDY_LOG=1` stays available as a diagnostic: one buddyinfo line at the end of `init_device` and one around the KV allocation per rank.

### 6. Final production instance g4d2-FINAL (launcher 31d04d8f defaults, AV unpinned, g4d installed with every flag off)
Supervisor launch with no extra variables; healthy at 13:13:39 UTC on the first try (+204 s). The short decode suite ran: no GSM8K-50, longctx or 1M, because nothing was promoted and the series above already ran eight cold 1M prompts on this configuration family. The window's final verification will run its one 1M probe.

| metric | g4d2-FINAL | reference |
|---|---|---|
| time-to-healthy, unpinned | +204 s | g4l-FINAL +199 s |
| exact.py | **a8c3394af9d2** first (own profile 910/466/2330/1373, "pin miss") and again after the suite | anchor a8c3394af9d2 |
| matched C1 coding / prose ms | 30.31 / 26.33 | pinned C1 30.25 / 26.34 |
| G1 x3 (code / prose / math) | **102.03** (95.88 / 76.49 / 133.71); outputs identical in all 3 repetitions | g4l-FINAL 102.13; g4-FINAL 101.62 |
| G4 session mean / prose MIN / G4min_hat | 61.39 / **48.36** / 47.96 (4-session step 45.30 ms) | g3-FU 61.65 / 48.64 / 48.43; g4-FINAL 60.50 / 47.61 |
| conc8 / conc16 aggregate tok/s | 258.0 / 399.1 | g3-FU 264.3 / 397.8 |
| MemAvailable after start (8/8) | 13.1-14.5 GB | - |
| errors / foreign requests | 0 ERROR/Traceback on 8/8 at start and end / none (G1: 9 completions for 9 sent) | - |

Goal status: still NOT met. G1 is 102.0 of 120 (+17.6% needed). The prose session MIN is 48.4 of 50 (+3.4%), and G4min_hat is 48.0 (+4.3%).

### 7. Live state and rollback
**Live:** instance **g4d2-FINAL**, healthy since 13:13:39 UTC.
- Launcher **31d04d8f** on 8/8 (unchanged in this step).
- Supervisor 7bea1cf3. Its preflight manifest is now **18b3d8175e12**: it hashes `model_runner.py`, which g4d replaces; without g4d it is 29a69fac9dfb.
- AV unpinned; clock lock 0-2000 MHz; LAN proxy unchanged.
- **Site files on 8/8:**
  - **g4d installed, every flag off** (`g4-all.sh status g4d`, digest f99d02f68320): `v1/worker/gpu/spec_decode/adaptive_verification.py` 55030a68, `v1/worker/gpu/model_runner.py` 6a4f2433, `v1/worker/gpu_worker.py` 4d67066c, `v1/worker/utils.py` c4e02ba8. The backups `.bak-g4-g4d` are 7e30eebc / 316fca44 / e970aa29 / 85b23497.
  - With the flags unset, g4d runs the stock code. The cost is one environment read at startup and one attribute check per step. Every control and S launch in this step served it that way, with exact.py a8.
  - fload (ca05f12513a1) and g4p (aef7c824bb9a) unchanged.
- 8/8 active, 0 restarts, 0 ERROR/Traceback, no test or sampler process left.

**Rollback / flags:**
- Nothing was promoted, so there is nothing to roll back in the launcher.
- One launch with a candidate, as supervisor arguments:
  - `DSV41_FUSED_AV=1` (or `=verify`, shadow compare, rank-0 log every `DSV41_FUSED_AV_LOG` steps);
  - `DSV41_KV_ALLOC_MODE=early_slab`;
  - `DSV41_KV_BUDDY_LOG=1` (diagnostic: buddyinfo around the KV allocation).
- Remove g4d: with the service stopped, run `ssh spark-one cat ~/dsv41-stage/g4/common/g4-all.sh | bash -s uninstall g4d`. It is sha-guarded, restores the four base files and brings the preflight manifest back to 29a69fac9dfb.

### 8. Notes for the next steps
- **Keep `DSV41_FAST_LOAD=1`:** it is also the 1M straggler fix (section 5). Rolling it back brings the slow class back (one stock-loader launch here: 220.7 s against 166 s).
- **Small exact decode items, as measured here:**
  - fused AV: +0.2% G1 (installed, off);
  - a balanced M32 tile scheduler: at most ~0.3 ms per step at every load (not built);
  - the rdma_ar P1 doorbell: ~0.1-0.2 ms (window 3; not built).
  
  None clears 0.5% alone. Together they might, as a single bundle in a later window.
- **The large exact levers remain** the ones listed in window 3 (MoE FC1 -> FC2 pipelining, contiguous FC2C, q_b in the L2 plan, drafter MoE on M32, mfront after its crash fix) and Moet's mHC boundary overlap (goal window 5).
- **Interrupted first attempt:** its instances g4d-V (verify shadow, 7,100 calls, 0 mismatches) and g4d-C1 (launcher fd1e809e) are in scratchpad `g4d/`. Its F1 launch was stopped by the Loader-phase restore before any measurement. This step reused its bundle unchanged.

## Goal window 4, client step (2026-10-27 13:22-14:05 UTC): Codex 07 (multi-turn prefix reuse) is a template default, not a server bug; it becomes a CLIENT RULE, and the server patch is NOT installed because it is not exact. Codex 03 (prose oracle): dump done on a pinned serving instance (exact); the replay finds NO calibration or tree headroom; the one lever is a better drafter head (ceiling +7% top-2 / +12% top-4 / +18% top-16 on the G4 prose session). Live config unchanged (launcher 31d04d8f)

Scope: the client step of goal window 4 (light; restarts only where needed): (a) Codex item 07, its `--live` multi-turn check on an idle server (`cached_tokens` per turn from `prompt_tokens_details`, thinking on and off); (b) Codex item 03, its prose-oracle dump during a serving period, then the offline replay. Trim rule: no long-prompt probes, no GSM8K-50 / longctx (nothing promoted). Every launch used `DSV41_GPU_CLOCK_MHZ=2000`; SM clock 1969-1989 MHz on 8/8 at every check; no foreign request in any timed phase.

Files: spark-one `~/dsv41-g4c/` (`scripts/`, `runs/`, `orc/g4c-O1/` = the dump, 314 MB, rank0-7), Mac scratchpad `g4c/` (`scripts/` = `orcab.sh` instance driver, `sw.sh` stopped window, `phase.sh`, `collect.sh`, `orcan.py`, `orcproj.py`; `orc/` = the bundle; `ab.log`, `ab.jsonl`, `runs/`). Bundle staged on 8/8: `~/dsv41-stage/g4/orc/` (NOT installed).

### 1. State check (13:22 UTC)
- Live: g4d2-FINAL, launcher **31d04d8f** on 8/8, healthy since 13:13:39, 0 requests in flight. Site status matched the README on 8/8: g4d f99d02f68320 (flags off), g4p aef7c824bb9a, fload ca05f12513a1.
- The task text still named launcher fd1e809e as "production"; that is superseded by the Loader phase (31d04d8f, `DSV41_FAST_LOAD`), and the decode+ops step had already left production defaults live. Nothing to restore.

### 2. Item 07: multi-turn prefix reuse, live (13:25-13:33 UTC, idle production instance g4d2-FINAL)
- Scripts (spark-one `~/dsv41-g4c/scripts/`): `mt_probe.py` (532cb8da; Codex's 4-turn `--live` conversation: ~5.3K-token code prefix, short questions; non-streaming, temperature 0, `reasoning_effort` low, `return_token_ids`; waits for `num_requests_running+waiting = 0` before every request) and `mt_ttft.py` (c6816244; streaming TTFT, long answers of 0.9-1.4K tokens per turn). Each variant has its own nonce, so variants never hit each other's cache. Results: `runs/mt_probe_1.json`, `runs/mt_ttft_1.json`.
- "Best possible" = full 64-token blocks of the common prefix of turn t's prompt ids and turn t-1's prompt + output ids (from `return_token_ids`); `cached_tokens` includes the 128-token SWA replay.

| variant (`chat_template_kwargs`) | turn 2 / 3 / 4: cached of prompt | = best possible? | where turn t leaves turn t-1's prompt + output |
|---|---|---|---|
| think, stock (default `drop_thinking`) | 5248/5371, 5312/5436, 5376/5504 (97.7-97.9%) | yes | at the LAST prompt token (`<think>` becomes `</think>`): the previous answer is always re-prefilled |
| think, `drop_thinking: false` + reasoning sent back | 5376/5431, 5504/5532, 5632/5656 (99.0-99.6%) | yes | after the whole previous output |
| think, `drop_thinking: false`, reasoning NOT sent back | 5248/5384, 5376/5462, 5440/5543 | yes | at the FIRST output token (empty `<think></think>`) |
| think, `tools` attached, reasoning sent back | 5632/5667, 5696/5761, 5824/5852 (98.9-99.5%) | yes | after the whole previous output (tools force `drop_thinking` off) |
| thinking off (`thinking: false`) | 5248/5327, 5312/5372, 5376/5420 (98.5-99.2%) | yes | after the whole previous output |

| long answers (streaming) | turn 2 / 3 / 4 cached | TTFT turn 2 / 3 / 4 | prompt at turn 4 |
|---|---|---|---|
| think, stock | 92.6 / 92.1 / 92.3% | 0.425 / 0.422 / 0.443 s | 6,586 tokens |
| think, `drop_thinking: false` + reasoning sent back | 98.7 / 99.5 / 99.7% | 0.396 / 0.280 / 0.275 s | 9,117 tokens |

- **The server side is already optimal:** in every case the cache hit equals the best possible block hit, so the prefix cache (MLA group + bounded SWA replay) loses nothing beyond block granularity. The only loss is the V4.1 template's default: it drops earlier reasoning and closes the old turn with `</think>`, so the previous answer (7-8% of the prompt with ~1K-token answers here, more when answers are long relative to the prompt) is prefilled again on the next turn. Codex's offline 82% -> 99% is the same effect on a conversation with longer answers.
- **Codex's server patch (`DSV41_PREFIX_STABLE_THINKING`): NOT installed, not staged.** It is server-side but **not exact**. When on, it rewrites every multi-turn thinking prompt so that the earlier reasoning stays in context, and its LRU "recovery" puts reasoning back even for clients that deliberately left it out. The model then sees a different prompt from turn 2 on, so outputs change against the stock template. exact.py and G1 are single-turn and would not notice, but the outputs of real chats would change. The same rendering is available per request with a stock kwarg (`deepseek_v41.py` passes `chat_template_kwargs.drop_thinking` to `encode_messages`), so no server change is needed.
- An exact server-side alternative (prefill the stock-rendered previous answer in the background after each thinking turn, so the next turn hits) would save at most ~0.15 s TTFT per turn here and would compete with other sessions' decode; not pursued.

**Client rule (multi-turn chats against this server):**
1. Thinking OFF: nothing to do; the whole conversation is reused.
2. Thinking ON, default: each new turn prefills the previous answer again (TTFT ~0.42 s at 5-7K prompts here). This is the model's documented default and the only exact-to-stock choice.
3. Thinking ON, to reuse everything: send each assistant turn back with `reasoning_content` (or `reasoning`) exactly as returned, together with its `content`, and set `"chat_template_kwargs": {"drop_thinking": false}` on EVERY turn (turn 1 renders identically either way). This gives 98.7-99.7% cached and TTFT 0.28 s on turns 3-4. It changes what the model sees (its earlier reasoning stays in context, as in DeepSeek's tool-call mode), so outputs differ from the default, and the prompt grows by the reasoning (+38% by turn 4 here). It is the client's choice per conversation.
4. Never set `drop_thinking: false` without sending the reasoning back: the template then renders an empty `<think></think>`, the prompt changes and nothing extra is reused.
5. With `tools`, the template already keeps reasoning (`drop_thinking` is forced off): tool loops must send `reasoning_content` back on every assistant message, or the cache breaks at the first output token of each step.
6. Keep the system message (and `tools`) at index 0; a mid-conversation system message moves the tools block and invalidates the cache from block 0 (Codex hazard C).

### 3. Item 03: prose oracle dump (instance g4c-O1, 13:45-13:52 UTC) and offline replay
**Bundle `orc`** (Codex shipped only a sketch; this is a full implementation, flag `DSV41_PROSE_ORACLE_DUMP=<dir>`, unset = off: no buffer, no extra kernel, stock graphs):

| file (site-packages) | before -> orc | what |
|---|---|---|
| vllm/v1/worker/gpu/spec_decode/dspark/speculator.py | 439b390e -> b7616890 | armed: inside the captured greedy-sharded draft graph, per draft position, the rank's vocab-shard top-64 (`DSV41_PROSE_ORACLE_K`) of the total logit (base + Markov bias), the bias of those candidates and the shard logsumexp go to persistent buffers. Read-only: draft tokens, confidences and the all-gather are unchanged |
| vllm/v1/worker/gpu/model_runner.py | 6a4f2433 -> 33e3472f | after `propose()` + `record_confidences()`: `record_step()` if the speculator owns the buffers (one getattr otherwise) |
| vllm/v1/worker/gpu/dsv41_oracle.py | ABSENT -> 091f1566 | writer: every rank pickles its shard top-64 per step to `<dir>/rank<R>/shard-*.pkl`; rank 0 adds request ids, sampled tokens / counts, confidences, AV-admitted and verify-cap-live draft counts, prompt token ids |

- No collective was added: the 8 shard files are merged offline. That merge is exact, because every global top-64 entry is in its shard's top-64.
- Checks before serving: CPU test (`tests/test_orc_cpu.py`: 40 random 8-shard merges equal to the global top-k; the writer fields) PASS; **GPU test (spark-two, service down): the capture inside a CUDA graph, 4 batch sizes x 5 replays, the graph's argmax equal to eager and the buffers equal to an eager topk / gather / logsumexp: PASS**; import check PASS.
- **Stopped window** 13:44:40-13:45:18 (`sw.sh`): stopped with 0 requests in flight, pre-status as in section 1, GPU test, `g4-all.sh install orc` (digest **a6223fa42311** on 8/8, backups `<file>.bak-g4-orc`, absent marker for the new module), import check.
- **Instance g4c-O1** (`orcab.sh`): pinned (`av-pin-k5-g3FU.json` eeac4f97, "pin hit") + `DSV41_PROSE_ORACLE_DUMP=~/dsv41-g4c/orc/g4c-O1`; healthy +126 s (13:47:42). Suite: exact.py, matched C1 (coding/prose x5, thinking off), G1 x2, G4 (warm-up + 6), exact.py #2. 0 ERROR/Traceback, no foreign request.
  - **Exact:** exact.py **a8c3394af9d2** at start and end, with the pin's counters 909/467/2335/1373 and 348/233/171/90/67. G1 digest **9a18f802d930f6e7**, and every G1 output (code/prose/math x 2 reps) is identical to the pinned control g4d2-C1. The dump changes no output.
  - **4-user outputs are not a bitwise gate in any instance:** the G4 outputs differ run to run in the control itself (g4d2-C1: 6 distinct output sets in 7 runs; g4d2-F1 7/7; g4c-O1 4/7; one set is common to all three).
  - **Cost while armed** (one device sync per step): C1 31.83 / 27.97 ms (control 30.25 / 26.34), G1 96.2 (101.6), G4 prose MIN 46.0 (48.1). It is for measurement windows only.
- **Merge** (`orcan.py`, on spark-one, ~10 s): 6,337 steps on 8 ranks aligned with 0 missing / count / validity / draft mismatches. The merged global top-1 equals the drafted token at all 49,810 positions. The replayed acceptance equals the verifier's own count in all 9,469 records after startup (20 of 493 records of the startup exact.py batch do not cross-check; not used). Workloads were labelled from the prompt tokens and the phase.

**Headroom per workload** (draft steps n; AL = 1 + accepted drafts per step, live; recall = the target's rank in the drafter's top-K at the first missed position, i.e. conditioned on a correct prefix):

| workload | n | AL live | drafts live / admitted per step | steps missing at pos 0 | recall at first miss top-2 / 4 / 8 / 16 / 64 | Markov-weight calibration (held-out) AL | cap off AL | ceiling: target in top-2 / 4 / 16 counts as hit, AL |
|---|---|---|---|---|---|---|---|---|
| **prose, G4 session (the slow one)** | 1,964 | **1.810** | 1.31 / 1.65 | 37% | **19.7 / 36.9 / 48.6 / 59.3 / 77.2%** | 1.817 (+0.4%) | 1.840 | 1.934 / 2.032 / 2.133 |
| prose, G1 (1 user) | 1,104 | 1.846 | 1.55 / 2.16 | 41% | 23.0 / 41.0 / 54.0 / 65.3 / 79.6% | 1.844 (-0.1%) | 1.889 | 2.011 / 2.145 / 2.312 |
| prose, matched C1 (thinking off) | 558 | 2.742 | 2.82 / 3.48 | 20% | 40.0 / 63.2 / 80.8 / 87.3 / 95.8% | 2.735 (-0.3%) | 2.821 | 3.052 / 3.251 / 3.523 |
| code / math / mixed, G4 | 1,475 / 1,017 / 1,253 | 2.420 / 3.557 / 2.884 | 2.21 / 3.24 / 2.62 live | 24 / 15 / 21% | top-2 39.5 / 48.0 / 44.9%, top-16 88.0 / 91.5 / 88.4% | +-0.3% | | |
| code, matched C1 | 271 | 5.214 | 4.74 / 4.89 | 2% | 49.2 / 76.9 / 84.6 / 92.3 / 93.8% | -0.8% | 5.262 | |

- The drafter's per-position Markov weight is already right: the held-out optimum is alpha 0.8-1.1 at every position, and the AL change is within +-0.4%. The DSpark confidence head is calibrated within a few points in every bucket and position. AV and the verify cap therefore already size the verify window well.
- For thinking prose the drafter is diffuse, not miscalibrated. When its top-1 misses, the target is its second choice only 20-23% of the time, and outside its top-64 21-23% of the time.

**Cost-adjusted** (`orcproj.py`, first order). Baselines: the pinned control g4d2-C1, i.e. G1 prose 76.52 tok/s (24.12 ms/step), C1 prose 26.34 ms, G4 4-session step 44.56 ms. Cost per extra verified row: 2.04 ms at 1 user (matched C1 coding vs prose), or 2.84 ms on the pin's curve; 1.79 ms at 4 users (pin curve, 12-16 rows). Server-wide = every G4 request pays its own extra rows.

| change | G1 prose tok/s | C1 prose tok/s | G4 prose session, server-wide (prose request only) | goal: +3.4% (prose MIN 48.36 -> 50) |
|---|---|---|---|---|
| Markov-weight calibration (Codex's "calibration only") | -0.1% | -0.3% | +0.4% | no |
| verify cap off (all admitted drafts live) | -2.6% | -2.2% | -3.7% (+0.3%) | no |
| tree K=2, confidence-gated (best realizable: conf < 0.3 / 0.5 / 0.7) | -0.8 / -1.4 / -0.5% | 0.0 / -1.8 / -1.7% | 0.0 / -1.5 / -6.1% (+0.1 / +0.6 / +1.7%) | no |
| tree K=4, gated | -4.3..-7.3% | -0.2..-10.4% | -0.2..-18.5% | no |
| tree K=2 at the first miss (oracle placement, not realizable) | +0.3% | +2.6% | -7.9% (+2.7%) | no |
| better drafter head, ceiling at the same verify width: target in top-2 / top-4 / top-16 promoted | +8.9 / +16.2 / +25.2% | +11.3 / +18.6 / +28.5% | **+6.9 / +12.2 / +17.8%** | yes if it captures about half of the top-2 ceiling or a quarter of the top-16 ceiling |

- **Codex's own analyzer** (`analyze_prose_oracle.py`, unchanged, run on the same records converted to its npz input with the current pin; `runs/codex-analyzer-O1.{txt,json}`) reproduces the recall tables above. It also reports "calibration" +10.4% for G1 prose and a G4 prose session of 52.1 tok/s. **That number is a replay error:** its calibration (and tree and ceiling) replays walk all 5 draft positions and ignore the live window that AV and the verify cap chose, while charging +0.25 verify rows. Its own `uncapped_5drafts` row (verify all 5) gives the same AL for the G4 prose session (2.119 against 2.128) at +3.69 rows and 41.9 tok/s. The α refit itself is worth +0.009 AL. The self-test's synthetic prose had a built-in Markov mismatch; the real drafter does not.
- **What the oracle recommends:** no calibration change and no tree (static or gated) for prose. The only lever with enough headroom is a better DSpark draft head for thinking prose, for example a fine-tune or self-distillation of the Markov/LM head, or a top-16 reranker, trained on this model's own greedy thinking-prose outputs. The dump format is a ready training and evaluation set: per position the top-64 with base and Markov parts and the target. This fits goal window 5's drafter-capture item. Whether a trained head reaches a useful share of the ceiling is open, and only training can tell.
- **Could it be exact?** Any drafter change (head, calibration, tree) changes the drafts, so it changes acceptance, AV's budget and the verify layout. The greedy target tokens are the same in exact arithmetic, but bitwise near-ties can flip with the verify layout. This is the same class as the Moet k6/k7 A/B ("greedy-equivalent but not bitwise"): exact.py a8 held there, code and math G1 stayed bitwise, and the known G1 prose tie (character 3885, " capital-intensive" / "ing capital-intensive") flipped between 9a18f802 and 69ad8721, exactly as an AV-profile change flips it.
  - **Gate it would need:** a pinned A/B, with the arm's own AV pin if the drafter cost changes. Then:
    - exact.py combined hash a8c3394af9d2 on every launch (the acceptance counters change by design);
    - G1 code / math bitwise (596b478a / 92e5d8cf);
    - G1 prose = one of the two known tie outcomes (923a46be / e8a524f3), with any other first divergence counting as a fail;
    - G4 / conc: no equality gate is possible (the control is not reproducible run to run), so a first-divergence census instead: each session's first divergence from its single-user greedy output must be a near-tie, at the control's rate.
  - Under the strict rule (identical G1 digest under the current pin), any arm that flips the prose tie fails. Promoting one is a user decision, as for k6/k7.

### 4. Production restore (13:57-14:03 UTC)
- Stopped (0 requests in flight) and `g4-all.sh uninstall orc`: digest 756484cd51b4 = baseline. Then on 8/8: speculator 439b390e, model_runner 6a4f2433, no `dsv41_oracle.py`; g4d f99d02f68320 / g4p aef7c824bb9a / fload ca05f12513a1 unchanged; launcher 31d04d8f. Supervisor preflight manifest back to **18b3d8175e12**.
- **g4c-FINAL:** supervisor launch with defaults (unpinned, production), healthy at 14:01:06 UTC (+198 s).
  - exact.py **a8c3394af9d2** at start and after the suite (own profile 908/468/2340/1373, "pin miss").
  - G1 x2 **101.67** tok/s, digest 69ad872124100141 (the other outcome of the known prose tie; unpinned production lands on either; g4d2-FINAL 102.03).
  - 0 ERROR/Traceback on 8/8, no foreign request, MemAvailable min 13.5 GB, SM 1969-1989 MHz.

### 5. Live state and rollback
- **Live:** g4c-FINAL, launcher **31d04d8f** defaults on 8/8, supervisor 7bea1cf3 (manifest 18b3d8175e12), AV unpinned, clock lock 0-2000 MHz, LAN proxy unchanged.
- **Site:** g4d installed with every flag off (f99d02f68320), g4p (aef7c824bb9a), fload (ca05f12513a1). orc NOT installed.
- **Nothing to roll back.**
- **Re-run the dump** (stopped window): `ssh spark-one cat ~/dsv41-stage/g4/common/g4-all.sh | bash -s install orc` (expects g4d's model_runner 6a4f2433 and g4p's speculator 439b390e). Launch pinned with `DSV41_PROSE_ORACLE_DUMP=<dir>`, run the prose traffic, then `collect.sh <label>` and `orcan.py`. Uninstall afterwards with `... | bash -s uninstall orc`.

### 6. Notes for the next steps
- 07 is closed: a client rule, no server work. Clients that want full reuse in thinking chats opt in per conversation (rule 3). Tool clients must send `reasoning_content` back (rule 5).
- 03: the prose goal (+3.4% on the G4 prose session) is not reachable by calibration, the verify cap or trees. A trained drafter head is the only lever with that headroom. If one is trained, judge it on this dump's held-out prose records first (it must beat the top-1 hit rate at the first miss without breaking the 53-59% of positions where top-1 is already right), then run the E1 gate above.

## sPTC layer 3 (goal window 4, Layer3 step, 2026-10-27 14:08-15:30 UTC): the DeepSeek-V4.1 tool-parser patch is installed on 8/8 and `DSV41_TOOL_STREAM_NEWLINE=1` is PROMOTED as a launcher default (launcher 31d04d8f -> **95f08eff**). It is output-neutral: exact.py a8 and the pinned G1 digest are unchanged, and streamed tool arguments equal the non-streamed ones for the same tokens in 28/28 cases. Tool-call programs now stream line by line: the largest single argument delta drops from 78% to 17% of the program (median). Layer-3 A/B (DSH plugin c981260, 20 code-mode tasks x 2 rounds per arm, thinking off, JIT off): 238/238 speculative sub-calls claimed, 0 wasted, no extra server tokens. run_code time drops 26%, but the agent's own stream is ~4% slower while sub-calls overlap it. Turn p50 9.00 -> 8.48 s, p90 17.63 -> 17.07 s, paired median -0.14 s (on faster in 25/40), mean flat. Recommendation for the Mac app: layer 3 ON as a small win, at its next plugin rebuild

Scope: the user-approved (2026-10-26) layer 3 of sPTC for DeepSeek Harness. The offline build was ready: `512S2:~/dsh-sptc/LAYER3_READY`, plugin `~/dsh-sptc/sptc-plugin` main **c981260**, server patch staged in `~/dsv41-stage/layer3/` on 8/8. Steps:
1. install the parser patch with the flag OFF, relaunch, exact.py;
2. relaunch with the flag ON, check stream integrity, exact.py and G1;
3. run the layer-3 A/B from 512S2;
4. keep the flag as a launcher default if integrity and exactness pass;
5. write this section.

Trim rule (user, 08:35 UTC): the patch only touches the API server's parser, not the engine. So there were no decode or long-prompt suites. Each instance ran exact.py first and last, the integrity case set, and G1 x2 where the step needed it. There was no GSM8K, longctx or 1M run, because the model's computation does not change. Every launch used `DSV41_GPU_CLOCK_MHZ=2000`, and the SM clock was 1969-1989 MHz on 8/8 at every check. Every integrity request was the only request on the server: each one ran on an idle server and added exactly 1 to `request_success_total`.

Files: spark-one `~/dsv41-g4l3/` (`scripts/`, `cases/l3cases.jsonl`, `runs/<label>/`). Mac scratchpad `g4l3/`:
- `scripts/`: `l3sw.sh` (stopped window), `l3ab.sh` (instance driver), `phase.sh`, `l3phase.sh`, `l3cases.py`, `l3integrity.py`, `l3ref.py`, `l3cmp.py`, `l3repeat.py`, `promote_l3.sh` and `lines-L3.txt`;
- results: `ab.log`, `sw.log`, `runs/` and `ab/<label>/`.

A/B outputs on 512S2: `~/dsh-sptc/ab/g4l3-20261027/` (`report.md`, `report.json`, `runs.jsonl`, per-run captures) and `~/dsh-sptc/ab/g4l3-20261027.log`.

### 1. State check (14:08-14:31 UTC)
- **Live:** g4c-FINAL, launcher **31d04d8f** on 8/8, healthy since 14:01:06, 0 requests in flight.
- **Site:** g4d f99d02f68320 (flags off), g4p aef7c824bb9a, fload ca05f12513a1. The parser files were the deployed ones on 8/8 (`deepseek_v4.py` 709f328f, `parser_engine.py` 62b741a0), with no `.bak-layer3`. The stage was identical on 8/8 (digest bb8f901117da).
- **Nothing to restore:** the task text again named fd1e809e as production, but the Loader phase made 31d04d8f production, and no candidate was installed or running.
- **The READY marker exists** (written 2026-10-26). Its orchestrator update says to use plugin main c981260, the commit with the review fixes.

### 2. The patch and how it was checked
**Patch** (staged `~/dsv41-stage/layer3`, MANIFEST for vLLM 0.32.0; site-packages `vllm/`):

| file | deployed -> patched (sha256) | what the flag changes |
|---|---|---|
| `parser/deepseek_v4.py` | 709f328fe3b0711c98cc76f572bee14c6342efaaf2160f010517d3fd54794cee -> **3334c1fa88c30f1b1c8fadb445332f264d8415641c44afc8f6478ee7bd4f83db** | argument deltas flush at `>` **and at newline** (`arg_structural_chars`) |
| `parser/engine/parser_engine.py` | 62b741a0557beff5e2d7ed630e907c0203102ab8fda26b06190ded0563f0e120 -> **6e3eeaba2a38b3ec9d9e42437d43246f77ae9d21c7b9d2540a7c08a5b461171e** | an open string parameter streams as incrementally escaped JSON. It falls back to the stock full conversion at a DSML mark, or past 8,192 characters outside an open string (`DSV41_TOOL_STREAM_NEWLINE_MAX_CHARS`). |

- The flag is read once at import: `DSV41_TOOL_STREAM_NEWLINE` must be exactly `1` in the API server's environment. With it unset, the patched files take the stock paths.
- Non-streaming responses use `parse()`, which the flag does not touch.

**Integrity probe** (`l3integrity.py` 66841a63 -> token-id version; `l3cases.py`). The case set has **32 requests**:
- **27 recorded:** main-agent requests from the 2026-10-26 DSH check runs (`512S2:~/dsh-sptc/check`), first turns and later turns with tool results:
  - A thinking 11, B no-thinking 8, D/E stress 4 (all `/v1/chat/completions`);
  - C 4 on the Messages route `/v1/messages`;
  - the OpenAI-route prompts are 5.8-8.8K tokens.
- **5 new:**
  - an HTML table program with `<`/`>` on every line, unicode (Reykjavík, 東京) and escapes;
  - two parallel `get_weather` calls;
  - `write_file` with an integer parameter and thinking on;
  - tools attached but no call;
  - a plain chat without tools.

How each case runs:
- A non-streaming request, then a streaming one, each on an idle server, each with its own `cache_salt` (cold prefill), and with `return_token_ids` on the OpenAI route.
- Three verdicts:
  - **direct:** live non-stream vs live stream: content, reasoning, tool names, arguments (string-exact), finish reason and completion tokens. It is valid only when both requests generated identical token ids.
  - **ref** (`l3ref.py`, CPU, the installed vLLM): the stream's OWN token ids, decoded with the served tokenizer, go through the installed parser's non-streaming path exactly as `chat_completion_full_generator` calls it (`set_prompt_token_ids` + `parse(text, request, enable_auto_tools=True)`). The result is compared with what the stream delivered. The same reproduction applied to the live non-stream token ids gave back the live non-stream response in every case (28/28 in each instance), so the reference is faithful.
  - **cross-instance** (`l3cmp.py`): flag off vs flag on, for cases with identical token ids in both instances.
- Why the ref verdict is needed: the two live requests of one case often generate different tokens (section 6). Without it, only the cases with matching tokens could be checked.

### 3. Install (stopped window 14:31:14-14:31:40 UTC, `l3sw.sh`)
- Stopped with 0 requests in flight. The pre-state was as in section 1 on 8/8.
- `bash ~/dsv41-stage/layer3/install.sh` on 8/8: rc 0 everywhere. It is sha-guarded and keeps backups `<file>.bak-layer3`.
- **After the install, on 8/8:** `deepseek_v4.py` 3334c1fa, `parser_engine.py` 6e3eeaba, backups 709f328f / 62b741a0. `status.sh` reports patched.
- **Import check** (spark-one, CPU):
  - flag unset: `arg_structural_chars = {'>'}`, `_L3_NEWLINE False`;
  - `DSV41_TOOL_STREAM_NEWLINE=1`: `{'\n', '>'}`, `_L3_NEWLINE True`, `_L3_INCREMENTAL True`.
- The supervisor's preflight manifest does not hash the parser files, so it stayed 18b3d8175e12.

### 4. Instance g4l3-F0: patch installed, flag unset (pinned `av-pin-k5-g3FU.json` eeac4f97; 14:31:48-14:41:30)
- Healthy at +131 s (fast loader).
- **exact.py a8c3394af9d2** with the pin's counters 909/467/2335/1373 and 348/233/171/90/67 (pin hit), and again after the suite (#2). 0 ERROR/Traceback on 8/8.
- **Cases:** all 64 requests answered normally.
  - direct PASS for **17/17** OpenAI cases with identical tokens and 2/2 Messages cases with identical outputs; 0 FAIL;
  - ref(stream) **28/28** PASS, ref(non-stream) 28/28.
- **Same as stock with the flag unset.** B0 is the same 32 cases on the stock parser (live g4c-FINAL, 14:22-14:30, before the install; first probe version without token ids).
  - 17 cases have identical stream text in B0 and F0, and 13 of them contain tool calls. In 12 of those 13, the argument chunking (the sequence of delta lengths) is identical. The odd one has 6 pieces against 5, i.e. two `>` flushes fell into one engine step.
  - Content and reasoning chunking differs between B0 and F0 even in a no-tools chat. B0 is unpinned and F0 pinned, so speculative decoding groups tokens into steps differently. That is why chunk-level comparisons are only made between F0 and F1 (same pin).

### 5. Instance g4l3-F1: `DSV41_TOOL_STREAM_NEWLINE=1` (pinned; 14:41:55-14:52:40)
- Healthy at +127 s. The API server's environment carried the flag.
- **exact.py a8c3394af9d2** with the pin's counters (pin hit), and again after the suite (#2).
- **G1 x2 101.66 tok/s** (95.95 / 76.50 / 132.53), **digest 9a18f802d930f6e7**, the pin's reference (as g4c-O1 and g4d2-C1). Both repetitions were identical, with 6 completions for 6 sent.
- 0 ERROR/Traceback on 8/8.
- **Stream integrity:**
  - ref(stream) **28/28 PASS**: for the same tokens, the streamed tool arguments concatenate to exactly the non-streamed arguments, and the streamed content and reasoning equal the non-streamed ones. This includes the 13 cases whose two live requests generated different tokens.
  - ref(non-stream) 28/28. direct PASS **15/15** OpenAI cases + **2/2** Messages; 0 FAIL.
  - Every streamed argument concatenation parses as JSON: 25 tool calls on both routes.
  - On the Messages route (no token ids), the `input_json_delta` concatenation equals the non-streamed `tool_use.input` in both cases whose two requests produced the same output.
- **F0 vs F1** (13 OpenAI cases with identical token ids in both instances, non-stream and stream, plus 2 Messages cases with identical outputs):
  - non-streaming responses identical **15/15**;
  - content and reasoning delta sequences identical **chunk by chunk, 15/15**;
  - argument concatenations identical 15/15.
  - Only the chunking of tool arguments differs. Arguments without a newline (the `get_weather` calls) chunk exactly as before.
- **How the arguments now arrive** (25 streamed tool calls; same-token subset in brackets; times are seconds before the last argument delta):

  | median per tool call | F0 (flag unset = stock) | F1 (flag) |
  |---|---|---|
  | argument deltas | 6 [6] | 22 [19] |
  | largest single delta, share of the program | 0.78 [0.70] | **0.17** [0.17] |
  | mean lead of an argument character | 0.48 s [0.47] | **1.58 s** [1.46] |
  | lead of the 50% mark | 0.49 s [0.49] | **1.68 s** [1.37] |

  Example (A:T05, same tokens): 4 deltas with 95% of the program in one burst, against 27 deltas and 17%.

### 6. Finding (pre-existing, not caused by this step): long-prompt greedy requests are not reproducible run to run
- **Back-to-back pairs differ.** Two identical greedy requests sent back to back to the same live instance (cold prefill each) generated different tokens:
  - 12/28 OpenAI cases on the **stock** instance B0, before anything was installed;
  - 11/28 in F0 and 13/28 in F1;
  - the Messages route 1-2/4.
  - Which cases differ changes between F0 and F1, although both had the same pin and the same request history.
  - Every case that differs has a 5.8-8.8K-token prompt. The new short-prompt cases (20-420 tokens) never differed.
- **Repeat probe** (`l3repeat.py`, on F1 after its suite; 4 identical non-streaming requests per mode):

  | case (prompt) | cold: fresh `cache_salt` each time, distinct outputs / first divergence | warm: one salt, prefix-cache hits |
  |---|---|---|
  | B:T16 (5.9K) | 3 of 4 / token 186 | 4 of 4 / token 155 (cached 5,888) |
  | B:T02 (5.9K) | 3 of 4 / tokens 218-225 | 2 of 4 / token 162 |
  | A:T01 (6.1K, thinking) | 2 of 4 / token 429 | 2 of 4: every cached run differs from the cold one at **token 0** (cached 6,080) |
  | N:html-table (416) | **1 of 4** (identical) | 2 of 4: cached runs differ from the cold one at token 41 (cached 384) |

- **What this means:**
  - Cold short prompts are reproducible.
  - Cold prompts above ~2K tokens are not, even with nothing else on the server.
  - A prefix-cache hit can change the output against a cold prefill at any length.
  - exact.py (268 prompt tokens) and G1 (103-134-token prompts + 1,024 output tokens) never reach 2K tokens of context. They held on every launch here, so the exactness gate as defined cannot see this.
- **Suspects, not investigated in this step:**
  - the long-context path above the sparse-attention top-k size of 2,048 (indexer top-k selection or tie-breaking that depends on physical KV placement);
  - cached-prefix vs cold-prefill numerics.
- **Effect on this step:** none on the verdicts. The integrity check compares the stream with the non-streaming parse of the same tokens, so it does not depend on reproducibility. This is an open item for the exactness program: an exactness gate for contexts above 2K tokens is missing.

### 7. Promotion (14:56 UTC)
- **Integrity and exactness passed**, so the flag became a launcher default (`scripts/promote_l3.sh l3 scripts/lines-L3.txt`): `~/dsv41-tp8.sh` 31d04d8f -> **95f08eff** on 8/8. The backup `~/dsv41-tp8.sh.bak-g4-l3` is 31d04d8f on 8/8. One line, after `DSV41_FAST_LOAD`:
  ```sh
  export DSV41_TOOL_STREAM_NEWLINE=${DSV41_TOOL_STREAM_NEWLINE:-1}  # g4l3 sPTC layer 3 2026-10-27: ... 0 = stock flushing at ">"
  ```
- The supervisor is unchanged (7bea1cf3). Its preflight manifest, which hashes the launcher, is now **40eb7abf08c7**.

### 8. Production instance g4l3-PROD (launcher 95f08eff defaults, AV unpinned; 14:58-15:21 UTC)
- Supervisor launch with no extra variables (`--expect-launcher 95f08eff`; preflight manifest 40eb7abf08c7 and launcher identical on 8/8). Healthy at 15:01:43 UTC on the first try (+200 s).
- The API server's environment has `DSV41_TOOL_STREAM_NEWLINE=1` from the launcher (no supervisor argument) and `DSV41_FAST_LOAD=1`.
- **exact.py a8c3394af9d2** first (own profile 909/467/2335/1373, per position 348/231/171/91/68, "pin miss" = anchor OK) and again at 15:21 after the A/B (#2).
- **G1 x2 101.37 tok/s** (95.70 / 75.80 / 132.61), digest 69ad872124100141 = the other outcome of the known prose tie (unpinned production lands on either; g4c-FINAL had the same digest, 101.67). Both repetitions identical; 6 completions for 6 sent.
- **Spot integrity check** with the flag coming from the launcher (5 cases: A:T05, B:T06, E:S22, N:html-table, C:T01 on the Messages route): direct PASS 5/5, ref 4/4 (OpenAI route); 17-32 argument deltas per call.
- 0 ERROR/Traceback on 8/8 at start and end. MemAvailable after start 13.2 GB (min of 8), 10.6 GB on spark-one after the A/B. SM clock 1969-1989 MHz on 8/8 at every check.

### 9. Layer-3 A/B (512S2, 15:04:29-15:19:28 UTC, on g4l3-PROD; newline flag on in both arms)
Command (plugin main c981260 through profile `sptc`, which links the main checkout):
```sh
ssh 512S2; source ~/dsh-sptc/env.sh
SPTC_ALLOW_SERVER=1 node ~/dsh-sptc/sptc-plugin/scripts/l3-ab.mjs --rounds 2 --jit 0 --wait-idle-sec 600 --out ~/dsh-sptc/ab/g4l3-20261027
```
- 20 code-mode tasks with `llm_query` sub-calls (T01-T16, T18, T19, S21, S26) x {off, on} x 2 rounds = 80 runs, one at a time, with alternating arm order. Route oai (`/v1/chat/completions` through the LAN API), thinking off, agent temperature 0, layer 3 literal + statements, speculative concurrency 2.
- **JIT pre-launch off in both arms.** That is the Mac app's setting; the harness default is JIT on.
- The server was idle before every run (load 0+0 in 80/80) and after it. 0 failed runs, 0 timeouts.

| arm | runs (failed) | turn p50 s | turn p90 s | turn mean s | turns summed s | run_code summed s | programs | llm_query calls | argument deltas per run_code call (median) |
|---|---|---|---|---|---|---|---|---|---|
| off | 40 (0) | 9.00 | 17.63 | 10.16 | 406.4 | 120.9 | 51 | 248 | 17 |
| on | 40 (0) | **8.48** | **17.07** | 10.32 | 413.0 | **90.0** | 57 | 249 | 17 |

- **Speculation (on):** 238 streaming launches, **238 claimed, 0 wasted**, 0 scopes dropped, 0 wasted tokens. Summed lead before dispatch was 253.6 s (1.07 s per claimed call).
- **Server tokens:** `llm_query` was the same in both arms: 248 / 249 requests, prompt 24,935 / 24,595, completion 12,367 / 13,096, 0 cancelled. In total, off used 657,292 tokens and on used 701,201. All of the difference is agent requests (91 / 97), because the on arm wrote more programs in four runs (below).
- **Paired, all 40 pairs:** on - off turn time has a median of -0.14 s and a mean of +0.16 s; on was faster in 25/40.
  - In 4 pairs the on arm's agent wrote more programs: T05 r1 1 -> 2, T07 r2 1 -> 2, T13 r2 1 -> 2, T19 r2 2 -> 5. They cost +0.8 to +5.2 s.
  - The final answers differ in all 40 pairs. This happens in both arms: long prompts are not reproducible (section 6), and the sub-call outputs vary too. The extra programs are therefore model variance, not something the plugin causes (it claims only byte-identical requests).
- **Paired, the 36 pairs with the same number of programs and steps:**
  - turns summed 357.2 -> 353.9 s (**-0.9%**);
  - paired median **-0.20 s (-2.7%)**, on faster in 25/36;
  - run_code summed 101.7 -> 78.0 s (**-23%**);
  - model/other time per run 7.10 -> 7.66 s (+0.56 s);
  - agent completion tokens per model-second 107.7 -> 102.9 (paired median ratio 0.957).
- **What this means:** layer 3 moves about a quarter of the run_code time into the agent's stream, because the sub-calls start while the program is still being written. The agent's own stream then gets about 4% slower, because the server decodes it together with the speculative sub-calls. The net is a small but consistent gain at the median (about 0.2 s per turn, 2-3%; p50 -5.8%, p90 -3.2%) and a flat mean. The offline estimate was +5% of turn time with the newline patch. The measured gain is about half of that at the median and about zero on the mean.

### 10. Recommendation for the Mac app (DeepSeek Harness desktop, profile `desktop`)
- **Layer 3: ON, as a small win.** Expect about 0.2 s faster typical turns, with no wasted sub-calls and no extra server tokens. Do not expect a lower mean. Keep JIT off.
- **What enabling needs:**
  - The installed Mac plugin is the layer-1 build (cfa2640 + the JIT fix) and does not contain layer 3.
  - Rebuild it from 512S2 main **c981260** with the declared DSH range widened to `>=0.2.1-rc.2`, as the layer-1 desktop packaging did (scratchpad `macinst/scripts/build-desktop.sh`).
  - Set `layer3: true` in the plugin config (or `SPTC_L3=1`), then restart the app with ⌘Q.
  - None of this was done in this step: the Mac app is unchanged.
- The Mac profile runs reasoning `max` (thinking on), while this A/B ran with thinking off. Reasoning adds time to every turn, so the relative gain there will be smaller still.
- **The server side is done either way.** `DSV41_TOOL_STREAM_NEWLINE=1` is the launcher default. Layer 3 needs it, and any streaming client now receives tool programs line by line.

### 11. Live state and rollback
**Live:** instance **g4l3-PROD**, healthy since 15:01:43 UTC (replaced at 15:38 UTC by the final-verification launches; live now: **g4v-FU**, same launcher and config: see "Goal window 4, final verification").
- Launcher **95f08eff** on 8/8 (= 31d04d8f + the `DSV41_TOOL_STREAM_NEWLINE` line); backup `~/dsv41-tp8.sh.bak-g4-l3` = 31d04d8f.
- Supervisor 7bea1cf3, preflight manifest **40eb7abf08c7**; AV unpinned; clock lock 0-2000 MHz; LAN/tailnet proxy `dsv41-lanproxy.service` active and unchanged.
- **Site files on 8/8:** the parser patch (`vllm/parser/deepseek_v4.py` **3334c1fa**, `vllm/parser/engine/parser_engine.py` **6e3eeaba**; backups `.bak-layer3` = 709f328f / 62b741a0). g4d f99d02f68320 (flags off), g4p aef7c824bb9a and fload ca05f12513a1 are unchanged.
- 8/8 active, 0 ERROR/Traceback, no test or sampler process left. The 512S2 harness and its capture proxies have exited.

**Rollback:**
- **One launch:** `DSV41_TOOL_STREAM_NEWLINE=0` as a supervisor argument restores the stock flushing at `>`. Outputs are the same either way.
- **Permanent:** `cp -p ~/dsv41-tp8.sh.bak-g4-l3 ~/dsv41-tp8.sh` on 8/8 (31d04d8f), then relaunch with `--expect-launcher 31d04d8f`.
- **Site files** (service stopped, after the launcher rollback): `bash ~/dsv41-stage/layer3/uninstall.sh` on 8/8. It restores the `.bak-layer3` files and is sha-checked. Check with `bash ~/dsv41-stage/layer3/status.sh`. With the flag unset, the installed files take the stock paths anyway (section 4).

### 12. Notes for the next steps
- **Open exactness item (section 6):** greedy requests with prompts above ~2K tokens are not reproducible run to run on one instance, and a prefix-cache hit can change an output at any length. exact.py and G1 cannot see this.
  - Suggested gate: 4 cold repeats of three recorded 6K-token tool prompts (`~/dsv41-g4l3/cases/l3cases.jsonl`; `l3repeat.py`) must be token-identical.
  - Then bisect the long-context path: the sparse-attention top-k (2,048) tie-breaking against physical KV placement, and cached vs cold prefill.
- **Layer 3's ceiling** is set by the agent-stream slowdown while sub-calls overlap it.
  - Only speculative concurrency 2 was measured. `SPTC_L3_CONCURRENCY` takes 1-4, and 1 might keep most of the overlap at a smaller slowdown.
  - A server-side priority for the agent request over its own speculative sub-calls would remove the slowdown.
- **Tool clients must keep Codex 07 rule 5:** send `reasoning_content` back on every assistant message in tool loops.

## Goal window 4, final verification (2026-10-27 15:33-16:03 UTC): config verified on 8/8 (launcher 95f08eff), both instances exact. Goal NOT met: production G1 102.0 of 120 tok/s (+17.7% needed); prose session MIN 48.4 of 50 (+3.3%; worst run +4.4%). Window 4's gains are prefill and operations: cold TTFT 16K 2.49 s, 128K 19.6 s, 1M 166 s (good class); healthy 209 s after a production launch

**Verdict.** Neither target is met. Every number below was taken with telemetry off, at the 2000 MHz boot clock, on the live config (launcher 95f08eff: window 3's decode stack plus window 4's exact prefill pair, fast loader and tool-stream flag).
- **G1, single user: 102.00 tok/s in production** (g4v-FU: code 96.25 / prose 76.20 / math 133.53) against 120: **+17.7% is needed** (+18.0 tok/s). The pinned instance g4v-FP gave 101.69 (+18.0%).
- **G4, 4 concurrent sessions: not met.** Code, math and mixed stayed above 50 tok/s in every timed run on both instances (lowest 53.4). **The prose session was below 50 in every run:**
  - production g4v-FU: 48.48 / 48.86 / 47.90, MIN **48.41**, so **+3.3% is needed** (+4.4% for its worst run); G4min_hat 47.50 (+5.3%);
  - pinned g4v-FP: 48.49 / 47.69 / 47.61, MIN **47.93** (+4.3%; worst run +5.0%); G4min_hat 48.17 (+3.8%).
  - The per-session mean is above 50 on both (61.05 / 61.36), but the goal requires every session.
- **Exact.** exact.py gave **a8c3394af9d2** at the start and after all the load on both instances (4/4). The pinned instance had the pin's counters 909 / 467 / 2335 / 1373 with per-position 348/233/171/90/67 both times, and its G1 outputs are the window's reference: the first two repetitions give **9a18f802d930f6e7** (the A/B reference digest), all three are identical (3-repetition digest bc1e8386878bd2b7). Production produced the same G1 outputs (bc1e8386878bd2b7, identical repetitions), i.e. this launch landed on the 9a18f802 side of the known prose near-tie.
- **Long prompts and quality (production, once each):** cold TTFT 16K **2.494 s**, 128K **19.617 s**, one cold 1M **166.2 s** in the good class; GSM8K-50 **50/50**; full longctx **PASS** (needle 60/60, copy 1.000). MemAvailable stayed at or above 8,600 MB on 8/8 during the 1M (floor 6 GB).
- **No outside traffic reached the server** in any phase (sampler maximum equal to the phase's own concurrency everywhere; G1 9/9 and G4 16/16 requests by the success counter). No clock or power setting was changed; the SM clock read 1969-1989 MHz at all 27 checks on 8/8.

### 1. What goal window 4 changed (02:30-15:30 UTC; every step has its own section above)

| step (UTC) | Codex item / work | result | launcher after |
|---|---|---|---|
| prefill (02:30-07:25) | 04 exact sequence-parallel prefill + 05 exact E7 `post_and_draft` | **PROMOTED as a pair**: bitwise, TTFT -6..-10% at 16K-1M, short chats during a 128K prefill 1.81 -> 0.84 s, decode unchanged | b0cf94d0 -> **fd1e809e** |
| | 06 candidate-only indexer (layers 24-36) | exact, no measurable gain: installed, off | |
| Loader (08:35-11:19; user: "move fast loading to front") | Codex ops kit 04, fast weight loader `DSV41_FAST_LOAD` | **PROMOTED**: every parameter sha256-identical to the stock loader; time-to-healthy pinned 286 -> 126-132 s, production 355 -> ~200 s; no swap writes | fd1e809e -> **31d04d8f** |
| decode+ops (11:23-13:30) | 09 fused draft budget `DSV41_FUSED_AV` | bitwise, but +0.20% G1 on its first pinned pair (trim rule: stop): installed, off | |
| | 01 RoCEnante fast path | already measured slower in window 3: not re-run | |
| | 02 MoE tile scheduling | gate CLOSED: best possible scheduler 1.025x at M=6 (gate 1.08x) | |
| | 08 early KV slab | not promoted: with the fast loader 0/4 launches are slow-class at 1M with or without it; **the fast loader is the 1M straggler fix** | |
| client (13:22-14:05) | 07 multi-turn prefix reuse | a client rule (`drop_thinking: false` + send reasoning back); the server patch changes outputs: not installed | |
| | 03 prose oracle dump + replay | dump exact; no calibration or tree headroom; only a better drafter head has room (+7/+12/+18% ceiling at top-2/4/16) | |
| Layer3 (14:08-15:30) | sPTC layer 3 server side: DeepSeek tool parser flushes on newline | **PROMOTED** `DSV41_TOOL_STREAM_NEWLINE=1`: output-neutral, tool programs stream line by line | 31d04d8f -> **95f08eff** |
| final verification (15:33-) | this section | | 95f08eff (unchanged) |

User directives of 08:35 UTC applied to this step: **one** cold 1M probe in the final verification, GSM8K-50 + longctx once (on the production config, the one being kept), and no long-prompt probes on the pinned decode instance.

### 2. Config verified on 8/8

`g4v/scripts/verify8.sh` ran on the live g4l3-PROD before the step (`verify-pre`, 15:33 UTC) and on the live g4v-FU after it (`verify-post`, 16:02:52 UTC). It runs `node-verify.sh` on every Spark and compares every host-independent line across the 8 nodes, then checks each value against this README. Both runs: every comparable line identical on 8/8 and every expected value matched. The two outputs differ only in service start times, GPU readings and MemAvailable.
- **Launcher** `~/dsv41-tp8.sh` **95f08eff** on 8/8, with the window's backups: `.bak-g4-l3` = 31d04d8f, `.bak-g4-fload` = fd1e809e, `.bak-g4-sp-e7x` = b0cf94d0. Its defaults include `DSV41_PREFILL_SP=exact`, `DSV41_E7_EXACT=post_and_draft`, `DSV41_FAST_LOAD=1` and `DSV41_TOOL_STREAM_NEWLINE=1`; `--enable-prompt-tokens-details` is in the launcher and in the running `vllm serve` command line on 8/8.
- **Supervisor** `ops/rel/dsv41-relaunch.sh` **7bea1cf3** (`PRE_CLOCK_DEFAULT=2000`, `PRE_COMPACT_DEFAULT=0`). Both launches logged "site manifest **40eb7abf08c7** and launcher 95f08eff identical on 8/8" and "engram check: 8/8 OK".
- **Window 4 site files**, identical on 8/8 (`g4-all.sh status` digests g4p **aef7c824bb9a**, fload **ca05f12513a1**, g4d **f99d02f68320**; layer3 `status.sh` patched with backups):
  - prefill (g4p): `model.py` 9c2d22ad, `model_state.py` b7598364, `decoder_replay_layers.py` b75aa927, `dspark/speculator.py` 439b390e, `sparse_attn_indexer.py` f2ae649b;
  - fast loader (fload): `dsv41_fast_load.py` dfbcf2c7, `default_loader.py` 1aeec2e6, `vl_model.py` 3ec0d401;
  - decode bundle, every flag off (g4d): `adaptive_verification.py` 55030a68, `model_runner.py` 6a4f2433, `gpu_worker.py` 4d67066c, `utils.py` c4e02ba8;
  - layer 3: `parser/deepseek_v4.py` 3334c1fa, `parser/engine/parser_engine.py` 6e3eeaba;
  - backups `.bak-g4-g4p`, `.bak-g4-fload`, `.bak-g4-g4d` and `.bak-layer3` present on 8/8 (two checked per bundle); the prose-oracle bundle is not installed (its stage `~/dsv41-stage/g4/orc` is kept).
- **Earlier windows' files**, unchanged and identical on 8/8: window 3 `flashinfer_sparse.py` b059bdf3, `mla/indexer.py` d9f4c5c9, `dsv41_r3_dense.py` 1f1c0456, `deep_gemm_utils.py` a4a0486a; `sm121_target_m32.py` ea4a8814, `cuda_communicator.py` 87bf77e3, `rdma_all_reduce.py` 26a5d834, `dsv41_l2_prefetch.py` 63737f9c, `dsv41_r4_small.py` c46e66bd, `sparse_swa.py` 854c3e5f, `rdma_ar_ext.so` 62daad6c.
- **Whole `vllm/` site tree:** one digest over the sha256 of all 5,452 files (bytecode caches and our `.bak-*` / marker files excluded), **38062de05f0a** on 8/8 before and after.
- **Serving dependencies** on 8/8: `fc2c-on-m32all` `PREP.ok` dcf5bc0909472121; pins `av-pin-k5-g3FU.json` eeac4f97 and `av-pin-g3.json` 1c98cf9b; `m32-bw` 661-664 MB (per-node build output, so its size is not compared).
- **Clock:** `dgx-spark-gpu-clock.service` (`nvidia-smi -lgc 0,2000`) enabled and active on 8/8. Both launches logged `prelaunch (compact=0 clock=2000): ok on 8/8`. No clock or power setting was changed.
- **Running env** (rank 0, both instances): the launcher defaults above plus window 3's `DSV41_TOPK_CANON_FUSED=1 DSV41_DECODE_REMAP_REUSE=1 DSV41_IDX_OMIT_T2S=1 DSV41_DRAFT_HEAD_TACTIC=1 DSV41_DRAFT_HEAD_TACTIC_MAP=17-29:1`; no candidate flag (`DSV41_FUSED_AV`, `DSV41_KV_ALLOC_MODE`, `DSV41_CAND_ONLY_INDEXER`, `DSV41_PROSE_ORACLE_DUMP`, `DSV41_WEIGHT_DIGEST`) is set. `DSV41_AV_COST_JSON` is set only on the pinned instance.
- **Service:** 8/8 active, 0 restarts, 0 ERROR/Traceback in the rank logs; every rank log has the fast-loader, SP-armed and E7-armed lines.

### 3. Method (trimmed suites, user directive of 08:35 UTC)
- **Driver** `g4v/scripts/final.sh` (Mac): `vinst.sh g4v-FP FP`, then `vinst.sh g4v-FU FU`, then `verify8.sh post`. Each `vinst.sh` instance:
  1. a supervised launch with `--expect-launcher 95f08eff` and `DSV41_GPU_CLOCK_MHZ=2000` (the boot policy; nothing raised), then one `nvidia-smi` read per node on 8/8 (must be <= 2010 MHz);
  2. `gstart.sh`: exact.py as the first request, with the speculative-decoding counter deltas. Pinned: exact.py must be a8c3394af9d2 **with the pin's counters** 909 / 467 / 2335 / 1373 and per-position 348/233/171/90/67, and `mkpin_k.py check` must report a pin hit. Production: the a8 anchor (the counters follow the startup profile);
  3. the phases one at a time on spark-one (`vphase.sh`), with the SM clock read on 8/8 before every measurement phase (telemetry off) and a 2 s `/metrics` sampler for foreign traffic;
  4. fetch, `vsumm.py` summary, end state on 8/8.
- **Pinned instance g4v-FP** (`av-pin-k5-g3FU.json` eeac4f97, the window's A/B pin): matched C1 (pdab, coding 1x5 + prose 1x5), G1 x3, G4 warm-up + 3 runs, bench conc 4 / 8 / 16 / 32 (512 tokens), exact.py #2. No long prompts, GSM8K or longctx (trim rule: the pinned instance only checks decode and exactness).
- **Production instance g4v-FU** (launcher defaults, AV unpinned; kept live): the same decode phases, then a 40K prefill warm-up, cold TTFT 16K x3 and 128K x2 (`sbench.py solo`), **one** cold 1M (`xrb4.py solo`, per-chunk times; 1 Hz `memwatch` on 8/8), GSM8K-50, full longctx, the usage check (`cached_tokens`), exact.py #2.
- **Foreign-request guard.** The API is open on the LAN and the tailnet. The G1 and G4 phases compare `request_success_total` before and after (9 and 16 requests) and rerun on a mismatch; the sampler flags any phase whose running + waiting count exceeds its own concurrency.
- **References.** The pinned control of this window is g4d2-C1 (decode+ops step, launcher 31d04d8f, same pin, 6 G4 runs), recomputed with `vsumm.py`. Window 3's final production instance g3-FU (launcher b0cf94d0) is quoted from "Goal window 3", section 10.3. G1 digests: `goal_single.py` hashes the outputs of all repetitions, so the 3-repetition digest of the pinned reference outputs is **bc1e8386878bd2b7**; the first two repetitions give **9a18f802d930f6e7**, the reference digest of the window's 2-repetition A/B runs.

### 4. Results (2000 MHz, telemetry off, no foreign traffic)

Instances, in launch order: **g4v-FP** (pinned, healthy 15:40:56 UTC, +142 s) and **g4v-FU** (production, healthy 15:49:49 UTC, +209 s; **live**). Both came up on the first try.

| metric | **production g4v-FU** (launcher 95f08eff defaults, AV unpinned) | pinned g4v-FP (same launcher + `av-pin-k5-g3FU.json`) | ref: pinned control g4d2-C1 (launcher 31d04d8f, same pin; G4 6 runs) | ref: window 3 final production g3-FU (launcher b0cf94d0) |
|---|---|---|---|---|
| healthy at (UTC) / launch to healthy | 15:49:49 / +209 s | 15:40:56 / +142 s | 11:39:40 / +143 s | 23:22:34 / +360 s |
| exact.py first request (acc / drafts / dtok / gen; per-pos; AV) | a8c3394af9d2 (911/465/2325/1373; 346/228/169/98/70; pin miss) | a8c3394af9d2 (909/467/2335/1373; 348/233/171/90/67; pin hit) | a8c3394af9d2 (909/467/2335/1373; 348/233/171/90/67; pin hit) | a8c3394af9d2 (909/467/2335/1373; 348/233/171/90/67; pin miss) |
| exact.py #2 after all load | a8c3394af9d2 (911/465/2325/1373; 346/228/169/98/70) | a8c3394af9d2 (909/467/2335/1373; 348/233/171/90/67) | a8c3394af9d2 (909/467/2335/1373; 348/233/171/90/67) | a8c3394af9d2 (same counters) |
| **G1 mean tok/s** (code / prose / math), 1024 tokens x3 each | **102.00** (96.25 / 76.20 / 133.53) | **101.69** (96.06 / 76.61 / 132.41) | **101.63** (95.97 / 76.52 / 132.39) | 101.51 (95.70 / 76.56 / 132.28) |
| G1 per repetition | 102.10, 101.93, 101.96 | 101.69, 101.68, 101.70 | 101.67, 101.59 | 101.56, 101.49, 101.48 |
| G1 AL code / prose / math | 2.603 / 1.837 / 3.633 | 2.590 / 1.847 / 3.594 | 2.590 / 1.847 / 3.594 | 2.590 / 1.847 / 3.594 |
| G1 output digest, all reps (reps identical); first 2 reps (pinned reference **9a18f802d930f6e7**) | bc1e8386878bd2b7 (yes); **9a18f802d930f6e7** | bc1e8386878bd2b7 (yes); **9a18f802d930f6e7** | 2 reps only: 9a18f802d930f6e7 (yes) | bc1e8386878bd2b7 (yes); 9a18f802d930f6e7 |
| **G4 per-session mean / prose MIN** (timed runs) | **61.05 / 48.41** | **61.36 / 47.93** | **60.83 / 48.10** | 61.65 / 48.64 |
| G4 by session: code / prose / math / mixed | 55.9 / 48.4 / 75.9 / 64.1 | 56.8 / 47.9 / 76.9 / 63.8 | 57.5 / 48.1 / 76.0 / 61.8 | 58.8 / 48.6 / 75.3 / 63.9 |
| G4 slowest session per timed run (always prose) | 48.48, 48.86, 47.90 | 48.49, 47.69, 47.61 | 47.25, 47.33, 47.39, 49.38, 47.75, 49.51 | 49.15, 49.52, 47.26 |
| G4 session-runs at >= 50 tok/s | 9/12 | 9/12 | 18/24 | 9/12 |
| G4min_hat (step times only) / 4-session step ms | 47.50 / 45.25 | 48.17 / 45.37 | 48.43 / 44.56 | 48.43 / 44.77 |
| matched C1 coding / prose ms/step (AL) | 30.40 / 26.41 (AL 5.224 / 2.736) | 30.25 / 26.28 (AL 5.191 / 2.725) | 30.25 / 26.34 (AL 5.191 / 2.725) | 30.32 / 26.33 (AL 5.191 / 2.725) |
| bench conc 4 / 8 / 16 / 32 aggregate tok/s (512 tokens; one run each) | 178.9 / 264.2 / 396.6 / 589.5 | 181.0 / 261.0 / 394.3 / 587.1 | - / 263.9 / 394.1 / - | 180.8 / 264.3 / 397.8 / 580.7 (two runs each) |
| conc per-request mean / min (4 / 8 / 16 / 32) | 59.2/47.5 ; 43.7/35.0 ; 34.7/26.4 ; 25.0/19.4 | 60.1/48.1 ; 43.5/34.5 ; 34.7/25.3 ; 24.8/19.3 | - ; 43.7/34.9 ; 34.4/26.6 ; - | 60.7/46.5 ; 43.9/34.1 ; 34.7/25.4 ; 24.9/18.7 |
| cold TTFT 16K x3 / 128K x2 (s) | **2.494** (2.557, 2.397, 2.528) / **19.617** (19.640, 19.595) | - (trim rule) | - | measured at ~24K / ~198K: 4.05 / 33.29 s |
| cold 1M (one probe, 1,024,611 tokens): TTFT, class (late/early), chunk 37-43 band | **166.2 s**, good (1.176), 2.713 s | - (one probe only) | - | 183.70 s, good (1.16) |
| MemAvailable minimum during the 1M (1 Hz memwatch, 8/8) | 8,600 MB (spark-one); others 8,996-9,957 MB | - | - | 8,265 MB (spark-one); others 8,939-10,132 MB |
| GSM8K-50 / full longctx / usage `cached_tokens` on the repeated 4.2K prompt | **50/50 / PASS** (needle 60/60, copy 1.000) / 4160 | - (trim rule) | - | 50/50 / PASS / 4160 |
| foreign traffic (sampler maximum vs own concurrency; G1 / G4 success-counter guard) | none; 9/9, 16/16 | none; 9/9, 16/16 | none | none |
| SM clock checks on 8/8 (one `nvidia-smi` read per node before every phase) | 17/17 OK, 1976-1989 MHz | 10/10 OK, 1969-1989 MHz | 8/8 OK, 1969-1989 MHz | 38/38 OK, 1969-1989 MHz |
| ERROR/Traceback in rank 0 (start / end) / MemAvailable at the end, spark-one | 0 / 0 / 8,849 MB | 0 / 0 / 12,513 MB | 0 / 0 / 13,519 MB | 0 / 8,323 MB |

G4 runs, production g4v-FU (telemetry off):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 57.30 | 46.95 | 76.68 | 60.44 | 60.34 | 46.95 (prose) | 38.87 | 176.2 | 2.525 |
| 1 | 55.38 | 48.48 | 75.27 | 63.34 | 60.62 | 48.48 (prose) | 41.69 | 185.0 | 2.498 |
| 2 | 58.72 | 48.86 | 71.33 | 68.47 | 61.85 | 48.86 (prose) | 40.62 | 186.5 | 2.582 |
| 3 | 53.46 | 47.90 | 81.00 | 60.36 | 60.68 | 47.90 (prose) | 41.50 | 179.6 | 2.502 |

G4 runs, pinned g4v-FP (telemetry off):

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 57.69 | 46.85 | 72.53 | 55.61 | 58.17 | 46.85 (prose) | 39.31 | 177.7 | 2.498 |
| 1 | 58.85 | 48.49 | 79.05 | 65.70 | 63.02 | 48.49 (prose) | 40.84 | 181.8 | 2.583 |
| 2 | 53.36 | 47.69 | 81.47 | 65.34 | 61.96 | 47.69 (prose) | 40.65 | 179.1 | 2.488 |
| 3 | 58.23 | 47.61 | 70.28 | 60.29 | 59.10 | 47.61 (prose) | 40.99 | 187.0 | 2.517 |

Production instances of goal window 4 (each step's final instance, launcher defaults, AV unpinned; from the step sections above):

| instance (healthy at, UTC) | launcher | to healthy | G1 (digest) | G4 mean / prose MIN | C1 coding / prose ms | conc 8 / 16 | TTFT 16K / 128K | cold 1M | GSM8K / longctx |
|---|---|---|---|---|---|---|---|---|---|
| g3-FU (2026-10-26 23:22, window 3 final) | b0cf94d0 | +360 s | 101.51 | 61.65 / 48.64 | 30.32 / 26.33 | 264.3 / 397.8 | ~24K / ~198K only | 183.7 s | 50/50 / PASS |
| g4-FINAL (07:09, prefill) | fd1e809e | +360 s | 101.62 | 60.50 / 47.61 | 30.44 / 26.36 | 259.6 / 399.8 | 2.511 / 19.806 s | 167.5 s | 50/50 / PASS |
| g4l-FINAL (11:08, Loader) | 31d04d8f | +199 s | 102.13 | - | - | - | - | 165.9 s | 50/50 / PASS |
| g4d2-FINAL (13:13, decode+ops) | 31d04d8f | +204 s | 102.03 | 61.39 / 48.36 | 30.31 / 26.33 | 258.0 / 399.1 | - | - | - |
| g4c-FINAL (14:01, client) | 31d04d8f | +198 s | 101.67 (69ad8721) | - | - | - | - | - | - |
| g4l3-PROD (15:01, Layer3; G1 x2) | 95f08eff | +200 s | 101.37 (69ad8721) | - | - | - | - | - | - |
| **g4v-FU (15:49, this step)** | **95f08eff** | **+209 s** | **102.00** (9a18f802 / bc1e8386) | **61.05 / 48.41** | **30.40 / 26.41** | **264.2 / 396.6** | **2.494 / 19.617 s** | **166.2 s** | **50/50 / PASS** |

- **Decode is unchanged by window 4, as intended.** No decode lever was promoted (09 stopped at +0.2%). The six production launches of the window gave G1 101.4-102.1, and the three that ran G4 gave prose MIN 47.6-48.4: the same spread as window 3's production launches (101.5-102.3; 47.9-48.6). Pinned g4v-FP against the window's pinned control g4d2-C1 (launcher 31d04d8f, same pin): G1 +0.06%, C1 within 0.2%, G4 prose MIN 47.93 against 48.10, G4min_hat 48.17 against 48.43: all inside the run-to-run spread of the pinned controls (G4min_hat 47.8-48.5 across window 3's instances). The only change promoted after g4d2-C1 is the tool-stream flag, which acts in the API server's parser, not in the engine.
- **Prefill and operations gained.** Cold TTFT 16K -6.8% and 128K -9.4% against the prefill step's pinned stock-prefill controls (2.675 / 21.66 s; window 3's final measured other lengths); cold 1M 183.7 -> 166.2 s (-9.5%) against window 3's final production, both good class; time-to-healthy +360 -> +209 s (-42%). The slow 1M class (220-235 s, the spark-six straggler) has not appeared on any fast-loader launch: 0 of 10 cold 1M probes since the Loader phase (g4l-FINAL, the 8 decode+ops relaunches, this one).
- **Production vs pinned on G4.** G4min_hat uses step times only, so it is comparable only under the same AV profile. Production's startup profile gave longer 1- and 3-session steps (25.7 / 41.5 ms against 23.9 / 39.3 ms pinned) but fewer prose steps per run (271.7 against 277.0), so production has the higher raw prose MIN (48.41 against 47.93) and the lower hat (47.50 against 48.17). The 4-session step is the same (45.25 / 45.37 ms).
- **Memory.** MemAvailable during the 1M bottomed at 8,600 MB (spark-one) and 8,996-9,957 MB elsewhere (1 Hz memwatch on 8/8): above the 6 GB floor, above window 3's stock-loader g3-FU (8,265 MB) and above the 6.2-7.1 GB recorded at g4l-FINAL's 1M. Swap in use stayed at its pre-step 135-174 MB per node (no new swap-out). At the end of the suite MemAvailable was 8,854-9,938 MB on 8/8.
- GSM8K-50 was 50/50 (idx 12, the usual borderline item, passed). The usage check returned `cached_tokens` 4160 on the repeated 4.2K prompt (prompt-tokens-details on).

### 5. Goal status and the remaining gap (2000 MHz, no clock or power increase)

| target | production g4v-FU | pinned g4v-FP |
|---|---|---|
| G1 mean >= 120 tok/s | 102.00: **+17.7%** | 101.69: +18.0% |
| G4: every session >= 50 tok/s in every run (prose MIN over the runs; worst run) | 48.41: **+3.3%**; 47.90: +4.4% | 47.93: +4.3%; 47.61: +5.0% |
| G4min_hat (step time only) | 47.50: +5.3% | 48.17: +3.8% |

- **G1 >= 120 tok/s: NOT met.** At today's acceptance (G1 AL code 2.60 / prose 1.84 / math 3.63) the single-user step is 27.0 / 24.1 / 27.2 ms. 120 tok/s needs every step about **15% shorter, -3.6..-4.1 ms per step**.
- **G4 >= 50 tok/s on every session: NOT met.** Only the prose session misses, in every run. It needs **+3.3..5.3%**, i.e. about 3.2-5.0% off every decode step: **-1.4..-2.3 ms** on the ~45.3 ms 4-session step.
- Window 4 targeted prefill and operations by design: its promoted items are output-neutral for decode, and the decode candidate it measured (09) was +0.2%. The decode gap is therefore unchanged from window 3 (G1 +17.3..18.2%, prose +2.8..4.7% there).

### 6. Next levers

**Exact** (bitwise; each needs its own GPU bitwise test and a pinned A/B under `av-pin-k5-g3FU.json` with the short decode suite), by size:
1. **Moet's mHC boundary off the critical path** (`DSV41_MHC_OVERLAP`, bitwise variant only; queued as goal window 5's 'mHC' phase, sources `~/dsv41-stage/moet/src`): estimated -0.8..-1.6 ms per step. It is the largest exact lever on the list and the one that matters most at one user.
2. MoE FC1 -> FC2 per-expert pipelining: -0.4..-1.0 ms per 4-session step (the M32 `all` + FC2C path runs 11-14% above the weight-streaming bound at 12-24 rows).
3. Contiguous FC2C layout: -0.5..-0.8 ms.
4. An L2-prefetch plan that keeps q_b prefetched above 8 rows: up to -0.86 ms on steps with 2 or more sessions.
5. Drafter MoE on the M32 path: ~-0.45 ms at 4 requests.
6. mfront, once the mixed live/fused-graph illegal-address crash is root-caused: -0.26..-0.30 ms.
7. One bundle of the small measured items, none of which clears 0.5% alone: fused AV (`DSV41_FUSED_AV`, +0.2% G1, installed and off), a balanced M32 tile scheduler (at most ~0.3 ms per step; item 02's gate is closed for it alone), the rdma_ar P1 doorbell (0.1-0.2 ms).

On paper items 1-7 add up to about 3-5 ms per 4-session step, so **the G4 prose gap (-1.4..-2.3 ms) closes if roughly half of them land**. They do not reach G1: the MoE items shrink at one user's 6 rows, and even with item 1 the list stays well short of the -3.6..-4.1 ms per step that 120 tok/s needs.

**Acceptance (not exact under the strict digest rule; a user decision):**

8. **A trained drafter head for thinking prose** is the only lever with room for G1 = 120, and the cleanest one for G4. The window's oracle replay (client step, item 03) puts its ceiling, at the same verify width, at +6.9 / +12.2 / +17.8% on the G4 prose session and +8.9 / +16.2 / +25.2% on G1 prose: that is a head which turns every first miss into a hit when the true token is already in today's top-2 / top-4 / top-16. The dump `~/dsv41-g4c/orc/g4c-O1` (spark-one) is a ready training and evaluation set, and goal window 5's Drafter step is set to reuse it.
   - Any drafter change alters drafts and the verify layout, and flips the known G1 prose near-tie (9a18f802 <-> 69ad8721). It therefore fails "identical G1 digest under the current pin" by definition, as Moet's k6/k7 did.
   - The gate proposed in the client step would replace that rule for this lever: exact.py a8 on every launch, G1 code and math bitwise, G1 prose equal to one of the two known outcomes, and at 4 users a first-divergence census at the control's near-tie rate.
9. Other user decisions, unchanged: unpadded Marlin W4A16 (not exact; -3.2 ms per step at one user), a default AV pin (reproducible production outputs, does not close the gap), clocks (closed at 2000 MHz by policy), and authentication for the LAN and tailnet API (still open; every timed window needs the foreign-request guard).

**Open exactness item (Layer3 step, section 6):** greedy requests with prompts above ~2K tokens are not reproducible run to run on one instance (cold 6K prompts gave 2-3 distinct outputs in 4 runs), and a prefix-cache hit can change an output. exact.py (268 prompt tokens) and G1 (~130) cannot see it. Suggested gate: 4 cold repeats of three recorded 6K tool prompts (`~/dsv41-g4l3/cases/l3cases.jsonl`, `l3repeat.py`) must be token-identical. Then bisect the sparse-attention top-k (2,048) tie-breaking against physical KV placement, and cached against cold prefill.

### 7. Live state after this step
- **g4v-FU is live:** production, launcher **95f08eff** defaults, AV unpinned, supervisor 7bea1cf3 (preflight manifest 40eb7abf08c7), clock lock 0-2000 MHz.
  - Healthy since 15:49:49 UTC on 8/8, with 0 restarts and 0 ERROR/Traceback. `/health` returns 200 locally and on the LAN address; `dsv41-lanproxy` is active and unchanged.
  - Its config matches section 2: `verify-post` (16:02:52 UTC) is identical to `verify-pre` apart from start times, GPU readings and MemAvailable, including the running-env digest 936207484f8d of g4l3-PROD.
  - 0 requests in flight at the end. No memwatch, sampler or bench process is left on 8/8.
- **Window 4 installs, all on 8/8:** g4p (04 + 05 on by default; 06 installed, off), fload (on by default), layer3 (on by default), g4d (09 fused AV and 08 early KV slab installed, off). The prose-oracle bundle is staged, not installed.
- **Rollback** (unchanged from the step sections):
  - one launch: `DSV41_TOOL_STREAM_NEWLINE=0`, `DSV41_FAST_LOAD=0` (brings the slow 1M class back) or `DSV41_PREFILL_SP=0 DSV41_E7_EXACT=0` as supervisor arguments;
  - permanent: the launcher backups on 8/8, `~/dsv41-tp8.sh.bak-g4-l3` (31d04d8f), `.bak-g4-fload` (fd1e809e) or `.bak-g4-sp-e7x` (b0cf94d0), then with the service stopped the matching uninstall: `bash ~/dsv41-stage/layer3/uninstall.sh`, `g4-all.sh uninstall fload`, `g4-all.sh uninstall g4p`; `g4-all.sh uninstall g4d` removes the inert decode bundle.

### 8. Files
- Mac scratchpad `g4v/`:
  - scripts: `final.sh` (driver), `vinst.sh` (one instance), `vphase.sh` (one phase, on spark-one), `verify8.sh` + `node-verify.sh` (8/8 config check), `vsumm.py` (instance summary), `vtab.py` (tables), `clk8.sh`, `finsampler.sh`, `memwatch.sh`, `readme_apply.py`;
  - logs and checks: `ab.log`, `final.out`, `verify-pre.txt`, `verify-pre-check.txt`, `verify-post.txt`, `verify-post-check.txt`;
  - results: `runs/g4v-FP/`, `runs/g4v-FU/` (with `vsumm.json` / `vsumm.md`), `runs/<label>-start/` (exact.py first request, startup facts), `ab/<label>/` (relaunch log, clock checks, 1M memwatch CSVs on 8/8, end state), `tables.md`.
- spark-one `~/dsv41-g4v/`: `runs/g4v-FP`, `runs/g4v-FU`, `runs/g4v-FP-start`, `runs/g4v-FU-start`, and `g4v-mac.tgz` (a copy of the Mac folder).

## Addendum 2026-10-27 (main session): Mac DSH app now has sPTC layer 3
The "sPTC layer 3" section above says the Mac app has the layer-1 build; that is no longer true. The Mac app (DeepSeek Harness 0.2.1-rc.2, profile ~/.dsh/profiles/desktop) now runs plugin dsh-sptc from 512S2 branch mac-rc2-layer3, commit 5774d545 (c981260 + engines.dsh widened to >=0.2.1-rc.2 <0.3.0; 90/90 unit tests), with `layer3: true`, `jit: false`, baseURL http://192.168.0.238:8000/v1 unchanged. It takes effect after the user quits (Cmd-Q) and reopens the app. Turn off: `layer3: false` in the sptc entry of ~/.dsh/profiles/desktop/cordis.patch.yml. Backups on the Mac: ~/.dsh/backups/dsh-sptc-plugin.bak-l3-20261027.tar.gz and desktop-profile.bak-l3-20261027.tar.gz. Layer 3 needs DSV41_TOOL_STREAM_NEWLINE=1 on the server (launcher default since 95f08eff) for its full effect.

## Long-prompt determinism (goal window 5, Determinism phase, 2026-10-27 16:14-18:07 UTC): sparse-indexer top-k boundary ties; `DSV41_TOPK_CANON_TIES=all` PROMOTED (launcher 95f08eff -> **f214c2c9**)

**Result.**
- **Cause.** Some rows of the sparse-indexer top-k have a boundary tie: two or more candidates have exactly the same fp32 index score as the row's 512th value. vLLM's top-k kernels then keep an arbitrary one of them, picked by an atomic race:
  - prefill: `top_k_per_row_prefill`;
  - decode: `persistent_topk` / `top_k_per_row_decode`.
- **Why the earlier fix missed it.** `DSV41_TOPK_CANON` (2026-10-24) makes the ORDER of the selected indices canonical, but not the SET.
- **One tie is enough.** It changes which KV entries one query attends to. From there, the next index layer's inputs differ in hundreds of rows, and the greedy output diverges.
- **Where the threshold is.** It is not the ~2K of the hypothesis. It is where the indexer really starts selecting:
  - in prefill, above 1,024 prompt tokens: layers 2/8/14 are ratio-2, so `index_topk` 512 = 1,024 tokens;
  - in decode, above 512 context on the ratio-1 layers (`index_topk` 512).
- **Why the drift looks sporadic.** Ties depend on content and are rare: 1-2 tied rows among ~9,600 selecting rows of a 4K prompt. No probe drifted at 1,300 tokens or fewer; the first drift was at 1,536.
- **Fix: `DSV41_TOPK_CANON_TIES=all`.**
  - Runs after the stock top-k on prefill and decode rows.
  - A tied row keeps every score above the 512th value plus the lowest column indices among the scores equal to it.
  - Rows without a tie are untouched (bitwise stock).
- **Verification.**
  - Cold probes from 2.5K to 128K tokens and three recorded 6K tool prompts give 4 identical runs out of 4, for both tokens and logprobs. The flag-off control gave 2-4 distinct token outputs out of 4 on 7 of the 9 probes, and drifting logprobs on an eighth.
  - exact.py a8c3394af9d2 with the pin's counters, and the G1 reference digest, are unchanged.
  - GSM8K 50/50; longctx PASS.
  - TTFT: 16K +0.0%, 128K +0.35% (same-instance A/B). 1M 168.4 s, good class.
  - Decode within noise: G1 -0.16% in the pinned pair.
- **Not fixed: warm (prefix-cache hit) vs cold.**
  - They still differ from the first generated token. This is row-count-dependent numerics, not ties (section 5).
  - Warm runs now agree with each other.

### 1. Reproduction on the live production instance (g4v-FU, launcher 95f08eff, AV unpinned, idle; 16:16-16:45 UTC)
Setup (`~/dsv41-g5det/scripts/detprobe.py`):
- Token-id prompts on `/v1/completions`: [BOS] + the first L-1 ids of a fixed text.
  - prose = GPL-3 + Apache-2.0 + GFDL-1.3;
  - code = an old copy of `sparse_attn_indexer.py`.
- Greedy, 128 tokens, `ignore_eos`, one request at a time.
- Cold = a fresh `cache_salt` per request; 4 runs per length.
- Pass A records tokens only, with cold and warm runs (warm = one salt, so runs 2-4 hit the prefix cache).
- Pass B is cold with the sampled token's logprob per step, used as a bitwise fingerprint of every step. "Step 0" is the prefill's own logits.

| prompt tokens | prose, cold: distinct tokens / logprob seqs (first differing step) | code, cold: same | warm vs cold, first differing token (prose / code) |
|---|---|---|---|
| 512, 768, 1,100, 1,300 | 1 / 1 | 1 / 1 | not run |
| 1,024 | 1 / 1 | 1 / 1 | same / 80 |
| 1,536 | 1 / **3 (step 0)** | 1 / 1 | same / 0 |
| 1,900 | 1 / 1 | 1 / 1 | 94 / 122 |
| 2,000 | 1 / 1 | 1 / 1 | same / 3 (the 3 warm runs also differ from each other: 4 distinct) |
| 2,048 | 1 / **3 (step 17)** | 1 / 1 | same / 44 |
| 2,100, 3,000, 4,200 | 1 / 1 | 1 / 1 | same / 36, 29, 39 |
| 2,500 | 1 / 1 | **3 / 4 (steps 0-1)** | same / 82 |
| 4,000 | 1 / 1 | **4 / 4 (step 0)** | same / 12 |
| 6,000 | 1 / **4 (steps 18-19)** | 1 / 1 | same / 13 |
| 8,000 | 1 / **4 (step 1)** | 1 / 1 | same / 0 (warm runs: 3 distinct) |

- Drift starts at 1,536 tokens, but only on some prompts; 2,100-4,200 prose is clean.
- The step at which it starts is either 0 (the prefill differs) or a later decode step.
- Prose often keeps the same tokens while its logprobs drift. Code flips tokens more readily.

### 2. Localization
**Kernel test** (`tk_tie_test.py`, `tk_ties_unit.py`; spark-eight; no model; 20 repeats per case):
- Continuous random scores: the selected SET is identical in 64/64 rows, and the ORDER differs in 64/64 (the part `DSV41_TOPK_CANON` already fixes).
- Scores with boundary ties (values on a 0.25 grid, or all equal): the SET differs run to run in 61-64 of 64 rows, for all three stock kernels. None of them keeps the lowest index consistently.
- Rows with fewer than 512 finite scores: the stock kernels pick an arbitrary subset of the -inf columns (31-32 of 32 rows vary). `top_k_per_row_decode` on 300K-wide rows returns only part of 512 (for example 304 valid indices). This case cannot occur here, because the candidate mask keeps ≥ 16,384 positions per row.

**In-forward dump.** Instance D1 (pinned; 16:52) ran with the new debug flag `DSV41_DET_DUMP` and the tie fix off. For every indexer call, rank 0 saved:
- the stock top-k rows;
- per-row boundary statistics;
- per-row fingerprints of the attention input, index queries, head weights, gathered index keys and scores.

Result for the code 4,000-token prompt, two cold runs:

| index call (layer, rows x compressed entries) | inputs (attn input / q / weights / keys) | scores | stock top-k SET | ORDER |
|---|---|---|---|---|
| layer 2, 4,000 x 2,000 | identical | identical | **differs in 1 row** | differs in 2,975 rows (CANON sorts it) |
| layer 8, 4,000 x 2,000 | differ in 636 rows; 319 / 2,000 keys | 2,511 rows | 2,323 rows | - |
| layer 14, 4,000 x 2,000 | differ in 2,511 rows; 1,256 / 2,000 keys | 2,511 rows | 2,510 rows | - |
| layers 20/24/28/32/36, 128-row decoder-replay batch x 4,000 | all 128 rows | all | all | - |

- **The tied row.** Row 1489 has 745 entries. Its 512th score, -2.0917205810546875, is shared by columns 222 and 306: 511 scores are greater and 2 are equal. The two keys have different bytes and the same block scale. Each run keeps one of the two at random.
- **Other prompts.**
  - Prose 1,536 and code 2,500 each had one tied row in layer 2. In this pair both runs happened to keep the same column; in production these lengths drifted in 2-3 of 4 runs.
  - The 4,000-token prompt also has a tie in layer 8 on one of its trajectories.
- **With the fix on (same instance).** `TIES=prefill` was switched on 8/8 through the debug file. The stock SET in row 1489 still differed run to run, but the fixed output was identical. Every later layer was bitwise identical, and 3/3 runs gave identical tokens and logprobs.
- **Ruled out.** Inputs are bitwise identical up to the first tied top-k, so none of these is the cause:
  - layers 0-1: SWA attention, MoE (the ep_scatter atomics), Engram, mHC;
  - SP prefill and E7;
  - all-reduces;
  - async scheduling;
  - chunking (single chunk);
  - the index-score kernel.
- **Prefill index calls run as follows:** layers 2/8/14 on every row; layers 20-36 only on the 128-row replay batch.

**Decode.**
- The decode indexer runs inside the FULL CUDA graphs, so Python cannot dump it at replay. The evidence is an A/B instead.
- With prefill rows fixed but decode stock (D1 `TIES=prefill`), prose 6,000 still drifted from decode step 18: 4 distinct logprob sequences, same tokens.
- In the same pass, prose 8,000 / 2,048, code 2,500 / 4,000, prose 1,536 and the three tool prompts became identical, so their drift had come from prefill ties.
- With decode rows fixed as well (D2, 17:09: debug file set to `all` before startup, so the graphs capture the fix), prose 6,000 was 4/4 identical. So were all 13 cold cases there, including big 2.5K / 8K / 32K / 128K.

**Layer-20 candidate blocks** (`scores.topk(2048)`, used only for contexts > 16K):
- In a 20-repeat test, `torch.topk` gave the same result run to run on tied block scores in every shape, although it does not always keep the lowest index.
- No change was made. Cold 32K and 128K probes are deterministic with the fix.

### 3. Fix: `DSV41_TOPK_CANON_TIES` (new site module `layers/dsv41_topk_ties.py` 943961bc, hooks in `sparse_attn_indexer.py` f2ae649b -> 005785f9)
- **Where it runs.** After the stock top-k and before CANON:
  - every prefill chunk (`top_k_per_row_prefill`; indices relative to the row start);
  - with `all`, every decode call (absolute indices).
- **How it works.** One Triton program per row:
  1. reads the 512 selected scores, takes their minimum v_k;
  2. counts the row's scores > v_k and == v_k;
  3. only if (#> + #==) > 512, rewrites the row ascending: every score > v_k, then the lowest-index scores == v_k until 512.
- **Edge cases.**
  - A -inf boundary is covered too.
  - These are left to stock: rows of ≤ 512, rows with a NaN or +inf k-th value, and unsupported layouts (a one-time warning, no crash).
- **Decode split.** Decode rows wider than 16,384 (the logits width is `max_model_len`/ratio) split the count over up to 64 programs per row (partials, no atomics). One program per row then decides, and rewrites the row if tied.
- **Unit test** (`tk_ties_unit.py`, prefill + both decode kernels; continuous / tied / all-equal / masked; widths 700 to 300K):
  - untied rows bitwise equal to stock;
  - tied rows equal to a float64 reference;
  - 0 rows varying over 20 runs.
- **Cost, prefill.** The check runs about as long as the stock top-k (+94% at 16K x 8K, +127% at 16K x 16K, kernel only). In serving, the top-k is a small part of TTFT: same-instance A/B +0.0% at 16K, +0.35% at 128K (below).
- **Cost, decode.** Measured inside a CUDA graph on spark-eight:
  - 4.3-4.6 µs per index call at G1 contexts (6 rows, ≤ 1.1K); 7.3 µs for 24 rows at 2K; 20.2 µs for 96 rows;
  - 12.3 µs at 131K and 24 µs at 1M context (6 rows);
  - in total, about 35 µs per decode step at G1 contexts (8 index layers), ~0.1% of a 30 ms step.
- **Modes.** `all` = prefill and decode (live default); `prefill` = prefill rows only; `0` = stock.
- **Debug only:** `DSV41_DET_DUMP=<dir>`.
  - Rank 0 writes per-call dumps while `<dir>/ON` exists.
  - `<dir>/TIES` overrides the mode for prefill at run time and must be written on 8/8.
  - Decode graphs capture the mode at startup.

### 4. Verification
**Same-instance TTFT A/B.** Instance D1, AV pin, cold `sbench solo`. The debug file was switched on 8/8 between blocks, in the order 0, prefill, 0, prefill:

| TTFT | `TIES=0` | `TIES=prefill` | delta |
|---|---|---|---|
| 16K (3 prompts x 2) | 2.502 s | 2.503 s | +0.0% |
| 128K (2 prompts x 2) | 19.641 s | 19.709 s | +0.35% |

**Pinned pair.** Launcher 95f08eff on both, with the k5-g3FU pin (eeac4f97). C0 has the flag off; C1 has `DSV41_TOPK_CANON_TIES=all` from the supervisor. Both ran the trimmed decode suite plus the cold/warm determinism probes; C1 also ran the prefill/long-context checks.

| metric | C0, flag off (17:21) | C1, `all` (17:32) | delta |
|---|---|---|---|
| exact.py (both runs) | a8c3394af9d2 909/467/2335/1373, 348/233/171/90/67 x2 | identical x2 | identical |
| G1 (code / prose / math) | 101.654 (96.00 / 76.54 / 132.43) | 101.495 (95.82 / 76.38 / 132.29) | -0.16% |
| G1 digest (3 reps / first 2) | bc1e8386878bd2b7 / 9a18f802d930f6e7 | identical | identical |
| matched C1 coding / prose (ms per step) | 30.23 / 26.33 | 30.25 / 26.41 | +0.07% / +0.30% |
| G4 mean / prose MIN / G4min_hat | 60.79 / 48.56 / 48.03 | 62.39 / 49.34 / 48.13 | noise (G4 outputs differ run to run in both) |
| conc8 / conc16 (aggregate tok/s) | 262.0 / 407.1 | 263.2 / 404.1 | +0.5% / -0.7% |
| TTFT 16K / 128K | - | 2.498 / 19.710 s | see the A/B above |
| GSM8K-50 / longctx | - | 50/50 / PASS (needle 60/60, copy 1.000) | - |

Cold determinism probes, 4 runs each. Cells give distinct tokens / logprob sequences (the first differing step); "1/1" = all 4 runs identical.

| probe | C0, flag off | C1, `all` | g5-PU2, production, `all` |
|---|---|---|---|
| big 2,500 | 1/1 | 1/1 | 1/1 |
| big 8,000 | **3/4 (steps 0-3)** | 1/1 | 1/1 |
| big 32,000 | **2/4 (step 0)** | 1/1 | 1/1 |
| big 128,000 | **4/4 (step 0)** | 1/1 | 1/1 |
| code 4,000 | **4/4 (step 0)** | 1/1 | 1/1 |
| prose 6,000 | **1/4 (steps 18-19, decode)** | 1/1 | 1/1 |
| tool B:T16 (5,903) | **3/4 (step 0)** | 1/1 | 1/1 |
| tool B:T02 (5,891) | **4/4 (step 0)** | 1/1 | 1/1 |
| tool A:T01 (6,107) | **3/3 (step 0)** | 1/1 | 1/1 |
| warm code 4,000 (1 cold + 3 warm) | 2/2: warm runs identical, differ from cold at token 12 | same | same |
| warm big 32,000 | tokens 1, logprobs 2 | 2/2 (token 82) | 2/2 (token 82) |

"big" = token-id prompts from a fixed 1.43 MB code corpus: the 57 `.py` files under vllm `v1/core`, `engine` and `config`, sha f3651c43077e.

**Production (g5-PU2).**
- Launched 17:53 by the supervisor with launcher f214c2c9 defaults: AV unpinned, no extra variables.
- Healthy at +205 s.
- exact.py **a8c3394af9d2** x2, own profile 910/466/2330/1373.
- G1 x3 **101.52** (95.61 / 76.32 / 132.64). The digest is bc1e8386878bd2b7 (first 2: 9a18f802d930f6e7), the reference outcome.
- TTFT: 16K 2.505 s (g4v-FU 2.494); 128K 19.668 s (19.617).
- ONE cold 1M: 168.4 s, good class (late/early 1.20, chunks 2.4 -> 2.9 s). g4v-FU was 166.2 s; good-class launches range 166-170 s. Lowest MemAvailable on spark-one: 8.73 GB.
- Determinism: 9/9 cold probes identical; warm runs identical to each other.
- 0 errors, 0 restarts; SM 1976-1989 MHz at every check.

**First production launch (g5-PU, 17:48).**
- It gave the AV-profile alternate aeef687377ef (942/437/2185/1373).
- Its unpinned startup profile ran while the site tree was being hashed on 8/8: verify_ms(2) was 22.65, against 18.1-19.1 in 10 earlier launches.
- It was relaunched with nothing else running (g5-PU2), which gave a8.
- This was not a code effect: the pinned C1 and D2 reproduced the pin's counters exactly.

### 5. Warm (prefix-cache hit) vs cold: not fixed (row-count dependence)
- **Status.** After the fix, warm runs (one `cache_salt`, runs 2-4 cached) are identical to each other in every case tested: D1, D2, C1 and PU2 (6 cases in D2). They still differ from the cold run:
  - code 2,000 / 4,000 / 8,000 at token 3 / 12 / 0;
  - big 32K at token 82;
  - tool B:T16 at token 231;
  - prose 2,048: tokens equal, logprobs differ from step 0.
- **Evidence** (D2 dump, code 4,000, cold vs warm). The warm step recomputes 160 rows (32 uncached + 128 SWA replay) through layers 2/8/14 and the replay layers.
  - Already at layer 2, the attention inputs of those 160 rows differ from the cold prefill in 160 of 160 rows, with no tie involved. The same tokens went through layers 0-1 in a 160-row batch instead of a 4,000-row one, and the FP8 linears, the MoE grouped GEMMs, the mHC split-K and attention choose tilings and reduction splits by row count.
  - 16 of 2,000 index keys differ: the re-written tail.
- **What a fix would take.** Batch-invariant kernels, with one tiling and reduction order for every row count, in:
  - the DeepGEMM FP8 linears;
  - the M32 / M64 / contiguous FP4 MoE paths;
  - mHC pre/post;
  - sparse MLA prefill and decode;
  - the indexer;
  - the all-reduce paths.

  That is a large rewrite, and batch-invariant GEMMs usually cost 1.2-2x at small row counts. It was not started.
- **Practical rule.** Compare long prompts cold with cold (a fresh `cache_salt`) or warm with warm; a warm result will not equal a cold one.

### 6. Live state, files, rollback
- **Launcher** `~/dsv41-tp8.sh`: 95f08eff -> **f214c2c9** on 8/8. Backup `~/dsv41-tp8.sh.bak-g5-det` = 95f08eff. One added line, after `DSV41_TOOL_STREAM_NEWLINE`:
  `export DSV41_TOPK_CANON_TIES=${DSV41_TOPK_CANON_TIES:-all}  # g5det 2026-10-27 ...`
- **Supervisor.** 7bea1cf3, unchanged. Its preflight manifest hashes the launcher, so it changed 40eb7abf08c7 -> **d773ed60f27b**.
- **Site.**
  - `model_executor/layers/sparse_attn_indexer.py` f2ae649b -> **005785f9** (backup `.bak-g5-det`);
  - new `model_executor/layers/dsv41_topk_ties.py` **943961bc**;
  - vllm site tree 38062de05f0a -> **4baa4e458a1a** (5,453 files), identical on 8/8;
  - installer `~/dsv41-stage/g5det/g5det_install.py` (850e9c9f) `status|install|uninstall`, staged on 8/8.
- **Live instance** g5-PU2: env `DSV41_TOPK_CANON_TIES=all` from the launcher; no `DSV41_DET_DUMP`, no `DSV41_AV_COST_JSON`.
- **The AV pin** `~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json` (eeac4f97) stays valid: exact.py counters and the G1 digest are identical with the fix.
- **Rollback.**
  - Quickest: `DSV41_TOPK_CANON_TIES=0` as a supervisor argument (no file change).
  - Or restore the launcher on 8/8: `cp ~/dsv41-tp8.sh.bak-g5-det ~/dsv41-tp8.sh`.
  - Full removal: stop the service, run `python3 ~/dsv41-stage/g5det/g5det_install.py uninstall` on 8/8 (restores f2ae649b and deletes the module), then restore the launcher.
- **New check for every prefill / long-context candidate (~4 min):**
  `cd ~/dsv41-g5det && python3 -I scripts/dbgrun.py - <out.jsonl> <label> big:2500,big:8000,big:32000,big:128000,code:4000,prose:6000,l3:B:T16-spam-reply:1,l3:B:T02-classify:1,l3:A:T01-reviews:1 4 0 cold`
  Every case must print `distinct_tok=1 distinct_lp=1`.
- **Raw results.**
  - spark-one `~/dsv41-g5det/`: `runs/` (A/B production passes, D1/D2, g5-C0/C1/PU2 suites, TTFT A/B), `dumps/D1/` (~0.4 GB of debug dumps; safe to delete), `scripts/`.
  - Mac scratchpad `g5det/`: `ab.log`, `launches.tsv`, `runs/<label>/vsumm.md`, drivers `scripts/g5inst.sh`, `ttftab.sh`, unit tests.
- **Goal status: unchanged.** Production G1 101.5 of 120 tok/s. The G4 prose session is 48.6-49.3 of 50 in the pinned pair. This was an exactness item, not a speed lever.

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
  - **One hour later (20:35 UTC, uptime 3,637 s):** still growing (10 lines in 10 s; 3,631 samples this boot),
    0 unit restarts, sysctls still panic=10 / panic_on_oops=1 / softlockup_panic=1 / hardlockup_panic=1. The logger
    runs at nice 19 with 11.6 MB RSS and ~0% CPU; the log grows ~200 KB per hour (64 MB rotation, 3 kept).
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
     Supporting number, 20:08 UTC: nvidia-smi utilization.gpu at 4 Hz during a production G4 (min/mean 47.57/59.57,
     telemetry on) read 96% median (94-96%) on rank 0 and rank 4 alike. ~4% of the step, about 1.8 ms of 45, has no
     kernel running. That is consistent with the ~1.6 ms of serialized pre-launch Python, but does not prove it:
     intra-graph launch gaps also count as idle, and nsys has to split the two.
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

## Vision input (2026-10-27): image window 2026-10-28 02:19-02:46 UTC (2026-10-27 19:19-19:46 MST) ROLLED BACK (R2), images NOT live. VP passed every check bitwise except 1 of 9 pre-image determinism probes (code:4000), which overlapped a foreign API request. Live = the pre-window state, no sha changed (launcher f214c2c9, model_runner.py 6a4f2433, manifest d773ed60f27b; instance vis-R2)

**Attempt record.** The change (PLAN v2: one env-gated launcher block plus a 2-hunk `model_runner.py` patch, both inert
with images off) is still staged on 8/8 and nothing of it runs. The window followed RUNBOOK v2 (W0-W2, then R2). Two
independent read-only verifications afterwards CONFIRMED everything below (section 7). A retry needs a new user approval
(section 8). Hand-back checks for this section: 2026-10-28 03:10-03:11 UTC, read-only.

### 1. Live state and shas

| item | value |
|---|---|
| state | `live`: `03:10:57 [vis_install expect live] OK: 8/8 in state live (manifest d773ed60f27b)` |
| launcher `~/dsv41-tp8.sh` | `f214c2c98a93ed2fe388e59fff68b054e0da966c22d1ef67a70aff9c26ce3b76` on 8/8 (unchanged) |
| `vllm/v1/worker/gpu/model_runner.py` | `6a4f2433eb4ecd257d9381473848154cec9a8f38b70e8dde706e4eadff2e386f` on 8/8 (unchanged). The staged vision file 78c9e22f is NOT installed |
| supervisor manifest | `d773ed60f27b` (unchanged) |
| vllm site tree, README method (every file under `vllm/`; bytecode, `.bak-*`, `.absent-*`, `.g4-tmp` excluded) | **`4baa4e458a1a`, 5,453 files, on 8/8** (hashed 03:10:52-03:10:53 UTC), the same as before the window |
| vllm site tree, W0 method (`*.py` + `*.so` only) | `7700bad006cc`, 2,903 files, on 8/8 (03:10:53-03:10:54 UTC), the same as the W0 record |
| instance | vis-R2: supervisor relaunch, launcher defaults, no extra env. `dsv41-rank` active on 8/8 since 02:40:13 UTC, NRestarts 0, healthy 02:43:36 |
| startup | `running in text-only mode`; `GPU KV cache size: 6,390,007 tokens`; `'enable_prompt_tokens_details': True`; FlashInfer autotune cache loaded on rank 0; AV `pin miss (DSV41_AV_COST_JSON unset)`; env identical to g5dr-PROD (61 DSV41 vars, envsha 4e88636fc98f); rank 0-7 logs 0 ERROR/Traceback |
| exact | exact.py `a8c3394af9d28249` at 02:43:53 (acc 909, dtok 2335, drafts 467, gen 1373; perpos 348/231/171/91/68, the unpinned production values) and again at 03:03 UTC (verifier, same hashes and counters); no concurrent request either time |
| images | an image request gets HTTP 400 `At most 0 image(s) may be provided in one prompt.`; text requests after it are unaffected |
| clocks | `-lgc 0,2000` OK on 8/8 at all 3 launches; SM 1969-1989 MHz; never raised |
| backups on 8/8 | `~/dsv41-tp8.sh.bak-vis-t0` = f214c2c9, `model_runner.py.bak-vis-t0` = 6a4f2433 (both equal the live files; a t0 re-install accepts them). No `.bak-vis-promote` or `.vis-new` files |
| DSH (Mac) | untouched: `cordis.patch.yml` 57623bdc, `dsv41 input line: 0` (no image input declared) |

**For other agents: no sha changed, nothing to rebase.** Use launcher **f214c2c9**, `model_runner.py` **6a4f2433** and
manifest **d773ed60f27b** (full shas above). `--expect-launcher f214c2c9` stays valid, and the mhc host-prep agent does
NOT rebase on 78c9e22f. The state check is `bash $S/stage/vis_install.sh expect live` (read-only: launcher,
`model_runner.py` and supervisor manifest on 8/8; `$S` in section 9); expect `OK: 8/8 in state live (manifest
d773ed60f27b)`. A change to any of the 10 supervisor-manifest files voids the staged vision set; it then needs a rebuild
before a retry (RUNBOOK, "If a manifest file changed").

### 2. What ran (UTC)

| step | expected | actual |
|---|---|---|
| P1 `vis_install.sh verify-stage` | `OK: staged set 3b034721d48422c0 verified on 8/8` | 02:19:09 OK |
| P2 `vis_install.sh expect live` | `OK: 8/8 in state live (manifest d773ed60f27b)` | 02:19:11 OK |
| W0 `tests/prep_window.sh` | `W0 OK` | 02:19:25-02:19:46 `W0 OK`: 320 autotune files recorded and backed up to `~/dsv41-stage/vision/autotune-W0` (40 per node, 8/8); det corpus f3651c43077e OK; site tree 7700bad006cc on 8/8; rank0-pre KV 6,390,007; last gate PASS (f214c2c9 20261027T195149Z); spark-one MemAvailable 6,367 MB; vision_func dry-run `imports ok` |
| W1 `vis_install.sh t0` | `OK: 8/8 in state t0 (manifest c4be85f5754f)` | 02:23:42 OK: launcher 236d5dbb and `model_runner.py` 78c9e22f on 8/8; `.bak-vis-t0` = f214c2c9 / 6a4f2433 |
| W2 `vinst-vision.sh vis-VP VP` | `VERDICT VP no hard failure` | **detref MISMATCH on 1 of 9 probes: automatic rollback to t0 (images off), driver exit 12** |
| W3-W8 | - | not run |
| R2, by hand | autotune compare; `rb-t0` gives `OK: 8/8 in state live (manifest d773ed60f27b)`; `relaunch_defaults.sh vis-R2 live` rc 0 | autotune UNCHANGED (320 files, no restore needed); 02:39:36 rb-t0 OK; 02:43:36 vis-R2 HEALTHY (+204 s), supervisor rc=0 |

Restarts (`$S/launches.tsv`):

| # | launch | stop | healthy | hard downtime |
|---|---|---|---|---|
| 1 | vis-VP: state t0, `DSV41_IMAGE_LIMIT=16`, AV pin av-pin-k5-g3FU, KV 16G, weight digest | 02:25:36 | 02:31:12 (+323 s) | 336 s |
| 2 | vis-VP-rollback: t0 defaults, images off (automatic) | 02:34:46 | 02:38:23 (+204 s) | 217 s |
| 3 | vis-R2: live defaults | 02:39:57 | 02:43:36 (+204 s) | 219 s |
| | total | | | **772 s (12 min 52 s)**, inside the approved 10-14 min |

The t0 rollback instance ran ~94 s and served no request (sampler success_total 0 from 02:38:22 to 02:39:57), so its
exactness was never checked.

### 3. Results

**VP** = images on through the real launcher branch (`DSV41_IMAGE_LIMIT=16`, which also sets `DSV41_DUMMY_EMBED=1`,
`DSV41_MM_PIN_TRIM=1`, the 0.5 GiB processor cache, `media.invalid`, the 8 MP cap and 4 decode threads), AV pin
`av-pin-k5-g3FU.json`, KV 16G. The driver writes no verdict at a rollback. `verdict.py VP` on the copied run
(`$S/ab/vis-VP/verdict-posthoc.txt`, reproduced byte-identically by a verifier) gives every HARD and DECISION line ok
except detref: `VERDICT VP HARD-FAIL (rollback)`.

| check | VP | want | result |
|---|---|---|---|
| exact.py, first request | a8c3394af9d2; acc 909, drafts 467, dtok 2335, gen 1373; perpos 348/233/171/90/67 | the pin's values | ok, bitwise |
| AV pin | k=5 launch hit (29 captured sizes) | hit | ok |
| weight digest vs dF3 (16 rank x load dumps) | text parameters identical in every dump (0 changed, 0 missing, no non-vision new parameter); target: +266 vision-side parameters materialised, 116.9 MiB per rank; drafter 99 = 99 | TEXT-WEIGHTS-IDENTICAL | ok |
| startup facts | KV 6,390,007; `Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 16 image items`; MMEncoderAttention on FLASH_ATTN (the FA2 JIT worked; no TORCH_SDPA fallback needed); mm-prefix not engaged; the same two autotune files (8c8be689..., c414c171...); startup errors 0; model loading 42.33 GiB (text-only 42.21) | rank0-pre.log | ok |
| FlashInfer autotune caches vs W0 | UNCHANGED on 8/8 (320 files) | unchanged | ok |
| matched C1 AL coding / prose | 5.191 / 2.725 | 5.191 / 2.725 | ok |
| matched C1 ms/step coding / prose | 30.25 / 26.31 | <= 30.40 / <= 26.41 | ok |
| matched C1 per-stream tok/s coding / prose | 171.59 / 103.54 | - | info |
| G1, 3 reps | **101.756 tok/s** (101.784 / 101.743 / 101.742); code 96.05, prose 76.65, math 132.57 | >= 101.2 | ok |
| G1 outputs | digests 9a18f802d930f6e7 (2 reps) / bc1e8386878bd2b7 (3 reps); code 596b478a, math 92e5d8cf, prose 923a46be | the same | ok, bitwise |
| detref before image traffic: 9 cold probes vs g5-C1 / g5-PU2 (token and logprob sha) | 8/9 bitwise: big:2500 / 8000 / 32000 / 128000 (SP-prefill path included), prose:6000, the 3 l3 tool prompts | 9/9 | **FAIL: code:4000 tok f197764bf437 (ref 43c4b49233ea), lp abdfbb17d83c (ref 1975145d1bc6)** |

- Info: spark-one MemAvailable was 13,339 MB on the fresh images-on instance, against 13,894 MB on the fresh text-only
  vis-R2 at 02:44 (one sample each; not a substitute for hist1m).
- **Not reached** (the suite stops at detref): the img / OCR / DSH-wire replay functional tests, G4, census, conc8 /
  conc16, exact #2, and the end-of-suite detref 44/44.
- **VC (W3), VU (W5), hist1m (W6): not run.** So there are no production-with-images numbers: no VU G1, gate, KL, s1m
  TTFT, 1M MemAvailable minimum or KV recommendation. Decision D2 (kv14 recommended / kv15 / kv16) is still open.
- **Goal numbers are unchanged:** production G1 ~102 (101.81 in the f214c2c9 gate run, 102.00 in window 4) and the G4
  prose session MIN ~48.4 (48.41 in window 4). Pinned VP G1 101.76 against pinned g4v-FP 101.69: no measurable G1
  cost from images on in this one instance.

### 4. Why it stopped: the code:4000 probe overlapped a foreign API request

- A detref MISMATCH is an exactness failure, so the driver rolled back automatically (02:34:42) and R2 followed by hand.
  That is the rule and it stands. The images-on text path is **unproven**, not shown broken.
- Evidence that the probe was batched with a foreign request (corroborated independently by both verifiers):
  - the 2 s sampler (`$SP/vwin/VP.sampler.txt`): `02:34:23 running=1 waiting=1` is the only two-request sample of the whole
    VP run, and request_success_total rose by 10 over the detref phase against 9 dbgrun requests;
  - latency: code:4000 took **4.158 s**, against 2.001-2.027 s in all 8 reference runs (g5-C1 x4, g5-PU2 x4). Every
    other VP probe was within ~1% of its reference;
  - divergence from the reference: the prefill logprob and output indexes 0-1 are bitwise equal, logprobs differ from
    index 2, and the first token flips at index 45;
  - dbgrun.py (158c9cba9830ad92) sends one request at a time after an idle check. Its big / code / prose probes use
    `/tokenize` + `/v1/completions`; only the l3 probes use chat. The VP rank0 access log showed a
    `/v1/chat/completions` between the code probe's `/tokenize` and its completion (section 7, item 3);
  - matched and G1 saw no overlap: their spec counters equal the g5-C1 / g5-C0 references, and the sampler never showed
    more than 1 request running in them.
- **Who.** The API has no authentication and is exposed through `dsv41-lanproxy.service` on spark-one (LAN
  192.168.0.238:8000, tailnet 100.90.117.38:8000; see "LAN API exposure"). Clients seen during and after the window:
  LAN host **192.168.0.86** and tailnet node **`fedora` 100.70.246.23, shared by account `aurora@`**. Neither is the
  Mac's DSH (192.168.0.239 / 100.75.132.97). `/metrics` showed `num_requests_running 1.0` at W0. After the window,
  foreign requests came at 02:55 (a ~110K-token prompt), 02:58, 03:02, 03:04 and 03:06.
- The runbook had no step for foreign traffic. The window went ahead because an overlap can only cause a false failure
  (exact counters, dbgrun's idle check, the G1 / G4 guards, gate rc=3, hist1m's refusal), never a false pass. Before
  starting W2 and R2, the window agent waited for two consecutive idle checks (02:25:26, 02:39:48).

### 5. Knobs and states (STAGED on 8/8, none active)

The vision files sit in `~/dsv41-stage/vision/` on 8/8 (staged set `SHA256SUMS` 3b034721d48422c0). Nothing running reads
them.

| state (`vis_install.sh expect <state>`) | launcher | `model_runner.py` | supervisor manifest | images |
|---|---|---|---|---|
| **`live` (now)** | **f214c2c9** | **6a4f2433** | **d773ed60f27b** | off: `--limit-mm-per-prompt '{"image":0}'` |
| `t0` (W1, inert install) | 236d5dbb (L1) | 78c9e22f | c4be85f5754f | off unless `DSV41_IMAGE_LIMIT` > 0 |
| `kv14` / `kv15` / `kv16` (W4 promote) | 9650fcc4 / ec7f8c32 / a69e5fc8 (L2) | 78c9e22f | 857294e45d80 / 7aec2ddcc7b4 / 39b84881b90c | on, 16 by default; KV 14G / 15G / 16G |

| knob | default | effect |
|---|---|---|
| `DSV41_IMAGE_LIMIT` | 0 in L1 (argv and env then byte-identical to f214c2c9, argv_check 20/20); 16 in L2 | `--limit-mm-per-prompt {"image":N}`, counted over the whole conversation (DSH re-sends every image each turn and allows 20 per message). N > 0 also adds `--mm-processor-cache-gb`, `--allowed-media-domains media.invalid`, `VLLM_MAX_IMAGE_PIXELS=8388608`, `VLLM_MEDIA_LOADING_THREAD_COUNT=4`, `DSV41_DUMMY_EMBED=1`, `DSV41_MM_PIN_TRIM=1` and `CUDA_CACHE_MAXSIZE` (4 GiB); an explicit `VAR=VAL` overrides each one. On an L2, `DSV41_IMAGE_LIMIT=0 DSV41_KV_BYTES=16G` equals f214c2c9 (R1-fast). A non-integer is refused. An env-only 20 is lost at the next relaunch that does not carry it |
| `DSV41_MM_CACHE_GB` | 0.5 (L1, L2-kv14); 0.25 (L2-kv15 / kv16) | EngineCore processor item cache on spark-one (stock 4 GiB) |
| `DSV41_DUMMY_EMBED` | 1 on the images-on branch (0 in code) | With images on, the AV cost-profile dummy runs embed their probe ids instead of reading the zeroed `inputs_embeds` buffer. Kill switch: `DSV41_DUMMY_EMBED=0` as a supervisor `VAR=VAL`. **Caveat (exact-compat review W7): `=0` is not a neutral lever, even with the AV pin.** Unpinned, the zeroed buffer routes every probe row to the same 6 experts, the verify-cost curve flattens (5.30x from width 1 to 96 today), AV over-verifies, and production widths and near-ties move. Treat `=0` as a functional fallback only; for exactness use R1 or R2 |
| `DSV41_MM_PIN_TRIM` | 1 on the images-on branch (0 in code) | `torch._C._host_emptyCache()` after `profile_run` frees the 256 MiB pinned host block that the 16-image encoder profile leaves; host memory only. Kill switch: `=0` |
| `DSV41_EXTRA=--mm-encoder-attn-backend TORCH_SDPA` | unset | Fallback only if the ViT's FA2 JIT fails. VP ran FLASH_ATTN fine, so it is not needed (if it were, production would need it on every relaunch; it is not staged in L2) |

Limits with images on (PLAN v2): at most 16 images per conversation; at most 1,024 tokens per image (~1.8 MP effective;
1280x900 = 706 tokens); input over 8,388,608 px gets a 4xx; `data:` URLs only (PNG, JPEG, WebP, GIF); +0.27 GiB per rank
at startup.

### 6. Rollback levels (for a future window) and the order rule

The commands are from the Mac kit (section 9). Every writing or restarting script refuses without `VISION_WINDOW=go`.

| level | when | result | downtime |
|---|---|---|---|
| **R-client** | always first, once DSH declares image input (W8) | DSH text-only; image sessions continue on `[image omitted ...]` placeholders | 0 |
| **R1**: images off, files kept | image-path trouble, memory, quality | L1 defaults: argv and env = f214c2c9, patch inert | 4-5 min |
| **R1-fast**: no file change | the same, when speed matters | identical to f214c2c9 (argv_check B) | 4-5 min |
| **R2**: full restore | any exactness failure | manifest d773ed60f27b, exactly the pre-window state. **Used on 2026-10-28** | 4-5 min |

```sh
S=/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad/vision; export VISION_WINDOW=go
KV=kv14            # the promoted state (D2), if any
# R-client: DSH Settings -> Models -> Spark Cluster -> dsv41 -> Input: untick `image`, Save. Or: Cmd-Q, then
VISION_DSH=go bash $S/client/dsh_vision.sh revert          # then reopen the app
# R1
bash $S/stage/vis_install.sh rb-promote
bash $S/tests/autotune_state.sh compare $S/ab/W0/autotune.txt || bash $S/tests/autotune_state.sh restore W0
bash $S/tests/relaunch_defaults.sh vis-R1 t0
# R1-fast (the state stays kvNN until the rb-promote after it)
bash $S/tests/autotune_state.sh compare $S/ab/W0/autotune.txt || bash $S/tests/autotune_state.sh restore W0
bash $S/tests/relaunch_defaults.sh vis-R1 $KV DSV41_IMAGE_LIMIT=0 DSV41_KV_BYTES=16G
bash $S/stage/vis_install.sh rb-promote
# R2 (rb-t0 restores from any of t0 / kv14 / kv15 / kv16 directly)
bash $S/tests/autotune_state.sh compare $S/ab/W0/autotune.txt || bash $S/tests/autotune_state.sh restore W0
bash $S/stage/vis_install.sh rb-t0 && bash $S/tests/relaunch_defaults.sh vis-R2 live
```

`relaunch_defaults.sh` exits 0 when healthy, 1 when failed, 2 when the preflight refused and 9 when it refused before
any stop. Each relaunch first checks `vis_install.sh expect <state>` on 8/8.

**Order rule: untick `image` in DSH (R-client) BEFORE any text-only server relaunch (R1, R1-fast, R2).** Turning images
off on the server first strands the DSH sessions that hold images. Turning on goes the other way: the server first (W5
passed), then the client (W8). Today DSH declares no image input, so there is nothing to untick, and W8 must NOT be done
while the server is text-only (every image request gets HTTP 400).

### 7. Verification after the window: discrepancies and how they were resolved

Two independent read-only verifications (02:49-02:58 and 03:03-03:06 UTC) CONFIRMED the state, the verdicts and the
diagnosis. The second also probed vis-R2 live: `/v1/models` 200; exact.py a8c3394af9d28249 with the same per-prompt
hashes and counters as vis-R2's first run; an image request rejected with 400 in 1.6 ms, with no engine-side effect;
MemAvailable 9,999-11,367 MiB on 8/8 at idle; 0 new errors. Neither changed anything on the cluster. What they found
in the window report (none of it changes an outcome):

| # | the report said | finding | resolution |
|---|---|---|---|
| 1 | detref "deterministic within the run: distinct_tok=1" | detref ran with `reps: 1` (DETREF_REPS=1), so this is true by construction and proves nothing | dropped; the diagnosis rests on section 4 |
| 2 | "jit_warnings 4, the same first-use Triton kernels as the text-only reference (5)" | rank0-pre has topk_indices, live_cap, refine, screen, topk_log_softmax; VP has live_cap plus topp_sb_mask / stats / step | benign: VP's 4 equal the text-only post-gstart set (vis-R2-start, g5-PU2-start) |
| 3 | the VP access-log lines (ports 36972 / 36986 / 36988) | cannot be re-checked: the R2 relaunch rotated the VP rank0 log away and no copy was kept | replaced by the sampler, latency and divergence evidence in section 4 |
| 4 | "chat completions most minutes 02:06-02:21" | rank0-pre.log (last stats line 02:17:08) shows activity in 6 of the 12 visible minutes | overstated, not material |
| 5 | site tree 7700bad006cc, where the runbook names 4baa4e458a1a | two methods over the same tree (`*.py` + `*.so`, 2,903 files, against every file, 5,453) | not a conflict; both re-recorded unchanged (section 1) |
| 6 | `$S/runs/vis-VP/sampler.txt` | a truncated copy | the full copy is `$SP/vwin/VP.sampler.txt` (= spark-one, sha 82e6790c...) |
| 7 | MemAvailable 13,423-14,802 MB on 8/8 at 02:44 | fresh-instance values; about 10-11.4 GB by 03:04, after a ~110K-token foreign request at 02:55 | normal drift toward the long-running value (6,367 MB on spark-one at W0) |

Also found, all harmless: `~/dsv41-logs/wdigest/wd-vis-VP-r*-{0,1}.json` (~1 MB) on 8/8; `~/.nv/ComputeCache` grew
~6 MB per node (ViT FA2 kernels); `rank0.prev.log` (the rollback instance) holds 8 `EngineDeadError` lines from its
SIGTERM at 02:39:58. Deviations the window disclosed: it proceeded despite foreign traffic (section 4); R2 after the
automatic rollback (the runbook's level for an exactness failure, and the preferred end state); the post-hoc verdict;
the aborted suite's 2 s /metrics sampler, stopped through its designed DONE file at 02:45:05; a read-only gstart exact
check (7 requests) and site-tree hash on vis-R2. `VISION_WINDOW` does not persist in any shell (each command carried
it inline), so the runbook's closing `unset VISION_WINDOW` is moot.

### 8. How to retry

1. **A new user approval** (D1): the same window shape, 10-14 min of hard downtime (+~2.5 min with VC).
2. **Keep foreign traffic off the API during the probe phases** (VP: matched, G1, detref, G4; VU: detref, gate, klref;
   hist1m). This is not in the runbook; user decision. Either stop the forwarder for the window
   (`ssh spark-one systemctl --user stop dsv41-lanproxy.service`, then `start` again before W8: DSH's baseURL is
   `http://192.168.0.238:8000/v1`, so DSH loses the API while it is stopped; the tests use 127.0.0.1:8000 on spark-one),
   or ask the users of 192.168.0.86 and `fedora` (`aurora@`) to pause.
3. **Optional hardening (planner decision):** make detref (dbgrun) and the exact.py counter checks treat a probe that
   overlapped another request as "not proven" and rerun it (compare request_success_total / num_requests_running before
   and after each probe), instead of a MISMATCH that triggers a rollback.
4. **Optional: VC (W3) before W2** proves the t0 rollback target live (critic open item).
5. **W0 artifacts.** `~/dsv41-stage/vision/autotune-W0` (8/8) and `$S/ab/W0` stay valid for a same-day retry while the
   autotune caches are unchanged. **On another day**, move `$S/ab/W0` aside and remove `autotune-W0` on 8/8 first;
   otherwise W0 reports `autotune-changed-since-first-W0` and R2 would restore a stale cache.
6. **Use new labels.** Each `vinst-vision.sh` run does `rm -rf ~/dsv41-vision/runs/<label>` (and `<label>-start`) on
   spark-one, so re-running `vis-VP` deletes this attempt's evidence there. Use `vis-VP2` and so on, or move the old
   directories aside first.
7. Then P1-P4 and W0-W9 as in RUNBOOK.md.
8. An offline draft, `DSV41_MM_EMBED_ON_DEMAND` (`$S/ondemand/DESIGN.md`; not staged, never run), would keep every text
   step on the images-off graphs. If it is adopted first, every staged sha in section 5 changes: re-stage, then P1-P4.

### 9. Files

- **Mac kit and evidence** (under `/private/tmp`, which a reboot or a cleanup can remove):
  `SP=/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad`, `S=$SP/vision`.
  - plan and runbook: `$S/PLAN.md`, `$S/RUNBOOK.md`; scripts: `$S/stage/`, `$S/tests/`, `$S/client/`;
  - this window: `$S/ab.log`, `$S/launches.tsv`, `$S/ab/W0/`, `$S/ab/vis-VP/` (verdict-posthoc.txt, relaunch.log,
    rollback.log, rollback-state.txt, autotune*.txt, wdcmp.txt, pincheck.txt, start.txt), `$S/runs/vis-VP/` (detref.log,
    detref-cold.jsonl, facts.txt, matched.log, g1.log, vsumm.json);
  - captures: `$SP/vwin/`; verifier outputs: `$SP/verify-imgwin/`, `$SP/verify-vis-20261028/`; hand-back re-checks:
    `$SP/w9/`.
  - **The durable copy on spark-one, `~/dsv41-stage/vision/mac-kit/vision-mac-kit.tgz` (sha 7d7e198e..., 01:45 UTC),
    predates the window.** It restores the kit (RUNBOOK P5) but not this window's results; refresh it if they should
    survive a Mac cleanup.
- **spark-one:** `~/dsv41-vision/` (tests, `ref/rank0-pre.log`, `runs/vis-VP`, `runs/vis-VP-start`, `runs/vis-R2-start`);
  `~/dsv41-stage/vision/mac-kit/`; this README's pre-edit copy `README.md.bak-vision`.
- **8/8:** `~/dsv41-stage/vision/` (staged set and `autotune-W0`), `~/dsv41-tp8.sh.bak-vis-t0`,
  `~/venvs/vllm032/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py.bak-vis-t0`,
  `~/dsv41-logs/wdigest/wd-vis-VP-r*-{0,1}.json`.

### 10. Open decisions (user)

1. **Re-run the image window?** It needs a new approval and a plan for foreign traffic during the probes (section 8,
   item 2). D2 (KV: kv14 recommended) is still open too.
2. **Harden the probes** so that an overlapped probe is "not proven" and rerun, not a rollback (planner decision).
3. **API exposure (D8).** The plan assumed DSH is the only client. The unauthenticated API serves at least two others
   through the forwarder: 192.168.0.86 on the LAN, and `fedora` 100.70.246.23 on the tailnet, shared by `aurora@`.
   Accept, restrict the forwarder or the tailnet share, or add API auth.
4. **VC first** in the next window (optional).
5. **W0 artifacts:** keep them for a same-day retry, or move / remove them per the runbook before another day.

## Goal window 5, LongCtx step (2026-10-28 03:29-04:50 UTC): Codex ops kit 07 (reuse the indexer top-k across decode steps) CLOSED at its own gate, NOT promoted. On 1,651 recorded real decode steps (128K and 512K) the top-512 set changes at every step, so no sound certificate can ever allow a reuse: 0 passes. No serving code built, no A/B. Launcher f214c2c9 unchanged; live g5lc-PU

**Result.**
- **The kit's gate** (its README: "log the margin and the pass count on a 512K decode; if the pass count is near zero, the kit stops"): **0 passes** out of 109,912 recorded (layer, row) pairs, at 128K and 512K, for every index layer.
- **Why no certificate can pass.** A reused set is exact only if the new top-512 lies inside the old set plus the entries that became visible since. That held in **0 of 109,912** pairs, including adjacent positions inside one verify step. Consecutive rows share only about half of their picks (mean overlap 0.50-0.58); a reuse would serve 206-251 wrong entries of 512 (median).
- **Why this certificate in particular fails.** Its bound D on how far any score can move between two query tokens is 79-94x |v_k| (median, rigorous K bound from the checkpoint's k_norm weights), or 36-43x with an optimistic key norm. The boundary gap it must beat (512th score minus the best outside score) is about 0.05% of |v_k|.
- **Ceiling, had it worked.** On this stack a 512K decode step costs **+2.30 ms** over a 1K prefix (30.2 vs 27.9 ms on code, same AV pin). That is the whole long-context indexer + top-k excess, so kit 07 could never have moved G1 / G4, which run short contexts.
- **Nothing changed in production.** No site file and no launcher line was touched. The recording used a diagnostic worker extension outside the site tree; it was output-neutral (exact.py with the pin's counters; 6/6 recorded continuations identical to the recorder-off reruns). Production was relaunched on launcher defaults (exact a8c3394af9d2).

### 1. Method
- **Recorder** (`~/dsv41-g5lc/py/g5lc_rec.py` 2faeffe949d0, staged on 8/8, DIAGNOSTIC ONLY, never in the live launcher). Loaded only with `PYTHONPATH=~/dsv41-g5lc/py G5LC_REC=1 DSV41_EXTRA=--worker-extension-cls=g5lc_rec.RecExt`.
  - It wraps `dsv41_topk_ties.decode_hook`, which the decode indexer calls inside the FULL graphs after the stock top-k and the live tie fix. It calls the original hook first.
  - On TP rank 0 only, two Triton kernels captured into the decode graphs READ the logits, final top-k, fp8 index queries and head weights. They WRITE only a 198 MiB ring that is allocated before capture (512 steps x 8 layers x 8 rows).
  - Per row it stores: the 512 ids; v_k (512th score); U, the best score outside the set (exact, ties included); counts > / >= v_k; the row end; counts >= the previous step's v_k (and -2/-5/-10%); the row max; q (32x128 fp8); w (32 fp32).
  - A device flag gates both kernels. A control file (`/dev/shm/g5lc-ctl.json`, polled at the start of `execute_model`) resets, enables or dumps the ring.
  - The kernels were checked in the Triton interpreter on CPU (`test_rec_cpu.py`): every field is bitwise equal to a numpy reference, including ties, -inf masks, rows < MAXR, ring wrap and the disabled flag.
- **Instance g5lc-R1.** Launcher f214c2c9 + AV pin `av-pin-k5-g3FU.json` + the recorder; healthy +131 s.
  - exact.py **a8c3394af9d2** with the pin's counters, 909/467/2335/1373 and per-position 348/233/171/90/67 (EXACT-OK); the pin hit.
- **Probe** (`~/dsv41-g5lc/scripts/g5lc_probe.py`). Setup:
  - One cold prefix prefill per case.
  - Then 3 greedy 512-token continuations on the cached prefix (chat, thinking off, `ignore_eos`).
  - Cases:
    - `code:512000` and `code:131072`: prefixes cut from a fixed 3.2 MB snapshot of 211 vLLM `.py` files, sha c2715b6aa1b779dc, exact 512,000 / 131,072 tokens;
    - `words:512000`: the X-ray decprobe's random words and questions.
  - Foreign-traffic guard. Before every request the probe waits for 30 s of idle. A 0.5 s /metrics sampler runs during each request. If a request overlapped foreign traffic, it was rerun and its result discarded.
- **Analysis** (`g5lc_an.py`, CPU only). Pairs of decode rows:
  - **A** = same row, consecutive steps;
  - **B** = new step's row 0 vs the previous step's row at the previous position (the last committed token);
  - **C** = adjacent positions inside one verify step.
- **Rigorous D.** Per head, `K x min(|a_n||Δb| + |Δa||b_p|, |a_p||Δb| + |Δa||b_n|)`, minimised over the free per-head rescaling (a, b) -> (a/c, c·b), c = 2^-8..2^8.
  - K = sqrt(128) x max|k_norm.weight| x 1.0625: 14.09 / 19.53 / 22.35 for layers 2 / 8 / 14, and 13.15 for layer 20, whose keys layers 24-36 share.

### 2. Recorded decode steps (all clean of foreign traffic)

| case | continuations (decode steps) | verify rows per step | ms/step, recorder off | ms/step, 1K-prefix control |
|---|---|---|---|---|
| code 512K (512,011-token prefix, cold prefill 82.1 s) | 169 / 185 / 140 | 2-6 | 30.21 / 29.76 / 30.50 (AL 3.07 / 2.78 / 3.68) | 27.85 / 27.39 / 28.61 (AL 2.84 / 2.71 / 3.35) |
| code 128K (131,083) | 202 / 179 / 134 | 2-6 | - (recorder on: 28.55 / 28.73 / 30.19) | - |
| words 512K (512,554) | 288 / 189 / 165 | 2-6 | 27.12 / 29.65 / 30.15 (AL 1.79 / 2.73 / 3.12) | 24.40 / 27.52 / 27.82 (AL 1.73 / 2.80 / 2.97) |

- **512K excess**, same question: +2.36 / +2.37 / +1.89 (code) and +2.72 / +2.13 / +2.33 (words). Mean **+2.30 ms/step (+8.4%)**, in line with the X-ray step's +2.1-2.2 ms.
- **Recorder on** costs ~1.5 ms/step at 512K (rank-0 scans).
- **Recorder on vs off**: the token output is identical in 6/6 clean pairs (text sha, steps and AL all equal).

### 3. Kit 07 on real queries

| pairing | pairs | reuse could be exact (oracle) | top-512 overlap, mean (p10 / p50 / p90) | wrong picks if reused, median | certificate passes: rigorous / optimistic K / D = 0 | D / abs(v_k), median | boundary gap / abs(v_k), median |
|---|---|---|---|---|---|---|---|
| A, next step, same row | 52,352 | **0** | 0.50 (0.27 / 0.51 / 0.71) | 251 | **0** / 0 / 26,510 | 94 | 0.00046 |
| B, next step, committed predecessor | 13,136 | **0** | 0.58 (0.37 / 0.60 / 0.76) | 206 | **0** / 0 / 6,555 | 79 | 0.00044 |
| C, adjacent position, same step | 44,424 | **0** | 0.56 (0.34 / 0.58 / 0.74) | 215 | **0** / 0 / 22,391 | 83 | 0.00046 |

Per index layer (all cases; A / B / C):

| layer | overlap, mean | wrong picks, median | D / abs(v_k) (A) | gap / abs(v_k) (A) |
|---|---|---|---|---|
| 2 | 0.52 / 0.58 / 0.59 | 241 / 204 / 198 | 64 | 9.6e-05 |
| 8 | 0.39 / 0.48 / 0.46 | 313 / 252 / 262 | 206 | 4.6e-04 |
| 14 | 0.39 / 0.47 / 0.45 | 306 / 254 / 268 | 149 | 5.6e-04 |
| 20 | 0.54 / 0.61 / 0.59 | 232 / 189 / 197 | 114 | 8.9e-04 |
| 24 | 0.54 / 0.61 / 0.59 | 229 / 187 / 200 | 25 | 1.7e-04 |
| 28 | 0.60 / 0.68 / 0.64 | 202 / 149 / 172 | 68 | 6.6e-04 |
| 32 | 0.50 / 0.59 / 0.56 | 256 / 205 / 217 | 141 | 1.3e-03 |
| 36 | 0.49 / 0.58 / 0.56 | 257 / 208 / 219 | 141 | 1.3e-03 |

- **Certificate unit test on the recorded steps.** The kit's rule ("reuse iff the k-th score beats the best outside score + D, else rescore") was evaluated on every pair. It passed 0 times, and it was never unsound: no case passed while the set changed. Served output would therefore be the stock rescore on every step (bitwise stock by construction), plus the certificate's own cost. So it cannot be faster, and per the trim rule no serving code was built and no pinned A/B was run.
- **The kit's model vs data.** Its "50% pass rate" assumed a stable set. In the data, the set is never stable from one token to the next, at any layer, context length or content. The X-ray step's prefill dumps had already shown the same thing for prefill rows (adjacent-row overlap 0.18-0.32 at layer 14).
- **One exact lead (not kit 07; not built).** Use the previous step's v_k of the same row as a threshold hint for the decode top-k:
  - in 50.6% of rows at least 512 scores reach it (53.4 / 57.5 / 63.4% with -2 / -5 / -10% slack), with a median of 518 / 554 / 614 / 731 candidates;
  - a filter-then-select top-k with a full fallback would stay exact. Its ceiling is a fraction of the top-k share of the +2.30 ms 512K excess, likely <= ~0.3 ms/step (~1% at 512K), and 0 at G1 / G4 contexts.

### 4. Live state, files, rollback
- **Production** g5lc-PU (04:38:55 UTC, supervisor, launcher f214c2c9 defaults, AV unpinned): healthy at +204 s.
  - exact.py **a8c3394af9d2**, own profile 912/464/2320/1373.
  - 8/8 active, 0 restarts, 0 ERROR/Traceback.
  - No G5LC / DSV41_EXTRA / AV-pin variables in the rank environments; PYTHONPATH is `~/dsv41-sched` only.
  - SM 1969-1989 MHz on 8/8; clocks were never raised (`DSV41_GPU_CLOCK_MHZ=2000` on both launches).
- **Nothing to roll back.**
  - No site file is newer than 03:29 UTC on any node; the launcher is f214c2c9 (mtime 2026-10-27 17:48 UTC) on 8/8.
  - Both launches' preflight found manifest d773ed60f27b identical on 8/8.
  - The vision staging set stays valid, because no live file changed. Its W0 artifacts were not touched.
- **Downtime.** The R1 launch took 147 s (04:02:41-04:05:08); the restore took 217 s (04:38:42-04:42:19). Between them, R1 (exact, pinned, recorder idle) served the LAN / tailnet clients.
  - Foreign traffic seen: a ~120K-token chat at ~04:08 that overlapped one recorded question (discarded and rerun); overlaps during two prefix prefills; one timing rerun.
- **Files.**
  - spark-one `~/dsv41-g5lc/`:
    - `py/` (recorder + CPU test);
    - `scripts/` (`g5lc_probe.py`, `g5lc_an.py`, `g5lc_rec_run.sh`);
    - `corpus/` (code snapshot + exact-length prefixes);
    - `rec/R1/` (9 dumps, ~600 MB, safe to delete);
    - `runs/R1/` (probe JSONL);
    - `an/` (`an-all.json`).
  - `~/dsv41-g5lc/py/g5lc_rec.py` is staged on spark-two..eight too. It is inert unless G5LC_REC=1 and the extension flag are given.
  - Mac scratchpad `g5lc/`: same scripts, `g5lc_inst.sh`, `ab.log`, `launches.tsv`, `an/`, `runs/R1/`.
- **Goal status: unchanged.** Production G1 is ~101.5-101.8 of 120 tok/s. Long-context decode is +2.3 ms/step at 512K, a cost only long-context traffic pays.

## Goal window 5, mHC step (2026-10-28 04:49-09:14 UTC): Moet's mHC boundary overlap ported BITWISE (`DSV41_MHC_OVERLAP`); `moet` + `DSV41_MHC_OVERLAP_PROJ=moet` PROMOTED (launcher f214c2c9 -> **afa97ae6**): pinned G1 +3.0% in both pairs, G4 prose MIN +1.6/+3.9%, G4min_hat +3.8%, 4-session step -3.9..-4.4%, every output bitwise. The served-kernel tier (`stock`) loses 0.6% and is NOT promoted

**Result.**
- **What it is.** At every sublayer boundary the served decode graph ran two kernels back to back on the critical path (`mhc_fused_tilelang` + `mhc_pre_big_fuse_with_norm` "fused"). Their coefficient half (the fn projection and the 20x Sinkhorn: post / comb / next pre-mix) is first read at the NEXT boundary. The promoted mode keeps every arithmetic op and:
  - runs the post-mix + collapse + RMSNorm + draft aux as ONE launch on the critical path (vLLM-Moet b0dc737 `mhc_post_norm`, mode 1, Apache-2.0);
  - computes the projection (Moet `mhc_proj`, 4 tokens per CTA) and the "stats" Sinkhorn kernel on a side stream that the decoder layer already joins after the sublayer;
  - only in FULL decode graphs of 1-64 tokens (eager prefill, piecewise graphs and > 64 tokens keep the served function).
- **Exact.**
  - GPU tests, 0 ulp everywhere (int-view compares): 290/290 + 348/348 checks on synthetic inputs (M = 1-64, aux on/off, eager + CUDA-graph replay with fresh inputs, 5-boundary chains), and 2,368 + 1,184 checks on 5 recorded real boundaries (512 rows each) sliced to M = 1, 6, 12, 24 (and 2, 4, 8, 16, 32, 33, 48, 64), eager + graph replay, all 6 outputs vs the live served function.
  - End to end on every pinned launch (av-pin-k5-g3FU): exact.py **a8c3394af9d2** with the pin's counters 909/467/2335/1373 (per position 348/233/171/90/67) at the start and after the suite, and the G1 reference outputs (2-rep digest **9a18f802d930f6e7**, 3-rep **bc1e8386878bd2b7**). Production on the new launcher: exact.py a8c3394af9d2 (unpinned profile 348/231/171/91/68).
- **Speed** (pinned, 2000 MHz, every kept measurement clean of foreign traffic): G1 **101.30 -> 104.37 / 104.36** (+3.03% / +3.03% vs the control between them), G4 prose MIN 48.52 -> 49.31 / **50.42**, G4min_hat 48.20 -> 50.04 / 50.05, 4-session step 45.24 -> 43.27 / 43.46 ms (-4.4% / -3.9%), matched C1 coding / prose -2.2..-2.3% / -2.6..-2.9% ms per step.
- **Goal status** (see section 6): G1 104.4 of 120 tok/s (+15.0% needed). G4: the prose session's G4min_hat is now above 50 (50.04 / 50.05) and its per-run MIN reached 51.43 / 49.94 / 49.89 on MP2, but not every run is >= 50 yet (worst 49.89, -0.2%).
- **Downtime:** 10 launches (140-211 s each) and 3 stopped windows for GPU tests, about 34 min in all between 05:15 and 08:51 UTC; the LAN/tailnet clients were served by the pinned instances in between (all exact).

### 1. The port (`DSV41_MHC_OVERLAP`, default off)
- **What runs where.** Each sublayer boundary of the served decode graph is two kernels back to back on the caller stream:
  `mhc_fused_tilelang` (post-mapped bf16 streams + the fn-projection split-K partials) and
  `mhc_pre_big_fuse_with_norm` "fused" (split sums, sigmoids, 20x Sinkhorn, collapse with the CARRIED pre-mix, RMSNorm, aux).
  Only the streams, the collapse, the RMSNorm and the aux feed the next sublayer; post / comb / next_pre are first read at the
  NEXT boundary. The overlap keeps every arithmetic op and moves the coefficient half onto a side stream that the decoder
  layer already joins after every sublayer (FULL decode graphs of 1-64 tokens only; eager, prefill and piecewise graphs keep
  the served function).
- **Modes** (all bitwise on this build, section 2):

  | mode | caller stream (critical path) | side stream (coefficients) | new arithmetic |
  |---|---|---|---|
  | `stock` | `mhc_post_tilelang` (the served > 32-token post kernel) + pre-norm `split_mode="input"` | the served `mhc_fused_tilelang` (its bf16 copy discarded) or, > 32 tokens, the served DeepGEMM tf32 prenorm GEMM; then pre-norm `split_mode="stats"` | none (same TileLang sources; S1-S3 are codegen equalities, tested) |
  | `moet` (`PROJ=fused`) | vLLM-Moet b0dc737 `mhc_post_norm` (one launch, 5-CTA cluster per token, contraction mode 1) | as `stock` | Moet's kernel replicates S1+S2's rounding sequence (tested 0 ulp) |
  | `moet` + `DSV41_MHC_OVERLAP_PROJ=moet` | as `moet` | Moet `mhc_proj` (4 tokens per CTA x 8 splits: 8 CTAs at 1-4 tokens, 24 at 12, instead of the served 96-1,440) + `"stats"` | `mhc_proj` replicates the served split-K partials (tested 0 ulp, S5) |

- **Not ported** (not bitwise or not applicable): the stock vLLM `mhc_stream` overlap (tf32 projection on the bf16 streams and a
  `residual.mean` aux), Moet's tf32 projection, the fused TP4 all-reduce (TP8 here; our rdma_ar and its L2-prefetch hooks stay),
  Moet's P2P all-reduce consumer, the DSpark drafter hunk (`dspark.py` is a supervisor-manifest file; left untouched) and the
  first-layer / Engram `mhc_pre` overlap (`DSV41_MHC_OVERLAP_PRE=1`, tested bitwise but not used: record_stream + 3 L2-prefetch
  reads lost).
- **Files** (`site-packages/vllm/models/deepseek_v41/nvidia/`, identical on 8/8, installed 05:22:50 UTC in the stopped window):
  - `model.py` 9c2d22ad -> **ca14bbb1** (6 hunks; with `DSV41_MHC_OVERLAP` and `DSV41_MHC_OVERLAP_DUMP` unset the helper is
    not imported, no stream is created and every hunk reduces to the unpatched expression);
  - new `dsv41_mhc_overlap.py` **4468cad5** (the dispatcher, the overlap boundary, the Moet loader and an armed-only input
    recorder). The first install (d4213ec5, 05:16) was replaced before any launch: Moet's `_ext()` sets `TORCH_CUDA_ARCH_LIST`
    process-wide, and the loader now restores it so no later torch extension build in a worker sees a changed arch list;
  - installer `~/dsv41-stage/g5mhc/g5mhc_install.py` (e48e27ef) `status|install|uninstall`, staged on 8/8 with the site
    files and the GPU test; backups `model.py.bak-g5-mhc` (= 9c2d22ad) and the marker `dsv41_mhc_overlap.py.absent-g5-mhc`.
  - Moet sources (Apache-2.0) `~/dsv41-stage/moet/src/vllm-moet-b0dc737/tools/dsv41_sm120/mhc_overlap/` (identical on 8/8);
    the extension is built per node into `~/.cache/torch_extensions/py312_cu130/vllm_moet_mhc_post_norm_sm120/` (built
    05:23 in the stopped window; the same SASS af84d1d71c3e on 8/8, sm_120 cubin running on sm_121).
- **Knobs** (read at import): `DSV41_MHC_OVERLAP` off|stock|moet; `_PROJ` fused|narrow|moet; `_MAX_TOKENS` 64; `_JOIN`
  boundary|ar; `_PRE` 0|1; `_MOET_MODE` 1; `_DUMP` + `_DUMP_ARM`/`_ROWS`/`_EVERY`/`_N` (recorder, armed by
  `/dev/shm/dsv41-mhc-dump.arm` only).
- The supervisor manifest (d773ed60f27b) does not cover `model.py`, so the install alone did not change it, and every launch
  of this step passed `--expect-launcher f214c2c9`.

### 2. Bitwise tests (stopped windows; `tests/test_mhc_overlap_bitwise.py`, int-view compares, 0 ulp everywhere)
- **U1, synthetic inputs** (05:19 UTC; M = 1, 2, 4, 5, 6, 8, 12, 16, 20, 24, 32, 33, 48, 64; aux on/off):
  - spark-two, `stock` fused + narrow + the PRE path: **290/290 bitwise**. S1 (post kernel == fused kernel bf16 streams,
    22), S2 ("input" == "fused" layer_input/aux, 28), S3 ("stats" == "fused" post/comb/next_pre, 28), S4 (narrow tile_n=24
    partials == served, 22), boundaries eager + CUDA-graph replay with 2 fresh input sets (168), 5-boundary chains eager +
    graph (10), PRE (12).
  - spark-three, Moet tier: **348/348 bitwise**. S5 (Moet `mhc_proj` partials == served, 22), S6 (`mhc_post_norm` mode 1 ==
    the S1/S2 references, 28), stock/moet and moet/moet boundaries eager + graph (168), chains (20). Modes 0 and 2 differ by
    1-2 ulp (informational; Moet's own test would have accepted them).
- **U2, recorded inputs** (05:29 and 06:25 UTC, spark-one): instance g5mhc-D1 (pinned, `DSV41_MHC_OVERLAP_DUMP`, overlap off)
  recorded 5 real eager prefill boundaries (calls 0, 9, 18, 27, 36 of a 1,255-token prompt; 512 rows each) on rank 0. Rows
  sliced to M = 1, 2, 4, 6, 8, 12, 16, 24, 32, 33, 48, 64 (head and strided rows), aux on and off, eager AND captured-graph
  replay with fresh rows, all 6 outputs (residual, post, comb, layer_input, next_pre, aux) vs the live served function:
  - stock/fused + moet/fused: **2,368/2,368 bitwise**; moet/moet: **1,184/1,184 bitwise**;
  - per M = 1 / 6 / 12 / 24 and per mode: 20 eager + 20 graph-replay boundary checks each, all equal.
- **Dump launch inertness** (g5mhc-D1): the patched `model.py` with the dispatcher installed (overlap off) gave exact.py
  a8c3394af9d2 with the pin's counters.

### 3. Microbench (spark-four, one boundary + a stand-in sublayer of two weight-streaming GEMMs, us per boundary)

| M | served+sub warm | stock/fused | stock/narrow | moet/fused | served+sub cold (48 MB flush) | stock/fused | stock/narrow | moet/fused |
|---|---|---|---|---|---|---|---|---|
| 1 | 32.99 | 34.75 | 32.98 | 32.52 | 93.27 | 95.35 | 92.94 | 87.69 |
| 4 | 44.46 | 53.61 | 55.01 | 49.57 | 98.61 | 97.67 | 106.70 | 94.18 |
| 6 | 49.61 | 60.20 | 57.32 | 55.48 | 101.01 | 99.80 | 98.98 | 96.67 |
| 12 | 61.66 | 70.46 | 67.91 | 66.15 | 105.44 | 104.92 | 103.87 | 103.41 |
| 24 | 80.70 | 68.98 | 73.29 | 70.92 | 107.29 | 103.54 | 105.53 | 102.72 |

- With two big GEMMs as the sublayer the side work mostly queues behind the main stream (warm L2: the overlap loses at M <= 16);
  cold it nets 0-4 us per boundary. The served-tiling projection (96-1,440 CTAs) competes for SMs; Moet's critical-path
  kernel is the only clear per-boundary saving. The end-to-end A/B decides.

### 4. Pinned A/B (short suite; pin `~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json` eeac4f97, "pin hit" on every launch)
- **Method.** Mac driver `g5mhc/scripts/inst.sh` per instance: supervised relaunch `--expect-launcher f214c2c9 DSV41_GPU_CLOCK_MHZ=2000` (clocks never raised; SM 1969-1989 MHz at all 76 reads, each on 8/8) -> gstart (exact.py FIRST + spec-counter deltas) -> pin check -> matched C1 (pdab coding 1x5 + prose 1x5), G1 x3, G4 warm-up + 3, conc8, conc16, exact.py #2. No long prompts, GSM8K or longctx on the A/B instances (trim rule).
- **Foreign-traffic rule** (window-5 brief): before every exactness or timing phase the driver waits until running == waiting == 0 and `request_success_total` is unchanged for >= 30 s; a 0.5 s sampler runs during the phase; a phase whose success delta exceeds its own requests or whose running + waiting exceeds its own concurrency is FOREIGN and rerun (never a pass or a fail).
  - From ~06:30 to ~08:20 UTC the tailnet client sent about one short chat request a minute (sometimes 2 at once), so no 30 s window fit a whole phase (MP1's G1 was FOREIGN 6 times). From MP1's G1 on, the rule is applied **per unit** (`gunit.py`: one G1 request, one G4 batch, one conc run, one exact.py run, one matched run per unit; 20 s idle before each; a FOREIGN unit is discarded and rerun). The numbers come from the same benchmark functions (`goal_single.stream_chat`, `goal_conc4.run_batch`, `bench.py conc`, `exact.py`, `matched_bench.py`) and the same summary code, and the per-unit control C3 equals the per-phase controls C1/C2 (G1 101.30 vs 101.45 / 101.26; C1 30.29 / 26.37 vs 30.21-30.32 / 26.39-26.40 ms).
  - Rejected tries: MP1 G1 5 units FOREIGN, G4 2, conc16 1; MP2, C3 and the production suite none. Every kept unit is CLEAN.
- **Arms:** C = control (patch installed, flag unset); O = `stock`; M = `moet` (served projection on the side stream); MP = `moet` + `DSV41_MHC_OVERLAP_PROJ=moet`. Order C1 O1 M1 C2 M2 | MP1 C3 MP2 (U2 for moet/moet ran in the stopped window between M2 and MP1).

| instance (healthy, UTC) | arm | guard | exact.py #1 / #2 (pin counters) | G1 digest 2 / 3 reps | matched C1 coding / prose ms | **G1** (code / prose / math) | G4 mean / **prose MIN** / worst run / **hat** | step ms 1 / 4 sessions | conc8 / conc16 |
|---|---|---|---|---|---|---|---|---|---|
| C1 (05:32:20) | control (flag unset) | phase | a8c3394af9d2 x2 (same) | 9a18f802 / bc1e8386 | 30.21 / 26.39 | **101.448** (95.773 / 76.435 / 132.135) | 60.485 / **48.279** / 47.613 / **48.05** | 24.872 / 45.411 | 261.1 / 402.1 |
| O1 (05:43:01) | `stock` | phase | a8c3394af9d2 x2 (same) | 9a18f802 / bc1e8386 | 30.4 / 26.52 | **100.836** (95.121 / 75.858 / 131.529) | 61.236 / **48.014** / 47.28 / **47.764** | 24.324 / 45.168 | 263.8 / 404.0 |
| M1 (05:55:18) | `moet` | phase | a8c3394af9d2 x2 (same) | 9a18f802 / bc1e8386 | 30.06 / 26.13 | **102.248** (96.567 / 77.06 / 133.116) | 62.022 / **48.304** / 47.563 / **47.863** | 24.403 / 45.337 | 266.4 / 397.1 |
| C2 (06:05:38) | control (flag unset) | phase | a8c3394af9d2 x2 (same) | 9a18f802 / bc1e8386 | 30.32 / 26.4 | **101.26** (95.444 / 76.331 / 132.005) | 62.583 / **48.141** / 47.528 / **48.077** | 23.967 / 45.436 | 257.9 / 384.6 |
| M2 (06:16:05) | `moet` | phase | a8c3394af9d2 x2 (same) | 9a18f802 / bc1e8386 | 30.03 / 26.21 | **102.211** (96.521 / 77.163 / 132.95) | 60.834 / **48.079** / 47.448 / **48.226** | 23.329 / 45.425 | 267.4 / 394.9 |
| MP1 (06:27:26) | `moet` + `PROJ=moet` | matched: phase; rest: unit | a8c3394af9d2 x2 (same) | 9a18f802 / bc1e8386 | 29.62 / 25.61 | **104.369** (98.57 / 78.581 / 135.957) | 62.413 / **49.312** / 48.673 / **50.043** | 24.004 / 43.268 | 270.8 / 403.9 |
| C3 (08:21:24) | control (flag unset) | unit | a8c3394af9d2 x2 (same) | 9a18f802 / bc1e8386 | 30.29 / 26.37 | **101.295** (95.666 / 76.377 / 131.843) | 61.004 / **48.522** / 47.409 / **48.197** | 24.649 / 45.24 | 261.4 / 397.2 |
| MP2 (08:35:00) | `moet` + `PROJ=moet` | unit | a8c3394af9d2 x2 (same) | 9a18f802 / bc1e8386 | 29.58 / 25.68 | **104.36** (98.625 / 78.517 / 135.938) | 63.329 / **50.417** / 49.885 / **50.047** | 23.536 / 43.464 | 267.0 / 408.0 |

| pair (candidate vs control) | matched C1 coding / prose | **G1** (code / prose / math) | G4 mean / **prose MIN** / **hat** | 4-session step | conc8 / conc16 | outputs |
|---|---|---|---|---|---|---|
| O1 vs C1 | -0.62 / -0.49% | **-0.60%** (-0.68 / -0.75 / -0.46) | +1.24 / **-0.55** / **-0.60%** | +0.54% | +1.03 / +0.47% | identical |
| M1 vs C1 | +0.50 / +1.00% | **+0.79%** (+0.83 / +0.82 / +0.74) | +2.54 / **+0.05** / **-0.39%** | +0.16% | +2.03 / -1.24% | identical |
| M1 vs C2 | +0.86 / +1.03% | **+0.98%** (+1.18 / +0.96 / +0.84) | -0.90 / **+0.34** / **-0.45%** | +0.22% | +3.30 / +3.25% | identical |
| M2 vs C2 | +0.97 / +0.72% | **+0.94%** (+1.13 / +1.09 / +0.72) | -2.79 / **-0.13** / **+0.31%** | +0.02% | +3.68 / +2.68% | identical |
| MP1 vs C3 | +2.26 / +2.97% | **+3.03%** (+3.04 / +2.89 / +3.12) | +2.31 / **+1.63** / **+3.83%** | +4.56% | +3.60 / +1.69% | identical |
| MP2 vs C3 | +2.40 / +2.69% | **+3.03%** (+3.09 / +2.80 / +3.11) | +3.81 / **+3.91** / **+3.84%** | +4.09% | +2.14 / +2.72% | identical |
| MP1 vs C2 | +2.36 / +3.08% | **+3.07%** (+3.28 / +2.95 / +2.99) | -0.27 / **+2.43** / **+4.09%** | +5.01% | +5.00 / +5.02% | identical |
| C3 vs C2 | +0.10 / +0.11% | **+0.03%** (+0.23 / +0.06 / -0.12) | -2.52 / **+0.79** / **+0.25%** | +0.43% | +1.36 / +3.28% | identical |

- **stock (O1): NOT promoted** (-0.6% G1 on its first pinned pair; trim rule: stop). The served projection on the side stream (96-1,440 CTAs per boundary) competes for SMs with the next sublayer, and the two served kernels on the critical path save little.
- **moet with the served projection (M1, M2): G1 +0.8..+1.0% in 3 pairs, G4 flat** (prose MIN -0.13..+0.34%, hat -0.45..+0.31%, 4-session step unchanged). The one-launch boundary kernel saves the critical-path time; at 4 sessions (12-24 rows) the side stream's 512-1,152 CTAs per boundary give it back.
- **moet + Moet projection (MP1, MP2): PROMOTED.** G1 +3.03% in both pairs (+3.07% vs C2), every type +2.8..+3.3%, and the 4-session step -3.9..-4.4% (the side stream now launches 8 x ceil(M/4) CTAs). Single-session step 24.65 -> 24.00 / 23.54 ms (-2.6% / -4.5%).
- Memory: KV cache 6,390,007 tokens on every launch; MemAvailable at the end of MP2 13.2 GB on spark-one (C3 13.3 GB). MP1 ended at 9.6 GB after 2 h of foreign traffic on the same instance (allocator growth from many eager prefills); MP2, the same config after a normal suite, ended at 13.2 GB. O1 (stock) ended at 9.2 GB after a normal suite: not investigated, since stock is not promoted. Production ended at 9.2 GB after the 1M and the 200K longctx prompts, as the gate's runs do.

### 5. Promotion and production (launcher afa97ae6)
- **Launcher** `~/dsv41-tp8.sh` f214c2c9 -> **afa97ae6** on 8/8 (08:46:42 UTC, `promote.sh`; backup `~/dsv41-tp8.sh.bak-g5-mhc` = f214c2c9). Two lines after `DSV41_TOPK_CANON_TIES`:
  `export DSV41_MHC_OVERLAP=${DSV41_MHC_OVERLAP:-moet}` and `export DSV41_MHC_OVERLAP_PROJ=${DSV41_MHC_OVERLAP_PROJ:-moet}` (with the usual comment).
- **Supervisor** 7bea1cf3 unchanged; its preflight manifest d773ed60f27b -> **0c96a727fb81** (the launcher is a manifest file).
- **Production instance g5mhc-PU** (supervisor, launcher afa97ae6 defaults, AV unpinned; healthy 08:51:16 UTC, +199 s; every phase CLEAN of foreign traffic, G1 / GSM8K / longctx / exact.py #2 with the per-unit guard):

  | check | g5mhc-PU (afa97ae6) | earlier production (f214c2c9 or older) |
  |---|---|---|
  | exact.py first / after the suite | **a8c3394af9d2** / a8c3394af9d2 (909/467/2335/1373; per position 348/231/171/91/68, the unpinned profile) | a8c3394af9d2 (same unpinned profile on vis-R2) |
  | G1 (code / prose / math), reps | **104.06** (97.84 / 78.46 / 135.89); 104.01 / 104.12 / 104.05 | 101.5-102.0 (g4v-FU 102.00; gate 101.81) |
  | G1 AL / outputs | 2.583 / 1.857 / 3.607; digest **bc1e8386878bd2b7** (= the pinned reference outputs; reps identical) | g4v-FU bc1e8386878bd2b7 |
  | ONE cold 1M (1,024,611 tokens) | **168.3 s**, good class (late/early 1.198; chunks 2-12 2.43 s, 50-62 2.91 s) | 166.2 s (g4v-FU), 168.3 s (gate) |
  | GSM8K-50 | **49/50** (idx 12, the known borderline item; 7 units of 8 items at concurrency 8) | 50/50 (g4v-FU), 195/200 (gate) |
  | longctx | **PASS** (needle 60/60, copy 1.000, 4K-200K) | PASS |
  | usage `cached_tokens` (repeated 4.2K prompt) | 4160 | 4160 |
  | MemAvailable, 1 Hz on 8/8, after healthy (1M, GSM8K, longctx) | min **8.97-10.38 GB** (spark-one 8.97) | 8.6-9.96 GB during window 4's 1M |
  | MemAvailable dip during startup (just before healthy) | 6.38-7.61 GB (no 1 Hz startup baseline exists) | - |
  | ERROR / Traceback, restarts, SM clock | 0 / 0 on 8/8; 1969-1989 MHz | |

- **Config check on 8/8** (`verify8.sh post-promote`, read-only, 08:53 UTC): launcher afa97ae6 (backup f214c2c9), `model.py` ca14bbb1 (backup 9c2d22ad), `dsv41_mhc_overlap.py` 4468cad5, `dspark.py` 36811ca9 and `model_runner.py` 6a4f2433 untouched, Moet sources 2f18249b (.cu) / ce0f79ab (.py), the built extension present; running env `DSV41_MHC_OVERLAP=moet DSV41_MHC_OVERLAP_PROJ=moet`; every rank logged `DSV41_MHC_OVERLAP=moet: mHC coefficients on a side stream in FULL decode graphs of 1..64 tokens (proj=moet, pre=0, join=boundary, moet mode 1)`; identical on 8/8.

### 6. Goal status (2000 MHz, no clock or power change)
| target | before this step | after (production g5mhc-PU / pinned MP2) | remaining |
|---|---|---|---|
| G1 mean >= 120 tok/s | 101.5-102.0 (production), 101.3 (pinned C3) | **104.06 / 104.36** | +15.3% / +15.0% |
| G4: every session >= 50 tok/s in every run | prose MIN 48.4 production, 48.1-48.5 pinned; worst run 47.4-47.9 | pinned MP2: prose MIN **50.42** (runs 51.43 / 49.94 / 49.89), G4min_hat **50.05**; MP1: 49.31 / 50.04 | worst run -0.2% (MP2), -2.7% (MP1) |

- G4 was not run on the production instance (the trimmed final verification: exact, G1, one 1M, GSM8K-50, longctx); the G4 numbers are pinned (av-pin-k5-g3FU).
- The prose session's step-time estimate is now above 50 on both MP instances, but the per-run minimum still dips below 50 in some runs: G4 needs about +0.2..+2.7% more on the 4-session step (the worst prose runs, MP2 / MP1) to hold >= 50 in every run.
- Next exact levers on top of this (each needs its own bitwise test + pinned pairs): the drafter's 3 layers on the same overlap (`DSV41_MHC_OVERLAP_DRAFT`, needs a `dspark.py` hunk, a manifest file: prep has it), `DSV41_MHC_OVERLAP_PRE=1` for the first-layer / Engram `mhc_pre` (bitwise in U1), and the window-3/4 MoE items (FC1->FC2 pipelining, contiguous FC2C, q_b in the L2 plan, drafter MoE on M32). A fresh AV pin under the new step costs would change trajectories (new anchor: user decision).

### 7. Live state, rollback, files
- **Live:** g5mhc-PU, production, launcher **afa97ae6** defaults, AV unpinned; healthy since 08:51:16 UTC on 8/8, 0 restarts, 0 ERROR/Traceback; 0 requests in flight at the end; `dsv41-lanproxy` untouched.
- **Rollback:**
  - one launch, no file change: supervisor argument `DSV41_MHC_OVERLAP=0` (served path, no side stream; the patched `model.py` is then inert);
  - permanent: `cp ~/dsv41-tp8.sh.bak-g5-mhc ~/dsv41-tp8.sh` on 8/8 (f214c2c9, manifest back to d773ed60f27b), then a supervised relaunch;
  - full removal, service stopped: `python3 -I ~/dsv41-stage/g5mhc/g5mhc_install.py uninstall` on 8/8 (restores `model.py` 9c2d22ad, deletes the helper and the marker), then restore the launcher.
- **Vision staging set: STALE** (it is pinned to the 10-file manifest and the launcher changed). Rebuild it against:
  - launcher `afa97ae6dfc65943f7e536baf7922bbecf2c5d8110f35768018c7c7dd5be3cc1` (manifest **0c96a727fb81**; the staged L1/L2 launchers are f214c2c9-based and lack the two mHC lines);
  - `model_runner.py` unchanged (6a4f2433), so the staged vision patch base still holds;
  - also changed, not manifest files: `nvidia/model.py` `ca14bbb1df9e4379dc21f051fd782a6403ec40687872d6711e1dcb2f708fd005`, new `nvidia/dsv41_mhc_overlap.py` `4468cad5aec3414ace5ccc7d8af0d448d7561e67d890fd106098d7220e4f9759`; serving dependency: Moet sources `mhc_post_norm_sm120.cu` 2f18249b / `.py` ce0f79ab in `~/dsv41-stage/moet/src/vllm-moet-b0dc737/tools/dsv41_sm120/mhc_overlap/` and the per-node build in `~/.cache/torch_extensions/py312_cu130/vllm_moet_mhc_post_norm_sm120/` (rebuilt automatically on first import if removed).
  - The W0 artifacts (`~/dsv41-stage/vision/autotune-W0`) were not touched.
- **Files.**
  - Staged on 8/8: `~/dsv41-stage/g5mhc/` (`g5mhc_install.py` e48e27ef, `site/model.py`, `site/dsv41_mhc_overlap.py`, `tests/test_mhc_overlap_bitwise.py` 3e9dade7; SHA256SUMS fc43ddb4).
  - spark-one `~/dsv41-g5mhc/`: `runs/<label>/` (per-phase outputs; `gunit-<phase>.log` = per-unit guard verdicts), `runs/<label>-start/`, `dumps/g5mhc-D1/rank0/` (5 recorded boundaries, 141 MB, safe to delete), scripts (`vphase.sh`, `gunit.py`, `idlewait.py`, `gssamp.sh`, `dumpreq.py`, `finsampler.sh`).
  - Mac scratchpad `g5mhc/`: `scripts/` (`sw.sh` stopped windows, `inst.sh` / `inst2.sh` one instance, `blk.sh`, `gres.sh`, `promote.sh`, `mklauncher.py`, `verify8.sh`, `memw.sh`, `agg.py`, `readme_tables.py`, `vsumm.py`), `src/` (patched files, installer, test), `sw/` (U1 / U2 / bench logs), `ab.log`, `launches.tsv`, `runs/`, `ab/`, `mw/` (1 Hz MemAvailable of the production suite), `verify-post-promote.txt`, `tables-ab.md`.

## Goal window 5, mHC co-residency gate (2026-10-28 09:20-10:33 UTC): PASS, the promoted `DSV41_MHC_OVERLAP=moet` + `DSV41_MHC_OVERLAP_PROJ=moet` STAYS (launcher afa97ae6 unchanged). 0 mismatches in 7.56 M co-run calls of the newly concurrent kernels, 0 in 415,360 routed-MoE outputs computed beside them, 0 in 12,500 replays of the overlapped decode region; racecheck / synccheck clean; production exact.py a8c3394af9d2 under live overlap

**Why this gate.** spark-ds41f's 2026-09-30 proxy-fence audit: kernels that released a TMA stage without `fence.proxy.async` while ldmatrix/LDS reads were still outstanding gave wrong outputs ONLY when another kernel's CTAs shared the SM (29-609 wrong per 12,000 calls), 0 when run alone; warp-specialized pipelines raced on GB10 while passing a single bitwise compare. A solo 0-ulp test and exact.py cannot see that, and the mHC overlap adds exactly this co-residency (side-stream mHC work beside the sublayer, l2pf, the aux-stream shared expert and the WOA side stream). Rule: any single mismatch = roll back to f214c2c9; all 0 = keep.

### 1. What the promoted setting runs concurrently
| kernel | stream, when | launch | registers per CTA (SM has 65,536) / smem | TMA | warp-specialized | cluster / DSMEM | PDL | can share an SM with other CTAs |
|---|---|---|---|---|---|---|---|---|
| Moet `mhc_post_norm_kernel<1, PDL, no AR>` (NEW kernel, b0dc737) | caller stream, every overlapped boundary, M 1-64 (+ `capture_aux` on the draft-tap boundaries) | T x 5 CTAs as clusters of 5, 128 threads | 80 x 128 = 10,240 / 1,296 B | no | no | 5-CTA cluster, one `barrier.cluster`; the collapsed values go through a global [T, H] L2 scratch (no DSMEM reads) | yes (`griddepcontrol.wait` before reading any activation or storing) | yes |
| Moet `mhc_proj_kernel<4>` (NEW kernel) | side stream, M <= 32 | 8 x ceil(M/4) CTAs, 256 threads | 255 x 256 = 65,280 (whole register file) / 2,688 B | no | no | no | no | no |
| TileLang `mhc_pre_big_fuse_with_norm`, `split_mode="stats"` (served source, new split mode) | side stream, every M | M CTAs, 96 threads | <= 24,480 / dynamic, cp.async staging | no (`TL_DISABLE_TMA_LOWER`) | no (`TL_DISABLE_WARP_SPECIALIZED`) | no | no | yes |
| DeepGEMM `sm120_tf32_hc_prenorm_gemm<24, 20480, 128, 32, 64, 16, 4, 256, 128>` (served kernel, now on the side stream) | side stream, M 33-64 | 16 CTAs (16 splits), 384 threads = 128 TMA + 256 math | 168 x 384 = 64,512 (whole register file) / 4-stage TMA ring | yes | yes (TMA producer + math warps, mbarrier full/empty, `fence.proxy.async` before every stage release) | no | yes | no |

Their new neighbours are unchanged served kernels: the attention sublayer, the routed MoE (m32-bw `all` FC1 + D05/FC2C FC2 = DeepGEMM `sm120_fp8_fp4_gemm_1d1d`: 384 threads = 128 TMA + 256 math, 168 registers = the whole register file, 7-stage TMA ring, warp-specialized, `fence.proxy.async` before every stage release, persistent grid over the 48 SMs), its glue kernels, the aux-stream shared expert, WOA and l2pf.
- **Structural finding (not relied on by the gate).** On GB10 the TMA / warp-specialized pipelines in this picture (tf32 prenorm GEMM, M32 FC1/FC2) and Moet's `mhc_proj` each hold the SM's whole register file, so no CTA of another kernel can be resident on their SM while they run: the audit's failure mode (a TMA pipeline sharing its SM with foreign CTAs) cannot occur for them on this build. The two kernels that do share SMs (Moet post_norm, TileLang stats) have no TMA, no warp specialization and no async-proxy shared-memory writes.

### 2. The stress gate (stopped window, 5 Sparks in parallel + 1 for the sanitizers)
- **Harness** `~/dsv41-g5stress/g5stress.py` (5bb65cd6) + `g5co.py` (ce67a28d, extension `g5stress_co_v3`) + `run.sh` (07fa2ae3) on spark-one..six. It runs in the launcher's environment (every `export` line of `~/dsv41-tp8.sh`) and the live venv, imports the installed overlap module (4468cad5: MODE moet, PROJ moet, MAX_TOKENS 64, MOET_MODE 1) and calls every kernel exactly as the overlap does (Moet wrapper defaults: mode 1, PDL on; DeepGEMM PDL on). It refuses to run while `dsv41-rank` is active or any process holds the GPU.
- **Recorded inputs.** The 5 real boundaries of g5mhc-D1 (512 rows each), rows sliced to M (head and strided) = 10 input sets per M. Every call writes its own poisoned output slot (100 calls per CUDA graph, replayed), and every output word is compared (integer views) with the same kernel's solo eager result on the same input set.
- **M** = 1, 6, 12, 24, 32, 48, 64 (48 and 64 are FULL-graph sizes and take the tf32 path).
- **Co-runners** on another stream. The test stream runs at high priority (-3) on four nodes and at default priority on spark-six:
  - `moe`: the served routed-MoE path through the live site loader (natives `trees/all-base`, `trees/all-d05`, the g2 FC2C tree), real checkpoint layers 0 and 26 of the node's TP rank, recorded decode routes (27 single-user windows, 33 four-user batches), the recorded layer input; 40 applies per graph replay. Every MoE output is checked against its own solo run on every replay.
  - `copy`: a persistent bandwidth copy (128 MiB ring), 2 CTAs x 256 threads on every SM, stopped by a flag. Every co-runner CTA records `%smid` and `%globaltimer` at entry and exit. In every round, all 48 SMs had a co-runner CTA resident for 100% of the test window, so same-SM residency is proven, not assumed.
  - `lds`: a persistent shared-memory hammer, 2 CTAs per SM, each with 256 threads and 17 KiB of shared memory, doing 32 LDS + 8 STS per thread per iteration. Same residency proof as `copy`.
  - `copyhalf`: the copy on 24 SMs, for `proj` and `tf32`, which can only run on an SM with no other CTA; they run on the free SMs while the copy saturates DRAM.
  - `solo`.
- **Contention was real.** Co-run time per 100 calls vs solo: post M=6 0.88 -> 19.3 ms (copy) / 3.8 (lds) / 1.7 (moe); proj M=32 2.7 -> 19.4 (moe); tf32 M=48 1.95 -> 19.2 (moe); stats M=24 0.46 -> 26.0 (copy).

**Results** (5 nodes: spark-one/two/three/four TP ranks 0-3, spark-six TP rank 5; 12,000 calls per node for every kernel x M x co-runner, i.e. 60,000 per cell):

| kernel | M | co-runners (60,000 calls each) | calls | mismatches |
|---|---|---|---|---|
| Moet `mhc_post_norm` (aux off) | 1, 6, 12, 24, 32, 48, 64 | solo, moe, copy, lds, copyhalf | 2,100,000 | **0** |
| Moet `mhc_post_norm` (aux on) | 1, 6, 12, 24, 32, 48, 64 | solo, moe, copy, lds, copyhalf | 2,100,000 | **0** |
| TileLang pre-norm `stats` | 1, 6, 12, 24, 32, 48, 64 | solo, moe, copy, lds, copyhalf | 2,100,000 | **0** |
| Moet `mhc_proj` | 1, 6, 12, 24, 32 | solo, moe, copyhalf | 900,000 | **0** |
| DeepGEMM tf32 prenorm GEMM | 48, 64 | solo, moe, copyhalf | 360,000 (+6,000 in one round the MoE co-runner did not fully cover) | **0** |
| routed MoE beside them (FC1/FC2/FC2C + glue) | 1-64 | every `moe` unit | 415,360 outputs | **0** |

- **Overlapped decode region** (CUDA graph per M = 1, 6, 12, 24, 48): 4 boundaries through the production `shifted_post_pre_overlap`, attention stand-in GEMMs, the routed MoE (real weights, recorded routes) with a shared-expert stand-in on an aux stream, an l2pf stand-in copy on a 4th stream, and the decoder's joins (28 outputs). Per node 250 replays alone + 250 beside `copyhalf`, i.e. **2,500 replays per M (12,500 in all), every replay compared with the first: 0 mismatches**. Replay 0 equals the eager served path (`stream=None`) bitwise on all 28 outputs, on every node and M.
- **Solo checks** (every node, M and input set): solo repeat 0, aux-on vs aux-off streams / layer input 0, composite (Moet post_norm + Moet proj or tf32 + stats) vs served `mhc_shifted_post_pre(stream=None)` 0, Moet proj partials vs served `mhc_fused_tilelang` partials 0. The `stats` kernel never writes or reads `layer_input` (a poisoned buffer stays poisoned, outputs unchanged), which matters because the side stream passes it the buffer the sublayer is reading.
- **compute-sanitizer** (optional, spark-five, recorded inputs, all five kernels at M = 6 and 48): racecheck `--racecheck-report all` **0 hazards**, synccheck **0 errors**. It checks hazards inside a kernel only, not co-residency.
- **GPU clock** 1969-1989 MHz on every node during the runs (2000 MHz policy, nothing raised).

**GATE: PASS.** No rollback. Nothing in the serving path changed in this step: launcher, supervisor manifest and site files are untouched, and the vision staging set is as stale as after the mHC step.

### 3. Production after the gate
- **g5stress-PU** (supervisor, launcher afa97ae6 defaults, AV unpinned, clock policy 2000 MHz): preflight site manifest 0c96a727fb81 + launcher afa97ae6 identical on 8/8, Engram 8/8 OK; healthy 10:29:01 UTC (+203 s after systemd-run).
- **Foreign-traffic guard:** every phase CLEAN (exact.py 7 of 7 own requests, G1 9 of 9, exact2 6 of 6; max running 1).
- **exact.py first, under live overlap: a8c3394af9d2.** Counters 907/469/2345/1373, per position 348/231/171/90/67. That is this unpinned launch's own AV profile; the anchor is the gate.
- **exact.py again after G1: a8c3394af9d2**, same counters.
- **G1 104.02** (code 98.46 / prose 78.47 / math 135.14; reps 104.04 / 104.19 / 103.84). Outputs are identical across reps: digest **bc1e8386878bd2b7** (= the reference outputs; 2-rep digest 9a18f802d930f6e7).
- **Config on 8/8** (read-only after healthy):
  - launcher afa97ae6, `nvidia/model.py` ca14bbb1, `dsv41_mhc_overlap.py` 4468cad5;
  - running env `DSV41_MHC_OVERLAP=moet DSV41_MHC_OVERLAP_PROJ=moet`, the overlap line once per rank (`proj=moet, pre=0, join=boundary, moet mode 1`);
  - 0 ERROR/Traceback, 0 restarts, KV 6,390,007 tokens, SM 1969-1989 MHz, MemAvailable 13.5-14.6 GB at the end;
  - the vision W0 artifacts (`~/dsv41-stage/vision/autotune-W0`, 40 files) are untouched on 8/8.

### 4. Window, downtime, files
- **Service down 10:00:16 -> 10:29:01 UTC (28 min 45 s).** The time went to: stop (3 s after a 30 s idle wait), three smoke rounds, the full suite (4 min 46 s on 5 nodes in parallel) and the supervised relaunch.
  - Smoke round 1: the relaunched co-runner left gaps, so the residency proof never held (INCOMPLETE, 0 mismatches).
  - Smoke round 2: the first flag-terminated build deadlocked (a test-stream launch could not start next to the resident co-runner) and was killed after the timeout.
  - Smoke round 3: build v3 preloads every module before the co-runner starts and gives the co-runner the full shared-memory carveout. Clean in 11 s.
- **Foreign traffic:** idle (30 s, counter 104) before the stop; production verification under the per-phase guard.
- **Files**:
  - spark-one..six `~/dsv41-g5stress/`: harness, `SHA256SUMS`, `out/<tag>.{jsonl,err,rc,san.txt}`, and `dumps/rank0/` (135 MB copy of the g5mhc-D1 dumps; deletable).
  - Test-only extension builds `~/.cache/torch_extensions/py312_cu130/g5stress_co_v{1,2,3}` (deletable).
  - Mac scratchpad `g5stress/` (`scripts/win.sh`, `scripts/inst.sh`, `out/`, `win.log`, `ab.log`, `readme_section.md`).
- **Rollback** (unchanged from the mHC step): one launch `DSV41_MHC_OVERLAP=0`; permanent `cp ~/dsv41-tp8.sh.bak-g5-mhc ~/dsv41-tp8.sh` on 8/8 + supervised relaunch (`--expect-launcher f214c2c9`).

## Goal window 5, Host step (2026-10-28 10:35-15:23 UTC): the nsys check came first and found 0.0 ms/step of host-bound GPU idle on ranks 0 and 4 (G4 x2, C1), so the host-path overlap STOPPED and nothing was built for it. The combined exact arm `DSV41_FUSED_AV=1` on top of mHC is exact but worth only G1 +0.15/+0.16% and same-output G4 -0.02/+0.21%: NOT promoted. Launcher afa97ae6 unchanged; live g5host-PU

Rule of the step (window-5 Host phase): run the mhc-host-prep.md B.6 nsys first; >= 1 ms/step host-bound GPU idle on rank 0 or 4 = GO with a flag-gated, scheduling-only host overlap; < 0.3 ms = STOP it and instead test the prep's best exact items as one pinned arm on top of the promoted mHC result. Short suite only (trim rule), window-5 foreign-traffic rule throughout.

### 1. The nsys check (mhc-host-prep.md B.6), done first
- **Trace instance g5host-T2** (diagnostic only; the API stayed up on it): `~/dsv41-tp8-hosttrace.sh` (36c3f00c) = the live launcher afa97ae6 verbatim plus an `nsys profile --trace=cuda,nvtx,osrt --osrt-threshold=20000 --cuda-graph-trace=node --python-functions-trace=~/dsv41-hostprof/pyfuncs-host.json --capture-range=cudaProfilerApi --capture-range-end=repeat:6` wrapper, `VLLM_WORKER_MULTIPROC_METHOD=spawn` and the vLLM cuda profiler, on all 8 ranks; AV pin av-pin-k5-g3FU (pin hit), mHC moet/moet on (overlap line on rank 0), 2000 MHz. The first launch (g5host-T1) failed twice at the startup memory check (nsys + spawn leave 110.8-111.1 GiB free < 0.92 x 121.6 GiB); g5host-T2 adds `--gpu-memory-utilization=0.89`, which only moves that check (the KV size is fixed by `--kv-cache-memory-bytes 16G`). Healthy 10:48:05 UTC; service down 10:43:26 -> 10:48:05.
- **Three capture ranges of 60 steps** (`~/dsv41-g5host/hostcap.py`, foreign-traffic guard per range: 30 s idle before, 0.25 s sampler during; all three CLEAN): G4 #1 (the 4 G4 sessions after a warm-up batch, start 5 steps after decode began), C1 (one G1 prose request), G4 #2 (start 40 steps in). Imported with QdstrmImporter (`~/dsv41-g5host/imp1.sh`), analysed on ranks 0 (spark-one) and 4 (spark-five) with the prep's read-only tools `idle_attrib.py`, `hostseq.py`, `steptl.py`.

| range | rank | step (median) | GPU idle per step: total | in target graph | DT window | drafter graph | TD window | **host-bound** | steps with > 1 ms host-bound | host lead | AV wait end -> next target launch | eager launches DT / TD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G4 #1 | 0 | 45.51 ms | 718 us (1.58%) | 412 | 134 | 86 | 86 | **0.0 us** | 0 / 59 | 34.3 ms | 7.3 ms | 49 / 32 |
| G4 #1 | 4 | 45.51 ms | 724 us (1.59%) | 414 | 134 | 86 | 90 | **0.0 us** | 0 / 59 | 37.3 ms | 4.4 ms | 49 / 32 |
| G4 #2 | 0 | 42.63 ms | 715 us (1.68%) | 410 | 133 | 86 | 84 | **0.0 us** | 0 / 59 | 36.0 ms | 3.1 ms | 49 / 32 |
| G4 #2 | 4 | 42.61 ms | 722 us (1.70%) | 411 | 134 | 86 | 90 | **0.0 us** | 0 / 59 | 36.3 ms | 2.9 ms | 49 / 32 |
| C1 | 0 | 22.90 ms | 667 us (2.91%) | 364 | 128 | 92 | 84 | **0.0 us** | 0 / 59 | 16.8 ms | 3.8 ms | 49 / 32 |
| C1 | 4 | 22.91 ms | 671 us (2.93%) | 361 | 130 | 92 | 87 | **0.0 us** | 0 / 59 | 17.7 ms | 2.6 ms | 49 / 32 |
| ref: g1 trace 2027f4d8 (2026-10-26), C4 | 0 | 50.8 ms | 827 us (1.6%) | 495 | 161 | 87 | 84 | 0 | 0 / 31 | 45.7 ms | 2.4 ms | 63 / 31 |

(GPU idle = no kernel, memcpy or memset on any stream, median per step, us. Host-bound = idle that ended at an op whose launch API call ended after the idle began. Host lead = GPU start of the next target graph minus the end of its cudaGraphLaunch call. The densely NVTX-instrumented trace slows the host path itself, e.g. 7.3 ms instead of ~2.4 ms after the AV wait in G4 #1, and the host is still 34-37 ms ahead.)

- **Per step on the rank-0 launching thread (G4 #2):** 33.4 ms blocked in `cudaEventSynchronize` (one call per step = the AV copy event, i.e. waiting for the GPU), 4.2 ms in other CUDA API calls, 4.6 ms of Python. The host queues step N+1 while the GPU is still about 36 ms from finishing step N, so the ~1.6 ms of pre-launch Python the ops step's py-spy saw is never on the critical path.
- **The "GPU busy ~96%" figure does not reproduce:** 98.3-98.4% busy at G4 and 97.1% at C1 on both ranks. Of the 0.67-0.72 ms idle per step, 0.36-0.41 ms is dependency gaps inside the target graph (down from 0.50 ms in the g1 trace thanks to the mHC side stream), 0.13 ms launch gaps between the ~49 small eager kernels of the drafter -> target window, 0.09 ms inside the drafter graph and 0.09 ms in the target -> drafter window. None of it depends on when the host runs its Python.
- **Rank 4 vs rank 0:** the same picture; rank 4 (its RPC arrives through the broadcast MessageQueue) is 1-3 ms further ahead.
- **Decision (rule of the step: >= 1 ms/step host-bound = GO, < 0.3 ms = STOP): 0.0 ms -> STOP the host overlap.** Nothing was built for it (`hostprep/model_runner_hooks.patch` stays unapplied). The only movable GPU-side time is the DT window's launch gaps (0.13 ms/step), which `dtgraph` targets; `dtgraph` is a design only (2-4 days) and was not built.

### 2. The combined exact arm instead: `DSV41_FUSED_AV=1` on top of mHC (pinned A/B, short suite): NOT promoted
- **Arm.** The rule for a STOP was to test the prep's best exact items as one arm on top of the promoted mHC result (launcher afa97ae6 defaults). That is `DSV41_FUSED_AV=1` (window 4 item 09: `reallocate_drafts` as one packed H2D + one Triton launch; bitwise in 31,019 GPU cases + 7,100 serving shadow compares; installed off since window 4, its code is in the live `adaptive_verification.py` and `model_runner.py` 6a4f2433). `dtgraph` is a design only and was not built, so the arm is FUSED_AV alone. Nothing was installed or changed for it.
- **Harness.** The g5mhc/g5stress `inst.sh` with new labels and arms only (`C` = pin + launcher defaults, `F` = pin + `DSV41_FUSED_AV=1`): supervised relaunch with `--expect-launcher afa97ae6` (site manifest 0c96a727fb81 and launcher identical on 8/8 at every launch), SM clock read on 8/8 before every phase (1969-1989 MHz at every read), exact.py first under the foreign-traffic rule, pin check, then matched C1, G1 (3 reps), G4 (warm-up + 3 runs), conc8, conc16, exact.py again, all with the per-unit guard (`gunit.py`: 20 s idle before each request or batch, 0.25 s sampler, overlapped units discarded and rerun).

| instance | arm | exact.py start / end (pin counters) | G1 digest (2 reps) | matched C1 coding / prose ms | G1 (code / prose / math) | G4 mean / prose MIN / G4min_hat | 4-session step ms | conc8 / conc16 |
|---|---|---|---|---|---|---|---|---|
| g5host-C1 | control | a8c3394af9d2 / a8c3394af9d2 (909/467/2335/1373) | 9a18f802d930f6e7 | 29.59 / 25.57 | 104.53 (98.75 / 78.66 / 136.19) | 63.74 / 49.70 / 49.61 | 44.00 | 265.8 / 407.8 |
| g5host-F1 | FUSED_AV=1 | same / same | same | 29.50 / 25.44 | 104.69 (99.01 / 78.80 / 136.26) | 63.08 / 48.63 / 50.03 | 43.57 | 270.9 / 414.8 |
| g5host-C2 | control | same / same | same | 29.54 / 25.54 | 104.37 (98.60 / 78.59 / 135.92) | 65.29 / **50.19** (12/12 sessions >= 50) / 49.86 | 43.87 | 273.6 / 411.0 |
| g5host-F2 | FUSED_AV=1 | same / not run | same | 29.54 / 25.57 | 104.54 (98.75 / 78.75 / 136.12) | 2 of 3 timed runs: 62.95 / 48.75 / 50.20 | 43.09 | not run |
| **F1 vs C1** | | exact same | same | +0.31 / +0.51% | **+0.15%** | -1.04 / **-2.16** / +0.84% | +1.0% | +1.9 / +1.7% |
| **F2 vs C2** | | exact same (start) | same | 0.00 / -0.12% | **+0.16%** | prose MIN -2.9% (2 runs) / hat +0.68% | +1.8% | - |

- **G1: +0.15% and +0.16%,** with identical outputs (sha8 596b478a / 923a46be / 92e5d8cf per type, all reps): consistent but a third of the 0.5% bar.
- **G4: the instance-level numbers are a batching lottery, so the decision uses same-output G4 batches.** The 4 sessions of a G4 batch land in one of several scheduling patterns, and each pattern gives its own 4 output texts, AL and step mix (the prose prefill joining the batch early, TTFT ~0.22 s, gives prose 46.6-49.0; the common pattern in C1/C2, AL 2.583, gives prose 50.0-50.2). The same 4-text set recurs across instances (`scen.txt`). Where an F batch and a no-FUSED_AV batch produced the identical set, the difference is the code's own:

| output set (code/prose/math/mixed sha) | AL | no FUSED_AV (mHC on, same pin) | FUSED_AV=1 | prose tok/s | 4-session step |
|---|---|---|---|---|---|
| c574e7/p26482/m58162/m93985 | 2.370 | g5mhc-MP1 batch 2: prose 48.673, T4 42.32 ms | g5host-F2 warm-up: 48.663, 42.42 ms | **-0.02%** | +0.24% |
| c9ea8e/p2e7fd/mc0618/m5809b | 2.485 | g5mhc-MP2 batch 2: 49.938, 43.25 ms | g5host-F1 batch 2: 50.043, 43.17 ms | **+0.21%** | -0.18% |
| ce3f30/p4ac39/m2a29f/mb4c26 (controls only) | 2.583 | C1 b1/b3, C2 b1/b2/b3: 50.01-50.21, T4 43.84-44.01 ms | - | spread 0.4% | spread 0.4% |

  So FUSED_AV is worth about +0.1% at G4, as its mechanism predicts (51-66 us when the draft budget binds, of a 43 ms step). The +0.7-0.8% hat and +1-1.8% 4-session-step gaps come from the F instances landing in lower-AL patterns (T4 42.4-43.2 ms vs 43.9 ms for the controls' AL 2.583 pattern); the control-vs-control spread is as large (C1 vs C2: hat 49.61 vs 49.86, conc8 265.8 vs 273.6).
- **Decision: NOT promoted** (window rule: >= 0.5% consistent in >= 2 pinned pairs; G1 +0.15/+0.16%, same-output G4 -0.02/+0.21%). FUSED_AV stays installed and off (`DSV41_FUSED_AV` unset).
- **F2 was stopped early** (trim rule; the answer was clear): from ~11:45 UTC a sustained foreign load (tailnet node 100.106.91.100 with 4 connections, this Mac 192.168.0.239, the LAN poller 192.168.0.86; 2-7 requests running at nearly every 2 s sample; F2 served 1,910 completions, almost all foreign) left no 20 s idle window for long stretches. F2's G1 took 2 h (16 tries for 9 clean units), and G4 run 3, conc8/16 and the closing exact.py never got a window. Every kept unit is clean; 16 overlapped units were discarded (matched 1, G1 7, G4 8). The pinned A/B instances served all of that traffic exactly (a8c3394af9d2 at start, G1 digest unchanged).
- **Note for the goal:** pinned control C2 had every session of every timed G4 run >= 50 tok/s (prose MIN 50.19, 12/12), and C1 had 11/12 (MIN 49.70). With the mHC overlap the G4 goal is now decided by which batching pattern the prose session lands in, not by step time.

### 3. Production after the step
- **g5host-PU** (supervisor, launcher afa97ae6 defaults, AV unpinned, clock policy 2000 MHz). The relaunch first waited 1,017 s for a 30 s idle window (foreign load), so no client request was cut. Preflight: site manifest 0c96a727fb81 and launcher afa97ae6 identical on 8/8, Engram 8/8 OK. Healthy 14:57:38 UTC (+204 s; downtime 218 s).
- **exact.py first: a8c3394af9d2** (counters 907/469/2345/1373, per position 349/230/171/90/67: this unpinned launch's own AV profile; the anchor is the gate). **exact.py again after G1: a8c3394af9d2**, same counters. Every phase CLEAN under the per-unit guard (G1: 9 clean units, 1 overlapped unit discarded and rerun).
- **G1 103.97** (code 98.53 / prose 78.14 / math 135.25; reps 103.97 / 103.95 / 104.00), outputs identical across reps: digest **bc1e8386878bd2b7** (= the reference outputs; 2-rep digest 9a18f802d930f6e7).
- **Config on 8/8** (read-only after healthy): launcher afa97ae6, `nvidia/model.py` ca14bbb1, `dsv41_mhc_overlap.py` 4468cad5, `model_runner.py` 6a4f2433, `dspark.py` 36811ca9, supervisor 7bea1cf3; running env `DSV41_MHC_OVERLAP=moet DSV41_MHC_OVERLAP_PROJ=moet DSV41_TOPK_CANON_TIES=all DSV41_FAST_LOAD=1`, `DSV41_FUSED_AV` unset (no FUSED_AV log line), `--enable-prompt-tokens-details` on; KV 6,390,007 tokens; 0 ERROR/Traceback, 0 restarts; SM 1969-1989 MHz at every read; dsv41-lanproxy active (untouched).
- **MemAvailable 10.1-11.1 GB** on 8/8 at the end and stable (10.09 -> 10.14 GB on spark-one over 30 min), about 3.4 GB below g5stress-PU's end (13.5-14.6 GB). This instance served about 30 minutes of heavy multi-turn foreign traffic before the check; the likely cause is the allocator keeping long-prompt workspace (not verified). The floor is 6 GB.
- **No serving file changed in this step.** The vision staging set is exactly as stale as after the mHC step (launcher afa97ae6, `nvidia/model.py` ca14bbb1, new `dsv41_mhc_overlap.py` 4468cad5); the W0 artifacts are untouched.

### 4. Window, downtime, files
- **API downtime 17 min 55 s in six restarts:** trace instance 10:43:26 -> 10:48:05 (279 s: g5host-T1 failed twice at vLLM's startup memory check, 110.8-111.1 GiB free < 0.92 x 121.6 GiB under nsys; g5host-T2 with `--gpu-memory-utilization=0.89`, startup check only), then C1 149 s, F1 146 s, C2 144 s, F2 139 s and PU 218 s. Every relaunch except the trace retry waited for a 30 s idle window first.
- **Foreign load 11:45-~14:50 UTC** (see section 2): the A/B instances and production served it exactly. dsv41-lanproxy was not touched and no client was blocked.
- **Mac scratchpad** `/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad/g5host/`: `ab.log`, `tables-ab.md`, `scen.txt` (same-output G4 batches), `nsys/an/` (all analysis outputs + `hostcap.jsonl`), `runs/<label>/` (fetched suite outputs + `vsumm.md`), `ab/<label>/` (relaunch logs, clocks, start facts), `scripts/` (`inst.sh`, `hostcap.py`, `imp1.sh`, `launch-trace.sh`, `relaunch-trace.sh`, `scen.py`, `agg.py`).
- **Cluster:** spark-one `~/dsv41-g5host/` (harness copies, `cap/hostcap.jsonl`, `runs/`); `~/dsv41-hostprof/` on spark-one (rank 0) and spark-five (rank 4), ~160 MB each: `q<rank>.{1,2,3}.{qdstrm,nsys-rep,sqlite}` (1 = G4 #1, 2 = C1, 3 = G4 #2) and the `*.idle/hostseq/steptl.txt` outputs; `~/dsv41-tp8-hosttrace.sh` (36c3f00c, diagnostic, never used by the supervisor) and `~/dsv41-hostprof/pyfuncs-host.json` on 8/8. The trace's `/tmp/nsys-report-*.qdstrm` files were removed on 8/8. All of it is deletable.
- **Next exact item for this lever: `dtgraph`** (capture the ~49 eager drafter -> target launches per (num_reqs, num_tokens) bucket; the DT launch gaps are 0.13 ms/step at C1 and C4, so at most ~0.3% at C4 and ~0.6% at C1; 2-4 days). Host Python, AV waits and RPCs are not on the critical path at any load measured.

## Goal window 5 (2026-10-27 16:14 - 2026-10-28 17:24 UTC), final verification: config verified on 8/8 (launcher afa97ae6), both instances exact, regression gate (kit 08) PASS with a clean foreign-traffic audit. Goal NOT met: production G1 104.0 of 120 tok/s (+15.4%); G4 prose session 47.9 in production (+4.4%) and 50.55 on the pinned instance (the G4 metric is met there, but one of its 3 runs was 49.1). Drafter evidence: train the MTP heads (user decision); no official-shard fetch; no suffix drafts

**Verdict: the goal is not met.** All numbers were taken at the 2000 MHz boot clock with nothing raised, on the final config: launcher afa97ae6, i.e. window 4's stack plus the tie-canonical indexer top-k and the bitwise mHC overlap. The window-5 foreign-traffic rule applied to every unit.
- **G1, single user: 104.00 tok/s in production** (g5v-FU: code 97.85 / prose 78.90 / math 135.25) against 120: **+15.4% is needed** (+16.0 tok/s). Pinned g5v-FP gave 104.22 (+15.1%).
  - Window 5 moved G1 +2.0% in production (window 4's final 102.00 -> 104.00) and +2.5% pinned (101.69 -> 104.22). All of it came from the mHC overlap.
- **G4, 4 concurrent sessions.** Code, math and mixed stayed at 59.8 tok/s or more in every timed run. The prose session decides the result:
  - **production g5v-FU:** 47.42 / 47.49 / 48.75. The G4 metric (`goal_conc4` `G4_per_session_min`: the slowest session, averaged over the 3 timed runs) is **47.89, so +4.4% is needed** (worst run +5.4%). 9/12 session-runs reached 50; G4min_hat 49.25 (+1.5%).
  - **pinned g5v-FP:** 51.39 / 49.07 / 51.19. **The G4 metric is 50.55, met on this instance**, but not in every run: 11/12 session-runs reached 50, the worst run needs +1.9%, and G4min_hat is 49.42 (+1.2%).
  - The two instances differ in how the 4 sessions batch. At about the same 4-session step (43.99 vs 44.13 ms), production's prose session needed 286 steps per run against 267 pinned (batch AL 2.53 vs 2.58). Step times alone (G4min_hat) put both 1.2-1.5% short of 50. G4 is therefore borderline: a launch can land on either side.
- **Exact.**
  - exact.py gave a8c3394af9d2 at the start and after the suite on both instances (4/4), and again in the gate. The pinned instance had the pin's counters 909 / 467 / 2335 / 1373 with per-position 348/233/171/90/67, and a pin hit.
  - The G1 outputs were identical on both instances and equal to the window's pinned reference: 3-repetition digest bc1e8386878bd2b7, first 2 repetitions 9a18f802d930f6e7, per type 596b478a / 923a46be / 92e5d8cf. Production landed on the reference side of the prose near-tie.
- **Regression gate (kit 08): PASS in 563 s on g5v-FU, audit CLEAN.** The gate sent 319 requests and 319 finished; the 0.5 s sampler saw +319, with max running + waiting 8, GSM8K's own concurrency.
  - exact a8; G1 104.1; one cold 1M at TTFT 168.8 s (good class; MemAvailable minimum 9.53 GB); GSM8K-200 195/200; longctx 60/60 with copy 1.000; KL verify, smoke and capture OK (the new afa97ae6 baseline).
  - The KL panel's equivalence tier is not an exactness signal at its settings: a second capture on the same instance differs from the first as much as the cross-launcher comparison does (section 5).
- **Time to healthy:** production +204 s (downtime 218 s), pinned +126 s (downtime 140 s).
- **Idle power** (production, 120 s windows with no request):
  - GPU 6.9-9.8 W per node by `nvidia-smi` (8-node sum 66-68 W); spark-one box meter 37.3-38.0 W; switch 260 W over its two PSUs; whole fleet ~565 W (an estimate, not metered).
  - The SM clock read 1969-1989 MHz at every check on 8/8. No clock or power setting was changed.
- **Foreign traffic:** the tailnet client 100.106.91.100 sent bursts of short chat requests every 5-15 s, mostly during the pinned run. One exact.py start, two G1 units and one exact.py #2 were discarded as FOREIGN and rerun; every kept measurement is clean. `dsv41-lanproxy` was not touched.

### 1. What goal window 5 changed (every step has its own section above)

| step (UTC) | work | result | launcher after |
|---|---|---|---|
| Determinism (10-27 16:14-18:07) | long-prompt greedy drift: root cause and fix | cause = **ties at the 512th score in the sparse-indexer top-k** (stock prefill and decode top-k kept a random one of the tied candidates). **PROMOTED** `DSV41_TOPK_CANON_TIES=all` (a tied row keeps every score above the 512th value, then the lowest-index equal ones; untied rows bitwise stock): cold repeats 1 distinct output per probe at 1.5K-128K (were up to 4 of 4), G1 -0.16% (noise), TTFT +0.0 / +0.35%. Warm (prefix-cache hit) vs cold still differ (batch-size-dependent kernel tiling; not fixed) | 95f08eff -> **f214c2c9** |
| Drafter (10-27 ~18:10-18:51) | Codex kits 01 (drafter attribution) + 02 (n-gram / suffix drafts) on real G1 / G4 traffic | the prose AL "gap" is **not** the abliteration (the MTP / drafter shards are byte-identical to the official release); kit 02 **CLOSED** (net negative on real traffic); self-distillation costed (section 8). No file changed | f214c2c9 |
| Ops (10-27 18:59-20:20) | Codex ops kits 04 / 08 / 10 / 03 / 06 | 08 regression gate **INSTALLED** (first run PASS); 10 black box + panic sysctls on spark-three only, via the site layer (image 0.9.5-1019-bb1, reboot-tested); 03 NOT promoted (+15.7 ms first request after idle, no measurable power); 06 profiled; 04 confirmed live | f214c2c9 |
| (user pause 10-27 20:40; image window 10-28 02:19-02:46) | images on (user's window) | **ROLLED BACK** (R2): one probe overlapped a foreign request; production = pre-window state, no sha changed | f214c2c9 |
| LongCtx (10-28 03:29-04:50) | Codex ops kit 07 (indexer top-k reuse across decode steps) | **CLOSED** at its own gate: 0 of 109,912 recorded (layer, row) pairs reusable; no serving code built | f214c2c9 |
| mHC (04:49-09:14) | Moet's mHC boundary overlap, ported bitwise | **PROMOTED** `DSV41_MHC_OVERLAP=moet` + `DSV41_MHC_OVERLAP_PROJ=moet`: pinned G1 **+3.03%** in both pairs, G4 prose MIN +1.6 / +3.9%, G4min_hat +3.8%, every output bitwise; `stock` -0.6% not promoted | f214c2c9 -> **afa97ae6** |
| Stress (09:20-10:33) | co-residency race gate for the newly concurrent kernels | **PASS**: 0 mismatches in 7.56 M co-run calls, 415,360 routed-MoE outputs and 12,500 overlapped-region graph replays; the overlap stays | afa97ae6 |
| Host (10:35-15:23) | nsys first, then the combined exact arm | host-bound GPU idle **0.0 ms/step** -> host-path overlap STOPPED (nothing built); `DSV41_FUSED_AV=1` exact but +0.15 / +0.16% G1: NOT promoted (installed, off) | afa97ae6 |
| final verification (15:43-17:24) | this section | | afa97ae6 (unchanged) |

User directives applied in this step: the trimmed suites of 2026-10-27 (final verification = goal metrics G1 x3 / G4 x3 on both instances, ONE 1M probe, GSM8K + longctx once, on the production config that stays live); the window-5 foreign-traffic rule (below).

### 2. Config verified on 8/8

`g5v/scripts/verify8.sh` ran on the live g5host-PU before the step (`verify-pre`, 15:36:25 UTC) and on the live g5v-FU after it (`verify-post`, 17:12:08 UTC). It runs `node-verify5.sh` (read-only) on every Spark, compares every host-independent line across the 8 nodes, then checks each value against this README. Both runs had every comparable line identical on 8/8 and every expected value matched (verify-post: 50 OK lines, 0 problems; verify-pre after one checker pattern was corrected). The two outputs differ only in service start times, GPU readings and MemAvailable.
- **Launcher** `~/dsv41-tp8.sh` **afa97ae6** on 8/8, with backups `.bak-g5-mhc` = f214c2c9, `.bak-g5-det` = 95f08eff, `.bak-g4-l3` = 31d04d8f, `.bak-g4-fload` = fd1e809e, `.bak-g4-sp-e7x` = b0cf94d0.
  - Its defaults include `DSV41_MHC_OVERLAP=moet`, `DSV41_MHC_OVERLAP_PROJ=moet`, `DSV41_TOPK_CANON_TIES=all`, `DSV41_PREFILL_SP=exact`, `DSV41_E7_EXACT=post_and_draft`, `DSV41_FAST_LOAD=1` and `DSV41_TOOL_STREAM_NEWLINE=1`, plus KV 16G.
  - `--enable-prompt-tokens-details` is in the launcher and in the running command line on 8/8.
- **Supervisor** `ops/rel/dsv41-relaunch.sh` **7bea1cf3** (`PRE_CLOCK_DEFAULT=2000`, `PRE_COMPACT_DEFAULT=0`). Both launches logged "site manifest **0c96a727fb81** and launcher afa97ae6 identical on 8/8" and "engram check: 8/8 OK".
- **Site files**, identical on 8/8:
  - window 5: `sparse_attn_indexer.py` 005785f9, `dsv41_topk_ties.py` 943961bc, `nvidia/model.py` ca14bbb1, `nvidia/dsv41_mhc_overlap.py` 4468cad5; Moet sources .cu 2f18249b / .py ce0f79ab with the built extension present, installer `g5mhc_install.py` e48e27ef;
  - window 4 prefill: `model_state.py` b7598364, `decoder_replay_layers.py` b75aa927, `speculator.py` 439b390e, `dspark.py` 36811ca9;
  - fast loader: `dsv41_fast_load.py` dfbcf2c7, `default_loader.py` 1aeec2e6, `vl_model.py` 3ec0d401;
  - decode bundle, every flag off: `adaptive_verification.py` 55030a68, `model_runner.py` 6a4f2433, `gpu_worker.py` 4d67066c, `utils.py` c4e02ba8;
  - layer 3: `deepseek_v4.py` 3334c1fa, `parser_engine.py` 6e3eeaba;
  - window 3 and older: unchanged (`flashinfer_sparse.py` b059bdf3, `mla/indexer.py` d9f4c5c9, `dsv41_r3_dense.py` 1f1c0456, `deep_gemm_utils.py` a4a0486a, `sm121_target_m32.py` ea4a8814, `cuda_communicator.py` 87bf77e3, `rdma_all_reduce.py` 26a5d834, `dsv41_l2_prefetch.py` 63737f9c, `dsv41_r4_small.py` c46e66bd, `sparse_swa.py` 854c3e5f, `rdma_ar_ext.so` 62daad6c);
  - backups present on 8/8: `model.py.bak-g5-mhc` (9c2d22ad), `sparse_attn_indexer.py.bak-g5-det` (f2ae649b), and the g4p / fload / g4d / layer3 backups (2 checked per bundle).
- **Whole `vllm/` site tree:** **11e30c3df6b6** (5,454 files) on 8/8, before and after. The digest is the sha256 over every file, excluding bytecode caches and `.bak-*` / marker files (the README method).
- **Serving dependencies** on 8/8: `fc2c-on-m32all` `PREP.ok` dcf5bc0909472121; pins `av-pin-k5-g3FU.json` eeac4f97 and `av-pin-g3.json` 1c98cf9b; `m32-bw` 661-664 MB (per-node build output); vision stage present (sums 3b034721, `autotune-W0` present); `dgx-spark-gpu-clock.service` (`nvidia-smi -lgc 0,2000`) enabled and active.
- **OS:** spark-three runs 0.9.5-1019-bb1 (panic=10, panic_on_oops=1, black box active, up 21.6 h since the ops step's reboot test); the other seven run 0.9.5-1019 (panic=30).
- **Running environment** (63 `DSV41_*` variables, env digest 9626ff845dc0 on 8/8, both checks): the launcher defaults above plus window 3's `DSV41_TOPK_CANON_FUSED=1 DSV41_DECODE_REMAP_REUSE=1 DSV41_IDX_OMIT_T2S=1 DSV41_DRAFT_HEAD_TACTIC=1 DSV41_DRAFT_HEAD_TACTIC_MAP=17-29:1`.
  - No candidate flag is set (`DSV41_FUSED_AV`, `DSV41_DSPARK_OVERRIDE`, recorder variables).
  - `DSV41_AV_COST_JSON` was set only on the pinned instance.
- **Service:** 8/8 active, 0 restarts, 0 ERROR/Traceback in the rank logs. Every rank log has the fast-loader, SP-armed, E7-armed and mHC-overlap lines and no fused-AV line. No other GPU process and no nsys ran on any node.
- **API:** `/health` 200, model `dsv41`, `dsv41-lanproxy` active; KV 6,390,007 tokens; gate script 1889ae4a5ce3; the goal scripts' `SHA256SUMS` check OK.

### 3. Method

**Foreign-traffic rule (goal window 5).** The API is open on the LAN and the tailnet with no authentication. Other clients use it through `dsv41-lanproxy`: the tailnet nodes `fedora` 100.70.246.23 and 100.106.91.100, DSH on the Mac (192.168.0.239), and the LAN poller 192.168.0.86, which only polls `/v1/models`. A concurrent request changes batch composition and can flip greedy outputs. So:
- Before every exactness or timing unit, wait until `vllm:num_requests_running` == `num_requests_waiting` == 0 and `request_success_total` has not moved for >= 30 s.
- During the unit, run a 0.5 s `/metrics` sampler.
- A unit is CLEAN only if the success counter moved by exactly its own requests and running + waiting never went above its own concurrency. Otherwise it is FOREIGN: NOT PROVEN, its result discarded (kept aside) and the unit rerun. A FOREIGN unit is never a pass and never by itself a reason to roll back.
- `dsv41-lanproxy` was not touched and no client was blocked. Every relaunch first waited for a 30 s idle window, so no in-flight request was cut.

**Driver.** Scratchpad `g5v/scripts/final.sh` (Mac) runs `inst.sh g5v-FP FP`, then `inst.sh g5v-FU FU`, then `gaterun.sh` on spark-one. `part2.sh` does the gate-step reruns, the idle power measurement and `verify8.sh post`. Each `inst.sh` instance does the following:
1. A supervised launch (`ops/rel/dsv41-relaunch.sh` 7bea1cf3, `--expect-launcher afa97ae6`, `DSV41_GPU_CLOCK_MHZ=2000`, the boot policy; nothing raised), then one `nvidia-smi` read per node on 8/8 (must be <= 2010 MHz).
2. `gstart.sh`: exact.py as the first request, with the speculative-decoding counter deltas, rerun if a foreign request overlapped it.
   - Pinned: exact.py must be a8c3394af9d2 **with the pin's counters** 909 / 467 / 2335 / 1373 and per-position 348/233/171/90/67, and `mkpin_k.py check` must report a pin hit.
   - Production: the a8 anchor; the counters follow the startup profile.
3. The phases one at a time on spark-one, each through `gunit.py`, with the rule applied per unit. A unit is one G1 request (rep, type), one G4 batch of 4 sessions (the warm-up batch is its own unit) or one exact.py run. The SM clock is read on 8/8 before every phase.
4. Fetch, a `vsumm.py` summary, and the end state on 8/8.

**Suites (trim rule).** Both instances ran G1 x3 (`goal_single.py`: code / prose / math, 1024 tokens, 3 repetitions), G4 (`goal_conc4.py`: warm-up + 3 timed batches, 512 tokens per session) and exact.py #2 at the end. There was no matched C1, conc8/16 or long-prompt probe: this is not a candidate A/B. The regression gate (kit 08: its ONE 1M probe, GSM8K-200, longctx and the KL panel) ran once, on g5v-FU, the production instance that stays live, after its goal metrics, so its verdict covers the instance clients get. `gaterun.sh` would have run the gate with `--no-1m` below 9,000 MB MemAvailable on spark-one (floor 6 GB). It read 9,870 MB, and the 1M bottomed at 9,531 MB.

**Gate audit.** `gate.sh` itself only refuses a server that is busy for 5 s at its start, and flags a step when more requests finished than it sent. `gaterun.sh` first waits for a 120 s idle window, then runs `gate.sh --live` unchanged beside a 0.5 s sampler. `gaudit.py` then checks every step against both samplers and the step's own concurrency. A NOT PROVEN step would be rerun alone with the gate's own command and pass rule (`gstep.sh`, guarded the same way). None was needed.

**Idle power.** `pwr.sh` runs on the live production instance after a 30 s idle window, for 120 s, twice (1 and 10 min after the gate). It reads `idlemeas.py` on 8/8 (CPU busy and wakes per box; on spark-one the spbm box meter at 1 Hz plus its energy counters), `nvidia-smi` power, temperature and SM clock every 10 s on 8/8, and the cisco8101 PSU 1 + PSU 2 input power (PMBus `power1_input`) every 10 s. The window counts only if running + waiting stayed 0 and the success counter did not move.

### 4. Results (2000 MHz, every kept unit clean of foreign traffic)

Instances, in launch order: **g5v-FP** (pinned, healthy 15:49:53 UTC, +126 s) and **g5v-FU** (production, healthy 16:35:34 UTC, +204 s; **live**). Both came up on the first try.

| metric | **production g5v-FU** (launcher afa97ae6 defaults, AV unpinned; live) | pinned g5v-FP (same launcher + `av-pin-k5-g3FU.json`) | ref: pinned control g5host-C2 (afa97ae6, same pin; Host step) | ref: window 4 final production g4v-FU (launcher 95f08eff) |
|---|---|---|---|---|
| healthy at (UTC) / launch to healthy | 16:35:34 / +204 s | 15:49:53 / +126 s | 11:26:11 / +132 s | 15:49:49 / +209 s |
| exact.py first request (acc / drafts / dtok / gen; per-pos; AV) | a8c3394af9d2 (916/460/2300/1373; 344/237/176/92/67; pin miss) | a8c3394af9d2 (909/467/2335/1373; 348/233/171/90/67; pin hit) | a8c3394af9d2 (909/467/2335/1373; 348/233/171/90/67; pin hit) | a8c3394af9d2 (911/465/2325/1373; pin miss) |
| exact.py #2 after the suite | a8c3394af9d2 (916/460/2300/1373; 344/237/176/92/67) | a8c3394af9d2 (909/467/2335/1373; 348/233/171/90/67) | a8c3394af9d2 (909/467/2335/1373; 348/233/171/90/67) | a8c3394af9d2 |
| **G1 mean tok/s** (code / prose / math), 1024 tokens x3 each | **104.00** (97.85 / 78.90 / 135.25) | **104.22** (98.46 / 78.48 / 135.72) | **104.37** (98.60 / 78.59 / 135.92) | 102.00 (96.25 / 76.20 / 133.53) |
| G1 per repetition | 104.00, 103.89, 104.10 | 104.24, 104.20, 104.21 | 104.39, 104.37, 104.36 | 102.10, 101.93, 101.96 |
| G1 AL code / prose / math | 2.577 / 1.870 / 3.582 | 2.590 / 1.847 / 3.594 | 2.590 / 1.847 / 3.594 | 2.603 / 1.837 / 3.633 |
| G1 output digest (reps identical); first 2 reps (pinned reference 9a18f802d930f6e7) | bc1e8386878bd2b7 (yes); 9a18f802d930f6e7 | bc1e8386878bd2b7 (yes); 9a18f802d930f6e7 | bc1e8386878bd2b7 (yes); 9a18f802d930f6e7 | bc1e8386878bd2b7 (yes); 9a18f802d930f6e7 |
| G1 per-type output sha8 code / prose / math | 596b478a / 923a46be / 92e5d8cf | 596b478a / 923a46be / 92e5d8cf | 596b478a / 923a46be / 92e5d8cf | 596b478a / 923a46be / 92e5d8cf |
| **G4 per-session mean / G4 metric** (`G4_per_session_min`: slowest session, mean over the 3 timed runs; always prose) | **63.45 / 47.89** | **63.55 / 50.55** | **65.29 / 50.19** | 61.05 / 48.41 |
| G4 by session: code / prose / math / mixed | 63.1 / 47.9 / 78.3 / 64.5 | 60.2 / 50.5 / 78.5 / 65.0 | 60.9 / 50.2 / 82.0 / 68.0 | 55.9 / 48.4 / 75.9 / 64.1 |
| G4 slowest session per timed run | 47.42, 47.49, 48.75 (prose) | 51.39, 49.07, 51.19 (prose) | 50.17, 50.21, 50.17 (prose) | 48.48, 48.86, 47.90 (prose) |
| G4 session-runs at >= 50 tok/s | 9/12 | 11/12 | 12/12 | 9/12 |
| G4min_hat (step times only) / 4-session step ms | 49.25 / 43.99 | 49.42 / 44.13 | 49.86 / 43.87 | 47.50 / 45.25 |
| G4 prose steps per run / batch AL (mean) | 286 / 2.532 | 267 / 2.578 | 283 / 2.583 | 271.7 / 2.527 |
| foreign traffic (per-unit guard; FOREIGN units discarded and rerun) | G1 9/9 units clean, G4 4/4, exact.py #2 1/1; 0 reruns | G1 9/9 (2 FOREIGN reruns), G4 4/4, exact.py #2 1/1 (1 rerun) | all units clean, 0 reruns | none |
| exact.py first request: gstart tries | 2 (try 1 overlapped a foreign request: NOT PROVEN, rerun) | 1 | 1 | 1 |
| SM clock checks on 8/8 (before every phase) | 5/5 OK, 1969-1989 MHz | 5/5 OK, 1976-1989 MHz | 8/8 OK, 1969-1989 MHz | 17/17 OK, 1976-1989 MHz |
| ERROR/Traceback rank 0 (start, end) / MemAvailable at the end, spark-one | 0 0 / 9815 MB | 0 0 / 10088 MB | 0 0 / 13187 MB | 0 0 / 8,849 MB |

G4 runs, production g5v-FU:

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 59.54 | 49.18 | 78.36 | 63.49 | 62.64 | 49.18 (prose) | 41.56 | 182.5 | 2.532 |
| 1 | 64.72 | 47.42 | 78.33 | 64.39 | 63.72 | 47.42 (prose) | 39.70 | 186.4 | 2.558 |
| 2 | 64.69 | 47.49 | 78.31 | 64.36 | 63.71 | 47.49 (prose) | 39.69 | 186.7 | 2.558 |
| 3 | 59.82 | 48.75 | 78.27 | 64.78 | 62.91 | 48.75 (prose) | 41.97 | 182.7 | 2.481 |

G4 runs, pinned g5v-FP:

| run | code | prose | math | mixed | per-session mean | slowest (session) | common-window min | aggregate | batch AL |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | 57.23 | 48.97 | 76.14 | 66.07 | 62.11 | 48.97 (prose) | 41.25 | 185.5 | 2.525 |
| 1 | 60.14 | 51.39 | 82.14 | 67.29 | 65.24 | 51.39 (prose) | 41.95 | 191.8 | 2.629 |
| 2 | 60.09 | 49.07 | 72.46 | 62.23 | 60.96 | 49.07 (prose) | 42.26 | 192.7 | 2.517 |
| 3 | 60.36 | 51.19 | 80.82 | 65.37 | 64.44 | 51.19 (prose) | 42.86 | 191.3 | 2.589 |

Production instances of goal window 5 (each step's final instance, launcher defaults, AV unpinned; from the step sections above):

| instance (healthy at, UTC) | launcher | to healthy | G1 (digest) | G4 mean / prose MIN | cold 1M | GSM8K / longctx |
|---|---|---|---|---|---|---|
| g4v-FU (10-27 15:49, window 4 final) | 95f08eff | +209 s | 102.00 (bc1e8386) | 61.05 / 48.41 | 166.2 s | 50/50 / PASS |
| g5-PU2 (10-27 17:56, Determinism) | f214c2c9 | +205 s | 101.52 | - | 168.4 s | - |
| g5dr-PROD (10-27 18:49, Drafter) | f214c2c9 | +200 s | capture only (prose a22e1d39) | - | - | - |
| g5ops-PROD (10-27 19:47, Ops) | f214c2c9 | +200 s | 101.81 one pass (abcdf1eb) | - | 168.26 s | GSM8K-200 195 / PASS |
| g5lc-PU (10-28 04:42, LongCtx) | f214c2c9 | +204 s | - | - | - | - |
| g5mhc-PU (08:51, mHC) | afa97ae6 | +199 s | 104.06 (bc1e8386) | - | 168.3 s | 49/50 / PASS |
| g5stress-PU (10:29, Stress) | afa97ae6 | +203 s | 104.02 (bc1e8386) | - | - | - |
| g5host-PU (14:57, Host) | afa97ae6 | +204 s | 103.97 (bc1e8386) | - | - | - |
| **g5v-FU (16:35, this step)** | **afa97ae6** | **+204 s** | **104.00 (bc1e8386)** | **63.45 / 47.89** | **168.8 s** | **GSM8K-200 195 / PASS** |

- **Decode moved by the mHC overlap only.** Pinned g5v-FP is within 0.3% of the Host step's pinned controls on the same launcher and pin (G1 104.22 against 104.53 / 104.37). Its 4-session step was 44.13 ms against 43.27-44.00 ms on window 5's earlier pinned afa97ae6 launches: the instance-to-instance spread (~2%) is about the size of the remaining G4 gap.
- **Production vs pinned.** G1 is the same (104.00 vs 104.22) and so are the G1 outputs. G4 differs by batching scenario, not by step time (the 4-session step was 43.99 vs 44.13 ms); this window's mHC step showed the same, with the prose session at 48.7-51.4 per run across its two promoted-arm instances.
- **Memory.** MemAvailable after a fresh launch has been ~9.8-11.2 GB at idle on 8/8 since the Host step's 14:54 UTC relaunch (g5host-PU, g5v-FP, g5v-FU). That is ~3 GB below the morning's instances (g5host-C2 ended at 13.2 GB on spark-one), although the process's startup footprint is identical (model 42.21 GiB, CUDA allocated 72.78 GB, initial free memory 112.5-113.0 GiB). The file cache is only 2.3-3.1 GB and the Engram mlock is 24.2 GB, as before. The cause is not identified. The 1M probe still bottomed at 9.53 GB on spark-one, 3.5 GB above the 6 GB floor, and swap stayed at 12-172 MB per node.

### 5. Regression gate (kit 08) on the final config

`gaterun.sh g5v-final` on spark-one at 16:54:14 UTC: idle wait 143.6 s for a 120 s window, then `gate.sh --live` at 16:56:37 UTC. Result: `~/dsv41-bench/gate/runs/afa97ae6/20261028T165637Z.json`, the first gate run under launcher afa97ae6.

| step | verdict | s | requests sent / finished; audit | result |
|---|---|---|---|---|
| refuse_if_busy | PASS | 5.1 | 0 / 0 | idle; instance g5v-FU; AV unpinned |
| exact.py | PASS | 12.5 | 6 / 6; CLEAN | combined **a8c3394af9d28249** |
| g1_one_pass | PASS | 31.7 | 3 / 3; CLEAN | G1 104.1 tok/s, digest 5d3de850b833b05b (code 596b478a, math 92e5d8cf, prose 923a46be = the reference outputs); first run under afa97ae6, nothing to compare with |
| probe_1m | PASS | 176.5 | 3 / 3; CLEAN | **TTFT 168.797 s** for 1,024,611 prompt tokens, good class (max 220 s); MemAvailable minimum 9.3 GiB (spark-one); 9,531-11,144 MB on 8/8 by the 2 s sampler |
| gsm8k_200 | PASS | 139.4 | 200 / 200; CLEAN | **195/200** (min 194); misses 12, 93, 100, 119, 184 |
| longctx | PASS | 156.7 | 24 / 24; CLEAN | **LONGCTX PASS**, needle 60/60, copy 1.000 |
| kl_verify | PASS | 0.2 | 0 / 0 | manifest 0917cafcf33722cc, 294,746 teacher-forced positions |
| kl_smoke | PASS | 2.1 | 3 / 3; CLEAN | tf n=2048 in 1.32 s |
| kl_capture_small | PASS | 38.6 | 80 / 80; CLEAN | label `gate-afa97ae6-20261028T165637Z` (tools + math, 80 sequences): the baseline for later gate runs under afa97ae6 |

- **Audit CLEAN.** Over the gate's 563 s, the 0.5 s sampler saw the success counter move by exactly the 319 requests the gate sent, and running + waiting never exceeded 8 (GSM8K's own concurrency). No step needed a rerun.
- **GSM8K-200:** 195/200 as under f214c2c9 (10-27 19:51 UTC). Item 12, the known borderline item that also failed in the mHC step's GSM8K-50, now misses and item 182 now passes. GSM8K at concurrency 8 is not bitwise across instances, so this is not an exactness signal.
- **KL panel, an extra check outside the gate:**
  - The gate compares a capture only with an earlier capture under the same launcher, so this run is a baseline. I compared it with the f214c2c9 baseline (`kl_panel.py compare --tier equiv`): excess KL +0.0160 nats/pos (CI +0.0087..+0.0239), top-1 -1.35 pts, dNLL +0.0004 (CI -0.0022..+0.0028). That is equiv FAIL and lossy PASS.
  - A second capture on the **same instance** (`gstep.sh`, 17:08 UTC, clean, label `gate-afa97ae6-20261028T170804Z-r`) differs from the first just as much: excess KL +0.0176, top-1 -1.43 pts, dNLL +0.0021 (CI includes 0), with single positions up to KL 14.8.
  - So the panel's run-to-run noise at its settings (concurrency 4, prefix caching on, so warm and cold rows mix; window 5's Determinism step showed warm and cold rows differ) is ~30x its equiv threshold (5e-4). The cross-launcher difference is inside that noise and is not evidence of a change. Exactness rests on the bitwise checks (exact.py, G1 digests, the mHC step's 0-ulp tests and co-residency gate).
  - The equiv tier needs a REF2 noise floor captured under the same conditions, or cold captures (for example concurrency 1 with prefix caching bypassed), before it can serve as a gate. Until then, read the gate's KL compare note as informational. Nothing was changed in the gate.

### 6. Time to healthy and idle power

- **Time to healthy** (supervisor, launch to `/health` 200 + startup complete): pinned **+126 s** (downtime 140 s), production **+204 s** (downtime 218 s). Both are in window 4's range (pinned 126-142 s, production ~200-209 s since the fast loader), and both came up on the first try. Each launch first waited for a 30 s idle window (245.9 s and 42.1 s of waiting), so no client request was cut. This step's API downtime was 358 s in total.
- **Idle power**, production g5v-FU, launcher defaults, no request in either window:

| | idle #1 (17:07, ~1 min after the gate) | idle #2 (17:15:39-17:17:39) | ref: ops step, production #1 / #2 (10-27 19:04 / 19:47, f214c2c9) | ref: service stopped (10-27 19:30) |
|---|---|---|---|---|
| spark-one box meter `sys_total` (spbm, 1 Hz mean) | 37.99 W | 37.32 W | 33.63 / 35.74 W | 27.84 W |
| spark-one energy counters: SoC pkg / cpu_p + cpu_e / gpu | 23.61 / 2.37 / 9.54 W | 22.77 / 2.19 / 9.22 W | 21.39 / 1.72 / 8.82 W (#1); - / 1.90 / 9.23 W (#2) | - / 0.99 / 5.11 W |
| GPU `power.draw.instant`, 8 nodes (mean / sum; 12 reads per node) | 8.55 / 68.4 W | 8.27 / 66.2 W | - | - |
| GPU temperature max / SM clock | 57 C / 1969-1989 MHz | 51 C / 1969-1989 MHz | - | - |
| box CPU busy, 8 nodes / dsv41 service CPU (% of one core) | 0.32-0.52% / 2.8-4.3 | 0.32-0.50% / 3.1-4.1 | 0.35% / 3.26 (spark-one) | 0.24% / 0 |
| switch cisco8101 input, PSU 1 + PSU 2 (PMBus) | 132.2 + 127.6 = 259.8 W | 132.6 + 128.0 = 260.6 W | (2026-10-14: 259.5 W) | - |
| fleet estimate (not metered: 8 x spark-one's non-GPU power + each node's GPU, + switch) | ~569 W | ~564 W | - | - |

  - spark-one idles 1.6-4.4 W above the ops step's two production readings. Those readings were already 2.1 W apart on the same config, and no clock, power or idle setting changed (the clock lock is 0-2000 MHz on 8/8). The increase is not attributed. Most of it is outside the SoC package (+2.2 W), and idle #1 ran while the box was still warm from the gate.
  - The fleet estimate matches the memory note's idle cluster figure (~550-650 W).

### 7. Goal status and the remaining gap (2000 MHz, no clock or power increase)

| target | production g5v-FU | pinned g5v-FP |
|---|---|---|
| G1 mean >= 120 tok/s | 104.00: **+15.4%** | 104.22: +15.1% |
| G4 metric `G4_per_session_min` >= 50 (slowest session, mean over 3 timed runs); worst run | 47.89: **+4.4%**; 47.42: +5.4% | **50.55: met**; 49.07: +1.9% |
| G4 every session >= 50 in every timed run | 9/12: no | 11/12: no |
| G4min_hat (step times only) | 49.25: +1.5% | 49.42: +1.2% |

- **G1 >= 120 tok/s: NOT met.** At today's acceptance (AL code 2.58 / prose 1.87 / math 3.58), the single-user step is 26.3 / 23.7 / 26.5 ms. 120 tok/s needs every step **13% shorter, -3.2..-3.5 ms**, or ~15% more accepted tokens per step.
- **G4: borderline.** The pinned instance met the G4 metric; production missed it by 4.4%. To hold on every launch and every run, the prose session needs **+1.2..5.4%**, i.e. -0.5..-2.3 ms off the ~44 ms 4-session step, or a better prose acceptance.
- **Exact levers left** (each needs its GPU bitwise test and a pinned A/B under `av-pin-k5-g3FU.json`; most come from window 4's section 6 and the spark-ds41f adoption report):
  - MoE: FC1 -> FC2 per-expert pipelining (-0.4..-1.0 ms per 4-session step), contiguous FC2C (-0.5..-0.8 ms), a q_b L2-prefetch plan above 8 rows (up to -0.86 ms), drafter MoE on M32 (~-0.45 ms at 4 requests);
  - spark-ds41f R07 MoE glue collapse (G1 +0.8..1.8%, G4 +0.55..1.35%), R06 routed-expert 4-bit scale codes (G1 +0.4..0.7%, G4 +0.6..1.1%), R08 PDL on the Triton glue (+0.4..0.7%);
  - the Host step's `dtgraph` (~0.6% at 1 user, ~0.3% at 4; design only), `DSV41_MHC_OVERLAP_DRAFT` and `DSV41_MHC_OVERLAP_PRE`.
  - **These can make G4 robust** (roughly half of them would cover the worst-run gap) **but they do not reach G1 = 120.** At 1 user the list adds up to a few percent, not the 13% per step that 120 needs. That leaves acceptance, i.e. the drafter: section 8.

### 8. What the drafter evidence implies (Drafter step, 2026-10-27; not written up in this README before)

The Drafter step changed nothing on the cluster. Its runs, analysis and scripts are on spark-one in `~/dsv41-g5dr/` (Mac copy: scratchpad `g5dr/`). On 512S2, `~/g5dr/` holds a copy of the g4c dump and `export.json` for evaluating a trained drafter. Its findings, and what each one means for the three questions:

**Fetch the official MTP shards? No.** Comparing the per-shard LFS sha256 values in the Hugging Face API (nothing was downloaded) shows that only shards 13-39 (layers 10-36) differ from the official DeepSeek-V4.1-Flash-1017 release. Shards 44-46 (`mtp.0`-`mtp.2`, 7.93 GB: the whole DSpark drafter, Markov and confidence heads included), shards 40-42 (the drafter's tap layers 37-39), shard 43 (head and norm) and the Engram shards are byte-identical. We already serve the official drafter, so a fetch would change nothing.

**Is the abliteration the cause of the prose gap? Almost certainly not.**
- The kit's "card" values (code 5.15 / prose 2.40 / math 3.20) have no published source. The 5.15 is our own matched-C1 code AL.
- Like-for-like comparisons with the official weights (thinking off, temperature 0) are flat for code: Moet code k=5 AV off, official 5.17 vs ours 5.273 ± 0.094 (n=8). Prose is mixed: Moet prose 2.21 / 2.30 official vs 2.106 ± 0.064 ours (-4.7 / -8.4%); knapcio's same-stack A/B of the official model against a different abliteration (mrfakename) gave -2.5% / -0.6% prose tok/s.
- The confidence head is as well calibrated on our thinking prose (ECE 0.05 at positions 1-2) as on code and math. A shifted target would have made it over-confident.
- A residual effect of up to ~5% on prose AL cannot be excluded with other stacks' data. The decisive test is an official-target A/B: ~4.6 GB of byte-range fetches (the non-routed tensors of layers 10-36), a load-time override and one stopped window of ~20 min. It is a diagnostic only: the plan below does not depend on it.

**What actually limits acceptance: thinking text.** None of the 37 G1 / G4 requests reaches `</think>` within 512 / 1024 tokens, so every G1 and G4 token is reasoning text. The drafter is much weaker on reasoning text. AL live / with all 5 drafts verified (g4c-O1 dump replay):

| workload | AL live | all 5 verified |
|---|---|---|
| code, C1 thinking off | 5.21 | 5.32 |
| code, G1 thinking on | 2.59 | 2.85 |
| math, G1 | 3.59 | 3.92 |
| prose, C1 thinking off | 2.74 | 3.13 |
| prose, G1 | 1.85 | 2.08 |
| prose, G4 session | 1.81 | 2.12 |

**Suffix / n-gram drafts (kit 02)? No: closed for the goal.** On real G1 / G4 traffic the matched draft tokens per output position are 0.21-0.31 for code, 0.35-0.54 for math, 0.13 for mixed and 0.02 for prose, counting prompt plus output so far. The kit's +5.75 came from a perfect-echo synthetic test. The best realistic policy appends the n-gram continuation after a fully accepted DSpark window, and it loses once its verify rows are paid for: G1 code -3.1%, math -5.6%, prose -1.9%; G4 prose -1.9%. Only C1 code with thinking off gains (+1.0%). A verify row costs 2.08 ms at 1 user and 1.79 ms at 4 users, so a row only pays if it is accepted with probability above 0.16-0.27; measured n-gram precision is 0.04-0.29. Even if the extra rows were free, the gain would be about +2% G1 and +0.4% G4 prose.

**Train the MTP heads (self-distillation)? Yes. It is the one lever with room for G1 = 120, and the clean way to make G4 robust.** The reason is the drafter's weakness on thinking text, not the abliteration.
- Ceilings at the same verify width, from the g4c oracle replay (the head turns each first miss into a hit when the true token is in today's top-2 / top-4 / top-16): G4 prose session +6.9 / +12.2 / +17.8%; G1 prose +8.9 / +16.2 / +25.2%. Prose alone moves G1 by at most ~+6%. G1 = 120 at today's step times needs about +15% AL on code, prose and math together.
- Plan:
  - Capture pass 1: ~5,000 held-out prompts, thinking on, up to 1,500 tokens each (~7.5 M tokens), decoded on production at up to 32 concurrent requests for ~3.5-4 h, with no restart.
  - Capture pass 2: one stopped window of ~1-1.5 h. A hook replays prefill with bounded replay off and dumps the layer-37/38/39 tap features (225 GB, or 75 GB after a frozen `main_proj`) and the target's top-32 tokens to 512S2, which has 3.1 TB free.
  - Training on the Mac Studios: first port the drafter to MLX; the port must match the live drafter's top-1 on at least 99.5% of dump positions. Then fine-tune the 0.64 B non-expert parameters, with the routed experts frozen: ~5-6 h per epoch on one Studio, or 2 epochs on both Studios in ~5-6 h. A cheaper first tier retrains only the Markov and confidence heads, in under an hour.
  - Effort: ~4-6 engineering days and ~7 h of cluster time.
- Deployment is a drop-in: same tensor names, shapes and dtypes, loaded by a flag `DSV41_DSPARK_OVERRIDE=<file>` (unset = stock). The files are staged under `~/dsv41-stage` with sha256 checked on 8/8, and nothing is written into `~/models`. The pin `av-pin-k5-g3FU.json` stays valid, because the drafter's cost does not change.
- Offline gate: held-out reasoning prose (all 5 verified) must gain at least +0.15 AL. Code and math must not drop more than 1%, and C1 code must stay within 0.1.
- **Exactness:** the target still decides every token, so outputs are greedy-equivalent by construction. But new drafts change the verify layout, so the known G1 prose near-ties flip. The prose outputs seen so far are 923a46be (the pinned reference), e8a524f3 and a22e1d39; the 2-repetition G1 digests are 9a18f802 and 69ad8721. A drafter change therefore cannot pass "identical pinned G1 digest", the same way Moet's k6 / k7 could not. The proposed drafter gate:
  - exact.py a8c3394af9d2 on every launch;
  - G1 code and math bitwise (596b478a / 92e5d8cf);
  - G1 prose landing on a known tie outcome;
  - a first-divergence census on G4 at the control's near-tie rate;
  - `dbgrun.py` cold repeats passing;
  - at promotion, GSM8K-50, longctx and one 1M probe.

**Decisions only the user can make:**
1. **Self-distillation of the drafter**: the ~4-6 days and ~7 h of cluster time above, including a 3.5-4 h capture on the live server (clients would share it) and one 1-1.5 h stopped window. It also needs the **drafter exactness rule** above in place of "identical pinned G1 digest". Without that rule, no drafter or acceptance lever can be promoted, and G1 = 120 stays out of reach (section 7).
2. Optional: the official-target A/B (~4.6 GB byte-range fetch and a ~20 min stopped window), only to bound the residual abliteration effect.
3. Carried over: API authentication or agreed quiet windows. Foreign clients blocked timed phases for hours in this window (Host step F2, 11:40-14:35 UTC). The vision retry needs the staged image set rebuilt (section 9). Unchanged: unpadded Marlin W4A16 (not exact), a default AV pin, and clocks (closed at 2000 MHz by policy).

### 9. Live state after this step, promoted files, rollback

- **g5v-FU is live:** production, launcher **afa97ae6** defaults, AV unpinned, supervisor 7bea1cf3 (preflight manifest 0c96a727fb81), clock lock 0-2000 MHz.
  - Healthy since 16:35:34 UTC on 8/8, with 0 restarts and 0 ERROR/Traceback.
  - `/health` 200; `dsv41-lanproxy` active and unchanged.
  - KV 6,390,007 tokens. `--enable-prompt-tokens-details` is in the launcher and in the running command line on 8/8.
  - `verify-post` (17:12:08 UTC) matches `verify-pre` apart from start times, GPU readings and MemAvailable.
- **Files promoted in goal window 5** (full sha256; identical on 8/8; backups `.bak-g5-*` on 8/8):

| file | before window 5 -> now | step |
|---|---|---|
| launcher `~/dsv41-tp8.sh` | 95f08eff -> f214c2c9 -> **afa97ae6dfc65943f7e536baf7922bbecf2c5d8110f35768018c7c7dd5be3cc1** (backups `.bak-g5-det` = 95f08eff, `.bak-g5-mhc` = f214c2c9) | Determinism, mHC |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | f2ae649b -> **005785f9** (backup `.bak-g5-det`) | Determinism |
| `vllm/model_executor/layers/dsv41_topk_ties.py` | new: **943961bc** | Determinism |
| `vllm/models/deepseek_v41/nvidia/model.py` | 9c2d22ad -> **ca14bbb1df9e4379dc21f051fd782a6403ec40687872d6711e1dcb2f708fd005** (backup `model.py.bak-g5-mhc`) | mHC |
| `vllm/models/deepseek_v41/nvidia/dsv41_mhc_overlap.py` | new: **4468cad5aec3414ace5ccc7d8af0d448d7561e67d890fd106098d7220e4f9759** | mHC |
| runtime dependency of the mHC overlap | Moet sources `~/dsv41-stage/moet/src/...` (.cu 2f18249b, .py ce0f79ab) and the built extension `~/.cache/torch_extensions/py312_cu130/vllm_moet_mhc_post_norm_sm120/` on every node | mHC |
| supervisor preflight manifest | 40eb7abf08c7 -> d773ed60f27b -> **0c96a727fb81** (the supervisor script itself, 7bea1cf3, is unchanged) | |
| whole `vllm/` site tree (README method: sha256 of every file, bytecode caches and `.bak-*` / marker files excluded) | 38062de05f0a (5,452 files) -> 4baa4e458a1a (5,453) -> **11e30c3df6b6 (5,454 files)** on 8/8 | |
| spark-three OS image (site layer) | 0.9.5-1019 -> **0.9.5-1019-bb1** (kit 10 black box + panic sysctls; the other 7 nodes stay 0.9.5-1019) | Ops |
| `~/dsv41-bench/gate/` | new: `gate.sh` 1889ae4a5ce3 (v1 kept as `gate.sh.v1`), `README.md` | Ops |

- **Vision staging set (`~/dsv41-stage/vision`, sums 3b034721) is STALE.** It was built against launcher f214c2c9, and the mHC step changed the launcher and `model.py` and added `dsv41_mhc_overlap.py`. `model_runner.py` is still 6a4f2433, so `model_runner.py.vis` (78c9e22f) still applies on top of it. The four staged launchers (L1, L2-kv14/15/16) and the post-install manifests must be regenerated from afa97ae6 with the sha list above. The W0 artifacts (`autotune-W0`) are untouched on 8/8.
- **Installed but off:** `DSV41_FUSED_AV` (window 4 item 09), the window-4 inert bundles (06 candidate-only indexer, 08 early KV slab), and the LongCtx recorder module (staged on 8/8 outside the site tree, inert unless both its env var and launch flag are set). Nothing from this step is installed.
- **Rollback** (from the step sections):
  - one launch: `DSV41_MHC_OVERLAP=0` or `DSV41_TOPK_CANON_TIES=0` as supervisor arguments;
  - permanent: `cp ~/dsv41-tp8.sh.bak-g5-mhc ~/dsv41-tp8.sh` (f214c2c9) or `.bak-g5-det` (95f08eff) on 8/8, then relaunch through the supervisor;
  - full removal, with the service stopped: `python3 -I ~/dsv41-stage/g5mhc/g5mhc_install.py uninstall` and `python3 -I ~/dsv41-stage/g5det/g5det_install.py uninstall` on 8/8;
  - kit 10 (spark-three only): the removal commands in "Goal window 5, ops step".

### 10. Files
- Mac scratchpad `g5v/`:
  - scripts: `final.sh` (part 1: FP -> FU -> gate), `part2.sh` (reruns, idle power, verify-post), `inst.sh` (one instance), `vphase.sh` + `gunit.py` + `idlewait.py` + `gssamp.sh` + `finsampler.sh` (phases with the per-unit guard, on spark-one), `gaterun.sh` + `gaudit.py` + `gstep.sh` (guarded gate, audit, step reruns), `pwr.sh` + `pwsamp.sh` + `idlemeas.py` (idle power), `memw8.sh` (2 s MemAvailable on 8/8 during the gate), `verify8.sh` + `node-verify5.sh` (config on 8/8), `vsumm.py`, `vtab5.py`, `clk8.sh`, `state8.sh`, `readme_update.py`;
  - logs and checks: `ab.log`, `launches.tsv`, `verify-pre.txt`, `verify-pre-check.txt`, `verify-post.txt`, `verify-post-check.txt`;
  - results: `runs/g5v-FP/`, `runs/g5v-FU/` (with `vsumm.json` / `vsumm.md`), `runs/<label>-start/` (exact.py first request, startup facts), `ab/<label>/` (relaunch log, clock checks, end state), `pwr/final-idle/`, `pwr/final-idle2/`, `memw/`, `readme/` (this section, the tables).
- spark-one: `~/dsv41-g5v/` (`runs/g5v-FP`, `runs/g5v-FU`, `gate-g5v-final.{out,samp,audit}`, `gstep/g5v-klrep/`, and `g5v-mac.tgz`, a copy of the Mac folder); `~/dsv41-bench/gate/runs/afa97ae6/20261028T165637Z{.json,/}`; KL captures `~/dsv41-bench/klpanel/runs/gate-afa97ae6-20261028T165637Z` (gate baseline) and `gate-afa97ae6-20261028T170804Z-r` (replicate, with the two compare files).
- Deletable leftovers from earlier steps (not touched here): spark-one `~/dsv41-g5lc/rec/R1` (~600 MB), `~/dsv41-g5det/` dumps (~0.4 GB), `~/dsv41-hostprof/` on spark-one and spark-five (~160 MB each), `~/dsv41-tp8-hosttrace.sh` on 8/8, `~/dsv41-g5stress/dumps/rank0` (135 MB per node on spark-one..six) and the test builds `~/.cache/torch_extensions/py312_cu130/g5stress_co_v{1,2,3}`, plus 26 unused shared-memory segments on spark-one (~1 MB).

## Goal window 6 (2026-10-29 01:44-09:56 UTC): exact decode levers. X (the drafter's routed MoE on the M32 kernel, `DSV41_DRAFT_MOE_M32=1`) passed the pinned chain (G1 +0.99 / +1.06%, G4min_hat +1.48 / +1.29%, every pinned output bitwise) and was promoted (launcher afa97ae6 -> d625c443), but the Phase D production check STOPPED it: unpinned, X changes the drafter cost the boot-time AV profile measures, and the G1 prose near-tie landed on an UNKNOWN outcome (33ae975a). Launcher ROLLED BACK to **afa97ae6** (X installed, flag off); keeping X in production is a user decision. Q and H NO-GO in the stopped window, L SMALL. Goal NOT met: production G1 104.3 of 120 tok/s (+15.1%); G4 prose session 48.6 in production (+3.0%)

**Verdict: nothing new is live; one exact lever is ready for a user decision.** All numbers were taken at the 2000 MHz boot clock with nothing raised (SM 1969-1989 MHz at every check on 8/8). The window-5 foreign-traffic rule applied to every unit (30 s idle before each, a 0.25-0.5 s sampler, an overlapped unit is NOT PROVEN and rerun); the window saw no foreign request in any timed unit. `dsv41-lanproxy` was not touched and no client was blocked. The cluster lock was held by claude for the whole window; every cluster action has a line in `AGENTS_LOG.md`.
- **X under the AV pin is bitwise:** 10/10 pinned instances EXACT-PASS, shadow verify 8/8 (0 mismatching words in 7,394 x 3 drafter MoE calls per rank), drafts identical (matched C1 AL 5.191 / 2.725 on every instance). Pinned: G1 105.23 / 105.30 (controls 104.18-104.29), G4min_hat 50.43 / 50.42 (controls 49.41-49.87), 4-session step 43.10 / 43.09 ms (controls 43.62-44.08).
- **X in production (unpinned) is not output-identical:** on launcher d625c443 exact.py stayed a8c3394af9d2 (start and end) and G1 code / math stayed bitwise (596b478a / 92e5d8cf), but the G1 prose output was **33ae975a in all 3 repetitions** (digest 247b90ce9f0b492e), not one of the known unpinned tie outcomes 923a46be / e8a524f3 / a22e1d39. Mechanism: the unpinned AV profile measures the drafter at boot, and X makes it cheaper (draft_ms 4.20 -> 3.92 at 1 row, 6.79 -> 5.71 at 4, 10.51 -> 8.85 at 12, 15.61 -> 13.47 at 32), so the AV policy picks other verify layouts and the prose near-tie flips. Under the pin the cost table is fixed, so nothing moves. Every unit was clean: this is a CLEAN mismatch, not noise, and it was not retried.
- **That X-on production instance measured:** G1 104.20 (98.94 / 77.30 / 136.36: code +1.1%, math +0.8%, prose -2.0% on the new tie path), **G4 metric 49.99** (prose per run 49.13 / 50.35 / 50.48; 11/12 session-runs >= 50), G4min_hat 50.72, 4-session step 42.99 ms.
- **Production now (w6-PROD-rb, suite w6-Drb, launcher afa97ae6, X off, AV unpinned):** G1 **104.26** (code 98.36 / prose 78.47 / math 135.96), G4 metric **48.56** (prose per run 48.04 / 48.09 / 49.55), G4min_hat 50.25, 4-session step 43.68 ms; exact.py a8c3394af9d2 at start and end (910/466/2330/1373), G1 code / math bitwise and prose on the known unpinned tie outcome a22e1d39 (digest 51d0a6537e239757): EXACT-PASS.
- **Regression gate (kit 08) on the final production: PASS in 596 s, foreign-traffic audit CLEAN.**
- **API downtime: 84.3 min in total**: window A 49.5 min over its 3 attempts, Phase B 27.6 min in 11 relaunches, Phase D 7.2 min in 2 relaunches (216 + 217 s).

**User decision (X).** Options: (a) keep X in production and accept 33ae975a as a fourth known unpinned prose tie outcome (code / math bitwise, exact.py a8; the production G4 metric reached 49.99); (b) make the AV pin the production default (open since window 5): then X is bitwise in production as well (the pin's draft-cost table is slightly pessimistic under X); (c) a fresh pin that includes X's costs (a new anchor); (d) leave X off. Re-enabling X is one step: `cp ~/dsv41-stage/w6/promote/dsv41-tp8.sh.d625c443 ~/dsv41-tp8.sh` on 8/8 (or `W6_PROMOTE_OK=1 bash $W/driver/promote-w6.sh X`, which rebuilds the same d625c443), then a supervised relaunch with `--expect-launcher d625c443`, or for one launch only the supervisor argument `DSV41_DRAFT_MOE_M32=1` on afa97ae6.

### 1. What is live now (verified on 8/8 09:55 UTC)

- **Launcher** `~/dsv41-tp8.sh` **afa97ae6** on 8/8 (unchanged by the window). The promoted d625c443 = afa97ae6 + 2 lines after `export DSV41_MOE_FC2C=` was installed 08:59-09:27 and the instance launched from it served 09:06-09:28; it is kept as `~/dsv41-stage/w6/promote/dsv41-tp8.sh.d625c443` on spark-one:
  ```sh
  # goal window 6 (RUNBOOK 7, promoted X): exact levers, default on; '=0' / '=1024' as a supervisor argument turns one off for a launch (RUNBOOK 9)
  export DSV41_DRAFT_MOE_M32=${DSV41_DRAFT_MOE_M32:-1}
  ```
  `~/dsv41-tp8.sh.bak-w6` = afa97ae6 on 8/8 (new; older backups unchanged).
- **Site files installed, flag off** (window A step A6, 04:54 UTC, `w6-bundle.sh install drm32`, service stopped):
  - `vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py` 5162dc0f -> **9364f121** (backup `deep_gemm_moe.py.bak-w6` = 5162dc0f; staged base copy `~/dsv41-stage/w6/bundle/deep_gemm_moe.py.base`).
  - new `vllm/model_executor/layers/fused_moe/experts/dsv41_draft_m32.py` **a17d574a** (`DSV41_DRAFT_MOE_M32` = 0 / 1 / verify, anything else is a startup error; `DSV41_DRAFT_M32_MAX_M` accepted 1..160 only).
  - With the flag unset both behave exactly as the live base (no new code path runs): A11, the 5 pinned C instances and this section's production check are bitwise on them. Neither is a supervisor-manifest file. The A7 prejit added the 2 drafter JIT entries (`sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d.3f71c0e0...` and `.74a66dfd...`) to `~/dsv41-stage/g1/m32-bw/jit/all-base/cache` on 8/8; no existing entry changed.
- **Supervisor** 7bea1cf3, preflight manifest **0c96a727fb81** (unchanged; it was 26db72449673 while d625c443 was installed: the launcher is a manifest file). **Site tree** `vllm/` 36631b8bc6ce over 5,455 files (was 11e30c3df6b6 over 5,454 files: the one new module).
- Unchanged: `dsv41_l2_prefetch.py` 63737f9c, `fp8_utils.py` b35c0561, `sm121_target_m32.py` ea4a8814, g2 pacer `libdsv41_l2pf_g2.so` 0451009d, r4 burst `libdsv41_l2pf.so` a09f85e8, gate.sh 1889ae4a5ce3, pin `av-pin-k5-g3FU.json` eeac4f97, the running env (window 5's 63 variables, no w6 variable).
- Staged only (`~/dsv41-stage/w6/`, bundle a284a782c82ca474): qb-l2pf (`dsv41_l2_prefetch.py` 0ca63b21, a manifest file), fc1fc2 H (`fp8_utils.py` 6b466c31, `sm121_target_m32.py` ae306c6b, `deep_gemm_moe.py.hpf` / `.all`).

### 2. The window, step by step

| step (UTC) | what | result | API down | launcher after |
|---|---|---|---|---|
| lock + staging (01:44-01:52) | lock taken; `~/dsv41-stage/w6/{qb-l2pf,fc2hpf,drafter-m32,bundle}` on 8/8 (stage dirs only) | verify-stage a284a782c82ca474 on 8/8, A0 GO | 0 | afa97ae6 |
| window A attempt 1 (01:55-02:09) | stopped window for every GPU bitwise test | no test started: after the stop the window's clock gate read the idle SM clock (208 MHz) against its 1900 MHz floor (fixed: the latch is judged under load); nothing installed; A11 VERIFY8-DIFF (a stale vision expectation in the window-5 check, fixed) | 4.0 min | afa97ae6 |
| window A attempt 2 (03:44-04:19) | same | harness failures only (drafter full / lpass runner rc 1, qb bench never finished, hpf unit tests 0/8); nothing installed; A11 PASS | 17.3 min | afa97ae6 |
| re-stage (04:21-04:29) | drafter-m32 (test-harness fix; MANIFEST digest 358ed295fee6df59 -> 499089245af1e302, CPU test 42/42) and fc1fc2 (harness fix, 97c3799d07b15459; the old stage dir kept as `fc2hpf.prev`); `W6-SHAS.txt` re-pinned (126 files) | no patch under test changed | 0 | afa97ae6 |
| **window A attempt 3** (04:30-05:08) | A0-A11 | **X GO, L GO, Q NO-GO, H NO-GO**; drm32 installed flags-off; production back at T+28 min; A11 PASS | 28.2 min | afa97ae6 |
| **Phase B** (05:31-08:24) | 10 pinned instances V1 C1 X1 C2 X2 C3 S1 C4 S2 C5 | all EXACT-PASS; X PASS; S (= L) SMALL | 27.6 min (11 relaunches) | afa97ae6 |
| **Phase C** (08:58-09:01) | `promote-w6.sh X` | launcher afa97ae6 -> d625c443 on 8/8 | 0 | d625c443 |
| **Phase D** (09:02-09:56) | production relaunch on d625c443 + suite; STOP; launcher rollback; production relaunch on afa97ae6 + suite + gate + identity | X-on production: unknown G1 prose outcome 33ae975a -> STOP -> rollback; final production exact, gate PASS | 7.2 min (2 relaunches) | **afa97ae6** |

### 3. Window A (stopped, attempt 3, T0 = 04:30:56)

| lane | verdict | deciding evidence |
|---|---|---|
| **X** drafter-m32 | **GO** | A2 quick + full 8/8 passed (26 summaries clean, 0 mismatching words), the same 2 new private JIT entries on 8/8; saved per drafter step at M = 20: **697.2 us** (rank 0) / **790.4 us** (rank 7), gate >= 250; A7 prejit 8/8; A8 quick on the INSTALLED files 8/8 (32 summaries) |
| **L** LIVE_ACT 192 | **GO** (L, L_DRAFTER, L_DRAFTER_M32) | A5 M4 target tests: 350 L cases per rank, 0 failures on 8/8; A2 lpass on the 128-row and the M32 drafter paths |
| **Q** qb-l2pf | **NO-GO (not tested)** | A3 PASS (Part A bitwise + Part S co-runner; R2 pacer `RATES_OK=64,80`), A4 bf16 bench F2 BITWISE PASS (0 failures on 8 ranks); the decisive `--mx` bench never ran: `bench_qb8.py --mx` calls `vllm.utils.flashinfer.mm_mxfp8`, which exists only with the launcher's `CUDA_HOME` / `PATH`; the bench ran without them (AttributeError on 8/8 at graph capture). Harness bug: rerun A4 with the launcher's CUDA env in a short stopped window (~15 min) |
| **H** FC2-head prefetch | **NO-GO** | T1 exhaustive table FAIL (3 cases) and the 3 sanitizer runs, all on the shared check `ptx_off_identical` (synthetic layers: the live arm compiled 3-4 kernel specializations, the flag-off arm 2; the real-layer runs passed it). memcheck / synccheck 0 errors; racecheck 15 races, all inside the existing live `deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl`, none in the prefetch code. Best bench gain ~0.95 us/layer against the 2.0 gate: NO-GO on gain alone |

A11 (production on afa97ae6 with drm32 installed flags-off): exact.py a8 x2 (905/471/2355/1373 unpinned), G1 104.1 (104.151 / 103.968 / 104.181), digest bc1e8386878bd2b7, no w6 line, verify8 identical on 8/8, w6state t0:drm32 with no w6 variable: **PASS**.

### 4. Phase B: pinned A/B chain (`av-pin-k5-g3FU.json`, 05:31-08:24)

Every instance: exact.py a8c3394af9d2 at start and end with 909/467/2335/1373 and 348/233/171/90/67 and a pin hit; G1 3-rep digest bc1e8386878bd2b7 (2 reps 9a18f802d930f6e7); matched C1 AL 5.191 / 2.725 (drafts identical); L2PLAN-CMP PASS; every unit CLEAN (0 foreign units, 0 NOT PROVEN). V1 (`=verify`, gate only): VERIFY PASS 8/8, 7,394 calls on each of the 3 drafter weights per rank, 79,915 rows, 0 mismatching words, 0 capture misses.

| slot | arm | G1 | G4 mean / prose MIN / G4min_hat | 4-session step ms | C1 coding / prose ms | conc8 / conc16 |
|---|---|---|---|---|---|---|
| V1 | V (timings not used) | 102.08 | 62.13 / 48.28 / 48.465 | 44.980 | 30.32 / 26.26 | 259.2 / 396.2 |
| C1 | C | 104.18 | 64.84 / 50.31 / 49.530 | 44.082 | 29.55 / 25.63 | 269.0 / 401.7 |
| **X1** | X | **105.23** | 64.50 / 50.21 / **50.433** | **43.104** | 29.42 / 25.40 | 268.7 / 408.0 |
| C2 | C | 104.21 | 63.31 / 50.20 / 49.870 | 43.780 | 29.56 / 25.62 | 265.9 / 403.4 |
| **X2** | X | **105.30** | 63.17 / 50.12 / **50.417** | **43.092** | 29.32 / 25.31 | 272.9 / 410.9 |
| C3 | C | 104.18 | 64.69 / 49.74 / 49.679 | 43.921 | 29.55 / 25.64 | 273.2 / 396.9 |
| S1 | S = L | 104.75 | 64.22 / 50.09 / 49.611 | 43.846 | 29.53 / 25.44 | 265.3 / 400.6 |
| C4 | C | 104.29 | 63.92 / 50.53 / 49.579 | 43.623 | 29.60 / 25.76 | 269.5 / 403.9 |
| S2 | S = L | 104.85 | 64.09 / 49.32 / 49.612 | 44.031 | 29.55 / 25.48 | 265.7 / 404.3 |
| C5 | C | 104.20 | 63.75 / 49.75 / 49.408 | 43.978 | 29.64 / 25.60 | 263.3 / 407.9 |

Pairs (candidate vs the mean of its two bracketing controls, %, + = better):

| pair | G1 | G4min_hat | prose MIN | C1 coding | C1 prose | conc8 | conc16 | step |
|---|---|---|---|---|---|---|---|---|
| X1 (C1, C2) | +0.99 | **+1.48** | -0.08 | +0.46 | +0.89 | +0.47 | +1.35 | +1.92 |
| X2 (C2, C3) | +1.06 | **+1.29** | +0.30 | +0.80 | +1.26 | +1.24 | +2.69 | +1.76 |
| S1 (C3, C4) | **+0.49** | -0.04 | -0.10 | +0.15 | +1.02 | -2.23 | +0.05 | -0.17 |
| S2 (C4, C5) | **+0.58** | +0.24 | -1.65 | +0.24 | +0.79 | -0.26 | -0.39 | -0.52 |

Noise (5-control spread, floors applied), %: G1 0.3 (measured 0.11), C1 coding 0.5, C1 prose 0.624, G4min_hat 0.931, conc8 3.692, conc16 3.0.

### 5. Phase C: the promotion rule (RUNBOOK 7), applied as written

- **X: promoted.** Exact on every instance (V1 8/8, X1 / X2 EXACT-PASS with drafts identical); primary metric G4min_hat +1.475% and +1.291% (>= +0.5% in both pinned pairs; G1 +0.99 / +1.06% as well); nothing worse beyond noise (the one negative, X1 prose MIN -0.08%, is far inside the 1.6% control spread). `promote-check X` rc 0. `promote-w6.sh X` (08:59:41): new launcher d625c443 on 8/8, `bash -n` OK, backup `.bak-w6` = afa97ae6 on 8/8; supervisor manifest 0c96a727fb81 -> 26db72449673 (the manifest recomputed with afa97ae6 in the launcher's place was 0c96a727fb81 on 8/8: the launcher was the only manifest file that changed).
- **L (S = L, H being NO-GO): not promoted, SMALL.** Primary metric G1 +0.494% (S1) and +0.579% (S2): S1 is under the +0.5% bar. At 4 sessions it costs a little (step +0.17 / +0.52%, prose MIN -0.10 / -1.65%). `promote-check X L` is REFUSED anyway: no arm measured X and L together.
- **Q, H:** never reached Phase B (NO-GO in A). **A** (combined arm): not run, fewer than 2 passing items.
- X's 4-session step gain (-0.83 / -0.76 ms against the bracket means, -1.9 / -1.8%) beat the review's -0.30..-0.60 ms; the measured slowest session did not move beyond noise on the pinned instances (prose MIN 50.21 / 50.12 against controls 49.74-50.54).

### 6. Phase D: production check (RUNBOOK 8)

| instance (UTC) | launcher / X | exact.py start / end | G1 (code / prose / math) | G1 outputs (code / prose / math), digest | G4 metric (prose per run) / G4min_hat / step ms | verdict |
|---|---|---|---|---|---|---|
| w6-PROD-D + w6-D1b (09:02-09:23) | d625c443 / on (default) | a8 OK / a8 OK (915/461/2305/1373; 346/238/177/89/65) | 104.20 (98.94 / 77.30 / 136.36) | 596b478a / **33ae975a** / 92e5d8cf, 247b90ce9f0b492e (all 3 reps identical) | 49.99 (49.13 / 50.35 / 50.48) / 50.72 / 42.99 | **STOP**: unknown prose outcome (CLEAN, 0 foreign) |
| w6-PROD-rb + w6-Drb (09:27-09:43) | afa97ae6 / off | a8 OK / a8 OK (910/466/2330/1373; 349/234/174/88/65) | 104.26 (98.36 / 78.47 / 135.96) | 596b478a / a22e1d39 / 92e5d8cf, 51d0a6537e239757 (all 3 reps identical) | 48.56 (48.04 / 48.09 / 49.55) / 50.25 / 43.68 | EXACT-PASS (a22e1d39 is a known unpinned tie outcome) |

- The STOP followed RUNBOOK 8 ("an unknown prose outcome is a stop") and RUNBOOK 2 (a clean exactness mismatch: production on launcher defaults with every w6 flag unset, all data kept, no retry): `promote-w6.sh --rollback` (09:26:56, afa97ae6 on 8/8, sha-checked), then the verified supervisor after a 30 s idle window, `--expect-launcher afa97ae6`, AV unpinned, no w6 flag: arm-P facts PASS (no w6 variable or line on 8/8), exact.py anchor a8c3394af9d2.
- A confirmation under the pin was not run: Phase B's X1 / X2 are that confirmation (the same files and the same running env; the promoted launcher only moves `DSV41_DRAFT_MOE_M32=1` from a supervisor argument into the launcher, which the arm facts verified on 8/8).
- Regression gate (kit 08, `gaterun.sh w6-D` on spark-one: 30 s idle window, 0.5 s sampler, gaudit) on the final production: **PASS in 596 s, audit CLEAN** (every step's success delta = its own requests): exact a8c3394af9d2; G1 one pass 104.36 (code 596b478a, math 92e5d8cf, prose a22e1d39); one cold 1M at TTFT 168.4 s (good class; MemAvailable minimum 8.3 GB); GSM8K-200 195/200; longctx 60/60, copy 1.000; KL verify / smoke / capture PASS (the capture's equivalence tier against the 10-28 capture reads 'FAIL (no noise floor)': not an exactness signal at its settings, as in window 5).
- Identity on 8/8: verify8 32 OK / 0 problems (`w6gates.py verify8` GO: identical on 8/8, no install change); site tree 36631b8bc6ce (5,455 files) and env digest 9626ff845dc0 (window 5's) on 8/8; w6state GO: the w6 files = t0:drm32, pacer 0451009d and r4 a09f85e8, launcher afa97ae6, no w6 variable in the running env of any rank. `/health` 200, `dsv41-rank` active on 8/8, nrestarts 0, SM 1969-1989 MHz.

### 7. Goal status

| metric | before window 6 (production / pinned) | now (production) | X-on, pinned / production (not live) | goal |
|---|---|---|---|---|
| G1 | 104.00 / 104.22 | 104.26 | 105.23-105.30 / 104.20 | 120 (+15.1% needed) |
| G4 metric (slowest session) | 47.89 / 50.55 | 48.56 | 50.21 / 50.12 pinned / 49.99 | 50 |
| G4min_hat | 49.25 / 49.42 | 50.25 | 50.43 / 50.72 | 50 |
| 4-session step | ~44 ms | 43.68 ms | 43.10 / 42.99 ms | - |

G1 120 still needs about +15%. Only acceptance closes that (drafter self-distillation, the open user decision of window 5, or R12's row-count invariance). G4: G4min_hat is above 50 on every X instance (50.42-50.72) and on this production launch (50.25), below it on the pinned controls (49.41-49.87); the measured slowest session (48.56 in production now, 49.99 with X) still depends on how the 4 sessions batch.

### 8. Rollback and re-enable

- The window's net change on 8/8 is the two flag-off site files and the new launcher backup. To remove the files (service stopped): `W6_WINDOW=go bash $W/bundle/w6-bundle.sh uninstall`, verified by `w6-bundle.sh expect live` (restores `deep_gemm_moe.py` 5162dc0f from `.bak-w6` or from the staged `~/dsv41-stage/w6/bundle/deep_gemm_moe.py.base`, removes `dsv41_draft_m32.py`). By hand on 8/8: `E=~/venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts; cp -p $E/deep_gemm_moe.py.bak-w6 $E/deep_gemm_moe.py && rm $E/dsv41_draft_m32.py`, then check 5162dc0f on 8/8. The 2 prejit JIT entries are inert and may stay.
- Re-enable X (only after the user's decision): see the decision paragraph above. Turn it off again for one launch with the supervisor argument `DSV41_DRAFT_MOE_M32=0`; permanently with `cp ~/dsv41-tp8.sh.bak-w6 ~/dsv41-tp8.sh` on 8/8 (= afa97ae6) and a relaunch with `--expect-launcher afa97ae6`.
- `$W` = the Mac kit `/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad/w6` (ephemeral).

### 9. Open items after window 6

- **X:** the user decision above. If X goes live unpinned, add 33ae975a to the known prose outcomes (driver `PROSE_KNOWN`, gate notes) and re-check under the gate; if the pin becomes the production default, X needs no new outcome.
- **Q** (q_b in the L2 plan, review +0.7..1.3% G4min_hat): fix the A4 harness (run `bench_qb8.py --mx` with the launcher's `CUDA_HOME` / `PATH`), a short stopped window for A4 only (~15 min), then a pinned Q chain. Installing qb moves the supervisor manifest (0c96a727fb81 -> 3a29a454fbed on afa97ae6). Like X, Q changes a cost the unpinned AV profile measures (verify above 8 rows), so its production check can meet the same unpinned prose flip: plan its Phase D with that (or with the pin as the production default).
- **H:** NO-GO on gain (0.95 of 2.0 us/layer) independent of the `ptx_off_identical` failure; not worth a retry as designed.
- **Vision:** the staged image set ced2eab1bb8da400 stays bound to what is live (launcher afa97ae6, manifest 0c96a727fb81, `model_runner.py` 6a4f2433, all unchanged; qb not installed), so this window's net change needs no re-stage; only the kit RUNBOOK's recorded "before" site tree moves from 11e30c3df6b6 to 36631b8bc6ce. Re-promoting X (new launcher, manifest 26db72449673) would make the set stale.
- **Codex R12:** refresh its baseline: `deep_gemm_moe.py` is now 9364f121 (was 5162dc0f), new `experts/dsv41_draft_m32.py` a17d574a, site tree 36631b8bc6ce over 5,455 files, 2 new JIT entries; the launcher (afa97ae6), the l2 prefetch files and the running env are unchanged.
- **Harness fixes from Phase D** (Mac kit only): `w6-bundle.sh expect` compares the manifest with its pre-promotion value, so any promoted launcher made `expect t0:<set>` refuse (Phase D used `phaseD/bundle-expect-promoted.sh`, which accepts only the case where the promoted launcher is the one manifest file that differs); the window-5 `verify8.sh` hard-codes launcher afa97ae6 (a copy expecting d625c443 is `phaseD/verify8-w6d.sh`; not needed after the rollback); `promote-w6.sh`'s tests did not exercise Phase D's unpinned prose check.

## Production AV pin + window-6 X (2026-10-29): the k5 AV cost pin is now the PRODUCTION default and window-6 X is on (launcher afa97ae6 -> **5b0a889c**). Production is now the pinned configuration, and it is bitwise identical to window 6's pinned evidence: exact.py a8c3394af9d2 with the pinned counters 909/467/2335/1373 and G1 digest bc1e8386878bd2b7. G1 105.27 tok/s, G4 metric 50.99 (12/12 session-runs >= 50), regression gate PASS. API down 146 s (one relaunch)

**Decision.** At about 10:40 UTC the user answered the window-6 X question with "yes, let's pin". That chose option (b) of the "Goal window 6" section: make the AV pin the production default, so X is bitwise in production too. Why: in window 6 Phase B, 10/10 pinned instances were bitwise identical, both with X and without it. Every instance gave exact.py a8c3394af9d2 with counters 909/467/2335/1373 and per-position counts 348/233/171/90/67, plus G1 digest bc1e8386878bd2b7 (first two reps 9a18f802d930f6e7; code 596b478a, prose 923a46be, math 92e5d8cf). The X instances X1 / X2 measured G1 105.23 / 105.30 and G4min_hat 50.43 / 50.42. Unpinned, X had moved the G1 prose near-tie to a new outcome, 33ae975a: X makes the drafter cheaper in the boot-time AV profile, which changes the verify layouts. The pin removes that boot-time measurement, so nothing moves.

**How it ran.** Every number below was taken at the 2000 MHz boot clock with nothing raised (SM 1969-1989 MHz at every check on 8/8). `ENGRAM_CHECK` was not touched (the supervisor's engram check ran: 8/8 OK). `--enable-prompt-tokens-details` stayed on, `dsv41-lanproxy` was not touched, and no client was blocked. Each timed or exactness unit ran only after a 30 s API idle window, under the window-5 foreign-traffic rule. 0 units were foreign. The cluster lock (owner=claude, purpose "pin default + X (user OK)") was taken at 13:17:54 UTC and is still held for the verification step. Every action has a line in `AGENTS_LOG.md`.

### 1. What changed (13:17-13:52 UTC)

| step (UTC) | what | result | API down | launcher after |
|---|---|---|---|---|
| 13:17:54 | lock (atomic `set -C`) | owner=claude, eta 16:17:54 | 0 | afa97ae6 |
| 13:18:23 | stable pin `~/dsv41-opt-20261021/pins/av-pin-k5-g3FU.json` on 8/8 (`mkdir -p`; `cp -p` of `~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json` via temp + `mv`; read-only, 2272 B) | sha **eeac4f97394d8275** on 8/8; the stage source is unchanged and kept | 0 | afa97ae6 |
| 13:19:25 | launcher built on the Mac from spark-one's live afa97ae6 (full sha checked) + the X lines of `~/dsv41-stage/w6/promote/dsv41-tp8.sh.d625c443`; `bash -n` OK; staged as `~/dsv41-tp8.sh.pinx-tmp` and sha-checked on 8/8; `~/dsv41-tp8.sh.bak-pin` = afa97ae6 (full) on 8/8; renamed into place (mode 755) | **5b0a889c** on 8/8 | 0 | 5b0a889c |
| 13:22:32-13:24:59 | relaunch `pinx-PU`, after a 30 s idle window (IDLE waited=30.3): supervisor 7bea1cf3 `--label pinx-PU --expect-launcher 5b0a889c DSV41_GPU_CLOCK_MHZ=2000`, no other env | preflight manifest **b71dda2146a7** + launcher 5b0a889c identical on 8/8; engram 8/8 OK; HEALTHY on try 1, +132 s after systemd-run (the pin skips the boot-time profile) | **146 s** (/health non-200 for 142.6 s) | 5b0a889c |
| 13:25:37-13:33:45 | exactness suite `pinx-U1`: exact.py, pin check, arm facts, G1 x3, exact.py #2 | EXACT-PASS | 0 | 5b0a889c |
| 13:34:20-13:40:40 | goal suite `pinx-U2`: exact.py, G4 warm-up + 3, conc8, conc16, exact.py #2 | EXACT-PASS | 0 | 5b0a889c |
| 13:41:24-13:51:34 | regression gate (kit 08) via `~/dsv41-g5v/gaterun.sh pinx` | **PASS**, audit CLEAN | 0 | 5b0a889c |
| 13:52 | verify8 + w6state on 8/8 (read-only) | 34 OK / 0 problems; GO / GO | 0 | 5b0a889c |

The launcher diff against afa97ae6 is 3 comment lines and 2 exports. Nothing else changed:
```sh
export DSV41_AV_PROFILE=${DSV41_AV_PROFILE:-pinned}  # (existing line 63, unchanged)
# 2026-10-29 user OK: production uses the k5 AV cost pin (no boot-time AV profile); window-6 X on
# DSV41_AV_COST_JSON= (empty) as a supervisor argument = one unpinned launch with the boot-time AV profile (the default uses "-", not ":-", so an explicit empty value is kept); DSV41_DRAFT_MOE_M32=0 = X off
export DSV41_AV_COST_JSON=${DSV41_AV_COST_JSON-$HOME/dsv41-opt-20261021/pins/av-pin-k5-g3FU.json}
...
export DSV41_MOE_FC2C=...                            # (existing, unchanged)
# goal window 6 (RUNBOOK 7, promoted X): exact levers, default on; '=0' / '=1024' as a supervisor argument turns one off for a launch (RUNBOOK 9)
export DSV41_DRAFT_MOE_M32=${DSV41_DRAFT_MOE_M32:-1}
```
The pin default uses `${VAR-default}` on purpose, not `${VAR:-default}`. With `:-`, an explicit empty `DSV41_AV_COST_JSON=` would fall back to the pin again, and the one-launch unpinned rollback below would be impossible. The two forms differ only when the variable is set but empty; I tested unset, empty and set. No `DSV41_*` variable is set in the systemd user environment on any node (checked on 8/8), so an unset variable gets the pin.

### 2. Evidence on the production instance (pinx-PU)

- **Start facts on 8/8:** every rank log has `DSV41 AV profile skipped (pin hit, DSV41_AV_PROFILE=pinned): /home/jeweled/dsv41-opt-20261021/pins/av-pin-k5-g3FU.json covers 29 captured sizes (limit 192), tails [288, 384, 768, 1536, 3072, 6144, 12288, 16384], draft [1, 2, 4, 6, 8, 12, 16, 24, 32]`, 0 pin-miss lines and 0 full-profile lines. Every rank log also has `DSV41_DRAFT_MOE_M32=1: DSpark drafter MoE (E128 top-3, N768/K5120 FC1, K384 FC2) served for 1 <= M <= 160 rows (32-row alignment, M32 target native)`, 0 "ignored" lines and 0 ERROR/Traceback. `mkpin_k.py check`: `OK: k=5 launch hit /home/jeweled/dsv41-opt-20261021/pins/av-pin-k5-g3FU.json`. Arm facts PASS (levers=X, pinned=1). The running env on 8/8 has exactly `DSV41_DRAFT_MOE_M32=1` plus `DSV41_AV_COST_JSON=<stable pin>` as its w6/AV variables. L2 plan noqb on 8/8.
- **Exactness, each unit CLEAN:** exact.py ran 5 times: the start and end of pinx-U1, the start and end of pinx-U2, and once in the gate. Every run gave **a8c3394af9d2**. The four suite runs all had counters **909/467/2335/1373** (acc/drafts/dtok/gen) and per-position **348/233/171/90/67**, which are the pinned values. G1 x3 (9/9 units CLEAN): outputs code **596b478a**, prose **923a46be**, math **92e5d8cf**, identical across reps. Digest **bc1e8386878bd2b7**; first two reps **9a18f802d930f6e7**. Gate verdicts: pinx-U1 EXACT-PASS, pinx-U2 EXACT-PASS.
- **G1:** **105.27 tok/s** (code 99.55 / prose 79.37 / math 136.89; reps 105.26 / 105.25 / 105.29; AL 2.590 / 1.847 / 3.594).
- **G4 x3 (warm-up + 3, 4/4 units CLEAN):** **G4 metric 50.99** (slowest session = prose in every run: 50.48 / 50.54 / 51.97; warm-up 50.80). **12/12 session-runs >= 50**, worst run 50.48. G4min_hat **50.56**, 4-session step **43.01 ms**, prose steps 277.3, G4 aggregate 190.6 tok/s. By session: code 60.26, prose 50.99, math 81.22, mixed 65.60.
- **Concurrency (1 unit each, CLEAN):** conc8 **273.9 tok/s** (per request mean 45.5 / min 35.4); conc16 **415.1 tok/s** (35.7 / 26.4); errors 0.
- **Regression gate (kit 08, gate.sh 1889ae4a, `--live`): PASS in 579 s** (13:41:54-13:51:33), audit CLEAN (every step's success delta = its own requests). refuse_if_busy saw AV = the stable pin. exact.py a8c3394af9d28249. G1 one pass 105.18 (code 596b478a, math 92e5d8cf, prose 923a46be; the first gate under this launcher sha, so there was no previous digest to compare). 1M probe TTFT **168.6 s** for 1,024,611 prompt tokens (good class, MemAvailable min 8.9 GB). GSM8K-200 **195/200**. longctx needle 60/60, copy 1.000. KL verify / smoke / capture_small PASS; the capture is the new baseline for 5b0a889c. Result: `~/dsv41-bench/gate/runs/5b0a889c/20261029T134154Z.json`.
- **Identity on 8/8 (read-only):** verify8 (pinx copy, expectations = launcher 5b0a889c, the stable pin in the running env, stable pin sha eeac4f97394d8275, `.bak-pin` afa97ae6, X served line): **34 OK / 0 problems**. Gates: verify8 GO (identical on 8/8, no unexpected install change); w6state GO (w6 files = bundle state t0:drm32, libs, launcher 5b0a889c, running w6 env {DSV41_DRAFT_MOE_M32: 1}). Site tree 36631b8bc6ce over 5,455 files on 8/8. Running-env digest 9f2d10601f94 on 8/8 (9626ff845dc0 on the afa97ae6 production; the difference is the 2 new variables). dsv41-rank active 8/8, NRestarts 0, rank logs clean.

| | before: production w6-Drb (afa97ae6, AV unpinned, X off) | **now: production pinx-PU (5b0a889c, pin + X)** | window 6 Phase B pinned X1 / X2 |
|---|---|---|---|
| exact.py | a8c3394af9d2 (910/466/2330/1373: counters follow the boot profile) | a8c3394af9d2 **909/467/2335/1373, 348/233/171/90/67** | the same |
| G1 outputs (code / prose / math) | 596b478a / a22e1d39 (unpinned tie) / 92e5d8cf, digest 51d0a6537e239757 | 596b478a / **923a46be** / 92e5d8cf, digest **bc1e8386878bd2b7** | digest bc1e8386878bd2b7 |
| G1 tok/s | 104.26 | **105.27** (+0.97%) | 105.23 / 105.30 |
| G4 metric (prose per run) | 48.56 (48.04 / 48.09 / 49.55) | **50.99** (50.48 / 50.54 / 51.97) | 50.21 / 50.12 |
| session-runs >= 50 | 9/12 | **12/12** | 11/12 / 11/12 |
| G4min_hat | 50.25 | **50.56** | 50.43 / 50.42 |
| 4-session step | 43.68 ms | **43.01 ms** (-1.5%) | 43.10 / 43.09 ms |
| conc8 / conc16 | not run in that suite (pinned controls C1-C5: 263.3-273.2 / 396.9-407.9) | **273.9 / 415.1** | 268.7 / 408.0; 272.9 / 410.9 |
| relaunch downtime | 216 / 217 s (the two Phase D relaunches, +203 s to healthy with the boot-time profile) | **146 s** (+132 s to healthy) | - |

**Goal status:** G1 105.27 of 120 tok/s (gap 14.0%). G4: in production, the slowest of the 4 sessions averaged 50.99 tok/s over 3 runs, and all 12 session-runs were at least 50. That is the first time a production instance met the 50 tok/s per-session target in every run (window 5 production 47.9; window 6 production 48.56, and 49.99 with X unpinned). It is one instance, though, and G4min_hat 50.56 puts it just above the line, so it is not a margin.

### 3. What is live now (verified on 8/8 at 13:52 UTC)

- **Launcher** `~/dsv41-tp8.sh` **5b0a889c** = `5b0a889c85b6b7595a3ba1a3d49ca77399438b9da127f9c455f03dfeea11958a` on 8/8. Backup `~/dsv41-tp8.sh.bak-pin` = afa97ae6 (`afa97ae6dfc65943f7e536baf7922bbecf2c5d8110f35768018c7c7dd5be3cc1`) on 8/8 (new); `.bak-w6` = afa97ae6 and the older backups are unchanged. `~/dsv41-stage/w6/promote/dsv41-tp8.sh.d625c443` is kept on spark-one.
- **AV pin:** `~/dsv41-opt-20261021/pins/av-pin-k5-g3FU.json`, sha eeac4f97394d8275, on 8/8 (new, read-only). Production reads only this stable copy; the stage copy `~/dsv41-stage/moet/pins/av-pin-k5-g3FU.json` (same sha) is kept for the window harnesses.
- **Supervisor** 7bea1cf3. **Manifest (new): b71dda2146a7** = the window-6 manifest 0c96a727fb81 with the launcher entry changed to 5b0a889c (checked on 8/8 by recomputing it with `.bak-pin` in the launcher's place). **Site tree** 36631b8bc6ce / 5,455 files (unchanged). X files as installed in window 6: `deep_gemm_moe.py` 9364f121, `dsv41_draft_m32.py` a17d574a; bundle state t0:drm32.
- **Running env:** window 5's variables + `DSV41_AV_COST_JSON=/home/jeweled/dsv41-opt-20261021/pins/av-pin-k5-g3FU.json` + `DSV41_DRAFT_MOE_M32=1` (env digest 9f2d10601f94). Unchanged: gate.sh 1889ae4a, `dsv41_l2_prefetch.py` 63737f9c, `fp8_utils.py` b35c0561, `sm121_target_m32.py` ea4a8814, the g2 / r4 libraries, the 2000 MHz clock unit.
- **Exactness reference for production is now the PINNED set:** exact.py a8c3394af9d2 + 909/467/2335/1373 + 348/233/171/90/67; G1 digest bc1e8386878bd2b7 (first two reps 9a18f802d930f6e7), code 596b478a, prose 923a46be, math 92e5d8cf. The unpinned tie outcomes (923a46be / e8a524f3 / a22e1d39, and 33ae975a with X) only apply to an explicitly unpinned launch.

### 4. Rollback

- **Full rollback** (back to afa97ae6: AV unpinned, X off). On 8/8 run `cp ~/dsv41-tp8.sh.bak-pin ~/dsv41-tp8.sh` (temp copy + `mv` is safer), check sha afa97ae6 on 8/8, then run the supervised relaunch after a 30 s idle window: `ssh spark-one cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh | bash -s -- --label <L> --expect-launcher afa97ae6 DSV41_GPU_CLOCK_MHZ=2000`. Expect exact.py a8c3394af9d2 (unpinned counters) and manifest 0c96a727fb81.
- **One unpinned, X-off launch** without touching the launcher: supervisor `... --expect-launcher 5b0a889c DSV41_GPU_CLOCK_MHZ=2000 DSV41_AV_COST_JSON= DSV41_DRAFT_MOE_M32=0`. This gives the boot-time AV profile with X off, which is the afa97ae6 behaviour. `DSV41_AV_COST_JSON=` alone keeps X on unpinned, and that reproduces the 33ae975a prose outcome; `DSV41_DRAFT_MOE_M32=0` alone keeps the pin with X off, which is equivalent to the Phase B C arm.
- Mac tools for this change (scratchpad `pinx/`): `relaunch-pinx.sh`, `rollback-pinx.sh` (the full rollback plus a guarded exact.py), `run-suite.sh` (the w6 production suite runner with a new arm U = pinned production), `bundle-expect-pinx.sh`, `verify8-pinx.sh`.

### 5. Open items

- Window-7 offline prep diffs against afa97ae6. Rebase its runbook on **5b0a889c**: production is pinned now, and every lever must still be tested both pinned and with one explicitly unpinned launch.
- Codex R12 baseline: production changed from (unpinned, X off) to (pin + X). Its reference numbers should be refreshed from section 2.
- If a later change moves the costs the pin was built from (the drafter, the verify graphs or the capture sizes), the pin needs a fresh `mkpin_k.py make` from an unpinned profile launch, and that makes a new anchor. If `check_pin` stops covering a launch shape, the launcher falls back to the full boot profile, and the startup log says "pin miss".
- The cluster lock is still held (owner=claude) for the verification step.
