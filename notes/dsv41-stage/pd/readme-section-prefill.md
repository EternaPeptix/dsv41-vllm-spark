
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
