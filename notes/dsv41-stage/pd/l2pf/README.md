# pd-l2pf: L2 prefetch re-plan arms (exact by construction: cache hints only, no math change)

Status 2026-10-25 (offline prep): arms defined, post-budget site file staged on 8/8 and NOT installed. Codex's 10 CPU planner tests rerun locally: OK.

| arm | env | needs install | prior evidence |
|---|---|---|---|
| C | none (pinned AV) | no | control |
| S | `DSV41_L2_PREFETCH_PRUNE_SHARED=1` | no (live 4da7745b) | Codex D01, one pinned instance pair, 2-repeat screen: code/prose/math +2.1/+2.1/+1.9%, target span -0.50 ms. C1 hashes and counters identical. The C2/C4 hash "fail" is a non-gate, since the control varies too (takeover audit §5 #14). Router backfill re-adds 150 of the 165 MiB removed. |
| SQ | S + `DSV41_L2_PREFETCH_PRUNE_STALE_QKV=1` | no | never measured. It drops the stale qkv_a-shard prefetch when r3 falls back by M (critic C6 "q_b never prefetched at 16u") |
| SR | S + `DSV41_L2_PREFETCH_PRUNE_ROUTER=1` | no | mHC-only post-attention window. Tests stretch (≈+18 µs/collective at 6 MB) against byte coverage |
| PB | `DSV41_L2_PREFETCH_PRUNE_SHARED_POST_BUDGET=1` | **yes**: `pd-all.sh install l2pf` (dsv41_l2_prefetch.py 4da7745b -> 514dcb23, Codex `post-budget`) | CPU: 500 randomized scenarios show flag-off parity with live. The arm removes 173 MB of selected shared hints per plan and backfills nothing: -26.9% selected bytes at M1-8, -33.5% at M12-24 (arithmetic prediction) |

`ops/arm.sh <arm>` restarts the 8 ranks with that arm. It pins the AV profile (`PD_AV_PIN`, default `~/dsv41-stage/codex-rdma-20261024/av-pin-c0.json`) and turns on `DSV41_L2_PREFETCH_PLAN_AUDIT=1`, which logs address-free plans at startup only.
- PB refuses to run unless the site file is installed. The pre-budget flags must stay 0 with PB; the candidate itself raises on mixed modes.
- Mixing arms with pd-d05/pd-pdl is fine: those are separate files and flags.

Suggested order: screen S, PB and SQ against C (1 pinned instance each, matched C1 x5 + bench single). Then ABAB the best one at 2 instances per arm.
- Gate: C1 at least 0.8% better in every pair, or conc16/32 at least 2%.
- exact.py must be identical under the pin.
- Expected: 1u -0.3..-0.6 ms, 16u -0.2..-0.5 ms (takuz0-compare #2).

Install/rollback for PB: `ssh spark-one cat ~/dsv41-stage/pd/common/pd-all.sh | bash -s install l2pf` / `... uninstall l2pf`. Flag off = live plan (CPU parity).
