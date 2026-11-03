# g130 rdma: window runbook

Nothing here has run on a GPU. `bench/bench_g130.py`, `tests/selftest_g130.py`, `ops/run-*.sh`, `ops/pair.sh`, `ops/stage8.sh` and `ops/verdict.py` are syntax-checked only; expect a first-run fix or two.

## Flags

All default off. Unset equals production.

| Env | Variant | Touches | Acts |
|---|---|---|---|
| `RDMA_AR_SPLIT_POST=1` | A | proxy | next message |
| `RDMA_AR_PUB_V1=1` | B, phase-1 `seq - 1` compare | kernel instantiation | next launch; a graph keeps what it captured |
| `RDMA_AR_PUB_V2=1` | B, no phase-2 spin | kernel instantiation | same |
| `RDMA_AR_HYG_PROXY=1` | C, doorbell spin + reap after post | proxy | next message |
| `RDMA_AR_HYG_POLL=1` | C, clock once per 1024 passes | kernel instantiation | same as V1 |

Select the .so with `RDMA_AR_EXT=/home/jeweled/dsv41-stage/g130/rdma/build/rdma_ar_ext.so`. Kernel flags need the parallel-load path; the host refuses them otherwise. The `RDMA_AR_` prefix puts every flag and the .so hash in the rank fingerprint, so a rank mismatch stops startup.

`pingpong_ns` is not comparable with `HYG_PROXY` on (ping is only checked in the full pass). Use the hand-off stat.

## Rule learned on 2026-11-02: no builds around a relaunch

vLLM needs 0.90 x 121.63 = 109.47 GiB free on the device at startup. On GB10 that is the same memory as the host. A compile on spark-two made rank 1 fail that check (109.0 GiB) with MemAvailable at 113 GB, so the 8 GB floor gives no protection there. The kit needs no further build. If you rebuild, do it with no relaunch running or due:

    cd ~/dsv41-stage/g130/rdma/build/ext && env PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda timeout -k 5 300 bash ../../ops/guard_run.sh ../../logs/variant-build.log -- ~/venvs/vllm032/bin/ninja -j1

A build uses about 5 GB.

## 0. Setup, from the Mac

    export PATH=/private/tmp/claude-501/-Users-jeweled/e29a16fa-1457-4e89-9250-a8791bc6d2a7/scratchpad/ts-ssh/bin:$PATH
    which ssh    # must print the shim
    K=~/Documents/dsv41-kits/g130-rdma

The shim is in a scratchpad that a sweep may reach around Nov 6-7. Copy `ts-ssh` out first if the window is later than that.

## 1. Stage to all 8 (disk only, safe while serving)

    bash $K/ops/stage8.sh

Every host must print `staged OK` with the same digest.

## 2. Stop the service (your call; ~1-2 min of the window is this)

    for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do ssh -n $h 'timeout 30 systemctl --user stop dsv41-rank; systemctl --user is-active dsv41-rank' & done; wait

## 3. Selftest on all 8 GPUs

    bash $K/ops/run-selftest8.sh st1

Every host must end `SELFTEST PASS`. This checks the data path of every kPub kernel against legacy and mask 0. It has no proxy, so it does not test the protocol.

## 4. Bench, control first and last

    bash $K/ops/run-bench8.sh ctlA control/rdma_ar_ext.so --phases X,L
    bash $K/ops/run-bench8.sh arms build/rdma_ar_ext.so --phases X,S,L
    TIMING=0 bash $K/ops/run-bench8.sh armsT0 build/rdma_ar_ext.so --phases L
    bash $K/ops/run-bench8.sh ctlB control/rdma_ar_ext.so --phases L

Every run must print `DONE failures per rank: [0, 0, 0, 0, 0, 0, 0, 0]`. Any failure in X or S stops that variant for good: it is not a noise question. In `[S]`, the victims' max proxy post must be in the ms range; if it is not, the stalls did not land and S proved nothing. Duration is unmeasured; my guess is 15-30 min.

Read `ctl` in `arms` against `ctlA`/`ctlB`. If the variant .so with flags off is further from control than `ctlA` is from `ctlB`, stop and investigate. A serving-like repeat is `--pin none`.

## 5. Kill thresholds

    ssh -n spark-one 'cd ~/dsv41-stage/g130/rdma && python3 -I ops/verdict.py results/arms.allranks.json'
    ssh -n spark-one 'cd ~/dsv41-stage/g130/rdma && python3 -I ops/verdict.py results/armsT0.allranks.json'

| Variant | GO | KILL |
|---|---|---|
| Split post | at least 1 us lower at AG M=6 and AR M=6 | less than that in any cell |
| V1+V2 | at least 0.6 us per publish | under 0.35 us per publish; between is HOLD |
| Hygiene | no threshold given | decide by hand; audit priced it at +0.04..+0.08% |

`verdict.py` applies split post to both the in-kernel `net` column and us/op, in b2b and l2pf: the strictest reading of "net". Relax it by reading the printed deltas if you meant only one. Per publish: AG = 1 publish, two-shot AR = 2. Trial spread is printed, and past single-run tables moved by about 1 us between identical arms; if a delta sits inside the spread, it is not a result.

Also read the `post` delta for split post: the audit's bound was +0.78 us per hop.

## 6. Pinned serving pair (restarts the service)

Only for what passed section 5, one bundle at a time. CTTC:

    bash $K/ops/pair.sh c1 C
    bash $K/ops/pair.sh t1 T RDMA_AR_PUB_V1=1 RDMA_AR_PUB_V2=1
    bash $K/ops/pair.sh t2 T RDMA_AR_PUB_V1=1 RDMA_AR_PUB_V2=1
    bash $K/ops/pair.sh c2 C

Split post: `T RDMA_AR_SPLIT_POST=1`. Hygiene: `T RDMA_AR_HYG_PROXY=1 RDMA_AR_HYG_POLL=1`.

Each arm must show `exact a8c3394af9d2` and digest `9a18f802d930f6e7`. A clean mismatch stops the variant. `c1`/`c2` also test flags-off in serving against today's numbers, at the production .so's kernels. `pair.sh` runs exact.py and goal_single only, so add your G4 and matched suites if the G1 gain is real. Results: spark-one `~/dsv41-g130ab/rdma/<label>/`.

## 7. Back to production

    ssh -n spark-one 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > /tmp/rel.sh && bash /tmp/rel.sh --label g130-prod --expect-launcher 37757dde DSV41_GPU_CLOCK_MHZ=2000

No `RDMA_AR_EXT` means the launcher's default path, production .so 0692daa5. Confirm `exact a8c3394af9d2`.

## Hazard reasoning

**V1.** Every op publishes phase 1 once and `st->seq` counts ops, so the last published value is `seq - 1`. The guard still spins on `posted_seq`, so the invariant is the same with one sys load fewer. Do not delete the guard: a rank's kernel only waits on the peers' proxies, so it can finish while its own proxy is parked, and `send_seq` plus the descriptor are a single mailbox. After a one-shot or gather, that guard is the only thing preventing a lost publish.

**V2.** A rank publishing phase 2 of `seq` has passed the phase-1 wait for `seq`, so all 7 peers published phase 1 of `seq`, so each finished every earlier op on its single stream, including the previous two-shot `seq'`, which waits for this rank's phase-2 flag. So this rank's proxy has read `send2_bytes` and posted `seq'` to all 7. `seq - 1` would hang, because `send2_seq` moves only on two-shot ops. The same chain already protects today's `send2` slot reuse, since the old spin does not cover NIC reads.

**Assumptions.** V2 depends on one stream per rank with kernels serialized, and on every rank waiting for every peer's phase-2 flag. The rdma_ar contract states the first. A graph with a non-linear chain, or a rank skipping the phase-2 wait, breaks it.

**Publish ordering.** The removed loads carried no data ordering. The publisher thread still has the `__threadfence_system` in `last_cta` and the release store.

**Split post.** The per-QP send queue executes in WQE order however the doorbell arrives. Emulator unit test: same WR list as the chain, and a failed payload post leaves no flag.

**Not covered on the CPU.** GPU weak ordering at sys scope (the emulator uses SC fences and acquire loads), C2C coherence, NIC, BlueFlame, write combining on Grace, back-to-back BF writes to one QP (ib_write_lat never does this), nonzero `uuarn` on our QPs, real timing. Only bench phases X and S exercise those, and only statistically.
