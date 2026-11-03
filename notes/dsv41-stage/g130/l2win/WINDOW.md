# g130 l2win: window runbook

Nothing here has run on a GPU, and nothing has been staged beyond spark-three. The CPU test passes on spark-three (`tests/test_l2win_plan_cpu.py`, no CUDA). `bench/bench_l2win.py` and `ops/*.sh` are syntax-checked only, so expect a first-run fix. Bitwise exactness is reasoning until phase F and a pinned pair say otherwise.

No native build. The arms use the installed pacer `~/rdma_ar_dsv41/build-g2ct/libdsv41_l2pf_g2.so` (0451009d) and `dsv41_spin` in the live r4 library (a09f85e8). `rdma_ar.cu` is untouched.

## Arms

| Bench mode | Env | Module | What it does |
|---|---|---|---|
| `p125_c64`, `p150_c64` | `DSV41_L2_PREFETCH_PACE_GBPS=R DSV41_L2_PREFETCH_PACE_CHUNK=65536 DSV41_L2_PREFETCH_PACE_KINDS=ar` | live | Env-only. Every AR window paced, gather stays burst. |
| `p150_c64_x26` | the above + `DSV41_L2_PREFETCH_PACE_MAX_US=26` | live | Env-only. Issue stops before the earliest Kernel2 start. |
| `p85_c16` | `DSV41_L2_PREFETCH_PACE_GBPS=85 DSV41_L2_PREFETCH_PACE_KINDS=ar` | live | Env-only. The gate window then ends about 12 us after AR end. Expect it to lose at M>=4. |
| `s85_c16` | `..PACE_GBPS=85 ..PACE_KINDS=ar ..PACE_MAX_WIN_KB=2048` | kit | Post-MoE windows paced, gate stays burst. No window passes AR end, so no new co-residency. |
| `s85_c64+b150` | `..PACE_GBPS=85 ..PACE_CHUNK=65536 ..PACE_KINDS=ar ..PACE_MAX_WIN_KB=2048 ..PACE_BIG_GBPS=150` | kit | Two rates. |
| `d0`, `d1`, `d3` | `..BURST_DELAY_US=D` | kit | Burst after a chained spin of D us, AR windows only. The launch gap adds about 1 us, unmeasured. |
| `s85_c16+d1`, `s85_c16+d3` | `s85_c16` + `..BURST_DELAY_US=D` | kit | Short paced, gate delayed. |

`..` is `DSV41_L2_PREFETCH_`. All new flags default off. With them unset `G2` stays False and none of the new lines run, apart from reading four env vars at import. Flags are read at startup and graphs are captured at each launch, so a relaunch is the only recapture needed. The rdma_ar fingerprint covers `DSV41_L2_PREFETCH*` and this module's hash, so a mismatch across ranks stops at startup.

## 0. Setup, from the Mac

    export PATH=/private/tmp/claude-501/-Users-jeweled/e29a16fa-1457-4e89-9250-a8791bc6d2a7/scratchpad/ts-ssh/bin:$PATH
    which ssh    # must print the shim
    K=~/Documents/dsv41-kits/g130-l2win

The shim is in a scratchpad that a sweep may reach around Nov 6-7. Copy `ts-ssh` out first if the window is later. Take CLUSTER_LOCK and log to AGENTS_LOG.md as usual. Ask for Codex and survey traffic to stop: this is a timed bench.

## 1. Stage to all 8 (disk only, safe while serving)

    bash $K/ops/stage8.sh

Every host must print `staged OK` with the same digest.

## 2. Stop the service

    for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do ssh -n $h 'timeout 30 systemctl --user stop dsv41-rank; systemctl --user is-active dsv41-rank' & done; wait

## 3. Ceiling first, about 2 minutes

    bash $K/ops/run-bench8.sh l2c --phases C

Read `[C]`. Does chunk 64 KiB reach 116 GB/s or more on `gate`, with `full 30/30`? This has never been measured.

- Yes: run all default modes in section 4.
- No, but 32 KiB does: replace `c64` with `c32` in `--modes`.
- Neither: drop every gate-pacing mode with `--modes none,live,d0,d1,d3,p85_c16,s85_c16,s85_c16+d1,s85_c16+d3`. The one-warp issuer without the divide is the only thing left that could lift the rate. It is not in this kit. Build it only if `s85_c16` shows a real gain and the gate window is still the larger pool.

`issued` counts instructions. It does not prove a 64 KiB prefetch landed: phase W is what shows that.

## 4. Stretch, late bytes, bitwise, about 20-25 minutes

    BENCH_TIMEOUT=2400 bash $K/ops/run-bench8.sh l2a --phases B,W,F

It must print `DONE failures per rank: [0, 0, 0, 0, 0, 0, 0, 0]`. `F` compares uint8 views of every output against `none`: AR outputs, mHC-like fp32 GEMM, router-like bf16 GEMM (cuBLAS Kernel2, the served kernel), shard GEMM, gather, q_b GEMM. It uses M = 1,2,4,6,8,12,16,24, 204 interleaved replays per mode per M, and 8 input sets. AR digests are compared across ranks.

## 5. GO and KILL

    ssh -n spark-one 'cd ~/dsv41-stage/g130/l2win && ~/venvs/vllm032/bin/python -I -B ops/verdict_l2win.py results/l2a.rank0.json'

Merge phase C first if it ran separately: `python3 -I -c "import json;a=json.load(open('results/l2a.rank0.json'));a['C']=json.load(open('results/l2c.rank0.json'))['C'];json.dump(a,open('results/l2a.rank0.json','w'))"` on spark-one.

| Gate | GO | KILL |
|---|---|---|
| Bitwise (F) | 0 mismatches on all ranks | any mismatch: that flag family is dead for good, not a noise question |
| Rate (C) | achieved >= 0.95 x requested, all bytes issued in every launch | below that |
| Late bytes (W) | per layer vs live: M<=8 at most +1.5 us, M>8 at most +3 us | above either |
| Gain (W) | saved >= 0.12 ms per step (40 x weighted us per layer; weights M2 .17, M4 .33, M6 .40, M8 .10) | below 0.06 ms; between is HOLD, rider only |

Prefer an env-only GO over a patched one of similar size.

Three more readings:
- `[B]`, gate class at M=4: the bench predicts +10 us for live, the trace shows +7. If live's stretch in B is off by more than 3 us from the trace, do not trust the bench's ranking between gate arms.
- `[B]`, `d*` modes: if stage returns to about 2.7 us but net grows by the same amount, the delay is moving the stretch rather than removing it. KILL d* regardless of small timing wins.
- `gather` rows in `[B]` are information only. Those arms are not GO-eligible from this kit, see co-residency below.

## 6. Install, only for a kit-module arm

Service stopped:

    bash $K/ops/l2win-all.sh install
    bash $K/ops/l2win-all.sh status   # 8 x e1fd76af0ea5c06c

The first control arm in section 7 then doubles as the flags-off check of the new module against today's numbers.

## 7. Pinned serving pair (restarts the service)

Only for one arm that passed section 5, or a bundle. CTTC, about 23 minutes per arm:

    bash $K/ops/pair.sh c1 C
    bash $K/ops/pair.sh t1 T DSV41_L2_PREFETCH_PACE_GBPS=125 DSV41_L2_PREFETCH_PACE_CHUNK=65536 DSV41_L2_PREFETCH_PACE_KINDS=ar
    bash $K/ops/pair.sh t2 T DSV41_L2_PREFETCH_PACE_GBPS=125 DSV41_L2_PREFETCH_PACE_CHUNK=65536 DSV41_L2_PREFETCH_PACE_KINDS=ar
    bash $K/ops/pair.sh c2 C

Every arm must show exact `a8c3394af9d28249`, digest `9a18f802d930f6e7` and `long 9 of 9 ALL==control`. A clean mismatch stops the arm. Promotion needs +0.5% G1 in both pairs. Alone this will often fall short: ship it in a bundle only if the bundle clears the bar and each flag passed its gates on its own.

## 8. Back to production

    bash $K/ops/l2win-all.sh uninstall    # only if section 6 ran, service stopped
    ssh -n spark-one 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > /tmp/rel.sh && bash /tmp/rel.sh --label g130-prod --expect-launcher 37757dde DSV41_GPU_CLOCK_MHZ=2000

Confirm exact `a8c3394af9d2`. Env-only arms leave no files behind. Release the lock.

## Co-residency and race reasoning

README L5468: TMA kernels were wrong only when another kernel's CTAs shared the SM, and exact.py cannot see that. So what matters is what each arm newly overlaps. Trace figures are prose, rank 0, 2026-11-02, see `analysis/`.

- **Today, in production:** gather bursts already run past AR end on about 18 windows per step, beside `_q_kv_norm_quant` (14.3 per step) and the MXFP8 `device_kernel` (9.4). AR-window bursts ended 8-31 us before AR end at the median.
- **`s85_c16`:** 1.18 MB at 85 GB/s is 14 us, inside every AR floor. No overlap with anything but `ar_kernel`, which the burst already overlaps.
- **`d*` at up to 3 us:** one-shot gate windows end 8.7 us before AR end at the median and 3.6 us at p90, today. If the AR shrinks 4 us as hoped, the p90 passes AR end by about 3 us, into `mhc_post_norm` and the quant kernel. The new spin kernel holds 1 CTA x 32 threads, registers only, no memory.
- **`p*` at gate rate:** 3.93 MB at 125-150 GB/s takes 26-31 us, against AR end at about 21-35 us. It can pass AR end by up to about 10 us. It can reach `mhc_post_norm` (+0.4 us), the quant kernel (+1.8), `mhc_proj` (+5.2, holds the whole register file, so it cannot share an SM) and Kernel2 (+5.4). `_x26` keeps issue clear of Kernel2 at the cost of the last chunk or two.
- **What the gate covers:** F's forced-overlap modes `p85_c16_d40` and `p150_c64_d40` delay the issuer into the consumers on every op, beside real cuBLAS Kernel2 and the stand-ins.
- **What it does not cover:** real `mhc_post_norm` and the quant kernel beside the pacer are not stress-tested here. g5 tested those kernels beside copy, LDS and MoE co-runners, not this one. Per the g5 table they have no TMA and no warp specialization, so the failure mode should not apply, but that is reasoning. The issuer is one CTA on one SM, so even the forced modes give no same-SM proof. A same-SM proof needs a persistent all-SM prefetch co-runner in `g5co.py`, about 5 GB to build and not done. If an arm that passes AR end is promoted, do that first, or use `s85_c16`, which does not need it.
- **Streams:** the pacer stream joins only at the end of the forward, so it never blocks the main stream. Consecutive pacers queue on it, and each table is kept alive by `S.keepalive`. A pacer never writes: `cp.async.bulk.prefetch.L2` only. Phase C checks the prefetched buffers are unchanged.
- **G2 path:** never served. CPU test [4] shows the same launches as the live module on a synthetic plan, with WOA. Real graph capture with three side streams is first exercised in section 7.
- **Fingerprint:** `DSV41_L2_PREFETCH_*` must be identical on all ranks. The supervisor passes VAR=VAL to all 8.
- **Overlap with other work:** rank 6 P1 doorbell (`~/Documents/dsv41-kits/g130-rdma`) hits the same first microseconds of the same bursts. Do not add the two estimates. Re-measure `[B]` if that kit lands first.

## Files

`analysis/` holds the three read-only sqlite queries and their outputs. `base/` is the live module. `patch/apply_l2win.py` refuses any base but 63737f9c. `site/` is its output, also as `patch/*.diff`.
