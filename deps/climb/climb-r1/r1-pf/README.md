# r1-pf: L2-prefetch levers G13, G15 and G11a (G29 refused under 08a249da, G01 dropped)

Climb round 1 kit. It has one installer, one patch of `vllm/distributed/device_communicators/dsv41_l2_prefetch.py`, and one new module `dsv41_l2pf_r1.py` placed next to it. Each lever sits behind its own `DSV41_L2_PREFETCH_R1_*` flag, and every flag is off by default.

**Production baseline: launcher `08a249da`** (8/8 since 2026-11-08 13:40:58Z, climb-w1): `DSV41_L2_PREFETCH_PACE_GBPS=125`, `PACE_KINDS=ar,attn,ag`, `PACE_CHUNK=65536`, `WOB_MB=6`. The kit was first written against `fda6d05a` (`PACE_KINDS=ar`, `WOB_MB=0`); the review fix (see `REVIEW.md` "Fixes") re-targets it:
- **G01 is dropped.** Its in-order wo_b fallback was superseded by the promoted paced wo_a+wo_b window. `DSV41_L2_PREFETCH_R1_WOB` is no longer a master flag and has no code behind it.
- **G29 refuses itself under 08a249da.** It is built for burst all-gather windows only. With `ag` paced, its tail would queue on pacer stream 3 ahead of the promoted wo_a+wo_b window (see the G29 section). It logs `INACTIVE` and changes nothing; it has no window arm on 08a249da.

| lever | flag | status | what it adds |
|---|---|---|---|
| **G13** (M2, L-DR, D5, C6a, M5), skeptic variant | `DSV41_L2_PREFETCH_R1_DRAFT=1` | built | Prefetch inside the DSpark drafter graph, for both graph sets (greedy-sharded included):<br>- every window paced at 125 GB/s<br>- wq_b moved to the previous window (qbprev)<br>- a drafter WOA window<br>- the first 5 MiB of the draft head in the last model all-reduce window<br>- no mHC fn hints<br>Also an eager E-phase pacer after the CERT screen: the main_proj split shard, then the ctxkv fold weight (18 MiB) |
| **G29** (M4) | `DSV41_L2_PREFETCH_R1_IDXTAIL=1` | skipped on 08a249da (refused, logged INACTIVE); code kept for burst-AG configs | With burst all-gather windows only (`ag` not in `PACE_KINDS`, e.g. fda6d05a): the rest of the indexer wq_b as a 2nd burst on side stream 1, after the head burst. Indexer-layer windows only |
| **G15** (L-B variant b) | `DSV41_L2_PREFETCH_R1_SHW2=1` | built | Shared-expert w2 (1.52 MB) as a burst in a post fork, right after the post-attention all-reduce |
| **G11 part a** (M3a) | `DSV41_L2_PREFETCH_R1_KVSRC=1` | partial: (a) only | The first 2.5 MiB of `compressor.fused_wkv_wgate`, paced into the preceding post-MoE all-reduce window (layers 2/8/14/20) |
| **G01, in-order form** (M1) | `DSV41_L2_PREFETCH_R1_WOB` | **dropped** | Nothing. Superseded by 08a249da's paced wo_a+wo_b window (`WOB_MB=6`). The flag is not a master flag: set alone it imports nothing; set next to a live lever it only shows up in the `levers` log line as `"wob": "dropped lever (G01); flag ignored"` |

Not built:
- G11(b), node priority / moving the join. It is outside this file, and the verifier says it needs a device test of the CTA dispatcher.
- G15(a), the cross-step expert prefetch.
- C6(b), persisting L2.

## Install (lead, 8/8)

```
python3 install.py install      # refuses unless the live base sha256 == 63737f9c413ebfa2… (KIT.json)
python3 install.py status       # prints a verdict line; exit 0 = INSTALLED or CLEAN, exit 2 = INCONSISTENT
python3 install.py dry
python3 install.py uninstall    # refuses (and changes nothing) after a post-install edit; --force overrides
```

What `install` does:
- It checks the kit module's sha against `kit_sha256` and the live base's sha against `base_sha256` before writing anything.
- It applies 4 anchors, each with an expected count of 1.
- It tags every inserted line `# r1-pf` (16 lines in total).
- It makes the backup `dsv41_l2_prefetch.py.bak-r1-pf`.
- It copies `dsv41_l2pf_r1.py` next to the base file.
- It checks the result against `patched_sha256` in KIT.json.

Running `install` a second time detects that the patch is already applied and checks it against patch(backup).

`uninstall` restores the base byte for byte, but only when the live file is exactly what the kit installed (`patched_sha256`, or patch(backup)) and the module is exactly `kit_sha256`. Both are checked before anything is touched. Otherwise it refuses, names the file, and leaves everything as it is. `--force` restores and removes anyway.

`status` prints each file's sha with what it matches (`= patched_sha256`, `= kit_sha256`, `= base_sha256`) and a final line `r1-pf status: INSTALLED | CLEAN | INCONSISTENT`. The new module is in neither the supervisor manifest nor the rdma fingerprint, so check 8/8 with:
```
for h in one two three four five six seven eight; do ssh -n spark-$h 'cd <kit> && python3 install.py status | tail -1'; done
```

File hashes:

| file | sha256 |
|---|---|
| base | `63737f9c413ebfa2c408b13cb0cbecddac2b42c6a5e219e6bad43bde095f4bee` (identical on spark-one..eight, checked read-only on 2026-11-08) |
| patched | `8bc9a1b5b73c808580a350fe8618fd6abb34a6f629de11c2053f640a7cd15596` |
| new module | see `KIT.json` (`kit_sha256`) and `TEST-RESULTS.txt` |

The patch differs from the reviewed one (`49f27151…`) in one tagged line only: `_R1_FLAGS` (line 1176) lists the 4 live master flags; `DSV41_L2_PREFETCH_R1_WOB` was removed.

The base is in the supervisor manifest and in the rdma fingerprint (`l2pf=` sha), so the install must be 8/8. The new flags start with `DSV41_L2_PREFETCH`, so the fingerprint covers them as well. No `sitecustomize` or `transform_apply` source transform touches these files. The only one live is the FC2 trial, and it patches `DeepGemmFP4Experts.apply`.

## Base edits (patched file line numbers)

| lines | edit |
|---|---|
| 1174-1183 | At module end:<br>`_R1 = None`<br>`_R1_FLAGS = (R1_DRAFT, R1_IDXTAIL, R1_SHW2, R1_KVSRC)`<br>`if ENABLED and any(master flag on): import dsv41_l2pf_r1 as _R1; _R1.bind(this module)` |
| 322-323 | In `install_model_module`:<br>`if _R1 is not None: _R1.install(mod)`<br>This runtime-wraps the call sites below. |
| 783-784 | In `make_plan`, right after `make_g2_plan`:<br>`if _R1 is not None: g2 = _R1.augment_target_g2(g2, events, sel, device)` |
| 1064-1065 | In `_fork_part`:<br>`if _R1 is not None and part[0] == "x": return _R1.fork_part(part, where)`<br>This handles parts with their own rate or stream. |

The new module also wraps three call sites at runtime. It does this only when the matching flag is on, and it edits no file:

| call site | lever | what the wrapper does |
|---|---|---|
| `DSparkSpeculator._generate_draft` and `_sample_sequential` | G13 | Brackets the drafter forward. Learns at each eager warm-up and plays back at the capture that follows. |
| `dsv41_r4_small.certified_logits` | G13 E-phase | Forks the E-phase pacer after the CERT screen. |
| `DeepseekV4Attention._run_parallel_input_projections` | G11a | Records the compressor weight in learn mode. |

## Why flag-off is identical

- With no master flag set, the import block does not run, so `_R1` stays `None`.
- The 7 inserted statements in function bodies are all `if _R1 is not None:` branches, so none of them execute.
- No module is imported, no hook is installed and no stream is created. Kernels, launch order, streams, events, allocations and graph contents are unchanged.

The tests check this three ways:
1. Removing the tagged lines from the patched file gives the live base byte for byte.
2. With flags unset, `dsv41_l2pf_r1` is never imported, and `DSV41_L2_PREFETCH=0` keeps it off even when the r1 flags are set.
3. A full simulated run compares the live base with the patched file, flags off, under 7 environments: production 08a249da, the earlier fda6d05a, fda6d05a + `WOB_MB=6` + `PACE_KINDS=ar,attn`, `REPLAN=qbprev`, a no-G2 config, sub-knobs with the masters unset, and the dropped `R1_WOB=1` alone. The run covers target learn+capture, drafter learn+capture for both sets, and an eager E-phase step. Every launch, every `wait_stream` and every log line is identical.

## Why flag-on is bit-exact

Every lever only launches the live hint kernels:
- `dsv41_l2pf`, one CTA per segment: LDG of the segment table, then `cp.async.bulk.prefetch.L2`.
- `dsv41_l2pf_paced`, one warp; its `stats` pointer is always `None`.

Both run on side streams. The G01 exactness verifier checked their SASS: no STG, ATOM or RED, so they write nothing.

Per output element:
- Main-stream kernels, their inputs, tactics, tiles, M and launch order are untouched.
- No reduction order, split, rounding point, verify width or draft changes.
- The drafter's drafts are computed by the same kernels on the same data, so acceptance and the digest cannot move.

Only cache state and timing change. What changes per lever:

| lever | change |
|---|---|
| G13 | Adds paced hint kernels inside the two drafter graph sets, plus one eager hint kernel per step. |
| G29 | Only with burst all-gather windows: adds one hint kernel per indexer layer on side stream 1. On 08a249da: nothing (refused). |
| G15 | Adds one post fork per layer on side stream 1. |
| G11a | Adds 4 paced hint kernels on side stream 3. |
| G01 | Dropped: nothing. |

## Race analysis

- **No consumer.** Nothing consumes what a hint kernel produces. The only device data a hint kernel reads is its segment table, an int64 tensor built at learn time (or at the first E-phase call). That is eager and outside capture. The table is kept alive in `S.keepalive` and never rewritten.
- **Forks and joins.** Every fork is `side.wait_stream(main)`, an event recorded at the fork point. Joins follow the base rules:
  - side stream 1 (burst windows, the G29 tail when active, the G15 post fork) joins at the next collective's `before_collective`;
  - side stream 2 (burst WOA windows) and side stream 3 (all paced parts: on 08a249da the AR, AG and WOA windows, plus G11a and the drafter windows) join at the end of the bracketed forward.
  - The drafter bracket calls the base `_join/_join2/_join3` in `finally`, so a capture never ends with an open branch.
- **Plan matching.** The two DSpark graph sets get separate plans, keyed `("r1-dspark", greedy?, num_reqs, tokens)`. So the greedy-sharded set (5 Markov all-gathers) never plays back the stock set's plan. The test makes the two sets differ in collective count to prove this: a shared key would log MISMATCH.
- **E-phase stream.** The E-phase uses its own stream, created once, eagerly. It is never captured, which is checked with `is_current_stream_capturing()`, and it is skipped inside a bracket. It is not joined, because nothing consumes it and it allocates nothing. It ends within `EPH_MAX_US=200` µs, long before the next target graph.
- **Fail closed.** A planning exception inside the module switches the affected levers off and keeps the base plan: target planning builds every table before it attaches any part (`augment_target_g2`), and a `make_drafter_plan` failure at a drafter warm-up turns `draft_graph` off, drops that set's plan and lets later warm-ups and captures run unbracketed. Graphs already captured keep their (joined) hints. Both paths are tested.
- **Allocations.** None inside capture. Plan tables are allocated at the eager warm-up, as the base does.
- **SM sharing.** The new risk is a longer hint CTA sharing an SM with a kernel that lacks TMA fences (README L5468 class). The G01 verifier checked the exposed kernels: `_sm120_wo_a_partial`, `wo_a_reduce`, MXQ and inv-rope. None of them uses cp.async.bulk, mbarrier, fence.proxy or atomics. The TMA GEMMs and sparse MLA fill the register file, so no hint CTA can co-reside with them. An optional g5co co-run is listed in the device checks.

## Expected saving (critical path) and the trace evidence

G1 changes by about +0.43% per 0.1 ms/step. Sources: `audit-20261106/data/verdicts-r2.json` and the `work/verify-*/NOTES.md` files.

**G13: 0.03 / 0.07 / 0.13 ms, G1 +0.13 / +0.30 / +0.56%.**
- The drafter graph, 2.10 ms serial per step, has 0 l2pf nodes today. Its kernels run cold:

  | kernel | drafter (cold) | same shape in the target (prefetched) |
  |---|---|---|
  | wq_b | 27.3-28.3 µs | 12.0-12.3 µs |
  | wo_a | 22-24 µs | 13.2 µs |

  The draft head takes 400 µs, streaming from DRAM.
- Paced AR windows show no stretch: 28 µs with the pacer against 29.5-30.3 µs without.
- A burst all-gather window would cancel the wq_b gain, hence qbprev.
- E-phase: main_proj MXG(1,5) takes 64.7 µs and ctxkv 44 µs, with about 160 µs of DRAM-light time after the screen.
- Removing the target mHC fn hints was worth +1.47%, so the drafter records none. Markov-embedding lookups are also excluded: markov_w1 is a 63 MiB nn.Embedding that the base hooks would otherwise record.

**G29: 0 on 08a249da (refused).** On fda6d05a (burst AG windows) the estimate was 0.03 / 0.065 / 0.09 ms, G1 +0.13 / +0.28 / +0.39%:
- Each step runs 8 indexer wq_b GEMMs at 24.0-24.9 µs, against 12.2 µs for the prefetched attention wq_b.
- The AG window's 6 MiB holds the attention wq_b (5.16 MiB) and only the first ~0.84 MiB of the indexer wq_b.
- The in-order tail keeps the attention wq_b at full burst rate. The L9 failure mode came from running segments in parallel.

Why it is refused on 08a249da: there the AG window is itself paced on side stream 3 (6 MiB at 125 GB/s, about 50 µs), and at the indexer wq_b post-hook the promoted WOA window (wo_a + wo_b, 9.28 MiB, paced) is launched on the same FIFO stream. A tail on side 3 would sit between them: it could start only about 50 µs after the AG fork, it would likely land after the indexer wq_b read, and it would delay the promoted wo_b prefetch by about 36 µs (4.4 MiB at 125 GB/s) in each of the 8 indexer layers. A burst tail on side 1 at the AG fork would instead run ahead of the paced head (out of read order) and burst during the all-gather, which is what climb-w1's AG pacing removed. Neither can be justified without a device trace, so the kit refuses: `bind()` logs `INACTIVE: {"idxtail": "built for burst all-gather windows; ..."}` and the plan is unchanged (tested: launch sequence identical to flag-off under 08a249da and under `PACE_KINDS=ar,ag`). A re-plan for paced AG windows (for example, appending the indexer wq_b remainder to the base paced AG part with a matching budget, or a delayed tail on side 2) is left to a later round with an nsys trace of the 08a249da indexer layers.

**G15: 0 / 0.13 / 0.22 ms, G1 0 / +0.56 / +0.95%.**
- FC1 is DRAM-bound at about 230 GB/s.
- The shared w2 (1.52 MB) is read cold during FC1 and costs up to 6.6 µs per layer.
- The front chain has about 4.5-5.4 MB of spare DRAM, nearly idle for the first 7 µs.

**G11a: about 0.01-0.04 ms, G1 0 / +0.08 / +0.17%.**
- In layers 2/8/14/20 the skinny waits 47/49/42/25 µs behind the 1024-CTA compressor Kernel2, which streams 10.5 MB in 54 µs.
- The post-MoE AR runs 28-37 µs, but its pacer is busy for only about 13 µs. 2.5 MiB at 125 GB/s takes about 21 µs.
- The verifier's realistic net for part (a) alone is 10-40 µs per step.

**G01: dropped.** 08a249da already prefetches wo_b in the paced WOA window (`WOB_MB=6`, climb-w1 W1b, +3.8% G1 together with paced AG). An A/B of the in-order form would need `WOB_MB=0` in the treatment arm, which reverts a promoted change; it is not a valid single-flag arm.

**Kit total on 08a249da, not additive: 0.13 / 0.8 / 1.6%** (G13 + G15 + G11a; G15 and G11a share the MoE front chain).

## Reading the effect in nsys (mid-request window, q*.2/.4/.6)

**G13**
- The drafter graph launch gains `dsv41_l2pf_paced` kernels on one side stream: one per non-empty drafter window (collective windows plus the 3 WOA windows). The `R1_DRAFT plan` log line gives the exact count.
- Kernel times:

  | kernel | now | expected |
  |---|---|---|
  | drafter wq_b, grid (1,32) | 27-28 µs | 12-16 µs |
  | drafter wo_a_partial | 22-24 µs | 13-16 µs |
  | draft head 1×127 | 400 µs | 385-392 µs |

- The drafter ARs (about 30 µs) and AGs (13-16 µs) must not grow, and the drafter graph wall should drop from 2.10 ms toward about 2.03 ms.
- E-phase: one `dsv41_l2pf_paced` of about 150 µs starts right after the CERT refine kernel. main_proj MXG(1,5) should drop from 64.7 to 30-45 µs, and ctxkv MXG(1,12) from 44 to 18-30 µs.
- The argmax AG must not stretch.

**G29** (only on a burst-AG launcher; on 08a249da there is nothing to see)
- In the 8 indexer layers, a second `dsv41_l2pf` (`IDXTAIL_SPLIT` CTAs, default 2) appears on side stream 1 right after the head burst, about 20-25 µs long.
- The indexer wq_b (between k-norm and `paged_mqa_logits`) should drop from about 24.5 to 12-18 µs.
- These must not change: the attention wq_b (about 12.2 µs), the qkv_a AG (30-31 µs), and the post-attention AR start, which must not wait on the side-1 join.

**G15**
- A burst of about 8 µs appears right after every post-attention AR, on side stream 1.
- The shared w2 GEMM (side stream, overlapping FC1) should speed up, and FC1 should drop by up to 4-5 µs per layer.
- The router GEMM must not slow down. It is on the critical path.

**G11a**
- In layers 2/8/14/20, a second paced kernel of about 21 µs follows the post-MoE window's pacer.
- The compressor Kernel2 should shorten, and the skinny delay after the qkv_a quant should shrink from 47/49/42/25 µs.
- The post-MoE AR must not grow.
- 08a249da-specific: side stream 3 now also carries the paced qkv_a AG window of the same layer. Check that this window's pacer starts at its AG fork and is not queued behind the G11a part (the part should end about 34 µs after the post-MoE AR fork, before the input projections reach the AG).

## rank0.log lines to check

- `DSV41_L2_PREFETCH_R1 (r1-pf) levers: {...}` must have no `INACTIVE`. A refused lever says why. The refusals:
  - G15 needs `PRUNE_SHARED=1`.
  - G11a needs paced AR windows.
  - G29 refuses when `ag` is in `PACE_KINDS` with `PACE_GBPS>0`, i.e. always on 08a249da.
  - A set `DSV41_L2_PREFETCH_R1_WOB` shows `"wob": "dropped lever (G01); flag ignored"`.
  - Every target lever needs G2 mode.
- `g2 plan tokens=N {...} r1={"kvsrc_n":4,"shw2_n":40,...}` (plus `idxtail_n":8` on a burst-AG launcher). The following `g2 captured ... forks (planned)` line must show equal counts.
- Fail-closed warnings that must NOT appear: `DSV41_L2_PREFETCH_R1: target planning failed` and `R1_DRAFT: drafter planning failed`.
- G13:
  - `R1_DRAFT plan set=stock|greedy reqs=R tokens=T collectives=C wq_b=3 ...`
  - `R1_DRAFT captured ... forks X (planned X)`, with no MISMATCH, for every shape in both sets
  - `R1_DRAFT E-phase table: main_proj_split:... ctxkv_fold:...` and `E-phase forks: 1`

## Window arms

These are the env arms of `KIT.json` `window_arms`, on top of the 08a249da launcher defaults:

| arm | env |
|---|---|
| G13 | `DSV41_L2_PREFETCH_R1_DRAFT=1` |
| G13 graph only | add `DSV41_L2_PREFETCH_R1_DRAFT_EPH=0` |
| G13 E-phase only | add `DSV41_L2_PREFETCH_R1_DRAFT_GRAPH=0` |
| G15 | `DSV41_L2_PREFETCH_R1_SHW2=1` |
| G11a | `DSV41_L2_PREFETCH_R1_KVSRC=1` |

No G29 arm (refused on 08a249da) and no G01 arm (dropped). Run G13 apart from r1-drafter's `MAINPROJ_HOIST` / `DRAFT_PROLOGUE_FORK` and from r1-attn's `IDXPAR`: their composition with r1-pf is untested.

## Risks

- **G13: windows skipped on mismatch.** If the drafter path changes between warm-up and capture, the remaining windows are skipped and logged. The cost is performance only.
- **G13: stale learned plans.** Another kit may change the drafter's collective sequence between eager and captured runs. This kit's plans are re-learned at every warm-up.
- **G13: wrapper overridden.** If a drafter kit replaces `_generate_draft` at runtime after this hook, the bracket is lost. You can see it: no `R1_DRAFT captured` lines.
- **G13: drafter collectives must reach the hook.** The plan is learned from the RdmaAllReduce `before_collective` calls made inside the drafter forward. If the drafter's collectives do not go through that hook, `S.idx` stays 0, no plan is made and there is no gain (speed only). The sign is no `R1_DRAFT plan` lines.
- **G13: E-phase table build.** The first E-phase call builds its 4-segment table with `torch.tensor(..., device=cuda)`: one H2D copy with a host sync, once per process, eager, outside capture. Time only.
- **G13: composition.** r1-drafter's `MAINPROJ_HOIST` and `DRAFT_PROLOGUE_FORK` add drafter streams and touch main_proj, which the E-phase prefetches; r1-attn's `IDXPAR` forks from the wq_b post-hook like `fork_part`. Untested together: A/B separately.
- **G13: E-phase timing.** The E-phase starts after the screen and before the argmax AG. If the sampler phase is shorter than modelled, ctxkv arrives only partly.
- **G13: unknowns.** The verifier's biggest unknown is how fast the 5-CTA and 12-CTA GEMMs run from L2.
- **G29.** Refused on 08a249da, so no risk there. On a burst-AG launcher: the tail adds DRAM traffic next to the wq_b GEMM, QNormRope and the compressor, and if the join at the post-attention AR ever waits on side 1, the AR is delayed (model: tail done at about +60 µs, join at +170-200 µs).
- **G15.** The prefetched lines must survive FC1's evict-first stream, and the router GEMM must not slow down. Kill the arm if it does.
- **G11a.** A small AR stretch is possible: the verifier measured +10 µs for a 1.6 MB burst. Here the bytes are paced, but the pacer runs about 21 µs longer. On 08a249da the same FIFO stream 3 also carries the next paced AG window; if the input projections reach the qkv_a AG within about 34 µs of the post-MoE AR fork, that window starts late (see the nsys check).
- **All levers.** No arithmetic risk: they are hints only. The payoff risk is the usual one for L2 work: L4 and L9 were flat.

## Tests (CPU, Mac Studio 512S1, torch 2.11 CPU)

Run `python3 tests/test_r1pf.py` (17 tests) and `bash tests/installer_cycle.sh <copy of the live base>`. The output is in `TEST-RESULTS.txt`.

`tests/harness.py` loads the live base, or the patched copy, under its real package name with stub vllm modules. Its `PROD_ENV` is the 08a249da l2pf config (`PROD_FDA_ENV` keeps fda6d05a). It replaces `torch.cuda` with a recorder (streams, `wait_stream`, capture flag) and the two native libraries with recorders. It then drives the exact hook protocol of the live code:
- pre-hook / sonar / r3 records only in learn mode;
- `_wq_b_post_hook`;
- `before_collective`, plus `after_collective` in G2 mode;
- vLLM's warm-up-then-capture order, for the target and for both DSpark graph sets.

The tests cover:
- the installer and anchors on the live base, including `uninstall` refusing after a post-install edit of either file (nothing touched), `--force`, and the `status` verdicts and exit codes;
- flag-off identity: byte-strip, never imported (also with only the dropped `R1_WOB=1` set), and an identical launch and log sequence under 7 environments;
- selection equivalence with `make_plan`;
- per-lever plan and launch-order tests, under both 08a249da and fda6d05a where the lever is active: G29 refusal under paced AG (identical launches) and, on burst AG, tail position, stream, bytes and split; G15 post fork and join; G11a window choice, including the Engram layer; G01 dropped (no effect, no code); G13 per-set plans, qbprev, head, no mHC or Markov hints, all paced, join at the end, E-phase table, never in capture;
- all levers together, under both launchers;
- the reviewer's stream-protocol checks for every lever alone and all together, under both launchers, in every captured section (target 4/6, stock and greedy 5/10): every side launch follows a `wait_stream(main)`, every side stream is joined before the end, a side-1 launch is joined at the next collective, and the base launches are an in-order subsequence of the patched ones;
- fail-closed: an exception in target planning or in `make_drafter_plan` leaves the flag-off launch sequence and logs one warning.

The tests do not cover:
- real kernels and real module hooks. The event protocol is emulated, so the first device log lines above are the real check.
- timing.
