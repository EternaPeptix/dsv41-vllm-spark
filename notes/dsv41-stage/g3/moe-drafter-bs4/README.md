# g3 / moe-drafter-bs4: where the 4-user step goes, exact MoE headroom at M=12-24, drafter cost at batch 2-4

Goal window 3 offline prep, 2026-10-26. Analysis only from existing data, no GPU, no restart. The two top items are
staged as flagged patches with tests next to this file: `../dhtac/` (drafter) and `../mfront/` (MoE).

Sources: Codex's nsys profile on 2027f4d8 (`~/dsv41-goal120-20261024/profile-control/r0.{1,2,3}`, rank 0, C1 / C2 / C4,
pinned AV, 30 target + 30 drafter graphs each); the k1 profile (10-24); the m32-bw bench (10-26 09:54,
`~/dsv41-stage/g1/m32-bw/results/`); the g2 fc2c bench (10-26 14:02); the pd Marlin bench (`~/mm-marlin-20261026/out/`);
the gw2 pinned A/B table (4-session step 46.02 C -> 45.13 ms X at 2000 MHz); the live rank-0 log (17:49 launch).

## 1. The 4-user step today (~45.1 ms at 2000 MHz, M ~12-16 rows)

No 4-user trace exists on the current config. The C4 trace (50.4 ms, M=12 in 26/30 graphs) predates D05, M32 'all',
FC2C and stepovh. Each of those has a measured or benched 4-user delta (D05 ~-1.9, 'all' -2.7 serving (bench
c4rec M=12 -75 us/layer x 0.9), FC2C -0.70 (gw2 pinned pairs), stepovh -0.51 (gw2), PRUNE_SHARED ~-0.1), which sums
to -5.9 ms -> 44.5 ms against the measured 45.1. Rolled forward:

| component | C4 trace (2027f4d8) | now (est.) | share | notes |
|---|---|---|---|---|
| target routed MoE GEMMs (FC1+FC2, 40 layers) | 24.10 | ~18.4 (17.5-19) | 41% | 18.9 from the trace deltas; 18.0 from the c4rec bench (15.6 ms GEMMs) x the trace/bench ratio 1.155 at equal code; FC1 ~60% |
| MoE glue (topk, count, fills, scatter x2, quant, pack, SiLU, gather) | 1.37 + front | ~1.4 | 3% | ~34 us/layer; 22 us/layer of it is pure data movement (mfront) |
| dense GEMMs (exposed; shared experts overlap FC1/FC2) | 9.96 (2.04 overlapped) | ~7.5 | 17% | above 8 rows q_b is no longer L2-prefetched (34 vs 12.5 us/layer: +0.86 ms; the replicated 9.46 MB qkv_a fills the 6 MB window). The replicated qkv_a itself (43.6 us) costs about what the 1-user shard + all-gather does (12 + 30 us) |
| collectives, exposed (rdma_ar, ~125 calls) | 5.96 | ~6.0 | 13% | RoCEnante task |
| drafter graph | 4.68 | ~4.7 | 10% | head 1.10, MoE 1.49, other dense 0.71, AR 0.50, mHC 0.18, rest 0.7 |
| mHC | 2.36 | ~2.4 | 5% | |
| attention + indexer | 2.25 | ~2.3 | 5% | |
| host / eager outside graphs | 1.47 | ~1.1 | 2.5% | CERT screen 0.51; SYNC_FREE removed the [M,16160] logits gather |
| other / Engram / idle in graph | 0.98 | ~1.0 | 2% | |
| **sum** | **50.4** | **~44.8 (measured 45.1)** | | +-1 ms: route diversity of the G4 texts, nsys overhead |

Per co-batched session (prose-c4 analysis): +9.0 ms (code joins), +6.2 (mixed), +6.1 (math). Almost all of it is MoE
(distinct experts grow ~linearly with rows at M <= 24), plus the q_b prefetch loss above 8 rows at the first join and
the drafter bs2 -> bs4 jump at the last.

First action of the next stopped window: a fresh 4-user nsys trace on 592793fd (k1trace launcher pattern, pinned AV,
`--wait=primary`), to replace the roll-forward with measured numbers.

## 2. MoE at 4-user row counts: M32 'all' + FC2C vs Marlin and vs streaming

Per routed-MoE layer on recorded 4-user batches (c4rec), us. Marlin numbers are normalized to the m32bw harness by the
two benches' common `live` arm (507.1 -> 498.4 at M=12, 819.5 -> 811.1 at M=24).

| | c4rec M=12 (34 experts) | c4rec M=24 (56 experts) |
|---|---|---|
| pre-'all' live (D05) | 498.4 | 811.1 |
| **M32 'all' + FC2C (live today)** | **409.1** (205 GB/s incl. glue) | **654.6** (211 GB/s) |
| Marlin W4A16, same layout (ma64, 2.61 MB/expert) | 406.6 | 658.5 |
| Marlin W4A16 unpadded (munpad, 2.44 MB/expert ~ our bytes) | 375.7 | 606.8 |
| weights-only streaming at 238 GB/s (our 2.45 MB/expert) | 352 | 580 |

- Distance to Marlin on the same bytes: **33 us/layer at M=12, 48 at M=24 (7-8%)** = 1.3 / 1.9 ms/step bench, about
  1.2 / 1.7 ms serving. Distance to the streaming bound: 57 / 75 us/layer (11-14%), 2.3 / 3.0 ms/step.
- Inside the live layer at M=24 ('all' profile, pre-FC2C): FC1 393.8 us = 88.4 MB at **224 GB/s (94% of 238)**; FC2
  270.6 us = 58.9 MB at 218 GB/s; with FC2C ~247.8 us = 49.7 MB at **~200 GB/s (84%)**; glue 23.1 us in 8 kernels.
- So FC1 is already near streaming. The exact headroom left is: (a) glue launches (~14-22 us/layer, of which ~9-16
  removable), (b) the FC1 -> SiLU -> FC2 serialization bubble (FC1 wave tail + SiLU launch + FC2 ramp, est. 10-25
  us/layer at M 12-24), (c) FC2C's lower streaming efficiency (~20 us/layer at M=24 vs FC1's rate).

## 3. Exact MoE kernel changes for M=12-24, ranked

| # | change | est. saving | exactness | effort | status |
|---|---|---|---|---|---|
| 1 | **mfront: fused permute** (count + fills + 2 scatters + FC1-internal UE8M0 pack -> 2 kernels) | 9-16 us/layer: -0.4 ms/step 1u, -0.55..-0.65 4u (G1 ~+1.3%, G4 prose MIN ~+1.2..1.6%) | by construction: byte copies + the same pack formula; row order inside an expert segment already arbitrary (atomics) | done | **staged ../mfront** |
| 2 | FC1 -> FC2 per-expert pipelining: one persistent kernel (or FC2 with per-expert ready flags) starts an expert's FC2 tiles when its 9 FC1 tiles and its SiLU-quant rows are done; SiLU-quant done by the last FC1 CTA of the expert | 10-25 us/layer at M 12-24: -0.4..-1.0 ms/step 4u, -0.2..-0.4 1u | same tiles / K order; SiLU-quant must reproduce `_silu_mul_quant_fp8_packed_kernel` bit for bit (Triton exp2 path) -> exhaustive bf16 check | 1-2 weeks | proposal |
| 3 | fold the activation quant into the mfront scatter (bf16 in, FP8 + UE8M0 out) | ~3 us/layer, -0.12 ms/step | needs the CUDA quant kernel's exact UE8M0 rounding replicated (log2f/ceil vs exponent bits differ just above powers of two) | days | proposal |
| 4 | FC2C contiguous tile layout (64 x 160 B per tile in one TMA box instead of 2 strided 64-B boxes + a 32-B tail) | ~20 us/layer at M=24, ~12 at M=12 (-0.5..-0.8 ms/step 4u) | same smem image, same MMA | needs the padded w2 to go (prefill on the compact layout, -2.3 GiB) or +12 GiB/rank (impossible): ties to prefill qualification | proposal |
| 5 | CTA scheduling for uneven expert loads | <= 5 us/layer | exact (scheduling only) | | **deprioritized**: at M <= 24 every expert is exactly one 32-row m-block, all tiles carry the same weight bytes, and the static round-robin is balanced up to the last wave (FC1 306 / 508 tiles on 48 CTAs); 8 tail CTAs still keep >200 GB/s in flight (m32-bw README). Only matters for experts with > 32 rows (M >= 48 or hot routes) |
| - | split-K / smaller K tiles for FC1 at small M | - | **not exact** (changes the fp32 accumulation order) | | excluded |

Microbench plans (all on the existing harness pattern: 40-layer CUDA graphs cycling layers 0/13/26/39 L2-cold,
recorded 10-21 routes M 1/6/12/24/48 + c4rec 12/24, ABAB 25 trials, A/A control within 1%, torch.profiler kernel
list at M 6 and 24, projection x0.96 (1u) / x0.90 (4u)):
- #1: `../mfront/tests/test_mfront_gpu.py --bench` (written). Gate >= 8 us/layer at c4rec M=12.
- #2: first measure the bubble: nsys `--cuda-graph-trace=node` of the bench graph at c4rec M=12/24, FC1 end -> SiLU
  -> FC2 first-CTA start per layer, and the FC1 last-wave duration. Build only if bubble + SiLU >= 10 us/layer. The
  candidate is benched against live with the same harness plus a bitwise unit test (all 40 layers, M 1..96, route
  kinds as in g2) and an exhaustive SiLU-quant table test (65536 bf16 gate values x 64 up values x clamp).
- #3: GPU table test of the reference quant over all exponents and mantissa edges (amax exactly 2^k, 2^k(1+2^-23),
  zeros, denormals, 448-boundary) before any bench.
- #4: FC2-only bench (the g2 `fc2` timing block) on a repacked layout at M 12/24/48, then a prefill-size sweep.

## 4. Drafter at batch 2-4

**The premise "with 3 requests the drafter still runs the bs4 graph" does not hold.** DSpark captures 22 FULL graphs
per set (stock + greedy-sharded; the live log shows 22 + 22): the capture sizes rounded up to multiples of 5 query
tokens give batch sizes 1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 15, 16, 18, 20, 21, 23, 24, 26, 28, 29, 31, 32, and
`dispatch()` maps 15 tokens to the 3-request graph. What is bs4-shaped at 3 requests is only the **AV cost model**:
draft_ms is profiled at [1, 2, 4, 6, 8, 12, 16, 24, 32] and stepped up, so a 3-request step is priced at the 4-request
draft cost (live 17:51 profile: 4.25 / 4.76 / 7.36 ms for bs 1 / 2 / 4). Pricing bs3 separately would change AV
budgets at 3-request steps, hence verify widths, hence target numerics at 4 users: not bitwise, excluded
(it would be a user decision like the prose-priority verify width).

Why bs2 -> bs4 costs +1.44 ms (trace; +2.6 ms in the AV profile, whose random routes touch more drafter experts):

| drafter kernel group | bs1 (C1) | bs2 (C2) | bs4 (C4) | bs2 -> bs4 |
|---|---|---|---|---|
| draft head `[M x 5120] x [5120 x 16160]` MXFP8 | 423 (tactic 1, 127 CTAs) | 433 (tactic 1) | **1104 (tactic 0, 505 CTAs, 77 GB/s)** | **+671** |
| drafter MoE FC1/FC2 (3 layers, E128 top-3, generic 128-row DeepGEMM, FC1 padded 768) | 546 | 926 | 1491 | +565 |
| other MXFP8 dense (qkv_a replicated above 8 rows, o-proj, ...) | 495 | 682 | 706 | +24 |
| rdma_ar | 394 | 393 | 499 | +106 |
| mHC fused (grid 5x12x8 / 10x12x8 / 20x4x8) | 100 | 127 | 177 | +50 |
| everything else | ~700 | ~680 | ~700 | ~+20 |
| **drafter span** | **2657** | **3241** | **4675** | **+1434** |

The head jump is an autotune artifact: the serving cache (8c8be689, loaded since 10-23) gives the head shape tactic 1
at buckets 1/2/8/16 but **tactic 0 at buckets 32/64/128**, i.e. every DSpark step with 4-25 requests.

Exactness rule for drafter changes: draft tokens and DSpark confidences feed the AV budget, so at 4 users any bit
change in the drafter changes verify widths and target numerics. exact.py and G1 are 1-user (head M=5, tactic 1)
and cannot see bs>=4 changes, so every drafter patch must prove bitwise drafter outputs itself.

Exact drafter cuts at bs4, ranked:
1. **dhtac** (staged ../dhtac): pin the head to the fastest tactic that is bitwise-equal to the serving one (all six
   CUTLASS SM120 MXFP8 tactics share one MMA chain per element: no split-K / stream-K; enforced by a startup
   self-check, a GPU equivalence-class test per rank and an audit mode). About **-0.65 ms per 4-request step**
   (~+0.9% G4 prose MIN), plus bs5-25 steps at higher concurrency.
2. Drafter MoE onto the M32 path (32-row tiles + compact FC1 576 + D05 / 'all' / FC2C): ~1491 -> ~1000-1100 us at
   bs4 (-0.4..-0.5 ms), -0.15 ms at bs1 (+0.5% G1). Needs an E128 / topk-3 instantiation of the M32 natives (the
   m32-bw 'all' features are compile-time gated to the kG1 E384 geometry), a compact drafter w13 at load (-189 MB,
   tails +63 MB) and a bitwise qualification M32 vs the current 128-row path at E128 (as was done for E384).
   Not staged: native build + 2-3 days.
3. Same tactic mechanism for the DSpark main_proj split `(M, 15360) x (15360, 640)`: the autotune cache has no entry
   above bucket 16, so target batches of 17-128 rows fall back to tactic 0 (20 CTAs; the AV profile logs the miss at
   M 24/40/72). Extend dhtac to it once the GPU test confirms tactic equivalence (the split already self-checks
   bitwise per M against the replicated GEMM).
4. Drafter qkv_a column shard above 8 rows (3 calls x ~45 us at bs2-4): only if the skinny shard is bitwise at
   M 10-20; same question as the target's QKV_A_SHARD_MAX_M (dense owner).
5. "Sharded head at bs4": the head is already vocab-sharded (16160 rows / rank, skip_gather, 16-B (max, id)
   all-gather per position). The 5 Markov steps' all-gathers are sequential by construction (each argmax feeds the
   next bias); no exact cut found.

## 5. What the staged items buy (pinned A/B still to run)

| item | 1 user (G1) | 4-request step | G4 prose MIN (prose-c4 sensitivities) |
|---|---|---|---|
| mfront | -0.4 ms (~+1.3%) | -0.55..-0.65 ms | ~+1.2..1.6% |
| dhtac | 0 | -0.65 ms (bs4 only) | ~+0.9% |
| both | ~+1.3% | ~-1.25 ms | **~+2.1..2.5%** |

Against the gap (G4 prose MIN +4..8%, G1 +19.5%) this is a third to a half of the G4 gap. The next exact candidates
by size: FC1 -> FC2 pipelining (#2 above, -0.4..-1.0 ms at 4u), an L2-prefetch plan that keeps q_b at 9-24 rows
(cache hints only, so exact by construction; up to -0.86 ms on every 3-4 request step; dense owner), drafter MoE on
M32 (-0.45 ms at bs4), and the collectives (RoCEnante task).

## Files

- `../dhtac/`, `../mfront/`: staged patches (README, FILES, make_patch.py, base copy, site file, CPU + GPU tests,
  ops/{site,all,prep-node,run-gpu-tests}.sh, results/cpu-test.txt).
- Mac scratchpad `g3md/`: profile summaries (`prof/`), copied bench results (`m32bw/`, `fc2c/`), sources read (`src/`).
