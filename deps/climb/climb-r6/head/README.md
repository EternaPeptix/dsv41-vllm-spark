# r6-head: certified LM-head refine load balance + screen dequant (climb round 6)

Production launcher 166d7aac, live `dsv41_r4_small.py` 3364d8c8 (`DSV41_R4_CERT_HEAD=1`, `_M=2-96`,
`DSV41_SYNC_FREE=argmax`). One live file patched (+7 tagged lines), one new module. Default off, inert when off.

| lever | flag | status | ΔG1 % low / central / high | ms/step (G1-weighted) |
|---|---|---|---|---|
| HEADREF: refine as candidate list + persistent CTAs (`row`; `tile` = pure bitwise) | `DSV41_R6_HEAD_REFINE=row\|tile` | built | −0.02 / 0.08 / 0.16 | −0.005 / 0.017 / 0.035 |
| SCREENDQ: exact integer dequant + launch-shape sweep | `DSV41_R6_HEAD_SCREEN=1` (+`_DQ`, `_CFG`) | built | −0.02 / 0.09 / 0.37 | −0.004 / 0.020 / 0.080 |
| CERTPRE: cert-head L2 pre-stage in the last MoE-AR window | none | partial: t2 evidence, design and device micro-bench; no hook | 0.03 / 0.14 / 0.25 | 0.006 / 0.030 / 0.052 |

**What ran and what did not.**

Ran, all passing:

| test | where | result |
|---|---|---|
| `tests/test_install.py` | Mac | 52/52 |
| `tests/test_cfg_guard.py` (tuning-flag guards + 7 must-fail mutants) | Mac (stub torch) and 512S1 (torch CPU) | 54/54 both |
| `tests/test_dequant.py` (exhaustive, 65,024 pairs) | Mac | ALL PASS |
| `tests/test_refine_logic.py` | 512S1, torch CPU | 216/216 |
| `tools/compile_check.py` (Triton → SASS sm_121, production specialization, no GPU) | spark-eight, nice 19, lock absent | ALL PASS |
| `tools/interp_test.py` (Triton interpreter) | spark-eight, CPU | 143/143 |
| `device_test.py --smoke` (CPU + interpreter run of the harness, now incl. anchor + dispatch; not a gate) | spark-eight | ALL PASS (row dispatch skips M=17: interpreter numerics invalid) |

Not run:
- `device_test.py` on a GPU. It needs a stopped node in a device window.
- Every serving arm.

No lever here has passed a device or serving test. Every number below is a trace- or model-based estimate. Round-4
estimates were off by 3–4× in both directions, so only a C-T-C-T pair decides.

## 1. Evidence from the t2 traces

Production traces: `512S1:~/climb-traces/t2/prod`, 60 steps each.
- Scripts: `scripts/headwait.py`, `scripts/prestage.py`, `scripts/rankratio.py`, `scripts/e1tail.py`.
- Outputs: `out/`.

Head chain on rank 0 and rank 4, means per step in µs (`out/headwait_t2.txt`):

| workload | screen r0 / r4 | refine r0 / r4 (p90 r4) | (max,id) gather r0 / r4 | screen start → gather end |
|---|---|---|---|---|
| code | 463.9 / 478.3 | 25.2 / 30.7 (44.5) | 53.6 / 31.6 | 556.7 |
| prose | 461.9 / 477.7 | 23.3 / 25.4 (30.7) | 41.3 / 21.8 | 541.2 |
| maths | 466.2 / 480.0 | 22.1 / **76.6** (144.2, max >300) | **128.0** / 58.2 | 630.1 |

- **Refine imbalance.** The maths refine on rank 4 is spread from 25 to >300 µs (histogram in `out/`). Rank 4 still
  waits 58 µs at the gather, so some untraced rank is slower still.
- **What the gather waits on.** The gather floor is about 6–10 µs, so rank 0's wait is about 119 µs in maths, 44 µs
  in code and 32 µs in prose. Prose (flat refine) shows the residual screen and launch skew, about 30 µs. Maths' excess
  over prose, about 87 µs/step, is the max-over-8-ranks refine. Code's excess is about 12 µs.
- **Why the refine is slow.** One live refine CTA computes a needed 16-row tile serially over K=5120: 40 dependent
  steps of 4 KB each, latency-bound at about 15–20 µs. Hundreds of needed tiles then turn DRAM-bound: 144 µs ≈ 26 MB ≈
  160 tiles × 164 KB.
- **Rows vs tiles.** The live predicate is per tile, so every tile with one candidate row reads all 16 rows. On the
  synthetic flat case in the CPU test, 22 of 63 tiles hold only 28 candidate rows (12.6× fewer bytes in `row` mode).
  The production rows/tiles ratio is unknown: no log has it, since `tiles computed` is logged every 2000 calls on rank
  0 only and is absent from the current logs. The r6 stats line provides it on every rank.
- **Screen.** 88.0 MB (MXINT8 + exponents + rt + nv) in 462–480 µs is about 185–190 GB/s.
  - SASS (`results/compile.log`): the live K loop is 540 instructions per K step per thread (8.4 per weight element:
    64 I2F, 83 LDS, 64 STS). At M = 7 that gives an issue-only floor of 47–60 µs (r6: 39–50 µs; 2.4–1.9 GHz) against a DRAM floor of 383–440 µs. The screen
    is memory-bound, not issue-bound, so instruction savings alone buy nothing.
  - The DQ kernel cuts shared memory per CTA from 40.0 to 24.8 KB (the live kernel round-trips the dequantized tile
    through smem). Residency goes from 2 to 3 CTAs/SM (smem-limited, 100 KB/SM) and 540 → 447 instructions per step.
    Any gain comes from more memory-level parallelism, and only the device sweep can tell.
  - Rank 4's screen is +14–16 µs/step for fixed work (r4/r0 = 1.031). Other kernels range from 0.97 to 1.06
    (`out/rankratio_t2_code.txt`), with no clean SM-clock or DRAM pattern. This is a node property that no exact
    kernel change removes. A faster screen helps every rank equally.
- **Pre-stage window.** From the end of the last FFN GEMM (fc2) to the screen start is 75–87 µs on every workload and
  rank, with **zero** L2-prefetch kernels (`out/prestage_t2.txt`, `out/e1tail_t2_maths_r0_s30.txt`).
  - The window holds the last MoE AR (28–38 µs), mHC/head reduce/rms_norm, an 11 µs host gap at the target-graph end,
    and about 16 µs of E1 glue.
  - Nothing stages the 88 MB table. The head weights are missing in L2 by construction, which meets the brief's
    condition for lever 3.

## 2. Mechanism

**HEADREF.** `certified_logits(full=False)` runs these kernels in order:
1. `_r6_xnorm_kernel`: the live body, plus program 0 zeroes the list counter.
2. The screen: live, or r6 with `SCREEN=1`.
3. `after_screen`.
4. `_r6_cand_kernel`: 64 CTAs of 256 rows at BMX 16.
   - Evaluates the live predicate per row (`row`) or per live 16-row tile (`tile`).
   - Writes −inf to the rows it does not select.
   - Appends the selected rows to a compact int32 list: one atomic per CTA plus an in-CTA prefix sum.
5. `_r6_refine_kernel`: a fixed grid of 4 × SMs = 192 persistent CTAs.
   - Each reads the list length and walks chunks of the list.
   - `spread=1` sizes chunks to ceil(n/192) rows (1..16), so the candidates are spread over up to 192 CTAs.
   - Each listed row is computed with the live refine's exact statement sequence.

Two effects:
- **Byte reduction (`row`).** Only candidate rows are read: 10 KB per row instead of 164 KB per tile. This is what
  shortens the maths outliers.
- **Light-case cost.** The light case becomes cand (about 3–4 µs) + one more launch gap (about 2.8 µs: the t2 gap
  between consecutive head kernels) + one persistent-CTA K chain that is no shorter than today's (same 40-step K
  chain, about 20 µs), against today's 1010-CTA kernel at about 22 µs. That is a loss of about 3–6 µs per step.
  - The chain itself can shrink only through BK/stages: BK 256/512, 3–4 stages compile without spills (§5).
  - Those configs are admitted only if bit-identical on the device.
  - Splitting K across CTAs is not allowed: it changes the reduction.

**SCREENDQ.** The live `_screen_kernel` with wd built from exact integer steps:
- the 0x4B400000 magic for f32(q);
- one exponent byte per 32 weights;
- DQ=2 takes the top 16 bits of the exact product.

The dot, bound, HI store and THR atomic are the live text. A launch-shape sweep (BN 32/64/128 × BK 128/256 × warps
4/8 × stages 2/3/4) runs in the device test. A config can be used only if it prints as ELIGIBLE (bit-identical at
every swept M and case).

**Self-check (default on).** The first eager call at each new M runs the live pass on the same input and compares:
HI/THR bits, tile output bits, `row` == where(predicate, live, −inf) plus the list length, and argmax. A failure logs
ERROR, serves the live result and turns r6 off for that process. It is rank-local, which is safe: no collective is
added or removed, and each shard is argmax-safe on its own.

## 3. Flag off is identical; flag on is exact

- **Flag off.** The 7 inserted lines evaluate two `os.environ.get` calls at import. Per call, `certified_logits`
  evaluates one False test. No import, kernel, launch, buffer or stream changes. The patched file minus the tagged
  lines equals live byte for byte, and the AST diff is exactly 3 statements (`tests/test_install.py`).
- **Flag on.** The op-by-op map is in `EXACTNESS.md`:
  - `tile` output == live, bit for bit.
  - `row` output == where(row predicate, live, −inf), each computed logit bit-identical.
  - The argmax is identical by the CERT certificate, ties included: every row tied with the max has hi ≥ T.
  - Screen HI/THR are bit-identical: exhaustive dequant proof, then the same HMMA.16816.F32.BF16 sequence.
- **Races.** One stream, launches in order, disjoint and complete write sets, per-call buffers. Graph-safe: fixed
  grids, no host read in `_run` (§5 of EXACTNESS.md).

## 4. Estimates, re-based on t2

1 % G1 ≈ 0.21 ms/step. ΔG1/G1 = Σ_i tok/s_i · d_i/T_i / Σ_i tok/s_i with T = 22.2 / 19.4 / 22.0 ms (code / prose /
maths).

- **HEADREF** (re-based in the round-6 review; the earlier 0.03 / 0.17 / 0.30 % was optimistic).

  | d (ms/step) | maths | code | prose | ΔG1 | G1-weighted ms |
  |---|---|---|---|---|---|
  | central | 0.045 | −0.002 | −0.004 | 0.08 % | 0.017 |
  | low | −0.004 | −0.006 | −0.006 | −0.02 % | −0.005 |
  | high | 0.080 | 0.008 | −0.001 | 0.16 % | 0.035 |

  - Every step pays the light-case cost: cand (3–4 µs) + one more launch gap (about 2.8 µs in t2) in front of a refine
    chain no shorter than live's. That is −3 to −6 µs, unless the time part finds a faster ELIGIBLE BK/stages.
  - Maths: the 87 µs excess is the 8-rank max of the refine. Central assumes `row` recovers about 56 % of it (net
    45 µs). The production rows/tiles ratio is unknown and the slowest rank is untraced, so this is a guess that the
    stats line settles. High (net 80 µs, about 92 %) was the old central.
  - Low: production candidates fill whole tiles, so `row` ≈ `tile`, and only the overhead remains.
  - Code: 12 µs excess; central recovers about 3 µs, which does not cover the overhead.
  - The reviewer's independent central was about 0.10 %. Both are well below the +0.3 % bar, so HEADREF alone will
    not promote. The combined arm reaches the bar only if SCREENDQ lands near its high end.
- **SCREENDQ.** Fixed work in every workload.
  - Central: 20 µs, from 2 → 3 CTAs/SM giving about 4 % more bandwidth.
  - High: 80 µs, if a sweep config reaches about 225 GB/s.
  - Low: −4 µs, if the extra residency only adds DRAM contention.
  - In ΔG1: −0.02 / 0.09 / 0.37 %.
- **CERTPRE** (not built).
  - About 8 MB staged in the 50–60 µs in-graph part of the window at about 150 GB/s.
  - Screen saving 8 MB / 190 GB/s ≈ 42 µs, minus about 4 µs of AR stretch and contention. Central about 30 µs,
    0.14 %.
  - The micro-bench (`--part l2`) measures the saving per MB.
- **Built set** (HEADREF + SCREENDQ): −0.04 / 0.17 / 0.53 % (−0.009 / 0.037 / 0.115 ms/step; G1 ≈ 130.2 → 130.4
  central).

## 5. Device test (the deliverable that decides)

    # stopped node, device window, lock held by the window tooling
    source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
    cd ~/climb-scratch/r6-head            # rsync of this kit folder
    python device_test.py --part all                          # synthetic head shard, ~25-35 min
    python device_test.py --part all --weights real --rank 4  # rank 4's real head.weight rows (165 MB read)

**Start-up assertions.** The run refuses with exit 3 unless all of these hold:
- the venv and CUDA_HOME are the mandatory ones, and ptxas comes from /usr/local/cuda/bin;
- the GPU is a GB10 (sm_121), and Triton is 3.7.1;
- no other compute process is on the GPU;
- the module under test is the serving `dsv41_r4_small` from the venv (3364d8c8 live, or c77844fb installed), with the
  Triton JIT `_xnorm/_screen/_refine` kernels and the launcher's tile defaults.

**Gates** (exit 0 only if all pass):
- **anchor:** with the r6 flags off, the serving `r4.certified_logits` equals the `r6._run(..., False, "0")` reference
  bit for bit (out, HI, THR, BAD) at every M in 2..96 × 4 cases. Every other part compares against that reference.
  Must-fail: the same compare against `full=True` must differ at every M.
- **bitwise:** every M in 2..96 × {flat, peaked, cluster, hostile}, for screen DQ1/DQ2 and refine tile/row.
- **Config sweeps, one M per served bucket:** M ∈ {2, 7, 24, 48, 96}, i.e. BM/BMX 16, 16, 32, 64, 128. That is every
  bucket up to `SWEPT_MAX_B` = 128, including BM 64, where the screen's `num_warps` becomes `max(w, 8)`.
  - Screen: BN × BK × warps × stages, with DQ2.
  - Refine: BK × stages (these act at BMX 16 only) × grid × spread (these act at every BMX), in tile and row mode,
    with warps 0.
  - A config is ELIGIBLE only if it is bit-identical at every swept M and case. Each eligible config prints as a
    ready-to-paste `ELIGIBLE-ENV` line. The default configs must be eligible.
  - The module enforces the same coverage at import and at launch, and `tests/test_cfg_guard.py` checks this:
    - a non-zero `SCREEN_CFG` without `SCREEN_DQ=2` is a ValueError;
    - non-zero refine warps is a ValueError;
    - above BM/BMX 128 the live screen shape, grid 0 and spread 1 serve.
- **ties:** duplicated rows across tiles, in one tile, across the cand-CTA boundary, across the sf 4096 block, at the
  shard end and far apart, plus all rows identical. The r6 chain plus the live `_sf_local_argmax_kernel` must give
  the stock leftmost argmax at every M.
- **graph:** a captured r6 pass replayed on 3 inputs must equal eager.
- **dispatch:** the installed path on the GPU. One child process per mode (row, tile) runs with the r6 flags set and
  imports the patched `dsv41_r4_small`:
  - with `--source installed`, the venv's file;
  - otherwise, the live file patched in a temp dir by `install.py`'s own edit, sha-checked against c77844fb.

  At M ∈ {2, 7, 24, 33, 48, 96} × 4 cases:
  - `r4.certified_logits` must dispatch to r6 and match live (tile) or where(pred, live, −inf) (row), bitwise;
  - the `_sf_local_argmax_kernel` pairs must match live;
  - the self-check must pass and log at the first call per M;
  - `after_screen` must run once, debug must be filled, and `full=True` must stay live;
  - the stats counters and the stats log line must appear.

  Must-fail mutants: a flipped r6 output bit and a flipped HI bit. Each must be caught by the serving self-check at
  every M: the live result is served, r6 turns off, and one FAILED line is logged.
- **time:** the times never gate, but the forced-set bitwise checks it runs first (tile == live, row ==
  where(sel, live, −inf)) do gate. That is stricter than INFO, on purpose.
- **mutants:** each must be caught at every M in 2..96:
  - `kreverse` and `ksplit`: reduction-order mutants, compared over all V×M logits by forcing every row;
  - `strict` (`>` instead of `>=`);
  - `dropfirst`;
  - `egroup` (wrong exponent group);
  - `tile8`.

**INFO** (never gates):
- per-logit mismatches against F.linear (cuBLAS), live vs r6 (r6 must add none);
- median timings, 200 runs warm and cold-L2: screen per config, refine at forced candidate sets (n 1..16160,
  scattered or clustered), and the full chain;
- the L2 pre-stage micro-bench;
- `results/device_test.json`.

`--smoke` runs the harness on CPU with the Triton interpreter (V 1000, K 512). It passed, but it is a code-path check
only: Triton 3.7.1's interpreter evaluates a bf16 `tl.dot` on raw bit patterns (the live kernels return ~5e11 for
|logit| < 2), so it says nothing numeric.

**Choose the window env from the output.** Copy `ELIGIBLE-ENV` lines; never type a config by hand.
- Screen: the fastest cold-L2 `ELIGIBLE-ENV screen` line. A non-default shape always comes with
  `DSV41_R6_HEAD_SCREEN_DQ=2`. The live shape is `SCREEN_CFG=0,0,0,0` with DQ 1 or 2.
- Refine: the fastest `ELIGIBLE-ENV refine` line (`BK,0,stages,grid,spread`), judged on the forced-set and chain
  timings.

## 6. Serving arms (lead)

C-T-C-T. On every arm: exact.py ec2f4437, G1 digest 2f7fdccbd388225a, 9 long prompts.

| arm | env |
|---|---|
| hcheck (first) | `DSV41_R4_CERT_HEAD=check DSV41_R6_HEAD_REFINE=row DSV41_R6_HEAD_SCREEN=1 DSV41_R6_HEAD_SELFCHECK=1 DSV41_R4_CERT_HEAD_LOG_EVERY=500` |
| hrow | `DSV41_R6_HEAD_REFINE=row DSV41_R6_HEAD_SELFCHECK=0` |
| htile | `DSV41_R6_HEAD_REFINE=tile DSV41_R6_HEAD_SELFCHECK=0` |
| hscr | `DSV41_R6_HEAD_SCREEN=1 DSV41_R6_HEAD_SELFCHECK=0` + an `ELIGIBLE-ENV screen` line |
| hall | hrow + the hscr screen line + an `ELIGIBLE-ENV refine` line |

- **Self-check and speed arms.** The self-check costs one extra head pass plus host syncs at the first call of each
  new M, on every rank, for up to 95 M values. That can bias short C-T-C-T segments.
  - Run hcheck first, with `SELFCHECK=1`.
  - Use `SELFCHECK=0` on the speed arms only after `device_test.py` exits 0 on both weight sets and hcheck is clean:
    8/8 ranks log `self-check passed`, none logs `FAILED`, and the check counters are 0.
  - Otherwise keep `SELFCHECK=1` and warm every served M before the measured segment.

- hcheck is a det/validation arm, not a speed arm. It is slower because check mode runs the stock head too. It counts
  argmax, value and computed-logit mismatches on device.
- **Boot and log lines to count, 8/8:**
  - `[r6-head] DSV41_R6_HEAD_REFINE=` (config line);
  - `[r6-head] self-check passed at M=`, which must never appear as `self-check FAILED`;
  - `[r6-head] r6 refine: ... rows computed (x/call), live-equivalent tiles (y/call)`. Its x vs 16·y per rank is the
    production rows-vs-tiles answer.

**Reading the effect in nsys:**
- `_refine_kernel` (grid 1010) is replaced by `_r6_cand_kernel` (grid 64 at M≤16) + `_r6_refine_kernel` (grid 192).
- `_screen_kernel` is replaced by `_r6_screen_kernel`.
- `_xnorm_kernel` is replaced by `_r6_xnorm_kernel`.
- The lever shows as the drop in the (max,id) gather (`ar_kernel_pl` after `_sf_local_argmax_kernel`) on rank 0 in
  maths: `scripts/headwait.py <dir> 3` gives `g` and "screen start → gather end".

## 7. Install and composition

    python3 install.py install | uninstall | status | dry     (R6H_VLLM=<vllm dir> for copies)

- **Install rules.**
  - The base must be exactly the sha in KIT.json, or the installer refuses with nothing written.
  - Every inserted line carries the `# dsv41-r6-head` tag, and the base is backed up to `.bak-r6-head`.
  - Install is idempotent. Uninstall restores the base byte-exact and refuses a file that is not exactly our patch.
  - The new module is copied when absent. A known earlier kit version (KIT.json `prior_sha256`) is upgraded. Any other
    bytes make both install and uninstall refuse, with nothing written (`status` shows FOREIGN).
  - Install 8/8. `install.py` is per host: it has no CLUSTER_LOCK check and no 8/8 wrapper (the r1/r2 pattern). Run it
    on each Spark inside a window that holds the lock (`climb-window.sh` KIT_INSTALL or a device-window MAC_PLAN step),
    on all 8 before a relaunch.
- **No conflicts with r5 or the window kits.**
  - No r5 kit (glue, ffnfront, prio, harness) touches `dsv41_r4_small.py`.
  - The r5-composed `model.py` call lines into `dsv41_r4_small` and the `dsv41_l2pf_r1._wrap_cert` lines are identical
    to live.
  - Installing on the r5-composed tree gives the same patched bytes, and the uninstall is byte-exact
    (`tests/test_install.py`). So either live state, with or without r5, takes this kit, in any order relative to
    `climb-r5/install-all.sh`.
- **Wrappers and the manifest.**
  - `dsv41_l2pf_r1._wrap_cert` (R1_DRAFT, off) and `dsv41_drafter_sched` (MAINPROJ_HOIST, off) wrap or call
    `certified_logits` by name. The dispatch sits inside its body, so they see the r6 path, with `after_screen` at the
    same point.
  - `dsv41_r4_small.py` is not in the supervisor manifest.
- **climb-r1 tooling: uninstall r6-head before r1-drafter.** While r6-head is installed:
  - `climb-r1/install-all.sh status` over the full kit set reports FOREIGN on `dsv41_r4_small.py`, because c77844fb
    is neither c46e66bd nor 3364d8c8. Subset runs such as `R1ALL_KITS=r1-pdl` are unaffected.
  - A direct `climb-r1/r1-drafter/install.py uninstall` restores dspark.py and speculator.py, then refuses on
    `dsv41_r4_small.py`, which leaves r1-drafter half removed.
  - So run `python3 install.py uninstall` here first, then the r1 tooling.

## 8. Lever 3 design (cert-head L2 pre-stage), for round 7

- **Where.** Put it in `dsv41_l2pf_r1` as a target lever, the same pattern as `R1_DRAFT`'s `DRAFT_HEAD_MB`, which
  already puts the first MB of the draft head into the drafter graph's last AR window.
  - At learn/capture of the target graph, append the first P MB of `model._dsv41_r4_cert.q` to the final layer's
    MoE-AR window. That CertState is built at the first eager `target_logits`, which runs before capture.
  - Pace it at about 150 GB/s on the pacer stream, joined like the base windows. It must end before the graph end,
    which leaves 50–60 µs of in-graph latency-bound time.
- **The screen needs no change.** CTA pid order reads q rows from 0 upwards, so the staged prefix is consumed in its
  first wave.
- **Exact by construction:** `cp.async.bulk.prefetch.L2` hints only.
- **Composition:** r5-ffnfront also patches `dsv41_l2pf_r1` (5af558fe → ffnfront base), so build it on the
  ffnfront-composed file.
- **Size P** from `device_test.py --part l2`: saving per MB and pre-read time per MB, against the L2 size the env part
  prints.

## 9. Risks

- **`row` gain depends on how production candidates are distributed.** If maths candidates on the heavy ranks fill
  whole tiles, `row` ≈ `tile`, and the gain is only the light-case delta (±3 µs). The stats line settles it in the
  first arm.
- **The certificate dependence of `row` is stronger than live's.** Live gets non-candidate tile-mates for free. `row`
  computes only certified candidates, so a certificate violation (a stock logit above its hi) could change the
  argmax under `row` where live might be rescued by a tile-mate.
  - The certificate is the same one live relies on: qualified by gpu_test_r4, with generous inflations.
  - hcheck counts it online.
  - Prefer `tile` if hcheck ever shows a mismatch.
- **The screen sweep may find nothing** (the screen is DRAM-bound at about 190 GB/s). Then SCREENDQ ≈ 0. The DQ kernel
  with the live shape is still exact and costs nothing.
- **New Triton specializations JIT at first use per BMX bucket** (eager, before serving; as live).
- **The self-check costs one extra head pass and several host syncs per new M** (about 0.5 ms each, up to 95 times per
  process and rank). See §6 for when speed arms use `DSV41_R6_HEAD_SELFCHECK=0`.

## 10. Where I see more

1. CERTPRE, sized by `--part l2`: 0.14 % central, and the head table is the only large stream with zero staging.
2. A smaller screen table:
   - MXINT8 is 88 MB. An MXINT4 screen (44 MB) with a looser bound would halve the 460 µs screen (about 1 % G1) but
     raise candidates. In `row` mode those cost about 10 KB each instead of 164 KB per tile, so this kit is what makes
     a coarser screen affordable.
   - It is exact by the same certificate argument. Building it needs a new table, a new bound proof and a device
     candidate census.
3. Rank 4's +15 µs screen (node property). The 8-rank census (r5 build list #2) should say whether another rank is
   slower, since the maths wait shows an untraced rank behind rank 4.

## Files

- `install.py`, `KIT.json`, `EXACTNESS.md`, `TEST-RESULTS.txt`
- `files/vllm/models/deepseek_v41/nvidia/dsv41_head_r6.py` (new module)
- `base/` (live copy 3364d8c8)
- `tests/`: `test_install.py`, `test_cfg_guard.py`, `test_dequant.py`, `test_refine_logic.py`, `fixtures/` (the
  earlier module 9bd17b03, for the installer's upgrade test)
- `tools/`: `compile_check.py`, `interp_test.py`, `run_spark8_checks.sh`
- `device_test.py`
- `scripts/` (t2 analysis), `out/` (t2 outputs), `results/` (logs, SASS/PTX), `live/` (read-only copies from
  spark-one)
