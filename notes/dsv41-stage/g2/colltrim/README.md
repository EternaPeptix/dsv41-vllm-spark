# g2 colltrim: collective trims (next-experiments #4), staged, NOT installed

Offline prep, 2026-10-26. No restart, no GPU job, no live edit. CPU builds and CPU tests ran on spark-four
under `nice -n 19` with `CUDA_VISIBLE_DEVICES` empty. Stage dir: `~/dsv41-stage/g2/colltrim/` on 8/8
(`SHA256SUMS` inside). Every flag defaults off. With the flags off, the installed files behave like the live ones:
- the l2-prefetch plan is byte-identical (CPU test: 32 env combinations);
- the dim-0 and all-reduce paths use the same kernels (SASS-identical, 11/11 kernels).

Exactness: (a) and (b) are exact by construction (data movement and cache hints only), so the serving
proof is the usual one: exact.py a8c3394af9d2 with identical pinned counters and an identical G1 output
digest per arm.

## Hashes and gates so far

| file | sha256[:8] | base (live, identical 8/8) |
|---|---|---|
| `site/.../cuda_communicator.py` | 87bf77e3 | 52fc8cf5 |
| `site/.../rdma_all_reduce.py` | 26a5d834 | 9056dcdf |
| `site/.../dsv41_l2_prefetch.py` | 63737f9c | 998af3d7 |
| `build/rdma_ar_ext.so` (from `native/rdma_ar.cu` 6173b362) | 0692daa5 | live k1 build 62daad6c, not replaced |
| `build/libdsv41_l2pf_g2.so` (from `native/l2pf_g2.cu` 0fa219e5) | 0451009d | live r4 lib a09f85e8, not replaced |

| gate | where | result |
|---|---|---|
| `tests/sass_check_g2.py` vs live `build-k1pl/rdma_ar_ext.so` | spark-four CPU | **PASS**: 11/11 live kernels instruction-identical; new kernels clean |
| `tests/test_plan_g2_cpu.py` | spark-four CPU | **PASS**: flags-off plan == live (32 combinations); re-plans conserve bytes; pacing split |
| `tests/test_cols_logic_cpu.py` | spark-four CPU | **PASS**: 24 gather calls; column path taken iff dim>0 and seg%16==0; result == `torch.cat`; one extension call, one audit, one before+after prefetch hook per gather; declined calls have no side effect |
| `tests/test_cols_selftest.py`, `tests/test_l2pf_g2_gpu.py` | one GPU, stopped window | not run (offline prep) |
| `bench/bench_g2ct.py` | 8 ranks, stopped window | not run (offline prep) |

## (a) In-place column all-gather: `DSV41_RDMA_AG_COLS=1`

**Today.** `cuda_communicator.all_gather(dim=-1)` on the rdma path works in two steps:
1. It gathers into a `[world, M, 224]` staging tensor.
2. It runs `movedim(0, dim).reshape(...)`, which is a `direct_copy` kernel. There are 40 of these per decode step for the qkv_a shard gathers, plus the logits / Engram / DSpark main_proj gathers.

**New.** The receiving kernel writes rank r's `[outer, seg]` slice straight into columns `[r*seg, (r+1)*seg)`
of the final contiguous tensor. This is knapcio's SPLIT_COMPACT_GATHER idea, and RoCEnante's `_allgather_cute` writes the same layout.
- **Same everywhere else.** The wire protocol, payload, sequence numbers, op order and L2-prefetch hook order are all unchanged.
- **Output.** The result tensor is bitwise identical to today's, just without the extra kernel and allocation.
- **When it applies.** Only when `dim > 0` and `seg = elem_size * prod(shape[dim:])` is a multiple of 16 B. Otherwise, and for dim 0, today's path is used.

**Native: `native/rdma_ar.cu`.** This is the k1 parallel-load source 1abfbc63 with the changes below; diff in `diffs/rdma_ar.cu.diff`.
- New mode `kGatherCols = 3` in both kernels (legacy and parallel-load).
- `Params.seg` is appended at the end of the struct.
- New `RdmaAR::all_gather_cols(out, x, seg_bytes)` and `SelfTest::all_gather_cols_into`.
- New module attribute `G2_VARIANT`.
- **Build.** `build/rdma_ar_ext.so` (ninja build, spark-four). Installed to `~/rdma_ar_dsv41/build-g2ct/` (new dir). The live `build/` and `build-k1pl/` are untouched.
- **SASS gate** (`tests/sass_check_g2.py`, `results/sass_check.txt`): all 11 kernels of the live k1 build are instruction-identical in the g2 build (all-reduce one-shot / two-shot bf16 + fp32, dim-0 gather, legacy and parallel-load, pingpong). New kernels:
  - `ar_kernel_pl<3,false,8>`: 96 regs, no spill, all 8 peer loads in flight before the stores.
  - `ar_kernel<3,false>`: 40 regs, no spill.

**Python.**
- `rdma_all_reduce.py` (9056dcdf -> staged):
  - New `_G2_EXT` path, chosen when `DSV41_RDMA_AG_COLS=1`. `DSV41_RDMA_PARLOAD` still selects parallel-load or legacy kernels inside it.
  - New `RdmaAllReduce.all_gather_cols()`. It returns None before any side effect when the shape does not qualify.
  - New post-collective prefetch hook, used by (b).
  - The fingerprint covers the new env var and the selected .so automatically.
- `cuda_communicator.py` (52fc8cf5 -> staged): a 6-line branch in `all_gather`.

**Expected gain.** About 2.5-3.5 µs per qkv gather (copy kernel + launch slot) × 40 = **-0.10..-0.15 ms per step**, plus the logits copy on CERT-armed steps (`[M,129280]` bf16, ~5-15 µs). That is about +0.4-0.6% at C1 and ~+0.3% G4: borderline for the 0.5% bar alone, so stack it with (b).

## (b) L2-prefetch pacing and window re-plan (`dsv41_l2_prefetch.py` 998af3d7 -> staged)

**Problem** (k1 nsys, difference-in-differences; `kernel-ideas-2.md` §5). The burst `l2pf_kernel` fires a whole window, up to 6 MB, at LPDDR5X peak. The collective's latency chain queues behind it: send-slot staging, NIC DMA, sys-scope flag polls, peer loads. Measured stretch per call:

| collective | window size | stretch |
|---|---|---|
| post-MoE AR | ~1.6 MB | +10 µs |
| post-attention AR | 6 MB (now mhc fn 1.88 MB + router gate 3.75 MB) | +18 µs |
| qkv_a gather | q_b 5.16 MB + wo_b | ~+8 µs (30.4 µs vs ≤22 µs expected) |

Total: about 1.1-1.5 ms per step at 1 user. The bytes are useful, though: q_b 35 → 12 µs, mhc 16 → 10 µs, and the router-gate backfill (PRUNE_SHARED) was worth -2% C1. So the arms move or pace the bytes rather than drop them.

**Env arms.** All need `DSV41_L2_PREFETCH=1` (live default). With all of them off, the module is behaviourally 998af3d7.

| env | arm |
|---|---|
| `DSV41_L2_PREFETCH_REPLAN=attn` | Each post-attention AR window (mhc fn + router gate, read right after that AR) is forked at the last wq_b of the attention block instead, on the WOA-style stream joined at forward end. The bytes then stream during rope / KV insert / indexer / sparse MLA, and the AR runs with no prefetch. It registers the wq_b post-hooks itself; with WOA=1, the wo_a bytes come first in that fork. |
| `DSV41_L2_PREFETCH_REPLAN=qbprev` | The q_b (wq_b) bytes of each qkv_a gather window move to the end of the previous collective's window. |
| `DSV41_L2_PREFETCH_REPLAN=qbpost` | The same bytes are forked right **after** the previous collective's kernel (new `after_collective` hook), so they overlap mhc / qkv_a GEMM instead of a collective. |
| `DSV41_L2_PREFETCH_REPLAN_MB` (16) | Cap on the bytes a re-plan adds to one fork; any remainder stays where it was. Windows that lose bytes are never backfilled. |
| `DSV41_L2_PREFETCH_PACE_GBPS=R` | Paced issue at R GB/s: one warp, `%globaltimer` deadlines, `build/libdsv41_l2pf_g2.so::dsv41_l2pf_paced`. It runs on a third side stream joined only at forward end, so a pacer never blocks the main stream. |
| `_PACE_KINDS` (ar,ag,post,attn) | Which forks are paced. |
| `_PACE_HEAD_KB` (0) | The first N KB of a paced fork are still issued as a burst, for the bytes the very next kernel reads. |
| `_PACE_DELAY_US` (0) | Start delay for the paced part. |
| `_PACE_MAX_US` (150) | A pacer stops issuing after this time. |
| `_PACE_CHUNK` (16384) | Bytes per paced prefetch instruction. |
| `_PACE_LIB` | Path to the pacing library. If it is missing, paced parts fall back to burst (logged). |
| existing diagnostics | `DSV41_L2_PREFETCH_AG=0` (no gather windows), `DSV41_L2_PREFETCH_MB=3\|4`. |

**Natives.** `native/l2pf_g2.cu` contains the r4 `l2pf_kernel` / `spin_kernel` verbatim plus the paced issuer. Built to `build/libdsv41_l2pf_g2.so` and installed to `~/rdma_ar_dsv41/build-g2ct/`. The live `~/dsv41-stage/r4/l2-prefetch/libdsv41_l2pf.so` stays the burst library.

**CPU planner test** (`tests/test_plan_g2_cpu.py`, PASS):
- Flags off: the staged `make_plan` equals the live module's, including the WOA tables, for 32 combinations of PRUNE_SHARED / PRUNE_ROUTER / budget / WOA / SKIP_N.
- Every re-plan conserves bytes: the prefetched byte ranges are identical to the base plan's. It moves only post-attention AR windows (attn) or q_b of gather windows (qb*), and leaves every other window byte-identical.
- The attn forks land at the last wq_b of each block.
- Pacing splits forks into a burst head ≤ HEAD and a paced tail, and only for PACE_KINDS.

**Expected gain** (estimates from the k1 stretch): `attn` removes up to +18 µs × 40 ≈ 0.7 ms per step if attention absorbs the bytes without its own stretch. `qbpost` / `qbprev` remove up to ~0.3 ms. Pacing is the fallback if the stretch follows burst depth rather than average rate. Range: **-0.3..-1.0 ms at 1 user (+1-3.5%)**, smaller at 16 users. The 8-rank bench (phase W) measures the per-layer trade-off before any serving arm.

## (c) RoCEnante vs rdma_ar: microbench plan only (needs your OK to run third-party code)

**Exactness check (read-only, `ext/knap-latest/.../b12x_next/comm/roce`).**
- RoCEnante's one-shot all-reduce sums in fixed rank order 0..W-1, in fp32, with one `cvt.rn.bf16.f32` at the store. That is our rule.
- **One difference: the accumulator start.** RoCEnante initialises with the rank-0 value; we start from +0.0. The results differ only when every rank's element is -0.0: they give -0.0, we give +0.0.
- Its all-gather writes column blocks, the same layout as (a).
- A port would keep our consumer code (zero init), so it stays bitwise. Two-shot vs one-shot is also exact under the same rule.

**Plan** (stopped window, ~20 min; nothing staged):

**Software.** RoCEnante is not standalone. It needs:
- knapcio's b12x runtime (`runtime/b12x_next`, Apache-2.0; local copy in scratchpad `ext/knap-latest/repo/runtime/`);
- the CuTe DSL (`nvidia-cutlass-dsl`), which JIT-compiles its kernels;
- `_roce_proxy.c`, built by `_proxy.load`;
- b12x's preparation flow: `RoceOneshotAllReduce(exchange_group=<gloo group>, device=0, max_size=2 MiB, hca_names=..., gid_index=5)`, then `_preparation.plan(RoceQuery(...), runtime=rt)` → publish → `rt.all_reduce(x, plan=p)` / `rt.all_gather(x, dim=-1, plan=p)`.

Install it into a separate venv on 8/8 (e.g. `~/venvs/roce-bench` with the same torch 2.13/cu130), never into vllm032.

**Measurements**, in one process per Spark, ABBA ×3, graphs of 40 ops:
- **Sizes:** all-reduce rows 1 / 6 / 24 / 48 × 5120 bf16 (10-480 KB), plus the (6,224) qkv gather (2.7 KB).
- **Arms:** RoCEnante with 2 HCAs (ours: `rocep1s0f0,roceP2p1s0f1`), RoCEnante with all 4, and our rdma_ar (one-shot below 32 KB, two-shot above, as served).
- **Background:** each runs with and without a background 6 MB burst prefetch (`bench_g2ct.py` phase B setup).
- **Bitwise check:** every output against our fixed-order fp32 reference (an NCCL gather of the inputs, summed in rank order); `-0.0` inputs are included to show the init difference.

**Port gate:** ≥10 µs saved per call at 6 and 48 rows (about -1.25 ms per step per 10 µs), with our accumulate code. A port means moving their striping and one-shot flag scheme into `rdma_ar.cu`, not running b12x in vLLM.

## Stopped-window procedure (service down ~25 min, then the serving A/B)

Run from the Mac, package dir = this directory (Mac copy:
`/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad/ct/pkg`).

1. Stop the service the usual way, recording the rank logs.
2. `bash ops/run-tests.sh gpu spark-two`, ~4 min, one GPU. It runs:
   - `test_cols_selftest.py`: 19 shapes × 3 ranks × 2 kernel sets bitwise vs the movedim path and the reference; guard bytes; seg = nbytes == dim-0; a 40-gather CUDA graph × 30 replays; bad seg refused; local timing of cols vs copy.
   - `test_l2pf_g2_gpu.py`: the paced issue timeline from the kernel's own stats (within 10% + 3 µs), data unchanged, L2 effect, then a 12-layer toy forward through 7 arms under CUDA graphs. Gates: graph == eager, all arms bitwise equal, side streams joined, planned fork counts issued, flags-off plan == live plan on recorded events.
   - **Gate:** `RESULT PASS` in both.
3. `bash ops/run-bench8.sh g2ct-w1`, 8 ranks, ~8-10 min, staged files only. It runs:
   - **A:** cols vs today vs PyNccl bitwise (eager, both kernel sets, fallbacks); 40-gather graph × 30 replays; timing of a 40 × (AR + qkv gather) step-like graph, today vs cols, M 1-8, plus logits gathers.
   - **B:** AR / AG latency and in-kernel stage / net / tail under none / burst 1.6 / burst 6 MB / paced 50-200 GB/s / delay / head.
   - **W:** per-layer window emulation:
     - W1 post-attention: none / live / pacehead / attn / attnpace.
     - W2 qkv/q_b: none / live / qbprev / qbpost / qbpostpace / ag0.
   - **Gate:** `DONE failures per rank: [0, 0, 0, 0, 0, 0, 0, 0]`.
   - **Choosing the B arms:**
     - Take the W1 / W2 modes that beat `live` by ≥2 µs per layer (≥0.08 ms per step) at M=1 and M=6.
     - If B shows paced(R) cuts in-kernel AR time by ≥5 µs vs burst6 at equal bytes, also try a paced arm at the best R.
     - If no W mode beats live, do not serve any (b) arm; report and stop.
4. `bash ops/g2ct-all.sh install`: preflight on 8/8 (service stopped, base shas, stage sums), then install, then an identical-digest check. Backups are `<file>.bak-g2-colltrim`. Nothing changes behaviour until a flag is set.
5. **Serving A/B.** Supervisor relaunches; pinned AV `DSV41_AV_COST_JSON=/home/jeweled/dsv41-stage/pdab/av-pin-pdfinal.json`; clock 2300 (supervisor default); telemetry OFF; exact.py first on every instance, then matched C1 (label pdab) ×5, G1 `goal_single.py --repeats 3`, G4 `goal_conc4.py`.
   - **Order:** C1, A1, C2, A2, then B arms ABAB against the best of C/A.
   - **C:** installed, flags off. Must give a8c3394af9d2, pinned counters 905/471/2355/1373, G1 digest 9a18f802d930f6e7, and the same C1/G1 as the live config within noise.
   - **A:** `DSV41_RDMA_AG_COLS=1`. Its log must show `rdma_ar g2 column all-gather: ON (variant colltrim-cols-v1)`.
   - **B candidates** (from step 3): `DSV41_L2_PREFETCH_REPLAN=attn`; `DSV41_L2_PREFETCH_REPLAN=attn,qbpost`; `DSV41_L2_PREFETCH_REPLAN=attn DSV41_L2_PREFETCH_PACE_GBPS=80 DSV41_L2_PREFETCH_PACE_KINDS=attn`.
   - **Pacing-only arm**, if B says so: `DSV41_L2_PREFETCH_PACE_GBPS=<R> DSV41_L2_PREFETCH_PACE_HEAD_KB=2048 DSV41_L2_PREFETCH_PACE_KINDS=ar`.
   - Each B instance logs `DSV41_L2_PREFETCH g2 plan tokens=...` and `g2 captured tokens=... forks {...} (planned {...})`; the two must match.
   - **Promotion rule (this window):** ≥0.5% on the G4 prose-session MIN or the G1 mean in every pinned pair (≥2 pairs), nothing worse beyond noise. Stack the winners and re-check the stack. GSM8K-50 ≥ 49/50 and longctx PASS before final promotion.
   - **Launcher** (`~/dsv41-tp8.sh` 395bf5b2 on 8/8, backup `.bak-g2-colltrim`): add `export DSV41_RDMA_AG_COLS=${DSV41_RDMA_AG_COLS:-1}` and/or the chosen `DSV41_L2_PREFETCH_*` defaults after `DSV41_MOE_M32_G1`.
6. Leave the server up on the best config. Write the README section.

## Rollback

- **One launch:** supervisor argument `DSV41_RDMA_AG_COLS=0` and/or `DSV41_L2_PREFETCH_REPLAN= DSV41_L2_PREFETCH_PACE_GBPS=0`.
- **Files** (service stopped): `bash ops/g2ct-all.sh uninstall` restores the three `.bak-g2-colltrim` backups (52fc8cf5 / 9056dcdf / 998af3d7). `--purge` also removes `~/rdma_ar_dsv41/build-g2ct/`.
- **Launcher:** `cp -p ~/dsv41-tp8.sh.bak-g2-colltrim ~/dsv41-tp8.sh` on 8/8.

## Risks and notes

**Pacer occupancy.** A pacer holds one warp on one SM for up to `PACE_MAX_US`. A grid of exactly 48 one-CTA-per-SM, full-register-file CTAs would wait for that SM. The bursts (8 CTAs × 64 threads for a few µs) and WOA already have this property briefly. Serving A/B is the judge, and `PACE_KINDS=attn` keeps pacing out of the MoE.

**`attn` assumptions.**
- mhc fn and router gate must stay L2-resident through the rest of attention, wo_a / wo_b and the AR: ~5 MB of traffic into a 24 MiB L2.
- Attention kernels may see some of the stretch the AR loses. W1 measures this only with a spin stand-in, so serving decides.

**qbprev on Engram layers.** In layers 1 and 14, the previous collective before the qkv gather is the Engram all-gather, so `qbprev` grows that window instead. `qbpost` forks after it.

**Interactions.**
- stepovh WOA (`DSV41_L2_PREFETCH_WOA`, installed inert in 998af3d7) is kept and composes with `attn`.
- The staged module is based on 998af3d7. Any other package that edits `dsv41_l2_prefetch.py`, `rdma_all_reduce.py` or `cuda_communicator.py` makes the preflight refuse (base sha check).
- fc2c-on-m32all (another g2 package) edits different files.

**Supervisor manifest.** The supervisor's site manifest covers the three site files and `build-k1pl/rdma_ar_ext.so`. It does not cover `build-g2ct/`, but the rdma fingerprint hashes the selected .so at startup on every rank.

## Files

| path | what |
|---|---|
| `native/rdma_ar.cu`, `native/l2pf_g2.cu`, `native/build_g2.py` | sources + CPU build (nvcc + torch cpp_extension; refuses live dirs) |
| `build/rdma_ar_ext.so`, `build/libdsv41_l2pf_g2.so` | built on spark-four |
| `site/vllm/distributed/device_communicators/{cuda_communicator,rdma_all_reduce,dsv41_l2_prefetch}.py` | staged site files |
| `base/` | the live bases (52fc8cf5, 9056dcdf, 998af3d7, r4 l2pf.cu) |
| `diffs/` | unified diffs against the bases |
| `tests/test_plan_g2_cpu.py`, `tests/sass_check_g2.py` | CPU gates (PASS) |
| `tests/test_cols_selftest.py`, `tests/test_l2pf_g2_gpu.py` | single-GPU gates (stopped window) |
| `bench/bench_g2ct.py` | 8-rank bench (stopped window) |
| `ops/build.sh`, `ops/stage.sh`, `ops/g2ct-all.sh`, `ops/g2ct-site.sh`, `ops/run-tests.sh`, `ops/run-bench8.sh` | Mac drivers + per-node site script |
| `results/` | build log, SASS check, CPU test log |
