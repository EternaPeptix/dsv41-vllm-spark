# DeepSeek V4.1 Flash on 8 × DGX Spark (vLLM, TP8)

DeepSeek-V4.1-Flash-1017 (FP8 attention / FP4 experts) served across eight NVIDIA DGX Sparks
with a patched vLLM 0.32.0: tensor parallel 8, DSpark speculative decoding from the checkpoint's
MTP heads, 1M-token context, image input, and an OpenAI-compatible API.

**Weights and outputs.** Since 4 November 2026, production runs the official
`deepseek-ai/DeepSeek-V4.1-Flash-1017` weights with the OBLITERATUS refusal-direction ablation applied as exact
runtime projections (`DSV41_OBLIT`, see `deps/dsv41-oblit/README.md`). That migration was a deliberate change
to the outputs, and it set a new reference.

**Exactness.** Every *speed* change since then is bitwise exact against that reference. Every promotion from
8 November on was verified on the production launcher with three checks, and all three matched:

- the G1 outputs digest `2f7fdccbd388225a` (code / prose / maths output hashes `20e3a46d` / `03f7d6fe` / `adffacdf`,
  unchanged since the first measurement on the new weights);
- the `bench/exact.py` combined hash `ec2f4437f8e1ed09` (the same on the 5 November gate runs);
- the 9-prompt determinism check (`bench/det/`): synthetic prompts of 2.5K–128K tokens plus three recorded chat
  requests, each run 4 times cold, all equal to the reference.

The two 5 November promotions (FC1 tile deal, FC2 barrier) were checked against the same G1 digest. So no speed
lever trades against output quality. Changes made before 4 November were exact in the same sense
against the earlier reference `a8c3394af9d2`, which was measured on the previous checkpoint.

## Performance

Measured **12 November 2026, 11:29 UTC** on production launcher `4a04b175`. The API was isolated (exactly the 6
benchmark requests reached the server), with outputs digest `2f7fdccbd388225a` and exact.py `ec2f4437f8e1ed09`:

| Workload (1 request) | tok/s | DSpark acceptance length |
|---|---:|---:|
| Coding | **133.4** | 2.93 |
| Prose | **97.0** | 1.84 |
| Maths | **166.8** | 3.61 |
| **Mean (G1)** | **132.4** | |

- **G1:** greedy decode, 1,024 output tokens, time to first token excluded, 2 runs per prompt type. The repeat runs
  gave byte-identical outputs (`bench/goal/goal_single.py`; raw record
  `notes/results-20261112/sparkos67-20261112T110449Z-g1-1.jsonl`).
- **The same launcher, measured at other times:** 131.99 at its promotion (11 November, 22:47 UTC; code 133.05 /
  prose 96.62 / maths 166.31, digest + exact + 9-prompt check all equal), and 132.24 at each of three restores on 12 November
  (03:57, 05:06 and 05:28 UTC). The 132.4 run followed a relaunch after the last two nodes moved to the spark-os
  image. No code changed between these runs, so the spread is restart-to-restart variation.
- **Time to first token** for the G1 prompts (103–134 tokens) in the same run: **0.20–0.23 s**. The harness
  sends no cache salt, so the repeat run of each prompt can hit the prefix cache.

### How we got here: 114.8 → 132.4 tok/s

| Date | Launcher | Change | Measured gain (paired A/B) | G1 after |
|---|---|---|---:|---:|
| 11-03 | `416e3ace` | Previous snapshot (dealignai checkpoint, reference `a8c3394af9d2`) | | 114.8 |
| 11-04 | `39e5c200` | **Official weights + OBLIT projections** (new output reference). The gain comes from acceptance, not kernels: the drafter matches the official weights better (code acceptance 2.59 → 2.93). | | 116.5 |
| 11-05 | `6d72bc33` | FC1 G-aware tile deal (`DSV41_MOE_FC1_DEAL=deal`) | +1.21% | 118.7 |
| 11-05 | `de7fcdee` | FC2 status barrier every 1,024 steps instead of 32 | +0.15% | 119.1 |
| 11-07 | `fda6d05a` | glibc malloc settings (`MALLOC_ARENA_MAX=2`, mmap threshold 1 MiB); no speed claim | ~0 | 118.9 (gate) |
| 11-08 | `08a249da` | L2 prefetch paced into the attention and all-gather windows too, including `wo_b` (`PACE_KINDS=ar,attn,ag`, `WOB_MB=6`) | +3.8% | 123.7 |
| 11-08 | `dcd09a30` | Drafter→target glue as CUDA graphs (`DTGRAPH`), shared-expert w2 and compressor-KV prefetch (`R1_SHW2`, `R1_KVSRC`), indexer q chain overlapped with the main q chain (`IDXPAR`) | +2.1%¹ | 126.1 |
| 11-08 | `b22e4aa9` | Skinny MXFP8 v2 for the qkv_a shard, attention-output RoPE + reduce/quant fusion, drafter MoE compaction, FC2 gather + shared add fused, MoE route-prep fusion | +2.2% | 128.9 |
| 11-08 | `166d7aac` | XSPLIT split-tile sparse-MLA decode kernel with in-order look-back, attention-window prefetch at 200 GB/s | +1.25% | 130.2 |
| 11-09 | `4f3cc1c6` | Glue kernels: one-store UE8M0 padding (`RQPAD`), route-prep split grid (`RPSPLIT=split`) | +0.3% | 130.5 |
| 11-10 | `b1306620` | Embedding all-reduce → column all-gather (`COLEMB`), compute-stream priority in captured graphs (`CMPPRIO`) | +0.14%, +0.09% | 130.6 |
| 11-11 | `58899745` | r7-comm2 RDMA all-reduce build with trimmed protocol (`DSV41_RDMA_TRIM`, `deps/dsv41-stage/r7comm2`) | +0.74% | 131.7 |
| 11-11 | `4a04b175` | Certified LM-head screen with exact integer dequant and a tuned launch shape, refine as a candidate-row list on persistent CTAs (`DSV41_R6_HEAD_*`, boot self-check) | +0.37% | 132.0 |
| 11-12 | `4a04b175` | Same launcher, after all 8 nodes moved to spark-os | | **132.4** |

- **Gains.** Each gain is the treatment mean over the mean of the bracketing controls, in one or two serving
  windows (the r5 glue row uses four arms across `r5-w1` and `r5-w2`). The `w1b` (11-08, `08a249da`) and `w4`
  (`166d7aac`) rows each rest on one treatment arm. Every arm matched the G1 digest, and every treatment arm
  matched the exact.py hash. The 9-prompt check ran on selected treatment arms and on every promotion verify, and
  it always matched.
- ¹ The two `w2b` arms behind the `dcd09a30` gain also had `DSV41_MHC_OVERLAP_DRAFT=1` set, and that flag was not
  promoted. The promotion verify (126.07) is +1.96% over the same controls.
- **G1 after.** This is the verify run right after the launcher edit; promotions roll back automatically if any
  check fails. The 11-07 row is a single gate pass.
- **Raw data.** The promotion records and window tables are in `notes/climb-2026-11/`. The 11-05 numbers come from
  `notes/climb-2026-11/campaign-20261105-REPORT.md`.
- **Measured and not promoted:** HT13 (AR + projection fusion, −0.5%), FC2 fused gather and FC2 bridge prefetch
  (killed by their timing rules), HT1 predicted-expert prefetch (lines evicted before use) and L5 4-bit LM-head screen
  (bound too loose). An external request that shared a batch with a verify run once caused a false rollback. Since
  then every verify run stops the LAN/tailnet forwarder and checks the server's request counter.

### Earlier measurements (not re-run on `4a04b175`)

These numbers predate the current launcher. Some were also taken on the previous checkpoint. Each row names its
date and launcher. Decode is faster now, and acceptance on prose and code differs between the two checkpoints, so
treat them as history.

| Measurement | Result | When / launcher / weights |
|---|---:|---|
| 4 concurrent sessions (code, prose, maths, mixed; 512 tokens each), per session mean / slowest | 68.6 / ≥53.1 tok/s | 11-03, `66eaa1d9`, dealignai |
| Same, on the new weights | 69.4 / 53.0 tok/s | 11-04, `39e5c200`, official + OBLIT |
| 32 concurrent requests × 400 greedy tokens, aggregate | 692 tok/s | 11-03, `66eaa1d9`, dealignai |
| TTFT, 16,384-token prompt | 2.49 s | 10-27, `fd1e809e`, dealignai |
| TTFT, 131,072-token prompt | 19.6 s | 10-27, `fd1e809e`, dealignai |
| TTFT, 1,024,611-token prompt (cold, one request) | **169.1 s** (6.1K tok/s) | 11-07, `fda6d05a`, official + OBLIT |
| Short request arriving during a 1M prefill (custom scheduler) | 2.3 s p50 (~130 s stock) | 10-24, `f4e4e214`, dealignai |
| Next turn of a 229K–355K-token conversation (prefix-cache hit) | 1.01 s (1.50 s before incremental tokenization) | 11-03, `416e3ace`, dealignai |

### Quality gates

`GATE_EXACT_ANCHOR=ec2f4437f8e1ed09 bench/gate/gate.sh --live` was last run in full on 7 November (launcher
`fda6d05a`, official weights + OBLIT, before the climb). The script's default anchor is still the old
`a8c3394af9d2`, so set the variable. It passed every step. Every change since then is a speed change and was gated bitwise as
described above.

| Check | Result |
|---|---:|
| exact.py | `ec2f4437f8e1ed09` (anchor) |
| GSM8K, first 200 | 195 / 200 |
| Needle retrieval + copy, 4K–200K | 60 / 60 needles, copy 1.000 |
| 1M prefill probe | 169.07 s, MemAvailable min 9.2 GB |
| MMLU, all 14,042 questions, thinking off (11-04, `39e5c200`) | 88.10% (official without projections 88.50%; previous dealignai checkpoint 84.01%) |

Image input was verified on 2 November (`cf9d04b1`, 16 images, OCR set rate 1.0), before the weight migration.

## Hardware and software

- 8 × DGX Spark (GB10, 128 GB unified memory each). ConnectX-7 links go to a Cisco 8101 switch running SONiC, and
  the all-reduce runs over RoCE.
- Since 12 November all 8 nodes boot the spark-os 0.9.5-1019 image (spark-three runs its `-bb1` build): Ubuntu
  24.04.5 userland, kernel `7.0.0-1019-nvidia`, driver 580.178.04, CUDA 13.0.
- GPU clocks are stock (SM 1969–1989 MHz under the 2000 MHz cap, read on 8/8 on 12 November), with no clock or
  power-limit raises.
- `vllm==0.32.0`, `flashinfer-python==0.7.0.post1`, `torch==2.13.0`, `triton==3.7.1`, `transformers==5.19.0`.
  `vllm-site/pip-freeze.txt` lists 198 packages, identical on 8/8.

## What is changed vs stock vLLM

Everything sits behind `DSV41_*` flags in `prod/dsv41-tp8.sh`. Each flag was promoted only after a paired A/B and the
exactness checks.

- **Weights / outputs:** official checkpoint plus OBLITERATUS refusal-direction projections, applied as exact BF16
  runtime projections (`dsv41_oblit.py`; `deps/dsv41-oblit`). The direction vectors are data and are not in this
  repo.
- **Communication:** custom RDMA all-reduce replacing NCCL for decode-size tensors. The production build is
  r7-comm2 with the trimmed protocol (`deps/dsv41-stage/r7comm2`, built from the g130 v2 lineage in
  `deps/dsv41-stage/g130/rdma` and `deps/rdma_ar_dsv41`). Also: column-wise all-gathers (including the embedding),
  and paced L2 prefetch of the next weights while the all-reduce, attention and all-gather windows run
  (`DSV41_L2_PREFETCH*`; `deps/dsv41-stage/r4/l2-prefetch`, `deps/climb/climb-r1/r1-pf`,
  `deps/climb/climb-r4/r4-xsretime`).
- **MoE:** compact FC1 with a G-aware tile deal (`deps/dsv41-stage/g130/fc1`), M32 native target kernels, Athena's
  packed FC2 kernels (`deps/athena-fc2`, `deps/dsv41-fc2`), FC2 status barrier every 1,024 steps, fused permute and
  route-prep, gather + shared-expert add fold, live-row activation skipping, drafter MoE compaction.
- **Attention / indexer:** XSPLIT split-tile sparse-MLA decode (`deps/climb/climb-r3/r3-xsplit`), deterministic
  indexer top-k with canonical tie-breaking (`deps/dsv41-stage/bs03`, from MoET's `det_topk`, Apache-2.0),
  indexer/main q-chain overlap, row-sharded indexer on long contexts, FP8 o_proj BMM, attention-output RoPE and
  reduce/quant fusion, boundary folds of context KV and mHC (`deps/dsv41-stage/g130/boundary`).
- **Dense:** skinny MXFP8 GEMV kernels for the sharded qkv_a projection (`deps/dsv41-r2`).
- **Speculative decoding:** DSpark (MTP heads, 5 drafts) with adaptive verification calibrated to measured verify
  cost (`prod/pins/`), and a sharded drafter with Markov fold. The target LM head is certified-screened: an exact
  integer-dequant screen, then a refine of only the candidate rows, with a self-check at boot.
- **Host / graphs:** drafter→target glue captured as CUDA graphs, compute-stream priority in captured graphs,
  sync-free argmax.
- **Prefill:** exact sequence-parallel prefill, and a custom scheduler that holds long prefills so short requests
  get through (`deps/dsv41-sched`).
- **Server:** incremental chat-prompt tokenization with an equality guard (`deps/dsv41-stage/incr-tok`), fast
  weight loader, Engram rows punched out of the weight files on seven nodes to free disk.

The climb kits behind the 8–11 November promotions are in `deps/climb/` (installer, exactness notes, tests,
review). `deps/climb/README.md` maps each kit to its flag.

## Layout

| Path | What | Lives on the Sparks at |
|---|---|---|
| `prod/dsv41-tp8.sh` | Production per-rank launcher, sha256 `4a04b175…`, identical on 8/8 | `~/dsv41-tp8.sh` |
| `prod/dsv41-tp8.sh.bak-*` | Every earlier launcher, named after the change that replaced it | `~/` |
| `prod/ops-rel/` | Supervised relaunch (`dsv41-relaunch.sh`, `7bea1cf3…`), status, replay detector | `~/dsv41-opt-20261021/ops/rel/` |
| `prod/pins/` | Adaptive-verification cost pin (prod: `av-pin-k5-g3FU.json`) | `~/dsv41-opt-20261021/pins/` |
| `prod/lanproxy/` | LAN/tailnet TCP forwarder for the API + systemd user unit | `~/dsv41-lanproxy/` |
| `vllm-site/` | Every live file in the vllm / flashinfer installs that differs from the wheel or was added (81 files). Backup and marker files (`.bak-*`, `.orig`, `.absent-*`) are left out. `MANIFEST.tsv` has their sha256; identical on 8/8 | `~/venvs/vllm032/lib/python3.12/site-packages/` |
| `deps/` | Kernel and patch kits the launcher loads (sources, build scripts, tests; no binaries) | `~/`, same relative paths |
| `deps/LOADED_SO_SHA256.tsv` | sha256 of all 12 custom `.so` files mapped by the serving worker (identical on 8/8), with their source and build entry point | |
| `deps/athena-fc2/` | Sources for the two prebuilt FC2 `.so` files in `~/dsv41-fc2` | `~/Athena Hillclimbing stuff/` |
| `deps/climb/` | Master copies of the climb overlay kits in production | operator's Mac |
| `deps/dsv41-oblit/` | OBLIT installer, tests, and the scripts that derive the direction vectors | `~/dsv41-oblit/kit`; the extraction scripts (`obl_dirs.py`, `obl_probe.py`, `official27.sha256`) come from the operator's OBLIT port folder on the Mac |
| `bench/` | Benchmark, goal and exactness harnesses, including the promotion gate | `~/dsv41-bench/` |
| `bench/det/` | 9-prompt determinism harness (`dbgrun.py`) + reference hashes (`ref9.json`, `exact.json`) | `~/dsv41-g5det/scripts/` |
| `notes/` | READMEs and result notes from the `~/dsv41-*` experiment folders, climb records, raw results behind this README | `~/dsv41-*/` |

Not included: model weights, the OBLIT direction vectors (`obliteratus_dirs.npz`, sha256 `6fe62c07…`), compiled
kernels (`.so` / `.cubin` / `.ptx` / SASS), JIT caches, traces, server and rank logs, and bulk result dumps. The
small files behind the tables are included as evidence: the promotion run logs and G1 summaries
(`notes/climb-2026-11/`), the G1 records (`notes/results-20261112/`) and each climb kit's test and compile
summaries. Three of the nine determinism prompts are recorded chat requests (`~/dsv41-g4l3/cases/l3cases.jsonl`),
which are also not included.
`notes/dsv41-opt-20261021/AGENTS_LOG.md` is kept at its 3 November version.

## Restore sketch

1. Create a fresh `vllm==0.32.0` venv at `~/venvs/vllm032`. Copy `vllm-site/` over its site-packages and check the
   `MANIFEST.tsv` hashes.
2. Put `deps/*` back at `~/…` (same relative paths) and rebuild every extension listed in
   `deps/LOADED_SO_SHA256.tsv`:
   - the FC2 pair, built into `~/dsv41-fc2` with `TORCH_CUDA_ARCH_LIST=12.1a`;
   - `rdma_ar_ext.so` from `deps/dsv41-stage/r7comm2` (its `KIT.json` pins the gated `.so` hash);
   - `l2pf` (`deps/dsv41-stage/r4/l2-prefetch`) and `l2pf_g2` (`deps/dsv41-stage/g2/colltrim`);
   - `bs03`;
   - XSPLIT (`deps/climb/climb-r3/r3-xsplit/src/build.sh`, into `~/dsv41-r3/xsplit/build/`);
   - skinny_mx v1 and v2;
   - the three DeepGEMM `_C` trees (`g1/m32-bw` `all-d05`, `g2/fc2c-on-m32all`, `g130/fc1` `deal-base`).

   Check each result against the table. The M32 and fc2c kits JIT their kernels on first launch.
3. Copy `prod/dsv41-tp8.sh` to `~/`, the pin to `~/dsv41-opt-20261021/pins/` and the supervisor to
   `~/dsv41-opt-20261021/ops/rel/`.
4. Weights go at `~/models/dsv41/official-fp8` (official `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
   `1bae0e10`). Generate the projections with `deps/dsv41-oblit/kit/obl_dirs.py` into
   `~/models/dsv41/oblit/obliteratus_dirs.npz` and check sha256 `6fe62c07…`. Engram rows are punched on seven nodes,
   so never rsync model files from a punched node.
5. Launch through the supervisor (`dsv41-relaunch.sh --expect-launcher 4a04b175`). Then check that
   `bench/goal/goal_single.py` gives digest `2f7fdccbd388225a`, `bench/exact.py` gives `ec2f4437f8e1ed09`, and
   `GATE_EXACT_ANCHOR=ec2f4437f8e1ed09 bench/gate/gate.sh --live` passes.

## Credits and licences

- FC2 packed MoE kernels by Athena (`deps/athena-fc2`).
- `det_topk` from MoET (`deps/dsv41-stage/bs03/moet-src-7f60bee`, Apache-2.0, licence included).
- Refusal directions from OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated (MIT), recovered by diffing against the
  official checkpoint. Their weights are not used.
- Ideas re-implemented, not copied, from knapcio's TP4 stack (certified LM head; their code is AGPL),
  christopherowen/spark-ds41f (routed-expert scale codes, MoE glue, top-512, expert L2 prefetch, paced prefetch,
  malloc settings) and rhys101's 8-Spark SGLang build.
- vLLM and FlashInfer files under `vllm-site/` keep their upstream Apache-2.0 licences. The DeepGEMM / CUTLASS
  header trees under `deps/dsv41-stage/*/tree*` keep their own licence files.
