# DeepSeek V4.1 Flash on 8 × DGX Spark (vLLM, TP8)

DeepSeek-V4.1-Flash-1017 (FP8 attention / FP4 experts) served across eight NVIDIA DGX Sparks
with a patched vLLM 0.32.0: tensor parallel 8, DSpark speculative decoding from the checkpoint's
MTP heads, 1M-token context, image input, and an OpenAI-compatible API.

Every optimisation here is **exact**: greedy output is bitwise identical to the unoptimised
reference build (anchor digest `a8c3394af9d2`), so speed never trades against output quality.

## Performance

Measured **3 November 2026** on production launcher `66eaa1d9` (kernels identical to the current
`416e3ace`, which only adds server-side incremental tokenization):

| Workload | 1 request | 4 concurrent sessions, per session | 32 concurrent requests, aggregate |
|---|---:|---:|---:|
| Coding | **108.6 tok/s** | **65.2 tok/s** | |
| Prose | **87.0 tok/s** | **54.6 tok/s** | |
| Maths | **148.8 tok/s** | **82.5 tok/s** | |
| Mean | **114.8 tok/s** | **68.6 tok/s** (204.8 tok/s total) | **692 tok/s** |

- **1 request:** greedy decode, 1,024 output tokens, time to first token excluded, mean of 2 runs per prompt type. Repeat runs gave byte-identical outputs. DSpark acceptance length: code 2.59, prose 1.85, maths 3.59.
- **4 sessions:** one code, one prose, one maths and one mixed session running together, 512 output tokens each, 3 runs. The mixed session ran at 72.1 tok/s. The slowest session in any run never fell below 53.1 tok/s.
- **32 requests:** 32 streams of 400 greedy tokens each, 12,800 tokens in 18.5 s wall clock (prefill included), 32/32 ok.

**Time to first token** (cold, one request, no prefix-cache hits):

| Input tokens | TTFT | Prefill rate |
|---:|---:|---:|
| ~100 | **0.21 s** | |
| 16,384 | **2.49 s** | 6.6K tok/s |
| 131,072 | **19.6 s** | 6.7K tok/s |
| 1,048,576 | **166–169 s** | 6.2–6.3K tok/s |

**TTFT under load and in multi-turn sessions:**

| Scenario | TTFT |
|---|---:|
| Short request arriving during a 128K prefill | **4.3 s** (21.7 s on stock scheduler) |
| Short request arriving during a 1M prefill | **2.3 s** p50 (~130 s on stock scheduler) |
| Next turn of a 229K–355K-token conversation (prefix-cache hit) | **1.01 s** (1.50 s before incremental tokenization) |

The ~100-token row is from the 3 November run. The 16K, 128K and 1M rows come from the
27 October prefill window, launcher `fd1e809e`. The 1M row was re-checked on 29 October,
launcher `5b0a889c`, at 168.6 s; decode-only changes since then don't touch prefill.

**Quality gates** run before any change is promoted (`bench/gate/gate.sh`):

| Check | Result |
|---|---:|
| Greedy exactness vs reference anchor | identical (`a8c3394af9d2`) |
| GSM8K, first 200 | 195 / 200 |
| Needle retrieval + copy, 4K–200K | 60 / 60 needles, copy 1.000 |
| Needle + copy at 500K and 1M | pass |
| Image input (OCR set) | rate 1.0 |

## Hardware

- 8 × DGX Spark (GB10, 128 GB unified memory each), ConnectX-7 links to a Cisco 8101 switch running SONiC; all-reduce over RoCE.
- DGX OS kernel `7.0.0-1019-nvidia`, driver 580.178.04, CUDA 13.0.
- GPU clocks stock; no clock or power-limit raises.
- `vllm==0.32.0`, `flashinfer-python==0.7.0.post1`, `torch==2.13.0`, `transformers==5.19.0` (`vllm-site/pip-freeze.txt`).

## What is changed vs stock vLLM

All behind `DSV41_*` flags in `prod/dsv41-tp8.sh`, each promoted only after a pinned A/B and the exactness gate:

- **Communication:** custom RDMA all-reduce (`deps/rdma_ar_dsv41`, v2 variant in `deps/dsv41-stage/g130/rdma`) replacing NCCL for decode-size tensors, column-wise all-gather, L2 prefetch of the next layer's weights while the all-reduce is in flight (`DSV41_L2_PREFETCH*`, paced).
- **MoE:** compact FC1, M32 native target kernels, Athena's packed FC2 kernels (`deps/athena-fc2`, `deps/dsv41-fc2`), thinned FC2 barriers, fused permute, live-row activation skipping.
- **Attention / indexer:** deterministic sparse-indexer top-k with canonical tie-breaking (`deps/dsv41-stage/bs03`, from MoET's `det_topk`, Apache-2.0), row-sharded indexer on long contexts, FP8 o_proj BMM, boundary folds of context KV and mHC (`deps/dsv41-stage/g130/boundary`).
- **Speculative decoding:** DSpark (MTP heads 0–2, 5 drafts) with adaptive verification calibrated to measured verify cost (`prod/pins/`), sharded drafter with Markov fold, screened FP8 LM head for drafts and a certified screened target head.
- **Prefill:** exact sequence-parallel prefill, a custom scheduler that holds long prefills so short requests get through (`deps/dsv41-sched`).
- **Server:** incremental chat-prompt tokenization with an equality guard (`deps/dsv41-stage/incr-tok`; server tokenize 375 ms → 6.5 ms on 229K–355K-token prompts), fast weight loader, Engram rows punched out of the weight files on three nodes to free disk.

## Layout

| Path | What | Lives on the Sparks at |
|---|---|---|
| `prod/dsv41-tp8.sh` | Production per-rank launcher, sha256 `416e3ace…` | `~/dsv41-tp8.sh` |
| `prod/dsv41-tp8.sh.bak-*` | Every earlier launcher, named after the change that replaced it | `~/` |
| `prod/ops-rel/` | Supervised relaunch (`dsv41-relaunch.sh`, `7bea1cf3…`), status, replay detector | `~/dsv41-opt-20261021/ops/rel/` |
| `prod/pins/` | Adaptive-verification cost pins (prod: `av-pin-k5-g3FU.json`) | `~/dsv41-opt-20261021/pins/` |
| `prod/lanproxy/` | LAN/tailnet TCP forwarder for the API + systemd user unit | `~/dsv41-lanproxy/` |
| `vllm-site/` | Every file in the vllm / flashinfer installs that differs from the wheel or was added; `MANIFEST.tsv` has sha256 | `~/venvs/vllm032/lib/python3.12/site-packages/` |
| `deps/` | Kernel and patch kits the launcher loads (sources, build scripts, tests; no binaries) | `~/` same relative paths |
| `deps/athena-fc2/` | Sources for the two prebuilt FC2 `.so` files in `~/dsv41-fc2`; `PROD_SO_SHA256.txt` pins the production builds | `~/Athena Hillclimbing stuff/` |
| `bench/` | Benchmark, goal and exactness harnesses, including the promotion gate | `~/dsv41-bench/` |
| `notes/` | READMEs and result notes from every `~/dsv41-*` experiment folder | `~/dsv41-*/` |

Not included: model weights, compiled kernels (`.so` / `.cubin` / `.ptx`), JIT caches, traces, logs, raw result dumps.

## Restore sketch

1. Fresh `vllm==0.32.0` venv at `~/venvs/vllm032`; copy `vllm-site/` over its site-packages and check `MANIFEST.tsv` hashes.
2. Put `deps/*` back at `~/…` (same relative paths). Rebuild the extensions: `rdma_ar_dsv41` (`build-k1pl/rdma_ar_ext.so`) and `dsv41-stage/g130/rdma`, `dsv41-stage/bs03/build.sh`, and the FC2 pair (`deps/athena-fc2/pack/build_fc2_pack.py`, `deps/athena-fc2/graph/run_fc2_graph_build.sh`, `TORCH_CUDA_ARCH_LIST=12.1a`) into `~/dsv41-fc2`; check against `PROD_SO_SHA256.txt`. The M32 / fc2c kits JIT on first launch.
3. Copy `prod/dsv41-tp8.sh` to `~/`, pins to `~/dsv41-opt-20261021/pins/`, supervisor to `~/dsv41-opt-20261021/ops/rel/`.
4. Weights at `~/models/dsv41/uncensored-fp8` (not in this repo). Engram rows are punched on three nodes, so never rsync model files from a punched node.
5. Launch through the supervisor: `dsv41-relaunch.sh --expect-launcher 416e3ace`, then `bench/gate/gate.sh --live`.

## Credits and licences

- FC2 packed MoE kernels by Athena (`deps/athena-fc2`).
- `det_topk` from MoET (`deps/dsv41-stage/bs03/moet-src-7f60bee`, Apache-2.0, licence included).
- Ideas re-implemented, not copied, from knapcio's TP4 stack (certified LM head; their code is AGPL), christopherowen/spark-ds41f (routed-expert scale codes, MoE glue, top-512, expert L2 prefetch) and rhys101's 8-Spark SGLang build.
- vLLM and FlashInfer files under `vllm-site/` keep their upstream Apache-2.0 licences.
