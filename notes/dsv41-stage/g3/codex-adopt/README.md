# g3 / codex-adopt: three adopted Codex items as default-off flags (staged, NOT installed)

Prepared 2026-10-26 ~19:15 UTC, offline only: no restart, no GPU job, no edit of a live file.
CPU work ran on spark-four (nice 19, 3 GiB cap, GPU hidden) while serving.
Live base (checked identical on 8/8 at prep time): launcher 592793fd, supervisor 7bea1cf3, and these three site files:

| # | flag (default off) | site file | live -> staged | what it does |
|---|---|---|---|---|
| 1 | `DSV41_TOPK_CANON_FUSED=1` (needs the live `DSV41_TOPK_CANON=1`) | `vllm/model_executor/layers/sparse_attn_indexer.py` | 7112c01d -> 481905c2 | Fused CANON, Codex R5 7fc2d1ab: one Triton launch per CANON call instead of where + radix sort + eq + masked_fill + copy |
| 2 | `DSV41_DECODE_REMAP_REUSE=1` (`check` = verify only) | `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` | c8a2da55 -> b059bdf3 | decode `compute_global_topk_indices_and_lens` once per (index source, KV source) interval and forward: 38 -> 8 launches per step |
| 3 | `DSV41_IDX_OMIT_T2S=1` | `vllm/v1/attention/backends/mla/indexer.py` | 4753707e -> d9f4c5c9 | the dead `token_to_seq` of the CUDA indexer prefill metadata is neither allocated nor written (env-gated constexpr) |

The flags are independent. With all three unset, the installed files run the live code: see "Exactness evidence" below.
Diffs: `diffs/1-fused-canon.diff`, `diffs/2-remap-reuse.diff`, `diffs/3-omit-t2s.diff`. The live copies are in `base/`.
Next to this package, on 8/8, are the two Codex standalone kits: `~/dsv41-stage/g3/fused-topk/` and `~/dsv41-stage/g3/candidate-only-indexer/` (see section 6).

## 1. Expected effect (projection, not measured)

| item | decode, 1 user | decode, 4 users | prefill / memory |
|---|---|---|---|
| 1 fused CANON | 8 CANON calls per step (index sources 2/8/14/20/24/28/32/36). Today about 20 us each (radix sort 14.4 us + 4 elementwise kernels), 0.17 ms per step. Fused is one kernel of about 3-5 us. **-0.12..-0.15 ms per step, G1 +0.4..0.5%** | **-0.15..-0.2 ms per step**, G4 prose MIN about +0.4..0.5% (at -1 ms on every step = +2.7%) | ~0 on TTFT (well under 0.1%). Removes the stock sort temporaries (17 B per element: int64 sort indices, int32 keys/values, bool mask), for example 17 MiB per 2,048-row prefill chunk. Decode memory: none |
| 2 remap reuse | 30 fewer remap launches per step (2-4 us each in the graph). **-0.06..-0.09 ms, G1 +0.2..0.3%** | **about -0.1 ms**, G4 prose MIN about +0.25% | Prefill path unchanged (the prefill version was rejected). Memory: none (at most 8 tensors of [rows, 512] int32 live per forward, a few hundred KiB). 30 fewer kernel nodes per captured decode graph |
| 3 omit token_to_seq | 0 | 0 | Late-1M step: about **-896 MiB** of allocations (ratio-1 builder: 128 subchunks x 4 MiB; three ratio-2 builders: 3 x 64 x 2 MiB) and the same bytes of single-CTA stores. Roughly 5-12 ms per late-1M step, **-0.2..-0.4 s per cold 1M prompt (~0.1-0.2%)**, about 0 at 128K or less. The 1M MemAvailable floor (gw2: 7.8 GB on spark-one) should rise by up to ~0.9 GB |
| 1+2 bundle | **-0.18..-0.24 ms per step, G1 about +0.6..0.9%** (100.4 -> ~101.1-101.3) | **-0.25..-0.3 ms**, G4 prose MIN about +0.65..0.8% | |

Neither 1 nor 2 alone reliably clears the 0.5% rule, so test them as the bundle "C+B" (with 3, which is decode-neutral).
`run-tests.sh bench` measures each lever in isolation: per-call and per-step us under graph replay for 1 and 2, held bytes and build ms of one late-1M step for 3.

## 2. Exactness

- **1 (CANON).** The kernel maps negatives to INT32_MAX, sorts each 512-wide row ascending in registers (Triton bitonic sort), then maps INT32_MAX to -1. That is the same integer multiset per row and the same order as stock, so the result is bitwise identical, positive INT32_MAX inputs included.
  - It runs only on 2-D int32 [n, 512] views with unit column stride and row stride >= 512, on CUDA SM12x device 0. Anything else keeps the stock path.
  - One compile key (`do_not_specialize` row_stride, `do_not_specialize_on_alignment` t_ptr), registered with the JIT warmup, so there is no compile at inference.
  - Flag off: the kernel object is not created and no warmup is registered.
- **2 (remap reuse).** The decode remap reads 5 inputs:
  - the index source's top-k rows;
  - the KV source's block table and block size;
  - the request map (`token_to_req_indices`);
  - the validity (`is_valid_token = slot_mapping >= 0`) of the decode rows.

  The first two are the same object for every layer of an interval. The last two come from the layer's SWA group, but their content is the same in every group: `token_to_req_indices` is built from the common `query_start_loc`; padding is -1 in every group; a real token has a slot in every group.
  - Per ForwardContext (weakref owner, reset when the context changes), the **index-source layer always recomputes and publishes**. Consumers reuse only an entry with the same key (index source, KV prefix, block size, num_decodes, num_decode_tokens) and the same metadata and top-k buffer objects. A consumer without a matching entry computes its own (stock).
  - Under CUDA graphs, the publisher's launch is captured and reruns on every replay. Consumers capture no remap.
  - `DSV41_DECODE_REMAP_REUSE=check`: every layer computes stock and consumes it, then compares with the publication (lens for every row, indices of the valid rows, `is_valid_token` across groups). An eager mismatch raises. During graph replays a device counter counts mismatches, and the next eager call raises.
- **3 (token_to_seq).** No CUDA code reads `chunk.token_to_seq`: `sparse_attn_indexer` and its `cp_gather_indexer_k_quant_cache` ABI take `block_table` + `local_cu_seq_lens` only. The one reader is the ROCm AITER gather, and the flag requires CUDA.
  - Grep of the install: the other `.token_to_seq` readers belong to other chunk types or models (MLA chunked context, dots3, kimi_k3, glm5next).
  - The `WRITE_TOKEN_TO_SEQ` constexpr is process-wide, so there is one warmup key per config, as today (72 keys).
  - Flag on: `token_to_seq` is an empty view of `cu_seq_lens`, and the False specialization never touches it. PCP/DCP builds raise (TP only).

### Exactness evidence collected offline (spark-four, while serving: nice 19, 3 GiB cap, GPU hidden)
- `tests/test_static.py` **PASS** (stdlib AST, 30+ checks). Each staged file is its live base plus flag-gated code only:
  - with the flags unset, `_canon_topk_`, `SparseAttnIndexer.__init__`, `_forward_decode`, the metadata kernel (True inlined) and `build_prefill_chunk_metadata` are AST-identical to live, and no other live class or function changed;
  - the fused kernel equals the kernel the Codex kit qualifies (name aside).
- `tests/sass_check.py` **PASS** (Triton -> ptxas sm_121, cuobjdump):
  - metadata kernel, staged with `WRITE_TOKEN_TO_SEQ=True`: SASS **identical** to live for 3 key shapes (192/208/560 instructions, 40/40/54 regs). With False it loses 48-72 instructions and 8 of 24 global stores (the token_to_seq loop).
  - fused CANON kernel: SASS identical to the Codex kit kernel (1064 instructions, 40 regs, 0 stack/local, 1 KiB shared, 20 barriers).
- Triton-interpreter tests (CPU) **PASS**:
  - `test_canon.py`: 72 cases, rows 1/3/7 x 8 value patterns x strides 512/527/1024, whole backing incl. canaries.
  - `test_remap.py`: real DSv4.1 topology from config.json (38 compressed layers, 8 intervals, 15 SWA groups). Stock == reuse with 8 remaps per forward in 14 eager scenarios (incl. padding rows with live-looking indices). Also: context isolation main/replay/main; an orphan consumer computes stock; a reused ForwardContext object with new top-k refreshes; check mode passes clean groups and raises on a corrupted `is_valid_token` or `token_to_req_indices`.
  - `test_t2s.py`: arms live (base/ copy) / staged flag off / staged flag on, 8 scenarios incl. sub-chunked 1M-shaped and odd start_idx. Off == live on every field incl. token_to_seq; on == live on every field except token_to_seq (an empty view sharing cu_seq_lens' storage).
- The staged CANON module loads as `vllm.model_executor.layers.sparse_attn_indexer` (custom op registers once), arms with the flags, and declines the fused path on CPU tensors.

## 3. Stopped window (GPU; ~8 min on one node, nothing installed)
From the Mac, with the service stopped on 8/8:
```sh
ssh spark-two 'bash ~/dsv41-stage/g3/codex-adopt/ops/run-tests.sh bench'     # gpu tests + microbenches
# in parallel on other nodes (all GPUs are free while stopped), the Codex kits (section 6):
ssh spark-three 'bash ~/dsv41-stage/g3/codex-adopt/ops/run-kits.sh gpu'
```
`run-tests.sh gpu|bench` refuses while dsv41-rank is active. It uses private Triton/DeepGEMM caches (DeepGEMM seeded by copying `~/.cache/vllm/deep_gemm`) and checks that the serving caches are unchanged. It runs:
- `test_canon.py --gpu`: warmup exactly as the registry does, then a compile trap (0 compiles allowed).
  - Staged `_canon_topk_` vs stock, byte for byte over the whole backing: rows 1-192, 256, 384, 1000, 2048, 16384; topk-buffer views at offsets 0/37; 527/1024-wide padded rows; 4-byte-misaligned storage; 8 value patterns (realistic top-k rows, -1, INT32_MIN, INT32_MAX, duplicates, full-range, sorted, reversed).
  - The fused path must run for each of these; width 256, int64, strided and 3-D inputs must stay stock.
  - CUDA graph replay with 5 fresh inputs per row count.
  - `--bench`: stock vs fused, us per call and per step (8 calls) at rows 1-16384.
- `test_remap.py --gpu`: the eager scenarios above plus 32/64-request batches.
  - CUDA-graph capture of the stock arm (38 remaps) and the reuse arm (8 remaps), 4 refills with new top-k / block tables / validity before each replay: bitwise equal.
  - Check mode under graphs: a clean replay counts 0, a corrupted replay is counted, and the next eager call raises.
  - `--bench`: one emulated 38-layer step, stock vs reuse, at 1-192 rows.
- `test_t2s.py --gpu`: the three arms in separate processes, each warming the 72 keys, then 0 compiles.
  - ks/ke/cu/... bitwise on full-size 1M scenarios (128 and 64 subchunks).
  - Real `fp8_fp4_mqa_logits` + `top_k_per_row_prefill` + CANON on the first, middle and last 1M subchunk (ratio 1 and 2): deterministic per arm and identical across arms.
  - Held bytes and build ms of one late-1M step (1 ratio-1 + 3 ratio-2 builders) per arm.

Gate to serve: all three PASS; serving caches unchanged; microbench savings of at least 0.08 ms per step at rows 6 (bench 1) and 0.04 ms (bench 2). If either is below, serve only the other.

## 4. Install (service stopped; inert until a flag is set) and serving A/B
```sh
ssh spark-one cat ~/dsv41-stage/g3/codex-adopt/ops/g3ca-all.sh | bash -s status      # expect baseline 8/8
ssh spark-one cat ~/dsv41-stage/g3/codex-adopt/ops/g3ca-all.sh | bash -s install     # backups <file>.bak-g3-codex-adopt
```
The preflight refuses unless all 3 files are at the live sha (or already staged) on that node; nothing is half-installed. The supervisor's site manifest does not cover these files, so it stays 0501bdcb13e0.
- First launch after the install: the indexer metadata kernel's source changed, so Triton compiles its 72 warmup keys once per node (seconds, at warmup, not at inference). Its SASS is identical (section 2).

Arms (fresh AV pin from this window for every arm; telemetry off; `--expect-launcher 592793fd`; exact.py first):
- C: no flags, the control. It must still give a8c3394af9d2 and the pin's counters, because the files are installed but inert.
- V (verification, untimed): `DSV41_DECODE_REMAP_REUSE=check DSV41_TOPK_CANON_FUSED=1 DSV41_IDX_OMIT_T2S=1`, then exact.py + G1 + G4 + a 128K prompt. Expect no `DSV41_DECODE_REMAP_REUSE=check` RuntimeError on 8/8.
- CB: `DSV41_TOPK_CANON_FUSED=1 DSV41_DECODE_REMAP_REUSE=1 DSV41_IDX_OMIT_T2S=1`, pinned ABAB against C (>= 2 pairs).
  - `ssh spark-one cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh | bash -s -- --label g3-CB DSV41_GPU_CLOCK_MHZ=2000 DSV41_TOPK_CANON_FUSED=1 DSV41_DECODE_REMAP_REUSE=1 DSV41_IDX_OMIT_T2S=1`
  - Log lines to require on 8/8:
    - `DSV41_TOPK_CANON_FUSED=1: fused width-512 CANON kernel registered for JIT warmup` and `DSV41_TOPK_CANON_FUSED: fused CANON kernel in use`;
    - `DSV41_DECODE_REMAP_REUSE=1: layer N reuses the decode remap`;
    - no `JIT compilation during inference: _dsv41_canon_topk512_kernel`.
- Optional single-lever arms if CB wins: C1 (`DSV41_TOPK_CANON_FUSED=1`), B1 (`DSV41_DECODE_REMAP_REUSE=1`).
- Item 3 is judged on the cold 1M probe of the final instance: TTFT, MemAvailable minimum with 1 Hz memwatch, and exactness. It does not change decode.

Promotion (if CB passes the rule): add to `~/dsv41-tp8.sh` after the `DSV41_TOPK_CANON` line, then relaunch with the new launcher sha:
```sh
export DSV41_TOPK_CANON_FUSED=${DSV41_TOPK_CANON_FUSED:-1}      # g3 codex-adopt #1 (needs sparse_attn_indexer.py 481905c2)
export DSV41_DECODE_REMAP_REUSE=${DSV41_DECODE_REMAP_REUSE:-1}  # g3 codex-adopt #2 (needs flashinfer_sparse.py b059bdf3)
export DSV41_IDX_OMIT_T2S=${DSV41_IDX_OMIT_T2S:-1}              # g3 codex-adopt #3 (needs mla/indexer.py d9f4c5c9)
```

## 5. Rollback
- One launch: set the flags to 0 (or unset them).
- Files (service stopped): `ssh spark-one cat ~/dsv41-stage/g3/codex-adopt/ops/g3ca-all.sh | bash -s uninstall`. It restores the `.bak-g3-codex-adopt` copies and verifies the live shas 7112c01d / c8a2da55 / 4753707e on 8/8.
- Run it before any other package's rollback of these three files.

## 6. The two Codex kits (standalone qualification only, nothing installed)
- `~/dsv41-stage/g3/fused-topk/` (Codex profile-targets-20261026, 476 qualification + 41 timed cases). It qualifies the same kernel as item 1: the sass_check shows identical SASS.
  - Stopped window: `ops/run-kits.sh smoke` (README short test) or `ops/run-kits.sh gpu`, which runs:
    - `run.py --mode all --execute-gpu` (full);
    - `compute-sanitizer --tool memcheck` (qualify, no graphs, timing off);
    - then the candidate-only qualifier.
  - Results go to `fused-topk/runs/`. A final status other than `PASS` is not a qualified run.
  - **Stage change:** `provenance.json` `ast_sha256` re-derived under the Sparks' Python 3.12.3. The kit was built under Python 3.14, whose `ast.dump` text differs, so `run.py` aborted with "AST pin mismatch" on the target. The 3.14 values are kept as `ast_sha256_py314`, `kernels_sha256` is unchanged and still checked, and `bundle-manifest.json` was updated for `provenance.json` only. Originals: Mac scratchpad `g3ca/fused-topk-*.orig.json`.
- `~/dsv41-stage/g3/candidate-only-indexer/`: prefill consumer layers 24/28/32/36 score only their layer-20 candidates.
  - GPU command (`run-kits.sh gpu`, with `DSV41_TOPK_CANON=1`): `~/venvs/vllm032/bin/python -B qualify_gpu.py --run --rows 1,3,4,5,128 --widths 131072,524288,1048576 --patterns random,zeros,duplicates,sentinels,edge,short,empty,special --warmup 2 --iterations 10 --samples 4 --cache-dir ./qualification-cache-<t> --output ./gpu-results-<t>.json`.
  - Its 13 pinned installed sources and the model config.json match on 8/8 (`ops/kit-pins.py`).
  - Its `candidate.patch` edits `sparse_attn_indexer.py` (7112c01d -> d3f3d2...), the same file as item 1. If both are adopted, the two need a merged file.
  - Do not run `capture.py` (Codex's ssh capture tool).
- CPU self-checks of both kits **PASS** on spark-four (`run-kits.sh cpu`, run in scratch copies because they rewrite their JSONs):
  - fused-topk `check_cpu.py`: 152 oracle fixtures / 719,872 values, 17 rejections; `run.py --mode plan` 476 + 41 cases;
  - candidate-only `check_candidate.py` / `check_harness.py` / `qualify_gpu.py --plan`.

## 7. Files
- `FILES`: the 3 site files with accepted-before and staged sha.
- `site/`, `base/`, `diffs/`.
- `SHA256SUMS`: the package digest, checked by `ops/stage.sh` on 8/8.
- `ops/`:
  - `g3ca-site.sh` (per node) and `g3ca-all.sh` (Mac, 8/8): install, uninstall, status;
  - `run-tests.sh` (cpu|gpu|bench|all) and `run-kits.sh` (cpu|smoke|gpu);
  - `kit-pins.py`, `stage.sh`.
- `tests/`: `test_static.py`, `test_canon.py`, `test_remap.py`, `test_t2s.py`, `sass_check.py`, `g3ca_common.py`.
- `results/<host>-<what>-<stamp>/`: per run.
