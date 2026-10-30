# engram-disk: A/B protocol (prepared 2026-10-22, reviewed 2026-10-23, NOT run)

> **Review changes (2026-10-23).** Workloads: the greedy single/conc runs and the synthetic-word prefill made Engram rows repeat far more than real traffic does (identical greedy text across repeats and across 32 copies of 3 prompts; ~5000 synthetic words), so in-step dedup and the row cache would have hidden most of the disk cost. The headline numbers now come from sampled runs (t1, distinct seeds, same seeds in every arm) and from `prefill_real.py` (distinct real source text per prompt). Instrumentation: the stats line now also gives `stage_ms` (host time after the sync), `gpu_gap_ms` (GPU-timed gap from hashes-on-host to rows-staged), cumulative totals for per-phase deltas, emits every 10 s, and `collect.sh` keeps every line from every rank. VERIFY: the budget went from 64 stages to all stages (warmup dummy runs alone could have used up 64), capture forwards are disarmed (they would have produced false mismatches), and `verify_reads` / `verify_fwd` counters prove the checks actually ran. The cost model is corrected for Little's law, for rank-dependent reuse, and for cache-tag RAM. The bench adds QD128, p99.9 and `e_max8_ms`.

Question: how much RAM per rank does moving the Engram tables to NVMe give back, and what does it cost in decode and prefill speed?

Mac stage: `$S=/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad/engram-disk`
Rank stage: `~/dsv41-stage/engram-disk/` on all 8 ranks. Results go to `~/dsv41-opt-20261021/engram-disk/<arm>/` on spark-one and `$S/results/<arm>/` on the Mac.

## 1. What the live code does (vLLM 0.32, read from site-packages)

| item | value |
|---|---|
| config | `vllm/config/engram.py`: `cpu_offload`, `use_thp`, `dp_shared_memory`, `embedding_across_dp`. **No disk or mmap mode exists.** The only mmaps are anonymous THP and the /dev/shm DP share. |
| tables | `layers.{1,14}.engram.embed.weight` F8_E4M3 [384,006,168 / 384,016,682 × 256] + `.scale` F8_E8M0 [.. × 8]. These are in shards 47 and 48 at data offset 0 (header 664 / 672 B). |
| hash columns | (max_ngram 4 − 1) × 8 heads = **24 per layer**. Each column owns a prime-sized bucket range of about 16M. |
| TP sharding | Whole heads, `part_n_hash_cols = 3` per rank per layer, as one contiguous row range. Rank 0, layer 1 is [0, 48,000,217); rank 7, layer 14 is [336,014,037, 384,016,682). |
| per-rank bytes | 96.0M rows × (256 + 8) B = **23.604 GiB**: 22.889 GiB fp8 rows + 0.715 GiB scales. Allocated as THP, cudaHostRegister'd and mlocked (DSV41_LOCK_ENGRAM_HOST). |
| rows per token per rank | **6** (2 layers × 3 heads), each a 256 B fp8 row plus an 8 B scale. 48 rows per token across the cluster, and each rank reads its own NVMe. |
| hash | `NgramHashState._hash_ids_kernel` (GPU). Rolling XOR of compressed token id × per-layer multiplier over lookbacks 0..3, then mod prime + offset per head. It depends only on token ids (plus lookback window) and positions. |
| where in the step | `DeepseekV4Model.forward`: hash → `Engram.prepare_embeddings` (both layers, on a side stream, `_engram_lookup_kernel` over UVA host memory) → layer 1 / 14 `_ready_rows` wait → TP all-gather (rdma_ar) → wkv. |
| graphs | `cudagraph_mode=FULL_AND_PIECEWISE`, breakable graphs on. Uniform decode is a **FULL** graph: the lookup kernel is captured, so no host code can run inside a decode step. |
| ahead of time? | At `prepare_inputs` the step's ids are final. The V2 runner has already combined the last sampled token with the drafted tokens on the GPU (`combine_sampled_and_draft_tokens`), so the hashes of the bonus token **and** all drafts can be computed there, before graph replay. They cannot be computed earlier: every n-gram includes the token sampled in the step just finished. Prompt chunks can be hashed one chunk ahead on the CPU, because the prompt ids are known at `add_request`. |

## 2. Design (behind `DSV41_ENGRAM_DISK=1`; inert otherwise)

Files: `payload/vllm/models/deepseek_v41/common/engram_disk.py` (new: numpy, ctypes and threads, no torch), `common/engram.py` (diff 481 lines), `nvidia/model_state.py` (diff 97 lines), and `engram_pread.c` → `libengram_pread.so`.

- **Storage.** Reads come from the checkpoint's own shard files, so no extra disk is needed (spark-one has 32 GB free). `EngramDiskTable` reads each rank's row range: offset = data_base + (vocab_start + local) × 256. This avoids tonyd2wild's first bug, where every rank read rank 0's rows. The 0.715 GiB of scales per rank stay in RAM and are loaded by the normal loader. The fp8 weight param is an empty placeholder with a no-op loader, so the mmap'd 98 GB tensor is never touched.
- **Honest IO.** The fd is opened with O_DIRECT and each row is read as its aligned 4 KiB block (8 KiB when it straddles; 1 row in 16) into a per-thread bounce buffer. This uses no page cache, so freed RAM stays freed and every miss is a real NVMe read. `DSV41_ENGRAM_DISK_DIRECT=0` switches to buffered reads for debugging only.
- **Reader.** `libengram_pread.so` is a persistent pthread pool (`DSV41_ENGRAM_DISK_THREADS`, default 32, plus the caller, so QD is about 33). Both layers' misses go out in **one wave** using per-row fds. ctypes drops the GIL. liburing is not installed on the Sparks (`linux/io_uring.h` is), so io_uring is a follow-up if the bench shows pread threads cap IOPS.
- **Cache.** `DSV41_ENGRAM_CACHE_GIB` is a set-associative cache (16-way, FIFO) of 256 B rows, keyed by (layer, local row). Tag and bytes are always written together, so hits are exact. One cached row costs 256 B, versus a 4 KiB page in the page cache.
- **Graph-safe staging** (`EngramDiskStager`, called from `DeepseekV41ModelState.prepare_inputs` after the lookback window gather and before the forward or graph replay, on the same stream):
  1. The forward's own hash kernel runs on the step's ids, positions, query_start_loc and lookback window.
  2. D2H of this rank's 3 hash columns per layer, then **one host sync**.
  3. Cache → O_DIRECT misses → token-major host staging, plus resident scales.
  4. Async H2D into a fixed device staging table.
  5. The **stock `_engram_lookup_kernel`** runs with identity ids (`ids[t, head_start+j] = vocab_start + 3t + j`) into the layer's persistent `staged_rows`.

  The FULL decode graph reads `staged_rows` at a fixed address. In disk mode `Engram.prepare_embeddings` is a no-op, and its side stream and event are off.
- **Exactness.** The bytes are verbatim checkpoint bytes and the dequant runs in the same kernel with the same arithmetic, so the output is bit-identical by construction. Checks:
  - CPU test (passed on the Mac).
  - `test gpu`: staged lookup equals resident lookup bitwise, in bf16 and packed modes.
  - `DSV41_ENGRAM_DISK_VERIFY=N` double-reads rows buffered and compares the forward's hashes with the staged ones on eager steps.
  - exact.py digest must equal control.
- **Prefetch** (`DSV41_ENGRAM_PREFETCH=1`): a numpy port of the hash kernel (checked against a scalar transliteration on the real primes and multipliers) hashes each prefilling request's next prompt chunk. A background thread warms the cache in 2048-row sub-batches, so it delays a stage by at most one sub-batch. It is only a hint: a wrong prefetch costs hit rate, never correctness. It forces a cache of at least `DSV41_ENGRAM_PREFETCH_GIB` (0.25 GiB). Decode cannot be prefetched (see §1).
- **Stats.** Every `DSV41_ENGRAM_DISK_STATS` stages (default 200) the rank log prints an `engram-disk stats:` line: rows, unique rows and reads per stage, cache hit rate, read_ms, host_ms, sync_ms, max_read_ms, preads, device MiB and prefetch counters.

## 3. Expected cost model (to be replaced by nvme_bench numbers)

Assumptions: random 4K O_DIRECT read about 80-100 µs at QD1. Synchronous pread threads give QD = threads + 1 = 33, so by Little's law IOPS ≤ 33 / latency-at-QD33 ≈ **300-400K** (not 400-600K; that needs QD ≥ 64). Spark NVMe ESL01TBTLCZ; the block scheduler is `none` (checked on spark-one), and the model is on ext4 /home on nvme0n1p2, where O_DIRECT is honoured. Step times: 28-34 ms single, about 110 ms at 16-32 users. Prefill runs at 5.4-6.1K tok/s, so a 16K chunk takes about 2.8 s.

| case | tokens/step | reads/rank/step (cache0) | added per step | relative |
|---|---|---|---|---|
| single decode (1 + ≤5 drafts) | ≤6 | ≤36 (1 wave at QD33, ~2 deep) | sync bubble 0.2-0.4 ms + IO 0.15-0.25 ms, max over 8 ranks (tail) → **0.5-1.0 ms** | **−1.5 to −3.5 % tok/s** |
| conc8 | ~32 | ~190 | ~0.4 + 0.4 ms | ~−1 % |
| conc32 | ≤192 (32 × 6) | 600-1150 | ~0.4 + 1.2-2.3 ms | **−1.5 to −2.5 %** |
| 16K prefill chunk | 16,384 | 98,304 (unique: ~90 %+ on the 4-gram ranks 6/7, far less on the 2-gram ranks 0/1; the step waits for 6/7) | 0.22-0.33 s IO (≈90K at 300-400K IOPS) + ~30 ms CPU | **+8 to +12 % TTFT** |
| 128K prompt (8 chunks) | | ~786K reads (~3 GiB device reads) | +1.9-2.7 s on ~22-24 s | **+8 to +12 %** |
| 128K with prefetch | | chunk n+1 read during chunk n | first chunk only, ~+0.3 s | **~+1-2 %** (16K single chunk: still ~+0.3 s) |

With a 4 GiB cache (16.8M rows, 17.5 % of the 96M rows per rank), Zipfian n-gram reuse should give roughly 55-75 % hits *averaged over ranks* (knapcio reports 67-76 % at 4 GiB). **Reuse is not uniform across ranks.** TP splits the 24 hash columns n-gram-major: ranks 0-1 hold 2-gram heads, 3-4 hold 3-gram heads, and 6-7 hold 4-gram heads; ranks 2 and 5 are mixed. Every step waits for the slowest rank, so the critical path is the 4-gram ranks, where new text has little reuse. Expect the cache to help ranks 0-2 much more than 6-7, and the decode gain from caching to be well below the averaged hit rate (perhaps −1.5 to −3 % single rather than −1 to −2 %). Read the per-rank hit rate and `max_read_ms`, not rank 0's.

**Noise floor.** Restart-to-restart spread on single-stream decode is about ±4 %. A −2 % decode effect cannot be resolved from one launch per arm. The decode cost is therefore read mechanistically, and the tok/s A/B is only a consistency check:
- added time per step ≈ `gpu_gap_ms/stage`, maxed over ranks, plus the runner tail of about 0.05 ms;
- divide that by the control step time (ITL × accepted tokens per step).

Prefill (about 10 %) and conc32 are above the noise floor. The per-step sync bubble does not shrink with caching. Below about 0.3 ms it is the floor of any exact disk design: the step's own sampled token is needed before any read can start.

**RAM given back per rank** (MemAvailable, since the THP tables are mlocked anonymous memory today): 23.60 − 0.715 (scales) − cache × (1 + 8/256 for int64 tags) − about 0.06 (staging, pinned plus device) GiB. That is **≈22.8 GiB with cache0, ≈18.7 with cache4G and ≈14.6 with cache8G**, and prefetch alone holds about 0.26 GiB. The cache is `np.zeros`, so it is calloc-lazy: MemAvailable shrinks as it fills, so read the under-load minimum, not the startup value. To show the freed RAM is usable, an optional extra arm is disk4pf with `DSV41_KV_BYTES=32G` (≈2× KV tokens).

## 4. Hard prerequisites

- The cluster is free: the 1M-context run is finished and the user or orchestrator has handed over the window.
- All commands run from the Mac unless marked. dsv41-launch-all.sh and launch-engram-arm.sh must run **from the Mac**, because spark-one cannot ssh to six, seven or eight.
- Live hashes at prep time (8/8 identical): engram.py `aa8aad1ba64f…`, model_state.py `53e4fe39b272…`, engram_disk.py absent. Launcher unchanged; no edit is needed, since arms set env through `systemd-run -E`.

## 5. Run sheet

```bash
S=/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad/engram-disk
H="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
# 0. (already done at prep) stage + build 8/8:   bash $S/push.sh
# 1. stop service, install 8/8, verify
for h in $H; do ssh -n $h 'systemctl --user stop dsv41-rank' & done; wait
for h in $H; do ssh -n $h 'bash ~/dsv41-stage/engram-disk/install.sh install' & done; wait     # expect "installed 3/3" x8
for h in $H; do ssh -n $h 'bash ~/dsv41-stage/engram-disk/install.sh status' & done; wait      # PATCHED x24
# 2. offline checks with the service stopped (GPU free)
ssh spark-one '~/venvs/vllm032/bin/python ~/dsv41-stage/engram-disk/test_engram_disk.py gpu --i-own-the-window'
i=0; for h in $H; do ssh -n $h "python3 ~/dsv41-stage/engram-disk/test_engram_disk.py real --rank $i --i-own-the-window" & i=$((i+1)); done; wait
# 3. NVMe microbench, all ranks in parallel (each its own disk), ~3 min
i=0; for h in $H; do ssh -n $h "mkdir -p ~/dsv41-opt-20261021/engram-disk/nvme && python3 ~/dsv41-stage/engram-disk/nvme_bench.py --i-own-the-window --rank $i --out ~/dsv41-opt-20261021/engram-disk/nvme/nvme_\$(hostname).jsonl" & i=$((i+1)); done; wait
# 4. correctness arm first (gate for everything else)
bash $S/launch-engram-arm.sh verify
ssh spark-one 'cd ~/dsv41-bench; D=~/dsv41-opt-20261021/engram-disk/verify; mkdir -p $D; until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 10; done;
  python3 exact.py $D/exact.json verify; python3 ~/gsm8k_gate.py $D/gsm8k50.jsonl 50 8 | tail -2;
  python3 ~/dsv41-stage/r4/prefill-bench/prefill_bench.py prefill --lens 16k --reps 1 --out $D/pf.jsonl | tail -1;
  grep -a "VERIFY\|engram-disk stats" ~/dsv41-logs/rank0.log | tail -5'
bash $S/collect.sh verify
#    GATE: exact digest == a8c3394af9d2 (control); 0 "VERIFY" error lines on all 8 ranks (collect.sh pulls engram_*.log);
#          and on EVERY rank the last stats line shows verify_reads > 0 AND verify_fwd > 0 (proves the checks ran;
#          forward checks only run on eager steps, i.e. the 16K prefill and warmup runs, never on FULL/PIECEWISE replays).
#    Optional, if the 16K prefill in step 4 is the only eager real step: also run prefill_real.py --lens 16k --reps 1.
# 5. perf arms (each: launch, ~6-8 min to health, run-arm ~28-30 min, collect). Priority order: the first four
#    answer the user's question (price without and with the practical mitigation, bracketed by controls);
#    the rest decompose it. Budget about 35-38 min per arm, so all 7 take about 4.3 h, plus verify, benches and restore: about 5.5 h.
for ARM in control disk0 disk4pf control2 disk4 disk0pf disk8; do
  bash $S/launch-engram-arm.sh $ARM
  ssh spark-one "bash ~/dsv41-stage/engram-disk/run-arm.sh $ARM"
  bash $S/collect.sh $ARM
done
#    (control2 = control config again, brackets restart drift: single spread is ~±4 %)
# 6. restore live
for h in $H; do ssh -n $h 'systemctl --user stop dsv41-rank engram-memwatch' & done; wait
for h in $H; do ssh -n $h 'bash ~/dsv41-stage/engram-disk/install.sh uninstall' & done; wait   # "restored 3/3" x8
bash $S/ops/dsv41-launch-all.sh     # (same file as ~/dsv41-opt-20261021/ops/dsv41-launch-all.sh)
ssh spark-one 'cd ~/dsv41-bench && python3 exact.py /tmp/exact_restore.json restore | tail -1 && python3 ~/gsm8k_gate.py /tmp/g50.jsonl 50 8 | tail -1'
```

Per arm, `run-arm.sh` runs:
- single code/prose/math: cold (first after restart, empty row cache) and warm (×2).
- conc8, conc32, and conc32 again at the end (warm cache).
- `prefill_bench.py prefill --lens 16k,128k --reps 2` (cold prompts with a unique nonce, so the prefix cache never hits; reports TTFT and tok/s).
- exact.py, GSM8K-50 at conc8, longctx 4k-128k gate.
- The rank0 engram-disk log lines.

`memwatch.sh` samples every 2 s on every rank: MemAvailable, Cached, SwapFree, nvme0n1 reads, sectors, read_ms, inflight and NVMe temperature. Startup and under-load MemAvailable come from it, aligned with `progress.txt` stamps. Disk IOPS = Δreads/Δt, and mean device latency = Δread_ms/Δreads.

## 6. Readout and gates

| metric | control | disk0 | disk4 | disk0pf | disk4pf | disk8 |
|---|---|---|---|---|---|---|
| MemAvailable after startup, min over ranks (GB) | | | | | | |
| MemAvailable under conc32, min (GB) | | | | | | |
| single t1 tok/s (headline, 3 reps × 3 types); greedy cold / greedy-repeat (anchors; cache arms inflated) | | | | | | |
| conc8 t1 / conc32 t1 / conc32 t1 steady (headline); conc32 greedy (anchor) | | | | | | |
| prefill_real TTFT 16K / 128K (s) (headline); synthetic prefill_bench (anchor) | | | | | | |
| engram-disk per rank (report ranks 0 and 7 + max): hit rate, uniq/stage, read_ms/stage, stage_ms/stage, **gpu_gap_ms/stage**, max_read_ms, taken as deltas over each workload phase (progress.txt stamps vs log timestamps) | – | | | | | |
| NVMe IOPS / mean latency during conc32 and 128K prefill | | | | | | |
| exact.py digest | a8c3394af9d2 | must equal | | | | |
| GSM8K-50 / longctx | | ≥49/50, same passes | | | | |

Interpretation:
- **The speed price of the freed RAM.** Compare disk vs control on single, conc32 and TTFT, against the ±4 % restart noise bracketed by control and control2.
- **Whether a cache or prefetch buys the cost back.** Compare disk4 vs disk0, and the pf arms vs the arms without pf.
- An exact.py digest that differs from control is a bug, not noise. Stop and rerun the verify arm.

## 7. Risks

1. **Per-step host sync.** Staging must wait for the step's ids, which removes CPU run-ahead under async scheduling. If the runner was hiding more than about 0.3 ms of CPU work behind the GPU, the single-stream cost will exceed the model. `sync_ms/stage` and `host_ms/stage` in the stats line show the split.
2. **Tail latency × 8 ranks.** Every step waits for the slowest of 8 NVMe waves. GC pauses, log writes, or anything else touching a disk show up directly in ITL. Watch `max_read_ms`.
3. **Grace CPU contention.** 32 pool threads plus a prefetch thread run on 20 cores next to the vLLM workers. If conc32 regresses more than the IO time explains, try `DSV41_ENGRAM_DISK_THREADS=16`.
4. **Prefetch heuristic.** The next-chunk start is computed as `num_computed_tokens_np[i] + num_scheduled_tokens[i]`, and those semantics have not been checked live. A mismatch only lowers the hit rate; the stats show prefetch rows vs hits.
5. **Untested on GPU so far.** The stager path ran only through the CPU unit test. The `test gpu` and verify-arm gates must pass before any perf arm. If startup fails, uninstall and relaunch (step 6) to restore live.
6. **Loader.** The no-op loader relies on the safetensors tensor for the 98 GB table being a lazy mmap view. The live path already narrows that same view, so this should hold. Watch startup RSS; an OOM at load means it does not.
7. **spark-one disk is 97 % full** (32 GB free). The experiment only reads the checkpoint and writes small logs, which is fine. Do not copy the tables.
8. **NVMe heat.** Sustained prefill is about 350K IOPS × 4 KiB ≈ 1.4 GB/s per rank; memwatch logs the temperature. Measured power on the feeds should barely change.
9. **Prefetch holds the read pool.** The C pool serializes callers. A decode stage that arrives while the prefetch thread runs a 2048-row sub-batch waits up to about 6-7 ms; this only matters when a long prompt prefills next to live decodes. Lower `DSV41_ENGRAM_PREFETCH_BATCH` (e.g. 512) if conc ITL p99 regresses in the pf arms only.
10. **Hangs.** A pread that never returns, or an OSError on one rank, stalls the other 7 in the next collective until the NCCL/rdma timeout. There is no per-read timeout. If a rank stops logging stats while others wait, check `dmesg` for NVMe errors, then do the step-6 restore.
11. **Disk mode also removes the live per-layer side-stream prefetch** (and its eager breaks in breakable PIECEWISE graphs). This is part of what "disk mode" costs or saves; it is not a separate effect to subtract.
