# g3 rocenante: RoCEnante (flashinfer 0.7.2) vs rdma_ar, staged for the stopped window. Nothing installed into serving

Offline prep, 2026-10-26. No restart, no GPU job, no live edit. CPU work ran on spark-four under `nice -n 19` with `CUDA_VISIBLE_DEVICES` empty.
- **Stage dir:** `~/dsv41-stage/g3/rocenante/` on 8/8 (`SHA256SUMS`).
- **Separate venv:** `~/venvs/roce072` on spark-four only. It holds flashinfer-python 0.7.2 and nvidia-cutlass-dsl 4.7.1 (+ libs), installed `--no-deps`. Torch 2.13.0+cu130, numpy and cuda-bindings come read-only from vllm032 through one `.pth`, so the live `rdma_ar_ext.so` loads in the same process. Manifest digest `1c7a6e9a3254` (10,663 files).
- The serving venv vllm032 (flashinfer 0.7.0.post1, cutlass-dsl 4.8.0) is untouched.

## Verdict so far (offline): do not adopt RoCEnante at TP8

RoCEnante uses the same architecture as our rdma_ar and the same reduction rule, except for one init detail. It is very likely **slower than our rdma_ar at every decode size at 8 ranks**. The stopped-window bench below settles this head to head.

**What the 2x in-situ gap really is.** knapcio's ~26 µs/call against our ~49 µs/call compares TP4 with TP8:
- At TP4 a one-shot sends 3× the payload instead of 7×, with half the QPs, WRs and flags and 3 instead of 7 ranks to wait for.
- Our in-situ figure also carries ~13.6 µs/call of L2-prefetch stretch, plus rank skew.

The transport design is not the cause.

**Evidence: component latency at 8 ranks, µs/op in CUDA graphs:**

| rows x 5120 bf16 | ours, k1 parallel-load (live), K1A bench 2026-10-24 | ours, legacy, A0 2026-10-21 (1-shot / 2-shot) | RoCEnante TP8, rhys101 SG17 component bench (rev 081b2359, incl. one copy/op, max-rank median) |
|---|---|---|---|
| 1 | **18.1** (1-shot) | 19.9 / 30.9 | 38.0 |
| 5-6 | **31.5** (2-shot); 39.0 1-shot | 40.5 / 35.5 | 50.4-50.6 |
| 24 | ~56 | 119.9 / **56.6** | 122.4 |
| 48 | **77.6** (2-shot) | 222.9 / 79.8 | 205.1 |

Sources:
- rhys101: `sglang/results/sg17-rocenante/component/comparison.json`, local copy in the Mac scratchpad `g3roce/dl-rhys/`. In the same harness NCCL measured 135-375 µs.
- K1A: `~/dsv41-stage/k1/3-rdma-loads/results/`.

## Why. Source reading of flashinfer 0.7.2 `experimental/b12x/comm/roce` (b12x b9e450f1) against rdma_ar.cu 1abfbc63

| | RoCEnante 0.7.2 | rdma_ar (live k1pl) | effect at TP8 |
|---|---|---|---|
| Initiation | GPU stages into pinned host memory and rings a host doorbell; a **CPU proxy** thread (C, libibverbs) posts the RDMA writes. Not GPU-initiated. | Same (C++ proxy thread). | none |
| Algorithm | **Always one-shot**: every rank writes its whole input to all 7 peers, i.e. 7× payload egress. | One-shot below 32 KiB, **two-shot** (reduce-scatter + all-gather) at or above it. | One-shot is bandwidth-bound from ~6 rows. Our A0: one-shot 40/68/120/223 µs vs two-shot 36/43/57/80 µs at 6/12/24/48 rows. **This is the main reason RoCEnante loses at TP8; at TP4 the payload is only 3×.** |
| Rails / QPs | Every peer payload is **striped over both HCAs**, with a 4-byte flag per stripe: 14 QPs and 28 WRs per op, every flag signaled, CQ drained after each op. | One HCA per peer pair ((rank+peer) % 2): 7 QPs and 14 WRs, signaled every 32nd. | Striping balances the 4:3 peer split per HCA (~12% less wire time on the busier HCA). In exchange the proxy posts twice as many WRs and polls more CQEs. Net ~0 at 1 row. |
| Doorbell | Level doorbell: per-slot byte counts, and the proxy catches up on missed sequence numbers. **No GPU wait on the proxy.** 2 slots. | The last CTA spins on `posted_seq >= send_seq` before it publishes: 1-2 sys-scope loads on the critical path. 4 slots. | RoCEnante saves ~0.5-1.5 µs/call. **Worth porting** (P1). |
| Consume loop | Rank- and world-specialized CuTe kernel. **SASS** (compiled offline here) issues the 7 peer `LDG.E.128.STRONG.SYS` 3 at a time and then 2 at a time between FADDs: ~3 serial C2C round trips. | k1pl SASS gate: **all 8 loads in flight** (depth 8), then accumulate: 1 round trip. | Ours is faster in the tail (K1A: tail 2.1 µs at 1 row). |
| Grid | Power-of-two grid, ~2 packs/thread, at most 8 CTAs (1 CTA at 1 row). | `ceil(nvec/512)`, clamped to 1-16 (2 CTAs at 1 row). | Measurable with the env-only arm `rdmaV` (`RDMA_AR_VEC_PER_CTA=1024`). |
| Timeout / fail-fast | `spin_limit` polls (~20 s), then a poison word plus host error word. Later launches become no-ops; `check_health()` after a replay raises. | `trap` after `RDMA_AR_TIMEOUT_MS` (600 s in serving), which kills the context; plus the 90 s capture watchdog. | RoCEnante fails faster and more gently, but only if the host checks after every step. |
| Sequence | uint32 epoch on the device. **The poison and error words store `seq`, so a timeout at the wrapped seq 0 goes unseen** (~every 2^32 ops ≈ 11 days at ~4.4K ops/s). sparkring f91b5c8 fixes this with a separate flag; 0.7.2 still has the hole. | uint64 seq, no wrap. | Adopting 0.7.2 would need the sparkring fix. |

**sparkring fixes** (knapcio ports them into `runtime/b12x/b12x/comm/roce_ring`, provenance file read):
1. **2d83bb6, atomic diagnostic counters.** In 0.7.2 the proxy counters are still plain `uint64` read from another thread. This affects diagnostics only.
2. **e4d0921, memory scope.** The sparkring kernel fences at system scope in every staging thread. 0.7.2 relies on `bar.sync` plus one `fence.sc.sys` by thread 0 (PTX cumulativity). That is likely fine, and our rdma_ar already fences in every thread.
3. **f91b5c8, sequence-wrap timeout.** See the table: not fixed in 0.7.2.

## Exactness

The rule is the same as ours: fixed rank order 0..7, an fp32 accumulator, and one round-to-nearest bf16 rounding. The one exception is the **accumulator init**: RoCEnante starts from rank 0's value, we start from +0.0. The outputs therefore differ **only where all 8 inputs are -0.0**: they give -0.0, we give +0.0. NaN, inf, subnormal and cancellation cases are identical.

Offline PTX/SASS check, `tests/ptx_offline.py` (CPU-only CuTe compile for sm_121a, `results/ptx_offline.txt`):
- bf16: **56 `add.f32` = 8 lanes × 7**, so source 0 initialises. **No `.ftz`**, which means subnormals are kept, as in ours.
- 8× `cvt.rn.bf16.f32`, which compiles to the same `F2FP.BF16.F32.PACK_AB` as our `__floats2bfloat162_rn`.
- 7 peer `ld.relaxed.sys.v4` plus the local operand.
- fp32: 28 adds, no FTZ.

**To adopt RoCEnante bitwise**, either patch `_accumulate_words(initialize=True)` to `acc = 0.0f + x0` (third-party code, Apache-2.0), or prove that all-(-0.0) never occurs; the second option is not acceptable for the exactness rule. The GPU bitwise phase X below confirms this per arm.

## How to get the RoCEnante ideas in rdma_ar (exact by construction: our accumulate code is unchanged)

- **P0, env only, measured by this bench.**
  - Phase C re-measures the one-shot/two-shot crossover (`RDMA_AR_TWOSHOT_MIN_BYTES`, live 32768, i.e. two-shot from 4 rows) with the k1pl kernels under prefetch. G1 steps run 3-5 rows, right at the crossover.
  - `rdma` vs `rdmaU`: proxy pinned to an X925 core vs floating (`RDMA_AR_PROXY_CORE`).
  - `rdmaV`: `RDMA_AR_VEC_PER_CTA=1024`.
  - Take a knob to a pinned serving A/B only if it saves ≥2 µs/call in the `gap` and `l2pf` modes.
- **P1, level doorbell (S).** Per-slot descriptors (bytes, mode, chunk) in `Ctrl`. The GPU publishes `send_seq` without waiting on `posted_seq`, and the proxy posts every sequence it missed, in order (at most kSlots-1 behind; refuse beyond that). Saves ~0.5-1.5 µs × ~125 calls/step ≈ 0.1-0.2 ms (0.3-0.6% G1). Wire protocol, slots and results are unchanged. Gates: k1 selftest, bitwise, a SASS diff of the consume loops.
- **P2, fail-fast like RoCEnante, seq-wrap safe (S-M).** A device poison word set to 1 on a timeout, `RDMA_AR_SPIN_LIMIT` in polls (default ~5 s), later launches become no-ops, and `RdmaAllReduce.check_health()` runs after every `execute_model`. This replaces the 600 s trap; the engine raises and the supervisor relaunches.
- **P3, stripe one-shot payloads over both HCAs with a flag per stripe (M).** Only one-shot sizes (≤3 rows) benefit, by <1 µs. Low priority.
- **Not from RoCEnante, but where the remaining per-call time actually is:**
  - L2-prefetch stretch: ~13.6 µs/call, up to 1.1-1.5 ms/step (colltrim (b) arms).
  - The proxy hop: K1A net 11.3 µs at 1 row includes ~0.7 µs GPU→CPU hand-off and ~1-2 µs of posting. GPU-initiated posting (DOCA GPUNetIO / IBGDA-style WQE + UAR doorbell from the kernel) would save ~2-3 µs/call, about 1-1.5% at 1 user. Effort L.
  - The stage copy: ~3 µs/AR, removable if the producer GEMM epilogue writes into the send slot. Effort L.

## Package

| path | what |
|---|---|
| `bench/bench_roce8.py` | 8-rank bench; phases X (bitwise), L (latency), C (crossover). The docstring lists the arms and modes. `--cpu-selftest` for the CPU gate. |
| `tests/check_env_cpu.py` | Import-resolution gate: b12x and cutlass from roce072, torch 2.13.0 from vllm032, no CUDA. |
| `tests/ptx_offline.py` | CPU-only compile of RoCEnante's kernel to PTX/cubin, plus the numerics checks above. |
| `ops/venv-node.sh` | On a node: `install` / `verify` / `uninstall` / `status` / `manifest` of `~/venvs/roce072`. Pinned no-deps uv install under nice/ionice; the proxy `.so` is prebuilt into `cache/roce`. |
| `ops/roce-all.sh` | Mac driver for venv-node.sh on 8/8, in parallel. Requires the same manifest everywhere. |
| `ops/run-bench8.sh` | Mac driver for the 8-rank bench. Refuses unless the service is stopped on 8/8 and the stage sums and venv are present. |
| `ops/stage.sh` | Mac: pushes the package and SHA256SUMS to 8/8. |
| `venv-manifest.sha256` | Spark-four reference manifest of the venv (digest 1c7a6e9a3254). |
| `results/ptx_offline.txt` | Offline PTX gate output. |

**CPU gates run on spark-four (all PASS):**
- `check_env_cpu.py`.
- `bench_roce8.py --cpu-selftest`: deterministic inputs; torch reference == numpy reference; zero-init vs rank-0-init differ exactly on the all-(-0.0) class; subnormal sums nonzero; order-dependent overflow gives +inf; the reversed-order reference differs (so the test can see order).
- `ptx_offline.py`, bf16 and fp32.
- RoCEnante proxy built (`roce_proxy-9725ab7a7a7e14d7.so`, ABI 5).
- Layout at 2 MiB slots: 37.75 MB pinned per runtime.
- pyflakes clean.

**Bench design.**
- **Inputs.** Every rank generates all 8 ranks' inputs from a seed, so the fixed-order reference is local and identical. Planted classes: all -0.0, mixed ±0, subnormals, cancellation, order-dependent overflow, inf/NaN, absorption, and a bf16-visible order probe.
- **Arms.**
  - rdma_ar, from one live `.so` (62daad6c): `rdma` (live policy), `rdma1` (always one-shot), `rdma2` (always two-shot), `rdmaU` (unpinned proxy), `rdmaV` (VEC_PER_CTA 1024), `rdmaT` (stage/net/tail timing).
  - RoCEnante: `roce2` (our two HCAs), `roce2k` (knapcio's f0 + P2f0), `roce1` (1 rail, isolates striping), `roce2U` (unpinned).
- **Shapes.** All-reduce rows 1/6/12/24/48 × 5120 bf16; gather [M,224] dim -1 (qkv_a); and a step pattern of AR, AG, AR.
- **Modes.**
  - `b2b`: back to back.
  - `gap`: a 100 µs GPU spin before each op.
  - `l2pf`: gap plus the **live** `libdsv41_l2pf.so` 6 MB, 2-segment `cp.async.bulk.prefetch.L2` burst forked on a side stream at every op, as in serving.
  - `dram`: gap plus a continuous 8-CTA prefetch reader over 512 MB.
- **Timing.** ABBA × 3 trials. µs/op is computed with the spin and prefetch-only graph subtracted.

## Stopped-window procedure (~25 min of downtime)

Run from the Mac in the package dir (Mac copy: `/private/tmp/claude-501/-Users-jeweled/64b45536-e9c4-4d44-9db6-63d2a8374326/scratchpad/g3roce/pkg`).

0. **Venv on 8/8**, if not done before. This step is safe while serving and takes ~2-4 min (PyPI download ~0.9 GB per node over the LAN/WiFi uplink; no GPU, no fabric):
   - `bash ops/roce-all.sh install`
   - **Gate:** `MANIFEST identical on 8/8:1c7a6e9a3254` and `VERIFY ... same=yes env=[RESULT PASS] ... selftest=[RESULT PASS]` on every node.
1. Stop the service the usual way.
2. **Smoke** (~3-5 min, including the first rank-specialized CuTe compile):
   - `bash ops/run-bench8.sh smoke --phases X --rows 1,6 --arms rdma,rdma2,roce2`
3. **Full run** (~15-20 min): `bash ops/run-bench8.sh g3roce-w1`. This runs phases X, L and C with all arms.
4. **Gates and decisions.**
   - The run must end with `DONE failures per rank: [0, 0, 0, 0, 0, 0, 0, 0]`.
   - Phase X:
     - every rdma arm reports `EXACT` (bf16 and fp32, eager == graph, identical on all ranks);
     - every roce arm reports `EXACT_EXCEPT_ALL_NEG0`: 0 mismatches against the rank-0-init reference, and against ours only the all-(-0.0) elements;
     - cpu-vs-gpu reference 0, reversed-order diffs > 0;
     - all gathers bitwise.
   - **Adopt or port RoCEnante itself only if** `roce2` beats `rdma` by ≥10 µs/call at 1 and 6 rows in the `gap` and `l2pf` modes; this is not expected. Otherwise **close RoCEnante** and keep P1/P2 as rdma_ar work items.
   - **Env-only exact knobs** (C crossover, pinning, VEC_PER_CTA) go to the pinned serving ABAB if they save ≥2 µs/call in gap and l2pf. Promotion rule: fresh-pin exact.py a8, identical G1 digest, ≥0.5% G4min_hat or G1 in every pair, ≥2 pairs.
5. Relaunch with the supervisor. Nothing in serving changed, so no rollback is needed.

## Integration plan, if a RoCEnante path ever wins (or for the P1/P2 port)

- **Flag.**
  - `DSV41_COLL_BACKEND=rdma_ar|rocenante` (default rdma_ar), plus an optional `DSV41_ROCE_MAX_BYTES` size gate, so hybrid routing is decided by size only. Every rank decides from shape and dtype alone, never from pointers (RoCEnante's `should_allreduce` contract).
  - Load a **vendored, sha-pinned copy** of `b12x/comm/roce` from flashinfer 0.7.2 with the init patch and the sparkring wrap fix. Do not use vllm032's flashinfer 0.7.0.post1, and do not upgrade it.
  - Add the backend env and the vendored file hashes to `RDMA_AR_FINGERPRINT`. Any rank difference means a guaranteed sequence desync, so the fingerprint must refuse to start.
- **Graph safety.**
  - Compile and allocate in warm-up. RoCEnante refuses CuTe compiles and scratch allocation under capture, so call `_prepare_resources(dtypes, padded_gather=True)` before vLLM's first capture.
  - Wrap vLLM's graph capture in `runtime.capture()`, as knapcio's `roce_capture` does.
  - All collectives of one runtime must sit on one stream inside a capture; RoCEnante raises otherwise. Aux-stream calls keep going to NCCL, as rdma_ar does.
  - Eager cross-stream calls are ordered by RoCEnante's own event, the equivalent of our `RDMA_AR_XSTREAM=wait`. Keep the rel-2 guard.
  - Each runtime has its own epoch/sequence in device memory, so op order per runtime must be identical on all ranks; the fingerprint enforces this.
- **Fail-fast.**
  - `B12X_ROCE_SPIN_LIMIT` ≈ 5 s of polls.
  - Call `check_health()` at every result boundary, after `execute_model` sync. This is knapcio's `synchronize_and_check_roce`, and it costs two host reads.
  - On poison, raise. The engine dies and the supervisor relaunches. A poisoned runtime turns later launches into no-ops, so output after a timeout is never consumed.
  - Keep the 90 s capture watchdog.
  - For rdma_ar P2: the same poison/health contract, replacing the 600 s `trap`.
- **Memory and CPU.**
  - 37.75 MB pinned per runtime at 2 MiB slots.
  - One more spinning proxy thread. The RoCEnante proxy spins 20 M idle polls before it naps; pin it to an X925 core.

## Rollback / uninstall

- Nothing in serving changes.
- `bash ops/roce-all.sh uninstall` removes `~/venvs/roce072` and the stage caches on 8/8.
- `rm -rf ~/dsv41-stage/g3/rocenante` drops the package.
