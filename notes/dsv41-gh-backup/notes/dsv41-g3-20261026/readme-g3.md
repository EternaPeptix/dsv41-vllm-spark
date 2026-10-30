
## Goal window 3 (2026-10-26 19:35-22:55 UTC): fresh AV pin; codex-adopt bundle + dhtac PROMOTED as a stack (launcher b0cf94d0); RoCEnante, mfront and the rdma_ar env knobs NOT adopted. Goal NOT met (G1 102.3 of 120 tok/s in production; prose session 47.9 of 50, G4min_hat 47.8)

Scope: (0) a fresh AV pin from the live config, (1) a stopped window with every GPU qualification test, (2) pinned ABAB A/Bs at 2000 MHz, (3) the stack and its confirmation, (4) promotion and a production instance. Clocks were never raised: every launch passed `DSV41_GPU_CLOCK_MHZ=2000`, and the SM clock was read on 8/8 before every timed phase (telemetry off). Mac drivers and raw results: scratchpad `g3w/` (`scripts/sw3.sh` stopped window, `g3ab.sh` one pinned instance, `g3suite.sh` + `g3phase.sh` per-phase suite, `g3blk.sh` serial block, `g3line.py`, `g3agg.py`, `promote.sh`, `fin/` final instance; `ab.log`, `ab.jsonl`, `pow.tsv`, `sw.log`, `ab/<label>/`, `runs/<label>/`, `sw/`). On spark-one: `~/dsv41-g3-20261026/runs/`.

### 0. Fresh AV pin (`~/dsv41-stage/g3/pin/av-pin-g3.json`, sha 1c98cf9b, on 8/8)
- The old pin `av-pin-pdfinal.json` predates M32 `all`, FC2C and stepovh: it priced verify at 8-96 rows 7-15% too high (new/old 0.85-0.93; 104-192 rows within 1%; draft within 2.6%), so pinned arms under-verified at 4 users.
- Source: the rank-0 startup profile of gw2-FU-2 (production, launcher 592793fd, unpinned "pin miss", healthy 17:51:44) via `moet/tools/mkpin_k.py make`; shape key identical to the old pin (29 capture sizes up to 192, 32 seqs, 16384 batched tokens, 5 spec tokens, TP 8).
- **Anchor under the new pin: exact.py a8c3394af9d2 with counters 909 / 467 / 2335 / 1373 (acc / drafts / draft tokens / gen) and per-position 348 / 224 / 167 / 101 / 69 ("pin hit")**: the same as gw2-FU-2's own profile, so the pin reproduces its source launch. G1 2-repetition output digest under the pin: **69ad872124100141**. All 12 pinned instances of this window reproduced both, at start and again after all the load (exact.py #2).

### 1. Stopped window (service down 19:50:20 -> first A/B instance healthy 20:09:38)
- **RoCEnante vs rdma_ar, 8 ranks, our fabric** (`g3/rocenante/ops/run-bench8.sh`; venv `~/venvs/roce072` installed on 8/8, manifest 1c7a6e9a3254; source read before running: its RDMA proxy is one C file over libibverbs, compiled with gcc into the stage cache; routes come from sysfs; no other I/O):
  - Exactness (phase X, bf16 and fp32, 1-48 rows, eager and graph): every rdma_ar arm **EXACT** against the fixed-order fp32 reference; every RoCEnante arm **EXACT_EXCEPT_ALL_NEG0** (it starts the accumulator from rank 0, so all-(-0.0) elements give -0.0 instead of +0.0; nothing else differs); gathers bitwise equal to torch.cat. `DONE failures per rank: [0, 0, 0, 0, 0, 0, 0, 0]`.
  - Latency, us per all-reduce (rows x 5120 bf16), ours / RoCEnante (best RoCEnante variant in brackets):

    | rows | gap (decode-like spacing) | l2pf (live L2 prefetch running) | b2b | dram (continuous reader) |
    |---|---|---|---|---|
    | 1 | **17.4** / 19.9 | **30.3** / 32.8 | 18.3 / 19.9 | 58.0 / 64.7 |
    | 6 | **31.6** / 40.9 | **44.2** / 45.7 | 30.5 / 41.3 | 84.1 / 91.7 |
    | 12 | 38.6 / 66.7 | 49.9 / 71.3 | 37.9 / 65.4 | 90.1 / 121.7 |
    | 24 | 53.0 / 107.4 | 60.1 / 117.4 | 53.0 / 107.9 | 106.1 / 180.2 |
    | 48 | 77.4 / 204.1 | 84.0 / 215.3 | 77.0 / 203.0 | 127.1 / 292.8 |

    Gathers (rows x 224) are within 1-3 us either way; the per-layer AR+AG+AR pattern at 24 rows is 51.6 vs 91.1 us (l2pf). RoCEnante is always one-shot, which loses from 6 rows on. Its only edge: l2pf mode at 3-5 rows, 2-3 us faster than both our modes.
  - **Decision: RoCEnante not adopted** (rule: >= 10 us/call faster at 1 and 6 rows in gap and l2pf; it is slower in all four).
  - **rdma_ar env knobs: none adopted** (rule: >= 2 us/call in the same modes). Proxy pinned to an X925 core vs unpinned (live): 17.4 vs 18.4 and 30.3 vs 31.8 us at 1 row, pinned 0.2 / 0.1 us slower at 6 rows. `RDMA_AR_VEC_PER_CTA=1024`: +1.2 / +1.1 us at 1 row, -0.7 / +1.1 at 6. Two-shot crossover (phase C): two-shot wins from 4 rows in gap mode = the live 32768-byte threshold, so `RDMA_AR_TWOSHOT_MIN_BYTES` stays.
  - The venv stays installed on 8/8 for later rdma_ar work (P1 doorbell / P2 fail-fast ports, prep README); `bash g3roce/pkg/ops/roce-all.sh uninstall` removes it. Results: `~/dsv41-stage/g3/rocenante/results/g3roce-{smoke,w1}.*` (spark-one), Mac `g3w/sw/roce/`.
- **dhtac** (`g3/dhtac/ops/run-gpu-tests.sh`): unit **PASS 8/8**. For every row count 1-160, all six CUTLASS SM120 MXFP8 tactics and the stock call give **bitwise-identical** draft-head outputs on every rank (one class, stable). Per-call times (8 ranks): tactic 0 is 993-1154 us at 5-25 rows and 372-396 us from 30 rows on; tactics 1/3 are 363-398 us everywhere.
  - **The test's own `recommend` line is wrong:** the rank logs had been rotated by the stop, so it fell back to the newest-mtime autotune file (0db863ad, bucket 32 -> tactic 3). Serving loads **8c8be689** (rank-0 log: "Using FlashInfer autotune cache file"), where buckets 32/64/128 -> tactic 0 on all 8 ranks. So at 17-29 rows (4 and 5 requests: 20 and 25 head rows) serving runs tactic 0 at 1046-1072 us instead of ~380 us. Map used: **`17-29:1`** (tactic 0 is already fastest from 30 rows).
- **mfront** (`g3/mfront/ops/run-gpu-tests.sh`): unit PASS 8/8 (1500 MoE cases + 4500 graph replays per rank, 40 layers, 0 failures), but the bench saved only **6.5-7.4 us/layer** (0.26-0.30 ms/step; gate 8 us/layer at c4rec M=12), and it then hit **CUDA illegal memory access** on ranks 0 and 7. Repro on spark-one (patched copy `~/dsv41-g3-20261026/repro/`): live and fused graphs captured and replayed alternately in one process crash at the M=48 set after M=24 passed; live-only and fused-only pass. Serving mixes the two paths in one process (fused for M*6 <= 1024, live for larger decode graphs and prefill), so **mfront was not installed and not served**. Next: bisect with `CUDA_LAUNCH_BLOCKING=1` + compute-sanitizer and `DSV41_MOE_FC2C=0`.
- **codex-adopt** (`run-tests.sh bench`, spark-two): canon / remap / t2s GPU tests **PASS**, no compile after warm-up, serving caches unchanged. Fused CANON 25.6-26.3 -> 3.84 us per call (-174..-180 us per decode step, 8 calls); remap reuse -34.8..-38.9 us per step; omit token_to_seq: 896.8 -> 0.8 MiB held at a late-1M step, metadata build 72.7 -> 15.8 ms.
- **Codex kits** (`run-kits.sh gpu`, spark-three): fused-topk full run **PASS** (93 s) and compute-sanitizer memcheck **PASS** (0 errors). candidate-only-indexer **FAIL / inconclusive** at its first case: "baseline top-k set unstable" on the all-zeros pattern (stock top-k of tied scores is not unique), so the kit cannot qualify; it is a prefill lever and edits the same file as #1, so it was not pursued.
- **Installs** (service stopped, flags default off): codex-adopt (`g3ca-all.sh install`: sparse_attn_indexer.py 7112c01d -> **481905c2**, flashinfer_sparse.py c8a2da55 -> **b059bdf3**, mla/indexer.py 4753707e -> **d9f4c5c9**; backups `.bak-g3-codex-adopt`; digest 9182ac06d964 on 8/8) and dhtac (`dsv41_r3_dense.py` 993023f2 -> **1f1c0456**, `.bak-g3-dhtac`, digest 95faa4a7255a on 8/8). mfront left at baseline (deep_gemm_utils.py a4a0486a). The supervisor site manifest does not cover these files (still 0501bdcb13e0).
- Stopped-window end state: 8/8 inactive, 0 GPU apps, no leftover test processes.

### 2. Method (pinned ABAB at 2000 MHz, fresh pin)
- **Launches.** 12 supervised launches through `g3blk.sh` / `g3ab.sh`, each with `DSV41_AV_COST_JSON=~/dsv41-stage/g3/pin/av-pin-g3.json` ("pin hit"), `DSV41_GPU_CLOCK_MHZ=2000`, `--expect-launcher 592793fd` and the arm's flags as env; all healthy at the first try (+279..300 s).
- **Order:** C1 V B1 D1 C2 D2 B2 C3, then X1 C4 X2 C5. Arms:
  - C: control (codex-adopt and dhtac files installed, flags off).
  - V: untimed verification, `DSV41_DECODE_REMAP_REUSE=check DSV41_TOPK_CANON_FUSED=1 DSV41_IDX_OMIT_T2S=1 DSV41_DRAFT_HEAD_TACTIC=audit DSV41_DRAFT_HEAD_TACTIC_MAP=17-29:1`.
  - B: codex-adopt bundle, `DSV41_TOPK_CANON_FUSED=1 DSV41_DECODE_REMAP_REUSE=1 DSV41_IDX_OMIT_T2S=1`.
  - D: dhtac, `DSV41_DRAFT_HEAD_TACTIC=1 DSV41_DRAFT_HEAD_TACTIC_MAP=17-29:1`.
  - X: B + D.
- **Per instance:** exact.py first (`gstart.sh`), then the timed suite with **telemetry off**, phase by phase from the Mac with one `nvidia-smi` read on 8/8 before each phase (`g3suite.sh`): matched C1 pdab x5, G1 x2, G4 warm-up + 6, conc8, conc16, conc8, conc16, TTFT ~24K / ~198K, exact.py #2. A 2 s `/metrics` sampler flags foreign requests. C and X instances then ran the power phase (2 s telemetry on 8/8; G1 x1, G4 x2; watts only).
- **Pairs:** each candidate against the mean of its bracketing controls. G4min_hat as in goal window 2 (511 / (148 T4 + 31.2 T3 + 30.6 T2 + 68.5 T1), the instance's per-phase step times).
- **Clocks:** 124 phase-start reads on 8/8 (12 instances), all OK at 1969-1989 MHz with event reasons 0x0; the 2 s telemetry of the power phases peaked at 1995 MHz once (C3, G1) and 68 C. No clock or power setting was changed.
- **Foreign traffic.** Since 20:16-20:21 UTC the API is also exposed on spark-one's LAN and tailnet addresses (`dsv41-lanproxy.service`, section "LAN API exposure"), and the DSH sPTC workflow (wf_cb6fe8d8-7fd) opened a 512S2 tunnel to 127.0.0.1:8000 at 20:23 (the workflow was killed at 20:24:19). Foreign requests hit V's G4 phase (up to 8 running; the access log shows browser-style probes `GET /`, `/favicon.ico`, `/models`) and probably C1's power phase (G4 aggregate 140 tok/s; excluded from the power means), and four short ones landed in D1's G4 warm-up (excluded from the metrics; the 6 timed runs were clean by the success counter). Every other timed phase of every instance was clean (sampler and success-counter checks).

### 3. Results (12 pinned instances, 2000 MHz, telemetry off)
**Exactness.** All 12 instances, V included, gave exact.py **a8c3394af9d2** with the pinned counters 909 / 467 / 2335 / 1373 and per-position 348 / 224 / 167 / 101 / 69 at start and again after all the load (exact.py #2), and the same 2-repetition G1 output digest **69ad872124100141**; 0 ERROR/Traceback on 8/8. V (fused CANON + omit token_to_seq on, remap in check mode, dhtac in audit mode) produced the control digest with 0 remap-check errors. The dhtac audit reporter crashed on 8/8 (`RuntimeError: Inplace update to inference tensor outside InferenceMode`: its pinned host buffer was made under inference mode; a diagnostics-only bug, the served path is unaffected), so there is no live 4-user audit count; the evidence for D's exactness at 4 users is the GPU test (all six tactics and stock bitwise-identical at every row count 1-160 on 8 ranks) plus the startup self-check of every D/X launch ("13 of 13 mapped row counts pinned (17-29:1) ... all tactics + stock bitwise" on 8/8). Fix for next time: wrap the reporter's `copy_` in `torch.inference_mode()`.

Log lines present on 8/8: B `DSV41_TOPK_CANON_FUSED=1: fused width-512 CANON kernel registered for JIT warmup` and `... fused CANON kernel in use (first call: rows=1030 row_stride=512)`; D `DSV41_DRAFT_HEAD_TACTIC=1: 13 of 13 mapped row counts pinned (17-29:1)`.

Per instance, in launch order (V is untimed: its G4 phase had foreign requests and verify-mode overhead):

| # | instance | arm | exact.py start / #2 | G1 digest | C1 coding / prose ms | G1 (code / prose / math) | G4 sess. mean / prose MIN / hat | 4-session step ms | G4 agg | conc8 / conc16 | TTFT 198K s | MemAvail min GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | g3-C1 | C | a8 OK / a8 OK | 69ad872124100141 | 30.84 / 26.66 | 100.53 (95.5 / 74.4 / 131.6) | 59.34 / 47.36 / 47.25 | 46.34 | 176.9 | 262.6 / 399.3 | 33.55 | 9.2 |
| 2 | g3-V | V | a8 OK / a8 OK | 69ad872124100141 | - / - | 98.69 (94.0 / 72.9 / 129.2) | 47.84 / 37.86 / 38.01 | 55.92 | 145.6 | 251.8 / 395.5 | 33.47 | 9.2 |
| 3 | g3-B1 | B | a8 OK / a8 OK | 69ad872124100141 | 30.52 / 26.35 | 101.74 (96.7 / 75.4 / 133.1) | 60.26 / 46.44 / 47.91 | 45.82 | 178.3 | 263.4 / 399.8 | 33.53 | 9.3 |
| 4 | g3-D1 | D | a8 OK / a8 OK | 69ad872124100141 | 30.87 / 26.79 | 100.45 (95.7 / 74.1 / 131.5) | 60.68 / 47.16 / 47.84 | 45.64 | 177.0 | 260.6 / 400.7 | 33.42 | 9.1 |
| 5 | g3-C2 | C | a8 OK / a8 OK | 69ad872124100141 | 30.88 / 26.71 | 100.27 (95.4 / 74.3 / 131.2) | 60.44 / 47.14 / 47.31 | 46.24 | 177.1 | 258.4 / 396.8 | 33.53 | 9.3 |
| 6 | g3-D2 | D | a8 OK / a8 OK | 69ad872124100141 | 30.85 / 26.68 | 100.58 (95.8 / 74.3 / 131.6) | 61.33 / 47.33 / 47.94 | 45.58 | 178.8 | 261.3 / 399.2 | 33.54 | 9.2 |
| 7 | g3-B2 | B | a8 OK / a8 OK | 69ad872124100141 | 30.57 / 26.40 | 101.69 (96.8 / 75.3 / 133.0) | 60.38 / 46.93 / 47.54 | 46.11 | 177.2 | 260.9 / 392.7 | 33.41 | 9.2 |
| 8 | g3-C3 | C | a8 OK / a8 OK | 69ad872124100141 | 30.81 / 26.67 | 100.55 (95.7 / 74.4 / 131.6) | 60.55 / 47.09 / 47.41 | 46.46 | 177.5 | 260.3 / 402.2 | 33.48 | 9.3 |
| 9 | g3-X1 | X | a8 OK / a8 OK | 69ad872124100141 | 30.60 / 26.43 | 101.53 (96.4 / 75.2 / 132.9) | 60.98 / 47.21 / 48.48 | 45.07 | 180.9 | 264.3 / 402.4 | 33.59 | 9.3 |
| 10 | g3-C4 | C | a8 OK / a8 OK | 69ad872124100141 | 30.83 / 26.71 | 100.55 (95.8 / 74.3 / 131.5) | 60.02 / 46.13 / 47.43 | 46.58 | 176.0 | 262.4 / 400.3 | 33.53 | 9.3 |
| 11 | g3-X2 | X | a8 OK / a8 OK | 69ad872124100141 | 30.55 / 26.38 | 101.61 (96.6 / 75.2 / 133.0) | 61.30 / 47.32 / 48.08 | 45.61 | 179.1 | 265.9 / 395.1 | 33.31 | 9.1 |
| 12 | g3-C5 | C | a8 OK / a8 OK | 69ad872124100141 | 30.90 / 26.72 | 100.12 (95.5 / 74.3 / 130.6) | 59.91 / 46.60 / 47.11 | 46.30 | 175.0 | 260.6 / 395.1 | 33.52 | 9.1 |

**Control (C, n=5):** G1 100.403 (sd 0.197), G1 code 95.575 (sd 0.178), G1 prose 74.321 (sd 0.073), G1 math 131.314 (sd 0.410), G4 MIN 46.862 (sd 0.493), G4 hat 47.300 (sd 0.129), G4 mean 60.055 (sd 0.481), cw min 40.027 (sd 0.587), G4 agg 176.494 (sd 0.999), conc8 260.896 (sd 1.716), conc16 398.726 (sd 2.818), C1 code 30.852 (sd 0.037), C1 prose 26.694 (sd 0.027), TTFT 198K 33.521 (sd 0.028)

Pairs (+ = better, in %; each instance against the mean of its bracketing controls; C1 and TTFT as speed-ups):

| inst | controls | G1 | G1 code | G1 prose | G1 math | G4 MIN | G4 hat | G4 mean | cw min | G4 agg | conc8 | conc16 | C1 code | C1 prose | TTFT 198K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | C1+C2 | +1.34 | +1.35 | +1.36 | +1.31 | -1.71 | +1.33 | +0.61 | -1.53 | +0.77 | +1.08 | +0.44 | +1.11 | +1.27 | +0.03 |
| D1 | C1+C2 | +0.06 | +0.31 | -0.31 | +0.08 | -0.19 | +1.18 | +1.30 | +0.44 | +0.02 | +0.00 | +0.68 | -0.03 | -0.39 | +0.36 |
| D2 | C2+C3 | +0.17 | +0.25 | +0.05 | +0.17 | +0.45 | +1.24 | +1.38 | +1.52 | +0.83 | +0.73 | -0.06 | -0.02 | +0.04 | -0.10 |
| B2 | C2+C3 | +1.27 | +1.31 | +1.30 | +1.24 | -0.38 | +0.38 | -0.19 | +0.64 | -0.06 | +0.57 | -1.70 | +0.90 | +1.10 | +0.26 |
| X1 | C3+C4 | +0.98 | +0.71 | +1.20 | +1.04 | +1.29 | +2.24 | +1.15 | +2.98 | +2.33 | +1.12 | +0.28 | +0.72 | +0.98 | -0.26 |
| X2 | C4+C5 | +1.28 | +1.05 | +1.27 | +1.45 | +2.07 | +1.71 | +2.22 | +3.80 | +2.06 | +1.69 | -0.65 | +1.03 | +1.27 | +0.64 |

**Per arm (means):**

| arm | n | G1 (code / prose / math) | G4 sess. mean / prose MIN / hat | 4-session step ms | G4 agg | conc8 / conc16 | C1 ms |
|---|---|---|---|---|---|---|---|
| C | 5 | 100.40 (95.58 / 74.32 / 131.31) | 60.06 / 46.86 / 47.30 | 46.38 | 176.5 | 260.9 / 398.7 | 30.85 / 26.69 |
| B | 2 | 101.72 (96.77 / 75.31 / 133.07) | 60.32 / 46.68 / 47.72 | 45.96 | 177.8 | 262.1 / 396.2 | 30.55 / 26.38 |
| D | 2 | 100.52 (95.77 / 74.23 / 131.56) | 61.00 / 47.24 / 47.89 | 45.61 | 177.9 | 260.9 / 400.0 | 30.86 / 26.74 |
| **X = B + D** | 2 | **101.57** (96.53 / 75.23 / 132.96) | **61.14 / 47.27 / 48.28** | **45.34** | **180.0** | 265.1 / 398.7 | 30.58 / 26.41 |

### 4. Decisions (rule: exact, and G1 or G4min_hat >= +0.5% in every pinned pair, >= 2 pairs)
- **B, codex-adopt bundle: PROMOTED.** G1 **+1.34 / +1.27%** (all three G1 prompts, matched C1 -1.0 / -1.2%), G4min_hat +1.33 / +0.38%. Larger than the +0.6-0.9% projection: the fused CANON saves ~0.18 ms and the remap reuse ~0.04 ms per step in the bench, and C1 coding fell 0.31 ms. The B2 conc16 dip (-1.7%) is acceptance, not step time (AL 2.39-2.41 against 2.43-2.47; tok/s per AL inside the control range).
- **D, dhtac: PROMOTED.** G4min_hat **+1.18 / +1.24%**, 4-session step -0.66 ms (46.38 -> 45.61 ms, matching the bench's 0.68 ms per 20-row head call), G4 mean +1.3 / +1.4%; G1 +0.06 / +0.17% as expected (one user drafts 5 head rows, outside the map).
- **X = B + D: PROMOTED.** G1 **+0.98 / +1.28%**, G4min_hat **+2.24 / +1.71%**, raw prose MIN +1.29 / +2.07%, G4 mean +1.2 / +2.2%, aggregate +2.3 / +2.1%, conc8 +1.1 / +1.7%, conc16 +0.3 / -0.7%, C1 coding -0.7 / -1.0%. The gains add: B's hat +0.89% and D's +1.24% against X's +2.07%.
- **mfront: NOT served** (section 1: illegal address with live and fused paths in one process; bench 6.5-7.4 us/layer, below its 8 us gate).
- **RoCEnante and the rdma_ar knobs: NOT adopted** (section 1).
- **candidate-only indexer: not qualified** (kit inconclusive on tied scores; prefill lever).

### 5. MoE and drafter at 4-user row counts (levers 3 and 4)
- **MoE (M=12-24 rows).** Today's M32 `all` + FC2C costs **409.1 / 654.6 us per routed layer at the recorded 4-user batches (c4rec M=12 / 24)**; this window's mfront bench re-measured the live path at 414.0 / 414.7 us (M=12, ranks 0 / 7). Against the same bytes: Marlin W4A16 in the same padded layout 406.6 / 658.5 us (equal), Marlin unpadded (2.44 MB/expert, our bytes) 375.7 / 606.8 us (**7-8% faster: 33 / 48 us per layer, about 1.2 / 1.7 ms per 4-user step**), weights-only streaming at 238 GB/s 352 / 580 us (11-14%). FC1 already streams at ~94%; the exact headroom is the glue kernels (~14-22 us/layer, of which mfront's fusion recovers ~7), the FC1 -> SiLU -> FC2 gap, and FC2C's lower streaming rate. (Marlin and streaming figures from the pd Marlin bench and the g1/g2 benches on recorded routes, normalized through their common `live` arm; prep analysis `g3/moe-drafter-bs4/README.md`.)
- **Drafter (bs2-4).** With 3 requests the drafter replays its own bs3 graph (the "bs3 runs the bs4 graph" premise was wrong; only the AV cost table prices 3 requests at the bs4 draft cost, and re-pricing it would change verify widths, so it is not bitwise). The real bs2 -> bs4 jump was the draft head: the serving autotune cache picks tactic 0 at 17-32 rows, 1.05 ms instead of 0.38 ms at 20 rows. D removes it (-0.66 ms per 4-request step, measured). Not built: drafter MoE on the M32 path (~-0.45 ms at bs4), the same tactic pin for the DSpark main_proj split.

### 6. Power and efficiency (power phase after the timed suite of every C and X instance; 2 s telemetry on 8/8, G1 x1 and G4 x2)
| arm | n | fleet GPU W idle / G1 / G4 | spark-one sys W G1 / G4 | J/token G1 (single) | J/token G4 (aggregate) |
|---|---|---|---|---|---|
| C | 5 (G4: 4) | 72.0 / 155.5 / 187.0 | 90.4 / 101.0 | 1.561 | 1.064 |
| **X = B + D** | 2 | 72.0 / 157.7 / 190.8 | 91.0 / 100.9 | 1.565 | 1.073 |

- C1's G4 power row is excluded (foreign requests, aggregate 140 tok/s). Control spread: J/token G1 1.524-1.621, G4 1.053-1.085.
- The stack changes no clock or power setting. Fleet GPU power is +2.2 W (G1) and +3.8 W (G4) over 8 nodes (+0.3 / +0.5 W per node) for +1.2% single-user and +2.0% aggregate throughput, so energy per token is flat inside the control spread. spark-one's metered whole-node power is unchanged (91.0 vs 90.4 W G1, 100.9 vs 101.0 W G4).

### 7. Promotion and final production instance
**Launcher** `~/dsv41-tp8.sh` 592793fd -> **b0cf94d0** on 8/8 (`g3w/scripts/promote.sh`); backup `~/dsv41-tp8.sh.bak-g3-xstack` = 592793fd on 8/8. Five lines after `DSV41_SYNC_FREE`, each commented:
```sh
export DSV41_TOPK_CANON_FUSED=${DSV41_TOPK_CANON_FUSED:-1}
export DSV41_DECODE_REMAP_REUSE=${DSV41_DECODE_REMAP_REUSE:-1}
export DSV41_IDX_OMIT_T2S=${DSV41_IDX_OMIT_T2S:-1}
export DSV41_DRAFT_HEAD_TACTIC=${DSV41_DRAFT_HEAD_TACTIC:-1}
export DSV41_DRAFT_HEAD_TACTIC_MAP=${DSV41_DRAFT_HEAD_TACTIC_MAP:-17-29:1}
```
The supervisor site manifest includes the launcher, so it moved 0501bdcb13e0 -> **b1150ecfcf49**. The supervisor itself is unchanged (7bea1cf3, clock default 2000, compaction 0). Dependencies: the installed site files of section 1 (codex-adopt and dhtac); `DSV41_TOPK_CANON_FUSED` needs `DSV41_TOPK_CANON=1` (launcher default).

**Final instance g3-FINAL** (production: launcher defaults, AV unpinned = startup profile, `DSV41_GPU_CLOCK_MHZ=2000`; driver `g3w/fin/scripts/fin.sh g3-FINAL prod 2`):
- Healthy at 22:36:32 UTC (+357 s, first try). Env on 8/8 carries the five new flags (verified by `verify8.sh post`).
- exact.py first: **a8c3394af9d2** (own profile, 921 / 455 / 2275 / 1373, per-position 342 / 234 / 175 / 100 / 70, "pin miss"); identical after all the load (exact.py #2).
- **Cold solo 1M: 183.07 s, good class** (ratio 1.16, first try). MemAvailable minimum during the prompt **8.32 GB (spark-one)**, 8.9-9.5 GB elsewhere (6 GB floor; gw2-FU-2 had 8.29 GB).
- Suite (`finsuite.sh`, telemetry off, 17 clock reads on 8/8 at 1976-1989 MHz, no foreign traffic):

  | metric | g3-FINAL (production, b0cf94d0) | ref: gw2-FU-2 (production, 592793fd) |
  |---|---|---|
  | G1 mean (code / prose / math), 3 reps | **102.26** (95.93 / 77.04 / 133.82); reps 102.19 / 102.35 / 102.25, identical outputs (digest eb6d32aab70a0191) | 100.39 (95.63 / 74.16 / 131.38) |
  | G1 AL code / prose / math | 2.616 / 1.889 / 3.658 | 2.603 / 1.792 / 3.607 |
  | G4 per-session mean / prose MIN (3 runs) | **60.86 / 47.87** (prose 48.70 / 45.80 / 49.10) | 60.34 / 46.42 |
  | G4min_hat (step times only) | **47.77** (4-session step 45.59 ms) | 47.57 (46.35 ms) |
  | G4 by session: code / prose / math / mixed | 55.2 / 47.9 / 76.0 / 64.4 | 55.5 / 46.4 / 75.0 / 64.4 |
  | G4 common window / aggregate | 58.92 / 40.62; 180.8 | 57.89 / 39.93; 176.9 |
  | matched C1 coding / prose ms (AL) | 30.59 / 26.58 (5.224 / 2.759); 2nd pass 30.52 / 26.64 | 30.91 / 26.90 |
  | bench conc 4 / 8 / 16 / 32 | **180.9 / 262.8 / 394.2 / 585.8** | 176.3 / 263.1 / 399.7 / 577.6 |
  | TTFT ~24K / ~198K | 4.04 s (warm, after the 1M) / 33.31 s | 4.09 / 33.30 s |
  | GSM8K-50 / full longctx / cached_tokens | **49/50** / **PASS** (needle 60/60, copy 1.000) / 4160 | 50/50 / PASS / 4160 |

  0 ERROR/Traceback on 8/8, 0 restarts; MemAvailable at the end 8.4-9.4 GB.
- Production numbers move with the startup AV profile (this launch: prose AL 1.889 vs 1.79-1.85 in earlier production launches), so the production-vs-production row is indicative; the pinned A/B above is the measurement (stack: G1 +1.2%, G4min_hat +2.1%).

### 8. Live state and rollback
**Live:** instance **g3-FINAL**, healthy since 22:36:32 UTC; launcher **b0cf94d0** on 8/8; supervisor 7bea1cf3; AV unpinned (the default pin stays a user decision; the fresh pin for A/Bs is `~/dsv41-stage/g3/pin/av-pin-g3.json`); clock lock 0-2000 MHz (boot unit active on 8/8). Site files on 8/8: sparse_attn_indexer.py 481905c2, flashinfer_sparse.py b059bdf3, mla/indexer.py d9f4c5c9, dsv41_r3_dense.py 1f1c0456 (in use); deep_gemm_utils.py a4a0486a (mfront not installed); everything from goal window 2 unchanged. No telemetry, sampler, logger or bench process left on 8/8. RoCEnante venv `~/venvs/roce072` remains on 8/8 (unused by serving).

Rollback:
- **One launch:** `DSV41_TOPK_CANON_FUSED=0 DSV41_DECODE_REMAP_REUSE=0 DSV41_IDX_OMIT_T2S=0` (codex bundle off) and/or `DSV41_DRAFT_HEAD_TACTIC=0` (dhtac off) as supervisor arguments. Exact either way.
- **Permanent:** `cp -p ~/dsv41-tp8.sh.bak-g3-xstack ~/dsv41-tp8.sh` on 8/8 (592793fd), relaunch with `--expect-launcher 592793fd`.
- **Site files** (service stopped, after the launcher rollback): `ssh spark-one cat ~/dsv41-stage/g3/codex-adopt/ops/g3ca-all.sh | bash -s uninstall` and `ssh spark-one cat ~/dsv41-stage/g3/dhtac/ops/all.sh | bash -s uninstall` (backups `.bak-g3-codex-adopt`, `.bak-g3-dhtac`).
- **RoCEnante venv:** `bash g3roce/pkg/ops/roce-all.sh uninstall` (Mac) removes it and its caches on 8/8.

### 9. Goal status and where the gap is (2000 MHz, no clock or power increase)
- **G1 >= 120 tok/s: NOT met.** 102.26 production (pinned stack 101.57): **+17.3%** still needed. The window's exact levers gave +1.2% pinned. 120 needs an acceptance lever (prose AL ~1.8-1.9) or MoE weight streaming much closer to the bound (FC1 already streams at ~94%).
- **G4 >= 50 tok/s on every session: NOT met.** Code / math / mixed stay at 52.7-78.2 in every run; prose is the binding session: production MIN 47.87 (worst run 45.80), pinned stack raw MIN 47.27, **G4min_hat 47.8-48.3**, so **+3.6..4.7%** (step time) and up to +9.2% for the worst prose run are still needed.
- Next exact levers, by size at 4 users: mfront after root-causing the mixed-path crash (~-0.27 ms/step bench), FC1 -> FC2 per-expert pipelining (-0.4..-1.0 ms), contiguous FC2C layout (-0.5..-0.8 ms), drafter MoE on the M32 path (~-0.45 ms), an L2-prefetch plan that keeps q_b above 8 rows (the replicated qkv_a fills the 6 MB window; +0.86 ms today), rdma_ar P1 doorbell (~0.1-0.2 ms). Together roughly 2.5-3.5 ms of the 45.3 ms 4-session step; the G4 target needs most of them to land.
- **Coordination:** the server is now reachable from the LAN and the tailnet without authentication (lanproxy), from 512S2 (the DSH tunnel, still open and idle) and from the Mac (the Codex app's `codex-dsv41-harness.sock` tunnel). Timed windows need these clients quiet: pause the Codex app and other users, or stop `dsv41-lanproxy.service` for the window (user decision), and keep the per-phase foreign-request check.
