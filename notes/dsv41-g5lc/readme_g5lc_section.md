
## Goal window 5, LongCtx step (2026-10-28 03:29-04:50 UTC): Codex ops kit 07 (reuse the indexer top-k across decode steps) CLOSED at its own gate, NOT promoted. On 1,651 recorded real decode steps (128K and 512K) the top-512 set changes at every step, so no sound certificate can ever allow a reuse: 0 passes. No serving code built, no A/B. Launcher f214c2c9 unchanged; live g5lc-PU

**Result.**
- **The kit's gate** (its README: "log the margin and the pass count on a 512K decode; if the pass count is near zero, the kit stops"): **0 passes** out of 109,912 recorded (layer, row) pairs, at 128K and 512K, for every index layer.
- **Why no certificate can pass.** A reused set is exact only if the new top-512 lies inside the old set plus the entries that became visible since. That held in **0 of 109,912** pairs, including adjacent positions inside one verify step. Consecutive rows share only about half of their picks (mean overlap 0.50-0.58); a reuse would serve 206-251 wrong entries of 512 (median).
- **Why this certificate in particular fails.** Its bound D on how far any score can move between two query tokens is 79-94x |v_k| (median, rigorous K bound from the checkpoint's k_norm weights), or 36-43x with an optimistic key norm. The boundary gap it must beat (512th score minus the best outside score) is about 0.05% of |v_k|.
- **Ceiling, had it worked.** On this stack a 512K decode step costs **+2.30 ms** over a 1K prefix (30.2 vs 27.9 ms on code, same AV pin). That is the whole long-context indexer + top-k excess, so kit 07 could never have moved G1 / G4, which run short contexts.
- **Nothing changed in production.** No site file and no launcher line was touched. The recording used a diagnostic worker extension outside the site tree; it was output-neutral (exact.py with the pin's counters; 6/6 recorded continuations identical to the recorder-off reruns). Production was relaunched on launcher defaults (exact a8c3394af9d2).

### 1. Method
- **Recorder** (`~/dsv41-g5lc/py/g5lc_rec.py` 2faeffe949d0, staged on 8/8, DIAGNOSTIC ONLY, never in the live launcher). Loaded only with `PYTHONPATH=~/dsv41-g5lc/py G5LC_REC=1 DSV41_EXTRA=--worker-extension-cls=g5lc_rec.RecExt`.
  - It wraps `dsv41_topk_ties.decode_hook`, which the decode indexer calls inside the FULL graphs after the stock top-k and the live tie fix. It calls the original hook first.
  - On TP rank 0 only, two Triton kernels captured into the decode graphs READ the logits, final top-k, fp8 index queries and head weights. They WRITE only a 198 MiB ring that is allocated before capture (512 steps x 8 layers x 8 rows).
  - Per row it stores: the 512 ids; v_k (512th score); U, the best score outside the set (exact, ties included); counts > / >= v_k; the row end; counts >= the previous step's v_k (and -2/-5/-10%); the row max; q (32x128 fp8); w (32 fp32).
  - A device flag gates both kernels. A control file (`/dev/shm/g5lc-ctl.json`, polled at the start of `execute_model`) resets, enables or dumps the ring.
  - The kernels were checked in the Triton interpreter on CPU (`test_rec_cpu.py`): every field is bitwise equal to a numpy reference, including ties, -inf masks, rows < MAXR, ring wrap and the disabled flag.
- **Instance g5lc-R1.** Launcher f214c2c9 + AV pin `av-pin-k5-g3FU.json` + the recorder; healthy +131 s.
  - exact.py **a8c3394af9d2** with the pin's counters, 909/467/2335/1373 and per-position 348/233/171/90/67 (EXACT-OK); the pin hit.
- **Probe** (`~/dsv41-g5lc/scripts/g5lc_probe.py`). Setup:
  - One cold prefix prefill per case.
  - Then 3 greedy 512-token continuations on the cached prefix (chat, thinking off, `ignore_eos`).
  - Cases:
    - `code:512000` and `code:131072`: prefixes cut from a fixed 3.2 MB snapshot of 211 vLLM `.py` files, sha c2715b6aa1b779dc, exact 512,000 / 131,072 tokens;
    - `words:512000`: the X-ray decprobe's random words and questions.
  - Foreign-traffic guard. Before every request the probe waits for 30 s of idle. A 0.5 s /metrics sampler runs during each request. If a request overlapped foreign traffic, it was rerun and its result discarded.
- **Analysis** (`g5lc_an.py`, CPU only). Pairs of decode rows:
  - **A** = same row, consecutive steps;
  - **B** = new step's row 0 vs the previous step's row at the previous position (the last committed token);
  - **C** = adjacent positions inside one verify step.
- **Rigorous D.** Per head, `K x min(|a_n||Δb| + |Δa||b_p|, |a_p||Δb| + |Δa||b_n|)`, minimised over the free per-head rescaling (a, b) -> (a/c, c·b), c = 2^-8..2^8.
  - K = sqrt(128) x max|k_norm.weight| x 1.0625: 14.09 / 19.53 / 22.35 for layers 2 / 8 / 14, and 13.15 for layer 20, whose keys layers 24-36 share.

### 2. Recorded decode steps (all clean of foreign traffic)

| case | continuations (decode steps) | verify rows per step | ms/step, recorder off | ms/step, 1K-prefix control |
|---|---|---|---|---|
| code 512K (512,011-token prefix, cold prefill 82.1 s) | 169 / 185 / 140 | 2-6 | 30.21 / 29.76 / 30.50 (AL 3.07 / 2.78 / 3.68) | 27.85 / 27.39 / 28.61 (AL 2.84 / 2.71 / 3.35) |
| code 128K (131,083) | 202 / 179 / 134 | 2-6 | - (recorder on: 28.55 / 28.73 / 30.19) | - |
| words 512K (512,554) | 288 / 189 / 165 | 2-6 | 27.12 / 29.65 / 30.15 (AL 1.79 / 2.73 / 3.12) | 24.40 / 27.52 / 27.82 (AL 1.73 / 2.80 / 2.97) |

- **512K excess**, same question: +2.36 / +2.37 / +1.89 (code) and +2.72 / +2.13 / +2.33 (words). Mean **+2.30 ms/step (+8.4%)**, in line with the X-ray step's +2.1-2.2 ms.
- **Recorder on** costs ~1.5 ms/step at 512K (rank-0 scans).
- **Recorder on vs off**: the token output is identical in 6/6 clean pairs (text sha, steps and AL all equal).

### 3. Kit 07 on real queries

| pairing | pairs | reuse could be exact (oracle) | top-512 overlap, mean (p10 / p50 / p90) | wrong picks if reused, median | certificate passes: rigorous / optimistic K / D = 0 | D / abs(v_k), median | boundary gap / abs(v_k), median |
|---|---|---|---|---|---|---|---|
| A, next step, same row | 52,352 | **0** | 0.50 (0.27 / 0.51 / 0.71) | 251 | **0** / 0 / 26,510 | 94 | 0.00046 |
| B, next step, committed predecessor | 13,136 | **0** | 0.58 (0.37 / 0.60 / 0.76) | 206 | **0** / 0 / 6,555 | 79 | 0.00044 |
| C, adjacent position, same step | 44,424 | **0** | 0.56 (0.34 / 0.58 / 0.74) | 215 | **0** / 0 / 22,391 | 83 | 0.00046 |

Per index layer (all cases; A / B / C):

| layer | overlap, mean | wrong picks, median | D / abs(v_k) (A) | gap / abs(v_k) (A) |
|---|---|---|---|---|
| 2 | 0.52 / 0.58 / 0.59 | 241 / 204 / 198 | 64 | 9.6e-05 |
| 8 | 0.39 / 0.48 / 0.46 | 313 / 252 / 262 | 206 | 4.6e-04 |
| 14 | 0.39 / 0.47 / 0.45 | 306 / 254 / 268 | 149 | 5.6e-04 |
| 20 | 0.54 / 0.61 / 0.59 | 232 / 189 / 197 | 114 | 8.9e-04 |
| 24 | 0.54 / 0.61 / 0.59 | 229 / 187 / 200 | 25 | 1.7e-04 |
| 28 | 0.60 / 0.68 / 0.64 | 202 / 149 / 172 | 68 | 6.6e-04 |
| 32 | 0.50 / 0.59 / 0.56 | 256 / 205 / 217 | 141 | 1.3e-03 |
| 36 | 0.49 / 0.58 / 0.56 | 257 / 208 / 219 | 141 | 1.3e-03 |

- **Certificate unit test on the recorded steps.** The kit's rule ("reuse iff the k-th score beats the best outside score + D, else rescore") was evaluated on every pair. It passed 0 times, and it was never unsound: no case passed while the set changed. Served output would therefore be the stock rescore on every step (bitwise stock by construction), plus the certificate's own cost. So it cannot be faster, and per the trim rule no serving code was built and no pinned A/B was run.
- **The kit's model vs data.** Its "50% pass rate" assumed a stable set. In the data, the set is never stable from one token to the next, at any layer, context length or content. The X-ray step's prefill dumps had already shown the same thing for prefill rows (adjacent-row overlap 0.18-0.32 at layer 14).
- **One exact lead (not kit 07; not built).** Use the previous step's v_k of the same row as a threshold hint for the decode top-k:
  - in 50.6% of rows at least 512 scores reach it (53.4 / 57.5 / 63.4% with -2 / -5 / -10% slack), with a median of 518 / 554 / 614 / 731 candidates;
  - a filter-then-select top-k with a full fallback would stay exact. Its ceiling is a fraction of the top-k share of the +2.30 ms 512K excess, likely <= ~0.3 ms/step (~1% at 512K), and 0 at G1 / G4 contexts.

### 4. Live state, files, rollback
- **Production** g5lc-PU (04:38:55 UTC, supervisor, launcher f214c2c9 defaults, AV unpinned): healthy at +204 s.
  - exact.py **a8c3394af9d2**, own profile 912/464/2320/1373.
  - 8/8 active, 0 restarts, 0 ERROR/Traceback.
  - No G5LC / DSV41_EXTRA / AV-pin variables in the rank environments; PYTHONPATH is `~/dsv41-sched` only.
  - SM 1969-1989 MHz on 8/8; clocks were never raised (`DSV41_GPU_CLOCK_MHZ=2000` on both launches).
- **Nothing to roll back.**
  - No site file is newer than 03:29 UTC on any node; the launcher is f214c2c9 (mtime 2026-10-27 17:48 UTC) on 8/8.
  - Both launches' preflight found manifest d773ed60f27b identical on 8/8.
  - The vision staging set stays valid, because no live file changed. Its W0 artifacts were not touched.
- **Downtime.** The R1 launch took 147 s (04:02:41-04:05:08); the restore took 217 s (04:38:42-04:42:19). Between them, R1 (exact, pinned, recorder idle) served the LAN / tailnet clients.
  - Foreign traffic seen: a ~120K-token chat at ~04:08 that overlapped one recorded question (discarded and rerun); overlaps during two prefix prefills; one timing rerun.
- **Files.**
  - spark-one `~/dsv41-g5lc/`:
    - `py/` (recorder + CPU test);
    - `scripts/` (`g5lc_probe.py`, `g5lc_an.py`, `g5lc_rec_run.sh`);
    - `corpus/` (code snapshot + exact-length prefixes);
    - `rec/R1/` (9 dumps, ~600 MB, safe to delete);
    - `runs/R1/` (probe JSONL);
    - `an/` (`an-all.json`).
  - `~/dsv41-g5lc/py/g5lc_rec.py` is staged on spark-two..eight too. It is inert unless G5LC_REC=1 and the extension flag are given.
  - Mac scratchpad `g5lc/`: same scripts, `g5lc_inst.sh`, `ab.log`, `launches.tsv`, `an/`, `runs/R1/`.
- **Goal status: unchanged.** Production G1 is ~101.5-101.8 of 120 tok/s. Long-context decode is +2.3 ms/step at 512K, a cost only long-context traffic pays.
