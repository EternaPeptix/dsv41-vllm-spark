## Engram punch (2026-10-23; one/two/three 07:10-08:20 UTC, five..eight 11:40-12:35 UTC). 7 nodes punched; spark-four whole

**What.** In vLLM 0.32 TP8 each rank copies only its own contiguous run of Engram rows (3 of 24 hash heads per layer) from shards 47/48 into mlocked host RAM at load, and never reads the shard files again. On **every node except spark-four** (spark-one..three = ranks 0..2, spark-five..eight = ranks 4..7) the rows other ranks own were released with `fallocate(PUNCH_HOLE)`. The file sizes and offsets are unchanged, and the unread rows now read back as zeros. Tool: `ops/engram-punch/engram_punch.py` (sha f79b3748; staged at `~/dsv41-stage/engram-punch/` on 8/8). The derivation and audit are in the Mac scratchpad `engram-punch/PLAN.md`.
- **Not punched.** spark-four (rank 3) is the pristine restore source and is never punched. After the second window its shards 47/48 still pass `sha256sum -c` against uncensored.sha.
- **Restore path for five..eight.** Each of spark-five..eight has a read-only key `~/.ssh/engram_restore`, which spark-four forces to `rrsync -ro` of the model dir, so remote paths are relative to the model dir. `restore_from_four.sh` (c36edacb) picks the key up automatically. Before punching, all 4 nodes listed both shards through that key (`rsync --list-only`, sizes 101,535,150,936 / 101,537,926,640).
- **Freed:** 165.23 GiB per node (82.61 + 82.62), **1,156.6 GiB in total over 7 nodes**. Shards 47+48 now allocate 23.90 GiB on a punched node (189.13 GiB on spark-four). Keep about 165 GiB free on each punched node, or a restore is impossible.

| node | rank | freed | free before -> after | `--check --deep` |
|---|---|---|---|---|
| spark-one | 0 | 165.23 GiB | 31.5 -> 196.8 GiB | OK |
| spark-two | 1 | 165.23 | 33.3 -> 198.5 | OK |
| spark-three | 2 | 165.23 | 149.8 -> 315.0 | OK |
| spark-four | 3 | - (pristine) | 197.5 | whole; sha256 47/48 OK |
| spark-five | 4 | 165.23 | 179.8 -> 345.0 | OK |
| spark-six | 5 | 165.23 | 185.2 -> 350.4 | OK |
| spark-seven | 6 | 165.23 | 135.6 -> 300.9 | OK |
| spark-eight | 7 | 165.23 | 149.2 -> 314.4 | OK |

- **Per-node proof** (state in `~/dsv41-stage/engram-punch/{state/,punch.log}`; five..eight copied to `./engram-punch/5to8/spark-*/`):
  - Before punching, the full sha256 matched uncensored.sha (47: 76a05106, 48: 9dc51433 on five..eight). The kept segments (header, own weight and scale rows, tail) matched spark-four's `--ref-all` manifest (ref.json dbdc3b09; spark-four full shas match published).
  - After punching, the kept segments were re-hashed with the page cache dropped, and pre == post. Punch ranges reported as holes: 100 % (0 B not holed).
  - The kept rows contain no holes.
  - The xattr `user.dsv41.engram_punch` = {rank, world 8, rows, state done}.
  - `--check --deep` OK on all 7. `check_all.sh` gives 8/8 OK (7 punched + spark-four whole).
  - Negative tests: `--rank 0 --check` on spark-two FAILS, and `--rank 4 --check` on spark-six FAILS ("punched for rank 5/8, not 4/8"), as they must.
  - Host guard in action: the first five..eight start used a bash-4 associative array, which the Mac's bash 3.2 silently ignores, so all four got `--rank 7`. The tool refused on five/six/seven ("spark-five runs rank 4, not --rank 7") before touching any byte; spark-eight (really rank 7) proceeded. Use `host:rank` pairs (`for p in five:4 six:5 ...`), not `declare -A`, in Mac-side loops.
- **Linux synthetic test.** `test_engram_punch.py` ran on spark-one ext4 (first run of the fallocate path) and PASSED, after a fix to the test harness only (cb7d89ca; was 489eb643). On Linux `shutil.copytree` used copy2, which also copied the rank-1 punch xattr into the "zeros as data" fixture. Now it uses `copy_function=shutil.copyfile`, which matches an rsync without -X. The tool itself was not changed.

**Guards now in place**
- `~/dsv41-tp8.sh` 85189840 -> **0f1ab42f** on 8/8 (backup `~/dsv41-tp8.sh.bak-punch` = 85189840). The Engram row check runs right after the venv activate line. It runs `engram_punch.py --rank $R --world 8 --check` (about 1 s; holes, xattr and a 64-row zero probe), logs to `~/dsv41-logs/engram-check.log`, and exits before `vllm serve` if this node lacks rank R's rows.
- `ops/dsv41-launch-all.sh` and `ops/conc/launch-conc.sh` (backups `*.bak-punch`) now begin with `ops/engram-punch/check_all.sh` over all 8 nodes and refuse to start anything if one fails. check_all.sh runs from the Mac with bash, the same way as the README launch: `ssh spark-one 'cat .../dsv41-launch-all.sh' | bash`.
- **ENGRAM_CHECK=0** skips both checks. For launch-all use `ENGRAM_CHECK=0 bash ...`; for the rank check pass `ENGRAM_CHECK=0` via `launch-conc.sh`, which goes through systemd-run -E. **Never use it on a punched node** (that is now every node except spark-four): vLLM 0.32 has no missing-row check and would silently serve zeros for those Engram rows.

**Verification**

Exactness method:
- On the live 1M config, exact.py is **not deterministic even within one instance**. With VERIFY_CAP=0 and ROUTE_LIVE=0 at 1M, 3 back-to-back runs gave 3 different hashes (49941910d174 / 61628a22d5b5 / 62162ae40c8b), before any punch.
- So the reference was taken at `DSV41_VERIFY_CAP=0 DSV41_ROUTE_LIVE=0 DSV41_MAX_MODEL_LEN=262144` (launch-conc.sh env). That config is stable within an instance (3/3) and reproduces the round-4 cross-restart anchor a8c3394af9d2.
- The 1M within-instance non-determinism existed before the punch. It is unexplained; a suspect is idxshard or kernels whose behaviour depends on max_model_len. It is not investigated here.

| check | before punch (unpunched) | after one..three | after five..eight (7 punched) |
|---|---|---|---|
| exact.py (cap 0, route 0, 262144) | a8c3394af9d2 x3 | **a8c3394af9d2 x3** | **a8c3394af9d2 x2** (per-prompt shas identical to b1) |
| GSM8K-50 conc8 (same instance) | 49/50 (item 12, known flip), 240.8 tok/s | **50/50**, 234.0 tok/s | **50/50**, 239.3 tok/s |
| `Engram table offloaded` rows (16 rank x layer) | as PLAN section 5 | unchanged, e.g. r0 48000217 / 48001463 | unchanged on 8/8 (r4 48000873 / 48002199 ... r7 48001319 / 48002645) |
| engram-check.log | - | OK 8/8 | OK 8/8 |

Live relaunch after one..three (1M, verify-cap auto, route skip; launch-all incl. check_all 8/8 OK, health 200 after 407 s):
- longctx.py default gate (4K-200K, 2 seeds): **PASS**, needles 60/60, copy 1.000 (wall 181 s).
- 500K needle run (lc 500k, 1 seed, 7 depths): **PASS**, 7/7 needles, copy exact (sim 1.000), 497,263 prompt tokens, 91.1 s request.

Live relaunch after five..eight (same live config; launch-all check_all 8/8 OK; /v1/models max_model_len 1048576). The first launch hung at the known rdma_ar graph capture (all ranks at `rdma_ar audit ... (160, ...)` for more than 6 min); stopping all 8 and relaunching once fixed it, health 200 after 388 s.
- longctx.py default gate: **PASS**, needles 60/60, copy 1.000 (wall 181 s).
- 500K needle run (1 seed, 7 depths): **PASS**, 7/7 needles, copy exact (sim 1.000), 497,263 prompt tokens, 90.1 s request.

Raw results: `./engram-punch/{b0 (1M non-determinism probe), b1 (baseline), p1 (post-punch), live}/` for one..three and `./engram-punch/5to8/{det, live, spark-five..eight, mac-logs}/` for five..eight; compare tool `engram-punch/excmp.py`.

**Restore one punched node** (`ops/engram-punch/restore_from_four.sh`, c36edacb):
1. Stop vLLM on that node.
2. On the node, run `bash ~/dsv41-stage/engram-punch/restore_from_four.sh`. It rsyncs `-I -W --inplace` from jeweled@10.77.0.41 (one..three with id_ed25519, five..eight with the read-only `~/.ssh/engram_restore` key), which needs about 165 GiB free. It then checks the full sha256 against uncensored.sha and removes the xattr.
3. Relaunch.

All 7 punched nodes can now reach 10.77.0.41.

**Rules**
- **Never rsync, scp or push model files FROM a punched node** (push-model.sh, fabric-pull.sh, rsync-*.sh). Holes are copied as real zeros. spark-four is the only valid source.
- `sha256sum -c uncensored.sha` **will report FAILED for shards 47/48 on the 7 punched nodes. This is expected.** Use `engram_punch.py --rank R --check --deep` there instead. All other shards must still match.
- **Any layout change requires restoring first.** That includes TP != 8, enabling DP or engram DP sharding, or changing which host runs which rank (launch-all HOSTS order). The head-to-rank map changes, and the check will refuse to launch.
- `engram-disk/nvme_bench.py` now computes exact per-rank row ranges (fixed during the Engram-on-disk run, below), so it is safe to run on punched nodes. Any other tool that reads shards 47/48 directly must likewise use exact per-rank ranges, never approximate `N//8` slices.
- spark-four is now the single pristine copy for 7 nodes. If its NVMe fails, each of them needs a 2 x 101.5 GB re-download (about 1.4 TB in total). Consider a second full copy of shards 47/48 elsewhere.

