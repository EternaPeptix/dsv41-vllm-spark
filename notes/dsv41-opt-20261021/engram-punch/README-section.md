
## Engram punch (2026-10-23 07:10-08:20 UTC). spark-one/two/three punched; spark-four..eight whole

**What.** In vLLM 0.32 TP8 each rank copies only its own contiguous run of Engram rows (3 of 24 hash heads per layer) from shards 47/48 into mlocked host RAM at load, and never reads the shard files again. On **spark-one (rank 0), spark-two (rank 1) and spark-three (rank 2)** the rows other ranks own were released with `fallocate(PUNCH_HOLE)`. The file sizes and offsets are unchanged, and the unread rows now read back as zeros. Tool: `ops/engram-punch/engram_punch.py` (sha f79b3748; staged at `~/dsv41-stage/engram-punch/` on 8/8). The derivation and audit are in the Mac scratchpad `engram-punch/PLAN.md`.
- **Not punched.** spark-four is the pristine restore source and is never punched. spark-five..eight were left whole in this window because they cannot ssh to spark-four (10.77.0.41), which leaves them no restore path. Fix their keys first, then punch them the same way.
- **Freed:** 165.23 GiB per node (82.61 + 82.62), **495.7 GiB in total**. Free space: spark-one 31.5 -> 196.8 GiB, spark-two 33.3 -> 198.5, spark-three 149.8 -> 315.0. Shards 47+48 now allocate 23.90 GiB on a punched node (189.13 GiB on spark-four). Keep about 165 GiB free on each punched node, or a restore is impossible.
- **Per-node proof** (state in `~/dsv41-stage/engram-punch/{state/,punch.log}`):
  - Before punching, the full sha256 matched uncensored.sha. The kept segments (header, own weight and scale rows, tail) matched spark-four's `--ref-all` manifest (ref.json dbdc3b09; spark-four full shas match published).
  - After punching, the kept segments were re-hashed with the page cache dropped, and pre == post.
  - The kept rows contain no holes.
  - The xattr `user.dsv41.engram_punch` = {rank, world 8, rows, state done}.
  - `--check --deep` OK on all 3. `check_all.sh` gives 8/8 OK (3 punched + 5 whole).
  - Negative test: `--rank 0 --check` on spark-two FAILS, as it must.
- **Linux synthetic test.** `test_engram_punch.py` ran on spark-one ext4 (first run of the fallocate path) and PASSED, after a fix to the test harness only (cb7d89ca; was 489eb643). On Linux `shutil.copytree` used copy2, which also copied the rank-1 punch xattr into the "zeros as data" fixture. Now it uses `copy_function=shutil.copyfile`, which matches an rsync without -X. The tool itself was not changed.

**Guards now in place**
- `~/dsv41-tp8.sh` 85189840 -> **0f1ab42f** on 8/8 (backup `~/dsv41-tp8.sh.bak-punch` = 85189840). The Engram row check runs right after the venv activate line. It runs `engram_punch.py --rank $R --world 8 --check` (about 1 s; holes, xattr and a 64-row zero probe), logs to `~/dsv41-logs/engram-check.log`, and exits before `vllm serve` if this node lacks rank R's rows.
- `ops/dsv41-launch-all.sh` and `ops/conc/launch-conc.sh` (backups `*.bak-punch`) now begin with `ops/engram-punch/check_all.sh` over all 8 nodes and refuse to start anything if one fails. check_all.sh runs from the Mac with bash, the same way as the README launch: `ssh spark-one 'cat .../dsv41-launch-all.sh' | bash`.
- **ENGRAM_CHECK=0** skips both checks. For launch-all use `ENGRAM_CHECK=0 bash ...`; for the rank check pass `ENGRAM_CHECK=0` via `launch-conc.sh`, which goes through systemd-run -E. **Never use it on a punched node**: vLLM 0.32 has no missing-row check and would silently serve zeros for those Engram rows.

**Verification**

Exactness method:
- On the live 1M config, exact.py is **not deterministic even within one instance**. With VERIFY_CAP=0 and ROUTE_LIVE=0 at 1M, 3 back-to-back runs gave 3 different hashes (49941910d174 / 61628a22d5b5 / 62162ae40c8b), before any punch.
- So the reference was taken at `DSV41_VERIFY_CAP=0 DSV41_ROUTE_LIVE=0 DSV41_MAX_MODEL_LEN=262144` (launch-conc.sh env). That config is stable within an instance (3/3) and reproduces the round-4 cross-restart anchor a8c3394af9d2.
- The 1M within-instance non-determinism existed before the punch. It is unexplained; a suspect is idxshard or kernels whose behaviour depends on max_model_len. It is not investigated here.

| check | before punch (unpunched) | after punch |
|---|---|---|
| exact.py (cap 0, route 0, 262144) | a8c3394af9d2 x3 | **a8c3394af9d2 x3** |
| GSM8K-50 conc8 (same instance) | 49/50 (item 12, known flip), 240.8 tok/s | **50/50**, 234.0 tok/s |
| `Engram table offloaded` rows (16 rank x layer) | as PLAN section 5 | unchanged, e.g. r0 48000217 / 48001463 |
| engram-check.log | - | OK 8/8 |

Live relaunch (1M, verify-cap auto, route skip; launch-all incl. check_all 8/8 OK, health 200 after 407 s):
- longctx.py default gate (4K-200K, 2 seeds): **PASS**, needles 60/60, copy 1.000 (wall 181 s).
- 500K needle run (lc 500k, 1 seed, 7 depths): **PASS**, 7/7 needles, copy exact (sim 1.000), 497,263 prompt tokens, 91.1 s request.
- Raw results: `./engram-punch/{b0 (1M non-determinism probe), b1 (baseline), p1 (post-punch), live}/`; compare tool `engram-punch/excmp.py`.

**Restore one punched node** (`ops/engram-punch/restore_from_four.sh`):
1. Stop vLLM on that node.
2. On the node, run `bash ~/dsv41-stage/engram-punch/restore_from_four.sh`. It rsyncs `-I -W --inplace` from jeweled@10.77.0.41, which needs about 165 GiB free. It then checks the full sha256 against uncensored.sha and removes the xattr.
3. Relaunch.

As of today only spark-one, two and three can reach 10.77.0.41.

**Rules**
- **Never rsync, scp or push model files FROM a punched node** (push-model.sh, fabric-pull.sh, rsync-*.sh). Holes are copied as real zeros. Use only spark-four, or another whole node, as a source.
- `sha256sum -c uncensored.sha` **will report FAILED for shards 47/48 on punched nodes. This is expected.** Use `engram_punch.py --rank R --check --deep` there instead. All other shards must still match.
- **Any layout change requires restoring first.** That includes TP != 8, enabling DP or engram DP sharding, or changing which host runs which rank (launch-all HOSTS order). The head-to-rank map changes, and the check will refuse to launch.
- `engram-disk/nvme_bench.py` uses approximate per-rank slices that run into neighbouring (punched) rows. Fix it to exact ranges before benchmarking on punched nodes.
- spark-four is the single pristine copy for 3 nodes. If its NVMe fails, they need a 2 x 101.5 GB re-download each.
