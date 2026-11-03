# g130 boundary: window runbook

Nothing here has run on a GPU, so nothing here is claimed exact. `tests/test_boundary_cpu.py` passed on spark-four with the Triton interpreter and torch CPU semantics; that checks wiring and my reading, not CUTLASS or CUDA bits. It also builds `Lin` from the GPU test on the real checkpoint scales. `tests/test_ops_stub.py` (stdlib, runs on the Mac) drives `ops/*.sh` against stub hosts and checks they fail when they should; it says nothing about real hosts. The rest of `tests/test_boundary_gpu.py` and `ops/*.sh` on the real 8 have never run, so expect a first-run fix or two, and a failure there should be loud. No compile: Triton kernels JIT at startup, before capture.

Launcher `b3976c4b`, 8/8 on 2026-11-02, with `dspark.py` 36811ca9 and `speculator.py` 439b390e live. Both anchors unchanged.

## Parts

| Part | Flag | Re-priced on prof4 (ms/step) | G1 | Status |
|---|---|---|---|---|
| BS06(a) context KV, one N=1536 GEMM, pinned tactic | `DSV41_CTXKV_FOLD=1` | -0.10..-0.12 | +0.42..+0.50% | built |
| Markov glue, keep K2, the bf16 add and the 16-byte all-gather | `DSV41_MK_FOLD=1` | -0.05..-0.10 | +0.22..+0.42% | built |
| Bundle | both | -0.15..-0.22 | +0.63..+0.92% | |
| BS06(b) winner-id verify | none | -0.03..-0.055 | +0.13..+0.23% | dropped, under 0.05 ms at central |
| 15 SWA metadata kernels | none | -0.026..-0.03 | +0.11..+0.13% | dropped, under 0.05 ms |
| BS06(c) and the fork | none | | | dropped per brief |

Traced, not measured. Neither built part clears +0.5% alone at its central value, so they ship together or with another lever.

Flags: all default off, and unset never imports `dsv41_boundary.py`. `DSV41_CTXKV_FOLD=split` is the no-copy fallback (three N=512 views); it has no audit path, so `arm.sh` will not serve it (section 5). `audit` on either flag serves the stock path and counts differing words on the device. `DSV41_MK_FOLD_TILES=16|1` picks the Markov scan shape; take `best_tiles` from the GPU test, and use the same value in the audit arm and the serving arms.

CUDA graph recapture: yes, both parts live in the FULL draft graphs, and the normal capture at each launch records them. `prebuild()` runs before capture. A fold that fails its self-check logs a warning and stays stock on that rank. That cannot desynchronise ranks: part (a) has no collective, and Markov issues the same all-gather either way. Check the log line, or the arm silently measures stock.

## 0. Setup, from the Mac

    export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
    which ssh    # must print the shim
    K=$HOME/Documents/dsv41-kits/g130-boundary
    python3 -I $K/tests/test_ops_stub.py | tail -1    # VERDICT PASS, no host contacted

Take CLUSTER_LOCK and log to AGENTS_LOG.md as usual, before the stop. Ask for Codex and survey traffic to stop: this is a timed bench. Release the lock only after section 7.

## 1. Stage to all 8 (disk only, safe while serving)

    bash $K/ops/stage8.sh

Must end `stage8: 8/8 staged OK <digest>` and exit 0; the digest is this mirror's. The stage also dry-runs the installer, which fails if a base sha moved (`dspark.py` 36811ca9, `speculator.py` 439b390e), so it also refuses on an installed site. Re-stage only after `site8.sh uninstall`.

**A failure in section 0 or 1 changed nothing.** The service is still up and unpatched on launcher `b3976c4b`; `stage8.sh` only wrote stage folders. Do not run section 2 or section 7: section 7 would stop a healthy service, and its `uninstall` needs the stage on every host (`stage8.sh` stops at the first bad host), so the relaunch would be skipped and the service would stay down. Fix the cause and rerun section 1, or release CLUSTER_LOCK and end the window.

## 2. Stop the service (your call)

Only after `stage8: 8/8 staged OK`. From this stop on, any failure, KILL or failed gate goes to section 7.

    for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do ssh -n $h 'timeout 30 systemctl --user stop dsv41-rank; systemctl --user is-active dsv41-rank' & done; wait

## 3. CPU and GPU tests, one node, no site install needed

    ssh -n spark-four 'set -o pipefail; cd ~/dsv41-stage/g130/boundary && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 19 timeout 600 ~/venvs/vllm032/bin/python tests/test_boundary_cpu.py 2>&1 | tail -30'; echo cpu_rc=$?
    ssh -n spark-four 'set -o pipefail; cd ~/dsv41-stage/g130/boundary && mkdir -p results && source ~/venvs/vllm032/bin/activate && CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH timeout 3000 python3 tests/test_boundary_gpu.py --tp-rank 3 2>&1 | tee results/gpu.jsonl | tail -30'; echo gpu_rc=$?

Read the Markov shape the arms will use. Keep `TILES` in this shell for sections 5 and 6; `arm.sh` refuses an empty value.

    TILES=$(ssh -n spark-four "grep '\"best_tiles\"' ~/dsv41-stage/g130/boundary/results/gpu.jsonl | tail -1" | python3 -I -c 'import json,sys;print(json.load(sys.stdin)["best_tiles"])'); echo TILES=$TILES

Rank 3 is spark-four. Duration is unmeasured; my guess is 15-40 min, mostly part (b) on 16160-column rows. Lower `--replays` only for a first smoke run: under 200 replays a GO reports as HOLD, since the replay count is part of the gate.

| Part | GO | HOLD | KILL |
|---|---|---|---|
| a | 0 differing words, pinned block at M=6 saves at least 90 us, GEMM at most 50 us | 0 differing words, saves 70-90 us; or merge misses and split saves at least 70 us (not servable in this kit: `split` has no audit path, the user decides) | any differing word on merge with no clean split, or saving under 70 us |
| b | 0 differing words in the normal classes and graph replays, saves at least 45 us at num_reqs=1 | saves 25-45 us; or a difference ONLY in the NaN or subnormal classes (do not serve until the user has seen it) | any difference in a normal class or graph replay, or saving under 25 us |
| audit plumbing | counters right in a captured graph, forced difference counted | | otherwise; do not trust an audit arm |

`gpu_rc` is 0 GO, 1 KILL or a crash (no `VERDICT` line), 2 HOLD, 124 timeout. If both parts KILL, skip to section 7. If one part KILLs, run the rest without its flag. A GO needs the serving autotune table on the node; without it part (a) reports HOLD.

## 4. Install on 8/8 (stopped service), flags still off

    bash $K/ops/site8.sh install
    bash $K/ops/site8.sh status

Must end `site8 install: 8/8 exit 0, identical, state patched` and exit 0. Any host that does not answer, exits nonzero, prints no `STATE` line, differs or is not `patched` makes it fail and names the host. A failed install can leave some hosts patched: run `uninstall`, then `status base`. The supervisor's manifest covers `dspark.py` and `speculator.py` but not `dsv41_boundary.py`, so `arm.sh` itself requires state `patched` on 8/8, which includes the side file hash, before every restart.

Rollback at any point (stopped service): `bash $K/ops/site8.sh uninstall`, which must end in state `base`.

## 5. Flags-off arm, then audit arm with concurrency

    bash $K/ops/arm.sh off
    bash $K/ops/arm.sh audit --conc DSV41_CTXKV_FOLD=audit DSV41_MK_FOLD=audit DSV41_MK_FOLD_TILES=$TILES DSV41_BOUNDARY_AUDIT_EVERY=64

Each `arm.sh` ends `ARM <label>: PASS` with exit 0, or says FAIL and exits 1. It fails on a missing or wrong exact (`a8c3394af9d28249`), G1 digest (`9a18f802d930f6e7`) or long prompt (against `final3/det.jsonl`, tokens and logprobs), a dead G4, any rank 0 error, or a rank that did not log the fold the flags asked for. The service stays up on either result. Stop on a FAIL; do not read its numbers as a measurement. Flags-off G1 should sit near the current 113.8; that one is your read.

The audit arm is the real exactness evidence. Digests are serial, at most 6 rows, and a weak detector for drafter flips. Its rows have real `main_x`, real logits and real concurrency: G4 at 4 requests and load32 at up to 32. Serving stays stock, so it must still hit both anchors, though G1 will be lower. Audit reports every 64 greedy steps per rank.

The audit only covers the kernels it ran, so `arm.sh` ties the serving arms to it. A passing audit arm leaves `~/dsv41-g130ab/boundary/audit/audit_pass.json` on spark-one (kit digest, parts audited, tiles, tactic map), and starting an audit arm deletes the old one. Before any restart, `arm.sh` refuses a `t` arm with no record, with another kit digest, serving a part the audit arm left out, with different tiles or a different tactic map, or with `split` (the audit shadow is always `merge`). After a dropped part, audit and serve only the other one. `audit` also refuses to run without `--conc`.

- GO (`arm.sh` checks all of it): `status OK` on 8/8, every `mk_*_mismatch*` and `ctx_mismatching_words` 0, at least 10,000 `mk_positions` per rank, `ctx_calls` nonzero, `load32` 32/32, and on all 8 rank logs `DSV41_CTXKV_FOLD=audit: rows pinned` and `DSV41_MK_FOLD=audit tiles N: self-check passed`. No such line means the arm measured nothing, and it fails.
- KILL: one mismatching word on any rank. Keep `~/dsv41-logs/boundary-audit.*.json` and rank logs, drop that part.

Compare the long prompts against the flags-off arm:

    ssh -n spark-one 'python3 -I ~/dsv41-stage/bs03/tools/cmp_cold.py ~/dsv41-g130ab/boundary/off/det.jsonl ~/dsv41-g130ab/boundary/audit/det.jsonl'

## 6. Pinned serving pairs, CTTC

    bash $K/ops/arm.sh c1
    bash $K/ops/arm.sh t1 DSV41_CTXKV_FOLD=1 DSV41_MK_FOLD=1 DSV41_MK_FOLD_TILES=$TILES
    bash $K/ops/arm.sh t2 DSV41_CTXKV_FOLD=1 DSV41_MK_FOLD=1 DSV41_MK_FOLD_TILES=$TILES
    bash $K/ops/arm.sh c2

Then:

    ssh -n spark-one 'cd ~/dsv41-g130ab/boundary; rc=0; for a in t1 t2 c2; do python3 -I ~/dsv41-stage/bs03/tools/cmp_cold.py c1/det.jsonl $a/det.jsonl || rc=1; done; exit $rc'; echo cmp_rc=$?
    ssh -n spark-one 'cd ~/dsv41-g130ab/boundary; for a in c1 t1 t2 c2; do tail -1 $a/g1.jsonl | python3 -I -c "import json,sys;print(sys.argv[1],json.load(sys.stdin)[\"G1_mean_tok_s\"])" $a; done'

Promotion: `ARM ...: PASS` on all four, `cmp_rc=0`, and at least +0.5% G1 in 2 pinned pairs (t1 vs c1, t2 vs c2). Expect the margin to be thin: the paper bundle is +0.63..+0.92% and pair noise is 0.1-0.2%. Any clean mismatch kills the arm's flag set. Quiet cluster, no survey ssh. No builds around a relaunch: on GB10 a compile can make a rank fail vLLM's 109.47 GiB startup check.

## 7. Back to production

Only for a window that reached section 2. A failure in section 0 or 1 ends the window with the service untouched, and this section must not run (see section 1). The arms leave the service up, and `uninstall` refuses while it runs, so stop first. Once section 2 has started, run these even after a KILL or a failed step. Skip the stop and the uninstall only if you are promoting.

    for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do ssh -n $h 'timeout 30 systemctl --user stop dsv41-rank; systemctl --user is-active dsv41-rank' & done; wait
    bash $K/ops/site8.sh uninstall && bash $K/ops/site8.sh status base && ssh -n spark-one 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > /tmp/rel.sh && bash /tmp/rel.sh --label g130-prod --expect-launcher b3976c4b DSV41_GPU_CLOCK_MHZ=2000; echo prod_rc=$?
    ssh -n spark-one 'cd ~/dsv41-bench && timeout 900 python3 exact.py /tmp/g130-prod-exact.json g130-prod > /dev/null 2>&1; python3 -I -c "import json;c=json.load(open(\"/tmp/g130-prod-exact.json\"))[\"combined\"];print(c[:16],\"OK\" if c.startswith(\"a8c3394af9d28249\") else \"MISMATCH\")"'

`prod_rc=0` and `a8c3394af9d28249 OK`, then release CLUSTER_LOCK. If `uninstall` fails, production must not come back on a patched site: read the named host, fix it, and repeat.

## Hazard reasoning

**Context KV bits.** Stock N=1792 and fold N=1536 tile output columns in 128s, and the wkv part starts on a tile boundary in both, so each CTA should do the same work on the same bytes. Scale slices are tile aligned (the CPU test checks the real `swizzle_mxfp8_scale`). `kv_norm` still reads a strided view in `merge`, and a contiguous one in `split`, so the GPU test compares after the norm too.

**Tactic.** N=1536 has no autotune entry, and a miss runs tactic 0 (about 3x slower on the draft head). Hence the direct runner call, as `_HeadTactic` does. Default map 1-64:1, 65-128:2, 129-192:4 mirrors serving for (5120, 1792). Rows outside the map, dummy runs and autotune passes use the stock path.

**Workspace.** The fold uses the same `mm_mxfp8_workspace` on the main stream. On prof4 ranks 0 and 4, 0 other-stream kernels overlapped the context block in 300 draft graphs, and on rank 0 none overlapped the Markov loop in 120 (prose and G4). No race is expected, and no side stream is added.

**Markov bits.** The torch bf16 add stays, so rounding and any flush behaviour cannot change. The new kernels compare integer keys and copy bits; there is no FP arithmetic. The stock max and argmax semantics I reproduce (NaN wins, first index wins, -0 == +0, value bits of the leftmost) are read from torch docs and CPU behaviour. CUDA behaviour on NaN and subnormal rows is what the HOLD class exists for. Real logits have a maximum far above the subnormal range, so that class should not arise in serving, but the gate reports it.

**AV.** Under the live pin the budget table is static, so a cheaper drafter should not move layouts. X and the earlier drafter changes kept the digest only under that pin. Do not run these flags with `DSV41_AV_COST_JSON=` empty or a refreshed pin.

**Not touched.** `K2`, the 16-byte all-gather, confidence head, sampled and mixed graphs (Markov fold is in the greedy path only; part (a) applies to both graph sets), `sparse_swa.py`, the rejection sampler. Memory +8.1 MB per rank in `merge`.

**JIT.** All kernels compile in the self-check before capture, with `do_not_specialize` on sizes and pointers, so a request should not JIT. Check for a rise in first-step latency in the first arm.
