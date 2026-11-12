# g130 fc1: G-aware tile deal, window runbook

Nothing here has run on a GPU, so no exactness is claimed beyond reasoning, and neither the fluid model's staircase nor the sanitizer runtime has been measured. On the CPU: `tests/test_cpu.py` passes on spark-six, all arms compile, and the SASS gate passes. `bench/fc1_probe.py`, `tests/test_deal_gpu.py`, `tests/test_route_rec_gpu.py`, `site/fc1_route_rec.py`, `ops/pair.sh`, `ops/run-gpu.sh`, `ops/stage8.sh`, `ops/site-node.sh install` and `tools/rec_g1.py` have only been syntax-checked or dry-run. The lock and log commands below ran only against scratch files on spark-six, never the real `CLUSTER_LOCK` or `AGENTS_LOG.md`, and the exact command there used a stub. Expect a first-run fix or two in the harness.

## What it is

FC1 runs on 48 CTAs. The live walk pins each CTA to one m residue, so tiles pile up on a few CTAs, and their tails run at the 396 ns stage floor. The deal hands out the same tiles in order to n CTAs, using G read from the layout in the kernel prologue. See `header/dsv41_fc1_deal.cuh`.

| Flag | Default | Effect |
|---|---|---|
| `DSV41_MOE_FC1_DEAL=deal` | unset | Target FC1 compiled from `trees/deal-base`: m-block's 9 N tiles on adjacent CTAs |
| `DSV41_MOE_FC1_DEAL=dealx` | unset | Same, one N tile across all m-blocks first |
| `DSV41_MOE_FC1_ROUTE_REC=1` | unset | Route recorder, diagnostic instance only, invalidates timing |

Unset means the production tree, JIT cache and manifest. `gemm()` gains one `is not None` test, which runs at capture or eagerly, never on a replay. The patch adds 36 lines to `sm121_target_m32.py` and removes none.

CUDA graphs: captured at every start as usual, so a relaunch with the flag is the recapture. No persisted graph is involved.

The arm is a tree swap, so the native `.so` is byte-identical (b1c4eff8), and only the JIT header changes. E128 drafter FC1 and FC2 compile to SASS identical to live (`build-deal.sh sass`). Target FC2 is not in this tree's path.

## Order and cost

| Step | Needs | API down | Decides |
|---|---|---|---|
| 1 stage | the lock from section 0, not during a relaunch | 0 | |
| 2 probe | stopped, no site change | about 15 min | GO / HOLD / KILL on timing |
| 3 exact + race + sync | stopped | about 30 min (sanitizer time at 19 replays unmeasured; 80 min at the timeouts) | KILL on any mismatch |
| 4 route record, optional before 5, required for a final number | site, one instance | about 15 min | real G1 weights |
| 5 pinned pairs | site, 4 instances | about 50 min | promotion |

Stop at the first KILL and go to section 7. Steps 2 and 3 fit one window with the other kits' stopped work.

## 0. Setup, from the Mac

    export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
    which ssh    # must print the shim
    K=~/Documents/dsv41-kits/g130-fc1
    ssh -n spark-one 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > /tmp/rel-fc1.sh
    test -s /tmp/rel-fc1.sh && bash -n /tmp/rel-fc1.sh && echo SUPERVISOR-OK || echo 'ABORT: no supervisor, stop nothing'
    for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do ssh -n $h 'echo $(hostname) $(sha256sum ~/dsv41-tp8.sh | cut -c1-8)'; done

Do not stop anything without `SUPERVISOR-OK`, `b3976c4b` on 8/8 and the lock below: section 7 needs `/tmp/rel-fc1.sh` to bring the service back, and it is fetched only here.

**Lock.** `CLUSTER_LOCK` and `AGENTS_LOG.md` are in `~/dsv41-opt-20261021/` on spark-one. Read both first:

    ssh -n spark-one 'cat ~/dsv41-opt-20261021/CLUSTER_LOCK 2>/dev/null || echo NO-LOCK; tail -n 5 ~/dsv41-opt-20261021/AGENTS_LOG.md'

| Lock reads | Do |
|---|---|
| `NO-LOCK` | Take it with the command below, which cannot overwrite a holder. It must print `LOCK-TAKEN`. |
| `purpose=goal 130: ...`, `start=2026-11-02T09:11:05Z`, and the user or the orchestrating session confirms this window runs under goal 130 | Do not take a second lock. Work under this one, and leave it in place at the end. |
| Anything else, including another `owner=claude` lock or one past its eta | Stop. Never overwrite or remove it. Wait for the holder or ask the user. |

Also wait if the log tail shows another agent mid-run. To take a free lock (the 6 h eta covers every timeout in steps 2 to 5):

    ssh -n spark-one 'set -C; printf "owner=claude\npurpose=g130 fc1 window\nstart=%s\neta=%s\n" "$(date -u +%FT%TZ)" "$(date -u -d +6hours +%FT%TZ)" > ~/dsv41-opt-20261021/CLUSTER_LOCK && echo LOCK-TAKEN'

Then, under either kind of lock, keep a copy of it and log the start with the request to stop foreign traffic. Say the same to the user, because a client that does not read the log will not see it.

    ssh -n spark-one 'cat ~/dsv41-opt-20261021/CLUSTER_LOCK' > /tmp/fc1-lock.txt; test -s /tmp/fc1-lock.txt && cat /tmp/fc1-lock.txt
    ssh -n spark-one 'D=~/dsv41-opt-20261021; echo "$(date -u +%FT%TZ) | agent=claude | action=g130 fc1 window start: lock $(grep ^purpose= $D/CLUSTER_LOCK) $(grep ^start= $D/CLUSTER_LOCK); all 8 ranks stop soon, API down about 2 h. REQUEST: Codex and survey, stop traffic to port 8000 until the line saying fc1 window done | launcher_before=$(sha256sum ~/dsv41-tp8.sh | cut -c1-8)" >> $D/AGENTS_LOG.md'

**Lock check.** Before each stop, GPU run and relaunch in sections 2 to 7, this must print `LOCK-OURS`:

    ssh -n spark-one 'cat ~/dsv41-opt-20261021/CLUSTER_LOCK' | cmp - /tmp/fc1-lock.txt && echo LOCK-OURS

Anything else means do nothing more and go to the user; section 7 says what to do with the service down. If the eta is under 45 min away, extend it after a passing check, keeping owner, purpose and start, then repeat the copy to `/tmp/fc1-lock.txt` and add a log line:

    ssh -n spark-one 'F=~/dsv41-opt-20261021/CLUSTER_LOCK; sed "s/^eta=.*/eta=$(date -u -d +3hours +%FT%TZ)/" $F > $F.tmp && mv -f $F.tmp $F'

## 1. Stage to all 8 (disk only, safe while serving; never during a relaunch)

    bash $K/ops/stage8.sh

Every host must print `staged OK` with the same digest. This needs no compile: `.so` copies come from each node's pinned all-base, and cubins come from the kit. Takes about 1 min and 180 MB per node. If you ever rebuild, `build-deal.sh` holds a guard that stops on low memory or if the service leaves `active`. Rule from 2026-11-02: no builds around a relaunch (vLLM needs 109.47 GiB free at startup).

## 2. Stop and probe (no serving change)

    ssh -n spark-one 'cat ~/dsv41-opt-20261021/CLUSTER_LOCK' | cmp - /tmp/fc1-lock.txt && echo LOCK-OURS    # no LOCK-OURS, no stop
    for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do ssh -n $h 'timeout 30 systemctl --user stop dsv41-rank; systemctl --user is-active dsv41-rank' & done; wait
    ssh -n spark-six 'cd ~/dsv41-stage/g130/fc1 && timeout -k 20 2800 bash ops/run-gpu.sh probe'
    ssh -n spark-three 'cd ~/dsv41-stage/g130/fc1 && timeout -k 20 2800 bash ops/run-gpu.sh probe'
    ssh -n spark-six 'cd ~/dsv41-stage/g130/fc1 && python3 -I tools/eval_probe.py results/probe-*.jsonl'
    ssh -n spark-three 'cd ~/dsv41-stage/g130/fc1 && python3 -I tools/eval_probe.py results/probe-*.jsonl'

Sections: A0/A5 per-G `all`, `deal`, `dealx`, paired; tokens 1,2,3,4,6,8,12,16,24; 31 paired trials of 40 launches. A1 is t0 (G=0). A2 is L2-warm G 1..8. A4 is no-QMMA. All G are synthetic expert sets over 4 real layers, since G1 routes do not exist yet. Section A3 (halfmask) is optional: copy `A/moet-native-halfmask-tree-v1` from spark-one first, SASS-check its branch, then `bash ops/run-gpu.sh probe --sections A3 --extra halfmask=<tree>`. It only matters if the deal is killed.

Thresholds (`tools/eval_probe.py`), weighted by the prof4 trace-inverted G prior (`results/g_prior.json`), replaced by recorded weights after step 4:

| Verdict | Prose saving per FC1 call | Code saving per FC1 call |
|---|---|---|
| **GO** | >= 3.5 us | and >= 2.5 us |
| **KILL** | < 1.5 us | and < 1.5 us |
| **HOLD** (bundle rider only) | anything else | |

Also KILL: an arm slower than `all` by more than 2 us per call on any G holding over 15% of a mix (that would mean C fell with fewer CTAs).

Read the staircase too. The model has one step below G 14, +18 us at G 8 to 9, and the next is +32 us at G 16 to 17 (+16 at tokens 3). This matters because the prior puts 68% of prose calls and 11% of code calls below G 14. A5 times `all` at every G from 1 to min(nmb, 40) at tokens up to 12, and to 72 above that. `eval_probe.py` prints a `STAIRCASE` line from those recorded G: `holds`, `model wrong` (median step under half the model's) or `not tested` (a G from 1 to min(nmb, 14) is missing, so rerun the probe before quoting it). The measured verdict stands either way, but the audit's arithmetic does not. Pick the arm with the larger minimum of the two savings; `dealx` costs an extra per-tile division, which the timing already includes. A difference under 0.5 us means take `deal`.

## 3. Exactness, only after GO or HOLD

Run the lock check first: these runs hold a GPU for up to 40 minutes each, and `g1lib.guard` looks at the service only when a process starts.

    ARM=deal
    for h in spark-six spark-three; do ssh -n $h "cd ~/dsv41-stage/g130/fc1 && timeout -k 20 3100 bash ops/run-gpu.sh exact --arms $ARM --replays 240" & done; wait
    ssh -n spark-six "cd ~/dsv41-stage/g130/fc1 && timeout -k 20 2500 bash ops/run-gpu.sh race $ARM"
    ssh -n spark-six "cd ~/dsv41-stage/g130/fc1 && timeout -k 20 2500 bash ops/run-gpu.sh sync $ARM"

Tokens are 1,2,4,6,8,12,16,24, on synthetic and real-layer weights. Each test compares the int view of the output, with the output poisoned before every replay. The direct phase has FC1 alone, with G rewritten between replays, whole buffer compared including dead rows; the apply phase runs the served path with six route patterns plus recorded routes. A/A control runs every 8th replay. At 240 replays the direct phase reaches every G from 0 to min(nmb, 96); each row records them in `Gs`, with median ms per G in `per_G` as a rough cross-check on the probe (unpaired, a few samples per G).

Race and sync run tokens 6, synthetic weights, FC1 alone, `--gs 0-18 --replays 19`: each G from 0 to 18 once, plus eager capture at G 8, so every model step below G 14 and the 16 to 17 step run under the sanitizer. Each replay prints an `event: replay` line, so a slow run can be told from a hung one. They give no timing (`timing_valid` false). Their runtime at 19 replays is unmeasured: each timeout is 2400 s, and if one fires with replay lines still advancing, that is a harness result, not a KILL. Rerun it with a shorter `--gs` and record what it left out.

- **GO**: last line `"status": "PASS"` on both exact ranks, and rc 0 on race and sync with `G_covered` listing 0 to 18 and 0 errors in `ERROR SUMMARY` (`run-gpu.sh` prints both lines, and returns 71 if the PASS line is missing).
- **KILL**: any byte mismatch in an arm (no retries, it is not a noise question), any hang or timeout, any racecheck or synccheck report that `all` does not also give. A mismatch in A/A means the harness is broken and is no verdict.

`run-gpu.sh` also fails if the production all-base JIT cache moved. The recorder test belongs here too if step 4 follows:

    ssh -n spark-six 'cd ~/dsv41-stage/g130/fc1 && bash ops/run-gpu.sh rec'

## 4. Record real G1 routes (one instance; gives no G1 number)

G1 routes have never been recorded. FC1 runs inside graphs, so Python cannot see replay layouts, and the old `physical_route_observer` script is gone. `site/fc1_route_rec.py` puts one small kernel before every FC1 launch, captured in the graphs, writing a ring. Its only effect is added time. A mismatch in exact.py under this flag is something to investigate; it does not count against the deal. Run the lock check first.

    for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do ssh -n $h 'cd ~/dsv41-stage/g130/fc1 && bash ops/site-node.sh install' & done; wait
    bash /tmp/rel-fc1.sh --label g130-fc1-rec --expect-launcher b3976c4b DSV41_GPU_CLOCK_MHZ=2000 DSV41_MOE_FC1_ROUTE_REC=1
    ssh -n spark-one 'cd ~/dsv41-stage/g130/fc1 && timeout 900 python3 -I tools/rec_g1.py results/rec1'
    ssh -n spark-one 'cd ~/dsv41-stage/g130/fc1 && python3 -I tools/routes_to_hist.py results/rec1/routes.bin | tail -8'

A `STOP: prefix contract violated` line kills the deal: `find_G` rests on that contract. Copy `results/rec1` to spark-six, then re-read the verdict on real weights, for prose and code:

    ssh -n spark-one 'tar cf - -C ~/dsv41-stage/g130/fc1/results rec1' | ssh spark-six 'tar xf - -C ~/dsv41-stage/g130/fc1/results'
    ssh -n spark-six 'cd ~/dsv41-stage/g130/fc1 && python3 -I tools/eval_probe.py results/probe-<stamp>.jsonl --hist results/rec1/hist-prose.json results/rec1/hist-code.json'

Then A6, which replays real layouts on a stopped node (`bash ops/run-gpu.sh probe --sections A6 --routes-bin results/rec1/routes.bin`). This also closes the math gap: `hist-math.json` is the first math weighting anyone has had.

## 5. Pinned serving pair, CTTC (restarts the service 4 times)

Site installed on 8/8 (step 4 or `site-node.sh install`). The launcher's default AV pin stays. `pair.sh` does not read the lock, so run the lock check before each of the four lines.

    bash $K/ops/pair.sh c1 C
    bash $K/ops/pair.sh t1 T DSV41_MOE_FC1_DEAL=deal
    bash $K/ops/pair.sh t2 T DSV41_MOE_FC1_DEAL=deal
    bash $K/ops/pair.sh c2 C

Each arm must show `exact a8c3394af9d2` and digest `9a18f802d930f6e7`. A clean mismatch stops the variant for good. `c1`/`c2` also test the installed site file with flags off. Check foreign traffic by the success delta (dbgrun = 36).

On a T instance before promoting, the checks digests do not cover (digests are serial, at most 6 rows):

    ssh -n spark-one 'mkdir -p ~/dsv41-g130ab/fc1/final; cd ~/dsv41-g130ab/fc1/final; for i in 1 2 3; do python3 -I ~/dsv41-bench/goal/goal_conc4.py --out g4.jsonl --label fc1 2>&1 | tail -1 | cut -c1-40; done; python3 -I ~/dsv41-g130ab/load32.py; cd ~/dsv41-g5det && timeout 900 python3 -I scripts/dbgrun.py - $HOME/dsv41-g130ab/fc1/final/det.jsonl fc1 big:2500,big:8000,big:32000,big:128000,code:4000,prose:6000,l3:B:T16-spam-reply:1,l3:B:T02-classify:1,l3:A:T01-reviews:1 4 0 cold'

The 9 `tok_sha` prefixes must equal the 11-02 control (`big:2500 3f4a9dd1`, `big:8000 d3e0f849`, `big:32000 7ba3d1a9`, `big:128000 89bd9111`, `code:4000 43c4b492`, `prose:6000 05e29037`, `l3:B:T16-spam-reply:1 37dd0438`, `l3:B:T02-classify:1 6b7032f2`, `l3:A:T01-reviews:1 6d8cf8a2`), each single-valued.

**Promote** only with exact a8, digest and long hashes on every instance and at least +0.5% G1 in both pairs. G4 slowest must not fall by more than its noise; the model gives G4 about +0.1%, so this is a G1-only lever. **Below +0.5%**: keep it for a bundle with another kit's riders. Each exactness gate still has to run in its own arm first.

## 6. Promotion

Add `export DSV41_MOE_FC1_DEAL=${DSV41_MOE_FC1_DEAL:-deal}` to the launcher through the usual `.bak` procedure, with a fresh `--expect-launcher` sha. Note: the site file is on the vision set's manifest, so re-stage that set before any image retry. Promotion runs under the same lock. It still ends with the last two commands of section 7.

## 7. Back to production

Run the lock check first, before any stop, uninstall or relaunch here. A different lock means another supervisor holds the cluster: do not stop or relaunch, and tell the user. A missing or unreadable lock does not hold up a restore, because the service is down: log it, tell the user and bring production back.

Flag-free relaunch is production, even with the site file installed:

    bash /tmp/rel-fc1.sh --label g130-prod --expect-launcher b3976c4b DSV41_GPU_CLOCK_MHZ=2000

`/tmp/rel-fc1.sh` comes from section 0. Exit 2 means pre-flight refused and nothing changed: the service is still down, so read the launcher sha on 8/8 and fix the cause. Do not leave it there.

If the site file was installed and the deal is dead, uninstall it with the service stopped, before that relaunch:

    for h in spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight; do ssh -n $h 'timeout 30 systemctl --user stop dsv41-rank'; ssh -n $h 'cd ~/dsv41-stage/g130/fc1 && bash ops/site-node.sh uninstall'; done

`uninstall` restores base `ea4a8814` and refuses if the live file is neither ours nor that base. If no site install happened (step 2 or 3 only), use the relaunch alone. The stage dir is inert and can stay.

If a kernel arm hangs in a stopped-window test, the wrapper's `timeout` handles it: check `nvidia-smi` is clean on that node before the relaunch.

Last, confirm exact on the restored service. It must print `a8c3394af9d28249 OK`, which is `exact a8c3394af9d2`:

    ssh -n spark-one 'rm -f /tmp/g130-fc1-prod-exact.json; cd ~/dsv41-bench && timeout 900 python3 exact.py /tmp/g130-fc1-prod-exact.json g130-fc1-prod > /dev/null 2>&1; python3 -I -c "import json;c=json.load(open(\"/tmp/g130-fc1-prod-exact.json\"))[\"combined\"];print(c[:16],\"OK\" if c.startswith(\"a8c3394af9d28249\") else \"MISMATCH\")"'

Only after that OK, release the lock and log it. A lock this runbook took is removed; the goal 130 lock stays, and the log line says which happened. The `cmp` keeps it from removing a lock that is not ours.

    ssh -n spark-one 'cat ~/dsv41-opt-20261021/CLUSTER_LOCK' | cmp - /tmp/fc1-lock.txt && ssh -n spark-one 'D=~/dsv41-opt-20261021; if grep -qx "purpose=g130 fc1 window" $D/CLUSTER_LOCK; then rm $D/CLUSTER_LOCK; S="lock released"; else S="goal lock kept"; fi; echo "$(date -u +%FT%TZ) | agent=claude | action=g130 fc1 window done, production restored, exact a8c3394af9d2, $S; traffic may resume | launcher_after=$(sha256sum ~/dsv41-tp8.sh | cut -c1-8)" >> $D/AGENTS_LOG.md; echo $S'

On exit 2, a mismatch or a missing OK, keep the lock and tell the user: the service is not verified, so nobody else should start on it.

## Reasoning on exactness and races (nothing tested on a GPU)

**Value argument.** Each output element comes from one tile's accumulators, which start at 0, walk k-blocks in order on that tile's A, B and scales, and then store. A tile's values do not depend on which CTA runs it or on the order of tiles. The deal changes only the (m, n) sequence of a CTA, and every tile still runs exactly once (`tests/deal_host_test.cpp`, G 0..420). SASS: same 8 QMMA with operands matching up to register names, and the 85 instructions of the stage math window have identical opcodes in both arms.

**Walker agreement.** The TMA leader, pf cursor, LA warp and 8 math warps must produce the same tile list, or barriers hang. All call the same pure functions on the same layout words, each with its own iterator, and those words are stable during the call.

**Layout timing.** `find_G` runs after `cudaGridDependencySynchronize` (checked in `test_cpu.py`), the same rule the live walker's per-tile layout read depends on. It adds reads, not a new kind of read.

**Prefix contract.** `grouped_layout[m*32] >= 0` iff m < G. Both layout builders (ep_scatter_1, mfront plan + scatter) lay 32-aligned segments contiguously from row 0 over a -1 fill, and `test_cpu.py` checks the rule on random routes. The live D04 stop already relies on it for full groups. The deal relies on it for all G, a wider use, so the recorder checks it on real routes. If it ever failed, tiles would be lost, not reordered, hence the KILL.

**Stale smem.** Junk A rows stay stale under `la` as they do today. A CTA may now see m-blocks of any residue. Junk outputs are never stored (`ds` stores live rows only), which the direct test checks by requiring dead rows to keep the poison.

**Co-residency.** Grid, smem, thread count and registers are unchanged (REG 168, SHARED 1024, STACK 8). CTAs past n exit earlier, as the zero-tile CTAs do today at small G. Any overlapped neighbour sees fewer busy SMs at the end of FC1 and more in the middle. The direct test does not exercise this, so the apply phase, the race and sync runs and exact.py on a serving instance carry it. `DSV41_L2_PREFETCH_MHC=0` does not turn off the mHC overlap: `DSV41_MHC_OVERLAP=moet` stays on in production, so its side stream can still co-reside with neighbouring kernels. A co-residency stress run against FC1 was not built: at promotion, rerun the pattern from w6's `stress_phase` (12,000 calls per overlapped kernel beside routed MoE, bitwise vs solo).

**Not checked by the SASS gate.** That values are equal. Only step 3 and step 5 can show that.

## Rider: scale byte selector, not built

The gr8 lesson saves about 26 ns per stage at the floor (F 396 to 370), and only there. On prof4 it is worth 0.018-0.024 ms per step on the live walk (`results/reprice-prof4.txt`, about +0.1% G1), and 0.000 on top of the deal, whose final round stays above the floor at G >= 3. It would need a new asm form with its own exactness proof, so it is not worth a tree. Revisit it only if the deal dies and A4/A3 show the floor is issue-bound.
