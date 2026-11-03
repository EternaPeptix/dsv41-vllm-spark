# BS03 stopped-rank window

This window answers one question: does `det_topk` match the live decode top-k chain bit for bit, and is it fast enough on GB10?

Nothing in this kit has run on a GPU yet. Every time and every number below is an estimate until `test_det_topk.py` prints its verdict.

- **Kit:** `spark-two:~/dsv41-stage/bs03/`. The Mac mirror is the path this file came from.
- **Node:** the test runs on spark-two only and needs no model. The service must be down on all 8 nodes, because TP8 cannot lose a rank.
- **API downtime:** about 35-65 min, estimated: quick run 3-5 min, full run 30-60 min, relaunch about 2.5 min.
- **Live files:** the test changes none. It writes only under `~/dsv41-stage/bs03/out/`, including its own Triton cache.

Commands run from the Mac over Tailscale. Plain `ssh spark-one` times out off the LAN.

## 1. Before stopping anything

These are safe beside the running service. Neither uses the GPU.

```sh
ssh spark-two.<tailnet>.ts.net 'cd ~/dsv41-stage/bs03 && sha256sum -c SHA256SUMS --quiet && echo KIT_OK && python3 -I bs03_install.py plan | head -1'
```
Expect `KIT_OK`, then `base shas OK; 3 anchors found once each; patched sparse_attn_indexer.py would be ebe2ba62 (expected ebe2ba62)`.

Optional, about 5 min at nice 19, no GPU: `ssh spark-two.<tailnet>.ts.net 'cd ~/dsv41-stage/bs03 && bash tools/cpu_check.sh'` should end in `CPU_CHECK_OK`. It runs the harness on a CPU mock, loads the extension, imports the live and patched files with the GPU hidden, and tries the installer on a throwaway copy. It stops itself if MemAvailable falls below 6 GB.

If you want a control for section 7, record it now, on a quiet API with the flag off. It takes about 4 min.

## 2. Stop the service

The supervisor has no stop-only mode, so this is its `stop_all` step. You hold `CLUSTER_LOCK`; log the window in `AGENTS_LOG.md` as usual.

```sh
ssh spark-one.<tailnet>.ts.net 'python3 -I ~/dsv41-w6ab/idlewait.py 30 1800'   # prints IDLE waited=...
for h in one two three four five six seven eight; do
  ssh -n spark-$h.<tailnet>.ts.net '
    timeout 30 systemctl --user stop dsv41-rank 2>/dev/null || { systemctl --user kill -s SIGKILL dsv41-rank 2>/dev/null; sleep 3; systemctl --user stop dsv41-rank 2>/dev/null; }
    systemctl --user reset-failed dsv41-rank 2>/dev/null
    pkill -9 -u $(id -u) -f "[v]llm serve" 2>/dev/null; pkill -9 -u $(id -u) -f "[V]LLM::" 2>/dev/null; true' &
done; wait
ssh spark-two.<tailnet>.ts.net 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; nvidia-smi --query-gpu=clocks.sm,memory.used --format=csv,noheader'
```
The compute-app list must be empty. The test refuses to start otherwise (exit 3).

## 3. Run the test on spark-two

**Quick run first.** It finds a broken launch or a hang before the long run.

```sh
ssh spark-two.<tailnet>.ts.net 'cd ~/dsv41-stage/bs03 && mkdir -p out && timeout -k 30 900 ~/venvs/vllm032/bin/python -I test_det_topk.py --quick 2>&1 | tee out/quick.log | tail -40'
```
- It prints the occupancy table first: `cudaOccupancyMaxActiveClusters` for C=16, 8, 4, 2, shared memory and registers. Save it. These numbers are unmeasured until now.
- Check `optin` in that table. Under 128 KiB is what the traces imply. At 128 KiB or more, `persistent_topk` takes other paths above 32 rows (FilteredTopK) and above 64 rows (sampled), and Moet reports wrong values from the sampled path on its GPU, so read a `STOP` at rows=128 with that in mind.
- It can end in `STOP` or `HOLD`, never `GO`. A `HOLD` that says "a --quick run never says GO" is the normal result.
- A `STOP` here means stop: do not start the full run. Bring production back (section 4), then see section 5.

**Full run.** It survives an ssh drop.

```sh
ssh spark-two.<tailnet>.ts.net 'cd ~/dsv41-stage/bs03 && nohup timeout -k 30 7200 ~/venvs/vllm032/bin/python -I test_det_topk.py > out/full.log 2>&1 &'
ssh spark-two.<tailnet>.ts.net 'tail -f ~/dsv41-stage/bs03/out/full.log'   # Ctrl-C leaves the run going
```
- Order: probe, smoke, timing (about 3 min), sweep, exact (most of the run). The exact stage prints one line per shape, and an elapsed-time line after each row count and width.
- The grid is what you asked for, plus rows 1, 12 and 16. Under the default map, rows 12 and 16 are the only rows on the C=2 path, and the hook would take all three.
- Results go to `out/<UTC>-full/results.jsonl` and `summary.json`. Up to 8 differing rows are saved as `mismatch-*.pt`.
- The last line is `VERDICT: GO`, `STOP` or `HOLD`. Exit codes are 0, 1 and 2; exit 3 means refused or a harness or launch error.

**If it hangs**, meaning no new line for 20 min (a stack dump every 20 min is normal):

```sh
ssh spark-two.<tailnet>.ts.net 'pkill -f "[t]est_det_topk.py"; sleep 5; nvidia-smi --query-compute-apps=pid --format=csv,noheader'
```
A cluster barrier that never releases looks like this. Treat it as `STOP`. If the process will not die or the GPU stays busy, reboot that node with `sudo -n /usr/sbin/reboot` before relaunching.

## 4. Bring production back

Do this whatever the verdict was.

```sh
ssh spark-one.<tailnet>.ts.net 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > /tmp/dsv41-relaunch.sh
shasum -a 256 /tmp/dsv41-relaunch.sh | cut -c1-8      # must print 7bea1cf3
# The supervisor calls plain `ssh spark-N`, so off the LAN it needs the Tailscale shim on PATH:
PATH=/private/tmp/claude-501/-Users-jeweled/e29a16fa-1457-4e89-9250-a8791bc6d2a7/scratchpad/ts-ssh/bin:$PATH \
  bash /tmp/dsv41-relaunch.sh --label bs03-restore-$(date -u +%m%d%H%M) --expect-launcher 62ac6866 DSV41_GPU_CLOCK_MHZ=2000
ssh spark-one.<tailnet>.ts.net 'bash ~/dsv41-opt-20261021/ops/rel/relstatus.sh'
ssh spark-one.<tailnet>.ts.net 'cd ~/dsv41-w6ab && python3 -I gunit.py ~/dsv41-w6ab/runs/check-$(date -u +%Y%m%dT%H%MZ) check exact2 --idle 30'
ssh spark-one.<tailnet>.ts.net 'cd ~/dsv41-w6ab && python3 -I gunit.py ~/dsv41-w6ab/runs/g1-$(date -u +%Y%m%dT%H%MZ) g1check g1 --idle 30 --reps 2'
```
- Expect exact.py `a8c3394af9d28249` and, over 2 repetitions, G1 digest `9a18f802d930f6e7`.
- Recreating the shim, if the scratchpad is gone: `bin/ssh` runs `/usr/bin/ssh -F <config> "$@"`, and `<config>` maps `Host spark-N` to `HostName spark-N.<tailnet>.ts.net`, then `Include ~/.ssh/config`.

## 5. Go / no-go

| Last line of the full run | What it means | Next step |
|---|---|---|
| `GO` | Weighted det_topk at or under 9 us/layer, alone and beside l2pf. No timed shape slower than the chain. 0 differing rows over the full grid at 200 replays. | Section 7: install flag-off, then the pinned A/B. |
| `STOP`: differing rows | The kernel differs from the live chain on some row. `chain==reference` / `det_topk==reference` in the log says which arm left the canonical function. If only `kind=denorm` rows differ and `chain==reference` is False, the live chain itself left the function on a denormal boundary, which real logits are not expected to reach. The test still says STOP; whether that case matters is your call. | Do not install. Bring `mismatch-*.pt` and `full.log` back for review. |
| `STOP`: weighted at or over 13 us | Too slow: below about +0.6% G1 by the gain skeptic's arithmetic. | Do not install. Check the sweep table first (next row). |
| `HOLD`: weighted 9-13 us | Exact so far, gain doubtful. | Read the sweep table. If another C is clearly cheaper at the G1 lengths, rerun with `--map`, for example `--map 5:4,11:4,22:2`. No rebuild needed, but run the whole test again: a new C is a new path, and `GO` needs smoke, timing and exact from one run. |
| `HOLD`: slower shape | Some timed shape lost to the chain. It prints as `(shape, det, chain)`. | A loss at 128K and above slows long-context decode, so that is a decision for you, not the script. If only rows 12 and 16 lose, `DSV41_BS03_MAX_ROWS=8` on the supervisor line keeps the live chain above 8 rows, where the 10-way merge does not run; the A/B then has to judge G4 on that basis. |
| `HOLD`: reference miss | det_topk and the chain agree, and both differ from the test's integer reference. | Treat as a test or domain question, look at the row. Not a go. |
| `HOLD`: incomplete | A stage did not finish, or `--max-minutes` hit. | Rerun the whole test. `GO` needs smoke, timing and exact from one run, so `--stages exact` alone can only end in `HOLD`. `--stages probe,smoke,timing,exact` skips the informational sweep. |
| exit 3 | Refused (service running, wrong base sha), a harness or launch error, or an input kind that lost what it promises. | Read the log. A launch error during `smoke` is a kernel problem, not a harness one. `--force` and `--allow-base-mismatch` skip a refusal, but such a run can only end in `STOP` or `HOLD`. |

What the numbers mean:
- **Weighted** is the G1 mix, using the code and prose row counts from the traces (rows 2: 22.7%, 4: 70.6%, 6: 6.7%) and lengths 300, 512, 700, 1150 with 57% of calls at 512 entries or fewer.
- **Beside l2pf** starts the real `l2pf_kernel` (2 CTAs, 64 threads, 6 MB per window) with the arm, so it is co-resident for the whole of a short arm. Production has it in flight at chain start on about 37% of chains, so this is pessimistic. The gate takes the worse of the two.
- **Slower** means `det > chain * 1.02 + 0.5 us`, to keep timer noise out. Strict comparison is in `results.jsonl`.
- The summary also prints the estimated saving per step. The chain figure should land near the 31 us/layer the traces show. If it does not, the graph setup differs from production; say so before trusting the saving.

## 6. What this test cannot tell you

exact.py and G1 cannot catch a wrong kernel here. exact.py never leaves the n <= 512 branch, G1 reaches n of about 1,163, and the tie rule is invisible to text hashes. That is why the int-view compare is the real gate. A `GO` is still a kernel-level result: it says nothing about the served stack until section 7 passes.

## 7. Standing long-prompt check

The rule (README, "New check for every prefill / long-context candidate", about 4 min): 9 cold probes, 4 repeats each, one at a time, on spark-one, with the API quiet. Every case must print `distinct_tok=1 distinct_lp=1`.

```sh
LABEL=bs03-control    # or bs03-arm
ssh spark-one.<tailnet>.ts.net "mkdir -p ~/dsv41-g5det/runs/bs03 && cd ~/dsv41-g5det && python3 -I scripts/dbgrun.py - runs/bs03/$LABEL.jsonl $LABEL big:2500,big:8000,big:32000,big:128000,code:4000,prose:6000,l3:B:T16-spam-reply:1,l3:B:T02-classify:1,l3:A:T01-reviews:1 4 0 cold"
```

That rule alone passes any deterministic kernel, including a wrong one. For BS03, also compare arms bit for bit. `tools/cmp_cold.py` needs only the stdlib, so copy it from the Mac mirror to spark-one's `~/dsv41-g5det/runs/bs03/`:

```sh
ssh spark-one.<tailnet>.ts.net 'cd ~/dsv41-g5det/runs/bs03 && python3 -I cmp_cold.py bs03-control.jsonl bs03-arm.jsonl'   # exit 0, COLD COMPARE OK
ssh spark-one.<tailnet>.ts.net 'python3 -I ~/dsv41-vision/tests/detref.py --corpus --expect 36 ~/dsv41-g5det/runs/bs03/bs03-control.jsonl'   # the stored references
```
- `prose:6000` is the decode-tie canary and `big:128000` covers the >16K path.
- The stored references (`det-ref-g5.json`) date from 10-27, and the README records no detref run after 10-28. If the control itself mismatches them, the control-vs-arm compare is what counts.
- These probes stop at 128K, so add one long prompt with the same recording: `big:500000`, 1 repeat, on both arms. Check the printed `pt`: the corpus may end before 500K tokens, and I could not check that here. A prompt that long takes minutes of cluster time.
- Run probes first after a launch, with the API quiet. A warm run never compares with a cold one.

## 8. Only after a GO: the A/B

None of this has been done. Site files change only with the service stopped.

1. **Copy the kit to the other 7 nodes**, from the Mac mirror: `rsync -a --exclude out/ --exclude refs/ --exclude live-base/ <mirror>/ spark-N.<tailnet>.ts.net:dsv41-stage/bs03/`.
2. **Install on 8/8, service stopped:** `python3 -I ~/dsv41-stage/bs03/bs03_install.py install`, then `status` on 8/8. Expect `sparse_attn_indexer.py ebe2ba62 bak-bs03 005785f9` on every node. It refuses on any other base sha or while vLLM runs. The supervisor manifest does not list these files, so `--expect-launcher 62ac6866` still holds.
3. **Flag off first:** relaunch as in section 4. The code path is unchanged, so expect exact.py a8c3394af9d28249 and the G1 digest.
4. **Flag on, one launch only:** add `DSV41_BS03_DET_TOPK=1` to the supervisor line. Check that the rank logs show `DSV41_BS03_DET_TOPK: det_topk in use` and no `live chain kept for` warning you do not expect. Rows above 128 keep the chain by design.
5. **Exactness:** exact.py and the G1 digest on every instance, section 7 against the control, and the cold compare on both arms.
6. **Speed:** a pinned ABAB on G1 against 62ac6866. The gain range is +0.5..+0.85%, and in the gain skeptic's simple noise model two pairs clear the 0.5% bar only 34-71% of the time at a true +0.55..+0.75%, so plan on more pairs or a bundle.
7. **Co-residency:** run the `~/dsv41-g5stress` gate if BS03 stays. It is a new cluster kernel, and a solo compare cannot see that failure mode.

**Rollback:**
- One launch: drop the flag from the supervisor line. The module is then not even imported.
- Files, service stopped, 8/8: `python3 -I ~/dsv41-stage/bs03/bs03_install.py uninstall`. It restores 005785f9 from `.bak-bs03` and removes the module.
