# r1-pf adversarial review (review-pf, 2026-11-08)

## Verdict

**Kit: ship. It is safe to install 8/8.**

| lever | verdict |
|---|---|
| G13 | ship |
| G15 | ship |
| G11a | ship |
| G29 | fix before its arm is run |
| G01 | drop |

Production changed under the kit while it was being built. At 13:40:58Z today the launcher went from fda6d05a to **08a249da** on 8/8 Sparks. That change is climb-w1:
- `PACE_KINDS=ar,attn,ag`, so all-gather and attention windows are now paced too;
- `WOB_MB=6`.

The kit, its README and its tests were written against fda6d05a, where `PACE_KINDS=ar` and `WOB_MB=0`. This does not affect flag-off identity, which I re-proved under 08a249da. It does affect G01 and G29: see below.

## What I verified myself

**Live base**
- `dsv41_l2_prefetch.py` sha256 is `63737f9c413ebfa2…` on spark-one..eight (read-only `sha256sum`).
- Neither `dsv41_l2pf_r1.py` nor `.bak-r1-pf` exists on any node. The existing backups are `.bak-g1-stepovh`, `.bak-g2-colltrim` and `.bak-pd-l2pf`.

**Kit files**
- base copy: `63737f9c…`
- patched: `49f27151b70e204e…`
- new module: `42c0f7b2e7e40cbd…`
- These match KIT.json.

**Diff**
- 4 hunks, 16 lines, all tagged `# r1-pf`.
- Every line inserted in a function body sits behind `_R1 is not None`.
- The block at the end of the module imports nothing unless `ENABLED` and a master flag are both set.
- `_env` is pure (`os.environ.get().strip()`).
- With flags off, the only new state is two module globals (`_R1`, `_R1_FLAGS`).

**Installer, run on a byte copy of the live spark-one base (cat over ssh)**
- `dry` lists 16 lines.
- `install` takes 63737f9c to 49f27151, and the module lands as 42c0f7b2.
- Stripping the tagged lines gives the base back byte for byte.
- `install` a second time reports "already patched".
- A tagged file that has been edited further is refused.
- `uninstall` restores 63737f9c.
- A drifted base (3130547d) is refused.

**Compile**
- Both kit files compile with spark-one's `~/venvs/vllm032/bin/python3` (3.12.14). The source was sent on stdin; nothing was written.
- Both prefetch `.so` files exist.

**Tests: rerun on 512S1 (exo-venv313, torch 2.11 CPU)**
- `tests/test_r1pf.py` passes 14/14, as claimed.

**Extra adversarial simulation under the 08a249da env** (scratch `512S1:~/climb-r1/r1-pf-review/r1-pf/adv.py`, `adv2.py`)
- **Flag-off:** live base and patched file give identical launch, wait and log sequences. This holds with sub-knobs set and the masters unset as well.
- **Each lever alone and all together:**
  - In every captured section (target 4/6; drafter stock 5/10 and greedy 5/10), every side-stream launch is preceded by `side.wait_stream(main)`.
  - Every launch is joined back to main before the end of the forward.
  - side1 is always joined before the next collective.
  - The base target launches are an in-order subsequence of the patched launches; r1 only adds.
  - The E-phase runs only in the eager step, on its own stream: one paced launch, unjoined.
- **Runtime hook targets** are present in the live refs:
  - `DSparkSpeculator._generate_draft` is in the class's own `__dict__` (speculator.py:557).
  - `capture()` passes `self._generate_draft`, which is looked up at capture time, after install.
  - `CudaGraphManager.capture` runs an eager warm-up and then the capture for each desc (cudagraph_utils.py:459-499).
  - `target_logits` looks up `certified_logits` as a module global and returns early while capturing (r4_small:824-856).
  - `_run_parallel_input_projections` holds the bare `torch.mm` on `compressor.fused_wkv_wgate.weight.T` (attention.py:947-975).
  - The compressor exists only when `is_kv_source`.
  - The attributes `_dsv41_r4_split`, `_dsv41_ctxkv`, `.parts`, `CTX_MODE` and `MAINPROJ_SPLIT` exist under the names used.
- **Compile mode:** production runs `CompilationMode.NONE` and `FULL_AND_PIECEWISE` (relfail rank0 log), with no breakable cudagraph. Python hooks and wrappers therefore run eagerly and are not traced by dynamo.

## Flag-off identity: holds

- Removing the tagged lines gives the base byte for byte.
- The simulated launch, wait and log sequences are identical under fda6d05a and 08a249da envs.
- With no master flag set, nothing is imported or wrapped, no stream is created and no table is allocated.

## Exactness when on: holds by construction

- The levers add only `dsv41_l2pf` and `dsv41_l2pf_paced` hint kernels on side streams. Their SASS, checked by the G01 verifier, does no stores or atomics.
- Main-stream kernels, their inputs, order, tactics, splits and verify width are untouched. The drafter wrapper only brackets the forward; drafts are unchanged.
- The remaining device-only risk is the known hint-CTA SM co-residency class. It is covered by the KIT device checks (exact.py, G1 digest, 9 long-prompt hashes, optional g5co).

## Races and capture: none found

- Every fork is a `wait_stream(main)` at the fork point.
- side1 joins at the next collective. side2 and side3 join at the end of the forward. The drafter bracket calls the base `_join/_join2/_join3` in `finally`.
- Segment tables are allocated at the eager learn step, kept alive and never rewritten.
- The E-phase is eager only (`is_current_stream_capturing` check plus `S.sig is None`). Its stream is created once, outside capture.
- Its table is built once. That build is one H2D copy with a host sync, at the first E-phase call; it costs time only.
- The drafter plan key `("r1-dspark", greedy, nreq, n)` cannot collide with the target key `(id(model), n)`. The two DSpark graph sets get separate plans (`_sharded_markov` is swapped around both the greedy capture and propose).

## Lever findings

### G01 (R1_WOB): drop

- Under 08a249da the default is `DSV41_L2_PREFETCH_WOB_MB=6`. `bind()` therefore always logs `INACTIVE: ... WOB_MB > 0` and the lever adds nothing. I confirmed this in the simulation.
- Its window arms (`G01-wob-inorder`, `-paced160`) would burn window time with no effect.
- A meaningful A/B would need `WOB_MB=0` in the treatment, which reverts the promoted climb-w1 W1b. That is not a valid single-flag arm.
- The in-order fallback has been superseded by the promoted paced attention window.

### G29 (R1_IDXTAIL): fix

Under 08a249da `"ag" in PACE_KINDS`, so the tail becomes a **paced part on side stream 3**, not the designed second burst on side1 joined at the post-attention all-reduce. The `adv2.py` launch sequence for an indexer layer shows the consequence, all on side3:

1. the base AG window, paced, 6.0 MiB;
2. the IDXTAIL tail, paced, about 4.3-4.4 MiB;
3. at the indexer wq_b post-hook, the promoted attention WOA window: wo_a + wo_b, paced, 9.28 MiB.

Side stream 3 is a FIFO, so the tail (about 36 µs at 125 GB/s) now delays the promoted wo_a/wo_b prefetch in all 8 indexer layers. It also cannot start until the paced head finishes, about 50 µs after the AG. It will likely land after the indexer wq_b read.

Other parts of the kit no longer match:
- `IDXTAIL_SPLIT` no longer means "CTAs": the pacer is a single warp.
- The README, KIT.json and nsys expectations ("2nd burst on side stream 1, 2 CTAs, joined at post-attention AR", "+0.13/+0.28/+0.39%") describe fda6d05a only.

The arm can regress the +3.8% climb-w1 gain. Before it is run:
- either re-model and re-plan G29 for paced AG windows (for example, a burst tail on side1, or placing it with explicit regard to the attention window);
- or make `bind()` refuse it when `"ag" in PACE_KINDS`, and say so.

### G13 (R1_DRAFT): ship

- Hooks, plan keys and E-phase gating are correct against the live refs.
- Under 08a249da the drafter attention window now also carries wo_b (base `_woa_segments` uses `WOB_BUDGET=6 MiB`). That is consistent with the target.
- Device confirmation is needed that the drafter's collectives go through the RdmaAllReduce `before_collective` hook. Otherwise `S.idx==0` and no plan is made; that costs speed only and shows up as missing `R1_DRAFT plan` lines.
- A `make_drafter_plan` exception at warm-up would crash the launch rather than fail closed. This is a nonblocking robustness item.

### G15 (R1_SHW2): ship

- The burst post-fork on side1 after the post-attention all-reduce is joined at the next collective. Its design is unaffected by climb-w1 (`post` is not paced).

### G11a (R1_KVSRC): ship

- The paced part is appended after the AR window's base paced part on side3. Recording goes to a side list and skips the drafter.
- Long-prompt hashes are listed in the device checks.

## Installer and composition (nonblocking)

**`uninstall` restores the backup blindly.**
- It does `os.replace(BAK, BASE)` without checking that the live file still equals `patch(BAK)` or `patched_sha256`.
- I demonstrated this: I appended a foreign line to the patched file, and `uninstall` silently discarded it.
- Fix: refuse unless the current sha equals `patched_sha256` (or `TAG` lines strip to the backup), unless `--force` is given.

**The new module is not covered by preflight or the fingerprint.**
- `dsv41_l2pf_r1.py` is in neither the supervisor manifest nor the rdma fingerprint (that hashes only `dsv41_l2_prefetch.py`).
- `status` should compare the module's sha with `kit_sha256` so the lead can check 8/8.
- The installer checks the source sha before copying, so the risk is low.

**Stale production facts.**
- `KIT.json depends_on` names launcher fda6d05a and `PACE_KINDS=ar`. It should be 08a249da: `PACE_KINDS=ar,attn,ag`, `WOB_MB=6`, `PACE_CHUNK=65536`.
- The harness `PROD_ENV` is also fda6d05a.

**Pinned base.**
- The base sha is pinned, so r1-pf cannot stack with any other kit that edits `dsv41_l2_prefetch.py`. No other climb-r1 kit does.

**r1-drafter.**
- r1-drafter source-patches speculator.py and r4_small, with `certified_logits` taking extra args. That composes with r1-pf's runtime wrappers, which use `*args, **kwargs`.
- Its `MAINPROJ_HOIST` and `DRAFT_PROLOGUE_FORK` add drafter streams and overlap G13's E-phase target (main_proj). The combination is untested; A/B them separately.

**r1-attn IDXPAR.**
- Its note about the wq_b post-hook forking from `current_stream()` also applies to r1-pf's `fork_part`. Run it separately.

## Fixes (fix-pf, 2026-11-08)

Production was re-checked read-only first: launcher `08a249da` on spark-one..eight, the live `dsv41_l2_prefetch.py` still `63737f9c413ebfa2…` on 8/8, and no r1-pf files or `.bak-r1-pf` on any node. The kit now targets 08a249da. New hashes:

| file | sha256 |
|---|---|
| patched `dsv41_l2_prefetch.py` | `8bc9a1b5b73c808580a350fe8618fd6abb34a6f629de11c2053f640a7cd15596` (was `49f27151…`; the only change is the tagged `_R1_FLAGS` line 1176) |
| `dsv41_l2pf_r1.py` | `5af558fe96edfcd630caa3e673b71788587649a4e0bf9b45b4e8963b39e10548` (was `42c0f7b2…`) |

Tests: `tests/test_r1pf.py` **17/17 PASS** on 512S1 (was 14), and `tests/installer_cycle.sh` on a byte copy of spark-one's live base. Both kit files compile with spark-one's Python 3.12.14 (stdin). Full output is in `TEST-RESULTS.txt`.

### Blocking: G29, tail on pacer stream 3 under paced AG windows → refused on 08a249da

- `bind()` now refuses G29 when `"ag" in PACE_KINDS and PACE_GBPS > 0` (`dsv41_l2pf_r1.py:146`). It logs `INACTIVE: {"idxtail": "built for burst all-gather windows; with 'ag' in PACE_KINDS (launcher 08a249da) the tail would queue on pacer stream 3 ahead of the paced wo_a+wo_b window"}` and the plan is unchanged.
- The paced branch is gone from planning. `_target_adds` only emits the tail as a burst (`"b"`, side stream 1, joined at the next collective), which is the configuration it was designed and modelled for. So `IDXTAIL_SPLIT` again means CTAs whenever the lever can be active.
- I did not re-plan for paced AG. Both candidates are unproven offline. A burst tail on side 1 would run ahead of the paced head, out of read order, and burst during the all-gather, which is what climb-w1 removed. A tail on side 3 delays the promoted wo_b prefetch, as you showed. Either needs an 08a249da nsys trace of the indexer layers.
- Status `skipped` for the current production. Window arms `G29-idxtail` and `G29-idxtail-4cta` are removed from KIT.json and listed in `window_arms_removed`. Expected gain on 08a249da is 0.
- README, KIT.json and the nsys expectations now describe both cases: refused on 08a249da, and the side-1 burst on fda6d05a.
- Tests:
  - `test_g29_refused_under_paced_ag`: under 08a249da and under `PACE_KINDS=ar,ag`, the full launch and wait log is identical to flag-off, with an INACTIVE line naming idxtail.
  - `test_g29_idxtail` and `test_g29_pure_rules` now run under fda6d05a.

### Blocking: G01 always INACTIVE under WOB_MB=6 → dropped

- The code is removed: `wob_tails`, `_subtract`, the `WOB*` constants, the `bind` branch and the `augment` branch.
- `DSV41_L2_PREFETCH_R1_WOB` is removed from the base gate `_R1_FLAGS`. Set alone, it imports nothing (tested). Set next to a live lever, it changes nothing: the launch log is identical to that lever alone, and the `levers` line names it `"dropped lever (G01); flag ignored"` (tested).
- The `DSV41_L2_PREFETCH_R1_WOB*` flags and both G01 window arms are removed from KIT.json. Status `skipped`, gain 0.

### Non-blocking items

| item | fix |
|---|---|
| Stale fda6d05a facts | KIT.json `depends_on` and the README name 08a249da (`PACE_KINDS=ar,attn,ag`, `WOB_MB=6`, `PACE_CHUNK=65536`). `harness.PROD_ENV` is now 08a249da; `PROD_FDA_ENV` keeps fda6d05a. Flag-off identity is tested under 7 environments, both launchers included. The G13, G15, G11a and all-levers tests run under both launchers. |
| `uninstall` restored blindly | It now checks both files before touching either. The base must equal `patched_sha256` or patch(backup), and the module must equal `kit_sha256`. Otherwise it refuses with nothing changed, and `--force` overrides. If the base already equals the backup, it only removes the backup. `install` now checks the module sha before writing anything; before, a bad module left the base patched without its module. Tested in `test_installer_cycle` and `installer_cycle.sh`: foreign line → refused, file intact; edited module → refused, nothing touched; `--force` → restored; plain cycle → `63737f9c`. |
| Module outside the manifest and fingerprint | `status` prints every sha with what it matches and ends with `r1-pf status: INSTALLED / CLEAN / INCONSISTENT`; it exits 2 on INCONSISTENT. It is a KIT.json device check for 8/8. |
| G13: `make_drafter_plan` exception fails the launch | `_draft_finish` is wrapped (`dsv41_l2pf_r1.py:709-714`). On error, `draft_graph` is turned off, that set's plan is dropped, and one warning is logged. Later warm-ups and captures run unbracketed. Target planning fails closed the same way: `augment_target_g2` builds every table before attaching any part, and on error it turns idxtail, shw2 and kvsrc off and returns the base g2 untouched. `test_fail_closed` covers both: the launch log equals flag-off, `S` is clean, and exactly one warning is logged. |
| G13: drafter collectives must reach `before_collective` | Documented as a device check and a risk (the sign is no `R1_DRAFT plan` lines; speed-only cost). |
| G13: E-phase H2D table build | Documented. It is one eager sync per process, outside capture, and costs time only. |
| Composition with r1-drafter (`MAINPROJ_HOIST`, `DRAFT_PROLOGUE_FORK`) and r1-attn `IDXPAR` | In KIT.json `conflicts_with` and the README window-arm notes: untested, A/B separately. |
| (found while fixing) G11a and side stream 3 on 08a249da | Side 3 now also carries the layer's paced qkv_a AG window. The G11a part (about 21 µs after the AR window's ~13 µs pacer) must finish before the input projections reach that AG fork. Expected yes (about 34 µs against at least about 40 µs), but not measured. Added as a device check and a README risk. |
| Your stream checks | `adv.py` is now `test_stream_protocol_every_env_and_lever`: every lever alone and all together, both launchers, every captured section. The side-1 rule is "joined at the next collective", because the base's own burst AG windows are pending across their collective by design. The checkers are self-tested on synthetic faults. |

Live window arms on 08a249da: G13-draft, G13-graph-only, G13-eph-only, G15-shw2, G11a-kvsrc. The kit total there is +0.13 / +0.8 / +1.6% G1.
