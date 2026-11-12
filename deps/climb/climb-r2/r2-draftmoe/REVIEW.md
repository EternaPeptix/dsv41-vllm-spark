# r2-draftmoe — adversarial review (G17, `DSV41_DRAFT_MOE_G17`)

Reviewer: review-draftmoe, 2026-11-08 ~14:20-15:00Z. Read-only on the cluster. CLUSTER_LOCK was present on spark-one
the whole time, so nothing was compiled or run on a Spark.

## Verdict: FIX (one blocking item, in the installer). The flag-off path is identical; the flag-on exactness argument holds but is unproven on hardware.

The lever code is sound. With the flag unset the executed path is identical to today. The flag-on argument holds up
element by element. The blocking problem is only in the installer: it cannot be installed together with r2-moefront
(either order), and its uninstall is not safe against order.

## What I verified myself

| check | result |
|---|---|
| Live base sha256 on spark-one (14:22Z, launcher 08a249da) | mxfp4.py 67339029, deep_gemm_moe.py 9364f121, dsv41_draft_m32.py a17d574a, deep_gemm_utils.py 58e264c8: all equal KIT.json and `base/`. sm121_target_m32.py 38fb644b and sm121_d05_verify.py c04c363d also equal `base/`. |
| Install into a scratch copy, then `diff -u` line by line | 9 / 11 / 6 / 17 tagged lines, as claimed. Every inserted statement is guarded by `_R2DM_FC1` or `bool(_R2DM_PERM)`. Every replaced condition is either `orig or (_R2DM_FC1 and …)` or `assert orig or (_R2DM_FC1 and …)`. One untagged blank line was inserted in deep_gemm_utils.py (before `# G17 perm`). |
| tests/test_install.py (Mac) | rerun: 37/37 PASS. This includes the flag-off AST fold for all 4 files and the r1-pdl-composed file, and Athena `transform_apply` on the patched apply. |
| tests/test_plan.py, tests/test_emulation.py, tests/test_device_harness.py (512S1, fresh rsync to ~/climb-r2/review-draftmoe) | rerun: 22/22, 14/14 and ALL PASS. The builder's TEST-RESULTS.txt claims are real. |
| Prejit source | live `38370f02…/kernel.cu` sha256 ed151295 (spark-one); the one-line N edit appears exactly once and gives fe5bee03 = KIT.json. |
| device_test.py | NOT run (no GPU). I reviewed it: it is sound (see below). |

## Flag-off identity: holds

- Each module computes `_R2DM_FC1, _R2DM_PERM = False, ""` from one env read at import. The new module is imported only when
  the flag is set. `os` is already imported in all 4 files. `_r2dm` is only referenced behind the flags (short-circuit).
- deep_gemm_utils.deepgemm_moe_permute adds one host-side local, `_r2dm_fp = bool("")` (False), before
  `expert_start_loc`. It is defined on every path that reaches the `verify` hunk, so there is no UnboundLocalError. No
  kernel, allocation, stream, event or graph-content change.
- Athena's runtime overlay execs the transformed `apply` in `dict(engine.__dict__)`, a snapshot taken after module
  import, so `_R2DM_FC1` and `_r2dm` resolve. Its 3 anchors still match once (tested). Its `try_launch` only fires for
  the 40 packed target layers (`_athena_fc2_packed`), so the drafter's new `compact=True` never reaches it.
- No circular-import risk: `fused_moe/experts/__init__.py` is empty (0 lines on spark-one) and the new module
  imports only os/threading at top level.
- None of the 4 files is sha-pinned by the launcher or the supervisor. The launcher pins only the Athena trial
  file (line 126).

## Exactness when on: argument holds, unproven on hardware

- **FC1 N 768→576 (M32 path):** the drafter instance has kNumGroups=128, so every dsv41 g1/deal feature is
  compile-time off. They all need `kNumGroups == 384` (live-ref header lines 148-165, 248), which I checked. The
  per-element chain (block-scaled m16n8k32, K-ascending, single accumulator from +0, one bf16 RN) does not depend on
  SHAPE_N or the tile config. The live default cache already serves the padded drafter with two different 128-row
  configs (see the prejit item), which is further evidence that the outputs do not depend on the config.
- **Weight load:** `_load_w13` narrows the checkpoint by `loaded_weight.shape/tp` (independent of the param size) and
  places w1/w3 at `shard = rows/2`. The fast loader (DSV41_FAST_LOAD) reads only those rank rows. Both are
  param-size independent, so the compact w13 rows equal the padded live rows.
- **SiLU writer:** the same Triton source. The compact branch masks pad columns with `other=0.0` and stores zeros up to
  `GROUPS_PER_ROW*128`. The grid (BM, packs_per_cta, grid_m) depends on M and expert_ends, not N. Bitwise equality
  needs the padded FC1 pad columns to be **+0**, not −0. A −0 would give FP8 0x80 vs 0x00. That holds by
  RN reasoning but is a hardware property. device_test asserts it, and nothing else can.
- **Permute:** the stock `transpose_and_pack` formula (smxx_layout.cuh:182-186) is byte-for-byte the scatter
  kernel's. `ep_scatter_1` writes layout=e only for live rows (lines 145-155), so the layout arrays are equal. Dead-row
  scale words are 0 in both. The only difference is the order of rows inside a segment (atomic order today too); FC1,
  SiLU and FC2 are per row; the gather goes through inv_perm. The gate is the target's with (128, 3); M≤160 gives
  ≤480 slots, under 1024.
- **AV / verify width:** pinned (`DSV41_AV_PROFILE=pinned` with av-pin-k5-g3FU.json, launcher lines 66-69), so a
  faster drafter cannot change the verify width.

## Races and CUDA graphs: no new hazards

No new streams, events or host syncs. The fused-permute allocations are on the same (drafter main) stream inside the
graph pool, which is the target's existing pattern. Workspaces are sized by K, so they are unchanged. Weights are
created before capture. The s16 shared-expert stream keeps its existing event join. Only overlap timing changes.

## Installer

OK: it checks every file before writing, refuses unknown bases, tags lines, writes `.bak-r2-draftmoe` once, is
idempotent, and uninstalls back to byte-identical files in LIFO order.

### BLOCKING
1. **It cannot be co-installed with r2-moefront, in either order.** r2-moefront patches the same
   `deep_gemm_utils.py` (same `_DSV41_FP_FILL_ROWS = 1024\n` anchor) and the same `deep_gemm_moe.py`.
   r2-draftmoe accepts only 58e264c8 or c3a05cca and refuses a moefront-patched file. r2-moefront strips only `# r1-pdl`
   tags and refuses ours. The texts do compose: I applied both `patched()` functions in both orders, all anchors
   matched once and both files parse. They are also semantically independent (moefront is E=384 only). The window
   protocol is "install all kits 8/8", so one of the two will refuse.
   **Fix:** accept a file whose lines tagged `# r1-pdl` / `# r2-moefront` strip back to a recorded base (as
   r2-moefront does). Tag or remove the one untagged blank line so that stripping is exact. Make uninstall strip our tagged
   lines, or at least verify `sha(file) == sha(patched(bak))` before restoring. Also declare the pair in KIT.json.
2. (Same fix.) **Uninstall is not safe against order.** It restores `.bak` blindly. Simulated (scratch) with r1-pdl's
   own logic: base 58e264c8 → r1-pdl c3a05cca → ours 4d032f79. If r1-pdl is uninstalled first, the file becomes
   58e264c8 and our edits are silently lost. Our uninstall then **restores c3a05cca**: r1-pdl lines remain after
   r1-pdl removed `dsv41_glue_pdl.py`, so production is not byte-restored. Until fixed, the README must require LIFO
   uninstall.

## Non-blocking (do these in the window)

- **device_test.py has not run.** It is the exactness gate: exit 0 on spark-eight, then `--real-weights` on every
  node, before any timing arm. I reviewed it. It compares FC1 live and pad columns (pad must be +0), FC2 input bytes
  and scales, FC2 rows and the MoE output per slot via each path's own inv_perm. It runs the M32 and 128-row paths,
  e2e through the patched `apply`, poisoned workspaces, and graph capture plus 4 replays with new inputs. It is not a
  false-pass design.
- **prejit is incomplete.** The live `~/.cache/vllm/deep_gemm/cache` holds two padded-drafter 128-row FC1 configs:
  5559d059 (`<0,768,5120,…,128 groups,128,128,…,2 stages>`) and 2201c753 (`…,128,64,…,3 stages>`). prejit
  rebuilds only 5559d059→3a6d4662, so a first long prompt can still JIT an N576 instance mid-serving. Run
  device_test (M=161…1024) on every node before relaunch, or add the 2201c753 analogue, and run `prejit-status` and
  `ls` the cache after the first long prompt. Never run prejit concurrently with a JIT-ing process
  (`rmtree(out)` race).
- No window arm for **G17 + r1-pdl (`DSV41_GLUE_PDL` 'fp' scope)**. With both flags on, the drafter permute goes
  through r1-pdl's PDL kernels, which were not qualified for E=128. Add an arm, or do not combine them.
- KIT.json does not record the new module's sha256 (caeb2f8c…). install copies the module *after* patching the
  files; r1-pdl copies it first. Uninstall leaves the prejit cache entries in place (harmless, but undocumented).
- If the lead collects `DSV41_WEIGHT_DIGEST`, the drafter w13 params differ with `fc1` on. This is expected.
- Gain: central +0.24% is below the +0.3% bar, so it may need stacking or long pairs.

## Per lever

G17: verdict **fix** (installer composition and uninstall only). Flag-off identical: **yes**. Exact when on: **unproven**
(strong argument; needs device_test exit 0, plus exact.py, the digest and the permverify shadow with the flag on).
Race-free: **yes**.

## Fixes (fix-draftmoe, 2026-11-08 ~15:00Z)

### Blocking 1: co-install with r2-moefront (and r1-pdl). FIXED on our side.
- **Pure insertions only.** `install.py` EDITS were rewritten so that every edit inserts lines ending in
  `# r2-draftmoe` and keeps each anchor verbatim. Nothing is replaced, so `strip_tag(patched(x)) == x` for any x.
  The three former replacements are now insertions:
  - The `apply()` w2 assert is a separate tagged block after the target's own asserts:
    `if _R2DM_FC1 and not compact_fc1 and geometry == DRAFT_COMPACT: assert w2 (128,5120,192); assert SILU; compact_fc1 = True`.
  - The `_workspace_m_alignment` condition is a tagged first statement:
    `if _R2DM_FC1 and compact drafter: return _workspace_m_alignment(M, 768, ...)`, which is exactly the padded
    drafter's decision.
  - The `_act_mul_quant` assert edit is gone. Instead, `drafter_compact_eligible` requires `DSV41_MOE_COMPACT == "1"`,
    so the original assert holds. It logs `NOT compacted ... DSV41_MOE_COMPACT=1:False` otherwise.
- **Blank line.** The untagged inserted blank line in `deep_gemm_utils.py` is removed.
- **Tags at line end.** The prelude comment now ends with the tag, so every tagged line ends with
  `# r2-draftmoe` and any kit that strips with `endswith` (as r2-moefront does) removes them exactly.
- **Strip-based base acceptance.** A file is accepted when its `# r1-pdl` / `# r2-moefront` lines strip back to the
  live sha in KIT.json. KIT.json records the composed shas: deep_gemm_utils r1 `c3a05cca`, mf `9ce425b5`,
  r1+mf `97f26924`; deep_gemm_moe mf `b2c3b833`. It also declares `composes_with`, `install_order` and
  `uninstall_order`.
- **Tests.** `tests/test_compose.py` (new, 33/33) runs the real r1-pdl and r2-moefront installers on scratch trees:
  - r1-pdl → r2-moefront → ours installs, and each composed file's flag-off AST equals the r1+mf AST.
  - LIFO uninstall restores every file of all three kits byte for byte.
  - S4/S5 cover moefront on top of ours.
- **Remaining limitation (r2-moefront's installer, not ours).** r2-moefront, as it is today, strips only `# r1-pdl`,
  so it refuses a file that already has our lines (S4: it refuses and changes nothing). Until it adds
  `"# r2-draftmoe"` to `ROUND1_TAGS`, the documented order is r1-pdl → r2-moefront → r2-draftmoe. With that one-word
  change (tested on a copy), either order works, and so do both uninstall orders. Both kits patch these files with
  pure insertions, so that is the only change needed.

### Blocking 2: uninstall is not safe against order. FIXED.
- **No restore from `.bak`.** `uninstall` no longer restores `.bak`. Per file, it:
  1. strips our tagged lines;
  2. requires the result to be an accepted base;
  3. requires the file to equal our patch of that result, up to the position of other kits' tagged lines at a
     shared anchor.
  Every file is checked before any file is written, and it refuses otherwise (test 3b: altered tagged line → refuses,
  nothing changed).
- **File without our lines.** It is left as it is (it reports whether it is an accepted base). Our `.bak` copies are
  informational, and uninstall deletes them.
- **Review scenario.** r1-pdl → ours, then a blind r1-pdl restore, then our uninstall: exit 0 and deep_gemm_utils
  ends at the live **58e264c8**, not c3a05cca (S3).
- **FIFO with today's kits.** r1-pdl's uninstall (it checks `sha == patched(bak)`) and r2-moefront's uninstall
  (strip + accept) both refuse while our lines are present and change nothing (S2). FIFO therefore cannot silently
  drop our edits, and our uninstall still gives the exact other-kits state.
- **Docs.** LIFO is documented in the README, the install.py docstring and KIT.json `uninstall_order`.

### Non-blocking
- **device_test.py:** not run, because I have no GPU and the brief reserves it for the lead on a stopped node. The
  README window protocol now makes this explicit: spark-eight exit 0, then `--real-weights` on every node, before
  any timing arm.
- **prejit:**
  - Third entry `dg128b` added: 2201c753 (128x64, 3 stages) → key `5a63770a66faeaa4f1203886b22d547d`, kernel.cu
    `fc936b78`. It was computed read-only on spark-eight; the source key reproduced itself.
  - The analogy also holds for the target's second config (4ea90a59 → 8135464c, a live entry).
  - Rmtree race removed: an entry that exists (committed or not) is never touched without `--force`; `--force`
    refuses while a vLLM process runs and renames the old entry aside.
  - The README asks for a cache check after the first long prompt.
- **G17 + r1-pdl:** added arm `g17+pdl` (`DSV41_DRAFT_MOE_G17=1 DSV41_GLUE_PDL=1`). Its gate is device_test.py
  with `DSV41_GLUE_PDL=fp` on an r1-pdl node; device_test calls `deep_gemm_utils.dsv41_fused_permute`, which r1-pdl
  hooks.
- **KIT.json module sha:** added `new_module_sha256` = `f83e055c…` (it changed from caeb2f8c because of the
  DSV41_MOE_COMPACT gate). install checks it and copies the module **before** patching any file. Uninstall leaves the
  prejit cache entries in place, and this is now documented.
- **DSV41_WEIGHT_DIGEST:** documented as expected for drafter w13 with fc1 on (README Risks, KIT.json `notes`).
- **Gain:** central +0.24% is still below the +0.3% bar. This is unchanged and documented (KIT.json `notes`).

### Tests rerun (TEST-RESULTS.txt, verbatim)
- test_install 39/39 (Mac).
- test_compose 33/33 (Mac).
- test_plan 24/24 (512S1). It adds the DSV41_MOE_COMPACT gate check and a documented assert case.
- test_emulation 14/14 (512S1).
- test_device_harness ALL PASS (512S1).
- device_test.py compiles but has not been run.
