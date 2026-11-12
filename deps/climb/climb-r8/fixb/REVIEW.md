# r8-fixb adversarial review (fixb:review, 2026-11-09)

**Verdict: SHIP to the device window, then to the serving window.** I found no blocking defect. The installed change
is one file, `dsv41_step.py` ebe2f7bb → 425c6515. It is inert when the flags are off and exact when they are on. The
items below are cheap fixes. The most useful ones should land before the windows: N1, N2, N3 and N5.

## What I checked myself (read-only on the cluster; light CPU on the Mac)

| check | result |
|---|---|
| live shas on 8/8 (ssh sha256sum, read-only) | `dsv41_step.py` ebe2f7bb, `engram.py` e99f623d, `dsv41_r4_small.py` ebe73cdb, launcher 4f3cc1c6, no `dsv41_step.py.*` strays. These match install.py's preconditions and KIT.json's base pin. |
| `ops/fleet.sh status` (live, read-only) | CLEAN 8/8 |
| `install.py pins` | 25/25 OK. TEST-RESULTS says 20/20, so the pin set grew after that file was written. |
| r6 → r8 module diff | Only the PACK pairing changed (`_PackState`, `_pack_ok` docstring, `_ident`, `embed`, `wkv_first`/`wkv_second`). COLEMB, C4, the flags and the entry points are textually unchanged. |
| flags-off import path (composed r5r6 copy) | `engram.py` imports `dsv41_step` only if COLEMB or PACK is set, and `dsv41_r4_small.py` only if C4 is set. r7-eager names the module in `_TRANSPARENT_MODS` and uses only `_capturing()`, which still has its 4 call sites. Flags off, the swap is unreachable. |
| `integrate/compose_fixb.py` (copies in the scratchpad) | PASS 78/78 |
| `integrate/test_fleet.sh` (local transport) | PASS 10/10 |
| GPU device tests, serving arms, any install | **not run by me.** They need a window. |

## Exactness when on: the r8 PACK pairing

- Every new condition only **refuses**: `fa(x)` or `fb(x)`, the stock forward. A refusal cannot change a value.
  - If `wkv_first` refuses, `PK.kv` is never set, so `wkv_second` runs the stock path too.
  - If `wkv_second` refuses after a pack, layer a already holds the correct kv1, and layer b recomputes the stock kv.
- Ranks cannot diverge in their collectives. The refusal tests compare tensor identity, which is the same structural
  fact on every rank; they do not compare address values.
- The identity holds on the live path. `Engram.forward` under r5-prio e99f623d (flag off) calls
  `self.wkv(self.embed(h).flatten(-2))`. `n == tp*H` (refusal rule), so `r1[:, :n]` is a full contiguous view, and
  `flatten` keeps its `data_ptr`. The stored rows stay referenced by `PK.rows`, and then by the returned view, until b's
  wkv call. So pointer reuse cannot alias a foreign operand.
- Graph replays: the decisions are made at capture time as in r6. Under a breakable capture `_pack_ok` refuses, and
  `_breakable_capture()` is unchanged. Nothing new crosses streams.
- The prefill-size claim holds for both levers in the code.
  - COLEMB: `T > max_t` → `orig(self, input_)` on the same tensor. The one-time table build happens at the first call,
    eagerly, never under capture.
  - PACK: `_pack_ok` is false → `orig_embed`. The state is reset at layer a.
  - The det loop runs `dbgrun.py ... cold`, so prefix-cache tails (a prefill with T ≤ 128) are not part of r6-w1's
    mismatch.
- **Overclaim (wording only).** README §3.1 says "C4 is the only lever in s1 with cross-stream data flow". PACK also
  consumes prefetch-stream rows on main. It waits on the same events as stock, but earlier (both layers' events at
  layer a). What the code supports is "no cross-stream flow beyond the stock lookup→main edge".

## Device-test realism

- **step det gate.** It uses the live `Engram` prefetch methods, the live `_engram_lookup_thresholds` and one shared
  prefetch stream (this matches serving: `prefetch_stream` is passed to `__init__`). It poisons the staging rows and
  sweeps 5 timing orders. The mutants are caught at every T they apply to on the CPU latest-scheduler simulation. That
  is a real gate for the ordering claims.
  - **Gap 1: the collective and the GEMM are emulated** (an XOR-concat AG and a matmul wkv). A race inside the real
    rdma AG at the new payloads would not be seen: [T,6,256] and [T,2·400] stay under `RDMA_AR_MAX_BYTES` 2 MiB, so they
    take the same backend as stock. Nor would a race involving DTGRAPH, MHC_OVERLAP or the l2pf side streams. Only the
    serving det on the solo arms can clear PACK and COLEMB, and the README says so.
  - **Gap 2: no graph capture or replay** in the det gate. r6's parts covered capture.
- **Serving op.** `--source kit` tests the kit's bytes, which are pinned and equal to what install.py writes. Fine.
- **Env rule.** `prodenv.py` mirrors 107 exports, asserts GLUE_RQPAD=1 / RPSPLIT=split and has two must-fail controls.
  plan.sh refuses node lines without the rule. Good.
- **head.** The ksplit rewrite is a genuine reduction-order mutant, and its smem equals the r6 refine's (mutfit log:
  ALL PASS, and the negative control reproduces 147456 B). The per-cfg sweep skips only configs where the unmutated
  kernel itself is out of resources. It still has to run on a GPU; whether "caught at every cfg × M" holds for tiny M is
  a GPU question.
- **skew.** The T4c classifier keeps every cpufreq, cpuidle and nvidia-smi difference fatal. It explains only
  kernel-timer IRQs that moved in an A/A control, IRQ directories created with the default mask, and removed IRQ
  directories. T4d adds a pre-action block. This is sound. Two small holes are in N6.

## Composition

- With r5 + r6 (the live state) and r7: proved on copies, and I re-ran it (78/78).
- r7 touches none of `dsv41_step.py`, `engram.py` or `dsv41_r4_small.py`.
- **Operationally it is not commutative.** climb-r6 and climb-r7 `install-all.sh` judge 425c6515 as FOREIGN and refuse.
  See N2.

## Non-blocking findings (fix order)

- **N1. fleet.sh lock check is owner-only.**
  - `lock_ok` greps `owner=claude`. Right now spark-one holds an `owner=claude` lock for a *different* window (r7-ht1
    ht1cap, 17:19Z), so `fleet.sh install` would pass today.
  - The r7 scripts pass the window's token. Do the same here: require `token=$DW_TOKEN`, or a `FIXB_LOCK_TOKEN` exported
    by device-window.sh or climb-window.sh.
- **N2. The device window auto-installs r8 8/8 (`fixb-install`), and serve.sh leaves it installed.** Both conflict
  with the r7 windows being run today: their `install-all.sh` preflight sees FOREIGN and aborts the window, safely but
  at the cost of a slot.
  - Drop `fixb-install` from the device plan. serve.sh's KIT_INSTALL installs it anyway, and the install is idempotent.
  - Make serve.sh uninstall after the restore unless a promotion is decided.
  - Or have the r7 integrator accept 425c6515 as an r6-step state.
- **N3. PACK=all can now degrade silently to look.** With r8, a refused identity check means no wkv pack, and nothing
  in serving reports it. The plan-time boot line ("share one rows gather and one wkv gather") prints before any
  refusal can happen.
  - Log the first `pack_wkv_refused`, or dump `COUNTS` once after N forwards.
  - Have the p arms check `pack_wkv > 0`, `pack_wkv_refused == 0` on 8/8, so the p arm measures what it is labelled.
- **N4. The control arms run no det.** A mismatch on e/p cannot be told apart from a rare production flake within the
  same window. Add `c3` to `DET_ARMS` (about 6 min).
- **N5. A must-fail mutant performs the forbidden action if the guard fails.**
  - The T4d audit mutant and `test_audit_blocks_and_records_mutations` run a real `nvidia-smi -lgc 1000,1000`, and
    TEST-RESULTS says these functions ran on spark-eight while it was serving.
  - It was blocked, and non-root `-lgc` normally fails anyway. Still, the owner rule forbids clock changes. Use a
    sentinel instead: a fake `nvidia-smi` early in PATH in the tmp dir, or a unit test of `_cmd_ok`.
  - Also note that the shim in spark-eight's `/tmp` was outside `~/climb-scratch/<kit>/`, which the brief does not
    allow.
- **N6. Skew edge cases.**
  - (a) `classify` compares nvidia-smi output per field with `zip`. If the field counts differ (for example an error
    string before and values after), the trailing fields drop and `all([])` comes out "explained". Emit an UNEXPLAINED
    row whenever the per-field compare finds no difference or the lengths differ.
  - (b) `auditrun` lets `os.setpriority(PRIO_USER|PRIO_PGRP, 0, …)` through, because `who=0` counts as own. That call
    would renice the vLLM workers. Block it unless `which == PRIO_PROCESS`.
- **N7. `tests/run_cpu.sh` cannot fail the head tests.** `... | tail -1` has no `pipefail`. Add `set -o pipefail`, or
  grep for the PASS line.
- **N8. Install edge cases.**
  - A failure between the backup copy and `os.replace` leaves that node in FOREIGN (r6 bytes plus a stray
    `.bak-r8-fixb`, possibly `.tmp-r8-fixb`). fleet.sh's rollback `uninstall` then refuses on that node. Serving is
    unaffected, but the r6 and r7 integrators then refuse. Let `uninstall` remove a stray backup or tmp file when the
    live file is exactly r6.
  - In `climb-window.sh` the identity arm logs det but gates only on digest and exact. That does not matter here,
    because flags-off is unreachable, but the serve.sh comment says the 9 cases "prove" identity.
- **N9. Doc nits.**
  - The module docstring for PACK still says "one persistent staging buffer so the packed operand needs no copy". The
    code does a `torch.cat`.
  - `device-window.sh` defaults to EXPECT 08a249da. The plan passes EXPECT, but call it out in the README.

## Estimates

- I re-checked them against `scripts/c3_rebase_census.txt`. The p5 numbers match the table: emb AR 24.7 / 16.1 /
  24.9, drafter AR 25.8–26.2, rows AG 15.5–18.2, wkv AG 18.2–24.0 µs.
- Saving only the transfer time is right when the replacement is still a collective, because the skew wait moves to
  the next collective.
- The COLEMB 0.04 / 0.09 / 0.13 % and PACK −0.03 / 0.07 / 0.13 % ranges are consistent with the one serving data point:
  r6-w1 s1 − k1 = +0.32 tok/s (+0.25 %) against ~0.18 % control spread.
- HEAD and ISO keep r6's ranges, which is reasonable.
- Only the C-T-C-T decides.
