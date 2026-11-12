# r8-fixb: repairs of the round-6 head, skew and step kits

Climb round 8, 2026-11-09. Production launcher is **4f3cc1c6**, G1 130.48. 1 % G1 is about 0.21 ms/step.

This kit fixes three round-6 kits, using the r6-dwA and r6-w1 window logs. It does not edit `climb-r6/`. The r6 kits
were copied here as `head/`, `skew/` and `step/` and changed in place.

**What has not run:**
- No GPU device test.
- No serving arm.
- No install on any Spark.

These need a stopped node or a cluster window. Every check that did run is listed in `TEST-RESULTS.txt`, with where
it ran.

| part | round-6 failure | fix | installed bytes change? |
|---|---|---|---|
| head | `ksplit` mutant needs 147456 B smem (limit 101376); the device test refuses the composed `dsv41_r4_small.py` ebe73cdb | `ksplit` rewritten; per-config mutant sweep (fit + caught); composed shas accepted | no (device test only) |
| skew | T4c "read-only snapshot differed", cause unknown | cause measured; every differing key printed and classified; write/exec audit added | no (device test + `node/auditrun.py` in the scratch stage) |
| step | s1 (C4 + COLEMB + PACK=all) differed on long prompts; COLEMB and PACK never served alone | race audit (none at prefill sizes); PACK=all pairing hardened; long-prompt determinism gate; solo arms | yes: `dsv41_step.py` ebe2f7bb → **425c6515** |

## 1. head (`head/device_test.py`, `head/tools/mutfit.py`)

**Cause.** The round-6 `ksplit` put two tile loads (two `tl.load` pairs) in each K iteration. Triton pipelines both
loads `num_stages` deep, which doubles the refine's shared memory:
- 73728 B → 147456 B at the live config (BK128, s3, BMX 128, M 96).
- The mutant part runs every M from 2 to 96, so it crashed at the first M ≥ 65.

**Fix.** The mutant is the same reduction-order change (two fp32 partial sums added at the end), written as two
sequential K-half loops, each with one tile load per iteration, then `acc = acc + acc2`. Its smem equals the unmutated
refine's at every config. It needs `(K // 2) % BK == 0`, which the harness asserts. K is 5120, so BK 128, 256 and 512
all divide 2560.

**Proof without a GPU** (`tools/mutfit.py`, spark-eight CPU, sm_121 compile). It compiles all 6 mutants against the
unmutated r6 kernel at every config the device test can run them at, 192 (mutant, config) pairs:
- refine: BMX 16 × BK {128, 256, 512} × stages {2, 3, 4}, plus BMX 32 / 64 / 128 at the live tile;
- screen: BM 16..128 × BN × BK × warps × stages;
- cand: every BMX.

Results:

| check | result |
|---|---|
| every mutant fits wherever its kernel fits | PASS |
| `ksplit` smem == the r6 refine's at all 12 refine configs (8192 … 98304 B; BMX 128: 73728 B) | PASS |
| negative control: the round-6 text at BMX 128 | still 147456 B, reproducing r6-dwA |

**New gate (`part_mutants_sweep`).**
- Every mutant must fit and be caught at every swept config × sweep M (BMX 16 / 16 / 32 / 64 / 128).
- Configs where the unmutated kernel itself does not fit are skipped. Such a config is never ELIGIBLE.
- The gated-M run at the live config (M 2..96) is unchanged.

**Composed states.** The serving-op check and the dispatch child now accept these `dsv41_r4_small.py` states:
- live 3364d8c8;
- r6-head c77844fb;
- **ebe73cdb**: r6-head → r6-step, the fleet's state. It is identical with and without r5, and no r7 kit touches it.

They refuse r6-step alone (d9bf5d59, no head lines), with a message.

**Environment rule** (`common/prodenv.py`, shared by all three tests):
- venv, CUDA_HOME, nvcc and ptxas from `/usr/local/cuda/bin`.
- The production launcher `~/dsv41-tp8.sh` must be `--expect` (default 4f3cc1c6, which is the live sha on all 8 nodes).
  Its 107 literal exports are mirrored into the test process, and `DSV41_GLUE_RQPAD=1 DSV41_GLUE_RPSPLIT=split` is
  asserted.
- Two must-fail controls: a stripped `env -i` interpreter, and a launcher copy without the glue lines.

## 2. skew (`skew/device_test.py`, `skew/node/auditrun.py`)

**Cause.** I measured it read-only on spark-eight while it was serving. irqbalance is inactive on spark-two, -four and
-eight. `/proc/irq/307` (the `arch_mem_timer` clockevent broadcast timer) changes its own `smp_affinity_list` about once
a second: 8 → 13 → 7 → 19 → 18 → 7 → 3 → 0 in 8 s. The kernel moves the broadcast timer to whichever CPU leaves deep
idle.

T4c compared whole snapshots taken ~5 s apart, so it fails at random on a correct tool. T3f has the same flaw and passed
by luck.

Two other ways a whole-snapshot compare differs without any write:
- IRQ directories appear or vanish when a driver requests or frees vectors. The nvidia driver does this when a CUDA
  context opens or closes, and T2's CUDA child and hostcensus's nvidia-smi run between the snapshots.
- The nvidia-smi line was compared as one string.

The A/A control and the classifier dry run on spark-eight show exactly one moving key, `/proc/irq/307`, classified
"kernel-timer, explained".

**Fix (no check weakened, one added):**
1. Every differing key is printed with its before value, after value and class.
2. A difference counts as explained only if one of these holds:
   - **(a)** the IRQ's `/proc/interrupts` action is a kernel clockevent timer **and** that key moved during an A/A
     control with no tool running (3 s before the test plus 1 s after);
   - **(b)** an IRQ directory appeared with the kernel's `default_smp_affinity`, so nobody wrote it;
   - **(c)** an IRQ directory vanished.

   Every other difference still fails, as before: any cpufreq or cpuidle key, any nvidia-smi limit field (now compared
   per field and named), or any other IRQ's mask.
3. **New T4d.** `hostcensus.py` and `iso.py plan --irq` run under `node/auditrun.py` (`sys.addaudithook` plus os
   wrappers). It records, and blocks *before they act*, all of these:
   - any write-open under /proc, /sys or /dev;
   - any exec other than read-only `nvidia-smi --query-gpu` and `systemctl is-active`;
   - affinity or priority calls on another pid;
   - an NVML dlopen.

   Zero blocked attempts is gated. This also covers the volatile keys.

**Mutants:**
- 4 audit mutants. These are tool copies that write-open an IRQ mask or the cpufreq governor, run `nvidia-smi -lgc`,
  or re-pin another pid. Each must be blocked and recorded.
- 4 classifier mutants. These are synthetic after-snapshots: a stable `scaling_max_freq`, a non-timer IRQ mask, an IRQ
  created with a non-default mask, and an nvidia-smi `clocks.max.sm` change. Each must come out UNEXPLAINED.

The audited real tools on spark-eight (serving, read-only) made 2 events: `systemctl is-active` and their own small-core
pin. Neither was blocked.

## 3. step (`step/files/.../dsv41_step.py` 425c6515, `step/det_gate.py`)

### 3.1 Can COLEMB or PACK race at prefill sizes? No: they are a pure pass-through there

Facts from the serving code (launcher 4f3cc1c6, rank logs, vllm032):
- Chunked prefill uses T up to `--max-num-batched-tokens 16384`. PREFILL_SP forwards have T ≥ 2048. On GB10,
  `_engram_lookup_thresholds` is (None, None), so every Engram lookup runs on the shared prefetch stream.
- **COLEMB** (`VocabParallelEmbedding.forward` hook): when `T == 0`, `T > ct.max_t` (128), the ids are not int or the
  ids are not CUDA, it returns `orig(self, input_)` on the same tensor. Above the cap it adds no kernel, collective,
  stream, event or allocation.
  - The column table is built once, eagerly, at the first call (the profile run), on every rank.
  - At decode sizes COLEMB is a single-stream lookup plus one column gather on the caller's stream, with nothing
    cross-stream.
  - The drafter's embedding runs inside the FULL drafter graph, which is the same at prefill and decode
    (num_reqs × query rows).
- **PACK**: `_pack_ok(T)` is `0 < T ≤ 128` and not inside a breakable capture.
  - Above the cap, layer a's embed resets the pairing state and calls the live `Engram.embed`. Layer b's embed finds
    nothing stored and calls the live embed. Both wkv calls are the stock forwards. So the only change is a Python
    counter.
  - At decode sizes, layer a's packed gather waits on **both** layers' live lookup events (`_ready_rows`). Both
    lookups were queued at the forward's start on the same prefetch stream (model.py, before `_run_layers`).
  - Layer b's next write (the next forward's prefetch) is ordered after main by `stream.wait_stream(current)`.

So neither lever can produce the s1 long-prompt mismatch at prefill sizes. C4 is the only lever in s1 with cross-stream
data flow (a side-stream GEMM reading the aux taps).

**The windows cannot attribute the mismatch:**
- c4a's det check ran with `DSV41_STEP_C4_AUDIT=2000`. During the audit window the **stock** y is served, so
  "det ALL==REF" on c4a does not clear C4 (r6 README §2.3).
- k1 (C4 alone) had no det check.
- C4 is dropped. Its premise, that the taps are graph-static outputs, is only argued for graph steps.

The solo arms below settle COLEMB and PACK.

### 3.2 What changed in the module: PACK=all pairing by operand identity

r6 paired layer 14's stored kv with its wkv call by T alone. `wkv_second` returned the stored kv to *any* operand with
the right T, and a second call of layer 1's wkv in one forward packed again. Neither happens in today's serving path:
Engram.forward is `wkv(embed(h).flatten(-2))`, and PREFILL_SP's shard self-check runs only at T ≥ 2048. But it was a
wrong-result hazard, not a race.

r8 changes:
- Records the identity (seq, data_ptr, shape) of each layer's packed embed output.
- Packs layer 1's wkv only on its own embed output, and only once per forward.
- Returns the stored kv only to the wkv call whose operand **is** layer 14's stored rows.
- Refusals go to the stock path and are counted (`pack_wkv_refused`).

A CPU test checks that COLEMB, C4, the AR map, the flags and the entry points are AST-identical to r6.

### 3.3 The long-prompt determinism gate (`device_test.py --part det`, in `all`)

**Shapes and timing:**
- Decode T: 1..128 (20 values). Prefill T: 129, 256, 1024, 2048, 4000, 8192, 13312, 16384.
- 5 timing reps per T, built from spin kernels. Each rep sets a different delay for the prefetch-stream lookup, the
  layer-0 stand-in and the layers 2–13 stand-in on main, so every ordering of lookup against consumer occurs.
- Staging rows are poisoned with a NaN payload before every forward, and COLEMB's ids buffer is poisoned with −1. Any
  unsynchronised read returns poison.

**PACK (look and all):**
- Two layers built from the **live vllm `Engram` class**: live `prepare_embeddings`, `_start_prefetch`, `_ready_rows`,
  `_finish_prefetch` and `embed`, on one shared prefetch stream.
- The forward is prepare(a), prepare(b), layer-0 work, `a.wkv(a.embed(h).flatten(-2))`, layers 2–13 work, then the same
  for b.
- Gates:
  - kit == stock bitwise in every rep;
  - decode T: the kit engaged (1 / 2 kit gathers; stock rows and wkv gathers 0/2 or 0/0);
  - prefill T: pure pass-through (0 kit gathers, pack counters unchanged, the stock 2 + 2 gathers ran);
  - PACK=all: a foreign operand gets the stock result.

**COLEMB:** the kit's own class hook on a stand-in class whose forward is the stock result, with a delayed ids producer
on main. Gates:
- kit == stock bitwise;
- decode T: lookup plus one column gather, and the stock forward not called;
- prefill T: the stock forward ran once on the **same** tensor, and the kit did nothing.

**Must-fail mutants** (each must be caught at every T it applies to):

| mutant | shapes | what it does |
|---|---|---|
| `pack-nowait` | decode | reads b's rows without b's event |
| `pack-cap` | prefill | PACK_MAX_T = 16384 |
| `pack-stale` | prefill | never clears pairing state and ignores (seq, T) |
| `pack-r6pair` | decode | the r6 module's T-only pairing |
| `colemb-side` | decode | lookup on a side stream not ordered after the ids producer |
| `colemb-cap` | prefill | max_t = 16384 |

**The CPU self-test** runs the whole PACK half on a *latest-possible scheduler* simulation, where a lookup executes
only when someone waits on its event:
- quick T set: 9 PACK gates pass and all 7 PACK mutants are caught;
- full T lists: pytest `test_det_sim_full_lists_pass_and_catch`.

COLEMB engages only on CUDA ids, so on CPU only its pass-through is checked. The colemb mutants run on the GPU.

### 3.4 Solo serving arms (`ops/serve.sh`, `ops/arms-fixb.txt`)

- **Arms:** `c1 e1 c2 p1 c3 e2 c4 p2 c5`. Every arm sets `RDMA_AR_AUDIT=200`, as r6-w1 did.
  - e = `DSV41_STEP_COLEMB=1`.
  - p = `DSV41_STEP_ENGRAM_PACK=all`.
- **Det:** the 9-case long-prompt check runs on **every T arm** (`DET_ARMS="e1 p1 e2 p2"`).
- **G4:** on c1, e1 and p1.
- **Install:** `KIT_INSTALL` is `ops/fleet.sh install`, so the identity arm proves the r8 module inert with flags off.
- **Boot lines to check:**
  - `DSV41_STEP_COLEMB rank r: full-table check ... PASS`;
  - `DSV41_STEP_ENGRAM_PACK=all: engram layers <a> and <b> share one rows gather and one wkv gather`.

Run the device window first (§5). serve.sh refuses without `EXPECT`.

## 4. Estimates (re-based on the 8-rank census; `scripts/c3_rebase.py` → `scripts/c3_rebase_census.txt`)

A collective's span includes the wait for the last rank, and an AG keeps that wait. So the saving is the transfer-alone
time (pooled p5) of the removed collective, minus the cost of what replaces it.

| lever | ms/step low / central / high | % G1 low / central / high | basis |
|---|---|---|---|
| COLEMB | 0.008 / 0.020 / 0.028 | 0.04 / 0.09 / 0.13 | eager AR p5 24.7 / 16.1 / 24.9 µs vs [T,640] AG ~16–18 µs, so 0 / 6 / 10 µs; drafter AR p5 25.8–26.2 µs vs in-graph AG ~12 µs, so 8 / 14 / 18 µs |
| PACK=all | −0.006 / 0.014 / 0.027 | −0.03 / 0.07 / 0.13 | rows AG p5 15.5–18.2, wkv AG p5 18.2–24.0 µs; costs: doubled payload + cat/split; lost paced-prefetch hiding (rows windows 100 % prefetch on t2) |
| head (r6 HEADREF + SCREENDQ, unblocked) | −0.009 / 0.037 / 0.115 | −0.04 / 0.17 / 0.53 | r6 range; census screen 460–486 µs and refine 21–59 µs agree with t2 |
| ISO (r6-skew, unblocked) | 0 / 0.017 / 0.086 | 0 / 0.08 / 0.40 | r6 range; census: 1.33–1.42 ms/step of last-arrival wait, rank 6 last 16–19 %, proxy unpinned |

**The only serving evidence:** in r6-w1, s1 (C4 + COLEMB + PACK) minus k1 (C4) = +0.32 tok/s (+0.25 %) for COLEMB and
PACK together. That is one unpaired pair, with noise ~0.2 %. It is consistent with the 0.16 % central sum.

Model estimates have been off by 3–4× in both directions; only the C-T-C-T decides.

**Where I see more:**
- The rows-AG windows carry 100 % of their paced prefetch, so every collective-removal lever (C3, C1, C2) loses part
  of its gain unless the l2pf planner re-places that DRAM work. That replan (§6 of the r6 step README) is worth more
  than PACK itself.
- The 1.3–1.4 ms/step of last-arrival wait is the biggest exact pool on the census. ISO addresses only its OS-noise
  part.

## 5. Install, compose, windows

**`install.py`** (one node; `FIXB_VLLM` overrides the vllm dir):
- Subcommands: `status | dry | install | uninstall | pins`.
- States: CLEAN = r6 module ebe2f7bb; INSTALLED = r8 module 425c6515 + `.bak-r8-fixb`; NO-R6STEP; FOREIGN.
- Install requires the r6-step patch on `engram.py` (34cf684e / e99f623d) and on `dsv41_r4_small.py` (d9bf5d59 /
  ebe73cdb). Every check runs before the first write.
- Atomic replace, then read-back. Uninstall is byte-exact.

**`ops/fleet.sh status | install | uninstall`** (8/8):
- Install and uninstall refuse unless the CLUSTER_LOCK has owner=claude.
- A preflight checks every node's state and preconditions first.
- Runs node by node, with rollback of every touched node on failure.
- `status` is read-only (the script is fed over ssh stdin; nothing is written).
- Today: **CLEAN 8/8**.

**LIFO.** `climb-r6/install-all.sh` and `climb-r7/install-all.sh` pin `dsv41_step.py` at ebe2f7bb. While r8-fixb is
installed they see 425c6515 as FOREIGN and refuse. Run `ops/fleet.sh uninstall` before any of their actions, or the
integrator adds 425c6515 as an accepted r6-step state.

**Composition** (`integrate/compose_fixb.py`, 78 checks): on copies of `climb-r7/integrate/composed/` in 10 states
(base / r5 / r6 / r5r6 / r5r6c, each with and without r7):
- install, no-op re-install and byte-exact uninstall on every r6-step state, with and without r7;
- only `dsv41_step.py` and its backup change;
- refusal on base and r5, and 4 drift refusals per state;
- no r7 kit touches `dsv41_step.py`, `engram.py` or `dsv41_r4_small.py`, so r8-fixb commutes with r7.

**Device window** (stopped, ~45–60 min). Render with `EXPECT=4f3cc1c6 WIN=r8-fixb-dw bash ops/plan.sh`. plan.sh
refuses a node line that runs Python without the environment rule; 4 mutants were tested. Then run:

```
WIN=r8-fixb-dw EXPECT=4f3cc1c6 NODE_PLAN=ops/device-plan.node MAC_PLAN=ops/device-plan.mac bash climb-common/device-window.sh
```

| lane | runs |
|---|---|
| spark-two | head synthetic + real (rank 4) |
| spark-three | step `--part all` (E0, C3a, C3b, C4, det) |
| spark-four | skew |
| Mac | installs the r8 module 8/8 only if the step device test passed, then status / collect / clean |

Order: run the device window first, then the head and ISO arms (r6 KIT.json `window_arms`) and the serving window
(`EXPECT=4f3cc1c6 bash ops/serve.sh`, ~1.5–2 h). r7 kits can be installed before or after this one; they commute.

After any promotion, pass the new `EXPECT` sha8 to every script. Nothing here defaults to a stale sha: 4f3cc1c6 is the
live launcher on 8/8 today.
