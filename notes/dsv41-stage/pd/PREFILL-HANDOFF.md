# pd prefill candidates: offline prep handoff (2026-10-25, ~19:00-20:15 UTC)

Offline only. Nothing here was installed, no restart, no GPU job, no request sent to the live server.
Staged on spark-one under `~/dsv41-stage/pd/{pf,sched-pd,nccl-arms,lone-chunk,p09-dualq,moe-glue,sp-v2,e7-tail}/`
(decode-prep tags d05/l2pf/pdl/rgate and `common/` belong to the decode step and are untouched).
Bases (identical 8/8 at 19:16 UTC): nvidia/model.py e7264781, model_state.py 53e4fe39,
decoder_replay_layers.py 2fedc0a6, dspark/speculator.py bbcee9e3, moe_runner.py 20b69fe5,
deep_gemm_utils.py a4a0486a, dsv41_sched.py 6074389f, launcher 2fd26e53.

## Ranking by expected TTFT gain (baseline 128K ~22.3 s / 8 chunks; 1M ~205-228 s / 61 chunks)

| # | candidate (tag, flag) | class | 128K TTFT | 1M TTFT | confidence |
|---|---|---|---|---|---|
| 1 | SP v2 fast (`pf`, `DSV41_PREFILL_SP=fast`) | EXPERIMENTAL, greedy-eq only (RS order) | -3.7..-4.2 s (-17%) | -28..-32 s (-14%) | P(pass) ~0.5: EV -2 s / -15 s |
| 1b | SP v2 exact (`DSV41_PREFILL_SP=exact`) | bitwise target, to verify (mHC split-K may differ at M/8) | -2.0..-2.4 s (-10%) | -15..-18 s (-7.5%) | medium-low |
| 2 | E7 final-chunks-only tail (`pf`+`sched-pd`, `DSV41_E7_TAIL=1`) | exact for every step it skips (needs guarantee G, enforced) | -1.0 s (-4.4%) | -9.7 s (-4.4%) | medium-high |
| 3 | Lone-prefill 24K chunks (`sched-pd`, `DSV41_PD_LONE_BUDGET=24576` + `--max-num-batched-tokens 24576`) | bitwise <=16K prompts, drift-class >16K | -0.3..-0.4 s (after E7; -0.4..-0.6 alone) | -2..-3 s (-3..-5 alone) | low-medium; HOLD on memory |
| 4 | MoE glue: shared+routed add folded into ep_gather (`pf`, `DSV41_MOE_GATHER_ADD=1`) | exact (bitwise RNE, GPU gate pending) | -0.26 s (-1.2%) | -2.0 s (-0.9%) | medium |
| 5 | P09 dual quant serving binding (`pf`+`p09-dualq/lib`, `DSV41_DUALQUANT_STAGE_A=1 DSV41_DUALQUANT_STAGE_B=1`) | exact (Codex: 297+66 quant cases, 132 MoE cases bitwise) | -0.12 s (-0.5%) | -0.9 s (-0.4%) | medium (binding untested on GPU) |
| 6 | NCCL arms (`nccl-arms`, launcher hook + `DSV41_NCCL_EXTRA`) | transport arms exact; proto/algo/channel arms change AR sum order | <= -0.3 s, likely ~0 | <= -3 s, likely ~0 | low |

Not additive in seconds with SP: SP shortens every chunk, so E7/glue/P09 keep their ms per chunk but SP-on
TTFTs shrink. E7 is independent of SP (tail vs layers 0-19). P09 and the gather-add fold do not stack on the
same MoE call (the P09 path returns before the fold arms): pick the fold first (2x the gain).
Already live and covering item (3)'s "live-row SiLU": `DSV41_MOE_LIVE_ACT=1` gates on >=1024 aligned rows, so
every prefill MoE call uses the live-row kernel (k1/2 table: M=16384 1017 -> 680 us/layer). Nothing to add.

## What is installable

* `pf/` — ONE site bundle (6 files, pd-site.sh format: `FILES` + `site/`), every feature default off:
  e7264781->fe7a7d12 model.py (E7 skip + SP v2), 53e4fe39->192bc014 model_state.py (E7 rule),
  2fedc0a6->580a4b58 decoder_replay_layers.py (skip flag + step mark), bbcee9e3->1a469da5 dspark/speculator.py
  (E7 draft skip), 20b69fe5->747cd5e4 moe_runner.py (P09 binding + fold), a4a0486a->a26655f3 deep_gemm_utils.py
  (fold kernel). Install (service STOPPED): `bash ~/dsv41-stage/pd/common/pd-site.sh install pf` on 8/8 via the
  decode step's `pd-all.sh`, or per rank. Backups `<file>.bak-pd-pf`. With no flags it must reproduce the anchor:
  first check after install = exact.py a8c3394af9d2 + identical bench-single spec counters under a pinned AV.
* `sched-pd/` — `~/dsv41-sched/dsv41_sched.py` 6074389f->85abae3d (+ new `dsv41_pd_policy.py` 3ac7b928):
  `bash ~/dsv41-stage/pd/sched-pd/sched-install.sh install` on 8/8 (backup `.bak-pd-sched-pd`). Flags off =
  stock dsv41_sched (test: identical scheduling decisions).
* `nccl-arms/launcher-hook.sh install` — launcher 2fd26e53 -> f5be18ba (inserts the DSV41_NCCL_EXTRA hook after
  `export NCCL_IB_HCA=`); round-trip tested. Needed because the launcher unsets every inherited NCCL_* var:
  `launch-conc.sh NCCL_PROTO=Simple` silently does nothing today.

## CPU tests run (all PASS)
Mac: `sched-pd/test_dsv41_pd_policy.py` (7 tests incl. 4000 random chunkings: no chunk E7 skips is ever read
under G; counterexample without G), `sched-pd/test_dsv41_sched_pd.py` (fake vLLM: flags off = stock, G defers
tiny non-final chunks, lone budget 24576 only when alone, refuses without the launcher flag),
`pf/tests/test_pf_static.py` (compiles; only whitelisted flag-gated base lines rewritten; flags default off;
E7 helper writes the last KV source's KV with the replay call's argument order, resets the mark, keeps the aux
layout; mark cleared before every return), `nccl-arms/test_nccl_hook.sh`, launcher hook round trip.
spark-eight (nice, 1 core, <1 GB): `moe-glue/test_gather_add_interp.py` (Triton interpreter; note the
interpreter rounds fp32->bf16 toward zero, so the CPU gate uses truncation; RNE equality is the GPU gate).
Re-run on spark-one after staging: see `RESULTS-cpu.txt`.

## 1. SP v2 (`sp-v2/`, in `pf/` model.py) — EXPERIMENTAL
r4 failure re-read: the only compare (hidden rel_l2 0.664) ran inside the FlashInfer autotune dummy forward
(02:55:12-23), where the reference pass double-writes state and tactics sweep; the init death was the known
rdma_ar DSpark-capture hang (now mitigated by Codex C02). knapcio's layer-20 rule (the last KV source writes
full-prompt compressed KV from gathered rows before the tail) already holds in our port: all five states are
all-gathered before `decoder_replay_layers(...)`, whose `write_kv` runs layer 20's KV/compressor/index-K
writes on every row; v2 keeps that and asserts it in bisect mode (the stock layers [k,20) also run on gathered
rows). Static review found no missed/doubled reduction (wo_b is a RowParallelLinear honouring reduce_results;
the shared expert is built with reduce_results=False; MoE runner honours skip_final_all_reduce; no routed
output transform for DSv4).
New: `DSV41_PREFILL_SP=exact` (every RS = the stock all-reduce + keep this rank's rows: same AR call and size,
so the reduced values are the stock bits; isolates sharding bugs), `=comm` (RS+AG at the same sites, nothing
sharded: fast-mode numerics without sharding), `=fast` (= r4 port), `DSV41_PREFILL_SP_LAYERS=k` (SP on layers
[0,k) only, bisect), DEBUG compares only on clean forwards (not capturing, not autotune, dict metadata, every
request's chunk starts at position 0, so the reference pass reads no state written by an earlier chunk).
Offline numerical test plan (one restart per row, 1 window ~1 h; 8 ranks; prompts 4K/8K/16K cold, salted):
1. `exact` + `DEBUG=4`: expect max_abs 0 on all 5 tensors; if not, the residue must be mHC split-K
   (compute_mhc_pre_num_splits keys on M: 1/4/16 splits by cdiv(M,64)) -> rel_l2 <1e-4; anything >1e-3 = bug,
   bisect with `DSV41_PREFILL_SP_LAYERS=1,2,4,10`. Then exact.py, lplong vs control: bitwise.
2. `comm` + DEBUG: rel_l2 <= ~1e-2 (RS vs ring AR order) -> transport numerics alone.
3. `fast` + DEBUG: same order as comm; then TTFT 16K/64K/128K x3, lplong |dlp| within base-vs-base 0.03,
   longctx 4K-200K, 500K/1M needles, GSM8K-50, decode unchanged (path needs >=2048 rows). Kill if fast < +12% at 128K.
Memory: +1 gathered state (~840 MB at 16K) transient; watch MemAvailable (observed 7.9-9.0 GB on 8/8 at 18:36).

## 2. E7 final-chunks-only decoder tail (`e7-tail/`, in `pf/` + `sched-pd/`) — top exact candidate
Per 16K chunk (B0prof): layer-20 write_kv on all rows 21.6 ms (kept), replay of 128 rows through layers 20-39
84.2 ms, post-replay full-T work + DSpark context/draft 84.3 ms (zero fill 6, mtp copy 5.9, hc_collapse 3.7,
norm 1.5, aux cat 4.6, MXFP8 quant 3.7, main_proj GEMM 43.0, 3 x wkv 10.5 ...). E7 removes ~165 ms per skipped chunk.
Rule (dsv41_pd_policy.e7_request_skippable): still prefilling, no prompt_logprobs, no per-request opt-out, and
tokens left after this chunk > max per-step tokens (max_num_batched_tokens) -> the next chunk cannot be final,
so this chunk is neither F nor F-1. Why F-1 must run: AV's first decode budget reads confidences recorded two
records back (F-1's propose). Why nothing else is read: trimmed later chunks attend only to their own kept rows
(kv_start = seq_len - window), decode reads only the final window, the drafter is all-SWA (window <= 128),
skipped steps' drafts/confidences/acceptance features are overwritten by F-1/F before use, the drafter and
replay-layer SWA groups are not prefix-cacheable (bounded replay). Guarantee G (scheduler hook on
`_reserve_prefill_lookahead`): a non-final prefill chunk is never <= 128 rows (deferred) -> F-1 is trimmed and
self-contained. Mode 1 skips only when EVERY request in the step qualifies (lone long prefill(s)); mixed steps
with decodes run stock, so co-scheduled rows never change shape. Model side refuses unless the dsv41 scheduler
is active (it enforces G), DP=PP=1. Skipped step: layer-20 write_kv still runs on every row; outputs are zeros;
DSpark propose returns its buffer (drafts of prefill chunks are ignored by the scheduler).
Per-request opt-out for an in-instance bitwise A/B: request `vllm_xargs: {"dsv41_e7_off": 1}`.
Serving test plan: (a) one instance, E7 on: same salted 40K/131K/300K prompts with and without dsv41_e7_off,
greedy 64 tokens + logprobs: bitwise identical text/logprobs and identical spec counters of the following
decode; (b) launch log shows `DSV41_E7_TAIL armed`, counter lines, sched stats `e7_defer`; (c) ABAB solo TTFT
64K/128K/500K/1M x3 per arm, gate >=3% at 128K and 1M; (d) exact.py, GSM8K-50, longctx, 1M needle, C1 unchanged.

## 3. Lone-prefill bigger chunks (`lone-chunk/`, `sched-pd/`)
`DSV41_PD_LONE_BUDGET=24576` + `DSV41_EXTRA="--max-num-batched-tokens 24576"`: the scheduler sets
max_num_scheduled_tokens to 24576 only when exactly one request exists and >16384 of it is left; any other
request -> 16384 (today's step). Refuses (logs, stays 16384) if the launcher flag is missing. First measure F on
the live server with `lone-chunk/fmeasure.py` (TTFT(16384+k), k=0/1/512/2048/4096/8192, 3 reps; go if F >= 0.10 s).
HOLD reasons: +~3 GB/rank workspace at 24K (+~6 GB at 32K) vs MemAvailable 7.9-9.0 GB observed today and
dgx-anti-oom's 4 GiB kill line on ranks 0-4; newcomer TTFT during a lone chunk rises ~+1.4 s (one longer chunk);
autotune/tactic tables at the new M must be diffed. With E7 on, F shrinks by the tail term, so the gain is ~1-2%.

## 4. MoE glue (`moe-glue/`, in `pf/`)
Fold: `_fwd_kernel_ep_gather_add` computes bf16(fp32(shared) + fp32(bf16(routed_acc))) = the stock
`shared_output + fused_output` bit for bit (RNE on GPU). Runner arms it only for eager calls with >= 2048 rows
(DSV41_MOE_GATHER_ADD_MIN_ROWS), NO_OVERLAP shared output present, routed scale 1.0, no transforms/SP/pre-reduced
output; the arm is cleared in `finally`, and a fold whose output is not the routed output raises (never a silent
double add). Saves ~1.6 ms/layer (one 503 MB pass -> one extra 168 MB read). GPU gate:
`test_gather_add_gpu.py <staged deep_gemm_utils.py> --timing` on one stopped rank (M 410..16384, dead routes
poisoned NaN, specials), require 0 bit diffs and fused <= 0.9x.
Not built (native, needs GPU): align-128 junk-free FC2 TMA stores on the 128-row prefill path (~24K padding rows
x 10 KB per layer at 16K ~ 1 ms/layer ~ -20 ms/chunk); design = R1 #5 arm A (8-row CD boxes, skip groups whose
first row has grouped_layout < 0) in the BLOCK_M=128 instantiation; LIVE_ACT leaves FC2-input padding stale, so
stores must stay skipped, never read. Gather-A FC1 / K2 quant-scatter (R3 #5, R1 #6) remain L-effort.

## 5. P09 dual quant (`p09-dualq/`, binding in `pf/` moe_runner.py)
`lib/` = Codex's stage_a.py, stage_b.py, dualquant_stage_a.cu b87d8696, libdualquant_stage_a.so 51ce3c4a +
build-manifest (from spark-four build-v1; PASS 297/66/132 cases; Stage B rank-local MoE -1.46% at 4K, -3.10% at
16K). Binding: `MoERunner._apply_quant_method` -> `stage_b.run_invocation(...)` with the live router closure,
`quant_method.moe_kernel`, `w13_weight/w2_weight`, `activation`; eligibility = Codex's checks (410..16384 rows,
same BF16 tensor for shared and routed, NO_OVERLAP, exact kernel classes) else stock. Copy `lib/` to the same
path on 8/8 before arming (`DSV41_DUALQUANT_DIR` default `~/dsv41-stage/pd/p09-dualq/lib`). GPU gate: Codex's
`gpu_compare_moe.py` on layers 0/19 + a serving A/B (counters bitwise under pinned AV, MoE time per 16K chunk).
Codex's own source snapshot equals today's live moe_runner/shared_experts/modular_kernel/no_dp_ep/deep_gemm_moe.

## 6. NCCL arms (`nccl-arms/`)
Facts: prefill ARs are 40-160 MiB (NCCL; rdma_ar caps at 2 MiB = 204 rows); B0prof: 41 ARs x 11.7 ms = 18% of a
16K chunk at 24.7 GB/s bus bandwidth vs ~27 GB/s for the twin PCIe x4 RoCE links, so the ceiling is ~8% of AR
time. "RING_LL" in traces is NCCL's kernel-name artifact (NCCL_PROTO=Simple is likely a no-op for 160 MiB).
Arms (via the hook): A0 probe `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,NET` (parse with
`nccl_tuning_parse.py`: chosen algo/proto/channels per size, GDR yes/no); A1 `NCCL_IB_QPS_PER_CONNECTION=4
NCCL_IB_SPLIT_DATA_ON_QPS=1` (transport only: reduction order unchanged -> exact); A2 `NCCL_BUFFSIZE=16777216`;
A3 `NCCL_MIN_NCHANNELS=8`/`MAX=8`; A4 `NCCL_PROTO=Simple`; A5 `NCCL_ALGO=Ring`. A2-A5 can change the bf16
sum order of every NCCL AR (prefill prompts > ~204 tokens and any aux-stream decode AR) -> output-changing,
measure only. Cheapest first test (service stopped, 8 ranks, no vLLM): `run_ar_bench.sh <tag> "<arm env>"`
(ar_bench.py: [T,5120] bf16 AR at 4K/8K/16K rows, median ms + result sha per arm, so order-changing arms show).
Promote only if an exact arm gains >= 5% AR time and the TTFT gate passes.

## Rollback
Everything is staged; nothing to roll back. After a later install: `pd-site.sh uninstall pf`,
`sched-install.sh uninstall`, `launcher-hook.sh uninstall` (each sha-guarded, refuses on unexpected state).
