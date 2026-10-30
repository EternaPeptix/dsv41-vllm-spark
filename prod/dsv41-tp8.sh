#!/bin/bash
# dsv41-tp8.sh <rank> — one vLLM 0.32.0 rank of DeepSeek-V4.1-Flash (uncensored-fp8) TP8 across the 8 Sparks.
# rank 0 = spark-one (API on 127.0.0.1:8000); ranks 1-7 headless. Run under systemd-run --user (unit dsv41-rank).
# Extra vllm args can be appended via DSV41_EXTRA (e.g. "--moe-backend marlin").
set -u
R=$1
MODEL=$HOME/models/dsv41/uncensored-fp8
LOGDIR=$HOME/dsv41-logs; mkdir -p "$LOGDIR"

# /etc/environment on spark-three..five carries stale NCCL_* (GID 3, enP7s7 is down); clear them so all ranks match.
for v in $(env | grep -oE '^(NCCL|GLOO)_[A-Z_]+'); do unset "$v"; done
# Fabric settings proven by the MiMo TP8 launcher (~/mimo-tp8.sh).
export NCCL_SOCKET_IFNAME=enp1s0f0np0 GLOO_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 NCCL_IB_GID_INDEX=5 NCCL_DEBUG=WARN
export VLLM_HOST_IP=$(ip -4 -br addr show enp1s0f0np0 | awk '{print $3}' | cut -d/ -f1)
export DSV41_SM121_VARLEN_INDEXER=1
export DSV41_SM121_TARGET_M64=1
export DSV41_SM121_TARGET_M32=${DSV41_SM121_TARGET_M32:-1}  # m32: 32-row target-MoE tiles (M<=48); needs fused_moe m32 site files; 0 = M64 path
export DSV41_TARGET_M32_MAX_M=${DSV41_TARGET_M32_MAX_M:-96}  # r3-e3: M32 target-MoE tiles up to 96 rows (16 seqs x 6); 48 = round-2 limit
export DSV41_TARGET_M32_NATIVE_DIR=${DSV41_TARGET_M32_NATIVE_DIR:-$HOME/dsv41-stage/m32/target-m32-native-v1}
export DSV41_RDMA_AR=${DSV41_RDMA_AR:-1}  # rdma-ar: custom ibverbs all-reduce/all-gather for <=512 KiB decode collectives (needs cuda_communicator.py rdma patch + ~/rdma_ar_dsv41); 0 = PyNccl only
export DSV41_WARM_SPEC_KERNELS=${DSV41_WARM_SPEC_KERNELS:-1}  # conc: extra warmup for greedy/temp-only sampler kernels + odd mixed prefill (needs warmup.py warmspec patch); 0 = stock warmup
export DSV41_MEMORY_DIAGNOSTICS=1
export DSV41_TRIM_STARTUP_CACHE=1
export DSV41_LOCK_ENGRAM_HOST=1
export DSV41_MOE_COMPACT=${DSV41_MOE_COMPACT:-1}  # exp1 moe-compact: unpadded target FC1 (N 768->576, -7.5 GiB/rank); needs moe-compact site files; 0 = padded layout
export DSV41_AV_VERIFY_SCALE=${DSV41_AV_VERIFY_SCALE:-0.6}  # exp5 av-calib: scale profiled verify-cost slope (needs adaptive_verification.py avcal patch); DSV41_AV_VERIFY_SCALE=1 DSV41_AV_FIXED_MS=0 = stock trimmer
export DSV41_AV_FIXED_MS=${DSV41_AV_FIXED_MS:-3.4}  # exp5 av-calib: outside-graph ms added to draft cost (r3-e1: 2.0 -> 3.4 with sharded Markov drafter)
export DSV41_DSPARK_SHARDED_MARKOV=${DSV41_DSPARK_SHARDED_MARKOV:-1}  # r3-e1: vocab-sharded Markov bias in DSpark drafter (needs dspark/speculator.py 774b34c2 patch); 0 = full-vocab Markov
export DSV41_AV_SCALE_N0=${DSV41_AV_SCALE_N0:-96}  # exp5 av-calib: full scale up to verify width N0 (= M32 max M; r3-e3 48 -> 96)
export DSV41_AV_SCALE_N1=${DSV41_AV_SCALE_N1:-104}  # exp5 av-calib: back to profiled cost by width N1 (r3-e3 56 -> 104)
export RDMA_AR_MAX_REDUCE_BYTES=${RDMA_AR_MAX_REDUCE_BYTES:-2097152}  # r3-e3: RDMA two-shot all-reduce up to 2 MiB (= RDMA_AR_MAX_BYTES slot, 204 rows) for conc17-32; 1048576 = round-2, 524288 = round-1
export DSV41_QKV_A_SHARD=${DSV41_QKV_A_SHARD:-1}  # r3-e4: column-shard replicated qkv_a (1792 -> 224 rows/rank) + rdma all-gather for M<=8 (needs r3dense site files: attention.py f2f48198, dsv41_r3_dense.py 993023f2); 0 = replicated GEMM
export DSV41_IDX_WPROJ=${DSV41_IDX_WPROJ:-bmm32}  # r3-e4: indexer weights_proj as 32-way split-K bmm (33 -> 8.5 us at M 2-8); empty = stock F.linear
export DSV41_FP8_LMHEAD=${DSV41_FP8_LMHEAD:-draft}  # r3-e4: MXFP8 copy of lm_head for DSpark draft logits only (lossless; target verifies; needs logits_processor.py caf6d216 + dspark.py fa56d07d); empty = bf16
export DSV41_SONAR_OPROJ=${DSV41_SONAR_OPROJ:-fp8}  # r3-e4: split-K W8A16 wo_a on exact MXFP8 re-encode (needs deepseek_v4 ops/o_proj.py 34b9e1a4 + sonar_bf16_bmm.py 9f6f3f84); empty = torch.bmm
export DSV41_SONAR_OPROJ_SPLITS=${DSV41_SONAR_OPROJ_SPLITS:-32}  # r3-e4: 32 splits fastest at every M (23-24 us vs bmm 40-55)
export DSV41_SONAR_OPROJ_MAX_M=${DSV41_SONAR_OPROJ_MAX_M:-32}  # r3-e4: above 32 rows fp8 kernel loses to torch.bmm
export DSV41_BLOCK_VERIFY_FRESH_U=${DSV41_BLOCK_VERIFY_FRESH_U:-1}  # r3-e5: fresh block-verify uniform per step/row (needs rejection_sampler_utils.py 09522889); 0 = stock (lossy with block)
export DSV41_DRAFT_FRESH_NOISE=${DSV41_DRAFT_FRESH_NOISE:-1}  # r3-e5: drafter Gumbel noise keyed by step+draft index (needs dspark/speculator.py 00cc6e2b); 0 = stock (lossy with block)
export DSV41_L2_PREFETCH=${DSV41_L2_PREFETCH:-1}  # r4-e1: L2 prefetch of next GEMM weights during collectives (needs rdma_all_reduce.py 99102ff1 + dsv41_l2_prefetch.py 7d2acdc8); 0 = off
export DSV41_R4_MAINPROJ_SPLIT=${DSV41_R4_MAINPROJ_SPLIT:-1}  # r4-e1: DSpark main_proj column split, bitwise self-checked per M (needs nvidia/model.py e7264781 + dsv41_r4_small.py 02f94f29)
export DSV41_R4_CERT_HEAD=${DSV41_R4_CERT_HEAD:-1}  # r4-e1: screened target lm-head, argmax-identical, armed only on all-greedy steps; 0 = stock
export DSV41_R4_CERT_HEAD_M=${DSV41_R4_CERT_HEAD_M:-2-96}  # r4-e1: gpu_test_r4 qualified row counts (M=1 not bitwise in full mode)
export DSV41_EAGER_GLUE=${DSV41_EAGER_GLUE:-all,swam}  # r4-e1: fused SWA metadata + 1024-wide dflash pad (needs sparse_swa.py 38b16c51 + dflash/speculator.py 17ddccf1); g2 2026-10-26: + swam = the 14 per-KV-group decode SWA metadata launches as one (stepovh C, needs sparse_swa.py 854c3e5f); all = without swam; empty = stock
export DSV41_DSPARK_GREEDY_SHARDED=${DSV41_DSPARK_GREEDY_SHARDED:-1}  # r4-e2: all-greedy steps replay a 2nd set of DSpark draft graphs on the vocab-sharded Markov path (needs dspark/speculator.py bbcee9e3); 0 = stock probabilistic drafter on every step
export DSV41_ROUTE_LIVE=${DSV41_ROUTE_LIVE:-skip}  # r4-e4: dead/padding verify rows read no experts (router ids -1; needs dsv4_topk.py 45c5fc4f + fused_topk_bias_router.py ea6720bc); 0 = stock. NEVER with b12x MoE
export DSV41_VERIFY_CAP=${DSV41_VERIFY_CAP:-auto}  # r4-e4: DSpark-confidence cap on drafts per step, tau from AV calibration (needs adaptive_verification.py 7e30eebc + model_runner.py 9fecbda4); 0 = off
export DSV41_IDX_ROW_SHARD_MIN=${DSV41_IDX_ROW_SHARD_MIN:-1024}  # r4-e5: indexer prefill row-shard threshold per rank (needs mla/indexer.py prefill-idxshard); stock 16384
export DSV41_IDX_ROW_SHARD_MIN_CTX=${DSV41_IDX_ROW_SHARD_MIN_CTX:-32768}  # r4-e5: only shard indexer rows when max ctx >= this; 0 = no ctx gate
export DSV41_TOPK_CANON=${DSV41_TOPK_CANON:-1}  # 1M determinism: canonical (ascending, -1 last) top-k index order so sparse MLA sums are run-independent at max_model_len >= ~398K (needs sparse_attn_indexer.py 7112c01d); 0 = stock order
export DSV41_CAND_LENAWARE=${DSV41_CAND_LENAWARE:-1}  # k1/1-indexer-mask: decode candidate flags + -inf mask stop at each row's end instead of the max_model_len-wide row (exact; needs dsa/candidate_blocks.py a5816163); 0 = stock
export DSV41_MOE_LIVE_ACT=${DSV41_MOE_LIVE_ACT:-1}  # k1/2-silu-liverows: routed FP4 MoE SiLU-mul-FP8-quant over live rows only (expert_ends from ep_scatter; exact; gate DSV41_MOE_LIVE_ACT_MIN_ROWS default 1024 aligned rows; needs fused_moe deep_gemm_utils.py a4a0486a + experts/deep_gemm_moe.py 5162dc0f); 0 = dense
export DSV41_RDMA_PARLOAD=${DSV41_RDMA_PARLOAD:-1}  # k1/3-rdma-loads: rdma_ar parallel-load kernels (all peer loads in flight before the fixed-order fp32 accumulate; bitwise identical; loads ~/rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so 62daad6c via rdma_all_reduce.py acbeeef7); 0 = legacy build/ .so
export DSV41_SCHED=${DSV41_SCHED:-1}  # short-request scheduling: AsyncScheduler subclass ~/dsv41-sched/dsv41_sched.py (+ dsv41_sched_policy.py) that admits short (<=4K uncached) requests during a long prefill in their own step and does not pre-schedule past a big prefill step; rank 0 runs it; 0 = stock AsyncScheduler
export DSV41_SCHED_LONG_CAP=${DSV41_SCHED_LONG_CAP:-0}  # sched: long-prefill tokens in a short step (0 = skip the long prefill in that step)
export DSV41_SCHED_HOLD=${DSV41_SCHED_HOLD:-1}  # sched: 1 = never pre-schedule past a big step, 2 = timed (pre-schedule 300 ms before its predicted end unless a request arrived)
# rel-2 / pd 2026-10-25 (startup only; needs model_runner.py 316fca44 + dsv41_rel.py f373f911 + rdma_all_reduce.py 9056dcdf + dspark.py 36811ca9).
# The Codex C02 device sync between decoder-replay and DSpark capture stays on in code (DSV41_CAPTURE_SYNC=0 = pre-C02 order, test only).
export DSV41_CAPTURE_WATCHDOG_S=${DSV41_CAPTURE_WATCHDOG_S:-90}  # rel-2: decoder-replay or DSpark capture phase > 90 s -> stack dump + worker exit (AV phase budget DSV41_CAPTURE_WATCHDOG_AV_S, 420); 0 = off (a desync then spins to the 600 s rdma_ar trap)
export RDMA_AR_XSTREAM=${RDMA_AR_XSTREAM:-wait}  # rel-2: an eager rdma_ar call on a new stream waits for the previous stream if it is still busy (the capture race); detect = log only; off = no guard
export RDMA_AR_FINGERPRINT=${RDMA_AR_FINGERPRINT:-1}  # rel-2: ranks compare a hash of the rdma env, rdma_all_reduce.py, the .so and the l2 prefetch module at init and refuse to start on a mismatch; 0 = off
export DSV41_AV_PROFILE=${DSV41_AV_PROFILE:-pinned}  # rel-2 arm P: with DSV41_AV_COST_JSON set and covering this shape set, skip the 5x AV cost profile (one warm replay per size, same calibrated tables); without a pin = full profile as before; force = always profile
# 2026-10-29 user OK: production uses the k5 AV cost pin (no boot-time AV profile); window-6 X on
# DSV41_AV_COST_JSON= (empty) as a supervisor argument = one unpinned launch with the boot-time AV profile (the default uses "-", not ":-", so an explicit empty value is kept); DSV41_DRAFT_MOE_M32=0 = X off
export DSV41_AV_COST_JSON=${DSV41_AV_COST_JSON-$HOME/dsv41-opt-20261021/pins/av-pin-k5-g3FU.json}
export DSV41_DSPARK_SHARD_FILTER=${DSV41_DSPARK_SHARD_FILTER:-1}  # rel-2 arm S: the DSpark draft load opens only the 3 shards holding mtp.* (45/48 skipped, ~33 s -> ~4 s); 0 = walk all 48
export DSV41_MOE_FC2_D05=${DSV41_MOE_FC2_D05:-1}  # pd-d05 2026-10-25: M32 target FC2 (E384 N5120 K384, <=96 tokens) skips the TMA store of every fully dead 8-row output group (Codex D05 native 6fea5091; needs sm121_target_m32.py 5acc5fde + sm121_d05_verify.py c04c363d); exact (written groups bitwise, dead rows never read); 0 = original FC2, verify = shadow compare
export DSV41_L2_PREFETCH_PRUNE_SHARED=${DSV41_L2_PREFETCH_PRUNE_SHARED:-1}  # pd-l2pf arm S 2026-10-25 (Codex D01): L2 prefetch plans drop the ffn.shared_experts weight hints and router hints backfill the window; cache hints only, exact (needs dsv41_l2_prefetch.py 4da7745b+; live 514dcb23 = pd-l2pf, also has PRUNE_SHARED_POST_BUDGET); 0 = full plan
export DSV41_MOE_M32_G1=${DSV41_MOE_M32_G1:-all}  # g1 m32-bw 2026-10-26: M32 target routed-MoE FC1/FC2 (<=96 tokens) JIT-built from the m32-bw 'all' tree (FC1: weight L2 prefetch 8 k-blocks ahead + evict-first, live-row A loads, live-row direct stores, D04 dead-walk stop; FC2: live-row stores, evict-first, dead-walk stop); exact (same MMA sequence and K order per element; bitwise unit test 8/8, exact.py a8 + G1 digest identical); needs sm121_target_m32.py db659d80 + ~/dsv41-stage/g1/m32-bw (PREP.ok bb9660e130330ec0) + DSV41_MOE_FC2_D05=1, excludes DSV41_MOE_PDL_EARLY_B; 0 = live D05 kernels
export DSV41_MOE_FC2C=${DSV41_MOE_FC2C:-1}  # g2 fc2c-on-m32all 2026-10-26: M32 target FC2 (M>=2, DSV41_MOE_FC2C_MIN_TOKENS=2) reads the compact K tail (-16.7% FC2 weight bytes; tails +62.9 MB/layer = +2.34 GiB/rank, built before graph capture) on top of the m32-bw 'all' FC2 kernel; exact (unit 8/8 PASS 2064 cases/rank, exact.py a8 + G1 digest identical); needs sm121_target_m32.py ea4a8814 + ~/dsv41-stage/g2/fc2c-on-m32all (PREP.ok dcf5bc0909472121; serving dependency) + DSV41_MOE_M32_G1=all|0; 0 = off
# goal window 6 (RUNBOOK 7, promoted X): exact levers, default on; '=0' / '=1024' as a supervisor argument turns one off for a launch (RUNBOOK 9)
export DSV41_DRAFT_MOE_M32=${DSV41_DRAFT_MOE_M32:-1}
export DSV41_L2_PREFETCH_WOA=${DSV41_L2_PREFETCH_WOA:-1}  # g2 stepovh A 2026-10-26 (knapcio WOA port): L2 prefetch of wo_a/wo_b during sparse MLA on a second side stream; cache hints only, exact; needs dsv41_l2_prefetch.py 998af3d7+ (live 63737f9c); 0 = off
export DSV41_SYNC_FREE=${DSV41_SYNC_FREE:-argmax}  # g2 stepovh B 2026-10-26: CERT-armed all-greedy steps all-gather 16 B/row (max, id) instead of the [M,16160] bf16 logits and return one-hot logits at the stock greedy token (same token and counts, ties included); exact; needs dsv41_r4_small.py c46e66bd + DSV41_R4_CERT_HEAD=1; audit = check only; 0 = off
export DSV41_TOPK_CANON_FUSED=${DSV41_TOPK_CANON_FUSED:-1}  # g3 codex-adopt #1 2026-10-26: fused width-512 CANON top-k kernel, one launch instead of five (needs sparse_attn_indexer.py 481905c2); exact (byte-identical to stock); 0 = stock
export DSV41_DECODE_REMAP_REUSE=${DSV41_DECODE_REMAP_REUSE:-1}  # g3 codex-adopt #2 2026-10-26: decode top-k remap once per (index source, KV source) interval, 38 -> 8 launches per step (needs flashinfer_sparse.py b059bdf3); exact; check = verify only; 0 = stock
export DSV41_IDX_OMIT_T2S=${DSV41_IDX_OMIT_T2S:-1}  # g3 codex-adopt #3 2026-10-26: the dead CUDA-indexer token_to_seq is neither allocated nor written (needs mla/indexer.py d9f4c5c9; ~-0.9 GB at late 1M steps); 0 = stock
export DSV41_DRAFT_HEAD_TACTIC=${DSV41_DRAFT_HEAD_TACTIC:-1}  # g3 dhtac 2026-10-26: DSpark draft head GEMM pinned to CUTLASS tactic 1 for the rows in the map (startup self-check: all 6 tactics + stock bitwise-equal, else stock); 0 = autotuned
export DSV41_DRAFT_HEAD_TACTIC_MAP=${DSV41_DRAFT_HEAD_TACTIC_MAP:-17-29:1}  # g3 dhtac: serving autotune cache (8c8be689) picks tactic 0 at 17-32 rows = 1.05 ms vs 0.38 ms at 20 rows (4 requests); >= 30 rows tactic 0 is already fastest
export DSV41_PREFILL_SP=${DSV41_PREFILL_SP:-exact}  # g4p #04 2026-10-27: layers 0-19 of eager prefill chunks >= 2048 rows run sequence-parallel (mHC/Engram per-row work on this rank's 1/8 of rows; attention, MoE and every all-reduce on all rows exactly as stock; split heuristics keyed on the logical row count); bitwise vs stock (QSP: 2048-16384 rows, 8/8 ranks); needs model.py 9c2d22ad; 0 = stock
export DSV41_PREFILL_SP_SELF_CHECK=${DSV41_PREFILL_SP_SELF_CHECK:-once}  # g4p #04: first SP forward of a launch checks every sharded mHC op against the full-row op bit for bit (rank-agreed); a mismatch falls back to full rows for that op; 1 = every new row count, 0 = off
export DSV41_E7_EXACT=${DSV41_E7_EXACT:-post_and_draft}  # g4p #05 2026-10-27: prefill chunks that are neither the final one nor the one before it skip the post-replay full-row work and the DSpark draft (layer-20 KV + 128-row replay still run, so every cache write is stock); bitwise vs stock (QE7); needs model_state.py b7598364, decoder_replay_layers.py b75aa927, speculator.py 439b390e; 0 = stock
export DSV41_FAST_LOAD=${DSV41_FAST_LOAD:-1}  # g4l fast loader 2026-10-27 (Codex ops kit 04): the target stream is pre-sorted (the VL wrapper no longer materialises 48 shards of mmap views); 8 threads read every tensor ahead of the stock weight loaders (O_DIRECT into pinned buffers; expert w1/w3 = this rank's rows only; Engram rows filled in place, holes never read); every parameter sha256-identical to the stock loader on 8/8 (DSV41_WEIGHT_DIGEST); load 175 -> 36 s, pinned time-to-healthy 286 -> 126-132 s, no startup swap-out; needs model_loader/dsv41_fast_load.py dfbcf2c7 + default_loader.py 1aeec2e6 + nvidia/vl_model.py 3ec0d401 (g4-all.sh status fload); 0 = stock mmap loader
export DSV41_TOOL_STREAM_NEWLINE=${DSV41_TOOL_STREAM_NEWLINE:-1}  # g4l3 sPTC layer 3 2026-10-27: the DeepSeek-V4.1 tool parser also flushes run_code/tool argument deltas at every newline (and streams open string arguments incrementally) instead of only at ">", so streaming clients get programs line by line; output-neutral (parser only: exact.py a8 + G1 digest unchanged, streamed arguments == non-streamed for the same tokens); needs site parser/deepseek_v4.py 3334c1fa + parser/engine/parser_engine.py 6e3eeaba (~/dsv41-stage/layer3, status.sh; backups .bak-layer3); read by the API server at import; 0 = stock flushing at ">"
export DSV41_TOPK_CANON_TIES=${DSV41_TOPK_CANON_TIES:-all}  # g5det 2026-10-27 (goal window 5, Determinism): boundary ties of the sparse-indexer top-k (several logits equal to a row's 512th value; the stock prefill top_k_per_row_prefill and decode persistent_topk / top_k_per_row_decode keep an arbitrary subset, chosen by an atomic race) keep the LOWEST column indices instead, so greedy output of long prompts (> ~1K tokens) is run-to-run identical; rows without a tie are untouched (bitwise stock), so exact.py a8 + the G1 digest are unchanged; needs site layers/dsv41_topk_ties.py 943961bc + sparse_attn_indexer.py 005785f9 (~/dsv41-stage/g5det/g5det_install.py status|uninstall); all = prefill + decode rows, prefill = prefill only, 0 = stock
export DSV41_MHC_OVERLAP=${DSV41_MHC_OVERLAP:-moet}  # g5mhc 2026-10-28 (goal window 5, mHC): the mHC boundary's coefficient half (fn projection + 20x Sinkhorn -> post/comb/next pre-mix, first read at the NEXT boundary) runs on a side stream joined after the sublayer, in FULL decode graphs of 1..64 tokens; the critical path keeps the post-mix, collapse, RMSNorm and draft aux in ONE launch (vLLM-Moet b0dc737 mhc_post_norm, mode 1, Apache-2.0, sources ~/dsv41-stage/moet/src/vllm-moet-b0dc737/tools/dsv41_sm120/mhc_overlap, built once per node into ~/.cache/torch_extensions). Bitwise vs the served pair: 0-ulp unit + recorded-input tests (eager + graph replay) and exact.py a8 + the pinned G1 digest; pinned A/B (av-pin-k5-g3FU, 2 pairs vs g5mhc-C3): G1 +3.0%, G4 prose MIN +1.6/+3.9%, G4min_hat +3.8%, 4-session step -4.1..-4.4%; needs site nvidia/model.py ca14bbb1 + dsv41_mhc_overlap.py 4468cad5 (~/dsv41-stage/g5mhc/g5mhc_install.py status|uninstall); stock = the served TileLang kernels on the critical path (slower at 1 user); 0 = off (served path, no side stream)
export DSV41_MHC_OVERLAP_PROJ=${DSV41_MHC_OVERLAP_PROJ:-moet}  # g5mhc: the side-stream projection is Moet's mhc_proj (4 tokens per CTA, the fn slice read once per block; partials bitwise = the served mhc_fused_tilelang partials); fused = the served kernel (96 CTAs per token)
[ "$DSV41_SCHED" = 1 ] && export PYTHONPATH=$HOME/dsv41-sched${PYTHONPATH:+:$PYTHONPATH}
export CUDA_HOME=/usr/local/cuda TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas VLLM_USE_DEEP_GEMM=1
export PATH=/usr/local/cuda/bin:$PATH
# SM12x needs 64-token KV blocks: DeepGEMM's indexer takes block_kv (block // ratio) <= 64.
# Local patches (originals kept as *.orig): vllm deepseek_v41/attention.py (SWA page 64),
# nvidia/flashinfer_sparse.py + v1/attention/backends/mla/indexer.py (allow 64-token blocks),
# flashinfer data/csrc/sparse_mla_sm120_prefill.cu (dual prefill for 32-row compressed pages).
source "$HOME/venvs/vllm032/bin/activate"
# --- Engram row check (insert in ~/dsv41-tp8.sh after "source .../activate", before ARGS=(serve ...)) ---
# After engram_punch.py a node holds only its own rank's Engram rows; a node started as another rank (or another
# TP size) would load zeros silently (vLLM 0.32 has no row check). Headers + hole map only, ~1 s. ENGRAM_CHECK=0 skips
# it -- never do that on a punched node.
if [ "${ENGRAM_CHECK:-1}" != 0 ]; then
  EP=$HOME/dsv41-stage/engram-punch/engram_punch.py
  [ -f "$EP" ] || { echo "refusing: no $EP for the Engram check (ENGRAM_CHECK=0 skips it)" | tee -a "$LOGDIR/rank$R.log" >&2; exit 1; }
  if ! python3 -I "$EP" --model "$MODEL" --rank "$R" --world 8 --check >> "$LOGDIR/engram-check.log" 2>&1; then
    echo "refusing: $(hostname) Engram tables lack rank $R's rows (see $LOGDIR/engram-check.log; restore: engram-punch/restore_from_four.sh)" | tee -a "$LOGDIR/rank$R.log" >&2
    exit 1
  fi
fi
# --- end Engram row check ---

# Page cache of the checkpoint competes with weights in unified memory: evict it before loading.
python3 - "$MODEL" <<'EOF'
import os, sys
for f in os.listdir(sys.argv[1]):
    p = os.path.join(sys.argv[1], f)
    if os.path.isfile(p):
        fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
EOF
pkill -f "^python3 $HOME/balloon.py" >/dev/null 2>&1 || true

ARGS=(serve "$MODEL" --served-model-name dsv41
  --tensor-parallel-size 8 --nnodes 8 --node-rank "$R" --master-addr 10.77.0.11 --master-port 29501
  --kv-cache-dtype fp8_ds_mla --block-size 64
  --max-model-len ${DSV41_MAX_MODEL_LEN:-1048576} --max-num-batched-tokens 16384 --max-num-seqs ${DSV41_MAX_NUM_SEQS:-32}
  --kv-cache-memory-bytes ${DSV41_KV_BYTES:-16G}  # r3-e3: 8G -> 16G (3.26M tokens) with max-num-seqs 16 -> 32
  --limit-mm-per-prompt '{"image":0}'
  --engram-config '{"cpu_offload": true, "use_thp": true}'
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --enable-prompt-tokens-details)  # user 2026-10-25: always on; usage gains prompt_tokens_details.cached_tokens (frontend only)
# DSpark speculative decoding from the checkpoint's mtp.{0,1,2} weights; DSV41_SPEC=0 disables.
SPEC=${DSV41_SPEC:-5}
if [ -n "${DSV41_SPEC_JSON:-}" ]; then
  # Full override of the speculative config (JSON), e.g. adaptive verification off.
  ARGS+=(--speculative-config "$DSV41_SPEC_JSON")
elif [ "$SPEC" != 0 ]; then
  # r3-e5: probabilistic drafting + block verification (pb5) for sampled requests; greedy rows unchanged.
  # DSV41_SPEC_SAMPLED=0 -> round-2 greedy-draft config (and the sharded Markov drafter becomes active again).
  if [ "${DSV41_SPEC_SAMPLED:-1}" = 1 ]; then
    ARGS+=(--speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": $SPEC, \"enable_adaptive_verification\": true, \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"block\"}")
  else
    ARGS+=(--speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": $SPEC, \"enable_adaptive_verification\": true}")
  fi
fi
if [ "$R" = 0 ]; then ARGS+=(--host 127.0.0.1 --port 8000); else ARGS+=(--headless); fi
[ "$DSV41_SCHED" = 1 ] && ARGS+=(--scheduler-cls dsv41_sched.DSV41Scheduler)
# shellcheck disable=SC2086
exec vllm "${ARGS[@]}" ${DSV41_EXTRA:-} > "$LOGDIR/rank$R.log" 2>&1
