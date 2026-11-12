// dsv41 climb r3, lever G02 (attn:L-XS): XSPLIT fused split-tile sparse MLA decode.
//
// One launch. A persistent grid (<= 46 CTAs) claims work items from a device counter in
// index order: first every A unit (token i, tile t) token-major and tile-ascending, then one
// B job per token, then END.
//
//   A unit  = the live prefill_mg_impl per-tile body for (i, t), verbatim, minus the four
//             cross-tile ops (guarded rescale, acc_o FFMA, l FADD, acc_rope MMA chain). The
//             running max is NOT parallelised: the unit publishes its tile max per head
//             (epoch-tagged, 64-bit single-copy word), then folds the published maxima of
//             tiles 0..t-1 of its token sequentially from -1e30 exactly as the live kernel
//             updates m_smem, and continues with w = exp2f(s - nm). It writes back the
//             per-tile operands the cross-tile ops need (alpha for all 16 rows, nm, sc, ls,
//             w, xv, rope B) and publishes a done flag (release).
//   B job   = for token i, replay tiles 0..n-1 in order (acquire on each done flag): the
//             guarded FMUL (alpha0 < 1 || alpha1 < 1, dead-row alpha1 carried), the FFMA
//             acc += xv * sc, the separate FADD l += ls, and the 4 chained bf16 MMAs with
//             C = acc_rope and A = bf16(w) staged in smem exactly as xv_rope_mma_mg does;
//             then the live epilogue (cross-warp l sum, sink fold, il, bf16, lse).
//
// Deadlock freedom: every wait (prefix max, done flag) targets an item with a smaller claim
// index, claimed by a CTA that is already running and that publishes it without waiting on
// any larger index. So the kernel cannot deadlock for any grid size, residency or co-runner.
// Stale state: flags carry the call epoch E = ctl[GEN] + 1; the last CTA to finish resets the
// claim/finished counters and publishes the next epoch (clears all flags before a 32-bit wrap).
//
// Only the production instantiation is supported: DSV4, BF16 QK, NUM_HEADS = 8 (one padded
// 16-row head group), main PBS 64, extra PBS 64 or 32, runtime topk_length/topk_length_extra.
#pragma once

#include <flashinfer/attention/sparse_mla_sm120/prefill_mg_kernel.cuh>

#ifndef XS_ROPE_IN_WS
#define XS_ROPE_IN_WS 1
#endif

// Timing-only arms (never shipped): 0 exact; 1 no prefix-max wait (wrong output);
// 2 A only, B jobs skip the replay and epilogue (no output); 3 A skips the row stores.
#ifndef XS_ARM
#define XS_ARM 0
#endif

// B replay loads the next tile's row while replaying the current one (load scheduling only).
#ifndef XS_B_PREFETCH
#define XS_B_PREFETCH 1
#endif

namespace flashinfer::sparse_mla_sm120 {
namespace dsv41_xsplit {

constexpr int XS_NUM_HEADS = 8;
constexpr int CTL_GEN = 0;
constexpr int CTL_CLAIM = 1;
constexpr int CTL_FINISHED = 2;
constexpr int CTL_WORDS = 64;  // 256 B, keeps the flag arrays 256 B aligned

// Row layout in 4-byte words. Per-thread arrays are [slot][math thread] so stores and loads
// are coalesced.
constexpr int ROW_ALPHA = 0;                    // [16] alpha of every head row (dead rows too)
constexpr int ROW_NM = 16;                      // [16] running max after this tile
constexpr int ROW_SC = 32;                      // [7][16] normalised V-scale per chunk/head
constexpr int ROW_LS = 32 + 7 * 16;             // [8 warps][8 live rows] warp row sums
constexpr int ROW_W = ROW_LS + 64;              // [2][256] w of live row gid, entries e0/e1
constexpr int ROW_XV = ROW_W + 2 * 256;         // [7 vc][2][256] xv of live row gid
constexpr int ROW_RB = ROW_XV + 7 * 2 * 256;    // [4 ks][2][256] rope-B u32 (XS_ROPE_IN_WS)
#if XS_ROPE_IN_WS
constexpr int ROW_USED = ROW_RB + 4 * 2 * 256;
#else
constexpr int ROW_USED = ROW_RB;
#endif
constexpr int ROW_WORDS = (ROW_USED + 31) / 32 * 32;  // 128 B rows

struct XsWs {
  unsigned* ctl;                // [CTL_WORDS]
  unsigned long long* tmpub;    // [cap_units][16] = (epoch << 32) | float bits of the tile max
  unsigned* done;               // [cap_units] epoch when the unit's row is complete
  float* rows;                  // [cap_units][ROW_WORDS]
  int cap_units;
  int pad;
};

// One tile's B-replay operands for this thread (registers).
struct XsBRow {
  float alpha0, alpha1;
  float xv[7][2];
  float sc[7];
  float ls0, w0, w1;
  uint32_t rb[4][2];
};

struct XsSmem {
  int slot[2];
  int is_last;
  int pad;
};

template <ModelType MT, ComputeMode CM>
struct XsSmemLayout {
  using LMG = SmemLayoutMG<MT, CM>;
  static constexpr size_t OFF_XS = (LMG::TOTAL + 15) / 16 * 16;
  static constexpr size_t TOTAL = OFF_XS + sizeof(XsSmem);
  static_assert(TOTAL <= 101376, "xsplit smem exceeds the 99 KB opt-in limit");
};

// ---- memory-model helpers. Workspace traffic never uses ld.global.nc / __ldg. ----
__device__ __forceinline__ unsigned xs_ld_acquire_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long xs_ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned xs_ld_relaxed_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void xs_st_release_u32(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void xs_st_release_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void xs_st_f32(float* p, float v) { __stcg(p, v); }
__device__ __forceinline__ void xs_st_u32(float* p, uint32_t v) {
  __stcg(reinterpret_cast<unsigned*>(p), v);
}
__device__ __forceinline__ float xs_ld_f32(const float* p) { return __ldcg(p); }
__device__ __forceinline__ uint32_t xs_ld_u32(const float* p) {
  return __ldcg(reinterpret_cast<const unsigned*>(p));
}
__device__ __forceinline__ unsigned long long xs_globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// Spin bound: a correct call waits microseconds. 2 s means a broken invariant; trap so the
// failure is a CUDA error, never a silent hang of the serving process.
#ifndef XS_SPIN_TRAP_NS_VALUE
#define XS_SPIN_TRAP_NS_VALUE 2000000000ull
#endif
constexpr unsigned long long XS_SPIN_TRAP_NS = XS_SPIN_TRAP_NS_VALUE;

// Per-token geometry, the same clamps and tile counts as prefill_mg_impl (DUAL_CACHE,
// !ASSUME_FULL_TILES).
struct XsGeo {
  int topk_len;
  int topk_len_extra;
  int main_ni;
  int loop_bound;
};
__device__ __forceinline__ XsGeo xs_token_geo(const PrefillColdParams& cold, int s_i) {
  XsGeo g;
  int topk_len = cold.topk_length ? __ldg(cold.topk_length + s_i) : cold.topk;
  topk_len = topk_len < 0 ? 0 : (topk_len > cold.topk ? cold.topk : topk_len);
  const int actual_ni = (topk_len + BI - 1) / BI;
  const int topk_extra_declared = cold.topk_extra;
  int topk_len_extra =
      cold.topk_length_extra ? __ldg(cold.topk_length_extra + s_i) : topk_extra_declared;
  topk_len_extra =
      topk_len_extra < 0
          ? 0
          : (topk_len_extra > topk_extra_declared ? topk_extra_declared : topk_len_extra);
  g.topk_len = topk_len;
  g.topk_len_extra = topk_len_extra;
  g.main_ni = actual_ni;
  g.loop_bound = actual_ni + (topk_len_extra + BI - 1) / BI;
  return g;
}

// Rope-V B operand for one k-step, the exact loads of xv_rope_mma_mg::load_rope_v
// (idx < 0 -> 0, never reads slot 0 for a -1 entry).
template <ModelType MT, int PBS>
__device__ __forceinline__ void xs_rope_b(const int32_t* __restrict__ tile_indices,
                                          const uint8_t* __restrict__ KV_cache, int mwarp,
                                          int lane, size_t stride_kv_block, int ks,
                                          uint32_t& b0, uint32_t& b1) {
  using KV = KVCacheTraits<MT>;
  using IO = KVIOTraits<MT>;
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int n_start = mwarp * 8;
  const int dim_n = n_start + gid;
  const int k_base = ks * 16;
  auto load_rope_v = [&](int entry_offset) -> uint16_t {
    int idx = tile_indices[entry_offset];
    if (idx < 0) return 0;
    const uint8_t* base;
    if constexpr (KV::SCALE_IN_KV_SMEM) {
      base = KV_cache + (size_t)idx * IO::IO_STRIDE;
    } else {
      constexpr int pbs = PBS;
      int bi = idx / pbs;
      int li = idx % pbs;
      base = KV_cache + (size_t)bi * stride_kv_block + (size_t)li * IO::IO_STRIDE;
    }
    const bf16* rp = reinterpret_cast<const bf16*>(base + KV::KV_ROPE_GMEM_OFFSET);
    return *reinterpret_cast<const uint16_t*>(&rp[dim_n]);
  };
  uint16_t v0 = load_rope_v(k_base + tid * 2);
  uint16_t v1 = load_rope_v(k_base + tid * 2 + 1);
  uint16_t v8 = load_rope_v(k_base + tid * 2 + 8);
  uint16_t v9 = load_rope_v(k_base + tid * 2 + 9);
  b0 = (uint32_t)v0 | ((uint32_t)v1 << 16);
  b1 = (uint32_t)v8 | ((uint32_t)v9 << 16);
}

__device__ __forceinline__ float xs_rcp_approx(float x) {
  // == 1.f / x under -use_fast_math (MUFU.RCP, no refinement), pinned.
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

template <int PAGE_BLOCK_SIZE_EXTRA>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
    dsv41_xsplit_mg_dual_kernel(const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
                                const int32_t* __restrict__ indices,
                                const uint8_t* __restrict__ KV_cache_extra,
                                const int32_t* __restrict__ indices_extra,
                                bf16* __restrict__ output, float* __restrict__ out_lse,
                                const float* __restrict__ attn_sink,
                                __grid_constant__ const PrefillColdParams cold,
                                __grid_constant__ const XsWs ws) {
  constexpr ModelType MT = ModelType::DSV4;
  constexpr ComputeMode CM = ComputeMode::BF16;
  constexpr int NUM_HEADS = XS_NUM_HEADS;
  constexpr int PAGE_BLOCK_SIZE = 64;
  constexpr int MG_N_HG = 1;
  using KV = KVCacheTraits<MT>;
  using CT = ComputeTraits<MT, ComputeMode::FP8>;
  using LMG = SmemLayoutMG<MT, CM>;
  using SMG = SmemPtrsMG<MT, CM>;
  using XL = XsSmemLayout<MT, CM>;
  static constexpr int VALID_HPB = (NUM_HEADS < HPB) ? NUM_HEADS : HPB;
  static_assert(VALID_HPB == 8, "xsplit serves the padded NH=8 head group only");
  static_assert(PAGE_BLOCK_SIZE_EXTRA == 64 || PAGE_BLOCK_SIZE_EXTRA == 32,
                "xsplit serves extra PBS 64/32 (no W_FP8 row xor)");
  static_assert(CT::N_V_CHUNKS == 7 && CT::NT_PER_WARP_XV == 1 && CT::XV_KSTEPS == 2 &&
                    CT::ACC_TILES == 7,
                "row layout assumes 7 V chunks x 1 n-tile x 2 k-steps");
  static_assert(KV::SCALE_FORMAT != ScaleFormat::ARBITRARY_FP32 && !KV::SCALE_IN_KV_SMEM &&
                    KV::V_HAS_ROPE && KV::D_ROPE == 64,
                "DSV4 UE8M0 footer-scale layout expected");
  static_assert(MATH_THREADS == 256 && BLOCK_THREADS == 384 && BI == 64 && HPB == 16 &&
                    N_MATH_WARPS == 8 && ENTRIES_PER_WARP == 8,
                "tile geometry");
  static constexpr bool USE_WFP8_ROW_XOR = false;  // DUAL_CACHE && PBSX == 2 only
  static_assert(2 * HPB * BI * sizeof(bf16) <= LMG::SMEM_W_FP8_MG,
                "B replay double-buffers the bf16 rope A operand inside the W_FP8 region");

  const float sm_scale = cold.sm_scale;
  const int num_tokens = cold.num_tokens;
  const size_t stride_kv_block = cold.stride_kv_block;
  const size_t stride_kv_block_extra = cold.stride_kv_block_extra;
  const int topk_extra_declared = cold.topk_extra;
  const int NMAX = cold.topk / BI + (topk_extra_declared + BI - 1) / BI;
  const int NA = num_tokens * NMAX;
  const int NTOT = NA + num_tokens;
  const unsigned E = xs_ld_relaxed_u32(ws.ctl + CTL_GEN) + 1u;
  // The host gate guarantees NA <= cap_units; a violation would write past the workspace.
  if (NA > ws.cap_units) __trap();

  const int warp_rank = threadIdx.x / 32;
  const int wy = warp_rank / 4;

  extern __shared__ char smem_raw[];
  auto sm = SMG::init(smem_raw);
  XsSmem* xs = reinterpret_cast<XsSmem*>(smem_raw + XL::OFF_XS);

  if (threadIdx.x == 0) {
    mbarrier_init(sm.mbar_kv(0), 1);
    mbarrier_init(sm.mbar_kv(1), 1);
  }
  bar_sync_t<3, BLOCK_THREADS>();

  // ── IO warps: claim items in index order, gather A tiles into the parity buffers ──
  if (wy == 2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(32));

    const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
    const uint64_t kv_l2_policy = create_l2_evict_first_policy();

    auto claim_and_issue = [&](int k) -> int {
      const int buf = k & 1;
      if (io_tid == 0) xs->slot[buf] = (int)atomicAdd(ws.ctl + CTL_CLAIM, 1u);
      bar_sync_t<4, IO_THREADS>();
      const int item = *reinterpret_cast<volatile int*>(&xs->slot[buf]);
      if (item < NA) {
        const int s_i = item / NMAX;
        const int t = item - s_i * NMAX;
        const XsGeo g = xs_token_geo(cold, s_i);
        if (t < g.loop_bound) {
          // load_idx + issue_tile of the live IO warps, logical tile t.
          const bool is_main = t < g.main_ni;
          int staged = -1;
          if (io_tid < BI) {
            const int32_t* p =
                is_main ? (indices + (size_t)s_i * cold.topk + t * BI)
                        : (indices_extra + (size_t)s_i * topk_extra_declared + (t - g.main_ni) * BI);
            staged = __ldg(p + io_tid);
          }
          if (is_main) {
            io_gather_scales<MT, PAGE_BLOCK_SIZE>(sm.kv_scale_buf(buf), staged, KV_cache, io_tid,
                                                  stride_kv_block);
            __threadfence_block();
            io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(sm.kv_buf(buf), staged, KV_cache,
                                                           sm.mbar_kv(buf), io_tid,
                                                           stride_kv_block, kv_l2_policy);
          } else {
            io_gather_scales<MT, PAGE_BLOCK_SIZE_EXTRA>(sm.kv_scale_buf(buf), staged,
                                                        KV_cache_extra, io_tid,
                                                        stride_kv_block_extra);
            __threadfence_block();
            io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE_EXTRA, true>(
                sm.kv_buf(buf), staged, KV_cache_extra, sm.mbar_kv(buf), io_tid,
                stride_kv_block_extra, kv_l2_policy);
          }
          return item;
        }
      }
      // Empty A unit, B job or END: complete the parity phase without a transfer.
      if (io_tid == 0) mbarrier_arrive(sm.mbar_kv(buf));
      return item;
    };

    int k = 0;
    int item = claim_and_issue(0);
#pragma unroll 1
    while (item < NTOT) {
      const int next = claim_and_issue(k + 1);
      bar_sync_alt<1, 5, BLOCK_THREADS>(k & 1);
      item = next;
      k++;
    }

    // ── Math warps ──────────────────────────────────────────────────
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(232));

    const int lane = threadIdx.x & 31;
    const int mwarp = warp_rank;
    const int gid = lane >> 2, tid = lane & 3;
    const float sm_scale_log2e = sm_scale * LOG2E;
    QRopeRegs<MT> q_rope_regs[MG_N_HG];
    int cur_tok = -1;

#pragma unroll 1
    for (int k = 0;; k++) {
      const int buf = k & 1;
      mbarrier_wait_parity(sm.mbar_kv(buf), (k >> 1) & 1);
      const int item = *reinterpret_cast<volatile int*>(&xs->slot[buf]);
      if (item >= NTOT) break;

      if (item < NA) {
        // ════════════════════════ A unit (s_i, ti) ════════════════════════
        const int s_i = item / NMAX;
        const int ti = item - s_i * NMAX;
        const XsGeo geo = xs_token_geo(cold, s_i);
        if (ti < geo.loop_bound) {
          const int topk_len = geo.topk_len;
          const int topk_len_extra = geo.topk_len_extra;
          const int main_ni = geo.main_ni;
          const unsigned u = (unsigned)item;
          float* row = ws.rows + (size_t)u * ROW_WORDS;
          const int32_t* idx_base = indices + (size_t)s_i * cold.topk;
          const int32_t* idx_base_extra = indices_extra + (size_t)s_i * topk_extra_declared;

          if (s_i != cur_tok) {
            // Q prologue of the live kernel (h_start = 0: REPLICATE_H = 1 at NH = 8). The
            // q_rope scratch aliases W_FP8: every earlier item ends in a math barrier after
            // its last W_FP8 read, and this barrier keeps the scratch until all lanes hold
            // their rope registers.
            const bf16* q_base_g = Q + (size_t)s_i * NUM_HEADS * KV::D_QK;
            load_q_bf16_to_smem<MT, MATH_THREADS>(sm.q_nope_bf16(0), sm.q_rope(), q_base_g,
                                                  VALID_HPB);
            q_rope_regs[0] = preload_q_rope_regs<MT>(sm.q_rope(), lane);
            cur_tok = s_i;
            bar_sync_t<2, MATH_THREADS>();
          }

          uint8_t* kv_smem = sm.kv_buf(buf);
          const int qk_nb = mwarp * ENTRIES_PER_WARP;
          uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;

          const int32_t* ib;
          const uint8_t* kv_global;
          size_t stride_kv_block_now;
          const bool is_main = (ti < main_ni);
          ib = is_main ? (idx_base + ti * BI) : (idx_base_extra + (ti - main_ni) * BI);
          kv_global = is_main ? KV_cache : KV_cache_extra;
          stride_kv_block_now = is_main ? stride_kv_block : stride_kv_block_extra;

          const uint8_t* entry_base_gid;
          {
            const int idx = ib[qk_nb + gid];
            if (is_main) {
              entry_base_gid =
                  prefill_kv_entry_base<MT, PAGE_BLOCK_SIZE>(KV_cache, idx, stride_kv_block);
            } else {
              entry_base_gid = prefill_kv_entry_base<MT, PAGE_BLOCK_SIZE_EXTRA>(
                  KV_cache_extra, idx, stride_kv_block_extra);
            }
          }

          KVRopePrefetch<MT> rope_pf = prefetch_kv_rope<MT>(
              reinterpret_cast<const bf16*>(entry_base_gid + KV::KV_ROPE_GMEM_OFFSET), lane);

          for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
            sm.w_head_sc_all()[i] = 0.f;

          float w_grp[MG_N_HG][4];
          float vsc_cache[CT::N_V_CHUNKS][2];

          // ── QK + softmax (live: MG_N_HG == 1 branch, CM == BF16) ──
#pragma unroll
          for (int g = 0; g < MG_N_HG; g++) {
            const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;
            float qk[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
              float scale_f;
              scale_f = ue8m0_to_fp32(
                  sm.kv_scale_buf(buf)[(qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk]);
#pragma unroll
              for (int ks = 0; ks < KV::QUANT_TILE / 16; ks++) {
                int ko = blk * KV::QUANT_TILE + ks * 16;
                uint32_t a0, a1, a2, a3;
                ldmatrix_load_A_bf16(a0, a1, a2, a3, sm.q_nope_bf16(g) + ko,
                                     KV::Q_NOPE_BF16_STRIDE, lane);
                uint16_t p0 = *reinterpret_cast<const uint16_t*>(kv_gid_base + ko + 2 * tid);
                uint16_t p1 = *reinterpret_cast<const uint16_t*>(kv_gid_base + ko + 2 * tid + 8);
                uint32_t f16x2_0, f16x2_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2_0) : "h"(p0));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2_1) : "h"(p1));
                __half2 h2_0 = *reinterpret_cast<__half2*>(&f16x2_0);
                __half2 h2_1 = *reinterpret_cast<__half2*>(&f16x2_1);
                float fk0 = __low2float(h2_0) * scale_f, fk1 = __high2float(h2_0) * scale_f;
                float fk2 = __low2float(h2_1) * scale_f, fk3 = __high2float(h2_1) * scale_f;
                uint32_t b0, b1;
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b0) : "f"(fk1), "f"(fk0));
                asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b1) : "f"(fk3), "f"(fk2));
                MmaBf16Result r =
                    mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, qk[0], qk[1], qk[2], qk[3]);
                qk[0] = r.d0;
                qk[1] = r.d1;
                qk[2] = r.d2;
                qk[3] = r.d3;
              }
            }

            compute_qk_rope<MT>(qk, q_rope_regs[g], rope_pf);

            {
              int e0 = qk_nb + tid * 2, e1 = e0 + 1;
              if (ib[e0] < 0) {
                qk[0] = -1e30f;
                qk[2] = -1e30f;
              }
              if (ib[e1] < 0) {
                qk[1] = -1e30f;
                qk[3] = -1e30f;
              }
              if (is_main) {
                if (cold.topk_length != nullptr) {
                  int a0 = ti * BI + e0, a1 = ti * BI + e1;
                  if (a0 >= topk_len) {
                    qk[0] = -1e30f;
                    qk[2] = -1e30f;
                  }
                  if (a1 >= topk_len) {
                    qk[1] = -1e30f;
                    qk[3] = -1e30f;
                  }
                }
              } else {
                int a0 = (ti - main_ni) * BI + e0, a1 = (ti - main_ni) * BI + e1;
                if (a0 >= topk_len_extra) {
                  qk[0] = -1e30f;
                  qk[2] = -1e30f;
                }
                if (a1 >= topk_len_extra) {
                  qk[1] = -1e30f;
                  qk[3] = -1e30f;
                }
              }
            }

            float s[4] = {qk[0] * sm_scale_log2e, qk[1] * sm_scale_log2e,
                          qk[2] * sm_scale_log2e, qk[3] * sm_scale_log2e};

            float lm0, lm1;
            softmax_warp_max(s, lm0, lm1);
            if (tid == 0) {
              sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid] = lm0;
              sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid + 8] = lm1;
            }
            w_grp[g][0] = s[0];
            w_grp[g][1] = s[1];
            w_grp[g][2] = s[2];
            w_grp[g][3] = s[3];
          }
          bar_sync_t<2, MATH_THREADS>();

          // Cross-warp tile max (live order), publish, then the in-order look-back fold.
          if (threadIdx.x < MG_N_HG * HPB) {
            int g = threadIdx.x / HPB, h = threadIdx.x % HPB;
            float tm = -1e30f;
#pragma unroll
            for (int w = 0; w < N_MATH_WARPS; w++)
              tm = fmaxf(tm, sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + w * HPB + h]);
            xs_st_release_u64(ws.tmpub + (size_t)u * HPB + h,
                              ((unsigned long long)E << 32) | (unsigned long long)__float_as_uint(tm));
            // old_m = m_smem after tiles 0..ti-1 of this token: m starts at -1e30 and each
            // tile does m = fmaxf(m, tm_tile), in tile order.
            float old_m = -1e30f;
#if XS_ARM != 1
#pragma unroll 1
            for (int tt = 0; tt < ti; tt++) {
              const unsigned long long* pp = ws.tmpub + ((size_t)s_i * NMAX + tt) * HPB + h;
              unsigned long long v = xs_ld_acquire_u64(pp);
              if ((unsigned)(v >> 32) != E) {
                const unsigned long long t0 = xs_globaltimer();
                int spins = 0;
                do {
                  if (++spins > 32) __nanosleep(32);
                  if (xs_globaltimer() - t0 > XS_SPIN_TRAP_NS) __trap();
                  v = xs_ld_acquire_u64(pp);
                } while ((unsigned)(v >> 32) != E);
              }
              old_m = fmaxf(old_m, __uint_as_float((unsigned)v));
            }
#endif
            float nm = fmaxf(old_m, tm);
            float alpha = exp2f(old_m - nm);
            sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + h] = alpha;
            sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + HPB + h] = nm;
#if XS_ARM != 3
            xs_st_f32(row + ROW_ALPHA + h, alpha);
            xs_st_f32(row + ROW_NM + h, nm);
#endif
          }
          bar_sync_t<2, MATH_THREADS>();

          const int e0i = qk_nb + tid * 2, e1i = e0i + 1;
#pragma unroll
          for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
            vsc_cache[vc][0] =
                ue8m0_to_fp32(sm.kv_scale_buf(buf)[e0i * KV::SCALE_BYTES_PER_TOKEN + vc]);
            vsc_cache[vc][1] =
                ue8m0_to_fp32(sm.kv_scale_buf(buf)[e1i * KV::SCALE_BYTES_PER_TOKEN + vc]);
          }

          // Exponentiate (the guarded rescale of acc_o / acc_rope / l moves to the B replay).
#pragma unroll
          for (int g = 0; g < MG_N_HG; g++) {
            float nm0 = sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + HPB + gid];
            float nm1 = sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + HPB + gid + 8];

            float w0 = exp2f(w_grp[g][0] - nm0), w1 = exp2f(w_grp[g][1] - nm0);
            float w2 = exp2f(w_grp[g][2] - nm1), w3 = exp2f(w_grp[g][3] - nm1);
            w_grp[g][0] = w0;
            w_grp[g][1] = w1;
            w_grp[g][2] = w2;
            w_grp[g][3] = w3;

            float ls0, ls1;
            softmax_warp_sum(w0, w1, w2, w3, ls0, ls1);
            // ls is identical in the 4 lanes of a quad (one commutative FADD per level).
#if XS_ARM != 3
            if (tid == 0) xs_st_f32(row + ROW_LS + mwarp * 8 + gid, ls0);
#endif
            (void)ls1;

#pragma unroll
            for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
              float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
              float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
              float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
              atomicMax(reinterpret_cast<int*>(
                            &sm.w_head_sc_all()[g * SMG::WSC_GRP_STRIDE + vc * HPB + gid]),
                        __float_as_int(fmaxf(ws00, ws01)));
              atomicMax(reinterpret_cast<int*>(
                            &sm.w_head_sc_all()[g * SMG::WSC_GRP_STRIDE + vc * HPB + gid + 8]),
                        __float_as_int(fmaxf(ws10, ws11)));
            }
          }
          bar_sync_t<2, MATH_THREADS>();

          for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
            sm.w_head_sc_all()[i] = fmaxf(sm.w_head_sc_all()[i], 1e-10f) / FP8_MAX;
          bar_sync_t<2, MATH_THREADS>();
#if XS_ARM != 3
          // Normalised V scales of this tile (sc0 = vc_sc[gid] of every XV FFMA).
          if (threadIdx.x < CT::N_V_CHUNKS * HPB)
            xs_st_f32(row + ROW_SC + threadIdx.x, sm.w_head_sc_all()[threadIdx.x]);
#endif

          // ── XV nope (live UE8M0 path, MG_N_HG == 1 branch); xv written back ──
#pragma unroll
          for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
            uint8_t* wfp8_parity = sm.w_fp8() + (vc & 1) * SMG::WFP8_PARITY_STRIDE;
            float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
#pragma unroll
            for (int g = 0; g < MG_N_HG; g++) {
              float* vc_sc = sm.w_head_sc_all() + g * SMG::WSC_GRP_STRIDE + vc * HPB;
              uint8_t* cur_wfp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
              float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
              float w0 = w_grp[g][0], w1 = w_grp[g][1];
              float w2 = w_grp[g][2], w3 = w_grp[g][3];
              float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
              float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
              __nv_fp8_e4m3 f00(ws00 * si0);
              __nv_fp8_e4m3 f01(ws01 * si0);
              __nv_fp8_e4m3 f10(ws10 * si1);
              __nv_fp8_e4m3 f11(ws11 * si1);
              int wrow0 = gid, wrow1 = gid + 8;
              if constexpr (USE_WFP8_ROW_XOR) {
                wrow0 = wfp8_row_xor(wrow0);
                wrow1 = wfp8_row_xor(wrow1);
              }
              cur_wfp8[wrow0 * (BI + 16) + e0i] = f00.__x;
              cur_wfp8[wrow0 * (BI + 16) + e1i] = f01.__x;
              cur_wfp8[wrow1 * (BI + 16) + e0i] = f10.__x;
              cur_wfp8[wrow1 * (BI + 16) + e1i] = f11.__x;
            }
            bar_sync_t<2, MATH_THREADS>();

#pragma unroll
            for (int g = 0; g < MG_N_HG; g++) {
              uint8_t* cur_wfp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
#pragma unroll
              for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
                float xv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kstep = 0; kstep < CT::XV_KSTEPS; kstep++) {
                  int ko = kstep * 32;
                  uint32_t a0, a1, a2, a3, b0, b1;
                  ldmatrix_load_A_fp8_layout<USE_WFP8_ROW_XOR>(a0, a1, a2, a3, cur_wfp8 + ko,
                                                               BI + 16, lane);
                  d2_load_b_fp8<KV::KV_SMEM_STRIDE>(b0, b1, kv_smem, kstep * 32, dim, lane);
                  MmaFp8Result r =
                      mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, xv[0], xv[1], xv[2], xv[3]);
                  xv[0] = r.d0;
                  xv[1] = r.d1;
                  xv[2] = r.d2;
                  xv[3] = r.d3;
                }
#if XS_ARM != 3
                xs_st_f32(row + ROW_XV + (vc * 2 + 0) * MATH_THREADS + threadIdx.x, xv[0]);
                xs_st_f32(row + ROW_XV + (vc * 2 + 1) * MATH_THREADS + threadIdx.x, xv[1]);
#endif
              }
            }
          }

#if XS_ARM != 3
          // w of live row gid (entries e0i/e1i) and the rope-V B operands for the B replay.
          xs_st_f32(row + ROW_W + 0 * MATH_THREADS + threadIdx.x, w_grp[0][0]);
          xs_st_f32(row + ROW_W + 1 * MATH_THREADS + threadIdx.x, w_grp[0][1]);
#if XS_ROPE_IN_WS
#pragma unroll
          for (int ks = 0; ks < BI / 16; ks++) {
            uint32_t b0, b1;
            if (is_main) {
              xs_rope_b<MT, PAGE_BLOCK_SIZE>(ib, kv_global, mwarp, lane, stride_kv_block_now, ks,
                                             b0, b1);
            } else {
              xs_rope_b<MT, PAGE_BLOCK_SIZE_EXTRA>(ib, kv_global, mwarp, lane,
                                                   stride_kv_block_now, ks, b0, b1);
            }
            xs_st_u32(row + ROW_RB + (ks * 2 + 0) * MATH_THREADS + threadIdx.x, b0);
            xs_st_u32(row + ROW_RB + (ks * 2 + 1) * MATH_THREADS + threadIdx.x, b1);
          }
#endif
#endif
          bar_sync_t<2, MATH_THREADS>();
          if (threadIdx.x == 0) {
            __threadfence();
            xs_st_release_u32(ws.done + u, E);
          }
        }
      } else {
        // ════════════════════════ B job: token s_i ════════════════════════
        const int s_i = item - NA;
#if XS_ARM != 2
        const XsGeo geo = xs_token_geo(cold, s_i);
        const int n = geo.loop_bound;
        const int qk_nb = mwarp * ENTRIES_PER_WARP;

        float acc_o[CT::ACC_TILES][2];
#pragma unroll
        for (int t = 0; t < CT::ACC_TILES; t++) acc_o[t][0] = acc_o[t][1] = 0.f;
        float acc_rope[4] = {0.f, 0.f, 0.f, 0.f};
        float warp_l = 0.f;

        // Wait for tile ti's row (thread 0 acquires the done flag, math barrier), then load this
        // thread's operands with ld.global.cg.
        auto wait_row = [&](int ti) {
          if (threadIdx.x == 0) {
            const unsigned* pd = ws.done + ((size_t)s_i * NMAX + ti);
            if (xs_ld_acquire_u32(pd) != E) {
              const unsigned long long t0 = xs_globaltimer();
              int spins = 0;
              do {
                if (++spins > 32) __nanosleep(32);
                if (xs_globaltimer() - t0 > XS_SPIN_TRAP_NS) __trap();
              } while (xs_ld_acquire_u32(pd) != E);
            }
          }
          bar_sync_t<2, MATH_THREADS>();
        };
        auto load_row = [&](int ti, XsBRow& r) {
          const float* row = ws.rows + ((size_t)s_i * NMAX + ti) * ROW_WORDS;
          r.alpha0 = xs_ld_f32(row + ROW_ALPHA + gid);
          r.alpha1 = xs_ld_f32(row + ROW_ALPHA + gid + 8);  // dead row: guard only
#pragma unroll
          for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
            r.xv[vc][0] = xs_ld_f32(row + ROW_XV + (vc * 2 + 0) * MATH_THREADS + threadIdx.x);
            r.xv[vc][1] = xs_ld_f32(row + ROW_XV + (vc * 2 + 1) * MATH_THREADS + threadIdx.x);
            r.sc[vc] = xs_ld_f32(row + ROW_SC + vc * HPB + gid);
          }
          r.ls0 = xs_ld_f32(row + ROW_LS + mwarp * 8 + gid);
          r.w0 = xs_ld_f32(row + ROW_W + 0 * MATH_THREADS + threadIdx.x);
          r.w1 = xs_ld_f32(row + ROW_W + 1 * MATH_THREADS + threadIdx.x);
#if XS_ROPE_IN_WS
#pragma unroll
          for (int ks = 0; ks < BI / 16; ks++) {
            r.rb[ks][0] = xs_ld_u32(row + ROW_RB + (ks * 2 + 0) * MATH_THREADS + threadIdx.x);
            r.rb[ks][1] = xs_ld_u32(row + ROW_RB + (ks * 2 + 1) * MATH_THREADS + threadIdx.x);
          }
#else
          const bool is_main = ti < geo.main_ni;
          const int32_t* ib =
              is_main ? (indices + (size_t)s_i * cold.topk + ti * BI)
                      : (indices_extra + (size_t)s_i * topk_extra_declared +
                         (ti - geo.main_ni) * BI);
#pragma unroll
          for (int ks = 0; ks < BI / 16; ks++) {
            if (is_main)
              xs_rope_b<MT, PAGE_BLOCK_SIZE>(ib, KV_cache, mwarp, lane, stride_kv_block, ks,
                                             r.rb[ks][0], r.rb[ks][1]);
            else
              xs_rope_b<MT, PAGE_BLOCK_SIZE_EXTRA>(ib, KV_cache_extra, mwarp, lane,
                                                   stride_kv_block_extra, ks, r.rb[ks][0],
                                                   r.rb[ks][1]);
          }
#endif
        };

        XsBRow cur, nxt;
        if (n > 0) {
          wait_row(0);
          load_row(0, cur);
        }
#pragma unroll 1
        for (int ti = 0; ti < n; ti++) {
#if XS_B_PREFETCH
          // Overlap the next row's L2 loads with this tile's replay (load scheduling only).
          if (ti + 1 < n) {
            wait_row(ti + 1);
            load_row(ti + 1, nxt);
          }
#endif
          // Live: if (alpha0 < 1 || alpha1 < 1) { acc *= alpha } -- FMUL.FTZ, guard includes
          // the dead row's alpha1 (F4).
          if (cur.alpha0 < 1.0f || cur.alpha1 < 1.0f) {
#pragma unroll
            for (int t = 0; t < CT::ACC_TILES; t++) {
              acc_o[t][0] = __fmul_rn(acc_o[t][0], cur.alpha0);
              acc_o[t][1] = __fmul_rn(acc_o[t][1], cur.alpha0);
            }
            acc_rope[0] = __fmul_rn(acc_rope[0], cur.alpha0);
            acc_rope[1] = __fmul_rn(acc_rope[1], cur.alpha0);
            acc_rope[2] = __fmul_rn(acc_rope[2], cur.alpha1);
            acc_rope[3] = __fmul_rn(acc_rope[3], cur.alpha1);
            warp_l = __fmul_rn(warp_l, cur.alpha0);
          }
          // Live: warp_l_partial += ls (FADD.FTZ, separate from the guarded FMUL; F2).
          warp_l = __fadd_rn(warp_l, cur.ls0);
          // Live: acc_o += xv * sc, one FFMA.FTZ per element.
#pragma unroll
          for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
            acc_o[vc][0] = __fmaf_rn(cur.xv[vc][0], cur.sc[vc], acc_o[vc][0]);
            acc_o[vc][1] = __fmaf_rn(cur.xv[vc][1], cur.sc[vc], acc_o[vc][1]);
          }
          // Live xv_rope_mma_mg: A = bf16(w) staged in smem, 4 chained m16n8k16 with
          // C = acc_rope. Dead rows 8-15 are zero here; MMA rows are independent, so live
          // rows see the same A, B and C bits as the live kernel (F3). The staging buffer
          // alternates by tile parity (2 x 2 KB inside the W_FP8 region), so this tile's
          // stores never meet the previous tile's ldmatrix reads.
          bf16* weight_smem = reinterpret_cast<bf16*>(sm.w_fp8()) + (ti & 1) * (HPB * BI);
          weight_smem[gid * BI + qk_nb + tid * 2] = __float2bfloat16(cur.w0);
          weight_smem[gid * BI + qk_nb + tid * 2 + 1] = __float2bfloat16(cur.w1);
          weight_smem[(gid + 8) * BI + qk_nb + tid * 2] = __float2bfloat16(0.f);
          weight_smem[(gid + 8) * BI + qk_nb + tid * 2 + 1] = __float2bfloat16(0.f);
          bar_sync_t<2, MATH_THREADS>();
#pragma unroll
          for (int ks = 0; ks < BI / 16; ks++) {
            uint32_t a0, a1, a2, a3;
            ldmatrix_load_A_bf16(a0, a1, a2, a3, weight_smem + ks * 16, BI, lane);
            MmaBf16Result r = mma_bf16_m16n8k16(a0, a1, a2, a3, cur.rb[ks][0], cur.rb[ks][1],
                                                acc_rope[0], acc_rope[1], acc_rope[2],
                                                acc_rope[3]);
            acc_rope[0] = r.d0;
            acc_rope[1] = r.d1;
            acc_rope[2] = r.d2;
            acc_rope[3] = r.d3;
          }
          if (ti + 1 < n) {
#if XS_B_PREFETCH
            cur = nxt;
#else
            wait_row(ti + 1);
            load_row(ti + 1, cur);
#endif
          }
        }
        // The epilogue's reduce_buf / staging writes are fenced from the last ldmatrix by the
        // barrier below (reduce_buf and the KV staging buffer do not alias the weight buffers).

        // ── Finalize deferred row_sum (live order) ──
        if (tid == 0) {
          sm.reduce_buf()[mwarp * HPB + gid] = warp_l;
          sm.reduce_buf()[mwarp * HPB + gid + 8] = 0.f;
        }
        bar_sync_t<2, MATH_THREADS>();
        if (threadIdx.x < HPB) {
          int h = threadIdx.x;
          float ts = 0.f;
#pragma unroll
          for (int w = 0; w < N_MATH_WARPS; w++) ts = __fadd_rn(ts, sm.reduce_buf()[w * HPB + h]);
          sm.l_smem()[h] = ts;
          sm.m_smem()[h] =
              (n > 0) ? xs_ld_f32(ws.rows + ((size_t)s_i * NMAX + (n - 1)) * ROW_WORDS + ROW_NM + h)
                      : -1e30f;
        }
        bar_sync_t<2, MATH_THREADS>();

        // ── Epilogue (live, g = 0, h_start = 0); staging in this item's own KV buffer ──
        bf16* staging_bf16 = reinterpret_cast<bf16*>(sm.kv_buf(buf));
        constexpr int BF16_STAGING_STRIDE = KV::D_V;
        constexpr size_t h_stride = KV::D_V;
        constexpr size_t token_stride = (size_t)NUM_HEADS * KV::D_V;
        {
          float il0;
          if (cold.attn_sink != nullptr) {
            int h0 = gid;
            // live SASS: FFMA(sink, LOG2E, -m), EX2, FADD(l, .), RCP.approx, d > 0 select
            float e0 = exp2f(__fmaf_rn(__ldg(cold.attn_sink + h0), LOG2E, -sm.m_smem()[gid]));
            float d0 = __fadd_rn(sm.l_smem()[gid], e0);
            il0 = (d0 > 0.f) ? xs_rcp_approx(d0) : 0.f;
          } else {
            float l0 = sm.l_smem()[gid];
            il0 = (l0 > 0.f) ? xs_rcp_approx(l0) : 0.f;
          }

#pragma unroll
          for (int t = 0; t < CT::ACC_TILES; t++) {
            constexpr int _NT8 = CT::NT_PER_WARP_XV * 8;
            int c = t / CT::NT_PER_WARP_XV, lnt = t % CT::NT_PER_WARP_XV;
            int d0 = c * CT::V_CHUNK + mwarp * _NT8 + lnt * 8 + tid * 2;
            staging_bf16[gid * BF16_STAGING_STRIDE + d0] =
                __float2bfloat16(__fmul_rn(acc_o[t][0], il0));
            staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1] =
                __float2bfloat16(__fmul_rn(acc_o[t][1], il0));
          }
          {
            int n_start = mwarp * 8;
            int d0 = KV::D_NOPE + n_start + tid * 2;
            staging_bf16[gid * BF16_STAGING_STRIDE + d0] =
                __float2bfloat16(__fmul_rn(acc_rope[0], il0));
            staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1] =
                __float2bfloat16(__fmul_rn(acc_rope[1], il0));
          }
          bar_sync_t<2, MATH_THREADS>();

          {
            const size_t out_base = (size_t)s_i * token_stride;
            constexpr int BF16_PER_STORE = 8;
            constexpr int STORES_PER_HEAD = KV::D_V / BF16_PER_STORE;
            for (int idx = threadIdx.x; idx < VALID_HPB * STORES_PER_HEAD; idx += MATH_THREADS) {
              int h = idx / STORES_PER_HEAD;
              int d8 = (idx - h * STORES_PER_HEAD) * BF16_PER_STORE;
              uint4 v =
                  *reinterpret_cast<const uint4*>(&staging_bf16[h * BF16_STAGING_STRIDE + d8]);
              *reinterpret_cast<uint4*>(&output[out_base + h * h_stride + d8]) = v;
            }
          }

          if (threadIdx.x < VALID_HPB) {
            int h = threadIdx.x;
            float lse = softmax_lse(sm.m_smem()[h], sm.l_smem()[h]);
            if (cold.attn_sink != nullptr) {
              // live SASS: FMUL(sink, LOG2E); if lse != -1e30: FADD(sl, -lse), EX2,
              // FADD(., 1), LG2, FADD(., lse); else lse = sl
              float sink_log2 = __fmul_rn(__ldg(cold.attn_sink + h), LOG2E);
              if (lse != -1e30f)
                lse = __fadd_rn(log2f(__fadd_rn(exp2f(__fadd_rn(sink_log2, -lse)), 1.f)), lse);
              else
                lse = sink_log2;
            }
            size_t lse_idx = (size_t)s_i * cold.stride_out_lse + h;
            out_lse[lse_idx] = lse;
          }
          // The next item's first smem write (weight/W_FP8/q scratch, staging) is fenced by
          // this barrier.
          bar_sync_t<2, MATH_THREADS>();
        }
#else
        (void)s_i;
#endif
      }
      bar_arrive_alt<1, 5, BLOCK_THREADS>(buf);
    }

    // ── All items done: last CTA resets the claim state and publishes the next epoch ──
    bar_sync_t<2, MATH_THREADS>();
    if (threadIdx.x == 0) {
      __threadfence();
      const unsigned prev = atomicAdd(ws.ctl + CTL_FINISHED, 1u);
      xs->is_last = (prev == gridDim.x - 1) ? 1 : 0;
    }
    bar_sync_t<2, MATH_THREADS>();
    if (xs->is_last) {
      __threadfence();
      unsigned next_gen = E;
      if (next_gen == 0xFFFFFFFFu) {
        // 32-bit epoch wrap: clear every flag so no stale word can match a future epoch.
        for (int j = threadIdx.x; j < ws.cap_units * HPB; j += MATH_THREADS) ws.tmpub[j] = 0ull;
        for (int j = threadIdx.x; j < ws.cap_units; j += MATH_THREADS) ws.done[j] = 0u;
        next_gen = 0u;
      }
      __threadfence();
      bar_sync_t<2, MATH_THREADS>();
      if (threadIdx.x == 0) {
        ws.ctl[CTL_CLAIM] = 0u;
        ws.ctl[CTL_FINISHED] = 0u;
        __threadfence();
        xs_st_release_u32(ws.ctl + CTL_GEN, next_gen);
      }
    }
  }
}

}  // namespace dsv41_xsplit
}  // namespace flashinfer::sparse_mla_sm120
