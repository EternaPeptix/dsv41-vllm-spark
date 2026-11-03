#!/usr/bin/env python3
"""mk_variant.py <control rdma_ar.cu 6173b362> <out rdma_ar.cu>: applies the g130 edits as exact, asserted replacements."""
import hashlib
import sys

src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
h = hashlib.sha256(s.encode()).hexdigest()
if not h.startswith("6173b362"):
    sys.exit(f"mk_variant.py: base sha {h[:8]} is not 6173b362")


def rep(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"mk_variant.py: expected {count} matches, found {n}:\n{old[:200]}")
    s = s.replace(old, new)


# ---- header
rep("""// safe to capture in CUDA graphs (kernel args never change between replays).
#include <torch/extension.h>""", """// safe to capture in CUDA graphs (kernel args never change between replays).
//
// g130 rdma (2026-11-02), on top of g2 colltrim 6173b362. Every flag defaults OFF; unset == production behaviour, and the
// production kernels (kPub == 0) are instruction-identical (tests/sass_gate.py). Wire protocol, Params, Ctrl, DevState,
// slots, sum order and rounding are untouched. Read once per instance at construction; WINDOW.md gives the commands.
//   RDMA_AR_SPLIT_POST=1  A   host only: each WR gets its own ibv_post_send (nreq == 1 qualifies for BlueFlame)
//   RDMA_AR_PUB_V1=1      B   kernel: phase-1 publish compares posted_seq with seq - 1 (no send_seq load)
//   RDMA_AR_PUB_V2=1      B   kernel: phase-2 publish has no spin at all (no send2_seq / posted2_seq load)
//   RDMA_AR_HYG_PROXY=1   C   host only: proxy spins on the one doorbell that rings next, reaps CQs after each post
//   RDMA_AR_HYG_POLL=1    C   kernel: GPU spins read the clock on the first miss and once per 1024 passes
// B and C-poll select other kernel instantiations at launch (parallel-load path only). See DESIGN notes in WINDOW.md.
#include <torch/extension.h>""")

# ---- kernel-variant bits
rep("""enum Mode : int { kOneShot = 0, kGather = 1, kTwoShot = 2, kGatherCols = 3 };  // kGatherCols: g2 colltrim
""", """enum Mode : int { kOneShot = 0, kGather = 1, kTwoShot = 2, kGatherCols = 3 };  // kGatherCols: g2 colltrim
// g130: kernel variants, chosen per launch by instantiation and never by a kernel argument, so kPub == 0 is production
enum PubBits : int { kPubV1 = 1, kPubV2 = 2, kPollThin = 4 };
""")

# ---- thinned poll
rep("""// True for exactly one thread: thread 0 of the last CTA through a grid-wide arrival counter""",
    """// g130 RDMA_AR_HYG_POLL: same wait and same timeout contract; the clock is read on the first miss and on every 1024th
// pass instead of on every pass, so a timeout can be late by about 1024 polls (about 1 ms against a 600 s limit).
__device__ __forceinline__ void spin_until_thin(const uint64_t* f, uint64_t seq, const Params& p, int who, int phase) {
  if (ld_acquire_sys(f) >= seq) return;
  const unsigned long long t0 = now_ns();
  unsigned int it = 0;
  while (ld_acquire_sys(f) < seq) {
    if ((++it & 1023u) == 0 && now_ns() - t0 > (unsigned long long)p.timeout_ns) {
      printf("rdma_ar rank %d: timeout waiting for %d phase %d (seq %llu)\\n", p.rank, who, phase,
             (unsigned long long)seq);
      asm volatile("trap;");
    }
  }
}
// kPub without kPollThin expands to exactly the production spin_until call.
#define G130_SPIN(f, s, who, phase)                   \\
  do {                                                \\
    if constexpr ((kPub & kPollThin) != 0)            \\
      spin_until_thin((f), (s), p, (who), (phase));   \\
    else                                              \\
      spin_until((f), (s), p, (who), (phase));        \\
  } while (0)

// True for exactly one thread: thread 0 of the last CTA through a grid-wide arrival counter""")

# ---- ar_kernel_pl only: the legacy ar_kernel stays byte-for-byte
rep("""template <int kMode, bool kF32, int kW>
__global__ void __launch_bounds__(kThreads, 1) ar_kernel_pl(Params p) {""",
    """template <int kMode, bool kF32, int kW, int kPub>
__global__ void __launch_bounds__(kThreads, 1) ar_kernel_pl(Params p) {""")

rep("""  if (last_cta(&p.st->arrive, &s_last)) {
    spin_until(&p.ctrl->posted_seq, ld_acquire_sys(&p.ctrl->send_seq), p, -1, 0);
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send_bytes) = (uint64_t)p.nvec * 16;""",
    """  if (last_cta(&p.st->arrive, &s_last)) {
    // g130 V1: every op publishes phase 1 exactly once and st->seq counts ops, so what was last published here is
    // seq - 1 and the send_seq load adds nothing. The guard itself must stay: this rank's kernel can finish while its own
    // proxy is stalled (its flags come from the peers' proxies), and the descriptor is a single mailbox.
    if constexpr ((kPub & kPubV1) != 0)
      G130_SPIN(&p.ctrl->posted_seq, seq - 1, -1, 0);
    else
      G130_SPIN(&p.ctrl->posted_seq, ld_acquire_sys(&p.ctrl->send_seq), -1, 0);
    *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send_bytes) = (uint64_t)p.nvec * 16;""")

rep("""  // 2) wait for every peer's phase-1 contribution
  if (threadIdx.x < p.world && threadIdx.x != p.rank)
    spin_until(p.flags + slot * p.world + threadIdx.x, seq, p, threadIdx.x, 1);
  __syncthreads();
  unsigned long long t_flags = 0;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) t_flags = now_ns();

  const char* rbase = p.recv + (long long)slot * p.world * p.max_bytes;
  if constexpr (kMode == kGather) {
    for (long long i = tid; i < p.nvec; i += stride) {""",
    """  // 2) wait for every peer's phase-1 contribution
  if (threadIdx.x < p.world && threadIdx.x != p.rank)
    G130_SPIN(p.flags + slot * p.world + threadIdx.x, seq, threadIdx.x, 1);
  __syncthreads();
  unsigned long long t_flags = 0;
  if (p.timing && threadIdx.x == 0 && blockIdx.x == 0) t_flags = now_ns();

  const char* rbase = p.recv + (long long)slot * p.world * p.max_bytes;
  if constexpr (kMode == kGather) {
    for (long long i = tid; i < p.nvec; i += stride) {""")

rep("""    if (last_cta(&p.st->arrive2, &s_last)) {
      spin_until(&p.ctrl->posted2_seq, ld_acquire_sys(&p.ctrl->send2_seq), p, -2, 0);
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send2_bytes) = (uint64_t)len * 16;
      st_release_sys(&p.ctrl->send2_seq, seq);
    }
    if (threadIdx.x < p.world && threadIdx.x != p.rank)
      spin_until(p.flags2 + slot * p.world + threadIdx.x, seq, p, threadIdx.x, 2);
    __syncthreads();
    const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;
    // legacy: for r != rank""",
    """    if (last_cta(&p.st->arrive2, &s_last)) {
      // g130 V2: no guard is needed. This thread is past the phase-1 wait of seq, so all 7 peers published phase 1 of seq,
      // so each of them finished the previous two-shot op seq' < seq on its single stream, so each of them saw this
      // rank's phase-2 flag of seq': this rank's proxy has already read send2_bytes and posted seq' to all 7 peers. (seq - 1
      // would be WRONG here: send2_seq only moves on two-shot ops, modes interleave, and the spin would never end.)
      if constexpr ((kPub & kPubV2) == 0)
        G130_SPIN(&p.ctrl->posted2_seq, ld_acquire_sys(&p.ctrl->send2_seq), -2, 0);
      *reinterpret_cast<volatile uint64_t*>(&p.ctrl->send2_bytes) = (uint64_t)len * 16;
      st_release_sys(&p.ctrl->send2_seq, seq);
    }
    if (threadIdx.x < p.world && threadIdx.x != p.rank)
      G130_SPIN(p.flags2 + slot * p.world + threadIdx.x, seq, threadIdx.x, 2);
    __syncthreads();
    const char* r2 = p.recv2 + (long long)slot * p.world * p.max_chunk;
    // legacy: for r != rank""")

# ---- launch dispatch
rep("""// Launch either kernel with the live grid rule: ceil(nvec / vec_per_cta) CTAs, clamped to [1, 16].
template <int kMode, bool kF32>
void launch_kernel(const Params& p, long long vec_per_cta, bool par, cudaStream_t stream) {
  long long blocks = (p.nvec + vec_per_cta - 1) / vec_per_cta;
  blocks = std::max(1LL, std::min(blocks, 16LL));
  if (par && p.world == kParW)
    ar_kernel_pl<kMode, kF32, kParW><<<(int)blocks, kThreads, 0, stream>>>(p);
  else
    ar_kernel<kMode, kF32><<<(int)blocks, kThreads, 0, stream>>>(p);
  CHECK_CUDA(cudaGetLastError());
}
""", """// g130: V2 has no meaning without a phase 2, so non-two-shot modes drop the bit instead of instantiating a duplicate kernel.
template <int kMode, bool kF32, int kPub>
void launch_pl(const Params& p, int blocks, cudaStream_t stream) {
  ar_kernel_pl<kMode, kF32, kParW, (kMode == kTwoShot ? kPub : (kPub & ~kPubV2))><<<blocks, kThreads, 0, stream>>>(p);
}

// Launch either kernel with the live grid rule: ceil(nvec / vec_per_cta) CTAs, clamped to [1, 16].
template <int kMode, bool kF32>
void launch_kernel(const Params& p, long long vec_per_cta, bool par, cudaStream_t stream, int pub = 0) {
  long long blocks = (p.nvec + vec_per_cta - 1) / vec_per_cta;
  blocks = std::max(1LL, std::min(blocks, 16LL));
  if (par && p.world == kParW) {
    switch (pub & 7) {
      case 0: launch_pl<kMode, kF32, 0>(p, (int)blocks, stream); break;
      case 1: launch_pl<kMode, kF32, 1>(p, (int)blocks, stream); break;
      case 2: launch_pl<kMode, kF32, 2>(p, (int)blocks, stream); break;
      case 3: launch_pl<kMode, kF32, 3>(p, (int)blocks, stream); break;
      case 4: launch_pl<kMode, kF32, 4>(p, (int)blocks, stream); break;
      case 5: launch_pl<kMode, kF32, 5>(p, (int)blocks, stream); break;
      case 6: launch_pl<kMode, kF32, 6>(p, (int)blocks, stream); break;
      default: launch_pl<kMode, kF32, 7>(p, (int)blocks, stream); break;
    }
  } else {
    ar_kernel<kMode, kF32><<<(int)blocks, kThreads, 0, stream>>>(p);
  }
  CHECK_CUDA(cudaGetLastError());
}
""")

rep("""int env_parload_default() {
  const char* e = getenv("RDMA_AR_PARLOAD");
  return e ? (atoi(e) != 0) : 1;  // this .so exists to run the parallel-load kernels
}
""", """int env_parload_default() {
  const char* e = getenv("RDMA_AR_PARLOAD");
  return e ? (atoi(e) != 0) : 1;  // this .so exists to run the parallel-load kernels
}

// g130: unset or 0 == off
bool env_flag(const char* name) {
  const char* e = getenv(name);
  return e && atoi(e) != 0;
}
int env_pub_mask() {
  return (env_flag("RDMA_AR_PUB_V1") ? kPubV1 : 0) | (env_flag("RDMA_AR_PUB_V2") ? kPubV2 : 0) |
         (env_flag("RDMA_AR_HYG_POLL") ? kPollThin : 0);
}
""")

# ---- host launch
rep("""    if (nbytes == 0) return;
    Params p;
    p.in = reinterpret_cast<const uint4*>(x.data_ptr());
    p.out = reinterpret_cast<uint4*>(out.data_ptr());
    p.nvec = nbytes / 16;
    p.chunk = (p.nvec + world_ - 1) / world_;
    p.send = arena_dev_ + send_off_;""", """    if (nbytes == 0) return;
    // g130: kernel variants exist on the parallel-load path only; never let a flag be silently ignored
    TORCH_CHECK(pub_mask_ == 0 || parload(), "rdma_ar: RDMA_AR_PUB_V1/V2 and RDMA_AR_HYG_POLL need the parallel-load kernels");
    Params p;
    p.in = reinterpret_cast<const uint4*>(x.data_ptr());
    p.out = reinterpret_cast<uint4*>(out.data_ptr());
    p.nvec = nbytes / 16;
    p.chunk = (p.nvec + world_ - 1) / world_;
    p.send = arena_dev_ + send_off_;""")
rep("""    launch_kernel<kMode, kF32>(p, vec_per_cta_, parload_ != 0, at::cuda::getCurrentCUDAStream());""",
    """    launch_kernel<kMode, kF32>(p, vec_per_cta_, parload_ != 0, at::cuda::getCurrentCUDAStream(), pub_mask_);""")

# ---- A: split post
rep("""    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(pr.qp, len ? &wr[0] : &wr[1], &bad)) {
      fprintf(stderr, "rdma_ar rank %d: ibv_post_send failed errno=%d\\n", rank_, errno);
      proxy_error_.store(true);
    }
  }
""", """    ibv_send_wr* bad = nullptr;
    if (len && split_post_.load(std::memory_order_relaxed)) {
      // g130 A: the same two WRs on the same QP in the same order, as two nreq == 1 posts. If the payload post fails the
      // flag is NOT posted, as in the chain: a flag without its payload would be silent wrong data at the peer.
      wr[0].next = nullptr;
      if (ibv_post_send(pr.qp, &wr[0], &bad) || ibv_post_send(pr.qp, &wr[1], &bad)) {
        fprintf(stderr, "rdma_ar rank %d: ibv_post_send failed errno=%d\\n", rank_, errno);
        proxy_error_.store(true);
      }
    } else if (ibv_post_send(pr.qp, len ? &wr[0] : &wr[1], &bad)) {
      fprintf(stderr, "rdma_ar rank %d: ibv_post_send failed errno=%d\\n", rank_, errno);
      proxy_error_.store(true);
    }
  }
""")

# ---- C: proxy
rep("""    unsigned idle_iters = 0;
    uint64_t last_ping = 0;
    while (!stop_.load(std::memory_order_relaxed)) {
      bool idle = true;
      const uint64_t pg = __atomic_load_n(&ctrl_->ping, __ATOMIC_ACQUIRE);""",
    """    unsigned idle_iters = 0;
    uint64_t last_ping = 0;
    bool last_two = false;  // g130 hygiene: the newest phase-1 message was two-shot, so its phase 2 rings next
    const auto reap = [&]() {
      for (auto& h : hcas_) {
        int n = ibv_poll_cq(h.cq, 32, wc);
        for (int i = 0; i < n; ++i)
          if (wc[i].status != IBV_WC_SUCCESS) {
            fprintf(stderr, "rdma_ar rank %d: completion error %s (qp %u)\\n", rank_,
                    ibv_wc_status_str(wc[i].status), wc[i].qp_num);
            proxy_error_.store(true);
          }
      }
    };
    while (!stop_.load(std::memory_order_relaxed)) {
      // g130 C (gr8 fabric_order.h walk_step): spin on the one word that rings next. Only a hint: after kHygSpin loads the
      // full pass below runs anyway, so a wrong guess costs a few ns, never a hang, and ping, stop and backoff still work.
      if (hyg_proxy_.load(std::memory_order_relaxed)) {
        const bool e2 = last_two && last2 != last1;
        const uint64_t* db = e2 ? &ctrl_->send2_seq : &ctrl_->send_seq;
        const uint64_t old = e2 ? last2 : last1;
        for (int k = 0; k < kHygSpin && __atomic_load_n(db, __ATOMIC_ACQUIRE) == old; ++k) {
        }
      }
      bool idle = true;
      const uint64_t pg = __atomic_load_n(&ctrl_->ping, __ATOMIC_ACQUIRE);""")

rep("""          post(pr, src, (uint32_t)len, pr.remote_base + recv_off_ + ((uint64_t)slot * world_ + rank_) * max_bytes_,
               pr.remote_base + flags_off_ + ((uint64_t)slot * world_ + rank_) * 8, s1);
        }
        __atomic_store_n(&ctrl_->posted_seq, s1, __ATOMIC_RELEASE);
""", """          post(pr, src, (uint32_t)len, pr.remote_base + recv_off_ + ((uint64_t)slot * world_ + rank_) * max_bytes_,
               pr.remote_base + flags_off_ + ((uint64_t)slot * world_ + rank_) * 8, s1);
        }
        __atomic_store_n(&ctrl_->posted_seq, s1, __ATOMIC_RELEASE);
        last_two = mode == kTwoShot;
""")
rep("""        px_n_.fetch_add(1, std::memory_order_relaxed);
        px_handoff_.fetch_add(t_seen - t_pub, std::memory_order_relaxed);
        px_post_.fetch_add(t_done - t_seen, std::memory_order_relaxed);
        if (t_done - t_seen > px_post_max_.load(std::memory_order_relaxed))
          px_post_max_.store(t_done - t_seen, std::memory_order_relaxed);
        last1 = s1;
      }
""", """        px_n_.fetch_add(1, std::memory_order_relaxed);
        px_handoff_.fetch_add(t_seen - t_pub, std::memory_order_relaxed);
        px_post_.fetch_add(t_done - t_seen, std::memory_order_relaxed);
        if (t_done - t_seen > px_post_max_.load(std::memory_order_relaxed))
          px_post_max_.store(t_done - t_seen, std::memory_order_relaxed);
        last1 = s1;
        if (hyg_proxy_.load(std::memory_order_relaxed)) reap();
      }
""")
rep("""        __atomic_store_n(&ctrl_->posted2_seq, s2, __ATOMIC_RELEASE);
        last2 = s2;
      }
      if (idle) {
        for (auto& h : hcas_) {
          int n = ibv_poll_cq(h.cq, 32, wc);
          for (int i = 0; i < n; ++i)
            if (wc[i].status != IBV_WC_SUCCESS) {
              fprintf(stderr, "rdma_ar rank %d: completion error %s (qp %u)\\n", rank_,
                      ibv_wc_status_str(wc[i].status), wc[i].qp_num);
              proxy_error_.store(true);
            }
        }
        if ((++idle_iters & 255) == 0""", """        __atomic_store_n(&ctrl_->posted2_seq, s2, __ATOMIC_RELEASE);
        last2 = s2;
        if (hyg_proxy_.load(std::memory_order_relaxed)) reap();
      }
      if (idle) {
        // g130 C: with hygiene on, the post path reaps; this slow reap stays so that a completion error with no traffic
        // behind it is still reported (gr8 has no such path, its walk ends on a timeout instead).
        if (!hyg_proxy_.load(std::memory_order_relaxed) || (idle_iters & 1023) == 0) reap();
        if ((++idle_iters & 255) == 0""")

# ---- members + setters
rep("""  int parload_ = env_parload_default();

 public:
  // k1/3-rdma-loads: 1 = parallel-load kernels""", """  int parload_ = env_parload_default();
  // g130 flags: see the file header. Off when unset. pub_mask_ acts at the next launch, the other two at the next message.
  static constexpr int kHygSpin = 64;
  int pub_mask_ = env_pub_mask();
  std::atomic<bool> split_post_{env_flag("RDMA_AR_SPLIT_POST")};
  std::atomic<bool> hyg_proxy_{env_flag("RDMA_AR_HYG_PROXY")};

 public:
  // g130: [split_post, hyg_proxy, pub_mask]. set_variant is for benches: call it with the stream drained and the same
  // values on every rank's quiet point. A CUDA graph keeps the kernel variant it was captured with.
  std::vector<int64_t> variant() const {
    return {split_post_.load() ? 1 : 0, hyg_proxy_.load() ? 1 : 0, pub_mask_};
  }
  void set_variant(bool split_post, bool hyg_proxy, int pub_mask) {
    TORCH_CHECK(pub_mask >= 0 && pub_mask <= 7, "rdma_ar: bad pub_mask ", pub_mask);
    split_post_.store(split_post);
    hyg_proxy_.store(hyg_proxy);
    pub_mask_ = pub_mask;
  }
  // k1/3-rdma-loads: 1 = parallel-load kernels""")

# ---- SelfTest
rep("""  int64_t max_chunk() const { return (int64_t)max_chunk_; }
  int64_t max_bytes() const { return max_bytes_; }
  int64_t max_reduce_bytes() const { return max_reduce_bytes_; }
  bool thp() const { return !host_alloc_; }
""", """  int64_t max_chunk() const { return (int64_t)max_chunk_; }
  int64_t max_bytes() const { return max_bytes_; }
  int64_t max_reduce_bytes() const { return max_reduce_bytes_; }
  bool thp() const { return !host_alloc_; }
  // g130: kernel variant for the following calls (par = true only); default 0 == production kernels
  void set_pub(int pub_mask) {
    TORCH_CHECK(pub_mask >= 0 && pub_mask <= 7, "selftest: bad pub_mask ", pub_mask);
    pub_ = pub_mask;
  }
""")
rep("""    launch_kernel<kMode, kF32>(p, std::max<long long>(64, vec_per_cta), par, at::cuda::getCurrentCUDAStream());""",
    """    TORCH_CHECK(pub_ == 0 || (par && world_ == kParW), "selftest: kernel variants need par = true and world 8");
    launch_kernel<kMode, kF32>(p, std::max<long long>(64, vec_per_cta), par, at::cuda::getCurrentCUDAStream(), pub_);""")
rep("""  int rank_, world_;
  int64_t max_bytes_, max_reduce_bytes_ = 0;
  int device_;
  size_t max_chunk_ = 0;
  void* arena_ = nullptr;
  void* raw_ = nullptr;""", """  int rank_, world_;
  int pub_ = 0;  // g130
  int64_t max_bytes_, max_reduce_bytes_ = 0;
  int device_;
  size_t max_chunk_ = 0;
  void* arena_ = nullptr;
  void* raw_ = nullptr;""")

# ---- bindings
rep("""      .def("set_parload", &RdmaAR::set_parload, py::arg("on"))
      .def("parload", &RdmaAR::parload);""", """      .def("set_parload", &RdmaAR::set_parload, py::arg("on"))
      .def("parload", &RdmaAR::parload)
      .def("variant", &RdmaAR::variant)
      .def("set_variant", &RdmaAR::set_variant, py::arg("split_post"), py::arg("hyg_proxy"), py::arg("pub_mask"));""")
rep("""      .def("max_reduce_bytes", &SelfTest::max_reduce_bytes)
      .def("thp", &SelfTest::thp);""", """      .def("max_reduce_bytes", &SelfTest::max_reduce_bytes)
      .def("thp", &SelfTest::thp)
      .def("set_pub", &SelfTest::set_pub, py::arg("pub_mask"));""")
rep("""  m.attr("G2_VARIANT") = "colltrim-cols-v1";  // g2 colltrim: all_gather_cols / all_gather_cols_into present
""", """  m.attr("G2_VARIANT") = "colltrim-cols-v1";  // g2 colltrim: all_gather_cols / all_gather_cols_into present
  m.attr("G130_VARIANT") = "rdma-flags-v1";   // g130: RdmaAR.variant / set_variant, SelfTest.set_pub
""")

open(out, "w").write(s)
print(f"mk_variant.py: {out} sha256 {hashlib.sha256(s.encode()).hexdigest()[:16]}")
