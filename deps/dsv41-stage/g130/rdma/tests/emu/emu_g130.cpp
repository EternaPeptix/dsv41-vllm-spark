// emu_g130.cpp: 8-rank CPU emulator of rdma_ar with FREE-RUNNING ranks and forced proxy stalls.
//
// Runs VERBATIM source (extract_g130.py): ar_kernel_pl<mode,f32,8,kPub>, RdmaAR::post and RdmaAR::proxy_loop. Verbs are fake:
// each QP is a FIFO drained by one "NIC" thread per rank, which reads the payload at execution time (not post time), so slot
// reuse before the NIC read is visible. Inline flags are captured at post time, as on the real device.
//
// Why not w7's emu_main.cpp launch(): it joins all 8 ranks between ops, so a rank can never run ahead of its own stalled
// proxy, which is exactly the hazard the removed spins guarded. Here each rank has its own driver and op list.
//
// Result checks are independent: acc = +0.0f, FADD in r = 0..7, one RN to bf16, computed from the generator.
// CANNOT catch (see WINDOW.md): GPU weak ordering at sys scope, NIC/BlueFlame/WC behaviour, C2C coherence, timing.
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>

#include <atomic>
#include <cerrno>
#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "emu_prims.h"

thread_local EmuCtx emu_ctx;
EmuBarrier* emu_bar = nullptr;
int emu_wmax[32];

namespace emu {
#include "emu_dev.inc"
}
using namespace emu;

// ---------------------------------------------------------------------------------------------- fake verbs -----
enum { IBV_WR_RDMA_WRITE = 0 };
enum { IBV_SEND_INLINE = 1, IBV_SEND_SIGNALED = 2 };
enum { IBV_WC_SUCCESS = 0 };
enum ibv_mtu { IBV_MTU_1024 = 3 };
struct ibv_context {};
struct ibv_pd {};
struct ibv_cq {};
struct ibv_mr {
  uint32_t lkey = 7;
};
struct ibv_gid {
  unsigned char raw[16];
};
struct ibv_sge {
  uint64_t addr;
  uint32_t length, lkey;
};
struct ibv_send_wr {
  uint64_t wr_id;
  ibv_send_wr* next;
  ibv_sge* sg_list;
  int num_sge;
  int opcode;
  unsigned send_flags;
  union {
    struct {
      uint64_t remote_addr;
      uint32_t rkey;
    } rdma;
  } wr;
};
struct ibv_wc {
  int status;
  unsigned qp_num;
};
struct QWr {
  bool inl;
  uint64_t src;
  uint32_t len;
  uint64_t dst;
  uint64_t val;
  unsigned flags;
};
struct ibv_qp {
  std::mutex m;
  std::deque<QWr> q;
  long calls = 0;       // ibv_post_send calls
  long nreq1 = 0;       // ... with one WR (BlueFlame-eligible on the real device)
  std::vector<QWr> log;  // unit test only
  bool keep_log = false;
};

struct StallCfg {
  int victim = -1;                      // -1 = every proxy
  unsigned load_n = 0, load_us = 0;     // at each proxy atomic load: 1/load_n odds, sleep up to load_us
  unsigned post_n = 0, post_us = 0;     // at each ibv_post_send
  unsigned nic_n = 0, nic_us = 0;       // before the NIC executes a WR
  unsigned jitter_us = 0;               // before each op, each rank but the victim
};
static StallCfg g_stall;
static thread_local int t_px_rank = -1;
static thread_local unsigned long long t_rng = 88172645463325252ull;
static inline unsigned rnd32() {
  t_rng ^= t_rng << 13;
  t_rng ^= t_rng >> 7;
  t_rng ^= t_rng << 17;
  return (unsigned)(t_rng >> 16);
}
static inline bool is_victim() { return g_stall.victim < 0 || g_stall.victim == t_px_rank; }
static inline void maybe_stall(unsigned n, unsigned us) {
  if (n && us && is_victim() && rnd32() % n == 0) usleep(1 + rnd32() % us);
}
static thread_local unsigned t_tick = 0;
// proxy-side atomic load: forced stall + a throttle so 8 spinning proxies stay gentle on the host
template <class T>
inline T emu_hload(const T* p) {
  maybe_stall(g_stall.load_n, g_stall.load_us);
  if ((++t_tick & 255u) == 0) usleep(20);
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

struct Counters {
  std::atomic<long> polls[8], msgs[8], ahead1[8], ahead2[8], bad[8], ops[8];
};
static Counters* g_ctr = nullptr;
static long g_post_fail_after = -1;  // unit test: fail the Nth WR-carrying call (ENOMEM, like a full SQ)

static inline int ibv_post_send(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad) {
  maybe_stall(g_stall.post_n, g_stall.post_us);
  if (g_post_fail_after == 0) {
    g_post_fail_after = -1;
    *bad = wr;
    errno = ENOMEM;
    return ENOMEM;
  }
  if (g_post_fail_after > 0) --g_post_fail_after;
  int n = 0;
  std::lock_guard<std::mutex> l(qp->m);
  for (ibv_send_wr* w = wr; w; w = w->next, ++n) {
    QWr q{};
    q.inl = (w->send_flags & IBV_SEND_INLINE) != 0;
    q.src = w->sg_list[0].addr;
    q.len = w->sg_list[0].length;
    q.dst = w->wr.rdma.remote_addr;
    q.flags = w->send_flags;
    if (q.inl) memcpy(&q.val, reinterpret_cast<const void*>(q.src), q.len);
    qp->q.push_back(q);
    if (qp->keep_log) qp->log.push_back(q);
  }
  ++qp->calls;
  if (n == 1) ++qp->nreq1;
  return 0;
}
static inline int ibv_poll_cq(ibv_cq*, int, ibv_wc*) {
  if (g_ctr && t_px_rank >= 0) g_ctr->polls[t_px_rank].fetch_add(1, std::memory_order_relaxed);
  return 0;
}
static inline const char* ibv_wc_status_str(int) { return "emu"; }

// verbatim host code: post + proxy_loop, as members of a small struct with the same member names as RdmaAR
#include "emu_hoststructs.inc"
#define __atomic_load_n(p, m) emu_hload(p)
struct EmuRdma {
  int rank_ = 0, world_ = 8;
  long long max_bytes_ = 0;
  int proxy_core_ = -1;
  size_t max_chunk_ = 0;
  void* arena_ = nullptr;
  size_t flags_off_ = 0, flags2_off_ = 0, send_off_ = 0, send2_off_ = 0, recv_off_ = 0, recv2_off_ = 0;
  Ctrl* ctrl_ = nullptr;
  std::vector<Hca> hcas_;
  std::vector<Peer> peers_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> proxy_error_{false};
  std::atomic<int64_t> px_n_{0}, px_handoff_{0}, px_post_{0}, px_post_max_{0};
  static constexpr int kHygSpin = 64;
  std::atomic<bool> split_post_{false};
  std::atomic<bool> hyg_proxy_{false};
#include "emu_host.inc"
};
#undef __atomic_load_n

// ---------------------------------------------------------------------------------------------- shared state -----
static constexpr int kWorld = 8;
static constexpr long long kMaxBytes = 4096;
static constexpr long long kBufVec = 2048;
static size_t al(size_t x, size_t a) { return (x + a - 1) / a * a; }
struct Rank {
  char* arena;
  DevState* st;
  uint4* in;
  uint4* out;
};
static Rank g_r[kWorld];
static char* g_mem = nullptr;
static size_t g_bytes = 0;
static volatile int* g_stop = nullptr;
static long long g_max_chunk;
static size_t g_flags_off, g_flags2_off, g_send_off, g_send2_off, g_recv_off, g_recv2_off, g_arena;
static pid_t g_proxy = -1;

struct Case {
  const char* name;
  int pub;
  bool split, hyg;
  StallCfg stall;
};
struct Op {
  int mode;
  bool f32;
  long long nvec, seg;
};
static std::vector<Op> g_ops;
static Case g_case;

static void proxy_main() {
  signal(SIGTERM, SIG_DFL);
  std::vector<std::thread> th;
  static ibv_cq cqs[kWorld][2];
  static ibv_mr mrs[kWorld][2];
  static std::vector<ibv_qp*> qps[kWorld];
  for (int R = 0; R < kWorld; ++R)
    for (int q = 0; q < kWorld; ++q) qps[R].push_back(new ibv_qp());
  static EmuRdma* obj[kWorld];
  for (int R = 0; R < kWorld; ++R) {
    EmuRdma* e = new EmuRdma();
    obj[R] = e;
    e->rank_ = R;
    e->max_bytes_ = kMaxBytes;
    e->max_chunk_ = (size_t)g_max_chunk;
    e->arena_ = g_r[R].arena;
    e->flags_off_ = g_flags_off;
    e->flags2_off_ = g_flags2_off;
    e->send_off_ = g_send_off;
    e->send2_off_ = g_send2_off;
    e->recv_off_ = g_recv_off;
    e->recv2_off_ = g_recv2_off;
    e->ctrl_ = reinterpret_cast<Ctrl*>(g_r[R].arena);
    e->hcas_.resize(2);
    for (int h = 0; h < 2; ++h) {
      e->hcas_[h].cq = &cqs[R][h];
      e->hcas_[h].mr = &mrs[R][h];
    }
    e->peers_.resize(kWorld);
    for (int q = 0; q < kWorld; ++q) {
      e->peers_[q].hca = (R + q) % 2;
      e->peers_[q].qp = qps[R][q];
      e->peers_[q].remote_base = (uint64_t)(uintptr_t)g_r[q].arena;
    }
    e->split_post_.store(g_case.split);
    e->hyg_proxy_.store(g_case.hyg);
    th.emplace_back([e, R] {
      t_px_rank = R;
      t_rng ^= (unsigned long long)(R + 1) * 0x9E3779B97F4A7C15ull;
      e->proxy_loop();
    });
    th.emplace_back([R] {  // the NIC of rank R: per QP in order, QPs interleaved
      t_px_rank = R;
      t_rng ^= (unsigned long long)(R + 11) * 0xC2B2AE3D27D4EB4Full;
      unsigned idle = 0;
      while (!*g_stop) {
        bool any = false;
        for (int q = 0; q < kWorld; ++q) {
          ibv_qp* qp = qps[R][q];
          QWr w;
          {
            std::lock_guard<std::mutex> l(qp->m);
            if (qp->q.empty()) continue;
            w = qp->q.front();
            qp->q.pop_front();
          }
          any = true;
          maybe_stall(g_stall.nic_n, g_stall.nic_us);
          if (w.inl) {
            __atomic_store_n(reinterpret_cast<uint64_t*>(w.dst), w.val, __ATOMIC_RELEASE);
          } else {
            memcpy(reinterpret_cast<void*>(w.dst), reinterpret_cast<const void*>(w.src), w.len);
            __atomic_thread_fence(__ATOMIC_RELEASE);
          }
        }
        if (!any && (++idle & 63u) == 0) usleep(20);
      }
    });
  }
  std::thread watcher([] {
    while (!*g_stop) usleep(1000);
    for (int R = 0; R < kWorld; ++R) obj[R]->stop_.store(true);
  });
  for (auto& t : th) t.join();
  watcher.join();
  bool err = false;
  for (int R = 0; R < kWorld; ++R) err |= obj[R]->proxy_error_.load();
  _exit(err ? 9 : 0);
}

static void setup() {
  g_max_chunk = (long long)al((size_t)((kMaxBytes / 16 + kWorld - 1) / kWorld) * 16, 64);
  const size_t fl = al(sizeof(uint64_t) * kSlots * kWorld, 4096);
  g_flags_off = 4096;
  g_flags2_off = g_flags_off + fl;
  g_send_off = g_flags2_off + fl;
  g_send2_off = g_send_off + al(kSlots * kMaxBytes, 4096);
  g_recv_off = g_send2_off + al(kSlots * g_max_chunk, 4096);
  g_recv2_off = g_recv_off + al((size_t)kSlots * kWorld * kMaxBytes, 4096);
  g_arena = g_recv2_off + al((size_t)kSlots * kWorld * g_max_chunk, 4096);
  const size_t dev = al(sizeof(DevState), 64) + al(16 * kBufVec, 64) * 2;
  g_bytes = 4096 + al(sizeof(Counters), 4096) + kWorld * (g_arena + dev);
  g_mem = (char*)mmap(nullptr, g_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
  if (g_mem == MAP_FAILED) {
    perror("mmap");
    exit(2);
  }
  g_stop = reinterpret_cast<volatile int*>(g_mem);
  g_ctr = reinterpret_cast<Counters*>(g_mem + 4096);
  char* p = g_mem + 4096 + al(sizeof(Counters), 4096);
  for (int r = 0; r < kWorld; ++r) {
    g_r[r].arena = p;
    p += g_arena;
    g_r[r].st = reinterpret_cast<DevState*>(p);
    p += al(sizeof(DevState), 64);
    g_r[r].in = reinterpret_cast<uint4*>(p);
    p += al(16 * kBufVec, 64);
    g_r[r].out = reinterpret_cast<uint4*>(p);
    p += al(16 * kBufVec, 64);
  }
}

// ---------------------------------------------------------------------------------------------- inputs + reference -----
static inline unsigned long long mix(unsigned long long x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  return x ^ (x >> 33);
}
// wide exponents: the fp32 sum depends on the order, so a reordered or missing contribution changes the bits
static inline unsigned in_word(int op, int rank, long long i, int k, bool f32) {
  unsigned long long h = mix(((unsigned long long)(op + 1) << 40) ^ ((unsigned long long)(rank + 1) << 32) ^ ((unsigned long long)i << 8) ^ (unsigned)k);
  float f = (float)((long long)(h & 0xffff) - 32768) / 32768.f;
  f = std::ldexp(f, (int)((h >> 16) % 33) - 16);
  if (f32) return __float_as_uint(f);
  return (unsigned)emu_f2bf(f);
}
static void fill_in(int r, int opi, const Op& o) {
  if (o.f32) {
    unsigned* w = reinterpret_cast<unsigned*>(g_r[r].in);
    for (long long i = 0; i < o.nvec * 4; ++i) w[i] = in_word(opi, r, i, 0, true);
  } else {
    unsigned short* h = reinterpret_cast<unsigned short*>(g_r[r].in);
    for (long long i = 0; i < o.nvec * 8; ++i) h[i] = (unsigned short)in_word(opi, r, i, 0, false);
  }
}
static long long verify(int r, int opi, const Op& o) {
  long long bad = 0;
  if (o.mode == kGather || o.mode == kGatherCols) {
    const unsigned short* out = reinterpret_cast<const unsigned short*>(g_r[r].out);
    for (int q = 0; q < kWorld; ++q)
      for (long long i = 0; i < o.nvec; ++i) {
        long long oi;
        if (o.mode == kGather) {
          oi = q * o.nvec + i;
        } else {
          const long long row = i / o.seg;
          oi = row * kWorld * o.seg + q * o.seg + (i - row * o.seg);
        }
        for (int k = 0; k < 8; ++k)
          if (out[oi * 8 + k] != (unsigned short)in_word(opi, q, i * 8 + k, 0, false)) ++bad;
      }
    return bad;
  }
  if (o.f32) {
    const unsigned* out = reinterpret_cast<const unsigned*>(g_r[r].out);
    for (long long i = 0; i < o.nvec * 4; ++i) {
      float acc = 0.f;
      for (int q = 0; q < kWorld; ++q) acc += __uint_as_float(in_word(opi, q, i, 0, true));
      if (out[i] != __float_as_uint(acc)) ++bad;
    }
    return bad;
  }
  const unsigned short* out = reinterpret_cast<const unsigned short*>(g_r[r].out);
  for (long long i = 0; i < o.nvec * 8; ++i) {
    float acc = 0.f;
    for (int q = 0; q < kWorld; ++q) acc += emu_bf2f((unsigned short)in_word(opi, q, i, 0, false));
    if (out[i] != emu_f2bf(acc)) ++bad;
  }
  return bad;
}

// ---------------------------------------------------------------------------------------------- kernels -----
using KFn = void (*)(Params);
template <int M, bool F, int P>
static void kcall(Params p) {
  ar_kernel_pl<M, F, kWorld, (M == kTwoShot ? P : (P & ~kPubV2))>(p);
}
template <int M, bool F>
static KFn pick(int pub) {
  switch (pub & 7) {
    case 0: return kcall<M, F, 0>;
    case 1: return kcall<M, F, 1>;
    case 2: return kcall<M, F, 2>;
    case 3: return kcall<M, F, 3>;
    case 4: return kcall<M, F, 4>;
    case 5: return kcall<M, F, 5>;
    case 6: return kcall<M, F, 6>;
    default: return kcall<M, F, 7>;
  }
}
static KFn kernel_of(const Op& o, int pub) {
  if (o.mode == kOneShot) return o.f32 ? pick<kOneShot, true>(pub) : pick<kOneShot, false>(pub);
  if (o.mode == kTwoShot) return o.f32 ? pick<kTwoShot, true>(pub) : pick<kTwoShot, false>(pub);
  if (o.mode == kGather) return pick<kGather, false>(pub);
  return pick<kGatherCols, false>(pub);
}

static Params params(int r, const Op& o) {
  Params p;
  p.in = g_r[r].in;
  p.out = g_r[r].out;
  p.nvec = o.nvec;
  p.chunk = (o.nvec + kWorld - 1) / kWorld;
  p.send = g_r[r].arena + g_send_off;
  p.send2 = g_r[r].arena + g_send2_off;
  p.recv = g_r[r].arena + g_recv_off;
  p.recv2 = g_r[r].arena + g_recv2_off;
  p.flags = reinterpret_cast<const uint64_t*>(g_r[r].arena + g_flags_off);
  p.flags2 = reinterpret_cast<const uint64_t*>(g_r[r].arena + g_flags2_off);
  p.ctrl = reinterpret_cast<Ctrl*>(g_r[r].arena);
  p.st = g_r[r].st;
  p.max_bytes = kMaxBytes;
  p.max_chunk = g_max_chunk;
  p.rank = r;
  p.world = kWorld;
  p.timeout_ns = 4000000000LL;  // a lost publish shows as a timeout trap (exit 43), not a hang
  p.timing = 0;
  p.seg = o.seg;
  return p;
}

// one rank's whole op list, with no join against the other ranks
static void rank_main(int r) {
  setpgid(0, 0);
  t_rng ^= (unsigned long long)(r + 101) * 0x9E3779B97F4A7C15ull;
  Ctrl* c = reinterpret_cast<Ctrl*>(g_r[r].arena);
  for (size_t opi = 0; opi < g_ops.size(); ++opi) {
    const Op& o = g_ops[opi];
    if (g_case.stall.jitter_us && r != g_case.stall.victim) usleep(rnd32() % g_case.stall.jitter_us);
    fill_in(r, (int)opi, o);
    const int grid = 1 + (int)(mix(opi * 31 + 7) % 2);
    const int block = 8;
    KFn fn = kernel_of(o, g_case.pub);
    Params p = params(r, o);
    std::vector<pid_t> kids;
    for (int b = 0; b < grid; ++b) {
      const pid_t pid = fork();
      if (pid == 0) {
        EmuBarrier bar;
        bar.n = (unsigned)block;
        emu_bar = &bar;
        std::vector<std::thread> th;
        for (int t = 0; t < block; ++t)
          th.emplace_back([&, t] {
            emu_ctx.tid = {(unsigned)t, 0, 0};
            emu_ctx.bid = {(unsigned)b, 0, 0};
            emu_ctx.bdim = {(unsigned)block, 1, 1};
            emu_ctx.gdim = {(unsigned)grid, 1, 1};
            fn(p);
          });
        for (auto& t : th) t.join();
        fflush(stdout);
        _exit(0);
      }
      kids.push_back(pid);
    }
    for (pid_t k : kids) {
      int st = 0;
      waitpid(k, &st, 0);
      if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        fflush(stdout);
        _exit(WIFEXITED(st) ? WEXITSTATUS(st) : 128);
      }
    }
    // coverage: this rank's kernel ended while its own proxy had not acknowledged the publish
    if (__atomic_load_n(&c->posted_seq, __ATOMIC_ACQUIRE) < __atomic_load_n(&c->send_seq, __ATOMIC_ACQUIRE))
      g_ctr->ahead1[r].fetch_add(1);
    if (__atomic_load_n(&c->posted2_seq, __ATOMIC_ACQUIRE) < __atomic_load_n(&c->send2_seq, __ATOMIC_ACQUIRE))
      g_ctr->ahead2[r].fetch_add(1);
    const long long bad = verify(r, (int)opi, o);
    if (bad) {
      g_ctr->bad[r].fetch_add(bad);
      printf("  rank %d op %zu mode %d f32 %d nvec %lld: %lld wrong values\n", r, opi, o.mode, (int)o.f32, o.nvec, bad);
      fflush(stdout);
      _exit(44);
    }
    g_ctr->ops[r].fetch_add(1);
  }
  _exit(0);
}

static std::vector<Op> make_ops(unsigned seed, int n) {
  std::mt19937 g(seed);
  std::vector<Op> v;
  const long long sizes[] = {4, 12, 16, 40, 64, 100, 128, 200, 256};
  for (int i = 0; i < n; ++i) {
    Op o{};
    switch (g() % 7) {  // two-shot heavy, modes interleaved, and runs of consecutive two-shot
      case 0: case 1: case 2: o.mode = kTwoShot; break;
      case 3: o.mode = kOneShot; break;
      case 4: o.mode = kGather; break;
      case 5: o.mode = kGatherCols; break;
      default: o.mode = (i && v.back().mode == kTwoShot) ? kTwoShot : kOneShot; break;
    }
    o.f32 = (o.mode == kOneShot || o.mode == kTwoShot) && g() % 4 == 0;
    o.nvec = sizes[g() % 9];
    if (o.mode == kGatherCols) {
      o.seg = 4;
      o.nvec -= o.nvec % 4;
      if (o.nvec > 128) o.nvec = 128;
    }
    if (o.mode == kGather && o.nvec > 128) o.nvec = 128;
    v.push_back(o);
  }
  return v;
}

// returns "" on pass
static std::string run_case(const Case& c, unsigned seed, int nops, int wall_s) {
  memset(g_mem, 0, g_bytes);
  g_case = c;
  g_stall = c.stall;
  g_ops = make_ops(seed, nops);
  fflush(stdout);
  g_proxy = fork();
  if (g_proxy == 0) proxy_main();
  std::map<pid_t, int> ranks;
  for (int r = 0; r < kWorld; ++r) {
    const pid_t pid = fork();
    if (pid == 0) rank_main(r);
    ranks[pid] = r;
  }
  std::string res;
  const auto t0 = std::chrono::steady_clock::now();
  while (!ranks.empty()) {
    int st = 0;
    const pid_t pid = waitpid(-1, &st, WNOHANG);
    if (pid > 0 && ranks.count(pid)) {
      const int r = ranks[pid];
      ranks.erase(pid);
      const int rc = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
      if (rc != 0 && res.empty()) res = "rank " + std::to_string(r) + (rc == 43 ? " timeout (lost publish or flag)" : rc == 44 ? " wrong data" : " exit " + std::to_string(rc));
      continue;
    }
    if (!res.empty() || std::chrono::steady_clock::now() - t0 > std::chrono::seconds(wall_s)) {
      if (res.empty()) res = "hung past the wall limit";
      for (auto& kv : ranks) {
        kill(-kv.first, SIGKILL);
        kill(kv.first, SIGKILL);
      }
      for (auto& kv : ranks) waitpid(kv.first, &st, 0);
      ranks.clear();
      break;
    }
    usleep(2000);
  }
  *g_stop = 1;
  int st = 0;
  waitpid(g_proxy, &st, 0);
  if (res.empty() && WIFEXITED(st) && WEXITSTATUS(st) == 9) res = "proxy reported an error";
  while (waitpid(-1, &st, WNOHANG) > 0) {
  }
  return res;
}

// ---------------------------------------------------------------------------------------------- post() unit test -----
static int unit_post() {
  int fails = 0;
  static char buf[4096];
  auto drive = [&](bool split, long fail_after, long iters, uint32_t len) {
    auto e = std::make_unique<EmuRdma>();
    e->hcas_.resize(1);
    static ibv_mr mr;
    e->hcas_[0].mr = &mr;
    e->peers_.resize(1);
    ibv_qp* qp = new ibv_qp();
    qp->keep_log = true;
    e->peers_[0].hca = 0;
    e->peers_[0].qp = qp;
    e->split_post_.store(split);
    g_post_fail_after = fail_after;
    for (long i = 0; i < iters; ++i)
      e->post(e->peers_[0], (uint64_t)(uintptr_t)buf, len, 0x1000 + (uint64_t)i * 64, 0x9000 + (uint64_t)i * 8, 100 + i);
    g_post_fail_after = -1;
    return std::make_pair(qp, e->proxy_error_.load());
  };
  auto same = [](const QWr& a, const QWr& b) {
    return a.inl == b.inl && a.src == b.src && a.len == b.len && a.dst == b.dst && a.flags == b.flags && (!a.inl || a.val == b.val);
  };
  for (uint32_t len : {4096u, 448u, 0u}) {
    auto [a, ea] = drive(false, -1, 70, len);
    auto [b, eb] = drive(true, -1, 70, len);
    bool eq = a->log.size() == b->log.size() && !ea && !eb;
    for (size_t i = 0; eq && i < a->log.size(); ++i) eq = same(a->log[i], b->log[i]);
    const long want_calls = len ? 140 : 70;
    const bool shape = a->calls == 70 && (len ? a->nreq1 == 0 : a->nreq1 == 70) && b->calls == want_calls && b->nreq1 == want_calls;
    printf("  post len %4u: chain %ld calls (nreq=1: %ld), split %ld calls (nreq=1: %ld), %zu WRs, WR lists identical: %s\n", len, a->calls,
           a->nreq1, b->calls, b->nreq1, b->log.size(), eq ? "yes" : "NO");
    if (!eq || !shape) ++fails;
  }
  {  // a failed payload post must not post the flag
    auto [q, err] = drive(true, 0, 1, 4096);
    const bool ok = err && q->log.empty();
    printf("  split, payload post fails: proxy_error %d, WRs queued %zu (want 1, 0): %s\n", (int)err, q->log.size(), ok ? "ok" : "BAD");
    if (!ok) ++fails;
  }
  return fails;
}

// ---------------------------------------------------------------------------------------------- main -----
int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const char* only = argc > 1 ? argv[1] : nullptr;  // mutants: one case name, must FAIL
  const int nops = getenv("EMU_OPS") ? atoi(getenv("EMU_OPS")) : 40;
  setup();
  int fails = 0;
  if (!only) fails += unit_post();

  StallCfg none;
  StallCfg all;
  all.load_n = 4000; all.load_us = 400; all.post_n = 12; all.post_us = 1500; all.nic_n = 40; all.nic_us = 300; all.jitter_us = 200;
  StallCfg vic;  // one proxy very late, and its own GPU side never jittered, so it runs ahead of its proxy
  vic.victim = 3; vic.load_n = 1500; vic.load_us = 3000; vic.post_n = 4; vic.post_us = 4000; vic.jitter_us = 400;
  struct V {
    const char* name;
    int pub;
    bool split, hyg;
  };
  const V vs[] = {{"control", 0, false, false}, {"split", 0, true, false},    {"v1", 1, false, false},
                  {"v2", 2, false, false},      {"v1v2", 3, false, false},    {"hygproxy", 0, false, true},
                  {"hygpoll", 4, false, false}, {"all", 7, true, true}};
  const struct R {
    const char* name;
    const StallCfg* s;
  } rs[] = {{"nostall", &none}, {"stallall", &all}, {"victim", &vic}};
  long tot_a1 = 0, tot_a2 = 0;
  for (const V& v : vs)
    for (const R& r : rs) {
      char nm[64];
      snprintf(nm, sizeof nm, "%s/%s", v.name, r.name);
      Case c{nm, v.pub, v.split, v.hyg, *r.s};
      if (only && strcmp(only, nm) != 0) continue;
      const unsigned seed = 20261102u + (unsigned)(&v - vs) * 13u + (unsigned)(&r - rs) * 101u;
      const auto t0 = std::chrono::steady_clock::now();
      const std::string res = run_case(c, seed, nops, 45);
      long a1 = 0, a2 = 0, ops = 0, polls = 0;
      for (int i = 0; i < kWorld; ++i) {
        a1 += g_ctr->ahead1[i].load();
        a2 += g_ctr->ahead2[i].load();
        ops += g_ctr->ops[i].load();
        polls += g_ctr->polls[i].load();
      }
      tot_a1 += a1;
      tot_a2 += a2;
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      printf("%-20s %s  ops %ld/%d  rank ended ahead of own proxy: phase1 %ld phase2 %ld  cq polls %ld  %.1fs\n", nm, res.empty() ? "PASS" : ("FAIL: " + res).c_str(),
             ops, nops * kWorld, a1, a2, polls, s);
      if (!res.empty()) ++fails;
    }
  if (!only) printf("coverage: ahead-of-own-proxy events phase1 %ld, phase2 %ld (0 would mean the hazard window was never exercised)\n", tot_a1, tot_a2);
  printf("EMU %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
