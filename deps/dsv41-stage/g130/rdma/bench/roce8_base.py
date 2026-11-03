#!/usr/bin/env python3
"""bench_roce8.py (g3 rocenante): RoCEnante (flashinfer 0.7.2 experimental b12x comm/roce) vs our rdma_ar
(live k1 parallel-load .so) on the 8-Spark fabric. STOPPED WINDOW ONLY: one process per Spark on GPU 0,
started by ops/run-bench8.sh in the separate venv ~/venvs/roce072 (flashinfer 0.7.2 + nvidia-cutlass-dsl
4.7.1; torch 2.13.0+cu130 comes from vllm032 through a .pth, so the live rdma_ar_ext.so loads unchanged).
Nothing is installed into vllm032 and no serving file is read or written.

Phases (--phases, default X,L):
  X  exactness. Every rank builds all 8 ranks' inputs from a seed (so the reference is local), runs each
     arm eagerly and from a CUDA graph, and compares bitwise against
       ref_zero : acc = +0.0f; for r in 0..7: acc += float(x_r); out = cvt.rn(acc)   (our rdma_ar rule)
       ref_r0   : acc = float(x_0);  for r in 1..7: acc += float(x_r)                  (RoCEnante's init)
     Inputs mix wide-range normals with planted classes: all-(-0.0), mixed +-0, subnormals (FTZ probe),
     exact cancellation, order-dependent overflow, inf/NaN, absorption. Mismatches are classified
     (all_neg0 / nan_bits / other). A CPU numpy reference cross-checks the GPU reference, and a
     reversed-order reference proves the test can see an order change.
     Gathers (dim -1, the qkv_a shape [M,224]) are compared with torch.cat of all ranks' inputs.
  L  latency: graphs of --n-ops collectives, ABBA-rotated arms x --trials, modes
       b2b   back to back (pure transport chain)
       gap   --gap-us GPU spin before every op (decode-like spacing; proxies hot, links idle)
       l2pf  gap + the live L2 prefetch (6 MB cp.async.bulk.prefetch.L2 burst, 8 CTAs) forked on a side
             stream at every op, joined at graph end (what serving does around each collective)
       dram  gap + a continuous side-stream DRAM reader (the same prefetch kernel looping over 512 MB)
     kinds: ar (rows x 5120 bf16 all-reduce), ag (rows x 224 bf16 gather along dim -1), step (per layer
     AR, AG, AR: the decode collective pattern). us/op = (T(graph) - T(same graph without ops)) / n.

  C  crossover: rdma1 (one-shot) vs rdma2 (two-shot) (+ roce2) all-reduce at --xover-rows in the gap and
     l2pf modes; prints the RDMA_AR_TWOSHOT_MIN_BYTES that the data supports (env-only, exact: both modes
     use the same fixed-order fp32 rule; live 32768 = two-shot from 4 rows).

Arms (--arms):
  rdma     live policy: one-shot < 32 KiB, else two-shot (RDMA_AR_TWOSHOT_MIN_BYTES default); gather =
           stacked kernel + movedim/reshape copy (live cuda_communicator path); proxy pinned (see --pin)
  rdma1    same instance, one-shot at every size          rdma2  same instance, two-shot at every size
  rdmaU    same instance as rdma with its proxy unpinned (affinity = all cores, the live default)
  rdmaV    separate instance with RDMA_AR_VEC_PER_CTA=1024 (RoCEnante-like grid: ~2 packs/thread)
  rdmaT    separate instance with RDMA_AR_TIMING=1 (stage / net / tail breakdown, policy as rdma)
  roce2    RoCEnante, hcas rocep1s0f0,roceP2p1s0f1 (= rdma_ar's two devices), 2 rails per peer
  roce2k   RoCEnante, hcas rocep1s0f0,roceP2p1s0f0 (knapcio's choice: both PCIe halves of port 0)
  roce1    RoCEnante, hca rocep1s0f0 only (1 rail: isolates striping)
  roce2U   roce2 with its proxy unpinned
"""
import argparse
import contextlib
import ctypes
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HIDDEN = 5120
QKV_COLS = 224
MB = 1 << 20
CH = 16384
MAX_BYTES = 2 * MB
TWOSHOT_MIN = int(os.environ.get("RDMA_AR_TWOSHOT_MIN_BYTES", str(32 * 1024)))
STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_RDMA_EXT = os.path.expanduser("~/rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so")
LIVE_L2PF = os.path.expanduser("~/dsv41-stage/r4/l2-prefetch/libdsv41_l2pf.so")
RDMA_HCAS = ["rocep1s0f0", "roceP2p1s0f1"]
ROCE_HCAS = {"roce2": ("rocep1s0f0", "roceP2p1s0f1"), "roce2k": ("rocep1s0f0", "roceP2p1s0f0"),
             "roce1": ("rocep1s0f0",)}
GID = int(os.environ.get("RDMA_AR_GID_INDEX", "5"))
X925 = [19, 18, 17, 16, 15, 9, 8, 7, 6]  # Cortex-X925 (3.9 GHz) cores on GB10; 0-4,10-14 are A725
ALL_ARMS = ["rdma", "rdma1", "rdma2", "rdmaU", "rdmaV", "rdmaT", "roce2", "roce2k", "roce1", "roce2U"]


# ----------------------------------------------------------------------------------- inputs / refs --
def finfo_max(dtype):
    return float(torch.finfo(dtype).max)


def gen_inputs(M, dtype, seed, world, cols=HIDDEN):
    """CPU tensor [world, M, cols] of dtype, identical on every rank (seeded), with planted classes in
    the first 512 columns (cols >= 512); see the module docstring. Returns (x, classes dict of col slices)."""
    g = torch.Generator().manual_seed(1000003 * seed + 7919 * M + (0 if dtype == torch.bfloat16 else 17))
    scale = torch.exp2(torch.randint(-14, 14, (world, M, cols), generator=g).float())
    x = (torch.randn((world, M, cols), generator=g) * scale).to(dtype)
    classes = {}
    if cols < 512:
        return x, classes
    ibits = torch.int16 if dtype in (torch.bfloat16, torch.float16) else torch.int32
    b = x.view(ibits)
    neg0 = -32768 if ibits == torch.int16 else -2147483648
    # A: all ranks -0.0
    b[:, :, 0:64] = neg0
    classes["all_neg0"] = (0, 64)
    # B: mixed +-0 (rank 3 always +0)
    sgn = torch.randint(0, 2, (world, M, 64), generator=g, dtype=torch.int64).to(ibits) * neg0
    sgn[min(3, world - 1)] = 0
    b[:, :, 64:128] = sgn
    classes["mixed_zero"] = (64, 128)
    # C: subnormals (exponent 0, nonzero mantissa), random signs
    if ibits == torch.int16:
        mant = torch.randint(1, 128, (world, M, 128), generator=g, dtype=torch.int64)
    else:
        mant = torch.randint(1, 1 << 23, (world, M, 128), generator=g, dtype=torch.int64)
    s = torch.randint(0, 2, (world, M, 128), generator=g, dtype=torch.int64)
    sub = (mant | (s << (15 if ibits == torch.int16 else 31))).to(torch.int64)
    if ibits == torch.int16:
        sub = torch.where(sub >= 32768, sub - 65536, sub)
    else:
        sub = torch.where(sub >= (1 << 31), sub - (1 << 32), sub)
    b[:, :, 128:256] = sub.to(ibits)
    classes["subnormal"] = (128, 256)
    # D: exact cancellation x_r = a (r even), -a (r odd)
    a = (torch.randn((M, 64), generator=g) * 4).to(dtype)
    for r in range(world):
        x[r, :, 256:320] = a if r % 2 == 0 else -a
    classes["cancel"] = (256, 320)
    # E: order-dependent overflow: +0.9 max on the first half of the ranks, -0.9 max on the rest
    big = 0.9 * finfo_max(dtype)
    for r in range(world):
        x[r, :, 320:384] = big if r < world // 2 else -big
    classes["overflow"] = (320, 384)
    if world < 8:  # the remaining classes index ranks up to 7
        return x, classes
    # F: inf / NaN placements
    x[:, :, 384:448] = 1.0
    x[0, :, 384:400] = float("inf")
    x[3, :, 384:392] = float("-inf")       # 384..391 -> NaN, 392..399 -> +inf
    x[5, :, 400:416] = float("nan")
    x[7, :, 416:432] = float("-inf")
    x[:, :, 432:448] = float("inf")
    classes["inf_nan"] = (384, 448)
    # G: absorption: one large value first, small ones after (order sensitive in fp32)
    x[:, :, 448:512] = (torch.rand((world, M, 64), generator=g) * 2 - 1).to(dtype)
    x[0, :, 448:512] = (2.0 ** 24) * (1 + torch.rand((M, 64), generator=g)).to(dtype)
    classes["absorb"] = (448, 512)
    # H: bf16-visible order dependence: x0 = 2^26, x1 = -2^26, x2..x7 = 1. Rank order 0..7 gives 6;
    #    reversed order gives 8 (-2^26 + 6 rounds to -2^26 + 8 in fp32). RoCEnante's init also gives 6.
    x[:, :, 512:576] = 1.0
    x[0, :, 512:576] = 2.0 ** 26
    x[1, :, 512:576] = -(2.0 ** 26)
    classes["order"] = (512, 576)
    return x, classes


def class_mismatch(out, ref, classes):
    mm = bits_of(out) != bits_of(ref)
    d = {k: int(mm[..., lo:hi].sum()) for k, (lo, hi) in classes.items()}
    d["random"] = int(mm[..., 576:].sum()) if mm.shape[-1] > 576 else 0
    return d


def ref_sum(xs32, init_zero, order=None):
    """fp32 fixed-order sum of a [world, ...] fp32 tensor (any device). init_zero: acc starts at +0.0
    (our rule); else acc = first operand (RoCEnante). order: rank order (default 0..world-1)."""
    world = xs32.shape[0]
    order = list(range(world)) if order is None else list(order)
    if init_zero:
        acc = torch.zeros_like(xs32[0])
        for r in order:
            acc = acc + xs32[r]
    else:
        acc = xs32[order[0]].clone()
        for r in order[1:]:
            acc = acc + xs32[r]
    return acc


def f32_to_bf16_bits_np(a):
    """numpy float32 -> uint16 bf16 bits, round to nearest even; NaN -> 0x7FFF (GPU cvt.rn canonical)."""
    u = a.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r[np.isnan(a)] = 0x7FFF
    return r


def cpu_ref_bits(x_cpu, init_zero):
    """numpy sequential fp32 sum (IEEE, subnormals kept) -> bf16/fp32 bits as an int array."""
    xs = x_cpu.float().numpy()
    _es = np.errstate(all="ignore")
    _es.__enter__()
    acc = np.zeros_like(xs[0]) if init_zero else xs[0].copy()
    for r in range(0 if init_zero else 1, xs.shape[0]):
        acc = (acc + xs[r]).astype(np.float32)
    if x_cpu.dtype == torch.bfloat16:
        return f32_to_bf16_bits_np(acc).astype(np.int64), np.isnan(acc)
    return acc.view(np.uint32).astype(np.int64), np.isnan(acc)


def bits_of(t):
    t = t.detach().contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16).to(torch.int64) & 0xFFFF
    return t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def compare(out, ref, all_neg0_mask):
    """bitwise compare -> dict(total, mismatch, all_neg0, nan_bits, other, first_other)."""
    ob, rb = bits_of(out), bits_of(ref)
    mm = ob != rb
    both_nan = torch.isnan(out.float()) & torch.isnan(ref.float())
    n0 = mm & all_neg0_mask
    nb = mm & both_nan & ~all_neg0_mask
    other = mm & ~all_neg0_mask & ~both_nan
    first = None
    if bool(other.any()):
        idx = other.nonzero()[0].tolist()
        first = dict(idx=idx, out=hex(int(ob[tuple(idx)])), ref=hex(int(rb[tuple(idx)])))
    return dict(total=int(mm.numel()), mismatch=int(mm.sum()), all_neg0=int(n0.sum()), nan_bits=int(nb.sum()),
                other=int(other.sum()), first_other=first)


def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------------------- utilities --
def log0(*a):
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*a, flush=True)


def tids():
    return set(int(t) for t in os.listdir("/proc/self/task"))


@contextlib.contextmanager
def inherit_affinity(core):
    """Threads created inside inherit {core} (pthread_create copies the creator's mask)."""
    if core is None or core < 0:
        yield
        return
    orig = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {core})
    try:
        yield
    finally:
        os.sched_setaffinity(0, orig)


def pinned_tids(before, core):
    out = []
    for t in tids() - before:
        try:
            if core is not None and core >= 0 and os.sched_getaffinity(t) == {core}:
                out.append(t)
        except OSError:
            pass
    return out


def set_tid_affinity(tlist, cores):
    for t in tlist:
        try:
            os.sched_setaffinity(t, cores)
        except OSError as e:
            log0(f"affinity({t}) failed: {e}")


def load_ext(path):
    spec = importlib.util.spec_from_file_location("rdma_ar_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# -------------------------------------------------------------------------------------------- arms --
class RdmaInst:
    """One RdmaAR instance (live .so), constructed collectively. Env read by the .so at construction
    (RDMA_AR_VEC_PER_CTA, RDMA_AR_TIMING) is set only around the constructor."""

    def __init__(self, ext, grp, rank, world, core, env=None, timeout_ms=120000):
        import torch.distributed as dist
        saved = {}
        for k, v in (env or {}).items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v
        before = tids()
        try:
            with inherit_affinity(core):
                self.h = ext.RdmaAR(rank, world, MAX_BYTES, MAX_BYTES, RDMA_HCAS, GID, 1, -1, timeout_ms, 0)
                infos = [None] * world
                dist.all_gather_object(infos, self.h.local_info(), group=grp)
                self.h.connect(infos)  # proxy thread starts here, inherits {core}
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        dist.barrier(group=grp)
        self.proxy_tids = pinned_tids(before, core)
        self.core = core
        self.world = world
        self.arena = list(self.h.arena_info())

    def ar(self, x, policy):
        nb = x.numel() * x.element_size()
        two = (TWOSHOT_MIN > 0 and nb >= TWOSHOT_MIN) if policy == "live" else (policy == "two")
        return self.h.all_reduce(x, two)

    def ag_cols(self, x):
        """live cuda_communicator path for dim=-1: stacked [world, *x.shape] + movedim/reshape copy."""
        out = torch.empty((self.world,) + tuple(x.shape), dtype=x.dtype, device=x.device)
        self.h.all_gather_into(out, x)
        d = x.dim() - 1
        return out.movedim(0, d).reshape(tuple(x.shape[:d]) + (self.world * x.shape[d],))


class RoceInst:
    """RoceOneshotAllReduce + the launchers its Plan would materialize (same calls as
    b12x.comm.roce._preparation.plan(...).materialize, without the session/publish machinery)."""

    def __init__(self, grp, hcas, core):
        from b12x.comm.roce import _allgather_cute as agc
        from b12x.comm.roce._oneshot_cute import get_launcher
        from b12x.comm.roce._preparation import _PreparedRoce
        from b12x.comm.roce.roce_oneshot import RoceOneshotAllReduce

        before = tids()
        with inherit_affinity(core):
            self.rt = RoceOneshotAllReduce(exchange_group=grp, device=torch.device("cuda", 0), max_size=MAX_BYTES,
                                           max_gather_bytes=MAX_BYTES, hca_names=tuple(hcas), gid_index=GID)
        self.proxy_tids = pinned_tids(before, core)
        self.core = core
        rt = self.rt
        t0 = time.time()
        rt._prepare_resources((torch.bfloat16, torch.float32), padded_gather=True)
        self.prep = _PreparedRoce(rt, {dt: get_launcher(*rt._launcher_key(dt)) for dt in (torch.bfloat16, torch.float32)},
                                  agc.get_launcher(*rt._gather_launcher_key()))
        self.compile_s = time.time() - t0

    def ar(self, x, out=None):
        return self.rt._run_prepared_all_reduce(x, prepared=self.prep, out=out)

    def ag_cols(self, x, out=None):
        return self.rt._run_prepared_all_gather(x, prepared=self.prep, dim=-1, out=out)

    def capture(self):
        return self.rt.capture()


class Arm:
    def __init__(self, name, inst, kind, policy=None, unpin=False):
        self.name, self.inst, self.kind, self.policy, self.unpin = name, inst, kind, policy, unpin

    def ar(self, x):
        if self.kind == "rdma":
            return self.inst.ar(x, self.policy)
        return self.inst.ar(x)

    def ag(self, x):
        return self.inst.ag_cols(x)

    def capture_ctx(self):
        return self.inst.capture() if self.kind == "roce" else contextlib.nullcontext()

    def enter(self):
        """affinity for this arm's measurement: unpinned arms float on all cores."""
        if self.unpin:
            set_tid_affinity(self.inst.proxy_tids, set(range(os.cpu_count())))

    def leave(self):
        if self.unpin and self.inst.core is not None and self.inst.core >= 0:
            set_tid_affinity(self.inst.proxy_tids, {self.inst.core})

    def health(self):
        if self.kind == "roce":
            self.inst.rt.check_health()


# -------------------------------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--phases", default="X,L,C")
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--rows", default="1,6,12,24,48")
    ap.add_argument("--xover-rows", default="2,3,4,5,6,8,10,12,16")
    ap.add_argument("--modes", default="b2b,gap,l2pf,dram")
    ap.add_argument("--kinds", default="ar,ag,step")
    ap.add_argument("--n-ops", type=int, default=40)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--gap-us", type=float, default=100.0)
    ap.add_argument("--pf-mb", type=float, default=6.0)
    ap.add_argument("--pf-segs", type=int, default=2)
    ap.add_argument("--pin", default="x925", help="x925: proxies on distinct X925 cores; none: unpinned")
    ap.add_argument("--rdma-ext", default=LIVE_RDMA_EXT)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--cpu-selftest", action="store_true", help="CPU only: generator + reference checks")
    a = ap.parse_args()
    if a.cpu_selftest:
        return cpu_selftest(a)

    import torch.distributed as dist

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    grp = dist.group.WORLD
    arms_req = [s for s in a.arms.split(",") if s]
    rows = [int(s) for s in a.rows.split(",") if s]
    res = {"rank": rank, "world": world, "args": vars(a), "host": os.uname().nodename,
           "env": {k: v for k, v in os.environ.items() if k.startswith(("RDMA_AR", "B12X", "NCCL_", "CUDA"))},
           "rdma_ext": a.rdma_ext, "rdma_ext_sha": file_sha(a.rdma_ext), "l2pf_sha": file_sha(LIVE_L2PF)}
    try:
        import importlib.metadata as md
        res["versions"] = {p: md.version(p) for p in ("flashinfer-python", "nvidia-cutlass-dsl", "torch")}
    except Exception as e:  # noqa: BLE001
        res["versions"] = str(e)
    cores = iter(X925 if a.pin == "x925" else [])

    def next_core():
        return next(cores, -1) if a.pin == "x925" else -1

    torch.cuda.synchronize()
    # ---- instances (collective, same order on every rank)
    ext = load_ext(a.rdma_ext)
    insts = {}
    need_rdma = any(n in arms_req for n in ("rdma", "rdma1", "rdma2", "rdmaU"))
    if need_rdma:
        insts["rdma"] = RdmaInst(ext, grp, rank, world, next_core())
    if "rdmaV" in arms_req:
        insts["rdmaV"] = RdmaInst(ext, grp, rank, world, next_core(), env={"RDMA_AR_VEC_PER_CTA": "1024"})
    if "rdmaT" in arms_req:
        insts["rdmaT"] = RdmaInst(ext, grp, rank, world, next_core(), env={"RDMA_AR_TIMING": "1"})
    for nm in ("roce2", "roce2k", "roce1"):
        if nm in arms_req or (nm == "roce2" and "roce2U" in arms_req):
            t0 = time.time()
            insts[nm] = RoceInst(grp, ROCE_HCAS[nm], next_core())
            log0(f"{nm}: hcas={ROCE_HCAS[nm]} rails={insts[nm].rt.rail_count} routes={insts[nm].rt.routes[(rank + 1) % world]} "
                 f"proxy_tids={insts[nm].proxy_tids} compile {insts[nm].compile_s:.1f} s (init {time.time() - t0:.1f} s)")
    arms = {}
    for nm in arms_req:
        if nm in ("rdma", "rdma1", "rdma2", "rdmaU"):
            arms[nm] = Arm(nm, insts["rdma"], "rdma", {"rdma": "live", "rdma1": "one", "rdma2": "two", "rdmaU": "live"}[nm],
                           unpin=(nm == "rdmaU"))
        elif nm in ("rdmaV", "rdmaT"):
            arms[nm] = Arm(nm, insts[nm], "rdma", "live")
        elif nm == "roce2U":
            arms[nm] = Arm(nm, insts["roce2"], "roce", unpin=True)
        else:
            arms[nm] = Arm(nm, insts[nm], "roce")
    res["instances"] = {k: dict(proxy_tids=v.proxy_tids, core=v.core) for k, v in insts.items()}
    if "rdma" in insts:
        res["pingpong_ns_pinned"] = insts["rdma"].h.pingpong(2000)
        set_tid_affinity(insts["rdma"].proxy_tids, set(range(os.cpu_count())))
        time.sleep(0.05)
        res["pingpong_ns_unpinned"] = insts["rdma"].h.pingpong(2000)
        if insts["rdma"].core >= 0:
            set_tid_affinity(insts["rdma"].proxy_tids, {insts["rdma"].core})
        res["rdma_arena"] = insts["rdma"].arena
    log0(f"init ok: world={world} arms={list(arms)} rdma_ext={a.rdma_ext} ({res['rdma_ext_sha']}) "
         f"pingpong pinned/unpinned={res.get('pingpong_ns_pinned', 0):.0f}/{res.get('pingpong_ns_unpinned', 0):.0f} ns "
         f"versions={res['versions']}")
    fails = 0

    # ================================================================================= phase X ===
    if "X" in a.phases.split(","):
        xres = []
        for dtype, rws in ((torch.bfloat16, rows), (torch.float32, [r for r in rows if r <= 12])):
            for M in rws:
                xc, classes = gen_inputs(M, dtype, a.seed, world)
                x32 = xc.to(dev).float()
                ref_zero = ref_sum(x32, True).to(dtype)
                ref_r0 = ref_sum(x32, False).to(dtype)
                ref_rev = ref_sum(x32, True, order=range(world - 1, -1, -1)).to(dtype)
                neg0 = (bits_of(xc.to(dev)) == (0x8000 if dtype == torch.bfloat16 else 0x80000000)).all(dim=0)
                cbits, cnan = cpu_ref_bits(xc, True)
                gbits = bits_of(ref_zero).cpu().numpy()
                cpu_vs_gpu = int(((cbits != gbits) & ~cnan).sum())
                sens = int((bits_of(ref_rev) != bits_of(ref_zero)).sum())
                n_allneg0 = int(neg0.sum())
                mine = xc[rank].to(dev).contiguous()
                row = dict(dtype=str(dtype), M=M, cpu_vs_gpu_ref=cpu_vs_gpu, reversed_order_diffs=sens,
                           all_neg0_elems=n_allneg0, arms={})
                if cpu_vs_gpu:
                    fails += 1
                for nm, arm in arms.items():
                    if nm in ("rdmaU", "roce2U", "rdmaT", "rdmaV") and dtype != torch.bfloat16:
                        continue
                    out_e = arm.ar(mine.clone())
                    torch.cuda.synchronize()
                    arm.health()
                    # graph: one op, 3 replays (stable input buffer)
                    xin = mine.clone()
                    st = torch.cuda.Stream()
                    st.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(st):
                        arm.ar(xin)
                    torch.cuda.current_stream().wait_stream(st)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with arm.capture_ctx():
                        with torch.cuda.graph(g):
                            out_g = arm.ar(xin)
                    outs_g = []
                    for _ in range(3):
                        g.replay()
                        torch.cuda.synchronize()
                        outs_g.append(out_g.clone())
                    arm.health()
                    cz = compare(out_e, ref_zero, neg0)
                    c0 = compare(out_e, ref_r0, neg0)
                    graph_eq = all(torch.equal(bits_of(o), bits_of(out_e)) for o in outs_g)
                    hs = [None] * world
                    dist.all_gather_object(hs, sha(out_e), group=grp)
                    same = all(h == hs[0] for h in hs)
                    if arm.kind == "rdma":
                        ok = cz["mismatch"] == 0 and graph_eq and same
                        verdict = "EXACT" if ok else "FAIL"
                    else:
                        ok = c0["mismatch"] == 0 and cz["other"] == 0 and cz["nan_bits"] == 0 and graph_eq and same
                        verdict = ("EXACT" if cz["mismatch"] == 0 else "EXACT_EXCEPT_ALL_NEG0") if ok else "FAIL"
                        if ok and cz["all_neg0"] != n_allneg0:
                            verdict = "FAIL"  # every all-(-0.0) element must differ (+0 vs -0) and nothing else
                            ok = False
                    if not ok:
                        fails += 1
                    row["arms"][nm] = dict(vs_ref_zero=cz, vs_ref_r0=c0, graph_equal=graph_eq, ranks_identical=same,
                                           verdict=verdict, by_class_vs_zero=class_mismatch(out_e, ref_zero, classes))
                    log0(f"[X] {str(dtype)[6:]:8s} M={M:2d} {nm:7s} {verdict:22s} vs ours: mm={cz['mismatch']} "
                         f"(all-neg0 {cz['all_neg0']}/{n_allneg0}, nan-bits {cz['nan_bits']}, other {cz['other']}) "
                         f"vs r0-init: mm={c0['mismatch']} graph==eager {graph_eq} ranks-identical {same}")
                xres.append(row)
                log0(f"[X] {str(dtype)[6:]:8s} M={M:2d} refs: cpu-vs-gpu {cpu_vs_gpu} (must be 0), reversed-order "
                     f"diffs {sens} (> 0: the test sees order), all-neg0 elems {n_allneg0}")
        # gathers: [M,224] along dim -1, random bits (incl. NaN payloads), bitwise vs torch.cat
        for M in rows:
            gq = torch.Generator().manual_seed(a.seed + 31 * M)
            allx = torch.randint(-32768, 32767, (world, M, QKV_COLS), generator=gq, dtype=torch.int32).to(torch.int16)
            allx = allx.view(torch.bfloat16)
            want = torch.cat([allx[r] for r in range(world)], dim=-1).to(dev)
            mine = allx[rank].to(dev).contiguous()
            grow = dict(M=M, arms={})
            for nm, arm in arms.items():
                if nm in ("rdma1", "rdma2", "rdmaU", "roce2U"):
                    continue
                outg = arm.ag(mine)
                torch.cuda.synchronize()
                arm.health()
                eq = torch.equal(bits_of(outg), bits_of(want))
                grow["arms"][nm] = eq
                if not eq:
                    fails += 1
            log0(f"[X] gather M={M:2d} x{QKV_COLS} dim=-1 bitwise == torch.cat: {grow['arms']}")
            xres.append(dict(gather=grow))
        res["X"] = xres

    # ================================================================================= phase L ===
    if set(a.phases.split(",")) & {"L", "C"}:
        lib = ctypes.CDLL(LIVE_L2PF)
        lib.dsv41_l2pf.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p]
        lib.dsv41_l2pf.restype = ctypes.c_int
        lib.dsv41_spin.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p]
        lib.dsv41_spin.restype = ctypes.c_int
        side, rdr = torch.cuda.Stream(), torch.cuda.Stream()
        n = a.n_ops
        pf_bytes = int(a.pf_mb * MB) & ~15
        wbufs = [torch.empty(pf_bytes // 2, dtype=torch.bfloat16, device=dev) for _ in range(n)]
        for t in wbufs:
            t.normal_()
        segb = (pf_bytes // max(1, a.pf_segs)) & ~15  # serving windows are 1-3 weight tensors (1 CTA each)
        tabs = [torch.tensor(sum(([t.data_ptr() + k * segb, segb] for k in range(a.pf_segs)), []), dtype=torch.int64,
                             device=dev) for t in wbufs]
        big = torch.empty(512 * MB // 2, dtype=torch.bfloat16, device=dev)
        seg = 64 * MB
        reader_tab = torch.tensor(sum(([big.data_ptr() + i * seg, seg] for i in range(8)), []) * 32,
                                  dtype=torch.int64, device=dev)  # 32 x 512 MB per launch

        def spin(us):
            if us > 0:
                assert lib.dsv41_spin(int(us * 1000), 8, torch.cuda.current_stream().cuda_stream) == 0

        def fork_pf(i):
            side.wait_stream(torch.cuda.current_stream())
            assert lib.dsv41_l2pf(tabs[i % len(tabs)].data_ptr(), a.pf_segs, CH, side.cuda_stream) == 0

        def reader_start():
            e = torch.cuda.Event()
            e.record()
            rdr.wait_event(e)
            for _ in range(6):  # ~0.4 s of DRAM reads at ~250 GB/s; drained after each timed block
                assert lib.dsv41_l2pf(reader_tab.data_ptr(), reader_tab.numel() // 2, CH, rdr.cuda_stream) == 0

        def reader_stop():
            rdr.synchronize()

        def body_fn(arm, kind, M, mode, with_ops):
            xa = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=dev)
            xg = torch.randn(M, QKV_COLS, dtype=torch.bfloat16, device=dev)
            seq = {"ar": ["ar"] * n, "ag": ["ag"] * n, "step": ["ar", "ag", "ar"] * (n // 3 or 1)}[kind]

            def body():
                for i, op in enumerate(seq):
                    if mode != "b2b":
                        spin(a.gap_us)
                    if mode == "l2pf":
                        fork_pf(i)
                    if with_ops:
                        if op == "ar":
                            arm.ar(xa)
                        else:
                            arm.ag(xg)
                if mode == "l2pf":
                    torch.cuda.current_stream().wait_stream(side)
            return body, len(seq)

        def graph_of(fn, ctx):
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                fn()
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with ctx:
                with torch.cuda.graph(g):
                    fn()
            torch.cuda.synchronize()
            return g

        def time_graph(g, nops, mode, arm=None):
            if arm is not None:
                arm.enter()
            try:
                dist.barrier(group=grp)
                for _ in range(2):
                    g.replay()
                torch.cuda.synchronize()
                dist.barrier(group=grp)
                if mode == "dram":
                    reader_start()
                if arm is not None and arm.name == "rdmaT":
                    arm.inst.h.stats(True)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(a.replays):
                    g.replay()
                e1.record()
                torch.cuda.synchronize()
                if mode == "dram":
                    reader_stop()
                parts = None
                if arm is not None and arm.name == "rdmaT":
                    parts = [v / 1000.0 for v in arm.inst.h.stats(True)[1:5]]
                if arm is not None:
                    arm.health()
                return e0.elapsed_time(e1) * 1000.0 / (a.replays * nops), parts
            finally:
                if arm is not None:
                    arm.leave()

        def run_cell(mode, kind, M, names, tag):
            base_body, nops = body_fn(None, kind, M, mode, False)
            gb = graph_of(base_body, contextlib.nullcontext()) if mode != "b2b" else None
            graphs = {}
            for nm in names:
                fn, nops = body_fn(arms[nm], kind, M, mode, True)
                graphs[nm] = graph_of(fn, arms[nm].capture_ctx())
            per = {nm: [] for nm in graphs}
            parts = {nm: [] for nm in graphs}
            basel = []
            for t in range(a.trials):
                order = names[t % len(names):] + names[:t % len(names)]
                if t % 2:
                    order = order[::-1]
                b = time_graph(gb, nops, mode)[0] if gb is not None else 0.0
                basel.append(b)
                for nm in order:
                    us, pt = time_graph(graphs[nm], nops, mode, arms[nm])
                    per[nm].append(us - b)
                    if pt is not None:
                        parts[nm].append(pt)
            med = {nm: float(np.median(v)) for nm, v in per.items()}
            row = dict(mode=mode, kind=kind, M=M, nops=nops, base_us=float(np.median(basel)), us_op=med,
                       trials=per, parts={k: np.median(np.array(v), axis=0).tolist() for k, v in parts.items() if v})
            best = min(med, key=med.get)
            log0(f"[{tag}] {mode:4s} {kind:4s} M={M:2d} " + " ".join(f"{k} {v:6.1f}" for k, v in med.items())
                 + f"  (us/op; best {best})" + (f" rdmaT stage/net/tail/total {['%.1f' % p for p in row['parts']['rdmaT']]}"
                                                if "rdmaT" in row["parts"] else ""))
            del graphs, gb
            torch.cuda.synchronize()
            return row

        lres = []
        if "L" in a.phases.split(","):
            for mode in [s for s in a.modes.split(",") if s]:
                for kind in [s for s in a.kinds.split(",") if s]:
                    for M in rows:
                        if M * HIDDEN * 2 <= MAX_BYTES:
                            lres.append(run_cell(mode, kind, M, list(arms), "L"))
        res["L"] = lres

        # ---- phase C: one-shot vs two-shot crossover of rdma_ar (an env-only, exact policy knob:
        # RDMA_AR_TWOSHOT_MIN_BYTES), serving-like modes, plus roce2 for reference
        if "C" in a.phases.split(",") and "rdma1" in arms and "rdma2" in arms:
            cnames = [n for n in ("rdma1", "rdma2", "roce2") if n in arms]
            crows = [int(s) for s in a.xover_rows.split(",") if s]
            cres = []
            for mode in ("gap", "l2pf"):
                for M in crows:
                    cres.append(run_cell(mode, "ar", M, cnames, "C"))
            # crossover: smallest M from which two-shot beats one-shot by >= 1 us in both modes for every larger M
            win = {M: all(r["us_op"]["rdma2"] + 1.0 <= r["us_op"]["rdma1"] for r in cres if r["M"] == M) for M in crows}
            cross = None
            for M in sorted(crows):
                if all(win[m] for m in crows if m >= M):
                    cross = M
                    break
            rec = None if cross is None else cross * HIDDEN * 2
            res["C"] = dict(rows=cres, two_shot_wins=win, crossover_rows=cross, recommend_twoshot_min_bytes=rec,
                            live_twoshot_min_bytes=TWOSHOT_MIN)
            log0(f"[C] two-shot wins by row count: {win}; crossover rows={cross} -> RDMA_AR_TWOSHOT_MIN_BYTES="
                 f"{rec} (live {TWOSHOT_MIN}: two-shot from {-(-TWOSHOT_MIN // (HIDDEN * 2))} rows)")
        res["roce_stats"] = {k: v.rt.stats() for k, v in insts.items() if isinstance(v, RoceInst)}

    # gather every rank's results to rank 0 (skew view) and write
    allres = [None] * world
    dist.all_gather_object(allres, {"fails": fails, "L": [dict(mode=r["mode"], kind=r["kind"], M=r["M"], us_op=r["us_op"])
                                                          for r in res.get("L", [])]}, group=grp)
    res["fails"] = fails
    if a.out:
        with open(f"{a.out}.rank{rank}.json", "w") as f:
            json.dump(res if rank == 0 else {k: v for k, v in res.items() if k != "X"}, f, indent=1, default=str)
        if rank == 0:
            with open(f"{a.out}.allranks.json", "w") as f:
                json.dump(allres, f, indent=1, default=str)
    log0(f"DONE failures per rank: {[r['fails'] for r in allres]}")
    for v in insts.values():
        if isinstance(v, RoceInst):
            with contextlib.suppress(Exception):
                v.rt.close()
    dist.barrier(group=grp)
    return 0 if all(r["fails"] == 0 for r in allres) else 1


def cpu_selftest(a):
    """No CUDA: checks the generator's planted classes and that the torch reference equals the numpy one."""
    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "CPU only"
    ok = True
    for dtype in (torch.bfloat16, torch.float32):
        for M in (1, 6):
            x, classes = gen_inputs(M, dtype, a.seed, 8)
            x2, _ = gen_inputs(M, dtype, a.seed, 8)
            det = torch.equal(bits_of(x), bits_of(x2))
            x32 = x.float()
            tz = ref_sum(x32, True)
            tr0 = ref_sum(x32, False)
            nb, nn = cpu_ref_bits(x, True)
            nb0, _ = cpu_ref_bits(x, False)
            if dtype == torch.bfloat16:
                tzb = f32_to_bf16_bits_np(tz.numpy()).astype(np.int64)
                tr0b = f32_to_bf16_bits_np(tr0.numpy()).astype(np.int64)
            else:
                tzb = tz.numpy().view(np.uint32).astype(np.int64)
                tr0b = tr0.numpy().view(np.uint32).astype(np.int64)
            agree = int((tzb != nb).sum()) == 0 and int((tr0b != nb0).sum()) == 0
            neg0 = (bits_of(x) == (0x8000 if dtype == torch.bfloat16 else 0x80000000)).all(dim=0).numpy()
            diff = tzb != tr0b
            only_neg0 = bool((diff == neg0).all())
            lo, hi = classes["all_neg0"]
            zb = tzb.reshape(M, -1)
            r0b = tr0b.reshape(M, -1)
            neg0_vals = (bool((zb[:, lo:hi] == 0).all()) and
                         bool((r0b[:, lo:hi] == (0x8000 if dtype == torch.bfloat16 else 0x80000000)).all()))
            s0, s1 = classes["subnormal"]
            sub_nonzero = bool((zb[:, s0:s1] != 0).any())
            o0, o1 = classes["overflow"]
            inf_bits = 0x7F80 if dtype == torch.bfloat16 else 0x7F800000
            ovf = bool((zb[:, o0:o1] == inf_bits).all())
            rev = ref_sum(x32, True, order=range(7, -1, -1))
            revb = (f32_to_bf16_bits_np(rev.numpy()) if dtype == torch.bfloat16
                    else rev.numpy().view(np.uint32)).astype(np.int64)
            sens = int((revb != tzb).sum())
            line = (f"{str(dtype)[6:]:8s} M={M}: deterministic {det}, torch==numpy {agree}, zero-vs-r0 differs exactly "
                    f"on all-neg0 {only_neg0} (+0 vs -0 {neg0_vals}), subnormal sums nonzero {sub_nonzero}, "
                    f"order-overflow -> +inf {ovf}, reversed-order diffs {sens}")
            print(line, flush=True)
            ok = ok and det and agree and only_neg0 and neg0_vals and sub_nonzero and ovf and sens > 0
    print("RESULT", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
