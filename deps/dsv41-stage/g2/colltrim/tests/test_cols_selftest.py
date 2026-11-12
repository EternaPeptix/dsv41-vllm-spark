#!/usr/bin/env python3
"""g2 colltrim (a) single-GPU bitwise test of the column all-gather. ONE GPU, no HCA, no network, ~1 min.
Run only in a stopped window (no serving on that GPU):

  source ~/venvs/vllm032/bin/activate && cd ~/dsv41-stage/g2/colltrim && python3 tests/test_cols_selftest.py

Uses the SelfTest class of build/rdma_ar_ext.so: the real ar_kernel / ar_kernel_pl with every peer flag preset,
so the kernel stages, "publishes" to nobody and goes straight to its consumer loop, reading what the test
wrote into the recv slots as the 7 peers' payload (k1's selftest_k1.py harness, same arena layout).
Gates (exit 1 on any failure), for both kernel sets (parload, legacy), several ranks and every shape:
  1 column gather == plain gather + movedim(0, dim).reshape (today's path) == the reference concat,
    compared as raw bytes;
  2 guard bytes around the output (64 x 16 B each side, sentinel 0x5A) are untouched;
  3 seg_bytes == nbytes reproduces the plain dim-0 gather;
  4 CUDA graph: a captured 40-gather graph (cols, M 1..8 x 224 bf16 = the qkv_a shard) replayed with fresh
    inputs and fresh peer images, every replay bitwise;
  5 bad seg_bytes (not a multiple of 16 / not dividing the size) is refused by the extension.
Timing (reported, not gated): 40-call graph, cols vs stacked + movedim/reshape copy, M 1..8 x 224, local
part only (no network): the saving per call is the copy kernel plus its launch slot.
"""
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SO = os.environ.get("G2CT_EXT", os.path.join(ROOT, "build", "rdma_ar_ext.so"))
WORLD = 8
MAX_BYTES = 2 * 1024 * 1024
MAX_REDUCE = 2 * 1024 * 1024
GUARD_VEC = 64
SENTINEL = 0x5A


def load_ext(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("rdma_ar_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ext = load_ext(SO)
assert getattr(ext, "G2_VARIANT", None), f"{SO} has no G2_VARIANT (not the colltrim build)"
dev = torch.device("cuda:0")
torch.cuda.set_device(0)
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL " + msg, flush=True)
    return bool(cond)


def guarded(numel_bytes):
    """uint8 buffer with GUARD_VEC*16 sentinel bytes on both sides; returns (buf, inner view)."""
    g = GUARD_VEC * 16
    buf = torch.full((g + numel_bytes + g,), SENTINEL, dtype=torch.uint8, device=dev)
    return buf, buf[g:g + numel_bytes]


def guards_ok(buf):
    g = GUARD_VEC * 16
    return bool((buf[:g] == SENTINEL).all().item()) and bool((buf[-g:] == SENTINEL).all().item())


def rand_like(shape, dtype, gen):
    if dtype.is_floating_point:
        x = torch.randn(shape, generator=gen) * 4.0
        return x.to(dtype)
    return torch.randint(-2**31, 2**31 - 1, shape, generator=gen, dtype=torch.int64).to(dtype)


CASES = [  # (shape, dim, dtype, what)
    ((1, 224), 1, torch.bfloat16, "qkv_a M1"), ((2, 224), 1, torch.bfloat16, "qkv_a M2"),
    ((4, 224), 1, torch.bfloat16, "qkv_a M4"), ((6, 224), 1, torch.bfloat16, "qkv_a M6"),
    ((7, 224), 1, torch.bfloat16, "qkv_a M7"), ((8, 224), 1, torch.bfloat16, "qkv_a M8"),
    ((6, 16160), 1, torch.bfloat16, "logits M6"), ((48, 16160), 1, torch.bfloat16, "logits M48"),
    ((64, 16160), 1, torch.bfloat16, "logits M64 (2.07 MB)"),
    ((6, 3, 256), 1, torch.bfloat16, "engram 3x256"), ((192, 3, 256), 1, torch.bfloat16, "engram M192"),
    ((6, 3, 512), 2, torch.bfloat16, "3-D dim 2"), ((3, 5, 40), 1, torch.bfloat16, "seg 400 B"),
    ((7, 24), 1, torch.bfloat16, "seg 48 B"), ((5, 8), 1, torch.bfloat16, "seg 16 B (minimum)"),
    ((6, 8), 1, torch.float32, "fp32 seg 32 B"), ((6, 640), 1, torch.float32, "fp32 main_proj-like"),
    ((6, 4), 1, torch.int32, "int32 seg 16 B"), ((96, 640), 1, torch.bfloat16, "main_proj-like M96"),
]

st_cache = {}


def selftest(rank):
    if rank not in st_cache:
        st_cache[rank] = ext.SelfTest(rank, WORLD, MAX_BYTES, MAX_REDUCE, 0)
    return st_cache[rank]


def run_case(shape, dim, dtype, rank, par, seed):
    st = selftest(rank)
    gen = torch.Generator().manual_seed(seed)
    numel = 1
    for s in shape:
        numel *= s
    esz = torch.tensor([], dtype=dtype).element_size()
    nbytes = numel * esz
    peers = [rand_like(shape, dtype, gen) for _ in range(WORLD)]
    img = torch.zeros((WORLD, nbytes), dtype=torch.uint8)
    for r in range(WORLD):
        img[r] = peers[r].contiguous().view(torch.uint8).view(-1)
    st.set_recv(img)
    x = peers[rank].to(dev).contiguous()
    stacked_ref = torch.stack(peers, 0)  # [world, *shape] (rank's own row = its input)
    ref = stacked_ref.movedim(0, dim).reshape(shape[:dim] + (WORLD * shape[dim],) + shape[dim + 1:]).contiguous()
    ref_b = ref.view(torch.uint8).view(-1)
    # today's path: plain gather into [world, *shape], then movedim/reshape
    old = torch.empty((WORLD,) + shape, dtype=dtype, device=dev)
    st.all_gather_into(old, x, par)
    old_final = old.movedim(0, dim).reshape(ref.shape)
    # column path into a guarded buffer
    seg = esz
    for s in shape[dim:]:
        seg *= s
    buf, inner = guarded(WORLD * nbytes)
    out = inner.view(dtype).view(ref.shape)
    st.all_gather_cols_into(out, x, seg, par)
    # seg == nbytes: the plain dim-0 gather
    buf0, inner0 = guarded(WORLD * nbytes)
    st.all_gather_cols_into(inner0.view(dtype).view((WORLD,) + shape), x, nbytes, par)
    torch.cuda.synchronize()
    ok = check(torch.equal(old_final.contiguous().view(torch.uint8).view(-1).cpu(), ref_b),
               f"plain+movedim != reference {shape} dim{dim} {dtype} rank{rank} par{par}")
    ok &= check(torch.equal(out.view(torch.uint8).view(-1).cpu(), ref_b),
                f"cols != reference {shape} dim{dim} {dtype} rank{rank} par{par}")
    ok &= check(guards_ok(buf), f"cols wrote outside its output {shape} dim{dim} rank{rank} par{par}")
    ok &= check(torch.equal(inner0.cpu(), torch.stack(peers, 0).contiguous().view(torch.uint8).view(-1)),
                f"cols with seg=nbytes != dim-0 gather {shape} rank{rank} par{par}")
    ok &= check(guards_ok(buf0), f"seg=nbytes wrote outside its output {shape} rank{rank} par{par}")
    return ok


print(f"[1-3] eager bitwise: {len(CASES)} shapes x ranks (0, 3, 7) x kernels (parload, legacy)", flush=True)
n = n_ok = 0
for shape, dim, dtype, what in CASES:
    for rank in (0, 3, 7):
        for par in (True, False):
            n += 1
            n_ok += run_case(shape, dim, dtype, rank, par, seed=hash((shape, dim, rank, par)) & 0xFFFF)
print(f"    {n_ok}/{n} cases bitwise, guards intact", flush=True)

# ---- 4 graph replay -----------------------------------------------------------------------------------
print("[4] CUDA graph: 40 column gathers (M 1..8 x 224 bf16), 30 replays with fresh inputs", flush=True)
rank = 5
st = selftest(rank)
Ms = [1 + (i % 8) for i in range(40)]
ins = [torch.zeros((m, 224), dtype=torch.bfloat16, device=dev) for m in Ms]
seg = 224 * 2
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for x in ins:  # warm-up (eager) on the capture stream
        st.all_gather_cols_into(torch.empty((x.shape[0], WORLD * 224), dtype=x.dtype, device=dev), x, seg, True)
torch.cuda.current_stream().wait_stream(side)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
outs = []
with torch.cuda.graph(g):
    for x in ins:
        o = torch.empty((x.shape[0], WORLD * 224), dtype=x.dtype, device=dev)
        st.all_gather_cols_into(o, x, seg, True)
        outs.append(o)
gfail = 0
for rep in range(30):
    gen = torch.Generator().manual_seed(9000 + rep)
    peers = [torch.randn((8, 224), generator=gen).to(torch.bfloat16) for _ in range(WORLD)]
    img = torch.zeros((WORLD, 8 * 224 * 2), dtype=torch.uint8)
    for r in range(WORLD):
        img[r] = peers[r].view(torch.uint8).view(-1)
    st.set_recv(img)  # every gather reads the first M rows of each peer image
    for x, m in zip(ins, Ms):
        x.copy_(peers[rank][:m].to(dev))
    g.replay()
    torch.cuda.synchronize()
    for o, m in zip(outs, Ms):
        ref = torch.stack([p[:m] for p in peers], 0).movedim(0, 1).reshape(m, WORLD * 224)
        if not torch.equal(o.cpu().view(torch.int16), ref.view(torch.int16)):
            gfail += 1
check(gfail == 0, f"graph replay: {gfail} mismatching outputs")
print(f"    {30 * len(ins) - gfail}/{30 * len(ins)} replayed gathers bitwise", flush=True)
del g

# ---- 5 refusal ------------------------------------------------------------------------------------------
print("[5] bad seg_bytes refused", flush=True)
x = torch.zeros((6, 224), dtype=torch.bfloat16, device=dev)
for bad in (0, 8, 24, 2688 + 16, 448 * 5):
    try:
        st.all_gather_cols_into(torch.empty((6, WORLD * 224), dtype=x.dtype, device=dev), x, bad, True)
        torch.cuda.synchronize()
        check(False, f"seg_bytes {bad} accepted")
    except RuntimeError:
        pass

# ---- timing (not gated) ----------------------------------------------------------------------------------
print("[T] local timing, 40-call graph (no network): cols vs plain + movedim/reshape copy", flush=True)


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    gg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gg):
        fn()
    return gg


def replay_us(gg, n=200):
    for _ in range(10):
        gg.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n):
            gg.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / n)
    return statistics.median(ts)


rows = []
for m in (1, 2, 4, 6, 8):
    xs = [torch.randn((m, 224), device=dev).to(torch.bfloat16) for _ in range(40)]

    def old_fn(xs=xs, m=m):
        for x in xs:
            o = torch.empty((WORLD, m, 224), dtype=x.dtype, device=dev)
            st.all_gather_into(o, x, True)
            o.movedim(0, 1).reshape(m, WORLD * 224)

    def cols_fn(xs=xs, m=m):
        for x in xs:
            o = torch.empty((m, WORLD * 224), dtype=x.dtype, device=dev)
            st.all_gather_cols_into(o, x, 448, True)

    g_old, g_new = graph_of(old_fn), graph_of(cols_fn)
    t = {"old": [], "cols": []}
    for order in (("old", "cols", "cols", "old"), ("cols", "old", "old", "cols")):
        for k in order:
            t[k].append(replay_us(g_old if k == "old" else g_new))
    a, b = statistics.median(t["old"]), statistics.median(t["cols"])
    rows.append((m, a, b))
    print(f"    M={m}: old {a / 40:6.2f} us/call  cols {b / 40:6.2f} us/call  saves {(a - b) / 40:5.2f} us/call "
          f"= {(a - b):6.1f} us per 40-gather step", flush=True)
    del g_old, g_new

print("RESULT", "PASS" if not FAILS else f"FAIL ({len(FAILS)})", flush=True)
sys.exit(1 if FAILS else 0)
