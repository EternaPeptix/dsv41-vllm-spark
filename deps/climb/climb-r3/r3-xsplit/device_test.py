#!/usr/bin/env python3
"""r3-xsplit device test (deliverable b): XSPLIT vs the live sparse_mla_prefill_mg_dual kernel, bitwise.

Run on ONE STOPPED node (service down, nothing else on the GPU), with the serving venv:

    cd ~/climb-r3/r3-xsplit && timeout 2400 ~/venvs/vllm032/bin/python3 -I device_test.py \
        --lib-dir build --out out/devtest.jsonl            # full, ~10-20 min
    ... --quick                                            # smoke, ~1-2 min

Reference = the live JIT module (flashinfer gen_sparse_mla_sm120_module().build_and_load(): the
cached sparse_mla_sm120.so the service runs), variant MG_DUAL, plus the live kernel instantiated
in the bench .so from the live headers (must agree with the JIT module: C0 at run time).

Phases (each compares output bf16 bits and LSE f32 bits; any mismatch fails the run):
  smoke      normal inputs, a few shapes
  exact      shapes x hostile classes x seeds; output/LSE poisoned (0xA5) and the XSPLIT workspace
             rows poisoned (0xFF = NaN) and its flags poisoned with stale epochs before every call
  lib        the serving library (libdsv41_xsplit.so) on the same inputs as the bench build
  grids      grid sizes 1..64 (dynamic claiming, T*n far above the grid)
  graph      CUDA graphs: >= 256 replays per shape, alternating input sets copied into the static
             buffers between replays, workspace/flags poisoned between replays
  residency  a 1-CTA/SM spinner on a side stream holds 46/47/48 SMs while XSPLIT launches (partial
             residency; must not deadlock), and the WOA l2pf stand-in beside every call
  wrap       epoch at 0xFFFFFFFE: three calls across the 32-bit wrap, flags cleared, outputs exact
  gate       (only with DSV41_ATTN_XSPLIT=1 and the kit installed) the patched flashinfer entry vs
             the raw binding
  timing     median of 200 runs, warm and cold-L2, live vs XSPLIT on production shapes
Exit code 0 only if every compare is bit-identical and nothing errored.
"""
import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np
import torch

H, DQK, DN, DR, RB, SC, BPT = 8, 512, 448, 64, 576, 8, 584
NUM_SCALES, QT = 7, 64
TOPK = 128
CAP_UNITS = 1152
CLS = ["normal", "fp8_nan", "scale_huge", "rope_special", "q_special", "slot0", "struct"]


# ---------------------------------------------------------------------------------------------
# Inputs (port of the g130-attn kit generator, inputs.h, with the extra width as a parameter)
# ---------------------------------------------------------------------------------------------
def f32_to_e4m3_bytes(x):
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch.float8_e4m3fn)
    return t.view(torch.uint8).numpy()


def f32_to_bf16_bits(x):
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.int16).numpy().view(np.uint16)


def gen_pages(rng, pages, pbs):
    page_bytes = pbs * (RB + SC)
    kv = np.zeros((pages, page_bytes), dtype=np.uint8)
    rows = pages * pbs
    sigma = np.exp2(rng.uniform(-2.0, 2.0, size=rows)).astype(np.float32)
    x = (rng.standard_normal((rows, NUM_SCALES, QT)).astype(np.float32) * sigma[:, None, None])
    amax = np.maximum(np.abs(x).max(axis=2), 1e-4).astype(np.float32)
    e = np.ceil(np.log2(amax / 448.0)).astype(np.int32)
    inv = np.ldexp(np.float32(1.0), -e).astype(np.float32)
    q = np.clip(x * inv[:, :, None], -448.0, 448.0)
    fp8 = f32_to_e4m3_bytes(q.reshape(rows, DN)).reshape(rows, DN)
    rope = f32_to_bf16_bits(rng.standard_normal((rows, DR)).astype(np.float32) * sigma[:, None])
    for p in range(pages):
        for r in range(pbs):
            i = p * pbs + r
            kv[p, r * RB:r * RB + DN] = fp8[i]
            kv[p, r * RB + DN:r * RB + RB] = rope[i].view(np.uint8)
            foot = pbs * RB + r * SC
            kv[p, foot:foot + NUM_SCALES] = (e[i] + 127).astype(np.uint8)
            kv[p, foot + 7] = 0
    return kv


def gen_inputs(seed, pbsx, topkx, T, n, cls):
    """Host arrays for one call. n = tiles per token (2 main + n-2 extra) before per-token jitter."""
    rng = np.random.default_rng(seed * 1000003 + pbsx * 7919 + topkx * 31 + T * 131 + n * 7 + cls)
    ub = lambda: int(rng.integers(0, 256))  # noqa: E731
    xpages = topkx // pbsx
    kv = gen_pages(rng, 2, 64)
    kvx = gen_pages(rng, xpages, pbsx)
    q = f32_to_bf16_bits(rng.standard_normal((T, H, DQK)).astype(np.float32))
    sink = (rng.standard_normal(H) * 2.0).astype(np.float32)
    has_sink = not (cls == 6 and (ub() & 1))
    main_rows = min(n, 2) * 64
    extra_rows = max(n - 2, 0) * 64
    idx = np.full((T, TOPK), -1, dtype=np.int32)
    idxx = np.full((T, topkx), -1, dtype=np.int32)
    ln = np.zeros(T, dtype=np.int32)
    lnx = np.zeros(T, dtype=np.int32)
    for t in range(T):
        m, x = main_rows, extra_rows
        if cls == 6:
            e = ub()
            if e < 24:
                m = 0
            elif e < 96:
                m = ub() % (TOPK + 1)
            f = ub()
            if f < 24:
                x = 0
            elif f < 120:
                x = (ub() * 2) % (topkx + 1)
        elif t > 0 and x >= 64:
            x -= t % 5
        ln[t], lnx[t] = m, x
        for i in range(m):
            v = i
            if cls in (5, 6):
                e = ub()
                if e < 8:
                    v = -1
                elif e < 16 and cls == 6:
                    v = ub() % 128
            idx[t, i] = v
        for i in range(x):
            v = i
            if cls in (5, 6):
                e = ub()
                if e < 8:
                    v = -1
                elif e < 16 and cls == 6:
                    v = (ub() * 2) % topkx
            idxx[t, i] = v
    if cls in (1, 2, 3):
        xmin = extra_rows - 4
        extra = xmin > 1 and (ub() & 1)
        span = (xmin if extra else main_rows) - 1
        if span >= 1:
            p = 1 + int(rng.integers(0, span))
            reader = int(rng.integers(0, T))
            ix = idxx if extra else idx
            for t in range(T):
                if t != reader:
                    ix[t, p] = -1
            pbs = pbsx if extra else 64
            buf = kvx if extra else kv
            page, r = p // pbs, p % pbs
            row = r * RB
            foot = pbs * RB + r * SC
            if cls == 1:
                for _ in range(1 + ub() % 8):
                    buf[page, row + int(rng.integers(0, DN))] = 0x7F if (ub() & 1) else 0xFF
            elif cls == 2:
                buf[page, foot + ub() % NUM_SCALES] = 254 if (ub() & 1) else 255
            else:
                sp = [0x7F80, 0xFF80, 0x7FC0, 0x0001, 0x8001, 0x8000, 0x0000]
                rope = buf[page, row + DN:row + RB].view(np.uint16)
                d0 = ub() % DR
                rope[d0] = sp[ub() % 3]
                for _ in range(ub() % 6):
                    d = ub() % DR
                    if d != d0:
                        rope[d] = sp[ub() % 7]
    elif cls == 4:
        sp = [0x7F80, 0xFF80, 0x7FC0, 0x0001]
        t0, h0 = int(rng.integers(0, T)), ub() % H
        d0 = int(rng.integers(0, DQK))
        q[t0, h0, d0] = sp[ub() % 3]
        for _ in range(ub() % 4):
            d = int(rng.integers(0, DQK))
            if d != d0:
                q[t0, h0, d] = sp[ub() % 4]
    elif cls == 5:
        idx[:, 0] = -1
        idxx[:, 0] = -1
        kv[0, DN:RB].view(np.uint16)[:] = 0x7FC0
        kvx[0, DN:RB].view(np.uint16)[:] = 0x7FC0
    elif cls == 6:
        if ub() < 64:
            zh, sh = ub() % H, ub() % H
            q[:, zh, :] = 0
            q[:, sh, :] = 0x8000
        r = ub() % 64
        kv[1, 64 * RB + r * SC + ub() % NUM_SCALES] = 0
    return dict(kv=kv.reshape(2, 64, BPT), kvx=kvx.reshape(xpages, pbsx, BPT), q=q, sink=sink,
                has_sink=has_sink, idx=idx, idxx=idxx, len=ln, lenx=lnx, pbsx=pbsx, topkx=topkx,
                T=T, cls=cls)


def to_dev(h, dev):
    d = {
        "kv": torch.from_numpy(h["kv"]).to(dev),
        "kvx": torch.from_numpy(h["kvx"]).to(dev),
        "q": torch.from_numpy(h["q"].view(np.int16)).to(dev).view(torch.bfloat16),
        "sink": torch.from_numpy(h["sink"]).to(dev) if h["has_sink"] else None,
        "idx": torch.from_numpy(h["idx"]).to(dev),
        "idxx": torch.from_numpy(h["idxx"]).to(dev),
        "len": torch.from_numpy(h["len"]).to(dev),
        "lenx": torch.from_numpy(h["lenx"]).to(dev),
    }
    d.update(pbsx=h["pbsx"], topkx=h["topkx"], T=h["T"], cls=h["cls"])
    return d


def copy_into(dst, src):
    """Copy a same-shaped input set into static buffers (graph replays)."""
    for k in ("kv", "kvx", "q", "idx", "idxx", "len", "lenx"):
        dst[k].copy_(src[k])
    if dst["sink"] is not None:
        dst["sink"].copy_(src["sink"] if src["sink"] is not None else torch.zeros_like(dst["sink"]))


# ---------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------
class Lib:
    def __init__(self, path):
        self.path = path
        self.l = ctypes.CDLL(path)
        vp, ull, ci = ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_int
        self.l.dsv41_xsplit_launch.restype = ci
        self.l.dsv41_xsplit_launch.argtypes = [ci, vp, vp, vp, vp, vp, vp, vp, vp, ctypes.c_float,
                                               ci, ci, ci, ull, ull, ull, vp, vp, vp, ci, ci, vp]
        self.l.dsv41_xsplit_ws_bytes.restype = ull
        self.l.dsv41_xsplit_ws_bytes.argtypes = [ci]
        self.l.dsv41_xsplit_ws_offset.restype = ull
        self.l.dsv41_xsplit_ws_offset.argtypes = [ci, ci]
        self.bench = hasattr(self.l, "dsv41_live_mg_dual_launch")
        if self.bench:
            self.l.dsv41_live_mg_dual_launch.restype = ci
            self.l.dsv41_live_mg_dual_launch.argtypes = [ci, vp, vp, vp, vp, vp, vp, vp, vp,
                                                         ctypes.c_float, ci, ci, ci, ull, ull, ull,
                                                         vp, vp, vp]
            self.l.dsv41_bench_spin.restype = ci
            self.l.dsv41_bench_spin.argtypes = [ci, ctypes.c_longlong, vp]
            self.l.dsv41_bench_l2pf.restype = ci
            self.l.dsv41_bench_l2pf.argtypes = [vp, ci, ctypes.c_longlong, ci, vp]
        self.row_words = self.l.dsv41_xsplit_row_words()
        self.arm = self.l.dsv41_xsplit_arm()

    def ws_new(self, dev, cap=CAP_UNITS):
        ws = torch.zeros(int(self.l.dsv41_xsplit_ws_bytes(cap)), dtype=torch.uint8, device=dev)
        off = [int(self.l.dsv41_xsplit_ws_offset(cap, i)) for i in range(4)]
        return dict(buf=ws, cap=cap, off=off)

    def xsplit(self, d, out, lse, ws, grid=46, stream=None, sm_scale=None):
        vp = ctypes.c_void_p
        T = d["T"]
        nmax = TOPK // 64 + (d["topkx"] + 63) // 64
        g = min(grid, T * nmax + T)
        s = stream if stream is not None else torch.cuda.current_stream()
        rc = self.l.dsv41_xsplit_launch(
            d["pbsx"], vp(d["q"].data_ptr()), vp(d["kv"].data_ptr()), vp(d["idx"].data_ptr()),
            vp(d["kvx"].data_ptr()), vp(d["idxx"].data_ptr()),
            vp(d["sink"].data_ptr() if d["sink"] is not None else 0), vp(out.data_ptr()),
            vp(lse.data_ptr()), ctypes.c_float(SM_SCALE if sm_scale is None else sm_scale), T,
            TOPK, d["topkx"], d["kv"].stride(0), d["kvx"].stride(0), lse.stride(0),
            vp(d["len"].data_ptr()), vp(d["lenx"].data_ptr()), vp(ws["buf"].data_ptr()),
            ws["cap"], g, vp(s.cuda_stream))
        if rc != 0:
            raise RuntimeError(f"xsplit launch rc={rc}")

    def live_bench(self, d, out, lse, stream=None):
        vp = ctypes.c_void_p
        s = stream if stream is not None else torch.cuda.current_stream()
        rc = self.l.dsv41_live_mg_dual_launch(
            d["pbsx"], vp(d["q"].data_ptr()), vp(d["kv"].data_ptr()), vp(d["idx"].data_ptr()),
            vp(d["kvx"].data_ptr()), vp(d["idxx"].data_ptr()),
            vp(d["sink"].data_ptr() if d["sink"] is not None else 0), vp(out.data_ptr()),
            vp(lse.data_ptr()), ctypes.c_float(SM_SCALE), d["T"], TOPK, d["topkx"],
            d["kv"].stride(0), d["kvx"].stride(0), lse.stride(0), vp(d["len"].data_ptr()),
            vp(d["lenx"].data_ptr()), vp(s.cuda_stream))
        if rc != 0:
            raise RuntimeError(f"live(bench) launch rc={rc}")


SM_SCALE = float(np.float32(1.0 / np.sqrt(512.0)))
_JIT = None


def live_jit(d, out, lse):
    global _JIT
    if _JIT is None:
        from flashinfer.jit.mla import gen_sparse_mla_sm120_module
        _JIT = gen_sparse_mla_sm120_module().build_and_load()
    _JIT.sparse_mla_sm120_paged_attention(d["q"], d["kv"], d["idx"], out, lse, SM_SCALE, 1, 3,
                                          d["len"], d["sink"], d["kvx"], d["idxx"], d["lenx"])


def new_out(T, dev):
    o = torch.empty((T, H, 512), dtype=torch.bfloat16, device=dev)
    lse_wide = torch.empty((T, H + 8), dtype=torch.float32, device=dev)  # strided LSE view
    lse = lse_wide[:, 3:3 + H]
    poison_out(o, lse)
    return o, lse


def poison_out(o, lse):
    o.view(torch.int16).fill_(-23131)  # 0xA5A5
    lse.view(torch.int32).fill_(-1515870811)  # 0xA5A5A5A5


def poison_ws(ws, rng):
    """Rows -> NaN bytes, flags -> stale epochs (current gen = previous call's tag) or random tags
    that are never the next epoch."""
    b = ws["buf"]
    gen = int(b[:4].view(torch.int32).item()) & 0xFFFFFFFF
    nxt = (gen + 1) & 0xFFFFFFFF
    o_tm, o_done, o_rows = ws["off"][1], ws["off"][2], ws["off"][3]
    b[o_rows:].fill_(0xFF)
    cap = ws["cap"]
    tm = b[o_tm:o_tm + cap * 16 * 8].view(torch.int64)
    dn = b[o_done:o_done + cap * 4].view(torch.int32)
    mode = int(rng.integers(0, 3))
    if mode == 0:
        tag = gen
    else:
        tag = int(rng.integers(0, 2**32))
        if tag == nxt:
            tag ^= 1
    lo = torch.randint(0, 2**31, (cap * 16,), dtype=torch.int64, device=b.device)
    tm.copy_(((tag << 32) | lo) if tag < 2**31 else (((tag - 2**32) << 32) | lo))
    dn.fill_(tag if tag < 2**31 else tag - 2**32)


def compare(o1, l1, o2, l2):
    a, b = o1.view(torch.int16), o2.view(torch.int16)
    if not torch.equal(a, b):
        i = int((a != b).flatten().nonzero()[0])
        return f"output mismatch at flat {i} (token {i // (H * 512)}, head {(i // 512) % H}, d {i % 512})"
    a, b = l1.contiguous().view(torch.int32), l2.contiguous().view(torch.int32)
    if not torch.equal(a, b):
        i = int((a != b).flatten().nonzero()[0])
        return f"lse mismatch at flat {i}"
    return None


def nonfinite_share(o):
    w = o.view(torch.int16).to(torch.int32) & 0x7F80
    return float((w == 0x7F80).float().mean())


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib-dir", default="build")
    ap.add_argument("--out", default="out/devtest.jsonl")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fo = open(args.out, "w")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    rng = np.random.default_rng(args.seed)
    bench = Lib(os.path.join(args.lib_dir, "libdsv41_xsplit_bench_a0r1.so"))
    serving = Lib(os.path.join(args.lib_dir, "libdsv41_xsplit.so"))
    assert bench.arm == 0 and serving.arm == 0
    ws = bench.ws_new(dev)
    ws_s = serving.ws_new(dev)
    fails, runs = [], 0
    t_start = time.time()

    def rec(**kw):
        fo.write(json.dumps(kw) + "\n")
        fo.flush()

    def check_one(d, phase, grid=46, lib=None, wsx=None, co=None):
        nonlocal runs
        lib = lib or bench
        wsx = wsx or ws
        o_ref, l_ref = new_out(d["T"], dev)
        o_b, l_b = new_out(d["T"], dev)
        o_x, l_x = new_out(d["T"], dev)
        live_jit(d, o_ref, l_ref)
        bench.live_bench(d, o_b, l_b)
        poison_ws(wsx, rng)
        if co is not None:
            co()
        lib.xsplit(d, o_x, l_x, wsx, grid=grid)
        torch.cuda.synchronize()
        runs += 1
        e0 = compare(o_ref, l_ref, o_b, l_b)
        e1 = compare(o_ref, l_ref, o_x, l_x)
        r = dict(phase=phase, pbsx=d["pbsx"], topkx=d["topkx"], T=d["T"], cls=CLS[d["cls"]],
                 grid=grid, live_c0=e0 or "ok", xsplit=e1 or "ok",
                 nonfinite=round(nonfinite_share(o_ref), 4))
        rec(**r)
        if e0 or e1:
            fails.append(r)
            print("MISMATCH", r, flush=True)

    shapes_quick = [(64, 512, 1, 10), (64, 512, 4, 10), (32, 1024, 2, 18), (32, 512, 6, 10)]
    shapes_full = []
    for pbsx, topkx in ((64, 512), (32, 512), (32, 1024)):
        nmax = 2 + topkx // 64
        for T in (1, 2, 3, 4, 5, 6, 8, 16, 33, 64):
            for n in sorted({0, 1, 2, 3, 5, nmax // 2, nmax - 1, nmax}):
                if T * nmax <= CAP_UNITS:
                    shapes_full.append((pbsx, topkx, T, n))

    # smoke ------------------------------------------------------------------------------
    for (pbsx, topkx, T, n) in shapes_quick:
        check_one(to_dev(gen_inputs(args.seed, pbsx, topkx, T, n, 0), dev), "smoke")
    print(f"smoke done runs={runs} fails={len(fails)}", flush=True)
    if fails:
        return finish(fo, fails, runs, t_start)

    # exact ------------------------------------------------------------------------------
    shapes = shapes_quick if args.quick else shapes_full
    seeds = 1 if args.quick else 2
    for (pbsx, topkx, T, n) in shapes:
        for cls in range(7):
            for s in range(seeds):
                check_one(to_dev(gen_inputs(args.seed * 100 + s, pbsx, topkx, T, n, cls), dev),
                          "exact")
    print(f"exact done runs={runs} fails={len(fails)}", flush=True)

    # serving library ----------------------------------------------------------------------
    for (pbsx, topkx, T, n) in shapes_quick:
        for cls in (0, 6):
            check_one(to_dev(gen_inputs(args.seed + 7, pbsx, topkx, T, n, cls), dev), "lib",
                      lib=serving, wsx=ws_s)

    # exact bench variants (rope-B re-read from the KV cache; B replay without row prefetch) ------
    for name in ("a0r0", "a0r1p0"):
        path = os.path.join(args.lib_dir, f"libdsv41_xsplit_bench_{name}.so")
        if not os.path.exists(path):
            continue
        var = Lib(path)
        wsv = var.ws_new(dev)
        for (pbsx, topkx, T, n) in shapes_quick:
            for cls in (0, 3, 5, 6):
                check_one(to_dev(gen_inputs(args.seed + 11, pbsx, topkx, T, n, cls), dev),
                          f"variant_{name}", lib=var, wsx=wsv)
    print(f"variants done runs={runs} fails={len(fails)}", flush=True)

    # grids ----------------------------------------------------------------------------
    for grid in (1, 2, 3, 7, 23, 46, 48, 64):
        for (pbsx, topkx, T, n) in ((32, 1024, 6, 18), (64, 512, 64, 10), (32, 512, 1, 10)):
            check_one(to_dev(gen_inputs(args.seed + grid, pbsx, topkx, T, n, 6 if grid & 1 else 0),
                             dev), "grids", grid=grid)
    print(f"grids done runs={runs} fails={len(fails)}", flush=True)

    # residency ------------------------------------------------------------------------
    side = torch.cuda.Stream()
    seg_buf = torch.empty(5 * 865 * 1024, dtype=torch.uint8, device=dev)
    segs = torch.tensor([[seg_buf.data_ptr() + i * 865 * 1024, 865 * 1024] for i in range(5)],
                        dtype=torch.int64, device=dev)
    for k in (46, 47, 48):
        def co(k=k):
            with torch.cuda.stream(side):
                rc = bench.l.dsv41_bench_spin(k, 300_000, ctypes.c_void_p(side.cuda_stream))
                assert rc == 0
            time.sleep(0.0005)

        for (pbsx, topkx, T, n) in ((32, 1024, 6, 18), (64, 512, 4, 10)):
            check_one(to_dev(gen_inputs(args.seed + k, pbsx, topkx, T, n, 0), dev),
                      f"residency{k}", co=co)

    def l2pf():
        rc = bench.l.dsv41_bench_l2pf(ctypes.c_void_p(segs.data_ptr()), 5, 16384, 2,
                                      ctypes.c_void_p(side.cuda_stream))
        assert rc == 0

    for (pbsx, topkx, T, n) in shapes_quick:
        check_one(to_dev(gen_inputs(args.seed + 3, pbsx, topkx, T, n, 0), dev), "l2pf", co=l2pf)
    torch.cuda.synchronize()
    print(f"residency done runs={runs} fails={len(fails)}", flush=True)

    # wrap -----------------------------------------------------------------------------
    ws["buf"][:4].view(torch.int32).fill_(-2)  # gen = 0xFFFFFFFE -> E = 0xFFFFFFFF
    for i in range(3):
        d = to_dev(gen_inputs(args.seed + 50 + i, 32, 1024, 4, 18, 0), dev)
        o_ref, l_ref = new_out(4, dev)
        o_x, l_x = new_out(4, dev)
        live_jit(d, o_ref, l_ref)
        bench.xsplit(d, o_x, l_x, ws)
        torch.cuda.synchronize()
        gen = int(ws["buf"][:4].view(torch.int32).item()) & 0xFFFFFFFF
        e = compare(o_ref, l_ref, o_x, l_x)
        rec(phase="wrap", call=i, gen_after=gen, xsplit=e or "ok")
        runs += 1
        if e or (i == 0 and gen != 0):
            fails.append(dict(phase="wrap", call=i, gen_after=gen, err=e))
    print(f"wrap done runs={runs} fails={len(fails)}", flush=True)

    # graph ----------------------------------------------------------------------------
    replays = 32 if args.quick else 256
    for (pbsx, topkx, T, n) in ((64, 512, 4, 10), (32, 1024, 6, 18), (32, 512, 2, 7),
                                (64, 512, 64, 10)):
        sets = [to_dev(gen_inputs(args.seed + 200 + j, pbsx, topkx, T, n, j % 7), dev)
                for j in range(4)]
        # static buffers shaped like set 0; sink present in all (zeros when a set has none)
        st = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sets[0].items()}
        if st["sink"] is None:
            st["sink"] = torch.zeros(H, dtype=torch.float32, device=dev)
        o_x, l_x = new_out(T, dev)
        o_ref, l_ref = new_out(T, dev)
        wsg = bench.ws_new(dev)
        # warm (eager) once, then capture XSPLIT alone and the live JIT call alone
        bench.xsplit(st, o_x, l_x, wsg)
        live_jit(st, o_ref, l_ref)
        torch.cuda.synchronize()
        gx, gl = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(gx):
            bench.xsplit(st, o_x, l_x, wsg)
        with torch.cuda.graph(gl):
            live_jit(st, o_ref, l_ref)
        bad = 0
        for r in range(replays):
            src = dict(sets[r % 4])
            if src["sink"] is None:
                src["sink"] = torch.zeros(H, dtype=torch.float32, device=dev)
            copy_into(st, src)
            poison_out(o_x, l_x)
            poison_out(o_ref, l_ref)
            if r % 2:
                poison_ws(wsg, rng)
            gl.replay()
            gx.replay()
            torch.cuda.synchronize()
            e = compare(o_ref, l_ref, o_x, l_x)
            runs += 1
            if e:
                bad += 1
                if bad <= 3:
                    fails.append(dict(phase="graph", pbsx=pbsx, T=T, n=n, replay=r, err=e))
        rec(phase="graph", pbsx=pbsx, topkx=topkx, T=T, n=n, replays=replays, mismatches=bad)
        del gx, gl
    print(f"graph done runs={runs} fails={len(fails)}", flush=True)

    # gate (patched flashinfer, flag on) -------------------------------------------------
    if os.environ.get("DSV41_ATTN_XSPLIT") == "1":
        try:
            from flashinfer.mla import _sparse_mla_sm120 as smod
            from flashinfer.mla import dsv41_xsplit as gate
            if not getattr(smod, "_DSV41_ATTN_XSPLIT", False):
                raise RuntimeError("flashinfer not patched or flag not read")
            pa = smod.get_sparse_mla_sm120_module().paged_attention
            for (pbsx, topkx, T, n) in shapes_quick:
                d = to_dev(gen_inputs(args.seed + 300, pbsx, topkx, T, n, 0), dev)
                o_ref, l_ref = new_out(T, dev)
                o_g, l_g = new_out(T, dev)
                live_jit(d, o_ref, l_ref)
                pa(d["q"], d["kv"], d["idx"], o_g, l_g, SM_SCALE, 512, 1, 0, d["len"], d["sink"],
                   d["kvx"], d["idxx"], d["lenx"], None, None)
                torch.cuda.synchronize()
                e = compare(o_ref, l_ref, o_g, l_g)
                runs += 1
                rec(phase="gate", pbsx=pbsx, T=T, n=n, xsplit=e or "ok", stats=gate.stats())
                if e:
                    fails.append(dict(phase="gate", T=T, err=e))
            st_g = gate.stats()
            print(f"gate done stats={st_g}", flush=True)
            if st_g["routed"] == 0:
                # The planner's calibrated crossover sent these shapes to the decode kernel, or
                # the envelope refused them: the gate path was not exercised. Not a mismatch,
                # but the window must not read this phase as evidence.
                rec(phase="gate", note="NOT EXERCISED (routed == 0)", stats=st_g)
                print("gate: NOT EXERCISED (routed == 0)", flush=True)
        except Exception as ex:  # noqa: BLE001
            fails.append(dict(phase="gate", err=repr(ex)))

    # timing ---------------------------------------------------------------------------
    if not args.skip_timing and not fails:
        thrash = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
        for (pbsx, topkx, T, n) in ((64, 512, 2, 10), (64, 512, 4, 10), (64, 512, 6, 10),
                                    (32, 1024, 2, 13), (32, 1024, 4, 16), (32, 1024, 6, 18)):
            d = to_dev(gen_inputs(args.seed + 400, pbsx, topkx, T, n, 0), dev)
            o, l = new_out(T, dev)
            res = {}
            for name, fn in (("live", lambda: bench.live_bench(d, o, l)),
                             ("xsplit", lambda: bench.xsplit(d, o, l, ws))):
                for cold in (False, True):
                    ts = []
                    for i in range(220):
                        if cold:
                            thrash.add_(1)
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                        fn()
                        b.record()
                        b.synchronize()
                        if i >= 20:
                            ts.append(a.elapsed_time(b) * 1000.0)
                    res[f"{name}_{'cold' if cold else 'warm'}_us"] = round(float(np.median(ts)), 2)
            rec(phase="timing", pbsx=pbsx, topkx=topkx, T=T, n=n, **res)
            print("timing", pbsx, topkx, T, n, res, flush=True)
    return finish(fo, fails, runs, t_start)


def finish(fo, fails, runs, t0):
    fo.close()
    print(f"RESULT runs={runs} mismatches/errors={len(fails)} elapsed={time.time() - t0:.0f}s",
          flush=True)
    for f in fails[:20]:
        print("  FAIL", f)
    print("DEVICE TEST", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
