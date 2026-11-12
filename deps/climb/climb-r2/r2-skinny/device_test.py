#!/usr/bin/env python3
"""r2-skinny device test: live skinny_mx_ext.so (v1) vs skinny_mx_v2_ext.so, bitwise, plus timing.

RUN ONLY ON A STOPPED NODE (no serving process on the GPU), e.g. spark-eight inside a window:
    cd ~/climb-scratch/r2-skinny && nice -n 5 ~/venvs/vllm032/bin/python device_test.py \
        [--v2-lib prebuilt (default)] [--model ~/models/dsv41/official-fp8 --layers 0,13,39 --rank 7] \
        [--acts recorded_acts.pt] [--quick] [--no-timing] [--json out.json] [--allow-any-so]
Expected runtime: ~2-4 min full (exhaustive quant ~17M elements x 2 FTZ modes, 8 M x 3 weight sets x 5 inputs x
grids/pdl, graph-timed medians of 200 replays warm + cold), ~40 s with --quick.

Exit code 0 only when every bitwise case is identical (the forced-mismatch control must MISMATCH) and the loaded
skinny_mx_v2_ext.so has the sha256 recorded in KIT.json (the binary install.py ships); --allow-any-so skips that
check (development only: such a run does not clear the kit for a window).

Reference gate (r4, 2026-11-08; before any section, exit 2 if it fails): the live quant this test compares against
  must be the op serving runs: vllm has_flashinfer() True, mxfp8_e4m3_quantize launches FlashInfer's
  MXFP8QuantizeSwizzled kernel (torch.profiler) and returns FlashInfer's bytes on a discriminating block (2^-120
  values: FlashInfer codes 0, torch fallback codes 0x70). The 18:00Z run of 2026-11-08 compared against the torch
  fallback (non-interactive ssh: no ninja/nvcc/CUDA_HOME) and failed Q falsely. Run it in the serving environment:
  source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda

Sections
  Q  fused quant (QF prologue device code via quant_debug) vs the live quant (vllm mxfp8_e4m3_quantize ->
     FlashInfer CuTe-DSL MXFP8QuantizeSwizzled): e4m3 bytes and unswizzled ue8m0 bytes, exhaustive over every bf16
     value x every reachable scale byte, every positive bf16 as block amax, NaN/Inf/zero blocks, random rows; both
     FTZ modes of amax * (1/448). The live reference is run in chunks of <= 384 rows so that it is FlashInfer's
     4T/SF variant (M * 160 < 65536), the one serving (M <= 8) and the self-check (256 rows) use. Reports which mode(s) match; the mode the serving self-check would pick
     (dsv41_r2_skinny._pick_ftz on hostile rows) must be one that matches exhaustively. Also deq bytes vs v1's
     A8 prologue formula (bf16_rn(float(e4m3) * 2^(e-127))) recomputed in torch. r4: exactly ONE mode must match
     and it must be --expect-ftz (default 0: the live 4T/SF cubin multiplies amax by 0x3B124925 with a non-FTZ
     FMUL, SASS read 2026-11-08); per mode the counts of differing code / scale / deq bytes and the first differing
     (row, block) with its input amax are printed and kept in the json.
  G  GEMV: for M = 1..8 (target verify widths 1..6, DSpark M = 5, plus 7, 8), weights {random, hostile, real
     checkpoint shards if --model}, activations {random, post-norm-like, hostile, zeros, recorded if --acts,
     ckpt-derived if --model: rmsnorm(embed[token]) * layers.L.attn_norm for real token ids, i.e. the qkv_a input
     distribution of layer L before any residual mixing}:
       ref = v1.skinny_mx_a8(*live_quant(x))
       v2.skinny_v2_a8(live_quant(x)) at grid 0 (= 48) and sweep grids, pdl False/True
       v2.skinny_v2_q(x) (fused quant) at grid 0, pdl False/True, picked FTZ mode
     bitwise on raw bytes; first mismatching (m, n) reported.
     r4 poison checks per (M, weights, input): v2 a8 with the swizzled-scale padding rows m >= M (and pad
     columns) set to 0xC3 must equal ref (v1 under the same poison is reported, informational); the fused entry
     on a row view of a larger buffer whose rows >= M are 0xFFFF (bf16 NaN) must equal ref.
  C  CUDA graphs: capture [quant -> v1] and [quant -> v2 pdl] and [producer -> v2_q pdl / v2_q] chains, replay,
     compare bitwise with eager v1 (PDL edges inside graphs).
  X  forced-mismatch control: one flipped weight bit for v2 -> must MISMATCH.
  T  timing (median of >= 200 graph replays, 20 calls per graph): v1 chain, v2 a8 chain, v2 a8 pdl chain,
     fused, fused pdl; warm (same weights) and cold-L2 (weights rotated over >= 96 MB of copies).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
N_SHARD, K = 224, 5120
FAILS: list[str] = []
RESULTS: dict = {"bitwise": [], "timing": [], "quant": {}}


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    RESULTS["bitwise"].append({"name": name, "ok": bool(ok), "detail": detail})
    if not ok:
        FAILS.append(name)


def load_so(d, stem):
    cands = sorted(glob.glob(os.path.join(os.path.expanduser(d), f"{stem}*.so")))
    if not cands:
        sys.exit(f"no {stem}*.so in {d}")
    spec = importlib.util.spec_from_file_location(stem, cands[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"loaded {cands[0]}")
    return mod


def first_diff(a, b):
    a16, b16 = a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
    ne = (a16 != b16).nonzero()
    if ne.numel() == 0:
        return "identical"
    i = tuple(ne[0].tolist())
    return f"{ne.shape[0]} differ, first at {i}: {a16[i].item() & 0xffff:#06x} vs {b16[i].item() & 0xffff:#06x}"


def same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


# ------------------------------------------------------------------ inputs
def unswizzle(sf1d, M, Kk):
    nm, nk = (M + 127) // 128, (Kk // 32 + 3) // 4
    t = sf1d.reshape(nm, nk, 32, 4, 4).transpose(1, 3).contiguous().view(nm * 128, nk * 4)
    return t[:M, : Kk // 32].contiguous()


def bf16_from_bits(bits):
    return bits.to(torch.int32).to(torch.int16).view(torch.bfloat16)


def rand_weights(seed, hostile, dev):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if hostile:  # every e4m3 code incl. +-0, subnormals, +-448 (NaN codes 0x7f/0xff excluded); wide scale spread
        codes = torch.randint(0, 256, (N_SHARD, K), generator=g, dtype=torch.int32)
        codes = torch.where((codes & 0x7F) == 0x7F, codes - 1, codes).to(torch.uint8)
        sc = torch.randint(100, 140, (N_SHARD, K // 32), generator=g, dtype=torch.int32).to(torch.uint8)
        sc[:, ::17] = 127
    else:
        w = (torch.randn(N_SHARD, K, generator=g) * 2).clamp(-448, 448).to(torch.float8_e4m3fn)
        codes = w.view(torch.uint8)
        sc = torch.randint(118, 124, (N_SHARD, K // 32), generator=g, dtype=torch.int32).to(torch.uint8)
    return codes.to(dev).contiguous(), sc.to(dev).contiguous()


def real_weights(model_dir, layer, rank, dev):
    """rows [224 rank, 224 rank + 224) of cat(wq_a, wkv) of one checkpoint layer (fused_wqa_wkv order)."""
    from safetensors import safe_open

    model_dir = os.path.expanduser(model_dir)
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    pre = f"layers.{layer}.attn." if not str(layer).startswith("mtp") else f"{layer}.attn."
    ws, ss = [], []
    for name in ("wq_a", "wkv"):
        for kind, out in (("weight", ws), ("scale", ss)):
            key = pre + f"{name}.{kind}"
            with safe_open(os.path.join(model_dir, idx[key]), "pt") as f:
                out.append(f.get_tensor(key))
    def rowmajor(wt, st):  # checkpoint e8m0 blocks (official-fp8: [N/32, K/32] = 32x32 blocks) -> [N, K/32]
        st = st.view(torch.uint8) if st.element_size() == 1 else st
        n = wt.shape[0]
        if st.shape == (n, K // 32):
            return st
        for rb, cb in ((32, 1), (128, 4), (1, 1)):
            if st.shape == (-(-n // rb), K // (32 * cb)):
                return st.repeat_interleave(rb, 0)[:n].repeat_interleave(cb, 1)
        raise RuntimeError(f"unrecognised scale shape {tuple(st.shape)} for weight {tuple(wt.shape)}")

    w = torch.cat(ws, 0)
    s2 = torch.cat([rowmajor(wt, st) for wt, st in zip(ws, ss)], 0)
    rows = slice(rank * N_SHARD, (rank + 1) * N_SHARD)
    return w.view(torch.uint8)[rows].contiguous().to(dev), s2[rows].contiguous().to(dev)


def ckpt_acts(model_dir, layer, n=64):
    """[n, K] bf16 production-shaped qkv_a inputs: rmsnorm(embed[tok]) * attn_norm.weight of `layer` (fp32 math,
    eps 1e-6, one bf16 rounding), for a fixed spread of token ids. Real weights and norms, not a serving record."""
    from safetensors import safe_open

    model_dir = os.path.expanduser(model_dir)
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]

    def get(key):
        with safe_open(os.path.join(model_dir, idx[key]), "pt") as f:
            return f.get_tensor(key)

    emb = get("embed.weight")
    pre = f"layers.{layer}." if not str(layer).startswith("mtp") else f"{layer}."
    nw = get(pre + "attn_norm.weight").float()
    toks = torch.linspace(0, emb.shape[0] - 1, n).long()
    toks[: min(n, 16)] = torch.arange(min(n, 16)) * 7 + 100  # some low (frequent) ids too
    h = emb[toks].float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * nw
    return h.to(torch.bfloat16)


def act_sets(M, seed, dev, r2mod, recorded):
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = [("random", (torch.randn(M, K, generator=g) * 0.05).to(torch.bfloat16))]
    x = torch.randn(M, K, generator=g)
    blk = torch.exp2(torch.randint(-6, 7, (M, K // 32), generator=g).float())
    x = x * blk.repeat_interleave(32, dim=1)
    x[:, ::97] = 0.0
    x[torch.arange(M), torch.randint(0, K, (M,), generator=g)] = 300.0
    out.append(("postnorm", x.to(torch.bfloat16)))
    out.append(("hostile", r2mod.hostile_rows(M, K, 9100 + seed, "cpu")))
    out.append(("zeros", torch.zeros(M, K, dtype=torch.bfloat16)))
    for j, r in enumerate(recorded or []):
        name, r = r if isinstance(r, tuple) else (f"recorded{j}", r)
        r = r.reshape(-1, K)
        if r.shape[0] >= M:
            o = (seed * 7) % (r.shape[0] - M + 1)  # a different row window per (M, weight set)
            out.append((name, r[o:o + M].to(torch.bfloat16)))
    return [(n, t.to(dev).contiguous()) for n, t in out]


# ------------------------------------------------------------------ reference gate (r4)
def assert_live_reference(live_quant, dev):
    """Exit 2 unless live_quant is the serving op: FlashInfer MXFP8QuantizeSwizzled with FlashInfer bytes."""
    from vllm.utils.flashinfer import has_flashinfer
    from torch.profiler import ProfilerActivity, profile

    print(f"reference env: VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV')} CUDA_HOME={os.environ.get('CUDA_HOME')} "
          f"PATH={os.environ.get('PATH')}", flush=True)
    why = []
    hf = bool(has_flashinfer())
    if not hf:
        why.append("has_flashinfer() is False (vLLM uses _mxfp8_e4m3_quantize_torch)")
    x = torch.randn(4, K, generator=torch.Generator().manual_seed(5)) * 0.5
    x[0, :32] = 2.0 ** -120
    x = x.to(torch.bfloat16).to(dev)
    live_quant(x, is_sf_swizzled_layout=True)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        q, s = live_quant(x, is_sf_swizzled_layout=True)
        torch.cuda.synchronize()
    names = {e.key for e in prof.key_averages()}
    hit = sorted(n for n in names if "MXFP8QuantizeSwizzled" in n)
    if not hit:
        why.append(f"no MXFP8QuantizeSwizzled kernel launched (saw {sorted(n[:50] for n in names)[:5]})")
    s0 = int(s.view(torch.uint8).reshape(-1)[0].item())
    c0 = int(q.view(torch.uint8)[0, :32].max().item())
    if s0 != 0 or c0 != 0:
        why.append(f"2^-120 block: scale {s0} codes max {c0:#04x} (FlashInfer: 0 / 0x00; torch fallback: 0 / 0x70)")
    RESULTS["reference"] = {"has_flashinfer": hf, "kernel": hit[0] if hit else None, "disc_scale": s0,
                            "disc_codes_max": c0, "ok": not why}
    if why:
        print("REFERENCE IS NOT THE SERVING OP -> no comparison is valid; exit 2:\n  " + "\n  ".join(why), flush=True)
        if RESULTS.get("_json"):
            json.dump({k: v for k, v in RESULTS.items() if k != "_json"}, open(RESULTS["_json"], "w"), indent=1)
        sys.exit(2)
    print(f"[PASS] reference is the serving op: has_flashinfer() True, kernel {hit[0][:90]}..., "
          f"discriminator block scale 0 codes 0x00", flush=True)


def diff_report(name, got, ref, R, nb_per_row, x_blocks):
    """count of differing bytes and the first differing (row, block) for [R, K] codes or [R, K/32] scales."""
    g, r = got.reshape(R, -1), ref.reshape(R, -1)
    ne = (g != r)
    n = int(ne.sum())
    if n == 0:
        return {"n": 0}, f"{name} 0 differ"
    idx = ne.nonzero()[0].tolist()
    row, col = idx
    blk = col if g.shape[1] == nb_per_row else col // 32
    amax_bits = int((x_blocks[row, blk].contiguous().view(torch.int16).to(torch.int32) & 0x7FFF).max().item()) \
        if x_blocks is not None else -1  # bf16 |x| bits: integer order == magnitude order (NaN > Inf)
    d = {"n": n, "row": row, "block": blk, "got": int(g[row, col]), "ref": int(r[row, col]),
         "block_absmax_bf16_bits": amax_bits}
    return d, f"{name} {n} differ (first row {row} block {blk}: got {d['got']:#x} ref {d['ref']:#x}, block |x| max bf16 bits {amax_bits:#06x})"


# ------------------------------------------------------------------ Q: exhaustive quant
def quant_section(v2, live_quant, r2mod, dev, quick, expect_ftz=0):
    allb = torch.arange(65536, dtype=torch.int32)
    vals = bf16_from_bits(allb).float()
    fin = torch.isfinite(vals)
    pos = vals[(vals > 0) & fin].unique().sort().values  # every positive finite bf16
    rows = []
    # (a) every positive bf16 as block amax, alone and with its negation, rest zero
    a = torch.zeros(pos.numel() * 2, 32)
    a[: pos.numel(), 5] = pos
    a[pos.numel():, 0] = -pos
    a[pos.numel():, 31] = pos
    rows.append(a)
    # (b) every bf16 value x every reachable scale byte: per amax A (largest bf16 of its ue8m0 class) all |v| <= A
    c448 = torch.tensor(1.0 / 448.0, dtype=torch.float32)
    prod = (pos.float() * c448).float()
    bits = prod.view(torch.int32)
    ex, man = (bits >> 23) & 255, bits & 0x7FFFFF
    bump = ((man != 0) & ~((ex == 0) & (man <= 0x400000))).int()
    ue = torch.clamp(ex + bump, max=254)
    allv = vals[fin]
    step = 1 if not quick else 8
    for e in ue.unique().tolist()[::step]:
        A = pos[ue == e].max()
        cand = allv[allv.abs() <= A]
        nb = (cand.numel() + 30) // 31
        blk = torch.zeros(nb * 31)
        blk[: cand.numel()] = cand
        blk = blk.view(nb, 31)
        rows.append(torch.cat([torch.full((nb, 1), float(A)), blk], 1))
    # (c) specials: NaN / Inf blocks, mixed
    sp = torch.zeros(8, 32)
    sp[0, 3] = float("nan")
    sp[1, 4] = float("inf")
    sp[2, 5] = float("-inf")
    sp[3, :] = float("nan")
    sp[4, 7], sp[4, 9] = float("inf"), 1.0
    sp[5, 1], sp[5, 2] = float("nan"), 3.0
    sp[6, :] = -0.0
    sp[7, 0] = 3.3895313892515355e38
    rows.append(sp)
    blocks = torch.cat(rows, 0)
    nblk = blocks.shape[0]
    pad = (-nblk) % (K // 32)
    blocks = torch.cat([blocks, torch.zeros(pad, 32)], 0)
    x = blocks.to(torch.bfloat16).view(-1, K)
    # random + hostile rows too
    x = torch.cat([x, r2mod.hostile_rows(512, K, 4243, "cpu"), (torch.randn(512, K) * 3).to(torch.bfloat16)], 0)
    x = x.to(dev).contiguous()
    R = x.shape[0]
    CH = 384  # live quant per chunk: 384 * 160 = 61440 SF blocks < 65536 -> FlashInfer 4T/SF (the serving variant)
    print(f"Q: {R} rows x {K} = {R * K / 1e6:.1f}M elements ({nblk} constructed blocks), live quant in "
          f"{(R + CH - 1) // CH} chunks of <= {CH} rows (4T/SF variant)", flush=True)
    qs, ss = [], []
    for r0 in range(0, R, CH):
        xc = x[r0:r0 + CH].contiguous()
        q_, s_ = live_quant(xc, is_sf_swizzled_layout=True)
        qs.append(q_.view(torch.uint8))
        ss.append(unswizzle(s_.view(torch.uint8), xc.shape[0], K))
    ref_q, ref_s = torch.cat(qs, 0), torch.cat(ss, 0)
    xq = ref_q.view(torch.float8_e4m3fn)
    # v1 A8 dequant formula in torch (exact: e4m3 * 2^(e-127) is exact in fp32; bf16 rn of an exact value)
    ef = ref_s.to(torch.int32)
    scale = torch.where(ef == 0, torch.tensor(2.0 ** -127), torch.exp2((ef - 127).float()))
    ref_d = (xq.float().view(R, K // 32, 32) * scale.unsqueeze(-1)).view(R, K).to(torch.bfloat16)
    match = []
    xb = x.view(R, K // 32, 32)  # per-block input (abs max reported for the first difference)
    for ftz in (0, 1):
        q, s, d = v2.quant_debug(x, ftz)
        okq, oks, okd = torch.equal(q, ref_q), torch.equal(s, ref_s), same(d, ref_d)
        dq, tq = diff_report("codes", q, ref_q, R, K // 32, xb)
        ds, ts = diff_report("scales", s, ref_s, R, K // 32, xb)
        dd, td = diff_report("deq", d.contiguous().view(torch.int16), ref_d.contiguous().view(torch.int16), R,
                             K // 32, xb)
        RESULTS["quant"][f"ftz{ftz}"] = {"codes": dq, "scales": ds, "deq": dd}
        print(f"   ftz={ftz}: codes {'==' if okq else '!='} live, scales {'==' if oks else '!='} live, "
              f"deq {'==' if okd else '!='} v1 formula | {tq}; {ts}; {td}", flush=True)
        if okq and oks and okd:
            match.append(ftz)
    RESULTS["quant"]["exhaustive_match_ftz"] = match
    # the mode the serving self-check would pick
    hx = r2mod.hostile_rows(256, K, 4242, dev)
    hq, hs = live_quant(hx, is_sf_swizzled_layout=True)
    picked = -1
    for ftz in (0, 1):
        q, s, _ = v2.quant_debug(hx, ftz)
        if torch.equal(q, hq.view(torch.uint8)) and torch.equal(s, unswizzle(hs.view(torch.uint8), 256, K)):
            picked = ftz
            break
    RESULTS["quant"]["self_check_pick"] = picked
    check("Q: fused quant == live quant (codes + scales + deq) exhaustively in exactly one FTZ mode, the expected one",
          match == [expect_ftz], f"matching modes {match}, expected [{expect_ftz}]")
    check("Q: the serving self-check's FTZ pick is that mode", picked in match and picked == expect_ftz,
          f"pick {picked}")
    return picked if (picked in match and picked == expect_ftz) else -1


# ------------------------------------------------------------------ G / C / X
def pad_mask(M, Kk, dev):
    """bool mask over the 1-D 128x4-swizzled scale tensor of an [M, Kk] quant: True where (row >= M or
    col >= Kk/32), i.e. the padding a kernel must never read."""
    nm, nk = (M + 127) // 128, (Kk // 32 + 3) // 4
    i = torch.arange(nm * nk * 512, device=dev)
    e, d, c, b, a = i % 4, (i // 4) % 4, (i // 16) % 32, (i // 512) % nk, i // (512 * nk)
    return ((a * 128 + d * 32 + c) >= M) | ((b * 4 + e) >= Kk // 32)


def gemv_section(v1, v2, live_quant, r2mod, wsets, dev, ftz, quick, recorded):
    Ms = [1, 2, 4, 5, 6, 8] if quick else list(range(1, 9))
    RESULTS.setdefault("poison_v1_info", [])
    grids = [0] if quick else [0, 28, 32, 40, 56, 64, 75, 112]
    for wname, (w, s) in wsets.items():
        for M in Ms:
            for xname, x in act_sets(M, 100 * M + len(wname), dev, r2mod, recorded):
                xq, xs = live_quant(x, is_sf_swizzled_layout=True)
                xq8, xs8 = xq.contiguous().view(torch.uint8), xs.view(torch.uint8)
                ref = v1.skinny_mx_a8(xq8, xs8, w, s)
                bad = []
                for g in grids:
                    if not v2.supports(M, N_SHARD, K, g):
                        continue
                    for pdl in (False, True):
                        y = v2.skinny_v2_a8(xq8, xs8, w, s, g, pdl)
                        if not same(y, ref):
                            bad.append(f"a8 grid={g} pdl={pdl}: {first_diff(y, ref)}")
                if ftz >= 0:
                    for pdl in (False, True):
                        y = v2.skinny_v2_q(x, w, s, 0, pdl, ftz)
                        if not same(y, ref):
                            bad.append(f"qf pdl={pdl}: {first_diff(y, ref)}")
                    # rows >= M of the underlying buffer poisoned (bf16 NaN 0xFFFF): must not be read
                    xbig = torch.empty((M + 8, K), dtype=torch.bfloat16, device=dev)
                    xbig.view(torch.int16).fill_(-1)
                    xbig[:M].copy_(x)
                    y = v2.skinny_v2_q(xbig[:M], w, s, 0, False, ftz)
                    if not same(y, ref):
                        bad.append(f"qf rows>=M poisoned: {first_diff(y, ref)}")
                # swizzled-scale padding (rows >= M, cols >= K/32) poisoned 0xC3: must not be read
                pm = pad_mask(M, K, dev)
                if pm.numel() == xs8.numel():
                    xsp = xs8.clone().reshape(-1)
                    xsp[pm] = 0xC3
                    xsp = xsp.view_as(xs8)
                    for pdl in (False, True):
                        y = v2.skinny_v2_a8(xq8, xsp, w, s, 0, pdl)
                        if not same(y, ref):
                            bad.append(f"a8 scale padding poisoned pdl={pdl}: {first_diff(y, ref)}")
                    y1 = v1.skinny_mx_a8(xq8, xsp, w, s)
                    if not same(y1, ref):
                        RESULTS["poison_v1_info"].append(f"M={M} w={wname} x={xname}: {first_diff(y1, ref)}")
                else:
                    bad.append(f"scale tensor size {xs8.numel()} != swizzled size {pm.numel()} (layout changed?)")
                check(f"G: M={M} w={wname} x={xname}", not bad, "; ".join(bad[:3]))
    # C: graphs (PDL edges captured)
    w, s = next(iter(wsets.values()))
    for M in (1, 2, 4, 5, 6):
        x = act_sets(M, 7 + M, dev, r2mod, None)[1][1]
        xq, xs = live_quant(x, is_sf_swizzled_layout=True)
        ref = v1.skinny_mx_a8(xq.view(torch.uint8), xs.view(torch.uint8), w, s)
        src = x.clone()
        st = torch.cuda.Stream()
        outs = {}
        with torch.cuda.stream(st):
            for name in ("v1", "v2pdl", "qfpdl", "qf"):  # r4: + qf without PDL (the V2=1 serving arm)
                if name.startswith("qf") and ftz < 0:
                    continue
                gph = torch.cuda.CUDAGraph()
                xin = torch.empty_like(x)
                for _ in range(2):  # warm-up allocations outside capture
                    xin.copy_(src)
                with torch.cuda.graph(gph, stream=st):
                    xin.copy_(src)  # producer kernel
                    if name.startswith("qf"):
                        o = v2.skinny_v2_q(xin, w, s, 0, name == "qfpdl", ftz)
                    else:
                        q_, s_ = live_quant(xin, is_sf_swizzled_layout=True)
                        o = (v1.skinny_mx_a8(q_.view(torch.uint8), s_.view(torch.uint8), w, s) if name == "v1"
                             else v2.skinny_v2_a8(q_.view(torch.uint8), s_.view(torch.uint8), w, s, 0, True))
                for _ in range(3):
                    gph.replay()
                st.synchronize()
                outs[name] = o.clone()
        for name, o in outs.items():
            check(f"C: graph replay {name} M={M} == eager v1", same(o, ref), first_diff(o, ref))
    # X: forced mismatch control
    x = act_sets(4, 3, dev, r2mod, None)[0][1]
    xq, xs = live_quant(x, is_sf_swizzled_layout=True)
    ref = v1.skinny_mx_a8(xq.view(torch.uint8), xs.view(torch.uint8), w, s)
    w2 = w.clone()
    w2[17, 1234] ^= 0x01
    y = v2.skinny_v2_a8(xq.view(torch.uint8), xs.view(torch.uint8), w2, s, 0, False)
    check("X: forced-mismatch control (one flipped weight bit) is detected", not same(y, ref), first_diff(y, ref))


# ------------------------------------------------------------------ T: timing
def time_graph(fn, reps=200, per=20):
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        for i in range(3):
            fn(i)
        st.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            for i in range(per):
                fn(i)
        g.replay()
        st.synchronize()
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(st)
            g.replay()
            b.record(st)
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3 / per)
    ts.sort()
    return ts[len(ts) // 2], ts[len(ts) // 10], ts[(len(ts) * 9) // 10]


def timing_section(v1, v2, live_quant, wsets, dev, ftz, quick):
    w0, s0 = next(iter(wsets.values()))
    ncopy = max(2, (96 << 20) // (w0.numel() + s0.numel()))
    cold = [(w0.clone(), s0.clone()) for _ in range(ncopy)]
    print(f"\nT: per-call us, median [p10, p90] of 200 graph replays x 20 calls; cold = {ncopy} weight copies",
          flush=True)
    Ms = [1, 2, 4, 6] if quick else [1, 2, 3, 4, 5, 6, 8]
    for M in Ms:
        x = (torch.randn(M, K, device=dev) * 0.05).to(torch.bfloat16)
        src = x.clone()
        row = {"M": M}
        for temp in ("warm", "cold"):
            W = (lambda i: (w0, s0)) if temp == "warm" else (lambda i: cold[i % ncopy])

            def prod():
                x.copy_(src)

            arms = {
                "producer": lambda i: prod(),
                "v1": lambda i: (prod(), v1.skinny_mx_a8(*[t.view(torch.uint8) for t in
                                                           live_quant(x, is_sf_swizzled_layout=True)], *W(i))),
                "v2": lambda i: (prod(), v2.skinny_v2_a8(*[t.view(torch.uint8) for t in
                                                           live_quant(x, is_sf_swizzled_layout=True)], *W(i), 0,
                                                         False)),
                "v2pdl": lambda i: (prod(), v2.skinny_v2_a8(*[t.view(torch.uint8) for t in
                                                              live_quant(x, is_sf_swizzled_layout=True)], *W(i), 0,
                                                            True)),
            }
            if ftz >= 0:
                arms["qf"] = lambda i: (prod(), v2.skinny_v2_q(x, *W(i), 0, False, ftz))
                arms["qfpdl"] = lambda i: (prod(), v2.skinny_v2_q(x, *W(i), 0, True, ftz))
            res = {k: time_graph(f) for k, f in arms.items()}
            p = res["producer"][0]
            row[temp] = {k: round(v[0] - (p if k != "producer" else 0), 2) for k, v in res.items()}
            print(f"  M={M} {temp}: " + "  ".join(f"{k} {v[0] - (p if k != 'producer' else 0):5.2f} "
                                                  f"[{v[1]:.1f},{v[2]:.1f}]" for k, v in res.items()) +
                  "   (chain minus producer)", flush=True)
        RESULTS["timing"].append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-lib", default="~/dsv41-r2/skinny_mx/build")
    ap.add_argument("--v2-lib", default=os.path.join(HERE, "prebuilt"))  # the binary install.py ships
    ap.add_argument("--model", default="")
    ap.add_argument("--layers", default="0,13,39")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--acts", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--allow-any-so", action="store_true")
    ap.add_argument("--expect-ftz", type=int, default=0, choices=(0, 1),
                    help="FTZ mode of the live amax*(1/448) multiply (0: non-FTZ FMUL in the live 4T/SF SASS)")
    args = ap.parse_args()
    RESULTS["_json"] = args.json

    os.environ.setdefault("DSV41_SKINNY_V2", "1")  # the kit module refuses to import without it
    spec = importlib.util.spec_from_file_location("dsv41_r2_skinny_kit", os.path.join(HERE, "new",
                                                                                      "dsv41_r2_skinny.py"))
    r2mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r2mod)
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize as live_quant

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    print(f"GPU {torch.cuda.get_device_name()} SMs={torch.cuda.get_device_properties(0).multi_processor_count} "
          f"torch {torch.__version__} cuda {torch.version.cuda}")
    v1 = load_so(args.v1_lib, "skinny_mx_ext")
    v2 = load_so(args.v2_lib, "skinny_mx_v2_ext")
    v2.prepare()
    import hashlib
    kitj = json.load(open(os.path.join(HERE, "KIT.json")))
    want = [f.get("sha256") for f in kitj["files"] if f["path"].endswith("skinny_mx_v2_ext.so")][0]
    got = hashlib.sha256(open(v2.__file__, "rb").read()).hexdigest()
    v1sha = hashlib.sha256(open(v1.__file__, "rb").read()).hexdigest()
    RESULTS["v2_so"], RESULTS["v1_so"] = {"path": v2.__file__, "sha256": got}, {"path": v1.__file__, "sha256": v1sha}
    print(f"v1 .so sha256 {v1sha} (live ref {kitj['live_refs']['skinny_mx_ext.so_sha256'][:16]}...)\n"
          f"v2 .so sha256 {got} (KIT.json {want})")
    check("v1 .so is the live skinny_mx_ext.so (KIT.json live_refs)", v1sha == kitj["live_refs"]["skinny_mx_ext.so_sha256"],
          v1sha[:16])
    if not args.allow_any_so:
        check("v2 .so is the KIT.json binary (the one install.py ships)", bool(want) and got == want, got[:16])
    check("v1 split for the shard is 8 (v2's supported class)", v1.chosen_split(N_SHARD, K) == 8 ==
          v2.v1_split(N_SHARD, K, 0))
    recorded = torch.load(args.acts) if args.acts else None
    if isinstance(recorded, torch.Tensor):
        recorded = [recorded]
    if args.model:
        recorded = list(recorded or []) + [(f"ckpt-L{L.strip()}", ckpt_acts(args.model, L.strip()))
                                         for L in args.layers.split(",")]
    assert_live_reference(live_quant, dev)  # exit 2 unless the reference is the serving FlashInfer quant
    t0 = time.time()
    ftz = quant_section(v2, live_quant, r2mod, dev, args.quick, args.expect_ftz)
    wsets = {"rand": rand_weights(1, False, dev), "hostile": rand_weights(2, True, dev)}
    if args.model:
        for L in args.layers.split(","):
            wsets[f"real-L{L}-r{args.rank}"] = real_weights(args.model, L.strip(), args.rank, dev)
    gemv_section(v1, v2, live_quant, r2mod, wsets, dev, ftz, args.quick, recorded)
    torch.cuda.synchronize()
    if not args.no_timing:
        timing_section(v1, v2, live_quant, wsets, dev, ftz, args.quick)
    RESULTS["fails"] = FAILS
    RESULTS["elapsed_s"] = round(time.time() - t0, 1)
    RESULTS.pop("_json", None)
    if RESULTS.get("poison_v1_info"):
        print(f"INFO (not a gate): live v1 output changed under scale-padding poison in "
              f"{len(RESULTS['poison_v1_info'])} cases, first: {RESULTS['poison_v1_info'][0]}")
    if args.json:
        json.dump(RESULTS, open(args.json, "w"), indent=1)
    n = len(RESULTS["bitwise"])
    print(f"\n{'ALL BITWISE CASES IDENTICAL' if not FAILS else 'MISMATCH'}: {n - len(FAILS)}/{n} checks passed "
          f"({RESULTS['elapsed_s']} s)" + ("" if not FAILS else "\n  " + "\n  ".join(FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
