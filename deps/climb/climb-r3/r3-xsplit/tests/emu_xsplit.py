#!/usr/bin/env python3
"""CPU emulation of the cross-tile op order: live prefill_mg_impl (serial tile loop) vs XSPLIT
(phase A per tile + epoch-free prefix fold + phase B in-order replay), bit for bit.

Scope. The per-tile body (QK, mask, warp max, cross-warp max, exp2, warp sum, V-scale atomicMax,
normalise, FP8 W quant, XV-nope MMA) is the same source in both kernels; what XSPLIT changes is
WHERE the cross-tile recurrences run and on WHICH operands. This emulation models every op of the
recurrences and of the per-tile values they consume, with explicit float32 rounding and FTZ at
every step, and checks that the decomposition hands each op identical operand bits in identical
order:
  running max      m = fmaxf(m, tm_t) from -1e30, alpha_t = ex2(old_m - nm)          (F5)
  guarded rescale  if (alpha[gid] < 1 || alpha[gid+8] < 1): acc *= alpha (FMUL.FTZ)  (F4: dead
                   row alpha1 carried)
  row sum          warp_l = guard ? warp_l*alpha : warp_l; warp_l += ls  (FMUL, FADD)  (F2)
  nope accumulate  acc_o = FFMA(xv, sc, acc_o)
  rope accumulate  acc_rope = MMA(A = bf16(w), B = rope, C = acc_rope), 4 k-steps     (F3)
  epilogue         ts = 0 + l_w0 + ... + l_w7; sink fold FFMA/EX2/FADD/RCP; FMUL il; bf16;
                   lse = lg2(l) + m, sink merge
MUFU (ex2/rcp/lg2), MMA and FFMA are deterministic stand-ins (float64 then one float32 rounding;
MMA output is not flushed, like HMMA); equality of the real instructions on identical operands is
the device test's job. Negative controls show the emulation detects each mandatory correction
being violated (l as one FFMA, guard without alpha1, rope MMA split into C=0 + FADD, a running
max folded in the wrong order with ±0 ties, a stale epoch read).

Run: python3 -I tests/emu_xsplit.py   (numpy only; ~1 min on 512S1)
"""
import sys

import numpy as np

F32 = np.float32
TINY = F32(2.0 ** -126)
NEG_BIG = F32(-1e30)
LOG2E = F32(1.44269504088896340736)
ROWS, LIVE, ENT, WARPS, EPW = 16, 8, 64, 8, 8
NVC, VCH, DN, DR = 7, 64, 448, 64


# ---------------------------------------------------------------- float32 + FTZ primitives
def ftz(x):
    x = np.asarray(x, dtype=F32)
    return np.where((np.abs(x) < TINY) & (x != 0), np.copysign(F32(0), x), x).astype(F32)


def fmul(a, b):
    return ftz(ftz(a) * ftz(b))


def fadd(a, b):
    return ftz(ftz(a) + ftz(b))


def fsub(a, b):
    return ftz(ftz(a) - ftz(b))


def ffma(a, b, c):  # stand-in: one rounding from float64
    a, b, c = (ftz(v).astype(np.float64) for v in (a, b, c))
    with np.errstate(invalid="ignore", over="ignore"):
        return ftz((a * b + c).astype(F32))


def fmax(a, b):  # FMNMX.FTZ max: NaN loses, max(-0,+0) = +0
    a, b = ftz(a), ftz(b)
    r = np.where(np.isnan(a), b, np.where(np.isnan(b), a, np.maximum(a, b)))
    both0 = (a == 0) & (b == 0)
    r = np.where(both0, np.where(np.signbit(a) & np.signbit(b), F32(-0.0), F32(0.0)), r)
    return r.astype(F32)


def ex2(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(np.exp2(ftz(x).astype(np.float64)).astype(F32))


def rcp(x):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return ftz((1.0 / ftz(x).astype(np.float64)).astype(F32))


def lg2(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return ftz(np.log2(ftz(x).astype(np.float64)).astype(F32))


def bf16(x):  # RN to bf16, returned as float32 values
    x = np.asarray(x, dtype=F32)
    u = x.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    r = np.where(np.isnan(x), np.uint64(0x7FC00000), r)
    return r.astype(np.uint32).view(F32)


def fp8(x):  # deterministic e4m3-like quant stand-in (3 mantissa bits, saturate 448)
    x = np.asarray(x, dtype=F32)
    with np.errstate(invalid="ignore", over="ignore"):
        m, e = np.frexp(x.astype(np.float64))
        q = np.ldexp(np.round(m * 16) / 16, e)
        q = np.clip(q, -448.0, 448.0)
    return np.where(np.isnan(x), np.nan, q).astype(F32)


def mma(A, B, C):  # A [16,k], B [k,n], C [16,n] -> one rounding per element, output not flushed
    with np.errstate(invalid="ignore", over="ignore"):
        return (A.astype(np.float64) @ B.astype(np.float64) + C.astype(np.float64)).astype(F32)


def imax_bits(vals):  # atomicMax on the int view (all candidates >= 0 or NaN)
    return vals.view(np.int32).max(axis=-1).view(F32)


# ---------------------------------------------------------------- per-tile body (shared code)
def tile_scores_max(s):
    """softmax_warp_max per warp + cross-warp fold from -1e30 (threads < 16)."""
    lm = np.full((WARPS, ROWS), NEG_BIG, F32)
    for w in range(WARPS):
        blk = s[:, w * EPW:(w + 1) * EPW]
        lane = [fmax(blk[:, 2 * t], blk[:, 2 * t + 1]) for t in range(4)]
        l01, l23 = fmax(lane[0], lane[1]), fmax(lane[2], lane[3])
        lm[w] = fmax(l01, l23)
    tm = np.full(ROWS, NEG_BIG, F32)
    for w in range(WARPS):
        tm = fmax(tm, lm[w])
    return tm


def tile_after_max(s, nm, vsc, vq):
    """exp2, warp sums, V-scale max, normalise, W quant, XV-nope MMA. Returns per-tile values."""
    w = ex2(fsub(s, nm[:, None]))
    ls = np.zeros((WARPS, ROWS), F32)
    for wp in range(WARPS):
        blk = w[:, wp * EPW:(wp + 1) * EPW]
        v = [fadd(blk[:, 2 * t], blk[:, 2 * t + 1]) for t in range(4)]
        ls[wp] = fadd(fadd(v[0], v[1]), fadd(v[2], v[3]))
    sc = np.zeros((NVC, ROWS), F32)
    xv = np.zeros((ROWS, DN), F32)
    for vc in range(NVC):
        ws = fmul(w, vsc[None, :, vc])
        pair = fmax(ws[:, 0::2], ws[:, 1::2])
        amax = imax_bits(np.ascontiguousarray(pair))
        sc[vc] = fmul(fmax(amax, F32(1e-10)), F32(1.0 / 448.0))
        si = rcp(sc[vc])
        wq = fp8(fmul(ws, si[:, None]))
        Vb = vq[:, vc * VCH:(vc + 1) * VCH]
        x = mma(wq[:, :32], Vb[:32], np.zeros((ROWS, VCH), F32))
        xv[:, vc * VCH:(vc + 1) * VCH] = mma(wq[:, 32:], Vb[32:], x)
    return w, ls, sc, xv


def guard_of(alpha):
    return (alpha[:LIVE] < 1) | (alpha[LIVE:] < 1)  # per gid


def epilogue(acc_o, acc_rope, warp_l, m, sink):
    l = np.zeros(ROWS, F32)
    for w in range(WARPS):
        l = fadd(l, warp_l[w])
    out = np.zeros((LIVE, DN + DR), F32)
    lse = np.zeros(LIVE, F32)
    for h in range(LIVE):
        if sink is not None:
            d0 = fadd(l[h], ex2(ffma(sink[h], LOG2E, -m[h])))
        else:
            d0 = l[h]
        il = rcp(d0) if d0 > 0 else F32(0)
        out[h, :DN] = bf16(fmul(acc_o[h], il))
        out[h, DN:] = bf16(fmul(acc_rope[h], il))
        x = fadd(lg2(l[h]), m[h]) if l[h] > 0 else NEG_BIG
        if sink is not None:
            sl = fmul(sink[h], LOG2E)
            x = fadd(lg2(fadd(ex2(fsub(sl, x)), F32(1))), x) if x != NEG_BIG else sl
        lse[h] = x
    return out, lse


# ---------------------------------------------------------------- live: serial tile loop
def live_token(tiles, sink, variant=None):
    acc_o = np.zeros((ROWS, DN), F32)
    acc_rope = np.zeros((ROWS, DR), F32)
    warp_l = np.zeros((WARPS, ROWS), F32)
    m = np.full(ROWS, NEG_BIG, F32)
    for tl in tiles:
        tm = tile_scores_max(tl["s"])
        nm = fmax(m, tm)
        alpha = ex2(fsub(m, nm))
        m = nm
        g = guard_of(alpha)
        for gid in range(LIVE):
            if g[gid]:
                for r in (gid, gid + LIVE):
                    acc_o[r] = fmul(acc_o[r], alpha[r])
                    acc_rope[r] = fmul(acc_rope[r], alpha[r])
                    warp_l[:, r] = fmul(warp_l[:, r], alpha[r])
        w, ls, sc, xv = tile_after_max(tl["s"], nm, tl["vsc"], tl["vq"])
        warp_l = fadd(warp_l, ls)
        for vc in range(NVC):
            sl = slice(vc * VCH, (vc + 1) * VCH)
            acc_o[:, sl] = ffma(xv[:, sl], sc[vc][:, None], acc_o[:, sl])
        A = bf16(w)
        for ks in range(4):
            k = slice(ks * 16, ks * 16 + 16)
            acc_rope = mma(A[:, k], tl["rope"][k], acc_rope)
    return epilogue(acc_o, acc_rope, warp_l, m, sink)


# ---------------------------------------------------------------- XSPLIT: A units + B replay
def xsplit_token(tiles, sink, rng, variant=None):
    n = len(tiles)
    pub = {}  # published tile maxima (the epoch-tagged words)
    order = list(range(n))
    rng.shuffle(order)  # concurrent CTAs: QK + cross-warp max of every tile in any order
    for t in order:
        pub[t] = tile_scores_max(tiles[t]["s"])
    if variant == "stale_epoch" and n >= 2:
        pub[n - 2] = pub[n - 2] - F32(1.0)  # a stale word from an earlier call accepted
    rows = [None] * n
    order = list(range(n))
    rng.shuffle(order)
    for t in order:  # phase A, any order
        tm = pub[t]
        old_m = np.full(ROWS, NEG_BIG, F32)
        prefix = list(range(t))
        if variant == "max_order":
            prefix = prefix[::-1]
        for tt in prefix:
            old_m = fmax(old_m, pub[tt]) if variant != "max_order" else fmax(pub[tt], old_m)
        nm = fmax(old_m, tm)
        alpha = ex2(fsub(old_m, nm))
        w, ls, sc, xv = tile_after_max(tiles[t]["s"], nm, tiles[t]["vsc"], tiles[t]["vq"])
        rows[t] = dict(alpha=alpha, nm=nm, sc=sc, ls=ls[:, :LIVE].copy(),
                       w=w[:LIVE].copy(), xv=xv[:LIVE].copy(), rb=tiles[t]["rope"])
    # phase B: one CTA per token, tiles in order, live rows only (dead rows never reach output)
    acc_o = np.zeros((LIVE, DN), F32)
    acc_rope = np.zeros((ROWS, DR), F32)
    warp_l = np.zeros((WARPS, LIVE), F32)
    for t in range(n):
        r = rows[t]
        a = r["alpha"]
        g = (a[:LIVE] < 1) if variant == "no_alpha1" else guard_of(a)
        for gid in range(LIVE):
            if g[gid]:
                acc_o[gid] = fmul(acc_o[gid], a[gid])
                acc_rope[gid] = fmul(acc_rope[gid], a[gid])
                acc_rope[gid + LIVE] = fmul(acc_rope[gid + LIVE], a[gid + LIVE])
                if variant != "l_ffma":
                    warp_l[:, gid] = fmul(warp_l[:, gid], a[gid])
        if variant == "l_ffma":
            gg = g[None, :]
            warp_l = np.where(gg, ffma(warp_l, a[None, :LIVE], r["ls"]), fadd(warp_l, r["ls"]))
        else:
            warp_l = fadd(warp_l, r["ls"])
        for vc in range(NVC):
            sl = slice(vc * VCH, (vc + 1) * VCH)
            acc_o[:, sl] = ffma(r["xv"][:, sl], r["sc"][vc][:LIVE, None], acc_o[:, sl])
        A = np.zeros((ROWS, ENT), F32)
        A[:LIVE] = bf16(r["w"])
        if variant == "rope_split":
            part = np.zeros((ROWS, DR), F32)
            for ks in range(4):
                k = slice(ks * 16, ks * 16 + 16)
                part = mma(A[:, k], r["rb"][k], part)
            acc_rope = fadd(acc_rope, part)
        else:
            for ks in range(4):
                k = slice(ks * 16, ks * 16 + 16)
                acc_rope = mma(A[:, k], r["rb"][k], acc_rope)
    m = rows[n - 1]["nm"] if n else np.full(ROWS, NEG_BIG, F32)
    full_o = np.zeros((ROWS, DN), F32)
    full_o[:LIVE] = acc_o
    full_l = np.zeros((WARPS, ROWS), F32)
    full_l[:, :LIVE] = warp_l
    return epilogue(full_o, acc_rope, full_l, m, sink)


# ---------------------------------------------------------------- inputs
SCALE = F32(1.0 / np.sqrt(512.0)) * LOG2E


def gen_token(rng, n, cls):
    tiles = []
    dead_seen_valid = False
    for t in range(n):
        qk = (rng.standard_normal((LIVE, ENT)) * rng.uniform(0.5, 8.0)).astype(F32)
        valid = rng.random(ENT) > (0.03 if cls != "struct" else 0.25)
        if cls == "struct" and rng.random() < 0.2:
            valid[:] = False
        if cls in ("ties",):
            qk = np.round(qk).astype(F32)
            if rng.random() < 0.5:
                qk[:] = qk[:, :1]
        if cls == "nan" and rng.random() < 0.3:
            qk[rng.integers(0, LIVE), rng.integers(0, ENT)] = np.nan
        if cls == "inf" and rng.random() < 0.3:
            qk[rng.integers(0, LIVE), rng.integers(0, ENT)] = np.inf * rng.choice([-1, 1])
        if cls == "zeroq" and rng.random() < 0.5:
            qk[rng.integers(0, LIVE)] = F32(0.0)
            qk[rng.integers(0, LIVE)] = F32(-0.0)
        full = np.zeros((ROWS, ENT), F32)  # dead rows: Q = 0 -> qk = 0
        full[:LIVE] = qk
        full[:, ~valid] = NEG_BIG
        s = fmul(full, SCALE)
        vsc = np.ldexp(1.0, rng.integers(-12, 4, size=(ENT, NVC))).astype(F32)
        if cls == "scale_huge" and rng.random() < 0.4:
            vsc[rng.integers(0, ENT), rng.integers(0, NVC)] = rng.choice([F32(2.0 ** 127), np.inf])
        vq = fp8(rng.standard_normal((ENT, DN)).astype(F32) * 64)
        if cls == "fp8_nan" and rng.random() < 0.4:
            vq[rng.integers(0, ENT), rng.integers(0, DN)] = np.nan
        rope = bf16(rng.standard_normal((ENT, DR)).astype(F32))
        rope[~valid] = 0.0  # idx < 0 -> 0 (slot-0 guard)
        if cls == "rope_special" and rng.random() < 0.4:
            e = rng.integers(0, ENT)
            rope[e, rng.integers(0, DR)] = rng.choice([np.inf, -np.inf, np.nan, F32(1e-40), F32(0)])
        if cls == "tiny":
            rope = (rope * F32(2.0 ** -120)).astype(F32)
        dead_seen_valid |= bool(valid.any())
        tiles.append(dict(s=s, vsc=vsc, vq=vq, rope=rope))
    return tiles


def gen_alpha1_case(rng):
    """Constructed F4 case. Tile 0 leaves a subnormal MMA output (2^-129) in the live rows of
    acc_rope. Tile 1: every live row keeps its max (alpha0 == 1 exactly) while the dead rows' max
    rises (alpha1 = 0), so the live guard fires on alpha1 alone and FMUL.FTZ by 1.0 flushes the
    subnormal; tile 1's MMA then lands at 2^-125, where a carried 2^-129 changes the bits (and
    il ~ 1 keeps the output normal). Dropping alpha1 from the guard must be detected here."""
    tiles = []
    for t in range(2):
        s = np.full((ROWS, ENT), NEG_BIG, F32)
        s[:LIVE, 0] = F32(0.0) if t == 0 else F32(-20.0)
        if t == 1:
            s[LIVE:, 0] = F32(0.0)
        vsc = np.ones((ENT, NVC), F32)
        vq = fp8(rng.standard_normal((ENT, DN)).astype(F32))
        rope = np.zeros((ENT, DR), F32)
        rope[0] = F32(2.0 ** -129) if t == 0 else F32(2.0 ** -105)
        tiles.append(dict(s=s, vsc=vsc, vq=vq, rope=rope))
    return tiles


def same(a, b):
    (o1, l1), (o2, l2) = a, b
    return np.array_equal(o1.view(np.uint32), o2.view(np.uint32)) and \
        np.array_equal(l1.view(np.uint32), l2.view(np.uint32))


def main():
    np.seterr(all="ignore")
    rng = np.random.default_rng(20261108)
    classes = ["normal", "ties", "nan", "inf", "zeroq", "scale_huge", "fp8_nan", "rope_special",
               "struct", "tiny"]
    runs = mism = 0
    for cls in classes:
        for rep in range(40):
            n = int(rng.choice([0, 1, 2, 3, 5, 10, 13, 18]))
            tiles = gen_token(rng, n, cls)
            sink = None if rng.random() < 0.3 else (rng.standard_normal(LIVE) * 2).astype(F32)
            a = live_token(tiles, sink)
            b = xsplit_token(tiles, sink, rng)
            runs += 1
            if not same(a, b):
                mism += 1
                print(f"MISMATCH cls={cls} rep={rep} n={n}")
    a1 = gen_alpha1_case(rng)
    a = live_token(a1, None)
    b = xsplit_token(a1, None, rng)
    runs += 1
    if not same(a, b):
        mism += 1
        print("MISMATCH alpha1 constructed case")
    print(f"exact decomposition: runs={runs} mismatches={mism}")

    # negative controls: each must be DETECTED on at least one input
    controls = {}
    for variant in ("l_ffma", "rope_split", "stale_epoch", "max_order"):
        hit = 0
        for rep in range(40):
            cls = classes[rep % len(classes)]
            n = int(rng.choice([2, 3, 5, 10]))
            tiles = gen_token(rng, n, cls)
            sink = (rng.standard_normal(LIVE) * 2).astype(F32)
            if not same(live_token(tiles, sink), xsplit_token(tiles, sink, rng, variant)):
                hit += 1
        controls[variant] = hit
    controls["no_alpha1"] = int(not same(live_token(a1, None),
                                         xsplit_token(a1, None, rng, "no_alpha1")))
    print("negative controls (inputs on which the broken variant is detected):")
    ok = True
    for k, v in controls.items():
        note = ""
        if k == "max_order":
            note = " (fmax is order-insensitive on these values: the fold order is safe, F5)"
        elif v == 0:
            ok = False
            note = "  <- NOT DETECTED"
        print(f"  {k:12s} {v}{note}")
    good = mism == 0 and ok
    print("EMULATION", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
