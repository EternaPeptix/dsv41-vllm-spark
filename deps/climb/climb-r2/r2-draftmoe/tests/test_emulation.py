#!/usr/bin/env python3
"""r2-draftmoe CPU emulation (torch, explicit roundings): old (padded N768 + stock permute) vs new (compact N576 +
fused permute) op sequences for the DSpark drafter routed MoE (E 128, top-3, TP8 rank: 288 live columns).

What is emulated (per output element, in the op order of the live kernels; see EXACTNESS.md):
  L  the live weight loader (_load_w13 / _narrow_expert_data_for_padding / _get_hidden_dim, executed from the live
     routed_experts.py) into a padded [768] and a compact [576] parameter, all 8 TP ranks;
  W  the per-row MN-major UE8M0 weight-scale pack (transform_sf_into_required_layout semantics);
  F1 FC1: per k32 chunk in K order  acc = RN32(acc + sum_32(a*b) * 2^(sfa-127) * 2^(sfb-127)), acc0 = +0.0f, then
     bf16 = RN(acc); IEEE signed zeros kept (numpy/torch float semantics);
  S  _silu_mul_quant_fp8_packed_kernel (padded: N=768, 3 groups; compact: N=576, padded_output_size=384, masked
     loads other=0.0) with bf16 round of y, group absmax, ceil-log2 UE8M0 scale, clamp, RN to e4m3;
  F2 FC2 (K=384, unchanged) and the ep_gather (fp32 sum in fixed top-k order, RN to bf16);
  P  stock permute (count, aligned cumsum, full(-1), zeros, atomic scatter, transpose_and_pack formula) vs the fused
     plan+scatter (histogram, aligned cumsum, atomic cursor, direct pack) with independent random atomic orders.
Inputs: random, hostile (max magnitude, zeros, -0 products, cancellation to 0, denormal-range scales, ties, all
tokens on one expert, M=160 touching all experts), NaN/Inf-free. Result: bitwise identical everywhere.
Run: python3 tests/test_emulation.py   (torch CPU; ~1-3 min on the Mac Studio)
"""
import ast
import os
import sys
import types

import torch

torch.manual_seed(20261108)
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def bits(t):
    return t.contiguous().view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


# ---------------------------------------------------------------------------------------------- decoders
FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                   dtype=torch.float64)


def fp4_decode(packed):  # [..., K/2] uint8 -> [..., K] float64; low nibble = even k (DeepGEMM kPackedFP4)
    lo = (packed & 0xF).long()
    hi = (packed >> 4).long()
    return torch.stack([FP4[lo], FP4[hi]], dim=-1).flatten(-2)


def fp8_decode(b):  # uint8 e4m3fn bytes -> float64 (NaN codes never generated)
    return b.view(torch.float8_e4m3fn).to(torch.float64)


def pow2(e):  # ue8m0 byte -> exact float64 2^(e-127)
    return torch.ldexp(torch.ones_like(e, dtype=torch.float64), e.long() - 127)


def rand_fp8(shape, lo=0, hi=256):
    b = torch.randint(lo, hi, shape, dtype=torch.uint8)
    b[(b & 0x7F) == 0x7F] = 0x7E  # no NaN
    return b


# ---------------------------------------------------------------------------------------------- F1/F2 GEMM model
def gemm_model(a_bytes, sfa, w_packed, sfb, k_scale_a=128):
    """D[r, n] = RN_bf16( chain over k32 chunks c of RN32(acc + chunk_sum) ), acc0 = +0.0f.

    a_bytes [R, K] e4m3 bytes, sfa [R, K/k_scale_a] ue8m0, w_packed [N, K/2] fp4 codes, sfb [N, K/32] ue8m0.
    Chunk sums of 32 exact products share one scale pair, so they are exact in float64."""
    R, K = a_bytes.shape
    C = K // 32
    a = fp8_decode(a_bytes).view(R, C, 32) * pow2(sfa).repeat_interleave(k_scale_a // 32, dim=1).view(R, C, 1)
    b = fp4_decode(w_packed).view(-1, C, 32) * pow2(sfb).view(-1, C, 1)
    S = torch.einsum("rck,nck->rnc", a, b)  # exact
    acc = torch.zeros(R, b.shape[0], dtype=torch.float32)  # +0.0f
    for c in range(C):
        acc = (acc.to(torch.float64) + S[:, :, c]).to(torch.float32)
    return acc.to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------- S model
def silu_writer(inp_bf16, padded_output_size=None, clamp=None, alpha=1.0, beta=0.0, group=128):
    """_silu_mul_quant_fp8_packed_kernel per element (fp32 math, masked loads other=0.0)."""
    R, N = inp_bf16.shape
    hidden = N // 2
    out_size = hidden if padded_output_size is None else -(-hidden // group) * group
    assert padded_output_size in (None, out_size)
    cols = torch.arange(out_size)
    live = cols < hidden
    g = torch.zeros(R, out_size, dtype=torch.float32)
    u = torch.zeros(R, out_size, dtype=torch.float32)
    g[:, live] = inp_bf16[:, :hidden].to(torch.float32)
    u[:, live] = inp_bf16[:, hidden:].to(torch.float32)
    if clamp is not None:
        g = torch.minimum(g, torch.tensor(clamp, dtype=torch.float32))
        u = torch.clamp(u, -clamp, clamp)
    a32, b32 = torch.tensor(alpha, dtype=torch.float32), torch.tensor(beta, dtype=torch.float32)
    glu = g / (torch.tensor(1.0, dtype=torch.float32) + torch.exp(-g * a32))
    y = (u + b32) * glu
    y = y.to(torch.bfloat16).to(torch.float32)
    yg = y.view(R, out_size // group, group)
    absmax = yg.abs().amax(dim=2)
    scale_raw = torch.maximum(absmax / torch.tensor(448.0, dtype=torch.float32), torch.tensor(1e-10, dtype=torch.float32))
    exponent = torch.ceil(torch.log2(scale_raw))
    scale = torch.exp2(exponent)
    yq = torch.clamp(yg / scale[:, :, None], -448.0, 448.0).view(R, out_size)
    q = yq.to(torch.float8_e4m3fn).view(torch.uint8)
    sbyte = torch.clamp(exponent + 127.0, 0.0, 255.0).to(torch.int32).to(torch.uint8)
    return q, sbyte


def gather_model(mm2_bf16, inv, w):
    out = torch.zeros(inv.shape[0], mm2_bf16.shape[1], dtype=torch.float32)
    for k in range(inv.shape[1]):  # fixed top-k order, fp32 FMA-free model: acc += fp32(x) * w
        out = out + mm2_bf16[inv[:, k].long()].to(torch.float32) * w[:, k:k + 1]
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------- L: live loader
src = open(os.path.join(KIT, "live-ref/model_executor/layers/fused_moe/routed_experts.py")).read()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(
    isinstance(f, ast.FunctionDef) and f.name == "_load_w13" for f in n.body))
ns = {}
g = {"torch": torch}
for f in cls.body:
    if isinstance(f, ast.FunctionDef) and f.name in ("_load_w13", "_load_w2", "_narrow_expert_data_for_padding",
                                                     "_get_hidden_dim"):
        exec(ast.unparse(f), g, ns)
Loader = type("Loader", (), ns)
loader = Loader()
loader.moe_config = types.SimpleNamespace(is_act_and_mul=True,
                                          moe_parallel_config=types.SimpleNamespace(tp_size=8))

ok_w, ok_s = True, True
for rank in range(8):
    w1 = torch.randint(0, 256, (2304, 2560), dtype=torch.uint8)
    w3 = torch.randint(0, 256, (2304, 2560), dtype=torch.uint8)
    s1 = torch.randint(0, 256, (2304, 160), dtype=torch.uint8)
    s3 = torch.randint(0, 256, (2304, 160), dtype=torch.uint8)
    pw, cw = torch.zeros(768, 2560, dtype=torch.uint8), torch.zeros(576, 2560, dtype=torch.uint8)
    ps, cs = torch.zeros(768, 160, dtype=torch.uint8), torch.zeros(576, 160, dtype=torch.uint8)
    for sid, lw, ls in (("w1", w1, s1), ("w3", w3, s3)):
        loader._load_w13(pw, 0, sid, lw, rank)
        loader._load_w13(cw, 0, sid, lw, rank)
        loader._load_w13(ps, 0, sid, ls, rank)
        loader._load_w13(cs, 0, sid, ls, rank)
    live = torch.cat([torch.arange(0, 288), torch.arange(384, 672)])
    pad = torch.cat([torch.arange(288, 384), torch.arange(672, 768)])
    sl = slice(rank * 288, (rank + 1) * 288)
    ok_w &= torch.equal(cw, pw[live]) and int(pw[pad].abs().sum()) == 0 and torch.equal(cw[:288], w1[sl]) \
        and torch.equal(cw[288:], w3[sl])
    ok_s &= torch.equal(cs, ps[live]) and int(ps[pad].abs().sum()) == 0
check("L: live _load_w13, 8 TP ranks: compact w13 == padded w13 rows [0:288]+[384:672]; padded pad rows stay 0",
      ok_w)
check("L: same for the UE8M0 weight scales (compact == padded live rows; pad scale bytes 0)", ok_s)


# ---------------------------------------------------------------------------------------------- W: weight-scale pack
def pack_rows(sf_bytes):  # [E, N, Kb] ue8m0 -> int32 [E, ceil(Kb/4), N] MN-major (the FP32 path packs exponents)
    E, N, Kb = sf_bytes.shape
    P = -(-Kb // 4)
    padb = torch.zeros(E, N, P * 4, dtype=torch.int64)
    padb[:, :, :Kb] = sf_bytes.long()
    w = padb[..., 0::4] | (padb[..., 1::4] << 8) | (padb[..., 2::4] << 16) | (padb[..., 3::4] << 24)
    return w.transpose(1, 2).contiguous()


sfp = torch.randint(0, 256, (4, 768, 160), dtype=torch.uint8)
sfp[:, 288:384] = 0
sfp[:, 672:768] = 0
sfc = torch.cat([sfp[:, :288], sfp[:, 384:672]], dim=1)
pp, pc = pack_rows(sfp), pack_rows(sfc)
check("W: packed weight-scale words are per row: compact columns == padded live columns",
      torch.equal(pc, torch.cat([pp[:, :, :288], pp[:, :, 384:672]], dim=2)))

# ---------------------------------------------------------------------------------------------- F1 + S + F2 + gather
LIVE = torch.cat([torch.arange(0, 288), torch.arange(384, 672)])
PADC = torch.cat([torch.arange(288, 384), torch.arange(672, 768)])


def make_case(kind, R=12):
    K = 5120
    a = rand_fp8((R, K))
    sfa = torch.randint(120, 131, (R, 40), dtype=torch.uint8)
    wl = torch.randint(0, 256, (576, K // 2), dtype=torch.uint8)
    sfl = torch.randint(118, 128, (576, 160), dtype=torch.uint8)
    if kind == "max":
        a = torch.where(torch.rand(R, K) < 0.5, torch.tensor(0x7E, dtype=torch.uint8), torch.tensor(0xFE, dtype=torch.uint8))
        wl = torch.where(torch.rand(576, K // 2) < 0.5, torch.tensor(0x77, dtype=torch.uint8), torch.tensor(0xFF, dtype=torch.uint8))
        sfa[:] = 127 + 4
        sfb_hi = 127 + 4
        sfl[:] = sfb_hi
    elif kind == "zeros":
        a[: R // 2] = 0
        wl[:100] = 0
    elif kind == "negzero":  # -0 products everywhere in some rows/cols: negative a, zero / -0 weights
        a[:] = 0xB8  # -1.0
        wl[:200] = 0x88  # two -0 codes
    elif kind == "cancel":  # pairs +x, -x in every chunk -> exact 0 sums
        a[:] = 0x38  # +1.0
        lo = torch.randint(1, 8, (576, K // 2), dtype=torch.uint8)
        wl = lo | ((lo | 8) << 4)  # (+v, -v) pairs
    elif kind == "tiny":  # denormal-range scales: fp32 underflow / signed-zero rounding
        sfa[:] = 1
        sfl[:] = 1
    elif kind == "ties":  # many equal magnitudes -> absmax ties
        a[:] = torch.where(torch.rand(R, K) < 0.5, torch.tensor(0x40, dtype=torch.uint8), torch.tensor(0xC0, dtype=torch.uint8))
        wl[:] = 0x22
    wp = torch.zeros(768, K // 2, dtype=torch.uint8)
    sfp_ = torch.zeros(768, 160, dtype=torch.uint8)
    wp[LIVE] = wl
    sfp_[LIVE] = sfl
    return a, sfa, wp, sfp_, wl, sfl


all_ok = True
for kind in ("random", "random2", "max", "zeros", "negzero", "cancel", "tiny", "ties"):
    a, sfa, wp, sfp_, wl, sfl = make_case(kind)
    d_pad = gemm_model(a, sfa, wp, sfp_)
    d_cmp = gemm_model(a, sfa, wl, sfl)
    f1_ok = same_bits(d_cmp, d_pad[:, LIVE])
    pad_zero = bool((bits(d_pad[:, PADC]) == 0).all())
    s_ok = True
    for clamp in (None, 10.0):
        q_p, s_p = silu_writer(d_pad, None, clamp)
        q_c, s_c = silu_writer(d_cmp, 384, clamp)
        s_ok &= same_bits(q_p, q_c) and same_bits(s_p, s_c)
    # FC2 (K=384, cols 288..383 zero) on the two FC2 inputs, subset of 256 output columns; then gather
    w2 = torch.randint(0, 256, (256, 192), dtype=torch.uint8)
    w2[:, 144:] = 0
    sfb2 = torch.randint(118, 128, (256, 12), dtype=torch.uint8)
    sfb2[:, 9:] = 0
    q_p, s_p = silu_writer(d_pad, None, 10.0)
    q_c, s_c = silu_writer(d_cmp, 384, 10.0)
    o_p = gemm_model(q_p, s_p, w2, sfb2)
    o_c = gemm_model(q_c, s_c, w2, sfb2)
    inv = torch.randperm(12)[:9].view(3, 3).to(torch.int32)
    tw = torch.rand(3, 3, dtype=torch.float32)
    g_ok = same_bits(gather_model(o_p, inv, tw), gather_model(o_c, inv, tw)) and same_bits(o_p, o_c)
    nz = int((bits(d_pad[:, LIVE]) == torch.tensor(-32768, dtype=torch.int16)).sum())  # -0.0 bf16 count (info)
    ok = f1_ok and pad_zero and s_ok and g_ok
    all_ok &= ok
    check(f"F1+S+F2+gather [{kind}]: compact FC1 cols == padded live cols; padded pad cols are +0 (0x0000); FC2 "
          f"input bytes + scales identical (clamp none/10); FC2 + gather identical", ok,
          f"f1={f1_ok} pad+0={pad_zero} silu={s_ok} fc2+gather={g_ok}; live -0.0 outputs: {nz}")

# hostile SiLU-only inputs straight in bf16 (huge, tiny, denormal, ties, signed zeros) through both writers
R = 64
live = torch.randn(R, 576) * torch.logspace(-40, 18, R).view(R, 1)
live[0] = 0.0
live[1] = -0.0
live[2, :] = 3.0
live[3, ::2] = -3.0
live[4] = torch.finfo(torch.bfloat16).tiny / 8  # bf16 denormals
live[5] = 1.0e18
live = live.to(torch.bfloat16)
padded = torch.zeros(R, 768, dtype=torch.bfloat16)
padded[:, LIVE] = live
okh = True
for clamp in (None, 10.0):
    q_p, s_p = silu_writer(padded, None, clamp)
    q_c, s_c = silu_writer(live, 384, clamp)
    okh &= same_bits(q_p, q_c) and same_bits(s_p, s_c)
check("S [hostile bf16 inputs: 1e-40..1e18, +/-0, ties, denormals, 1e18]: identical FP8 bytes and UE8M0 scales",
      okh)
# the one condition the compaction relies on: the padded FC1 writes +0 (not -0) into its pad columns
padded_m0 = padded.clone()
padded_m0[:, PADC] = -0.0
q_m, _ = silu_writer(padded_m0, None, None)
q_c, _ = silu_writer(live, 384, None)
diff = int((q_m != q_c).sum())
check("S [control]: if the padded FC1 ever produced -0 in its pad columns the FC2 input would differ (0x80 vs 0x00) "
      "-> the device test asserts the pad columns are +0", diff > 0, f"{diff} bytes differ in the -0 control")


# ---------------------------------------------------------------------------------------------- P: permute
def pack_word(v):  # 4 fp32 scale bit patterns -> packed int32 (transpose_and_pack / mfront formula), uint32 math
    v = [x & 0xFFFFFFFF for x in v]
    return ((v[0] >> 23) | (v[1] >> 15) | (v[2] >> 7) | (v[3] << 1)) & 0xFFFFFFFF


def ue8m0_scales(M, sf_k=40):
    e = torch.randint(100, 150, (M, sf_k), dtype=torch.int64)
    return (e << 23).to(torch.int32)  # fp32 bit patterns of 2^(e-127): zero sign and mantissa


def stock_permute(ids, E, align, H_words, aq_words, sf_bits, order):
    M, topk = ids.shape
    cnt = torch.zeros(E, dtype=torch.int64)
    for e in ids.flatten().tolist():
        if 0 <= e < E:
            cnt[e] += 1
    aligned = (cnt + align - 1) // align * align
    start = torch.cumsum(aligned, 0) - aligned
    M_sum = -(-(M * topk + min(M * topk, E) * (align - 1)) // align) * align
    layout = torch.full((M_sum,), -1, dtype=torch.int32)
    for e in range(E):
        layout[start[e]:start[e] + cnt[e]] = e
    out = torch.zeros(M_sum, H_words, dtype=torch.int32)
    sf_out = torch.zeros(M_sum, sf_bits.shape[1], dtype=torch.int32)  # torch.zeros fp32 bit patterns
    inv = torch.empty(M, topk, dtype=torch.int32)
    cur = start.clone()
    for s in order:  # atomic arrival order
        t, k = divmod(s, topk)
        e = int(ids[t, k])
        if 0 <= e < E:
            d = int(cur[e])
            cur[e] += 1
            inv[t, k] = d
            out[d] = aq_words[t]
            sf_out[d] = sf_bits[t]
    P = -(-sf_bits.shape[1] // 4)
    packed = torch.zeros(M_sum, P, dtype=torch.int64)  # transpose_and_pack inside FC1
    for r in range(M_sum):
        row = sf_out[r].tolist() + [0] * (P * 4 - sf_out.shape[1])
        for p in range(P):
            packed[r, p] = pack_word(row[4 * p:4 * p + 4])
    return out, packed, layout, inv, cur, M_sum


def fused_permute(ids, E, align, H_words, aq_words, sf_bits, order):
    M, topk = ids.shape
    M_sum = -(-(M * topk + min(M * topk, E) * (align - 1)) // align) * align
    P = -(-sf_bits.shape[1] // 4)
    layout = torch.full((M_sum,), -1, dtype=torch.int32)  # plan: fill
    packed = torch.zeros(M_sum, P, dtype=torch.int64)     # plan: zero packed scales
    flat = ids.flatten()
    valid = (flat >= 0) & (flat < E)
    cnt = torch.bincount(torch.where(valid, flat, torch.tensor(E)).long(), minlength=E + 1)[:E]  # histogram
    aligned = (cnt + align - 1) // align * align
    cursor = torch.cumsum(aligned, 0) - aligned
    out = torch.zeros(M_sum, H_words, dtype=torch.int32)  # (poisoned below: rows never written stay garbage)
    out[:] = 0x5A5A5A5A
    inv = torch.empty(M, topk, dtype=torch.int32)
    for s in order:
        t, k = divmod(s, topk)
        e = int(ids[t, k])
        if 0 <= e < E:
            d = int(cursor[e])
            cursor[e] += 1
            inv[t, k] = d
            layout[d] = e
            out[d] = aq_words[t]
            row = sf_bits[t].tolist() + [0] * (P * 4 - sf_bits.shape[1])
            for p in range(P):
                packed[d, p] = pack_word(row[4 * p:4 * p + 4])
    return out, packed, layout, inv, cursor, M_sum


def topk_cases():
    E = 128
    for M in (1, 2, 3, 5, 8, 16, 32, 64, 96, 128, 160):
        yield f"random M={M}", torch.stack([torch.randperm(E)[:3] for _ in range(M)]).to(torch.int32)
    yield "all tokens on experts 0,1,2 M=160", torch.tensor([[0, 1, 2]] * 160, dtype=torch.int32)
    yield "edge experts 127,126,0 M=40", torch.tensor([[127, 126, 0]] * 40, dtype=torch.int32)
    all_e = torch.arange(160 * 3) % 128
    yield "M=160 touching all 128 experts", all_e.view(160, 3).to(torch.int32)
    ids = torch.stack([torch.randperm(E)[:3] for _ in range(20)]).to(torch.int32)
    ids[3, 1] = -1
    ids[7, 0] = -1
    yield "M=20 with -1 (invalid) slots", ids


H_words = 64  # row payload (word copies are dtype-agnostic; 5120 B rows = 1280 words, a 64-word slice suffices)
perm_ok = True
for name, ids in topk_cases():
    M = ids.shape[0]
    aq_words = torch.randint(-2 ** 31, 2 ** 31 - 1, (M, H_words), dtype=torch.int64).to(torch.int32)
    sf_bits = ue8m0_scales(M)
    slots = list(range(M * 3))
    o1 = torch.randperm(len(slots)).tolist()
    o2 = torch.randperm(len(slots)).tolist()
    s_out, s_pk, s_lay, s_inv, s_end, M_sum = stock_permute(ids, 128, 32, H_words, aq_words, sf_bits, o1)
    f_out, f_pk, f_lay, f_inv, f_end, M_sum2 = fused_permute(ids, 128, 32, H_words, aq_words, sf_bits, o2)
    ok = M_sum == M_sum2 and torch.equal(s_lay, f_lay) and torch.equal(s_end, f_end)
    for t in range(M):
        for k in range(3):
            if 0 <= int(ids[t, k]) < 128:
                ds, df = int(s_inv[t, k]), int(f_inv[t, k])
                ok &= torch.equal(s_out[ds], f_out[df]) and torch.equal(s_pk[ds], f_pk[df])
    dead = s_lay < 0
    ok &= bool((s_pk[dead] == 0).all()) and bool((f_pk[dead] == 0).all())
    # downstream: any per-row function of (row, packed scales) gathered through inv in fixed top-k order
    rowf = lambda out, pk: (out.to(torch.float64).sum(1) * 1e-9 + pk.to(torch.float64).sum(1)).to(torch.float32)
    ys, yf = rowf(s_out, s_pk), rowf(f_out, f_pk)
    w = torch.rand(M, 3, dtype=torch.float32)
    acc_s = torch.zeros(M, dtype=torch.float32)
    acc_f = torch.zeros(M, dtype=torch.float32)
    for k in range(3):
        m = ids[:, k] >= 0
        acc_s = torch.where(m, acc_s + ys[s_inv[:, k].clamp(min=0).long()] * w[:, k], acc_s)
        acc_f = torch.where(m, acc_f + yf[f_inv[:, k].clamp(min=0).long()] * w[:, k], acc_f)
    ok &= same_bits(acc_s, acc_f)
    perm_ok &= ok
    if not ok:
        print("   mismatch:", name)
check("P: stock vs fused permute, E=128 top-3 align 32, M=1..160 + hostile routings, independent atomic orders: "
      "same grouped_layout, expert ends (LIVE_ACT), M_sum; per slot identical row bytes and packed scale words; dead "
      "rows' scale words 0; gathered output identical", perm_ok)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
