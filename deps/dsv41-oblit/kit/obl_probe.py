#!/usr/bin/env python3
"""Does OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated (BF16) differ from deepseek-ai/DeepSeek-V4.1-Flash-1017 (native
FP8/FP4) by clean rank-1 projections?  Fetches single tensors with HTTP Range requests (no full download).

usage: python3 -I obl_probe.py --out DIR [--layers 5,38] [--head]
For each tensor pair: base = dequantised native tensor, obl = BF16 tensor, delta = obl - base.
  writer (output dim 5120: attention wo_b, shared w2, expert w2):  failspy  W' = (I - d d^T) W   ->  delta = -d (d^T W)
  reader (input dim 5120: wq_a, wkv, router, shared w1/w3, expert w1/w3, head): W' = W (I - d d^T) -> delta = -(W d) d^T
Reports |delta|/|W|, the top-2 singular values of delta, the residual of the best rank-1 projection fit, the fitted
scale alpha (1.0 = exact projection), and |cos| between the directions found in different tensors of a layer.
Saves the directions as .npy in --out.
"""
import argparse
import json
import os
import struct
import sys
import time
import urllib.request

import numpy as np

OBL = "OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated"
OFF = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
UA = {"User-Agent": "obl-probe/1.0"}
_hdr = {}


def get(repo, path, start=None, end=None):
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    h = dict(UA)
    if start is not None:
        h["Range"] = f"bytes={start}-{end}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=600) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == 5:
                raise
            print(f"  retry {path} {start}: {e!r}"[:200], file=sys.stderr)
            time.sleep(3 * (attempt + 1))


def index(repo):
    return json.loads(get(repo, "model.safetensors.index.json"))["weight_map"]


def header(repo, shard):
    k = (repo, shard)
    if k not in _hdr:
        n = struct.unpack("<Q", get(repo, shard, 0, 7))[0]
        _hdr[k] = (json.loads(get(repo, shard, 8, 8 + n - 1)), 8 + n)
    return _hdr[k]


def raw(repo, wm, name, first_of_dim0=False):
    shard = wm[name]
    h, base = header(repo, shard)
    meta = h[name]
    a, b = meta["data_offsets"]
    shape = list(meta["shape"])
    if first_of_dim0:  # only element 0 along dim 0 (one expert of a fused expert tensor)
        b = a + (b - a) // shape[0]
        shape = shape[1:]
    return get(repo, shard, base + a, base + b - 1), meta["dtype"], shape


def bf16(buf, shape):
    u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(shape)


def lut_e4m3():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        v = np.nan if (e == 15 and m == 7) else ((m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
        t[b] = -v if s else v
    return t


E4M3 = lut_e4m3()
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e8m0(buf, shape):
    return np.exp2(np.frombuffer(buf, dtype=np.uint8).astype(np.float32) - 127).reshape(shape)


def deq_fp8(w, s):
    """w [R, C] e4m3 bytes, s [R/br, C/bc] e8m0 -> f32 (block sizes from the shapes)."""
    R, C = w.shape
    br, bc = R // s.shape[0], C // s.shape[1]
    return E4M3[w] * np.repeat(np.repeat(s, br, 0), bc, 1)


def deq_fp4(w, s, low_first=True):
    """w [R, C/2] packed e2m1, s [R, C/32] e8m0 -> f32 [R, C]."""
    lo, hi = E2M1[w & 15], E2M1[w >> 4]
    v = np.stack([lo, hi] if low_first else [hi, lo], -1).reshape(w.shape[0], -1)
    return v * np.repeat(s, v.shape[1] // s.shape[1], 1)


def native(wm, name):
    """dequantised native tensor (BF16 / FP8+scale / FP4+scale)."""
    buf, dt, shape = raw(OFF, wm, name + ".weight")
    if dt == "BF16":
        return {"v": bf16(buf, shape)}
    sbuf, _, sshape = raw(OFF, wm, name + ".scale")
    s = e8m0(sbuf, sshape)
    w = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
    if dt == "F8_E4M3":
        return {"v": deq_fp8(w, s)}
    if dt == "I8":
        return {"lo": deq_fp4(w, s, True), "hi": deq_fp4(w, s, False)}
    raise ValueError(dt)


def top_sv(D, k=2, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    out, Dw = [], D.astype(np.float64)
    for _ in range(k):
        v = rng.standard_normal(Dw.shape[1])
        for _ in range(iters):
            v = Dw.T @ (Dw @ v)
            v /= np.linalg.norm(v) + 1e-300
        u = Dw @ v
        s = np.linalg.norm(u)
        u /= s + 1e-300
        out.append((s, u, v))
        Dw = Dw - s * np.outer(u, v)
    return out


def fit(base, obl, kind):
    """kind 'writer': rows are the 5120 residual dims; 'reader': columns are."""
    W, D = base.astype(np.float64), (obl - base).astype(np.float64)
    nW, nD = np.linalg.norm(W), np.linalg.norm(D)
    (s1, u1, v1), (s2, _, _) = top_sv(D, 2)
    if kind == "writer":
        d = u1
        dW = d @ W                          # d^T W
        alpha = -(d @ D @ dW) / (dW @ dW)   # D ~ -alpha d d^T W
        resid = np.linalg.norm(D + alpha * np.outer(d, dW)) / max(nD, 1e-300)
    else:
        d = v1
        Wd = W @ d
        alpha = -(Wd @ D @ d) / (Wd @ Wd)
        resid = np.linalg.norm(D + alpha * np.outer(Wd, d)) / max(nD, 1e-300)
    return {"rel_delta": nD / nW, "s1": s1, "s2_over_s1": s2 / max(s1, 1e-300), "alpha": alpha, "resid": resid}, d


def orient(base, obl):
    if base.shape == obl.shape:
        return obl
    if base.shape == obl.shape[::-1]:
        return obl.T
    raise ValueError(f"shape mismatch base {base.shape} obl {obl.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="5,38")
    ap.add_argument("--head", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wo, wf = index(OBL), index(OFF)
    rows, dirs = [], {}

    def pair(tag, nat_name, obl_name, kind, first=False, split=None, fp4=False):
        nb = native(wf, nat_name)
        buf, dt, shape = raw(OBL, wo, obl_name, first_of_dim0=first)
        ob = bf16(buf, shape)
        if split is not None:   # fused gate_up: take the gate (0) or up (1) half
            R, C = next(iter(nb.values())).shape
            if ob.shape == (2 * R, C):
                ob = ob[:R] if split == 0 else ob[R:]
            elif ob.shape == (C, 2 * R):
                ob = ob[:, :R] if split == 0 else ob[:, R:]
            else:
                print(f"{tag}: unexpected fused shape {ob.shape} for native {(R, C)}")
                return
        best = None
        for key, base in nb.items():
            try:
                o = orient(base, ob)
            except ValueError as e:
                print(f"{tag}: {e}")
                return
            r, d = fit(base, o, kind)
            r["fp4_order"] = key if fp4 else None
            if best is None or r["resid"] < best[0]["resid"]:
                best = (r, d)
        r, d = best
        rows.append((tag, kind, r))
        dirs[tag] = d
        np.save(os.path.join(a.out, tag.replace("/", "_") + ".npy"), d.astype(np.float32))
        print(f"{tag:26} {kind:6} |dW|/|W| {r['rel_delta']:.4f}  s2/s1 {r['s2_over_s1']:.4f}  alpha {r['alpha']:+.4f}  "
              f"rank1-resid {r['resid']:.4f}" + (f"  fp4 {r['fp4_order']}" if fp4 else ""), flush=True)

    for L in [int(x) for x in a.layers.split(",") if x]:
        p, q = f"layers.{L}.", f"model.layers.{L}."
        pair(f"L{L}/wo_b", p + "attn.wo_b", q + "self_attn.o_b_proj.weight", "writer")
        pair(f"L{L}/wq_a", p + "attn.wq_a", q + "self_attn.q_a_proj.weight", "reader")
        pair(f"L{L}/wkv", p + "attn.wkv", q + "self_attn.kv_proj.weight", "reader")
        pair(f"L{L}/router", p + "ffn.gate", q + "mlp.gate.weight", "reader")
        pair(f"L{L}/shared_w2", p + "ffn.shared_experts.w2", q + "mlp.shared_experts.down_proj.weight", "writer")
        pair(f"L{L}/shared_w1", p + "ffn.shared_experts.w1", q + "mlp.shared_experts.gate_proj.weight", "reader")
        pair(f"L{L}/shared_w3", p + "ffn.shared_experts.w3", q + "mlp.shared_experts.up_proj.weight", "reader")
        pair(f"L{L}/expert0_w2", p + "ffn.experts.0.w2", q + "mlp.experts.down_proj", "writer", first=True, fp4=True)
        pair(f"L{L}/expert0_w1", p + "ffn.experts.0.w1", q + "mlp.experts.gate_up_proj", "reader", first=True, split=0, fp4=True)
        for nn, on in ((p + "attn_norm.weight", q + "input_layernorm.weight"), (p + "ffn_norm.weight", q + "post_attention_layernorm.weight")):
            nbuf, _, ns = raw(OFF, wf, nn)
            obuf, _, os_ = raw(OBL, wo, on)
            nv, ov = bf16(nbuf, ns), bf16(obuf, os_)
            print(f"L{L} norm {on.split('.')[-2]:26} max|obl-native| {np.abs(ov - nv).max():.3g}  (obl mean {ov.mean():.4f}, native mean {nv.mean():.4f})")
        keys = [k for k in dirs if k.startswith(f"L{L}/")]
        cs = [(k1.split('/')[1], k2.split('/')[1], abs(float(dirs[k1] @ dirs[k2]))) for i, k1 in enumerate(keys) for k2 in keys[i + 1:]]
        print(f"L{L} |cos| between directions: min {min(c for *_, c in cs):.4f}  max {max(c for *_, c in cs):.4f}")
    if a.head:
        pair("head", "head", "lm_head.weight", "reader")
        nbuf, _, ns = raw(OFF, wf, "norm.weight")
        obuf, _, os_ = raw(OBL, wo, "model.norm.weight")
        print(f"final norm max|obl-native| {np.abs(bf16(obuf, os_) - bf16(nbuf, ns)).max():.3g}")
    Ls = sorted({k.split('/')[0] for k in dirs})
    if len(Ls) > 1:
        for t in ("wo_b", "router"):
            ks = [f"{l}/{t}" for l in Ls if f"{l}/{t}" in dirs]
            if len(ks) > 1:
                print(f"|cos| {t} across {ks}: {abs(float(dirs[ks[0]] @ dirs[ks[-1]])):.4f}")
    if "head" in dirs:
        for l in Ls:
            if l != "head" and f"{l}/wo_b" in dirs:
                print(f"|cos| head vs {l}/wo_b: {abs(float(dirs['head'] @ dirs[l + '/wo_b'])):.4f}")
    json.dump([{"tensor": t, "kind": k, **{x: float(v) if isinstance(v, (int, float, np.floating)) else v for x, v in r.items()}}
               for t, k, r in rows], open(os.path.join(a.out, "probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
