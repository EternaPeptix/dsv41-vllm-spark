#!/usr/bin/env python3
"""Extract OBLITERATUS's per-layer refusal directions for DeepSeek-V4.1-Flash-1017 by diffing its BF16 tensors against the
official native checkpoint (HTTP Range fetches, nothing downloaded in full). Uses obl_probe.py's helpers.

usage: python3 obl_dirs.py --out DIR
For layers 1-39: direction from the router (BF16 both sides, reader), wq_a (FP8 native, reader) and shared w2 (FP8
native, writer); requires pairwise |cos| >= 0.999 and fitted alpha within 1 +- 0.01, then averages the sign-aligned
three. Layer 0 must be unedited (router delta == 0). Head: reader direction of lm_head.
Writes DIR/obliteratus_dirs.npz (float32 unit vectors, keys L1..L39 and head) and DIR/obliteratus_dirs.json (per-layer
fit stats, cosines, source repo revisions, sha256 of the npz).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obl_probe as P  # noqa: E402


def model_sha(repo):
    return json.loads(P.urllib.request.urlopen(P.urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}", headers=P.UA), timeout=60).read())["sha"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=",".join(str(i) for i in range(40)))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wo, wf = P.index(P.OBL), P.index(P.OFF)
    dirs, stats, bad = {}, {}, []

    def one(nat, obl, kind):
        nb = P.native(wf, nat)["v"]
        buf, _, shape = P.raw(P.OBL, wo, obl)
        ob = P.orient(nb, P.bf16(buf, shape))
        if not np.any(ob != nb):
            return None, {"identical": True}
        r, d = P.fit(nb, ob, kind)
        return d, r

    for L in [int(x) for x in a.layers.split(",")]:
        p, q = f"layers.{L}.", f"model.layers.{L}."
        parts = {
            "router": one(p + "ffn.gate", q + "mlp.gate.weight", "reader"),
            "wq_a": one(p + "attn.wq_a", q + "self_attn.q_a_proj.weight", "reader"),
            "shared_w2": one(p + "ffn.shared_experts.w2", q + "mlp.shared_experts.down_proj.weight", "writer"),
        }
        st = {k: v[1] for k, v in parts.items()}
        if all(v[0] is None for v in parts.values()):
            stats[f"L{L}"] = {"edited": False, **st}
            print(f"L{L:2d}: unedited (identical to official)", flush=True)
            continue
        if any(v[0] is None for v in parts.values()):
            bad.append(f"L{L}: some tensors identical, some edited")
            stats[f"L{L}"] = {"edited": "partial", **st}
            print(f"L{L:2d}: PARTIAL edit {st}", flush=True)
            continue
        ds = [v[0] for v in parts.values()]
        ref = ds[0]
        ds = [d if d @ ref >= 0 else -d for d in ds]
        cos = [float(abs(ds[i] @ ds[j])) for i in range(3) for j in range(i + 1, 3)]
        alphas = [v[1]["alpha"] for v in parts.values()]
        ok = min(cos) >= 0.999 and all(abs(x - 1) <= 0.01 for x in alphas)
        d = np.mean(ds, 0)
        d /= np.linalg.norm(d)
        dirs[f"L{L}"] = d.astype(np.float32)
        stats[f"L{L}"] = {"edited": True, "ok": ok, "min_cos": min(cos), "alphas": alphas,
                          "resid": [v[1]["resid"] for v in parts.values()]}
        if not ok:
            bad.append(f"L{L}: min_cos {min(cos):.4f} alphas {alphas}")
        print(f"L{L:2d}: min|cos| {min(cos):.5f}  alpha {min(alphas):.4f}..{max(alphas):.4f}  "
              f"resid {max(stats[f'L{L}']['resid']):.3f}  {'ok' if ok else 'CHECK'}", flush=True)
    dh, rh = one("head", "lm_head.weight", "reader")
    if dh is not None:
        dirs["head"] = (dh / np.linalg.norm(dh)).astype(np.float32)
    stats["head"] = rh
    print(f"head: {rh}", flush=True)
    path = os.path.join(a.out, "obliteratus_dirs.npz")
    np.savez(path, **dirs)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = {"source_obliterated": {"repo": P.OBL, "sha": model_sha(P.OBL)},
            "source_official": {"repo": P.OFF, "sha": model_sha(P.OFF)},
            "edit": "W' = W (I - d d^T) for readers (attn wq_a, wkv; router; routed+shared w1/w3; lm_head) and "
                    "W' = (I - d d^T) W for writers (attn wo_b; routed+shared w2); one unit d per layer, layers 1-39 "
                    "plus a head direction; layer 0, norms, embeddings, Engram, MTP untouched",
            "npz": os.path.basename(path), "npz_sha256": sha, "keys": sorted(dirs), "problems": bad, "stats": stats}
    json.dump(meta, open(os.path.join(a.out, "obliteratus_dirs.json"), "w"), indent=1, default=float)
    print(f"wrote {path} sha256 {sha[:16]}  keys {len(dirs)}  problems {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
