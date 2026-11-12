#!/usr/bin/env python3
"""r2-attnout device test (lever G19): live attention-output kernels vs the kit's kernels, bitwise + timing.

Run by the lead on a STOPPED node (no vLLM process on the GPU), e.g. spark-eight, kit NOT installed needed:
    cd <kit dir> && ~/venvs/vllm032/bin/python device_test.py [--variants all] [--iters 200] [--ckpt DIR --layer 0 --rank 0]
Expected runtime: ~3-6 min (Triton JIT of the new kernels ~30 s, 9 variants x 32 M x 4 input sets bitwise,
timing at 8 M values x 3 modes).

Live side (imported from the installed venv, unchanged): fused_inv_rope_fp8_quant (quantize=False) ->
sonar _sm120_wo_a_partial (num_warps 4, num_stages 2, fp fusion off) -> _sm120_wo_a_reduce -> flashinfer
mxfp8_quantize (cute-dsl, swizzled 128x4), i.e. exactly what deep_gemm_fp8_o_proj + wo_b's quant run today
(DSV41_SONAR_OPROJ=fp8, SPLITS=32). Kit side: files/models/deepseek_v4/nvidia/ops/dsv41_attnout.py loaded by
path once per variant (its module flags are read at import).

Inputs: weights = an exact MXFP8 re-encoding (sonar.mxfp8_requant_exact, as production builds it) of either
synthetic MXFP8-dequantised bf16 weights or, with --ckpt, the checkpoint's layers.<L>.attn.wo_a group <rank>;
activations o[M, 8, 512] bf16 for every M = 1..32 (the sonar range; target verify M=1..6 and DSpark M=5 are
inside it) in four sets: normal, hostile (wide exponent range, +-0, bf16 denormals, large), rope-hostile
(positions at 0 / max / identity rows), and an rq FTZ-corner set (32-blocks whose amax/448 is an fp32 denormal;
flashinfer's CuTe-DSL multiply may flush it -- reported separately).
Compares raw bytes (torch.equal on uint8 views) of: the fp32 partial buffer (all splits), z (bf16) when the
reduce is live, and (e4m3 bytes, UE8M0 bytes) when rq is on. Prints the first differing byte index.
Timing: per chain (inverse RoPE .. quant, wo_b GEMM excluded), median of --iters runs: warm (eager, CUDA
events), cold-L2 (64 MiB write between runs), and graph (one CUDA graph of 40 chains, replay time / 40, which
includes the inter-kernel graph gaps the serving path sees).
Exit code 0 only if every compared case is bit-identical (FTZ-corner cases included; see the printout).
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys

import torch

KIT = os.environ.get("DSV41_ATTNOUT_KIT") or os.path.dirname(os.path.abspath(__file__))
MOD = os.path.join(KIT, "files/models/deepseek_v4/nvidia/ops/dsv41_attnout.py")
VARIANTS = ["rope", "rq", "st1", "vs", "bn128", "st1,vs", "st1,vs,bn128", "rope,rq", "rope,rq,st1,vs",
            "rope,rq,st1,vs,bn128"]
G, HPG, HD, NOPE, ROPE_DIM, N = 1, 8, 512, 448, 64, 1024
K = HPG * HD
MAX_POS = 65536


def load_variant(v):
    os.environ["DSV41_ATTNOUT_OP"] = v
    spec = importlib.util.spec_from_file_location("dsv41_attnout_" + v.replace(",", "_"), MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.PARTS == m.parse(v)
    return m


def u8(t):
    return t.contiguous().view(torch.uint8)


def first_diff(a, b):
    a8, b8 = u8(a).flatten(), u8(b).flatten()
    idx = torch.nonzero(a8 != b8)
    return None if idx.numel() == 0 else (int(idx[0]), int(idx.numel()), a8.numel())


# ------------------------------------------------------------------ inputs
def rope_cache(dev):
    half = ROPE_DIM // 2
    inv = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float64) / half))
    ang = torch.arange(MAX_POS, dtype=torch.float64)[:, None] * inv[None, :]
    cs = torch.cat([ang.cos(), ang.sin()], 1).float()
    cs[1, :half], cs[1, half:] = 1.0, 0.0
    cs[2, :half] = -0.0
    return cs.to(dev)


def synthetic_weight(gen):
    q = (torch.randn((G, N, K), generator=gen) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    e = torch.randint(110, 125, (G, N, K // 32), generator=gen, dtype=torch.int32)
    w = (q.float().view(G, N, K // 32, 32) * torch.ldexp(torch.ones(()), e - 127)[..., None]).view(G, N, K)
    return w.to(torch.bfloat16)


def ckpt_weight(path, layer, rank):
    from safetensors import safe_open

    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    name = f"layers.{layer}.attn.wo_a.weight"
    with safe_open(os.path.join(path, idx[name]), "pt") as f:
        q = f.get_tensor(name)
    sname = f"layers.{layer}.attn.wo_a.scale"
    with safe_open(os.path.join(path, idx[sname]), "pt") as f:
        s = f.get_tensor(sname)
    rows, cols = q.shape
    sr, sc = s.shape
    br, bc = rows // sr, cols // sc
    sf = s.float() if s.dtype != torch.uint8 else torch.ldexp(torch.ones(()), s.to(torch.int32) - 127)
    w = q.float() * sf.repeat_interleave(br, 0).repeat_interleave(bc, 1)
    w = w.to(torch.bfloat16).view(-1, N, cols)[rank: rank + 1]  # group `rank` of o_groups (TP8: 1 group/rank)
    assert w.shape == (1, N, K), w.shape
    return w.contiguous()


def hostile_o(M, gen):
    x = torch.randn((M, G * HPG, HD), generator=gen) * torch.exp2(torch.randint(-24, 14, (M, G * HPG, HD),
                                                                              generator=gen).float())
    f = x.view(-1)
    for frac, val in ((40, 0.0), (40, -0.0), (80, 65280.0), (80, 2.0 ** -133), (80, -2.0 ** -130)):
        f[torch.randint(0, f.numel(), (max(1, f.numel() // frac),), generator=gen)] = val
    return x.to(torch.bfloat16)


def input_sets(M, gen, dev):
    sets = {}
    sets["normal"] = ((torch.randn((M, G * HPG, HD), generator=gen) * 0.05).to(torch.bfloat16),
                      torch.randint(0, MAX_POS, (M,), generator=gen))
    sets["hostile"] = (hostile_o(M, gen), torch.randint(0, MAX_POS, (M,), generator=gen))
    pos = torch.randint(0, MAX_POS, (M,), generator=gen)
    pos[0] = 0
    if M > 1:
        pos[1] = MAX_POS - 1
    if M > 2:
        pos[2] = 1
    if M > 3:
        pos[3] = 2
    sets["rope-hostile"] = (hostile_o(M, gen), pos)
    # FTZ corner for rq: tiny activations -> partial sums ~1e-38 -> 32-blocks with amax/448 below FLT_MIN
    sets["ftz-corner"] = ((torch.randn((M, G * HPG, HD), generator=gen) * 2.0 ** -118).to(torch.bfloat16),
                          torch.randint(0, MAX_POS, (M,), generator=gen))
    return {k: (o.to(dev), p.to(dev)) for k, (o, p) in sets.items()}


# ------------------------------------------------------------------ chains
class Live:
    def __init__(self, mod, sonar, wq, ws, cs):
        self.mod, self.sonar, self.wq, self.ws, self.cs = mod, sonar, wq, ws, cs

    def inv_rope(self, o, pos):
        from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

        out, _ = fused_inv_rope_fp8_quant(o, pos, self.cs, n_groups=G, heads_per_group=HPG, nope_dim=NOPE,
                                          rope_dim=ROPE_DIM, quant_group_size=128, tma_aligned_scales=False,
                                          quantize=False)
        return out

    def chain(self, o, pos):
        M = o.shape[0]
        splits = self.sonar._splits_for(M)
        a = self.inv_rope(o, pos)
        P = torch.empty((splits, G, M, N), device=o.device, dtype=torch.float32)
        self.mod._launch_partial(a, None, self.wq, self.ws, P, M, G, N, K, splits, True, new=False)
        z = torch.empty((M, G, N), device=o.device, dtype=torch.bfloat16)
        self.mod._launch_live_reduce(P, z, M, G, N, splits)
        q, sf = self.mod._live_quant(z.flatten(1))
        return dict(partial=P, z=z, q=q, sf=sf)


def new_chain(mod, live, o, pos):
    M = o.shape[0]
    splits = live.sonar._splits_for(M)
    if mod.ROPE:
        a, rope = o, (pos, live.cs, HPG, NOPE, ROPE_DIM // 2)
    else:
        a, rope = live.inv_rope(o, pos), None
    P = torch.empty((splits, G, M, N), device=o.device, dtype=torch.float32)
    mod._launch_partial(a, rope, live.wq, live.ws, P, M, G, N, K, splits, True,
                        new=bool(mod.ROPE or mod.TUNE))
    out = dict(partial=P)
    if mod.RQ:
        q, sf = mod._launch_rq(P, M, N, splits)
        out.update(q=q, sf=sf)
    else:
        z = torch.empty((M, G, N), device=o.device, dtype=torch.bfloat16)
        mod._launch_live_reduce(P, z, M, G, N, splits)
        q, sf = mod._live_quant(z.flatten(1))
        out.update(z=z, q=q, sf=sf)
    return out


# ------------------------------------------------------------------ timing
def time_eager(fn, iters, flush=None):
    ts = []
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        if flush is not None:
            flush.add_(1)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def time_graph(fn, iters, chains=40):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(chains):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(max(20, iters // 4)):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / chains)
    return statistics.median(ts)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="all")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ms", default="1-32")
    ap.add_argument("--time-ms", default="1,2,3,4,5,6,16,32")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    torch.cuda.set_device(0)
    a, b = args.ms.split("-") if "-" in args.ms else (args.ms, args.ms)
    Ms = list(range(int(a), int(b) + 1))
    tMs = [int(x) for x in args.time_ms.split(",") if x]
    variants = VARIANTS if args.variants == "all" else args.variants.split(";")

    os.environ.setdefault("DSV41_SONAR_OPROJ", "fp8")
    os.environ.setdefault("DSV41_SONAR_OPROJ_SPLITS", "32")
    os.environ.setdefault("DSV41_SONAR_OPROJ_MAX_M", "32")
    from vllm.models.deepseek_v4.nvidia.ops import sonar_bf16_bmm as sonar

    assert sonar._splits_for(1) == 32, "run with DSV41_SONAR_OPROJ_SPLITS=32 (production)"
    gen = torch.Generator().manual_seed(20261108)
    weights = {"synthetic": synthetic_weight(gen)}
    if args.ckpt:
        try:
            weights[f"ckpt-L{args.layer}-g{args.rank}"] = ckpt_weight(args.ckpt, args.layer, args.rank)
        except Exception as e:  # noqa: BLE001
            print(f"WARN checkpoint weight not loaded ({e!r}); synthetic only")
    cs = rope_cache(dev)
    base_mod = load_variant("rope")  # helper functions (live launches are variant independent)
    fails, total, ftz_fails = [], 0, []
    timings = {}
    for wname, w in weights.items():
        res = sonar.mxfp8_requant_exact(w.to(dev))
        assert res is not None, f"{wname}: weight is not an exact MXFP8 re-encoding"
        wq, ws = res
        live = Live(base_mod, sonar, wq, ws, cs)
        for v in variants:
            # A variant that fails to load / compile / launch is recorded as a failure (exit 1) and the run goes
            # on, so one broken part does not hide the results of the others.
            try:
                mod = load_variant(v)
                for M in Ms:
                    for sname, (o, pos) in input_sets(M, gen, dev).items():
                        ref = live.chain(o, pos)
                        got = new_chain(mod, live, o, pos)
                        keys = ["partial"] + (["q", "sf"] if mod.RQ else ["z", "q", "sf"])
                        for key in keys:
                            total += 1
                            d = first_diff(got[key], ref[key])
                            if d is not None:
                                rec = (wname, v, M, sname, key, d)
                                (ftz_fails if sname == "ftz-corner" and key in ("q", "sf") else fails).append(rec)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                fails.append((wname, v, "ERROR", repr(e)))
                print(f"[bitwise] weights={wname} variant={v}: ERROR {e!r} (variant counted as failed)", flush=True)
                continue
            print(f"[bitwise] weights={wname} variant={v}: M={Ms[0]}..{Ms[-1]} x 4 sets done; "
                  f"mismatches so far {len(fails)} (+{len(ftz_fails)} ftz-corner)", flush=True)
            if args.no_timing or wname != "synthetic":
                continue
            flush = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
            for M in tMs:
                o, pos = input_sets(M, gen, dev)["normal"]
                for side, fn in (("live", lambda: live.chain(o, pos)), (v, lambda: new_chain(mod, live, o, pos))):
                    key = (side, M)
                    if key in timings:
                        continue
                    timings[key] = (time_eager(fn, args.iters), time_eager(fn, args.iters, flush),
                                    time_graph(fn, args.iters))
    print("\n=== bitwise ===")
    print(f"compared {total} (variant, weights, M, input set, tensor) cases")
    bad_variants = sorted({f[1] for f in fails} | {f[1] for f in ftz_fails})
    print("variants with a mismatch or error:", ", ".join(bad_variants) if bad_variants else "none")
    for f in fails:
        print("MISMATCH", f)
    for f in ftz_fails:
        print("MISMATCH ftz-corner (CuTe-DSL multiply FTZ semantics; real activations never reach it)", f)
    if not fails and not ftz_fails:
        print("ALL BIT-IDENTICAL")
    if timings:
        print("\n=== timing per chain, us (median): warm eager / cold-L2 eager / in-graph (40 chains) ===")
        for M in tMs:
            lw = timings.get(("live", M))
            if lw is None:
                continue
            print(f"M={M:2d} live            {lw[0]:7.2f} {lw[1]:7.2f} {lw[2]:7.2f}")
            for v in variants:
                t = timings.get((v, M))
                if t:
                    print(f"M={M:2d} {v:22s}{t[0]:7.2f} {t[1]:7.2f} {t[2]:7.2f}   graph delta {t[2] - lw[2]:+6.2f}")
    sys.exit(0 if not fails and not ftz_fails else 1)


if __name__ == "__main__":
    main()
