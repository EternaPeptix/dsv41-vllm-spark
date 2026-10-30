#!/usr/bin/env python3
"""g1/m32-bw GPU unit test: every variant BITWISE vs live (STOPPED-SERVICE WINDOW ONLY; refuses otherwise).

Arms (one process, each native loaded once from its own tree; see g1lib.py):
  live  serving trees + private copies of the serving JIT caches (FC1 b1c4eff8 tree, FC2 Codex D05 6fea5091 tree)
  off   patch mechanics only (SASS identical to live offline): an A/A control
  pair / pf / la / ds / all   the exact bandwidth variants (tools/make_tree.py)
Checks, all bitwise on uint8 views:
  * whole DeepGemmFP4Experts.apply as served (serving input quant, COMPACT FC1 + LIVE_ACT + D05 / g1 FC2), eager, with
    every workspace poisoned (0xFF = NaN in bf16 and fp8) before each run; live references must be finite;
  * per GEMM: FC1 and FC2 outputs on live rows (grouped_layout >= 0) vs live (localizes any mismatch);
  * CUDA graph captured once per (arm, M), then --replays replays cycling through all route patterns with the
    workspaces re-poisoned before each replay (races / stale-state are timing dependent: every replay compared).
Weights: synthetic random FP4 (all codes) and, with --real-layer L, one TP slice of checkpoint layer L.
Routes: valid / spec / half (dead -1 slots) / hot (multi-block experts) / skew / one, plus recorded 10-21 decode
routes (6-row windows and 4-user batches) when routes/ is present.
Usage: source ~/venvs/vllm032/bin/activate; python ~/dsv41-stage/g1/m32-bw/tests/test_m32bw_unit.py
          [--variants off,pair,pf,la,ds,all] [--tokens 1,2,...,96] [--replays 60] [--real-layer 0|-1]
Sanitizer (slow, once, small case; -x = also check the variants' extra producer warp):
  compute-sanitizer --tool racecheck python .../test_m32bw_unit.py --variants la,all --tokens 6 --replays 2 --real-layer -1 --force-gpu-shared
  compute-sanitizer --tool synccheck python .../test_m32bw_unit.py --variants la,all --tokens 6 --replays 2 --real-layer -1 --force-gpu-shared
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1lib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="off,pair,pf,la,ds,all")
ap.add_argument("--tokens", default="1,2,3,4,5,6,8,12,16,24,32,48,64,96")
ap.add_argument("--replays", type=int, default=60)
ap.add_argument("--real-layer", type=int, default=0, help="checkpoint layer for real weights (-1: synthetic only)")
ap.add_argument("--tp-rank", type=int, default=0)
ap.add_argument("--routes", default=str(g1lib.G / "routes"))
ap.add_argument("--force", action="store_true")
ap.add_argument("--force-gpu-shared", action="store_true", help="skip only the GPU-busy check (sanitizer runs)")
a = ap.parse_args()
g1lib.guard(a.force, a.force_gpu_shared)
g1lib.setup_env(g1lib.G / "work")

import torch  # noqa: E402

variants = [v for v in a.variants.split(",") if v]
assert all(v in g1lib.VARIANTS for v in variants), variants
nat = g1lib.Natives(variants)
g1lib.install_live_getter(nat)
T0 = time.time()
fails = cases = 0


def rec(ok, **kw):
    global fails, cases
    cases += 1
    fails += not ok
    print(json.dumps(dict(ok=bool(ok), **kw)), flush=True)


@contextlib.contextmanager
def capture(kernel):
    """Record (N, output clone, grouped_layout clone) of every grouped GEMM of one eager apply."""
    calls = []
    orig = kernel.gemm

    def g(*args, **kw):
        r = orig(*args, **kw)
        (_aq, _sfa), (_bw, _sfb), d, layout = args
        calls.append((int(d.shape[1]), d.clone(), layout.clone()))
        return r
    kernel.gemm = g
    try:
        yield calls
    finally:
        del kernel.gemm


def poison(ws):
    for t in ws:
        t.view(torch.uint8).fill_(255)


def same(x, y):
    return x.shape == y.shape and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


_REC = {}


def recorded(m, gen):
    p = Path(a.routes)
    if not p.exists():
        return {}
    if "x" not in _REC:
        _REC["x"] = g1lib.load_recorded(p)
    wins, c4 = _REC["x"]
    out = {}
    if wins:
        layer = int(torch.randint(0, 40, (1,), generator=gen))
        if m == 1:
            out["rec1"] = wins[0][layer, :1]
        elif m % 6 == 0:
            idx = torch.randperm(len(wins), generator=gen).tolist()
            out["rec6x"] = torch.cat([wins[idx[i % len(idx)]][layer] for i in range(m // 6)], 0)
    if m in c4:
        out["rec_c4"] = c4[m][0][int(torch.randint(0, 40, (1,), generator=gen))]
    return out


weight_sets = [("synthetic", g1lib.synthetic_weights())]
if a.real_layer >= 0 and g1lib.MODEL.exists():
    t0 = time.time()
    weight_sets.append((f"layer{a.real_layer}r{a.tp_rank}", g1lib.real_layer(a.real_layer, a.tp_rank)))
    print(json.dumps(dict(event="real_layer_loaded", layer=a.real_layer, tp_rank=a.tp_rank,
                          s=round(time.time() - t0, 1))), flush=True)

for wname, (w1c, s1c, w2, s2) in weight_sets:
    arms = {n: g1lib.Arm(k, w1c, s1c, w2, s2) for n, k in nat.kernels.items()}
    for m in [int(t) for t in a.tokens.split(",")]:
        gen = torch.Generator().manual_seed(1000 + m)
        x = (torch.randn(m, g1lib.H, generator=gen) * 0.15).to(torch.bfloat16).cuda()
        tw = (torch.softmax(torch.randn(m, g1lib.TOPK, generator=gen), -1) * 1.5).float().cuda()
        routes = g1lib.synthetic_routes(m, gen)
        routes.update(recorded(m, gen))
        routes = {k: v.cuda() for k, v in routes.items()}
        ids = torch.empty(m, g1lib.TOPK, dtype=torch.int32, device="cuda")
        live = arms["live"]
        refs, ref_gemm = {}, {}
        for r, v in routes.items():
            ids.copy_(v)
            ws = live.workspaces(m)
            poison(ws)
            with capture(live.ex._dsv41_target_m32) as calls:
                out = live.run(x, ids, tw, ws).clone()
            torch.cuda.synchronize()
            refs[r], ref_gemm[r] = out, calls
            live_tok = (v >= 0).any(1)            # fully dead tokens may legitimately keep workspace bytes
            rec(bool(torch.isfinite(out[live_tok].float()).all()), w=wname, m=m, arm="live", route=r,
                mode="ref_finite")
        for name in variants:
            arm = arms[name]
            ws = arm.workspaces(m)
            for r, v in routes.items():                     # eager, poisoned workspaces
                ids.copy_(v)
                poison(ws)
                with capture(arm.ex._dsv41_target_m32) as calls:
                    got = arm.run(x, ids, tw, ws)
                torch.cuda.synchronize()
                detail = {}
                for i, ((n0, d0, l0), (n1, d1, l1)) in enumerate(zip(ref_gemm[r], calls)):
                    tag = {576: "fc1", 5120: "fc2"}.get(n0, f"n{n0}")
                    live_rows = l0 >= 0
                    detail[tag] = bool(n0 == n1 and same(l0, l1) and same(d0[live_rows], d1[live_rows]))
                ok = same(got, refs[r]) and len(calls) == len(ref_gemm[r]) and all(detail.values())
                rec(ok, w=wname, m=m, arm=name, route=r, mode="eager", gemm=detail,
                    live_rows=int((ref_gemm[r][0][2] >= 0).sum()) if ref_gemm[r] else 0)
            # CUDA graph: capture once, replay over all route patterns, poison before every replay
            ids.copy_(routes["valid"])
            arm.run(x, ids, tw, ws)
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                arm.run(x, ids, tw, ws)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                arm.run(x, ids, tw, ws)
            keys = list(routes)
            bad = []
            for i in range(a.replays):
                r = keys[i % len(keys)]
                ids.copy_(routes[r])
                poison(ws)
                gr.replay()
                torch.cuda.synchronize()
                if not same(ws[2], refs[r]):
                    bad.append(r)
            rec(not bad, w=wname, m=m, arm=name, mode="graph", replays=a.replays, mismatching=len(bad),
                routes_bad=sorted(set(bad))[:6])
            del gr
        torch.cuda.synchronize()
    del arms
    torch.cuda.empty_cache()

print(json.dumps(dict(event="summary", status="PASS" if not fails else "FAIL", cases=cases, failures=fails,
                      variants=variants, s=round(time.time() - T0, 1))), flush=True)
sys.exit(1 if fails else 0)
