#!/usr/bin/env python3
"""g1/m32-bw routed-MoE microbench on recorded routes (STOPPED-SERVICE WINDOW ONLY; refuses otherwise).

Same method as mm_marlin_bench.py (2026-10-26 pd Marlin MoE measurement), restricted to the live M32 path and the
exact g1 variants:
  * one TP rank slice of K real checkpoint layers (default 0/13/26/39) loaded with vLLM's RoutedExperts loader;
  * recorded 10-21 decode routes (rank-0 recorder, 40 layers): M=1 = first row of a 6-row verify window, M=6k = k
    single-user windows side by side, c4rec = real 4-user batches (M 12/24); optional uniform-route sets (--rand);
  * one CUDA graph per arm = 40 sequential layer applies (DeepGemmFP4Experts.apply + serving input quant), cycling
    the K layers' weights (L2-cold), each with its own layer's routes; arms interleaved ABAB over --trials replays;
  * a bitwise pre-check (eager, poisoned workspaces) per arm and M on the first route set: arms that differ from live
    are reported and NOT timed;
  * per-kernel profile (torch.profiler) at --profile M values: FC1 / FC2 / glue us per layer for every arm.
Arms: live (serving trees, private cache copies), off (A/A control: same SASS from a g1 tree), pair, pf, la, ds, all.
JSON lines on stdout. Projection to serving ms/step: bench/project.py <this output>.
Usage: source ~/venvs/vllm032/bin/activate
       python ~/dsv41-stage/g1/m32-bw/bench/m32bw_bench.py [--arms live,off,pair,pf,la,ds,all] [--Ms 1,6,12,24,48]
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import g1lib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="live,off,pair,pf,la,ds,all")
ap.add_argument("--layers", default="0,13,26,39")
ap.add_argument("--Ms", default="1,6,12,24,48")
ap.add_argument("--sets", type=int, default=2)
ap.add_argument("--trials", type=int, default=25)
ap.add_argument("--profile", default="6,24")
ap.add_argument("--tp-rank", type=int, default=0)
ap.add_argument("--routes", default=str(g1lib.G / "routes"))
ap.add_argument("--rand", action="store_true", help="also uniform-random route sets (high expert diversity bound)")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
g1lib.guard(a.force)
g1lib.setup_env(g1lib.G / "work")

import torch  # noqa: E402


def emit(**kw):
    print(json.dumps(kw), flush=True)


names = [n for n in a.arms.split(",") if n]
assert names[0] == "live", "live must be the first arm"
variants = [n for n in names if n != "live"]
assert all(v in g1lib.VARIANTS for v in variants), variants
assert torch.cuda.get_device_capability() == (12, 1)
assert torch.cuda.get_device_properties(0).multi_processor_count == 48
emit(event="start", host=__import__("socket").gethostname(), tp_rank=a.tp_rank, arms=names, torch=torch.__version__)
t0 = time.time()
nat = g1lib.Natives(variants)
g1lib.install_live_getter(nat)

layers = [int(x) for x in a.layers.split(",")]
per_layer = []
for layer in layers:
    w1c, s1c, w2, s2 = g1lib.real_layer(layer, a.tp_rank)
    per_layer.append({n: g1lib.Arm(nat.kernels[n], w1c, s1c, w2, s2) for n in names})
    free, tot = torch.cuda.mem_get_info()
    emit(event="layer_ready", layer=layer, s=round(time.time() - t0, 1), free_gb=round(free / 2**30, 1))
K = len(per_layer)
bpe = per_layer[0]["live"].bytes_per_expert


def route_sets(Ms, nsets):
    wins, c4 = g1lib.load_recorded(Path(a.routes))
    emit(event="routes", windows6=len(wins), c4={k: len(v) for k, v in c4.items()})
    g = torch.Generator().manual_seed(20261026)
    sets = []
    for m in Ms:
        for s in range(nsets):
            perm = torch.randperm(len(wins), generator=g).tolist()
            if m == 1:
                ids = wins[perm[0]][:, :1]
            else:
                assert m % 6 == 0, m
                ids = torch.cat([wins[perm[i % len(perm)]] for i in range(m // 6)], 1)
            sets.append((m, f"u6x{max(1, m // 6)}-s{s}", ids))
        if m in c4:
            sets.append((m, "c4rec", c4[m][0]))
        if a.rand:
            ids = torch.stack([torch.stack([torch.randperm(g1lib.E, generator=g)[:g1lib.TOPK] for _ in range(m)])
                               for _ in range(40)]).int()
            sets.append((m, "uniform", ids))
    return sets


def poison(ws):
    for t in ws:
        t.view(torch.uint8).fill_(255)


def same(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


gen = torch.Generator().manual_seed(11)
excluded = {}
checked_m = set()
for (m, setname, ids40c) in route_sets([int(x) for x in a.Ms.split(",")], a.sets):
    ids40 = ids40c.cuda().contiguous()
    tw40 = torch.stack([(torch.softmax(torch.randn(m, g1lib.TOPK, generator=gen), -1) * 1.5).float()
                        for _ in range(40)]).cuda().contiguous()
    x = (torch.randn(m, g1lib.H, generator=gen) * 0.15).to(torch.bfloat16).cuda()
    distinct = [int(torch.unique(ids40c[i][ids40c[i] >= 0]).numel()) for i in range(40)]
    if m not in checked_m:                        # bitwise pre-check vs live (eager, 8 layers' routes, poisoned)
        checked_m.add(m)
        for i in range(8):
            ws = per_layer[i % K]["live"].workspaces(m)
            poison(ws)
            ref = per_layer[i % K]["live"].run(x, ids40[i], tw40[i], ws).clone()
            for n in variants:
                ws2 = per_layer[i % K][n].workspaces(m)
                poison(ws2)
                got = per_layer[i % K][n].run(x, ids40[i], tw40[i], ws2)
                if not same(got, ref):
                    excluded.setdefault(n, []).append((m, i))
        torch.cuda.synchronize()
        emit(event="bitwise_precheck", m=m, mismatching_arms={k: len(v) for k, v in excluded.items()})
    arms_now = [n for n in names if n not in excluded]
    graphs = {}
    for n in arms_now:
        try:
            ws = per_layer[0][n].workspaces(m)

            def body(n=n, ws=ws):
                for i in range(40):
                    per_layer[i % K][n].run(x, ids40[i], tw40[i], ws)
            body()
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                body()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                body()
            graphs[n] = (gr, ws)
        except Exception as ex:
            emit(event="capture_error", arm=n, m=m, set=setname, error=repr(ex)[:800])
    order = list(graphs)
    for _ in range(3):
        for n in order:
            graphs[n][0].replay()
    torch.cuda.synchronize()
    ts = {n: [] for n in order}
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for t in range(a.trials):
        for n in (order if t % 2 == 0 else order[::-1]):
            e0.record()
            graphs[n][0].replay()
            e1.record()
            e1.synchronize()
            ts[n].append(e0.elapsed_time(e1))
    res = {}
    for n, v in ts.items():
        med = statistics.median(v)
        res[n] = dict(step40_ms_med=round(med, 4), step40_ms_min=round(min(v), 4),
                      step40_ms_p90=round(sorted(v)[int(0.9 * (len(v) - 1))], 4),
                      layer_us_med=round(med * 1000 / 40, 2),
                      gbps=round(sum(distinct) * bpe / (med / 1000) / 1e9, 1))
    if "live" in res:
        for n in res:
            res[n]["speedup_vs_live"] = round(res["live"]["step40_ms_med"] / res[n]["step40_ms_med"], 4)
            res[n]["delta_us_per_layer"] = round(res[n]["layer_us_med"] - res["live"]["layer_us_med"], 2)
    emit(event="timing", m=m, set=setname, distinct_experts_mean=round(statistics.mean(distinct), 1),
         trials=a.trials, results=res)
    if str(m) in a.profile.split(",") and (setname.endswith("s0") or setname == "c4rec"):
        try:
            from torch.profiler import profile, ProfilerActivity
            for n in order:
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    graphs[n][0].replay()
                    torch.cuda.synchronize()
                agg = {}
                for ev in prof.events():
                    if ev.device_type.name == "CUDA" and ev.device_time > 0:
                        k = ev.name
                        tag = ("FC1" if "1d1d_impl<0u, 576u, 5120u" in k else
                               "FC2" if "1d1d_impl<0u, 5120u, 384u" in k else k[:60])
                        c, tt = agg.get(tag, (0, 0.0))
                        agg[tag] = (c + 1, tt + ev.device_time)
                top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:10]
                emit(event="profile", m=m, set=setname, arm=n,
                     kernels=[dict(k=k, calls=c, us_per_layer=round(tt / 40, 2)) for k, (c, tt) in top])
        except Exception as ex:
            emit(event="profile_error", m=m, error=repr(ex)[:400])
    del graphs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
emit(event="done", s=round(time.time() - t0, 1), excluded={k: v[:8] for k, v in excluded.items()})
