#!/usr/bin/env python3
"""g130 fc1 GPU exactness test: deal arm vs production 'all' FC1, bitwise (STOPPED-SERVICE WINDOW ONLY; refuses otherwise).

Phase direct: FC1 alone, CUDA graph captured once per (arm, tokens, weights). Before each replay the layout is rewritten (the
  kernel reads it at run time, as in serving) and the output is poisoned to 0xFF. Both arms replay on identical inputs and the
  WHOLE output buffer must match as bytes: live rows, and dead rows, which must keep the poison. G cycles over 0..min(nmb, 96),
  with rows one/few/any/full and multi-block experts. Every 8th replay is an A/A rerun of 'all'. --gs replaces that G schedule with
  an explicit list, run in full or refused. Each row records the G that ran (Gs, read back from the layout), the capture G and,
  without --sanitizer, median ms per G (per_G: G -> [n, ms_all, ms_arm]).
Phase apply: the whole DeepGemmFP4Experts.apply as served, eager and graph replays, workspaces poisoned, routes valid / spec /
  half / hot / skew / one and the recorded ones, output buffer compared as bytes against the 'all' arm.

GO   = status PASS: 0 mismatches across all tokens, >= 200 replays per (arm, tokens, weights), A/A clean.
KILL = any mismatch in a deal arm, or any hang (the wrapper's timeout). A mismatch in A/A means the harness is broken: fix it,
       it is not a verdict.
Usage: source ~/venvs/vllm032/bin/activate; python tests/test_deal_gpu.py [--arms deal] [--tokens 1,2,4,6,8,12,16,24]
          [--replays 240] [--real-layer 0|-1] [--phase direct,apply]
Sanitizers (slow, small case; each needs PASS and 0 errors in its log):
  compute-sanitizer --tool racecheck python tests/test_deal_gpu.py --arms deal --tokens 6 --replays 19 --gs 0-18 --sanitizer --real-layer -1 --phase direct --force-gpu-shared
  compute-sanitizer --tool synccheck python tests/test_deal_gpu.py --arms deal --tokens 6 --replays 19 --gs 0-18 --sanitizer --real-layer -1 --phase direct --force-gpu-shared
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deallib  # noqa: E402
import gsched  # noqa: E402

g1lib = deallib.g1lib
ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="deal")
ap.add_argument("--tokens", default="1,2,4,6,8,12,16,24")
ap.add_argument("--replays", type=int, default=240)
ap.add_argument("--gs", default="", help="direct phase: explicit G list, such as 0-18 or 0,1,8,9, one line per replay")
ap.add_argument("--sanitizer", action="store_true", help="running under compute-sanitizer: no timings, they would mean nothing")
ap.add_argument("--real-layer", type=int, default=0)
ap.add_argument("--tp-rank", type=int, default=0)
ap.add_argument("--phase", default="direct,apply")
ap.add_argument("--routes", default=str(g1lib.G / "routes"))
ap.add_argument("--force", action="store_true")
ap.add_argument("--force-gpu-shared", action="store_true")
a = ap.parse_args()
torch = deallib.start(a.force, a.force_gpu_shared)
arms = [x for x in a.arms.split(",") if x]
G_LIST = gsched.parse_gs(a.gs)
G_DONE = set()
assert all(x in ("deal", "dealx") for x in arms), "nomma changes values and has no place here"
nat, kernels = deallib.load_arms(arms)
T0 = time.time()
fails = cases = 0


def rec(ok, **kw):
    global fails, cases
    cases += 1
    fails += not ok
    print(json.dumps(dict(ok=bool(ok), **kw)), flush=True)


def poison(*ts):
    for t in ts:
        t.view(torch.uint8).fill_(255)


def phase_direct(wname, w, tokens):
    w1c, s1c, w2, s2 = w
    arm_all = g1lib.Arm(kernels["all"], w1c, s1c, w2, s2)
    inp = deallib.Fc1Input(torch, arm_all, tokens, 7000 + tokens)
    gen = torch.Generator().manual_seed(9000 + tokens)
    graphs = {}
    capture_G = inp.nmb if inp.nmb < 3 else min(inp.nmb, 8)
    top = min(inp.nmb, 96)
    if G_LIST:
        gsched.check(G_LIST, a.replays, top)
    for n, k in kernels.items():
        def body(k=k):
            inp.call(k)
        inp.layout.copy_(deallib.make_layout(torch, inp.M_sum, capture_G, gen).cuda())
        graphs[n] = deallib.graph_of(torch, body)
    ref = torch.empty_like(inp.d)
    shapes = ("few", "one", "any", "full")
    for arm in arms:
        bad, aa_bad, gs, t_all, t_arm, by_g = [], 0, set(), [], [], {}
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for r in range(a.replays):
            G = gsched.g_for(r, top, G_LIST, lambda n: int(torch.randint(0, n, (1,), generator=gen)))
            rows = shapes[(r // 2) % 4]
            lay = deallib.make_layout(torch, inp.M_sum, G, gen, rows=rows, multi=0.25 if r % 5 == 0 else 0.0)
            assert int((lay[::32] >= 0).sum()) == G, (r, G)    # the G the kernel reads, not only the G asked for
            inp.layout.copy_(lay.cuda())
            gs.add(G)
            poison(inp.d)
            torch.cuda.synchronize()
            e0.record(); graphs["all"].replay(); e1.record(); e1.synchronize()
            t_all.append(e0.elapsed_time(e1))
            ref.copy_(inp.d)
            if r % 8 == 7:
                poison(inp.d)
                torch.cuda.synchronize()
                graphs["all"].replay()
                torch.cuda.synchronize()
                if not deallib.same_bytes(torch, inp.d, ref):
                    aa_bad += 1
            poison(inp.d)
            torch.cuda.synchronize()
            e0.record(); graphs[arm].replay(); e1.record(); e1.synchronize()
            t_arm.append(e0.elapsed_time(e1))
            by_g.setdefault(G, []).append((t_all[-1], t_arm[-1]))
            same = deallib.same_bytes(torch, inp.d, ref)
            if not same:
                live = lay.cuda() >= 0
                bad.append(dict(r=r, G=G, rows=rows, live_same=bool(deallib.same_bytes(torch, inp.d[live], ref[live]))))
            if G_LIST:
                print(json.dumps(dict(event="replay", arm=arm, r=r, G=G, rows=rows, same=bool(same))), flush=True)
        missing = sorted(set(G_LIST) - gs)
        G_DONE.update(gs)
        timing = {} if a.sanitizer else dict(
            ms_all=round(statistics.median(t_all), 4), ms_arm=round(statistics.median(t_arm), 4),
            per_G={str(g): [len(v), round(statistics.median(x[0] for x in v), 4), round(statistics.median(x[1] for x in v), 4)]
                   for g, v in sorted(by_g.items())})
        rec(not bad and not aa_bad and not missing, w=wname, tokens=tokens, arm=arm, phase="direct", replays=a.replays, nmb=inp.nmb,
            distinct_G=len(gs), Gs=sorted(gs), capture_G=capture_G, missing_G=missing, mismatches=len(bad), aa_mismatches=aa_bad,
            first_bad=bad[:4], timing_valid=not a.sanitizer, **timing)
    for g in graphs.values():
        del g
    del graphs, arm_all, inp


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


def phase_apply(wname, w, m):
    w1c, s1c, w2, s2 = w
    allk = g1lib.Arm(kernels["all"], w1c, s1c, w2, s2)
    gen = torch.Generator().manual_seed(1000 + m)
    x = (torch.randn(m, g1lib.H, generator=gen) * 0.15).to(torch.bfloat16).cuda()
    tw = (torch.softmax(torch.randn(m, g1lib.TOPK, generator=gen), -1) * 1.5).float().cuda()
    routes = g1lib.synthetic_routes(m, gen)
    routes.update(recorded(m, gen))
    routes = {k: v.cuda() for k, v in routes.items()}
    ids = torch.empty(m, g1lib.TOPK, dtype=torch.int32, device="cuda")
    refs = {}
    ws = allk.workspaces(m)
    for r, v in routes.items():
        ids.copy_(v)
        poison(*ws)
        refs[r] = allk.run(x, ids, tw, ws).clone()
        torch.cuda.synchronize()
        tok = (v >= 0).any(1)
        rec(bool(torch.isfinite(refs[r][tok].float()).all()), w=wname, tokens=m, arm="all", route=r, mode="ref_finite")
    for arm in arms:
        ex = g1lib.Arm(kernels[arm], w1c, s1c, w2, s2)
        ws = ex.workspaces(m)
        for r, v in routes.items():
            ids.copy_(v)
            poison(*ws)
            got = ex.run(x, ids, tw, ws)
            torch.cuda.synchronize()
            rec(deallib.same_bytes(torch, got, refs[r]), w=wname, tokens=m, arm=arm, route=r, phase="apply", mode="eager")
        ids.copy_(routes["valid"])

        def body(ex=ex, ws=ws):
            ex.run(x, ids, tw, ws)
        gr = deallib.graph_of(torch, body)
        keys = list(routes)
        bad = []
        for i in range(a.replays):
            r = keys[i % len(keys)]
            ids.copy_(routes[r])
            poison(*ws)
            gr.replay()
            torch.cuda.synchronize()
            if not deallib.same_bytes(torch, ws[2], refs[r]):
                bad.append(r)
        rec(not bad, w=wname, tokens=m, arm=arm, phase="apply", mode="graph", replays=a.replays, mismatches=len(bad),
            routes_bad=sorted(set(bad))[:6])
        del gr, ex
    del allk


weight_sets = [("synthetic", g1lib.synthetic_weights())]
if a.real_layer >= 0 and g1lib.MODEL.exists():
    t0 = time.time()
    weight_sets.append((f"layer{a.real_layer}r{a.tp_rank}", g1lib.real_layer(a.real_layer, a.tp_rank)))
    print(json.dumps(dict(event="real_layer_loaded", s=round(time.time() - t0, 1))), flush=True)
tokens = [int(t) for t in a.tokens.split(",")]
phases = a.phase.split(",")
for wname, w in weight_sets:
    for t in tokens:
        if "direct" in phases:
            phase_direct(wname, w, t)
        if "apply" in phases:
            phase_apply(wname, w, t)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
print(json.dumps(dict(event="summary", status="PASS" if not fails else "FAIL", cases=cases, failures=fails, arms=arms,
                      s=round(time.time() - T0, 1), G_covered=sorted(G_DONE))), flush=True)
sys.exit(1 if fails else 0)
