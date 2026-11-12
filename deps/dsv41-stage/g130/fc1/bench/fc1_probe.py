#!/usr/bin/env python3
"""g130 fc1 probe (STOPPED-SERVICE WINDOW, one GPU; no serving change): FC1 time per call by G, on real checkpoint layers.

Each arm runs one CUDA graph per tokens value: 40 FC1 launches, cycling 4 real layers, each launch with its own layout buffer and
a shared output. A graph is reusable across G because the kernel reads the layout at run time, so a point is a rewrite plus
paired replays (arm A, arm B, ... per trial). The graph holds FC1 only, so us/call = replay time / 40.

Sections (--sections, comma list):
  A0  staircase: arm 'all', G 1..min(nmb, 40) at tokens 1,2,3,4,6,8,12 (below G 14 has never been measured).
  A1  t0: G = 0 at every tokens value (40 launches that do no tile), then G = 1.
  A2  L2-warm: 40 launches of ONE layer on ONE expert set, G 1..8. If the floor does not move, it is not DRAM.
  A3  halfmask: optional, --extra halfmask=<tree>, G 1..24. Is the floor issue-bound?
  A4  nomma: QMMAs removed, G 1..24, timing only. Load-and-sync bound.
  A5  deal / dealx vs all, paired, all G of A0 and tokens 16,24 (this is the decision).
  A6  recorded routes: --routes-bin <fc1-routes*.bin>, replays real (tokens, layout) records, arm vs all (real-route check).
Output: JSONL on stdout, one line per (section, tokens, G, arm). tools/eval_probe.py reads it.
Usage: source ~/venvs/vllm032/bin/activate; python bench/fc1_probe.py --arms all,deal,dealx,nomma > results/probe.jsonl
"""
import argparse
import json
from pathlib import Path
import statistics
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import deallib  # noqa: E402

g1lib = deallib.g1lib
ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="all,deal,dealx,nomma")
ap.add_argument("--sections", default="A0,A1,A2,A4,A5")
ap.add_argument("--tp-rank", type=int, default=3)
ap.add_argument("--layers", default="0,13,26,39")
ap.add_argument("--tokens", default="1,2,3,4,6,8,12")
ap.add_argument("--tokens-a5", default="16,24")
ap.add_argument("--trials", type=int, default=31)
ap.add_argument("--extra", default="", help="name=tree,... (timing-only trees)")
ap.add_argument("--routes-bin", default="")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
torch = deallib.start(a.force)
arm_names = [x for x in a.arms.split(",") if x]
assert "all" in arm_names
extra = [tuple(x.split("=", 1)) for x in a.extra.split(",") if x]
nat, kernels = deallib.load_arms(arm_names, extra)
arm_names += [n for n, _ in extra]
t0 = time.time()


def emit(**kw):
    print(json.dumps(kw), flush=True)


layers = [g1lib.real_layer(int(x), a.tp_rank) for x in a.layers.split(",")]
emit(event="ready", s=round(time.time() - t0, 1), host=__import__("socket").gethostname(), arms=arm_names)
all_arm = [g1lib.Arm(kernels["all"], *w) for w in layers]
gen = torch.Generator().manual_seed(20261102)


class Bench:
    def __init__(self, tokens):
        self.tokens = tokens
        self.inp = [deallib.Fc1Input(torch, ar, tokens, 5000 + tokens + i) for i, ar in enumerate(all_arm)]
        self.base = self.inp[0]
        self.M_sum, self.nmb = self.base.M_sum, self.base.nmb
        self.lay = [torch.full((self.M_sum,), -1, dtype=torch.int32, device="cuda") for _ in range(40)]
        self.graphs = {}
        self.warm = {}

    def graph(self, arm, warm=False):
        cache = self.warm if warm else self.graphs
        if arm not in cache:
            k = kernels[arm]
            base = self.base
            lay, inp = self.lay, self.inp

            def body():
                for i in range(40):
                    w = inp[0] if warm else inp[i % len(inp)]
                    base.call(k, layout=lay[i], bw=w.bw, sfb=w.sfb)
            self.fill(1)
            cache[arm] = deallib.graph_of(torch, body)
        return cache[arm]

    def fill(self, G, rows="few", same=False):
        ls = [deallib.make_layout(torch, self.M_sum, G, gen, rows=rows)]
        for i in range(1, 40):
            ls.append(ls[0] if same else deallib.make_layout(torch, self.M_sum, G, gen, rows=rows))
        for t, s in zip(self.lay, ls):
            t.copy_(s.cuda())
        torch.cuda.synchronize()

    def point(self, section, G, arms, rows="few", warm=False, tag=None):
        gs = {n: self.graph(n, warm) for n in arms}
        self.fill(G, rows, same=warm)
        for g in gs.values():      # warm-up: not timed
            g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = {n: [] for n in arms}
        for tr in range(a.trials):
            order = arms if tr % 2 == 0 else list(reversed(arms))
            for n in order:
                e0.record()
                gs[n].replay()
                e1.record()
                e1.synchronize()
                ts[n].append(e0.elapsed_time(e1) * 1000 / 40)
        med = {n: statistics.median(v) for n, v in ts.items()}
        for n in arms:
            d = {}
            if n != "all" and "all" in ts:
                diffs = [x - y for x, y in zip(ts["all"], ts[n])]
                d = dict(saving_us=round(statistics.median(diffs), 3))
            emit(event="p", section=section, tokens=self.tokens, M_sum=self.M_sum, nmb=self.nmb, G=G, arm=n, rows=rows, tag=tag,
                 us=round(med[n], 3), p10=round(sorted(ts[n])[len(ts[n]) // 10], 3), **d)


sections = a.sections.split(",")
benches = {}


def bench(t):
    if t not in benches:
        benches[t] = Bench(t)
    return benches[t]


tokens = [int(x) for x in a.tokens.split(",")]
if "A0" in sections or "A5" in sections:
    arms5 = [n for n in ("all", "deal", "dealx") if n in arm_names] if "A5" in sections else ["all"]
    for t in tokens + ([int(x) for x in a.tokens_a5.split(",")] if "A5" in sections else []):
        b = bench(t)
        top = min(b.nmb, 40 if t <= 12 else 72)
        for G in range(1, top + 1):
            b.point("A0" if arms5 == ["all"] else "A5", G, arms5)
if "A1" in sections:
    for t in tokens:
        for G in (0, 1):
            bench(t).point("A1", G, ["all"] + [n for n in ("deal", "dealx") if n in arm_names])
if "A2" in sections:
    for t in (2, 4):
        for G in range(1, 9):
            bench(t).point("A2", G, ["all"], warm=True)
if "A3" in sections or "A4" in sections:
    arms4 = (["all"]
             + (["nomma"] if "A4" in sections and "nomma" in arm_names else [])
             + ([n for n, _ in extra] if "A3" in sections else []))
    for t in (2, 4):
        b = bench(t)
        for G in range(1, min(b.nmb, 24) + 1):
            b.point("A34", G, arms4)
if "A6" in sections and a.routes_bin:
    # records as dumped by site/fc1_route_rec.py: see tools/routes_to_hist.py for the layout
    raw = Path(a.routes_bin).read_bytes()
    REC, CAP, total, _ = struct.unpack_from("<4I", raw, 0)
    n = min(total, CAP)
    vals = torch.frombuffer(bytearray(raw[16:16 + n * REC * 4]), dtype=torch.int32).reshape(n, REC)
    by = {}
    for i in range(0, n - n % 40, 40):
        nmb = int(vals[i, 0])
        by.setdefault(nmb, []).append(i)
    for nmb, starts in sorted(by.items()):
        toks = next((t for t in range(1, 97) if deallib.m_sum(t) // 32 == nmb), None)
        if toks is None or len(starts) < 2:
            continue
        b = bench(toks)
        arms6 = [x for x in ("all", "deal", "dealx") if x in arm_names]
        gs = {x: b.graph(x) for x in arms6}
        ts = {x: [] for x in arms6}
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        g_sum = 0
        picks = starts[:: max(1, len(starts) // 200)]
        for i0 in picks:
            for li in range(40):
                r = vals[i0 + li]
                G = int(r[1])
                g_sum += G
                lay = torch.full((b.M_sum,), -1, dtype=torch.int32)
                for m in range(min(G, (REC - 2) // 2)):
                    lay[m * 32:m * 32 + int(r[3 + 2 * m])] = int(r[2 + 2 * m])
                b.lay[li].copy_(lay.cuda())
            torch.cuda.synchronize()
            for x in arms6:
                gs[x].replay()
            torch.cuda.synchronize()
            for x in arms6:
                e0.record()
                gs[x].replay()
                e1.record()
                e1.synchronize()
                ts[x].append(e0.elapsed_time(e1) * 1000 / 40)
        for x in arms6:
            d = {}
            if x != "all":
                d = dict(saving_us=round(statistics.median([p - q for p, q in zip(ts["all"], ts[x])]), 3))
            emit(event="p", section="A6", tokens=toks, nmb=nmb, arm=x, steps=len(picks), meanG=round(g_sum / (40 * len(picks)), 2),
                 us=round(statistics.median(ts[x]), 3), **d)
emit(event="done", s=round(time.time() - t0, 1))
