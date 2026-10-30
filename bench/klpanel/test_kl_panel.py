#!/usr/bin/env python3
"""CPU unit tests for kl_panel.py: synthetic logprob dumps and a tiny mock vLLM server only.
No GPU, no real server, a few seconds of one core.

  python3 test_kl_panel.py                        # numpy bootstrap if numpy imports
  KLPANEL_NO_NUMPY=1 python3 test_kl_panel.py     # pure-stdlib path
  python3 test_kl_panel.py -v Estimator           # one group
  KLPANEL_TEST_KEEP=1 python3 test_kl_panel.py    # keep the mock runs/ + compare reports for inspection
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import random
import shutil
import struct
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kl_panel as kp  # noqa: E402


# ---------------------------------------------------------------------------------------- helpers

def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def log_softmax(lg):
    m = max(lg)
    s = math.fsum(math.exp(x - m) for x in lg)
    lse = m + math.log(s)
    return [x - lse for x in lg]


def topk(lp, K):
    return sorted(((t, v) for t, v in enumerate(lp)), key=lambda e: (-e[1], e[0]))[:K]


def full_kl(lp, lq):
    return math.fsum(math.exp(a) * (a - b) for a, b in zip(lp, lq))


def rand_logits(rng, V, scale):
    return [rng.gauss(0.0, 1.0) * scale for _ in range(V)]


def dists(rng, V, scale, eps, pert=None):
    """Reference / candidate log-probs (float64 and float32-rounded) for one position."""
    lg = rand_logits(rng, V, scale)
    lg2 = [x + eps * rng.gauss(0.0, 1.0) for x in lg] if pert is None else pert(list(lg))
    lp, lq = log_softmax(lg), log_softmax(lg2)
    return lp, lq, [f32(x) for x in lp], [f32(x) for x in lq]


def entries(lp32, K, y):
    return topk(lp32, K), lp32[y]


@contextlib.contextmanager
def quiet():
    buf_o, buf_e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_o), contextlib.redirect_stderr(buf_e):
        yield buf_o


# -------------------------------------------------------------------------------------- estimator

class Estimator(unittest.TestCase):
    K = 20

    def test_identical_is_exactly_zero(self):
        rng = random.Random(1)
        for trial in range(300):
            V = rng.choice([20, 21, 50, 300])
            lp32 = [f32(x) for x in log_softmax(rand_logits(rng, V, rng.choice([0.3, 2, 6, 12])))]
            y = rng.randrange(V)
            pt, py = entries(lp32, self.K, y)
            m = dict(zip(kp.METRICS, kp.pos_metrics(pt, py, pt, py, y)))
            self.assertEqual(m["kl"], 0.0, trial)
            self.assertEqual(m["kl_rev"], 0.0, trial)
            self.assertAlmostEqual(m["kl_norm"], 0.0, delta=1e-15)
            self.assertEqual(m["top1"], 1.0)
            self.assertEqual(m["top5"], 1.0)
            self.assertEqual(m["dlp1"], 0.0)
            self.assertEqual(m["nll_r"], m["nll_c"])
            self.assertEqual(m["miss"], 0.0)

    def test_full_support_equals_exact_kl(self):
        """V == K: the support is the whole vocabulary, so the estimator is the exact KL."""
        rng = random.Random(2)
        for _ in range(200):
            _, _, lp32, lq32 = dists(rng, self.K, rng.choice([0.5, 3, 8]), rng.choice([0.01, 0.2, 1.0]))
            y = rng.randrange(self.K)
            pt, py = entries(lp32, self.K, y)
            qt, qy = entries(lq32, self.K, y)
            kl, kln, miss, _, tail = kp.bucket_kl(pt, py, qt, qy, y)
            # exact KL of the float32 inputs (they are normalised to ~1e-7, hence the tolerance)
            exact = full_kl(lp32, lq32)
            self.assertEqual(miss, 0)
            self.assertAlmostEqual(kl, exact, delta=2e-6 + 1e-6 * exact)
            self.assertAlmostEqual(kln, exact, delta=2e-6 + 1e-6 * exact)
            self.assertLess(tail, 1e-6)

    def test_lower_bound_on_full_vocab_kl(self):
        """V >> K: never above the exact full-vocabulary KL (both directions), close when the tail is light."""
        rng = random.Random(3)
        close = 0
        n = 600
        for trial in range(n):
            V = rng.choice([64, 200, 1000])
            scale = rng.choice([1.0, 3.0, 6.0, 12.0])
            lp, lq, lp32, lq32 = dists(rng, V, scale, rng.choice([0.003, 0.05, 0.3, 1.5]))
            y = rng.randrange(V) if rng.random() < 0.3 else topk(lp, 1)[0][0]
            pt, py = entries(lp32, self.K, y)
            qt, qy = entries(lq32, self.K, y)
            kl, kln, miss, _, tail = kp.bucket_kl(pt, py, qt, qy, y)
            exact = full_kl(lp, lq)
            self.assertGreaterEqual(kl, -1e-7, trial)
            self.assertLessEqual(kl, exact + 2e-6 + 1e-6 * exact, (trial, V, scale, kl, exact))
            klr = kp.bucket_kl(qt, qy, pt, py, y)[0]
            self.assertLessEqual(klr, full_kl(lq, lp) + 2e-6, trial)
            if tail < 1e-3 and exact > 1e-4:
                close += kl >= 0.9 * exact
        self.assertGreater(close, 0)

    def test_lower_bound_light_tail_is_tight(self):
        rng = random.Random(4)
        for _ in range(200):
            lp, lq, lp32, lq32 = dists(rng, 500, 12.0, 0.2)
            y = topk(lp, 1)[0][0]
            pt, py = entries(lp32, self.K, y)
            qt, qy = entries(lq32, self.K, y)
            kl = kp.bucket_kl(pt, py, qt, qy, y)[0]
            exact = full_kl(lp, lq)
            if exact > 1e-3:
                self.assertGreater(kl, 0.8 * exact)

    def test_top1_swap(self):
        rng = random.Random(5)
        for _ in range(100):
            lg = rand_logits(rng, self.K, 2.0)
            o = sorted(range(self.K), key=lambda t: -lg[t])
            a, b = o[0], o[1]
            lg2 = list(lg)
            lg2[a], lg2[b] = lg2[b], lg2[a]
            lp32 = [f32(x) for x in log_softmax(lg)]
            lq32 = [f32(x) for x in log_softmax(lg2)]
            y = o[2]
            pt, py = entries(lp32, self.K, y)
            qt, qy = entries(lq32, self.K, y)
            m = dict(zip(kp.METRICS, kp.pos_metrics(pt, py, qt, qy, y)))
            p1, p2 = math.exp(lp32[a]), math.exp(lp32[b])
            expect = (p1 - p2) * (lp32[a] - lp32[b])
            if lp32[a] != lp32[b]:
                self.assertEqual(m["top1"], 0.0)
                self.assertEqual(m["top5"], 0.0)
            self.assertAlmostEqual(m["kl"], expect, delta=1e-6)
            self.assertAlmostEqual(m["dlp1"], abs(lp32[a] - lq32[a]), delta=1e-12)
            self.assertAlmostEqual(m["dlp1"], abs(lp32[a] - lp32[b]), delta=1e-6)
            self.assertEqual(m["nll_r"], -lp32[y])
            self.assertEqual(m["nll_c"], -lq32[y])

    def test_missing_tokens_still_counted(self):
        """The candidate pushes the reference argmax far down (out of its top-K): large KL, still <= exact."""
        rng = random.Random(6)
        for _ in range(100):
            V = 300
            lg = rand_logits(rng, V, 3.0)
            a = max(range(V), key=lambda t: lg[t])
            lg2 = list(lg)
            lg2[a] -= 25.0
            lp, lq = log_softmax(lg), log_softmax(lg2)
            lp32, lq32 = [f32(x) for x in lp], [f32(x) for x in lq]
            y = rng.randrange(V)
            pt, py = entries(lp32, self.K, y)
            qt, qy = entries(lq32, self.K, y)
            kl, _, miss, q_r1, _ = kp.bucket_kl(pt, py, qt, qy, y)
            exact = full_kl(lp, lq)
            self.assertLessEqual(kl, exact + 1e-6)
            # {a} is its own bucket: log-sum inequality on {a} vs rest, with q_a <= the candidate's K-th
            # listed probability when a is not listed (it is always listed when a == y)
            pa = math.exp(lp32[a])
            if y == a:
                qa = math.exp(lq32[a])
                bound = pa * math.log(pa / qa) + (1 - pa) * math.log((1 - pa) / (1 - qa))
            else:
                self.assertGreaterEqual(miss, 1)
                cap = math.exp(min(v for _, v in qt))
                bound = pa * math.log(pa / cap) + (1 - pa) * math.log(1 - pa)
                self.assertLessEqual(q_r1, math.log(cap) + 1e-9)
            self.assertGreaterEqual(kl, bound - 1e-6)

    def test_complete_missing_is_the_minimum(self):
        rng = random.Random(7)

        def obj(p, x, D, R, Po):
            s = sum(x)
            v = math.fsum(pi * math.log(pi / xi) for pi, xi in zip(p, x))
            if Po > 0:
                q = D + R - s
                if q <= 0:
                    return math.inf
                v += Po * math.log(Po / q)
            return v

        for trial in range(400):
            n = rng.randint(1, 4)
            p = [rng.uniform(1e-4, 0.3) for _ in range(n)]
            cap = 10 ** rng.uniform(-5, -0.7)
            R = 10 ** rng.uniform(-4, -0.3)
            D = rng.choice([0.0, rng.uniform(0, 0.6)])
            Po = rng.choice([0.0, 10 ** rng.uniform(-5, -0.5)])
            x = kp.complete_missing(p, cap, D, R, Po)
            self.assertEqual(len(x), n)
            for xi in x:
                self.assertLessEqual(xi, cap * (1 + 1e-9) + 1e-30)
                self.assertGreater(xi, 0)
            self.assertLessEqual(sum(x), R * (1 + 1e-6) + 1e-30, trial)
            best = obj(p, x, D, R, Po)
            for _ in range(300):
                z = [rng.uniform(1e-12, cap) for _ in range(n)]
                s = sum(z)
                lim = R * rng.choice([1.0, rng.random()])
                if s > lim:
                    z = [zi * lim / s for zi in z]
                self.assertLessEqual(best, obj(p, z, D, R, Po) + 1e-9 + 1e-9 * abs(best), (trial, p, cap, D, R, Po))

    def test_kl_norm_by_hand(self):
        # ref top-3 (K=3) over 4 tokens; candidate lists the same three tokens
        K = 3
        p = [0.5, 0.3, 0.15, 0.05]
        q = [0.4, 0.35, 0.2, 0.05]
        lp = [math.log(x) for x in p]
        lq = [math.log(x) for x in q]
        pt, qt = topk(lp, K), topk(lq, K)
        kl, kln, miss, _, tail = kp.bucket_kl(pt, lp[0], qt, lq[0], 0)
        P, Q = sum(p[:3]), sum(q[:3])
        want_n = sum(p[i] / P * math.log((p[i] / P) / (q[i] / Q)) for i in range(3))
        want_b = sum(p[i] * math.log(p[i] / q[i]) for i in range(3)) + 0.05 * math.log(0.05 / 0.05)
        self.assertAlmostEqual(kln, want_n, places=12)
        self.assertAlmostEqual(kl, want_b, places=12)
        self.assertEqual(miss, 0)
        self.assertAlmostEqual(tail, 0.05, places=12)

    def test_seq_metrics_roundtrip(self):
        """Packed records -> seq_metrics_tf gives the per-position values of pos_metrics."""
        rng = random.Random(8)
        K, V, n = self.K, 80, 30
        ids = [0] + [rng.randrange(V) for _ in range(n - 1)]
        recs = []
        for eps in (0.0, 0.4):
            ylp, tid, tlp = [], [], []
            for t in range(1, n):
                r2 = random.Random(t)
                lg = rand_logits(r2, V, 3.0)
                lg = [x + eps * rng.gauss(0, 1) for x in lg]
                lp32 = [f32(x) for x in log_softmax(lg)]
                ylp.append(lp32[ids[t]])
                for tok, v in topk(lp32, K):
                    tid.append(tok)
                    tlp.append(v)
            recs.append({"set": "s", "idx": 0, "ylp": kp.pack_arr("f", ylp), "tid": kp.pack_arr("i", tid),
                         "tlp": kp.pack_arr("f", tlp)})
        same = kp.seq_metrics_tf(recs[0], recs[0], ids, K)
        self.assertEqual(sum(same["kl"]), 0.0)
        self.assertEqual(sum(same["top1"]), n - 1)
        diff = kp.seq_metrics_tf(recs[0], recs[1], ids, K)
        self.assertEqual(len(diff["kl"]), n - 1)
        self.assertGreater(sum(diff["kl"]), 0.0)
        self.assertEqual(len(diff["_worst"]), 5)
        self.assertAlmostEqual(sum(diff["dnll"]), sum(diff["nll_c"]) - sum(diff["nll_r"]), places=9)


# ---------------------------------------------------------------------------------------- parsing

def vllm_position(lp32, y, K, flat_first=False):
    """One prompt position the way vLLM 0.32 serialises it: scored token first, then the top-K."""
    top = topk(lp32, K)
    rank_y = 1 + sum(1 for v in lp32 if v > lp32[y])
    d = {str(y): {"logprob": lp32[y], "rank": rank_y, "decoded_token": f"<{y}>"}}
    for r, (t, v) in enumerate(top, 1):
        d[str(t)] = {"logprob": v, "rank": r, "decoded_token": f"<{t}>"}
    return d


class Parsing(unittest.TestCase):
    def _resp(self, ids, lps, K, first=None):
        plp = [first] + [vllm_position(lps[t], ids[t], K) for t in range(1, len(ids))]
        return {"choices": [{"index": 0, "text": "", "prompt_logprobs": plp, "prompt_token_ids": list(ids),
                             "token_ids": [7], "finish_reason": "length"}], "usage": {"prompt_tokens": len(ids)}}

    def test_parse_tf(self):
        rng = random.Random(11)
        K, V, n = 5, 30, 12
        ids = [0] + [rng.randrange(V) for _ in range(n - 1)]
        lps = [None] + [[f32(x) for x in log_softmax(rand_logits(rng, V, 2.0))] for _ in range(n - 1)]
        for first in (None, {}):
            p = kp.parse_tf(self._resp(ids, lps, K, first), ids, K)
            self.assertEqual(len(p["ylp"]), n - 1)
            self.assertEqual(len(p["tid"]), (n - 1) * K)
            for t in range(1, n):
                self.assertEqual(p["ylp"][t - 1], lps[t][ids[t]])
                want = topk(lps[t], K)
                self.assertEqual(list(zip(p["tid"][(t - 1) * K:t * K], p["tlp"][(t - 1) * K:t * K])), want)
                self.assertEqual(p["yrank"][t - 1], 1 + sum(1 for v in lps[t] if v > lps[t][ids[t]]))
            self.assertEqual(p["gen"], [7])

    def test_parse_tf_errors(self):
        rng = random.Random(12)
        K, V, n = 5, 30, 6
        ids = [0] + [rng.randrange(V) for _ in range(n - 1)]
        lps = [None] + [[f32(x) for x in log_softmax(rand_logits(rng, V, 2.0))] for _ in range(n - 1)]
        r = self._resp(ids, lps, K)
        with self.assertRaises(kp.ParseError):
            kp.parse_tf(r, ids[:-1], K)                        # length mismatch
        r2 = json.loads(json.dumps(r))
        r2["choices"][0]["prompt_token_ids"] = [1] + ids
        with self.assertRaises(kp.ParseError):
            kp.parse_tf(r2, ids, K)                            # server added a BOS
        r3 = json.loads(json.dumps(r))
        d = r3["choices"][0]["prompt_logprobs"][2]
        del d[next(iter(d))]
        if len(d) in (K, K + 1):
            del d[next(iter(d))]
        with self.assertRaises(kp.ParseError):
            kp.parse_tf(r3, ids, K)                            # wrong entry count / missing scored token
        with self.assertRaises(kp.ParseError):
            kp.parse_tf({"error": "x"}, ids, K)

    def test_json_roundtrip_of_float32(self):
        """Logprobs survive JSON (vLLM float32 -> Python float -> JSON -> float) and pack_arr exactly."""
        rng = random.Random(13)
        xs = [f32(rng.uniform(-30, 0)) for _ in range(1000)] + [-9999.0, 0.0, f32(-1e-7)]
        back = json.loads(json.dumps(xs))
        self.assertEqual(back, xs)
        self.assertEqual(list(kp.unpack_arr("f", kp.pack_arr("f", back))), xs)
        ints = [rng.randrange(129280) for _ in range(1000)] + [-1]
        self.assertEqual(list(kp.unpack_arr("i", kp.pack_arr("i", ints))), ints)

    def test_parse_gen(self):
        K = 4
        lps = [[f32(x) for x in log_softmax([0.1 * i * (j + 1) for i in range(12)])] for j in range(3)]
        out = [11, 11, 11]
        tops = []
        for lp, y in zip(lps, out):
            d = {f"token_id:{y}": lp[y]}
            for t, v in topk(lp, K):
                d[f"token_id:{t}"] = v
            tops.append(d)
        resp = {"choices": [{"token_ids": out, "logprobs": {"top_logprobs": tops, "tokens": []},
                             "finish_reason": "length"}]}
        g = kp.parse_gen(resp, K)
        self.assertEqual(g["out"], out)
        self.assertEqual(g["ylp"], [lp[11] for lp in lps])
        bad = json.loads(json.dumps(resp))
        bad["choices"][0]["logprobs"]["top_logprobs"][0] = {"x": 1.0}
        with self.assertRaises(kp.ParseError):
            kp.parse_gen(bad, K)


# ---------------------------------------------------------------------------------------- storage

class Storage(unittest.TestCase):
    def test_scan_and_repair_torn_tail(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "s.jsonl")
            recs = [{"idx": i, "seq_sha": f"h{i}", "x": "y" * (50 + i)} for i in range(3)]
            with open(p, "wb") as f:
                for r in recs[:2]:
                    f.write((json.dumps(r) + "\n").encode())
                f.write(json.dumps(recs[2]).encode()[:30])          # torn write
            index, bad = kp.scan_run_file(p)
            self.assertEqual(sorted(index), [0, 1])
            self.assertEqual(bad, 0)
            size_before = os.path.getsize(p)
            index, _ = kp.scan_run_file(p, repair=True)
            self.assertLess(os.path.getsize(p), size_before)
            with open(p, "ab") as f:
                f.write((json.dumps(recs[2]) + "\n").encode())
                f.write((json.dumps({"idx": 0, "seq_sha": "dup"}) + "\n").encode())   # duplicate: first wins
                f.write(b"{not json}\n")
            index, bad = kp.scan_run_file(p)
            self.assertEqual(sorted(index), [0, 1, 2])
            self.assertEqual(index[0][2], "h0")
            self.assertEqual(bad, 1)
            for k, (off, n, _) in index.items():
                self.assertEqual(kp.read_record(p, off, n), recs[k])
        finally:
            shutil.rmtree(d)


# ------------------------------------------------------------------------------------------ stats

class Stats(unittest.TestCase):
    def test_cluster_boot(self):
        strata = [[{"m": (2.0, 4)} for _ in range(5)], [{"m": (1.0, 2)} for _ in range(3)]]
        lo, hi = kp.cluster_boot(strata, ("m",), 200, 1)["m"]
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)
        rng = random.Random(2)
        strata = [[{"m": (rng.gauss(0.1, 0.05) * 100, 100)} for _ in range(40)]]
        mean = sum(r["m"][0] for r in strata[0]) / (40 * 100)
        lo, hi = kp.cluster_boot(strata, ("m",), 500, 3)["m"]
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)
        self.assertLess(hi - lo, 0.05)

    def test_position_ci_both_paths(self):
        rng = random.Random(4)
        xs = [rng.gauss(1.0, 0.5) for _ in range(5000)]
        mu = sum(xs) / len(xs)
        saved = kp.np
        try:
            for np_ in ([saved] if saved is not None else []) + [None]:
                kp.np = np_
                lo, hi, how = kp.position_ci({"m": xs}, ("m",), 300, 5)["m"]
                self.assertLess(lo, mu)
                self.assertGreater(hi, mu)
                self.assertAlmostEqual(hi - lo, 2 * 1.96 * 0.5 / math.sqrt(5000), delta=0.006)
                self.assertEqual(how, "normal" if np_ is None else "bootstrap")
        finally:
            kp.np = saved

    def test_gate(self):
        cand = {"kl": 3e-4, "top1": 0.985, "dnll": 0.0005}
        noise = {"kl": 1e-4, "top1": 0.99, "dnll": 0.0}
        g = kp.gate("equiv", cand, noise, {"kl": (1e-4, 3e-4), "top1": (-0.008, -0.002), "dnll": (-0.001, 0.001)})
        self.assertEqual(g["verdict"], "PASS")
        self.assertEqual(g["criteria"][0]["thr"], 5e-4)
        self.assertAlmostEqual(g["criteria"][1]["thr"], -0.01)
        g = kp.gate("equiv", cand, noise, {"kl": (4e-4, 7e-4), "top1": (-0.008, -0.002), "dnll": (-0.001, 0.001)})
        self.assertEqual(g["verdict"], "BORDERLINE")
        g = kp.gate("equiv", cand, noise, {"kl": (6e-4, 9e-4), "top1": (-0.008, -0.002), "dnll": (-0.001, 0.001)})
        self.assertEqual(g["verdict"], "FAIL")
        g = kp.gate("equiv", cand, noise, {"kl": (1e-4, 3e-4), "top1": (-0.008, -0.002), "dnll": (0.0025, 0.004)})
        self.assertEqual(g["verdict"], "FAIL")
        g = kp.gate("equiv", cand, noise, {"kl": (1e-4, 3e-4), "top1": (-0.008, -0.002), "dnll": (-0.004, -0.0025)})
        self.assertEqual(g["verdict"], "FAIL")                 # two-sided in equiv
        g = kp.gate("lossy", {"kl": 0.02, "top1": 0.95, "dnll": -0.01}, noise,
                    {"kl": (0.015, 0.025), "top1": (-0.045, -0.035), "dnll": (-0.02, -0.005)})
        self.assertEqual(g["verdict"], "PASS")                 # lower NLL is fine in lossy
        g = kp.gate("lossy", {"kl": 0.04, "top1": 0.95, "dnll": 0.0}, None,
                    {"kl": (0.035, 0.045), "top1": (-0.05, -0.04), "dnll": (-0.001, 0.001)})
        self.assertEqual(g["verdict"], "FAIL")
        noise_big = {"kl": 2e-3, "top1": 0.97, "dnll": 0.0}   # floor above the absolute slack -> relative
        g = kp.gate("equiv", cand, noise_big, {"kl": (1e-3, 1.9e-3), "top1": (-0.02, -0.01), "dnll": (-0.001, 0.001)})
        self.assertEqual(g["criteria"][0]["thr"], 2e-3)
        self.assertEqual(g["verdict"], "PASS")


# ------------------------------------------------------------------------------- mock vLLM server

class MockModel:
    """Deterministic causal toy LM over V tokens; eps > 0 adds a seeded logit perturbation."""

    def __init__(self, V=64, eps=0.0, pert_seed=0, tie_every=0):
        self.V, self.eps, self.pert_seed, self.tie_every = V, eps, pert_seed, tie_every

    def logprobs(self, ctx):
        a, b, t = ctx[-1], (ctx[-2] if len(ctx) > 1 else -1), len(ctx)
        h = hashlib.sha256(f"{a},{b},{t % 5}".encode()).digest()
        rng = random.Random(h)
        scale = (3.0, 5.0, 8.0, 12.0)[h[0] % 4]
        lg = [rng.gauss(0.0, 1.0) * scale for _ in range(self.V)]
        if self.tie_every and t % self.tie_every == 0:
            o = sorted(range(self.V), key=lambda i: -lg[i])
            lg[o[1]] = lg[o[0]]                                # exact top-1 tie (bf16-style)
        if self.eps:
            r2 = random.Random(hashlib.sha256(f"p{self.pert_seed},{a},{b},{t}".encode()).digest())
            lg = [x + self.eps * r2.gauss(0.0, 1.0) for x in lg]
        return [f32(x) for x in log_softmax(lg)]


class MockState:
    model = MockModel()
    requests = 0
    max_logprobs = 20


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": "dsv41", "max_model_len": 4096}]})
        elif self.path == "/version":
            self._send(200, {"version": "mock"})
        else:
            self._send(404, {"error": "no"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockState.requests += 1
        m = MockState.model
        ids = body["prompt"]
        K = body.get("prompt_logprobs")
        L = body.get("logprobs")
        for v in (K, L):
            if v is not None and v > MockState.max_logprobs:
                return self._send(400, {"error": {"message": f"Requested prompt logprobs of {v}, which is greater "
                                                             f"than max allowed: {MockState.max_logprobs}"}})
        ch = {"index": 0, "text": "", "finish_reason": "length"}
        if K is not None:
            ch["prompt_logprobs"] = [None] + [vllm_position(m.logprobs(ids[:t]), ids[t], K) for t in range(1, len(ids))]
        ctx, out, tops = list(ids), [], []
        for _ in range(body.get("max_tokens", 16)):
            lp = m.logprobs(ctx)
            y = topk(lp, 1)[0][0]
            out.append(y)
            if L is not None:
                d = {f"token_id:{y}": lp[y]}
                for t, v in topk(lp, L):
                    d[f"token_id:{t}"] = v
                tops.append(d)
            ctx.append(y)
        if L is not None:
            ch["logprobs"] = {"tokens": [f"token_id:{y}" for y in out], "token_logprobs": [], "top_logprobs": tops}
        if body.get("return_token_ids"):
            ch["prompt_token_ids"], ch["token_ids"] = list(ids), out
        self._send(200, {"choices": [ch], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out)}})


def write_sets(base, V=64, seed=9):
    rng = random.Random(seed)
    plan = {"alpha": ("tf", [24, 40, 33, 17, 90, 120]), "beta": ("tf", [50, 29, 75, 110]), "g": ("gen", [8, 11])}
    os.makedirs(os.path.join(base, "sets"))
    man = {"builder_version": "test", "sets": {}, "files": {}}
    tot = 0
    for name, (panel, lens) in plan.items():
        rows = []
        for i, n in enumerate(lens):
            ids = [0] + [rng.randrange(1, V) for _ in range(n - 1)]
            row = {"set": name, "idx": i, "n": n, "sha256": kp.seq_sha(ids), "src": [f"synthetic:{i}"], "ids": ids}
            if panel == "gen":
                row["max_tokens"] = 12
            rows.append(row)
        fn = os.path.join(base, "sets", name + ".jsonl")
        with open(fn, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        pos = sum(n - 1 for n in lens) if panel == "tf" else 12 * len(lens)
        tot += pos if panel == "tf" else 0
        man["sets"][name] = {"panel": panel, "n_seq": len(rows), "positions": pos, "sha256": kp.set_sha(rows)}
        man["files"][name + ".jsonl"] = kp.file_sha(fn)
    man["tf_positions"] = tot
    with open(os.path.join(base, "sets", "manifest.json"), "w") as f:
        json.dump(man, f)
    return plan


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.th = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.th.start()
        cls.base = tempfile.mkdtemp(prefix="klpanel-test-")
        cls.plan = write_sets(cls.base)
        cls._orig_find = kp.find_server_process

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        kp.find_server_process = cls._orig_find
        if os.environ.get("KLPANEL_TEST_KEEP"):
            print(f"\nkept test runs in {cls.base}", file=sys.stderr)
        else:
            shutil.rmtree(cls.base)

    def run_kp(self, *args, instance="mock@1"):
        kp.find_server_process = lambda: {"pid": 1, "argv": ["mock"], "env": {}, "start_epoch": 0,
                                          "instance": instance}
        with quiet() as out:
            rc = kp.main(["--base", self.base] + list(args))
        return rc, out.getvalue()

    def capture(self, label, model, instance, *extra):
        MockState.model = model
        return self.run_kp("capture", "--label", label, "--url", self.url, *extra, instance=instance)

    def report(self, cand, ref="REF", panel="tf"):
        tag = f"{panel}_vs_{ref}" if panel != "tf" else f"vs_{ref}"
        with open(os.path.join(self.base, "runs", cand, f"compare_{tag}.json")) as f:
            return json.load(f)

    def test_end_to_end(self):
        base = self.base
        rc, out = self.run_kp("verify")
        self.assertEqual(rc, 0, out)
        # REF: deterministic toy model
        rc, out = self.capture("REF", MockModel(), "mock@1", "--concurrency", "3")
        self.assertEqual(rc, 0, out)
        p = kp.run_file(base, "REF", "alpha")
        idx, _ = kp.scan_run_file(p)
        n_alpha = len(self.plan["alpha"][1])
        self.assertEqual(sorted(idx), list(range(n_alpha)))
        # resume: nothing to do; then simulate a crash (torn tail + a missing record) and resume again
        n0 = MockState.requests
        rc, out = self.capture("REF", MockModel(), "mock@1")
        self.assertEqual(MockState.requests, n0, out)
        with open(p, "rb") as f:
            lines = f.read().splitlines(keepends=True)
        keep = {json.loads(x)["idx"]: x for x in lines}
        with open(p, "wb") as f:
            f.write(keep[0] + keep[2] + keep[3][:100])        # idx 1, 4, 5 lost, idx 3 torn
        rc, out = self.capture("REF", MockModel(), "mock@1", "--concurrency", "2")
        self.assertEqual(rc, 0, out)
        self.assertEqual(MockState.requests - n0, n_alpha - 2)
        idx, bad = kp.scan_run_file(p)
        self.assertEqual((sorted(idx), bad), (list(range(n_alpha)), 0))
        for k in (1, 3, 5):                                    # deterministic model: same records again
            off, n, _ = idx[k]
            self.assertEqual(kp.read_record(p, off, n)["tlp"], json.loads(keep[k])["tlp"])
        # a restarted server must not silently extend a run
        rc, out = self.capture("REF", MockModel(), "mock@9", "--sets", "beta")
        self.assertEqual(rc, 0, out)                          # beta complete already -> nothing scored, fine
        os.remove(kp.run_file(base, "REF", "beta"))
        rc, _ = self.capture("REF", MockModel(), "mock@9", "--sets", "beta")
        self.assertEqual(rc, 2)
        rc, out = self.capture("REF", MockModel(), "mock@1", "--sets", "beta")
        self.assertEqual(rc, 0, out)
        # REF2: same config after a "restart" (tiny numerical noise); bitwise twin; candidates
        self.assertEqual(self.capture("REF2", MockModel(eps=1e-3, pert_seed=1), "mock@2", "--replicate-of", "REF")[0], 0)
        self.assertEqual(self.capture("TWIN", MockModel(), "mock@3")[0], 0)
        self.assertEqual(self.capture("SAME", MockModel(eps=1e-3, pert_seed=2), "mock@4")[0], 0)
        self.assertEqual(self.capture("LOSSY", MockModel(eps=0.05, pert_seed=3), "mock@5")[0], 0)
        self.assertEqual(self.capture("BAD", MockModel(eps=1.5, pert_seed=4), "mock@6")[0], 0)
        self.assertEqual(self.capture("C4", MockModel(), "mock@7", "--concurrency", "1")[0], 0)

        # bitwise twin: KL exactly 0, agreement 1, gate PASS
        rc, out = self.run_kp("compare", "--ref", "REF", "--label", "TWIN", "--boot", "200", "--cluster-boot", "200",
                              "--jobs", "2")
        self.assertEqual(rc, 0, out)
        r = self.report("TWIN")
        ov = r["sets"]["overall"]["cand"]
        self.assertEqual(ov["kl"], 0.0)
        self.assertEqual(ov["top1"], 1.0)
        self.assertEqual(ov["dlp1"], 0.0)
        self.assertEqual(ov["dnll"], 0.0)
        self.assertEqual(r["noise_used"], "REF2")
        self.assertTrue(r["noise_runs"]["REF2"]["cross_instance"])
        self.assertEqual(r["gates"]["equiv"]["verdict"], "PASS")
        self.assertEqual(ov["positions"], sum(n - 1 for nm, (pl, lens) in self.plan.items() if pl == "tf"
                                              for n in lens))
        # concurrency does not change which positions are scored (bitwise identical toy model)
        rc, out = self.run_kp("compare", "--ref", "REF", "--label", "C4", "--boot", "50", "--cluster-boot", "50")
        self.assertEqual(self.report("C4")["sets"]["overall"]["cand"]["kl"], 0.0)
        self.assertIn("concurrency differs", out)

        # same config, other instance: within the noise floor
        self.run_kp("compare", "--ref", "REF", "--label", "SAME", "--boot", "200", "--cluster-boot", "300")
        r = self.report("SAME")
        self.assertEqual(r["gates"]["equiv"]["verdict"], "PASS", json.dumps(r["gates"]))
        self.assertLess(r["sets"]["overall"]["cand"]["kl"], 1e-4)

        # mildly lossy: fails equivalence, passes the lossy tier
        rc, out = self.run_kp("compare", "--ref", "REF", "--label", "LOSSY", "--tier", "lossy", "--strict",
                              "--boot", "200", "--cluster-boot", "300")
        r = self.report("LOSSY")
        self.assertEqual(r["gates"]["equiv"]["verdict"], "FAIL", out)
        self.assertEqual(r["gates"]["lossy"]["verdict"], "PASS", out)
        self.assertEqual(rc, 0)
        self.assertIn("GATE lossy -> PASS <==", out)

        # broken: fails both, --strict exits 1
        rc, out = self.run_kp("compare", "--ref", "REF", "--label", "BAD", "--tier", "lossy", "--strict",
                              "--boot", "200", "--cluster-boot", "300")
        r = self.report("BAD")
        self.assertEqual(rc, 1, out)
        self.assertEqual(r["gates"]["lossy"]["verdict"], "FAIL")
        self.assertLess(r["sets"]["overall"]["cand"]["top1"], 0.9)

        # estimator stays below the exact full-vocab KL at every position, and is close to it
        sets = kp.load_sets(base, ["alpha", "beta"])[1]
        ref_m, bad_m = MockModel(), MockModel(eps=1.5, pert_seed=4)
        ir, ib = kp.run_index(base, "REF", ["alpha", "beta"], sets), kp.run_index(base, "BAD", ["alpha", "beta"], sets)
        est_sum = exact_sum = 0.0
        for nm in ("alpha", "beta"):
            for row in sets[nm]:
                ids = row["ids"]
                m = kp.seq_metrics_tf(kp.read_record(*ir[nm][row["idx"]]), kp.read_record(*ib[nm][row["idx"]]),
                                      ids, 20)
                for t in range(1, len(ids)):
                    ex = full_kl(ref_m.logprobs(ids[:t]), bad_m.logprobs(ids[:t]))
                    self.assertLessEqual(m["kl"][t - 1], ex + 1e-5)
                    est_sum += m["kl"][t - 1]
                    exact_sum += ex
        self.assertGreater(est_sum, 0.7 * exact_sum)

        # REF vs its own replicate, and explicit --no-noise
        rc, out = self.run_kp("compare", "--ref", "REF", "--label", "REF2", "--boot", "50", "--cluster-boot", "50")
        self.assertEqual(rc, 0, out)
        self.assertIsNone(self.report("REF2")["noise_used"])
        self.assertIn("no noise floor", out)

        # decode panel: greedy outputs + logprobs, compared up to the first divergence
        self.assertEqual(self.capture("REF", MockModel(), "mock@1", "--panel", "gen", "--gen-max-tokens", "12")[0], 0)
        self.assertEqual(self.capture("BAD", MockModel(eps=1.5, pert_seed=4), "mock@6", "--panel", "gen",
                                      "--gen-max-tokens", "12")[0], 0)
        rc, out = self.run_kp("compare", "--ref", "REF", "--label", "BAD", "--panel", "gen", "--boot", "50",
                              "--cluster-boot", "50")
        self.assertEqual(rc, 0, out)
        g = self.report("BAD", panel="gen")["gen"]["BAD"]
        self.assertEqual(g["prompts"], 2)
        rc, out = self.run_kp("list")
        self.assertIn("REF2", out)
        self.assertIn("replicate_of=REF", out)

    def test_server_limit_is_fatal(self):
        MockState.max_logprobs = 5
        try:
            rc, out = self.capture("LIMIT", MockModel(), "mock@1", "--sets", "alpha")
        finally:
            MockState.max_logprobs = 20
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
