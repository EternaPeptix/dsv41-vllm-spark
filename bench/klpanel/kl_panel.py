#!/usr/bin/env python3
"""kl_panel.py - teacher-forced logprob fidelity panel for the dsv41 vLLM server.

A run scores the fixed token-id sets in sets/ (built once by build_sets.py) through
/v1/completions with prompt_logprobs=K, max_tokens=1, temperature=0, token-id prompts and no chat
template, so every position gets the server's top-K next-token distribution plus the logprob of
the true next token. `compare` lines two runs up position by position (KL, top-1 agreement,
|d logprob| of the reference argmax, NLL / perplexity delta, bootstrap CIs, noise floor, gate).

  python3 kl_panel.py verify                                    # sets/ vs manifest (no server)
  python3 kl_panel.py smoke                                     # 3 small requests: API shape + runtime projection
  python3 kl_panel.py capture --label REF                       # resumable -> runs/REF/<set>.jsonl
  python3 kl_panel.py capture --label REF2 --replicate-of REF   # noise floor: same config, after a restart
  python3 kl_panel.py capture --label CAND --note "what changed"
  python3 kl_panel.py compare --ref REF --label CAND --tier lossy
  python3 kl_panel.py capture --panel gen --label REF           # optional decode-path panel (greedy + logprobs)
  python3 kl_panel.py list

stdlib only; numpy (if importable) is used for the position-level bootstrap. README.md documents
the estimators, their bias, and the gate thresholds.
"""
from __future__ import annotations

import argparse
import array
import base64
import hashlib
import heapq
import json
import math
import multiprocessing
import os
import random
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    if os.environ.get("KLPANEL_NO_NUMPY"):
        raise ImportError("disabled by KLPANEL_NO_NUMPY")
    import numpy as np
except ImportError:  # pure-Python fallback (normal-approximation position CIs)
    np = None

FORMAT = 2                           # 2 = runs/<label>/<set>.jsonl (one record per sequence per line)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URL = os.environ.get("KLPANEL_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.environ.get("KLPANEL_MODEL", "dsv41")
DEFAULT_K = 20                       # vLLM --max-logprobs default; the dsv41 launcher does not raise it
EPS_MASS = 1e-9                      # masses below this are float32 round-off (see README "estimator")
FLOOR = 1e-30
POS_EDGES = (0, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
LITTLE = sys.byteorder == "little"
METRICS = ("kl", "kl_norm", "kl_rev", "top1", "top5", "dlp1", "nll_r", "nll_c", "acc_r", "acc_c",
           "miss", "tail", "tie")
GATE_METRICS = ("kl", "top1", "dnll")
BOOT_METRICS = ("kl", "kl_norm", "kl_rev", "top1", "dlp1", "dnll")

# Gate tiers (README "Gate"). "excess" = candidate comparison minus the REF-vs-REF2 noise floor,
# paired over identical positions; verdicts use the sequence-cluster bootstrap CI of the excess.
TIERS = {
    # numerically-equivalent changes (kernel swaps, fusions, greedy-eq experiments)
    "equiv": {"kl_abs": 5e-4, "kl_rel": 1.0, "top1_abs": 0.002, "top1_rel": 1.0, "nll_abs": 0.002,
              "nll_two_sided": True},
    # intentionally lossy changes (e.g. W4A16 MoE): knapcio-style budget on this estimator
    "lossy": {"kl_abs": 0.030, "kl_rel": 0.0, "top1_abs": 0.050, "top1_rel": 0.0, "nll_abs": 0.015,
              "nll_two_sided": False},
}


class Fatal(Exception):
    """Stop the whole command with a message."""


class ParseError(Exception):
    """A server response did not have the expected shape."""


# ----------------------------------------------------------------------------------------- utils

def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def seq_sha(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def set_sha(rows) -> str:
    return hashlib.sha256("\n".join(f"{i}:{r['sha256']}" for i, r in enumerate(rows)).encode()).hexdigest()


def file_sha(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def pack_arr(typecode: str, values) -> str:
    a = array.array(typecode, values)
    if a.itemsize != 4:
        raise RuntimeError(f"array('{typecode}') itemsize {a.itemsize} != 4 on this platform")
    if not LITTLE:
        a.byteswap()
    return base64.b64encode(a.tobytes()).decode("ascii")


def unpack_arr(typecode: str, s: str) -> array.array:
    a = array.array(typecode)
    a.frombytes(base64.b64decode(s))
    if not LITTLE:
        a.byteswap()
    return a


def write_json_atomic(path: str, obj) -> None:
    tmp = f"{path}.tmp{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def http_json(url: str, body=None, timeout: float = 60.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ------------------------------------------------------------------------------------------ sets

def read_manifest(base: str) -> dict:
    mpath = os.path.join(base, "sets", "manifest.json")
    if not os.path.isfile(mpath):
        raise Fatal(f"no {mpath}: build the token sets first (build_sets.py, see README)")
    with open(mpath) as f:
        return json.load(f)


def manifest_sha(man: dict) -> str:
    """One digest for the whole panel: the per-set token digests."""
    return hashlib.sha256(json.dumps({n: s["sha256"] for n, s in sorted(man["sets"].items())},
                                     sort_keys=True).encode()).hexdigest()


def set_names(man: dict, panel: str) -> list[str]:
    return [n for n, s in man["sets"].items() if s.get("panel", "tf") == panel]


def load_sets(base: str, names, verify: bool = True):
    man = read_manifest(base)
    out = {}
    for name in names:
        if name not in man["sets"]:
            raise Fatal(f"set {name!r} not in manifest (have: {', '.join(man['sets'])})")
        rows = []
        with open(os.path.join(base, "sets", name + ".jsonl")) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        if verify:
            for i, r in enumerate(rows):
                if r["idx"] != i:
                    raise Fatal(f"{name}: row {i} has idx {r['idx']} (sets/ corrupted?)")
                if seq_sha(r["ids"]) != r["sha256"]:
                    raise Fatal(f"{name}[{r['idx']}]: token sha256 mismatch (sets/ corrupted?)")
            if set_sha(rows) != man["sets"][name]["sha256"]:
                raise Fatal(f"set {name}: sha256 differs from manifest")
        out[name] = rows
    return man, out


# ------------------------------------------------------------------------------ request / parse

def tf_body(model: str, ids, K: int) -> dict:
    return {"model": model, "prompt": list(ids), "max_tokens": 1, "temperature": 0.0,
            "prompt_logprobs": K, "add_special_tokens": False, "return_token_ids": True}


def gen_body(model: str, ids, K: int, max_tokens: int) -> dict:
    return {"model": model, "prompt": list(ids), "max_tokens": max_tokens, "temperature": 0.0,
            "logprobs": K, "add_special_tokens": False, "return_token_ids": True,
            "return_tokens_as_token_ids": True}


def split_topk(entries, y, K: int, where=""):
    """One position's (token, logprob) entries -> (logprob of y, top-K sorted by (-logprob, token)).

    vLLM inserts the scored token (the true next token / the sampled token) first and then the
    top-K into one dict, so a position has K entries when y is in the top-K and K+1 when it is not.
    Sorting by (-logprob, token id) makes the first entry the greedy argmax with torch.argmax's
    first-index tie-break (bf16 logits tie often)."""
    y_lp = None
    for tok, lp in entries:
        if tok == y:
            y_lp = lp
    if len(entries) == K + 1:
        top = [(t, lp) for t, lp in entries if t != y]
        if len(top) != K:
            raise ParseError(f"{where}: {K + 1} entries but the scored token is not among them")
    elif len(entries) == K:
        top = list(entries)
    else:
        raise ParseError(f"{where}: {len(entries)} logprob entries, expected {K} or {K + 1}")
    if y is not None and y_lp is None:
        raise ParseError(f"{where}: scored token {y} missing from the logprob entries")
    top.sort(key=lambda e: (-e[1], e[0]))
    return y_lp, top


def parse_tf(resp: dict, ids, K: int) -> dict:
    try:
        ch = resp["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise ParseError(f"unexpected response: {str(resp)[:300]}") from None
    plp = ch.get("prompt_logprobs")
    if not isinstance(plp, list):
        raise ParseError(f"no prompt_logprobs list in the response (got {type(plp).__name__})")
    n = len(ids)
    if len(plp) != n:
        raise ParseError(f"prompt_logprobs has {len(plp)} entries for {n} prompt tokens")
    pti = ch.get("prompt_token_ids")
    if pti is not None and list(pti) != list(ids):
        raise ParseError("server prompt_token_ids differ from the panel tokens (BOS added? truncation?)")
    ylp, yrank, tid, tlp = [], [], [], []
    for t in range(1, n):
        d = plp[t]
        if not isinstance(d, dict):
            raise ParseError(f"position {t}: expected a dict, got {type(d).__name__}")
        y = ids[t]
        entries, rank_y = [], -1
        for k, v in d.items():
            tok = int(k)
            entries.append((tok, float(v["logprob"])))
            if tok == y and v.get("rank") is not None:
                rank_y = int(v["rank"])
        y_lp, top = split_topk(entries, y, K, where=f"position {t}")
        ylp.append(y_lp)
        yrank.append(rank_y)
        for tok, lp in top:
            tid.append(tok)
            tlp.append(lp)
    return {"ylp": ylp, "yrank": yrank, "tid": tid, "tlp": tlp,
            "gen": ch.get("token_ids"), "finish": ch.get("finish_reason")}


def _tok_id(key: str) -> int:
    if key.startswith("token_id:"):
        return int(key[9:])
    raise ParseError(f"top_logprobs key {key!r} is not 'token_id:N' (return_tokens_as_token_ids ignored?)")


def parse_gen(resp: dict, K: int) -> dict:
    try:
        ch = resp["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise ParseError(f"unexpected response: {str(resp)[:300]}") from None
    out, lp = ch.get("token_ids"), ch.get("logprobs")
    if out is None or not isinstance(lp, dict):
        raise ParseError("generation response lacks token_ids or logprobs")
    tops = lp.get("top_logprobs") or []
    if len(tops) != len(out):
        raise ParseError(f"{len(tops)} top_logprobs rows for {len(out)} generated tokens")
    ylp, tid, tlp = [], [], []
    for j, (y, d) in enumerate(zip(out, tops)):
        if not isinstance(d, dict):
            raise ParseError(f"generated token {j}: no top_logprobs")
        entries = [(_tok_id(k), float(v)) for k, v in d.items()]
        y_lp, top = split_topk(entries, y, K, where=f"generated token {j}")
        ylp.append(y_lp)
        for tok, v in top:
            tid.append(tok)
            tlp.append(v)
    return {"out": list(out), "ylp": ylp, "tid": tid, "tlp": tlp, "finish": ch.get("finish_reason")}


# ------------------------------------------------------------------------------------ estimators

def complete_missing(mp, cap: float, D: float, R: float, Po: float):
    """Minimum-KL completion of the candidate on support tokens it did not list.

    mp:  reference probabilities of the missing tokens (in the reference top-K, not in the
         candidate's top-K list).
    cap: each missing candidate probability is <= the candidate's K-th listed probability.
    D:   candidate mass on listed tokens outside the support (always lands in the 'other' bucket).
    R:   candidate mass outside everything it listed; the missing tokens' mass comes out of R.
    Po:  reference 'other' bucket mass.
    Minimises  sum_m p_m log(p_m / x_m) + Po log(Po / (D + R - sum x))  s.t. 0 < x_m <= cap,
    sum x <= R.  KKT: x_m = min(cap, p_m / lam) (water-filling), closed form below."""
    n = len(mp)
    if n == 0:
        return []
    cap = max(cap, FLOOR)
    order = sorted(range(n), key=lambda i: -mp[i])
    p = [mp[i] for i in order]
    suf = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = suf[i + 1] + p[i]
    xs = None
    if Po > 0.0:
        for k in range(n + 1):                      # k largest capped
            den = D + R - k * cap
            if den <= 0.0:
                break
            lam = (Po + suf[k]) / den
            lc = lam * cap
            if (k == 0 or p[k - 1] >= lc) and (k == n or p[k] <= lc):
                xs = [cap] * k + [p[i] / lam for i in range(k, n)]
                break
        if xs is not None and sum(xs) > R * (1.0 + 1e-9):
            xs = None                               # the residual-mass constraint binds
    if xs is None:
        if n * cap <= R:
            xs = [cap] * n
        else:                                       # water-fill exactly R
            for k in range(n):
                rem = R - k * cap
                if rem <= 0.0:
                    break
                lam = suf[k] / rem
                lc = lam * cap
                if (k == 0 or p[k - 1] >= lc) and p[k] <= lc:
                    xs = [cap] * k + [p[i] / lam for i in range(k, n)]
                    break
            if xs is None:                          # numerical corner: proportional split
                s = suf[0] or 1.0
                xs = [min(cap, R * pi / s) for pi in p]
    out = [0.0] * n
    for j, i in enumerate(order):
        out[i] = max(xs[j], FLOOR)
    return out


def bucket_kl(pt, p_ylp, qt, q_ylp, y):
    """KL(P||Q) on the support S = top-K(P) + {y} plus one 'other' bucket (see README).

    pt, qt: top-K (token, logprob) lists sorted by (-logprob, token); p_ylp / q_ylp: logprob of y
    (the true next token) under P / Q, or y=None. Q values for support tokens Q did not list are
    filled by complete_missing(), so the result is a certified lower bound on the full-vocabulary
    KL (data-processing inequality + minimum over every completion consistent with Q's top-K).
    Returns (kl, kl_norm, n_missing, Q logprob of P's argmax, P 'other' mass), where kl_norm is the
    reference-normalised truncated KL on top-K(P) alone."""
    exp, log = math.exp, math.log
    qd = dict(qt)
    if y is not None:
        qd[y] = q_ylp
    sup = pt
    if y is not None and all(t != y for t, _ in pt):
        sup = pt + [(y, p_ylp)]
    q_listed = 0.0
    for v in qd.values():
        q_listed += exp(v)
    R = 1.0 - q_listed
    if R < EPS_MASS:
        R = EPS_MASS
    cap = exp(min(v for _, v in qt))
    sup_ids = {t for t, _ in sup}
    D = 0.0
    for t, v in qd.items():
        if t not in sup_ids:
            D += exp(v)
    kl = 0.0
    PS = 0.0
    qv = {}
    mt, mp = [], []
    for t, lp in sup:
        p = exp(lp)
        PS += p
        qlp = qd.get(t)
        if qlp is None:
            mt.append(t)
            mp.append(p)
        else:
            qv[t] = qlp
            if p > 0.0:
                kl += p * (lp - qlp)
    Po = 1.0 - PS
    if Po < EPS_MASS:
        Po = 0.0
    X = 0.0
    if mt:
        for t, p, x in zip(mt, mp, complete_missing(mp, cap, D, R, Po)):
            X += x
            lx = log(x)
            qv[t] = lx
            if p > 0.0:
                kl += p * (log(p) - lx)
    if Po > 0.0:
        Qo = D + R - X
        if Qo < FLOOR:
            Qo = FLOOR
        kl += Po * log(Po / Qo)
    PA = 0.0
    QA = 0.0
    for t, lp in pt:
        PA += exp(lp)
        QA += exp(qv[t])
    lPA, lQA = log(max(PA, FLOOR)), log(max(QA, FLOOR))
    kln = 0.0
    for t, lp in pt:
        a = lp - lPA
        kln += exp(a) * (a - (qv[t] - lQA))
    return kl, kln, len(mt), qv[pt[0][0]], Po


def pos_metrics(rt, r_ylp, ct, c_ylp, y):
    """All per-position metrics for reference top-K rt vs candidate top-K ct (see METRICS)."""
    kl, kln, miss, q_r1, tail = bucket_kl(rt, r_ylp, ct, c_ylp, y)
    klr = bucket_kl(ct, c_ylp, rt, r_ylp, y)[0]
    r1, r1lp = rt[0]
    c1 = ct[0][0]
    top5 = 1.0 if [t for t, _ in rt[:5]] == [t for t, _ in ct[:5]] else 0.0
    tie = 1.0 if len(rt) > 1 and rt[1][1] == r1lp else 0.0
    if y is None:
        nan = float("nan")
        nll_r = nll_c = acc_r = acc_c = nan
    else:
        nll_r, nll_c = -r_ylp, -c_ylp
        acc_r = 1.0 if r1 == y else 0.0
        acc_c = 1.0 if c1 == y else 0.0
    return (kl, kln, klr, 1.0 if r1 == c1 else 0.0, top5, abs(r1lp - q_r1), nll_r, nll_c,
            acc_r, acc_c, float(miss), tail, tie)


def _tops(tid, tlp, P: int, K: int):
    return [list(zip(tid[o:o + K], tlp[o:o + K])) for o in range(0, P * K, K)]


def seq_metrics_tf(r: dict, c: dict, ids, K: int) -> dict:
    rylp, cylp = unpack_arr("f", r["ylp"]), unpack_arr("f", c["ylp"])
    P = len(ids) - 1
    if len(rylp) != P or len(cylp) != P:
        raise Fatal(f"{r['set']}[{r['idx']}]: position count mismatch ({len(rylp)}/{len(cylp)} vs {P})")
    rtop = _tops(unpack_arr("i", r["tid"]), unpack_arr("f", r["tlp"]), P, K)
    ctop = _tops(unpack_arr("i", c["tid"]), unpack_arr("f", c["tlp"]), P, K)
    cols = [array.array("d") for _ in METRICS]
    for p in range(P):
        for col, v in zip(cols, pos_metrics(rtop[p], rylp[p], ctop[p], cylp[p], ids[p + 1])):
            col.append(v)
    out = dict(zip(METRICS, cols))
    out["dnll"] = array.array("d", (b - a for a, b in zip(out["nll_r"], out["nll_c"])))
    worst = heapq.nlargest(5, range(P), key=lambda q: out["kl"][q])
    out["_worst"] = [{"pos": q, "kl": out["kl"][q], "y": ids[q + 1], "ref": rtop[q][:3], "cand": ctop[q][:3]}
                     for q in worst]
    return out


def seq_metrics_gen(r: dict, c: dict, K: int) -> dict:
    ro, co = r["out"], c["out"]
    L = min(len(ro), len(co))
    s = next((j for j in range(L) if ro[j] != co[j]), L)
    upto = min(s + 1, L)       # the divergence position still has an identical context
    rtop = _tops(unpack_arr("i", r["tid"]), unpack_arr("f", r["tlp"]), len(ro), K)
    ctop = _tops(unpack_arr("i", c["tid"]), unpack_arr("f", c["tlp"]), len(co), K)
    cols = [array.array("d") for _ in METRICS]
    for j in range(upto):
        for col, v in zip(cols, pos_metrics(rtop[j], None, ctop[j], None, None)):
            col.append(v)
    out = dict(zip(METRICS, cols))
    out["dnll"] = array.array("d", [float("nan")] * upto)
    worst = heapq.nlargest(5, range(upto), key=lambda q: out["kl"][q])
    out["_worst"] = [{"pos": q, "kl": out["kl"][q], "ref": rtop[q][:3], "cand": ctop[q][:3]} for q in worst]
    out["_gen"] = {"identical": ro == co, "first_div": s if s < L else None, "len_r": len(ro),
                   "len_c": len(co), "shared": upto}
    return out


# ---------------------------------------------------------------------------------------- stats

def _quantile(sorted_xs, q):
    if not sorted_xs:
        return float("nan")
    i = min(len(sorted_xs) - 1, max(0, int(math.ceil(q * len(sorted_xs))) - 1))
    return sorted_xs[i]


def _pct(sorted_reps, alpha, B):
    return (sorted_reps[int(math.floor(alpha / 2 * B))],
            sorted_reps[max(0, int(math.ceil((1 - alpha / 2) * B)) - 1)])


def cluster_boot(strata, metrics, B: int, seed: int, alpha: float = 0.05):
    """Sequence-level (cluster) bootstrap, stratified by set: resample whole sequences with
    replacement inside each set, statistic = pooled mean (sum of sums / sum of counts).
    strata: list (per set) of lists of per-sequence dicts {metric: (sum, count)}.
    Returns {metric: (lo, hi)}; the same resamples are used for every metric."""
    rng = random.Random(seed)
    S = {m: [[row[m][0] for row in g] for g in strata] for m in metrics}
    N = {m: [[row[m][1] for row in g] for g in strata] for m in metrics}
    reps = {m: [] for m in metrics}
    for _ in range(B):
        picks = [[rng.randrange(len(g)) for _ in range(len(g))] for g in strata]
        for m in metrics:
            s = 0.0
            n = 0
            for gi, pk in enumerate(picks):
                Sg, Ng = S[m][gi], N[m][gi]
                for j in pk:
                    s += Sg[j]
                    n += Ng[j]
            reps[m].append(s / n if n else float("nan"))
    return {m: _pct(sorted(reps[m]), alpha, B) for m in metrics}


def position_ci(cols: dict, metrics, B: int, seed: int, alpha: float = 0.05):
    """Position-level i.i.d. bootstrap CI (as requested; optimistic because positions inside a
    sequence are correlated - the gate uses cluster_boot). numpy: multinomial resampling (bincount
    weights); without numpy: normal approximation (n is ~1e5, so the two agree closely)."""
    out = {}
    if np is None:
        for m in metrics:
            xs = cols[m]
            n = len(xs)
            if n < 2:
                out[m] = (float("nan"), float("nan"), "normal")
                continue
            mu = math.fsum(xs) / n
            sd = math.sqrt(math.fsum((x - mu) ** 2 for x in xs) / (n - 1))
            h = 1.959964 * sd / math.sqrt(n)
            out[m] = (mu - h, mu + h, "normal")
        return out
    X = np.array([cols[m] for m in metrics], dtype=np.float64)       # (metrics, n)
    n = X.shape[1]
    if n < 2:
        return {m: (float("nan"), float("nan"), "bootstrap") for m in metrics}
    rng = np.random.default_rng(seed)
    reps = np.empty((B, len(metrics)))
    for b in range(B):
        w = np.bincount(rng.integers(0, n, size=n), minlength=n)
        reps[b] = (X @ w) / n
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2], axis=0)
    return {m: (float(lo[i]), float(hi[i]), "bootstrap") for i, m in enumerate(metrics)}


def summarize(seqs, metrics=METRICS):
    """Pooled means + KL tail stats + position bins over a list of per-sequence results."""
    n = sum(len(s["kl"]) for s in seqs)
    res = {"positions": n, "sequences": len(seqs)}
    for m in metrics + ("dnll",):
        tot = 0.0
        cnt = 0
        for s in seqs:
            col = s[m]
            tot += math.fsum(col)
            cnt += len(col)
        res[m] = tot / cnt if cnt else float("nan")
    if n and not math.isnan(res["nll_r"]):
        res["ppl_r"] = math.exp(res["nll_r"])
        res["ppl_c"] = math.exp(res["nll_c"])
        res["ppl_ratio"] = math.exp(res["dnll"])
    allkl = sorted(v for s in seqs for v in s["kl"])
    res["kl_p50"], res["kl_p90"] = _quantile(allkl, 0.5), _quantile(allkl, 0.9)
    res["kl_p99"], res["kl_p999"] = _quantile(allkl, 0.99), _quantile(allkl, 0.999)
    res["kl_max"] = allkl[-1] if allkl else float("nan")
    best = max(seqs, key=lambda s: max(s["kl"]) if s["kl"] else -1.0, default=None)
    if best is not None and best["kl"]:
        q = max(range(len(best["kl"])), key=lambda i: best["kl"][i])
        res["kl_max_at"] = {"set": best["_set"], "idx": best["_idx"], "pos": q}
    bins = []
    edges = list(POS_EDGES) + [1 << 40]
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = t1 = 0.0
        c = 0
        for s in seqs:
            col = s["kl"][lo:hi]
            k += math.fsum(col)
            t1 += math.fsum(s["top1"][lo:hi])
            c += len(col)
        if c:
            bins.append({"pos": f"{lo}-{hi - 1}" if hi < 1 << 40 else f">={lo}", "n": c, "kl": k / c,
                         "top1": t1 / c})
    res["by_position"] = bins
    return res


def seq_sums(s, metrics):
    return {m: (math.fsum(s[m]), len(s[m])) for m in metrics}


# ------------------------------------------------------------------------------- runs (JSONL)

def run_dir(base: str, label: str) -> str:
    return os.path.join(base, "runs", label)


def run_file(base: str, label: str, name: str) -> str:
    return os.path.join(run_dir(base, label), name + ".jsonl")


def scan_run_file(path: str, repair: bool = False):
    """Index a run's <set>.jsonl (append-only, one record per line) -> ({idx: (offset, nbytes,
    seq_sha)}, unreadable_lines). The first record of an idx wins. A torn last line (crash in the
    middle of a write) is skipped and, with repair=True, truncated so appends start on a line."""
    index, bad = {}, 0
    if not os.path.isfile(path):
        return index, bad
    off = good = 0
    with open(path, "rb") as f:
        for line in f:
            n = len(line)
            if not line.endswith(b"\n"):
                break
            try:
                rec = json.loads(line)
                k, sha = int(rec["idx"]), rec["seq_sha"]
            except (ValueError, KeyError, TypeError):
                bad += 1
            else:
                index.setdefault(k, (off, n, sha))
            off += n
            good = off
    if repair and os.path.getsize(path) > good:
        with open(path, "r+b") as f:
            f.truncate(good)
    return index, bad


def read_record(path: str, off: int, n: int) -> dict:
    with open(path, "rb") as f:
        f.seek(off)
        return json.loads(f.read(n))


def run_index(base: str, label: str, names, sets) -> dict:
    """{set: {idx: (path, offset, nbytes)}} for one run; Fatal if a record's tokens differ from sets/."""
    out = {}
    for nm in names:
        path = run_file(base, label, nm)
        index, bad = scan_run_file(path)
        if bad:
            print(f"warning: {path}: {bad} unreadable line(s) ignored", file=sys.stderr)
        want = {r["idx"]: r["sha256"] for r in sets[nm]}
        locs = {}
        for k, (off, n, sha) in index.items():
            if want.get(k) != sha:
                raise Fatal(f"run {label}: {nm}[{k}] was scored on other tokens than sets/{nm}.jsonl holds")
            locs[k] = (path, off, n)
        out[nm] = locs
    return out


def load_meta(base: str, label: str) -> dict:
    p = os.path.join(run_dir(base, label), "meta.json")
    if not os.path.isfile(p):
        raise Fatal(f"no run {label!r} ({p})")
    with open(p) as f:
        meta = json.load(f)
    if meta.get("format") != FORMAT:
        raise Fatal(f"run {label!r} has format {meta.get('format')}, this kl_panel.py reads format {FORMAT}")
    return meta


def run_instances(meta: dict, panel: str) -> set:
    """Server instances (pid@start) whose sessions wrote records of this panel."""
    return {s.get("instance") for s in meta.get("sessions", [])
            if s.get("panel") == panel and s.get("done")} - {None}


def _proc_start_epoch(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        ticks = int(fields[19])
        with open("/proc/stat") as f:
            btime = next(int(line.split()[1]) for line in f if line.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, StopIteration, IndexError):
        return None


def find_server_process() -> dict | None:
    """The local `vllm serve` process (argv + DSV41_/VLLM_/RDMA_/NCCL_ env), for provenance."""
    if not os.path.isdir("/proc"):
        return None
    hits = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [c.decode(errors="replace") for c in f.read().split(b"\0") if c]
        except OSError:
            continue
        if len(argv) >= 2 and "serve" in argv[1:4] and any(a.endswith("vllm") for a in argv[:2]):
            hits.append((int(pid), argv))
    if not hits:
        return None
    pid, argv = min(hits)
    env = {}
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            for kv in f.read().split(b"\0"):
                k, _, v = kv.decode(errors="replace").partition("=")
                if k.startswith(("DSV41_", "VLLM_", "RDMA_AR", "NCCL_", "CUDA_VISIBLE")):
                    env[k] = v
    except OSError:
        pass
    start = _proc_start_epoch(pid)
    return {"pid": pid, "argv": argv, "env": dict(sorted(env.items())), "start_epoch": start,
            "instance": f"{pid}@{int(start) if start else 'unknown'}"}


def collect_provenance(url: str) -> dict:
    prov = {"host": socket.gethostname(), "utc": now_utc(), "url": url}
    for path in ("/v1/models", "/version"):
        try:
            prov[path] = http_json(url.rstrip("/") + path, timeout=15)
        except Exception as e:  # noqa: BLE001 - provenance is best effort
            prov[path] = f"ERROR {type(e).__name__}: {e}"
    launcher = os.path.expanduser("~/dsv41-tp8.sh")
    prov["launcher"] = {"path": launcher, "sha256": file_sha(launcher)}
    prov["server_process"] = find_server_process()
    return prov


def _post_with_retries(url: str, body: dict, timeout: float, retries: int = 3):
    delay = 5.0
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            resp = http_json(url, body, timeout=timeout)
            return resp, time.time() - t0
        except urllib.error.HTTPError as e:
            with e:
                msg = e.read()[:1500].decode(errors="replace")
            if 400 <= e.code < 500:
                raise Fatal(f"HTTP {e.code} from the server: {msg}") from None
            err = f"HTTP {e.code}: {msg[:200]}"
        except (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout) as e:
            err = f"{type(e).__name__}: {e}"
        if attempt == retries:
            raise Fatal(f"request failed after {retries + 1} attempts: {err}")
        print(f"  retry {attempt + 1}/{retries} in {delay:.0f}s ({err})", flush=True)
        time.sleep(delay)
        delay *= 3


# --------------------------------------------------------------------------------------- capture

def cmd_capture(a) -> int:
    base, panel, K = a.base, a.panel, a.k
    man = read_manifest(base)
    names = a.sets.split(",") if a.sets else set_names(man, panel)
    for nm in names:
        if nm not in man["sets"]:
            raise Fatal(f"set {nm!r} not in manifest (have: {', '.join(man['sets'])})")
        if man["sets"][nm].get("panel", "tf") != panel:
            raise Fatal(f"set {nm!r} is not a {panel!r} set")
    man, sets = load_sets(base, names)
    rdir = run_dir(base, a.label)
    os.makedirs(rdir, exist_ok=True)
    mpath = os.path.join(rdir, "meta.json")
    gen_tokens = a.gen_max_tokens
    template = (tf_body(a.model, [], K) if panel == "tf" else gen_body(a.model, [], K, gen_tokens))
    template["prompt"] = "<token ids from sets/>"
    if os.path.isfile(mpath):
        meta = load_meta(base, a.label)
        if a.replicate_of and meta.get("replicate_of") not in (None, a.replicate_of):
            raise Fatal(f"run {a.label} is a replicate of {meta['replicate_of']}, not {a.replicate_of}")
    else:
        meta = {"format": FORMAT, "label": a.label, "created_utc": now_utc(), "replicate_of": None,
                "note": a.note or "", "manifest_sha256": manifest_sha(man), "panels": {}, "sessions": []}
    if a.replicate_of:
        meta["replicate_of"] = a.replicate_of
    if a.note and a.note not in meta.get("note", ""):
        meta["note"] = (meta.get("note", "") + " | " + a.note).strip(" |")
    pm = meta["panels"].get(panel)
    set_info = {nm: {"sha256": man["sets"][nm]["sha256"], "n_seq": len(sets[nm]),
                     "positions": man["sets"][nm].get("positions")} for nm in names}
    if pm is None:
        meta["panels"][panel] = pm = {"K": K, "model": a.model, "request": template, "sets": {}}
    elif pm["K"] != K or pm["request"] != template:
        raise Fatal(f"run {a.label} was captured with a different request ({pm['request']}); use a new label")
    for nm, inf in set_info.items():
        old = pm["sets"].get(nm)
        if old and old["sha256"] != inf["sha256"]:
            raise Fatal(f"run {a.label} holds set {nm} with different tokens; use a new label")
        pm["sets"][nm] = inf

    # resume state: what each <set>.jsonl already holds (a torn last line is cut off)
    jobs, total_pos, have = [], 0, 0
    for nm in names:
        path = run_file(base, a.label, nm)
        index, bad = scan_run_file(path, repair=not a.dry_run)
        if bad:
            print(f"warning: {path}: {bad} unreadable line(s) ignored (their sequences are re-scored)")
        want = {r["idx"]: r["sha256"] for r in sets[nm]}
        for k, (_, _, sha) in index.items():
            if want.get(k) != sha:
                raise Fatal(f"{path}: record {k} was scored on other tokens; use a new label")
        rows = sets[nm][: a.limit] if a.limit else sets[nm]
        for r in rows:
            if r["idx"] in index:
                have += 1
                continue
            jobs.append((nm, r))
            total_pos += (len(r["ids"]) - 1) if panel == "tf" else r.get("max_tokens", gen_tokens)
    unit = "positions" if panel == "tf" else "max new tokens"
    if a.dry_run:
        print(f"[{a.label}] dry run, panel {panel}: {len(jobs)} sequences to score ({total_pos:,} {unit}), "
              f"{have} already done; request {json.dumps(template)}")
        return 0
    if not jobs:                         # complete already: no server contact needed
        write_json_atomic(mpath, meta)
        _finish_session(mpath, None, 0, 0, 0.0, base, a.label, panel, sets)
        print(f"[{a.label}] panel {panel}: nothing to score, {have} sequences already done")
        return 0

    prov = collect_provenance(a.url)
    if not isinstance(prov.get("/v1/models"), dict):
        raise Fatal(f"server not reachable at {a.url}: {prov.get('/v1/models')}")
    sp = prov.get("server_process") or {}
    inst = sp.get("instance")
    prev = [s for s in meta["sessions"] if s.get("panel") == panel and s.get("done", 0) > 0]
    if prev and inst and prev[-1].get("instance") and prev[-1]["instance"] != inst and not a.allow_instance_change:
        raise Fatal(f"run {a.label} was started on server instance {prev[-1]['instance']} but the server is now "
                    f"{inst} (restarted). Mixing instances corrupts the run: use a new label, or pass "
                    f"--allow-instance-change if the restart was intentional and the config identical.")
    if a.replicate_of:
        try:
            rinst = run_instances(load_meta(base, a.replicate_of), panel)
            if inst and inst in rinst:
                print(f"note: {a.label} runs on the same server instance as {a.replicate_of} "
                      f"(same-instance replicate; the gate prefers a replicate captured after a restart)")
        except Fatal:
            print(f"warning: --replicate-of {a.replicate_of}: no such run yet")
    session = {"utc": now_utc(), "panel": panel, "concurrency": a.concurrency, "url": a.url,
               "instance": inst, "provenance": prov, "kl_panel_sha256": file_sha(os.path.abspath(__file__)),
               "done": 0, "positions": 0, "secs": 0.0}
    meta["sessions"].append(session)
    sid = len(meta["sessions"]) - 1
    write_json_atomic(mpath, meta)
    print(f"[{a.label}] panel {panel}: {len(jobs)} sequences to score ({total_pos:,} {unit}), {have} already done;"
          f" K={K} concurrency={a.concurrency} instance={inst}", flush=True)
    url = a.url.rstrip("/") + "/v1/completions"

    def run_one(job):
        nm, r = job
        ids = r["ids"]
        if panel == "tf":
            resp, dt = _post_with_retries(url, tf_body(a.model, ids, K), a.timeout)
            try:
                parsed = parse_tf(resp, ids, K)
            except ParseError as e:
                raise Fatal(f"{nm}[{r['idx']}]: {e}") from None
            npos = len(ids) - 1
            rec = {"v": FORMAT, "panel": panel, "set": nm, "idx": r["idx"], "seq_sha": r["sha256"],
                   "n": len(ids), "K": K, "ylp": pack_arr("f", parsed["ylp"]),
                   "yrank": pack_arr("i", parsed["yrank"]), "tid": pack_arr("i", parsed["tid"]),
                   "tlp": pack_arr("f", parsed["tlp"]), "gen": parsed["gen"], "finish": parsed["finish"],
                   "secs": round(dt, 3), "session": sid, "usage": resp.get("usage")}
        else:
            mt = r.get("max_tokens", gen_tokens)
            resp, dt = _post_with_retries(url, gen_body(a.model, ids, K, mt), a.timeout)
            try:
                parsed = parse_gen(resp, K)
            except ParseError as e:
                raise Fatal(f"{nm}[{r['idx']}]: {e}") from None
            npos = len(parsed["out"])
            rec = {"v": FORMAT, "panel": panel, "set": nm, "idx": r["idx"], "seq_sha": r["sha256"],
                   "n": len(ids), "K": K, "out": parsed["out"], "ylp": pack_arr("f", parsed["ylp"]),
                   "tid": pack_arr("i", parsed["tid"]), "tlp": pack_arr("f", parsed["tlp"]),
                   "finish": parsed["finish"], "secs": round(dt, 3), "session": sid, "usage": resp.get("usage")}
        line = (json.dumps(rec, separators=(",", ":")) + "\n").encode()
        return nm, line, npos

    # Only this thread appends to the files (completion order); records are keyed by idx, so the
    # file order does not matter and the scored positions do not depend on --concurrency.
    fhs = {nm: open(run_file(base, a.label, nm), "ab") for nm in sorted({j[0] for j in jobs})}
    t0 = time.time()
    st = {"done": 0, "npos": 0}
    written = set()

    def write(fut):
        nm, line, npos = fut.result()
        fh = fhs[nm]
        fh.write(line)
        fh.flush()
        written.add(fut)
        st["done"] += 1
        st["npos"] += npos

    def sync():
        for fh in fhs.values():
            fh.flush()
            os.fsync(fh.fileno())

    ex = ThreadPoolExecutor(max_workers=a.concurrency)
    futs = [ex.submit(run_one, j) for j in jobs]
    err, rc, last, last_sync = None, 0, 0.0, time.time()
    try:
        for fut in as_completed(futs):
            write(fut)
            el = time.time() - t0
            if time.time() - last_sync > 30:
                sync()
                last_sync = time.time()
            if el - last > 20 or st["done"] == len(jobs):
                last = el
                rate = st["npos"] / el if el > 0 else 0.0
                eta = (total_pos - st["npos"]) / rate / 60 if rate > 0 else float("nan")
                print(f"  {st['done']}/{len(jobs)} seqs, {st['npos']:,}/{total_pos:,} {unit}, {rate:,.0f}/s, "
                      f"eta {eta:.1f} min", flush=True)
    except KeyboardInterrupt:
        rc = 130
    except Exception as e:  # noqa: BLE001 - Fatal or unexpected: stop, but keep what finished
        err = e
    finally:
        if rc or err:
            for f in futs:
                f.cancel()
            ex.shutdown(wait=True)
            for f in futs:   # requests that were in flight and finished cleanly are kept too
                if f not in written and f.done() and not f.cancelled() and f.exception() is None:
                    write(f)
        else:
            ex.shutdown(wait=True)
        sync()
        for fh in fhs.values():
            fh.close()
        wall = time.time() - t0
        _finish_session(mpath, sid, st["done"], st["npos"], wall, base, a.label, panel, sets)
    if err is not None:
        raise err
    if rc:
        print(f"\ninterrupted after {st['done']} sequences; they are saved - rerun the same command to resume")
        return rc
    print(f"[{a.label}] done: {st['done']} sequences, {st['npos']:,} {unit} in {wall:.0f}s "
          f"({st['npos'] / max(wall, 1e-9):,.0f}/s)", flush=True)
    return 0


def _finish_session(mpath, sid, done, npos, secs, base, label, panel, sets):
    with open(mpath) as f:
        meta = json.load(f)
    if sid is not None:
        s = meta["sessions"][sid]
        s.update({"done": done, "positions": npos, "secs": round(secs, 1), "end_utc": now_utc()})
    pm = meta["panels"][panel]
    complete = True
    for nm, inf in pm["sets"].items():
        index, _ = scan_run_file(run_file(base, label, nm))
        if len(index) < inf["n_seq"]:
            complete = False
    pm["complete"] = complete
    write_json_atomic(mpath, meta)


# --------------------------------------------------------------------------------------- compare

def _seq_job(args):
    panel, nm, idx, ids, K, rloc, cloc = args
    r, c = read_record(*rloc), read_record(*cloc)
    for x in (r, c):
        if x["K"] != K:
            raise Fatal(f"{nm}[{idx}]: K mismatch")
    out = seq_metrics_tf(r, c, ids, K) if panel == "tf" else seq_metrics_gen(r, c, K)
    out["_set"], out["_idx"] = nm, idx
    return out


def _compute_pair(ridx, cidx, panel, keys, rows, K, jobs):
    args = [(panel, nm, i, rows[nm][i]["ids"], K, ridx[nm][i], cidx[nm][i]) for nm, i in keys]
    if jobs > 1 and len(args) > 1:
        ctx = multiprocessing.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        with ctx.Pool(jobs) as pool:
            res = pool.map(_seq_job, args, chunksize=1)
    else:
        res = [_seq_job(x) for x in args]
    return {(s["_set"], s["_idx"]): s for s in res}


def _verdict(lo, hi, thr, op):
    if op == "<=":
        return "PASS" if hi <= thr else ("FAIL" if lo > thr else "BORDERLINE")
    if op == ">=":
        return "PASS" if lo >= thr else ("FAIL" if hi < thr else "BORDERLINE")
    # two-sided |x| <= thr
    if -thr <= lo and hi <= thr:
        return "PASS"
    return "FAIL" if (lo > thr or hi < -thr) else "BORDERLINE"


def gate(tier: str, cand: dict, noise: dict | None, ex_ci: dict) -> dict:
    t = TIERS[tier]
    has_noise = noise is not None
    kl_noise = noise["kl"] if has_noise else 0.0
    top1_noise = noise["top1"] if has_noise else 1.0
    dnll_noise = noise["dnll"] if has_noise else 0.0
    kl_thr = max(t["kl_abs"], t["kl_rel"] * kl_noise)
    top1_thr = max(t["top1_abs"], t["top1_rel"] * (1.0 - top1_noise))
    crit = []
    v = cand["kl"] - kl_noise
    lo, hi = ex_ci["kl"]
    crit.append({"name": "excess KL (nats/pos)", "value": v, "ci": [lo, hi], "op": "<=", "thr": kl_thr,
                 "verdict": _verdict(lo, hi, kl_thr, "<=")})
    v = cand["top1"] - top1_noise
    lo, hi = ex_ci["top1"]
    crit.append({"name": "top-1 agreement vs noise", "value": v, "ci": [lo, hi], "op": ">=", "thr": -top1_thr,
                 "verdict": _verdict(lo, hi, -top1_thr, ">=")})
    if not math.isnan(cand.get("dnll", float("nan"))):
        v = cand["dnll"] - dnll_noise
        lo, hi = ex_ci["dnll"]
        op = "|<=|" if t["nll_two_sided"] else "<="
        crit.append({"name": "excess dNLL (nats/token)", "value": v, "ci": [lo, hi], "op": op,
                     "thr": t["nll_abs"], "verdict": _verdict(lo, hi, t["nll_abs"], op)})
    order = {"PASS": 0, "BORDERLINE": 1, "FAIL": 2}
    overall = max((c["verdict"] for c in crit), key=order.get)
    return {"tier": tier, "noise_floor": has_noise, "criteria": crit, "verdict": overall}


def _fmt(x, nd=5):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def cmd_compare(a) -> int:
    base, panel = a.base, a.panel
    rmeta, cmeta = load_meta(base, a.ref), load_meta(base, a.label)
    if panel not in rmeta["panels"] or panel not in cmeta["panels"]:
        raise Fatal(f"both runs need panel {panel!r}")
    rp, cp = rmeta["panels"][panel], cmeta["panels"][panel]
    if rp["K"] != cp["K"]:
        raise Fatal(f"K differs: {rp['K']} vs {cp['K']}")
    K = rp["K"]
    names = [n for n in rp["sets"] if n in cp["sets"]]
    if a.sets:
        names = [n for n in names if n in a.sets.split(",")]
    if not names:
        raise Fatal("the two runs have no set in common")
    for nm in names:
        if rp["sets"][nm]["sha256"] != cp["sets"][nm]["sha256"]:
            raise Fatal(f"set {nm}: the two runs scored different tokens")
    man, sets = load_sets(base, names)
    for nm in names:
        if man["sets"][nm]["sha256"] != rp["sets"][nm]["sha256"]:
            raise Fatal(f"set {nm}: sets/ changed since {a.ref} was captured")
    rows = {nm: {r["idx"]: r for r in sets[nm]} for nm in names}
    warnings = []
    idx_r = run_index(base, a.ref, names, sets)
    idx_c = run_index(base, a.label, names, sets)

    def present(ix):
        return {(nm, i) for nm in names for i in ix[nm]}

    keys = sorted(present(idx_r) & present(idx_c))
    total = sum(len(sets[nm]) for nm in names)
    if not keys:
        raise Fatal("no sequences captured in both runs")
    if len(keys) < total:
        warnings.append(f"only {len(keys)}/{total} sequences are in both runs (incomplete capture?)")
    # noise floor: explicit --noise, else every replicate of REF
    if a.no_noise:
        noise_labels = []
    elif a.noise:
        noise_labels = a.noise.split(",")
    else:
        noise_labels = []
        rdir = os.path.join(base, "runs")
        for lab in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
            if lab in (a.ref, a.label):
                continue
            try:
                m = load_meta(base, lab)
            except (Fatal, ValueError):
                continue
            if m.get("replicate_of") == a.ref and panel in m.get("panels", {}):
                noise_labels.append(lab)
    t0 = time.time()
    cand = _compute_pair(idx_r, idx_c, panel, keys, rows, K, a.jobs)
    noises, noise_meta = {}, {}
    for lab in noise_labels:
        nmeta = load_meta(base, lab)
        npn = nmeta["panels"].get(panel, {})
        if npn.get("K") != K or any(npn.get("sets", {}).get(nm, {}).get("sha256") != rp["sets"][nm]["sha256"]
                                    for nm in names if nm in npn.get("sets", {})):
            warnings.append(f"noise run {lab}: different K or tokens: skipped")
            continue
        idx_n = run_index(base, lab, [nm for nm in names if nm in npn.get("sets", {})], sets)
        for nm in names:
            idx_n.setdefault(nm, {})
        nk = sorted(set(keys) & present(idx_n))
        if len(nk) < len(keys):
            warnings.append(f"noise run {lab} lacks {len(keys) - len(nk)} of the compared sequences: skipped")
            continue
        noises[lab] = _compute_pair(idx_r, idx_n, panel, keys, rows, K, a.jobs)
        noise_meta[lab] = nmeta
    print(f"computed {1 + len(noises)} comparison(s) over {len(keys)} sequences in {time.time() - t0:.0f}s",
          flush=True)

    groups = {nm: [k for k in keys if k[0] == nm] for nm in names}
    groups = {nm: g for nm, g in groups.items() if g}

    def summ(pair):
        out = {nm: summarize([pair[k] for k in g]) for nm, g in groups.items()}
        out["overall"] = summarize([pair[k] for k in keys])
        return out

    cs = summ(cand)
    ns = {lab: summ(p) for lab, p in noises.items()}
    # Which replicate is the noise floor: one captured on another server instance than REF (a
    # restart replicate, like every candidate) if there is one; among those the largest KL.
    r_inst = run_instances(rmeta, panel)
    cross = {}
    for lab in ns:
        n_inst = run_instances(noise_meta[lab], panel)
        cross[lab] = None if not (r_inst and n_inst) else r_inst.isdisjoint(n_inst)
    pool = [lab for lab in ns if cross[lab] is not False] or list(ns)
    noise_lab = max(pool, key=lambda lab: ns[lab]["overall"]["kl"]) if pool else None
    if noise_lab is not None and cross[noise_lab] is False:
        warnings.append(f"noise floor {noise_lab} ran on the same server instance as {a.ref}: restart variance "
                        f"is not in the floor (capture a replicate after a restart)")
    nsum = ns.get(noise_lab)
    npair = noises.get(noise_lab)
    gm = ("kl", "kl_norm", "kl_rev", "top1", "dlp1") if panel == "gen" else BOOT_METRICS

    # CIs: position-level (requested) and sequence-cluster (used by the gate)
    def cis(pair, sel, with_pos=True):
        pos = None
        if with_pos:
            cols = {m: [v for k in sorted(sel) for v in pair[k][m]] for m in gm}
            pos = position_ci(cols, gm, a.boot, a.seed)
        strata = [[seq_sums(pair[k], gm) for k in groups[nm] if k in sel] for nm in groups]
        strata = [s for s in strata if s]
        clu = cluster_boot(strata, gm, a.cluster_boot, a.seed)
        return pos, clu

    def excess_ci(sel):
        strata = []
        for nm in groups:
            rows_ = []
            for k in groups[nm]:
                if k not in sel:
                    continue
                row = {}
                for m in GATE_METRICS:
                    sc, nc = math.fsum(cand[k][m]), math.fsum(npair[k][m]) if npair else 0.0
                    row[m] = (sc - nc, len(cand[k][m]))
                rows_.append(row)
            if rows_:
                strata.append(rows_)
        return cluster_boot(strata, GATE_METRICS, a.cluster_boot, a.seed + 1)

    report = {"ref": a.ref, "cand": a.label, "panel": panel, "K": K, "utc": now_utc(),
              "noise_runs": {lab: {"cross_instance": cross[lab], "kl": ns[lab]["overall"]["kl"]} for lab in ns},
              "noise_used": noise_lab, "sets": {}, "warnings": warnings,
              "estimator": "bucket KL on top-K(ref)+{true token} plus 'other' bucket, min-KL completion"
                           " (certified lower bound on full-vocab KL); see README",
              "ci": {"position": "multinomial bootstrap" if np is not None else "normal approximation",
                     "cluster": f"sequence bootstrap stratified by set, B={a.cluster_boot}",
                     "B_position": a.boot, "seed": a.seed},
              "ref_note": rmeta.get("note", ""), "cand_note": cmeta.get("note", "")}
    for nm in list(groups) + ["overall"]:
        sel = set(keys) if nm == "overall" else set(groups[nm])
        pos, clu = cis(cand, sel)
        ent = {"cand": cs[nm], "ci_position": pos, "ci_cluster": clu}
        if nsum is not None:
            ent["noise"] = nsum[nm]
            ent["noise_ci_cluster"] = cis(npair, sel, with_pos=False)[1]
            if panel == "tf":
                ent["excess_ci_cluster"] = excess_ci(sel)
        report["sets"][nm] = ent
    if panel == "gen":
        gstats = {}
        for lab, pair in [(a.label, cand)] + list(noises.items()):
            g = [pair[k]["_gen"] for k in keys]
            fd = sorted(x["first_div"] for x in g if x["first_div"] is not None)
            gstats[lab] = {"identical": sum(x["identical"] for x in g), "prompts": len(g),
                           "median_first_div": _quantile(fd, 0.5) if fd else None,
                           "shared_positions": sum(x["shared"] for x in g)}
        report["gen"] = gstats

    # gate (teacher-forced panel only; the decode panel is reported, not gated)
    ov = report["sets"]["overall"]
    if panel == "tf":
        if "excess_ci_cluster" in ov:
            ex = ov["excess_ci_cluster"]
        else:   # no replicate: noise floor taken as 0 (KL 0, agreement 1, dNLL 0)
            ex = {"kl": ov["ci_cluster"]["kl"], "top1": tuple(x - 1.0 for x in ov["ci_cluster"]["top1"]),
                  "dnll": ov["ci_cluster"]["dnll"]}
            warnings.append("no noise floor (no REF2 replicate): gate uses absolute thresholds; capture REF2")
        report["gates"] = {t: gate(t, ov["cand"], ov.get("noise"), ex) for t in TIERS}
        # per-set and tail warnings
        thr = report["gates"][a.tier]["criteria"][0]["thr"]
        for nm in groups:
            e = report["sets"][nm]
            exk = e["cand"]["kl"] - (e["noise"]["kl"] if "noise" in e else 0.0)
            if exk > 2 * thr:
                warnings.append(f"set {nm}: excess KL {exk:.4g} > 2x the {a.tier} threshold ({thr:.3g})")
        if nsum is not None:
            nk = nsum["overall"]["kl"]
            if nk > 0.005:
                warnings.append(f"noise floor KL {nk:.4g} is high for a same-config replicate: check {noise_lab}")
            c999, n999 = ov["cand"]["kl_p999"], nsum["overall"]["kl_p999"]
            if a.tier == "equiv" and c999 > 4 * n999 + 0.05:
                warnings.append(f"KL p99.9 {c999:.3g} vs noise {n999:.3g}: localized divergence (see worst positions)")
        miss = ov["cand"]["miss"] / (K + 1)
        if miss > 0.10:
            warnings.append(f"{100 * miss:.1f}% of support entries were completed (candidate top-K misses many "
                            f"reference top-K tokens): KL is a loose lower bound here")
    rc_inst, cc_inst = r_inst, run_instances(cmeta, panel)
    if rc_inst & cc_inst:
        warnings.append("REF and CAND share a server instance (same-instance comparison)")
    rconc = {s.get("concurrency") for s in rmeta.get("sessions", []) if s.get("panel") == panel and s.get("done")}
    cconc = {s.get("concurrency") for s in cmeta.get("sessions", []) if s.get("panel") == panel and s.get("done")}
    if rconc and cconc and rconc != cconc:
        warnings.append(f"capture concurrency differs ({sorted(rconc)} vs {sorted(cconc)}): batch composition "
                        f"adds noise the REF2 floor may not include")
    worst = heapq.nlargest(25, (dict(w, set=s["_set"], idx=s["_idx"]) for s in cand.values() for w in s["_worst"]),
                           key=lambda w: w["kl"])
    report["worst_positions"] = worst
    tag = f"{panel}_vs_{a.ref}" if panel != "tf" else f"vs_{a.ref}"
    out_json = os.path.join(run_dir(base, a.label), f"compare_{tag}.json")
    write_json_atomic(out_json, report)
    text = render_report(report, a.tier)
    out_txt = os.path.join(run_dir(base, a.label), f"compare_{tag}.txt")
    with open(out_txt, "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {out_txt} and {out_json}")
    if a.strict and panel == "tf" and report["gates"][a.tier]["verdict"] == "FAIL":
        return 1
    return 0


def _table(headers, rows):
    w = [max([len(str(h))] + [len(str(r[i])) for r in rows]) for i, h in enumerate(headers)]

    def line(cells):
        return "  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(cells))

    return [line(headers)] + [line(r) for r in rows]


def _ci(c, nd=5):
    return f"[{_fmt(c[0], nd)},{_fmt(c[1], nd)}]" if c else "-"


def render_report(rep: dict, tier: str) -> str:
    L = []
    noise = rep.get("noise_used")
    L.append(f"KL panel [{rep['panel']}]  ref={rep['ref']}  cand={rep['cand']}  K={rep['K']}  "
             f"noise floor={noise or 'none'}  ({rep['utc']})")
    if rep.get("cand_note"):
        L.append(f"cand note: {rep['cand_note']}")
    L.append("KL = bucket KL(ref||cand) on top-K(ref) + true token + 'other' bucket, a lower bound on the "
             "full-vocab KL (nats/position); [..] = 95% sequence-cluster bootstrap CI")
    K1 = rep["K"] + 1
    r1, r2, r3 = [], [], []
    for nm, e in rep["sets"].items():
        c, nz = e["cand"], e.get("noise")
        exc = e.get("excess_ci_cluster", {})
        r1.append([nm, f"{c['positions']:,}", _fmt(c["kl"]), _ci(e["ci_cluster"]["kl"]),
                   _fmt(nz["kl"] if nz else None), _fmt(c["kl"] - nz["kl"] if nz else None), _ci(exc.get("kl")),
                   _fmt(c["kl_norm"]), _fmt(c["kl_rev"]), _fmt(c["kl_p99"], 3), _fmt(c["kl_p999"], 3),
                   _fmt(c["kl_max"], 3), f"{100 * c['miss'] / K1:.2f}", f"{c['tail']:.4f}"])
        dn = e["ci_cluster"].get("dnll")
        r2.append([nm, _fmt(c["top1"], 4), _fmt(nz["top1"] if nz else None, 4), _ci(exc.get("top1"), 4),
                   _fmt(c["top5"], 4), _fmt(c["dlp1"], 4), _fmt(c.get("nll_r"), 4), _fmt(c.get("dnll"), 5),
                   _ci(dn, 5), _fmt(nz.get("dnll") if nz else None, 5), _fmt(c.get("ppl_ratio"), 4),
                   _fmt(c.get("acc_r"), 4), _fmt(c.get("acc_c"), 4), f"{100 * c['tie']:.2f}"])
        if nz:
            ncl = e.get("noise_ci_cluster", {})
            r3.append([nm, _fmt(nz["kl"]), _ci(ncl.get("kl")), _fmt(nz["kl_p999"], 3), _fmt(nz["kl_max"], 3),
                       _fmt(nz["top1"], 4), _ci(ncl.get("top1"), 4), _fmt(nz.get("dnll"), 5)])
    L += _table(["set", "positions", "KL", "KL CI", "noiseKL", "excessKL", "excess CI", "KLnorm", "KLrev",
                 "p99", "p99.9", "max", "miss%", "refTail"], r1)
    L.append("")
    L += _table(["set", "top1", "noise", "excess top1 CI", "top5", "|dlp1|", "NLLref", "dNLL", "dNLL CI",
                 "noise dNLL", "ppl x", "acc ref", "acc cand", "tie%"], r2)
    if r3:
        L.append("")
        L.append(f"noise floor = {rep['ref']} vs {noise} (same config):")
        L += _table(["set", "KL", "KL CI", "p99.9", "max", "top1", "top1 CI", "dNLL"], r3)
    ov = rep["sets"]["overall"]
    pci = ov["ci_position"]["kl"]
    L.append(f"overall KL position-level 95% CI ({pci[2]}): {_ci(pci)}  (optimistic: positions in a sequence "
             f"are correlated; the gate uses the cluster CI)")
    L.append("KL by position: " + "  ".join(f"{b['pos']}:{_fmt(b['kl'], 4)}" for b in ov["cand"]["by_position"]))
    if ov["cand"].get("kl_max_at"):
        m = ov["cand"]["kl_max_at"]
        L.append(f"max-position KL {ov['cand']['kl_max']:.4f} at {m['set']}[{m['idx']}] pos {m['pos']}")
    if rep.get("noise_runs"):
        L.append("noise floor runs: " + ", ".join(
            f"{lab} (KL {v['kl']:.3g}, {'restart' if v['cross_instance'] else 'same-instance' if v['cross_instance'] is False else 'instance unknown'})"
            for lab, v in rep["noise_runs"].items()) + f"; used: {noise}")
    if "gen" in rep:
        for lab, g in rep["gen"].items():
            L.append(f"decode panel {lab}: identical greedy outputs {g['identical']}/{g['prompts']}, median first "
                     f"divergence {g['median_first_div']}, shared positions {g['shared_positions']}")
    if "gates" in rep:
        for t, g in rep["gates"].items():
            mark = " <==" if t == tier else ""
            L.append(f"GATE {t:5} -> {g['verdict']}{mark}" + ("" if g["noise_floor"] else "  (no noise floor)"))
            for c in g["criteria"]:
                L.append(f"    {c['name']:28} {c['value']:+.5f}  CI [{c['ci'][0]:+.5f},{c['ci'][1]:+.5f}]  "
                         f"{c['op']} {c['thr']:+.5f}  {c['verdict']}")
    for w in rep["warnings"]:
        L.append(f"WARNING: {w}")
    if rep.get("worst_positions"):
        w = rep["worst_positions"][0]
        L.append(f"worst position: {w['set']}[{w['idx']}] pos {w['pos']} KL {w['kl']:.3f} "
                 f"ref top {w['ref'][0]} cand top {w['cand'][0]} (more in the JSON)")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------- smoke/list/verify

def cmd_smoke(a) -> int:
    base, K = a.base, a.k
    man = read_manifest(base)
    tf = set_names(man, "tf")
    gen = set_names(man, "gen")
    man, sets = load_sets(base, tf[:1] + gen[:1])
    sdir = run_dir(base, "_smoke")
    os.makedirs(sdir, exist_ok=True)
    print(f"server {a.url}:")
    prov = collect_provenance(a.url)
    models = prov.get("/v1/models")
    if not isinstance(models, dict):
        raise Fatal(f"server not reachable: {models}")
    ent = (models.get("data") or [{}])[0]
    print(f"  model {ent.get('id')} max_model_len {ent.get('max_model_len')} instance "
          f"{(prov.get('server_process') or {}).get('instance')}")
    seq = sets[tf[0]][0]["ids"]
    url = a.url.rstrip("/") + "/v1/completions"
    rate = None
    for n in (min(64, len(seq)), len(seq)):
        ids = seq[:n]
        resp, dt = _post_with_retries(url, tf_body(a.model, ids, K), a.timeout, retries=0)
        with open(os.path.join(sdir, f"tf_{n}.json"), "w") as f:
            json.dump(resp, f)
        p = parse_tf(resp, ids, K)
        nk = len(resp["choices"][0]["prompt_logprobs"][1])
        print(f"  tf n={n}: {dt:.2f}s ({(n - 1) / dt:,.0f} pos/s), entries/pos {nk} (K={K}), "
              f"pos1 y={ids[1]} lp={p['ylp'][0]:.4f} rank={p['yrank'][0]} top1={p['tid'][0]}:{p['tlp'][0]:.4f}, "
              f"mean NLL {-sum(p['ylp']) / len(p['ylp']):.3f}")
        rate = (n - 1) / dt
    tot = sum(s.get("positions") or 0 for s in man["sets"].values() if s.get("panel", "tf") == "tf")
    if rate:
        print(f"  projected full tf capture at concurrency 1: {tot:,} positions / {rate:,.0f} pos/s = "
              f"{tot / rate / 60:.1f} min (concurrency 4 is usually faster)")
    if gen:
        ids = sets[gen[0]][0]["ids"]
        resp, dt = _post_with_retries(url, gen_body(a.model, ids, K, 16), a.timeout, retries=0)
        with open(os.path.join(sdir, "gen_16.json"), "w") as f:
            json.dump(resp, f)
        g = parse_gen(resp, K)
        print(f"  gen 16 tokens: {dt:.2f}s, out {g['out'][:8]}..., top1 lp {[round(x, 4) for x in g['ylp'][:4]]}")
    print(f"raw responses in {sdir}/")
    return 0


def cmd_list(a) -> int:
    rdir = os.path.join(a.base, "runs")
    if not os.path.isdir(rdir):
        print("no runs yet")
        return 0
    for lab in sorted(os.listdir(rdir)):
        try:
            m = load_meta(a.base, lab)
        except (Fatal, ValueError):
            continue
        for panel, pm in m.get("panels", {}).items():
            n = sum(len(scan_run_file(run_file(a.base, lab, nm))[0]) for nm in pm["sets"])
            tot = sum(s["n_seq"] for s in pm["sets"].values())
            inst = sorted(run_instances(m, panel)) or ["?"]
            print(f"{lab:16} {panel:4} {n:>4}/{tot:<4} complete={pm.get('complete', False)!s:5} "
                  f"replicate_of={m.get('replicate_of') or '-':10} instance={','.join(inst)} "
                  f"created={m['created_utc']} {m.get('note', '')}")
    return 0


def cmd_verify(a) -> int:
    man = read_manifest(a.base)
    man, sets = load_sets(a.base, list(man["sets"]))
    bad = 0
    for fn, want in (man.get("files") or {}).items():
        got = file_sha(os.path.join(a.base, "sets", fn))
        if got != want:
            bad += 1
            print(f"MISMATCH {fn}: {got} != manifest {want}")
    for nm, rows in sets.items():
        s = man["sets"][nm]
        lens = sorted({len(r["ids"]) for r in rows})
        print(f"  {nm:6} {s.get('panel', 'tf'):3} {len(rows):3} seqs  lengths {lens[0]}..{lens[-1]}  "
              f"{s.get('positions') or 0:>7,} {'positions' if s.get('panel', 'tf') == 'tf' else 'max new tokens'}  "
              f"sha {s['sha256'][:12]}")
    print(f"manifest sha {manifest_sha(man)[:16]}; tf positions {man.get('tf_positions')}; "
          f"{'OK' if not bad else f'{bad} FILE MISMATCH(ES)'}")
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.environ.get("KLPANEL_BASE", HERE), help="dir holding sets/ and runs/")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def server_args(p):
        p.add_argument("--url", default=DEFAULT_URL)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--k", type=int, default=DEFAULT_K, help="prompt_logprobs / logprobs K (server max 20)")
        p.add_argument("--timeout", type=float, default=1800.0)

    p = sub.add_parser("capture", help="score the sets on the live server (resumable)")
    p.add_argument("--label", required=True)
    p.add_argument("--panel", choices=("tf", "gen"), default="tf",
                   help="tf = teacher-forced prompt scoring (default); gen = greedy decode with logprobs")
    p.add_argument("--sets", help="comma list (default: every set of the panel)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--replicate-of", help="mark this run as a same-config replicate (noise floor) of REF")
    p.add_argument("--note", default="")
    p.add_argument("--limit", type=int, default=0, help="first N sequences per set only (partial run)")
    p.add_argument("--gen-max-tokens", type=int, default=256)
    p.add_argument("--allow-instance-change", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="show what would be scored; no server contact")
    server_args(p)

    p = sub.add_parser("compare", help="compare CAND against REF (+ noise floor)")
    p.add_argument("--ref", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--panel", choices=("tf", "gen"), default="tf")
    p.add_argument("--noise", help="comma list of noise-floor runs (default: every --replicate-of REF run)")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--tier", choices=tuple(TIERS), default="equiv")
    p.add_argument("--sets")
    p.add_argument("--boot", type=int, default=1000, help="position-level bootstrap replicates")
    p.add_argument("--cluster-boot", type=int, default=2000, help="sequence-cluster bootstrap replicates")
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--strict", action="store_true", help="exit 1 when the chosen tier FAILs")

    p = sub.add_parser("smoke", help="3 small requests: checks the response shape, projects runtime")
    server_args(p)
    sub.add_parser("list", help="list runs")
    sub.add_parser("verify", help="check sets/ against manifest.json (no server contact)")
    a = ap.parse_args(argv)
    try:
        return {"capture": cmd_capture, "compare": cmd_compare, "smoke": cmd_smoke, "list": cmd_list,
                "verify": cmd_verify}[a.cmd](a)
    except Fatal as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
