#!/usr/bin/env python3
"""dbgrun.py <dumpdir|-> <out.jsonl> <label> <cases> <reps> <dump 0|1> [mode cold|warm] [max_tokens]
(ON spark-one, stdlib; goal window 5 Determinism phase)

cases: comma list of code:<L> | prose:<L> | l3:<case id in ~/dsv41-g4l3/cases/l3cases.jsonl>
Token-id prompts = [BOS] + the first L-1 ids of the source text (same sources as detprobe.py), greedy,
ignore_eos, logprobs of the sampled token per step. l3 cases = the recorded chat request, non-streaming.
cold = fresh cache_salt per request; warm = one salt per case (run 0 prefills, later runs hit the cache).
With dump=1: <dumpdir>/ON exists only while each request runs (rank 0 dumps), then that request's
r0_*.pt files move to <dumpdir>/<label>-<case>-<i>/. One request at a time on an idle server.
Prints per case: distinct token sequences, distinct logprob sequences, first logprob divergence vs run 0.
"""
import glob
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
ddir, out_path, label, cases_s, reps_s, dump_s = sys.argv[1:7]
mode = sys.argv[7] if len(sys.argv) > 7 else "cold"
max_tokens = int(sys.argv[8]) if len(sys.argv) > 8 else 128
reps, dump = int(reps_s), dump_s == "1"
SRC = {
    "prose": ["/usr/share/common-licenses/GPL-3", "/usr/share/common-licenses/Apache-2.0",
              "/usr/share/common-licenses/GFDL-1.3"],
    "code": ["/home/jeweled/venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py.bak-nondet"],
}
L3 = "/home/jeweled/dsv41-g4l3/cases/l3cases.jsonl"
_V = "/home/jeweled/venvs/vllm032/lib/python3.12/site-packages/vllm"
# "big": every .py under vllm/{v1/core,engine,config} in sorted order (~1.43 MB, untouched by g5det)
SRC["big"] = sorted(os.path.join(dp, f) for top in ("v1/core", "engine", "config")
                    for dp, _, fs in os.walk(os.path.join(_V, top)) for f in fs if f.endswith(".py"))


def post(path, body, timeout=3600):
    r = urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=timeout)
    return json.loads(r.read().decode())


def idle():
    for _ in range(900):
        t = urllib.request.urlopen(BASE + "/metrics", timeout=5).read().decode()
        s = sum(float(l.split()[-1]) for l in t.splitlines()
                if l.startswith("vllm:num_requests_running{") or l.startswith("vllm:num_requests_waiting{"))
        if s == 0:
            return
        time.sleep(1)
    raise SystemExit("server not idle")


_ids = {}


def src_ids(name):
    if name not in _ids:
        text = "".join(open(p, encoding="utf-8", errors="replace").read() for p in SRC[name])
        _ids[name] = post("/tokenize", {"model": "dsv41", "prompt": text, "add_special_tokens": False})["tokens"]
    return _ids[name]


def l3case(cid):
    for line in open(L3):
        c = json.loads(line)
        if c["id"] == cid:
            return c
    raise SystemExit("no l3 case " + cid)


def bits(x):
    return None if x is None else struct.pack("<d", float(x)).hex()


def first_div(a, b):
    for j in range(min(len(a), len(b))):
        if a[j] != b[j]:
            return j
    return None if len(a) == len(b) else min(len(a), len(b))


def one(case, salt):
    kind, arg = case.split(":", 1)
    if kind in SRC:
        L = int(arg)
        body = {"model": "dsv41", "prompt": [0] + src_ids(kind)[: L - 1], "max_tokens": max_tokens,
                "temperature": 0, "ignore_eos": True, "return_token_ids": True, "cache_salt": salt, "logprobs": 1}
        d = post("/v1/completions", body)
        ch = d["choices"][0]
        toks = ch.get("token_ids") or []
        lps = [bits(x) for x in ((ch.get("logprobs") or {}).get("token_logprobs") or [])]
    else:
        c = l3case(arg)
        body = dict(c["body"])
        body.update({"stream": False, "return_token_ids": True, "cache_salt": salt, "logprobs": True,
                     "top_logprobs": 1, "temperature": 0})
        d = post(c["path"], body)
        ch = d["choices"][0]
        toks = ch.get("token_ids") or []
        lps = [bits(t.get("logprob")) for t in (((ch.get("logprobs") or {}).get("content")) or [])]
    u = d.get("usage") or {}
    return toks, lps, u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens")


fo = open(out_path, "a")
for case in cases_s.split(","):
    tag = case.replace(":", "_")
    fixed = "g5dbg-%s-warm-%s-%d" % (label, tag, time.time_ns())
    seqs, lpss, cached = [], [], []
    pt = None
    for i in range(reps):
        salt = fixed if mode == "warm" else "g5dbg-%s-%s-%d-%d" % (label, tag, i, time.time_ns())
        idle()
        if dump:
            for f in glob.glob(os.path.join(ddir, "r0_*.pt")):
                os.remove(f)
            open(os.path.join(ddir, "ON"), "w").close()
        t0 = time.time()
        try:
            toks, lps, pt, ct = one(case, salt)
        finally:
            if dump:
                os.remove(os.path.join(ddir, "ON"))
        dt = time.time() - t0
        if dump:
            rd = os.path.join(ddir, "%s-%s-%d" % (label, tag, i))
            os.makedirs(rd, exist_ok=True)
            for f in glob.glob(os.path.join(ddir, "r0_*.pt")):
                shutil.move(f, rd)
        seqs.append(toks)
        lpss.append(lps)
        cached.append(ct)
        fo.write(json.dumps({"label": label, "case": case, "mode": mode, "run": i, "pt": pt, "cached": ct,
                             "lat_s": round(dt, 3), "tok_sha": hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:12],
                             "lp_sha": hashlib.sha256(json.dumps(lps).encode()).hexdigest()[:12],
                             "toks": toks, "lps": lps}) + "\n")
        fo.flush()
    dt_ = []
    for q in seqs:
        if q not in dt_:
            dt_.append(q)
    dl = []
    for q in lpss:
        if q not in dl:
            dl.append(q)
    summ = {"label": label, "summary": 1, "case": case, "mode": mode, "reps": reps, "pt": pt,
            "distinct_tok": len(dt_), "distinct_lp": len(dl),
            "tokdiv": [first_div(q, seqs[0]) for q in seqs[1:]], "lpdiv": [first_div(q, lpss[0]) for q in lpss[1:]],
            "lp0_distinct": len(set(q[0] for q in lpss if q)), "cached": cached}
    fo.write(json.dumps(summ) + "\n")
    fo.flush()
    print("%-10s %-14s pt=%-5s %-4s distinct_tok=%d distinct_lp=%d lp0_distinct=%d tokdiv=%s lpdiv=%s cached=%s"
          % (label, case, pt, mode, len(dt_), len(dl), summ["lp0_distinct"], summ["tokdiv"], summ["lpdiv"], cached),
          flush=True)
