#!/usr/bin/env python3
"""detprobe.py: long-prompt greedy determinism probe (goal window 5, Determinism phase). ON spark-one, stdlib only.

usage: detprobe.py <out.jsonl> <lens csv> <srcs csv: prose,code> <reps> <modes csv: cold,warm> [logprobs 0|1] [tag]

Per (source, length): the prompt is [BOS] + the first (L-1) token ids of the source text (exact length, token-id
prompt on /v1/completions, no chat template). Greedy, max_tokens 128, ignore_eos. One request at a time on an idle
server (waits for running+waiting == 0 before every request).
  cold = a fresh cache_salt per request (full prefill every time)
  warm = one cache_salt per (source, length): run 0 prefills cold, runs 1.. hit the prefix cache
With logprobs=1 each request also returns the sampled token's logprob per step (a bitwise fingerprint of each step):
first_lp_div = first step whose logprob bits differ from run 0 (0 = the prefill's own logits differ).
Writes one JSON line per request and one summary line per (source, length, mode).
"""
import hashlib
import json
import struct
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
out_path, lens_s, srcs_s, reps_s, modes_s = sys.argv[1:6]
use_lp = len(sys.argv) > 6 and sys.argv[6] == "1"
tag = sys.argv[7] if len(sys.argv) > 7 else "g5det"
lens = [int(x) for x in lens_s.split(",")]
srcs = srcs_s.split(",")
reps = int(reps_s)
modes = modes_s.split(",")

SRC_FILES = {
    "prose": ["/usr/share/common-licenses/GPL-3", "/usr/share/common-licenses/Apache-2.0",
              "/usr/share/common-licenses/GFDL-1.3"],
    "code": ["/home/jeweled/venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py.bak-nondet"],
}


def post(path, body, timeout=3600):
    r = urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=timeout)
    return json.loads(r.read().decode())


def idle():
    for _ in range(600):
        t = urllib.request.urlopen(BASE + "/metrics", timeout=5).read().decode()
        s = sum(float(l.split()[-1]) for l in t.splitlines()
                if l.startswith("vllm:num_requests_running{") or l.startswith("vllm:num_requests_waiting{"))
        if s == 0:
            return
        time.sleep(1)
    raise SystemExit("server not idle for 600 s")


def src_ids(name):
    text = "".join(open(p, encoding="utf-8", errors="replace").read() for p in SRC_FILES[name])
    return post("/tokenize", {"model": "dsv41", "prompt": text, "add_special_tokens": False})["tokens"]


def bits(x):
    return None if x is None else struct.pack("<d", float(x)).hex()


def first_div(a, b):
    for j in range(min(len(a), len(b))):
        if a[j] != b[j]:
            return j
    return None if len(a) == len(b) else min(len(a), len(b))


fo = open(out_path, "a")
ids_cache = {s: src_ids(s) for s in srcs}
for s in srcs:
    print("source %s: %d ids available" % (s, len(ids_cache[s])), flush=True)
for s in srcs:
    for L in lens:
        if L - 1 > len(ids_cache[s]):
            print("skip %s L=%d (source too short)" % (s, L), flush=True)
            continue
        prompt = [0] + ids_cache[s][: L - 1]
        for mode in modes:
            fixed = "%s-warm-%s-%d-%d" % (tag, s, L, time.time_ns())
            seqs, lps, cached, ttfts = [], [], [], []
            for i in range(reps):
                salt = fixed if mode == "warm" else "%s-cold-%s-%d-%d-%d" % (tag, s, L, i, time.time_ns())
                body = {"model": "dsv41", "prompt": prompt, "max_tokens": 128, "temperature": 0,
                        "ignore_eos": True, "return_token_ids": True, "cache_salt": salt}
                if use_lp:
                    body["logprobs"] = 1
                idle()
                t0 = time.time()
                d = post("/v1/completions", body)
                dt = time.time() - t0
                ch = d["choices"][0]
                toks = ch.get("token_ids") or []
                lp = [bits(x) for x in ((ch.get("logprobs") or {}).get("token_logprobs") or [])]
                u = d.get("usage") or {}
                ct = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                seqs.append(toks)
                lps.append(lp)
                cached.append(ct)
                ttfts.append(round(dt, 3))
                rec = {"tag": tag, "src": s, "L": L, "pt": u.get("prompt_tokens"), "mode": mode, "run": i,
                       "salt": salt, "cached": ct, "lat_s": round(dt, 3), "ntok": len(toks),
                       "tok_sha": hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:12],
                       "lp_sha": hashlib.sha256(json.dumps(lp).encode()).hexdigest()[:12] if use_lp else None,
                       "toks": toks, "lp0": lp[0] if lp else None}
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
            distinct = []
            for q in seqs:
                if q not in distinct:
                    distinct.append(q)
            dl = []
            for q in lps:
                if q not in dl:
                    dl.append(q)
            summ = {"tag": tag, "summary": 1, "src": s, "L": L, "mode": mode, "reps": reps,
                    "distinct_tok": len(distinct), "first_tok_div": [first_div(q, seqs[0]) for q in seqs[1:]],
                    "distinct_lp": len(dl) if use_lp else None,
                    "first_lp_div": [first_div(q, lps[0]) for q in lps[1:]] if use_lp else None,
                    "distinct_lp0": len(set(q[0] for q in lps if q)) if use_lp else None,
                    "cached": cached, "lat_s": ttfts}
            fo.write(json.dumps(summ) + "\n")
            fo.flush()
            print("%-5s L=%-6d %-4s distinct_tok=%d tokdiv=%s distinct_lp=%s lpdiv=%s lp0_distinct=%s cached=%s lat=%s"
                  % (s, L, mode, summ["distinct_tok"], summ["first_tok_div"], summ["distinct_lp"],
                     summ["first_lp_div"], summ["distinct_lp0"], cached, ttfts), flush=True)
