#!/usr/bin/env python3
"""goal_single.py -- G1 goal metric (stdlib only): single-user decode tok/s on the dsv41 server.

G1 = bench.py "single" with the SAME three prompts (code / prose / math, copied verbatim from bench.py so the
numbers stay comparable with every earlier README row), 1024 max tokens, temperature 0, top_p 1.0, thinking left
at the server default (ON), each type x2 in bench.py order (code, prose, math, code, prose, math), serial.
Per request: decode tok/s = (completion_tokens - 1) / (last content chunk - first content chunk), TTFT,
spec-decode deltas (AL) from /metrics and sha256 of reasoning + NUL + content (exactness check across arms).
Summary: per-type mean tok/s, G1 = mean of the three type means, per-type AL, digest of all output hashes.

The prompts are fixed: their canonical JSON must hash to PROMPTS_SHA256 or the script refuses to run.
Usage: python3 goal_single.py --out runs/x/g1.jsonl --label x [--url http://127.0.0.1:8000] [--repeats 2]
"""
import argparse, hashlib, json, re, sys, time, urllib.request

PROMPTS = {
    "code": ("Write a complete, well-documented Python module implementing a thread-safe LRU cache with TTL expiry, "
             "a decorator interface, statistics counters (hits, misses, evictions), and a small command-line demo. "
             "Include type hints, docstrings for every public function, and a set of unittest test cases covering "
             "eviction order, TTL expiry, concurrency, and the decorator. Then explain the design choices and the "
             "complexity of each operation in detail."),
    "prose": ("Write a long, thoughtful essay about how the invention of the printing press changed European society, "
              "covering religion, science, politics, literacy, and the economy. Use vivid historical examples, discuss "
              "both intended and unintended consequences, compare it to the rise of the internet in the late twentieth "
              "century, and finish with a reflective conclusion about what lessons it holds for new technologies today."),
    "math": ("Solve the following carefully, showing every step of your reasoning. A sequence is defined by a_1 = 3 and "
             "a_{n+1} = 2a_n + n^2 for n >= 1. Find a closed-form expression for a_n, prove it by induction, compute "
             "a_10 exactly, and then determine the smallest n for which a_n exceeds one million. After that, compute "
             "the sum of the first 12 terms and verify it using your closed form. Double-check all arithmetic."),
}
TYPES = ("code", "prose", "math")
PROMPTS_SHA256 = "dc744bc29273cc072ffd3f9e29786f6c854141c0d4083001bebfb8ebee1291e2"  # sha256 of json.dumps(PROMPTS, sort_keys=True)
MAX_TOKENS = 1024


def prompts_sha():
    return hashlib.sha256(json.dumps(PROMPTS, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def scrape(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        txt = r.read().decode()
    m = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for key, name in (("drafts", "vllm:spec_decode_num_drafts_total"),
                          ("draft_tokens", "vllm:spec_decode_num_draft_tokens_total"),
                          ("accepted", "vllm:spec_decode_num_accepted_tokens_total")):
            if line.startswith(name + "{") or line.startswith(name + " "):
                m[key] += float(line.split()[-1])
    return m


def spec_delta(a, b):
    d = b["drafts"] - a["drafts"]
    if d <= 0:
        return {"drafts": 0}
    acc = b["accepted"] - a["accepted"]
    dt = b["draft_tokens"] - a["draft_tokens"]
    return {"drafts": int(d), "draft_tokens": int(dt), "accepted": int(acc), "AL": round(1 + acc / d, 3)}


def stream_chat(base, model, prompt, max_tokens, timeout=1800):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "top_p": 1.0, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = t_last = None; nchunks = 0; usage = None; finish = None
    reasoning, content = [], []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                dl = ch.get("delta") or {}
                c = dl.get("content") or ""
                rc = dl.get("reasoning_content") or dl.get("reasoning") or ""
                if c or rc:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    nchunks += 1
                    content.append(c); reasoning.append(rc)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.perf_counter()
    ct = (usage or {}).get("completion_tokens")
    text = "".join(reasoning) + "\x00" + "".join(content)
    res = {"prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": ct, "finish": finish,
           "ttft_s": round(t_first - t0, 4) if t_first else None, "e2e_s": round(t_end - t0, 4),
           "chunks": nchunks, "sha256": hashlib.sha256(text.encode()).hexdigest()}
    if t_first and ct and nchunks > 1 and t_last > t_first:
        res["decode_s"] = round(t_last - t_first, 4)
        res["decode_tok_s"] = round((ct - 1) / (t_last - t_first), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--print-sha", action="store_true")
    a = ap.parse_args()
    sha = prompts_sha()
    if a.print_sha:
        print(sha); return
    if sha != PROMPTS_SHA256:
        sys.exit(f"goal_single: prompt set changed (sha256 {sha} != pinned {PROMPTS_SHA256}); refusing")
    base = a.url.rstrip("/")
    rows = []
    def emit(row):
        row.update({"label": a.label, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        s = json.dumps(row); print(s, flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(s + "\n")
    t_start = time.time()
    for rep in range(a.repeats):
        for ty in TYPES:
            m0 = scrape(base)
            r = stream_chat(base, a.model, PROMPTS[ty], MAX_TOKENS)
            r.update({"metric": "G1", "type": ty, "rep": rep, "max_tokens": MAX_TOKENS,
                      "t_wall_start": round(t_start, 3), "spec": spec_delta(m0, scrape(base))})
            rows.append(r); emit(r)
    per = {}
    for ty in TYPES:
        xs = [r["decode_tok_s"] for r in rows if r["type"] == ty and r.get("decode_tok_s")]
        als = [r["spec"].get("AL") for r in rows if r["type"] == ty and r["spec"].get("AL")]
        per[ty] = {"tok_s": round(sum(xs) / len(xs), 3) if xs else None, "runs": xs,
                   "AL": round(sum(als) / len(als), 3) if als else None,
                   "sha8": sorted({r["sha256"][:8] for r in rows if r["type"] == ty}),
                   "tokens": [r["completion_tokens"] for r in rows if r["type"] == ty]}
    means = [per[t]["tok_s"] for t in TYPES if per[t]["tok_s"]]
    summ = {"metric": "G1_summary", "label": a.label, "prompts_sha256": sha,
            "G1_mean_tok_s": round(sum(means) / len(means), 3) if len(means) == len(TYPES) else None,
            "per_type": per, "outputs_digest": hashlib.sha256("".join(r["sha256"] for r in rows).encode()).hexdigest()[:16],
            "all_reps_identical": all(len(per[t]["sha8"]) == 1 for t in TYPES),
            "elapsed_s": round(time.time() - t_start, 1), "t_start": round(t_start, 3), "t_end": round(time.time(), 3)}
    emit(summ)


if __name__ == "__main__":
    main()
