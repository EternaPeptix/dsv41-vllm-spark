#!/usr/bin/env python3
"""fraser_bench.py -- vLLM-Moet's "Fraser bench" (single-stream code decode on one fixed prompt) for our dsv41 server.

Method (kacper-daftcode/vLLM-Moet tools/sm120_perf/fraser_bench.py @ b0dc737, Apache-2.0; prompt fraser_prompt.txt
verbatim, sha256 5a6b7bc1...): one fixed 2,678-token code-generation prompt, temperature 0, max_tokens 2048, thinking
OFF, ONE stream, a cache-bust nonce line first, a fresh prefill per rep. tok/s = completion tokens over the
first-to-last-token interval (Moet's headline "median tok/s out" is completion_tokens / (last - first); we also give
(n-1)/(last-first) like our goal scripts), DSpark AL = 1 + accepted/drafts and steps/s = drafts/(last-first) from the
server's spec-decode counters (/metrics deltas around each request). Report the MEDIAN over reps, not the best.

Our differences (all switchable):
  * --nonce-mode fixed (default): rep i uses nonce sha256("fraser-<i>")[:32], so every arm sees the same prompts and the
    output digest is a bitwise cross-arm check; a random cache_salt per request still forces a fresh prefill
    (no prefix-cache hit, same tokens). --nonce-mode random = Moet's behaviour (uuid4 nonce, no cache_salt).
  * --warmup 1 (default): one excluded rep first (nonce "warmup").
  * refuses to start a rep while the server has running/waiting requests (foreign traffic), unless --allow-busy.
  * the prompt is pinned by sha256 (refuses another file unless --any-prompt).

usage (spark-one; stdlib only):
  python3 -I ~/dsv41-bench/fraser/fraser_bench.py --label X --out fraser.jsonl [--reps 10] [--base http://127.0.0.1:8000]
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.request
import uuid

PROMPT_SHA256 = "5a6b7bc1cbbc73a75d661fb7055e690c0afaeda06f3d3ea75cdae89c110802fd"
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_RE = re.compile(r'^vllm:(spec_decode_num_drafts_total|spec_decode_num_draft_tokens_total|'
                     r'spec_decode_num_accepted_tokens_total|spec_decode_num_accepted_tokens_per_pos_total|'
                     r'num_requests_running|num_requests_waiting)(\{[^}]*\})? ([0-9.eE+-]+)$')


def scrape(base):
    out = {}
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            m = SPEC_RE.match(line)
            if not m:
                continue
            name, labels, v = m.groups()
            key = name.replace("spec_decode_num_", "").replace("_total", "")
            pos = re.search(r'position="(\d+)"', labels or "")
            if pos:
                key += "_p" + pos.group(1)
            out[key] = out.get(key, 0.0) + float(v)
    return out


def wait_idle(base, allow_busy, timeout_s=60):
    t0 = time.time()
    while True:
        m = scrape(base)
        busy = m.get("num_requests_running", 0) + m.get("num_requests_waiting", 0)
        if busy == 0 or allow_busy:
            return busy
        if time.time() - t0 > timeout_s:
            sys.exit(f"fraser_bench: server busy for {timeout_s}s (running+waiting={busy}); foreign traffic? "
                     "(--allow-busy to ignore)")
        time.sleep(2)


def one(a, text, nonce, rep):
    body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "messages": [{"role": "user", "content": f"[LocalMaxxing cache-bust nonce: {nonce}]\n{text}"}]}
    if a.thinking == "off":
        body["chat_template_kwargs"] = {"thinking": False}
    if a.nonce_mode == "fixed" and not a.no_cache_salt:
        body["cache_salt"] = uuid.uuid4().hex
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    busy = wait_idle(a.base, a.allow_busy)
    before = scrape(a.base)
    t0 = time.perf_counter()
    first = last = None
    count = 0
    usage = {}
    pieces = []
    finish = None
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                pieces.append((d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or ""))
                finish = ch.get("finish_reason") or finish
            n = usage.get("completion_tokens", 0) if usage else 0
            if n > count:
                now = time.perf_counter()
                first = first or now
                last = now
                count = n
    after = scrape(a.base)
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after
         if k.startswith(("drafts", "draft_tokens", "accepted"))}
    drafts, dtok, acc = d.get("drafts", 0.0), d.get("draft_tokens", 0.0), d.get("accepted_tokens", 0.0)
    per_pos = [int(d[k]) for k in sorted((k for k in d if k.startswith("accepted_tokens_p")),
                                         key=lambda s: int(s.rsplit("_p", 1)[1]))]
    span = (last - first) if (first is not None and last is not None and last > first) else None
    out_text = "".join(pieces)
    return {"metric": "FRASER", "label": a.label, "rep": rep, "nonce": nonce,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": count, "finish": finish,
            "ttft_ms": round((first - t0) * 1000, 1) if first is not None else None,
            "tok_s_out_lm": round(count / span, 2) if span else None,  # Moet's headline (n / (last - first))
            "tok_s_out": round((count - 1) / span, 2) if span and count > 1 else None,  # our goal-script convention
            "tok_s_total": round((usage.get("prompt_tokens", 0) + count) / (last - t0), 1) if last else None,
            "acceptance": round(acc / dtok, 4) if dtok else None,
            "AL": round(1 + acc / drafts, 3) if drafts else None,
            "steps_s": round(drafts / span, 2) if drafts and span else None,
            "drafts": int(drafts), "draft_tokens": int(dtok), "accepted": int(acc), "accepted_per_pos": per_pos,
            "busy_before": busy, "sha256": hashlib.sha256(out_text.encode()).hexdigest(),
            **({"text": out_text} if a.save_text else {}),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--prompt-file", default=os.path.join(HERE, "fraser_prompt.txt"))
    ap.add_argument("--any-prompt", action="store_true", help="skip the prompt sha256 pin")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--thinking", choices=("off", "on"), default="off")
    ap.add_argument("--nonce-mode", choices=("fixed", "random"), default="fixed")
    ap.add_argument("--no-cache-salt", action="store_true")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--label", default="fraser")
    ap.add_argument("--out", default=None, help="append JSON lines (per rep + summary)")
    ap.add_argument("--save-text", action="store_true")
    a = ap.parse_args()
    raw = open(a.prompt_file, "rb").read()
    if hashlib.sha256(raw).hexdigest() != PROMPT_SHA256 and not a.any_prompt:
        sys.exit(f"fraser_bench: {a.prompt_file} sha256 != {PROMPT_SHA256[:16]} (pinned prompt; --any-prompt overrides)")
    text = raw.decode()

    def nonce_for(i):
        if a.nonce_mode == "random":
            return uuid.uuid4().hex
        return hashlib.sha256(f"fraser-{i}".encode()).hexdigest()[:32]

    sink = open(a.out, "a") if a.out else None

    def emit(rec):
        line = json.dumps(rec)
        print(line, flush=True)
        if sink:
            sink.write(line + "\n")
            sink.flush()

    t_start = time.time()
    for w in range(a.warmup):
        r = one(a, text, "warmup" if a.nonce_mode == "fixed" else uuid.uuid4().hex, -1 - w)
        r["metric"] = "FRASER_warmup"
        emit(r)
    runs = []
    for i in range(a.reps):
        r = one(a, text, nonce_for(i), i)
        runs.append(r)
        emit(r)
    digest = hashlib.sha256("".join(r["sha256"] for r in runs).encode()).hexdigest()[:16]
    lm = [r["tok_s_out_lm"] for r in runs if r["tok_s_out_lm"] is not None]
    pp = [sum(r["accepted_per_pos"][j] for r in runs if j < len(r["accepted_per_pos"]))
          for j in range(max((len(r["accepted_per_pos"]) for r in runs), default=0))]
    drafts = sum(r["drafts"] for r in runs)
    emit({"metric": "FRASER_summary", "label": a.label, "prompt_sha256": PROMPT_SHA256, "reps": a.reps,
          "nonce_mode": a.nonce_mode, "thinking": a.thinking, "max_tokens": a.max_tokens,
          "median_tok_s_out_lm": med(lm), "best_tok_s_out_lm": max(lm) if lm else None,
          "mean_tok_s_out_lm": round(statistics.mean(lm), 3) if lm else None,
          "median_tok_s_out": med(r["tok_s_out"] for r in runs), "median_steps_s": med(r["steps_s"] for r in runs),
          "median_AL": med(r["AL"] for r in runs), "median_acceptance": med(r["acceptance"] for r in runs),
          "pooled_AL": round(1 + sum(r["accepted"] for r in runs) / drafts, 3) if drafts else None,
          "pooled_pos_accept_rate": [round(x / drafts, 4) for x in pp] if drafts else None,
          "median_ttft_ms": med(r["ttft_ms"] for r in runs),
          "completion_tokens": [r["completion_tokens"] for r in runs],
          "outputs_digest": digest, "elapsed_s": round(time.time() - t_start, 1),
          "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    main()
