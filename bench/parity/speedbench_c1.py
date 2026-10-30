#!/usr/bin/env python3
"""speedbench_c1.py -- stdlib-only reproduction of knapcio/speedbench 0.3 `run --mode full` (one arm) for OUR endpoint.

Source reproduced (read, never executed): github.com/knapcio/speedbench e95a638b (0.3; knapcio's README cites
6733653): speedbench/run.py (build_body, prepare, derive, summarize), client.py (stream_chat), prompts.py (WARMUP).
Prompt set: speedbench-v1.json next to this file, copied byte-for-byte from the repo (Apache-2.0),
sha256 07dc37be5a53d2681d462412fc5f94c69540e7c20233eee84df8f6b7118b73e2 (checked at start).

Protocol (full mode, c1, T=0, 256 tokens; knapcio ran it "thinking off" on SGLang):
  body = {model, messages, stream:true, max_tokens, temperature:0,
          stream_options:{include_usage:true, continuous_usage_stats:true}, [tools],
          chat_template_kwargs:{reasoning_effort:"low", thinking:false}}
    speedbench's default --effort low sends chat_template_kwargs.reasoning_effort; our deepseek_v41 tokenizer
    turns thinking ON when neither `thinking` nor `enable_thinking` is given, so "thinking off" needs
    {"thinking": false} (speedbench --extra-body merges it into the same dict). No top_p, no seed, no ignore_eos.
  prepare: probe (WARMUP[0], 8 tokens; continuous usage off if rejected) -> tool probe (first prompt with tools,
    16 tokens) -> 3 warm-ups (64 tokens) -> one cold ~8K-token unique document, max_tokens 1 (speedbench sizes its
    seeded document with /tokenize; any unique ~8K document serves, it is discarded).
  jobs: the 73 prompts in file order; per job a priming request (same body, max_tokens 1) then the measured one;
    then the first 4 jobs again as sentinels (drift check, not in the categories).
  per request: t_first / t_last = first / last SSE event carrying content, reasoning or tool-call text;
    decode tokens = cumulative usage at the last text event - at the first (exact boundary, continuous usage)
    else completion_tokens - 1; tok/s = tokens / (t_last - t_first); a request counts only with a final usage and
    a finish_reason; valid decode needs >= 32 decode tokens (MIN_DECODE_TOKENS).
  headline: per category and ALL, geometric mean of per-request tok/s over valid prompts; median TTFT (warm:
    the priming request cached the prompt).
Extras: /metrics spec counters around each measured request (c1 -> exact): AL and server-free step time
ms/step = decode_s * 1000 / (steps * dt / (completion - 1)) as speedbench's decompose(); text sha256 per prompt.

  python3 speedbench_c1.py --out runs/X/sb_full.jsonl              # ~3 min at ~100 tok/s
  python3 speedbench_c1.py --quick --out runs/X/sb_quick.jsonl     # 30-prompt subset x 160 tokens
"""
import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "speedbench-v1.json")
PROMPTS_SHA = "07dc37be5a53d2681d462412fc5f94c69540e7c20233eee84df8f6b7118b73e2"
MIN_DECODE_TOKENS = 32
WARMUP = [
    [{"role": "user", "content": "Count from one to twenty in words, one per line."}],
    [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."}],
    [{"role": "user", "content": "Here is a list of fictional cities and their populations:\n" +
      "\n".join("City %d: %d inhabitants, founded in %d." % (i, 1000 + 37 * i * i, 1200 + 3 * i) for i in range(220)) +
      "\n\nWhich three cities are the largest?"}],
]
CATS = ["prose", "code", "structured", "agentic", "multilingual", "longctx"]
# knapcio README 2026-10-23: speedbench 0.3 full, c1, T=0, 256 tokens, thinking off; geomean tok/s per category.
KNAPCIO = {"official 1017": {"prose": 64.7, "code": 106.8, "structured": 109.5, "agentic": 98.3,
                             "multilingual": 70.0, "longctx": 91.0, "ALL": 87.5, "median_ttft_ms": 205},
           "abliterated": {"prose": 64.3, "code": 110.2, "structured": 105.1, "agentic": 100.6,
                           "multilingual": 68.4, "longctx": 92.2, "ALL": 87.6, "median_ttft_ms": 208}}


class Failed(Exception):
    pass


def iter_sse(resp):
    buf = []
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if buf:
                yield "\n".join(buf)
                buf = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            buf.append(line[5:].lstrip(" "))
    if buf:
        yield "\n".join(buf)


def stream_chat(base, body, timeout, headers):
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    h.update(headers)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h)
    S = {"t_first": None, "t_last": None, "usage": None, "finish": None, "chunks": [], "content": [],
         "reasoning": [], "tools": {}}
    t_send = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise Failed("HTTP %d: %s" % (e.code, e.read().decode(errors="replace")[:300])) from None
    except Exception as e:  # noqa: BLE001
        raise Failed("connection: %r" % (e,)) from None
    try:
        with resp:
            for data in iter_sse(resp):
                if data.strip() == "[DONE]":
                    break
                now = time.perf_counter()
                j = json.loads(data)
                if j.get("error"):
                    raise Failed("stream error: %s" % (j["error"],))
                if j.get("usage"):
                    S["usage"] = j["usage"]
                got, chan = 0, ""
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    c = d.get("content")
                    if c:
                        S["content"].append(c)
                        got += len(c)
                        chan += "c"
                    for k in ("reasoning_content", "reasoning"):
                        rc = d.get(k)
                        if rc:
                            S["reasoning"].append(rc)
                            got += len(rc)
                            chan += "r"
                            break
                    for tc in d.get("tool_calls") or []:
                        slot = S["tools"].setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                        got += len(fn.get("arguments") or "")
                        chan += "t"
                    if ch.get("finish_reason"):
                        S["finish"] = ch["finish_reason"]
                if got or "t" in chan:
                    cum = (j.get("usage") or {}).get("completion_tokens")
                    if S["t_first"] is None:
                        S["t_first"] = now
                    S["t_last"] = now
                    S["chunks"].append([round((now - t_send) * 1000, 2), got, -1 if cum is None else cum, chan])
    except Failed:
        raise
    except Exception as e:  # noqa: BLE001
        raise Failed("stream interrupted: %r" % (e,)) from None
    u = S["usage"]
    if not u or not u.get("completion_tokens"):
        raise Failed("stream ended without usage.completion_tokens")
    if S["finish"] is None:
        raise Failed("stream ended without a finish_reason")
    ch = S["chunks"]
    exact = bool(ch) and ch[0][2] >= 0 and ch[-1][2] >= 0
    tools = [S["tools"][k] for k in sorted(S["tools"])]
    text = json.dumps(["".join(S["reasoning"]), "".join(S["content"]), tools], ensure_ascii=False)
    r = {"ttft_s": None if S["t_first"] is None else S["t_first"] - t_send,
         "decode_s": None if S["t_first"] is None else S["t_last"] - S["t_first"],
         "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u["completion_tokens"],
         "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
         "boundary": "exact" if exact else "assumed", "finish_reason": S["finish"], "n_chunks": len(ch),
         "first_chunk_tokens": ch[0][2] if exact else None, "last_chunk_tokens": ch[-1][2] if exact else None,
         "reasoning_chars": sum(map(len, S["reasoning"])), "content_chars": sum(map(len, S["content"])),
         "tool_calls": len(tools), "sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
    return r


def derive(r):
    n = r["completion_tokens"]
    dt = (r["last_chunk_tokens"] - r["first_chunk_tokens"]) if r["boundary"] == "exact" else n - 1
    r["decode_tokens"] = dt
    r["tok_s"] = dt / r["decode_s"] if r["decode_s"] and r["decode_s"] > 0 and dt > 0 else None
    r["valid_decode"] = r["tok_s"] is not None and dt >= MIN_DECODE_TOKENS
    return r


SPEC = (("drafts", "vllm:spec_decode_num_drafts_total"), ("draft_tokens", "vllm:spec_decode_num_draft_tokens_total"),
        ("accepted", "vllm:spec_decode_num_accepted_tokens_total"), ("finished", "vllm:request_success_total"),
        ("running", "vllm:num_requests_running"))


def scrape(base, headers):
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/metrics", headers=headers), timeout=10) as r:
            txt = r.read().decode(errors="replace")
    except Exception:  # noqa: BLE001
        return None
    m = {k: 0.0 for k, _ in SPEC}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k, name in SPEC:
            if line.startswith(name + "{") or line.startswith(name + " "):
                try:
                    m[k] += float(line.split()[-1])
                except ValueError:
                    pass
    return m


class Bench:
    def __init__(self, a):
        self.a, self.base = a, a.url.rstrip("/")
        self.headers = {"Authorization": "Bearer " + a.api_key} if a.api_key else {}
        self.continuous = not a.no_continuous_usage

    def body(self, prompt, max_tokens):
        b = {"model": self.a.model, "messages": copy.deepcopy(prompt["messages"]), "stream": True,
             "max_tokens": max_tokens, "temperature": 0.0, "stream_options": {"include_usage": True}}
        if self.continuous:
            b["stream_options"]["continuous_usage_stats"] = True
        if prompt.get("tools"):
            b["tools"] = prompt["tools"]
        ctk = {}
        if self.a.effort:
            ctk["reasoning_effort"] = self.a.effort
        if not self.a.thinking:
            ctk["thinking"] = False
        if ctk:
            b["chat_template_kwargs"] = ctk
        return b

    def one(self, prompt, max_tokens, metrics=False):
        before = scrape(self.base, self.headers) if metrics else None
        try:
            r = stream_chat(self.base, self.body(prompt, max_tokens), self.a.timeout, self.headers)
        except Failed as e:
            return {"ok": False, "error": str(e)[:300]}
        r["ok"] = True
        after = scrape(self.base, self.headers) if metrics else None
        if before and after:
            st = after["drafts"] - before["drafts"]
            fin = after["finished"] - before["finished"]
            r["spec"] = {"steps": int(st), "accepted": int(after["accepted"] - before["accepted"]),
                         "draft_tokens": int(after["draft_tokens"] - before["draft_tokens"]), "finished": fin}
        return r


def long_doc(n_tokens, salt, cpt=3.6):
    words = ("minutes log config schedule review budget sensor deploy rollback incident queue replica shard "
             "latency window cohort ledger invoice tensor kernel cache index vector batch").split()
    out, state, i = ["Document %s\n" % salt], 12345, 0
    while sum(map(len, out)) < n_tokens * cpt:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append("%d. %s %s %d; " % (i, words[state % len(words)], words[(state >> 8) % len(words)], state % 997))
        i += 1
    return [{"role": "user", "content": "".join(out) + "\n\nSummarize the document in one sentence."}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--quick", action="store_true", help="speedbench --mode quick: 30-prompt subset x 160 tokens")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--effort", default="low", help="chat_template_kwargs.reasoning_effort ('' = none)")
    ap.add_argument("--thinking", action="store_true", help="leave thinking on (default: thinking off)")
    ap.add_argument("--no-continuous-usage", action="store_true")
    ap.add_argument("--no-prime", action="store_true")
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--warmup-prefill", type=int, default=8192)
    ap.add_argument("--sentinels", type=int, default=None)
    ap.add_argument("--skip-tools", action="store_true")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default="sb_full.jsonl")
    a = ap.parse_args()
    raw = open(PROMPTS, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != PROMPTS_SHA:
        raise SystemExit("prompt set sha256 mismatch: %s" % sha)
    body = json.loads(raw.decode("utf-8"))
    prompts = [p for p in body["prompts"] if (not a.quick or p.get("quick")) and not (a.skip_tools and p.get("tools"))]
    max_tokens = a.max_tokens or (160 if a.quick else 256)
    n_sent = a.sentinels if a.sentinels is not None else (2 if a.quick else 4)
    B = Bench(a)
    out = open(a.out, "a")

    def emit(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    emit({"kind": "meta", "tool": "speedbench_c1.py", "reproduces": "knapcio/speedbench 0.3 run --mode %s" %
          ("quick" if a.quick else "full"), "prompt_set": body.get("version"), "prompt_set_sha256": sha,
          "args": {k: v for k, v in vars(a).items() if k != "api_key"}, "max_tokens": max_tokens,
          "request_template": {k: v for k, v in B.body(prompts[0], max_tokens).items() if k != "messages"},
          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    # ---- prepare (all discarded)
    pr = B.one({"messages": WARMUP[0]}, 8)
    if not pr.get("ok") and B.continuous:
        B.continuous = False
        pr = B.one({"messages": WARMUP[0]}, 8)
    if not pr.get("ok"):
        raise SystemExit("probe failed: %s" % pr.get("error"))
    print("probe ok, continuous usage %s, boundary %s" % (B.continuous, pr.get("boundary")), flush=True)
    tp = next((p for p in prompts if p.get("tools")), None)
    if tp is not None:
        t = B.one(tp, 16)
        if not t.get("ok"):
            raise SystemExit("tool probe failed (%s); use --skip-tools" % t.get("error"))
    for i in range(3):
        w = B.one({"messages": WARMUP[i]}, 64)
        print("warm-up %d: %s" % (i + 1, "ok" if w.get("ok") else w.get("error")), flush=True)
    if a.warmup_prefill:
        w = B.one({"messages": long_doc(a.warmup_prefill, uuid.uuid4().hex)}, 1)
        print("warm-up long prefill: %s" % ("ok, %s prompt tokens, %.1f s" % (w.get("prompt_tokens"), w.get("ttft_s") or 0)
                                           if w.get("ok") else w.get("error")), flush=True)
    # ---- jobs
    jobs = [(p, "measure") for p in prompts] + [(p, "sentinel") for p in prompts[:n_sent]]
    recs = []
    t0 = time.time()
    for n, (p, kind) in enumerate(jobs):
        rec = {"kind": kind, "id": p["id"], "category": p["category"], "pos": n, "t_wall": round(time.time() - t0, 2)}
        if not a.no_prime:
            q = B.one(p, 1)
            rec["prime"] = {k: q.get(k) for k in ("ok", "ttft_s", "prompt_tokens", "cached_tokens", "error")}
        r = B.one(p, max_tokens, metrics=not a.no_metrics)
        if r.get("ok"):
            derive(r)
            sp = r.get("spec") or {}
            if sp.get("steps") and sp.get("finished") == 1 and r["completion_tokens"] > 1 and r["decode_tokens"] > 0:
                st, nn = sp["steps"], r["completion_tokens"] - 1
                r["AL"] = round(nn / st, 3)
                r["step_ms"] = round(r["decode_s"] * 1000 / (st * r["decode_tokens"] / nn), 2)
        rec.update(r)
        emit(rec)
        recs.append(rec)
        print("%3d/%d %-16s %-12s %7s tok/s  ttft %6.0f ms  ct %4s  AL %-6s %s" % (
            n + 1, len(jobs), p["id"], p["category"], "%.1f" % rec["tok_s"] if rec.get("tok_s") else "-",
            (rec.get("ttft_s") or 0) * 1000, rec.get("completion_tokens"), rec.get("AL", "-"),
            "" if kind == "measure" else "(sentinel)" if rec.get("ok") else rec.get("error")), flush=True)
    # ---- summary (speedbench summarize())
    meas = [r for r in recs if r["kind"] == "measure"]
    ok = [r for r in meas if r.get("ok")]
    summ = {"requests": len(meas), "failed": len(meas) - len(ok), "valid_decode": sum(bool(r.get("valid_decode")) for r in ok),
            "exact_boundary": sum(r.get("boundary") == "exact" for r in ok), "elapsed_s": round(time.time() - t0, 1)}
    print("\nspeedbench-v1 %s, c1, T=0, %d tokens, thinking %s: geomean tok/s (n)   [knapcio official / abliterated]" % (
        "quick" if a.quick else "full", max_tokens, "on" if a.thinking else "off"))
    for c in CATS + ["ALL"]:
        rs = [r for r in ok if (c == "ALL" or r["category"] == c) and r.get("valid_decode")]
        if not rs:
            continue
        toks = [r["tok_s"] for r in rs]
        s = {"n": len(rs), "geomean_tok_s": round(math.exp(statistics.fmean(math.log(x) for x in toks)), 2),
             "median_tok_s": round(statistics.median(toks), 2),
             "aggregate_tok_s": round(sum(r["decode_tokens"] for r in rs) / sum(r["decode_s"] for r in rs), 2),
             "median_ttft_ms": round(statistics.median(r["ttft_s"] for r in rs) * 1000, 1),
             "truncated_frac": round(sum(r.get("finish_reason") == "length" for r in rs) / len(rs), 3)}
        al = [r["AL"] for r in rs if r.get("AL")]
        sm = [r["step_ms"] for r in rs if r.get("step_ms")]
        if al:
            s["median_AL"] = round(statistics.median(al), 3)
            s["median_step_ms"] = round(statistics.median(sm), 2)
        summ[c] = s
        print("  %-12s %7.1f (%2d)  AL %-6s step %-6s ms  ttft %6.0f ms   [%s / %s]" % (
            c, s["geomean_tok_s"], s["n"], s.get("median_AL", "-"), s.get("median_step_ms", "-"), s["median_ttft_ms"],
            KNAPCIO["official 1017"].get(c), KNAPCIO["abliterated"].get(c)))
    sent = [r for r in recs if r["kind"] == "sentinel" and r.get("tok_s")]
    orig = {r["id"]: r for r in ok if r.get("tok_s")}
    pairs = [(orig[s["id"]], s) for s in sent if s["id"] in orig]
    if pairs:
        summ["sentinel_drift_pct"] = round((math.exp(statistics.fmean(math.log(s["tok_s"] / o["tok_s"]) for o, s in pairs)) - 1) * 100, 3)
        summ["sentinel_identical"] = sum(o["sha256"] == s["sha256"] for o, s in pairs)
        print("  sentinels: drift %+.2f %%, identical text %d/%d" % (summ["sentinel_drift_pct"], summ["sentinel_identical"], len(pairs)))
    print("  requests %d, failed %d, valid %d, exact boundary %d, %.0f s" % (
        summ["requests"], summ["failed"], summ["valid_decode"], summ["exact_boundary"], summ["elapsed_s"]))
    emit({"kind": "summary", **summ, "knapcio": KNAPCIO, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    out.close()


if __name__ == "__main__":
    main()
