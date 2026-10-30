#!/usr/bin/env python3
"""sparkdash_decode.py -- stdlib-only reproduction of sparkDash's DecodeBench for OUR OpenAI endpoint.

Source reproduced (read, never executed): MiaAI-Lab/sparkDash 1.8.8 = commit 754f40a7 ("Release 1.8.8"),
server/collectors/DecodeBench.js, server/collectors/LlmStreaming.js, src/shared/llmPrompts.js. knapcio's
README (2026-10-23) used "sparkDash 1.8.8: 256 new tokens, temperature 0, thinking off". sparkDash >= 2.70
defaults to temperature 0.7 + thinking on, so do NOT compare with a current sparkDash run unless it was set to 0/off.

Request (identical fields to sparkDash 1.8.8; JSON key order is irrelevant to the server):
  POST /v1/chat/completions
  {model, messages:[{role:"user", content:P}], max_tokens:256, temperature:0, top_p:1, stream:true,
   stream_options:{include_usage:true},
   chat_template_kwargs:{enable_thinking:false, thinking:false, thinking_mode:"disabled"},
   min_tokens:256, ignore_eos:true, stop:[]}      <- the three fill-force fields only on measured waves
  HTTP 400 with thinking fields -> one retry with sparkDash's thinking-off fallback body;
  HTTP 400 with fill-force fields -> one retry without min_tokens/ignore_eos/stop (as sparkDash).
  Our vLLM deepseek_v41 tokenizer reads chat_template_kwargs.thinking / enable_thinking (thinking_mode is ignored),
  so these flags select chat mode. NOTE: min_tokens>0 keeps EOS in all_stop_token_ids even with ignore_eos, so every
  measured sparkDash request sets needs_logits_processing and DISARMS our CERT lm-head (dsv41_r4_small.step_armed);
  --no-fill runs the same prompts without the fill-force fields to measure that cost (text should be identical:
  all four prompts naturally run past 256 tokens).
Job = one 32-token warm-up request (type prompt; code -> the separate warmup_noop task), then the selected
concurrency waves in ascending order, one after another.
Wave c = c simultaneous streams. Prompts: code -> DECODE_CODE_TASKS[i] (c1 = binary_search); prose / structured /
json -> the base prompt at c1, and at c>1 "<base> (stream i/c)" (1.8.8; --prompt-version head gives
"Request ID: decode-i\\n<base>", sparkDash >= 2.4x).
Per stream: completion_tokens from the final usage (fallback chars/4), decode_tokens = ct - 1,
decode_tps = decode_tokens / (t_last - t_first), t_first / t_last = arrival of the first / last SSE event that
carried content or reasoning text. Wave aggregate = sum(decode_tokens) / (max t_last - min t_first) (sparkDash
"agg"); per-stream = mean decode_tps (sparkDash "/str", knapcio's bracketed value).

knapcio protocol (--protocol knapcio, default): two discarded prose c1 jobs; then per type
  job A = c1,c2,c4,c8,c16; job B = c1,c4; job C = c1   ->  c1 = median of 3, c4 = median of 2, c2/c8/c16 one run.
(knapcio: "after two discarded warm-up prose c1 runs; c1 is the median of three runs, c4 the median of two,
c2 / c8 / c16 one run each"; the split into jobs is our choice, the cells are the same.)
Extras (not in sparkDash): /metrics spec-decode counters around every wave -> AL (1 + accepted/drafts) and implied
ms/step = AL / per-stream tok/s (exact at c1, approximate at c>1); text sha256 per stream (determinism check);
vllm:num_requests_running checked before every wave (foreign traffic warning).

Run ON spark-one (API bound to 127.0.0.1:8000), cluster otherwise idle:
  python3 sparkdash_decode.py --out runs/X/sd_decode.jsonl              # knapcio protocol, ~3 min
  python3 sparkdash_decode.py --protocol single --types prose --conc 1  # one quick job
  python3 sparkdash_decode.py --no-fill --protocol single --conc 1 --out runs/X/sd_nofill.jsonl
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------------------------------------- prompts
# Extracted programmatically from llmPrompts.js at 754f40a7 (identical at HEAD c6e825cd except the c>1 tag);
# PROMPT_SET_SHA256 below is checked at start-up.
DECODE_STRUCTURED_PROMPT = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
DECODE_PROSE_PROMPT = ("Write a detailed step-by-step explanation of how a hash map works, "
                       "including collision handling, resizing, and time complexity. Be thorough.")
CODE_TASK_TAIL = ("Output only Python source. No comments, no docstrings, no markdown fences. "
                  "Then add tests and the helpers this needs. Keep writing code.")
DECODE_JSON_PROMPT = ("Emit only a JSON array of fake GPU metrics rows. Each object needs host, gpuIndex, utilPct, "
                      "tempC, powerW, memUsedMb. Invent many rows. No markdown. Keep expanding the array.")
_CODE = [
    ("binary_search", "def binary_search(nums, target) -> int: index of target in a sorted list, or -1."),
    ("merge_sort", "def merge_sort(nums) -> list: stable sort of a list of ints, returning a new list."),
    ("lru_cache", "class LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used."),
    ("token_bucket", "class TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False."),
    ("ring_buffer", "class RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow."),
    ("dijkstra", "def dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph."),
    ("edit_distance", "def edit_distance(a, b) -> int: Levenshtein distance between two strings."),
    ("semver_cmp", "def semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b."),
    ("url_parse", "def url_parse(url) -> dict: scheme, host, port, path, and query pairs. No extra libraries."),
    ("json_pointer", "def json_pointer(doc, pointer) -> object: follow an RFC 6901 pointer, or None if missing."),
    ("glob_match", "def glob_match(pattern, text) -> bool: * and ? wildcards, no character classes."),
    ("csv_parse", "def csv_parse(text) -> list: rows of fields, honoring double-quoted commas and escaped quotes."),
    ("rle", "def rle_encode(s) -> str and rle_decode(s) -> str: run-length encoding of single-byte runs."),
    ("top_k", "def top_k(nums, k) -> list: the k largest ints, unordered, using a bounded heap."),
    ("interval_merge", "def merge_intervals(spans) -> list: merge overlapping [start, end] pairs."),
    ("topo_sort", "def topo_sort(nodes, edges) -> list: a valid order, or None if the graph has a cycle."),
    ("bloom_filter", "class BloomFilter: add(item) and might_contain(item) with two hash functions over a bit array."),
    ("moving_average", "class MovingAverage: next(x) returns the mean of the last window values."),
    ("retry_backoff", "def backoff_delays(attempts, base_ms, cap_ms) -> list: exponential delays clipped at the cap."),
    ("base64_encode", "def b64_encode(data) -> str and b64_decode(text) -> bytes: standard base64, no libraries."),
    ("expr_eval", "def eval_expr(text) -> int: evaluate non-negative ints with + - * / and parentheses."),
    ("histogram_percentile", "def percentile(samples, p) -> float: nearest-rank percentile of a list of numbers."),
    ("redact_secrets", "def redact(text) -> str: replace AWS-looking keys and password= values with ***. Keep the rest."),
    ("chunk_text", "def chunk_text(text, size) -> list: split into chunks of at most size chars, breaking on spaces when possible."),
    ("route_match", "def route_match(pattern, path) -> dict or None: /users/:id style params."),
    ("crc32", "def crc32(data) -> int: IEEE CRC-32 of a bytes object."),
    ("fixed_window", "class FixedWindow: allow() is True up to limit events per window_s, else False."),
    ("diff_lines", "def diff_lines(a, b) -> list: line diff as equal/delete/insert ops using a simple LCS."),
    ("infix_postfix", "def infix_to_postfix(tokens) -> list: shunting-yard for + - * / and parentheses."),
    ("consistent_hash", "class ConsistentHash: add_node, remove_node, and get_node(key) on a ring of virtual nodes."),
    ("utf8_decode", "def utf8_decode(data) -> str: decode UTF-8 bytes, replacing invalid sequences with U+FFFD."),
    ("dependency_closure", "def closure(root, deps) -> list: packages reachable from root, each name once, in visit order."),
]
DECODE_CODE_TASKS = ["%s\n%s\n%s" % (n, s, CODE_TASK_TAIL) for n, s in _CODE]
DECODE_CODE_WARMUP_PROMPT = "warmup_noop\ndef warmup_noop(x): return x unchanged.\n" + CODE_TASK_TAIL
PROMPT_SET_SHA256 = "56f109ee933f952b1ce6aa0d7e6f8bc8d2e0191f33d1ee39619d3b59de41f922"
BASE = {"structured": DECODE_STRUCTURED_PROMPT, "prose": DECODE_PROSE_PROMPT,
        "code": DECODE_CODE_TASKS[0], "json": DECODE_JSON_PROMPT}
TYPES = ["prose", "code", "structured", "json"]          # knapcio's table order

# knapcio README 2026-10-23 (sparkDash 1.8.8, 4x GB10 TP4, SGLang), aggregate (per-stream); c1 = median of 3.
KNAPCIO = {
    "official": {"prose": {1: 91.3, 2: (115.0, 58.1), 4: (174.0, 44.8), 8: (242.8, 32.0), 16: (359.7, 24.2)},
                 "code": {1: 142.9, 2: (186.1, 94.1), 4: (255.2, 66.0), 8: (341.4, 45.5), 16: (471.6, 31.9)},
                 "structured": {1: 160.8, 2: (222.2, 114.3), 4: (215.7, 67.1), 8: (248.7, 41.7), 16: (393.5, 35.2)},
                 "json": {1: 130.6, 2: (184.9, 96.3), 4: (322.8, 82.5), 8: (461.4, 59.6), 16: (726.9, 47.2)}},
    "abliterated": {"prose": {1: 89.0, 16: (355.7, None)}, "code": {1: 145.2, 16: (479.1, None)},
                    "structured": {1: 160.1}, "json": {1: 131.4}},
}


def prompt_set_sha():
    d = {"DECODE_STRUCTURED_PROMPT": DECODE_STRUCTURED_PROMPT, "DECODE_PROSE_PROMPT": DECODE_PROSE_PROMPT,
         "CODE_TASK_TAIL": CODE_TASK_TAIL, "DECODE_JSON_PROMPT": DECODE_JSON_PROMPT,
         "DECODE_CODE_TASKS": DECODE_CODE_TASKS, "DECODE_CODE_WARMUP_PROMPT": DECODE_CODE_WARMUP_PROMPT}
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pick_prompts(n, kind, version="1.8.8"):
    """pickDecodeBenchPrompts(count, type)"""
    if kind == "code":
        out = []
        for i in range(n):
            task = DECODE_CODE_TASKS[i % len(DECODE_CODE_TASKS)]
            cycle = i // len(DECODE_CODE_TASKS)
            out.append(task if cycle == 0 else "[stream %d]\n%s" % (i + 1, task))
        return out
    base = BASE[kind]
    if n <= 1:
        return [base]
    if version == "head":
        return ["Request ID: decode-%d\n%s" % (i + 1, base) for i in range(n)]
    return ["%s (stream %d/%d)" % (base, i + 1, n) for i in range(n)]


def warmup_prompt(kind):
    return DECODE_CODE_WARMUP_PROMPT if kind == "code" else BASE[kind]


# ----------------------------------------------------------------------------------------------- client
THINK_OFF = {"enable_thinking": False, "thinking": False, "thinking_mode": "disabled"}


def request_body(model, prompt, max_tokens, fill=False, think_flags="sparkdash"):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1, "stream": True, "stream_options": {"include_usage": True}}
    if think_flags == "sparkdash":
        body["chat_template_kwargs"] = dict(THINK_OFF)
    elif think_flags == "minimal":
        body["chat_template_kwargs"] = {"thinking": False}
    if fill:
        body.update({"min_tokens": max_tokens, "ignore_eos": True, "stop": []})
    return body


def thinking_off_fallback(body):
    nb = {k: v for k, v in body.items() if k not in ("enable_thinking", "thinking", "thinking_mode",
                                                       "reasoning_effort", "reasoning_budget")}
    nb["chat_template_kwargs"] = dict(THINK_OFF)
    nb["thinking"] = {"type": "disabled"}
    nb["enable_thinking"] = False
    return nb


def _iter_sse(resp):
    buf = []
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if buf:
                yield "\n".join(buf)
                buf = []
            continue
        if line.startswith("data:"):
            buf.append(line[5:].strip())
    if buf:
        yield "\n".join(buf)


def stream_once(base, body, timeout, api_key=None):
    """One streaming request, timed like sparkDash runStreamingRequestOnce (perf_counter, one clock for all threads)."""
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        h["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h)
    t0 = time.perf_counter()
    r = {"t0": t0, "t_first": None, "t_last": None, "usage": None, "finish_reason": None, "error": None,
         "http_status": None, "n_text_events": 0, "chars": 0}
    text = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            r["http_status"] = resp.status
            for data in _iter_sse(resp):
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except ValueError:
                    continue
                if isinstance(j.get("usage"), dict):
                    r["usage"] = j["usage"]
                for ch in (j.get("choices") or [])[:1]:
                    d = ch.get("delta") or {}
                    piece = d.get("content") or ""
                    rs = d.get("reasoning") or d.get("reasoning_content") or ""
                    if piece or rs:
                        now = time.perf_counter()
                        if r["t_first"] is None:
                            r["t_first"] = now
                        r["t_last"] = now
                        r["n_text_events"] += 1
                        r["chars"] += len(piece) + len(rs)
                        text.append(rs + piece)
                    if ch.get("finish_reason"):
                        r["finish_reason"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        r["http_status"] = e.code
        r["error"] = "HTTP %d: %s" % (e.code, e.read().decode(errors="replace")[:200])
    except Exception as e:  # noqa: BLE001 - one failed stream must not kill the wave
        r["error"] = "stream: %r" % (e,)
    r["t_end"] = time.perf_counter()
    u = r["usage"] or {}
    ct = u.get("completion_tokens") or 0
    if ct <= 0:                                   # sparkDash fallback: ~4 chars per token
        ct = max(1, round(r["chars"] / 4)) if r["chars"] else 0
    r["completion_tokens"] = ct
    r["prompt_tokens"] = u.get("prompt_tokens")
    r["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    r["decode_tokens"] = max(0, ct - (1 if ct > 0 else 0))
    dec = (r["t_last"] - r["t_first"]) if (r["t_first"] is not None and r["t_last"] > r["t_first"]) else 0.0
    r["decode_ms"] = dec * 1000
    r["decode_tps"] = r["decode_tokens"] / dec if dec > 0 and r["decode_tokens"] > 0 else 0.0
    r["ttft_ms"] = (r["t_first"] - t0) * 1000 if r["t_first"] is not None else 0.0
    pt = r["prompt_tokens"] or 0
    r["prefill_tps"] = pt / (r["ttft_ms"] / 1000) if r["ttft_ms"] > 0 and pt > 0 else 0.0
    r["text_sha"] = hashlib.sha256("".join(text).encode()).hexdigest()[:16]
    if r["error"] is None and r["t_first"] is None:
        r["error"] = "no text received"
    return r


def stream_request(base, body, timeout, api_key=None):
    """runStreamingRequest with retryOnThinking400, then the wave-level fill-force retry."""
    r = stream_once(base, body, timeout, api_key)
    if r["error"] and r["error"].startswith("HTTP 400"):
        r2 = stream_once(base, thinking_off_fallback(body), timeout, api_key)
        r2["retried"] = "thinking-fallback"
        r = r2
        if r["error"] and r["error"].startswith("HTTP 400") and ("min_tokens" in body or "ignore_eos" in body):
            nb = {k: v for k, v in body.items() if k not in ("min_tokens", "ignore_eos", "stop")}
            r = stream_once(base, nb, timeout, api_key)
            r["retried"] = "fill-fields-stripped"
    return r


# ----------------------------------------------------------------------------------------------- metrics
SPEC = (("drafts", "vllm:spec_decode_num_drafts_total"), ("draft_tokens", "vllm:spec_decode_num_draft_tokens_total"),
        ("accepted", "vllm:spec_decode_num_accepted_tokens_total"), ("running", "vllm:num_requests_running"),
        ("waiting", "vllm:num_requests_waiting"))


def scrape(base, api_key=None):
    try:
        h = {"Authorization": "Bearer " + api_key} if api_key else {}
        with urllib.request.urlopen(urllib.request.Request(base + "/metrics", headers=h), timeout=10) as r:
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


def spec_delta(a, b):
    if not a or not b:
        return None
    d = b["drafts"] - a["drafts"]
    if d <= 0:
        return {"steps": 0}
    acc, dt = b["accepted"] - a["accepted"], b["draft_tokens"] - a["draft_tokens"]
    return {"steps": int(d), "AL": round(1 + acc / d, 4), "drafts_per_step": round(dt / d, 3),
            "accept_rate": round(acc / dt, 4) if dt else None}


def wait_idle(base, api_key, log, max_wait=120):
    t = time.time()
    while True:
        m = scrape(base, api_key)
        if m is None or (m["running"] + m["waiting"]) == 0:
            return m
        if time.time() - t > max_wait:
            log("WARNING: %d running / %d waiting requests on the server before the wave (foreign traffic?)"
                % (m["running"], m["waiting"]))
            return m
        time.sleep(2)


# ----------------------------------------------------------------------------------------------- waves
def run_wave(a, kind, c, job, run_idx, log, out):
    prompts = pick_prompts(c, kind, a.prompt_version)
    m0 = None if a.no_metrics else wait_idle(a.base, a.api_key, log)
    res = [None] * c
    barrier = threading.Barrier(c)

    def worker(i):
        body = request_body(a.model, prompts[i], a.max_tokens, fill=not a.no_fill, think_flags=a.think_flags)
        try:
            barrier.wait(timeout=30)
        except threading.BrokenBarrierError:
            pass
        res[i] = stream_request(a.base, body, a.timeout, a.api_key)

    th = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(c)]
    wall0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join(a.timeout + 60)
    wall = time.perf_counter() - wall0
    m1 = None if a.no_metrics else scrape(a.base, a.api_key)
    ok = [r for r in res if r and not r["error"] and r["decode_tokens"] > 0]
    tps = [r["decode_tps"] for r in ok]
    agg = 0.0
    if ok:
        win = max(r["t_last"] for r in ok) - min(r["t_first"] for r in ok)
        agg = sum(r["decode_tokens"] for r in ok) / win if win > 0 else 0.0
    sd = spec_delta(m0, m1)
    rec = {"kind": "wave", "type": kind, "conc": c, "job": job, "run": run_idx, "fill": not a.no_fill,
           "prompt_version": a.prompt_version, "max_tokens": a.max_tokens,
           "streams_ok": len(ok), "streams_failed": c - len(ok),
           "aggregate_tps": round(agg, 2), "mean_tps": round(statistics.fmean(tps), 2) if tps else 0.0,
           "median_tps": round(statistics.median(tps), 2) if tps else 0.0,
           "min_tps": round(min(tps), 2) if tps else 0.0, "max_tps": round(max(tps), 2) if tps else 0.0,
           "mean_ttft_ms": round(statistics.fmean(r["ttft_ms"] for r in ok), 1) if ok else None,
           "median_ttft_ms": round(statistics.median(r["ttft_ms"] for r in ok), 1) if ok else None,
           "total_decode_tokens": sum(r["decode_tokens"] for r in ok), "wall_s": round(wall, 3), "spec": sd,
           "errors": sorted({r["error"] for r in res if r and r["error"]})[:3],
           "streams": [{"i": i, "prompt_sha": hashlib.sha256(prompts[i].encode()).hexdigest()[:12],
                        **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in (r or {}).items()
                           if k in ("completion_tokens", "prompt_tokens", "cached_tokens", "decode_tokens",
                                    "decode_tps", "ttft_ms", "decode_ms", "finish_reason", "error", "text_sha",
                                    "n_text_events", "retried")}} for i, r in enumerate(res)]}
    if sd and sd.get("steps") and rec["mean_tps"]:
        rec["implied_ms_per_step"] = round(1000 * sd["AL"] / rec["mean_tps"], 2)
    out.write(json.dumps(rec) + "\n")
    out.flush()
    log("%-10s c%-2d job %-9s run %d  agg %7.1f  /str %6.1f  ttft %6.1f ms  AL %s  ms/step %s  ok %d/%d%s" % (
        kind, c, job, run_idx, rec["aggregate_tps"], rec["mean_tps"], rec["mean_ttft_ms"] or 0,
        sd.get("AL") if sd else "-", rec.get("implied_ms_per_step", "-"), len(ok), c,
        ("  ERR " + "; ".join(rec["errors"])) if rec["errors"] else ""))
    return rec


def run_job(a, kind, concs, job, run_idx, log, out):
    w = stream_request(a.base, request_body(a.model, warmup_prompt(kind), 32, fill=False,
                                            think_flags=a.think_flags), a.timeout, a.api_key)
    out.write(json.dumps({"kind": "warmup", "type": kind, "job": job, "ok": not w["error"], "error": w["error"],
                          "completion_tokens": w["completion_tokens"]}) + "\n")
    return [run_wave(a, kind, c, job, run_idx, log, out) for c in sorted(concs)]


def summarize(waves, log):
    cells = {}
    for w in waves:
        if w["job"] == "discard":
            continue
        cells.setdefault((w["type"], w["conc"]), []).append(w)
    log("\nsparkDash 1.8.8 decode (%s), aggregate tok/s (per-stream mean); c1/c4 = median over runs" %
        ("fill-force on" if waves and waves[0]["fill"] else "fill-force OFF"))
    concs = sorted({c for _, c in cells})
    log("%-11s" % "type" + "".join("%20s" % ("c%d" % c) for c in concs) + "   | knapcio c1 off/abl  c16 off")
    summary = {}
    for kind in TYPES:
        row, srow = "%-11s" % kind, {}
        for c in concs:
            ws = cells.get((kind, c))
            if not ws:
                row += "%20s" % "-"
                continue
            agg = statistics.median(x["aggregate_tps"] for x in ws)
            per = statistics.median(x["mean_tps"] for x in ws)
            als = [x["spec"]["AL"] for x in ws if x.get("spec") and x["spec"].get("AL")]
            mss = [x["implied_ms_per_step"] for x in ws if x.get("implied_ms_per_step")]
            srow[c] = {"aggregate": round(agg, 1), "per_stream": round(per, 1), "runs": len(ws),
                       "all_aggregate": [x["aggregate_tps"] for x in ws],
                       "AL": round(statistics.median(als), 3) if als else None,
                       "ms_per_step": round(statistics.median(mss), 2) if mss else None,
                       "text_identical_across_runs": len({tuple(s["text_sha"] for s in x["streams"]) for x in ws}) == 1}
            row += "%20s" % ("%.1f" % agg if c == 1 else "%.1f (%.1f)" % (agg, per))
        k1o = KNAPCIO["official"].get(kind, {}).get(1)
        k1a = KNAPCIO["abliterated"].get(kind, {}).get(1)
        k16 = KNAPCIO["official"].get(kind, {}).get(16)
        row += "   | %s / %s   %s" % (k1o, k1a, k16[0] if k16 else "-")
        log(row)
        summary[kind] = srow
    log("\nc1 decomposition (AL from /metrics; ms/step = AL / tok/s):")
    for kind in TYPES:
        s = summary.get(kind, {}).get(1)
        if s:
            log("  %-10s c1 %6.1f tok/s  AL %s  %s ms/step  runs %s  text identical %s" % (
                kind, s["aggregate"], s["AL"], s["ms_per_step"], s["all_aggregate"], s["text_identical_across_runs"]))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default="sd_decode.jsonl")
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--conc", default="1,2,4,8,16", help="levels for --protocol single / job A")
    ap.add_argument("--protocol", choices=["knapcio", "single"], default="knapcio")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-version", choices=["1.8.8", "head"], default="1.8.8")
    ap.add_argument("--think-flags", choices=["sparkdash", "minimal"], default="sparkdash",
                    help="sparkdash: {enable_thinking,thinking:false,thinking_mode:disabled}; minimal: {thinking:false}")
    ap.add_argument("--no-fill", action="store_true", help="omit min_tokens/ignore_eos/stop (diagnostic, not sparkDash)")
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--timeout", type=float, default=360.0)
    a = ap.parse_args()
    a.base = a.url.rstrip("/")
    sha = prompt_set_sha()
    if sha != PROMPT_SET_SHA256:
        sys.exit("prompt set hash mismatch: %s" % sha)
    types = [t for t in a.types.split(",") if t]
    concs = sorted({int(x) for x in a.conc.split(",") if x})
    out = open(a.out, "a")

    def log(msg):
        print(msg, flush=True)

    out.write(json.dumps({"kind": "meta", "tool": "sparkdash_decode.py", "reproduces": "sparkDash 1.8.8 754f40a7",
                          "prompt_set_sha256": sha, "args": {k: v for k, v in vars(a).items() if k != "api_key"},
                          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    waves = []
    if a.protocol == "knapcio":
        log("[warm-up] two discarded prose c1 jobs")
        for i in range(2):
            waves += run_job(a, "prose", [1], "discard", i, log, out)
        for kind in types:
            waves += run_job(a, kind, concs, "A", 0, log, out)
            waves += run_job(a, kind, [c for c in (1, 4) if c in concs], "B", 1, log, out)
            if 1 in concs:
                waves += run_job(a, kind, [1], "C", 2, log, out)
    else:
        for kind in types:
            waves += run_job(a, kind, concs, "single", 0, log, out)
    summary = summarize(waves, log)
    out.write(json.dumps({"kind": "summary", "cells": {k: {str(c): v for c, v in d.items()} for k, d in summary.items()},
                          "knapcio": {k: {t: {str(c): v for c, v in d.items()} for t, d in tv.items()}
                                      for k, tv in KNAPCIO.items()},
                          "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    out.close()


if __name__ == "__main__":
    main()
