#!/usr/bin/env python3
"""dsv41 live benchmark (stdlib only). Streams SSE from an OpenAI-compatible vLLM server.

Usage:
  python3 bench.py single  [--temp 0] [--max-tokens 1024] [--repeats 2] [--types code,prose,math] [--min-tokens N]
  python3 bench.py conc    --conc 4 [--max-tokens 512] [--temp 0]
  python3 bench.py ttft    --lens 4096,16384,32768 [--max-tokens 16]
  python3 bench.py metrics                      # dump spec-decode counters
  common: --url http://127.0.0.1:8000 --model dsv41 --out results.jsonl --label X --no-think
  sampled decoding (r3): [--top-p P] [--seed S] or --preset t1 (temp 1.0, top_p 0.95) / t06 (temp 0.6, top_p 1.0)
    --top-p unset keeps the old behaviour (top_p=1.0 sent only when temp==0, server default otherwise).
    --seed S sends seed=S+offset per request (offset = rep*1000+type index / request index) for paired arms.
Each request row (JSON) is appended to --out and printed. Every mode also scrapes /metrics before/after
and prints spec-decode deltas (mean acceptance length, per-position acceptance).
"""
import argparse, json, random, re, sys, threading, time, urllib.request

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

WORDS = ("alpha river stone copper lantern violet harbor meadow quartz falcon ember cedar willow orbit prism tundra "
         "saffron canyon glacier marble thistle cobalt nectar summit ripple juniper basalt comet fern zephyr opal "
         "granite heron lagoon mosaic pepper quill raven sierra tidal umber velvet walnut yonder zinc amber birch "
         "cinder dune estuary fjord garnet hazel iris jade kelp lilac maple nimbus ochre pine").split()


def http_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def scrape(base):
    txt = http_json(base + "/metrics")
    m = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "pos": {}}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            m["drafts"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            m["draft_tokens"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            m["accepted"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            p = re.search(r'position="(\d+)"', line)
            if p:
                m["pos"][int(p.group(1))] = float(line.split()[-1])
    return m


def spec_delta(a, b):
    d = b["drafts"] - a["drafts"]
    if d <= 0:
        return {"drafts": 0}
    acc = b["accepted"] - a["accepted"]
    dt = b["draft_tokens"] - a["draft_tokens"]
    pos = {k: round((b["pos"].get(k, 0) - a["pos"].get(k, 0)) / d, 4) for k in sorted(b["pos"])}
    return {"drafts": int(d), "draft_tokens": int(dt), "accepted": int(acc),
            "mean_accept_len": round(1 + acc / d, 3), "draft_accept_rate": round(acc / dt, 4) if dt else None,
            "drafts_per_step": round(dt / d, 3), "per_pos": pos}


def stream_chat(base, model, messages, max_tokens, temp, min_tokens=None, no_think=False, timeout=1800,
                top_p=None, seed=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temp,
            "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": False}}
    if top_p is not None:
        body["top_p"] = top_p
    elif temp == 0:
        body["top_p"] = 1.0
    if seed is not None:
        body["seed"] = int(seed)
    if min_tokens:
        body["min_tokens"] = min_tokens
    if no_think:
        body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    chunk_times = []
    usage = None
    finish = None
    nchars = {"content": 0, "reasoning": 0}
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
                    chunk_times.append(now)
                    nchars["content"] += len(c)
                    nchars["reasoning"] += len(rc)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.perf_counter()
    ct = usage.get("completion_tokens") if usage else None
    pt = usage.get("prompt_tokens") if usage else None
    res = {"prompt_tokens": pt, "completion_tokens": ct, "finish": finish,
           "ttft_s": round(t_first - t0, 4) if t_first else None, "e2e_s": round(t_end - t0, 4),
           "chunks": len(chunk_times), "chars": nchars}
    if t_first and ct and len(chunk_times) > 1:
        dec = chunk_times[-1] - t_first
        res["decode_s"] = round(dec, 4)
        res["decode_tok_s"] = round((ct - 1) / dec, 2) if dec > 0 else None
        gaps = sorted(b - a for a, b in zip(chunk_times, chunk_times[1:]))
        res["itl_chunk_ms_p50"] = round(1000 * gaps[len(gaps) // 2], 2)
        res["itl_chunk_ms_p90"] = round(1000 * gaps[int(len(gaps) * 0.9)], 2)
        res["tokens_per_chunk"] = round(ct / len(chunk_times), 3)
    return res


def emit(args, row):
    row["label"] = args.label
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    s = json.dumps(row)
    print(s, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(s + "\n")


def synth_text(target_tokens, seed):
    rng = random.Random(seed)
    parts = [f"Document nonce {seed}-{rng.random():.12f}. Read the numbered notes below."]
    # ~1 token per word + numbering; empirically ~14-15 tokens per sentence
    n = max(1, int(target_tokens / 14.5))
    for i in range(1, n + 1):
        w = " ".join(rng.choice(WORDS) for _ in range(9))
        parts.append(f"{i}. Note {rng.randint(1000, 99999)}: {w}.")
    parts.append("Question: in one short phrase, what is note 1 about?")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "conc", "ttft", "metrics"])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--min-tokens", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--types", default="code,prose,math")
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--lens", default="4096,16384,32768")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--preset", choices=["t1", "t06"], default=None,
                    help="t1 = temp 1.0 top_p 0.95; t06 = temp 0.6 top_p 1.0 (overrides --temp/--top-p)")
    args = ap.parse_args()
    if args.preset == "t1":
        args.temp, args.top_p = 1.0, 0.95
    elif args.preset == "t06":
        args.temp, args.top_p = 0.6, 1.0
    def req_seed(off):
        return None if args.seed is None else args.seed + off
    base = args.url.rstrip("/")
    m0 = scrape(base)
    if args.mode == "metrics":
        print(json.dumps(m0))
        return
    t_start = time.perf_counter()
    if args.mode == "single":
        mt = args.max_tokens or 1024
        for rep in range(args.repeats):
            for ty in args.types.split(","):
                a = scrape(base)
                r = stream_chat(base, args.model, [{"role": "user", "content": PROMPTS[ty]}], mt, args.temp,
                                args.min_tokens, args.no_think, top_p=args.top_p,
                                seed=req_seed(rep * 1000 + list(PROMPTS).index(ty)))
                r.update({"mode": "single", "type": ty, "rep": rep, "temp": args.temp, "max_tokens": mt,
                          "spec": spec_delta(a, scrape(base))})
                if args.top_p is not None:
                    r["top_p"] = args.top_p
                emit(args, r)
    elif args.mode == "conc":
        mt = args.max_tokens or 512
        tys = args.types.split(",")
        results = [None] * args.conc
        def worker(i):
            ty = tys[i % len(tys)]
            msg = [{"role": "user", "content": f"(request {i}) " + PROMPTS[ty]}]
            try:
                results[i] = stream_chat(base, args.model, msg, mt, args.temp, args.min_tokens, args.no_think,
                                         top_p=args.top_p, seed=req_seed(i))
            except Exception as e:
                results[i] = {"error": repr(e)}
            results[i].update({"type": ty, "i": i})
        t0 = time.perf_counter()
        th = [threading.Thread(target=worker, args=(i,)) for i in range(args.conc)]
        for t in th: t.start()
        for t in th: t.join()
        wall = time.perf_counter() - t0
        tot = sum(r.get("completion_tokens") or 0 for r in results)
        per = [r.get("decode_tok_s") for r in results if r.get("decode_tok_s")]
        emit(args, {"mode": "conc", "conc": args.conc, "temp": args.temp, "top_p": args.top_p,
                    "max_tokens": mt, "wall_s": round(wall, 3),
                    "total_completion_tokens": tot, "agg_tok_s": round(tot / wall, 2),
                    "per_req_decode_tok_s_mean": round(sum(per) / len(per), 2) if per else None,
                    "per_req_decode_tok_s_min": min(per) if per else None,
                    "ttft_s": [r.get("ttft_s") for r in results], "requests": results,
                    "spec": spec_delta(m0, scrape(base))})
    elif args.mode == "ttft":
        mt = args.max_tokens or 16
        for L in [int(x) for x in args.lens.split(",")]:
            seed = int(time.time() * 1000) % 10**9 + L
            a = scrape(base)
            r = stream_chat(base, args.model, [{"role": "user", "content": synth_text(L, seed)}], mt, 0.0,
                            None, args.no_think)
            r.update({"mode": "ttft", "target_len": L, "spec": spec_delta(a, scrape(base))})
            if r.get("prompt_tokens") and r.get("ttft_s"):
                r["prefill_tok_s"] = round(r["prompt_tokens"] / r["ttft_s"], 1)
            emit(args, r)
    print(json.dumps({"summary_spec_delta": spec_delta(m0, scrape(base)),
                      "elapsed_s": round(time.perf_counter() - t_start, 1)}), flush=True)


if __name__ == "__main__":
    main()
