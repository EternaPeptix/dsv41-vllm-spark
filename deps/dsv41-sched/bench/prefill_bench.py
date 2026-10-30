#!/usr/bin/env python3
"""r4 prefill/TTFT/starvation bench for the DSv4.1 vLLM endpoint (stdlib only).

Run on spark-one (or anywhere that reaches the API). Every prompt starts with a
per-run nonce and is built from random words, so the prefix cache never hits.
Token counts are calibrated from the server's usage.prompt_tokens.

Modes
  prefill   cold TTFT sweep, one request at a time:
              python3 prefill_bench.py prefill --lens 16k,64k,128k,200k --reps 2 --out pf.jsonl
            reports prefill tok/s = prompt_tokens / TTFT.
  starve    short-request latency while a long prompt prefills (+ ITL of a
            request that was already decoding):
              python3 prefill_bench.py starve --long 128k --shorts 6 --short-every 2 \\
                  --short-delay 3 --decoder --out st.jsonl
  logprobs  fixed prompt set (2.5k..24k tokens, deterministic text), greedy,
            top-5 logprobs of the first --max-tokens tokens -> JSON for A/B:
              python3 prefill_bench.py logprobs --out lp_base.json
  compare   compare two logprobs files:
              python3 prefill_bench.py compare lp_base.json lp_new.json
Last stdout line of every mode is a one-line JSON summary.
"""
import argparse, json, random, statistics, sys, threading, time, urllib.request

WORDS = None


def words(rng, n):
    global WORDS
    if WORDS is None:
        r = random.Random(1234)
        syl = ["ka", "lo", "ve", "mar", "tin", "sel", "dor", "an", "bri", "cus", "el", "fen", "gar",
               "hal", "is", "jor", "kel", "lun", "mo", "nor", "ost", "pra", "quel", "ros", "sun", "tar",
               "ul", "vin", "wen", "yr", "zan", "bel", "cor", "dra", "esh", "fal", "gol", "hen"]
        WORDS = sorted({"".join(r.choice(syl) for _ in range(r.choice([1, 2, 2, 3]))) for _ in range(6000)})
    out = []
    for i in range(n):
        out.append(rng.choice(WORDS))
        if i % 14 == 13:
            out[-1] += "."
    return " ".join(out)


def parse_len(s):
    s = s.strip().lower()
    return int(float(s[:-1]) * 1024) if s.endswith("k") else int(s)


TOK_PER_WORD = [1.9]  # refined after each response (usage.prompt_tokens)


def make_prompt(target_tokens, seed, question="Summarize the text above in one sentence."):
    rng = random.Random(seed)
    nonce = f"[run {time.time_ns()} seed {seed}]"
    nw = max(int(target_tokens / TOK_PER_WORD[0]), 1)
    return f"{nonce}\n{words(rng, nw)}\n\n{question}", nw


def stream(base, prompt, max_tokens, timeout=1800, logprobs=0, model="dsv41"):
    """Returns dict(ttft, t_end, chunk_times, text, usage, logprobs)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = logprobs
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    res = {"t0": t0, "ttft": None, "chunk_times": [], "text": "", "usage": None, "logprobs": []}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                res["usage"] = d["usage"]
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                piece = (dl.get("content") or "") + (dl.get("reasoning_content") or dl.get("reasoning") or "")
                lp = (ch.get("logprobs") or {}).get("content") or []
                if piece or lp:
                    now = time.monotonic()
                    if res["ttft"] is None:
                        res["ttft"] = now - t0
                    res["chunk_times"].append(now - t0)
                    res["text"] += piece
                    for e in lp:
                        res["logprobs"].append({"tok": e.get("token"), "lp": e.get("logprob"),
                                                "top": [(t.get("token"), t.get("logprob"))
                                                        for t in e.get("top_logprobs") or []]})
    res["t_end"] = time.monotonic() - t0
    return res


def calibrate(usage, nw):
    if usage and usage.get("prompt_tokens") and nw > 100:
        TOK_PER_WORD[0] = 0.5 * TOK_PER_WORD[0] + 0.5 * (usage["prompt_tokens"] / nw)


def mode_prefill(a):
    out = open(a.out, "a") if a.out else None
    rows = []
    stream(a.base, make_prompt(512, 7)[0], 4)  # warm, discarded
    for L in [parse_len(x) for x in a.lens.split(",")]:
        for rep in range(a.reps):
            p, nw = make_prompt(L, 1000 * rep + L)
            r = stream(a.base, p, a.max_tokens)
            calibrate(r["usage"], nw)
            pt = (r["usage"] or {}).get("prompt_tokens", 0)
            row = {"mode": "prefill", "label": a.label, "target": L, "prompt_tokens": pt,
                   "ttft_s": round(r["ttft"], 3), "prefill_tok_s": round(pt / r["ttft"], 1)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            if out:
                out.write(json.dumps(row) + "\n")
    summ = {"mode": "prefill", "label": a.label}
    for L in sorted({r["target"] for r in rows}):
        v = [r["prefill_tok_s"] for r in rows if r["target"] == L]
        summ[f"{L // 1024}k_tok_s"] = round(statistics.median(v), 1)
    print(json.dumps(summ))


def mode_starve(a):
    L = parse_len(a.long)
    # calibrate tokens/word on a mid prompt first
    p, nw = make_prompt(8192, 3)
    calibrate(stream(a.base, p, 2)["usage"], nw)
    results = {"long": None, "shorts": [], "decoder": None}
    t_base = time.monotonic()

    def run_long():
        p, nw = make_prompt(L, 99)
        r = stream(a.base, p, 8)
        r["start"] = r["t0"] - t_base
        results["long"] = r

    def run_short(i, delay):
        time.sleep(delay)
        p, _ = make_prompt(a.short_tokens, 500 + i, "Reply with one word: ok")
        r = stream(a.base, p, a.short_max_tokens)
        r["start"] = r["t0"] - t_base
        results["shorts"].append(r)

    def run_decoder():
        p, _ = make_prompt(256, 77, "Write a long story about a lighthouse keeper. Do not stop early.")
        r = stream(a.base, p, a.decoder_tokens)
        r["start"] = r["t0"] - t_base
        results["decoder"] = r

    th = []
    if a.decoder:
        th.append(threading.Thread(target=run_decoder))
        th[-1].start()
        time.sleep(a.decoder_lead)
    t_long = time.monotonic()
    th.append(threading.Thread(target=run_long))
    th[-1].start()
    for i in range(a.shorts):
        th.append(threading.Thread(target=run_short, args=(i, a.short_delay + i * a.short_every)))
        th[-1].start()
    for t in th:
        t.join()
    lg = results["long"]
    long_window = (lg["start"], lg["start"] + lg["ttft"])
    sh = sorted(results["shorts"], key=lambda r: r["start"])
    row = {"mode": "starve", "label": a.label,
           "long_prompt_tokens": (lg["usage"] or {}).get("prompt_tokens"),
           "long_ttft_s": round(lg["ttft"], 2),
           "short_ttft_s": [round(r["ttft"], 2) for r in sh],
           "short_e2e_s": [round(r["t_end"], 2) for r in sh],
           "short_started_inside_long_prefill": sum(1 for r in sh if long_window[0] <= r["start"] < long_window[1])}
    if results["decoder"]:
        d = results["decoder"]
        ts = [d["start"] + t for t in d["chunk_times"]]
        gaps = [(t1 - t0, t0) for t0, t1 in zip(ts, ts[1:])]
        inside = [g for g, t0 in gaps if long_window[0] <= t0 < long_window[1]]
        row.update({"decoder_itl_inside_ms_p50": round(1000 * statistics.median(inside), 1) if inside else None,
                    "decoder_itl_inside_ms_max": round(1000 * max(inside), 1) if inside else None,
                    "decoder_itl_outside_ms_p50": round(1000 * statistics.median([g for g, t0 in gaps if not long_window[0] <= t0 < long_window[1]] or [0]), 1),
                    "decoder_tokens": (d["usage"] or {}).get("completion_tokens")})
    row["short_ttft_p50_s"] = round(statistics.median(row["short_ttft_s"]), 2) if sh else None
    row["short_ttft_max_s"] = round(max(row["short_ttft_s"]), 2) if sh else None
    if a.out:
        open(a.out, "a").write(json.dumps(row) + "\n")
    print(json.dumps(row))


# deterministic prompt set for logprob A/B (no nonce: identical across runs;
# send each prompt once per server boot so the prefix cache cannot matter, or
# set --salt to a fresh value per run to defeat it explicitly).
def lp_prompts(salt):
    out = []
    for i, L in enumerate([2600, 4100, 8300, 12500, 16500, 24600]):
        rng = random.Random(42 + i)
        nw = int(L / 1.9)
        q = ["List the first five words of the text above.", "Which word appears right after the third period?",
             "Count the sentences in the first paragraph-like block and explain briefly.",
             "Write a haiku inspired by the text above.", "Repeat the last sentence of the text above verbatim.",
             "Give a one-line summary."][i]
        out.append((L, f"[salt {salt}]\n{words(rng, nw)}\n\n{q}"))
    return out


def mode_logprobs(a):
    res = []
    for L, p in lp_prompts(a.salt):
        r = stream(a.base, p, a.max_tokens, logprobs=5)
        res.append({"target": L, "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
                    "text": r["text"], "logprobs": r["logprobs"], "ttft": r["ttft"]})
        print(L, res[-1]["prompt_tokens"], repr(r["text"][:60]), flush=True)
    json.dump({"label": a.label, "salt": a.salt, "results": res}, open(a.out, "w"))
    print(json.dumps({"mode": "logprobs", "label": a.label, "n": len(res), "out": a.out}))


def mode_compare(a):
    A = json.load(open(a.files[0]))["results"]
    B = json.load(open(a.files[1]))["results"]
    rows, first_eq, dlp_all = [], 0, []
    for x, y in zip(A, B):
        la, lb = x["logprobs"], y["logprobs"]
        div = next((i for i, (u, v) in enumerate(zip(la, lb)) if u["tok"] != v["tok"]), min(len(la), len(lb)))
        d = [abs(u["lp"] - v["lp"]) for u, v in zip(la[:div], lb[:div]) if u["lp"] is not None and v["lp"] is not None]
        # top-5 overlap at token 0
        t0a = {t for t, _ in la[0]["top"]} if la else set()
        t0b = {t for t, _ in lb[0]["top"]} if lb else set()
        first_eq += int(bool(la and lb and la[0]["tok"] == lb[0]["tok"]))
        dlp_all += d
        rows.append({"target": x["target"], "diverge_at": div, "n": min(len(la), len(lb)),
                     "mean_abs_dlp": round(statistics.mean(d), 4) if d else None,
                     "max_abs_dlp": round(max(d), 4) if d else None,
                     "top5_overlap_tok0": len(t0a & t0b)})
        print(json.dumps(rows[-1]))
    print(json.dumps({"mode": "compare", "first_token_equal": f"{first_eq}/{len(rows)}",
                      "mean_abs_dlp_prediv": round(statistics.mean(dlp_all), 4) if dlp_all else None,
                      "max_abs_dlp_prediv": round(max(dlp_all), 4) if dlp_all else None,
                      "min_diverge_at": min(r["diverge_at"] for r in rows)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prefill", "starve", "logprobs", "compare"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--lens", default="16k,64k,128k,200k")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--long", default="128k")
    ap.add_argument("--shorts", type=int, default=6)
    ap.add_argument("--short-tokens", type=int, default=300)
    ap.add_argument("--short-max-tokens", type=int, default=16)
    ap.add_argument("--short-delay", type=float, default=3.0)
    ap.add_argument("--short-every", type=float, default=2.0)
    ap.add_argument("--decoder", action="store_true")
    ap.add_argument("--decoder-lead", type=float, default=3.0)
    ap.add_argument("--decoder-tokens", type=int, default=3000)
    ap.add_argument("--salt", default="r4")
    a = ap.parse_args()
    if a.mode == "logprobs" and a.max_tokens == 8:
        a.max_tokens = 48
    {"prefill": mode_prefill, "starve": mode_starve, "logprobs": mode_logprobs, "compare": mode_compare}[a.mode](a)


if __name__ == "__main__":
    main()
