#!/usr/bin/env python3
"""sparkdash_prefill.py -- stdlib-only reproduction of sparkDash's PrefillBench for OUR OpenAI endpoint.

Source reproduced (read, never executed): MiaAI-Lab/sparkDash 1.8.8 (754f40a7) server/collectors/PrefillBench.js;
--filler words reproduces HEAD (c6e825cd) server/collectors/contextFill.js.

Per size n (tokens, estimated as chars/4 the sparkDash way):
  salt = uuid4; header = "[prefill-bench <salt>]\\nIgnore the filler below. Reply with the single word OK.\\n"
  footer = "\\nReply OK."; fill = max(1, max(8, n) - max(1, round(len(header + footer) / 4)))
  1.8.8 filler: " the" * fill  (ONE repeated token -- knapcio: "inflates these numbers", MiaAI-Lab#21)
  HEAD  filler: LCG-ordered common words (seed 0x2545f491), " " + " ".join(words)
  request: max_tokens 8, temperature 0, top_p 1, stream, include_usage, thinking-off flags (as decode bench)
  prefill tok/s = usage.prompt_tokens / TTFT, TTFT = request start -> first SSE event with text.
sparkDash job = one 512-token warm-up, then the sizes in ascending order, one request each (c1).
knapcio protocol (default): "cold (salted prompts), after a 4k warm-up, median of the second and third of three
runs per length", sizes 4k/16k/32k/64k/128k -> here: 512 + 4096 warm-ups, then 3 passes over the sizes,
reported value = median(pass 2, pass 3) (= their mean). cached_tokens is recorded (our server runs
--enable-prompt-tokens-details) and must be ~0 for a cold number.

  python3 sparkdash_prefill.py --out runs/X/sd_prefill.jsonl                 # ~2.5 min at ~5.5K tok/s
  python3 sparkdash_prefill.py --filler words --out runs/X/sd_prefill_words.jsonl
"""
import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
import uuid

KNAPCIO = {4096: 4798, 16384: 5976, 32768: 6196, 65536: 6241, 131072: 6291}   # 2026-10-23, official 1017
KNAPCIO_ABL = (4855, 6214)                                                     # abliterated 4k..128k range
FILLER_WORDS = ("the of and to in is you that it he was for on are as with his they at be this have from or one "
                "had by word but not what all were we when your can said there use an each which she do how their "
                "if will up other about out many then them these so some her would make like him into time").split(" ")
THINK_OFF = {"enable_thinking": False, "thinking": False, "thinking_mode": "disabled"}


def js_round(x):
    return math.floor(x + 0.5)


def estimate_tokens(text):
    return max(1, js_round(len(text) / 4)) if text else 0


def filler_words(count):
    words, state = [], 0x2545F491
    for _ in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        words.append(FILLER_WORDS[(state >> 8) % len(FILLER_WORDS)])
    return " " + " ".join(words)


def build_prompt(target, salt, filler="the"):
    n = max(8, js_round(float(target)))
    header = "[prefill-bench %s]\nIgnore the filler below. Reply with the single word OK.\n" % salt
    footer = "\nReply OK."
    fill = max(1, n - estimate_tokens(header + footer))
    body = " the" * fill if filler == "the" else filler_words(fill)
    return header + body + footer


def timeout_s(n):
    return min(2700.0, max(90.0, 60.0 + n * 0.008))


def one(base, model, target, filler, think_flags, api_key=None):
    salt = str(uuid.uuid4())
    prompt = build_prompt(target, salt, filler)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 8, "temperature": 0,
            "top_p": 1, "stream": True, "stream_options": {"include_usage": True}}
    body["chat_template_kwargs"] = dict(THINK_OFF) if think_flags == "sparkdash" else {"thinking": False}
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        h["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h)
    t0 = time.perf_counter()
    t_first, usage, err = None, None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s(target)) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except ValueError:
                    continue
                if isinstance(j.get("usage"), dict):
                    usage = j["usage"]
                for ch in (j.get("choices") or [])[:1]:
                    d = ch.get("delta") or {}
                    if (d.get("content") or d.get("reasoning") or d.get("reasoning_content")) and t_first is None:
                        t_first = time.perf_counter()
    except urllib.error.HTTPError as e:
        err = "HTTP %d: %s" % (e.code, e.read().decode(errors="replace")[:200])
    except Exception as e:  # noqa: BLE001
        err = "stream: %r" % (e,)
    u = usage or {}
    pt = u.get("prompt_tokens") or 0
    ttft = (t_first - t0) if t_first else None
    return {"target": target, "filler": filler, "prompt_chars": len(prompt), "prompt_tokens": pt,
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "ttft_s": round(ttft, 4) if ttft else None,
            "prefill_tps": round(pt / ttft, 1) if ttft and pt else 0.0,
            "completion_tokens": u.get("completion_tokens"), "error": err or (None if ttft else "no first token")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--sizes", default="4096,16384,32768,65536,131072")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--filler", choices=["the", "words"], default="the", help="the = sparkDash 1.8.8, words = HEAD")
    ap.add_argument("--think-flags", choices=["sparkdash", "minimal"], default="sparkdash")
    ap.add_argument("--no-4k-warmup", action="store_true")
    ap.add_argument("--out", default="sd_prefill.jsonl")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    sizes = sorted(int(s) for s in a.sizes.split(","))
    out = open(a.out, "a")

    def emit(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    emit({"kind": "meta", "tool": "sparkdash_prefill.py", "reproduces": "sparkDash 1.8.8 754f40a7 PrefillBench",
          "args": {k: v for k, v in vars(a).items() if k != "api_key"},
          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    warm = [512] + ([] if a.no_4k_warmup else [4096])
    for w in warm:
        r = one(base, a.model, w, a.filler, a.think_flags, a.api_key)
        emit(dict(r, kind="warmup"))
        print("warm-up %6d: %s" % (w, r["error"] or "%.0f tok/s" % r["prefill_tps"]), flush=True)
    res = {s: [] for s in sizes}
    for p in range(a.passes):
        for s in sizes:
            r = one(base, a.model, s, a.filler, a.think_flags, a.api_key)
            r.update(kind="measure", pass_=p)
            emit(r)
            res[s].append(r)
            print("pass %d %6d tok: prompt %7d cached %s ttft %7.3f s -> %7.0f tok/s %s" % (
                p + 1, s, r["prompt_tokens"], r["cached_tokens"], r["ttft_s"] or 0, r["prefill_tps"],
                r["error"] or ""), flush=True)
    print("\nsize      ours (median of passes 2-3)   all passes            knapcio 1017")
    summ = {}
    for s in sizes:
        v = [r["prefill_tps"] for r in res[s] if not r["error"]]
        tail = v[1:3] if len(v) >= 3 else v
        med = statistics.median(tail) if tail else 0.0
        summ[s] = {"tps": round(med, 1), "all": v, "ttft_s": [r["ttft_s"] for r in res[s]]}
        print("%-8d %10.0f                    %-22s %s" % (s, med, ",".join("%.0f" % x for x in v), KNAPCIO.get(s, "-")))
    emit({"kind": "summary", "filler": a.filler, "sizes": {str(k): v for k, v in summ.items()},
          "knapcio": {str(k): v for k, v in KNAPCIO.items()},
          "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    out.close()


if __name__ == "__main__":
    main()
