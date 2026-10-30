#!/usr/bin/env python3
"""Short-request starvation bench (stdlib only; reuses r4-e5 prefill_bench.py helpers).

starve: one cold long prompt + N short chats injected at spaced offsets + one live
        decoder stream (started before the long prompt, closed when the long prompt's
        first token arrives).  One JSON row per run.
solo:   the same long prompt alone (TTFT), --reps times.
warm:   one ~24K-token prompt + one short (post-restart first-use warmup; discarded).

  python3 sbench.py starve --label X --long 128k --offsets 2,5,8,11,14,17 --out st.jsonl
  python3 sbench.py solo   --label X --long 128k --reps 2 --out solo.jsonl
Every prompt starts with a nonce, so the prefix cache never hits.
"""
import argparse, json, os, statistics, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill_bench as pb  # noqa: E402


def pct(v, q):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def stream_until(base, prompt, max_tokens, stop_ev, model="dsv41"):
    """Streaming decoder that returns when stop_ev is set (closes the connection)."""
    import urllib.request
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    times = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                if (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")):
                    times.append(time.monotonic())
            if stop_ev.is_set():
                break
    return t0, times


# Fixed tokens/word (measured 2.909 on this tokenizer: 31,462 words -> 91,534 tokens), so every
# arm gets the same prompt sizes; calib() only sends two small settle requests.
pb.TOK_PER_WORD[0] = 2.909


def calib(a):
    for s in (3, 4):
        pb.stream(a.base, pb.make_prompt(8192, s)[0], 2)


def mode_starve(a):
    L = pb.parse_len(a.long)
    offs = [float(x) for x in a.offsets.split(",")]
    for run in range(a.runs):
        calib(a)
        res = {"long": None, "shorts": [], "dec": None}
        stop_ev = threading.Event()
        T0 = time.monotonic()

        def run_long():
            p, nw = pb.make_prompt(L, 99 + run)
            try:
                res["long"] = pb.stream(a.base, p, 8, timeout=3600)
            finally:
                stop_ev.set()

        def run_short(i, d):
            time.sleep(max(0.0, T0 + d - time.monotonic()))
            p, _ = pb.make_prompt(a.short_tokens, 500 + i + 37 * run, "Reply with one word: ok")
            r = pb.stream(a.base, p, a.short_max_tokens, timeout=3600)
            r["off"] = d
            res["shorts"].append(r)

        def run_dec():
            p, _ = pb.make_prompt(256, 77 + run, "Write a long story about a lighthouse keeper. Do not stop early.")
            res["dec"] = stream_until(a.base, p, a.decoder_tokens, stop_ev)

        th = []
        if a.decoder:
            th.append(threading.Thread(target=run_dec)); th[-1].start()
            time.sleep(a.decoder_lead)
            T0 = time.monotonic()
        th.append(threading.Thread(target=run_long)); th[-1].start()
        for i, d in enumerate(offs):
            th.append(threading.Thread(target=run_short, args=(i, d))); th[-1].start()
        for t in th:
            t.join()
        lg = res["long"]
        lw = (lg["t0"], lg["t0"] + lg["ttft"])
        sh = sorted(res["shorts"], key=lambda r: r["off"])
        st = [r["ttft"] for r in sh]
        row = {"mode": "starve", "label": a.label, "run": run, "t": time.time(),
               "long_prompt_tokens": (lg["usage"] or {}).get("prompt_tokens"),
               "long_ttft_s": round(lg["ttft"], 3),
               "short_offsets_s": offs,
               "short_ttft_s": [round(x, 3) for x in st],
               "short_inside": sum(1 for r in sh if lw[0] <= r["t0"] < lw[1]),
               "short_ttft_p50_s": round(statistics.median(st), 3),
               "short_ttft_p95_s": round(pct(st, 0.95), 3),
               "short_ttft_max_s": round(max(st), 3)}
        if res["dec"]:
            t0, ts = res["dec"]
            gaps = [(b - c, c) for c, b in zip(ts, ts[1:])]
            ins = [g for g, c in gaps if lw[0] <= c < lw[1]]
            row.update({"dec_gap_inside_p50_ms": round(1000 * statistics.median(ins), 1) if ins else None,
                        "dec_gap_inside_max_ms": round(1000 * max(ins), 1) if ins else None,
                        "dec_chunks_inside": len(ins)})
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        time.sleep(a.gap)


def mode_solo(a):
    L = pb.parse_len(a.long)
    for rep in range(a.reps):
        calib(a)
        p, nw = pb.make_prompt(L, 4242 + rep)
        r = pb.stream(a.base, p, 8, timeout=3600)
        pt = (r["usage"] or {}).get("prompt_tokens", 0)
        row = {"mode": "solo", "label": a.label, "rep": rep, "t": time.time(), "prompt_tokens": pt,
               "ttft_s": round(r["ttft"], 3), "tok_s": round(pt / r["ttft"], 1)}
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        time.sleep(a.gap)


def mode_warm(a):
    p, nw = pb.make_prompt(24576, 11)
    r = pb.stream(a.base, p, 4)
    pb.stream(a.base, pb.make_prompt(300, 12, "Reply with one word: ok")[0], 4)
    print(json.dumps({"mode": "warm", "ttft_s": round(r["ttft"], 3),
                      "prompt_tokens": (r["usage"] or {}).get("prompt_tokens")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["starve", "solo", "warm"])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--long", default="128k")
    ap.add_argument("--offsets", default="2,5,8,11,14,17")
    ap.add_argument("--short-tokens", type=int, default=300)
    ap.add_argument("--short-max-tokens", type=int, default=16)
    ap.add_argument("--decoder", type=int, default=1)
    ap.add_argument("--decoder-lead", type=float, default=3.0)
    ap.add_argument("--decoder-tokens", type=int, default=20000)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--gap", type=float, default=2.0)
    a = ap.parse_args()
    {"starve": mode_starve, "solo": mode_solo, "warm": mode_warm}[a.mode](a)


if __name__ == "__main__":
    main()
