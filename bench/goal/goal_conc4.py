#!/usr/bin/env python3
"""goal_conc4.py -- G4 goal metric (stdlib only): 4 concurrent individual sessions on the dsv41 server.

G4 = 4 DIFFERENT fixed prompts (code, prose, math = the G1/bench.py prompts; mixed = one fixed RSA
prose+math+code prompt), started together behind a barrier, 512 max tokens each, temperature 0, top_p 1.0,
thinking at the server default (ON). One warm-up batch (excluded: it puts all 4 prompts in the prefix cache, so
every measured batch starts its 4 decodes together), then --runs measured batches (default 3).
Per session: decode tok/s = (completion_tokens - 1) / (last content chunk - first content chunk), as bench.py.
Also the common-window rate: tokens each session produced while ALL 4 were decoding (continuous usage stats give
the exact completion-token count on every chunk) divided by that window.
Per run: min and mean of the per-session rates (target: 50 tok/s per session). Summary: mean over runs of the
per-run min and mean, the same for the common-window rates, per-session AL is not separable (one shared
/metrics counter), so the batch AL is reported.
Output text sha256 per session is recorded (informational: batched greedy output can depend on batch timing).

The prompts are fixed: their canonical JSON must hash to PROMPTS_SHA256 or the script refuses to run.
Usage: python3 goal_conc4.py --out runs/x/g4.jsonl --label x [--runs 3] [--url http://127.0.0.1:8000]
"""
import argparse, hashlib, json, sys, threading, time, urllib.request

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
    "mixed": ("Explain how the RSA public-key cryptosystem works. First describe in plain prose where it came from and "
              "why it matters. Then derive the mathematics of key generation, encryption and decryption, including "
              "Euler's theorem and a proof that decryption recovers the message. Work a complete numeric example by "
              "hand with p = 61, q = 53 and e = 17, showing every modular computation. Finally write a short, "
              "well-commented Python implementation with a unittest that encrypts and decrypts a sample message."),
}
SESSIONS = ("code", "prose", "math", "mixed")
PROMPTS_SHA256 = "42c93c19844208cfba09d51266b32ee9af339362eeba6a6266a3a6dbd41cf2b3"  # sha256 of json.dumps(PROMPTS, sort_keys=True)
MAX_TOKENS = 512


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
    return {"drafts": int(d), "accepted": int(acc), "AL": round(1 + acc / d, 3)}


def stream_chat(base, model, prompt, max_tokens, barrier, timeout=1800):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "top_p": 1.0, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    barrier.wait()
    t0 = time.time(); trace = []; usage = None; finish = None; reasoning, content = [], []
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
            got = False
            for ch in ev.get("choices") or []:
                dl = ch.get("delta") or {}
                c = dl.get("content") or ""
                rc = dl.get("reasoning_content") or dl.get("reasoning") or ""
                if c or rc:
                    got = True; content.append(c); reasoning.append(rc)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if got:
                trace.append((time.time(), (ev.get("usage") or {}).get("completion_tokens")))
    t_end = time.time()
    ct = (usage or {}).get("completion_tokens")
    text = "".join(reasoning) + "\x00" + "".join(content)
    res = {"t0": t0, "t_end": t_end, "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": ct,
           "finish": finish, "chunks": len(trace), "sha256": hashlib.sha256(text.encode()).hexdigest(),
           "ttft_s": round(trace[0][0] - t0, 4) if trace else None, "trace": trace}
    if trace and ct and len(trace) > 1 and trace[-1][0] > trace[0][0]:
        res["first"], res["last"] = trace[0][0], trace[-1][0]
        res["decode_tok_s"] = round((ct - 1) / (trace[-1][0] - trace[0][0]), 3)
    return res


def tokens_at(trace, t):
    """completion tokens delivered by wall time t (continuous usage); None if the counts are missing."""
    n = 0
    for ts, c in trace:
        if ts > t:
            break
        if c is None:
            return None
        n = c
    return n


def run_batch(a, base):
    barrier = threading.Barrier(len(SESSIONS))
    res = [None] * len(SESSIONS)
    def one(i):
        try:
            res[i] = stream_chat(base, a.model, PROMPTS[SESSIONS[i]], MAX_TOKENS, barrier)
        except Exception as e:  # keep the other sessions' rows
            res[i] = {"error": repr(e)}
        res[i]["session"] = SESSIONS[i]
    m0 = scrape(base); t0 = time.time()
    th = [threading.Thread(target=one, args=(i,)) for i in range(len(SESSIONS))]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0; sd = spec_delta(m0, scrape(base))
    per = [r.get("decode_tok_s") for r in res]
    ok = all(p for p in per)
    out = {"wall_s": round(wall, 3), "spec": sd, "sessions": []}
    if ok:
        w0 = max(r["first"] for r in res); w1 = min(r["last"] for r in res)
        cw = []
        for r in res:
            n0, n1 = tokens_at(r["trace"], w0), tokens_at(r["trace"], w1)
            cw.append(round((n1 - n0) / (w1 - w0), 3) if (n0 is not None and n1 is not None and w1 > w0) else None)
        out.update({"per_session_min": min(per), "per_session_mean": round(sum(per) / len(per), 3),
                    "common_window_s": round(w1 - w0, 3), "cw_tok_s": cw,
                    "cw_min": min(cw) if all(c is not None for c in cw) else None,
                    "cw_mean": round(sum(cw) / len(cw), 3) if all(c is not None for c in cw) else None,
                    "start_skew_s": round(w0 - min(r["first"] for r in res), 3),
                    "agg_tok_s": round(sum(r["completion_tokens"] for r in res) / wall, 2)})
    for r in res:
        out["sessions"].append({k: v for k, v in r.items() if k not in ("trace",)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--print-sha", action="store_true")
    a = ap.parse_args()
    sha = prompts_sha()
    if a.print_sha:
        print(sha); return
    if sha != PROMPTS_SHA256:
        sys.exit(f"goal_conc4: prompt set changed (sha256 {sha} != pinned {PROMPTS_SHA256}); refusing")
    base = a.url.rstrip("/")
    def emit(row):
        row.update({"label": a.label, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        s = json.dumps(row); print(s, flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(s + "\n")
    t_start = time.time()
    if not a.no_warmup:
        w = run_batch(a, base); w.update({"metric": "G4_warmup"}); emit(w)
    runs = []
    for k in range(a.runs):
        b = run_batch(a, base); b.update({"metric": "G4", "run": k}); runs.append(b); emit(b)
    good = [b for b in runs if b.get("per_session_min")]
    f = lambda key: round(sum(b[key] for b in good) / len(good), 3) if good and all(b.get(key) for b in good) else None
    by_sess = {}
    for i, s in enumerate(SESSIONS):
        xs = [b["sessions"][i].get("decode_tok_s") for b in good]
        by_sess[s] = {"tok_s": round(sum(xs) / len(xs), 3) if xs and all(xs) else None,
                      "tokens": [b["sessions"][i].get("completion_tokens") for b in good],
                      "sha8": sorted({b["sessions"][i].get("sha256", "")[:8] for b in good})}
    emit({"metric": "G4_summary", "prompts_sha256": sha, "runs_ok": len(good), "runs": a.runs,
          "G4_per_session_min": f("per_session_min"), "G4_per_session_mean": f("per_session_mean"),
          "G4_cw_min": f("cw_min"), "G4_cw_mean": f("cw_mean"), "agg_tok_s": f("agg_tok_s"),
          "per_run_min": [b.get("per_session_min") for b in runs], "per_run_mean": [b.get("per_session_mean") for b in runs],
          "batch_AL": [b["spec"].get("AL") for b in runs], "by_session": by_sess,
          "elapsed_s": round(time.time() - t_start, 1), "t_start": round(t_start, 3), "t_end": round(time.time(), 3)})


if __name__ == "__main__":
    main()
