#!/usr/bin/env python3
"""ttft_ab.py --out FILE --label L [--start-tokens 230000] [--steps 56] [--max-tokens 8]: first-token wait on a synthetic,
growing agent conversation. ON spark-one, stdlib only, talks to http://127.0.0.1:8000. One request at a time, idle check
before each. Synthetic data only (synth.py, fixed seed): the same conversation byte for byte in every arm.

Step 0 builds the conversation to about --start-tokens (estimated from chars; the cold prefill of step 0 is slow, about
40 s at 230K). Each later step appends one synthetic assistant step + its tool results (never the model's own output, so
both arms send identical requests) and sends: stream, temperature 0, max_tokens, tools, return_token_ids,
stream_options.include_usage. Per step it records ttft_s (request sent -> first SSE data line, which carries the
prompt_token_ids in this server), e2e_s, prompt_tokens, cached_tokens, sha256 of the returned prompt_token_ids
(ids_sha: compare across arms = exactness of the server's prompt ids) and of the output text (out_sha).

Needs about 56 steps: with DSV41_INCR_TOK=1 the first 32 cache hits of a process are verified by a full tokenize (no
saving), so only later steps show the saving. ab_compare.py labels steps from the overlay's per-request log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BASE = "http://127.0.0.1:8000"


def busy() -> float:
    t = urllib.request.urlopen(BASE + "/metrics", timeout=5).read().decode()
    return sum(float(line.split()[-1]) for line in t.splitlines()
               if line.startswith("vllm:num_requests_running{") or line.startswith("vllm:num_requests_waiting{"))


def wait_idle(limit_s: int = 600) -> float:
    t0 = time.time()
    while busy() > 0:
        if time.time() - t0 > limit_s:
            raise SystemExit("ABORT: server not idle for 10 min (foreign traffic?)")
        time.sleep(1)
    return time.time() - t0


def send(body: dict, timeout: float) -> dict:
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    out = []
    ids = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ch = json.loads(data)
            if ch.get("prompt_token_ids") is not None and ids is None:
                ids = ch["prompt_token_ids"]
            if ch.get("usage"):
                usage = ch["usage"]
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                for k in ("reasoning_content", "reasoning", "content"):
                    if d.get(k):
                        out.append(f"{k}:{d[k]}")
    e2e = time.perf_counter() - t0
    return {"ttft_s": round(ttft or -1, 4), "e2e_s": round(e2e, 4),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "n_ids": len(ids) if ids is not None else None,
            "ids_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16] if ids is not None else None,
            "out_sha": hashlib.sha256("\x00".join(out).encode()).hexdigest()[:16], "out_chars": sum(len(x) for x in out)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--start-tokens", type=int, default=230000)
    ap.add_argument("--steps", type=int, default=56)
    ap.add_argument("--max-tokens", type=int, default=8)
    a = ap.parse_args()
    from synth import SYSTEM, TOOLS, USER0, make_step  # deterministic: module-level Random(20261103)

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER0}]
    while sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) / 3.25 < a.start_tokens:
        messages.extend(make_step())
    with open(a.out, "a") as f:
        for step in range(a.steps):
            if step:
                messages.extend(make_step())
            waited = wait_idle()
            body = {"model": "dsv41", "messages": messages, "tools": TOOLS, "stream": True,
                    "stream_options": {"include_usage": True}, "max_tokens": a.max_tokens, "temperature": 0,
                    "return_token_ids": True}
            rec = {"label": a.label, "step": step, "messages": len(messages), "idle_wait_s": round(waited, 1),
                   **send(body, timeout=1800 if step == 0 else 600)}
            rec["foreign_after"] = busy()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)
            if rec["ids_sha"] is None:
                print("ABORT: server returned no prompt_token_ids (return_token_ids unsupported?)", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
