"""exact.py <out.json> [label] — determinism probe: 6 fixed benign prompts, serial, temperature 0,
max_tokens 256; records sha256 of reasoning+content per prompt (and a combined digest)."""
import hashlib, json, sys, time, urllib.request

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "What is 17 * 23 + 144 / 12? Show your steps.",
    "Explain in two paragraphs why the sky appears blue.",
    "List five prime numbers greater than 100 and explain how you checked each one.",
    "Write a short haiku about autumn leaves, then explain its imagery.",
    "Implement binary search in C and describe its time complexity.",
]
out = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else ""
rows = []
for i, p in enumerate(PROMPTS):
    body = {"model": "dsv41", "temperature": 0, "max_tokens": 256,
            "messages": [{"role": "user", "content": p}]}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    m = r["choices"][0]["message"]
    text = (m.get("reasoning_content") or m.get("reasoning") or "") + "\x00" + (m.get("content") or "")
    rows.append({"text": text, "i": i, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                 "tokens": r["usage"]["completion_tokens"], "s": round(time.time() - t, 2)})
combined = hashlib.sha256("".join(x["sha256"] for x in rows).encode()).hexdigest()
res = {"label": label, "combined": combined, "rows": rows}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({"label": label, "combined": combined[:16], "per": [x["sha256"][:8] for x in rows]}))
