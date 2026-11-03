"""Which cut points make tokenization restartable for the DSv4.1 tokenizer?

1. Exhaustive structural check of the added vocabulary (no added-token string
   overlaps another by suffix/prefix, none contains another).  Under that
   property every literal occurrence of an added-token string is matched as that
   token by the AddedVocabulary split, so any occurrence boundary is a hard
   piece boundary.
2. Randomised split test: encode(S) == encode(S[:c]) + encode(S[c:]) for every
   added-token occurrence start/end c, on adversarial strings; negative controls
   at whitespace/newline/'<tool_result>' cuts (expected to fail sometimes).
3. HF __call__ (as base.py:637 calls it) == raw backend encode.
4. End-to-end: rendered prompts of multi-step conversations in several chat
   modes, incremental (cut at last added-token literal inside the common
   prefix) == full, every step.

CPU only, synthetic data, bounded.  usage: CUDA_VISIBLE_DEVICES= nice -n 19 python boundary_check.py
"""
import json
import os
import random
import sys
import time

T0 = time.time()
DEADLINE = T0 + float(os.environ.get("DEADLINE_S", "420"))
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/dsv41/uncensored-fp8"))


def mem_ok():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            if int(line.split()[1]) < 8_000_000:
                print("ABORT: MemAvailable", line.strip(), flush=True)
                sys.exit(3)
    if time.time() > DEADLINE:
        print("ABORT: deadline", flush=True)
        sys.exit(4)


mem_ok()
from vllm.tokenizers.registry import get_tokenizer  # noqa: E402
from vllm.tokenizers import deepseek_v41_encoding as E  # noqa: E402

tok = get_tokenizer(MODEL, tokenizer_mode="deepseek_v41")
be = tok._tokenizer
added = [t.content for t in be.get_added_tokens_decoder().values()]
print("added tokens:", len(added), flush=True)

# ---------------------------------------------------------------- 1. structure
aset = set(added)
prefixes = {}
for a in added:
    for i in range(1, len(a)):
        prefixes.setdefault(a[:i], a)
overlaps = []
for a in added:
    for i in range(1, len(a)):
        suf = a[i:]
        if suf in prefixes:
            overlaps.append((a, prefixes[suf]))
contain = []
for a in added:
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if (i, j) != (0, n) and a[i:j] in aset:
                contain.append((a, a[i:j]))
print("suffix/prefix overlaps:", len(overlaps), overlaps[:5])
print("containments:", len(contain), contain[:5])
structure_ok = not overlaps and not contain

# ---------------------------------------------------------------- 2. random split test
R = random.Random(7)
SPECIALS = ["<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>", "<think>", "｜DSML｜",
            "<｜System｜>", "<｜begin▁of▁sentence｜>", "<｜deepseek_image｜>", "<｜latest_reminder｜>", "<｜action｜>"]
FRAGS = ["\n", "\n\n", " ", "  ", "\t", "<", ">", "</", "｜", "<｜", "｜>", "<｜Use", "r｜>", "<thi", "nk>", "DSML",
         " calls", " invoke", 'name="bash"', "<tool_result>", "</tool_result>", "123", "4567", "0x1f", "é", "ß",
         "的一是", "🙂", "def f(x):", "    return x", "{\"a\": 1}", "Hello", "world", ".", ",", "::", "--", "==",
         "\r\n", " \n", "\n ", "  \n\n  "]


def rand_string():
    parts = []
    for _ in range(R.randint(5, 60)):
        r = R.random()
        if r < 0.25:
            parts.append(R.choice(SPECIALS))
        elif r < 0.3:
            parts.append(R.choice(added))
        else:
            parts.append(R.choice(FRAGS))
    return "".join(parts)


def enc(s):
    return be.encode(s, add_special_tokens=False).ids


SELF_OVERLAP = {a for a, b in overlaps if a == b}  # '｜DSML｜' (shares its edge char '｜' with itself)


def occ_bounds(s):
    """cut -> True if it is a boundary of an occurrence of a non-self-overlapping added token."""
    out = {}
    for a in aset:
        j = s.find(a)
        while j >= 0:
            for c in (j, j + len(a)):
                out[c] = out.get(c, False) or (a not in SELF_OVERLAP)
            j = s.find(a, j + 1)
    return out


n_cuts = fails = 0
dsml_only = [0, 0]  # cuts that are only '｜DSML｜' occurrence boundaries: [cuts, fails]
neg = {"after_space": [0, 0], "after_newline": [0, 0], "after_blankline": [0, 0], "before_tool_result": [0, 0],
       "random_pos": [0, 0]}
hf_mismatch = 0
n_strings = 0
while time.time() - T0 < 120 and n_strings < 4000:
    s = rand_string()
    n_strings += 1
    full = enc(s)
    for c, clean in occ_bounds(s).items():
        bad = enc(s[:c]) + enc(s[c:]) != full
        if clean:
            n_cuts += 1
            fails += bad
            if bad and fails <= 3:
                print("SPLIT FAIL (non-DSML) at", c, repr(s[max(0, c - 20):c + 20]))
        else:
            dsml_only[0] += 1
            dsml_only[1] += bad
    for name, finder in (("after_space", lambda: [i + 1 for i, ch in enumerate(s) if ch == " "]),
                         ("after_newline", lambda: [i + 1 for i, ch in enumerate(s) if ch == "\n"]),
                         ("after_blankline", lambda: [i + 2 for i in range(len(s) - 1) if s[i:i + 2] == "\n\n"]),
                         ("before_tool_result", lambda: [i for i in range(len(s)) if s.startswith("<tool_result>", i)]),
                         ("random_pos", lambda: [R.randint(1, max(1, len(s) - 1))])):
        for c in finder()[:3]:
            neg[name][0] += 1
            if enc(s[:c]) + enc(s[c:]) != full:
                neg[name][1] += 1
    if n_strings % 50 == 0:
        if tok(s, truncation=True, max_length=1048577, add_special_tokens=False)["input_ids"] != full:
            hf_mismatch += 1
print(f"random strings={n_strings} added-token cuts (excl. DSML-only)={n_cuts} fails={fails}; DSML-only cuts [n, fails]={dsml_only}")
print("negative controls [cuts, fails]:", neg)
print("HF __call__ vs backend mismatches:", hf_mismatch)
mem_ok()

# ---------------------------------------------------------------- 4. end-to-end incremental over rendered prompts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CUTS = ("<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>", "<think>")  # not ｜DSML｜ (self-overlapping)


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    B = 1 << 14
    while i < n and a[i:i + B] == b[i:i + B]:
        i += B
    hi = min(i + B, n)
    while i < hi and a[i] == b[i]:
        i += 1
    return min(i, n)


def incremental(prev_s, prev_ids, s):
    L = lcp(prev_s, s)
    c = max((s.rfind(x, 0, L) + len(x) for x in CUTS if s.rfind(x, 0, L) >= 0), default=-1)
    if c <= 0:
        return None, 0
    tail = enc(prev_s[c:]) if c < len(prev_s) else []
    k = len(prev_ids) - len(tail)
    if k < 0 or prev_ids[k:] != tail:
        return None, 0
    return prev_ids[:k] + enc(s[c:]), k


def words(n):
    W = "alpha beta gamma delta the of and to in is model token cache 12 345 0x7f <tag> é 的 🙂".split()
    return " ".join(R.choice(W) for _ in range(n))


TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}}}}]


def tricky_text():
    t = words(R.randint(5, 80))
    if R.random() < 0.4:
        t = R.choice(["", "\n", " ", "<｜Use", "｜>", "<", "think>", "\n\n"]) + t + R.choice(
            ["", "\n", " ", "<｜Use", "｜>", "<", "</", "<think", R.choice(SPECIALS)])
    return t


def step(i, mode):
    cid = f"call_{i}"
    a = {"role": "assistant", "content": tricky_text() if R.random() < 0.5 else "",
         "reasoning_content": tricky_text(),
         "tool_calls": [{"id": cid, "type": "function",
                         "function": {"name": "bash", "arguments": json.dumps({"command": tricky_text()})}}]}
    out = [a, {"role": "tool", "tool_call_id": cid, "content": tricky_text()}]
    if mode == "mid_system" and R.random() < 0.2:
        out.append({"role": "system", "content": "reminder " + tricky_text()})
    if mode == "plain_chat":
        out = [{"role": "assistant", "content": tricky_text(), "reasoning_content": tricky_text()},
               {"role": "user", "content": tricky_text()}]
    return out


MODES = {  # name: (tools?, chat_template_kwargs)
    "tools_thinking": (True, {}),
    "tools_chat_effort_none": (True, {"reasoning_effort": "none"}),
    "notools_thinking_drop": (False, {}),
    "notools_thinking_keep": (False, {"drop_thinking": False}),
    "mid_system": (True, {}),
    "plain_chat": (False, {"thinking": False}),
    "history_edit": (True, {}),
}
e2e = {}
for mode, (use_tools, kw) in MODES.items():
    mem_ok()
    msgs = [{"role": "system", "content": "sys " + words(50)}, {"role": "user", "content": tricky_text()}]
    prev_s = prev_ids = None
    stats = {"steps": 0, "identical": 0, "fallback": 0, "extension": 0, "reused_frac": []}
    for i in range(40):
        msgs.extend(step(i, mode))
        if mode == "history_edit" and i % 7 == 6:
            j = R.randrange(2, len(msgs))
            msgs[j] = dict(msgs[j], content=(msgs[j].get("content") or "") + " edited")
        s = tok.apply_chat_template(msgs, tools=TOOLS if use_tools else None, tokenize=False, **kw)
        full = enc(s)
        if prev_s is not None:
            inc, k = incremental(prev_s, prev_ids, s)
            stats["steps"] += 1
            stats["extension"] += s.startswith(prev_s)
            if inc is None:
                stats["fallback"] += 1
            else:
                stats["identical"] += inc == full
                stats["reused_frac"].append(k / len(full))
        prev_s, prev_ids = s, full
    rf = stats.pop("reused_frac")
    stats["mean_reused_frac"] = round(sum(rf) / max(len(rf), 1), 3)
    e2e[mode] = stats
    print(mode, stats, flush=True)

summary = {"structure_ok": structure_ok, "overlaps": len(overlaps), "containments": len(contain),
           "random_strings": n_strings, "added_token_cuts": n_cuts, "added_token_cut_failures": fails, "dsml_only_cuts_and_failures": dsml_only,
           "negative_controls": neg, "hf_call_vs_backend_mismatch": hf_mismatch, "e2e": e2e,
           "elapsed_s": round(time.time() - T0, 1)}
json.dump(summary, open("boundary_check.json", "w"), indent=1)
print(json.dumps(summary))
