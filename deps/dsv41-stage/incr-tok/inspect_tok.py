"""Inspect the DSv4.1 tokenizer.json: normalizer, pre-tokenizer, post-processor,
added tokens (flags) and which encoding-module literals are single added tokens.
CPU only, reads files only. Run with the vllm032 venv python on spark-two."""
import json
import os
import sys

MODEL = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/models/dsv41/uncensored-fp8")
tj = json.load(open(os.path.join(MODEL, "tokenizer.json")))
print("normalizer:", json.dumps(tj.get("normalizer"))[:400])
print("pre_tokenizer:", json.dumps(tj.get("pre_tokenizer"))[:1500])
print("post_processor:", json.dumps(tj.get("post_processor"))[:400])
print("decoder:", json.dumps(tj.get("decoder"))[:200])
m = tj["model"]
print("model:", m.get("type"), "vocab", len(m.get("vocab", {})), "merges", len(m.get("merges", [])),
      "byte_fallback", m.get("byte_fallback"), "ignore_merges", m.get("ignore_merges"),
      "dropout", m.get("dropout"))
at = tj["added_tokens"]
print("added_tokens:", len(at))
flags = {}
for t in at:
    k = (t["special"], t["normalized"], t["lstrip"], t["rstrip"], t["single_word"])
    flags[k] = flags.get(k, 0) + 1
print("flag combos (special, normalized, lstrip, rstrip, single_word): count")
for k, v in sorted(flags.items(), key=lambda kv: -kv[1]):
    print("  ", k, v)
maxlen = max(len(t["content"]) for t in at)
print("max added token length (chars):", maxlen)
contents = {t["content"]: t for t in at}

sys.path.insert(0, "")
from vllm.tokenizers import deepseek_v41_encoding as E  # noqa: E402
lits = [E.bos_token, E.eos_token, E.thinking_start_token, E.thinking_end_token,
        E.dsml_token, E.USER_SP_TOKEN, E.ASSISTANT_SP_TOKEN, E.SYSTEM_SP_TOKEN,
        E.LATEST_REMINDER_SP_TOKEN, E.IMAGE_PLACEHOLDER, "<tool_result>", "</tool_result>",
        "<｜DSML｜", "</｜DSML｜"] + list(E.DS_TASK_SP_TOKENS.values())
for s in lits:
    t = contents.get(s)
    print(f"  {s!r:40} added={t is not None}",
          "" if t is None else f"id={t['id']} special={t['special']} norm={t['normalized']} l/r={t['lstrip']}/{t['rstrip']}")
# Added tokens that are a prefix of / contain other added tokens (longest-match relevance)
pref = [(a, b) for a in contents for b in contents if a != b and b.startswith(a)]
print("added-token prefix pairs (a is prefix of b):", len(pref), pref[:10])
