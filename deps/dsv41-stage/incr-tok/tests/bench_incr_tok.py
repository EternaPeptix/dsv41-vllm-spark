"""incr-tok bench on a Spark CPU: ms saved per request at ~150K / 300K / 500K tokens, through the real DeepseekV4Renderer.

For each size a synthetic agent conversation (synth.py: tools + thinking, one assistant step + tool results per request)
is grown to the target, one request primes the cache, then STEPS requests each add one step. Per request:
  stock_ms   BaseRenderer._tokenize_prompt (the HF __call__ at base.py:638), median of REPS runs
  hit_ms     the overlay's cache hit (verification off), checked identical to stock (whole TokensPrompt dict)
  verify_ms  a verified hit (full tokenize + incremental + compare), what the first 32 hits and 1 in 64 cost
  miss_ms    a miss on an empty cache (lookup + full tokenize + store), i.e. the cost on a new conversation
The overlay is attached exactly as the patched DeepseekV4Renderer.__init__ does (attach(renderer)).

usage: cd ~/dsv41-stage/incr-tok && CUDA_VISIBLE_DEVICES= nice -n 19 timeout 600 ~/venvs/vllm032/bin/python tests/bench_incr_tok.py
env: SIZES=150000,300000,500000 STEPS=4 REPS=3 DEADLINE_S=540 OUT=results/bench_incr_tok.json
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

T0 = time.time()
DEADLINE = T0 + float(os.environ.get("DEADLINE_S", "540"))
KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/dsv41/uncensored-fp8"))
SIZES = [int(x) for x in os.environ.get("SIZES", "150000,300000,500000").split(",")]
STEPS = int(os.environ.get("STEPS", "4"))
REPS = int(os.environ.get("REPS", "3"))
OUT = os.environ.get("OUT", str(KIT / "results/bench_incr_tok.json"))


def mem_kb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return 0


def guard(where: str) -> None:
    if mem_kb() < 8_000_000:
        print(f"ABORT at {where}: MemAvailable {mem_kb()} kB", flush=True)
        sys.exit(3)
    if time.time() > DEADLINE:
        print(f"ABORT at {where}: deadline", flush=True)
        sys.exit(4)


guard("start")
from vllm.config import ModelConfig, VllmConfig  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
from vllm.renderers import merge_kwargs  # noqa: E402
from vllm.renderers.base import BaseRenderer  # noqa: E402
from vllm.renderers.registry import renderer_from_config  # noqa: E402

spec = importlib.util.spec_from_file_location("vllm.renderers.dsv41_incr_tok", KIT / "site/vllm/renderers/dsv41_incr_tok.py")
OVL = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = OVL
spec.loader.exec_module(OVL)

mc = ModelConfig(model=MODEL, tokenizer_mode="deepseek_v41", max_model_len=1048576, limit_mm_per_prompt={"image": 16})
r = renderer_from_config(VllmConfig(model_config=mc))
TOK = r.get_tokenizer()
os.environ.update(DSV41_INCR_TOK="1", DSV41_INCR_TOK_VERIFY_FIRST="0", DSV41_INCR_TOK_VERIFY_EVERY="0")
inc = OVL.attach(r)
assert inc is not None and r._tokenize_prompt is inc, "attach failed"

from synth import SYSTEM, USER0, body_for, make_step  # noqa: E402


def render(messages):
    req = ChatCompletionRequest.model_validate(json.loads(body_for(messages)))
    tool_dicts = None if req.tools is None else [t.model_dump() for t in req.tools]
    chat_params = req.build_chat_params(None, "auto").with_defaults(
        merge_kwargs({}, dict(tools=tool_dicts, tokenize=False)),
        default_media_io_kwargs=(mc.multimodal_config.media_io_kwargs if mc.multimodal_config else None),
        default_mm_processor_kwargs=getattr(req, "mm_processor_kwargs", None))
    tok_params = req.build_tok_params(mc)
    _, prompt = r.render_messages(req.messages, chat_params)
    return tok_params.apply_pre_tokenization(TOK, prompt), tok_params


def ms(f, *a):
    t = time.perf_counter()
    out = f(*a)
    return (time.perf_counter() - t) * 1e3, out


def stock(prompt, params):
    return BaseRenderer._tokenize_prompt(r, dict(prompt), params)


results = {"host": os.uname().nodename, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "sizes": []}
messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER0}]
cpt = 3.25
for target in SIZES:
    guard(f"grow {target}")
    while True:  # grow by chars, confirm with one real tokenize
        chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
        if chars / cpt >= target * 0.99:
            p, prm = render(messages)
            n = len(stock(p, prm)["prompt_token_ids"])
            if n >= target * 0.97:
                break
            cpt = chars / n
            continue
        messages.extend(make_step())
    p, prm = render(messages)
    inc(dict(p), prm)  # prime (miss -> stored)
    rows = []
    for step in range(STEPS):
        guard(f"{target} step {step}")
        messages.extend(make_step())
        p, prm = render(messages)
        gc.collect()
        st = [ms(stock, p, prm) for _ in range(REPS)]
        want = st[0][1]
        t_hit, got = ms(inc, dict(p), prm)
        rec_inc = dict(inc.c)
        # verified hit cost: same request again with forced verification (entry for p now cached; tail check + suffix)
        inc.cfg.verify_first = 10**9
        t_ver, got_v = ms(inc, dict(p), prm)
        inc.cfg.verify_first = 0
        rows.append({"tokens": len(want["prompt_token_ids"]), "messages": len(messages), "chars": len(p["prompt"]),
                     "stock_ms": round(statistics.median(x[0] for x in st), 2), "hit_ms": round(t_hit, 2),
                     "verify_ms": round(t_ver, 2), "identical": got == want and got_v == want})
        print(f"[{target}] {rows[-1]}", flush=True)
    # miss on an empty cache: lookup + full tokenize + store
    with inc.lock:
        saved = (inc.entries.copy(), inc.bytes)
        inc.entries.clear()
        inc.bytes = 0
    t_miss, got_m = ms(inc, dict(p), prm)
    with inc.lock:
        entry_mb = round(next(reversed(inc.entries.values())).nbytes / 2**20, 2) if inc.entries else None
        inc.entries, inc.bytes = saved
    row = {"target": target, "steps": rows,
           "stock_ms": round(statistics.median(x["stock_ms"] for x in rows), 1),
           "hit_ms": round(statistics.median(x["hit_ms"] for x in rows), 1),
           "verify_ms": round(statistics.median(x["verify_ms"] for x in rows), 1),
           "miss_ms": round(t_miss, 1), "entry_mb": entry_mb, "all_identical": all(x["identical"] for x in rows) and got_m == want,
           "tokens": rows[-1]["tokens"]}
    row["saved_ms"] = round(row["stock_ms"] - row["hit_ms"], 1)
    row["miss_overhead_ms"] = round(row["miss_ms"] - rows[-1]["stock_ms"], 1)
    results["sizes"].append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "steps"}), flush=True)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(results, indent=1))
results["cache_stats"] = inc.stats()
results["elapsed_s"] = round(time.time() - T0, 1)
results["mem_available_kb_end"] = mem_kb()
Path(OUT).write_text(json.dumps(results, indent=1))
ok = all(s["all_identical"] for s in results["sizes"])
print(f"BENCH {'PASS' if ok else 'FAIL'} " + " ".join(f"{s['tokens']}tok saved {s['saved_ms']}ms" for s in results["sizes"]), flush=True)
sys.exit(0 if ok else 1)
