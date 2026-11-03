"""Stage profile of the DSv4.1 /v1/chat/completions prompt path, CPU only.

Replays, in-process and in the same order, the calls the API server makes from
the HTTP body to the EngineCoreRequest (see PROFILE.md for file:line), on
synthetic agent conversations at ~50K/150K/300K/500K tokens that grow by one
assistant step + its tool result(s) per request.  Also times a prototype of
incremental tokenization (cut at the last added-token literal inside the common
prefix) and checks it is bit-identical to the full tokenization on every
request.

Safety: run with CUDA_VISIBLE_DEVICES= (no GPU context), nice -n 19, bounded by
a wall deadline; aborts if MemAvailable < 8,000,000 kB.  Synthetic data only.

usage: CUDA_VISIBLE_DEVICES= nice -n 19 python profile_stages.py [out.json]
env:   SIZES=50000,150000,300000,500000  REPS=5  STEPS=4  DEADLINE_S=540
"""
import copy
import gc
import json
import os
import random
import statistics
import sys
import time

T_START = time.time()
DEADLINE = T_START + float(os.environ.get("DEADLINE_S", "540"))
MEM_FLOOR_KB = 8_000_000
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/dsv41/uncensored-fp8"))
SIZES = [int(x) for x in os.environ.get("SIZES", "50000,150000,300000,500000").split(",")]
REPS = int(os.environ.get("REPS", "5"))
STEPS = int(os.environ.get("STEPS", "4"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "profile_out.json"


def mem_available_kb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 0


def guard(where: str) -> None:
    m = mem_available_kb()
    if m < MEM_FLOOR_KB:
        print(f"ABORT at {where}: MemAvailable {m} kB < {MEM_FLOOR_KB}", flush=True)
        sys.exit(3)
    if time.time() > DEADLINE:
        print(f"ABORT at {where}: wall deadline reached", flush=True)
        sys.exit(4)


guard("start")

# --------------------------------------------------------------------------- server objects
from vllm.config import ModelConfig  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
from vllm.parser.parser_manager import ParserManager  # noqa: E402
from vllm.renderers.chat_utils import parse_chat_messages  # noqa: E402
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt  # noqa: E402
from vllm.tokenizers import deepseek_v41 as V41  # noqa: E402
from vllm.tokenizers import deepseek_v41_encoding as E  # noqa: E402
from vllm.tokenizers.registry import get_tokenizer  # noqa: E402
from vllm.renderers import merge_kwargs  # noqa: E402
from vllm.inputs import tokens_input  # noqa: E402

tok = get_tokenizer(MODEL, tokenizer_mode="deepseek_v41")  # == cached_tokenizer_from_config
mc = ModelConfig(model=MODEL, tokenizer_mode="deepseek_v41", max_model_len=1048576,
                 limit_mm_per_prompt={"image": 16})
ParserCls = ParserManager.get_parser(tool_parser_name="deepseek_v41", reasoning_parser_name="deepseek_v41",
                                     enable_auto_tools=True, model_name=MODEL, tokenizer=tok)
backend = tok._tokenizer  # tokenizers.Tokenizer (Rust)
print("objects ready", round(time.time() - T_START, 1), "s", flush=True)

# --------------------------------------------------------------------------- synthetic data (synth.py)
from synth import R, TOOLS, SYSTEM, USER0, make_step, body_for  # noqa: E402


# --------------------------------------------------------------------------- the server path
def T():
    return time.perf_counter()


def run_full(body: bytes, timings: dict | None) -> tuple[str, list[int], object]:
    """Replays serving.py -> online_renderer.py -> renderers -> tokenizer, stage by stage."""
    t0 = T()
    obj = json.loads(body)                                              # FastAPI: await request.json()
    req = ChatCompletionRequest.model_validate(obj)                     # FastAPI: pydantic body model
    t1 = T()
    ctk = req.build_chat_params(None, "auto").with_defaults({}).chat_template_kwargs   # serving.py:192
    parser = ParserCls(tok, req.tools, chat_template_kwargs=ctk, model_config=mc)       # serving.py:307
    tool_dicts = None if req.tools is None else [t.model_dump() for t in req.tools]     # online_renderer.py:287
    dtk = merge_kwargs({}, dict(tools=tool_dicts, tokenize=False))                      # online_renderer.py:773
    tok_params = req.build_tok_params(mc)                                               # online_renderer.py:784
    chat_params = req.build_chat_params(None, "auto").with_defaults(
        dtk, default_media_io_kwargs=(mc.multimodal_config.media_io_kwargs if mc.multimodal_config else None),
        default_mm_processor_kwargs=getattr(req, "mm_processor_kwargs", None))           # online_renderer.py:785
    t2 = T()
    conversation, mm_data, mm_uuids = parse_chat_messages(                              # deepseek_v4.py:74
        req.messages, mc, content_format="openai",
        media_io_kwargs=chat_params.media_io_kwargs, mm_processor_kwargs=chat_params.mm_processor_kwargs)
    t3 = T()
    prompt_raw = tok.apply_chat_template(conversation=conversation, messages=req.messages,   # deepseek_v4.py:82
                                         **chat_params.get_apply_chat_template_kwargs())
    t4 = T()
    prompt = parse_dec_only_prompt(prompt_raw)                                          # deepseek_v4.py:88
    prompt = tok_params.apply_pre_tokenization(tok, prompt)                              # base.py:725
    enc = tok(prompt["prompt"], **tok_params.get_encode_kwargs())                        # base.py:637
    ids = list(enc["input_ids"])                                                         # base.py:604
    t5 = T()
    tp = {"prompt_token_ids": ids, "prompt": prompt["prompt"]}
    tp = tok_params.apply_post_tokenization(tok, tp)                                     # base.py:736
    ei = tokens_input(tp["prompt_token_ids"])                                            # base.py:1017
    ei["prompt"] = tp["prompt"]
    t6 = T()
    parser2 = ParserCls(tok, req.tools, model_config=mc, chat_template_kwargs=chat_params.chat_template_kwargs)
    req = parser2.adjust_request(request=req)                                            # online_renderer.py:847
    t7 = T()
    req.resolve_reasoning_ended(parser, ids)                                             # serving.py:250
    max(ids), min(ids)                                                                   # base.py:420 via input_processor.py:646
    t8 = T()
    if timings is not None:
        for k, v in (("http_json_pydantic", t1 - t0), ("params_parser_tools", t2 - t1), ("parse_chat_messages", t3 - t2),
                     ("render_apply_chat_template", t4 - t3), ("tokenize", t5 - t4), ("post_tok_tokens_input", t6 - t5),
                     ("adjust_request", t7 - t6), ("reasoning_end_validate_ids", t8 - t7)):
            timings.setdefault(k, []).append(v * 1e3)
    return prompt["prompt"], ids, (req, chat_params, tok_params, mm_data)


def render_breakdown(req, chat_params, timings: dict) -> None:
    """Splits deepseek_v41.py apply_chat_template into its parts (same inputs)."""
    kw = chat_params.get_apply_chat_template_kwargs()
    t0 = T()
    conv = V41._normalize_messages(req.messages)                       # deepseek_v41.py:62 (deepcopy)
    if kw.get("tools"):
        sysm = next((m for m in conv if m["role"] == "system"), None)
        sysm["tools"] = kw["tools"]
    t1 = T()
    merged = E.merge_tool_messages(conv)                                # encoding.py:545 (deepcopy per msg)
    t2 = T()
    merged = E.sort_tool_results_by_call_order(merged)                  # encoding.py:546
    t3 = T()
    eff_drop = not any(m.get("tools") for m in merged)
    p = E.bos_token
    for i in range(len(merged)):                                        # encoding.py:568
        p += E.render_message(i, merged, thinking_mode="thinking", drop_thinking=eff_drop, reasoning_effort="high")
    t4 = T()
    for k, v in (("r_normalize_deepcopy", t1 - t0), ("r_merge_tool_messages", t2 - t1),
                 ("r_sort_tool_results", t3 - t2), ("r_render_loop", t4 - t3)):
        timings.setdefault(k, []).append(v * 1e3)


def engine_side(ids: list[int], timings: dict) -> None:
    """What the same ids cost after the renderer (API-server and EngineCore side)."""
    import tokenizers
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
    t0 = T()
    ds = tokenizers.decoders.DecodeStream(ids=ids, skip_special_tokens=True)   # detokenizer.py:195
    t1 = T()
    enc = MsgpackEncoder()
    bufs = enc.encode(ids)                                                     # EngineCoreRequest.prompt_token_ids
    t2 = T()
    MsgpackDecoder(list[int]).decode(bufs)                                     # EngineCore side
    t3 = T()
    import hashlib
    import pickle
    h = b"\0" * 32
    for i in range(0, len(ids) - 63, 64):                                      # kv_cache_utils.hash_block_tokens, sha256
        h = hashlib.sha256(pickle.dumps((h, tuple(ids[i:i + 64]), None), protocol=pickle.HIGHEST_PROTOCOL)).digest()
    t4 = T()
    del ds
    for k, v in (("x_decodestream_prime", t1 - t0), ("x_msgpack_encode", t2 - t1), ("x_msgpack_decode", t3 - t2),
                 ("x_block_hash_sha256_enginecore", t4 - t3)):
        timings.setdefault(k, []).append(v * 1e3)


# --------------------------------------------------------------------------- incremental prototype
# not "｜DSML｜": it overlaps itself ("｜DSML｜DSML｜"), so a literal occurrence is not always a match
CUT_LITERALS = ("<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>", "<think>")


def lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    if b.startswith(a[:n]) if len(a) > len(b) else b.startswith(a):
        return n
    lo, B = 0, 1 << 16
    while lo < n and a[lo:lo + B] == b[lo:lo + B]:
        lo += B
    hi = min(lo + B, n)
    while lo < hi:                       # first mismatch in [lo, hi)
        mid = (lo + hi) // 2
        if a[lo:mid + 1] == b[lo:mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return lo


def enc_raw(s: str) -> list[int]:
    return backend.encode(s, add_special_tokens=False).ids


def incremental(prev_s: str, prev_ids: list[int], s: str):
    L = lcp(prev_s, s)
    c = -1
    for lit in CUT_LITERALS:
        j = s.rfind(lit, 0, L)
        if j >= 0:
            c = max(c, j + len(lit))
    if c <= 0:
        return None, {"L": L, "c": c}
    tail_prev = enc_raw(prev_s[c:]) if c < len(prev_s) else []
    k = len(prev_ids) - len(tail_prev)
    if k < 0 or prev_ids[k:] != tail_prev:
        return None, {"L": L, "c": c, "selfcheck": False}
    suffix = enc_raw(s[c:])
    return prev_ids[:k] + suffix, {"L": L, "c": c, "reused": k, "new": len(suffix), "prev_tail": len(tail_prev)}


# --------------------------------------------------------------------------- driver
def med(xs):
    return round(statistics.median(xs), 2)


results = {"host": os.uname().nodename, "sizes": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER0}]
est_chars_per_tok = 3.25
for target in SIZES:
    guard(f"size {target}")
    # grow (cheap estimate by chars, then confirm with one real tokenize)
    while True:
        chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
        if chars / est_chars_per_tok >= target * 0.99:
            _, ids0, _ = run_full(body_for(messages), None)
            if len(ids0) >= target * 0.97:
                break
            est_chars_per_tok = chars / max(len(ids0), 1)
            continue
        messages.extend(make_step())
    row = {"target": target}
    prev_s, prev_ids, _ = run_full(body_for(messages), None)
    row["start_tokens"] = len(prev_ids)
    row["start_messages"] = len(messages)
    steps = []
    tim: dict = {}
    for step in range(STEPS):
        guard(f"size {target} step {step}")
        messages.extend(make_step())
        body = body_for(messages)
        gc.collect()
        for _ in range(REPS):
            s, ids, ctx = run_full(body, tim)
        render_breakdown(ctx[0], ctx[1], tim)
        engine_side(ids, tim)
        # raw backend encode vs the HF __call__ wrapper on the same string
        t0 = T(); enc_raw(s); tim.setdefault("tok_raw_backend_encode", []).append((T() - t0) * 1e3)
        # incremental prototype
        t0 = T()
        inc, info = incremental(prev_s, prev_ids, s)
        dt = (T() - t0) * 1e3
        same = inc == ids
        tim.setdefault("incr_tokenize_total", []).append(dt)
        steps.append({"tokens": len(ids), "messages": len(messages), "body_bytes": len(body), "prompt_chars": len(s),
                      "incr_ms": round(dt, 2), "incr_identical": same, **info})
        print(f"[{target}] step {step}: tokens={len(ids)} msgs={len(messages)} body={len(body) / 1e6:.2f}MB "
              f"incr={dt:.1f}ms identical={same} info={info}", flush=True)
        prev_s, prev_ids = s, ids
    row["steps"] = steps
    if os.environ.get("DUMP_PROMPT"):
        with open(f"prompt_{target}.txt", "w") as f:
            f.write(prev_s)
    row["ms_median"] = {k: med(v) for k, v in tim.items()}
    row["ms_min"] = {k: round(min(v), 2) for k, v in tim.items()}
    results["sizes"].append(row)
    print(json.dumps(row["ms_median"]), flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)

results["elapsed_s"] = round(time.time() - T_START, 1)
results["mem_available_kb_end"] = mem_available_kb()
with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print("done", results["elapsed_s"], "s", flush=True)
