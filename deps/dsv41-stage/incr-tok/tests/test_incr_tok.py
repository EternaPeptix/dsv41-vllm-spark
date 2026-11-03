"""incr-tok tests on a Spark CPU with the real DSv4.1 tokenizer and the real DeepseekV4Renderer. Never touches the live site.

  1. installer: dry-run / install / status / uninstall against a temp copy of the site file; refuses a moved base sha;
     flag-off AST difference is exactly the import, the env read and one `if` block.
  2. flag off: the patched renderer is stock (no instance override, overlay never imported).
  3. load checks: pinned tokenizer.json sha, structure digest and bad env values each keep the renderer stock.
  4. equality fuzz: thousands of synthetic conversations (7 chat modes, tools, reasoning, unicode, code, literal and
     partial special tokens), 4 interleaved at a time sharing one system prompt and tool list, with growth, middle edits,
     system edits, rollbacks, resends, compaction and mode switches. Every request goes server-style through pydantic,
     the renderer and the cache with verification OFF (so cache hits are really served), and is compared with the stock
     `_tokenize_prompt` result (whole TokensPrompt dict). Edits before the reuse floor must miss.
  5. concurrency: 8 conversations on 4 threads through one cache.
  6. eviction: 1 MB / 4-entry cache under 10 interleaved conversations; bounds hold after every request.
  7. string-level fuzz: random strings with all added tokens, random edits / truncations / extensions.
  8. fallbacks: images (multi_modal_data / uuids), truncate_prompt_tokens, add_special_tokens, the truncation max_length
     edge (n = max_length - 1 / = max_length / > max_length), short prompts; sync and async base paths use the cache.
  9. guard: a forced wrong id is caught by the first-N verification and by the 1-in-K check, the full result is served,
     the cache disables itself for the process and later requests run stock; audit mode always serves stock.

usage: cd ~/dsv41-stage/incr-tok && CUDA_VISIBLE_DEVICES= nice -n 19 timeout 600 ~/venvs/vllm032/bin/python tests/test_incr_tok.py
env: N_CONV (2000)  DEADLINE_S (540)  OUT (results/test_incr_tok.json)
Exit 0 = all pass. Aborts (exit 3) if MemAvailable < 8,000,000 kB.
"""
from __future__ import annotations

import ast
import asyncio
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

T0 = time.time()
DEADLINE = T0 + float(os.environ.get("DEADLINE_S", "540"))
KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT / "tests"))
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/dsv41/uncensored-fp8"))
OUT = os.environ.get("OUT", str(KIT / "results/test_incr_tok.json"))
N_CONV = int(os.environ.get("N_CONV", "2000"))
RESULTS: dict = {"host": os.uname().nodename, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
FAILS: list[str] = []


def mem_kb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return 0


def guard(where: str) -> None:
    m = mem_kb()
    if m < 8_000_000:
        print(f"ABORT at {where}: MemAvailable {m} kB < 8000000", flush=True)
        sys.exit(3)


def check(cond: bool, what: str) -> bool:
    if not cond:
        FAILS.append(what)
        print("FAIL", what, flush=True)
    return cond


def section(name: str) -> None:
    guard(name)
    print(f"== {name}  (+{time.time() - T0:.0f}s, MemAvailable {mem_kb() // 1024} MB)", flush=True)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


guard("start")
import vllm  # noqa: E402

REAL_SITE = Path(vllm.__file__).resolve().parent
OVERLAY_NAME = "vllm.renderers.dsv41_incr_tok"

# ============================================================================== 1. installer
section("1 installer")
TMP = Path(tempfile.mkdtemp(prefix="incrtok-test-"))
SITE = TMP / "vllm"
(SITE / "renderers").mkdir(parents=True)
shutil.copy2(REAL_SITE / "renderers/deepseek_v4.py", SITE / "renderers/deepseek_v4.py")
BASE_BYTES = (SITE / "renderers/deepseek_v4.py").read_bytes()
check(hashlib.sha256(BASE_BYTES).hexdigest().startswith("f7175f99"), "live deepseek_v4.py is the f7175f99 base")


def inst(*args) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-I", str(KIT / "tools/incr_install.py"), *args], capture_output=True, text=True,
                       env={**os.environ, "INCR_SITE": str(SITE)})
    return r.returncode, r.stdout + r.stderr


rc, out = inst("status", "base"); check(rc == 0 and "STATE base" in out, f"status base on a clean copy: {out}")
rc, out = inst("dry-run"); check(rc == 0 and "dry-run OK" in out, f"dry-run: {out}")
check((SITE / "renderers/deepseek_v4.py").read_bytes() == BASE_BYTES, "dry-run wrote nothing")
rc, out = inst("install"); check(rc == 0 and "STATE patched" in out, f"install: {out}")
check((SITE / "renderers/deepseek_v4.py.bak-incrtok").read_bytes() == BASE_BYTES, "backup is the base, byte for byte")
check((SITE / "renderers/dsv41_incr_tok.py").read_bytes() == (KIT / "site/vllm/renderers/dsv41_incr_tok.py").read_bytes(), "side file installed")
rc, out = inst("install"); check(rc != 0 and "ABORT" in out, "second install refuses (base sha moved)")
rc, out = inst("status", "patched"); check(rc == 0, "status patched")
PATCHED_BYTES = (SITE / "renderers/deepseek_v4.py").read_bytes()
rc, out = inst("uninstall"); check(rc == 0 and "STATE base" in out, f"uninstall: {out}")
check((SITE / "renderers/deepseek_v4.py").read_bytes() == BASE_BYTES and not (SITE / "renderers/dsv41_incr_tok.py").exists()
      and not (SITE / "renderers/deepseek_v4.py.bak-incrtok").exists(), "uninstall restores the base exactly")
(SITE / "renderers/deepseek_v4.py").write_bytes(BASE_BYTES + b"# moved\n")
rc, out = inst("dry-run"); check(rc != 0 and "kit expects base" in out, "refuses an unexpected base sha")
rc, out = inst("install"); check(rc != 0 and (SITE / "renderers/deepseek_v4.py").read_bytes() == BASE_BYTES + b"# moved\n"
                                 and not (SITE / "renderers/dsv41_incr_tok.py").exists(), "install on a moved base writes nothing")
rc, out = inst("status"); check(rc != 0 and "STATE UNKNOWN" in out, "status UNKNOWN on a foreign file")
(SITE / "renderers/deepseek_v4.py").write_bytes(BASE_BYTES)
rc, out = inst("install"); check(rc == 0, "reinstall")


def ast_flag_off_diff() -> bool:
    base = ast.parse(BASE_BYTES.decode())
    pat = ast.parse(PATCHED_BYTES.decode())
    # expected additions: `import os`, `_DSV41_INCR_TOK = ...`, and `if _DSV41_INCR_TOK: ...` at the end of __init__
    body = [n for n in pat.body if not (isinstance(n, ast.Import) and [a.name for a in n.names] == ["os"])
            and not (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_DSV41_INCR_TOK")]
    for n in body:
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4Renderer":
            init = next(f for f in n.body if isinstance(f, ast.FunctionDef) and f.name == "__init__")
            last = init.body[-1]
            if not (isinstance(last, ast.If) and getattr(last.test, "id", "") == "_DSV41_INCR_TOK"):
                return False
            init.body = init.body[:-1]
    pat.body = body
    return ast.dump(pat) == ast.dump(base)


check(ast_flag_off_diff(), "patched file == base plus exactly: import os, the flag read, one if-block in __init__")
RESULTS["installer"] = "ok" if not FAILS else "fail"

# ============================================================================== 2. flag off
section("2 flag off")
from vllm.config import ModelConfig, VllmConfig  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
from vllm.renderers import merge_kwargs  # noqa: E402
from vllm.renderers.base import BaseRenderer  # noqa: E402
from vllm.tokenizers.registry import get_tokenizer  # noqa: E402

mc = ModelConfig(model=MODEL, tokenizer_mode="deepseek_v41", max_model_len=1048576, limit_mm_per_prompt={"image": 16})
VC = VllmConfig(model_config=mc)
TOK = get_tokenizer(MODEL, tokenizer_mode="deepseek_v41")
for k in [k for k in os.environ if k.startswith("DSV41_INCR_TOK")]:
    del os.environ[k]
DV4_OFF = load("vllm.renderers.deepseek_v4_incrtest_off", SITE / "renderers/deepseek_v4.py")
r_off = DV4_OFF.DeepseekV4Renderer(VC, TOK)
check(DV4_OFF._DSV41_INCR_TOK is False, "flag reads off when unset")
check("_tokenize_prompt" not in vars(r_off) and OVERLAY_NAME not in sys.modules, "flag off: no override, overlay never imported")
for v in ("0", "off", "", " 0 "):
    os.environ["DSV41_INCR_TOK"] = v
    m = load("vllm.renderers.deepseek_v4_incrtest_v", SITE / "renderers/deepseek_v4.py")
    check(m._DSV41_INCR_TOK is False, f"DSV41_INCR_TOK={v!r} is off")
del os.environ["DSV41_INCR_TOK"]

# ============================================================================== 3. load checks
section("3 load checks")
OVL = load(OVERLAY_NAME, SITE / "renderers/dsv41_incr_tok.py")
os.environ["DSV41_INCR_TOK"] = "1"
DV4 = load("vllm.renderers.deepseek_v4_incrtest_on", SITE / "renderers/deepseek_v4.py")
check(DV4._DSV41_INCR_TOK is True, "flag reads on")
ok, why = OVL.check_tokenizer(TOK)
check(ok, f"tokenizer checks pass: {why}")
RESULTS["tokenizer_check"] = why
for attr, bad in (("TOKENIZER_JSON_SHA256", "0" * 64), ("STRUCT_SHA256", "0" * 64)):
    keep = getattr(OVL, attr)
    setattr(OVL, attr, bad)
    ok2, why2 = OVL.check_tokenizer(TOK)
    r_bad = DV4.DeepseekV4Renderer(VC, TOK)
    setattr(OVL, attr, keep)
    check(not ok2 and "_tokenize_prompt" not in vars(r_bad), f"wrong {attr} keeps the renderer stock ({why2})")
keep = OVL.CUTS
OVL.CUTS = {**keep, "｜DSML｜": 128825}
check(not OVL.check_tokenizer(TOK)[0], "｜DSML｜ as a cut is refused by the overlap check")
OVL.CUTS = keep
for k, v in (("DSV41_INCR_TOK_MAX_MB", "abc"), ("DSV41_INCR_TOK_VERIFY_EVERY", "-1"), ("DSV41_INCR_TOK", "maybe")):
    old = os.environ.get(k)
    os.environ[k] = v
    r_bad = DV4.DeepseekV4Renderer(VC, TOK)
    check("_tokenize_prompt" not in vars(r_bad), f"{k}={v} keeps the renderer stock")
    if old is None:
        del os.environ[k]
    else:
        os.environ[k] = old


# ============================================================================== helpers
def make_renderer(**env):
    for k in [k for k in os.environ if k.startswith("DSV41_INCR_TOK_")]:
        del os.environ[k]
    os.environ["DSV41_INCR_TOK"] = env.pop("mode", "1")
    for k, v in env.items():
        os.environ["DSV41_INCR_TOK_" + k.upper()] = str(v)
    r = DV4.DeepseekV4Renderer(VC, TOK)
    inc = vars(r).get("_dsv41_incr_tok")
    assert inc is not None, "attach failed"
    recs: list[dict] = []
    inc._record = lambda rec: recs.append(rec)
    return r, inc, recs


def render(r, body: dict):
    """Server path: pydantic -> chat/tok params -> renderer.render_messages -> pre-tokenization (as profile_stages.py)."""
    req = ChatCompletionRequest.model_validate(body)
    tool_dicts = None if req.tools is None else [t.model_dump() for t in req.tools]
    chat_params = req.build_chat_params(None, "auto").with_defaults(
        merge_kwargs({}, dict(tools=tool_dicts, tokenize=False)),
        default_media_io_kwargs=(mc.multimodal_config.media_io_kwargs if mc.multimodal_config else None),
        default_mm_processor_kwargs=getattr(req, "mm_processor_kwargs", None))
    tok_params = req.build_tok_params(mc)
    _, prompt = r.render_messages(req.messages, chat_params)
    prompt = tok_params.apply_pre_tokenization(TOK, prompt)
    return prompt, tok_params


def stock(r, prompt, params):
    return BaseRenderer._tokenize_prompt(r, dict(prompt), params)


def lcp(a: str, b: str) -> int:
    return OVL.lcp(a, b)


ADDED = [t.content for t in TOK._tokenizer.get_added_tokens_decoder().values()]
from fuzzgen import Conv, Gen  # noqa: E402

# ============================================================================== 4. equality fuzz
section("4 equality fuzz")
r, inc, recs = make_renderer(verify_first=0, verify_every=0, min_chars=2048)
g = Gen(20261103, ADDED)
shared_sys = g.text(1800)
shared_tools = g.tools()
fz = {"conversations": 0, "requests": 0, "identical": 0, "kinds": {}, "ops": {}, "tokens_total": 0, "tokens_reused": 0,
      "early_edit_requests": 0, "early_edit_missed": 0, "singleton_path": 0, "modes": {}}
prev_text: dict[int, str] = {}
pool: list[Conv] = []
t_fuzz = time.time()
while fz["conversations"] < N_CONV or pool:
    if time.time() > DEADLINE - 200:  # leave time for sections 5-9
        print("fuzz stopped by deadline", flush=True)
        break
    while len(pool) < 4 and fz["conversations"] < N_CONV:
        c = Conv(g, shared_sys if g.R.random() < 0.7 else None, shared_tools if g.R.random() < 0.7 else None)
        pool.append(c)
        fz["conversations"] += 1
        fz["modes"][c.mode] = fz["modes"].get(c.mode, 0) + 1
    if not pool:
        break
    c = g.R.choice(pool)
    op = "new" if id(c) not in prev_text else c.advance()
    fz["ops"][op] = fz["ops"].get(op, 0) + 1
    prompt, params = render(r, c.body())
    text = prompt["prompt"]
    want = stock(r, prompt, params)
    n_before = len(recs)
    with inc.lock:
        cached_texts = [e.text for e in inc.entries.values()]
    if fz["requests"] % 10 == 0:
        got = r._tokenize_singleton_prompt(dict(prompt), params)  # the base sync path, via the instance override
        fz["singleton_path"] += 1
        same = got["prompt_token_ids"] == want["prompt_token_ids"]
    else:
        got = r._tokenize_prompt(dict(prompt), params)
        same = got == want
    fz["requests"] += 1
    fz["identical"] += same
    fz["tokens_total"] += len(want["prompt_token_ids"])
    if not check(same, f"fuzz request {fz['requests']} op {op} mode {c.mode}: cached != stock"):
        break
    rec = recs[-1] if len(recs) > n_before else {"kind": "skip"}
    fz["kinds"][rec["kind"]] = fz["kinds"].get(rec["kind"], 0) + 1
    if rec["kind"] == "hit":
        fz["tokens_reused"] += rec["reused"]
        # the reused text must lie inside the common prefix with some cached prompt
        shared = max((lcp(text, t) for t in cached_texts), default=0)
        check(rec["cut"] <= shared, f"hit cut {rec['cut']} beyond the longest shared prefix {shared}")
    if op == "edit_system" or (op == "edit_middle" and c.tools):
        # Tools modes render each message independently (drop_thinking off), so the first changed char against this
        # conversation's previous prompt is the edit point; no cached prompt can share text past it.
        e = lcp(prev_text[id(c)], text)
        fz["edit_requests"] = fz.get("edit_requests", 0) + 1
        ok_edit = rec["kind"].startswith("miss") or rec["kind"] == "skip" or (rec["kind"] == "hit" and rec["cut"] <= e)
        check(ok_edit, f"{op} at char {e} of {len(text)} reused past the edit: {rec}")
        if e < 0.5 * len(text):
            fz["early_edit_requests"] += 1
            fz["early_edit_missed"] += rec["kind"].startswith("miss")
            check(rec["kind"].startswith("miss"), f"{op} at char {e} of {len(text)} (before the reuse floor) did not miss: {rec}")
    prev_text[id(c)] = text
    if c.done():
        pool.remove(c)
        prev_text.pop(id(c), None)
fz["seconds"] = round(time.time() - t_fuzz, 1)
fz["reused_frac"] = round(fz["tokens_reused"] / max(fz["tokens_total"], 1), 3)
fz["cache_stats"] = inc.stats()
RESULTS["fuzz"] = fz
print(json.dumps(fz), flush=True)
check(fz["conversations"] >= min(N_CONV, 1000) and fz["identical"] == fz["requests"], "fuzz: >= 1000 conversations, all identical")
check(fz["kinds"].get("hit", 0) > fz["requests"] // 3, "fuzz: cache hits are a large share (the test exercises the cache)")
check(inc.stats().get("mismatch", 0) == 0 and not inc.disabled, "fuzz: no mismatch recorded")

# async base path uses the override
c = Conv(g, shared_sys, shared_tools)
c.steps.append(g.step(c))
p1, prm = render(r, c.body())
asyncio.run(r._tokenize_singleton_prompt_async(dict(p1), prm))
c.steps.append(g.step(c))
p2, prm = render(r, c.body())
nrec = len(recs)
got = asyncio.run(r._tokenize_singleton_prompt_async(dict(p2), prm))
check(len(recs) > nrec and recs[-1]["kind"] == "hit" and got["prompt_token_ids"] == stock(r, p2, prm)["prompt_token_ids"],
      "async base path (_tokenize_prompt_async on the executor) is served by the cache and identical")

# ============================================================================== 5. concurrency
section("5 concurrency")
r, inc, recs = make_renderer(verify_first=0, verify_every=0, min_chars=2048)
convs = [Conv(Gen(500 + i, ADDED), shared_sys, shared_tools) for i in range(8)]
for c in convs:
    c.target_chars = 10**9
    c.requests_left = 10**9
errors: list[str] = []
counts = {"requests": 0}
clock = threading.Lock()


def run_conv(c: Conv, n: int) -> None:
    for i in range(n):
        if i:
            c.advance() if c.g.R.random() < 0.8 else c.steps.append(c.g.step(c))
        prompt, params = render(r, c.body())
        got = r._tokenize_prompt(dict(prompt), params)
        want = stock(r, prompt, params)
        with clock:
            counts["requests"] += 1
            if got != want:
                errors.append(f"conv {id(c)} req {i}")


with ThreadPoolExecutor(4) as ex:
    list(ex.map(lambda c: run_conv(c, 14), convs))
RESULTS["concurrency"] = {"threads": 4, "conversations": 8, "requests": counts["requests"], "errors": len(errors), "stats": inc.stats()}
check(not errors and counts["requests"] == 8 * 14, f"concurrency: all identical ({errors[:3]})")
check(inc.stats().get("hit", 0) > 0, "concurrency: hits happened")

# ============================================================================== 6. eviction
section("6 eviction")
r, inc, recs = make_renderer(verify_first=0, verify_every=0, min_chars=2048, max_mb=1, max_entries=4)
convs = [Conv(Gen(900 + i, ADDED), None, None) for i in range(10)]
ev = {"requests": 0, "identical": 0, "max_entries_seen": 0, "max_bytes_seen": 0}
for rnd in range(12):
    for c in convs:
        c.requests_left = 10**9
        if rnd:
            c.steps.append(c.g.step(c))
        prompt, params = render(r, c.body())
        got = r._tokenize_prompt(dict(prompt), params)
        ev["requests"] += 1
        ev["identical"] += got == stock(r, prompt, params)
        with inc.lock:
            ev["max_entries_seen"] = max(ev["max_entries_seen"], len(inc.entries))
            ev["max_bytes_seen"] = max(ev["max_bytes_seen"], inc.bytes)
            check(inc.bytes == sum(e.nbytes for e in inc.entries.values()), "eviction: byte count is exact")
ev["stats"] = inc.stats()
RESULTS["eviction"] = ev
check(ev["identical"] == ev["requests"], "eviction: all identical")
check(ev["max_entries_seen"] <= 4 and ev["max_bytes_seen"] <= 1 << 20, f"eviction: bounds held ({ev['max_entries_seen']}, {ev['max_bytes_seen']})")
check(ev["stats"].get("evicted", 0) > 0, "eviction: evictions happened")

# ============================================================================== 7. string-level fuzz
section("7 string fuzz")
r, inc, recs = make_renderer(verify_first=0, verify_every=0, min_chars=2048, min_reuse=0.0)
params0 = ChatCompletionRequest.model_validate({"model": "x", "messages": [{"role": "user", "content": "x"}]}).build_tok_params(mc)
R = random.Random(77)
gs = Gen(78, ADDED)
head = gs.text(2100)
sf = {"cases": 0, "identical": 0, "hits": 0}
cur = head + gs.text(3000)
t7 = time.time()
while sf["cases"] < 6000 and time.time() < DEADLINE - 120:
    r_ = R.random()
    if r_ < 0.4:
        cur = cur + gs.text(R.randint(1, 800))
    elif r_ < 0.6:
        p = R.randint(len(head), len(cur))
        cur = cur[:p] + gs.fragment() + cur[p:]
    elif r_ < 0.75:
        cur = cur[:R.randint(len(head), len(cur))] + gs.text(R.randint(0, 300))
    elif r_ < 0.85:
        p = R.randint(len(head), len(cur))  # cut through a special token on purpose
        cur = cur[:p] + R.choice(["<｜User", "｜>", "</thi", "nk>", "｜DSML｜", "<｜end▁of▁sentence｜>", "<think>"]) + cur[p:]
    else:
        cur = head + gs.text(R.randint(100, 4000))
    if len(cur) > 16000:
        cur = head + cur[-6000:]
    prompt = {"prompt": cur}
    got = r._tokenize_prompt(dict(prompt), params0)
    want = stock(r, prompt, params0)
    sf["cases"] += 1
    sf["identical"] += got == want
    sf["hits"] += bool(recs) and recs[-1]["kind"] == "hit"
    if got != want:
        check(False, f"string fuzz case {sf['cases']}")
        break
sf["seconds"] = round(time.time() - t7, 1)
RESULTS["string_fuzz"] = sf
check(sf["cases"] >= 2000 and sf["identical"] == sf["cases"] and sf["hits"] > sf["cases"] // 3, f"string fuzz: {sf}")

# ============================================================================== 8. fallbacks
section("8 fallbacks")
r, inc, recs = make_renderer(verify_first=0, verify_every=0, min_chars=2048)
calls = {"n": 0}
real_orig = inc.orig


def spy(prompt, params):
    calls["n"] += 1
    return real_orig(prompt, params)


inc.orig = spy
c = Conv(Gen(4242, ADDED), shared_sys, shared_tools)
for _ in range(3):
    c.steps.append(c.g.step(c))
p_a, prm = render(r, c.body())
r._tokenize_prompt(dict(p_a), prm)  # miss, cached
c.steps.append(c.g.step(c))
p_b, prm = render(r, c.body())
fb = {}


def served_by_stock(prompt, params, label):
    n0 = calls["n"]
    got = r._tokenize_prompt(dict(prompt), params)
    used = calls["n"] - n0
    fb[label] = {"stock_calls": used, "identical": got == stock(r, prompt, params)}
    return used == 1 and fb[label]["identical"]


check(served_by_stock({**p_b, "multi_modal_data": {"image": [object()]}}, prm, "multi_modal_data"), "images fall back (multi_modal_data)")
check(served_by_stock({**p_b, "multi_modal_uuids": {"image": ["u"]}}, prm, "multi_modal_uuids"), "images fall back (multi_modal_uuids)")
check(served_by_stock(p_b, dataclasses.replace(prm, truncate_prompt_tokens=1000), "truncate_prompt_tokens"), "truncate_prompt_tokens falls back")
check(served_by_stock(p_b, dataclasses.replace(prm, add_special_tokens=True), "add_special_tokens"), "add_special_tokens=True falls back")
check(served_by_stock({"prompt": p_b["prompt"][:1500]}, prm, "short"), "short prompt falls back")
n_full = len(stock(r, p_b, prm)["prompt_token_ids"])
for label, slack in (("maxlen_minus_1", 2), ("maxlen_equal", 1), ("maxlen_over", 0)):
    # max_length = max_total - max_output + 1; pick max_total so that n_full = max_length - (slack - 1) - 1 ...
    max_len = {2: n_full + 1, 1: n_full, 0: n_full - 5}[slack]
    pr = dataclasses.replace(prm, max_total_tokens=max_len - 1, max_output_tokens=0)  # max_length = max_input_tokens + 1 = max_total + 1
    check(pr.get_encode_kwargs()["max_length"] == max_len, f"{label}: max_length set up")
    n0 = calls["n"]
    nrec = len(recs)
    got = r._tokenize_prompt(dict(p_b), pr)
    want = stock(r, p_b, pr)
    kind = recs[-1]["kind"] if len(recs) > nrec else "skip"
    fb[label] = {"max_length": max_len, "n_full": n_full, "kind": kind, "identical": got == want,
                 "got_len": len(got["prompt_token_ids"])}
    if slack == 2:
        check(kind == "hit" and got == want, f"{label}: n = max_length - 1 is served from the cache and identical ({fb[label]})")
    else:
        check(kind == "miss_maxlen" and calls["n"] - n0 == 1 and got == want, f"{label}: truncation edge falls back to stock ({fb[label]})")
RESULTS["fallbacks"] = fb
inc.orig = real_orig

# ============================================================================== 9. guard
section("9 guard")
gd = {}


def corrupting(inc_, which):
    """Wrap _incremental so the which-th successful incremental result has one wrong id."""
    real = inc_._incremental
    st = {"n": 0}

    def f(text, kwargs, max_len):
        ids, info = real(text, kwargs, max_len)
        if ids is not None:
            st["n"] += 1
            if st["n"] == which:
                ids = list(ids)
                ids[len(ids) // 2] = (ids[len(ids) // 2] + 1) % 128000
        return ids, info

    inc_._incremental = f
    return st


def guard_run(label, which, **env):
    r, inc, recs = make_renderer(min_chars=2048, **env)
    st = corrupting(inc, which)
    c = Conv(Gen(31337, ADDED), shared_sys, shared_tools)
    c.requests_left = 10**9
    res = {"served_identical": 0, "requests": 0}
    for i in range(which + 4):
        c.steps.append(c.g.step(c))
        prompt, params = render(r, c.body())
        got = r._tokenize_prompt(dict(prompt), params)
        res["requests"] += 1
        res["served_identical"] += got == stock(r, prompt, params)
    res.update(stats=inc.stats(), disabled=inc.disabled, entries=len(inc.entries), incremental_calls=st["n"],
               mismatch_records=sum(1 for x in recs if x["kind"] == "MISMATCH"))
    gd[label] = res
    return res


res = guard_run("first_n", 2, verify_first=3, verify_every=0)
check(res["served_identical"] == res["requests"] and res["disabled"] and res["entries"] == 0
      and res["stats"].get("mismatch") == 1 and res["mismatch_records"] == 1 and res["incremental_calls"] == 2,
      f"guard (first-N verify): caught, full result served, cache off, later requests stock: {res}")
res = guard_run("one_in_k", 5, verify_first=0, verify_every=1)
check(res["served_identical"] == res["requests"] and res["disabled"] and res["stats"].get("mismatch") == 1 and res["incremental_calls"] == 5,
      f"guard (1-in-K verify): caught and disabled: {res}")
res = guard_run("audit", 3, mode="audit")
check(res["served_identical"] == res["requests"] and res["disabled"] and res["stats"].get("mismatch") == 1,
      f"audit mode: always serves stock, catches the wrong id, disables: {res}")
res = guard_run("unguarded_control", 2, verify_first=0, verify_every=0)
check(res["served_identical"] == 2 and not res["disabled"],
      f"control: with verification off the wrong id IS served (and carried into the later hits of that conversation, so the next "
      f"sampled check would catch it), so the harness's comparison is sensitive: {res}")
# audit with no corruption: served object is the stock one, hits verified, nothing disabled
r, inc, recs = make_renderer(mode="audit", min_chars=2048)
c = Conv(Gen(999, ADDED), shared_sys, shared_tools)
for i in range(6):
    c.steps.append(c.g.step(c))
    prompt, params = render(r, c.body())
    check(r._tokenize_prompt(dict(prompt), params) == stock(r, prompt, params), "audit serves the stock result")
gd["audit_clean"] = inc.stats()
check(inc.stats().get("verified", 0) >= 4 and not inc.disabled, f"audit mode: clean run verifies hits: {inc.stats()}")
# default production settings: first 32 hits verified
r, inc, recs = make_renderer(min_chars=2048)
c = Conv(Gen(1000, ADDED), shared_sys, shared_tools)
for i in range(6):
    c.steps.append(c.g.step(c))
    prompt, params = render(r, c.body())
    r._tokenize_prompt(dict(prompt), params)
gd["defaults"] = inc.stats()
check(inc.stats().get("verified", 0) == 5 and inc.stats().get("hit", 0) == 0, f"defaults: early hits are all verified: {inc.stats()}")
RESULTS["guard"] = gd

# ============================================================================== done
RESULTS["fails"] = FAILS
RESULTS["elapsed_s"] = round(time.time() - T0, 1)
RESULTS["mem_available_kb_end"] = mem_kb()
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
Path(OUT).write_text(json.dumps(RESULTS, indent=1, default=str))
shutil.rmtree(TMP, ignore_errors=True)
print(f"VERDICT {'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failures, {RESULTS['elapsed_s']} s, results {OUT}", flush=True)
sys.exit(0 if not FAILS else 1)
