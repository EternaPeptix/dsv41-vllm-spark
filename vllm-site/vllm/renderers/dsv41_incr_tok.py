# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 incremental prompt tokenization for the API server (incr-tok kit).

Imported only when DSV41_INCR_TOK is set (anchored edit in renderers/deepseek_v4.py). Unset, this file is never
imported and the renderer runs stock code.

What it does. An agent sends the whole conversation on every step, and the API server tokenizes the whole rendered
prompt again (renderers/base.py `_tokenize_prompt`, about 2.8 us per token: 0.9 s at 300K tokens). This module keeps
(prompt string, prompt ids) for recent prompts. For a new prompt S it finds the cached prompt P with the longest common
prefix L, cuts at the end c of the last literal occurrence of a cut token inside S[:L], checks that P's ids end with
enc(P[c:]), and returns ids(P)[:k] + enc(S[c:]).

Why that is exact. For the DSv4.1 tokenizer (identity normalizer, Split + ByteLevel pre-tokenizers without a prefix
space, BPE without byte fallback, no post-processor ids, all added tokens without lstrip/rstrip/single_word), a literal
occurrence of an added-token string T is always matched as T, provided no added token overlaps T (a proper suffix of one
equal to a proper prefix of the other) or contains it, and T contains no other one. Then enc(X) == enc(X[:c]) + enc(X[c:])
at the end c of any occurrence (PROFILE.md section 4). The five cut tokens are checked for that against the whole added
vocabulary when this module loads. ｜DSML｜ overlaps itself and is never a cut. The V4.1 encoder writes a cut token at
every message boundary.

Guards. The module disables itself at load (and the renderer stays stock) unless tokenizer.json has the pinned sha256,
the loaded tokenizer's pipeline has the pinned structure digest, the cut tokens pass the overlap check against all added
tokens, and a known-answer split test passes. At run time it serves the full tokenize for anything it cannot prove:
non-string prompts, images (multi_modal_data / multi_modal_uuids), offsets, truncate_prompt_tokens, add_special_tokens,
any unexpected encode kwarg, and any result whose length reaches the truncation max_length. The first VERIFY_FIRST hits
of the process, then 1 in VERIFY_EVERY at random, are also tokenized in full and compared; on any difference it logs,
disables itself for the life of the process and serves the full result.

Modes (env DSV41_INCR_TOK): 1/on = serve cache hits (with the guard); audit = always serve the full tokenize, compute
the incremental result too and compare (shadow mode, exactness evidence on live traffic). Tunables:
DSV41_INCR_TOK_VERIFY_FIRST (32), DSV41_INCR_TOK_VERIFY_EVERY (64, 0 = no random checks after the first ones),
DSV41_INCR_TOK_MAX_MB (256), DSV41_INCR_TOK_MAX_ENTRIES (32), DSV41_INCR_TOK_MIN_CHARS (32768),
DSV41_INCR_TOK_MIN_REUSE (0.5: fraction of the new prompt's chars that must be reused, else full tokenize),
DSV41_INCR_TOK_STATS_EVERY (200 requests per INFO stats line), DSV41_INCR_TOK_LOG (path: one JSON line per eligible
request with timings; never prompt text).
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

KIT = "incr-tok v1"
TOKENIZER_JSON_SHA256 = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
# sha256 of the canonical JSON of the loaded backend's normalizer, pre_tokenizer, post_processor, model and
# added_tokens (structure_digest below), measured on spark-two with the vllm032 venv (tokenizers 0.23.3).
STRUCT_SHA256 = "fbfe19ded3dacda625833a3988f1e6953afefc13d0381270c6214ee494cc34bc"
# Cut tokens and their ids. Not ｜DSML｜ (128825): it overlaps itself, so a literal occurrence is not always a match.
CUTS: dict[str, int] = {
    "<｜end▁of▁sentence｜>": 1,
    "<｜User｜>": 128803,
    "<｜Assistant｜>": 128804,
    "</think>": 128822,
    "<think>": 128821,
}
_ALLOWED_KWARGS = frozenset(("truncation", "max_length", "add_special_tokens"))
_HEAD = 2048  # chars compared first to skip unrelated cached prompts
_CUT_WINDOW = 1 << 16  # chars searched back from the common prefix end before widening to the whole prefix


# ---------------------------------------------------------------------------------------------------- config
def _env_mode() -> str | None:
    v = os.environ.get("DSV41_INCR_TOK", "0").strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return None
    if v in ("1", "on", "true", "yes"):
        return "on"
    if v == "audit":
        return "audit"
    raise ValueError(f"DSV41_INCR_TOK={v!r}: want 1, audit or 0")


def _env_num(name: str, default, cast, lo, hi):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    val = cast(raw)
    if not (lo <= val <= hi):
        raise ValueError(f"{name}={raw!r} outside [{lo}, {hi}]")
    return val


class Config:
    def __init__(self, mode: str):
        self.mode = mode
        self.verify_first = _env_num("DSV41_INCR_TOK_VERIFY_FIRST", 32, int, 0, 10**9)
        self.verify_every = _env_num("DSV41_INCR_TOK_VERIFY_EVERY", 64, int, 0, 10**9)
        self.max_bytes = _env_num("DSV41_INCR_TOK_MAX_MB", 256, int, 1, 65536) << 20
        self.max_entries = _env_num("DSV41_INCR_TOK_MAX_ENTRIES", 32, int, 1, 4096)
        self.min_chars = _env_num("DSV41_INCR_TOK_MIN_CHARS", 32768, int, _HEAD, 10**9)
        self.min_reuse = _env_num("DSV41_INCR_TOK_MIN_REUSE", 0.5, float, 0.0, 1.0)
        self.stats_every = _env_num("DSV41_INCR_TOK_STATS_EVERY", 200, int, 1, 10**9)
        self.log_path = os.environ.get("DSV41_INCR_TOK_LOG", "").strip() or None

    def describe(self) -> str:
        return (f"verify first {self.verify_first} then 1/{self.verify_every or 'never'}, cache {self.max_bytes >> 20} MB / "
                f"{self.max_entries} entries, min {self.min_chars} chars, min reuse {self.min_reuse}")


# ---------------------------------------------------------------------------------------------------- load checks
def structure_digest(d: dict) -> str:
    keep = {k: d.get(k) for k in ("normalizer", "pre_tokenizer", "post_processor", "model", "added_tokens")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cut_overlaps(added: list[str]) -> list[tuple[str, str, str]]:
    """(cut, other, kind) for every overlap or containment between a cut token and any added token (itself included)."""
    bad = []
    for t in CUTS:
        for x in added:
            if x != t and (t in x or x in t):
                bad.append((t, x, "contains"))
            for a, b in ((x, t), (t, x)):  # proper suffix of a == proper prefix of b
                for i in range(1, len(a)):
                    suf = a[i:]
                    if len(suf) < len(b) and b.startswith(suf):
                        bad.append((t, x, "overlap"))
                        break
    return bad


def _probe_strings() -> list[str]:
    frags = ["\n", "\n\n", " ", "  ", "<", ">", "</", "｜", "<｜", "｜>", "<｜Use", "r｜>", "<thi", "nk>", "</think",
             "<think", "｜DSML｜", "｜DSML｜DSML｜", "<｜end▁of▁sentence", "▁", "123", "4567", "é", "的一是", "🙂",
             "def f(x):", "    return x", '{"a": 1}', "Hello", "<tool_result>", "\r\n", " \n", "x"]
    cuts = list(CUTS)
    rng = random.Random(4101)
    out = []
    for _ in range(64):
        out.append("".join(rng.choice(cuts) if rng.random() < 0.3 else rng.choice(frags) for _ in range(rng.randint(4, 40))))
    return out


def check_tokenizer(tok) -> tuple[bool, str]:
    """True only if every assumption of the cut proof holds for this tokenizer object. Never raises."""
    try:
        backend = getattr(tok, "_tokenizer", None)
        if backend is None or not hasattr(backend, "to_str"):
            return False, "no Rust tokenizers backend"
        path = os.path.join(str(getattr(tok, "name_or_path", "") or ""), "tokenizer.json")
        if not os.path.isfile(path):
            return False, f"no tokenizer.json at {path!r}"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != TOKENIZER_JSON_SHA256:
            return False, f"tokenizer.json sha256 {h.hexdigest()[:16]} != pinned {TOKENIZER_JSON_SHA256[:16]}"
        d = json.loads(backend.to_str())
        if d.get("normalizer") not in (None, {"type": "Sequence", "normalizers": []}):
            return False, f"normalizer {d.get('normalizer')!r} is not the identity"
        pre = d.get("pre_tokenizer") or {}
        steps = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else [pre]
        for st in steps:
            if st.get("type") not in ("Split", "ByteLevel"):
                return False, f"pre_tokenizer step {st.get('type')!r}"
            if st.get("type") == "ByteLevel" and st.get("add_prefix_space"):
                return False, "ByteLevel pre_tokenizer adds a prefix space"
        pp = d.get("post_processor")
        if pp is not None and pp.get("type") != "ByteLevel":
            return False, f"post_processor {pp.get('type')!r}"
        model = d.get("model") or {}
        if model.get("type") != "BPE" or model.get("byte_fallback") or model.get("dropout") not in (None, 0, 0.0):
            return False, "model is not plain BPE"
        added = d.get("added_tokens") or []
        for t in added:
            if t.get("lstrip") or t.get("rstrip") or t.get("single_word"):
                return False, f"added token {t.get('content')!r} has lstrip/rstrip/single_word"
        ids = {t["content"]: t["id"] for t in added}
        for t, i in CUTS.items():
            if ids.get(t) != i:
                return False, f"cut token {t!r} is id {ids.get(t)}, expected {i}"
        bad = cut_overlaps(list(ids))
        if bad:
            return False, f"cut token overlaps: {bad[:3]}"
        digest = structure_digest(d)
        if digest != STRUCT_SHA256:
            return False, f"tokenizer structure digest {digest[:16]} != pinned {STRUCT_SHA256[:16]}"
        # Known-answer split test through the same HF __call__ the server uses.
        def enc(s):
            return list(tok(s, add_special_tokens=False)["input_ids"]) if s else []
        for s in _probe_strings():
            full = enc(s)
            for t in CUTS:
                j = s.find(t)
                while j >= 0:
                    for c in (j, j + len(t)):
                        if enc(s[:c]) + enc(s[c:]) != full:
                            return False, f"known-answer split failed at {t!r}"
                    j = s.find(t, j + 1)
        return True, f"tokenizer.json {TOKENIZER_JSON_SHA256[:8]}, structure {digest[:8]}, {len(added)} added tokens, 5 cuts clean"
    except Exception as e:  # noqa: BLE001 - any surprise disables the cache, never the server
        return False, f"check raised {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------------------------------- string helpers
def lcp(a: str, b: str) -> int:
    """Length of the longest common prefix (galloping, then binary search on slices)."""
    n = min(len(a), len(b))
    if (b.startswith(a) if len(a) <= len(b) else a.startswith(b)):
        return n
    lo, step = 0, 4096
    while True:
        hi = min(lo + step, n)
        if a[lo:hi] != b[lo:hi]:
            break
        lo, step = hi, step * 2
    while hi - lo > 64:  # invariant: a[:lo] == b[:lo], first difference in [lo, hi)
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    while lo < hi and a[lo] == b[lo]:
        lo += 1
    return lo


def last_cut(s: str, end: int) -> int:
    """End of the last occurrence of a cut token lying wholly inside s[:end], or -1."""
    lo = max(0, end - _CUT_WINDOW)
    while True:
        best = -1
        for t in CUTS:
            j = s.rfind(t, lo, end)
            if j >= 0 and j + len(t) > best:
                best = j + len(t)
        if best >= 0 or lo == 0:
            return best
        lo = 0


class _Entry:
    __slots__ = ("text", "ids", "head", "nbytes")

    def __init__(self, text: str, ids: list[int]):
        self.text = text
        self.ids = array.array("I", ids)
        self.head = text[:_HEAD]
        self.nbytes = sys.getsizeof(text) + self.ids.itemsize * len(self.ids) + 256


# ---------------------------------------------------------------------------------------------------- the cache
class IncrTokenizer:
    """Wraps a renderer's stock `_tokenize_prompt(prompt, params)`; thread-safe."""

    def __init__(self, tok, cfg: Config, orig: Callable, build: Callable, wants_offsets: Callable):
        self.tok = tok
        self.cfg = cfg
        self.orig = orig
        self.build = build
        self.wants_offsets = wants_offsets
        self.disabled = False
        self.lock = threading.Lock()
        self.entries: OrderedDict[int, _Entry] = OrderedDict()
        self.bytes = 0
        self.next_key = 0
        self.rng = random.Random(int.from_bytes(os.urandom(8), "little"))
        self.c: dict[str, float] = {}
        self.calls = 0
        self.log_f = None
        self.log_lock = threading.Lock()
        if cfg.log_path:
            try:
                self.log_f = open(os.path.expanduser(cfg.log_path), "a", buffering=1)
            except OSError as e:
                logger.warning("DSV41_INCR_TOK_LOG %s: %s (no per-request log)", cfg.log_path, e)

    # ------------------------------------------------------------------ bookkeeping
    def _bump(self, key: str, by: float = 1) -> None:
        self.c[key] = self.c.get(key, 0) + by

    def stats(self) -> dict[str, Any]:
        with self.lock:
            out = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in sorted(self.c.items())}
            out.update(entries=len(self.entries), cache_mb=round(self.bytes / 2**20, 1), disabled=self.disabled,
                       mode=self.cfg.mode)
        return out

    def _record(self, rec: dict) -> None:
        if self.log_f is None:
            return
        rec["t"] = round(time.time(), 3)
        try:
            with self.log_lock:
                self.log_f.write(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _tick(self) -> None:
        with self.lock:
            self.calls += 1
            due = self.calls % self.cfg.stats_every == 0
        if due:
            logger.info("DSV41_INCR_TOK stats %s", json.dumps(self.stats()))

    # ------------------------------------------------------------------ cache entries
    def _lookup(self, text: str):
        head = text[:_HEAD]
        with self.lock:
            snap = list(reversed(self.entries.items()))  # most recent first
        best_key, best_e, best_l = None, None, 0
        for key, e in snap:
            if len(e.text) <= best_l or e.head != head:
                continue
            L = len(e.text) if text.startswith(e.text) else lcp(text, e.text)
            if L > best_l:
                best_key, best_e, best_l = key, e, L
        if best_e is not None:
            with self.lock:
                if best_key in self.entries:
                    self.entries.move_to_end(best_key)
        return best_e, best_l

    def _store(self, text: str, ids: list[int], max_len: int | None) -> None:
        if len(text) < self.cfg.min_chars or (max_len is not None and len(ids) >= max_len):
            return
        e = _Entry(text, ids)
        if e.nbytes > self.cfg.max_bytes:
            return
        with self.lock:
            if self.disabled:
                return
            # Drop entries this one supersedes: any cached prompt that is a prefix of the new one.
            for k in [k for k, o in self.entries.items() if len(o.text) <= len(text) and o.head == e.head
                      and text.startswith(o.text)]:
                self.bytes -= self.entries.pop(k).nbytes
            key = self.next_key
            self.next_key += 1
            self.entries[key] = e
            self.bytes += e.nbytes
            while self.bytes > self.cfg.max_bytes or len(self.entries) > self.cfg.max_entries:
                _, old = self.entries.popitem(last=False)
                self.bytes -= old.nbytes
                self._bump("evicted")

    def _disable(self, why: str, detail: dict) -> None:
        with self.lock:
            first = not self.disabled
            self.disabled = True
            self.entries.clear()
            self.bytes = 0
            self._bump("mismatch")
        if first:
            logger.error("DSV41_INCR_TOK MISMATCH (%s): incremental ids differ from the full tokenize; the full result was "
                         "served and the cache is disabled for this process. %s", why, json.dumps(detail))
            self._record({"kind": "MISMATCH", "why": why, **detail})

    # ------------------------------------------------------------------ the request path
    def _ineligible(self, prompt, params, text) -> str | None:
        if not isinstance(text, str):
            return "skip_not_text"
        if len(text) < self.cfg.min_chars:
            return "skip_short"
        if any(prompt.get(k) for k in ("multi_modal_data", "multi_modal_uuids", "mm_processor_kwargs",
                                       "prompt_token_ids", "prompt_embeds")):
            return "skip_mm"
        if getattr(params, "truncate_prompt_tokens", None) is not None:
            return "skip_truncate"
        if self.wants_offsets(prompt, params):
            return "skip_offsets"
        return None

    def _incremental(self, text: str, kwargs: dict, max_len: int | None):
        """(ids, info) or (None, reason)."""
        e, L = self._lookup(text)
        if e is None or L <= 0:
            return None, "miss_none"
        c = last_cut(text, L)
        if c <= 0:
            return None, "miss_nocut"
        if c < self.cfg.min_reuse * len(text):
            return None, "miss_low_reuse"

        def enc(s: str) -> list[int]:
            return list(self.tok(s, **kwargs)["input_ids"]) if s else []

        tail = enc(e.text[c:])
        k = len(e.ids) - len(tail)
        if k <= 0 or e.ids[k:] != array.array("I", tail):
            return None, "miss_tailcheck"
        suffix = enc(text[c:])
        if max_len is not None and k + len(suffix) >= max_len:
            return None, "miss_maxlen"
        return e.ids[:k].tolist() + suffix, {"cut": c, "reused": k, "new": len(suffix), "tail": len(tail)}

    def _verify_now(self) -> bool:
        with self.lock:
            n = self.c.get("verified", 0)
            if n < self.cfg.verify_first:
                return True
        return self.cfg.verify_every > 0 and self.rng.random() * self.cfg.verify_every < 1.0

    def __call__(self, prompt, params):
        if self.disabled:
            return self.orig(prompt, params)
        text = prompt.get("prompt")
        why = self._ineligible(prompt, params, text)
        kwargs = None
        if why is None:
            kwargs = params.get_encode_kwargs()
            if set(kwargs) - _ALLOWED_KWARGS or kwargs.get("add_special_tokens") is not False:
                why = "skip_kwargs"
            elif kwargs.get("truncation") and not isinstance(kwargs.get("max_length"), int):
                why = "skip_kwargs"
            elif not kwargs.get("truncation") and kwargs.get("max_length") is not None:
                why = "skip_kwargs"
        if why is not None:
            with self.lock:
                self._bump(why)
            self._tick()
            return self.orig(prompt, params)
        max_len = kwargs.get("max_length") if kwargs.get("truncation") else None
        t0 = time.perf_counter()
        try:
            ids, info = self._incremental(text, kwargs, max_len)
        except Exception as ex:  # noqa: BLE001 - never fail a request because of the cache
            ids, info = None, "miss_error"
            logger.warning("DSV41_INCR_TOK internal error, serving the full tokenize: %r", ex)
        t_inc = (time.perf_counter() - t0) * 1e3
        audit = self.cfg.mode == "audit"
        verify = ids is not None and (audit or self._verify_now())
        if ids is None or verify or audit:
            t1 = time.perf_counter()
            res = self.orig(prompt, params)
            t_full = (time.perf_counter() - t1) * 1e3
            full = res["prompt_token_ids"]
            kind = info if ids is None else "verified"
            if ids is not None and ids != full:
                n = min(len(ids), len(full))
                first = next((i for i in range(n) if ids[i] != full[i]), n)
                self._disable("audit" if audit else "verify", {"chars": len(text), "full_tokens": len(full),
                              "incr_tokens": len(ids), "first_diff": first, **info})
                kind = "verify_failed"  # the details are in the MISMATCH record written by _disable
            else:
                self._store(text, full, max_len)
            with self.lock:
                if kind != "verify_failed":  # counted as "mismatch" by _disable
                    self._bump(kind)
                self._bump("ms_full", t_full)
                self._bump("ms_incr", t_inc)
                self._bump("n_full_timed")
            self._record({"kind": kind, "chars": len(text), "tokens": len(full), "ms_full": round(t_full, 2),
                          "ms_incr": round(t_inc, 2), **(info if isinstance(info, dict) else {})})
            self._tick()
            return res
        self._store(text, ids, max_len)
        with self.lock:
            self._bump("hit")
            self._bump("ms_incr", t_inc)
            self._bump("tokens_reused", info["reused"])
        self._record({"kind": "hit", "chars": len(text), "tokens": len(ids), "ms_incr": round(t_inc, 2), **info})
        self._tick()
        return self.build(ids, prompt)


# ---------------------------------------------------------------------------------------------------- hook
def attach(renderer) -> IncrTokenizer | None:
    """Called from DeepseekV4Renderer.__init__ when DSV41_INCR_TOK is set. Never raises."""
    try:
        mode = _env_mode()
        if mode is None:
            return None
        cfg = Config(mode)
        tok = renderer.get_tokenizer()
        t0 = time.perf_counter()
        ok, why = check_tokenizer(tok)
        dt = (time.perf_counter() - t0) * 1e3
        if not ok:
            logger.warning("DSV41_INCR_TOK=%s requested but DISABLED (stock tokenize): %s", mode, why)
            return None
        from vllm.utils.async_utils import make_async

        inc = IncrTokenizer(tok, cfg, renderer._tokenize_prompt, renderer._build_tokens_prompt, renderer._wants_offsets)
        renderer._tokenize_prompt = inc
        renderer._tokenize_prompt_async = make_async(inc, executor=renderer._executor)
        renderer._dsv41_incr_tok = inc
        logger.info("DSV41_INCR_TOK=%s enabled (%s): %s; %s; checks %.0f ms", mode, KIT, why, cfg.describe(), dt)
        return inc
    except Exception as e:  # noqa: BLE001
        logger.warning("DSV41_INCR_TOK attach failed, stock tokenize: %r", e)
        return None
