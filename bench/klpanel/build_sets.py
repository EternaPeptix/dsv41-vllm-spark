#!/usr/bin/env python3
"""build_sets.py - build the fixed token-id sets for kl_panel.py. Run ONCE, on a Spark CPU:

  cd ~/dsv41-bench/klpanel && TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 \
    nice -n 19 ~/venvs/vllm032/bin/python build_sets.py --model ~/models/dsv41/uncensored-fp8

Sources (local files only, benign, permissively licensed or shipped with the OS):
  code   CPython 3.12 stdlib sources (PSF) + C headers under /usr/include
  math   GSM8K test split ~/gsm8k_test.jsonl (MIT), calculator annotations <<..>> stripped
  prose  Python's own documentation text (pydoc_data/topics.py, PSF) + man pages (de-roffed here)
  tools  synthetic benign tool-use conversations, rendered by vLLM's own DeepSeek-V4.1 chat encoder
         (vllm/tokenizers/deepseek_v41_encoding.py, loaded by file path - vLLM itself is not imported)
  long   whole large stdlib files and concatenated doc topics, 8K tokens each
  gen    chat prompts for the optional decode panel (kl_panel.py capture --panel gen)
Every sequence starts with BOS (id 0). Token ids come from the model's tokenizer.json via
transformers.TokenizersBackend (the class vLLM's deepseek_v41 tokenizer mode wraps; AutoTokenizer as
a fallback), cross-checked against the raw `tokenizers` library. Output: sets/<name>.jsonl (token
ids + source provenance, no text) + sets/manifest.json (sha256 per sequence, per set and per file)
+ sets/SHA256SUMS. Refuses to overwrite an existing manifest: REF/candidate runs must score
identical tokens. --preview-out FILE writes decoded heads of every sequence for a human check
(keep it outside sets/).
"""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import sysconfig
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDER_VERSION = 1

# name: (panel, n_seq, seq_len)   ~295K teacher-forced positions in total
DEFAULT_PLAN = {
    "code": ("tf", 40, 2048),
    "math": ("tf", 48, 1024),
    "prose": ("tf", 40, 2048),
    "tools": ("tf", 32, 1024),
    "long": ("tf", 6, 8192),
    "gen": ("gen", 24, 0),
}
CODE_PY, CODE_C = 32, 8          # of the 40 code sequences
PROSE_DOC, PROSE_MAN = 24, 16    # of the 40 prose sequences
LONG_PY, LONG_DOC = 4, 2         # of the 6 long sequences
GEN_MAX_TOKENS = 256
SEED = 20261024


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def seq_sha(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def set_sha(rows) -> str:
    return hashlib.sha256("\n".join(f"{i}:{r['sha256']}" for i, r in enumerate(rows)).encode()).hexdigest()


def file_sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def by_hash(items, key=lambda x: x):
    """Deterministic pseudo-random order that does not depend on the filesystem's listing order."""
    return sorted(items, key=lambda x: sha(str(key(x))))


# ---------------------------------------------------------------------------------------- code

PY_SKIP_DIRS = {"test", "tests", "idlelib", "lib2to3", "pydoc_data", "encodings", "site-packages",
                "dist-packages", "turtledemo", "__pycache__", "__phello__", "ensurepip", "tkinter"}
PY_SKIP_FILES = {"html/entities.py", "_pydecimal.py", "typing.py", "inspect.py", "tarfile.py"}
LONG_PY_FILES = ("_pydecimal.py", "typing.py", "inspect.py", "tarfile.py")


def stdlib_files(root: str) -> list[str]:
    out = []
    for dp, dns, fns in os.walk(root):
        rel = os.path.relpath(dp, root)
        parts = [] if rel == "." else rel.split(os.sep)
        if any(p in PY_SKIP_DIRS or p.startswith("config-") for p in parts):
            dns[:] = []
            continue
        dns.sort()
        for fn in sorted(fns):
            if fn.endswith(".py"):
                out.append(os.path.relpath(os.path.join(dp, fn), root))
    return out


_TOPLEVEL = re.compile(r"(def |class |async def |@)")


def code_window(text: str, frac: float = 0.2) -> str:
    """Start at the first top-level def/class/decorator after `frac` of the file (skips the
    module docstring and import block, which are near-identical across files)."""
    lines = text.splitlines(keepends=True)
    start = int(frac * len(lines))
    for i in range(start, len(lines)):
        if _TOPLEVEL.match(lines[i]):
            return "".join(lines[i:])
    return text


def header_window(text: str, frac: float = 0.15) -> str:
    lines = text.splitlines(keepends=True)
    for i in range(int(frac * len(lines)), len(lines)):
        if not lines[i].strip():
            return "".join(lines[i + 1:])
    return text


VENDOR_HDR = re.compile(r"(cuda|cudnn|cublas|cusparse|cufft|curand|nccl|nvidia|nvml|nvtx|tensorrt|nv_|mlx|"
                        r"rdma/hfi|infiniband/mlx|mellanox|doca|ucx|ucp|ucs|gdr)", re.I)


def c_headers(root: str) -> list[str]:
    out = []
    for dp, dns, fns in os.walk(root):
        rel = os.path.relpath(dp, root)
        parts = [] if rel == "." else rel.split(os.sep)
        if parts and (parts[0] in ("c++",) or parts[0].startswith("python")):
            dns[:] = []
            continue
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            if fn.endswith(".h") and 12_000 <= os.path.getsize(p) <= 400_000 and not VENDOR_HDR.search(rel):
                out.append(rel)
    return out


# ---------------------------------------------------------------------------------------- math

def gsm8k_docs(path: str):
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            it = json.loads(line)
            ans = re.sub(r"<<[^>]*>>", "", it["answer"])
            yield f"gsm8k:{i}", f"Question: {it['question'].strip()}\nAnswer: {ans.strip()}"


# --------------------------------------------------------------------------------------- prose

def pydoc_topics(stdlib: str) -> dict:
    path = os.path.join(stdlib, "pydoc_data", "topics.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "topics":
            return ast.literal_eval(node.value)
    raise RuntimeError(f"no topics dict in {path}")


_ROFF_CHARS = {
    "em": "—", "en": "–", "hy": "-", "mi": "-", "aq": "'", "dq": '"', "lq": "“", "rq": "”",
    "oq": "‘", "cq": "’", "bu": "•", "co": "©", "rg": "®", "tm": "™", "de": "°",
    "+-": "±", "mu": "×", "di": "÷", "<=": "≤", ">=": "≥", "!=": "≠",
    "->": "→", "<-": "←", "ga": "`", "ha": "^", "ti": "~", "sl": "/", "rs": "\\", "ba": "|",
    "br": "|", "ul": "_", "pl": "+", "eq": "=", "lB": "[", "rB": "]", "lC": "{", "rC": "}", "at": "@",
    "sh": "#", "Do": "$", "ct": "¢", "Po": "£", "Ye": "¥", "ss": "ß", "fo": "«",
    "fc": "»", "dg": "†", "sc": "§", "ps": "¶", "ru": "_", "va": "↕",
}


def _skip_arg(s: str, i: int) -> int:
    """i points at the char after an escape letter that takes (x | (xx | [name]); return the end."""
    if i < len(s) and s[i] == "(":
        return i + 3
    if i < len(s) and s[i] == "[":
        j = s.find("]", i + 1)
        return j + 1 if j >= 0 else len(s)
    return i + 1


def roff_inline(s: str) -> str:
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            break
        c = s[i + 1]
        if c == '"':                         # comment
            break
        if c in "fFnk*gmMY$V":               # font / register / string / color: drop
            i = _skip_arg(s, i + 2)
            if c in "n" and i < n and s[i - 1] in "+-":
                pass
            continue
        if c == "s":                         # size change
            j = i + 2
            if j < n and s[j] in "+-":
                j += 1
            if j < n and s[j] in "([":
                j = _skip_arg(s, j)
            else:
                while j < n and s[j].isdigit() and j < i + 4:
                    j += 1
            i = j
            continue
        if c == "(":
            out.append(_ROFF_CHARS.get(s[i + 2:i + 4], ""))
            i += 4
            continue
        if c == "[":
            j = s.find("]", i + 2)
            out.append(_ROFF_CHARS.get(s[i + 2:j], "") if j >= 0 else "")
            i = j + 1 if j >= 0 else n
            continue
        if c in "e\\":
            out.append("\\")
        elif c == "-":
            out.append("-")
        elif c in "~0 ":
            out.append(" ")
        elif c in "'`":
            out.append(c)
        elif c in "hvwlLDxNXZoAbBCRSz":     # escapes with a delimited argument: drop the argument
            d = s[i + 2] if i + 2 < n else ""
            j = s.find(d, i + 3) if d else -1
            i = j + 1 if j >= 0 else n
            continue
        elif c in "&|^%:cdupt)/,!":
            pass                            # zero-width / hyphenation / continuation
        else:
            out.append(c)
        i += 2
    return "".join(out)


def _roff_args(rest: str) -> list[str]:
    args, cur, q, i = [], [], False, 0
    while i < len(rest):
        ch = rest[i]
        if ch == '"':
            if q and i + 1 < len(rest) and rest[i + 1] == '"':
                cur.append('"')
                i += 2
                continue
            q = not q
        elif ch in " \t" and not q:
            if cur:
                args.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        args.append("".join(cur))
    return args


def deroff(src: str) -> str | None:
    """man(7) page source -> plain text. Returns None for mdoc pages and .so redirects."""
    lines = src.splitlines()
    head = lines[:120]
    if any(ln.startswith((".Dd", ".Sh ", ".Nm")) for ln in head):
        return None
    if any(ln.startswith(".so ") for ln in head) and len(lines) < 40:
        return None
    out, para = [], []
    nofill = False
    in_table = False

    def flush():
        if para:
            t = "\n".join(para) if nofill else re.sub(r"[ \t]+", " ", " ".join(x for x in para if x)).strip()
            if t.strip():
                out.append(t.rstrip())
            para.clear()

    for raw in lines:
        if in_table:
            if raw.startswith(".TE"):
                in_table = False
            continue
        if raw.startswith(('.\\"', "'\\\"", ".\\#")) or raw.strip() in (".", "'"):
            continue
        if raw[:1] in ".'":
            body = raw[1:].lstrip()
            m, _, rest = body.partition(" ")
            args = _roff_args(rest)
            if m in ("SH", "SS"):
                flush()
                h = roff_inline(" ".join(args)).strip()
                if h:
                    out.append(h)
            elif m in ("B", "I", "SM", "SB"):
                para.append(roff_inline(" ".join(args)))
            elif m in ("BR", "RB", "IR", "RI", "BI", "IB"):
                para.append(roff_inline("".join(args)))
            elif m in ("PP", "LP", "P", "HP", "sp", "br", "RS", "RE", "TP", "TQ", "PD", "in", "ti", "ne"):
                flush()
            elif m == "IP":
                flush()
                if args:
                    para.append(roff_inline(args[0]))
            elif m in ("nf", "EX"):
                flush()
                nofill = True
            elif m in ("fi", "EE"):
                flush()
                nofill = False
            elif m == "TS":
                flush()
                in_table = True
            elif m == "so":
                return None
            continue                         # every other request/macro is layout: dropped
        para.append(roff_inline(raw))
    flush()
    text = "\n\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def man_pages(root: str) -> list[str]:
    out = []
    for sec in ("man1", "man5", "man7", "man8"):
        d = os.path.join(root, sec)
        if os.path.isdir(d):
            out += [os.path.join(sec, f) for f in sorted(os.listdir(d)) if f.endswith(".gz")]
    return out


def read_man(root: str, rel: str) -> str | None:
    try:
        raw = gzip.decompress(open(os.path.join(root, rel), "rb").read())
    except (OSError, EOFError):
        return None
    try:
        src = raw.decode("utf-8")
    except UnicodeDecodeError:
        src = raw.decode("latin-1")
    return deroff(src)


# --------------------------------------------------------------------------------------- tools

CITIES = ["Lisbon", "Osaka", "Toronto", "Nairobi", "Reykjavik", "Santiago", "Melbourne", "Krakow", "Hanoi",
          "Denver", "Edinburgh", "Montevideo", "Tallinn", "Kyoto", "Cape Town", "Vancouver", "Porto", "Seville"]
CONDITIONS = ["clear", "partly cloudy", "overcast", "light rain", "showers", "fog", "light snow", "windy"]
UNITS = [("km", "mi", 0.621371), ("kg", "lb", 2.20462), ("l", "gal", 0.264172), ("m", "ft", 3.28084),
         ("cm", "in", 0.393701)]
CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CAD", "AUD", "CHF", "SEK"]
BOOK_WORDS = ["Garden", "River", "Algorithms", "Winter", "Lighthouse", "Patterns", "Orchard", "Compiler",
              "Harbor", "Atlas", "Clockwork", "Meadow", "Signals", "Lantern", "Quiet", "Northern"]
AUTHORS = ["A. Rivera", "M. Okafor", "L. Chen", "S. Novak", "R. Haddad", "J. Lindqvist", "P. Moreau", "K. Tanaka"]
EVENTS = ["Team standup", "Dentist appointment", "Lunch with Sam", "Code review", "Gym", "Book club",
          "Project planning", "Piano lesson", "Grocery run"]
ITEMS = [("SKU-1042", "stainless water bottle"), ("SKU-2210", "wool socks"), ("SKU-3307", "desk lamp"),
         ("SKU-4471", "notebook, A5 dotted"), ("SKU-5583", "USB-C cable 2 m"), ("SKU-6619", "ceramic mug")]
RECIPES = {"pancakes": [("flour", 200, "g"), ("milk", 300, "ml"), ("eggs", 2, ""), ("sugar", 25, "g")],
           "tomato soup": [("tomatoes", 800, "g"), ("onion", 1, ""), ("stock", 500, "ml"), ("olive oil", 2, "tbsp")],
           "lentil curry": [("red lentils", 250, "g"), ("coconut milk", 400, "ml"), ("curry paste", 2, "tbsp"),
                            ("spinach", 100, "g")]}
FILES = ["README.md", "setup.cfg", "src/", "tests/", "docs/", "pyproject.toml", "LICENSE", "notes.txt",
         "data.csv", "Makefile"]

TOOL_SPECS = {
    "get_weather": ("Get the current weather for a city.",
                    {"city": {"type": "string", "description": "City name"},
                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    "convert_units": ("Convert a numeric value between units.",
                      {"value": {"type": "number"}, "from_unit": {"type": "string"},
                       "to_unit": {"type": "string"}}, ["value", "from_unit", "to_unit"]),
    "search_library": ("Search the city library catalog.",
                       {"query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1}}, ["query"]),
    "get_calendar_events": ("List calendar events on a date.",
                            {"date": {"type": "string", "description": "YYYY-MM-DD"}}, ["date"]),
    "convert_currency": ("Convert an amount between currencies using today's reference rate.",
                         {"amount": {"type": "number"}, "from_currency": {"type": "string"},
                          "to_currency": {"type": "string"}}, ["amount", "from_currency", "to_currency"]),
    "calculate": ("Evaluate an arithmetic expression.", {"expression": {"type": "string"}}, ["expression"]),
    "list_directory": ("List the files in a project directory.", {"path": {"type": "string"}}, ["path"]),
    "get_stock_level": ("Look up the stock level of an item in the shop inventory.",
                        {"sku": {"type": "string"}}, ["sku"]),
    "scale_recipe": ("Scale a recipe's ingredient list to a number of servings.",
                     {"recipe": {"type": "string"}, "servings": {"type": "integer", "minimum": 1}},
                     ["recipe", "servings"]),
}


def tool_schema(name: str) -> dict:
    desc, props, req = TOOL_SPECS[name]
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": req}}}


def _scenario(rng: random.Random, name: str):
    """-> (user text, arguments, tool result, thinking text, final answer)."""
    if name == "get_weather":
        city, unit = rng.choice(CITIES), rng.choice(["celsius", "fahrenheit"])
        t = round(rng.uniform(-4, 31), 1)
        t = t if unit == "celsius" else round(t * 9 / 5 + 32, 1)
        cond, hum, wind = rng.choice(CONDITIONS), rng.randint(25, 95), round(rng.uniform(0, 35), 1)
        user = rng.choice([f"What's the weather like in {city} right now?",
                           f"I'm heading out in {city} - how warm is it, in {unit}?",
                           f"Quick check: is it raining in {city}?"])
        res = {"city": city, "temperature": t, "unit": unit, "condition": cond, "humidity": hum, "wind_kph": wind}
        sym = "C" if unit == "celsius" else "F"
        return (user, {"city": city, "unit": unit}, res,
                f"The user wants the current weather in {city}. I'll call get_weather with unit {unit}.",
                f"It's currently {t}°{sym} and {cond} in {city}, with {hum}% humidity and wind around {wind} km/h.")
    if name == "convert_units":
        a, b, f = rng.choice(UNITS)
        if rng.random() < 0.5:
            a, b, f = b, a, 1 / f
        v = rng.choice([1, 2.5, 5, 10, 12, 42, 100, 250])
        r = round(v * f, 3)
        return (f"How many {b} is {v} {a}?", {"value": v, "from_unit": a, "to_unit": b},
                {"value": v, "from_unit": a, "to_unit": b, "result": r},
                f"This is a unit conversion from {a} to {b}; convert_units will give an exact figure.",
                f"{v} {a} is about {r} {b}.")
    if name == "search_library":
        w = rng.sample(BOOK_WORDS, 2)
        q = f"{w[0].lower()} {w[1].lower()}"
        books = [{"title": f"The {rng.choice(BOOK_WORDS)} of {rng.choice(BOOK_WORDS)}", "author": rng.choice(AUTHORS),
                  "year": rng.randint(1962, 2024), "available": rng.random() < 0.6} for _ in range(rng.randint(2, 4))]
        avail = [b for b in books if b["available"]]
        ans = (f"I found {len(books)} matches. Available now: " + "; ".join(f"{b['title']} by {b['author']} ({b['year']})"
                                                                         for b in avail) + "."
               if avail else f"I found {len(books)} matches, but none are on the shelf right now.")
        return (f"Can you look for books about {q} at the library?", {"query": q, "max_results": 5},
                {"query": q, "results": books}, f"I'll search the library catalog for '{q}'.", ans)
    if name == "get_calendar_events":
        d = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        evs = sorted([{"time": f"{rng.randint(8, 18):02d}:{rng.choice(['00', '15', '30', '45'])}",
                       "title": rng.choice(EVENTS)} for _ in range(rng.randint(1, 4))], key=lambda e: e["time"])
        return (f"What's on my calendar for {d}?", {"date": d}, {"date": d, "events": evs},
                f"The user asks about their schedule on {d}; I'll fetch the events for that date.",
                f"On {d} you have: " + ", ".join(f"{e['title']} at {e['time']}" for e in evs) + ".")
    if name == "convert_currency":
        a, b = rng.sample(CURRENCIES, 2)
        amt = rng.choice([20, 75, 120, 499.99, 1000, 2500])
        rate = round(rng.uniform(0.005, 160), 4)
        return (f"Roughly how much is {amt} {a} in {b}?", {"amount": amt, "from_currency": a, "to_currency": b},
                {"rate": rate, "converted": round(amt * rate, 2), "as_of": "reference rate, demo data"},
                f"Currency conversion {a} to {b}; I'll call convert_currency.",
                f"At the reference rate of {rate}, {amt} {a} is about {round(amt * rate, 2)} {b}.")
    if name == "calculate":
        x, y, z = rng.randint(12, 999), rng.randint(2, 99), rng.randint(2, 40)
        expr = f"({x} + {y}) * {z}"
        return (f"What is ({x} plus {y}) times {z}?", {"expression": expr}, {"expression": expr, "result": (x + y) * z},
                "Simple arithmetic, but I'll use the calculator to be exact.", f"({x} + {y}) × {z} = {(x + y) * z}.")
    if name == "list_directory":
        p = rng.choice(["~/projects/garden-planner", "~/projects/recipe-box", "~/notes", "~/projects/weather-cli"])
        ents = [{"name": f, "type": "dir" if f.endswith("/") else "file", "size": 0 if f.endswith("/") else
                 rng.randint(120, 48000)} for f in rng.sample(FILES, rng.randint(3, 6))]
        return (f"What's in {p}?", {"path": p}, {"path": p, "entries": ents},
                f"I'll list the directory {p}.",
                f"{p} contains " + ", ".join(e["name"] for e in ents) + ".")
    if name == "get_stock_level":
        sku, item = rng.choice(ITEMS)
        n = rng.randint(0, 140)
        return (f"Do we still have the {item} in stock?", {"sku": sku},
                {"sku": sku, "name": item, "in_stock": n, "warehouse": rng.choice(["north", "central", "harbor"])},
                f"The {item} is {sku}; I'll check the inventory.",
                f"We have {n} units of the {item} ({sku}) in stock." if n else f"The {item} ({sku}) is out of stock.")
    if name == "scale_recipe":
        rec = rng.choice(sorted(RECIPES))
        s = rng.randint(2, 12)
        ing = [{"item": i, "amount": round(q * s / 4, 1), "unit": u} for i, q, u in RECIPES[rec]]
        return (f"Scale my {rec} recipe (it serves 4) to {s} people.", {"recipe": rec, "servings": s},
                {"recipe": rec, "servings": s, "ingredients": ing},
                f"The base recipe serves 4; scale_recipe will compute amounts for {s}.",
                f"For {s} servings of {rec}: " + ", ".join(f"{x['amount']} {x['unit']} {x['item']}".replace("  ", " ")
                                                         for x in ing) + ".")
    raise KeyError(name)


def tool_conversation(rng: random.Random, n_turns: int):
    """Benign multi-turn tool-use conversation in OpenAI message format (system carries the tools)."""
    names = rng.sample(sorted(TOOL_SPECS), rng.randint(3, 5))
    system = {"role": "system", "content": rng.choice([
        "You are a helpful assistant with access to tools. Use them when they help, and answer concisely.",
        "You are a friendly home-and-office assistant. Call tools for facts you do not know.",
        "You help with everyday questions. Prefer calling a tool over guessing."]),
        "tools": [tool_schema(n) for n in names]}
    msgs = [system]
    call_no = 0
    for turn in range(n_turns):
        nm = rng.choice(names)
        user, args, res, think, ans = _scenario(rng, nm)
        if turn and rng.random() < 0.5:
            user = rng.choice(["Thanks! ", "Great. ", "One more thing: ", "Also, "]) + user[0].lower() + user[1:]
        msgs.append({"role": "user", "content": user})
        calls = [(nm, args, res)]
        if nm in ("get_weather", "get_stock_level", "convert_units") and rng.random() < 0.3:
            u2, a2, r2, _, ans2 = _scenario(rng, nm)
            calls.append((nm, a2, r2))
            ans = ans + " " + ans2
        tcs = []
        for cn, ca, _ in calls:
            call_no += 1
            tcs.append({"id": f"call_{call_no:03d}", "type": "function",
                        "function": {"name": cn, "arguments": json.dumps(ca)}})
        msgs.append({"role": "assistant", "content": "", "reasoning_content": think, "tool_calls": tcs})
        for tc, (_, _, cr) in zip(tcs, calls):
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(cr, ensure_ascii=False)})
        msgs.append({"role": "assistant", "content": ans,
                     "reasoning_content": "The tool returned what I need; I'll summarize it for the user."})
    return msgs


def load_encoder(path: str | None):
    """vLLM's DeepSeek-V4.1 chat encoder, loaded by file (stdlib-only module; vLLM is not imported)."""
    if path is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.origin:
            return None, None
        path = os.path.join(os.path.dirname(spec.origin), "tokenizers", "deepseek_v41_encoding.py")
    if not os.path.isfile(path):
        return None, None
    s = importlib.util.spec_from_file_location("dsv41_encoding", path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod, path


def render_chat(enc, msgs, thinking: bool) -> str:
    """Same steps as vllm/tokenizers/deepseek_v41.py apply_chat_template (effort 'high')."""
    m = copy.deepcopy(msgs)
    return enc.encode_messages(m, thinking_mode="thinking" if thinking else "chat", drop_thinking=True,
                               reasoning_effort="high")


def render_plain(msgs) -> str:
    """Fallback when the encoder file is missing: a JSON-lines transcript (still JSON/tool-call text)."""
    return "<｜begin▁of▁sentence｜>" + "\n".join(json.dumps(x, ensure_ascii=False) for x in msgs)


# ----------------------------------------------------------------------------------------- gen

GEN_PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list. Include a docstring and a short example.",
    "Implement a small LRU cache class in Python using collections.OrderedDict, with get and put methods.",
    "Write a C function that reverses a singly linked list in place, and explain how it works.",
    "Write a shell one-liner that counts the lines in all .py files under the current directory, then explain each part.",
    "Explain the difference between a process and a thread, with one example of when to use each.",
    "Write a Python generator that yields the prime numbers below n using a sieve, and state its complexity.",
    "Write a SQL query that returns the three most recent orders per customer from a table orders(id, customer_id, created_at).",
    "Explain what a hash table is and how it handles collisions, in two short paragraphs.",
    "Describe the water cycle for a middle-school student in about 150 words.",
    "Give three practical tips for writing clear commit messages, with an example of each.",
    "Summarize how public-key cryptography lets two people agree on a secret over an open channel, in plain language.",
    "Explain why the sky looks blue during the day and red at sunset.",
    "Write a JSON Schema for a library book with title, author, year, ISBN and a list of tags.",
    "Convert this to JSON with keys name, age and city: Ana is 34 and lives in Porto; Kenji is 29 and lives in Osaka.",
    "Write a short, friendly email reminding a colleague about tomorrow's 10:00 design review.",
    "List five common English idioms about time and explain what each means.",
]


def gen_prompts(gsm_path: str):
    items = [json.loads(x) for x in open(gsm_path) if x.strip()]
    qs = [items[i]["question"] for i in range(len(items) - 1, len(items) - 9, -1)]   # last 8 test items
    prompts = [("gen:prompt%02d" % i, p) for i, p in enumerate(GEN_PROMPTS)]
    prompts += [(f"gen:gsm8k:{len(items) - 1 - i}", "Solve the following math problem step by step. End with "
                 "'Answer: <number>'.\n\n" + q) for i, q in enumerate(qs)]
    return prompts


# ------------------------------------------------------------------------------------ assembly

class Tok:
    def __init__(self, model_dir: str):
        try:   # what vllm/tokenizers/deepseek_v41.py loads (transformers 5.x)
            from transformers import TokenizersBackend
            self.tok = TokenizersBackend.from_pretrained(model_dir)
        except ImportError:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.bos = self.tok.bos_token_id
        self.model_dir = model_dir
        self.raw = None
        try:
            import tokenizers
            self.raw = tokenizers.Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        except Exception as e:  # noqa: BLE001
            print(f"warning: raw tokenizers cross-check unavailable: {e}")
        self.checked = 0
        self.mismatch = 0

    def encode(self, text: str) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        if self.raw is not None and self.checked < 64:
            self.checked += 1
            if self.raw.encode(text, add_special_tokens=False).ids != ids:
                self.mismatch += 1
        return list(ids)

    def decode(self, ids) -> str:
        return self.tok.decode(ids)


def pack(docs, L: int, n_seq: int, tok, bos: int, allow_pack: bool = True, sep: str = "\n\n",
         chat: bool = False):
    """docs: iterable of (src, text). Each sequence starts at a document start; short documents are
    concatenated (allow_pack) or skipped; every sequence is cut to exactly L tokens."""
    seqs = []
    buf, srcs = None, []
    for src, text in docs:
        buf = text if buf is None else buf + sep + text
        srcs.append(src)
        ids = tok.encode(buf) if chat else [bos] + tok.encode(buf)
        if len(ids) >= L:
            seqs.append((ids[:L], list(srcs)))
            buf, srcs = None, []
            if len(seqs) == n_seq:
                break
        elif not allow_pack:
            buf, srcs = None, []
    return seqs


def build(args) -> int:
    out_dir = args.out
    mpath = os.path.join(out_dir, "manifest.json")
    if os.path.exists(mpath) and not args.force:
        print(f"refusing: {mpath} exists (runs must score identical tokens; --force only before any REF capture)")
        return 1
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    tok = Tok(args.model)
    bos = tok.bos
    if bos != 0:
        print(f"warning: BOS id is {bos}, expected 0")
    stdlib = args.stdlib or sysconfig.get_paths()["stdlib"]
    enc, enc_path = load_encoder(args.encoder)
    print(f"tokenizer {args.model} (bos {bos}); stdlib {stdlib}; chat encoder {enc_path or 'MISSING (plain fallback)'}")
    plan = dict(DEFAULT_PLAN)
    sets = {}
    sources = {}

    # long (first, so its files are excluded from code)
    _, n_long, L_long = plan["long"]
    long_seqs = []
    for fn in LONG_PY_FILES[:LONG_PY]:
        text = open(os.path.join(stdlib, fn), encoding="utf-8").read()
        long_seqs += pack([(f"stdlib:{fn}", text)], L_long, 1, tok, bos, allow_pack=False)
    topics = pydoc_topics(stdlib)
    tkeys = by_hash(sorted(topics))
    half = len(tkeys) // 2
    long_topics = sorted(tkeys[half:])           # alphabetical neighbours read like one document
    long_seqs += pack([(f"pydoc:{k}", topics[k]) for k in long_topics], L_long, LONG_DOC, tok, bos)
    sets["long"] = long_seqs[:n_long]

    # code
    _, n_code, L_code = plan["code"]
    py = [f for f in by_hash(stdlib_files(stdlib))
          if f not in PY_SKIP_FILES and os.path.getsize(os.path.join(stdlib, f)) >= 12_000]
    code = pack(((f"stdlib:{f}", code_window(open(os.path.join(stdlib, f), encoding="utf-8").read())) for f in py),
                L_code, CODE_PY, tok, bos, allow_pack=False)
    inc = by_hash(c_headers(args.include)) if os.path.isdir(args.include) else []

    def read_h(f):
        return open(os.path.join(args.include, f), encoding="utf-8", errors="replace").read()

    code += pack(((f"include:{f}", header_window(read_h(f))) for f in inc), L_code, n_code - len(code), tok, bos,
                 allow_pack=False)
    if len(code) < n_code:   # not enough headers: top up with more stdlib files
        used = {s for _, ss in code for s in ss}
        code += pack(((f"stdlib:{f}", code_window(open(os.path.join(stdlib, f), encoding="utf-8").read()))
                      for f in py if f"stdlib:{f}" not in used), L_code, n_code - len(code), tok, bos,
                     allow_pack=False)
    sets["code"] = code

    # math
    _, n_math, L_math = plan["math"]
    sets["math"] = pack(gsm8k_docs(args.gsm8k), L_math, n_math, tok, bos)

    # prose
    _, n_prose, L_prose = plan["prose"]
    prose = pack(((f"pydoc:{k}", topics[k]) for k in tkeys[:half]), L_prose, PROSE_DOC, tok, bos)
    mans = by_hash(man_pages(args.man)) if os.path.isdir(args.man) else []

    def man_docs():
        for rel in mans:
            t = read_man(args.man, rel)
            if t and len(t) >= 1500:
                yield f"man:{rel}", t

    prose += pack(man_docs(), L_prose, n_prose - len(prose), tok, bos)
    if len(prose) < n_prose:
        print(f"warning: only {len(prose)} prose sequences (man pages short?)")
    sets["prose"] = prose

    # tools
    _, n_tools, L_tools = plan["tools"]
    rng = random.Random(SEED)
    tools = []
    for i in range(n_tools):
        thinking = (i % 4) != 3                  # server default is thinking mode (effort high)
        turns = 2
        while True:
            r2 = random.Random(rng.random())
            msgs = tool_conversation(r2, turns)
            text = render_chat(enc, msgs, thinking) if enc else render_plain(msgs)
            ids = tok.encode(text)
            if len(ids) >= L_tools or turns >= 12:
                break
            turns += 1
        if ids[0] != bos:
            raise RuntimeError("chat encoder output does not start with BOS")
        if len(ids) >= L_tools:
            tools.append((ids[:L_tools], [f"tools:conv{i}:{'thinking' if thinking else 'chat'}:{turns}turns"]))
    sets["tools"] = tools

    # gen prompts (decode panel)
    gen = []
    for j, (src, p) in enumerate(gen_prompts(args.gsm8k)[: plan["gen"][1]]):
        thinking = j % 2 == 0
        msgs = [{"role": "user", "content": p}]
        text = render_chat(enc, msgs, thinking) if enc else ("<｜begin▁of▁sentence｜>" + p)
        gen.append((tok.encode(text), [f"{src}:{'thinking' if thinking else 'chat'}"]))
    sets["gen"] = gen

    # write
    man = {"builder_version": BUILDER_VERSION, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "builder_sha256": file_sha(os.path.abspath(__file__)), "seed": SEED,
           "tokenizer": {"model_dir": args.model, "class": type(tok.tok).__name__,
                         "tokenizer_json_sha256": file_sha(os.path.join(args.model, "tokenizer.json")),
                         "tokenizer_config_sha256": file_sha(os.path.join(args.model, "tokenizer_config.json")),
                         "bos_id": bos, "vocab_size": len(tok.tok), "cross_check_docs": tok.checked,
                         "cross_check_mismatches": tok.mismatch},
           "python": sys.version.split()[0], "chat_encoder": {"path": enc_path,
                                                             "sha256": file_sha(enc_path) if enc_path else None},
           "sources": {"stdlib": stdlib, "include": args.include, "man": args.man, "gsm8k": args.gsm8k,
                       "gsm8k_sha256": file_sha(args.gsm8k)},
           "sets": {}}
    try:
        import transformers
        man["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    total = 0
    previews = []
    for name in ("code", "math", "prose", "tools", "long", "gen"):
        panel = plan[name][0]
        rows = []
        for i, (ids, srcs) in enumerate(sets[name]):
            row = {"set": name, "idx": i, "n": len(ids), "sha256": seq_sha(ids), "src": srcs, "ids": ids}
            if args.preview_out:
                head = tok.decode(ids[:160]) if panel == "tf" else tok.decode(ids)
                previews.append(f"=== {name}[{i}] n={len(ids)} src={srcs[:3]}{' ...' if len(srcs) > 3 else ''}\n"
                                f"{head}\n")
            if panel == "gen":
                row["max_tokens"] = GEN_MAX_TOKENS
            rows.append(row)
        with open(os.path.join(out_dir, name + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        pos = sum(r["n"] - 1 for r in rows) if panel == "tf" else sum(r.get("max_tokens", 0) for r in rows)
        if panel == "tf":
            total += pos
        man["sets"][name] = {"panel": panel, "n_seq": len(rows), "seq_len": plan[name][2] or None,
                             "positions": pos, "sha256": set_sha(rows),
                             "src_files": sorted({s.split(":")[0] for r in rows for s in r["src"]})}
        print(f"  {name:6} {panel:3} {len(rows):3} seqs  {pos:>7,} {'positions' if panel == 'tf' else 'max new tokens'}")
    man["tf_positions"] = total
    man["files"] = {name + ".jsonl": file_sha(os.path.join(out_dir, name + ".jsonl")) for name in man["sets"]}
    with open(os.path.join(out_dir, "SHA256SUMS"), "w") as f:
        for fn, h in sorted(man["files"].items()):
            f.write(f"{h}  {fn}\n")
    with open(mpath, "w") as f:
        json.dump(man, f, indent=1)
    if args.preview_out:
        with open(args.preview_out, "w") as f:
            f.write("\n".join(previews))
        print(f"previews -> {args.preview_out}")
    if tok.mismatch:
        print(f"WARNING: transformers vs tokenizers disagreed on {tok.mismatch}/{tok.checked} documents")
    print(f"wrote {out_dir}: {total:,} teacher-forced positions in {time.time() - t0:.0f}s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser("~/models/dsv41/uncensored-fp8"))
    ap.add_argument("--out", default=os.path.join(HERE, "sets"))
    ap.add_argument("--gsm8k", default=os.path.expanduser("~/gsm8k_test.jsonl"))
    ap.add_argument("--stdlib", default=None, help="default: this interpreter's stdlib")
    ap.add_argument("--include", default="/usr/include")
    ap.add_argument("--man", default="/usr/share/man")
    ap.add_argument("--encoder", default=None, help="path to deepseek_v41_encoding.py (default: from the vllm install)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preview-out", default=None,
                    help="write decoded heads of every sequence here for a human check (not into sets/)")
    return build(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
