#!/usr/bin/env python3
"""Long-context correctness probe for an OpenAI-compatible chat server (stdlib only).

For each (length, seed) it builds a non-repetitive haystack of procedurally generated
facts about fictional towns, plants K key->value facts at fixed depths and a uniquely
labelled 40-word passage at depth 50%, then asks (greedy, non-thinking):
  needle: "give the code for each of these K vaults"  -> exact recall per depth
  copy:   "quote archivist's note N verbatim"           -> word-sequence similarity / exact
Both questions share the same document prefix (prefix cache friendly).

Usage:
  python3 longctx.py [--base http://127.0.0.1:8000] [--lengths 4k,16k,32k,64k,128k,200k]
                     [--seeds 2] [--depths 5,25,50,75,95] [--conc 1] [--out results.jsonl]
                     [--min-needle 0.9] [--min-copy 0.9] [--pass-upto 128k]
Last stdout line: "LONGCTX PASS|FAIL ..." (exit code 0 on PASS, 1 on FAIL).
"""
import argparse, difflib, json, random, re, sys, threading, time, urllib.request

# ----------------------------------------------------------------- haystack text
SYL = ["ka", "lo", "ve", "mar", "tin", "sel", "dor", "an", "bri", "cus", "el", "fen", "gar", "hal", "is",
       "jor", "kel", "lun", "mo", "nor", "ost", "pra", "quel", "ros", "sun", "tar", "ul", "vin", "wen", "yr",
       "zan", "bel", "cor", "dra", "esh", "fal", "gol", "hen", "ith", "lar", "mel", "nim", "or", "pel", "ra"]
GOODS = ["copper bells", "salt cod", "blue glass", "walnut furniture", "wool blankets", "clockwork toys",
         "pickled plums", "river pearls", "slate roofing", "linen sails", "honey wine", "iron nails",
         "paper lanterns", "cheese wheels", "tin whistles", "cedar boats", "dyed silk", "maps and charts",
         "barley beer", "glazed pottery", "hand-carved chess sets", "smoked eels", "brass telescopes"]
FEATURES = ["a leaning stone tower", "a covered market built over the river", "an observatory on the hill",
            "seven arched bridges", "a library with a painted ceiling", "a lighthouse painted in red stripes",
            "a botanical garden of desert plants", "a fortress converted into a school", "hot springs",
            "a clock that runs backwards", "a cathedral with no bells", "terraced vineyards",
            "a canal lined with lime trees", "a museum of musical instruments", "an abandoned silver mine"]
DIRS = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
SEASONS = ["spring", "summer", "autumn", "winter"]
JOBS = ["fishing", "weaving", "mining", "shipbuilding", "printing", "glassmaking", "beekeeping", "brewing",
        "tanning", "clockmaking", "stonecutting", "rope making"]
WEATHER = ["foggy mornings", "sudden hailstorms", "long dry summers", "gentle sea breezes", "heavy snowfall",
           "warm southern winds", "frequent thunderstorms", "mild winters"]
FIRST = ["Ada", "Bram", "Celia", "Doran", "Elsa", "Feliks", "Greta", "Hugo", "Ines", "Jonas", "Kira", "Lev",
         "Mira", "Nils", "Odile", "Piet", "Rhea", "Sven", "Talia", "Ulrich", "Vera", "Wim", "Yara", "Zeno"]


def name(rng, n=None):
    n = n or rng.choice([2, 2, 3])
    return "".join(rng.choice(SYL) for _ in range(n)).capitalize()


def sentence(rng):
    a, b = name(rng), name(rng)
    t = rng.randrange(14)
    if t == 0:
        return f"The town of {a}, founded in {rng.randint(1100, 1899)}, lies {rng.randint(3, 900)} km {rng.choice(DIRS)} of {b}."
    if t == 1:
        return f"{a} is best known for its {rng.choice(GOODS)}, which merchants carry as far as {b}."
    if t == 2:
        return f"At the last census {a} counted {rng.randint(800, 990000):,} inhabitants, most of them working in {rng.choice(JOBS)}."
    if t == 3:
        return f"Visitors to {a} usually come to see {rng.choice(FEATURES)}."
    if t == 4:
        return f"The road from {a} to {b} climbs {rng.randint(40, 2900)} metres and takes about {rng.randint(2, 30)} hours on foot."
    if t == 5:
        return f"In {rng.choice(SEASONS)} the people of {a} hold a festival in honour of {rng.choice(FIRST)} {name(rng)}, who once {rng.choice(['rebuilt the harbour', 'mapped the coast', 'founded the guild of ' + rng.choice(JOBS), 'saved the town from a flood', 'wrote its first charter'])}."
    if t == 6:
        return f"{a} has {rng.choice(WEATHER)} and an average of {rng.randint(20, 240)} rainy days per year."
    if t == 7:
        return f"The mayor of {a}, {rng.choice(FIRST)} {name(rng)}, was elected in {rng.randint(1950, 2024)} with {rng.randint(51, 89)} percent of the vote."
    if t == 8:
        return f"A ferry links {a} with the island of {b} {rng.randint(2, 14)} times a day, except on {rng.choice(['Sundays', 'Mondays', 'feast days', 'stormy days'])}."
    if t == 9:
        return f"The river {name(rng)} flows through {a} before joining the sea near {b}, {rng.randint(5, 400)} km downstream."
    if t == 10:
        return f"Local cooks in {a} are proud of a stew made with {rng.choice(['lentils', 'chestnuts', 'mussels', 'turnips', 'smoked pork', 'wild mushrooms'])} and {rng.choice(['cider', 'saffron', 'juniper', 'dill', 'black pepper', 'sour cream'])}."
    if t == 11:
        return f"The old guild hall of {a} was rebuilt {rng.randint(2, 6)} times, most recently in {rng.randint(1700, 2015)} after a fire."
    if t == 12:
        return f"Children in {a} learn the {rng.choice(['fiddle', 'flute', 'accordion', 'drum', 'harp'])} from the age of {rng.randint(4, 11)}, and the town orchestra has {rng.randint(12, 140)} members."
    return f"Trade between {a} and {b} grew by {rng.randint(2, 60)} percent after the {rng.choice(['bridge', 'railway', 'canal', 'new port', 'mountain tunnel'])} opened in {rng.randint(1820, 2020)}."


def paragraph(rng):
    return " ".join(sentence(rng) for _ in range(rng.randint(3, 7)))


COLORS = ["amber", "cobalt", "crimson", "ivory", "jade", "onyx", "saffron", "teal", "violet", "scarlet",
          "silver", "umber", "indigo", "coral", "olive"]
ANIMALS = ["falcon", "otter", "lynx", "heron", "badger", "marten", "ibex", "condor", "gecko", "walrus",
           "bison", "puffin", "jackal", "tapir", "osprey"]
PASSAGE_WORDS = ("lantern quiet harbor stitched ledger crimson orchard whistle granite meadow ferry copper "
                 "almanac thimble cobbled winter beacon saddle orchard velvet compass hollow ember tidewater "
                 "parchment willow kettle barrow glimmer anvil rook spindle marrow fable sextant juniper "
                 "brine loom quarry lattice cinder harrow tallow ripple").split()


def make_passage(rng):
    # 40 words of plausible but unpredictable prose (not memorizable).
    subj = [f"{rng.choice(FIRST)} {name(rng)}", f"the keeper of {name(rng)}", f"a cartographer from {name(rng)}"]
    parts = [
        f"On the {rng.randint(2, 28)}th night of {rng.choice(['March', 'April', 'October', 'November'])}, {rng.choice(subj)} wrote that",
        f"the {rng.choice(PASSAGE_WORDS)} near the {rng.choice(PASSAGE_WORDS)} had turned {rng.choice(['green', 'grey', 'gold', 'black'])},",
        f"that {rng.randint(3, 97)} {rng.choice(['barrels', 'lanterns', 'sheep', 'letters', 'oars'])} were missing,",
        f"and that nobody in {name(rng)} would speak of the {rng.choice(PASSAGE_WORDS)} until the {rng.choice(PASSAGE_WORDS)} returned",
        f"with a {rng.choice(PASSAGE_WORDS)} of {rng.choice(PASSAGE_WORDS)} and {rng.choice(PASSAGE_WORDS)} for the {rng.choice(PASSAGE_WORDS)} {rng.choice(PASSAGE_WORDS)} {rng.choice(PASSAGE_WORDS)}",
    ]
    words = " ".join(parts).split()
    while len(words) < 40:
        words.append(rng.choice(PASSAGE_WORDS))
    words = words[:40]
    return " ".join(words).rstrip(",") + "."


# ----------------------------------------------------------------- HTTP
def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def count_tokens(base, model, messages, kwargs):
    r = post(base, "/tokenize", {"model": model, "messages": messages, "chat_template_kwargs": kwargs})
    return r["count"]


def chat(base, model, messages, max_tokens, kwargs):
    # ctx1m variant: streaming, records TTFT (first content/reasoning delta) in usage["ttft_s"].
    t0 = time.time()
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": kwargs, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    parts, usage, fin, ttft = [], {}, None, None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                c = d.get("content") or ""
                if (c or d.get("reasoning_content") or d.get("reasoning")) and ttft is None:
                    ttft = time.time() - t0
                parts.append(c)
                if ch.get("finish_reason"):
                    fin = ch["finish_reason"]
    dt = time.time() - t0
    usage = dict(usage or {}); usage["ttft_s"] = round(ttft, 2) if ttft is not None else None
    text = "".join(parts).split("</think>")[-1]
    return text, usage, fin, dt


# ----------------------------------------------------------------- document
def parse_len(s):
    s = s.strip().lower()
    return int(float(s[:-1]) * 1000) if s.endswith("k") else int(s)


def build_doc(target_tokens, seed, depths, base, model, kwargs, tpp=None, n_par=None):
    rng = random.Random(seed * 1000003 + target_tokens)
    keys, vals, used = [], [], set()
    while len(keys) < len(depths):
        k = f"{rng.choice(COLORS)}-{rng.choice(ANIMALS)}".upper()
        if k in used:
            continue
        used.add(k)
        keys.append(k)
        vals.append(f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}")
    note_id = rng.randint(100, 999)
    passage = make_passage(rng)
    # Estimate paragraphs needed (tokens per paragraph calibrated on a sample if not given).
    if tpp is None:
        sample = "\n\n".join(paragraph(random.Random(1)) for _ in range(40))
        tpp = count_tokens(base, model, [{"role": "user", "content": sample}], kwargs) / 40.0
    overhead = 400
    if n_par is None:
        n_par = max(4, int((target_tokens - overhead) / tpp))
    paras = [paragraph(rng) for _ in range(n_par)]
    # Insert planted facts at depth fractions; and the archivist note at 50% (after the 50% needle).
    inserts = []
    for d, k, v in zip(depths, keys, vals):
        inserts.append((d / 100.0, 0, f"Security registry entry: the access code for the vault named {k} is {v}. Keep this code exact."))
    inserts.append((0.5, 1, f"Archivist's note {note_id}: \"{passage}\""))
    for frac, sub, text in sorted(inserts, key=lambda x: (-x[0], -x[1])):
        pos = min(len(paras), max(0, round(frac * len(paras))))
        paras.insert(pos, text)
    doc = "\n\n".join(paras)
    return doc, keys, vals, note_id, passage, tpp, n_par


HEADER = ("Below is a long gazetteer of towns. Somewhere inside it are security registry entries and "
          "archivist's notes. Read it carefully; you will be asked about it afterwards.\n\n<document>\n")


def needle_question(keys):
    return ("\n</document>\n\nUsing only the document above, give the access code for each of these vaults: "
            + ", ".join(keys) + ". Answer with exactly one line per vault in the form NAME: CODE, "
            "in the order given, and nothing else.")


def copy_question(note_id):
    return (f"\n</document>\n\nUsing only the document above, quote the full text of archivist's note {note_id} "
            "verbatim, word for word, without the quotation marks and without any other text.")


def score_needles(text, keys, vals):
    """A needle counts only if its exact value appears on a line naming its key."""
    norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    lines = [norm(ln) for ln in text.splitlines()]
    return [any(norm(k) in ln and norm(v) in ln for ln in lines) for k, v in zip(keys, vals)]


def norm_words(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def score_copy(text, passage):
    # Ignore an echoed "Archivist's note N:" label and surrounding quotes.
    text = re.sub(r"^\s*archivist'?s note \d+\s*:?\s*", "", text.strip(), flags=re.I)
    a, b = norm_words(passage), norm_words(text)
    sim = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    return sim, a == b


# ----------------------------------------------------------------- main
def run_case(args, L, seed, kwargs, tpp_cache, out_rows, lock):
    doc, keys, vals, note_id, passage, tpp, n_par = build_doc(L, seed, args.depths, args.base, args.model,
                                                              kwargs, tpp_cache.get("tpp"))
    m1 = [{"role": "user", "content": HEADER + doc + needle_question(keys)}]
    ntok = count_tokens(args.base, args.model, m1, kwargs)
    for _ in range(2):  # rescale paragraph count until within 2% of the target
        if abs(ntok - L) <= 0.02 * L:
            break
        n_par = max(4, int(n_par * (L - 200) / max(1, ntok - 200)))
        doc, keys, vals, note_id, passage, tpp, n_par = build_doc(L, seed, args.depths, args.base, args.model,
                                                                  kwargs, tpp, n_par)
        m1 = [{"role": "user", "content": HEADER + doc + needle_question(keys)}]
        ntok = count_tokens(args.base, args.model, m1, kwargs)
    row = {"target": L, "seed": seed, "prompt_tokens": ntok}
    if ntok + 300 > args.max_model_len:
        row["skipped"] = "too long"
    else:
        t_case0 = time.time()
        text, usage, fin, dt = chat(args.base, args.model, m1, 300, kwargs)
        hits = score_needles(text, keys, vals)
        row.update(needle_hits=hits, needle_text=text[:600], needle_s=round(dt, 1), needle_finish=fin,
                   needle_ttft_s=usage.get("ttft_s"), needle_usage=usage,
                   prefill_tok_s=round(ntok / usage["ttft_s"], 1) if usage.get("ttft_s") else None,
                   t_start=round(t_case0, 1), t_end_needle=round(time.time(), 1))
        if not args.no_copy:
            m2 = [{"role": "user", "content": HEADER + doc + copy_question(note_id)}]
            text2, usage2, fin2, dt2 = chat(args.base, args.model, m2, 200, kwargs)
            sim, exact = score_copy(text2, passage)
            row.update(copy_sim=round(sim, 4), copy_exact=exact, copy_text=text2[:600], copy_ref=passage,
                       copy_s=round(dt2, 1), copy_ttft_s=usage2.get("ttft_s"), copy_usage=usage2,
                       t_end=round(time.time(), 1))
    with lock:
        out_rows.append(row)
        hs = "".join("1" if h else "0" for h in row.get("needle_hits", []))
        print(f"len~{L:>7} seed={seed} tok={ntok:>7} needles[{','.join(map(str, args.depths))}]={hs or '-'} "
              f"copy_sim={row.get('copy_sim', '-')} exact={row.get('copy_exact', '-')} "
              f"t={row.get('needle_s', '-')}s/{row.get('copy_s', '-')}s ttft={row.get('needle_ttft_s', '-')}s "
              f"prefill={row.get('prefill_tok_s', '-')}tok/s", flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")


def summarize(args, lengths, rows, upto, t0):
    print("\n== recall by length x depth (hits/trials) ==")
    print("length   tokens  " + "  ".join(f"d{d:>3}%" for d in args.depths) + "   needle   copy_sim  copy_exact")
    fails = []
    for L in lengths:
        rs = [r for r in rows if r["target"] == L and "needle_hits" in r]
        errs = [r for r in rows if r["target"] == L and "error" in r]
        if not rs:
            print(f"{L:>7}  (no data{', errors=' + str(len(errs)) if errs else ''})")
            if L <= upto:
                fails.append(f"{L}:nodata")
            continue
        per_d = [sum(r["needle_hits"][i] for r in rs) for i in range(len(args.depths))]
        rec = sum(per_d) / (len(rs) * len(args.depths))
        cs = [r["copy_sim"] for r in rs if "copy_sim" in r]
        ce = sum(1 for r in rs if r.get("copy_exact"))
        csim = sum(cs) / len(cs) if cs else float("nan")
        toks = sum(r["prompt_tokens"] for r in rs) // len(rs)
        print(f"{L:>7} {toks:>7}  " + "  ".join(f"{h}/{len(rs):<4}" for h in per_d)
              + f"   {rec:6.1%}   {csim:7.3f}   {ce}/{len(cs)}" + (f"  errors={len(errs)}" if errs else ""))
        if L <= upto:
            if rec < args.min_needle:
                fails.append(f"{L}:needle={rec:.2f}")
            if cs and csim < args.min_copy:
                fails.append(f"{L}:copy={csim:.2f}")
            if errs:
                fails.append(f"{L}:errors={len(errs)}")
    allr = [r for r in rows if "needle_hits" in r]
    tot = sum(sum(r["needle_hits"]) for r in allr)
    n = sum(len(r["needle_hits"]) for r in allr)
    cs = [r["copy_sim"] for r in allr if "copy_sim" in r]
    verdict = "PASS" if not fails else "FAIL"
    print(f"LONGCTX {verdict} needle={tot}/{n} copy_mean={sum(cs) / max(1, len(cs)):.3f} "
          f"gate(<= {args.pass_upto}: needle>={args.min_needle}, copy>={args.min_copy}) "
          f"fails={';'.join(fails) or 'none'} wall={time.time() - t0:.0f}s")
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--lengths", default="4k,16k,32k,64k,128k,200k")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--depths", default="5,25,50,75,95")
    ap.add_argument("--conc", type=int, default=1, help="cases run concurrently")
    ap.add_argument("--no-copy", action="store_true")
    ap.add_argument("--thinking", action="store_true", help="leave thinking on (default: off)")
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--out", default="")
    ap.add_argument("--min-needle", type=float, default=0.9, help="min needle recall per length <= --pass-upto")
    ap.add_argument("--min-copy", type=float, default=0.9, help="min mean copy similarity per length <= --pass-upto")
    ap.add_argument("--rescore", default="", help="re-score an existing --out jsonl (copy metric) and print the summary")
    ap.add_argument("--pass-upto", default="128k", help="lengths above this are reported but not gated")
    args = ap.parse_args()
    args.depths = [int(x) for x in args.depths.split(",")]
    lengths = [parse_len(x) for x in args.lengths.split(",")]
    upto = parse_len(args.pass_upto)
    kwargs = {"thinking": bool(args.thinking), "enable_thinking": bool(args.thinking)}
    if args.rescore:
        rows = [json.loads(l) for l in open(args.rescore) if l.strip()]
        for r in rows:
            if "copy_ref" in r:
                r["copy_sim"], r["copy_exact"] = score_copy(r["copy_text"], r["copy_ref"])
        lengths = sorted({r["target"] for r in rows})
        sys.exit(0 if summarize(args, lengths, rows, upto, time.time()) == "PASS" else 1)
    cases = [(L, s) for L in lengths for s in range(args.seed0, args.seed0 + args.seeds)]
    rows, lock, tpp_cache = [], threading.Lock(), {}
    # calibrate once (serially) so concurrent workers share it
    sample = "\n\n".join(paragraph(random.Random(1)) for _ in range(40))
    tpp_cache["tpp"] = count_tokens(args.base, args.model, [{"role": "user", "content": sample}], kwargs) / 40.0
    t0 = time.time()
    it = iter(cases)

    def worker():
        while True:
            with lock:
                c = next(it, None)
            if c is None:
                return
            try:
                run_case(args, c[0], c[1], kwargs, tpp_cache, rows, lock)
            except Exception as e:  # keep going; record the failure
                with lock:
                    rows.append({"target": c[0], "seed": c[1], "error": repr(e)[:300]})
                    print(f"len~{c[0]} seed={c[1]} ERROR {e!r}"[:300], flush=True)

    ths = [threading.Thread(target=worker) for _ in range(max(1, args.conc))]
    [t.start() for t in ths]
    [t.join() for t in ths]

    verdict = summarize(args, lengths, rows, upto, t0)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
