"""Synthetic agent-conversation generator (text, code, logs, JSON, CJK/emoji, tool calls,
tool results, reasoning, literal special-token strings).  Deterministic seed.
No real user data."""
import json
import random

R = random.Random(20261103)
WORDS = ("the of and to in is that for it as with was on be by this are from at or have an not which but all "
         "were when we there can been has more if will one would so no out up into than them only other new "
         "some could time these two may then first any like now my such make over our even most also after "
         "model token cache prefix request server render layer kernel tensor batch decode prefill latency "
         "throughput memory buffer offset stride block scheduler worker thread queue socket config patch "
         "install backup verify checksum rollback deploy service launcher overlay context window budget").split()
IDENT = ("alpha beta gamma delta block_size num_tokens prefix_len hidden kv_cache rank world_size offset "
         "stride chunk head dim weight scale bias logits ids mask seq_lens slot cursor").split()
CJK = "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女问力机给等几很业最间新什打便位因重被走电四第门相次东政海口使教西再平真听世气信北少关并内加化由却代军产入先山五太水万市眼体别处总才场师书比住员九笑性通目华报立马命张活难神数件安表原车白应路期叫死常提感金何更反合放做系计或司利受光王果亲界及今京务制解各任至清物台象记边共风战干接它许八特觉望直服毛林题建南度统色字请交爱让认算论百吃义科怎元社术结六功指思非流每青管夫连远资队跟带花快条院变联言权往展该领传近留红治决周保达办运武半候七必城父强步完革深区"
TRICKY = ["<think>", "</think>", "<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>", "｜DSML｜",
          "<｜DSML｜ invoke name=\"x\">", "<｜begin▁of▁sentence｜>", "<｜System｜>", "<tool_result>"]


def words(n: int) -> str:
    out = []
    for i in range(n):
        w = R.choice(WORDS)
        if i % 17 == 0:
            w = w.capitalize()
        out.append(w)
        if i % 13 == 12:
            out[-1] += R.choice([".", ",", ";", ":", "?"])
    return " ".join(out)


def code_block(lines: int) -> str:
    out = []
    ind = 0
    for _ in range(lines):
        r = R.random()
        a, b = R.choice(IDENT), R.choice(IDENT)
        if r < 0.15:
            out.append("    " * ind + f"def {a}_{b}(self, {b}: int = {R.randint(0, 4096)}) -> list[int]:")
            ind = min(ind + 1, 4)
        elif r < 0.3:
            out.append("    " * ind + f"# {words(R.randint(4, 12))}")
        elif r < 0.45:
            out.append("    " * ind + f"for {a} in range({R.randint(1, 999)}, {b}, {R.randint(1, 8)}):")
            ind = min(ind + 1, 4)
        elif r < 0.55:
            ind = max(ind - 1, 0)
            out.append("    " * ind + f"return {a}[{b}:{R.randint(0, 64)}]")
        elif r < 0.65:
            out.append("    " * ind + f'{a} = {{"{b}": {R.random():.6f}, "{a}": [{R.randint(0, 9)}, 0x{R.randint(0, 2**32):08x}]}}')
        else:
            out.append("    " * ind + f"{a}.{b}({R.choice(IDENT)}={R.randint(-100, 100000)}, dtype=torch.bfloat16)")
    return "\n".join(out)


def log_lines(n: int) -> str:
    out = []
    for i in range(n):
        out.append(f"2026-11-0{R.randint(1, 3)} {R.randint(0, 23):02d}:{R.randint(0, 59):02d}:{R.randint(0, 59):02d}."
                   f"{R.randint(0, 999999):06d} INFO [{R.choice(IDENT)}.py:{R.randint(1, 3000)}] "
                   f"{words(R.randint(3, 10))} pid={R.randint(1000, 4000000)} t={R.random() * 100:.3f}ms "
                   f"addr=0x{R.getrandbits(48):012x}")
    return "\n".join(out)


def cjk(n: int) -> str:
    return "".join(R.choice(CJK) for _ in range(n)) + "。"


def tool_result(target_chars: int) -> str:
    parts = []
    size = 0
    while size < target_chars:
        r = R.random()
        if r < 0.35:
            s = code_block(R.randint(10, 60))
        elif r < 0.6:
            s = log_lines(R.randint(5, 30))
        elif r < 0.8:
            s = json.dumps({R.choice(IDENT): [R.randint(0, 10**6) for _ in range(R.randint(3, 20))],
                            "path": f"/home/user/{R.choice(IDENT)}/{R.choice(IDENT)}.py",
                            "ok": R.random() < 0.8, "msg": words(R.randint(5, 30))}, indent=R.choice([None, 2]))
        elif r < 0.9:
            s = words(R.randint(30, 150))
        elif r < 0.95:
            s = cjk(R.randint(20, 120)) + " " + "🙂✅ é ü ß ñ"
        else:
            # literal special-token strings inside tool output (e.g. grep over tokenizer files)
            s = f"{R.choice(IDENT)}.py:{R.randint(1, 900)}: tok = \"{R.choice(TRICKY)}\"{R.choice(['', ' ', chr(10)])}{R.choice(TRICKY)}x"
        parts.append(s)
        size += len(s) + 1
    return "\n".join(parts)


TOOLS = []
for name, desc in [("bash", "Run a shell command"), ("read", "Read a file"), ("write", "Write a file"),
                   ("edit", "Exact string replacement in a file"), ("grep", "Search file contents"),
                   ("glob", "Find files by pattern"), ("web_fetch", "Fetch a URL"), ("todo", "Track tasks"),
                   ("ssh_exec", "Run a command on a remote host"), ("python", "Run Python"),
                   ("notebook", "Edit a notebook cell"), ("ask_user", "Ask the user a question")]:
    props = {}
    for _ in range(R.randint(2, 6)):
        p = R.choice(IDENT)
        props[p] = {"type": R.choice(["string", "integer", "boolean", "array"]), "description": words(R.randint(8, 40))}
    TOOLS.append({"type": "function", "function": {"name": name, "description": desc + ". " + words(R.randint(40, 160)),
                                                   "parameters": {"type": "object", "properties": props,
                                                                  "required": list(props)[:1]}}})
SYSTEM = "You are a coding agent.\n\n" + "\n\n".join(words(R.randint(60, 200)) for _ in range(30))
USER0 = "Please investigate the latency regression.\n\n" + words(300) + "\n\n" + code_block(40)
_call_n = [0]


def make_step() -> list[dict]:
    ncalls = 1 if R.random() < 0.8 else R.randint(2, 3)
    calls = []
    for _ in range(ncalls):
        _call_n[0] += 1
        cid = f"call_{_call_n[0]:05d}_{R.getrandbits(32):08x}"
        name = R.choice(TOOLS)["function"]["name"]
        args = {"command": f"grep -rn {R.choice(IDENT)} ~/src | head -{R.randint(5, 80)}"} if name in ("bash", "ssh_exec") \
            else {"path": f"/home/user/{R.choice(IDENT)}/{R.choice(IDENT)}.py", "limit": R.randint(10, 400)}
        if R.random() < 0.2:
            args["content"] = code_block(R.randint(5, 30))
        calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}})
    asst = {"role": "assistant", "content": words(R.randint(0, 60)) if R.random() < 0.4 else "",
            "reasoning": words(R.randint(30, 300)) + ("\n\n" + code_block(R.randint(2, 10)) if R.random() < 0.3 else ""),
            "tool_calls": calls}
    msgs = [asst]
    for c in calls:
        msgs.append({"role": "tool", "tool_call_id": c["id"], "content": tool_result(int(R.expovariate(1 / 2200)) + 150)})
    return msgs


def body_for(messages: list[dict], tools=True) -> bytes:
    b = {"model": "dsv41", "messages": messages, "stream": True, "max_tokens": 32768}
    if tools:
        b["tools"] = TOOLS
    return json.dumps(b).encode()


