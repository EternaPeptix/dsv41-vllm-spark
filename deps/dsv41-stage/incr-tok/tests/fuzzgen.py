"""Synthetic conversations for the incr-tok equality fuzz. Stdlib only, deterministic per seed, no real user data.

Content: prose, code, logs, JSON, CJK, emoji (incl. ZWJ sequences), RTL, combining marks, CRLF, whitespace runs, digits,
literal special-token strings (all cut tokens, random added tokens, ｜DSML｜ chains) and partial special-token-like text.
Structure: system (+ tools), a first user turn, then steps. Each step ends with a tool result or a user message, so any
prefix of whole steps is a valid request. Ops between requests: grow, edit a middle message, edit the system message,
roll back steps, resend, compact history, switch chat mode.
"""
from __future__ import annotations

import copy
import json
import random

CUTS = ["<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>", "<think>"]
PARTIAL = ["<｜Use", "r｜>", "<thi", "nk>", "</think", "<think", "</thin", "<｜end▁of▁sentence", "▁of▁sentence｜>",
           "｜>", "<｜", "｜", "｜DSML", "DSML｜", "<｜Assistant", "Assistant｜>", "<｜begin▁of", "<tool_result>",
           "</tool_result>", "<｜｜>", "<<think>>", "<｜User｜", "｜User｜>"]
WORDS = ("the of and to in is that for it as with was on be by this are from at or have an not which but all were when "
         "model token cache prefix request server render layer kernel tensor batch decode prefill latency throughput "
         "memory buffer offset stride block scheduler worker thread queue socket config patch install backup verify").split()
IDENT = "alpha beta gamma delta block_size num_tokens prefix_len hidden kv_cache rank world_size offset stride".split()
CJK = "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本見経頭面公同三已老従動両長知民様現分将外但身些与高意進把法此実回二理美点月明其種声全工己話児者向情部正名定女問力機給等幾很業最間新什打便位因重被走電四第門相次東政海口使教西再平真聴世気信北少関并内加化由却代軍産入先山五太水万市眼体別処総才場師書比住員九笑性通目華報立馬命張活難神数件安表原車白応路期叫死常提感金何更反合放做系計或司利受光王果親界及今京務制解各任至清物台象記辺共風戦干接它許八特覚望直服毛林題建南度統色字請交愛譲認算論百吃義科怎元社術結六功指思非流毎青管夫連遠資隊跟帯花快条院変聯言権往展該領伝近留紅治決周保達弁運武半候七必城父強歩完革深区"
EMOJI = ["🙂", "✅", "🚀", "👩‍💻", "👨‍👩‍👧‍👦", "🏳️‍🌈", "🇯🇵", "❤️", "1️⃣", "🤖", "⚠️"]
RTL = ["مرحبا بالعالم", "שלום עולם", "سلام دنیا", "‏RTL mark‏"]
COMBINING = ["é", "ä", "ñ", "ộ", "Z̵̡", "क्ष"]
WS = [" ", "  ", "    ", "\t", "\n", "\n\n", "\n\n\n", "\r\n", " \n", "\n ", "  \n\n  ", "　", " "]


class Gen:
    def __init__(self, seed: int, added_tokens: list[str]):
        self.R = random.Random(seed)
        self.added = added_tokens
        self.call_n = 0

    # ------------------------------------------------------------------ text
    def words(self, n: int) -> str:
        R = self.R
        return " ".join(R.choice(WORDS).capitalize() if i % 11 == 0 else R.choice(WORDS) for i in range(n)) + R.choice([".", "", "?", ":"])

    def code(self, lines: int) -> str:
        R = self.R
        out, ind = [], 0
        for _ in range(lines):
            a, b = R.choice(IDENT), R.choice(IDENT)
            r = R.random()
            if r < 0.2:
                out.append("    " * ind + f"def {a}_{b}(self, {b}: int = {R.randint(0, 4096)}) -> list[int]:")
                ind = min(ind + 1, 4)
            elif r < 0.35:
                out.append("    " * ind + f"# {self.words(R.randint(3, 9))}")
            elif r < 0.5:
                ind = max(ind - 1, 0)
                out.append("    " * ind + f"return {a}[{b}:{R.randint(0, 64)}]")
            elif r < 0.6:
                out.append("\t" * ind + f'{a} = {{"{b}": {R.random():.6f}, "x": 0x{R.getrandbits(32):08x}}}')
            else:
                out.append("    " * ind + f"{a}.{b}({R.choice(IDENT)}={R.randint(-100, 100000)})  # {R.choice(CUTS + PARTIAL) if R.random() < 0.1 else ''}")
        return "\n".join(out)

    def fragment(self) -> str:
        R = self.R
        r = R.random()
        if r < 0.22:
            return self.words(R.randint(5, 60))
        if r < 0.36:
            return self.code(R.randint(3, 25))
        if r < 0.44:
            return "\n".join(f"2026-11-03 {R.randint(0, 23):02d}:{R.randint(0, 59):02d}:{R.randint(0, 59):02d} INFO "
                             f"[{R.choice(IDENT)}.py:{R.randint(1, 3000)}] {self.words(R.randint(2, 8))} t={R.random() * 100:.3f}ms"
                             for _ in range(R.randint(1, 12)))
        if r < 0.52:
            return json.dumps({R.choice(IDENT): [R.randint(0, 10**6) for _ in range(R.randint(1, 8))], "msg": self.words(R.randint(2, 12)),
                               "ok": R.random() < 0.5}, indent=R.choice([None, 2]), ensure_ascii=R.random() < 0.5)
        if r < 0.58:
            return "".join(R.choice(CJK) for _ in range(R.randint(5, 80))) + R.choice(["。", "，", "", "！"])
        if r < 0.62:
            return " ".join(R.choice(EMOJI) for _ in range(R.randint(1, 8)))
        if r < 0.65:
            return R.choice(RTL) + " " + "".join(R.choice(COMBINING) for _ in range(R.randint(1, 6)))
        if r < 0.71:
            return "".join(R.choice(WS) for _ in range(R.randint(1, 6)))
        if r < 0.75:
            return str(R.randint(0, 10**R.randint(1, 12)))
        if r < 0.83:  # literal special-token strings
            k = R.random()
            if k < 0.6:
                return R.choice(CUTS)
            if k < 0.8:
                return R.choice(self.added)
            return "｜DSML｜" * R.randint(1, 4) + R.choice(["", "DSML｜", "｜"])
        return R.choice(PARTIAL)

    def text(self, n_chars: int) -> str:
        R = self.R
        parts, size = [], 0
        while size < n_chars:
            f = self.fragment()
            parts.append(f)
            size += len(f) + 1
        joiner = R.choice(["", " ", "\n", "\n\n"])
        s = joiner.join(parts)
        if R.random() < 0.3:  # tricky edges
            s = R.choice(PARTIAL + CUTS + WS) + s
        if R.random() < 0.3:
            s = s + R.choice(PARTIAL + CUTS + WS)
        return s

    def content(self, n_chars: int):
        if self.R.random() < 0.12:  # content parts, joined with "\n\n" by the V4.1 normalizer
            return [{"type": "text", "text": self.text(max(1, n_chars // 2))} for _ in range(self.R.randint(1, 3))]
        return self.text(n_chars)

    # ------------------------------------------------------------------ conversation pieces
    def tools(self) -> list[dict]:
        R = self.R
        out = []
        for name in R.sample(["bash", "read", "write", "edit", "grep", "glob", "web_fetch", "todo", "ssh_exec", "python"], R.randint(1, 6)):
            props = {R.choice(IDENT): {"type": R.choice(["string", "integer", "boolean"]), "description": self.words(R.randint(3, 20))}
                     for _ in range(R.randint(1, 4))}
            out.append({"type": "function", "function": {"name": name, "description": self.words(R.randint(10, 60)),
                                                         "parameters": {"type": "object", "properties": props, "required": list(props)[:1]}}})
        return out

    def step(self, conv: "Conv") -> list[dict]:
        R = self.R
        size = int(R.expovariate(1 / 900)) + 20
        if conv.tools and R.random() < 0.8:
            calls = []
            for _ in range(1 if R.random() < 0.75 else R.randint(2, 3)):
                self.call_n += 1
                cid = f"call_{self.call_n:06d}_{R.getrandbits(24):06x}"
                args = {"command": self.text(R.randint(5, 200))}
                if R.random() < 0.2:
                    args["content"] = self.code(R.randint(2, 20))
                calls.append({"id": cid, "type": "function", "function": {"name": R.choice(conv.tools)["function"]["name"],
                                                                         "arguments": json.dumps(args, ensure_ascii=R.random() < 0.5)}})
            a = {"role": "assistant", "content": self.text(R.randint(1, 300)) if R.random() < 0.4 else "", "tool_calls": calls}
            if R.random() < 0.85:
                a["reasoning_content" if R.random() < 0.5 else "reasoning"] = self.text(R.randint(20, 1200))
            msgs = [a] + [{"role": "tool", "tool_call_id": c["id"], "content": self.content(size)} for c in calls]
            if R.random() < 0.1:
                msgs.append({"role": "user", "content": self.content(R.randint(10, 600))})
            return msgs
        a = {"role": "assistant", "content": self.text(R.randint(10, 1500))}
        if R.random() < 0.8:
            a["reasoning_content" if R.random() < 0.5 else "reasoning"] = self.text(R.randint(20, 1200))
        msgs = [a]
        if conv.mode == "mid_system" and R.random() < 0.3:
            msgs.append({"role": "system", "content": "reminder: " + self.text(R.randint(10, 300))})
        msgs.append({"role": "user", "content": self.content(R.randint(10, size))})
        return msgs


MODES = {  # name: (tools, chat_template_kwargs)
    "tools_thinking": (True, {}),
    "tools_effort_none": (True, {"reasoning_effort": "none"}),
    "tools_effort_max": (True, {"reasoning_effort": "max"}),
    "notools_thinking_drop": (False, {}),
    "notools_thinking_keep": (False, {"drop_thinking": False}),
    "mid_system": (True, {}),
    "plain_chat": (False, {"thinking": False}),
}


class Conv:
    def __init__(self, g: Gen, shared_system: str | None, shared_tools: list | None):
        R = g.R
        self.g = g
        self.mode = R.choice(list(MODES))
        use_tools, self.ctk = MODES[self.mode]
        self.ctk = dict(self.ctk)
        self.tools = (shared_tools or g.tools()) if use_tools else None
        self.system = shared_system if shared_system is not None else g.text(R.randint(300, 2500))
        self.user0 = g.content(R.randint(50, 1500))
        self.steps: list[list[dict]] = []
        self.target_chars = R.choice([6000, 12000, 24000, 48000, 120000])
        self.requests_left = R.randint(3, 12)
        self.last_op = "new"

    def messages(self) -> list[dict]:
        out = [{"role": "system", "content": self.system}, {"role": "user", "content": self.user0}]
        for s in self.steps:
            out.extend(s)
        return copy.deepcopy(out)

    def body(self) -> dict:
        b = {"model": "dsv41", "messages": self.messages(), "max_tokens": 32768, "stream": True}
        if self.tools:
            b["tools"] = self.tools
        if self.ctk:
            b["chat_template_kwargs"] = self.ctk
        return b

    def chars(self) -> int:
        return len(json.dumps(self.messages(), ensure_ascii=False))

    def advance(self) -> str:
        """Apply one op; returns its name."""
        g, R = self.g, self.g.R
        r = R.random()
        if not self.steps or r < 0.66:
            for _ in range(R.randint(1, 2)):
                self.steps.append(g.step(self))
            op = "grow"
        elif r < 0.72:
            s = R.randrange(len(self.steps))
            m = R.choice(self.steps[s])
            if isinstance(m.get("content"), str):
                p = R.randint(0, len(m["content"]))
                m["content"] = m["content"][:p] + R.choice(["X", " edited ", R.choice(CUTS), R.choice(PARTIAL), "\n"]) + m["content"][p:]
            else:
                m["content"] = g.text(R.randint(10, 300))
            self.steps.append(g.step(self))
            op = "edit_middle"
        elif r < 0.76:
            p = R.randint(0, len(self.system))
            self.system = self.system[:p] + "Z" + self.system[p:]
            self.steps.append(g.step(self))
            op = "edit_system"
        elif r < 0.81:
            del self.steps[len(self.steps) - R.randint(1, min(3, len(self.steps))):]
            if R.random() < 0.5:
                self.steps.append(g.step(self))
            op = "rollback"
        elif r < 0.86:
            op = "resend"
        elif r < 0.91 and len(self.steps) > 3:
            keep = R.randint(1, 3)
            self.steps = [[{"role": "user", "content": "Summary of earlier work: " + g.text(R.randint(100, 800))}]] + self.steps[-keep:]
            op = "compact"
        elif r < 0.95:
            self.ctk = dict(self.ctk)
            if "thinking" in self.ctk:
                self.ctk["thinking"] = not self.ctk["thinking"]
            else:
                self.ctk["reasoning_effort"] = R.choice(["none", "high", "max"])
            self.steps.append(g.step(self))
            op = "mode_switch"
        else:
            self.steps.append(g.step(self))
            op = "grow"
        self.requests_left -= 1
        self.last_op = op
        return op

    def done(self) -> bool:
        return self.requests_left <= 0 or self.chars() > self.target_chars
