#!/usr/bin/env python3
"""sassir.py: a small SASS (sm_12x, cuobjdump -sass text) intermediate representation for the r6-comm dataflow gate.

What it does, per kernel:
  * parses the instruction text (predicate guard, opcode with modifiers, operands);
  * builds the CFG (BRA targets, EXIT / RET / BPT.TRAP terminators; a CALL clobbers every register);
  * runs a symbolic forward analysis to a fixpoint: every register value is a hash-consed node
    ('c', v) constant | ('cb', bank, off) constant-bank word | ('sr', name) special register | ('init', reg)
    | ('phi', block, reg) loop / join value | ('ld', idx, k, addr) word k of the load at instruction idx
    | ('op', idx, k, inputs) output k of instruction idx applied to the input nodes | ('ite', p, a, b) predicated write
    | ('clob', idx, reg) value after a CALL.
    A predicated write `@P0 X Rd` gives Rd = ite(P0, X, old Rd), so a `@P LDG Rd` / `@!P LDG Rd` pair is ONE node that
    names both loads and the predicate that selects between them;
  * evaluates any integer node concretely under an environment (constant bank image, special registers, values of
    loads), with loop phis taken at iteration 1 (forward incoming) or 2 (back edge applied once).
Anything it does not model becomes an opaque node: it can never make a check pass, only fail it (fail closed).
CPU only, no CUDA needed (the .sass text comes from cuobjdump on spark-eight)."""
import hashlib
import re

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1

INS_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
GUARD_RE = re.compile(r"^@(!?)(U?P[T0-6])\s+(.*)$")
FP_CLASS = ("FADD", "FMUL", "FFMA", "F2FP", "F2F", "I2F", "F2I", "FMNMX", "FSETP", "FSEL", "FSET", "FCHK", "FRND",
            "MUFU", "HADD2", "HMUL2", "HFMA2", "HSETP2", "HMNMX2", "DADD", "DMUL", "DFMA", "DSETP", "I2FP", "F2IP")


def split_functions(text):
    """{mangled name: [(addr, raw instruction text)]} from cuobjdump -sass output."""
    out, cur = {}, None
    for line in text.splitlines():
        if "Function :" in line:
            cur = out.setdefault(line.split("Function :", 1)[1].strip(), [])
            continue
        if cur is None:
            continue
        m = INS_RE.search(line)
        if m:
            cur.append((int(m.group(1), 16), re.sub(r"\s+", " ", m.group(2)).strip()))
    return out


def split_operands(s):
    ops, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            ops.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        ops.append(cur.strip())
    return ops


def _half_bits(txt):
    """fp16 bit pattern of a cuobjdump HFMA2 immediate (decimal text, +/-INF, QNAN), or None."""
    import struct
    t = txt.strip()
    special = {"+INF": 0x7C00, "INF": 0x7C00, "-INF": 0xFC00, "QNAN": 0x7FFF, "+QNAN": 0x7FFF, "-QNAN": 0xFFFF}
    if t.upper() in special:
        return special[t.upper()]
    try:
        f = float(t)
    except ValueError:
        return None
    b = struct.unpack("<H", struct.pack("<e", f))[0]
    if struct.unpack("<e", struct.pack("<H", b))[0] != f:
        return None  # not exactly representable: refuse (fail closed)
    return b


class Opnd:
    """One parsed operand. kind: reg | ureg | pred | upred | cb | imm | fimm | mem | sr | bar | label | other."""
    __slots__ = ("kind", "name", "num", "neg", "inv", "abs", "notp", "val", "bank", "off", "base", "ubase", "wide",
                 "raw", "suffix", "idx")

    def __init__(self, raw):
        self.raw = raw
        self.kind = "other"
        self.name = None
        self.num = None
        self.neg = self.inv = self.abs = self.notp = self.wide = False
        self.val = None
        self.bank = self.off = None
        self.base = self.ubase = None
        self.suffix = ""
        self.idx = None


def _reg(tok):
    m = re.match(r"^(R\d+|RZ|UR\d+|URZ)((?:\.[A-Za-z0-9]+)*)$", tok)
    if not m:
        return None
    return m.group(1), m.group(2)


def parse_operand(raw, fp_imm=False):
    o = Opnd(raw)
    s = raw
    if s.startswith("!") and re.match(r"^!U?P[T0-6]$", s):
        o.kind = "upred" if s[1] == "U" else "pred"
        o.notp = True
        o.name = s[1:]
        return o
    if re.match(r"^U?P[T0-6]$", s):
        o.kind = "upred" if s[0] == "U" else "pred"
        o.name = s
        return o
    if s.startswith("desc[") or s.startswith("["):
        o.kind = "mem"
        inner = s[s.rfind("[") + 1:s.rfind("]")]
        if s.startswith("desc["):
            o.idx = s[5:s.index("]")]
        parts = inner.replace("+-", "+-").split("+")
        o.off = 0
        for p in parts:
            p = p.strip()
            if not p:
                continue
            r = _reg(p)
            if r:
                if r[0].startswith("U"):
                    o.ubase = r[0]
                else:
                    o.base = r[0]
                    o.wide = ".64" in r[1]
            else:
                try:
                    o.off += int(p, 16) if "0x" in p else int(p)
                except ValueError:
                    o.kind = "other"
                    return o
        return o
    neg = inv = ab = False
    if s.startswith("-") and not re.match(r"^-0x|^-\d|^-QNAN|^-INF", s):
        neg, s = True, s[1:]
    if s.startswith("~"):
        inv, s = True, s[1:]
    if s.startswith("|") and s.endswith("|"):
        ab, s = True, s[1:-1]
    o.neg, o.inv, o.abs = neg, inv, ab
    r = _reg(s)
    if r:
        o.name = r[0]
        o.suffix = r[1].replace(".reuse", "")
        o.kind = "ureg" if r[0].startswith("U") else "reg"
        if r[0] in ("RZ", "URZ"):
            o.num = None
        else:
            o.num = int(r[0][2:] if r[0].startswith("UR") else r[0][1:])
        return o
    m = re.match(r"^c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]$", s)
    if m:
        o.kind, o.bank, o.off = "cb", int(m.group(1), 16), int(m.group(2), 16)
        return o
    if s.startswith("c["):
        o.kind = "other"  # indexed constant bank: opaque
        return o
    if s.startswith("SR"):
        o.kind, o.name = "sr", s
        return o
    if re.match(r"^B\d+$", s):
        o.kind, o.name = "bar", s
        return o
    if fp_imm:
        o.kind, o.val = "fimm", raw  # keep the literal text: +0.0 vs -0.0 vs 1 etc.
        return o
    m = re.match(r"^(-?)0x([0-9a-f]+)$", raw)
    if m:
        v = int(m.group(2), 16)
        o.kind, o.val = "imm", (-v if m.group(1) else v)
        return o
    m = re.match(r"^-?\d+$", raw)
    if m:
        o.kind, o.val = "imm", int(raw)
        return o
    o.kind = "other"
    return o


class Ins:
    __slots__ = ("idx", "addr", "text", "gneg", "gpred", "op", "base", "mods", "ops", "vals", "block", "bpred", "bdiv")

    def __init__(self, idx, addr, text):
        self.idx, self.addr, self.text = idx, addr, text
        m = GUARD_RE.match(text)
        if m:
            self.gneg, self.gpred, rest = m.group(1) == "!", m.group(2), m.group(3)
        else:
            self.gneg, self.gpred, rest = False, "PT", text
        parts = rest.split(" ", 1)
        self.op = parts[0]
        self.base = self.op.split(".")[0]
        self.mods = self.op.split(".")[1:]
        rawops = split_operands(parts[1]) if len(parts) > 1 else []
        # r7: `BRA.U UP0, target` / `BRA.U !UP0, target` / `BRA P0, target` carry their condition as the first operand,
        # not as an @-guard. The r6 CFG read them as unconditional (losing the fall-through edge, so the code after them
        # was statically unreachable and never checked). The operand becomes the guard; a second guard is refused.
        self.bpred = None
        # r8-comm2fix: `BRA.DIV URx, target` (emitted before a __shfl_*_sync in code the compiler cannot prove
        # warp-converged) is taken iff the warp is NOT converged over the mask in URx, into a WARPSYNC.COLLECTIVE
        # fallback. r7's sassir read it as an unconditional BRA (no @-guard): the converged fast path, the one that
        # runs, was statically unreachable and no gate checked it. It is now a two-way branch on the environment's
        # special register SR_WARPDIV (unset: opaque, both paths possible); its mask operand is dropped.
        self.bdiv = self.base == "BRA" and "DIV" in self.op.split(".")[1:]
        if self.bdiv:
            if len(rawops) != 2 or not re.fullmatch(r"U?R\w+", rawops[0].strip()):
                raise ValueError(f"unexpected BRA.DIV form: {text}")
            rawops = rawops[1:]
        if self.base == "BRA" and len(rawops) >= 2 and re.fullmatch(r"!?U?P[T0-6]", rawops[0].strip()):
            c = rawops[0].strip()
            if self.gpred == "PT" and not self.gneg:
                if c.lstrip("!") not in ("PT", "UPT"):
                    self.gneg, self.gpred = c.startswith("!"), c.lstrip("!")
                elif c.startswith("!"):
                    self.gneg, self.gpred = True, "PT"  # never taken
            else:
                self.bpred = c  # @guard BRA Pc, target: taken iff guard AND Pc (combined in _guard)
            rawops = rawops[1:]
        fp = self.base in ("FADD", "FMUL", "FFMA", "FMNMX", "FSETP", "FSEL", "HFMA2", "HADD2", "HMUL2", "MUFU")
        self.ops = [parse_operand(r, fp_imm=fp) for r in rawops]
        self.vals = None
        self.block = None

    def is_fp(self):
        return self.base in FP_CLASS


# ------------------------------------------------------------------------------------------------------------------
class Nodes:
    def __init__(self):
        self.tab = {}
        self.lst = []

    def mk(self, t):
        i = self.tab.get(t)
        if i is None:
            i = len(self.lst)
            self.tab[t] = i
            self.lst.append(t)
        return i

    def __getitem__(self, i):
        return self.lst[i]


class Kernel:
    """Parsed kernel + CFG + fixpoint symbolic state."""

    def __init__(self, name, insns_text):
        self.name = name
        self.ins = [Ins(i, a, t) for i, (a, t) in enumerate(insns_text)]
        self.by_addr = {x.addr: x.idx for x in self.ins}
        self.N = Nodes()
        self.ZERO = self.N.mk(("c", 0))
        self.TRUE = self.N.mk(("pc", 1))
        self.FALSE = self.N.mk(("pc", 0))
        self.phi_in = {}  # phi node id -> list of (pred block, node id, is_back)
        self._build_cfg()
        self._fixpoint()

    # ---------------- CFG
    def _build_cfg(self):
        n = len(self.ins)
        leaders = {0}
        for x in self.ins:
            if x.base == "BRA":
                tgt = [o for o in x.ops if o.kind == "imm"]
                if tgt:
                    t = self.by_addr.get(tgt[-1].val)
                    if t is not None:
                        leaders.add(t)
                if x.idx + 1 < n:
                    leaders.add(x.idx + 1)
            elif x.base in ("EXIT", "RET", "BPT", "CALL", "BRX", "JMP", "JMX") and x.idx + 1 < n:
                leaders.add(x.idx + 1)
        starts = sorted(leaders)
        self.blocks = []
        for bi, s in enumerate(starts):
            e = starts[bi + 1] if bi + 1 < len(starts) else n
            self.blocks.append(list(range(s, e)))
            for i in range(s, e):
                self.ins[i].block = bi
        start_of = {s: bi for bi, s in enumerate(starts)}
        self.succ = [[] for _ in self.blocks]
        for bi, b in enumerate(self.blocks):
            last = self.ins[b[-1]]
            nxt = bi + 1 if bi + 1 < len(self.blocks) else None
            uncond = last.gpred == "PT" and not last.gneg and not last.bdiv
            never = last.gpred == "PT" and last.gneg
            if last.base == "BRA" and not never:
                tgt = [o for o in last.ops if o.kind == "imm"]
                t = self.by_addr.get(tgt[-1].val) if tgt else None
                if t is not None:
                    self.succ[bi].append(start_of[t])
                if not uncond and nxt is not None:
                    self.succ[bi].append(nxt)
            elif last.base in ("EXIT", "RET", "BPT", "BRX", "JMP", "JMX") and not never:
                if not uncond and nxt is not None:
                    self.succ[bi].append(nxt)
            elif nxt is not None:
                self.succ[bi].append(nxt)
        self.pred = [[] for _ in self.blocks]
        for bi, ss in enumerate(self.succ):
            for s in ss:
                self.pred[s].append(bi)
        # reachability, DFS order, back edges
        self.reach, self.back, order = set(), set(), []
        state = {}
        stack = [(0, iter(self.succ[0]))]
        state[0] = 1
        self.reach.add(0)
        while stack:
            b, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[b] = 2
                order.append(b)
                stack.pop()
                continue
            if state.get(nxt) == 1:
                self.back.add((b, nxt))
            elif nxt not in state:
                state[nxt] = 1
                self.reach.add(nxt)
                stack.append((nxt, iter(self.succ[nxt])))
        self.rpo = list(reversed(order))

    def _callee_writes(self, addr):
        start = self.by_addr.get(addr)
        if start is None:
            return None
        out = set()
        for x in self.ins[start:]:
            if x.base in ("RET",):
                return out
            if x.base in ("CALL", "BRX", "JMX"):
                return None
            mods = set(x.mods)
            width = 4 if "128" in mods else (2 if ({"64", "WIDE", "S64", "U64"} & mods) else 1)
            if x.base in ("CS2R", "LDS", "LDG", "LDL") and width == 1 and x.base == "CS2R":
                width = 2
            for o in x.ops[:3]:
                if o.kind in ("reg", "ureg") and o.num is not None:
                    pre = "UR" if o.kind == "ureg" else "R"
                    out |= {f"{pre}{o.num + j}" for j in range(width)}
                elif o.kind in ("pred", "upred") and o.name not in ("PT", "UPT"):
                    out.add(o.name)
                if x.base not in ("IADD3", "LEA", "ISETP", "UISETP", "LOP3", "ULOP3", "IMNMX", "PLOP3", "UIADD3", "ULEA"):
                    break  # only these write predicates / registers beyond the first operand
        return None

    # ---------------- symbolic transfer
    def _read(self, st, o, wide=False, n=1):
        """value node(s) of an operand: a list of `n` (or 2 if wide) 32-bit words, low first."""
        N = self.N
        cnt = 2 if wide else n
        if o.kind in ("reg", "ureg"):
            if o.num is None:
                w = [self.ZERO] * cnt
            else:
                pre = "UR" if o.kind == "ureg" else "R"
                w = [st.get(f"{pre}{o.num + j}", N.mk(("init", f"{pre}{o.num + j}"))) for j in range(cnt)]
            if o.neg or o.inv or o.abs:
                tag = ("neg" if o.neg else "") + ("inv" if o.inv else "") + ("abs" if o.abs else "")
                return [N.mk(("mod", tag, tuple(w)))] + ([] if cnt == 1 else [N.mk(("modhi", tag, tuple(w)))])
            return w
        if o.kind == "cb":
            w = [N.mk(("cb", o.bank, o.off + 4 * j)) for j in range(cnt)]
            if o.neg or o.inv:
                tag = "neg" if o.neg else "inv"
                return [N.mk(("mod", tag, tuple(w)))] + ([] if cnt == 1 else [N.mk(("modhi", tag, tuple(w)))])
            return w
        if o.kind == "imm":
            v = o.val & (M64 if cnt == 2 else M32)
            if cnt == 2:
                return [N.mk(("c", v & M32)), N.mk(("c", (v >> 32) & M32))]
            return [N.mk(("c", v))]
        if o.kind == "fimm":
            return [N.mk(("fimm", o.raw))]
        if o.kind == "sr":
            return [N.mk(("sr", o.name + (f"#{j}" if j else ""))) for j in range(cnt)]
        if o.kind in ("pred", "upred"):
            p = self._pred(st, o)
            return [p]
        return [N.mk(("opaque_operand", o.raw))] * cnt

    def _pred(self, st, o):
        N = self.N
        name = o.name
        if name in ("PT", "UPT"):
            v = self.TRUE
        else:
            v = st.get(name, N.mk(("init", name)))
        if o.notp:
            v = self._not(v)
        return v

    def _not(self, v):
        N = self.N
        t = N[v]
        if t == ("pc", 1):
            return self.FALSE
        if t == ("pc", 0):
            return self.TRUE
        if t[0] == "pnot":
            return t[1]
        return N.mk(("pnot", v))

    def _guard(self, st, x):
        g = self._guard1(st, x.gpred, x.gneg)
        if getattr(x, "bdiv", False):  # r8: taken iff the warp is diverged (environment SR_WARPDIV)
            g = self.N.mk(("pand", g, self.N.mk(("sr", "SR_WARPDIV"))))
        if getattr(x, "bpred", None):
            c = x.bpred
            g2 = self._guard1(st, c.lstrip("!"), c.startswith("!"))
            g = self.N.mk(("pand", g, g2))
        return g

    def _guard1(self, st, name, neg):
        if name == "PT":
            return self.FALSE if neg else self.TRUE
        o = Opnd(name)
        o.kind = "upred" if name.startswith("U") else "pred"
        o.name = name
        o.notp = neg
        return self._pred(st, o)

    def under(self, g, n):
        """n simplified under the knowledge that predicate g holds (top-level ite on g or on !g only)."""
        N = self.N
        for _ in range(64):
            t = N[n]
            if t[0] != "ite":
                return n
            if t[1] == g:
                n = t[2]
            elif N[t[1]] == ("pnot", g) or N[g] == ("pnot", t[1]):
                n = t[3]
            else:
                return n
        return n

    def _write(self, st, reg, val, g):
        if reg in ("RZ", "URZ", "PT", "UPT") or reg is None:
            return
        if g == self.TRUE:
            st[reg] = val
        elif g == self.FALSE:
            return
        else:
            old = st.get(reg, self.N.mk(("init", reg)))
            st[reg] = self.N.mk(("ite", g, val, old))

    @staticmethod
    def _dreg(o, j=0):
        if o.kind not in ("reg", "ureg") or o.num is None:
            return None
        return ("UR" if o.kind == "ureg" else "R") + str(o.num + j)

    def _exec(self, st, x, record):
        """symbolic execution of one instruction on state st (dict)."""
        N = self.N
        g = self._guard(st, x)
        b, op, ops = x.base, x.op, x.ops
        mods = set(x.mods)
        if record:
            x.vals = {"guard": g}

        def opnode(k, ins):
            return N.mk(("op", x.idx, k, tuple(ins)))

        # ---- control / no-register-effect
        if b in ("BRA", "EXIT", "RET", "BPT", "NOP", "BAR", "MEMBAR", "ERRBAR", "CGAERRBAR", "CCTL", "BSSY", "BSYNC",
                 "WARPSYNC", "YIELD", "DEPBAR", "FENCE", "BMOV", "ACQBULK", "ELECT", "PREEXIT", "NANOSLEEP", "UCGABAR_ARV",
                 "UCGABAR_WAIT", "ACQSHMINIT"):
            if record and b == "BRA":
                x.vals["cond"] = g
            if b == "BMOV" and ops and ops[0].kind in ("reg", "ureg"):
                self._write(st, self._dreg(ops[0]), opnode(0, []), g)
            return
        if b == "CALL":
            # CALL.REL to a local subroutine (e.g. the 64-bit division helper) clobbers exactly the registers that
            # subroutine writes (every first-operand register / predicate between its entry and its RET, widened to
            # 4 registers to cover pairs and quads); any other call (vprintf) clobbers every register. The clobbered
            # values become opaque nodes: they can never make a check pass.
            tgt = [o for o in ops if o.kind == "imm"]
            regs = self._callee_writes(tgt[-1].val) if ("REL" in mods and tgt) else None
            if regs is None:
                regs = set(k for k in st.keys() if k.startswith("R") or k.startswith("UR") or k.startswith("P")) | \
                    {f"R{r}" for r in range(256)}
            for k in regs:
                st[k] = N.mk(("clob", x.idx, k))
            return
        # ---- memory
        if b in ("LDG", "LD", "LDS", "LDL", "LDSM", "ATOMG", "ATOM", "ATOMS", "ATOMG", "RED", "REDG", "LDGSTS", "LDC",
                 "LDCU", "ULDC"):
            if b in ("LDC", "LDCU", "ULDC"):
                dst = ops[0]
                src = ops[1]
                n = 4 if "128" in mods else (2 if "64" in mods else 1)
                if src.kind == "cb":
                    for j in range(n):
                        self._write(st, self._dreg(dst, j), N.mk(("cb", src.bank, src.off + 4 * j)), g)
                else:
                    for j in range(n):
                        self._write(st, self._dreg(dst, j), opnode(j, []), g)
                return
            n = 4 if "128" in mods else (2 if "64" in mods else 1)
            if b in ("ATOMG", "ATOM", "ATOMS"):
                # ATOMG.E.ADD... Pd, Rd, [mem], Rsrc
                regs = [o for o in ops if o.kind in ("reg", "ureg")]
                mem = [o for o in ops if o.kind == "mem"][0]
                dst = regs[0]
                data = [w for o in regs[1:] for w in self._read(st, o, n=n)]
            elif b in ("RED", "REDG"):
                mem = [o for o in ops if o.kind == "mem"][0]
                dst, data = None, []
            else:
                dst, mem = ops[0], ops[1]
                data = []
            addr = self._addr(st, mem)
            if record:
                x.vals["addr"] = addr
                x.vals["data"] = data
            if dst is not None and dst.kind in ("reg", "ureg"):
                for j in range(n):
                    self._write(st, self._dreg(dst, j), N.mk(("ld", x.idx, j, addr)), g)
            return
        if b in ("STG", "ST", "STS", "STL", "STSM"):
            mem, src = ops[0], ops[1]
            n = 4 if "128" in mods else (2 if "64" in mods else 1)
            if record:
                x.vals["addr"] = self._addr(st, mem)
                # a guarded store writes only when its guard holds: ite(guard, v, old) is v there
                x.vals["data"] = [self.under(g, d) for d in self._read(st, src, n=n)]
            return
        # ---- HFMA2 move idiom: HFMA2 Rd, -RZ, RZ, h0, h1 == the constant (h0 << 16) | h1 (exact: -0*0 + h = h).
        #      r8-comm2fix: cuobjdump prints the HIGH half first. r7 had (h1 << 16) | h0: `HFMA2 R3, -RZ, RZ, 0,
        #      5.96e-08` (the +1 of every arrival atomic) evaluated to 0x10000 and the BRA.DIV fallback's lane masks
        #      4 / 2 / 1 to 0x40000 / 0x20000 / 0x10000. (The familiar `HFMA2 R, -RZ, RZ, 1.875, 0` = 1.0f = 0x3f800000.)
        if b == "HFMA2" and len(ops) == 5 and ops[1].raw in ("-RZ", "RZ") and ops[2].raw == "RZ" and \
                all(o.kind == "fimm" for o in ops[3:5]):
            hv = [_half_bits(o.raw) for o in ops[3:5]]
            if None not in hv:
                self._write(st, self._dreg(ops[0]), N.mk(("c", (hv[0] << 16) | hv[1])), g)
                return
        # ---- predicates
        if b in ("ISETP", "UISETP"):
            pds = [o for o in ops[:2]]
            wide = any(m in mods for m in ("S64", "U64"))
            a = self._read(st, ops[2], wide=wide)
            c = self._read(st, ops[3], wide=wide)
            p = self._pred(st, ops[4])
            extra = [self._pred(st, ops[5])] if len(ops) > 5 else []
            ins = a + c + [p] + extra
            if record:
                x.vals["ins"] = ins
            self._write(st, pds[0].name if not pds[0].notp else None, opnode(0, ins), g)
            if pds[1].name not in ("PT", "UPT"):
                self._write(st, pds[1].name, opnode(1, ins), g)
            return
        if b in ("PLOP3", "UPLOP3", "P2R", "R2P", "VOTE", "VOTEU", "PSETP"):
            if ops and ops[0].kind in ("pred", "upred"):
                self._write(st, ops[0].name, opnode(0, []), g)
                if len(ops) > 1 and ops[1].kind in ("pred", "upred"):
                    self._write(st, ops[1].name, opnode(1, []), g)
            elif ops and ops[0].kind in ("reg", "ureg"):
                self._write(st, self._dreg(ops[0]), opnode(0, []), g)
            return
        if b in ("LOP3", "ULOP3"):
            # LOP3.LUT [Pd,] Rd, a, b, c, lut, Pc
            i0 = 1 if ops[0].kind in ("pred", "upred") else 0
            ins = [self._read(st, o)[0] for o in ops[i0 + 1:i0 + 4]] + [self._read(st, ops[i0 + 4])[0]]
            ins += [self._pred(st, ops[i0 + 5])] if len(ops) > i0 + 5 else []
            if record:
                x.vals["ins"] = ins
            self._write(st, self._dreg(ops[i0]), opnode(0, ins), g)
            if i0:
                self._write(st, ops[0].name, opnode(1, ins), g)
            return
        # ---- r7: SHFL.{BFLY,IDX,UP,DOWN} Pd, Rd, Ra, b, c: the first operand is a predicate destination, Rd the value
        #      (the generic rule below wrote nothing to Rd, so a shuffled value silently kept Rd's older value)
        if b == "SHFL" and len(ops) >= 5 and ops[0].kind in ("pred", "upred") and ops[1].kind == "reg":
            ins = []
            for o in ops[2:]:
                ins.extend(self._read(st, o) if o.kind not in ("pred", "upred") else [self._pred(st, o)])
            if record:
                x.vals["ins"] = ins
            self._write(st, self._dreg(ops[1]), opnode(0, ins), g)
            if ops[0].name not in ("PT", "UPT"):
                self._write(st, ops[0].name, opnode(1, ins), g)
            return
        # ---- generic ALU: destination first; inputs = the remaining operands
        dests, srcs = self._split_dest(x)
        ins = []
        for o, wide in srcs:
            if o.kind in ("pred", "upred"):
                ins.append(self._pred(st, o))
            else:
                ins.extend(self._read(st, o, wide=wide))
        if record:
            x.vals["ins"] = ins
        for k, (reg, isp) in enumerate(dests):
            self._write(st, reg, opnode(k, ins), g)

    def _split_dest(self, x):
        """[(dest register name, is_pred)], [(source operand, wide)] for the generic ALU instructions."""
        b, mods, ops = x.base, set(x.mods), x.ops
        w64 = "64" in mods or "S64" in mods or "U64" in mods
        dests, srcs = [], []
        if b in ("IADD3", "UIADD3"):
            d = ops[0]
            dests.append((self._dreg(d), False))
            k = 1
            while k < len(ops) and ops[k].kind in ("pred", "upred") and k < 3:
                if ops[k].name not in ("PT", "UPT"):
                    dests.append((ops[k].name, True))
                else:
                    dests.append((None, True))
                k += 1
            srcs = [(o, False) for o in ops[k:]]
            return dests, srcs
        if b in ("LEA", "ULEA"):
            d = ops[0]
            dests.append((self._dreg(d), False))
            k = 1
            if k < len(ops) and ops[k].kind in ("pred", "upred"):
                dests.append((ops[k].name if ops[k].name not in ("PT", "UPT") else None, True))
                k += 1
            srcs = [(o, False) for o in ops[k:]]
            return dests, srcs
        if b in ("IMNMX", "UIMNMX", "VIMNMX", "UVIMNMX") and ops and ops[0].kind in ("pred", "upred"):
            # new form: IMNMX.S64 Pa, Pb, Rd, Ra, Rb, Psel, Pc
            d = ops[2]
            n = 2 if w64 else 1
            for j in range(n):
                dests.append((self._dreg(d, j), False))
            srcs = [(ops[3], w64), (ops[4], w64)] + [(o, False) for o in ops[5:]]
            return dests, srcs
        d = ops[0] if ops else None
        nd = 1
        if b in ("IMAD", "UIMAD") and "WIDE" in mods:
            nd = 2
        elif b in ("IADD", "MOV", "UMOV", "SEL", "USEL", "IMNMX", "UIMNMX", "CS2R") and w64:
            nd = 2
        elif b in ("CS2R",) and d is not None and d.kind == "reg":
            nd = 1 if "32" in mods else 2  # r7: CS2R.32 writes one register (was 2: clobbered Rd+1, fail-closed)
        elif b == "F2I" and "U64" in mods or b == "F2I" and "S64" in mods:
            nd = 2
        if d is not None and d.kind in ("reg", "ureg"):
            for j in range(nd):
                dests.append((self._dreg(d, j), False))
        elif d is not None and d.kind in ("pred", "upred"):
            dests.append((d.name, True))
        for i, o in enumerate(ops[1:]):
            wide = False
            if b in ("IADD", "MOV", "UMOV", "SEL", "USEL", "IMNMX", "UIMNMX") and w64 and o.kind in ("reg", "ureg", "cb", "imm"):
                wide = True
            if b in ("IMAD", "UIMAD") and "WIDE" in mods and i == 2:
                wide = True
            if b == "I2F" and ("U64" in mods or "S64" in mods) and i == 0:
                wide = True
            srcs.append((o, wide))
        return dests, srcs

    def _addr(self, st, mem):
        N = self.N
        if mem.kind != "mem":
            return (N.mk(("opaque_operand", mem.raw)),)
        parts = []
        if mem.base is not None and mem.base != "RZ":
            o = Opnd(mem.base)
            o.kind, o.name, o.num = "reg", mem.base, int(mem.base[1:])
            parts += self._read(st, o, wide=mem.wide)
            if not mem.wide:
                parts.append(self.ZERO)
        else:
            parts += [self.ZERO, self.ZERO]
        if mem.ubase is not None and mem.ubase != "URZ":
            o = Opnd(mem.ubase)
            o.kind, o.name, o.num = "ureg", mem.ubase, int(mem.ubase[2:])
            u = self._read(st, o, wide=mem.idx is not None and mem.base in (None, "RZ"))
            parts += u if len(u) == 2 else u + [self.ZERO]
        else:
            parts += [self.ZERO, self.ZERO]
        parts.append(N.mk(("c", mem.off & M64)))
        return tuple(parts)

    # ---------------- fixpoint
    def _merge(self, bi, outs):
        N = self.N
        preds = [p for p in self.pred[bi] if p in self.reach and p in outs]
        if bi == 0 and not self.pred[0]:
            return {}
        if not preds:
            return {}
        keys = set()
        for p in preds:
            keys |= set(outs[p].keys())
        st = {}
        for k in keys:
            vals = [(p, outs[p].get(k, N.mk(("init", k)))) for p in preds]
            if bi == 0:
                vals.append((-1, N.mk(("init", k))))
            vs = {v for _, v in vals}
            if len(vs) == 1:
                st[k] = vals[0][1]
            else:
                ph = N.mk(("phi", bi, k))
                st[k] = ph
                self.phi_in[ph] = [(p, v, (p, bi) in self.back) for p, v in vals]
        return st

    def _fixpoint(self):
        outs = {}
        for _ in range(60):
            changed = False
            for bi in self.rpo:
                st = self._merge(bi, outs)
                for i in self.blocks[bi]:
                    self._exec(st, self.ins[i], False)
                if outs.get(bi) != st:
                    outs[bi] = st
                    changed = True
            if not changed:
                break
        else:
            raise RuntimeError(f"{self.name}: symbolic fixpoint did not converge")
        self.outs = outs
        for bi in self.rpo:
            st = self._merge(bi, outs)
            for i in self.blocks[bi]:
                self._exec(st, self.ins[i], True)
        # branch condition of every CFG edge (taken edge: the guard; fall-through: its negation)
        self.edge_cond = {}
        for bi in self.reach:
            last = self.ins[self.blocks[bi][-1]]
            if last.vals is None or (last.gpred == "PT" and not last.bdiv):
                continue
            # r8-comm2fix: only a control instruction decides an edge. A block can end in a PREDICATED non-control
            # instruction when the next one is a branch target (r7: `@!P4 STS [R22+0x8]` before the BRA.DIV fallback's
            # re-entry at 0x4a10); its fall-through is unconditional. r7's sassir gave it the condition !guard, so the
            # threads whose guard held (lane 0, the s writer) never reached the rest of the kernel.
            if last.base not in ("BRA", "EXIT", "RET", "BPT", "BRX", "JMP", "JMX"):
                continue
            g = last.vals["guard"]
            nxt = bi + 1
            for s_ in self.succ[bi]:
                if last.base == "BRA" and s_ != nxt:
                    self.edge_cond[(bi, s_)] = g
                elif s_ == nxt:
                    self.edge_cond[(bi, s_)] = self._not(g)

    # ---------------- walking
    def children(self, n, value_only=False):
        """input node ids of node n. value_only: the data inputs only (no load addresses, no ite predicates)."""
        t = self.N[n]
        k = t[0]
        if k == "op":
            if value_only and self.ins[t[1]].base in ("SEL", "USEL", "FSEL") and t[3]:
                return list(t[3][:-1])  # the select predicate is control, not value
            return list(t[3])
        if k == "ite":
            return [t[2], t[3]] if value_only else [t[1], t[2], t[3]]
        if k == "ld":
            return [] if value_only else list(t[3])
        if k in ("mod", "modhi"):
            return list(t[2])
        if k == "pnot":
            return [t[1]]
        if k == "pand":
            return [t[1], t[2]]
        if k == "phi":
            return [v for _, v, _ in self.phi_in.get(n, [])]
        return []

    def slice_ins(self, roots, value_only=False):
        """instruction indices of every op / ld node reachable from roots."""
        seen, out, stack = set(), set(), list(roots)
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            t = self.N[n]
            if t[0] in ("op", "ld"):
                out.add(t[1])
            stack.extend(self.children(n, value_only))
        return out


# ------------------------------------------------------------------------------------------------------------------
class Env:
    """Concrete environment: constant-bank words, special registers, loads (by instruction idx) and opaque values."""

    def __init__(self, cb, sr, loads=None, seed=0):
        self.cb, self.sr, self.loads, self.seed = cb, sr, loads or {}, seed

    def opaque(self, key):
        h = hashlib.sha256(repr((key, self.seed)).encode()).digest()
        return int.from_bytes(h[:4], "little")


class Ambiguous(Exception):
    pass


class Unknown(Exception):
    pass


def _s32(v):
    v &= M32
    return v - (1 << 32) if v >> 31 else v


def _s64(v):
    v &= M64
    return v - (1 << 64) if v >> 63 else v


def _lut3(a, b, c, lut):
    r = 0
    for i in range(32):
        idx = (((a >> i) & 1) << 2) | (((b >> i) & 1) << 1) | ((c >> i) & 1)
        r |= ((lut >> idx) & 1) << i
    return r


def _cmp(op, a, b):
    return {"EQ": a == b, "NE": a != b, "LT": a < b, "LE": a <= b, "GT": a > b, "GE": a >= b}[op]


class Evaluator:
    def __init__(self, k, env, it=1):
        self.k, self.env, self.it = k, env, it
        self.memo = {}
        self.is_strict = False
        self._strict = None
        self._reach = {}

    def _opq(self, key):
        if self.is_strict:
            raise Unknown(str(key)[:80])
        return self.env.opaque(key)

    def strict(self, n):
        """concrete value, or None when it depends on anything the environment does not fix (loads, clocks, ...)."""
        sub = self._strict
        if sub is None:
            sub = self._strict = Evaluator(self.k, self.env, self.it)
            sub.is_strict = True
        try:
            return sub(n)
        except (Unknown, Ambiguous):
            return None

    def edge(self, p, s):
        c = self.k.edge_cond.get((p, s), self.k.TRUE)
        v = self.strict(c)
        return None if v is None else (1 if v else 0)

    def reach(self, b):
        """1 / 0 / None: does block b execute on the forward (iteration-1) path under this environment?"""
        if b in self._reach:
            return self._reach[b]
        self._reach[b] = None  # cycle guard
        K = self.k
        if b == 0:
            self._reach[b] = 1
            return 1
        res = 0
        for p in K.pred[b]:
            if (p, b) in K.back or p not in K.reach:
                continue
            r = self.reach(p)
            e = self.edge(p, b)
            if r == 0 or e == 0:
                continue
            if r == 1 and e == 1:
                res = 1
                break
            res = None
        self._reach[b] = res
        return res

    def __call__(self, n):
        key = (n, self.it)
        if key in self.memo:
            return self.memo[key]
        v = self._eval(n)
        self.memo[key] = v
        return v

    def _eval(self, n):
        K, E = self.k, self.env
        t = K.N[n]
        kind = t[0]
        if kind == "c":
            return t[1]
        if kind == "pc":
            return t[1]
        if kind == "cb":
            key = (t[1], t[2])
            if key in E.cb:
                return E.cb[key] & M32
            return self._opq(("cb",) + key)
        if kind == "sr":
            if t[1] in E.sr:
                return E.sr[t[1]] & M32
            return self._opq(("sr", t[1]))
        if kind == "init":
            return self._opq(("init", t[1]))
        if kind == "clob":
            return self._opq(("clob", t[1], t[2]))
        if kind == "fimm" or kind == "opaque_operand":
            return self._opq((kind, t[1]))
        if kind == "pnot":
            return 1 - self(t[1])
        if kind == "pand":
            return 1 if (self(t[1]) and self(t[2])) else 0
        if kind == "ite":
            return self(t[2]) if self(t[1]) else self(t[3])
        if kind in ("mod", "modhi"):
            w = [self(x) for x in t[2]]
            tag = t[1]
            if len(w) == 1:
                v = w[0]
                if "inv" in tag:
                    v = ~v & M32
                if "neg" in tag:
                    v = (-v) & M32
                return v
            v = w[0] | (w[1] << 32)
            if "inv" in tag:
                v = ~v & M64
            if "neg" in tag:
                v = (-v) & M64
            return (v & M32) if kind == "mod" else (v >> 32)
        if kind == "ld":
            key = t[1]
            if key in E.loads:
                vals = E.loads[key]
                return vals[t[2]] if isinstance(vals, (list, tuple)) else vals
            return self._opq(("ld", t[1], t[2], tuple(self(a) for a in t[3])))
        if kind == "phi":
            sel, sev = self.phi_pick(n)
            return sev(sel)
        if kind == "op":
            return self._op(t[1], t[2], t[3])
        raise Ambiguous(f"unknown node kind {kind}")

    def phi_pick(self, n):
        """(incoming node, evaluator to evaluate it with) that phi n takes under this environment."""
        K, E = self.k, self.env
        t = K.N[n]
        if True:
            inc = K.phi_in.get(n, [])
            fwd = [(p, v) for p, v, back in inc if not back]
            bck = [(p, v) for p, v, back in inc if back]
            if self.it > 1 and bck:
                sub = self._sub()
                vals = {sub(v) for _, v in bck}
                if len(vals) != 1:
                    raise Ambiguous(f"phi {t[1]}/{t[2]}: {len(vals)} distinct back-edge values")
                return bck[0][1], sub
            # join of forward paths: keep the incoming edges this environment can take (path-sensitive, iteration 1)
            definite, maybe = [], []
            for p, v in fwd:
                r = self.reach(p)
                e = self.edge(p, t[1])
                if r == 0 or e == 0:
                    continue
                (definite if (r == 1 and e == 1) else maybe).append(v)
            pick = definite if definite else maybe
            vals = {self(v) for v in pick}
            if len(vals) != 1:
                raise Ambiguous(f"phi {t[1]}/{t[2]} has {len(vals)} distinct forward values")
            return pick[0], self

    def _sub(self):
        if getattr(self, "_subev", None) is None:
            self._subev = Evaluator(self.k, self.env, self.it - 1)
            self._subev.is_strict = self.is_strict
        return self._subev

    def _op(self, idx, k, ins):
        x = self.k.ins[idx]
        b, mods = x.base, set(x.mods)
        v = [self(i) for i in ins]
        U = b.startswith("U") and b not in ("UMOV",) and len(b) > 1
        bb = b[1:] if b.startswith("U") and b[1:] in ("IADD3", "IMAD", "LEA", "SHF", "LOP3", "SEL", "ISETP", "IMNMX",
                                                         "VIMNMX", "MOV", "PRMT", "FLO", "POPC") else b
        if b == "UMOV":
            bb = "MOV"
        if bb in ("MOV", "R2UR", "S2R", "S2UR", "CS2R", "CS2UR"):
            if bb == "MOV":
                return v[k] if k < len(v) else v[0]
            if k == 0:
                return v[0]  # R2UR copy; S2R / CS2R: the special register's value from the environment
            return self._opq(("sr-hi", idx, k))
        if bb == "IMAD":
            if "WIDE" in mods:
                a, c = v[0], v[1]
                add = (v[2] | (v[3] << 32)) if len(v) >= 4 else 0
                if "U32" in mods:
                    r = (a * c + add) & M64
                else:
                    r = (_s32(a) * _s32(c) + add) & M64
                return (r & M32) if k == 0 else (r >> 32)
            if "HI" in mods:
                a, c, d = v[0], v[1], v[2]
                if "U32" in mods:
                    return (((a * c) >> 32) + d) & M32
                return (((_s32(a) * _s32(c)) >> 32) + d) & M32
            if "X" in mods:
                cin = v[3] if len(v) > 3 else 0
                return (v[0] * v[1] + v[2] + cin) & M32
            return (v[0] * v[1] + v[2]) & M32
        if bb == "IADD3":
            s = v[0] + v[1] + v[2]
            if "X" in mods:
                s += sum(1 for c in v[3:5] if c)
                if k == 0:
                    return s & M32
                return 1 if (s >> 32) else 0
            if k == 0:
                return s & M32
            return 1 if s >> 32 else 0
        if b == "IADD" and not mods and len(v) == 2:  # r7: the 2-operand 32-bit IADD form newer ptxas emits
            return (v[0] + v[1]) & M32 if k == 0 else (1 if (v[0] + v[1]) >> 32 else 0)
        if b == "IADD" and "64" in mods:
            a = v[0] | (v[1] << 32)
            c = v[2] | (v[3] << 32)
            r = (a + c) & M64
            return (r & M32) if k == 0 else (r >> 32)
        if bb == "LEA":
            if "HI" in mods:
                a, c_, hi, s = v[0], v[1], v[2], v[3]
                cin = v[4] if ("X" in mods and len(v) > 4) else 0
                full = ((hi << 32) | a) << s
                r = ((full >> 32) & M32) + c_ + (1 if cin else 0)
                return r & M32
            a, c_, s = v[0], v[1], v[2]
            r = (a << s) + c_
            if k == 0:
                return r & M32
            return 1 if (r >> 32) else 0
        if bb == "SHF":
            a, s, hi = v[0], v[1] & 0xFF, v[2]
            val = (hi << 32) | a
            if "L" in mods:
                r = (val << min(s, 64)) & ((1 << 128) - 1)
                return ((r >> 32) & M32) if "HI" in mods else (r & M32)
            if "S32" in mods or "S64" in mods:
                val = _s64(val)
            r = val >> min(s, 64)
            r &= M64
            return ((r >> 32) & M32) if "HI" in mods else (r & M32)
        if bb == "LOP3":
            r = _lut3(v[0], v[1], v[2], v[3])
            if k == 0:
                return r
            return 1 if r else 0
        if bb == "SEL":
            if "64" in mods:
                p = v[4]
                return (v[k] if p else v[2 + k])
            return v[0] if v[2] else v[1]
        if bb == "PRMT":
            a, sel, c = v[0], v[1], v[2]
            by = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(c >> (8 * i)) & 0xFF for i in range(4)]
            r = 0
            for i in range(4):
                nib = (sel >> (4 * i)) & 0xF
                bv = by[nib & 7]
                if nib & 8:
                    bv = 0xFF if (bv & 0x80) else 0
                r |= bv << (8 * i)
            return r
        if bb in ("IMNMX", "VIMNMX"):
            if "S64" in mods or "U64" in mods:
                a = v[0] | (v[1] << 32)
                c = v[2] | (v[3] << 32)
                p = v[4]
                if "S64" in mods:
                    a2, c2 = _s64(a), _s64(c)
                else:
                    a2, c2 = a, c
                r = (a if a2 <= c2 else c) if p else (a if a2 >= c2 else c)
                return (r & M32) if k == 0 else (r >> 32)
            a, c, p = v[0], v[1], v[2]
            if "U32" in mods:
                a2, c2 = a, c
            else:
                a2, c2 = _s32(a), _s32(c)
            return (a if a2 <= c2 else c) if p else (a if a2 >= c2 else c)
        if bb == "ISETP":
            cmpop = [m for m in x.mods if m in ("EQ", "NE", "LT", "LE", "GT", "GE")][0]
            bop = [m for m in x.mods if m in ("AND", "OR", "XOR")][0]
            if "S64" in mods or "U64" in mods:
                a = v[0] | (v[1] << 32)
                c = v[2] | (v[3] << 32)
                if "S64" in mods:
                    a, c = _s64(a), _s64(c)
                r = _cmp(cmpop, a, c)
                p = v[4]
            elif "EX" in mods:
                a, c, p, plo = v[0], v[1], v[2], v[3] if len(v) > 3 else 1
                if "U32" in mods:
                    a2, c2 = a, c
                else:
                    a2, c2 = _s32(a), _s32(c)
                if cmpop in ("EQ",):
                    r = (a2 == c2) and plo
                elif cmpop == "NE":
                    r = (a2 != c2) or plo
                else:
                    r = _cmp(cmpop[0] + "T", a2, c2) or (a2 == c2 and plo)
                    if cmpop in ("LE", "GE"):
                        r = _cmp(cmpop[0] + "T", a2, c2) or (a2 == c2 and plo)
                p = v[2]
            else:
                a, c, p = v[0], v[1], v[2]
                if "U32" in mods:
                    r = _cmp(cmpop, a, c)
                else:
                    r = _cmp(cmpop, _s32(a), _s32(c))
            r = bool(r)
            if k == 1:
                r = not r
            if bop == "AND":
                return 1 if (r and p) else 0
            if bop == "OR":
                return 1 if (r or p) else 0
            return 1 if (r != bool(p)) else 0
        if bb in ("FLO", "POPC", "VOTEU", "VOTE", "UFLO", "UPOPC", "LEPC", "P2R", "R2P", "SHFL"):
            return self._opq(("op", idx, k, tuple(v)))
        if bb in ("I2F", "F2I", "MUFU", "FADD", "FMUL", "FFMA", "F2FP", "HFMA2", "F2F"):
            return self._opq(("fp", idx, k, tuple(v)))
        return self._opq(("op", idx, k, tuple(v)))


def addr_value(ev, addr):
    """concrete byte address of a memory operand node tuple (lo, hi, ulo, uhi, off)."""
    lo, hi, ulo, uhi, off = addr
    a = ev(lo) | (ev(hi) << 32)
    u = ev(ulo) | (ev(uhi) << 32)
    return (a + u + ev(off)) & M64
