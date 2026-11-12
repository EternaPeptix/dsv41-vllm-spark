#!/usr/bin/env python3
"""fpflow.py: the r6-comm C1 reduction-dataflow gate (replaces r3's sass_r3.py ARITH regex, which failed on non-FP idioms).

    python3 -I fpflow.py --ref live.sass --cand cand.sass [--json out.json] [--mutant] [--show KERNEL]

Inputs are `cuobjdump -sass` texts (or .so/.o/.cubin files when cuobjdump is on PATH). CPU only.

The spec (DESIGN §3 exactness frame, rdma_ar.cu acc8/accv/pack8): every element of an all-reduce output is
    bf16_rn(((((((( +0.0 + x_0) + x_1) + x_2) + x_3) + x_4) + x_5) + x_6) + x_7)       (fp32-out: the fp32 sum, no rounding)
in fp32 FADD (RN, no FTZ, no operand modifier), where x_r is rank r's term: the exact bf16->fp32 widening of the same
half of the same 32-bit word of the same 16-byte vector, read from this rank's p.in when r == rank and from receive slot
[slot][r] otherwise. A gather / column-gather element is a copy (no FP op on the path).

For each kernel the gate PROVES this from the SASS dataflow, not from mnemonic counts:
  1. every STG whose stored value depends on an FP instruction is a reduction store; each of its 32-bit words is
     F2FP.BF16.F32.PACK_AB(chain_hi, chain_lo) (bf16) or a chain (fp32);
  2. each chain is a serial FADD list: the first FADD adds the +0.0 seed (RZ) to leaf 0, every later FADD adds exactly one
     leaf to the previous FADD (two FADD operands = a tree -> FAIL; FFMA / FMUL / modifiers / another seed -> FAIL);
  3. each leaf is a pure bit move of one loaded word (IMAD.U32 x 0x10000 / PRMT / SHF / LOP3 masks, or the word itself
     for fp32) selected by a predicate: evaluating the predicate for rank = 0..7 shows leaf r is the local p.in load when
     rank == r and the receive-slot load of rank r otherwise; addresses are evaluated concretely (random parameter
     sets, loop iterations 1 and 2) and decoded into region / slot / rank / element / word, and must match the stored
     element, word and half (two-shot: local element b + i, receive element i, out[b + i] and send2[slot][i]);
  4. every FADD / F2FP / FFMA / FMUL in the kernel belongs to a proven chain, and every other FP-class instruction is
     either an HFMA2 move idiom (inputs only -RZ / RZ / immediates, i.e. a constant) or I2F / F2I / MUFU whose forward
     dataflow reaches no stored value (only branch conditions / addresses), else FAIL;
  5. gathers: no FP-class instruction in the value slice of any store.
Then every candidate reduction kernel's canonical record must equal its production twin's (ar_kernel_r3<m,f,8,mask> vs
ar_kernel_pl<m,f,8,7> two-shot / <m,f,8,5> otherwise), and every ref kernel must be instruction-identical in the
candidate (the g130 flag-off identity gate). Exit 0 only if all hold. --mutant inverts the exit code for mutant builds
(exit 0 only if the gate FAILS on at least one candidate r3 kernel of every reduction variant)."""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sassir as S  # noqa: E402

# Params layout (rdma_ar.cu struct Params) at the kernel-parameter base of constant bank 0
PBASE = 0x380
FIELDS = {"in": 0x00, "out": 0x08, "nvec": 0x10, "chunk": 0x18, "send": 0x20, "send2": 0x28, "recv": 0x30,
          "recv2": 0x38, "flags": 0x40, "flags2": 0x48, "ctrl": 0x50, "st": 0x58, "max_bytes": 0x60,
          "max_chunk": 0x68, "rank": 0x70, "world": 0x74, "timeout_ns": 0x78, "timing": 0x80, "seg": 0x88}
REGIONS = ["in", "out", "send", "send2", "recv", "recv2", "flags", "flags2", "ctrl", "st", "qd", "qy"]
Q_OFF = 0x90  # r7: R7Proj q {d, y, rows} follows Params (sizeof 0x90); only kR7Proj (mask & 512) kernels read it
W = 8
# r8-comm2fix: {kernel name: projgate.prove_r7 result} for the kR7Proj kernels of the candidate (set by main(); None =
# not computed: then a kR7Proj kernel's projection is not exempt and fails here, as in r7)
PROJ_PROOFS = None


def make_env(rank, seed, tid=37, cta=0, nvec=4000, max_bytes=0x200000, max_chunk=0x40000, nctaid=3):
    cb = {}

    def put64(off, v):
        cb[(0, PBASE + off)] = v & S.M32
        cb[(0, PBASE + off + 4)] = (v >> 32) & S.M32

    for i, r in enumerate(REGIONS):
        if r in FIELDS:
            put64(FIELDS[r], (i + 1) << 40)
    put64(Q_OFF, (REGIONS.index("qd") + 1) << 40)
    put64(Q_OFF + 8, (REGIONS.index("qy") + 1) << 40)
    put64(Q_OFF + 16, 6)
    chunk = (nvec + W - 1) // W
    put64(FIELDS["nvec"], nvec)
    put64(FIELDS["chunk"], chunk)
    put64(FIELDS["max_bytes"], max_bytes)
    put64(FIELDS["max_chunk"], max_chunk)
    cb[(0, PBASE + FIELDS["rank"])] = rank
    cb[(0, PBASE + FIELDS["world"])] = W
    put64(FIELDS["timeout_ns"], 600 * 10 ** 9)
    cb[(0, PBASE + FIELDS["timing"])] = 0
    put64(FIELDS["seg"], 28)
    # launch geometry: ntid at c[0x0][0x360..], nctaid at c[0x0][0x370..] on sm_12x (checked in the live SASS)
    cb[(0, 0x360)], cb[(0, 0x364)], cb[(0, 0x368)] = 512, 1, 1
    cb[(0, 0x370)], cb[(0, 0x374)], cb[(0, 0x378)] = nctaid, 1, 1
    sr = {"SR_TID.X": tid, "SR_TID.Y": 0, "SR_TID.Z": 0, "SR_CTAID.X": cta, "SR_CTAID.Y": 0, "SR_CTAID.Z": 0,
          "SR_LANEID": tid & 31}
    return S.Env(cb, sr, seed=seed), dict(nvec=nvec, chunk=chunk, max_bytes=max_bytes, max_chunk=max_chunk,
                                          i=cta * 512 + tid, stride=nctaid * 512)


def decode(addr, info):
    """byte address -> (region, slot, rank, element, word) or None."""
    reg = addr >> 40
    if not 1 <= reg <= len(REGIONS):
        return None
    name = REGIONS[reg - 1]
    off = addr & ((1 << 40) - 1)
    word = (off % 16) // 4
    if off % 4:
        return None
    if name in ("in", "out", "qy"):
        return (name, None, None, off // 16, word)
    if name == "qd":
        return (name, None, None, off // 4, None)
    if name == "send":
        s, rem = divmod(off, info["max_bytes"])
        return (name, s, None, rem // 16, word)
    if name == "send2":
        s, rem = divmod(off, info["max_chunk"])
        return (name, s, None, rem // 16, word)
    if name in ("recv", "recv2"):
        row = info["max_bytes"] if name == "recv" else info["max_chunk"]
        q, rem = divmod(off, row)
        s, r = divmod(q, W)
        return (name, s, r, rem // 16, word)
    return (name, None, None, off, None)


class Fail(Exception):
    pass


# ------------------------------------------------------------------------------------------------------------------
class KernelGate:
    def __init__(self, name, text):
        self.name = name
        self.k = S.Kernel(name, text)
        m = re.search(r"ar_kernel_(pl|r3)ILi(\d)ELb(\d)ELi8ELi(\d+)EE", name)
        self.fam, self.mode, self.f32, self.mask = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))) \
            if m else (None, None, None, None)
        self.errors = []
        self.notes = []
        self.records = []
        self.stream = []
        self.ignored = []
        self.copies = []

    def err(self, msg):
        self.errors.append(msg)

    # -- bit-level view of a leaf ---------------------------------------------------------------------------------
    # A bit is 0 / 1 / (word source node, bit) / "X" (unknown). A word source is a load, or a predicated select whose
    # arms are loads (or a register's older value); which load a rank reads is decided later by evaluation (resolve).
    def is_wordsrc(self, n, depth=0):
        t = self.k.N[n]
        if t[0] == "ld":
            return True
        term = lambda q: self.k.N[q][0] in ("phi", "init", "clob", "ld", "ite")
        if t[0] == "ite" and depth < 32:
            a, b = self.is_wordsrc(t[2], depth + 1), self.is_wordsrc(t[3], depth + 1)
            return (a and (b or term(t[3]))) or (b and term(t[2]))
        if t[0] == "phi" and depth < 8:
            arms = [v for _, v, _ in self.k.phi_in.get(n, [])]
            return any(self.is_wordsrc(v, depth + 1) for v in arms) and all(term(v) for v in arms)
        return False

    def bits(self, n, depth=0):
        K = self.k
        X = ["X"] * 32
        t = K.N[n]
        if depth > 48:
            return X
        if t[0] == "c":
            return [(t[1] >> j) & 1 for j in range(32)]
        if self.is_wordsrc(n):
            return [(n, j) for j in range(32)]
        if t[0] == "ite":
            a, b = self.bits(t[2], depth + 1), self.bits(t[3], depth + 1)
            return a if a == b else X
        if t[0] != "op":
            return X
        x = K.ins[t[1]]
        ins = t[3]
        mods = set(x.mods)
        if x.base == "IMAD" and mods <= {"U32", "SHL", "MOV"}:
            if any(o.neg or o.inv or o.abs for o in x.ops[1:4]):
                return X
            a, m, c = (self.bits(i, depth + 1) for i in ins[:3])
            mc, ac, cc = self._const(m), self._const(a), self._const(c)
            if ac == 0 or mc == 0:
                return c  # IMAD.MOV.U32 Rd, RZ, RZ, Rc
            if mc is None or cc != 0 or mc & (mc - 1):
                return X
            s = mc.bit_length() - 1
            return [0] * s + a[:32 - s]
        if x.base == "MOV" and "64" not in mods and not x.ops[1].neg:
            return self.bits(ins[0], depth + 1)
        if x.base == "SHF" and len(ins) >= 3 and not ({"S32", "S64"} & mods):
            a, hi = self.bits(ins[0], depth + 1), self.bits(ins[2], depth + 1)
            s_ = self._const(self.bits(ins[1], depth + 1))
            if s_ is None or s_ > 63:
                return X
            v = a + hi
            full = ([0] * s_ + v)[:64] if "L" in mods else v[s_:] + [0] * s_
            return full[32:64] if "HI" in mods else full[:32]
        if x.base == "PRMT":
            a, c = self.bits(ins[0], depth + 1), self.bits(ins[2], depth + 1)
            sel = self._const(self.bits(ins[1], depth + 1))
            if sel is None or x.mods:
                return X
            out = []
            for i in range(4):
                nib = (sel >> (4 * i)) & 0xF
                src = a if (nib & 7) < 4 else c
                j = nib & 3
                out += (["X"] * 8) if nib & 8 else src[8 * j:8 * j + 8]
            return out
        if x.base == "LOP3":
            a, b, c = (self.bits(i, depth + 1) for i in ins[:3])
            lut = self._const(self.bits(ins[3], depth + 1))
            if lut is None:
                return X
            out = []
            for j in range(32):
                bs = (a[j], b[j], c[j])
                unk = [q for q in bs if not isinstance(q, int)]
                syms = {q for q in unk if q != "X"}
                if len(syms) > 1:
                    out.append("X")
                    continue
                sym = syms.pop() if syms else None
                res = {}
                import itertools
                for combo in itertools.product((0, 1), repeat=len(unk)):
                    env = dict(zip(range(len(unk)), combo))
                    it = iter(combo)
                    vals = []
                    symv = None
                    for q in bs:
                        if isinstance(q, int):
                            vals.append(q)
                        else:
                            v_ = next(it)
                            if q == sym:
                                if symv is None:
                                    symv = v_
                                v_ = symv  # the same symbolic bit in two inputs takes one value
                            vals.append(v_)
                    r = (lut >> ((vals[0] << 2) | (vals[1] << 1) | vals[2])) & 1
                    res.setdefault(symv, set()).add(r)
                if all(len(v) == 1 for v in res.values()):
                    outs = {k: next(iter(v)) for k, v in res.items()}
                    if len(set(outs.values())) == 1:
                        out.append(next(iter(outs.values())))
                    elif sym is not None and outs.get(0) == 0 and outs.get(1) == 1:
                        out.append(sym)
                    else:
                        out.append("X")
                else:
                    out.append("X")
            return out
        return X

    @staticmethod
    def _const(b):
        if b is None or any(not isinstance(q, int) for q in b):
            return None
        return sum(q << j for j, q in enumerate(b))

    @staticmethod
    def _is_word(b):
        if b is None or not b or not isinstance(b[0], tuple):
            return False
        src = b[0][0]
        return all(q == (src, j) for j, q in enumerate(b))

    def leaf(self, n):
        """('lo'|'hi'|'raw', word source node) or None: the operand is the exact bf16->fp32 widening of one half of
        one loaded word (low 16 bits zero, high 16 bits = that half), or (fp32) the loaded word itself."""
        b = self.bits(n)
        if self._is_word(b):
            return ("raw", b[0][0])
        lo, hi = b[:16], b[16:]
        if lo != [0] * 16 or not all(isinstance(q, tuple) for q in hi):
            return None
        src = hi[0][0]
        if hi == [(src, j) for j in range(16)]:
            return ("lo", src)
        if hi == [(src, 16 + j) for j in range(16)]:
            return ("hi", src)
        return None


    # -- chains -----------------------------------------------------------------------------------------------------
    def chain(self, n):
        """[seed, leaf_0, ..., leaf_7] (leaves as node ids) of a serial FADD list, or raises Fail."""
        K = self.k
        seq, cur = [], n
        while True:
            t = K.N[cur]
            if t[0] != "op":
                raise Fail(f"chain node is {t[0]}, not an FADD")
            x = K.ins[t[1]]
            if x.base != "FADD":
                raise Fail(f"{x.op} at /*{x.addr:04x}*/ in the reduction chain (only FADD allowed)")
            if x.op != "FADD":
                raise Fail(f"{x.op} at /*{x.addr:04x}*/: rounding/FTZ modifier (production uses plain FADD = RN, no FTZ)")
            if any(o.neg or o.abs or o.inv for o in x.ops[1:3]):
                raise Fail(f"operand modifier in {x.text} at /*{x.addr:04x}*/")
            if x.gpred != "PT" or x.gneg:
                raise Fail(f"predicated FADD at /*{x.addr:04x}*/")
            a, b = t[3][0], t[3][1]
            fa = self._is_fadd(a)
            fb = self._is_fadd(b)
            seq.append(x.idx)
            if fa and fb:
                raise Fail(f"FADD at /*{x.addr:04x}*/ adds two partial sums (tree, not the serial rank order)")
            if fa or fb:
                prev, lf = (a, b) if fa else (b, a)
                seq[-1] = (x.idx, lf)
                cur = prev
                continue
            # first FADD: one operand must be the +0.0 seed, the other leaf 0
            za, zb = a == K.ZERO, b == K.ZERO
            if za == zb:
                raise Fail(f"first FADD at /*{x.addr:04x}*/ has no +0.0 (RZ) seed operand ({x.text}): rank-0 seed or "
                           f"constant seed")
            seq[-1] = (x.idx, b if za else a)
            break
        seq.reverse()
        return seq  # [(fadd idx, leaf node)] from the first FADD (seed) to the last

    def _is_fadd(self, n):
        t = self.k.N[n]
        return t[0] == "op" and self.k.ins[t[1]].base in ("FADD", "FFMA", "FMUL")

    # -- leaf resolution under a concrete rank ------------------------------------------------------------------------
    def resolve(self, src, ev):
        """(load instruction, word, byte address) that word source `src` holds under evaluator ev: selects follow the
        evaluated predicate, joins follow the incoming edge this environment takes (iteration-aware)."""
        K = self.k
        for _ in range(256):
            t = K.N[src]
            if t[0] == "ld":
                x = K.ins[t[1]]
                return x, t[2], S.addr_value(ev, t[3])
            if t[0] == "ite":
                src = t[2] if ev(t[1]) else t[3]
                continue
            if t[0] == "phi":
                src, ev = ev.phi_pick(src)
                continue
            raise Fail(f"word source resolves to {t[0]}, not a load")
        raise Fail("word-source select chain too deep")

    def envs(self, block):
        """(rank, info, evaluator) for 3 parameter sets x loop iterations 1, 2 x rank 0..7 where `block` executes."""
        for tid, cta in ((37, 0), (5, 0), (300, 1)):
            for it in (1, 2):
                for rank in range(W):
                    env, info = make_env(rank, tid * 7 + cta, tid=tid, cta=cta)
                    ev = S.Evaluator(self.k, env, it)
                    if ev.reach(block) == 0:
                        continue
                    info["elem"] = cta * 512 + tid + (it - 1) * 3 * 512  # the element this thread owns (grid-stride)
                    yield rank, info, ev

    # -- r7 (review item: the concrete environment must enter the unrolled staging loop) ---------------------------
    # (tid, cta, gridDim.x, nvec): the r6 sets (3 CTAs, nvec 4000: grid staging never enters the 4x-unrolled loop),
    # 16-CTA sets that enter it once (nvec 26112 > tid + 3 * 8192) and twice (60000 > tid + 4 * 8192 + 3 * 8192), and
    # single-CTA S1 sets (CTA 0 stages alone with stride 512: 1536 = the 24 KiB S1 bound, 4000 = all four b1..b3 arms).
    STAGE_ENVS = ((37, 0, 3, 4000), (300, 1, 3, 4000), (37, 0, 16, 26112), (511, 15, 16, 26112), (37, 0, 16, 60000),
                  (200, 3, 16, 60000), (37, 0, 1, 1536), (5, 0, 1, 4000))

    def check_staging(self):
        """Every phase-1 staging store (STG to the send region) copies p.in[e] word w to send[slot][e] word w, for every
        environment it executes in; every such store instruction must execute in at least one environment, and the
        r3 grid-staging kernels must execute a block holding four send stores (the unrolled body). Returns the number
        of (store, environment) pairs proven."""
        K = self.k
        stg = [x for bi in K.rpo for i in K.blocks[bi] for x in [K.ins[i]]
               if x.base == "STG" and x.vals and "data" in x.vals]
        n_ok, hit = 0, {}
        for tid, cta, nct, nvec in self.STAGE_ENVS:
            for it in (1, 2):
                for rank in (0, 5):
                    env, info = make_env(rank, tid * 31 + cta + nct, tid=tid, cta=cta, nvec=nvec, nctaid=nct)
                    ev = S.Evaluator(K, env, it)
                    for x in stg:
                        if ev.reach(x.block) == 0:  # as envs(): None (load-dependent edges before it) still runs
                            continue
                        g = ev.strict(x.vals["guard"])
                        if g == 0:
                            continue
                        try:
                            sa = decode(S.addr_value(ev, x.vals["addr"]), info)
                        except (S.Unknown, S.Ambiguous, KeyError):
                            continue
                        if sa is None or sa[0] != "send":
                            continue
                        for w, d in enumerate(x.vals["data"]):
                            if not self.is_wordsrc(d):
                                self.err(f"staging store /*{x.addr:04x}*/ word {w}: not a copy of a loaded word")
                                return n_ok
                            try:
                                lx, kk, la = self.resolve(d, ev)
                            except Fail as e:
                                self.err(f"staging store /*{x.addr:04x}*/ word {w}: {e}")
                                return n_ok
                            ld = decode(la + 4 * kk, info)
                            if ld is None or ld[0] != "in" or ld[3] != sa[3] or ld[4] != sa[4] + w:
                                self.err(f"staging store /*{x.addr:04x}*/ (tid {tid} cta {cta} grid {nct} nvec {nvec} "
                                         f"it {it}): send element {sa[3]} word {sa[4] + w} holds {ld}")
                                return n_ok
                        n_ok += 1
                        hit.setdefault(x.idx, set()).add((nct, nvec, it))
        send_stores = set(hit)
        blocks = {}
        for i in send_stores:
            blocks.setdefault(K.ins[i].block, []).append(i)
        self.staging = {"proven": n_ok, "stores": len(send_stores),
                        "unrolled_blocks": sorted(b for b, v in blocks.items() if len(v) >= 4)}
        grid_staging = self.fam == "r3" and self.mask is not None and not (self.mask & (16 | 64)) or self.fam == "pl"
        if self.mode in (0, 2, 1, 3) and self.fam in ("r3", "pl"):
            if not send_stores and not (self.fam == "r3" and self.mask & 64):
                self.err("no staging store proven in any environment")
            if grid_staging and not self.staging["unrolled_blocks"]:
                self.err("no environment executes the 4x-unrolled staging body (gate cannot see it)")
        return n_ok

    # -- main ---------------------------------------------------------------------------------------------------------
    def run(self):
        K = self.k
        reach_ins = [x for bi in K.rpo for i in K.blocks[bi] for x in [K.ins[i]]]
        fp_ins = [x for x in reach_ins if x.is_fp()]
        stores = [x for x in reach_ins if x.base in ("STG", "ST", "STS", "STL") and x.vals and "data" in x.vals]
        atoms = [x for x in reach_ins if x.base in ("ATOMG", "ATOM", "ATOMS", "RED", "REDG") and x.vals]
        fp_idx = {x.idx for x in fp_ins}
        red_stores, copy_stores, proven = [], [], set()
        # r7 HT13: projection-tail stores (to q.y) are proven by gate/projgate.py, not here; their FP slice is exempt
        # from the stray-FP rule below only where no reduction chain uses it (projgate requires every FP op of that
        # slice to be in its proven chain / tree / y form)
        self.proj_stores, proj_fp = [], set()
        proj_sts = set()
        if self.mask is not None and self.mask & 512 and PROJ_PROOFS is not None:
            # r8-comm2fix: the projection tail is exempt exactly where gate/projgate.py PROVED it, in every one of its
            # environments: the y stores it proved, the shared partial-sum STS it proved (level-1 / level-2 slots;
            # r7 sent them to check_store, "bf16 word 0 is not an F2FP pack"), and the FP ops of its proven chain /
            # tree / y form (r7 took only the y stores' register slice, which stops at the LDS of s: the dot chain
            # and the level-1 tree were "stray"). No proof (or a failed one) -> the kernel FAILS here. Every q.y store
            # the address decode finds must be one the proof executed.
            pr = PROJ_PROOFS.get(self.name)
            if pr is None or not pr.get("ok"):
                self.err("projection tail not proven by projgate: " + (pr or {}).get("error", "no proof"))
            else:
                env0, info0 = make_env(0, 1)
                ev0 = S.Evaluator(K, env0, 1)
                proven_y = set(pr["ystores"])
                for x in stores:
                    try:
                        sa = decode(S.addr_value(ev0, x.vals["addr"]), info0)
                    except (S.Unknown, S.Ambiguous, KeyError):
                        sa = None
                    if sa is not None and sa[0] == "qy" and x.idx not in proven_y:
                        self.err(f"q.y store /*{x.addr:04x}*/ {x.text} is not one projgate proved")
                self.proj_stores = [K.ins[i] for i in sorted(proven_y)]
                proj_sts = set(pr["sts"])
                proj_fp = set(pr["covered"]) & fp_idx
                if not self.proj_stores:
                    self.err("kR7Proj kernel without any proven store to q.y")
        elif self.mask is not None and self.mask & 512:
            env0, info0 = make_env(0, 1)
            ev0 = S.Evaluator(K, env0, 1)
            for x in stores:
                try:
                    sa = decode(S.addr_value(ev0, x.vals["addr"]), info0)
                except (S.Unknown, S.Ambiguous, KeyError):
                    sa = None
                if sa is not None and sa[0] == "qy":
                    self.proj_stores.append(x)
                    proj_fp |= K.slice_ins(x.vals["data"], value_only=True) & fp_idx
            if not self.proj_stores:
                self.err("kR7Proj kernel without any store to q.y")
        elif self.mask is not None:
            for x in stores:  # a non-projection kernel must never store to q.y (it has no such operand)
                pass
        pstore_idx = {x.idx for x in self.proj_stores} | proj_sts
        for x in stores:
            if x.idx in pstore_idx:
                proven.add(x.idx)
                continue
            vs = K.slice_ins(x.vals["data"], value_only=True)
            if not vs & fp_idx:
                continue
            tops = {K.ins[K.N[d][1]].base for d in x.vals["data"] if K.N[d][0] == "op"}
            (red_stores if tops & {"F2FP", "FADD", "FFMA", "FMUL"} else copy_stores).append(x)
        chain_ins = set()
        if self.mode in (0, 2):
            if not red_stores:
                self.err("reduction kernel without any store of an FP value")
            for x in red_stores:
                try:
                    chain_ins |= self.check_store(x)
                    proven.add(x.idx)
                except (Fail, S.Ambiguous) as e:
                    self.err(f"store /*{x.addr:04x}*/ {x.text}: {e}")
        elif self.mode in (1, 3):
            for x in red_stores:
                self.err(f"gather kernel stores an FP-derived value at /*{x.addr:04x}*/ {x.text}")
        else:
            self.notes.append("not an rdma_ar reduction/gather kernel: only the FP-op census is reported")
        # stores whose naive slice reaches an FP op only through the never-selected arm of a select: prove they copy
        for x in copy_stores:
            try:
                self.copies.append(self.check_copy(x))
                proven.add(x.idx)
            except (Fail, S.Ambiguous) as e:
                self.err(f"store /*{x.addr:04x}*/ {x.text}: FP-derived value that is not a proven copy: {e}")
        # every FADD/F2FP/FFMA/FMUL must be in a proven chain
        proj_only = proj_fp - chain_ins
        self.proj_exempt = len(proj_only)
        for x in fp_ins:
            if x.base in ("FADD", "F2FP", "FFMA", "FMUL") and x.idx not in chain_ins and x.idx not in proj_only:
                self.err(f"stray {x.op} at /*{x.addr:04x}*/ outside every proven reduction chain")
        # every other FP-class instruction: HFMA2 move idiom or I2F/F2I/MUFU with no stored value in its forward slice
        sinks = {}
        for x in stores + atoms:
            if x.idx not in proven:  # proven stores: their real value is the proven chain / the selected load
                for i in K.slice_ins(x.vals.get("data", []), value_only=True):
                    sinks.setdefault(i, set()).add(f"stored value /*{x.addr:04x}*/")
            for i in K.slice_ins(x.vals.get("addr", ()), value_only=False) - K.slice_ins(x.vals.get("data", []), True):
                sinks.setdefault(i, set()).add("address")
        for x in reach_ins:
            if x.base == "BRA" and x.vals and "cond" in x.vals:
                for i in K.slice_ins([x.vals["cond"]]):
                    sinks.setdefault(i, set()).add("branch")
            if x.base in ("LDG", "LDS", "LD") and x.vals and "addr" in x.vals:
                for i in K.slice_ins(x.vals["addr"]):
                    sinks.setdefault(i, set()).add("address")
        for x in fp_ins:
            if x.base in ("FADD", "F2FP", "FFMA", "FMUL"):
                continue
            uses = sorted(sinks.get(x.idx, set()))
            stored = [u for u in uses if u.startswith("stored value")]
            if x.base == "HFMA2" and len(x.ops) >= 5 and x.ops[1].raw in ("-RZ", "RZ") and x.ops[2].raw == "RZ" and \
                    all(o.kind in ("fimm", "imm") for o in x.ops[3:5]):
                kind = "HFMA2 move idiom (constant: inputs -RZ, RZ, immediates)"
                ok = True
            elif x.base in ("I2F", "F2I", "MUFU"):
                kind = "integer-division / timeout conversion"
                ok = not stored
            else:
                kind = "other FP-class instruction"
                ok = False
            self.ignored.append({"addr": f"{x.addr:04x}", "ins": x.text, "kind": kind,
                                 "reaches": uses or ["nothing stored or branched on"], "ok": ok})
            if not ok:
                self.err(f"{x.text} at /*{x.addr:04x}*/ ({kind}) reaches {uses}")
        # r7: phase-1 staging copies, in environments that enter the unrolled loop
        self.staging = {}
        if self.fam in ("r3", "pl"):
            self.check_staging()
        # ordered FP stream with dataflow (program order)
        lane_of = {}
        for rec in self.records:
            for pos, fi in enumerate(rec["fadds"]):
                lane_of[fi] = (rec["lane"], pos)
            if rec.get("pack") is not None:
                lane_of[rec["pack"]] = (rec["lane"].split("/")[0] + "/pack", -1)
        for x in fp_ins:
            if x.idx in lane_of:
                ln, pos = lane_of[x.idx]
                self.stream.append(f"{x.op} {ln} " + ("pack" if pos < 0 else ("seed+r0" if pos == 0 else f"+r{pos}")))
        return not self.errors

    def check_store(self, x):
        """prove one reduction store; returns the FP instruction indices it covers."""
        K = self.k
        data = x.vals["data"]
        covered = set()
        lanes = []  # (word, half, chain list)
        for w, d in enumerate(data):
            t = K.N[d]
            if self.f32:
                lanes.append((w, "raw", d))
                continue
            if t[0] != "op" or K.ins[t[1]].base != "F2FP":
                raise Fail(f"bf16 word {w} is not an F2FP pack ({t[0]})")
            px = K.ins[t[1]]
            if px.op != "F2FP.BF16.F32.PACK_AB" or any(o.neg or o.abs for o in px.ops[1:3]):
                raise Fail(f"word {w}: {px.text} (production: F2FP.BF16.F32.PACK_AB, RN, no modifier)")
            covered.add(px.idx)
            lanes.append((w, "hi", t[3][0], px.idx))
            lanes.append((w, "lo", t[3][1], px.idx))
        # decode the store address under each rank and two parameter sets
        recs = []
        for ln in lanes:
            w, half, root = ln[0], ln[1], ln[2]
            ch = self.chain(root)
            if len(ch) != W:
                raise Fail(f"word {w} {half}: chain of {len(ch)} terms (expected {W})")
            leaves = []
            for pos, (fi, lf) in enumerate(ch):
                covered.add(fi)
                L = self.leaf(lf)
                if L is None:
                    raise Fail(f"word {w} {half} term {pos}: not a pure bit move of one loaded word")
                want = "raw" if self.f32 else half
                if L[0] != want:
                    raise Fail(f"word {w} {half} term {pos}: leaf is the {L[0]} half (expected {want})")
                leaves.append(L[1])
            loadkinds = set()
            n_exec = 0
            for rank, info, ev in self.envs(x.block):
                n_exec += 1
                sa = decode(S.addr_value(ev, x.vals["addr"]) + 4 * w, info)
                if sa is None:
                    raise Fail(f"store address does not decode (rank {rank})")
                b = rank * info["chunk"] if self.mode == 2 else 0
                if sa[0] == "out":
                    e = sa[3] - b
                elif sa[0] == "send2" and self.mode == 2:
                    e = sa[3]
                else:
                    raise Fail(f"reduction value stored to region {sa[0]}")
                if sa[4] != w:
                    raise Fail(f"word {w} stored at word position {sa[4]}")
                if e != info["elem"]:
                    raise Fail(f"thread of element {info['elem']} stores element {e}")
                slot = sa[1]
                slots = set()
                for r, src in enumerate(leaves):
                    lx, kk, la = self.resolve(src, ev)
                    ld = decode(la + 4 * kk, info)
                    if ld is None:
                        raise Fail(f"term {r} load address does not decode (rank {rank})")
                    if r == rank:
                        if ld[0] != "in" or ld[3] != b + e or ld[4] != w:
                            raise Fail(f"term {r} for rank {rank}: reads {ld} (expected local in[{b + e}] word {w})")
                        loadkinds.add(("local", lx.op))
                    else:
                        if ld[0] != "recv" or ld[2] != r or ld[3] != e or ld[4] != w:
                            raise Fail(f"term {r} for rank {rank}: reads {ld} (expected recv[slot][{r}][{e}] word {w})")
                        slots.add(ld[1])
                        loadkinds.add(("recv", lx.op))
                if len(slots) != 1 or (slot is not None and slot not in slots):
                    raise Fail(f"receive slots {sorted(slots)} vs store slot {slot}")
            if n_exec < 16:
                raise Fail(f"store executes in only {n_exec} environments")
            recs.append({"lane": f"{'f32' if self.f32 else 'bf16'} w{w}{'' if self.f32 else '.' + half}",
                         "fadds": [fi for fi, _ in ch], "pack": ln[3] if len(ln) > 3 else None,
                         "loads": sorted(loadkinds)})
        region = None
        env, info = make_env(0, 1)
        sa = decode(S.addr_value(S.Evaluator(K, env, 1), x.vals["addr"]), info)
        region = sa[0] if sa else "?"
        for r in recs:
            r["store"] = region
            self.records.append(r)
        return covered

    def check_copy(self, x):
        """every data word of store x is a select of loads that, for every rank where the store executes, resolves to
        a load (never to the select's default arm): the stored value is a bit copy."""
        K = self.k
        seen = set()
        n_exec = 0
        for d in x.vals["data"]:
            if not self.is_wordsrc(d):
                raise Fail(f"data word {K.N[d][0]} is not a select of loads")
        for rank, info, ev in self.envs(x.block):
            g = ev.strict(x.vals["guard"])
            if g == 0:
                continue
            n_exec += 1
            sa = decode(S.addr_value(ev, x.vals["addr"]), info)
            for w, d in enumerate(x.vals["data"]):
                lx, kk, la = self.resolve(d, ev)
                ld = decode(la + 4 * kk, info)
                seen.add((sa[0] if sa else "?", ld[0] if ld else "?", lx.op))
        if not n_exec:
            raise Fail("the store's guard is false in every environment (cannot prove)")
        return {"store": f"/*{x.addr:04x}*/", "copies": sorted(seen)}

    def canonical(self):
        """order-free canonical record of the proven reduction (what must match the production twin)."""
        out = []
        for r in self.records:
            out.append(f"{r['store']}:{r['lane']}: bf16_rn? {'no' if self.f32 else 'yes'} = FADD.RN serial "
                       f"(+0.0, r0..r7) loads {r['loads']}")
        return sorted(out)


# ------------------------------------------------------------------------------------------------------------------
def load_text(path):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt").read()
    if path.endswith((".so", ".o", ".cubin", ".fatbin")):
        return subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True, check=True).stdout
    return open(path).read()


def norm_ins(s):
    s = re.sub(r"c\[0x4\]\[0x[0-9a-f]+\]", "c[0x4][X]", s)
    return re.sub(r"\(\*\"[^\"]*\"\*\)", "", s)


def kernel_key(name):
    m = re.search(r"(ar_kernel_r3|ar_kernel_pl|ar_kernel|pingpong_kernel)(I\w+?EE)?", name)
    k = m.group(0) if m else name
    return re.sub(r"ELi0EE$", "EE", k)


def twin_key(mode, f32):
    return f"ar_kernel_plILi{mode}ELb{f32}ELi8ELi{7 if mode == 2 else 5}EE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--json")
    ap.add_argument("--mutant", action="store_true", help="expect FAIL: exit 0 only if every reduction variant fails")
    ap.add_argument("--show", default=None, help="print the ordered FP stream of kernels matching this regex")
    ap.add_argument("--no-identity", action="store_true", help="skip the ref-kernels-identical check (mutant runs)")
    ap.add_argument("--mutant-skip-mask", type=int, default=0,
                    help="--mutant only: kernels whose mask has any of these bits are not required to fail "
                         "(m_stage: 64 = the send-buffer kernels, which have no staging copy)")
    ap.add_argument("--expect", default=None,
                    help="--mutant only: every reduction kernel must fail with an error matching this regex")
    a = ap.parse_args()
    ref = {kernel_key(n): (n, b) for n, b in S.split_functions(load_text(a.ref)).items()}
    cand = {kernel_key(n): (n, b) for n, b in S.split_functions(load_text(a.cand)).items()}
    report = {"ref": a.ref, "cand": a.cand, "identity": {}, "kernels": {}}
    ok = True
    # 1. identity of every ref kernel in the candidate (flag-off path)
    nid = 0
    if not a.no_identity:
        for k in sorted(ref):
            x = [norm_ins(t) for _, t in ref[k][1]]
            y = [norm_ins(t) for _, t in cand.get(k, ("", []))[1]]
            same = x == y and len(x) > 0
            nid += same
            report["identity"][k] = same
            if not same:
                ok = False
                print(f"IDENTITY FAIL {k}: {len(x)} vs {len(y)} instructions")
        print(f"IDENTITY: {nid}/{len(ref)} ref kernels instruction-identical in the candidate")
    # 2. dataflow proof of every production twin (ref) and every candidate reduction / gather kernel
    def gate(tag, key, name, body):
        g = KernelGate(name, body)
        g.run()
        report["kernels"][f"{tag}:{key}"] = {"errors": g.errors, "records": g.canonical(), "stream": g.stream,
                                            "ignored": g.ignored, "fp_ops": len(g.stream), "staging": g.staging}
        return g

    global PROJ_PROOFS
    if not a.mutant:  # r8: prove the kR7Proj projection tails first (cached, parallel); --mutant keeps r7's behaviour
        import projgate as P
        PROJ_PROOFS = P.prove_all([(n, b) for n, b in cand.values()
                                   if re.search(r"ar_kernel_r3ILi\d+ELb\d+ELi8ELi(\d+)EE", n) and
                                   int(re.search(r"ELi8ELi(\d+)EE", n).group(1)) & 512])
    twins = {}
    for k, (n, body) in sorted(ref.items()):
        m = re.search(r"ar_kernel_plILi(\d)ELb(\d)ELi8ELi(\d+)EE", k)
        if not m or int(m.group(3)) not in (5, 7):
            continue
        g = gate("ref", k, n, body)
        twins[k] = g
        status = "PASS" if not g.errors else "FAIL"
        print(f"ref  {k:40s} spec {status} lanes {len(g.records):2d} fp-ops {len(g.stream):3d} ignored "
              f"{len(g.ignored):2d}" + ("" if not g.errors else f"  first error: {g.errors[0]}"))
        if g.errors:
            ok = False
    results = []
    for k, (n, body) in sorted(cand.items()):
        m = re.search(r"ar_kernel_r3ILi(\d)ELb(\d)ELi8ELi(\d+)EE", k)
        if not m:
            continue
        mode, f32 = int(m.group(1)), int(m.group(2))
        g = gate("cand", k, n, body)
        tw = twins.get(twin_key(mode, f32))
        same = tw is not None and g.canonical() == tw.canonical()
        good = not g.errors and same
        if not g.errors and not same:
            g.errors.append(f"canonical dataflow differs from production twin {twin_key(mode, f32)}")
            report["kernels"][f"cand:{k}"]["errors"] = g.errors
        order = tw is not None and [s for s in g.stream] == [s for s in tw.stream]
        results.append((mode, f32, good))
        report["kernels"][f"cand:{k}"]["pass"] = good
        print(f"cand {k:40s} {'PASS' if good else 'FAIL'} lanes {len(g.records):2d} fp-ops {len(g.stream):3d} "
              f"ignored {len(g.ignored):2d} same-dataflow-as-twin {'YES' if same else 'NO '} "
              f"same-program-order {'YES' if order else 'no '} staging {g.staging.get('proven', 0):3d} copies/"
              f"{g.staging.get('stores', 0)} stores, unrolled {'YES' if g.staging.get('unrolled_blocks') else 'no '}"
              + ("" if good else f"  first error: {g.errors[0]}"))
        if a.show and re.search(a.show, k):
            for s_ in g.stream:
                print("   ", s_)
            for ig in g.ignored:
                print("    ignored:", ig)
    if a.show:
        for k, g in twins.items():
            if re.search(a.show, k):
                for s_ in g.stream:
                    print("   ref", s_)
    n_cand = len(results)
    if n_cand == 0:
        print("no ar_kernel_r3 kernels in the candidate")
        ok = False
    cand_ok = all(gd for _, _, gd in results)
    if a.mutant:
        # KILLED only if EVERY candidate reduction kernel (one-shot and two-shot, bf16 and fp32, every mask) fails, for
        # the expected reason when --expect is given, while the production twins in the same binary still pass
        red = [(k, v) for k, v in report["kernels"].items() if re.search(r"^cand:ar_kernel_r3ILi[02]E", k)
               and not (int(re.search(r"ELi(\d+)EE", k).group(1)) & a.mutant_skip_mask)]
        caught = [k for k, v in red if not v.get("pass") and
                  (a.expect is None or any(re.search(a.expect, e) for e in v["errors"]))]
        verdict = bool(red) and len(caught) == len(red) and all(not g.errors for g in twins.values())
        print(f"MUTANT: reduction kernels caught {len(caught)}/{len(red)}"
              + (f" (reason /{a.expect}/)" if a.expect else "") + f" -> {'KILLED' if verdict else 'SURVIVED'}")
        report["verdict"] = "KILLED" if verdict else "SURVIVED"
        rc = 0 if verdict else 1
    else:
        verdict = ok and cand_ok
        print(f"FPFLOW_GATE {'PASS' if verdict else 'FAIL'}: {sum(gd for _, _, gd in results)}/{n_cand} r3 kernels "
              f"proven, {sum(not g.errors for g in twins.values())}/{len(twins)} production twins proven")
        report["verdict"] = "PASS" if verdict else "FAIL"
        rc = 0 if verdict else 1
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
