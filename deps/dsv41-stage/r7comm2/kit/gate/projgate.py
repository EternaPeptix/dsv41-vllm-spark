#!/usr/bin/env python3
"""projgate.py: the r7-comm2 HT13 dataflow gate (CPU only, from `cuobjdump -sass` text).

    python3 -I projgate.py --ref ref/triton/proj_rows.<hash>/_proj_rows.sass --cand sass/<r7>.sass [--json out.json]
                           [--mutant --expect REGEX]

The served projection is the Triton kernel dsv41_oblit._proj_rows (H = 5120, BLOCK 8192, 8 warps, one CTA per row).
HT13 computes the same thing inside ar_kernel_r3 (instantiation bit kR7Proj = 512) on the AR's own bf16 output. This
gate PROVES, separately for the Triton cubin and for every kR7Proj kernel of the candidate, the complete per-row op
order from the SASS dataflow, and requires the two canonical records to be equal:

  1. every y store (STG whose words are F2FP.BF16.F32.PACK_AB of FFMA) stores, for element e of row r,
       bf16_rn( FFMA(-s_r, d[e], x_r[e]) )        (one FFMA, the negation on the s operand, no other modifier)
     x_r[e] = the exact bf16 widening of the matching half of the loaded word of row r element e (Triton: X; r7: p.out),
     d[e]  = the fp32 word e of D (Triton: D; r7: q.d); addresses are evaluated concretely per thread and decoded;
  2. s_r is a shared-memory word; the gate finds, by evaluating every STS address and guard for every thread of the
     row's 256-thread group, the ONE thread that writes it; its value must be the level-2 tree
       v = LDS(slot of lane l); v = v + SHFL.BFLY(v, m) for m = 4, 2, 1   (FADD, plain)
     and lane l (l = 0..7) of that warp must read the slot written by exactly one thread, lane 0 of warp w(l); the
     record keeps the map l -> w(l);
  3. each level-1 value is the butterfly a = a + SHFL.BFLY(a, m) for m = 16, 8, 4, 2, 1 of the thread's chain;
  4. each thread's chain is FMUL(x_i0, d_j0) followed by FFMA(x_i, d_j, acc) (acc in the addend slot, plain FFMA, no
     predicate, no modifier); the record keeps, for every thread t of the group, the ordered list of (x element, d
     element) pairs, a masked (zero) operand recorded as Z (the mask predicates are evaluated per thread);
  5. every FFMA / FMUL / FADD / F2FP reachable in the projection (Triton: the whole kernel) is in a proven chain, tree
     or y form, and nothing else FP-class is (I2F/MUFU/HFMA2 constants are reported, never in a value slice).
The record is evaluated for rows 0..5 (r7: q.rows = 6, groups g = 0/1, row = 2k + g; Triton: CTA r) and threads
t = 0..255 of each group. Equal records mean equal IEEE op sequences on equal operands for every output element, i.e.
bitwise-identical y (modulo NaN payload propagation, which the same FFMA/FADD on the same operands reproduces).
Anything not modelled fails closed. --mutant: exit 0 only if every candidate kR7Proj kernel FAILS (for --expect)."""
import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sassir as S  # noqa: E402
import fpflow as F  # noqa: E402

H = 5120
ROWS = 6
RX, RY, RD, ROUT = 21, 22, 23, 2  # region tags (address >> 40); ROUT = fpflow's "out" (index 1 + 1)
Q_OFF = 0x380 + 0x90  # r7: R7Proj q follows Params (sizeof 0x90) in the parameter bank


class Fail(Exception):
    pass


def put64(cb, off, v):
    cb[(0, off)] = v & S.M32
    cb[(0, off + 4)] = (v >> 32) & S.M32


def env_triton(t, row):
    cb = {}
    put64(cb, 0x380, RX << 40)
    put64(cb, 0x388, RY << 40)
    put64(cb, 0x390, RD << 40)
    cb[(0, 0x398)] = H
    cb[(0, 0x39c)] = H
    cb[(0, 0x3a0)] = H
    cb[(0, 0x360)], cb[(0, 0x364)], cb[(0, 0x368)] = 256, 1, 1
    cb[(0, 0x370)], cb[(0, 0x374)], cb[(0, 0x378)] = ROWS, 1, 1
    sr = {"SR_TID.X": t, "SR_TID.Y": 0, "SR_TID.Z": 0, "SR_CTAID.X": row, "SR_CTAID.Y": 0, "SR_CTAID.Z": 0,
          "SR_LANEID": t & 31, "SR_CgaCtaId": 0}
    return S.Env(cb, sr, seed=t * 7 + row)


def env_r7(tid, cta=0, nctaid=1, nvec=512, div=0):
    """r8-comm2fix: the r7 environment must REACH the projection tail. sassir follows loops on their iteration-1 path
    only, so code after a loop is reachable only when the loop runs 0 or 1 times for this thread. r7's env used
    nvec = ROWS * H / 8 = 3840 on one CTA: every thread entered the one-shot reduce loop and never left it at
    iteration 1, so every y store was unreachable and projgate found 0/5120 stores (R3 0/18) on a correct kernel.
    The projection reads only threadIdx, q and p.out (through memory), so the AR's message length does not enter it;
    the gate runs it in several loop / grid / divergence environments (CONFIGS) and requires the same record in all.
    div: SR_WARPDIV, the BRA.DIV condition (0: the warp is converged, the fast path; 1: the WARPSYNC.COLLECTIVE
    fallback)."""
    env, _ = F.make_env(0, tid, tid=tid, cta=cta, nvec=nvec, nctaid=nctaid)
    cb = env.cb
    put64(cb, Q_OFF + 0x00, RD << 40)
    put64(cb, Q_OFF + 0x08, RY << 40)
    put64(cb, Q_OFF + 0x10, ROWS)
    env.sr["SR_LANEID"] = tid & 31
    env.sr["SR_CgaCtaId"] = 0
    env.sr["SR_WARPDIV"] = div
    return env


# r8: (name, projecting CTA (non-P2; P2 kernels always project in CTA 0), gridDim.x, nvec, SR_WARPDIV). Each config
# makes every AR loop run 0 or 1 times for every thread of the projecting CTA (see env_r7), with the arrive3 atomic
# returning gridDim.x - 1 (this CTA is the last one).
CONFIGS = [
    ("g1-trip1", 0, 1, 512, 0),      # 1 CTA, every thread runs each AR loop once (two-shot: tid < chunk 64 only)
    ("g1-trip1-div", 0, 1, 512, 1),  # the same, the level-2 shuffles on their divergence fallback
    ("g1-trip0", 0, 1, 0, 0),        # no AR loop runs at all
    ("g3-last2", 2, 3, 1536, 0),     # 3 CTAs, CTA 2 projects (blockIdx.x != 0); its threads run each loop once
]


def decode(addr, kind):
    """-> (array, row, element) for a byte address of x / y / d; None when it does not decode."""
    reg, off = addr >> 40, addr & ((1 << 40) - 1)
    if reg == RD:
        return ("d", None, off // 4) if off % 4 == 0 else None
    if (kind == "triton" and reg == RX) or (kind == "r7" and reg == ROUT):
        a = "x"
    elif reg == RY:
        a = "y"
    else:
        return None
    if off % 2:
        return None
    return (a, off // (2 * H), (off % (2 * H)) // 2)


class ProjGate(F.KernelGate):
    def __init__(self, name, text, kind):
        super().__init__(name, text)
        self.kind = kind
        self.covered = set()
        self.rec = None
        self.used_loads = set()   # r8: global loads proven to be x / d operands (ordgate O3 gates exactly these)
        self.proven_sts = set()   # r8: the level-1 / level-2 shared partial stores proven above (fpflow exempts them)
        self.ystore_exec = set()  # r8: the y stores that executed (and were proven) in some environment

    def is_zero(self, n):
        t = self.k.N[n]
        if t == ("c", 0) or n == self.k.ZERO:
            return True
        if t[0] == "op" and self.k.ins[t[1]].base == "CS2R":
            return all(self.k.N[i] == ("sr", "SRZ") for i in t[3]) and len(t[3]) > 0
        if t[0] == "op":  # r8: `MOV Rd, RZ` (r7's masked x words: @!P LDG Rd after MOV Rd, RZ)
            x = self.k.ins[t[1]]
            if x.op == "MOV" and len(t[3]) == 1 and len(x.ops) == 2 and not (x.ops[1].neg or x.ops[1].abs or
                                                                              getattr(x.ops[1], "inv", False)):
                return self.is_zero(t[3][0])
        return False

    def unwrap_neg(self, n):
        """(inner node, negated?) for an operand node: sassir wraps -R as ('mod', ('neg', ...), inner)."""
        t = self.k.N[n]
        if t[0] == "mod":
            inner = t[2] if not isinstance(t[2], tuple) else t[2][0]
            return inner, "neg" in str(t[1])
        return n, False

    # a word source may also be a select (ite / SEL) whose other arm is the constant 0 (a masked load)
    def is_wordsrc(self, n, depth=0):
        t = self.k.N[n]
        if t[0] == "ld":
            return True
        if depth > 32:
            return False
        if t[0] == "ite":
            arms = (t[2], t[3])
        elif t[0] == "op" and self.k.ins[t[1]].base == "SEL" and len(t[3]) >= 3:
            arms = (t[3][0], t[3][1])
        else:
            return False
        ok = [self.is_wordsrc(a, depth + 1) or self.is_zero(a) for a in arms]
        return all(ok) and any(self.is_wordsrc(a, depth + 1) for a in arms)

    def resolve0(self, src, ev):
        """(load ins, word, address) or 'Z' (the zero arm of a masked load) under evaluator ev."""
        K = self.k
        for _ in range(256):
            t = K.N[src]
            if t[0] == "ld":
                return K.ins[t[1]], t[2], S.addr_value(ev, t[3])
            if self.is_zero(src):
                return "Z"
            if t[0] == "ite":
                p = ev.strict(t[1])
                if p is None:
                    raise Fail("mask predicate not concrete")
                src = t[2] if p else t[3]
                continue
            if t[0] == "op" and K.ins[t[1]].base == "SEL":
                x = K.ins[t[1]]
                p = ev.strict(t[3][2])
                if p is None:
                    raise Fail("SEL predicate not concrete")
                neg = x.ops[3].notp if hasattr(x.ops[3], "notp") else False
                p = (not p) if neg else bool(p)
                src = t[3][0] if p else t[3][1]
                continue
            if t[0] == "phi":
                src, ev = ev.phi_pick(src)
                continue
            raise Fail(f"word source resolves to {t[0]}")
        raise Fail("select chain too deep")

    def operand(self, n, ev, want):
        """x / d leaf of a chain or y FFMA under ev -> ('Z',) | (array, row, element)."""
        L = self.leaf(n)
        if L is None:
            raise Fail(f"{want} operand is not a pure bit move of one loaded word")
        half, src = L
        if (want == "d") != (half == "raw"):
            raise Fail(f"{want} operand is a {half} leaf")
        r = self.resolve0(src, ev)
        if r == "Z":
            return ("Z",)
        lx, k, a = r
        self.used_loads.add(lx.idx)
        dd = decode(a + 4 * k, self.kind)
        if dd is None:
            raise Fail(f"{want} load address {a:#x} does not decode")
        arr, row, e = dd
        if arr != want:
            raise Fail(f"{want} operand reads array {arr}")
        if want == "x":
            if half not in ("lo", "hi"):
                raise Fail("x operand is not a bf16 half")
            e += 1 if half == "hi" else 0
        elif half != "raw":
            raise Fail("d operand is not a raw fp32 word")
        return (arr, row, e)

    def _op(self, n, base):
        t = self.k.N[n]
        if t[0] != "op":
            return None
        x = self.k.ins[t[1]]
        return x if x.base == base else None

    def plain(self, x, allow_neg=()):
        if x.op not in ("FFMA", "FMUL", "FADD"):
            raise Fail(f"{x.text} /*{x.addr:04x}*/: modifier (served: plain RN, no FTZ)")
        if x.gpred != "PT" or x.gneg:
            raise Fail(f"predicated {x.base} /*{x.addr:04x}*/")
        for i, o in enumerate(x.ops[1:], 1):
            if (o.neg and i not in allow_neg) or o.abs or getattr(o, "inv", False):
                raise Fail(f"operand modifier in {x.text} /*{x.addr:04x}*/")

    def dot_chain(self, n, ev):
        """[(x, d)] of the serial chain rooted at n (FMUL first)."""
        K = self.k
        out = []
        cur = n
        for _ in range(64):
            t = K.N[cur]
            x = K.ins[t[1]] if t[0] == "op" else None
            if x is not None and x.base == "FADD" and len(t[3]) == 2:
                # r8-comm2fix: a term whose x AND d are the compile-time constant +0.0 (r7: chunk c = 3, elements
                # 6144 + 8t >= 5120, masked for every thread) is folded by nvcc from FFMA(+0, +0, acc) into
                # FADD(+0, acc) (RZ, no modifier). Bitwise the same op: +0 * +0 = +0 exactly, then one RN add of +0
                # to acc (acc = -0 -> +0 in both; NaN, Inf, subnormals identical; neither flushes). Recorded as the
                # masked term (Z, Z), exactly what the served kernel's runtime-masked FFMA(+0, +0, acc) records.
                # FADD.FTZ / a negated or |.| operand / a nonzero constant are refused (plain + is_zero).
                self.plain(x)
                z = [i for i in (0, 1) if self.is_zero(t[3][i])]
                if len(z) != 1:
                    raise Fail(f"chain node {x.text} /*{x.addr:04x}*/ is an FADD that is not acc + RZ")
                self.covered.add(x.idx)
                out.append((("Z",), ("Z",)))
                cur = t[3][1 - z[0]]
                continue
            if x is None or x.base not in ("FFMA", "FMUL"):
                raise Fail(f"chain node is {t[0]} {x.text if x else ''}")
            self.plain(x)
            self.covered.add(x.idx)
            a, b = t[3][0], t[3][1]
            pair = []
            for o in (a, b):
                L = self.leaf(o)
                if L is None:
                    raise Fail(f"/*{x.addr:04x}*/ operand is not a pure bit move of one loaded word")
                kind = "d" if L[0] == "raw" else "x"  # fp32 word: d; bf16 half: x
                pair.append((kind, self.operand(o, ev, kind)))
            kinds = sorted(p[0] for p in pair)
            if kinds != ["d", "x"]:
                raise Fail(f"/*{x.addr:04x}*/ multiplies {kinds} (expected one x and one d leaf)")
            xv = [p[1] for p in pair if p[0] == "x"][0]
            dv = [p[1] for p in pair if p[0] == "d"][0]
            out.append((xv, dv))
            if x.base == "FMUL":
                break
            cur = t[3][2]
        else:
            raise Fail("chain longer than 64")
        out.reverse()
        return out

    def see(self, n, ev):
        """r8: n through value-preserving plumbing under ev: a phi (the incoming value this environment takes: r7's
        level-2 tree re-joins the BRA.DIV fallback at its last FADD) and a plain `MOV Rd, Rs` (the fallback copies v
        into the SHFL source register; its lane-mask / clamp operands are MOV / HFMA2 constants)."""
        K = self.k
        for _ in range(64):
            t = K.N[n]
            if t[0] == "phi" and ev is not None:
                n, ev = ev.phi_pick(n)
                continue
            if t[0] == "op":
                x = K.ins[t[1]]
                if x.op == "MOV" and len(t[3]) == 1 and len(x.ops) == 2 and x.gpred == "PT" and not x.gneg and \
                        not (x.ops[1].neg or x.ops[1].abs or getattr(x.ops[1], "inv", False)):
                    n = t[3][0]
                    continue
            return n
        raise Fail("phi / MOV chain too deep")

    def butterfly(self, n, masks, ev=None):
        """root of v = v + SHFL.BFLY(v, m) for m in masks (outermost last) -> innermost v node."""
        K = self.k
        cur = self.see(n, ev)
        for m in reversed(masks):
            x = self._op(cur, "FADD")
            if x is None:
                raise Fail(f"tree level xor {m}: not an FADD")
            self.plain(x)
            self.covered.add(x.idx)
            a, b = self.see(K.N[cur][3][0], ev), self.see(K.N[cur][3][1], ev)
            got = None
            for v, w in ((a, b), (b, a)):
                sx = self._op(w, "SHFL")
                if sx is not None and "BFLY" in sx.mods and len(K.N[w][3]) == 3 and self.see(K.N[w][3][0], ev) == v:
                    mk = K.N[self.see(K.N[w][3][1], ev)]
                    cl = K.N[self.see(K.N[w][3][2], ev)]
                    if mk == ("c", m) and cl == ("c", 0x1f):
                        got = v
            if got is None:
                raise Fail(f"tree level: expected FADD(v, SHFL.BFLY(v, {m}, 0x1f)) at /*{x.addr:04x}*/")
            cur = got
        return cur

    def fwd(self, i, stop_bar=False):
        """r8: instruction indices reachable from instruction i (exclusive) over forward CFG edges (no back edges);
        stop_bar: paths end at a BAR.SYNC (the BAR itself is included). Replaces r7's program-order tests, which
        cannot see an out-of-line path (the BRA.DIV fallback stores s at /*5800*/, after its reader at /*4a70*/)."""
        key = (i, stop_bar)
        memo = self.__dict__.setdefault("_fwd", {})
        if key in memo:
            return memo[key]
        K = self.k
        out, stack, seen = set(), [(K.ins[i].block, i + 1)], set()
        while stack:
            b, j = stack.pop()
            blk = K.blocks[b]
            stopped = False
            for q in range(j, blk[-1] + 1):
                out.add(q)
                if stop_bar and K.ins[q].base == "BAR":
                    stopped = True
                    break
            if stopped:
                continue
            for s_ in K.succ[b]:
                if (b, s_) in K.back or s_ in seen:
                    continue
                seen.add(s_)
                stack.append((s_, K.blocks[s_][0]))
        memo[key] = out
        return out

    def sts_writers(self, envs_of, addr, reader_idx):
        """[(thread, STS ins)]: the LAST store to shared address `addr` before the reading LDS (guard true, block not
        unreachable) among threads envs_of; a BAR.SYNC must separate it from the reader on every path. Slots reused
        later (Triton writes the total into slot 0 after reading it) are thereby not mistaken for the writer.
        r8: "before" / "last" / "separated" are CFG properties (forward edges), not program order."""
        K = self.k
        sts = [x for x in K.ins if x.base == "STS" and x.vals and "addr" in x.vals and x.block in K.reach and
               reader_idx in self.fwd(x.idx)]
        hits = []
        for tid, ev in envs_of:
            for x in sts:
                if ev.reach(x.block) == 0:
                    continue
                g = ev.strict(x.vals["guard"])
                if g == 0:
                    continue
                try:
                    a = S.addr_value(ev, x.vals["addr"])
                except Exception:  # noqa: BLE001
                    raise Fail(f"STS /*{x.addr:04x}*/ address not concrete")
                if a == addr:
                    if g is None:
                        raise Fail(f"STS /*{x.addr:04x}*/ guard not concrete for thread {tid}")
                    hits.append((tid, x))
        if not hits:
            return hits
        ins = {x.idx for _, x in hits}
        # the last writer: no other writing instruction lies on a path from it to the reader
        last = [i for i in ins if not any(j != i and j in self.fwd(i) and reader_idx in self.fwd(j) for j in ins)]
        if len(last) != 1:
            raise Fail(f"shared slot {addr:#x}: {len(last)} last writers {[f'/*{K.ins[i].addr:04x}*/' for i in last]}")
        last = last[0]
        hits = [h for h in hits if h[1].idx == last]
        if reader_idx in self.fwd(last, stop_bar=True):
            raise Fail(f"no BAR between the STS /*{K.ins[last].addr:04x}*/ and its reader on every path")
        return hits

    def lds_addr(self, n, ev):
        """shared byte address read by tree input n under ev (a predicated LDS is ite(p, ld, old): p must hold)."""
        K = self.k
        for _ in range(16):
            t = K.N[n]
            if t[0] == "ite":
                p = ev.strict(t[1])
                if p is None:
                    raise Fail("predicated shared load: predicate not concrete")
                n = t[2] if p else t[3]
                continue
            break
        t = K.N[n]
        if t[0] != "ld" or K.ins[t[1]].base != "LDS":
            raise Fail(f"tree input is {t[0]}, not a shared-memory load")
        return S.addr_value(ev, t[3]) + 4 * t[2]

    def lds_idx(self, n, ev):
        K = self.k
        while K.N[n][0] == "ite":
            n = K.N[n][2] if ev.strict(K.N[n][1]) else K.N[n][3]
        return K.N[n][1]

    # -------------------------------------------------------------------------------------------------------------
    def arrivals(self):
        """r8: indices of the arrive3 atomic (ATOM*.ADD at st + 0x54) of a kR7Proj kernel."""
        if getattr(self, "_arr", None) is None:
            K = self.k
            env, _ = F.make_env(0, 1)
            st = (F.REGIONS.index("st") + 1) << 40
            arr = []
            for x in K.ins:
                if x.base in ("ATOMG", "ATOM") and "ADD" in x.mods and x.vals and "addr" in x.vals:
                    try:
                        a = S.addr_value(S.Evaluator(K, env, 1), x.vals["addr"])
                    except Exception:  # noqa: BLE001
                        continue
                    if a == st + 0x54:
                        arr.append(x.idx)
            self._arr = arr
        return self._arr

    def groups(self, cfg=None):
        """[(row, [(t, tid, evaluator)])] with tid the hardware thread of group thread t."""
        out = []
        for row in range(ROWS):
            thr = []
            for t in range(256):
                if self.kind == "triton":
                    ev = S.Evaluator(self.k, env_triton(t, row), 1)
                    thr.append((t, t, ev))
                else:
                    g, k = row % 2, row // 2
                    tid = g * 256 + t
                    name, cta, G, nvec, div = cfg
                    if self.mask & 32:
                        cta = 0  # P2: CTA 0 alone reduces and projects
                    env = env_r7(tid, cta=cta, nctaid=G, nvec=nvec, div=div)
                    for i in self.arrivals():
                        env.loads[i] = G - 1
                    ev = S.Evaluator(self.k, env, 1)
                    thr.append((t, tid, ev))
            out.append((row, thr))
        return out

    def ystores(self):
        K = self.k
        res = []
        for x in K.ins:
            if x.base != "STG" or not x.vals or "data" not in x.vals:
                continue
            ok = True
            for d in x.vals["data"]:
                px = self._op(d, "F2FP")
                if px is None:
                    ok = False
            if ok:
                res.append(x)
        return res

    def run_proj(self):
        """Triton: one proof. r7 (r8): the proof in every CONFIGS environment; the records must be equal. The union
        of what each proof covered (FP ops, shared partial stores, operand loads) is kept for fpflow / ordgate."""
        if self.kind == "triton":
            return self.run_proj_env(None)
        if self.mask & 32 == 0 and len(self.arrivals()) != 1:
            raise Fail(f"expected one arrive3 atomic (st + 0x54), found {len(self.arrivals())}")
        recs = {}
        for cfg in CONFIGS:
            try:
                recs[cfg[0]] = canon(self.run_proj_env(cfg))
            except (Fail, F.Fail) as e:
                raise Fail(f"[{cfg[0]}] {e}")
            except (S.Ambiguous, S.Unknown) as e:
                raise Fail(f"[{cfg[0]}] {type(e).__name__}: {e}")
        if len(set(recs.values())) != 1:
            raise Fail(f"the projection record depends on the environment: {sorted(recs)}")
        return self.rec

    def run_proj_env(self, cfg):
        K = self.k
        ys = self.ystores()
        if not ys:
            raise Fail("no y store (STG of F2FP words)")
        rec = {}
        cache_chain = {}
        executed = set()
        for row, thr in self.groups(cfg):
            envs = [(tid, ev) for _, tid, ev in thr]
            ev_of = {tid: ev for _, tid, ev in thr}
            elems = {}
            s_info = None
            for t, tid, ev in thr:
                for x in ys:
                    if ev.reach(x.block) == 0:
                        continue
                    g = ev.strict(x.vals["guard"])
                    if g == 0:
                        continue
                    a = S.addr_value(ev, x.vals["addr"])
                    dy = decode(a, self.kind)
                    if dy is None or dy[0] != "y":
                        continue  # not this row's projection store (or another region)
                    executed.add(x.idx)
                    self.ystore_exec.add(x.idx)
                    if dy[1] != row:
                        continue  # r7: the same store instruction serves another row slot; it is checked there
                    for w, dn in enumerate(x.vals["data"]):
                        px = self._op(dn, "F2FP")
                        if px.op != "F2FP.BF16.F32.PACK_AB" or any(o.neg or o.abs for o in px.ops[1:3]):
                            raise Fail(f"{px.text}: not the plain RN pack")
                        self.covered.add(px.idx)
                        for half, fn in (("hi", K.N[dn][3][0]), ("lo", K.N[dn][3][1])):
                            fx = self._op(fn, "FFMA")
                            if fx is None:
                                raise Fail(f"y word {w} {half}: not an FFMA (served: one FFMA, single rounding)")
                            self.plain(fx, allow_neg=(1, 2))
                            self.covered.add(fx.idx)
                            e = dy[2] + 2 * w + (1 if half == "hi" else 0)
                            ops3 = K.N[fn][3]
                            negs = [fx.ops[1].neg, fx.ops[2].neg]
                            if sum(negs) != 1:
                                raise Fail(f"{fx.text}: expected exactly one negated product operand (-s)")
                            sn, dn_ = (ops3[0], ops3[1]) if negs[0] else (ops3[1], ops3[0])
                            dv = self.operand(dn_, ev, "d")
                            xv = self.operand(ops3[2], ev, "x")
                            sn, sneg = self.unwrap_neg(sn)
                            if not sneg:
                                raise Fail(f"{fx.text}: the negated operand is not a plain -s")
                            sa = self.lds_addr(sn, ev)
                            s_reader = self.k.N[sn][1]
                            if xv != ("x", row, e) or dv != ("d", None, e):
                                raise Fail(f"row {row} y[{e}] = fma(-s, {dv}, {xv})")
                            if s_info is None:
                                s_info = sa
                            elif s_info != sa:
                                raise Fail(f"row {row}: two s slots {s_info:#x} {sa:#x}")
                            if e in elems:
                                raise Fail(f"row {row} y[{e}] stored twice")
                            elems[e] = (t, f"/*{x.addr:04x}*/")
            if sorted(elems) != list(range(H)):
                missing = sorted(set(range(H)) - set(elems))
                raise Fail(f"row {row}: y elements stored {len(elems)}/{H} (first missing {missing[:3]})")
            # level 2: the single writer of s
            wr = self.sts_writers(envs, s_info, s_reader)
            if len(wr) != 1:
                raise Fail(f"row {row}: s slot written by {len(wr)} threads/instructions (expected 1)")
            wtid, wx = wr[0]
            self.proven_sts.add(wx.idx)
            v0 = self.butterfly(wx.vals["data"][0], [4, 2, 1], ev_of[wtid])
            lanes = {}
            base = wtid & ~31
            for l in range(8):
                lev = ev_of.get(base + l)
                if lev is None:
                    raise Fail("level-2 lane outside the group")
                la = self.lds_addr(v0, lev)
                w1 = self.sts_writers(envs, la, self.lds_idx(v0, lev))
                if len(w1) != 1:
                    raise Fail(f"row {row}: level-2 lane {l} slot written by {len(w1)} (expected 1)")
                t1, x1 = w1[0]
                self.proven_sts.add(x1.idx)
                if t1 & 31:
                    raise Fail(f"row {row}: level-1 slot written by lane {t1 & 31} (expected lane 0)")
                lanes[l] = (t1 - (wtid & ~255)) >> 5
                cn = self.butterfly(x1.vals["data"][0], [16, 8, 4, 2, 1], ev_of[t1])
                for L in range(32):
                    ct = t1 + L
                    key = (row, ct)
                    if key not in cache_chain:
                        cache_chain[key] = self.dot_chain(cn, ev_of[ct])
            if sorted(lanes.values()) != list(range(8)):
                raise Fail(f"row {row}: level-2 lanes read warps {lanes}")
            chains = [cache_chain[(row, base_t)] for base_t in sorted(k[1] for k in cache_chain if k[0] == row)]
            if len(chains) != 256:
                raise Fail(f"row {row}: {len(chains)} chains (expected 256)")
            norm = []
            for ch in chains:
                norm.append(tuple((("Z",) if xv == ("Z",) else ("x", xv[2]) if xv[1] == row else ("x?", xv),
                                   ("Z",) if dv == ("Z",) else ("d", dv[2])) for xv, dv in ch))
            rec[row] = {"level2_lane_to_warp": [lanes[l] for l in range(8)], "level2_masks": [4, 2, 1],
                        "level1_masks": [16, 8, 4, 2, 1], "chains": norm,
                        "y": "bf16_rn(FFMA(-s, d[e], x[e])) for e in 0..5119, one store each"}
        self.rec = rec
        # 5. stray FP inside the projection. A y store whose guard is false for every thread and row (Triton's 4th
        #    8192-block chunk is fully masked at H = 5120) is dead: its FP slice is exempt, but only the part that no
        #    executed store uses, and it is reported.
        fp = [x for x in K.ins if x.base in ("FFMA", "FMUL", "FADD", "F2FP")]
        dead = [x for x in ys if x.idx not in executed]
        live_slice = set()
        for x in ys:
            if x.idx in executed:
                live_slice |= K.slice_ins(x.vals["data"], value_only=True)
        dead_only = set()
        for x in dead:
            dead_only |= K.slice_ins(x.vals["data"], value_only=True)
        dead_only -= live_slice
        self.dead = [f"/*{x.addr:04x}*/ {x.text}" for x in dead]
        if self.kind == "triton":
            stray = [x for x in fp if x.idx not in self.covered and x.idx not in dead_only]
        else:
            stray = [x for x in fp if x.idx in live_slice and x.idx not in self.covered]
        if stray:
            raise Fail(f"FP instruction outside the proven projection: {stray[0].text} /*{stray[0].addr:04x}*/")
        return rec

    def ystore_slice(self):
        s = set()
        for x in self.ystores():
            s |= self.k.slice_ins(x.vals["data"], value_only=True)
        return s


# ---- r8: one proof per kR7Proj kernel, shared by projgate (R3), fpflow (R0/R1/MR6: the projection exemption) and
#      ordgate (R4: O3 gates exactly the proven operand loads); cached by the kernel's SASS text, run in parallel.
def _code_sha():
    h = hashlib.sha256()
    for f in ("projgate.py", "sassir.py", "fpflow.py"):  # a proof is only reused by the gate code that made it
        h.update(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), "rb").read())
    return h.hexdigest()


def _proof_key(name, body):
    return hashlib.sha256(repr((name, body, _code_sha())).encode()).hexdigest()


def prove_r7(item):
    name, body = item
    g = ProjGate(name, body, "r7")
    try:
        rec = g.run_proj()
        return {"ok": True, "canon": canon(rec), "summary": summary(rec), "covered": sorted(g.covered),
                "sts": sorted(g.proven_sts), "loads": sorted(g.used_loads), "ystores": sorted(g.ystore_exec)}
    except (Fail, F.Fail, S.Ambiguous, S.Unknown) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


CACHE = os.environ.get("PROJGATE_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                                      ".projcache"))


def prove_all(items, jobs=None):
    """{name: proof} for [(name, body)]; proofs are cached under CACHE by sha256(name, SASS text)."""
    out, todo = {}, []
    for name, body in items:
        f = os.path.join(CACHE, _proof_key(name, body) + ".json")
        if os.path.exists(f):
            out[name] = json.load(open(f))
        else:
            todo.append((name, body))
    if todo:
        jobs = jobs or int(os.environ.get("PROJGATE_JOBS", "0")) or min(len(todo), os.cpu_count() or 1)
        if jobs > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(jobs) as pool:
                res = pool.map(prove_r7, todo)
        else:
            res = [prove_r7(t) for t in todo]
        os.makedirs(CACHE, exist_ok=True)
        for (name, body), r in zip(todo, res):
            out[name] = r
            json.dump(r, open(os.path.join(CACHE, _proof_key(name, body) + ".json"), "w"))
    return out


def canon(rec):
    return json.dumps({str(r): rec[r] for r in sorted(rec)}, sort_keys=True)


def summary(rec):
    r0 = rec[0]
    ch = r0["chains"][5]
    return (f"rows {len(rec)}; thread 5 chain {len(ch)} terms, first {ch[:2]} last {ch[-1]}; masked terms row 0: "
            f"{sum(1 for c in r0['chains'] for p in c if p[0] == ('Z',))}; level-2 lane->warp {r0['level2_lane_to_warp']}")


# --selftest: must-fail edits of the served Triton SASS itself, run through the same proof (CPU, no build needed).
# Each must either FAIL the proof or produce a record that differs from the unedited kernel's.
SELFTEST = [
    ("T1 chain starts with element 0 (FMUL x0*d0, then FFMA x1*d1)",
     [("FMUL R52, R0, R5 ;", "FMUL R52, R53, R4 ;"), ("FFMA R52, R53, R4, R52 ;", "FFMA R52, R0, R5, R52 ;")]),
    ("T2 warp butterfly xor 1 first and 16 last",
     [("SHFL.BFLY PT, R3, R36, 0x10, 0x1f", "SHFL.BFLY PT, R3, R36, 0x1, 0x1f"),
      ("SHFL.BFLY PT, R75, R72, 0x1, 0x1f", "SHFL.BFLY PT, R75, R72, 0x10, 0x1f")]),
    ("T3 second-level butterfly xor 1 then 4", [("SHFL.BFLY PT, R75, R2, 0x4, 0x1f", "SHFL.BFLY PT, R75, R2, 0x1, 0x1f"),
                                             ("SHFL.BFLY PT, R77, R72, 0x1, 0x1f", "SHFL.BFLY PT, R77, R72, 0x4, 0x1f")]),
    ("T4 one y FFMA flushes denormals (.FTZ)", [("FFMA R5, -R64.reuse, R5, R0 ;", "FFMA.FTZ R5, -R64.reuse, R5, R0 ;")]),
    ("T5 one y FFMA without the negation (x + s*d)", [("FFMA R5, -R64.reuse, R5, R0 ;", "FFMA R5, R64.reuse, R5, R0 ;")]),
    ("T6 d elements 4..7 read 16 bytes too far", [("LDG.E.128 R8, desc[UR10][R32.64+0x10]", "LDG.E.128 R8, desc[UR10][R32.64+0x20]")]),
]


def selftest(path):
    text = F.load_text(path)
    fs = S.split_functions(text)
    n = [k for k in fs if "_proj_rows" in k][0]
    base = canon(ProjGate(n, fs[n], "triton").run_proj())
    ok = True
    for name, edits in SELFTEST:
        t = text
        for a, b in edits:
            if t.count(a) != 1:
                print(f"selftest {name}: anchor {a!r} found {t.count(a)} times: BROKEN")
                ok = False
                break
            t = t.replace(a, b)
        else:
            f2 = S.split_functions(t)
            try:
                r = canon(ProjGate(n, f2[n], "triton").run_proj())
                caught, why = r != base, "record differs" if r != base else "record EQUAL"
            except (Fail, F.Fail, S.Ambiguous, S.Unknown) as e:
                caught, why = True, f"proof fails: {e}"
            print(f"selftest {name}: {'CAUGHT' if caught else 'NOT CAUGHT'} ({why})")
            ok &= caught
    print(f"PROJGATE_SELFTEST {'PASS' if ok else 'FAIL'} (base record {hashlib.sha256(base.encode()).hexdigest()[:16]})")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        return selftest(sys.argv[sys.argv.index("--selftest") + 1])
    if "--records" in sys.argv:  # the served Triton cubins (one per cache entry) must all have the same record
        hs = []
        for path in sys.argv[sys.argv.index("--records") + 1:]:
            fs = S.split_functions(F.load_text(path))
            n = [k for k in fs if "_proj_rows" in k][0]
            try:
                hs.append(hashlib.sha256(canon(ProjGate(n, fs[n], "triton").run_proj()).encode()).hexdigest()[:16])
            except (Fail, F.Fail, S.Ambiguous, S.Unknown) as e:
                hs.append(f"FAIL {e}")
            print(f"{path}: record {hs[-1]}")
        same = len(hs) > 1 and len(set(hs)) == 1 and not hs[0].startswith("FAIL")
        print(f"RECORDS {'EQUAL' if same else 'DIFFER'} ({len(hs)} served cubins)")
        return 0 if same else 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Triton _proj_rows SASS (cuobjdump -sass of the served cubin)")
    ap.add_argument("--cand", required=True, help="candidate rdma_ar SASS (full build or device-TU cubins)")
    ap.add_argument("--json")
    ap.add_argument("--mutant", action="store_true")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--only", default=None, help="regex: gate only matching candidate kernels")
    a = ap.parse_args()
    rep = {"ref": a.ref, "cand": a.cand, "kernels": {}}
    ref = S.split_functions(F.load_text(a.ref))
    rn = [n for n in ref if "_proj_rows" in n]
    if len(rn) != 1:
        print(f"ref: expected one _proj_rows kernel, found {len(rn)}")
        return 2
    rg = ProjGate(rn[0], ref[rn[0]], "triton")
    try:
        rrec = rg.run_proj()
        print(f"ref  _proj_rows PROVEN: {summary(rrec)}")
    except (Fail, F.Fail, S.Ambiguous) as e:
        print(f"ref  _proj_rows FAIL: {e}")
        return 2
    rep["ref_record"] = json.loads(canon(rrec))
    cand = S.split_functions(F.load_text(a.cand))
    res = []
    sel = []
    for n, body in sorted(cand.items()):
        m = re.search(r"ar_kernel_r3ILi(\d)ELb(\d)ELi8ELi(\d+)EE", n)
        if not m or not (int(m.group(3)) & 512):
            continue
        if a.only and not re.search(a.only, F.kernel_key(n)):
            continue
        sel.append((n, body))
    proofs = prove_all(sel)
    rref = canon(rrec)
    for n, body in sel:
        key = F.kernel_key(n)
        pr = proofs[n]
        if pr["ok"]:
            same = pr["canon"] == rref
            err = None if same else "canonical projection record differs from the served Triton _proj_rows"
            if not same:
                crec = {int(k): v for k, v in json.loads(pr["canon"]).items()}
                for r in sorted(rrec):
                    for fld in ("level2_lane_to_warp", "level2_masks", "level1_masks"):
                        if rrec[r][fld] != crec[r][fld]:
                            err += f" (row {r} {fld}: {crec[r][fld]} vs {rrec[r][fld]})"
                            break
                    else:
                        for t in range(256):
                            if [list(map(list, p)) for p in rrec[r]["chains"][t]] != crec[r]["chains"][t]:
                                err += f" (row {r} thread {t} chain: {crec[r]['chains'][t][:3]} vs {rrec[r]['chains'][t][:3]})"
                                break
                    if err.endswith(")"):
                        break
        else:
            same, err = False, pr["error"]
        res.append((key, same, err))
        rep["kernels"][key] = {"pass": same, "error": err}
        print(f"cand {key:44s} {'PASS' if same else 'FAIL'}" + ("" if same else f"  {err}"))
    if not res:
        print("no kR7Proj kernels in the candidate")
        return 1
    if a.mutant:
        caught = [k for k, ok, e in res if not ok and (a.expect is None or (e and re.search(a.expect, e)))]
        v = len(caught) == len(res)
        print(f"MUTANT: projection kernels caught {len(caught)}/{len(res)}" + (f" (/{a.expect}/)" if a.expect else "")
              + f" -> {'KILLED' if v else 'SURVIVED'}")
        rep["verdict"] = "KILLED" if v else "SURVIVED"
        rc = 0 if v else 1
    else:
        v = all(ok for _, ok, _ in res)
        print(f"PROJGATE {'PASS' if v else 'FAIL'}: {sum(ok for _, ok, _ in res)}/{len(res)} kR7Proj kernels equal the served "
              f"_proj_rows dataflow")
        rep["verdict"] = "PASS" if v else "FAIL"
        rc = 0 if v else 1
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, sort_keys=True, default=str)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
