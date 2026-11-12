"""det_gate.py (r8-fixb step): the LONG-PROMPT DETERMINISM gate for COLEMB and PACK (device_test.py --part det).

Why: r6-w1's s1 arm (C4 + COLEMB + PACK=all) differed in 1 of 4 reps on the 128K and code-4000 cases. COLEMB and PACK
had never been served alone, so the race could be theirs or C4's. From the code (README section 2): above their caps
(T > 128, i.e. every chunked-prefill forward, T up to max_num_batched_tokens = 16384) both levers are the stock call with
no kit kernel, collective, stream, event or allocation; at decode sizes COLEMB is single-stream and PACK waits on exactly
the stock lookup events. This gate checks both claims on the GPU with an adversarial scheduler, and must-fail mutants
show that the gate would see each kind of race.

Per T in DECODE_T (kit engages) and PREFILL_T (kit must pass through), REPS reps with different timing:
  PACK (look and all): two Engram layers built from the LIVE vllm Engram class (live prepare_embeddings / _start_prefetch
    / _ready_rows / _finish_prefetch / embed; one shared prefetch stream as in serving; the lookup is a table gather
    delayed by a spin kernel on the prefetch stream), staging rows poisoned (NaN payload bits) on main before every
    forward, main-stream stand-ins for layer 0 and layers 2..13 with their own delays. Forward = prepare(a), prepare(b),
    layer-0 work, kv_a = a.wkv(a.embed(h).flatten(-2)), layers 2..13 work, kv_b = b.wkv(b.embed(h).flatten(-2)) (the
    live Engram.forward line). Gates: kit embeds and kvs == stock bitwise in every rep; decode T: the kit facade made
    exactly 1 (look) / 2 (all) gathers and the stock gathers ran 0 / 0 (rows) and 2 / 0 (wkv) times; prefill T: the kit
    facade made 0 gathers, the pack counters did not move, the stock made 2 rows + 2 wkv gathers (pass-through).
    PACK=all identity (decode T): after a packed forward, layer b's wkv called on a FOREIGN operand must return the stock
    result for that operand.
  COLEMB: the wrapped VocabParallelEmbedding.forward (the kit's own class hook, installed on a stand-in class whose
    forward is the stock emulation), ids written by a delayed producer on main into a poisoned (-1) buffer. Gates: kit ==
    stock bitwise in every rep; decode T: the kit lookup + one column gather ran and the stock forward did not; prefill T:
    the stock forward ran once on the very same tensor and the kit did nothing.
Must-fail mutants (each caught at EVERY T it applies to):
  pack-nowait   (decode T)  the packed gather reads layer b's staging rows without waiting on b's lookup event
  pack-cap      (prefill T) PACK_MAX_T raised to 16384 (the kit engages at prefill sizes)
  pack-stale    (prefill T) the pairing state is never cleared (a's stock path and b's take keep it) and b ignores
                            (seq, T): b serves the previous decode forward's rows
  pack-r6pair   (decode T)  the r6 module (T-only pairing): layer b's wkv returns the stored kv for a foreign operand
  colemb-side   (decode T)  the lookup on a side stream that does not wait for the ids producer
  colemb-cap    (prefill T) ColTable.max_t raised to 16384
The CPU self-test (device_test.py --cpu-selftest) runs the whole PACK part (gates + 4 mutants) on a simulated runtime in
which the prefetch stream runs as LATE as the events allow (a lookup executes only when someone waits on its event), so
every unsynchronised read sees the poison. COLEMB engages only on CUDA ids, so on CPU it checks the pass-through only;
its engaged gates and both colemb mutants run on the GPU.
"""
from __future__ import annotations

import importlib.util
import os

DECODE_T = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 30, 32, 40, 48, 64, 96, 128]
PREFILL_T = [129, 256, 1024, 2048, 4000, 8192, 13312, 16384]
DECODE_T_QUICK = [1, 6, 30, 128]
PREFILL_T_QUICK = [129, 4000, 16384]
TMAX = 16384  # launcher --max-num-batched-tokens
H, HD, TP, ME = 3, 256, 8, 3  # Engram rows [T, 3, 256] per rank; this process plays rank 3
WKV_OUT = 400  # local wkv columns (serving 3200; the GEMM itself is gated bitwise by part pack, not here)
# (prefetch delay, layer-0 delay, layers 2..13 delay) in spin cycles (~2 GHz): every ordering of lookup vs consumer
REPS = [(0, 0, 0), (40_000_000, 0, 0), (0, 20_000_000, 0), (40_000_000, 0, 40_000_000), (8_000_000, 2_000_000, 0)]
POISON = 0x7FC1  # a NaN payload no lookup produces
HERE = os.path.dirname(os.path.abspath(__file__))
R6_MODULE = os.path.join(HERE, "prev", "dsv41_step.py.r6-ebe2f7bb")


def load_module(path, name, replace=()):
    src = open(path).read()
    for old, new in replace:
        assert src.count(old) == 1, (name, old[:60], src.count(old))
        src = src.replace(old, new)
    spec = importlib.util.spec_from_loader(name, loader=None)
    m = importlib.util.module_from_spec(spec)
    m.__file__ = path
    exec(compile(src, path, "exec"), m.__dict__)
    return m


STALE_MUT = (("            PK.rows = PK.kv = PK.a_out = PK.b_out = None\n", "            PK.kv = PK.a_out = None\n"),
             ("            rec, PK.rows = PK.rows, None\n", "            rec = PK.rows\n"),
             ("            if rec is None or rec[0] != PK.seq or rec[1] != T:\n", "            if rec is None:\n"))


# ================================================================================================ runtimes
class GpuRT:
    """Real CUDA streams/events and the live vllm Engram class."""
    name = "gpu"

    def __init__(self, torch):
        from vllm.models.deepseek_v41.common import engram as E

        self.torch, self.E, self.dev = torch, E, "cuda"
        self.base = E.Engram
        self._saved = E.tensor_model_parallel_all_gather

    def set_engram_ag(self, fn):
        self.E.tensor_model_parallel_all_gather = fn

    def restore(self):
        self.E.tensor_model_parallel_all_gather = self._saved

    def spin(self, cycles):
        if cycles:
            self.torch.cuda._sleep(int(cycles))

    def stream(self):
        return self.torch.cuda.Stream()

    def event(self):
        return self.torch.cuda.Event()

    def sync(self):
        self.torch.cuda.synchronize()

    def init_engram(self, obj, stream):
        obj._prefetch_stream, obj._prefetch_done = stream, self.torch.cuda.Event()


class _SimStream:
    def __init__(self):
        self.q = []  # pending closures (run lazily)

    def flush_to(self, n):
        while self.q and n > 0:
            self.q.pop(0)()
            n -= 1


class _SimEvent:
    def __init__(self, stream=None, pos=0):
        self.stream, self.pos = stream, pos


class SimEngramBase:
    """CPU mirror of the live Engram prefetch semantics with a LATEST-POSSIBLE scheduler: the prefetch stream's lookup
    runs only when a consumer waits on its event (_ready_rows); an unsynchronised read sees the poison."""

    def __init__(self, *a, **k):
        pass

    def prepare_embeddings(self, hash_ids):
        rows = self.staged_rows[: hash_ids.shape[0]]
        st = self._prefetch_stream
        st.q.append(lambda h=hash_ids.clone(), r=rows: self.embed_tokens.lookup(h, r))
        self._prefetch_done = _SimEvent(st, st.done + len(st.q))

    def _ready_rows(self, num_tokens):
        ev = self._prefetch_done
        executed = getattr(ev.stream, "done", 0)
        ev.stream.flush_to(ev.pos - executed)
        ev.stream.done = executed + max(0, ev.pos - executed)
        return self.staged_rows[:num_tokens]

    def embed(self, hash_ids):
        rows = self._ready_rows(hash_ids.shape[0])
        rows = SIM_AG[0](rows, 1)
        return rows[:, : self.embed_tokens.n_hash_cols]


SIM_AG = [None]


class SimRT:
    name = "cpu-sim"

    def __init__(self, torch):
        self.torch, self.dev, self.base = torch, "cpu", SimEngramBase
        self.streams = []

    def set_engram_ag(self, fn):
        SIM_AG[0] = fn

    def restore(self):
        SIM_AG[0] = None

    def spin(self, cycles):
        pass

    def stream(self):
        s = _SimStream()
        s.done = 0
        self.streams.append(s)
        return s

    def sync(self):  # device synchronize: every queued lookup completes
        for s in self.streams:
            k = len(s.q)
            s.flush_to(k)
            s.done += k

    def init_engram(self, obj, stream):
        obj._prefetch_stream, obj._prefetch_done = stream, None


# ================================================================================================ PACK world
def _bits(t):
    import torch

    return t.contiguous().view(torch.int16)


def _beq(a, b):
    return tuple(a.shape) == tuple(b.shape) and bool((_bits(a) == _bits(b)).all())


class PackWorld:
    def __init__(self, rt, m, mode):
        torch = rt.torch
        self.rt, self.m, self.mode, self.torch = rt, m, mode, torch
        dev = rt.dev
        g = torch.Generator(device="cpu").manual_seed(8)
        R = 4096
        tabs = (torch.randn(2, R, H, HD, generator=g) * 0.5).to(torch.bfloat16)
        tb = tabs.view(torch.int16)
        tb[:, 0, 0, :8] = -32768  # -0
        tb[:, 0, 1, :8] = 1  # smallest denormal
        self.tabs = [tabs[i].contiguous().to(dev) for i in range(2)]  # the poison (a NaN payload) never occurs
        self.W = [(torch.randn(WKV_OUT, TP * H * HD, generator=g) * 0.02).to(torch.bfloat16).to(dev) for _ in range(2)]
        self.cnt = {"kit": 0, "rows": 0, "wkv": 0}
        xor = [((r * 0x0101) ^ 0x5A5A) & 0x7FFF for r in range(TP)]

        def ag_core(x, dim):  # 8-rank SPMD emulation: rank r's block = this rank's operand with bits XOR c_r (c_ME = 0)
            xi = x.contiguous().view(torch.int16)
            blocks = [(xi if r == ME else (xi ^ (xor[r] ^ xor[ME]))).view(torch.bfloat16) for r in range(TP)]
            return torch.cat(blocks, dim=dim)

        def ag_rows(x, dim):
            self.cnt["rows"] += 1
            return ag_core(x, dim)

        def ag_kit(x, dim):
            self.cnt["kit"] += 1
            return ag_core(x, dim)

        self.ag_core, self.ag_rows, self.ag_kit = ag_core, ag_rows, ag_kit
        world = self

        class _QM:
            def apply(self, layer, x, bias):
                return torch.matmul(x, layer.W.t())

        class ColumnParallelLinear:  # qualname ends in ColumnParallelLinear.forward (the kit's refusal test)
            def __init__(self, W):
                self.W, self.gather_output, self.skip_bias_add, self.bias, self.quant_method = W, True, False, None, _QM()

            def forward(self, x):
                world.cnt["wkv"] += 1
                return world.ag_core(self.quant_method.apply(self, x, None), -1)

            def __call__(self, x):
                return self.forward(x)

        class FakeTab:
            tp_size, dp_size, part_n_hash_cols, n_hash_cols, dim, cpu_offload = TP, 1, H, TP * H, HD, True

            def __init__(self, tab):
                self.tab, self.delay = tab, 0

            def lookup(self, hash_ids, rows, background=False):
                rt.spin(self.delay)
                rows.copy_(self.tab.index_select(0, hash_ids.remainder(self.tab.shape[0])))

        def make_cls(tag):
            def __init__(self, idx, stream):
                if rt.name == "gpu":
                    torch.nn.Module.__init__(self)
                self.layer_hash_index, self._batch, self.use_sequence_parallel = idx, None, False
                self.embed_tokens = FakeTab(world.tabs[idx])
                self.staged_rows = torch.empty(TMAX, H, HD, dtype=torch.bfloat16, device=dev)
                rt.init_engram(self, stream)
                self.wkv = ColumnParallelLinear(world.W[idx])

            return type(f"DetEngram_{tag}", (rt.base,), {"__init__": __init__})

        self.Stock = make_cls("stock")
        self.Kit = make_cls(f"kit_{mode}_{id(m)}")
        m.PK.__init__()
        m.PACK = mode
        m._all_gather = ag_kit
        m._breakable_capture = lambda: False
        m._install_engram(self.Kit)
        s_stock, s_kit = rt.stream(), rt.stream()
        self.stock = (self.Stock(0, s_stock), self.Stock(1, s_stock))
        self.kit = (self.Kit(0, s_kit), self.Kit(1, s_kit))
        rt.set_engram_ag(ag_rows)

    def forward(self, pair, h, delays):
        torch, rt = self.torch, self.rt
        a, b = pair
        pf, d0, d1 = delays
        for e in pair:
            e.staged_rows.view(torch.int16).fill_(POISON)
            e.embed_tokens.delay = pf
        a.prepare_embeddings(h[:, 0])
        b.prepare_embeddings(h[:, 1])
        rt.spin(d0)  # layer 0
        ea = a.embed(h)
        ka = a.wkv(ea.flatten(-2))
        rt.spin(d1)  # layers 2..13
        eb = b.embed(h)
        kb = b.wkv(eb.flatten(-2))
        out = [t.clone() for t in (ea, ka, eb, kb)]
        rt.sync()
        return out

    def hashes(self, T, rep):
        g = self.torch.Generator(device="cpu").manual_seed(1000 * T + rep)
        return self.torch.randint(0, 1 << 30, (T, 2), generator=g, dtype=self.torch.int64).to(self.rt.dev)


def pack_run(world, Ts, reps, decode, report=None):
    """Returns (bad_T, counter_bad_T). bad_T: T where kit != stock in some rep; counter_bad_T: T where the gather
    counts / pack counters are not the expected engaged (decode) or pass-through (prefill) pattern."""
    m, mode = world.m, world.mode
    bad, cbad = [], []
    for T in Ts:
        diff = False
        cnt_ok = True
        for rep, dl in enumerate(reps):
            h = world.hashes(T, rep)
            ref = world.forward(world.stock, h, dl)
            c0 = dict(world.cnt)
            k0 = (m.COUNTS["pack_rows"], m.COUNTS["pack_wkv"])
            got = world.forward(world.kit, h, dl)
            dk = {x: world.cnt[x] - c0[x] for x in c0}
            dp = (m.COUNTS["pack_rows"] - k0[0], m.COUNTS["pack_wkv"] - k0[1])
            if not all(_beq(x, y) for x, y in zip(ref, got)):
                diff = True
            if decode:
                want = {"kit": 1, "rows": 0, "wkv": 2} if mode == "look" else {"kit": 2, "rows": 0, "wkv": 0}
                want_p = (1, 0) if mode == "look" else (1, 1)
            else:
                want, want_p = {"kit": 0, "rows": 2, "wkv": 2}, (0, 0)
            if dk != want or dp != want_p:
                cnt_ok = False
                if report is not None and len(report) < 4:
                    report.append((T, rep, dk, dp))
        if diff:
            bad.append(T)
        if not cnt_ok:
            cbad.append(T)
    return bad, cbad


def pack_foreign(world, Ts):
    """PACK=all: after a packed forward's embeds, layer b's wkv on a FOREIGN operand must give the stock result."""
    torch = world.torch
    bad = []
    for T in Ts:
        h = world.hashes(T, 99)
        a, b = world.kit
        sa, sb = world.stock
        for e in world.kit + world.stock:
            e.embed_tokens.delay = 0
        a.prepare_embeddings(h[:, 0])
        b.prepare_embeddings(h[:, 1])
        ea = a.embed(h)
        a.wkv(ea.flatten(-2))
        b.embed(h)
        g = torch.Generator(device="cpu").manual_seed(T)
        xf = (torch.randn(T, TP * H * HD, generator=g)).to(torch.bfloat16).to(world.rt.dev)
        got = b.wkv(xf)
        ref = sb.wkv(xf)
        world.rt.sync()
        if not _beq(got, ref):
            bad.append(T)
    return bad


def part_pack_det(m, rt, gate, mutant, quick=False, r6_path=R6_MODULE):
    dT = DECODE_T_QUICK if quick else DECODE_T
    pT = PREFILL_T_QUICK if quick else PREFILL_T
    reps = REPS[:3] if quick else REPS
    import types

    for mode in ("look", "all"):
        w = PackWorld(rt, m, mode)
        try:
            rep = []
            bad, cbad = pack_run(w, dT, reps, True, rep)
            gate(f"DET pack={mode} decode T {dT[0]}..{dT[-1]} ({len(dT)} T x {len(reps)} timing reps): kit == stock "
                 f"bitwise in every rep", not bad, f"differs at T={bad[:8]}")
            gate(f"DET pack={mode} decode: engaged (kit gathers {1 if mode == 'look' else 2}, stock rows 0, stock wkv "
                 f"{2 if mode == 'look' else 0}) at every T", not cbad, f"T={cbad[:8]} {rep[:2]}")
            rep = []
            bad, cbad = pack_run(w, pT, reps, False, rep)
            gate(f"DET pack={mode} prefill T {pT[0]}..{pT[-1]} ({len(pT)} T): kit == stock bitwise in every rep",
                 not bad, f"differs at T={bad[:8]}")
            gate(f"DET pack={mode} prefill: pure pass-through (0 kit gathers, pack counters unchanged, stock 2 rows + "
                 f"2 wkv gathers) at every T", not cbad, f"T={cbad[:8]} {rep[:2]}")
            if mode == "all":
                bad = pack_foreign(w, dT)
                gate("DET pack=all identity: layer b's wkv on a foreign operand returns the stock result at every "
                     "decode T", not bad, f"T={bad[:8]}")
            # ---- mutants
            real = m.packed_rows_gather

            def nowait(a, b, T, _m=m):
                ra = a._ready_rows(T)
                rb = b.staged_rows[:T]  # MUTANT: no wait on layer b's lookup event
                Hh, hd = ra.shape[1], ra.shape[2]
                g = _m._all_gather(rt.torch.cat([ra, rb], dim=1), 1)
                tp = g.shape[1] // (2 * Hh)
                s = g.view(T, tp, 2, Hh * hd).permute(2, 0, 1, 3).contiguous()
                return s[0].view(T, tp * Hh, hd), s[1].view(T, tp * Hh, hd)

            m.packed_rows_gather = nowait
            try:
                bad, _ = pack_run(w, dT, reps, True)
            finally:
                m.packed_rows_gather = real
            mutant(f"DET pack={mode} pack-nowait (b's rows read without its event) caught at every decode T",
                   bad == dT, f"missed at T={[t for t in dT if t not in bad][:8]}")
            cap = m.PACK_MAX_T
            m.PACK_MAX_T = TMAX
            try:
                bad, cbad = pack_run(w, pT, reps[:1], False)
            finally:
                m.PACK_MAX_T = cap
            caught = sorted(set(bad) | set(cbad))
            mutant(f"DET pack={mode} pack-cap (engages at prefill sizes) caught at every prefill T", caught == pT,
                   f"missed at T={[t for t in pT if t not in caught][:8]}")
        finally:
            rt.restore()
        # pack-stale: text mutant of THIS module (a keeps the stored rows on its stock path, b ignores seq/T)
        ms = load_module(m.__file__, f"dsv41_step_stale_{mode}", STALE_MUT)
        _copy_facades(ms, m)
        ws = PackWorld(rt, ms, mode)
        try:
            missed = []
            for T in pT:
                h6 = ws.hashes(6, 7)
                ws.forward(ws.kit, h6, (0, 0, 0))  # a decode forward stores rows (T=6)
                h = ws.hashes(T, 0)
                ref = ws.forward(ws.stock, h, (0, 0, 0))
                try:
                    got = ws.forward(ws.kit, h, (0, 0, 0))
                    same = all(_beq(x, y) for x, y in zip(ref, got))
                except Exception:  # noqa: BLE001 (a shape error is a catch)
                    same = False
                if same:
                    missed.append(T)
            mutant(f"DET pack={mode} pack-stale (b serves the previous forward's rows) caught at every prefill T",
                   not missed, f"missed at T={missed[:8]}")
        finally:
            rt.restore()
    # pack-r6pair: the r6 module (T-only pairing) on the foreign-operand identity check
    if os.path.exists(r6_path):
        m6 = load_module(r6_path, "dsv41_step_r6pair")
        _copy_facades(m6, m)
        w6 = PackWorld(rt, m6, "all")
        try:
            bad = pack_foreign(w6, dT)
        finally:
            rt.restore()
        mutant("DET pack=all pack-r6pair (r6's T-only pairing returns the stored kv for a foreign operand) caught at "
               "every decode T", bad == dT, f"missed at T={[t for t in dT if t not in bad][:8]}")
    else:
        gate("DET r6 module for the pack-r6pair mutant is staged (prev/dsv41_step.py.r6-ebe2f7bb)", False, r6_path)


def _copy_facades(dst, src):
    for k in ("_cu", "_tp", "_tp_agree_min", "_tp_all_equal"):
        if hasattr(src, k):
            setattr(dst, k, getattr(src, k))


# ================================================================================================ COLEMB
def part_colemb_det(m, rt, gate, mutant, quick=False, vpe_mod=None):
    torch = rt.torch
    dev = rt.dev
    dT = DECODE_T_QUICK if quick else DECODE_T
    pT = PREFILL_T_QUICK if quick else PREFILL_T
    reps = REPS[:3] if quick else REPS
    V, D = 32768, 5120
    n = D // TP
    g = torch.Generator(device="cpu").manual_seed(5)
    full = (torch.randn(V, D, generator=g) * 0.05).to(torch.bfloat16)
    full = full.to(dev)
    cts = [m.ColTable(full[:, r * n:(r + 1) * n].contiguous(), r, TP, n, V, D, 128) for r in range(TP)]
    if rt.name == "gpu":
        lookup = m._colemb_lookup_fn()
    else:
        def lookup(ids, ct):  # CPU stand-in with the kernel's semantics (table[id] if 0 <= id < V else +0)
            i = ids.reshape(-1)
            ok = (i >= 0) & (i < ct.V)
            out = ct.table.index_select(0, torch.where(ok, i, torch.zeros_like(i)))
            return torch.where(ok[:, None], out, torch.zeros_like(out))
    for ct in cts:
        ct.lookup = lookup
    cur = {"ids": None}
    cnt = {"kit": 0, "stock": 0, "stock_same_obj": 0}

    def ag_cols(local, dim):
        cnt["kit"] += 1
        assert dim == -1
        blocks = [local if r == ME else lookup(cur["ids"], cts[r]) for r in range(TP)]
        return torch.cat(blocks, dim=-1)

    class FakeVPE:
        def forward(self, input_):  # the stock path's result: the AR output row (canon map; table rows are canonical)
            cnt["stock"] += 1
            cnt["stock_same_obj"] += int(input_ is cur["obj"])
            i = input_.reshape(-1)
            ok = (i >= 0) & (i < V)
            out = full.index_select(0, torch.where(ok, i, torch.zeros_like(i)))
            return torch.where(ok[:, None], out, torch.zeros_like(out)).view(*input_.shape, D)

        def __call__(self, x):
            return self.forward(x)

    stock_fwd = FakeVPE.forward
    import types

    if vpe_mod is None:
        import sys

        vpe_mod = types.ModuleType("vllm.model_executor.layers.vocab_parallel_embedding")
        for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers"):
            sys.modules.setdefault(name, types.ModuleType(name))
        saved_mod = sys.modules.get(vpe_mod.__name__)
        sys.modules[vpe_mod.__name__] = vpe_mod
    else:
        saved_mod = None
    saved_cls = getattr(vpe_mod, "VocabParallelEmbedding", None)
    vpe_mod.VocabParallelEmbedding = FakeVPE
    real_ag = m._all_gather
    m._all_gather = ag_cols
    try:
        m._install_colemb()
        emb = FakeVPE()
        m._COL[id(emb)] = cts[ME]
        ids_buf = torch.empty(TMAX, dtype=torch.int64, device=dev)

        def produce(T, rep, delay):
            gg = torch.Generator(device="cpu").manual_seed(77 * T + rep)
            src = torch.randint(-3, V + 3, (T,), generator=gg, dtype=torch.int64)
            src[0] = 7 if T > 1 else src[0]
            src = src.to(dev)
            ids_buf.fill_(-1)
            rt.sync()  # the poison is in place before the delayed producer is queued
            rt.spin(delay)
            ids_buf[:T].copy_(src)
            v = ids_buf[:T]
            cur["ids"], cur["obj"] = v, v
            return v

        def run(Ts, decode, reps_):
            bad, cbad = [], []
            for T in Ts:
                diff, cok = False, True
                for rep, dl in enumerate(reps_):
                    v = produce(T, rep, dl[0])
                    ref = stock_fwd(emb, v).clone()
                    c0 = dict(cnt)
                    k0 = m.COUNTS["colemb"]
                    v = produce(T, rep, dl[0])
                    got = emb.forward(v).clone()
                    rt.sync()
                    d = {k: cnt[k] - c0[k] for k in cnt}
                    dk = m.COUNTS["colemb"] - k0
                    if not _beq(ref, got):
                        diff = True
                    want = ({"kit": 1, "stock": 0, "stock_same_obj": 0}, 1) if decode else \
                        ({"kit": 0, "stock": 1, "stock_same_obj": 1}, 0)
                    if (d, dk) != want:
                        cok = False
                if diff:
                    bad.append(T)
                if not cok:
                    cbad.append(T)
            return bad, cbad

        if rt.name != "gpu":
            # the kit engages only on CUDA ids (input_.is_cuda): on the CPU simulation every T must pass through
            bad, cbad = run(dT + pT, False, reps[:1])
            gate("DET colemb (cpu-sim) every T passes through on CPU ids (the kit engages only on CUDA tensors); the "
                 "engaged gates and both colemb mutants run on the GPU only", not bad and not cbad, f"{bad} {cbad}")
            return
        bad, cbad = run(dT, True, reps)
        gate(f"DET colemb decode T {dT[0]}..{dT[-1]} ({len(dT)} T x {len(reps)} reps, delayed ids producer): kit == "
             f"stock bitwise", not bad, f"T={bad[:8]}")
        gate("DET colemb decode: kit lookup + 1 column gather, stock forward not called, at every T", not cbad,
             f"T={cbad[:8]}")
        bad, cbad = run(pT, False, reps)
        gate(f"DET colemb prefill T {pT[0]}..{pT[-1]}: kit == stock bitwise", not bad, f"T={bad[:8]}")
        gate("DET colemb prefill: pure pass-through (stock forward once on the SAME tensor, 0 kit gathers, counter "
             "unchanged) at every T", not cbad, f"T={cbad[:8]}")
        if rt.name == "gpu":
            side = torch.cuda.Stream()

            def side_lookup(ids, ct):  # MUTANT: lookup on a side stream that does not wait for the ids producer
                with torch.cuda.stream(side):
                    out = lookup(ids, ct)
                torch.cuda.current_stream().wait_stream(side)
                return out

            cts[ME].lookup = side_lookup
            try:
                bad, _ = run(dT, True, [r for r in reps if r[0] >= 8_000_000])
            finally:
                cts[ME].lookup = lookup
            mutant("DET colemb-side (lookup not ordered after the ids producer) caught at every decode T", bad == dT,
                   f"missed at T={[t for t in dT if t not in bad][:8]}")
        cts[ME].max_t = TMAX
        try:
            bad, cbad = run(pT, False, reps[:1])
        finally:
            cts[ME].max_t = 128
        caught = sorted(set(bad) | set(cbad))
        mutant("DET colemb-cap (engages at prefill sizes) caught at every prefill T", caught == pT,
               f"missed at T={[t for t in pT if t not in caught][:8]}")
    finally:
        m._all_gather = real_ag
        m._COL.clear()
        if saved_cls is not None:
            vpe_mod.VocabParallelEmbedding = saved_cls
        if saved_mod is not None:
            import sys

            sys.modules[vpe_mod.__name__] = saved_mod
