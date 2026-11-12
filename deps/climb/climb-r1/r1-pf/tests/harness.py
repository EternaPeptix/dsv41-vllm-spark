"""CPU harness for r1-pf: loads the live base dsv41_l2_prefetch.py (or the kit's patched copy) under its real
package name with stub vllm modules, replaces torch.cuda with a recording fake (streams, waits, capture flag)
and the two native prefetch libraries with recorders, and drives the exact hook protocol the live code uses:

  module pre-hook records  -> S.events.append(("w", name, items))     (only in learn mode, like pre_hook)
  sonar o_proj wrapper     -> S.events.append(("woa", name, items))   (only in learn mode)
  wq_b post-hook           -> base._wq_b_post_hook(...)
  RdmaAllReduce            -> base.before_collective(kind); base.after_collective(kind) when base.G2
  vLLM capture             -> eager warm-up forward (learn) then the same forward with capturing=True

Every launch is logged as (op, stream, [(ptr, bytes)...], extra) so two runs can be compared exactly.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys
import types
from pathlib import Path

import torch

KIT = Path(__file__).resolve().parent.parent
BASE_LIVE = KIT / "base" / "dsv41_l2_prefetch.py"
MODDIR = KIT / "files" / "vllm" / "distributed" / "device_communicators"
PATCHED = MODDIR / "dsv41_l2_prefetch.py"
R1_FILE = MODDIR / "dsv41_l2pf_r1.py"
STUBS = KIT / "tests" / "stubs"
PKG = "vllm.distributed.device_communicators"
BASE_NAME = PKG + ".dsv41_l2_prefetch"
R1_NAME = PKG + ".dsv41_l2pf_r1"
MI = 1 << 20

PROD_ENV = {  # the l2pf part of ~/dsv41-tp8.sh 08a249da (launcher defaults since climb-w1, 2026-11-08 13:40:58Z)
    "DSV41_L2_PREFETCH": "1",
    "DSV41_L2_PREFETCH_PRUNE_SHARED": "1",
    "DSV41_L2_PREFETCH_WOA": "1",
    "DSV41_L2_PREFETCH_MHC": "0",
    "DSV41_L2_PREFETCH_PACE_GBPS": "125",
    "DSV41_L2_PREFETCH_PACE_KINDS": "ar,attn,ag",
    "DSV41_L2_PREFETCH_PACE_CHUNK": "65536",
    "DSV41_L2_PREFETCH_WOB_MB": "6",
}
PROD_FDA_ENV = dict(PROD_ENV, DSV41_L2_PREFETCH_PACE_KINDS="ar")  # launcher fda6d05a (before climb-w1)
del PROD_FDA_ENV["DSV41_L2_PREFETCH_WOB_MB"]

LOG: list = []


class FakeStream:
    _ids = itertools.count(1000)

    def __init__(self, name):
        self.name = name
        self.cuda_stream = next(FakeStream._ids)
        REG[self.cuda_stream] = self

    def wait_stream(self, other):
        LOG.append(("wait", self.name, other.name))


REG: dict[int, FakeStream] = {}


class FakeCuda:
    def __init__(self):
        self.main = FakeStream("main")
        self.capturing = False
        self.n = 0

    def current_stream(self, device=None):
        return self.main

    def Stream(self, device=None):  # noqa: N802
        self.n += 1
        return FakeStream(f"new{self.n}")

    def current_device(self):
        return 0

    def is_current_stream_capturing(self):
        return self.capturing

    def is_available(self):
        return False


class TorchProxy(types.ModuleType):
    def __init__(self, cuda):
        super().__init__("torch_proxy")
        self.cuda = cuda

    def __getattr__(self, k):
        return getattr(torch, k)

    @staticmethod
    def tensor(data, dtype=None, device=None):
        return torch.tensor(data, dtype=dtype)  # every table lives on the CPU here


def _table(B, ptr, nseg):
    for t in reversed(B.S.keepalive):
        if t.data_ptr() == ptr:
            v = t.tolist()
            assert len(v) == 2 * nseg, (len(v), nseg)
            return tuple((v[2 * i], v[2 * i + 1]) for i in range(nseg))
    raise AssertionError(f"segment table {ptr:#x} not kept alive")


class FakeLib:
    def __init__(self, B):
        self.B = B

    def dsv41_l2pf(self, ptr, nseg, chunk, stream):
        LOG.append(("burst", REG[stream].name, _table(self.B, ptr, nseg), chunk))
        return 0

    def dsv41_l2pf_paced(self, ptr, nseg, chunk, period_ps, delay_ns, max_ns, stats, stream):
        assert stats is None
        LOG.append(("paced", REG[stream].name, _table(self.B, ptr, nseg), chunk, period_ps, delay_ns, max_ns))
        return 0

    def dsv41_spin(self, *a):
        return 0


def load(which: str = "patched", env: dict | None = None):
    """Fresh import of the base (live or patched) with exactly `env` as the DSV41_L2_PREFETCH* environment."""
    for k in list(os.environ):
        if k.startswith("DSV41_L2_PREFETCH"):
            del os.environ[k]
    os.environ.update(env or {})
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            del sys.modules[name]
    if str(STUBS) not in sys.path:
        sys.path.insert(0, str(STUBS))
    import vllm.distributed.device_communicators as pkg  # noqa: F401  (stub package)

    pkg.__path__.append(str(MODDIR))  # dsv41_l2pf_r1 is found next to the base, as installed
    path = {"base": BASE_LIVE, "patched": PATCHED}[which]
    spec = importlib.util.spec_from_file_location(BASE_NAME, path)
    B = importlib.util.module_from_spec(spec)
    sys.modules[BASE_NAME] = B
    spec.loader.exec_module(B)
    return B


class Sim:
    """A small DSv4.1-shaped target (SWA / indexer / kv-source+indexer / Engram layers) and a 3-layer DSpark drafter."""

    TARGET_LAYERS = ("swa", "swa_engram", "kvsrc_idx", "idx", "swa", "kvsrc_idx_engram")

    def __init__(self, B):
        LOG.clear()
        REG.clear()
        self.B = B
        self.R1 = sys.modules.get(R1_NAME)
        self.cuda = FakeCuda()
        proxy = TorchProxy(self.cuda)
        B.torch = proxy
        if self.R1 is not None:
            self.R1.torch = proxy
            self.R1.ALLOW_CPU = True
        B.S.lib = FakeLib(B)
        B.S.lib_g2 = FakeLib(B)
        B.S.pace_ok = bool(B.G2 and B.PACE_GBPS > 0)
        self._ptr = 0x7F0000000000
        self._ptrs: dict = {}
        self.keep = []
        import vllm.models.deepseek_v41.attention as attn_mod
        import vllm.v1.worker.gpu.spec_decode.dspark.speculator as spec_mod
        import vllm.models.deepseek_v41.nvidia.dsv41_r4_small as r4
        import vllm.models.deepseek_v41.nvidia.dsv41_boundary  # noqa: F401

        attn_mod.SIM = self
        spec_mod.SIM = self
        self.attn_mod, self.spec_mod, self.r4 = attn_mod, spec_mod, r4
        self.attn = []
        for L, kind in enumerate(self.TARGET_LAYERS):
            comp = None
            if "kvsrc" in kind:
                w = self.cpu_tensor(int(10.5 * MI))
                comp = types.SimpleNamespace(fused_wkv_wgate=types.SimpleNamespace(
                    weight=w, prefix=f"model.layers.{L}.attn.compressor.fused_wkv_wgate"))
            self.attn.append(attn_mod.DeepseekV4Attention(L, comp))

        sim = self

        class DeepseekV4Model(torch.nn.Module):
            def forward(self, input_ids, positions):
                sim.target_forward(int(positions.shape[0]))
                return "hidden"

        self.target_cls = DeepseekV4Model
        self.target = DeepseekV4Model()
        self.draft_model = self._draft_model()

    # ---------------------------------------------------------------- helpers
    def cpu_tensor(self, nbytes):
        t = torch.empty(nbytes, dtype=torch.uint8)
        self.keep.append(t)
        return t

    def alloc(self, key, nbytes):
        p = self._ptrs.get(key)
        if p is None:
            p = self._ptrs[key] = self._ptr
            self._ptr += (nbytes + 4095) & ~4095
        return p

    def w(self, name, sizes, kind="w"):
        S = self.B.S
        if S.mode == "learn" and S.events is not None:
            S.events.append((kind, name, [(self.alloc((name, i), sz), sz, None) for i, sz in enumerate(sizes)]))

    def coll(self, kind):
        self.B.before_collective(kind)
        LOG.append(("COLL", kind))
        if getattr(self.B, "G2", False):  # rdma_all_reduce wires after_collective only in G2 mode
            self.B.after_collective(kind)

    def wqb(self):
        self.B._wq_b_post_hook(None, None, None)

    # ---------------------------------------------------------------- target
    def install(self):
        mod = types.SimpleNamespace(DeepseekV4Model=self.target_cls)
        self.B.install_model_module(mod)
        S = self.B.S
        for d, nm in ((S.side, "side1"), (S.side2, "side2"), (S.side3, "side3")):
            if 0 in d:
                d[0].name = nm

    def input_projections(self, attn, hidden_states):
        L = attn.layer
        kind = self.TARGET_LAYERS[L]
        self.w(f"language_model.model.layers.{L}.attn.qkv_a_shard", [32 * 1024, int(1.1 * MI)])
        if "idx" in kind:
            self.w("idx.weights_proj", [int(0.08 * MI)])
        self.coll("ag")
        return None

    def target_forward(self, n):
        for L, kind in enumerate(self.TARGET_LAYERS):
            if "engram" in kind:
                self.w(f"layers.{L}.engram", [40 * 1024, 40 * 1024])
                self.coll("ag")
                self.w(f"layers.{L}.engram.wkv", [int(0.59 * MI), int(5.41 * MI)])
            self.attn[L]._run_parallel_input_projections(None)
            self.w(f"layers.{L}.attn.wq_b", [int(0.16 * MI), 5 * MI])
            self.wqb()
            if "kvsrc" in kind:  # the indexer that owns K projects it (wk) before its wq_b
                self.w(f"layers.{L}.attn.indexer.wk", [128 * 1024])
            if "idx" in kind:
                self.w(f"layers.{L}.attn.indexer.wq_b", [int(0.16 * MI), 5 * MI])
                self.wqb()
            self.w(f"language_model.model.layers.{L}.attn.wo_a", [int(0.12 * MI), 4 * MI], kind="woa")
            self.w(f"layers.{L}.attn.wo_b", [int(0.16 * MI), 5 * MI])
            self.coll("ar")
            self.w(f"layers.{L}.ffn.gate", [int(3.75 * MI)])
            self.w(f"layers.{L}.ffn.shared_experts.gate_up_proj", [int(0.09 * MI), int(2.9 * MI)])
            self.w(f"layers.{L}.ffn.shared_experts.down_proj", [int(0.05 * MI), int(1.47 * MI)])
            self.coll("ar")

    def capture_target(self, sizes=(4, 6)):
        for n in sizes:
            self.cuda.capturing = False
            LOG.append(("WARMUP", "target", n))
            self.target.forward(None, positions=torch.zeros(n))
            self.cuda.capturing = True
            LOG.append(("CAPTURE", "target", n))
            self.target.forward(None, positions=torch.zeros(n))
            LOG.append(("END", "target", n))
            self.cuda.capturing = False

    # ---------------------------------------------------------------- drafter
    def _draft_model(self):
        head = types.SimpleNamespace(s_swz=self.cpu_tensor(int(2.6 * MI)), w=self.cpu_tensor(int(82.7 * MI)))
        lm_head = types.SimpleNamespace(_dsv41_fp8_head=head, weight=self.cpu_tensor(1 * MI))
        split = types.SimpleNamespace(s_swz=self.cpu_tensor(int(0.31 * MI)), w=self.cpu_tensor(int(9.8 * MI)))
        main_proj = types.SimpleNamespace(_dsv41_r4_split=split, weight=self.cpu_tensor(MI))
        wkv = self.cpu_tensor(2048 * 4096).view(-1, 4096)  # [rows, K] contiguous; the fold keeps w.t()
        fold = types.SimpleNamespace(parts=[(wkv.t(), self.cpu_tensor(int(0.25 * MI)), 1536)])
        inner = types.SimpleNamespace(main_proj=main_proj, _dsv41_ctxkv=fold)
        m = torch.nn.Module()
        m.lm_head = lm_head
        m.model = inner
        self.eph_expect = [(split.s_swz.data_ptr(), split.s_swz.numel()), (split.w.data_ptr(), split.w.numel()),
                           (fold.parts[0][1].data_ptr(), fold.parts[0][1].numel()),
                           (wkv.data_ptr(), wkv.numel())]
        self.head_expect = [(head.s_swz.data_ptr(), head.s_swz.numel()), (head.w.data_ptr(), head.w.numel())]
        return m

    def specs(self):
        S = self.spec_mod.DSparkSpeculator
        return S(self.draft_model, None), S(self.draft_model, {"greedy": True})

    def drafter_backbone(self, spec, n):
        self.coll("ar")  # VocabParallelEmbedding all-reduce
        for L in (40, 41, 42):
            self.w("mhc_fn", [int(1.97 * MI)])  # recorded only with DSV41_L2_PREFETCH_MHC=1; must never be planned
            self.w(f"model.layers.{L}.attn.qkv_a_shard", [32 * 1024, int(1.1 * MI)])
            self.coll("ag")
            self.w(f"model.layers.{L}.attn.wq_b", [int(0.16 * MI), 5 * MI])
            self.wqb()
            self.w(f"model.layers.{L}.attn.wo_a", [int(0.12 * MI), 4 * MI], kind="woa")
            self.w(f"model.layers.{L}.attn.wo_b", [int(0.16 * MI), 5 * MI])
            self.coll("ar")
            self.w(f"model.layers.{L}.ffn.gate", [int(1.3 * MI)])
            self.w(f"model.layers.{L}.ffn.shared_experts.gate_up_proj", [int(0.09 * MI), int(2.9 * MI)])
            self.w(f"model.layers.{L}.ffn.shared_experts.down_proj", [int(0.05 * MI), int(1.47 * MI)])
            self.coll("ar")

    def drafter_markov(self, spec, num_reqs):
        # the Markov embedding (nn.Embedding, 63 MiB, a few rows read) is module-called before every position
        self.w("model.model.markov_head.markov_w1", [63 * MI])
        if spec._sharded_markov is not None:
            for _ in range(5):  # greedy-sharded: (max, id) all-gather per draft position
                self.coll("ag")
        else:
            self.coll("ag")  # stock: base-logit gather, then per-position bias all-reduce (one more collective
            for _ in range(5):  # than the greedy set, so a plan shared across the two sets would mismatch)
                self.coll("ar")

    def capture_drafter(self, shapes=((1, 5), (2, 10))):
        stock, greedy = self.specs()
        for tag, spec in (("stock", stock), ("greedy", greedy)):  # DSparkSpeculator.capture: stock set, then greedy
            for nreq, n in shapes:
                self.cuda.capturing = False
                LOG.append(("WARMUP", tag, n))
                spec._generate_draft(nreq, n, None, None, None)
                self.cuda.capturing = True
                LOG.append(("CAPTURE", tag, n))
                spec._generate_draft(nreq, n, None, None, None)
                LOG.append(("END", tag, n))
                self.cuda.capturing = False
        return stock, greedy

    def eager_step_logits(self):
        LOG.append(("EPHASE",))
        return self.r4.target_logits(None, "h")

    # ---------------------------------------------------------------- analysis
    def section(self, tag, n):
        """LOG entries of one captured forward."""
        i = LOG.index(("CAPTURE", tag, n))
        j = LOG.index(("END", tag, n), i)
        return LOG[i + 1:j]


def launches(entries):
    return [e for e in entries if e[0] in ("burst", "paced")]


def nbytes(entry):
    return sum(t for _p, t in entry[2])
