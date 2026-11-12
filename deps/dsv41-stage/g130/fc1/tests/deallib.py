"""g130 fc1 helpers for the probe and the GPU test (STOPPED-SERVICE WINDOW ONLY; g1lib.guard refuses otherwise).

Arms, each native loaded once from its own tree in one process (g1lib's mechanism):
  all    the production FC1 tree (m32-bw all-base, jit/all-base)
  deal / dealx / nomma   kit trees (trees/<arm>-base, jit/<arm>-base, pinned in MANIFESTS.json)
FC2 is not under test: every arm shares the all-d05 native.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sys

KIT = Path(__file__).resolve().parent.parent
G1 = Path.home() / "dsv41-stage/g1/m32-bw"
sys.path.insert(0, str(G1 / "tests"))
import g1lib  # noqa: E402

MANIFESTS = json.loads((KIT / "MANIFESTS.json").read_text())
N576 = 576


def start(force=False, force_gpu_shared=False):
    g1lib.guard(force, force_gpu_shared)
    g1lib.setup_env(KIT / "work")
    # as the production launcher: FC1 receives packed scales and launches no transpose kernel
    os.environ["DSV41_MOE_FUSED_PERMUTE"] = os.environ.get("G130_FC1_FUSED_PERMUTE", "1")
    os.environ["DSV41_MOE_LIVE_ACT_MIN_ROWS"] = "192"
    import torch
    return torch


def load_arms(arms, extra=()):
    """-> (g1lib.Natives, {arm: kernel}). extra = [(name, tree root)]: unpinned timing-only trees (A3)."""
    nat = g1lib.Natives(["all"])
    g1lib.install_live_getter(nat)
    m32 = nat.m32
    kernels = {"all": nat.kernels["all"]}
    ld = nat.mods["g1all_d05._C"]
    for a in arms:
        if a == "all":
            continue
        vb = nat._load(f"g130{a}_base._C", KIT / f"trees/{a}-base", KIT / f"jit/{a}-base", MANIFESTS[f"{a}-base"],
                       m32._NATIVE_SHA256, alignment=128)
        k = m32.TargetM32Kernel(vb, ld, "1")
        k.g1_la = True
        kernels[a] = k
    for name, root in extra:
        root = Path(root).resolve()
        import hashlib
        raw = (root / "manifest.json").read_bytes()
        jit = KIT / f"jit/{name}-extra"
        (jit / "tmp").mkdir(parents=True, exist_ok=True)
        (jit / "cache").mkdir(parents=True, exist_ok=True)
        vb = nat._load(f"g130{name}_extra._C", root, jit, hashlib.sha256(raw).hexdigest(),
                       json.loads(raw)["native_sha256"], alignment=128)
        k = m32.TargetM32Kernel(vb, ld, "1")
        k.g1_la = True
        kernels[name] = k
    return nat, kernels


def m_sum(tokens):
    routes = 6 * tokens
    rows = routes + min(routes, g1lib.E) * 31
    return -(-rows // 32) * 32


@contextlib.contextmanager
def capture_fc1(kernel, out):
    """Record the FC1 call of one eager apply: (aq, sfa), (bw, sfb), d, layout, kwargs."""
    orig = kernel.gemm

    def g(*args, **kw):
        if int(args[2].shape[1]) == N576:
            out.append((args, dict(kw)))
        return orig(*args, **kw)
    kernel.gemm = g
    try:
        yield out
    finally:
        del kernel.gemm


class Fc1Input:
    """Real FC1 inputs for one (weights, tokens), captured from the production apply, every row filled with real rows."""

    def __init__(self, torch, arm, tokens, seed):
        self.torch = torch
        gen = torch.Generator().manual_seed(seed)
        x = (torch.randn(tokens, g1lib.H, generator=gen) * 0.15).to(torch.bfloat16).cuda()
        tw = (torch.softmax(torch.randn(tokens, g1lib.TOPK, generator=gen), -1) * 1.5).float().cuda()
        ids = g1lib.synthetic_routes(tokens, gen)["valid"].cuda()
        ws = arm.workspaces(tokens)
        calls = []
        kernel = arm.ex._dsv41_target_m32
        with capture_fc1(kernel, calls):
            arm.run(x, ids, tw, ws)
        torch.cuda.synchronize()
        assert len(calls) == 1, len(calls)
        (aq, sfa), (self.bw, self.sfb), d, layout = calls[0][0]
        self.kw = calls[0][1]
        self.M_sum = int(aq.shape[0])
        assert self.M_sum == m_sum(tokens) and self.M_sum % 32 == 0, (self.M_sum, tokens)
        self.nmb = self.M_sum // 32
        live = (layout >= 0).nonzero()[:, 0]
        idx = live[torch.randint(len(live), (self.M_sum,), generator=gen).to(live.device)]
        idx[live] = live
        self.aq = aq[idx].contiguous()
        self.sfa = sfa                     # keeps its strides (a fused-permute scale tensor is private to that call)
        self.sfa.copy_(sfa[idx].clone())
        assert self.aq.data_ptr() % 16 == 0
        self.d = torch.empty((self.M_sum, N576), dtype=torch.bfloat16, device="cuda")
        self.layout = torch.full((self.M_sum,), -1, dtype=torch.int32, device="cuda")

    def call(self, kernel, layout=None, d=None, bw=None, sfb=None):
        with kernel.alignment_scope():
            kernel.gemm((self.aq, self.sfa), (self.bw if bw is None else bw, self.sfb if sfb is None else sfb),
                        self.d if d is None else d, self.layout if layout is None else layout, **self.kw)


def make_layout(torch, M_sum, G, gen, rows="few", multi=0.0):
    """Layout with exactly G live m-blocks, as ep_scatter_1 / the mfront plan lay it out: experts in ascending id, 32-aligned
    contiguous segments from row 0, live rows a prefix of each segment. rows: one | few (1..6) | full | any (1..32).
    multi: probability that an expert takes two m-blocks (first full), as a hot expert does above 32 rows."""
    nmb = M_sum // 32
    assert 0 <= G <= nmb
    out = torch.full((M_sum,), -1, dtype=torch.int32)
    if G == 0:
        return out
    ex = sorted(torch.randperm(g1lib.E, generator=gen)[:G].tolist())
    m = 0
    for e in ex:
        if m >= G:
            break
        two = m + 1 < G and multi > 0 and float(torch.rand(1, generator=gen)) < multi
        if two:
            out[m * 32:(m + 1) * 32] = e
            m += 1
        hi = {"one": 2, "few": 7, "full": 33, "any": 33}[rows]
        lo = 32 if rows == "full" else 1
        c = 1 if rows == "one" else int(torch.randint(lo, hi, (1,), generator=gen))
        out[m * 32:m * 32 + c] = e
        m += 1
    assert m == G, (m, G)
    return out


def same_bytes(torch, x, y):
    return x.shape == y.shape and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


def graph_of(torch, body):
    """As m32_gsweep / serving capture: one eager run, one on a side stream, then capture."""
    body()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        body()
    return gr
