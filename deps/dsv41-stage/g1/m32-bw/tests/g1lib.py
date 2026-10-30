"""g1/m32-bw shared helpers for the GPU unit test and the microbench (STOPPED-SERVICE WINDOW ONLY).

Loads, in ONE process and each exactly once (every tree has its own .so copy, so dlopen gives each its own state):
  live      byte-identical copies (trees/live-base, trees/live-d05; the live pinned manifests 5f939d00 / fc230f26) of
            the serving trees (b1c4eff8 FC1 tree, Codex D05 6fea5091 FC2 tree) through the live site file's own
            TargetM32Kernel, with PRIVATE copies of the serving JIT caches (jit/live-base, jit/live-d05): no shared
            cache is read-modified or written and no serving tree file is dlopen'ed.
  <variant> trees/<variant>-base (+ trees/<variant>-d05 for ds/all; FC1-only variants reuse the live D05 native) with
            the pre-JIT caches jit/<tree> built by ops/build-jit.sh.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

G = Path(__file__).resolve().parent.parent            # ~/dsv41-stage/g1/m32-bw
HOME = Path.home()
LIVE_BASE = HOME / "dsv41-stage/m32/target-m32-native-v1"
LIVE_D05 = HOME / "dsv41-goal120-20261024/d05-shadow-v1/native"
# byte-identical copies of the two live trees (ops/prep-node.sh; same pinned manifests): the test processes never
# dlopen the serving tree files themselves, so a staged loader can load those in the same process without sharing state
LIVE_BASE_COPY = G / "trees/live-base"
LIVE_D05_COPY = G / "trees/live-d05"
LIVE_BASE_CACHE = HOME / "dsv41-stage/m32/target-m32-jit-cache-v1"
LIVE_D05_CACHE = HOME / "dsv41-stage/pd/d05/jit-cache"
VARIANTS = ("off", "pair", "pf", "la", "ds", "all")
HAS_D05_TREE = ("off", "ds", "all")                    # pair/pf/la change FC1 only
MANIFESTS = json.loads((G / "MANIFESTS.json").read_text()) if (G / "MANIFESTS.json").exists() else {}


def guard(force=False, force_gpu_shared=False):
    if force:
        return
    st = subprocess.run(["systemctl", "--user", "is-active", "dsv41-rank"], capture_output=True, text=True)
    assert st.stdout.strip() != "active", "dsv41-rank is active: stopped-service window only"
    if not force_gpu_shared:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True)
        assert not apps.stdout.strip(), "GPU busy: " + apps.stdout.strip()


def setup_env(work: Path):
    """MoE env exactly as launcher 04b6738e sets it for the target routed experts; private Triton/DeepGEMM caches."""
    os.environ.update(
        DSV41_SM121_TARGET_M32="1", DSV41_TARGET_M32_MAX_M="96", DSV41_SM121_TARGET_M64="1",
        DSV41_MOE_COMPACT="1", DSV41_MOE_LIVE_ACT="1", DSV41_MOE_FC2_D05="1", VLLM_USE_DEEP_GEMM="1",
        CUDA_HOME="/usr/local/cuda", TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas")
    for k in ("DSV41_MOE_PDL_EARLY_B", "DSV41_MOE_M32_G1", "VLLM_MARLIN_INPUT_DTYPE"):
        os.environ.pop(k, None)
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["DSV41_TARGET_M32_NATIVE_DIR"] = str(LIVE_BASE)
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(work / "triton-cache"))
    os.environ.setdefault("DG_JIT_CACHE_DIR", str(work / "dg-cache"))     # installed deep_gemm (not used by M32)


def private_cache(src: Path, dst: Path):
    """Private copy of a serving JIT cache (cache/ only), refreshed if the source gained entries."""
    if not (dst / "cache").is_dir() or sorted(p.name for p in (dst / "cache").iterdir()) != sorted(
            p.name for p in (src / "cache").iterdir()):
        shutil.rmtree(dst, ignore_errors=True)
        (dst / "tmp").mkdir(parents=True)
        shutil.copytree(src / "cache", dst / "cache")
    return dst


def tree_sha(root: Path) -> str:
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


class Natives:
    """All natives of this process. Mirrors sm121_target_m32.get_target_m32_kernel / load_d05_native."""

    def __init__(self, variants):
        import torch  # noqa: F401
        import vllm.model_executor.layers.fused_moe.experts.sm121_target_m32 as m32
        from vllm.utils import deep_gemm as installed
        self.m32 = m32
        installed._lazy_init()
        old = installed._import_deep_gemm()
        assert old is not None
        self.pdl = old.get_pdl()
        self.old = old
        self.mods = {}
        # live trees: pinned shas from the live site file
        lb = self._load("g1live_base._C", LIVE_BASE_COPY, private_cache(LIVE_BASE_CACHE, G / "jit/live-base"),
                        m32._MANIFEST_SHA256, m32._NATIVE_SHA256, alignment=128)
        ld = self._load("g1live_d05._C", LIVE_D05_COPY, private_cache(LIVE_D05_CACHE, G / "jit/live-d05"),
                        m32._D05_MANIFEST_SHA256, m32._D05_NATIVE_SHA256, alignment=32, tc_from=lb)
        self.kernels = {"live": m32.TargetM32Kernel(lb, ld, "1")}

        class LAKernel(m32.TargetM32Kernel):
            """g1-la reads A rows with cp.async at row stride SHAPE_K: contiguous, 16-B aligned A only."""

            def gemm(self, *args, **kwargs):
                assert _la_ok(args), "g1-la needs contiguous 16-B aligned A"
                return super().gemm(*args, **kwargs)
        for v in variants:
            vb = self._load(f"g1{v}_base._C", G / f"trees/{v}-base", G / f"jit/{v}-base",
                            MANIFESTS.get(f"{v}-base"), m32._NATIVE_SHA256, alignment=128)
            vd = ld
            if v in HAS_D05_TREE:
                vd = self._load(f"g1{v}_d05._C", G / f"trees/{v}-d05", G / f"jit/{v}-d05",
                                MANIFESTS.get(f"{v}-d05"), m32._D05_NATIVE_SHA256, alignment=32, tc_from=vb)
            self.kernels[v] = (LAKernel if v in ("la", "all") else m32.TargetM32Kernel)(vb, vd, "1")
        # every module must hold its own runtime state
        for name, mod in self.mods.items():
            assert mod.get_num_sms() == 48, name

    def _load(self, name, root: Path, cache: Path, manifest_sha, native_sha, alignment, tc_from=None):
        """As the live loaders: base natives keep their own tc_util; D05 natives copy the base native's."""
        root, cache = root.resolve(), cache.resolve()
        assert cache.exists(), cache
        if manifest_sha is None:
            raise AssertionError(f"no pinned manifest for {root} (MANIFESTS.json)")
        self.m32._verify_tree(root, manifest_sha, native_sha)
        man = json.loads((root / "manifest.json").read_text())
        prior = os.environ.get("DG_JIT_CACHE_DIR")
        os.environ["DG_JIT_CACHE_DIR"] = str(cache)
        try:
            assert name not in sys.modules
            spec = importlib.util.spec_from_file_location(name, root / man["native"])
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            mod.init(str(root))
            mod.set_pdl(self.pdl)
            if tc_from is not None and hasattr(mod, "set_tc_util") and hasattr(tc_from, "get_tc_util"):
                mod.set_tc_util(tc_from.get_tc_util())
            mod.set_mk_alignment_for_contiguous_layout(alignment)
        finally:
            if prior is None:
                os.environ.pop("DG_JIT_CACHE_DIR", None)
            else:
                os.environ["DG_JIT_CACHE_DIR"] = prior
        for other in self.mods.values():       # separate dlopen state: alignment must not leak across modules
            assert other is not mod
        self.mods[name] = mod
        return mod


def _la_ok(args):
    (aq, _sfa) = args[0]
    return aq.dim() == 2 and aq.is_contiguous() and aq.data_ptr() % 16 == 0


def install_live_getter(nat: Natives):
    """DeepGemmFP4Experts.__init__ calls get_target_m32_kernel(); hand it our privately loaded live kernel."""
    nat.m32.get_target_m32_kernel = lambda: nat.kernels["live"]


# ------------------------------------------------------------------ model geometry / weights
E, H, I, P, TOPK, LIMIT = 384, 5120, 288, 384, 6, 10.0
MODEL = HOME / "models/dsv41/uncensored-fp8"


def make_cfg(ipp=P):
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig, FusedMoEParallelConfig,
                                                            RoutingMethodType)
    par = FusedMoEParallelConfig(tp_size=8, pcp_size=1, dp_size=1, ep_size=1, tp_rank=0, pcp_rank=0, dp_rank=0,
                                 ep_rank=0, sp_size=1, use_ep=False, all2all_backend="naive", enable_eplb=False)
    cfg = FusedMoEConfig(num_experts=E, experts_per_token=TOPK, hidden_dim=H, intermediate_size=I * 8,
                         num_local_experts=E, num_logical_experts=E, activation=MoEActivation.SILU, device="cuda",
                         routing_method=list(RoutingMethodType)[0], moe_parallel_config=par,
                         in_dtype=torch.bfloat16, swiglu_limit=LIMIT)
    cfg.intermediate_size_per_partition = ipp
    return cfg


def sf(raw, n, k):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import deepgemm_post_process_weight_scale_block
    return deepgemm_post_process_weight_scale_block(ws=raw, mn=n, k=k, quant_block_shape=(1, 32), num_groups=E)


def synthetic_weights(seed=20261026):
    """Compact FC1 [E, 576, H/2] + FC2 [E, H, 192] (K 288 + 96 zero pad) uint8, e8m0 scales 119..123."""
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    w1 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda", generator=g)
    s1 = torch.randint(119, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device="cuda", generator=g)
    w2 = torch.randint(0, 256, (E, H, P // 2), dtype=torch.uint8, device="cuda", generator=g)
    s2 = torch.randint(119, 124, (E, H, P // 32), dtype=torch.uint8, device="cuda", generator=g)
    w2[:, :, I // 2:] = 0
    s2[:, :, I // 32:] = 0
    return w1, s1, w2, s2


_readers = {}


def _get_tensor(index, key):
    from safetensors import safe_open
    f = index[key]
    if f not in _readers:
        _readers[f] = safe_open(str(MODEL / f), framework="pt", device="cpu")
    return _readers[f].get_tensor(key)


def real_layer(layer, tp_rank):
    """One TP rank slice of a real checkpoint layer via vLLM's own RoutedExperts loader (as mm_marlin_bench)."""
    import torch
    from types import SimpleNamespace
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    full = {}
    for proj in ("w1", "w3", "w2"):
        for kind in ("weight", "scale"):
            full[(proj, kind)] = torch.stack([_get_tensor(index, f"layers.{layer}.ffn.experts.{e}.{proj}.{kind}")
                                              .view(torch.uint8) for e in range(E)])
    loader = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(loader)
    loader.moe_config = SimpleNamespace(is_act_and_mul=True, tp_shard_with_padding=False,
                                        moe_parallel_config=SimpleNamespace(tp_size=8))
    w1 = torch.zeros(E, 2 * P, H // 2, dtype=torch.uint8)
    s1 = torch.zeros(E, 2 * P, H // 32, dtype=torch.uint8)
    w2 = torch.zeros(E, H, P // 2, dtype=torch.uint8)
    s2 = torch.zeros(E, H, P // 32, dtype=torch.uint8)
    for e in range(E):
        for proj in ("w1", "w3", "w2"):
            for kind, dst in (("weight", w2 if proj == "w2" else w1), ("scale", s2 if proj == "w2" else s1)):
                loader._load_model_weight_or_group_weight_scale(
                    1 if proj == "w2" else 0, dst[e], proj, full[(proj, kind)][e], tp_rank, is_scale=kind == "scale")
    del full
    w1, s1, w2, s2 = (x.cuda() for x in (w1, s1, w2, s2))
    w1[:, I:P] = 0
    w1[:, P + I:] = 0
    s1[:, I:P] = 0
    s1[:, P + I:] = 0
    w2[:, :, I // 2:] = 0
    s2[:, :, I // 32:] = 0
    w1c = torch.cat((w1[:, :I], w1[:, P:P + I]), 1).contiguous()
    s1c = torch.cat((s1[:, :I], s1[:, P:P + I]), 1).contiguous()
    return w1c, s1c, w2, s2


class Arm:
    """DeepGemmFP4Experts.apply with the serving input quant, bound to one TargetM32Kernel."""

    def __init__(self, kernel, w1c, s1c, w2, s2):
        from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmFP4Experts
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (Mxfp4MoeBackend,
                                                                      make_mxfp4_moe_quant_config)
        self.qc = make_mxfp4_moe_quant_config(Mxfp4MoeBackend.DEEPGEMM_MXFP4, sf(s1c, 2 * I, H), sf(s2, H, P),
                                              swiglu_limit=LIMIT)
        self.ex = DeepGemmFP4Experts(make_cfg(P), self.qc)
        self.ex._dsv41_target_m32 = kernel
        self.w1, self.w2 = w1c, w2
        self.bytes_per_expert = 2 * I * (H // 2 + H // 32) + H * (P // 2 + P // 32)

    def workspaces(self, m):
        import torch
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        sizes = self.ex.workspace_shapes(m, 2 * I, H, TOPK, E, E, None, MoEActivation.SILU)
        return [torch.empty(s, dtype=torch.bfloat16, device="cuda") for s in sizes]

    def run(self, x, ids, tw, ws):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import _quantize_input
        aq, sfa = _quantize_input(x, self.qc)
        x13, y2, out = ws
        self.ex.apply(output=out, hidden_states=aq, w1=self.w1, w2=self.w2, topk_weights=tw, topk_ids=ids,
                      activation=MoEActivation.SILU, global_num_experts=E, expert_map=None, a1q_scale=sfa,
                      a2_scale=None, workspace13=x13, workspace2=y2, expert_tokens_meta=None,
                      apply_router_weight_on_input=False)
        return out


# ------------------------------------------------------------------ routes
def load_recorded(path: Path):
    """10-21 recorded decode routes (rank-0 recorder): 6-row single-user windows and 4-user batches."""
    import torch
    wins, c4 = [], {}
    for fn in sorted(path.glob("*.jsonl")):
        for line in open(fn):
            d = json.loads(line)
            if d.get("event") != "routes" or d.get("validation_errors") not in ([], "[]", None):
                continue
            if d.get("has_prefill") in (True, "True"):
                continue
            ids = d["ids_by_layer"]
            ids = json.loads(ids) if isinstance(ids, str) else ids
            t = torch.tensor(ids, dtype=torch.int32)
            rows, nreq = int(d["physical_rows"]), int(d["num_requests"])
            assert t.shape == (40, rows, TOPK), t.shape
            if rows == 6 and nreq == 1:
                wins.append(t)
            elif nreq == 4 and rows in (12, 24):
                c4.setdefault(rows, []).append(t)
    return wins, c4


def synthetic_routes(m, gen):
    """Route patterns that stress the M32 layout (experts distinct within every row):
    valid (uniform), spec (4 experts shared by all rows), half (odd rows all-dead: ROUTE_LIVE -1 slots),
    hot (all rows on the same 6 experts: multi-block experts at m > 32), skew (one expert on every row, a second on
    half the rows), one (a single live slot in the whole batch)."""
    import torch

    def rows_with(shared):
        out = []
        for _ in range(m):
            rest = [e for e in torch.randperm(E, generator=gen).tolist() if e not in shared][:TOPK - len(shared)]
            out.append(list(shared) + rest)
        return torch.tensor(out, dtype=torch.int32)

    valid = rows_with([])
    base = valid[0].tolist()
    spec = rows_with(base[:4])
    half = valid.clone()
    half[1::2] = -1
    hot = valid[:1].repeat(m, 1)
    skew = torch.cat((rows_with(base[:2])[: (m + 1) // 2], rows_with(base[:1])[: m // 2]), 0)
    one = torch.full_like(valid, -1)
    one[0, 0] = base[0]
    return dict(valid=valid, spec=spec, half=half, hot=hot, skew=skew, one=one)
