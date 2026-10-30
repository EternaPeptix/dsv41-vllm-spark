#!/usr/bin/env python3
"""g2 fc2c-on-m32all GPU unit test: staged sm121_target_m32.py with DSV41_MOE_M32_G1=all + DSV41_MOE_FC2C
(FC2-compact v2 tail on the m32-bw 'all' FC2 tree) against the LIVE serving FC2 (m32-bw all-d05 tree).

STOPPED-SERVICE WINDOW ONLY: refuses to run while dsv41-rank is active or any GPU compute process exists.
One rank per invocation (all 8 in parallel with --layers all --layer-offset R --layer-stride 8 --tp-rank R covers
the 40 target layers once). Private JIT/Triton/DeepGEMM caches under ~/dsv41-stage/g2/fc2c-on-m32all/jit-test/
(ops/prep-node.sh pre makes them; the serving caches are never written: the staged loader's _G1_ROOT is pointed at
jit-test/g1root, whose trees/ is a symlink to the real m32-bw trees and whose jit/ holds private cache copies).

  ~/venvs/vllm032/bin/python ~/dsv41-stage/g2/fc2c-on-m32all/tests/test_g2_unit.py                   # synthetic
  ~/venvs/vllm032/bin/python .../test_g2_unit.py --layers all --layer-offset R --layer-stride 8 --tp-rank R

Natives (one process, each loaded once, every .so its own file): base = m32-bw trees/all-base (FC1, 'all'),
d05 = trees/all-d05 (live target FC2), fc2c = g2 tree/ (_C 33d4eb19 + merged header), all three through the
STAGED loader get_target_m32_kernel() with DSV41_MOE_FC2C=1 (the serving construction); orig = m32-bw
trees/live-base (byte copy of the serving base tree, original header) for an independent FC2 reference.
Arms (identical inputs):
  live       the INSTALLED live site file's TargetM32Kernel (db659d80) on base + d05: the serving FC2 today
  staged_off staged TargetM32Kernel with FC2C off on the same natives (must equal live)
  fc2c       the kernel get_target_m32_kernel() returned: serving gate 2 tokens (M=1 -> d05, M>=2 -> compact)
  fc2c_all   gate 1 token (compact at every M incl. M=1), tails shared with fc2c
  orig       original FC2 kernel (no D05, no g1 features): live rows only
Checks (bitwise, uint8 views; NaN/Inf compared as bits):
  tail      per layer: tail rows == w2 bytes 128..143 in the .b4x16_p64 image, pads 0, 62,914,560 B
  direct    scatter32 layouts (deepgemm_moe_permute), LIVE_ACT SiLU, then STALE PADDING ROWS as LIVE_ACT leaves
            them (dead-row FP8 inputs and packed UE8M0 scales poisoned with NaN bytes / random bytes). The 'all'
            kernels store live rows only: with ONE sentinel for live/staged_off/fc2c/fc2c_all the WHOLE output
            buffers must be bitwise equal (same rows written, same bytes, rest untouched); live rows == orig and
            finite. Rows written by live are reported (expected: exactly the live rows).
  moe       DeepGemmFP4Experts.apply (COMPACT + LIVE_ACT as served), NaN-poisoned workspaces: all arms == live
  replay    CUDA-graph capture + replay with changing routes, workspaces poisoned between replays
  miss      fresh FC2Compact whose weight is first seen inside a capture: d05 fallback, no tail allocated,
            replay == live; the next eager call builds the tail and runs compact
  verify    DSV41_MOE_FC2C=verify (poison_served as get_target_m32_kernel sets it for 'all'): direct calls and full
            MoE apply: device counters clean, served output == live; control without the poison must count
            mismatches (proves the compare sees padding rows)
  gating    M=1 never compact under the serving gate; M=2 compact; M=97 reaches neither
  engaged   the private fc2c JIT cache holds exactly the pre-JIT entry (cache HIT: the serving pre-JIT cubin is the
            kernel that ran, no compile); compact calls > 0 in both compact arms
Exit status 0 only on PASS. JSON lines on stdout; the last line is the summary.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
HOME = Path.home()
SITE = HOME / "venvs/vllm032/lib/python3.12/site-packages"
EXP = SITE / "vllm/model_executor/layers/fused_moe/experts"
LIVE_M32_SHA = "db659d809cbcc3949f94dd810d9f00359f1770af082aadd8f7037816371f02da"
LIVE_VER_SHA = "c04c363dfa54bd8a01002e59974503a9bc08d09b34e517ae0bba277fd9af86cd"
LIVE_MOE_SHA = "5162dc0fdb59ea83d413411e861623d0c3a68f3fe2e87c2b20bc1ad0d937b93f"
LIVE_UTILS_SHA = "a4a0486a655d9c6953d4429ed3cf7fd760d2def2d56103438adca7bccdbcd580"
M32BW = HOME / "dsv41-stage/g1/m32-bw"
ORIG_TREE = M32BW / "trees/live-base"           # byte copy of the serving base tree (manifest 5f939d00)
TAIL_BYTES = 384 * 5120 * 32
BAK = ".bak-g2-fc2c-on-m32all"

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--tokens", default="1,2,4,6,12,24,48,96")
p.add_argument("--layers", default="synthetic", help="synthetic | all | comma list of layer ids")
p.add_argument("--layer-offset", type=int, default=0)
p.add_argument("--layer-stride", type=int, default=1)
p.add_argument("--tp-rank", type=int, default=0)
p.add_argument("--replays", type=int, default=2, help="graph replay cycles over all route sets")
p.add_argument("--seed", type=int, default=20261026)
p.add_argument("--force", action="store_true", help="skip the stopped-service check (never on a serving node)")
a = p.parse_args()


def emit(**kw):
    print(json.dumps(kw), flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if not a.force:
    st = subprocess.run(["systemctl", "--user", "is-active", "dsv41-rank"], capture_output=True, text=True)
    assert st.stdout.strip() in ("inactive", "failed", "unknown", ""), "dsv41-rank is active: " + st.stdout
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True)
    assert not apps.stdout.strip(), "GPU occupied: " + apps.stdout.strip()

JT = STAGE / "jit-test"
for sub in ("g1root/jit/all-base", "g1root/jit/all-d05", "fc2c-all", "live-base", "dg", "triton"):
    assert (JT / sub).is_dir(), f"missing {JT / sub}: run ops/prep-node.sh pre first"
assert (JT / "g1root/trees").resolve() == (M32BW / "trees").resolve(), "jit-test/g1root/trees must link the m32-bw trees"
os.environ.update(
    DSV41_SM121_TARGET_M32="1", DSV41_TARGET_M32_MAX_M="96", DSV41_SM121_TARGET_M64="1",
    DSV41_MOE_COMPACT="1", DSV41_MOE_LIVE_ACT="1", VLLM_USE_DEEP_GEMM="1",
    CUDA_HOME="/usr/local/cuda", TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
    DSV41_MOE_FC2_D05="1", DSV41_MOE_M32_G1="all", DSV41_MOE_FC2C="1",
    DSV41_MOE_FC2C_JIT_DIR=str(JT / "fc2c-all"),
    DG_JIT_CACHE_DIR=str(JT / "dg"), TRITON_CACHE_DIR=str(JT / "triton"),
    DSV41_MOE_FC2_D05_VERIFY_DIR=str(STAGE / "results"))
for k in ("DSV41_MOE_PDL_EARLY_B", "DSV41_MOE_FC2C_MIN_TOKENS", "DSV41_MOE_FC2C_NATIVE_DIR", "VLLM_MARLIN_INPUT_DTYPE",
          "DSV41_MOE_FC2_D05_NATIVE_DIR", "DSV41_MOE_FC2_D05_JIT_DIR"):
    os.environ.pop(k, None)
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ["DSV41_TARGET_M32_NATIVE_DIR"] = str(HOME / "dsv41-stage/m32/target-m32-native-v1")

import torch  # noqa: E402

assert torch.cuda.get_device_capability() == (12, 1)
assert torch.cuda.get_device_properties(0).multi_processor_count == 48
torch.manual_seed(a.seed)
T_START = time.time()

import vllm.model_executor.layers.fused_moe.experts as _experts_pkg  # noqa: E402,F401


def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PKG = "vllm.model_executor.layers.fused_moe.experts"
STAGED_M32 = STAGE / "site/vllm/model_executor/layers/fused_moe/experts/sm121_target_m32.py"
INSTALLED_M32 = EXP / "sm121_target_m32.py"
# the live reference is db659d80: the installed file before an install, its .bak-g2 backup after one
LIVE_SRC = INSTALLED_M32 if sha(INSTALLED_M32) == LIVE_M32_SHA else INSTALLED_M32.with_name(INSTALLED_M32.name + BAK)
assert sha(LIVE_SRC) == LIVE_M32_SHA, f"no db659d80 live site file ({LIVE_SRC})"
assert sha(INSTALLED_M32) in (LIVE_M32_SHA, sha(STAGED_M32)), "unexpected installed sm121_target_m32.py"
assert sha(EXP / "sm121_d05_verify.py") == LIVE_VER_SHA
assert f"{PKG}.sm121_target_m32" not in sys.modules
m32 = load_as(f"{PKG}.sm121_target_m32", STAGED_M32)
live_m32 = load_as("g2live_sm121_target_m32", LIVE_SRC)
m32._G1_ROOT = JT / "g1root"          # trees/ -> real m32-bw trees (pinned manifests); jit/ -> private cache copies
from vllm.model_executor.layers.fused_moe.experts import sm121_d05_verify as _kv  # noqa: E402,F401

from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmFP4Experts  # noqa: E402
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: E402
    Mxfp4MoeBackend, make_mxfp4_moe_quant_config)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    deepgemm_post_process_weight_scale_block, per_token_group_quant_fp8)
from vllm.model_executor.layers.fused_moe import deep_gemm_utils as utils  # noqa: E402
from vllm.utils import deep_gemm as installed_dg  # noqa: E402

assert sha(sys.modules[DeepGemmFP4Experts.__module__].__file__) == LIVE_MOE_SHA
assert sha(utils.__file__) == LIVE_UTILS_SHA

E, H, I, P = 384, 5120, 288, 384
failures = cases = 0


def beq(x, y):
    return (x.shape == y.shape and x.dtype == y.dtype
            and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)))


def poison(*ts, value=255):
    for t in ts:
        t.view(torch.uint8).fill_(value)


def record(ok, **kw):
    global failures, cases
    cases += 1
    failures += not ok
    emit(ok=bool(ok), **kw)
    return ok


def cache_entries(root: Path):
    c = root / "cache"
    return sorted(x.name for x in c.iterdir()) if c.is_dir() else []


CACHE_BEFORE = {k: cache_entries(JT / k) for k in ("fc2c-all", "g1root/jit/all-base", "g1root/jit/all-d05", "live-base")}
PREJIT = sorted(x.name for x in (STAGE / "jit-cache/cache").iterdir())
assert CACHE_BEFORE["fc2c-all"] == PREJIT, ("private fc2c cache must start as a copy of the pre-JIT cache",
                                            CACHE_BEFORE["fc2c-all"], PREJIT)

# ---------------------------------------------------------------- kernels (serving construction)
k_serv = m32.get_target_m32_kernel()                  # G1=all, D05=1, FC2C=1: exactly what serving builds
assert k_serv.d05 == "1" and k_serv.fc2c is not None and k_serv.fc2c.mode == "1" and k_serv.g1_la
assert getattr(k_serv.fc2c, "poison_served", None) is True
assert k_serv.fc2c.min_rows == m32.fc2c_min_rows(2) == 384
base_native, d05_native, fc2c_native = k_serv.native, k_serv.fc2_native, k_serv.fc2c.native
pdl = installed_dg._import_deep_gemm().get_pdl()
paths = {n: str(Path(mod.__file__).resolve()) for n, mod in
         (("base", base_native), ("d05", d05_native), ("fc2c", fc2c_native))}
assert paths["base"] == str((M32BW / "trees/all-base").resolve() / Path(paths["base"]).name)
assert paths["d05"] == str((M32BW / "trees/all-d05").resolve() / Path(paths["d05"]).name)
assert paths["fc2c"] == str((STAGE / "tree").resolve() / Path(paths["fc2c"]).name)


def load_orig():
    """Pristine base native (original JIT header, no D05, no g1 features) with a private serving-cache copy."""
    root, cache = ORIG_TREE.resolve(), (JT / "live-base").resolve()
    m32._verify_tree(root, m32._MANIFEST_SHA256, m32._NATIVE_SHA256)
    man = json.loads((root / "manifest.json").read_text())
    prior = os.environ.get("DG_JIT_CACHE_DIR")
    os.environ["DG_JIT_CACHE_DIR"] = str(cache)
    try:
        name = "g2orig_base._C"
        spec = importlib.util.spec_from_file_location(name, root / man["native"])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mod.init(str(root))
        mod.set_pdl(pdl)
        mod.set_mk_alignment_for_contiguous_layout(128)
    finally:
        if prior is None:
            os.environ.pop("DG_JIT_CACHE_DIR", None)
        else:
            os.environ["DG_JIT_CACHE_DIR"] = prior
    return mod


orig_native = load_orig()
assert len({id(x) for x in (base_native, d05_native, fc2c_native, orig_native)}) == 4


class Counting:
    """Call counter around any TargetM32Kernel (live or staged class); everything else is delegated."""

    def __init__(self, kern):
        self.k = kern
        self.calls = self.d05_calls = self.fc2c_calls = 0

    def __getattr__(self, name):
        return getattr(self.k, name)

    def alignment_scope(self):
        return self.k.alignment_scope()

    def gemm(self, *args, **kwargs):
        self.calls += 1
        fc = getattr(self.k, "fc2c", None)
        before = fc.eager_calls if fc is not None else 0
        if self.k.d05 != "0" and m32.d05_eligible(args, kwargs):
            self.d05_calls += 1
        out = self.k.gemm(*args, **kwargs)
        if fc is not None:
            self.fc2c_calls += fc.eager_calls - before
        return out


def with_la(k):
    k.g1_la = True
    return k


min2, min1 = m32.fc2c_min_rows(2), m32.fc2c_min_rows(1)
fc_serv = k_serv.fc2c
fc_all = m32.FC2Compact(fc2c_native, "1", min1)
fc_all.entries = fc_serv.entries                       # one tail per weight, shared by both gates
K = {
    "orig": Counting(m32.TargetM32Kernel(orig_native)),
    "live": Counting(with_la(live_m32.TargetM32Kernel(base_native, d05_native, "1"))),
    "staged_off": Counting(with_la(m32.TargetM32Kernel(base_native, d05_native, "1"))),
    "fc2c": Counting(k_serv),
    "fc2c_all": Counting(with_la(m32.TargetM32Kernel(base_native, d05_native, "1", None, fc_all))),
}
emit(event="kernels", pdl=pdl, natives={n: sha(p_)[:16] for n, p_ in paths.items()},
     orig_native=sha(orig_native.__file__)[:16], tree_manifest=sha(STAGE / "tree/manifest.json")[:16],
     min_rows_serving=min2, min_rows_all=min1, jit_test=str(JT), live_site=str(LIVE_SRC), prejit=PREJIT)

parallel = FusedMoEParallelConfig(tp_size=8, pcp_size=1, dp_size=1, ep_size=1, tp_rank=0, pcp_rank=0,
                                  dp_rank=0, ep_rank=0, sp_size=1, use_ep=False, all2all_backend="naive",
                                  enable_eplb=False)
cfg = FusedMoEConfig(num_experts=E, experts_per_token=6, hidden_dim=H, intermediate_size=I * 8,
                     num_local_experts=E, num_logical_experts=E, activation=MoEActivation.SILU,
                     device="cuda", routing_method=list(RoutingMethodType)[0],
                     moe_parallel_config=parallel, in_dtype=torch.bfloat16, swiglu_limit=10.)
cfg.intermediate_size_per_partition = P


def sf(raw, n, k):
    return deepgemm_post_process_weight_scale_block(ws=raw, mn=n, k=k, quant_block_shape=(1, 32), num_groups=E)


def checkpoint_layer_ids():
    model = HOME / "models/dsv41/uncensored-fp8"
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    return sorted({int(k.split(".")[1]) for k in index
                   if k.startswith("layers.") and ".ffn.experts.0.w2.weight" in k})


def weights(layer):
    """compact w1 [E,576,H/2], w1 scale, w2 [E,H,192], w2 scale (raw UE8M0 bytes); same loader as the D05 test."""
    if layer is None:
        w1 = torch.randint(0, 256, (E, 2 * P, H // 2), dtype=torch.uint8, device="cuda")
        s1 = torch.randint(119, 124, (E, 2 * P, H // 32), dtype=torch.uint8, device="cuda")
        w2 = torch.randint(0, 256, (E, H, P // 2), dtype=torch.uint8, device="cuda")
        s2 = torch.randint(119, 124, (E, H, P // 32), dtype=torch.uint8, device="cuda")
    else:
        from safetensors import safe_open
        from types import SimpleNamespace
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        model = HOME / "models/dsv41/uncensored-fp8"
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        loader = object.__new__(RoutedExperts)
        torch.nn.Module.__init__(loader)
        loader.moe_config = SimpleNamespace(is_act_and_mul=True, tp_shard_with_padding=False,
                                            moe_parallel_config=SimpleNamespace(tp_size=8))
        w1 = torch.zeros(E, 2 * P, H // 2, dtype=torch.uint8)
        s1 = torch.zeros(E, 2 * P, H // 32, dtype=torch.uint8)
        w2 = torch.zeros(E, H, P // 2, dtype=torch.uint8)
        s2 = torch.zeros(E, H, P // 32, dtype=torch.uint8)
        readers = {}
        for e in range(E):
            for proj in ("w1", "w3", "w2"):
                for kind, dst in (("weight", w2 if proj == "w2" else w1), ("scale", s2 if proj == "w2" else s1)):
                    key = f"layers.{layer}.ffn.experts.{e}.{proj}.{kind}"
                    f = index[key]
                    if f not in readers:
                        readers[f] = safe_open(str(model / f), framework="pt", device="cpu")
                    src = readers[f].get_tensor(key).view(torch.uint8)
                    loader._load_model_weight_or_group_weight_scale(
                        1 if proj == "w2" else 0, dst[e], proj, src, a.tp_rank, is_scale=kind == "scale")
        w1, s1, w2, s2 = (x.cuda() for x in (w1, s1, w2, s2))
    w1[:, I:P] = 0
    w1[:, P + I:] = 0
    s1[:, I:P] = 0
    s1[:, P + I:] = 0
    w2[:, :, I // 2:] = 0
    s2[:, :, I // 32:] = 0
    w1c = torch.cat((w1[:, :I], w1[:, P:P + I]), dim=1).contiguous()
    s1c = torch.cat((s1[:, :I], s1[:, P:P + I]), dim=1).contiguous()
    return w1c, s1c, w2, s2


def routes(m):
    valid = torch.stack([torch.randperm(E, device="cuda")[:6] for _ in range(m)]).int()
    skew = torch.tensor([0, 1, 63, 127, 255, 383], device="cuda", dtype=torch.int32).repeat(m, 1)
    spec = valid.clone()
    spec[:, :4] = valid[0, :4]                      # C1 verify rows share most experts
    half = valid.clone()
    half[1::2] = -1                                 # ROUTE_LIVE=skip style dead verify rows
    holes = valid.clone()
    holes[torch.rand(m, 6, device="cuda") < .4] = -1
    one = torch.zeros_like(valid)                   # every route to expert 0 (multi-tile expert)
    one[:, 1:] = torch.arange(1, 6, device="cuda", dtype=torch.int32)
    return dict(valid=valid, skewed=skew, spec=spec, half=half, holes=holes, one=one,
                all_dead=torch.full_like(valid, -1))


def stale(a2q, a2s, live, kind):
    """LIVE_ACT leaves dead rows of the FC2 input (FP8 q and packed UE8M0 scales) stale: emulate it."""
    dead = ~live
    if kind == "nan":                               # e4m3 0x7F = NaN, UE8M0 0xFF = NaN scale
        a2q.view(torch.uint8).masked_fill_(dead[:, None], 0x7F)
        a2s.masked_fill_(dead[:, None], -1)
    else:                                           # random bytes / random (mostly finite) scales
        r = torch.randint(0, 256, a2q.shape, dtype=torch.uint8, device="cuda")
        a2q.view(torch.uint8).copy_(torch.where(dead[:, None], r, a2q.view(torch.uint8)))
        rs = torch.randint(-2 ** 31, 2 ** 31 - 1, a2s.shape, dtype=torch.int32, device="cuda")
        a2s.copy_(torch.where(dead[:, None], rs, a2s))


def fc2_inputs(control, aq, sfa, ids):
    perm = utils.deepgemm_moe_permute(aq=aq, aq_scale=sfa, topk_ids=ids, local_num_experts=E, expert_map=None,
                                      expert_tokens_meta=None, alignment_m=32, return_expert_ends=True)
    layout = perm[2]
    msum = layout.numel()
    mm1 = torch.randn(msum, 576, dtype=torch.bfloat16, device="cuda")
    q = torch.empty(msum, 384, dtype=torch.float8_e4m3fn, device="cuda")
    a2q, a2s = control._act_mul_quant(input=mm1, output=q, activation=MoEActivation.SILU,
                                      expert_ends=perm[5], expert_alignment=32)
    return a2q, a2s, layout, msum


def partial_groups(layout):
    g = layout.view(-1, 8)
    return int(((g[:, 0] >= 0) & (g < 0).any(dim=1)).sum())


tokens = [int(x) for x in a.tokens.split(",") if x]
if a.layers == "synthetic":
    layer_list = [None]
elif a.layers == "all":
    layer_list = checkpoint_layer_ids()[a.layer_offset::a.layer_stride]
else:
    layer_list = [int(x) for x in a.layers.split(",")][a.layer_offset::a.layer_stride]
emit(event="plan", layers=layer_list, tokens=tokens, tp_rank=a.tp_rank)

ARMS = ("live", "staged_off", "fc2c", "fc2c_all")
DS_SENTINEL = 0xA5
for layer in layer_list:
    t_load = time.time()
    w1, s1, w2, s2 = weights(layer)
    qc = make_mxfp4_moe_quant_config(Mxfp4MoeBackend.DEEPGEMM_MXFP4, sf(s1, 2 * I, H), sf(s2, H, P),
                                     swiglu_limit=10.)
    experts = {}
    for key in ARMS:
        ex = DeepGemmFP4Experts(cfg, qc)
        ex._dsv41_target_m32 = K[key]
        experts[key] = ex
    control = experts["live"]
    emit(event="layer", layer=layer, load_s=round(time.time() - t_load, 1))

    # ---- tail tensor of this weight (built by the serving FC2Compact, eager)
    w2i8 = w2.view(torch.int8)
    mem0 = torch.cuda.memory_allocated()
    descs = fc_serv.descs(w2i8)
    tail = fc_serv.entries[w2i8.data_ptr()][1]
    grow = torch.cuda.memory_allocated() - mem0
    u8 = w2.view(torch.uint8).reshape(E * H, P // 2)
    ok = (tuple(tail.shape) == (E * H, 32) and beq(tail[:, 0:8], u8[:, 128:136]) and beq(tail[:, 16:24], u8[:, 136:144])
          and not bool(tail[:, 8:16].any()) and not bool(tail[:, 24:32].any()) and not bool(u8[:, 144:].any())
          and tuple(descs.shape) == (6, 128) and descs.is_cuda and tail.numel() == TAIL_BYTES)
    record(ok, event="tail", layer=layer, tail_bytes=tail.numel(), alloc_growth=int(grow))

    for m in tokens:
        align = control._workspace_m_alignment(m, 576, H, 6, E)
        hidden = torch.randn(m, H, dtype=torch.bfloat16, device="cuda") * 2
        aq, sfa = per_token_group_quant_fp8(hidden, 128, use_ue8m0=True)
        router = torch.softmax(torch.randn(m, 6, device="cuda"), -1)
        ids = torch.empty(m, 6, dtype=torch.int32, device="cuda")
        rs = routes(m)
        sizes = control.workspace_shapes(m, 576, H, 6, E, E, None, MoEActivation.SILU)

        def pooled():
            # Mimic WorkspaceManager.get_simultaneous: disjoint 256-B aligned views of ONE allocation.
            nbytes = [int(torch.Size(s).numel()) * 2 for s in sizes]
            offs, tot = [], 0
            for n in nbytes:
                offs.append(tot)
                tot += (n + 255) // 256 * 256
            pool = torch.empty(tot, dtype=torch.uint8, device="cuda")
            return [pool[o:o + n].view(torch.bfloat16).view(s) for o, n, s in zip(offs, nbytes, sizes)]

        ws = {key: pooled() for key in ARMS}

        def run(key, ws=ws, experts=experts):  # bound per iteration (both are del'ed at the loop end)
            x, y, out = ws[key]
            experts[key].apply(output=out, hidden_states=aq, w1=w1, w2=w2, topk_weights=router,
                               topk_ids=ids, activation=MoEActivation.SILU, global_num_experts=E,
                               expert_map=None, a1q_scale=sfa, a2_scale=None, workspace13=x,
                               workspace2=y, expert_tokens_meta=None, apply_router_weight_on_input=False)
            return out

        # ---- direct FC2 on real scatter32 layouts with stale LIVE_ACT padding rows (align 32 only)
        if align == 32:
            for route, values in rs.items():
                ids.copy_(values)
                for kind in ("nan", "random"):
                    a2q, a2s, layout, msum = fc2_inputs(control, aq, sfa, ids)   # fresh inputs per kind
                    flags = (layout[::32] >= 0).cpu().tolist()
                    assert flags == [True] * sum(flags) + [False] * (len(flags) - sum(flags)), "prefix contract"
                    live = layout >= 0
                    stale(a2q, a2s, live, kind)
                    outs = {}
                    for key in ("orig",) + ARMS:
                        o = torch.empty(msum, H, dtype=torch.bfloat16, device="cuda")
                        poison(o, value=0x3C if key == "orig" else DS_SENTINEL)
                        kern = K[key]
                        with kern.alignment_scope():
                            kern.gemm((a2q, a2s), (w2i8, control.w2_scale), o, layout,
                                      recipe_a=(1, 128), recipe_b=(1, 32))
                        outs[key] = o
                    torch.cuda.synchronize()
                    ref = outs["live"]
                    whole = {key: beq(ref, outs[key]) for key in ARMS[1:]}
                    written = (ref.view(torch.uint8) != DS_SENTINEL).any(dim=1)
                    live_orig = beq(outs["orig"][live], ref[live])
                    finite = bool(torch.isfinite(ref[live]).all())
                    ok = all(whole.values()) and live_orig and finite
                    record(ok, event="direct_fc2", layer=layer, m=m, route=route, stale=kind, m_sum=msum,
                           live_rows=int(live.sum()), rows_written_by_live=int(written.sum()),
                           written_eq_live=bool(torch.equal(written, live)), partial_groups=partial_groups(layout),
                           compact_engaged=msum >= min2, whole_buffer_eq_live=whole, live_eq_orig=live_orig,
                           finite_live=finite)

        # ---- whole MoE, eager (COMPACT + LIVE_ACT as served), poisoned workspaces
        refs = {}
        for route, values in rs.items():
            ids.copy_(values)
            got = {}
            for key in ARMS:
                poison(*ws[key])
                got[key] = run(key).clone()
            torch.cuda.synchronize()
            ref = got["live"]
            eq = {k: beq(ref, got[k]) for k in ARMS[1:]}
            ok = all(eq.values()) and bool(torch.isfinite(ref).all())
            if route == "all_dead":
                ok = ok and not bool(ref.any())
            refs[route] = ref
            record(ok, event="moe_eager", layer=layer, m=m, align=align, route=route, eq_live=eq)

        # ---- whole MoE, graph capture + replay with changing routes
        for key in ("fc2c", "fc2c_all"):
            ids.copy_(rs["valid"])
            run(key)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run(key)
            for route, values in list(rs.items()) * a.replays + [("valid", rs["valid"])]:
                ids.copy_(values)
                poison(*ws[key], value=0x7F)
                g.replay()
                torch.cuda.synchronize()
                record(beq(ws[key][2], refs[route]), event="moe_replay", layer=layer, m=m, arm=key, route=route)
            del g
        del ws, refs
        torch.cuda.empty_cache()

    # ---- capture miss: weight first seen inside a capture -> d05 fallback inside the graph, still bitwise
    m = 6
    fc_miss = m32.FC2Compact(fc2c_native, "1", min2)
    k_miss = Counting(with_la(m32.TargetM32Kernel(base_native, d05_native, "1", None, fc_miss)))
    ex = DeepGemmFP4Experts(cfg, qc)
    ex._dsv41_target_m32 = k_miss
    hidden = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
    aq, sfa = per_token_group_quant_fp8(hidden, 128, use_ue8m0=True)
    router = torch.softmax(torch.randn(m, 6, device="cuda"), -1)
    ids = torch.stack([torch.randperm(E, device="cuda")[:6] for _ in range(m)]).int()
    sizes = ex.workspace_shapes(m, 576, H, 6, E, E, None, MoEActivation.SILU)
    x, y, out = (torch.empty(s, dtype=torch.bfloat16, device="cuda") for s in sizes)

    def run_miss():
        ex.apply(output=out, hidden_states=aq, w1=w1, w2=w2, topk_weights=router, topk_ids=ids,
                 activation=MoEActivation.SILU, global_num_experts=E, expert_map=None, a1q_scale=sfa,
                 a2_scale=None, workspace13=x, workspace2=y, expert_tokens_meta=None,
                 apply_router_weight_on_input=False)
        return out

    # Reference first (eager, live arm): every kernel module of this size is loaded before the capture below,
    # so the capture only exercises the missing-tail path (k_miss itself never runs eagerly before it).
    ex_ref = DeepGemmFP4Experts(cfg, qc)
    ex_ref._dsv41_target_m32 = K["live"]
    xr, yr, outr = (torch.empty(s_, dtype=torch.bfloat16, device="cuda") for s_ in sizes)
    ex_ref.apply(output=outr, hidden_states=aq, w1=w1, w2=w2, topk_weights=router, topk_ids=ids,
                 activation=MoEActivation.SILU, global_num_experts=E, expert_map=None, a1q_scale=sfa,
                 a2_scale=None, workspace13=xr, workspace2=yr, expert_tokens_meta=None,
                 apply_router_weight_on_input=False)
    torch.cuda.synchronize()
    mem_before_capture = torch.cuda.memory_allocated()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        run_miss()
    poison(x, y, out)
    g.replay()
    torch.cuda.synchronize()
    got_graph = out.clone()
    misses, entries_after_capture = fc_miss.capture_misses, len(fc_miss.entries)
    tail_alloc_in_capture = fc_miss.tail_bytes
    run_miss()                                      # eager: builds the tail, runs compact
    torch.cuda.synchronize()
    ok = (beq(got_graph, outr) and beq(out, outr) and misses >= 1 and entries_after_capture == 0
          and tail_alloc_in_capture == 0 and len(fc_miss.entries) == 1 and k_miss.fc2c_calls >= 1)
    record(ok, event="capture_miss", layer=layer, misses=misses, entries_after_capture=entries_after_capture,
           tail_bytes_during_capture=tail_alloc_in_capture, compact_calls_after=k_miss.fc2c_calls,
           mem_before_capture=int(mem_before_capture))
    for e_ in list(fc_miss.entries):
        fc_miss.entries.pop(e_)
    del g, fc_miss, k_miss, ex, ex_ref

    # ---- FC2C=verify (poison_served as get_target_m32_kernel sets it for 'all'): direct calls, then full MoE
    ver = m32._D05Verifier("fc2c")
    fc_ver = m32.FC2Compact(fc2c_native, "verify", min2, ver)
    fc_ver.entries = fc_serv.entries
    fc_ver.poison_served = True
    k_ver = with_la(m32.TargetM32Kernel(base_native, d05_native, "1", None, fc_ver))
    ver_c = m32._D05Verifier("fc2c")                # control: same compare, served buffer NOT poisoned
    fc_ctl = m32.FC2Compact(fc2c_native, "verify", min2, ver_c)
    fc_ctl.entries = fc_serv.entries
    fc_ctl.poison_served = False
    k_ctl = with_la(m32.TargetM32Kernel(base_native, d05_native, "1", None, fc_ctl))
    live_rows = partial = 0
    served_ok = True
    for m in (2, 6, 24):
        hidden = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        aq, sfa = per_token_group_quant_fp8(hidden, 128, use_ue8m0=True)
        ids = torch.stack([torch.randperm(E, device="cuda")[:6] for _ in range(m)]).int()
        a2q, a2s, layout, msum = fc2_inputs(control, aq, sfa, ids)
        stale(a2q, a2s, layout >= 0, "nan")
        live_rows += int((layout >= 0).sum())
        partial += partial_groups(layout)
        o_ref = torch.empty(msum, H, dtype=torch.bfloat16, device="cuda")
        poison(o_ref, value=DS_SENTINEL)
        o_ver = torch.zeros(msum, H, dtype=torch.bfloat16, device="cuda")
        o_ctl = torch.zeros(msum, H, dtype=torch.bfloat16, device="cuda")
        for kern, o in ((K["live"], o_ref), (k_ver, o_ver), (k_ctl, o_ctl)):
            with kern.alignment_scope():
                kern.gemm((a2q, a2s), (w2i8, control.w2_scale), o, layout, recipe_a=(1, 128), recipe_b=(1, 32))
        torch.cuda.synchronize()
        lv = layout >= 0
        served_ok &= beq(o_ver[lv], o_ref[lv]) and beq(o_ctl[lv], o_ref[lv])
    torch.cuda.synchronize()
    c = ver.counters[:len(ver.index)].cpu().tolist()
    cc = ver_c.counters[:len(ver_c.index)].cpu().tolist()
    ok = (len(c) == 1 and c[0][0] == 3 and c[0][1] == live_rows and c[0][2] == 0 and c[0][3] == 0 and served_ok
          and len(cc) == 1 and cc[0][0] == 3 and ((cc[0][2] > 0) == (partial > 0)) and cc[0][3] == 0)
    record(ok, event="verify_mode", layer=layer, counters=c, control_counters=cc, live_rows=live_rows,
           partial_groups=partial, served_live_rows_eq_live=served_ok)
    # full MoE apply in verify mode (the poison lands on workspace2 = FC2 output, after FC1/SiLU consumed it)
    ex_v = DeepGemmFP4Experts(cfg, qc)
    ex_v._dsv41_target_m32 = k_ver
    ex_l = experts["live"]
    good = True
    for m in (2, 6, 24):
        hidden = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        aq, sfa = per_token_group_quant_fp8(hidden, 128, use_ue8m0=True)
        router = torch.softmax(torch.randn(m, 6, device="cuda"), -1)
        ids = torch.stack([torch.randperm(E, device="cuda")[:6] for _ in range(m)]).int()
        sizes = ex_v.workspace_shapes(m, 576, H, 6, E, E, None, MoEActivation.SILU)
        res = []
        for exx in (ex_l, ex_v):
            x, y, out = (torch.empty(s_, dtype=torch.bfloat16, device="cuda") for s_ in sizes)
            poison(x, y, out)
            exx.apply(output=out, hidden_states=aq, w1=w1, w2=w2, topk_weights=router, topk_ids=ids,
                      activation=MoEActivation.SILU, global_num_experts=E, expert_map=None, a1q_scale=sfa,
                      a2_scale=None, workspace13=x, workspace2=y, expert_tokens_meta=None,
                      apply_router_weight_on_input=False)
            res.append(out)
        torch.cuda.synchronize()
        good &= beq(res[0], res[1])
    c2 = ver.counters[:len(ver.index)].cpu().tolist()
    record(good and c2[0][0] == 6 and c2[0][2] == 0 and c2[0][3] == 0, event="verify_moe", layer=layer,
           counters=c2, out_eq_live=good)
    del k_ver, fc_ver, ver, k_ctl, fc_ctl, ver_c, ex_v

    # release this weight's tails before the next layer (the registry holds the weight alive)
    fc_serv.entries.pop(w2i8.data_ptr(), None)
    del w1, s1, w2, s2, w2i8, experts, control, tail, descs
    torch.cuda.empty_cache()

# ---- gating: M=97 (128-row path) reaches neither D05 nor compact; M=1 never compact under the serving gate
w1, s1, w2, s2 = weights(None)
qc = make_mxfp4_moe_quant_config(Mxfp4MoeBackend.DEEPGEMM_MXFP4, sf(s1, 2 * I, H), sf(s2, H, P), swiglu_limit=10.)
for m, want_d05, want_fc2c in ((97, 0, 0), (1, 1, 0), (2, 1, 1)):
    kern = K["fc2c"]
    b_calls, b_d05, b_fc = kern.calls, kern.d05_calls, kern.fc2c_calls
    ex = DeepGemmFP4Experts(cfg, qc)
    ex._dsv41_target_m32 = kern
    hidden = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
    aq, sfa = per_token_group_quant_fp8(hidden, 128, use_ue8m0=True)
    ids = torch.stack([torch.randperm(E, device="cuda")[:6] for _ in range(m)]).int()
    sizes = ex.workspace_shapes(m, 576, H, 6, E, E, None, MoEActivation.SILU)
    x, y, out = (torch.empty(s_, dtype=torch.bfloat16, device="cuda") for s_ in sizes)
    ex.apply(output=out, hidden_states=aq, w1=w1, w2=w2, topk_weights=torch.softmax(torch.randn(m, 6, device="cuda"), -1),
             topk_ids=ids, activation=MoEActivation.SILU, global_num_experts=E, expert_map=None, a1q_scale=sfa,
             a2_scale=None, workspace13=x, workspace2=y, expert_tokens_meta=None, apply_router_weight_on_input=False)
    torch.cuda.synchronize()
    got_d05, got_fc = kern.d05_calls - b_d05, kern.fc2c_calls - b_fc
    record(got_d05 == want_d05 and got_fc == want_fc2c and bool(torch.isfinite(out).all()), event="gating", m=m,
           m32_calls=kern.calls - b_calls, d05_eligible_calls=got_d05, compact_calls=got_fc)

# ---- engaged: the compact kernel ran from the PRE-JIT entry (cache hit, nothing compiled)
after = {k: cache_entries(JT / k) for k in CACHE_BEFORE}
new = {k: sorted(set(after[k]) - set(CACHE_BEFORE[k])) for k in after}
committed = STAGE / "jit-test/fc2c-all/cache" / PREJIT[0] / ".committed"
hit = committed.stat().st_mtime >= T_START - 1
tmp_left = sorted(x.name for x in (JT / "fc2c-all/tmp").iterdir()) if (JT / "fc2c-all/tmp").is_dir() else []
record(not new["fc2c-all"] and hit and not tmp_left and K["fc2c"].fc2c_calls > 0 and K["fc2c_all"].fc2c_calls > 0,
       event="fc2c_engaged", prejit=PREJIT, new_entries=new, prejit_hit=hit, tmp_left=tmp_left,
       calls={k: K[k].fc2c_calls for k in K})
emit(event="summary", status="PASS" if failures == 0 else "FAIL", cases=cases, failures=failures,
     layers=layer_list, tokens=tokens, tp_rank=a.tp_rank, host=os.uname().nodename, at=time.time())
sys.exit(1 if failures else 0)
