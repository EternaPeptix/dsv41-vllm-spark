#!/usr/bin/env python3
"""r6-step device test (single GPU node, serving stopped). Bitwise gates for C3 (COLEMB, ENGRAM_PACK) and C4.

Run on a STOPPED node (device window), in the mandatory environment:

    source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
    cd ~/climb-scratch/r6-step && python3 device_test.py --source kit --model ~/models/dsv41/official-fp8

    --source kit|installed   dsv41_step.py from this kit folder (default) or the one installed in the venv
    --part all|colemb|pack|c4
    --quick                  fewer shapes (smoke run)
    --cpu-selftest           harness logic only, on CPU (no GPU, no vllm): checks that every mutant is caught

r8-fixb additions:
    --part det               the LONG-PROMPT DETERMINISM gate for COLEMB and PACK (det_gate.py): decode T 1..128 and
                             prefill T 129..16384 (chunked prefill up to --max-num-batched-tokens), adversarial timing,
                             live Engram prefetch methods, pass-through at prefill sizes, 6 must-fail mutants.
    The environment rule (../common/prodenv.py): venv + CUDA_HOME + nvcc/ptxas, the production launcher (--expect sha8)
    env mirrored into this process (DSV41_GLUE_RQPAD=1 DSV41_GLUE_RPSPLIT=split asserted), and its must-fail controls.
    --source installed also asserts the venv's dsv41_step.py IS this kit's r8 module and engram.py / dsv41_r4_small.py
    carry the r6-step patch (the serving op under test).

Exit: 0 = every gate PASS and every mutant caught at every gated shape; 1 = a gate failed or a mutant passed;
2 = environment assertion failed (wrong venv / CUDA / the serving op is not the one under test). ~4-8 min.

Gates
  E0  environment: venv vllm032, CUDA_HOME, nvcc on PATH, GPU sm_12x, the live vllm modules (not this kit's
      copies) for the serving ops: torch.ops._C.vocab_parallel_embedding (fused CUDA kernel, not the torch
      fallback), FlashInfer mm_mxfp8, the live dsv41_r4_small.ColSplit; DSV41_DENSE_TACTIC and
      VLLM_DISABLED_KERNELS unset (the launcher sets neither, so the kernel choice and mm_mxfp8's tactic are the
      serving ones).
      E0 serving linears (fix r6-review blockers 1+2): mtp.0.main_proj and layers.{1,14}.engram.wkv (all 8
      ColumnParallel rank slices) are built through the LIVE quant-method lifecycle from the checkpoint:
      DeepseekV4FP8Config(checkpoint quantization_config).get_quant_method -> ModelOptLinearMethod(kMxfp8Static,
      scale block 32x32); the live scale loader KMxfp8Static.get_scale_weight_loader (e8m0 [N/32, K/32] ->
      uint8 repeat_interleave(32, 0) -> [N, K/32]); ColumnParallelLinear's narrow on output_dim 0 for a rank
      slice; the live select_linear_kernel; the live ModelOptLinearMethod.process_weights_after_loading (the
      swizzled 1-D serving scale). Gates: the method/kernel are ModelOptLinearMethod /
      FlashInferCutlassMxfp8LinearKernel (the serving choice on sm_12x), ColumnParallelLinear.forward and
      ReplicatedLinear.forward call exactly quant_method.apply(self, x, bias) (bias None here), and
      concat_r ColSplit(main_proj, r, 8).local(x) == main_proj.quant_method.apply(x) bitwise (so the C4 GEMM is the
      serving GEMM), with a rank-order mutant.
  C3a COLEMB, full table: embed.weight (129280 x 5120) from the checkpoint. For every rank r: the live fused
      kernel's partial for ALL ids (owner rows must equal W bitwise, other rows +-0 only), the AR is emulated
      with a Triton kernel issuing the same instruction kinds as ar_kernel_pl (fp32 FADD from a +0.0 seed in rank
      order, cvt.rn.bf16 = F2FP) for all 129280 x 5120 elements; the kit's column table (canon map) through the
      kit's Triton lookup + column concatenation must equal it bitwise; then replays at every served row count T
      with random, hostile (planted -0 row, NaN payloads, denormals, inf, max id) and invalid ids (-1, V, V+7).
      Mutants that must FAIL, at EVERY served T (T = 1 included): table without the -0/NaN map (the -0 row
      planted at index 0, the NaN row at index 1), lookup shifted by one column shard (an ordinary row at index 0),
      invalid id not zeroed (-1 at index 0, V+7 at index 1); each mutant's id vector also passes the real gate.
      (The real 8-rank rdma AR cannot run on one node: the serving-time builder re-proves the map on the real AR at
      every startup, owner rows x all 5120 columns on every rank, and refuses on any mismatch.)
  C3b ENGRAM_PACK: packed rows gather and packed wkv gather vs the stock per-layer gathers (8 ranks emulated on one
      GPU, hostile bit patterns). Mutants: halves swapped, rank order reversed (every T).
      PACK=all GEMM (fix r6-review blocker 2): the kit's own _wrap_wkv/wkv_first/wkv_second on the E0 serving
      wkv layers of layers 1 and 14, for every rank and every served T: layer 14's GEMM runs inside layer 1's
      wkv forward (interleaved), then layers 2..13 stand-in GEMMs, then layer 14 takes the stored kv; the result
      must equal the stock order (layer 1 GEMM, layers 2..13, layer 14 GEMM, each gathered) bitwise for both
      layers. Mutants (every T): layer 14 fed layer 1's rows, layer 14's GEMM run with layer 1's weights.
  C4  main_proj: the E0 serving main_proj (serving-layout scale), the live ColSplit for each rank slice, the kit's
      _Launch / c4_take with a real side stream and events, a DRAM-heavy screen stand-in and refine stand-in on
      main; hoisted y == stock local GEMM on main, bitwise, at every served M. Mutants at EVERY served M and rank:
      dropped join (side stream delayed), side stream not waiting for the taps' producer, a tap written on main
      after the side stream read it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    return ok


def mutant(name: str, caught: bool, detail: str = "") -> bool:
    RESULTS.append((f"mutant {name}", bool(caught), detail))
    print(f"{'CAUGHT' if caught else 'NOT-CAUGHT'}  mutant {name}  {detail}", flush=True)
    return caught


def load_step(source: str):
    if source == "installed":
        from vllm.models.deepseek_v41.nvidia import dsv41_step as m

        return m
    p = os.path.join(HERE, "files", "vllm", "models", "deepseek_v41", "nvidia", "dsv41_step.py")
    spec = importlib.util.spec_from_file_location("dsv41_step_kit", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ E0
KIT_MODULE = os.path.join(HERE, "files", "vllm", "models", "deepseek_v41", "nvidia", "dsv41_step.py")
ENGRAM_PATCHED = {"34cf684e87b650f7a7d28b66649bf32b8f6620502297453ee3e0ac621a004077",
                  "e99f623d3c461fa6d82002750f573dfac2a9f13ffc1e5b5341d090d9c322a148"}
R4_PATCHED = {"d9bf5d596c261771379d1ce5c8fb505cd3b5edff19540f7d8ccf1973fa845e74",
              "ebe73cdb57f06670370fb539ce77a836eadce74f51da29c1a15767086990247c"}


def _sha(p):
    import hashlib

    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def env_rule(args) -> bool:
    sys.path.insert(0, os.path.join(HERE, "..", "common"))
    import prodenv

    ok = True
    for name, good, det in prodenv.env_rule():
        ok &= gate(f"E0r {name}", good, str(det))
    _, mch = prodenv.mirror_launcher(args.launcher, args.expect)
    for name, good, det in mch:
        ok &= gate(f"E0r {name}", good, str(det))
    for name, caught, det in prodenv.selftest_mutants(args.launcher, args.expect):
        ok &= mutant(f"E0r {name}", caught, str(det))
    flags = [k for k in ("DSV41_STEP_COLEMB", "DSV41_STEP_ENGRAM_PACK", "DSV41_STEP_C4")
             if os.environ.get(k, "").strip().lower() not in ("", "0", "off", "false", "no")]
    ok &= gate("E0r r6-step flags unset in this process (the kit module is driven directly)", not flags, str(flags))
    return ok


def serving_op_check(args) -> bool:
    """--source installed: the module under test must be the r8 module installed in the venv, wired by r6-step's patch."""
    if args.source != "installed":
        return gate("E0s --source kit: the kit's r8 module is tested directly", os.path.exists(KIT_MODULE), KIT_MODULE)
    import vllm

    v = os.path.dirname(vllm.__file__)
    mod = os.path.join(v, "models/deepseek_v41/nvidia/dsv41_step.py")
    ok = gate("E0s installed dsv41_step.py == this kit's r8 module", os.path.exists(mod) and _sha(mod) == _sha(KIT_MODULE),
              _sha(mod)[:16] if os.path.exists(mod) else "absent")
    ok &= gate("E0s engram.py carries the r6-step C3 patch (34cf684e / e99f623d)",
               _sha(os.path.join(v, "models/deepseek_v41/common/engram.py")) in ENGRAM_PATCHED)
    ok &= gate("E0s dsv41_r4_small.py carries the r6-step patch (d9bf5d59 / composed ebe73cdb)",
               _sha(os.path.join(v, "models/deepseek_v41/nvidia/dsv41_r4_small.py")) in R4_PATCHED)
    return ok


def env_check() -> bool:
    import torch

    ok = True
    pre = sys.prefix
    ok &= gate("E0 venv", pre.endswith("venvs/vllm032"), pre)
    ok &= gate("E0 CUDA_HOME", os.environ.get("CUDA_HOME") == "/usr/local/cuda", str(os.environ.get("CUDA_HOME")))
    ok &= gate("E0 nvcc on PATH", "/usr/local/cuda/bin" in os.environ.get("PATH", "").split(":"))
    ok &= gate("E0 cuda", torch.cuda.is_available())
    if not ok:
        return False
    cap = torch.cuda.get_device_capability()
    ok &= gate("E0 sm_12x", cap[0] == 12, str(cap))
    import vllm
    import vllm._custom_ops as ops

    ok &= gate("E0 vllm from venv", vllm.__file__.startswith(pre), vllm.__file__)
    import inspect

    src = inspect.getsource(ops.vocab_parallel_embedding)
    has = hasattr(torch.ops._C, "vocab_parallel_embedding")
    ok &= gate("E0 fused vocab_parallel_embedding CUDA op", has and "torch.ops._C.vocab_parallel_embedding" in src)
    from vllm.utils import flashinfer as vfi

    ok &= gate("E0 FlashInfer mm_mxfp8", hasattr(vfi, "mm_mxfp8"))
    ok &= gate("E0 DSV41_DENSE_TACTIC unset (serving tactic path)",
               os.environ.get("DSV41_DENSE_TACTIC", "0").strip() in ("", "0"))
    from vllm.models.deepseek_v41.nvidia import dsv41_r4_small as r4

    ok &= gate("E0 live dsv41_r4_small", r4.__file__.startswith(pre) and hasattr(r4, "ColSplit"), r4.__file__)
    ok &= gate("E0 VLLM_DISABLED_KERNELS unset (serving kernel choice)",
               os.environ.get("VLLM_DISABLED_KERNELS", "").strip() == "")
    return ok


# ------------------------------------------------------------------------------------------------ E0 serving linears
SERVING_MXFP8_KERNEL = "FlashInferCutlassMxfp8LinearKernel"
LAYERS: dict = {}  # "main_proj" -> layer; "wkv1" / "wkv14" -> [8 rank-slice layers]
_CK: dict = {}


def serving_linear(model_dir: str, prefix: str, column_rows=None, dev="cuda"):
    """One checkpoint linear through the LIVE quant-method lifecycle (what model loading does for it on one rank):
    DeepseekV4FP8Config.get_quant_method -> ModelOptLinearMethod; the live MXFP8 scale loader (32x32 e8m0 block
    scale -> uint8 repeat_interleave(32, 0)); ColumnParallelLinear's load narrow (output_dim 0, both params) when
    column_rows is a slice; the live kernel selection; the live process_weights_after_loading. Returns the layer;
    layer.quant_method.apply(layer, x, None) is the serving GEMM (linear.py forward with bias None)."""
    import torch
    from torch.nn import Parameter
    from vllm.model_executor.layers.fusion.quant_activation import expose_input_quant_key
    from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
    from vllm.model_executor.layers.quantization import modelopt as mo
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

    qc = json.load(open(os.path.join(model_dir, "config.json")))["quantization_config"]
    cfg = DeepseekV4FP8Config.from_config(dict(qc))
    cls = ColumnParallelLinear if column_rows is not None else ReplicatedLinear
    lyr = cls.__new__(cls)  # a real LinearBase instance for get_quant_method; no TP group needed (no __init__)
    torch.nn.Module.__init__(lyr)
    method = cfg.get_quant_method(lyr, prefix)
    if type(method).__name__ != "ModelOptLinearMethod":
        raise RuntimeError(f"{prefix}: quant method is {type(method).__name__}, not ModelOptLinearMethod")
    if prefix not in _CK:  # one checkpoint read per tensor (the 8 rank slices share it)
        _CK.clear()
        _CK[prefix] = (ckpt_tensor(model_dir, prefix + ".weight"), ckpt_tensor(model_dir, prefix + ".scale"))
    W, S = _CK[prefix]
    srep = mo.KMxfp8Static.get_scale_weight_loader(lambda p, lw, *a, **k: lw, method.ctx)(None, S)
    N, K = W.shape
    if tuple(srep.shape) != (N, K // mo.MXFP8_BLOCK_SIZE) or srep.dtype != mo.MXFP8_SCALE_DTYPE:
        raise RuntimeError(f"{prefix}: loaded scale {tuple(srep.shape)} {srep.dtype} is not [N, K/32] "
                           f"{mo.MXFP8_SCALE_DTYPE} (raw {tuple(S.shape)} {S.dtype})")
    if column_rows is not None:
        W, srep = W[column_rows], srep[column_rows]
    lyr.weight = Parameter(W.contiguous().to(dev), requires_grad=False)
    lyr.weight_scale = Parameter(srep.contiguous().to(dev), requires_grad=False)
    lyr.register_parameter("bias", None)
    lyr.skip_bias_add = False
    method.input_dtype = method.out_dtype = torch.bfloat16  # model dtype (only the fp8 family reads it)
    rt = mo.RuntimeDtypes(torch.bfloat16, torch.bfloat16, None)
    method.kernel = mo.select_linear_kernel(method.spec, lyr, rt, weight_shape=method.fmt.kernel_weight_shape(lyr))
    expose_input_quant_key(lyr, method.kernel)
    method.process_weights_after_loading(lyr)
    lyr.quant_method = method
    lyr._raw = (tuple(S.shape), str(S.dtype))
    return lyr


def env_check_ops(args) -> bool:
    """E0 (second half): build the serving linears and prove the test GEMMs are the serving GEMMs."""
    import inspect

    import torch
    from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
    from vllm.model_executor.layers.quantization import modelopt as mo
    from vllm.models.deepseek_v41 import quant_config as qcm
    from vllm.models.deepseek_v41.nvidia import dsv41_r4_small as r4

    ok = True
    pre = sys.prefix
    ok &= gate("E0 live deepseek_v41.quant_config / modelopt", qcm.__file__.startswith(pre)
               and mo.__file__.startswith(pre), qcm.__file__)
    ok &= gate("E0 ColumnParallelLinear.forward GEMM == quant_method.apply(self, input_, bias)",
               "self.quant_method.apply(self, input_, bias)" in inspect.getsource(ColumnParallelLinear.forward))
    ok &= gate("E0 ReplicatedLinear.forward GEMM == quant_method.apply(self, x, bias)",
               "self.quant_method.apply(self, x, bias)" in inspect.getsource(ReplicatedLinear.forward))
    dev = torch.device("cuda")
    try:
        mp = serving_linear(args.model, "mtp.0.main_proj", None, dev)
        tp = 8
        wkv = {}
        for li in (1, 14):
            n = 25600 // tp
            wkv[li] = [serving_linear(args.model, f"layers.{li}.engram.wkv", slice(r * n, (r + 1) * n), dev)
                       for r in range(tp)]
    except Exception as e:  # noqa: BLE001
        gate("E0 serving linears built through the live quant-method lifecycle", False, repr(e))
        return False
    for name, L in [("mtp.0.main_proj", mp)] + [(f"layers.{li}.engram.wkv rank {r}", wkv[li][r])
                                                for li in (1, 14) for r in range(tp)]:
        qm = L.quant_method
        kn = type(qm.kernel).__name__
        good = (qm.spec.weight == mo.kMxfp8Static and qm.ctx.scale_block_size == (32, 32)
                and type(qm.fmt) is mo.FormatScheme and kn == SERVING_MXFP8_KERNEL
                and type(qm.kernel).apply_weights.__module__.startswith("vllm.model_executor.kernels.linear.mxfp8")
                and L.weight_scale.dim() == 1)
        if not good or name in ("mtp.0.main_proj", "layers.1.engram.wkv rank 0", "layers.14.engram.wkv rank 7"):
            ok &= gate(f"E0 {name}: ModelOptLinearMethod(kMxfp8Static, 32x32) -> {SERVING_MXFP8_KERNEL}, "
                       f"swizzled serving scale", good,
                       f"kernel {kn}, weight {tuple(L.weight.shape)}, scale {tuple(L.weight_scale.shape)} "
                       f"(raw {L._raw[0]} {L._raw[1]})")
    if not ok:
        return False
    # ColSplit (the r4 split C4 hoists) on the serving-layout layer == the stock replicated forward, bitwise
    try:
        splits = [r4.ColSplit(mp, r, tp) for r in range(tp)]
    except Exception as e:  # noqa: BLE001
        return gate("E0 live ColSplit accepts the serving main_proj", False, repr(e))
    for M in (1, 7, 32):
        x = hostile_bf16((M, 15360), 4242 + M, dev).nan_to_num(0.0, 4.0, -4.0)
        stock = mp.quant_method.apply(mp, x, None)
        cols = [s.local(x)[0] for s in splits]
        good, d = beq(torch.cat(cols, -1), stock)
        ok &= gate(f"E0 concat_r ColSplit.local == stock ReplicatedLinear main_proj GEMM M={M}", good, d)
        mutant(f"E0 ColSplit rank order reversed M={M}", not beq(torch.cat(cols[::-1], -1), stock)[0])
    LAYERS["main_proj"], LAYERS["wkv1"], LAYERS["wkv14"] = mp, wkv[1], wkv[14]
    return ok


# ------------------------------------------------------------------------------------------------ helpers
def ckpt_tensor(model_dir: str, name: str):
    from safetensors import safe_open

    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(model_dir, idx[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def ar_emulator():
    """Triton: out = bf16_rn(((0.f + x0) + x1) + ... + x7): bf16 -> f32 by a 16-bit shift (exact, as the live kernel's
    integer widening), fp32 FADD without .FTZ from a +0.0 seed in rank order, cvt.rn.bf16.f32 (F2FP.BF16.F32.PACK_AB,
    the live kernel's pack). tools/compile_check.py asserts these SASS kinds for sm_121."""
    import triton
    import triton.language as tl

    @triton.jit
    def _ar8(X, OUT, numel, stride_r, BLOCK: tl.constexpr):
        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = o < numel
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for r in tl.static_range(8):
            h = tl.load(X + r * stride_r + o, mask=m, other=0)  # raw bf16 bits (int16)
            acc = acc + ((h.to(tl.int32) & 0xFFFF) << 16).to(tl.float32, bitcast=True)  # exact widen, no FP op
        tl.store(OUT + o, acc.to(tl.bfloat16), mask=m)

    def run(parts):  # parts [8, ...] bf16
        import torch

        out = torch.empty(parts.shape[1:], dtype=torch.bfloat16, device=parts.device)
        n = out.numel()
        _ar8[(triton.cdiv(n, 4096),)](parts.view(torch.int16), out, n, parts[0].numel(), BLOCK=4096)
        return out

    return run


def bits(t):
    import torch

    return t.contiguous().view(torch.int16)


def beq(a, b) -> tuple[bool, str]:
    import torch

    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    d = bits(a) != bits(b)
    if not bool(d.any()):
        return True, ""
    i = int(torch.nonzero(d.flatten())[0])
    return False, f"first mismatch at flat index {i} of {d.numel()} ({int(d.sum())} differ)"


SERVED_T = list(range(1, 17)) + [20, 24, 25, 30, 32, 40, 48, 64, 96, 128]
SERVED_T_QUICK = [1, 2, 3, 4, 5, 6, 8, 10, 15, 30]


# ------------------------------------------------------------------------------------------------ C3a
HOSTILE_NEG0_ROW = 7  # part_colemb plants a whole -0 row here (and a NaN-payload row at per + 3)
PLAIN_ROW = 12345  # an ordinary row (never planted; < per): its column shards differ


def c3a_mutant_ids(T: int, V: int, per: int, g):
    """Per C3a mutant, T ids (int32, CPU) whose index 0 is the row that mutant must change, so the mutant is
    catchable at every T including T = 1: map -> the -0 row (then the NaN row), shard -> an ordinary row,
    invalid -> -1 (then V + 7). The rest are random valid ids."""
    import torch

    base = torch.randint(0, V, (T,), generator=g, dtype=torch.int32)
    out = {k: base.clone() for k in ("map", "shard", "invalid")}
    out["map"][0] = HOSTILE_NEG0_ROW
    out["shard"][0] = PLAIN_ROW
    out["invalid"][0] = -1
    if T > 1:
        out["map"][1] = per + 3
        out["invalid"][1] = V + 7
    return out


def part_colemb(m, args) -> None:
    import torch
    import vllm._custom_ops as ops

    dev = torch.device("cuda")
    W = ckpt_tensor(args.model, "embed.weight").to(torch.bfloat16)
    V, D = W.shape
    tp, n = 8, D // 8
    per = V // tp
    gate("C3a table shape", V == 129280 and D == 5120 and V % tp == 0, f"{V} x {D}")
    sc = m.special_counts(W)
    print(f"      embed.weight specials: {sc}")
    W = W.to(dev)
    # Hostile copy: plant -0 / NaN payload / denormal / inf rows so the map is exercised whatever the real table has.
    Wh = W.clone()
    hb = Wh.view(torch.int16)
    hb[HOSTILE_NEG0_ROW] = -32768
    hb[per + 3, ::2] = 0x7FC1
    hb[per + 3, 1::2] = -16  # 0xFFF0: NaN with payload, sign set
    hb[2 * per + 1] = 0x0001
    hb[3 * per + 2, :8] = 0x7F80
    hb[V - 1, :] = -32768
    emu = ar_emulator()
    for label, Wt in (("real", W), ("hostile", Wh)):
        # live path per rank (fused kernel) for every id, chunked; emulated AR
        ref = torch.empty((V, D), dtype=torch.bfloat16, device=dev)
        owner_ok, zero_ok = True, True
        CH = 4096
        for s in range(0, V, CH):
            e = min(V, s + CH)
            ids = torch.arange(s, e, dtype=torch.int32, device=dev)
            parts = []
            for r in range(tp):
                p = ops.vocab_parallel_embedding(ids, Wt[r * per:(r + 1) * per].contiguous(), r * per,
                                                 (r + 1) * per, 0, V, V)
                own = (ids >= r * per) & (ids < (r + 1) * per)
                if bool(own.any()):
                    owner_ok &= bool(torch.equal(bits(p[own]), bits(Wt[s:e][own])))
                if bool((~own).any()):
                    zero_ok &= bool(((bits(p[~own]) & 0x7FFF) == 0).all())
                parts.append(p)
            ref[s:e] = emu(torch.stack(parts))
        gate(f"C3a[{label}] live fused kernel: owner rows == W bitwise", owner_ok)
        gate(f"C3a[{label}] live fused kernel: non-owner rows are +-0", zero_ok)
        canon = m.canon_bits(Wt)
        ok, d = beq(canon.view(torch.bfloat16), ref)
        gate(f"C3a[{label}] FULL TABLE canon(W) == emulated live AR, {V} x {D}", ok, d)
        # the kit's per-rank column tables + Triton lookup + column concat
        cts = [m.ColTable(canon.view(torch.bfloat16)[:, r * n:(r + 1) * n].contiguous(), r, tp, n, V, D, 128)
               for r in range(tp)]
        look = m._colemb_lookup_fn()
        allid = torch.arange(V, dtype=torch.int32, device=dev)
        full = torch.cat([torch.cat([look(allid[s:s + 8192], ct) for s in range(0, V, 8192)]) for ct in cts], -1)
        ok, d = beq(full, ref)
        gate(f"C3a[{label}] FULL TABLE kit lookup+concat == emulated live AR", ok, d)
        g = torch.Generator(device="cpu").manual_seed(1)
        raw = [m.ColTable(Wt[:, r * n:(r + 1) * n].contiguous(), r, tp, n, V, D, 128) for r in range(tp)]

        def live_ar(ids):  # the live fused kernel per rank + the emulated AR
            return emu(torch.stack([ops.vocab_parallel_embedding(ids, Wt[r * per:(r + 1) * per].contiguous(),
                                                                 r * per, (r + 1) * per, 0, V, V)
                                    for r in range(tp)]))

        for T in (SERVED_T_QUICK if args.quick else SERVED_T):
            ids = torch.randint(0, V, (T,), generator=g, dtype=torch.int32)
            ids[0] = HOSTILE_NEG0_ROW
            if T > 2:
                ids[1], ids[2] = per + 3, V - 1
            if T > 4:
                ids[3], ids[4] = -1, V + 7
            ids = ids.to(dev)
            got = torch.cat([look(ids, ct) for ct in cts], -1)
            ok, d = beq(got, live_ar(ids))
            gate(f"C3a[{label}] replay T={T}", ok, d)
            if label != "hostile":
                continue
            # mutants at EVERY served T: each gets an id vector whose index 0 is the row it must trip on
            mids = c3a_mutant_ids(T, V, per, g)
            muts = {
                "map": ("table without -0/NaN map", lambda i: torch.cat([look(i, ct) for ct in raw], -1)),
                "shard": ("column shard shifted",
                          lambda i: torch.cat([look(i, cts[(r + 1) % tp]) for r in range(tp)], -1)),
                "invalid": ("invalid id not zeroed",
                            lambda i: torch.cat([ct.table[i.long().clamp(0, V - 1)] for ct in cts], -1)),
            }
            for k, (what, fn) in muts.items():
                i = mids[k].to(dev)
                ref = live_ar(i)
                ok, d = beq(torch.cat([look(i, ct) for ct in cts], -1), ref)
                gate(f"C3a[hostile] replay T={T} ({k} ids)", ok, d)
                mutant(f"C3a {what} T={T}", not beq(fn(i), ref)[0])


# ------------------------------------------------------------------------------------------------ C3b
def hostile_bf16(shape, seed, dev):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g).to(torch.bfloat16)
    b = x.view(torch.int16).flatten()
    k = max(1, b.numel() // 50)
    pats = torch.tensor([-32768, 0x7FC1, -16, 0x0001, 0x7F80, -128, 0x7FFF], dtype=torch.int16)
    b[torch.randint(0, b.numel(), (k,), generator=g)] = pats[torch.randint(0, len(pats), (k,), generator=g)]
    return x.to(dev)


def part_pack(m, args) -> None:
    import torch

    dev = torch.device("cuda")
    tp, H, hd, nw = 8, 3, 256, 3200
    for T in (SERVED_T_QUICK if args.quick else SERVED_T):
        ra = [hostile_bf16((T, H, hd), 10 * T + r, dev) for r in range(tp)]
        rb = [hostile_bf16((T, H, hd), 1000 + 10 * T + r, dev) for r in range(tp)]
        me = T % tp

        class E:
            def __init__(self, rows):
                self.rows = rows

            def _ready_rows(self, k):
                return self.rows[:k]

        m._all_gather = lambda x, dim: torch.cat([torch.cat([ra[r], rb[r]], 1) for r in range(tp)], dim)
        o1, o2 = m.packed_rows_gather(E(ra[me]), E(rb[me]), T)
        s1, s2 = torch.cat(ra, 1), torch.cat(rb, 1)  # == stock tensor_model_parallel_all_gather(rows, dim=1)
        ok = beq(o1, s1)[0] and beq(o2, s2)[0]
        gate(f"C3b rows pack T={T}", ok)
        mutant(f"C3b rows halves swapped T={T}", not beq(o2, s1)[0])
        m._all_gather = lambda x, dim: torch.cat([torch.cat([ra[r], rb[r]], 1) for r in reversed(range(tp))], dim)
        r1, _ = m.packed_rows_gather(E(ra[me]), E(rb[me]), T)
        mutant(f"C3b rows rank order reversed T={T}", not beq(r1, s1)[0])
        y1 = [hostile_bf16((T, nw), 5000 + 10 * T + r, dev) for r in range(tp)]
        y2 = [hostile_bf16((T, nw), 9000 + 10 * T + r, dev) for r in range(tp)]
        m._all_gather = lambda x, dim: torch.cat([torch.cat([y1[r], y2[r]], -1) for r in range(tp)], dim)
        k1, k2 = m.packed_cols_gather(y1[me], y2[me])
        gate(f"C3b wkv pack T={T}", beq(k1, torch.cat(y1, -1))[0] and beq(k2, torch.cat(y2, -1))[0])
        mutant(f"C3b wkv halves swapped T={T}", not beq(k2, torch.cat(y1, -1))[0])
    part_pack_gemm(m, args)


def part_pack_gemm(m, args, dev=None) -> None:
    """PACK=all: the kit's own wkv wrappers on the E0 serving wkv layers (live quant_method.apply, serving-layout
    scale) of layers 1 and 14, all 8 rank slices: interleaved (kit) order == stock order, bitwise, every T.
    (dev="cpu" only from cpu_selftest, with stand-in layers, to check the harness plumbing and mutants.)"""
    import types

    import torch

    dev = torch.device(dev or "cuda")
    L1, L14, mp = LAYERS["wkv1"], LAYERS["wkv14"], LAYERS["main_proj"]
    tp, ncols, hd, nloc = 8, 24, 256, 3200
    K = ncols * hd

    def apply(L, x):  # == ColumnParallelLinear.forward's GEMM (bias None; E0 checked the call)
        return L.quant_method.apply(L, x, None)

    def others(T, seed):  # stand-ins for layers 2..13's GEMMs between the two Engram layers (same kernel)
        g = torch.Generator(device="cpu").manual_seed(seed)
        apply(mp, torch.randn(T, 15360, generator=g).to(dev, torch.bfloat16))
        for r, TT in ((2, max(1, T // 2)), (5, 64), (6, T)):
            apply(L1[r], torch.randn(TT, K, generator=g).to(dev, torch.bfloat16))
            apply(L14[r], torch.randn(TT, K, generator=g).to(dev, torch.bfloat16))

    def no_fallback(x):
        raise AssertionError("the kit took the stock wkv path (pairing refused)")

    def proxy(L):  # what _wrap_wkv touches on a ColumnParallelLinear (forward is rebound per test)
        return types.SimpleNamespace(quant_method=L.quant_method, weight=L.weight, weight_scale=L.weight_scale,
                                     bias=None, skip_bias_add=False, forward=no_fallback)

    def run_kit(me, T, x1, x2, rows_b, wb_me):
        """Rank `me` under PACK=all: layer 1's wkv forward runs both GEMMs and the packed gather; layers 2..13;
        layer 14's wkv forward returns the stored kv. rows_b / wb_me are what the kit uses for layer 14."""
        a = types.SimpleNamespace(wkv=proxy(L1[me]))
        b = types.SimpleNamespace(wkv=proxy(wb_me), embed_tokens=types.SimpleNamespace(n_hash_cols=ncols))
        m.PK.__init__()
        m._wrap_wkv(a, b)

        def ag(x, dim):  # the column all-gather: rank r contributes its [y1_r | y2_r]; r == me is the kit's operand
            assert dim == -1 and tuple(x.shape) == (T, 2 * nloc), (dim, tuple(x.shape))
            return torch.cat([x if r == me else torch.cat([apply(L1[r], x1), apply(L14[r], x2)], -1)
                              for r in range(tp)], -1)

        m._all_gather = ag
        m.PK.seq = 1
        m.PK.rows = (1, T, rows_b)  # what layer 1's packed embed stored for layer 14 this forward
        # r8-fixb operand identity: in serving, layer 1's wkv operand IS its packed embed output and layer 14's IS the
        # stored rows its embed returned; the harness passes x1 / x2 as those operands, so it records them as such
        m.PK.a_out = (1, int(x1.data_ptr()), (T, ncols, x1.shape[1] // ncols))
        kv1 = a.wkv.forward(x1)
        others(T, 1000 + T)
        m.PK.second_seq = 1  # layer 14's embed took this forward's stored rows
        m.PK.b_out = (1, int(x2.data_ptr()), (T, ncols, x2.shape[1] // ncols))
        kv2 = b.wkv.forward(x2)
        return kv1, kv2

    ranks = (0, 3, 7) if args.quick else range(tp)
    for T in (SERVED_T_QUICK if args.quick else SERVED_T):
        rows1 = hostile_bf16((T, ncols, hd), 300 + T, dev).nan_to_num(0.0, 1.0, -1.0)  # gathered rows, layer 1
        rows2 = hostile_bf16((T, ncols, hd), 700 + T, dev).nan_to_num(0.0, 1.0, -1.0)  # gathered rows, layer 14
        x1, x2 = rows1.flatten(-2), rows2.flatten(-2)  # the stock wkv operands: embed(hash_ids).flatten(-2)
        # stock order: layer 1's GEMM (every rank slice) + gather, layers 2..13, layer 14's GEMM + gather
        kv1_ref = torch.cat([apply(L1[r], x1) for r in range(tp)], -1)
        others(T, T)
        kv2_ref = torch.cat([apply(L14[r], x2) for r in range(tp)], -1)
        for me in ranks:
            kv1, kv2 = run_kit(me, T, x1, x2, rows2, L14[me])
            ok1, d1 = beq(kv1, kv1_ref)
            ok2, d2 = beq(kv2, kv2_ref)
            gate(f"C3b PACK=all rank {me} T={T}: layer-1 kv == stock", ok1, d1)
            gate(f"C3b PACK=all rank {me} T={T}: layer-14 kv (GEMM run at layer 1) == stock", ok2, d2)
        me = T % tp
        _, bad = run_kit(me, T, x1, x2, rows1, L14[me])
        mutant(f"C3b layer 14 fed layer 1's rows T={T}", not beq(bad, kv2_ref)[0])
        _, bad = run_kit(me, T, x1, x2, rows2, L1[me])
        mutant(f"C3b layer 14's GEMM with layer 1's weights T={T}", not beq(bad, kv2_ref)[0])
    m.PK.__init__()


# ------------------------------------------------------------------------------------------------ C4
def part_c4(m, args) -> None:
    import torch
    from vllm.models.deepseek_v41.nvidia import dsv41_r4_small as r4

    dev = torch.device("cuda")
    # The serving layer from E0: live quant-method lifecycle, serving-layout (swizzled 1-D) scale. The raw
    # checkpoint scale (e8m0 [160, 480], 32x32 blocks) is NOT what ColSplit sees in serving (r6-review blocker 1).
    lin = LAYERS["main_proj"]
    N, K = lin.weight.shape
    gate("C4 main_proj shape (serving layer)", (N, K) == (5120, 15360),
         f"{N} x {K} {lin.weight.dtype} scale {tuple(lin.weight_scale.shape)} {lin.weight_scale.dtype} "
         f"(raw {lin._raw[0]} {lin._raw[1]})")
    big = torch.empty(96 * 2 ** 20 // 2, dtype=torch.bfloat16, device=dev)  # screen stand-in: ~85 MB read
    big2 = torch.empty_like(big)
    m._cu = torch.cuda
    for rank in ((0, 3, 7) if args.quick else range(8)):
        st = r4.ColSplit(lin, rank, 8)

        class Split:  # the live ColSplit's local GEMM; the all-gather (a byte copy) is identity on one node
            checked = None

            def local(self, x):
                return st.local(x)

            def gather(self, y, shape):
                return y

        sp = Split()
        for M in (SERVED_T_QUICK if args.quick else SERVED_T):
            for it in range(3):
                aux = [hostile_bf16((M, 5120), 31 * M + 7 * it + j + rank, dev).nan_to_num(0.0, 4.0, -4.0)
                       for j in range(3)]
                m.C4S.__init__()
                big2.copy_(big)  # "screen"
                m._Launch(sp, M, aux).after_screen()
                big.copy_(big2)  # "refine + sampling glue" on main while the side stream runs
                x = torch.cat(aux, -1)
                y = m.c4_take(sp, x, M)
                ref, _ = st.local(x)  # the stock path at combine time
                ok, d = beq(y, ref)
                if it == 0 or not ok:
                    gate(f"C4 rank {rank} M={M} hoisted == stock local GEMM", ok, d)
            _c4_mutants(m, sp, st, M, rank, big, big2)  # every served M, every rank
        torch.cuda.synchronize()
    # timing (information only): main-stream span of the chain, stock vs hoisted
    st = r4.ColSplit(lin, 0, 8)
    M = 4
    aux = [torch.randn(M, 5120, device=dev).to(torch.bfloat16) for _ in range(3)]
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def span(fn, reps=200):
        ts = []
        for _ in range(reps):
            big2.copy_(big)
            fn(prep=True)
            ev0.record()
            fn(prep=False)
            ev1.record()
            torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    class Split0:
        checked = None

        def local(self, x):
            return st.local(x)

        def gather(self, y, shape):
            return y

    sp0 = Split0()

    def stock(prep):
        if not prep:
            st.local(torch.cat(aux, -1))

    def hoisted(prep):
        if prep:
            m.C4S.__init__()
            m._Launch(sp0, M, aux).after_screen()
            big.copy_(big2)
        else:
            m.c4_take(sp0, torch.cat(aux, -1), M)

    print(f"      timing M=4 (median of 200, us, main stream from the cat to the gather operand): "
          f"stock {span(stock):.1f}  hoisted {span(hoisted):.1f}")


def _c4_mutants(m, sp, st, M, rank, big, big2) -> None:
    import torch

    dev = big.device
    # (1) dropped join: the side stream is delayed, the consumer reads y without the event wait
    aux = [torch.randn(M, 5120, device=dev).to(torch.bfloat16) for _ in range(3)]
    m.C4S.__init__()
    m.C4S.stream, m.C4S.ev_screen, m.C4S.ev_done = torch.cuda.Stream(), torch.cuda.Event(), torch.cuda.Event()
    with torch.cuda.stream(m.C4S.stream):
        torch.cuda._sleep(20_000_000)
    m._Launch(sp, M, aux).after_screen()
    p, m.C4S.pending = m.C4S.pending, None
    early = p.y.clone()  # main reads y with no wait
    ref, _ = st.local(torch.cat(aux, -1))
    torch.cuda.synchronize()
    mutant(f"C4 dropped join rank {rank} M={M}", not beq(early, ref)[0])
    # (2) side stream does not wait for the taps' producer (main writes aux after a delay)
    m.C4S.__init__()
    m.C4S.stream, m.C4S.ev_screen, m.C4S.ev_done = torch.cuda.Stream(), torch.cuda.Event(), torch.cuda.Event()
    m.C4S.stream.wait_event = lambda ev: None
    aux = [torch.zeros(M, 5120, device=dev, dtype=torch.bfloat16) for _ in range(3)]
    src = [torch.randn(M, 5120, device=dev).to(torch.bfloat16) for _ in range(3)]
    torch.cuda._sleep(20_000_000)
    for a, s in zip(aux, src):
        a.copy_(s)  # "target graph" writes the taps late on main
    m._Launch(sp, M, aux).after_screen()
    y = m.c4_take(sp, torch.cat(aux, -1), M)
    ref, _ = st.local(torch.cat(aux, -1))
    torch.cuda.synchronize()
    mutant(f"C4 no wait for the taps rank {rank} M={M}", not beq(y, ref)[0])
    # (3) a tap written on main after the side stream read it
    m.C4S.__init__()
    aux = [torch.randn(M, 5120, device=dev).to(torch.bfloat16) for _ in range(3)]
    m._Launch(sp, M, aux).after_screen()
    torch.cuda.synchronize()
    aux[1].add_(1.0)
    y = m.c4_take(sp, torch.cat(aux, -1), M)
    ref, _ = st.local(torch.cat(aux, -1))
    torch.cuda.synchronize()
    mutant(f"C4 tap written after the fork rank {rank} M={M}", not beq(y, ref)[0])
    m.C4S.__init__()


# ------------------------------------------------------------------------------------------------ CPU self-test
def cpu_selftest() -> int:
    """Harness logic on CPU: layout gates pass, every layout / map mutant is caught (no GPU kernels)."""
    import torch

    m = load_step("kit")
    W = torch.randn(64, 16).to(torch.bfloat16)
    W.view(torch.int16)[3] = -32768
    W.view(torch.int16)[9, 0] = 0x7FC1
    c = m.canon_bits(W)
    gate("cpu canon -0 row -> +0", int(c[3].abs().sum()) == 0)
    gate("cpu canon NaN -> 0x7FFF", int(c[9, 0]) == 0x7FFF)
    mutant("cpu table without map", not torch.equal(W.view(torch.int16), c))
    T, H, hd, tp = 5, 3, 4, 8
    ra = [hostile_bf16((T, H, hd), r, "cpu") for r in range(tp)]
    rb = [hostile_bf16((T, H, hd), 100 + r, "cpu") for r in range(tp)]

    class E:
        def __init__(self, rows):
            self.rows = rows

        def _ready_rows(self, k):
            return self.rows[:k]

    m._all_gather = lambda x, dim: torch.cat([torch.cat([ra[r], rb[r]], 1) for r in range(tp)], dim)
    o1, o2 = m.packed_rows_gather(E(ra[0]), E(rb[0]), T)
    gate("cpu rows pack", beq(o1, torch.cat(ra, 1))[0] and beq(o2, torch.cat(rb, 1))[0])
    mutant("cpu rows halves swapped", not beq(o2, torch.cat(ra, 1))[0])
    _cpu_c3a_mutant_ids(m)
    _cpu_pack_gemm(m)
    # r8-fixb: the det gate's PACK part and colemb pass-through on the latest-possible-scheduler simulation
    import det_gate as DG

    md = load_step("kit")
    DG.part_pack_det(md, DG.SimRT(torch), gate, mutant, quick=True)
    mc = load_step("kit")
    DG.part_colemb_det(mc, DG.SimRT(torch), gate, mutant, quick=True)
    return summary()


def _cpu_pack_gemm(m) -> None:
    """part_pack_gemm's plumbing on CPU: stand-in wkv layers whose quant_method.apply is a deterministic bf16 matmul
    (the real run uses the E0 serving layers). Gates must pass and both mutants must be caught at every T."""
    import types

    import torch

    class QM:
        @staticmethod
        def apply(layer, x, bias):
            assert bias is None
            return (x.float() @ layer.weight.float().t()).to(torch.bfloat16)

    def lay(seed, N, K):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return types.SimpleNamespace(quant_method=QM, weight=(torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16),
                                     weight_scale=None)

    LAYERS["wkv1"] = [lay(100 + r, 3200, 6144) for r in range(8)]
    LAYERS["wkv14"] = [lay(200 + r, 3200, 6144) for r in range(8)]
    LAYERS["main_proj"] = lay(300, 640, 15360)
    n0 = len(RESULTS)
    part_pack_gemm(m, types.SimpleNamespace(quick=True), dev="cpu")
    res = RESULTS[n0:]
    del RESULTS[n0:]
    g_ok = all(ok for name, ok, _ in res if not name.startswith("mutant"))
    mu = [ok for name, ok, _ in res if name.startswith("mutant")]
    ng = sum(1 for name, _, _ in res if not name.startswith("mutant"))
    gate(f"cpu C3b PACK=all harness (stand-in GEMM): {ng} kit-vs-stock gates pass", g_ok and ng > 0)
    mutant(f"cpu C3b PACK=all harness: {len(mu)} mutants (2 per T) all caught", len(mu) == 2 * len(SERVED_T_QUICK)
           and all(mu))
    LAYERS.clear()


def _cpu_c3a_mutant_ids(m) -> None:
    """The C3a mutant id vectors catch their mutant at EVERY served T (T = 1 included). CPU stand-ins: the reference
    row is canon(W) (proved equal to the AR map by tests/test_step.py's exhaustive test), +0 for invalid ids."""
    import torch

    tp, n = 8, 4
    V, D = 8 * 16384, tp * n
    per = V // tp
    assert PLAIN_ROW < per and HOSTILE_NEG0_ROW < per
    g0 = torch.Generator(device="cpu").manual_seed(3)
    W = torch.randn(V, D, generator=g0).to(torch.bfloat16)
    hb = W.view(torch.int16)
    hb[HOSTILE_NEG0_ROW] = -32768
    hb[per + 3, ::2] = 0x7FC1
    hb[per + 3, 1::2] = -16
    canon = m.canon_bits(W).view(torch.bfloat16)
    cts = [canon[:, r * n:(r + 1) * n] for r in range(tp)]
    raw = [W[:, r * n:(r + 1) * n] for r in range(tp)]

    def look(i, tab):  # the kit kernel's semantics: table[id] if 0 <= id < V else +0
        ok = (i >= 0) & (i < V)
        return torch.where(ok.unsqueeze(1), tab[i.long().clamp(0, V - 1)], torch.zeros((), dtype=tab.dtype))

    def ref(i):
        return torch.cat([look(i, ct) for ct in cts], -1)

    muts = {
        "map": lambda i: torch.cat([look(i, t) for t in raw], -1),
        "shard": lambda i: torch.cat([look(i, cts[(r + 1) % tp]) for r in range(tp)], -1),
        "invalid": lambda i: torch.cat([ct[i.long().clamp(0, V - 1)] for ct in cts], -1),
    }
    g = torch.Generator(device="cpu").manual_seed(1)
    good, caught, worst = True, {k: 0 for k in muts}, []
    for T in SERVED_T:
        mids = c3a_mutant_ids(T, V, per, g)
        for k, fn in muts.items():
            i = mids[k]
            good &= beq(torch.cat([look(i, ct) for ct in cts], -1), ref(i))[0]
            if not beq(fn(i), ref(i))[0]:
                caught[k] += 1
            else:
                worst.append((k, T))
    gate("cpu C3a mutant-id vectors: the correct lookup passes at every T", good)
    for k in muts:
        mutant(f"cpu C3a {k} mutant caught at all {len(SERVED_T)} served T", caught[k] == len(SERVED_T),
               f"missed at {[t for kk, t in worst if kk == k]}")


def summary() -> int:
    fails = [r for r in RESULTS if not r[1]]
    nm = sum(1 for r in RESULTS if r[0].startswith("mutant"))
    print(f"\nSUMMARY: {len(RESULTS)} checks ({nm} mutants), {len(fails)} failed")
    for f in fails:
        print(f"  FAILED: {f[0]} {f[2]}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("kit", "installed"), default="kit")
    ap.add_argument("--part", choices=("all", "colemb", "pack", "c4", "det"), default="all")
    ap.add_argument("--launcher", default=os.path.expanduser("~/dsv41-tp8.sh"))
    ap.add_argument("--expect", default=os.environ.get("FIXB_EXPECT", "4f3cc1c6"),
                    help="production launcher sha8 (pass the new one after any promotion)")
    ap.add_argument("--model", default=os.path.expanduser("~/models/dsv41/official-fp8"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cpu-selftest", action="store_true")
    args = ap.parse_args()
    if args.cpu_selftest:
        return cpu_selftest()
    if not (env_rule(args) & env_check() & serving_op_check(args)):
        print("ENVIRONMENT ASSERTION FAILED: not the serving environment / serving op; nothing was tested")
        return 2
    if not env_check_ops(args):
        print("SERVING-OP ASSERTION FAILED (E0 serving linears / ColSplit == stock): nothing else was tested")
        summary()
        return 2
    m = load_step(args.source)
    t0 = time.time()
    if args.part in ("all", "colemb"):
        part_colemb(m, args)
    if args.part in ("all", "pack"):
        part_pack(m, args)
    if args.part in ("all", "c4"):
        part_c4(m, args)
    if args.part in ("all", "det"):
        import torch
        import det_gate as DG
        import vllm.model_executor.layers.vocab_parallel_embedding as vpe

        DG.part_pack_det(m, DG.GpuRT(torch), gate, mutant, quick=args.quick)
        DG.part_colemb_det(m, DG.GpuRT(torch), gate, mutant, quick=args.quick, vpe_mod=vpe)
    print(f"elapsed {time.time() - t0:.0f} s")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
