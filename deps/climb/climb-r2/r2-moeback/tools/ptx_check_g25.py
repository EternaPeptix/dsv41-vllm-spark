#!/usr/bin/env python3
"""G25 compile check (no GPU, no kernel launch): lower the live _gated_gather and the kit _gated_gather_add for
sm_121 with Triton's own compiler, then compare the arithmetic instruction streams.

    CUDA_VISIBLE_DEVICES= nice -n 19 ~/venvs/vllm032/bin/python ptx_check_g25.py <kit_dir> <out_dir>

Checks:
  1. our lowering of the live _gated_gather reproduces the production Triton cache (same counts of fma.rn.f32x2,
     cvt.rn.bf16x2.f32, no .ftz anywhere): the compile setup matches production;
  2. _gated_gather_add contains the same accumulation (48 fma.rn.f32x2 per BLOCK_D=1024 / 64 threads, no ftz, no
     extra fp ops between the loads and the fmas) and its epilogue is EXACTLY one of the two lowerings proven
     bit-identical to torch's CUDAFunctor_add<bf16> by tests/emul_add_exhaustive.py, counted per element
     (rule in ptx_rule_g25.py, unit-tested by tests/test_ptx_rule.py):
       A: 16 cvt.rn.bf16x2.f32 in total (8 live + 8 for the sum) and 16 extra fp32 add elements, no bf16 add;
       B: 8 cvt.rn.bf16x2.f32 (live only) and 16 bf16 add elements (8 add.bf16x2), no extra fp32 add.
     The known-wrong lowering (shared folded into the fp32 accumulator before the single rounding) fails.
Writes both PTX files and a summary to <out_dir>. Exit 0 only when every check holds.
"""
import importlib.util
import os
import re
import sys
import types

kit, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(out, exist_ok=True)
# Never write into the production Triton cache (~/.triton/cache): use a private one under out_dir.
os.environ["TRITON_CACHE_DIR"] = os.path.join(out, "triton-cache")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

# vllm.triton_utils shim so the two modules import without pulling in vllm
pkg = types.ModuleType("vllm")
pkg.__path__ = []
tu = types.ModuleType("vllm.triton_utils")
tu.tl, tu.triton = tl, triton
sys.modules.setdefault("vllm", pkg)
sys.modules["vllm.triton_utils"] = tu


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


live = load("live_gather", os.path.expanduser("~/dsv41-fc2/fc2_status_gated_gather_trial.py"))
fold = load("kit_fold", os.path.join(kit, "files/vllm/model_executor/layers/fused_moe/runner/dsv41_moe_fold.py"))
target = GPUTarget("cuda", 121, 32)

base_sig = {"total_token_num": "i32", "input_tensor": "*bf16", "input_stride0": "i32", "input_stride1": "constexpr",
            "recv_topk_ids": "*i32", "ids_stride0": "i32", "ids_stride1": "constexpr",
            "recv_topk_weight": "*fp32", "weights_stride0": "i32", "weights_stride1": "constexpr",
            "input_index": "*i32", "index_stride0": "i32", "index_stride1": "constexpr",
            "output_tensor": "*bf16", "output_stride0": "i32", "output_stride1": "constexpr"}
tail_sig = {"step_status": "*i32", "topk_num": "constexpr", "BLOCK_D": "constexpr"}
consts = {"input_stride1": 1, "ids_stride1": 1, "weights_stride1": 1, "index_stride1": 1, "output_stride1": 1,
          "topk_num": 6, "BLOCK_D": 1024}


def compile_kernel(fn, sig, extra_consts):
    c = dict(consts, **extra_consts)
    names = list(sig)
    # 16-byte divisibility on every pointer and on the leading strides (what the runtime specializes on)
    div = [[i] for i, n in enumerate(names) if sig[n].startswith("*") or n.endswith("stride0")]
    attrs = {tuple(d): [["tt.divisibility", 16]] for d in div}
    try:
        src = ASTSource(fn=fn, signature=sig, constexprs={(names.index(k),): v for k, v in c.items()}, attrs=attrs)
    except TypeError:
        src = ASTSource(fn=fn, signature=sig, constexprs=c)
    return triton.compile(src, target=target, options={"num_warps": 2})


k_live = compile_kernel(live._gated_gather, dict(base_sig, **tail_sig), {})
add_sig = dict(base_sig)
add_sig.update({"shared_tensor": "*bf16", "shared_stride0": "i32", "shared_stride1": "constexpr"})
add_sig.update(tail_sig)
k_add = compile_kernel(fold._gated_gather_add, add_sig, {"shared_stride1": 1})
open(os.path.join(out, "_gated_gather.ptx"), "w").write(k_live.asm["ptx"])
open(os.path.join(out, "_gated_gather_add.ptx"), "w").write(k_add.asm["ptx"])

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptx_rule_g25 import census, verdict  # noqa: E402

c_live, c_add = census(k_live.asm["ptx"]), census(k_add.asm["ptx"])
print("live  _gated_gather     :", c_live)
print("kit   _gated_gather_add :", c_add)
fails, info = verdict(c_live, c_add, k_live.asm["ptx"], k_add.asm["ptx"])
fp_extra = info["extra"]
print("epilogue / extra ops in _gated_gather_add:", fp_extra, "path", info["path"])
summary = (f"live {c_live}\nadd {c_add}\nextra {fp_extra}\nrule {info}\n" + ("PASS" if not fails else "FAIL " + "; ".join(fails)))
open(os.path.join(out, "summary.txt"), "w").write(summary + "\n")
print(summary)
sys.exit(1 if fails else 0)
