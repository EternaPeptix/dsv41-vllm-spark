#!/usr/bin/env python3
"""Triton CPU-interpreter check (TRITON_INTERPRET=1, CPU tensors only, no GPU / CUDA context): run the REAL kernel
sources -- the live inverse-RoPE kernel (extracted verbatim from base/.../fused_inv_rope_fp8_quant.py), the live
_sm120_wo_a_partial (base/ sonar copy) and the kit's _r2_wo_a_partial -- and compare the partial buffers bitwise.
RoPE uses quarter-turn cos/sin (exact either way). The interpreter's tl.dot is numpy fp32 matmul, identical for identical A/W tiles, so this checks the kit kernel's
indexing / masking / RoPE fold / scale vectorisation / BN=128 tiling as Triton actually executes the source
(the MMA order itself is the compile_check + device_test's job).

    TRITON_INTERPRET=1 nice -n 19 ~/venvs/vllm032/bin/python scripts/interp_check.py
"""
import ast
import hashlib
import json
import logging
import os
import sys
import types

os.environ["TRITON_INTERPRET"] = "1"
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _n in ("vllm", "vllm.logger", "vllm.triton_utils"):
    sys.modules[_n] = types.ModuleType(_n)
sys.modules["vllm"].__path__ = []
sys.modules["vllm.logger"].init_logger = logging.getLogger
sys.modules["vllm.triton_utils"].tl = tl
sys.modules["vllm.triton_utils"].triton = triton
import importlib.util  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


kit = json.load(open(os.path.join(KIT, "KIT.json")))
SON = os.path.join(KIT, "base/models/deepseek_v4/nvidia/ops/sonar_bf16_bmm.py")
assert hashlib.sha256(open(SON, "rb").read()).hexdigest() == [f["base_sha256"] for f in kit["files"]
                                                             if f["path"].endswith("sonar_bf16_bmm.py")][0]
sonar = _load("sonar_live", SON)
# The kit's _launch_partial / _sonar() import vllm.models.deepseek_v4.nvidia.ops.sonar_bf16_bmm: register the
# loaded base copy under that name (review fix; without it the stubbed `vllm` raises ModuleNotFoundError).
for _n in ("vllm.models", "vllm.models.deepseek_v4", "vllm.models.deepseek_v4.nvidia",
           "vllm.models.deepseek_v4.nvidia.ops"):
    sys.modules[_n] = types.ModuleType(_n)
    sys.modules[_n].__path__ = []
sys.modules["vllm.models.deepseek_v4.nvidia.ops"].sonar_bf16_bmm = sonar
sys.modules["vllm.models.deepseek_v4.nvidia.ops.sonar_bf16_bmm"] = sonar
results = {}
for parts in ("rope", "vs", "bn128", "rope,vs,bn128"):
    os.environ["DSV41_ATTNOUT_OP"] = parts
    results[parts] = _load("ao_" + parts.replace(",", "_"), os.path.join(KIT, "files/models/deepseek_v4/nvidia/ops/dsv41_attnout.py"))

# live inverse-RoPE kernel, verbatim source of FusedInvRopeFP8QuantKernel.kernel
INV = os.path.join(KIT, "base/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py")
src = open(INV).read()
fn = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "kernel"][0]
body = ast.get_source_segment(src, fn)
import textwrap  # noqa: E402

ns = {"tl": tl, "triton": triton}
exec("@triton.jit(do_not_specialize=['num_tokens', 'scale_stride_k'])\n" + textwrap.dedent(body), ns)
inv_kernel = ns["kernel"]

G, HPG, HD, NOPE, HALF, N = 1, 8, 512, 448, 32, 1024
K = HPG * HD
SPLITS = 32


def inv_rope(o, pos, cs):
    M = o.shape[0]
    out = torch.empty((G, M, K), dtype=torch.bfloat16)
    inv_kernel[(M, G * HPG)](o, pos, cs, out, torch.empty(1, dtype=torch.int32), M, HPG, o.stride(0), o.stride(1),
                              cs.stride(0), out.stride(0), out.stride(1), 0, 0, 448.0, 1e-10, 128, 4, NOPE, HALF,
                              False, False, False)
    return out.transpose(0, 1)


def live_partial(a, wq, ws, M):
    P = torch.empty((SPLITS, G, M, N), dtype=torch.float32)
    sonar._sm120_wo_a_partial[(N // 64, G * triton.cdiv(M, 16), SPLITS)](
        a, wq, ws, P, M, G, N, K, a.stride(1), a.stride(0), a.stride(2), K // SPLITS, True)
    return P


def main():
    gen = torch.Generator().manual_seed(7)
    q = (torch.randn((G, N, K), generator=gen) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    e = torch.randint(110, 125, (G, N, K // 32), generator=gen).to(torch.uint8)
    # quarter-turn rotations only (cos, sin in {0, +-1}, random per (pos, pair)): every RoPE product/sum is exact,
    # so the interpreter's lack of FMA contraction cannot matter and any index error (partner, cos vs sin
    # column, position row) shows up as a different value.
    quarter = torch.randint(0, 4, (4096, HALF), generator=gen)
    cos = torch.tensor([1.0, 0.0, -1.0, 0.0])[quarter]
    sin = torch.tensor([0.0, 1.0, 0.0, -1.0])[quarter]
    cs = torch.cat([cos, sin], 1).float().contiguous()
    ok = True
    for M in (1, 5, 17):
        o = (torch.randn((M, G * HPG, HD), generator=gen) * torch.exp2(
            torch.randint(-12, 8, (M, G * HPG, HD), generator=gen).float())).to(torch.bfloat16)
        pos = torch.randint(0, 4096, (M,), generator=gen)
        a = inv_rope(o, pos, cs)
        ref = live_partial(a, q, e, M)
        for parts, m in results.items():
            P = torch.empty_like(ref)
            rope = (pos, cs, HPG, NOPE, HALF) if m.ROPE else None
            m._launch_partial(o if m.ROPE else a, rope, q, e, P, M, G, N, K, SPLITS, True, new=True)
            same = torch.equal(P.view(torch.int32), ref.view(torch.int32))
            ok &= same
            print(f"M={M:2d} parts={parts:14s} partial bit-identical to live (interpreter): {same}", flush=True)
    print("INTERP CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
