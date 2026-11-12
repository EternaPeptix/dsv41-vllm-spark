#!/usr/bin/env python3
"""Run the ACTUAL r2 _dsv41_rp_kernel and the kit _r5g_rp_kernel (hoist and split) in the Triton interpreter
(TRITON_INTERPRET=1: numpy on CPU, no GPU) and compare every output byte (rows, packed scale words, layout, inv_perm,
expert ends) on poisoned buffers. Checks the kernels' logic and indexing (not the PTX lowering: compile_check.py /
device_test.py). PDL and the FC1 prefetch are off here (no device intrinsics in the interpreter). RQPAD is not run here:
its kernel body is inline PTX (the interpreter cannot execute it); tests/test_writesets.py covers its index math.

    CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 ~/venvs/vllm032/bin/python tools/interp_test.py
"""
import importlib.util
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRITON_INTERPRET"] = "1"
os.environ["DSV41_MOE_ROUTEPREP"] = "1"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

E, TOPK, H = 384, 6, 5120
FAILS = []


class _Platform:
    @staticmethod
    def is_arch_support_pdl():
        return False

    @staticmethod
    def fp8_dtype():
        return torch.float8_e4m3fn


def stub(name, **kw):
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m
    return m


for n in ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fused_moe"):
    stub(n)
stub("vllm.platforms", current_platform=_Platform())
stub("vllm.triton_utils", tl=tl, triton=triton)
stub("vllm.logger", init_logger=lambda n: types.SimpleNamespace(info=lambda *a, **k: None,
                                                               error=lambda *a, **k: None))


def load(name, path, env):
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    setattr(sys.modules["vllm.model_executor.layers.fused_moe"], name.rsplit(".", 1)[1], m)
    spec.loader.exec_module(m)
    return m


RPL = load("vllm.model_executor.layers.fused_moe.dsv41_routeprep", os.path.join(HERE, "base/dsv41_routeprep.py"), {})
KIT = load("vllm.model_executor.layers.fused_moe.dsv41_r5g_rp",
           os.path.join(HERE, "files/vllm/model_executor/layers/fused_moe/dsv41_r5g_rp.py"),
           {"DSV41_GLUE_RPSPLIT": "split"})
RNG = np.random.default_rng(11)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def m_sum(M):
    s = M * TOPK + min(M * TOPK, E) * 31
    return (s + 31) // 32 * 32


def bufs(M, Ms, p8, p32):
    mn = (Ms + 3) // 4 * 4
    return dict(out=torch.full((Ms, H), p8, dtype=torch.uint8), sfp=torch.full((10 * mn,), p32, dtype=torch.int32),
                layout=torch.full((Ms,), p32, dtype=torch.int32), inv=torch.full((M, TOPK), p32, dtype=torch.int32),
                ends=torch.full((E,), p32, dtype=torch.int32), mn=mn)


def run(fn, x, ids, M, Ms, b, **kw):
    fn(x, x.stride(0), ids, ids.stride(0), ids.stride(1), b["out"], b["out"].stride(0), b["sfp"], b["mn"], b["layout"],
       b["inv"], b["inv"].stride(0), b["inv"].stride(1), b["ends"], None, None, M, Ms, **kw)


def ids_for(M, kind):
    if kind == "same":
        ids = np.tile(RNG.choice(E, TOPK, replace=False), (M, 1))
    elif kind == "dead":
        ids = np.stack([RNG.choice(E, TOPK, replace=False) for _ in range(M)])
        ids[RNG.random(ids.shape) < 0.3] = -1
    else:
        ids = np.stack([RNG.choice(E, TOPK, replace=False) for _ in range(M)])
    return torch.from_numpy(ids).to(torch.int32)


def x_for(M, kind):
    r = RNG.standard_normal((M, H)).astype(np.float32)
    if kind == "hostile":
        r = r * np.exp2(RNG.integers(-130, 120, size=(M, H)))
        r[:, ::5] = 0.0
        r = np.where(np.isfinite(r), r, 0).astype(np.float32)
    return torch.from_numpy(r.astype(np.float32)).to(torch.bfloat16)


def main():
    n = 0
    bad = []
    for M in (1, 2, 3, 4, 6, 10):
        Ms = m_sum(M)
        for kind in ("distinct", "same", "dead"):
            for xk in ("normal", "hostile"):
                x, ids = x_for(M, xk), ids_for(M, kind)
                for split in (False, True):
                    for p8, p32 in ((0xA5, 0x5A5A5A5A), (0, 0)):
                        a, b = bufs(M, Ms, p8, p32), bufs(M, Ms, p8, p32)
                        run(RPL._launch, x, ids, M, Ms, a)
                        run(KIT._launch_new, x, ids, M, Ms, b, split=split)
                        n += 1
                        for k in ("out", "sfp", "layout", "inv", "ends"):
                            if not torch.equal(a[k], b[k]):
                                bad.append(f"M={M} {kind} {xk} {'split' if split else 'hoist'} poison {p8:#x}: {k}")
                                break
    check(f"interpreter: kit hoist + split == r2 kernel on every output byte ({n} cases)", not bad, "; ".join(bad[:5]))
    # one must-fail mutant through the interpreter: the harness has to see it
    src = open(os.path.join(HERE, "files/vllm/model_executor/layers/fused_moe/dsv41_r5g_rp.py")).read()
    old = "(v3 << 1)\n"
    assert src.count(old) == 1
    p = os.path.join(HERE, "results", "interp_mutant_rp.py")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(src.replace(old, "(v3 << 2)\n"))
    mut = load("vllm.model_executor.layers.fused_moe.interp_mutant_rp", p, {"DSV41_GLUE_RPSPLIT": "split"})
    missed = []
    for M in (1, 2, 4, 6, 10):
        Ms = m_sum(M)
        x, ids = x_for(M, "normal"), ids_for(M, "distinct")
        for split in (False, True):
            a, b = bufs(M, Ms, 0, 0), bufs(M, Ms, 0, 0)
            run(RPL._launch, x, ids, M, Ms, a)
            run(mut._launch_new, x, ids, M, Ms, b, split=split)
            if torch.equal(a["sfp"], b["sfp"]):
                missed.append((M, split))
    os.remove(p)
    check("interpreter: mutant (packed word v3 << 2) caught at every M, both modes", not missed, str(missed))
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
