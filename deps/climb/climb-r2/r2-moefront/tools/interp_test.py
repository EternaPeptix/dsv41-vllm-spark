#!/usr/bin/env python3
"""Run the ACTUAL _dsv41_rp_kernel source in the Triton interpreter (TRITON_INTERPRET=1, CPU numpy, no GPU) and
compare it with the bit-level emulation of the live chain (tests/emul.py live_quant + the mfront plan/scatter
contract). The interpreter executes the same tl ops on numpy (div_rn, bitcasts, histogram, cumsum, integer
encode), so this checks the kernel's logic and indexing, not the PTX lowering (compile_check.py / device_test.py).
PDL and the lever-2 prefetch are off here (no device intrinsics in the interpreter).

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
sys.path.insert(0, os.path.join(HERE, "tests"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from emul import live_quant, pack_live  # noqa: E402


class _Platform:
    @staticmethod
    def is_arch_support_pdl():
        return False

    @staticmethod
    def fp8_dtype():
        return torch.float8_e4m3fn


for n, kw in (("vllm", {}), ("vllm.platforms", {"current_platform": _Platform()}),
              ("vllm.triton_utils", {"tl": tl, "triton": triton}),
              ("vllm.logger", {"init_logger": lambda n: types.SimpleNamespace(info=lambda *a, **k: None)})):
    m = types.ModuleType(n)
    m.__dict__.update(kw)
    sys.modules[n] = m
spec = importlib.util.spec_from_file_location("dsv41_routeprep",
                                              os.path.join(HERE, "files/vllm/model_executor/layers/fused_moe",
                                                           "dsv41_routeprep.py"))
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)
RNG = np.random.default_rng(3)
E, TOPK, H = 384, 6, 5120
FAILS = []


def m_sum(M):
    s = M * TOPK + min(M * TOPK, E) * 31
    return (s + 31) // 32 * 32


def live_ref(xb, ids):
    """Live chain emulated: quant (SASS order) + mfront plan + scatter (in slot order; any order is equivalent)."""
    M = xb.shape[0]
    q, s = live_quant(xb.reshape(M, 40, 128))
    q = q.reshape(M, H)
    words = pack_live(s.reshape(M, 10, 4))
    Ms = m_sum(M)
    flat = ids.reshape(-1)
    valid = flat >= 0
    cnt = np.bincount(flat[valid], minlength=E)
    al = (cnt + 31) // 32 * 32
    start = np.cumsum(al) - al
    cur = start.copy()
    layout = np.full(Ms, -1, np.int32)
    out = {}
    for slot in range(M * TOPK):
        e = flat[slot]
        if e >= 0:
            d = cur[e]
            cur[e] += 1
            layout[d] = e
            out[slot] = (q[slot // TOPK], words[slot // TOPK])
    return layout, cur, out


def run(M, ids_np, xb):
    Ms = m_sum(M)
    x = torch.from_numpy(xb.astype(np.int16)).view(torch.bfloat16).reshape(M, H)
    ids = torch.from_numpy(ids_np.astype(np.int32))
    aq_out = torch.zeros((Ms, H), dtype=torch.float8_e4m3fn)
    out, sfp, layout, inv, _, ends = rp.fused_permute(x, rp.Deferred(None, None), ids, E, None, None, aq_out, Ms, 32,
                                                      128, False, True)
    lay_ref, ends_ref, rows_ref = live_ref(xb.reshape(M, H), ids_np)
    ok = np.array_equal(layout.numpy(), lay_ref) and np.array_equal(ends.numpy(), ends_ref)
    dead = lay_ref < 0
    ok &= bool((sfp.numpy()[dead] == 0).all())
    ob = out.view(torch.uint8).numpy()
    sw = sfp.numpy().view(np.uint32)
    for slot, (row, words) in rows_ref.items():
        d = inv.numpy().reshape(-1)[slot]
        ok &= np.array_equal(ob[d], row) and np.array_equal(sw[d], words) and lay_ref[d] == ids_np.reshape(-1)[slot]
    return ok


def main():
    cases = 0
    for M in (1, 2, 4, 6, 11):
        for kind in ("random", "dead", "same", "hostile"):
            ids = np.stack([RNG.choice(E, TOPK, replace=False) for _ in range(M)])
            if kind == "dead":
                ids[RNG.random(M) < 0.5] = -1
            if kind == "same":
                ids[:] = ids[0]
            xf = RNG.standard_normal((M, H)).astype(np.float32)
            if kind == "hostile":
                xf = (xf * np.exp2(RNG.integers(-130, 120, size=(M, H)))).astype(np.float32)
                xf[:, :128] = 448.0
                xf[:, 128:256] = -0.0
            with np.errstate(all="ignore"):
                xf = np.where(np.isfinite(xf), xf, 0).astype(np.float32)
            xb = (xf.view(np.uint32) >> 16).astype(np.uint16)
            ok = run(M, ids, xb)
            cases += 1
            print(f"{'PASS' if ok else 'FAIL'} interpreter M={M} {kind}", flush=True)
            if not ok:
                FAILS.append((M, kind))
    print(f"{cases} cases: {'ALL PASS' if not FAILS else 'FAILED ' + str(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
