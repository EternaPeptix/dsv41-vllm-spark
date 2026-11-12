#!/usr/bin/env python3
"""Tests for dsv41_oblit.py + install_oblit.py. Run on a Spark with the vllm032 venv python:

    ~/venvs/vllm032/bin/python test_oblit.py            # CPU tests + GPU tests if a GPU is free
Runs each mode in a fresh subprocess so the module-level DSV41_OBLIT switch is read per case.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
V = os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")


def run(code, env_extra):
    env = dict(os.environ, PYTHONPATH=HERE, **env_extra)
    r = subprocess.run([PY, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    return r.returncode, out


def main():
    tmp = tempfile.mkdtemp(prefix="oblit-test-")
    fails = 0
    try:
        npz = os.path.join(tmp, "dirs.npz")
        subprocess.run([PY, "-c", textwrap.dedent(f"""
            import numpy as np
            r = np.random.default_rng(0)
            np.savez({npz!r}, **{{f"L{{i}}": r.standard_normal(5120).astype(np.float32) * 3 for i in range(2, 40)}},
                     head=r.standard_normal(5120).astype(np.float32))
        """)], check=True)

        cases = {
            "binding": ("""
                import dsv41_oblit as o
                assert o.ON
                assert o.layer_for("model.layers.5") == 5
                assert o.layer_for("language_model.model.layers.39") == 39
                assert o.layer_for("model.layers.0") == -1 and o.layer_for("model.layers.1") == -1
                assert o.layer_for("model.layers.40") == -1 and o.layer_for("model.layers.5.attn") == -1
                assert o.layer_for("mtp.0") == -1 and o.layer_for("dspark.layers.3") == -1
                assert o.bound_layers() == [5, 39], o.bound_layers()
                print("ok")
            """, {"DSV41_OBLIT": npz}),
            "off": ("""
                import dsv41_oblit as o
                assert not o.ON and o.layer_for("model.layers.5") == -1
                print("ok")
            """, {"DSV41_OBLIT": ""}),
            "math_cpu": ("""
                import numpy as np, torch, dsv41_oblit as o
                d = np.load(o.PATH)["L7"].astype(np.float64); d /= np.linalg.norm(d)
                x = torch.randn(33, 5120, dtype=torch.float32).to(torch.bfloat16) * 4 + torch.from_numpy(d * 50).to(torch.bfloat16)
                y = o.project(x, 7)
                assert y.dtype == x.dtype and y.shape == x.shape and y.data_ptr() != x.data_ptr()
                xr = x.double().numpy(); ref = xr - np.outer(xr @ d, d)
                err = np.abs(y.double().numpy() - ref).max() / np.abs(ref).max()
                comp = np.abs(y.double().numpy() @ d).max()
                assert err < 1e-2, err
                assert comp < 0.2, comp           # was ~50 before; bf16 rounding leaves O(0.1)
                assert torch.equal(o.project(x, 7), y)   # deterministic
                print(f"ok rel_err {err:.2e} residual component {comp:.3f}")
            """, {"DSV41_OBLIT": npz}),
            "moe_output": ("""
                import dataclasses, torch, dsv41_oblit as o
                from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput, UnfinalizedMoEOutput
                fields = {f.name for f in dataclasses.fields(UnfinalizedMoEOutput)}
                kw = {}
                for f in dataclasses.fields(UnfinalizedMoEOutput):
                    kw[f.name] = torch.randn(12, 5120).to(torch.bfloat16) if f.name == "gemm2_permuted" else torch.zeros(4, 6)
                u = UnfinalizedMoEOutput(**kw)
                mo = MoEOutput(routed=u, shared_output=torch.randn(4, 5120).to(torch.bfloat16))
                p = o.project_moe(mo, 3)
                d = torch.from_numpy(o._load()["L3"]).double()
                assert p.routed.gemm2_permuted.double().matmul(d).abs().max() < 0.05
                assert p.shared_output.double().matmul(d).abs().max() < 0.05
                for f in fields - {"gemm2_permuted"}:
                    assert getattr(p.routed, f) is getattr(u, f), f   # untouched fields kept
                mo2 = MoEOutput(routed=torch.randn(4, 5120).to(torch.bfloat16), shared_output=None)
                p2 = o.project_moe(mo2, 3)
                assert p2.shared_output is None and p2.routed.double().matmul(d).abs().max() < 0.05
                t = o.project_moe(torch.randn(4, 5120).to(torch.bfloat16), 3)
                assert t.double().matmul(d).abs().max() < 0.05
                print("ok", sorted(fields))
            """, {"DSV41_OBLIT": npz}),
            "gpu": ("""
                import numpy as np, torch, dsv41_oblit as o
                if not torch.cuda.is_available():
                    print("skip: no cuda"); raise SystemExit(0)
                assert o._TRITON
                dev = torch.device("cuda")
                d = np.load(o.PATH)["L9"].astype(np.float64); d /= np.linalg.norm(d)
                for shape in [(1, 5120), (7, 5120), (16384, 5120)]:
                    x = (torch.randn(*shape, device=dev) * 3).to(torch.bfloat16)
                    y = o.project(x, 9)
                    xr = x.double().cpu().numpy(); ref = xr - np.outer(xr @ d, d)
                    tol = 2.0 ** -7 * np.abs(ref).max()          # one bf16 rounding at the largest magnitude
                    err = np.abs(y.double().cpu().numpy() - ref).max()
                    assert err <= tol, (shape, err, tol)
                    yc = o.project(x.cpu(), 9).to(dev)        # torch fallback agrees with triton
                    assert (y.float() - yc.float()).abs().max() <= tol
                xs = torch.randn(20, 5120, device=dev).to(torch.bfloat16)[::2]   # strided rows
                assert (o.project(xs, 9).float() - o.project(xs.contiguous(), 9).float()).abs().max() == 0
                # graph capture with a pre-initialised direction
                xg = torch.randn(64, 5120, device=dev).to(torch.bfloat16)
                o.project(xg, 11)                              # eager first use creates the vector
                g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    o.project(xg, 11)
                torch.cuda.current_stream().wait_stream(s)
                with torch.cuda.graph(g):
                    yg = o.project(xg, 11)
                xg.copy_(torch.randn(64, 5120, device=dev).to(torch.bfloat16)); g.replay(); torch.cuda.synchronize()
                assert (yg.float() - o.project(xg, 11).float()).abs().max() == 0
                # first use inside capture must raise
                g2 = torch.cuda.CUDAGraph(); raised = False
                try:
                    with torch.cuda.graph(g2):
                        o.project(xg, 13)
                except RuntimeError as e:
                    raised = "capture" in str(e)
                assert raised
                # speed: decode-like and prefill-like rows
                import time
                for rows in (96, 16384):
                    xb = torch.randn(rows, 5120, device=dev).to(torch.bfloat16); o.project(xb, 9); torch.cuda.synchronize()
                    t = time.perf_counter()
                    for _ in range(200): o.project(xb, 9)
                    torch.cuda.synchronize(); us = (time.perf_counter() - t) / 200 * 1e6
                    print(f"rows {rows}: {us:.1f} us/projection (eager, incl. launch)")
                print("ok")
            """, {"DSV41_OBLIT": npz}),
        }
        for name, (code, env) in cases.items():
            rc, out = run(code, env)
            status = "PASS" if rc == 0 else "FAIL"
            fails += rc != 0
            print(f"[{status}] {name}: {out[-1500:] if rc else out.splitlines()[-1] if out else ''}")
            if rc == 0 and name == "gpu":
                print("    " + "\n    ".join(out.splitlines()[:-1]))

        # installer: dry run + install/uninstall on a COPY of the real model.py
        fake = os.path.join(tmp, "vllm")
        os.makedirs(os.path.join(fake, "models/deepseek_v41/nvidia"))
        real = os.path.join(V, "models/deepseek_v41/nvidia/model.py")
        dst = os.path.join(fake, "models/deepseek_v41/nvidia/model.py")
        shutil.copy2(real, dst)
        env = dict(os.environ, OBLIT_VLLM=fake)
        inst = os.path.join(HERE, "install_oblit.py")
        r1 = subprocess.run([PY, inst, "install"], env=env, capture_output=True, text=True)
        txt = open(dst).read()
        r2 = subprocess.run([PY, "-m", "py_compile", dst], capture_output=True, text=True)
        r3 = subprocess.run([PY, inst, "install"], env=env, capture_output=True, text=True)   # idempotent
        r4 = subprocess.run([PY, inst, "uninstall"], env=env, capture_output=True, text=True)
        same = open(dst).read() == open(real).read()
        ok = (r1.returncode == 0 and txt.count("# dsv41-oblit") == 16 and r2.returncode == 0
              and "already patched" in r3.stdout and same and not os.path.exists(dst.replace("model.py", "dsv41_oblit.py")))
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] installer: {r1.stdout.strip().splitlines()[0] if r1.stdout else r1.stderr[-400:]}"
              f" | tagged {txt.count('# dsv41-oblit')} | compile rc {r2.returncode} {r2.stderr[-300:]} | idempotent "
              f"{'already patched' in r3.stdout} | uninstall restores {same}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
