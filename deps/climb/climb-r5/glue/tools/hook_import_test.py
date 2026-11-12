#!/usr/bin/env python3
"""r5-glue device test part H (review N5): the REAL installed hook path, flags on, on a copy of the venv's vllm.

Builds a symlink tree of site-packages/vllm in a temp dir (directories real, files symlinked: the venv itself is never
written), runs `install.py install` on it (R5G_VLLM), then in a FRESH python with PYTHONPATH=<copy> and the flags on:
  H1 vllm resolves to the copy; importing deep_gemm_utils (production's module-level import site of dsv41_routeprep)
     leaves dsv41_routeprep._launch wrapped (.r5g) by the copy's dsv41_r5g_rp;
  H2 importing dsv41_attnout the way o_proj() does (lazily, after vllm is up) runs the hook, which imports dsv41_r5g_rq
     while dsv41_attnout is still initialising (the circular import); _launch_rq is wrapped (.r5g);
  H3 dsv41_attnout._launch_rq (the rebound name every caller resolves) at M=1,2,4,6 on hostile partials == the r2
     launch (__wrapped__), q + SF bytes; the boot INFO line "...bit-identical ... at every M 1..32 (synthetic sweep"
     is emitted through vLLM's logger;
  H4 fused_permute end to end (M=1,2,4,6, int32 and int64 ids) with the wrapped _launch == with the r2 _launch:
     rows, packed scales, layout, inv_perm, expert ends; the RPSPLIT INFO line names the sweep; no DISABLED line.
Run by device_test.py (subprocess) or alone, in the mandatory environment, on a stopped node:
    python3 tools/hook_import_test.py [--mode split|hoist]
Exit 0 = all pass (prints "HOOK ALL PASS").
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_ENV = {"DSV41_MOE_ROUTEPREP": "1", "DSV41_ATTNOUT_OP": "rope,rq", "DSV41_SONAR_OPROJ": "fp8",
            "DSV41_SONAR_OPROJ_SPLITS": "32", "DSV41_SONAR_OPROJ_MAX_M": "32"}

CHILD = r'''
import importlib, logging, os, sys
T = os.environ["R5G_COPY"]
FAILS = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + str(detail)) if detail and not ok else ''}", flush=True)
    if not ok:
        FAILS.append(name)
import torch
import vllm
check("H1 vllm resolves to the installed copy", vllm.__file__.startswith(T), vllm.__file__)
import vllm.model_executor.layers.fused_moe.deep_gemm_utils  # noqa: F401  production's import site
rp = sys.modules.get("vllm.model_executor.layers.fused_moe.dsv41_routeprep")
check("H1 deep_gemm_utils imported dsv41_routeprep from the copy", rp is not None and rp.__file__.startswith(T),
      getattr(rp, "__file__", None))
check("H1 dsv41_routeprep._launch is the r5-glue wrapper", getattr(rp._launch, "r5g", False)
      and sys.modules["vllm.model_executor.layers.fused_moe.dsv41_r5g_rp"].__file__.startswith(T))
ao = importlib.import_module("vllm.models.deepseek_v4.nvidia.ops.dsv41_attnout")  # what o_proj() does lazily
check("H2 dsv41_attnout._launch_rq is the r5-glue wrapper (circular import resolved)",
      getattr(ao._launch_rq, "r5g", False) and ao.__file__.startswith(T)
      and sys.modules["vllm.models.deepseek_v4.nvidia.ops.dsv41_r5g_rq"]._ao is ao)
# capture vLLM's log records of the two kit modules
recs = []
class H(logging.Handler):
    def emit(self, r):
        recs.append((r.levelname, r.getMessage()))
for n in ("vllm.models.deepseek_v4.nvidia.ops.dsv41_r5g_rq", "vllm.model_executor.layers.fused_moe.dsv41_r5g_rp"):
    logging.getLogger(n).addHandler(H())
def u8(t):
    return t.contiguous().view(torch.uint8).flatten()
gen = torch.Generator().manual_seed(77)
r2rq = ao._launch_rq.__wrapped__
for M in (2, 1, 4, 6):  # production's first eager call is M=2
    P = ao._hostile_partials((32, 1, M, 1024), gen).cuda()
    q, sf = ao._launch_rq(P, M, 1024, 32)
    q2, sf2 = r2rq(P, M, 1024, 32)
    torch.cuda.synchronize()
    check(f"H3 dsv41_attnout._launch_rq M={M} == r2 launch (q + SF bytes)", torch.equal(u8(q), u8(q2))
          and torch.equal(u8(sf), u8(sf2)))
check("H3 RQPAD sweep INFO line through vLLM's logger", any(l == "INFO" and "DSV41_GLUE_RQPAD=1: _r5g_reduce_quant "
      "bit-identical to _r2_reduce_quant at every M 1..32 (synthetic sweep" in m for l, m in recs), recs)
E, T6, H5 = 384, 6, 5120
for dt in (torch.int32, torch.int64):
    for M in (2, 1, 4, 6):
        aq = (torch.randn(M, H5, generator=gen) * 3).to(torch.bfloat16).cuda()
        ids = torch.stack([torch.randperm(E, generator=gen)[:T6] for _ in range(M)]).to(dt).cuda()
        Ms, al = rp._m_sum(M)
        outs = []
        for launch in (rp._launch, rp._launch.__wrapped__):
            keep = rp._launch
            rp._launch = launch
            try:
                aq_out = torch.full((Ms, H5), 0x5A, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
                outs.append(rp.fused_permute(aq, rp.Deferred(None, None), ids, E, None, None, aq_out, Ms, 32, 128,
                                             False, True))
            finally:
                rp._launch = keep
        torch.cuda.synchronize()
        names = ("rows", "scales", "layout", "inv_perm", "align", "ends")
        bad = [nm for nm, a, b in zip(names, outs[0], outs[1]) if nm != "align" and not torch.equal(u8(a), u8(b))]
        check(f"H4 fused_permute M={M} {dt}: wrapped _launch == r2 _launch (rows/scales/layout/inv_perm/ends)",
              not bad and al == 32, (bad, al))
check("H4 RPSPLIT sweep INFO line through vLLM's logger", any(l == "INFO" and "_r5g_rp_kernel bit-identical to "
      "_dsv41_rp_kernel at every M 1..10 (synthetic sweep" in m for l, m in recs), recs)
check("H no DISABLED line", not any("DISABLED" in m for _, m in recs), recs)
print("HOOK ALL PASS" if not FAILS else f"HOOK FAILED: {FAILS}", flush=True)
sys.exit(1 if FAILS else 0)
'''


def symlink_tree(src, dst):
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            os.symlink(os.path.join(root, f), os.path.join(dst, rel, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="split", choices=("split", "hoist"))
    args = ap.parse_args()
    import vllm  # the venv's (mandatory environment: ~/venvs/vllm032)

    site_v = os.path.dirname(vllm.__file__)
    tmp = tempfile.mkdtemp(prefix="r5g-hook-")
    try:
        copy = os.path.join(tmp, "vllm")
        symlink_tree(site_v, copy)
        r = subprocess.run([sys.executable, os.path.join(HERE, "install.py"), "install"], capture_output=True,
                           text=True, env=dict(os.environ, R5G_VLLM=copy))
        print(r.stdout + r.stderr, flush=True)
        if r.returncode != 0 or "OVERALL INSTALLED" not in r.stdout:
            print("HOOK FAILED: install.py install on the copy did not report OVERALL INSTALLED", flush=True)
            return 1
        env = dict(os.environ, R5G_COPY=copy, PYTHONPATH=tmp + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   DSV41_GLUE_RQPAD="1", DSV41_GLUE_RPSPLIT=args.mode)
        for k, v in PROD_ENV.items():  # production's launcher defaults (device_test.py passes the launcher's own)
            env.setdefault(k, v)
        r = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=1800,
                           cwd=tmp)
        print(r.stdout, flush=True)
        if r.returncode != 0:
            print(r.stderr[-4000:], flush=True)
        return r.returncode
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
