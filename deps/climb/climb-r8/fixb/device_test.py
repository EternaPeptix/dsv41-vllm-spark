#!/usr/bin/env python3
"""r8-fixb device tests: one entry point for the three repaired device tests (run on a STOPPED node, device window).

    source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
    cd ~/climb-scratch/r8-fixb && python3 device_test.py head|step|skew [--expect <sha8>] [sub-test args]

The environment rule is checked HERE first (common/prodenv.py: venv vllm032, CUDA_HOME, nvcc/ptxas from
/usr/local/cuda/bin, the production launcher ~/dsv41-tp8.sh == --expect, GLUE_RQPAD=1 / RPSPLIT=split in its env) and
again inside each sub-test, which also asserts its serving op and carries its must-fail mutants:
    head  head/device_test.py --part all [--weights real --rank 4]   anchor, bitwise, sweeps, ties, mutants (+ the r8
          per-config mutant sweep: fit + caught at every swept cfg), graph, dispatch, time, l2
    step  step/device_test.py --source kit|installed --part all       E0, C3a, C3b, C4, det (long-prompt determinism)
    skew  skew/device_test.py --core 16                               T-1..T4 incl. the r8 T4c classifier + T4d audit
Exit: the sub-test's exit code; 2 when the environment rule fails here (nothing is run).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
import prodenv  # noqa: E402

SUB = {"head": ("head", ["python", "device_test.py"]), "step": ("step", ["python3", "device_test.py"]),
       "skew": ("skew", ["python3", "device_test.py"])}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SUB:
        print(__doc__)
        return 2
    kit, rest = sys.argv[1], sys.argv[2:]
    expect = prodenv.EXPECT
    if "--expect" in rest:
        expect = rest[rest.index("--expect") + 1]
    else:
        rest += ["--expect", expect]
    bad = [c for c in prodenv.env_rule() if not c[1]]
    _, mch = prodenv.mirror_launcher(prodenv.LAUNCHER, expect, apply=False)
    bad += [c for c in mch if not c[1]]
    for name, ok, det in prodenv.env_rule() + mch:
        print(f"{'PASS' if ok else 'FAIL'} r8-fixb {name}: {det}")
    if bad:
        print("ENVIRONMENT RULE FAILED: source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH "
              "CUDA_HOME=/usr/local/cuda (and the production launcher must be --expect); nothing was run")
        return 2
    d, cmd = SUB[kit]
    cmd = [sys.executable if c in ("python", "python3") else c for c in cmd]
    return subprocess.call(cmd + rest, cwd=os.path.join(HERE, d))


if __name__ == "__main__":
    sys.exit(main())
