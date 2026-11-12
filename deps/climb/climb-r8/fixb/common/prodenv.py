"""prodenv.py (r8-fixb): the device-test environment rule, shared by head/, skew/ and step/ device tests.

Rule (climb-r8 BRIEF "Device-test environment (mandatory)"): every device test runs with
    source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
asserts at start that the serving op is the one under test (each test does that itself), mirrors the production
launcher's env (including DSV41_GLUE_RQPAD=1 DSV41_GLUE_RPSPLIT=split) and carries must-fail mutants.

    env_rule()                      -> [(name, ok, detail)]   venv / CUDA_HOME / nvcc + ptxas from /usr/local/cuda/bin
    mirror_launcher(path, expect)   -> (env dict, [(name, ok, detail)])  parses `export A=${A:-v}` / `${A-v}` / plain
                                       `export A=v B=w` lines of the launcher whose sha256[:8] must be `expect`, and
                                       os.environ.setdefault()s every value (an explicit value in the caller's env wins,
                                       so an arm env can be layered on top; the checks report it).
    selftest_mutants(launcher)      -> [(name, caught, detail)]  the stripped-env subprocess and a launcher copy without
                                       the glue lines must both FAIL the rule.
stdlib only; no side effect except os.environ.setdefault in mirror_launcher.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

VENV = os.path.expanduser("~/venvs/vllm032")
CUDA = "/usr/local/cuda"
LAUNCHER = os.path.expanduser("~/dsv41-tp8.sh")
EXPECT = os.environ.get("FIXB_EXPECT", "4f3cc1c6")  # production launcher sha8 on 2026-11-09; pass --expect after a promotion
REQUIRED = {"DSV41_GLUE_RQPAD": "1", "DSV41_GLUE_RPSPLIT": "split"}
_SKIP = {"PATH", "VLLM_HOST_IP", "LD_LIBRARY_PATH", "PYTHONPATH"}

_RX_DEF = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=\$\{\1:?-([^}]*)\}")


def env_rule():
    out = []
    out.append(("env: running in ~/venvs/vllm032", os.path.realpath(sys.prefix) == os.path.realpath(VENV), sys.prefix))
    out.append(("env: CUDA_HOME=/usr/local/cuda", os.environ.get("CUDA_HOME") == CUDA, os.environ.get("CUDA_HOME")))
    for tool in ("nvcc", "ptxas"):
        w = shutil.which(tool) or ""
        out.append((f"env: {tool} on PATH is {CUDA}/bin/{tool}", os.path.realpath(w) == os.path.realpath(f"{CUDA}/bin/{tool}"),
                    w or "none"))
    return out


def parse_launcher(text):
    """{name: value} for the launcher's exports with a literal default or a plain literal value ($HOME expanded)."""
    env = {}
    for ln in text.splitlines():
        s = ln.strip()
        if not s.startswith("export "):
            continue
        m = _RX_DEF.match(s)
        if m:
            name, val = m.group(1), m.group(2)
            if "$(" in val or "`" in val:
                continue
            env[name] = val.replace("$HOME", os.path.expanduser("~"))
            continue
        body = s[len("export "):].split("#", 1)[0]
        for tok in body.split():
            if "=" not in tok:
                continue
            name, val = tok.split("=", 1)
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) or name in _SKIP:
                continue
            if "$" in val or "`" in val or "(" in val:
                continue
            env[name] = val
    return env


def mirror_launcher(path=LAUNCHER, expect=EXPECT, apply=True):
    checks = []
    try:
        raw = open(path, "rb").read()
    except OSError as e:
        return {}, [(f"launcher {path} readable", False, repr(e))]
    s8 = hashlib.sha256(raw).hexdigest()[:8]
    checks.append((f"launcher {path} is production {expect}", s8 == expect, s8))
    env = parse_launcher(raw.decode("utf-8", "replace"))
    overridden = []
    if apply:
        for k, v in env.items():
            if k in os.environ and os.environ[k] != v:
                overridden.append(k)
            os.environ.setdefault(k, v)
    eff = {k: (os.environ.get(k) if apply else env.get(k)) for k in REQUIRED}
    checks.append(("launcher env mirrored: DSV41_GLUE_RQPAD=1 DSV41_GLUE_RPSPLIT=split in effect",
                   all(eff[k] == v for k, v in REQUIRED.items()), str(eff)))
    checks.append((f"launcher env mirrored: {len(env)} exports applied (arm overrides: {overridden or 'none'})",
                   len(env) >= 40, ""))
    return env, checks


_CHILD = r"""
import os, shutil, sys
ok = (os.path.realpath(sys.prefix) == os.path.realpath(%r) and os.environ.get('CUDA_HOME') == %r
      and os.path.realpath(shutil.which('nvcc') or '') == os.path.realpath(%r))
print('ENV', 'OK' if ok else 'BAD', sys.prefix, os.environ.get('CUDA_HOME'), shutil.which('nvcc'))
sys.exit(0 if ok else 2)
""" % (VENV, CUDA, CUDA + "/bin/nvcc")


def selftest_mutants(path=LAUNCHER, expect=EXPECT):
    """Must-fail controls: (1) the env check in a stripped environment (a non-interactive ssh shell) fails;
    (2) a launcher copy without the glue lines fails the mirror check; (3) the real check in this env passes."""
    res = []
    r = subprocess.run(["env", "-i", "PATH=/usr/bin:/bin", "/usr/bin/python3", "-I", "-c", _CHILD],
                       capture_output=True, text=True)
    res.append(("env rule: stripped environment (env -i, system python) must FAIL", r.returncode != 0, r.stdout.strip()))
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        txt = ""
    bad = "\n".join(ln for ln in txt.splitlines() if "DSV41_GLUE_RQPAD" not in ln and "DSV41_GLUE_RPSPLIT" not in ln)
    d = tempfile.mkdtemp(prefix="fixb-env-")
    p = os.path.join(d, "dsv41-tp8.sh")
    open(p, "w").write(bad)
    saved = {k: os.environ.pop(k, None) for k in REQUIRED}
    try:
        _, ch = mirror_launcher(p, hashlib.sha256(bad.encode()).hexdigest()[:8], apply=False)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        shutil.rmtree(d, ignore_errors=True)
    glue = [c for c in ch if "GLUE" in c[0]]
    res.append(("launcher mirror: a launcher copy without the glue exports must FAIL", bool(glue) and not glue[0][1],
                glue[0][2] if glue else "no glue check"))
    return res
