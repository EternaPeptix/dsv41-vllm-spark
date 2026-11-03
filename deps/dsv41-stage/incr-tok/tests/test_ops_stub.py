"""Drive ops/*.sh against 8 stub hosts on this Mac (stdlib only, seconds, no host contacted) and unit-test the two verdict
scripts. A stub `ssh` runs the remote command locally with HOME=<tmp>/<host>, whose ~/venvs/vllm032/... holds a copy of
the base deepseek_v4.py (base/deepseek_v4.py, f7175f99) and whose stub `systemctl` answers from <host>/.svc.

Checks: stage8 stages 8/8; stage8 refuses a host with a moved base; site8 install refuses on one running host and writes
nothing anywhere; site8 install/status/uninstall round trip; arm.sh usage errors exit 2 before any ssh; arm_check.py and
ab_compare.py pass good data and fail each gate.

usage: python3 -I tests/test_ops_stub.py     (prints VERDICT PASS|FAIL)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
HOSTS = "spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight".split()
SITE_REL = "venvs/vllm032/lib/python3.12/site-packages/vllm"
FAILS = []


def check(c, what):
    print(("ok   " if c else "FAIL ") + what, flush=True)
    if not c:
        FAILS.append(what)


ROOT = Path(tempfile.mkdtemp(prefix="incrtok-ops-"))
STUB = ROOT / "bin"
STUB.mkdir()
(STUB / "ssh").write_text(f"""#!/bin/bash
nostdin=0
while [ $# -gt 0 ]; do case "$1" in -n) nostdin=1; shift ;; -o|-J|-p|-i) shift 2 ;; -*) shift ;; *) break ;; esac; done
h=$1; shift
[ -d "{ROOT}/$h" ] || {{ echo "ssh: no host $h" >&2; exit 255; }}
if [ $nostdin = 1 ]; then exec env HOME="{ROOT}/$h" PATH="{STUB}:$PATH" bash -c "$*" < /dev/null; fi
exec env HOME="{ROOT}/$h" PATH="{STUB}:$PATH" bash -c "$*"
""")
(STUB / "systemctl").write_text("#!/bin/bash\ncat \"$HOME/.svc\" 2>/dev/null || echo inactive\n")
(STUB / "timeout").write_text("#!/bin/bash\nshift\nexec \"$@\"\n")
for f in STUB.iterdir():
    f.chmod(0o755)
BASE = (KIT / "base/deepseek_v4.py").read_bytes()
check(hashlib.sha256(BASE).hexdigest().startswith("f7175f99"), "kit base copy is f7175f99")
for h in HOSTS:
    d = ROOT / h / SITE_REL / "renderers"
    d.mkdir(parents=True)
    (d / "deepseek_v4.py").write_bytes(BASE)
ENV = {**os.environ, "PATH": f"{STUB}:{os.environ['PATH']}"}


def run(*cmd):
    r = subprocess.run(list(cmd), capture_output=True, text=True, env=ENV, cwd=str(KIT))
    return r.returncode, r.stdout + r.stderr


def site(h):
    return ROOT / h / SITE_REL / "renderers"


rc, out = run("bash", "ops/stage8.sh")
check(rc == 0 and "stage8: 8/8 staged OK" in out, f"stage8 stages 8/8 ({out.strip().splitlines()[-1][:120] if out.strip() else ''})")
check((ROOT / "spark-five/dsv41-stage/incr-tok/site/vllm/renderers/dsv41_incr_tok.py").exists(), "stage8 wrote the kit on a host")

(site("spark-six") / "deepseek_v4.py").write_bytes(BASE + b"#x\n")
rc, out = run("bash", "ops/stage8.sh")
check(rc == 1 and "spark-six failed to stage" in out, "stage8 refuses a host with a moved base")
(site("spark-six") / "deepseek_v4.py").write_bytes(BASE)

(ROOT / "spark-three/.svc").write_text("active\n")
rc, out = run("bash", "ops/site8.sh", "install")
check(rc == 1 and "spark-three precheck" in out and "nothing written" in out, "site8 install refuses while one host runs")
check(all((site(h) / "deepseek_v4.py").read_bytes() == BASE and not (site(h) / "dsv41_incr_tok.py").exists() for h in HOSTS),
      "refused install wrote nothing on any host")
(ROOT / "spark-three/.svc").unlink()

rc, out = run("bash", "ops/site8.sh", "install")
check(rc == 0 and "site8 install: 8/8 exit 0, identical, state patched" in out, "site8 install 8/8 patched")
rc, out = run("bash", "ops/site8.sh", "status", "patched")
check(rc == 0, "site8 status patched")
rc, out = run("bash", "ops/site8.sh", "status", "base")
check(rc == 1, "site8 status base fails on a patched site")
(site("spark-eight") / "dsv41_incr_tok.py").write_bytes(b"# tampered\n")
rc, out = run("bash", "ops/site8.sh", "status", "patched")
check(rc == 1 and "spark-eight" in out, "site8 status names a host with a tampered side file")
shutil.copy2(KIT / "site/vllm/renderers/dsv41_incr_tok.py", site("spark-eight") / "dsv41_incr_tok.py")
(ROOT / "spark-one/.svc").write_text("active\n")
rc, out = run("bash", "ops/site8.sh", "uninstall")
check(rc == 1 and "spark-one" in out, "site8 uninstall refuses while a host runs")
(ROOT / "spark-one/.svc").unlink()
rc, out = run("bash", "ops/site8.sh", "uninstall")
check(rc == 0 and "state base" in out, "site8 uninstall 8/8 base")
check(all((site(h) / "deepseek_v4.py").read_bytes() == BASE and not (site(h) / "dsv41_incr_tok.py").exists()
          and not (site(h) / "deepseek_v4.py.bak-incrtok").exists() for h in HOSTS), "uninstall restored every host byte for byte")

for args, what in ((["x", "FOO=1"], "foreign flag"), (["x", "DSV41_INCR_TOK=2"], "bad flag value"),
                   (["x", "DSV41_INCR_TOK_VERIFY_FIRST=3"], "tunable without the flag"), (["a b"], "bad label"),
                   (["x", "DSV41_INCR_TOK=1", "DSV41_INCR_TOK_LOG=/tmp/x"], "log path override")):
    rc, out = run("bash", "ops/arm.sh", *args)
    check(rc == 2, f"arm.sh usage error exits 2 before any ssh: {what}")

# ---------------------------------------------------------------- verdict scripts
sys.path.insert(0, str(KIT / "ops"))
import ab_compare  # noqa: E402
import arm_check  # noqa: E402


def arm_dir(mode, **bad):
    d = Path(tempfile.mkdtemp(dir=ROOT))
    for n in ("exact", "g1", "det", "ab"):
        (d / f"{n}.rc").write_text("0\n")
    (d / "exact.json").write_text(json.dumps({"combined": "a8c3394af9d28249" + "0" * 48}))
    (d / "g1.jsonl").write_text(json.dumps({"metric": "G1_summary", "outputs_digest": "9a18f802d930f6e7aa", "G1_mean_tok_s": 113.8}) + "\n")
    (d / "det.jsonl").write_text("".join(json.dumps({"case": c, "tok_sha": s + "ffff"}) + "\n" for c, s in arm_check.LONG_REF.items() for _ in range(4)))
    (d / "ab.jsonl").write_text("".join(json.dumps({"step": i, "ids_sha": "ab", "prompt_tokens": 230000 + i}) + "\n" for i in range(3)))
    (d / "errors.rank0").write_text("0\n")
    (d / "rank0.incr").write_text("" if mode == "off" else f"INFO DSV41_INCR_TOK={mode} enabled (incr-tok v1): ...\n")
    for k, v in bad.items():
        (d / k).write_text(v)
    return str(d)


def verdict(d, mode):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return arm_check.main(d, mode)


check(verdict(arm_dir("off"), "off") == 0, "arm_check: good off arm passes")
check(verdict(arm_dir("on"), "on") == 0, "arm_check: good on arm passes")
check(verdict(arm_dir("audit"), "audit") == 0, "arm_check: good audit arm passes")
for k, v, mode, what in (
        ("exact.json", json.dumps({"combined": "deadbeef"}), "off", "exact anchor"),
        ("g1.jsonl", json.dumps({"metric": "G1_summary", "outputs_digest": "0000"}) + "\n", "off", "G1 digest"),
        ("det.jsonl", json.dumps({"case": "big:2500", "tok_sha": "00000000"}) + "\n", "off", "long prompts"),
        ("errors.rank0", "2\n", "off", "rank0 errors"),
        ("ab.rc", "1\n", "off", "ttft_ab exit"),
        ("rank0.incr", "INFO DSV41_INCR_TOK=on enabled\n", "off", "overlay logging with the flag off"),
        ("rank0.incr", "WARNING DSV41_INCR_TOK=on requested but DISABLED (stock tokenize): x\n", "on", "overlay disabled"),
        ("rank0.incr", "INFO DSV41_INCR_TOK=on enabled\nERROR DSV41_INCR_TOK MISMATCH (verify)\n", "on", "overlay mismatch")):
    check(verdict(arm_dir(mode, **{k: v}), mode) == 1, f"arm_check fails on: {what}")


def ab_files(off, on, log=None):
    d = Path(tempfile.mkdtemp(dir=ROOT))
    (d / "off").write_text("".join(json.dumps(r) + "\n" for r in off))
    (d / "on").write_text("".join(json.dumps(r) + "\n" for r in on))
    args = ["x", str(d / "off"), str(d / "on")]
    if log is not None:
        (d / "log").write_text("".join(json.dumps(r) + "\n" for r in log))
        args.append(str(d / "log"))
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return ab_compare.main(args)


steps_off = [{"step": i, "ids_sha": f"s{i}", "prompt_tokens": 230000 + 2000 * i, "out_sha": "o", "ttft_s": 1.5} for i in range(40)]
steps_on = [dict(r, ttft_s=1.5 if i < 20 else 0.8) for i, r in enumerate(steps_off)]
log = [{"kind": "miss_none", "tokens": 230000}] + [{"kind": "verified" if i < 20 else "hit", "tokens": 230000 + 2000 * i, "ms_full": 430, "ms_incr": 10}
                                                 for i in range(1, 40)]
check(ab_files(steps_off, steps_on, log) == 0, "ab_compare: identical ids + hits passes")
check(ab_files(steps_off, [dict(r, ids_sha="x") if r["step"] == 7 else r for r in steps_on], log) == 1, "ab_compare fails on one differing prompt id hash")
check(ab_files(steps_off, steps_on[:-1], log) == 1, "ab_compare fails on a missing step")
check(ab_files(steps_off, steps_on, log + [{"kind": "MISMATCH"}]) == 1, "ab_compare fails on a mismatch record")
check(ab_files(steps_off, steps_on, [{"kind": "verified", "tokens": 1}]) == 1, "ab_compare fails with too few hits")

shutil.rmtree(ROOT, ignore_errors=True)
print(f"VERDICT {'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failures")
sys.exit(0 if not FAILS else 1)
