#!/usr/bin/env python3
"""g130 boundary kit, ops test. Stdlib only, no torch, no GPU, no network: a stub `ssh` first on PATH runs each "remote" command locally with HOME set to a scratch
folder per host, and the real boundary_install.py, site8.sh, stage8.sh, arm.sh and arm_check.py run against those. It proves the scripts FAIL when they should.
It does not prove the real hosts, the supervisor or the benchmarks behave like the stubs. Light enough for the Mac:

  python3 -I tests/test_ops_stub.py

Writes only under results/ops-stub, removed at the end.
"""
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
ROOT = KIT / "results" / "ops-stub"
HOSTS = ("spark-one", "spark-two", "spark-three", "spark-four", "spark-five", "spark-six", "spark-seven", "spark-eight")
DSPARK = "models/deepseek_v41/nvidia/dspark.py"
SPEC = "v1/worker/gpu/spec_decode/dspark/speculator.py"
SIDE = "models/deepseek_v41/nvidia/dsv41_boundary.py"
EXACT = "a8c3394af9d28249" + "0" * 48
FAIL = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{' ' + detail if detail and not ok else ''}", flush=True)
    if not ok:
        FAIL.append(name)


def script(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


STUB_SSH = r"""#!/bin/bash
# Stub ssh: options, then host, then the remote command. STUB_ssh_<host> = "down" fails like an unreachable host, "silent" exits 0 and prints nothing.
while [ $# -gt 0 ]; do case "$1" in -n) shift ;; -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
h=$1; shift
var=STUB_ssh_$(echo "$h" | tr - _)
case "${!var:-}" in
  down) echo "ssh: connect to host $h: timed out" >&2; exit 255 ;;
  silent) cat >/dev/null 2>&1 < /dev/null; exit 0 ;;
esac
export HOME=$STUB_ROOT/$h
export BOUNDARY_SITE=$HOME/venvs/vllm032/lib/python3.12/site-packages/vllm
export STUB_HOST=$h
exec bash -c "$*"
"""


def prepare():
    shutil.rmtree(ROOT, ignore_errors=True)
    bin_ = ROOT / "bin"
    script(bin_ / "ssh", STUB_SSH)
    script(bin_ / "hostname", '#!/bin/sh\necho "$STUB_HOST"\n')
    script(bin_ / "sha256sum", '#!/bin/bash\nexec shasum -a 256 "$@"\n')
    script(bin_ / "timeout", '#!/bin/bash\nshift\nexec "$@"\n')
    script(bin_ / "systemctl", '#!/bin/bash\ncat "$HOME/service-state" 2>/dev/null\n')
    for h in HOSTS:
        reset_host(h)


def reset_host(h):
    home = ROOT / h
    shutil.rmtree(home, ignore_errors=True)
    site = home / "venvs/vllm032/lib/python3.12/site-packages/vllm"
    for rel, name in ((DSPARK, "dspark.py"), (SPEC, "speculator.py")):
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(KIT / "base" / name, site / rel)
    (home / "service-state").write_text("inactive\n")
    stage = home / "dsv41-stage/g130"
    stage.mkdir(parents=True)
    os.symlink(KIT, stage / "boundary")


def run(*args, env=None, cwd=KIT):
    e = dict(os.environ, PATH=f"{ROOT / 'bin'}:{os.environ['PATH']}", STUB_ROOT=str(ROOT), TMPDIR=str(ROOT))
    e.update(env or {})
    r = subprocess.run(args, cwd=cwd, env=e, capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def site8(*args, env=None):
    return run("bash", str(KIT / "ops/site8.sh"), *args, env=env)


def site(h):
    return ROOT / h / "venvs/vllm032/lib/python3.12/site-packages/vllm"


def reset_all():
    for h in HOSTS:
        reset_host(h)


def test_site8():
    reset_all()
    rc, out = site8("install", env={"STUB_ssh_" + h.replace("-", "_"): "silent" for h in HOSTS})
    check("site8 all hosts silent exit 0 is a FAIL (the old false 8/8)", rc == 1 and "FAILED" in out and "8/8" not in out, out)

    rc, out = site8("install", env={"STUB_ssh_spark_five": "down"})
    check("site8 one host unreachable", rc == 1 and "FAIL spark-five: exit 255" in out and "8/8" not in out, out)

    reset_all()
    (ROOT / "spark-three/service-state").write_text("active\n")
    rc, out = site8("install")
    check("site8 service active on one host", rc == 1 and "FAIL spark-three: exit 1" in out, out)
    check("site8 refused host wrote nothing", not (site("spark-three") / SIDE).exists())

    reset_all()
    (ROOT / "spark-two/service-state").write_text("")
    rc, out = site8("install")
    check("site8 unreadable service state fails closed", rc == 1 and "FAIL spark-two" in out, out)

    reset_all()
    with open(site("spark-six") / DSPARK, "a") as f:
        f.write("# moved\n")
    rc, out = site8("install")
    check("site8 moved base on one host", rc == 1 and "FAIL spark-six: exit 1" in out and "kit expects base" in out, out)

    reset_all()
    rc, out = site8("status", "patched")
    check("site8 status patched on a base site", rc == 1, out)
    rc, out = site8("status", "base")
    check("site8 status base on a base site", rc == 0 and "8/8" in out and "state base" in out, out)
    rc, out = site8("install")
    check("site8 install on 8", rc == 0 and "8/8 exit 0, identical, state patched" in out, out)
    rc, out = site8("status", "patched")
    check("site8 status patched after install", rc == 0 and "state patched" in out, out)
    rc, out = site8("status")
    check("site8 status with no state", rc == 0, out)

    good = (site("spark-four") / SIDE).read_bytes()
    (site("spark-four") / SIDE).write_bytes(good + b"\n")
    rc, out = site8("status", "patched")
    check("site8 stale side file on one host", rc == 1 and "FAIL spark-four" in out, out)
    (site("spark-four") / SIDE).write_bytes(good)

    (ROOT / "spark-one/service-state").write_text("active\n")
    rc, out = site8("uninstall")
    check("site8 uninstall refused on one host", rc == 1 and "FAIL spark-one" in out, out)
    (ROOT / "spark-one/service-state").write_text("inactive\n")
    rc, out = site8("uninstall")
    check("site8 uninstall", rc == 0 and "state base" in out, out)

    rc, out = site8("status", "nonsense")
    check("site8 bad state is a usage error", rc == 2, out)
    rc, out = site8("reinstall")
    check("site8 bad command is a usage error", rc == 2, out)


def test_stage8():
    reset_all()
    for h in HOSTS:   # stage into a real folder per host
        (ROOT / h / "dsv41-stage/g130/boundary").unlink()
    rc, out = run("bash", str(KIT / "ops/stage8.sh"))
    check("stage8 on 8", rc == 0 and "stage8: 8/8 staged OK" in out, out)
    with open(site("spark-seven") / SPEC, "a") as f:
        f.write("# moved\n")
    rc, out = run("bash", str(KIT / "ops/stage8.sh"))
    check("stage8 stops on a moved base", rc == 1 and "spark-seven failed" in out, out)
    rc, out = run("bash", str(KIT / "ops/stage8.sh"), env={"STUB_ssh_spark_one": "silent"})
    check("stage8 silent host", rc == 1, out)
    reset_all()


def fixture_home(h, i):
    """A fake host: supervisor, benchmarks and rank log. Spark-one runs the benchmarks."""
    home = ROOT / h
    logs = home / "dsv41-logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"rank{i}.log").write_text("startup\n")
    if i:
        return
    script(home / "dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh", REL_STUB)
    det = []
    cases = ["big:2500", "big:8000", "big:32000", "big:128000", "code:4000", "prose:6000", "l3:B:T16-spam-reply:1", "l3:B:T02-classify:1", "l3:A:T01-reviews:1"]
    for c in cases:
        for run_ in range(4):
            det.append({"case": c, "mode": "cold", "run": run_, "tok_sha": "t" + c, "lp_sha": "l" + c, "pt": 1})
        det.append({"summary": 1, "case": c, "mode": "cold", "distinct_tok": 1, "distinct_lp": 1})
    ref = home / "dsv41-g130ab/final3/det.jsonl"
    ref.parent.mkdir(parents=True)
    ref.write_text("".join(json.dumps(r) + "\n" for r in det))
    script(home / "dsv41-bench/exact.py", f"""import json,os,sys
if os.environ.get("STUB_NO_EXACT"): sys.exit(1)
c = os.environ.get("STUB_EXACT", "{EXACT}")
json.dump({{"combined": c}}, open(sys.argv[1], "w"))
""")
    script(home / "dsv41-bench/goal/goal_single.py", """import json,os,sys
a = sys.argv; p = a[a.index("--out") + 1]
d = os.environ.get("STUB_DIGEST", "9a18f802d930f6e7")
open(p, "a").write(json.dumps({"metric": "G1_summary", "G1_mean_tok_s": 113.8, "outputs_digest": d, "all_reps_identical": True}) + "\\n")
""")
    script(home / "dsv41-bench/goal/goal_conc4.py", """import json,os,sys
if os.environ.get("STUB_G4_DEAD"): sys.exit(3)
a = sys.argv; p = a[a.index("--out") + 1]
open(p, "a").write(json.dumps({"metric": "G4_summary", "runs": 3, "runs_ok": 3, "G4_per_session_min": 54, "G4_per_session_mean": 68}) + "\\n")
""")
    script(home / "dsv41-g5det/scripts/dbgrun.py", """import json,os,sys
if os.environ.get("STUB_DET_DEAD"): sys.exit(3)
out = sys.argv[2]; cases = sys.argv[4].split(",")
if os.environ.get("STUB_DET_SHORT"): cases = cases[:-1]
with open(out, "w") as f:
    for c in cases:
        tok = "tX" if os.environ.get("STUB_DET_DIFF") and c == "big:8000" else "t" + c
        for r in range(2):
            f.write(json.dumps({"case": c, "mode": "cold", "run": r, "tok_sha": tok, "lp_sha": "l" + c}) + "\\n")
        f.write(json.dumps({"summary": 1, "case": c, "distinct_tok": 1, "distinct_lp": 1}) + "\\n")
""")
    script(home / "dsv41-g130ab/load32.py", """import os
print("LOAD32 ok", os.environ.get("STUB_LOAD32", "32"), "/32 tokens 100 wall 1.0s aggregate 100 tok/s")
""")


def arm_env(reset=True):
    if reset:
        reset_all()
        for i, h in enumerate(HOSTS):
            fixture_home(h, i)
        rc, out = site8("install")
        assert rc == 0, out


def set_logs(*lines):
    for i, h in enumerate(HOSTS):
        (ROOT / h / f"dsv41-logs/rank{i}.log").write_text("startup\n" + "".join(x + "\n" for x in lines))


def set_audit(**over):
    for h in HOSTS:
        a = {"host": h, "status": "OK", "ctx_calls": 900, "ctx_words": 1, "ctx_mismatching_words": 0, "mk_positions": 20000, "mk_rows": 20000,
             "mk_pair_mismatching_words": 0, "mk_token_mismatches": 0, "mk_embed_mismatching_words": 0, "step": 4096}
        a.update(over)
        (ROOT / h / "dsv41-logs/boundary-audit.x.1.json").write_text(json.dumps(a))


def arm(*args, env=None):
    return run("bash", str(KIT / "ops/arm.sh"), *args, env=env)


FOLD_LINES = ("x DSV41_CTXKV_FOLD=1: rows pinned [1, 6], kept stock [], weight copy 8.10 MB", "x DSV41_MK_FOLD=1 tiles 16: self-check passed at M (1,), V 16160, tp 8")
AUDIT_LINES = ("x DSV41_CTXKV_FOLD=audit: rows pinned [1, 6], kept stock [], weight copy 8.10 MB", "x DSV41_MK_FOLD=audit tiles 16: self-check passed at M (1,), V 16160, tp 8")
T = ("DSV41_CTXKV_FOLD=1", "DSV41_MK_FOLD=1", "DSV41_MK_FOLD_TILES=16")
AUD = ("DSV41_CTXKV_FOLD=audit", "DSV41_MK_FOLD=audit", "DSV41_BOUNDARY_AUDIT_EVERY=64")
REL_STUB = r"""#!/bin/bash
# stub supervisor: records its arguments; exit $STUB_REL_RC.
echo "$*" >> "$STUB_ROOT/rel.args"
exit "${STUB_REL_RC:-0}"
"""
REL_PATH = "spark-one/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh"
AUDIT_PASS = "spark-one/dsv41-g130ab/boundary/audit/audit_pass.json"


def relaunch_writes_audit(**over):
    # arm.sh clears old reports before the restart, so the supervisor stub has to write the fresh ones
    a = {"status": "OK", "ctx_calls": 900, "ctx_words": 1, "ctx_mismatching_words": 0, "mk_positions": 20000, "mk_rows": 20000, "mk_pair_mismatching_words": 0,
         "mk_token_mismatches": 0, "mk_embed_mismatching_words": 0, "step": 4096}
    a.update(over)
    body = json.dumps(a)
    lines = "".join(f"echo '{body}' > \"$STUB_ROOT/{h}/dsv41-logs/boundary-audit.x.1.json\"\n" for h in HOSTS)
    script(ROOT / REL_PATH, "#!/bin/bash\n" + lines)


def tile_lines(lines, tiles):
    return tuple(x.replace("tiles 16:", f"tiles {tiles}:") for x in lines)


def pass_audit(*flags, lines=AUDIT_LINES):
    """A real audit arm through arm.sh, then the plain supervisor stub back."""
    relaunch_writes_audit()
    set_logs(*lines)
    rc, out = arm("audit", "--conc", *flags)
    assert rc == 0 and "ARM audit: PASS" in out, out
    script(ROOT / REL_PATH, REL_STUB)


def restarts():
    p = ROOT / "rel.args"
    return len(p.read_text().splitlines()) if p.exists() else 0


def refused(name, *args, want="ABORT"):
    n = restarts()
    rc, out = arm(*args)
    check(f"arm gate: {name}", rc == 1 and "ABORT" in out and want in out and restarts() == n, out)


def test_arm():
    arm_env()
    for label, args, why in (("t1", (), "t1 with an empty flag list"), ("c1", T, "c1 with flags"), ("audit", T, "audit with serving flags"), ("t1", AUD, "t1 with audit flags"),
                             ("x", (), "unknown label"), ("t1", ("DSV41_MK_FOLD_TILES=1",), "t1 with no fold flag"),
                             ("audit", AUD, "audit without --conc"), ("t1", ("DSV41_MK_FOLD=1", "DSV41_MK_FOLD_TILES="), "empty tiles, an unset $TILES"),
                             ("t1", ("DSV41_MK_FOLD=1", "DSV41_MK_FOLD_TILES=<best_tiles>"), "tiles placeholder left in"), ("t1", ("DSV41_MK_FOLD=1", "DSV41_MK_FOLD_TILES=8"), "tiles 8")):
        rc, out = arm(label, *args)
        check(f"arm usage: {why}", rc == 2, out)

    set_logs()
    rc, out = arm("c1")
    check("arm flags-off passes on good evidence", rc == 0 and "ARM c1: PASS" in out, out)
    args = (ROOT / "rel.args").read_text().strip().splitlines()[-1]
    check("arm passes launcher b3976c4b and clock 2000", "--expect-launcher b3976c4b DSV41_GPU_CLOCK_MHZ=2000" in args, args)

    pass_audit(*AUD)
    set_logs()
    rc, out = arm("t1", *T)
    check("arm fold flags with no fold line in the logs", rc == 1 and "this rank measured stock" in out, out)
    set_logs(*FOLD_LINES)
    rc, out = arm("t1", *T)
    check("arm fold passes with the lines on 8", rc == 0 and "ARM t1: PASS" in out, out)
    (ROOT / "spark-six/dsv41-logs/rank5.log").write_text("startup\n" + FOLD_LINES[0] + "\n")
    rc, out = arm("t1", *T)
    check("arm one rank fell back to stock", rc == 1 and "FAIL spark-six" in out, out)
    set_logs(*FOLD_LINES, "x DSV41_CTXKV_FOLD: M=48 kept on the stock path: layer 0 differs")
    rc, out = arm("t1", *T)
    check("arm a row kept stock", rc == 1 and "kept on the stock path" in out, out)
    set_logs(*FOLD_LINES)
    rc, out = arm("c1")
    check("arm fold line in a flags-off arm", rc == 1 and "fold line in a flags-off arm" in out, out)

    set_logs()
    for name, env, want in (("no exact result", {"STUB_NO_EXACT": "1"}, "FAIL exact"), ("exact differs", {"STUB_EXACT": "b" * 64}, "FAIL exact"),
                            ("digest differs", {"STUB_DIGEST": "dead"}, "FAIL g1"), ("G4 died", {"STUB_G4_DEAD": "1"}, "FAIL g4"),
                            ("long prompts died", {"STUB_DET_DEAD": "1"}, "FAIL long prompts"), ("long prompt missing", {"STUB_DET_SHORT": "1"}, "FAIL long prompts"),
                            ("long prompt differs", {"STUB_DET_DIFF": "1"}, "FAIL long prompts"), ("supervisor refused", {"STUB_REL_RC": "2"}, "")):
        rc, out = arm("c1", env=env)
        check(f"arm {name}", rc != 0 and want in out and "PASS\n" not in out.split("ARM")[-1], out)
    # a good earlier result under the same label must not pass a dead rerun
    rc, _ = arm("c9")
    rc, out = arm("c9", env={"STUB_NO_EXACT": "1"})
    check("arm stale exact.json is not reused", rc == 1 and "FAIL exact" in out, out)

    with open(site("spark-eight") / SPEC, "a") as f:
        f.write("# moved\n")
    n = len((ROOT / "rel.args").read_text().splitlines())
    rc, out = arm("c1")
    check("arm refuses a site not patched on 8/8, before any restart", rc == 1 and "ABORT" in out and len((ROOT / "rel.args").read_text().splitlines()) == n, out)

    arm_env()
    set_logs(*AUDIT_LINES)
    rc, out = arm("audit", "--conc", *AUD)
    check("arm audit with no report", rc == 1 and "no audit report" in out, out)

    relaunch_writes_audit()
    rc, out = arm("audit", "--conc", *AUD)
    check("arm audit passes on 8 clean reports", rc == 0 and "ARM audit: PASS" in out, out)
    relaunch_writes_audit(mk_token_mismatches=1, status="MISMATCH")
    rc, out = arm("audit", "--conc", *AUD)
    check("arm audit one mismatch", rc == 1 and "mk_token_mismatches 1" in out, out)
    relaunch_writes_audit(mk_positions=9999)
    rc, out = arm("audit", "--conc", *AUD)
    check("arm audit under 10000 positions", rc == 1 and "under 10000" in out, out)
    relaunch_writes_audit(ctx_calls=0)
    rc, out = arm("audit", "--conc", *AUD)
    check("arm audit ctx_calls 0", rc == 1 and "ctx_calls 0" in out, out)
    relaunch_writes_audit()
    rc, out = arm("audit", "--conc", *AUD, env={"STUB_LOAD32": "31"})
    check("arm load32 short", rc == 1 and "FAIL load32" in out, out)
    set_logs()
    relaunch_writes_audit()
    rc, out = arm("c1")
    check("arm audit report in a flags-off arm", rc == 1 and "audit report in a non-audit arm" in out, out)

    rc, out = run("bash", str(KIT / "ops/arm.sh"), "c1", env={"STUB_ssh_spark_one": "down"})
    check("arm unreachable host", rc == 1 and "ABORT" in out, out)


def test_audit_gate():
    """A t arm must serve what a passing audit arm on this kit ran. Refusals happen before any restart and leave the audit reports alone."""
    T1 = ("DSV41_CTXKV_FOLD=1", "DSV41_MK_FOLD=1", "DSV41_MK_FOLD_TILES=1")
    MK_ONLY = ("DSV41_MK_FOLD=1", "DSV41_MK_FOLD_TILES=16")
    CTX_ONLY = ("DSV41_CTXKV_FOLD=1",)
    arm_env()
    set_logs(*FOLD_LINES)
    report = ROOT / "spark-three/dsv41-logs/boundary-audit.keep.json"
    report.write_text("{}")
    refused("t arm with no audit arm", "t1", *T, want="no passing audit arm")
    check("arm gate: a refused t arm keeps the audit reports", report.exists())

    # audit default 16: serving 16 passes whether spelled out or left to the default, serving 1 is refused
    pass_audit(*AUD)
    refused("audit 16, t tiles 1", "t1", *T1, want="tiles same as audit")
    set_logs(*FOLD_LINES)
    rc, out = arm("t1", *T)
    check("arm gate: audit default 16, t tiles 16", rc == 0 and "ARM t1: PASS" in out, out)
    rc, out = arm("t2", "DSV41_CTXKV_FOLD=1", "DSV41_MK_FOLD=1")
    check("arm gate: audit default 16, t default 16", rc == 0 and "ARM t2: PASS" in out, out)

    # audit tiles 1: the reviewer's case, serving 1 needs an audit at 1
    pass_audit(*AUD, "DSV41_MK_FOLD_TILES=1", lines=tile_lines(AUDIT_LINES, 1))
    refused("audit 1, t tiles 16", "t1", *T, want="tiles same as audit")
    refused("audit 1, t default 16", "t1", "DSV41_CTXKV_FOLD=1", "DSV41_MK_FOLD=1", want="tiles same as audit")
    set_logs(*tile_lines(FOLD_LINES, 1))
    rc, out = arm("t1", *T1)
    check("arm gate: audit 1, t tiles 1", rc == 0 and "ARM t1: PASS" in out, out)
    set_logs(*FOLD_LINES)
    rc, out = arm("t1", *T1)
    check("arm tiles asked differ from tiles logged", rc == 1 and "tiles 1: self-check passed" in out, out)

    # a part the audit arm left out is not covered
    pass_audit("DSV41_CTXKV_FOLD=audit", lines=AUDIT_LINES[:1])
    refused("ctx-only audit, t both", "t1", *T, want="Markov audited")
    refused("ctx-only audit, t Markov", "t1", *MK_ONLY, want="Markov audited")
    set_logs(FOLD_LINES[0])
    rc, out = arm("t1", *CTX_ONLY)
    check("arm gate: ctx-only audit, ctx-only t", rc == 0 and "ARM t1: PASS" in out, out)
    pass_audit("DSV41_MK_FOLD=audit", lines=AUDIT_LINES[1:])
    refused("Markov-only audit, t ctx", "t1", *CTX_ONLY, want="context KV audited")
    set_logs(FOLD_LINES[1])
    rc, out = arm("t1", *MK_ONLY)
    check("arm gate: Markov-only audit, Markov-only t", rc == 0 and "ARM t1: PASS" in out, out)

    # split and a moved tactic map run other kernels than the audit did
    pass_audit(*AUD)
    refused("split t arm", "t1", "DSV41_CTXKV_FOLD=split", want="split has no audit path")
    refused("tactic map differs", "t1", *T, "DSV41_CTXKV_TACTIC_MAP=1-192:1", want="tactic map same as audit")

    # the record is bound to these kit bytes and survives only a passing audit
    rec = ROOT / AUDIT_PASS
    good = rec.read_text()
    rec.write_text(good.replace('"digest": "', '"digest": "old'))
    refused("audit from another kit digest", "t1", *T, want="audit arm on this kit")
    rec.write_text("not json")
    refused("unreadable audit record", "t1", *T, want="no passing audit arm")
    rec.write_text(good)
    relaunch_writes_audit(mk_token_mismatches=1, status="MISMATCH")
    set_logs(*AUDIT_LINES)
    rc, out = arm("audit", "--conc", *AUD)
    check("arm gate: failed audit arm writes no record and clears the old one", rc == 1 and not rec.exists(), out)
    script(ROOT / REL_PATH, REL_STUB)
    refused("t arm after a failed audit", "t1", *T, want="no passing audit arm")


def main():
    assert not (Path.home() / "dsv41-tp8.sh").exists(), "this is a Spark: run it on the Mac"
    prepare()
    which = run("bash", "-c", "command -v ssh")[1].strip()
    assert which == str(ROOT / "bin/ssh"), which   # never reach a real host
    test_site8()
    test_stage8()
    test_arm()
    test_audit_gate()
    shutil.rmtree(ROOT, ignore_errors=True)
    print("VERDICT", "FAIL " + "; ".join(FAIL) if FAIL else "PASS")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
