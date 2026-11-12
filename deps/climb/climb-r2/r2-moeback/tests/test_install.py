#!/usr/bin/env python3
"""Installer / anchor / flag-off tests (stdlib only).

1. Every anchor applies exactly once to the LIVE base files (copied from spark-one, sha256 in KIT.json),
   the result compiles, and the exact inverse edit restores the base bytes.
2. install -> status -> install (idempotent) -> uninstall on a scratch tree; after uninstall every target is
   byte-identical to its base; a target with an unknown base sha is refused; new files appear/disappear.
3. Flag-off AST check: every tagged line sits inside an `if` whose test is the lever flag (or is the flag
   definition / the flag-first `if _FLAG and ...` whose else branch is the original code).
4. Athena: the launcher-pinned trial overlay and deep_gemm_moe.py are not targets; the trial's own
   transform_apply() (fail-closed anchors) still applies to DeepGemmFP4Experts.apply of the live base.
5. Triton: the _gated_gather kernel source block (lines 1..51) of the patched gather module is byte-identical
   and at the same line numbers (flag-off kernel cache key and binary unchanged).
6. sitecustomize (G18): run the patched file in subprocesses with a fake HOME: flag off -> the trial is
   installed with ATHENA_FC2_GRAPH_SO untouched; flag on + good .so/sha -> GRAPH_SO points at the PDL .so
   and DSV41_MOE_PDL_TRIGGER=start; flag on + sha mismatch / missing .so -> exit 97 (fail closed);
   explicit DSV41_MOE_PDL_TRIGGER=0 is respected.
7. (review fixes) preflight: a bad base on a LATER target refuses the whole install with nothing written;
   PDL binary gate on a temp kit copy: build/*.so with KIT.json pdl_binary_sha256 null -> refused (nothing
   written); sha mismatch -> refused; vetted but no/failed/quick/other-sha device stamp -> binary skipped and a
   stale installed copy removed; vetted + passing stamp -> installed with a matching .sha256 sidecar.
8. vet-pdl on synthetic build_pdl.sh outputs: STRICT and SCHEDULE-ONLY accepted (KIT.json gets the sha), and
   refused (KIT.json unchanged) for: live != control, an extra/changed opcode, no ACQBULK, ACQBULK already in
   control, a global store before the wait, a live .so that is not the pinned production binary, a .so that
   is not the one the log hashed.
Run: python3 test_install.py <kit_dir>
"""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

KIT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
B = os.path.join(KIT, "base")
FAILS = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def make_tree(root):
    v = os.path.join(root, "vllm/model_executor/layers/fused_moe/runner")
    f = os.path.join(root, "fc2/site")
    os.makedirs(v)
    os.makedirs(f)
    shutil.copy2(os.path.join(B, "model_executor_layers_fused_moe_runner_moe_runner.py"), os.path.join(v, "moe_runner.py"))
    shutil.copy2(os.path.join(B, "fc2_status_gated_gather_trial.py"), os.path.join(root, "fc2"))
    shutil.copy2(os.path.join(B, "sitecustomize.py"), f)
    return {"moe_runner": os.path.join(v, "moe_runner.py"),
            "gather": os.path.join(root, "fc2/fc2_status_gated_gather_trial.py"),
            "site": os.path.join(f, "sitecustomize.py")}


def inst(root, cmd):
    env = dict(os.environ, R2MB_VLLM=os.path.join(root, "vllm"), R2MB_FC2=os.path.join(root, "fc2"))
    return subprocess.run([sys.executable, os.path.join(KIT, "install.py"), cmd], env=env, capture_output=True, text=True)


def main():
    sys.path.insert(0, KIT)
    os.environ.setdefault("R2MB_VLLM", "/nonexistent")
    import install as I
    kit = json.load(open(os.path.join(KIT, "KIT.json")))
    bases = {e["key"]: e["base_sha256"] for e in kit["files"] if e["action"] == "patch"}
    live = {"moe_runner": "model_executor_layers_fused_moe_runner_moe_runner.py",
            "gather": "fc2_status_gated_gather_trial.py", "site": "sitecustomize.py"}
    edits = {"moe_runner": I.RUNNER_EDITS, "gather": I.GATHER_EDITS, "site": I.SITE_EDITS}
    tags = {"moe_runner": I.T25, "gather": I.T25, "site": I.T18}

    # 1. anchors on live bases
    for key, fn in live.items():
        p = os.path.join(B, fn)
        src = open(p).read()
        check(f"[{key}] live base sha256 recorded in KIT.json", sha(p) in bases[key], sha(p)[:16])
        for old, new, count in edits[key]:
            check(f"[{key}] anchor x{count}: {old.strip().splitlines()[0][:60]!r}", src.count(old) == count)
        new = I.apply_edits(src, edits[key], key)
        compile(new, key, "exec")
        check(f"[{key}] patched compiles; {new.count(tags[key])} tagged lines", True)
        check(f"[{key}] inverse edit restores base bytes", I.revert_edits(new, edits[key], key) == src)
        check(f"[{key}] every changed line carries the tag",
              all(tags[key] in l for l in set(new.splitlines()) - set(src.splitlines()) if l.strip()),
              str([l for l in set(new.splitlines()) - set(src.splitlines()) if l.strip() and tags[key] not in l]))

    # 2. install / idempotent / uninstall / refuse
    with tempfile.TemporaryDirectory() as root:
        t = make_tree(root)
        before = {k: sha(p) for k, p in t.items()}
        r = inst(root, "install")
        check("install rc 0", r.returncode == 0, r.stderr[-300:])
        fold = os.path.join(root, "vllm/model_executor/layers/fused_moe/runner/dsv41_moe_fold.py")
        check("new module copied", os.path.exists(fold))
        check("all three targets patched", all(sha(p) != before[k] for k, p in t.items()))
        mid = {k: sha(p) for k, p in t.items()}
        r = inst(root, "install")
        check("second install is a no-op", r.returncode == 0 and {k: sha(p) for k, p in t.items()} == mid)
        r = inst(root, "status")
        check("status lists PATCHED x3", r.stdout.count("PATCHED") == 3, r.stdout)
        r = inst(root, "uninstall")
        check("uninstall rc 0", r.returncode == 0, r.stderr[-300:])
        check("uninstall restores base bytes", {k: sha(p) for k, p in t.items()} == before)
        check("new module removed", not os.path.exists(fold))
        with open(t["moe_runner"], "a") as f:
            f.write("# foreign edit\n")
        r = inst(root, "install")
        check("unknown base sha refused", r.returncode != 0 and "refusing" in (r.stdout + r.stderr))
        # composition: another kit's tagged edit made AFTER ours survives our uninstall
        t2root = os.path.join(root, "t2")
        t2 = make_tree(t2root)
        inst(t2root, "install")
        with open(t2["moe_runner"], "a") as f:
            f.write("_OTHER_KIT = 1  # r1-other\n")
        r = inst(t2root, "uninstall")
        txt = open(t2["moe_runner"]).read()
        check("uninstall keeps a later kit's edit", r.returncode == 0 and "# r1-other" in txt and I.T25 not in txt)

    # 3. flag-off AST check
    flag_names = {"_DSV41_FC2_GATHER_ADD"}
    for key, fn in live.items():
        new = I.apply_edits(open(os.path.join(B, fn)).read(), edits[key], key)
        tree = ast.parse(new)
        lines = new.splitlines()
        tagged = {i + 1 for i, l in enumerate(lines) if tags[key] in l and not l.strip().startswith("#")}
        parents = {}
        for node in ast.walk(tree):
            for ch in ast.iter_child_nodes(node):
                parents[ch] = node

        def flag_test(test):
            if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
                test = test.values[0]
            if isinstance(test, ast.Name) and test.id in flag_names:
                return True
            s = ast.unparse(test)
            return 'DSV41_FC2_PDL' in s and s.endswith("== '1'")

        ok = True
        for node in ast.walk(tree):
            if not isinstance(node, ast.stmt) or node.lineno not in tagged:
                continue
            good = False
            if isinstance(node, ast.Assign) and any(isinstance(tg, ast.Name) and tg.id in flag_names for tg in node.targets):
                good = True
            if isinstance(node, ast.If) and flag_test(node.test):
                good = True
            par = parents.get(node)
            if (isinstance(node, ast.If) and isinstance(par, ast.If) and flag_test(par.test) and par.orelse == [node]
                    and not any(getattr(n, "lineno", 0) in tagged for b in node.body for n in ast.walk(b))):
                good = True  # the original `if` turned `elif` under a flag-first test: flag off runs it unchanged
            p = parents.get(node)
            while p is not None and not good:
                if isinstance(p, ast.If) and flag_test(p.test) and node not in p.orelse:
                    # inside the flag-true body
                    good = any(node is n or any(node is m for m in ast.walk(n)) for n in p.body)
                p = parents.get(p)
            if not good:
                ok = False
                print("   ungated tagged statement:", ast.unparse(node)[:100])
        check(f"[{key}] every tagged statement is the flag definition or flag-gated", ok)

    # 4. Athena anchors still apply; pinned files untouched
    trial_src = open(os.path.join(B, "athena_fc2_forty_layer_trial.py")).read()
    check("trial overlay sha unchanged (launcher pin ae2089724597)", sha(os.path.join(B, "athena_fc2_forty_layer_trial.py")).startswith("ae2089724597"))
    ns = {}
    exec(compile(trial_src, "trial", "exec"), ns)
    dg = open(os.path.join(B, "model_executor_layers_fused_moe_experts_deep_gemm_moe.py")).read()
    mod = ast.parse(dg)
    cls = next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == "DeepGemmFP4Experts")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "apply")
    seg = "\n".join(dg.splitlines()[fn.lineno - 1 if not fn.decorator_list else fn.decorator_list[0].lineno - 1: fn.end_lineno]) + "\n"
    try:
        out = ns["transform_apply"](seg)
        check("Athena transform_apply anchors apply to live DeepGemmFP4Experts.apply", "athena_fc2_gather" in out)
    except Exception as e:
        check("Athena transform_apply anchors apply to live DeepGemmFP4Experts.apply", False, repr(e))
    targets = {os.path.basename(p) for p, _, _ in I.TARGETS.values()}
    check("kit targets exclude trial overlay and deep_gemm_moe.py",
          not targets & {"athena_fc2_forty_layer_trial.py", "deep_gemm_moe.py", "dsv41-tp8.sh"}, str(targets))

    # 5. Triton kernel block unchanged and in place
    g0 = open(os.path.join(B, "fc2_status_gated_gather_trial.py")).read().splitlines()
    g1 = I.apply_edits("\n".join(g0) + "\n", I.GATHER_EDITS, "gather").splitlines()
    end = next(i for i, l in enumerate(g0) if l.startswith("@torch.no_grad()"))
    check(f"_gated_gather kernel lines 1..{end - 1} identical and unmoved", g0[:end - 1] == g1[:end - 1])

    # 6. sitecustomize behaviour
    site_new = I.apply_edits(open(os.path.join(B, "sitecustomize.py")).read(), I.SITE_EDITS, "site")
    with tempfile.TemporaryDirectory() as home:
        fc2 = os.path.join(home, "dsv41-fc2")
        os.makedirs(os.path.join(fc2, "site"))
        open(os.path.join(fc2, "site", "sitecustomize.py"), "w").write(site_new)
        open(os.path.join(fc2, "athena_fc2_forty_layer_trial.py"), "w").write(textwrap.dedent("""
            import os
            def install():
                print("TRIAL", os.environ.get("ATHENA_FC2_GRAPH_SO"), os.environ.get("DSV41_MOE_PDL_TRIGGER"))
        """))
        so = os.path.join(fc2, "athena_tp8_fc2_packed_graph_pdl_sm121.so")
        open(so, "wb").write(b"\x7fELF fake")
        open(so + ".sha256", "w").write(sha(so) + "  x\n")

        def run(extra):
            env = {"HOME": home, "PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.path.join(fc2, "site"),
                   "ATHENA_FC2_SERVING_TRIAL": "1", "ATHENA_FC2_GRAPH_SO": "/live.so"}
            env.update(extra)
            return subprocess.run([sys.executable, "-c", "pass"], env=env, capture_output=True, text=True)

        r = run({})
        check("site flag off: trial installed, GRAPH_SO untouched, no trigger", r.returncode == 0 and "TRIAL /live.so None" in r.stdout, r.stdout + r.stderr[-200:])
        r = run({"DSV41_FC2_PDL": "1"})
        check("site flag on: GRAPH_SO -> pdl .so, trigger=start", r.returncode == 0 and f"TRIAL {so} start" in r.stdout, r.stdout + r.stderr[-200:])
        r = run({"DSV41_FC2_PDL": "1", "DSV41_MOE_PDL_TRIGGER": "0"})
        check("site flag on: explicit DSV41_MOE_PDL_TRIGGER=0 kept", r.returncode == 0 and f"TRIAL {so} 0" in r.stdout, r.stdout)
        open(so + ".sha256", "w").write("0" * 64 + "  x\n")
        r = run({"DSV41_FC2_PDL": "1"})
        check("site flag on + sha mismatch: exit 97", r.returncode == 97, str(r.returncode))
        os.remove(so)
        r = run({"DSV41_FC2_PDL": "1"})
        check("site flag on + missing .so: exit 97", r.returncode == 97, str(r.returncode))
        r = run({"DSV41_FC2_PDL": "0"})
        check("site flag off with missing .so: still boots", r.returncode == 0 and "TRIAL /live.so None" in r.stdout)

    test_gates(I)
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


def temp_kit(root):
    k = os.path.join(root, "kit")
    os.makedirs(os.path.join(k, "files/vllm/model_executor/layers/fused_moe/runner"))
    shutil.copy2(os.path.join(KIT, "install.py"), k)
    shutil.copy2(os.path.join(KIT, "KIT.json"), k)
    shutil.copy2(os.path.join(KIT, "files/vllm/model_executor/layers/fused_moe/runner/dsv41_moe_fold.py"),
                 os.path.join(k, "files/vllm/model_executor/layers/fused_moe/runner"))
    j = json.load(open(os.path.join(k, "KIT.json")))
    j["pdl_binary_sha256"] = None
    j.pop("pdl_vet", None)
    json.dump(j, open(os.path.join(k, "KIT.json"), "w"), indent=1)
    return k


def kinst(k, root, *args):
    env = dict(os.environ, R2MB_VLLM=os.path.join(root, "vllm"), R2MB_FC2=os.path.join(root, "fc2"))
    return subprocess.run([sys.executable, os.path.join(k, "install.py"), *args], env=env, capture_output=True,
                          text=True)


def set_kit(k, **kv):
    p = os.path.join(k, "KIT.json")
    j = json.load(open(p))
    j.update(kv)
    json.dump(j, open(p, "w"), indent=1)


def test_gates(I):
    good_stamp = {"pass": True, "device_test_rc": 0, "racecheck_rc": 0, "synccheck_rc": 0, "reps": 200,
                  "graph_cases_ok": 8}
    # 7a preflight: later-target mismatch -> nothing written
    with tempfile.TemporaryDirectory() as root:
        t = make_tree(root)
        k = temp_kit(root)
        with open(t["site"], "a") as f:
            f.write("# foreign\n")
        before = {key: sha(p) for key, p in t.items()}
        r = kinst(k, root, "install")
        check("preflight: bad base on the last target refuses the whole install",
              r.returncode != 0 and "preflight failed" in r.stderr and "refusing" in r.stderr, r.stderr[-200:])
        check("preflight: no target written, no backup, no new module",
              {key: sha(p) for key, p in t.items()} == before
              and not any(os.path.exists(p + I.BAK) for p in t.values())
              and not os.path.exists(os.path.join(root, "vllm/model_executor/layers/fused_moe/runner/dsv41_moe_fold.py")))
    # 7b PDL gate
    with tempfile.TemporaryDirectory() as root:
        t = make_tree(root)
        k = temp_kit(root)
        before = {key: sha(p) for key, p in t.items()}
        dst = os.path.join(root, "fc2/athena_tp8_fc2_packed_graph_pdl_sm121.so")
        r = kinst(k, root, "install")
        check("no build/ .so: install rc 0, binary skipped with warning",
              r.returncode == 0 and "WARNING: G18 binary not installed" in r.stdout and not os.path.exists(dst), r.stdout[-300:])
        kinst(k, root, "uninstall")
        check("uninstall after skip restores bases", {key: sha(p) for key, p in t.items()} == before)
        os.makedirs(os.path.join(k, "build"))
        so = os.path.join(k, "build/athena_tp8_fc2_packed_graph_pdl_sm121.so")
        open(so, "wb").write(b"\x7fELF unvetted")
        r = kinst(k, root, "install")
        check("build/ .so with pdl_binary_sha256 null: install REFUSED",
              r.returncode != 0 and "pdl_binary_sha256 is null" in r.stderr, r.stderr[-200:])
        check("  ... and nothing written (targets, module, binary)",
              {key: sha(p) for key, p in t.items()} == before and not os.path.exists(dst)
              and not os.path.exists(os.path.join(root, "vllm/model_executor/layers/fused_moe/runner/dsv41_moe_fold.py")))
        set_kit(k, pdl_binary_sha256="0" * 64)
        r = kinst(k, root, "install")
        check("build/ .so sha != KIT.json: install REFUSED, nothing written",
              r.returncode != 0 and "!= KIT.json pdl_binary_sha256" in r.stderr
              and {key: sha(p) for key, p in t.items()} == before, r.stderr[-200:])
        set_kit(k, pdl_binary_sha256=sha(so))
        open(dst, "wb").write(b"stale")
        open(dst + ".sha256", "w").write(sha(dst) + "  x\n")
        r = kinst(k, root, "install")
        check("vetted but no device stamp: install rc 0, binary skipped, stale installed copy removed",
              r.returncode == 0 and "no device gate stamp" in r.stdout and not os.path.exists(dst)
              and not os.path.exists(dst + ".sha256"), r.stdout[-300:])
        stamp_p = os.path.join(k, "build/DEVICE-G18.json")
        for label, st in (("failed racecheck", dict(good_stamp, racecheck_rc=99)),
                          ("pass=false", dict(good_stamp, **{"pass": False})),
                          ("quick reps", dict(good_stamp, reps=20)),
                          ("too few graph cases", dict(good_stamp, graph_cases_ok=2)),
                          ("stamp for another sha", dict(good_stamp, pdl_so_sha256="1" * 64))):
            st.setdefault("pdl_so_sha256", sha(so))
            json.dump(st, open(stamp_p, "w"))
            r = kinst(k, root, "install")
            check(f"stamp {label}: binary skipped", r.returncode == 0 and "does not pass" in r.stdout
                  and not os.path.exists(dst), r.stdout[-200:])
        json.dump(dict(good_stamp, pdl_so_sha256=sha(so)), open(stamp_p, "w"))
        r = kinst(k, root, "install")
        check("vetted + passing device stamp: binary installed with matching sidecar",
              r.returncode == 0 and os.path.exists(dst) and sha(dst) == sha(so)
              and open(dst + ".sha256").read().split()[0] == sha(so), r.stdout[-300:] + r.stderr[-200:])
        r = kinst(k, root, "uninstall")
        check("uninstall removes binary + sidecar and restores bases",
              r.returncode == 0 and not os.path.exists(dst) and {key: sha(p) for key, p in t.items()} == before)

    # 8 vet-pdl
    ctl = [f"IMAD R{i % 30}, R{(i + 1) % 30}, 0x{i:x}, RZ;" for i in range(300)]
    ctl[5] = "LDG.E.128.CONSTANT R4, desc[UR4][R2.64];"
    ctl[6] = "STS.128 [R3], R4;"
    ctl[7] = "@!P0 BRA 0x1480;"
    ctl[8] = "RED.E.OR.STRONG.GPU desc[UR6][R8.64], R9;"
    ctl[200] = "STG.E.U16 desc[UR4][R10.64], R12;"
    ctl[250] = "BRA 0x2f00;"
    strict = ctl[:20] + ["ACQBULK;", "NOP;"] + ctl[20:]
    strict = [l.replace("0x1480", "0x14a0").replace("0x2f00", "0x2f20") for l in strict]
    sched = ctl[:20] + ["ACQBULK;"] + ctl[21:40] + [ctl[20]] + ctl[40:]
    sched = [l.replace("R1,", "R31,") if i == 30 else l for i, l in enumerate(sched)]
    variants = {
        "STRICT accepted": (ctl, ctl, strict, True, "STRICT"),
        "SCHEDULE-ONLY accepted": (ctl, ctl, sched, True, "SCHEDULE-ONLY"),
        "live != control refused": (ctl[:-1] + ["EXIT;"], ctl, strict, False, "NOT IDENTICAL"),
        "extra FFMA refused": (ctl, ctl, strict + ["FFMA R1, R2, R3, R4;"], False, "opcode multiset differs"),
        "changed opcode refused": (ctl, ctl, [l.replace("IMAD R7,", "IADD3 R7,") for l in strict], False, "opcode multiset differs"),
        "no ACQBULK refused": (ctl, ctl, ctl, False, "no griddepcontrol.wait"),
        "ACQBULK in control refused": (["ACQBULK;"] + ctl[1:], ["ACQBULK;"] + ctl[1:], ["ACQBULK;"] + ctl[1:], False, "control already has"),
        "STG before wait refused": (ctl, ctl, ctl[:3] + ["STG.E desc[UR4][R2.64], R5;"] + ctl[3:20] + ["ACQBULK;"] + ctl[20:], False, "global write before"),
    }
    live_so_line = f"{I.PINNED_LIVE_SO}  /home/jeweled/dsv41-fc2/athena_tp8_fc2_packed_graph_sm121.so"
    cases = [(n, v, None) for n, v in variants.items()] + [
        ("live .so not pinned refused", variants["STRICT accepted"], "badlive"),
        (".so not the one hashed refused", variants["STRICT accepted"], "badso")]
    for name, (live, control, pdl, ok, needle), mod in cases:
        ok = ok and mod is None
        with tempfile.TemporaryDirectory() as root:
            k = temp_kit(root)
            b = os.path.join(root, "out")
            os.makedirs(b)
            for fn, lines in (("ins_athena_tp8_fc2_packed_graph_sm121.txt", live),
                              ("ins_control_athena_tp8_fc2_packed_graph_sm121.txt", control),
                              ("ins_athena_tp8_fc2_packed_graph_pdl_sm121.txt", pdl)):
                open(os.path.join(b, fn), "w").write("\n".join(lines) + "\n")
            so = os.path.join(b, "athena_tp8_fc2_packed_graph_pdl_sm121.so")
            open(so, "wb").write(b"\x7fELF pdl build")
            so_sha = sha(so) if mod != "badso" else "2" * 64
            lv = live_so_line if mod != "badlive" else live_so_line.replace(I.PINNED_LIVE_SO, "3" * 64)
            open(os.path.join(b, "COMPILE-LOG.txt"), "w").write(
                "== build_pdl.sh 2026-11-08T19:00:00Z on spark-eight\n== diff live vs control (expect empty)\n"
                "IDENTICAL\n== sha256\n" + lv + "\n" + "4" * 64 + "  control_athena_tp8_fc2_packed_graph_sm121.so\n"
                + so_sha + "  athena_tp8_fc2_packed_graph_pdl_sm121.so\n")
            kj0 = open(os.path.join(k, "KIT.json")).read()
            r = kinst(k, root, "vet-pdl", b)
            kj = json.load(open(os.path.join(k, "KIT.json")))
            if ok:
                good = (r.returncode == 0 and kj["pdl_binary_sha256"] == sha(so) and kj["pdl_vet"]["mode"] == needle
                        and kj["pdl_vet"]["host"] == "spark-eight"
                        and sha(os.path.join(k, "build/athena_tp8_fc2_packed_graph_pdl_sm121.so")) == sha(so))
            else:
                needle = {"badlive": "pinned production", "badso": "not the one COMPILE-LOG hashed"}.get(mod, needle)
                good = (r.returncode != 0 and needle in r.stderr and open(os.path.join(k, "KIT.json")).read() == kj0
                        and not os.path.exists(os.path.join(k, "build")))
            check(f"vet-pdl: {name}", good, (r.stdout + r.stderr)[-300:])


if __name__ == "__main__":
    main()
