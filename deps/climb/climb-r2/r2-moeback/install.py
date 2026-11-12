#!/usr/bin/env python3
"""r2-moeback installer: G25 (DSV41_FC2_GATHER_ADD) and G18-PDL (DSV41_FC2_PDL), both default off.

    python3 install.py install | uninstall | status | dry | render <outdir> | vet-pdl <build_dir>

install    anchor-patch 3 live files (backup .bak-r2-moeback), copy the new module and the PDL FC2 binary.
uninstall  reverse the anchored edits (exact inverse replacement, so files that other kits patched after us
           keep their edits), remove the new module and the PDL binary.
status     sha256 + patched/clean per target (compare 8/8 in the window).
dry        print the tagged lines that install would write (no changes).
render     write the patched copies of the three targets into <outdir> (used by the CPU tests).

Targets (roots overridable for tests: R2MB_VLLM=<.../site-packages/vllm>, R2MB_FC2=<~/dsv41-fc2>):
  T1 vllm/model_executor/layers/fused_moe/runner/moe_runner.py        patch  G25
  T2 ~/dsv41-fc2/fc2_status_gated_gather_trial.py                      patch  G25
  T3 ~/dsv41-fc2/site/sitecustomize.py                                 patch  G18
  N1 vllm/model_executor/layers/fused_moe/runner/dsv41_moe_fold.py     new    G25
  N2 ~/dsv41-fc2/athena_tp8_fc2_packed_graph_pdl_sm121.so (+ .sha256)  new    G18 (binary). Installed ONLY when
     KIT.json pdl_binary_sha256 is set by `vet-pdl` and equals build/*.so, AND build/DEVICE-G18.json
     (tools/g18_gate.py) passed for that sha. A build/*.so with a null or different KIT.json sha is REFUSED
     (whole install, nothing written). Not vetted/gated yet: skipped with a warning and any stale installed copy
     removed, so DSV41_FC2_PDL=1 fails closed at boot (exit 97).
vet-pdl <dir>  check a build_pdl.sh output dir (live == control SASS, control vs PDL only the wait, pinned live
           .so) and record the PDL .so sha in KIT.json.
All bases are checked and all edits applied in memory before the first write (no partial install).
~/dsv41-fc2/athena_fc2_forty_layer_trial.py (sha-pinned by the launcher) and deep_gemm_moe.py (Athena's
transform_apply anchors) are NOT touched.
"""
import hashlib
import json
import os
import shutil
import sys

KIT = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("R2MB_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
FC2 = os.environ.get("R2MB_FC2") or os.path.expanduser("~/dsv41-fc2")
BAK = ".bak-r2-moeback"
T25 = "# r2-moeback-g25"
T18 = "# r2-moeback-g18"

# ---------------------------------------------------------------- T1 moe_runner.py (G25)
RUNNER_EDITS = [
    ("logger = init_logger(__name__)\n",
     "logger = init_logger(__name__)\n"
     f"# DSV41_FC2_GATHER_ADD=1: routed+shared add folded into Athena's FC2 gather (default off).  {T25}\n"
     f"_DSV41_FC2_GATHER_ADD = __import__(\"os\").environ.get(\"DSV41_FC2_GATHER_ADD\", \"0\").strip() == \"1\"  {T25}\n",
     1),
    ("        self._maybe_apply_shared_experts(\n"
     "            shared_experts_input, SharedExpertsOrder.NO_OVERLAP\n"
     "        )\n",
     "        self._maybe_apply_shared_experts(\n"
     "            shared_experts_input, SharedExpertsOrder.NO_OVERLAP\n"
     "        )\n"
     f"        if _DSV41_FC2_GATHER_ADD:  {T25}\n"
     f"            from vllm.model_executor.layers.fused_moe.runner import dsv41_moe_fold as _g25  {T25}\n"
     f"            _g25_tok = _g25.arm(self, shared_experts_overlapping)  {T25}\n",
     1),
    ("        if shared_experts_overlapping:\n"
     "            assert self._shared_experts is not None\n"
     "            self._shared_experts.wait()\n",
     f"        if _DSV41_FC2_GATHER_ADD:  {T25}\n"
     f"            self._dsv41_g25_folded = _g25.finish(_g25_tok, fused_out)  {T25}\n"
     "        if shared_experts_overlapping:\n"
     "            assert self._shared_experts is not None\n"
     "            self._shared_experts.wait()\n",
     1),
    ("        if shared_output is not None:\n"
     "            result = shared_output + fused_output\n"
     "        else:\n"
     "            result = fused_output\n",
     f"        if _DSV41_FC2_GATHER_ADD and shared_output is not None and self.__dict__.pop(\"_dsv41_g25_folded\", False):  {T25}\n"
     f"            result = fused_output  {T25}\n"
     f"        elif shared_output is not None:  {T25}\n"
     "            result = shared_output + fused_output\n"
     "        else:\n"
     "            result = fused_output\n",
     1),
]

# ---------------------------------------------------------------- T2 gather (G25)
# Inserted after the @triton.jit kernel so its source lines (and Triton's cache key) do not move.
GATHER_EDITS = [
    ("@torch.no_grad()\ndef gated_gather(",
     f"_DSV41_FC2_GATHER_ADD = __import__(\"os\").environ.get(\"DSV41_FC2_GATHER_ADD\", \"0\").strip() == \"1\"  {T25}\n"
     "\n\n@torch.no_grad()\ndef gated_gather(",
     1),
    # Inserted BEFORE the grid line (grid recomputed with the same expression), so round-1 r1-pdl's two-line
    # anchor "grid = ...\n    _gated_gather[grid](" stays intact whichever kit is installed first.
    ("    grid = (triton.cdiv(input_tensor.shape[1], block_d), min(n, 1024))\n",
     f"    if _DSV41_FC2_GATHER_ADD:  {T25}\n"
     f"        from vllm.model_executor.layers.fused_moe.runner import dsv41_moe_fold as _g25  {T25}\n"
     f"        _shared = _g25.take(output)  {T25}\n"
     f"        if _shared is not None:  {T25}\n"
     f"            return _g25.launch_gather_add((triton.cdiv(input_tensor.shape[1], block_d), min(n, 1024)), n, input_tensor, topk_ids, topk_weights, inv_perm, output, status, _shared, block_d)  {T25}\n"
     "    grid = (triton.cdiv(input_tensor.shape[1], block_d), min(n, 1024))\n",
     1),
]

# ---------------------------------------------------------------- T3 sitecustomize (G18)
SITE_EDITS = [
    ('        sys.path.insert(0, os.path.expanduser("~/dsv41-fc2"))\n',
     '        sys.path.insert(0, os.path.expanduser("~/dsv41-fc2"))\n'
     f'        if os.environ.get("DSV41_FC2_PDL", "0").strip() == "1":  {T18}\n'
     f'            import hashlib  {T18}\n'
     f'            _pdl_so = os.path.expanduser("~/dsv41-fc2/athena_tp8_fc2_packed_graph_pdl_sm121.so")  {T18}\n'
     f'            with open(_pdl_so, "rb") as _f: _pdl_sha = hashlib.sha256(_f.read()).hexdigest()  {T18}\n'
     f'            with open(_pdl_so + ".sha256") as _f: _pdl_want = _f.read().split()[0]  {T18}\n'
     f'            if _pdl_sha != _pdl_want: raise RuntimeError(f"DSV41_FC2_PDL: {{_pdl_so}} sha256 {{_pdl_sha}} != {{_pdl_want}}")  {T18}\n'
     f'            os.environ["ATHENA_FC2_GRAPH_SO"] = _pdl_so  {T18}\n'
     f'            os.environ.setdefault("DSV41_MOE_PDL_TRIGGER", "start")  {T18}\n'
     f'            print(f"DSV41_FC2_PDL on: {{_pdl_so}} {{_pdl_sha[:12]}} trigger={{os.environ[\'DSV41_MOE_PDL_TRIGGER\']}}", file=sys.stderr, flush=True)  {T18}\n',
     1),
]

TARGETS = {
    "moe_runner": (f"{V}/model_executor/layers/fused_moe/runner/moe_runner.py", RUNNER_EDITS, T25),
    "gather": (f"{FC2}/fc2_status_gated_gather_trial.py", GATHER_EDITS, T25),
    "site": (f"{FC2}/site/sitecustomize.py", SITE_EDITS, T18),
}
NEW_MODULE = (os.path.join(KIT, "files/vllm/model_executor/layers/fused_moe/runner/dsv41_moe_fold.py"),
              f"{V}/model_executor/layers/fused_moe/runner/dsv41_moe_fold.py")
PDL_SO_SRC = os.path.join(KIT, "build/athena_tp8_fc2_packed_graph_pdl_sm121.so")
PDL_SO_DST = f"{FC2}/athena_tp8_fc2_packed_graph_pdl_sm121.so"


def sha(path, n=64):
    if not os.path.exists(path):
        return "-"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:n]


def kit():
    with open(os.path.join(KIT, "KIT.json")) as f:
        return json.load(f)


def accepted_bases(key):
    """sha256 values this kit accepts as the base of a target: live production and live + round-1 kits."""
    for entry in kit()["files"]:
        if entry.get("key") == key:
            return set(entry["base_sha256"] if isinstance(entry["base_sha256"], list) else [entry["base_sha256"]])
    raise SystemExit(f"KIT.json has no entry for {key}")


def apply_edits(src, edits, label):
    for old, new, count in edits:
        n = src.count(old)
        if n != count:
            raise SystemExit(f"{label}: anchor found {n}x, expected {count}: {old[:80]!r}")
        src = src.replace(old, new)
    return src


def revert_edits(src, edits, label):
    for old, new, count in reversed(edits):
        n = src.count(new)
        if n != count:
            raise SystemExit(f"{label}: installed block found {n}x, expected {count}: {new[:80]!r}")
        src = src.replace(new, old)
    return src


def write_atomic(path, text, mode_from):
    tmp = path + ".tmp-r2-moeback"
    with open(tmp, "w") as f:
        f.write(text)
    shutil.copymode(mode_from, tmp)
    os.replace(tmp, path)


PDL_STAMP = os.path.join(KIT, "build/DEVICE-G18.json")
PINNED_LIVE_SO = "85d888ae36fdfbc1eb126e3c9048f6e2dc9b89a5189aae2488bc89d630b515e0"  # live FC2 .so (KIT.json)


def pdl_gate():
    """Decide what install does with the G18 binary. Refuses (SystemExit, before anything is written) when the
    kit holds a PDL .so that is not vetted: KIT.json pdl_binary_sha256 null, or a different sha. Returns
    ("install", sha) only when the binary is vetted (vet-pdl) AND device-tested (build/DEVICE-G18.json from
    tools/g18_gate.py: full device_test --part g18 with >= 200 replays, racecheck and synccheck all rc 0, for
    this exact sha). Otherwise ("skip", reason): the binary is not installed (and a stale installed copy is
    removed), so DSV41_FC2_PDL=1 fails closed at boot (exit 97); flag off is unaffected."""
    if not os.path.exists(PDL_SO_SRC):
        return "skip", f"{PDL_SO_SRC} missing (G18 binary not built)"
    want = kit().get("pdl_binary_sha256")
    got = sha(PDL_SO_SRC)
    if not want:
        raise SystemExit(f"{PDL_SO_SRC} is present but KIT.json pdl_binary_sha256 is null: unvetted binary; "
                         "run `python3 install.py vet-pdl <build_pdl.sh out dir>` or remove it; refusing")
    if got != want:
        raise SystemExit(f"{PDL_SO_SRC}: sha256 {got} != KIT.json pdl_binary_sha256 {want}; refusing")
    try:
        with open(PDL_STAMP) as f:
            st = json.load(f)
    except (OSError, ValueError) as exc:
        return "skip", f"no device gate stamp ({PDL_STAMP}: {exc.__class__.__name__}); run tools/g18_gate.py"
    problems = [k for k, ok in (
        ("pdl_so_sha256", st.get("pdl_so_sha256") == want),
        ("pass", st.get("pass") is True),
        ("device_test_rc", st.get("device_test_rc") == 0),
        ("racecheck_rc", st.get("racecheck_rc") == 0),
        ("synccheck_rc", st.get("synccheck_rc") == 0),
        ("reps>=200", isinstance(st.get("reps"), int) and st["reps"] >= 200),
        ("graph_cases>=8", isinstance(st.get("graph_cases_ok"), int) and st["graph_cases_ok"] >= 8),
    ) if not ok]
    if problems:
        return "skip", f"device gate stamp {PDL_STAMP} does not pass: {problems}"
    return "install", got


def preflight():
    """Check every target base, apply every edit in memory and run the PDL gate BEFORE writing anything, so a
    mismatch anywhere refuses the whole install (no partial install)."""
    plan, errors = [], []
    for key, (path, edits, tag) in TARGETS.items():
        try:
            src = open(path).read()
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if tag in src:
            plan.append((key, path, None, tag))
            continue
        base = sha(path)
        if base not in accepted_bases(key):
            errors.append(f"{path}: base sha256 {base} not in KIT.json accepted bases for {key}")
            continue
        try:
            new = apply_edits(src, edits, path)
            compile(new, path, "exec")
        except (SystemExit, SyntaxError) as exc:
            errors.append(str(exc))
            continue
        plan.append((key, path, new, tag))
    if errors:
        raise SystemExit("preflight failed, nothing written:\n  " + "\n  ".join(errors) + "\nrefusing")
    return plan, pdl_gate()


def install():
    plan, (pdl_action, pdl_info) = preflight()
    for key, path, new, tag in plan:
        if new is None:
            print(f"{path}: already patched ({sha(path, 8)})")
            continue
        base = sha(path)
        bak = path + BAK
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        write_atomic(path, new, bak)
        print(f"{path}: patched ({base[:8]} -> {sha(path, 8)}), {new.count(tag)} tagged lines")
    src, dst = NEW_MODULE
    shutil.copy2(src, dst)
    print(f"{dst}: {sha(dst, 8)}")
    if pdl_action == "install":
        shutil.copy2(PDL_SO_SRC, PDL_SO_DST)
        with open(PDL_SO_DST + ".sha256", "w") as f:
            f.write(f"{pdl_info}  {os.path.basename(PDL_SO_DST)}\n")
        print(f"{PDL_SO_DST}: {pdl_info[:8]} (+ .sha256; vetted + device-gated)")
    else:
        for p in (PDL_SO_DST, PDL_SO_DST + ".sha256"):
            if os.path.exists(p):
                os.remove(p)
                print(f"removed stale {p}")
        print(f"WARNING: G18 binary not installed: {pdl_info}; DSV41_FC2_PDL=1 would exit 97 at boot")


# ---------------------------------------------------------------- vet-pdl (build_pdl.sh output -> KIT.json)
WAIT_OPS = {"ACQBULK"}                  # griddepcontrol.wait on sm_12x (seen in production sm_120/121 SASS)
PLUMBING_OPS = {"NOP"}                  # scheduling padding
BRANCH_OPS = {"BRA", "BRX", "JMP", "JMX", "CALL", "BSSY", "BREAK", "BSYNC", "SSY", "PBK", "WARPSYNC", "RET"}


def _opcode(line):
    for tok in line.replace(";", " ").split():
        if not tok.startswith("@"):
            return tok
    return ""


def _is_global_write(op):
    base = op.split(".")[0]
    if base in ("STG", "ST", "STAS"):
        return True
    if base in ("RED", "ATOMG", "ATOM", "REDG"):
        return ".OR" not in op           # the structural-error status atomicOr is the only allowed pre-wait write
    return False


def vet_compare(live, control, pdl):
    """Pure check of build_pdl.sh's instruction lists (one SASS instruction per line, addresses stripped).
    Returns (mode, failures). mode STRICT: PDL == control with only wait/NOP lines inserted; SCHEDULE-ONLY:
    same opcode multiset once wait/NOP are removed (register allocation / order may differ)."""
    from collections import Counter
    fails = []
    if min(len(live), len(control), len(pdl)) < 100:
        fails.append(f"instruction lists too short (live {len(live)}, control {len(control)}, pdl {len(pdl)})")
    if live != control:
        fails.append("live vs control SASS NOT IDENTICAL (toolchain does not reproduce production)")
    ops_c = [_opcode(l) for l in control]
    ops_p = [_opcode(l) for l in pdl]
    n_wait_c = sum(o in WAIT_OPS for o in ops_c)
    n_wait_p = sum(o in WAIT_OPS for o in ops_p)
    if n_wait_c:
        fails.append(f"control already has {n_wait_c} wait instructions")
    if n_wait_p < 1:
        fails.append("PDL SASS has no griddepcontrol.wait (ACQBULK)")
    else:
        first = ops_p.index(next(o for o in ops_p if o in WAIT_OPS))
        early = [o for o in ops_p[:first] if _is_global_write(o)]
        if early:
            fails.append(f"global write before the first wait in listing order: {early}")
    import re

    def norm(line, op):  # branch/reconvergence targets shift by the inserted instructions
        return re.sub(r"0x[0-9a-f]+", "0xADDR", line) if op.split(".")[0] in BRANCH_OPS else line

    def keep(lines, ops):
        return [norm(l, o) for l, o in zip(lines, ops) if o not in WAIT_OPS | PLUMBING_OPS]

    if keep(pdl, ops_p) == keep(control, ops_c):
        mode = "STRICT"
    else:
        cc = Counter(o for o in ops_c if o not in WAIT_OPS | PLUMBING_OPS)
        cp = Counter(o for o in ops_p if o not in WAIT_OPS | PLUMBING_OPS)
        if cc == cp:
            mode = "SCHEDULE-ONLY"
        else:
            mode = "DIFFERENT"
            fails.append(f"opcode multiset differs beyond the wait: control-only {dict(cc - cp)}, "
                         f"pdl-only {dict(cp - cc)}")
    return mode, fails


def vet_pdl(build_dir):
    """Vet a build_pdl.sh output dir: SASS gates (vet_compare), the live .so in the log is production's pinned
    binary, the PDL .so on disk is the one the log hashed. On success copy the .so + evidence into the kit's
    build/ and record pdl_binary_sha256 + pdl_vet in KIT.json. Refuses (nothing written) otherwise."""
    import re
    b = os.path.abspath(os.path.expanduser(build_dir))
    names = {"live": "ins_athena_tp8_fc2_packed_graph_sm121.txt",
             "control": "ins_control_athena_tp8_fc2_packed_graph_sm121.txt",
             "pdl": "ins_athena_tp8_fc2_packed_graph_pdl_sm121.txt"}
    so = os.path.join(b, "athena_tp8_fc2_packed_graph_pdl_sm121.so")
    logp = os.path.join(b, "COMPILE-LOG.txt")
    for p in list(os.path.join(b, n) for n in names.values()) + [so, logp]:
        if not os.path.exists(p):
            raise SystemExit(f"vet-pdl: {p} missing; refusing")
    lists = {k: [l.rstrip("\n") for l in open(os.path.join(b, n)) if l.strip()] for k, n in names.items()}
    mode, fails = vet_compare(lists["live"], lists["control"], lists["pdl"])
    log = open(logp).read()
    run = log[log.rfind("== build_pdl.sh"):] if "== build_pdl.sh" in log else log
    shas = {os.path.basename(p): h for h, p in re.findall(r"^([0-9a-f]{64})\s+(\S+)$", run, re.M)}
    if shas.get("athena_tp8_fc2_packed_graph_sm121.so") != PINNED_LIVE_SO:
        fails.append(f"live .so in COMPILE-LOG is {shas.get('athena_tp8_fc2_packed_graph_sm121.so')}, "
                     f"not the pinned production binary {PINNED_LIVE_SO[:12]}")
    got = sha(so)
    if shas.get("athena_tp8_fc2_packed_graph_pdl_sm121.so") != got:
        fails.append("PDL .so on disk is not the one COMPILE-LOG hashed")
    if "IDENTICAL" not in run:
        fails.append("COMPILE-LOG has no IDENTICAL line for live vs control")
    host = re.search(r"== build_pdl.sh (\S+) on (\S+)", run)
    print(f"vet-pdl {b}: mode {mode}; live {len(lists['live'])} control {len(lists['control'])} pdl "
          f"{len(lists['pdl'])} instructions; pdl .so {got[:16]}")
    if fails:
        raise SystemExit("vet-pdl FAILED, KIT.json unchanged:\n  " + "\n  ".join(fails))
    os.makedirs(os.path.join(KIT, "build"), exist_ok=True)
    for n in list(names.values()) + ["COMPILE-LOG.txt"]:
        shutil.copy2(os.path.join(b, n), os.path.join(KIT, "build", n))
    shutil.copy2(so, PDL_SO_SRC)
    kp = os.path.join(KIT, "KIT.json")
    k = kit()
    k["pdl_binary_sha256"] = got
    k["pdl_vet"] = {"mode": mode, "built_at": host.group(1) if host else None, "host": host.group(2) if host else None,
                    "live_so_sha256": PINNED_LIVE_SO, "compile_log_sha256": sha(logp),
                    "instructions": {n: len(v) for n, v in lists.items()}}
    tmp = kp + ".tmp-r2-moeback"
    with open(tmp, "w") as f:
        json.dump(k, f, indent=1)
        f.write("\n")
    os.replace(tmp, kp)
    print(f"vet-pdl OK ({mode}): KIT.json pdl_binary_sha256 = {got}; next: tools/g18_gate.py on a stopped node")


def uninstall():
    for key, (path, edits, tag) in TARGETS.items():
        src = open(path).read()
        if tag not in src:
            print(f"{path}: clean ({sha(path, 8)})")
            continue
        old = revert_edits(src, edits, path)
        if tag in old:
            raise SystemExit(f"{path}: tagged lines remain after revert; refusing")
        write_atomic(path, old, path)
        print(f"{path}: reverted -> {sha(path, 8)}")
        if os.path.exists(path + BAK) and sha(path + BAK) == sha(path):  # climb-r2 integrate: drop the redundant backup
            os.remove(path + BAK)
    for p in (NEW_MODULE[1], PDL_SO_DST, PDL_SO_DST + ".sha256"):
        if os.path.exists(p):
            os.remove(p)
            print(f"removed {p}")


def status():
    for key, (path, edits, tag) in TARGETS.items():
        on = os.path.exists(path) and tag in open(path).read()
        print(f"{sha(path, 16)} {'PATCHED' if on else 'clean  '} {path}")
    for p in (NEW_MODULE[1], PDL_SO_DST, PDL_SO_DST + ".sha256"):
        print(f"{sha(p, 16)} {'present' if os.path.exists(p) else 'absent '} {p}")


def dry():
    for key, (path, edits, tag) in TARGETS.items():
        src = open(path).read()
        print(f"== {path} base {sha(path, 16)} accepted={sha(path) in accepted_bases(key)}")
        new = src if tag in src else apply_edits(src, edits, path)
        for i, line in enumerate(new.splitlines(), 1):
            if tag in line:
                print(f"{i:5d}: {line}")


def render(outdir):
    os.makedirs(outdir, exist_ok=True)
    for key, (path, edits, tag) in TARGETS.items():
        new = apply_edits(open(path).read(), edits, path)
        with open(os.path.join(outdir, os.path.basename(path)), "w") as f:
            f.write(new)
        print(os.path.join(outdir, os.path.basename(path)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "render":
        render(sys.argv[2])
    elif cmd == "vet-pdl":
        vet_pdl(sys.argv[2])
    elif cmd in ("install", "uninstall", "status", "dry"):
        {"install": install, "uninstall": uninstall, "status": status, "dry": dry}[cmd]()
    else:
        raise SystemExit(__doc__)
