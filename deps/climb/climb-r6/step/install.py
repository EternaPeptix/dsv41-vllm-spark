#!/usr/bin/env python3
"""r6-step (climb round 6): install / uninstall / check the step-start collective levers (C3) and the E1 main_proj
side stream (C4).

    python3 install.py install | uninstall | status | dry | compose-check

Patches two live files under site-packages/vllm (every inserted line tagged '# r6-step', one backup per file
'<file>.bak-r6-step') and copies models/deepseek_v41/nvidia/dsv41_step.py next to them:

    models/deepseek_v41/common/engram.py           C3: module-level flag + install_c3(Engram) at the end of the file
    models/deepseek_v41/nvidia/dsv41_r4_small.py   C4: flag, fork after the screen, take in mainproj_split, install

With DSV41_STEP_COLEMB, DSV41_STEP_ENGRAM_PACK and DSV41_STEP_C4 unset, dsv41_step is never imported and every
patched function is AST-identical to the live one after folding the two module flags to False
(tests/test_step.py checks this).

Base check: each file's sha256 must be its live 166d7aac sha (KIT.json base_sha256) or a compose sha (the same file
with the r5 kits installed first: engram.py c8170bd9 = r5-prio). Install is all-or-nothing: every anchor of every
file is checked before the first byte is written. Uninstall restores a backup only when the live file is exactly
patch(backup). Env: STEP_VLLM overrides the vllm directory.
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("STEP_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
TAG = "# r6-step"
BAKSUF = ".bak-r6-step"
REL_MOD = "models/deepseek_v41/nvidia/dsv41_step.py"
MOD_SRC = os.path.join(HERE, "files", "vllm", REL_MOD)
_FLAG = 'not in ("", "0", "off", "false", "no")'
IMP = "from vllm.models.deepseek_v41.nvidia import dsv41_step as _dsv41_step"

EDITS = {
    "models/deepseek_v41/common/engram.py": [
        ("            num_warps=num_warps,\n"
         "        )\n"
         "        return output\n",
         "            num_warps=num_warps,\n"
         "        )\n"
         "        return output\n"
         "\n"
         "\n"
         f"# climb r6 (r6-step) C3, default off: unset = dsv41_step is never imported.  {TAG}\n"
         f'_DSV41_STEP_C3 = any(__import__("os").environ.get(_k, "").strip().lower() {_FLAG}  {TAG}\n'
         f'                     for _k in ("DSV41_STEP_COLEMB", "DSV41_STEP_ENGRAM_PACK"))  {TAG}\n'
         f"if _DSV41_STEP_C3:  {TAG}\n"
         f"    {IMP}  {TAG}\n"
         f"    _dsv41_step.install_c3(Engram)  {TAG}\n", 1),
    ],
    "models/deepseek_v41/nvidia/dsv41_r4_small.py": [
        ('_DSV41_MAINPROJ_HOIST = os.environ.get("DSV41_MAINPROJ_HOIST", "0").strip().lower() not in '
         '("", "0", "off", "false", "no")  # dsv41-r1-drafter\n',
         '_DSV41_MAINPROJ_HOIST = os.environ.get("DSV41_MAINPROJ_HOIST", "0").strip().lower() not in '
         '("", "0", "off", "false", "no")  # dsv41-r1-drafter\n'
         f"# climb r6 (r6-step) C4, default off: unset = dsv41_step is never imported.  {TAG}\n"
         f'_DSV41_STEP_C4 = os.environ.get("DSV41_STEP_C4", "").strip().lower() {_FLAG}  {TAG}\n', 1),
        ("    if MAINPROJ_SPLIT:\n"
         "        _install_dspark_hook()\n",
         "    if MAINPROJ_SPLIT:\n"
         "        _install_dspark_hook()\n"
         f"    if _DSV41_STEP_C4:  {TAG}\n"
         f"        {IMP}  {TAG}\n"
         f"        _dsv41_step.install_c4()  {TAG}\n", 1),
        ("            _dsv41_hk = mainproj_target_hook()  # dsv41-r1-drafter\n"
         "        local = certified_logits(st, hidden_states,  # dsv41-r1-drafter\n",
         "            _dsv41_hk = mainproj_target_hook()  # dsv41-r1-drafter\n"
         f"        if _DSV41_STEP_C4 and _dsv41_hk is None and armed and not skip_gather and head.tp_size > 1:  {TAG}\n"
         f"            {IMP}  {TAG}\n"
         f"            _dsv41_hk = _dsv41_step.c4_target_hook()  {TAG}\n"
         "        local = certified_logits(st, hidden_states,  # dsv41-r1-drafter\n", 1),
        ("    if not st:\n"
         "        return None\n"
         "    verdict = st.checked.get(M)\n",
         "    if not st:\n"
         "        return None\n"
         f"    if _DSV41_STEP_C4:  {TAG}\n"
         f"        {IMP}  {TAG}\n"
         f"        _dsv41_y = _dsv41_step.c4_take(st, x, M)  {TAG}\n"
         f"        if _dsv41_y is not None:  {TAG}\n"
         f"            return _dsv41_y  {TAG}\n"
         "    verdict = st.checked.get(M)\n", 1),
    ],
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def kit() -> dict:
    with open(os.path.join(HERE, "KIT.json")) as f:
        return json.load(f)


def allowed_bases(rel: str) -> set:
    for f in kit()["files"]:
        if f["path"] == rel:
            return {f["base_sha256"], *f.get("compose_sha256", [])}
    raise KeyError(rel)


def patch_text(rel: str, text: str) -> str:
    if TAG in text:
        raise ValueError(f"{rel}: already carries {TAG} lines")
    for old, new, count in EDITS[rel]:
        n = text.count(old)
        if n != count:
            raise ValueError(f"{rel}: anchor found {n}x, expected {count}x: {old.splitlines()[0][:80]!r}")
        text = text.replace(old, new)
    return text


def unpatch_text(rel: str, text: str) -> str:
    for old, new, count in reversed(EDITS[rel]):
        n = text.count(new)
        if n != count:
            raise ValueError(f"{rel}: patched block found {n}x, expected {count}x")
        text = text.replace(new, old)
    return text


def read(p: str) -> str:
    with open(p, encoding="utf-8") as f:
        return f.read()


def write_atomic(p: str, text: str, mode_from: str) -> None:
    tmp = p + ".tmp-r6-step"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    shutil.copymode(mode_from, tmp)
    os.replace(tmp, p)


def state(rel: str) -> str:
    p = os.path.join(V, rel)
    cur = read(p)
    bak = p + BAKSUF
    if os.path.exists(bak):
        b = read(bak)
        try:
            ok = patch_text(rel, b) == cur
        except ValueError:
            ok = False
        if ok and sha(b.encode()) in allowed_bases(rel):
            return "INSTALLED"
        return "INCONSISTENT"
    if TAG in cur:
        return "INCONSISTENT"
    return "BASE" if sha(cur.encode()) in allowed_bases(rel) else "FOREIGN"


def mod_state() -> str:
    dst = os.path.join(V, REL_MOD)
    if not os.path.exists(dst):
        return "ABSENT"
    with open(dst, "rb") as f, open(MOD_SRC, "rb") as g:
        return "INSTALLED" if f.read() == g.read() else "FOREIGN"


def cmd_status() -> int:
    bad = 0
    for rel in EDITS:
        s = state(rel)
        print(f"{s:12s} {rel}")
        bad |= s in ("INCONSISTENT", "FOREIGN")
    m = mod_state()
    print(f"{m:12s} {REL_MOD}")
    bad |= m == "FOREIGN"
    return 1 if bad else 0


def cmd_install(dry: bool = False) -> int:
    plan = []
    for rel in EDITS:
        s = state(rel)
        if s == "INSTALLED":
            continue
        if s != "BASE":
            print(f"REFUSE {rel}: state {s} (base sha not in KIT.json base/compose list)")
            return 2
        p = os.path.join(V, rel)
        cur = read(p)
        plan.append((rel, p, cur, patch_text(rel, cur)))  # raises before any write on a missing anchor
    if mod_state() == "FOREIGN":
        print(f"REFUSE {REL_MOD}: a different file is already there")
        return 2
    for rel, p, cur, new in plan:
        print(f"{'would patch' if dry else 'patch'} {rel}: {sha(cur.encode())[:8]} -> {sha(new.encode())[:8]}")
    if dry:
        return 0
    for rel, p, cur, new in plan:
        shutil.copy2(p, p + BAKSUF)
        write_atomic(p, new, p)
    dst = os.path.join(V, REL_MOD)
    if mod_state() == "ABSENT":
        shutil.copy2(MOD_SRC, dst)
        print(f"copy {REL_MOD}")
    return cmd_status()


def cmd_uninstall() -> int:
    for rel in EDITS:
        s = state(rel)
        if s == "INCONSISTENT" or s == "FOREIGN":
            print(f"REFUSE {rel}: state {s} (another kit patched it after r6-step? uninstall that kit first)")
            return 2
    for rel in EDITS:
        p = os.path.join(V, rel)
        if state(rel) == "INSTALLED":
            write_atomic(p, read(p + BAKSUF), p)
            os.remove(p + BAKSUF)
            print(f"restore {rel}")
    dst = os.path.join(V, REL_MOD)
    if mod_state() == "INSTALLED":
        os.remove(dst)
        print(f"remove {REL_MOD}")
    return cmd_status()


def cmd_compose_check() -> int:
    """Print the sha each target would get from the current file (to review before adding a compose sha)."""
    for rel in EDITS:
        p = os.path.join(V, rel)
        cur = read(p)
        try:
            new = patch_text(rel, cur)
            back = unpatch_text(rel, new)
            print(f"{rel}: base {sha(cur.encode())[:16]} -> patched {sha(new.encode())[:16]}; "
                  f"unpatch round trip {'OK' if back == cur else 'FAIL'}")
        except ValueError as e:
            print(f"{rel}: {e}")
    return 0


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "install":
        return cmd_install()
    if cmd == "dry":
        return cmd_install(dry=True)
    if cmd == "uninstall":
        return cmd_uninstall()
    if cmd == "status":
        return cmd_status()
    if cmd == "compose-check":
        return cmd_compose_check()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
