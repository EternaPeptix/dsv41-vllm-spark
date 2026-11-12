#!/usr/bin/env python3
"""r8-fixb installer (one node): swaps the INSTALLED r6-step module dsv41_step.py (ebe2f7bb) for the r8 module (425c6515,
PACK operand-identity pairing) and back. Nothing else is installed: the head and skew repairs are device-test-only
(staged by ops/stage.sh into ~/climb-scratch/r8-fixb/, never into the venv).

    python3 install.py status | dry | install | uninstall | pins
    env: FIXB_VLLM = the vllm package dir (default ~/venvs/vllm032/lib/python3.12/site-packages/vllm)

States of models/deepseek_v41/nvidia/dsv41_step.py (relative to FIXB_VLLM):
    CLEAN       the r6 module ebe2f7bb, no backup                      -> install allowed
    INSTALLED   the r8 module 425c6515 + backup .bak-r8-fixb == ebe2f7bb  -> uninstall allowed; install = no-op
    NO-R6STEP   r6-step is not installed (module absent)               -> refuse (install climb-r6 step first)
    FOREIGN     anything else (another module, a stray backup, an unpatched engram.py / dsv41_r4_small.py)
Preconditions for install (all checked before the first byte is written): r6-step's patch is on engram.py (34cf684e on
live, e99f623d on r5-prio) and dsv41_r4_small.py (d9bf5d59 on live, ebe73cdb = r6-head -> r6-step composed); the local
r8 module bytes are the KIT.json pin. Install = backup copy + atomic replace (same mode) + read-back; uninstall restores
the backup atomically only when the live file is exactly the r8 pin and the backup exactly the r6 pin.
LIFO: climb-r6/install-all.sh and climb-r7/install-all.sh pin dsv41_step.py at ebe2f7bb, so while r8-fixb is installed
they report it FOREIGN and refuse: run `install.py uninstall` (8/8, ops/fleet.sh uninstall) before any of their actions.
No CLUSTER_LOCK check here (one node); ops/fleet.sh checks it before touching any node.
"""
import hashlib
import json
import os
import shutil
import stat
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("FIXB_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
REL = "models/deepseek_v41/nvidia/dsv41_step.py"
SRC = os.path.join(HERE, "step", "files", "vllm", REL)
BAK = ".bak-r8-fixb"
KIT = json.load(open(os.path.join(HERE, "KIT.json")))
F = KIT["install_files"][0]
R6_SHA, R8_SHA = F["base_sha256"], F["sha256"]
ENGRAM = ("models/deepseek_v41/common/engram.py", set(F["requires"]["models/deepseek_v41/common/engram.py"]))
R4 = ("models/deepseek_v41/nvidia/dsv41_r4_small.py", set(F["requires"]["models/deepseek_v41/nvidia/dsv41_r4_small.py"]))


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def state():
    p = os.path.join(V, REL)
    b = p + BAK
    if not os.path.exists(p):
        return "NO-R6STEP" if not os.path.exists(b) else "FOREIGN"
    s = sha(p)
    if s == R6_SHA and not os.path.exists(b):
        return "CLEAN"
    if s == R8_SHA and os.path.exists(b) and sha(b) == R6_SHA:
        return "INSTALLED"
    return "FOREIGN"


def preconds():
    errs = []
    if sha(SRC) != R8_SHA:
        errs.append(f"local r8 module {sha(SRC)[:16]} != KIT.json pin {R8_SHA[:16]}")
    for rel, ok in (ENGRAM, R4):
        p = os.path.join(V, rel)
        if not os.path.exists(p):
            errs.append(f"{rel} missing")
        elif sha(p) not in ok:
            errs.append(f"{rel} {sha(p)[:16]} is not an r6-step patched state {sorted(x[:8] for x in ok)}")
    return errs


def write_atomic(dst, data, mode_from):
    tmp = dst + ".tmp-r8-fixb"
    with open(tmp, "wb") as f:
        f.write(data)
    os.chmod(tmp, stat.S_IMODE(os.stat(mode_from).st_mode))
    os.replace(tmp, dst)


def cmd_status():
    s = state()
    p = os.path.join(V, REL)
    print(f"{s:10s} {REL} ({sha(p)[:16] if os.path.exists(p) else 'absent'})")
    for e in preconds():
        print(f"  note: {e}")
    return 0 if s in ("CLEAN", "INSTALLED") else 1


def cmd_install(dry=False):
    s = state()
    if s == "INSTALLED":
        print(f"INSTALLED {REL}: nothing to do")
        return 0
    if s != "CLEAN":
        print(f"REFUSE {REL}: state {s} (need the r6-step module {R6_SHA[:8]} and no {BAK})")
        return 2
    errs = preconds()
    if errs:
        print("REFUSE: " + "; ".join(errs))
        return 2
    p = os.path.join(V, REL)
    print(f"{'would replace' if dry else 'replace'} {REL}: {R6_SHA[:8]} -> {R8_SHA[:8]} (backup {BAK})")
    if dry:
        return 0
    shutil.copy2(p, p + BAK)
    if sha(p + BAK) != R6_SHA:
        os.remove(p + BAK)
        print("REFUSE: backup read-back differs")
        return 2
    write_atomic(p, open(SRC, "rb").read(), p)
    ok = state() == "INSTALLED"
    print("INSTALLED" if ok else "FAILED read-back")
    return 0 if ok else 3


def cmd_uninstall():
    s = state()
    if s == "CLEAN":
        print(f"CLEAN {REL}: nothing to do")
        return 0
    if s != "INSTALLED":
        print(f"REFUSE {REL}: state {s}")
        return 2
    p = os.path.join(V, REL)
    write_atomic(p, open(p + BAK, "rb").read(), p)
    if sha(p) != R6_SHA:
        print("FAILED: restored bytes differ")
        return 3
    os.remove(p + BAK)
    print(f"restored {REL} {R6_SHA[:8]}; CLEAN")
    return 0


def cmd_pins():
    """Every file the kit ships or a device test runs, with its KIT.json pin (exit 1 on any drift)."""
    bad = 0
    for rel, want in sorted(KIT["pins"].items()):
        got = sha(os.path.join(HERE, rel)) if os.path.exists(os.path.join(HERE, rel)) else "absent"
        ok = got == want
        bad |= not ok
        print(f"{'OK  ' if ok else 'DIFF'} {want[:16]} {rel}" + ("" if ok else f" (local {got[:16]})"))
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    fn = {"status": cmd_status, "dry": lambda: cmd_install(True), "install": cmd_install, "uninstall": cmd_uninstall,
          "pins": cmd_pins}.get(cmd)
    if fn is None:
        print(__doc__)
        sys.exit(2)
    sys.exit(fn())
