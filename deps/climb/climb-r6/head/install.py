#!/usr/bin/env python3
"""r6-head (climb round 6: certified LM-head refine load balance + screen dequant): install / uninstall / status / dry.

    python3 install.py install | uninstall | status | dry

Edits ONE live file by pure tagged insertion (`# dsv41-r6-head` on every inserted line, exact anchor counts), backs it
up once to `<file>.bak-r6-head`, and copies one new module next to it:

  models/deepseek_v41/nvidia/dsv41_r4_small.py   (live 3364d8c8, r4-e1 + g1-stepovh + r1-drafter)  + 7 tagged lines
      module level: _DSV41_R6_HEAD = any of DSV41_R6_HEAD_REFINE / DSV41_R6_HEAD_SCREEN set; if so, import
                    dsv41_head_r6 (validates every r6 flag at import, ValueError on a bad value)
      certified_logits(): `if _DSV41_R6_HEAD and not full: return dsv41_head_r6.certified_logits(...)` as its first
                    statement (after the docstring). full=True (tests / check paths that ask for every tile) stays live.
  models/deepseek_v41/nvidia/dsv41_head_r6.py    NEW

With both flags unset the inserted lines evaluate two os.environ.get at import (module-level False) and one False test
per certified_logits call; no import, no kernel, launch, buffer or stream changes.

Interplay: dsv41_l2pf_r1._wrap_cert (DSV41_L2_PREFETCH_R1_DRAFT, off in production) and dsv41_drafter_sched
(DSV41_MAINPROJ_HOIST, off) wrap or call certified_logits by name: the dispatch sits inside the function body, so
both see the r6 path, and after_screen is passed through to the same point (between screen and refine). No r5 kit
touches dsv41_r4_small.py (r5-glue / r5-ffnfront / r5-prio / harness KIT.json), so the order against
climb-r5/install-all.sh does not matter; tests/test_install.py proves both orders on copies.

Accepted base: exactly the sha256 in KIT.json (the live file). Any other bytes -> refuse, nothing written. Idempotent.
Uninstall checks first that removing our tagged lines gives the accepted base, then writes, then deletes the module.

The new module: install copies it when absent, leaves it when it is the kit's bytes, replaces it ("upgraded") only when
it is a known earlier kit version (KIT.json `prior_sha256`), and refuses with nothing written when it is anything else.
Uninstall refuses with nothing written when the module present is neither the kit's bytes nor a known earlier version.
Per host only: there is no CLUSTER_LOCK check and no 8/8 wrapper here (r1/r2 pattern). Run it on each Spark inside a
window that holds the lock (climb-window.sh KIT_INSTALL or a device-window MAC_PLAN step), all 8 before a relaunch.

Env: R6H_VLLM (default ~/venvs/vllm032/lib/python3.12/site-packages/vllm) for tests on copies.
"""
import hashlib
import json
import os
import shutil
import sys

KIT = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("R6H_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
TAG = "# dsv41-r6-head"
BAK = ".bak-r6-head"
TMP = ".tmp-r6-head"

R4 = "models/deepseek_v41/nvidia/dsv41_r4_small.py"
NEW = {
    "models/deepseek_v41/nvidia/dsv41_head_r6.py":
        os.path.join(KIT, "files/vllm/models/deepseek_v41/nvidia/dsv41_head_r6.py"),
}

A1 = ('_DSV41_MAINPROJ_HOIST = os.environ.get("DSV41_MAINPROJ_HOIST", "0").strip().lower() not in '
      '("", "0", "off", "false", "no")  # dsv41-r1-drafter\n')
H1 = (
    f"# climb-r6 r6-head, default off: unset = dsv41_head_r6.py is never imported.  {TAG}\n"
    '_DSV41_R6_HEAD = any(os.environ.get(_n, "0").strip().lower() not in _OFF  '
    f"{TAG}\n"
    '                     for _n in ("DSV41_R6_HEAD_REFINE", "DSV41_R6_HEAD_SCREEN"))  '
    f"{TAG}\n"
    f"if _DSV41_R6_HEAD:  {TAG}\n"
    f"    from vllm.models.deepseek_v41.nvidia import dsv41_head_r6 as _r6_head  {TAG}\n"
)
A2 = ('    (full=True: exact kernel on every tile). No host sync; graph-capturable."""\n'
      '    k = _kernels()\n')
H2 = (
    '    (full=True: exact kernel on every tile). No host sync; graph-capturable."""\n'
    f"    if _DSV41_R6_HEAD and not full:  {TAG}\n"
    f"        return _r6_head.certified_logits(st, x, debug=debug, after_screen=after_screen)  {TAG}\n"
    '    k = _kernels()\n'
)
EDITS = {R4: [(A1, A1 + H1, 1), (A2, H2, 1)]}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    return sha_bytes(open(p, "rb").read()) if os.path.exists(p) else "-"


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def wr(p, text):
    tmp = p + TMP
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    shutil.copymode(p, tmp)
    os.replace(tmp, p)


def kit():
    return json.load(open(os.path.join(KIT, "KIT.json")))


def module_shas(rel):
    """(kit sha, set of known earlier kit shas) of a NEW module."""
    for f in kit()["files"]:
        if f["action"] == "new" and f["path"].split("/", 1)[1] == rel:
            return f["sha256"], set(f.get("prior_sha256") or [])
    raise SystemExit(f"{rel}: not in KIT.json")


def module_state(rel):
    p = os.path.join(V, rel)
    if not os.path.exists(p):
        return "absent"
    cur, prior = module_shas(rel)
    s = sha(p)
    return "present" if s == cur else "STALE-PRIOR" if s in prior else "FOREIGN"


def bases():
    out = {}
    for f in kit()["files"]:
        if f["action"] == "patch":
            out.setdefault(f["path"].split("/", 1)[1], set()).add(f["base_sha256"])
    return out


def strip_tag(src, tag=TAG):
    return "".join(ln for ln in src.splitlines(keepends=True) if not ln.rstrip("\r\n").endswith(tag))


def accepted(rel, src):
    return TAG not in src and sha_bytes(src.encode("utf-8")) in bases()[rel]


def patched(rel, src):
    for old, new, count in EDITS[rel]:
        n = src.count(old)
        if n != count:
            raise SystemExit(f"{rel}: anchor found {n}x, expected {count}: {old[:70]!r}")
        src = src.replace(old, new)
    return src


def ours_ok(rel, src):
    base = strip_tag(src)
    if not accepted(rel, base):
        return False
    try:
        return patched(rel, base) == src
    except SystemExit:
        return False


def state(rel):
    p = os.path.join(V, rel)
    if not os.path.exists(p):
        return "ABSENT"
    src = rd(p)
    if TAG in src:
        return "patched" if ours_ok(rel, src) else "FOREIGN-PATCHED"
    return "base" if accepted(rel, src) else "UNKNOWN-BASE"


def status():
    rows = []
    for rel in EDITS:
        p = os.path.join(V, rel)
        rows.append((rel, state(rel), sha(p)[:16], "bak" if os.path.exists(p + BAK) else "-"))
    mods = []
    for rel, srcp in NEW.items():
        p = os.path.join(V, rel)
        m = module_state(rel)
        mods.append(m)
        rows.append((rel, m, sha(p)[:16], "-"))
    st = [state(rel) for rel in EDITS]
    overall = ("INSTALLED" if all(s == "patched" for s in st) and all(m == "present" for m in mods)
               else "NOT-INSTALLED" if all(s == "base" for s in st) and all(m == "absent" for m in mods)
               else "PARTIAL")
    rows.append(("OVERALL", overall, "", ""))
    return rows


def plan_install():
    plan = []
    for rel in EDITS:
        p = os.path.join(V, rel)
        if not os.path.exists(p):
            raise SystemExit(f"{rel}: absent; refusing")
        src = rd(p)
        if TAG in src:
            if state(rel) != "patched":
                raise SystemExit(f"{rel}: carries {TAG} but is not exactly our patch of an accepted base; refusing")
            continue
        if not accepted(rel, src):
            raise SystemExit(f"{rel}: sha {sha(p)} is not an accepted base (KIT.json); refusing")
        out = patched(rel, src)
        if strip_tag(out) != src:
            raise SystemExit(f"{rel}: internal: edit is not a pure tagged insertion; refusing")
        plan.append((rel, p, out))
    for rel, srcp in NEW.items():
        if os.path.exists(os.path.join(V, rel) + TMP) or any(os.path.exists(os.path.join(V, r) + TMP) for r in EDITS):
            raise SystemExit(f"stale {TMP} file next to {rel}; refusing")
        if sha(srcp) != module_shas(rel)[0]:
            raise SystemExit(f"{rel}: kit file sha {sha(srcp)[:16]} != KIT.json sha256; refusing")
        if module_state(rel) == "FOREIGN":
            raise SystemExit(f"{rel}: present with sha {sha(os.path.join(V, rel))[:16]}, neither this kit's nor a known "
                             f"earlier kit version (KIT.json prior_sha256); refusing, nothing written. Inspect it.")
    return plan


def dry():
    plan = plan_install()
    for rel, p, out in plan:
        print(f"{rel}: OK, +{len(out.splitlines()) - len(rd(p).splitlines())} tagged lines")
    for rel in EDITS:
        if rel not in [r for r, _, _ in plan]:
            print(f"{rel}: already patched")
    for rel, srcp in NEW.items():
        print(f"{rel}: {module_state(rel)}; would copy {sha(srcp)[:16]}")


def install():
    plan = plan_install()
    for rel, srcp in NEW.items():  # the module first, then the hook that imports it
        dst = os.path.join(V, rel)
        ms = module_state(rel)  # absent / present / STALE-PRIOR (FOREIGN refused in plan_install)
        if ms != "present":
            shutil.copy2(srcp, dst + TMP)
            os.replace(dst + TMP, dst)
            print(f"{'upgraded (was a known earlier kit version)' if ms == 'STALE-PRIOR' else 'copied'} {rel}")
    for rel, p, out in plan:
        if not os.path.exists(p + BAK):
            shutil.copy2(p, p + BAK)
        wr(p, out)
        print(f"patched {rel}")
    for row in status():
        print(*row)


def uninstall():
    plan = []
    for rel in EDITS:
        p = os.path.join(V, rel)
        if not os.path.exists(p):
            continue
        src = rd(p)
        if TAG not in src:
            if not accepted(rel, src):
                raise SystemExit(f"{rel}: not ours and not an accepted base; nothing was changed")
            continue
        out = strip_tag(src)
        if not accepted(rel, out) or patched(rel, out) != src:
            raise SystemExit(f"{rel}: removing our {TAG} lines would not give an accepted base, or the file is not "
                             f"exactly our patch; nothing was changed. Inspect it (do NOT restore {p + BAK} blindly).")
        plan.append((rel, p, out))
    for rel in NEW:
        if module_state(rel) == "FOREIGN":
            raise SystemExit(f"{rel}: present with sha {sha(os.path.join(V, rel))[:16]}, neither this kit's nor a known "
                             f"earlier kit version; nothing was changed. Inspect it.")
    for rel, p, out in plan:
        wr(p, out)
        print(f"restored {rel} ({sha(p)[:16]})")
    for rel in EDITS:
        p = os.path.join(V, rel)
        if os.path.exists(p + BAK) and sha(p + BAK) == sha(p):
            os.remove(p + BAK)
    for rel in NEW:
        dst = os.path.join(V, rel)
        if os.path.exists(dst):
            os.remove(dst)
            print(f"removed {rel}")
        pyc = os.path.join(os.path.dirname(dst), "__pycache__")
        if os.path.isdir(pyc):
            for f in os.listdir(pyc):
                if f.startswith("dsv41_head_r6."):
                    os.remove(os.path.join(pyc, f))
    for row in status():
        print(*row)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "install":
        install()
    elif cmd == "uninstall":
        uninstall()
    elif cmd == "dry":
        dry()
    elif cmd == "status":
        rows = status()
        for row in rows:
            print(*row)
        sys.exit(0 if rows[-1][1] in ("INSTALLED", "NOT-INSTALLED") else 2)
    else:
        raise SystemExit(__doc__)
