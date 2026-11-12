#!/usr/bin/env python3
"""r5-glue (climb round 5: RQPAD + RPSPLIT): install / uninstall / status / dry.

    python3 install.py install | uninstall | status | dry

Appends tagged lines (`# r5-glue`) after the LAST lines of two round-2 kernel modules (pure insertion, exact anchor
counts), backs each file up once to `<file>.bak-r5-glue`, and copies two new modules next to them:

  models/deepseek_v4/nvidia/ops/dsv41_attnout.py          (r2-attnout, live 308942eb)  + 4 tagged lines
      if DSV41_GLUE_RQPAD: import dsv41_r5g_rq; _launch_rq = dsv41_r5g_rq.wrap(_launch_rq)
  model_executor/layers/fused_moe/dsv41_routeprep.py      (r2-moefront, live 842f49fb) + 4 tagged lines
      if DSV41_GLUE_RPSPLIT: import dsv41_r5g_rp; _launch = dsv41_r5g_rp.wrap(_launch)
  models/deepseek_v4/nvidia/ops/dsv41_r5g_rq.py           NEW
  model_executor/layers/fused_moe/dsv41_r5g_rp.py         NEW

With both flags unset the inserted lines only evaluate one os.environ.get each at import (module-level False) and
nothing else changes: no import, no rebinding, same kernels, same launches.

Accepted base for each patched file: exactly the live sha256 in KIT.json. Lines tagged by another kit are REFUSED
(FOREIGN_TAGS is empty, review N4): the kit re-implements _r2_reduce_quant and _dsv41_rp_kernel, so a foreign edit of
these modules (e.g. a future r1-pdl rebase onto the r2 kernels, r4 REPORT #13) would silently not reach the kit kernels;
such a rebase needs a new r5-glue base instead. Idempotent. Uninstall checks EVERY file first (stripping our tagged lines must give an
accepted base) and only then writes; it removes exactly our tagged lines and deletes the two new modules.

Interplay with the parent kits (they own the two patched modules):
  * r2-attnout `install.py install` and r2-moefront `install.py install` re-copy their module unconditionally, which
    silently drops our hooks (the flags then do nothing; `status` here prints HOOKS-LOST). Do not re-run them while
    r5-glue is installed; if you must, run `python3 install.py install` here again afterwards.
  * Their `uninstall` deletes their module; uninstall r5-glue FIRST.
  * climb-r2/install-all.sh `status` hashes these two modules against the r2 kits' shas: with r5-glue installed it
    reports FOREIGN for dsv41_attnout.py and dsv41_routeprep.py on every node and exits 2. That is expected (review
    N3); use `install-all.sh status` here (it checks 8/8 agreement of the r5-glue state and of the stripped bases).

Env: R5G_VLLM (default ~/venvs/vllm032/lib/python3.12/site-packages/vllm) for tests on copies.
"""
import hashlib
import json
import os
import shutil
import sys

KIT = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("R5G_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
TAG = "# r5-glue"
BAK = ".bak-r5-glue"
FOREIGN_TAGS = ()  # review N4: no foreign-tagged lines accepted in the two modules whose kernels the kit re-implements

AO = "models/deepseek_v4/nvidia/ops/dsv41_attnout.py"
RP = "model_executor/layers/fused_moe/dsv41_routeprep.py"
NEW = {
    "models/deepseek_v4/nvidia/ops/dsv41_r5g_rq.py": os.path.join(KIT, "files/vllm/models/deepseek_v4/nvidia/ops",
                                                                  "dsv41_r5g_rq.py"),
    "model_executor/layers/fused_moe/dsv41_r5g_rp.py": os.path.join(KIT, "files/vllm/model_executor/layers/fused_moe",
                                                                    "dsv41_r5g_rp.py"),
}

AO_ANCHOR = "    x[0, ..., 64:96] = 1.0 + 2.0 ** -8\n    return x.float()\n"
AO_HOOK = (
    '_R5G_RQPAD = os.environ.get("DSV41_GLUE_RQPAD", "0").strip().lower() not in ("", "0", "off", "false")'
    f"  {TAG}\n"
    f"if _R5G_RQPAD:  {TAG}\n"
    f"    from vllm.models.deepseek_v4.nvidia.ops import dsv41_r5g_rq as _r5g_rq  {TAG}\n"
    f"    _launch_rq = _r5g_rq.wrap(_launch_rq)  {TAG}\n"
)
RP_ANCHOR = ("        return aq_out, sfp, layout, inv_perm, align_used, ends\n"
             "    return aq_out, sfp, layout, inv_perm, align_used\n")
RP_HOOK = (
    '_R5G_RPSPLIT = os.environ.get("DSV41_GLUE_RPSPLIT", "0").strip().lower() not in ("", "0", "off", "false")'
    f"  {TAG}\n"
    f"if _R5G_RPSPLIT:  {TAG}\n"
    f"    from vllm.model_executor.layers.fused_moe import dsv41_r5g_rp as _r5g_rp  {TAG}\n"
    f"    _launch = _r5g_rp.wrap(_launch)  {TAG}\n"
)
# relpath -> [(anchor, replacement, expected count)]; every replacement = anchor + tagged lines
EDITS = {
    AO: [(AO_ANCHOR, AO_ANCHOR + AO_HOOK, 1)],
    RP: [(RP_ANCHOR, RP_ANCHOR + RP_HOOK, 1)],
}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    return sha_bytes(open(p, "rb").read()) if os.path.exists(p) else "-"


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def wr(p, text):
    tmp = p + ".tmp-r5-glue"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    shutil.copymode(p, tmp)
    os.replace(tmp, p)


def kit():
    return json.load(open(os.path.join(KIT, "KIT.json")))


def bases():
    out = {}
    for f in kit()["files"]:
        if f["action"] == "patch":
            out[f["path"].split("/", 1)[1]] = f["base_sha256"]
    return out


def strip_tag(src, tag):
    return "".join(ln for ln in src.splitlines(keepends=True) if not ln.rstrip("\r\n").endswith(tag))


def strip_foreign(src):
    for t in FOREIGN_TAGS:
        src = strip_tag(src, t)
    return src


def accepted(rel, src):
    return TAG not in src and sha_bytes(strip_foreign(src).encode("utf-8")) == bases()[rel]


def patched(rel, src):
    for old, new, count in EDITS[rel]:
        n = src.count(old)
        if n != count:
            raise SystemExit(f"{rel}: anchor found {n}x, expected {count}: {old[:70]!r}")
        src = src.replace(old, new)
    return src


def ours_ok(rel, src):
    if not accepted(rel, strip_tag(src, TAG)):
        return False
    try:
        mine = patched(rel, strip_foreign(strip_tag(src, TAG)))
    except SystemExit:
        return False
    return strip_foreign(src) == mine


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
        rows.append((rel, state(rel), sha(p)[:16], os.path.exists(p + BAK)))
    any_new = False
    for rel, srcp in NEW.items():
        p = os.path.join(V, rel)
        any_new |= os.path.exists(p)
        rows.append((rel, "present" if os.path.exists(p) and sha(p) == sha(srcp) else
                     ("STALE" if os.path.exists(p) else "absent"), sha(p)[:16], False))
    st = [state(rel) for rel in EDITS]
    if any_new and any(s == "base" for s in st):
        rows.append(("NOTE", "HOOKS-LOST: a new module is present but a parent module is unpatched (r2 kit "
                     "re-installed?): run `install.py install` again", "", False))
    overall = ("PARENT-ABSENT (r2-attnout / r2-moefront module missing)" if "ABSENT" in st
               else "INSTALLED" if all(s == "patched" for s in st) and all(r[1] == "present" for r in rows[2:4])
               else "NOT-INSTALLED" if all(s == "base" for s in st) and not any_new else "PARTIAL")
    rows.append(("OVERALL", overall, "", False))
    return rows


def dry():
    for rel in EDITS:
        p = os.path.join(V, rel)
        src = rd(p)
        if TAG in src:
            print(f"{rel}: already patched ({state(rel)})")
            continue
        if not accepted(rel, src):
            raise SystemExit(f"{rel}: sha {sha(p)} is not an accepted base (KIT.json); refusing")
        out = patched(rel, src)
        assert strip_tag(out, TAG) == src
        print(f"{rel}: OK, +{len(out.splitlines()) - len(src.splitlines())} tagged lines")
    for rel, srcp in NEW.items():
        print(f"{rel}: would copy {sha(srcp)[:16]}")


def install():
    plan = []
    for rel in EDITS:
        p = os.path.join(V, rel)
        if not os.path.exists(p):
            raise SystemExit(f"{rel}: absent (the parent r2 kit is not installed); refusing")
        src = rd(p)
        if TAG in src:
            if state(rel) != "patched":
                raise SystemExit(f"{rel}: carries {TAG} but does not strip to an accepted base; refusing")
            continue
        if not accepted(rel, src):
            raise SystemExit(f"{rel}: sha {sha(p)} is not an accepted base (KIT.json); refusing")
        out = patched(rel, src)
        if strip_tag(out, TAG) != src:
            raise SystemExit(f"{rel}: internal: edit is not a pure tagged insertion; refusing")
        plan.append((rel, p, out))
    for rel, srcp in NEW.items():  # copy the modules before the hooks that import them
        dst = os.path.join(V, rel)
        if sha(dst) != sha(srcp):
            shutil.copy2(srcp, dst + ".tmp-r5-glue")
            os.replace(dst + ".tmp-r5-glue", dst)
            print(f"copied {rel}")
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
            continue
        out = strip_tag(src, TAG)
        if not accepted(rel, out):
            raise SystemExit(f"{rel}: removing our {TAG} lines would not give an accepted base; nothing was changed. "
                             f"Inspect the file (do NOT restore {p + BAK} blindly).")
        plan.append((rel, p, out))
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
        for row in status():
            print(*row)
    else:
        raise SystemExit(__doc__)
