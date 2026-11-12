#!/usr/bin/env python3
"""r2-moefront (climb round 2: G04 Stage A route-prep fusion + G18 FC1 front): install / uninstall / status / dry.

    python3 install.py install | uninstall | status | dry

Patches two vLLM site files by INSERTING tagged lines (`# r2-moefront`) around unchanged anchors with exact expected
counts, backs each file up once to `<file>.bak-r2-moefront`, and copies one new module next to deep_gemm_utils.py.

Accepted base for each patched file (refuses anything else):
  * the live sha256 recorded in KIT.json (`base_sha256`), or
  * a file whose lines tagged by another known kit (`# r1-pdl`, `# r2-draftmoe`: pure tagged insertions) strip back to
    that live sha (other kits installed first; rule 5). KIT.json also records the composed shas seen at build time.
Idempotent: a file that already carries our tag, and is exactly our patch of an accepted base (other kits' tagged lines
ignored), is left alone.
Uninstall checks EVERY file first (stripping our tagged lines must give an accepted base) and only then writes, so a
refusal changes nothing; it removes exactly our tagged lines, which leaves other kits' lines in place in either install
order; it never restores a .bak. The new module is removed. Recommended order (both orders are tested): install r1-pdl
-> r2-moefront -> r2-draftmoe, uninstall in reverse. With DSV41_MOE_ROUTEPREP unset every inserted statement sits under
`if _DSV41_RP:` (module-level False) and nothing else changes. Files are read and written as UTF-8 with newline=''.

Env: R2MF_VLLM (default ~/venvs/vllm032/lib/python3.12/site-packages/vllm) for tests on copies.
"""
import hashlib
import json
import os
import shutil
import sys

KIT = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("R2MF_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
TAG = "# r2-moefront"
BAK = ".bak-r2-moefront"
# other kits that may have inserted tagged lines (pure insertions) into our two files, before or after us
FOREIGN_TAGS = ("# r1-pdl", "# r2-draftmoe")

FLAG = ('_DSV41_RP = os.environ.get("DSV41_MOE_ROUTEPREP", "0").strip().lower() not in ("", "0", "off", "false")'
        f"  {TAG}\n"
        f"if _DSV41_RP:  {TAG}\n"
        f"    from vllm.model_executor.layers.fused_moe import dsv41_routeprep as _dsv41_rp  {TAG}\n")

UTILS = "model_executor/layers/fused_moe/deep_gemm_utils.py"
MOE = "model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
NEW_REL = "model_executor/layers/fused_moe/dsv41_routeprep.py"
NEW_SRC = os.path.join(KIT, "files/vllm", NEW_REL)

MFRONT = "    # g3 mfront (DSV41_MOE_FUSED_PERMUTE): fused plan + scatter on the qualified M32 geometry.\n"
FP4_ASSERTS = ("        assert a1q_scale is not None\n"
               "        assert a2_scale is None\n"
               "        assert self.w1_scale is not None\n")
EOF_ANCHOR = "            self._dsv41_draft_verify.shadow(self, output, dsv41_call)\n"

# relpath -> [(anchor, replacement, expected count)]; every replacement = anchor + tagged lines or tagged lines + anchor
EDITS = {
    UTILS: [
        ("_DSV41_FP_FILL_ROWS = 1024\n", "_DSV41_FP_FILL_ROWS = 1024\n" + FLAG, 1),
        (MFRONT,
         f"    if _DSV41_RP and isinstance(aq_scale, _dsv41_rp.Deferred):  {TAG}\n"
         "        return _dsv41_rp.fused_permute(aq, aq_scale, topk_ids, local_num_experts, expert_map, "
         "expert_tokens_meta, aq_out, M_sum, align_used, block_k, use_psum_layout, return_expert_ends)"
         f"  {TAG}\n" + MFRONT, 1),
    ],
    MOE: [
        ("logger = init_logger(__name__)\n", "logger = init_logger(__name__)\n" + FLAG, 1),
        (FP4_ASSERTS,
         f"        if _DSV41_RP and a1q_scale is None and self.expects_unquantized_inputs:  {TAG}\n"
         "            hidden_states, a1q_scale = _dsv41_rp.front(self, hidden_states, w1, topk_ids, expert_map, "
         f"expert_tokens_meta)  {TAG}\n" + FP4_ASSERTS, 1),
        (EOF_ANCHOR,
         EOF_ANCHOR
         + f"if _DSV41_RP:  {TAG}\n"
         + "    DeepGemmFP4Experts.expects_unquantized_inputs = property(_dsv41_rp.expects_unquantized_inputs)"
         + f"  {TAG}\n", 1),
    ],
}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    return sha_bytes(open(p, "rb").read()) if os.path.exists(p) else "-"


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def wr(p, text):
    tmp = p + ".tmp-r2-moefront"
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
    """True when src (without our tag) is the live base, or the live base plus other kits' tagged insertions only."""
    return TAG not in src and sha_bytes(strip_foreign(src).encode("utf-8")) == bases()[rel]


def patched(rel, src):
    for old, new, count in EDITS[rel]:
        n = src.count(old)
        if n != count:
            raise SystemExit(f"{rel}: anchor found {n}x, expected {count}: {old[:70]!r}")
        src = src.replace(old, new)
    return src


def ours_ok(rel, src):
    """src carries our tag: True when, ignoring other kits' tagged lines, it is exactly our patch of the live base."""
    if not accepted(rel, strip_tag(src, TAG)):
        return False
    try:
        mine = patched(rel, strip_foreign(strip_tag(src, TAG)))
    except SystemExit:
        return False
    return strip_foreign(src) == mine


def state(rel):
    src = rd(os.path.join(V, rel))
    if TAG in src:
        return "patched" if ours_ok(rel, src) else "FOREIGN-PATCHED"
    return "base" if accepted(rel, src) else "UNKNOWN-BASE"


def status():
    rows = []
    for rel in EDITS:
        p = os.path.join(V, rel)
        rows.append((rel, state(rel), sha(p)[:16], os.path.exists(p + BAK)))
    new = os.path.join(V, NEW_REL)
    rows.append((NEW_REL, "present" if os.path.exists(new) and sha(new) == sha(NEW_SRC) else
                 ("STALE" if os.path.exists(new) else "absent"), sha(new)[:16], False))
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
        added = len(out.splitlines()) - len(src.splitlines())
        assert strip_tag(out, TAG) == src
        print(f"{rel}: OK, +{added} tagged lines")
    print(f"{NEW_REL}: would copy {sha(NEW_SRC)[:16]}")


def install():
    plan = []
    for rel in EDITS:
        p = os.path.join(V, rel)
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
    for rel, p, out in plan:  # every anchor checked before any write
        if not os.path.exists(p + BAK):
            shutil.copy2(p, p + BAK)
        wr(p, out)
        print(f"patched {rel}")
    new = os.path.join(V, NEW_REL)
    if sha(new) != sha(NEW_SRC):
        shutil.copy2(NEW_SRC, new + ".tmp-r2-moefront")
        os.replace(new + ".tmp-r2-moefront", new)
        print(f"copied {NEW_REL}")
    for row in status():
        print(*row)


def uninstall():
    plan = []
    for rel in EDITS:  # check every file before any write: a refusal changes nothing
        p = os.path.join(V, rel)
        src = rd(p)
        if TAG not in src:
            continue
        out = strip_tag(src, TAG)
        if not accepted(rel, out):
            raise SystemExit(f"{rel}: removing our {TAG} lines would not give the live base (plus only {FOREIGN_TAGS} "
                             f"lines); nothing was changed. Do NOT restore {p + BAK} blindly: it predates any kit "
                             f"installed after this one. Inspect the file.")
        plan.append((rel, p, out))
    for rel, p, out in plan:
        wr(p, out)
        print(f"restored {rel} ({sha(p)[:16]})")
        if os.path.exists(p + BAK) and sha(p + BAK) == sha(p):  # climb-r2 integrate: drop the backup once it is redundant
            os.remove(p + BAK)
    new = os.path.join(V, NEW_REL)
    if os.path.exists(new):
        os.remove(new)
        print(f"removed {NEW_REL}")
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
