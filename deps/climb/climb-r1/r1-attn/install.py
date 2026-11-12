#!/usr/bin/env python3
"""r1-attn kit: install / uninstall / check DSV41_IDXPAR (audit G20, attn:L-IP) in the vllm032 venv.

    python3 install.py install | uninstall | status | dry

Edits vllm/models/deepseek_v41/attention.py (backup attention.py.bak-r1attn; refuses unless the file's
sha256 is the base recorded in KIT.json or already this kit's patched result) and copies
nvidia/dsv41_idxpar.py next to dsv41_r3_dense.py. Every inserted line carries the TAG comment.
With DSV41_IDXPAR unset/0 the patch is one module-level bool test per attention-layer __init__ and per
_prepare_and_attn call: the helper module is never imported, nothing else changes.
Set R1ATTN_VLLM=<dir> to operate on another vllm tree (tests).
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("R1ATTN_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
ATTN_REL = "models/deepseek_v41/attention.py"
MOD_REL = "models/deepseek_v41/nvidia/dsv41_idxpar.py"
ATTN = f"{V}/{ATTN_REL}"
MOD_DST = f"{V}/{MOD_REL}"
MOD_SRC = os.path.join(HERE, "files", MOD_REL)
TAG = "# r1-attn"
BAK = ATTN + ".bak-r1attn"

E1_OLD = '_DSV41_IDX_WPROJ = bool(__import__("os").environ.get("DSV41_IDX_WPROJ", ""))\n'
E1_NEW = (E1_OLD +
          f"# r1-attn IDXPAR (audit G20; default off; see nvidia/dsv41_idxpar.py)  {TAG}\n"
          '_DSV41_IDXPAR = __import__("os").environ.get("DSV41_IDXPAR", "0").strip() not in ("", "0")'
          f"  {TAG}\n")

E2_OLD = "        if vllm_config.kernel_config.enable_jit_warmup:\n"
E2_NEW = (f"        if _DSV41_IDXPAR:  {TAG}\n"
          f"            from vllm.models.deepseek_v41.nvidia import dsv41_idxpar as _idxpar  {TAG}\n"
          f"            self._dsv41_idxpar = _idxpar.LayerPlan.create(self)  {TAG}\n"
          + E2_OLD)

E3_OLD = ('        both cache writes finish before sparse attention reads either cache.\n'
          '        """\n'
          '        attn_metadata = get_forward_context().attn_metadata\n'
          '        indexer = self.indexer\n')
E3_NEW = ('        both cache writes finish before sparse attention reads either cache.\n'
          '        """\n'
          f"        if _DSV41_IDXPAR and self._dsv41_idxpar is not None:  {TAG}\n"
          f"            if self._dsv41_idxpar.run(  {TAG}\n"
          f"                hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, attn_out  {TAG}\n"
          f"            ):  {TAG}\n"
          f"                return  {TAG}\n"
          '        attn_metadata = get_forward_context().attn_metadata\n'
          '        indexer = self.indexer\n')

# (old, new, expected count in the base file)
EDITS = [(E1_OLD, E1_NEW, 1), (E2_OLD, E2_NEW, 1), (E3_OLD, E3_NEW, 1)]


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "-"


def kit():
    return json.load(open(os.path.join(HERE, "KIT.json")))


def base_sha(rel):
    for f in kit()["files"]:
        if f["path"] == rel:
            return f["base_sha256"]
    raise KeyError(rel)


def patched(src):
    for old, new, count in EDITS:
        n = src.count(old)
        if n != count:
            sys.exit(f"{ATTN}: anchor found {n}x, expected {count}: {old[:70]!r}")
        src = src.replace(old, new)
    return src


def unpatched(src):
    """Exact inverse of patched(): drop every TAG line."""
    out = src
    for old, new, _ in EDITS:
        out = out.replace(new, old)
    return out


def install():
    src = open(ATTN).read()
    want = base_sha(ATTN_REL)
    if TAG in src:
        if hashlib.sha256(unpatched(src).encode()).hexdigest() != want:
            sys.exit(f"{ATTN}: patched, but its base is not {want[:8]}; refusing")
        print(f"{ATTN}: already patched ({sha256(ATTN)[:8]})")
    else:
        have = sha256(ATTN)
        if have != want:
            sys.exit(f"{ATTN}: sha256 {have[:8]} != KIT.json base {want[:8]}; refusing")
        new = patched(src)
        if not os.path.exists(BAK):
            shutil.copy2(ATTN, BAK)
        tmp = ATTN + ".tmp-r1attn"
        open(tmp, "w").write(new)
        shutil.copymode(BAK, tmp)
        os.replace(tmp, ATTN)
        print(f"{ATTN}: patched ({have[:8]} -> {sha256(ATTN)[:8]}), {new.count(TAG)} tagged lines")
    shutil.copy2(MOD_SRC, MOD_DST)
    print(f"{MOD_DST}: {sha256(MOD_DST)[:8]}")


def uninstall():
    if os.path.exists(BAK):
        os.replace(BAK, ATTN)
        print(f"{ATTN}: restored {sha256(ATTN)[:8]}")
    elif TAG in open(ATTN).read():
        new = unpatched(open(ATTN).read())
        open(ATTN, "w").write(new)
        print(f"{ATTN}: tagged lines removed (no backup) -> {sha256(ATTN)[:8]}")
    if os.path.exists(MOD_DST):
        os.remove(MOD_DST)
        print(f"{MOD_DST}: removed")


def status():
    a = open(ATTN).read()
    on = TAG in a
    base_ok = hashlib.sha256((unpatched(a) if on else a).encode()).hexdigest() == base_sha(ATTN_REL)
    print(f"{sha256(ATTN)[:8]} {'PATCHED' if on else 'clean  '} base={'ok' if base_ok else 'MISMATCH'} {ATTN}")
    m = os.path.exists(MOD_DST)
    same = m and sha256(MOD_DST) == sha256(MOD_SRC)
    print(f"{sha256(MOD_DST)[:8]} {'present' if m else 'absent '} {'(= kit)' if same else ''} {MOD_DST}")


def dry():
    src = open(ATTN).read()
    if TAG in src:
        src = unpatched(src)
    for i, line in enumerate(patched(src).splitlines(), 1):
        if TAG in line:
            print(f"{i:5d}: {line}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("install", "uninstall", "status", "dry"):
        sys.exit(__doc__)
    {"install": install, "uninstall": uninstall, "status": status, "dry": dry}[sys.argv[1]]()
