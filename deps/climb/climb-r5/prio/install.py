#!/usr/bin/env python3
"""r5-prio (climb round 5): install / uninstall / check the default-off stream-priority and ordering levers.

    python3 install.py install | uninstall [--force] | status | dry | compose-check

Patches five live files under site-packages/vllm (every inserted line tagged '# r5-prio', one backup per file
'<file>.bak-r5-prio') and copies models/deepseek_v41/nvidia/dsv41_prio.py next to them:

    models/deepseek_v41/attention.py                       DSV41_PRIO_CMP  (KV-source qkv_a vs compressor)
    models/deepseek_v41/nvidia/dsv41_r3_dense.py           DSV41_PRIO_CMP  (shard-GEMM enter/leave hooks)
    model_executor/layers/fused_moe/runner/shared_experts.py  DSV41_PRIO_SHARED
    models/deepseek_v41/common/engram.py                   DSV41_PRIO_ENGRAM_HOIST (take the hoisted wkv)
    models/deepseek_v41/nvidia/model.py                    DSV41_PRIO_ENGRAM_HOIST (install the hoist hooks)

Every inserted statement sits under a module-level flag read once at import; with DSV41_PRIO_CMP,
DSV41_PRIO_SHARED and DSV41_PRIO_ENGRAM_HOIST unset the new module is never imported and every patched function
is AST-identical to the live one after constant-folding the flags to False (tests/test_prio.py checks this).

Base check: each file's sha256 must be in KIT.json files[].base_sha256 (live 166d7aac) or in
files[].compose_sha256 (the same file with another kit installed first; add the sha printed by
'compose-check' after reviewing it). Install is all-or-nothing: every anchor of every file is checked before
the first byte is written. Uninstall restores a backup only when the live file is exactly patch(backup).
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("PRIO_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
TAG = "# r5-prio"
BAKSUF = ".bak-r5-prio"
REL_MOD = "models/deepseek_v41/nvidia/dsv41_prio.py"
MOD_SRC = os.path.join(HERE, "files", "vllm", REL_MOD)

_FLAG = 'not in ("", "0", "off", "false", "no")'
IMP = "from vllm.models.deepseek_v41.nvidia import dsv41_prio as _prio"

EDITS = {
    # ------------------------------------------------------------------------------------------------ CMPPRIO
    "models/deepseek_v41/attention.py": [
        ('_DSV41_IDXPAR = __import__("os").environ.get("DSV41_IDXPAR", "0").strip() not in ("", "0")  # r1-attn\n',
         '_DSV41_IDXPAR = __import__("os").environ.get("DSV41_IDXPAR", "0").strip() not in ("", "0")  # r1-attn\n'
         f'_DSV41_PRIO_CMP = __import__("os").environ.get("DSV41_PRIO_CMP", "").strip().lower() {_FLAG}  {TAG}\n', 1),
        ("        qr_kv, (kv_score, indexer_weights) = execute_in_parallel(\n"
         "            lambda: self._fused_wqa_wkv_gemm(hidden_states),\n",
         f"        if _DSV41_PRIO_CMP and self.compressor is not None:  {TAG}\n"
         f"            {IMP}  {TAG}\n"
         f"            return _prio.cmp_parallel_input_projections(  {TAG}\n"
         f"                self, hidden_states, aux_fns, aux_streams,  {TAG}\n"
         f"                hidden_states.shape[0] <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD)  {TAG}\n"
         "        qr_kv, (kv_score, indexer_weights) = execute_in_parallel(\n"
         "            lambda: self._fused_wqa_wkv_gemm(hidden_states),\n", 1),
    ],
    "models/deepseek_v41/nvidia/dsv41_r3_dense.py": [
        ('R2_QFUSE = R2_V2 and os.environ.get("DSV41_SKINNY_QFUSE", "0").strip() == "1"  # r2-skinny\n',
         'R2_QFUSE = R2_V2 and os.environ.get("DSV41_SKINNY_QFUSE", "0").strip() == "1"  # r2-skinny\n'
         f'_DSV41_PRIO_CMP = os.environ.get("DSV41_PRIO_CMP", "").strip().lower() {_FLAG}  {TAG}\n'
         f"\n\ndef _dsv41_prio_enter(st, x):  {TAG}\n"
         f"    {IMP}  {TAG}\n"
         f"    return _prio.cmp_enter(st, x)  {TAG}\n"
         f"\n\ndef _dsv41_prio_leave(tok, y):  {TAG}\n"
         f"    {IMP}  {TAG}\n"
         f"    return _prio.cmp_leave(tok, y)  {TAG}\n", 1),
        ("        if R2_QFUSE and self._r2 is not None:  # r2-skinny\n"
         "            _y = self._r2.fused(x)  # r2-skinny\n"
         "            if _y is not None:  # r2-skinny\n"
         "                return tensor_model_parallel_all_gather(_y, dim=-1).view(*x.shape[:-1], self.N)  # r2-skinny\n"
         "        xq, xs, shape = _quantize_act(x, self.K)\n"
         "        y = self.gemm(xq, xs)  # [M, n] bf16: this rank's contiguous output rows\n",
         f"        _dsv41_pt = _dsv41_prio_enter(self, x) if _DSV41_PRIO_CMP else None  {TAG}\n"
         "        if R2_QFUSE and self._r2 is not None:  # r2-skinny\n"
         "            _y = self._r2.fused(x)  # r2-skinny\n"
         "            if _y is not None:  # r2-skinny\n"
         f"                if _DSV41_PRIO_CMP and _dsv41_pt is not None:  {TAG}\n"
         f"                    _y = _dsv41_prio_leave(_dsv41_pt, _y)  {TAG}\n"
         "                return tensor_model_parallel_all_gather(_y, dim=-1).view(*x.shape[:-1], self.N)  # r2-skinny\n"
         "        xq, xs, shape = _quantize_act(x, self.K)\n"
         "        y = self.gemm(xq, xs)  # [M, n] bf16: this rank's contiguous output rows\n"
         f"        if _DSV41_PRIO_CMP and _dsv41_pt is not None:  {TAG}\n"
         f"            y = _dsv41_prio_leave(_dsv41_pt, y)  {TAG}\n", 1),
    ],
    # ------------------------------------------------------------------------------------------------- SHARED
    "model_executor/layers/fused_moe/runner/shared_experts.py": [
        ("logger = init_logger(__name__)\n",
         "logger = init_logger(__name__)\n"
         f'_DSV41_PRIO_SHARED = __import__("os").environ.get("DSV41_PRIO_SHARED", "").strip().lower() {_FLAG}  {TAG}\n',
         1),
        ("            self._stream = aux_stream()\n",
         "            self._stream = aux_stream()\n"
         f"            if _DSV41_PRIO_SHARED:  {TAG}\n"
         f"                {IMP}  {TAG}\n"
         f"                self._stream = _prio.shared_stream(self._stream, layer)  {TAG}\n", 1),
    ],
    # ------------------------------------------------------------------------------------------- ENGRAM HOIST
    "models/deepseek_v41/common/engram.py": [
        ("logger = init_logger(__name__)\n",
         "logger = init_logger(__name__)\n"
         f'_DSV41_PRIO_EH = __import__("os").environ.get("DSV41_PRIO_ENGRAM_HOIST", "").strip().lower() {_FLAG}  {TAG}\n',
         1),
        ("            kv = self.wkv(self.embed(hash_ids).flatten(-2))\n",
         f"            kv = (self.wkv(self.embed(hash_ids).flatten(-2))  {TAG}\n"
         f'                  if not _DSV41_PRIO_EH or self.__dict__.get("_dsv41_prio_take") is None  {TAG}\n'
         f"                  else self._dsv41_prio_take(hash_ids))  {TAG}\n", 1),
    ],
    "models/deepseek_v41/nvidia/model.py": [
        ("logger = init_logger(__name__)\n",
         "logger = init_logger(__name__)\n"
         f'_DSV41_PRIO_EH = __import__("os").environ.get("DSV41_PRIO_ENGRAM_HOIST", "").strip().lower() {_FLAG}  {TAG}\n',
         1),
        ("        if self.fuse_mhc_all_reduce:\n"
         "            # A MoE's top-k finalize folds into the next layer's first mHC\n",
         f"        if _DSV41_PRIO_EH:  {TAG}\n"
         f"            {IMP}  {TAG}\n"
         f"            _prio.engram_install(self)  {TAG}\n"
         "        if self.fuse_mhc_all_reduce:\n"
         "            # A MoE's top-k finalize folds into the next layer's first mHC\n", 1),
    ],
}
FILES = list(EDITS)


def kit():
    return json.load(open(os.path.join(HERE, "KIT.json")))


def kit_file(rel):
    for f in kit()["files"]:
        if f["path"] == rel:
            return f
    raise SystemExit(f"KIT.json has no entry for {rel}")


def sha(p, n=64):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:n] if os.path.exists(p) else "-"


def sha_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def patched(src, rel, where=None):
    where = where or rel
    for old, new, count in EDITS[rel]:
        n = src.count(old)
        if n != count:
            raise SystemExit(f"{where}: anchor found {n}x, expected {count}: {old[:90]!r}")
        src = src.replace(old, new)
    return src


def accepted_bases(rel):
    f = kit_file(rel)
    return {f["base_sha256"], *f.get("compose_sha256", [])}


def paths(rel):
    p = f"{V}/{rel}"
    return p, p + BAKSUF


def plan_install():
    """Return [(rel, path, new_text or None (already patched))]; raises before anything is written."""
    want = kit_file(REL_MOD)
    if sha(MOD_SRC) != want.get("kit_sha256"):
        raise SystemExit(f"{MOD_SRC}: sha {sha(MOD_SRC, 16)} != KIT.json kit_sha256; refusing")
    out = []
    for rel in FILES:
        p, bak = paths(rel)
        if not os.path.exists(p):
            raise SystemExit(f"{p}: missing")
        src = open(p).read()
        if TAG in src:
            if not os.path.exists(bak):
                raise SystemExit(f"{p}: tagged but no {bak}; refusing")
            if sha(bak) not in accepted_bases(rel):
                raise SystemExit(f"{bak}: sha {sha(bak, 16)} not an accepted base; refusing")
            if src != patched(open(bak).read(), rel, bak):
                raise SystemExit(f"{p}: tagged but differs from patch(backup); refusing")
            out.append((rel, p, None))
        else:
            if sha(p) not in accepted_bases(rel):
                raise SystemExit(f"{p}: sha256 {sha(p, 16)} is not KIT.json base_sha256/compose_sha256; refusing "
                                 f"(run 'install.py compose-check' and review)")
            out.append((rel, p, patched(src, rel, p)))
    return out


def install():
    plan = plan_install()
    for rel, p, new in plan:
        bak = p + BAKSUF
        if new is None:
            print(f"{p}: already patched ({sha(p, 8)})")
            continue
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
        tmp = p + ".tmp-r5-prio"
        open(tmp, "w").write(new)
        shutil.copymode(bak, tmp)
        os.replace(tmp, p)
        print(f"{p}: patched ({sha(bak, 8)} -> {sha(p, 8)}), {new.count(TAG)} tagged lines")
        exp = kit_file(rel).get("patched_sha256")
        if sha(bak) == kit_file(rel)["base_sha256"] and exp and sha(p) != exp:
            raise SystemExit(f"{p}: patched sha {sha(p, 16)} != KIT.json patched_sha256 {exp[:16]}")
    dst = f"{V}/{REL_MOD}"
    shutil.copy2(MOD_SRC, dst)
    print(f"{dst}: {sha(dst, 8)}")


def _force():
    return "--force" in sys.argv[2:]


def uninstall():
    """Restore each backup only when the live file is exactly patch(backup) (or KIT patched_sha256); remove the
    module only when it is exactly the kit's. Everything is checked before anything is touched."""
    errs, acts = [], []
    for rel in FILES:
        p, bak = paths(rel)
        if os.path.exists(bak):
            cur = open(p).read() if os.path.exists(p) else None
            try:
                exp = patched(open(bak).read(), rel, bak)
            except SystemExit:
                exp = None
            if cur is not None and sha(p) == sha(bak):
                acts.append(("drop", p, bak))
            elif cur is not None and (cur == exp or sha(p) == kit_file(rel).get("patched_sha256")):
                acts.append(("restore", p, bak))
            else:
                acts.append(("restore", p, bak))
                errs.append(f"{p}: sha {sha(p, 16)} is not patch(backup); edited after install")
        elif os.path.exists(p) and TAG in open(p).read():
            raise SystemExit(f"{p}: tagged but no backup; refusing")
    dst = f"{V}/{REL_MOD}"
    if os.path.exists(dst) and sha(dst) != kit_file(REL_MOD).get("kit_sha256"):
        errs.append(f"{dst}: sha {sha(dst, 16)} != KIT.json kit_sha256")
    if errs and not _force():
        raise SystemExit("refusing to uninstall (nothing changed; --force overrides):\n  " + "\n  ".join(errs))
    for what, p, bak in acts:
        if what == "restore":
            os.replace(bak, p)
            print(f"{p}: restored {sha(p, 8)}")
        else:
            os.remove(bak)
            print(f"{p}: already {sha(p, 8)}; removed {bak}")
    if os.path.exists(dst):
        os.remove(dst)
        print(f"{dst}: removed")


def status():
    """Exit 0 when the node is consistently CLEAN or consistently INSTALLED, 2 otherwise."""
    inst, clean = True, True
    for rel in FILES:
        p, bak = paths(rel)
        f = kit_file(rel)
        sp, sb = sha(p), sha(bak)
        tagged = os.path.exists(p) and TAG in open(p).read()
        ok_patch = tagged and sb in accepted_bases(rel) and open(p).read() == patched(open(bak).read(), rel, bak)
        tag = ("= patched(backup)" if ok_patch else "= base_sha256" if sp == f["base_sha256"]
               else "= compose base" if sp in f.get("compose_sha256", []) else "UNKNOWN sha")
        print(f"{sp[:8]} {'PATCHED' if tagged else 'clean  '} {p}  [{tag}]")
        inst &= ok_patch
        clean &= (not tagged) and sp in accepted_bases(rel) and sb == "-"
    dst = f"{V}/{REL_MOD}"
    sm = sha(dst)
    print(f"{sm[:8]} {'present' if sm != '-' else 'absent '} {dst}"
          f"  [{'= kit_sha256' if sm == kit_file(REL_MOD).get('kit_sha256') else '-' if sm == '-' else 'UNKNOWN sha'}]")
    inst &= sm == kit_file(REL_MOD).get("kit_sha256")
    clean &= sm == "-"
    verdict = "INSTALLED" if inst else "CLEAN" if clean else "INCONSISTENT"
    print(f"r5-prio status: {verdict}")
    if verdict == "INCONSISTENT":
        sys.exit(2)


def dry():
    for rel in FILES:
        p, bak = paths(rel)
        src_path = bak if os.path.exists(p) and TAG in open(p).read() else p
        for i, line in enumerate(patched(open(src_path).read(), rel, src_path).splitlines(), 1):
            if TAG in line:
                print(f"{rel}:{i}: {line}")


def compose_check():
    """For a node where another kit was installed first: verify every anchor applies to the live file and
    print its sha256 (to be reviewed and added to KIT.json files[].compose_sha256). Writes nothing."""
    for rel in FILES:
        p, bak = paths(rel)
        src_path = bak if os.path.exists(p) and TAG in open(p).read() else p
        src = open(src_path).read()
        patched(src, rel, src_path)
        known = sha(src_path) in accepted_bases(rel)
        print(f"{sha(src_path)}  {rel}  anchors OK  {'(accepted)' if known else '(NOT in KIT.json)'}")


if __name__ == "__main__":
    {"install": install, "uninstall": uninstall, "status": status, "dry": dry,
     "compose-check": compose_check}[sys.argv[1]]()
