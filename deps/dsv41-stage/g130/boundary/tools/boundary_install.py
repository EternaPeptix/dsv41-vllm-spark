#!/usr/bin/env python3
"""g130 boundary kit: anchored install for ONE node. status|install|uninstall|dry-run. Stdlib only, no torch, no GPU.

install / uninstall refuse while dsv41-rank is active. Bases are guarded by sha256: a changed file stops the install and
nothing is written. Every flag defaults off, so an installed site with flags unset runs the same statements as production
apart from the module-level flag reads and three false `if`s (tests/test_boundary_cpu.py checks the AST difference).

Patched files: models/deepseek_v41/nvidia/dspark.py, v1/worker/gpu/spec_decode/dspark/speculator.py.
New file: models/deepseek_v41/nvidia/dsv41_boundary.py. Backups: <file>.bak-boundary.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
SITE = Path(os.environ.get("BOUNDARY_SITE", str(Path.home() / "venvs/vllm032/lib/python3.12/site-packages/vllm")))
DSPARK = "models/deepseek_v41/nvidia/dspark.py"
SPEC = "v1/worker/gpu/spec_decode/dspark/speculator.py"
SIDE = "models/deepseek_v41/nvidia/dsv41_boundary.py"
BASE = {DSPARK: "36811ca9", SPEC: "439b390e"}

FLAG_LINE = '{name} = os.environ.get("{env}", "0").strip().lower() not in ("", "0", "off")\n'

EDITS = {
    DSPARK: [
        (
            '_MTP_NAME_RE = re.compile(r"mtp\\.\\d+\\.")\n',
            '_MTP_NAME_RE = re.compile(r"mtp\\.\\d+\\.")\n'
            "# g130 boundary (a), default off: unset = dsv41_boundary.py is never imported.\n"
            + FLAG_LINE.format(name="_DSV41_CTXKV_FOLD", env="DSV41_CTXKV_FOLD"),
        ),
        (
            "        for i, layer in enumerate(self.layers):\n            slot_mapping = (\n",
            "        if _DSV41_CTXKV_FOLD:\n"
            "            from .dsv41_boundary import ctxkv_precompute\n\n"
            "            if ctxkv_precompute(\n"
            "                self, main_x, context_positions, context_slot_mappings, _insert_context_kv\n"
            "            ):\n"
            "                return\n"
            "        for i, layer in enumerate(self.layers):\n            slot_mapping = (\n",
        ),
    ],
    SPEC: [
        (
            '_DSV41_GREEDY_SHARDED = os.environ.get("DSV41_DSPARK_GREEDY_SHARDED", "0") == "1"\n',
            '_DSV41_GREEDY_SHARDED = os.environ.get("DSV41_DSPARK_GREEDY_SHARDED", "0") == "1"\n'
            "# g130 boundary, default off: unset = dsv41_boundary.py is never imported.\n"
            + FLAG_LINE.format(name="_DSV41_MK_FOLD", env="DSV41_MK_FOLD")
            + FLAG_LINE.format(name="_DSV41_CTXKV_FOLD", env="DSV41_CTXKV_FOLD"),
        ),
        (
            "    def capture(self) -> None:\n        super().capture()\n",
            "    def capture(self) -> None:\n"
            "        if _DSV41_MK_FOLD or _DSV41_CTXKV_FOLD:\n"
            "            from vllm.models.deepseek_v41.nvidia.dsv41_boundary import prebuild\n\n"
            "            prebuild(self)\n"
            "        super().capture()\n",
        ),
        (
            "        self._dsv41_greedy_steps += 1\n",
            "        self._dsv41_greedy_steps += 1\n"
            "        if _DSV41_MK_FOLD or _DSV41_CTXKV_FOLD:\n"
            "            from vllm.models.deepseek_v41.nvidia.dsv41_boundary import audit_tick\n\n"
            "            audit_tick(self._dsv41_greedy_steps)\n",
        ),
        (
            "        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]\n"
            "        for i in range(n_spec):\n"
            "            markov_embed = self.model.markov_embed(prev)\n"
            "            if self.use_confidence_head:\n"
            "                confidence_markov_embeds.append(markov_embed)\n"
            '            bias = lp(st["shim"], markov_embed, skip_gather=True)\n',
            "        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]\n"
            "        _mk_done = False\n"
            "        if _DSV41_MK_FOLD:\n"
            "            from vllm.models.deepseek_v41.nvidia.dsv41_boundary import markov_fold_loop\n\n"
            "            _mk_done = markov_fold_loop(\n"
            "                self, st, lp, base, prev, num_reqs, n_spec,\n"
            "                confidence_markov_embeds, tensor_model_parallel_all_gather,\n"
            "            )\n"
            "        for i in range(0 if _mk_done else n_spec):\n"
            "            markov_embed = self.model.markov_embed(prev)\n"
            "            if self.use_confidence_head:\n"
            "                confidence_markov_embeds.append(markov_embed)\n"
            '            bias = lp(st["shim"], markov_embed, skip_gather=True)\n',
        ),
    ],
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def patched(rel: str, text: str) -> str:
    for old, new in EDITS[rel]:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"ABORT: anchor found {n} times in {rel}: {old[:60]!r}")
        text = text.replace(old, new)
    return text


def service_active() -> bool:
    """Fails closed: only a clean "inactive" or "failed" counts as stopped. No bus, no answer or "activating" counts as running."""
    r = subprocess.run(["systemctl", "--user", "is-active", "dsv41-rank"], capture_output=True, text=True)
    return r.stdout.strip() not in ("inactive", "failed")


def write(p: Path, b: bytes) -> None:
    tmp = p.with_name(p.name + ".boundary-tmp")
    tmp.write_bytes(b)
    shutil.copymode(p if p.exists() else tmp, tmp)
    tmp.replace(p)


def status(want: str | None = None) -> int:
    """Last line is STATE patched|base|UNKNOWN. Exit 1 on UNKNOWN, and on anything other than `want` when it is given."""
    states = []
    for rel, base in BASE.items():
        b = (SITE / rel).read_bytes()
        h = sha(b)
        if h.startswith(base):
            state = "base (not installed)"
        else:
            bak = SITE / (rel + ".bak-boundary")
            ok = bak.exists() and sha(bak.read_bytes()).startswith(base) and patched(rel, bak.read_text()).encode() == b
            state = "patched" if ok else "UNKNOWN"
        states.append(state)
        print(f"{rel}: {h[:8]} {state}")
    side = SITE / SIDE
    side_h = sha(side.read_bytes()) if side.exists() else None
    print(f"{SIDE}: {side_h[:8] if side_h else 'absent'}")
    if all(s == "patched" for s in states) and side_h == sha((KIT / "site/vllm" / SIDE).read_bytes()):
        overall = "patched"
    elif all(s.startswith("base") for s in states) and side_h is None:
        overall = "base"
    else:
        overall = "UNKNOWN"
    print(f"STATE {overall}")
    if overall == "UNKNOWN":
        return 1
    if want and overall != want:
        print(f"ABORT: state {overall}, wanted {want}")
        return 1
    return 0


def install(dry: bool) -> int:
    plan = {}
    for rel, base in BASE.items():
        b = (SITE / rel).read_bytes()
        if not sha(b).startswith(base):
            raise SystemExit(f"ABORT: {rel} is {sha(b)[:8]}, kit expects base {base}. Nothing written.")
        plan[rel] = (b, patched(rel, b.decode()).encode())
    side = (KIT / "site/vllm" / SIDE).read_bytes()
    for rel, (old, new) in plan.items():
        compile(new, rel, "exec")
        print(f"{rel}: {sha(old)[:8]} -> {sha(new)[:8]}")
    compile(side, SIDE, "exec")
    print(f"{SIDE}: new {sha(side)[:8]}")
    if dry:
        return 0
    if service_active():
        raise SystemExit("ABORT: dsv41-rank is active. Stopped service only. Nothing written.")
    for rel, (old, new) in plan.items():
        bak = SITE / (rel + ".bak-boundary")
        if not bak.exists():
            bak.write_bytes(old)
        write(SITE / rel, new)
    (SITE / SIDE).write_bytes(side)
    print("installed; flags default off")
    return status("patched")


def uninstall() -> int:
    if service_active():
        raise SystemExit("ABORT: dsv41-rank is active. Stopped service only. Nothing written.")
    for rel, base in BASE.items():
        bak = SITE / (rel + ".bak-boundary")
        if sha((SITE / rel).read_bytes()).startswith(base):
            print(f"{rel}: already base")
            continue
        if not bak.exists() or not sha(bak.read_bytes()).startswith(base):
            raise SystemExit(f"ABORT: no good backup for {rel}")
        write(SITE / rel, bak.read_bytes())
        bak.unlink()
        print(f"{rel}: restored {base}")
    (SITE / SIDE).unlink(missing_ok=True)
    return status("base")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        sys.exit(status(sys.argv[2] if len(sys.argv) > 2 else None))
    if cmd == "dry-run":
        sys.exit(install(True))
    if cmd == "install":
        sys.exit(install(False))
    if cmd == "uninstall":
        sys.exit(uninstall())
    raise SystemExit(__doc__)
