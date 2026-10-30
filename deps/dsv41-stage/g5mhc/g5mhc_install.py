#!/usr/bin/env python3
"""g5mhc_install.py status|install|uninstall  -- DSV41_MHC_OVERLAP (goal window 5, mHC step), sha-guarded.

Installs, in ~/venvs/vllm032/lib/python3.12/site-packages/vllm/models/deepseek_v41/nvidia:
  model.py               9c2d22ad -> the patched model.py (6 hunks, every one inert while DSV41_MHC_OVERLAP and
                         DSV41_MHC_OVERLAP_DUMP are unset)
  dsv41_mhc_overlap.py   new file (imported only when one of those two variables is set)
dspark.py (a supervisor-manifest file) is NOT touched by this bundle.
Backups: model.py.bak-g5-mhc (= 9c2d22ad); dsv41_mhc_overlap.py.absent-g5-mhc (marker: the file did not exist).
Stage dir = the directory of this script (files under ./site/).
Exit 0 = ok / state printed; 1 = refused or a sha check failed.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parent
SITE = Path(os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm"))
NV = SITE / "models/deepseek_v41/nvidia"
TAG = "g5-mhc"
# (name, base sha256 or None = absent before the install, file in the stage dir)
FILES = [
    ("model.py", "9c2d22ad94a79dbb7135cf6537c8c7251f401e9af275c30ad6b9b461089b2600", "site/model.py"),
    ("dsv41_mhc_overlap.py", None, "site/dsv41_mhc_overlap.py"),
]


def sha(p: Path) -> str | None:
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def state() -> tuple[str, list[str]]:
    lines, states = [], set()
    for name, base, src in FILES:
        live = sha(NV / name)
        new = sha(STAGE / src)
        if live == new:
            st = "installed"
        elif live == base:
            st = "base"
        else:
            st = "OTHER"
        states.add(st)
        bak = NV / f"{name}.bak-{TAG}"
        absent = NV / f"{name}.absent-{TAG}"
        lines.append(
            f"{name}: live {(live or 'absent')[:12]} base {(base or 'absent')[:12]} new {(new or '?')[:12]} -> {st}"
            f" | bak {(sha(bak) or '-')[:12]} absent-marker {'yes' if absent.exists() else 'no'}"
        )
    total = states.pop() if len(states) == 1 else "MIXED"
    return total, lines


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    host = os.uname().nodename
    total, lines = state()
    if cmd == "status":
        for ln in lines:
            print(f"[{host}] {ln}")
        print(f"[{host}] STATE {total}")
        return 0 if total in ("base", "installed") else 1
    if cmd == "install":
        if total == "installed":
            print(f"[{host}] already installed")
            return 0
        if total != "base":
            print(f"[{host}] REFUSED: state {total}")
            for ln in lines:
                print(f"[{host}]   {ln}")
            return 1
        for name, base, src in FILES:
            dst = NV / name
            if base is None:
                (NV / f"{name}.absent-{TAG}").write_text("absent before g5mhc install\n")
            else:
                shutil.copy2(dst, NV / f"{name}.bak-{TAG}")
                if sha(NV / f"{name}.bak-{TAG}") != base:
                    print(f"[{host}] backup sha mismatch for {name}")
                    return 1
            tmp = NV / f".{name}.g5mhc-tmp"
            shutil.copyfile(STAGE / src, tmp)
            os.replace(tmp, dst)
        total, lines = state()
        for ln in lines:
            print(f"[{host}] {ln}")
        print(f"[{host}] STATE {total}")
        return 0 if total == "installed" else 1
    if cmd == "uninstall":
        if total == "base":
            print(f"[{host}] already at base")
            return 0
        for name, base, src in FILES:
            dst = NV / name
            if base is None:
                if dst.exists():
                    dst.unlink()
                (NV / f"{name}.absent-{TAG}").unlink(missing_ok=True)
            else:
                bak = NV / f"{name}.bak-{TAG}"
                if sha(bak) != base:
                    print(f"[{host}] REFUSED: backup {bak} is not the base")
                    return 1
                tmp = NV / f".{name}.g5mhc-tmp"
                shutil.copyfile(bak, tmp)
                os.replace(tmp, dst)
        total, lines = state()
        for ln in lines:
            print(f"[{host}] {ln}")
        print(f"[{host}] STATE {total}")
        return 0 if total == "base" else 1
    print(f"unknown command {cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
