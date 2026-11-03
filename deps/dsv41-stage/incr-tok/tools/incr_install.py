#!/usr/bin/env python3
"""incr-tok kit: anchored install for ONE node. status [patched|base] | dry-run | precheck | install | uninstall. Stdlib only.

precheck = dry-run plus "the service is stopped": site8.sh install runs it on all 8 before it writes anywhere.

install / uninstall refuse while dsv41-rank is active. The base file is guarded by sha256: any other sha stops the install
and nothing is written. The flag defaults off: with DSV41_INCR_TOK unset, an installed site runs the stock statements plus
one module-level env read and one false `if` in DeepseekV4Renderer.__init__, and never imports dsv41_incr_tok.py
(tests/test_incr_tok.py checks the AST difference).

Patched file: renderers/deepseek_v4.py (base f7175f99). New file: renderers/dsv41_incr_tok.py. Backup:
renderers/deepseek_v4.py.bak-incrtok (byte copy of the base, written once, before the first write).

INCR_SITE=<dir> points the tool at another vllm tree (tests only); the service check is skipped there.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
DEFAULT_SITE = Path.home() / "venvs/vllm032/lib/python3.12/site-packages/vllm"
SITE = Path(os.environ.get("INCR_SITE", str(DEFAULT_SITE)))
TARGET = "renderers/deepseek_v4.py"
SIDE = "renderers/dsv41_incr_tok.py"
BASE_SHA = "f7175f9980550e5ccfcc94fcebfdeee113eee66efc7cd1ec539d5b2af1f0d8cd"
BAK = TARGET + ".bak-incrtok"

EDITS = [
    (
        "from vllm.config import VllmConfig\n",
        "import os\n\nfrom vllm.config import VllmConfig\n",
    ),
    (
        '_FIM_END = "<｜fim▁end｜>"\n',
        '_FIM_END = "<｜fim▁end｜>"\n'
        "# incr-tok kit, default off: unset = dsv41_incr_tok.py is never imported.\n"
        '_DSV41_INCR_TOK = os.environ.get("DSV41_INCR_TOK", "0").strip().lower() not in ("", "0", "off")\n',
    ),
    (
        "        self._apply_chat_template_async = make_async(\n"
        "            self._apply_chat_template, executor=self._executor\n"
        "        )\n",
        "        self._apply_chat_template_async = make_async(\n"
        "            self._apply_chat_template, executor=self._executor\n"
        "        )\n"
        "        if _DSV41_INCR_TOK:\n"
        "            from .dsv41_incr_tok import attach\n\n"
        "            attach(self)\n",
    ),
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def patched(text: str) -> str:
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"ABORT: anchor found {n} times in {TARGET}: {old[:60]!r}. Nothing written.")
        text = text.replace(old, new)
    return text


def service_active() -> bool:
    """Fails closed: only a clean "inactive" or "failed" counts as stopped."""
    if SITE.resolve() != DEFAULT_SITE.resolve():
        return False
    r = subprocess.run(["systemctl", "--user", "is-active", "dsv41-rank"], capture_output=True, text=True)
    return r.stdout.strip() not in ("inactive", "failed")


def write(p: Path, b: bytes) -> None:
    tmp = p.with_name(p.name + ".incrtok-tmp")
    tmp.write_bytes(b)
    shutil.copymode(p if p.exists() else tmp, tmp)
    tmp.replace(p)


def side_bytes() -> bytes:
    return (KIT / "site/vllm" / SIDE).read_bytes()


def status(want: str | None = None) -> int:
    """Prints one line per file, then STATE patched|base|UNKNOWN. Exit 1 on UNKNOWN or on a state other than `want`."""
    b = (SITE / TARGET).read_bytes()
    h = sha(b)
    bak = SITE / BAK
    if h == BASE_SHA:
        tstate = "base"
    elif bak.exists() and sha(bak.read_bytes()) == BASE_SHA and patched(bak.read_text()).encode() == b:
        tstate = "patched"
    else:
        tstate = "UNKNOWN"
    print(f"{TARGET}: {h[:8]} {tstate}")
    side = SITE / SIDE
    sh = sha(side.read_bytes()) if side.exists() else None
    print(f"{SIDE}: {sh[:8] if sh else 'absent'}")
    if tstate == "patched" and sh == sha(side_bytes()):
        overall = "patched"
    elif tstate == "base" and sh is None:
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


def install(dry: bool, precheck: bool = False) -> int:
    old = (SITE / TARGET).read_bytes()
    if sha(old) != BASE_SHA:
        raise SystemExit(f"ABORT: {TARGET} is {sha(old)[:8]}, kit expects base {BASE_SHA[:8]}. Nothing written.")
    new = patched(old.decode()).encode()
    side = side_bytes()
    compile(new, TARGET, "exec")
    compile(side, SIDE, "exec")
    if b"__STRUCT_SHA256__" in side:
        raise SystemExit("ABORT: overlay has no pinned structure digest. Nothing written.")
    print(f"{TARGET}: {sha(old)[:8]} -> {sha(new)[:8]}")
    print(f"{SIDE}: new {sha(side)[:8]}")
    if (SITE / SIDE).exists():
        raise SystemExit(f"ABORT: {SIDE} already exists on a base site. Nothing written.")
    if dry:
        if precheck and service_active():
            raise SystemExit("ABORT: dsv41-rank is active. Stopped service only. Nothing written.")
        print("precheck OK" if precheck else "dry-run OK")
        return 0
    if service_active():
        raise SystemExit("ABORT: dsv41-rank is active. Stopped service only. Nothing written.")
    bak = SITE / BAK
    if bak.exists() and sha(bak.read_bytes()) != BASE_SHA:
        raise SystemExit(f"ABORT: {BAK} exists and is not the base. Nothing written.")
    if not bak.exists():
        bak.write_bytes(old)
        shutil.copymode(SITE / TARGET, bak)
    write(SITE / SIDE, side)
    write(SITE / TARGET, new)
    print("installed; DSV41_INCR_TOK defaults off")
    return status("patched")


def uninstall() -> int:
    if service_active():
        raise SystemExit("ABORT: dsv41-rank is active. Stopped service only. Nothing written.")
    cur = (SITE / TARGET).read_bytes()
    bak = SITE / BAK
    if sha(cur) == BASE_SHA:
        print(f"{TARGET}: already base")
    else:
        if not bak.exists() or sha(bak.read_bytes()) != BASE_SHA:
            raise SystemExit(f"ABORT: no good backup {BAK}. Nothing written.")
        write(SITE / TARGET, bak.read_bytes())
        print(f"{TARGET}: restored {BASE_SHA[:8]}")
    if bak.exists() and sha(bak.read_bytes()) == BASE_SHA:
        bak.unlink()
    (SITE / SIDE).unlink(missing_ok=True)
    for pyc in (SITE / "renderers/__pycache__").glob("dsv41_incr_tok.*.pyc"):
        pyc.unlink(missing_ok=True)
    return status("base")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        sys.exit(status(sys.argv[2] if len(sys.argv) > 2 else None))
    if cmd == "dry-run":
        sys.exit(install(True))
    if cmd == "precheck":
        sys.exit(install(True, precheck=True))
    if cmd == "install":
        sys.exit(install(False))
    if cmd == "uninstall":
        sys.exit(uninstall())
    raise SystemExit(__doc__)
