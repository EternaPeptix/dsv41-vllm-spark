#!/usr/bin/env python3
"""climb r1 kit r1-eager: install / uninstall / check the DSV41_DTGRAPH lever (G08 "dtgraph").

    python3 install.py install | uninstall | status | dry

Patches vllm/v1/worker/gpu/model_runner.py (backup model_runner.py.bak-r1-eager; refuses unless the file's
sha256 is the base recorded in KIT.json or it is already patched by this kit) and copies dsv41_dtgraph.py
next to it (vllm/v1/worker/gpu/dsv41_dtgraph.py). Every inserted line carries the tag "# dsv41-r1eager".
With DSV41_DTGRAPH unset/0 the patch is one module-level bool assignment plus two `if False:` tests per
execute_model and one at import: same kernels, launches, streams, events, allocations and graphs.
Idempotent; uninstall restores the backup byte for byte and removes the module.
Env: R1EAGER_VLLM overrides the vllm package directory (tests use a scratch copy).
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("R1EAGER_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
MR = f"{V}/v1/worker/gpu/model_runner.py"
MOD_DST = f"{V}/v1/worker/gpu/dsv41_dtgraph.py"
MOD_SRC = os.path.join(HERE, "dsv41_dtgraph.py")
TAG = "# dsv41-r1eager"
BAK = ".bak-r1-eager"


def _kit():
    return json.load(open(os.path.join(HERE, "KIT.json")))


def _base_sha(rel):
    for f in _kit()["files"]:
        if f["path"] == rel:
            return f.get("base_sha256")
    raise SystemExit(f"KIT.json has no entry for {rel}")


E1_OLD = "    return None if m is None else m.STEER  # dsv41-steer\n"
E1_NEW = (E1_OLD
          + "_DSV41_DTGRAPH = __import__(\"os\").environ.get(\"DSV41_DTGRAPH\", \"\").strip().lower() not in "
          f"(\"\", \"0\", \"off\", \"false\", \"no\")  {TAG}\n"
          f"if _DSV41_DTGRAPH:  {TAG}\n"
          f"    from vllm.v1.worker.gpu import dsv41_dtgraph as _dsv41_dtg  {TAG}\n"
          f"    _DSV41_DTGRAPH = _dsv41_dtg.ON  {TAG}\n")
E2_OLD = ("            assert batch_req_state is not None\n"
          "            input_batch = self.prepare_inputs(\n")
E2_NEW = ("            assert batch_req_state is not None\n"
          f"            if _DSV41_DTGRAPH:  {TAG}\n"
          f"                _dsv41_dtg.dt_begin(self, scheduler_output, batch_req_state, batch_desc)  {TAG}\n"
          "            input_batch = self.prepare_inputs(\n")
E3_OLD = ("        input_ids = input_batch.input_ids\n"
          "        inputs_embeds = None\n")
E3_NEW = (f"        if _DSV41_DTGRAPH:  {TAG}\n"
          f"            _dsv41_dtg.dt_end(self)  {TAG}\n"
          + E3_OLD)
E4_APPEND = (f"\n\nif _DSV41_DTGRAPH:  {TAG}\n"
             f"    _dsv41_dtg.wrap_execute_model(GPUModelRunner)  {TAG}\n")
EDITS = [(E1_OLD, E1_NEW, 1), (E2_OLD, E2_NEW, 1), (E3_OLD, E3_NEW, 1)]


def sha(p, n=64):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:n] if os.path.exists(p) else "-"


def patched(src):
    for old, new, count in EDITS:
        n = src.count(old)
        if n != count:
            sys.exit(f"{MR}: anchor found {n}x, expected {count}: {old[:70]!r}")
        src = src.replace(old, new)
    if not src.endswith("\n"):
        sys.exit(f"{MR}: no trailing newline")
    return src + E4_APPEND


def unpatched(src):
    """Inverse of patched(): drop every tagged line and the two blank lines before the appended block."""
    if src.endswith(E4_APPEND):
        src = src[: -len(E4_APPEND)]
    return "".join(l for l in src.splitlines(keepends=True) if TAG not in l)


def _check_base():
    want = _base_sha("vllm/v1/worker/gpu/model_runner.py")
    have = sha(MR)
    if have != want:
        sys.exit(f"{MR}: sha256 {have[:16]} != KIT.json base {want[:16]}; refusing")


def install():
    src = open(MR).read()
    if TAG in src:
        if hashlib.sha256(unpatched(src).encode()).hexdigest() != _base_sha("vllm/v1/worker/gpu/model_runner.py"):
            sys.exit(f"{MR}: tagged but its untagged text is not the KIT.json base; refusing")
        print(f"{MR}: already patched ({sha(MR, 8)})")
    else:
        _check_base()
        new = patched(src)
        assert unpatched(new) == src, "patch is not reversible by tag"
        bak = MR + BAK
        if not os.path.exists(bak):
            shutil.copy2(MR, bak)
        tmp = MR + ".tmp-r1-eager"
        open(tmp, "w").write(new)
        shutil.copymode(bak, tmp)
        os.replace(tmp, MR)
        print(f"{MR}: patched ({sha(bak, 8)} -> {sha(MR, 8)}), {new.count(TAG)} tagged lines")
    if os.path.exists(MOD_DST) and sha(MOD_DST) != sha(MOD_SRC):
        print(f"{MOD_DST}: replacing {sha(MOD_DST, 8)}")
    shutil.copy2(MOD_SRC, MOD_DST)
    print(f"{MOD_DST}: {sha(MOD_DST, 8)}")


def uninstall():
    bak = MR + BAK
    if os.path.exists(bak):
        os.replace(bak, MR)
        print(f"{MR}: restored {sha(MR, 8)}")
    elif TAG in open(MR).read():
        sys.exit(f"{MR}: patched but {bak} is missing; restore by hand")
    if os.path.exists(MOD_DST):
        os.remove(MOD_DST)
        print(f"{MOD_DST}: removed")


def status():
    base = _base_sha("vllm/v1/worker/gpu/model_runner.py")
    src = open(MR).read() if os.path.exists(MR) else ""
    state = "PATCHED" if TAG in src else ("base   " if sha(MR) == base else "OTHER  ")
    print(f"{sha(MR, 8)} {state} {MR}")
    m = "present" if os.path.exists(MOD_DST) else "absent "
    same = " (= kit)" if os.path.exists(MOD_DST) and sha(MOD_DST) == sha(MOD_SRC) else ""
    print(f"{sha(MOD_DST, 8)} {m} {MOD_DST}{same}")


def dry():
    _check_base()
    new = patched(open(MR).read())
    for i, line in enumerate(new.splitlines(), 1):
        if TAG in line:
            print(f"{i:5d}: {line}")
    print(f"result sha256 {hashlib.sha256(new.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("install", "uninstall", "status", "dry"):
        sys.exit(__doc__)
    {"install": install, "uninstall": uninstall, "status": status, "dry": dry}[sys.argv[1]]()
