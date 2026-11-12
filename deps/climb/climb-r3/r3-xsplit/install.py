#!/usr/bin/env python3
"""r3-xsplit installer (lever G02, attn:L-XS, flag DSV41_ATTN_XSPLIT).

    python3 install.py install | uninstall | status | dry

* patches  flashinfer/mla/_sparse_mla_sm120.py  (2 anchored edits, every inserted line tagged
           '# dsv41-r3-xsplit', backup .bak-r3-xsplit; refuses unless the file's sha256 is the
           recorded base)
* copies   flashinfer/mla/dsv41_xsplit.py       (new, the gate; imported only when the flag is 1)
* copies   ~/dsv41-r3/xsplit/build/libdsv41_xsplit.so (+ .sha256 sidecar), the prebuilt kernel
           library from build/ in this kit; refuses unless its sha256 is the one in KIT.json

With DSV41_ATTN_XSPLIT unset or 0 the patched function does one extra module-level bool test
per call and nothing else: same planner, same binding call, same kernels, same graphs.
Idempotent; never touches the JIT cache or the live kernel source.
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.environ.get("R3_XSPLIT_SP") or os.path.expanduser(
    "~/venvs/vllm032/lib/python3.12/site-packages")
TARGET = f"{SP}/flashinfer/mla/_sparse_mla_sm120.py"
MOD_DST = f"{SP}/flashinfer/mla/dsv41_xsplit.py"
MOD_SRC = os.path.join(HERE, "files/flashinfer/mla/dsv41_xsplit.py")
LIB_DIR = os.environ.get("R3_XSPLIT_LIB_DIR") or os.path.expanduser("~/dsv41-r3/xsplit/build")
LIB_SRC = os.path.join(HERE, "build/libdsv41_xsplit.so")
LIB_DST = os.path.join(LIB_DIR, "libdsv41_xsplit.so")
TAG = "# dsv41-r3-xsplit"
BAK = TARGET + ".bak-r3-xsplit"

KIT = json.load(open(os.path.join(HERE, "KIT.json")))


def _file_entry(suffix):
    for f in KIT["files"]:
        if f["path"].endswith(suffix):
            return f
    raise KeyError(suffix)


BASE_SHA = _file_entry("_sparse_mla_sm120.py")["base_sha256"]
LIB_SHA = _file_entry("libdsv41_xsplit.so")["sha256"]

FLAG_OLD = "import torch\n\nfrom ..api_logging import flashinfer_api\n"
FLAG_NEW = (
    "import torch\n"
    "_DSV41_ATTN_XSPLIT = __import__('os').environ.get('DSV41_ATTN_XSPLIT', '0') == '1'  "
    f"{TAG}\n"
    "\nfrom ..api_logging import flashinfer_api\n")
GATE_OLD = "        if planned.variant is KernelVariant.DECODE_SPLITK:\n"
GATE_NEW = (
    f"        if _DSV41_ATTN_XSPLIT and planned.variant is KernelVariant.PREFILL_MG_DUAL:  {TAG}\n"
    f"            from . import dsv41_xsplit as _dsv41_xs  {TAG}\n"
    f"            if _dsv41_xs.try_run(module, int(planned.variant), q, kv_cache, indices,  {TAG}\n"
    f"                                 output, out_lse, sm_scale, model_type, topk_length,  {TAG}\n"
    f"                                 attn_sink, extra_kv_cache, extra_indices,  {TAG}\n"
    f"                                 extra_topk_length):  {TAG}\n"
    f"                return  {TAG}\n"
    "        if planned.variant is KernelVariant.DECODE_SPLITK:\n")
EDITS = [(FLAG_OLD, FLAG_NEW, 1), (GATE_OLD, GATE_NEW, 1)]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def patched(src):
    for old, new, count in EDITS:
        n = src.count(old)
        if n != count:
            sys.exit(f"{TARGET}: anchor found {n}x, expected {count}: {old[:70]!r}")
        src = src.replace(old, new)
    return src


def install():
    src = open(TARGET).read()
    if TAG in src:
        bsha = sha(BAK)
        if bsha != BASE_SHA:
            sys.exit(f"{TARGET}: tagged but backup sha {bsha} != base {BASE_SHA}; refusing")
        if patched(open(BAK).read()) != src:
            sys.exit(f"{TARGET}: tagged but differs from this kit's patch; refusing")
        print(f"{TARGET}: already patched ({sha(TARGET)[:12]})")
    else:
        cur = sha(TARGET)
        if cur != BASE_SHA:
            sys.exit(f"{TARGET}: sha256 {cur} != recorded base {BASE_SHA}; refusing")
        new = patched(src)
        if not os.path.exists(BAK):
            shutil.copy2(TARGET, BAK)
        tmp = TARGET + ".tmp-r3-xsplit"
        with open(tmp, "w") as f:
            f.write(new)
        shutil.copymode(BAK, tmp)
        os.replace(tmp, TARGET)
        print(f"{TARGET}: patched ({cur[:12]} -> {sha(TARGET)[:12]}), {new.count(TAG)} tagged lines")
    shutil.copy2(MOD_SRC, MOD_DST)
    print(f"{MOD_DST}: {sha(MOD_DST)[:12]}")
    lsha = sha(LIB_SRC)
    if lsha is None:
        sys.exit(f"{LIB_SRC} missing: build it first (src/build.sh on spark-eight)")
    if lsha != LIB_SHA:
        sys.exit(f"{LIB_SRC}: sha256 {lsha} != KIT.json {LIB_SHA}; refusing")
    os.makedirs(LIB_DIR, exist_ok=True)
    shutil.copy2(LIB_SRC, LIB_DST)
    with open(LIB_DST + ".sha256", "w") as f:
        f.write(LIB_SHA + "\n")
    if sha(LIB_DST) != LIB_SHA:
        sys.exit(f"{LIB_DST}: copy sha mismatch")
    print(f"{LIB_DST}: {LIB_SHA[:12]}")


def uninstall():
    if os.path.exists(BAK):
        os.replace(BAK, TARGET)
        print(f"{TARGET}: restored {sha(TARGET)[:12]}")
    elif os.path.exists(TARGET) and TAG in open(TARGET).read():
        sys.exit(f"{TARGET}: tagged but no backup; refusing to guess")
    for p in (MOD_DST, LIB_DST, LIB_DST + ".sha256"):
        if os.path.exists(p):
            os.remove(p)
            print(f"removed {p}")
    try:
        os.rmdir(LIB_DIR)
    except OSError:
        pass


def status():
    t = sha(TARGET)
    on = os.path.exists(TARGET) and TAG in open(TARGET).read()
    print(f"{(t or '-')[:12]} {'PATCHED' if on else 'clean  '} {TARGET}"
          f"{'  (base ok)' if t == BASE_SHA else ''}")
    for p in (MOD_DST, LIB_DST):
        s = sha(p)
        print(f"{(s or '-')[:12]} {'present' if s else 'absent '} {p}")
    s = sha(LIB_DST)
    if s is not None:
        print(f"lib sha {'OK' if s == LIB_SHA else 'MISMATCH'}")


def dry():
    for i, line in enumerate(patched(open(TARGET).read()).splitlines(), 1):
        if TAG in line:
            print(f"{i:5d}: {line}")


if __name__ == "__main__":
    {"install": install, "uninstall": uninstall, "status": status, "dry": dry}[sys.argv[1]]()
