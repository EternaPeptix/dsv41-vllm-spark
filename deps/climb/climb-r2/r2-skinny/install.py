#!/usr/bin/env python3
"""r2-skinny installer (climb round 2): skinny_mx v2 for the qkv_a shard (G10) + fused qkv_a quant (G09/D2).

    python3 install.py install | uninstall | status | dry

* patches  vllm/models/deepseek_v41/nvidia/dsv41_r3_dense.py   (4 anchor edits, inserted lines only, each tagged
           "# r2-skinny"; backup .bak-r2-skinny; refuses unless the file's sha256 is one of KIT.json base_sha256s
           or it is already patched by this kit)
* copies   new/dsv41_r2_skinny.py -> vllm/models/deepseek_v41/nvidia/dsv41_r2_skinny.py
* copies   prebuilt/skinny_mx_v2_ext.so (sha256 must equal KIT.json's; a null/missing KIT.json sha256 or any
           mismatch is refused) and kernel/skinny_mx_v2.cu -> ~/dsv41-r2/skinny_mx_v2/{build/skinny_mx_v2_ext.so, ...}
           An .so already at the destination is accepted only if its sha256 equals KIT.json's.
           `install --build` (escape hatch, NOT device-tested) builds an .so on this node instead; status then
           reports it as UNVERIFIED because its bytes differ from the device-tested prebuilt.

With DSV41_SKINNY_V2 unset the patched file runs exactly the old code: every inserted branch is behind the
module-level bool R2_V2 / R2_QFUSE (False), dsv41_r2_skinny is never imported and the .so is never loaded.
Uninstall removes exactly the tagged lines (so it composes with kits installed after this one), checks the result
against the recorded base sha256s, and removes the copied module and .so directory.
An already-patched file is accepted only if re-patching its tag-stripped base reproduces it byte for byte
(a stale patch from an older kit revision is refused: uninstall it with that revision's install.py first).
Env: R2SK_VLLM (vllm package dir), R2SK_HOME (home for ~/dsv41-r2), R2SK_KIT_JSON, R2SK_PREBUILT (tests).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = json.load(open(os.environ.get("R2SK_KIT_JSON") or os.path.join(HERE, "KIT.json")))
V = os.environ.get("R2SK_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
HOME = os.environ.get("R2SK_HOME") or os.path.expanduser("~")
TAG = "# r2-skinny"
DENSE = f"{V}/models/deepseek_v41/nvidia/dsv41_r3_dense.py"
MOD_SRC = os.path.join(HERE, "new", "dsv41_r2_skinny.py")
MOD_DST = f"{V}/models/deepseek_v41/nvidia/dsv41_r2_skinny.py"
SO_SRC = os.environ.get("R2SK_PREBUILT") or os.path.join(HERE, "prebuilt", "skinny_mx_v2_ext.so")
CU_SRC = os.path.join(HERE, "kernel", "skinny_mx_v2.cu")
LIB = f"{HOME}/dsv41-r2/skinny_mx_v2"
SO_DST = f"{LIB}/build/skinny_mx_v2_ext.so"

EDITS = [  # (anchor, replacement, expected count); replacements only ADD tagged lines around the anchor
    (
        '_SKINNY_DIR = os.environ.get("DSV41_R3_SKINNY_LIB", "/home/jeweled/dsv41-r2/skinny_mx/build")\n',
        '_SKINNY_DIR = os.environ.get("DSV41_R3_SKINNY_LIB", "/home/jeweled/dsv41-r2/skinny_mx/build")\n'
        f'R2_V2 = os.environ.get("DSV41_SKINNY_V2", "0").strip().lower() in ("1", "pdl")  {TAG}\n'
        f'if not R2_V2 and os.environ.get("DSV41_SKINNY_V2", "0").strip().lower() not in ("", "0", "off"):  {TAG}\n'
        f'    raise ValueError("DSV41_SKINNY_V2 must be 1|pdl (or 0/unset)")  {TAG}\n'
        f'R2_QFUSE = R2_V2 and os.environ.get("DSV41_SKINNY_QFUSE", "0").strip() == "1"  {TAG}\n'
        f'if not R2_V2 and os.environ.get("DSV41_SKINNY_QFUSE", "0").strip() == "1":  {TAG}\n'
        f'    logger.warning("DSV41_SKINNY_QFUSE=1 ignored: it needs DSV41_SKINNY_V2=1|pdl")  {TAG}\n',
        1,
    ),
    (
        '        if QKV_GEMM == "skinny":\n'
        "            skinny_ext()\n",
        '        if QKV_GEMM == "skinny":\n'
        "            skinny_ext()\n"
        f"        if R2_V2:  {TAG}\n"
        f"            try:  {TAG}\n"
        f"                from vllm.models.deepseek_v41.nvidia import dsv41_r2_skinny as _r2s  {TAG}\n"
        f'                self._r2 = _r2s.build(self) if QKV_GEMM == "skinny" else None  {TAG}\n'
        f"            except Exception as _e:  # noqa: BLE001 - never let v2 disable the live shard path  {TAG}\n"
        f'                _warn_once("r2_skinny", "DSV41_SKINNY_V2 disabled (live skinny kept): %r", _e)  {TAG}\n'
        f"                self._r2 = None  {TAG}\n",
        1,
    ),
    (
        "    def gemm(self, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:\n"
        "        M = xq.shape[0]\n",
        "    def gemm(self, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:\n"
        "        M = xq.shape[0]\n"
        f"        if R2_V2 and self._r2 is not None:  {TAG}\n"
        f"            _y = self._r2.gemm_a8(xq, xs)  {TAG}\n"
        f"            if _y is not None:  {TAG}\n"
        f"                return _y  {TAG}\n",
        1,
    ),
    (
        "        xq, xs, shape = _quantize_act(x, self.K)\n"
        "        y = self.gemm(xq, xs)  # [M, n] bf16: this rank's contiguous output rows\n",
        f"        if R2_QFUSE and self._r2 is not None:  {TAG}\n"
        f"            _y = self._r2.fused(x)  {TAG}\n"
        f"            if _y is not None:  {TAG}\n"
        f"                return tensor_model_parallel_all_gather(_y, dim=-1).view(*x.shape[:-1], self.N)  {TAG}\n"
        "        xq, xs, shape = _quantize_act(x, self.K)\n"
        "        y = self.gemm(xq, xs)  # [M, n] bf16: this rank's contiguous output rows\n",
        1,
    ),
]
N_TAGGED = 21


def sha(p, n=64):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:n] if os.path.exists(p) else "-"


def base_shas(rel):
    for f in KIT["files"]:
        if f["path"] == rel:
            return set(f.get("base_sha256_accepted") or [f["base_sha256"]])
    return set()


def patched(src, path=DENSE):
    for old, new, count in EDITS:
        n = src.count(old)
        if n != count:
            sys.exit(f"{path}: anchor found {n}x, expected {count}: {old[:80]!r}")
        src = src.replace(old, new)
    if src.count(TAG) != N_TAGGED:
        sys.exit(f"{path}: {src.count(TAG)} tagged lines after patch, expected {N_TAGGED}")
    return src


def unpatched(src):
    return "".join(line for line in src.splitlines(keepends=True) if TAG not in line)


def so_expected():
    for f in KIT["files"]:
        if f["path"].endswith("skinny_mx_v2_ext.so"):
            return f.get("sha256")
    return None


def write_atomic(path, text, mode_from):
    tmp = path + ".tmp-r2-skinny"
    open(tmp, "w").write(text)
    shutil.copymode(mode_from, tmp)
    os.replace(tmp, path)


def install():
    build = "--build" in sys.argv
    src = open(DENSE).read()
    rel = "models/deepseek_v41/nvidia/dsv41_r3_dense.py"
    want = so_expected()
    if not build:  # check the binary before touching anything
        if not want:
            sys.exit("KIT.json has no sha256 for skinny_mx_v2_ext.so (no device-tested binary); refusing "
                     "(install --build builds an untested one)")
        if os.path.exists(SO_SRC):
            if sha(SO_SRC) != want:
                sys.exit(f"{SO_SRC}: sha256 {sha(SO_SRC)} != KIT.json {want}; refusing")
        elif not os.path.exists(SO_DST):
            sys.exit(f"no prebuilt .so in the kit and none at {SO_DST}; refusing (or rerun with --build)")
        elif sha(SO_DST) != want:
            sys.exit(f"{SO_DST}: sha256 {sha(SO_DST)} != KIT.json {want} and no prebuilt in the kit; refusing")
    if TAG in src:
        base = unpatched(src)
        if hashlib.sha256(base.encode()).hexdigest() not in base_shas(rel):
            sys.exit(f"{DENSE}: carries {TAG} lines but its base is not a recorded base sha256; refusing")
        if patched(base) != src:
            sys.exit(f"{DENSE}: carries {TAG} lines that differ from this kit revision's edits (stale patch?); "
                     "refusing: uninstall first")
        print(f"{DENSE}: already patched ({sha(DENSE, 16)})")
    else:
        if sha(DENSE) not in base_shas(rel):
            sys.exit(f"{DENSE}: sha256 {sha(DENSE)} is not a recorded base ({sorted(base_shas(rel))}); refusing")
        new = patched(src)
        bak = DENSE + ".bak-r2-skinny"
        if not os.path.exists(bak):
            shutil.copy2(DENSE, bak)
        write_atomic(DENSE, new, bak)
        print(f"{DENSE}: patched ({sha(bak, 16)} -> {sha(DENSE, 16)}), {new.count(TAG)} tagged lines")
    shutil.copy2(MOD_SRC, MOD_DST)
    print(f"{MOD_DST}: {sha(MOD_DST, 16)}")
    os.makedirs(f"{LIB}/build", exist_ok=True)
    shutil.copy2(CU_SRC, f"{LIB}/skinny_mx_v2.cu")
    shutil.copy2(os.path.join(HERE, "kernel", "build.py"), f"{LIB}/build.py")
    if build:
        py = os.path.expanduser("~/venvs/vllm032/bin/python")
        subprocess.run(["nice", "-n", "19", py, f"{LIB}/build.py"], check=True,
                       env=dict(os.environ, MAX_JOBS="1"))
    elif os.path.exists(SO_SRC):
        shutil.copy2(SO_SRC, SO_DST)
    ok = bool(want) and sha(SO_DST) == want
    print(f"{SO_DST}: {sha(SO_DST, 16)}" + ("  (== KIT.json, device-tested binary)" if ok else
                                           "  (WARNING: UNVERIFIED, != KIT.json sha256: not the device-tested binary)"))


def uninstall():
    rel = "models/deepseek_v41/nvidia/dsv41_r3_dense.py"
    src = open(DENSE).read()
    if TAG in src:
        clean = unpatched(src)
        h = hashlib.sha256(clean.encode()).hexdigest()
        if h not in base_shas(rel):
            print(f"{DENSE}: WARNING tag-stripped sha256 {h[:16]} is not a recorded base (another kit changed it?)")
        write_atomic(DENSE, clean, DENSE)
        print(f"{DENSE}: {TAG} lines removed -> {sha(DENSE, 16)}")
    else:
        print(f"{DENSE}: not patched ({sha(DENSE, 16)})")
    bak = DENSE + ".bak-r2-skinny"
    if os.path.exists(bak) and sha(bak) == sha(DENSE):
        os.remove(bak)
    if os.path.exists(MOD_DST):
        os.remove(MOD_DST)
        print(f"{MOD_DST}: removed")
    if os.path.isdir(LIB):
        shutil.rmtree(LIB)
        print(f"{LIB}: removed")


def status():
    on = os.path.exists(DENSE) and TAG in open(DENSE).read()
    print(f"{sha(DENSE, 16)} {'PATCHED' if on else 'clean  '} {DENSE}")
    print(f"{sha(MOD_DST, 16)} {'present' if os.path.exists(MOD_DST) else 'absent '} {MOD_DST}")
    want = so_expected()
    if not os.path.exists(SO_DST):
        st = "absent "
    elif not want:
        st = "UNVERIFIED (KIT.json sha256 null)"
    else:
        st = "present (== KIT.json)" if sha(SO_DST) == want else "MISMATCH (!= KIT.json)"
    print(f"{sha(SO_DST, 16)} {st} {SO_DST}")
    if on:
        cur = open(DENSE).read()
        try:
            same = patched(unpatched(cur)) == cur
        except SystemExit as e:
            same = False
            print(f"  ({e})")
        print("patch matches this kit revision" if same else
              "WARNING: tagged lines differ from this kit revision (stale patch)")


def dry():
    src = open(DENSE).read()
    if TAG in src:
        src = unpatched(src)
    rel = "models/deepseek_v41/nvidia/dsv41_r3_dense.py"
    h = hashlib.sha256(src.encode()).hexdigest()
    print(f"base sha256 {h[:16]} {'ACCEPTED' if h in base_shas(rel) else 'NOT A RECORDED BASE'}")
    for i, line in enumerate(patched(src).splitlines(), 1):
        if TAG in line:
            print(f"{i:5d}: {line}")


if __name__ == "__main__":
    {"install": install, "uninstall": uninstall, "status": status, "dry": dry}[sys.argv[1]]()
