#!/usr/bin/env python3
"""r6-comm installer (climb round 6): rdma_ar C1 (G06 trim, DSV41_RDMA_TRIM) + C2 (PDL on the collective edges,
DSV41_RDMA_PDL). Supersedes climb-r3 r3-comm (never installed: its build failed the old regex SASS gate).

    python3 install.py install | uninstall | status | dry

* patches  vllm/distributed/device_communicators/rdma_all_reduce.py   (anchor edits, inserted lines only, each tagged
           "# r6-comm"; backup .bak-r6-comm; refuses unless the file's sha256 is the recorded base 26a5d834... (live,
           unchanged by every climb r1..r5 kit) or the file is already exactly this kit's patch of that base; refuses
           while r3-comm lines ("# r3-comm") are present)
* stages   ~/dsv41-stage/r6comm/{native/rdma_ar.cu, native/mk_r6.py, build.sh, build/rdma_ar_ext.so}
           build/rdma_ar_ext.so = prebuilt/rdma_ar_ext.so from the kit: the ONE binary built on spark-eight by
           ops/build8-r6.sh, whose sha256 KIT.json records and which gate/run_gate.py passed (cases D3-D6). Never compiles.
           NEVER touches the live .so ~/dsv41-stage/g130/rdma/build/rdma_ar_ext.so (14d049dd) or its tree.
Order: every check runs before the first write; then the stage dir; rdma_all_reduce.py is patched LAST and restored from
the backup on any later failure (a failed install leaves it byte-identical).
Nothing selects the new .so: the launcher default stays the live one. An arm sets
RDMA_AR_EXT=$HOME/dsv41-stage/r6comm/build/rdma_ar_ext.so (+ DSV41_RDMA_TRIM=1 [+ DSV41_RDMA_PDL=1|trig]).

Flags off (DSV41_RDMA_TRIM, DSV41_RDMA_SENDBUF, DSV41_RDMA_PDL unset): the patched module runs exactly the old per-call
code: the inserted lines are module-level flag parsing, startup-only checks behind the module bools, a startup log line
and one method nobody calls (send_buffer, G26, bench only). Uninstall removes exactly the tagged lines, checks the result
against the base sha256 and removes what install staged; ~/dsv41-stage/r6comm/kit (ops/stage8-r6.sh, with results) stays.
Gate (review N4): install also refuses unless gate/results/SUMMARY.txt (staged with the kit by ops/stage8-r6.sh) says
"RUN_GATE PASS" and carries "GATED_SO_SHA256 <the sha KIT.json records>", which gate/run_gate.py writes only when D3-D6
passed on that exact build's SASS. (build8-r6.sh records the sha in KIT.json right after compiling, before the gate.)
Env (tests): R6C_VLLM (vllm package dir), R6C_HOME (home), R6C_KIT_JSON, R6C_PREBUILT, R6C_GATE_SUMMARY,
R6C_TEST_FAIL_AFTER_PATCH.
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = json.load(open(os.environ.get("R6C_KIT_JSON") or os.path.join(HERE, "KIT.json")))
V = os.environ.get("R6C_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
HOME = os.environ.get("R6C_HOME") or os.path.expanduser("~")
TAG = "# r6-comm"
R3TAG = "# r3-comm"
REL = "distributed/device_communicators/rdma_all_reduce.py"
RAR = f"{V}/{REL}"
STAGE = f"{HOME}/dsv41-stage/r6comm"
SO_SRC = os.environ.get("R6C_PREBUILT") or os.path.join(HERE, "prebuilt", "rdma_ar_ext.so")
SO_DST = f"{STAGE}/build/rdma_ar_ext.so"
LIVE_SO = f"{HOME}/dsv41-stage/g130/rdma/build/rdma_ar_ext.so"
GATE_SUMMARY = os.environ.get("R6C_GATE_SUMMARY") or os.path.join(HERE, "gate", "results", "SUMMARY.txt")

T = TAG
ON = '("1", "on", "true", "yes")'
EDITS = [  # (anchor, replacement, expected count); replacements only ADD tagged lines around the anchor
    (   # 1. header documentation
        "#    RDMA_AR_MAX_SLEEP_US, RDMA_AR_TIMING)\n",
        "#    RDMA_AR_MAX_SLEEP_US, RDMA_AR_TIMING)\n"
        f"#   DSV41_RDMA_TRIM            climb r6 C1 = r3 G06 (default 0): 1 = the r6 .so runs ar_kernel_r3 (early posted_seq  {T}\n"
        f"#                              read, seq advance at entry, single-CTA publish <= 24 KiB, single-CTA two-shot phase 2).  {T}\n"
        f"#                              Needs RDMA_AR_EXT = the r6 .so and RDMA_AR_PUB_V1/V2/HYG_POLL=1. Bitwise identical.  {T}\n"
        f"#   DSV41_RDMA_PDL             climb r6 C2 (default 0, needs TRIM): 1/in = ar_kernel_r3 launches with PDL (only  {T}\n"
        f"#                              ctrl->posted_seq is read before griddepcontrol.wait); trig = also triggers its  {T}\n"
        f"#                              dependents after the last flag wait. Scheduling only, bitwise identical.  {T}\n"
        f"#   DSV41_RDMA_SENDBUF         r3 G26 (default 0, needs TRIM): registered send-buffer region; an input that  {T}\n"
        f"#                              IS a send_buffer() is published without staging. BENCH ONLY: no producer uses it.  {T}\n",
        1,
    ),
    (   # 2. module-level flags, read once (the same parser as the .so's env_r3_flag / env_r6_pdl)
        "_EXT_PATH = None\n",
        "_EXT_PATH = None\n"
        f"_R3_ENV = {{k: os.environ.get(k, \"\").strip().lower() for k in (\"DSV41_RDMA_TRIM\", \"DSV41_RDMA_SENDBUF\")}}  {T}\n"
        f"_R3_BAD = sorted(k for k, v in _R3_ENV.items() if v not in (\"\", \"0\", \"off\", \"false\", \"no\", \"1\", \"on\", \"true\", \"yes\"))  {T}\n"
        f"if _R3_BAD:  {T}\n"
        f"    raise RuntimeError(f\"rdma_ar r6: {{_R3_BAD}} must be one of 0/1/off/on/false/true/no/yes\")  {T}\n"
        f"_R3_TRIM = _R3_ENV[\"DSV41_RDMA_TRIM\"] in {ON}  {T}\n"
        f"_R3_SENDBUF = _R3_ENV[\"DSV41_RDMA_SENDBUF\"] in {ON}  {T}\n"
        f"_R6_PDL_ENV = os.environ.get(\"DSV41_RDMA_PDL\", \"\").strip().lower()  {T}\n"
        f"if _R6_PDL_ENV not in (\"\", \"0\", \"off\", \"false\", \"no\", \"1\", \"on\", \"true\", \"yes\", \"in\", \"trig\"):  {T}\n"
        f"    raise RuntimeError(f\"rdma_ar r6: DSV41_RDMA_PDL={{_R6_PDL_ENV!r}} must be one of 0/1/off/on/false/true/no/yes/in/trig\")  {T}\n"
        f"_R6_PDL = _R6_PDL_ENV in (\"1\", \"on\", \"true\", \"yes\", \"in\", \"trig\")  {T}\n"
        f"if (_R6_PDL or _R3_SENDBUF) and not _R3_TRIM:  {T}\n"
        f"    raise RuntimeError(\"rdma_ar r6: DSV41_RDMA_PDL / DSV41_RDMA_SENDBUF need DSV41_RDMA_TRIM=1\")  {T}\n"
        f"_R3_ANY = _R3_TRIM or _R3_SENDBUF or _R6_PDL  {T}\n",
        1,
    ),
    (   # 3. the selected .so must carry the r3 (and, for PDL, the r6) kernels: else the live .so would ignore the flags
        "            ext = _load_ext()\n",
        "            ext = _load_ext()\n"
        f"            if _R3_ANY and getattr(ext, \"R3_VARIANT\", None) is None:  {T}\n"
        f"                raise RuntimeError(f\"DSV41_RDMA_TRIM/SENDBUF/PDL set but {{_EXT_PATH}} has no r3 kernels\")  {T}\n"
        f"            if _R6_PDL and getattr(ext, \"R6_VARIANT\", None) is None:  {T}\n"
        f"                raise RuntimeError(f\"DSV41_RDMA_PDL set but {{_EXT_PATH}} has no r6 (PDL) kernels\")  {T}\n",
        1,
    ),
    (   # 4. never fall back to NCCL silently with a flag set (the arm would measure something else)
        "        if bad:\n"
        "            logger.warning(\"rdma_ar disabled on all ranks; setup failed on %s\", bad)\n",
        "        if bad:\n"
        f"            if _R3_ANY:  {T}\n"
        f"                raise RuntimeError(f\"rdma_ar r6: setup failed on {{bad}}; refusing to start without rdma_ar\")  {T}\n"
        "            logger.warning(\"rdma_ar disabled on all ranks; setup failed on %s\", bad)\n",
        1,
    ),
    (   # 5. same for connect
        "        if bad:\n"
        "            logger.warning(\"rdma_ar disabled on all ranks; connect failed on %s\", bad)\n",
        "        if bad:\n"
        f"            if _R3_ANY:  {T}\n"
        f"                raise RuntimeError(f\"rdma_ar r6: connect failed on {{bad}}; refusing to start without rdma_ar\")  {T}\n"
        "            logger.warning(\"rdma_ar disabled on all ranks; connect failed on %s\", bad)\n",
        1,
    ),
    (   # 6. startup log of the r3 / r6 state (one line per rank; the window's boot count greps it)
        "            os.environ.get(\"DSV41_RDMA_PARLOAD\", \"0\"),\n"
        "        )\n",
        "            os.environ.get(\"DSV41_RDMA_PARLOAD\", \"0\"),\n"
        "        )\n"
        f"        if _R3_ANY:  {T}\n"
        f"            logger.info(\"rdma_ar r6 %s %s: r3 [mask, s1_bytes, sendbuf_bytes, used, buffers] = %s, pdl bits = %s\",  {T}\n"
        f"                        getattr(_EXT, \"R3_VARIANT\", None), getattr(_EXT, \"R6_VARIANT\", None), list(self.h.r3()),  {T}\n"
        f"                        self.h.r6() if hasattr(self.h, \"r6\") else None)  {T}\n",
        1,
    ),
    (   # 7. G26 API (bench only; see climb-r3 README "G26")
        "    def stats(self, reset: bool = False) -> list:\n",
        f"    def send_buffer(self, shape, dtype: torch.dtype):  {T}\n"
        f"        \"\"\"r3 G26: a tensor of `shape`/`dtype` in a registered send buffer (graph capture only, one buffer per  {T}\n"
        f"        call, never reused by another call), or None. A collective whose input IS that tensor skips staging.\"\"\"  {T}\n"
        f"        if not _R3_SENDBUF or self.disabled or not torch.cuda.is_current_stream_capturing():  {T}\n"
        f"            return None  {T}\n"
        f"        n = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()  {T}\n"
        f"        if n <= 0 or n % 16 or n > self.max_bytes:  {T}\n"
        f"            return None  {T}\n"
        f"        b = self.h.send_buffer(n)  {T}\n"
        f"        return None if b is None else b.view(dtype).view(shape)  {T}\n"
        f"  {T}\n"
        "    def stats(self, reset: bool = False) -> list:\n",
        1,
    ),
]
N_TAGGED = 44


def sha(p, n=64):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:n] if os.path.exists(p) else "-"


def entry(pred):
    for f in KIT["files"]:
        if pred(f["path"]):
            return f
    return {}


def base_shas():
    f = entry(lambda p: p == REL)
    return set(f.get("base_sha256_accepted") or [f["base_sha256"]])


def patched(src, path=RAR):
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


def write_atomic(path, text, mode_from):
    tmp = path + ".tmp-r6-comm"
    open(tmp, "w").write(text)
    shutil.copymode(mode_from, tmp)
    os.replace(tmp, path)


def copy_atomic(src, dst):
    tmp = dst + ".tmp-r6-comm"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def so_expected():
    return entry(lambda p: p.endswith("r6comm/build/rdma_ar_ext.so")).get("sha256")


def patched_sha():
    return entry(lambda p: p == REL).get("patched_sha256")


def gate_problem(want):
    """None when the gate summary proves D3-D6 passed on the full build of exactly `want`, else the reason"""
    try:
        lines = open(GATE_SUMMARY).read().splitlines()
    except OSError:
        return "missing"
    if not any(l.startswith("RUN_GATE PASS") for l in lines):
        return "no RUN_GATE PASS"
    if not any(l.split()[:2] == ["GATED_SO_SHA256", want] for l in lines if l.strip()):
        return f"no 'GATED_SO_SHA256 {want[:16]}..' line (D3-D6 not passed on this build)"
    d = {l.split()[0]: l.split()[1] for l in lines if l[:2] in ("D3", "D4", "D5", "D6") and len(l.split()) > 1}
    if sorted(d) != ["D3_live_vs_r6so", "D4_r3so_vs_r6so", "D5_pdl_r6so", "D6_devr6_vs_r6so"] or set(d.values()) != {"OK"}:
        return f"D3-D6 not all OK ({d})"
    return None


def install():
    if len(sys.argv) > 2:
        sys.exit(f"install takes no options (got {sys.argv[2:]}); the kit installs the prebuilt .so only")
    live_before = sha(LIVE_SO)
    # ---- 1. preflight: nothing is written until every check has passed
    src = open(RAR).read()
    if R3TAG in src:
        sys.exit(f"{RAR}: carries r3-comm lines; r6-comm supersedes r3-comm: uninstall r3-comm first; nothing written")
    if TAG in src:
        base = unpatched(src)
        if hashlib.sha256(base.encode()).hexdigest() not in base_shas():
            sys.exit(f"{RAR}: carries {TAG} lines but its base is not a recorded base sha256; refusing")
        if patched(base) != src:
            sys.exit(f"{RAR}: carries {TAG} lines but is not this kit's patch of its base ({sha(RAR, 16)}); refusing")
        new = None
    else:
        if sha(RAR) not in base_shas():
            sys.exit(f"{RAR}: sha256 {sha(RAR)} is not a recorded base ({sorted(base_shas())}); refusing")
        new = patched(src)
    base_txt, result_txt = (src, new) if new is not None else (unpatched(src), src)
    want_py, got_py = patched_sha(), hashlib.sha256(result_txt.encode()).hexdigest()
    if hashlib.sha256(base_txt.encode()).hexdigest() == entry(lambda p: p == REL).get("base_sha256") and got_py != want_py:
        sys.exit(f"{RAR}: patch result {got_py[:16]} != KIT.json patched_sha256 {str(want_py)[:16]}; refusing")
    cu_want = entry(lambda p: p.endswith("r6comm/native/rdma_ar.cu")).get("sha256")
    if not cu_want or sha(os.path.join(HERE, "src/rdma_ar.cu")) != cu_want:
        sys.exit(f"{HERE}/src/rdma_ar.cu: sha256 != KIT.json {cu_want}; refusing")
    want = so_expected()
    if not want:
        sys.exit("KIT.json records no sha256 for the r6 .so (not built yet: ops/build8-r6.sh r6 on spark-eight in a window "
                 "with the service stopped, then gate/run_gate.py D cases); nothing written")
    gate_bad = gate_problem(want)
    if gate_bad:
        sys.exit(f"{GATE_SUMMARY}: {gate_bad}; the .so {want[:16]} is not gated (run gate/run_gate.py on its full-build "
                 "SASS: D3-D6); nothing written")
    if os.path.exists(SO_SRC):
        if sha(SO_SRC) != want:
            sys.exit(f"{SO_SRC}: sha256 {sha(SO_SRC)} != KIT.json {want}; refusing, nothing written")
        so_from = SO_SRC
    elif os.path.exists(SO_DST) and sha(SO_DST) == want:
        so_from = None  # already staged by an earlier install
    else:
        sys.exit(f"no prebuilt .so at {SO_SRC} and {SO_DST} is "
                 f"{'absent' if not os.path.exists(SO_DST) else 'not ' + want[:16]}; nothing written")
    # ---- 2. stage dir (outside vllm; the launcher default never points here)
    os.makedirs(f"{STAGE}/native", exist_ok=True)
    os.makedirs(f"{STAGE}/build", exist_ok=True)
    for s_, d in (("src/rdma_ar.cu", "native/rdma_ar.cu"), ("src/mk_r6.py", "native/mk_r6.py"), ("build.sh", "build.sh")):
        copy_atomic(os.path.join(HERE, s_), f"{STAGE}/{d}")
    if so_from:
        copy_atomic(so_from, SO_DST)
    if sha(SO_DST) != want or sha(f"{STAGE}/native/rdma_ar.cu") != cu_want:
        sys.exit(f"{STAGE}: staged files do not match KIT.json after copy; rdma_all_reduce.py NOT patched")
    print(f"{SO_DST}: {sha(SO_DST, 16)} (== KIT.json)")
    # ---- 3. patch rdma_all_reduce.py last; roll back on any later failure
    bak = RAR + ".bak-r6-comm"
    wrote = False
    try:
        if new is None:
            print(f"{RAR}: already patched ({sha(RAR, 16)})")
        else:
            if not os.path.exists(bak):
                shutil.copy2(RAR, bak)
            if sha(bak) not in base_shas():
                sys.exit(f"{bak}: exists but is not a recorded base; refusing")
            wrote = True
            write_atomic(RAR, new, bak)
            print(f"{RAR}: patched ({sha(bak, 16)} -> {sha(RAR, 16)}), {new.count(TAG)} tagged lines")
        if sha(LIVE_SO) != live_before:
            sys.exit(f"BUG: {LIVE_SO} changed during install")
        if os.environ.get("R6C_TEST_FAIL_AFTER_PATCH"):  # tests only: prove the rollback
            raise RuntimeError("R6C_TEST_FAIL_AFTER_PATCH")
    except BaseException:
        if wrote:
            write_atomic(RAR, open(bak).read(), bak)
            os.remove(bak)
            print(f"{RAR}: install failed after patching; restored {sha(RAR, 16)}", file=sys.stderr)
        raise
    print(f"{LIVE_SO}: untouched ({sha(LIVE_SO, 16)})")


def uninstall():
    src = open(RAR).read()
    if TAG in src:
        clean = unpatched(src)
        h = hashlib.sha256(clean.encode()).hexdigest()
        if h not in base_shas():
            print(f"{RAR}: WARNING tag-stripped sha256 {h[:16]} is not a recorded base (another kit changed it?)")
        write_atomic(RAR, clean, RAR)
        print(f"{RAR}: {TAG} lines removed -> {sha(RAR, 16)}")
    else:
        print(f"{RAR}: not patched ({sha(RAR, 16)})")
    bak = RAR + ".bak-r6-comm"
    if os.path.exists(bak) and sha(bak) == sha(RAR):
        os.remove(bak)
    for d in ("native", "build"):
        if os.path.isdir(f"{STAGE}/{d}"):
            shutil.rmtree(f"{STAGE}/{d}")
    if os.path.exists(f"{STAGE}/build.sh"):
        os.remove(f"{STAGE}/build.sh")
    if os.path.isdir(STAGE) and not os.listdir(STAGE):
        os.rmdir(STAGE)
    print(f"{STAGE}: staged files removed" + (" (kit/ and its results kept)" if os.path.isdir(STAGE) else ""))


def status():
    src = open(RAR).read() if os.path.exists(RAR) else ""
    st = "PATCHED" if TAG in src else ("r3-comm!" if R3TAG in src else "clean  ")
    print(f"{sha(RAR, 16)} {st} {RAR}")
    want = so_expected()
    ok = os.path.exists(SO_DST) and bool(want) and sha(SO_DST) == want
    print(f"{sha(SO_DST, 16)} {'present' if ok else 'ABSENT/MISMATCH' if os.path.exists(SO_DST) else 'absent '} {SO_DST}")
    print(f"{sha(LIVE_SO, 16)} live .so (never touched) {LIVE_SO}")


def dry():
    src = open(RAR).read()
    if TAG in src:
        src = unpatched(src)
    h = hashlib.sha256(src.encode()).hexdigest()
    print(f"base sha256 {h[:16]} {'ACCEPTED' if h in base_shas() else 'NOT A RECORDED BASE'}")
    for i, line in enumerate(patched(src).splitlines(), 1):
        if TAG in line:
            print(f"{i:5d}: {line}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("install", "uninstall", "status", "dry"):
        sys.exit("usage: install.py install | uninstall | status | dry")
    {"install": install, "uninstall": uninstall, "status": status, "dry": dry}[sys.argv[1]]()
