#!/usr/bin/env python3
"""r7-comm2 installer (climb round 7): r6-comm (C1 DSV41_RDMA_TRIM, C2 DSV41_RDMA_PDL) + HT13 (DSV41_RDMA_PROJ: the
dsv41_oblit `_proj_rows` projection folded into the all-reduce's tail), one .so.

    python3 install.py install | uninstall | status | dry

* patches  vllm/distributed/device_communicators/rdma_all_reduce.py
             - base 26a5d834 (live; untouched by every r1..r6 kit but r6-comm): r6-comm's OWN edits are applied first
               (imported from climb-r6/comm/install.py, byte-identical result 67498e6b, tag "# r6-comm"), then this
               kit's (tag "# r7-comm2"). state.json records that r7 put the r6 lines in, so uninstall takes them out.
             - base 67498e6b (r6-comm already installed): only the r7 lines; uninstall leaves the r6 lines.
           vllm/models/deepseek_v41/nvidia/dsv41_oblit.py   (base 7824af1f = live, r1-pdl lines present; tag "# r7-comm2")
* stages   ~/dsv41-stage/r7comm2/{native/rdma_ar.cu, native/mk_r7.py, build.sh, build/rdma_ar_ext.so, state.json}
           build/rdma_ar_ext.so = prebuilt/rdma_ar_ext.so: the ONE binary ops/build8-r7.sh built, whose sha256 KIT.json
           records and for which gate/results/SUMMARY.txt carries RUN_GATE PASS and "GATED_SO_SHA256 <sha>". Never
           compiles, never touches the live .so ~/dsv41-stage/g130/rdma/build/rdma_ar_ext.so (14d049dd) or r6comm.
Every check runs before the first write; stage dir first; then dsv41_oblit.py, then rdma_all_reduce.py; any failure
after the first patch restores both files byte for byte.
Flags off (DSV41_RDMA_PROJ unset): every inserted line is flag parsing, a startup check / log behind the flag, or code
behind `if _R7_PROJ:` / `if _R7 is not None:` (dsv41_oblit imports nothing new). An arm sets
RDMA_AR_EXT=$HOME/dsv41-stage/r7comm2/build/rdma_ar_ext.so DSV41_RDMA_TRIM=1 [DSV41_RDMA_PDL=1] DSV41_RDMA_PROJ=1.
Env (tests): R7C_VLLM, R7C_HOME, R7C_KIT_JSON, R7C_PREBUILT, R7C_GATE_SUMMARY, R7C_R6_INSTALL, R7C_TEST_FAIL_AFTER_PATCH.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = json.load(open(os.environ.get("R7C_KIT_JSON") or os.path.join(HERE, "KIT.json")))
V = os.environ.get("R7C_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
HOME = os.environ.get("R7C_HOME") or os.path.expanduser("~")
TAG = "# r7-comm2"
R6TAG = "# r6-comm"
R3TAG = "# r3-comm"
REL_RAR = "distributed/device_communicators/rdma_all_reduce.py"
REL_OBL = "models/deepseek_v41/nvidia/dsv41_oblit.py"
RAR, OBL = f"{V}/{REL_RAR}", f"{V}/{REL_OBL}"
STAGE = f"{HOME}/dsv41-stage/r7comm2"
SO_SRC = os.environ.get("R7C_PREBUILT") or os.path.join(HERE, "prebuilt", "rdma_ar_ext.so")
SO_DST = f"{STAGE}/build/rdma_ar_ext.so"
LIVE_SO = f"{HOME}/dsv41-stage/g130/rdma/build/rdma_ar_ext.so"
GATE_SUMMARY = os.environ.get("R7C_GATE_SUMMARY") or os.path.join(HERE, "gate", "results", "SUMMARY.txt")
R6_INSTALL = os.environ.get("R7C_R6_INSTALL") or os.path.join(HERE, "r6", "install.py")
STATE = f"{STAGE}/state.json"

T = TAG
ON = '("1", "on", "true", "yes")'
EDITS_RAR = [
    (   # 1. header documentation (after r6-comm's block)
        "#                              IS a send_buffer() is published without staging. BENCH ONLY: no producer uses it.  # r6-comm\n",
        "#                              IS a send_buffer() is published without staging. BENCH ONLY: no producer uses it.  # r6-comm\n"
        f"#   DSV41_RDMA_PROJ            climb r7 HT13 (default 0, needs TRIM and the r7 .so): an all-reduce of a bf16  {T}\n"
        f"#                              [rows <= 6, 5120] tensor while dsv41_oblit has armed a direction d also writes  {T}\n"
        f"#                              y = _proj_rows(out, d) from the same kernel; dsv41_oblit.project(out, d) then  {T}\n"
        f"#                              returns y instead of launching _proj_rows. Same op order (gate/projgate.py).  {T}\n",
        1,
    ),
    (   # 2. module-level flag and the arm / take state (host-side only; graph-capture safe: decided at capture time)
        "_R3_ANY = _R3_TRIM or _R3_SENDBUF or _R6_PDL  # r6-comm\n",
        "_R3_ANY = _R3_TRIM or _R3_SENDBUF or _R6_PDL  # r6-comm\n"
        f"_R7_PROJ_ENV = os.environ.get(\"DSV41_RDMA_PROJ\", \"\").strip().lower()  {T}\n"
        f"if _R7_PROJ_ENV not in (\"\", \"0\", \"off\", \"false\", \"no\", \"1\", \"on\", \"true\", \"yes\"):  {T}\n"
        f"    raise RuntimeError(f\"rdma_ar r7: DSV41_RDMA_PROJ={{_R7_PROJ_ENV!r}} must be one of 0/1/off/on/false/true/no/yes\")  {T}\n"
        f"_R7_PROJ = _R7_PROJ_ENV in {ON}  {T}\n"
        f"if _R7_PROJ and not _R3_TRIM:  {T}\n"
        f"    raise RuntimeError(\"rdma_ar r7: DSV41_RDMA_PROJ needs DSV41_RDMA_TRIM=1\")  {T}\n"
        f"_R7_MAX_ROWS = 6  {T}\n"
        f"_R7 = {{\"d\": None, \"tag\": None, \"folded\": 0, \"taken\": 0, \"seg_folded\": 0, \"seg_taken\": 0, \"seg_rows\": set(),  {T}\n"
        f"       \"seg_first\": None, \"eager_logs\": 0}}  {T}\n"
        f"  {T}\n"
        f"  {T}\n"
        f"def r7_arm(d, rows=-1):  {T}\n"
        f"    \"\"\"dsv41_oblit: the next all-reduces may fold the projection by direction d (an fp32 [5120] device tensor);  {T}\n"
        f"    rows = token rows of the projected tensor (the first arm of a forward names the forward's rows in the log)\"\"\"  {T}\n"
        f"    if _R7_PROJ:  {T}\n"
        f"        _R7[\"d\"] = d  {T}\n"
        f"        if _R7[\"seg_first\"] is None:  {T}\n"
        f"            _R7[\"seg_first\"] = rows  {T}\n"
        f"  {T}\n"
        f"  {T}\n"
        f"def r7_disarm():  {T}\n"
        f"    \"\"\"dsv41_oblit.project_head: end of the bound layers. Logs this forward's fold counts (review: the promote  {T}\n"
        f"    precondition is taken == 76 in every captured decode forward on every rank; folded - taken = wasted tails)\"\"\"  {T}\n"
        f"    _R7[\"d\"] = None  {T}\n"
        f"    _R7[\"tag\"] = None  {T}\n"
        f"    if not _R7_PROJ:  {T}\n"
        f"        return  {T}\n"
        f"    f, t, rows, fr = _R7[\"seg_folded\"], _R7[\"seg_taken\"], sorted(_R7[\"seg_rows\"]), _R7[\"seg_first\"]  {T}\n"
        f"    _R7[\"seg_folded\"], _R7[\"seg_taken\"], _R7[\"seg_rows\"], _R7[\"seg_first\"] = 0, 0, set(), None  {T}\n"
        f"    try:  {T}\n"
        f"        cap = bool(torch.cuda.is_current_stream_capturing())  {T}\n"
        f"    except Exception:  # noqa: BLE001  (CPU-only builds)  {T}\n"
        f"        cap = False  {T}\n"
        f"    if cap or (f and _R7[\"eager_logs\"] < 8):  {T}\n"
        f"        _R7[\"eager_logs\"] += 0 if cap else 1  {T}\n"
        f"        logger.info(\"rdma_ar-r7 fwd: capturing=%d fwd_rows=%s ar_rows=%s folded=%d taken=%d wasted=%d \"  {T}\n"
        f"                    \"(totals folded=%d taken=%d)\", int(cap), fr, rows, f, t, f - t, _R7[\"folded\"], _R7[\"taken\"])  {T}\n"
        f"  {T}\n"
        f"  {T}\n"
        f"def r7_take(x, d):  {T}\n"
        f"    \"\"\"y when x IS the output of the last all-reduce (same storage, size, unchanged version counter) and that  {T}\n"
        f"    all-reduce folded direction d (the same tensor object); else None. One use: the tag is consumed. The tag  {T}\n"
        f"    holds the output tensor itself, so its memory cannot be reused by another tensor while the tag lives.\"\"\"  {T}\n"
        f"    t = _R7[\"tag\"]  {T}\n"
        f"    _R7[\"tag\"] = None  {T}\n"
        f"    if t is None or not _R7_PROJ:  {T}\n"
        f"        return None  {T}\n"
        f"    out, ver, y, dd = t  {T}\n"
        f"    if (dd is not d or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() != out.data_ptr() or  {T}\n"
        f"            x.numel() != out.numel() or x.is_inference() or out.is_inference() or  {T}\n"
        f"            x._version != ver or out._version != ver):  {T}\n"
        f"        return None  {T}\n"
        f"    _R7[\"taken\"] += 1  {T}\n"
        f"    _R7[\"seg_taken\"] += 1  {T}\n"
        f"    return y  {T}\n",
        1,
    ),
    (   # 3. the selected .so must carry the projection kernels
        "                raise RuntimeError(f\"DSV41_RDMA_PDL set but {_EXT_PATH} has no r6 (PDL) kernels\")  # r6-comm\n",
        "                raise RuntimeError(f\"DSV41_RDMA_PDL set but {_EXT_PATH} has no r6 (PDL) kernels\")  # r6-comm\n"
        f"            if _R7_PROJ and getattr(ext, \"R7_VARIANT\", None) is None:  {T}\n"
        f"                raise RuntimeError(f\"DSV41_RDMA_PROJ set but {{_EXT_PATH}} has no r7 projection kernels\")  {T}\n",
        1,
    ),
    (   # 4. startup log (the window's boot count greps "rdma_ar r7")
        "                        self.h.r6() if hasattr(self.h, \"r6\") else None)  # r6-comm\n",
        "                        self.h.r6() if hasattr(self.h, \"r6\") else None)  # r6-comm\n"
        f"        if _R7_PROJ:  {T}\n"
        f"            logger.info(\"rdma_ar r7 %s: projection fold on (bf16 [rows <= %d, 5120], DSV41_RDMA_PROJ), so r7 %s\",  {T}\n"
        f"                        getattr(_EXT, \"R7_VARIANT\", None), _R7_MAX_ROWS, self.h.r7() if hasattr(self.h, \"r7\") else None)  {T}\n",
        1,
    ),
    (   # 5. the fold: after the before-collective hook, before the plain call; returns early when it folded
        "        if _L2PF is not None:\n            _L2PF.before_collective(\"ar\")\n        out = self.h.all_reduce(x, two_shot)\n",
        "        if _L2PF is not None:\n            _L2PF.before_collective(\"ar\")\n"
        f"        if _R7_PROJ:  {T}\n"
        f"            _r7 = self._r7_fold(x, two_shot)  {T}\n"
        f"            if _r7 is not None:  {T}\n"
        f"                if _L2PF_AFTER is not None:  {T}\n"
        f"                    _L2PF_AFTER(\"ar\")  {T}\n"
        f"                return _r7  {T}\n"
        f"            _R7[\"tag\"] = None  {T}\n"
        "        out = self.h.all_reduce(x, two_shot)\n",
        1,
    ),
    (   # 6. the fold method (before all_reduce)
        "    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:\n",
        f"    def _r7_fold(self, x: torch.Tensor, two_shot: bool):  {T}\n"
        f"        d = _R7[\"d\"]  {T}\n"
        f"        if (d is None or x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[-1] != 5120 or  {T}\n"
        f"                not (1 <= x.shape[0] <= _R7_MAX_ROWS) or not x.is_contiguous()):  {T}\n"
        f"            return None  {T}\n"
        f"        # Served under torch.inference_mode, whose tensors carry no version counter and may not be written  {T}\n"
        f"        # in place outside it: allocate y and the op's output (torch::empty_like) as normal tensors, so  {T}\n"
        f"        # r7_take's in-place guard works and no inference tensor is written. x is only read. Same values.  {T}\n"
        f"        with torch.inference_mode(False):  {T}\n"
        f"            y = torch.empty_like(x)  {T}\n"
        f"            out = self.h.all_reduce_proj(x, two_shot, d, y)  {T}\n"
        f"        _R7[\"tag\"] = None if out.is_inference() else (out, out._version, y, d)  {T}\n"
        f"        _R7[\"folded\"] += 1  {T}\n"
        f"        _R7[\"seg_folded\"] += 1  {T}\n"
        f"        _R7[\"seg_rows\"].add(int(x.shape[0]))  {T}\n"
        f"        return out  {T}\n"
        f"  {T}\n"
        "    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:\n",
        1,
    ),
]
EDITS_OBL = [
    (   # a. the rdma module, only with the flag (nothing imported otherwise). Fail closed (review B1 / item 3) when
        #    the flag meets a state whose served projection is not the proven one: DSV41_GLUE_PDL with its 'proj'
        #    family (the served op becomes dsv41_glue_pdl.proj_rows, gdc_wait + launch_pdl, a separately compiled
        #    kernel whose SASS record nobody compared with f29c9ff9), or r2-oblit's DSV41_OBLIT_FOLD (only project_moe
        #    runs per layer, so the previous layer's d stays armed and no take could ever match: zero gain, wasted tails)
        "ON = bool(PATH)\n",
        "ON = bool(PATH)\n"
        f"_R7 = None  {T}\n"
        f"_R7_ON = os.environ.get(\"DSV41_RDMA_PROJ\", \"\").strip().lower() in {ON}  {T}\n"
        f"if _R7_ON and _DSV41_GLUE_PDL:  {T}\n"
        f"    raise RuntimeError(\"DSV41_RDMA_PROJ=1 refused: DSV41_GLUE_PDL enables its 'proj' family, so the served projection \"  {T}\n"
        f"                       \"would be dsv41_glue_pdl.proj_rows (PDL), whose SASS record is unproven against f29c9ff9 \"  {T}\n"
        f"                       \"(r7-comm2 gate/projgate.py); unset one of them\")  {T}\n"
        f"if _R7_ON and os.environ.get(\"DSV41_OBLIT_FOLD\", \"0\").strip().lower() not in (\"\", \"0\", \"off\", \"false\", \"no\"):  {T}\n"
        f"    raise RuntimeError(\"DSV41_RDMA_PROJ=1 refused with DSV41_OBLIT_FOLD (r2-oblit): the folded layers never arm \"  {T}\n"
        f"                       \"their own direction, so no fold would be taken; unset one of them\")  {T}\n"
        f"if _R7_ON:  {T}\n"
        f"    from vllm.distributed.device_communicators import rdma_all_reduce as _R7  {T}\n",
        1,
    ),
    (   # b. take the folded y when x is the last all-reduce's output for this direction; then arm this direction
        "    H = x.shape[-1]\n    if x.numel() == 0:\n        return x\n",
        "    H = x.shape[-1]\n    if x.numel() == 0:\n        return x\n"
        f"    if _R7 is not None:  {T}\n"
        f"        y7 = _R7.r7_take(x, d)  {T}\n"
        f"        _R7.r7_arm(d, x.shape[0] if x.dim() == 2 else -1)  {T}\n"
        f"        if y7 is not None:  {T}\n"
        f"            return y7.view(x.shape)  {T}\n",
        1,
    ),
    (   # c. the head projection ends the layers: disarm (the drafter's collectives never fold)
        "    return _project(h, _vec(\"head\", h))\n",
        f"    if _R7 is not None:  {T}\n"
        f"        y = _project(h, _vec(\"head\", h))  {T}\n"
        f"        _R7.r7_disarm()  {T}\n"
        f"        return y  {T}\n"
        "    return _project(h, _vec(\"head\", h))\n",
        1,
    ),
]
N_TAGGED = {REL_RAR: None, REL_OBL: None}  # filled from the edits


def _count(edits):
    return sum(new.count(TAG) - old.count(TAG) for old, new, _ in edits)


N_TAGGED[REL_RAR] = _count(EDITS_RAR)
N_TAGGED[REL_OBL] = _count(EDITS_OBL)


def sha(p, n=64):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:n] if os.path.exists(p) else "-"


def shat(text):
    return hashlib.sha256(text.encode()).hexdigest()


def entry(rel):
    for f in KIT["files"]:
        if f["path"] == rel:
            return f
    return {}


def apply(src, edits, path, rel):
    for old, new, count in edits:
        n = src.count(old)
        if n != count:
            sys.exit(f"{path}: anchor found {n}x, expected {count}: {old[:90]!r}; nothing written")
        src = src.replace(old, new)
    if src.count(TAG) != N_TAGGED[rel]:
        sys.exit(f"{path}: {src.count(TAG)} tagged lines after patch, expected {N_TAGGED[rel]}")
    return src


def strip(src, tag=TAG):
    return "".join(line for line in src.splitlines(keepends=True) if tag not in line)


def r6_module():
    spec = importlib.util.spec_from_file_location("r6comm_install", R6_INSTALL)
    m = importlib.util.module_from_spec(spec)
    old = os.environ.get("R6C_KIT_JSON")
    os.environ["R6C_KIT_JSON"] = os.path.join(os.path.dirname(R6_INSTALL), "KIT.json")
    try:
        spec.loader.exec_module(m)
    finally:
        if old is None:
            os.environ.pop("R6C_KIT_JSON", None)
        else:
            os.environ["R6C_KIT_JSON"] = old
    return m


def plan_rar(src):
    """(new text or None if already installed, r6 lines added by us?)"""
    e = entry(REL_RAR)
    live, r6p = e["base_sha256_live"], e["base_sha256_r6"]
    if R3TAG in src:
        sys.exit(f"{RAR}: carries r3-comm lines; nothing written")
    if TAG in src:
        base = strip(src)
        if shat(base) == r6p:
            new = apply(base, EDITS_RAR, RAR, REL_RAR)
            if new == src:
                return None, None
            if shat(src) in e.get("prior_patched_sha256", []):
                # in-place upgrade of this kit's earlier patch (HT13 inference fix): same r6 base, the .so and
                # state.json stay as they are (None = do not rewrite state.json)
                return new, None
        sys.exit(f"{RAR}: carries {TAG} lines but is not this kit's patch of r6 {r6p[:8]}; nothing written")
    h = shat(src)
    if h == r6p:
        return apply(src, EDITS_RAR, RAR, REL_RAR), False
    if h == live:
        r6 = r6_module().patched(src, RAR)
        if shat(r6) != r6p:
            sys.exit(f"{RAR}: r6-comm's edits gave {shat(r6)[:16]}, not {r6p[:16]}; nothing written")
        return apply(r6, EDITS_RAR, RAR, REL_RAR), True
    sys.exit(f"{RAR}: sha256 {h[:16]} is neither the live base {live[:16]} nor r6-comm's {r6p[:16]}; nothing written")


def plan_obl(src):
    e = entry(REL_OBL)
    if TAG in src:
        base = strip(src)
        if shat(base) not in e["base_sha256_accepted"] or apply(base, EDITS_OBL, OBL, REL_OBL) != src:
            sys.exit(f"{OBL}: carries {TAG} lines but is not this kit's patch; nothing written")
        return None
    if shat(src) not in e["base_sha256_accepted"]:
        sys.exit(f"{OBL}: sha256 {shat(src)[:16]} is not a recorded base; nothing written")
    return apply(src, EDITS_OBL, OBL, REL_OBL)


def write_atomic(path, text, mode_from):
    tmp = path + ".tmp-r7-comm2"
    open(tmp, "w").write(text)
    shutil.copymode(mode_from, tmp)
    os.replace(tmp, path)


def copy_atomic(src, dst):
    tmp = dst + ".tmp-r7-comm2"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def so_expected():
    return entry("$HOME/dsv41-stage/r7comm2/build/rdma_ar_ext.so").get("sha256")


def gate_problem(want):
    try:
        lines = open(GATE_SUMMARY).read().splitlines()
    except OSError:
        return "missing"
    if not any(l.startswith("RUN_GATE7 PASS") for l in lines):
        return "no RUN_GATE7 PASS"
    if not any(l.split()[:2] == ["GATED_SO_SHA256", want] for l in lines if l.strip()):
        return f"no 'GATED_SO_SHA256 {want[:16]}..' line (D cases not passed on this build)"
    d = {l.split()[0]: l.split()[1] for l in lines if l.startswith("D7_") and len(l.split()) > 1}
    need = {"D7_live_vs_r7so", "D7_pdl_r7so", "D7_proj_r7so", "D7_ord_r7so", "D7_devr7_vs_r7so"}
    if set(d) != need or set(d.values()) != {"OK"}:
        return f"D7 cases not all OK ({d})"
    return None


def install():
    if len(sys.argv) > 2:
        sys.exit(f"install takes no options (got {sys.argv[2:]})")
    live_before = sha(LIVE_SO)
    rar_src, obl_src = open(RAR).read(), open(OBL).read()
    rar_new, r6_added = plan_rar(rar_src)
    obl_new = plan_obl(obl_src)
    for rel, txt, cur in ((REL_RAR, rar_new, rar_src), (REL_OBL, obl_new, obl_src)):
        res = txt if txt is not None else cur
        want = entry(rel).get("patched_sha256")
        if shat(res) != want:
            sys.exit(f"{rel}: result {shat(res)[:16]} != KIT.json patched_sha256 {str(want)[:16]}; nothing written")
    cu_want = entry("$HOME/dsv41-stage/r7comm2/native/rdma_ar.cu").get("sha256")
    if not cu_want or sha(os.path.join(HERE, "src/rdma_ar.cu")) != cu_want:
        sys.exit(f"{HERE}/src/rdma_ar.cu: sha256 != KIT.json {cu_want}; nothing written")
    want = so_expected()
    if not want:
        sys.exit("KIT.json records no sha256 for the r7 .so (not built yet: ops/build8-r7.sh in a stopped window, then "
                 "gate/run_gate7.py D7 cases); nothing written")
    bad = gate_problem(want)
    if bad:
        sys.exit(f"{GATE_SUMMARY}: {bad}; the .so {want[:16]} is not gated; nothing written")
    if os.path.exists(SO_SRC):
        if sha(SO_SRC) != want:
            sys.exit(f"{SO_SRC}: sha256 != KIT.json {want[:16]}; nothing written")
        so_from = SO_SRC
    elif os.path.exists(SO_DST) and sha(SO_DST) == want:
        so_from = None
    else:
        sys.exit(f"no prebuilt .so at {SO_SRC} and {SO_DST} is not {want[:16]}; nothing written")
    # ---- stage dir
    os.makedirs(f"{STAGE}/native", exist_ok=True)
    os.makedirs(f"{STAGE}/build", exist_ok=True)
    for s_, d in (("src/rdma_ar.cu", "native/rdma_ar.cu"), ("src/mk_r7.py", "native/mk_r7.py"), ("build.sh", "build.sh")):
        copy_atomic(os.path.join(HERE, s_), f"{STAGE}/{d}")
    if so_from:
        copy_atomic(so_from, SO_DST)
    if sha(SO_DST) != want or sha(f"{STAGE}/native/rdma_ar.cu") != cu_want:
        sys.exit(f"{STAGE}: staged files do not match KIT.json after copy; no vllm file patched")
    if r6_added is not None:
        st = {"r6_lines_added_by_r7": bool(r6_added)}
        open(STATE + ".tmp", "w").write(json.dumps(st))
        os.replace(STATE + ".tmp", STATE)
    # ---- patch dsv41_oblit.py, then rdma_all_reduce.py; restore both on any later failure
    wrote = []
    try:
        for path, new in ((OBL, obl_new), (RAR, rar_new)):
            if new is None:
                print(f"{path}: already patched ({sha(path, 16)})")
                continue
            bak = path + ".bak-r7-comm2"
            made = not os.path.exists(bak)
            if made:
                shutil.copy2(path, bak)
            # restore target = the bytes on disk now (the base on a fresh install, the prior patch on an upgrade);
            # the .bak keeps the pre-r7 base for uninstall and is removed on rollback only if this run made it
            wrote.append((path, bak, open(path).read(), made))
            before = sha(path, 16)
            write_atomic(path, new, bak)
            print(f"{path}: patched ({before} -> {sha(path, 16)}), {new.count(TAG)} {TAG} lines")
        if sha(LIVE_SO) != live_before:
            sys.exit(f"BUG: {LIVE_SO} changed during install")
        if os.environ.get("R7C_TEST_FAIL_AFTER_PATCH"):
            raise RuntimeError("R7C_TEST_FAIL_AFTER_PATCH")
    except BaseException:
        for path, bak, prev, made in reversed(wrote):
            write_atomic(path, prev, bak)
            if made:
                os.remove(bak)
            print(f"{path}: install failed; restored {sha(path, 16)}", file=sys.stderr)
        raise
    print(f"{LIVE_SO}: untouched ({sha(LIVE_SO, 16)})")


def uninstall():
    added = False
    try:
        added = json.load(open(STATE)).get("r6_lines_added_by_r7", False)
    except OSError:
        pass
    for path, rel in ((RAR, REL_RAR), (OBL, REL_OBL)):
        src = open(path).read()
        if TAG in src:
            clean = strip(src)
            if path == RAR and added:
                clean = strip(clean, R6TAG)
            write_atomic(path, clean, path)
            print(f"{path}: {TAG}{' and ' + R6TAG if path == RAR and added else ''} lines removed -> {sha(path, 16)}")
        else:
            print(f"{path}: not patched ({sha(path, 16)})")
        bak = path + ".bak-r7-comm2"
        if os.path.exists(bak) and sha(bak) == sha(path):
            os.remove(bak)
        e = entry(rel)
        okb = {e.get("base_sha256_live"), e.get("base_sha256_r6")} | set(e.get("base_sha256_accepted", []))
        if sha(path) not in okb:
            print(f"{path}: WARNING result {sha(path, 16)} is not a recorded base (another kit changed it?)")
    for d in ("native", "build"):
        if os.path.isdir(f"{STAGE}/{d}"):
            shutil.rmtree(f"{STAGE}/{d}")
    for f in ("build.sh", "state.json"):
        if os.path.exists(f"{STAGE}/{f}"):
            os.remove(f"{STAGE}/{f}")
    if os.path.isdir(STAGE) and not os.listdir(STAGE):
        os.rmdir(STAGE)
    print(f"{STAGE}: staged files removed")


def status():
    """per file: PATCHED / clean (+r6 when r6-comm's lines are present). HYBRID (review item 5): rdma_all_reduce.py
    carries this kit's lines but not r6-comm's, which is what r6-comm's own `uninstall` leaves on a node where r7
    installed on top of the r6 lines. It imports with the flags off, and DSV41_RDMA_PROJ=1 then fails at import
    (NameError, fail-closed), but it is no recorded state: run this kit's uninstall, then reinstall what is wanted.
    Exit status 4 for HYBRID / not-a-recorded-state, else 0."""
    rc = 0
    added = None
    try:
        added = json.load(open(STATE)).get("r6_lines_added_by_r7")
    except OSError:
        pass
    for path, rel in ((RAR, REL_RAR), (OBL, REL_OBL)):
        src = open(path).read() if os.path.exists(path) else ""
        st = ("PATCHED" if TAG in src else "clean  ") + (" +r6" if R6TAG in src else "")
        e = entry(rel)
        known = ({e.get("base_sha256_live"), e.get("base_sha256_r6"), e.get("patched_sha256")} |
                 set(e.get("base_sha256_accepted", [])) | set(e.get("prior_patched_sha256", [])))
        if src and shat(src) in e.get("prior_patched_sha256", []):
            st += " PRIOR-PATCH (install upgrades it in place)"
        if path == RAR and TAG in src and R6TAG not in src:
            st += " HYBRID (r7 lines without r6-comm's: r6-comm's uninstall ran under r7; run r7 uninstall first)"
            rc = 4
        elif src and shat(src) not in known:
            st += " NOT-A-RECORDED-STATE"
            rc = 4
        if path == RAR and added is not None:
            st += f" (state.json: r6 lines added by r7 = {added})"
        print(f"{sha(path, 16)} {st} {path}")
    want = so_expected()
    ok = os.path.exists(SO_DST) and bool(want) and sha(SO_DST) == want
    print(f"{sha(SO_DST, 16)} {'present' if ok else 'ABSENT/MISMATCH'} {SO_DST}")
    print(f"{sha(LIVE_SO, 16)} live .so (never touched) {LIVE_SO}")
    if rc:
        sys.exit(rc)


def downgrade():
    """undo an in-place upgrade: rdma_all_reduce.py at this kit's patch -> the recorded prior patch (byte-exact copy
    of prior_patched_ref). Touches nothing else: the .so, dsv41_oblit.py and state.json stay. Used as the revert of
    an upgrade window, where `uninstall` would delete the .so that the production launcher loads."""
    if len(sys.argv) > 2:
        sys.exit(f"downgrade takes no options (got {sys.argv[2:]})")
    e = entry(REL_RAR)
    prior = e.get("prior_patched_sha256", [])
    ref = os.path.join(HERE, e.get("prior_patched_ref", ""))
    if not prior or not os.path.isfile(ref) or sha(ref) != prior[0]:
        sys.exit(f"{ref}: missing or != prior_patched_sha256; nothing written")
    src = open(RAR).read()
    if shat(src) == prior[0]:
        print(f"{RAR}: already the prior patch ({sha(RAR, 16)})")
        return
    if shat(src) != e["patched_sha256"]:
        sys.exit(f"{RAR}: sha256 {shat(src)[:16]} is not this kit's patch {e['patched_sha256'][:16]}; nothing written")
    bak = RAR + ".bak-r7-comm2"
    write_atomic(RAR, open(ref).read(), bak if os.path.exists(bak) else RAR)
    if sha(RAR) != prior[0]:
        sys.exit(f"BUG: {RAR} is {sha(RAR, 16)} after downgrade, not {prior[0][:16]}")
    print(f"{RAR}: downgraded to the prior patch ({sha(RAR, 16)})")


def dry():
    rar_new, r6_added = plan_rar(open(RAR).read())
    obl_new = plan_obl(open(OBL).read())
    for path, new in ((RAR, rar_new), (OBL, obl_new)):
        if new is None:
            print(f"{path}: already patched")
            continue
        print(f"{path}: result {shat(new)[:16]}" + (" (r6-comm lines added too)" if path == RAR and r6_added else ""))
        for i, line in enumerate(new.splitlines(), 1):
            if TAG in line:
                print(f"{i:5d}: {line}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("install", "uninstall", "status", "dry", "downgrade"):
        sys.exit("usage: install.py install | uninstall | status | dry | downgrade")
    {"install": install, "uninstall": uninstall, "status": status, "dry": dry, "downgrade": downgrade}[sys.argv[1]]()
