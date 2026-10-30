#!/usr/bin/env python3
"""g2 fc2c-on-m32all CPU test (no CUDA): dispatch of the staged sm121_target_m32.py, provenance of every
function it shares with the two qualified site files, the tree pin, and the tail byte mapping.

  CUDA_VISIBLE_DEVICES= ~/venvs/vllm032/bin/python -I test_g2_cpu.py <staged> <live db659d80> <g1 fc2c 9e250455> [stage]

Checks:
  * DSV41_MOE_FC2C unset: the staged module issues exactly the live module's (db659d80) native calls, D05 0/1/verify,
    with and without g1-la (the 'all' A-layout guard).
  * provenance: every function/class the staged file shares with db659d80 (g1 loaders, D05 dispatch helpers) or with
    the g1 fc2c-d05 file 9e250455 (fc2c_mode, fc2c_min_rows, tail builder, FC2Compact, verifier) is source-identical;
    load_fc2c_native differs from 9e250455's only in the tree selection; TargetM32Kernel.gemm is 9e250455's plus the
    db659d80 g1-la guard plus the verify-mode served-buffer poison.
  * mode parsing; fc2c_min_rows(t) == M_sum(t) at alignment 32, t = 1..120; the gate is monotonic (M >= 2).
  * FC2C=1 routing (fake natives): compact native with dsv41_fc2_live_store=True and the weight's descriptor table
    for M_sum >= min rows; M=1 -> D05 native; FC1 / other shapes -> base native; no tail during capture -> D05.
  * FC2C=verify: D05 served first, then the compact shadow; with poison_served the served buffer is filled with the
    verifier's POISON before the D05 call (and only then).
  * tree selection: G1=all -> g2 tree + its JIT cache + pin; G1 off -> g1 fc2c-d05 tree; pair/pf/la/ds refused;
    env overrides honoured; G1=all without D05=1 refused.
  * [stage given] the staged pin equals tree/manifest.json, the whole tree verifies, and tree/ header ==
    make_fc2c_tree.fc2c_on_all(live all-d05 header) (the tree is reproducible from the live trees).
  * tail bytes (torch CPU) == the padded kernel's last-k-block shared-memory data bytes; non-zero K padding rejected.
"""
import difflib
import hashlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import torch

staged_path, live_path, g1_path = sys.argv[1:4]
stage = Path(sys.argv[4]) if len(sys.argv) > 4 else None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                  # inspect.getsource(class) resolves the file through sys.modules
    spec.loader.exec_module(mod)
    return mod


for k in ("DSV41_MOE_FC2C", "DSV41_MOE_FC2C_MIN_TOKENS", "DSV41_MOE_FC2_D05", "DSV41_MOE_M32_G1",
          "DSV41_MOE_PDL_EARLY_B", "DSV41_MOE_FC2C_NATIVE_DIR", "DSV41_MOE_FC2C_JIT_DIR"):
    os.environ.pop(k, None)
assert sha(live_path) == "db659d809cbcc3949f94dd810d9f00359f1770af082aadd8f7037816371f02da", "live is not db659d80"
assert sha(g1_path) == "9e2504551c6a42f28bc4c66aca1f9c8ad085eb5888b4c6c7178bca6584fe9dc7", "g1 file is not 9e250455"
staged = load("staged_m32", staged_path)
live = load("live_m32", live_path)
g1 = load("g1_m32", g1_path)
fails = 0


def check(cond, what):
    global fails
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    fails += not cond


# ---- 0. provenance (source identity)
def src(mod, qual):
    obj = mod
    for part in qual.split("."):
        obj = getattr(obj, part)
    return inspect.getsource(obj)


for qual in ("g1_variant", "pdl_early_b", "d05_mode", "d05_eligible", "_verify_tree", "load_d05_native",
             "TargetM32Kernel.alignment_scope", "_D05Verifier.check"):
    check(src(staged, qual) == src(live, qual), f"provenance: {qual} == db659d80")
for qual in ("fc2c_mode", "fc2c_min_rows", "make_fc2c_tail", "fc2c_tail_from_bytes", "FC2Compact",
             "_D05Verifier.__init__", "_D05Verifier._start_reporter", "_D05Verifier.check_compact"):
    check(src(staged, qual) == src(g1, qual), f"provenance: {qual} == 9e250455")
for name in ("_MANIFEST_SHA256", "_NATIVE_SHA256", "_D05_MANIFEST_SHA256", "_D05_NATIVE_SHA256", "_G1_MANIFESTS",
             "_G1_VARIANTS", "_G1_HAS_D05", "_G1_HAS_LA", "_PDL_BASE_MANIFEST_SHA256", "_PDL_D05_MANIFEST_SHA256",
             "_D05_MAX_ROWS", "_FC2_E", "_FC2_N", "_FC2_KP", "_FC2_KB"):
    check(getattr(staged, name) == getattr(live, name), f"constant {name} == db659d80")
for name in ("_FC2C_MANIFEST_SHA256", "_FC2C_NATIVE_SHA256", "_FC2C_TOPK", "_FC2C_ALIGN", "_FC2C_KREAL_B"):
    check(getattr(staged, name) == getattr(g1, name), f"constant {name} == 9e250455")


def line_delta(a_src, b_src):
    d = list(difflib.ndiff(a_src.splitlines(), b_src.splitlines()))
    return [x for x in d if x[:1] in "+-"]


delta = line_delta(src(g1, "load_fc2c_native"), src(staged, "load_fc2c_native"))
removed = sorted(x[2:].strip() for x in delta if x.startswith("- "))
added = sorted(x[2:].strip() for x in delta if x.startswith("+ "))
check(all(("root" in x or "cache" in x or "DSV41_MOE_FC2C" in x or "fc2c-d05" in x or "_verify_tree" in x
           or x.startswith('"""') or x.startswith("for other")) for x in removed + added) and len(removed) <= 8,
      f"load_fc2c_native == 9e250455 except tree selection ({len(removed)} lines out, {len(added)} in)")
delta = line_delta(src(g1, "TargetM32Kernel.gemm"), src(staged, "TargetM32Kernel.gemm"))
check(not [x for x in delta if x.startswith("- ")]
      and [x[2:].strip() for x in delta if x.startswith("+ ")] == [
          "if self.g1_la:  # g1-la cp.async's A rows at row stride K: contiguous, 16-B aligned A only",
          "aq = args[0][0]",
          'assert aq.dim() == 2 and aq.is_contiguous() and aq.data_ptr() % 16 == 0, "g1-la: bad A layout"',
          'if getattr(fc2c, "poison_served", False):',
          "# g2: a g1 'ds'/'all' D05 kernel stores live rows only, so the served buffer gets the",
          "# shadow's poison first: rows neither kernel writes then compare equal. Never read.",
          "import torch", "", "args[2].view(torch.int16).fill_(_D05Verifier.POISON)"],
      "TargetM32Kernel.gemm == 9e250455 + g1-la guard (db659d80) + verify poison (g2)")


class DT:
    def __init__(self, s):
        self.s = s

    def __str__(self):
        return self.s


LOG = []


class T:
    _ptr = 0x1000

    def __init__(self, shape, dtype="torch.bfloat16", contiguous=True, ptr=None, name=None):
        self.shape = tuple(shape)
        self.dtype = DT(dtype)
        self._c = contiguous
        T._ptr += 0x1000
        self._p = T._ptr if ptr is None else ptr
        self.name = name

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._c

    def data_ptr(self):
        return self._p

    def view(self, dtype):
        return TV(self, dtype)


class TV:
    def __init__(self, base, dtype):
        self.base, self.vdtype = base, dtype

    def fill_(self, value):
        LOG.append(("poison", self.base.name, str(self.vdtype), value))
        return self


class Native:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def m_grouped_fp8_fp4_gemm_nt_contiguous(self, *args, **kwargs):
        self.log.append((self.name, len(args), tuple(sorted(kwargs.items(), key=lambda kv: kv[0]))))
        return self.name


class Verifier:
    def __init__(self, log):
        self.log = log

    def check(self, fc2_native, args, kwargs):
        self.log.append(("verify", fc2_native.name, tuple(sorted(kwargs))))

    def check_compact(self, fc2c, args, kwargs):
        self.log.append(("verify_compact", fc2c.native.name, tuple(sorted(kwargs))))


W2 = T((384, 5120, 192), "torch.int8")


def fc2_args(m=1152, kb=192, kp=384, n=5120, e=384, out_dtype="torch.bfloat16", layout_dtype="torch.int32",
             contiguous=True, w2=None):
    w = w2 if w2 is not None else (W2 if (e, n, kb) == (384, 5120, 192) else T((e, n, kb), "torch.int8"))
    return ((T((m, kp), "torch.float8_e4m3fn"), T((m, 1), "torch.int32")),
            (w, T((e, n, 3), "torch.int32")),
            T((m, n), out_dtype, contiguous, name="d"), T((m,), layout_dtype))


def fc1_args(m=1152):
    return ((T((m, 5120), "torch.float8_e4m3fn"), T((m, 40), "torch.int32")),
            (T((384, 576, 2560), "torch.int8"), T((384, 576, 160), "torch.int32")),
            T((m, 576)), T((m,), "torch.int32"))


RK = dict(recipe_a=(1, 128), recipe_b=(1, 32))

# ---- 1. FC2C unset: staged == live, call for call (D05 0 / 1 / verify; g1-la off/on)
cases = [(fc2_args(), RK), (fc1_args(), RK), (fc2_args(m=64), RK), (fc2_args(m=192), RK), (fc2_args(m=12480), RK),
         (fc2_args(), dict(RK, disable_ue8m0_cast=False)), (fc2_args(kb=144, kp=288), RK)]
for d05 in ("0", "1", "verify"):
    for la in (False, True):
        log_live, log_st = [], []
        if d05 == "0":
            k_live = live.TargetM32Kernel(Native("orig", log_live))
            k_st = staged.TargetM32Kernel(Native("orig", log_st))
        else:
            k_live = live.TargetM32Kernel(Native("orig", log_live), Native("d05", log_live), d05,
                                          Verifier(log_live) if d05 == "verify" else None)
            k_st = staged.TargetM32Kernel(Native("orig", log_st), Native("d05", log_st), d05,
                                          Verifier(log_st) if d05 == "verify" else None)
        k_live.g1_la = k_st.g1_la = la
        r_live = [k_live.gemm(*a, **kw) for a, kw in cases]
        r_st = [k_st.gemm(*a, **kw) for a, kw in cases]
        check(log_live == log_st and r_live == r_st and len(log_live) >= len(cases) and k_st.fc2c is None,
              f"FC2C unset, D05={d05}, g1_la={la}: staged issues exactly the live calls ({len(log_live)})")
k_la = staged.TargetM32Kernel(Native("orig", []), Native("d05", []), "1")
k_la.g1_la = True
check(k_la.gemm(*fc2_args(), **RK) == "d05", "g1-la guard passes an aligned contiguous A")
bad = fc2_args()
bad[0][0]._p += 4
try:
    k_la.gemm(*bad, **RK)
    check(False, "g1-la guard rejects a misaligned A")
except AssertionError:
    check(True, "g1-la guard rejects a misaligned A")

# ---- 2. mode parsing, min rows
for raw, want in (("", "0"), ("0", "0"), ("off", "0"), ("1", "1"), ("on", "1"), ("verify", "verify")):
    os.environ["DSV41_MOE_FC2C"] = raw
    check(staged.fc2c_mode() == want, f"fc2c mode {raw!r} -> {want}")
os.environ["DSV41_MOE_FC2C"] = "bogus"
try:
    staged.fc2c_mode()
    check(False, "bogus fc2c mode rejected")
except ValueError:
    check(True, "bogus fc2c mode rejected")
os.environ.pop("DSV41_MOE_FC2C")


def msum(m, topk=6, e=384, align=32):  # deep_gemm_utils.compute_aligned_M_and_alignment (no CPU meta)
    rows = m * topk + min(m * topk, e) * (align - 1)
    return -(-rows // align) * align


check(all(staged.fc2c_min_rows(t) == msum(t) for t in range(1, 121)), "fc2c_min_rows(t) == M_sum(t), t=1..120")
check(staged.fc2c_min_rows() == 384 and staged.fc2c_min_rows(1) == 192, "default gate 2 tokens = 384 rows")
check(all((msum(t) >= staged.fc2c_min_rows(2)) == (t >= 2) for t in range(1, 121)), "gate monotonic: M>=2")
os.environ["DSV41_MOE_FC2C_MIN_TOKENS"] = "1"
check(staged.fc2c_min_rows() == 192, "DSV41_MOE_FC2C_MIN_TOKENS=1 -> 192 rows (compact at M=1)")
os.environ.pop("DSV41_MOE_FC2C_MIN_TOKENS")

# ---- 3. FC2C=1 routing with fake natives and a fake tail registry (g1-la on, as with 'all')
log = []
fc2c_native = Native("fc2c", log)
fc = staged.FC2Compact(fc2c_native, "1", staged.fc2c_min_rows(2))
built = []


def fake_descs(bw, capturing=False, fc=fc):
    e = fc.entries.get(bw.data_ptr())
    if e is not None:
        return e[2]
    if capturing:
        fc.capture_misses += 1
        return None
    d = ("descs", bw.data_ptr())
    fc.entries[bw.data_ptr()] = (bw, "tail", d)
    built.append(bw.data_ptr())
    return d


fc.descs = fake_descs
k = staged.TargetM32Kernel(Native("orig", log), Native("d05", log), "1", None, fc)
k.g1_la = True
for m, want in ((192, "d05"), (384, "fc2c"), (1152, "fc2c"), (9216, "fc2c"), (12480, "fc2c")):
    log.clear()
    r = k.gemm(*fc2_args(m=m), **RK)
    kw = dict(log[-1][2])
    ok = r == want and log[-1][0] == want and kw.get("dsv41_fc2_live_store") is True and kw.get(
        "ensure_zero_padding") is False
    if want == "fc2c":
        ok = ok and kw.get("dsv41_fc2c_tail_descs") == ("descs", W2.data_ptr()) and kw.get("recipe_a") == (1, 128)
    else:
        ok = ok and "dsv41_fc2c_tail_descs" not in kw
    check(ok and len(log) == 1, f"FC2C=1: M_sum {m} -> {want}")
check(built == [W2.data_ptr()], "one tail per FC2 weight, reused")
w2b = T((384, 5120, 192), "torch.int8")
log.clear()
k.gemm(*fc2_args(m=1152, w2=w2b), **RK)
check(dict(log[-1][2]).get("dsv41_fc2c_tail_descs") == ("descs", w2b.data_ptr()) and len(built) == 2,
      "second FC2 weight gets its own tail")
for name, args, kw in (("FC1", fc1_args(), RK), ("other K", fc2_args(kb=144, kp=288), RK),
                       ("other E", fc2_args(e=128), RK), ("rows > cap", fc2_args(m=12896), RK),
                       ("extra kwarg", fc2_args(), dict(RK, disable_ue8m0_cast=False))):
    log.clear()
    r = k.gemm(*args, **kw)
    check(r == "orig" and len(log) == 1 and "dsv41_fc2c_tail_descs" not in dict(log[-1][2]),
          f"FC2C=1: {name} stays on the base native")
fc.descs = lambda bw: fake_descs(bw, capturing=True)
w2c = T((384, 5120, 192), "torch.int8")
log.clear()
r = k.gemm(*fc2_args(m=1152, w2=w2c), **RK)
check(r == "d05" and len(log) == 1 and "dsv41_fc2c_tail_descs" not in dict(log[-1][2])
      and fc.capture_misses == 1, "FC2C=1: weight without tail during capture -> D05 fallback")
fc.descs = fake_descs
try:
    staged.TargetM32Kernel(Native("orig", log), None, "0", None, fc)
    check(False, "FC2C without D05 rejected")
except AssertionError:
    check(True, "FC2C without D05 rejected")

# ---- 4. FC2C=verify: D05 served (served buffer poisoned first when poison_served), then the compact shadow
for poison_served in (False, True):
    LOG.clear()
    fv = staged.FC2Compact(Native("fc2c", LOG), "verify", staged.fc2c_min_rows(2), Verifier(LOG))
    fv.descs = lambda bw, fv=fv: ("descs", bw.data_ptr())
    if poison_served:
        fv.poison_served = True
    kv = staged.TargetM32Kernel(Native("orig", LOG), Native("d05", LOG), "1", None, fv)
    kv.g1_la = True
    r = kv.gemm(*fc2_args(m=1152), **RK)
    want = (["poison", "d05", "verify_compact"] if poison_served else ["d05", "verify_compact"])
    pz = [x for x in LOG if x[0] == "poison"]
    check(r == "d05" and [x[0] for x in LOG] == want
          and all(x[1] == "d" and x[2] == "torch.int16" and x[3] == staged._D05Verifier.POISON == -1 for x in pz),
          f"FC2C=verify poison_served={poison_served}: order {want}")
    LOG.clear()
    r = kv.gemm(*fc2_args(m=192), **RK)
    check(r == "d05" and [x[0] for x in LOG] == ["d05"], f"FC2C=verify poison_served={poison_served}: M=1 serves D05 only")

# ---- 5. tree selection (no file access beyond the env; pins as constants)
home = Path.home()
os.environ["DSV41_MOE_FC2_D05"] = "1"
os.environ["DSV41_MOE_M32_G1"] = "all"
root, cache, pin = staged.fc2c_tree()
check(root == (home / "dsv41-stage/g2/fc2c-on-m32all/tree").resolve()
      and cache == (home / "dsv41-stage/g2/fc2c-on-m32all/jit-cache").resolve()
      and pin == staged._G2_FC2C_MANIFEST_SHA256, "G1=all -> g2 tree, g2 jit-cache, g2 pin")
os.environ["DSV41_MOE_FC2C_JIT_DIR"] = "/tmp/x-jit"
os.environ["DSV41_MOE_FC2C_NATIVE_DIR"] = "/tmp/x-tree"
root, cache, pin = staged.fc2c_tree()
check(str(root) == str(Path("/tmp/x-tree").resolve()) and str(cache) == str(Path("/tmp/x-jit").resolve())
      and pin == staged._G2_FC2C_MANIFEST_SHA256, "env overrides honoured, pin unchanged")
os.environ.pop("DSV41_MOE_FC2C_JIT_DIR")
os.environ.pop("DSV41_MOE_FC2C_NATIVE_DIR")
os.environ["DSV41_MOE_M32_G1"] = "0"
root, cache, pin = staged.fc2c_tree()
check(root == (home / "dsv41-stage/g1/fc2c-d05/native").resolve()
      and cache == (home / "dsv41-stage/g1/fc2c-d05/jit-cache").resolve()
      and pin == staged._FC2C_MANIFEST_SHA256, "G1 off -> g1 fc2c-d05 tree (9e250455 behaviour)")
for v in ("pair", "pf", "la", "ds"):
    os.environ["DSV41_MOE_M32_G1"] = v
    try:
        staged.fc2c_tree()
        check(False, f"G1={v}: refused")
    except AssertionError:
        check(True, f"G1={v}: refused")
os.environ["DSV41_MOE_M32_G1"] = "all"
os.environ["DSV41_MOE_FC2_D05"] = "0"
try:
    staged.fc2c_tree()
    check(False, "G1=all with D05=0 refused")
except AssertionError:
    check(True, "G1=all with D05=0 refused")
os.environ["DSV41_MOE_FC2_D05"] = "1"
os.environ["DSV41_MOE_PDL_EARLY_B"] = "1"
try:
    staged.load_fc2c_native(None, False)
    check(False, "FC2C with PDL_EARLY_B refused")
except AssertionError:
    check(True, "FC2C with PDL_EARLY_B refused")
for k_ in ("DSV41_MOE_PDL_EARLY_B", "DSV41_MOE_M32_G1", "DSV41_MOE_FC2_D05"):
    os.environ.pop(k_)

# ---- 6. the g2 tree on this node: pin, full verify, reproducible from the live trees
if stage is not None:
    sys.path.insert(0, str(stage / "tools"))
    import make_fc2c_tree as mft
    tree = stage / "tree"
    check(sha(tree / "manifest.json") == staged._G2_FC2C_MANIFEST_SHA256, "staged pin == tree/manifest.json")
    try:
        man = staged._verify_tree(tree.resolve(), staged._G2_FC2C_MANIFEST_SHA256, staged._FC2C_NATIVE_SHA256)
        check(True, f"tree verifies ({len(man['files'])} files, native {man['native_sha256'][:8]})")
    except AssertionError as exc:
        check(False, f"tree verifies: {exc}")
    alld05 = home / "dsv41-stage/g1/m32-bw/trees/all-d05" / mft.HDR
    check((tree / mft.HDR).read_text() == mft.fc2c_on_all(alld05.read_text()),
          "tree header == make_fc2c_tree.fc2c_on_all(live all-d05 header)")
    d05h = home / "dsv41-goal120-20261024/d05-shadow-v1/native" / mft.HDR
    g1h = home / "dsv41-stage/g1/fc2c-d05/native" / mft.HDR
    try:
        mft.self_check(d05h.read_text(), g1h.read_text(), alld05.read_text())
        check(True, "hunks reproduce the qualified g1 header from the D05 header")
    except AssertionError as exc:
        check(False, f"hunk self-check: {exc}")

# ---- 7. tail bytes vs the padded kernel's shared-memory data bytes (16U4_ALIGN16B model)
g = torch.Generator().manual_seed(7)
E_, N_ = 3, 64
w2 = torch.randint(0, 256, (E_, N_, 192), dtype=torch.uint8, generator=g)
w2[:, :, 144:] = 0                                   # TP padding k >= 288
tail = staged.fc2c_tail_from_bytes(w2)
check(tuple(tail.shape) == (E_ * N_, 32) and tail.dtype == torch.uint8, "tail shape [E*N, 32] uint8")
rows = w2.reshape(E_ * N_, 192)


def padded_kblock2_data(row):
    """Data bytes the u4a16 TMA writes for k 256..383 (64 packed bytes -> 8 chunks x [8 data | 8 pad])."""
    img = torch.full((128,), -1, dtype=torch.int16)    # -1 = pad byte (left untouched by the TMA)
    for c in range(8):
        img[16 * c:16 * c + 8] = row[128 + 8 * c:136 + 8 * c].to(torch.int16)
    return img


def tail_kblock2_data(trow):
    """Bytes the UINT8 tail TMA writes: 32 in-bounds bytes, then OOB zero fill to 128."""
    img = torch.zeros((128,), dtype=torch.int16)
    img[:32] = trow.to(torch.int16)
    return img


same = True
for r in range(rows.shape[0]):
    a_, b_ = padded_kblock2_data(rows[r]), tail_kblock2_data(tail[r])
    data = a_ >= 0                                   # positions ldmatrix .b4x16_p64 reads
    same &= bool(torch.equal(a_[data], b_[data]))
check(same, "every data byte of the last k-block equals the padded kernel's (all rows)")
check(bool((tail[:, 8:16] == 0).all() and (tail[:, 24:32] == 0).all()), "tail pad bytes are zero")
w2bad = w2.clone()
w2bad[1, 5, 150] = 1
try:
    staged.fc2c_tail_from_bytes(w2bad)
    check(False, "non-zero K padding rejected")
except AssertionError:
    check(True, "non-zero K padding rejected")

print("SUMMARY", "PASS" if fails == 0 else f"FAIL {fails}", flush=True)
sys.exit(1 if fails else 0)
