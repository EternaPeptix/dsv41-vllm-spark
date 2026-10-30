#!/usr/bin/env python3
"""g1/m32-bw: staged sm121_target_m32.py = live 5acc5fde + the DSV41_MOE_M32_G1 switch (default off). CPU only.

  python3 -I make_site.py <live sm121_target_m32.py> <out file> <MANIFESTS.json>

With DSV41_MOE_M32_G1 unset/0 every code path is the live one (the flag is read only inside the two loaders and a
False attribute check in gemm). With DSV41_MOE_M32_G1=<pair|pf|la|ds|all> the M32 base native comes from
~/dsv41-stage/g1/m32-bw/trees/<v>-base (+ jit/<v>-base) and, for ds/all, the target-FC2 D05 native from
trees/<v>-d05 (+ jit/<v>-d05); both trees are manifest-pinned below and carry the unchanged native .so binaries.
"""
import hashlib
import json
import sys
from pathlib import Path

LIVE_SHA = "5acc5fde"
src, dst, mf = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
s = src.read_text()
assert hashlib.sha256(s.encode()).hexdigest().startswith(LIVE_SHA), "live sm121_target_m32.py is not 5acc5fde"
man = json.loads(mf.read_text())


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


sub('''Pair with DSV41_MOE_PDL_TRIGGER=start|end (fp8_utils.py SiLU trigger) for overlap.
"""''', '''Pair with DSV41_MOE_PDL_TRIGGER=start|end (fp8_utils.py SiLU trigger) for overlap.

g1/m32-bw (default off): DSV41_MOE_M32_G1=pair|pf|la|ds|all loads the EXACT bandwidth variants of the same native
binaries from ~/dsv41-stage/g1/m32-bw (trees/<v>-base for FC1, trees/<v>-d05 for target FC2 with ds/all; pinned
manifests below; pre-JIT caches jit/<tree>). Requires DSV41_MOE_FC2_D05=1, excludes DSV41_MOE_PDL_EARLY_B.
  pair  FC1 TMA leader issues k-block pairs back to back (whole 128-B weight lines per request pair)
  pf    FC1 B/SFB L2 prefetch cursor 8 k-blocks ahead in 4-k-block groups + EVICT_FIRST weight loads
  la    FC1 loads only live A rows (cp.async, producer warp 9) instead of the 32-row TMA box
  ds    FC1 + FC2 store only live rows (32-bit st.global of the same BF16 pairs; no smem staging/TMA store)
  all   FC1 pf+la+ds + D04 dead-walk stop; FC2 ds + D04 + EVICT_FIRST weight loads
Every live output row is bitwise identical (same MMA sequence per element); see the g1/m32-bw README.
"""''')

sub('''_PDL_D05_MANIFEST_SHA256 = "2009e01179eacde44e10de2d2be2ca1b05dea72c04f9dbfd1bbcf57915c89ee9"
''', '''_PDL_D05_MANIFEST_SHA256 = "2009e01179eacde44e10de2d2be2ca1b05dea72c04f9dbfd1bbcf57915c89ee9"
# g1/m32-bw variant trees (tools/make_tree.py over the two immutable trees above; same .so binaries).
_G1_ROOT = Path.home() / "dsv41-stage/g1/m32-bw"
_G1_MANIFESTS = %s
_G1_VARIANTS = ("pair", "pf", "la", "ds", "all")
_G1_HAS_D05 = ("ds", "all")
_G1_HAS_LA = ("la", "all")


def g1_variant() -> str:
    raw = os.environ.get("DSV41_MOE_M32_G1", "0").strip().lower()
    if raw in ("", "0", "off"):
        return ""
    assert raw in _G1_VARIANTS, f"DSV41_MOE_M32_G1={raw!r}: use 0 or one of {_G1_VARIANTS}"
    assert not pdl_early_b(), "DSV41_MOE_M32_G1 and DSV41_MOE_PDL_EARLY_B are exclusive"
    assert d05_mode() == "1", "DSV41_MOE_M32_G1 is qualified with DSV41_MOE_FC2_D05=1 only"
    return raw
''' % json.dumps({k: v for k, v in man.items() if not k.startswith("off-")}, indent=4).replace("\n}", ",\n}"))

sub('''        self.d05 = d05
        self.verifier = verifier
''', '''        self.d05 = d05
        self.verifier = verifier
        self.g1_la = False
''')

sub('''        assert "ensure_zero_padding" not in kwargs
''', '''        assert "ensure_zero_padding" not in kwargs
        if self.g1_la:  # g1-la cp.async's A rows at row stride K: contiguous, 16-B aligned A only
            aq = args[0][0]
            assert aq.dim() == 2 and aq.is_contiguous() and aq.data_ptr() %% 16 == 0, "g1-la: bad A layout"
'''.replace("%%", "%"))

sub('''        manifest_sha = _PDL_D05_MANIFEST_SHA256
    else:
        root = Path(os.environ.get(''', '''        manifest_sha = _PDL_D05_MANIFEST_SHA256
    elif g1_variant() in _G1_HAS_D05:
        root = (_G1_ROOT / f"trees/{g1_variant()}-d05").resolve()
        cache = (_G1_ROOT / f"jit/{g1_variant()}-d05").resolve()
        manifest_sha = _G1_MANIFESTS[f"{g1_variant()}-d05"]
    else:
        root = Path(os.environ.get(''')

sub('''        manifest_sha = _PDL_BASE_MANIFEST_SHA256
    else:
        root = Path(os.environ["DSV41_TARGET_M32_NATIVE_DIR"]).resolve()''', '''        manifest_sha = _PDL_BASE_MANIFEST_SHA256
    elif g1_variant():
        root = (_G1_ROOT / f"trees/{g1_variant()}-base").resolve()
        jit_cache = (_G1_ROOT / f"jit/{g1_variant()}-base").resolve()
        manifest_sha = _G1_MANIFESTS[f"{g1_variant()}-base"]
    else:
        root = Path(os.environ["DSV41_TARGET_M32_NATIVE_DIR"]).resolve()''')

sub('''    fc2_native = load_d05_native(native, pdl)
    verifier = _D05Verifier() if mode == "verify" else None
    return TargetM32Kernel(native, fc2_native, mode, verifier)''', '''    fc2_native = load_d05_native(native, pdl)
    verifier = _D05Verifier() if mode == "verify" else None
    kernel = TargetM32Kernel(native, fc2_native, mode, verifier)
    kernel.g1_la = g1_variant() in _G1_HAS_LA
    return kernel''')

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(s)
print("SITE", dst, hashlib.sha256(dst.read_bytes()).hexdigest())
