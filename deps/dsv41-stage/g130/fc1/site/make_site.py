#!/usr/bin/env python3
"""g130 fc1: anchored patch of the live sm121_target_m32.py.  python3 -I make_site.py <base> <out> <MANIFESTS.json>
Base sha is pinned. With DSV41_MOE_FC1_DEAL and DSV41_MOE_FC1_ROUTE_REC unset, the patched file takes the same path as production:
same tree, same JIT cache, same manifest, and one extra attribute test in gemm(), which only runs at capture or eagerly."""
import hashlib
import json
import sys
from pathlib import Path

BASE_SHA = "ea4a8814986c3618d26d7acb748561d75e50e839cbea225d9f06154febb290e5"


def patch(s, manifests):
    def sub(old, new):
        nonlocal s
        assert s.count(old) == 1, (old[:80], s.count(old))
        s = s.replace(old, new)

    table = {k.split("-")[0]: v for k, v in manifests.items() if k in ("deal-base", "dealx-base")}
    assert set(table) == {"deal", "dealx"}, table
    sub('''def g1_variant() -> str:
''', '''# g130 fc1 G-aware tile deal (default off; kit ~/dsv41-stage/g130/fc1): DSV41_MOE_FC1_DEAL=deal|dealx swaps only the tree that
# compiles the target FC1 (same native binary, patched JIT header); FC2 and the E128 drafter kernels are SASS-identical.
_FC1DEAL_ROOT = Path.home() / "dsv41-stage/g130/fc1"
_FC1DEAL_MANIFESTS = %s


def fc1_deal_arm() -> str:
    raw = os.environ.get("DSV41_MOE_FC1_DEAL", "0").strip().lower()
    if raw in ("", "0", "off"):
        return ""
    assert raw in _FC1DEAL_MANIFESTS, f"DSV41_MOE_FC1_DEAL={raw!r}: use 0 or one of {tuple(_FC1DEAL_MANIFESTS)}"
    assert g1_variant() == "all", "DSV41_MOE_FC1_DEAL is qualified with DSV41_MOE_M32_G1=all only"
    return raw


def fc1_route_rec():
    """DSV41_MOE_FC1_ROUTE_REC=1: side-file recorder (diagnostic instance only); None otherwise."""
    if os.environ.get("DSV41_MOE_FC1_ROUTE_REC", "0").strip() != "1":
        return None
    spec = importlib.util.spec_from_file_location("dsv41_fc1_route_rec", _FC1DEAL_ROOT / "site/fc1_route_rec.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RouteRec()


def g1_variant() -> str:
''' % json.dumps(table, indent=4))
    sub('''        self.fc2c = fc2c
        self.g1_la = False
''', '''        self.fc2c = fc2c
        self.g1_la = False
        self.fc1_rec = None
''')
    sub('''        assert "ensure_zero_padding" not in kwargs
        if self.g1_la:''', '''        assert "ensure_zero_padding" not in kwargs
        if self.fc1_rec is not None:
            self.fc1_rec.observe(args)
        if self.g1_la:''')
    sub('''        manifest_sha = _G1_MANIFESTS[f"{g1_variant()}-base"]
    else:
        root = Path(os.environ["DSV41_TARGET_M32_NATIVE_DIR"]).resolve()
''', '''        manifest_sha = _G1_MANIFESTS[f"{g1_variant()}-base"]
        if fc1_deal_arm():
            root = (_FC1DEAL_ROOT / f"trees/{fc1_deal_arm()}-base").resolve()
            jit_cache = (_FC1DEAL_ROOT / f"jit/{fc1_deal_arm()}-base").resolve()
            manifest_sha = _FC1DEAL_MANIFESTS[fc1_deal_arm()]
    else:
        root = Path(os.environ["DSV41_TARGET_M32_NATIVE_DIR"]).resolve()
''')
    sub('''    kernel.g1_la = g1_variant() in _G1_HAS_LA
    return kernel
''', '''    kernel.g1_la = g1_variant() in _G1_HAS_LA
    kernel.fc1_rec = fc1_route_rec()
    return kernel
''')
    return s


def main():
    base, out, man = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    raw = base.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA, f"base sha {hashlib.sha256(raw).hexdigest()} != pinned {BASE_SHA}"
    new = patch(raw.decode(), json.loads(man.read_text()))
    out.write_text(new)
    compile(new, str(out), "exec")
    print("SITE", out, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
