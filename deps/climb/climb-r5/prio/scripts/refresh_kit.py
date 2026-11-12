"""Recompute KIT.json patched_sha256 / kit_sha256 and regenerate patched/ (review copies) from base/ + install.py."""
import hashlib, importlib.util, json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("inst", os.path.join(HERE, "install.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
kit = json.load(open(os.path.join(HERE, "KIT.json")))
for f in kit["files"]:
    if f["action"] == "patch":
        src = open(os.path.join(HERE, "base", "vllm", f["path"])).read()
        assert hashlib.sha256(src.encode()).hexdigest() == f["base_sha256"], f["path"]
        new = m.patched(src, f["path"])
        out = os.path.join(HERE, "patched", "vllm", f["path"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write(new)
        f["patched_sha256"] = hashlib.sha256(new.encode()).hexdigest()
    else:
        f["kit_sha256"] = hashlib.sha256(open(m.MOD_SRC, "rb").read()).hexdigest()
    print(f["path"], f.get("patched_sha256") or f.get("kit_sha256"))
json.dump(kit, open(os.path.join(HERE, "KIT.json"), "w"), indent=1)
open(os.path.join(HERE, "KIT.json"), "a").write("\n")
