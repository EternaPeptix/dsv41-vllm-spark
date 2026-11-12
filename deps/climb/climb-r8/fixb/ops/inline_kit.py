"""Turn install.py (stdin) into a self-contained script for a read-only `status` over ssh: KIT.json is embedded, so
nothing is written on the node (python3 - status < this output)."""
import json
import os
import sys

kit = json.load(open(os.path.join(sys.argv[1], "KIT.json")))
src = sys.stdin.read()
src = src.replace('KIT = json.load(open(os.path.join(HERE, "KIT.json")))', "KIT = " + repr(kit))
src = src.replace('SRC = os.path.join(HERE, "step", "files", "vllm", REL)', "SRC = None")
src = src.replace("    if sha(SRC) != R8_SHA:\n", "    if SRC is not None and sha(SRC) != R8_SHA:\n")
assert "KIT = {" in src and "SRC = None" in src
sys.stdout.write(src)
