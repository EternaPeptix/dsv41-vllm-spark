#!/usr/bin/env python3
"""Regenerate KIT.json's computed parts (install_files shas, pins) from the files on disk; the prose parts live in
tools/kit_template.json. Run after any change: python3 tools/mkkit.py"""
import hashlib
import json
import os

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINNED = [
    "install.py", "device_test.py", "tests/run_cpu.sh", "ops/fleet.sh", "ops/inline_kit.py", "ops/stage.sh", "ops/plan.sh", "ops/serve.sh", "ops/arms-fixb.txt", "common/prodenv.py",
    "head/device_test.py", "head/tools/mutfit.py", "head/install.py",
    "head/files/vllm/models/deepseek_v41/nvidia/dsv41_head_r6.py",
    "head/base/vllm/models/deepseek_v41/nvidia/dsv41_r4_small.py",
    "step/device_test.py", "step/det_gate.py", "step/install.py",
    "step/files/vllm/models/deepseek_v41/nvidia/dsv41_step.py", "step/prev/dsv41_step.py.r6-ebe2f7bb",
    "skew/device_test.py", "skew/node/iso.py", "skew/node/hostcensus.py", "skew/node/nsx_extract.py",
    "skew/node/auditrun.py",
]


def sha(p):
    return hashlib.sha256(open(os.path.join(KIT, p), "rb").read()).hexdigest()


t = json.load(open(os.path.join(KIT, "tools", "kit_template.json")))
f = t["install_files"][0]
f["sha256"] = sha("step/files/vllm/models/deepseek_v41/nvidia/dsv41_step.py")
f["base_sha256"] = sha("step/prev/dsv41_step.py.r6-ebe2f7bb")
t["pins"] = {p: sha(p) for p in PINNED}
json.dump(t, open(os.path.join(KIT, "KIT.json"), "w"), indent=1)
print("KIT.json:", len(t["pins"]), "pins; r8 module", f["sha256"][:16], "r6", f["base_sha256"][:16])
