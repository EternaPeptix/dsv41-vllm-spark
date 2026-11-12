#!/usr/bin/env python3
"""r2-attnout (climb round 2, lever G19 / attn:L-OP): install / uninstall / check the DSV41_ATTNOUT_OP hooks.

    python3 install.py install | uninstall | status | dry

Patches two files (anchor edits with exact expected counts, every inserted or modified line tagged
`# r2-attnout`, backup `<file>.bak-r2-attnout`) and copies one new module next to sonar_bf16_bmm.py:

  vllm/models/deepseek_v4/nvidia/ops/o_proj.py          module flag + a guarded dispatch in deep_gemm_fp8_o_proj
  vllm/models/deepseek_v4/nvidia/ops/sonar_bf16_bmm.py  try_project(..., r2ao=None) + a guarded dispatch
  vllm/models/deepseek_v4/nvidia/ops/dsv41_attnout.py   NEW (kernels + host logic, imported only when the flag is on)

Every patched file's sha256 must equal one of the bases recorded in KIT.json (the live file, or the file with the
round-1 kits installed: r1-pdl patches sonar_bf16_bmm.py) -- refuses otherwise. Idempotent. With DSV41_ATTNOUT_OP
unset/0, o_proj.py's hook is `if <module-level False> and ...:` and nobody passes r2ao, so the live code runs.

Env: R2AO_VLLM (default ~/venvs/vllm032/lib/python3.12/site-packages/vllm) for tests.
"""
import hashlib
import json
import os
import shutil
import sys

KIT = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("R2AO_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
TAG = "# r2-attnout"
BAK = ".bak-r2-attnout"

O_PROJ = "models/deepseek_v4/nvidia/ops/o_proj.py"
SONAR = "models/deepseek_v4/nvidia/ops/sonar_bf16_bmm.py"
NEW = {"models/deepseek_v4/nvidia/ops/dsv41_attnout.py":
       os.path.join(KIT, "files/models/deepseek_v4/nvidia/ops/dsv41_attnout.py")}

FLAG_LINE = ('_DSV41_ATTNOUT_OP = __import__("os").environ.get("DSV41_ATTNOUT_OP", "0").strip().lower() '
             f'not in ("", "0", "off")  {TAG}\n')

DISPATCH = (
    f"    if _DSV41_ATTNOUT_OP and not use_fp8 and not isinstance(o, QuantizedActivation):  {TAG}\n"
    f"        from vllm.models.deepseek_v4.nvidia.ops import dsv41_attnout as _r2ao  {TAG}\n"
    f"        _r2y = _r2ao.o_proj(  {TAG}\n"
    f"            o, positions, cos_sin_cache, wo_a, wo_b, n_groups=n_groups,  {TAG}\n"
    f"            heads_per_group=heads_per_group, nope_dim=nope_dim, rope_dim=rope_dim,  {TAG}\n"
    f"            o_lora_rank=o_lora_rank, quant_group_size=einsum_recipe[2],  {TAG}\n"
    f"            tma_aligned_scales=tma_aligned_scales,  {TAG}\n"
    f"        )  {TAG}\n"
    f"        if _r2y is not None:  {TAG}\n"
    f"            return _r2y  {TAG}\n"
)

# relpath -> list of (old, new, expected count)
EDITS = {
    O_PROJ: [
        ("from vllm.utils.deep_gemm import fp8_einsum, get_tma_aligned_size\n",
         "from vllm.utils.deep_gemm import fp8_einsum, get_tma_aligned_size\n" + FLAG_LINE, 1),
        ("    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn\n"
         "    if isinstance(o, QuantizedActivation):\n",
         "    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn\n" + DISPATCH
         + "    if isinstance(o, QuantizedActivation):\n", 1),
    ],
    SONAR: [
        ("def try_project(o_proj_input: torch.Tensor, wo_a, grouped_weight: torch.Tensor, z: torch.Tensor) -> bool:\n",
         "def try_project(o_proj_input: torch.Tensor, wo_a, grouped_weight: torch.Tensor, z: torch.Tensor,  "
         f"{TAG}\n"
         f"                r2ao=None) -> bool:  {TAG}\n", 1),
        ("    grouped_bmm(o_proj_input, weight, z, scale)\n    return True\n",
         f"    if r2ao is not None:  {TAG}\n"
         f"        return r2ao.run(o_proj_input, weight, scale, z)  {TAG}\n"
         "    grouped_bmm(o_proj_input, weight, z, scale)\n    return True\n", 1),
    ],
}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "-"


def kit_json():
    return json.load(open(os.path.join(KIT, "KIT.json")))


def allowed_bases(rel):
    for f in kit_json()["files"]:
        if f["path"] == f"vllm/{rel}" and f["action"] == "patch":
            return set([f["base_sha256"]] + list(f.get("base_sha256_with_round1", {}).values()))
    raise SystemExit(f"KIT.json has no patch entry for vllm/{rel}")


def patched(rel, src):
    for old, new, count in EDITS[rel]:
        n = src.count(old)
        if n != count:
            raise SystemExit(f"{rel}: anchor found {n}x, expected {count}: {old[:80]!r}")
        src = src.replace(old, new)
    return src


def install():
    plans = []
    for rel in EDITS:
        p = os.path.join(V, rel)
        src = open(p).read()
        if TAG in src:
            bak = p + BAK
            if not os.path.exists(bak) or sha(bak) not in allowed_bases(rel):
                raise SystemExit(f"{p}: tagged but its backup is missing or not a recorded base; refusing")
            print(f"{p}: already patched ({sha(p)[:8]})")
            continue
        if sha(p) not in allowed_bases(rel):
            raise SystemExit(f"{p}: sha256 {sha(p)[:16]} is not a recorded base {sorted(b[:16] for b in allowed_bases(rel))}; "
                             "refusing")
        plans.append((rel, p, src, patched(rel, src)))  # all anchors checked before any write
    for rel, p, src, new in plans:
        bak = p + BAK
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
        tmp = p + ".tmp-r2-attnout"
        with open(tmp, "w") as f:
            f.write(new)
        shutil.copymode(bak, tmp)
        os.replace(tmp, p)
        print(f"{p}: patched ({sha(bak)[:8]} -> {sha(p)[:8]}), {new.count(TAG)} tagged lines")
    for rel, srcp in NEW.items():
        dst = os.path.join(V, rel)
        shutil.copy2(srcp, dst)
        print(f"{dst}: {sha(dst)[:8]} (new)")


def uninstall():
    # Restore a backup only when the current file is exactly patched(backup): if another kit edited the file
    # after this one (e.g. r1-pdl installed AFTER r2-attnout), restoring would silently drop its edits -> refuse,
    # before any write. Order: install r1-pdl first, uninstall r2-attnout first.
    restores = []
    for rel in EDITS:
        p = os.path.join(V, rel)
        bak = p + BAK
        if os.path.exists(bak):
            want = hashlib.sha256(patched(rel, open(bak).read()).encode()).hexdigest()
            if sha(p) != want:
                raise SystemExit(f"{p}: sha256 {sha(p)[:16]} != patched(backup) {want[:16]}: the file changed after "
                                 "r2-attnout was installed (another kit installed on top?); uninstall that first. "
                                 "Refusing; nothing restored")
            restores.append((p, bak))
        elif os.path.exists(p) and TAG in open(p).read():
            raise SystemExit(f"{p}: tagged but no backup {bak}; refusing to guess")
    for p, bak in restores:
        os.replace(bak, p)
        print(f"{p}: restored {sha(p)[:8]}")
    for rel in NEW:
        dst = os.path.join(V, rel)
        if os.path.exists(dst):
            os.remove(dst)
            print(f"{dst}: removed")


def status():
    for rel in EDITS:
        p = os.path.join(V, rel)
        on = os.path.exists(p) and TAG in open(p).read()
        base = "base" if sha(p) in allowed_bases(rel) else ("" if on else "UNKNOWN-BASE")
        print(f"{sha(p)[:8]} {'PATCHED' if on else 'clean  '} {base:12s} {p}")
    for rel, srcp in NEW.items():
        dst = os.path.join(V, rel)
        state = "present" if os.path.exists(dst) else "absent "
        same = "" if not os.path.exists(dst) else ("(kit copy)" if sha(dst) == sha(srcp) else "(DIFFERS from kit)")
        print(f"{sha(dst)[:8]} {state} {same} {dst}")


def dry():
    for rel in EDITS:
        p = os.path.join(V, rel)
        src = open(p).read()
        if TAG in src:
            print(f"{p}: already patched")
            continue
        if sha(p) not in allowed_bases(rel):
            print(f"{p}: sha256 {sha(p)[:16]} not a recorded base (install would refuse)")
        for i, line in enumerate(patched(rel, src).splitlines(), 1):
            if TAG in line:
                print(f"{rel}:{i:5d}: {line}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("install", "uninstall", "status", "dry"):
        raise SystemExit(__doc__)
    {"install": install, "uninstall": uninstall, "status": status, "dry": dry}[sys.argv[1]]()
