#!/usr/bin/env python3
"""r2-draftmoe installer (climb round 2, lever G17 = moe:L-D, flag DSV41_DRAFT_MOE_G17, default off).

    python3 install.py install | uninstall | status | dry | prejit [--force] | prejit-status

install    copies experts/dsv41_r2_draftmoe.py (sha-checked) next to dsv41_draft_m32.py, then anchor-edits 4 live
           files. Every edit is a pure insertion of lines ending in '# r2-draftmoe' (anchors kept verbatim, nothing
           replaced), so removing the tagged lines gives back the exact pre-install file. Accepted base per file: the
           live sha256 in KIT.json, or a file whose '# r1-pdl' / '# r2-moefront' tagged lines strip back to it (those
           kits installed first). Refuses anything else before writing. Idempotent. <file>.bak-r2-draftmoe is an
           informational copy of the pre-install file; nothing restores it.
uninstall  removes exactly our tagged lines, after checking (per file, all files before any write) that the file is our
           patch of an accepted base modulo the other kits' tagged lines; removes the backups and the new module.
           Never restores a .bak, so it cannot resurrect or drop another kit's lines. Documented order: LIFO
           (install r1-pdl -> r2-moefront -> r2-draftmoe; uninstall r2-draftmoe -> r2-moefront -> r1-pdl). The
           pre-JIT cache entries are left in place (unreachable with the kit off).
status     sha8 + PATCHED/clean/FOREIGN per file, plus the pre-JIT cache entries.
dry        prints the tagged lines each edit would produce, touching nothing.
prejit     CPU-only offline pre-JIT (same nvcc, flags, include tree and cache-key function as the DeepGEMM JIT, the
           procedure of ~/dsv41-stage/g130/fc1/ops/build-deal.sh) of the three new drafter FC1 instances (N 576,
           E 128): the M32 one into ~/dsv41-stage/g130/fc1/jit/deal-base/cache and the two 128-row tile configs
           (128x128 2-stage, 128x64 3-stage) into ~/.cache/vllm/deep_gemm/cache. nice 19, CUDA_VISIBLE_DEVICES empty,
           one nvcc at a time. An entry that already exists (committed or not) is never touched without --force;
           --force refuses while a vLLM process runs and moves the old entry aside (no rmtree). Without prejit the
           runtime JITs them on first use (boot or first long drafter call). Optional, never needed for exactness.

Env: R2DM_VLLM (default ~/venvs/vllm032/lib/python3.12/site-packages/vllm), R2DM_HOME (default ~) for tests.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("R2DM_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
HOME = os.environ.get("R2DM_HOME") or os.path.expanduser("~")
TAG = "# r2-draftmoe"
BAK = ".bak-r2-draftmoe"
KIT = json.load(open(os.path.join(HERE, "KIT.json")))

MXFP4 = "model_executor/layers/quantization/mxfp4.py"
DGMOE = "model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
DRM32 = "model_executor/layers/fused_moe/experts/dsv41_draft_m32.py"
DGUTL = "model_executor/layers/fused_moe/deep_gemm_utils.py"
NEWMOD = "model_executor/layers/fused_moe/experts/dsv41_r2_draftmoe.py"

PRELUDE = (
    f"_R2DM_FC1, _R2DM_PERM = False, \"\"  # G17 DSV41_DRAFT_MOE_G17, default off (experts/dsv41_r2_draftmoe.py)  {TAG}\n"
    f"if os.environ.get(\"DSV41_DRAFT_MOE_G17\", \"0\").strip().lower() not in (\"\", \"0\", \"off\", \"false\"):  {TAG}\n"
    f"    from vllm.model_executor.layers.fused_moe.experts import dsv41_r2_draftmoe as _r2dm  {TAG}\n"
    f"    _R2DM_FC1, _R2DM_PERM = _r2dm.FC1, _r2dm.PERM  {TAG}\n"
)

# Every edit is a PURE INSERTION of lines that end with TAG (anchor kept verbatim, nothing replaced, no untagged
# line added), so strip_tag(patched(x), TAG) == x for any x. install asserts it; uninstall relies on it.
EDITS = {
    MXFP4: [
        ("logger = init_logger(__name__)\n", "logger = init_logger(__name__)\n" + PRELUDE, 1),
        ('            self.w13_intermediate_size = 288\n'
         '            logger.info_once(\n'
         '                "DSV41_MOE_COMPACT: target compact FC1 enabled: logical width 288, "\n'
         '                "FC1 rows 576, FC2 input width 384"\n'
         '            )\n'
         '        weight_loader = extra_weight_attrs.pop("weight_loader")\n',
         '            self.w13_intermediate_size = 288\n'
         '            logger.info_once(\n'
         '                "DSV41_MOE_COMPACT: target compact FC1 enabled: logical width 288, "\n'
         '                "FC1 rows 576, FC2 input width 384"\n'
         '            )\n'
         f'        if _R2DM_FC1 and _r2dm.drafter_compact_eligible(  {TAG}\n'
         f'                self, num_experts, hidden_size, intermediate_size_per_partition,  {TAG}\n'
         f'                current_platform.is_device_capability(121),  {TAG}\n'
         f'                self.mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4, MoEActivation.SILU):  {TAG}\n'
         f'            self.w13_intermediate_size = 288  {TAG}\n'
         '        weight_loader = extra_weight_attrs.pop("weight_loader")\n', 1),
    ],
    DGMOE: [
        ("logger = init_logger(__name__)\n", "logger = init_logger(__name__)\n" + PRELUDE, 1),
        # M32 alignment: the compact drafter takes exactly the padded drafter's decision (same env gates, same M cap).
        ("    def _workspace_m_alignment(M: int, N: int, K: int, topk: int, experts: int) -> int:\n",
         "    def _workspace_m_alignment(M: int, N: int, K: int, topk: int, experts: int) -> int:\n"
         f"        if _R2DM_FC1 and (N, K, topk, experts) == _r2dm.DRAFT_COMPACT_GEOMETRY:  {TAG}\n"
         f"            return DeepGemmFP4Experts._workspace_m_alignment(M, 768, K, topk, experts)  {TAG}\n", 1),
        # apply(): the compact drafter sets compact_fc1 (same two asserts as the target, with the drafter's w2 shape).
        ("        if compact_fc1:\n"
         "            assert tuple(w2.shape) == (384, 5120, 192)\n"
         "            assert activation == MoEActivation.SILU\n",
         "        if compact_fc1:\n"
         "            assert tuple(w2.shape) == (384, 5120, 192)\n"
         "            assert activation == MoEActivation.SILU\n"
         f"        if _R2DM_FC1 and not compact_fc1 and (  {TAG}\n"
         f"                N, K, topk_ids.size(1), local_num_experts) == _r2dm.DRAFT_COMPACT_GEOMETRY:  {TAG}\n"
         f"            assert tuple(w2.shape) == (128, 5120, 192)  {TAG}\n"
         f"            assert activation == MoEActivation.SILU  {TAG}\n"
         f"            compact_fc1 = True  {TAG}\n", 1),
        # _act_mul_quant's compact branch asserts DSV41_MOE_COMPACT == "1": the drafter gate requires it (module).
    ],
    DRM32: [
        ("DRAFT_GEOMETRY = (768, 5120, 3, 128)\n", "DRAFT_GEOMETRY = (768, 5120, 3, 128)\n" + PRELUDE, 1),
        ("    return (N, K, topk, experts) == DRAFT_GEOMETRY\n",
         f"    if _R2DM_FC1 and (N, K, topk, experts) == _r2dm.DRAFT_COMPACT_GEOMETRY:  {TAG}\n"
         f"        return True  {TAG}\n"
         "    return (N, K, topk, experts) == DRAFT_GEOMETRY\n", 1),
    ],
    DGUTL: [
        ("_DSV41_FP_FILL_ROWS = 1024\n", "_DSV41_FP_FILL_ROWS = 1024\n" + PRELUDE, 1),
        ("        if return_expert_ends:\n"
         "            return f_out, f_sfp, f_layout, f_inv, align_used, f_ends\n"
         "        return f_out, f_sfp, f_layout, f_inv, align_used\n"
         "\n"
         "    expert_start_loc = torch.empty(\n",
         "        if return_expert_ends:\n"
         "            return f_out, f_sfp, f_layout, f_inv, align_used, f_ends\n"
         "        return f_out, f_sfp, f_layout, f_inv, align_used\n"
         "\n"
         f"    # G17 perm: the same two kernels for the drafter geometry (E 128, top-3, M32 path).  {TAG}\n"
         f"    _r2dm_fp = bool(_R2DM_PERM) and _r2dm.perm_eligible(  {TAG}\n"
         f"        aq, aq_scale, topk_ids, local_num_experts, expert_map, expert_tokens_meta, block_k,  {TAG}\n"
         f"        use_psum_layout, align_used, _DSV41_FP_SLOTS)  {TAG}\n"
         f"    if _r2dm_fp and _R2DM_PERM == \"1\":  {TAG}\n"
         f"        f_out, f_sfp, f_layout, f_inv, f_ends = dsv41_fused_permute(  {TAG}\n"
         f"            aq, aq_scale, topk_ids, local_num_experts, aq_out, M_sum, align_used)  {TAG}\n"
         f"        if return_expert_ends:  {TAG}\n"
         f"            return f_out, f_sfp, f_layout, f_inv, align_used, f_ends  {TAG}\n"
         f"        return f_out, f_sfp, f_layout, f_inv, align_used  {TAG}\n"
         "    expert_start_loc = torch.empty(\n", 1),
        ("    if return_expert_ends:\n"
         "        return aq_out, aq_scale_out, grouped_layout, inv_perm, align_used, expert_start_loc\n",
         f"    if _r2dm_fp and _R2DM_PERM == \"verify\":  {TAG}\n"
         f"        _dsv41_fp_verify(aq, aq_scale, topk_ids, local_num_experts, M_sum, align_used,  {TAG}\n"
         f"                         (aq_out, aq_scale_out, grouped_layout, inv_perm, expert_start_loc))  {TAG}\n"
         "    if return_expert_ends:\n"
         "        return aq_out, aq_scale_out, grouped_layout, inv_perm, align_used, expert_start_loc\n", 1),
    ],
}

# Other kits that may insert tagged lines into the same files (pure insertions, anchors verified to compose):
# r1-pdl (deep_gemm_utils.py) and r2-moefront (deep_gemm_utils.py, deep_gemm_moe.py). A file is an accepted base when
# stripping these tags gives the recorded live sha256.
FOREIGN_TAGS = ("# r1-pdl", "# r2-moefront")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rd(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def strip_tag(src, tag):
    return "".join(ln for ln in src.splitlines(keepends=True) if not ln.rstrip("\r\n").endswith(tag))


def strip_foreign(src):
    for t in FOREIGN_TAGS:
        src = strip_tag(src, t)
    return src


def live_base(rel):
    for f in KIT["files"]:
        if f["path"] == rel:
            return f["base_sha256"]
    raise KeyError(rel)


def accepted(rel, src):
    """src is the recorded live base, or the live base plus tagged insertions of the FOREIGN_TAGS kits only."""
    return TAG not in src and sha_text(strip_foreign(src)) == live_base(rel)


def patched(rel, src):
    for old, new, count in EDITS[rel]:
        n = src.count(old)
        if n != count:
            sys.exit(f"{rel}: anchor found {n}x, expected {count}: {old[:90]!r}")
        src = src.replace(old, new)
    return src


def ours_state(rel, src):
    """For a file carrying TAG: (ok, pre) where pre = src without our lines. ok when pre is an accepted base and src
    equals our patch of pre up to the position of foreign-tagged lines (another kit installed on top of ours may have
    inserted its lines next to our block at a shared anchor)."""
    pre = strip_tag(src, TAG)
    if TAG in pre or not accepted(rel, pre):
        return False, pre
    try:
        mine = patched(rel, pre)
    except SystemExit:
        return False, pre
    return strip_foreign(mine) == strip_foreign(src), pre


def _write(path, text, mode_from):
    tmp = path + ".tmp-r2-draftmoe"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    shutil.copymode(mode_from, tmp)
    os.replace(tmp, path)


def install():
    # Check everything first, then write: never leave a half-installed kit.
    plan = []
    for rel in EDITS:
        path = os.path.join(V, rel)
        src = rd(path)
        if TAG in src:
            ok, _ = ours_state(rel, src)
            if not ok:
                sys.exit(f"REFUSED {rel}: carries {TAG} but is not our patch of an accepted base; nothing was changed")
            plan.append((rel, path, None))
            continue
        if not accepted(rel, src):
            sys.exit(f"REFUSED {rel}: sha256 {sha256(path)} is not a recorded base, nor one with only {FOREIGN_TAGS} "
                     f"lines added (KIT.json); nothing was changed")
        new = patched(rel, src)
        if strip_tag(new, TAG) != src:
            sys.exit(f"REFUSED {rel}: internal: edit is not a pure tagged insertion")
        plan.append((rel, path, new))
    src_mod = os.path.join(HERE, "new", NEWMOD)
    if sha256(src_mod) != KIT["new_module_sha256"]:
        sys.exit(f"REFUSED: kit module {src_mod} sha256 {sha256(src_mod)} != KIT.json new_module_sha256")
    dst = os.path.join(V, NEWMOD)
    tmp = dst + ".tmp-r2-draftmoe"  # module first: no patched file ever points at a missing module
    shutil.copy2(src_mod, tmp)
    os.replace(tmp, dst)
    print(f"{NEWMOD}: {sha256(dst)[:8]} (new)")
    for rel, path, new in plan:
        if new is None:
            print(f"{rel}: already patched ({sha256(path)[:8]})")
            continue
        shutil.copy2(path, path + BAK)  # informational copy of the pre-install file; uninstall never restores it
        _write(path, new, path + BAK)
        print(f"{rel}: patched ({sha256(path + BAK)[:8]} -> {sha256(path)[:8]}), {new.count(TAG)} tagged lines")


def uninstall():
    """Remove exactly our tagged lines (all edits are pure insertions), after checking the file is our patch of an
    accepted base. Works in any order relative to r1-pdl / r2-moefront as long as their own uninstallers are safe;
    LIFO is the documented order. Never restores the .bak copy (it may predate another kit's install/uninstall)."""
    plan = []
    for rel in EDITS:
        path = os.path.join(V, rel)
        src = rd(path)
        if TAG not in src:
            plan.append((rel, path, None))
            continue
        ok, pre = ours_state(rel, src)
        if not ok:
            sys.exit(f"REFUSED {rel}: tagged lines do not match our patch of an accepted base; nothing was changed "
                     f"(inspect by hand; {BAK} holds the file as it was before our install)")
        plan.append((rel, path, pre))
    for rel, path, pre in plan:
        bak = path + BAK
        if pre is None:
            print(f"{rel}: no {TAG} lines ({sha256(path)[:8]}, "
                  f"{'accepted base' if accepted(rel, rd(path)) else 'NOT an accepted base: check by hand'}); left as is")
        else:
            _write(path, pre, path)
            print(f"{rel}: removed our lines ({sha256(path)[:8]}"
                  f"{', = live base' if sha256(path) == live_base(rel) else ', = live base + other kits'})")
        if os.path.exists(bak):
            os.remove(bak)
    dst = os.path.join(V, NEWMOD)
    if os.path.exists(dst):
        os.remove(dst)
        print(f"{NEWMOD}: removed")
    print("pre-JIT cache entries (if any) are left in place: unreachable with the kit off, harmless")


def status():
    for rel in list(EDITS) + [NEWMOD]:
        path = os.path.join(V, rel)
        if not os.path.exists(path):
            print(f"-------- absent   {rel}")
            continue
        if rel == NEWMOD:
            st = "PATCHED" if sha256(path) == KIT["new_module_sha256"] else "STALE  "
        else:
            src = rd(path)
            if TAG in src:
                st = "PATCHED" if ours_state(rel, src)[0] else "FOREIGN-PATCHED (!)"
            else:
                st = "clean  " if accepted(rel, src) else "UNKNOWN-BASE (!)"
        print(f"{sha256(path)[:8]} {st} {rel}")
    prejit_status()


def dry():
    for rel in EDITS:
        src = rd(os.path.join(V, rel))
        if TAG in src:
            print(f"== {rel}: already patched ({'ours' if ours_state(rel, src)[0] else 'NOT OURS'})")
            continue
        base = sha256(os.path.join(V, rel))
        ok = accepted(rel, src)
        print(f"== {rel}: base {base[:8]} {'OK' if ok else 'NOT A RECORDED BASE (install would refuse)'}")
        for i, line in enumerate(patched(rel, src).splitlines(), 1):
            if TAG in line:
                print(f"{i:5d}: {line}")


# ------------------------------------------------------------------------------------------------ pre-JIT
JIT_TAG = "sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d"


def _prejit_entries():
    caches = {
        "deal": (os.path.join(HOME, "dsv41-stage/g130/fc1/jit/deal-base/cache"),
                 os.path.join(HOME, "dsv41-stage/g130/fc1/trees/deal-base/include")),
        "dg": (os.path.join(HOME, ".cache/vllm/deep_gemm/cache"), os.path.join(V, "third_party/deep_gemm/include")),
    }
    out = []
    for name, e in KIT["prejit"].items():
        cache, include = caches[e["cache"]]
        out.append(dict(e, name=name, cache=cache, include=include))
    return out


def _vllm_running():
    """True when a vLLM/serving process is alive on this node (a DeepGEMM JIT could be writing the same cache)."""
    r = subprocess.run(["pgrep", "-f", "vllm.entrypoints|vllm serve|VLLM::|dsv41-rank"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _djkey():
    import importlib.util

    spec = importlib.util.spec_from_file_location("r2dm_djkey", os.path.join(HERE, "tools", "djkey.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flags(meta):
    o = meta["compiler_options"]
    assert o["arch"] == "120f" and o["optimize_level"] == "3" and not o["use_fast_math"]
    assert o["ptxas_register_usage_level"] == 10 and not o["ptxas_verbose"] and not o["with_line_info"]
    assert o["post_hook"] is None
    return (["--gpu-architecture=sm_%s" % o["arch"], "-O3", "--compiler-options=-O3",
             "--ptxas-options=--register-usage-level=10"] + o["nvcc_flags"] + o["extra_nvcc_flags"])


def prejit_status():
    for e in _prejit_entries():
        out = os.path.join(e["cache"], f"{JIT_TAG}.{e['new_key']}")
        state = "absent"
        if os.path.isfile(os.path.join(out, ".committed")):
            state = "present " + (sha256(os.path.join(out, "kernel.cubin")) or "?")[:8]
        print(f"prejit {e['name']}: {state} {out}")


def prejit(force=False):
    dj = _djkey()
    nvcc = "/usr/local/cuda/bin/nvcc"
    for e in _prejit_entries():
        src_dir = os.path.join(e["cache"], f"{JIT_TAG}.{e['src_key']}")
        out = os.path.join(e["cache"], f"{JIT_TAG}.{e['new_key']}")
        if os.path.exists(out) and not force:
            # committed = done; present but uncommitted = a DeepGEMM JIT may be writing it right now: never touch it
            print(f"{'HIT' if os.path.isfile(os.path.join(out, '.committed')) else 'SKIP (uncommitted, JIT in progress?)'}"
                  f" {e['name']} {out}")
            continue
        if force and os.path.exists(out) and _vllm_running():
            sys.exit(f"REFUSED {e['name']}: --force would replace {out} while a vLLM process runs on this node")
        kcu = os.path.join(src_dir, "kernel.cu")
        if sha256(kcu) != e["src_kernel_cu_sha256"]:
            sys.exit(f"REFUSED {e['name']}: source entry {kcu} sha {sha256(kcu)} != {e['src_kernel_cu_sha256']}")
        meta = json.load(open(os.path.join(src_dir, "meta.json")))
        src = rd(kcu)
        old, new = "        0, 768, 5120,\n", "        0, 576, 5120,\n"
        if src.count(old) != 1:
            sys.exit(f"REFUSED {e['name']}: N anchor not found once in {kcu}")
        code = src.replace(old, new)
        if hashlib.sha256(code.encode()).hexdigest() != e["new_kernel_cu_sha256"]:
            sys.exit(f"REFUSED {e['name']}: generated kernel.cu sha differs from KIT.json")
        flags = _flags(meta)
        version = meta["compiler_info"]["version"]
        got = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
        if got != version:
            sys.exit(f"REFUSED {e['name']}: nvcc --version differs from the source entry's compiler")
        key = dj.key(e["include"], code.encode(), version, flags)
        if key != e["new_key"]:
            sys.exit(f"REFUSED {e['name']}: cache key {key} != {e['new_key']} (include tree changed?)")
        avail = int([l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemAvailable:")][0])
        if avail < 7500000:
            sys.exit(f"REFUSED {e['name']}: MemAvailable {avail} kB < 7500000 kB")
        tmp = tempfile.mkdtemp(prefix=f"{JIT_TAG}.{key}.", dir=os.path.dirname(e["cache"]))
        os.chmod(tmp, 0o755)
        open(os.path.join(tmp, "kernel.cu"), "w").write(code)
        cmd = [nvcc, os.path.join(tmp, "kernel.cu"), "-o", os.path.join(tmp, "kernel.cubin"), "--cubin"] + flags + \
              ["--include-path", e["include"]]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""))
        print(f"BUILD {e['name']} key {key} (nice 19, timeout 1500 s)", flush=True)
        r = subprocess.run(["timeout", "-k", "10", "1500", "nice", "-n", "19"] + cmd, env=env)
        if r.returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            sys.exit(f"FAIL {e['name']} nvcc rc={r.returncode}")
        meta["command"] = " ".join([nvcc, f"{out}/kernel.cu", "--output-file", f"{out}/kernel.cubin", "--cubin"]
                                   + flags + ["--include-path", e["include"]])
        meta["r2_draftmoe"] = "pre-JIT by r2-draftmoe install.py prejit (offline nvcc, same flags/compiler/include as JIT)"
        json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"))
        open(os.path.join(tmp, ".committed"), "w").close()
        if os.path.exists(out):  # only with --force and no vLLM process: move the old entry aside, never rmtree it
            os.rename(out, f"{out}.old-r2dm-{int(time.time())}")
        try:
            os.rename(tmp, out)
        except OSError:  # someone (a JIT) committed it meanwhile: keep theirs
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"RACE {e['name']}: {out} appeared during the build; kept the existing entry")
            continue
        cubin = sha256(os.path.join(out, "kernel.cubin"))
        want = e.get("cubin_sha256")
        print(f"BUILT {e['name']} {out} cubin {cubin}" + ("" if not want else (" == KIT.json" if cubin == want
                                                                               else f" (KIT.json has {want})")))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"install": install, "uninstall": uninstall, "status": status, "dry": dry,
          "prejit-status": prejit_status}.get(cmd)
    if cmd == "prejit":
        prejit(force="--force" in sys.argv[2:])
    elif fn is None:
        sys.exit(__doc__)
    else:
        fn()
