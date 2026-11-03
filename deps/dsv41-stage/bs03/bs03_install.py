#!/usr/bin/env python3
"""bs03_install.py plan|status|install|uninstall [--site DIR] (ON each Spark, python3 -I, stdlib only)

BS03 Design B hook for DSV41_BS03_DET_TOPK=1. Inert while the flag is unset.

plan      : default. Checks every base sha and anchor, prints the patch. Writes nothing. Safe while serving.
status    : sha256[:8] of the three base files, the backup and the module. Writes nothing. Safe while serving.
install   : service stopped only. model_executor/layers/sparse_attn_indexer.py 005785f9 -> patched (backup .bak-bs03),
            + layers/dsv41_bs03_det_topk.py from this kit. Refuses on any other base sha (sparse_attn_indexer.py 005785f9,
            dsv41_topk_ties.py 943961bc, indexer_topk.py 4dc91f83), if a vLLM process runs, or if a kit file fails SHA256SUMS.
uninstall : service stopped only. Restores sparse_attn_indexer.py from .bak-bs03 and removes the module.

--site DIR works on a copy of the vllm package directory (how the kit was tested), and skips the running-process check.
All 8 nodes must hold the same state: run it on each, then compare `status` on 8/8.
"""
import difflib
import hashlib
import os
import shutil
import subprocess
import sys

DEFAULT_SITE = os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
KIT = os.path.dirname(os.path.abspath(__file__))
REL_SAI = "model_executor/layers/sparse_attn_indexer.py"
REL_TIES = "model_executor/layers/dsv41_topk_ties.py"
REL_TOPK = "model_executor/layers/indexer_topk.py"
REL_MOD = "model_executor/layers/dsv41_bs03_det_topk.py"
BAK_SUFFIX = ".bak-bs03"
BASE = {
    REL_SAI: "005785f9c352961b79ed6a4af5628a4b8d82a9168afe0d06a11670c540ea5113",
    REL_TIES: "943961bcc1b76613cc65b3a99c72cb94e790e6f5d71a0aefa830ecd454aedc99",
    REL_TOPK: "4dc91f83043c659304e0f0808bd9082574f919361b0b12f7c87fef5e261469bb",
}
PATCHED_SAI_SHA = "ebe2ba62ecb2b261667eb7899e8b230719d41fca5c489fd95a6244bb75a8bac0"

A_IMPORT = """from vllm.model_executor.layers import dsv41_topk_ties as _g5det
"""
B_IMPORT = A_IMPORT + """# BS03 (2026-11-02): one canonical top-512 kernel replaces top-k + TIES + CANON on decode rows; inert unless
# DSV41_BS03_DET_TOPK=1 (the module is not even imported otherwise; see dsv41_bs03_det_topk.py).
if os.environ.get("DSV41_BS03_DET_TOPK", "0") == "1":
    from vllm.model_executor.layers import dsv41_bs03_det_topk as _bs03
else:
    _bs03 = None
"""

A_DECODE = """        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # The backend comes from the layer (config is only readable at model
        # construction); dispatchers are cached per backend.
        get_indexer_topk(topk_backend)(
            logits,
            seq_lens,
            next_n,
            topk_indices,
            topk_tokens,
            attn_metadata_narrowed.max_seq_len,
        )
        if _g5det.ENABLED:
            _g5det.decode_hook(
"""
B_DECODE = """        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # BS03: True = the canonical kernel wrote topk_indices (top-k + TIES + CANON in one launch). False, always so
        # with the flag unset, = the chain below runs exactly as before. try_decode refuses DCP, so the merge is unaffected.
        bs03_done = _bs03 is not None and _bs03.try_decode(
            logits,
            seq_lens,
            next_n,
            topk_indices,
            topk_tokens,
            topk_backend,
            decode_metadata.global_seq_lens,
        )
        # The backend comes from the layer (config is only readable at model
        # construction); dispatchers are cached per backend.
        if not bs03_done:
            get_indexer_topk(topk_backend)(
                logits,
                seq_lens,
                next_n,
                topk_indices,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        if _g5det.ENABLED and not bs03_done:
            _g5det.decode_hook(
"""

A_CANON = """                cp_kv_cache_interleave_size,
            )
        if _DSV41_TOPK_CANON:
            _canon_topk_(topk_indices)

        if needs_padded_path:
"""
B_CANON = """                cp_kv_cache_interleave_size,
            )
        if _DSV41_TOPK_CANON and not bs03_done:
            _canon_topk_(topk_indices)

        if needs_padded_path:
"""
HUNKS = ((A_IMPORT, B_IMPORT), (A_DECODE, B_DECODE), (A_CANON, B_CANON))


def sha(p):
    if not os.path.exists(p):
        return "absent"
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def die(msg):
    sys.exit("refusing: " + msg)


def patched_text(s):
    for a, b in HUNKS:
        n = s.count(a)
        if n != 1:
            die("anchor found %d times:\n%s" % (n, a[:300]))
        s = s.replace(a, b)
    return s


def check_bases(site):
    for rel, want in BASE.items():
        got = sha(os.path.join(site, rel))
        if got != want:
            die("%s is %s, expected %s" % (rel, got[:8], want[:8]))


def check_kit():
    sums = os.path.join(KIT, "SHA256SUMS")
    if not os.path.exists(sums):
        die("no %s" % sums)
    need = {"site/dsv41_bs03_det_topk.py", "build/vllm_bs03_det_topk_sm121.so"}
    with open(sums) as f:
        for line in f:
            want, name = line.split()
            name = name.lstrip("*").lstrip("./")
            if sha(os.path.join(KIT, name)) != want:
                die("kit file %s fails SHA256SUMS" % name)
            need.discard(name)
    if need:
        die("SHA256SUMS does not list %s" % sorted(need))


def check_stopped():
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "[v]llm serve|[V]LLM::"], stdout=subprocess.PIPE, text=True)
    if r.returncode == 0:
        die("vLLM processes are running (pids %s); site files change only with the service stopped" % r.stdout.split())


def status(site):
    sai = os.path.join(site, REL_SAI)
    print("%s sparse_attn_indexer.py %s bak-bs03 %s dsv41_topk_ties.py %s indexer_topk.py %s dsv41_bs03_det_topk.py %s" % (
        os.uname().nodename, sha(sai)[:8], sha(sai + BAK_SUFFIX)[:8], sha(os.path.join(site, REL_TIES))[:8],
        sha(os.path.join(site, REL_TOPK))[:8], sha(os.path.join(site, REL_MOD))[:8]))


def plan(site):
    check_bases(site)
    sai = os.path.join(site, REL_SAI)
    with open(sai) as f:
        old = f.read()
    new = patched_text(old)
    got = hashlib.sha256(new.encode()).hexdigest()
    print("base shas OK; 3 anchors found once each; patched sparse_attn_indexer.py would be %s (expected %s)" % (got[:8], PATCHED_SAI_SHA[:8]))
    if got != PATCHED_SAI_SHA:
        die("patched sha %s differs from the reviewed %s" % (got, PATCHED_SAI_SHA))
    print("would back up to %s%s and install %s" % (REL_SAI, BAK_SUFFIX, REL_MOD))
    sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "a/" + REL_SAI, "b/" + REL_SAI))
    status(site)


def install(site, is_live):
    sai = os.path.join(site, REL_SAI)
    with open(sai) as f:
        old = f.read()
    if "_bs03" in old:
        print("already installed")
        status(site)
        return
    if is_live:
        check_stopped()
    check_kit()
    check_bases(site)
    new = patched_text(old)
    if hashlib.sha256(new.encode()).hexdigest() != PATCHED_SAI_SHA:
        die("patched sha differs from the reviewed %s" % PATCHED_SAI_SHA)
    bak = sai + BAK_SUFFIX
    if os.path.exists(bak) and sha(bak) != BASE[REL_SAI]:
        die("backup %s exists with sha %s" % (bak, sha(bak)[:8]))
    if not os.path.exists(bak):
        shutil.copy2(sai, bak)
    shutil.copy2(os.path.join(KIT, "site", "dsv41_bs03_det_topk.py"), os.path.join(site, REL_MOD))
    tmp = sai + ".bs03tmp"
    with open(tmp, "w") as f:
        f.write(new)
    shutil.copystat(sai, tmp)
    os.replace(tmp, sai)
    status(site)


def uninstall(site, is_live):
    if is_live:
        check_stopped()
    sai = os.path.join(site, REL_SAI)
    bak = sai + BAK_SUFFIX
    if not os.path.exists(bak):
        die("no backup %s" % bak)
    if sha(bak) != BASE[REL_SAI]:
        die("backup sha %s != %s" % (sha(bak)[:8], BASE[REL_SAI][:8]))
    tmp = sai + ".bs03tmp"
    shutil.copy2(bak, tmp)
    os.replace(tmp, sai)
    mod = os.path.join(site, REL_MOD)
    if os.path.exists(mod):
        os.remove(mod)
    status(site)


def main():
    args = sys.argv[1:]
    site = DEFAULT_SITE
    is_live = True
    if "--site" in args:
        i = args.index("--site")
        site = args[i + 1]
        del args[i : i + 2]
        is_live = False
    cmd = args[0] if args else "plan"
    if cmd == "plan":
        plan(site)
    elif cmd == "status":
        status(site)
    elif cmd == "install":
        install(site, is_live)
    elif cmd == "uninstall":
        uninstall(site, is_live)
    else:
        sys.exit(__doc__)


main()
