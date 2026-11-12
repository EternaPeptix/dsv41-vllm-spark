#!/usr/bin/env python3
"""r4-xsretime (climb round 4): install / uninstall / check the default-off attention-window re-timing levers.

    python3 install.py install | uninstall [--force] | status | dry

Edits vllm/distributed/device_communicators/dsv41_l2_prefetch.py (6 anchors, every inserted line tagged
'# r4-xsretime', backup .bak-r4-xsretime) and copies dsv41_l2pf_xsr.py next to it. The base is the LIVE file of
launcher b22e4aa9 (sha256 8bc9a1b5..., i.e. r1-pf installed on 63737f9c); refuses when the live file's sha256 differs
from KIT.json. With every master flag (DSV41_L2_PREFETCH_PACE_GBPS_ATTN, DSV41_L2_PREFETCH_PACE_ATTN_STREAM,
DSV41_L2_PREFETCH_ATTN_DEDUP) unset the new module is never imported (_XSR stays None) and every inserted branch is
dead: same kernels, streams, events, graph contents. Uninstall this kit BEFORE uninstalling r1-pf.
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("XSR_VLLM") or os.path.expanduser("~/venvs/vllm032/lib/python3.12/site-packages/vllm")
REL_BASE = "distributed/device_communicators/dsv41_l2_prefetch.py"
REL_MOD = "distributed/device_communicators/dsv41_l2pf_xsr.py"
BASE = f"{V}/{REL_BASE}"
MOD_DST = f"{V}/{REL_MOD}"
MOD_SRC = os.path.join(HERE, "files", "vllm", REL_MOD)
TAG = "# r4-xsretime"
BAK = BASE + ".bak-r4-xsretime"

MASTER = ("DSV41_L2_PREFETCH_PACE_GBPS_ATTN", "DSV41_L2_PREFETCH_PACE_ATTN_STREAM", "DSV41_L2_PREFETCH_ATTN_DEDUP")


def _kit_sha(rel):
    kit = json.load(open(os.path.join(HERE, "KIT.json")))
    for f in kit["files"]:
        if f["path"] == rel:
            return f
    raise SystemExit(f"KIT.json has no entry for {rel}")


EDITS = [
    # 1. install hook (first collective, outside capture): create the own attn pacer stream
    ("        _R1.install(mod)  # r1-pf\n"
     "    S.installed = True\n",
     "        _R1.install(mod)  # r1-pf\n"
     f"    if _XSR is not None:  {TAG}\n"
     f"        _XSR.install()  {TAG}\n"
     "    S.installed = True\n", 1),
    # 2. end-of-forward join of the own attn pacer stream (after the base joins, before _finish)
    ("        _join3()\n"
     "        try:\n"
     "            _finish(sig, mode)\n",
     "        _join3()\n"
     f"        if _XSR is not None:  {TAG}\n"
     f"            _XSR.join()  {TAG}\n"
     "        try:\n"
     "            _finish(sig, mode)\n", 1),
    # 3. target g2 plan: attention-window dedup against the owner collective window (pure list edit)
    ("    fl = g2_fork_lists(events, sel)\n"
     "    kinds = fl[\"kind\"]\n",
     "    fl = g2_fork_lists(events, sel)\n"
     f"    if _XSR is not None:  {TAG}\n"
     f"        fl = _XSR.retime_lists(fl, events)  {TAG}\n"
     "    kinds = fl[\"kind\"]\n", 1),
    # 4. attention parts: paced tail as a "y" part (attn rate / own stream); None = base path unchanged
    ("    pace = PACE_GBPS > 0 and S.pace_ok and fork_kind in PACE_KINDS\n",
     f"    if _XSR is not None and fork_kind == \"attn\":  {TAG}\n"
     f"        _xp = _XSR.attn_parts(segs, device)  {TAG}\n"
     f"        if _xp is not None:  {TAG}\n"
     f"            return _xp  {TAG}\n"
     "    pace = PACE_GBPS > 0 and S.pace_ok and fork_kind in PACE_KINDS\n", 1),
    # 5. fork dispatch for "y" parts; must precede the 4-tuple unpack
    ("        return _R1.fork_part(part, where)  # r1-pf\n"
     "    mode, t, nseg, _nb = part\n",
     "        return _R1.fork_part(part, where)  # r1-pf\n"
     f"    if _XSR is not None and part[0] == \"y\":  {TAG}\n"
     f"        return _XSR.fork_part(part, where)  {TAG}\n"
     "    mode, t, nseg, _nb = part\n", 1),
    # 6. gate + import at the end of the module (after the r1-pf gate; all base names exist by then)
    ("        logger.warning(\"DSV41_L2_PREFETCH_R1: disabled (%r)\", _e)  # r1-pf\n",
     "        logger.warning(\"DSV41_L2_PREFETCH_R1: disabled (%r)\", _e)  # r1-pf\n"
     f"# climb r4 (r4-xsretime): attention-window re-timing in dsv41_l2pf_xsr.py, imported only when a master flag is set  {TAG}\n"
     f"_XSR = None  {TAG}\n"
     f"_XSR_FLAGS = {MASTER!r}  {TAG}\n"
     f"if ENABLED and any(_env(_k, \"0\").lower() not in (\"\", \"0\", \"off\", \"false\") for _k in _XSR_FLAGS):  {TAG}\n"
     f"    try:  {TAG}\n"
     f"        from vllm.distributed.device_communicators import dsv41_l2pf_xsr as _XSR  {TAG}\n"
     f"        _XSR.bind(sys.modules[__name__])  {TAG}\n"
     f"        if not _XSR.any_active():  {TAG}\n"
     f"            _XSR = None  {TAG}\n"
     f"    except Exception as _e:  # noqa: BLE001  {TAG}\n"
     f"        _XSR = None  {TAG}\n"
     f"        logger.warning(\"DSV41_L2_PREFETCH r4-xsretime: disabled (%r)\", _e)  {TAG}\n", 1),
]


def sha(p, n=64):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:n] if os.path.exists(p) else "-"


def patched(src, where=BASE):
    for old, new, count in EDITS:
        n = src.count(old)
        if n != count:
            sys.exit(f"{where}: anchor found {n}x, expected {count}: {old[:80]!r}")
        src = src.replace(old, new)
    if not src.endswith(f"{TAG}\n"):
        sys.exit(f"{where}: the end-of-module anchor is not the last line; refusing")
    return src


def _check_base(src_path):
    want = _kit_sha(REL_BASE)["base_sha256"]
    got = sha(src_path)
    if got != want:
        sys.exit(f"{src_path}: sha256 {got[:16]} != KIT.json base {want[:16]}; refusing (rebase the kit)")


def install():
    want = _kit_sha(REL_MOD)
    if sha(MOD_SRC) != want.get("kit_sha256"):  # checked before anything is written
        sys.exit(f"{MOD_SRC}: sha differs from KIT.json kit_sha256; refusing")
    src = open(BASE).read()
    if TAG in src:
        if not os.path.exists(BAK):
            sys.exit(f"{BASE}: tagged but no {BAK}; refusing")
        _check_base(BAK)
        if src != patched(open(BAK).read()):
            sys.exit(f"{BASE}: tagged but differs from patch(backup); refusing")
        print(f"{BASE}: already patched ({sha(BASE, 8)})")
    else:
        _check_base(BASE)
        new = patched(src)
        if not os.path.exists(BAK):
            shutil.copy2(BASE, BAK)
        tmp = BASE + ".tmp-r4-xsretime"
        open(tmp, "w").write(new)
        shutil.copymode(BAK, tmp)
        os.replace(tmp, BASE)
        print(f"{BASE}: patched ({sha(BAK, 8)} -> {sha(BASE, 8)}), {new.count(TAG)} tagged lines")
    shutil.copy2(MOD_SRC, MOD_DST)
    print(f"{MOD_DST}: {sha(MOD_DST, 8)}")
    exp = _kit_sha(REL_BASE).get("patched_sha256")
    if exp and sha(BASE) != exp:
        sys.exit(f"{BASE}: patched sha {sha(BASE, 16)} != KIT.json patched_sha256 {exp[:16]}")


def _force():
    return "--force" in sys.argv[2:]


def uninstall():
    """Restore the backup only when the live file is exactly what this kit installed (patched_sha256, or
    patch(backup)); remove the module only when it is exactly the kit's. Both are checked before anything is
    touched. --force overrides both checks."""
    want_p = _kit_sha(REL_BASE).get("patched_sha256")
    want_m = _kit_sha(REL_MOD).get("kit_sha256")
    errs = []
    restore = drop_bak = False
    if os.path.exists(BAK):
        cur = open(BASE).read() if os.path.exists(BASE) else None
        try:
            exp = patched(open(BAK).read(), BAK)
        except SystemExit:  # backup does not take the anchors: only the sha check can match
            exp = None
        if cur is not None and sha(BASE) == sha(BAK):
            drop_bak = True  # already the backup's bytes (e.g. restored by hand): just drop the backup
        elif cur is not None and (sha(BASE) == want_p or cur == exp):
            restore = True
        else:
            restore = True
            errs.append(f"{BASE}: sha {sha(BASE, 16)} is neither KIT.json patched_sha256 {str(want_p)[:16]} nor "
                        f"patch(backup); it was edited after install (restoring {BAK} would discard those edits)")
    elif os.path.exists(BASE) and TAG in open(BASE).read():
        sys.exit(f"{BASE}: tagged but no backup; refusing")
    if os.path.exists(MOD_DST) and sha(MOD_DST) != want_m:
        errs.append(f"{MOD_DST}: sha {sha(MOD_DST, 16)} != KIT.json kit_sha256 {str(want_m)[:16]}")
    if errs and not _force():
        sys.exit("refusing to uninstall (nothing changed; rerun with --force to override):\n  " + "\n  ".join(errs))
    if restore:
        os.replace(BAK, BASE)
        print(f"{BASE}: restored {sha(BASE, 8)}" + (" (--force)" if errs else ""))
    elif drop_bak:
        os.remove(BAK)
        print(f"{BASE}: already {sha(BASE, 8)}; removed {BAK}")
    if os.path.exists(MOD_DST):
        os.remove(MOD_DST)
        print(f"{MOD_DST}: removed" + (" (--force)" if errs else ""))


def status():
    """One line per file plus a verdict line; exit 0 when the node is consistently clean or consistently
    installed (base == patched_sha256 AND module == kit_sha256 AND backup == base_sha256), 2 otherwise."""
    fb, fm = _kit_sha(REL_BASE), _kit_sha(REL_MOD)
    sb, sm, sk = sha(BASE), sha(MOD_DST), sha(BAK)
    tagged = os.path.exists(BASE) and TAG in open(BASE).read()
    print(f"{sb[:8]} {'PATCHED' if tagged else 'clean  '} {BASE}"
          f"  [{'= patched_sha256' if sb == fb.get('patched_sha256') else '= base_sha256' if sb == fb['base_sha256'] else 'UNKNOWN sha'}]")
    print(f"{sm[:8]} {'present' if os.path.exists(MOD_DST) else 'absent '} {MOD_DST}"
          f"  [{'= kit_sha256' if sm == fm.get('kit_sha256') else '-' if sm == '-' else 'UNKNOWN sha'}]")
    if os.path.exists(BAK):
        print(f"{sk[:8]} backup  {BAK}  [{'= base_sha256' if sk == fb['base_sha256'] else 'UNKNOWN sha'}]")
    installed = sb == fb.get("patched_sha256") and sm == fm.get("kit_sha256") and sk == fb["base_sha256"]
    clean = sb == fb["base_sha256"] and sm == "-" and sk == "-"
    verdict = "INSTALLED" if installed else "CLEAN" if clean else "INCONSISTENT"
    print(f"r4-xsretime status: {verdict} (base {sb[:16]} module {sm[:16]})")
    if verdict == "INCONSISTENT":
        sys.exit(2)


def dry():
    src_path = BAK if TAG in open(BASE).read() else BASE
    _check_base(src_path)
    for i, line in enumerate(patched(open(src_path).read()).splitlines(), 1):
        if TAG in line:
            print(f"{i:5d}: {line}")


if __name__ == "__main__":
    {"install": install, "uninstall": uninstall, "status": status, "dry": dry}[sys.argv[1]]()
