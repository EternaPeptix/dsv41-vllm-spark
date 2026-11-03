#!/usr/bin/env python3
"""apply_l2win.py <base> <out>: anchored patch of dsv41_l2_prefetch.py for g130 l2win. Refuses any base other than sha 63737f9c and any
anchor that is not found exactly once. Every new behaviour sits behind a flag that defaults off; with all of them unset G2 stays False and
none of the new lines run, apart from reading four env vars at import. Standard library only."""
import hashlib
import sys

BASE_SHA = "63737f9c413ebfa2c408b13cb0cbecddac2b42c6a5e219e6bad43bde095f4bee"

EDITS = [
    # 1. module doc
    ("  DSV41_L2_PREFETCH_PACE_LIB      ~/rdma_ar_dsv41/build-g2ct/libdsv41_l2pf_g2.so (missing -> burst, logged)\n",
     "  DSV41_L2_PREFETCH_PACE_LIB      ~/rdma_ar_dsv41/build-g2ct/libdsv41_l2pf_g2.so (missing -> burst, logged)\n"
     "  g130 l2win (all default off; cache hints only):\n"
     "  DSV41_L2_PREFETCH_PACE_MAX_WIN_KB  forks with more bytes than this are not paced at PACE_GBPS (0 = no limit)\n"
     "  DSV41_L2_PREFETCH_PACE_BIG_GBPS    those larger forks are paced at this rate instead of staying burst (0 = burst)\n"
     "  DSV41_L2_PREFETCH_BURST_DELAY_US   chain dsv41_spin(this long, 1 CTA) before the burst on the side stream (0 = off)\n"
     "  DSV41_L2_PREFETCH_BURST_DELAY_KINDS fork kinds the delay applies to (ar)\n"),
    # 2. env
    ('PACE_CHUNK = max(16, int(_env("DSV41_L2_PREFETCH_PACE_CHUNK", "16384")) & ~15)\n',
     'PACE_CHUNK = max(16, int(_env("DSV41_L2_PREFETCH_PACE_CHUNK", "16384")) & ~15)\n'
     '# g130 l2win (default off; see module doc)\n'
     'PACE_MAX_WIN = int(float(_env("DSV41_L2_PREFETCH_PACE_MAX_WIN_KB", "0") or "0") * 1024)\n'
     'PACE_BIG_GBPS = float(_env("DSV41_L2_PREFETCH_PACE_BIG_GBPS", "0") or "0")\n'
     'BURST_DELAY_NS = int(float(_env("DSV41_L2_PREFETCH_BURST_DELAY_US", "0") or "0") * 1000)\n'
     'BURST_DELAY_KINDS = frozenset(k.strip() for k in _env("DSV41_L2_PREFETCH_BURST_DELAY_KINDS", "ar").split(",")\n'
     '                              if k.strip())\n'),
    # 3. gate
    ("G2 = PACE_GBPS > 0 or bool(REPLAN)\n",
     "G2 = PACE_GBPS > 0 or bool(REPLAN) or BURST_DELAY_NS > 0\n"),
    # 4. init log
    ('            ",".join(sorted(REPLAN)) or "-", REPLAN_BUDGET / 2**20,\n        )\n',
     '            ",".join(sorted(REPLAN)) or "-", REPLAN_BUDGET / 2**20,\n        )\n'
     '        if PACE_MAX_WIN > 0 or BURST_DELAY_NS > 0:\n'
     '            logger.info("DSV41_L2_PREFETCH l2win: pace_max_win=%d KB pace_big=%g GB/s burst_delay=%.1f us kinds=%s",\n'
     '                        PACE_MAX_WIN // 1024, PACE_BIG_GBPS, BURST_DELAY_NS / 1e3, ",".join(sorted(BURST_DELAY_KINDS)))\n'),
    # 5. _parts: size gate
    ("    pace = PACE_GBPS > 0 and S.pace_ok and fork_kind in PACE_KINDS\n    if pace and PACE_HEAD > 0:\n",
     "    pace = PACE_GBPS > 0 and S.pace_ok and fork_kind in PACE_KINDS\n"
     '    pmode = "p"\n'
     "    if pace and PACE_MAX_WIN > 0 and sum(tk for _p, tk, _n in segs) > PACE_MAX_WIN:\n"
     "        pace = PACE_BIG_GBPS > 0\n"
     '        pmode = "q"  # paced at PACE_BIG_GBPS\n'
     "    if pace and PACE_HEAD > 0:\n"),
    # 6. _parts: mode tags
    ('    for mode, ss in (("b", head), ("p", tail)):\n',
     '    bmode = "d" if BURST_DELAY_NS > 0 and fork_kind in BURST_DELAY_KINDS else "b"\n'
     '    for mode, ss in ((bmode, head), (pmode, tail)):\n'),
    # 7. plan key
    ("            PACE_GBPS if S.pace_ok else 0.0, PACE_HEAD, tuple(sorted(PACE_KINDS)))\n",
     "            PACE_GBPS if S.pace_ok else 0.0, PACE_HEAD, tuple(sorted(PACE_KINDS)),\n"
     "            PACE_MAX_WIN, PACE_BIG_GBPS, BURST_DELAY_NS, tuple(sorted(BURST_DELAY_KINDS)))\n"),
    # 8. _fork_part: paced rate
    ('    if mode == "p" and S.pace_ok:\n'
     "        side = _side_stream3()\n"
     "        side.wait_stream(main)\n"
     "        period_ps = int(round(PACE_CHUNK * 1000.0 / PACE_GBPS))\n",
     '    if mode in ("p", "q") and S.pace_ok:\n'
     "        side = _side_stream3()\n"
     "        side.wait_stream(main)\n"
     '        period_ps = int(round(PACE_CHUNK * 1000.0 / (PACE_BIG_GBPS if mode == "q" else PACE_GBPS)))\n'),
    # 9. _fork_part: delayed burst
    ('        side = _side_stream2() if where == "attn" else _side_stream()\n'
     "        side.wait_stream(main)\n",
     '        side = _side_stream2() if where == "attn" else _side_stream()\n'
     "        side.wait_stream(main)\n"
     '        if mode == "d":\n'
     "            rc = _load_lib().dsv41_spin(BURST_DELAY_NS, 1, side.cuda_stream)\n"
     "            if rc:\n"
     '                raise RuntimeError(f"DSV41_L2_PREFETCH_BURST_DELAY: spin launch failed (cuda error {rc})")\n'),
]


def apply(text: str) -> str:
    for old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"anchor found {text.count(old)} times, want 1: {old[:70]!r}")
        text = text.replace(old, new)
    return text


def main() -> None:
    base, out = sys.argv[1], sys.argv[2]
    raw = open(base, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != BASE_SHA:
        raise SystemExit(f"refusing: base sha {got[:16]} is not {BASE_SHA[:16]}")
    new = apply(raw.decode()).encode()
    with open(out, "wb") as f:
        f.write(new)
    print("patched", out, hashlib.sha256(new).hexdigest())


if __name__ == "__main__":
    main()
