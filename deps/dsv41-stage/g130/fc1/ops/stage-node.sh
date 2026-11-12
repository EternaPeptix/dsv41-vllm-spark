#!/bin/bash
# g130 fc1 node staging: rebuild the arm trees from this node's pinned m32-bw all-base (no compile) and verify them and the pre-JIT
# caches. Disk only, nice 19, safe while serving, but not while a relaunch is running (MemAvailable at startup).
# Usage: cd ~/dsv41-stage/g130/fc1 && bash ops/stage-node.sh
set -u
K=$HOME/dsv41-stage/g130/fc1
SRC=$HOME/dsv41-stage/g1/m32-bw/trees/all-base
cd "$K" || exit 1
fail() { echo "STAGE $(hostname) FAIL $*"; exit 1; }
sha256sum -c --quiet SHA256SUMS || fail "SHA256SUMS"
[ "$(sha256sum "$SRC/manifest.json" | cut -c1-64)" = 74d0d4f2a09247a668f876d4d5e0993afa9e93462f4d7f6da70276738a218f5c ] || fail "all-base manifest"
mkdir -p trees
for t in $(python3 -I -c 'import json; print(" ".join(sorted(json.load(open("MANIFESTS.json")))))'); do
  want=$(python3 -I -c "import json; print(json.load(open('MANIFESTS.json'))['$t'])")
  if [ -f "trees/$t/manifest.json" ] && [ "$(sha256sum "trees/$t/manifest.json" | cut -c1-64)" = "$want" ]; then continue; fi
  rm -rf "trees/$t"
  nice -n 19 python3 -I tools/make_deal_tree.py "${t%-base}" "$SRC" "trees/$t" header/dsv41_fc1_deal.cuh > /dev/null || fail "make $t"
  [ "$(sha256sum "trees/$t/manifest.json" | cut -c1-64)" = "$want" ] || fail "tree $t manifest"
done
python3 -I - <<'PY' || fail "tree file shas"
import hashlib, json, pathlib
for t in json.load(open("MANIFESTS.json")):
    r = pathlib.Path("trees") / t
    m = json.loads((r / "manifest.json").read_text())
    for rel, h in m["files"].items():
        assert hashlib.sha256((r / rel).read_bytes()).hexdigest() == h, (t, rel)
PY
( cd jit && sha256sum -c --quiet ../jit.sha256 ) || fail "jit caches"
d=$(cat MANIFESTS.json jit.sha256 SHA256SUMS | sha256sum | cut -c1-16)
echo "$d $(date -u +%FT%TZ)" > PREP.ok
echo "staged OK $(hostname) digest $d trees $(ls trees | wc -l) jit $(find jit -name .committed | wc -l)"
