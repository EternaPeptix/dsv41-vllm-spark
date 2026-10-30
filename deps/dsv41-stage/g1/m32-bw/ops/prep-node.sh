#!/bin/bash
# g1/m32-bw node prep (CPU only, niced; safe while serving): build the variant trees from the immutable live trees with
# tools/make_tree.py, require the pinned manifest shas (MANIFESTS.json), verify the pre-JIT caches (jit.sha256), the
# staged site file and the staged file list (FILES), then write PREP.ok.  Usage: bash ops/prep-node.sh
set -u
G=$HOME/dsv41-stage/g1/m32-bw
cd "$G" || exit 1
BASE=$HOME/dsv41-stage/m32/target-m32-native-v1
D05=$HOME/dsv41-goal120-20261024/d05-shadow-v1/native
fail() { echo "PREP $(hostname) FAIL $*"; exit 1; }
[ "$(sha256sum $BASE/manifest.json | cut -c1-64)" = 5f939d0021986cf071741012d49e4ad5acaafa814597aba94649ad4207a83aaf ] || fail "live base tree manifest"
[ "$(sha256sum $D05/manifest.json | cut -c1-64)" = fc230f267293c94f101d48dffe78113276de017425e9923d120014dd1a69dd2c ] || fail "live D05 tree manifest"
sha256sum -c --quiet FILES || fail "staged files (FILES)"
mkdir -p trees
# byte-identical copies of the two live trees for the test processes (own inodes; same pinned manifests)
for spec in "live-base:$BASE:5f939d0021986cf071741012d49e4ad5acaafa814597aba94649ad4207a83aaf" \
            "live-d05:$D05:fc230f267293c94f101d48dffe78113276de017425e9923d120014dd1a69dd2c"; do
  t=${spec%%:*}; rest=${spec#*:}; src=${rest%%:*}; want=${rest##*:}
  if [ ! -f "trees/$t/manifest.json" ] || [ "$(sha256sum trees/$t/manifest.json | cut -c1-64)" != "$want" ]; then
    rm -rf "trees/$t"; nice -n 19 cp -a "$src" "trees/$t" || fail "copy $t"
  fi
  python3 -I - "trees/$t" "$want" <<'PY' || fail "live copy $t"
import hashlib, json, pathlib, sys
r = pathlib.Path(sys.argv[1]); raw = (r / "manifest.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == sys.argv[2]
for rel, h in json.loads(raw)["files"].items():
    assert hashlib.sha256((r / rel).read_bytes()).hexdigest() == h, rel
PY
done
for t in $(python3 -I -c 'import json; print(" ".join(sorted(json.load(open("MANIFESTS.json")))))'); do
  want=$(python3 -I -c "import json; print(json.load(open('MANIFESTS.json'))['$t'])")
  if [ -f "trees/$t/manifest.json" ] && [ "$(sha256sum trees/$t/manifest.json | cut -c1-64)" = "$want" ]; then continue; fi
  rm -rf "trees/$t"
  src=$BASE; case $t in *-d05) src=$D05;; esac
  nice -n 19 python3 -I tools/make_tree.py "${t%-*}" "$src" "trees/$t" > /dev/null || fail "make_tree $t"
  [ "$(sha256sum trees/$t/manifest.json | cut -c1-64)" = "$want" ] || fail "tree $t manifest"
done
python3 -I - <<'PY' || fail "tree file shas"
import hashlib, json, pathlib
for t in json.load(open("MANIFESTS.json")):
    r = pathlib.Path("trees") / t
    m = json.loads((r / "manifest.json").read_text())
    for rel, h in m["files"].items():
        assert hashlib.sha256((r / rel).read_bytes()).hexdigest() == h, (t, rel)
PY
( cd jit && sha256sum -c --quiet ../jit.sha256 ) || fail "jit caches (jit.sha256)"
d=$(cat MANIFESTS.json jit.sha256 FILES | sha256sum | cut -c1-16)
echo "$d $(date -u +%FT%TZ)" > PREP.ok
echo "PREP $(hostname) OK digest $d trees $(ls trees | wc -l) jit $(find jit -name .committed | wc -l)"
