#!/bin/bash
# prep-node.sh <build|pre|post|clean> -- run ON one Spark. g2 fc2c-on-m32all (~/dsv41-stage/g2/fc2c-on-m32all).
#  build: CPU only, nice 19, safe while serving. Verify the staged files (MANIFEST.sha256), build tree/ from this
#         node's two immutable inputs (g1 fc2c-d05 native 3b6f30bc + m32-bw all-d05 073232ef) with
#         tools/make_fc2c_tree.py, require the pinned tree manifest (PINS), verify the pre-JIT cache list (jit.sha256),
#         run tests/test_g2_cpu.py, then write PREP.ok.
#  pre  : STOPPED WINDOW. Private test caches under jit-test/ (g1root/trees -> the real m32-bw trees, g1root/jit/
#         all-{base,d05} + fc2c-all + live-base + dg + triton = copies of the serving caches), and listings of every
#         shared/serving cache the tests could touch (path, size, mtime).
#  post : diff those caches against the pre listings (expected: nothing new/changed/removed, mtimes included).
#  clean: remove jit-test/ (private copies, ~180 MB) after the window.
set -u
G=$HOME/dsv41-stage/g2/fc2c-on-m32all
JT=$G/jit-test
R=$G/results
M32=$HOME/dsv41-stage/g1/m32-bw
FC2C=$HOME/dsv41-stage/g1/fc2c-d05
PY=$HOME/venvs/vllm032/bin/python
SP=$HOME/venvs/vllm032/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts
H=$(hostname)
mkdir -p "$R"
SHARED="$M32/jit/all-base $M32/jit/all-d05 $G/jit-cache $HOME/dsv41-stage/m32/target-m32-jit-cache-v1 $HOME/dsv41-stage/pd/d05/jit-cache $FC2C/jit-cache $HOME/.triton/cache $HOME/.cache/vllm/deep_gemm"
lst() { echo "$R/$1-$H-$(echo "$2" | tr / _).lst"; }
pin() { awk -v k="$1" '$1 == k {print $2}' "$G/PINS"; }
case "${1:-build}" in
build)
  (cd "$G" && sha256sum -c --quiet MANIFEST.sha256) || { echo "PREP $H FAIL staged files (MANIFEST.sha256)"; exit 1; }
  if [ ! -f "$G/tree/manifest.json" ]; then
    rm -rf "$G/tree.partial"
    (cd "$G" && nice -n 19 python3 -I tools/make_fc2c_tree.py tree "$FC2C/native" "$M32/trees/all-d05" tree.partial) \
      > "$R/tree-build-$H.txt" 2>&1 && mv "$G/tree.partial" "$G/tree" \
      || { echo "PREP $H FAIL tree build ($R/tree-build-$H.txt)"; tail -5 "$R/tree-build-$H.txt"; exit 1; }
  fi
  t=$(sha256sum "$G/tree/manifest.json" | cut -d' ' -f1)
  [ "$t" = "$(pin tree_manifest)" ] || { echo "PREP $H FAIL tree manifest $t != pin $(pin tree_manifest)"; exit 1; }
  j=$(cd "$G" && ( cd jit-cache && find . -type f ! -name '.committed' | sort | xargs sha256sum; find . -name .committed | sort ) | sha256sum | cut -d' ' -f1)
  [ "$j" = "$(pin jit_list)" ] || { echo "PREP $H FAIL jit-cache list $j != pin $(pin jit_list)"; exit 1; }
  CUDA_VISIBLE_DEVICES= nice -n 19 "$PY" -I "$G/tests/test_g2_cpu.py" \
    "$G/site/vllm/model_executor/layers/fused_moe/experts/sm121_target_m32.py" \
    "$( [ "$(sha256sum "$SP/sm121_target_m32.py" | cut -c1-8)" = db659d80 ] && echo "$SP/sm121_target_m32.py" || echo "$SP/sm121_target_m32.py.bak-g2-fc2c-on-m32all")" \
    "$FC2C/site/vllm/model_executor/layers/fused_moe/experts/sm121_target_m32.py" "$G" > "$R/cpu-test-$H.txt" 2>&1
  tail -1 "$R/cpu-test-$H.txt" | grep -q "SUMMARY PASS" || { echo "PREP $H FAIL cpu test ($R/cpu-test-$H.txt)"; grep -v '^ok' "$R/cpu-test-$H.txt" | head; exit 1; }
  d=$(cat "$G/MANIFEST.sha256" "$G/tree/manifest.json" "$G/jit.sha256" | sha256sum | cut -c1-16)
  echo "$d $(date -u +%FT%TZ) tree=${t:0:8} jit=${j:0:8} cpu=$(grep -c '^ok' "$R/cpu-test-$H.txt")/$(grep -cE '^(ok|FAIL)' "$R/cpu-test-$H.txt")" > "$G/PREP.ok"
  echo "PREP $H OK $(cat "$G/PREP.ok")";;
pre)
  [ -f "$G/PREP.ok" ] || { echo "PRE $H FAIL no PREP.ok (run build)"; exit 1; }
  (cd "$G" && sha256sum -c --quiet MANIFEST.sha256) || { echo "PRE $H FAIL staged files"; exit 1; }
  mkdir -p "$JT/g1root/jit"
  [ -L "$JT/g1root/trees" ] || ln -s "$M32/trees" "$JT/g1root/trees"
  cpcache() {  # <serving cache root (has cache/)> <private root>: fresh copy of cache/ (mtimes kept), no tmp/
    rm -rf "$2"; mkdir -p "$2"; cp -a "$1/cache" "$2/cache"
  }
  cpcache "$M32/jit/all-base" "$JT/g1root/jit/all-base"
  cpcache "$M32/jit/all-d05" "$JT/g1root/jit/all-d05"
  cpcache "$G/jit-cache" "$JT/fc2c-all"
  cpcache "$HOME/dsv41-stage/m32/target-m32-jit-cache-v1" "$JT/live-base"
  [ -d "$JT/dg" ] || cp -a "$HOME/.cache/vllm/deep_gemm" "$JT/dg"
  [ -d "$JT/triton" ] || cp -a "$HOME/.triton/cache" "$JT/triton"
  for d in $SHARED; do
    [ -d "$d" ] && find "$d" -type f -printf "%P %s %T@\n" | sort > "$(lst pre "$d")"
  done
  echo "PRE $H OK $(du -sh "$JT" | cut -f1) private caches; fc2c-all: $(ls "$JT/fc2c-all/cache")"
  ;;
post)
  bad=0
  for d in $SHARED; do
    [ -d "$d" ] || continue
    pre=$(lst pre "$d"); post=$(lst post "$d")
    [ -f "$pre" ] || { echo "  $d: no pre listing"; bad=1; continue; }
    find "$d" -type f -printf "%P %s %T@\n" | sort > "$post"
    n=$(comm -3 "$pre" "$post" | wc -l)
    [ "$n" = 0 ] || { echo "  $d: $n changed lines"; comm -3 "$pre" "$post" | head -5 | sed 's/^/     /'; bad=1; }
  done
  [ $bad = 0 ] && echo "CACHES $H OK unchanged" || echo "CACHES $H CHANGED";;
clean)
  rm -rf "$JT" && echo "CLEAN $H OK";;
*) echo "usage: prep-node.sh build|pre|post|clean"; exit 2;;
esac
