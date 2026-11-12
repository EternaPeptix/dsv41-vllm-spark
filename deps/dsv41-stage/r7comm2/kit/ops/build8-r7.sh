#!/bin/bash
# build8-r7.sh [target ...]: build r7-comm2 .so files on spark-eight (the only node allowed to compile), from the Mac.
# Targets (default: all, in this order):
#   r7          src/rdma_ar.cu (r6 C1 + C2 + HT13)   -> prebuilt/rdma_ar_ext.so        + sass/r7-<sha8>.sass
#   m_rev m_seed0 m_negzero m_ffma m_tree m_waitlate m_stage mp_first mp_warp mp_l2 mp_unfused mp_chunk
#               (src/mutants/rdma_ar.<m>.cu, from src/mk_mutants7.py) -> prebuilt/mutants/<m>.so + sass/<m>-<sha8>.sass
#   ctl_r6      the r6-comm source (2bff8a89) rebuilt now -> sass/ctl_r6-<sha8>.sass (r6 kernels' identity control; not shipped)
# 14 full builds of ~5 GB / several minutes each: the stopped window only (build.sh refuses below 13 GB MemAvailable).
# Rules (climb-r1/r2 briefs): spark-eight only, ~/climb-scratch/r7-comm2/ only, nice 19, ninja -j1, CUDA_VISIBLE_DEVICES=
# (no GPU), never while ~/dsv41-opt-20261021/CLUSTER_LOCK exists on spark-one: a positive FREE before every build and a
# poll every 10 s during it; a lock (or 3 polls without an answer) kills that build's own process group and exits 3.
# SASS is dumped with cuobjdump on spark-eight (CPU) to stdout and saved here; the gate itself runs on the Mac / 512S1.
set -euo pipefail
cd "$(dirname "$0")/.."
TS=$HOME/Documents/dsv41-kits/ts-ssh/bin
[ -x "$TS/ssh" ] && export PATH=$TS:$PATH
B=spark-eight
W='$HOME/climb-scratch/r7-comm2'
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p build-logs prebuilt/mutants sass
LOG=build-logs/build8-$STAMP.log
TARGETS=${*:-r7 m_rev m_seed0 m_negzero m_ffma m_tree m_waitlate m_stage mp_first mp_warp mp_l2 mp_unfused mp_chunk ctl_r6}
# DW_TOKEN (exported by climb-common/device-window.sh for its MAC_PLAN steps): inside the lead's own STOPPED device
# window the lock carries that token; the service is stopped, so a build disturbs nothing and has the memory it needs.
lock() { ssh -n -o ConnectTimeout=8 -o BatchMode=yes spark-one \
  "if [ -e ~/dsv41-opt-20261021/CLUSTER_LOCK ]; then if [ -n '${DW_TOKEN:-}' ] && grep -qx 'token=${DW_TOKEN:-}' ~/dsv41-opt-20261021/CLUSTER_LOCK; then echo FREE; else echo LOCKED; fi; else echo FREE; fi" 2>/dev/null || echo NOANSWER; }
srcof() { case $1 in r7) echo src/rdma_ar.cu;; ctl_r6) echo ref/rdma_ar.r6-2bff8a89.cu;; *) echo src/mutants/rdma_ar.$1.cu;; esac; }
for t in $TARGETS; do [ -f "$(srcof $t)" ] || { echo "build8-r7: no source for $t ($(srcof $t))"; exit 2; }; done
l=$(lock); [ "$l" = FREE ] || { echo "build8-r7: CLUSTER_LOCK check says $l: not compiling" | tee -a "$LOG"; exit 3; }
# stage every source as <target>/rdma_ar.cu plus build.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cp build.sh "$TMP/"
for t in $TARGETS; do mkdir -p "$TMP/src-$t"; cp "$(srcof $t)" "$TMP/src-$t/rdma_ar.cu"; done
( cd "$TMP" && COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf - . ) | \
  ssh "$B" "mkdir -p $W/kit && cd $W/kit && tar xzf - && sha256sum src-*/rdma_ar.cu" | tee -a "$LOG"

build() {  # $1 target
  local t=$1 out="build-$1"
  ssh -n "$B" "cd $W/kit && rm -rf $W/$out && R7C_NO_LOCK_CHECK=1 R7C_SRC=$W/kit/src-$t R7C_OUT=$W/$out bash build.sh" \
    >> "$LOG" 2>&1 &
  local sp=$! miss=0
  while kill -0 "$sp" 2>/dev/null; do
    sleep 10
    l=$(lock)
    if [ "$l" = FREE ]; then miss=0; continue; fi
    [ "$l" = NOANSWER ] && miss=$((miss + 1)) && [ "$miss" -lt 3 ] && continue
    echo "build8-r7: lock state $l during the $t build: killing its process group on $B" | tee -a "$LOG"
    ssh -n "$B" "p=\$(cat $W/$out/build.pgid 2>/dev/null) && kill -TERM -- -\$p; sleep 3; kill -KILL -- -\$p 2>/dev/null; rm -f $W/$out/*.o" || true
    wait "$sp" || true
    exit 3
  done
  wait "$sp"
}

for t in $TARGETS; do
  l=$(lock); [ "$l" = FREE ] || { echo "build8-r7: lock state $l before $t: stopping" | tee -a "$LOG"; exit 3; }
  t0=$(date +%s)
  echo "build8-r7: $STAMP build $t on $B ($(shasum -a 256 "$(srcof $t)" | cut -c1-16))" | tee -a "$LOG"
  build "$t"
  SO=$(ssh -n "$B" "sha256sum $W/build-$t/rdma_ar_ext.so" | cut -d' ' -f1)
  echo "build8-r7: $t rdma_ar_ext.so sha256 $SO ($(( $(date +%s) - t0 )) s)" | tee -a "$LOG"
  ssh -n "$B" "nice -n 19 /usr/local/cuda/bin/cuobjdump -sass $W/build-$t/rdma_ar_ext.so" > "sass/$t-${SO:0:8}.sass"
  ssh -n "$B" "nice -n 19 /usr/local/cuda/bin/cuobjdump -res-usage $W/build-$t/rdma_ar_ext.so" > "sass/$t-${SO:0:8}.resusage"
  case $t in
    r7) dst=prebuilt/rdma_ar_ext.so ;;
    ctl_r6) dst= ;;
    *) dst=prebuilt/mutants/$t.so ;;
  esac
  if [ -n "$dst" ]; then
    ssh -n "$B" "cat $W/build-$t/rdma_ar_ext.so" > "$dst.tmp"
    [ "$(shasum -a 256 "$dst.tmp" | cut -d' ' -f1)" = "$SO" ] || { echo "build8-r7: fetch hash mismatch ($t)"; exit 2; }
    mv "$dst.tmp" "$dst"
  fi
  echo "$SO  $t  $(srcof $t)  $STAMP" >> build-logs/BUILT.txt
  # record the shipped binaries in KIT.json (install.py and device_test.py refuse anything else)
  python3 - "$t" "$SO" "$STAMP" <<'PY'
import json, sys
t, so, stamp = sys.argv[1:4]
k = json.load(open("KIT.json"))
if t == "r7":
    for f in k["files"]:
        if f["path"].endswith("r7comm2/build/rdma_ar_ext.so"):
            f["sha256"] = so
            f["note"] = (f"built once on spark-eight by ops/build8-r7.sh at {stamp} (build-logs/build8-{stamp}.log); "
                         "prebuilt/rdma_ar_ext.so, installed byte-identical on 8/8 by install.py")
elif t.startswith(("m_", "mp_")):
    k.setdefault("mutant_so_sha256", {})[t] = so
json.dump(k, open("KIT.json", "w"), indent=1)
open("KIT.json", "a").write("\n")
PY
done
echo "build8-r7: done ($TARGETS). Next: python3 -I gate/run_gate7.py (512S1), then KIT.json shas" | tee -a "$LOG"
