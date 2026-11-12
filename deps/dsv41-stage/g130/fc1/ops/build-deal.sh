#!/bin/bash
# g130 fc1 offline build: make the arm trees and pre-JIT cubins under ~/dsv41-stage/g130/fc1. CPU only, nice 19, one nvcc at a
# time, each under guard_run.sh and an explicit timeout. Never run it while a relaunch may be in progress.
# Usage: bash ops/build-deal.sh trees              make the arm trees (no compile)
#        bash ops/build-deal.sh fc1 <arm>          compile the FC1 kernel only (bench/test arm)
#        bash ops/build-deal.sh e128 <arm>         compile the two E128 drafter kernels + copy the transpose entries (serving)
#        bash ops/build-deal.sh sass               report; requires the entries above
set -u
K=$HOME/dsv41-stage/g130/fc1
G=$HOME/dsv41-stage/g1/m32-bw
SRC=$G/trees/all-base
LC=$G/jit/all-base/cache
TAG=sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d
FC1=$LC/$TAG.bed3be1011439a54aad5926d82bf1e35
DF1=$LC/$TAG.3f71c0e022a9d679ea003abd40f16f6b
DF2=$LC/$TAG.74a66dfd00d7eaf972893598d529bfc5
NV=/usr/local/cuda/bin
FL="--cubin --gpu-architecture=sm_120f -O3 --compiler-options=-O3 --ptxas-options=--register-usage-level=10 -std=c++20 --compiler-options=-fPIC --compiler-options=-fconcepts --expt-relaxed-constexpr --expt-extended-lambda --diag-suppress=39,161,174,177,186,940 --compiler-options=-Wno-deprecated-declarations,-Wno-abi"
cd "$K" || exit 1
mkdir -p trees jit cc logs

sass() { $NV/cuobjdump -sass "$1" 2>/dev/null | grep -vE '^\s*$|Function|\.headerflags|code for|arch =' | sed 's@/\*[0-9a-f]*\*/@@g'; }

compile() { # arm src_entry_dir label
  local arm=$1 src=$2 lab=$3 tree=$K/trees/$1-base
  local key; key=$(python3 -I "$G/tools/djkey.py" --include "$tree/include" --source "$src/kernel.cu" --meta "$src/meta.json") || return 1
  local out=$K/jit/$arm-base/cache/$TAG.$key
  if [ -f "$out/.committed" ]; then echo "HIT $arm $lab $key"; return 0; fi
  local tmp=$K/jit/$arm-base/tmp/$TAG.$key.$$; rm -rf "$tmp"; mkdir -p "$tmp"
  cp "$src/kernel.cu" "$tmp/kernel.cu"
  local t0; t0=$(date +%s)
  timeout -k 10 "${NVCC_TIMEOUT:-1500}" bash "$K/ops/guard_run.sh" "$K/logs/$arm.$lab.log" -- \
    env PATH=$NV:$PATH $NV/nvcc "$tmp/kernel.cu" -o "$tmp/kernel.cubin" $FL --include-path "$tree/include"
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAIL $arm $lab rc=$rc (logs/$arm.$lab.log)"; tail -15 "$K/logs/$arm.$lab.log"; return 1; fi
  python3 -I - "$src/meta.json" "$tmp/meta.json" "$tree" "$out" "$FL" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
root, out, fl = sys.argv[3], sys.argv[4], sys.argv[5]
m["command"] = "/usr/local/cuda/bin/nvcc %s/kernel.cu --output-file %s/kernel.cubin %s --include-path %s/include" % (out, out, fl, root)
m["config"]["python_library_root"] = root
m["config"]["include_dirs"] = [root + "/include"]
m["g130_fc1"] = "pre-JIT by ops/build-deal.sh (offline nvcc, same flags and compiler as the JIT)"
json.dump(m, open(sys.argv[2], "w"))
PY
  : > "$tmp/.committed"
  mkdir -p "$K/jit/$arm-base/cache"; mv "$tmp" "$out"; rmdir "$K/jit/$arm-base/tmp" 2>/dev/null
  sass "$out/kernel.cubin" > "$K/cc/$arm.$lab.sass"
  local regs; regs=$($NV/cuobjdump -res-usage "$out/kernel.cubin" 2>/dev/null | grep -oE 'REG:[0-9]+|STACK:[0-9]+|LOCAL:[0-9]+|SHARED:[0-9]+' | tr '\n' ' ')
  echo "BUILT $arm $lab $key $(( $(date +%s) - t0 ))s $regs lines=$(wc -l < "$K/cc/$arm.$lab.sass")"
}

case "${1:-}" in
  trees)
    for arm in deal dealx nomma; do
      [ -f "trees/$arm-base/manifest.json" ] || nice -n 19 python3 -I tools/make_deal_tree.py "$arm" "$SRC" "trees/$arm-base" header/dsv41_fc1_deal.cuh || exit 1
    done
    ;;
  fc1) compile "$2" "$FC1" fc1 ;;
  e128)
    compile "$2" "$DF1" df1 && compile "$2" "$DF2" df2 || exit 1
    for tp in "$LC"/transpose_and_pack_fp32_into_ue8m0.*; do
      k=$(python3 -I "$G/tools/djkey.py" --include "$K/trees/$2-base/include" --source "$tp/kernel.cu" --meta "$tp/meta.json")
      [ "$k" = "${tp##*.}" ] || { echo "FAIL transpose key moved $k"; exit 1; }
      mkdir -p "jit/$2-base/cache"; [ -d "jit/$2-base/cache/${tp##*/}" ] || cp -a "$tp" "jit/$2-base/cache/"
    done
    ;;
  sass)
    sass "$FC1/kernel.cubin" > cc/live.fc1.sass; sass "$DF1/kernel.cubin" > cc/live.df1.sass; sass "$DF2/kernel.cubin" > cc/live.df2.sass
    for f in cc/*.sass; do
      printf '%-24s lines=%5d UTMALDG=%3d UTMAPF=%3d LDGSTS=%3d STG=%3d QMMA=%3d LDL/STL=%d\n' "${f#cc/}" "$(wc -l < "$f")" \
        "$(grep -c UTMALDG "$f")" "$(grep -c UTMAPF "$f")" "$(grep -c LDGSTS "$f")" "$(grep -cE '\bSTG\.' "$f")" "$(grep -c QMMA "$f")" "$(grep -cE '\b(LDL|STL)\b' "$f")"
    done
    for a in deal dealx; do for l in df1 df2; do
      [ -f "cc/$a.$l.sass" ] && { cmp -s "cc/live.$l.sass" "cc/$a.$l.sass" && echo "CHECK $a $l SASS == live" || echo "CHECK-FAIL $a $l SASS differs from live"; }
    done; done
    ;;
  *) echo "usage: build-deal.sh trees|fc1 <arm>|e128 <arm>|sass"; exit 2 ;;
esac
