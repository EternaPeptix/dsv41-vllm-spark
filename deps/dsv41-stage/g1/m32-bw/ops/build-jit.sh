#!/bin/bash
# g1/m32-bw offline build (CPU only, niced, 3 GiB cap per nvcc; safe while serving). Run on ONE node (spark-four):
#   1. build every variant tree from the immutable live trees (tools/make_tree.py; .so copied, manifest regenerated)
#   2. pre-JIT: compile the live JIT kernel.cu (FC1 compact 576x5120; D05 FC2 5120x384) against each variant tree
#      with the JIT's own nvcc flags, into jit/<tree>/cache/<tag>.<deep_jit key>/{kernel.cu,kernel.cubin,meta.json,
#      .committed} (key = tools/djkey.py, verified against 6 live entries), plus the unchanged transpose entries
#   3. SASS report: the 'off' trees must give SASS identical to the live cubins (patch mechanics are inert)
# Usage: bash ops/build-jit.sh [variants...]   (default: off pair pf la ds all)
set -u
G=$HOME/dsv41-stage/g1/m32-bw
cd "$G" || exit 1
BASE=$HOME/dsv41-stage/m32/target-m32-native-v1
D05=$HOME/dsv41-goal120-20261024/d05-shadow-v1/native
LC=$HOME/dsv41-stage/m32/target-m32-jit-cache-v1/cache
DC=$HOME/dsv41-stage/pd/d05/jit-cache/cache
FC1=$LC/sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d.4681d5cc667a19725f0a8f2aa4a459e1
FC2=$LC/sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d.95b1b5958457ab43764c03ae562a3a7a
FD5=$DC/dsv41_fc2_live_store_v1.690c25bb46fc5afb213dfd120a12d1aa
TP1=$LC/transpose_and_pack_fp32_into_ue8m0.b384fe9bd0f290b740a467a6ac044821
TP2=$LC/transpose_and_pack_fp32_into_ue8m0.d4c37c50ff509f4a3690ea2cd19b2e25
VARS=${*:-off pair pf la ds all}
FL="--cubin --gpu-architecture=sm_120f -O3 --compiler-options=-O3 --ptxas-options=--register-usage-level=10 -std=c++20 --compiler-options=-fPIC --compiler-options=-fconcepts --expt-relaxed-constexpr --expt-extended-lambda --diag-suppress=39,161,174,177,186,940 --compiler-options=-Wno-deprecated-declarations,-Wno-abi"
mkdir -p trees jit cc
NV=/usr/local/cuda/bin

sass() { $NV/cuobjdump -sass "$1" 2>/dev/null | grep -vE '^\s*$|Function|\.headerflags|code for|arch =' | sed 's@/\*[0-9a-f]*\*/@@g'; }

compile() { # tree tag src_entry_dir label
  local tree=$1 tag=$2 src=$3 lab=$4
  local key; key=$(python3 -I tools/djkey.py --include "$G/trees/$tree/include" --source "$src/kernel.cu" --meta "$src/meta.json") || return 1
  local out=$G/jit/$tree/cache/$tag.$key
  if [ -f "$out/.committed" ]; then echo "HIT $tree $lab $key"; return 0; fi
  local tmp=$G/jit/$tree/tmp/$tag.$key.$$; rm -rf "$tmp"; mkdir -p "$tmp"
  cp "$src/kernel.cu" "$tmp/kernel.cu"
  local t0=$(date +%s)
  systemd-run --user --scope --quiet -p MemoryMax=3G -p MemorySwapMax=0 -- nice -n 19 \
    $NV/nvcc "$tmp/kernel.cu" -o "$tmp/kernel.cubin" $FL --include-path "$G/trees/$tree/include" > "$G/cc/$tree.$lab.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAIL $tree $lab rc=$rc (cc/$tree.$lab.log)"; tail -20 "$G/cc/$tree.$lab.log"; return 1; fi
  python3 -I - "$src/meta.json" "$tmp/meta.json" "$G/trees/$tree" "$out" "$FL" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
root, out, fl = sys.argv[3], sys.argv[4], sys.argv[5]
m["command"] = "/usr/local/cuda/bin/nvcc %s/kernel.cu --output-file %s/kernel.cubin %s --include-path %s/include" % (
    out, out, fl, root)
m["config"]["python_library_root"] = root
m["config"]["include_dirs"] = [root + "/include"]
m["g1_m32bw"] = "pre-JIT by ops/build-jit.sh (offline nvcc, same flags and compiler as the JIT)"
json.dump(m, open(sys.argv[2], "w"))
PY
  : > "$tmp/.committed"
  mkdir -p "$G/jit/$tree/cache"; mv "$tmp" "$out"; rmdir "$G/jit/$tree/tmp" 2>/dev/null
  sass "$out/kernel.cubin" > "$G/cc/$tree.$lab.sass"
  local regs; regs=$($NV/cuobjdump -res-usage "$out/kernel.cubin" 2>/dev/null | grep -oE 'REG:[0-9]+|STACK:[0-9]+|LOCAL:[0-9]+|SHARED:[0-9]+' | tr '\n' ' ')
  echo "BUILT $tree $lab $key $(( $(date +%s) - t0 ))s $regs lines=$(wc -l < "$G/cc/$tree.$lab.sass")"
}

for v in $VARS; do
  for kind in base d05; do
    case "$kind:$v" in d05:pair|d05:pf|d05:la) continue;; esac   # FC1-only variants reuse the live D05 native
    src=$BASE; [ $kind = d05 ] && src=$D05
    if [ ! -f "trees/$v-$kind/manifest.json" ]; then
      nice -n 19 python3 -I tools/make_tree.py "$v" "$src" "trees/$v-$kind" || { echo "FAIL tree $v-$kind"; exit 1; }
    fi
  done
done

pids=()
for v in $VARS; do
  ( compile "$v-base" sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d "$FC1" fc1
    for tp in "$TP1" "$TP2"; do    # transpose kernels do not include the patched header: same key, copy as is
      k=$(python3 -I tools/djkey.py --include "$G/trees/$v-base/include" --source "$tp/kernel.cu" --meta "$tp/meta.json")
      [ "$k" = "${tp##*.}" ] || { echo "FAIL transpose key moved $v $k"; exit 1; }
      mkdir -p "jit/$v-base/cache"; [ -d "jit/$v-base/cache/${tp##*/}" ] || cp -a "$tp" "jit/$v-base/cache/"
    done
    if [ -d "trees/$v-d05" ]; then compile "$v-d05" dsv41_fc2_live_store_v1 "$FD5" fc2d05; fi
    if [ "$v" = off ]; then compile "$v-base" sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d "$FC2" fc2; fi
  ) > "cc/build-$v.log" 2>&1 &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 5; done
done
wait
cat cc/build-*.log

# SASS identity of the inert ('off') trees vs the live cubins
sass "$FC1/kernel.cubin" > cc/live.fc1.sass
sass "$FD5/kernel.cubin" > cc/live.d05.sass
sass "$FC2/kernel.cubin" > cc/live.fc2.sass
if [ -f cc/off-base.fc1.sass ]; then
  cmp -s cc/live.fc1.sass cc/off-base.fc1.sass \
    && echo "CHECK off FC1 SASS == live ($(wc -l < cc/live.fc1.sass) lines)" || echo "CHECK-FAIL off FC1 SASS differs from live"
fi
if [ -f cc/off-d05.fc2d05.sass ]; then
  cmp -s cc/live.d05.sass cc/off-d05.fc2d05.sass \
    && echo "CHECK off D05-FC2 SASS == live ($(wc -l < cc/live.d05.sass) lines)" || echo "CHECK-FAIL off D05-FC2 SASS differs from live"
fi
if [ -f cc/off-base.fc2.sass ]; then
  cmp -s cc/live.fc2.sass cc/off-base.fc2.sass \
    && echo "CHECK off FC2 SASS == live ($(wc -l < cc/live.fc2.sass) lines)" || echo "CHECK-FAIL off FC2 SASS differs from live"
fi
for f in cc/*.sass; do
  printf '%-62s lines=%5d UTMALDG=%3d UTMAPF=%3d LDGSTS=%3d ARRIVES=%3d UTMASTG=%3d STG=%3d LDL/STL=%d\n' "${f#cc/}" \
    "$(wc -l < "$f")" "$(grep -c 'UTMALDG' "$f")" "$(grep -c 'UTMAPF' "$f")" "$(grep -c 'LDGSTS' "$f")" \
    "$(grep -c 'ARRIVES' "$f")" "$(grep -c 'UTMASTG' "$f")" "$(grep -cE '\bSTG\.' "$f")" "$(grep -cE '\b(LDL|STL)\b' "$f")"
done
( cd jit && find . -type f | sort | xargs sha256sum ) > jit.sha256
echo "JIT manifest $(sha256sum jit.sha256 | cut -c1-16) entries=$(find jit -name .committed | wc -l)"
