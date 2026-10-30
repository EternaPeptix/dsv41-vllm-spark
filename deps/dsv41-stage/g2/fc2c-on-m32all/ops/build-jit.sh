#!/bin/bash
# g2 fc2c-on-m32all offline pre-JIT + SASS checks (CPU only, nice 19, 3 GiB cap per nvcc, serial; safe while
# serving). Run on ONE node (spark-four) after tools/make_fc2c_tree.py built tree/:
#   1. deep_jit key check: tools/djkey.py (m32-bw copy) reproduces the keys of the live all-d05 FC2 entry
#      (c9b9ccac) and the g1 fc2c-d05 entry (16a0ffc6) from their own kernel.cu + include trees
#   2. A/A: the live all-d05 FC2 kernel.cu (no kBMainK -> 0) compiled against tree/include must give SASS identical
#      to the SERVING all-d05 cubin (the merged header is inert for every non-compact instantiation)
#   3. pre-JIT: the fc2c kernel.cu (kBMainK = 256; the code the fc2c native emits) against tree/include, with the
#      JIT's own nvcc flags, into jit-cache/cache/dsv41_fc2c_d05_v1.<key>/{kernel.cu,kernel.cubin,meta.json,.committed}
#   4. SASS diff live all-d05 -> compact (normalized) next to the qualified g1 diff (D05 -> fc2c-d05)
# Usage: bash ops/build-jit.sh            (idempotent: an existing committed entry is kept: "HIT")
set -u
G=$HOME/dsv41-stage/g2/fc2c-on-m32all
cd "$G" || exit 1
M32=$HOME/dsv41-stage/g1/m32-bw
FC2C=$HOME/dsv41-stage/g1/fc2c-d05
LIVE_E=$M32/jit/all-d05/cache/dsv41_fc2_live_store_v1.c9b9ccac684672ef9f5685ec1f8e12dc
FC2C_E=$FC2C/jit-cache/cache/dsv41_fc2c_d05_v1.16a0ffc69b95c7356325536c241eea18
NV=/usr/local/cuda/bin
FL="--cubin --gpu-architecture=sm_120f -O3 --compiler-options=-O3 --ptxas-options=--register-usage-level=10 -std=c++20 --compiler-options=-fPIC --compiler-options=-fconcepts --expt-relaxed-constexpr --expt-extended-lambda --diag-suppress=39,161,174,177,186,940 --compiler-options=-Wno-deprecated-declarations,-Wno-abi"
mkdir -p cc jit-cache/cache
[ -f tree/manifest.json ] || { echo "FAIL no tree/ (run tools/make_fc2c_tree.py tree first)"; exit 1; }
[ "$(systemctl --user is-active dsv41-rank)" = active ] && echo "note: dsv41-rank active here; CPU-only nice-19 build, 3 GiB cap"

# normalized SASS: instruction text only (no addresses/encodings), as g1 fc2c-d05 offline-nvcc
sass() { $NV/cuobjdump -sass "$1" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#^\s+/\*[0-9a-f]{4}\*/\s+##; s#\s*/\*.*\*/\s*$##; s# ;$##'; }
# + branch/BSSY targets folded to ADDR (shift with code size)
sassn() { sass "$1" | sed -E 's#(BSSY(\.[A-Z]+)* B[0-9]+, )0x[0-9a-f]+#\1ADDR#; s#(BRA(\.[A-Z]+)* (!?U?P[0-9T], )?)0x[0-9a-f]+#\1ADDR#; s#(CALL\.REL(\.[A-Z]+)* )0x[0-9a-f]+#\1ADDR#'; }
cap() { systemd-run --user --scope --quiet -p MemoryMax=3G -p MemorySwapMax=0 -- nice -n 19 "$@"; }

# 1. key checks
k1=$(python3 -I tools/djkey.py --include "$M32/trees/all-d05/include" --source "$LIVE_E/kernel.cu" --meta "$LIVE_E/meta.json")
k2=$(python3 -I tools/djkey.py --include "$FC2C/native/include" --source "$FC2C_E/kernel.cu" --meta "$FC2C_E/meta.json")
[ "$k1" = "${LIVE_E##*.}" ] && echo "KEY live all-d05 reproduced $k1" || { echo "FAIL key live all-d05 $k1 != ${LIVE_E##*.}"; exit 1; }
[ "$k2" = "${FC2C_E##*.}" ] && echo "KEY g1 fc2c-d05 reproduced $k2" || { echo "FAIL key g1 fc2c-d05 $k2 != ${FC2C_E##*.}"; exit 1; }
cp "$LIVE_E/kernel.cu" cc/d05-kernel.cu
cp "$FC2C_E/kernel.cu" cc/fc2c-kernel.cu
# the two kernel.cu differ only by the trailing kBMainK argument (fc2c native, b_main_k = 256)
diff cc/d05-kernel.cu cc/fc2c-kernel.cu > cc/kernel-cu.diff
python3 -I - cc/d05-kernel.cu cc/fc2c-kernel.cu <<'PY' || exit 1
import sys
a, b = (open(x).read() for x in sys.argv[1:3])
assert b == a.replace("\n        true\n    >);", "\n        true,\n        256\n    >);"), "kernel.cu differs beyond kBMainK"
print("KERNEL.CU fc2c == d05 + ', 256' (kBMainK)")
PY

# 2. A/A: D05 kernel.cu against the merged tree == serving all-d05 cubin
t0=$(date +%s)
cap $NV/nvcc cc/d05-kernel.cu -o cc/aa-d05-on-merged.cubin $FL --include-path "$G/tree/include" --resource-usage > cc/aa-d05-on-merged.res 2>&1 \
  || { echo "FAIL nvcc A/A"; tail -20 cc/aa-d05-on-merged.res; exit 1; }
sass "$LIVE_E/kernel.cubin" > cc/live-alld05.sass
sass cc/aa-d05-on-merged.cubin > cc/aa-d05-on-merged.sass
if cmp -s cc/live-alld05.sass cc/aa-d05-on-merged.sass; then
  echo "AA D05 code on merged tree: SASS == serving all-d05 cubin ($(wc -l < cc/live-alld05.sass) lines) $(( $(date +%s) - t0 ))s"
else
  echo "FAIL AA: D05 code on the merged tree changes SASS"; diff cc/live-alld05.sass cc/aa-d05-on-merged.sass | head -20; exit 1
fi

# 3. pre-JIT the compact kernel
key=$(python3 -I tools/djkey.py --include "$G/tree/include" --source cc/fc2c-kernel.cu --meta "$FC2C_E/meta.json") || exit 1
out=$G/jit-cache/cache/dsv41_fc2c_d05_v1.$key
if [ -f "$out/.committed" ]; then
  echo "HIT $out"
else
  tmp=$G/jit-cache/tmp/prejit.$$; rm -rf "$tmp"; mkdir -p "$tmp"
  cp cc/fc2c-kernel.cu "$tmp/kernel.cu"
  t0=$(date +%s)
  cap $NV/nvcc "$tmp/kernel.cu" -o "$tmp/kernel.cubin" $FL --include-path "$G/tree/include" > cc/prejit.log 2>&1 \
    || { echo "FAIL nvcc pre-JIT"; tail -20 cc/prejit.log; exit 1; }
  python3 -I - "$FC2C_E/meta.json" "$tmp/meta.json" "$G/tree" "$out" "$FL" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
root, out, fl = sys.argv[3], sys.argv[4], sys.argv[5]
m["command"] = "/usr/local/cuda/bin/nvcc %s/kernel.cu --output-file %s/kernel.cubin %s --include-path %s/include" % (
    out, out, fl, root)
m["config"]["python_library_root"] = root
m["config"]["include_dirs"] = [root + "/include"]
m["g2_fc2c_on_m32all"] = "pre-JIT by ops/build-jit.sh (offline nvcc, same flags and compiler as the JIT)"
json.dump(m, open(sys.argv[2], "w"))
PY
  : > "$tmp/.committed"
  mv "$tmp" "$out"; rmdir "$G/jit-cache/tmp" 2>/dev/null
  echo "BUILT $out $(( $(date +%s) - t0 ))s"
fi
cap $NV/nvcc cc/fc2c-kernel.cu -o cc/fc2c-on-merged.cubin $FL --include-path "$G/tree/include" --resource-usage > cc/fc2c-on-merged.res 2>&1
sass "$out/kernel.cubin" > cc/prejit.sass
sass cc/fc2c-on-merged.cubin > cc/fc2c-on-merged.sass
cmp -s cc/prejit.sass cc/fc2c-on-merged.sass && echo "PREJIT SASS reproducible (second compile identical, $(wc -l < cc/prejit.sass) lines)" \
  || echo "WARN pre-JIT SASS differs between two compiles"

# 4. SASS diffs: live all-d05 -> compact (this build) and plain D05 -> fc2c-d05 (qualified g1 build)
sassn "$LIVE_E/kernel.cubin" > cc/live-alld05.sassn
sassn "$out/kernel.cubin" > cc/prejit.sassn
diff cc/live-alld05.sassn cc/prejit.sassn > cc/sass-diff-alld05-to-g2.txt
sassn "$HOME/dsv41-stage/pd/d05/jit-cache/cache/dsv41_fc2_live_store_v1.690c25bb46fc5afb213dfd120a12d1aa/kernel.cubin" > cc/live-d05.sassn
sassn "$FC2C_E/kernel.cubin" > cc/g1-fc2c.sassn
diff cc/live-d05.sassn cc/g1-fc2c.sassn > cc/sass-diff-d05-to-g1fc2c.txt
for f in cc/live-alld05.sassn cc/prejit.sassn cc/live-d05.sassn cc/g1-fc2c.sassn; do
  printf '%-26s lines=%4d UTMALDG=%2d UTMAPF=%2d UTMACCTL=%2d UTMASTG=%2d STG=%2d LDL/STL=%d\n' "${f#cc/}" "$(wc -l < "$f")" \
    "$(grep -c UTMALDG "$f")" "$(grep -c UTMAPF "$f")" "$(grep -c UTMACCTL "$f")" "$(grep -c UTMASTG "$f")" \
    "$(grep -cE '\bSTG\.' "$f")" "$(grep -cE '\b(LDL|STL)\b' "$f")"
done
echo "== resources"; grep -hE "registers|spill" cc/aa-d05-on-merged.res cc/fc2c-on-merged.res
echo "== diff live all-d05 -> g2 compact ($(grep -c '^[<>]' cc/sass-diff-alld05-to-g2.txt) changed lines)"; cat cc/sass-diff-alld05-to-g2.txt
echo "== diff live D05 -> g1 fc2c-d05 ($(grep -c '^[<>]' cc/sass-diff-d05-to-g1fc2c.txt) changed lines)"; cat cc/sass-diff-d05-to-g1fc2c.txt
( cd jit-cache && find . -type f ! -name '.committed' | sort | xargs sha256sum; find . -name .committed | sort ) > jit.sha256
echo "JIT entries=$(find jit-cache -name .committed | wc -l) list $(sha256sum jit.sha256 | cut -c1-16)"
