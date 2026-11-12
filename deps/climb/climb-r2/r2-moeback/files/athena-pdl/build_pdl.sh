#!/bin/bash
# [r2-moeback G18] Build the PDL early-B Athena FC2 graph binding for sm_121a, mirroring the live build.ninja
# (~/"Athena Hillclimbing stuff"/moet-fc2packed-snapshot/fc2-graph-build/build.ninja, sha256 2de8ccdb) exactly,
# plus -DATHENA_FC2_PDL_EARLY_B=1 and the PDL binding source. Also builds a CONTROL .so from the unmodified live
# sources with the same commands, and dumps SASS of the live .so, the control and the PDL build so the kernel
# instruction streams can be diffed (control == live proves the toolchain reproduces production; PDL vs control
# must differ only by the griddepcontrol.wait and its predicate/branch plumbing).
#
# Usage (spark-eight only, never while ~/dsv41-opt-20261021/CLUSTER_LOCK exists on spark-one):
#   bash build_pdl.sh <src_dir = this athena-pdl folder> <out_dir = ~/climb-scratch/r2-moeback/build>
# Never loads the result into a GPU process. Single job, nice 19.
set -euo pipefail
# Host guard (review r2-moeback non-blocking #6): compiling is allowed on spark-eight only (climb-r2 brief rule 4).
if [ "$(hostname -s)" != spark-eight ]; then echo "build_pdl.sh: host $(hostname -s) is not spark-eight; refusing" >&2; exit 4; fi
SRC=$(cd "$1" && pwd)
OUT=$2
mkdir -p "$OUT"
cd "$OUT"
# one build per log: install.py vet-pdl reads the last run and checks the shas it printed
[ -f COMPILE-LOG.txt ] && mv COMPILE-LOG.txt "COMPILE-LOG.$(date -u +%Y%m%dT%H%M%SZ).txt"
exec > >(tee -a COMPILE-LOG.txt) 2>&1
echo "== build_pdl.sh $(date -u +%FT%TZ) on $(hostname)"
VENV=$HOME/venvs/vllm032
TORCH=$VENV/lib/python3.12/site-packages/torch
PYINC=$($VENV/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
NVCC=/usr/local/cuda/bin/nvcc
LIVE_SO=$HOME/dsv41-fc2/athena_tp8_fc2_packed_graph_sm121.so
LIVE_SRC="$HOME/Athena Hillclimbing stuff/moet-fc2packed-snapshot"
COMMON=(-DTORCH_EXTENSION_NAME=athena_tp8_fc2_packed_graph_sm121 -DTORCH_API_INCLUDE_EXTENSION_H
  -isystem "$TORCH/include" -isystem "$TORCH/include/torch/csrc/api/include" -isystem /usr/local/cuda/include
  -isystem "$PYINC" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__
  -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_121a,code=sm_121a
  --compiler-options '-fPIC' -O3 -std=c++20 -DATHENA_EXPERIMENTAL_TP8_NATIVE_FC2=1
  -DATHENA_EXPERIMENTAL_TP8_FC2_SERIAL32PACKED_K288=1 -DATHENA_EXPERIMENTAL_TP8_FC2_GRAPH_BINDING=1)
LINK=(-shared -L"$TORCH/lib" -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python
  -L/usr/local/cuda/lib64 -lcudart)
# Fail closed: anything but an explicit FREE from spark-one counts as locked, unless the caller holds the lock
# (the lead building inside its own window): R2MB_LOCK_TOKEN must equal the token= line of the lock file.
lock() {
  local st
  st=$(ssh -n -o ConnectTimeout=5 spark-one 'if test -e ~/dsv41-opt-20261021/CLUSTER_LOCK; then grep "^token=" ~/dsv41-opt-20261021/CLUSTER_LOCK || echo token=?; else echo FREE; fi' 2>/dev/null)
  [ "$st" = FREE ] && return 1
  [ -n "${R2MB_LOCK_TOKEN:-}" ] && [ "$st" = "token=$R2MB_LOCK_TOKEN" ] && return 1
  return 0
}
if lock; then echo "CLUSTER_LOCK present: refusing to compile" >&2; exit 3; fi
echo "== toolchain"; $NVCC --version | tail -2; c++ --version | head -1; echo "torch $TORCH"; echo "pyinc $PYINC"

# Control: the live sources, with the include path rewritten from /tmp/athena-fc2-graph-1000 to the snapshot.
mkdir -p control/moet-baseline-research
cp "$LIVE_SRC/moet-baseline-research/fc2_serial32packed_k288_candidate.cu" control/moet-baseline-research/
cp -r "$SRC/moet-sm121-port" control/
sed 's#/tmp/athena-fc2-graph-1000/moet-baseline-research/fc2_serial32packed_k288_candidate.cu#fc2_serial32packed_k288_candidate.cu#' \
  "$LIVE_SRC/moet-baseline-research/fc2_packed_graph_binding.generated.cu" > control/moet-baseline-research/binding.cu
echo "== control compile"
nice -n 19 $NVCC "${COMMON[@]}" -c control/moet-baseline-research/binding.cu -o control.o
lock && { echo "CLUSTER_LOCK appeared: stopping" >&2; exit 3; }
nice -n 19 c++ control.o "${LINK[@]}" -o control_athena_tp8_fc2_packed_graph_sm121.so

echo "== PDL compile"
nice -n 19 $NVCC "${COMMON[@]}" -DATHENA_FC2_PDL_EARLY_B=1 \
  -c "$SRC/moet-baseline-research/fc2_packed_graph_binding_pdl.cu" -o pdl.o
lock && { echo "CLUSTER_LOCK appeared: stopping" >&2; exit 3; }
nice -n 19 c++ pdl.o "${LINK[@]}" -o athena_tp8_fc2_packed_graph_pdl_sm121.so

echo "== SASS"
# whole fatbin (every kernel in the .so, today only athena_tp8_fc2_serial32packed_k288_candidate), so nothing
# depends on how cuobjdump -fun matches namespaced names; addresses may exceed 4 hex digits.
for so in "$LIVE_SO" control_athena_tp8_fc2_packed_graph_sm121.so athena_tp8_fc2_packed_graph_pdl_sm121.so; do
  b=$(basename "$so" .so)
  /usr/local/cuda/bin/cuobjdump -sass "$so" > "sass_$b.txt" 2>&1 || true
  grep -E 'Function :' "sass_$b.txt" | sed "s#^#  $b #" || true
  # instruction text only (drop addresses/encodings) for diffing
  grep -E '^\s+/\*[0-9a-f]{4,}\*/' "sass_$b.txt" | sed -E 's#^\s+/\*[0-9a-f]{4,}\*/\s+##; s#\s*/\*.*##; s#\s+;#;#' > "ins_$b.txt" || true
  echo "$b: $(wc -l < "ins_$b.txt") instructions, $(grep -c ACQBULK "ins_$b.txt" || true) ACQBULK"
done
echo "== diff live vs control (expect empty)"
diff "ins_$(basename "$LIVE_SO" .so).txt" ins_control_athena_tp8_fc2_packed_graph_sm121.txt && echo "IDENTICAL"
echo "== diff control vs pdl"
diff ins_control_athena_tp8_fc2_packed_graph_sm121.txt ins_athena_tp8_fc2_packed_graph_pdl_sm121.txt || true
echo "== sha256"
sha256sum "$LIVE_SO" control_athena_tp8_fc2_packed_graph_sm121.so athena_tp8_fc2_packed_graph_pdl_sm121.so
rm -f control.o pdl.o
echo "== next: python3 <kit>/install.py vet-pdl $OUT   (records the PDL .so sha in KIT.json only if every gate holds)"
