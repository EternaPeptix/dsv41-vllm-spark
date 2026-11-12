#!/usr/bin/env bash
# Build the XSPLIT .so files with the live sparse_mla_sm120 JIT flags (build.ninja of
# ~/.cache/flashinfer/0.7.0.post1/121a/cached_ops/sparse_mla_sm120) against the live headers.
# CPU only. Compile on spark-eight only, never while CLUSTER_LOCK exists on spark-one:
#   nice -n 19 bash build.sh <out_dir> [all|bench|<variant>...]
# variants: serving a0r1 a0r0 a1r1 a2r1 a3r1 a0r1p0
set -euo pipefail
OUT=${1:?out dir}
WHAT=${2:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
SP=${SP:-$HOME/venvs/vllm032/lib/python3.12/site-packages}
FI=$SP/flashinfer/data
CUDA=${CUDA_HOME:-/usr/local/cuda}
NVCC=$CUDA/bin/nvcc
mkdir -p "$OUT"

# Same order and content as the live cuda_cflags (python / tvm_ffi includes dropped: unused here).
FLAGS=(
  -I"$FI/cccl/cub" -I"$FI/cccl/libcudacxx/include" -I"$FI/cccl/thrust"
  -isystem "$CUDA/include" -isystem "$FI/include" -isystem "$FI/csrc"
  -isystem "$FI/cutlass/include" -isystem "$FI/cutlass/tools/util/include"
  --compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false
  -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1
  -use_fast_math -Xfatbin=-compress-all --compress-mode=size
  -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3
  -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -gencode=arch=compute_121a,code=sm_121a
  -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1
  -I"$HERE"
)
LD=(-shared -L"$CUDA/lib64" -L"$CUDA/lib64/stubs" -lcudart -lcuda)

one() {  # name, extra defines...
  local name=$1; shift
  echo "== $name $*"
  "$NVCC" "${FLAGS[@]}" "$@" -c "$HERE/dsv41_xsplit.cu" -o "$OUT/$name.o"
  c++ "$OUT/$name.o" "${LD[@]}" -o "$OUT/$name.so"
  rm -f "$OUT/$name.o"
  sha256sum "$OUT/$name.so" | awk '{print $1}' > "$OUT/$name.so.sha256"
  "$CUDA/bin/cuobjdump" --dump-resource-usage "$OUT/$name.so" > "$OUT/$name.resource.txt" 2>&1 || true
  "$CUDA/bin/cuobjdump" -sass "$OUT/$name.so" > "$OUT/$name.sass" 2>&1 || true
  cat "$OUT/$name.so.sha256"
}

variant() {
  case "$1" in
    serving) one libdsv41_xsplit -DXS_ARM=0 -DXS_ROPE_IN_WS=1 -DXS_B_PREFETCH=1 ;;
    a0r1) one libdsv41_xsplit_bench_a0r1 -DXS_BENCH -DXS_ARM=0 -DXS_ROPE_IN_WS=1 -DXS_B_PREFETCH=1 ;;
    a0r0) one libdsv41_xsplit_bench_a0r0 -DXS_BENCH -DXS_ARM=0 -DXS_ROPE_IN_WS=0 ;;
    a1r1) one libdsv41_xsplit_bench_a1r1 -DXS_BENCH -DXS_ARM=1 -DXS_ROPE_IN_WS=1 ;;
    a2r1) one libdsv41_xsplit_bench_a2r1 -DXS_BENCH -DXS_ARM=2 -DXS_ROPE_IN_WS=1 ;;
    a3r1) one libdsv41_xsplit_bench_a3r1 -DXS_BENCH -DXS_ARM=3 -DXS_ROPE_IN_WS=1 ;;
    a0r1p0) one libdsv41_xsplit_bench_a0r1p0 -DXS_BENCH -DXS_ARM=0 -DXS_ROPE_IN_WS=1 -DXS_B_PREFETCH=0 ;;
    *) echo "unknown variant $1"; exit 2 ;;
  esac
}

shift  # out dir
[ $# -gt 0 ] && shift  # WHAT
case "$WHAT" in
  all) set -- serving a0r1 a0r0 a1r1 a2r1 a3r1 a0r1p0 ;;
  bench) set -- a0r1 a0r0 a1r1 a2r1 a3r1 a0r1p0 ;;
  *) set -- "$WHAT" "$@" ;;
esac
for v in "$@"; do variant "$v"; done
echo BUILD OK
