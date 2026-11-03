#!/bin/bash
# dev_build.sh <tu.cu> <out.cubin> <log>: guarded device-only compile (no torch headers, no host compile) with the
# recorded device flags. Used for SASS evidence while the service runs; safe to rerun.
set -u
TU=$1; OUT=$2; LOG=$3
HERE=$(cd "$(dirname "$0")" && pwd)
PATH=/usr/local/cuda/bin:$PATH bash "$HERE/guard_run.sh" "$LOG" -- /usr/local/cuda/bin/nvcc -cubin -arch=compute_121 -code=sm_121 \
  -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -O3 -lineinfo -std=c++20 "$TU" -o "$OUT"
