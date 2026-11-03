#!/bin/bash
# build.sh: CPU-only build of the BS03 kit into build/ (no GPU access, safe beside the running service).
#
# Recipe: ~/.cache/torch_extensions/py312_cu130/vllm_moet_mhc_post_norm_sm120/build.ninja, the Moet extension production
# loads. Same nvcc rule (cuda_cflags), g++ rule (cflags) and link line (ldflags). Differences:
#   - -gencode=arch=compute_121,code=sm_121 (that build says compute_120) and the extension name;
#   - two translation units, kernels without torch headers: one nvcc run over torch/extension.h took MemAvailable below the
#     floor on this node, in its first attempt on 2026-11-02;
#   - --param ggc-* on the g++ glue step only: GCC memory, no effect on generated code.
# Everything runs at nice 19, one job at a time, and stops if MemAvailable falls below the site's 6 GB floor.
# Nothing is written outside this directory.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
B=$HERE/build
NAME=vllm_bs03_det_topk_sm121
T=$HOME/venvs/vllm032/lib/python3.12/site-packages/torch
PYINC=$HOME/.local/share/uv/python/cpython-3.12.14-linux-aarch64-gnu/include/python3.12
CUDA=/usr/local/cuda
NVCC=$CUDA/bin/nvcc
FLOOR_KB=${BS03_MEM_FLOOR_KB:-6291456}   # 6 GiB
VLIMIT_KB=${BS03_VLIMIT_KB:-8388608}     # 8 GiB per process

mkdir -p "$B"
export CUDA_VISIBLE_DEVICES=
export TMPDIR=$B/tmp
mkdir -p "$TMPDIR"
ulimit -v "$VLIMIT_KB"

mem_kb() { awk '/MemAvailable/{print $2}' /proc/meminfo; }

# run_guarded <log> <cmd...>: nice 19 in its own session; killed (exit 99) when MemAvailable drops below the floor.
run_guarded() {
  local log=$1; shift
  local low start t0
  start=$(mem_kb); low=$start; t0=$(date +%s)
  setsid nice -n 19 ionice -c3 "$@" >>"$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    local m
    m=$(mem_kb)
    [ "$m" -lt "$low" ] && low=$m
    if [ "$m" -lt "$FLOOR_KB" ]; then
      echo "MEMORY GUARD: MemAvailable ${m} kB < floor ${FLOOR_KB} kB, killing this step" | tee -a "$log"
      kill -TERM -- "-$pid" 2>/dev/null; sleep 1; kill -KILL -- "-$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 99
    fi
    sleep 0.5
  done
  wait "$pid"
  local rc=$?
  echo "step rc=$rc: $(( $(date +%s) - t0 )) s, MemAvailable start ${start} kB, lowest ${low} kB" | tee -a "$LOG"
  return $rc
}

LOG=$B/build.log
: >"$LOG"
{
  echo "host $(hostname) $(date -u +%FT%TZ)"
  echo "MemAvailable start: $(mem_kb) kB"
  "$NVCC" --version | sed -n 1,3p
  c++ --version | head -1
  sha256sum "$HERE/src/det_topk_bs03.cu" "$HERE/src/det_topk_bs03.h" "$HERE/src/det_topk_bs03_py.cpp" "$HERE/src/bs03_stamp.cu"
} | tee -a "$LOG"

INCS=(-isystem "$T/include" -isystem "$T/include/torch/csrc/api/include" -isystem "$CUDA/include" -isystem "$PYINC")
DEFS=(-DTORCH_EXTENSION_NAME=$NAME -DTORCH_API_INCLUDE_EXTENSION_H)
CUDA_CFLAGS=("${DEFS[@]}" "${INCS[@]}"
  -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
  --expt-relaxed-constexpr -gencode=arch=compute_121,code=sm_121 --compiler-options -fPIC -O3 -std=c++20)
CFLAGS=("${DEFS[@]}" "${INCS[@]}" -fPIC -std=c++20 -O3 -std=c++20)
LDFLAGS=(-shared -L"$T/lib" -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L"$CUDA/lib64" -lcudart)

cd "$HERE/src" || exit 1

echo "== nvcc: kernels (no torch headers)" | tee -a "$LOG"
run_guarded "$LOG" "$NVCC" "${CUDA_CFLAGS[@]}" -c det_topk_bs03.cu -o "$B/det_topk_bs03.cuda.o" || { echo "nvcc FAILED, see $LOG"; exit 1; }

# The glue parses all of torch/extension.h (the step that comes closest to the memory floor), so it reruns only if it changed.
if [ "$B/det_topk_bs03_py.o" -nt det_topk_bs03_py.cpp ] && [ "$B/det_topk_bs03_py.o" -nt det_topk_bs03.h ]; then
  echo "== g++: torch glue up to date, kept" | tee -a "$LOG"
else
  echo "== g++: torch glue" | tee -a "$LOG"
  run_guarded "$LOG" c++ "${CFLAGS[@]}" --param ggc-min-expand=10 --param ggc-min-heapsize=32768 -c det_topk_bs03_py.cpp -o "$B/det_topk_bs03_py.o" || { echo "g++ FAILED, see $LOG"; exit 1; }
fi

echo "== link" | tee -a "$LOG"
run_guarded "$LOG" c++ "$B/det_topk_bs03.cuda.o" "$B/det_topk_bs03_py.o" "${LDFLAGS[@]}" -o "$B/$NAME.so" || { echo "link FAILED, see $LOG"; exit 1; }

if [ "$B/libbs03_stamp.so" -nt bs03_stamp.cu ]; then
  echo "== nvcc: test stamp helper up to date, kept" | tee -a "$LOG"
else
  echo "== nvcc: test stamp helper (recipe of ~/dsv41-stage/r4/l2-prefetch/l2pf.cu)" | tee -a "$LOG"
  run_guarded "$LOG" "$NVCC" -O3 -arch=sm_121a -shared -Xcompiler -fPIC -o "$B/libbs03_stamp.so" bs03_stamp.cu || { echo "stamp FAILED, see $LOG"; exit 1; }
fi

echo "== offline facts (cuobjdump, nm; no GPU)" | tee -a "$LOG"
{
  "$CUDA/bin/cuobjdump" --list-elf "$B/$NAME.so"
  "$CUDA/bin/cuobjdump" --dump-resource-usage "$B/$NAME.so"
} >"$B/resource-usage.txt" 2>&1
nm -D --defined-only "$B/$NAME.so" | grep -E 'PyInit' | tee -a "$LOG"
ldd "$B/$NAME.so" | grep -E 'cudart' | tee -a "$LOG"
sha256sum "$B/$NAME.so" "$B/libbs03_stamp.so" | tee -a "$LOG"
echo "MemAvailable end: $(mem_kb) kB" | tee -a "$LOG"
echo "BUILD_OK"
