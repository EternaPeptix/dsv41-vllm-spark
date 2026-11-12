#!/bin/bash
# build.sh: build an r7-comm2 rdma_ar .so (the r7 source or one of its mutants) into $R6_OUT (climb-scratch on spark-eight).
# Same compiler, flags and link line as the live g130 .so (14d049dd) and the r3 .so (7ec43164), only the source and
# the output directory differ. Never touches ~/dsv41-stage/g130/rdma (the live .so is a serving dependency).
#
# Rules (climb-r2 brief): compile on spark-eight only, nice 19, ninja -j1, never while
# ~/dsv41-opt-20261021/CLUSTER_LOCK exists on spark-one (checked here over ssh when run on a Spark), never near a relaunch
# (WINDOW.md: a build uses ~5 GB of the unified memory). The guard (ops/guard_run.sh semantics) aborts below 8 GB free.
#   bash build.sh                 build, print sha256
#   bash build.sh --expect <sha>  also fail unless the result has that sha256 (reproducibility: a clean rebuild)
#   R7C_SRC=<dir> R7C_OUT=<dir>   override source / output dirs (ops/build8-r7.sh: ~/climb-scratch/r7-comm2/...)
# Normally driven from the Mac by ops/build8-r7.sh, which checks the lock on spark-one before and DURING the build and
# kills the build's process group if a lock appears. Refuses to run on any host but spark-eight (review N1).
# The memory watchdog and a lock-stop only kill this build's own process group (review N2): the compile runs under
# setsid (its pgid is written to $OUT/build.pgid), so a concurrent build of another kit on the node is never touched.
set -euo pipefail
if [ "$(hostname)" != spark-eight ]; then echo "build.sh: compiling is allowed on spark-eight only (this is $(hostname))"; exit 7; fi
SRC=${R7C_SRC:?R7C_SRC: directory holding rdma_ar.cu}
OUT=${R7C_OUT:?R7C_OUT: output directory}
EXPECT=""
[ "${1:-}" = "--expect" ] && EXPECT=${2:?}
VENV=$HOME/venvs/vllm032
PYINC=$(ls -d $HOME/.local/share/uv/python/cpython-3.12.*-linux-aarch64-gnu/include/python3.12 | head -1)
TORCH=$VENV/lib/python3.12/site-packages/torch
if [ -z "${R7C_NO_LOCK_CHECK:-}" ]; then
  # fail closed: compile only on a positive "FREE" answer from spark-one (an ssh failure is not a free cluster)
  lk=$(ssh -n -o ConnectTimeout=5 spark-one 'if [ -e ~/dsv41-opt-20261021/CLUSTER_LOCK ]; then echo LOCKED; else echo FREE; fi' 2>/dev/null || true)
  if [ "$lk" != FREE ]; then
    echo "build.sh: CLUSTER_LOCK check on spark-one says '${lk:-no answer}': not compiling (R7C_NO_LOCK_CHECK=1 only if you checked by hand)"; exit 3
  fi
fi
MEMINFO=${R7C_MEMINFO:-/proc/meminfo}  # tests only (tests/build_sh_test.sh): a fake meminfo drives the watchdog
avail=$(awk '/MemAvailable/ {print $2}' "$MEMINFO")
if [ "$avail" -lt 13000000 ]; then echo "build.sh: MemAvailable ${avail} kB < 13 GB: not compiling"; exit 4; fi
case "$OUT" in *dsv41-stage*) echo "build.sh: r7-comm2 compiles only under ~/climb-scratch (never into ~/dsv41-stage)"; exit 5;; esac
case "$OUT" in */climb-scratch/r7-comm2/*) ;; *) echo "build.sh: output must be under ~/climb-scratch/r7-comm2/"; exit 5;; esac
mkdir -p "$OUT"
cd "$OUT"
cat > build.ninja <<EOF
ninja_required_version = 1.3
cxx = c++
nvcc = /usr/local/cuda/bin/nvcc
cuda_cflags = -DTORCH_EXTENSION_NAME=rdma_ar_ext -DTORCH_API_INCLUDE_EXTENSION_H -isystem $TORCH/include -isystem $TORCH/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem $PYINC -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_121,code=sm_121 --compiler-options '-fPIC' -O3 -lineinfo -std=c++20
ldflags = -shared -L/usr/lib/aarch64-linux-gnu -l:libibverbs.so.1 -L$TORCH/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart
rule cuda_compile
  depfile = \$out.d
  deps = gcc
  command = \$nvcc -MD -MF \$out.d \$cuda_cflags -c \$in -o \$out
rule link
  command = \$cxx \$in \$ldflags -o \$out
build rdma_ar.cuda.o: cuda_compile $SRC/rdma_ar.cu
build rdma_ar_ext.so: link rdma_ar.cuda.o
default rdma_ar_ext.so
EOF
LOG=$OUT/build.log
echo "build.sh: $(date -u +%FT%TZ) $(hostname) src $(sha256sum $SRC/rdma_ar.cu | cut -c1-16) MemAvailable ${avail} kB" | tee "$LOG"
# the compile runs in its own session / process group: setsid execs env -> nice -> timeout -> ninja (nvcc, cicc, ptxas
# inherit the group), so pgid == $BPID (timeout's own setpgid is a no-op on a session leader, and on expiry it signals
# that same group), and a kill of -$BPID reaches every child and nothing else
setsid env PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES= nice -n 19 \
  timeout -k 5 900 $VENV/bin/ninja -j1 >> "$LOG" 2>&1 &
BPID=$!
echo "$BPID" > "$OUT/build.pgid"
# watchdog: stop THIS build (its process group only) if free memory falls under 8 GB (a relaunch needs ~109.5 GiB free)
( while kill -0 "$BPID" 2>/dev/null; do sleep 2; a=$(awk '/MemAvailable/ {print $2}' "$MEMINFO"); if [ "$a" -lt 8000000 ]; then
    echo "build.sh: MemAvailable $a kB < 8 GB: killing the build (pgid $BPID)" >> "$LOG"; kill -TERM -- "-$BPID" 2>/dev/null || true;
    sleep 3; kill -KILL -- "-$BPID" 2>/dev/null || true; exit 0; fi; done ) &
WD=$!
trap 'kill $WD 2>/dev/null || true; rm -f "$OUT/build.pgid"' EXIT
rc=0; wait "$BPID" || rc=$?
cat "$LOG"
rm -f rdma_ar.cuda.o rdma_ar.cuda.o.d .ninja_deps .ninja_log
if [ "$rc" -ne 0 ]; then echo "build.sh: ninja failed or was stopped (rc $rc)"; exit "$rc"; fi
S=$(sha256sum rdma_ar_ext.so | cut -d' ' -f1)
echo "build.sh: rdma_ar_ext.so sha256 $S" | tee -a "$LOG"
if [ -n "$EXPECT" ] && [ "$S" != "$EXPECT" ]; then echo "build.sh: sha256 differs from --expect $EXPECT"; exit 6; fi
