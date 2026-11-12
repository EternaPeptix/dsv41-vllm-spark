#!/bin/bash
# dev8-r7.sh <name> [<name> ...]: device-only (torch-free) cubin compiles of rdma_ar sources on spark-eight while the service
# runs, for SASS evidence (gate/device_tu7.py). name = ctl_r6 (r6 src 2bff8a89, method control) | r7 (src/rdma_ar.cu) | m_* (mutants).
# Each source -> 7 TUs (pingpong + launch_kernel<mode, f32> x 6), each compiled alone under ops/guard_run.sh: nice 19, own
# session, CUDA_VISIBLE_DEVICES= (no GPU), killed as soon as MemAvailable < GUARD_FLOOR_KB (default here 8500000 kB), refused
# below floor + 300 MB, `timeout 600`. Positive FREE from spark-one before every TU (lock never held while compiling).
# Output: sass/dev-<name>.sass (cuobjdump -sass of the 7 cubins, concatenated), build-logs/dev8-<UTC>.log. Never writes
# outside ~/climb-scratch/r7-comm2/ on spark-eight.
set -euo pipefail
cd "$(dirname "$0")/.."
TS=$HOME/Documents/dsv41-kits/ts-ssh/bin
[ -x "$TS/ssh" ] && export PATH=$TS:$PATH
B=spark-eight
W='$HOME/climb-scratch/r7-comm2'
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p build-logs sass
LOG=build-logs/dev8-$STAMP.log
FLOOR=${GUARD_FLOOR_KB:-8500000}
# DW_TOKEN (exported by climb-common/device-window.sh for its MAC_PLAN steps): inside the lead's own STOPPED device
# window the lock carries that token; the service is stopped, so a build disturbs nothing and has the memory it needs.
lock() { ssh -n -o ConnectTimeout=8 -o BatchMode=yes spark-one \
  "if [ -e ~/dsv41-opt-20261021/CLUSTER_LOCK ]; then if [ -n '${DW_TOKEN:-}' ] && grep -qx 'token=${DW_TOKEN:-}' ~/dsv41-opt-20261021/CLUSTER_LOCK; then echo FREE; else echo LOCKED; fi; else echo FREE; fi" 2>/dev/null || echo NOANSWER; }
srcof() { case $1 in r7) echo src/rdma_ar.cu;; ctl_r6) echo ref/rdma_ar.r6-2bff8a89.cu;; *) echo src/mutants/rdma_ar.$1.cu;; esac; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cp ops/guard_run.sh "$TMP/"
for n in "$@"; do python3 -I gate/device_tu7.py "$(srcof $n)" "$TMP/$n" | tee -a "$LOG"; done
( cd "$TMP" && COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf - . ) | ssh "$B" "mkdir -p $W/dev && cd $W/dev && tar xzf -"
for n in "$@"; do
  : > "sass/dev-$n.sass.tmp"; : > "sass/dev-$n.resusage"
  for tu in pp os_bf16 os_f32 ts_bf16 ts_f32 ag agc; do
    l=$(lock); [ "$l" = FREE ] || { echo "dev8-r7: lock state $l: stopping before $n/$tu" | tee -a "$LOG"; exit 3; }
    r=$(ssh -n "$B" "cd $W/dev/$n && a=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo); \
      if [ \$a -lt $((FLOOR + 300000)) ]; then echo NOSTART \$a; exit 0; fi; \
      GUARD_FLOOR_KB=$FLOOR timeout -k 5 600 bash ../guard_run.sh $tu.log -- /usr/local/cuda/bin/nvcc -cubin -arch=compute_121 -code=sm_121 \
        -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
        --expt-relaxed-constexpr -O3 -lineinfo -std=c++20 $tu.cu -o $tu.cubin > /dev/null 2>&1; echo rc=\$? \$(tail -1 $tu.log)")
    echo "dev8-r7: $n/$tu $r" | tee -a "$LOG"
    case "$r" in rc=0*) ;; *) echo "dev8-r7: $n/$tu did not build; stopping" | tee -a "$LOG"; exit 4;; esac
    ssh -n "$B" "nice -n 19 /usr/local/cuda/bin/cuobjdump -sass $W/dev/$n/$tu.cubin" >> "sass/dev-$n.sass.tmp"
    # r7: registers / local memory per kernel (a spill in r7_project would put LDL/STL into the projection dataflow,
    # which gate/projgate.py refuses: fail closed; this file says why)
    ssh -n "$B" "nice -n 19 /usr/local/cuda/bin/cuobjdump -res-usage $W/dev/$n/$tu.cubin" >> "sass/dev-$n.resusage"
  done
  mv "sass/dev-$n.sass.tmp" "sass/dev-$n.sass"
  echo "dev8-r7: $n -> sass/dev-$n.sass ($(grep -c 'Function :' sass/dev-$n.sass) kernels)" | tee -a "$LOG"
done
