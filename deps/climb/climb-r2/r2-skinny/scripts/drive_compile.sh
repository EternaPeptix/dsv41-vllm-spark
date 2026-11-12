#!/bin/bash
# Mac-side driver: lock check -> copy sources -> start build on spark-eight -> watch lock -> fetch log.
export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
KIT=$(cd "$(dirname "$0")/.." && pwd)
SP=$KIT/scripts; OUT=${OUT:-$KIT/build-out}  # compile_spark8.sh lives next to this script
lock() { ssh -n -o ConnectTimeout=15 spark-one 'test -e ~/dsv41-opt-20261021/CLUSTER_LOCK && echo LOCKED || echo FREE' 2>/dev/null; }
L=$(lock); echo "lock: $L $(date -u +%FT%TZ)"
[ "$L" = "FREE" ] || { echo "ABORT: lock"; exit 3; }
ssh -n spark-eight 'mkdir -p ~/climb-scratch/r2-skinny/kernel' || exit 2
scp -q $KIT/kernel/skinny_mx_v2.cu $KIT/kernel/build.py spark-eight:climb-scratch/r2-skinny/kernel/ || exit 2
scp -q $SP/compile_spark8.sh spark-eight:climb-scratch/r2-skinny/compile_spark8.sh || exit 2
ssh -n spark-eight 'cd ~/climb-scratch/r2-skinny && sha256sum kernel/skinny_mx_v2.cu kernel/build.py && nohup bash compile_spark8.sh > compile.out 2>&1 < /dev/null & echo started'
for i in $(seq 1 120); do
  sleep 10
  if [ "$(lock)" = "LOCKED" ]; then
    echo "LOCK APPEARED $(date -u +%FT%TZ): killing build"
    ssh -n spark-eight 'pkill -f climb-scratch/r2-skinny; pkill -f r2-skinny/kernel; rm -rf ~/climb-scratch/r2-skinny/kernel/build; echo killed'
    exit 4
  fi
  if ssh -n spark-eight 'grep -q "^build end" ~/climb-scratch/r2-skinny/kernel/build.log'; then break; fi
done
mkdir -p $OUT
scp -q spark-eight:climb-scratch/r2-skinny/kernel/build.log $OUT/build.log
scp -q 'spark-eight:climb-scratch/r2-skinny/kernel/build/*.so' $OUT/ 2>/dev/null
ssh -n spark-eight 'ls -la ~/climb-scratch/r2-skinny/kernel/build/'
grep -E "BUILD_RC|build (start|end)|BUILD_OK" $OUT/build.log
shasum -a 256 $OUT/*.so
ssh -n spark-eight 'rm -rf ~/climb-scratch/r2-skinny/kernel/build && echo "spark-eight build dir removed"'  # brief rule 4: no large leftovers
