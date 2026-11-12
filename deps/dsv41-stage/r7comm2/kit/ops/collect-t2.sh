#!/bin/bash
# collect-t2.sh: (re)run the N1/N2 SASS collection on spark-eight and the binary checks (no window; read-only on the
# node: cuobjdump streams to this Mac; nothing is written on any Spark).
# Review B2 fix: the CLUSTER_LOCK lives on spark-one only, and spark-eight cannot ssh to spark-one, so the old per-dump
# check inside collect_sass.py (run on spark-eight) could never fire. audit/collect_driver.py now runs the collection one
# step per ssh call (each scan root, each cuobjdump) and asks spark-one FROM THIS MAC before every step and every 15 s
# while a step runs; anything but FREE (LOCKED / no answer) stops it, tears the step down, and the worker kills its
# cuobjdump at its next heartbeat (<= 5 s). Exit 3 = stopped on the lock (rerun later; the partial output is kept as
# audit/out/t2_sass.jsonl.partial and NOT used).
# The 2026-11-09 run was stopped by hand when the r5-dw window took the lock: the vllm _C / _moe_C / cuBLASLt fat
# binaries were not reached, which is why Kernel2, act_and_mul, fusedDeepseekV4QNormRopeKV are NO-SASS (N2 OPEN), and
# the flashinfer MXFP8 quant (CuTe DSL) kernel has no on-disk cubin in the searched roots (N1 / N2 OPEN).
set -euo pipefail
cd "$(dirname "$0")/.."
TS=$HOME/Documents/dsv41-kits/ts-ssh/bin
[ -x "$TS/ssh" ] && export PATH=$TS:$PATH
set +e
python3 -I audit/collect_driver.py --names audit/out/mangled.json --out audit/out/t2_sass.jsonl.new 2> audit/out/collect.err
rc=$?
set -e
if [ $rc != 0 ]; then
  mv audit/out/t2_sass.jsonl.new audit/out/t2_sass.jsonl.partial
  echo "collect-t2: stopped (rc $rc): $(tail -1 audit/out/collect.err)"; exit $rc
fi
gzip -9 -c audit/out/t2_sass.jsonl.new > audit/out/t2_sass.jsonl.gz && rm audit/out/t2_sass.jsonl.new
python3 -I gate/acqcheck.py --sass audit/out/t2_sass.jsonl.gz --pdl audit/out/t2_pdl.json --json audit/out/acqcheck.json --selftest | tee audit/out/acqcheck.txt
