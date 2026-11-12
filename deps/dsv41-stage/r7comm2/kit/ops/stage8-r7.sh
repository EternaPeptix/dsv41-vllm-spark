#!/bin/bash
# stage8-r7.sh: copy the kit (device tests, gate, prebuilt .so + mutants, installer) to ~/dsv41-stage/r7comm2/kit on 8/8.
# Touches no vllm file and no live .so. Verified per node with SHA256SUMS.stage.
set -euo pipefail
cd "$(dirname "$0")/.."
TS=$HOME/Documents/dsv41-kits/ts-ssh/bin
[ -x "$TS/ssh" ] && export PATH=$TS:$PATH
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
[ -f prebuilt/rdma_ar_ext.so ] || { echo "stage8-r7: no prebuilt/rdma_ar_ext.so (ops/build8-r7.sh first)"; exit 2; }
FILES=$(ls device_test7.py device_test6.py proj_emu.py shapes.py tests/device/graph_edges.py ops/*.sh gate/*.py \
  src/rdma_ar.cu src/mk_r7.py src/mk_mutants7.py build.sh install.py KIT.json r6/install.py r6/KIT.json \
  gate/results/SUMMARY.txt prebuilt/rdma_ar_ext.so prebuilt/mutants/*.so ref/triton/*/_proj_rows.cubin 2>/dev/null)
shasum -a 256 $FILES > SHA256SUMS.stage
for h in $HOSTS; do
  COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf - $FILES SHA256SUMS.stage | \
    ssh -o ConnectTimeout=10 "$h" 'mkdir -p ~/dsv41-stage/r7comm2/kit && cd ~/dsv41-stage/r7comm2/kit && nice -n 19 tar xzf - && sha256sum -c --quiet SHA256SUMS.stage && echo "$(hostname) staged OK $(sha256sum SHA256SUMS.stage | cut -c1-12)"'
done
