#!/bin/bash
# build.sh (g2 colltrim) — run FROM THE MAC with bash from anywhere. CPU-only work on spark-four (nice -n 19,
# CUDA_VISIBLE_DEVICES empty, safe while serving): copies the sources, builds build/rdma_ar_ext.so and
# build/libdsv41_l2pf_g2.so (native/build_g2.py), runs the SASS gate against the live k1 build and the CPU
# planner test, and brings the two .so files and the logs back into this package (results/).
set -euo pipefail
cd "$(dirname "$0")/.."
B=${BUILD_HOST:-spark-four}
D='$HOME/dsv41-stage/g2/colltrim'
ssh -n "$B" "mkdir -p $D/results"
COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf - native site base tests bench ops diffs | ssh "$B" "cd $D && tar xzf -"
ssh -n "$B" "cd $D && env CUDA_HOME=/usr/local/cuda PATH=\$HOME/venvs/vllm032/bin:/usr/local/cuda/bin:\$PATH \
  CUDA_VISIBLE_DEVICES= nice -n 19 \$HOME/venvs/vllm032/bin/python native/build_g2.py > results/build.log 2>&1 \
  || { tail -20 results/build.log; exit 1; }; tail -1 results/build.log"
ssh -n "$B" "cd $D && nice -n 19 \$HOME/venvs/vllm032/bin/python tests/sass_check_g2.py \
  \$HOME/rdma_ar_dsv41/build-k1pl/rdma_ar_ext.so build/rdma_ar_ext.so > results/sass_check.txt 2>&1; tail -1 results/sass_check.txt"
ssh -n "$B" "cd $D && CUDA_VISIBLE_DEVICES= nice -n 19 \$HOME/venvs/vllm032/bin/python tests/test_plan_g2_cpu.py \
  > results/test_plan_g2_cpu.log 2>&1; tail -1 results/test_plan_g2_cpu.log"
ssh -n "$B" "cd $D && CUDA_VISIBLE_DEVICES= nice -n 19 \$HOME/venvs/vllm032/bin/python tests/test_cols_logic_cpu.py \
  > results/test_cols_logic_cpu.log 2>&1; tail -1 results/test_cols_logic_cpu.log"
mkdir -p build results
scp -q "$B:dsv41-stage/g2/colltrim/build/rdma_ar_ext.so" "$B:dsv41-stage/g2/colltrim/build/libdsv41_l2pf_g2.so" build/
scp -q "$B:dsv41-stage/g2/colltrim/results/build.log" "$B:dsv41-stage/g2/colltrim/results/sass_check.txt" \
  "$B:dsv41-stage/g2/colltrim/results/test_plan_g2_cpu.log" "$B:dsv41-stage/g2/colltrim/results/test_cols_logic_cpu.log" results/
shasum -a 256 build/*.so
