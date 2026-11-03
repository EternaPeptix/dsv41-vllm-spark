#!/bin/bash
set -euo pipefail
root="$HOME/Athena Hillclimbing stuff/moet-fc2packed-snapshot"
short_root="/tmp/athena-fc2-graph-$(id -u)"
if [ -L "$short_root" ]; then
  [ "$(readlink -f "$short_root")" = "$(readlink -f "$root")" ] || exit 31
elif [ -e "$short_root" ]; then
  exit 32
else
  ln -s "$root" "$short_root"
fi
export CUDA_HOME=/usr/local/cuda
export PATH="$HOME/venvs/vllm032/bin:/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=1
python "$short_root/moet-baseline-research/build_fc2_graph_binding.py" --build-dir "$short_root/fc2-graph-build" --run-experimental-build
