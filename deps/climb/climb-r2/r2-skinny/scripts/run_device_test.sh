#!/bin/bash
# r2-skinny: run device_test.py on a STOPPED node (lead only, inside a window; brief rule 3). From the Mac:
#   bash scripts/run_device_test.sh [node=spark-eight] [rank=7]
# Copies the kit (incl. prebuilt/skinny_mx_v2_ext.so) to ~/climb-scratch/r2-skinny on the node, refuses if any
# vLLM process or any GPU compute process is running there, runs the full test with real shards, fetches the log + json into
# tests/device-<node>-<UTC>.{log,json}. Exit code = device_test.py's (0 = every case bit-identical).
set -u
NODE=${1:-spark-eight}; RANK=${2:-7}
KIT=$(cd "$(dirname "$0")/.." && pwd)
export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
TS=$(date -u +%Y%m%dT%H%M%SZ)
if ssh -n "$NODE" 'pgrep -f "[v]llm serve|[E]ngineCore|[V]LLM::Worker" >/dev/null || nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .'; then
  echo "REFUSING: a vLLM process is running on $NODE (the device test needs a stopped node)"; exit 2
fi
ssh -n "$NODE" 'mkdir -p ~/climb-scratch/r2-skinny' || exit 2
tar --exclude='._*' --exclude='tests/*.log' -C "$KIT" -czf - device_test.py KIT.json new prebuilt | \
  ssh "$NODE" 'cd ~/climb-scratch/r2-skinny && tar xzf - && sha256sum prebuilt/skinny_mx_v2_ext.so' || exit 2
# r4: the serving environment (venv bin = ninja, /usr/local/cuda/bin, CUDA_HOME): without it vLLM's has_flashinfer() is
# False and the live quant reference silently becomes the torch fallback (device_test.py now exits 2 in that case).
ssh -n "$NODE" "cd ~/climb-scratch/r2-skinny && . ~/venvs/vllm032/bin/activate && export PATH=/usr/local/cuda/bin:\$PATH CUDA_HOME=/usr/local/cuda && nice -n 5 ~/venvs/vllm032/bin/python device_test.py \
  --model ~/models/dsv41/official-fp8 --layers 0,13,39 --rank $RANK --json dt-$TS.json > dt-$TS.log 2>&1; \
  echo EXIT=\$? >> dt-$TS.log"
scp -q "$NODE:climb-scratch/r2-skinny/dt-$TS.log" "$KIT/tests/device-$NODE-$TS.log"
scp -q "$NODE:climb-scratch/r2-skinny/dt-$TS.json" "$KIT/tests/device-$NODE-$TS.json" 2>/dev/null
tail -25 "$KIT/tests/device-$NODE-$TS.log"
RC=$(grep -o 'EXIT=[0-9]*' "$KIT/tests/device-$NODE-$TS.log" | tail -1 | cut -d= -f2)
exit "${RC:-1}"
