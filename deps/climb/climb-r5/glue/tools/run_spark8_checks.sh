#!/bin/bash
# r5-glue: compile check (Triton -> PTX/cubin/SASS for sm_121 in production's specialization, no GPU use) + Triton-interpreter run of the real kernel
# sources, on spark-eight only, outside measurement windows. Run from this Mac in the kit folder:
#     bash tools/run_spark8_checks.sh
# Results: results/compile.log, results/interp.log, results/ptx/*. The Triton cache on spark-eight is deleted.
# The CLUSTER_LOCK on spark-one is checked before each step and every 20 s while a step runs; if it appears the remote
# step is killed and the script exits 3. Exit 2 = lock present at start (nothing done).
set -u
export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
KIT=$(cd "$(dirname "$0")/.." && pwd)
RDIR='~/climb-scratch/r5-glue'
lock() { ssh -n -o ConnectTimeout=15 spark-one 'test -e ~/dsv41-opt-20261021/CLUSTER_LOCK'; }

# the mandatory device-test environment (BRIEF r5), sourced for these CPU-only steps as well (review N8)
ENVSRC='source ~/venvs/vllm032/bin/activate && export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda && \
    echo "env: $(command -v python) CUDA_HOME=$CUDA_HOME ptxas=$(command -v ptxas)"'

run_step() {
    local name=$1 cmd=$2 log=$3
    ssh -n spark-eight "$ENVSRC && cd $RDIR && $cmd" > "$log" 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
        sleep 20
        if lock; then
            echo "CLUSTER_LOCK appeared during $name: killing it" | tee -a "$log"
            ssh -n spark-eight "pkill -f '[c]limb-scratch/r5-glue/tools/$name'" || true
            wait $pid 2>/dev/null
            return 3
        fi
    done
    wait $pid
    local rc=$?
    echo "$name exit $rc" | tee -a "$log"
    return $rc
}

if lock; then echo "CLUSTER_LOCK present: not compiling" >&2; exit 2; fi
ssh -n spark-eight "mkdir -p $RDIR" || exit 4
rsync -a --delete --exclude live --exclude results --exclude '__pycache__' "$KIT/" spark-eight:climb-scratch/r5-glue/ || exit 4
mkdir -p "$KIT/results/ptx"
hdr="# $(date -u +%FT%TZ) spark-eight; kit module sha256: $(shasum -a 256 "$KIT"/files/vllm/models/deepseek_v4/nvidia/ops/dsv41_r5g_rq.py "$KIT"/files/vllm/model_executor/layers/fused_moe/dsv41_r5g_rp.py | awk '{print substr($1,1,16)}' | tr '\n' ' ')"
run_step compile_check.py "CUDA_VISIBLE_DEVICES= TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CACHE_DIR=\$PWD/tcache timeout 2400 nice -n 19 ~/venvs/vllm032/bin/python \$HOME/climb-scratch/r5-glue/tools/compile_check.py" \
    "$KIT/results/compile.log"
rc1=$?
{ echo "$hdr"; cat "$KIT/results/compile.log"; } > "$KIT/results/compile.log.tmp" && mv "$KIT/results/compile.log.tmp" "$KIT/results/compile.log"
rsync -a spark-eight:climb-scratch/r5-glue/results/ptx/ "$KIT/results/ptx/" 2>/dev/null
if [ $rc1 -eq 3 ]; then ssh -n spark-eight "rm -rf $RDIR/tcache $RDIR/results"; exit 3; fi
if lock; then echo "lock appeared: stopping before the interpreter run" | tee -a "$KIT/results/compile.log"; exit 3; fi
run_step interp_test.py "CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 TRITON_CACHE_DIR=\$PWD/tcache timeout 1800 \
    nice -n 19 ~/venvs/vllm032/bin/python \$HOME/climb-scratch/r5-glue/tools/interp_test.py" "$KIT/results/interp.log"
rc2=$?
{ echo "$hdr"; cat "$KIT/results/interp.log"; } > "$KIT/results/interp.log.tmp" && mv "$KIT/results/interp.log.tmp" "$KIT/results/interp.log"
ssh -n spark-eight "rm -rf $RDIR/tcache $RDIR/results; du -sh $RDIR"
echo "compile_check rc=$rc1 interp_test rc=$rc2"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
