#!/bin/bash
# Compile check (Triton -> PTX/cubin for sm_121, no GPU use) + Triton-interpreter run of the real kernel source, on
# spark-eight only, outside measurement windows (rule 4). Run from this Mac in the kit folder:
#     bash tools/run_spark8_checks.sh
# Results land in results/compile.log, results/interp.log and ptx/*.ptx; the Triton cache on spark-eight is deleted.
# The CLUSTER_LOCK on spark-one is checked before each step and every 20 s while a step runs; if it appears the remote
# step is killed (pkill on its exact script path) and the script exits 3. Exit 2 = lock present at start.
set -u
export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
KIT=$(cd "$(dirname "$0")/.." && pwd)
RDIR='~/climb-scratch/r2-moefront'
lock() { ssh -n -o ConnectTimeout=15 spark-one 'test -e ~/dsv41-opt-20261021/CLUSTER_LOCK'; }

# run_step <name> <remote command> <log>: runs on spark-eight in the background, polls the lock, kills on lock
run_step() {
    local name=$1 cmd=$2 log=$3
    ssh -n spark-eight "cd $RDIR && $cmd" > "$log" 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
        sleep 20
        if lock; then
            echo "CLUSTER_LOCK appeared during $name: killing it" | tee -a "$log"
            ssh -n spark-eight "pkill -f '[c]limb-scratch/r2-moefront/tools/$name'" || true
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
rsync -a --delete --exclude base --exclude ref --exclude results --exclude ptx --exclude '__pycache__' \
    "$KIT/" spark-eight:climb-scratch/r2-moefront/ || exit 4
mkdir -p "$KIT/results" "$KIT/ptx"
{ echo "# $(date -u +%FT%TZ) spark-eight; kit module sha256:"; shasum -a 256 \
    "$KIT/files/vllm/model_executor/layers/fused_moe/dsv41_routeprep.py"; } > "$KIT/results/compile.header"
run_step compile_check.py "CUDA_VISIBLE_DEVICES= TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CACHE_DIR=\$PWD/tcache timeout 900 nice -n 19 ~/venvs/vllm032/bin/python \$HOME/climb-scratch/r2-moefront/tools/compile_check.py" \
    "$KIT/results/compile.log"
rc1=$?
cat "$KIT/results/compile.header" "$KIT/results/compile.log" > "$KIT/results/compile.log.tmp" && \
    mv "$KIT/results/compile.log.tmp" "$KIT/results/compile.log" && rm -f "$KIT/results/compile.header"
rsync -a spark-eight:climb-scratch/r2-moefront/ptx/ "$KIT/ptx/" 2>/dev/null
if [ $rc1 -eq 3 ]; then ssh -n spark-eight "rm -rf $RDIR/tcache $RDIR/ptx"; exit 3; fi
if lock; then echo "lock appeared: stopping before the interpreter run" | tee -a "$KIT/results/compile.log"; exit 3; fi
run_step interp_test.py "CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 TRITON_CACHE_DIR=\$PWD/tcache timeout 1800 \
    nice -n 19 ~/venvs/vllm032/bin/python \$HOME/climb-scratch/r2-moefront/tools/interp_test.py" "$KIT/results/interp.log"
rc2=$?
ssh -n spark-eight "rm -rf $RDIR/tcache $RDIR/ptx; du -sh $RDIR"
echo "compile_check rc=$rc1 interp_test rc=$rc2"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
