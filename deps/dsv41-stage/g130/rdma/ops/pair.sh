#!/bin/bash
# pair.sh <label> <C|T> [RDMA_AR_x=1 ...]: one pinned serving arm through the supervisor. RESTARTS THE SERVICE (the launcher's default AV
# pin, GPU clock 2000). C = control: the kit's .so with every flag unset. T = the same .so with the flags given. Then, on spark-one: exact.py
# (anchor a8c3394af9d2) and goal_single.py (G1 + outputs_digest, must be 9a18f802d930f6e7). Run ABBA from WINDOW.md. Run it from the Mac.
set -euo pipefail
L=${1:?label}; ARM=${2:?C or T}; shift 2
[ "$ARM" = C ] && [ $# -gt 0 ] && { echo "C takes no flags"; exit 2; }
[ "$ARM" = T ] && [ $# -eq 0 ] && { echo "T needs flags"; exit 2; }
for kv in "$@"; do case "$kv" in RDMA_AR_SPLIT_POST=1|RDMA_AR_PUB_V1=1|RDMA_AR_PUB_V2=1|RDMA_AR_HYG_PROXY=1|RDMA_AR_HYG_POLL=1) ;; *) echo "not a g130 flag: $kv"; exit 2 ;; esac; done
EXT=/home/jeweled/dsv41-stage/g130/rdma/build/rdma_ar_ext.so
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
for h in $HOSTS; do ssh -n -o ConnectTimeout=10 "$h" "cd ~/dsv41-stage/g130/rdma && sha256sum -c --quiet SHA256SUMS" || { echo "ABORT: $h fails SHA256SUMS"; exit 1; }; done
S=$(mktemp -d "${TMPDIR:-/tmp}/g130pair.XXXXXX"); trap 'rm -rf "$S"' EXIT
ssh -n spark-one 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > "$S/rel.sh"; [ -s "$S/rel.sh" ] || { echo "no supervisor"; exit 1; }
bash "$S/rel.sh" --label "g130-$L" --expect-launcher 37757dde RDMA_AR_EXT=$EXT DSV41_GPU_CLOCK_MHZ=2000 "$@"
ssh -n spark-one "O=\$HOME/dsv41-g130ab/rdma/$L; mkdir -p \$O; grep -E 'rdma_ar (ext|fingerprint)' ~/dsv41-logs/rank0.log | tail -2 | cut -c1-200; cd ~/dsv41-bench && python3 exact.py \$O/exact.json $L > \$O/exact.log 2>&1; python3 -c \"import json,sys;print('exact',json.load(open(sys.argv[1]))['combined'][:12],'(want a8c3394af9d2)')\" \$O/exact.json; python3 -I goal/goal_single.py --out \$O/g1.jsonl --label $L > \$O/g1.log 2>&1; python3 -c \"import json,sys;r=json.loads(open(sys.argv[1]).read().splitlines()[-1]);print({k:r.get(k) for k in ('G1','g1','outputs_digest','per_type')},'(want digest 9a18f802d930f6e7)')\" \$O/g1.jsonl; echo errors_rank0=\$(grep -c ' ERROR \|Traceback' ~/dsv41-logs/rank0.log)"
