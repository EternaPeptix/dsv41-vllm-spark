#!/bin/bash
# pair.sh <label> <C|T> [DSV41_MOE_FC1_DEAL=deal|dealx] [other bundle VAR=VAL ...]: one pinned serving arm through the supervisor. RESTARTS THE
# SERVICE. Needs the site file installed on 8/8. C = control: flags unset. T = with the flag. Then on spark-one exact.py (a8c3394af9d2) and
# goal_single.py (G1, digest 9a18f802d930f6e7). Run CTTC from the Mac. WINDOW.md lists the G4 and long-prompt checks for a promotion.
set -euo pipefail
L=${1:?label}; ARM=${2:?C or T}; shift 2
[ "$ARM" = C ] && [ $# -gt 0 ] && { echo "C takes no flags"; exit 2; }
[ "$ARM" = T ] && [ $# -eq 0 ] && { echo "T needs flags"; exit 2; }
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
for h in $HOSTS; do ssh -n -o ConnectTimeout=10 "$h" "cd ~/dsv41-stage/g130/fc1 && cat PREP.ok >/dev/null && bash ops/site-node.sh status | grep -q INSTALLED" || { echo "ABORT: $h not staged+installed"; exit 1; }; done
S=$(mktemp -d "${TMPDIR:-/tmp}/g130fc1.XXXXXX"); trap 'rm -rf "$S"' EXIT
ssh -n spark-one 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > "$S/rel.sh"; [ -s "$S/rel.sh" ] || { echo "no supervisor"; exit 1; }
bash "$S/rel.sh" --label "g130-fc1-$L" --expect-launcher b3976c4b DSV41_GPU_CLOCK_MHZ=2000 "$@"
ssh -n spark-one "O=\$HOME/dsv41-g130ab/fc1/$L; mkdir -p \$O; cd ~/dsv41-bench && python3 exact.py \$O/exact.json $L > \$O/exact.log 2>&1; python3 -c \"import json,sys;print('exact',json.load(open(sys.argv[1]))['combined'][:12],'(want a8c3394af9d2)')\" \$O/exact.json; python3 -I goal/goal_single.py --out \$O/g1.jsonl --label $L > \$O/g1.log 2>&1; python3 -c \"import json,sys;r=json.loads(open(sys.argv[1]).read().splitlines()[-1]);print({k:r.get(k) for k in ('G1_mean_tok_s','outputs_digest','per_type')},'(want digest 9a18f802d930f6e7)')\" \$O/g1.jsonl; echo errors_rank0=\$(grep -c ' ERROR \|Traceback' ~/dsv41-logs/rank0.log)"
