#!/usr/bin/env bash
# stage.sh (WINDOW ONLY: called from the device plan while device-window.sh holds CLUSTER_LOCK)
#   stage.sh stage        copy the kit to ~/climb-scratch/r8-fixb/ on $HEAD_NODE / $STEP_NODE / $SKEW_NODE, verify
#                         every pinned file there (install.py pins == KIT.json), then write .staged (plan lines wait on it)
#   stage.sh collect DIR  copy the device tests' JSON results back (read-only on the nodes)
#   stage.sh clean        remove ~/climb-scratch/r8-fixb on the three nodes
# env: HEAD_NODE (spark-two), STEP_NODE (spark-three), SKEW_NODE (spark-four): the r6-dwA lanes
set -u -o pipefail
export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
KIT=$(cd "$(dirname "$0")/.." && pwd)
HEAD_NODE=${HEAD_NODE:-spark-two}; STEP_NODE=${STEP_NODE:-spark-three}; SKEW_NODE=${SKEW_NODE:-spark-four}
NODES=$(printf '%s\n' "$HEAD_NODE" "$STEP_NODE" "$SKEW_NODE" | sort -u)
case "${1:-}" in
  stage)
    ( cd "$KIT" && python3 install.py pins >/dev/null ) || { echo "LOCAL kit files differ from KIT.json pins"; exit 3; }
    rc=0
    for n in $NODES; do
      ssh -n -o ConnectTimeout=15 "$n" "rm -f \$HOME/climb-scratch/r8-fixb/.staged; mkdir -p \$HOME/climb-scratch/r8-fixb" || { rc=4; continue; }
      rsync -a --delete --exclude __pycache__ --exclude live --exclude results --exclude out --exclude tcache \
        --exclude 'head/results-r8' --exclude 'skew/analysis' --exclude 'skew/window' --exclude 'step/live' \
        --exclude 'scripts' "$KIT/" "$n:climb-scratch/r8-fixb/" || { rc=4; continue; }
      if ssh -n -o ConnectTimeout=15 "$n" "cd \$HOME/climb-scratch/r8-fixb && /usr/bin/python3 install.py pins >/dev/null && mkdir -p head/results step/results && date -u +%FT%TZ > .staged"; then
        echo "staged r8-fixb on $n: $(python3 -c "import json;print(len(json.load(open('$KIT/KIT.json'))['pins']))") pinned files verified"
      else
        echo "STAGED PINS DIFFER on $n"; rc=5
      fi
    done
    exit $rc ;;
  collect)
    d=${2:?dir}; mkdir -p "$d"
    scp -q "$HEAD_NODE:climb-scratch/r8-fixb/head/results/*.json" "$d/" 2>/dev/null || echo "no head JSON on $HEAD_NODE"
    ls "$d"/*.json 2>/dev/null ;;
  clean) for n in $NODES; do ssh -n -o ConnectTimeout=15 "$n" "rm -rf \$HOME/climb-scratch/r8-fixb"; done ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac
