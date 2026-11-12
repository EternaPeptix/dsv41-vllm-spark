#!/usr/bin/env bash
# fleet.sh status|install|uninstall   (from the Mac; install/uninstall ONLY inside a window: climb-window.sh KIT_INSTALL /
# KIT_UNINSTALL, while CLUSTER_LOCK on spark-one has owner=claude)
#   - verifies the local kit pins (python3 install.py pins) before anything;
#   - status: copies nothing, runs the node-side state check read-only over ssh (python3 - < install.py with FIXB_VLLM);
#   - install/uninstall: refuses unless CLUSTER_LOCK exists with owner=claude; ships install.py + KIT.json + the r8 module
#     to ~/dsv41-stage/r8-fixb/ on each node, runs `install.py <cmd>` node by node, checks the state after each node;
#     on any failure every node already changed is put back (uninstall after a failed install, install after a failed
#     uninstall) and the script exits 1.
# env: FIXB_HOSTS (default the 8 Sparks), FIXB_TRANSPORT=ssh|local (local: each "host" is a directory holding
#      vllm/ and stage/, used by tests/test_fleet.sh), FIXB_LOCK_FILE (local transport: a lock file path)
set -u -o pipefail
export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
KIT=$(cd "$(dirname "$0")/.." && pwd)
CMD=${1:-status}
HOSTS=${FIXB_HOSTS:-"spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"}
TR=${FIXB_TRANSPORT:-ssh}
MOD=step/files/vllm/models/deepseek_v41/nvidia/dsv41_step.py
log() { echo "$(date -u +%T) [r8-fixb fleet] $*"; }
( cd "$KIT" && python3 install.py pins >/dev/null ) || { log "local kit files differ from KIT.json pins: refusing"; exit 2; }

run_node() {  # host cmd -> prints the node's output, returns its rc
  local h=$1 c=$2
  if [ "$TR" = local ]; then
    ( cd "$h/stage" && FIXB_VLLM="$h/vllm" python3 install.py "$c" )
  else
    ssh -n -o ConnectTimeout=15 "$h" "cd \$HOME/dsv41-stage/r8-fixb && /usr/bin/python3 install.py $c"
  fi
}
ship() {
  local h=$1
  if [ "$TR" = local ]; then
    mkdir -p "$h/stage/$(dirname $MOD)" && cp "$KIT/install.py" "$KIT/KIT.json" "$h/stage/" && cp "$KIT/$MOD" "$h/stage/$MOD"
  else
    ssh -n -o ConnectTimeout=15 "$h" "mkdir -p \$HOME/dsv41-stage/r8-fixb/$(dirname $MOD)" &&
      rsync -a "$KIT/install.py" "$KIT/KIT.json" "$h:dsv41-stage/r8-fixb/" &&
      rsync -a "$KIT/$MOD" "$h:dsv41-stage/r8-fixb/$MOD"
  fi
}
lock_ok() {
  if [ "$TR" = local ]; then grep -q '^owner=claude' "${FIXB_LOCK_FILE:-/nonexistent}" 2>/dev/null; return; fi
  ssh -n -o ConnectTimeout=15 spark-one 'grep -q "^owner=claude" ~/dsv41-opt-20261021/CLUSTER_LOCK' 2>/dev/null
}
node_status() {  # read-only: the state line only
  local h=$1
  if [ "$TR" = local ]; then
    ( cd "$KIT" && FIXB_VLLM="$h/vllm" python3 install.py status ) | awk 'NR==1{s=$1} /note:/{p=1} END{print s (p ? "(PRECOND-FAIL)" : "")}'
  else
    ssh -o ConnectTimeout=15 "$h" "cd /tmp && FIXB_VLLM=\$HOME/venvs/vllm032/lib/python3.12/site-packages/vllm /usr/bin/python3 - status" \
      < <(sed "s#^HERE = .*#HERE = '/nonexistent'#" "$KIT/install.py" | python3 "$KIT/ops/inline_kit.py" "$KIT") \
      | awk 'NR==1{s=$1} /note:/{p=1} END{print s (p ? "(PRECOND-FAIL)" : "")}'
  fi
}

case "$CMD" in
  status)
    rc=0
    for h in $HOSTS; do s=$(node_status "$h"); echo "$h: ${s:-UNREACHABLE}"; case "$s" in CLEAN|INSTALLED) ;; *) rc=1;; esac; done
    exit $rc ;;
  install|uninstall)
    lock_ok || { log "no CLUSTER_LOCK with owner=claude: refusing $CMD"; exit 2; }
    want=$([ "$CMD" = install ] && echo INSTALLED || echo CLEAN)
    back=$([ "$CMD" = install ] && echo uninstall || echo install)
    # preflight: every node must be in a state the command accepts
    for h in $HOSTS; do
      s=$(node_status "$h")
      case "$CMD:$s" in install:CLEAN|install:INSTALLED|uninstall:INSTALLED|uninstall:CLEAN) ;;
        *) log "$h is ${s:-UNREACHABLE}: refusing $CMD (nothing written)"; exit 2;; esac
    done
    done_hosts=""
    for h in $HOSTS; do
      if ship "$h" && run_node "$h" "$CMD" && [ "$(node_status "$h")" = "$want" ]; then
        log "$h: $want"; done_hosts="$done_hosts $h"
      else
        log "$h: $CMD FAILED; putting back:$done_hosts $h"
        for g in $done_hosts $h; do run_node "$g" "$back" >/dev/null 2>&1; log "  $g -> $(node_status "$g")"; done
        exit 1
      fi
    done
    log "DONE: $CMD on $(echo $HOSTS | wc -w | tr -d ' ') nodes" ;;
  *) sed -n 2,12p "$0"; exit 2 ;;
esac
