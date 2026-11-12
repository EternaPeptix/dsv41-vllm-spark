#!/bin/bash
# install8-r7.sh install|uninstall|status: install.py on 8/8 from the staged kit (ops/stage8-r7.sh), node by node;
# install needs CLUSTER_LOCK held by owner=claude (a window); a failure uninstalls every touched node and verifies.
set -uo pipefail
cd "$(dirname "$0")/.."
TS=$HOME/Documents/dsv41-kits/ts-ssh/bin
[ -x "$TS/ssh" ] && export PATH=$TS:$PATH
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
CMD=${1:?install|uninstall|status}
K='cd ~/dsv41-stage/r7comm2/kit && sha256sum -c --quiet SHA256SUMS.stage && R7C_R6_INSTALL=$HOME/dsv41-stage/r7comm2/kit/r6/install.py python3 -I install.py'
if [ "$CMD" != status ]; then
  own=$(ssh -n -o ConnectTimeout=10 spark-one 'grep -x "owner=claude" ~/dsv41-opt-20261021/CLUSTER_LOCK 2>/dev/null' || true)
  [ "$own" = owner=claude ] || { echo "install8-r7: $CMD needs CLUSTER_LOCK held by owner=claude (a window); refusing"; exit 3; }
fi
done_hosts=""
for h in $HOSTS; do
  echo "== $h"
  if ! ssh -n -o ConnectTimeout=15 "$h" "$K $CMD"; then
    echo "install8-r7: $CMD failed on $h"
    if [ "$CMD" = install ]; then
      # R7C_ROLLBACK=downgrade when r7 is already PROD (C1 promoted 2026-11-11, launcher 58899745 loads
      # ~/dsv41-stage/r7comm2/build/rdma_ar_ext.so): install is then an in-place upgrade of the prior patch, and
      # `uninstall` would rmtree that build dir under the serving launcher. downgrade restores the prior patch
      # byte-exact and keeps build/ + state.json. (The failing host was already restored by install.py itself;
      # its downgrade then refuses harmlessly.) Default stays uninstall (fresh install on a clean node).
      RB=${R7C_ROLLBACK:-uninstall}
      for d in $done_hosts $h; do ssh -n "$d" "$K $RB" || echo "install8-r7: rollback ($RB) on $d FAILED or not needed"; done
      [ "$RB" = uninstall ] && { bash ops/verify8-r7.sh --uninstalled || true; }
    fi
    exit 1
  fi
  done_hosts="$done_hosts $h"
done
case $CMD in install) bash ops/verify8-r7.sh;; uninstall) bash ops/verify8-r7.sh --uninstalled;; esac
