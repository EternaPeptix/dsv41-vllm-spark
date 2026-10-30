#!/bin/bash
# g1/m32-bw site install / uninstall / status on all 8 Sparks -- run from the Mac (bash 3.2 OK):
#   ssh spark-one cat ~/dsv41-stage/g1/m32-bw/ops/g1-site.sh | bash -s <install|uninstall|status>
# install  : per node (ops/g1-site-node.sh) refuses while dsv41-rank is active (FORCE=1 overrides), requires the
#            live file = 5acc5fde and PREP.ok, backs up sm121_target_m32.py.bak-g1-m32bw, installs the staged file
#            db659d80 (DSV41_MOE_M32_G1 unset = live behaviour) and verifies the sha.
# uninstall: restores the backup (5acc5fde) and verifies.  status: sha per host.
# Exit 0 only if 8/8 OK with one identical digest.
set -u
ACTION=${1:?install|uninstall|status}
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
FORCE=${FORCE:-0}
out=""; bad=0
for h in $HOSTS; do
  line=$(ssh -n -o ConnectTimeout=15 -o BatchMode=yes "$h" "bash \$HOME/dsv41-stage/g1/m32-bw/ops/g1-site-node.sh $ACTION $FORCE" 2>&1)
  echo "$line" | sed "s/^/[$h] /"
  last=$(echo "$line" | grep '^G1 ' | tail -1)
  case "$last" in "G1 OK "*) ;; *) bad=1;; esac
  out="$out $(echo "$last" | awk '{print $NF}')"
done
n=$(echo $out | tr ' ' '\n' | sort -u | wc -l | tr -d ' ')
if [ $bad = 0 ] && [ "$n" = 1 ]; then echo "G1-SITE $ACTION OK 8/8 identical $(echo $out | awk '{print substr($1,1,8)}')"; exit 0; fi
echo "G1-SITE $ACTION FAIL (bad=$bad, distinct=$n:$out)"; exit 1
