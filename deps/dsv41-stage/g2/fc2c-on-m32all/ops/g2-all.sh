#!/bin/bash
# g2-all.sh <install|uninstall|status|build>  -- run from the Mac (bash 3.2 OK), all 8 Sparks:
#   install|uninstall|status: ops/g2-site.sh on 8/8, requires the same end-state digest on 8/8
#   build:                    ops/prep-node.sh build on 8/8 (CPU only, nice 19; tree + pins + CPU test -> PREP.ok)
# Usage: ssh spark-one cat ~/dsv41-stage/g2/fc2c-on-m32all/ops/g2-all.sh | bash -s status
set -u
ACTION=${1:?action}
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
out=""; bad=0
for h in $HOSTS; do
  if [ "$ACTION" = build ]; then
    line=$(ssh -n -o ConnectTimeout=15 -o BatchMode=yes "$h" "bash \$HOME/dsv41-stage/g2/fc2c-on-m32all/ops/prep-node.sh build" 2>&1)
    last=$(echo "$line" | grep '^PREP ' | tail -1)
    tag=$(echo "$last" | awk '{print $4}')        # PREP.ok digest
  else
    line=$(ssh -n -o ConnectTimeout=15 -o BatchMode=yes "$h" "G2_ALLOW_ACTIVE=${G2_ALLOW_ACTIVE:-0} bash \$HOME/dsv41-stage/g2/fc2c-on-m32all/ops/g2-site.sh $ACTION" 2>&1)
    last=$(echo "$line" | grep '^G2 ' | tail -1)
    tag=$(echo "$last" | awk '{print $NF}')
  fi
  echo "$line" | sed "s/^/[$h] /"
  case "$last" in *" OK "*) ;; *) bad=1;; esac
  out="$out $tag"
done
n=$(echo $out | tr ' ' '\n' | sort -u | wc -l | tr -d ' ')
if [ $bad = 0 ] && [ "$n" = 1 ]; then echo "G2-ALL fc2c-on-m32all $ACTION OK 8/8 identical digest $(echo $out | awk '{print $1}')"; exit 0; fi
echo "G2-ALL fc2c-on-m32all $ACTION FAIL (bad=$bad, distinct digests=$n:$out)"; exit 1
