#!/usr/bin/env bash
# Mac-side: build on spark-eight only while no CLUSTER_LOCK exists on spark-one; kill the build the
# moment a lock appears. Usage: bash ops/build_guarded.sh <variant>...   (see src/build.sh)
set -u
export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
KIT=$(cd "$(dirname "$0")/.." && pwd)
locked() { ssh -n spark-one 'test -e ~/dsv41-opt-20261021/CLUSTER_LOCK'; }
if locked; then echo "CLUSTER_LOCK present: not building"; exit 3; fi
COPYFILE_DISABLE=1 tar cf - -C "$KIT" src tools | ssh spark-eight 'mkdir -p ~/climb-scratch/r3-xsplit && cd ~/climb-scratch/r3-xsplit && tar xf - 2>/dev/null; find . -name "._*" -delete'
ssh -n spark-eight "cd ~/climb-scratch/r3-xsplit && rm -f build.done build.rc build.pgid && setsid bash -c 'echo \$\$ > build.pgid; nice -n 19 bash src/build.sh build $* > build.log 2>&1; echo \$? > build.rc; touch build.done' > /dev/null 2>&1 < /dev/null &"
while :; do
  sleep 15
  if locked; then
    echo "CLUSTER_LOCK appeared: killing the build"
    ssh -n spark-eight 'cd ~/climb-scratch/r3-xsplit && kill -TERM -- -$(cat build.pgid) 2>/dev/null; sleep 2; kill -KILL -- -$(cat build.pgid) 2>/dev/null; pgrep -g $(cat build.pgid) >/dev/null && echo STILL-RUNNING || echo KILLED'
    exit 4
  fi
  if ssh -n spark-eight 'test -e ~/climb-scratch/r3-xsplit/build.done'; then
    ssh -n spark-eight 'cd ~/climb-scratch/r3-xsplit && echo rc=$(cat build.rc) && tail -20 build.log'
    exit 0
  fi
done
