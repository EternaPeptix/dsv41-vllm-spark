#!/bin/bash
# verify8-r7.sh [--uninstalled]: the r7 .so, rdma_all_reduce.py, dsv41_oblit.py and the live .so on 8/8 against KIT.json.
# Installed: .so = KIT.json r7 sha, both files = patched_sha256. --uninstalled: no r7 .so; dsv41_oblit.py = live base;
# rdma_all_reduce.py = the live base or r6-comm's 67498e6b (whichever the node had before install).
set -euo pipefail
cd "$(dirname "$0")/.."
TS=$HOME/Documents/dsv41-kits/ts-ssh/bin
[ -x "$TS/ssh" ] && export PATH=$TS:$PATH
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
read -r SO RP RL R6 OP OB < <(python3 -c '
import json; k = json.load(open("KIT.json")); f = {x["path"]: x for x in k["files"]}
r = f["distributed/device_communicators/rdma_all_reduce.py"]; o = f["models/deepseek_v41/nvidia/dsv41_oblit.py"]
print(f["$HOME/dsv41-stage/r7comm2/build/rdma_ar_ext.so"]["sha256"] or "-", r["patched_sha256"], r["base_sha256_live"], r["base_sha256_r6"], o["patched_sha256"], o["base_sha256_accepted"][0])')
LIVE=14d049dda41a2a8f
bad=0
for h in $HOSTS; do
  out=$(ssh -n -o ConnectTimeout=10 "$h" 'f() { [ -e "$1" ] && sha256sum "$1" | cut -d" " -f1 || echo -; }
    V=~/venvs/vllm032/lib/python3.12/site-packages/vllm
    echo "$(f ~/dsv41-stage/r7comm2/build/rdma_ar_ext.so) $(f $V/distributed/device_communicators/rdma_all_reduce.py) $(f $V/models/deepseek_v41/nvidia/dsv41_oblit.py) $(f ~/dsv41-stage/g130/rdma/build/rdma_ar_ext.so)"') || out="- - - -"
  read -r so rar obl live <<<"$out"
  ok=OK
  if [ "${1:-}" = "--uninstalled" ]; then
    [ "$so" = - ] && { [ "$rar" = "$RL" ] || [ "$rar" = "$R6" ]; } && [ "$obl" = "$OB" ] || { ok=BAD; bad=1; }
  else
    [ "$so" = "$SO" ] && [ "$rar" = "$RP" ] && [ "$obl" = "$OP" ] || { ok=BAD; bad=1; }
  fi
  [ "${live:0:16}" = "$LIVE" ] || { ok=BAD; bad=1; }
  printf '%-12s r7.so %.16s rdma_all_reduce.py %.16s dsv41_oblit.py %.16s live.so %.16s %s\n' "$h" "$so" "$rar" "$obl" "$live" "$ok"
done
[ $bad = 0 ] && echo "VERIFY8_R7 PASS" || { echo "VERIFY8_R7 FAIL"; exit 1; }
