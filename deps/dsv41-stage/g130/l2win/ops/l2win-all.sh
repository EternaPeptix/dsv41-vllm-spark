#!/bin/bash
# l2win-all.sh install|uninstall|status: from the Mac, service STOPPED on all 8 for install and uninstall. Only needed for a patched arm (s*, d*); a p* arm is
# env-only and installs nothing. Checks all 8 before touching any; the rdma_ar fingerprint hashes this module, so a partial install stops at startup rather
# than desynchronising. With every flag unset the new module behaves as 63737f9c. NOT RUN YET.
set -euo pipefail
CMD=${1:?install|uninstall|status}
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
T='$HOME/venvs/vllm032/lib/python3.12/site-packages/vllm/distributed/device_communicators/dsv41_l2_prefetch.py'
S='$HOME/dsv41-stage/g130/l2win/site/dsv41_l2_prefetch.py'
BASE=63737f9c413ebfa2c408b13cb0cbecddac2b42c6a5e219e6bad43bde095f4bee
NEW=$(shasum -a 256 "$(dirname "$0")/../site/dsv41_l2_prefetch.py" | cut -c1-64)
sha() { ssh -n -o ConnectTimeout=10 "$1" "timeout 20 sha256sum $2 2>/dev/null | cut -c1-64"; }
if [ "$CMD" = status ]; then for h in $HOSTS; do echo "$h $(sha "$h" "$T" | cut -c1-16)"; done; exit 0; fi
for h in $HOSTS; do
  st=$(ssh -n -o ConnectTimeout=10 "$h" 'timeout 10 systemctl --user is-active dsv41-rank 2>/dev/null || true')
  { [ "$st" = active ] || [ "$st" = activating ]; } && { echo "ABORT: dsv41-rank is $st on $h"; exit 1; }
  cur=$(sha "$h" "$T")
  case "$CMD" in
    install)   [ "$cur" = "$BASE" ] || { echo "ABORT: $h target is ${cur:0:16}, not base ${BASE:0:16}"; exit 1; }
               [ "$(sha "$h" "$S")" = "$NEW" ] || { echo "ABORT: $h stage file is not ${NEW:0:16}"; exit 1; } ;;
    uninstall) [ "$cur" = "$NEW" ] || { echo "ABORT: $h target is ${cur:0:16}, not the kit's ${NEW:0:16}"; exit 1; }
               [ "$(sha "$h" "$T.bak-g130-l2win")" = "$BASE" ] || { echo "ABORT: $h has no good backup"; exit 1; } ;;
    *) echo "unknown $CMD"; exit 2 ;;
  esac
done
for h in $HOSTS; do
  if [ "$CMD" = install ]; then ssh -n "$h" "timeout 20 cp -n $T $T.bak-g130-l2win && timeout 20 cp $S $T"; want=$NEW
  else ssh -n "$h" "timeout 20 cp $T.bak-g130-l2win $T && rm -f $T.bak-g130-l2win"; want=$BASE; fi
  [ "$(sha "$h" "$T")" = "$want" ] || { echo "FAILED on $h: run $0 status"; exit 1; }
  echo "$h $CMD OK ${want:0:16}"
done
