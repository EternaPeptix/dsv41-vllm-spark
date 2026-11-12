#!/bin/bash
# g2ct-all.sh <status|install|uninstall> [--purge] (g2 colltrim) — run FROM THE MAC with bash. Runs
# ops/g2ct-site.sh on all 8 Sparks: install/uninstall do a preflight on all 8 first and change nothing
# unless all 8 pass; then apply on all 8 and require the same status digest everywhere (the rdma
# fingerprint hashes rdma_all_reduce.py, the .so and the l2 prefetch module: they must match 8/8).
# Service stopped for install/uninstall (G2CT_ALLOW_ACTIVE=1 overrides). Mac bash 3.2: plain host list.
set -euo pipefail
CMD=${1:?usage: g2ct-all.sh status|install|uninstall [--purge]}
EXTRA=${2:-}
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
S='$HOME/dsv41-stage/g2/colltrim/ops/g2ct-site.sh'
AE="G2CT_ALLOW_ACTIVE=${G2CT_ALLOW_ACTIVE:-0}"
if [ "$CMD" = install ] || [ "$CMD" = uninstall ]; then
  bad=0
  for h in $HOSTS; do
    if [ "$CMD" = install ]; then out=$(ssh -n "$h" "$AE bash $S preflight" 2>&1 || true)
    else out=$(ssh -n "$h" "systemctl --user is-active dsv41-rank 2>/dev/null || true"); fi
    echo "preflight $h: $out"
    case "$CMD:$out" in
      install:PREFLIGHT_OK*) ;;
      uninstall:active|uninstall:activating) [ "${G2CT_ALLOW_ACTIVE:-0}" = 1 ] || bad=1 ;;
      uninstall:*) ;;
      *) bad=1 ;;
    esac
  done
  [ $bad = 0 ] || { echo "preflight failed on at least one node: nothing changed"; exit 1; }
fi
digests=""
for h in $HOSTS; do
  line=$(ssh -n "$h" "$AE bash $S $CMD $EXTRA" 2>&1 | tail -1)
  echo "$line"
  d=$(printf '%s' "$line" | cut -d' ' -f2- | sed 's/ service=.*//' | shasum -a 256 | cut -c1-12)
  digests="$digests $d"
done
n=$(echo $digests | tr ' ' '\n' | sort -u | wc -l | tr -d ' ')
if [ "$n" = 1 ]; then echo "DIGEST identical on 8/8:$(echo $digests | cut -d' ' -f1)"; else echo "DIGEST DIFFERS across nodes:$digests"; exit 1; fi
