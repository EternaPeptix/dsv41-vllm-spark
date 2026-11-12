#!/usr/bin/env bash
# plan.sh EXPECT=<sha8> [WIN=r8-fixb-dw] plan.sh      -> renders ops/device-plan.{node,mac} for climb-common/device-window.sh
#   refuses without EXPECT (production is 4f3cc1c6 on 2026-11-09; pass the live sha8 after any promotion), and refuses to
#   write a plan in which a node line that runs Python lacks the environment rule (venv + PATH + CUDA_HOME).
#   then: WIN=$WIN EXPECT=$EXPECT NODE_PLAN=ops/device-plan.node MAC_PLAN=ops/device-plan.mac bash climb-common/device-window.sh
set -u -o pipefail
D=$(cd "$(dirname "$0")" && pwd); KIT=$(cd "$D/.." && pwd)
EXPECT=${EXPECT:?set EXPECT=<production launcher sha8>}; WIN=${WIN:-r8-fixb-dw}
for f in node mac; do
  sed -e "s#@KIT@#$KIT#g" -e "s#@EXPECT@#$EXPECT#g" -e "s#@WIN@#$WIN#g" "$D/device-plan.$f.in" > "$D/device-plan.$f.tmp"
done
# the command is everything after the second '|' (commands contain '||' themselves)
bad=$(awk -F'|' '{c=$0; sub(/^[^|]*\|[^|]*\|/, "", c)} $1 ~ /^spark-/ && c ~ /python/ && !(c ~ /source \$HOME\/venvs\/vllm032\/bin\/activate && export PATH=\/usr\/local\/cuda\/bin:\$PATH CUDA_HOME=\/usr\/local\/cuda/) {print $2}' "$D/device-plan.node.tmp")
if [ -n "$bad" ]; then echo "REFUSE: node lines without the environment rule: $bad"; rm -f "$D"/device-plan.*.tmp; exit 2; fi
grep -q -- "--expect $EXPECT" "$D/device-plan.node.tmp" || { echo "REFUSE: EXPECT not in the device-test lines"; exit 2; }
mv "$D/device-plan.node.tmp" "$D/device-plan.node"; mv "$D/device-plan.mac.tmp" "$D/device-plan.mac"
echo "wrote $D/device-plan.node ($(grep -c . "$D/device-plan.node") lines) and device-plan.mac; next:"
echo "  WIN=$WIN EXPECT=$EXPECT NODE_PLAN=$D/device-plan.node MAC_PLAN=$D/device-plan.mac bash $KIT/../../climb-common/device-window.sh"
