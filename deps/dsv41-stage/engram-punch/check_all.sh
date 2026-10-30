#!/bin/bash
# check_all.sh — run FROM THE MAC before dsv41-launch-all.sh: every node must hold the Engram rows of the rank
# launch-all gives it (HOSTS[i] -> rank i). Exit 1 if any node fails; nothing is started or stopped.
# Add to dsv41-launch-all.sh as the first line after HOSTS=:  bash ~/.../check_all.sh || exit 1
HOSTS=(spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight)
D=$(mktemp -d); rc=0
for i in "${!HOSTS[@]}"; do
  ssh -n -o ConnectTimeout=10 "${HOSTS[$i]}" "python3 -I \$HOME/dsv41-stage/engram-punch/engram_punch.py --rank $i --world 8 --check" > "$D/$i" 2>&1 &
done; wait
for i in "${!HOSTS[@]}"; do
  if grep -q "^engram check OK" "$D/$i"; then echo "${HOSTS[$i]} rank $i: $(grep '^engram check OK' "$D/$i" | sed 's/.*(\(.*\))/\1/')"
  else echo "${HOSTS[$i]} rank $i: FAILED"; sed 's/^/    /' "$D/$i"; rc=1; fi
done
rm -rf "$D"; [ $rc = 0 ] && echo "engram check: 8/8 OK" || echo "engram check: REFUSING to launch"
exit $rc
