#!/bin/bash
# takecheck-r7.sh [--watch [SECONDS]]: review item 1 promote precondition for the HT13 (p) arms. READ-ONLY (ssh grep
# of the live rank logs; nothing is written on any node; output goes to build-logs/ on this Mac).
#
# With DSV41_RDMA_PROJ=1 every rank logs, at the end of each forward (dsv41_oblit.project_head -> r7_disarm),
#   "rdma_ar-r7 fwd: capturing=1 fwd_rows=R ar_rows=[..] folded=F taken=T wasted=W ..."
# for every CUDA-graph capture (and for the first 8 eager forwards that folded). PASS needs, on EVERY rank (8/8):
#   * the boot line "rdma_ar r7 r7comm2-ht13-v1: projection fold on";
#   * at least one capturing line with 1 <= fwd_rows <= 6 (a decode graph the fold covers);
#   * taken == EXPECT_TAKEN (default 76 = 2 x 38 bound layers) on EVERY capturing line with 1 <= fwd_rows <= 6, and
#     wasted == 0 there (t2 predicts 0: the Engram AR is a gather and never folds);
# else FAIL: the folds would cost tails without removing the _proj_rows launches (do not run or promote the p arms).
# The rank logs are replaced at every relaunch, so run it while a p arm is up: `--watch` polls every SECONDS (default
# 60) until a p arm's capture lines are present on all 8 ranks, checks once, and exits (run it next to climb-window.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
TS=$HOME/Documents/dsv41-kits/ts-ssh/bin
[ -x "$TS/ssh" ] && export PATH=$TS:$PATH
HOSTS=(spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight)
EXPECT_TAKEN=${EXPECT_TAKEN:-76}
mkdir -p build-logs
once() {
  local d; d=$(mktemp -d)
  for h in "${HOSTS[@]}"; do
    ssh -n -o ConnectTimeout=10 -o BatchMode=yes "$h" \
      "grep -h -E 'rdma_ar r7 .*projection fold on|rdma_ar-r7 fwd:' ~/dsv41-logs/rank*.log 2>/dev/null" > "$d/$h.txt" 2>/dev/null || true
  done
  python3 -I - "$d" "$EXPECT_TAKEN" "${HOSTS[@]}" <<'EOF'
import re, sys
d, want, hosts = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
bad, ready = 0, 0
for h in hosts:
    L = open(f"{d}/{h}.txt").read().splitlines()
    boot = any("projection fold on" in l for l in L)
    caps = []
    for l in L:
        m = re.search(r"rdma_ar-r7 fwd: capturing=(\d) fwd_rows=(-?\d+|None) ar_rows=\[([\d, ]*)\] folded=(\d+) taken=(\d+) wasted=(\d+)", l)
        if m and m.group(1) == "1" and m.group(2) not in ("None",) and 1 <= int(m.group(2)) <= 6:
            caps.append((int(m.group(2)), int(m.group(4)), int(m.group(5)), int(m.group(6))))
    ok = boot and caps and all(t == want and w == 0 for _, _, t, w in caps)
    ready += bool(boot and caps)
    bad += not ok
    print(f"{h:12s} boot {'yes' if boot else 'NO '} decode captures {len(caps):3d} "
          f"(rows, folded, taken, wasted) {sorted(set(caps))[:6]} {'OK' if ok else 'BAD'}")
print(f"TAKECHECK_R7 {'PASS' if not bad else 'FAIL'}: {len(hosts) - bad}/{len(hosts)} ranks taken == {want} on every "
      f"captured decode forward (fwd_rows 1..6)")
sys.exit(0 if not bad else (3 if ready < len(hosts) else 1))
EOF
  local rc=$?; rm -rf "$d"; return $rc
}
if [ "${1:-}" = --watch ]; then
  P=${2:-60}
  while :; do
    out=$(once); rc=$?
    if [ $rc != 3 ]; then
      echo "$out" | tee "build-logs/takecheck-$(date -u +%Y%m%dT%H%M%SZ).txt"; exit $rc
    fi
    sleep "$P"
  done
fi
out=$(once); rc=$?
echo "$out" | tee "build-logs/takecheck-$(date -u +%Y%m%dT%H%M%SZ).txt"
exit $rc
