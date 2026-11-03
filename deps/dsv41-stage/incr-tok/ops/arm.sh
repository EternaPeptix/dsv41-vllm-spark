#!/bin/bash
# arm.sh <label> [DSV41_INCR_TOK=1|audit] [DSV41_INCR_TOK_<X>=v ...]: one serving arm through the supervisor. RESTARTS THE SERVICE (launcher 66eaa1d9,
# GPU clock 2000) and leaves it up. Before the restart: this mirror's SHA256SUMS digest staged on 8/8 and site state `patched` on 8/8. With the flag
# on, the overlay's per-request log goes to spark-one ~/dsv41-logs/incr-tok.jsonl (emptied first). Then on spark-one: exact.py (a8c3394af9d28249),
# goal_single.py (G1 digest 9a18f802d930f6e7), the 9 long-prompt cases x4 cold (dbgrun.py, reference hashes), ttft_ab.py (synthetic growing
# conversation, about 230K -> 330K tokens, 56 steps), the DSV41_INCR_TOK lines and the error count of rank0.log; ops/arm_check.py decides.
# Results: spark-one ~/dsv41-incrtok/<label>/. Exit 0 = ARM PASS, 1 = FAIL, 2 = bad usage. From the Mac. NOT RUN YET on the real 8.
set -euo pipefail
L=${1:?label}; shift
MODE=off
for kv in "$@"; do
  case "$kv" in
    DSV41_INCR_TOK=1) MODE=on ;;
    DSV41_INCR_TOK=audit) MODE=audit ;;
    DSV41_INCR_TOK=*) echo "DSV41_INCR_TOK must be 1 or audit here (leave it out for the off arm): $kv"; exit 2 ;;
    DSV41_INCR_TOK_LOG=*) echo "arm.sh sets DSV41_INCR_TOK_LOG itself"; exit 2 ;;
    DSV41_INCR_TOK_*=*) ;;
    *) echo "not an incr-tok flag: $kv"; exit 2 ;;
  esac
done
case "$L" in *[!A-Za-z0-9._-]*|'') echo "label must be [A-Za-z0-9._-]+"; exit 2 ;; esac
[ "$MODE" = off ] && [ $# -gt 0 ] && { echo "tunables without DSV41_INCR_TOK=1|audit do nothing"; exit 2; }
K=$(cd "$(dirname "$0")/.." && pwd)
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
WANT=$(shasum -a 256 "$K/SHA256SUMS" | cut -c1-12)
for h in $HOSTS; do
  got=$(ssh -n -o ConnectTimeout=10 "$h" "cd ~/dsv41-stage/incr-tok && sha256sum -c --quiet SHA256SUMS && sha256sum SHA256SUMS | cut -c1-12") || { echo "ABORT: $h fails SHA256SUMS"; exit 1; }
  [ "$got" = "$WANT" ] || { echo "ABORT: $h stage digest $got, mirror $WANT"; exit 1; }
done
bash "$K/ops/site8.sh" status patched || { echo "ABORT: site is not patched on 8/8. Nothing restarted."; exit 1; }
EXTRA=()
[ "$MODE" != off ] && EXTRA=("DSV41_INCR_TOK_LOG=/home/jeweled/dsv41-logs/incr-tok.jsonl")
ssh -n -o ConnectTimeout=10 spark-one 'mkdir -p ~/dsv41-logs && rm -f ~/dsv41-logs/incr-tok.jsonl' || { echo "ABORT: cannot clear the overlay log"; exit 1; }
S=$(mktemp -d "${TMPDIR:-/tmp}/incrtok.XXXXXX"); trap 'rm -rf "$S"' EXIT
ssh -n spark-one 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > "$S/rel.sh"; [ -s "$S/rel.sh" ] || { echo "no supervisor"; exit 1; }
bash "$S/rel.sh" --label "incrtok-$L" --expect-launcher 66eaa1d9 DSV41_GPU_CLOCK_MHZ=2000 "$@" ${EXTRA[@]+"${EXTRA[@]}"}

rc=0
ssh spark-one 'bash -s' "$L" "$MODE" <<'R' || rc=1
set -u; L=$1; MODE=$2; O=$HOME/dsv41-incrtok/$L; K=$HOME/dsv41-stage/incr-tok; mkdir -p "$O"; cd "$O"
rm -f exact.* g1.* det.* ab.* rank0.incr errors.rank0 incr-tok.jsonl
(cd ~/dsv41-bench && timeout 900 python3 exact.py "$O/exact.json" "$L") > exact.log 2>&1; echo $? > exact.rc
(cd ~/dsv41-bench && timeout 900 python3 -I goal/goal_single.py --out "$O/g1.jsonl" --label "$L") > g1.log 2>&1; echo $? > g1.rc
C=$(python3 -I -c "import sys; sys.path.insert(0, '$K/ops'); import arm_check; print(arm_check.LONG_CASES)")
(cd ~/dsv41-g5det && timeout 1500 python3 -I scripts/dbgrun.py - "$O/det.jsonl" "$L" "$C" 4 0 cold) > det.log 2>&1; echo $? > det.rc
timeout 2400 python3 -I "$K/ops/ttft_ab.py" --out "$O/ab.jsonl" --label "$L" > ab.log 2>&1; echo $? > ab.rc
grep -a 'DSV41_INCR_TOK' ~/dsv41-logs/rank0.log | cut -c1-400 > rank0.incr || true
grep -ac ' ERROR \|Traceback' ~/dsv41-logs/rank0.log > errors.rank0 || true
[ -f ~/dsv41-logs/incr-tok.jsonl ] && cp ~/dsv41-logs/incr-tok.jsonl "$O/incr-tok.jsonl"
python3 -I "$K/ops/arm_check.py" "$O" "$MODE"
R
if [ "$rc" = 0 ]; then echo "ARM $L: PASS"; else echo "ARM $L: FAIL, do not use this arm"; fi
exit $rc
