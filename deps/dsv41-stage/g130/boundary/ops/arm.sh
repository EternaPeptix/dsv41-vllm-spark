#!/bin/bash
# arm.sh <label> [--conc] [VAR=VAL ...]: one serving arm through the supervisor. RESTARTS THE SERVICE and leaves it up. Launcher default AV pin, clock 2000.
# Labels: off and c<N> take no flags, t<N> needs a fold flag, audit needs audit flags and --conc. Before the restart: mirror digest and SHA256SUMS on 8/8, site state
# patched on 8/8, and for a t arm a passing audit arm on this kit that ran the same kernels (same folds, tiles and tactic map; ops/arm_check.py pairing). Then on spark-one: exact.py, goal_single.py (G1), goal_conc4.py (G4), the long prompts (dbgrun.py, cases read from final3/det.jsonl), and with --conc
# load32.py, the only load here that drives num_reqs well past 4 (for audit arms). Then the 8 rank logs and audit reports. ops/arm_check.py decides.
# Exit 0 = every anchor hit and the flags really took effect; 1 = anything else; 2 = bad usage. Results: spark-one ~/dsv41-g130ab/boundary/<label>/.
# Run from the Mac. Tested against stub hosts by tests/test_ops_stub.py; NOT RUN YET on the real 8.
set -euo pipefail
L=${1:?label}; shift
CONC=0
[ "${1:-}" = --conc ] && { CONC=1; shift; }
CTX=0; MK=0
for kv in "$@"; do
  case "$kv" in
    DSV41_CTXKV_FOLD=*) CTX=${kv#*=} ;;
    DSV41_MK_FOLD=*) MK=${kv#*=} ;;
    DSV41_MK_FOLD_TILES=*) case "${kv#*=}" in 1|16) ;; *) echo "DSV41_MK_FOLD_TILES must be 1 or 16, got: $kv"; exit 2 ;; esac ;;
    DSV41_CTXKV_TACTIC_MAP=*|DSV41_BOUNDARY_AUDIT_EVERY=*) ;;
    *) echo "not a boundary flag: $kv"; exit 2 ;;
  esac
done
case "$L" in
  off|c[0-9]*) [ $# -eq 0 ] || { echo "$L is a flags-off arm, got: $*"; exit 2; } ;;
  t[0-9]*) case "$CTX$MK" in 00) echo "$L needs a fold flag: an empty \$T would measure stock"; exit 2 ;; *audit*) echo "$L is a serving arm, not audit"; exit 2 ;; esac ;;
  audit) case "$CTX:$MK" in audit:audit|audit:0|0:audit) ;; *) echo "audit needs DSV41_CTXKV_FOLD=audit and/or DSV41_MK_FOLD=audit, no serving fold"; exit 2 ;; esac
    [ "$CONC" = 1 ] || { echo "audit needs --conc: without load32 it never sees num_reqs past 4"; exit 2; } ;;
  *) echo "label must be off, audit, c<N> or t<N>"; exit 2 ;;
esac
K=$(cd "$(dirname "$0")/.." && pwd)
HOSTS="spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"
WANT=$(shasum -a 256 "$K/SHA256SUMS" | cut -c1-12)
for h in $HOSTS; do
  got=$(ssh -n -o ConnectTimeout=10 "$h" "cd ~/dsv41-stage/g130/boundary && sha256sum -c --quiet SHA256SUMS && sha256sum SHA256SUMS | cut -c1-12") || { echo "ABORT: $h fails SHA256SUMS"; exit 1; }
  [ "$got" = "$WANT" ] || { echo "ABORT: $h stage digest $got, mirror $WANT"; exit 1; }
done
bash "$K/ops/site8.sh" status patched || { echo "ABORT: site is not patched on 8/8, the arm would measure stock"; exit 1; }
S=$(mktemp -d "${TMPDIR:-/tmp}/boundary.XXXXXX"); trap 'rm -rf "$S"' EXIT
AUDIT_PASS='~/dsv41-g130ab/boundary/audit/audit_pass.json'   # expanded by the remote shell
case "$L" in
  t[0-9]*)
    ssh -n -o ConnectTimeout=10 spark-one "cat $AUDIT_PASS" > "$S/audit_pass.json" 2>/dev/null || true
    python3 -I "$K/ops/arm_check.py" pairing "$S/audit_pass.json" "$WANT" "$CTX" "$MK" "$@" || { echo "ABORT: $L would serve kernels no passing audit arm ran. Nothing restarted."; exit 1; } ;;
  audit)
    ssh -n -o ConnectTimeout=10 spark-one "rm -f $AUDIT_PASS" || { echo "ABORT: cannot clear the old audit record"; exit 1; } ;;
esac
for h in $HOSTS; do ssh -n -o ConnectTimeout=10 "$h" 'rm -f ~/dsv41-logs/boundary-audit.*.json' || { echo "ABORT: $h: cannot clear old audit reports"; exit 1; }; done
ssh -n spark-one 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > "$S/rel.sh"; [ -s "$S/rel.sh" ] || { echo "no supervisor"; exit 1; }
bash "$S/rel.sh" --label "g130-$L" --expect-launcher b3976c4b DSV41_GPU_CLOCK_MHZ=2000 "$@"

rc=0
ssh spark-one 'bash -s' "$L" "$CONC" <<'R' || rc=1
set -u; L=$1; CONC=$2; O=$HOME/dsv41-g130ab/boundary/$L; mkdir -p "$O"; cd "$O"
rm -f exact.* g1.* g4.* det.* load32.*
(cd ~/dsv41-bench && timeout 900 python3 exact.py "$O/exact.json" "$L") > exact.log 2>&1; echo $? > exact.rc
(cd ~/dsv41-bench && timeout 900 python3 -I goal/goal_single.py --out "$O/g1.jsonl" --label "$L") > g1.log 2>&1; echo $? > g1.rc
(cd ~/dsv41-bench && timeout 900 python3 -I goal/goal_conc4.py --out "$O/g4.jsonl" --label "$L") > g4.log 2>&1; echo $? > g4.rc
C=$(python3 -I -c "import json,os;print(','.join(dict.fromkeys(json.loads(l)['case'] for l in open(os.path.expanduser('~/dsv41-g130ab/final3/det.jsonl')))))")
(cd ~/dsv41-g5det && timeout 1200 python3 -I scripts/dbgrun.py - "$O/det.jsonl" "$L" "$C" 2 0 cold) > det.log 2>&1; echo $? > det.rc
if [ "$CONC" = 1 ]; then timeout 900 python3 -I ~/dsv41-g130ab/load32.py > load32.log 2>&1; echo $? > load32.rc; fi
grep -c ' ERROR \|Traceback' ~/dsv41-logs/rank0.log > errors.rank0
python3 -I ~/dsv41-stage/g130/boundary/ops/arm_check.py serving "$O" "$CONC" ~/dsv41-g130ab/final3/det.jsonl
R

i=0
for h in $HOSTS; do
  ssh -n -o ConnectTimeout=10 "$h" "L=~/dsv41-logs/rank$i.log; if [ -s \$L ]; then grep -E 'DSV41_(CTXKV|MK)_FOLD|DSV41 boundary' \$L | cut -c1-300; echo LOG_READ; else echo LOG_MISSING; fi" > "$S/$h.log" 2>&1 || echo SSH_FAILED >> "$S/$h.log"
  ssh -n -o ConnectTimeout=10 "$h" 'for f in $(ls -t ~/dsv41-logs/boundary-audit.*.json 2>/dev/null | head -1); do cat "$f"; done' > "$S/$h.audit" 2>&1 || echo SSH_FAILED >> "$S/$h.audit"
  i=$((i + 1))
done
python3 -I "$K/ops/arm_check.py" fleet "$S" "$CTX" "$MK" "$@" || rc=1
if [ "$rc" = 0 ] && [ "$L" = audit ]; then
  python3 -I "$K/ops/arm_check.py" record "$WANT" "$CTX" "$MK" "$@" | ssh -o ConnectTimeout=10 spark-one "cat > $AUDIT_PASS.tmp && mv $AUDIT_PASS.tmp $AUDIT_PASS" || { echo "FAIL could not record the audit pass, no t arm can run"; rc=1; }
fi
if [ "$rc" = 0 ]; then echo "ARM $L: PASS"; else echo "ARM $L: FAIL, do not use this arm"; fi
exit $rc
