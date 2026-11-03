#!/bin/bash
# pair.sh <label> <C|T> [DSV41_L2_PREFETCH_x=v ...]: one pinned serving arm through the supervisor. RESTARTS THE SERVICE (launcher's default AV pin, GPU clock
# 2000). Then on spark-one: exact.py, goal_single x2, goal_conc4, the 9 long-prompt cases. Prints every anchor. Output under ~/dsv41-g130ab/l2win/<label>/. From the
# Mac; run CTTC from WINDOW.md. NOT RUN YET.
set -euo pipefail
L=${1:?label}; ARM=${2:?C or T}; shift 2
[ "$ARM" = C ] && [ $# -gt 0 ] && { echo "C takes no flags"; exit 2; }
[ "$ARM" = T ] && [ $# -eq 0 ] && { echo "T needs flags"; exit 2; }
for kv in "$@"; do case "$kv" in DSV41_L2_PREFETCH_PACE_*=*|DSV41_L2_PREFETCH_BURST_DELAY_*=*) ;; *) echo "not an l2win flag: $kv"; exit 2 ;; esac; done
S=$(mktemp -d "${TMPDIR:-/tmp}/l2winpair.XXXXXX"); trap 'rm -rf "$S"' EXIT
ssh -n spark-one 'cat ~/dsv41-opt-20261021/ops/rel/dsv41-relaunch.sh' > "$S/rel.sh"; [ -s "$S/rel.sh" ] || { echo "no supervisor"; exit 1; }
bash "$S/rel.sh" --label "g130-l2win-$L" --expect-launcher 37757dde DSV41_GPU_CLOCK_MHZ=2000 "$@"
ssh spark-one 'bash -s' "$L" <<'R'
set -u; L=$1; O=$HOME/dsv41-g130ab/l2win/$L; mkdir -p "$O"; cd "$O"
grep -E 'DSV41_L2_PREFETCH (g2|l2win)' ~/dsv41-logs/rank0.log | tail -3 | cut -c1-220
timeout 900 python3 -I ~/dsv41-bench/exact.py exact.json > exact.log 2>&1
python3 -I -c "import json;print('exact',json.load(open('exact.json'))['combined'][:16],'(want a8c3394af9d28249)')"
for i in 1 2; do timeout 900 python3 -I ~/dsv41-bench/goal/goal_single.py --out g1.jsonl --label "$L" > g1.$i.log 2>&1; done
for i in 1 2; do timeout 900 python3 -I ~/dsv41-bench/goal/goal_conc4.py --out g4.jsonl --label "$L" > g4.$i.log 2>&1; done
(cd ~/dsv41-g5det && timeout 900 python3 -I scripts/dbgrun.py - "$O/det.jsonl" "$L" big:2500,big:8000,big:32000,big:128000,code:4000,prose:6000,l3:B:T16-spam-reply:1,l3:B:T02-classify:1,l3:A:T01-reviews:1 4 0 cold >/dev/null 2>&1)
python3 -I - <<'P'
import json, collections
S = [json.loads(l) for l in open("g1.jsonl")]; S = [r for r in S if r.get("metric") == "G1_summary"]
for r in S: print("G1", r["G1_mean_tok_s"], r["outputs_digest"][:16], "(want 9a18f802d930f6e7)", {k: v["tok_s"] for k, v in r["per_type"].items()})
G = [json.loads(l) for l in open("g4.jsonl")]; G = [r for r in G if r.get("metric") == "G4_summary"]; print("G4 min", [r["G4_per_session_min"] for r in G], "mean", [r["G4_per_session_mean"] for r in G])
REF = {"big:2500":"3f4a9dd1","big:8000":"d3e0f849","big:32000":"7ba3d1a9","big:128000":"89bd9111","code:4000":"43c4b492","prose:6000":"05e29037","l3:B:T16-spam-reply:1":"37dd0438","l3:B:T02-classify:1":"6b7032f2","l3:A:T01-reviews:1":"6d8cf8a2"}
C = collections.defaultdict(collections.Counter)
for l in open("det.jsonl"):
    r = json.loads(l)
    if "tok_sha" in r: C[r["case"]][r["tok_sha"][:8]] += 1
bad = {c: dict(v) for c, v in C.items() if set(v) != {REF[c]}}; print("long", len(C), "of 9", "ALL==control" if not bad and len(C) == 9 else bad)
P
echo "errors_rank0=$(grep -c ' ERROR \|Traceback' ~/dsv41-logs/rank0.log)"
R
