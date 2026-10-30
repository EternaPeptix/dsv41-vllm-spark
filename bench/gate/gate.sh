#!/bin/bash
# gate.sh -- dsv41 regression gate (Codex ops kit 08), ONE command, ON spark-one:
#
#   bash ~/dsv41-bench/gate/gate.sh --live [--label L] [--no-1m]     run it (refuses a busy server)
#   bash ~/dsv41-bench/gate/gate.sh --plan                            print the steps and the budget
#   bash ~/dsv41-bench/gate/gate.sh --last                            print the newest result
#
# Steps, each stopping the gate on a hard failure (first run 570 s of the kit's 900 s budget):
#   refuse_if_busy      nothing running/waiting and no request finished for 5 s, else exit 3 (REFUSED)
#   exact.py            combined digest must start with GATE_EXACT_ANCHOR (default a8c3394af9d2)
#   g1_one_pass         goal_single.py --repeats 1: G1 tok/s, outputs digest; digest compared with the previous
#                       PASS under the same launcher sha (informational: an unpinned production launch can land
#                       on either side of the known G1 prose near-tie)
#   probe_1m            (moved before GSM8K/longctx after the first run, 2026-10-27: a 1M prefill right after
#                       the 200K longctx requests bottomed at MemAvailable 5.98 GB on spark-one, against 8.6-8.7 GB
#                       when it runs first, as in every earlier verification suite; see below for the probe itself)
#   gsm8k_200           ~/gsm8k_gate.py 200 8: accuracy >= GATE_GSM_MIN (default 194; reference 196/200,
#                       misses 93,100,119,184)
#   longctx             ~/dsv41-bench/longctx.py (needle + copy, 4K-200K): LONGCTX PASS
#   probe_1m            ONE cold ~1M prompt (xrb4.py solo): TTFT <= GATE_1M_MAX_S (default 220 s = the good
#                       class; a slower launch fails as the straggler class, it is not dropped); MemAvailable
#                       minimum on spark-one from a 2 s sampler. --no-1m skips it (recorded as skipped).
#   kl_verify           kl_panel.py verify (no server)
#   kl_smoke            kl_panel.py smoke (3 requests)
#   kl_capture_small    kl_panel.py capture --sets GATE_KL_SETS (default tools,math = 81,840 teacher-forced
#                       positions); compared (--tier equiv) with the previous gate capture under the same
#                       launcher sha when there is one
# Not in this command (the nightly set): a full KL capture, G1 x3, G4, decode speed A/Bs.
#
# Results: ~/dsv41-bench/gate/runs/<launcher sha8>/<UTC>.json (summary) and <UTC>/ (raw files). Two launchers
# never share a directory. A 2 s /metrics sampler records foreign traffic (the LAN/tailnet API has no auth):
# each step records the requests that finished during it against the number it sent.
# Exit: 0 PASS, 1 FAIL (summary names the step), 3 REFUSED (busy), 2 usage.
set -u
MODE=--plan; LABEL=gate; DO1M=1
while [ $# -gt 0 ]; do
  case "$1" in
    --live|--plan|--last) MODE=$1; shift ;;
    --label) LABEL=$2; shift 2 ;;
    --no-1m) DO1M=0; shift ;;
    *) echo "usage: gate.sh [--plan|--live|--last] [--label L] [--no-1m]" >&2; exit 2 ;;
  esac
done
G=$HOME/dsv41-bench/gate
B=$HOME/dsv41-bench
URL=http://127.0.0.1:8000
EXACT_ANCHOR=${GATE_EXACT_ANCHOR:-a8c3394af9d2}
GSM_MIN=${GATE_GSM_MIN:-194}
MAX1M=${GATE_1M_MAX_S:-220}
KLSETS=${GATE_KL_SETS:-tools,math}
LS=$(sha256sum $HOME/dsv41-tp8.sh | cut -c1-8)

if [ "$MODE" = --plan ]; then
  cat <<EOF
launcher $LS -> results under $G/runs/$LS/
   2 s  refuse_if_busy
  ~30 s exact.py (anchor $EXACT_ANCHOR)
  ~35 s G1 one pass (digest vs previous PASS under $LS)
 ~180 s ONE cold 1M probe (TTFT <= $MAX1M s)$([ $DO1M = 0 ] && echo " [skipped: --no-1m]")
 ~150 s GSM8K-200 (>= $GSM_MIN)
 ~160 s longctx needle + copy (LONGCTX PASS)
   2 s  kl_panel verify
  ~10 s kl_panel smoke
  ~60 s kl_panel capture --sets $KLSETS (+ compare --tier equiv vs the previous gate capture)
EOF
  exit 0
fi
if [ "$MODE" = --last ]; then
  f=$(ls -t $G/runs/*/*.json 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "$f"; python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ('verdict','failed_step','launcher','instance','utc')}, indent=1)); [print(f\"  {s['name']:18} {s['verdict']:8} {s['seconds']:7.1f}s  {s.get('note','')}\") for s in d['steps']]" "$f"; } || echo "no gate runs yet"
  exit 0
fi

UTC=$(date -u +%Y%m%dT%H%M%SZ)
D=$G/runs/$LS/$UTC; mkdir -p "$D"
SUM=$G/runs/$LS/$UTC.json
STEPS=$D/steps.jsonl; : > "$STEPS"
met() { curl -s -m 5 $URL/metrics; }
succ() { met | awk '/^vllm:request_success_total\{/{s+=$2} END{printf "%d", s}'; }
busy() { met | awk '/^vllm:num_requests_(running|waiting)\{/{s+=$2} END{printf "%d", s}'; }
PID=$(pgrep -u "$(id -u)" -f "[v]llm serve" | head -1)
INSTANCE=$(systemctl --user show dsv41-rank -p Description --value 2>/dev/null)
AVPIN=$( [ -n "$PID" ] && tr '\0' '\n' < /proc/$PID/environ | sed -n 's/^DSV41_AV_COST_JSON=//p')
finish() {  # $1 verdict, $2 failed step
  python3 -I - "$STEPS" "$SUM" "$1" "$2" "$LS" "$INSTANCE" "$UTC" "$LABEL" "${AVPIN:-unpinned}" "$D" <<'PY'
import json, sys
steps, out, verdict, failed, ls, inst, utc, label, av, d = sys.argv[1:11]
rows = [json.loads(l) for l in open(steps) if l.strip()]
json.dump({"verdict": verdict, "failed_step": failed or None, "launcher": ls, "instance": inst, "utc": utc,
           "label": label, "av_pin": av, "raw_dir": d, "steps": rows,
           "total_seconds": round(sum(r["seconds"] for r in rows), 1)}, open(out, "w"), indent=1)
print(f"GATE {verdict}{' at ' + failed if failed else ''}: launcher {ls}, instance '{inst}', {utc}, "
      f"{sum(r['seconds'] for r in rows):.0f} s -> {out}")
for r in rows:
    print(f"  {r['name']:18} {r['verdict']:8} {r['seconds']:7.1f}s  {r.get('note', '')}")
PY
  touch "$D/DONE"
}
rec() {  # name verdict seconds sent note  (json-escaped by python)
  python3 -I -c 'import json,sys; n,v,s,sent,fin,note=sys.argv[1:7]; print(json.dumps({"name":n,"verdict":v,"seconds":round(float(s),1),"requests_sent":int(sent),"requests_finished":int(fin),"foreign":int(fin)>int(sent),"note":note}))' "$@" >> "$STEPS"
}
now() { date +%s.%N; }
el() { python3 -I -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$1" "$(now)"; }

# ---- refuse_if_busy -------------------------------------------------------------------------------
t=$(now)
if [ -z "$PID" ] || [ "$(curl -s -m 5 -o /dev/null -w %{http_code} $URL/health)" != 200 ]; then
  rec refuse_if_busy FAIL "$(el $t)" 0 0 "server not healthy"; finish FAIL refuse_if_busy; exit 1
fi
c0=$(succ); b=$(busy); sleep 5; c1=$(succ); b2=$(busy)
if [ "$b" != 0 ] || [ "$b2" != 0 ] || [ "$c0" != "$c1" ]; then
  echo "REFUSED: server busy (running+waiting $b/$b2, finished $((c1 - c0)) in 5 s); nothing was run"
  rec refuse_if_busy REFUSED "$(el $t)" 0 $((c1 - c0)) "busy: running+waiting $b/$b2"; finish REFUSED refuse_if_busy
  exit 3
fi
rec refuse_if_busy PASS "$(el $t)" 0 0 "idle; launcher $LS; instance '$INSTANCE'; AV ${AVPIN:-unpinned}"
# 2 s sampler (running, waiting, success_total) until DONE
( while [ ! -e "$D/DONE" ]; do m=$(met); echo "$(date +%s) $(echo "$m" | awk '/^vllm:num_requests_running\{/{r=$2} /^vllm:num_requests_waiting\{/{w=$2} /^vllm:request_success_total\{/{s+=$2} END{print r+0, w+0, s+0}')" >> "$D/sampler.txt"; sleep 2; done ) &

prev=$(ls -t $G/runs/$LS/*.json 2>/dev/null | grep -v "/$UTC.json" | head -1)
pval() { [ -n "$prev" ] && python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); s=[x for x in d['steps'] if x['name']==sys.argv[2]]; print(s[0].get(sys.argv[3],'') if s else '')" "$prev" "$1" "$2"; }

# ---- exact.py --------------------------------------------------------------------------------------
t=$(now); c0=$(succ)
( cd $B && python3 exact.py "$D/exact.json" "$LABEL-exact" > "$D/exact.log" 2>&1 ); rc=$?
comb=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['combined'][:16])" "$D/exact.json" 2>/dev/null)
v=PASS; case "$comb" in $EXACT_ANCHOR*) ;; *) v=FAIL ;; esac; [ $rc = 0 ] || v=FAIL
rec exact.py $v "$(el $t)" 6 $(( $(succ) - c0 )) "combined $comb (anchor $EXACT_ANCHOR)"
[ $v = PASS ] || { finish FAIL exact.py; exit 1; }

# ---- G1 one pass -----------------------------------------------------------------------------------
t=$(now); c0=$(succ)
python3 -I $B/goal/goal_single.py --repeats 1 --out "$D/g1.jsonl" --label "$LABEL-g1" > "$D/g1.log" 2>&1; rc=$?
read -r g1 dig per <<<"$(python3 -I -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
s=[r for r in rows if r.get('metric')=='G1_summary']
if not s: print('- - -'); sys.exit()
s=s[0]; pt=s.get('per_type',{})
print(round(s.get('G1_mean_tok_s') or 0, 2), s.get('outputs_digest','-'), ','.join(f\"{k}:{'/'.join(v.get('sha8',[]))}\" for k,v in sorted(pt.items())))
" "$D/g1.jsonl" 2>/dev/null)"
pd=$(pval g1_one_pass digest)
cmp=none; [ -n "$pd" ] && { [ "$pd" = "$dig" ] && cmp=same || cmp="differs (prev $pd)"; }
v=PASS; [ $rc = 0 ] && [ "$dig" != - ] || v=FAIL
rec g1_one_pass $v "$(el $t)" 3 $(( $(succ) - c0 )) "G1 $g1 tok/s; digest $dig vs previous PASS: $cmp; $per"
python3 -I -c "import json,sys; f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; r=rows[-1]; r['digest']=sys.argv[2]; r['g1']=sys.argv[3]; open(f,'w').write(''.join(json.dumps(x)+'\n' for x in rows[:-1]+[r]))" "$STEPS" "$dig" "$g1"
[ $v = PASS ] || { finish FAIL g1_one_pass; exit 1; }

# ---- ONE cold 1M probe -------------------------------------------------------------------------------
if [ $DO1M = 1 ]; then
  t=$(now); c0=$(succ)
  ( while [ ! -e "$D/.memstop" ]; do echo "$(date +%s) $(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$D/memlog-1m.txt"; sleep 2; done ) &
  MP=$!
  python3 -I $HOME/dsv41-g4p/scripts/xrb4.py solo --len 1000k --tag "$LABEL-$LS-$UTC" --out "$D/s1m.json" > "$D/s1m.log" 2>&1; rc=$?
  touch "$D/.memstop"; wait $MP 2>/dev/null; rm -f "$D/.memstop"
  read -r ttft pt <<<"$(python3 -I -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith('{')]
r=[x for x in rows if 'ttft_s' in x]
print(r[-1]['ttft_s'] if r else '-', r[-1].get('prompt_tokens','-') if r else '-')" "$D/s1m.log" 2>/dev/null)"
  memmin=$(awk 'NR==1||$2<m{m=$2} END{printf "%.1f", m/1048576}' "$D/memlog-1m.txt" 2>/dev/null)
  v=FAIL; cls=straggler
  if [ $rc = 0 ] && [ "$ttft" != - ] && python3 -I -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$ttft" "$MAX1M"; then v=PASS; cls=good; fi
  rec probe_1m $v "$(el $t)" 3 $(( $(succ) - c0 )) "TTFT ${ttft}s for ${pt} prompt tokens ($cls class, max ${MAX1M}s); MemAvailable min ${memmin} GB"
  [ $v = PASS ] || { finish FAIL probe_1m; exit 1; }
else
  rec probe_1m SKIPPED 0 0 0 "--no-1m"
fi

# ---- GSM8K-200 --------------------------------------------------------------------------------------
t=$(now); c0=$(succ)
python3 $HOME/gsm8k_gate.py "$D/gsm8k200.jsonl" 200 8 > "$D/gsm8k200.log" 2>&1; rc=$?
acc=$(sed -n 's/^accuracy \([0-9]*\)\/200.*/\1/p' "$D/gsm8k200.log" | head -1)
wrong=$(grep '^wrong:' "$D/gsm8k200.log" | head -1 | cut -c1-200)
v=PASS; [ $rc = 0 ] && [ -n "$acc" ] && [ "$acc" -ge "$GSM_MIN" ] || v=FAIL
rec gsm8k_200 $v "$(el $t)" 200 $(( $(succ) - c0 )) "accuracy ${acc:-?}/200 (min $GSM_MIN; ref 196, misses 93,100,119,184); $wrong"
[ $v = PASS ] || { finish FAIL gsm8k_200; exit 1; }

# ---- longctx ------------------------------------------------------------------------------------------
t=$(now); c0=$(succ)
( cd $B && python3 longctx.py --out "$D/longctx.jsonl" > "$D/longctx.log" 2>&1 ); rc=$?
lc=$(grep '^LONGCTX' "$D/longctx.log" | tail -1 | cut -c1-160)
n=$(grep -c . "$D/longctx.jsonl" 2>/dev/null)
v=FAIL; case "$lc" in "LONGCTX PASS"*) [ $rc = 0 ] && v=PASS ;; esac
rec longctx $v "$(el $t)" $(( ${n:-0} * 2 )) $(( $(succ) - c0 )) "$lc"
[ $v = PASS ] || { finish FAIL longctx; exit 1; }

# ---- KL panel ----------------------------------------------------------------------------------------
K=$B/klpanel/kl_panel.py
t=$(now)
python3 $K verify > "$D/kl_verify.log" 2>&1; rc=$?
v=PASS; [ $rc = 0 ] || v=FAIL
rec kl_verify $v "$(el $t)" 0 0 "$(tail -1 "$D/kl_verify.log" | cut -c1-120)"
[ $v = PASS ] || { finish FAIL kl_verify; exit 1; }
t=$(now); c0=$(succ)
python3 $K smoke > "$D/kl_smoke.log" 2>&1; rc=$?
v=PASS; [ $rc = 0 ] || v=FAIL
rec kl_smoke $v "$(el $t)" 3 $(( $(succ) - c0 )) "$(grep 'tf n=' "$D/kl_smoke.log" | tail -1 | cut -c1-110)"
[ $v = PASS ] || { finish FAIL kl_smoke; exit 1; }
t=$(now); c0=$(succ)
KLAB="gate-$LS-$UTC"
python3 $K capture --label "$KLAB" --sets "$KLSETS" --concurrency 4 --note "regression gate $LS $UTC instance $INSTANCE" > "$D/kl_capture.log" 2>&1; rc=$?
nseq=$(python3 -I -c "
import json,os,sys
b=os.path.join(os.path.dirname(sys.argv[1]),'runs',sys.argv[2]); n=0
for s in sys.argv[3].split(','):
    p=os.path.join(b,s+'.jsonl'); n+=sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0
print(n)" "$K" "$KLAB" "$KLSETS" 2>/dev/null)
pk=$(pval kl_capture_small kl_label)
cmpv=baseline
if [ $rc = 0 ] && [ -n "$pk" ]; then
  python3 $K compare --ref "$pk" --label "$KLAB" --tier equiv > "$D/kl_compare.log" 2>&1
  cmpv="vs $pk: $(grep -m1 -E '^ *GATE equiv' "$D/kl_compare.log" | sed 's/^ *//' | cut -c1-100)"
fi
v=PASS; [ $rc = 0 ] || v=FAIL
rec kl_capture_small $v "$(el $t)" "${nseq:-0}" $(( $(succ) - c0 )) "label $KLAB, sets $KLSETS, $nseq sequences; $cmpv"
python3 -I -c "import json,sys; f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; r=rows[-1]; r['kl_label']=sys.argv[2]; open(f,'w').write(''.join(json.dumps(x)+'\n' for x in rows[:-1]+[r]))" "$STEPS" "$KLAB"
[ $v = PASS ] || { finish FAIL kl_capture_small; exit 1; }

finish PASS ""
exit 0
