#!/usr/bin/env bash
# r5-glue install-all (review N3): install / uninstall / check r5-glue on all 8 Sparks, with an 8/8 agreement check.
#
#   ./install-all.sh status      read-only: per-node verdict CLEAN | INSTALLED | HOOKS-LOST | FOREIGN | ABSENT | STALE-TMP;
#                                exit 0 only if 8/8 agree on CLEAN or 8/8 on INSTALLED
#   ./install-all.sh preflight   status + refuse unless every node is CLEAN or INSTALLED (no FOREIGN / HOOKS-LOST / tmp)
#   ./install-all.sh install     preflight, stage the kit (~/dsv41-stage/climb-r5/r5-glue only), then node by node:
#                                install.py install + probe == INSTALLED; any failure uninstalls every touched node
#   ./install-all.sh uninstall   preflight, then node by node: install.py uninstall + probe == CLEAN
#
# Verdicts are computed HERE from a read-only remote probe (sha256 of the two patched r2 modules, of the two new modules,
# stale .tmp-r5-glue files): CLEAN = both r2 modules at their live base sha (KIT.json) and no r5-glue module;
# INSTALLED = both r2 modules at the sha that install.py produces from the base (computed locally on a temp copy, so
# nothing is pinned twice) and both new modules at KIT.json new_sha256; HOOKS-LOST = a new module present but an r2
# module back at its base (an r2 kit was re-installed on top); FOREIGN = anything else.
# Note: climb-r2/install-all.sh status reports FOREIGN for dsv41_attnout.py / dsv41_routeprep.py while r5-glue is
# installed (it hashes them against the r2 kits' shas). That is expected; use this script for these two files.
# The relaunch preflight manifest does not cover these files: run `status` before every relaunch with an r5-glue arm.
#
# Test hooks (never set them for the cluster): R5ALL_TRANSPORT=local R5ALL_FAKE=<dir> runs every "remote" command with
# HOME=<dir>/<host> on this machine; R5ALL_HOSTS overrides the host list; R5ALL_PY the remote python.
set -u -o pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
HOSTS=${R5ALL_HOSTS:-"spark-one spark-two spark-three spark-four spark-five spark-six spark-seven spark-eight"}
TRANSPORT=${R5ALL_TRANSPORT:-ssh}
RPY=${R5ALL_PY:-/usr/bin/python3}
LPY=${R5ALL_LPY:-python3}
STAGE_REL=dsv41-stage/climb-r5/r5-glue
VREL=venvs/vllm032/lib/python3.12/site-packages/vllm
export PATH=$HOME/Documents/dsv41-kits/ts-ssh/bin:$PATH
AO=models/deepseek_v4/nvidia/ops/dsv41_attnout.py
RP=model_executor/layers/fused_moe/dsv41_routeprep.py
NRQ=models/deepseek_v4/nvidia/ops/dsv41_r5g_rq.py
NRP=model_executor/layers/fused_moe/dsv41_r5g_rp.py

die() { echo "install-all: $*" >&2; exit 1; }

rsh() {  # rsh <host> <command...>   (stdin is passed through)
    local h=$1; shift
    if [ "$TRANSPORT" = local ]; then
        HOME="$R5ALL_FAKE/$h" bash -c "$*"
    else
        ssh -o ConnectTimeout=15 -o BatchMode=yes "$h" "$*"
    fi
}

stage() {  # stage <host>: install.py, KIT.json and files/ only
    local h=$1
    if [ "$TRANSPORT" = local ]; then
        mkdir -p "$R5ALL_FAKE/$h/$STAGE_REL" &&
            rsync -a --delete "$TMP/kit/" "$R5ALL_FAKE/$h/$STAGE_REL/"
    else
        rsh "$h" "mkdir -p ~/$STAGE_REL" </dev/null && rsync -a --delete "$TMP/kit/" "$h:$STAGE_REL/"
    fi
}

PROBE='
import hashlib, json, os
V = os.path.expanduser("~/'"$VREL"'")
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "-"
out = {r: sha(os.path.join(V, r)) for r in ("'"$AO"'", "'"$RP"'", "'"$NRQ"'", "'"$NRP"'")}
out["tmp"] = sorted(f for d in ("models/deepseek_v4/nvidia/ops", "model_executor/layers/fused_moe")
                    if os.path.isdir(os.path.join(V, d)) for f in os.listdir(os.path.join(V, d)) if "tmp-r5-glue" in f)
print(json.dumps(out))
'

TMP=$(mktemp -d "${TMPDIR:-/tmp}/r5all.XXXXXX") || die "mktemp"
trap 'rm -rf "$TMP"' EXIT
# the kit payload, and the expected installed shas (install.py applied to base/ copies here)
mkdir -p "$TMP/kit" "$TMP/v/$(dirname $AO)" "$TMP/v/$(dirname $RP)"
cp "$HERE/install.py" "$HERE/KIT.json" "$TMP/kit/" && cp -R "$HERE/files" "$TMP/kit/files" || die "copy kit"
cp "$HERE/base/dsv41_attnout.py" "$TMP/v/$AO" && cp "$HERE/base/dsv41_routeprep.py" "$TMP/v/$RP" || die "copy base"
R5G_VLLM="$TMP/v" "$LPY" "$TMP/kit/install.py" install >"$TMP/local-install.log" 2>&1 ||
    die "local install on base copies failed: $(cat "$TMP/local-install.log")"
EXPECT=$("$LPY" - "$HERE/KIT.json" "$TMP/v" "$AO" "$RP" "$NRQ" "$NRP" <<'EOF'
import hashlib, json, sys
k = json.load(open(sys.argv[1]))
v, ao, rp, nrq, nrp = sys.argv[2:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
by = {f["path"].split("/", 1)[1]: f for f in k["files"]}
print(json.dumps({"base": {ao: by[ao]["base_sha256"], rp: by[rp]["base_sha256"]},
                  "inst": {ao: sha(f"{v}/{ao}"), rp: sha(f"{v}/{rp}"), nrq: by[nrq]["new_sha256"],
                           nrp: by[nrp]["new_sha256"]}}))
EOF
) || die "expected shas"
for f in "$NRQ" "$NRP"; do  # KIT.json new_sha256 must be the shipped module
    want=$(printf '%s' "$EXPECT" | "$LPY" -c "import json,sys; print(json.load(sys.stdin)['inst']['$f'])")
    got=$(shasum -a 256 "$HERE/files/vllm/$f" | awk '{print $1}')
    [ "$want" = "$got" ] || die "files/vllm/$f sha $got != KIT.json new_sha256 $want"
done

verdict() {  # verdict <probe json> -> CLEAN | INSTALLED | HOOKS-LOST | FOREIGN | ABSENT | STALE-TMP
    "$LPY" - "$1" "$EXPECT" "$AO" "$RP" "$NRQ" "$NRP" <<'EOF'
import json, sys
p, e = json.loads(sys.argv[1]), json.loads(sys.argv[2])
ao, rp, nrq, nrp = sys.argv[3:]
if p["tmp"]:
    print("STALE-TMP"); sys.exit()
if p[ao] == "-" or p[rp] == "-":
    print("ABSENT"); sys.exit()
base = p[ao] == e["base"][ao] and p[rp] == e["base"][rp]
new_absent = p[nrq] == "-" and p[nrp] == "-"
if base and new_absent:
    print("CLEAN")
elif all(p[f] == e["inst"][f] for f in (ao, rp, nrq, nrp)):
    print("INSTALLED")
elif not new_absent and (p[ao] == e["base"][ao] or p[rp] == e["base"][rp]):
    print("HOOKS-LOST")
else:
    print("FOREIGN")
EOF
}

probe() { printf '%s' "$PROBE" | rsh "$1" "$RPY -"; }

status() {  # prints one line per host; sets AGREE (CLEAN / INSTALLED / '')
    local h j v first=""
    AGREE=""
    local same=1
    for h in $HOSTS; do
        j=$(probe "$h" 2>"$TMP/err") || j=""
        if [ -z "$j" ]; then v="UNREACHABLE"; else v=$(verdict "$j"); fi
        printf '%-12s %-11s %s\n' "$h" "$v" "${j:-$(head -c 200 "$TMP/err")}"
        [ -z "$first" ] && first=$v
        [ "$v" = "$first" ] || same=0
        eval "V_${h//-/_}=$v"
    done
    if [ $same = 1 ] && { [ "$first" = CLEAN ] || [ "$first" = INSTALLED ]; }; then AGREE=$first; fi
}

preflight() {
    status
    local h v
    for h in $HOSTS; do
        eval "v=\$V_${h//-/_}"
        case $v in CLEAN|INSTALLED) ;; *) die "preflight: $h is $v; nothing changed" ;; esac
    done
    echo "preflight OK (expected installed shas: $EXPECT)"
}

node_cmd() {  # node_cmd <host> install|uninstall
    rsh "$1" "cd ~/$STAGE_REL && $RPY install.py $2" </dev/null
}

cmd=${1:-status}
case $cmd in
status)
    status
    if [ -n "$AGREE" ]; then echo "OVERALL $AGREE, $(echo $HOSTS | wc -w | tr -d ' ') nodes agree"; exit 0; fi
    echo "OVERALL DISAGREE"; exit 2 ;;
preflight)
    preflight ;;
install)
    preflight
    touched=""
    for h in $HOSTS; do
        if ! stage "$h"; then echo "$h: staging failed" >&2; fail=$h; break; fi
        touched="$touched $h"
        if node_cmd "$h" install >"$TMP/$h.log" 2>&1 && [ "$(verdict "$(probe "$h")")" = INSTALLED ]; then
            echo "$h: INSTALLED"
        else
            echo "$h: install FAILED:"; cat "$TMP/$h.log"; fail=$h; break
        fi
    done
    if [ -n "${fail:-}" ]; then
        echo "rolling back:$touched"
        for h in $touched; do node_cmd "$h" uninstall >/dev/null 2>&1; echo "$h: $(verdict "$(probe "$h")")"; done
        exit 1
    fi
    status; [ "$AGREE" = INSTALLED ] || die "post-install status does not agree"
    echo "OVERALL INSTALLED, $(echo $HOSTS | wc -w | tr -d ' ') nodes agree" ;;
uninstall)
    preflight
    for h in $HOSTS; do
        eval "v=\$V_${h//-/_}"
        [ "$v" = CLEAN ] && { echo "$h: already CLEAN"; continue; }
        stage "$h" || die "$h: staging failed"
        node_cmd "$h" uninstall >"$TMP/$h.log" 2>&1 && [ "$(verdict "$(probe "$h")")" = CLEAN ] ||
            { cat "$TMP/$h.log"; die "$h: uninstall failed (nodes before it are CLEAN)"; }
        echo "$h: CLEAN"
    done
    status; [ "$AGREE" = CLEAN ] || die "post-uninstall status does not agree"
    echo "OVERALL CLEAN, $(echo $HOSTS | wc -w | tr -d ' ') nodes agree" ;;
*)
    sed -n '2,20p' "$0"; exit 2 ;;
esac
