#!/bin/bash
# Installer cycle on a byte copy of the live base: bash tests/installer_cycle.sh <live dsv41_l2_prefetch.py copy>
set -u
KIT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python3}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
D=$T/vllm/distributed/device_communicators; mkdir -p "$D"; cp "$1" "$D/dsv41_l2_prefetch.py"
export R1PF_VLLM=$T/vllm
run() { echo "== $*"; "$PY" "$KIT/install.py" "$@" 2>&1 | sed "s#$T#<tmp>#g"; echo "   (exit ${PIPESTATUS[0]})"; }
run status; run dry; run install; run install; run status
echo "== append a foreign line to the patched file"; echo "# foreign" >> "$D/dsv41_l2_prefetch.py"
run status; run uninstall
echo "== remove the foreign line again"; "$PY" - "$D/dsv41_l2_prefetch.py" <<'P'
import sys; p=sys.argv[1]; s=open(p).read(); open(p,"w").write(s[:-len("# foreign\n")])
P
run status; run uninstall; run status
sha256sum "$D/dsv41_l2_prefetch.py" 2>/dev/null | sed "s#$T#<tmp>#g" || shasum -a 256 "$D/dsv41_l2_prefetch.py" | sed "s#$T#<tmp>#g"
echo "== drifted base"; echo "# drift" >> "$D/dsv41_l2_prefetch.py"; run install; grep -c "# r1-pf" "$D/dsv41_l2_prefetch.py" || true
