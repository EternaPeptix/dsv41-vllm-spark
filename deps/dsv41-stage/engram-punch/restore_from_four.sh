#!/bin/bash
# restore_from_four.sh — run ON a punched node (never on spark-four), with vLLM stopped. Refills shards 47/48 from
# spark-four's pristine copy over the fabric, then checks the full sha256 against ~/models/dsv41/uncensored.sha.
# Needs: spark-one..three: id_ed25519 accepted by jeweled@10.77.0.41; spark-five..eight: ~/.ssh/engram_restore (read-only rrsync key).
# Needs ~165 GiB free (the punched space comes back as data).
set -euo pipefail
SRC=${SRC:-jeweled@10.77.0.41}
M=$HOME/models/dsv41/uncensored-fp8
# spark-five..eight (2026-10-23): dedicated key ~/.ssh/engram_restore, forced to "rrsync -ro <model dir>" on spark-four,
# so remote paths are relative to the model dir and the key can only read.
SSH="ssh -o ConnectTimeout=10"; RDIR=models/dsv41/uncensored-fp8/
if [ -f "$HOME/.ssh/engram_restore" ] && [ "$SRC" = jeweled@10.77.0.41 ]; then
  SSH="ssh -o ConnectTimeout=10 -o IdentitiesOnly=yes -i $HOME/.ssh/engram_restore"; RDIR=
fi
F="model-00047-of-00048.safetensors model-00048-of-00048.safetensors"
[ "$(hostname)" = spark-four ] && { echo "refusing: spark-four is the pristine source"; exit 1; }
pgrep -f "vllm serve" >/dev/null && { echo "refusing: vLLM is running; stop dsv41-rank first"; exit 1; }
need=0; for f in $F; do a=$(stat -c %b "$M/$f"); s=$(stat -c %s "$M/$f"); need=$((need + s - a*512)); done
free=$(df -B1 --output=avail "$M" | tail -1)
echo "need ~$((need/1073741824)) GiB more on disk, free $((free/1073741824)) GiB"
[ "$free" -gt $((need + 8*1073741824)) ] || { echo "refusing: not enough free space"; exit 1; }
for f in $F; do
  # -I: punching changed mtime anyway, but never trust size+mtime quick-check here; -W: whole file, no delta reads;
  # --inplace: rewrite the existing (holed) file in place so offsets/inode stay the same.
  nice -n 19 rsync -I -W --inplace --no-compress -e "$SSH" --info=progress2 "$SRC:$RDIR$f" "$M/$f"
  python3 -c "import os,sys; os.removexattr(sys.argv[1], 'user.dsv41.engram_punch')" "$M/$f" 2>/dev/null || true
done
cd "$M" && grep -E "model-0004[78]-of-00048" ../uncensored.sha | nice -n 19 sha256sum -c -
python3 -I "$HOME/dsv41-stage/engram-punch/engram_punch.py" --rank 0 --world 8 --dry-run | grep "holes now"
echo "restored; every rank's --check now passes on this node"
