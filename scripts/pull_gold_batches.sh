#!/usr/bin/env bash
# Copy new gold-benchmark archives (scripts/goldgen_batcher.py) from the GPU nodes to this machine.
# Meant for cron every 5 minutes:
#   */5 * * * * $HOME/Desktop/feagent/scripts/pull_gold_batches.sh >> $HOME/Desktop/gold_benchmarks/pull.log 2>&1
# Needs the `autor` CLI, logged in once (`autor login`). Only RUNNING nodes are contacted: a stopped
# node is never woken (that would bill it). Each archive is checked against the node's sha256,
# kept in archives/ and unpacked into DEST. <node>.complete is written once the node's batcher has
# finished (batches/DONE) and every one of its archives is here: then that node is safe to stop.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
NODES=${GOLD_NODES:-"goldgen-l2 goldgen-l2b"}
REMOTES=${GOLD_REMOTES:-"gold gold2 gold_l3"}   # run folders under ~ on each node (gold2 = second batch, gold_l3 = L1-L3 run)
DEST=${GOLD_DEST:-$HOME/Desktop/gold_benchmarks}
mkdir -p "$DEST/archives"
exec 9>"$DEST/.pull.lock"
flock -n 9 || { echo "$(date -Is) previous pull still running"; exit 0; }

for N in $NODES; do
  state=$(autor node get "$N" --json --quiet 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>/dev/null)
  case "$state" in running*) ;; *) echo "$(date -Is) $N: ${state:-unknown}, skipped"; continue ;; esac
  for R in $REMOTES; do
  [ -f "$DEST/$N.$R.complete" ] || { [ "$R" = gold ] && [ -f "$DEST/$N.complete" ]; } && continue
  listing=$(autor run "$N" --timeout 60 -- sh -c "cd ~/$R/batches 2>/dev/null && { sha256sum *.tgz 2>/dev/null; [ -f DONE ] && echo DONE; }")
  [ -n "$listing" ] || continue
  new=0
  while read -r sha name; do
    [[ "${name:-}" == *.tgz && ${#sha} -eq 64 ]] || continue
    grep -qx "$name" "$DEST/pulled.txt" 2>/dev/null && continue
    part="$DEST/archives/$name.part"
    if autor cp "$N:~/$R/batches/$name" "$part" >/dev/null && echo "$sha  $part" | sha256sum -c --quiet; then
      mv "$part" "$DEST/archives/$name" && tar -xzf "$DEST/archives/$name" -C "$DEST" \
        && echo "$name" >> "$DEST/pulled.txt" && new=$((new + 1))
    else
      echo "$(date -Is) $N: $name failed (will retry)"; rm -f "$part"
    fi
  done < <(printf '%s\n' "$listing" | grep -v '^DONE$')
  total=$(printf '%s\n' "$listing" | grep -c '\.tgz$')
  have=$(printf '%s\n' "$listing" | awk '/\.tgz$/{print $2}' | grep -cxFf "$DEST/pulled.txt" 2>/dev/null); have=${have:-0}
  echo "$(date -Is) $N:$R: +$new new, $have/$total archives here, $(ls -d "$DEST"/out/*_bc* 2>/dev/null | wc -l) BC sets on this machine"
  if printf '%s\n' "$listing" | grep -qx DONE && [ "$have" -eq "$total" ]; then
    date -Is > "$DEST/$N.$R.complete"; echo "$(date -Is) $N:$R: run finished and fully copied -> safe to stop"
  fi
  done
done
