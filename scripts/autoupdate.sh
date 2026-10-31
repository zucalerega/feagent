#!/usr/bin/env bash
# Pull new commits of the checked-out branch and restart the feacore service. Run every minute by
# the feacore-autoupdate systemd timer (installed by scripts/serve.sh); safe to run by hand.
#   - fast-forward only; a dirty working tree or a diverged branch is logged and skipped
#   - re-runs scripts/setup_gpu_box.sh when dependencies or the setup script changed
#   - defers the restart (up to ~10 min) while the service reports active jobs
# Env: FEACORE_BRANCH (default: current branch), PORT (8000), FEACORE_RESTART_CMD (for testing).
set -uo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8000}
BRANCH=${FEACORE_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}
STATE=.feacore_autoupdate_deferrals
RESTART=${FEACORE_RESTART_CMD:-"systemctl restart feacore-ui"}
log() { echo "[autoupdate $(date -u +%H:%M:%S)] $*"; }

git fetch -q origin "$BRANCH" || { log "fetch failed"; exit 0; }
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")
if [ "$LOCAL" = "$REMOTE" ] && [ ! -f "$STATE" ]; then exit 0; fi

if [ "$LOCAL" != "$REMOTE" ]; then
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    log "local changes in the working tree; not updating"; exit 0
  fi
  if ! git merge-base --is-ancestor "$LOCAL" "$REMOTE"; then
    log "local branch has diverged from origin/$BRANCH; not updating"; exit 0
  fi
  CHANGED=$(git diff --name-only "$LOCAL" "$REMOTE")
  git merge -q --ff-only "origin/$BRANCH" || { log "fast-forward failed"; exit 0; }
  log "updated ${LOCAL:0:7} -> ${REMOTE:0:7} ($(echo "$CHANGED" | wc -l) files)"
  if echo "$CHANGED" | grep -q -E '^(pyproject.toml|uv.lock|requirements.txt|scripts/setup_gpu_box.sh)$'; then
    log "dependencies changed: re-running setup"
    bash scripts/setup_gpu_box.sh >/dev/null 2>&1 || log "setup failed (service keeps the old environment)"
  fi
  echo 0 > "$STATE"
fi

ACTIVE=$(curl -fs "http://localhost:$PORT/api/health" 2>/dev/null \
  | python3 -c 'import json,sys; print(int(json.load(sys.stdin).get("active_jobs", 0)))' 2>/dev/null || echo 0)
N=$(cat "$STATE" 2>/dev/null || echo 0)
if [ "$ACTIVE" -gt 0 ] && [ "$N" -lt 10 ]; then
  echo $((N + 1)) > "$STATE"; log "$ACTIVE active job(s); restart deferred ($((N + 1))/10)"; exit 0
fi
rm -f "$STATE"
$RESTART && log "service restarted"
