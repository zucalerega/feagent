#!/usr/bin/env bash
# Run the feacore UI + HTTP solver service as a boot-persistent service on this box, and keep it
# up to date with the checked-out branch (pulls new commits every minute, see scripts/autoupdate.sh).
#   bash scripts/serve.sh             install/refresh the services and (re)start them
#   bash scripts/serve.sh logs        follow the service log      (update-logs: the auto-update log)
#   bash scripts/serve.sh stop        stop the service and the auto-update
# Env: PORT (8000), FEACORE_UI_TOKEN (optional shared token; strongly advised behind a public tunnel),
#      AUTOUPDATE=0 to skip the auto-update, FEACORE_BRANCH to follow a branch other than the current one.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
PORT=${PORT:-8000}
PY=$(cat .feacore_python 2>/dev/null || echo "$REPO/.venv/bin/python")
THREADS=$(( $(nproc) / 4 )); [ "$THREADS" -ge 1 ] || THREADS=1
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo "sudo")
UI_CMD="OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS $PY -m feacore.ui --host 0.0.0.0 --port $PORT"
HAVE_SYSTEMD=$(command -v systemctl >/dev/null && [ -d /run/systemd/system ] && echo 1 || echo 0)

case "${1:-start}" in
  stop)
    if [ "$HAVE_SYSTEMD" = 1 ]; then $SUDO systemctl stop feacore-autoupdate.timer feacore-ui 2>/dev/null || true
    else tmux kill-session -t feacore-update 2>/dev/null; tmux kill-session -t feacore-ui 2>/dev/null; fi
    exit 0 ;;
  logs) [ "$HAVE_SYSTEMD" = 1 ] && exec journalctl -u feacore-ui -f || exec tmux attach -t feacore-ui ;;
  update-logs) [ "$HAVE_SYSTEMD" = 1 ] && exec journalctl -u feacore-autoupdate -f || exec tmux attach -t feacore-update ;;
  ui-only)  # used by the tmux-mode auto-update to restart just the UI
    tmux kill-session -t feacore-ui 2>/dev/null || true
    FEACORE_UI_TOKEN=${FEACORE_UI_TOKEN:-$(cat .feacore_token 2>/dev/null || true)} \
      tmux new -d -s feacore-ui "$UI_CMD"
    exit 0 ;;
esac

[ -x "$PY" ] || { echo "no Python env yet: run  bash scripts/setup_gpu_box.sh  first"; exit 1; }
[ -n "${FEACORE_UI_TOKEN:-}" ] && { echo "$FEACORE_UI_TOKEN" > .feacore_token; chmod 600 .feacore_token; }

if [ "$HAVE_SYSTEMD" = 1 ]; then
  $SUDO tee /etc/systemd/system/feacore-ui.service >/dev/null <<UNIT
[Unit]
Description=feacore UI and solver service
After=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$REPO
Environment=OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS FEACORE_UI_TOKEN=${FEACORE_UI_TOKEN:-}
ExecStart=$PY -m feacore.ui --host 0.0.0.0 --port $PORT
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT
  if [ "${AUTOUPDATE:-1}" != 0 ]; then
    $SUDO tee /etc/systemd/system/feacore-autoupdate.service >/dev/null <<UNIT
[Unit]
Description=feacore auto-update: pull new commits and restart the service

[Service]
Type=oneshot
User=$(id -un)
WorkingDirectory=$REPO
Environment=PORT=$PORT FEACORE_BRANCH=${FEACORE_BRANCH:-} "FEACORE_RESTART_CMD=sudo -n systemctl restart feacore-ui"
ExecStart=/bin/bash $REPO/scripts/autoupdate.sh
UNIT
    $SUDO tee /etc/systemd/system/feacore-autoupdate.timer >/dev/null <<UNIT
[Unit]
Description=Run the feacore auto-update every minute

[Timer]
OnBootSec=2min
OnUnitActiveSec=60s

[Install]
WantedBy=timers.target
UNIT
  fi
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now feacore-ui >/dev/null 2>&1
  $SUDO systemctl restart feacore-ui
  [ "${AUTOUPDATE:-1}" != 0 ] && $SUDO systemctl enable --now feacore-autoupdate.timer >/dev/null 2>&1
  echo "feacore-ui: systemd service (starts on boot). Logs: bash scripts/serve.sh logs"
else
  bash "$0" ui-only
  if [ "${AUTOUPDATE:-1}" != 0 ]; then
    tmux kill-session -t feacore-update 2>/dev/null || true
    tmux new -d -s feacore-update "while true; do PORT=$PORT FEACORE_RESTART_CMD='bash scripts/serve.sh ui-only' bash scripts/autoupdate.sh; sleep 60; done"
  fi
  echo "feacore-ui: tmux session 'feacore-ui' (no systemd here; rerun this script after a reboot)"
fi
[ "${AUTOUPDATE:-1}" != 0 ] && echo "auto-update: follows origin/${FEACORE_BRANCH:-$(git rev-parse --abbrev-ref HEAD)} every minute (bash scripts/serve.sh update-logs)"
for _ in $(seq 1 30); do curl -fs "http://localhost:$PORT/api/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fs "http://localhost:$PORT/api/health" >/dev/null 2>&1 && echo "healthy on port $PORT" \
  || echo "not answering yet on port $PORT; check: bash scripts/serve.sh logs"
echo "Expose it: Brev console > your instance > HTTP Ports > add port $PORT (or: brev port-forward <instance> --port $PORT:$PORT)"
