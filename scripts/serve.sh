#!/usr/bin/env bash
# Run the feacore UI + HTTP solver service as a boot-persistent service on this box.
#   bash scripts/serve.sh             install/refresh a systemd service (or tmux if no systemd) and start it
#   bash scripts/serve.sh stop|logs   stop it / follow its logs
# Env: PORT (8000), FEACORE_UI_TOKEN (optional shared token; strongly advised behind a public tunnel).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
PORT=${PORT:-8000}
PY=$(cat .feacore_python 2>/dev/null || echo "$REPO/.venv/bin/python")
[ -x "$PY" ] || { echo "no Python env yet: run  bash scripts/setup_gpu_box.sh  first"; exit 1; }
THREADS=$(( $(nproc) / 4 )); [ "$THREADS" -ge 1 ] || THREADS=1
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo "sudo")
UNIT=/etc/systemd/system/feacore-ui.service

case "${1:-start}" in
  stop) ($SUDO systemctl stop feacore-ui 2>/dev/null || tmux kill-session -t feacore-ui 2>/dev/null) || true; exit 0 ;;
  logs) (journalctl -u feacore-ui -f 2>/dev/null) || tmux attach -t feacore-ui; exit 0 ;;
esac

if command -v systemctl >/dev/null && [ -d /run/systemd/system ]; then
  $SUDO tee "$UNIT" >/dev/null <<UNITEOF
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
UNITEOF
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now feacore-ui >/dev/null
  $SUDO systemctl restart feacore-ui
  echo "feacore-ui running as a systemd service (starts on boot). Logs: bash scripts/serve.sh logs"
else
  tmux kill-session -t feacore-ui 2>/dev/null || true
  tmux new -d -s feacore-ui "OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS $PY -m feacore.ui --host 0.0.0.0 --port $PORT"
  echo "feacore-ui running in tmux session 'feacore-ui' (no systemd here; restart after reboots)"
fi
for i in $(seq 1 30); do curl -fs "http://localhost:$PORT/api/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fs "http://localhost:$PORT/api/health" >/dev/null 2>&1 && echo "healthy on port $PORT" || echo "not answering yet on port $PORT; check: bash scripts/serve.sh logs"
echo "Expose it: Brev console > instance > Access > Using Tunnels > port $PORT (or: brev port-forward <instance> --port $PORT:$PORT)"
