#!/usr/bin/env bash
# One-command setup of feagent / feacore on ANY fresh Ubuntu box (any Brev account, a Launchable, or
# your own machine). It clones the repo into ~/feagent (or updates it), installs everything, and starts
# the UI/solver service on port 8000 with an auto-update that follows the branch (main by default).
#
#   curl -fsSL https://raw.githubusercontent.com/zucalerega/feagent/main/scripts/bootstrap.sh | bash
#
# Then on Brev: console > your instance > HTTP Ports > add port 8000, and open the https link.
# Env: REPO_URL, BRANCH (main), FEACORE_UI_TOKEN (optional; not needed behind Brev's HTTP Ports sign-in).
set -euo pipefail
REPO_URL=${REPO_URL:-https://github.com/zucalerega/feagent}
BRANCH=${BRANCH:-main}
USER_NAME=${SUDO_USER:-$(id -un)}
[ "$USER_NAME" = root ] && id ubuntu >/dev/null 2>&1 && USER_NAME=ubuntu
HOME_DIR=$(getent passwd "$USER_NAME" | cut -d: -f6)
DIR="$HOME_DIR/feagent"
run() { if [ "$(id -un)" = "$USER_NAME" ]; then bash -c "$1"; else sudo -u "$USER_NAME" -H bash -c "$1"; fi; }
command -v git >/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq git)
[ -d "$DIR/.git" ] || run "git clone --branch $BRANCH $REPO_URL $DIR"
run "cd $DIR && git pull --ff-only || true"
if [ "$(id -u)" = 0 ]; then   # as root (e.g. a Launchable setup script): install system packages now
  apt-get update -qq && apt-get install -y -qq $(grep -v '^#' "$DIR/scripts/system_packages.txt") >/dev/null
fi
run "cd $DIR && bash scripts/setup_gpu_box.sh"
run "cd $DIR && PORT=${PORT:-8000} FEACORE_UI_TOKEN=${FEACORE_UI_TOKEN:-} bash scripts/serve.sh"
echo
echo "feagent is set up in $DIR and serving on port 8000."
echo "Brev: console > your instance > HTTP Ports > add port 8000, then open the https link."
