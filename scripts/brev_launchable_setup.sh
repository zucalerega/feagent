#!/usr/bin/env bash
# Brev Launchable "VM mode" setup script: paste this into the Launchable wizard. Anyone who deploys the
# Launchable gets a box with the repo, all dependencies and the UI/solver service running on port 8000.
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
run "cd $DIR && bash scripts/setup_gpu_box.sh"
run "cd $DIR && bash scripts/serve.sh"
