#!/bin/bash
# One command: run manifest -> ~/feagent-viz/twin.usdc (+ .usdz, legends, summary.json) -> GPU preview renders.
#   ~/feagent-viz/export.sh path/to/run_manifest.json [extra export_twin.py args, e.g. --scale 50]
set -eu
D=$(cd "$(dirname "$0")" && pwd)
M=${1:?usage: export.sh run_manifest.json [export_twin.py args]}; shift
systemd-run --user --scope -q -p MemoryMax=8G ~/micromamba/envs/feagent/bin/python -W ignore \
  "$D/export_twin.py" "$M" --out "$D" "$@"
"$D/render.sh" "$D/twin.usdc" "$D/renders"
