#!/usr/bin/env bash
# Gold-benchmark pilot on one GPU node: one ABC part, nested tet10 levels with L2 ~250k cells,
# a few BC sets solved on every level (cuDSS when available), plus renders of the finest mesh.
#   bash scripts/goldgen_pilot.sh [PART=00000003] [BCS=2] [H0=0.237]
# Output: ~/pilot/ (out/ benchmarks, timing.log, L2.png, L2.html, summary.json)
set -euo pipefail
PART=${1:-00000003}; BCS=${2:-2}; H0=${3:-0.237}
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=~/pilot; mkdir -p "$W"; cd "$W"
T0=$(date +%s)
stamp() { echo "$(( $(date +%s) - T0 ))s $*" | tee -a "$W/timing.log"; }

stamp "setup start"
if [ ! -x "$REPO/.venv/bin/python" ]; then (cd "$REPO" && bash scripts/setup_gpu_box.sh) > "$W/setup.log" 2>&1; fi
sudo apt-get install -y -qq p7zip-full >/dev/null
PY="$REPO/.venv/bin/python"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$W/timing.log"
"$PY" -c "from feacore.solver import linsolve; print('linear solvers:', linsolve.available_methods())" | tee -a "$W/timing.log"
stamp "setup done"

if [ ! -f abc_0000_step_v00.7z ]; then
  curl -sSL -o abc_0000_step_v00.7z https://archive.nyu.edu/rest/bitstreams/88598/retrieve
fi
7z x -y -oabc "abc_0000_step_v00.7z" "$PART/*" >/dev/null
stamp "STEP downloaded + extracted ($PART)"

"$PY" -m feacore.goldgen screen abc --out parts.jsonl --workers 1
stamp "screened"

THREADS=${GMN_CPU_LIMIT:-8}
PYTHONUNBUFFERED=1 "$PY" -W ignore -m feacore.goldgen run parts.jsonl abc --out out --parts 1 \
  --bcs "$BCS" --min-bcs 1 --levels 3 --h0-frac "$H0" --threads "${THREADS%.*}" --linear-solver cudss --mesh-workers 1 \
  2>&1 | grep -av "^\s*$" | tee -a "$W/run.log"
stamp "goldgen run done"

L2=$(ls out/_meshes/*/L2.msh | head -1)
"$PY" -W ignore "$REPO/scripts/view_mesh.py" "$L2" --out "$W/L2.html" --title "abc_$PART  (level L2, H100 pilot)"
stamp "viewer written"
"$PY" -W ignore "$REPO/scripts/render_mesh.py" "$L2" --out "$W/L2.png" --title "abc_$PART  (level L2, H100 pilot)" || true
stamp "render written"

"$PY" - <<EOF > "$W/summary.json"
import json
solves = [json.loads(l) for l in open("out/solves.jsonl")]
summ = [json.loads(l) for l in open("out/summary.jsonl")]
print(json.dumps({"meshes": [s for s in solves if "mesh_level" in s],
                  "solves": [s for s in solves if "level" in s], "bcsets": summ}, indent=1))
EOF
cp "$W/summary.json" "${GMN_RESULT_PATH:-/dev/null}" 2>/dev/null || true
stamp "all done"
