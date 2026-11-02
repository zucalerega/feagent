#!/usr/bin/env bash
# Gold-benchmark production pilot on one H100 node: CPU cores mesh (12 gmsh processes) while solve
# processes share the GPU (cuDSS). Screens ABC chunk 0000, then N parts x BCS BC sets x 3 levels.
#   bash scripts/goldgen_h100.sh [NPARTS=10] [BCS=10] [MESH_WORKERS=12] [SOLVE_WORKERS=3] [CANDIDATES=300] \
#                                [SHARD=0/1] [GPU_SLOTS=4]   (RENDER=1 also writes the abc_00000003 L2 results page)
# GOLD_W=~/gold2 CAND_START=1500 runs a fresh batch on ABC files CAND_START..CAND_START+CANDIDATES-1.
# Output: ~/gold/ (out/ benchmarks + solves.jsonl + gpu.jsonl, run.log, L2_results.html, summary.json)
set -euo pipefail
NPARTS=${1:-10}; BCS=${2:-10}; MW=${3:-12}; SW=${4:-3}; NCAND=${5:-300}; SHARD=${6:-0/1}; SLOTS=${7:-4}
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=${GOLD_W:-$HOME/gold}; CAND0=${CAND_START:-0}; mkdir -p "$W"; cd "$W"
T0=$(date +%s)
NOISE='No multithreading interface|Statistics on Transfer|Transfer Mode|Transferring Shape|WorkSession|Step File Name|^\s*(\x1b\[[0-9;]*m)?\**(\x1b\[0m)?\s*$' 
stamp() { echo "$(( $(date +%s) - T0 ))s $*" | tee -a "$W/timing.log"; }

stamp "setup start"
if [ ! -x "$REPO/.venv/bin/python" ]; then (cd "$REPO" && bash scripts/setup_gpu_box.sh) > "$W/setup.log" 2>&1; fi
# a woken node keeps only ~ (the venv): system packages (gmsh's GL/X libs, 7z) must be reinstalled
if ! command -v 7z >/dev/null || ! ldconfig -p | grep -q libGLU.so.1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq p7zip-full libglu1-mesa libxcursor1 libxft2 \
    libxinerama1 libxrender1 libxi6 libgl1 >/dev/null
fi
PY="$REPO/.venv/bin/python"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$W/timing.log"
echo "cpu_limit=${GMN_CPU_LIMIT:-?} mem_limit=${GMN_MEMORY_LIMIT_BYTES:-?}" | tee -a "$W/timing.log"
stamp "setup done"

[ -f abc.7z ] || { [ -f ~/gold/abc.7z ] && ln -s ~/gold/abc.7z abc.7z; } || curl -sSL -o abc.7z https://archive.nyu.edu/rest/bitstreams/88598/retrieve
7z x -y -oabc abc.7z $(for i in $(seq "$CAND0" $((CAND0 + NCAND - 1))); do printf '%08d/* ' "$i"; done) >/dev/null
stamp "STEP downloaded + $NCAND candidates extracted (from $CAND0)"

CPUS=${GMN_CPU_LIMIT:-8}; CPUS=${CPUS%.*}
"$PY" -W ignore -m feacore.goldgen screen abc --out parts.jsonl --workers "$CPUS" | tee -a "$W/run.log"
stamp "screened"

PYTHONUNBUFFERED=1 "$PY" -W ignore -m feacore.goldgen run parts.jsonl abc --out out --parts "$NPARTS" \
  --bcs "$BCS" --min-bcs 5 --levels 3 --h0-frac 0.237 --mesh-workers "$MW" --solve-workers "$SW" \
  --shard "$SHARD" --gpu-slots "$SLOTS" --linear-solver cudss 2>&1 | grep -av -E "$NOISE" | tee -a "$W/run.log"
stamp "goldgen run done"

L2=out/_meshes/abc_00000003/L2.msh
if [ "${RENDER:-0}" = 1 ] && [ -f "$L2" ]; then
  PYTHONUNBUFFERED=1 "$PY" -W ignore "$REPO/scripts/view_results.py" parts.jsonl abc_00000003 "$L2" --bcs 10 \
    --out "$W/L2_results.html" --title "abc_00000003  (L2 curved Tet10, 10 BC sets, H100)" 2>&1 \
    | grep -av -E "$NOISE" | tee -a "$W/run.log"
  stamp "L2 results page written"
fi

"$PY" - <<'EOF' > "$W/summary.json"
import json, collections, statistics
meshes = [json.loads(l) for l in open("out/meshes.jsonl")]
summ = [json.loads(l) for l in open("out/summary.jsonl")]
gpu = [json.loads(l) for l in open("out/gpu.jsonl")] if __import__("os").path.exists("out/gpu.jsonl") else []
parts = collections.defaultdict(list)
for m in meshes:
    parts[m["part"]].append(m)
plan = {p: ("straight" if any(m.get("fallback_reason") for m in ms) else "curved") for p, ms in parts.items()}
print(json.dumps({
    "parts_meshed": len(parts), "parts_curved": sum(v == "curved" for v in plan.values()),
    "parts_straight": sum(v == "straight" for v in plan.values()),
    "bcsets": len(summ), "bcsets_ok": sum(s.get("result") == "ok" for s in summ),
    "bcsets_invalid": sum(s.get("result") == "invalid_setup" for s in summ),
    "bcsets_error": sum(s.get("result") == "error" for s in summ),
    "gold_qois": sum(len(s.get("gold") or []) for s in summ),
    "errors": [s.get("error") for s in summ if s.get("result") == "error"][:5],
    "gpu_util_mean": statistics.mean(g["gpu_util"] for g in gpu) if gpu else None,
    "gpu_util_samples": len(gpu), "gpu_mem_max_mib": max((g["gpu_mem_mib"] for g in gpu), default=None),
    "max_cells": max(m["cells"] for m in meshes) if meshes else None,
}, indent=1))
EOF
cat "$W/summary.json"
cp "$W/summary.json" "${GMN_RESULT_PATH:-/dev/null}" 2>/dev/null || true
stamp "all done"
