#!/bin/bash
# Prepare + run the platen-compression experiment from a fresh clone (CPU only).
#   experiments/physical/prepare.sh [WORKDIR] [part ...]       default WORKDIR=experiments/physical/work, parts: abc_00000003 abc_00000065
# Env: COMP_E_PA (Young's modulus in Pa, default 2.23e9 = Alpha Fast Resin), COMP_FORCE_N (default 100).
# Steps: copy the two gold STEPs into WORKDIR -> `feagent automesh` + group/BC specs + feacore solve (run_comp.py)
#        -> WORKDIR/<part>/{mesh.yaml,case.json,result.vtu,summary.json}.
# Then (optional) crush-to-snap: see experiments/README.md.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); cd "$ROOT"
WORK=${1:-$HERE/work}; shift || true
parts=("$@"); [ ${#parts[@]} -eq 0 ] && parts=(abc_00000003 abc_00000065)
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
for p in "${parts[@]}"; do cp "benchmarks/gold/${p}_bc0/public/geometry.step" "$WORK/$p.step"; done
uv run python experiments/physical/run_comp.py "$WORK" "${parts[@]}"
