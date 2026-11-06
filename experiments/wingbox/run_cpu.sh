#!/bin/bash
# Wingbox stage 1 from scratch on a CPU machine (no GPU needed for the small cases).
#   experiments/wingbox/run_cpu.sh [case ...]     cases: assembly_c2 assembly_c1 assembly assembly_xf (default: assembly_c2)
# Steps: build geometry (gmsh OCC) -> mesh each case (feagent check) -> feacore case.json -> solve (PARDISO) ->
# result in <case>/cpu/{result.json,result.vtu,watchdog.txt}. Then verify with
#   SOLVE_DIR=cpu uv run python experiments/wingbox/verify/verify_all.py     (needs all four cases solved)
# The expected GCI/verdict numbers in experiments/README.md come from the Spark cuDSS solves of all four meshes
# (assembly_xf is ~1.1 M DOFs: use the Spark, or a CPU with >= 64 GB RAM).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE/../.."          # repo root
cases=("$@"); [ ${#cases[@]} -eq 0 ] && cases=(assembly_c2)
uv run python experiments/wingbox/geometry/build_wingbox.py
for c in "${cases[@]}"; do
  d=experiments/wingbox/$c
  uv run feagent check "$d/mesh.yaml" --no-render --json > "$d/check.json" || true
  uv run python -m feacore.adapters.mesh_harness "$d/out" --out "$d/case.json"
  python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['mesh']='out/mesh.msh'; json.dump(d,open(p,'w'),indent=2)" "$d/case.json"
  mkdir -p "$d/cpu"; t0=$(date +%s)
  uv run python -m feacore.cli "$d/case.json" --out "$d/cpu/result.json" --vtu "$d/cpu/result.vtu" --linear-solver pardiso
  echo "wall_s=$(( $(date +%s) - t0 )) min_MemAvailable_GiB=0 drop_GiB=0 cgroup_peak_GiB=0 killed=0 (cpu)" > "$d/cpu/watchdog.txt"
done
