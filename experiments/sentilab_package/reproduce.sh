#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python_bin="${PACKAGE_PYTHON:-.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
output_dir="${PACKAGE_OUT:-evaluation_runs/sentilab_package_reproduced}"
solver="${PACKAGE_SOLVER:-pardiso}"
(cd experiments/sentilab_package/inputs && sha256sum -c SHA256SUMS)
for refinement in 1 2 3; do
  for caseid in A B; do
    "$python_bin" experiments/sentilab_package/run.py --case "$caseid" --refinement "$refinement" \
      --solver "$solver" --out "$output_dir/${caseid}_r${refinement}"
  done
done
for caseid in A B; do
  "$python_bin" experiments/sentilab_package/run.py --case "$caseid" --refinement 3 \
    --variant surface --solver "$solver" --out "$output_dir/${caseid}_surface_r3"
done
"$python_bin" experiments/sentilab_package/summarize.py "$output_dir"
