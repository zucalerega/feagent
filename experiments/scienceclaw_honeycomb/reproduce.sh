#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python_bin="${HONEYCOMB_PYTHON:-.venv/bin/python}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${TMPDIR:-/tmp}/feagent-mpl" MPLBACKEND=Agg
"$python_bin" - <<'PY'
import hashlib
from pathlib import Path
expected={
    'SM-W1.stl':'5a60deb5dc602bed4e0a4de44ac803b3c8b56b7db098b5008a631b4797577307',
    'INPUTS.md':'7a5d923a8091721e210760783337ec989bee80d74a1662a09604625650ffc9f4',
}
for name,sha in expected.items():
    p=Path('experiments/scienceclaw_honeycomb/inputs')/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==sha, f'Input changed: {name}'
print('Both frozen input hashes verified.')
PY
"$python_bin" experiments/scienceclaw_honeycomb/verify.py
"$python_bin" experiments/scienceclaw_honeycomb/run.py --h .6 .3 .15 .1 .075 --nu .35
"$python_bin" experiments/scienceclaw_honeycomb/run.py --h .075 --nu .4
"$python_bin" experiments/scienceclaw_honeycomb/summarize.py
