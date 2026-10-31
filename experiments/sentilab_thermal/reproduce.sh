#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../.."
"${PYTHON_BIN:-python3}" -m venv .venv-sentilab
.venv-sentilab/bin/python -m pip install -r experiments/sentilab_thermal/requirements.txt
.venv-sentilab/bin/python experiments/sentilab_thermal/run.py --out evaluation_runs/sentilab_thermal_reproduced
