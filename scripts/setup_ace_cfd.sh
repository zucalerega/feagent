#!/usr/bin/env bash
# Prepare a Linux box (Brev / Ubuntu 22.04-24.04) to run the ACE loop on the CFD suites with the
# Claude API. Run from the repo root, after scripts/setup_gpu_box.sh (or bootstrap.sh) made .venv:
#
#   bash scripts/setup_ace_cfd.sh
#
# 1. OpenFOAM, ESI line: Ubuntu's own `openfoam` package (v1912 on 24.04, also built for arm64), which
#    the CFD benchmarks were validated with. Skipped when an ESI OpenFOAM is already found.
# 2. The ace and cfd extras (anthropic SDK, foamlib) into the existing Python environment.
# 3. python -m ace.preflight on both CFD suites: every task's good mesh solves within tolerance.
# On a Mac use scripts/openfoam_docker.sh instead of step 1 (ace/README.md, "The CFD suite").
set -euo pipefail
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo "sudo")

if [ -x .venv/bin/python ]; then
  PY="$PWD/.venv/bin/python"
elif [ -x "$HOME/micromamba/envs/feacore/bin/python" ]; then
  PY="$HOME/micromamba/envs/feacore/bin/python"     # aarch64 (DGX Spark) environment of setup_gpu_box.sh
else
  echo "!! no Python environment: run bash scripts/setup_gpu_box.sh first"; exit 1
fi

echo "== OpenFOAM"
if $PY -c "from feacore.solver._foam_env import find_openfoam; import sys; sys.exit(find_openfoam() is None)" 2>/dev/null; then
  echo "   already installed"
else
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq openfoam >/dev/null
fi

echo "== Python extras: ace, cfd"
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
if [ "$PY" = "$PWD/.venv/bin/python" ] && [ -x "$UV" ]; then
  "$UV" pip install -q --python "$PY" -e ".[dev,ace,cfd]"
elif [ "$PY" = "$PWD/.venv/bin/python" ]; then
  "$PY" -m pip install -q -e ".[dev,ace,cfd]"
else   # aarch64 conda env: gmsh came from conda-forge, so only add the extras' packages
  "$PY" -m pip install -q anthropic "pydantic>=2" rapidfuzz matplotlib streamlit plotly altair foamlib
fi

echo "== pre-flight (no API calls)"
$PY -m ace.preflight --suite cfd
$PY -m ace.preflight --suite cfd3d
echo
echo "Next: export ANTHROPIC_API_KEY=..., then  $PY -m ace.preflight --suite cfd --ping  (one call, ~\$0.01)"
echo "and the run itself, e.g.  $PY -m ace.run_loop --suite cfd --budget-usd 30   (ace/README.md, 'The CFD suite')"
