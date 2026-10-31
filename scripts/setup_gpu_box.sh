#!/usr/bin/env bash
# Set up feacore on a fresh NVIDIA box: a Brev instance (usually x86_64) or a Dell GB10
# "DGX Spark" (aarch64). Run from the repo root:  bash scripts/setup_gpu_box.sh
#
# x86_64: everything from pip.
# aarch64: PyPI has no Linux ARM wheel for gmsh (only macOS arm64), so gmsh comes from conda-forge
#          via micromamba, and MKL PARDISO (x86-only) is skipped; the solver falls back to SciPy or
#          to cuDSS on the GPU.
# GPU: if nvidia-smi works, installs nvmath-python (cuDSS) + CuPy matching the driver's CUDA major.
set -euo pipefail

ARCH=$(uname -m)
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo "sudo")
echo "== arch: $ARCH"

echo "== system packages (gmsh runtime libs, CalculiX)"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq libglu1-mesa libxcursor1 libxft2 libxinerama1 libxrender1 libxi6 libgl1 \
  calculix-ccx curl bzip2 >/dev/null

if [ "$ARCH" = "x86_64" ]; then
  PY=python3
  $PY -m venv .venv && source .venv/bin/activate && PY=python
  $PY -m pip install -q --upgrade pip
  $PY -m pip install -q -e ".[dev]" pypardiso
else
  echo "== aarch64: python + gmsh from conda-forge (micromamba)"
  if ! command -v micromamba >/dev/null; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj -C "$HOME" bin/micromamba >/dev/null
    export PATH="$HOME/bin:$PATH"
  fi
  export MAMBA_ROOT_PREFIX="$HOME/micromamba"
  micromamba create -y -q -n feacore -c conda-forge python=3.12 gmsh python-gmsh numpy scipy meshio >/dev/null
  PY="$MAMBA_ROOT_PREFIX/envs/feacore/bin/python"
  $PY -m pip install -q -e ".[dev]" --no-deps
  $PY -m pip install -q "scikit-fem>=10" pytest matplotlib sympy
fi

if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  CUDA_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -1)
  echo "== GPU found, driver CUDA $CUDA_MAJOR: installing cuDSS (nvmath-python) + CuPy"
  if [ "${CUDA_MAJOR:-12}" -ge 13 ]; then
    $PY -m pip install -q "nvmath-python[cu13]" cupy-cuda13x || echo "!! GPU extras failed; CPU solve still works"
  else
    $PY -m pip install -q "nvmath-python[cu12]" cupy-cuda12x || echo "!! GPU extras failed; CPU solve still works"
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "== no GPU visible: CPU solve only"
fi

echo "== smoke check"
$PY - <<'EOF'
import gmsh, skfem, feacore
from feacore.solver import BACKENDS
print("gmsh", gmsh.__version__, "| scikit-fem", skfem.__version__, "| backends", sorted(BACKENDS))
try:
    from feacore.solver import linsolve
    print("linear solvers:", getattr(linsolve, "available_methods", lambda: "n/a")())
except Exception as e:  # the module may not exist on older checkouts
    print("linsolve check skipped:", e)
EOF
echo "== done. Python: $PY"
echo "   UI:  $PY -m feacore.ui --host 0.0.0.0 --port 8000   (then expose port 8000 with Brev)"
