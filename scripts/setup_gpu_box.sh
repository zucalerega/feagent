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
PKGS=$(grep -v '^#' "$(dirname "$0")/system_packages.txt")
MISSING=$(for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || echo "$p"; done)
if [ -n "$MISSING" ]; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq $MISSING >/dev/null
else
  echo "   all present"
fi

if [ "$ARCH" = "x86_64" ]; then
  # uv gives Python 3.12 regardless of the image's system Python (the mesh harness needs >= 3.12)
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || python3 -m pip install -q --user uv \
      || python3 -m pip install -q --break-system-packages --user uv
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv venv -q --python ">=3.12" .venv
  PY="$PWD/.venv/bin/python"
  PIP="uv pip install -q --python $PY"   # uv venvs have no pip module
  $PIP -e ".[dev]" pypardiso fastapi uvicorn python-multipart httpx
else
  echo "== aarch64: python + gmsh from conda-forge (micromamba)"
  if ! command -v micromamba >/dev/null; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj -C "$HOME" bin/micromamba >/dev/null
    export PATH="$HOME/bin:$PATH"
  fi
  export MAMBA_ROOT_PREFIX="$HOME/micromamba"
  micromamba create -y -q -n feacore -c conda-forge python=3.12 gmsh python-gmsh numpy scipy meshio >/dev/null
  PY="$MAMBA_ROOT_PREFIX/envs/feacore/bin/python"
  PIP="$PY -m pip install -q"
  $PIP -e ".[dev]" --no-deps
  $PIP "scikit-fem>=10" pytest matplotlib sympy fastapi uvicorn python-multipart httpx
fi

if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  CUDA_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -1)
  echo "== GPU found, driver CUDA $CUDA_MAJOR: installing cuDSS (nvmath-python) + CuPy"
  if [ "${CUDA_MAJOR:-12}" -ge 13 ]; then
    $PIP "nvmath-python[cu13]" cupy-cuda13x || echo "!! GPU extras failed; CPU solve still works"
  else
    $PIP "nvmath-python[cu12]" cupy-cuda12x || echo "!! GPU extras failed; CPU solve still works"
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "== no GPU visible: CPU solve only"
fi

echo "== smoke check"
trap 'echo "!! setup FAILED at the smoke check (error above). A missing system library usually means apt"; echo "   could not install one of the packages listed; fix that and re-run this script."' ERR
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
echo "$PY" > .feacore_python   # read by scripts/serve.sh
echo "== done. Python: $PY"
echo "   GPU check:  $PY scripts/gpu_check.py"
echo "   Headless renders (mesh harness / pyvista) need a virtual display: prefix with  xvfb-run -a"
echo "   UI:         tmux new -d -s ui '$PY -m feacore.ui --host 0.0.0.0 --port 8000'"
echo "               then expose port 8000: Brev console > your instance > HTTP Ports > add port 8000,"
echo "               or from your laptop: brev port-forward <instance> --port 8000:8000"
