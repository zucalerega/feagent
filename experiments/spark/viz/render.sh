#!/bin/bash
# GPU (EGL on the GB10) offscreen preview of a twin stage; falls back to CPU (llvmpipe) if EGL fails.
#   ~/feagent-viz/render.sh [twin.usdc] [outdir] [gpu|cpu]
set -u
D=$(cd "$(dirname "$0")" && pwd)
USD=${1:-$D/twin.usdc}; OUT=${2:-$D/renders}; MODE=${3:-gpu}
PY=~/micromamba/envs/feagent/bin/python
run() { systemd-run --user --scope -q -p MemoryMax=8G env "$@" $PY -W ignore "$D/render_twin.py" "$USD" "$OUT"; }
if [ "$MODE" = gpu ]; then
  run -u DISPLAY -u WAYLAND_DISPLAY __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
      VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow && exit 0
  echo "render.sh: EGL/GPU render failed; falling back to CPU (llvmpipe)" >&2
fi
run -u DISPLAY -u WAYLAND_DISPLAY __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow \
    VTK_DEFAULT_EGL_DEVICE_INDEX=${CPU_EGL_DEVICE:-1}  # device 0 is the GB10; 1 is Mesa llvmpipe here
