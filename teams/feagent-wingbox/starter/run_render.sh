#!/bin/bash
# usage: run_render.sh MODE stage png [reps]   MODE = x11 | nvidia | cpu
# Spike code from the GPU/visualization study (2026-10-31): measured 5M tris in 11.9 ms on an RTX 4060 via EGL vs 673 ms on llvmpipe.
PY="${PYTHON:-python}"   # a venv with usd-core + pyvista: uv pip install usd-core
D=$(dirname "$0")
M=$1; shift
unset WAYLAND_DISPLAY
case $M in
  x11) exec $PY -W ignore $D/render_usd.py "$@";;
  nvidia) unset DISPLAY
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
    export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
    exec $PY -W ignore $D/render_usd.py "$@";;
  cpu) export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe; exec $PY -W ignore $D/render_usd.py "$@";;
  cpu_egl) unset DISPLAY
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless VTK_DEFAULT_EGL_DEVICE_INDEX=${CPU_DEV:-0}
    export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
    exec $PY -W ignore $D/render_usd.py "$@";;
esac
