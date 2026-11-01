#!/bin/bash
# Omniverse RTX crush-to-snap video of one crush.py run, on the DGX Spark (<spark-host>, aarch64, GB10):
# build the USD time series (host, CPU) -> render the frames with Isaac Sim 6.2 headless (Docker, RTX Real-Time)
# -> caption + encode (host).
#
#   experiments/omniverse/make_crush_video.sh RUN_DIR NAME "mesh system label"
#   ONLY=0,240,-1 ...     render just these frames (a look first); the PNGs land in $WORK/NAME/frames
#   SKIP_BUILD=1 / SKIP_RENDER=1 / RESUME=1   reuse the stage / the frames / the PNGs already written
#   RENDER_EXTRA="--rt-subframes 12"          passed to render_crush_rtx.py
#   ALLOW_CONTAINER=ridgeline-isaac           tolerate exactly this running container (it is never touched)
#
# Env: WORK (default ~/feagent-viz/crush/work/rtx), OUT_DIR (default ~/feagent-viz/videos/mesh_study/crush),
# PY (python with pyvista + usd-core + Pillow + matplotlib; default ~/micromamba/envs/feagent/bin/python),
# FFMPEG (default ~/micromamba/envs/feagent-ffmpeg/bin/ffmpeg), MEM_LIMIT (container cap, default 24g),
# MIN_FREE_GIB (MemAvailable needed before the render starts, default 44: the 24 GiB cap + 20 GiB headroom).
# Rules of the box: no sudo, nothing persistent (one --rm container `feagent-crush`), every job memory-capped,
# one large GPU job at a time, the image is used as present (never pulled).
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
RUN=${1:?RUN_DIR}; NAME=${2:?NAME}; LABEL=${3:-}
WORK=${WORK:-$HOME/feagent-viz/crush/work/rtx}
OUT_DIR=${OUT_DIR:-$HOME/feagent-viz/videos/mesh_study/crush}
PY=${PY:-$HOME/micromamba/envs/feagent/bin/python}
FFMPEG=${FFMPEG:-$HOME/micromamba/envs/feagent-ffmpeg/bin/ffmpeg}
IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
CNAME=feagent-crush
MEM_LIMIT=${MEM_LIMIT:-24g}
MIN_FREE_GIB=${MIN_FREE_GIB:-44}
CACHE=$HOME/.cache/feagent-twin-isaac
ALLOW_CONTAINER=${ALLOW_CONTAINER:-}
W=$WORK/$NAME
FR=$W/frames
mkdir -p "$W" "$FR" "$OUT_DIR" "$W/logs"
say() { echo "make_crush_video: $*" >&2; }
cap() { CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=24G "$@"; }

if [ -z "${SKIP_BUILD:-}" ]; then
  say "build USD time series -> $W"
  cap "$PY" "$HERE/build_crush_usd.py" "$RUN" "$W" ${LABEL:+--label "$LABEL"} 2>&1 | grep -v -e WARN -e '^$' | tee "$W/logs/build.log"
fi

if [ -z "${SKIP_RENDER:-}" ]; then
  # preflight
  c=$(docker ps --format '{{.Names}} {{.Image}}' | grep -iE 'isaac|feagent' \
      | { if [ -n "$ALLOW_CONTAINER" ]; then grep -vE "^${ALLOW_CONTAINER} " || true; else cat; fi; } || true)
  [ -z "$c" ] || { say "REFUSE: another Isaac/feagent container is running:"; echo "$c" >&2; exit 3; }
  mem=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) / 1048576 ))
  [ "$mem" -ge "$MIN_FREE_GIB" ] || { say "REFUSE: MemAvailable ${mem} GiB < ${MIN_FREE_GIB} GiB"; exit 3; }
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { say "REFUSE: $IMAGE not present (not pulling)"; exit 3; }
  exec 9>"$WORK/.render.lock"; flock -n 9 || { say "another crush render is running"; exit 3; }
  mkdir -p $CACHE/{main,computecache,kit,logs,config,data,pkg,hub}
  chmod a+rwX $CACHE $CACHE/* "$FR" 2>/dev/null || true
  [ -n "${RESUME:-}${ONLY:-}" ] || find "$FR" -mindepth 1 -delete
  RARGS=(--usd /work/crush.usdc --meta /work/crush_meta.json --out /out)
  [ -n "${ONLY:-}" ] && RARGS+=(--only "$ONLY")
  [ -n "${RESUME:-}" ] && RARGS+=(--resume)
  [ -n "${RENDER_EXTRA:-}" ] && read -r -a EX <<< "$RENDER_EXTRA" && RARGS+=("${EX[@]}")
  say "render: Isaac Sim 6.2 headless, MemAvailable ${mem} GiB, container cap $MEM_LIMIT"
  trap 'docker rm -f "$CNAME" >/dev/null 2>&1 || true' EXIT INT TERM
  set +e
  timeout --signal=TERM --kill-after=60 7200 docker run --name "$CNAME" --rm --gpus all --entrypoint bash \
    --memory "$MEM_LIMIT" --memory-swap "$MEM_LIMIT" \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -v "$HERE":/crush:ro -v "$W":/work:ro -v "$FR":/out:rw \
    -v $CACHE/main:/isaac-sim/.cache:rw -v $CACHE/computecache:/isaac-sim/.nv/ComputeCache:rw \
    -v $CACHE/kit:/isaac-sim/kit/cache:rw -v $CACHE/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
    -v $CACHE/config:/isaac-sim/.nvidia-omniverse/config:rw -v $CACHE/data:/isaac-sim/.local/share/ov/data:rw \
    -v $CACHE/pkg:/isaac-sim/.local/share/ov/pkg:rw -v $CACHE/hub:/var/cache/hub:rw \
    -u 1234:1234 "$IMAGE" \
    -c 'export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1; exec ./python.sh /crush/render_crush_rtx.py "$@"' \
    bash "${RARGS[@]}" > "$W/logs/render.log" 2>&1
  rc=$?
  set -e
  trap - EXIT INT TERM
  grep '\[render_crush_rtx\]' "$W/logs/render.log" | tail -4 >&2 || true
  [ "$rc" = 0 ] || { say "render failed ($rc): $W/logs/render.log"; exit "$rc"; }
  [ -n "${ONLY:-}" ] && { say "frames: $FR"; exit 0; }
fi

cap "$PY" "$HERE/assemble_crush.py" --frames "$FR" --meta "$W/crush_meta.json" --out "$OUT_DIR/${NAME}_crush_rtx.mp4" \
  --ffmpeg "$FFMPEG" 2>&1 | tee "$W/logs/assemble.log"
say "done: $OUT_DIR/${NAME}_crush_rtx.mp4"
