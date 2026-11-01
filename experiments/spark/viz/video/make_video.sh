#!/bin/bash
# FEA virtual twin -> ~/feagent-viz/video/wingbox_twin.mp4, rendered by Omniverse RTX (Isaac Sim 6.2.0, headless,
# Docker) on the GB10, captioned and encoded by ffmpeg (H.264, yuv420p, 30 fps).
#
#   ~/feagent-viz/video/make_video.sh                 preflight, render all frames, assemble the MP4
#   DRY_RUN=1  ~/feagent-viz/video/make_video.sh      print every command (and the preflight verdict); run nothing
#   PREVIEW=1  ...                                    one frame per shot + a pt/rt timing probe -> frames_preview/
#   SKIP_RENDER=1 ...                                 assemble only, from existing frames/
#   RENDERER=auto|pt|rt  (default auto: path tracing if <= PT_BUDGET_S s/frame on the first frames, else RTX Real-Time)
#   STAGES=/abs/a.usdc,/abs/b.usdc                    gallery hook: a 3 s turntable per extra stage, appended
#   RESUME=1  re-use PNGs already written;  RENDER_TIMEOUT=7200 (s);  MEM_LIMIT=64g (container cap)
#   ALLOW_OTHER_CONTAINERS=1  start even if a non-Isaac container is running
#   ALLOW_CONTAINER=ridgeline-isaac  tolerate exactly this running container and its GPU processes (user-approved sharing)
#   GOLD_WAIT_S=900  wait (poll 30 s) for gold_case.py / run_all.py / gold.py to finish before refusing
#
# Preflight refuses (exit 3) if: an Isaac Sim container runs (e.g. ridgeline-isaac); any other container runs (unless
# ALLOW_OTHER_CONTAINERS=1); a process holds the GPU; MemAvailable < 40 GiB; a gold run is in progress (lock/pid
# files in ~/gold-runs, or a gold_case.py / run_all.py / gold.py process after a 15 min wait); < 10 GB free disk;
# image or ffmpeg missing.
# No sudo, nothing persistent: one foreground container `feagent-video` (--rm); caches in ~/.cache/feagent-twin-isaac
# (shared with open.sh). ~/feagent-viz is mounted read-only; only the frames dir is mounted read-write.
set -euo pipefail
V=$(cd "$(dirname "$0")" && pwd)          # ~/feagent-viz/video
D=$(dirname "$V")                          # ~/feagent-viz
IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
NAME=feagent-video
FFMPEG=${FFMPEG:-$(command -v ffmpeg || echo ~/micromamba/envs/feagent-ffmpeg/bin/ffmpeg)}
PLANPY=${PLANPY:-~/micromamba/envs/feagent/bin/python}   # usd-core, CPU only: shot plan for DRY_RUN
CACHE=~/.cache/feagent-twin-isaac
GOLD=~/gold-runs
MIN_MEM_GIB=${MIN_MEM_GIB:-40}
RENDERER=${RENDERER:-auto}
PT_BUDGET_S=${PT_BUDGET_S:-4.0}
RENDER_TIMEOUT=${RENDER_TIMEOUT:-7200}
MEM_LIMIT=${MEM_LIMIT:-64g}
if [ -n "${PREVIEW:-}" ]; then FR=$V/frames_preview; else FR=$V/frames; fi
OUT=$V/wingbox_twin.mp4
# RAMPS=1: the load-ramp video instead (stages ~/feagent-viz/ramps/<set>/<set>.usdc from prep_ramps.sh, in RAMP_SETS order)
RD=$D/ramps
if [ -n "${RAMPS:-}" ]; then
  FR=$RD/frames${PREVIEW:+_preview}; OUT=$RD/load_ramps.mp4
  RENDERER=${RENDERER_RAMPS:-rt}
fi
LOGS=$V/logs
TS=$(date +%Y%m%d-%H%M%S)
DRY=${DRY_RUN:-}

say() { echo "make_video: $*" >&2; }
show() { printf '+'; printf ' %q' "$@"; echo; }
run() { show "$@"; [ -n "$DRY" ] || "$@"; }

# ------------------------------------------------------------------ preflight
# ALLOW_CONTAINER=<exact name> tolerates that one running container (and its GPU processes), e.g. ridgeline-isaac
# when the user has chosen to share the GPU. Nothing else is relaxed; the container is never touched.
ALLOW_CONTAINER=${ALLOW_CONTAINER:-}
GOLD_PAT='gold_case\.py|run_all\.py|scripts/gold\.py'
GOLD_WAIT_S=${GOLD_WAIT_S:-900}

wait_for_gold() {   # poll every 30 s, at most GOLD_WAIT_S, until no gold driver process runs
  local waited=0 gp
  while :; do
    gp=$(pgrep -af "$GOLD_PAT" | grep -v -e pgrep -e make_video || true)
    [ -z "$gp" ] && return 0
    if [ "$waited" -ge "$GOLD_WAIT_S" ] || [ -n "$DRY" ]; then
      say "gold run still in progress after ${waited} s:"; echo "$gp" | cut -c1-160 | sed 's/^/      /' >&2; return 1
    fi
    say "waiting for the gold run to finish ($(echo "$gp" | wc -l) processes; ${waited}/${GOLD_WAIT_S} s)"
    sleep 30; waited=$((waited + 30))
  done
}

preflight() {
  local bad=0 c others
  c=$(docker ps --format '{{.Names}} {{.Image}}' | grep -iE 'isaac|feagent-twin|feagent-video' \
      | { if [ -n "$ALLOW_CONTAINER" ]; then grep -vE "^${ALLOW_CONTAINER} " || true; else cat; fi; } || true)
  if [ -n "$ALLOW_CONTAINER" ] && docker ps --format '{{.Names}}' | grep -qx "$ALLOW_CONTAINER"; then
    say "note: ALLOW_CONTAINER=$ALLOW_CONTAINER is running and tolerated (user chose to share the GPU)"
  fi
  if [ -n "$c" ]; then
    say "REFUSE: an Isaac Sim container is running (the GB10 runs one large GPU job at a time):"
    echo "$c" | sed 's/^/      /' >&2; bad=1
  fi
  others=$(docker ps --format '{{.Names}} {{.Image}}' | grep -viE 'isaac|feagent-twin|feagent-video' \
      | { if [ -n "$ALLOW_CONTAINER" ]; then grep -vE "^${ALLOW_CONTAINER} " || true; else cat; fi; } || true)
  if [ -n "$others" ] && [ -z "${ALLOW_OTHER_CONTAINERS:-}" ]; then
    say "REFUSE: other containers are running (AGENTS.md: one large job at a time; ALLOW_OTHER_CONTAINERS=1 to override):"
    echo "$others" | sed 's/^/      /' >&2; bad=1
  fi
  local gpu
  gpu=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
  if [ -n "$ALLOW_CONTAINER" ] && [ -n "$gpu" ]; then   # drop the GPU processes that belong to the tolerated container
    local apids
    apids=$(docker top "$ALLOW_CONTAINER" -eo pid 2>/dev/null | tail -n +2 | tr -s ' \n' '|' | sed 's/^|//;s/|$//')
    [ -n "$apids" ] && gpu=$(echo "$gpu" | grep -vE "^(${apids})," || true)
  fi
  if [ -n "$gpu" ]; then
    say "REFUSE: processes are using the GPU:"; echo "$gpu" | sed 's/^/      /' >&2; bad=1
  fi
  local mem_kb mem_gib
  mem_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  mem_gib=$((mem_kb / 1048576))
  if [ "$mem_gib" -lt "$MIN_MEM_GIB" ]; then
    say "REFUSE: MemAvailable is ${mem_gib} GiB < ${MIN_MEM_GIB} GiB (shared CPU/GPU memory; an OOM GB10 can freeze)"; bad=1
  else
    say "ok: MemAvailable ${mem_gib} GiB >= ${MIN_MEM_GIB} GiB"
  fi
  # gold runs: lock / pid / in-progress markers in ~/gold-runs (live pid only), or a gold process
  if [ -d "$GOLD" ]; then
    local f pid
    while IFS= read -r f; do
      case "$f" in
        *.pid) pid=$(tr -dc '0-9' < "$f" | head -c 12); if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                 say "REFUSE: gold run in progress: $f (pid $pid alive)"; bad=1; fi ;;
        *) say "REFUSE: gold run in progress: lock/marker $f"; bad=1 ;;
      esac
    done < <(find "$GOLD" -maxdepth 3 \( -name '*.lock' -o -name '*.pid' -o -name 'LOCK' -o -name 'RUNNING' \
                                        -o -name '*.inprogress' -o -name '.running' \) 2>/dev/null)
  fi
  if ! wait_for_gold; then say "REFUSE: a gold run is in progress (waited up to ${GOLD_WAIT_S} s)"; bad=1; fi
  local free_gb
  free_gb=$(df -BG --output=avail "$V" | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt 10 ]; then say "REFUSE: only ${free_gb} GB free on $V (frames need ~3 GB)"; bad=1; fi
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then say "REFUSE: image $IMAGE not present (do not pull)"; bad=1; fi
  if [ ! -x "$FFMPEG" ]; then say "REFUSE: no ffmpeg ($FFMPEG); see README (micromamba env feagent-ffmpeg)"; bad=1; fi
  [ -f "$D/twin.usdc" ] || { say "REFUSE: $D/twin.usdc missing"; bad=1; }
  if [ "$bad" = 0 ]; then say "preflight: OK"; fi
  return $bad
}

if [ -z "${SKIP_RENDER:-}" ]; then
  if ! preflight; then
    if [ -n "$DRY" ]; then say "preflight would REFUSE now; DRY_RUN continues printing the commands"
    else say "not starting (nothing was run). Wait for the jobs above to finish, then run again."; exit 3; fi
  fi
fi

# one make_video at a time
if [ -z "$DRY" ]; then
  mkdir -p "$LOGS"
  exec 9>"$V/.make_video.lock"
  flock -n 9 || { say "another make_video.sh is running"; exit 3; }
fi

# ------------------------------------------------------------------ shot plan (CPU only, usd-core; no Isaac Sim)
STAGE_ARGS=() ; STAGE_MOUNTS=() ; IN_STAGES=""
if [ -n "${STAGES:-}" ]; then
  k=0
  IFS=',' read -r -a SL <<< "$STAGES"
  for s in "${SL[@]}"; do
    [ -f "$s" ] || { say "STAGES: $s not found"; exit 2; }
    STAGE_MOUNTS+=(-v "$(cd "$(dirname "$s")" && pwd)":/stages/$k:ro)
    IN_STAGES+="${IN_STAGES:+,}/stages/$k/$(basename "$s")"
    k=$((k + 1))
  done
  STAGE_ARGS=(--stages "$IN_STAGES")
fi
if [ -n "$DRY" ]; then
  say "plan (CPU, read-only stage read with usd-core):"
  show "$PLANPY" -W ignore "$V/render_twin.py" --plan-only --usd "$D/twin.usdc" --out "$V/plan" ${STAGES:+--stages "$STAGES"}
  "$PLANPY" -W ignore "$V/render_twin.py" --plan-only --usd "$D/twin.usdc" --out "$V/plan" ${STAGES:+--stages "$STAGES"}
fi

# ------------------------------------------------------------------ render (Isaac Sim 6.2 headless, Docker)
if [ -z "${SKIP_RENDER:-}" ]; then
  if [ -z "$DRY" ]; then
    mkdir -p $CACHE/{main,computecache,kit,logs,config,data,pkg,hub} "$FR"
    # container runs as uid 1234; no chown/sudo. Only our top dirs: what uid 1234 wrote stays its own.
    chmod a+rwX $CACHE $CACHE/* "$FR" 2>/dev/null || true
    if [ -z "${RESUME:-}" ] && [ -d "$FR" ]; then find "$FR" -mindepth 1 -delete; fi
  else
    show mkdir -p $CACHE/{main,computecache,kit,logs,config,data,pkg,hub} "$FR"
    show chmod a+rwX $CACHE "$CACHE/*" "$FR"
    [ -n "${RESUME:-}" ] || show find "$FR" -mindepth 1 -delete
  fi
  RARGS=(--usd /twin/twin.usdc --summary /twin/summary.json --out /out --renderer "$RENDERER" --pt-budget-s "$PT_BUDGET_S"
         "${STAGE_ARGS[@]}")
  if [ -n "${RAMPS:-}" ]; then
    RL=""; for s in ${RAMP_SETS:?RAMP_SETS="set1 set2 ..."}; do RL+="${RL:+,}/ramps_in/$s/$s.usdc"; done
    RARGS=(--usd /twin/twin.usdc --out /out --renderer "$RENDERER" --ramps "$RL")
    STAGE_MOUNTS+=(-v "$RD":/ramps_in:ro)
  fi
  [ -n "${PREVIEW:-}" ] && RARGS+=(--preview)
  [ -n "${RESUME:-}" ] && RARGS+=(--resume)
  [ -n "${RENDER_EXTRA:-}" ] && read -r -a EXTRA <<< "$RENDER_EXTRA" && RARGS+=("${EXTRA[@]}")
  CMD=(timeout --signal=TERM --kill-after=60 "$RENDER_TIMEOUT"
    docker run --name "$NAME" --rm --gpus all --entrypoint bash --memory "$MEM_LIMIT"
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES
    -v "$D":/twin:ro
    -v "$FR":/out:rw
    "${STAGE_MOUNTS[@]}"
    -v $CACHE/main:/isaac-sim/.cache:rw
    -v $CACHE/computecache:/isaac-sim/.nv/ComputeCache:rw
    -v $CACHE/kit:/isaac-sim/kit/cache:rw
    -v $CACHE/logs:/isaac-sim/.nvidia-omniverse/logs:rw
    -v $CACHE/config:/isaac-sim/.nvidia-omniverse/config:rw
    -v $CACHE/data:/isaac-sim/.local/share/ov/data:rw
    -v $CACHE/pkg:/isaac-sim/.local/share/ov/pkg:rw
    -v $CACHE/hub:/var/cache/hub:rw
    -u 1234:1234
    "$IMAGE" -c 'export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1; exec ./python.sh /twin/video/render_twin.py "$@"'
    bash "${RARGS[@]}")
  say "render: Isaac Sim 6.2 headless (renderer $RENDERER), frames -> $FR; first start compiles shaders (~3 min)"
  if [ -n "$DRY" ]; then show "${CMD[@]}"; echo "    (stdout/stderr tee'd to $LOGS/render-$TS.log)"
  else
    trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT INT TERM
    show "${CMD[@]}"
    set +e
    "${CMD[@]}" 2>&1 | tee "$LOGS/render-$TS.log"
    rc=${PIPESTATUS[0]}
    set -e
    trap - EXIT INT TERM
    [ "$rc" = 0 ] || { say "render failed (exit $rc); log: $LOGS/render-$TS.log"; exit "$rc"; }
  fi
fi

# ------------------------------------------------------------------ assemble (host ffmpeg)
if [ -n "${RAMPS:-}" ] && [ -z "${PREVIEW:-}" ]; then
  if [ -n "$DRY" ]; then python3 "$RD/ramp_assemble.py" --frames "$FR" --out "$OUT" --stills "$RD/stills" --ffmpeg "$FFMPEG" --dry-run
  else python3 "$RD/ramp_assemble.py" --frames "$FR" --out "$OUT" --stills "$RD/stills" --ffmpeg "$FFMPEG" 2>&1 | tee "$LOGS/ramps-$TS.log"; fi
  say "done: $OUT"; exit 0
fi
if [ -n "${PREVIEW:-}" ]; then
  run "$FFMPEG" -hide_banner -v error -y -pattern_type glob -i "$FR/*/frame_*.png" -vf "scale=384:-1,tile=5x4" \
      -frames:v 1 "$V/${RAMPS:+ramps_}preview_sheet.png"
  say "preview: frames in $FR, contact sheet $V/preview_sheet.png, timing in $FR/manifest_preview.json"
  exit 0
fi
AARGS=(--frames "$FR" --summary "$D/summary.json" --out "$OUT" --work "$V/work" --ffmpeg "$FFMPEG")
if [ -n "$DRY" ]; then
  say "assemble (commands only; the captions below are written to $V/work at run time):"
  python3 "$V/assemble.py" "${AARGS[@]}" --dry-run
  say "DRY_RUN: nothing was started."
else
  python3 "$V/assemble.py" "${AARGS[@]}" 2>&1 | tee "$LOGS/assemble-$TS.log"
  say "done: $OUT"
fi
