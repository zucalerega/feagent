#!/bin/bash
# Open the FEA virtual twin (~/feagent-viz/twin.usdc) in Isaac Sim 6.2.0 (Omniverse RTX) on the Spark's display.
# Runs ONE foreground container, `feagent-twin` (--rm): close the Isaac Sim window or press Ctrl-C to stop it.
#   ~/feagent-viz/open.sh [path/to/twin.usdc inside ~/feagent-viz]      (default twin.usdc)
#   DRY_RUN=1 ~/feagent-viz/open.sh   prints the docker command without running it
# No sudo, no services, nothing left behind except Isaac Sim's caches in ~/.cache/feagent-twin-isaac.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
IMAGE=nvcr.io/nvidia/isaac-sim:6.2.0
NAME=feagent-twin
USD_REL=${1:-twin.usdc}
[ -f "$D/$USD_REL" ] || { echo "open.sh: $D/$USD_REL not found (run export.sh first)" >&2; exit 2; }

# 1. One large GPU job at a time: refuse if any Isaac Sim container (e.g. ridgeline-isaac) is running.
busy=$(docker ps --format '{{.Names}} {{.Image}}' | grep -iE 'isaac|feagent-twin' || true)
if [ -n "$busy" ]; then
  echo "open.sh: NOT starting. Another Isaac Sim / GPU container is running:" >&2
  echo "$busy" | sed 's/^/    /' >&2
  echo "  The GB10 runs one large GPU job at a time (shared 128 GB memory; an out-of-memory GB10 can freeze)." >&2
  echo "  Wait until it has finished (docker ps), then run open.sh again. Do not stop someone else's container." >&2
  exit 3
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "open.sh: image $IMAGE not present (26.8 GB; do not pull without approval)" >&2; exit 4; }

# 2. Display: the Spark's desktop is Xorg :1 (/tmp/.X11-unix/X1, auth /run/user/$UID/gdm/Xauthority).
export DISPLAY=${DISPLAY:-:1}
XA_SRC=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
CACHE=~/.cache/feagent-twin-isaac
mkdir -p $CACHE/{main,computecache,kit,logs,config,data,pkg,hub}
chmod -R a+rwX $CACHE                # the container runs as uid 1234 (isaac-sim); no chown/sudo needed
XA=$(mktemp "$CACHE/xauth.XXXXXX")   # short-lived X cookie for the container, wildcard host family
trap 'rm -f "$XA"' EXIT
if [ -r "$XA_SRC" ] && xauth -f "$XA_SRC" nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$XA" nmerge - 2>/dev/null \
   && [ -s "$XA" ]; then
  chmod 644 "$XA"
else
  echo "open.sh: could not copy the X cookie from $XA_SRC. If the window does not appear, allow local clients once" >&2
  echo "  in a terminal of the desktop session:   xhost +local:     (and afterwards:  xhost -local:)" >&2
fi

cat <<EOF
Opening $D/$USD_REL in Isaac Sim 6.2.0 (container $NAME, display $DISPLAY). First start compiles shaders: several minutes.
  Fields:   Stage panel -> select /World -> Property panel -> Variants -> field = von_mises | u_mag | gate_status | part_colors
            (the legend quad next to the wingbox switches with it).
  Deformed: the timeline plays automatically (Play/Pause at the bottom, or Space). t = 0 undeformed, t = 24 = deformation x$(python3 -c "import json;print('%g' % json.load(open('$D/summary.json'))['deformation_scale'])" 2>/dev/null).
  Camera:   viewport camera menu -> /World/Camera.  Per-part data: select /World/Wingbox/<part>, Raw USD Properties (fea:*).
  Stop:     close the Isaac Sim window or press Ctrl-C here.
  No window / "Authorization required"? run  xhost +local:  in the desktop session, then again: xhost -local:
EOF

CMD=(docker run --name "$NAME" --rm --gpus all --entrypoint bash
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES
  -e DISPLAY="$DISPLAY" -e XAUTHORITY=/isaac-sim/.Xauthority -v "$XA":/isaac-sim/.Xauthority:ro
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro
  -e TWIN_USD="/twin/$USD_REL"
  -v "$D":/twin:ro
  -v $CACHE/main:/isaac-sim/.cache:rw
  -v $CACHE/computecache:/isaac-sim/.nv/ComputeCache:rw
  -v $CACHE/kit:/isaac-sim/kit/cache:rw
  -v $CACHE/logs:/isaac-sim/.nvidia-omniverse/logs:rw
  -v $CACHE/config:/isaac-sim/.nvidia-omniverse/config:rw
  -v $CACHE/data:/isaac-sim/.local/share/ov/data:rw
  -v $CACHE/pkg:/isaac-sim/.local/share/ov/pkg:rw
  -v $CACHE/hub:/var/cache/hub:rw
  -u 1234:1234
  "$IMAGE" -c 'export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
    if [ -x ./isaac-sim.sh ]; then exec ./isaac-sim.sh --exec "/twin/open_in_kit.py"; fi
    exec ./runapp.sh --exec "/twin/open_in_kit.py"')
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
