#!/usr/bin/env bash
# OpenFOAM for feacore's openfoam backend on a machine without it (e.g. a Mac): OpenFOAM v1912
# (Ubuntu 24.04's package) runs in a Docker container, and shims make its applications look local.
#
#   bash scripts/openfoam_docker.sh            # build the image, start the container, write the shims
#   export FEACORE_OPENFOAM_BASHRC=~/.feacore_foam/bashrc TMPDIR=~/.feacore_foam/tmp
#   python -m ace.run_loop --suite cfd ...     # or any feacore flow solve
#   bash scripts/openfoam_docker.sh stop       # remove the container (the image stays)
#
# The container mounts ~/.feacore_foam at the same path, so OpenFOAM's case directories must live
# there: TMPDIR does that (the backend works in a temp dir). Docker Desktop / colima usually share
# the home folder, but not ~/Documents, ~/Desktop or ~/Downloads (macOS privacy), hence the hidden
# folder. Each shim forwards stop-and-write (SIGUSR1: 30 on macOS, 10 on Linux) to the solver.
set -euo pipefail
F="$HOME/.feacore_foam"
NAME=feacore-foam
IMAGE=feacore-openfoam:1912

if [ "${1:-}" = "stop" ]; then
  docker rm -f "$NAME" >/dev/null && echo "removed container $NAME"
  exit 0
fi

mkdir -p "$F/bin" "$F/tmp"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" - <<'EOF'
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends openfoam && rm -rf /var/lib/apt/lists/*
EOF
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -v "$F:$F" "$IMAGE" sleep infinity >/dev/null

cat > "$F/bin/foam-shim" <<'EOF'
#!/bin/bash
# Run the OpenFOAM application named like this link inside the feacore-foam container, in the same
# directory (under ~/.feacore_foam, mounted at the same path), forwarding SIGUSR1 to the solver.
app=$(basename "$0")
pidfile="$PWD/.foam_pid_$$"
args=()
for a in "$@"; do
  case "$a" in stopAtWriteNowSignal=*) args+=("stopAtWriteNowSignal=10") ;; *) args+=("$a") ;; esac
done
docker exec -w "$PWD" feacore-foam bash -c \
  '. /usr/share/openfoam/etc/bashrc >/dev/null 2>&1; echo $$ > "$0"; exec "$@"' "$pidfile" "$app" "${args[@]}" &
child=$!
trap 'docker exec feacore-foam bash -c "kill -USR1 \$(cat \"$pidfile\") 2>/dev/null"' USR1
while kill -0 "$child" 2>/dev/null; do wait "$child"; done
wait "$child"; rc=$?
rm -f "$pidfile"
exit $rc
EOF
chmod +x "$F/bin/foam-shim"
for app in simpleFoam gmshToFoam postProcess checkMesh foamDictionary; do ln -sf foam-shim "$F/bin/$app"; done

cat > "$F/bashrc" <<EOF
# Stand-in OpenFOAM environment for feacore (FEACORE_OPENFOAM_BASHRC): the applications are shims
# that run in the $NAME Docker container (see bin/foam-shim).
export WM_PROJECT=OpenFOAM
export WM_PROJECT_VERSION=v1912
export FOAM_API=1912
export WM_PROJECT_DIR=/usr/share/openfoam
export PATH="$F/bin:\$PATH"
EOF
echo "OpenFOAM v1912 in container $NAME; now:"
echo "  export FEACORE_OPENFOAM_BASHRC=$F/bashrc TMPDIR=$F/tmp"
