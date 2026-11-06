#!/bin/bash
# Gold-parts gallery stages for the video: re-solve each pilot set's existing case with feacore (cuDSS) writing a .vtu,
# convert SI (m, Pa) -> mm / MPa, export a single-part twin stage.  ~/gold-catalog is only read.
#   ~/feagent-viz/gallery/prep_gallery.sh [set ...]      -> ~/feagent-viz/gallery/<set>/<set>.usdc
set -euo pipefail
G=$(cd "$(dirname "$0")" && pwd)
PY=~/micromamba/envs/feagent/bin/python
PP=~/feagent-mh/meshing-harness/src:~/feagent-mh
SETS=("$@"); [ ${#SETS[@]} -gt 0 ] || SETS=(abc_00000627_bc4 abc_00000974_bc4 abc_00000330_bc1 abc_00000264_bc9)
for s in "${SETS[@]}"; do
  part=${s%_bc*}; src=~/gold-catalog/pilot/$part
  d=$G/$s; mkdir -p "$d"
  cp "$src/out/mesh.msh" "$d/mesh.msh"
  cp "$src/$s/result.json" "$d/pilot_result.json"
  cp "$src/$s/mapping.json" "$d/mapping.json" 2>/dev/null || true
  python3 - "$src/$s/case.json" "$d/case.json" "$d/mesh.msh" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["mesh"] = sys.argv[3]; json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
  echo "[gallery] $s: solve (cuDSS, 40G scope + 12 GiB watchdog)"
  t0=$(date +%s.%N)
  "$G/wd.sh" "solve-$s" "$d/solve.log" PYTHONPATH="$PP" -- -m feacore.cli "$d/case.json" --out "$d/r.json" --vtu "$d/r.vtu" --linear-solver cudss
  echo "[gallery] $s: solved in $(python3 -c "import time;print(round(time.time()-$t0,1))") s"
  "$G/wd.sh" "export-$s" "$d/export.log" -- -W ignore "$G/to_twin.py" "$d" "$s"
  tail -3 "$d/export.log"
done
