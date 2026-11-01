#!/usr/bin/env bash
# CalculiX smoke check: mesh.inp + tiny linear-elastic static step (fixed bottom, 0.01 mm-ish top displacement
# along AXIS) -> must run without errors. usage: ccx_smoke.sh OUT_DIR AXIS(1|2|3 = x|y|z)
set -euo pipefail
out=$1; dof=$2; w=$(mktemp -d)
cp "$out/mesh.inp" "$w/mesh.inp"
cat > "$w/job.inp" <<DECK
*INCLUDE, INPUT=mesh.inp
*MATERIAL, NAME=RESIN
*ELASTIC
1879.8E6, 0.35
*SOLID SECTION, ELSET=domain, MATERIAL=RESIN
*BOUNDARY
bottom, 1, 3
*BOUNDARY
top, $dof, $dof, -1.0E-6
*STEP
*STATIC
*NODE FILE
U
*EL FILE
S
*END STEP
DECK
cd "$w"; CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 ~/micromamba-local/envs/ccx/bin/ccx job > ccx.log 2>&1 || { tail -20 ccx.log; exit 1; }
grep -iE "error|warning" ccx.log | head -5 || true
tail -3 ccx.log; grep -c . job.dat 2>/dev/null || true; ls
