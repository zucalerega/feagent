#!/usr/bin/env bash
# CalculiX smoke check of a nonlinear netgen mesh: linear-elastic static step, fixed bottom,
# small top displacement along AXIS (1=x, 2=y, 3=z). Usage: ccx_smoke.sh OUT_DIR AXIS_NUMBER
set -euo pipefail
d=$1; ax=$2
cat > "$d/smoke.inp" <<EOF
*INCLUDE, INPUT=mesh_ccx.inp
*MATERIAL, NAME=RESIN
*ELASTIC
1.8798E9, 0.35
*SOLID SECTION, ELSET=domain, MATERIAL=RESIN
*STEP
*STATIC
*BOUNDARY
bottom, 1, 3, 0.
top, $ax, $ax, -1.E-5
*NODE FILE
U
*EL FILE
S
*END STEP
EOF
cd "$d" && OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" ~/micromamba-local/envs/ccx/bin/ccx smoke 2>&1 | tail -15
