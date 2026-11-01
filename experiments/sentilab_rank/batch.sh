#!/usr/bin/env bash
set -u
cd /home/ubuntu/feagent
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
base=evaluation_runs/sentilab_rank
mkdir -p "$base"
run() {
  local label="$1" input="$2" refinement="$3"
  timeout 900 .venv/bin/python experiments/sentilab_package/run.py \
    --input "$input" --case B --refinement "$refinement" --order 2 \
    --solver pardiso --out "$base/$label" > "$base/$label.log" 2>&1
  local result=$?
  echo "$label $result" >> "$base/status.txt"
  return "$result"
}
run raw_r1 experiments/sentilab_package/inputs/case.json 1 || exit 2
.venv/bin/python - <<'PY'
import json
x=json.load(open('evaluation_runs/sentilab_rank/raw_r1/result.json'))
assert abs(x['M_C']-122.34100707914376)<1e-6, x['M_C']
print('Raw FE reference reproduces previous independent FE run')
PY
if [ "$?" -ne 0 ]; then exit 3; fi
for layout in reference near-01 any-01; do
  run "${layout}_r1" "experiments/sentilab_rank/inputs/B_${layout}.json" 1 || true
done
run reference_r2 experiments/sentilab_rank/inputs/B_reference.json 2 &
p1=$!
run near-01_r2 experiments/sentilab_rank/inputs/B_near-01.json 2 &
p2=$!
wait "$p1" || true
wait "$p2" || true
run any-01_r2 experiments/sentilab_rank/inputs/B_any-01.json 2 || true
date -u > "$base/FINISHED"
