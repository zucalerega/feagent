#!/usr/bin/env bash
# The student study end to end: a small open model taught to mesh by a large teacher (wsp/README.md).
#
#   bash scripts/wsp_study.sh                       # Qwen/Qwen3.5-0.8B, every method, CPU is enough
#   WSP_MODEL=Qwen/Qwen3-0.6B OUT=wsp_runs/q3 bash scripts/wsp_study.sh
#   SKIP_WSP=1 bash scripts/wsp_study.sh            # no weight-space editing session (needs an editor)
#
# Needs: the model from huggingface.co (or a local directory as WSP_MODEL), `pip install -e ".[ace,wsp]"`.
# The weight-space programming session needs an editor: ANTHROPIC_API_KEY for the automated Claude Opus
# editor (wsp.editor run), or run `python -m wsp.editor serve ...` and drive it from a Claude Code session.
# Times on a 4-core CPU (fp32, Qwen3.5-0.8B): ~10 min per evaluation, ~1-2 h per LoRA run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
MODEL=${WSP_MODEL:-Qwen/Qwen3.5-0.8B}
OUT=${OUT:-wsp_runs/study}
PASSES=${PASSES:-25}
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

step() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

step "references + automesher baselines (cached in ace/field_refs/)"
$PY -m wsp.refs --workers 3 > "$OUT/refs.log"
step "teacher demonstrations -> wsp/data/demos.jsonl"
$PY -m wsp.demos | tail -3

ev() {  # ev NAME [evaluate args...]
  local name=$1; shift
  [ -f "$OUT/eval_$name/summary.json" ] || $PY -m wsp.evaluate --variant "$name" --model "$MODEL" --out "$OUT/eval_$name" "$@" \
    | $PY -c "import json,sys; s=json.load(sys.stdin); print('$name', round(s['all']['mean_x'],3), 'x;', 'valid', s['all']['valid'])"
}

step "baselines that need no model"
[ -f "$OUT/eval_teacher_zeroshot/summary.json" ] || \
  $PY -m wsp.evaluate --variant teacher_zeroshot --recipes wsp/results/opus_zeroshot_recipes.jsonl --out "$OUT/eval_teacher_zeroshot" > /dev/null
[ -f "$OUT/eval_heuristic/summary.json" ] || { $PY -m wsp.heuristic > "$OUT/heuristic.jsonl" && \
  $PY -m wsp.evaluate --variant heuristic --recipes "$OUT/heuristic.jsonl" --out "$OUT/eval_heuristic" > /dev/null; }

step "base model and in-context playbook (no weight change)"
ev base
ev playbook --playbook wsp/data/playbook.md

step "LoRA distillation of the teacher's verified recipes"
[ -d "$OUT/sft/adapter" ] || $PY -m wsp.sft --data wsp/data/demos.jsonl --model "$MODEL" --out "$OUT/sft" > "$OUT/sft.log"
ev sft --adapter "$OUT/sft/adapter"

step "rejection-sampling fine-tuning (solver as verifier), one round"
[ -f "$OUT/rft/keep.jsonl" ] || $PY -m wsp.rft --model "$MODEL" --adapter "$OUT/sft/adapter" --out "$OUT/rft"
[ -d "$OUT/sft_rft/adapter" ] || $PY -m wsp.sft --data "wsp/data/demos.jsonl,$OUT/rft/keep.jsonl" --model "$MODEL" \
  --out "$OUT/sft_rft" > "$OUT/sft_rft.log"
ev sft_rft --adapter "$OUT/sft_rft/adapter"

step "best of 8 samples, picked by the reference-free ZZ estimate"
ev sft_rft_k8 --adapter "$OUT/sft_rft/adapter" --n 8 --temperature 0.7
[ -f "$OUT/eval_sft_rft_zz/summary.json" ] || $PY -m wsp.evaluate --variant x --select-zz "$OUT/eval_sft_rft_k8" \
  --out "$OUT/eval_sft_rft_zz" > /dev/null

if [ -z "${SKIP_WSP:-}" ] && [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  step "weight-space programming: Claude Opus edits the base weights ($PASSES forward passes)"
  [ -f "$OUT/wsp_base/diff.json" ] || $PY -m wsp.editor run --session "$OUT/wsp_base" --model "$MODEL" --passes "$PASSES"
  ev wsp --edits "$OUT/wsp_base/edits"
  step "weight-space programming on the distilled student"
  [ -f "$OUT/wsp_sft/diff.json" ] || $PY -m wsp.editor run --session "$OUT/wsp_sft" --model "$MODEL" \
    --adapter "$OUT/sft_rft/adapter" --passes "$PASSES"
  ev sft_rft_wsp --adapter "$OUT/sft_rft/adapter" --edits "$OUT/wsp_sft/edits"
fi

step "figures and tables -> $OUT/report"
names=() dirs=()
for v in base playbook wsp sft sft_rft sft_rft_zz sft_rft_wsp heuristic teacher_zeroshot; do
  [ -f "$OUT/eval_$v/summary.json" ] && { names+=("$v"); dirs+=("$OUT/eval_$v"); }
done
$PY -m wsp.report --evals "$(IFS=,; echo "${dirs[*]}")" --names "$(IFS=,; echo "${names[*]}")" --out "$OUT/report" \
  --mesh-methods base,sft_rft,teacher_zeroshot --meshes cooks_membrane,nafems_le1,lbracket_reentrant,notched_bar_t10_r2
echo "done: $OUT/report/results.md"
