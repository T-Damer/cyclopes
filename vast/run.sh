#!/usr/bin/env bash
set -euo pipefail

[[ -f /venv/main/bin/activate ]] && source /venv/main/bin/activate

: "${HF_READ_ONLY_TOKEN:?HF_READ_ONLY_TOKEN is required}"
: "${TRAIN_MANIFEST:?TRAIN_MANIFEST is required}"
: "${INITIAL_CHECKPOINT:?INITIAL_CHECKPOINT is required}"

MAX_SECONDS="${MAX_SECONDS:-5400}"
RUN_ROOT="${RUN_ROOT:-runs/vit-experts}"
EVAL_MANIFEST="${EVAL_MANIFEST:-}"
START_SECONDS=$SECONDS
mkdir -p "$RUN_ROOT" reports

run_budgeted() {
  local remaining=$((MAX_SECONDS - (SECONDS - START_SECONDS)))
  if (( remaining <= 60 )); then
    echo "wall-clock budget exhausted" >&2
    return 124
  fi
  timeout --foreground "$remaining" "$@"
}

export HF_TOKEN="$HF_READ_ONLY_TOKEN"
run_budgeted python -m cyclopes.cli train \
  --architecture vit_multilayer_scalepair \
  --model-revision ac6ee457bea904a373065754107451793b56db00 \
  --initial-checkpoint "$INITIAL_CHECKPOINT" --experts-only \
  --manifest "$TRAIN_MANIFEST" \
  --output "$RUN_ROOT/cyclopes-vit.pt" \
  --report "$RUN_ROOT/train.json" \
  --device cuda --batch-size 64 --workers 16 \
  --epochs 3 --max-steps 800 \
  --freeze-steps 1000000 --unfreeze-last-blocks 0 \
  --backbone-lr 0 --head-lr 0.0002 --consistency-weight 0.05

run_budgeted python -m cyclopes.cli calibrate \
  --manifest "$TRAIN_MANIFEST" --split calibration \
  --checkpoint "$RUN_ROOT/cyclopes-vit.pt" \
  --output "$RUN_ROOT/calibration.json" --report "$RUN_ROOT/calibrate-report.json" \
  --device cuda --batch-size 96 --workers 16

run_budgeted python -m cyclopes.cli evaluate \
  --manifest "$TRAIN_MANIFEST" --split validation --paired-views \
  --checkpoint "$RUN_ROOT/cyclopes-vit.pt" --calibration "$RUN_ROOT/calibration.json" \
  --report "$RUN_ROOT/validation.json" --device cuda --batch-size 96 --workers 16

if [[ -n "$EVAL_MANIFEST" ]]; then
  run_budgeted python -m cyclopes.cli evaluate \
    --manifest "$EVAL_MANIFEST" --split test --paired-views \
    --checkpoint "$RUN_ROOT/cyclopes-vit.pt" --calibration "$RUN_ROOT/calibration.json" \
    --report "$RUN_ROOT/evaluation.json" --device cuda --batch-size 96 --workers 16
fi

run_budgeted python -m cyclopes.cli export \
  --checkpoint "$RUN_ROOT/cyclopes-vit.pt" --calibration "$RUN_ROOT/calibration.json" \
  --output "$RUN_ROOT/cyclopes-vit.onnx" --report "$RUN_ROOT/export.json"

find "$RUN_ROOT" -maxdepth 1 -type f ! -name SHA256SUMS -exec sha256sum {} + > "$RUN_ROOT/SHA256SUMS"
tar --exclude="$RUN_ROOT/checkpoints" -czf "$RUN_ROOT.tar.gz" "$RUN_ROOT"
echo "completed in $((SECONDS - START_SECONDS)) seconds: $RUN_ROOT.tar.gz"
