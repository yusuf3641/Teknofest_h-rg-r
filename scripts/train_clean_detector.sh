#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA="${HURGOR_TRAIN_DATA:-artifacts/datasets/hurgor_detector_grouped/data.yaml}"
MODEL="${HURGOR_TRAIN_MODEL:-artifacts/pretrained/yolo26n.pt}"
NAME="${HURGOR_TRAIN_NAME:-hurgor-yolo26n-clean-v2}"
EPOCHS="${HURGOR_TRAIN_EPOCHS:-50}"
IMAGE_SIZE="${HURGOR_TRAIN_IMAGE_SIZE:-640}"
BATCH="${HURGOR_TRAIN_BATCH:-8}"
DEVICE="${HURGOR_TRAIN_DEVICE:-mps}"
WORKERS="${HURGOR_TRAIN_WORKERS:-0}"
PATIENCE="${HURGOR_TRAIN_PATIENCE:-12}"

"$ROOT/.venv/bin/python" tools/train_detector.py \
  --data "$DATA" \
  --model "$MODEL" \
  --project artifacts/training \
  --name "$NAME" \
  --epochs "$EPOCHS" \
  --image-size "$IMAGE_SIZE" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --patience "$PATIENCE" \
  --no-cache \
  --plots

"$ROOT/.venv/bin/python" tools/export_detector.py \
  "artifacts/training/$NAME/weights/best.pt" \
  --target onnx \
  --image-size "$IMAGE_SIZE" \
  --device cpu \
  --manifest "artifacts/training/$NAME/weights/best.json"

"$ROOT/.venv/bin/python" tools/evaluate_detector.py \
  "artifacts/training/$NAME/weights/best.pt" \
  --data "$DATA" \
  --split test \
  --image-size "$IMAGE_SIZE" \
  --output "artifacts/training/$NAME/test_metrics.json"
