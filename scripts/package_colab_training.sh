#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${HURGOR_COLAB_DATA_DIR:-artifacts/datasets/hurgor_detector_grouped}"
PRETRAINED="${HURGOR_COLAB_MODEL:-artifacts/pretrained/yolo26n.pt}"
OUT_DIR="${HURGOR_COLAB_OUT_DIR:-artifacts/colab}"
OUT_ZIP="$OUT_DIR/hurgor_colab_training.zip"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Dataset not found: $DATA_DIR" >&2
  exit 1
fi

if [[ ! -f "$PRETRAINED" ]]; then
  echo "Pretrained checkpoint not found: $PRETRAINED" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
rm -f "$OUT_ZIP"

zip -qr "$OUT_ZIP" \
  "$DATA_DIR" \
  "$PRETRAINED" \
  tools/train_detector.py \
  tools/export_detector.py \
  tools/evaluate_detector.py \
  src/hurgor/export_models.py \
  pyproject.toml \
  README.md

du -sh "$OUT_ZIP"
