#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" tools/download_external_assets.py --skip-roboflow --skip-drive
"$PYTHON_BIN" tools/build_detector_dataset.py --clean
"$PYTHON_BIN" tools/validate_yolo_labels.py artifacts/datasets/hurgor_detector/labels --report logs/hurgor-detector-labels.json
"$PYTHON_BIN" tools/deduplicate_images.py artifacts/datasets/hurgor_detector/images --report logs/hurgor-detector-duplicates.json
"$PYTHON_BIN" tools/resplit_yolo_dataset.py artifacts/datasets/hurgor_detector artifacts/datasets/hurgor_detector_grouped --clean
"$PYTHON_BIN" tools/validate_yolo_labels.py artifacts/datasets/hurgor_detector_grouped/labels --report logs/hurgor-detector-grouped-labels.json
"$PYTHON_BIN" tools/deduplicate_images.py artifacts/datasets/hurgor_detector_grouped/images --report logs/hurgor-detector-grouped-duplicates.json
