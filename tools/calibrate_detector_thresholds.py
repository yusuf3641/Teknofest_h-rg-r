from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from hurgor.detector_calibration import EXPECTED_DETECTOR_CLASSES, select_operating_point

LOGGER = logging.getLogger("hurgor.calibration")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation eğrilerinden model-bağlı sınıf confidence profili üretir"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--runtime-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-name", choices=EXPECTED_DETECTOR_CLASSES, default="insan")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--base-confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("artifacts/calibration"))
    args = parser.parse_args()

    import numpy as np
    from ultralytics import YOLO

    weights = args.weights.expanduser().resolve()
    runtime_model = args.runtime_model.expanduser().resolve()
    data = args.data.expanduser().resolve()
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        imgsz=args.image_size,
        device=args.device,
        split="val",
        batch=args.batch,
        workers=args.workers,
        plots=False,
        verbose=False,
        conf=0.001,
        project=str(args.project),
        name=f"{weights.stem}-{args.class_name}-threshold",
        exist_ok=True,
    )
    class_id = EXPECTED_DETECTOR_CLASSES.index(args.class_name)
    curve_class_ids = list(map(int, np.asarray(metrics.box.ap_class_index).tolist()))
    if class_id not in curve_class_ids:
        raise RuntimeError(f"validation curve does not contain class {args.class_name}")
    row = curve_class_ids.index(class_id)
    thresholds = np.asarray(metrics.box.px, dtype=float).tolist()
    precision = np.asarray(metrics.box.p_curve[row], dtype=float).tolist()
    recall = np.asarray(metrics.box.r_curve[row], dtype=float).tolist()
    selected = select_operating_point(
        thresholds,
        precision,
        recall,
        beta=args.beta,
    )
    baseline_index = min(
        range(len(thresholds)),
        key=lambda index: abs(thresholds[index] - args.base_confidence),
    )
    baseline = select_operating_point(
        [thresholds[baseline_index]],
        [precision[baseline_index]],
        [recall[baseline_index]],
        beta=args.beta,
    )
    selected_threshold = round(float(selected["threshold"]), 3)
    class_thresholds = {
        class_name: args.base_confidence for class_name in EXPECTED_DETECTOR_CLASSES
    }
    class_thresholds[args.class_name] = selected_threshold
    payload = {
        "schema_version": 1,
        "runtime_model_sha256": _sha256(runtime_model),
        "calibration_checkpoint_sha256": _sha256(weights),
        "classes": list(EXPECTED_DETECTOR_CLASSES),
        "thresholds": class_thresholds,
        "calibration": {
            "dataset": str(data),
            "split": "val",
            "image_size": args.image_size,
            "class_name": args.class_name,
            "class_id": class_id,
            "objective": f"f_beta_{args.beta:g}",
            "selected": selected,
            "runtime_threshold": selected_threshold,
            "baseline": {
                **baseline,
                "threshold": float(thresholds[baseline_index]),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "threshold_profile_written output=%s class=%s threshold=%.3f model_sha256=%s",
        args.output,
        args.class_name,
        selected_threshold,
        payload["runtime_model_sha256"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
