from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector holdout değerlendirmesi")
    parser.add_argument("weights", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/metrics.json"))
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        raise SystemExit("weights and data YAML must exist")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("install export dependencies: pip install -e '.[export]'") from exc
    output_parent = args.output.parent.resolve()
    metrics = YOLO(str(args.weights)).val(
        data=str(args.data),
        imgsz=args.image_size,
        device=args.device,
        split=args.split,
        batch=args.batch,
        workers=args.workers,
        plots=args.plots,
        project=str(output_parent),
        name=f"{args.output.stem}-{args.split}",
        exist_ok=True,
    )
    results = getattr(metrics, "results_dict", {})
    names = dict(getattr(metrics, "names", {}))
    # ``class_result`` accepts the compact result-array index, not the original
    # class id.  They only differ when a holdout contains no instances for one
    # or more configured classes (for example HIT-UAV has no UAP/UAI labels).
    # Treat absent classes as unmeasured instead of crashing or reporting a
    # misleading zero score.
    measured_indices = {
        int(class_id): result_index
        for result_index, class_id in enumerate(metrics.ap_class_index)
    }
    per_class = {}
    for class_id, name in sorted(names.items()):
        result_index = measured_indices.get(int(class_id))
        class_report = {
            "class_id": int(class_id),
            "measured": result_index is not None,
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
        }
        if result_index is not None:
            precision, recall, map50, map50_95 = metrics.class_result(result_index)
            class_report.update(
                {
                    "precision": float(precision),
                    "recall": float(recall),
                    "map50": float(map50),
                    "map50_95": float(map50_95),
                }
            )
        per_class[str(name)] = class_report
    report = {
        "split": args.split,
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "image_size": args.image_size,
        "aggregate": results,
        "per_class": per_class,
        "speed_ms": getattr(metrics, "speed", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
