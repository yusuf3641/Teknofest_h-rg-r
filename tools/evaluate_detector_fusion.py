from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _dataset(data_yaml: Path, split: str) -> tuple[Path, list[Path], dict[int, str]]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root_value = Path(str(payload.get("path", data_yaml.parent)))
    root = root_value if root_value.is_absolute() else (data_yaml.parent / root_value)
    root = root.expanduser().resolve()
    split_value = payload.get(split)
    if not isinstance(split_value, str):
        raise ValueError(f"dataset split {split!r} must be a directory path")
    image_dir_value = Path(split_value)
    image_dir = (
        image_dir_value if image_dir_value.is_absolute() else root / image_dir_value
    ).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    images = sorted(
        path for path in image_dir.rglob("*") if path.suffix.casefold() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"no images found under {image_dir}")
    raw_names = payload.get("names", {})
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError("dataset names must be a list or mapping")
    return root, images, names


def _label_path(root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(root)
    parts = list(relative.parts)
    try:
        image_index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"image path does not contain an images directory: {image_path}") from exc
    parts[image_index] = "labels"
    return root.joinpath(*parts).with_suffix(".txt")


def _load_targets(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    classes: list[int] = []
    boxes: list[list[float]] = []
    if label_path.is_file():
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"invalid YOLO label at {label_path}:{line_number}")
            raw_class, center_x, center_y, box_width, box_height = map(float, values)
            class_id = int(raw_class)
            if raw_class != class_id:
                raise ValueError(f"non-integer class at {label_path}:{line_number}")
            center_x *= width
            box_width *= width
            center_y *= height
            box_height *= height
            classes.append(class_id)
            boxes.append(
                [
                    center_x - box_width / 2,
                    center_y - box_height / 2,
                    center_x + box_width / 2,
                    center_y + box_height / 2,
                ]
            )
    return np.asarray(classes, dtype=np.int64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _predictions(result: Any) -> np.ndarray:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.column_stack(
        (
            boxes.xyxy.detach().cpu().numpy(),
            boxes.conf.detach().cpu().numpy(),
            boxes.cls.detach().cpu().numpy(),
        )
    ).astype(np.float32, copy=False)


def _match_predictions(
    predictions: np.ndarray,
    target_classes: np.ndarray,
    target_boxes: np.ndarray,
) -> np.ndarray:
    correct = np.zeros((len(predictions), len(IOU_THRESHOLDS)), dtype=bool)
    if len(predictions) == 0 or len(target_classes) == 0:
        return correct
    pairwise_iou = box_iou(
        torch.from_numpy(target_boxes),
        torch.from_numpy(predictions[:, :4]),
    ).numpy()
    pairwise_iou *= target_classes[:, None] == predictions[:, 5].astype(np.int64)[None, :]
    for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
        matches = np.argwhere(pairwise_iou >= threshold)
        if len(matches) == 0:
            continue
        if len(matches) > 1:
            scores = pairwise_iou[matches[:, 0], matches[:, 1]]
            matches = matches[np.argsort(scores)[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), threshold_index] = True
    return correct


def _summarize(
    records: dict[str, list[np.ndarray]],
    names: dict[int, str],
) -> dict[str, Any]:
    correct = np.concatenate(records["correct"], axis=0)
    confidence = np.concatenate(records["confidence"], axis=0)
    predicted_classes = np.concatenate(records["predicted_classes"], axis=0)
    target_classes = np.concatenate(records["target_classes"], axis=0)
    (
        _,
        _,
        precision,
        recall,
        f1,
        ap,
        class_ids,
        precision_curve,
        recall_curve,
        f1_curve,
        confidence_axis,
        _,
    ) = ap_per_class(
        correct,
        confidence,
        predicted_classes,
        target_classes,
        names=names,
    )
    per_class: dict[str, Any] = {}
    for result_index, class_id in enumerate(class_ids):
        best_index = int(np.argmax(f1_curve[result_index]))
        operating_points: dict[str, dict[str, float]] = {}
        for threshold in (0.04, 0.05, 0.10, 0.15, 0.20, 0.25):
            threshold_index = int(round(threshold * (len(confidence_axis) - 1)))
            operating_points[f"{threshold:.2f}"] = {
                "precision": float(precision_curve[result_index, threshold_index]),
                "recall": float(recall_curve[result_index, threshold_index]),
                "f1": float(f1_curve[result_index, threshold_index]),
            }
        per_class[names.get(int(class_id), str(class_id))] = {
            "class_id": int(class_id),
            "precision": float(precision[result_index]),
            "recall": float(recall[result_index]),
            "f1": float(f1[result_index]),
            "map50": float(ap[result_index, 0]),
            "map50_95": float(ap[result_index].mean()),
            "class_specific_best_f1": {
                "confidence": float(confidence_axis[best_index]),
                "precision": float(precision_curve[result_index, best_index]),
                "recall": float(recall_curve[result_index, best_index]),
                "f1": float(f1_curve[result_index, best_index]),
            },
            "operating_points": operating_points,
        }
    return {
        "aggregate": {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "map50": float(ap[:, 0].mean()),
            "map50_95": float(ap.mean()),
        },
        "per_class": per_class,
        "prediction_count": int(len(predicted_classes)),
        "target_count": int(len(target_classes)),
    }


def evaluate(
    main_weights: Path,
    specialist_weights: Path,
    data_yaml: Path,
    *,
    split: str,
    image_size: int,
    device: str,
    chunk_size: int,
) -> dict[str, Any]:
    root, image_paths, names = _dataset(data_yaml, split)
    main = YOLO(str(main_weights), task="detect")
    specialist = YOLO(str(specialist_weights), task="detect")
    predict_options = {
        "stream": True,
        "imgsz": image_size,
        "device": device,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "verbose": False,
    }
    modes = {
        name: {key: [] for key in ("correct", "confidence", "predicted_classes", "target_classes")}
        for name in ("main", "specialist", "fusion")
    }
    main_seconds = 0.0
    specialist_seconds = 0.0
    started = time.perf_counter()
    for chunk_start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[chunk_start : chunk_start + chunk_size]
        main_stream = iter(
            main.predict(source=[str(path) for path in chunk], **predict_options)
        )
        specialist_stream = iter(
            specialist.predict(source=[str(path) for path in chunk], **predict_options)
        )
        for image_path in chunk:
            prediction_started = time.perf_counter()
            main_result = next(main_stream)
            main_seconds += time.perf_counter() - prediction_started
            prediction_started = time.perf_counter()
            specialist_result = next(specialist_stream)
            specialist_seconds += time.perf_counter() - prediction_started
            main_predictions = _predictions(main_result)
            specialist_predictions = _predictions(specialist_result)
            fused_predictions = np.concatenate(
                (
                    main_predictions[main_predictions[:, 5] != 1],
                    specialist_predictions[specialist_predictions[:, 5] == 1],
                ),
                axis=0,
            )
            height, width = main_result.orig_shape
            target_classes, target_boxes = _load_targets(
                _label_path(root, image_path),
                width,
                height,
            )
            for mode, predictions in (
                ("main", main_predictions),
                ("specialist", specialist_predictions),
                ("fusion", fused_predictions),
            ):
                modes[mode]["correct"].append(
                    _match_predictions(predictions, target_classes, target_boxes)
                )
                modes[mode]["confidence"].append(predictions[:, 4])
                modes[mode]["predicted_classes"].append(predictions[:, 5])
                modes[mode]["target_classes"].append(target_classes)
        if next(main_stream, None) is not None or next(specialist_stream, None) is not None:
            raise RuntimeError("model yielded more results than dataset images")
    return {
        "schema_version": 1,
        "policy": "thermal insan=uzman; arac/uap/uai=ana model",
        "main_weights": str(main_weights.resolve()),
        "specialist_weights": str(specialist_weights.resolve()),
        "data": str(data_yaml.resolve()),
        "split": split,
        "image_size": image_size,
        "image_count": len(image_paths),
        "chunk_size": chunk_size,
        "metrics": {mode: _summarize(records, names) for mode, records in modes.items()},
        "timing": {
            "total_seconds": time.perf_counter() - started,
            "main_ms_per_image": main_seconds * 1000 / len(image_paths),
            "specialist_ms_per_image": specialist_seconds * 1000 / len(image_paths),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ana YOLO + termal insan uzmanı birleşim testi")
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.main, args.specialist, args.data):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    report = evaluate(
        args.main,
        args.specialist,
        args.data,
        split=args.split,
        image_size=args.image_size,
        device=args.device,
        chunk_size=args.chunk_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
