from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXPECTED_DETECTOR_CLASSES = ("arac", "insan", "uap", "uai")


def load_detector_thresholds(
    path: str,
    *,
    runtime_model_sha256: str,
    class_names: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Load a model-bound per-class threshold profile and fail closed on drift."""

    target = Path(path).expanduser().resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("detector threshold profile schema_version must be 1")
    expected_classes = list(class_names)
    if expected_classes != list(EXPECTED_DETECTOR_CLASSES):
        raise ValueError("unsafe runtime detector class order")
    if payload.get("classes") != expected_classes:
        raise ValueError("detector threshold profile class order does not match runtime")
    profile_sha = str(payload.get("runtime_model_sha256", "")).lower()
    if profile_sha != runtime_model_sha256.lower():
        raise ValueError("detector threshold profile is bound to a different model")

    raw_thresholds = payload.get("thresholds")
    if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != set(expected_classes):
        raise ValueError("detector threshold profile must define every runtime class exactly once")
    thresholds: dict[str, float] = {}
    for class_name in expected_classes:
        raw_value = raw_thresholds[class_name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError(f"detector threshold for {class_name} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"detector threshold for {class_name} must be in (0, 1]")
        thresholds[class_name] = value
    return thresholds, payload


def select_operating_point(
    thresholds: Sequence[float],
    precision: Sequence[float],
    recall: Sequence[float],
    *,
    beta: float = 1.0,
) -> dict[str, float | int]:
    """Select the validation threshold that maximizes an F-beta score."""

    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("beta must be a finite positive number")
    if not thresholds or len(thresholds) != len(precision) or len(thresholds) != len(recall):
        raise ValueError("threshold, precision and recall curves must be non-empty and aligned")
    beta_squared = beta * beta
    best_index = 0
    best_score = -1.0
    for index, (threshold, p_value, r_value) in enumerate(
        zip(thresholds, precision, recall, strict=True)
    ):
        values = (float(threshold), float(p_value), float(r_value))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("calibration curves must contain only finite values")
        denominator = beta_squared * values[1] + values[2]
        score = (
            (1 + beta_squared) * values[1] * values[2] / denominator
            if denominator > 0
            else 0.0
        )
        if score > best_score:
            best_index = index
            best_score = score
    return {
        "index": best_index,
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f_beta": best_score,
        "beta": float(beta),
    }
