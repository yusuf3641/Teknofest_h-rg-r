from __future__ import annotations

import json

import pytest

from hurgor.detector_calibration import load_detector_thresholds, select_operating_point


def test_select_operating_point_uses_validation_f1_maximum() -> None:
    selected = select_operating_point(
        [0.1, 0.2, 0.3],
        [0.4, 0.7, 0.9],
        [0.8, 0.6, 0.2],
    )

    assert selected["index"] == 1
    assert selected["threshold"] == 0.2


def test_threshold_profile_is_bound_to_model_hash_and_class_order(tmp_path) -> None:
    profile = tmp_path / "thresholds.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_model_sha256": "abc",
                "classes": ["arac", "insan", "uap", "uai"],
                "thresholds": {"arac": 0.25, "insan": 0.158, "uap": 0.25, "uai": 0.25},
            }
        ),
        encoding="utf-8",
    )

    thresholds, _ = load_detector_thresholds(
        str(profile),
        runtime_model_sha256="abc",
        class_names=["arac", "insan", "uap", "uai"],
    )

    assert thresholds["insan"] == 0.158
    with pytest.raises(ValueError, match="different model"):
        load_detector_thresholds(
            str(profile),
            runtime_model_sha256="def",
            class_names=["arac", "insan", "uap", "uai"],
        )
