from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from hurgor.models import DetectedObject, FrameMetadata


def _payload() -> dict[str, object]:
    return {
        "url": "http://test/frames/1/",
        "image_url": "/media/frame_000001.jpg",
        "video_name": "test",
        "session": "http://test/session/1/",
        "translation_x": "NaN",
        "translation_y": "NaN",
        "translation_z": "NaN",
        "health_status": 0,
    }


def test_pdf_health_status_alias_and_nan_strings_are_supported() -> None:
    frame = FrameMetadata.model_validate(_payload())
    assert frame.gps_health_status == 0
    assert math.isnan(frame.translation_x)
    assert frame.reference_translation is None


def test_gps_health_status_name_is_also_supported() -> None:
    payload = _payload()
    payload.pop("health_status")
    payload["gps_health_status"] = 1
    payload.update(translation_x=1.0, translation_y=2.0, translation_z=3.0)
    frame = FrameMetadata.model_validate(payload)
    assert frame.reference_translation == (1.0, 2.0, 3.0)


def test_optional_orientation_quaternion_is_normalized_and_exposes_heading() -> None:
    payload = _payload()
    payload.update(
        orientation_x=0.0,
        orientation_y=0.0,
        orientation_z=math.sin(math.pi / 8.0) * 2.0,
        orientation_w=math.cos(math.pi / 8.0) * 2.0,
    )

    frame = FrameMetadata.model_validate(payload)

    assert frame.orientation_quaternion is not None
    assert math.isclose(frame.orientation_heading_rad or 0.0, math.pi / 4.0)


def test_partial_or_invalid_orientation_is_treated_as_unavailable() -> None:
    payload = _payload()
    payload.update(orientation_x=0.0, orientation_y=0.0, orientation_z=0.0)
    assert FrameMetadata.model_validate(payload).orientation_heading_rad is None

    payload["orientation_w"] = 0.0
    assert FrameMetadata.model_validate(payload).orientation_heading_rad is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_outgoing_bounding_boxes_reject_non_finite_coordinates(invalid: float) -> None:
    with pytest.raises(ValidationError):
        DetectedObject(
            cls="http://test/classes/0/",
            top_left_x=invalid,
            top_left_y=0,
            bottom_right_x=10,
            bottom_right_y=10,
        )
