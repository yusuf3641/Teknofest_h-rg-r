from __future__ import annotations

import math

from hurgor.models import FrameMetadata


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
