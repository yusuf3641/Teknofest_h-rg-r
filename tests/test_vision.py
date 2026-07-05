from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image

from hurgor.mock_server import VideoFrameSource
from hurgor.models import DetectedObject, FrameMetadata
from hurgor.vision import (
    FrustumProjector,
    ONNXYoloDetector,
    OpticalFlowSE3Estimator,
    OptimizedObjectDetector,
    ORBReferenceMatcher,
    TopologicalNoiseFilter,
)


def _frame(health: int, index: int = 0) -> FrameMetadata:
    healthy = health == 1
    return FrameMetadata.model_validate(
        {
            "url": f"http://test/frames/{index}/",
            "image_url": f"/media/{index}.jpg",
            "video_name": "vision-test",
            "session": "http://test/session/1/",
            "translation_x": 0.0 if healthy else "NaN",
            "translation_y": 0.0 if healthy else "NaN",
            "translation_z": 10.0 if healthy else "NaN",
            "gps_health_status": health,
        }
    )


def _feature_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    image = rng.integers(0, 80, size=(256, 256), dtype=np.uint8)
    cv2.rectangle(image, (35, 40), (210, 190), 240, 4)
    cv2.circle(image, (120, 120), 45, 180, 5)
    cv2.putText(image, "HURGOR", (55, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    return image


def test_optical_flow_updates_se3_translation_when_gps_is_unhealthy() -> None:
    first = _feature_image()
    transform = np.float32([[1, 0, 6], [0, 1, -4]])
    second = cv2.warpAffine(first, transform, (256, 256))
    estimator = OpticalFlowSE3Estimator(500.0, 500.0, 10.0)

    initial = estimator.estimate(Image.fromarray(first), _frame(1, 0))
    estimated = estimator.estimate(Image.fromarray(second), _frame(0, 1))

    assert initial == (0.0, 0.0, 10.0)
    assert all(math.isfinite(value) for value in estimated)
    assert abs(estimated[0]) + abs(estimated[1]) > 0.01


def test_orb_homography_writes_undefined_object_box(tmp_path) -> None:
    reference = _feature_image()
    reference_path = tmp_path / "object_7.png"
    assert cv2.imwrite(str(reference_path), reference)
    matcher = ORBReferenceMatcher(str(tmp_path))

    matches = matcher.match(Image.fromarray(reference), _frame(1))

    assert len(matches) == 1
    assert matches[0].object_id == 7
    assert matches[0].bottom_right_x > matches[0].top_left_x


def test_volumetric_overlap_marks_landing_area_blocked() -> None:
    detections = [
        DetectedObject.from_class_id(
            2,
            base_url="http://test",
            top_left_x=80,
            top_left_y=80,
            bottom_right_x=200,
            bottom_right_y=200,
        ),
        DetectedObject.from_class_id(
            1,
            base_url="http://test",
            top_left_x=100,
            top_left_y=100,
            bottom_right_x=160,
            bottom_right_y=180,
        ),
    ]

    class FixedDetector:
        def detect(self, image, frame):
            del image, frame
            return detections

    detector = OptimizedObjectDetector(
        FixedDetector(), TopologicalNoiseFilter(), FrustumProjector(500, 500, 10)
    )
    output = detector.detect_fast(Image.new("RGB", (256, 256)), _frame(1))

    assert output[0].landing_status == "0"


def test_video_frame_source_reads_real_video(tmp_path) -> None:
    video_path = tmp_path / "mock.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 7.5, (96, 64))
    assert writer.isOpened()
    for index in range(4):
        writer.write(np.full((64, 96, 3), index * 50, dtype=np.uint8))
    writer.release()

    source = VideoFrameSource(str(video_path))
    try:
        content = source.render(2)
    finally:
        source.close()

    assert source.frame_count == 4
    with Image.open(io.BytesIO(content)) as image:
        assert image.size == (96, 64)


def test_runtime_rejects_raw_training_weight(tmp_path) -> None:
    raw_weight = tmp_path / "model.pt"
    raw_weight.write_bytes(b"not-a-model")
    try:
        ONNXYoloDetector(str(raw_weight))
    except ValueError as exc:
        assert "ONNX" in str(exc)
    else:
        raise AssertionError("raw .pt runtime weight must be rejected")
