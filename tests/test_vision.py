from __future__ import annotations

import io
import math
import time

import cv2
import numpy as np
from PIL import Image

from hurgor.config import ClientSettings
from hurgor.inference import (
    LastKnownPositionEstimator,
    NoopUndefinedObjectMatcher,
    PipelineInferenceEngine,
)
from hurgor.mock_server import VideoFrameSource
from hurgor.models import DetectedObject, DetectedUndefinedObject, FrameMetadata
from hurgor.vision import (
    FrustumProjector,
    MotionCompensatedMotionClassifier,
    ONNXYoloDetector,
    OpticalFlowSE3Estimator,
    OptimizedObjectDetector,
    ORBReferenceMatcher,
    ThermalHumanFusionDetector,
    TopologicalNoiseFilter,
    _select_nms_indices,
    build_vision_components,
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


def test_volumetric_overlap_marks_clear_landing_area_available() -> None:
    detections = [
        DetectedObject.from_class_id(
            3,
            base_url="http://test",
            top_left_x=80,
            top_left_y=80,
            bottom_right_x=200,
            bottom_right_y=200,
        ),
        DetectedObject.from_class_id(
            0,
            base_url="http://test",
            top_left_x=10,
            top_left_y=10,
            bottom_right_x=40,
            bottom_right_y=40,
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

    assert output[0].landing_status == "1"


def test_landing_area_touching_image_boundary_is_not_safe() -> None:
    detection = DetectedObject.from_class_id(
        2,
        base_url="http://test",
        top_left_x=0,
        top_left_y=80,
        bottom_right_x=120,
        bottom_right_y=200,
    )

    class FixedDetector:
        def detect(self, image, frame):
            del image, frame
            return [detection]

    detector = OptimizedObjectDetector(
        FixedDetector(), TopologicalNoiseFilter(), FrustumProjector(500, 500, 10)
    )

    output = detector.detect_fast(Image.new("RGB", (256, 256)), _frame(1))

    assert output[0].landing_status == "0"


def test_undefined_object_overlap_blocks_landing_after_reference_matching() -> None:
    landing_area = DetectedObject.from_class_id(
        3,
        base_url="http://test",
        top_left_x=60,
        top_left_y=60,
        bottom_right_x=180,
        bottom_right_y=180,
    )

    class FixedDetector:
        def detect(self, image, frame):
            del image, frame
            return [landing_area]

    class FixedMatcher:
        def match(self, image, frame):
            del image, frame
            return [
                DetectedUndefinedObject(
                    object_id=9,
                    top_left_x=100,
                    top_left_y=100,
                    bottom_right_x=140,
                    bottom_right_y=140,
                )
            ]

    detector = OptimizedObjectDetector(
        FixedDetector(), TopologicalNoiseFilter(), FrustumProjector(500, 500, 10)
    )
    engine = PipelineInferenceEngine(object_detector=detector, undefined_matcher=FixedMatcher())
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256)).save(buffer, format="JPEG")

    outcome = engine.infer_timed(_frame(1), buffer.getvalue(), "http://test/users/1/")

    assert outcome.prediction.detected_objects[0].landing_status == "0"
    assert outcome.timings_ms["landing_analysis_ms"] >= 0


def test_optimized_detector_removes_duplicate_boxes() -> None:
    detections = [
        DetectedObject.from_class_id(
            1,
            base_url="http://test",
            top_left_x=40 + offset,
            top_left_y=40 + offset,
            bottom_right_x=100 + offset,
            bottom_right_y=120 + offset,
        )
        for offset in (0, 2)
    ]
    detections.append(
        DetectedObject.from_class_id(
            1,
            base_url="http://test",
            top_left_x=160,
            top_left_y=40,
            bottom_right_x=210,
            bottom_right_y=120,
        )
    )

    class FixedDetector:
        def detect(self, image, frame):
            del image, frame
            return detections

    detector = OptimizedObjectDetector(
        FixedDetector(), TopologicalNoiseFilter(), FrustumProjector(500, 500, 10)
    )

    output = detector.detect_fast(Image.new("RGB", (256, 256)), _frame(1))

    assert len(output) == 2


def test_score_ordered_nms_suppresses_same_and_cross_class_duplicates() -> None:
    boxes = [[10, 10, 50, 50], [11, 11, 50, 50], [10, 10, 50, 50], [100, 100, 20, 20]]
    scores = [0.70, 0.95, 0.90, 0.60]
    classes = [0, 0, 1, 1]

    selected = _select_nms_indices(
        boxes,
        scores,
        classes,
        same_class_iou=0.45,
        cross_class_iou=0.90,
    )

    assert selected == [1, 3]


def test_thermal_fusion_replaces_only_humans_and_keeps_rgb_on_main() -> None:
    class RecordingDetector:
        def __init__(self, detections):
            self.detections = detections
            self.calls = 0

        def warmup(self):
            return None

        def health(self):
            return {"ok": True}

        def close(self):
            return None

        def model_info(self):
            return {"type": "recording"}

        def detect(self, image, frame):
            del image, frame
            self.calls += 1
            return self.detections

    main = RecordingDetector(
        [
            DetectedObject.from_class_id(
                class_id,
                base_url="http://test",
                top_left_x=10 + class_id * 20,
                top_left_y=10,
                bottom_right_x=25 + class_id * 20,
                bottom_right_y=30,
            )
            for class_id in (0, 1, 2, 3)
        ]
    )
    specialist_human = DetectedObject.from_class_id(
        1,
        base_url="http://test",
        top_left_x=100,
        top_left_y=100,
        bottom_right_x=130,
        bottom_right_y=150,
    )
    specialist = RecordingDetector(
        [
            DetectedObject.from_class_id(
                0,
                base_url="http://test",
                top_left_x=200,
                top_left_y=200,
                bottom_right_x=230,
                bottom_right_y=230,
            ),
            specialist_human,
        ]
    )
    detector = ThermalHumanFusionDetector(main, specialist)
    image = Image.new("RGB", (256, 256))

    rgb = detector.detect(image, _frame(1).model_copy(update={"modality": "rgb"}))
    thermal = detector.detect(
        image,
        _frame(1).model_copy(update={"modality": "thermal"}),
    )

    assert [item.class_id for item in rgb] == ["0", "1", "2", "3"]
    assert [item.class_id for item in thermal] == ["0", "2", "3", "1"]
    assert thermal[-1].top_left_x == 100
    assert main.calls == 2
    assert specialist.calls == 1
    detector.close()


def test_thermal_fusion_falls_back_and_degraded_mode_bypasses_specialist() -> None:
    main_human = DetectedObject.from_class_id(
        1,
        base_url="http://test",
        top_left_x=10,
        top_left_y=10,
        bottom_right_x=30,
        bottom_right_y=40,
    )

    class MainDetector:
        def warmup(self):
            return None

        def health(self):
            return {"ok": True}

        def close(self):
            return None

        def model_info(self):
            return {"type": "main"}

        def detect(self, image, frame):
            del image, frame
            return [main_human]

    class BrokenSpecialist(MainDetector):
        calls = 0

        def detect(self, image, frame):
            del image, frame
            self.calls += 1
            raise RuntimeError("synthetic specialist failure")

    specialist = BrokenSpecialist()
    detector = ThermalHumanFusionDetector(MainDetector(), specialist)
    frame = _frame(1).model_copy(update={"modality": "thermal"})

    normal = detector.detect(Image.new("RGB", (64, 64)), frame)
    second_normal = detector.detect(Image.new("RGB", (64, 64)), frame)
    degraded = detector.detect_fast(Image.new("RGB", (64, 64)), frame)

    assert normal == [main_human]
    assert second_normal == [main_human]
    assert degraded == [main_human]
    assert specialist.calls == 1
    assert detector.health()["ok"] is True
    assert detector.health()["thermal_specialist"]["active"] is False
    detector.close()


def test_thermal_fusion_sheds_specialist_load_before_watchdog_timeout() -> None:
    class Detector:
        def __init__(self, *, delay: float = 0.0):
            self.delay = delay
            self.calls = 0

        def warmup(self):
            return None

        def health(self):
            return {"ok": True}

        def close(self):
            return None

        def model_info(self):
            return {"type": "fixed"}

        def detect(self, image, frame):
            del image, frame
            self.calls += 1
            if self.delay:
                import time

                time.sleep(self.delay)
            return []

    main = Detector()
    specialist = Detector(delay=0.005)
    detector = ThermalHumanFusionDetector(
        main,
        specialist,
        slow_threshold_ms=1.0,
        cooldown_frames=2,
        cooldown_seconds=0.0,
    )
    frame = _frame(1).model_copy(update={"modality": "thermal"})
    image = Image.new("RGB", (64, 64))

    for _ in range(4):
        detector.detect(image, frame)

    assert main.calls == 4
    assert specialist.calls == 2
    assert detector.health()["specialist_cooldown_remaining"] == 2
    detector.close()


def test_thermal_specialist_timeout_returns_main_before_outer_watchdog() -> None:
    main_human = DetectedObject.from_class_id(
        1,
        base_url="http://test",
        top_left_x=5,
        top_left_y=5,
        bottom_right_x=15,
        bottom_right_y=20,
    )

    class Detector:
        def __init__(self, *, delay: float = 0.0):
            self.delay = delay

        def warmup(self):
            return None

        def health(self):
            return {"ok": True}

        def close(self):
            return None

        def model_info(self):
            return {"type": "fixed"}

        def detect(self, image, frame):
            del image, frame
            if self.delay:
                import time

                time.sleep(self.delay)
            return [main_human]

    detector = ThermalHumanFusionDetector(
        Detector(),
        Detector(delay=0.05),
        specialist_timeout_ms=5.0,
        cooldown_frames=2,
        cooldown_seconds=0.0,
    )
    frame = _frame(1).model_copy(update={"modality": "thermal"})
    started = time.monotonic()
    output = detector.detect(Image.new("RGB", (64, 64)), frame)
    elapsed = time.monotonic() - started

    assert output == [main_human]
    assert elapsed < 0.04
    detector.close()


def test_optimized_detector_delegates_lifecycle_methods() -> None:
    class FixedDetector:
        warmed = False
        closed = False

        def warmup(self):
            self.warmed = True

        def health(self):
            return {"ok": True, "backend": "fixed"}

        def close(self):
            self.closed = True

        def model_info(self):
            return {"type": "fixed"}

        def detect(self, image, frame):
            del image, frame
            return []

    fixed = FixedDetector()
    detector = OptimizedObjectDetector(
        fixed, TopologicalNoiseFilter(), FrustumProjector(500, 500, 10)
    )

    detector.warmup()
    detector.close()

    assert fixed.warmed is True
    assert fixed.closed is True
    assert detector.health()["wrapper"] == "optimized"
    assert detector.model_info()["type"] == "fixed"


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


def test_motion_classifier_compensates_global_camera_motion_with_hysteresis() -> None:
    classifier = MotionCompensatedMotionClassifier(hysteresis_frames=2)

    def vehicles(offsets: tuple[float, float, float]) -> list[DetectedObject]:
        return [
            DetectedObject.from_class_id(
                0,
                base_url="http://test",
                top_left_x=base + offset,
                top_left_y=20,
                bottom_right_x=base + offset + 40,
                bottom_right_y=60,
            )
            for base, offset in zip((20, 120, 220), offsets, strict=True)
        ]

    first = classifier.update(vehicles((0, 0, 0)), "video")
    second = classifier.update(vehicles((10, 10, 30)), "video")
    third = classifier.update(vehicles((20, 20, 60)), "video")

    assert [item.motion_status for item in first] == ["-1", "-1", "-1"]
    assert [item.motion_status for item in second] == ["-1", "-1", "-1"]
    assert [item.motion_status for item in third] == ["0", "0", "1"]


def test_motion_classifier_uses_non_vehicle_landmarks_for_camera_compensation() -> None:
    classifier = MotionCompensatedMotionClassifier(hysteresis_frames=2)

    def scene(vehicle_extra_dx: float) -> list[DetectedObject]:
        global_dx = 10
        return [
            DetectedObject.from_class_id(
                0,
                base_url="http://test",
                top_left_x=40 + global_dx + vehicle_extra_dx,
                top_left_y=40,
                bottom_right_x=80 + global_dx + vehicle_extra_dx,
                bottom_right_y=80,
            ),
            DetectedObject.from_class_id(
                1,
                base_url="http://test",
                top_left_x=140 + global_dx,
                top_left_y=40,
                bottom_right_x=170 + global_dx,
                bottom_right_y=90,
            ),
            DetectedObject.from_class_id(
                2,
                base_url="http://test",
                top_left_x=220 + global_dx,
                top_left_y=40,
                bottom_right_x=280 + global_dx,
                bottom_right_y=100,
            ),
        ]

    first = classifier.update(scene(0), "video")
    second = classifier.update(scene(30), "video")
    third = classifier.update(scene(60), "video")

    assert first[0].motion_status == "-1"
    assert second[0].motion_status == "-1"
    assert third[0].motion_status == "1"
    assert [item.motion_status for item in third[1:]] == ["-1", "-1"]


def test_motion_classifier_compensates_camera_rotation_and_scale() -> None:
    classifier = MotionCompensatedMotionClassifier(hysteresis_frames=2)
    base_items = [
        (0, 50.0, 60.0, 30.0, 30.0),
        (0, 130.0, 70.0, 30.0, 30.0),
        (1, 80.0, 170.0, 24.0, 40.0),
        (2, 210.0, 80.0, 40.0, 40.0),
        (3, 220.0, 190.0, 40.0, 40.0),
    ]

    def scene(step: int) -> list[DetectedObject]:
        angle = step * 0.025
        scale = 1.0 + step * 0.01
        cosine = math.cos(angle) * scale
        sine = math.sin(angle) * scale
        output = []
        for index, (class_id, x, y, width, height) in enumerate(base_items):
            center_x = cosine * x - sine * y + step * 7
            center_y = sine * x + cosine * y - step * 4
            if index == 1:
                center_x += step * 18
            output.append(
                DetectedObject.from_class_id(
                    class_id,
                    base_url="http://test",
                    top_left_x=center_x - width / 2,
                    top_left_y=center_y - height / 2,
                    bottom_right_x=center_x + width / 2,
                    bottom_right_y=center_y + height / 2,
                )
            )
        return output

    classifier.update(scene(0), "affine-video")
    classifier.update(scene(1), "affine-video")
    third = classifier.update(scene(2), "affine-video")

    assert [item.motion_status for item in third[:2]] == ["0", "1"]


def test_motion_classifier_keeps_unknown_without_camera_motion_evidence() -> None:
    classifier = MotionCompensatedMotionClassifier(hysteresis_frames=2)

    def vehicle(offset: float) -> list[DetectedObject]:
        return [
            DetectedObject.from_class_id(
                0,
                base_url="http://test",
                top_left_x=30 + offset,
                top_left_y=30,
                bottom_right_x=80 + offset,
                bottom_right_y=80,
            )
        ]

    classifier.update(vehicle(0), "single-vehicle")
    classifier.update(vehicle(10), "single-vehicle")
    third = classifier.update(vehicle(20), "single-vehicle")

    assert third[0].motion_status == "-1"


def test_motion_classifier_resets_tracks_between_videos() -> None:
    classifier = MotionCompensatedMotionClassifier(hysteresis_frames=2)
    vehicle = DetectedObject.from_class_id(
        0,
        base_url="http://test",
        top_left_x=20,
        top_left_y=20,
        bottom_right_x=60,
        bottom_right_y=60,
    )

    classifier.update([vehicle], "first-video")
    reset = classifier.update(
        [vehicle.model_copy(update={"top_left_x": 40, "bottom_right_x": 80})],
        "new-video",
    )

    assert reset[0].motion_status == "-1"


def test_repeated_frozen_frames_never_claim_vehicle_is_stationary() -> None:
    detections = [
        DetectedObject.from_class_id(
            class_id,
            base_url="http://test",
            top_left_x=20 + index * 60,
            top_left_y=20,
            bottom_right_x=60 + index * 60,
            bottom_right_y=70,
        )
        for index, class_id in enumerate((0, 1, 2))
    ]

    class FixedDetector:
        def detect(self, image, frame):
            del image, frame
            return detections

    detector = OptimizedObjectDetector(
        FixedDetector(), TopologicalNoiseFilter(), FrustumProjector(500, 500, 10)
    )
    frozen = Image.new("RGB", (256, 256), (80, 90, 100))

    first = detector.detect_fast(frozen, _frame(1, 0))
    second = detector.detect_fast(frozen, _frame(1, 1))
    third = detector.detect_fast(frozen, _frame(1, 2))

    assert first[0].motion_status == "-1"
    assert second[0].motion_status == "-1"
    assert third[0].motion_status == "-1"


def test_unvalidated_visual_odometry_is_disabled_by_default() -> None:
    _, position, _ = build_vision_components(ClientSettings())
    assert isinstance(position, LastKnownPositionEstimator)


def test_missing_optional_reference_directory_falls_back_to_noop(tmp_path) -> None:
    _, _, matcher = build_vision_components(
        ClientSettings(reference_images_dir=str(tmp_path / "missing"))
    )

    assert isinstance(matcher, NoopUndefinedObjectMatcher)
