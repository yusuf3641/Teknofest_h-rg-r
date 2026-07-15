from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hurgor.config import ClientSettings
from hurgor.models import FrameMetadata
from hurgor.odometry import (
    CalibratedHomographySE3Estimator,
    HomographyMotionEstimator,
    MotionObservation,
    RobustMotionCalibration,
)
from hurgor.vision import build_vision_components
from tools.evaluate_odometry import build_parser, evaluate
from tools.generate_odometry_fixture import generate


def _textured_frame() -> np.ndarray:
    rng = np.random.default_rng(20260713)
    image = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)
    for _ in range(80):
        x, y = rng.integers(20, 620), rng.integers(20, 460)
        cv2.circle(
            image,
            (int(x), int(y)),
            int(rng.integers(2, 8)),
            int(rng.integers(0, 256)),
            -1,
        )
    return image


def _observation(
    feature: tuple[float, float, float],
    *,
    yaw_rad: float = 0.0,
) -> MotionObservation:
    return MotionObservation(
        feature=feature,
        yaw_rad=yaw_rad,
        tracked_points=200,
        inlier_points=190,
        inlier_ratio=0.95,
        reprojection_error_px=0.1,
        valid=True,
    )


def _frame(
    index: int,
    position: tuple[float, float, float],
    *,
    healthy: bool,
    orientation_yaw: float | None = None,
) -> FrameMetadata:
    values: tuple[float | str, float | str, float | str]
    values = position if healthy else ("NaN", "NaN", "NaN")
    half_yaw = orientation_yaw / 2.0 if orientation_yaw is not None else None
    return FrameMetadata(
        url=f"http://test/frames/{index}/",
        image_url=f"frame_{index:06d}",
        video_name="odometry-test",
        session="http://test/session/1/",
        translation_x=values[0],
        translation_y=values[1],
        translation_z=values[2],
        gps_health_status=1 if healthy else 0,
        orientation_x=0.0 if half_yaw is not None else None,
        orientation_y=0.0 if half_yaw is not None else None,
        orientation_z=math.sin(half_yaw) if half_yaw is not None else None,
        orientation_w=math.cos(half_yaw) if half_yaw is not None else None,
    )


@dataclass
class _MotionSequence:
    observations: list[MotionObservation]
    index: int = 0

    def observe(self, previous_gray, current_gray, principal_point) -> MotionObservation:
        del previous_gray, current_gray, principal_point
        observation = self.observations[self.index]
        self.index += 1
        return observation


def _mapped_delta(feature: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, scale = feature
    return (
        0.30 * x - 0.12 * y + 8.0 * scale,
        0.08 * x + 0.25 * y - 3.0 * scale,
        -0.02 * x + 0.04 * y + 45.0 * scale,
    )


def test_homography_motion_removes_rotation_and_scale_from_translation() -> None:
    previous = _textured_frame()
    transform = cv2.getRotationMatrix2D((320, 240), 3.0, 1.01)
    transform[:, 2] += np.asarray((7.0, -4.0))
    current = cv2.warpAffine(
        previous,
        transform,
        (640, 480),
        borderMode=cv2.BORDER_REFLECT,
    )

    observation = HomographyMotionEstimator().observe(previous, current, (320.0, 240.0))

    assert observation.valid is True
    assert observation.inlier_ratio > 0.9
    assert observation.reprojection_error_px < 0.2
    assert np.allclose(observation.feature[:2], (7.0, -4.0), atol=0.2)
    assert math.isclose(observation.feature[2], math.log(1.01), abs_tol=0.002)
    assert math.isclose(abs(observation.yaw_rad), math.radians(3.0), abs_tol=0.003)


def test_homography_projective_mode_emits_stable_grid_flow_features() -> None:
    previous = _textured_frame()
    transform = cv2.getRotationMatrix2D((320, 240), 2.0, 1.005)
    transform[:, 2] += np.asarray((5.0, -3.0))
    current = cv2.warpAffine(previous, transform, (640, 480), borderMode=cv2.BORDER_REFLECT)

    observation = HomographyMotionEstimator(projective_features=True).observe(
        previous,
        current,
        (320.0, 240.0),
    )

    assert observation.valid is True
    assert len(observation.feature) == 18
    assert all(math.isfinite(value) for value in observation.feature)


def test_robust_calibration_supports_projective_feature_vectors() -> None:
    calibration = RobustMotionCalibration(min_samples=8, max_samples=40, calibrate_gain=True)
    for index in range(24):
        feature = tuple(
            math.sin(index * 0.23 + component * 0.41) * (component + 1) for component in range(18)
        )
        target = (
            0.02 * feature[0] - 0.01 * feature[5],
            0.03 * feature[1] + 0.005 * feature[8],
            -0.01 * feature[4] + 0.004 * feature[12],
        )
        calibration.add(feature, target, quality=0.95)

    probe = tuple(
        math.sin(25 * 0.23 + component * 0.41) * (component + 1) for component in range(18)
    )
    predicted = calibration.predict(probe)

    assert calibration.ready is True
    assert predicted is not None
    assert 0.5 <= calibration.prediction_gain <= 2.0
    assert len(calibration.diagnostics()["metric_mapping"]) == 18


def test_robust_calibration_learns_camera_axes_and_metric_scale() -> None:
    calibration = RobustMotionCalibration(min_samples=8, max_samples=40)
    for index in range(24):
        feature = (
            2.0 + (index % 5) * 0.7,
            -1.5 + (index % 7) * 0.4,
            -0.008 + (index % 4) * 0.006,
        )
        calibration.add(feature, _mapped_delta(feature), quality=0.95)

    probe = (3.3, -0.4, 0.011)
    predicted = calibration.predict(probe)

    assert calibration.ready is True
    assert calibration.feature_rank == 3
    assert calibration.fit_mae_m < 0.001
    assert predicted is not None
    assert np.allclose(predicted, _mapped_delta(probe), atol=0.002)


def test_calibrated_odometry_beats_hold_and_reanchors_after_dropout() -> None:
    features = [
        (
            2.0 + (index % 5) * 0.55,
            -1.2 + (index % 4) * 0.45,
            -0.006 + (index % 3) * 0.007,
        )
        for index in range(1, 19)
    ]
    observations = [_observation(feature) for feature in features]
    estimator = CalibratedHomographySE3Estimator(
        1000.0,
        1000.0,
        10.0,
        min_calibration_samples=6,
        max_step_m=20.0,
        require_cross_validation=False,
    )
    estimator._motion = _MotionSequence(observations)  # type: ignore[assignment]
    image = Image.fromarray(np.full((64, 64), 120, dtype=np.uint8))

    truth = [(10.0, 20.0, 30.0)]
    for feature in features:
        delta = _mapped_delta(feature)
        truth.append(tuple(left + right for left, right in zip(truth[-1], delta, strict=True)))

    estimates: list[tuple[float, float, float]] = []
    for index in range(16):
        healthy = index < 10
        estimates.append(estimator.estimate(image, _frame(index, truth[index], healthy=healthy)))

    hold = truth[9]
    candidate_errors = [
        np.linalg.norm(np.asarray(estimates[index]) - np.asarray(truth[index]))
        for index in range(10, 16)
    ]
    hold_errors = [
        np.linalg.norm(np.asarray(hold) - np.asarray(truth[index])) for index in range(10, 16)
    ]

    assert estimator.diagnostics()["calibration"]["ready"] is True
    assert float(np.mean(candidate_errors)) < 0.01
    assert float(np.mean(candidate_errors)) < float(np.mean(hold_errors))

    recovered = estimator.estimate(image, _frame(16, truth[16], healthy=True))
    assert recovered == truth[16]
    assert estimator.diagnostics()["reanchors"] == 1


def test_odometry_checkpoint_restores_calibration_pose_and_previous_frame() -> None:
    features = [
        (
            1.7 + (index % 4) * 0.45,
            -0.9 + (index % 5) * 0.31,
            -0.005 + (index % 3) * 0.006,
        )
        for index in range(12)
    ]
    truth = [(3.0, -2.0, 20.0)]
    for feature in features:
        delta = _mapped_delta(feature)
        truth.append(tuple(left + right for left, right in zip(truth[-1], delta, strict=True)))

    estimator = CalibratedHomographySE3Estimator(
        1000.0,
        1000.0,
        10.0,
        min_calibration_samples=6,
        require_cross_validation=False,
    )
    estimator._motion = _MotionSequence(  # type: ignore[assignment]
        [_observation(feature) for feature in features[:9]]
    )
    image = Image.fromarray(np.full((64, 64), 95, dtype=np.uint8))
    for index in range(10):
        estimator.estimate(image, _frame(index, truth[index], healthy=True))

    checkpoint = estimator.export_recovery_state()
    restored = CalibratedHomographySE3Estimator(
        1000.0,
        1000.0,
        10.0,
        min_calibration_samples=6,
        require_cross_validation=False,
    )
    restored._motion = _MotionSequence(  # type: ignore[assignment]
        [_observation(features[9])]
    )
    restored.restore_recovery_state(
        checkpoint,
        image,
        _frame(9, truth[9], healthy=True),
    )

    estimate = restored.estimate(image, _frame(10, truth[10], healthy=False))

    assert restored.diagnostics()["calibration"]["ready"] is True
    assert np.allclose(estimate, truth[10], atol=0.002)


def test_camera_local_motion_is_rotated_into_world_after_axis_calibration() -> None:
    local_features = [
        (
            1.5 + (index % 4) * 0.45,
            -0.8 + (index % 5) * 0.32,
            -0.005 + (index % 3) * 0.006,
        )
        for index in range(1, 18)
    ]
    heading = 0.0
    observations: list[MotionObservation] = []
    truth = [(0.0, 0.0, 20.0)]
    for index, local_feature in enumerate(local_features, start=1):
        yaw = math.radians(2.0 + (index % 3) * 0.5)
        heading += yaw
        cosine, sine = math.cos(heading), math.sin(heading)
        local_delta = _mapped_delta(local_feature)
        world_delta = (
            cosine * local_delta[0] - sine * local_delta[1],
            sine * local_delta[0] + cosine * local_delta[1],
            local_delta[2],
        )
        truth.append(
            tuple(left + right for left, right in zip(truth[-1], world_delta, strict=True))
        )
        observations.append(_observation(local_feature, yaw_rad=yaw))

    estimator = CalibratedHomographySE3Estimator(
        1000.0,
        1000.0,
        10.0,
        min_calibration_samples=6,
        max_step_m=20.0,
        require_cross_validation=False,
    )
    estimator._motion = _MotionSequence(observations)  # type: ignore[assignment]
    image = Image.fromarray(np.full((64, 64), 100, dtype=np.uint8))

    estimates = []
    for index in range(16):
        estimates.append(estimator.estimate(image, _frame(index, truth[index], healthy=index < 10)))

    outage_errors = [
        np.linalg.norm(np.asarray(estimates[index]) - np.asarray(truth[index]))
        for index in range(10, 16)
    ]
    assert float(np.mean(outage_errors)) < 0.01
    assert abs(float(estimator.diagnostics()["image_heading_deg"])) > 30.0


def test_optional_orientation_telemetry_overrides_drifting_image_yaw() -> None:
    local_features = [
        (1.2 + (index % 4) * 0.3, -0.7 + (index % 5) * 0.2, 0.002 * (index % 3))
        for index in range(1, 16)
    ]
    headings = [0.0]
    observations: list[MotionObservation] = []
    truth = [(0.0, 0.0, 20.0)]
    for local_feature in local_features:
        headings.append(headings[-1] + math.radians(3.0))
        local_delta = _mapped_delta(local_feature)
        cosine, sine = math.cos(headings[-1]), math.sin(headings[-1])
        world_delta = (
            cosine * local_delta[0] - sine * local_delta[1],
            sine * local_delta[0] + cosine * local_delta[1],
            local_delta[2],
        )
        truth.append(
            tuple(left + right for left, right in zip(truth[-1], world_delta, strict=True))
        )
        # Deliberately wrong image-derived yaw: telemetry must take precedence.
        observations.append(_observation(local_feature, yaw_rad=math.radians(7.0)))

    estimator = CalibratedHomographySE3Estimator(
        1000.0,
        1000.0,
        10.0,
        min_calibration_samples=6,
        max_step_m=20.0,
        require_cross_validation=False,
    )
    estimator._motion = _MotionSequence(observations)  # type: ignore[assignment]
    image = Image.fromarray(np.full((64, 64), 100, dtype=np.uint8))

    estimates = [
        estimator.estimate(
            image,
            _frame(
                index,
                truth[index],
                healthy=index < 10,
                orientation_yaw=headings[index],
            ),
        )
        for index in range(16)
    ]

    outage_errors = [
        np.linalg.norm(np.asarray(estimates[index]) - np.asarray(truth[index]))
        for index in range(10, 16)
    ]
    diagnostics = estimator.diagnostics()
    assert float(np.mean(outage_errors)) < 0.01
    assert diagnostics["heading_source"] == "telemetry"
    assert diagnostics["orientation_telemetry_frames"] == 16


def test_feature_failure_uses_bounded_decaying_fallback_without_jump() -> None:
    valid_features = [
        (1.0 + (index % 3), 0.4 * (index % 4), 0.002 * (index % 2)) for index in range(1, 9)
    ]
    observations = [_observation(feature) for feature in valid_features]
    observations.extend(
        [
            MotionObservation.invalid("insufficient_features"),
            MotionObservation.invalid("insufficient_features"),
        ]
    )
    estimator = CalibratedHomographySE3Estimator(
        1000.0,
        1000.0,
        10.0,
        min_calibration_samples=6,
        max_step_m=5.0,
        fallback_decay=0.5,
        require_cross_validation=False,
    )
    estimator._motion = _MotionSequence(observations)  # type: ignore[assignment]
    image = Image.fromarray(np.full((64, 64), 80, dtype=np.uint8))
    truth = [(0.0, 0.0, 10.0)]
    for feature in valid_features:
        delta = _mapped_delta(feature)
        truth.append(tuple(left + right for left, right in zip(truth[-1], delta, strict=True)))

    for index in range(9):
        estimator.estimate(image, _frame(index, truth[index], healthy=True))
    first = estimator.estimate(image, _frame(9, truth[-1], healthy=False))
    second = estimator.estimate(image, _frame(10, truth[-1], healthy=False))
    first_step = np.linalg.norm(np.asarray(first) - np.asarray(truth[-1]))
    second_step = np.linalg.norm(np.asarray(second) - np.asarray(first))

    assert 0 < second_step < first_step < 5.0
    assert all(math.isfinite(value) for value in second)
    assert estimator.diagnostics()["status"] == "velocity_fallback"


def test_unqualified_calibration_holds_instead_of_drifting() -> None:
    rng = np.random.default_rng(20260715)
    features = [tuple(float(value) for value in rng.normal(size=3)) for _ in range(55)]
    observations = [_observation(feature) for feature in features]
    estimator = CalibratedHomographySE3Estimator(
        1000.0,
        1000.0,
        10.0,
        min_calibration_samples=10,
        min_validation_samples=10,
        max_step_m=5.0,
    )
    estimator._motion = _MotionSequence(observations)  # type: ignore[assignment]
    image = Image.fromarray(np.full((64, 64), 80, dtype=np.uint8))
    truth = [(0.0, 0.0, 10.0)]
    for index in range(55):
        unrelated_delta = (
            0.7 + 0.1 * math.sin(index * 0.2),
            -0.4 + 0.1 * math.cos(index * 0.3),
            0.05,
        )
        truth.append(
            tuple(left + right for left, right in zip(truth[-1], unrelated_delta, strict=True))
        )

    for index in range(46):
        estimator.estimate(image, _frame(index, truth[index], healthy=True))
    held = estimator.estimate(image, _frame(46, truth[46], healthy=False))
    diagnostics = estimator.diagnostics()

    assert diagnostics["calibration"]["ready"] is True
    assert diagnostics["navigation_qualified"] is False
    assert held == truth[45]
    assert diagnostics["status"] == "hold:calibration_unqualified"


def test_experimental_flag_builds_new_estimator_without_changing_default() -> None:
    _, position, _ = build_vision_components(ClientSettings(enable_experimental_vo=True))

    assert isinstance(position, CalibratedHomographySE3Estimator)


def test_external_camera_mode_does_not_apply_registered_profile(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("registered camera profile must be bypassed")

    monkeypatch.setattr("hurgor.odometry.select_camera_profile", fail_if_called)
    estimator = CalibratedHomographySE3Estimator(
        893.39,
        898.33,
        10.0,
        principal_x=31.0,
        principal_y=23.0,
        use_registered_camera_profile=False,
    )
    image = Image.fromarray(np.full((48, 64), 80, dtype=np.uint8))

    estimated = estimator.estimate(image, _frame(0, (1.0, 2.0, 3.0), healthy=True))

    assert estimated == (1.0, 2.0, 3.0)


def test_video_csv_dropout_evaluation_passes_synthetic_thermal_gate(tmp_path) -> None:
    video, translation_csv, _ = generate(
        tmp_path,
        frames=90,
        fps=7.5,
        width=640,
        height=512,
        seed=20260713,
        modality="thermal",
    )
    args = build_parser().parse_args(
        [
            str(video),
            str(translation_csv),
            "--dropout-start",
            "40",
            "--dropout-end",
            "75",
            "--recovery-frames",
            "2",
            "--min-calibration-samples",
            "10",
        ]
    )

    report = evaluate(args)

    assert report["acceptance"]["passed"] is True
    assert report["candidate"]["mae_m"] < report["last_known_baseline"]["mae_m"]
    assert report["reanchor_max_error_m"] == 0.0
    assert report["camera_profile"]["modality"] == "thermal"
