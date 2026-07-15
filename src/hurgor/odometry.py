from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .camera import select_camera_profile
from .models import FrameMetadata

LOGGER = logging.getLogger("hurgor.odometry")


def _require_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("visual odometry requires OpenCV and NumPy") from exc
    return cv2, np


@dataclass(frozen=True, slots=True)
class MotionObservation:
    """Rotation-compensated image motion between two consecutive frames."""

    feature: tuple[float, ...]
    yaw_rad: float
    tracked_points: int
    inlier_points: int
    inlier_ratio: float
    reprojection_error_px: float
    valid: bool
    reason: str = "ok"

    @classmethod
    def invalid(cls, reason: str, *, tracked_points: int = 0) -> MotionObservation:
        return cls(
            feature=(0.0, 0.0, 0.0),
            yaw_rad=0.0,
            tracked_points=tracked_points,
            inlier_points=0,
            inlier_ratio=0.0,
            reprojection_error_px=math.inf,
            valid=False,
            reason=reason,
        )


@dataclass(slots=True)
class HomographyMotionEstimator:
    min_inliers: int = 24
    min_inlier_ratio: float = 0.45
    ransac_threshold_px: float = 2.5
    max_reprojection_error_px: float = 3.0
    max_corners: int = 1200
    forward_backward_threshold_px: float = 1.5
    projective_features: bool = False

    def observe(
        self,
        previous_gray: Any,
        current_gray: Any,
        principal_point: tuple[float, float],
    ) -> MotionObservation:
        cv2, np = _require_cv()
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        if points is None or len(points) < self.min_inliers:
            return MotionObservation.invalid("insufficient_features")

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        )
        tracked, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=criteria,
        )
        if tracked is None or forward_status is None:
            return MotionObservation.invalid("forward_flow_failed")
        backwards, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            tracked,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=criteria,
        )
        if backwards is None or backward_status is None:
            return MotionObservation.invalid("backward_flow_failed")

        old_all = points.reshape(-1, 2)
        new_all = tracked.reshape(-1, 2)
        back_all = backwards.reshape(-1, 2)
        valid = (forward_status.reshape(-1) == 1) & (backward_status.reshape(-1) == 1)
        valid &= np.linalg.norm(old_all - back_all, axis=1) <= self.forward_backward_threshold_px
        old_points = old_all[valid]
        new_points = new_all[valid]
        tracked_count = len(old_points)
        if tracked_count < self.min_inliers:
            return MotionObservation.invalid(
                "insufficient_bidirectional_tracks",
                tracked_points=tracked_count,
            )

        homography, mask = cv2.findHomography(
            old_points,
            new_points,
            cv2.RANSAC,
            self.ransac_threshold_px,
            maxIters=2000,
            confidence=0.995,
        )
        if homography is None or mask is None:
            return MotionObservation.invalid(
                "homography_failed",
                tracked_points=tracked_count,
            )
        inliers = mask.reshape(-1).astype(bool)
        old_inliers = old_points[inliers]
        new_inliers = new_points[inliers]
        inlier_count = len(old_inliers)
        inlier_ratio = inlier_count / tracked_count
        if inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return MotionObservation(
                feature=(0.0, 0.0, 0.0),
                yaw_rad=0.0,
                tracked_points=tracked_count,
                inlier_points=inlier_count,
                inlier_ratio=inlier_ratio,
                reprojection_error_px=math.inf,
                valid=False,
                reason="weak_homography_consensus",
            )

        projected = cv2.perspectiveTransform(old_inliers.reshape(-1, 1, 2), homography).reshape(
            -1, 2
        )
        errors = np.linalg.norm(projected - new_inliers, axis=1)
        reprojection_error = float(np.median(errors))
        if not math.isfinite(reprojection_error) or (
            reprojection_error > self.max_reprojection_error_px
        ):
            return MotionObservation(
                feature=(0.0, 0.0, 0.0),
                yaw_rad=0.0,
                tracked_points=tracked_count,
                inlier_points=inlier_count,
                inlier_ratio=inlier_ratio,
                reprojection_error_px=reprojection_error,
                valid=False,
                reason="high_reprojection_error",
            )

        affine, affine_mask = cv2.estimateAffinePartial2D(
            old_inliers,
            new_inliers,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold_px,
            maxIters=1500,
            confidence=0.995,
            refineIters=10,
        )
        if affine is None:
            return MotionObservation.invalid(
                "similarity_decomposition_failed",
                tracked_points=tracked_count,
            )
        if affine_mask is not None:
            affine_inliers = affine_mask.reshape(-1).astype(bool)
            if int(affine_inliers.sum()) >= self.min_inliers:
                old_inliers = old_inliers[affine_inliers]
                new_inliers = new_inliers[affine_inliers]

        linear = np.asarray(affine[:, :2], dtype=np.float64)
        determinant = float(np.linalg.det(linear))
        if not math.isfinite(determinant) or determinant <= 1e-8:
            return MotionObservation.invalid(
                "invalid_similarity_scale", tracked_points=tracked_count
            )
        scale = math.sqrt(determinant)
        rotation = linear / scale
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        if not (0.5 <= scale <= 2.0):
            return MotionObservation.invalid(
                "implausible_frame_scale", tracked_points=tracked_count
            )

        center = np.asarray(principal_point, dtype=np.float64)
        rotation_scale_only = (old_inliers - center) @ linear.T + center
        residual_flow = new_inliers - rotation_scale_only
        translation_px = np.median(residual_flow, axis=0)
        if self.projective_features:
            height, width = previous_gray.shape[:2]
            center_x, center_y = principal_point
            grid = np.asarray(
                [
                    (
                        center_x + x_offset * width,
                        center_y + y_offset * height,
                    )
                    for y_offset in (-0.30, 0.0, 0.30)
                    for x_offset in (-0.30, 0.0, 0.30)
                ],
                dtype=np.float32,
            )
            projected_grid = cv2.perspectiveTransform(
                grid.reshape(-1, 1, 2),
                homography,
            ).reshape(-1, 2)
            grid_flow = projected_grid - grid
            feature = tuple(float(value) for value in grid_flow.reshape(-1))
        else:
            feature = (
                float(translation_px[0]),
                float(translation_px[1]),
                float(math.log(scale)),
            )
        if not all(math.isfinite(value) for value in feature):
            return MotionObservation.invalid("non_finite_motion", tracked_points=tracked_count)
        return MotionObservation(
            feature=feature,
            yaw_rad=yaw,
            tracked_points=tracked_count,
            inlier_points=inlier_count,
            inlier_ratio=inlier_ratio,
            reprojection_error_px=reprojection_error,
            valid=True,
        )


@dataclass(slots=True)
class RobustMotionCalibration:
    min_samples: int = 20
    max_samples: int = 450
    ridge: float = 1e-3
    calibrate_gain: bool = False
    validation_fraction: float = 0.20
    min_validation_samples: int = 20
    max_step_skill_ratio: float = 0.85
    max_trajectory_skill_ratio: float = 0.75
    max_bias_ratio: float = 0.50
    _features: list[tuple[float, ...]] = field(default_factory=list)
    _targets: list[tuple[float, float, float]] = field(default_factory=list)
    _qualities: list[float] = field(default_factory=list)
    _headings: list[float] = field(default_factory=list)
    _feature_scale: Any | None = None
    _coefficients: Any | None = None
    fit_mae_m: float = math.inf
    feature_rank: int = 0
    sample_step_median: float = 0.0
    sample_step_mad: float = 0.0
    prediction_gain: float = 1.0
    validation_samples: int = 0
    validation_mae_m: float = math.inf
    validation_hold_mae_m: float = math.inf
    validation_trajectory_mae_m: float = math.inf
    validation_hold_trajectory_mae_m: float = math.inf
    validation_bias_m: float = math.inf

    @property
    def sample_count(self) -> int:
        return len(self._features)

    @property
    def ready(self) -> bool:
        return (
            self.sample_count >= self.min_samples
            and self.feature_rank >= 2
            and self._coefficients is not None
            and math.isfinite(self.fit_mae_m)
        )

    @property
    def navigation_ready(self) -> bool:
        if not self.ready or self.validation_samples < self.min_validation_samples:
            return False
        hold_step = max(self.validation_hold_mae_m, 1e-9)
        hold_trajectory = max(self.validation_hold_trajectory_mae_m, 1e-9)
        return (
            self.validation_mae_m / hold_step <= self.max_step_skill_ratio
            and self.validation_trajectory_mae_m / hold_trajectory
            <= self.max_trajectory_skill_ratio
            and self.validation_bias_m / hold_step <= self.max_bias_ratio
        )

    def add(
        self,
        feature: tuple[float, ...],
        target_delta: tuple[float, float, float],
        quality: float,
        heading_rad: float = 0.0,
    ) -> None:
        _, np = _require_cv()
        if not all(math.isfinite(value) for value in (*feature, *target_delta, heading_rad)):
            return
        if self._features and len(feature) != len(self._features[0]):
            return
        feature_norm = float(np.linalg.norm(np.asarray(feature, dtype=np.float64)))
        target_norm = float(np.linalg.norm(np.asarray(target_delta, dtype=np.float64)))
        if feature_norm < 1e-4 and target_norm < 1e-4:
            return
        self._features.append(feature)
        self._targets.append(target_delta)
        self._qualities.append(max(0.05, min(1.0, quality)))
        self._headings.append(heading_rad)
        if len(self._features) > self.max_samples:
            self._features.pop(0)
            self._targets.pop(0)
            self._qualities.pop(0)
            self._headings.pop(0)
        if self.sample_count >= max(6, self.min_samples // 2):
            self._fit()

    def _fit(self) -> None:
        _, np = _require_cv()
        features = np.asarray(self._features, dtype=np.float64)
        targets = np.asarray(self._targets, dtype=np.float64)
        qualities = np.asarray(self._qualities, dtype=np.float64)
        scales, coefficients, predictions, weights, gain = self._solve(
            features,
            targets,
            qualities,
        )
        residuals = np.linalg.norm(predictions - targets, axis=1)
        steps = np.linalg.norm(targets, axis=1)
        normalized = features / scales
        self.feature_rank = int(np.linalg.matrix_rank(normalized, tol=1e-3))
        self._feature_scale = scales
        self._coefficients = coefficients
        self.prediction_gain = gain
        self.fit_mae_m = float(np.mean(residuals))
        self.sample_step_median = float(np.median(steps))
        self.sample_step_mad = float(np.median(np.abs(steps - self.sample_step_median)))
        self._validate_chronologically(features, targets, qualities)

    def _solve(
        self,
        features: Any,
        targets: Any,
        qualities: Any,
    ) -> tuple[Any, Any, Any, Any, float]:
        _, np = _require_cv()
        scales = np.std(features, axis=0)
        floors = (
            np.asarray((0.05, 0.05, 1e-5), dtype=np.float64)
            if features.shape[1] == 3
            else np.full(features.shape[1], 0.05, dtype=np.float64)
        )
        scales = np.maximum(scales, floors)
        normalized = features / scales
        base_weights = np.asarray(qualities, dtype=np.float64)
        weights = base_weights.copy()
        coefficients = np.zeros((features.shape[1], 3), dtype=np.float64)
        identity = np.eye(features.shape[1], dtype=np.float64)
        for _ in range(6):
            weighted = normalized * weights[:, None]
            lhs = normalized.T @ weighted + self.ridge * identity
            rhs = normalized.T @ (targets * weights[:, None])
            try:
                coefficients = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                coefficients = np.linalg.pinv(lhs) @ rhs
            residuals = np.linalg.norm(normalized @ coefficients - targets, axis=1)
            median = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - median)))
            robust_scale = max(1e-4, 1.4826 * mad)
            huber_limit = 1.5 * robust_scale
            robust = np.ones_like(residuals)
            large = residuals > huber_limit
            robust[large] = huber_limit / np.maximum(residuals[large], 1e-9)
            weights = base_weights * robust

        predictions = normalized @ coefficients
        gain = 1.0
        if self.calibrate_gain:
            denominator = float(np.sum(weights[:, None] * predictions * predictions))
            if denominator > 1e-12:
                numerator = float(np.sum(weights[:, None] * predictions * targets))
                gain = max(0.5, min(2.0, numerator / denominator))
                predictions *= gain
        return scales, coefficients, predictions, weights, gain

    def _validate_chronologically(
        self,
        features: Any,
        targets: Any,
        qualities: Any,
    ) -> None:
        _, np = _require_cv()
        requested = max(
            self.min_validation_samples,
            int(round(len(features) * self.validation_fraction)),
        )
        validation_count = min(requested, len(features) - max(6, self.min_samples))
        if validation_count < self.min_validation_samples:
            self._clear_validation()
            return
        split = len(features) - validation_count
        train_features = features[:split]
        train_targets = targets[:split]
        train_qualities = qualities[:split]
        scales, coefficients, _, _, validation_gain = self._solve(
            train_features,
            train_targets,
            train_qualities,
        )
        predicted_local = (features[split:] / scales) @ coefficients * validation_gain
        target_local = targets[split:]
        headings = np.asarray(self._headings[split:], dtype=np.float64)
        predicted_world = self._rotate_xy(predicted_local, headings)
        target_world = self._rotate_xy(target_local, headings)
        residuals = predicted_world - target_world
        step_errors = np.linalg.norm(residuals, axis=1)
        hold_step_errors = np.linalg.norm(target_world, axis=1)
        trajectory_errors = np.linalg.norm(np.cumsum(residuals, axis=0), axis=1)
        hold_trajectory_errors = np.linalg.norm(np.cumsum(target_world, axis=0), axis=1)
        self.validation_samples = validation_count
        self.validation_mae_m = float(np.mean(step_errors))
        self.validation_hold_mae_m = float(np.mean(hold_step_errors))
        self.validation_trajectory_mae_m = float(np.mean(trajectory_errors))
        self.validation_hold_trajectory_mae_m = float(np.mean(hold_trajectory_errors))
        self.validation_bias_m = float(np.linalg.norm(np.mean(residuals, axis=0)))

    @staticmethod
    def _rotate_xy(vectors: Any, angles: Any) -> Any:
        _, np = _require_cv()
        result = np.asarray(vectors, dtype=np.float64).copy()
        cosine = np.cos(angles)
        sine = np.sin(angles)
        source_x = np.asarray(vectors, dtype=np.float64)[:, 0]
        source_y = np.asarray(vectors, dtype=np.float64)[:, 1]
        result[:, 0] = cosine * source_x - sine * source_y
        result[:, 1] = sine * source_x + cosine * source_y
        return result

    def _clear_validation(self) -> None:
        self.validation_samples = 0
        self.validation_mae_m = math.inf
        self.validation_hold_mae_m = math.inf
        self.validation_trajectory_mae_m = math.inf
        self.validation_hold_trajectory_mae_m = math.inf
        self.validation_bias_m = math.inf

    def predict(self, feature: tuple[float, ...]) -> tuple[float, float, float] | None:
        _, np = _require_cv()
        if not self.ready or self._feature_scale is None or self._coefficients is None:
            return None
        vector = np.asarray(feature, dtype=np.float64) / self._feature_scale
        predicted = (vector @ self._coefficients) * self.prediction_gain
        values = tuple(float(value) for value in predicted)
        return values if all(math.isfinite(value) for value in values) else None

    def reasonable_step_limit(self, hard_limit_m: float) -> float:
        if self.sample_count < self.min_samples:
            return hard_limit_m
        robust_limit = max(
            self.sample_step_median * 4.0,
            self.sample_step_median + 8.0 * max(self.sample_step_mad, 0.01),
        )
        return max(0.25, min(hard_limit_m, robust_limit))

    def diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "ready": self.ready,
            "navigation_ready": self.navigation_ready,
            "samples": self.sample_count,
            "feature_rank": self.feature_rank,
            "fit_mae_m": self.fit_mae_m if math.isfinite(self.fit_mae_m) else None,
            "step_median_m": self.sample_step_median,
            "step_mad_m": self.sample_step_mad,
            "prediction_gain": self.prediction_gain,
            "validation": {
                "samples": self.validation_samples,
                "mae_m": (self.validation_mae_m if math.isfinite(self.validation_mae_m) else None),
                "hold_mae_m": (
                    self.validation_hold_mae_m
                    if math.isfinite(self.validation_hold_mae_m)
                    else None
                ),
                "trajectory_mae_m": (
                    self.validation_trajectory_mae_m
                    if math.isfinite(self.validation_trajectory_mae_m)
                    else None
                ),
                "hold_trajectory_mae_m": (
                    self.validation_hold_trajectory_mae_m
                    if math.isfinite(self.validation_hold_trajectory_mae_m)
                    else None
                ),
                "bias_m": (
                    self.validation_bias_m if math.isfinite(self.validation_bias_m) else None
                ),
                "max_step_skill_ratio": self.max_step_skill_ratio,
                "max_trajectory_skill_ratio": self.max_trajectory_skill_ratio,
                "max_bias_ratio": self.max_bias_ratio,
            },
        }
        if self._feature_scale is not None and self._coefficients is not None:
            diagnostics["feature_scale"] = [float(value) for value in self._feature_scale]
            metric_mapping = (
                self._coefficients / self._feature_scale[:, None] * self.prediction_gain
            )
            diagnostics["metric_mapping"] = [
                [float(value) for value in row] for row in metric_mapping
            ]
        return diagnostics


@dataclass(slots=True)
class CalibratedHomographySE3Estimator:
    """GPS-supervised monocular odometry with guarded SE(3) accumulation.

    During healthy GPS frames the estimator learns camera-axis direction and metric scale
    directly from consecutive image motion and reference translation deltas. A chronological
    holdout must prove that the learned mapping beats a last-known-position baseline before it
    may navigate. During an outage, camera-local motion is rotated into the world frame using
    the accumulated image heading and then composed on SE(3).
    """

    fx: float
    fy: float
    default_altitude_m: float
    principal_x: float | None = None
    principal_y: float | None = None
    use_registered_camera_profile: bool = True
    min_calibration_samples: int = 20
    max_calibration_samples: int = 450
    calibration_ridge: float = 1e-3
    require_cross_validation: bool = True
    validation_fraction: float = 0.20
    min_validation_samples: int = 20
    max_step_skill_ratio: float = 0.85
    max_trajectory_skill_ratio: float = 0.75
    max_bias_ratio: float = 0.50
    min_inliers: int = 24
    min_inlier_ratio: float = 0.45
    ransac_threshold_px: float = 2.5
    max_reprojection_error_px: float = 3.0
    max_step_m: float = 20.0
    fallback_decay: float = 0.85
    projective_features: bool = False
    previous_gray: Any | None = None
    pose: Any | None = None
    last_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _stream_key: tuple[str, str] | None = None
    _previous_reference: tuple[float, float, float] | None = None
    _previous_was_healthy: bool = False
    _last_trusted_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _fallback_steps: int = 0
    _image_heading_rad: float = 0.0
    _previous_orientation_yaw_rad: float | None = None
    _orientation_telemetry_frames: int = 0
    _heading_source: str = "image"
    _last_status: str = "uninitialized"
    _reanchor_count: int = 0
    _rejected_motion_count: int = 0
    _calibration_ready_logged: bool = False
    _motion: HomographyMotionEstimator = field(init=False)
    _calibration: RobustMotionCalibration = field(init=False)

    def __post_init__(self) -> None:
        _, np = _require_cv()
        self.pose = np.eye(4, dtype=np.float64)
        self._motion = HomographyMotionEstimator(
            min_inliers=self.min_inliers,
            min_inlier_ratio=self.min_inlier_ratio,
            ransac_threshold_px=self.ransac_threshold_px,
            max_reprojection_error_px=self.max_reprojection_error_px,
            projective_features=self.projective_features,
        )
        self._calibration = RobustMotionCalibration(
            min_samples=self.min_calibration_samples,
            max_samples=self.max_calibration_samples,
            ridge=self.calibration_ridge,
            calibrate_gain=self.projective_features,
            validation_fraction=self.validation_fraction,
            min_validation_samples=self.min_validation_samples,
            max_step_skill_ratio=self.max_step_skill_ratio,
            max_trajectory_skill_ratio=self.max_trajectory_skill_ratio,
            max_bias_ratio=self.max_bias_ratio,
        )

    def estimate(self, image: Image.Image, frame: FrameMetadata) -> tuple[float, float, float]:
        _, np = _require_cv()
        stream_key = (frame.session, frame.video_name)
        if self._stream_key != stream_key:
            self._reset_stream(stream_key)

        current, principal = self._prepare_gray(image, frame)

        observation = None
        if self.previous_gray is not None:
            observation = self._motion.observe(self.previous_gray, current, principal)
        calibrated_feature = None
        if observation is not None and observation.valid:
            calibrated_feature = observation.feature
        telemetry_heading = frame.orientation_heading_rad
        if telemetry_heading is not None:
            self._advance_telemetry_orientation(telemetry_heading)
        else:
            self._previous_orientation_yaw_rad = None
            self._heading_source = "image"
            if observation is not None and observation.valid:
                self._advance_orientation(observation.yaw_rad)

        reference = frame.reference_translation
        healthy = frame.gps_health_status == 1 and reference is not None
        if healthy:
            assert reference is not None
            if self._previous_was_healthy and self._previous_reference is not None:
                target = tuple(
                    reference[index] - self._previous_reference[index] for index in range(3)
                )
                self._update_trusted_delta(target)
                if observation is not None and calibrated_feature is not None:
                    quality = self._observation_quality(observation)
                    local_target = self._world_to_heading_frame(target)
                    self._calibration.add(
                        calibrated_feature,
                        local_target,
                        quality,
                        heading_rad=self._image_heading_rad,
                    )
                    if self._calibration_is_qualified() and not self._calibration_ready_logged:
                        LOGGER.info(
                            "vo_navigation_qualified samples=%d fit_mae_m=%.4f rank=%d",
                            self._calibration.sample_count,
                            self._calibration.fit_mae_m,
                            self._calibration.feature_rank,
                        )
                        self._calibration_ready_logged = True
            if not self._previous_was_healthy and self.previous_gray is not None:
                self._reanchor_count += 1
            self._anchor(reference)
            self._previous_reference = reference
            self._previous_was_healthy = True
            self._fallback_steps = 0
            self._last_status = "gps_anchor"
            self.previous_gray = current
            return reference

        self._previous_reference = None
        self._previous_was_healthy = False
        delta = None
        calibration_qualified = self._calibration_is_qualified()
        if calibrated_feature is not None and calibration_qualified:
            local_delta = self._calibration.predict(calibrated_feature)
            if local_delta is not None:
                delta = self._heading_frame_to_world(local_delta)
            if delta is not None and not self._delta_is_safe(delta):
                delta = None
                self._rejected_motion_count += 1
                self._last_status = "visual_outlier_rejected"
        if delta is not None:
            self._apply_delta(delta)
            self._update_trusted_delta(delta, alpha=0.05)
            self._fallback_steps = 0
            self._last_status = "visual_odometry"
        else:
            fallback = self._controlled_fallback() if calibration_qualified else None
            if fallback is not None:
                self._apply_delta(fallback)
                self._last_status = "velocity_fallback"
            elif observation is None:
                self._last_status = "first_frame_hold"
            elif not observation.valid:
                self._last_status = f"hold:{observation.reason}"
            elif self._calibration.ready and not calibration_qualified:
                self._last_status = "hold:calibration_unqualified"
            else:
                self._last_status = "hold:calibration_not_ready"
        self.previous_gray = current
        return self.last_position

    def export_recovery_state(self) -> dict[str, Any]:
        """Return a compact, pickle-safe checkpoint for watchdog restarts.

        The previous image itself is deliberately excluded. The watchdog already owns
        the compressed bytes of the last successful frame and supplies them only after
        a worker restart, avoiding a full grayscale-frame copy on every IPC response.
        """

        _, np = _require_cv()
        pose = self.pose if self.pose is not None else np.eye(4, dtype=np.float64)
        return {
            "schema_version": 1,
            "stream_key": list(self._stream_key) if self._stream_key is not None else None,
            "pose": np.asarray(pose, dtype=np.float64).tolist(),
            "last_position": list(self.last_position),
            "previous_reference": (
                list(self._previous_reference) if self._previous_reference is not None else None
            ),
            "previous_was_healthy": self._previous_was_healthy,
            "last_trusted_delta": list(self._last_trusted_delta),
            "fallback_steps": self._fallback_steps,
            "image_heading_rad": self._image_heading_rad,
            "previous_orientation_yaw_rad": self._previous_orientation_yaw_rad,
            "orientation_telemetry_frames": self._orientation_telemetry_frames,
            "heading_source": self._heading_source,
            "last_status": self._last_status,
            "reanchor_count": self._reanchor_count,
            "rejected_motion_count": self._rejected_motion_count,
            "calibration_ready_logged": self._calibration_ready_logged,
            "calibration": {
                "features": [list(values) for values in self._calibration._features],
                "targets": [list(values) for values in self._calibration._targets],
                "qualities": list(self._calibration._qualities),
                "headings": list(self._calibration._headings),
            },
        }

    def restore_recovery_state(
        self,
        state: dict[str, Any],
        previous_image: Image.Image,
        previous_frame: FrameMetadata,
    ) -> None:
        """Restore a validated checkpoint and rebuild the prior optical-flow frame."""

        _, np = _require_cv()
        if state.get("schema_version") != 1:
            raise ValueError("unsupported odometry recovery state")
        raw_stream_key = state.get("stream_key")
        if not isinstance(raw_stream_key, list) or len(raw_stream_key) != 2:
            raise ValueError("invalid odometry recovery stream key")
        stream_key = (str(raw_stream_key[0]), str(raw_stream_key[1]))
        if stream_key != (previous_frame.session, previous_frame.video_name):
            raise ValueError("odometry recovery frame belongs to a different stream")

        pose = np.asarray(state.get("pose"), dtype=np.float64)
        if pose.shape != (4, 4) or not bool(np.all(np.isfinite(pose))):
            raise ValueError("invalid odometry recovery pose")

        def vector(name: str, *, optional: bool = False) -> tuple[float, float, float] | None:
            raw = state.get(name)
            if optional and raw is None:
                return None
            if not isinstance(raw, list) or len(raw) != 3:
                raise ValueError(f"invalid odometry recovery vector: {name}")
            values = tuple(float(value) for value in raw)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite odometry recovery vector: {name}")
            return values

        calibration_state = state.get("calibration")
        if not isinstance(calibration_state, dict):
            raise ValueError("missing odometry recovery calibration")
        raw_features = calibration_state.get("features")
        raw_targets = calibration_state.get("targets")
        raw_qualities = calibration_state.get("qualities")
        raw_headings = calibration_state.get("headings")
        if not all(
            isinstance(values, list)
            for values in (raw_features, raw_targets, raw_qualities, raw_headings)
        ):
            raise ValueError("invalid odometry recovery calibration arrays")
        sample_count = len(raw_features)
        if not (
            sample_count
            == len(raw_targets)
            == len(raw_qualities)
            == len(raw_headings)
            <= self.max_calibration_samples
        ):
            raise ValueError("inconsistent odometry recovery calibration lengths")

        features = [tuple(float(value) for value in row) for row in raw_features]
        targets = [tuple(float(value) for value in row) for row in raw_targets]
        qualities = [float(value) for value in raw_qualities]
        headings = [float(value) for value in raw_headings]
        feature_size = len(features[0]) if features else 3
        if feature_size not in {3, 18} or any(len(row) != feature_size for row in features):
            raise ValueError("invalid odometry recovery feature dimensions")
        if any(len(row) != 3 for row in targets) or not all(
            math.isfinite(value) for row in (*features, *targets) for value in row
        ):
            raise ValueError("invalid odometry recovery calibration values")
        if not all(math.isfinite(value) for value in (*qualities, *headings)):
            raise ValueError("non-finite odometry recovery calibration metadata")

        calibration = RobustMotionCalibration(
            min_samples=self.min_calibration_samples,
            max_samples=self.max_calibration_samples,
            ridge=self.calibration_ridge,
            calibrate_gain=self.projective_features,
            validation_fraction=self.validation_fraction,
            min_validation_samples=self.min_validation_samples,
            max_step_skill_ratio=self.max_step_skill_ratio,
            max_trajectory_skill_ratio=self.max_trajectory_skill_ratio,
            max_bias_ratio=self.max_bias_ratio,
        )
        calibration._features = features
        calibration._targets = targets
        calibration._qualities = qualities
        calibration._headings = headings
        if sample_count >= max(6, self.min_calibration_samples // 2):
            calibration._fit()

        last_position = vector("last_position")
        previous_reference = vector("previous_reference", optional=True)
        trusted_delta = vector("last_trusted_delta")
        assert last_position is not None and trusted_delta is not None
        image_heading = float(state.get("image_heading_rad", 0.0))
        if not math.isfinite(image_heading):
            raise ValueError("non-finite odometry recovery heading")
        raw_orientation_yaw = state.get("previous_orientation_yaw_rad")
        previous_orientation_yaw = (
            None if raw_orientation_yaw is None else float(raw_orientation_yaw)
        )
        if previous_orientation_yaw is not None and not math.isfinite(previous_orientation_yaw):
            raise ValueError("non-finite odometry recovery telemetry heading")

        self.pose = pose
        self.last_position = last_position
        self._stream_key = stream_key
        self._previous_reference = previous_reference
        self._previous_was_healthy = bool(state.get("previous_was_healthy", False))
        self._last_trusted_delta = trusted_delta
        self._fallback_steps = max(0, int(state.get("fallback_steps", 0)))
        self._image_heading_rad = image_heading
        self._previous_orientation_yaw_rad = previous_orientation_yaw
        self._orientation_telemetry_frames = max(
            0,
            int(state.get("orientation_telemetry_frames", 0)),
        )
        heading_source = str(state.get("heading_source", "image"))
        self._heading_source = (
            heading_source if heading_source in {"image", "telemetry"} else "image"
        )
        self._last_status = str(state.get("last_status", "worker_recovery"))
        self._reanchor_count = max(0, int(state.get("reanchor_count", 0)))
        self._rejected_motion_count = max(0, int(state.get("rejected_motion_count", 0)))
        self._calibration_ready_logged = bool(
            state.get("calibration_ready_logged", calibration.navigation_ready)
        )
        self._calibration = calibration
        self.previous_gray, _ = self._prepare_gray(previous_image, previous_frame)

    def _prepare_gray(
        self,
        image: Image.Image,
        frame: FrameMetadata,
    ) -> tuple[Any, tuple[float, float]]:
        cv2, np = _require_cv()
        rgb = np.asarray(image.convert("RGB"))
        current = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        principal = (
            self.principal_x if self.principal_x is not None else image.width / 2.0,
            self.principal_y if self.principal_y is not None else image.height / 2.0,
        )
        if self.use_registered_camera_profile:
            try:
                profile = select_camera_profile(
                    image.width,
                    image.height,
                    video_name=frame.video_name,
                )
                camera_matrix = np.asarray(profile.camera_matrix, dtype=np.float64)
                distortion = np.asarray(profile.distortion, dtype=np.float64)
                current = cv2.undistort(current, camera_matrix, distortion)
                principal = (profile.cx, profile.cy)
            except ValueError:
                pass
        current = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(current)
        return current, principal

    def _reset_stream(self, stream_key: tuple[str, str]) -> None:
        _, np = _require_cv()
        self.previous_gray = None
        self.pose = np.eye(4, dtype=np.float64)
        self.last_position = (0.0, 0.0, 0.0)
        self._stream_key = stream_key
        self._previous_reference = None
        self._previous_was_healthy = False
        self._last_trusted_delta = (0.0, 0.0, 0.0)
        self._fallback_steps = 0
        self._image_heading_rad = 0.0
        self._previous_orientation_yaw_rad = None
        self._orientation_telemetry_frames = 0
        self._heading_source = "image"
        self._last_status = "stream_reset"
        self._reanchor_count = 0
        self._rejected_motion_count = 0
        self._calibration_ready_logged = False
        self._calibration = RobustMotionCalibration(
            min_samples=self.min_calibration_samples,
            max_samples=self.max_calibration_samples,
            ridge=self.calibration_ridge,
            calibrate_gain=self.projective_features,
            validation_fraction=self.validation_fraction,
            min_validation_samples=self.min_validation_samples,
            max_step_skill_ratio=self.max_step_skill_ratio,
            max_trajectory_skill_ratio=self.max_trajectory_skill_ratio,
            max_bias_ratio=self.max_bias_ratio,
        )

    def _anchor(self, reference: tuple[float, float, float]) -> None:
        _, np = _require_cv()
        if self.pose is None:
            self.pose = np.eye(4, dtype=np.float64)
        self.pose[:3, 3] = np.asarray(reference, dtype=np.float64)
        self.last_position = reference

    def _advance_orientation(self, image_yaw_rad: float) -> None:
        cv2, np = _require_cv()
        if self.pose is None:
            self.pose = np.eye(4, dtype=np.float64)
        self._image_heading_rad = math.atan2(
            math.sin(self._image_heading_rad + image_yaw_rad),
            math.cos(self._image_heading_rad + image_yaw_rad),
        )
        rotation_delta, _ = cv2.Rodrigues(np.asarray((0.0, 0.0, -image_yaw_rad), dtype=np.float64))
        self.pose[:3, :3] = self.pose[:3, :3] @ rotation_delta

    def _advance_telemetry_orientation(self, orientation_yaw_rad: float) -> None:
        """Fuse optional quaternion yaw without making it an API requirement.

        The first telemetry sample is aligned to the estimator's current heading. If
        telemetry disappears, image yaw continues the trajectory; a later telemetry
        sample establishes a fresh alignment instead of applying the gap twice.
        """

        previous = self._previous_orientation_yaw_rad
        self._previous_orientation_yaw_rad = orientation_yaw_rad
        self._orientation_telemetry_frames += 1
        self._heading_source = "telemetry"
        if previous is None:
            return
        delta = math.atan2(
            math.sin(orientation_yaw_rad - previous),
            math.cos(orientation_yaw_rad - previous),
        )
        self._advance_orientation(delta)

    def _world_to_heading_frame(
        self,
        delta: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        x, y, z = delta
        cosine = math.cos(-self._image_heading_rad)
        sine = math.sin(-self._image_heading_rad)
        return (
            cosine * x - sine * y,
            sine * x + cosine * y,
            z,
        )

    def _heading_frame_to_world(
        self,
        delta: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        x, y, z = delta
        cosine = math.cos(self._image_heading_rad)
        sine = math.sin(self._image_heading_rad)
        return (
            cosine * x - sine * y,
            sine * x + cosine * y,
            z,
        )

    def _calibration_is_qualified(self) -> bool:
        return self._calibration.ready and (
            not self.require_cross_validation or self._calibration.navigation_ready
        )

    def _apply_delta(self, world_delta: tuple[float, float, float]) -> None:
        _, np = _require_cv()
        if self.pose is None:
            self.pose = np.eye(4, dtype=np.float64)
        rotation = self.pose[:3, :3]
        world = np.asarray(world_delta, dtype=np.float64)
        local_translation = rotation.T @ world
        delta_transform = np.eye(4, dtype=np.float64)
        delta_transform[:3, 3] = local_translation
        self.pose = self.pose @ delta_transform
        values = tuple(float(value) for value in self.pose[:3, 3])
        if all(math.isfinite(value) for value in values):
            self.last_position = values

    def _delta_is_safe(self, delta: tuple[float, float, float]) -> bool:
        _, np = _require_cv()
        norm = float(np.linalg.norm(np.asarray(delta, dtype=np.float64)))
        limit = self._calibration.reasonable_step_limit(self.max_step_m)
        return math.isfinite(norm) and norm <= limit

    def _controlled_fallback(self) -> tuple[float, float, float] | None:
        _, np = _require_cv()
        self._fallback_steps += 1
        velocity = np.asarray(self._last_trusted_delta, dtype=np.float64)
        if float(np.linalg.norm(velocity)) < 1e-5:
            return None
        decay = self.fallback_decay**self._fallback_steps
        candidate = tuple(float(value * decay) for value in velocity)
        return candidate if self._delta_is_safe(candidate) else None

    def _update_trusted_delta(
        self,
        delta: tuple[float, float, float],
        *,
        alpha: float = 0.25,
    ) -> None:
        self._last_trusted_delta = tuple(
            (1.0 - alpha) * previous + alpha * current
            for previous, current in zip(self._last_trusted_delta, delta, strict=True)
        )

    @staticmethod
    def _observation_quality(observation: MotionObservation) -> float:
        reprojection = math.exp(-max(0.0, observation.reprojection_error_px) / 2.0)
        return max(0.05, min(1.0, observation.inlier_ratio * reprojection))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": self._last_status,
            "position": self.last_position,
            "fallback_steps": self._fallback_steps,
            "image_heading_deg": math.degrees(self._image_heading_rad),
            "heading_source": self._heading_source,
            "orientation_telemetry_frames": self._orientation_telemetry_frames,
            "reanchors": self._reanchor_count,
            "rejected_motion": self._rejected_motion_count,
            "cross_validation_required": self.require_cross_validation,
            "navigation_qualified": self._calibration_is_qualified(),
            "calibration": self._calibration.diagnostics(),
        }
