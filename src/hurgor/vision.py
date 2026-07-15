from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .config import ClientSettings
from .detector_calibration import load_detector_thresholds
from .inference import (
    LastKnownPositionEstimator,
    NoopObjectDetector,
    NoopUndefinedObjectMatcher,
    ObjectDetector,
    PositionEstimator,
    UndefinedObjectMatcher,
)
from .modality import frame_modality
from .models import DetectedObject, DetectedUndefinedObject, FrameMetadata
from .odometry import CalibratedHomographySE3Estimator

LOGGER = logging.getLogger("hurgor.vision")


def _require_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "AI görüntü modülleri için `pip install -e '.[ai]'` çalıştırılmalıdır"
        ) from exc
    return cv2, np


def _pil_to_bgr(image: Image.Image) -> Any:
    cv2, np = _require_cv()
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@dataclass(slots=True)
class TopologicalNoiseFilter:
    """Lightweight persistence-inspired noise detector across intensity thresholds.

    This is not a full persistent-homology library. It measures short-lived connected
    components over several thresholds and only filters when their density is high.
    """

    component_ratio_threshold: float = 0.015
    max_analysis_side: int = 320

    def apply(self, image: Image.Image) -> Image.Image:
        cv2, np = _require_cv()
        bgr = _pil_to_bgr(image)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        longest_side = max(gray.shape)
        if longest_side > self.max_analysis_side:
            scale = self.max_analysis_side / longest_side
            gray = cv2.resize(
                gray,
                (
                    max(1, int(round(gray.shape[1] * scale))),
                    max(1, int(round(gray.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        tiny_components = 0
        total_components = 0
        tiny_limit = max(4, int(gray.size * 0.0001))
        for threshold in (48, 96, 144, 192):
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            if count <= 1:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA]
            tiny_components += int(np.count_nonzero(areas <= tiny_limit))
            total_components += int(areas.size)
        ratio = tiny_components / max(total_components, 1)
        if total_components > 30 and ratio >= self.component_ratio_threshold:
            LOGGER.info(
                "tda_preprocess applied=true component_ratio=%.4f components=%d",
                ratio,
                total_components,
            )
            filtered = cv2.medianBlur(bgr, 3)
            return Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        return image


@dataclass(slots=True)
class FrustumProjector:
    fx: float
    fy: float
    altitude_m: float

    def aabb(
        self,
        detection: DetectedObject,
        image_size: tuple[int, int],
        object_height_m: float,
        altitude_m: float | None = None,
    ) -> tuple[Any, Any]:
        _, np = _require_cv()
        width, height = image_size
        cx, cy = width / 2, height / 2
        altitude = altitude_m if altitude_m is not None and altitude_m > 0 else self.altitude_m
        near = max(0.05, altitude - object_height_m)
        far = altitude + 0.15
        pixels = (
            (detection.top_left_x, detection.top_left_y),
            (detection.bottom_right_x, detection.top_left_y),
            (detection.bottom_right_x, detection.bottom_right_y),
            (detection.top_left_x, detection.bottom_right_y),
        )
        points = []
        for depth in (near, far):
            for x, y in pixels:
                points.append(((x - cx) * depth / self.fx, (y - cy) * depth / self.fy, depth))
        array = np.asarray(points, dtype=np.float64)
        return array.min(axis=0), array.max(axis=0)

    @staticmethod
    def iou3d(first: tuple[Any, Any], second: tuple[Any, Any]) -> float:
        _, np = _require_cv()
        first_min, first_max = first
        second_min, second_max = second
        overlap = np.maximum(
            0.0, np.minimum(first_max, second_max) - np.maximum(first_min, second_min)
        )
        intersection = float(np.prod(overlap))
        first_volume = float(np.prod(np.maximum(0.0, first_max - first_min)))
        second_volume = float(np.prod(np.maximum(0.0, second_max - second_min)))
        union = first_volume + second_volume - intersection
        return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class OptimizedObjectDetector:
    detector: ObjectDetector
    noise_filter: TopologicalNoiseFilter
    projector: FrustumProjector
    motion_classifier: MotionCompensatedMotionClassifier = field(
        default_factory=lambda: MotionCompensatedMotionClassifier()
    )
    duplicate_iou_threshold: float = 0.45
    cross_class_duplicate_iou_threshold: float = 0.90
    landing_boundary_margin_px: float = 1.0
    _last_frame_digest: bytes | None = field(default=None, init=False, repr=False)
    _last_video_name: str | None = field(default=None, init=False, repr=False)

    def warmup(self) -> None:
        width = int(getattr(self.detector, "input_width", 640))
        height = int(getattr(self.detector, "input_height", 640))
        self.noise_filter.apply(Image.new("RGB", (width, height)))
        warmup = getattr(self.detector, "warmup", None)
        if warmup is not None:
            warmup()

    def health(self) -> dict[str, object]:
        health = getattr(self.detector, "health", None)
        if health is None:
            return {"ok": True, "wrapper": "optimized"}
        payload = dict(health())
        payload["wrapper"] = "optimized"
        return payload

    def close(self) -> None:
        close = getattr(self.detector, "close", None)
        if close is not None:
            close()

    def model_info(self) -> dict[str, object]:
        model_info = getattr(self.detector, "model_info", None)
        if model_info is None:
            return {"type": "unknown", "wrapper": "optimized"}
        payload = dict(model_info())
        payload["wrapper"] = "optimized"
        return payload

    def detect(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]:
        frozen_frame = self._is_frozen_frame(image, frame.video_name)
        filtered = self.noise_filter.apply(image)
        detections = _deduplicate_detections(
            self.detector.detect(filtered, frame),
            self.duplicate_iou_threshold,
            self.cross_class_duplicate_iou_threshold,
        )
        detections = self._landing_status(detections, image.size, frame)
        return self.motion_classifier.update(
            detections,
            frame.video_name,
            frozen_frame=frozen_frame,
        )

    def detect_fast(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]:
        # Degraded mode bypasses topology filtering, but keeps the light detector.
        frozen_frame = self._is_frozen_frame(image, frame.video_name)
        fast_detect = getattr(self.detector, "detect_fast", self.detector.detect)
        detections = _deduplicate_detections(
            fast_detect(image, frame),
            self.duplicate_iou_threshold,
            self.cross_class_duplicate_iou_threshold,
        )
        detections = self._landing_status(detections, image.size, frame)
        return self.motion_classifier.update(
            detections,
            frame.video_name,
            frozen_frame=frozen_frame,
        )

    def _is_frozen_frame(self, image: Image.Image, video_name: str) -> bool:
        if self._last_video_name != video_name:
            self._last_video_name = video_name
            self._last_frame_digest = None
        thumbnail = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        digest = hashlib.blake2b(thumbnail.tobytes(), digest_size=16).digest()
        frozen = self._last_frame_digest == digest
        self._last_frame_digest = digest
        return frozen

    def apply_unknown_obstacles(
        self,
        detections: list[DetectedObject],
        undefined_objects: list[DetectedUndefinedObject],
        image_size: tuple[int, int],
    ) -> list[DetectedObject]:
        """Apply reference/unknown obstacles after the matcher has produced its boxes."""

        output: list[DetectedObject] = []
        for item in detections:
            if item.class_id not in {"2", "3"}:
                output.append(item)
                continue
            blocked = item.landing_status == "0" or _touches_image_boundary(
                item,
                image_size,
                self.landing_boundary_margin_px,
            )
            if not blocked:
                blocked = any(
                    _bbox_intersection_area(item, obstacle) > 0
                    for obstacle in undefined_objects
                )
            output.append(item.model_copy(update={"landing_status": "0" if blocked else "1"}))
        return output

    def _landing_status(
        self,
        detections: list[DetectedObject],
        image_size: tuple[int, int],
        frame: FrameMetadata,
    ) -> list[DetectedObject]:
        reference = frame.reference_translation
        if reference is not None and math.isfinite(reference[2]) and abs(reference[2]) > 0.01:
            self.projector.altitude_m = abs(reference[2])
        obstacles = [item for item in detections if item.class_id in {"0", "1"}]
        obstacle_frustums = [
            self.projector.aabb(item, image_size, 1.7 if item.class_id == "1" else 1.5)
            for item in obstacles
        ]
        output: list[DetectedObject] = []
        for item in detections:
            if item.class_id not in {"2", "3"}:
                output.append(item)
                continue
            landing_frustum = self.projector.aabb(item, image_size, 0.30)
            blocked = _touches_image_boundary(
                item,
                image_size,
                self.landing_boundary_margin_px,
            )
            blocked = blocked or any(
                _bbox_intersection_area(item, obstacle_box) > 0
                or self.projector.iou3d(landing_frustum, obstacle_frustum) > 0.0001
                for obstacle_box, obstacle_frustum in zip(
                    obstacles,
                    obstacle_frustums,
                    strict=True,
                )
            )
            output.append(item.model_copy(update={"landing_status": "0" if blocked else "1"}))
        return output


@dataclass(slots=True)
class ThermalHumanFusionDetector:
    """Use a thermal specialist for humans while retaining the main model elsewhere.

    RGB frames always use the main four-class detector. On thermal frames, main-model
    human boxes are replaced by specialist human boxes; vehicle, UAP and UAI boxes
    continue to come from the main detector. A specialist failure fails open to the
    complete main-model result so one bad model call can never block the POST cycle.
    """

    main_detector: ObjectDetector
    thermal_specialist: ObjectDetector
    specialist_timeout_ms: float = 200.0
    slow_threshold_ms: float = 180.0
    cooldown_frames: int = 30
    cooldown_seconds: float = 20.0
    _specialist_available: bool = field(default=True, init=False, repr=False)
    _cooldown_remaining: int = field(default=0, init=False, repr=False)
    _cooldown_until: float = field(default=0.0, init=False, repr=False)
    _cooldown_announced: bool = field(default=False, init=False, repr=False)
    _specialist_future: Future[list[DetectedObject]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _specialist_executor: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._specialist_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="thermal-specialist",
        )

    @property
    def input_width(self) -> int:
        return int(getattr(self.main_detector, "input_width", 640))

    @property
    def input_height(self) -> int:
        return int(getattr(self.main_detector, "input_height", 640))

    def warmup(self) -> None:
        self.main_detector.warmup()
        try:
            self._specialist_executor.submit(self.thermal_specialist.warmup).result()
        except Exception:
            self._specialist_available = False
            LOGGER.exception("thermal_specialist_warmup_failed fallback=main")

    def health(self) -> dict[str, object]:
        main = dict(self.main_detector.health())
        if self._specialist_available:
            try:
                specialist = dict(self.thermal_specialist.health())
            except Exception as exc:
                specialist = {"ok": False, "active": True, "error": str(exc)}
        else:
            specialist = {"ok": False, "active": False, "fallback": "main"}
        return {
            # The specialist is an optional accuracy enhancement. The complete main
            # model remains a valid operational fallback.
            "ok": bool(main.get("ok", True)),
            "mode": "thermal_human_fusion",
            "specialist_cooldown_remaining": self._cooldown_remaining,
            "specialist_cooldown_seconds_remaining": max(
                0.0,
                self._cooldown_until - time.monotonic(),
            ),
            "main": main,
            "thermal_specialist": specialist,
        }

    def close(self) -> None:
        self._specialist_executor.shutdown(wait=True, cancel_futures=True)
        try:
            self.main_detector.close()
        finally:
            self.thermal_specialist.close()

    def model_info(self) -> dict[str, object]:
        return {
            "type": "thermal_human_fusion",
            "specialist_available": self._specialist_available,
            "load_shedding": {
                "specialist_timeout_ms": self.specialist_timeout_ms,
                "slow_threshold_ms": self.slow_threshold_ms,
                "cooldown_frames": self.cooldown_frames,
                "cooldown_seconds": self.cooldown_seconds,
                "cooldown_remaining": self._cooldown_remaining,
            },
            "policy": {
                "rgb": "main_all_classes",
                "thermal": {
                    "main": ["arac", "uap", "uai"],
                    "specialist": ["insan"],
                },
                "degraded": "main_all_classes",
                "specialist_failure": "main_all_classes",
            },
            "main": self.main_detector.model_info(),
            "thermal_specialist": self.thermal_specialist.model_info(),
        }

    def detect(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]:
        if frame_modality(frame) != "thermal" or not self._specialist_available:
            return self.main_detector.detect(image, frame)
        self._drain_late_specialist_result()
        main = self.main_detector.detect(image, frame)
        if self._cooldown_active():
            return main
        specialist_started = time.perf_counter()
        self._specialist_future = self._specialist_executor.submit(
            self.thermal_specialist.detect,
            image,
            frame,
        )
        try:
            specialist = self._specialist_future.result(
                timeout=self.specialist_timeout_ms / 1000,
            )
            self._specialist_future = None
        except FuturesTimeoutError:
            elapsed_ms = (time.perf_counter() - specialist_started) * 1000
            self._start_cooldown("timeout", elapsed_ms)
            return main
        except Exception:
            self._specialist_future = None
            self._specialist_available = False
            LOGGER.exception(
                "thermal_specialist_failed disabled=true fallback=main frame=%s",
                frame.url,
            )
            return main
        elapsed_ms = (time.perf_counter() - specialist_started) * 1000
        if elapsed_ms >= self.slow_threshold_ms:
            self._start_cooldown("slow", elapsed_ms)
        return [item for item in main if item.class_id != "1"] + [
            item for item in specialist if item.class_id == "1"
        ]

    def detect_fast(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]:
        """SLA protection: bypass the second model while degraded."""

        return self.main_detector.detect(image, frame)

    def _cooldown_active(self) -> bool:
        frame_cooldown = self._cooldown_remaining > 0
        if frame_cooldown:
            self._cooldown_remaining -= 1
        time_cooldown = time.monotonic() < self._cooldown_until
        future_running = (
            self._specialist_future is not None and not self._specialist_future.done()
        )
        active = frame_cooldown or time_cooldown or future_running
        if not active and self._cooldown_announced:
            self._cooldown_announced = False
            LOGGER.info("thermal_specialist_cooldown complete=true")
        return active

    def _start_cooldown(self, reason: str, elapsed_ms: float) -> None:
        self._cooldown_remaining = max(self._cooldown_remaining, self.cooldown_frames)
        self._cooldown_until = max(
            self._cooldown_until,
            time.monotonic() + self.cooldown_seconds,
        )
        self._cooldown_announced = True
        LOGGER.warning(
            "thermal_specialist_cooldown triggered=true reason=%s elapsed_ms=%.3f "
            "frames=%d seconds=%.3f",
            reason,
            elapsed_ms,
            self.cooldown_frames,
            self.cooldown_seconds,
        )

    def _drain_late_specialist_result(self) -> None:
        future = self._specialist_future
        if future is None or not future.done():
            return
        try:
            future.result()
        except Exception as exc:
            self._specialist_available = False
            LOGGER.error(
                "thermal_specialist_late_failure disabled=true fallback=main error=%r",
                exc,
            )
        finally:
            self._specialist_future = None


@dataclass(slots=True)
class _MotionTrack:
    track_id: int
    detection: DetectedObject
    moving_streak: int = 0
    stationary_streak: int = 0
    status: str = "-1"


@dataclass(slots=True)
class MotionCompensatedMotionClassifier:
    """Bounded vehicle motion state with robust camera-motion compensation."""

    residual_threshold: float = 0.08
    hysteresis_frames: int = 2
    max_tracks: int = 256
    _tracks: list[_MotionTrack] = field(default_factory=list, init=False)
    _next_track_id: int = field(default=1, init=False)
    _video_name: str | None = field(default=None, init=False)

    def update(
        self,
        detections: list[DetectedObject],
        video_name: str,
        *,
        frozen_frame: bool = False,
    ) -> list[DetectedObject]:
        if self._video_name != video_name:
            self._tracks.clear()
            self._video_name = video_name
        matches: list[
            tuple[DetectedObject, _MotionTrack, float, float, float, float]
        ] = []
        unmatched_tracks = list(self._tracks)
        for detection in detections:
            candidates = [
                track
                for track in unmatched_tracks
                if track.detection.class_id == detection.class_id
            ]
            track = max(
                candidates, key=lambda item: _bbox_iou(item.detection, detection), default=None
            )
            if track is None or _bbox_iou(track.detection, detection) < 0.05:
                continue
            old_x, old_y = _bbox_center(track.detection)
            new_x, new_y = _bbox_center(detection)
            matches.append((detection, track, old_x, old_y, new_x, new_y))
            unmatched_tracks.remove(track)

        camera_transform = _estimate_camera_transform(matches)
        updated: list[DetectedObject] = []
        next_tracks: list[_MotionTrack] = []
        matched_ids: set[int] = set()
        for detection, track, old_x, old_y, new_x, new_y in matches:
            matched_ids.add(id(detection))
            status = "-1"
            if (
                detection.class_id == "0"
                and not frozen_frame
                and camera_transform is not None
            ):
                width = detection.bottom_right_x - detection.top_left_x
                height = detection.bottom_right_y - detection.top_left_y
                scale = max(math.hypot(width, height), 1.0)
                predicted_x = (
                    float(camera_transform[0, 0]) * old_x
                    + float(camera_transform[0, 1]) * old_y
                    + float(camera_transform[0, 2])
                )
                predicted_y = (
                    float(camera_transform[1, 0]) * old_x
                    + float(camera_transform[1, 1]) * old_y
                    + float(camera_transform[1, 2])
                )
                residual = math.hypot(new_x - predicted_x, new_y - predicted_y) / scale
                moving = residual > self.residual_threshold
                track.moving_streak = track.moving_streak + 1 if moving else 0
                track.stationary_streak = track.stationary_streak + 1 if not moving else 0
                if track.moving_streak >= self.hysteresis_frames:
                    track.status = "1"
                elif track.stationary_streak >= self.hysteresis_frames:
                    track.status = "0"
                status = track.status
            elif detection.class_id == "0":
                # One vehicle alone cannot disambiguate target motion from camera motion.
                track.moving_streak = 0
                track.stationary_streak = 0
                track.status = "-1"
            track.detection = detection.model_copy(update={"motion_status": status})
            updated.append(track.detection)
            next_tracks.append(track)

        for detection in detections:
            if id(detection) in matched_ids:
                continue
            track = _MotionTrack(self._next_track_id, detection)
            self._next_track_id += 1
            updated.append(detection.model_copy(update={"motion_status": "-1"}))
            next_tracks.append(track)
        self._tracks = next_tracks[-self.max_tracks :]
        return updated


def _bbox_center(item: DetectedObject) -> tuple[float, float]:
    return (
        (item.top_left_x + item.bottom_right_x) / 2,
        (item.top_left_y + item.bottom_right_y) / 2,
    )


def _bbox_iou(first: DetectedObject, second: DetectedObject) -> float:
    x1 = max(first.top_left_x, second.top_left_x)
    y1 = max(first.top_left_y, second.top_left_y)
    x2 = min(first.bottom_right_x, second.bottom_right_x)
    y2 = min(first.bottom_right_y, second.bottom_right_y)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first.bottom_right_x - first.top_left_x) * (
        first.bottom_right_y - first.top_left_y
    )
    second_area = (second.bottom_right_x - second.top_left_x) * (
        second.bottom_right_y - second.top_left_y
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_intersection_area(first: Any, second: Any) -> float:
    width = max(
        0.0,
        min(float(first.bottom_right_x), float(second.bottom_right_x))
        - max(float(first.top_left_x), float(second.top_left_x)),
    )
    height = max(
        0.0,
        min(float(first.bottom_right_y), float(second.bottom_right_y))
        - max(float(first.top_left_y), float(second.top_left_y)),
    )
    return width * height


def _touches_image_boundary(
    item: DetectedObject,
    image_size: tuple[int, int],
    margin_px: float,
) -> bool:
    width, height = image_size
    margin = max(0.0, margin_px)
    return (
        item.top_left_x <= margin
        or item.top_left_y <= margin
        or item.bottom_right_x >= width - margin
        or item.bottom_right_y >= height - margin
    )


def _bbox_overlap_over_smaller(first: DetectedObject, second: DetectedObject) -> float:
    intersection = _bbox_intersection_area(first, second)
    first_area = (first.bottom_right_x - first.top_left_x) * (
        first.bottom_right_y - first.top_left_y
    )
    second_area = (second.bottom_right_x - second.top_left_x) * (
        second.bottom_right_y - second.top_left_y
    )
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller > 0 else 0.0


def _deduplicate_detections(
    detections: list[DetectedObject],
    same_class_iou: float,
    cross_class_iou: float,
) -> list[DetectedObject]:
    output: list[DetectedObject] = []
    for detection in detections:
        duplicate = False
        for kept in output:
            iou = _bbox_iou(detection, kept)
            if detection.class_id == kept.class_id:
                duplicate = iou >= same_class_iou or _bbox_overlap_over_smaller(
                    detection, kept
                ) >= 0.90
            else:
                duplicate = iou >= cross_class_iou
            if duplicate:
                break
        if not duplicate:
            output.append(detection)
    return output


def _estimate_camera_transform(
    matches: list[tuple[DetectedObject, _MotionTrack, float, float, float, float]],
) -> Any | None:
    if not matches:
        return None
    stable_landmarks = [item for item in matches if item[0].class_id in {"2", "3"}]
    non_vehicle_landmarks = [item for item in matches if item[0].class_id != "0"]
    if len(stable_landmarks) >= 3:
        landmarks = stable_landmarks
    elif len(non_vehicle_landmarks) >= 3:
        landmarks = non_vehicle_landmarks
    elif len(stable_landmarks) >= 2:
        landmarks = stable_landmarks
    else:
        landmarks = non_vehicle_landmarks
    if len(landmarks) >= 3:
        cv2, np = _require_cv()
        source = np.asarray([(item[2], item[3]) for item in landmarks], dtype=np.float32)
        target = np.asarray([(item[4], item[5]) for item in landmarks], dtype=np.float32)
        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is not None and inliers is not None:
            inlier_count = int(inliers.sum())
            if inlier_count >= 2 and inlier_count / len(landmarks) >= 0.5:
                return affine
    if len(landmarks) >= 2:
        _, np = _require_cv()
        dx = statistics.median(item[4] - item[2] for item in landmarks)
        dy = statistics.median(item[5] - item[3] for item in landmarks)
        return np.asarray(((1.0, 0.0, dx), (0.0, 1.0, dy)), dtype=np.float64)
    if len(matches) >= 3:
        _, np = _require_cv()
        dx = statistics.median(item[4] - item[2] for item in matches)
        dy = statistics.median(item[5] - item[3] for item in matches)
        return np.asarray(((1.0, 0.0, dx), (0.0, 1.0, dy)), dtype=np.float64)
    return None


def _box_iou_xywh(first: list[float], second: list[float]) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    x1 = max(first_x, second_x)
    y1 = max(first_y, second_y)
    x2 = min(first_x + first_width, second_x + second_width)
    y2 = min(first_y + first_height, second_y + second_height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


def _select_nms_indices(
    boxes: list[list[float]],
    scores: list[float],
    classes: list[int],
    *,
    same_class_iou: float,
    cross_class_iou: float,
) -> list[int]:
    """Score-ordered NMS plus a conservative cross-class duplicate guard."""

    selected: list[int] = []
    for index in sorted(range(len(boxes)), key=lambda item: scores[item], reverse=True):
        duplicate = False
        for kept_index in selected:
            iou = _box_iou_xywh(boxes[index], boxes[kept_index])
            threshold = same_class_iou if classes[index] == classes[kept_index] else cross_class_iou
            if iou >= threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append(index)
    return selected


class ONNXYoloDetector:
    """Manifest-driven YOLO ONNX adapter with explicit output decoding."""

    def __init__(
        self,
        model_path: str,
        *,
        base_url: str = "http://127.0.0.1:5000",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        num_classes: int | None = None,
        expected_classes: tuple[str, ...] = ("arac", "insan", "uap", "uai"),
        manifest_path: str | None = None,
        thresholds_path: str | None = None,
        cross_class_iou_threshold: float = 0.90,
        providers: tuple[str, ...] = (),
        intra_op_threads: int = 0,
        inter_op_threads: int = 1,
    ) -> None:
        cv2, np = _require_cv()
        del cv2, np
        path = Path(model_path).expanduser().resolve()
        if path.suffix.lower() in {".pt", ".pth", ".h5"}:
            raise ValueError("runtime .pt/.h5 kabul etmez; modeli ONNX'e export edin")
        if path.suffix.lower() != ".onnx":
            raise ValueError("YOLO runtime modeli .onnx olmalıdır")
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX modeli için `pip install -e '.[ai]'` gerekir") from exc

        available = ort.get_available_providers()
        requested = providers or (
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        )
        preferred = [provider for provider in requested if provider in available]
        if not preferred:
            raise RuntimeError(f"none of the requested ONNX providers are available: {requested}")
        provider_options: list[dict[str, Any]] = []
        for provider in preferred:
            if provider == "TensorrtExecutionProvider":
                provider_options.append(
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(path.parent / ".trt_cache"),
                        "trt_fp16_enable": True,
                    }
                )
            else:
                provider_options.append({})
        session_options = ort.SessionOptions()
        if intra_op_threads > 0:
            session_options.intra_op_num_threads = intra_op_threads
        session_options.inter_op_num_threads = max(1, inter_op_threads)
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=preferred,
            provider_options=provider_options,
        )
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        self.input_height = int(shape[2]) if isinstance(shape[2], int) else 640
        self.input_width = int(shape[3]) if isinstance(shape[3], int) else 640
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.cross_class_iou_threshold = cross_class_iou_threshold
        if not expected_classes or len(set(expected_classes)) != len(expected_classes):
            raise ValueError("expected model classes must be non-empty and unique")
        if num_classes is not None and num_classes != len(expected_classes):
            raise ValueError("num_classes must match expected_classes")
        self.num_classes = len(expected_classes)
        self.base_url = base_url
        self.model_path = path
        self.model_sha256 = _file_sha256(path)
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = max(1, inter_op_threads)
        self.output_format = "yolo_one_to_many"
        self.class_names = list(expected_classes)
        self.thresholds_path: Path | None = None
        self.threshold_profile: dict[str, Any] | None = None
        if manifest_path:
            manifest = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
            self.output_format = str(manifest.get("output_format", ""))
            manifest_classes = list(manifest.get("classes", []))
            if self.output_format not in {"yolo_one_to_many", "yolo_end2end"}:
                raise ValueError("manifest output_format must be yolo_one_to_many or yolo_end2end")
            if manifest_classes != self.class_names:
                raise ValueError(
                    f"model class order mismatch: expected {self.class_names}, "
                    f"got {manifest_classes}"
                )
            expected_sha = str(manifest.get("sha256", "")).lower()
            if expected_sha and expected_sha != self.model_sha256:
                raise ValueError("model checksum does not match manifest")
        self.class_confidences = {name: confidence for name in self.class_names}
        if thresholds_path:
            thresholds, profile = load_detector_thresholds(
                thresholds_path,
                runtime_model_sha256=self.model_sha256,
                class_names=self.class_names,
            )
            self.class_confidences = thresholds
            self.threshold_profile = profile
            self.thresholds_path = Path(thresholds_path).expanduser().resolve()
        LOGGER.info(
            "yolo_backend providers=%s intra_threads=%d inter_threads=%d model=%s",
            self.session.get_providers(),
            self.intra_op_threads,
            self.inter_op_threads,
            path,
        )

    def warmup(self) -> None:
        _, np = _require_cv()
        tensor = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
        self.session.run(None, {self.input.name: tensor})
        warmup_frame = FrameMetadata.model_validate(
            {
                "url": "http://warmup.invalid/frames/0/",
                "image_url": "/warmup.jpg",
                "video_name": "warmup",
                "session": "http://warmup.invalid/session/0/",
                "translation_x": 0.0,
                "translation_y": 0.0,
                "translation_z": 10.0,
                "gps_health_status": 1,
            }
        )
        self.detect(
            Image.new("RGB", (self.input_width, self.input_height), (114, 114, 114)),
            warmup_frame,
        )

    def health(self) -> dict[str, Any]:
        return {"ok": True, "providers": self.session.get_providers()}

    def close(self) -> None:
        self.session = None

    def model_info(self) -> dict[str, Any]:
        return {
            "path": str(self.model_path),
            "sha256": self.model_sha256,
            "classes": self.class_names,
            "output_format": self.output_format,
            "input_size": [self.input_width, self.input_height],
            "providers": self.session.get_providers(),
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": self.inter_op_threads,
            "class_confidences": self.class_confidences,
            "thresholds_path": str(self.thresholds_path) if self.thresholds_path else None,
        }

    def detect(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]:
        cv2, np = _require_cv()
        del frame
        bgr = _pil_to_bgr(image)
        original_height, original_width = bgr.shape[:2]
        scale = min(self.input_width / original_width, self.input_height / original_height)
        resized_width = int(round(original_width * scale))
        resized_height = int(round(original_height * scale))
        resized = cv2.resize(bgr, (resized_width, resized_height))
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        tensor = cv2.dnn.blobFromImage(
            canvas, 1 / 255.0, (self.input_width, self.input_height), swapRB=True
        )
        raw = self.session.run(None, {self.input.name: tensor})[0]
        predictions = np.squeeze(raw)
        if predictions.ndim != 2:
            raise ValueError(f"unexpected YOLO output shape: {raw.shape}")
        if (
            self.output_format == "yolo_one_to_many"
            and predictions.shape[0] < predictions.shape[1]
            and predictions.shape[0] <= 128
        ):
            predictions = predictions.T

        boxes: list[list[float]] = []
        scores: list[float] = []
        classes: list[int] = []
        for row in predictions:
            if self.output_format == "yolo_end2end":
                if row.shape[0] != 6:
                    raise ValueError(f"end-to-end row must contain 6 values, got {row.shape[0]}")
                x1, y1, x2, y2, score, raw_class = map(float, row)
                class_id = int(raw_class)
                left, top = (x1 - pad_x) / scale, (y1 - pad_y) / scale
                width, height = (x2 - x1) / scale, (y2 - y1) / scale
            else:
                if row.shape[0] < 4 + self.num_classes:
                    continue
                class_scores = row[4 : 4 + self.num_classes]
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])
                center_x, center_y, width, height = map(float, row[:4])
                left = (center_x - width / 2 - pad_x) / scale
                top = (center_y - height / 2 - pad_y) / scale
                width, height = width / scale, height / scale
            if class_id < 0 or class_id >= self.num_classes:
                continue
            if score < self.class_confidences[self.class_names[class_id]]:
                continue
            if width <= 0 or height <= 0:
                continue
            boxes.append([left, top, width, height])
            scores.append(score)
            classes.append(class_id)
        indices = _select_nms_indices(
            boxes,
            scores,
            classes,
            same_class_iou=self.iou_threshold,
            cross_class_iou=self.cross_class_iou_threshold,
        )
        detections: list[DetectedObject] = []
        for index in indices:
            idx = int(index)
            left, top, width, height = boxes[idx]
            x1 = max(0.0, min(float(original_width - 1), left))
            y1 = max(0.0, min(float(original_height - 1), top))
            x2 = max(x1 + 1, min(float(original_width), left + width))
            y2 = max(y1 + 1, min(float(original_height), top + height))
            class_id = classes[idx]
            detections.append(
                DetectedObject.from_class_id(
                    class_id,
                    base_url=self.base_url,
                    landing_status="-1",
                    motion_status="-1",
                    top_left_x=x1,
                    top_left_y=y1,
                    bottom_right_x=x2,
                    bottom_right_y=y2,
                )
            )
        return detections


def _se3_exp(xi: Any) -> Any:
    cv2, np = _require_cv()
    translation = np.asarray(xi[:3], dtype=np.float64).reshape(3, 1)
    rotation_vector = np.asarray(xi[3:], dtype=np.float64).reshape(3, 1)
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3:] = translation
    return transform


@dataclass(slots=True)
class OpticalFlowSE3Estimator:
    fx: float
    fy: float
    default_altitude_m: float
    previous_gray: Any | None = None
    pose: Any | None = None
    last_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _stream_key: tuple[str, str] | None = None

    def estimate(self, image: Image.Image, frame: FrameMetadata) -> tuple[float, float, float]:
        cv2, np = _require_cv()
        current = cv2.cvtColor(_pil_to_bgr(image), cv2.COLOR_BGR2GRAY)
        focal_x, focal_y = self.fx, self.fy
        try:
            from .camera import select_camera_profile

            profile = select_camera_profile(image.width, image.height, video_name=frame.video_name)
            camera_matrix = np.asarray(profile.camera_matrix, dtype=np.float64)
            distortion = np.asarray(profile.distortion, dtype=np.float64)
            current = cv2.undistort(current, camera_matrix, distortion)
            focal_x, focal_y = profile.fx, profile.fy
        except ValueError:
            # Unknown/synthetic resolutions retain explicit configured intrinsics.
            pass
        stream_key = (frame.session, frame.video_name)
        if self._stream_key != stream_key:
            self.previous_gray = None
            self.pose = np.eye(4, dtype=np.float64)
            self._stream_key = stream_key
        if self.pose is None:
            self.pose = np.eye(4, dtype=np.float64)

        reference = frame.reference_translation
        if frame.gps_health_status == 1 and reference is not None:
            self.pose[:3, 3] = np.asarray(reference)
            self.last_position = reference
            self.previous_gray = current
            return reference

        if self.previous_gray is None:
            self.previous_gray = current
            return self.last_position

        points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        if points is None or len(points) < 8:
            self.previous_gray = current
            return self.last_position
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, current, points, None)
        if tracked is None or status is None:
            self.previous_gray = current
            return self.last_position
        valid = status.reshape(-1) == 1
        old_points = points.reshape(-1, 2)[valid]
        new_points = tracked.reshape(-1, 2)[valid]
        if len(old_points) < 8:
            self.previous_gray = current
            return self.last_position

        flow = new_points - old_points
        median_dx, median_dy = np.median(flow, axis=0)
        last_altitude = abs(float(self.last_position[2]))
        altitude = last_altitude if last_altitude > 0.01 else self.default_altitude_m
        dx = -float(median_dx) * altitude / focal_x
        dy = -float(median_dy) * altitude / focal_y
        affine, _ = cv2.estimateAffinePartial2D(old_points, new_points, method=cv2.RANSAC)
        yaw = 0.0
        dz = 0.0
        if affine is not None:
            scale = math.hypot(float(affine[0, 0]), float(affine[0, 1]))
            yaw = math.atan2(float(affine[1, 0]), float(affine[0, 0]))
            if scale > 1e-6:
                dz = altitude * (1.0 - 1.0 / scale)
        self.pose = self.pose @ _se3_exp((dx, dy, dz, 0.0, 0.0, yaw))
        values = tuple(float(value) for value in self.pose[:3, 3])
        self.last_position = values
        self.previous_gray = current
        return values


@dataclass(slots=True)
class ORBReferenceMatcher:
    reference_dir: str
    _references: list[tuple[int, Any, Any, tuple[int, int], str | None, str | None]] = field(
        default_factory=list, init=False
    )

    def __post_init__(self) -> None:
        cv2, _ = _require_cv()
        orb = cv2.ORB_create(nfeatures=1500)
        directory = Path(self.reference_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        windows: dict[str, tuple[int, str, str]] = {}
        manifest = directory / "references_manifest.json"
        if manifest.is_file():
            for item in json.loads(manifest.read_text(encoding="utf-8")):
                windows[str(Path(item["image_path"]).resolve())] = (
                    int(item["object_id"]),
                    str(item["frame_start"]),
                    str(item["frame_end"]),
                )
        image_paths = sorted(
            item
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        for fallback_id, path in enumerate(image_paths, start=1):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            normalized = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
            keypoints, descriptors = orb.detectAndCompute(normalized, None)
            if descriptors is None or len(keypoints) < 8:
                continue
            match = re.search(r"\d+", path.stem)
            window = windows.get(str(path.resolve()))
            object_id = window[0] if window else int(match.group()) if match else fallback_id
            self._references.append(
                (
                    object_id,
                    keypoints,
                    descriptors,
                    (gray.shape[1], gray.shape[0]),
                    window[1] if window else None,
                    window[2] if window else None,
                )
            )
        LOGGER.info("reference_matcher loaded=%d dir=%s", len(self._references), directory)

    def warmup(self) -> None:
        cv2, np = _require_cv()
        checkerboard = np.zeros((128, 128), dtype=np.uint8)
        checkerboard[::16, :] = 255
        checkerboard[:, ::16] = 255
        normalized = cv2.createCLAHE(2.0, (8, 8)).apply(checkerboard)
        cv2.ORB_create(nfeatures=2000).detectAndCompute(normalized, None)
        cv2.BFMatcher(cv2.NORM_HAMMING)

    def match(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedUndefinedObject]:
        cv2, np = _require_cv()
        gray = cv2.cvtColor(_pil_to_bgr(image), cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        orb = cv2.ORB_create(nfeatures=2000)
        current_keypoints, current_descriptors = orb.detectAndCompute(gray, None)
        if current_descriptors is None or len(current_descriptors) < 2:
            return []
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        output: list[DetectedUndefinedObject] = []
        for (
            object_id,
            ref_keypoints,
            ref_descriptors,
            (width, height),
            frame_start,
            frame_end,
        ) in self._references:
            if frame_start is not None and frame_end is not None:
                from .models import ReferenceDefinition

                active = ReferenceDefinition(
                    url=f"http://local/reference/{object_id}/",
                    session="http://local/session/0/",
                    image_url="/local",
                    frame_start=frame_start,
                    frame_end=frame_end,
                    order=object_id,
                ).is_active(frame.url)
                if not active:
                    continue
            pairs = matcher.knnMatch(ref_descriptors, current_descriptors, k=2)
            good = [
                pair[0]
                for pair in pairs
                if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
            ]
            if len(good) < 8:
                continue
            source = np.float32([ref_keypoints[item.queryIdx].pt for item in good])
            target = np.float32([current_keypoints[item.trainIdx].pt for item in good])
            homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
            if homography is None or mask is None:
                continue
            inliers = mask.reshape(-1).astype(bool)
            inlier_count = int(inliers.sum())
            if inlier_count < 6 or inlier_count / len(good) < 0.40:
                continue
            projected_inliers = cv2.perspectiveTransform(
                source[inliers].reshape(-1, 1, 2),
                homography,
            ).reshape(-1, 2)
            reprojection_errors = np.linalg.norm(projected_inliers - target[inliers], axis=1)
            if float(np.median(reprojection_errors)) > 4.0:
                continue
            corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(
                -1, 1, 2
            )
            transformed = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
            if not np.isfinite(transformed).all():
                continue
            contour = transformed.astype(np.float32).reshape(-1, 1, 2)
            if not cv2.isContourConvex(contour):
                continue
            polygon_area = abs(float(cv2.contourArea(contour)))
            image_area = float(gray.shape[0] * gray.shape[1])
            if polygon_area < 16.0 or polygon_area > image_area * 4.0:
                continue
            x1, y1 = transformed.min(axis=0)
            x2, y2 = transformed.max(axis=0)
            raw_width = max(0.0, float(x2 - x1))
            raw_height = max(0.0, float(y2 - y1))
            visible_width = max(0.0, min(float(gray.shape[1]), float(x2)) - max(0.0, float(x1)))
            visible_height = max(
                0.0,
                min(float(gray.shape[0]), float(y2)) - max(0.0, float(y1)),
            )
            raw_box_area = raw_width * raw_height
            visible_box_area = visible_width * visible_height
            if raw_box_area <= 0 or visible_box_area / raw_box_area < 0.10:
                continue
            x1 = max(0.0, min(float(gray.shape[1] - 1), float(x1)))
            y1 = max(0.0, min(float(gray.shape[0] - 1), float(y1)))
            x2 = max(x1 + 1, min(float(gray.shape[1]), float(x2)))
            y2 = max(y1 + 1, min(float(gray.shape[0]), float(y2)))
            output.append(
                DetectedUndefinedObject(
                    object_id=object_id,
                    top_left_x=x1,
                    top_left_y=y1,
                    bottom_right_x=x2,
                    bottom_right_y=y2,
                )
            )
        return output


def build_vision_components(
    settings: ClientSettings,
) -> tuple[ObjectDetector, PositionEstimator, UndefinedObjectMatcher]:
    try:
        cv2, _ = _require_cv()
        cv2.setNumThreads(settings.opencv_num_threads)
    except RuntimeError as exc:
        LOGGER.warning("vision_dependencies_missing fallback=noop error=%s", exc)
        return (
            NoopObjectDetector(),
            LastKnownPositionEstimator(),
            NoopUndefinedObjectMatcher(),
        )

    if settings.yolo_onnx_path:
        base_detector: ObjectDetector = ONNXYoloDetector(
            settings.yolo_onnx_path,
            base_url=settings.base_url,
            manifest_path=settings.model_manifest_path,
            confidence=settings.detector_confidence,
            iou_threshold=settings.detector_iou_threshold,
            thresholds_path=settings.detector_thresholds_path,
            cross_class_iou_threshold=settings.detector_cross_class_iou_threshold,
            providers=settings.onnx_providers,
            intra_op_threads=settings.onnx_intra_op_threads,
            inter_op_threads=settings.onnx_inter_op_threads,
        )
        if settings.thermal_specialist_onnx_path:
            try:
                specialist_providers = settings.thermal_specialist_onnx_providers
                if not specialist_providers:
                    main_providers = tuple(
                        str(item) for item in base_detector.model_info().get("providers", [])
                    )
                    # Two CoreML sessions can starve one another during long runs.
                    # Isolate only that backend; NVIDIA targets remain on their
                    # automatically selected TensorRT/CUDA provider.
                    if "CoreMLExecutionProvider" in main_providers:
                        specialist_providers = ("CPUExecutionProvider",)
                thermal_specialist: ObjectDetector = ONNXYoloDetector(
                    settings.thermal_specialist_onnx_path,
                    base_url=settings.base_url,
                    manifest_path=settings.thermal_specialist_manifest_path,
                    confidence=settings.thermal_specialist_confidence,
                    iou_threshold=settings.detector_iou_threshold,
                    cross_class_iou_threshold=settings.detector_cross_class_iou_threshold,
                    expected_classes=("arac", "insan"),
                    providers=specialist_providers or settings.onnx_providers,
                    intra_op_threads=settings.thermal_specialist_onnx_intra_op_threads,
                    inter_op_threads=settings.onnx_inter_op_threads,
                )
            except Exception:
                LOGGER.exception(
                    "thermal_human_fusion initialization_failed fallback=main specialist=%s",
                    settings.thermal_specialist_onnx_path,
                )
            else:
                base_detector = ThermalHumanFusionDetector(
                    base_detector,
                    thermal_specialist,
                    specialist_timeout_ms=settings.thermal_specialist_timeout_ms,
                    slow_threshold_ms=settings.thermal_specialist_slow_threshold_ms,
                    cooldown_frames=settings.thermal_specialist_cooldown_frames,
                    cooldown_seconds=settings.thermal_specialist_cooldown_seconds,
                )
                LOGGER.info(
                    "thermal_human_fusion enabled=true specialist=%s",
                    settings.thermal_specialist_onnx_path,
                )
        detector: ObjectDetector = OptimizedObjectDetector(
            base_detector,
            TopologicalNoiseFilter(),
            FrustumProjector(settings.camera_fx, settings.camera_fy, settings.camera_altitude_m),
            duplicate_iou_threshold=settings.detector_iou_threshold,
            cross_class_duplicate_iou_threshold=settings.detector_cross_class_iou_threshold,
        )
    else:
        LOGGER.warning("yolo_model_missing fallback=noop")
        detector = NoopObjectDetector()
    if settings.enable_experimental_vo:
        LOGGER.info("visual_odometry enabled=true guarded_cross_validation=true")
        position: PositionEstimator = CalibratedHomographySE3Estimator(
            fx=settings.camera_fx,
            fy=settings.camera_fy,
            default_altitude_m=settings.camera_altitude_m,
            min_calibration_samples=settings.vo_min_calibration_samples,
            max_calibration_samples=settings.vo_max_calibration_samples,
            validation_fraction=settings.vo_validation_fraction,
            min_validation_samples=settings.vo_min_validation_samples,
            max_step_skill_ratio=settings.vo_max_step_skill_ratio,
            max_trajectory_skill_ratio=settings.vo_max_trajectory_skill_ratio,
            max_bias_ratio=settings.vo_max_bias_ratio,
            min_inliers=settings.vo_min_inliers,
            min_inlier_ratio=settings.vo_min_inlier_ratio,
            ransac_threshold_px=settings.vo_ransac_threshold_px,
            max_reprojection_error_px=settings.vo_max_reprojection_error_px,
            max_step_m=settings.vo_max_step_m,
            fallback_decay=settings.vo_fallback_decay,
            projective_features=settings.vo_projective_features,
        )
    else:
        position = LastKnownPositionEstimator()
    matcher: UndefinedObjectMatcher = NoopUndefinedObjectMatcher()
    if settings.reference_images_dir:
        try:
            matcher = ORBReferenceMatcher(settings.reference_images_dir)
        except (OSError, ValueError) as exc:
            LOGGER.warning(
                "reference_matcher_unavailable fallback=noop dir=%s error=%s",
                settings.reference_images_dir,
                exc,
            )
    return detector, position, matcher


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
