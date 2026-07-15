from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from PIL import Image, UnidentifiedImageError

from .models import (
    DetectedObject,
    DetectedTranslation,
    DetectedUndefinedObject,
    FrameMetadata,
    Prediction,
    prediction_id_from_frame_url,
)

if TYPE_CHECKING:
    from .config import ClientSettings


class CorruptFrameError(ValueError):
    pass


class ObjectDetector(Protocol):
    def warmup(self) -> None: ...

    def detect(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]: ...

    def health(self) -> dict[str, object]: ...

    def close(self) -> None: ...

    def model_info(self) -> dict[str, object]: ...


class PositionEstimator(Protocol):
    def estimate(self, image: Image.Image, frame: FrameMetadata) -> tuple[float, float, float]: ...


class UndefinedObjectMatcher(Protocol):
    def match(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedUndefinedObject]: ...


class NoopObjectDetector:
    def warmup(self) -> None:
        return None

    def detect(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]:
        return []

    def health(self) -> dict[str, object]:
        return {"ok": True, "mode": "noop"}

    def close(self) -> None:
        return None

    def model_info(self) -> dict[str, object]:
        return {"type": "noop", "classes": []}


@dataclass(slots=True)
class LastKnownPositionEstimator:
    """Fail-safe production default until calibrated odometry passes the real-data gate."""

    last_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def estimate(self, image: Image.Image, frame: FrameMetadata) -> tuple[float, float, float]:
        if frame.gps_health_status == 1 and frame.reference_translation is not None:
            self.last_position = frame.reference_translation
        return self.last_position


class NoopUndefinedObjectMatcher:
    def match(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedUndefinedObject]:
        return []


@dataclass(frozen=True, slots=True)
class InferenceOutcome:
    prediction: Prediction
    timings_ms: dict[str, float]


@dataclass(slots=True)
class PipelineInferenceEngine:
    object_detector: ObjectDetector = field(default_factory=NoopObjectDetector)
    position_estimator: PositionEstimator = field(default_factory=LastKnownPositionEstimator)
    undefined_matcher: UndefinedObjectMatcher = field(default_factory=NoopUndefinedObjectMatcher)
    _last_safe_position: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0), init=False)

    @classmethod
    def from_settings(cls, settings: ClientSettings) -> PipelineInferenceEngine:
        from .vision import build_vision_components

        detector, position_estimator, matcher = build_vision_components(settings)
        return cls(detector, position_estimator, matcher)

    def warmup(self) -> None:
        self.object_detector.warmup()
        matcher_warmup = getattr(self.undefined_matcher, "warmup", None)
        if matcher_warmup is not None:
            matcher_warmup()

    def export_recovery_state(self) -> dict[str, object] | None:
        exporter = getattr(self.position_estimator, "export_recovery_state", None)
        if exporter is None:
            return None
        state = exporter()
        if not isinstance(state, dict):
            raise TypeError("position recovery exporter must return a dictionary")
        return state

    def restore_recovery_state(
        self,
        state: dict[str, object],
        previous_image_bytes: bytes,
        previous_frame: FrameMetadata,
    ) -> None:
        restorer = getattr(self.position_estimator, "restore_recovery_state", None)
        if restorer is None:
            return
        previous_image = self._decode_image(previous_image_bytes)
        restorer(state, previous_image, previous_frame)

    def infer(
        self,
        frame: FrameMetadata,
        image_bytes: bytes,
        user_url: str,
        *,
        degraded: bool = False,
    ) -> Prediction:
        return self.infer_timed(
            frame,
            image_bytes,
            user_url,
            degraded=degraded,
        ).prediction

    def infer_timed(
        self,
        frame: FrameMetadata,
        image_bytes: bytes,
        user_url: str,
        *,
        degraded: bool = False,
    ) -> InferenceOutcome:
        total_started = time.perf_counter()
        started = time.perf_counter()
        image = self._decode_image(image_bytes)
        decode_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        if degraded and hasattr(self.object_detector, "detect_fast"):
            objects = self.object_detector.detect_fast(image, frame)  # type: ignore[attr-defined]
        else:
            objects = self.object_detector.detect(image, frame)
        detection_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        position = self.position_estimator.estimate(image, frame)
        odometry_ms = (time.perf_counter() - started) * 1000
        if all(math.isfinite(value) for value in position):
            self._last_safe_position = position

        started = time.perf_counter()
        undefined_objects = [] if degraded else self.undefined_matcher.match(image, frame)
        reference_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        apply_unknown_obstacles = getattr(self.object_detector, "apply_unknown_obstacles", None)
        if apply_unknown_obstacles is not None:
            objects = apply_unknown_obstacles(objects, undefined_objects, image.size)
        landing_analysis_ms = (time.perf_counter() - started) * 1000
        prediction = self._prediction(frame, user_url, objects, position, undefined_objects)
        return InferenceOutcome(
            prediction=prediction,
            timings_ms={
                "image_decode_ms": decode_ms,
                "preprocessing_ms": 0.0,
                "detection_ms": detection_ms,
                "tracking_ms": 0.0,
                "landing_analysis_ms": landing_analysis_ms,
                "odometry_ms": odometry_ms,
                "reference_matching_ms": reference_ms,
                "inference_ms": (time.perf_counter() - total_started) * 1000,
            },
        )

    def fallback(self, frame: FrameMetadata, user_url: str) -> Prediction:
        # Do not call model components here: a timed-out worker may still be running.
        if frame.gps_health_status == 1 and frame.reference_translation is not None:
            position = frame.reference_translation
            self._last_safe_position = position
        else:
            position = self._last_safe_position
        return self._prediction(frame, user_url, [], position, [])

    @staticmethod
    def _decode_image(image_bytes: bytes) -> Image.Image:
        if not image_bytes:
            raise CorruptFrameError("image body is empty")
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                source.verify()
            with Image.open(io.BytesIO(image_bytes)) as source:
                return source.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise CorruptFrameError("image cannot be decoded") from exc

    @staticmethod
    def prediction_id(frame_url: str) -> int:
        return prediction_id_from_frame_url(frame_url)

    def _prediction(
        self,
        frame: FrameMetadata,
        user_url: str,
        objects: list[DetectedObject],
        position: tuple[float, float, float],
        undefined_objects: list[DetectedUndefinedObject],
    ) -> Prediction:
        if not all(math.isfinite(value) for value in position):
            position = (0.0, 0.0, 0.0)
        return Prediction(
            id=self.prediction_id(frame.url),
            user=user_url,
            frame=frame.url,
            detected_objects=objects,
            detected_translations=[
                DetectedTranslation(
                    translation_x=position[0],
                    translation_y=position[1],
                    translation_z=position[2],
                )
            ],
            detected_undefined_objects=undefined_objects,
        )
