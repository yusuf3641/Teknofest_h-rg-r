from __future__ import annotations

import io
import math
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
    def detect(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]: ...


class PositionEstimator(Protocol):
    def estimate(self, image: Image.Image, frame: FrameMetadata) -> tuple[float, float, float]: ...


class UndefinedObjectMatcher(Protocol):
    def match(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedUndefinedObject]: ...


class NoopObjectDetector:
    def detect(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]:
        return []


@dataclass(slots=True)
class LastKnownPositionEstimator:
    """Placeholder; replace with the visual odometry implementation."""

    last_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def estimate(self, image: Image.Image, frame: FrameMetadata) -> tuple[float, float, float]:
        if frame.gps_health_status == 1 and frame.reference_translation is not None:
            self.last_position = frame.reference_translation
        return self.last_position


class NoopUndefinedObjectMatcher:
    def match(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedUndefinedObject]:
        return []


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

    def infer(
        self,
        frame: FrameMetadata,
        image_bytes: bytes,
        user_url: str,
        *,
        degraded: bool = False,
    ) -> Prediction:
        image = self._decode_image(image_bytes)
        if degraded and hasattr(self.object_detector, "detect_fast"):
            objects = self.object_detector.detect_fast(image, frame)  # type: ignore[attr-defined]
        else:
            objects = self.object_detector.detect(image, frame)
        position = self.position_estimator.estimate(image, frame)
        if all(math.isfinite(value) for value in position):
            self._last_safe_position = position
        undefined_objects = [] if degraded else self.undefined_matcher.match(image, frame)
        return self._prediction(frame, user_url, objects, position, undefined_objects)

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
