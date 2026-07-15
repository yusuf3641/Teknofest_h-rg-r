from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

Status = Literal["-1", "0", "1"]
CLASS_IDS = frozenset({"0", "1", "2", "3"})
API_CLASS_IDS = frozenset({"0", "1", "2", "3", "4"})
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
_CLASS_PATH_PATTERN = re.compile(r"^/classes/(?P<class_id>[0-4])/$")


def prediction_id_from_frame_url(frame_url: str) -> int:
    """Return a deterministic SHA-256 ID that is safe in JSON/JavaScript integers."""
    digest_value = int.from_bytes(hashlib.sha256(frame_url.encode("utf-8")).digest(), "big")
    return 1 + (digest_value % MAX_SAFE_JSON_INTEGER)


def class_url_from_id(base_url: str, class_id: int | str) -> str:
    normalized_id = str(class_id)
    if normalized_id not in CLASS_IDS:
        raise ValueError(f"class_id must be one of {sorted(CLASS_IDS)}")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    return f"{parsed.scheme}://{parsed.netloc}/classes/{normalized_id}/"


class FrameMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    url: str
    image_url: str
    video_name: str
    session: str
    modality: Literal["rgb", "thermal", "unknown"] = "unknown"
    translation_x: float
    translation_y: float
    translation_z: float
    gps_health_status: Literal[0, 1] = Field(
        validation_alias=AliasChoices("gps_health_status", "health_status")
    )
    # The general 2026 specification says that per-frame orientation accompanies
    # the sample positioning video, while the technical API table does not assign
    # these values a mandatory field.  Keep quaternion telemetry optional so an
    # undocumented/extended server response can improve heading without making it
    # a protocol dependency.
    orientation_x: float | None = None
    orientation_y: float | None = None
    orientation_z: float | None = None
    orientation_w: float | None = None

    @field_validator("gps_health_status", mode="before")
    @classmethod
    def normalize_health_status(cls, value: Any) -> int:
        if value in {"0", 0} or value is False:
            return 0
        if value in {"1", 1} or value is True:
            return 1
        raise ValueError("gps_health_status/health_status must be 0 or 1")

    @property
    def reference_translation(self) -> tuple[float, float, float] | None:
        values = (self.translation_x, self.translation_y, self.translation_z)
        if all(math.isfinite(value) for value in values):
            return values
        return None

    @property
    def orientation_quaternion(self) -> tuple[float, float, float, float] | None:
        values = (
            self.orientation_x,
            self.orientation_y,
            self.orientation_z,
            self.orientation_w,
        )
        if any(value is None for value in values):
            return None
        quaternion = tuple(float(value) for value in values if value is not None)
        if len(quaternion) != 4 or not all(math.isfinite(value) for value in quaternion):
            return None
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm <= 1e-9:
            return None
        return tuple(value / norm for value in quaternion)

    @property
    def orientation_heading_rad(self) -> float | None:
        quaternion = self.orientation_quaternion
        if quaternion is None:
            return None
        x, y, z, w = quaternion
        heading = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        return heading if math.isfinite(heading) else None


class BoundingBox(BaseModel):
    top_left_x: float = Field(ge=0, allow_inf_nan=False)
    top_left_y: float = Field(ge=0, allow_inf_nan=False)
    bottom_right_x: float = Field(ge=0, allow_inf_nan=False)
    bottom_right_y: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_order(self) -> BoundingBox:
        if self.bottom_right_x <= self.top_left_x:
            raise ValueError("bottom_right_x must be greater than top_left_x")
        if self.bottom_right_y <= self.top_left_y:
            raise ValueError("bottom_right_y must be greater than top_left_y")
        return self


class DetectedObject(BoundingBox):
    cls: str
    landing_status: Status = "-1"
    motion_status: Status = "-1"

    @field_validator("cls")
    @classmethod
    def validate_class_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
            or _CLASS_PATH_PATTERN.fullmatch(parsed.path) is None
        ):
            raise ValueError("cls must match http://SERVER/classes/CLASS_ID/")
        return value

    @classmethod
    def from_class_id(
        cls,
        class_id: int | str,
        *,
        base_url: str,
        **data: Any,
    ) -> DetectedObject:
        return cls(cls=class_url_from_id(base_url, class_id), **data)

    @property
    def class_id(self) -> str:
        match = _CLASS_PATH_PATTERN.fullmatch(urlsplit(self.cls).path)
        if match is None:  # pragma: no cover - guarded by Pydantic validation
            raise ValueError(f"invalid class URL: {self.cls}")
        return match.group("class_id")


class DetectedTranslation(BaseModel):
    translation_x: float
    translation_y: float
    translation_z: float

    @model_validator(mode="after")
    def validate_finite(self) -> DetectedTranslation:
        values = (self.translation_x, self.translation_y, self.translation_z)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("submitted translations must be finite")
        return self


class DetectedUndefinedObject(BoundingBox):
    object_id: int = Field(strict=True, ge=0)


class ReferenceDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    session: str
    image_url: str
    frame_start: str
    frame_end: str
    frame_start_image_url: str | None = None
    frame_end_image_url: str | None = None
    order: int = Field(ge=0)

    def is_active(self, frame_url: str) -> bool:
        current = _trailing_identifier(frame_url)
        start = _trailing_identifier(self.frame_start)
        end = _trailing_identifier(self.frame_end)
        return start <= current <= end


class Prediction(BaseModel):
    id: int = Field(strict=True, ge=0, le=MAX_SAFE_JSON_INTEGER)
    user: str
    frame: str
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    detected_translations: list[DetectedTranslation] = Field(min_length=1, max_length=1)
    detected_undefined_objects: list[DetectedUndefinedObject] = Field(default_factory=list)

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def official_dict(
        self,
        base_url: str,
        reference_urls: Mapping[int, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "detected_objects": [
                {
                    "cls": _official_class_url(item.cls, base_url),
                    "landing_status": item.landing_status,
                    "moving_status": item.motion_status,
                    "top_left_x": str(item.top_left_x),
                    "top_left_y": str(item.top_left_y),
                    "bottom_right_x": str(item.bottom_right_x),
                    "bottom_right_y": str(item.bottom_right_y),
                }
                for item in self.detected_objects
            ],
            "detected_translations": [
                {
                    "translation_x": str(item.translation_x),
                    "translation_y": str(item.translation_y),
                    "translation_z": str(item.translation_z),
                }
                for item in self.detected_translations
            ],
            "reference_predictions": [
                {
                    "reference": _official_reference_url(
                        item.object_id,
                        base_url,
                        reference_urls,
                    ),
                    "top_left_x": str(item.top_left_x),
                    "top_left_y": str(item.top_left_y),
                    "bottom_right_x": str(item.bottom_right_x),
                    "bottom_right_y": str(item.bottom_right_y),
                }
                for item in self.detected_undefined_objects
            ],
        }


def _official_class_url(cls_url: str, base_url: str) -> str:
    match = _CLASS_PATH_PATTERN.fullmatch(urlsplit(cls_url).path)
    if match is None:
        raise ValueError(f"invalid class URL: {cls_url}")
    internal_id = int(match.group("class_id"))
    # Internal pipeline IDs follow the documented constants 0..3. The official
    # 2026 connection interface submits API URLs as classes/1..4.
    api_id = internal_id + 1 if str(internal_id) in CLASS_IDS else internal_id
    normalized_id = str(api_id)
    if normalized_id not in API_CLASS_IDS:
        raise ValueError(f"official class id must be one of {sorted(API_CLASS_IDS)}")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    return f"{parsed.scheme}://{parsed.netloc}/classes/{normalized_id}/"


def _official_reference_url(
    object_id: int,
    base_url: str,
    reference_urls: Mapping[int, str] | None,
) -> str:
    if reference_urls and object_id in reference_urls:
        value = reference_urls[object_id]
    else:
        parsed_base = urlsplit(base_url)
        if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        value = f"{parsed_base.scheme}://{parsed_base.netloc}/reference/{object_id}/"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid reference URL: {value}")
    if not re.fullmatch(r"/reference/\d+/", parsed.path):
        raise ValueError(f"reference URL must end with /reference/<id>/: {value}")
    return value


def _trailing_identifier(value: str) -> int:
    match = re.search(r"(?P<identifier>\d+)/?$", value)
    if match is None:
        raise ValueError(f"URL does not end with a numeric identifier: {value}")
    return int(match.group("identifier"))
