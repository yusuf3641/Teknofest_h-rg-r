from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

Status = Literal["-1", "0", "1"]
CLASS_IDS = frozenset({"0", "1", "2", "3"})
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
_CLASS_PATH_PATTERN = re.compile(r"^/classes/(?P<class_id>[0-3])/$")


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
    translation_x: float
    translation_y: float
    translation_z: float
    gps_health_status: Literal[0, 1] = Field(
        validation_alias=AliasChoices("gps_health_status", "health_status")
    )

    @property
    def reference_translation(self) -> tuple[float, float, float] | None:
        values = (self.translation_x, self.translation_y, self.translation_z)
        if all(math.isfinite(value) for value in values):
            return values
        return None


class BoundingBox(BaseModel):
    top_left_x: float = Field(ge=0)
    top_left_y: float = Field(ge=0)
    bottom_right_x: float = Field(ge=0)
    bottom_right_y: float = Field(ge=0)

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


class Prediction(BaseModel):
    id: int = Field(strict=True, ge=0, le=MAX_SAFE_JSON_INTEGER)
    user: str
    frame: str
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    detected_translations: list[DetectedTranslation] = Field(min_length=1, max_length=1)
    detected_undefined_objects: list[DetectedUndefinedObject] = Field(default_factory=list)

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
