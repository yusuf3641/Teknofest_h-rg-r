from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

Status = Literal["-1", "0", "1"]


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
    object_id: int


class Prediction(BaseModel):
    id: int = Field(ge=0)
    user: str
    frame: str
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    detected_translations: list[DetectedTranslation] = Field(min_length=1, max_length=1)
    detected_undefined_objects: list[DetectedUndefinedObject] = Field(default_factory=list)

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
