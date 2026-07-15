from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CameraProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    modality: Literal["rgb", "thermal"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    distortion: tuple[float, float, float, float]
    expected_channel_layout: Literal["RGB", "L"]

    @property
    def camera_matrix(self) -> tuple[tuple[float, float, float], ...]:
        return ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0))


CAMERA_PROFILES = (
    CameraProfile(
        name="thermal_640x512",
        modality="thermal",
        width=640,
        height=512,
        fx=731.7965,
        fy=732.0172,
        cx=319.2367,
        cy=251.2424,
        distortion=(-0.3507, 0.1137, 0.0, 0.0),
        expected_channel_layout="L",
    ),
    CameraProfile(
        name="rgb_4k",
        modality="rgb",
        width=4000,
        height=3000,
        fx=2792.2,
        fy=2795.2,
        cx=1988.0,
        cy=1562.2,
        distortion=(0.0798, -0.1867, 0.0, 0.0),
        expected_channel_layout="RGB",
    ),
    CameraProfile(
        name="rgb_1080p",
        modality="rgb",
        width=1920,
        height=1080,
        # veri/Kalibrasyon_Parametreleri.txt, 33 patterns, 0.645 px reprojection error.
        fx=1413.3,
        fy=1418.8,
        cx=950.0639,
        cy=543.3796,
        distortion=(-0.0091, 0.0666, 0.0, 0.0),
        expected_channel_layout="RGB",
    ),
)


def select_camera_profile(
    width: int,
    height: int,
    *,
    video_name: str = "",
) -> CameraProfile:
    candidates = [item for item in CAMERA_PROFILES if (item.width, item.height) == (width, height)]
    if len(candidates) == 1:
        return candidates[0]
    tokens = set(re.findall(r"[a-z0-9]+", video_name.casefold()))
    modality = (
        "thermal"
        if tokens.intersection({"thermal", "termal", "infrared", "ir"})
        else "rgb"
    )
    for profile in candidates:
        if profile.modality == modality:
            return profile
    raise ValueError(f"no calibrated camera profile for {width}x{height} video={video_name!r}")
