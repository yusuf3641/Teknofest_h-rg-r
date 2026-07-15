from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import logging
import math
import re
import secrets
import threading
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import parse_qs, urljoin, urlsplit

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import MockSettings
from .models import Prediction, ReferenceDefinition

logger = logging.getLogger(__name__)


class OfficialObjectPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cls: str
    landing_status: Literal["-1", "0", "1"]
    moving_status: Literal["-1", "0", "1"]
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float

    @field_validator("cls")
    @classmethod
    def validate_cls(cls, value: str) -> str:
        if urlsplit(value).path not in {f"/classes/{index}/" for index in range(1, 5)}:
            raise ValueError("official class URL must end with /classes/1..4/")
        return value


class OfficialTranslationPayload(BaseModel):
    translation_x: float
    translation_y: float
    translation_z: float

    @field_validator("translation_x", "translation_y", "translation_z")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("translation must be finite")
        return value


class OfficialReferencePredictionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: str
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float

    @field_validator("reference")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        if re.fullmatch(r"/reference/\d+/", urlsplit(value).path) is None:
            raise ValueError("official reference URL must end with /reference/<id>/")
        return value


class OfficialPredictionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame: str
    detected_objects: list[OfficialObjectPayload] = Field(default_factory=list)
    detected_translations: list[OfficialTranslationPayload] = Field(min_length=1, max_length=1)
    reference_predictions: list[OfficialReferencePredictionPayload] = Field(default_factory=list)


@dataclass(slots=True)
class MockState:
    settings: MockSettings
    frame_source: FrameSource
    next_index: int = 0
    outstanding_index: int | None = None
    accepted_count: int = 0
    last_accepted_frame: str | None = None
    last_accepted_digest: str | None = None
    last_empty_fault_index: int | None = None
    recent_frame_urls: deque[str] = field(default_factory=lambda: deque(maxlen=100))
    valid_token: str | None = None
    token_request_count: int = 0
    request_count: int = 0
    request_get_count: int = 0
    request_post_count: int = 0
    injected_401_count: int = 0
    injected_429_count: int = 0
    injected_5xx_count: int = 0
    frame_issue_count: int = 0
    frame_response_count: int = 0
    repeated_frame_get_count: int = 0
    empty_metadata_fault_count: int = 0
    image_request_count: int = 0
    corrupt_image_fault_count: int = 0
    empty_image_fault_count: int = 0
    prediction_payload_count: int = 0
    duplicate_prediction_count: int = 0
    rejected_prediction_count: int = 0
    order_violation_count: int = 0
    position_errors_m: list[float] = field(default_factory=list)
    outage_position_errors_m: list[float] = field(default_factory=list)
    outage_hold_errors_m: list[float] = field(default_factory=list)
    first_position_error_m: float | None = None
    final_position_error_m: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @staticmethod
    def _error_summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {
                "count": 0,
                "mae_m": None,
                "rmse_m": None,
                "p95_m": None,
                "max_m": None,
            }
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return {
            "count": len(values),
            "mae_m": sum(values) / len(values),
            "rmse_m": math.sqrt(sum(value * value for value in values) / len(values)),
            "p95_m": ordered[p95_index],
            "max_m": ordered[-1],
        }

    def record_position(
        self,
        index: int,
        predicted: tuple[float, float, float],
    ) -> None:
        """Score the submitted translation against truth hidden from the client."""
        truth = self.frame_source.translation(index)
        if truth is None or not all(math.isfinite(value) for value in (*truth, *predicted)):
            return
        error = math.dist(truth, predicted)
        self.position_errors_m.append(error)
        self.final_position_error_m = error
        if index == 0:
            self.first_position_error_m = error

        if index < self.settings.healthy_frames:
            return
        self.outage_position_errors_m.append(error)
        if self.settings.healthy_frames <= 0:
            return
        anchor = self.frame_source.translation(self.settings.healthy_frames - 1)
        if anchor is not None and all(math.isfinite(value) for value in anchor):
            self.outage_hold_errors_m.append(math.dist(truth, anchor))

    def position_summary(self) -> dict[str, Any]:
        outage = self._error_summary(self.outage_position_errors_m)
        hold = self._error_summary(self.outage_hold_errors_m)
        outage_mae = outage["mae_m"]
        hold_mae = hold["mae_m"]
        improvement = None
        if isinstance(outage_mae, float) and isinstance(hold_mae, float) and hold_mae > 0.0:
            improvement = 100.0 * (hold_mae - outage_mae) / hold_mae
        return {
            "all": self._error_summary(self.position_errors_m),
            "outage": outage,
            "outage_hold_baseline": hold,
            "outage_improvement_percent": improvement,
            "first_error_m": self.first_position_error_m,
            "final_error_m": self.final_position_error_m,
        }

    def metadata(self, request: Request, index: int) -> dict[str, Any]:
        root = str(request.base_url).rstrip("/")
        healthy = index < self.settings.healthy_frames
        translation = self.frame_source.translation(index)
        if translation is None:
            if healthy:
                # The official reference frame starts at x0=y0=z0=0 m.
                translation = (round(index * 0.02, 4), round(index * 0.01, 4), 0.0)
            else:
                translation = ("NaN", "NaN", "NaN")
        elif not healthy:
            # In unhealthy windows the competition may hide/poison GPS values.
            # Keeping NaN here exercises the client's visual-odometry fallback path.
            translation = ("NaN", "NaN", "NaN")
        payload = {
            "url": f"{root}/frames/{index}/",
            "image_url": f"/media/frame_{index:06d}.jpg",
            "video_name": self.settings.video_name,
            "session": urljoin(f"{root}/", self.settings.session_url),
            "translation_x": translation[0],
            "translation_y": translation[1],
            "translation_z": translation[2],
            # The PDF figure uses health_status; the prose uses gps_health_status.
            "health_status": 1 if healthy else 0,
        }
        orientation_reader = getattr(self.frame_source, "orientation", None)
        orientation = orientation_reader(index) if callable(orientation_reader) else None
        if orientation is not None:
            payload.update(
                dict(
                    zip(
                        (
                            "orientation_x",
                            "orientation_y",
                            "orientation_z",
                            "orientation_w",
                        ),
                        orientation,
                        strict=True,
                    )
                )
            )
        return payload

    def official_frame_metadata(self, request: Request, index: int) -> dict[str, Any]:
        payload = self.metadata(request, index)
        for key in (
            "translation_x",
            "translation_y",
            "translation_z",
            "health_status",
        ):
            payload.pop(key)
        return payload

    def official_translation_metadata(self, request: Request, index: int) -> dict[str, Any]:
        frame = self.metadata(request, index)
        root = str(request.base_url).rstrip("/")
        payload = {
            "url": f"{root}/translation/{index}/",
            "frame": frame["url"],
            "image_url": frame["image_url"],
            "video_name": frame["video_name"],
            "session": frame["session"],
            "translation_x": frame["translation_x"],
            "translation_y": frame["translation_y"],
            "translation_z": frame["translation_z"],
            "health_status": frame["health_status"],
        }
        for key in (
            "orientation_x",
            "orientation_y",
            "orientation_z",
            "orientation_w",
        ):
            if key in frame:
                payload[key] = frame[key]
        return payload


def _prediction_digest(prediction: Prediction) -> str:
    canonical = json.dumps(prediction.canonical_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class FrameSource(Protocol):
    frame_count: int

    def render(self, index: int) -> bytes: ...

    def translation(self, index: int) -> tuple[float, float, float] | None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class SyntheticFrameSource:
    frame_count: int
    modality: Literal["rgb", "thermal"] = "rgb"

    def render(self, index: int) -> bytes:
        if self.modality == "thermal":
            return self._render_thermal(index)
        image = Image.new("RGB", (640, 480), color=(15, 23, 42))
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 80, 560, 400), outline=(34, 211, 238), width=4)
        draw.text((100, 110), f"HURGOR MOCK FRAME {index:06d}", fill=(248, 250, 252))
        draw.text((100, 150), "TEK KARE / IDEMPOTENT API", fill=(148, 163, 184))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82)
        return buffer.getvalue()

    def _render_thermal(self, index: int) -> bytes:
        image = Image.new("L", (640, 512), color=34)
        draw = ImageDraw.Draw(image)
        offset = (index * 5) % 220
        draw.rectangle((32, 42, 608, 470), outline=96, width=3)
        draw.rectangle((80 + offset, 120, 160 + offset, 175), fill=210)
        draw.rectangle((360, 290, 500, 350), fill=72, outline=150, width=2)
        draw.ellipse((250, 185, 320, 255), fill=180)
        draw.text((40, 22), f"HURGOR THERMAL MOCK {index:06d}", fill=230)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        return buffer.getvalue()

    def close(self) -> None:
        return None

    def translation(self, index: int) -> tuple[float, float, float] | None:
        del index
        return None


class DirectoryFrameSource:
    """Serve a deterministic, bounded sequence of real images from a directory."""

    SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        image_dir: str,
        *,
        frame_stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        root = Path(image_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"görüntü klasörü bulunamadı: {root}")
        stride = max(1, frame_stride)
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in self.SUPPORTED_SUFFIXES
        )[::stride]
        if max_frames is not None and max_frames > 0:
            paths = paths[:max_frames]
        if not paths:
            raise ValueError(f"görüntü klasöründe desteklenen dosya yok: {root}")
        self.root = root
        self.paths = paths
        self.frame_count = len(paths)

    def render(self, index: int) -> bytes:
        try:
            path = self.paths[index]
        except IndexError as exc:
            raise ValueError(f"görüntü frame bulunamadı: {index}") from exc
        with Image.open(path) as source:
            source.load()
            image = source.copy()
        if image.mode not in {"L", "RGB"}:
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()

    def translation(self, index: int) -> tuple[float, float, float] | None:
        del index
        return None

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class TranslationRow:
    translation_x: float
    translation_y: float
    translation_z: float
    frame_number: str
    orientation: tuple[float, float, float, float] | None = None


class TranslationTrack:
    def __init__(self, csv_path: str, *, frame_stride: int = 1) -> None:
        self.path = csv_path
        self.frame_stride = max(1, frame_stride)
        self.rows = self._load(csv_path)
        if not self.rows:
            raise ValueError(f"translation CSV boş: {csv_path}")

    @staticmethod
    def _load(csv_path: str) -> list[TranslationRow]:
        rows: list[TranslationRow] = []
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"translation_x", "translation_y", "translation_z", "frame_numbers"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"translation CSV eksik kolonlar: {sorted(missing)}")
            orientation_columns = (
                "orientation_x",
                "orientation_y",
                "orientation_z",
                "orientation_w",
            )
            available_orientation_columns = set(orientation_columns).intersection(
                reader.fieldnames or []
            )
            if available_orientation_columns and len(available_orientation_columns) != 4:
                missing_orientation = set(orientation_columns).difference(
                    available_orientation_columns
                )
                raise ValueError(
                    f"translation CSV eksik oryantasyon kolonları: {sorted(missing_orientation)}"
                )
            for row in reader:
                orientation = None
                if len(available_orientation_columns) == 4:
                    orientation = tuple(float(row[column]) for column in orientation_columns)
                    if not all(math.isfinite(value) for value in orientation):
                        raise ValueError("translation CSV sonlu oryantasyon içermiyor")
                rows.append(
                    TranslationRow(
                        translation_x=float(row["translation_x"]),
                        translation_y=float(row["translation_y"]),
                        translation_z=float(row["translation_z"]),
                        frame_number=row["frame_numbers"],
                        orientation=orientation,
                    )
                )
        return rows

    @property
    def frame_count(self) -> int:
        return (len(self.rows) + self.frame_stride - 1) // self.frame_stride

    def source_index(self, index: int) -> int:
        return min(index * self.frame_stride, len(self.rows) - 1)

    def translation(self, index: int) -> tuple[float, float, float]:
        row = self.rows[self.source_index(index)]
        return (row.translation_x, row.translation_y, row.translation_z)

    def orientation(self, index: int) -> tuple[float, float, float, float] | None:
        return self.rows[self.source_index(index)].orientation


class VideoFrameSource:
    def __init__(
        self,
        video_path: str,
        *,
        frame_stride: int = 1,
        translation_track: TranslationTrack | None = None,
        max_frames: int | None = None,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Video mock için `pip install -e '.[ai]'` çalıştırılmalıdır"
            ) from exc
        self.cv2 = cv2
        self.path = str(video_path)
        self.frame_stride = max(1, frame_stride)
        self.translation_track = translation_track
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise ValueError(f"video açılamadı: {self.path}")
        self.raw_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.raw_frame_count <= 0:
            raise ValueError(f"videoda frame bulunamadı: {self.path}")
        self.frame_count = (self.raw_frame_count + self.frame_stride - 1) // self.frame_stride
        if self.translation_track is not None:
            self.frame_count = min(self.frame_count, self.translation_track.frame_count)
        if max_frames is not None and max_frames > 0:
            self.frame_count = min(self.frame_count, max_frames)
        self.next_index = 0
        self.lock = threading.Lock()

    def render(self, index: int) -> bytes:
        source_index = min(index * self.frame_stride, self.raw_frame_count - 1)
        with self.lock:
            expected_source = self.next_index * self.frame_stride
            if source_index != expected_source:
                self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = self.capture.read()
            if not ok:
                raise ValueError(f"video frame okunamadı: logical={index} source={source_index}")
            self.next_index = index + 1
            ok, encoded = self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise ValueError(f"video frame JPEG'e çevrilemedi: {index}")
            return encoded.tobytes()

    def close(self) -> None:
        with self.lock:
            self.capture.release()

    def translation(self, index: int) -> tuple[float, float, float] | None:
        if self.translation_track is None:
            return None
        return self.translation_track.translation(index)

    def orientation(self, index: int) -> tuple[float, float, float, float] | None:
        if self.translation_track is None:
            return None
        return self.translation_track.orientation(index)


def create_app(settings: MockSettings | None = None) -> FastAPI:
    configured = settings or MockSettings.from_env()
    source: FrameSource
    if configured.image_dir:
        try:
            source = DirectoryFrameSource(
                configured.image_dir,
                frame_stride=configured.frame_stride,
                max_frames=configured.frame_count,
            )
            configured = replace(configured, frame_count=source.frame_count)
        except (OSError, ValueError) as exc:
            logger.warning(
                "mock image source unavailable; using synthetic frames: %s",
                exc,
            )
            source = SyntheticFrameSource(configured.frame_count, configured.modality)
    elif configured.video_path:
        try:
            translation_track = (
                TranslationTrack(
                    configured.translation_csv_path,
                    frame_stride=configured.frame_stride,
                )
                if configured.translation_csv_path
                else None
            )
            source = VideoFrameSource(
                configured.video_path,
                frame_stride=configured.frame_stride,
                translation_track=translation_track,
                max_frames=configured.frame_count,
            )
            configured = replace(configured, frame_count=source.frame_count)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "mock video/translation source unavailable; using synthetic frames: %s",
                exc,
            )
            source = SyntheticFrameSource(configured.frame_count, configured.modality)
    else:
        source = SyntheticFrameSource(configured.frame_count, configured.modality)
    state = MockState(configured, source)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            state.frame_source.close()

    app = FastAPI(title="HürGör Mock Yarışma Sunucusu", version="0.1.0", lifespan=lifespan)
    app.state.mock = state

    async def require_token(request: Request) -> Response | None:
        async with state.lock:
            state.request_count += 1
            if request.method == "GET":
                state.request_get_count += 1
            elif request.method == "POST":
                state.request_post_count += 1
            if (
                state.settings.rate_limit_every
                and state.request_count % state.settings.rate_limit_every == 0
            ):
                state.injected_429_count += 1
                return JSONResponse(
                    {"detail": "rate limited"},
                    status_code=429,
                    headers={"Retry-After": str(state.settings.retry_after_seconds)},
                )
            if (
                state.settings.server_error_every
                and state.request_count % state.settings.server_error_every == 0
            ):
                state.injected_5xx_count += 1
                return JSONResponse(
                    {"detail": "injected failure"},
                    status_code=state.settings.server_error_status,
                )
            authorization = request.headers.get("authorization", "")
            if state.valid_token is None or authorization != f"Token {state.valid_token}":
                state.injected_401_count += 1
                return JSONResponse(
                    {"detail": "Authentication credentials were not provided."},
                    status_code=401,
                    headers={"WWW-Authenticate": "Token"},
                )
            state.token_request_count += 1
            if (
                state.settings.token_expire_after_requests
                and state.token_request_count > state.settings.token_expire_after_requests
            ):
                state.valid_token = None
                state.injected_401_count += 1
                return JSONResponse(
                    {"detail": "Token expired"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Token"},
                )
        return None

    @app.post("/auth/")
    async def auth(request: Request) -> Response:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                credentials = await request.json()
            except ValueError:
                credentials = {}
        else:
            parsed = parse_qs((await request.body()).decode("utf-8", errors="replace"))
            credentials = {key: values[0] for key, values in parsed.items() if values}
        username = credentials.get("username") if isinstance(credentials, dict) else None
        password = credentials.get("password") if isinstance(credentials, dict) else None
        if username != state.settings.mock_username or password != state.settings.mock_password:
            return JSONResponse({"detail": "invalid credentials"}, status_code=400)
        async with state.lock:
            state.valid_token = secrets.token_urlsafe(24)
            state.token_request_count = 0
            token = state.valid_token
        return JSONResponse({"token": token})

    @app.get("/progress/")
    async def official_progress(request: Request) -> Response:
        rejection = await require_token(request)
        if rejection is not None:
            return rejection
        async with state.lock:
            return JSONResponse(
                {
                    "total_frames": state.settings.frame_count,
                    "processed_frames": state.accepted_count,
                }
            )

    @app.get("/classes/")
    async def official_classes(request: Request) -> Response:
        rejection = await require_token(request)
        if rejection is not None:
            return rejection
        root = str(request.base_url).rstrip("/")
        names = ("Taşıt", "İnsan", "UAP", "UAİ", "Referans_Obje")
        return JSONResponse(
            [
                {"url": f"{root}/classes/{index}/", "id": index, "name": name}
                for index, name in enumerate(names, start=1)
            ]
        )

    @app.get("/reference/")
    async def official_references(request: Request) -> Response:
        rejection = await require_token(request)
        if rejection is not None:
            return rejection
        if state.settings.frame_count == 0:
            return JSONResponse([])
        root = str(request.base_url).rstrip("/")
        reference = ReferenceDefinition(
            url=f"{root}/reference/1/",
            session=urljoin(f"{root}/", state.settings.session_url),
            image_url="/media/reference_1.png",
            frame_start=f"{root}/frames/0/",
            frame_end=f"{root}/frames/{max(0, state.settings.frame_count - 1)}/",
            frame_start_image_url="/media/frame_000000.jpg",
            frame_end_image_url=(f"/media/frame_{max(0, state.settings.frame_count - 1):06d}.jpg"),
            order=1,
        )
        return JSONResponse([reference.model_dump(mode="json")])

    @app.get("/frames/")
    async def official_frame(request: Request) -> Response:
        rejection = await require_token(request)
        if rejection is not None:
            return rejection
        async with state.lock:
            if state.outstanding_index is not None:
                index = state.outstanding_index
                state.repeated_frame_get_count += 1
            elif state.next_index >= state.settings.frame_count:
                return Response(status_code=204)
            else:
                index = state.next_index
                should_fault = (
                    state.settings.empty_every > 0
                    and (index + 1) % state.settings.empty_every == 0
                    and state.last_empty_fault_index != index
                )
                if should_fault:
                    state.last_empty_fault_index = index
                    state.empty_metadata_fault_count += 1
                    return JSONResponse([])
                state.outstanding_index = index
                state.frame_issue_count += 1
            state.frame_response_count += 1
            payload = state.official_frame_metadata(request, index)
        if state.settings.get_delay_ms:
            await asyncio.sleep(state.settings.get_delay_ms / 1000)
        return JSONResponse([payload])

    @app.get("/translation/")
    async def official_translation(request: Request) -> Response:
        rejection = await require_token(request)
        if rejection is not None:
            return rejection
        async with state.lock:
            if state.outstanding_index is None:
                return JSONResponse({"detail": "no outstanding frame"}, status_code=409)
            payload = state.official_translation_metadata(request, state.outstanding_index)
        return JSONResponse([payload])

    @app.get("/api/frames/next")
    async def get_next_frame(request: Request) -> Response:
        async with state.lock:
            if state.outstanding_index is not None:
                index = state.outstanding_index
                state.repeated_frame_get_count += 1
            elif state.next_index >= state.settings.frame_count:
                return Response(status_code=204)
            else:
                index = state.next_index
                should_fault = (
                    state.settings.empty_every > 0
                    and (index + 1) % state.settings.empty_every == 0
                    and state.last_empty_fault_index != index
                )
                if should_fault:
                    state.last_empty_fault_index = index
                    state.empty_metadata_fault_count += 1
                    return JSONResponse([])
                state.outstanding_index = index
                state.frame_issue_count += 1

            state.frame_response_count += 1
            payload = state.metadata(request, index)
        if state.settings.get_delay_ms:
            await asyncio.sleep(state.settings.get_delay_ms / 1000)
        return JSONResponse([payload])

    @app.get("/media/frame_{index}.jpg")
    async def get_frame_image(index: int) -> Response:
        async with state.lock:
            if index < 0 or index >= state.settings.frame_count:
                raise HTTPException(status_code=404, detail="frame does not exist")
            state.image_request_count += 1
            empty = (
                state.settings.empty_image_every > 0
                and (index + 1) % state.settings.empty_image_every == 0
            )
            corrupt = (
                state.settings.corrupt_every > 0 and (index + 1) % state.settings.corrupt_every == 0
            )
            if empty:
                state.empty_image_fault_count += 1
            elif corrupt:
                state.corrupt_image_fault_count += 1
        content = b"" if empty else b"corrupt-jpeg" if corrupt else state.frame_source.render(index)
        return Response(content=content, media_type="image/jpeg")

    @app.get("/media/reference_1.png")
    async def get_reference_image() -> Response:
        return Response(content=state.frame_source.render(0), media_type="image/jpeg")

    @app.post("/prediction/")
    async def official_prediction(request: Request, payload: Annotated[Any, Body()]) -> Response:
        rejection = await require_token(request)
        if rejection is not None:
            return rejection
        if state.settings.post_delay_ms:
            await asyncio.sleep(state.settings.post_delay_ms / 1000)
        async with state.lock:
            state.prediction_payload_count += 1
        if not isinstance(payload, dict):
            async with state.lock:
                state.rejected_prediction_count += 1
            return JSONResponse({"detail": "prediction must be one JSON object"}, status_code=422)
        try:
            prediction = OfficialPredictionPayload.model_validate(payload)
        except ValidationError as exc:
            async with state.lock:
                state.rejected_prediction_count += 1
            return JSONResponse({"detail": exc.errors()}, status_code=422)
        root = str(request.base_url).rstrip("/")
        if any(not item.cls.startswith(f"{root}/classes/") for item in prediction.detected_objects):
            async with state.lock:
                state.rejected_prediction_count += 1
            return JSONResponse({"detail": "class URL host mismatch"}, status_code=422)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        async with state.lock:
            if prediction.frame == state.last_accepted_frame:
                state.duplicate_prediction_count += 1
                return JSONResponse(
                    {
                        "accepted": True,
                        "duplicate": True,
                        "same_payload": digest == state.last_accepted_digest,
                    }
                )
            if state.outstanding_index is None:
                state.rejected_prediction_count += 1
                state.order_violation_count += 1
                return JSONResponse({"detail": "no outstanding frame"}, status_code=409)
            expected = f"/frames/{state.outstanding_index}/"
            if not prediction.frame.endswith(expected):
                state.rejected_prediction_count += 1
                state.order_violation_count += 1
                return JSONResponse(
                    {"detail": f"expected frame suffix {expected}"}, status_code=409
                )
            submitted_translation = prediction.detected_translations[0]
            state.record_position(
                state.outstanding_index,
                (
                    submitted_translation.translation_x,
                    submitted_translation.translation_y,
                    submitted_translation.translation_z,
                ),
            )
            state.last_accepted_frame = prediction.frame
            state.last_accepted_digest = digest
            state.recent_frame_urls.append(prediction.frame)
            state.accepted_count += 1
            state.next_index = state.outstanding_index + 1
            state.outstanding_index = None
        return JSONResponse({"accepted": True, "duplicate": False})

    @app.post("/api/predictions")
    async def post_prediction(
        request: Request,
        payload: Annotated[Any, Body()],
    ) -> dict[str, Any]:
        if state.settings.post_delay_ms:
            await asyncio.sleep(state.settings.post_delay_ms / 1000)
        async with state.lock:
            state.prediction_payload_count += 1
        if not isinstance(payload, list) or len(payload) != 1:
            async with state.lock:
                state.rejected_prediction_count += 1
            raise HTTPException(
                status_code=422,
                detail="official contract requires a one-item prediction list",
            )
        raw_prediction: Any = payload[0]
        try:
            prediction = Prediction.model_validate(raw_prediction)
        except ValidationError as exc:
            async with state.lock:
                state.rejected_prediction_count += 1
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        root = str(request.base_url).rstrip("/")
        expected_user = urljoin(f"{root}/", state.settings.user_url)
        if prediction.user != expected_user:
            async with state.lock:
                state.rejected_prediction_count += 1
            raise HTTPException(
                status_code=422,
                detail=f"expected user {expected_user}",
            )
        expected_class_prefix = f"{root}/classes/"
        class_url_mismatch = any(
            not item.cls.startswith(expected_class_prefix) for item in prediction.detected_objects
        )
        if class_url_mismatch:
            async with state.lock:
                state.rejected_prediction_count += 1
            raise HTTPException(
                status_code=422,
                detail=f"class URLs must start with {expected_class_prefix}",
            )

        digest = _prediction_digest(prediction)
        async with state.lock:
            if prediction.frame == state.last_accepted_frame:
                state.duplicate_prediction_count += 1
                return {
                    "accepted": True,
                    "duplicate": True,
                    "same_payload": digest == state.last_accepted_digest,
                }
            if state.outstanding_index is None:
                state.rejected_prediction_count += 1
                state.order_violation_count += 1
                raise HTTPException(status_code=409, detail="no outstanding frame")
            expected = f"/frames/{state.outstanding_index}/"
            if not prediction.frame.endswith(expected):
                state.rejected_prediction_count += 1
                state.order_violation_count += 1
                raise HTTPException(
                    status_code=409,
                    detail=f"expected frame suffix {expected}",
                )

            submitted_translation = prediction.detected_translations[0]
            state.record_position(
                state.outstanding_index,
                (
                    submitted_translation.translation_x,
                    submitted_translation.translation_y,
                    submitted_translation.translation_z,
                ),
            )
            state.last_accepted_frame = prediction.frame
            state.last_accepted_digest = digest
            state.recent_frame_urls.append(prediction.frame)
            state.accepted_count += 1
            state.next_index = state.outstanding_index + 1
            state.outstanding_index = None
            return {"accepted": True, "duplicate": False}

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        async with state.lock:
            return {
                "next_index": state.next_index,
                "outstanding_index": state.outstanding_index,
                "accepted_count": state.accepted_count,
                "frame_count": state.settings.frame_count,
                "recent_state_size": len(state.recent_frame_urls),
                "request_count": state.request_count,
                "request_get_count": state.request_get_count,
                "request_post_count": state.request_post_count,
                "injected_401_count": state.injected_401_count,
                "injected_429_count": state.injected_429_count,
                "injected_5xx_count": state.injected_5xx_count,
                "frame_issue_count": state.frame_issue_count,
                "frame_response_count": state.frame_response_count,
                "repeated_frame_get_count": state.repeated_frame_get_count,
                "empty_metadata_fault_count": state.empty_metadata_fault_count,
                "image_request_count": state.image_request_count,
                "corrupt_image_fault_count": state.corrupt_image_fault_count,
                "empty_image_fault_count": state.empty_image_fault_count,
                "prediction_payload_count": state.prediction_payload_count,
                "duplicate_prediction_count": state.duplicate_prediction_count,
                "rejected_prediction_count": state.rejected_prediction_count,
                "order_violation_count": state.order_violation_count,
                "position": state.position_summary(),
            }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="HürGör mock yarışma sunucusu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--image-dir", default=None, help="Mock için gerçek görüntü klasörü")
    parser.add_argument("--video", default=None, help="Mock için yerel video dosyası")
    parser.add_argument("--translation-csv", default=None, help="Frame translation CSV dosyası")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="30 FPS videodan 7.5 FPS için 4",
    )
    parser.add_argument("--video-name", default=None)
    parser.add_argument("--healthy-frames", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    settings = MockSettings.from_env()
    if args.image_dir:
        settings = replace(settings, image_dir=args.image_dir)
    if args.video:
        settings = replace(settings, video_path=args.video)
    if args.translation_csv:
        settings = replace(settings, translation_csv_path=args.translation_csv)
    if args.frame_stride is not None:
        settings = replace(settings, frame_stride=max(1, args.frame_stride))
    if args.video_name:
        settings = replace(settings, video_name=args.video_name)
    if args.healthy_frames is not None:
        settings = replace(settings, healthy_frames=max(0, args.healthy_frames))
    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
