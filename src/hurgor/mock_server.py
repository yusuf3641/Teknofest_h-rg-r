from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import threading
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Protocol
from urllib.parse import urljoin

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from pydantic import ValidationError

from .config import MockSettings
from .models import Prediction


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
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def metadata(self, request: Request, index: int) -> dict[str, Any]:
        root = str(request.base_url).rstrip("/")
        healthy = index < self.settings.healthy_frames
        translation = self.frame_source.translation(index)
        if translation is None:
            if healthy:
                translation = (round(index * 0.02, 4), round(index * 0.01, 4), 10.0)
            else:
                translation = ("NaN", "NaN", "NaN")
        elif not healthy:
            # In unhealthy windows the competition may hide/poison GPS values.
            # Keeping NaN here exercises the client's visual-odometry fallback path.
            translation = ("NaN", "NaN", "NaN")
        return {
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

    def render(self, index: int) -> bytes:
        image = Image.new("RGB", (640, 480), color=(15, 23, 42))
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 80, 560, 400), outline=(34, 211, 238), width=4)
        draw.text((100, 110), f"HURGOR MOCK FRAME {index:06d}", fill=(248, 250, 252))
        draw.text((100, 150), "TEK KARE / IDEMPOTENT API", fill=(148, 163, 184))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82)
        return buffer.getvalue()

    def close(self) -> None:
        return None

    def translation(self, index: int) -> tuple[float, float, float] | None:
        del index
        return None


@dataclass(frozen=True, slots=True)
class TranslationRow:
    translation_x: float
    translation_y: float
    translation_z: float
    frame_number: str


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
            for row in reader:
                rows.append(
                    TranslationRow(
                        translation_x=float(row["translation_x"]),
                        translation_y=float(row["translation_y"]),
                        translation_z=float(row["translation_z"]),
                        frame_number=row["frame_numbers"],
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


class VideoFrameSource:
    def __init__(
        self,
        video_path: str,
        *,
        frame_stride: int = 1,
        translation_track: TranslationTrack | None = None,
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


def create_app(settings: MockSettings | None = None) -> FastAPI:
    configured = settings or MockSettings.from_env()
    source: FrameSource
    if configured.video_path:
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
        )
        configured = replace(configured, frame_count=source.frame_count)
    else:
        source = SyntheticFrameSource(configured.frame_count)
    state = MockState(configured, source)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            state.frame_source.close()

    app = FastAPI(title="HürGör Mock Yarışma Sunucusu", version="0.1.0", lifespan=lifespan)
    app.state.mock = state

    @app.get("/api/frames/next")
    async def get_next_frame(request: Request) -> Response:
        async with state.lock:
            if state.outstanding_index is not None:
                index = state.outstanding_index
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
                    return JSONResponse([])
                state.outstanding_index = index

            payload = state.metadata(request, index)
        if state.settings.get_delay_ms:
            await asyncio.sleep(state.settings.get_delay_ms / 1000)
        return JSONResponse([payload])

    @app.get("/media/frame_{index}.jpg")
    async def get_frame_image(index: int) -> Response:
        async with state.lock:
            if index < 0 or index >= state.settings.frame_count:
                raise HTTPException(status_code=404, detail="frame does not exist")
            corrupt = (
                state.settings.corrupt_every > 0 and (index + 1) % state.settings.corrupt_every == 0
            )
        content = b"corrupt-jpeg" if corrupt else state.frame_source.render(index)
        return Response(content=content, media_type="image/jpeg")

    @app.post("/api/predictions")
    async def post_prediction(
        request: Request,
        payload: Annotated[Any, Body()],
    ) -> dict[str, Any]:
        if state.settings.post_delay_ms:
            await asyncio.sleep(state.settings.post_delay_ms / 1000)
        if not isinstance(payload, list) or len(payload) != 1:
            raise HTTPException(
                status_code=422,
                detail="official contract requires a one-item prediction list",
            )
        raw_prediction: Any = payload[0]
        try:
            prediction = Prediction.model_validate(raw_prediction)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        root = str(request.base_url).rstrip("/")
        expected_user = urljoin(f"{root}/", state.settings.user_url)
        if prediction.user != expected_user:
            raise HTTPException(
                status_code=422,
                detail=f"expected user {expected_user}",
            )
        expected_class_prefix = f"{root}/classes/"
        class_url_mismatch = any(
            not item.cls.startswith(expected_class_prefix)
            for item in prediction.detected_objects
        )
        if class_url_mismatch:
            raise HTTPException(
                status_code=422,
                detail=f"class URLs must start with {expected_class_prefix}",
            )

        digest = _prediction_digest(prediction)
        async with state.lock:
            if prediction.frame == state.last_accepted_frame:
                return {
                    "accepted": True,
                    "duplicate": True,
                    "same_payload": digest == state.last_accepted_digest,
                }
            if state.outstanding_index is None:
                raise HTTPException(status_code=409, detail="no outstanding frame")
            expected = f"/frames/{state.outstanding_index}/"
            if not prediction.frame.endswith(expected):
                raise HTTPException(
                    status_code=409,
                    detail=f"expected frame suffix {expected}",
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
            }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="HürGör mock yarışma sunucusu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
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
