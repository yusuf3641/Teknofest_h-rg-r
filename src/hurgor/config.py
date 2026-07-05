from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class ClientSettings:
    base_url: str = "http://127.0.0.1:5000"
    frame_endpoint: str = "/api/frames/next"
    prediction_endpoint: str = "/api/predictions"
    user_url: str = "/users/1/"
    session_url: str = "/session/1/"
    http_timeout_seconds: float = 2.0
    max_retries: int = 3
    retry_base_seconds: float = 0.1
    error_cooldown_seconds: float = 0.25
    inference_timeout_seconds: float = 0.9
    sla_seconds: float = 1.0
    log_every: int = 25
    queue_maxsize: int = 3
    thread_join_timeout_seconds: float = 10.0
    degrade_threshold_ms: float = 800.0
    degrade_after_frames: int = 5
    recover_threshold_ms: float = 250.0
    recover_after_frames: int = 10
    log_file: str = "system.log"
    yolo_onnx_path: str | None = None
    reference_images_dir: str | None = None
    camera_fx: float = 1000.0
    camera_fy: float = 1000.0
    camera_altitude_m: float = 10.0

    @classmethod
    def from_env(cls) -> ClientSettings:
        defaults = cls()
        return cls(
            base_url=os.getenv("HURGOR_BASE_URL", defaults.base_url).rstrip("/"),
            frame_endpoint=os.getenv("HURGOR_FRAME_ENDPOINT", defaults.frame_endpoint),
            prediction_endpoint=os.getenv(
                "HURGOR_PREDICTION_ENDPOINT", defaults.prediction_endpoint
            ),
            user_url=os.getenv("HURGOR_USER_URL", defaults.user_url),
            session_url=os.getenv("HURGOR_SESSION_URL", defaults.session_url),
            http_timeout_seconds=float(
                os.getenv("HURGOR_HTTP_TIMEOUT_SECONDS", defaults.http_timeout_seconds)
            ),
            max_retries=int(os.getenv("HURGOR_MAX_RETRIES", defaults.max_retries)),
            retry_base_seconds=float(
                os.getenv("HURGOR_RETRY_BASE_SECONDS", defaults.retry_base_seconds)
            ),
            error_cooldown_seconds=float(
                os.getenv(
                    "HURGOR_ERROR_COOLDOWN_SECONDS",
                    defaults.error_cooldown_seconds,
                )
            ),
            inference_timeout_seconds=float(
                os.getenv(
                    "HURGOR_INFERENCE_TIMEOUT_SECONDS",
                    defaults.inference_timeout_seconds,
                )
            ),
            sla_seconds=float(os.getenv("HURGOR_SLA_SECONDS", defaults.sla_seconds)),
            log_every=max(1, int(os.getenv("HURGOR_LOG_EVERY", defaults.log_every))),
            queue_maxsize=max(1, int(os.getenv("HURGOR_QUEUE_MAXSIZE", defaults.queue_maxsize))),
            thread_join_timeout_seconds=max(
                1.0,
                float(
                    os.getenv(
                        "HURGOR_THREAD_JOIN_TIMEOUT_SECONDS",
                        defaults.thread_join_timeout_seconds,
                    )
                ),
            ),
            degrade_threshold_ms=max(
                1.0,
                float(
                    os.getenv(
                        "HURGOR_DEGRADE_THRESHOLD_MS",
                        defaults.degrade_threshold_ms,
                    )
                ),
            ),
            degrade_after_frames=max(
                1,
                int(
                    os.getenv(
                        "HURGOR_DEGRADE_AFTER_FRAMES",
                        defaults.degrade_after_frames,
                    )
                ),
            ),
            recover_threshold_ms=max(
                1.0,
                float(
                    os.getenv(
                        "HURGOR_RECOVER_THRESHOLD_MS",
                        defaults.recover_threshold_ms,
                    )
                ),
            ),
            recover_after_frames=max(
                1,
                int(
                    os.getenv(
                        "HURGOR_RECOVER_AFTER_FRAMES",
                        defaults.recover_after_frames,
                    )
                ),
            ),
            log_file=os.getenv("HURGOR_LOG_FILE", defaults.log_file),
            yolo_onnx_path=os.getenv("HURGOR_YOLO_ONNX_PATH") or None,
            reference_images_dir=os.getenv("HURGOR_REFERENCE_IMAGES_DIR") or None,
            camera_fx=max(1.0, float(os.getenv("HURGOR_CAMERA_FX", defaults.camera_fx))),
            camera_fy=max(1.0, float(os.getenv("HURGOR_CAMERA_FY", defaults.camera_fy))),
            camera_altitude_m=max(
                0.01,
                float(os.getenv("HURGOR_CAMERA_ALTITUDE_M", defaults.camera_altitude_m)),
            ),
        )


@dataclass(frozen=True, slots=True)
class MockSettings:
    frame_count: int = 2250
    healthy_frames: int = 450
    user_url: str = "/users/1/"
    session_url: str = "/session/1/"
    corrupt_every: int = 0
    empty_every: int = 0
    get_delay_ms: int = 0
    post_delay_ms: int = 0
    video_path: str | None = None

    @classmethod
    def from_env(cls) -> MockSettings:
        defaults = cls()
        return cls(
            frame_count=max(
                0,
                int(os.getenv("HURGOR_MOCK_FRAME_COUNT", defaults.frame_count)),
            ),
            healthy_frames=max(
                0,
                int(os.getenv("HURGOR_MOCK_HEALTHY_FRAMES", defaults.healthy_frames)),
            ),
            user_url=os.getenv("HURGOR_USER_URL", defaults.user_url),
            session_url=os.getenv("HURGOR_SESSION_URL", defaults.session_url),
            corrupt_every=max(
                0,
                int(os.getenv("HURGOR_MOCK_CORRUPT_EVERY", defaults.corrupt_every)),
            ),
            empty_every=max(
                0,
                int(os.getenv("HURGOR_MOCK_EMPTY_EVERY", defaults.empty_every)),
            ),
            get_delay_ms=max(
                0,
                int(os.getenv("HURGOR_MOCK_GET_DELAY_MS", defaults.get_delay_ms)),
            ),
            post_delay_ms=max(
                0,
                int(os.getenv("HURGOR_MOCK_POST_DELAY_MS", defaults.post_delay_ms)),
            ),
            video_path=os.getenv("HURGOR_MOCK_VIDEO_PATH") or None,
        )
