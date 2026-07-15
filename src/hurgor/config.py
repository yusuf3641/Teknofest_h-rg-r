from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class ClientSettings:
    base_url: str = "http://127.0.0.1:5000"
    frame_endpoint: str = "/api/frames/next"
    translation_endpoint: str | None = None
    prediction_endpoint: str = "/api/predictions"
    progress_endpoint: str = "/api/status"
    reference_endpoint: str | None = None
    user_url: str = "/users/1/"
    session_url: str = "/session/1/"
    team_name: str | None = None
    password: str | None = None
    session_name: str | None = None
    auth_scheme: str = "auto"
    auth_token: str | None = None
    token_endpoint: str | None = None
    api_contract: str = "local"
    http_timeout_seconds: float = 2.0
    max_retries: int = 3
    retry_base_seconds: float = 0.1
    error_cooldown_seconds: float = 0.25
    inference_timeout_seconds: float = 0.9
    inference_startup_timeout_seconds: float = 15.0
    inference_circuit_breaker_threshold: int = 2
    inference_circuit_breaker_cooldown_frames: int = 30
    sla_seconds: float = 1.0
    target_fps: float = 0.0
    log_every: int = 25
    queue_maxsize: int = 3
    thread_join_timeout_seconds: float = 10.0
    degrade_threshold_ms: float = 800.0
    degrade_after_frames: int = 5
    recover_threshold_ms: float = 250.0
    recover_after_frames: int = 10
    log_file: str = "system.log"
    yolo_onnx_path: str | None = None
    model_manifest_path: str | None = None
    model_sha256: str | None = None
    detector_thresholds_path: str | None = None
    detector_confidence: float = 0.25
    detector_iou_threshold: float = 0.45
    detector_cross_class_iou_threshold: float = 0.90
    thermal_specialist_onnx_path: str | None = None
    thermal_specialist_manifest_path: str | None = None
    thermal_specialist_sha256: str | None = None
    thermal_specialist_confidence: float = 0.25
    thermal_specialist_timeout_ms: float = 200.0
    thermal_specialist_slow_threshold_ms: float = 180.0
    thermal_specialist_cooldown_frames: int = 30
    thermal_specialist_cooldown_seconds: float = 20.0
    thermal_specialist_onnx_providers: tuple[str, ...] = ()
    thermal_specialist_onnx_intra_op_threads: int = 1
    allow_noop_detector: bool = False
    inference_process_enabled: bool = True
    enable_experimental_vo: bool = False
    multiprocessing_start_method: str = "spawn"
    onnx_providers: tuple[str, ...] = ()
    onnx_intra_op_threads: int = 0
    onnx_inter_op_threads: int = 1
    opencv_num_threads: int = 1
    diagnostics_dir: str = "diagnostics"
    metrics_file: str = "logs/metrics.jsonl"
    reference_images_dir: str | None = None
    reference_cache_dir: str = "artifacts/references"
    camera_fx: float = 1000.0
    camera_fy: float = 1000.0
    camera_altitude_m: float = 10.0
    vo_min_calibration_samples: int = 20
    vo_max_calibration_samples: int = 450
    vo_validation_fraction: float = 0.20
    vo_min_validation_samples: int = 20
    vo_max_step_skill_ratio: float = 0.85
    vo_max_trajectory_skill_ratio: float = 0.75
    vo_max_bias_ratio: float = 0.50
    vo_min_inliers: int = 24
    vo_min_inlier_ratio: float = 0.45
    vo_ransac_threshold_px: float = 2.5
    vo_max_reprojection_error_px: float = 3.0
    vo_max_step_m: float = 20.0
    vo_fallback_decay: float = 0.85
    vo_projective_features: bool = True

    @classmethod
    def from_env(cls) -> ClientSettings:
        defaults = cls()
        official_base_url = os.getenv("EVALUATION_SERVER_URL")
        team_name = os.getenv("TEAM_NAME") or None
        password = os.getenv("PASSWORD") or None
        session_name = os.getenv("SESSION_NAME") or None
        # Credentials alone must never redirect a local run to the official contract.
        # The official mode is selected by the committee URL or an explicit contract.
        requested_contract = os.getenv("HURGOR_API_CONTRACT", "").strip().lower()
        official_mode = bool(official_base_url) or requested_contract == "official"
        return cls(
            base_url=os.getenv(
                "HURGOR_BASE_URL",
                official_base_url or defaults.base_url,
            ).rstrip("/"),
            frame_endpoint=os.getenv(
                "HURGOR_FRAME_ENDPOINT",
                "/frames/" if official_mode else defaults.frame_endpoint,
            ),
            translation_endpoint=os.getenv(
                "HURGOR_TRANSLATION_ENDPOINT",
                "/translation/" if official_mode else defaults.translation_endpoint or "",
            )
            or None,
            prediction_endpoint=os.getenv(
                "HURGOR_PREDICTION_ENDPOINT",
                "/prediction/" if official_mode else defaults.prediction_endpoint,
            ),
            progress_endpoint=os.getenv(
                "HURGOR_PROGRESS_ENDPOINT",
                "/progress/" if official_mode else defaults.progress_endpoint,
            ),
            reference_endpoint=os.getenv(
                "HURGOR_REFERENCE_ENDPOINT",
                "/reference/" if official_mode else defaults.reference_endpoint or "",
            )
            or None,
            user_url=os.getenv("HURGOR_USER_URL", defaults.user_url),
            session_url=os.getenv("HURGOR_SESSION_URL", defaults.session_url),
            team_name=team_name,
            password=password,
            session_name=session_name,
            auth_scheme=os.getenv("HURGOR_AUTH_SCHEME", defaults.auth_scheme).lower(),
            auth_token=os.getenv("HURGOR_AUTH_TOKEN") or None,
            token_endpoint=os.getenv(
                "HURGOR_TOKEN_ENDPOINT",
                "/auth/" if official_mode else "",
            )
            or None,
            api_contract=os.getenv(
                "HURGOR_API_CONTRACT",
                "official" if official_mode else defaults.api_contract,
            ).lower(),
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
            inference_startup_timeout_seconds=max(
                0.1,
                float(
                    os.getenv(
                        "HURGOR_INFERENCE_STARTUP_TIMEOUT_SECONDS",
                        defaults.inference_startup_timeout_seconds,
                    )
                ),
            ),
            inference_circuit_breaker_threshold=max(
                1,
                int(
                    os.getenv(
                        "HURGOR_INFERENCE_CIRCUIT_BREAKER_THRESHOLD",
                        defaults.inference_circuit_breaker_threshold,
                    )
                ),
            ),
            inference_circuit_breaker_cooldown_frames=max(
                1,
                int(
                    os.getenv(
                        "HURGOR_INFERENCE_CIRCUIT_BREAKER_COOLDOWN_FRAMES",
                        defaults.inference_circuit_breaker_cooldown_frames,
                    )
                ),
            ),
            sla_seconds=float(os.getenv("HURGOR_SLA_SECONDS", defaults.sla_seconds)),
            target_fps=max(0.0, float(os.getenv("HURGOR_TARGET_FPS", defaults.target_fps))),
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
            model_manifest_path=os.getenv("HURGOR_MODEL_MANIFEST_PATH") or None,
            model_sha256=(os.getenv("HURGOR_MODEL_SHA256") or "").lower() or None,
            detector_thresholds_path=os.getenv("HURGOR_DETECTOR_THRESHOLDS_PATH") or None,
            detector_confidence=float(
                os.getenv("HURGOR_DETECTOR_CONFIDENCE", defaults.detector_confidence)
            ),
            detector_iou_threshold=float(
                os.getenv("HURGOR_DETECTOR_IOU_THRESHOLD", defaults.detector_iou_threshold)
            ),
            detector_cross_class_iou_threshold=float(
                os.getenv(
                    "HURGOR_DETECTOR_CROSS_CLASS_IOU_THRESHOLD",
                    defaults.detector_cross_class_iou_threshold,
                )
            ),
            thermal_specialist_onnx_path=(
                os.getenv("HURGOR_THERMAL_SPECIALIST_ONNX_PATH") or None
            ),
            thermal_specialist_manifest_path=(
                os.getenv("HURGOR_THERMAL_SPECIALIST_MANIFEST_PATH") or None
            ),
            thermal_specialist_sha256=(
                (os.getenv("HURGOR_THERMAL_SPECIALIST_SHA256") or "").lower() or None
            ),
            thermal_specialist_confidence=float(
                os.getenv(
                    "HURGOR_THERMAL_SPECIALIST_CONFIDENCE",
                    defaults.thermal_specialist_confidence,
                )
            ),
            thermal_specialist_timeout_ms=max(
                1.0,
                float(
                    os.getenv(
                        "HURGOR_THERMAL_SPECIALIST_TIMEOUT_MS",
                        defaults.thermal_specialist_timeout_ms,
                    )
                ),
            ),
            thermal_specialist_slow_threshold_ms=max(
                1.0,
                float(
                    os.getenv(
                        "HURGOR_THERMAL_SPECIALIST_SLOW_THRESHOLD_MS",
                        defaults.thermal_specialist_slow_threshold_ms,
                    )
                ),
            ),
            thermal_specialist_cooldown_frames=max(
                1,
                int(
                    os.getenv(
                        "HURGOR_THERMAL_SPECIALIST_COOLDOWN_FRAMES",
                        defaults.thermal_specialist_cooldown_frames,
                    )
                ),
            ),
            thermal_specialist_cooldown_seconds=max(
                0.0,
                float(
                    os.getenv(
                        "HURGOR_THERMAL_SPECIALIST_COOLDOWN_SECONDS",
                        defaults.thermal_specialist_cooldown_seconds,
                    )
                ),
            ),
            thermal_specialist_onnx_providers=tuple(
                item.strip()
                for item in os.getenv(
                    "HURGOR_THERMAL_SPECIALIST_ONNX_PROVIDERS",
                    ",".join(defaults.thermal_specialist_onnx_providers),
                ).split(",")
                if item.strip()
            ),
            thermal_specialist_onnx_intra_op_threads=max(
                0,
                int(
                    os.getenv(
                        "HURGOR_THERMAL_SPECIALIST_ONNX_INTRA_OP_THREADS",
                        defaults.thermal_specialist_onnx_intra_op_threads,
                    )
                ),
            ),
            allow_noop_detector=_env_bool("HURGOR_ALLOW_NOOP_DETECTOR", False),
            inference_process_enabled=_env_bool("HURGOR_INFERENCE_PROCESS_ENABLED", True),
            enable_experimental_vo=_env_bool("HURGOR_ENABLE_EXPERIMENTAL_VO", False),
            multiprocessing_start_method=os.getenv(
                "HURGOR_MULTIPROCESSING_START_METHOD",
                defaults.multiprocessing_start_method,
            ).lower(),
            onnx_providers=tuple(
                item.strip()
                for item in os.getenv("HURGOR_ONNX_PROVIDERS", "").split(",")
                if item.strip()
            ),
            onnx_intra_op_threads=max(
                0,
                int(
                    os.getenv(
                        "HURGOR_ONNX_INTRA_OP_THREADS",
                        defaults.onnx_intra_op_threads,
                    )
                ),
            ),
            onnx_inter_op_threads=max(
                1,
                int(
                    os.getenv(
                        "HURGOR_ONNX_INTER_OP_THREADS",
                        defaults.onnx_inter_op_threads,
                    )
                ),
            ),
            opencv_num_threads=max(
                1,
                int(os.getenv("HURGOR_OPENCV_NUM_THREADS", defaults.opencv_num_threads)),
            ),
            diagnostics_dir=os.getenv("HURGOR_DIAGNOSTICS_DIR", defaults.diagnostics_dir),
            metrics_file=os.getenv("HURGOR_METRICS_FILE", defaults.metrics_file),
            reference_images_dir=os.getenv("HURGOR_REFERENCE_IMAGES_DIR") or None,
            reference_cache_dir=os.getenv(
                "HURGOR_REFERENCE_CACHE_DIR", defaults.reference_cache_dir
            ),
            camera_fx=max(1.0, float(os.getenv("HURGOR_CAMERA_FX", defaults.camera_fx))),
            camera_fy=max(1.0, float(os.getenv("HURGOR_CAMERA_FY", defaults.camera_fy))),
            camera_altitude_m=max(
                0.01,
                float(os.getenv("HURGOR_CAMERA_ALTITUDE_M", defaults.camera_altitude_m)),
            ),
            vo_min_calibration_samples=max(
                6,
                int(
                    os.getenv(
                        "HURGOR_VO_MIN_CALIBRATION_SAMPLES",
                        defaults.vo_min_calibration_samples,
                    )
                ),
            ),
            vo_max_calibration_samples=max(
                6,
                int(
                    os.getenv(
                        "HURGOR_VO_MAX_CALIBRATION_SAMPLES",
                        defaults.vo_max_calibration_samples,
                    )
                ),
            ),
            vo_validation_fraction=float(
                os.getenv("HURGOR_VO_VALIDATION_FRACTION", defaults.vo_validation_fraction)
            ),
            vo_min_validation_samples=max(
                1,
                int(
                    os.getenv(
                        "HURGOR_VO_MIN_VALIDATION_SAMPLES",
                        defaults.vo_min_validation_samples,
                    )
                ),
            ),
            vo_max_step_skill_ratio=float(
                os.getenv(
                    "HURGOR_VO_MAX_STEP_SKILL_RATIO",
                    defaults.vo_max_step_skill_ratio,
                )
            ),
            vo_max_trajectory_skill_ratio=float(
                os.getenv(
                    "HURGOR_VO_MAX_TRAJECTORY_SKILL_RATIO",
                    defaults.vo_max_trajectory_skill_ratio,
                )
            ),
            vo_max_bias_ratio=float(
                os.getenv("HURGOR_VO_MAX_BIAS_RATIO", defaults.vo_max_bias_ratio)
            ),
            vo_min_inliers=max(
                6,
                int(os.getenv("HURGOR_VO_MIN_INLIERS", defaults.vo_min_inliers)),
            ),
            vo_min_inlier_ratio=float(
                os.getenv("HURGOR_VO_MIN_INLIER_RATIO", defaults.vo_min_inlier_ratio)
            ),
            vo_ransac_threshold_px=max(
                0.1,
                float(
                    os.getenv(
                        "HURGOR_VO_RANSAC_THRESHOLD_PX",
                        defaults.vo_ransac_threshold_px,
                    )
                ),
            ),
            vo_max_reprojection_error_px=max(
                0.1,
                float(
                    os.getenv(
                        "HURGOR_VO_MAX_REPROJECTION_ERROR_PX",
                        defaults.vo_max_reprojection_error_px,
                    )
                ),
            ),
            vo_max_step_m=max(
                0.01,
                float(os.getenv("HURGOR_VO_MAX_STEP_M", defaults.vo_max_step_m)),
            ),
            vo_fallback_decay=float(
                os.getenv("HURGOR_VO_FALLBACK_DECAY", defaults.vo_fallback_decay)
            ),
            vo_projective_features=_env_bool(
                "HURGOR_VO_PROJECTIVE_FEATURES",
                defaults.vo_projective_features,
            ),
        )

    @property
    def is_official(self) -> bool:
        return self.api_contract == "official"

    def validate(self, *, for_runtime: bool = False) -> None:
        """Fail closed on unsafe competition configuration.

        Dataclass construction remains lightweight for unit tests and adapters. Runtime
        entrypoints call this method before opening network connections.
        """

        errors: list[str] = []
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            errors.append("base_url must be an absolute HTTP(S) URL")
        if self.api_contract not in {"local", "official"}:
            errors.append("api_contract must be local or official")
        if self.auth_scheme not in {"auto", "basic", "token", "bearer", "none", "off", "disabled"}:
            errors.append("unsupported auth_scheme")
        if (
            self.http_timeout_seconds <= 0
            or self.inference_timeout_seconds <= 0
            or self.inference_startup_timeout_seconds <= 0
        ):
            errors.append("HTTP, inference and inference startup timeouts must be positive")
        if (
            self.inference_circuit_breaker_threshold < 1
            or self.inference_circuit_breaker_cooldown_frames < 1
        ):
            errors.append("inference circuit-breaker values must be positive")
        if self.sla_seconds <= 0 or self.thread_join_timeout_seconds <= 0:
            errors.append("SLA and thread join timeout must be positive")
        if self.target_fps < 0:
            errors.append("target_fps must not be negative")
        if self.max_retries < 0 or self.retry_base_seconds < 0 or self.error_cooldown_seconds < 0:
            errors.append("retry values must not be negative")
        if self.queue_maxsize < 1:
            errors.append("queue_maxsize must be at least one")
        if not 0 < self.detector_confidence <= 1:
            errors.append("detector confidence must be in (0, 1]")
        if not 0 < self.detector_iou_threshold <= 1:
            errors.append("detector IoU threshold must be in (0, 1]")
        if not 0 < self.detector_cross_class_iou_threshold <= 1:
            errors.append("detector cross-class IoU threshold must be in (0, 1]")
        if not 0 < self.thermal_specialist_confidence <= 1:
            errors.append("thermal specialist confidence must be in (0, 1]")
        if (
            self.thermal_specialist_timeout_ms <= 0
            or self.thermal_specialist_timeout_ms >= self.inference_timeout_seconds * 1000
            or self.thermal_specialist_slow_threshold_ms <= 0
            or self.thermal_specialist_cooldown_frames < 1
            or self.thermal_specialist_cooldown_seconds < 0
        ):
            errors.append(
                "thermal specialist load-shedding values must be positive and "
                "the specialist timeout must be below the inference timeout"
            )
        if self.multiprocessing_start_method not in {"spawn", "fork", "forkserver"}:
            errors.append("unsupported multiprocessing start method")
        if self.vo_max_calibration_samples < self.vo_min_calibration_samples:
            errors.append("VO max calibration samples must be at least the minimum")
        if self.vo_max_calibration_samples < (
            self.vo_min_calibration_samples + self.vo_min_validation_samples
        ):
            errors.append(
                "VO max calibration samples must fit both training and validation samples"
            )
        if self.vo_min_calibration_samples < 6 or self.vo_min_inliers < 6:
            errors.append("VO calibration samples and inliers must be at least six")
        if not 0 < self.vo_validation_fraction < 1:
            errors.append("VO validation fraction must be in (0, 1)")
        if not all(
            0 < value < 1
            for value in (
                self.vo_max_step_skill_ratio,
                self.vo_max_trajectory_skill_ratio,
                self.vo_max_bias_ratio,
            )
        ):
            errors.append("VO validation skill ratios must be in (0, 1)")
        if not 0 < self.vo_min_inlier_ratio <= 1:
            errors.append("VO minimum inlier ratio must be in (0, 1]")
        if (
            self.vo_ransac_threshold_px <= 0
            or self.vo_max_reprojection_error_px <= 0
            or self.vo_max_step_m <= 0
        ):
            errors.append("VO pixel and metric safety thresholds must be positive")
        if not 0 <= self.vo_fallback_decay < 1:
            errors.append("VO fallback decay must be in [0, 1)")

        if self.is_official:
            if not os.getenv("EVALUATION_SERVER_URL") and for_runtime:
                errors.append("official mode requires EVALUATION_SERVER_URL")
            if not self.team_name or not self.password:
                errors.append("official mode requires TEAM_NAME and PASSWORD")
            if not self.session_name:
                errors.append("official mode requires SESSION_NAME")
            if for_runtime and not self.yolo_onnx_path and not self.allow_noop_detector:
                errors.append(
                    "official mode requires HURGOR_YOLO_ONNX_PATH; "
                    "set HURGOR_ALLOW_NOOP_DETECTOR=true only for an explicit dry test"
                )

        runtime_model_sha: str | None = None
        if self.yolo_onnx_path:
            model_path = Path(self.yolo_onnx_path).expanduser()
            if not model_path.is_file():
                errors.append(f"model file does not exist: {model_path}")
            elif model_path.suffix.lower() != ".onnx":
                errors.append("runtime detector model must be ONNX")
            else:
                runtime_model_sha = _sha256_file(model_path)
                if self.model_sha256 and runtime_model_sha != self.model_sha256:
                    errors.append("model SHA-256 does not match HURGOR_MODEL_SHA256")
        if self.model_manifest_path:
            try:
                manifest = json.loads(
                    Path(self.model_manifest_path).expanduser().read_text(encoding="utf-8")
                )
                classes = manifest.get("classes")
                if classes != ["arac", "insan", "uap", "uai"]:
                    errors.append("model manifest classes must be [arac, insan, uap, uai]")
                manifest_sha = str(manifest.get("sha256", "")).lower()
                if self.model_sha256 and manifest_sha and manifest_sha != self.model_sha256:
                    errors.append("model manifest checksum disagrees with configuration")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"model manifest cannot be read: {exc}")
        elif for_runtime and self.is_official and self.yolo_onnx_path:
            errors.append("official mode requires HURGOR_MODEL_MANIFEST_PATH")
        if self.detector_thresholds_path:
            thresholds_path = Path(self.detector_thresholds_path).expanduser()
            if not thresholds_path.is_file():
                errors.append(f"detector threshold profile does not exist: {thresholds_path}")
            elif runtime_model_sha is None:
                errors.append("detector threshold profile requires a valid ONNX model")
            else:
                try:
                    from .detector_calibration import load_detector_thresholds

                    load_detector_thresholds(
                        str(thresholds_path),
                        runtime_model_sha256=runtime_model_sha,
                        class_names=["arac", "insan", "uap", "uai"],
                    )
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(f"detector threshold profile is invalid: {exc}")

        specialist_metadata_present = any(
            (self.thermal_specialist_manifest_path, self.thermal_specialist_sha256)
        )
        if specialist_metadata_present and not self.thermal_specialist_onnx_path:
            errors.append("thermal specialist metadata requires an ONNX model")
        if self.thermal_specialist_onnx_path:
            specialist_path = Path(self.thermal_specialist_onnx_path).expanduser()
            specialist_runtime_sha: str | None = None
            if not specialist_path.is_file():
                errors.append(f"thermal specialist model file does not exist: {specialist_path}")
            elif specialist_path.suffix.lower() != ".onnx":
                errors.append("thermal specialist runtime model must be ONNX")
            else:
                specialist_runtime_sha = _sha256_file(specialist_path)
                if (
                    self.thermal_specialist_sha256
                    and specialist_runtime_sha != self.thermal_specialist_sha256
                ):
                    errors.append(
                        "thermal specialist SHA-256 does not match configuration"
                    )
            if not self.thermal_specialist_manifest_path:
                errors.append("thermal specialist requires a model manifest")
            else:
                try:
                    specialist_manifest = json.loads(
                        Path(self.thermal_specialist_manifest_path)
                        .expanduser()
                        .read_text(encoding="utf-8")
                    )
                    if specialist_manifest.get("classes") != ["arac", "insan"]:
                        errors.append(
                            "thermal specialist manifest classes must be [arac, insan]"
                        )
                    manifest_sha = str(specialist_manifest.get("sha256", "")).lower()
                    if not manifest_sha:
                        errors.append("thermal specialist manifest requires SHA-256")
                    elif specialist_runtime_sha and manifest_sha != specialist_runtime_sha:
                        errors.append(
                            "thermal specialist manifest checksum does not match model"
                        )
                    if (
                        self.thermal_specialist_sha256
                        and manifest_sha
                        and manifest_sha != self.thermal_specialist_sha256
                    ):
                        errors.append(
                            "thermal specialist manifest checksum disagrees with configuration"
                        )
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(f"thermal specialist manifest cannot be read: {exc}")

        if errors:
            raise ValueError("invalid HürGör configuration: " + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class MockSettings:
    frame_count: int = 2250
    healthy_frames: int = 450
    user_url: str = "/users/1/"
    session_url: str = "/session/1/"
    video_name: str = "hurgor_mock_v1"
    modality: Literal["rgb", "thermal"] = "rgb"
    corrupt_every: int = 0
    empty_every: int = 0
    empty_image_every: int = 0
    get_delay_ms: int = 0
    post_delay_ms: int = 0
    image_dir: str | None = None
    video_path: str | None = None
    translation_csv_path: str | None = None
    frame_stride: int = 1
    mock_username: str = "hurgor_test"
    mock_password: str = "test_password"
    token_expire_after_requests: int = 0
    rate_limit_every: int = 0
    server_error_every: int = 0
    server_error_status: int = 503
    retry_after_seconds: float = 0.01

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
            video_name=os.getenv("HURGOR_MOCK_VIDEO_NAME", defaults.video_name),
            modality=_env_choice(
                "HURGOR_MOCK_MODALITY",
                defaults.modality,
                {"rgb", "thermal"},
            ),
            corrupt_every=max(
                0,
                int(os.getenv("HURGOR_MOCK_CORRUPT_EVERY", defaults.corrupt_every)),
            ),
            empty_every=max(
                0,
                int(os.getenv("HURGOR_MOCK_EMPTY_EVERY", defaults.empty_every)),
            ),
            empty_image_every=max(
                0,
                int(
                    os.getenv(
                        "HURGOR_MOCK_EMPTY_IMAGE_EVERY",
                        defaults.empty_image_every,
                    )
                ),
            ),
            get_delay_ms=max(
                0,
                int(os.getenv("HURGOR_MOCK_GET_DELAY_MS", defaults.get_delay_ms)),
            ),
            post_delay_ms=max(
                0,
                int(os.getenv("HURGOR_MOCK_POST_DELAY_MS", defaults.post_delay_ms)),
            ),
            image_dir=os.getenv("HURGOR_MOCK_IMAGE_DIR") or None,
            video_path=os.getenv("HURGOR_MOCK_VIDEO_PATH") or None,
            translation_csv_path=os.getenv("HURGOR_MOCK_TRANSLATION_CSV_PATH") or None,
            frame_stride=max(
                1,
                int(os.getenv("HURGOR_MOCK_FRAME_STRIDE", defaults.frame_stride)),
            ),
            mock_username=os.getenv("HURGOR_MOCK_USERNAME", defaults.mock_username),
            mock_password=os.getenv("HURGOR_MOCK_PASSWORD", defaults.mock_password),
            token_expire_after_requests=max(
                0,
                int(
                    os.getenv(
                        "HURGOR_MOCK_TOKEN_EXPIRE_AFTER_REQUESTS",
                        defaults.token_expire_after_requests,
                    )
                ),
            ),
            rate_limit_every=max(
                0,
                int(os.getenv("HURGOR_MOCK_RATE_LIMIT_EVERY", defaults.rate_limit_every)),
            ),
            server_error_every=max(
                0,
                int(os.getenv("HURGOR_MOCK_SERVER_ERROR_EVERY", defaults.server_error_every)),
            ),
            server_error_status=_env_int_range(
                "HURGOR_MOCK_SERVER_ERROR_STATUS",
                defaults.server_error_status,
                minimum=500,
                maximum=599,
            ),
            retry_after_seconds=max(
                0.0,
                float(os.getenv("HURGOR_MOCK_RETRY_AFTER_SECONDS", defaults.retry_after_seconds)),
            ),
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in allowed:
        return normalized
    raise ValueError(f"{name} must be one of {sorted(allowed)}")


def _env_int_range(name: str, default: int, *, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, default))
    if minimum <= value <= maximum:
        return value
    raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
