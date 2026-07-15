from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from hurgor.client import HeaderTokenAuth, build_http_auth, main, override_base_url_for_cli
from hurgor.config import ClientSettings, MockSettings
from hurgor.mock_server import TranslationTrack


def test_official_env_uses_evaluation_server_defaults(monkeypatch) -> None:
    monkeypatch.delenv("HURGOR_BASE_URL", raising=False)
    monkeypatch.delenv("HURGOR_FRAME_ENDPOINT", raising=False)
    monkeypatch.delenv("HURGOR_TRANSLATION_ENDPOINT", raising=False)
    monkeypatch.delenv("HURGOR_PREDICTION_ENDPOINT", raising=False)
    monkeypatch.delenv("HURGOR_PROGRESS_ENDPOINT", raising=False)
    monkeypatch.delenv("HURGOR_REFERENCE_ENDPOINT", raising=False)
    monkeypatch.delenv("HURGOR_TOKEN_ENDPOINT", raising=False)
    monkeypatch.delenv("HURGOR_API_CONTRACT", raising=False)
    monkeypatch.setenv("TEAM_NAME", "team")
    monkeypatch.setenv("PASSWORD", "secret")
    monkeypatch.setenv("EVALUATION_SERVER_URL", "http://official.example:1025/")
    monkeypatch.setenv("SESSION_NAME", "ONLINE_YARISMA_2026")

    settings = ClientSettings.from_env()

    assert settings.base_url == "http://official.example:1025"
    assert settings.frame_endpoint == "/frames/"
    assert settings.translation_endpoint == "/translation/"
    assert settings.prediction_endpoint == "/prediction/"
    assert settings.progress_endpoint == "/progress/"
    assert settings.reference_endpoint == "/reference/"
    assert settings.token_endpoint == "/auth/"
    assert settings.api_contract == "official"
    assert settings.team_name == "team"
    assert settings.password == "secret"
    assert settings.session_name == "ONLINE_YARISMA_2026"


def test_auth_token_env_is_read(monkeypatch) -> None:
    monkeypatch.setenv("HURGOR_AUTH_SCHEME", "token")
    monkeypatch.setenv("HURGOR_AUTH_TOKEN", "token-value")
    monkeypatch.setenv("HURGOR_TOKEN_ENDPOINT", "/auth/token/")

    settings = ClientSettings.from_env()

    assert settings.auth_scheme == "token"
    assert settings.auth_token == "token-value"
    assert settings.token_endpoint == "/auth/token/"


def test_visual_odometry_tuning_is_loaded_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("HURGOR_VO_MIN_CALIBRATION_SAMPLES", "30")
    monkeypatch.setenv("HURGOR_VO_MAX_CALIBRATION_SAMPLES", "300")
    monkeypatch.setenv("HURGOR_VO_VALIDATION_FRACTION", "0.25")
    monkeypatch.setenv("HURGOR_VO_MIN_VALIDATION_SAMPLES", "24")
    monkeypatch.setenv("HURGOR_VO_MAX_STEP_SKILL_RATIO", "0.8")
    monkeypatch.setenv("HURGOR_VO_MAX_TRAJECTORY_SKILL_RATIO", "0.7")
    monkeypatch.setenv("HURGOR_VO_MAX_BIAS_RATIO", "0.45")
    monkeypatch.setenv("HURGOR_VO_MIN_INLIERS", "36")
    monkeypatch.setenv("HURGOR_VO_MIN_INLIER_RATIO", "0.6")
    monkeypatch.setenv("HURGOR_VO_RANSAC_THRESHOLD_PX", "1.8")
    monkeypatch.setenv("HURGOR_VO_MAX_REPROJECTION_ERROR_PX", "2.2")
    monkeypatch.setenv("HURGOR_VO_MAX_STEP_M", "12")
    monkeypatch.setenv("HURGOR_VO_FALLBACK_DECAY", "0.7")
    monkeypatch.setenv("HURGOR_VO_PROJECTIVE_FEATURES", "false")

    settings = ClientSettings.from_env()

    assert settings.vo_min_calibration_samples == 30
    assert settings.vo_max_calibration_samples == 300
    assert settings.vo_validation_fraction == 0.25
    assert settings.vo_min_validation_samples == 24
    assert settings.vo_max_step_skill_ratio == 0.8
    assert settings.vo_max_trajectory_skill_ratio == 0.7
    assert settings.vo_max_bias_ratio == 0.45
    assert settings.vo_min_inliers == 36
    assert settings.vo_min_inlier_ratio == 0.6
    assert settings.vo_ransac_threshold_px == 1.8
    assert settings.vo_max_reprojection_error_px == 2.2
    assert settings.vo_max_step_m == 12
    assert settings.vo_fallback_decay == 0.7
    assert settings.vo_projective_features is False


def test_inference_circuit_breaker_tuning_is_loaded_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("HURGOR_INFERENCE_CIRCUIT_BREAKER_THRESHOLD", "3")
    monkeypatch.setenv("HURGOR_INFERENCE_CIRCUIT_BREAKER_COOLDOWN_FRAMES", "12")

    settings = ClientSettings.from_env()

    assert settings.inference_circuit_breaker_threshold == 3
    assert settings.inference_circuit_breaker_cooldown_frames == 12


def test_detector_threshold_tuning_is_loaded_from_environment(monkeypatch, tmp_path) -> None:
    profile = tmp_path / "thresholds.json"
    profile.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HURGOR_DETECTOR_THRESHOLDS_PATH", str(profile))
    monkeypatch.setenv("HURGOR_DETECTOR_CONFIDENCE", "0.2")
    monkeypatch.setenv("HURGOR_DETECTOR_IOU_THRESHOLD", "0.4")
    monkeypatch.setenv("HURGOR_DETECTOR_CROSS_CLASS_IOU_THRESHOLD", "0.92")

    settings = ClientSettings.from_env()

    assert settings.detector_thresholds_path == str(profile)
    assert settings.detector_confidence == 0.2
    assert settings.detector_iou_threshold == 0.4
    assert settings.detector_cross_class_iou_threshold == 0.92


def test_thermal_specialist_settings_are_loaded_from_environment(monkeypatch, tmp_path) -> None:
    model = tmp_path / "thermal.onnx"
    manifest = tmp_path / "thermal.json"
    model.write_bytes(b"thermal-model")
    manifest.write_text("{}", encoding="utf-8")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    monkeypatch.setenv("HURGOR_THERMAL_SPECIALIST_ONNX_PATH", str(model))
    monkeypatch.setenv("HURGOR_THERMAL_SPECIALIST_MANIFEST_PATH", str(manifest))
    monkeypatch.setenv("HURGOR_THERMAL_SPECIALIST_SHA256", model_sha)
    monkeypatch.setenv("HURGOR_THERMAL_SPECIALIST_CONFIDENCE", "0.18")
    monkeypatch.setenv("HURGOR_THERMAL_SPECIALIST_TIMEOUT_MS", "410")
    monkeypatch.setenv("HURGOR_THERMAL_SPECIALIST_SLOW_THRESHOLD_MS", "420")
    monkeypatch.setenv("HURGOR_THERMAL_SPECIALIST_COOLDOWN_FRAMES", "17")
    monkeypatch.setenv("HURGOR_THERMAL_SPECIALIST_COOLDOWN_SECONDS", "12")
    monkeypatch.setenv(
        "HURGOR_THERMAL_SPECIALIST_ONNX_PROVIDERS",
        "CPUExecutionProvider",
    )
    monkeypatch.setenv("HURGOR_THERMAL_SPECIALIST_ONNX_INTRA_OP_THREADS", "1")

    settings = ClientSettings.from_env()

    assert settings.thermal_specialist_onnx_path == str(model)
    assert settings.thermal_specialist_manifest_path == str(manifest)
    assert settings.thermal_specialist_sha256 == model_sha
    assert settings.thermal_specialist_confidence == 0.18
    assert settings.thermal_specialist_timeout_ms == 410
    assert settings.thermal_specialist_slow_threshold_ms == 420
    assert settings.thermal_specialist_cooldown_frames == 17
    assert settings.thermal_specialist_cooldown_seconds == 12
    assert settings.thermal_specialist_onnx_providers == ("CPUExecutionProvider",)
    assert settings.thermal_specialist_onnx_intra_op_threads == 1


def test_thermal_specialist_manifest_and_checksum_are_validated(tmp_path) -> None:
    model = tmp_path / "thermal.onnx"
    model.write_bytes(b"thermal-model")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = tmp_path / "thermal.json"
    manifest.write_text(
        json.dumps(
            {
                "classes": ["arac", "insan"],
                "sha256": model_sha,
                "output_format": "yolo_end2end",
            }
        ),
        encoding="utf-8",
    )

    ClientSettings(
        thermal_specialist_onnx_path=str(model),
        thermal_specialist_manifest_path=str(manifest),
        thermal_specialist_sha256=model_sha,
    ).validate()

    manifest.write_text(
        json.dumps({"classes": ["insan", "arac"], "sha256": model_sha}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"classes must be \[arac, insan\]"):
        ClientSettings(
            thermal_specialist_onnx_path=str(model),
            thermal_specialist_manifest_path=str(manifest),
        ).validate()


def test_detector_threshold_profile_hash_mismatch_fails_before_runtime(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"runtime-model")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    profile = tmp_path / "thresholds.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_model_sha256": "0" * 64,
                "classes": ["arac", "insan", "uap", "uai"],
                "thresholds": {"arac": 0.25, "insan": 0.15, "uap": 0.25, "uai": 0.25},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different model"):
        ClientSettings(
            yolo_onnx_path=str(model),
            model_sha256=model_sha,
            detector_thresholds_path=str(profile),
        ).validate()


def test_visual_odometry_tuning_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="VO max calibration samples"):
        ClientSettings(
            vo_min_calibration_samples=40,
            vo_max_calibration_samples=20,
        ).validate()
    with pytest.raises(ValueError, match="VO minimum inlier ratio"):
        ClientSettings(vo_min_inlier_ratio=1.2).validate()
    with pytest.raises(ValueError, match="VO fallback decay"):
        ClientSettings(vo_fallback_decay=1.0).validate()
    with pytest.raises(ValueError, match="VO validation fraction"):
        ClientSettings(vo_validation_fraction=1.0).validate()
    with pytest.raises(ValueError, match="VO validation skill ratios"):
        ClientSettings(vo_max_bias_ratio=1.0).validate()


def test_mock_server_error_status_must_be_5xx(monkeypatch) -> None:
    monkeypatch.setenv("HURGOR_MOCK_SERVER_ERROR_STATUS", "499")

    with pytest.raises(ValueError, match="between 500 and 599"):
        MockSettings.from_env()


def test_build_http_auth_prefers_token_in_auto_mode() -> None:
    settings = ClientSettings(
        team_name="team",
        password="secret",
        auth_scheme="auto",
        auth_token="token-value",
    )

    auth = build_http_auth(settings)

    assert isinstance(auth, HeaderTokenAuth)
    request = httpx.Request("GET", "http://example.test/")
    authorized_request = next(auth.auth_flow(request))
    assert authorized_request.headers["Authorization"] == "Token token-value"


def test_client_main_exits_nonzero_when_pipeline_has_fatal_error(monkeypatch) -> None:
    import hurgor.threaded_pipeline as threaded_pipeline

    class FatalPipeline:
        def __init__(self, settings):
            self.settings = settings

        def stop(self):
            return None

        def run(self, max_frames=None):
            del max_frames
            return SimpleNamespace(
                frames_submitted=0,
                elapsed=0.1,
                fps=0.0,
                sla_misses=0,
                degraded_frames=0,
                fatal_error="forced fatal",
            )

    monkeypatch.setattr(threaded_pipeline, "ThreadedEdgePipeline", FatalPipeline)
    monkeypatch.setattr(sys, "argv", ["hurgor-client", "--max-frames", "1"])
    monkeypatch.delenv("EVALUATION_SERVER_URL", raising=False)
    monkeypatch.setenv("HURGOR_API_CONTRACT", "local")
    monkeypatch.setenv("HURGOR_AUTH_SCHEME", "none")

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1


def test_hurgor_endpoint_env_overrides_official_defaults(monkeypatch) -> None:
    monkeypatch.setenv("TEAM_NAME", "team")
    monkeypatch.setenv("PASSWORD", "secret")
    monkeypatch.setenv("EVALUATION_SERVER_URL", "http://official.example:1025/")
    monkeypatch.setenv("HURGOR_FRAME_ENDPOINT", "/custom/get")
    monkeypatch.setenv("HURGOR_PREDICTION_ENDPOINT", "/custom/post")

    settings = ClientSettings.from_env()

    assert settings.frame_endpoint == "/custom/get"
    assert settings.prediction_endpoint == "/custom/post"


def test_cli_base_url_override_restores_local_endpoints_without_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("HURGOR_FRAME_ENDPOINT", raising=False)
    monkeypatch.delenv("HURGOR_PREDICTION_ENDPOINT", raising=False)
    official_settings = ClientSettings(
        base_url="http://official.example:1025",
        frame_endpoint="/frames/",
        translation_endpoint="/translation/",
        prediction_endpoint="/prediction/",
        progress_endpoint="/progress/",
        reference_endpoint="/reference/",
        api_contract="official",
        team_name="team",
        password="secret",
    )

    settings = override_base_url_for_cli(official_settings, "http://127.0.0.1:5126/")

    assert settings.base_url == "http://127.0.0.1:5126"
    assert settings.frame_endpoint == "/api/frames/next"
    assert settings.translation_endpoint is None
    assert settings.prediction_endpoint == "/api/predictions"
    assert settings.progress_endpoint == "/api/status"
    assert settings.reference_endpoint is None
    assert settings.api_contract == "local"
    assert settings.auth_scheme == "none"
    assert settings.auth_token is None
    assert settings.token_endpoint is None


def test_cli_base_url_override_keeps_explicit_endpoint_env(monkeypatch) -> None:
    monkeypatch.setenv("HURGOR_FRAME_ENDPOINT", "/official-frame")
    monkeypatch.setenv("HURGOR_PREDICTION_ENDPOINT", "/official-post")
    official_settings = ClientSettings(
        base_url="http://official.example:1025",
        frame_endpoint="/official-frame",
        prediction_endpoint="/official-post",
    )

    settings = override_base_url_for_cli(official_settings, "http://127.0.0.1:5126/")

    assert settings.base_url == "http://127.0.0.1:5126"
    assert settings.frame_endpoint == "/official-frame"
    assert settings.prediction_endpoint == "/official-post"


def test_cli_private_server_override_preserves_official_contract(monkeypatch) -> None:
    for name in (
        "HURGOR_FRAME_ENDPOINT",
        "HURGOR_TRANSLATION_ENDPOINT",
        "HURGOR_PREDICTION_ENDPOINT",
        "HURGOR_PROGRESS_ENDPOINT",
        "HURGOR_REFERENCE_ENDPOINT",
        "HURGOR_API_CONTRACT",
    ):
        monkeypatch.delenv(name, raising=False)
    official_settings = ClientSettings(
        base_url="http://official.example:1025",
        frame_endpoint="/frames/",
        translation_endpoint="/translation/",
        prediction_endpoint="/prediction/",
        progress_endpoint="/progress/",
        reference_endpoint="/reference/",
        api_contract="official",
        auth_scheme="auto",
    )

    settings = override_base_url_for_cli(official_settings, "http://192.168.50.25:5000")

    assert settings.base_url == "http://192.168.50.25:5000"
    assert settings.api_contract == "official"
    assert settings.frame_endpoint == "/frames/"
    assert settings.translation_endpoint == "/translation/"
    assert settings.prediction_endpoint == "/prediction/"
    assert settings.progress_endpoint == "/progress/"
    assert settings.reference_endpoint == "/reference/"
    assert settings.auth_scheme == "auto"


def test_translation_track_stride(tmp_path: Path) -> None:
    csv_path = tmp_path / "translation.csv"
    csv_path.write_text(
        "\n".join(
            [
                "translation_x,translation_y,translation_z,frame_numbers",
                "0,0,10,frame_000000",
                "1,2,11,frame_000001",
                "2,4,12,frame_000002",
                "3,6,13,frame_000003",
                "4,8,14,frame_000004",
            ]
        ),
        encoding="utf-8",
    )

    track = TranslationTrack(str(csv_path), frame_stride=2)

    assert track.frame_count == 3
    assert track.translation(0) == (0.0, 0.0, 10.0)
    assert track.translation(1) == (2.0, 4.0, 12.0)
    assert track.translation(2) == (4.0, 8.0, 14.0)


def test_translation_track_preserves_optional_orientation(tmp_path: Path) -> None:
    csv_path = tmp_path / "translation-orientation.csv"
    csv_path.write_text(
        "translation_x,translation_y,translation_z,frame_numbers,"
        "orientation_x,orientation_y,orientation_z,orientation_w\n"
        "0,0,0,frame_000000,0,0,0,1\n"
        "1,2,3,frame_000001,0,0,0.3826834324,0.9238795325\n",
        encoding="utf-8",
    )

    track = TranslationTrack(str(csv_path))

    assert track.orientation(0) == (0.0, 0.0, 0.0, 1.0)
    assert track.orientation(1) == pytest.approx((0.0, 0.0, 0.3826834324, 0.9238795325))
