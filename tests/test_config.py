from __future__ import annotations

from pathlib import Path

import httpx

from hurgor.client import HeaderTokenAuth, build_http_auth, override_base_url_for_cli
from hurgor.config import ClientSettings
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
