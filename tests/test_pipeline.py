from __future__ import annotations

import math

import httpx
import pytest

from hurgor.client import CompetitionAPI, PermanentAPIError, RetryExhausted, SessionComplete
from hurgor.config import ClientSettings, MockSettings
from hurgor.mock_server import create_app
from hurgor.models import DetectedTranslation, Prediction


def _client_settings() -> ClientSettings:
    return ClientSettings(
        base_url="http://testserver",
        http_timeout_seconds=1.0,
        max_retries=1,
        retry_base_seconds=0.001,
        error_cooldown_seconds=0.001,
    )


def _prediction(frame_url: str, identifier: int) -> list[dict]:
    prediction = Prediction(
        id=identifier,
        user="http://testserver/users/1/",
        frame=frame_url,
        detected_objects=[],
        detected_translations=[
            DetectedTranslation(translation_x=0, translation_y=0, translation_z=0)
        ],
        detected_undefined_objects=[],
    )
    return [prediction.canonical_dict()]


@pytest.mark.asyncio
async def test_repeated_get_returns_same_frame_until_post() -> None:
    app = create_app(MockSettings(frame_count=2, healthy_frames=1))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        first = (await client.get("/api/frames/next")).json()[0]
        second = (await client.get("/api/frames/next")).json()[0]
        assert first["url"] == second["url"]
        status = (await client.get("/api/status")).json()
        assert status["next_index"] == 0
        assert status["outstanding_index"] == 0


@pytest.mark.asyncio
async def test_mock_faults_are_deterministic_and_do_not_advance_frame() -> None:
    app = create_app(MockSettings(frame_count=3, healthy_frames=1, corrupt_every=2, empty_every=2))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        first = (await client.get("/api/frames/next")).json()[0]
        response = await client.post("/api/predictions", json=_prediction(first["url"], 1))
        assert response.status_code == 200

        assert (await client.get("/api/frames/next")).json() == []
        second = (await client.get("/api/frames/next")).json()[0]
        corrupt_image = await client.get(second["image_url"])
        assert corrupt_image.content == b"corrupt-jpeg"
        status = (await client.get("/api/status")).json()
        assert status["next_index"] == 1
        assert status["outstanding_index"] == 1


@pytest.mark.asyncio
async def test_official_mock_counts_empty_metadata_empty_image_and_exact_post() -> None:
    settings = MockSettings(frame_count=1, empty_every=1, empty_image_every=1)
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        token = (
            await client.post(
                "/auth/",
                data={"username": settings.mock_username, "password": settings.mock_password},
            )
        ).json()["token"]
        headers = {"Authorization": f"Token {token}"}

        assert (await client.get("/frames/", headers=headers)).json() == []
        frame = (await client.get("/frames/", headers=headers)).json()[0]
        image = await client.get(frame["image_url"])
        assert image.content == b""
        prediction = {
            "frame": frame["url"],
            "detected_objects": [],
            "detected_translations": [
                {"translation_x": 0.0, "translation_y": 0.0, "translation_z": 0.0}
            ],
            "reference_predictions": [],
        }
        response = await client.post("/prediction/", headers=headers, json=prediction)
        status = (await client.get("/api/status")).json()

    assert response.status_code == 200
    assert status["empty_metadata_fault_count"] == 1
    assert status["empty_image_fault_count"] == 1
    assert status["frame_issue_count"] == 1
    assert status["frame_response_count"] == 1
    assert status["prediction_payload_count"] == 1
    assert status["accepted_count"] == 1
    assert status["duplicate_prediction_count"] == 0
    assert status["rejected_prediction_count"] == 0


@pytest.mark.asyncio
async def test_official_mock_injects_and_counts_real_http_500() -> None:
    settings = MockSettings(frame_count=1, server_error_every=1, server_error_status=500)
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        token = (
            await client.post(
                "/auth/",
                data={"username": settings.mock_username, "password": settings.mock_password},
            )
        ).json()["token"]
        response = await client.get("/progress/", headers={"Authorization": f"Token {token}"})
        status = (await client.get("/api/status")).json()

    assert response.status_code == 500
    assert status["injected_5xx_count"] == 1


@pytest.mark.asyncio
async def test_mock_server_history_is_bounded() -> None:
    frame_count = 300
    app = create_app(MockSettings(frame_count=frame_count, healthy_frames=20))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        for index in range(frame_count):
            frame = (await client.get("/api/frames/next")).json()[0]
            response = await client.post("/api/predictions", json=_prediction(frame["url"], index))
            assert response.status_code == 200
        status = (await client.get("/api/status")).json()

    assert status["accepted_count"] == frame_count
    assert status["recent_state_size"] == 100


@pytest.mark.asyncio
async def test_competition_api_classifies_permanent_error() -> None:
    app = create_app(MockSettings(frame_count=1))
    settings = ClientSettings(base_url="http://testserver", frame_endpoint="/missing")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        api = CompetitionAPI(settings, client)
        with pytest.raises(PermanentAPIError):
            await api.fetch_frame()


@pytest.mark.asyncio
async def test_competition_api_rejects_empty_metadata() -> None:
    app = create_app(MockSettings(frame_count=2, empty_every=1))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        api = CompetitionAPI(_client_settings(), client)
        with pytest.raises(RetryExhausted):
            await api.fetch_frame()


@pytest.mark.asyncio
async def test_competition_api_treats_official_empty_list_as_session_complete() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/frames/":
            return httpx.Response(200, json=[], request=request)
        assert request.url.path == "/progress/"
        return httpx.Response(
            200,
            json={"frame_index": 2250, "total_frames": 2250, "completed": True},
            request=request,
        )

    settings = ClientSettings(
        base_url="http://official.test",
        frame_endpoint="/frames/",
        translation_endpoint="/translation/",
        progress_endpoint="/progress/",
        api_contract="official",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=settings.base_url,
    ) as client:
        with pytest.raises(SessionComplete):
            await CompetitionAPI(settings, client).fetch_frame()


@pytest.mark.asyncio
async def test_competition_api_retries_official_empty_list_during_live_session() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/frames/":
            return httpx.Response(200, json=[], request=request)
        assert request.url.path == "/progress/"
        return httpx.Response(
            200,
            json={"frame_index": 156, "total_frames": 2250, "completed": False},
            request=request,
        )

    settings = ClientSettings(
        base_url="http://official.test",
        frame_endpoint="/frames/",
        translation_endpoint="/translation/",
        progress_endpoint="/progress/",
        api_contract="official",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=settings.base_url,
    ) as client:
        with pytest.raises(RetryExhausted, match="before session completion"):
            await CompetitionAPI(settings, client).fetch_frame()


@pytest.mark.asyncio
async def test_official_frame_fetch_merges_translation_without_overwriting_frame_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/frames/":
            return httpx.Response(
                200,
                json=[
                    {
                        "url": "http://official.test/frames/4/",
                        "image_url": "/session/frame_000000.webp",
                        "video_name": "official-session",
                        "session": "http://official.test/session/1/",
                    }
                ],
            )
        if request.url.path == "/translation/":
            return httpx.Response(
                200,
                json=[
                    {
                        "url": "http://official.test/translation/2/",
                        "frame": "http://official.test/frames/4/",
                        "image_url": "/session/frame_000000.webp",
                        "video_name": "official-session",
                        "session": "http://official.test/session/1/",
                        "translation_x": "0.044",
                        "translation_y": "0.003",
                        "translation_z": "-0.001",
                        "health_status": "1",
                        "orientation_x": "0.0",
                        "orientation_y": "0.0",
                        "orientation_z": "0.3826834324",
                        "orientation_w": "0.9238795325",
                    }
                ],
            )
        return httpx.Response(404)

    settings = ClientSettings(
        base_url="http://official.test",
        frame_endpoint="/frames/",
        translation_endpoint="/translation/",
        api_contract="official",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://official.test",
    ) as client:
        frame = await CompetitionAPI(settings, client).fetch_frame()

    assert frame.url == "http://official.test/frames/4/"
    assert frame.image_url == "/session/frame_000000.webp"
    assert frame.gps_health_status == 1
    assert frame.reference_translation == (0.044, 0.003, -0.001)
    assert math.isclose(frame.orientation_heading_rad or 0.0, math.pi / 4.0)
