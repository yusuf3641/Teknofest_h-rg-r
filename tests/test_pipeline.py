from __future__ import annotations

import httpx
import pytest

from hurgor.client import CompetitionAPI, PermanentAPIError, RetryExhausted
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
