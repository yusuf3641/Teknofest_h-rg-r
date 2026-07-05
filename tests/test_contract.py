from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from hurgor.config import MockSettings
from hurgor.mock_server import create_app
from hurgor.models import (
    MAX_SAFE_JSON_INTEGER,
    DetectedObject,
    DetectedTranslation,
    DetectedUndefinedObject,
    Prediction,
    class_url_from_id,
    prediction_id_from_frame_url,
)


@pytest.mark.asyncio
async def test_page_25_get_contract_is_a_list_with_health_status() -> None:
    app = create_app(
        MockSettings(
            frame_count=1,
            healthy_frames=1,
            user_url="/users/4/",
            session_url="/session/2/",
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/frames/next")

    assert response.status_code == 200
    assert response.json() == [
        {
            "url": "http://testserver/frames/0/",
            "image_url": "/media/frame_000000.jpg",
            "video_name": "hurgor_mock_v1",
            "session": "http://testserver/session/2/",
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 10.0,
            "health_status": 1,
        }
    ]
    assert "gps_health_status" not in response.json()[0]


@pytest.mark.asyncio
async def test_page_27_post_contract_matches_official_shape() -> None:
    settings = MockSettings(
        frame_count=1,
        healthy_frames=1,
        user_url="/users/4/",
        session_url="/session/2/",
    )
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        frame = (await client.get("/api/frames/next")).json()[0]
        prediction = Prediction(
            id=22246,
            user="http://testserver/users/4/",
            frame=frame["url"],
            detected_objects=[
                DetectedObject.from_class_id(
                    1,
                    base_url="http://testserver",
                    landing_status="-1",
                    motion_status="-1",
                    top_left_x=262.87,
                    top_left_y=734.47,
                    bottom_right_x=405.2,
                    bottom_right_y=847.3,
                )
            ],
            detected_translations=[
                DetectedTranslation(
                    translation_x=0.02,
                    translation_y=0.01,
                    translation_z=0.03,
                )
            ],
            detected_undefined_objects=[
                DetectedUndefinedObject(
                    object_id=1,
                    top_left_x=262.87,
                    top_left_y=734.47,
                    bottom_right_x=405.2,
                    bottom_right_y=847.3,
                )
            ],
        )
        payload = [prediction.canonical_dict()]

        assert payload == [
            {
                "id": 22246,
                "user": "http://testserver/users/4/",
                "frame": "http://testserver/frames/0/",
                "detected_objects": [
                    {
                        "top_left_x": 262.87,
                        "top_left_y": 734.47,
                        "bottom_right_x": 405.2,
                        "bottom_right_y": 847.3,
                        "cls": "http://testserver/classes/1/",
                        "landing_status": "-1",
                        "motion_status": "-1",
                    }
                ],
                "detected_translations": [
                    {
                        "translation_x": 0.02,
                        "translation_y": 0.01,
                        "translation_z": 0.03,
                    }
                ],
                "detected_undefined_objects": [
                    {
                        "top_left_x": 262.87,
                        "top_left_y": 734.47,
                        "bottom_right_x": 405.2,
                        "bottom_right_y": 847.3,
                        "object_id": 1,
                    }
                ],
            }
        ]
        assert "session" not in payload[0]

        direct_object_response = await client.post("/api/predictions", json=payload[0])
        assert direct_object_response.status_code == 422

        response = await client.post("/api/predictions", json=payload)
        assert response.status_code == 200
        assert response.json() == {"accepted": True, "duplicate": False}


def test_prediction_id_is_strict_deterministic_and_json_safe() -> None:
    frame_url = "http://127.0.0.25:5000/frames/4000/"
    first = prediction_id_from_frame_url(frame_url)
    second = prediction_id_from_frame_url(frame_url)

    assert type(first) is int
    assert first == second
    assert 1 <= first <= MAX_SAFE_JSON_INTEGER

    with pytest.raises(ValidationError):
        Prediction.model_validate(
            {
                "id": str(first),
                "user": "http://127.0.0.25:5000/users/4/",
                "frame": frame_url,
                "detected_objects": [],
                "detected_translations": [
                    {
                        "translation_x": 0.0,
                        "translation_y": 0.0,
                        "translation_z": 0.0,
                    }
                ],
                "detected_undefined_objects": [],
            }
        )


def test_class_url_is_derived_from_server_base_url() -> None:
    assert (
        class_url_from_id("http://127.0.0.25:5000/api", 3)
        == "http://127.0.0.25:5000/classes/3/"
    )
    with pytest.raises(ValidationError):
        DetectedObject(
            cls="3",
            top_left_x=1,
            top_left_y=1,
            bottom_right_x=2,
            bottom_right_y=2,
        )
