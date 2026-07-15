from __future__ import annotations

import io
import json
import math
import time

import httpx
import pytest
from PIL import Image

from hurgor.camera import select_camera_profile
from hurgor.client import CompetitionAPI, PermanentAPIError
from hurgor.config import ClientSettings, MockSettings
from hurgor.inference import PipelineInferenceEngine
from hurgor.metrics import current_rss_mb
from hurgor.mock_server import DirectoryFrameSource, SyntheticFrameSource, create_app
from hurgor.models import FrameMetadata, Prediction, ReferenceDefinition
from hurgor.references import ReferenceManager
from hurgor.vision import OpticalFlowSE3Estimator, ORBReferenceMatcher
from hurgor.watchdog import (
    InferenceReply,
    InferenceTimeoutError,
    InferenceWatchdog,
    InferenceWorkerError,
)


def _sleep_worker(settings, input_queue, output_queue) -> None:
    del settings
    output_queue.put(InferenceReply("__ready__", None, {}, None))
    input_queue.get()
    time.sleep(2)


def _crash_worker(settings, input_queue, output_queue) -> None:
    del settings, input_queue, output_queue
    raise SystemExit(9)


def _delayed_ready_worker(settings, input_queue, output_queue) -> None:
    del settings
    time.sleep(0.2)
    output_queue.put(InferenceReply("__ready__", None, {}, None))
    job = input_queue.get()
    frame = FrameMetadata.model_validate(job.frame)
    prediction = PipelineInferenceEngine().fallback(frame, job.user_url)
    output_queue.put(
        InferenceReply(
            job.job_id,
            prediction.canonical_dict(),
            {"inference_ms": 1.0},
            None,
        )
    )


def _recovering_worker(settings, input_queue, output_queue) -> None:
    del settings
    output_queue.put(InferenceReply("__ready__", None, {}, None))
    while True:
        job = input_queue.get()
        if job is None:
            return
        frame = FrameMetadata.model_validate(job.frame)
        frame_index = int(frame.url.rstrip("/").rsplit("/", 1)[-1])
        if frame_index == 1:
            time.sleep(2)
            continue
        previous_count = int((job.recovery_state or {}).get("count", 0))
        prediction = PipelineInferenceEngine().fallback(frame, job.user_url)
        output_queue.put(
            InferenceReply(
                job.job_id,
                prediction.canonical_dict(),
                {"inference_ms": 1.0},
                None,
                {"count": previous_count + 1},
            )
        )


def _frame() -> FrameMetadata:
    return FrameMetadata.model_validate(
        {
            "url": "http://test/frames/0/",
            "image_url": "/media/frame_000000.jpg",
            "video_name": "test",
            "session": "http://test/session/1/",
            "translation_x": 0,
            "translation_y": 0,
            "translation_z": 10,
            "health_status": 1,
        }
    )


def _process_method() -> str:
    return "spawn"


def test_current_rss_reports_the_live_process_tree() -> None:
    assert current_rss_mb() > 0


def test_credentials_without_evaluation_url_do_not_select_official(monkeypatch) -> None:
    for name in (
        "EVALUATION_SERVER_URL",
        "HURGOR_BASE_URL",
        "HURGOR_API_CONTRACT",
        "SESSION_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TEAM_NAME", "local-team")
    monkeypatch.setenv("PASSWORD", "secret")

    settings = ClientSettings.from_env()

    assert settings.api_contract == "local"
    assert settings.frame_endpoint == "/api/frames/next"


def test_official_runtime_without_model_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("EVALUATION_SERVER_URL", "http://official.test:1025/")
    settings = ClientSettings(
        base_url="http://official.test:1025",
        api_contract="official",
        team_name="team",
        password="secret",
        session_name="ONLINE_YARISMA_2026",
    )

    with pytest.raises(ValueError, match="HURGOR_YOLO_ONNX_PATH"):
        settings.validate(for_runtime=True)


def test_mock_falls_back_to_synthetic_frames_when_video_assets_are_missing(tmp_path) -> None:
    settings = MockSettings(
        frame_count=7,
        video_path=str(tmp_path / "missing.mp4"),
        translation_csv_path=str(tmp_path / "missing.csv"),
    )

    app = create_app(settings)

    assert isinstance(app.state.mock.frame_source, SyntheticFrameSource)
    assert app.state.mock.settings.frame_count == 7


@pytest.mark.asyncio
async def test_mock_serves_a_bounded_real_image_directory(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        Image.new("RGB", (32, 24), color=color).save(image_dir / f"{index:02d}.png")
    settings = MockSettings(frame_count=2, image_dir=str(image_dir))

    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        frame = (await client.get("/api/frames/next")).json()[0]
        image_response = await client.get(frame["image_url"])

    assert isinstance(app.state.mock.frame_source, DirectoryFrameSource)
    assert app.state.mock.settings.frame_count == 2
    with Image.open(io.BytesIO(image_response.content)) as image:
        assert image.mode == "RGB"
        assert image.size == (32, 24)


@pytest.mark.asyncio
async def test_local_mock_can_serve_thermal_frames() -> None:
    settings = MockSettings(
        frame_count=1,
        modality="thermal",
        video_name="Ornek-Veri-2-Termal",
    )
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        frame = (await client.get("/api/frames/next")).json()[0]
        image_response = await client.get(frame["image_url"])

    assert frame["video_name"] == "Ornek-Veri-2-Termal"
    with Image.open(io.BytesIO(image_response.content)) as image:
        assert image.mode == "L"
        assert image.size == (640, 512)


def test_inference_timeout_retires_worker_before_deferred_restart() -> None:
    settings = ClientSettings(
        inference_timeout_seconds=0.05,
        multiprocessing_start_method=_process_method(),
    )
    watchdog = InferenceWatchdog(settings, worker_target=_sleep_worker)
    try:
        # Model/process startup is governed by a separate startup timeout and can
        # legitimately take over a second on macOS.  Measure only the per-frame
        # watchdog path here.
        watchdog.start()
        started = time.monotonic()
        with pytest.raises(InferenceTimeoutError):
            watchdog.infer_timed(_frame(), b"image", "http://test/users/1/")
        elapsed = time.monotonic() - started
        assert watchdog.timeout_count == 1
        assert watchdog.restart_count >= 1
        assert watchdog.process is None
        assert watchdog.ready is False
        assert elapsed < 1.0

        watchdog.start()
        assert watchdog.process is not None and watchdog.process.is_alive()
    finally:
        watchdog.close()


def test_worker_startup_time_is_not_charged_to_per_frame_timeout() -> None:
    settings = ClientSettings(
        inference_timeout_seconds=0.05,
        # Process spawn can exceed two seconds on a loaded macOS CI/desktop host.
        inference_startup_timeout_seconds=5.0,
        multiprocessing_start_method=_process_method(),
    )
    watchdog = InferenceWatchdog(settings, worker_target=_delayed_ready_worker)
    try:
        outcome = watchdog.infer_timed(_frame(), b"image", "http://test/users/1/")

        assert outcome.prediction.frame == _frame().url
        assert watchdog.ready is True
        assert watchdog.timeout_count == 0
        assert watchdog.restart_count == 0
    finally:
        watchdog.close()


def test_inference_crash_is_detected_and_restarted() -> None:
    settings = ClientSettings(
        inference_timeout_seconds=3.0,
        multiprocessing_start_method=_process_method(),
    )
    watchdog = InferenceWatchdog(settings, worker_target=_crash_worker)
    try:
        with pytest.raises(InferenceWorkerError):
            watchdog.infer_timed(_frame(), b"image", "http://test/users/1/")
        assert watchdog.restart_count >= 1
    finally:
        watchdog.close()


def test_worker_preserves_structured_corrupt_frame_error_type() -> None:
    settings = ClientSettings(
        allow_noop_detector=True,
        inference_timeout_seconds=3.0,
        inference_startup_timeout_seconds=5.0,
        multiprocessing_start_method=_process_method(),
    )
    watchdog = InferenceWatchdog(settings)
    try:
        with pytest.raises(InferenceWorkerError) as caught:
            watchdog.infer_timed(
                _frame(),
                b"not-an-image",
                "http://test/users/1/",
            )

        assert caught.value.error_type == "CorruptFrameError"
    finally:
        watchdog.close()


def test_watchdog_restores_position_checkpoint_after_timeout_restart() -> None:
    settings = ClientSettings(
        inference_timeout_seconds=0.05,
        inference_startup_timeout_seconds=5.0,
        multiprocessing_start_method=_process_method(),
    )
    watchdog = InferenceWatchdog(settings, worker_target=_recovering_worker)
    try:
        watchdog.infer_timed(_frame(), b"frame-0", "http://test/users/1/")
        with pytest.raises(InferenceTimeoutError):
            watchdog.infer_timed(
                _frame().model_copy(update={"url": "http://test/frames/1/"}),
                b"frame-1",
                "http://test/users/1/",
            )
        watchdog.infer_timed(
            _frame().model_copy(update={"url": "http://test/frames/2/"}),
            b"frame-2",
            "http://test/users/1/",
        )

        assert watchdog.recovery_state == {"count": 2}
        assert watchdog.state_restore_count == 1
        assert watchdog.restart_count >= 1
    finally:
        watchdog.close()


def test_default_worker_restores_odometry_before_next_frame_budget_starts() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (96, 64), (90, 120, 150)).save(buffer, format="JPEG")
    settings = ClientSettings(
        allow_noop_detector=True,
        enable_experimental_vo=True,
        inference_timeout_seconds=0.5,
        inference_startup_timeout_seconds=5.0,
        multiprocessing_start_method=_process_method(),
    )
    watchdog = InferenceWatchdog(settings)
    try:
        watchdog.infer_timed(_frame(), buffer.getvalue(), "http://test/users/1/")
        assert watchdog.recovery_state is not None

        watchdog._restart_worker("test_checkpoint_restore")
        watchdog.start()

        assert watchdog.state_restore_count == 1
        assert watchdog.ready is True
        watchdog.infer_timed(
            _frame().model_copy(update={"url": "http://test/frames/1/"}),
            buffer.getvalue(),
            "http://test/users/1/",
        )
    finally:
        watchdog.close()


@pytest.mark.asyncio
async def test_official_mock_contract_auth_frame_translation_and_prediction() -> None:
    settings = MockSettings(frame_count=2, healthy_frames=1)
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        unauthorized = await client.get("/frames/")
        assert unauthorized.status_code == 401
        login = await client.post(
            "/auth/",
            data={"username": settings.mock_username, "password": settings.mock_password},
        )
        token = login.json()["token"]
        headers = {"Authorization": f"Token {token}"}
        progress = await client.get("/progress/", headers=headers)
        classes = await client.get("/classes/", headers=headers)
        references = await client.get("/reference/", headers=headers)
        frame = (await client.get("/frames/", headers=headers)).json()[0]
        translation = (await client.get("/translation/", headers=headers)).json()[0]
        response = await client.post(
            "/prediction/",
            headers=headers,
            json={
                "frame": frame["url"],
                "detected_objects": [],
                "detected_translations": [
                    {"translation_x": "0", "translation_y": "0", "translation_z": "0"}
                ],
                "reference_predictions": [
                    {
                        "reference": "http://testserver/reference/1/",
                        "top_left_x": "1",
                        "top_left_y": "2",
                        "bottom_right_x": "3",
                        "bottom_right_y": "4",
                    }
                ],
            },
        )

    assert progress.json()["total_frames"] == 2
    assert [item["id"] for item in classes.json()] == [1, 2, 3, 4, 5]
    assert len(references.json()) == 1
    assert translation["frame"] == frame["url"]
    assert (
        translation["translation_x"],
        translation["translation_y"],
        translation["translation_z"],
    ) == (0.0, 0.0, 0.0)
    assert response.status_code == 200
    assert response.json() == {"accepted": True, "duplicate": False}


@pytest.mark.asyncio
async def test_reference_manager_downloads_hashes_and_filters_active_window(tmp_path) -> None:
    mock_settings = MockSettings(frame_count=3)
    app = create_app(mock_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        token = (
            await client.post(
                "/auth/",
                data={
                    "username": mock_settings.mock_username,
                    "password": mock_settings.mock_password,
                },
            )
        ).json()["token"]
        client.headers["Authorization"] = f"Token {token}"
        api = CompetitionAPI(
            ClientSettings(
                base_url="http://testserver",
                api_contract="official",
                reference_endpoint="/reference/",
            ),
            client,
        )
        manager = ReferenceManager(str(tmp_path))
        assets = await manager.bootstrap(api)

    assert len(assets) == 1
    assert len(assets[0].sha256) == 64
    assert (tmp_path / "references_manifest.json").is_file()
    assert len(manager.active("http://testserver/frames/1/")) == 1
    assert manager.active("http://testserver/frames/3/") == []


@pytest.mark.asyncio
async def test_reference_matcher_emits_detected_undefined_object_from_mock_reference(
    tmp_path,
) -> None:
    mock_settings = MockSettings(frame_count=3)
    app = create_app(mock_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        token = (
            await client.post(
                "/auth/",
                data={
                    "username": mock_settings.mock_username,
                    "password": mock_settings.mock_password,
                },
            )
        ).json()["token"]
        client.headers["Authorization"] = f"Token {token}"
        api = CompetitionAPI(
            ClientSettings(
                base_url="http://testserver",
                api_contract="official",
                reference_endpoint="/reference/",
            ),
            client,
        )
        manager = ReferenceManager(str(tmp_path))
        await manager.bootstrap(api)
        frame_content = (await client.get("/media/frame_000000.jpg")).content

    matcher = ORBReferenceMatcher(str(tmp_path))
    frame = FrameMetadata.model_validate(
        {
            "url": "http://testserver/frames/0/",
            "image_url": "/media/frame_000000.jpg",
            "video_name": "reference-test",
            "session": "http://testserver/session/1/",
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 10.0,
            "gps_health_status": 1,
        }
    )
    matches = matcher.match(Image.open(io.BytesIO(frame_content)).convert("RGB"), frame)

    assert len(matches) == 1
    assert matches[0].object_id == 1
    assert matches[0].bottom_right_x > matches[0].top_left_x
    assert matches[0].bottom_right_y > matches[0].top_left_y


@pytest.mark.asyncio
async def test_client_refreshes_once_after_401() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.headers.get("Authorization") != "Token fresh":
            return httpx.Response(401, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    class FakeAuthManager:
        def refresh(self, stale_token: str | None = None) -> str:
            assert stale_token == "stale"
            return "fresh"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
        headers={"Authorization": "Token stale"},
    ) as client:
        api = CompetitionAPI(ClientSettings(max_retries=0), client, FakeAuthManager())
        response = await api._request("GET", "/progress/")

    assert response.status_code == 200
    assert calls == 2


@pytest.mark.asyncio
async def test_client_retries_after_connection_drop() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("injected connection drop", request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    settings = ClientSettings(max_retries=1, retry_base_seconds=0.0)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
    ) as client:
        api = CompetitionAPI(settings, client)
        response = await api._request("GET", "/progress/")

    assert response.status_code == 200
    assert calls == 2
    assert api.retry_count == 1


@pytest.mark.asyncio
async def test_translation_frame_mismatch_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/frames/":
            return httpx.Response(
                200,
                json=[
                    {
                        "url": "http://test/frames/1/",
                        "image_url": "/frame.webp",
                        "video_name": "video",
                        "session": "http://test/session/1/",
                    }
                ],
            )
        return httpx.Response(
            200,
            json=[
                {
                    "frame": "http://test/frames/2/",
                    "translation_x": 0,
                    "translation_y": 0,
                    "translation_z": 0,
                    "health_status": 1,
                }
            ],
        )

    settings = ClientSettings(
        base_url="http://test",
        frame_endpoint="/frames/",
        translation_endpoint="/translation/",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as client:
        with pytest.raises(Exception, match="translation frame mismatch"):
            await CompetitionAPI(settings, client).fetch_frame()


@pytest.mark.asyncio
async def test_permanent_422_writes_sanitized_contract_diagnostic(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "invalid schema"}, request=request)

    settings = ClientSettings(
        base_url="http://official.test",
        api_contract="official",
        prediction_endpoint="/prediction/",
        diagnostics_dir=str(tmp_path),
    )
    prediction = {
        "id": 1,
        "user": "http://official.test/users/1/",
        "frame": "http://official.test/frames/1/",
        "detected_objects": [],
        "detected_translations": [{"translation_x": 0, "translation_y": 0, "translation_z": 0}],
        "detected_undefined_objects": [],
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as client:
        with pytest.raises(PermanentAPIError):
            await CompetitionAPI(settings, client).submit(Prediction.model_validate(prediction))

    diagnostic = json.loads((tmp_path / "contract_422_1.json").read_text(encoding="utf-8"))
    assert "password" not in json.dumps(diagnostic).casefold()
    assert "token" not in json.dumps(diagnostic).casefold()
    assert "user" not in diagnostic["prediction"]


def test_camera_profile_selection_is_calibration_safe() -> None:
    rgb = select_camera_profile(1920, 1080, video_name="RGB")
    assert rgb.name == "rgb_1080p"
    assert (rgb.fx, rgb.fy, rgb.cx, rgb.cy) == (1413.3, 1418.8, 950.0639, 543.3796)
    assert select_camera_profile(640, 512, video_name="Termal").modality == "thermal"
    with pytest.raises(ValueError, match="no calibrated camera profile"):
        select_camera_profile(1280, 720, video_name="unknown")


def test_optical_flow_se3_moves_from_last_healthy_position() -> None:
    base = Image.effect_noise((320, 240), 80).convert("RGB")
    shifted = Image.new("RGB", base.size)
    shifted.paste(base, (8, 0))
    estimator = OpticalFlowSE3Estimator(1000.0, 1000.0, 10.0)
    healthy = _frame().model_copy(
        update={
            "translation_x": 100.0,
            "translation_y": 200.0,
            "translation_z": 50.0,
            "gps_health_status": 1,
        }
    )
    unhealthy = healthy.model_copy(
        update={
            "url": "http://test/frames/1/",
            "translation_x": float("nan"),
            "translation_y": float("nan"),
            "translation_z": float("nan"),
            "gps_health_status": 0,
        }
    )

    assert estimator.estimate(base, healthy) == (100.0, 200.0, 50.0)
    estimated = estimator.estimate(shifted, unhealthy)

    assert all(math.isfinite(value) for value in estimated)
    assert estimated[0] < 99.8
    assert abs(estimated[1] - 200.0) < 0.25


def test_reference_definition_active_window() -> None:
    reference = ReferenceDefinition(
        url="http://test/reference/1/",
        session="http://test/session/1/",
        image_url="/media/reference.png",
        frame_start="http://test/frames/10/",
        frame_end="http://test/frames/20/",
        order=1,
    )
    assert reference.is_active("http://test/frames/10/")
    assert reference.is_active("http://test/frames/15/")
    assert not reference.is_active("http://test/frames/21/")
