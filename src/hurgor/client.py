from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
import os
import random
import signal
import threading
import time
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import ValidationError

from .config import ClientSettings
from .logging_utils import configure_logging
from .models import FrameMetadata, Prediction

LOGGER = logging.getLogger("hurgor.client")


class SessionComplete(Exception):
    pass


class RetryExhausted(RuntimeError):
    pass


class PermanentAPIError(RuntimeError):
    pass


class AuthenticationError(RuntimeError):
    pass


class CompetitionAPI:
    def __init__(
        self,
        settings: ClientSettings,
        client: httpx.AsyncClient,
        auth_manager: AuthenticationManager | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.auth_manager = auth_manager
        self.retry_count = 0
        self.status_counts: dict[int, int] = {}
        self.last_fetch_timings_ms: dict[str, float] = {}
        self._reference_urls: dict[int, str] | None = None

    async def fetch_frame(self) -> FrameMetadata:
        self.last_fetch_timings_ms = {}
        started = time.perf_counter()
        response = await self._request("GET", self.settings.frame_endpoint)
        self.last_fetch_timings_ms["frame_metadata_ms"] = (time.perf_counter() - started) * 1000
        if response.status_code == 204:
            raise SessionComplete
        try:
            payload: Any = response.json()
            if isinstance(payload, list):
                if self.settings.is_official and not payload:
                    # The official server uses [] at normal completion, but a
                    # transiently empty response must not terminate a live session.
                    # Progress is the only safe discriminator because there is no
                    # outstanding frame to acknowledge in either case.
                    progress = await self.fetch_progress()
                    if self._progress_is_complete(progress):
                        raise SessionComplete
                    raise ValueError("frame response is empty before session completion")
                if len(payload) != 1:
                    raise ValueError("frame response list must contain exactly one item")
                payload = payload[0]
            if not isinstance(payload, dict) or not payload:
                raise ValueError("frame response is empty")
            if self.settings.translation_endpoint and not self._has_translation(payload):
                started = time.perf_counter()
                translation = await self.fetch_translation()
                self.last_fetch_timings_ms["translation_ms"] = (
                    time.perf_counter() - started
                ) * 1000
                payload = self._merge_translation(payload, translation)
            else:
                self.last_fetch_timings_ms["translation_ms"] = 0.0
            return FrameMetadata.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            raise RetryExhausted(f"invalid frame metadata: {exc}") from exc

    async def fetch_translation(self) -> dict[str, Any]:
        if not self.settings.translation_endpoint:
            raise RetryExhausted("translation endpoint is not configured")
        response = await self._request("GET", self.settings.translation_endpoint)
        try:
            payload: Any = response.json()
            if isinstance(payload, list):
                if len(payload) != 1:
                    raise ValueError("translation response list must contain exactly one item")
                payload = payload[0]
            if not isinstance(payload, dict) or not payload:
                raise ValueError("translation response is empty")
            return payload
        except ValueError as exc:
            raise RetryExhausted(f"invalid translation metadata: {exc}") from exc

    async def fetch_progress(self) -> Any:
        response = await self._request("GET", self.settings.progress_endpoint)
        try:
            return response.json()
        except ValueError as exc:
            raise RetryExhausted(f"invalid progress response: {exc}") from exc

    @staticmethod
    def _progress_is_complete(progress: Any) -> bool:
        if isinstance(progress, list) and len(progress) == 1:
            progress = progress[0]
        if not isinstance(progress, dict):
            return False
        if progress.get("completed") is True:
            return True
        current = progress.get("frame_index", progress.get("processed_frames"))
        total = progress.get("total_frames")
        try:
            return total is not None and int(total) > 0 and int(current) >= int(total)
        except (TypeError, ValueError):
            return False

    async def fetch_image(self, image_url: str) -> bytes:
        absolute_url = self._image_url(image_url)
        response = await self._request("GET", absolute_url)
        return response.content

    async def submit(self, prediction: Prediction) -> None:
        if self.settings.api_contract == "official":
            payload: Any = prediction.official_dict(
                self.settings.base_url,
                self._load_reference_urls(),
            )
        else:
            payload = [prediction.canonical_dict()]
        try:
            await self._request("POST", self.settings.prediction_endpoint, json=payload)
        except PermanentAPIError as exc:
            if "HTTP 422" in str(exc):
                self._write_contract_diagnostic(prediction, str(exc))
            raise

    def _load_reference_urls(self) -> dict[int, str]:
        if self._reference_urls is not None:
            return self._reference_urls
        manifest = Path(self.settings.reference_cache_dir).expanduser() / "references_manifest.json"
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            self._reference_urls = {
                int(item["object_id"]): str(item["source_url"])
                for item in raw
                if isinstance(item, dict) and "object_id" in item and "source_url" in item
            }
        except (OSError, ValueError, TypeError):
            self._reference_urls = {}
        return self._reference_urls

    def _image_url(self, image_url: str) -> str:
        if (
            self.settings.api_contract == "official"
            and image_url.startswith("/")
            and not image_url.startswith("/media/")
        ):
            image_url = f"/media{image_url}"
        return urljoin(f"{self.settings.base_url}/", image_url)

    @staticmethod
    def _has_translation(payload: dict[str, Any]) -> bool:
        return {
            "translation_x",
            "translation_y",
            "translation_z",
        }.issubset(payload) and ("gps_health_status" in payload or "health_status" in payload)

    @staticmethod
    def _merge_translation(frame: dict[str, Any], translation: dict[str, Any]) -> dict[str, Any]:
        expected_frame = frame.get("url")
        translated_frame = translation.get("frame")
        if translated_frame is not None and translated_frame != expected_frame:
            raise ValueError(
                f"translation frame mismatch: expected {expected_frame}, got {translated_frame}"
            )
        for key in ("image_url", "video_name", "session"):
            if key in translation and key in frame and translation[key] != frame[key]:
                raise ValueError(f"translation {key} mismatch")
        merged = dict(frame)
        for key in ("translation_x", "translation_y", "translation_z"):
            if key not in translation:
                raise ValueError(f"translation response missing {key}")
            merged[key] = translation[key]
        if "health_status" in translation:
            merged["health_status"] = translation["health_status"]
        elif "gps_health_status" in translation:
            merged["gps_health_status"] = translation["gps_health_status"]
        else:
            raise ValueError("translation response missing health_status/gps_health_status")
        for key in (
            "orientation_x",
            "orientation_y",
            "orientation_z",
            "orientation_w",
        ):
            if key in translation:
                merged[key] = translation[key]
        return merged

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        refreshed = False
        total_attempts = self.settings.max_retries + 1 + int(self.auth_manager is not None)
        for attempt in range(total_attempts):
            try:
                response = await self.client.request(method, url, **kwargs)
                self.status_counts[response.status_code] = (
                    self.status_counts.get(response.status_code, 0) + 1
                )
                if response.status_code == 401 and self.auth_manager is not None and not refreshed:
                    refreshed = True
                    authorization = self.client.headers.get("Authorization", "")
                    stale_token = authorization.removeprefix("Token ")
                    token = await asyncio.to_thread(self.auth_manager.refresh, stale_token)
                    self.client.headers["Authorization"] = f"Token {token}"
                    self.retry_count += 1
                    continue
                if response.status_code == 204:
                    return response
                if response.status_code >= 500 or response.status_code in {408, 425, 429}:
                    raise httpx.HTTPStatusError(
                        f"retryable HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise PermanentAPIError(
                        f"{method} {url} returned permanent HTTP "
                        f"{response.status_code}: {response.text[:300]}"
                    ) from exc
                return response
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code >= 500 or exc.response.status_code in {408, 425, 429}
                )
                if not retryable or attempt >= self.settings.max_retries:
                    break
                self.retry_count += 1
                retry_after = _retry_after_seconds(
                    exc.response.headers.get("Retry-After")
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                delay = (
                    retry_after
                    if retry_after is not None
                    else min(5.0, self.settings.retry_base_seconds * (2**attempt))
                    + random.uniform(0.0, self.settings.retry_base_seconds)
                )
                await asyncio.sleep(delay)
        raise RetryExhausted(f"{method} {url} failed: {last_error}") from last_error

    def _write_contract_diagnostic(self, prediction: Prediction, error: str) -> None:
        directory = Path(self.settings.diagnostics_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        identifier = prediction.id
        payload = {
            "error": error[:1000],
            "frame": prediction.frame,
            "prediction": (
                prediction.official_dict(
                    self.settings.base_url,
                    self._load_reference_urls(),
                )
                if self.settings.is_official
                else prediction.canonical_dict()
            ),
        }
        target = directory / f"contract_422_{identifier}.json"
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )


class AuthenticationManager:
    """Thread-safe in-memory token owner shared by all network roles."""

    def __init__(self, settings: ClientSettings) -> None:
        self.settings = settings
        self._token: str | None = settings.auth_token
        self._lock = threading.Lock()
        self.auth_ms = 0.0
        self.auth_count = 0
        self._reported_auth_count = 0

    def token(self) -> str:
        with self._lock:
            if self._token is None:
                self._token = self._login()
            return self._token

    def refresh(self, stale_token: str | None = None) -> str:
        with self._lock:
            if stale_token and self._token and self._token != stale_token:
                return self._token
            self._token = self._login()
            return self._token

    def take_auth_ms(self) -> float:
        with self._lock:
            if self.auth_count <= self._reported_auth_count:
                return 0.0
            self._reported_auth_count = self.auth_count
            return self.auth_ms

    def _login(self) -> str:
        if not self.settings.token_endpoint:
            raise AuthenticationError("token endpoint is required")
        if not self.settings.team_name or not self.settings.password:
            raise AuthenticationError("TEAM_NAME and PASSWORD are required")
        started = time.perf_counter()
        try:
            response = httpx.post(
                urljoin(f"{self.settings.base_url}/", self.settings.token_endpoint),
                data={
                    "username": self.settings.team_name,
                    "password": self.settings.password,
                },
                timeout=self.settings.http_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise AuthenticationError("token login failed") from exc
        finally:
            self.auth_ms = (time.perf_counter() - started) * 1000
            self.auth_count += 1
        token = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise AuthenticationError("token login response did not include token")
        return token


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, min(float(value), 30.0))
    except ValueError:
        return None


class HeaderTokenAuth(httpx.Auth):
    def __init__(self, scheme: str, token: str) -> None:
        self.scheme = scheme
        self.token = token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"{self.scheme} {self.token}"
        yield request


def build_http_auth(settings: ClientSettings) -> httpx.Auth | None:
    """Build authentication without exposing credentials in logs.

    The official Team Connection Interface advertises TEAM_NAME/PASSWORD.
    The public test server advertises DRF Token auth, so HURGOR_AUTH_TOKEN is
    supported explicitly. If no token is available, auto mode falls back to
    Basic Auth for local/mock or committee-provided variants.
    """

    if settings.auth_scheme in {"none", "off", "disabled"}:
        return None
    if settings.auth_scheme not in {"auto", "basic", "token", "bearer"}:
        raise ValueError(f"unsupported HURGOR_AUTH_SCHEME={settings.auth_scheme!r}")
    if settings.auth_scheme in {"auto", "token"} and settings.auth_token:
        return HeaderTokenAuth("Token", settings.auth_token)
    if settings.auth_scheme == "bearer" and settings.auth_token:
        return HeaderTokenAuth("Bearer", settings.auth_token)
    if settings.auth_scheme in {"token", "bearer"}:
        raise ValueError(f"HURGOR_AUTH_SCHEME={settings.auth_scheme!r} requires HURGOR_AUTH_TOKEN")
    if not settings.team_name or not settings.password:
        return None
    return httpx.BasicAuth(settings.team_name, settings.password)


async def fetch_auth_token(settings: ClientSettings) -> str:
    if not settings.token_endpoint:
        raise AuthenticationError("HURGOR_TOKEN_ENDPOINT is required for token login")
    if not settings.team_name or not settings.password:
        raise AuthenticationError("TEAM_NAME and PASSWORD are required for token login")
    async with httpx.AsyncClient(
        base_url=settings.base_url,
        timeout=httpx.Timeout(settings.http_timeout_seconds),
    ) as client:
        try:
            response = await client.post(
                settings.token_endpoint,
                data={"username": settings.team_name, "password": settings.password},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise AuthenticationError("token login failed") from exc
    token = payload.get("token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        raise AuthenticationError("token login response did not include token")
    return token


async def build_http_auth_async(settings: ClientSettings) -> httpx.Auth | None:
    if settings.auth_scheme in {"none", "off", "disabled"}:
        return None
    if settings.auth_token:
        scheme = "Bearer" if settings.auth_scheme == "bearer" else "Token"
        return HeaderTokenAuth(scheme, settings.auth_token)
    if settings.auth_scheme in {"auto", "token"} and settings.token_endpoint:
        return HeaderTokenAuth("Token", await fetch_auth_token(settings))
    if settings.auth_scheme in {"token", "bearer"}:
        raise ValueError(
            f"HURGOR_AUTH_SCHEME={settings.auth_scheme!r} requires "
            "HURGOR_AUTH_TOKEN or token login settings"
        )
    return build_http_auth(settings)


def override_base_url_for_cli(settings: ClientSettings, base_url: str) -> ClientSettings:
    """Override target server while keeping explicit endpoint env overrides intact."""

    defaults = ClientSettings()
    updates: dict[str, Any] = {"base_url": base_url.rstrip("/")}
    loopback = _is_loopback_base_url(base_url)
    # The committee may announce a private Ethernet IP on competition day. When
    # the loaded environment is already official, changing only host/port must not
    # silently switch /frames/ and /prediction/ back to the local mock contract.
    preserve_official_contract = settings.is_official and not loopback
    if not preserve_official_contract:
        if "HURGOR_FRAME_ENDPOINT" not in os.environ:
            updates["frame_endpoint"] = defaults.frame_endpoint
        if "HURGOR_TRANSLATION_ENDPOINT" not in os.environ:
            updates["translation_endpoint"] = defaults.translation_endpoint
        if "HURGOR_PREDICTION_ENDPOINT" not in os.environ:
            updates["prediction_endpoint"] = defaults.prediction_endpoint
        if "HURGOR_PROGRESS_ENDPOINT" not in os.environ:
            updates["progress_endpoint"] = defaults.progress_endpoint
        if "HURGOR_REFERENCE_ENDPOINT" not in os.environ:
            updates["reference_endpoint"] = defaults.reference_endpoint
        if "HURGOR_API_CONTRACT" not in os.environ:
            updates["api_contract"] = defaults.api_contract
    if loopback:
        updates["auth_scheme"] = "none"
        updates["auth_token"] = None
        updates["token_endpoint"] = None
    return replace(settings, **updates)


def _is_loopback_base_url(base_url: str) -> bool:
    host = urlsplit(base_url).hostname
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="HürGör Edge AI istemcisi")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--server-ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    settings = ClientSettings.from_env()
    if args.base_url:
        settings = override_base_url_for_cli(settings, args.base_url)
    elif args.server_ip is not None or args.port is not None:
        server_ip = args.server_ip or "127.0.0.1"
        port = args.port or 5000
        settings = override_base_url_for_cli(settings, f"http://{server_ip}:{port}")
    configure_logging(level=args.log_level, log_file=settings.log_file)
    try:
        settings.validate(for_runtime=True)
    except ValueError as exc:
        LOGGER.error("preflight_config_failed error=%s", exc)
        raise SystemExit(2) from exc

    # Local import avoids a module cycle: the threaded pipeline reuses CompetitionAPI.
    from .threaded_pipeline import ThreadedEdgePipeline

    pipeline = ThreadedEdgePipeline(settings)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _signum, _frame: pipeline.stop())
    stats = pipeline.run(max_frames=args.max_frames)
    LOGGER.info(
        "finished frames=%d elapsed_s=%.3f fps=%.2f sla_misses=%d degraded_frames=%d fatal=%s",
        stats.frames_submitted,
        stats.elapsed,
        stats.fps,
        stats.sla_misses,
        stats.degraded_frames,
        stats.fatal_error,
    )
    if stats.fatal_error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
