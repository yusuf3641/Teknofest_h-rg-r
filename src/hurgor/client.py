from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from collections.abc import Generator
from dataclasses import replace
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
    def __init__(self, settings: ClientSettings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    async def fetch_frame(self) -> FrameMetadata:
        response = await self._request("GET", self.settings.frame_endpoint)
        if response.status_code == 204:
            raise SessionComplete
        try:
            payload: Any = response.json()
            if isinstance(payload, list):
                if len(payload) != 1:
                    raise ValueError("frame response list must contain exactly one item")
                payload = payload[0]
            if not isinstance(payload, dict) or not payload:
                raise ValueError("frame response is empty")
            if self.settings.translation_endpoint and not self._has_translation(payload):
                translation = await self.fetch_translation()
                payload = self._merge_translation(payload, translation)
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

    async def fetch_image(self, image_url: str) -> bytes:
        absolute_url = self._image_url(image_url)
        response = await self._request("GET", absolute_url)
        return response.content

    async def submit(self, prediction: Prediction) -> None:
        if self.settings.api_contract == "official":
            payload: Any = prediction.official_dict(self.settings.base_url)
        else:
            payload = [prediction.canonical_dict()]
        await self._request("POST", self.settings.prediction_endpoint, json=payload)

    def _image_url(self, image_url: str) -> str:
        if self.settings.api_contract == "official" and image_url.startswith(
            "/"
        ) and not image_url.startswith("/media/"):
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
        return merged

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = await self.client.request(method, url, **kwargs)
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
                await asyncio.sleep(self.settings.retry_base_seconds * (2**attempt))
        raise RetryExhausted(f"{method} {url} failed: {last_error}") from last_error


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
        raise ValueError(
            f"HURGOR_AUTH_SCHEME={settings.auth_scheme!r} requires HURGOR_AUTH_TOKEN"
        )
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
    if _is_loopback_base_url(base_url):
        updates["auth_scheme"] = "none"
        updates["auth_token"] = None
        updates["token_endpoint"] = None
    return replace(settings, **updates)


def _is_loopback_base_url(base_url: str) -> bool:
    host = urlsplit(base_url).hostname
    return host in {"127.0.0.1", "localhost", "::1"}


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


if __name__ == "__main__":
    main()
