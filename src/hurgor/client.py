from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from dataclasses import replace
from typing import Any
from urllib.parse import urljoin

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
            return FrameMetadata.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            raise RetryExhausted(f"invalid frame metadata: {exc}") from exc

    async def fetch_image(self, image_url: str) -> bytes:
        absolute_url = urljoin(f"{self.settings.base_url}/", image_url)
        response = await self._request("GET", absolute_url)
        return response.content

    async def submit(self, prediction: Prediction) -> None:
        payload: Any = [prediction.canonical_dict()]
        await self._request("POST", self.settings.prediction_endpoint, json=payload)

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
        settings = replace(settings, base_url=args.base_url.rstrip("/"))
    elif args.server_ip is not None or args.port is not None:
        server_ip = args.server_ip or "127.0.0.1"
        port = args.port or 5000
        settings = replace(settings, base_url=f"http://{server_ip}:{port}")
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
