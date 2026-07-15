from __future__ import annotations

import argparse
import asyncio
import logging
import socket
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

import httpx

from .client import CompetitionAPI, build_http_auth_async
from .config import ClientSettings
from .logging_utils import configure_logging

LOGGER = logging.getLogger("hurgor.official_probe")


def _host(base_url: str) -> str:
    parsed = urlsplit(base_url)
    return parsed.hostname or base_url


def _credential_state(settings: ClientSettings) -> str:
    has_user_pass = bool(settings.team_name and settings.password)
    has_token = bool(settings.auth_token)
    if has_user_pass and has_token:
        return "user_password_and_token_present"
    if has_token:
        return "token_present"
    if has_user_pass:
        return "user_password_present"
    if settings.team_name or settings.password:
        return "partial"
    return "missing"


@dataclass(frozen=True, slots=True)
class ProbeHTTPResult:
    status: int | None
    body: str
    www_authenticate: str
    allow: str


async def _get(client: httpx.AsyncClient, path: str) -> ProbeHTTPResult:
    try:
        response = await client.get(path, follow_redirects=True)
        body = response.text[:240].replace("\n", " ") if response.text else ""
        return ProbeHTTPResult(
            status=response.status_code,
            body=body,
            www_authenticate=response.headers.get("www-authenticate", ""),
            allow=response.headers.get("allow", ""),
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic command must not crash.
        return ProbeHTTPResult(
            status=None,
            body=f"{type(exc).__name__}: {exc}",
            www_authenticate="",
            allow="",
        )


async def probe(settings: ClientSettings, *, fetch_frame: bool = False) -> int:
    host = _host(settings.base_url)
    LOGGER.info(
        "probe_start base_url=%s host=%s credentials=%s",
        settings.base_url,
        host,
        _credential_state(settings),
    )
    try:
        addresses = socket.getaddrinfo(host, urlsplit(settings.base_url).port or 80)
        LOGGER.info("dns_ok host=%s first_address=%s", host, addresses[0][4][0])
    except socket.gaierror as exc:
        LOGGER.error("dns_failed host=%s error=%s", host, exc)
        if "havaciliktyapayzeka" in host:
            LOGGER.error(
                "host_hint official DNS currently resolves as havaciliktayapayzeka.teknofest.org"
            )
        return 2

    try:
        auth = await build_http_auth_async(settings)
    except Exception as exc:  # noqa: BLE001 - diagnostic command must not crash.
        LOGGER.error("auth_setup_failed error=%s", exc)
        return 4

    async with httpx.AsyncClient(
        base_url=settings.base_url,
        auth=auth,
        timeout=httpx.Timeout(settings.http_timeout_seconds),
    ) as client:
        saw_401 = False
        auth_headers: list[str] = []
        # A frame GET changes competition state by reserving the current frame. Keep the
        # default probe strictly read-only; --fetch-frame uses CompetitionAPI explicitly.
        for path in _read_only_probe_paths(settings):
            result = await _get(client, path)
            LOGGER.info(
                "probe_get path=%s status=%s www_authenticate=%s allow=%s body=%s",
                path,
                result.status,
                result.www_authenticate or "-",
                result.allow or "-",
                result.body,
            )
            saw_401 = saw_401 or result.status == 401
            if result.www_authenticate:
                auth_headers.append(result.www_authenticate)

        if saw_401:
            if _credential_state(settings) == "missing":
                LOGGER.warning("auth_required credentials_missing_in_env")
                return 3
            if any("token" in header.lower() for header in auth_headers):
                if not settings.auth_token:
                    LOGGER.warning(
                        "token_auth_required HURGOR_AUTH_TOKEN_missing "
                        "team_password_present=%s token_endpoint=%s",
                        bool(settings.team_name and settings.password),
                        settings.token_endpoint or "-",
                    )
                    return 4
                LOGGER.warning("token_auth_failed HURGOR_AUTH_TOKEN_present_but_rejected")
                return 4
            LOGGER.warning("auth_failed_or_non_basic credentials_present_but_server_returned_401")
            return 4

        if fetch_frame:
            try:
                api = CompetitionAPI(settings, client)
                frame = await api.fetch_frame()
                LOGGER.info(
                    "frame_fetch_ok frame=%s image_url=%s video_name=%s health=%s",
                    frame.url,
                    frame.image_url,
                    frame.video_name,
                    frame.gps_health_status,
                )
            except Exception as exc:  # noqa: BLE001 - probe should report, not crash.
                LOGGER.error("frame_fetch_failed error=%s", exc)
                return 5

    LOGGER.info("probe_complete")
    return 0


def _read_only_probe_paths(settings: ClientSettings) -> tuple[str, ...]:
    return (settings.progress_endpoint,)


def main() -> None:
    parser = argparse.ArgumentParser(description="TEKNOFEST resmi sunucu bağlantı probu")
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--fetch-frame",
        action="store_true",
        help="Durum değiştirebilecek GET frame testi",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    settings = ClientSettings.from_env()
    if args.base_url:
        settings = replace(settings, base_url=args.base_url.rstrip("/"))
    configure_logging(level=args.log_level, log_file=settings.log_file)
    raise SystemExit(asyncio.run(probe(settings, fetch_frame=args.fetch_frame)))


if __name__ == "__main__":
    main()
