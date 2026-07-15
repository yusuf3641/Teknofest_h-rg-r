from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from .client import AuthenticationManager, CompetitionAPI
from .config import ClientSettings
from .inference import PipelineInferenceEngine
from .logging_utils import configure_logging

LOGGER = logging.getLogger("hurgor.preflight")


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def _model_providers(model_info: dict[str, Any]) -> list[str]:
    providers: set[str] = set()

    def walk(value: Any) -> None:
        if not isinstance(value, dict):
            return
        raw_providers = value.get("providers")
        if isinstance(raw_providers, list | tuple):
            providers.update(str(item) for item in raw_providers)
        for nested in value.values():
            walk(nested)

    walk(model_info)
    return sorted(providers)


async def run_preflight(
    settings: ClientSettings,
    *,
    network: bool = False,
    fetch_frame: bool = False,
) -> tuple[int, list[Check]]:
    checks: list[Check] = []
    try:
        settings.validate(for_runtime=True)
        checks.append(Check("config", True, "validated"))
    except ValueError as exc:
        checks.append(Check("config", False, str(exc)))
        return 2, checks

    log_path = Path(settings.log_file).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        probe = log_path.parent / ".hurgor_write_probe"
        probe.touch()
        probe.unlink()
        checks.append(Check("log_write", True, str(log_path.parent)))
    except OSError as exc:
        checks.append(Check("log_write", False, str(exc)))

    disk = shutil.disk_usage(log_path.parent)
    checks.append(Check("disk", disk.free >= 512 * 1024 * 1024, f"free_mb={disk.free // 2**20}"))
    checks.append(Check("python", sys.version_info[:2] >= (3, 11), sys.version.split()[0]))

    engine: PipelineInferenceEngine | None = None
    try:
        engine = PipelineInferenceEngine.from_settings(settings)
        engine.warmup()
        detector = engine.object_detector
        model_info = detector.model_info()
        providers = _model_providers(model_info) or ["noop"]
        is_noop = type(detector).__name__ == "NoopObjectDetector"
        allowed = not (settings.is_official and is_noop and not settings.allow_noop_detector)
        checks.append(Check("model", allowed, f"detector={type(detector).__name__}"))
        checks.append(Check("provider", True, ",".join(providers)))
    except Exception as exc:  # model/provider diagnostics must be reported, not hidden
        checks.append(Check("model", False, f"{type(exc).__name__}: {exc}"))
    finally:
        if engine is not None:
            engine.object_detector.close()

    if network:
        auth_manager = AuthenticationManager(settings) if settings.is_official else None
        headers = {}
        if auth_manager is not None:
            try:
                token = await asyncio.to_thread(auth_manager.token)
                headers["Authorization"] = f"Token {token}"
                checks.append(Check("auth", True, f"elapsed_ms={auth_manager.auth_ms:.3f}"))
            except Exception as exc:
                checks.append(Check("auth", False, f"{type(exc).__name__}: {exc}"))
                return 3, checks
        async with httpx.AsyncClient(
            base_url=settings.base_url,
            headers=headers,
            timeout=settings.http_timeout_seconds,
        ) as client:
            api = CompetitionAPI(settings, client, auth_manager)
            for name, endpoint in (
                ("progress", settings.progress_endpoint),
                ("reference", settings.reference_endpoint),
                ("classes", "/classes/" if settings.is_official else None),
            ):
                if not endpoint:
                    continue
                try:
                    response = await api._request("GET", endpoint)
                    payload: Any = response.json()
                    size = len(payload) if isinstance(payload, list | dict) else 1
                    checks.append(Check(name, True, f"status={response.status_code} items={size}"))
                except Exception as exc:
                    checks.append(Check(name, False, f"{type(exc).__name__}: {exc}"))
            if fetch_frame:
                try:
                    frame = await api.fetch_frame()
                    content = await api.fetch_image(frame.image_url)
                    image = PipelineInferenceEngine._decode_image(content)
                    checks.append(Check("image_decode", True, f"size={image.width}x{image.height}"))
                except Exception as exc:
                    checks.append(Check("image_decode", False, f"{type(exc).__name__}: {exc}"))

    return (0 if all(check.ok for check in checks) else 1), checks


def write_report(checks: list[Check], path: str) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="HürGör yarışma öncesi güvenlik kontrolü")
    parser.add_argument("--network", action="store_true", help="salt-okunur endpointleri denetle")
    parser.add_argument(
        "--fetch-frame",
        action="store_true",
        help="GET frame ve görsel decode et; POST yapmaz",
    )
    parser.add_argument("--report", default="logs/preflight.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    settings = ClientSettings.from_env()
    configure_logging(level=args.log_level, log_file=settings.log_file)
    code, checks = asyncio.run(
        run_preflight(settings, network=args.network, fetch_frame=args.fetch_frame)
    )
    write_report(checks, args.report)
    for check in checks:
        LOGGER.info("preflight_check name=%s ok=%s detail=%s", check.name, check.ok, check.detail)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
