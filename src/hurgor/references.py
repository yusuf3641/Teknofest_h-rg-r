from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, UnidentifiedImageError
from pydantic import TypeAdapter, ValidationError

from .models import ReferenceDefinition

if TYPE_CHECKING:
    from .client import CompetitionAPI

LOGGER = logging.getLogger("hurgor.references")


@dataclass(frozen=True, slots=True)
class ReferenceAsset:
    object_id: int
    image_path: str
    sha256: str
    frame_start: str
    frame_end: str
    source_url: str


class ReferenceManager:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.assets: list[ReferenceAsset] = []

    async def bootstrap(self, api: CompetitionAPI) -> list[ReferenceAsset]:
        if not api.settings.reference_endpoint:
            return []
        response = await api._request("GET", api.settings.reference_endpoint)
        try:
            raw: Any = response.json()
            definitions = TypeAdapter(list[ReferenceDefinition]).validate_python(raw)
        except (ValueError, ValidationError) as exc:
            raise ValueError(f"invalid reference contract: {exc}") from exc

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        assets: list[ReferenceAsset] = []
        for definition in definitions:
            content = await api.fetch_image(definition.image_url)
            extension = self._verified_extension(content)
            digest = hashlib.sha256(content).hexdigest()
            path = self.cache_dir / f"object_{definition.order}_{digest[:12]}.{extension}"
            path.write_bytes(content)
            assets.append(
                ReferenceAsset(
                    object_id=definition.order,
                    image_path=str(path),
                    sha256=digest,
                    frame_start=definition.frame_start,
                    frame_end=definition.frame_end,
                    source_url=definition.url,
                )
            )
        manifest = self.cache_dir / "references_manifest.json"
        manifest.write_text(
            json.dumps([asdict(asset) for asset in assets], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.assets = assets
        LOGGER.info("references_ready count=%d cache=%s", len(assets), self.cache_dir)
        return list(assets)

    def active(self, frame_url: str) -> list[ReferenceAsset]:
        output: list[ReferenceAsset] = []
        for asset in self.assets:
            definition = ReferenceDefinition(
                url=asset.source_url,
                session="http://local/session/0/",
                image_url=asset.image_path,
                frame_start=asset.frame_start,
                frame_end=asset.frame_end,
                order=asset.object_id,
            )
            if definition.is_active(frame_url):
                output.append(asset)
        return output

    @staticmethod
    def _verified_extension(content: bytes) -> str:
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
                image_format = (image.format or "").lower()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("reference image cannot be decoded") from exc
        if image_format not in {"jpeg", "png", "webp"}:
            raise ValueError(f"unsupported reference image format: {image_format}")
        return "jpg" if image_format == "jpeg" else image_format
