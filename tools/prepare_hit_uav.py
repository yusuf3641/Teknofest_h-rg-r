from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

SOURCE_URL = (
    "https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/"
    "releases/download/v1.2.1/HIT-UAV.zip"
)
SOURCE_PAGE = "https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset"
RELEASE = "v1.2.1"
EXPECTED_ARCHIVE_BYTES = 812_468_971
HIT_TO_HURGOR_CLASS = {
    0: 1,  # Person -> insan
    1: 0,  # Car -> arac
    2: 0,  # Bicycle/rider is a vehicle under the competition definition.
    3: 0,  # OtherVehicle -> arac
}


class HTTPRangeReader(io.RawIOBase):
    """Seekable, bounded HTTP reader suitable for selective ZIP extraction."""

    def __init__(
        self,
        url: str,
        *,
        expected_size: int,
        block_size: int = 4 * 1024 * 1024,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__()
        self.url = url
        self.expected_size = expected_size
        self.block_size = max(64 * 1024, block_size)
        self.position = 0
        self.cache_start = 0
        self.cache = b""
        self.client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": "Hurgor-dataset-audit/1.0"},
        )
        response = self.client.head(self.url)
        response.raise_for_status()
        size = int(response.headers.get("Content-Length", "0"))
        if size != self.expected_size:
            self.client.close()
            raise ValueError(
                f"unexpected HIT-UAV archive size: expected {self.expected_size}, got {size}"
            )
        if "bytes" not in response.headers.get("Accept-Ranges", "").casefold():
            self.client.close()
            raise ValueError("HIT-UAV host does not advertise byte-range support")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.expected_size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = min(position, self.expected_size)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.expected_size:
            return b""
        if size is None or size < 0:
            size = self.expected_size - self.position
        size = min(size, self.expected_size - self.position)
        output = bytearray()
        while len(output) < size:
            if not self._cache_contains(self.position):
                self._fetch_block(self.position, size - len(output))
            cache_offset = self.position - self.cache_start
            available = min(len(self.cache) - cache_offset, size - len(output))
            if available <= 0:
                raise OSError("empty HTTP range block")
            output.extend(self.cache[cache_offset : cache_offset + available])
            self.position += available
        return bytes(output)

    def close(self) -> None:
        if not self.closed:
            self.client.close()
        super().close()

    def _cache_contains(self, position: int) -> bool:
        return self.cache_start <= position < self.cache_start + len(self.cache)

    def _fetch_block(self, position: int, requested: int) -> None:
        start = (position // self.block_size) * self.block_size
        length = max(self.block_size, requested + (position - start))
        end = min(self.expected_size - 1, start + length - 1)
        response = self.client.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
        )
        if response.status_code != 206:
            raise OSError(
                "range request was not honored; refusing an unbounded archive download "
                f"(HTTP {response.status_code})"
            )
        expected = end - start + 1
        if len(response.content) != expected:
            raise OSError(
                f"short HTTP range: expected {expected} bytes, got {len(response.content)}"
            )
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise OSError(f"unexpected Content-Range: {content_range!r}")
        self.cache_start = start
        self.cache = response.content


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member: {name}")


def convert_annotations(
    annotation_path: Path,
    image_dir: Path,
    label_dir: Path,
) -> dict[str, Any]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    expected_categories = {
        0: "Person",
        1: "Car",
        2: "Bicycle",
        3: "OtherVehicle",
        4: "DontCare",
    }
    if categories != expected_categories:
        raise ValueError(f"unexpected HIT-UAV categories: {categories}")
    images = {
        int(item["id"]): {
            "filename": str(item["filename"]),
            "width": int(item["width"]),
            "height": int(item["height"]),
        }
        for item in payload["images"]
    }
    available = {path.name for path in image_dir.iterdir() if path.is_file()}
    selected_ids = {
        image_id for image_id, item in images.items() if item["filename"] in available
    }
    labels: dict[int, list[tuple[int, int, str]]] = {image_id: [] for image_id in selected_ids}
    ignored_annotations = 0
    class_counts = {"arac": 0, "insan": 0}
    for annotation in payload["annotation"]:
        image_id = int(annotation["image_id"])
        if image_id not in selected_ids:
            continue
        source_class = int(annotation["category_id"])
        target_class = HIT_TO_HURGOR_CLASS.get(source_class)
        if target_class is None:
            ignored_annotations += 1
            continue
        image = images[image_id]
        width = image["width"]
        height = image["height"]
        x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + box_width))
        y2 = max(0.0, min(float(height), y + box_height))
        if x2 <= x1 or y2 <= y1:
            ignored_annotations += 1
            continue
        normalized = (
            (x1 + x2) / (2.0 * width),
            (y1 + y2) / (2.0 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
        )
        line = f"{target_class} " + " ".join(f"{value:.8f}" for value in normalized)
        labels[image_id].append((int(annotation["id"]), target_class, line))
        class_counts["arac" if target_class == 0 else "insan"] += 1

    label_dir.mkdir(parents=True, exist_ok=True)
    label_hash = hashlib.sha256()
    empty_images = 0
    for image_id in sorted(selected_ids, key=lambda item: images[item]["filename"]):
        filename = images[image_id]["filename"]
        target = label_dir / f"{Path(filename).stem}.txt"
        lines = [item[2] for item in sorted(labels[image_id])]
        if not lines:
            empty_images += 1
        content = "\n".join(lines) + ("\n" if lines else "")
        target.write_text(content, encoding="utf-8")
        label_hash.update(target.name.encode("utf-8"))
        label_hash.update(content.encode("utf-8"))
    return {
        "selected_images": len(selected_ids),
        "label_files": len(selected_ids),
        "empty_images": empty_images,
        "class_counts": class_counts,
        "ignored_annotations": ignored_annotations,
        "label_tree_sha256": label_hash.hexdigest(),
    }


def prepare(
    output: Path,
    *,
    split: str,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    output = output.expanduser().resolve()
    image_dir = output / "images" / split
    annotation_path = output / "annotations" / f"{split}.json"
    label_dir = output / "labels" / split
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and not force and limit is None:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") == 2
            and payload.get("release") == RELEASE
            and payload.get("split") == split
        ):
            return payload

    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    if force:
        for existing in image_dir.iterdir():
            if existing.is_file():
                existing.unlink()

    prefix = f"HIT-UAV/normal_json/{split}/"
    annotation_member = f"HIT-UAV/normal_json/annotations/{split}.json"
    extracted: list[Path] = []
    with HTTPRangeReader(SOURCE_URL, expected_size=EXPECTED_ARCHIVE_BYTES) as remote:
        with zipfile.ZipFile(remote) as archive:
            # Preserve archive order so the range cache turns hundreds of small
            # image reads into a few sequential HTTP range requests.
            image_members = [
                item
                for item in archive.infolist()
                if item.filename.startswith(prefix)
                and not item.is_dir()
                and PurePosixPath(item.filename).suffix.casefold()
                in {".jpg", ".jpeg", ".png"}
            ]
            if limit is not None:
                image_members = image_members[:limit]
            try:
                annotation_info = archive.getinfo(annotation_member)
            except KeyError as exc:
                raise ValueError(f"annotation member missing: {annotation_member}") from exc
            for item in [annotation_info, *image_members]:
                _safe_member_name(item.filename)

            temporary_annotation = annotation_path.with_suffix(".json.part")
            with archive.open(annotation_info) as source, temporary_annotation.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary_annotation.replace(annotation_path)

            for item in image_members:
                target = image_dir / PurePosixPath(item.filename).name
                temporary = target.with_suffix(target.suffix + ".part")
                with archive.open(item) as source, temporary.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
                temporary.replace(target)
                extracted.append(target)

    image_hash = hashlib.sha256()
    total_bytes = 0
    for path in sorted(extracted):
        total_bytes += path.stat().st_size
        image_hash.update(path.name.encode("utf-8"))
        image_hash.update(bytes.fromhex(_sha256(path)))
    conversion = convert_annotations(annotation_path, image_dir, label_dir)
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            (
                f"path: {output}",
                f"train: images/{split}",
                f"val: images/{split}",
                f"test: images/{split}",
                "names:",
                "  0: arac",
                "  1: insan",
                "  2: uap",
                "  3: uai",
                "",
            )
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 2,
        "dataset": "HIT-UAV",
        "release": RELEASE,
        "source_url": SOURCE_PAGE,
        "archive_url": SOURCE_URL,
        "archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "license": "CC-BY-4.0",
        "modality": "thermal",
        "split": split,
        "image_count": len(extracted),
        "image_bytes": total_bytes,
        "image_tree_sha256": image_hash.hexdigest(),
        "annotation_path": str(annotation_path),
        "annotation_sha256": _sha256(annotation_path),
        "selection_limit": limit,
        "conversion": conversion,
        "data_yaml": str(data_yaml),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Selectively extract HIT-UAV without storing its 812 MB ZIP archive"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/datasets/hit_uav_thermal"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    payload = prepare(**vars(build_parser().parse_args()))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
