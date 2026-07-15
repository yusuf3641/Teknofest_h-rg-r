from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8))
        pixels = gray.tobytes()
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Görüntü kopyalarını perceptual hash ile bul")
    parser.add_argument("images", type=Path)
    parser.add_argument("--quarantine", type=Path, default=Path("artifacts/duplicates"))
    parser.add_argument("--apply", action="store_true", help="kopyaları silmeden karantinaya taşı")
    parser.add_argument("--report", type=Path, default=Path("logs/duplicates.json"))
    args = parser.parse_args()
    seen: dict[int, Path] = {}
    duplicates: list[dict[str, str]] = []
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for path in sorted(
        item for item in args.images.rglob("*") if item.suffix.lower() in extensions
    ):
        fingerprint = difference_hash(path)
        original = seen.get(fingerprint)
        if original is None:
            seen[fingerprint] = path
            continue
        duplicates.append({"duplicate": str(path), "original": str(original)})
        if args.apply:
            target = args.quarantine / path.relative_to(args.images)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(duplicates, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
