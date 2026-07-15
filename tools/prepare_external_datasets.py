from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def tree_checksum(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dış veri kaynağı için lisanslı manifest kaydı")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--modality", choices=("rgb", "thermal", "paired"), required=True)
    parser.add_argument("--original-classes", required=True, help="virgülle ayrılmış")
    parser.add_argument("--usage-notes", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.source_dir.is_dir():
        raise SystemExit(f"source directory missing: {args.source_dir}")
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [
        path for path in args.source_dir.rglob("*") if path.suffix.lower() in image_extensions
    ]
    labels = list(args.source_dir.rglob("*.txt"))
    payload = {
        "dataset_name": args.dataset_name,
        "source_url": args.source_url,
        "license": args.license,
        "modality": args.modality,
        "original_classes": [item.strip() for item in args.original_classes.split(",")],
        "mapped_classes": ["arac", "insan", "uap", "uai"],
        "image_count": len(images),
        "label_count": len(labels),
        "checksum": tree_checksum(args.source_dir),
        "usage_notes": args.usage_notes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
