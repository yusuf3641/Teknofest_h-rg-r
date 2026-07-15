from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_CLASSES = ["arac", "insan", "uap", "uai"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="HürGör ONNX model manifesti üret")
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-format",
        choices=("yolo_one_to_many", "yolo_end2end", "rfdetr"),
        required=True,
    )
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--source-checkpoint", default="unknown")
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Virgülle ayrılmış, eğitimde kullanılan kesin sınıf sırası",
    )
    args = parser.parse_args()
    if not args.model.is_file() or args.model.suffix.lower() != ".onnx":
        raise SystemExit("model must be an existing .onnx file")
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    if not classes or len(set(classes)) != len(classes):
        raise SystemExit("classes must be non-empty and unique")
    payload = {
        "schema_version": 1,
        "model_file": args.model.name,
        "sha256": sha256_file(args.model),
        "classes": classes,
        "output_format": args.output_format,
        "image_size": args.image_size,
        "batch": 1,
        "source_checkpoint": args.source_checkpoint,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
