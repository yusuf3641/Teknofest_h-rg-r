from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger("hurgor.export")


def export_yolo(
    source: str,
    *,
    target: str,
    image_size: int,
    device: str,
    half: bool,
) -> Path:
    path = Path(source).expanduser().resolve()
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("YOLO export kaynağı .pt/.pth olmalıdır")
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Export aracı için `pip install -e '.[export]'` çalıştırılmalıdır"
        ) from exc
    if target not in {"onnx", "engine"}:
        raise ValueError("target yalnızca onnx veya engine olabilir")
    model = YOLO(str(path))
    output = model.export(
        format=target,
        imgsz=image_size,
        device=device,
        half=half,
        dynamic=False,
        simplify=target == "onnx",
    )
    exported = Path(str(output)).resolve()
    LOGGER.info("export_complete source=%s target=%s output=%s", path, target, exported)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO modelini runtime için ONNX veya TensorRT engine'e çevir"
    )
    parser.add_argument("source", help="Eğitilmiş .pt model")
    parser.add_argument("--target", choices=("onnx", "engine"), default="onnx")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    export_yolo(
        args.source,
        target=args.target,
        image_size=args.image_size,
        device=args.device,
        half=args.half,
    )


if __name__ == "__main__":
    main()
