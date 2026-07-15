from __future__ import annotations

import argparse
from pathlib import Path

from hurgor.export_models import export_yolo


def main() -> None:
    parser = argparse.ArgumentParser(description="HürGör detector export")
    parser.add_argument("weights", type=Path)
    parser.add_argument("--target", choices=("onnx", "engine", "coreml"), required=True)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"weights missing: {args.weights}")
    export_yolo(
        str(args.weights),
        target=args.target,
        image_size=args.image_size,
        device=args.device,
        half=args.half,
        dynamic=args.dynamic,
        int8=args.int8,
        end2end=args.end2end,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
