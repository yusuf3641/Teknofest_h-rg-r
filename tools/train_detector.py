from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tekrarlanabilir Ultralytics detector eğitimi")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="yerel pretrained checkpoint")
    parser.add_argument("--project", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--name", default="yolo26s-960")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="training split fraction; use a small value only for smoke tests",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.data.is_file() or not args.model.is_file():
        raise SystemExit(
            "data YAML and local checkpoint must exist; implicit downloads are disabled"
        )
    if not 0 < args.fraction <= 1:
        raise SystemExit("fraction must be greater than zero and at most one")
    if args.epochs < 1 or args.image_size < 32 or args.batch < 1 or args.workers < 0:
        raise SystemExit(
            "epochs, image-size and batch must be positive; workers cannot be negative"
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("install export dependencies: pip install -e '.[export]'") from exc
    random.seed(args.seed)
    project = args.project.resolve()
    config = vars(args) | {
        "data": str(args.data.resolve()),
        "model": str(args.model.resolve()),
        "project": str(project),
    }
    project.mkdir(parents=True, exist_ok=True)
    (project / f"{args.name}-config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        cache=args.cache,
        amp=args.amp,
        project=str(project),
        name=args.name,
        patience=args.patience,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
