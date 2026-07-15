from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Videodan sızıntı güvenli frame grubu çıkar")
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--modality", choices=("rgb", "thermal"), required=True)
    parser.add_argument("--group", default=None, help="video/scene group ID")
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("stride must be positive")
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python-headless is required") from exc
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    args.output.mkdir(parents=True, exist_ok=True)
    group = args.group or args.video.stem
    records: list[dict[str, object]] = []
    source_index = 0
    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % args.stride == 0:
            target = args.output / f"{args.video.stem}_{source_index:08d}.jpg"
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise SystemExit(f"cannot write {target}")
            records.append(
                {
                    "image": str(target.resolve()),
                    "source_video": str(args.video.resolve()),
                    "source_frame": source_index,
                    "group": group,
                    "modality": args.modality,
                }
            )
            written += 1
            if args.max_frames is not None and written >= args.max_frames:
                break
        source_index += 1
    capture.release()
    manifest = args.output / "frames.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
