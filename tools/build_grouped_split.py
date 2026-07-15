from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Video/scene gruplu train-val-test split oluştur")
    parser.add_argument("manifest", type=Path, help="extract_frames JSONL veya birleşik manifest")
    parser.add_argument("output", type=Path)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    test_ratio = 1.0 - args.train - args.val
    if min(args.train, args.val, test_ratio) <= 0:
        raise SystemExit("train/val/test ratios must all be positive")
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        group = str(record.get("group") or record.get("source_video") or "")
        if not group:
            raise SystemExit("every record requires group or source_video")
        groups[group].append(record)
    names = sorted(groups)
    random.Random(args.seed).shuffle(names)
    targets = {"train": args.train * len(records), "val": args.val * len(records)}
    splits: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    for name in names:
        if len(splits["train"]) < targets["train"]:
            split = "train"
        elif len(splits["val"]) < targets["val"]:
            split = "val"
        else:
            split = "test"
        splits[split].extend(groups[name])
    args.output.mkdir(parents=True, exist_ok=True)
    assigned_groups: dict[str, str] = {}
    for split, items in splits.items():
        (args.output / f"{split}.txt").write_text(
            "\n".join(str(item["image"]) for item in items) + ("\n" if items else ""),
            encoding="utf-8",
        )
        for item in items:
            assigned_groups[str(item.get("group") or item.get("source_video"))] = split
    (args.output / "split_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "counts": {key: len(value) for key, value in splits.items()},
                "groups": assigned_groups,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
