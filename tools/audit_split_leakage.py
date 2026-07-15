from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from resplit_yolo_dataset import IMAGE_EXTENSIONS, SPLITS, difference_hash


def audit(dataset: Path, *, max_distance: int, example_limit: int = 20) -> dict[str, object]:
    if max_distance < 0 or max_distance > 8:
        raise ValueError("max_distance must be in 0..8")
    records: list[tuple[str, Path, int]] = []
    for split in SPLITS:
        root = dataset / "images" / split
        if not root.is_dir():
            continue
        for path in sorted(
            item for item in root.rglob("*") if item.suffix.casefold() in IMAGE_EXTENSIONS
        ):
            records.append((split, path, difference_hash(path)))

    block_count = max_distance + 1
    base_width, remainder = divmod(64, block_count)
    widths = [base_width + int(index < remainder) for index in range(block_count)]
    offsets: list[int] = []
    offset = 0
    for width in widths:
        offsets.append(offset)
        offset += width

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    counts: dict[str, Counter[int]] = {
        "train-val": Counter(),
        "train-test": Counter(),
        "val-test": Counter(),
    }
    examples: dict[str, list[dict[str, object]]] = {key: [] for key in counts}
    split_order = {name: index for index, name in enumerate(SPLITS)}

    for index, (split, path, fingerprint) in enumerate(records):
        candidates: set[int] = set()
        for block, (block_offset, width) in enumerate(zip(offsets, widths, strict=True)):
            value = (fingerprint >> block_offset) & ((1 << width) - 1)
            candidates.update(buckets[(block, value)])
        for candidate_index in candidates:
            other_split, other_path, other_fingerprint = records[candidate_index]
            if split == other_split:
                continue
            distance = (fingerprint ^ other_fingerprint).bit_count()
            if distance > max_distance:
                continue
            first, second = sorted((split, other_split), key=split_order.__getitem__)
            pair = f"{first}-{second}"
            counts[pair][distance] += 1
            if len(examples[pair]) < example_limit:
                examples[pair].append(
                    {
                        "distance": distance,
                        split: path.name,
                        other_split: other_path.name,
                    }
                )
        for block, (block_offset, width) in enumerate(zip(offsets, widths, strict=True)):
            value = (fingerprint >> block_offset) & ((1 << width) - 1)
            buckets[(block, value)].append(index)

    return {
        "dataset": str(dataset.resolve()),
        "max_distance": max_distance,
        "image_counts": {split: sum(record[0] == split for record in records) for split in SPLITS},
        "pairs": {
            pair: {
                "total": sum(distance_counts.values()),
                "by_distance": {
                    str(distance): distance_counts[distance] for distance in range(max_distance + 1)
                },
                "examples": examples[pair],
            }
            for pair, distance_counts in counts.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit perceptual leakage across YOLO splits")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-distance", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset, max_distance=args.max_distance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
