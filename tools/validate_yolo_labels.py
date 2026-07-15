from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

EXPECTED_CLASS_IDS = {0, 1, 2, 3}


def validate_label(path: Path) -> tuple[Counter[int], list[str]]:
    counts: Counter[int] = Counter()
    errors: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number}: expected 5 columns")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric value")
            continue
        if class_id not in EXPECTED_CLASS_IDS:
            errors.append(f"{path}:{line_number}: class {class_id} outside 0..3")
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{path}:{line_number}: NaN/Infinity")
        x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{path}:{line_number}: invalid normalized box")
        if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
            errors.append(f"{path}:{line_number}: box exceeds image bounds")
        counts[class_id] += 1
    return counts, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO etiketlerini 4 sınıflı kontrata göre doğrula"
    )
    parser.add_argument("labels", type=Path)
    parser.add_argument("--report", type=Path, default=Path("logs/label-validation.json"))
    args = parser.parse_args()
    files = sorted(args.labels.rglob("*.txt"))
    total: Counter[int] = Counter()
    errors: list[str] = []
    empty = 0
    for path in files:
        counts, current_errors = validate_label(path)
        total.update(counts)
        errors.extend(current_errors)
        empty += int(not path.read_text(encoding="utf-8").strip())
    report = {
        "label_files": len(files),
        "empty_negative_files": empty,
        "class_counts": {str(index): total[index] for index in range(4)},
        "error_count": len(errors),
        "errors": errors[:1000],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        raise SystemExit(f"label validation failed with {len(errors)} errors; see {args.report}")


if __name__ == "__main__":
    main()
