import argparse
import json
from pathlib import Path
from typing import Any

from mig.difficulty import (
    annotate_record_difficulty,
    collect_difficulty_scores,
    fit_difficulty_thresholds,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate candidate pool with difficulty labels from DEITA complexity scores."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Input pool jsonl.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output pool jsonl with annotation.difficulty.",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="Optional metadata json output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise FileNotFoundError(f"Input not found: {args.src}")

    rows = read_jsonl(args.src)
    scores = collect_difficulty_scores(rows)
    thresholds = fit_difficulty_thresholds(scores)

    for row in rows:
        annotate_record_difficulty(row, thresholds)

    write_jsonl(args.out, rows)

    meta = {
        "source": str(args.src),
        "rows": len(rows),
        "difficulty_thresholds": {
            "easy_max": thresholds.low,
            "medium_max": thresholds.high,
        },
        "score_min": min(scores),
        "score_max": max(scores),
        "score_avg": sum(scores) / len(scores),
        "label_counts": {
            "easy": sum(1 for row in rows if row["annotation"]["difficulty"]["label"] == "easy"),
            "medium": sum(1 for row in rows if row["annotation"]["difficulty"]["label"] == "medium"),
            "hard": sum(1 for row in rows if row["annotation"]["difficulty"]["label"] == "hard"),
        },
        "output": str(args.out),
    }

    if args.meta_out is not None:
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Rows: {len(rows)}")
    print(f"Thresholds: easy<= {thresholds.low:.6f}, medium<= {thresholds.high:.6f}, hard> {thresholds.high:.6f}")
    print(f"Label counts: {meta['label_counts']}")
    print(f"Output: {args.out}")
    if args.meta_out is not None:
        print(f"Meta: {args.meta_out}")


if __name__ == "__main__":
    main()

