import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a valid_tag_path JSON list from JSONL records.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON file (JSON list of tags).")
    parser.add_argument(
        "--labels-key",
        type=str,
        default="query_labels",
        help="Primary list field to read labels from. Defaults to query_labels.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum corpus frequency required for a tag to be kept. Defaults to 1.",
    )
    return parser.parse_args()


def extract_labels(row: dict, labels_key: str) -> list[str]:
    raw = row.get(labels_key)
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if isinstance(item, str) and str(item).strip()]

    raw_fallback = row.get("query_tags_raw")
    if isinstance(raw_fallback, list):
        return [str(item).strip() for item in raw_fallback if isinstance(item, str) and str(item).strip()]

    ann = row.get("annotation") or {}
    instag = ann.get("instag") or {}
    content = instag.get("content")
    if isinstance(content, list):
        return [str(item).strip() for item in content if isinstance(item, str) and str(item).strip()]

    return []


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    counts: dict[str, int] = {}
    row_count = 0
    rows_with_labels = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row_count += 1
            row = json.loads(line)
            labels = extract_labels(row, args.labels_key)
            if labels:
                rows_with_labels += 1
                for tag in labels:
                    counts[tag] = counts.get(tag, 0) + 1

    sorted_tags = sorted(tag for tag, count in counts.items() if count >= args.min_freq)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sorted_tags, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.out),
                "labels_key": args.labels_key,
                "min_freq": args.min_freq,
                "row_count": row_count,
                "rows_with_labels": rows_with_labels,
                "tag_count": len(sorted_tags),
                "unique_tag_count_before_filter": len(counts),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
