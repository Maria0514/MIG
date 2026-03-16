import argparse
import json
import os
from pathlib import Path
from typing import Any

from mig.query_tagger import (
    HFInsTagger,
    OpenAICompatInsTagger,
    tag_queries,
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
        description="Tag input ICL queries with InsTagger and project to valid_tag_path."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/icl/humaneval_eval.jsonl"),
        help="Input query dataset (.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/icl/humaneval_eval_tagged.jsonl"),
        help="Output tagged query dataset (.jsonl).",
    )
    parser.add_argument(
        "--valid-tag-path",
        type=Path,
        default=Path("configs/valid_tag_path_openhermes_python.json"),
        help="Path to valid tag list JSON.",
    )
    parser.add_argument(
        "--backend",
        choices=("hf", "openai_compat"),
        default="hf",
        help="Tagging backend.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="OFA-Sys/InsTagger",
        help="Model name/path for backend.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base (when backend=openai_compat).",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Env var name for API key (backend=openai_compat).",
    )
    parser.add_argument(
        "--api-mode",
        choices=("auto", "chat", "completion"),
        default="auto",
        help="OpenAI-compatible endpoint mode when backend=openai_compat.",
    )
    parser.add_argument(
        "--text-key",
        type=str,
        default="prompt",
        help="Text field used as query input.",
    )
    parser.add_argument(
        "--id-key",
        type=str,
        default="id",
        help="ID field for output mapping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for local HF backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise FileNotFoundError(f"Input not found: {args.src}")
    if not args.valid_tag_path.exists():
        raise FileNotFoundError(f"valid_tag_path not found: {args.valid_tag_path}")

    queries = read_jsonl(args.src)
    with open(args.valid_tag_path, "r", encoding="utf-8") as f:
        valid_tags = json.load(f)

    if args.backend == "hf":
        tagger = HFInsTagger(model_name_or_path=args.model)
    else:
        api_key = os.environ.get(args.api_key_env, "EMPTY")
        tagger = OpenAICompatInsTagger(
            api_base=args.api_base,
            model=args.model,
            api_key=api_key,
            api_mode=args.api_mode,
        )

    tagged = tag_queries(
        query_records=queries,
        valid_tags=valid_tags,
        tagger=tagger,
        text_key=args.text_key,
        id_key=args.id_key,
        batch_size=args.batch_size,
    )

    out_rows: list[dict[str, Any]] = []
    for src_row, t in zip(queries, tagged):
        row = dict(src_row)
        row["query_labels"] = t.valid_tags
        row["query_tags_raw"] = t.raw_tags
        row["query_tagger_raw_text"] = t.raw_generation
        out_rows.append(row)

    write_jsonl(args.out, out_rows)

    n = len(out_rows)
    avg_raw = (sum(len(x.raw_tags) for x in tagged) / n) if n else 0.0
    avg_valid = (sum(len(x.valid_tags) for x in tagged) / n) if n else 0.0
    print(f"Tagged queries: {n}")
    print(f"Avg raw tags/query: {avg_raw:.2f}")
    print(f"Avg valid tags/query: {avg_valid:.2f}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
