import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from mig.query_tagger import (
    HFInsTagger,
    OpenAICompatInsTagger,
    build_instag_prompt,
    parse_instag_output,
    project_tags_to_valid_set,
)


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def load_done_ids(path: Path, id_key: str) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for _, row in iter_jsonl(path):
        rid = str(row.get(id_key, "")).strip()
        if rid:
            done.add(rid)
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-tag ICL queries with InsTagger and write JSONL incrementally."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/icl/apps_test_eval.jsonl"),
        help="Input query dataset (.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/icl/apps_test_eval_tagged.jsonl"),
        help="Output tagged dataset (.jsonl).",
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
    parser.add_argument("--text-key", type=str, default="prompt", help="Text field used as query input.")
    parser.add_argument("--id-key", type=str, default="id", help="ID field.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max-records", type=int, default=0, help="Tag at most N records. 0 means all.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N tagged rows.")
    parser.add_argument(
        "--output-mode",
        choices=("full", "minimal"),
        default="full",
        help="full keeps original row; minimal keeps only id/prompt/tests/entry_point/source/task_type/meta.",
    )
    parser.add_argument("--resume", action="store_true", help="Append and skip IDs already in output.")
    return parser.parse_args()


def build_out_row(row: dict[str, Any], *, output_mode: str, id_key: str, text_key: str) -> dict[str, Any]:
    if output_mode == "full":
        return dict(row)

    out: dict[str, Any] = {}
    for key in (id_key, text_key, "tests", "entry_point", "source", "task_type", "meta"):
        if key in row:
            out[key] = row[key]
    return out


def main() -> None:
    args = parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"Input not found: {args.src}")
    if not args.valid_tag_path.exists():
        raise FileNotFoundError(f"valid_tag_path not found: {args.valid_tag_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    with open(args.valid_tag_path, "r", encoding="utf-8") as f:
        valid_tags = json.load(f)
    if not isinstance(valid_tags, list):
        raise ValueError("valid_tag_path must contain a JSON list of tags.")

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

    done_ids: set[str] = set()
    open_mode = "w"
    if args.resume:
        done_ids = load_done_ids(args.out, args.id_key)
        open_mode = "a"
        print(f"[resume] already tagged rows: {len(done_ids)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.out, open_mode, encoding="utf-8")
    try:
        buffered_rows: list[dict[str, Any]] = []
        buffered_prompts: list[str] = []
        total_seen = 0
        total_tagged = 0
        total_raw_tags = 0
        total_valid_tags = 0
        start_t = time.time()

        def flush_batch() -> None:
            nonlocal total_tagged, total_raw_tags, total_valid_tags
            if not buffered_rows:
                return
            generations = tagger.generate(buffered_prompts)
            if len(generations) != len(buffered_rows):
                raise RuntimeError(
                    f"Generation size mismatch: got={len(generations)} expected={len(buffered_rows)}"
                )
            for src_row, gen in zip(buffered_rows, generations):
                raw_tags = parse_instag_output(gen)
                valid = project_tags_to_valid_set(raw_tags, valid_tags)
                out_row = build_out_row(
                    src_row,
                    output_mode=args.output_mode,
                    id_key=args.id_key,
                    text_key=args.text_key,
                )
                out_row["query_labels"] = valid
                out_row["query_tags_raw"] = raw_tags
                out_row["query_tagger_raw_text"] = gen
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                rid = str(src_row.get(args.id_key, "")).strip()
                if rid:
                    done_ids.add(rid)
                total_tagged += 1
                total_raw_tags += len(raw_tags)
                total_valid_tags += len(valid)

            fout.flush()
            buffered_rows.clear()
            buffered_prompts.clear()

        for line_no, row in iter_jsonl(args.src):
            total_seen += 1
            rid = str(row.get(args.id_key, "")).strip()
            if args.resume and rid and rid in done_ids:
                continue

            text = str(row.get(args.text_key, "") or "").strip()
            prompt = build_instag_prompt(text)
            buffered_rows.append(row)
            buffered_prompts.append(prompt)

            if len(buffered_rows) >= args.batch_size:
                flush_batch()
                if args.progress_every > 0 and total_tagged > 0 and total_tagged % args.progress_every == 0:
                    elapsed = time.time() - start_t
                    speed = total_tagged / elapsed if elapsed > 0 else 0.0
                    print(f"[progress] tagged={total_tagged} seen={total_seen} speed={speed:.2f} rows/s")

            if args.max_records > 0 and total_tagged >= args.max_records:
                break

        flush_batch()

        elapsed = time.time() - start_t
        avg_raw = (total_raw_tags / total_tagged) if total_tagged else 0.0
        avg_valid = (total_valid_tags / total_tagged) if total_tagged else 0.0
        print(f"Done. seen={total_seen} tagged={total_tagged} elapsed_s={elapsed:.1f}")
        print(f"Avg raw tags/query: {avg_raw:.2f}")
        print(f"Avg valid tags/query: {avg_valid:.2f}")
        print(f"Output: {args.out}")
    finally:
        fout.close()


if __name__ == "__main__":
    main()
