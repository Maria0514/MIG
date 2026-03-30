import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _extract_entry_point_from_code(code: str) -> str | None:
    if not code:
        return None
    try:
        module = ast.parse(code)
    except SyntaxError:
        return None
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def _length_stats(prompt: str, solution: str) -> dict[str, int]:
    prompt = prompt or ""
    solution = solution or ""
    return {
        "prompt_chars": len(prompt),
        "solution_chars": len(solution),
        "prompt_lines": prompt.count("\n") + (1 if prompt else 0),
        "solution_lines": solution.count("\n") + (1 if solution else 0),
    }


def _parse_json_field(value: Any) -> tuple[Any, bool]:
    """Return parsed object and whether parsing succeeded (or was unnecessary)."""
    if value is None:
        return None, True
    if isinstance(value, (dict, list)):
        return value, True
    if isinstance(value, str):
        try:
            return json.loads(value), True
        except json.JSONDecodeError:
            return value, False
    return value, False


def _normalize_solutions(raw: Any) -> tuple[list[str], bool]:
    parsed, ok = _parse_json_field(raw)
    if isinstance(parsed, list):
        sols = [str(x) for x in parsed if str(x).strip()]
        return sols, ok
    if isinstance(parsed, str) and parsed.strip():
        return [parsed], ok
    return [], ok


def _build_prompt(question: str, starter_code: str) -> str:
    question = (question or "").strip()
    starter_code = (starter_code or "").strip()
    if starter_code:
        return f"{question}\n\n# Starter Code\n{starter_code}\n"
    return f"{question}\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert APPS test.jsonl to a HumanEval-like eval JSONL schema."
    )
    parser.add_argument("--src", type=Path, default=Path("data/raw/apps/test.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/icl/apps_test_eval.jsonl"))
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=Path("data/icl/apps_test_eval.meta.json"),
        help="Conversion metadata output path.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-name", type=str, default="codeparrot/apps")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"Input file not found: {args.src}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    parse_ok_solutions = 0
    parse_ok_input_output = 0
    missing_question = 0
    empty_solution = 0
    with_entry_point = 0

    with open(args.src, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            if args.limit > 0 and written >= args.limit:
                break

            row = json.loads(line)
            raw_id = row.get("id", written)
            question = row.get("question") or ""
            starter_code = row.get("starter_code") or ""
            if not str(question).strip():
                missing_question += 1

            solutions, ok_solutions = _normalize_solutions(row.get("solutions"))
            if ok_solutions:
                parse_ok_solutions += 1
            canonical_solution = solutions[0] if solutions else ""
            if not canonical_solution.strip():
                empty_solution += 1

            input_output, ok_io = _parse_json_field(row.get("input_output"))
            if ok_io:
                parse_ok_input_output += 1

            prompt = _build_prompt(str(question), str(starter_code))
            entry_point = _extract_entry_point_from_code(canonical_solution)
            if entry_point:
                with_entry_point += 1

            out_row = {
                "id": f"apps/{args.split}/{raw_id}",
                "source": {
                    "dataset": args.dataset_name,
                    "split": args.split,
                    "origin": str(args.src),
                },
                "task_type": "python_code_generation",
                "prompt": prompt,
                "canonical_solution": canonical_solution,
                "entry_point": entry_point,
                "tests": {
                    "test": "",
                    "input_output": input_output,
                },
                "meta": {
                    "difficulty": row.get("difficulty", "unknown"),
                    "quality_score": None,
                    "url": row.get("url", ""),
                    "length": _length_stats(prompt, canonical_solution),
                    "solution_count": len(solutions),
                },
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1

    meta = {
        "src": str(args.src),
        "out": str(args.out),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "limit": args.limit,
        "stats": {
            "total_rows_scanned": total,
            "rows_written": written,
            "missing_question_rows": missing_question,
            "empty_canonical_solution_rows": empty_solution,
            "rows_with_entry_point": with_entry_point,
            "solutions_parse_ok_rows": parse_ok_solutions,
            "input_output_parse_ok_rows": parse_ok_input_output,
        },
    }
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Done. wrote={written} -> {args.out}")
    print(f"Meta: {args.meta_out}")


if __name__ == "__main__":
    main()
