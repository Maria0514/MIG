import ast
import gzip
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class BuildICLResult:
    mbpp_path: Path
    humaneval_path: Path
    mbpp_count: int
    humaneval_count: int


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


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".jsonl.gz") or path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "items", "records"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON structure in {path}")

    raise ValueError(f"Unsupported file type: {path}")


def _length_stats(prompt: str, solution: str) -> dict[str, int]:
    prompt = prompt or ""
    solution = solution or ""
    return {
        "prompt_chars": len(prompt),
        "solution_chars": len(solution),
        "prompt_lines": prompt.count("\n") + (1 if prompt else 0),
        "solution_lines": solution.count("\n") + (1 if solution else 0),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_mbpp_rows(records: list[dict[str, Any]], source: str, config_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(records):
        prompt = item.get("text") or item.get("prompt") or ""
        solution = item.get("code") or item.get("canonical_solution") or item.get("solution") or ""
        entry_point = _extract_entry_point_from_code(solution)
        tests_public = item.get("test_list") or []
        tests_hidden = item.get("challenge_test_list") or []
        tests_setup = item.get("test_setup_code") or ""
        task_id = item.get("task_id") or idx

        rows.append({
            "id": f"mbpp/{task_id}",
            "source": {
                "dataset": "mbpp",
                "config": config_name,
                "split": "train",
                "origin": source,
            },
            "task_type": "python_code_generation",
            "prompt": prompt,
            "canonical_solution": solution,
            "entry_point": entry_point,
            "tests": {
                "setup_code": tests_setup,
                "public_tests": tests_public,
                "hidden_tests": tests_hidden,
            },
            "meta": {
                "difficulty": "unknown",
                "quality_score": None,
                "length": _length_stats(prompt, solution),
            },
        })
    return rows


def _build_humaneval_rows(records: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(records):
        prompt = item.get("prompt") or ""
        solution = item.get("canonical_solution") or item.get("solution") or ""
        task_id = item.get("task_id") or idx
        rows.append({
            "id": f"humaneval/{task_id}",
            "source": {
                "dataset": "openai_humaneval",
                "split": "test",
                "origin": source,
            },
            "task_type": "python_code_generation",
            "prompt": prompt,
            "canonical_solution": solution,
            "entry_point": item.get("entry_point"),
            "tests": {
                "test": item.get("test") or "",
            },
            "meta": {
                "difficulty": "unknown",
                "quality_score": None,
                "length": _length_stats(prompt, solution),
            },
        })
    return rows


def _try_load_from_hf_dataset(path: str, split: str, config_name: Optional[str] = None) -> Optional[list[dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    try:
        if config_name is not None:
            ds = load_dataset(path, config_name, split=split)
        else:
            ds = load_dataset(path, split=split)
    except Exception:
        return None

    return [dict(item) for item in ds]


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_mbpp_records(mbpp_src: Optional[Path], mbpp_config: str) -> tuple[list[dict[str, Any]], str]:
    if mbpp_src is not None:
        return _read_json_records(mbpp_src), str(mbpp_src)

    local_default = _first_existing([
        Path("data/raw/mbpp_train.jsonl"),
        Path("data/raw/mbpp_train.jsonl.gz"),
        Path("data/raw/mbpp.jsonl"),
        Path("data/mbpp_train.jsonl"),
        Path("data/mbpp.jsonl"),
    ])
    if local_default is not None:
        return _read_json_records(local_default), str(local_default)

    hf_rows = _try_load_from_hf_dataset("mbpp", split="train", config_name=mbpp_config)
    if hf_rows is not None:
        return hf_rows, f"hf://mbpp/{mbpp_config}/train"

    raise RuntimeError(
        "Cannot load MBPP. Provide --mbpp-src (json/jsonl/jsonl.gz) or install `datasets` with network access."
    )


def _load_humaneval_records(humaneval_src: Optional[Path]) -> tuple[list[dict[str, Any]], str]:
    if humaneval_src is not None:
        return _read_json_records(humaneval_src), str(humaneval_src)

    local_default = _first_existing([
        Path("data/raw/humaneval_test.jsonl"),
        Path("data/raw/humaneval_test.jsonl.gz"),
        Path("data/raw/HumanEval.jsonl"),
        Path("data/raw/HumanEval.jsonl.gz"),
        Path("data/humaneval_test.jsonl"),
    ])
    if local_default is not None:
        return _read_json_records(local_default), str(local_default)

    hf_rows = _try_load_from_hf_dataset("openai_humaneval", split="test")
    if hf_rows is not None:
        return hf_rows, "hf://openai_humaneval/test"

    raise RuntimeError(
        "Cannot load HumanEval. Provide --humaneval-src (json/jsonl/jsonl.gz) or install `datasets` with network access."
    )


def build_icl_datasets(
    out_dir: str | Path = Path("data/icl"),
    mbpp_src: str | Path | None = None,
    humaneval_src: str | Path | None = None,
    mbpp_config: str = "sanitized",
    overwrite: bool = False,
) -> BuildICLResult:
    out_dir = Path(out_dir)
    mbpp_src_path = Path(mbpp_src) if mbpp_src is not None else None
    humaneval_src_path = Path(humaneval_src) if humaneval_src is not None else None

    mbpp_records, mbpp_origin = _load_mbpp_records(mbpp_src=mbpp_src_path, mbpp_config=mbpp_config)
    humaneval_records, humaneval_origin = _load_humaneval_records(humaneval_src=humaneval_src_path)

    mbpp_rows = _build_mbpp_rows(mbpp_records, source=mbpp_origin, config_name=mbpp_config)
    humaneval_rows = _build_humaneval_rows(humaneval_records, source=humaneval_origin)

    mbpp_path = out_dir / "mbpp_candidate_pool.jsonl"
    humaneval_path = out_dir / "humaneval_eval.jsonl"

    _write_jsonl(mbpp_path, mbpp_rows, overwrite=overwrite)
    _write_jsonl(humaneval_path, humaneval_rows, overwrite=overwrite)

    return BuildICLResult(
        mbpp_path=mbpp_path,
        humaneval_path=humaneval_path,
        mbpp_count=len(mbpp_rows),
        humaneval_count=len(humaneval_rows),
    )
