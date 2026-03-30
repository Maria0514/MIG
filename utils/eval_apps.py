import argparse
import io
import json
import math
import multiprocessing as mp
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_code_fence(text: str) -> str:
    src = "" if text is None else str(text)
    if not src.strip():
        return ""
    probe = src.strip()
    m = re.match(r"^\s*```[^\n`]*\n?(.*?)\n?```\s*$", probe, flags=re.DOTALL)
    if m:
        return m.group(1).rstrip("\r\n")
    if probe.startswith("```"):
        first_newline = probe.find("\n")
        if first_newline >= 0:
            return probe[first_newline + 1 :].lstrip("\r\n")
        return ""
    return src.rstrip("\r\n")


def extract_completion(raw_text: str) -> str:
    src = "" if raw_text is None else str(raw_text)
    if not src.strip():
        return ""

    blocks = re.findall(r"```(?:python)?\s*(.*?)```", src, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        src = max(blocks, key=len).rstrip("\r\n")
    else:
        src = strip_code_fence(src)

    return src.rstrip("\r\n")


def normalize_output_text(text: str) -> str:
    src = "" if text is None else str(text)
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in src.split("\n")]
    return "\n".join(lines).strip()


def expected_output_matches(actual: str, expected: Any) -> bool:
    actual_norm = normalize_output_text(actual)
    if isinstance(expected, list):
        return any(actual_norm == normalize_output_text(item) for item in expected)
    return actual_norm == normalize_output_text(expected)


def _run_code_with_stdin(code: str, input_data: str) -> str:
    import builtins

    old_stdin = sys.stdin
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_argv = sys.argv

    stdin_obj = io.StringIO(input_data)
    stdout_obj = io.StringIO()
    stderr_obj = io.StringIO()

    try:
        sys.stdin = stdin_obj
        sys.stdout = stdout_obj
        sys.stderr = stderr_obj
        sys.argv = ["solution.py"]
        ns: dict[str, Any] = {"__name__": "__main__"}
        exec(code, ns)  # noqa: S102
    except SystemExit:
        pass
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.argv = old_argv

    return stdout_obj.getvalue()


def _eval_worker(
    completion: str,
    input_output: dict[str, Any],
    queue: mp.Queue,
) -> None:
    started_at = time.perf_counter()
    try:
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise RuntimeError("tests.input_output.inputs/outputs must both be lists")
        if len(inputs) != len(outputs):
            raise RuntimeError("tests.input_output inputs/outputs length mismatch")

        passed_count = 0
        case_results: list[dict[str, Any]] = []
        for idx, (inp, exp) in enumerate(zip(inputs, outputs)):
            actual = _run_code_with_stdin(completion, str(inp))
            passed = expected_output_matches(actual, exp)
            case_results.append(
                {
                    "case_index": idx,
                    "passed": passed,
                    "input_preview": str(inp)[:200],
                    "expected_preview": (exp[:200] if isinstance(exp, str) else exp),
                    "actual_preview": actual[:200],
                }
            )
            if not passed:
                queue.put(
                    {
                        "passed": False,
                        "error_type": "WrongAnswer",
                        "error_message": f"Mismatch at case {idx}",
                        "duration_s": round(time.perf_counter() - started_at, 4),
                        "passed_case_count": passed_count,
                        "total_case_count": len(inputs),
                        "case_results": case_results,
                    }
                )
                return
            passed_count += 1

        queue.put(
            {
                "passed": True,
                "error_type": "",
                "error_message": "",
                "duration_s": round(time.perf_counter() - started_at, 4),
                "passed_case_count": passed_count,
                "total_case_count": len(inputs),
                "case_results": case_results,
            }
        )
    except Exception as e:  # noqa: BLE001
        queue.put(
            {
                "passed": False,
                "error_type": type(e).__name__,
                "error_message": "".join(traceback.format_exception_only(type(e), e)).strip(),
                "duration_s": round(time.perf_counter() - started_at, 4),
                "passed_case_count": 0,
                "total_case_count": 0,
                "case_results": [],
            }
        )


def eval_in_current_process(
    completion: str,
    input_output: dict[str, Any],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise RuntimeError("tests.input_output.inputs/outputs must both be lists")
        if len(inputs) != len(outputs):
            raise RuntimeError("tests.input_output inputs/outputs length mismatch")

        passed_count = 0
        case_results: list[dict[str, Any]] = []
        for idx, (inp, exp) in enumerate(zip(inputs, outputs)):
            actual = _run_code_with_stdin(completion, str(inp))
            passed = expected_output_matches(actual, exp)
            case_results.append(
                {
                    "case_index": idx,
                    "passed": passed,
                    "input_preview": str(inp)[:200],
                    "expected_preview": (exp[:200] if isinstance(exp, str) else exp),
                    "actual_preview": actual[:200],
                }
            )
            if not passed:
                return {
                    "passed": False,
                    "error_type": "WrongAnswer",
                    "error_message": f"Mismatch at case {idx}",
                    "duration_s": round(time.perf_counter() - started_at, 4),
                    "passed_case_count": passed_count,
                    "total_case_count": len(inputs),
                    "case_results": case_results,
                }
            passed_count += 1

        return {
            "passed": True,
            "error_type": "",
            "error_message": "",
            "duration_s": round(time.perf_counter() - started_at, 4),
            "passed_case_count": passed_count,
            "total_case_count": len(inputs),
            "case_results": case_results,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "passed": False,
            "error_type": type(e).__name__,
            "error_message": "".join(traceback.format_exception_only(type(e), e)).strip(),
            "duration_s": round(time.perf_counter() - started_at, 4),
            "passed_case_count": 0,
            "total_case_count": 0,
            "case_results": [],
        }


def run_single_eval(
    *,
    completion: str,
    input_output: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    try:
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_eval_worker,
            args=(completion, input_output, queue),
        )
        t0 = time.perf_counter()
        proc.start()
        proc.join(timeout=timeout_s)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)
            return {
                "passed": False,
                "error_type": "Timeout",
                "error_message": f"Execution exceeded {timeout_s}s",
                "duration_s": round(time.perf_counter() - t0, 4),
                "passed_case_count": 0,
                "total_case_count": 0,
                "case_results": [],
            }

        if queue.empty():
            return {
                "passed": False,
                "error_type": "NoResult",
                "error_message": "Worker exited without returning result",
                "duration_s": round(time.perf_counter() - t0, 4),
                "passed_case_count": 0,
                "total_case_count": 0,
                "case_results": [],
            }

        return queue.get()
    except PermissionError:
        return eval_in_current_process(completion, input_output)


def pass_at_k_estimator(n: int, c: int, k: int) -> float | None:
    if n <= 0 or c < 0 or c > n or k <= 0:
        return None
    if n < k:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def parse_k_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k > 0:
            values.append(k)
    if not values:
        values = [1]
    return sorted(set(values))


def mean_or_none(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {}


def build_unique_run_dir(base_dir: Path, run_name: str) -> Path:
    target = base_dir / run_name
    if not target.exists():
        return target
    idx = 1
    while True:
        cand = base_dir / f"{run_name}_{idx:02d}"
        if not cand.exists():
            return cand
        idx += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate APPS outputs from API inference JSONL and compute pass@k metrics."
    )
    parser.add_argument(
        "--inference",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Fallback source for tests/query_prompt by query_id.",
    )
    parser.add_argument("--base-out-dir", type=Path, default=Path("data/icl/eval_metrics"))
    parser.add_argument("--run-name", type=str, default="", help="Optional run name. If empty, auto-generated.")
    parser.add_argument(
        "--selector-meta",
        type=Path,
        default=Path("data/icl/metadata/humaneval_to_demos.meta.json"),
        help="Selector metadata file used to annotate penalty and MIG settings.",
    )
    parser.add_argument("--is-mig-algorithm", dest="is_mig_algorithm", action="store_true")
    parser.add_argument("--not-mig-algorithm", dest="is_mig_algorithm", action="store_false")
    parser.set_defaults(is_mig_algorithm=True)
    parser.add_argument(
        "--difficulty-label-scheme",
        type=str,
        default="easy/medium/hard from deita complexity quantiles",
    )
    parser.add_argument("--k-list", type=str, default="1,5")
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument("--max-samples-per-query", type=int, default=0, help="0 means use all samples.")
    parser.add_argument("--query-limit", type=int, default=0, help="0 means evaluate all query groups.")
    parser.add_argument("--status-ok-only", dest="status_ok_only", action="store_true")
    parser.add_argument("--include-error-status", dest="status_ok_only", action="store_false")
    parser.set_defaults(status_ok_only=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.inference.exists():
        raise FileNotFoundError(f"Inference file not found: {args.inference}")
    if not args.prompts.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompts}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"{args.inference.stem}_{ts}"
    out_dir = build_unique_run_dir(args.base_out_dir, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    selector_meta = load_json(args.selector_meta)
    params_obj = selector_meta.get("params")
    if isinstance(params_obj, dict):
        selector_params = params_obj
    elif isinstance(selector_meta, dict):
        selector_params = selector_meta
    else:
        selector_params = {}

    lambda_quality = selector_params.get("lambda_quality")
    lambda_len = selector_params.get("lambda_len")
    lambda_red = selector_params.get("lambda_red")
    lambda_diff = selector_params.get("lambda_diff")
    use_difficulty_penalty = selector_params.get("use_difficulty_penalty")

    k_list = parse_k_list(args.k_list)
    infer_rows = read_jsonl(args.inference)
    prompt_rows = read_jsonl(args.prompts)
    prompt_map = {str(r.get("query_id", "")).strip(): r for r in prompt_rows if str(r.get("query_id", "")).strip()}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    query_order: list[str] = []
    for row in infer_rows:
        qid = str(row.get("query_id", "")).strip()
        if not qid:
            continue
        if args.status_ok_only and str(row.get("status", "")).lower() != "ok":
            continue
        if qid not in grouped:
            query_order.append(qid)
        grouped[qid].append(row)

    if args.query_limit > 0:
        query_order = query_order[: args.query_limit]

    per_sample: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []

    topk_values: dict[int, list[float]] = {k: [] for k in k_list}
    topk_strict_values: dict[int, list[float]] = {k: [] for k in k_list}
    estimator_values: dict[int, list[float]] = {k: [] for k in k_list}

    eval_latencies: list[float] = []
    infer_latencies: list[float] = []
    prompt_tokens: list[float] = []
    completion_tokens: list[float] = []
    total_tokens: list[float] = []

    missing_case_count = 0
    sample_total = 0
    sample_passed = 0

    for qid in query_order:
        rows = grouped.get(qid, [])
        if args.max_samples_per_query > 0:
            rows = rows[: args.max_samples_per_query]

        query_pass_flags: list[bool] = []
        query_eval_count = 0
        query_method = str(rows[0].get("method", "")) if rows else ""

        for rank, row in enumerate(rows):
            sample_total += 1
            prompt_row = prompt_map.get(qid, {})

            tests = row.get("tests")
            if not isinstance(tests, dict) or "input_output" not in tests:
                tests = prompt_row.get("tests", {})
            input_output = (tests or {}).get("input_output", {})

            completion_raw = str(row.get("completion_cleaned") or row.get("raw_output") or "")
            completion = extract_completion(completion_raw)

            usage = row.get("usage")
            if isinstance(usage, dict):
                pt = usage.get("prompt_tokens")
                ct = usage.get("completion_tokens")
                tt = usage.get("total_tokens")
                if isinstance(pt, (int, float)):
                    prompt_tokens.append(float(pt))
                if isinstance(ct, (int, float)):
                    completion_tokens.append(float(ct))
                if isinstance(tt, (int, float)):
                    total_tokens.append(float(tt))

            inf_lat = row.get("latency_s")
            if isinstance(inf_lat, (int, float)):
                infer_latencies.append(float(inf_lat))

            if not isinstance(input_output, dict) or not completion:
                missing_case_count += 1
                result = {
                    "passed": False,
                    "error_type": "MissingField",
                    "error_message": "tests.input_output or completion missing",
                    "duration_s": 0.0,
                    "passed_case_count": 0,
                    "total_case_count": 0,
                    "case_results": [],
                }
            else:
                result = run_single_eval(
                    completion=completion,
                    input_output=input_output,
                    timeout_s=args.timeout_s,
                )
                query_eval_count += 1
                eval_latencies.append(float(result.get("duration_s", 0.0) or 0.0))

            passed = bool(result.get("passed", False))
            if passed:
                sample_passed += 1
            query_pass_flags.append(passed)

            per_sample.append(
                {
                    "query_id": qid,
                    "method": query_method,
                    "sample_rank": rank,
                    "passed": passed,
                    "error_type": result.get("error_type", ""),
                    "error_message": result.get("error_message", ""),
                    "eval_duration_s": result.get("duration_s", 0.0),
                    "passed_case_count": result.get("passed_case_count", 0),
                    "total_case_count": result.get("total_case_count", 0),
                }
            )

        n = len(query_pass_flags)
        c = sum(1 for x in query_pass_flags if x)
        query_summary: dict[str, Any] = {
            "query_id": qid,
            "method": query_method,
            "n_samples": n,
            "n_evaluated_samples": query_eval_count,
            "n_correct": c,
            "top1_pass": bool(query_pass_flags[0]) if n > 0 else False,
            "any_pass": c > 0,
        }

        for k in k_list:
            topk_any_available = any(query_pass_flags[: min(k, n)]) if n > 0 else False
            query_summary[f"top{k}_pass_any_available"] = topk_any_available
            topk_values[k].append(1.0 if topk_any_available else 0.0)

            if n >= k:
                topk_strict = any(query_pass_flags[:k])
                query_summary[f"top{k}_pass_strict"] = topk_strict
                topk_strict_values[k].append(1.0 if topk_strict else 0.0)
            else:
                query_summary[f"top{k}_pass_strict"] = None

            est = pass_at_k_estimator(n=n, c=c, k=k)
            query_summary[f"pass@{k}_estimator"] = est
            if est is not None:
                estimator_values[k].append(est)

        per_query.append(query_summary)

    summary: dict[str, Any] = {
        "run_name": out_dir.name,
        "inference_file": str(args.inference),
        "prompts_file": str(args.prompts),
        "selector_meta_file": str(args.selector_meta) if args.selector_meta.exists() else "",
        "out_dir": str(out_dir),
        "k_list": k_list,
        "timeout_s": args.timeout_s,
        "max_samples_per_query": args.max_samples_per_query,
        "status_ok_only": args.status_ok_only,
        "query_total": len(query_order),
        "sample_total": sample_total,
        "sample_passed": sample_passed,
        "sample_pass_rate": (sample_passed / sample_total) if sample_total > 0 else None,
        "missing_case_count": missing_case_count,
        "avg_eval_duration_s": mean_or_none(eval_latencies),
        "avg_infer_latency_s": mean_or_none(infer_latencies),
        "avg_prompt_tokens": mean_or_none(prompt_tokens),
        "avg_completion_tokens": mean_or_none(completion_tokens),
        "avg_total_tokens": mean_or_none(total_tokens),
        "experiment_context": {
            "is_mig_algorithm": bool(args.is_mig_algorithm),
            "methods_seen": sorted(
                {
                    str(r.get("method", "")).strip()
                    for r in infer_rows
                    if str(r.get("method", "")).strip()
                }
            ),
            "difficulty_label_scheme": args.difficulty_label_scheme,
            "penalty_config": {
                "lambda_quality": lambda_quality,
                "lambda_len": lambda_len,
                "lambda_red": lambda_red,
                "lambda_diff": lambda_diff,
                "length_penalty_enabled": bool(isinstance(lambda_len, (int, float)) and float(lambda_len) != 0.0),
                "redundancy_penalty_enabled": bool(isinstance(lambda_red, (int, float)) and float(lambda_red) != 0.0),
                "difficulty_penalty_enabled": bool(use_difficulty_penalty) or bool(isinstance(lambda_diff, (int, float)) and float(lambda_diff) != 0.0),
            },
            "selector_params_raw": selector_params,
        },
        "metrics": {},
    }

    for k in k_list:
        summary["metrics"][f"pass@{k}_topk_any_available"] = mean_or_none(topk_values[k])
        summary["metrics"][f"pass@{k}_topk_strict"] = mean_or_none(topk_strict_values[k])
        summary["metrics"][f"pass@{k}_estimator"] = mean_or_none(estimator_values[k])
        summary["metrics"][f"pass@{k}_strict_denominator"] = len(topk_strict_values[k])
        summary["metrics"][f"pass@{k}_estimator_denominator"] = len(estimator_values[k])

    failed_cases = [row for row in per_sample if not bool(row.get("passed", False))]
    summary["failed_case_count"] = len(failed_cases)

    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "per_query.jsonl", per_query)
    write_jsonl(out_dir / "per_sample.jsonl", per_sample)
    write_jsonl(out_dir / "failed_cases.jsonl", failed_cases)
    write_json(
        out_dir / "run_manifest.json",
        {
            "run_name": out_dir.name,
            "created_at_local": datetime.now().isoformat(timespec="seconds"),
            "eval_dataset": "apps_test",
            "files": {
                "summary": "summary.json",
                "per_query": "per_query.jsonl",
                "per_sample": "per_sample.jsonl",
                "failed_cases": "failed_cases.jsonl",
            },
        },
    )

    print(f"Evaluated queries: {len(query_order)}")
    print(f"Evaluated samples: {sample_total}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"Per-query: {out_dir / 'per_query.jsonl'}")
    print(f"Per-sample: {out_dir / 'per_sample.jsonl'}")
    print(f"Failed-cases: {out_dir / 'failed_cases.jsonl'}")


if __name__ == "__main__":
    main()
