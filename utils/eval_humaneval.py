import argparse
import json
import math
import multiprocessing as mp
import re
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
    src = (text or "").strip()
    if not src:
        return ""
    m = re.match(r"^\s*```[^\n`]*\n?(.*?)\n?```\s*$", src, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return src


def extract_completion(raw_text: str, entry_point: str | None = None) -> str:
    src = (raw_text or "").strip()
    if not src:
        return ""

    # Prefer fenced code when present.
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", src, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        src = max(blocks, key=len).strip()
    else:
        src = strip_code_fence(src)

    # If model emits long prose, cut from function definition when available.
    if entry_point:
        anchor = f"def {entry_point}"
        pos = src.find(anchor)
        if pos >= 0:
            src = src[pos:]

    return src.strip()


def _eval_worker(
    query_prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    queue: mp.Queue,
) -> None:
    started_at = time.perf_counter()
    ns: dict[str, Any] = {}
    try:
        program = f"{query_prompt}\n{completion}\n"
        exec(program, ns)  # noqa: S102
        exec(test_code, ns)  # noqa: S102

        check = ns.get("check")
        candidate = ns.get(entry_point)
        if not callable(check):
            raise RuntimeError("check(...) not found after executing tests")
        if not callable(candidate):
            raise RuntimeError(f"entry_point '{entry_point}' is not callable")

        check(candidate)
        queue.put(
            {
                "passed": True,
                "error_type": "",
                "error_message": "",
                "duration_s": round(time.perf_counter() - started_at, 4),
            }
        )
    except Exception as e:  # noqa: BLE001
        queue.put(
            {
                "passed": False,
                "error_type": type(e).__name__,
                "error_message": "".join(traceback.format_exception_only(type(e), e)).strip(),
                "duration_s": round(time.perf_counter() - started_at, 4),
            }
        )


def eval_in_current_process(
    query_prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    ns: dict[str, Any] = {}
    try:
        program = f"{query_prompt}\n{completion}\n"
        exec(program, ns)  # noqa: S102
        exec(test_code, ns)  # noqa: S102

        check = ns.get("check")
        candidate = ns.get(entry_point)
        if not callable(check):
            raise RuntimeError("check(...) not found after executing tests")
        if not callable(candidate):
            raise RuntimeError(f"entry_point '{entry_point}' is not callable")

        check(candidate)
        return {
            "passed": True,
            "error_type": "",
            "error_message": "",
            "duration_s": round(time.perf_counter() - started_at, 4),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "passed": False,
            "error_type": type(e).__name__,
            "error_message": "".join(traceback.format_exception_only(type(e), e)).strip(),
            "duration_s": round(time.perf_counter() - started_at, 4),
        }


def run_single_eval(
    *,
    query_prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    timeout_s: float,
) -> dict[str, Any]:
    try:
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_eval_worker,
            args=(query_prompt, completion, test_code, entry_point, queue),
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
            }

        if queue.empty():
            return {
                "passed": False,
                "error_type": "NoResult",
                "error_message": "Worker exited without returning result",
                "duration_s": round(time.perf_counter() - t0, 4),
            }

        return queue.get()
    except PermissionError:
        # Fallback for restricted Windows sandbox where multiprocessing pipe creation is denied.
        return eval_in_current_process(query_prompt, completion, test_code, entry_point)


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
        description="Evaluate HumanEval outputs from API inference JSONL and compute pass@k metrics."
    )
    parser.add_argument(
        "--inference",
        type=Path,
        default=Path("data/icl/eval_outputs/siliconflow_deepseek_v32_mig_humaneval_k8.jsonl"),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/icl/eval_prompts/mig_humaneval_k8.jsonl"),
        help="Fallback source for query_prompt/tests/entry_point by query_id.",
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
    selector_params = selector_meta.get("params", {}) if isinstance(selector_meta.get("params", {}), dict) else {}
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

            entry_point = str(row.get("entry_point") or prompt_row.get("entry_point") or "").strip()
            tests = row.get("tests")
            if not isinstance(tests, dict) or "test" not in tests:
                tests = prompt_row.get("tests", {})
            test_code = str((tests or {}).get("test", "") or "")
            query_prompt = str(row.get("query_prompt") or prompt_row.get("query_prompt") or "").rstrip()

            completion_raw = str(row.get("completion_cleaned") or row.get("raw_output") or "")
            completion = extract_completion(completion_raw, entry_point=entry_point or None)

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

            if not query_prompt or not entry_point or not test_code or not completion:
                missing_case_count += 1
                result = {
                    "query_id": qid,
                    "method": query_method,
                    "sample_rank": rank,
                    "passed": False,
                    "error_type": "MissingField",
                    "error_message": "query_prompt/entry_point/test/completion missing",
                    "duration_s": 0.0,
                    "entry_point": entry_point,
                }
            else:
                result = run_single_eval(
                    query_prompt=query_prompt,
                    completion=completion,
                    test_code=test_code,
                    entry_point=entry_point,
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
                    "entry_point": entry_point,
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
            "methods_seen": sorted({str(r.get("method", "")).strip() for r in infer_rows if str(r.get("method", "")).strip()}),
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

    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "per_query.jsonl", per_query)
    write_jsonl(out_dir / "per_sample.jsonl", per_sample)
    write_json(
        out_dir / "run_manifest.json",
        {
            "run_name": out_dir.name,
            "created_at_local": datetime.now().isoformat(timespec="seconds"),
            "files": {
                "summary": "summary.json",
                "per_query": "per_query.jsonl",
                "per_sample": "per_sample.jsonl",
            },
        },
    )

    print(f"Evaluated queries: {len(query_order)}")
    print(f"Evaluated samples: {sample_total}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"Per-query: {out_dir / 'per_query.jsonl'}")
    print(f"Per-sample: {out_dir / 'per_sample.jsonl'}")


if __name__ == "__main__":
    main()
