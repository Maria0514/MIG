import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def run_cmd(cmd: list[str]) -> None:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[RUN] {printable}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {printable}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return obj
    return {}


def load_eval_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def method_k_from_config(config: dict[str, Any], method: str) -> int | None:
    section = config.get("icl_eval")
    if not isinstance(section, dict):
        section = config
    kb = section.get("k_by_method")
    if not isinstance(kb, dict):
        return None
    raw = kb.get(method)
    if isinstance(raw, int) and raw >= 0:
        return raw
    return None


def parse_methods(text: str) -> list[str]:
    methods: list[str] = []
    for raw in text.split(","):
        m = raw.strip().lower()
        if not m:
            continue
        if m == "similarity":
            m = "sim"
        if m not in ("zero", "random", "sim"):
            raise ValueError(f"Unsupported method: {m}")
        methods.append(m)
    if not methods:
        raise ValueError("No method specified.")
    return methods


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "k",
        "run_name",
        "pass@1_topk_strict",
        "pass@5_topk_strict",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_total_tokens",
        "avg_infer_latency_s",
        "sample_pass_rate",
        "sample_total",
        "sample_passed",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| method | k | pass@1 | pass@5 | avg_prompt_tokens | avg_completion_tokens | "
        "avg_total_tokens | avg_infer_latency_s | sample_pass_rate |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row.get('method','')} | {row.get('k','')} | {row.get('pass@1_topk_strict')} | "
            f"{row.get('pass@5_topk_strict')} | {row.get('avg_prompt_tokens')} | "
            f"{row.get('avg_completion_tokens')} | {row.get('avg_total_tokens')} | "
            f"{row.get('avg_infer_latency_s')} | {row.get('sample_pass_rate')} |\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline end-to-end pipeline: mapping -> prompts -> API inference -> HumanEval eval."
    )
    parser.add_argument("--methods", type=str, default="zero,random,sim")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/icl_eval_config.json"),
        help="Optional experiment config. When k_by_method exists, it overrides --k per method.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--pool-limit", type=int, default=0)

    parser.add_argument("--pool", type=Path, default=Path("data/icl/openhermes_python_related.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/icl/humaneval_eval_tagged.jsonl"))
    parser.add_argument("--query-id-key", type=str, default="id")
    parser.add_argument("--query-text-keys", type=str, default="prompt")

    parser.add_argument("--mapping-dir", type=Path, default=Path("data/icl/mappings"))
    parser.add_argument("--prompts-dir", type=Path, default=Path("data/icl/eval_prompts"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("data/icl/eval_outputs"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("data/icl/eval_metrics"))
    parser.add_argument("--report-dir", type=Path, default=Path("data/icl/eval_reports"))

    parser.add_argument("--run-prefix", type=str, default="v32_baseline")
    parser.add_argument("--timestamp", type=str, default="")

    parser.add_argument("--model", type=str, default="nvidia/openai/gpt-oss-120b")
    parser.add_argument("--base-url", type=str, default="https://inference-api.nvidia.com")
    parser.add_argument("--api-key-env", type=str, default="SILICONFLOW_API_KEY")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Override max_tokens for every request. <=0 means use prompt/config task defaults.",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--retry", type=int, default=5)
    parser.add_argument("--retry-backoff-s", type=float, default=8.0)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--rate-limit-qps", type=float, default=0.3)
    parser.add_argument("--bucket-capacity", type=float, default=1.0)
    parser.add_argument("--save-raw-response", action="store_true")
    parser.add_argument("--dry-run-infer", action="store_true")

    parser.add_argument("--eval-k-list", type=str, default="1,5")
    parser.add_argument("--eval-timeout-s", type=float, default=8.0)
    parser.add_argument("--eval-max-samples-per-query", type=int, default=0)

    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")

    parser.add_argument("--embedding-model", type=str, default="models/e5-mistral-7b-instruct")
    parser.add_argument("--embedding-device", type=str, default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-max-seq-length", type=int, default=512)
    parser.add_argument("--pool-text-max-chars", type=int, default=1200)
    parser.add_argument("--query-text-max-chars", type=int, default=1200)
    parser.add_argument("--similarity-query-prompt-name", type=str, default="sts_query")
    parser.add_argument("--similarity-query-prompt", type=str, default="")
    parser.add_argument(
        "--pool-emb-cache",
        type=Path,
        default=Path("data/icl/cache/openhermes_e5m7b_pool_emb_f16.npy"),
    )
    parser.add_argument("--embedding-dtype", type=str, default="float16", choices=("float16", "float32"))
    return parser.parse_args()


def build_common_map_cmd(args: argparse.Namespace, method: str, k_eff: int, out_map: Path, out_meta: Path) -> list[str]:
    cmd = [
        sys.executable,
        "utils/build_baseline_mappings.py",
        "--method",
        method,
        "--k",
        str(k_eff),
        "--pool",
        str(args.pool),
        "--queries",
        str(args.queries),
        "--query-id-key",
        args.query_id_key,
        "--query-text-keys",
        args.query_text_keys,
        "--seed",
        str(args.seed),
        "--query-limit",
        str(args.query_limit),
        "--pool-limit",
        str(args.pool_limit),
        "--out",
        str(out_map),
        "--meta-out",
        str(out_meta),
    ]
    if method == "sim":
        cmd.extend(
            [
                "--embedding-model",
                args.embedding_model,
                "--embedding-device",
                args.embedding_device,
                "--embedding-batch-size",
                str(args.embedding_batch_size),
                "--embedding-max-seq-length",
                str(args.embedding_max_seq_length),
                "--pool-text-max-chars",
                str(args.pool_text_max_chars),
                "--query-text-max-chars",
                str(args.query_text_max_chars),
                "--similarity-query-prompt-name",
                args.similarity_query_prompt_name,
                "--similarity-query-prompt",
                args.similarity_query_prompt,
                "--pool-emb-cache",
                str(args.pool_emb_cache),
                "--embedding-dtype",
                args.embedding_dtype,
            ]
        )
    return cmd


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    eval_config = load_eval_config(args.config)
    timestamp = args.timestamp.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

    report_rows: list[dict[str, Any]] = []
    pipeline_meta: dict[str, Any] = {
        "run_prefix": args.run_prefix,
        "timestamp": timestamp,
        "methods": methods,
        "k": args.k,
        "config": str(args.config),
        "query_limit": args.query_limit,
        "pool_limit": args.pool_limit,
        "dry_run_infer": args.dry_run_infer,
        "skip_infer": args.skip_infer,
        "skip_eval": args.skip_eval,
        "runs": [],
    }

    for method in methods:
        cfg_k = method_k_from_config(eval_config, method)
        method_k = cfg_k if cfg_k is not None else (0 if method == "zero" else args.k)
        map_name = f"humaneval_{method}_k{method_k}.jsonl"
        mapping_path = args.mapping_dir / map_name
        meta_path = args.mapping_dir / f"{map_name}.meta.json"
        prompt_path = args.prompts_dir / f"{method}_humaneval_k{method_k}.jsonl"

        run_name = f"{args.run_prefix}_{method}_k{method_k}_{timestamp}"
        infer_dir = args.outputs_dir / run_name
        infer_path = infer_dir / "outputs.jsonl"
        eval_run_name = f"{run_name}_eval"
        eval_dir = args.metrics_dir / eval_run_name
        eval_summary_path = eval_dir / "summary.json"

        run_record: dict[str, Any] = {
            "method": method,
            "k": method_k,
            "mapping_path": str(mapping_path),
            "mapping_meta_path": str(meta_path),
            "prompt_path": str(prompt_path),
            "inference_path": str(infer_path),
            "eval_dir": str(eval_dir),
        }

        run_cmd(build_common_map_cmd(args, method, method_k, mapping_path, meta_path))

        run_cmd(
            [
                sys.executable,
                "utils/build_icl_prompts.py",
                "--pool",
                str(args.pool),
                "--queries",
                str(args.queries),
                "--mapping",
                str(mapping_path),
                "--config",
                str(args.config),
                "--k",
                str(method_k),
                "--out",
                str(prompt_path),
                "--method-from-mapping",
            ]
        )

        if not args.skip_infer:
            infer_cmd = [
                sys.executable,
                "utils/run_api_infer.py",
                "--prompts",
                str(prompt_path),
                "--out",
                str(infer_path),
                "--selector-meta",
                str(meta_path),
                "--not-mig-algorithm",
                "--difficulty-label-scheme",
                "n/a for baseline",
                "--model",
                args.model,
                "--base-url",
                args.base_url,
                "--api-key-env",
                args.api_key_env,
                "--env-file",
                str(args.env_file),
                "--temperature",
                str(args.temperature),
                "--top-p",
                str(args.top_p),
                "--timeout-s",
                str(args.timeout_s),
                "--retry",
                str(args.retry),
                "--retry-backoff-s",
                str(args.retry_backoff_s),
                "--parallelism",
                str(args.parallelism),
                "--rate-limit-qps",
                str(args.rate_limit_qps),
                "--bucket-capacity",
                str(args.bucket_capacity),
                "--config",
                str(args.config),
            ]
            if args.max_tokens > 0:
                infer_cmd.extend(["--max-tokens", str(args.max_tokens)])
            if args.save_raw_response:
                infer_cmd.append("--save-raw-response")
            if args.dry_run_infer:
                infer_cmd.append("--dry-run")
            run_cmd(infer_cmd)
        else:
            print(f"[SKIP] inference for {method}")

        if not args.skip_eval:
            eval_cmd = [
                sys.executable,
                "utils/eval_humaneval.py",
                "--inference",
                str(infer_path),
                "--prompts",
                str(prompt_path),
                "--selector-meta",
                str(meta_path),
                "--not-mig-algorithm",
                "--difficulty-label-scheme",
                "n/a for baseline",
                "--run-name",
                eval_run_name,
                "--k-list",
                args.eval_k_list,
                "--timeout-s",
                str(args.eval_timeout_s),
                "--max-samples-per-query",
                str(args.eval_max_samples_per_query),
            ]
            run_cmd(eval_cmd)
        else:
            print(f"[SKIP] eval for {method}")

        summary = load_json(eval_summary_path)
        metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
        report_row = {
            "method": method,
            "k": method_k,
            "run_name": eval_run_name,
            "pass@1_topk_strict": metrics.get("pass@1_topk_strict"),
            "pass@5_topk_strict": metrics.get("pass@5_topk_strict"),
            "avg_prompt_tokens": summary.get("avg_prompt_tokens"),
            "avg_completion_tokens": summary.get("avg_completion_tokens"),
            "avg_total_tokens": summary.get("avg_total_tokens"),
            "avg_infer_latency_s": summary.get("avg_infer_latency_s"),
            "sample_pass_rate": summary.get("sample_pass_rate"),
            "sample_total": summary.get("sample_total"),
            "sample_passed": summary.get("sample_passed"),
        }
        report_rows.append(report_row)
        run_record["summary"] = report_row
        pipeline_meta["runs"].append(run_record)

    report_base = args.report_dir / f"{args.run_prefix}_{timestamp}_baseline_compare"
    write_json(report_base.with_suffix(".json"), {"rows": report_rows, "meta": pipeline_meta})
    write_csv(report_base.with_suffix(".csv"), report_rows)
    write_markdown(report_base.with_suffix(".md"), report_rows)

    print("\nPipeline finished.")
    print(f"Report JSON: {report_base.with_suffix('.json')}")
    print(f"Report CSV : {report_base.with_suffix('.csv')}")
    print(f"Report MD  : {report_base.with_suffix('.md')}")


if __name__ == "__main__":
    main()
