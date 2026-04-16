import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mig.icl_config import detect_prompt_task, load_eval_config, resolve_generation_profile


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_api_key(env_name: str) -> str:
    key = os.getenv(env_name, "").strip()
    if key:
        return key
    raise RuntimeError(f"Environment variable '{env_name}' is empty or missing.")


def load_env_file(path: Path) -> int:
    if not path.exists():
        return 0
    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return obj
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


def sanitize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content)


_FENCE_RE = re.compile(r"^\s*```[^\n`]*\n?(.*?)\n?```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    src = "" if text is None else str(text)
    if not src.strip():
        return ""
    probe = src.strip()
    m = _FENCE_RE.match(probe)
    if m:
        # Keep leading indentation in code body; only trim trailing line breaks.
        return m.group(1).rstrip("\r\n")
    if probe.startswith("```"):
        first_newline = probe.find("\n")
        if first_newline >= 0:
            return probe[first_newline + 1 :].lstrip("\r\n")
        return ""
    return src.rstrip("\r\n")


def resolve_row_generation_config(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    eval_config: dict[str, Any],
) -> tuple[str, dict[str, Any], int]:
    prompt_task = str(row.get("prompt_task", "")).strip()
    if not prompt_task:
        prompt_task = detect_prompt_task(row)

    _, resolved_profile = resolve_generation_profile(eval_config, prompt_task)
    row_profile = row.get("generation_config")
    if isinstance(row_profile, dict):
        max_tokens = row_profile.get("max_tokens")
        if isinstance(max_tokens, float):
            max_tokens = int(max_tokens)
        if isinstance(max_tokens, int) and max_tokens > 0:
            resolved_profile["max_tokens"] = max_tokens

    if args.max_tokens > 0:
        resolved_profile["max_tokens"] = args.max_tokens

    max_tokens = resolved_profile.get("max_tokens", 0)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = 0

    return str(prompt_task), resolved_profile, max_tokens


def post_chat_completions(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


class TokenBucket:
    """Thread-safe token bucket for request rate limiting."""

    def __init__(self, rate: float, capacity: float):
        self.rate = max(float(rate), 0.0)
        self.capacity = max(float(capacity), 1.0)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        need = max(float(tokens), 0.0)
        if need <= 0:
            return
        if self.rate <= 0:
            return

        while True:
            wait_s = 0.0
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.updated
                self.updated = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                if self.tokens >= need:
                    self.tokens -= need
                    return
                wait_s = (need - self.tokens) / self.rate
            time.sleep(max(wait_s, 0.01))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run API inference from prompt JSONL using SiliconFlow-compatible chat completions with parallel workers and token-bucket rate limit."
    )
    parser.add_argument("--prompts", type=Path, default=Path("data/icl/eval_prompts/mig_humaneval_k5.jsonl"))
    parser.add_argument("--out", type=Path, default=None, help="Optional explicit output JSONL path.")
    parser.add_argument("--base-out-dir", type=Path, default=Path("data/icl/eval_outputs"))
    parser.add_argument("--run-name", type=str, default="", help="Optional run name. If empty, auto-generated.")
    parser.add_argument("--out-file-name", type=str, default="outputs.jsonl")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/icl_eval_config.json"),
        help="Optional experiment config for task-aware generation defaults.",
    )

    parser.add_argument("--model", type=str, default="nvidia/openai/gpt-oss-120b")
    parser.add_argument("--base-url", type=str, default="https://inference-api.nvidia.com")
    parser.add_argument("--api-key-env", type=str, default="SILICONFLOW_API_KEY")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))

    parser.add_argument(
        "--selector-meta",
        type=Path,
        default=Path("data/icl/metadata/humaneval_to_demos.meta.json"),
        help="Optional selector metadata file for experiment context.",
    )
    parser.add_argument("--is-mig-algorithm", dest="is_mig_algorithm", action="store_true")
    parser.add_argument("--not-mig-algorithm", dest="is_mig_algorithm", action="store_false")
    parser.set_defaults(is_mig_algorithm=True)
    parser.add_argument(
        "--difficulty-label-scheme",
        type=str,
        default="easy/medium/hard from deita complexity quantiles",
    )

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
    parser.add_argument("--sleep-after-s", type=float, default=0.0)

    parser.add_argument("--parallelism", type=int, default=1, help="Concurrent worker count.")
    parser.add_argument("--rate-limit-qps", type=float, default=0.3, help="Token bucket refill rate (requests/sec).")
    parser.add_argument(
        "--bucket-capacity",
        type=float,
        default=1.0,
        help="Token bucket burst capacity (tokens).",
    )

    parser.add_argument("--query-limit", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--resume", action="store_true", help="Skip query_id already in output JSONL.")
    parser.add_argument("--save-raw-response", action="store_true")
    parser.add_argument("--strip-code-fence", dest="strip_code_fence", action="store_true")
    parser.add_argument("--keep-code-fence", dest="strip_code_fence", action="store_false")
    parser.set_defaults(strip_code_fence=True)
    parser.add_argument("--dry-run", action="store_true", help="Do not call API, only emit placeholder outputs.")
    return parser.parse_args()


def infer_one(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    eval_config: dict[str, Any],
    api_key: str,
    bucket: TokenBucket | None,
) -> tuple[dict[str, Any], str]:
    query_id = str(row.get("query_id", "")).strip()
    method = row.get("method", "")
    request_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return (
            {
                "query_id": query_id,
                "method": method,
                "model": args.model,
                "status": "error",
                "error_type": "invalid_prompt",
                "error_message": "messages field is missing or invalid",
            },
            "error",
        )

    prompt_task, generation_config, resolved_max_tokens = resolve_row_generation_config(
        row,
        args=args,
        eval_config=eval_config,
    )

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if resolved_max_tokens > 0:
        payload["max_tokens"] = resolved_max_tokens

    t0 = time.perf_counter()
    base_info = {
        "query_id": query_id,
        "method": method,
        "prompt_task": prompt_task,
        "model": args.model,
        "request_time_utc": request_ts,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": resolved_max_tokens,
        "generation_config": generation_config,
        "prompt_text": row.get("prompt_text", ""),
        "messages": messages,
        "demo_ids": row.get("demo_ids", []),
        "demo_pool_indices": row.get("demo_pool_indices", []),
        "entry_point": row.get("entry_point"),
        "query_prompt": row.get("query_prompt", ""),
        "tests": row.get("tests", {}),
    }

    if args.dry_run:
        completion_raw = "# DRY_RUN\npass\n"
        out_row = {
            **base_info,
            "latency_s": round(time.perf_counter() - t0, 4),
            "status": "ok",
            "raw_output": completion_raw,
            "completion_cleaned": strip_code_fence(completion_raw) if args.strip_code_fence else completion_raw,
            "usage": None,
            "finish_reason": "dry_run",
        }
        return out_row, "ok"

    last_error_type = ""
    last_error_msg = ""
    response_json: dict[str, Any] | None = None

    for attempt in range(args.retry + 1):
        try:
            if bucket is not None:
                bucket.acquire(1.0)
            response_json = post_chat_completions(
                base_url=args.base_url,
                api_key=api_key,
                payload=payload,
                timeout_s=args.timeout_s,
            )
            break
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_error_type = f"http_{e.code}"
            last_error_msg = err_body or str(e)
        except urllib.error.URLError as e:
            last_error_type = "url_error"
            last_error_msg = str(e)
        except Exception as e:  # noqa: BLE001
            last_error_type = "exception"
            last_error_msg = str(e)

        if attempt < args.retry:
            time.sleep(args.retry_backoff_s * (2 ** attempt))

    latency_s = round(time.perf_counter() - t0, 4)

    if response_json is None:
        out_row = {
            **base_info,
            "latency_s": latency_s,
            "status": "error",
            "error_type": last_error_type,
            "error_message": last_error_msg,
        }
        return out_row, "error"

    choices = response_json.get("choices") or []
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    completion_raw = sanitize_content(message.get("content", ""))
    completion_cleaned = strip_code_fence(completion_raw) if args.strip_code_fence else completion_raw

    out_row: dict[str, Any] = {
        **base_info,
        "latency_s": latency_s,
        "status": "ok",
        "response_id": response_json.get("id"),
        "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
        "usage": response_json.get("usage"),
        "raw_output": completion_raw,
        "completion_cleaned": completion_cleaned,
    }
    if args.save_raw_response:
        out_row["raw_response"] = response_json

    if args.sleep_after_s > 0:
        time.sleep(args.sleep_after_s)
    return out_row, "ok"


def main() -> None:
    args = parse_args()
    if not args.prompts.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompts}")

    eval_config = load_eval_config(args.config)
    prompt_rows = read_jsonl(args.prompts)
    if args.query_limit > 0:
        prompt_rows = prompt_rows[: args.query_limit]

    if args.out is not None:
        out_path = args.out
        run_dir = out_path.parent
        run_name = out_path.stem
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_name = args.run_name.strip() or f"{args.model.replace('/', '_')}_{ts}"
        run_dir = build_unique_run_dir(args.base_out_dir, auto_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / args.out_file_name
        run_name = run_dir.name

    selector_meta = load_json(args.selector_meta)
    params_obj = selector_meta.get("params")
    if isinstance(params_obj, dict):
        selector_params = params_obj
    elif isinstance(selector_meta, dict):
        # Compatibility: baseline meta may store fields at top-level.
        selector_params = selector_meta
    else:
        selector_params = {}
    methods_seen = sorted(
        {
            str(r.get("method", "")).strip()
            for r in prompt_rows
            if str(r.get("method", "")).strip()
        }
    )

    done: set[str] = set()
    if args.resume and out_path.exists():
        for row in read_jsonl(out_path):
            qid = str(row.get("query_id", "")).strip()
            if qid:
                done.add(qid)

    loaded_count = load_env_file(args.env_file)
    if loaded_count > 0:
        print(f"Loaded {loaded_count} env var(s) from {args.env_file}")

    api_key = "" if args.dry_run else resolve_api_key(args.api_key_env)
    started_at = datetime.now().isoformat(timespec="seconds")

    total_input_rows = len(prompt_rows)
    rows_to_process: list[dict[str, Any]] = []
    skip_count = 0
    for row in prompt_rows:
        qid = str(row.get("query_id", "")).strip()
        if args.resume and qid and qid in done:
            skip_count += 1
            continue
        rows_to_process.append(row)

    ok_count = 0
    err_count = 0
    total_to_process = len(rows_to_process)

    bucket: TokenBucket | None = None
    if not args.dry_run and args.rate_limit_qps > 0:
        bucket = TokenBucket(rate=args.rate_limit_qps, capacity=args.bucket_capacity)

    workers = max(1, args.parallelism)

    if workers == 1:
        for idx, row in enumerate(rows_to_process, start=1):
            out_row, status = infer_one(row, args=args, eval_config=eval_config, api_key=api_key, bucket=bucket)
            append_jsonl(out_path, out_row)
            if status == "ok":
                ok_count += 1
            else:
                err_count += 1
            if idx % 10 == 0 or idx == total_to_process:
                print(
                    f"Progress {idx}/{total_to_process} | ok={ok_count} error={err_count} "
                    f"skip={skip_count} out={out_path}"
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(infer_one, row, args=args, eval_config=eval_config, api_key=api_key, bucket=bucket)
                for row in rows_to_process
            ]
            processed = 0
            for fut in as_completed(futures):
                processed += 1
                try:
                    out_row, status = fut.result()
                except Exception as e:  # noqa: BLE001
                    status = "error"
                    out_row = {
                        "query_id": "",
                        "method": "",
                        "model": args.model,
                        "status": "error",
                        "error_type": "worker_exception",
                        "error_message": str(e),
                    }
                append_jsonl(out_path, out_row)
                if status == "ok":
                    ok_count += 1
                else:
                    err_count += 1
                if processed % 10 == 0 or processed == total_to_process:
                    print(
                        f"Progress {processed}/{total_to_process} | ok={ok_count} error={err_count} "
                        f"skip={skip_count} out={out_path}"
                    )

    finished_at = datetime.now().isoformat(timespec="seconds")
    manifest = {
        "run_name": run_name,
        "created_at_local": started_at,
        "finished_at_local": finished_at,
        "outputs_file": str(out_path),
        "prompts_file": str(args.prompts),
        "config_file": str(args.config),
        "selector_meta_file": str(args.selector_meta) if args.selector_meta.exists() else "",
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "query_limit": args.query_limit,
        "resume": args.resume,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens_override": args.max_tokens,
        "timeout_s": args.timeout_s,
        "retry": args.retry,
        "retry_backoff_s": args.retry_backoff_s,
        "sleep_after_s": args.sleep_after_s,
        "parallelism": workers,
        "rate_limit_qps": args.rate_limit_qps,
        "bucket_capacity": args.bucket_capacity,
        "strip_code_fence": args.strip_code_fence,
        "save_raw_response": args.save_raw_response,
        "dry_run": args.dry_run,
        "stats": {
            "total_input_rows": total_input_rows,
            "total_processed_rows": total_to_process,
            "written_ok": ok_count,
            "written_error": err_count,
            "skipped_resume": skip_count,
        },
        "experiment_context": {
            "is_mig_algorithm": bool(args.is_mig_algorithm),
            "methods_seen": methods_seen,
            "difficulty_label_scheme": args.difficulty_label_scheme,
            "penalty_config": {
                "lambda_quality": selector_params.get("lambda_quality"),
                "lambda_len": selector_params.get("lambda_len"),
                "lambda_red": selector_params.get("lambda_red"),
                "lambda_diff": selector_params.get("lambda_diff"),
                "length_penalty_enabled": bool(
                    isinstance(selector_params.get("lambda_len"), (int, float))
                    and float(selector_params.get("lambda_len")) != 0.0
                ),
                "redundancy_penalty_enabled": bool(
                    isinstance(selector_params.get("lambda_red"), (int, float))
                    and float(selector_params.get("lambda_red")) != 0.0
                ),
                "difficulty_penalty_enabled": bool(selector_params.get("use_difficulty_penalty"))
                or bool(
                    isinstance(selector_params.get("lambda_diff"), (int, float))
                    and float(selector_params.get("lambda_diff")) != 0.0
                ),
            },
            "selector_params_raw": selector_params,
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)

    print("Done.")
    print(f"Total input rows: {total_input_rows}")
    print(f"Total processed rows: {total_to_process}")
    print(f"Written ok: {ok_count}")
    print(f"Written error: {err_count}")
    print(f"Skipped (resume): {skip_count}")
    print(f"Output: {out_path}")
    print(f"Manifest: {run_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
