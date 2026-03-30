import json
from collections import defaultdict
from pathlib import Path


RUN_DIRS = [
    "v32_zero_k0_full_eval",
    "v32_random_k5_full_eval",
    "v32_sim_k5_full_eval",
    "v32_mig_k5_full_eval",
]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_run_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def index_inference_rows(
    inference_path: Path,
    *,
    status_ok_only: bool,
    max_samples_per_query: int,
) -> dict[str, list[dict]]:
    rows = read_jsonl(inference_path)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        qid = str(row.get("query_id", "")).strip()
        if not qid:
            continue
        if status_ok_only and str(row.get("status", "")).lower() != "ok":
            continue
        grouped[qid].append(row)

    if max_samples_per_query > 0:
        for qid in list(grouped.keys()):
            grouped[qid] = grouped[qid][:max_samples_per_query]
    return grouped


def main() -> None:
    base_dir = Path("data/icl/eval_metrics")
    total_failed = 0
    touched = 0

    for run_name in RUN_DIRS:
        run_dir = base_dir / run_name
        per_sample_path = run_dir / "per_sample.jsonl"
        out_path = run_dir / "failed_cases.jsonl"
        summary = load_run_summary(run_dir)

        if not per_sample_path.exists():
            print(f"[SKIP] missing per_sample: {per_sample_path}")
            continue

        inference_rel = str(summary.get("inference_file", "")).strip()
        if not inference_rel:
            print(f"[SKIP] missing inference_file in summary: {run_dir / 'summary.json'}")
            continue

        inference_path = Path(inference_rel)
        if not inference_path.exists():
            print(f"[SKIP] inference file not found: {inference_path}")
            continue

        status_ok_only = bool(summary.get("status_ok_only", True))
        max_samples_per_query = int(summary.get("max_samples_per_query", 0) or 0)
        infer_map = index_inference_rows(
            inference_path,
            status_ok_only=status_ok_only,
            max_samples_per_query=max_samples_per_query,
        )

        rows = read_jsonl(per_sample_path)
        failed_rows: list[dict] = []
        for row in rows:
            if bool(row.get("passed", False)):
                continue

            qid = str(row.get("query_id", "")).strip()
            rank = int(row.get("sample_rank", -1))
            infer_row = {}
            bucket = infer_map.get(qid, [])
            if 0 <= rank < len(bucket):
                infer_row = bucket[rank]

            enriched = dict(row)
            enriched["generated_code"] = (
                str(infer_row.get("completion_cleaned", "")).strip()
                or str(infer_row.get("raw_output", "")).strip()
            )
            enriched["completion_cleaned"] = infer_row.get("completion_cleaned", "")
            enriched["raw_output"] = infer_row.get("raw_output", "")
            enriched["infer_status"] = infer_row.get("status", "")
            enriched["response_id"] = infer_row.get("response_id", "")
            enriched["latency_s"] = infer_row.get("latency_s")
            failed_rows.append(enriched)

        write_jsonl(out_path, failed_rows)

        touched += 1
        total_failed += len(failed_rows)
        print(
            f"[OK] {run_name}: failed={len(failed_rows)} / total={len(rows)} -> {out_path}"
        )

    print(f"done. runs={touched}, total_failed={total_failed}")


if __name__ == "__main__":
    main()
