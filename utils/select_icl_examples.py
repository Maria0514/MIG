import argparse
import json
from pathlib import Path
from typing import Any

from mig.data import DataPoint, load_dataset
from mig.icl_selector import QueryAwareICLSelector
from mig.label import LabelGraphType, create_label_graph


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


def resolve_demo_id(dp: DataPoint, pool_index: int) -> str:
    raw = dp.raw if isinstance(dp.raw, dict) else {}
    for key in ("demo_id", "id", "_id"):
        value = raw.get(key)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    return f"pool_{pool_index}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run query-aware ICL greedy selection and export query->demo mappings.")
    parser.add_argument("--pool", type=Path, default=Path("data/icl/openhermes_python_related.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/icl/humaneval_eval_tagged.jsonl"))
    parser.add_argument("--graph-pkl", type=Path, default=Path("outputs/label_graph_openhermes_python_t08.pkl"))
    parser.add_argument("--valid-tag-path", type=Path, default=Path("configs/valid_tag_path_openhermes_python.json"))
    parser.add_argument("--out", type=Path, default=Path("data/icl/mappings/humaneval_to_demos.jsonl"))

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--tau-q", type=float, default=0.8)

    parser.add_argument("--phi-type", choices=("pow", "exp"), default="pow")
    parser.add_argument("--phi-alpha", type=float, default=1.0)
    parser.add_argument("--phi-a", type=float, default=1e-6)
    parser.add_argument("--phi-b", type=float, default=0.8)

    parser.add_argument("--lambda-quality", type=float, default=0.1)
    parser.add_argument("--lambda-len", type=float, default=0.05)
    parser.add_argument("--lambda-red", type=float, default=0.1)
    parser.add_argument("--lambda-diff", type=float, default=0.0)
    parser.add_argument("--prop-weight", type=float, default=1.0)

    parser.add_argument("--use-difficulty-penalty", action="store_true")
    parser.add_argument("--difficulty-penalty-weight", type=float, default=0.0)

    parser.add_argument("--query-labels-key", type=str, default="query_labels")
    parser.add_argument("--query-id-key", type=str, default="id")
    parser.add_argument("--query-limit", type=int, default=0, help="0 means all queries")
    parser.add_argument("--with-debug-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pool.exists():
        raise FileNotFoundError(f"Pool not found: {args.pool}")
    if not args.queries.exists():
        raise FileNotFoundError(f"Queries not found: {args.queries}")
    if not args.graph_pkl.exists():
        raise FileNotFoundError(f"Graph pkl not found: {args.graph_pkl}")
    if not args.valid_tag_path.exists():
        raise FileNotFoundError(f"valid_tag_path not found: {args.valid_tag_path}")

    print(f"Loading pool: {args.pool}")
    pool = load_dataset(args.pool, valid_tag_path=str(args.valid_tag_path))
    print(f"Pool size: {len(pool)}")

    print(f"Loading label graph: {args.graph_pkl}")
    label_graph = create_label_graph(LabelGraphType.SIM, load_from=str(args.graph_pkl), sim_threshold=args.tau_q)

    print(f"Loading queries: {args.queries}")
    queries = read_jsonl(args.queries)
    if args.query_limit and args.query_limit > 0:
        queries = queries[: args.query_limit]
    print(f"Query size: {len(queries)}")

    selector = QueryAwareICLSelector(
        pool=pool,
        label_graph=label_graph,
        prop_weight=args.prop_weight,
        phi_type=args.phi_type,
        phi_alpha=args.phi_alpha,
        phi_a=args.phi_a,
        phi_b=args.phi_b,
        use_difficulty_penalty=args.use_difficulty_penalty,
        difficulty_penalty_weight=args.difficulty_penalty_weight,
    )

    out_rows: list[dict[str, Any]] = []
    for q in queries:
        qid = str(q.get(args.query_id_key, ""))
        q_labels = q.get(args.query_labels_key) or []
        if not isinstance(q_labels, list):
            q_labels = []

        indices, steps = selector.select_k_for_query(
            query_labels=q_labels,
            k=args.k,
            tau_q=args.tau_q,
            lambda_quality=args.lambda_quality,
            lambda_len=args.lambda_len,
            lambda_red=args.lambda_red,
            lambda_diff=args.lambda_diff,
        )

        demo_ids = [resolve_demo_id(pool[i], i) for i in indices]
        row: dict[str, Any] = {
            "query_id": qid,
            "ordered_demo_ids": demo_ids,
            "ordered_pool_indices": indices,
        }

        if args.with_debug_scores:
            row["debug_scores"] = [
                {
                    "pool_index": s.pool_index,
                    "score": s.score,
                    "gain": s.gain,
                    "quality": s.quality,
                    "len_penalty": s.len_penalty,
                    "red_penalty": s.red_penalty,
                }
                for s in steps
            ]

        out_rows.append(row)

    write_jsonl(args.out, out_rows)
    print(f"Output rows: {len(out_rows)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
