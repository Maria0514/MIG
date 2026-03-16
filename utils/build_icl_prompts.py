import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a Python coding assistant. Complete the target function correctly. "
    "Output only Python code with no markdown fences and no explanation."
)
DEFAULT_TARGET_INSTRUCTION = "Please provide the completion for the target function only."


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


def resolve_demo_id(raw: dict[str, Any], pool_index: int) -> str:
    for key in ("demo_id", "id", "_id"):
        value = raw.get(key)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    return f"pool_{pool_index}"


def extract_role_content(raw: dict[str, Any], role: str) -> str:
    dialogs = raw.get("dialogs")
    if isinstance(dialogs, list):
        for turn in dialogs:
            if not isinstance(turn, dict):
                continue
            if str(turn.get("role", "")).strip().lower() != role:
                continue
            content = turn.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def extract_demo_problem(raw: dict[str, Any]) -> str:
    text = extract_role_content(raw, "user")
    if text:
        return text
    for key in ("prompt", "instruction", "question"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_demo_solution(raw: dict[str, Any]) -> str:
    text = extract_role_content(raw, "assistant")
    if text:
        return text
    for key in ("canonical_solution", "solution", "answer", "output"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


_FENCE_RE = re.compile(r"^\s*```[^\n`]*\n?(.*?)\n?```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    src = (text or "").strip()
    if not src:
        return ""
    m = _FENCE_RE.match(src)
    if m:
        return m.group(1).strip()
    return src


def build_user_prompt(
    demos: list[dict[str, Any]],
    query_prompt: str,
    target_instruction: str,
) -> str:
    blocks: list[str] = []
    for i, demo in enumerate(demos, start=1):
        blocks.append(
            f"[Example {i}]\n"
            f"Problem:\n{demo['problem']}\n\n"
            f"Reference Solution:\n{demo['solution']}"
        )

    blocks.append(
        f"[Target Problem]\n{query_prompt}\n\n"
        f"{target_instruction}"
    )
    return "\n\n".join(blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build evaluation prompts from query->demo mappings."
    )
    parser.add_argument("--pool", type=Path, default=Path("data/icl/openhermes_python_related.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/icl/humaneval_eval_tagged.jsonl"))
    parser.add_argument("--mapping", type=Path, default=Path("data/icl/mappings/humaneval_to_demos.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/icl/eval_prompts/mig_humaneval_k8.jsonl"))

    parser.add_argument("--query-id-key", type=str, default="id")
    parser.add_argument("--mapping-query-id-key", type=str, default="query_id")
    parser.add_argument("--k", type=int, default=8, help="Number of demos per query. <=0 means use all mapped demos.")
    parser.add_argument("--query-limit", type=int, default=0, help="Only export first N mapping rows. 0 means all.")

    parser.add_argument("--method", type=str, default="mig")
    parser.add_argument(
        "--method-from-mapping",
        dest="method_from_mapping",
        action="store_true",
        help="Use per-row method from mapping when provided.",
    )
    parser.add_argument(
        "--ignore-mapping-method",
        dest="method_from_mapping",
        action="store_false",
        help="Ignore mapping method field and always use --method.",
    )
    parser.set_defaults(method_from_mapping=True)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--target-instruction", type=str, default=DEFAULT_TARGET_INSTRUCTION)

    parser.add_argument("--strip-code-fence", dest="strip_code_fence", action="store_true")
    parser.add_argument("--keep-code-fence", dest="strip_code_fence", action="store_false")
    parser.set_defaults(strip_code_fence=True)

    parser.add_argument("--skip-missing-query", action="store_true")
    parser.add_argument("--strict-demo-match", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for p in (args.pool, args.queries, args.mapping):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    pool = read_jsonl(args.pool)
    queries = read_jsonl(args.queries)
    mappings = read_jsonl(args.mapping)

    query_map: dict[str, dict[str, Any]] = {}
    for q in queries:
        qid = str(q.get(args.query_id_key, "")).strip()
        if qid:
            query_map[qid] = q

    pool_id_to_index: dict[str, int] = {}
    resolved_pool_ids: list[str] = []
    for idx, raw in enumerate(pool):
        did = resolve_demo_id(raw, idx)
        resolved_pool_ids.append(did)
        # keep first occurrence to avoid unstable remap on duplicated ids
        if did not in pool_id_to_index:
            pool_id_to_index[did] = idx

    if args.query_limit > 0:
        mappings = mappings[: args.query_limit]

    out_rows: list[dict[str, Any]] = []
    missing_query_count = 0
    missing_demo_count = 0

    for m in mappings:
        qid = str(m.get(args.mapping_query_id_key, "")).strip()
        if not qid:
            continue

        query = query_map.get(qid)
        if query is None:
            missing_query_count += 1
            if args.skip_missing_query:
                continue
            raise KeyError(f"Query id from mapping not found in queries: {qid}")

        mapped_indices: list[int] = []
        raw_indices = m.get("ordered_pool_indices")
        raw_demo_ids = m.get("ordered_demo_ids")

        if isinstance(raw_indices, list):
            for x in raw_indices:
                if isinstance(x, int) and 0 <= x < len(pool):
                    mapped_indices.append(x)
                else:
                    missing_demo_count += 1
                    if args.strict_demo_match:
                        raise IndexError(f"Invalid pool index for query {qid}: {x}")

        if not mapped_indices and isinstance(raw_demo_ids, list):
            for did in raw_demo_ids:
                idx = pool_id_to_index.get(str(did))
                if idx is None:
                    missing_demo_count += 1
                    if args.strict_demo_match:
                        raise KeyError(f"Demo id not found in pool for query {qid}: {did}")
                    continue
                mapped_indices.append(idx)

        if args.k > 0:
            mapped_indices = mapped_indices[: args.k]

        demos: list[dict[str, Any]] = []
        for idx in mapped_indices:
            raw_demo = pool[idx]
            problem = extract_demo_problem(raw_demo)
            solution = extract_demo_solution(raw_demo)
            if args.strip_code_fence:
                solution = strip_code_fence(solution)
            demos.append(
                {
                    "pool_index": idx,
                    "demo_id": resolved_pool_ids[idx],
                    "problem": problem,
                    "solution": solution,
                }
            )

        query_prompt = str(query.get("prompt", "") or "")
        user_prompt = build_user_prompt(
            demos=demos,
            query_prompt=query_prompt,
            target_instruction=args.target_instruction,
        )
        row_method = args.method
        if args.method_from_mapping:
            m_method = str(m.get("method", "")).strip()
            if m_method:
                row_method = m_method
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        out_rows.append(
            {
                "method": row_method,
                "query_id": qid,
                "query_labels": query.get("query_labels", []),
                "entry_point": query.get("entry_point"),
                "tests": query.get("tests", {}),
                "query_prompt": query_prompt,
                "demo_ids": [d["demo_id"] for d in demos],
                "demo_pool_indices": [d["pool_index"] for d in demos],
                "demos": demos,
                "system_prompt": args.system_prompt,
                "user_prompt": user_prompt,
                "messages": messages,
                "prompt_text": f"SYSTEM:\n{args.system_prompt}\n\nUSER:\n{user_prompt}",
            }
        )

    write_jsonl(args.out, out_rows)
    print(f"Queries loaded: {len(queries)}")
    print(f"Mappings loaded: {len(mappings)}")
    print(f"Prompt rows written: {len(out_rows)}")
    print(f"Missing queries: {missing_query_count}")
    print(f"Missing demos: {missing_demo_count}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
