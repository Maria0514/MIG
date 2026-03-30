import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np


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


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def format_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return obj
    return {}


def read_jsonl_ids(path: Path) -> list[str]:
    ids: list[str] = []
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            value = row.get("id")
            if isinstance(value, (str, int)) and str(value):
                ids.append(str(value))
    return ids


def load_eval_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return obj
    return {}


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


def extract_query_text(query: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = query.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build baseline query->demo mappings for Zero-shot / Random-ICL / Similarity-ICL."
    )
    parser.add_argument("--pool", type=Path, default=Path("data/icl/openhermes_python_related.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/icl/humaneval_eval_tagged.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/icl/mappings/humaneval_random_k5.jsonl"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/icl_eval_config.json"),
        help="Optional experiment config. Used to resolve per-method k when --k is omitted.",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="Optional metadata path. Default: <out>.meta.json",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=("zero", "random", "similarity", "sim"),
        required=True,
    )
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--query-id-key", type=str, default="id")
    parser.add_argument("--query-text-keys", type=str, default="prompt")
    parser.add_argument("--query-limit", type=int, default=0, help="0 means all queries")
    parser.add_argument("--pool-limit", type=int, default=0, help="0 means all pool samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pool-text-max-chars",
        type=int,
        default=1200,
        help="Max chars per pool sample text for similarity embedding. <=0 means no truncation.",
    )
    parser.add_argument(
        "--query-text-max-chars",
        type=int,
        default=1200,
        help="Max chars per query text for similarity embedding. <=0 means no truncation.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="models/e5-mistral-7b-instruct",
        help="Used only for similarity baseline.",
    )
    parser.add_argument("--embedding-device", type=str, default="cpu", help="Used only for similarity baseline.")
    parser.add_argument("--embedding-batch-size", type=int, default=64, help="Used only for similarity baseline.")
    parser.add_argument(
        "--embedding-max-seq-length",
        type=int,
        default=512,
        help="Used only for similarity baseline. <=0 means keep model default.",
    )
    parser.add_argument(
        "--similarity-query-prompt-name",
        type=str,
        default="sts_query",
        help="SentenceTransformer prompt_name for query encoding. Empty means no prompt_name.",
    )
    parser.add_argument(
        "--similarity-query-prompt",
        type=str,
        default="",
        help="SentenceTransformer prompt text for query encoding. Overrides --similarity-query-prompt-name when set.",
    )
    parser.add_argument(
        "--pool-emb-cache",
        type=Path,
        default=None,
        help="Optional .npy cache for pool embeddings (similarity baseline).",
    )
    parser.add_argument(
        "--query-emb-cache",
        type=Path,
        default=None,
        help="Optional .npy cache for query embeddings (similarity baseline).",
    )
    parser.add_argument(
        "--embedding-dtype",
        type=str,
        choices=("float32", "float16"),
        default="float16",
        help="Saved/loaded dtype for pool embedding matrix when using similarity baseline.",
    )
    return parser.parse_args()


def normalize_method_name(method: str) -> str:
    m = method.strip().lower()
    if m == "similarity":
        return "sim"
    return m


def build_zero_rows(
    queries: list[dict[str, Any]],
    query_id_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in queries:
        qid = str(q.get(query_id_key, "")).strip()
        if not qid:
            continue
        rows.append(
            {
                "method": "zero",
                "query_id": qid,
                "ordered_demo_ids": [],
                "ordered_pool_indices": [],
            }
        )
    return rows


def build_random_rows(
    queries: list[dict[str, Any]],
    query_id_key: str,
    pool_size: int,
    pool_ids: list[str],
    *,
    k: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    k_eff = min(max(k, 0), pool_size)
    for q in queries:
        qid = str(q.get(query_id_key, "")).strip()
        if not qid:
            continue
        rng = random.Random(f"{seed}:{qid}")
        selected = rng.sample(range(pool_size), k_eff) if k_eff > 0 else []
        rows.append(
            {
                "method": "random",
                "query_id": qid,
                "ordered_demo_ids": [pool_ids[i] for i in selected],
                "ordered_pool_indices": selected,
            }
        )
    return rows


def _encode_texts(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    prompt_name: str | None = None,
    prompt: str | None = None,
) -> np.ndarray:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": True,
    }
    if prompt:
        kwargs["prompt"] = prompt
    elif prompt_name:
        kwargs["prompt_name"] = prompt_name
    arr = model.encode(texts, **kwargs)
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    return arr


def load_or_build_pool_embeddings(
    *,
    model: Any | None,
    pool_texts: list[str],
    batch_size: int,
    cache_path: Path | None,
    dtype: str,
) -> np.ndarray:
    np_dtype = np.float16 if dtype == "float16" else np.float32
    if cache_path is not None and cache_path.exists():
        emb = np.load(cache_path)
        if emb.ndim != 2 or emb.shape[0] != len(pool_texts):
            raise ValueError(
                f"Invalid cache shape {emb.shape}; expected ({len(pool_texts)}, dim). Remove cache and retry."
            )
        if emb.dtype != np_dtype:
            emb = emb.astype(np_dtype, copy=False)
        return emb

    if model is None:
        raise ValueError("Pool embedding cache unavailable; model is required to build pool embeddings.")

    emb = _encode_texts(
        model=model,
        texts=pool_texts,
        batch_size=batch_size,
        prompt_name=None,
        prompt=None,
    )
    emb = emb.astype(np_dtype, copy=False)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
    return emb


def load_or_build_query_embeddings(
    *,
    model: Any | None,
    query_ids: list[str],
    query_texts: list[str],
    batch_size: int,
    prompt_name: str,
    prompt: str,
    cache_path: Path | None,
    dtype: str,
) -> np.ndarray:
    np_dtype = np.float16 if dtype == "float16" else np.float32
    meta_path = None if cache_path is None else cache_path.with_suffix(cache_path.suffix + ".meta.json")
    legacy_meta_path = None if cache_path is None else cache_path.with_suffix(".meta.json")
    legacy_ids_path = None
    if cache_path is not None:
        stem = cache_path.stem
        if "_emb_" in stem:
            legacy_ids_name = stem.split("_emb_", 1)[0] + "_ids.jsonl"
            legacy_ids_path = cache_path.with_name(legacy_ids_name)

    active_meta_path = None
    if meta_path is not None and meta_path.exists():
        active_meta_path = meta_path
    elif legacy_meta_path is not None and legacy_meta_path.exists():
        active_meta_path = legacy_meta_path

    if cache_path is not None and cache_path.exists() and active_meta_path is not None:
        emb = np.load(cache_path)
        meta = read_json(active_meta_path)
        cached_ids = meta.get("query_ids")
        if not isinstance(cached_ids, list) and legacy_ids_path is not None and legacy_ids_path.exists():
            cached_ids = read_jsonl_ids(legacy_ids_path)
        if not isinstance(cached_ids, list):
            raise ValueError(f"Invalid query cache meta: {active_meta_path}")
        if emb.ndim != 2 or emb.shape[0] != len(query_ids):
            raise ValueError(
                f"Invalid query cache shape {emb.shape}; expected ({len(query_ids)}, dim). Remove cache and retry."
            )
        if [str(x) for x in cached_ids] != query_ids:
            raise ValueError(
                "Query cache ids mismatch with current query set/order. "
                f"Cache: {cache_path}, Meta: {active_meta_path}. Remove cache and retry."
            )
        if emb.dtype != np_dtype:
            emb = emb.astype(np_dtype, copy=False)
        print(f"Loaded query embeddings from cache: {cache_path}")
        return emb

    if model is None:
        raise ValueError("Query embedding cache unavailable; model is required to build query embeddings.")

    emb = _encode_texts(
        model=model,
        texts=query_texts,
        batch_size=batch_size,
        prompt_name=prompt_name if prompt_name else None,
        prompt=prompt if prompt else None,
    )
    emb = emb.astype(np_dtype, copy=False)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        if meta_path is not None:
            write_json(
                meta_path,
                {
                    "query_ids": query_ids,
                    "shape": [int(x) for x in emb.shape],
                    "dtype": str(emb.dtype),
                    "prompt_name": prompt_name,
                    "prompt": prompt,
                },
            )
        print(f"Saved query embeddings cache: {cache_path}")
    return emb


def build_similarity_rows(
    queries: list[dict[str, Any]],
    query_id_key: str,
    query_text_keys: list[str],
    pool_ids: list[str],
    pool_embeddings: np.ndarray,
    *,
    model: Any | None,
    k: int,
    batch_size: int,
    query_prompt_name: str,
    query_prompt: str,
    query_text_max_chars: int,
    query_emb_cache: Path | None,
    embedding_dtype: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pool_size = len(pool_ids)
    k_eff = min(max(k, 0), pool_size)

    query_pairs: list[tuple[str, str]] = []
    for q in queries:
        qid = str(q.get(query_id_key, "")).strip()
        if not qid:
            continue
        query_pairs.append((qid, truncate_text(extract_query_text(q, query_text_keys), query_text_max_chars)))

    qids = [x[0] for x in query_pairs]
    q_texts = [x[1] for x in query_pairs]
    q_embeddings = load_or_build_query_embeddings(
        model=model,
        query_ids=qids,
        query_texts=q_texts,
        batch_size=batch_size,
        prompt_name=query_prompt_name,
        prompt=query_prompt,
        cache_path=query_emb_cache,
        dtype=embedding_dtype,
    )
    q_embeddings = q_embeddings.astype(pool_embeddings.dtype, copy=False)
    total_queries = len(qids)
    progress_step = 100 if total_queries >= 100 else max(total_queries, 1)
    started_at = time.perf_counter()
    print(f"Starting similarity retrieval for {total_queries} queries (k={k_eff}, pool={pool_size})...")

    for i, qid in enumerate(qids):
        if k_eff <= 0:
            rows.append(
                {
                    "method": "sim",
                    "query_id": qid,
                    "ordered_demo_ids": [],
                    "ordered_pool_indices": [],
                }
            )
        else:
            scores = pool_embeddings @ q_embeddings[i]
            cand = np.argpartition(scores, -k_eff)[-k_eff:]
            order = np.lexsort((cand, -scores[cand]))
            top_idx = cand[order].tolist()
            rows.append(
                {
                    "method": "sim",
                    "query_id": qid,
                    "ordered_demo_ids": [pool_ids[j] for j in top_idx],
                    "ordered_pool_indices": top_idx,
                }
            )

        done = i + 1
        if done == 1 or done % progress_step == 0 or done == total_queries:
            elapsed = time.perf_counter() - started_at
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total_queries - done) / rate if rate > 0 else 0.0
            print(
                "Similarity retrieval progress: "
                f"{done}/{total_queries} ({done / total_queries:.1%}) | "
                f"elapsed {format_duration(elapsed)} | "
                f"eta {format_duration(eta)}"
            )
    return rows


def main() -> None:
    args = parse_args()
    method = normalize_method_name(args.method)
    eval_config = load_eval_config(args.config)
    selected_k = args.k if isinstance(args.k, int) else method_k_from_config(eval_config, method)
    if selected_k is None:
        selected_k = 0 if method == "zero" else 5
    if selected_k < 0:
        raise ValueError("--k must be >= 0")

    if not args.pool.exists():
        raise FileNotFoundError(f"Pool not found: {args.pool}")
    if not args.queries.exists():
        raise FileNotFoundError(f"Queries not found: {args.queries}")
    if method == "sim" and not Path(args.embedding_model).exists():
        raise FileNotFoundError(f"Embedding model not found: {args.embedding_model}")

    pool = read_jsonl(args.pool)
    queries = read_jsonl(args.queries)
    if args.pool_limit > 0:
        pool = pool[: args.pool_limit]
    if args.query_limit > 0:
        queries = queries[: args.query_limit]

    pool_ids = [resolve_demo_id(raw, i) for i, raw in enumerate(pool)]
    query_text_keys = [k.strip() for k in args.query_text_keys.split(",") if k.strip()]
    if not query_text_keys:
        query_text_keys = ["prompt"]

    if method == "zero":
        out_rows = build_zero_rows(queries=queries, query_id_key=args.query_id_key)
    elif method == "random":
        out_rows = build_random_rows(
            queries=queries,
            query_id_key=args.query_id_key,
            pool_size=len(pool),
            pool_ids=pool_ids,
            k=selected_k,
            seed=args.seed,
        )
    elif method == "sim":
        pool_embeddings: np.ndarray | None = None
        out_rows: list[dict[str, Any]] | None = None

        can_try_cache_only = (
            args.pool_emb_cache is not None
            and args.query_emb_cache is not None
            and args.pool_emb_cache.exists()
            and args.query_emb_cache.exists()
            and (
                args.query_emb_cache.with_suffix(args.query_emb_cache.suffix + ".meta.json").exists()
                or args.query_emb_cache.with_suffix(".meta.json").exists()
            )
        )
        if can_try_cache_only:
            try:
                # Cache-only fast path: skip loading the embedding model when both caches are valid.
                pool_embeddings = load_or_build_pool_embeddings(
                    model=None,
                    pool_texts=[""] * len(pool),
                    batch_size=args.embedding_batch_size,
                    cache_path=args.pool_emb_cache,
                    dtype=args.embedding_dtype,
                )
                out_rows = build_similarity_rows(
                    queries=queries,
                    query_id_key=args.query_id_key,
                    query_text_keys=query_text_keys,
                    pool_ids=pool_ids,
                    pool_embeddings=pool_embeddings,
                    model=None,
                    k=selected_k,
                    batch_size=args.embedding_batch_size,
                    query_prompt_name=args.similarity_query_prompt_name,
                    query_prompt=args.similarity_query_prompt,
                    query_text_max_chars=args.query_text_max_chars,
                    query_emb_cache=args.query_emb_cache,
                    embedding_dtype=args.embedding_dtype,
                )
                print("Loaded pool/query embeddings from cache. Skipped embedding model loading.")
            except Exception as e:
                print(f"Cache-only path unavailable: {e}")
                print("Falling back to embedding model loading.")

        if out_rows is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(args.embedding_model, device=args.embedding_device)
            if args.embedding_max_seq_length > 0:
                model.max_seq_length = args.embedding_max_seq_length

            pool_texts = [truncate_text(extract_demo_problem(raw), args.pool_text_max_chars) for raw in pool]
            pool_embeddings = load_or_build_pool_embeddings(
                model=model,
                pool_texts=pool_texts,
                batch_size=args.embedding_batch_size,
                cache_path=args.pool_emb_cache,
                dtype=args.embedding_dtype,
            )
            out_rows = build_similarity_rows(
                queries=queries,
                query_id_key=args.query_id_key,
                query_text_keys=query_text_keys,
                pool_ids=pool_ids,
                pool_embeddings=pool_embeddings,
                model=model,
                k=selected_k,
                batch_size=args.embedding_batch_size,
                query_prompt_name=args.similarity_query_prompt_name,
                query_prompt=args.similarity_query_prompt,
                query_text_max_chars=args.query_text_max_chars,
                query_emb_cache=args.query_emb_cache,
                embedding_dtype=args.embedding_dtype,
            )
    else:
        raise ValueError(f"Unsupported method: {method}")

    write_jsonl(args.out, out_rows)
    meta_out = args.meta_out if args.meta_out is not None else args.out.with_suffix(args.out.suffix + ".meta.json")
    write_json(
        meta_out,
        {
            "method": method,
            "pool_file": str(args.pool),
            "query_file": str(args.queries),
            "output_file": str(args.out),
            "query_id_key": args.query_id_key,
            "query_text_keys": query_text_keys,
            "k": selected_k,
            "seed": args.seed,
            "pool_size": len(pool),
            "query_size": len(queries),
            "pool_limit": args.pool_limit,
            "query_limit": args.query_limit,
            "similarity": {
                "embedding_model": args.embedding_model,
                "embedding_device": args.embedding_device,
                "embedding_batch_size": args.embedding_batch_size,
                "embedding_max_seq_length": args.embedding_max_seq_length,
                "query_prompt_name": args.similarity_query_prompt_name,
                "query_prompt": args.similarity_query_prompt,
                "pool_emb_cache": str(args.pool_emb_cache) if args.pool_emb_cache else "",
                "query_emb_cache": str(args.query_emb_cache) if args.query_emb_cache else "",
                "embedding_dtype": args.embedding_dtype,
                "pool_text_max_chars": args.pool_text_max_chars,
                "query_text_max_chars": args.query_text_max_chars,
            },
        },
    )

    print(f"Method: {method}")
    print(f"Selected k: {selected_k}")
    print(f"Pool rows: {len(pool)}")
    print(f"Query rows: {len(queries)}")
    print(f"Output rows: {len(out_rows)}")
    print(f"Output: {args.out}")
    print(f"Meta: {meta_out}")


if __name__ == "__main__":
    main()
