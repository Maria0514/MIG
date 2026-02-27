import argparse
import gzip
import json

from glob import glob
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export parquet files to sharded JSONL files (one output shard per parquet file)."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        required=True,
        help="Glob for parquet files, e.g. data/raw/openhermes-2.5-pool-annotated/default/train/*.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for output shards.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="shard",
        help="Output shard filename prefix.",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Write .jsonl.gz instead of .jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every N rows for each shard.",
    )
    return parser.parse_args()


def build_output_path(out_dir: Path, prefix: str, shard_idx: int, use_gzip: bool) -> Path:
    suffix = ".jsonl.gz" if use_gzip else ".jsonl"
    return out_dir / f"{prefix}-{shard_idx:04d}{suffix}"


def write_shard(
    parquet_path: str,
    out_path: Path,
    use_gzip: bool,
    progress_every: int,
) -> int:
    ds = load_dataset("parquet", data_files={"train": parquet_path}, split="train")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if use_gzip else open

    num_rows = 0
    with opener(out_path, "wt", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            num_rows += 1
            if progress_every > 0 and num_rows % progress_every == 0:
                print(f"[{out_path.name}] rows={num_rows}")

    return num_rows


def main() -> None:
    args = parse_args()

    parquet_files = sorted(glob(args.input_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matched: {args.input_glob}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    print(f"Found parquet files: {len(parquet_files)}")

    for idx, parquet_path in enumerate(parquet_files):
        out_path = build_output_path(args.out_dir, args.prefix, idx, args.gzip)
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists: {out_path}. Use --overwrite to replace existing shards."
            )

        print(f"[{idx+1}/{len(parquet_files)}] {parquet_path} -> {out_path}")
        rows = write_shard(
            parquet_path=parquet_path,
            out_path=out_path,
            use_gzip=args.gzip,
            progress_every=args.progress_every,
        )
        total_rows += rows
        print(f"[{out_path.name}] done rows={rows}")

    print(f"All done. shards={len(parquet_files)} total_rows={total_rows}")


if __name__ == "__main__":
    main()
