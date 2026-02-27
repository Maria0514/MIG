import argparse
from pathlib import Path

from mig.label import SimLabelGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast re-threshold for a dumped MIG label graph pkl (no embedding recompute)."
    )
    parser.add_argument(
        "--graph-pkl",
        type=Path,
        required=True,
        help="Source label graph pkl path.",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        required=True,
        help="New similarity threshold.",
    )
    parser.add_argument(
        "--out-pkl",
        type=Path,
        default=None,
        help="Output graph pkl path. Default: <graph-pkl>_tXX.pkl",
    )
    return parser.parse_args()


def default_out_path(src: Path, threshold: float) -> Path:
    t = int(round(threshold * 100))
    return src.with_name(f"{src.stem}_t{t:02d}{src.suffix}")


def edge_stats(graph: SimLabelGraph) -> tuple[int, float]:
    wam = graph.wam
    n = len(graph.labels)
    if n <= 1:
        return 0, 0.0
    edge_count = int(((wam > 0).sum().item() - n) // 2)
    total_possible = n * (n - 1) // 2
    density = edge_count / total_possible
    return edge_count, density


def main() -> None:
    args = parse_args()
    src = args.graph_pkl
    if not src.exists():
        raise FileNotFoundError(f"Graph pkl not found: {src}")

    out = args.out_pkl if args.out_pkl is not None else default_out_path(src, args.sim_threshold)
    out.parent.mkdir(parents=True, exist_ok=True)

    graph = SimLabelGraph.__new__(SimLabelGraph)
    graph.load(str(src))
    graph.rethreshold(args.sim_threshold)
    graph.dump(str(out))

    edges, density = edge_stats(graph)
    print(f"Input graph: {src}")
    print(f"Output graph: {out}")
    print(f"Labels: {len(graph.labels)}")
    print(f"Threshold: {args.sim_threshold}")
    print(f"Edges: {edges}")
    print(f"Density: {density:.6f}")


if __name__ == "__main__":
    main()
