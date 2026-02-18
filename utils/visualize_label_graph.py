import argparse
from pathlib import Path

from mig.label import SimLabelGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a serialized MIG label graph (.pkl)."
    )
    parser.add_argument(
        "--graph-pkl",
        type=Path,
        default=Path("outputs/label_graph.pkl"),
        help="Path to the dumped label graph pkl file.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="jaal",
        help="Visualization backend. Current project supports: jaal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = args.graph_pkl

    if not graph_path.exists():
        raise FileNotFoundError(f"Label graph file not found: {graph_path}")

    # Load graph from pickle without initializing embedding model again.
    graph = SimLabelGraph.__new__(SimLabelGraph)
    graph.load(str(graph_path))

    print(f"Loaded graph: {graph_path}")
    print(f"Number of labels: {len(graph.labels)}")
    print(f"WAM shape: {tuple(graph.wam.shape)}")

    graph.visualize(args.backend)


if __name__ == "__main__":
    main()
