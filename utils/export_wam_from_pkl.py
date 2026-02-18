import argparse
import csv
from pathlib import Path

import numpy as np
from mig.label import SimLabelGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export WAM (weighted adjacency matrix) from a label graph pkl."
    )
    parser.add_argument(
        "--graph-pkl",
        type=Path,
        default=Path("outputs/label_graph.pkl"),
        help="Path to the dumped label graph pkl file.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/wam.csv"),
        help="Path to save WAM csv file (with row/column labels).",
    )
    parser.add_argument(
        "--out-npy",
        type=Path,
        default=Path("outputs/wam.npy"),
        help="Path to save WAM npy file.",
    )
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=None,
        help="Optional path to save labels as txt (one label per line).",
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

    labels = graph.labels
    wam = graph.wam
    if hasattr(wam, "detach"):
        wam = wam.detach().cpu().numpy()
    elif not isinstance(wam, np.ndarray):
        wam = np.array(wam)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_npy.parent.mkdir(parents=True, exist_ok=True)

    np.save(args.out_npy, wam)

    with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + labels)
        for label, row in zip(labels, wam):
            writer.writerow([label] + row.tolist())

    if args.labels_out is not None:
        args.labels_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.labels_out, "w", encoding="utf-8") as f:
            for label in labels:
                f.write(f"{label}\n")

    print(f"Loaded graph: {graph_path}")
    print(f"Number of labels: {len(labels)}")
    print(f"WAM shape: {tuple(wam.shape)}")
    print(f"Saved npy: {args.out_npy}")
    print(f"Saved csv: {args.out_csv}")
    if args.labels_out is not None:
        print(f"Saved labels: {args.labels_out}")


if __name__ == "__main__":
    main()
