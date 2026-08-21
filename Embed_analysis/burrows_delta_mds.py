#!/usr/bin/env python3
"""Map JSON queries with classical MDS over 100-MFW Burrows' Delta.

This script reuses the exact extraction, Delta matrix, average-linkage cut,
and synthetic filenames from ``burrows_delta_dendrogram.py``.
Only each selected input record's ``query`` value participates in the
Delta/MDS and clustering calculations. The binary ``label`` is used for plot
styling, cluster-quality evaluation, and optional balanced sampling; it never
directly affects positions or cluster membership.

python3 burrows_delta_mds.py --no-labels
or
python3 burrows_delta_mds.py 
or 
python3 burrows_delta_mds.py --balance-labels --no-labels
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np



from burrows_delta_dendrogram import (
    DEFAULT_INPUT,
    DEFAULT_MATRIX,
    DEFAULT_METADATA,
    N_MFW,
    average_linkage,
    canonical_groups,
    load_queries,
    obtain_distance_matrix,
    query_labels,
    save_groups,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PLOT = HERE / f"burrows_delta_mds_{N_MFW}.png"
DEFAULT_COORDINATES = HERE / "burrows_delta_mds_coordinates.json"
DEFAULT_GROUPS = HERE / "burrows_delta_groups.json"
DEFAULT_LABEL_SUMMARY = HERE / "burrows_delta_cluster_label_summary.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 2D classical-MDS plot from Burrows' Delta."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--matrix-output", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--coordinates-output", type=Path, default=DEFAULT_COORDINATES)
    parser.add_argument("--groups-output", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--label-summary-output", type=Path, default=DEFAULT_LABEL_SUMMARY)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--matrix-block-size", type=int, default=512)
    parser.add_argument(
        "--force", action="store_true",
        help="recompute the matrix even if compatible cached outputs exist",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="omit point labels from the PNG (coordinate JSON still labels every query)",
    )
    parser.add_argument(
        "--balance-labels", action="store_true",
        help=(
            "use equal numbers of labels 0 and 1 by randomly downsampling the "
            "majority label"
        ),
    )
    parser.add_argument(
        "--random-seed", type=int, default=0,
        help="random seed used by --balance-labels (default: 0)",
    )
    return parser.parse_args(argv)


def load_ground_truth_labels(path: Path) -> np.ndarray:
    """Load binary labels for display/evaluation, never for clustering."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array")
    result: list[int] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"record {index}: expected an object")
        value = record.get("label")
        if isinstance(value, bool) or value not in (0, 1):
            raise ValueError(f"record {index}: 'label' must be integer 0 or 1")
        result.append(int(value))
    return np.asarray(result, dtype=np.int8)


def balanced_indices(labels: np.ndarray, random_seed: int) -> np.ndarray:
    """Return source-order indices with equally many examples of both labels."""
    label_0 = np.flatnonzero(labels == 0)
    label_1 = np.flatnonzero(labels == 1)
    sample_size = min(len(label_0), len(label_1))
    if sample_size == 0:
        raise ValueError("--balance-labels requires at least one query of each label")

    rng = np.random.default_rng(random_seed)
    selected_0 = rng.choice(label_0, size=sample_size, replace=False)
    selected_1 = rng.choice(label_1, size=sample_size, replace=False)
    return np.sort(np.concatenate((selected_0, selected_1)))


def classical_mds(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a scalable two-dimensional classical (Torgerson) MDS solution."""
    try:
        from scipy.sparse.linalg import LinearOperator, eigsh
    except ImportError as exc:
        raise RuntimeError("missing SciPy; install dependencies with: pip install -r requirements.txt") from exc
    count = matrix.shape[0]
    # B = -0.5 J D^2 J. Applying centering in the matvec avoids materializing
    # another full n-by-n matrix and makes all-query MDS feasible.
    def multiply(vector: np.ndarray) -> np.ndarray:
        centered = vector - vector.mean()
        if not np.all(np.isfinite(centered)):
            raise RuntimeError("MDS eigensolver supplied a non-finite vector")
        product = np.empty(count, dtype=np.float64)
        block = 512
        for start in range(0, count, block):
            stop = min(start + block, count)
            distances = np.asarray(matrix[start:stop], dtype=np.float64)
            # Some BLAS builds leak harmless floating-point status flags into
            # NumPy here. Validate explicitly instead of printing false alarms.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                block_product = (distances * distances) @ centered
            if not np.all(np.isfinite(block_product)):
                raise RuntimeError("non-finite values encountered during MDS")
            product[start:stop] = block_product
        product *= -0.5
        return product - product.mean()

    operator = LinearOperator((count, count), matvec=multiply, dtype=np.float64)
    eigenvalues, eigenvectors = eigsh(operator, k=2, which="LA", tol=1e-6)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.any(eigenvalues <= 0):
        raise RuntimeError(
            "classical MDS did not find two positive dimensions for this distance matrix"
        )
    coordinates = eigenvectors * np.sqrt(eigenvalues)
    # Resolve arbitrary eigenvector signs for repeatable plots.
    for dimension in range(2):
        pivot = int(np.argmax(np.abs(coordinates[:, dimension])))
        if coordinates[pivot, dimension] < 0:
            coordinates[:, dimension] *= -1
    return coordinates, eigenvalues


def save_coordinates(
    path: Path, labels: Sequence[str], groups: np.ndarray,
    ground_truth: np.ndarray, coordinates: np.ndarray, eigenvalues: np.ndarray,
) -> None:
    records = [
        {
            "filename": label,
            "group": int(group),
            "label": int(true_label),
            "mds_1": float(point[0]),
            "mds_2": float(point[1]),
        }
        for label, group, true_label, point
        in zip(labels, groups, ground_truth, coordinates)
    ]
    payload = {
        "method": "classical MDS of the 100-MFW Burrows' Delta matrix",
        "eigenvalues": [float(value) for value in eigenvalues],
        "points": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_label_summary(
    path: Path, groups: np.ndarray, ground_truth: np.ndarray
) -> None:
    """Save a contingency table and standard cluster/label agreement metrics."""
    try:
        from sklearn.metrics import (
            adjusted_rand_score,
            homogeneity_completeness_v_measure,
            normalized_mutual_info_score,
        )
    except ImportError as exc:
        raise RuntimeError(
            "missing scikit-learn; install dependencies with: pip install -r requirements.txt"
        ) from exc

    rows: list[dict[str, int | float]] = []
    correctly_assigned_by_majority = 0
    for group in sorted(set(int(value) for value in groups)):
        selected = ground_truth[groups == group]
        label_0 = int(np.count_nonzero(selected == 0))
        label_1 = int(np.count_nonzero(selected == 1))
        size = label_0 + label_1
        correctly_assigned_by_majority += max(label_0, label_1)
        rows.append({
            "group": group,
            "size": size,
            "label_0": label_0,
            "label_1": label_1,
            "label_1_percentage": 100.0 * label_1 / size,
            "dominant_label": 0 if label_0 >= label_1 else 1,
            "purity": max(label_0, label_1) / size,
        })

    total = len(ground_truth)
    total_0 = int(np.count_nonzero(ground_truth == 0))
    total_1 = total - total_0
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        ground_truth, groups
    )
    payload = {
        "note": (
            "After any optional label-balanced sampling, labels did not participate "
            "in Burrows' Delta, MDS, or average-linkage clustering."
        ),
        "overall": {
            "queries": total,
            "label_0": total_0,
            "label_1": total_1,
            "label_1_percentage": 100.0 * total_1 / total,
            "majority_class_baseline": max(total_0, total_1) / total,
            "cluster_purity": correctly_assigned_by_majority / total,
            "adjusted_rand_index": float(adjusted_rand_score(ground_truth, groups)),
            "normalized_mutual_information": float(
                normalized_mutual_info_score(ground_truth, groups)
            ),
            "homogeneity": float(homogeneity),
            "completeness": float(completeness),
            "v_measure": float(v_measure),
        },
        "clusters": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_mds(
    coordinates: np.ndarray, labels: Sequence[str], ground_truth: np.ndarray,
    output: Path, show_labels: bool,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError("missing Matplotlib; install dependencies with: pip install -r requirements.txt") from exc
    figure, axis = plt.subplots(figsize=(18, 14))
    label_styles = {
        0: {"color": "#1d79d5", "marker": "o", "size": 42},
        1: {"color": "#d62728", "marker": "o", "size": 42},
    }
    for true_label, style in label_styles.items():
        selected = ground_truth == true_label
        axis.scatter(
            coordinates[selected, 0], coordinates[selected, 1],
            s=style["size"], marker=style["marker"], color=style["color"],
            edgecolors="none" if true_label == 0 else "black",
            alpha=0.52 if true_label == 0 else 0.95,
            linewidths=0 if true_label == 0 else 0.35,
            zorder=2 if true_label == 0 else 3,
        )
    if show_labels:
        for label, true_label, (x_value, y_value) in zip(
            labels, ground_truth, coordinates
        ):
            axis.annotate(
                f"{label} [L{int(true_label)}]", (x_value, y_value), xytext=(2, 2),
                textcoords="offset points", fontsize=3,
                fontweight="bold" if true_label == 1 else "normal",
                alpha=0.9 if true_label == 1 else 0.45,
            )
    axis.set_title(
        "Query stylometry: 100-MFW Burrows' Delta MDS"
    )
    axis.set_xlabel("MDS dimension 1")
    axis.set_ylabel("MDS dimension 2")
    axis.grid(alpha=0.2)
    label_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=label_styles[0]["color"],
               markersize=6, label="Easy"),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=label_styles[1]["color"],
                markersize=6, label="Hard"),
    ]
    axis.legend(handles=label_handles, title="Ground-truth label", loc="upper right")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queries = load_queries(args.input)
        ground_truth = load_ground_truth_labels(args.input)
        if len(ground_truth) != len(queries):
            raise ValueError("the number of query and label values does not match")
        labels = query_labels(len(queries))
        if args.balance_labels:
            selected = balanced_indices(ground_truth, args.random_seed)
            queries = [queries[index] for index in selected]
            labels = [labels[index] for index in selected]
            ground_truth = ground_truth[selected]
            print(
                f"balanced sample: {len(ground_truth) // 2:,} label 0 + "
                f"{len(ground_truth) // 2:,} label 1",
                file=sys.stderr,
            )
        matrix, _ = obtain_distance_matrix(
            queries, args.matrix_output, args.metadata_output,
            args.matrix_block_size, args.force,
        )
        print("shape of the distance matrix is : ", matrix.shape)
        print("that matrix contains the distance between every pair of queries, and it is symmetric")
        # The same average-linkage rule and canonical numbering as the dendrogram.
        linkage_matrix = average_linkage(matrix)
        groups = canonical_groups(linkage_matrix, args.n_clusters)
        coordinates, eigenvalues = classical_mds(matrix)
        save_groups(args.groups_output, labels, groups)
        save_coordinates(
            args.coordinates_output, labels, groups, ground_truth,
            coordinates, eigenvalues
        )
        save_label_summary(args.label_summary_output, groups, ground_truth)
        plot_mds(
            coordinates, labels, ground_truth, args.plot_output, not args.no_labels
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"distance matrix: {args.matrix_output}")
    print(f"MDS coordinates: {args.coordinates_output}")
    print(f"cluster/label summary: {args.label_summary_output}")
    print(f"MDS plot: {args.plot_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
