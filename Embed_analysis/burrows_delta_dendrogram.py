#!/usr/bin/env python3
"""Cluster JSON queries with Burrows' Delta and draw a dendrogram.

The value of each record's ``query`` key drives the analysis. Input positions
are used as synthetic filenames (query_00001, query_00002, ...). The binary
``label`` is used only for plot styling and optional balanced sampling; it
never directly affects distances or cluster membership.

The implementation uses the 100 corpus-wide most frequent words (MFW),
relative word frequencies, sample-standardized z-scores, Burrows' Delta
(mean absolute z-score difference), and average-linkage clustering.
python burrows_delta_dendrogram.py --balance-labels --no-labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import re

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "all_queries.json"
DEFAULT_MATRIX = HERE / "burrows_delta_distances.npy"
DEFAULT_METADATA = HERE / "burrows_delta_metadata.json"
DEFAULT_PLOT = HERE / "burrows_delta_dendrogram.png"
DEFAULT_GROUPS = HERE / "burrows_delta_groups.json"
N_MFW = 100

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an average-linkage Burrows' Delta dendrogram."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--matrix-output", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--groups-output", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument(
        "--n-clusters", type=int, default=10,
        help="number of groups obtained by cutting the tree (default: 10)",
    )
    parser.add_argument(
        "--truncate", type=int,
        help="show only the final N merged clusters; calculations still use every query",
    )
    parser.add_argument(
        "--matrix-block-size", type=int, default=512,
        help="rows computed at once when writing the symmetric matrix (default: 512)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="recompute the matrix even if compatible cached outputs exist",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="omit query filenames from the PNG",
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


def load_queries(path: Path) -> list[str]:
    """Load only the query strings; deliberately ignore all other attributes."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"{path}: expected a JSON array containing at least two records")

    queries: list[str] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"record {index}: expected an object")
        # This is intentionally the sole field access on each source record.
        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"record {index}: 'query' must be a non-empty string")
        queries.append(query)
    return queries


def load_ground_truth_labels(path: Path) -> np.ndarray:
    """Load binary labels for display/sampling, never for clustering."""
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


def query_labels(count: int) -> list[str]:
    width = max(5, len(str(count)))
    return [f"query_{index:0{width}d}" for index in range(1, count + 1)]


def query_fingerprint(queries: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        encoded = query.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def tokenize_queries(queries: Sequence[str]) -> list[list[str]]:
    try:
        from nltk.tokenize import RegexpTokenizer
    except ImportError as exc:
        raise RuntimeError("missing NLTK; install dependencies with: pip install -r requirements.txt") from exc
    # Unicode letters with optional internal apostrophes; no NLTK data download needed.
    tokenizer = RegexpTokenizer(r"[^\W\d_]+(?:['’][^\W\d_]+)*|,")
    return [tokenizer.tokenize(query.lower()) for query in queries]


def burrows_zscores(
    queries: Sequence[str], n_mfw: int = N_MFW
) -> tuple[np.ndarray, list[str]]:
    print("first, we tokenize each query")
    tokenized = tokenize_queries(queries)
    print("so aftern tokenization, this : ", len(tokenized), " tokenized queries")
    print("8 first tokenized queries : ", tokenized[0], " and ", tokenized[1], " and ", tokenized[2], " and ", tokenized[3], " and ", tokenized[4], " and ", tokenized[5], " and ", tokenized[6], " and ", tokenized[7])
    print(tokenized[680], " and ", tokenized[681], " and ", tokenized[682], " and ", tokenized[683], " and ", tokenized[684], " and ", tokenized[685], " and ", tokenized[686], " and ", tokenized[687])
    print("then we pick the 100 most frequent words across all queries")
    vocabulary = [word for word, _ in Counter(
        word for tokens in tokenized for word in tokens
    ).most_common(n_mfw)]
    print("vocabulary length is : ", len(vocabulary))
    print("first 10 words in vocabulary : ", vocabulary[:10])
    if not vocabulary:
        raise ValueError("no word tokens were found in the queries")

    word_to_column = {word: column for column, word in enumerate(vocabulary)}
    print("then we create a dictionnary to know the column index of each word in the vocabulary")
    frequencies = np.zeros((len(queries), len(vocabulary)), dtype=np.float64)
    print("now we intilize a matrix called frequencies, and in this matrix each row will correspond to a query and each column will correspond to a word in the vocabulary")
    print("now for each query")
    for row, tokens in enumerate(tokenized):
        if not tokens:
            continue
        counts = Counter(tokens) #return a dict, contains how many times each  token  appears in tokens
        denominator = float(len(tokens))
        for word, count in counts.items():
            column = word_to_column.get(word)
            if column is not None:
                frequencies[row, column] = count / denominator

    means = frequencies.mean(axis=0) #compute the mean of each column
    standard_deviations = frequencies.std(axis=0, ddof=1) #std on each column
    usable = standard_deviations > 0.0
    zero_std_count = np.count_nonzero(~usable)
    print(f"Number of features with standard deviation = 0: {zero_std_count}")
    if not np.all(usable):
        vocabulary = [word for word, keep in zip(vocabulary, usable) if keep]
        frequencies = frequencies[:, usable]
        means = means[usable]
        standard_deviations = standard_deviations[usable]
    if frequencies.shape[1] == 0:
        raise ValueError("the MFW features have no variation between queries")
    return (frequencies - means) / standard_deviations, vocabulary


def write_distance_matrix(
    zscores: np.ndarray, output: Path, block_size: int
) -> np.memmap:
    """Write a symmetric Delta matrix without holding two copies in RAM."""
    try:
        from scipy.spatial.distance import cdist
        from numpy.lib.format import open_memmap
    except ImportError as exc:
        raise RuntimeError("missing NumPy/SciPy; install dependencies with: pip install -r requirements.txt") from exc
    if block_size < 1:
        raise ValueError("--matrix-block-size must be at least 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    count, feature_count = zscores.shape
    matrix = open_memmap(output, mode="w+", dtype=np.float32, shape=(count, count))
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        for other_start in range(start, count, block_size):
            other_stop = min(other_start + block_size, count)
            distances = (
                cdist(
                    zscores[start:stop], zscores[other_start:other_stop],
                    metric="cityblock",
                )
                / feature_count
            ).astype(np.float32)
            if other_start == start:
                distances = (distances + distances.T) / 2.0
            matrix[start:stop, other_start:other_stop] = distances
            matrix[other_start:other_stop, start:stop] = distances.T
        print(f"distance rows {stop:,}/{count:,}", file=sys.stderr)
    np.fill_diagonal(matrix, 0.0)
    matrix.flush()
    return matrix


def obtain_distance_matrix(
    queries: Sequence[str], matrix_path: Path, metadata_path: Path,
    block_size: int, force: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    fingerprint = query_fingerprint(queries)
    if not force and matrix_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            matrix = np.load(matrix_path, mmap_mode="r")
            compatible = (
                metadata.get("query_sha256") == fingerprint
                and metadata.get("n_mfw_requested") == N_MFW
                and matrix.shape == (len(queries), len(queries))
            )
            if compatible:
                print(f"reusing compatible distance matrix: {matrix_path}", file=sys.stderr)
                return matrix, metadata
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    print("queries shape is : ", len(queries))
    zscores, vocabulary = burrows_zscores(queries)
    print("after burrow zscores, we have zscores and vocabulary")
    print("zscores shape is : ", zscores.shape)
    print("vocabulary length is : ", len(vocabulary))
    print("that zscore matrix has a row for each query and a column for each word in the top100 vocabulary")
    print("each cell is like how percentage of that query that top100 word represents" )
    print("and it is normalized")
    print("then we compute the distance between every pair of row and put this into what we call a distance matrix")
    matrix = write_distance_matrix(zscores, matrix_path, block_size)
    print("shape of the distance matrix is : ", matrix.shape)
    metadata: dict[str, object] = {
        "method": "Burrows' Delta (mean absolute difference of MFW z-scores)",
        "tokenizer": "NLTK RegexpTokenizer; lowercase Unicode alphabetic words",
        "n_queries": len(queries),
        "n_mfw_requested": N_MFW,
        "n_mfw_used": len(vocabulary),
        "mfw": vocabulary,
        "labels": query_labels(len(queries)),
        "query_sha256": fingerprint,
        "matrix_dtype": "float32",
        "matrix_file": str(matrix_path.resolve()),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return matrix, metadata


def condensed_distances(matrix: np.ndarray) -> np.ndarray:
    count = matrix.shape[0]
    result = np.empty(count * (count - 1) // 2, dtype=np.float64)
    cursor = 0
    for row in range(count - 1):
        width = count - row - 1
        result[cursor:cursor + width] = matrix[row, row + 1:]
        cursor += width
    return result


def average_linkage(matrix: np.ndarray) -> np.ndarray:
    try:
        from scipy.cluster.hierarchy import linkage
    except ImportError as exc:
        raise RuntimeError("missing SciPy; install dependencies with: pip install -r requirements.txt") from exc
    return linkage(condensed_distances(matrix), method="average")


def canonical_groups(linkage_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster
    count = linkage_matrix.shape[0] + 1
    if not 2 <= n_clusters <= count:
        raise ValueError(f"--n-clusters must be between 2 and {count}")
    raw = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    members: dict[int, list[int]] = {}
    for index, group in enumerate(raw):
        members.setdefault(int(group), []).append(index)
    ordered = sorted(members, key=lambda group: (-len(members[group]), members[group][0]))
    mapping = {old: new for new, old in enumerate(ordered, start=1)}
    return np.asarray([mapping[int(group)] for group in raw], dtype=np.int32)


def save_groups(path: Path, labels: Sequence[str], groups: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"filename": label, "group": int(group)}
        for label, group in zip(labels, groups)
    ]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def plot_dendrogram(
    linkage_matrix: np.ndarray, labels: Sequence[str], ground_truth: np.ndarray,
    output: Path, truncate: int | None, show_labels: bool,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from scipy.cluster.hierarchy import dendrogram
    except ImportError as exc:
        raise RuntimeError("missing Matplotlib/SciPy; install dependencies with: pip install -r requirements.txt") from exc

    count = len(labels)
    label_colors = {0: "#1f77b4", 1: "#d62728"}
    uniform_label: dict[int, int | None] = {
        index: int(ground_truth[index]) for index in range(count)
    }
    for offset, row in enumerate(linkage_matrix):
        left, right = int(row[0]), int(row[1])
        uniform_label[count + offset] = (
            uniform_label[left]
            if uniform_label[left] == uniform_label[right]
            else None
        )

    if truncate is not None and truncate < 2:
        raise ValueError("--truncate must be at least 2")
    visible = min(count, truncate or count)
    width = max(18.0, min(120.0, visible * 0.055))
    figure, axis = plt.subplots(figsize=(width, 12))
    kwargs: dict[str, object] = {}
    if truncate is not None:
        kwargs.update(truncate_mode="lastp", p=truncate, show_contracted=True)
    tree = dendrogram(
        linkage_matrix, labels=list(labels), leaf_rotation=90, leaf_font_size=4,
        no_labels=not show_labels, color_threshold=0,
        above_threshold_color="#808080", ax=axis, **kwargs,
    )
    if show_labels:
        label_to_truth = {
            label: int(true_label) for label, true_label in zip(labels, ground_truth)
        }
        for tick in axis.get_xmajorticklabels():
            true_label = label_to_truth.get(tick.get_text())
            if true_label is not None:
                tick.set_color(label_colors[true_label])

    # Keep label identity visible even with --no-labels. With a truncated tree,
    # a collapsed leaf is gray when it contains a mixture of labels.
    leaf_colors = [
        label_colors.get(uniform_label.get(int(node_id)), "#808080")
        for node_id in tree["leaves"]
    ]
    leaf_x = np.arange(5, 10 * len(leaf_colors) + 5, 10)
    axis.scatter(
        leaf_x, np.zeros(len(leaf_colors)), c=leaf_colors, marker="o", s=18,
        edgecolors="none", zorder=3, clip_on=False,
    )
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=label_colors[value],
               markersize=6, label=f"Label {value}")
        for value in (0, 1)
    ]
    axis.legend(handles=legend_handles, title="Ground-truth label", loc="upper right")
    axis.set_title(
        "Query stylometry: 100-MFW Burrows' Delta / average linkage\n"
        "Leaf color = ground-truth label"
    )
    axis.set_xlabel("Query filename (input position)")
    axis.set_ylabel("Burrows' Delta")
    axis.grid(axis="y", alpha=0.2)
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
        linkage_matrix = average_linkage(matrix)
        groups = canonical_groups(linkage_matrix, args.n_clusters)
        save_groups(args.groups_output, labels, groups)
        plot_dendrogram(
            linkage_matrix, labels, ground_truth, args.plot_output,
            args.truncate, not args.no_labels,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"distance matrix: {args.matrix_output}")
    print(f"group assignments: {args.groups_output}")
    print(f"dendrogram: {args.plot_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
