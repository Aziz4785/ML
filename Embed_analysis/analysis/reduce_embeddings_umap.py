#!/usr/bin/env python3
"""Reduce query embeddings to five dimensions with UMAP.

By default, this reads ``hard_queries_embedded.json`` from this script's
directory and writes ``hard_queries_embedded_umap_5d.json`` beside it. Each
output record preserves its existing fields, with ``embedding`` replaced by
the corresponding five-dimensional UMAP vector.

Example:
    python3 analysis/reduce_embeddings_umap.py

    python3 analysis/reduce_embeddings_umap.py \
        analysis/hard_queries_embedded.json \
        analysis/hard_queries_embedded_umap_5d.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
N_COMPONENTS = 500
DEFAULT_INPUT = HERE / "hard_queries_embedded.json"
DEFAULT_OUTPUT = HERE / f"hard_queries_embedded_umap_{N_COMPONENTS}d.json"



def load_records(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load records and return their embeddings as a numeric matrix."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise ValueError("Input JSON must be a non-empty top-level array.")

    embeddings: list[list[float]] = []
    expected_size: int | None = None

    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a JSON object.")

        embedding = record.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(
                f"Record {index} must contain a non-empty 'embedding' list."
            )

        if expected_size is None:
            expected_size = len(embedding)
        elif len(embedding) != expected_size:
            raise ValueError(
                f"Record {index} has {len(embedding)} dimensions; "
                f"expected {expected_size}."
            )

        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in embedding
        ):
            raise ValueError(f"Record {index} contains a non-numeric embedding value.")

        embeddings.append(embedding)

    matrix = np.asarray(embeddings, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must not contain NaN or infinite values.")

    return data, matrix


def reduce_embeddings(
    embeddings: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    """Fit UMAP and transform the embedding matrix to five dimensions."""
    try:
        from umap import UMAP
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'umap-learn'. Install it with: "
            "python3 -m pip install umap-learn"
        ) from exc

    if len(embeddings) < 3:
        raise ValueError("UMAP requires at least three records.")
    if not 2 <= n_neighbors < len(embeddings):
        raise ValueError(
            f"--n-neighbors must be between 2 and {len(embeddings) - 1}."
        )
    if min_dist < 0:
        raise ValueError("--min-dist must be non-negative.")

    reducer = UMAP(
        n_components=N_COMPONENTS,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def save_records(
    records: list[dict[str, Any]],
    reduced_embeddings: np.ndarray,
    output_path: Path,
) -> None:
    """Write records with their reduced embeddings to a JSON file."""
    output_records = []
    if len(records) != len(reduced_embeddings):
        raise ValueError("The number of reduced embeddings does not match the records.")

    for record, embedding in zip(records, reduced_embeddings):
        output_record = dict(record)
        output_record["embedding"] = embedding.tolist()
        output_records.append(output_record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_records, file, ensure_ascii=False)

    print(
        f"Saved {len(output_records):,} records with {N_COMPONENTS}D embeddings "
        f"to {output_path}"
    )


def plot_pairwise_distances(reduced_embeddings: np.ndarray) -> None:
    """Compute and display Euclidean distances between every unique point pair."""
    try:
        import matplotlib.pyplot as plt
        from scipy.spatial.distance import pdist
    except ImportError as exc:
        raise SystemExit(
            "Missing plotting dependencies. Install them with: "
            "python3 -m pip install matplotlib scipy"
        ) from exc

    distances = pdist(reduced_embeddings, metric="euclidean")
    mean_distance = float(np.mean(distances))
    std_distance = float(np.std(distances))

    print(f"Pairwise Euclidean distance mean: {mean_distance:.6f}")
    print(f"Pairwise Euclidean distance std:  {std_distance:.6f}")

    _, axis = plt.subplots(figsize=(10, 6))
    axis.hist(distances, bins="auto", edgecolor="black", alpha=0.75)
    axis.axvline(
        mean_distance,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_distance:.4f}",
    )
    axis.set(
        title="Pairwise distances between reduced embeddings",
        xlabel="Euclidean distance",
        ylabel="Frequency",
    )
    axis.text(
        0.98,
        0.95,
        f"Mean: {mean_distance:.4f}\nStd: {std_distance:.4f}",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axis.legend()
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce JSON embeddings to five dimensions with UMAP."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Input JSON file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Size of UMAP's local neighborhood. Default: 15.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="Minimum distance between points in the result. Default: 0.1.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="Distance metric used on the input embeddings. Default: cosine.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible results. Default: 42.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, embeddings = load_records(args.input)
    print(
        f"Reducing {embeddings.shape[0]:,} embeddings from "
        f"{embeddings.shape[1]}D to {N_COMPONENTS}D..."
    )
    reduced_embeddings = reduce_embeddings(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )
    save_records(records, reduced_embeddings, args.output)
    plot_pairwise_distances(reduced_embeddings)


if __name__ == "__main__":
    main()
