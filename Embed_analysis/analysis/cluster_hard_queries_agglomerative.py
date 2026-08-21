#!/usr/bin/env python3
"""Cluster query embeddings with agglomerative hierarchical clustering.

The default invocation is directly comparable to the existing HDBSCAN run: it
reads the same five-dimensional UMAP embeddings and creates ten clusters.

Examples:
    python3 analysis/cluster_hard_queries_agglomerative.py

    python3 analysis/cluster_hard_queries_agglomerative.py \
        --n-clusters 15 \
        --linkage average

    python3 analysis/cluster_hard_queries_agglomerative.py \
        --select-k 2 30

    python3 analysis/cluster_hard_queries_agglomerative.py \
        --input analysis/hard_queries_embedded.json \
        --linkage average \
        --metric cosine \
        --select-k 2 30

The output contains every source record, its assigned cluster, distances to the
cluster centroid, representative queries, and a silhouette score. Clusters are
renumbered from largest to smallest so that cluster 0 is always the largest.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "hard_queries_embedded_umap_5d.json"
DEFAULT_OUTPUT = HERE / "hard_queries_agglomerative_clusters.json"
DEFAULT_N_CLUSTERS = 10


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster precomputed query embeddings hierarchically."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"input JSON array (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output JSON report (default: {DEFAULT_OUTPUT})",
    )

    cut_group = parser.add_mutually_exclusive_group()
    cut_group.add_argument(
        "--n-clusters",
        type=int,
        help=f"number of clusters (default: {DEFAULT_N_CLUSTERS})",
    )
    cut_group.add_argument(
        "--distance-threshold",
        type=float,
        help="cut the hierarchy at this distance instead of choosing a cluster count",
    )
    cut_group.add_argument(
        "--select-k",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="try every cluster count in this inclusive range and use the best silhouette score",
    )

    parser.add_argument(
        "--linkage",
        choices=("ward", "complete", "average", "single"),
        default="ward",
        help="linkage rule (default: ward)",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="distance metric; ward linkage requires euclidean (default: euclidean)",
    )
    parser.add_argument(
        "--representatives-per-cluster",
        type=int,
        default=6,
        help="number of centroid-nearest representative queries (default: 6)",
    )
    return parser.parse_args(argv)


def load_records(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load records and their finite, consistently sized embeddings."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc

    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON array")

    records: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []
    expected_dimensions: int | None = None

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"record {index}: expected a JSON object")
        query = item.get("query")
        if not isinstance(query, str):
            raise ValueError(f"record {index}: 'query' must be a string")
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"record {index}: 'embedding' must be a non-empty list")

        if expected_dimensions is None:
            expected_dimensions = len(embedding)
        elif len(embedding) != expected_dimensions:
            raise ValueError(
                f"record {index}: expected {expected_dimensions} embedding values, "
                f"found {len(embedding)}"
            )

        numeric_embedding: list[float] = []
        for dimension, value in enumerate(embedding):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"record {index}: embedding value {dimension} is not numeric"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(
                    f"record {index}: embedding value {dimension} is not finite"
                )
            numeric_embedding.append(number)

        records.append(dict(item))
        embeddings.append(numeric_embedding)

    return records, np.asarray(embeddings, dtype=np.float64)


def validate_options(args: argparse.Namespace, record_count: int) -> None:
    if args.n_clusters is not None and not 2 <= args.n_clusters < record_count:
        raise ValueError(
            f"--n-clusters must be between 2 and {record_count - 1}"
        )
    if args.distance_threshold is not None and args.distance_threshold <= 0:
        raise ValueError("--distance-threshold must be positive")
    if args.select_k is not None:
        minimum, maximum = args.select_k
        if not 2 <= minimum <= maximum < record_count:
            raise ValueError(
                "--select-k must satisfy "
                f"2 <= MIN <= MAX <= {record_count - 1}"
            )
    if args.linkage == "ward" and args.metric != "euclidean":
        raise ValueError("ward linkage only supports the euclidean metric")
    if args.representatives_per_cluster < 1:
        raise ValueError("--representatives-per-cluster must be at least 1")


def fit_labels(
    embeddings: np.ndarray,
    *,
    n_clusters: int | None,
    distance_threshold: float | None,
    linkage: str,
    metric: str,
) -> np.ndarray:
    """Fit one agglomerative model and return integer labels."""
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise RuntimeError(
            "missing dependency; install it with: "
            "python3 -m pip install 'scikit-learn>=1.2'"
        ) from exc

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_full_tree=True,
        compute_distances=True,
    )
    return np.asarray(model.fit_predict(embeddings), dtype=np.int64)


def canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters by descending size, then first input position."""
    counts = Counter(int(label) for label in labels)
    first_positions: dict[int, int] = {}
    for index, raw_label in enumerate(labels):
        first_positions.setdefault(int(raw_label), index)
    ordered = sorted(
        counts,
        key=lambda label: (-counts[label], first_positions[label]),
    )
    mapping = {old_label: new_label for new_label, old_label in enumerate(ordered)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)


def pairwise_distance_matrix(
    embeddings: np.ndarray,
    metric: str,
) -> np.ndarray | None:
    """Build a reusable distance matrix for small datasets when SciPy supports it."""
    if len(embeddings) > 5_000:
        return None
    try:
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(embeddings, metric=metric))
    except (ImportError, TypeError, ValueError):
        return None
    if not np.isfinite(distances).all():
        return None
    return np.asarray(distances, dtype=np.float64)


def silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str,
    *,
    distances: np.ndarray | None = None,
) -> float:
    try:
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise RuntimeError(
            "missing dependency; install it with: "
            "python3 -m pip install 'scikit-learn>=1.2'"
        ) from exc

    cluster_count = len(set(int(label) for label in labels))
    if not 2 <= cluster_count < len(labels):
        return float("nan")
    if distances is None:
        distances = pairwise_distance_matrix(embeddings, metric)
    if distances is not None:
        return float(silhouette_score(distances, labels, metric="precomputed"))
    return float(silhouette_score(embeddings, labels, metric=metric))


def choose_k(
    embeddings: np.ndarray,
    *,
    minimum: int,
    maximum: int,
    linkage: str,
    metric: str,
) -> tuple[int, list[dict[str, float | int]]]:
    """Select the cluster count with the highest silhouette score."""
    evaluations: list[dict[str, float | int]] = []
    distances = pairwise_distance_matrix(embeddings, metric)
    for cluster_count in range(minimum, maximum + 1):
        labels = fit_labels(
            embeddings,
            n_clusters=cluster_count,
            distance_threshold=None,
            linkage=linkage,
            metric=metric,
        )
        score = silhouette(embeddings, labels, metric, distances=distances)
        evaluations.append(
            {"n_clusters": cluster_count, "silhouette_score": score}
        )

    best = max(
        evaluations,
        key=lambda result: (float(result["silhouette_score"]), -int(result["n_clusters"])),
    )
    return int(best["n_clusters"]), evaluations


def distance_summary(distances: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
    }


def distances_to_centroid(
    embeddings: np.ndarray,
    centroid: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Calculate centroid distances without a large pairwise distance matrix."""
    centroid_vector = centroid.reshape(-1)
    if metric == "euclidean":
        return np.linalg.norm(embeddings - centroid_vector, axis=1)
    if metric == "cosine":
        row_norms = np.linalg.norm(embeddings, axis=1)
        centroid_norm = float(np.linalg.norm(centroid_vector))
        similarities = np.zeros(len(embeddings), dtype=np.float64)
        valid = (row_norms > 0) & (centroid_norm > 0)
        similarities[valid] = np.sum(
            embeddings[valid] * centroid_vector,
            axis=1,
        ) / (row_norms[valid] * centroid_norm)
        return np.clip(1.0 - similarities, 0.0, 2.0)

    try:
        from sklearn.metrics import pairwise_distances
    except ImportError as exc:
        raise RuntimeError(
            "missing dependency; install it with: "
            "python3 -m pip install 'scikit-learn>=1.2'"
        ) from exc
    return pairwise_distances(embeddings, centroid, metric=metric).reshape(-1)


def rank_summary(records: Sequence[dict[str, Any]]) -> dict[str, float | int] | None:
    ranks = [
        float(record["rank"])
        for record in records
        if isinstance(record.get("rank"), (int, float))
        and not isinstance(record.get("rank"), bool)
        and math.isfinite(float(record["rank"]))
    ]
    if not ranks:
        return None
    return {
        "min": min(ranks),
        "mean": float(statistics.fmean(ranks)),
        "median": float(statistics.median(ranks)),
        "max": max(ranks),
    }


def build_report(
    records: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    input_path: Path,
    requested_n_clusters: int | None,
    distance_threshold: float | None,
    linkage: str,
    metric: str,
    representatives_per_cluster: int,
    k_evaluations: Sequence[dict[str, float | int]],
) -> dict[str, Any]:
    """Build a report containing cluster members and representatives."""
    labels = canonicalize_labels(labels)
    cluster_ids = sorted(set(int(label) for label in labels))
    clusters: list[dict[str, Any]] = []

    for cluster_id in cluster_ids:
        member_indices = np.flatnonzero(labels == cluster_id)
        cluster_embeddings = embeddings[member_indices]
        centroid = np.mean(cluster_embeddings, axis=0, keepdims=True)
        distances = distances_to_centroid(cluster_embeddings, centroid, metric)
        order = np.argsort(distances, kind="stable")
        representative_positions = order[:representatives_per_cluster]

        members: list[dict[str, Any]] = []
        for local_position, source_index in enumerate(member_indices):
            member = dict(records[int(source_index)])
            member["agglomerative"] = {
                "cluster_id": cluster_id,
                "distance_to_centroid": float(distances[local_position]),
            }
            members.append(member)

        representatives = []
        for local_position in representative_positions:
            source_index = int(member_indices[int(local_position)])
            representatives.append(
                {
                    "id": records[source_index].get("id"),
                    "query": records[source_index]["query"],
                    "distance_to_centroid": float(distances[int(local_position)]),
                }
            )

        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(member_indices),
                "centroid": centroid.reshape(-1).tolist(),
                "distance_to_centroid_summary": distance_summary(distances),
                "rank_summary": rank_summary(members),
                "representative_queries": representatives,
                "queries": members,
            }
        )

    score = silhouette(embeddings, labels, metric)
    return {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": "AgglomerativeClustering",
            "parameters": {
                "n_clusters": requested_n_clusters,
                "distance_threshold": distance_threshold,
                "linkage": linkage,
                "metric": metric,
                "compute_full_tree": True,
                "label_order": "descending_cluster_size_then_first_input_index",
            },
            "embedding_dimensions": int(embeddings.shape[1]),
            "total_queries": len(records),
            "cluster_count": len(cluster_ids),
            "silhouette_score": None if math.isnan(score) else score,
            "input_file": str(input_path.resolve()),
            "k_selection": list(k_evaluations),
        },
        "clusters": clusters,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records, embeddings = load_records(args.input)
        validate_options(args, len(records))

        k_evaluations: list[dict[str, float | int]] = []
        if args.select_k is not None:
            selected_k, k_evaluations = choose_k(
                embeddings,
                minimum=args.select_k[0],
                maximum=args.select_k[1],
                linkage=args.linkage,
                metric=args.metric,
            )
            print(
                f"Selected {selected_k} clusters by maximum silhouette score "
                f"from the range {args.select_k[0]}..{args.select_k[1]}."
            )
            n_clusters: int | None = selected_k
        elif args.distance_threshold is not None:
            n_clusters = None
        else:
            n_clusters = (
                DEFAULT_N_CLUSTERS if args.n_clusters is None else args.n_clusters
            )

        labels = fit_labels(
            embeddings,
            n_clusters=n_clusters,
            distance_threshold=args.distance_threshold,
            linkage=args.linkage,
            metric=args.metric,
        )
        report = build_report(
            records,
            embeddings,
            labels,
            input_path=args.input,
            requested_n_clusters=n_clusters,
            distance_threshold=args.distance_threshold,
            linkage=args.linkage,
            metric=args.metric,
            representatives_per_cluster=args.representatives_per_cluster,
            k_evaluations=k_evaluations,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    metadata = report["metadata"]
    score_text = (
        "unavailable"
        if metadata["silhouette_score"] is None
        else f"{metadata['silhouette_score']:.4f}"
    )
    print(
        f"Clustered {metadata['total_queries']} queries into "
        f"{metadata['cluster_count']} clusters; "
        f"silhouette={score_text}."
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
