#!/usr/bin/env python3
"""Cluster precomputed query embeddings with HDBSCAN.

When ``DIMENSION_OF_EMBEDDINGS`` is ``"max"``, this script clusters the original
embeddings in ``hard_queries_embedded.json`` directly. For a numeric dimension,
it instead reads the corresponding UMAP-reduced embeddings.

Install the dependency, if needed, with:

    python3 -m pip install "hdbscan>=0.8.42,<0.9"

Then run:

    python3 analysis/cluster_hard_queries_hdbscan.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
DIMENSION_OF_EMBEDDINGS = 2
DEFAULT_INPUT = (
    HERE / "hard_queries_embedded.json"
    if DIMENSION_OF_EMBEDDINGS == "max"
    else HERE / f"hard_queries_embedded_umap_{DIMENSION_OF_EMBEDDINGS}d.json"
)
DEFAULT_OUTPUT = HERE / f"hard_queries_hdbscan_clusters_{DIMENSION_OF_EMBEDDINGS}d.json"

DEFAULT_MIN_CLUSTER_SIZE = 10

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster precomputed query embeddings with HDBSCAN."
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
        help=f"output JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help=f"minimum number of queries in a cluster (default: {DEFAULT_MIN_CLUSTER_SIZE})",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help=(
            "HDBSCAN min_samples; by default HDBSCAN uses min_cluster_size "
            "(5 unless overridden)"
        ),
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="distance metric used by HDBSCAN (default: euclidean)",
    )
    parser.add_argument(
        "--cluster-selection-method",
        choices=("eom", "leaf"),
        default="eom",
        help="HDBSCAN cluster selection method (default: eom)",
    )
    return parser.parse_args(argv)


def load_records(path: Path) -> tuple[list[dict[str, Any]], list[list[float]]]:
    """Load and validate a JSON array containing numeric embeddings."""
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

    return records, embeddings


def make_member(
    record: dict[str, Any],
    *,
    label: int,
    probability: float,
    outlier_score: float,
) -> dict[str, Any]:
    """Copy a source record and attach its HDBSCAN results."""
    member = dict(record)
    member["hdbscan"] = {
        "cluster_id": label,
        "is_noise": label == -1,
        "probability": probability,
        "outlier_score": outlier_score,
    }
    return member


def cluster_records(
    records: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
    *,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
    cluster_selection_method: str,
) -> dict[str, Any]:
    """Fit HDBSCAN and build a JSON-serializable report."""
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    if min_samples is not None and min_samples < 1:
        raise ValueError("min_samples must be at least 1")

    try:
        import hdbscan
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "missing dependency; install it with: "
            "python3 -m pip install 'hdbscan>=0.8.42,<0.9'"
        ) from exc

    matrix = np.asarray(embeddings, dtype=np.float64)
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        max_cluster_size=50,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    labels = model.fit_predict(matrix)
    probabilities = model.probabilities_
    outlier_scores = model.outlier_scores_

    # HDBSCAN labels non-noise clusters with consecutive integers beginning at 0.
    cluster_ids = sorted(int(value) for value in set(labels) if int(value) >= 0)
    persistence = {
        cluster_id: float(model.cluster_persistence_[position])
        for position, cluster_id in enumerate(cluster_ids)
    }
    members_by_label: dict[int, list[dict[str, Any]]] = {
        cluster_id: [] for cluster_id in cluster_ids
    }
    noise: list[dict[str, Any]] = []

    if not (
        len(records) == len(labels) == len(probabilities) == len(outlier_scores)
    ):
        raise RuntimeError("HDBSCAN returned result arrays with inconsistent lengths")

    for record, raw_label, raw_probability, raw_outlier_score in zip(
        records, labels, probabilities, outlier_scores
    ):
        label = int(raw_label)
        member = make_member(
            record,
            label=label,
            probability=float(raw_probability),
            outlier_score=float(raw_outlier_score),
        )
        if label == -1:
            noise.append(member)
        else:
            members_by_label[label].append(member)

    clusters = []
    for cluster_id in cluster_ids:
        members = members_by_label[cluster_id]
        cluster_probabilities = [item["hdbscan"]["probability"] for item in members]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "cluster_persistence": persistence[cluster_id],
                "probability_summary": {
                    "min": min(cluster_probabilities),
                    "mean": sum(cluster_probabilities) / len(cluster_probabilities),
                    "max": max(cluster_probabilities),
                },
                # Full source records are included so no input information is lost.
                "queries": members,
            }
        )

    counts = Counter(int(label) for label in labels)
    return {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": "HDBSCAN",
            "parameters": {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "effective_min_samples": (
                    min_cluster_size if min_samples is None else min_samples
                ),
                "metric": metric,
                "cluster_selection_method": cluster_selection_method,
                "prediction_data": True,
            },
            "embedding_dimensions": int(matrix.shape[1]),
            "total_queries": len(records),
            "cluster_count": len(cluster_ids),
            "clustered_query_count": len(records) - counts.get(-1, 0),
            "noise_query_count": counts.get(-1, 0),
            "noise_fraction": counts.get(-1, 0) / len(records),
        },
        "clusters": clusters,
        "noise": {
            "cluster_id": -1,
            "size": len(noise),
            "queries": noise,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records, embeddings = load_records(args.input)
        report = cluster_records(
            records,
            embeddings,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric=args.metric,
            cluster_selection_method=args.cluster_selection_method,
        )
        report["metadata"]["input_file"] = str(args.input.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    metadata = report["metadata"]
    print(
        f"Clustered {metadata['total_queries']} queries into "
        f"{metadata['cluster_count']} clusters; "
        f"{metadata['noise_query_count']} queries were labeled as noise."
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
