#!/usr/bin/env python3
"""Extract query text and high-confidence examples by HDBSCAN cluster ID."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence



"""
python analysis/extract_cluster_queries.py
"""
HERE = Path(__file__).resolve().parent

DIMENSION_OF_EMBEDDINGS = 2
DEFAULT_INPUT = HERE / f"hard_queries_hdbscan_clusters_{DIMENSION_OF_EMBEDDINGS}d.json"
DEFAULT_OUTPUT = HERE / f"hard_queries_by_cluster_{DIMENSION_OF_EMBEDDINGS}d.json"




def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write only the query strings grouped by cluster ID."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exclude-noise",
        action="store_true",
        help="omit HDBSCAN noise queries (cluster -1)",
    )
    parser.add_argument(
        "--examples-per-cluster",
        type=int,
        default=6,
        help="number of highest-probability example prompts per group (default: 5)",
    )
    return parser.parse_args(argv)


def read_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc

    if not isinstance(report, dict) or not isinstance(report.get("clusters"), list):
        raise ValueError(f"{path}: expected an HDBSCAN cluster report")
    return report


def validated_queries(group: dict[str, Any], group_name: str) -> list[dict[str, Any]]:
    queries = group.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"{group_name}: 'queries' must be a list")

    for index, record in enumerate(queries):
        if not isinstance(record, dict):
            raise ValueError(f"{group_name}: query {index} must be an object")
        query = record.get("query")
        if not isinstance(query, str):
            raise ValueError(f"{group_name}: query {index} has no string 'query' field")
        hdbscan = record.get("hdbscan")
        if not isinstance(hdbscan, dict) or not isinstance(
            hdbscan.get("probability"), (int, float)
        ):
            raise ValueError(
                f"{group_name}: query {index} has no numeric HDBSCAN probability"
            )
    return queries


def format_group(
    group: dict[str, Any], group_name: str, examples_per_cluster: int
) -> dict[str, Any]:
    """Return all query strings and the highest-probability examples."""
    records = validated_queries(group, group_name)
    ranked = sorted(
        enumerate(records),
        key=lambda item: (-float(item[1]["hdbscan"]["probability"]), item[0]),
    )
    return {
        "queries": [record["query"] for record in records],
        "example_prompts": [
            record["query"] for _, record in ranked[:examples_per_cluster]
        ],
        "description": "",
    }


def extract_queries(
    report: dict[str, Any], *, include_noise: bool = True, examples_per_cluster: int = 5
) -> dict[str, dict[str, Any]]:
    """Return all query text and high-confidence examples for each cluster."""
    if examples_per_cluster < 1:
        raise ValueError("examples_per_cluster must be at least 1")

    output: dict[str, dict[str, Any]] = {}

    for index, cluster in enumerate(report["clusters"]):
        if not isinstance(cluster, dict) or "cluster_id" not in cluster:
            raise ValueError(f"cluster {index}: expected an object with 'cluster_id'")
        cluster_id = str(cluster["cluster_id"])
        output[cluster_id] = format_group(
            cluster, f"cluster {cluster_id}", examples_per_cluster
        )

    if include_noise:
        noise = report.get("noise")
        if not isinstance(noise, dict):
            raise ValueError("report has no valid 'noise' group")
        output[str(noise.get("cluster_id", -1))] = format_group(
            noise, "noise", examples_per_cluster
        )

    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = read_report(args.input)
        result = extract_queries(
            report,
            include_noise=not args.exclude_noise,
            examples_per_cluster=args.examples_per_cluster,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total = sum(len(group["queries"]) for group in result.values())
    print(f"Wrote {total} queries across {len(result)} groups to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
