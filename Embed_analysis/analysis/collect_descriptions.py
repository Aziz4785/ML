#!/usr/bin/env python3
"""Collect cluster descriptions from the hard-query clustering reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = (
    HERE / "hard_queries_by_cluster_2d.json",
    HERE / "hard_queries_by_cluster_50d.json",
    HERE / "hard_queries_by_cluster_maxd.json",
)
DEFAULT_OUTPUT = HERE / "hard_query_descriptions.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="cluster JSON files (defaults to the 2d, 50d, and maxd reports)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output JSON file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> Any:
    """Read JSON from *path*, adding useful file and location context to errors."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc


def extract_descriptions(report: Any, source: Path | str = "input") -> list[str]:
    """Extract descriptions from one cluster report in its existing order."""
    if not isinstance(report, dict):
        raise ValueError(f"{source}: expected a JSON object keyed by cluster ID")

    descriptions: list[str] = []
    for cluster_id, cluster in report.items():
        if not isinstance(cluster, dict):
            raise ValueError(f"{source}: cluster {cluster_id!r} must be an object")
        description = cluster.get("description")
        if not isinstance(description, str):
            raise ValueError(
                f"{source}: cluster {cluster_id!r} has no string 'description' field"
            )
        descriptions.append(description)

    return descriptions


def collect_descriptions(paths: Sequence[Path]) -> list[str]:
    """Read *paths* and return one flat list containing every description."""
    descriptions: list[str] = []
    for path in paths:
        descriptions.extend(extract_descriptions(read_json(path), path))
    return descriptions


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        descriptions = collect_descriptions(args.inputs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(descriptions, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {len(descriptions)} descriptions from {len(args.inputs)} files "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
