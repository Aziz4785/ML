#!/usr/bin/env python3
"""Keep all non-MS MARCO queries and a sample of MS MARCO queries."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = HERE / "all_queries.json"
DEFAULT_OUTPUT_PATH = HERE / "all_queries_fewMS.json"


def fraction(value: str) -> float:
    """Parse a sampling fraction between 0 and 1."""
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("fraction must be between 0 and 1")
    return parsed


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSON array of query objects with string IDs."""
    with path.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not isinstance(queries, list):
        raise ValueError(f"Expected a JSON array in {path}")

    for index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f"Expected an object at index {index} in {path}")
        if not isinstance(query.get("id"), str):
            raise ValueError(f"Expected a string 'id' at index {index} in {path}")

    return queries


def filter_queries(
    queries: list[dict[str, Any]],
    *,
    keyword: str = "msmarco",
    keep_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return all other queries plus a seeded sample of matching queries."""
    matching_indexes = [
        index for index, query in enumerate(queries) if keyword in query["id"]
    ]
    keep_count = round(len(matching_indexes) * keep_fraction)
    kept_matching_indexes = set(
        random.Random(seed).sample(matching_indexes, keep_count)
    )

    filtered = [
        query
        for index, query in enumerate(queries)
        if index in kept_matching_indexes or keyword not in query["id"]
    ]
    return filtered, len(matching_indexes), keep_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Keep all queries except for IDs containing a keyword, of which only "
            "a seeded sample is retained."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSON array (default: {DEFAULT_INPUT_PATH.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT_PATH.name})",
    )
    parser.add_argument(
        "--keyword",
        default="msmarco",
        help="Case-sensitive substring to match in each query ID (default: msmarco)",
    )
    parser.add_argument(
        "--fraction",
        type=fraction,
        default=0.1,
        help="Fraction of matching queries to retain (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    args = parser.parse_args()

    queries = load_queries(args.input)
    filtered, matching_count, kept_matching_count = filter_queries(
        queries,
        keyword=args.keyword,
        keep_fraction=args.fraction,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(filtered, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(
        f"Created {args.output} with {len(filtered)} queries: "
        f"kept {kept_matching_count}/{matching_count} IDs containing "
        f"{args.keyword!r}, plus {len(queries) - matching_count} other queries."
    )


if __name__ == "__main__":
    main()
