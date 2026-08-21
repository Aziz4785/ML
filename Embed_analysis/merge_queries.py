#!/usr/bin/env python3
"""Merge easy and hard query datasets into one JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def load_queries(path: Path) -> list[Any]:
    """Load a JSON array from ``path``."""
    with path.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not isinstance(queries, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge easy_queries.json and hard_queries.json."
    )
    parser.add_argument(
        "--easy",
        type=Path,
        default=HERE / "easy_queries.json",
        help="Path to the easy queries JSON file",
    )
    parser.add_argument(
        "--hard",
        type=Path,
        default=HERE / "hard_queries.json",
        help="Path to the hard queries JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "all_queries.json",
        help="Path for the merged JSON file",
    )
    args = parser.parse_args()

    easy_queries = [
        {**query, "label": 0} for query in load_queries(args.easy)
    ]
    hard_queries = [
        {**query, "label": 1} for query in load_queries(args.hard)
    ]
    all_queries = easy_queries + hard_queries

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(all_queries, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(
        f"Created {args.output} with {len(all_queries)} queries "
        f"({len(easy_queries)} easy + {len(hard_queries)} hard)."
    )


if __name__ == "__main__":
    main()
