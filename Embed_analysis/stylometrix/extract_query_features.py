#!/usr/bin/env python3
"""Extract StyloMetrix feature vectors from ``all_queries.json``.

By default, the script reads the repository's ``all_queries.json`` and writes
``stylometrix/all_queries_stylometrix.json``.  All English metrics exposed by
StyloMetrix are enabled; version 0.1.9.1 currently produces 196 features.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "all_queries.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "all_queries_stylometrix.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract all English StyloMetrix features for every query and "
            "store the vectors with their labels."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON array (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Queries passed to StyloMetrix per transform call (default: 256)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N queries (useful for a smoke test)",
    )
    return parser.parse_args()


def load_queries(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than zero")
        data = data[:limit]
    if not data:
        raise ValueError(f"No queries found in {path}")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} is not a JSON object")
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ValueError(f"Item {index} has no non-empty string 'query'")
        if "label" not in item:
            raise ValueError(f"Item {index} has no 'label'")

    return data


def batches(items: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def json_number(value: Any) -> float | None:
    """Return a JSON number, mapping missing, NaN, and infinite values to null."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def extract_features(
    queries: list[dict[str, Any]], batch_size: int
) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        import stylo_metrix as sm
    except ImportError as error:
        raise RuntimeError(
            "StyloMetrix is not installed. Run: "
            "python -m pip install -r stylometrix/requirements.txt"
        ) from error

    try:
        extractor = sm.StyloMetrix("en")
    except OSError as error:
        raise RuntimeError(
            "The English spaCy model is unavailable. Run: "
            "python -m spacy download en_core_web_trf"
        ) from error

    feature_names: list[str] | None = None
    output_records: list[dict[str, Any]] = []

    for query_batch in batches(queries, batch_size):
        frame = extractor.transform(item["query"] for item in query_batch)
        current_names = [str(column) for column in frame.columns if column != "text"]

        if feature_names is None:
            feature_names = current_names
            if len(feature_names) <= 190:
                raise RuntimeError(
                    "Expected more than 190 StyloMetrix features, but got "
                    f"{len(feature_names)}"
                )
            if len(set(feature_names)) != len(feature_names):
                raise RuntimeError("StyloMetrix returned duplicate feature names")
        elif current_names != feature_names:
            raise RuntimeError("StyloMetrix feature columns changed between batches")

        if len(frame) != len(query_batch):
            raise RuntimeError(
                f"StyloMetrix returned {len(frame)} rows for "
                f"a batch of {len(query_batch)} queries"
            )

        vectors = frame[feature_names].to_numpy().tolist()
        for item, values in zip(query_batch, vectors):
            record: dict[str, Any] = {
                "query": item["query"],
                "label": item["label"],
                "feature_vector": [json_number(value) for value in values],
            }
            if "id" in item:
                record = {"id": item["id"], **record}
            output_records.append(record)

    if feature_names is None:
        raise RuntimeError("No feature names were returned")
    return feature_names, output_records


def write_output(
    path: Path,
    source: Path,
    feature_names: list[str],
    records: list[dict[str, Any]],
) -> None:
    result = {
        "metadata": {
            "source": str(source.resolve()),
            "language": "en",
            "feature_count": len(feature_names),
            "query_count": len(records),
            "null_values_represent_missing_or_non_finite_metrics": True,
        },
        "feature_names": feature_names,
        "queries": records,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    queries = load_queries(args.input, args.limit)
    print(f"Loaded {len(queries)} queries from {args.input}")

    feature_names, records = extract_features(queries, args.batch_size)
    write_output(args.output, args.input, feature_names, records)

    print(
        f"Wrote {len(records)} queries with {len(feature_names)} features each "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
