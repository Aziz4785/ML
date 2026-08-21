#!/usr/bin/env python3
"""Measure retrieval quality using the baseline OpenAI embeddings.

For every row in queries_msmarco.jsonl, queries_manual.jsonl and queries_generated.jsonl,
look up its generated-text embedding in llm_embeddings.jsonl / llm_msmarco_embeddings.jsonl
(queries without such an embedding are skipped).

Each query embedding is then compared against every corpus embedding
(embedded_corpus.jsonl + embedded_corpus_msmarco.jsonl) with cosine similarity, and we
report where the query's own corpus paragraph ranks.

Examples:
    python3 measure_baseline.py
    python3 measure_baseline.py --output baseline_report.json
    python3 measure_baseline.py --queries queries_manual.jsonl
    python3 measure_baseline.py --hard-only

"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_QUERY_PATHS = [
    HERE / "queries_msmarco.jsonl",
    HERE / "queries_manual.jsonl",
    HERE / "queries_generated.jsonl",
]
DEFAULT_LLM_EMBEDDING_PATHS = [
    HERE / "llm_embeddings.jsonl",
    HERE / "llm_msmarco_embeddings.jsonl",
]
DEFAULT_CORPUS_EMBEDDING_PATHS = [
    HERE / "embedded_corpus_msmarco.jsonl",
    HERE / "embedded_corpus.jsonl",
]

# Similarities are computed one query block at a time so the score matrix stays small.
QUERY_BATCH_SIZE = 256

RANK_FAILURE_THRESHOLD = 60
HARD_QUERIES_PATH = HERE / "hard_queries.json"
EASY_QUERIES_PATH = HERE / "easy_queries.json"
QUERY_ID_SUFFIXES = (
    "_prompt1",
    "_prompt2",
    "_m",
    "_prompt3",
    "_prompta",
    "_prompt4",
    "_prompt5",
    "_prompt6",
    "_prompt7",
    "_prompt8",
)


@dataclass(frozen=True)
class EmbeddingIndex:
    """L2-normalized embeddings for a set of files, keyed by record id."""

    ids: list[str]
    positions: dict[str, int]
    matrix: np.ndarray
    paragraph_ids: list[str | None]
    model: str | None

    @property
    def dimensions(self) -> int:
        return int(self.matrix.shape[1])

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    paragraph_id: str
    query: str
    source: str


@dataclass(frozen=True)
class RankingResult:
    query_id: str
    paragraph_id: str
    query: str
    source: str
    rank: int
    cosine_similarity: float


def iter_jsonl(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(location, record)`` pairs so errors can name file and line."""
    seen_any = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{location}: expected a JSON object")
            seen_any = True
            yield location, record
    if not seen_any:
        raise ValueError(f"{path}: file contains no records")


def _non_empty_string(value: Any, *, location: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be a non-empty string")
    return value.strip()


def validate_embedding(embedding: Any, *, location: str) -> np.ndarray:
    """Return a finite, non-zero embedding as a float32 vector."""
    if not isinstance(embedding, list) or not embedding:
        raise ValueError(f"{location}: embedding must be a non-empty list")
    try:
        vector = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: embedding contains a non-numeric value") from exc
    if vector.ndim != 1:
        raise ValueError(f"{location}: embedding must be a flat list of numbers")
    if not np.isfinite(vector).all():
        raise ValueError(f"{location}: embedding contains a non-finite value")
    if float(np.linalg.norm(vector)) == 0.0:
        raise ValueError(f"{location}: embedding has zero magnitude")
    return vector


def load_embedding_index(
    paths: Sequence[Path], *, description: str, verbose: bool = False
) -> EmbeddingIndex:
    """Load and L2-normalize every embedding row across ``paths``, keyed by ``id``."""
    ids: list[str] = []
    positions: dict[str, int] = {}
    paragraph_ids: list[str | None] = []
    vectors: list[np.ndarray] = []
    origins: dict[str, str] = {}
    models: set[str] = set()
    expected_dimensions: int | None = None

    for path in paths:
        if verbose:
            print(f"Loading {description}: {path.name}", file=sys.stderr, flush=True)
        for location, record in iter_jsonl(path):
            try:
                raw_id = record["id"]
                raw_embedding = record["embedding"]
            except KeyError as exc:
                raise ValueError(f"{location}: missing {exc.args[0]}") from exc

            record_id = _non_empty_string(raw_id, location=location, field="id")
            if record_id in positions:
                raise ValueError(
                    f"{location}: duplicate id {record_id} "
                    f"(first seen at {origins[record_id]})"
                )
            embedding = validate_embedding(raw_embedding, location=location)
            if expected_dimensions is None:
                expected_dimensions = embedding.size
            elif embedding.size != expected_dimensions:
                raise ValueError(
                    f"{location}: embedding has {embedding.size} dimensions; "
                    f"expected {expected_dimensions}"
                )

            raw_model = record.get("embedding_model")
            if raw_model is not None:
                models.add(
                    _non_empty_string(
                        raw_model, location=location, field="embedding_model"
                    )
                )
            raw_paragraph_id = record.get("paragraph_id")
            paragraph_id = (
                _non_empty_string(
                    raw_paragraph_id, location=location, field="paragraph_id"
                )
                if raw_paragraph_id is not None
                else None
            )

            positions[record_id] = len(ids)
            origins[record_id] = location
            ids.append(record_id)
            paragraph_ids.append(paragraph_id)
            vectors.append(embedding)

    if not ids:
        raise ValueError(f"{description}: no embeddings were loaded")
    if len(models) > 1:
        raise ValueError(
            f"{description} contain multiple embedding models: {sorted(models)}"
        )

    matrix = np.vstack(vectors)
    del vectors
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return EmbeddingIndex(
        ids=ids,
        positions=positions,
        matrix=matrix,
        paragraph_ids=paragraph_ids,
        model=next(iter(models), None),
    )


def load_queries(
    paths: Sequence[Path],
    *,
    keep_ids: set[str] | None = None,
    hard_only: bool = False,
    verbose: bool = False,
) -> tuple[list[QueryRecord], dict[str, int]]:
    """Load queries, optionally keeping only ids in ``keep_ids`` or containing ``_hard``.

    Returns the kept queries and a per-source count of the skipped ones.
    """
    queries: list[QueryRecord] = []
    skipped: dict[str, int] = defaultdict(int)
    origins: dict[str, str] = {}
    selected_count = 0

    for path in paths:
        if verbose:
            print(f"Loading queries: {path.name}", file=sys.stderr, flush=True)
        source = path.stem
        for location, record in iter_jsonl(path):
            try:
                query_id = _non_empty_string(record["id"], location=location, field="id")
            except KeyError as exc:
                raise ValueError(f"{location}: missing {exc.args[0]}") from exc
            if hard_only and "_hard" not in query_id:
                continue
            selected_count += 1
            try:
                paragraph_id = _non_empty_string(
                    record["paragraph_id"], location=location, field="paragraph_id"
                )
                query = _non_empty_string(
                    record["query"], location=location, field="query"
                )
            except KeyError as exc:
                raise ValueError(f"{location}: missing {exc.args[0]}") from exc
            if query_id in origins:
                raise ValueError(
                    f"{location}: duplicate id {query_id} "
                    f"(first seen at {origins[query_id]})"
                )
            origins[query_id] = location
            if keep_ids is not None and query_id not in keep_ids:
                skipped[source] += 1
                continue
            queries.append(QueryRecord(query_id, paragraph_id, query, source))
    if hard_only and verbose:
        print(f"Hard queries found: {selected_count}", file=sys.stderr, flush=True)
        if keep_ids is not None:
            print(
                f"Hard queries with matching LLM embeddings: {len(queries)}",
                file=sys.stderr,
                flush=True,
            )
    return queries, dict(skipped)


def rank_expected_paragraphs(
    query_matrix: np.ndarray,
    expected_positions: np.ndarray,
    corpus_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-based ranks and scores of each query's expected corpus paragraph.

    Both matrices must be L2-normalized, so the dot product is the cosine similarity.
    Ties are broken by corpus order, matching a stable descending sort.
    """
    ranks = np.empty(query_matrix.shape[0], dtype=np.int64)
    scores = np.empty(query_matrix.shape[0], dtype=np.float64)
    for start in range(0, query_matrix.shape[0], QUERY_BATCH_SIZE):
        block = query_matrix[start : start + QUERY_BATCH_SIZE]
        # macOS Accelerate BLAS leaves stray FP flags set, which numpy reports as
        # spurious divide/overflow warnings; the products themselves are finite.
        with np.errstate(all="ignore"):
            similarities = block @ corpus_matrix.T
        for offset in range(block.shape[0]):
            row = similarities[offset]
            position = int(expected_positions[start + offset])
            score = float(row[position])
            better = int(np.count_nonzero(row > score))
            ties_before = int(np.count_nonzero(row[:position] == score))
            ranks[start + offset] = better + ties_before + 1
            scores[start + offset] = score
    return ranks, scores


def measure_baseline(
    query_paths: Sequence[Path] = DEFAULT_QUERY_PATHS,
    llm_embedding_paths: Sequence[Path] = DEFAULT_LLM_EMBEDDING_PATHS,
    corpus_embedding_paths: Sequence[Path] = DEFAULT_CORPUS_EMBEDDING_PATHS,
    *,
    hard_only: bool = False,
    verbose: bool = False,
) -> tuple[list[RankingResult], dict[str, int], int]:
    """Rank each query's own corpus paragraph by cosine similarity.

    Returns the rankings, the per-source count of queries skipped for want of an LLM
    embedding, and the corpus size.
    """
    llm_index = load_embedding_index(
        llm_embedding_paths, description="LLM embeddings", verbose=verbose
    )
    queries, skipped = load_queries(
        query_paths,
        keep_ids=set(llm_index.positions),
        hard_only=hard_only,
        verbose=verbose,
    )
    if not queries:
        if hard_only:
            hard_query_count = sum(skipped.values())
            if hard_query_count:
                raise ValueError(
                    f"found {hard_query_count} queries containing '_hard', but none "
                    "has a matching LLM text embedding; generate their LLM text and "
                    "embeddings before measuring the baseline"
                )
            raise ValueError("no query id containing '_hard' was found")
        raise ValueError("no query has a matching LLM text embedding")
    corpus_index = load_embedding_index(
        corpus_embedding_paths, description="corpus embeddings", verbose=verbose
    )

    if (
        llm_index.model is not None
        and corpus_index.model is not None
        and llm_index.model != corpus_index.model
    ):
        raise ValueError(
            f"embedding model mismatch: LLM uses {llm_index.model!r}, "
            f"corpus uses {corpus_index.model!r}"
        )
    if llm_index.dimensions != corpus_index.dimensions:
        raise ValueError(
            f"embedding dimension mismatch: LLM has {llm_index.dimensions}, "
            f"corpus has {corpus_index.dimensions}"
        )

    # Validate every query up front so problems surface before the expensive scoring.
    missing_paragraphs: list[str] = []
    mismatched: list[str] = []
    for query in queries:
        if query.paragraph_id not in corpus_index.positions:
            missing_paragraphs.append(f"{query.query_id} -> {query.paragraph_id}")
        llm_paragraph_id = llm_index.paragraph_ids[llm_index.positions[query.query_id]]
        if llm_paragraph_id is not None and llm_paragraph_id != query.paragraph_id:
            mismatched.append(
                f"{query.query_id}: query says {query.paragraph_id!r}, "
                f"LLM embedding says {llm_paragraph_id!r}"
            )
    if missing_paragraphs:
        raise ValueError(
            f"{len(missing_paragraphs)} queries target paragraphs missing from the "
            f"corpus embeddings, e.g. {missing_paragraphs[:5]}"
        )
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} queries disagree with their LLM embedding's "
            f"paragraph_id, e.g. {mismatched[:5]}"
        )

    query_matrix = llm_index.matrix[
        [llm_index.positions[query.query_id] for query in queries]
    ]
    expected_positions = np.fromiter(
        (corpus_index.positions[query.paragraph_id] for query in queries),
        dtype=np.int64,
        count=len(queries),
    )
    if verbose:
        print(
            f"Scoring {len(queries)} queries against {len(corpus_index)} paragraphs",
            file=sys.stderr,
            flush=True,
        )
    ranks, scores = rank_expected_paragraphs(
        query_matrix, expected_positions, corpus_index.matrix
    )

    rankings = [
        RankingResult(
            query_id=query.query_id,
            paragraph_id=query.paragraph_id,
            query=query.query,
            source=query.source,
            rank=int(rank),
            cosine_similarity=float(score),
        )
        for query, rank, score in zip(queries, ranks, scores)
    ]
    return rankings, skipped, len(corpus_index)


def summarize_rankings(rankings: Sequence[RankingResult]) -> dict[str, Any]:
    """Calculate standard aggregate retrieval metrics."""
    if not rankings:
        raise ValueError("cannot summarize an empty rankings list")
    ranks = [result.rank for result in rankings]
    query_count = len(ranks)
    return {
        "query_count": query_count,
        "mean_rank": statistics.fmean(ranks),
        "median_rank": statistics.median(ranks),
        "mean_reciprocal_rank": statistics.fmean(1.0 / rank for rank in ranks),
        "recall_at_1": sum(rank <= 1 for rank in ranks) / query_count,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / query_count,
        "recall_at_10": sum(rank <= 10 for rank in ranks) / query_count,
    }


def summarize_by_source(
    rankings: Sequence[RankingResult],
) -> dict[str, dict[str, Any]]:
    """Break the aggregate metrics down per query file."""
    grouped: dict[str, list[RankingResult]] = defaultdict(list)
    for result in rankings:
        grouped[result.source].append(result)
    return {source: summarize_rankings(group) for source, group in sorted(grouped.items())}


def summarize_ranks_worse_than(
    rankings: Sequence[RankingResult], threshold: int
) -> dict[str, Any]:
    """Count queries above ``threshold`` and group them by query-id suffix."""
    worse_rankings = [result for result in rankings if result.rank > threshold]
    worse_by_suffix = {
        suffix: sum(result.query_id.endswith(suffix) for result in worse_rankings)
        for suffix in QUERY_ID_SUFFIXES
    }
    total_by_suffix = {
        suffix: sum(result.query_id.endswith(suffix) for result in rankings)
        for suffix in QUERY_ID_SUFFIXES
    }
    return {
        "threshold": threshold,
        "query_count": len(worse_rankings),
        "query_count_by_id_suffix": worse_by_suffix,
        "total_query_count_by_id_suffix": total_by_suffix,
        "percentage_by_id_suffix": {
            suffix: (
                100.0 * worse_by_suffix[suffix] / total
                if total
                else 0.0
            )
            for suffix, total in total_by_suffix.items()
        },
    }


def build_report(
    rankings: Sequence[RankingResult],
    *,
    corpus_size: int,
    skipped: dict[str, int],
) -> dict[str, Any]:
    """Build the serializable detailed baseline report."""
    return {
        "corpus_size": corpus_size,
        "skipped_queries_without_llm_embedding": skipped,
        "metrics": summarize_rankings(rankings),
        "metrics_by_source": summarize_by_source(rankings),
        "ranks_worse_than_100": summarize_ranks_worse_than(
            rankings, RANK_FAILURE_THRESHOLD
        ),
        "rankings": [asdict(result) for result in rankings],
    }


def build_hard_queries(
    rankings: Sequence[RankingResult],
) -> list[dict[str, Any]]:
    """Return query text, id, and rank for results ranked worse than 100."""
    return [
        {"id": result.query_id, "query": result.query, "rank": result.rank}
        for result in rankings
        if result.rank > RANK_FAILURE_THRESHOLD
    ]


def build_easy_queries(
    rankings: Sequence[RankingResult],
) -> list[dict[str, Any]]:
    """Return query text, id, and rank for results ranked first."""
    return [
        {"id": result.query_id, "query": result.query, "rank": result.rank}
        for result in rankings
        if result.rank == 1
    ]


def write_json(path: Path, data: Any) -> None:
    """Atomically write JSON data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(
        f"{label}: n={metrics['query_count']} "
        f"mean={metrics['mean_rank']:.2f} "
        f"median={metrics['median_rank']:.1f} "
        f"MRR={metrics['mean_reciprocal_rank']:.4f} "
        f"R@1={metrics['recall_at_1']:.2%} "
        f"R@5={metrics['recall_at_5']:.2%} "
        f"R@10={metrics['recall_at_10']:.2%}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, nargs="+", default=DEFAULT_QUERY_PATHS)
    parser.add_argument(
        "--llm-embeddings", type=Path, nargs="+", default=DEFAULT_LLM_EMBEDDING_PATHS
    )
    parser.add_argument(
        "--corpus-embeddings",
        type=Path,
        nargs="+",
        default=DEFAULT_CORPUS_EMBEDDING_PATHS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a detailed JSON report with every query rank.",
    )
    parser.add_argument(
        "--hard-only",
        action="store_true",
        help="Only load queries whose id contains '_hard'.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages while the JSONL files load.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rankings, skipped, corpus_size = measure_baseline(
        args.queries,
        args.llm_embeddings,
        args.corpus_embeddings,
        hard_only=args.hard_only,
        verbose=not args.quiet,
    )
    metrics = summarize_rankings(rankings)

    if args.hard_only:
        print(f"Hard queries evaluated: {metrics['query_count']}")
    print(f"Evaluated {metrics['query_count']} queries against {corpus_size} paragraphs")
    if skipped:
        total_skipped = sum(skipped.values())
        detail = ", ".join(f"{source}: {count}" for source, count in sorted(skipped.items()))
        print(f"Skipped {total_skipped} queries without an LLM embedding ({detail})")
    print(f"Mean rank: {metrics['mean_rank']:.3f}")
    print(f"Median rank: {metrics['median_rank']:.3f}")
    print(f"MRR: {metrics['mean_reciprocal_rank']:.4f}")
    print(f"Recall@1: {metrics['recall_at_1']:.2%}")
    print(f"Recall@5: {metrics['recall_at_5']:.2%}")
    print(f"Recall@10: {metrics['recall_at_10']:.2%}")
    print()
    worse_summary = summarize_ranks_worse_than(rankings, RANK_FAILURE_THRESHOLD)
    print(
        f"Queries with rank worse than {RANK_FAILURE_THRESHOLD}: "
        f"{worse_summary['query_count']}"
    )
    for suffix, count in worse_summary["query_count_by_id_suffix"].items():
        total = worse_summary["total_query_count_by_id_suffix"][suffix]
        percentage = worse_summary["percentage_by_id_suffix"][suffix]
        print(
            f"  IDs ending with {suffix}: {count} / {total} "
            f"({percentage:.2f}%)"
        )
    print()
    print("By source:")
    for source, source_metrics in summarize_by_source(rankings).items():
        _print_metrics(f"  {source}", source_metrics)

    print("Worst 20 queries:")
    for result in sorted(rankings, key=lambda item: item.rank, reverse=True)[:20]:
        print(f"  Rank {result.rank} [{result.source}] {result.query[:100]}")

    write_json(HARD_QUERIES_PATH, build_hard_queries(rankings))
    print(f"Hard queries: {HARD_QUERIES_PATH}")
    write_json(EASY_QUERIES_PATH, build_easy_queries(rankings))
    print(f"Easy queries: {EASY_QUERIES_PATH}")

    if args.output is not None:
        write_json(
            args.output,
            build_report(rankings, corpus_size=corpus_size, skipped=skipped),
        )
        print(f"Detailed report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
