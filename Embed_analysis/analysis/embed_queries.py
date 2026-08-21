#!/usr/bin/env python3
"""
Embed every query in a JSON file using either a local SentenceTransformer or
OpenAI's text-embedding-3-small model.

Usage:
    python analysis/embed_queries.py analysis/hard_queries_embedded.json
Optional:
    python analysis/embed_queries.py analysis/hard_queries_embedded.json \
        --batch-size 64 \
        --normalize
    python analysis/embed_queries.py analysis/hard_queries_embedded.json \
        --model text-embedding-3-small

"""

import argparse
import json
from pathlib import Path
from typing import Any

SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"
OPENAI_MODEL = "text-embedding-3-small"
SUPPORTED_MODELS = (SENTENCE_TRANSFORMER_MODEL, OPENAI_MODEL)
INPUT_PATH = Path(__file__).resolve().parents[1] / "hard_queries.json"


def load_records(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate the input JSON records."""
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("The input JSON must contain a top-level array.")

    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a JSON object.")

        query = record.get("query")
        if not isinstance(query, str):
            raise ValueError(
                f"Record {index} has no valid string 'query' field."
            )

    return data


def embed_queries(
    records: list[dict[str, Any]],
    model: Any,
    batch_size: int,
    normalize: bool,
) -> None:
    """Generate embeddings and add them to the records in place."""
    queries = [record["query"] for record in records]

    embeddings = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    for record, embedding in zip(records, embeddings):
        # Convert NumPy float values to normal Python floats so JSON can
        # serialize them.
        record["embedding"] = embedding.tolist()


def embed_queries_openai(
    records: list[dict[str, Any]],
    client: Any,
    batch_size: int,
    normalize: bool,
) -> None:
    """Generate OpenAI embeddings and add them to the records in place."""
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        response = client.embeddings.create(
            model=OPENAI_MODEL,
            input=[record["query"] for record in batch],
            encoding_format="float",
        )

        if len(response.data) != len(batch):
            raise RuntimeError("OpenAI returned an unexpected embedding count.")

        embeddings: list[list[float] | None] = [None] * len(batch)
        for fallback_index, item in enumerate(response.data):
            index = getattr(item, "index", fallback_index)
            if not isinstance(index, int) or not 0 <= index < len(batch):
                raise RuntimeError(
                    f"OpenAI returned an invalid embedding index: {index}"
                )
            if embeddings[index] is not None:
                raise RuntimeError(
                    f"OpenAI returned duplicate embedding index: {index}"
                )

            embedding = [float(value) for value in item.embedding]
            if normalize:
                magnitude = sum(value * value for value in embedding) ** 0.5
                if magnitude == 0:
                    raise RuntimeError("OpenAI returned a zero-length embedding.")
                embedding = [value / magnitude for value in embedding]
            embeddings[index] = embedding

        if any(embedding is None for embedding in embeddings):
            raise RuntimeError("OpenAI response omitted an embedding.")

        for record, embedding in zip(batch, embeddings):
            record["embedding"] = embedding

        print(f"Embedded {min(start + len(batch), len(records)):,}/{len(records):,}")


def save_records(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write the enriched records to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False)

    print(f"Saved {len(records):,} embedded queries to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed query fields with a local or OpenAI embedding model."
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=SENTENCE_TRANSFORMER_MODEL,
        help=f"Embedding model to use. Default: {SENTENCE_TRANSFORMER_MODEL}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of queries encoded per batch. Default: 32.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings for cosine-similarity search.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device such as 'cuda', 'mps', or 'cpu'. Auto-detected by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    records = load_records(INPUT_PATH)

    if args.model == OPENAI_MODEL:
        if args.device is not None:
            raise ValueError("--device is only supported for SentenceTransformer.")
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        print(f"Embedding with OpenAI model: {OPENAI_MODEL}")
        embed_queries_openai(
            records=records,
            client=OpenAI(),
            batch_size=args.batch_size,
            normalize=args.normalize,
        )
    else:
        from sentence_transformers import SentenceTransformer

        print(f"Loading model: {args.model}")
        model = SentenceTransformer(args.model, device=args.device)
        embed_queries(
            records=records,
            model=model,
            batch_size=args.batch_size,
            normalize=args.normalize,
        )
    save_records(records, args.output)


if __name__ == "__main__":
    main()
