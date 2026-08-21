#!/usr/bin/env python3
"""Generate and store an OpenAI embedding for every text in ``corpus.jsonl``.

The output is a resumable JSONL file. A saved embedding is reused only when its
corpus id, model, and exact-text SHA-256 hash still match the current corpus.

Examples:
    python3 baseline/embed_corpus.py
    python3 baseline/embed_corpus.py --batch-size 32
    python3 baseline/embed_corpus.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_PRICE_PER_MILLION_TOKENS = 0.02


@dataclass(frozen=True)
class CorpusRecord:
    record_id: str
    text: str
    text_sha256: str


@dataclass
class EmbeddingRunStats:
    input_tokens: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        return self.input_tokens * EMBEDDING_PRICE_PER_MILLION_TOKENS / 1_000_000


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def load_corpus(path: Path) -> list[CorpusRecord]:
    corpus: list[CorpusRecord] = []
    seen_ids: set[str] = set()
    for line_number, record in enumerate(read_jsonl(path), start=1):
        try:
            record_id = record["id"]
            text = record["text"]
        except KeyError as exc:
            raise ValueError(f"{path}:{line_number}: missing {exc.args[0]}") from exc
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
        if record_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {record_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number}: text must be a non-empty string")
        seen_ids.add(record_id)
        corpus.append(CorpusRecord(record_id, text, text_sha256(text)))
    if not corpus:
        raise ValueError(f"{path}: corpus contains no records")
    return corpus


def validate_embedding(embedding: Any, *, location: str) -> list[float]:
    if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"{location}: embedding must contain {EMBEDDING_DIMENSIONS} values"
        )
    try:
        vector = [float(value) for value in embedding]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: embedding contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{location}: embedding contains a non-finite value")
    return vector


def load_existing_embeddings(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    embeddings: dict[str, dict[str, Any]] = {}
    for line_number, record in enumerate(read_jsonl(path), start=1):
        location = f"{path}:{line_number}"
        try:
            record_id = record["id"]
            model = record["embedding_model"]
            digest = record["text_sha256"]
            embedding = record["embedding"]
        except KeyError as exc:
            raise ValueError(f"{location}: missing {exc.args[0]}") from exc
        if model != EMBEDDING_MODEL:
            continue
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{location}: id must be a non-empty string")
        if record_id in embeddings:
            raise ValueError(f"{location}: duplicate id {record_id}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{location}: invalid text_sha256")
        embeddings[record_id] = {
            "id": record_id,
            "embedding_model": EMBEDDING_MODEL,
            "text_sha256": digest,
            "embedding": validate_embedding(embedding, location=location),
        }
    return embeddings


def request_embeddings(
    client: Any,
    texts: list[str],
    *,
    stats: EmbeddingRunStats | None = None,
) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if stats is not None and isinstance(prompt_tokens, int):
        stats.input_tokens += prompt_tokens

    if len(response.data) != len(texts):
        raise RuntimeError("OpenAI returned an unexpected embedding count")

    vectors: list[list[float] | None] = [None] * len(texts)
    for fallback_index, item in enumerate(response.data):
        index = getattr(item, "index", fallback_index)
        if not isinstance(index, int) or not 0 <= index < len(texts):
            raise RuntimeError(f"OpenAI returned an invalid embedding index: {index}")
        if vectors[index] is not None:
            raise RuntimeError(f"OpenAI returned duplicate embedding index: {index}")
        vectors[index] = validate_embedding(
            item.embedding, location=f"OpenAI response item {index}"
        )
    if any(vector is None for vector in vectors):
        raise RuntimeError("OpenAI response omitted an embedding")
    return [vector for vector in vectors if vector is not None]


def write_embeddings(
    path: Path,
    corpus: list[CorpusRecord],
    embeddings: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for corpus_record in corpus:
                embedding = embeddings.get(corpus_record.record_id)
                if embedding is not None:
                    handle.write(json.dumps(embedding, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_corpus_embeddings(
    corpus_path: Path,
    output_path: Path,
    *,
    batch_size: int,
    limit: int | None = None,
    included_record_ids: set[str] | None = None,
    force: bool = False,
    client: Any | None = None,
    stats: EmbeddingRunStats | None = None,
) -> tuple[int, int]:
    """Return ``(selected corpus rows, newly generated embeddings)``."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    corpus = load_corpus(corpus_path)
    if included_record_ids is not None:
        corpus = [
            record
            for record in corpus
            if record.record_id in included_record_ids
        ]
        if not corpus:
            raise ValueError(
                f"{corpus_path}: none of its corpus IDs occur in the paragraph-ID "
                "filter; output was left unchanged"
            )
    existing = {} if force else load_existing_embeddings(output_path)
    embeddings = {
        record.record_id: existing[record.record_id]
        for record in corpus
        if record.record_id in existing
        and existing[record.record_id]["text_sha256"] == record.text_sha256
    }
    missing = [record for record in corpus if record.record_id not in embeddings]
    if limit is not None:
        missing = missing[:limit]

    if missing and client is None:
        from openai import OpenAI

        client = OpenAI()

    generated = 0
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = request_embeddings(
            client,
            [record.text for record in batch],
            stats=stats,
        )
        for record, vector in zip(batch, vectors):
            embeddings[record.record_id] = {
                "id": record.record_id,
                "embedding_model": EMBEDDING_MODEL,
                "text_sha256": record.text_sha256,
                "embedding": vector,
            }
        generated += len(batch)
        write_embeddings(output_path, corpus, embeddings)
        print(f"Embedded {generated}/{len(missing)} missing texts")

    # Canonicalize the file even when every row was already present, removing
    # stale records and preserving the current corpus order.
    write_embeddings(output_path, corpus, embeddings)
    return len(corpus), generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--force", action="store_true", help="regenerate every embedding"
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        pass
    else:
        load_dotenv()
    total, generated = generate_corpus_embeddings(
        args.corpus,
        args.output,
        batch_size=args.batch_size,
        force=args.force,
    )
    reused = total - generated
    print(
        f"Stored {total} {EMBEDDING_MODEL} embeddings in {args.output} "
        f"({generated} generated, {reused} reused)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
