#!/usr/bin/env python3
"""Generate LLM text for rows in training_questions.jsonl.

Examples:
    python generate_llm_text.py --limit 300
    python generate_llm_text.py --limit 400 --output generated_training_text.jsonl

The script is resumable: it reads the existing output JSONL file, skips rows
that have already been generated, and appends only new results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = HERE / "queries_manual.jsonl"
DEFAULT_OUTPUT_PATH = HERE / "llm.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

INPUT_PRICE_PER_MILLION = 0.75
CACHED_INPUT_PRICE_PER_MILLION = 0.075
OUTPUT_PRICE_PER_MILLION = 4.50
EMBEDDING_PRICE_PER_MILLION = 0.02


def count_words(text: str) -> int:
    return len(text.split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def get_query(record: dict[str, Any]) -> str:
    query = record.get("query", record.get("question"))
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"input row has no non-empty query/question: {record}")
    return query.strip()


def get_record_id(record: dict[str, Any]) -> str:
    record_id = record.get("paragraph_id", record.get("id"))
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError(f"input row has no non-empty paragraph_id/id: {record}")
    return record_id.strip()


def get_query_id(record: dict[str, Any]) -> str:
    query_id = record.get("id")
    if not isinstance(query_id, str) or not query_id.strip():
        raise ValueError(f"input row has no non-empty id: {record}")
    return query_id.strip()


def record_key(record_id: str, query: str) -> tuple[str, str]:
    return record_id, query


def load_processed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()

    processed: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                record_id = get_record_id(record)
                query = get_query(record)
            except (json.JSONDecodeError, ValueError) as exc:
                print(
                    f"Warning: ignoring malformed output row "
                    f"{path}:{line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            processed.add(record_key(record_id, query))
    return processed


def default_embeddings_path(output: Path) -> Path:
    return output.with_name(output.stem + "_embeddings.jsonl")


def load_embedded_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()

    embedded: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                query_id = get_query_id(record)
                model = record["embedding_model"]
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                print(
                    f"Warning: ignoring malformed embedding row "
                    f"{path}:{line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            embedded.add((query_id, model))
    return embedded


def generate_embedding(
    client: OpenAI,
    text: str,
    *,
    model: str,
) -> tuple[list[float], float | None]:
    response = client.embeddings.create(model=model, input=text)
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    cost = (
        prompt_tokens * EMBEDDING_PRICE_PER_MILLION / 1_000_000
        if prompt_tokens is not None
        else None
    )
    return response.data[0].embedding, cost


def embed_and_store(
    client: OpenAI,
    record: dict[str, Any],
    *,
    embeddings_path: Path,
    embedding_model: str,
) -> float | None:
    embedding, cost = generate_embedding(
        client, record["generated_text"], model=embedding_model
    )
    append_result(
        embeddings_path,
        {
            "id": record["id"],
            "paragraph_id": record["paragraph_id"],
            "embedding_model": embedding_model,
            "embedding": embedding,
        },
    )
    return cost


def calculate_cost(usage: Any) -> float | None:
    if usage is None:
        return None

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None or output_tokens is None:
        return None

    details = getattr(usage, "input_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) if details else 0
    uncached_tokens = input_tokens - cached_tokens

    return (
        uncached_tokens * INPUT_PRICE_PER_MILLION
        + cached_tokens * CACHED_INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000


def generate_text(
    client: OpenAI,
    query: str,
    *,
    model: str,
    max_words: int,
) -> tuple[str, float | None]:
    response = client.responses.create(
        model=model,
        instructions=(
            "Generate a text corresponding to the query. "
            "Return only the requested text. "
            "No intro, no explanation. "
            f"The text must be no more than {max_words} words."
        ),
        input=f"Generate a text corresponding to the following query: {query}",
    )
    return response.output_text.strip(), calculate_cost(response.usage)


def append_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        fh.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM text for unprocessed training queries."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSONL file. Default: {DEFAULT_INPUT_PATH.name}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL file. Default: {DEFAULT_OUTPUT_PATH.name}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Number of new, unprocessed queries to generate in this run.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"OpenAI embedding model to use. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--embeddings-output",
        type=Path,
        default=None,
        help=(
            "Output JSONL file for embeddings. "
            "Default: <output>_embeddings.jsonl next to the output file."
        ),
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=300,
        help="Maximum words requested from the model. Default: 300",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        print("--limit must be at least 1", file=sys.stderr)
        return 2
    if args.max_words < 1:
        print("--max-words must be at least 1", file=sys.stderr)
        return 2

    input_records = read_jsonl(args.input)
    processed = load_processed_keys(args.output) #this returns set of tuples of (record_id, query) ex of tuple: ('news-000661', 'AP ne...idency')

    #maybe it is better to return the id of the training question no ?
    #print("processed keys : ") 
    #print(processed)
    print()
    already_processed_count = sum(
        1
        for record in input_records
        if record_key(get_record_id(record), get_query(record)) in processed
    )
    unprocessed_count = len(input_records) - already_processed_count
    client = OpenAI()

    embeddings_path = args.embeddings_output or default_embeddings_path(args.output)
    embedded = load_embedded_keys(embeddings_path)

    generated_count = 0
    total_cost = 0.0

    # Backfill embeddings for texts generated in earlier runs.
    if args.output.exists():
        for record in read_jsonl(args.output):
            key = (get_query_id(record), args.embedding_model)
            if key in embedded:
                continue
            cost = embed_and_store(
                client,
                record,
                embeddings_path=embeddings_path,
                embedding_model=args.embedding_model,
            )
            if cost is not None:
                total_cost += cost
            embedded.add(key)
            print(f"backfilled embedding for {record['paragraph_id']}")

    for input_index, record in enumerate(input_records):
        record_id = get_record_id(record)
        query_id = get_query_id(record)
        query = get_query(record)
        key = record_key(record_id, query)

        if key in processed:
            continue
        text, cost = generate_text(
            client,
            query,
            model=args.model,
            max_words=args.max_words,
        )
        result = {
            "id": query_id,
            "paragraph_id": record_id,
            "query": query,
            "generated_text": text,
            "word_count": count_words(text),
            "model": args.model,
            "input_index": input_index,
        }
        if cost is not None:
            result["cost_usd"] = cost
            total_cost += cost

        append_result(args.output, result)
        processed.add(key)

        embedding_cost = embed_and_store(
            client,
            result,
            embeddings_path=embeddings_path,
            embedding_model=args.embedding_model,
        )
        if embedding_cost is not None:
            total_cost += embedding_cost
        embedded.add((query_id, args.embedding_model))

        generated_count += 1
        print(
            f"[{generated_count}/{args.limit}] generated {record_id} "
            f"({count_words(text)} words)"
        )

        if generated_count >= args.limit:
            break

    remaining_count = max(unprocessed_count - generated_count, 0)
    print(
        f"Done. Existing: {already_processed_count}. "
        f"Generated: {generated_count}. Remaining: {remaining_count}."
    )
    if total_cost:
        print(f"Estimated cost for this run: ${total_cost:.8f}")
    print(f"Output: {args.output}")
    print(f"Embeddings: {embeddings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# run it like : 
# python generate_llm_text.py --limit 30

# you can also run :
"""
python3 generate_llm_text.py \
   --input queries_manual.jsonl \
   --output llm.jsonl \
   --limit 500

python3 generate_llm_text.py \
   --input queries_msmarco.jsonl \
   --output llm_msmarco.jsonl \
   --limit 500

python3 generate_llm_text.py --input queries_generated.jsonl \
   --output llm.jsonl \
   --limit 600

"""
