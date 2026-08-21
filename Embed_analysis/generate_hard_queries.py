#!/usr/bin/env python3
"""Generate hard retrieval queries for not-yet-annotated corpus paragraphs.

The generation instructions are loaded from
``Prompt for Generating Hard Retrieval Queries.md``. By default, one query is
generated per paragraph; use ``-n`` to request several distinct queries.

Run with:

    python generate_hard_queries.py
    python generate_hard_queries.py --limit 10
    python generate_hard_queries.py --limit 10 -n 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

from annotation_data import QUESTIONS_PER_PARAGRAPH, annotation_counts


HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = HERE / "corpus_initial.jsonl"
DEFAULT_TRAINING_DATA_PATH = HERE / "queries_manual.jsonl"
DEFAULT_OUTPUT_PATH = HERE / "queries_generated.jsonl"
DEFAULT_PROMPT_PATH = HERE / "Prompt for Generating Hard Retrieval Queries.md"
DEFAULT_MODEL = "gpt-5.6-terra"
QUERY_ID_PATTERN = re.compile(r"^(?P<paragraph_id>.+)_hard(?P<number>[1-9]\d*)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects from ``path`` and report malformed rows clearly."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def read_prompt(path: Path) -> str:
    """Load and minimally validate the hard-query prompt template."""
    prompt = path.read_text(encoding="utf-8")
    missing = [token for token in ("{{TEXT}}", "{{N}}") if token not in prompt]
    if missing:
        raise ValueError(
            f"Prompt template {path} is missing placeholder(s): {', '.join(missing)}"
        )
    return prompt


def render_prompt(prompt: str, text: str, number_of_queries: int) -> str:
    """Substitute the two placeholders used by the Markdown prompt."""
    # Replace N first so placeholder-like text inside the passage stays verbatim.
    return prompt.replace("{{N}}", str(number_of_queries)).replace(
        "{{TEXT}}", text
    )


def available_records(
    records: list[dict[str, Any]], training_data_path: Path
) -> list[dict[str, Any]]:
    """Return records that have fewer than the required manual questions."""
    counts = annotation_counts(training_data_path)
    return [
        record
        for record in records
        if counts[record["id"]] < QUESTIONS_PER_PARAGRAPH
    ]


def generated_query_numbers(path: Path) -> dict[str, set[int]]:
    """Return existing ``_hardN`` query numbers grouped by paragraph id."""
    generated: dict[str, set[int]] = {}
    if not path.exists():
        return generated

    for line_number, record in enumerate(read_jsonl(path), start=1):
        record_id = record.get("id")
        if not isinstance(record_id, str):
            continue
        match = QUERY_ID_PATTERN.fullmatch(record_id)
        if match is None:
            continue

        paragraph_id = record.get("paragraph_id")
        id_paragraph = match.group("paragraph_id")
        if paragraph_id != id_paragraph:
            raise ValueError(
                f"Inconsistent hard-query id in output record {path}:{line_number}"
            )
        generated.setdefault(id_paragraph, set()).add(int(match.group("number")))
    return generated


def parse_query_response(response: Any, number_of_queries: int) -> list[str]:
    """Parse and validate the structured query array returned by the model."""
    status = getattr(response, "status", "completed")
    if status != "completed":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        suffix = f" ({reason})" if reason else ""
        raise ValueError(f"OpenAI response status was {status!r}{suffix}")

    output_text = getattr(response, "output_text", "")
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("OpenAI returned no output text")
    try:
        result = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI returned malformed JSON: {exc.msg}") from exc

    if not isinstance(result, dict) or not isinstance(result.get("queries"), list):
        raise ValueError("OpenAI response must contain a 'queries' array")
    queries = result["queries"]
    if len(queries) != number_of_queries:
        raise ValueError(
            f"OpenAI returned {len(queries)} queries; expected {number_of_queries}"
        )
    if any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("OpenAI returned a non-string or empty query")

    cleaned = [query.strip() for query in queries]
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("OpenAI returned duplicate queries")
    return cleaned


def generate_queries(
    client: OpenAI,
    text: str,
    *,
    model: str,
    prompt: str,
    number_of_queries: int,
) -> list[str]:
    """Generate the requested number of hard queries for one passage."""
    response = client.responses.create(
        model=model,
        input=render_prompt(prompt, text, number_of_queries),
        text={
            "format": {
                "type": "json_schema",
                "name": "hard_retrieval_queries",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": number_of_queries,
                            "maxItems": number_of_queries,
                        }
                    },
                    "required": ["queries"],
                    "additionalProperties": False,
                },
            }
        },
    )
    return parse_query_response(response, number_of_queries)


def append_questions(
    path: Path,
    paragraph_id: str,
    numbered_queries: list[tuple[int, str]],
) -> None:
    """Append UTF-8 JSONL hard-query records for one paragraph."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for query_number, query in numbered_queries:
            record = {
                "id": f"{paragraph_id}_hard{query_number}",
                "paragraph_id": paragraph_id,
                "query": query,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def generate_available_questions(
    records: list[dict[str, Any]],
    *,
    training_data_path: Path,
    output_path: Path,
    number_of_queries: int,
    query_generator: Callable[[str, int], list[str]],
    limit: int | None = None,
) -> int:
    """Generate missing hard queries, returning the number of rows written."""
    generated = generated_query_numbers(output_path)
    required_numbers = set(range(1, number_of_queries + 1))
    pending = [
        record
        for record in available_records(records, training_data_path)
        if not required_numbers.issubset(generated.get(record["id"], set()))
    ]
    random.shuffle(pending)
    if limit is not None:
        pending = pending[:limit]

    total = len(pending)
    written = 0
    for number, record in enumerate(pending, start=1):
        paragraph_id = record.get("id")
        text = record.get("text")
        if not isinstance(paragraph_id, str) or not paragraph_id.strip():
            raise ValueError(f"Corpus record has no non-empty id: {record}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Corpus record {paragraph_id!r} has no non-empty text")

        missing_numbers = sorted(
            required_numbers - generated.get(paragraph_id, set())
        )
        queries = query_generator(text, len(missing_numbers))
        if len(queries) != len(missing_numbers):
            raise ValueError(
                f"Query generator returned {len(queries)} queries; "
                f"expected {len(missing_numbers)}"
            )
        numbered_queries = list(zip(missing_numbers, queries))
        append_questions(output_path, paragraph_id, numbered_queries)
        written += len(numbered_queries)
        print(
            f"[{number}/{total}] wrote {len(numbered_queries)} hard "
            f"query/queries for {paragraph_id}"
        )

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument(
        "--training-data", type=Path, default=DEFAULT_TRAINING_DATA_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "-n",
        "--number-of-queries",
        type=int,
        default=1,
        help="Number of distinct hard queries to generate per paragraph (default: 1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many new paragraphs (useful for a test run).",
    )
    args = parser.parse_args()
    if args.number_of_queries < 1:
        parser.error("--number-of-queries must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    load_dotenv()
    records = read_jsonl(args.corpus)
    prompt = read_prompt(args.prompt)
    client = OpenAI()
    written = generate_available_questions(
        records,
        training_data_path=args.training_data,
        output_path=args.output,
        number_of_queries=args.number_of_queries,
        query_generator=lambda text, count: generate_queries(
            client,
            text,
            model=args.model,
            prompt=prompt,
            number_of_queries=count,
        ),
        limit=args.limit,
    )
    print(f"Done. Wrote {written} hard queries to {args.output}")


if __name__ == "__main__":
    main()
