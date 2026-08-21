#!/usr/bin/env python3
"""Add deterministic and model-classified binary features to query records."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# Run with: python3 analysis/annotate_queries.py
# or python3 analysis/annotate_queries.py --force-follow-template

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "all_queries.json"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_BATCH_SIZE = 50

NUMBER_OF_WORDS_RE = re.compile(r"\b\d+\s+words\b", re.IGNORECASE)
LENGTH_DESCRIPTION_RE = re.compile(r"\b(?:short|long)\s+text\b", re.IGNORECASE)


def query_about_length(query: str) -> int:
    """Return 1 when *query* describes the requested text length."""
    if NUMBER_OF_WORDS_RE.search(query):
        return 1

    return int(
        len(query.split()) < 5 and LENGTH_DESCRIPTION_RE.search(query) is not None
    )


# Add the four forthcoming feature annotators here as their rules are defined.
FEATURE_ANNOTATORS: dict[str, Callable[[str], int]] = {
    "query_about_length": query_about_length,
}

FOLLOW_TEMPLATE_INSTRUCTIONS = """\
Classify whether each query can reasonably be associated with this writing-query
template, even when the match is approximate rather than perfect:

    [LENGTH/DENSITY] [REGISTER] [DOMAIN] TEXT-TYPE
        optionally followed by one or more of:
        - in a ___ style/tone/register
        - with ___
        - a comma plus an -ing rhetorical-action clause
        - about / focused on a broad topic

Interpret the slots broadly:
- LENGTH/DENSITY (optional) includes short, long, concise, detailed, a word count, etc.
- REGISTER (optional)includes formal, academic, casual, technical, poetic, etc.
- DOMAIN (optional) is a subject area or broad topic.
- Modifiers may appear before or after the text type and do not need to use the
  template's exact wording.

Return 1 when the query mainly describes a requested written artifact using a text
type and one or more compatible slots/modifiers. Be generous about word order,
missing optional slots, and slightly imperfect grammar. Return 0 for a bare factual
question, an unrelated task, or a query that describe the requested text with good details of its content.
For example these queries should receive 0:
-"text containing the answer for : what kind of oil do hibachi chefs use" (because the topic is too specific and not broad)
-"short text answering directly or indirectly the question : what does condenser mean" (same the content of the text is clearly specified)
- "story about unrest in Lleida after a court said a museum must return art bought decades ago from nuns" (because the content of the text is clearly specified)

Classify every supplied query independently. Return each input index exactly once.
"""

FOLLOW_TEMPLATE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "follow_template_annotations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "follow_template": {"type": "integer", "enum": [0, 1]},
                    },
                    "required": ["index", "follow_template"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}

QUERY_TYPE_INSTRUCTIONS = """\
Classify each writing query into exactly one query type. The match may be
approximate rather than perfect.

1 = Attributive: the query specifies what kind of text is wanted without going
deeply into the text's subject matter. This includes register, genre, tone,
broad domain or topic, audience, or purpose, often expressed through stacked
pre-modifiers (for example, encyclopedic, journalistic, or dictionary-style),
phrases such as "in ... style", "with ...", "about ...", or "focused on ...",
or participles such as explaining, describing, or summarizing. A query whose
main distinguishing feature is detailed, concrete topic content--such as a
specific question to answer, facts to include, or an event to recount--is not
Attributive merely because it uses "about", "focused on", or similar wording.

2 = Structural/sequential: the query specifies how the text unfolds. This often
uses a relative clause such as "text that begins...", "text where...", or
"passage which opens..." and gives an ordered discourse skeleton: opens with X,
then Y, followed by Z, and ends with W. The stages are primarily argumentative
or rhetorical functions--such as a claim, qualification, correction, contrast,
concession, reversal, or example--rather than merely a list of topics.

0 = Neither: the query does not reasonably fit either definition.

Use 2 when a query contains a material structural/sequential skeleton even if it
also has attributive wording. Do not label a mere list of topics as structural.
Be generous about wording, incomplete sequences, and slightly imperfect grammar.
Classify every supplied query independently and return each input index exactly
once.
"""

QUERY_TYPE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "query_type_annotations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "query_type": {"type": "integer", "enum": [0, 1, 2]},
                    },
                    "required": ["index", "query_type"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}

SUBJECT_MATTER_UNSPECIFIED_INSTRUCTIONS = """\
Classify whether each writing query leaves the main subject matter of the
requested text unspecified.

Return subject_matter_unspecified = 1 when the query says how the text should be
written but does not say what it should be about. Length, tone, register, style,
format, text type, audience, and rhetorical structure do not by themselves
specify subject matter. For example, "a concise formal explanation", "an AP-style
news report", and "a passage that opens with a claim and ends with a concession"
all receive 1.

Return subject_matter_unspecified = 0 when the query names either:
- a concrete topic, entity, event, question, or issue the text should address; or
- a recognizable subject/domain or purpose that constrains its content, such as
  mythology, science, history, linguistics, politics, sports, religion,
  promotional/commercial copy, or product marketing.

The subject may be expressed by an "about" phrase, a requested question or
answer, a domain adjective, or any other wording. It need not be detailed. For
example, "a mythological explanation", "a science-news article", "promotional
copy for a new camera", and "text answering what is a conifer" all receive 0.

Focus on whether the content is identified, not whether the query is otherwise
complete or well written. Classify every supplied query independently. Return
each input index exactly once.
"""

SUBJECT_MATTER_UNSPECIFIED_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "subject_matter_unspecified_annotations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "subject_matter_unspecified": {
                            "type": "integer",
                            "enum": [0, 1],
                        },
                    },
                    "required": ["index", "subject_matter_unspecified"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSON array of query objects."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array")

    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be an object")
        if not isinstance(record.get("query"), str):
            raise ValueError(f"{path}: record {index} must have a string 'query' field")

    return data


def annotate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copies of *records* containing deterministic features."""
    return [
        {
            **record,
            **{
                feature_name: annotate(record["query"])
                for feature_name, annotate in FEATURE_ANNOTATORS.items()
            },
        }
        for record in records
    ]


def classify_follow_template(
    client: Any,
    queries: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
) -> list[int]:
    """Classify one batch of queries using an OpenAI structured response."""
    indexed_queries = [
        {"index": index, "query": query} for index, query in enumerate(queries)
    ]
    response = client.responses.create(
        model=model,
        store=False,
        input=[
            {"role": "system", "content": FOLLOW_TEMPLATE_INSTRUCTIONS},
            {
                "role": "user",
                "content": json.dumps(indexed_queries, ensure_ascii=False),
            },
        ],
        text={"format": FOLLOW_TEMPLATE_FORMAT},
    )

    if getattr(response, "status", "completed") != "completed":
        raise ValueError(f"OpenAI response status was {response.status!r}")

    try:
        payload = json.loads(response.output_text)
        results = payload["results"]
    except (AttributeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("OpenAI returned an invalid follow_template result") from exc

    if not isinstance(results, list) or len(results) != len(queries):
        raise ValueError(
            "OpenAI returned an unexpected number of follow_template results"
        )

    labels: list[int | None] = [None] * len(queries)
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("OpenAI returned a non-object classification")
        index = result.get("index")
        label = result.get("follow_template")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(queries)
            or labels[index] is not None
            or label not in (0, 1)
            or isinstance(label, bool)
        ):
            raise ValueError("OpenAI returned an invalid or duplicate classification")
        labels[index] = label

    if any(label is None for label in labels):
        raise ValueError("OpenAI omitted a follow_template classification")
    return [int(label) for label in labels]


def classify_with_retries(
    client: Any,
    queries: Sequence[str],
    *,
    model: str,
    max_attempts: int = 3,
) -> list[int]:
    """Retry transient API and response-validation failures for one batch."""
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return classify_follow_template(client, queries, model=model)
        except Exception as exc:  # The SDK exposes several retryable error types.
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def classify_query_type(
    client: Any,
    queries: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
) -> list[int]:
    """Classify one batch as neither, attributive, or structural/sequential."""
    indexed_queries = [
        {"index": index, "query": query} for index, query in enumerate(queries)
    ]
    response = client.responses.create(
        model=model,
        store=False,
        input=[
            {"role": "system", "content": QUERY_TYPE_INSTRUCTIONS},
            {
                "role": "user",
                "content": json.dumps(indexed_queries, ensure_ascii=False),
            },
        ],
        text={"format": QUERY_TYPE_FORMAT},
    )

    if getattr(response, "status", "completed") != "completed":
        raise ValueError(f"OpenAI response status was {response.status!r}")

    try:
        payload = json.loads(response.output_text)
        results = payload["results"]
    except (AttributeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("OpenAI returned an invalid query_type result") from exc

    if not isinstance(results, list) or len(results) != len(queries):
        raise ValueError("OpenAI returned an unexpected number of query_type results")

    labels: list[int | None] = [None] * len(queries)
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("OpenAI returned a non-object classification")
        index = result.get("index")
        label = result.get("query_type")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(queries)
            or labels[index] is not None
            or label not in (0, 1, 2)
            or isinstance(label, bool)
        ):
            raise ValueError("OpenAI returned an invalid or duplicate classification")
        labels[index] = label

    if any(label is None for label in labels):
        raise ValueError("OpenAI omitted a query_type classification")
    return [int(label) for label in labels]


def classify_query_type_with_retries(
    client: Any,
    queries: Sequence[str],
    *,
    model: str,
    max_attempts: int = 3,
) -> list[int]:
    """Retry transient API and response-validation failures for one batch."""
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return classify_query_type(client, queries, model=model)
        except Exception as exc:  # The SDK exposes several retryable error types.
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def classify_subject_matter_unspecified(
    client: Any,
    queries: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
) -> list[int]:
    """Classify whether each query omits the requested text's subject matter."""
    indexed_queries = [
        {"index": index, "query": query} for index, query in enumerate(queries)
    ]
    response = client.responses.create(
        model=model,
        store=False,
        input=[
            {
                "role": "system",
                "content": SUBJECT_MATTER_UNSPECIFIED_INSTRUCTIONS,
            },
            {
                "role": "user",
                "content": json.dumps(indexed_queries, ensure_ascii=False),
            },
        ],
        text={"format": SUBJECT_MATTER_UNSPECIFIED_FORMAT},
    )

    if getattr(response, "status", "completed") != "completed":
        raise ValueError(f"OpenAI response status was {response.status!r}")

    try:
        payload = json.loads(response.output_text)
        results = payload["results"]
    except (AttributeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            "OpenAI returned an invalid subject_matter_unspecified result"
        ) from exc

    if not isinstance(results, list) or len(results) != len(queries):
        raise ValueError(
            "OpenAI returned an unexpected number of "
            "subject_matter_unspecified results"
        )

    labels: list[int | None] = [None] * len(queries)
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("OpenAI returned a non-object classification")
        index = result.get("index")
        label = result.get("subject_matter_unspecified")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(queries)
            or labels[index] is not None
            or label not in (0, 1)
            or isinstance(label, bool)
        ):
            raise ValueError("OpenAI returned an invalid or duplicate classification")
        labels[index] = label

    if any(label is None for label in labels):
        raise ValueError("OpenAI omitted a subject_matter_unspecified classification")
    return [int(label) for label in labels]


def classify_subject_matter_unspecified_with_retries(
    client: Any,
    queries: Sequence[str],
    *,
    model: str,
    max_attempts: int = 3,
) -> list[int]:
    """Retry failures while classifying one subject-matter batch."""
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return classify_subject_matter_unspecified(client, queries, model=model)
        except Exception as exc:  # The SDK exposes several retryable error types.
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def annotate_follow_template(
    records: list[dict[str, Any]],
    client: Any,
    *,
    output: Path,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> int:
    """Annotate missing records in batches, checkpointing after every batch."""
    pending = [
        index
        for index, record in enumerate(records)
        if force or record.get("follow_template") not in (0, 1)
    ]

    completed = 0
    for start in range(0, len(pending), batch_size):
        indices = pending[start : start + batch_size]
        labels = classify_with_retries(
            client,
            [records[index]["query"] for index in indices],
            model=model,
        )
        for index, label in zip(indices, labels):
            records[index]["follow_template"] = label
        completed += len(indices)
        write_records(output, records)
        print(
            f"Classified {completed}/{len(pending)} pending queries",
            file=sys.stderr,
        )

    return completed


def annotate_query_type(
    records: list[dict[str, Any]],
    client: Any,
    *,
    output: Path,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> int:
    """Annotate missing query types in batches, checkpointing each batch."""
    pending = [
        index
        for index, record in enumerate(records)
        if force
        or record.get("query_type") not in (0, 1, 2)
        or isinstance(record.get("query_type"), bool)
    ]

    completed = 0
    for start in range(0, len(pending), batch_size):
        indices = pending[start : start + batch_size]
        labels = classify_query_type_with_retries(
            client,
            [records[index]["query"] for index in indices],
            model=model,
        )
        for index, label in zip(indices, labels):
            records[index]["query_type"] = label
        completed += len(indices)
        write_records(output, records)
        print(
            f"Classified query_type for {completed}/{len(pending)} pending queries",
            file=sys.stderr,
        )

    return completed


def annotate_subject_matter_unspecified(
    records: list[dict[str, Any]],
    client: Any,
    *,
    output: Path,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> int:
    """Annotate missing subject-matter labels, checkpointing each batch."""
    pending = [
        index
        for index, record in enumerate(records)
        if force
        or record.get("subject_matter_unspecified") not in (0, 1)
        or isinstance(record.get("subject_matter_unspecified"), bool)
    ]

    completed = 0
    for start in range(0, len(pending), batch_size):
        indices = pending[start : start + batch_size]
        labels = classify_subject_matter_unspecified_with_retries(
            client,
            [records[index]["query"] for index in indices],
            model=model,
        )
        for index, label in zip(indices, labels):
            records[index]["subject_matter_unspecified"] = label
        completed += len(indices)
        write_records(output, records)
        print(
            "Classified subject_matter_unspecified for "
            f"{completed}/{len(pending)} pending queries",
            file=sys.stderr,
        )

    return completed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"input JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output JSON file (default: update the input file)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model for model-classified features (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"queries per API request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--force-follow-template",
        action="store_true",
        help="reclassify records that already have a follow_template value",
    )
    parser.add_argument(
        "--force-query-type",
        action="store_true",
        help="reclassify records that already have a query_type value",
    )
    parser.add_argument(
        "--force-subject-matter-unspecified",
        action="store_true",
        help=(
            "reclassify records that already have a "
            "subject_matter_unspecified value"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or args.input

    if args.batch_size < 1:
        print("error: --batch-size must be at least 1", file=sys.stderr)
        return 2

    try:
        records = annotate_records(load_records(args.input))
        write_records(output, records)

        needs_follow_template = args.force_follow_template or any(
            record.get("follow_template") not in (0, 1) for record in records
        )
        needs_query_type = args.force_query_type or any(
            record.get("query_type") not in (0, 1, 2)
            or isinstance(record.get("query_type"), bool)
            for record in records
        )
        needs_subject_matter_unspecified = (
            args.force_subject_matter_unspecified
            or any(
                record.get("subject_matter_unspecified") not in (0, 1)
                or isinstance(record.get("subject_matter_unspecified"), bool)
                for record in records
            )
        )
        follow_template_classified = 0
        query_type_classified = 0
        subject_matter_unspecified_classified = 0
        if (
            needs_follow_template
            or needs_query_type
            or needs_subject_matter_unspecified
        ):
            try:
                from dotenv import load_dotenv
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI dependencies are missing; "
                    "run pip install -r requirements.txt"
                ) from exc

            load_dotenv(ROOT / ".env")
            client = OpenAI()
            if needs_follow_template:
                follow_template_classified = annotate_follow_template(
                    records,
                    client,
                    output=output,
                    model=args.model,
                    batch_size=args.batch_size,
                    force=args.force_follow_template,
                )
            if needs_query_type:
                query_type_classified = annotate_query_type(
                    records,
                    client,
                    output=output,
                    model=args.model,
                    batch_size=args.batch_size,
                    force=args.force_query_type,
                )
            if needs_subject_matter_unspecified:
                subject_matter_unspecified_classified = (
                    annotate_subject_matter_unspecified(
                        records,
                        client,
                        output=output,
                        model=args.model,
                        batch_size=args.batch_size,
                        force=args.force_subject_matter_unspecified,
                    )
                )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Annotated {len(records)} queries in {output}; "
        f"classified {follow_template_classified} follow_template and "
        f"{query_type_classified} query_type and "
        f"{subject_matter_unspecified_classified} "
        f"subject_matter_unspecified values with {args.model}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
