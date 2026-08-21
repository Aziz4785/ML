#!/usr/bin/env python3
"""Generate one search query for each not-yet-annotated corpus paragraph.


Run with:

    python generate_queries1.py --prompt 1
    python generate_queries1.py --prompt 1 --limit 10
    python generate_queries1.py --prompt 2 --limit 10
    python generate_queries1.py --prompt 3 --limit 10
    python generate_queries1.py --prompt 5 --limit 200
    python generate_queries1.py --prompt 6 --limit 200
    python generate_queries1.py --prompt 7 --limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
import random
from openai import OpenAI

from annotation_data import QUESTIONS_PER_PARAGRAPH, annotation_counts

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = HERE / "corpus_initial.jsonl"
DEFAULT_TRAINING_DATA_PATH = HERE / "queries_manual.jsonl"
# Every prompt appends to this one file; the id suffix keeps them apart.
DEFAULT_OUTPUT_PATH = HERE / "queries_generated.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini"

PROMPT1 = """Imagine you have a search engine for text. You can write a query in plain English and it will retrieve the corresponding text. What single query can you write in that search engine so that it retrieves this text:

{text}

The query can be about the core meaning of the text or another dimension such as its format, style, distinctive features, or details. It could start with \"text about...\", \"text talking about...\", \"text where...\", etc."""

PROMPT2 = """Someone read the text below a while ago and now only vaguely remembers it. They're describing it out loud to a friend who has read everything on the internet, hoping the friend can identify it.

Write what they'd say. Rules:

Natural spoken sentences, not keywords.
Do not use technical or rare terms from the text — a person who forgot it wouldn't remember those words. Describe things in everyday language instead.
The text : 
{text}
 Dont end with a question. Just describe it as if you were telling a friend about it. """

PROMPT3 = """Someone is trying to find the text below, but may remember it with different levels of detail.

Give 5 different ways they might describe it, ordered from very vague to fairly specific. Each description should be a natural spoken sentence, not a list of keywords. Do not copy distinctive technical or rare terms from the text when an everyday description would work. Do not end the descriptions with a question.

The text:
{text}"""

PROMPT4 = """Read the text below and consider:
- What is this text talking about?
- What are its distinctive features?
- What is the "engine" of the passage—the central idea, tension, mechanism, or progression that drives it?
- What tone and emotions does it convey?
- What possible questions could this text answer?
- How is it structured?

Using that analysis, write a single natural-language search query that would retrieve this text. Capture its most identifying combination of subject, distinctive features, underlying engine, tone, and structure. Return only one query, not the analysis or a list of alternatives.

The text:
{text}"""


PROMPT5 = """Imagine you have a search engine for text. You can write a query in plain English and it will retrieve the corresponding text. What single query can you write in that search engine so that it retrieves this text:

{text}

The query shoud be about the format, style, distinctive features, or details. It could start with \"text about...\", \"text talking about...\", \"text where...\", etc."""

PROMPT6 = """Imagine you have a search engine for text. Write one natural-language query that would retrieve the text below based primarily on its style and genre, rather than its subject matter.

Identify the text's overall linguistic identity and express the most distinctive combination of:
- genre or recognizable text type, such as an encyclopedia article, news report, academic essay, personal letter, product description, dialogue, instruction manual, or biography;
- style, including formality, vocabulary, sentence complexity, objectivity or subjectivity, emotional character, humor or asides, information density, and technicality;
- register, such as formal, informal, academic, conversational, professional, technical, or journalistic;
- tone, such as neutral, enthusiastic, skeptical, reassuring, critical, humorous, respectful, or cautious;
- communicative purpose, such as informing, explaining, teaching, persuading, narrating, or advising.

Styles may be hybrid. Preserve important balances such as \"formal but friendly,\" \"academic but conversational,\" \"factual but narrative,\" \"technical but accessible,\" or \"authoritative but cautious.\" Describe the qualities demonstrated by the writing; do not merely copy conspicuous words from it. Mention formatting or evidence conventions, such as citations, dialogue, direct address, greetings, or sign-offs, only when they help identify the text type.

Return only one concise query. It should sound like something a person would type into a semantic text search engine, for example \"text written in an encyclopedic, factual narrative style\" or \"conversational academic writing with casual asides.\"

The text:
{text}"""

PROMPT7 = """Imagine you have a search engine for text. Write one concise natural-language query that would retrieve the text below based primarily on its discourse or rhetorical structure: the ordered progression of communicative moves across the text as a whole.

Identify what each larger segment does and how the text moves from one segment to the next. Relevant rhetorical moves may include:
- opening or introducing a situation;
- making a claim, giving evidence, or drawing an inference;
- hedging or cautiously reformulating a statement;
- conceding a limitation or opposing point;
- correcting, clarifying, or replacing an earlier statement;
- promising a repair or corrective action;
- reversing an initial position;
- qualifying or narrowing a broad claim;
- introducing an example or aside, then resuming the main point;
- presenting an event, reaction, consequence, and conclusion;
- alternating quoted speech with a reporting tag;
- returning to the main topic or closing in a particular way.

Express the most distinctive sequence in the order it occurs, such as "text that opens with a claim, concedes a limitation, promises a correction, and then resumes the main point." Include only moves genuinely present in the text, and distinguish stages that perform different functions. Pay attention to conspicuous changes of direction, including an abrupt or forced final turn.

Focus on relationships between sentences or larger units, not grammar within a single sentence. Describe the rhetorical progression rather than mainly summarizing the topic, copying distinctive wording, or merely listing tone and genre. If the structure is simple, describe it accurately without inventing extra stages.

Return only one concise query suitable for a semantic text search engine.

The text:
{text}"""

PROMPTS = {
    1: PROMPT1,
    2: PROMPT2,
    3: PROMPT3,
    4: PROMPT4,
    5: PROMPT5,
    6: PROMPT6,
    7: PROMPT7,
}

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


def generated_paragraph_ids(path: Path, prompt_id: int) -> set[str]:
    """Return paragraph ids already generated with ``prompt_id`` in the output."""
    if not path.exists():
        return set()

    suffix = f"_prompt{prompt_id}"
    paragraph_ids: set[str] = set()
    for line_number, record in enumerate(read_jsonl(path), start=1):
        paragraph_id = record.get("paragraph_id")
        if not isinstance(paragraph_id, str) or not paragraph_id.strip():
            raise ValueError(
                f"Missing paragraph_id in output record {path}:{line_number}"
            )
        record_id = record.get("id")
        # An output file may hold several prompts; only skip the current one.
        if isinstance(record_id, str) and not record_id.endswith(suffix):
            continue
        paragraph_ids.add(paragraph_id)
    return paragraph_ids


def generate_query(
    client: OpenAI,
    text: str,
    *,
    model: str,
    prompt: str,
    prompt_id: int,
) -> str:
    """Generate queries and select the one required by ``prompt_id``."""
    query_count = 5 if prompt_id == 3 else 1
    response = client.responses.create(
        model=model,
        input=prompt.format(text=text),
        text={
            "format": {
                "type": "json_schema",
                "name": "query_idea",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": query_count,
                            "maxItems": query_count,
                        }
                    },
                    "required": ["queries"],
                    "additionalProperties": False,
                },
            }
        },
    )
    queries = json.loads(response.output_text)["queries"]
    # Prompt 3 is ordered vague-to-specific. Keep the penultimate attempt,
    # i.e. the second-most-specific description, for every paragraph.
    query = queries[-2 if prompt_id == 3 else 0].strip()
    if not query:
        raise ValueError("OpenAI returned an empty query")
    return query


def append_question(
    path: Path, paragraph_id: str, query: str, prompt_id: int
) -> None:
    """Append one UTF-8 JSONL question record."""
    record = {
        "id": f"{paragraph_id}_prompt{prompt_id}",
        "paragraph_id": paragraph_id,
        "query": query,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def generate_available_questions(
    records: list[dict[str, Any]],
    *,
    training_data_path: Path,
    output_path: Path,
    prompt_id: int,
    query_generator: Callable[[str], str],
    limit: int | None = None,
) -> int:
    """Generate and persist missing questions, returning the number written."""
    available = available_records(records, training_data_path)
    already_generated = generated_paragraph_ids(output_path, prompt_id)
    pending = [
        record for record in available if record["id"] not in already_generated
    ]
    #shuffle pending :
    random.shuffle(pending)
    if limit is not None:
        pending = pending[:limit]

    total = len(pending)
    
    for number, record in enumerate(pending, start=1):
        paragraph_id = record.get("id")
        text = record.get("text")
        if not isinstance(paragraph_id, str) or not paragraph_id.strip():
            raise ValueError(f"Corpus record has no non-empty id: {record}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Corpus record {paragraph_id!r} has no non-empty text")

        query = query_generator(text)
        append_question(output_path, paragraph_id, query, prompt_id)
        print(f"[{number}/{total}] wrote query for {paragraph_id}")

    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument(
        "--training-data", type=Path, default=DEFAULT_TRAINING_DATA_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        type=int,
        choices=sorted(PROMPTS),
        default=1,
        help="Which prompt to generate queries with.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate at most this many new rows (useful for a test run).",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    load_dotenv()
    records = read_jsonl(args.corpus)
    client = OpenAI()
    prompt = PROMPTS[args.prompt]
    written = generate_available_questions(
        records,
        training_data_path=args.training_data,
        output_path=args.output,
        prompt_id=args.prompt,
        query_generator=lambda text: generate_query(
            client,
            text,
            model=args.model,
            prompt=prompt,
            prompt_id=args.prompt,
        ),
        limit=args.limit,
    )
    print(
        f"Done. Wrote {written} queries with prompt {args.prompt} to {args.output}"
    )


if __name__ == "__main__":
    main()
