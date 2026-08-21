#!/usr/bin/env python3
"""Generate form-based retrieval queries and append them to the auto training data.

Only corpus texts with more than 40 words are considered. Each eligible passage
gets at most one structure question, stored with the id
``<paragraph_id>_prompta``. Passages that already have such a question are
skipped.

Run with:

    python generate_queries2.py
    python generate_queries2.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = HERE / "corpus_initial.jsonl"
DEFAULT_OUTPUT_PATH = HERE / "queries_generated.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini"
MIN_WORDS = 40
# Every prompt appends to the same file; the id suffix keeps them apart.
QUERY_ID_SUFFIX = "_prompta"

MAX_GENERATION_ATTEMPTS = 3
MAX_QUERY_WORDS = 35
MAX_TOPIC_OVERLAP = 3
RARE_WORD_MAX_DF = 10

PROMPTA = """You write retrieval queries for a benchmark that tests whether a search system can find a passage by its FORM rather than its TOPIC.

Write as someone who read this passage weeks ago. They remember how it moves, or a phrase that stuck to them — and they have completely forgotten what it was about. Asked for the subject, they could not name it. They can only describe the shape.

---

THE RULE THAT MATTERS

The query must not leak the topic. Someone reading only your query must be unable to guess what the passage discusses, and a search engine must not be able to walk back to this passage on subject matter alone.

So: no proper nouns, no technical terms, no domain vocabulary, no objects, no places, no numbers taken from the passage — and no disguised versions of them either. Naming the referent in vaguer words is the same leak: "a metal", "a study", "an album", "a hotel", "a superconductor", "two students" are all forbidden. If you must point at something, say "a claim", "an example", "a name", "a figure".

Your vocabulary is the vocabulary of rhetoric and sentence shape, and nothing else:
claim, qualification, hedge, concession, reversal, exception, aside, digression, resumption, restatement, escalation, deflation, enumeration, address to the reader, imperative, rhetorical question, quotation, attribution tag, parenthesis, repeated connective, abruptly short sentence, sentence that runs on, closing that returns to the opening.

---

STEP 1: choose the type

DISCOURSE — the passage makes 2 or 3 identifiable rhetorical moves in a fixed order, and at least one of them is a turn (contrast, exception, qualification, reversal, return) rather than plain continuation.

SURFACE — the passage carries a distinctive literal marker: an unusual opening clause built from ordinary words, a quotation followed by an attribution tag, a rare punctuation habit, a repeated connective, an odd sentence-final construction.

If both apply, prefer DISCOURSE.

Output {"type": "none", "query": null} if neither is true. A passage that simply states things in order has no form worth querying, and a passage whose only distinctive marker is its subject matter cannot be queried without leaking. Returning none is a correct answer, not a failure — use it often.

---

STEP 2: draft

Begin with "text where", "text that", or "text beginning with".
Lowercase, plain, slightly rough — a note to oneself, not a catalogue entry.
At most three moves, at most 30 words. If the passage makes six moves, keep the two or three most peculiar and drop the rest; narrating all of them turns the query into a summary.
For SURFACE you may quote a short fragment verbatim, but only if it is built from ordinary function words and carries no subject matter. "i daresay you can understand" is quotable. An opening sentence that names its subject is not.

GOOD:
text where the writing opens with a comparative statement, then turns to a general contrast, then to a peculiar exception
text beginning with a quote and then "added mr.."
text where there is an announcement of a change, then a long explanation of what does not change, and then the announcement resumes
text beginning with "i daresay you can understand"
text that opens with a sequence of adjectives before a plain, pathetic, and truthful recital

BAD — every one of these is a summary wearing the words "text where":
text where it starts with a general statement about algae distribution, then shifts to local labels for species, and ends with an example about algae traveling to hawaii on ship hulls
text where it starts by saying a specific iron arsenide is a bulk superconductor, then moves to estimating an upper critical field from onset of diamagnetism
text beginning with milwaukee (ap) — federal prosecutors allege
text where a comedian has died of cardiac arrest, his manager confirms

---

STEP 3: calibrate the difficulty

Imagine a corpus of a hundred thousand passages on every subject. Your query should fit a handful of them — five, maybe ten. If only this passage could ever match, you leaked its content. If hundreds would match ("text that makes a claim and then supports it", "text with a list"), it is too general to be a retrieval target at all. Difficulty comes from an unusual combination of ordinary moves, never from naming things.

---

STEP 4: reject and redraft if ANY of these fail

LEAK — a reader of the query could guess the subject, or some word in the query belongs to the passage's subject matter rather than to its form.
SUMMARY — the query narrates what is said instead of how it goes, or runs past three moves or 30 words.
UNVERIFIABLE — a clause that cannot be pointed at in the passage.
TOO GENERAL — the moves are so ordinary that most passages would match.

---

here is the text :"""


STOPWORDS = frozenset(
    """
    a an the this that these those it its it's they them their there here
    i you he she we us me my your his her our
    is are was were be been being am do does did done doing have has had
    will would shall should can could may might must
    of in on at to from by for with without into onto out up down over under
    about after before during between through against within across along
    and or but nor so yet if then than as because while when where why how
    not no none nor only just very much more most less least such same other
    another both each either neither all any some many few several one two
    what which who whom whose whether
    """.split()
)

# Words that describe form rather than subject matter: they may echo the
# passage without giving away what it is about.
FORM_VOCABULARY = frozenset(
    """
    text passage writing line lines sentence sentences clause clauses phrase
    phrases word words paragraph section quote quotation quoted quoting
    attribution tag aside asides digression parenthesis parenthetical
    claim claims statement statements point remark note observation
    qualification hedge concession reversal exception contrast comparative
    comparison announcement explanation resumption restatement escalation
    enumeration list listing question answer imperative address reader
    opens opening open begins beginning begin starts start ends ending end
    turns turn shifts shift moves move returns return resumes resume
    repeats repeat repeated repetition follows following followed
    first second third last final finally next again still
    general specific broad narrow long short brief abrupt plain odd peculiar
    unusual strange sudden slow flat run running
    """.split()
)


def normalize_word(word: str) -> str:
    """Fold a token to a crude stem so ``transported``/``transport`` match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def topic_words(text: str) -> set[str]:
    """Return the stemmed content words of ``text``, ignoring form vocabulary."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {
        normalize_word(word)
        for word in words
        if len(word) > 2 and word not in STOPWORDS and word not in FORM_VOCABULARY
    }


def rare_corpus_words(
    records: list[dict[str, Any]], *, max_document_frequency: int = RARE_WORD_MAX_DF
) -> frozenset[str]:
    """Return words occurring in few corpus passages, i.e. words that name a topic."""
    document_frequency: Counter[str] = Counter()
    for record in records:
        document_frequency.update(topic_words(record["text"]))
    return frozenset(
        word
        for word, count in document_frequency.items()
        if count <= max_document_frequency
    )


def difficulty_problem(
    query: str, text: str, *, rare_words: frozenset[str] = frozenset()
) -> str | None:
    """Return why ``query`` is too easy a retrieval target, or ``None`` if it holds."""
    word_count = len(query.split())
    if word_count > MAX_QUERY_WORDS:
        return (
            f"the query runs to {word_count} words, which means it is narrating the "
            f"passage instead of describing its shape; stay under {MAX_QUERY_WORDS}"
        )

    shared = sorted(topic_words(query) & topic_words(text))

    distinctive = sorted(word for word in shared if word in rare_words)
    if distinctive:
        return (
            f"the query borrows words that barely occur anywhere else in the corpus "
            f"({', '.join(distinctive[:8])}), which points at this passage on its "
            "subject alone; name no part of what the passage is about"
        )

    if len(shared) > MAX_TOPIC_OVERLAP:
        leaked = ", ".join(shared[:8])
        return (
            f"the query reuses the passage's own subject-matter words ({leaked}), so a "
            "plain keyword search finds the passage; describe the form with rhetorical "
            "vocabulary only"
        )
    return None


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


def eligible_records(
    records: list[dict[str, Any]], *, min_words: int = MIN_WORDS
) -> list[dict[str, Any]]:
    """Return records whose text contains strictly more than ``min_words`` words."""
    eligible: list[dict[str, Any]] = []
    for record in records:
        paragraph_id = record.get("id")
        text = record.get("text")
        if not isinstance(paragraph_id, str) or not paragraph_id.strip():
            raise ValueError(f"Corpus record has no non-empty id: {record}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Corpus record {paragraph_id!r} has no non-empty text")
        if len(text.split()) > min_words:
            eligible.append(record)
    return eligible


def generated_paragraph_ids(path: Path) -> set[str]:
    """Return paragraph ids that already have a ``_prompta`` query in ``path``."""
    if not path.exists():
        return set()

    paragraph_ids: set[str] = set()
    for line_number, record in enumerate(read_jsonl(path), start=1):
        paragraph_id = record.get("paragraph_id")
        question_id = record.get("id")
        if not isinstance(paragraph_id, str) or not paragraph_id.strip():
            raise ValueError(
                f"Missing paragraph_id in output record {path}:{line_number}"
            )
        # An output file may hold several prompts; only skip this one.
        if isinstance(question_id, str) and question_id.endswith(QUERY_ID_SUFFIX):
            paragraph_ids.add(paragraph_id)
    return paragraph_ids


def parse_query_response(response: Any) -> str | None:
    """Validate and parse a completed structured response."""
    status = getattr(response, "status", None)
    if status != "completed":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        suffix = f" ({reason})" if reason else ""
        raise ValueError(f"OpenAI response status was {status!r}{suffix}")

    output_text = getattr(response, "output_text", "")
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("OpenAI returned no structured output text")

    try:
        result = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI returned malformed JSON: {exc.msg}") from exc

    query_type = result["type"]
    query = result["query"]

    if query_type == "none":
        if query is not None:
            raise ValueError("OpenAI returned a query for a passage of type 'none'")
        return None
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"OpenAI returned no query for type {query_type!r}")

    cleaned = query.strip()
    allowed_openings = ("text where", "text that", "text beginning with")
    if not cleaned.startswith(allowed_openings):
        raise ValueError(f"OpenAI returned a query with an invalid opening: {cleaned!r}")
    return cleaned


def generate_query(
    client: OpenAI,
    text: str,
    *,
    model: str,
    rare_words: frozenset[str] = frozenset(),
) -> str | None:
    """Generate one form query, retrying malformed or too-easy responses."""
    last_error: Exception | None = None
    feedback = ""
    rejected = False
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        response = client.responses.create(
            model=model,
            input=f"{PROMPTA}\n\n{text}{feedback}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "structure_query",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["surface", "discourse", "none"],
                            },
                            "query": {"type": ["string", "null"]},
                        },
                        "required": ["type", "query"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        try:
            query = parse_query_response(response)
        except (KeyError, TypeError, ValueError) as exc:
            last_error = exc
            feedback = ""
            rejected = False
            if attempt < MAX_GENERATION_ATTEMPTS:
                print(
                    "OpenAI returned incomplete or invalid structured output; "
                    f"retrying ({attempt}/{MAX_GENERATION_ATTEMPTS})..."
                )
            continue

        if query is None:
            return None

        problem = difficulty_problem(query, text, rare_words=rare_words)
        if problem is None:
            return query

        last_error = ValueError(f"{problem}: {query!r}")
        rejected = True
        feedback = (
            "\n\n---\n\nYour previous attempt was rejected.\n"
            f"rejected query: {query}\n"
            f"reason: {problem}\n"
            "Write a different query that fixes this. Keep only the shape and drop the "
            "subject matter entirely; return type 'none' if the passage has no form "
            "that can be described without naming what it is about."
        )
        if attempt < MAX_GENERATION_ATTEMPTS:
            print(
                f"  rejected query ({problem.split(',')[0]}); "
                f"retrying ({attempt}/{MAX_GENERATION_ATTEMPTS})..."
            )

    if rejected:
        # The passage has no form that survives the difficulty checks: skip it
        # rather than store a query a keyword search would solve.
        return None

    raise RuntimeError(
        f"OpenAI did not return a valid structure query after "
        f"{MAX_GENERATION_ATTEMPTS} attempts"
    ) from last_error


def append_question(path: Path, paragraph_id: str, query: str) -> None:
    """Append one UTF-8 JSONL structure-question record."""
    record = {
        "id": f"{paragraph_id}{QUERY_ID_SUFFIX}",
        "paragraph_id": paragraph_id,
        "query": query,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def add_structure_questions(
    records: list[dict[str, Any]],
    *,
    output_path: Path,
    query_generator: Callable[[str], str | None],
    limit: int | None = None,
) -> int:
    """Generate structure questions and return the number appended."""
    already_generated = generated_paragraph_ids(output_path)
    pending = [
        record
        for record in eligible_records(records)
        if record["id"] not in already_generated
    ]
    random.shuffle(pending)
    if limit is not None:
        pending = pending[:limit]

    total = len(pending)
    written = 0
    for number, record in enumerate(pending, start=1):
        paragraph_id = record["id"]
        query = query_generator(record["text"])
        if query is None:
            print(f"[{number}/{total}] skipped {paragraph_id}: no clean structure")
            continue

        append_question(output_path, paragraph_id, query)
        written += 1
        print(f"[{number}/{total}] wrote structure query for {paragraph_id}")

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many eligible passages.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    load_dotenv()
    records = read_jsonl(args.corpus)
    #each record is a line of corpus.jsonl
    client = OpenAI()
    rare_words = rare_corpus_words(records)
    written = add_structure_questions(
        records,
        output_path=args.output,
        query_generator=lambda text: generate_query(
            client, text, model=args.model, rare_words=rare_words
        ),
        limit=args.limit,
    )
    print(f"Done. Wrote {written} structure queries to {args.output}")


if __name__ == "__main__":
    main()
