"""Persistence helpers for paragraph/question training annotations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

QUESTIONS_PER_PARAGRAPH = 3
MIN_ANNOTATION_WORDS = 20


def is_eligible_for_annotation(
    record: Mapping[str, object], *, min_words: int = MIN_ANNOTATION_WORDS
) -> bool:
    """Return whether a corpus record has enough text to be annotated."""
    text = record.get("text")
    return isinstance(text, str) and len(text.split()) >= min_words


def annotation_counts(path: Path) -> Counter[str]:
    """Return the number of saved questions for each paragraph id."""
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts

    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                counts[item["paragraph_id"]] += 1
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid annotation at {path}:{line_number}"
                ) from exc
    return counts


def save_questions(
    path: Path, paragraph_id: str, questions: Iterable[str]
) -> None:
    """Validate and append one JSONL record per question."""
    cleaned = [question.strip() for question in questions]
    if len(cleaned) != QUESTIONS_PER_PARAGRAPH:
        raise ValueError(f"Exactly {QUESTIONS_PER_PARAGRAPH} questions are required.")
    if any(not question for question in cleaned):
        raise ValueError("All questions must be non-empty.")

    start_idx = annotation_counts(path)[paragraph_id] + 1
    payload = "".join(
        json.dumps(
            {
                "id": f"{paragraph_id}_{query_idx}",
                "paragraph_id": paragraph_id,
                "query": question,
            },
            ensure_ascii=False,
        )
        + "\n"
        for query_idx, question in enumerate(cleaned, start=start_idx)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        # A single write keeps each question batch together.
        fh.write(payload)
