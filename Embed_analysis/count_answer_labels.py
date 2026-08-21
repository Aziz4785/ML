#!/usr/bin/env python3
"""Count labels for queries containing the word "answer" (case-insensitive)."""

import json
import re
from pathlib import Path


INPUT_FILE = Path(__file__).with_name("all_queries.json")
ANSWER_WORD = re.compile(r"\banswer\b", re.IGNORECASE)


def main() -> None:
    with INPUT_FILE.open(encoding="utf-8") as file:
        queries = json.load(file)

    counts = {0: 0, 1: 0}

    for item in queries:
        query = str(item.get("query", "")).lower()
        label = item.get("label")

        if ANSWER_WORD.search(query) and label in counts:
            counts[label] += 1

    print(f"Label 1: {counts[1]}")
    print(f"Label 0: {counts[0]}")
    print(f"Total:   {counts[0] + counts[1]}")


if __name__ == "__main__":
    main()
