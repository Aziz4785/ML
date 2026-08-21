#!/usr/bin/env python3
"""Append ``_m`` to every top-level ``id`` in one or more JSONL files."""

import argparse
import json
import os
import tempfile
from pathlib import Path


def append_suffix(path: Path) -> tuple[int, int]:
    updated = 0
    unchanged = 0

    with path.open("r", encoding="utf-8") as source, tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as destination:
        temporary_path = Path(destination.name)

        try:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    destination.write(line)
                    continue

                record = json.loads(line)
                record_id = record.get("id")
                if not isinstance(record_id, str):
                    raise ValueError(
                        f"Line {line_number}: expected a string in the 'id' field"
                    )

                if record_id.endswith("_m"):
                    unchanged += 1
                else:
                    record["id"] = f"{record_id}_m"
                    updated += 1

                destination.write(json.dumps(record, ensure_ascii=False) + "\n")

            destination.flush()
            os.fsync(destination.fileno())
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    return updated, unchanged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append '_m' to every top-level id in one or more JSONL files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path(__file__).with_name("queries_manual.jsonl"),
            Path(__file__).with_name("generated_llm_text.jsonl"),
            Path(__file__).with_name("generated_llm_text_embeddings.jsonl"),
        ],
        help="JSONL files to update in place (default: the three project datasets)",
    )
    args = parser.parse_args()

    for path in args.paths:
        updated, unchanged = append_suffix(path)
        print(
            f"{path}: updated {updated} IDs; "
            f"{unchanged} already ended with '_m'."
        )


if __name__ == "__main__":
    main()
