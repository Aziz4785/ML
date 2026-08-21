import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from measure_baseline import load_queries, parse_args


class HardQueryFilteringTests(unittest.TestCase):
    def test_hard_only_keeps_ids_containing_hard(self) -> None:
        records = [
            {"id": "query_easy", "paragraph_id": "p1", "query": "Easy"},
            {"id": "query_hard_1", "paragraph_id": "p2", "query": "Hard"},
            {"id": "prefix_hard_suffix", "paragraph_id": "p3", "query": "Harder"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            queries, skipped = load_queries([path], hard_only=True)

        self.assertEqual(
            [query.query_id for query in queries],
            ["query_hard_1", "prefix_hard_suffix"],
        )
        self.assertEqual(skipped, {})

    def test_hard_only_flag_is_opt_in(self) -> None:
        self.assertFalse(parse_args([]).hard_only)
        self.assertTrue(parse_args(["--hard-only"]).hard_only)

    def test_hard_only_verbose_reports_found_and_matching_counts(self) -> None:
        records = [
            {"id": "query_easy", "paragraph_id": "p1", "query": "Easy"},
            {"id": "query_hard_1", "paragraph_id": "p2", "query": "Hard"},
            {"id": "query_hard_2", "paragraph_id": "p3", "query": "Harder"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                load_queries(
                    [path],
                    keep_ids={"query_hard_2"},
                    hard_only=True,
                    verbose=True,
                )

        self.assertIn("Hard queries found: 2", stderr.getvalue())
        self.assertIn(
            "Hard queries with matching LLM embeddings: 1", stderr.getvalue()
        )


if __name__ == "__main__":
    unittest.main()
