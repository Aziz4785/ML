from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from topic_model_hard_queries import (
    PreparedRecord,
    force_assign_by_embedding_centroids,
    prepare_records,
    read_json_records,
    reduce_outliers_two_stage,
    request_topic_label,
    select_representative_indices,
)


class RecordLoadingTests(unittest.TestCase):
    def test_reads_json_array_and_jsonl(self) -> None:
        rows = [
            {"id": "a", "query": "first prompt", "rank": 10},
            {"id": "b", "query": "second prompt", "rank": 20},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "rows.json"
            jsonl_path = root / "rows.jsonl"
            json_path.write_text(json.dumps(rows), encoding="utf-8")
            jsonl_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(read_json_records(json_path), rows)
            self.assertEqual(read_json_records(jsonl_path), rows)

    def test_rejects_non_object_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('[{"id": "a"}, 3]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record 1"):
                read_json_records(path)


class PreprocessingTests(unittest.TestCase):
    def test_filters_language_and_normalized_duplicates(self) -> None:
        rows = [
            {"id": "a", "query": "  Hello   WORLD  ", "language": "EN"},
            {"id": "b", "query": "hello world", "language": "english"},
            {"id": "c", "query": "bonjour", "language": "fr"},
            {"id": "d", "query": "missing tag"},
            {"id": "e", "query": "Another English prompt", "language": "eng"},
        ]
        records, stats = prepare_records(
            rows,
            id_field="id",
            text_field="query",
            language_field="language",
        )
        self.assertEqual([record.record_id for record in records], ["a", "e"])
        self.assertEqual(stats["duplicates_removed"], 1)
        self.assertEqual(stats["language_filter"]["non_english_removed"], 1)
        self.assertEqual(stats["language_filter"]["missing_tag_removed"], 1)

    def test_rejects_duplicate_ids_with_different_prompts(self) -> None:
        rows = [
            {"id": "same", "query": "first"},
            {"id": "same", "query": "second"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_records(rows, id_field="id", text_field="query")


class RepresentativeSelectionTests(unittest.TestCase):
    def test_uses_strict_top_fraction_then_word_limit(self) -> None:
        records = [
            PreparedRecord(i, str(i), f"prompt {i}", {"id": str(i)})
            for i in range(10)
        ]
        records[9] = PreparedRecord(
            9,
            "9",
            "one two three four five six",
            {"id": "9"},
        )
        topics = [0] * 10
        probabilities = [0.1 * i for i in range(10)]
        selected = select_representative_indices(
            records,
            topics,
            probabilities,
            examples_per_topic=20,
            high_probability_fraction=0.2,
            max_words=5,
        )[0]

        # Top 20% is indices 9 and 8. Index 9 is then removed by the word limit.
        self.assertEqual(selected["probability_pool_size"], 2)
        self.assertEqual(selected["indices"], [8])
        self.assertEqual(selected["shortfall"], 19)


class FakeTopicModel:
    def __init__(self) -> None:
        self.calls = []

    def reduce_outliers(self, documents, topics, **kwargs):
        self.calls.append((list(documents), list(topics), kwargs))
        if kwargs["strategy"] == "probabilities":
            return [0, -1, 0, 1]
        if kwargs["strategy"] == "embeddings":
            return [0, 1, 0, 1]
        raise AssertionError("unexpected strategy")


class OutlierReductionTests(unittest.TestCase):
    def test_chains_probability_and_embedding_strategies(self) -> None:
        model = FakeTopicModel()
        result = reduce_outliers_two_stage(
            model,
            ["a", "b", "c", "d"],
            [-1, -1, 0, 1],
            [[0.8, 0.2], [0.1, 0.2], [0.9, 0.1], [0.2, 0.8]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            probability_threshold=0.05,
            embedding_threshold=0.0,
        )
        self.assertEqual(result.final_topics, [0, 1, 0, 1])
        self.assertEqual(
            result.methods,
            [
                "hdbscan_probability",
                "embedding_similarity",
                "original_cluster",
                "original_cluster",
            ],
        )
        self.assertEqual([call[2]["strategy"] for call in model.calls], [
            "probabilities",
            "embeddings",
        ])
        self.assertEqual(model.calls[0][2]["threshold"], 0.05)

    def test_centroid_fallback_assigns_remaining_outlier(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.8, 0.2],
            ]
        )
        final, forced = force_assign_by_embedding_centroids(
            embeddings,
            core_topics=[0, 0, 1, 1, -1],
            current_topics=[0, 0, 1, 1, -1],
        )
        self.assertEqual(final, [0, 0, 1, 1, 0])
        self.assertEqual(forced, 1)


class FakeResponses:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return type(
            "Response",
            (),
            {
                "status": "completed",
                "output_text": json.dumps(
                    {
                        "name": "Database Query Help",
                        "description": "Prompts asking for help constructing SQL queries.",
                    }
                ),
            },
        )()


class OpenAILabelTests(unittest.TestCase):
    def test_requests_and_parses_strict_structured_label(self) -> None:
        responses = FakeResponses()
        client = type("Client", (), {"responses": responses})()
        label = request_topic_label(
            client,
            model="gpt-4o",
            topic_id=3,
            examples=["help me query a table"],
            keywords=["sql", "query"],
        )
        self.assertEqual(label["name"], "Database Query Help")
        self.assertEqual(responses.request["model"], "gpt-4o")
        response_format = responses.request["text"]["format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["strict"])
        self.assertFalse(response_format["schema"]["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
