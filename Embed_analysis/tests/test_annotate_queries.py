from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.annotate_queries import (
    FOLLOW_TEMPLATE_FORMAT,
    QUERY_TYPE_FORMAT,
    SUBJECT_MATTER_UNSPECIFIED_FORMAT,
    annotate_records,
    annotate_follow_template,
    annotate_query_type,
    annotate_subject_matter_unspecified,
    classify_follow_template,
    classify_query_type,
    classify_subject_matter_unspecified,
    load_records,
    query_about_length,
)


class QueryAboutLengthTests(unittest.TestCase):
    def test_detects_number_followed_by_words(self) -> None:
        self.assertEqual(query_about_length("write an answer in 150 words please"), 1)
        self.assertEqual(query_about_length("Summarize this in 20 WORDS."), 1)

    def test_number_must_be_followed_by_words(self) -> None:
        self.assertEqual(query_about_length("words about 150 topics"), 0)
        self.assertEqual(query_about_length("use 150 characters"), 0)

    def test_detects_short_or_long_text_only_below_five_words(self) -> None:
        self.assertEqual(query_about_length("a short text"), 1)
        self.assertEqual(query_about_length("LONG TEXT please"), 1)
        self.assertEqual(query_about_length("please write a short text"), 0)

    def test_annotates_without_mutating_source_records(self) -> None:
        source = [{"id": "one", "query": "short text"}]
        annotated = annotate_records(source)
        self.assertNotIn("query_about_length", source[0])
        self.assertEqual(annotated[0]["query_about_length"], 1)


class LoadingTests(unittest.TestCase):
    def test_rejects_a_record_without_a_string_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.json"
            path.write_text(json.dumps([{"query": None}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "string 'query'"):
                load_records(path)


class FakeResponses:
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        output = {
            "results": [
                {"index": index, "follow_template": label}
                for index, label in enumerate(self.labels)
            ]
        }
        return type(
            "Response",
            (),
            {"status": "completed", "output_text": json.dumps(output)},
        )()


class FollowTemplateTests(unittest.TestCase):
    def test_requests_strict_structured_labels_from_gpt_5_6_luna(self) -> None:
        responses = FakeResponses([1, 0])
        client = type("Client", (), {"responses": responses})()

        labels = classify_follow_template(
            client,
            ["a concise academic history essay", "what is 2 + 2?"],
        )

        self.assertEqual(labels, [1, 0])
        request = responses.requests[0]
        self.assertEqual(request["model"], "gpt-5.6-luna")
        self.assertEqual(request["text"], {"format": FOLLOW_TEMPLATE_FORMAT})
        self.assertFalse(request["store"])

    def test_batches_checkpoints_and_skips_existing_values(self) -> None:
        records = [
            {"query": "already done", "follow_template": 1},
            {"query": "formal science explanation"},
            {"query": "a short story"},
        ]
        responses = FakeResponses([0, 1])
        client = type("Client", (), {"responses": responses})()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "annotated.json"
            count = annotate_follow_template(
                records,
                client,
                output=output,
                batch_size=2,
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(count, 2)
        self.assertEqual([row["follow_template"] for row in saved], [1, 0, 1])
        sent = json.loads(responses.requests[0]["input"][1]["content"])
        self.assertEqual(
            [item["query"] for item in sent],
            ["formal science explanation", "a short story"],
        )


class FakeQueryTypeResponses:
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        output = {
            "results": [
                {"index": index, "query_type": label}
                for index, label in enumerate(self.labels)
            ]
        }
        return type(
            "Response",
            (),
            {"status": "completed", "output_text": json.dumps(output)},
        )()


class QueryTypeTests(unittest.TestCase):
    def test_requests_strict_structured_labels_from_gpt_5_6_luna(self) -> None:
        responses = FakeQueryTypeResponses([1, 2, 0])
        client = type("Client", (), {"responses": responses})()

        labels = classify_query_type(
            client,
            [
                "a journalistic explanation about migration",
                "a passage that opens with a claim then gives a concession",
                "what is 2 + 2?",
            ],
        )

        self.assertEqual(labels, [1, 2, 0])
        request = responses.requests[0]
        self.assertEqual(request["model"], "gpt-5.6-luna")
        self.assertEqual(request["text"], {"format": QUERY_TYPE_FORMAT})
        self.assertFalse(request["store"])

    def test_batches_checkpoints_and_preserves_existing_values(self) -> None:
        records = [
            {"query": "already done", "query_type": 1},
            {"query": "text that begins with a claim and ends with a reversal"},
            {"query": "what is 2 + 2?"},
        ]
        responses = FakeQueryTypeResponses([2, 0])
        client = type("Client", (), {"responses": responses})()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "annotated.json"
            count = annotate_query_type(
                records,
                client,
                output=output,
                batch_size=2,
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(count, 2)
        self.assertEqual([row["query_type"] for row in saved], [1, 2, 0])
        sent = json.loads(responses.requests[0]["input"][1]["content"])
        self.assertEqual(
            [item["query"] for item in sent],
            [
                "text that begins with a claim and ends with a reversal",
                "what is 2 + 2?",
            ],
        )


class FakeSubjectMatterResponses:
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        output = {
            "results": [
                {"index": index, "subject_matter_unspecified": label}
                for index, label in enumerate(self.labels)
            ]
        }
        return type(
            "Response",
            (),
            {"status": "completed", "output_text": json.dumps(output)},
        )()


class SubjectMatterUnspecifiedTests(unittest.TestCase):
    def test_requests_strict_structured_binary_labels(self) -> None:
        responses = FakeSubjectMatterResponses([1, 0, 0])
        client = type("Client", (), {"responses": responses})()

        labels = classify_subject_matter_unspecified(
            client,
            [
                "a concise formal explanation",
                "a concise explanation about mythology",
                "promotional copy for a new camera",
            ],
        )

        self.assertEqual(labels, [1, 0, 0])
        request = responses.requests[0]
        self.assertEqual(request["model"], "gpt-5.6-luna")
        self.assertEqual(
            request["text"], {"format": SUBJECT_MATTER_UNSPECIFIED_FORMAT}
        )
        self.assertFalse(request["store"])

    def test_batches_checkpoints_and_preserves_existing_values(self) -> None:
        records = [
            {"query": "already done", "subject_matter_unspecified": 0},
            {"query": "an AP-style report"},
            {"query": "a scientific explanation of photosynthesis"},
        ]
        responses = FakeSubjectMatterResponses([1, 0])
        client = type("Client", (), {"responses": responses})()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "annotated.json"
            count = annotate_subject_matter_unspecified(
                records,
                client,
                output=output,
                batch_size=2,
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(count, 2)
        self.assertEqual(
            [row["subject_matter_unspecified"] for row in saved], [0, 1, 0]
        )
        sent = json.loads(responses.requests[0]["input"][1]["content"])
        self.assertEqual(
            [item["query"] for item in sent],
            ["an AP-style report", "a scientific explanation of photosynthesis"],
        )


if __name__ == "__main__":
    unittest.main()
