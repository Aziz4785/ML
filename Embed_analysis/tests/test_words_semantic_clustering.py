import json
from pathlib import Path
import tempfile
import unittest

from analysis.words_semantic_clustering import (
    build_report,
    extract_terms,
    grammatical_signature,
    load_queries,
    semantic_root,
    signatures_are_compatible,
)


class WordsSemanticClusteringTests(unittest.TestCase):
    def test_loads_both_files_and_validates_query_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first.write_text(json.dumps([{"query": "first query"}]))
            second.write_text(json.dumps([{"query": "second query"}]))
            self.assertEqual(load_queries([first, second]), ["first query", "second query"])

    def test_extracts_words_removes_stop_words_and_keeps_clause_phrases(self) -> None:
        counts = extract_terms(
            ["Text beginning with an aside, then a long appeal, then an enumeration of ideas"]
        )
        self.assertNotIn("text", counts)
        self.assertNotIn("an", counts)
        self.assertEqual(counts["beginning"], 1)
        self.assertEqual(counts["an aside"], 1)
        self.assertEqual(counts["a long appeal"], 1)
        self.assertEqual(counts["an enumeration"], 1)

    def test_extracts_auxiliary_with_progressive_verb_as_one_unit(self) -> None:
        counts = extract_terms(["A text that is starting with a question"])
        self.assertEqual(counts["is starting"], 1)

    def test_does_not_treat_ing_nouns_as_progressive_verbs(self) -> None:
        counts = extract_terms(["What is something useful?"])
        self.assertNotIn("is something", counts)

    def test_only_compatible_surface_forms_can_share_a_cluster(self) -> None:
        starts = grammatical_signature("starts")
        is_starting = grammatical_signature("is starting")
        self.assertTrue(signatures_are_compatible(starts, is_starting))
        self.assertFalse(
            signatures_are_compatible(starts, grammatical_signature("starting"))
        )
        self.assertFalse(
            signatures_are_compatible(starts, grammatical_signature("start"))
        )

    def test_inflected_replacement_forms_have_same_semantic_root(self) -> None:
        self.assertEqual(semantic_root("starts"), "start")
        self.assertEqual(semantic_root("is starting"), "start")
        self.assertEqual(semantic_root("begins"), "begin")
        self.assertEqual(semantic_root("is beginning"), "begin")

    def test_report_has_stable_cluster_ids_and_frequency_order(self) -> None:
        terms = ["academic", "factual", "starting"]
        counts = {"academic": 2, "factual": 5, "starting": 1}
        report = build_report(
            terms,
            counts,  # type: ignore[arg-type]
            [[0, 1]],
            input_paths=[Path("queries.json")],
            query_count=3,
            model_name="test-model",
            threshold=0.7,
        )
        self.assertEqual(report["groups"][0]["id"], "cluster1")
        self.assertEqual(report["groups"][0]["words"], ["factual", "academic"])
        self.assertEqual(report["metadata"]["grouped_candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
