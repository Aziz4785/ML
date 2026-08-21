"""Pipeline for loading, preprocessing, and featurizing query texts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from spacy.tokens import Doc

from CLARIN.feature_extraction import count_features
from CLARIN.preprocess_spacy import preprocess_queries
from CLARIN.text_classify import (
    ClassifierName,
    ClassificationResult,
    classify_features,
    feature_importance as calculate_feature_importance,
)
from CLARIN.text_visualise import plot_feature_importance, plot_shap

if TYPE_CHECKING:
    from matplotlib.figure import Figure


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "all_queries_fewMS.json"
PREPROCESSED_QUERIES_PATH = Path(__file__).resolve().parent / "preprocessed_queries_fewMS.json"


class QueryPipeline:
    """Run the CLARIN query-processing stages and retain their results.

    Parameters
    ----------
    queries_path:
        Path to records containing a ``query`` field. Running classification
        also requires the configured binary target field (``label`` by default).
    feature_scheme:
        Optional scheme passed to :func:`CLARIN.feature_extraction.count_features`.
        When omitted, the extractor's default scheme is used.
    target_field:
        Binary record field to predict.
    test_size:
        Fraction of records reserved for held-out evaluation.
    random_state:
        Seed used for the reproducible train/test split and model.
    classifier:
        ``"lgbm"`` (the default) or ``"decision_tree"``.
    verbose:
        Verbosity passed to the feature extractor.
    """

    def __init__(
        self,
        queries_path: str | Path = DEFAULT_QUERIES_PATH,
        feature_scheme: dict[str, Any] | None = None,
        target_field: str = "label",
        test_size: float = 0.2,
        random_state: int = 42,
        classifier: ClassifierName = "lgbm",
        verbose: int = 0,
    ) -> None:
        self.queries_path = Path(queries_path)
        self.feature_scheme = feature_scheme
        self.target_field = target_field
        self.test_size = test_size
        self.random_state = random_state
        self.classifier = classifier
        self.verbose = verbose

        self.records: list[dict[str, Any]] = []
        self.queries: list[str] = []
        self.docs: list[Doc] = []
        self.features: pd.DataFrame | None = None
        self.feature_names: list[str] = []
        self.feature_names_waterfall: list[str] = []
        self.labels: list[int] = []
        self.classification: ClassificationResult | None = None
        self.feature_importance: pd.DataFrame | None = None
        self.explanation_figure: Figure | None = None
        self.shap_figure: Figure | None = None

    def texts_load(self, queries_path: str | Path | None = None) -> list[str]:
        """Load query records from the selected JSON file."""
        if queries_path is not None:
            self.queries_path = Path(queries_path)

        with self.queries_path.open(encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list):
            raise ValueError(f"Expected a list of query records in {self.queries_path}")
        if not all(
            isinstance(record, dict) and isinstance(record.get("query"), str)
            for record in records
        ):
            raise ValueError(
                f"Every record in {self.queries_path} must contain a string 'query' field"
            )

        self.records = records
        self.queries = [record["query"] for record in records]

        # Loading a new input invalidates every downstream pipeline result.
        self.docs = []
        self.features = None
        self.feature_names = []
        self.feature_names_waterfall = []
        self.labels = []
        self.classification = None
        self.feature_importance = None
        self.explanation_figure = None
        self.shap_figure = None
        return self.queries

    def texts_preprocess(self) -> list[Doc]:
        """Convert the loaded query strings into annotated spaCy documents."""
        if not self.records:
            raise RuntimeError("No queries loaded. Call texts_load() first.")

        records, docs = preprocess_queries(self.queries_path)
        self.records = records
        self.queries = [record["query"] for record in records]
        self.docs = docs

        # Reprocessing invalidates features produced from earlier documents.
        self.features = None
        self.feature_names = []
        self.feature_names_waterfall = []
        self.labels = []
        self.classification = None
        self.feature_importance = None
        self.explanation_figure = None
        self.shap_figure = None
        return self.docs

    def extract_features(
        self,
        features: int | list[int] | None = None,
    ) -> pd.DataFrame:
        """Extract document features and store their model and display names.

        Parameters
        ----------
        features:
            Feature ID or list of feature IDs to extract for this call. When
            omitted, the IDs from ``self.feature_scheme`` (or the extractor's
            default scheme) are used.
        """
        print("Extracting features...")
        if not self.docs:
            raise RuntimeError("No spaCy documents available. Call texts_preprocess() first.")

        kwargs: dict[str, Any] = {"verbose": self.verbose}
        if self.feature_scheme is not None:
            kwargs["feature_scheme"] = self.feature_scheme
        if features is not None:
            kwargs["features"] = features

        feature_frame = count_features(self.docs, **kwargs)
        if feature_frame is None:
            raise RuntimeError("Feature extraction failed to produce a DataFrame.")
        print("shape of features:", feature_frame.shape)

        self.features = feature_frame
        self.feature_names = list(
            feature_frame.attrs.get("feature_names", feature_frame.columns)
        )
        self.feature_names_waterfall = list(
            feature_frame.attrs.get("feature_names_waterfall", self.feature_names)
        )
        self.classification = None
        self.feature_importance = None
        self.explanation_figure = None
        self.shap_figure = None
        print("we return features with shape:", self.features.shape)
        return self.features

    def classify(self) -> ClassificationResult:
        """Predict the records' binary target from the extracted text features."""
        print()
        print("Classifying features...")
        if self.features is None:
            raise RuntimeError("No features available. Call extract_features() first.")

        labels = [record.get(self.target_field) for record in self.records]
        print("labels[:100]", labels[:100])
        if any(isinstance(label, bool) or label not in (0, 1) for label in labels):
            raise ValueError(
                f"Every record must contain a binary integer {self.target_field!r} field"
            )

        self.labels = [int(label) for label in labels]
        self.classification = classify_features(
            self.features,
            self.labels,
            test_size=self.test_size,
            random_state=self.random_state,
            classifier=self.classifier,
        )
        self.feature_importance = None
        self.explanation_figure = None
        self.shap_figure = None
        return self.classification

    def explain(
        self,
        *,
        max_display: int = 20,
        show: bool = False,
        output_path: str | Path | None = None,
        shap_output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Rank features and render importance and SHAP plots."""
        if self.classification is None:
            raise RuntimeError("No classification available. Call classify() first.")

        if self.features is None:
            raise RuntimeError("No features available. Call extract_features() first.")
        held_out_features = self.features.iloc[self.classification.test_indices]
        self.feature_importance = calculate_feature_importance(
            self.classification,
            held_out_features,
            self.feature_names,
            label=1,
        )
        self.explanation_figure = plot_feature_importance(
            self.feature_importance,
            max_display=max_display,
            show=show,
            output_path=output_path,
        )
        self.shap_figure = plot_shap(
            self.classification.model,
            held_out_features,
            self.feature_names,
            label=1,
            max_display=max_display,
            show=show,
            output_path=shap_output_path,
        )
        return self.feature_importance

    def run(
        self,
        *,
        features: int | list[int] | None = None,
        show_explanation: bool = False,
    ) -> "QueryPipeline":
        """Run all pipeline stages in order."""
        self.texts_load()
        self.texts_preprocess()
        self.extract_features(features=features)
        self.classify()
        self.explain(show=show_explanation)
        return self


# A concise alias for callers that prefer ``Pipeline(...)``.
Pipeline = QueryPipeline
