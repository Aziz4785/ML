#!/usr/bin/env python3
"""Train a logistic regression model that predicts labels in ``all_queries.json``.

The four manually annotated features and the query's word count are used.
``id``, ``rank``, and the query text itself are excluded; rank reveals the label
in this dataset and would make an evaluation misleading.

Example:
    python3 prediction/logistic_regression.py
    python3 prediction/logistic_regression.py path/to/queries.json --folds 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Sequence

import numpy as np
from scipy.special import expit
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "all_queries.json"
ANNOTATION_FEATURES = (
    "query_about_length",
    "follow_template",
    "query_type",
    "subject_matter_unspecified",
)
RAW_FEATURE_NAMES = (*ANNOTATION_FEATURES, "query_word_count")

# query_type is categorical, not ordinal.  The model therefore gets one
# indicator for each of its possible values instead of the raw value 0/1/2.
MODEL_FEATURE_NAMES = (
    "query_about_length",
    "follow_template",
    "subject_matter_unspecified",
    "query_word_count",
    "query_type=0",
    "query_type=1",
    "query_type=2",
)


@dataclass(frozen=True)
class FeatureTestResult:
    """Result of a univariate categorical association test."""

    name: str
    test_name: str
    statistic_name: str
    statistic: float
    degrees_of_freedom: int | None
    p_value: float
    adjusted_p_value: float
    cramers_v: float
    label_one_rates: tuple[tuple[int, float, int], ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"JSON input file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="number of stratified cross-validation folds (default: 10)",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0,
        metavar="C",
        help="inverse L2 regularization strength (default: 1.0)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="reproducible cross-validation and fit seed (default: 42)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="family-wise significance level for feature tests (default: 0.05)",
    )
    args = parser.parse_args(argv)

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.regularization <= 0.0:
        parser.error("--regularization must be greater than 0")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be between 0 and 1")
    return args


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate records, returning raw features and binary labels."""
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc

    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty JSON array")

    feature_rows: list[list[float]] = []
    labels: list[int] = []
    for index, record in enumerate(records):
        location = f"{path}: record {index}"
        if not isinstance(record, dict):
            raise ValueError(f"{location} must be an object")

        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{location} must have a non-empty string 'query'")

        label = record.get("label")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"{location} must have a binary integer 'label'")

        values: list[float] = []
        for feature in ANNOTATION_FEATURES:
            value = record.get(feature)
            allowed = (0, 1, 2) if feature == "query_type" else (0, 1)
            if isinstance(value, bool) or value not in allowed:
                raise ValueError(
                    f"{location} has invalid {feature!r}; expected one of {allowed}"
                )
            values.append(float(value))

        values.append(float(len(query.split())))
        feature_rows.append(values)
        labels.append(int(label))

    class_counts = Counter(labels)
    if set(class_counts) != {0, 1}:
        raise ValueError(f"{path}: both labels 0 and 1 are required")
    if min(class_counts.values()) < 2:
        raise ValueError(f"{path}: each label needs at least two records")

    return np.asarray(feature_rows, dtype=np.float64), np.asarray(labels, dtype=int)


def encode_features(features: np.ndarray) -> np.ndarray:
    """One-hot encode query_type while retaining the other raw features."""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(RAW_FEATURE_NAMES):
        raise ValueError(
            f"features must have shape (n, {len(RAW_FEATURE_NAMES)})"
        )

    query_type = array[:, 2].astype(int)
    if not np.all(array[:, 2] == query_type) or not np.all(
        np.isin(query_type, (0, 1, 2))
    ):
        raise ValueError("query_type values must be 0, 1, or 2")
    one_hot_query_type = np.eye(3, dtype=np.float64)[query_type]
    return np.column_stack(
        (array[:, 0], array[:, 1], array[:, 3], array[:, 4], one_hot_query_type)
    )


def predict_labels(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    """Predict binary labels from already encoded features."""
    scores = np.sum(features * model.coef_[0], axis=1) + model.intercept_[0]
    return np.where(scores > 0.0, model.classes_[1], model.classes_[0])


def coefficient_confidence_intervals(
    model: LogisticRegression,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    confidence: float = 0.95,
) -> np.ndarray:
    """Return approximate Wald intervals for coefficients and the intercept.

    The covariance matrix is the inverse observed information of the fitted,
    class-weighted model, including its L2 penalty.  The final row corresponds
    to the intercept.  Because these are intervals for a regularized estimate,
    they should be interpreted as approximate rather than classical maximum-
    likelihood confidence intervals.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    encoded = encode_features(features)
    target = np.asarray(labels, dtype=int)
    if target.ndim != 1 or len(target) != len(encoded):
        raise ValueError("labels must be a one-dimensional array matching features")

    scores = (
        np.einsum("ij,j->i", encoded, model.coef_[0], optimize=False)
        + model.intercept_[0]
    )
    probabilities = expit(scores)
    classes = np.asarray(model.classes_, dtype=int)
    class_weights = compute_class_weight("balanced", classes=classes, y=target)
    weight_by_class = dict(zip(classes, class_weights))
    sample_weights = np.asarray(
        [weight_by_class[label] for label in target], dtype=np.float64
    )

    # liblinear represents the intercept as a synthetic feature and therefore
    # regularizes it too (intercept_scaling is 1 by default).
    design = np.column_stack((encoded, np.ones(len(encoded), dtype=np.float64)))
    variance_weights = sample_weights * probabilities * (1.0 - probabilities)
    information = np.einsum(
        "ni,n,nj->ij", design, variance_weights, design, optimize=False
    )
    information += np.eye(design.shape[1], dtype=np.float64) / model.C
    covariance = np.linalg.inv(information)

    estimates = np.concatenate((model.coef_[0], model.intercept_))
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    critical_value = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    margin = critical_value * standard_errors
    return np.column_stack((estimates - margin, estimates + margin))


def cross_validate_and_train(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int,
    regularization: float,
    random_state: int,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Run stratified cross-validation, then fit a model on all records."""
    encoded = encode_features(features)
    class_counts = Counter(int(label) for label in labels)
    if min(class_counts.values()) < folds:
        raise ValueError(
            f"{folds}-fold cross-validation requires at least {folds} "
            "records of each label"
        )

    def new_model() -> LogisticRegression:
        return LogisticRegression(
            C=regularization,
            class_weight="balanced",
            max_iter=1_000,
            random_state=random_state,
            solver="liblinear",
        )

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    predictions = np.empty_like(labels)
    fold_scores: list[float] = []
    for train_indices, test_indices in splitter.split(encoded, labels):
        fold_model = new_model()
        fold_model.fit(encoded[train_indices], labels[train_indices])
        fold_predictions = predict_labels(fold_model, encoded[test_indices])
        predictions[test_indices] = fold_predictions
        fold_scores.append(
            balanced_accuracy_score(labels[test_indices], fold_predictions)
        )

    final_model = new_model()
    final_model.fit(encoded, labels)
    return final_model, predictions, np.asarray(fold_scores, dtype=np.float64)


def test_feature_associations(
    features: np.ndarray, labels: np.ndarray
) -> list[FeatureTestResult]:
    """Test each original feature for association with the target label.

    Pearson's chi-square test is used because all annotations are categorical;
    sparse 2x2 tables use Fisher's exact test instead. Holm adjustment controls
    the family-wise error rate, and Cramer's V reports association strength.
    """
    array = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=int)
    if array.ndim != 2 or array.shape[1] != len(RAW_FEATURE_NAMES):
        raise ValueError(
            f"features must have shape (n, {len(RAW_FEATURE_NAMES)})"
        )
    if target.ndim != 1 or len(target) != len(array):
        raise ValueError("labels must be a one-dimensional array matching features")
    if not np.all(np.isin(target, (0, 1))):
        raise ValueError("labels must contain only 0 and 1")

    unadjusted: list[
        tuple[str, str, str, float, int | None, float, float, tuple]
    ] = []
    for column, name in enumerate(ANNOTATION_FEATURES):
        values = array[:, column].astype(int)
        levels = np.unique(values)
        observed = np.asarray(
            [
                [
                    np.count_nonzero((values == level) & (target == label))
                    for label in (0, 1)
                ]
                for level in levels
            ],
            dtype=np.int64,
        )
        chi_square, chi_square_p, degrees_of_freedom, expected = chi2_contingency(
            observed, correction=False
        )
        if observed.shape == (2, 2) and np.any(expected < 5):
            statistic, p_value = fisher_exact(observed, alternative="two-sided")
            test_name = "Fisher's exact"
            statistic_name = "odds ratio"
            reported_degrees_of_freedom = None
        else:
            statistic, p_value = chi_square, chi_square_p
            test_name = "Pearson chi-square"
            statistic_name = "chi2"
            reported_degrees_of_freedom = int(degrees_of_freedom)
        denominator = len(target) * min(
            observed.shape[0] - 1, observed.shape[1] - 1
        )
        cramers_v = float(np.sqrt(chi_square / denominator)) if denominator else 0.0
        rates = tuple(
            (
                int(level),
                float(observed[row, 1] / observed[row].sum()),
                int(observed[row].sum()),
            )
            for row, level in enumerate(levels)
        )
        unadjusted.append(
            (
                name,
                test_name,
                statistic_name,
                float(statistic),
                reported_degrees_of_freedom,
                float(p_value),
                cramers_v,
                rates,
            )
        )

    # Holm's step-down adjustment. The cumulative maximum preserves the
    # required monotonicity of adjusted p-values in sorted order.
    adjusted = [0.0] * len(unadjusted)
    running_max = 0.0
    for rank, index in enumerate(
        sorted(range(len(unadjusted)), key=lambda item: unadjusted[item][5])
    ):
        candidate = (len(unadjusted) - rank) * unadjusted[index][5]
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)

    return [
        FeatureTestResult(
            name=name,
            test_name=test_name,
            statistic_name=statistic_name,
            statistic=statistic,
            degrees_of_freedom=degrees_of_freedom,
            p_value=p_value,
            adjusted_p_value=adjusted[index],
            cramers_v=cramers_v,
            label_one_rates=rates,
        )
        for index, (
            name,
            test_name,
            statistic_name,
            statistic,
            degrees_of_freedom,
            p_value,
            cramers_v,
            rates,
        ) in enumerate(unadjusted)
    ]


def describe_effect(cramers_v: float) -> str:
    """Return a conventional, coarse interpretation of Cramer's V."""
    if cramers_v < 0.1:
        return "negligible"
    if cramers_v < 0.3:
        return "small"
    if cramers_v < 0.5:
        return "medium"
    return "large"


def format_p_value(value: float) -> str:
    """Format tiny p-values without displaying floating-point underflow as zero."""
    return "<1e-300" if value == 0.0 else f"={value:.3g}"


def print_feature_tests(results: Sequence[FeatureTestResult], *, alpha: float) -> None:
    """Display per-feature tests with multiplicity-adjusted conclusions."""
    print("\nUnivariate feature association tests")
    print("Chi-square/Fisher tests; Holm-adjusted p-values across four features.")
    for result in results:
        conclusion = (
            "associated" if result.adjusted_p_value < alpha else "not significant"
        )
        rates = ", ".join(
            f"{level}: {rate:.1%} (n={count:,})"
            for level, rate, count in result.label_one_rates
        )
        print(f"\n{result.name}")
        statistic = f"{result.statistic_name}={result.statistic:.3f}"
        if result.degrees_of_freedom is not None:
            statistic = (
                f"{result.statistic_name}({result.degrees_of_freedom})="
                f"{result.statistic:.3f}"
            )
        print(
            f"  {result.test_name}: {statistic}, "
            f"p{format_p_value(result.p_value)}, "
            f"Holm p{format_p_value(result.adjusted_p_value)}"
        )
        print(
            f"  Cramer's V={result.cramers_v:.3f} "
            f"({describe_effect(result.cramers_v)}); {conclusion} at alpha={alpha:g}"
        )
        print(f"  label-1 rate by value: {rates}")


def print_results(
    model: LogisticRegression,
    features: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    fold_scores: np.ndarray,
) -> None:
    """Print class balance, cross-validation metrics, and final coefficients."""
    counts = Counter(int(label) for label in labels)
    classes = np.asarray(sorted(counts), dtype=int)
    weights = compute_class_weight("balanced", classes=classes, y=labels)

    print(f"Loaded {len(labels):,} records")
    print(
        "Label distribution: "
        + ", ".join(f"{label}={counts[label]:,}" for label in classes)
    )
    print(
        "Balanced class weights (full dataset): "
        + ", ".join(
            f"{label}={weight:.3f}" for label, weight in zip(classes, weights)
        )
    )
    print("Features: " + ", ".join(RAW_FEATURE_NAMES))
    print("Excluded columns: id, rank, query text (except word count)")
    score = balanced_accuracy_score(labels, predictions)
    print(
        f"\n{len(fold_scores)}-fold cross-validation balanced accuracy: "
        f"{fold_scores.mean():.3f} +/- {fold_scores.std(ddof=1):.3f}"
    )
    print(f"Out-of-fold balanced accuracy (all predictions): {score:.3f}")
    print("Confusion matrix (rows=true, columns=predicted; labels 0, 1):")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))
    print("\nOut-of-fold classification report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["label 0", "label 1"],
            digits=3,
            zero_division=0,
        )
    )
    intervals = coefficient_confidence_intervals(model, features, labels)
    estimates = np.concatenate((model.coef_[0], model.intercept_))
    names = (*MODEL_FEATURE_NAMES, "intercept")
    print(
        "Final model coefficients fitted on all records "
        "(positive favors label 1):"
    )
    print(f"  {'term':<30} {'coefficient':>12} {'95% CI':>25}")
    for name, coefficient, (lower, upper) in zip(names, estimates, intervals):
        print(
            f"  {name:<30} {coefficient:+12.6f} "
            f"[{lower:+.6f}, {upper:+.6f}]"
        )
    print("  Approximate Wald intervals include class weights and L2 regularization.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        features, labels = load_dataset(args.input)
        model, predictions, fold_scores = cross_validate_and_train(
            features,
            labels,
            folds=args.folds,
            regularization=args.regularization,
            random_state=args.random_state,
        )
        print_results(model, features, labels, predictions, fold_scores)
        print_feature_tests(test_feature_associations(features, labels), alpha=args.alpha)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
