#!/usr/bin/env python3
"""Train and display a decision tree that predicts query difficulty labels.

The model is deliberately kept in memory: this script does not create a model,
checkpoint, plot, or predictions file.

Examples:
    python3 prediction/decisiontree.py
    python3 prediction/decisiontree.py --max-depth 4
    python3 prediction/decisiontree.py --no-show  # useful in CI/headless shells
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "all_queries.json"
ANNOTATION_FEATURES = (
    "query_about_length",
    "follow_template",
    "query_type",
    "subject_matter_unspecified",
)
FEATURE_NAMES = (*ANNOTATION_FEATURES, "query_word_count")


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
        "--test-size",
        type=float,
        default=0.2,
        help="fraction reserved for evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="maximum tree depth; kept small for a readable UI (default: 5)",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=10,
        help="minimum training rows in a leaf (default: 10)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="reproducible train/test split and fit seed (default: 42)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="train and evaluate without opening the matplotlib tree window",
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size must be between 0 and 1")
    if args.max_depth < 1:
        parser.error("--max-depth must be at least 1")
    if args.min_samples_leaf < 1:
        parser.error("--min-samples-leaf must be at least 1")
    return args


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate records, returning the feature matrix and labels."""
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

        # This cheap textual summary adds signal without creating a huge,
        # unreadable TF-IDF tree. ID and rank are intentionally excluded. In this
        # dataset rank directly reveals the label, which would leak the target.
        values.append(float(len(query.split())))
        feature_rows.append(values)
        labels.append(int(label))

    class_counts = Counter(labels)
    if set(class_counts) != {0, 1}:
        raise ValueError(f"{path}: both labels 0 and 1 are required")
    if min(class_counts.values()) < 2:
        raise ValueError(f"{path}: each label needs at least two records")

    return np.asarray(feature_rows, dtype=np.float64), np.asarray(labels, dtype=int)


def train_and_evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    test_size: float,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> tuple[DecisionTreeClassifier, np.ndarray, np.ndarray]:
    """Fit a class-weighted tree and return it with held-out truth/predictions."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    model = DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model, y_test, model.predict(x_test)


def print_results(
    model: DecisionTreeClassifier,
    labels: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """Print imbalance information, held-out metrics, and a textual tree."""
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
    print("Excluded leakage columns: id, rank")
    print(f"\nHeld-out balanced accuracy: {balanced_accuracy_score(y_test, predictions):.3f}")
    print("Confusion matrix (rows=true, columns=predicted; labels 0, 1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["label 0", "label 1"],
            digits=3,
            zero_division=0,
        )
    )
    print("Decision rules:")
    print(export_text(model, feature_names=list(FEATURE_NAMES)))


def show_tree(model: DecisionTreeClassifier) -> None:
    """Open the fitted tree in a matplotlib UI without writing it to disk."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for the tree UI; install requirements.txt "
            "or run with --no-show"
        ) from exc

    width = min(32.0, max(14.0, 2.0 * (2 ** min(model.get_depth(), 5))))
    figure, axis = plt.subplots(figsize=(width, 10))
    plot_tree(
        model,
        feature_names=FEATURE_NAMES,
        class_names=("label 0", "label 1"),
        filled=True,
        rounded=True,
        proportion=True,
        precision=3,
        fontsize=8,
        ax=axis,
    )
    axis.set_title("Class-weighted decision tree for query labels")
    figure.tight_layout()
    plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        features, labels = load_dataset(args.input)
        model, y_test, predictions = train_and_evaluate(
            features,
            labels,
            test_size=args.test_size,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
        )
        print_results(model, labels, y_test, predictions)
        if not args.no_show:
            show_tree(model)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
