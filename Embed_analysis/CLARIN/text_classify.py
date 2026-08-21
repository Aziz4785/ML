"""Small, local classifier for the CLARIN query features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


ClassifierName = Literal["lgbm", "decision_tree"]
TreeClassifier = LGBMClassifier | DecisionTreeClassifier


@dataclass
class ClassificationResult:
    """Model and held-out predictions produced by :func:`classify_features`."""

    model: TreeClassifier
    classifier: ClassifierName
    train_indices: np.ndarray
    test_indices: np.ndarray
    true_labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    accuracy: float
    balanced_accuracy: float
    confusion_matrix: np.ndarray
    report: dict


def classify_features(
    features: pd.DataFrame,
    labels: list[int] | np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    classifier: ClassifierName = "lgbm",
) -> ClassificationResult:
    """Train a binary classifier and evaluate it on a stratified holdout set.

    Stratification preserves the label distribution in both partitions.  For
    example, with 700 label-1 rows and ``test_size=0.2``, approximately 560 are
    assigned to training and 140 to testing. LightGBM is used by default;
    ``classifier="decision_tree"`` retains the previous model for comparison.
    """
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(features):
        raise ValueError("labels must contain one value per feature row")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("labels must contain both binary values 0 and 1")
    if classifier not in {"lgbm", "decision_tree"}:
        raise ValueError("classifier must be 'lgbm' or 'decision_tree'")

    indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if classifier == "lgbm":
        model = LGBMClassifier(
            class_weight="balanced",
            learning_rate=0.05,
            n_estimators=200,
            num_leaves=15,
            min_child_samples=1,
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        )
    elif classifier == "decision_tree":
        model = DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=10,
            random_state=random_state,
        )
    # Generated n-gram column names can contain characters that LightGBM does
    # not accept in its internal JSON metadata. The names are retained by the
    # DataFrame for reporting, while the estimators receive numeric matrices.
    train_features = features.iloc[train_indices].to_numpy()
    test_features = features.iloc[test_indices].to_numpy()
    model.fit(train_features, y[train_indices])

    true_labels = y[test_indices]
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]

    return ClassificationResult(
        model=model,
        classifier=classifier,
        train_indices=train_indices,
        test_indices=test_indices,
        true_labels=true_labels,
        predictions=predictions,
        probabilities=probabilities,
        accuracy=float(accuracy_score(true_labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(true_labels, predictions)),
        confusion_matrix=confusion_matrix(true_labels, predictions, labels=[0, 1]),
        report=classification_report(
            true_labels,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        ),
    )


def feature_importance(
    result: ClassificationResult,
    features: pd.DataFrame,
    feature_names: list[str],
    *,
    label: int = 1,
) -> pd.DataFrame:
    """Return tree importance with a SHAP-derived direction for one class.

    Tree importance has no sign. The ``coefficient`` column therefore keeps its
    magnitude but receives a sign based on the association between each
    feature's value and its SHAP value on ``features``. Positive values mean
    that higher feature counts push toward ``label``; negative values mean that
    higher counts push away from it.
    """
    importances = result.model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError("feature_names do not match the trained model")
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame")
    if features.shape[1] != len(feature_names):
        raise ValueError("features do not match the trained model")

    explanation = shap_explanation(
        result.model,
        features,
        feature_names,
        label=label,
    )
    feature_values = features.to_numpy(dtype=float)
    centered_values = feature_values - feature_values.mean(axis=0)
    direction_scores = np.mean(centered_values * explanation.values, axis=0)
    signed_importances = importances * np.sign(direction_scores)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            # Retain the legacy column name for compatibility with callers.
            "coefficient": signed_importances,
            "importance": importances,
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def shap_explanation(
    model: TreeClassifier,
    features: pd.DataFrame,
    feature_names: list[str],
    *,
    label: int = 1,
):
    """Return a two-dimensional SHAP explanation for one model class."""
    if not isinstance(model, (LGBMClassifier, DecisionTreeClassifier)):
        raise TypeError("model must be a fitted LGBMClassifier or DecisionTreeClassifier")
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame")
    if features.shape[1] != model.n_features_in_:
        raise ValueError("features do not match the trained model")
    if len(feature_names) != model.n_features_in_:
        raise ValueError("feature_names do not match the trained model")

    # SHAP remains optional until an explanation is explicitly requested.
    import shap

    feature_values = features.to_numpy()
    if isinstance(model, LGBMClassifier):
        matching_classes = np.flatnonzero(model.classes_ == label)
        if not len(matching_classes):
            raise ValueError(f"label {label!r} is not present in the trained model")

        # LightGBM's native contribution prediction is its TreeSHAP
        # implementation. Besides being faster, it avoids a native-library
        # conflict between SHAP's LightGBM adapter and spaCy on macOS.
        contributions = model.predict(feature_values, pred_contrib=True)
        values = contributions[:, :-1]
        base_values = contributions[:, -1]
        if int(matching_classes[0]) == 0:
            values = -values
            base_values = -base_values
        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=feature_values,
            feature_names=feature_names,
        )

    explainer = shap.TreeExplainer(model, feature_names=feature_names)
    explanation = explainer(
        feature_values,
        check_additivity=False,
    )

    if explanation.values.ndim == 3:
        matching_classes = np.flatnonzero(model.classes_ == label)
        if not len(matching_classes):
            raise ValueError(f"label {label!r} is not present in the trained model")
        explanation = explanation[:, :, int(matching_classes[0])]
    elif explanation.values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {explanation.values.shape}")

    return explanation
