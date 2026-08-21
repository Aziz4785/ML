"""Find the best non-empty combination of candidate CLARIN feature families.

The expensive preprocessing and feature extraction stages run only once.  The
resulting columns are then sliced for every candidate combination before the
classifier is trained.  No explanations, SHAP values, or plots are produced.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd


# Support both ``python CLARIN/search_feature_combinations.py`` and
# ``python -m CLARIN.search_feature_combinations``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CLARIN.base import Pipeline
from CLARIN.feature_extraction import VALID_FEATURES
from CLARIN.text_classify import classify_features


# Every non-empty subset of these feature families will be evaluated.
FEATURES = [10,11,13, 20, 22,30,31,40,50,51, 52, 60]
#21 and 12 and 23 are not good


def feature_combinations(features: list[int]):
    """Yield all non-empty subsets, from the smallest to the largest."""
    for size in range(1, len(features) + 1):
        yield from combinations(features, size)


def columns_for_features(columns, features: tuple[int, ...]) -> list[str]:
    """Select extracted columns whose final suffix is a requested feature ID."""
    suffixes = tuple(f"_{feature_id}" for feature_id in features)
    return [column for column in columns if str(column).endswith(suffixes)]


def validate_candidates(features: list[int]) -> None:
    """Reject invalid or duplicate candidate IDs before expensive work starts."""
    invalid = sorted(set(features) - VALID_FEATURES)
    if invalid:
        raise ValueError(
            f"Invalid feature IDs: {invalid}. Valid IDs are: {sorted(VALID_FEATURES)}"
        )
    if not features:
        raise ValueError("FEATURES must contain at least one feature ID")
    if len(features) != len(set(features)):
        raise ValueError("FEATURES must not contain duplicate feature IDs")


def main() -> None:
    validate_candidates(FEATURES)

    pipeline = Pipeline()
    pipeline.texts_load()
    pipeline.texts_preprocess()
    all_features = pipeline.extract_features(features=FEATURES)

    labels = [record.get(pipeline.target_field) for record in pipeline.records]
    if any(isinstance(label, bool) or label not in (0, 1) for label in labels):
        raise ValueError(
            f"Every record must contain a binary integer {pipeline.target_field!r} field"
        )
    labels = [int(label) for label in labels]

    rows: list[dict] = []
    candidate_combinations = list(feature_combinations(FEATURES))
    print(f"\nTesting {len(candidate_combinations)} feature combinations (no plots)...")

    for index, combination in enumerate(candidate_combinations, start=1):
        selected_columns = columns_for_features(all_features.columns, combination)
        if not selected_columns:
            print(
                f"[{index:>2}/{len(candidate_combinations)}] {combination}: "
                "skipped (no columns)"
            )
            continue

        result = classify_features(
            all_features[selected_columns],
            labels,
            test_size=pipeline.test_size,
            random_state=pipeline.random_state,
        )
        rows.append(
            {
                "features": combination,
                "feature_family_count": len(combination),
                "column_count": len(selected_columns),
                "label_1_recall": result.report["1"]["recall"],
                "accuracy": result.accuracy,
                "balanced_accuracy": result.balanced_accuracy,
                "confusion_matrix": result.confusion_matrix,
            }
        )
        print(
            f"[{index:>2}/{len(candidate_combinations)}] {combination}: "
            f"label_1_recall={result.report['1']['recall']:.3f}, "
            f"balanced_accuracy={result.balanced_accuracy:.3f}, "
            f"accuracy={result.accuracy:.3f}"
        )

    if not rows:
        raise RuntimeError("None of the candidate combinations produced feature columns")

    # Label-1 recall is the primary criterion: among records whose true label is
    # 1, it measures the fraction predicted as 1. For an exact tie, prefer the
    # simpler combination, then balanced and ordinary accuracy, then ID order.
    ranking = pd.DataFrame(rows).sort_values(
        by=[
            "label_1_recall",
            "feature_family_count",
            "balanced_accuracy",
            "accuracy",
            "features",
        ],
        ascending=[False, True, False, False, True],
        ignore_index=True,
    )
    best = ranking.iloc[0]

    print("\nTop 10 combinations:")
    print(
        ranking.drop(columns="confusion_matrix")
        .head(10)
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nBest feature combination:", list(best["features"]))
    print(f"Label-1 recall: {best['label_1_recall']:.3f}")
    print(f"Balanced accuracy: {best['balanced_accuracy']:.3f}")
    print(f"Accuracy: {best['accuracy']:.3f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(best["confusion_matrix"])


if __name__ == "__main__":
    main()

#best combination when depth = None
"""
Best feature combination: [13, 20, 30, 52]
Balanced accuracy: 0.851


Best feature combination: [10, 11, 13, 20, 30, 40, 52]
Balanced accuracy: 0.875
"""
#max depth 6:
"""
 [11, 31, 40, 52]
 Balanced accuracy: 0.878
"""
#max depth 7:
"""
Best feature combination: [11, 20, 30, 31, 40, 52, 60]
Balanced accuracy: 0.887
"""
#max depth 8:
"""
Best feature combination: [11, 22, 30, 40, 51]
Balanced accuracy: 0.898
"""

#max depth 9:
"""
Best feature combination: [11, 20, 22, 30, 31, 40, 51, 60]
Balanced accuracy: 0.904
"""

#max depth 10:
"""
Best feature combination: [10, 11, 13, 22, 30, 40, 50, 51, 52]
Balanced accuracy: 0.898
"""

#lgbm: Best feature combination: [13, 20, 51, 60]
#Balanced accuracy: 0.944

"""
Best feature combination: [20, 30, 31, 40, 50, 51, 52]
Label-1 recall: 0.964
"""