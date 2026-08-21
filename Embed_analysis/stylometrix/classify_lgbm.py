#!/usr/bin/env python3
"""Evaluate a class-balanced LGBM classifier with 10-fold cross-validation."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate


DATA_PATH = Path(__file__).resolve().parent / "all_queries_stylometrix.json"


with DATA_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

X = np.asarray([item["feature_vector"] for item in data["queries"]], dtype=float)
y = np.asarray([item["label"] for item in data["queries"]], dtype=int)

# Compute class weights separately inside every training fold. This makes an
# error on the minority class more costly without duplicating observations or
# using information from a validation fold during training.
classifier = LGBMClassifier(
    class_weight="balanced", random_state=37, verbosity=-1
)
cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=37)

# Balanced accuracy is the mean of the recalls for labels 0 and 1, so neither
# class dominates the score because it has more observations. Keep both recalls
# visible as diagnostics for which side of the separation is weaker.
scoring = {
    "balanced_accuracy": "balanced_accuracy",
    "recall_label_0": make_scorer(recall_score, pos_label=0),
    "recall_label_1": "recall",
}
scores = cross_validate(classifier, X, y, cv=cross_validation, scoring=scoring)

label_counts = Counter(y)
print(
    "Label distribution: "
    + ", ".join(
        f"{label}={count:,} ({count / len(y):.1%})"
        for label, count in sorted(label_counts.items())
    )
)

for fold in range(cross_validation.n_splits):
    print(
        f"Fold {fold + 1}: "
        f"balanced_accuracy={scores['test_balanced_accuracy'][fold]:.4f}, "
        f"recall_0={scores['test_recall_label_0'][fold]:.4f}, "
        f"recall_1={scores['test_recall_label_1'][fold]:.4f}"
    )

print("\nMean cross-validation scores:")
for name in scoring:
    values = scores[f"test_{name}"]
    print(f"{name}: {values.mean():.4f} (+/- {values.std():.4f})")

"""
Mean cross-validation scores:
balanced_accuracy: 0.9098 (+/- 0.0290)
recall_label_0: 0.9745 (+/- 0.0051)
recall_label_1: 0.8450 (+/- 0.0579)
"""