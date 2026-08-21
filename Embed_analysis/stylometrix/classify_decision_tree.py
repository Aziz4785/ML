#!/usr/bin/env python3
"""Evaluate and explain a class-balanced Decision Tree."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree


DATA_PATH = Path(__file__).resolve().parent / "all_queries_stylometrix.json"


with DATA_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

X = np.asarray([item["feature_vector"] for item in data["queries"]], dtype=float)
y = np.asarray([item["label"] for item in data["queries"]], dtype=int)

# Compute class weights separately inside every training fold. This makes an
# error on the minority class more costly without duplicating observations or
# using information from a validation fold during training.
classifier = DecisionTreeClassifier(
    class_weight="balanced", max_depth=6, min_samples_split=2, random_state=42
)
cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

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

# Cross-validation fits copies of the estimator, so fit the original classifier
# on all available data before drawing it.
classifier.fit(X, y)

top_feature_indices = np.argsort(classifier.feature_importances_)[::-1][:4]
print("\nTop 4 most important features:")
for rank, feature_index in enumerate(top_feature_indices, start=1):
    print(
        f"{rank}. {data['feature_names'][feature_index]}: "
        f"{classifier.feature_importances_[feature_index]:.4f}"
    )

figure, axis = plt.subplots(figsize=(32, 18))
plot_tree(
    classifier,
    feature_names=data["feature_names"],
    class_names=[str(label) for label in classifier.classes_],
    filled=True,
    rounded=True,
    fontsize=7,
    ax=axis,
)
axis.set_title("Decision Tree Trained on All Stylometric Queries")
figure.tight_layout()

tree_image_path = Path(__file__).resolve().parent / "decision_tree.png"
figure.savefig(tree_image_path, dpi=200, bbox_inches="tight")
print(f"\nDecision tree visualization saved to: {tree_image_path}")

# Explain the final model fitted on all observations. For a classifier, SHAP
# returns one explanation per class; focus on label 1 so positive SHAP values
# consistently mean "pushes the model toward label 1".
explainer = shap.TreeExplainer(
    classifier,
    feature_names=data["feature_names"],
)
shap_values = explainer(X)
label_to_explain = 1 if 1 in classifier.classes_ else classifier.classes_[-1]

if shap_values.values.ndim == 3:
    class_index = int(np.flatnonzero(classifier.classes_ == label_to_explain)[0])
    label_shap_values = shap_values[:, :, class_index]
elif shap_values.values.ndim == 2:
    # Some SHAP/model combinations return only the positive-class explanation.
    label_shap_values = shap_values
else:
    raise ValueError(
        f"Unexpected SHAP value shape: {shap_values.values.shape}"
    )

mean_absolute_shap = np.abs(label_shap_values.values).mean(axis=0)
top_shap_indices = np.argsort(mean_absolute_shap)[::-1][:10]
top_shap_values = label_shap_values[:, top_shap_indices]
print(f"\nTop features by mean absolute SHAP value for label {label_to_explain}:")
for rank, feature_index in enumerate(top_shap_indices, start=1):
    print(
        f"{rank}. {data['feature_names'][feature_index]}: "
        f"{mean_absolute_shap[feature_index]:.6f}"
    )

# The bar chart answers "which features matter most overall?". The beeswarm
# additionally shows direction: points to the right push toward label 1, while
# points to the left push away from it; color represents the feature value.
plt.figure(figsize=(12, 9))
shap.plots.bar(top_shap_values, max_display=10, show=False)
plt.title(f"Global SHAP Feature Importance for Label {label_to_explain}")
plt.tight_layout()
shap_bar_path = Path(__file__).resolve().parent / "shap_feature_importance.png"
plt.savefig(shap_bar_path, dpi=200, bbox_inches="tight")
print(f"SHAP feature-importance plot saved to: {shap_bar_path}")

plt.figure(figsize=(12, 9))
shap.plots.beeswarm(
    top_shap_values,
    max_display=10,
    show=False,
    plot_size=None,
)
plt.title(f"SHAP Feature Impact on Predictions for Label {label_to_explain}")
plt.tight_layout()
shap_beeswarm_path = Path(__file__).resolve().parent / "shap_beeswarm.png"
plt.savefig(shap_beeswarm_path, dpi=200, bbox_inches="tight")
print(f"SHAP beeswarm plot saved to: {shap_beeswarm_path}")

plt.show()

"""

Mean cross-validation scores:
balanced_accuracy: 0.9002 (+/- 0.0193)
recall_label_0: 0.9028 (+/- 0.0128)
recall_label_1: 0.8976 (+/- 0.0304)

balanced_accuracy = (percentage of 0s correctly classified + percentage of 1s correctly classified) / 2

Top 4 most important features:
1. L_PUNCT_COL: 0.6669 (Frequency of the colon : in the text)
2. L_PROPER_NAME: 0.1048 (Frequency of proper nouns/names such as London, Microsoft, France, John, etc.)
3. QAS: 0.0610 (Vocabulary with potential negative connotations)
4. L_PUNCT: 0.0498 (Frequency of punctuation marks in general)


Top features by mean absolute SHAP value for label 1:
1. L_PUNCT_COL: 0.315540 
2. L_PROPER_NAME: 0.084063
3. QAS: 0.076970
4. L_PUNCT_COM: 0.028197 (Frequency of commas relative to text length)
5. L_PUNCT: 0.025971
6. ST_SENT_WRDSPERSENT: 0.022647 (sentence length)
7. ST_TYPE_TOKEN_RATIO_LEMMAS: 0.020005 (Ratio of unique lemmas to total text length)


conclusion:
because low values of L_PUNCT_COL, L_PROPER_NAME, and QAS tend to push the model 
toward Label 1 we can say that Hard queries are characterized by 
fewer colons, 
fewer proper nouns or named entities, => queries are less specific and less about the content of the text
and less vocabulary with potentially negative connotations, => less specific

At the same time, Label 1 is associated with more punctuation overall (L_PUNCT),
 slightly more commas (L_PUNCT_COM), 
 and greater lexical diversity (ST_TYPE_TOKEN_RATIO_LEMMAS).
"""
