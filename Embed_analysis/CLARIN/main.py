"""Run the CLARIN feature-extraction pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


# When this file is executed as ``python CLARIN/main.py``, Python adds the
# CLARIN directory (rather than the repository root) to sys.path. Add the root
# so the package's absolute imports work in both direct and module execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CLARIN.base import Pipeline

# Feature IDs to extract. Edit this list to change the feature selection.
FEATURES = [11, 20, 22, 30, 31, 40, 51, 60]
FEATURES = [20, 30, 31, 40, 50, 51, 52]

pipeline = Pipeline().run(features=FEATURES, show_explanation=True)
df = pipeline.features

print(df.shape)
print(df.head())

#print(pipeline.feature_names)
#print(pipeline.feature_names_waterfall)

print(df.dtypes)
print(df.describe())

result = pipeline.classification
print(f"Classifier: {result.classifier}")
print(f"Accuracy: {result.accuracy:.3f}")
print(f"Balanced accuracy: {result.balanced_accuracy:.3f}")
print("Confusion matrix (rows=true, columns=predicted):")
print(result.confusion_matrix)
print("\nMost influential features:")
print(pipeline.feature_importance.head(20).to_string(index=False))


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
