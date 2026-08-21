import pandas as pd

from CLARIN.base import QueryPipeline
from CLARIN.text_classify import classify_features, feature_importance


def test_classify_features_returns_holdout_metrics_and_importance():
    features = pd.DataFrame(
        {
            "easy_10": [4, 3, 4, 3, 0, 0, 1, 0],
            "hard_10": [0, 0, 1, 0, 3, 4, 3, 4],
        }
    )
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    result = classify_features(features, labels, test_size=0.5, random_state=1)
    importance = feature_importance(result, features.columns.tolist())

    assert len(result.predictions) == 4
    assert result.confusion_matrix.shape == (2, 2)
    assert 0.0 <= result.balanced_accuracy <= 1.0
    assert set(importance.columns) == {"feature", "coefficient", "importance"}


def test_pipeline_classify_and_explain_use_record_labels():
    pipeline = QueryPipeline(test_size=0.5, random_state=1)
    pipeline.records = [
        {"query": str(index), "label": label}
        for index, label in enumerate([0, 0, 0, 0, 1, 1, 1, 1])
    ]
    pipeline.features = pd.DataFrame(
        {
            "easy_10": [4, 3, 4, 3, 0, 0, 1, 0],
            "hard_10": [0, 0, 1, 0, 3, 4, 3, 4],
        }
    )
    pipeline.feature_names = pipeline.features.columns.tolist()
    pipeline.feature_names_waterfall = ["easy", "hard"]

    result = pipeline.classify()
    importance = pipeline.explain()

    assert pipeline.labels == [0, 0, 0, 0, 1, 1, 1, 1]
    assert result is pipeline.classification
    assert set(importance["feature"]) == {"easy", "hard"}
