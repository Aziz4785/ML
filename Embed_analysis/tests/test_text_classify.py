import pandas as pd
import spacy
from lightgbm import LGBMClassifier
from matplotlib.figure import Figure

from CLARIN.base import QueryPipeline
from CLARIN.text_classify import classify_features, feature_importance


def test_pipeline_extract_features_accepts_per_call_feature_ids():
    pipeline = QueryPipeline(verbose=2)
    pipeline.docs = [spacy.blank("en")("A cat sat")]

    features = pipeline.extract_features(features=[10])

    assert not features.empty
    assert all(column.endswith("_10") for column in features.columns)


def test_classify_features_returns_holdout_metrics_and_importance():
    features = pd.DataFrame(
        {
            "easy_10": [4, 3, 4, 3, 0, 0, 1, 0],
            "hard_10": [0, 0, 1, 0, 3, 4, 3, 4],
        }
    )
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    result = classify_features(features, labels, test_size=0.5, random_state=1)
    held_out_features = features.iloc[result.test_indices]
    importance = feature_importance(
        result,
        held_out_features,
        features.columns.tolist(),
    )

    assert len(result.predictions) == 4
    assert result.classifier == "lgbm"
    assert isinstance(result.model, LGBMClassifier)
    assert result.confusion_matrix.shape == (2, 2)
    assert 0.0 <= result.balanced_accuracy <= 1.0
    assert set(importance.columns) == {"feature", "coefficient", "importance"}


def test_classify_features_can_still_use_decision_tree():
    features = pd.DataFrame({"value": [0, 1, 2, 3, 6, 7, 8, 9]})
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    result = classify_features(
        features,
        labels,
        test_size=0.5,
        random_state=1,
        classifier="decision_tree",
    )

    assert result.classifier == "decision_tree"


def test_classify_features_stratifies_hard_queries_between_both_sets():
    labels = [0] * 300 + [1] * 700
    features = pd.DataFrame({"value": range(len(labels))})

    result = classify_features(features, labels, test_size=0.2, random_state=1)

    train_labels = pd.Series(labels).iloc[result.train_indices]
    test_labels = pd.Series(labels).iloc[result.test_indices]
    assert (train_labels == 1).sum() == 560
    assert (test_labels == 1).sum() == 140


def test_feature_importance_direction_points_higher_easy_feature_left():
    features = pd.DataFrame(
        {
            "answer_13": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    result = classify_features(features, labels, test_size=0.5, random_state=1)
    held_out_features = features.iloc[result.test_indices]
    importance = feature_importance(
        result,
        held_out_features,
        ["answer"],
    )

    assert importance.loc[0, "importance"] > 0
    assert importance.loc[0, "coefficient"] < 0


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
    assert set(importance["feature"]) == {"easy_10", "hard_10"}
    assert isinstance(pipeline.explanation_figure, Figure)
