import pandas as pd
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeClassifier

from CLARIN.text_visualise import plot_feature_importance, plot_shap


def test_plot_feature_importance_builds_coefficient_chart(tmp_path):
    importance = pd.DataFrame(
        {
            "feature": ["hard phrase", "easy phrase", "neutral"],
            "coefficient": [1.2, -0.8, 0.1],
            "importance": [1.2, 0.8, 0.1],
        }
    )
    output_path = tmp_path / "importance.png"

    figure = plot_feature_importance(
        importance,
        max_display=2,
        output_path=output_path,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes[0].patches) == 2
    assert sorted(patch.get_width() for patch in figure.axes[0].patches) == [-0.8, 1.2]
    assert "left = easy, right = hard" in figure.axes[0].get_xlabel()
    assert figure.axes[0].get_legend() is not None
    assert output_path.exists()


def test_plot_shap_builds_beeswarm_for_hard_label(tmp_path):
    features = pd.DataFrame(
        {
            "question words": [0, 1, 0, 2, 0, 3, 1, 2],
            "proper nouns": [3, 0, 2, 0, 4, 0, 1, 0],
            "punctuation": [0, 2, 1, 3, 0, 4, 1, 3],
        }
    )
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(
        features, labels
    )
    output_path = tmp_path / "shap.png"

    figure = plot_shap(
        model,
        features,
        list(features.columns),
        output_path=output_path,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) >= 1
    assert output_path.exists()
