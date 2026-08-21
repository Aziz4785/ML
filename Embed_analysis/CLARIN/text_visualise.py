"""Visualise the CLARIN decision-tree query classifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from CLARIN.text_classify import TreeClassifier, shap_explanation


def plot_feature_importance(
    importance: pd.DataFrame,
    *,
    max_display: int = 20,
    show: bool = False,
    output_path: str | Path | None = None,
    title: str = "Feature importance and direction for hard-query predictions",
):
    """Plot tree-importance magnitude with SHAP-derived class direction."""
    required_columns = {"feature", "coefficient", "importance"}
    if not isinstance(importance, pd.DataFrame) or not required_columns.issubset(
        importance.columns
    ):
        raise ValueError(
            "importance must contain feature, coefficient, and importance columns"
        )
    if importance.empty:
        raise ValueError("importance must not be empty")
    if max_display < 1:
        raise ValueError("max_display must be at least 1")

    # Import lazily so loading the processing pipeline does not initialize a
    # plotting backend unless an explanation is actually requested.
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    displayed = importance.nlargest(max_display, "importance").sort_values("importance")
    colors = [
        "#d97706" if value > 0 else "#2563eb" if value < 0 else "#6b7280"
        for value in displayed["coefficient"]
    ]
    positions = range(len(displayed))

    height = max(4.0, 0.38 * len(displayed) + 1.5)
    figure, axis = plt.subplots(figsize=(10, height))
    axis.barh(positions, displayed["coefficient"], color=colors)
    axis.set_yticks(list(positions), displayed["feature"])
    axis.axvline(0, color="#374151", linewidth=0.8)
    axis.set_xlabel(
        "Signed tree importance (left = easy, right = hard)"
    )
    axis.set_ylabel("")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(
        handles=[
            Patch(color="#d97706", label="higher value → hard (label 1)"),
            Patch(color="#2563eb", label="higher value → easy (label 0)"),
        ],
        loc="best",
    )
    axis.text(
        0.5,
        -0.1,
        "Bar length is tree split importance; side and color come from SHAP "
        "direction on held-out queries.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(Path(output_path), bbox_inches="tight")
    if show:
        plt.show()
    return figure


def plot_shap(
    model: TreeClassifier,
    features: pd.DataFrame,
    feature_names: list[str],
    *,
    label: int = 1,
    max_display: int = 20,
    show: bool = False,
    output_path: str | Path | None = None,
):
    """Render a SHAP beeswarm showing feature impact for one class."""
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame")
    if features.shape[1] != model.n_features_in_:
        raise ValueError("features do not match the trained model")
    if len(feature_names) != model.n_features_in_:
        raise ValueError("feature_names do not match the trained model")
    if max_display < 1:
        raise ValueError("max_display must be at least 1")

    import shap
    from matplotlib import pyplot as plt

    # Use the same class-selection logic as the directional bar chart.
    explanation = shap_explanation(
        model,
        features,
        feature_names,
        label=label,
    )

    plt.figure(figsize=(12, max(6.0, 0.42 * min(max_display, features.shape[1]) + 2)))
    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    figure = plt.gcf()
    figure.axes[0].set_title(
        f"SHAP feature impact on hard-query predictions (label {label}) "
        "(right = toward class, left = away)"
    )
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(Path(output_path), dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure
