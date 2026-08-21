#!/usr/bin/env python3
"""Plot two- and three-dimensional query embeddings from JSON files.

Each input must be a non-empty JSON array whose records contain an
``embedding`` field with the expected number of finite numeric values. If
every record also contains a cluster field, points are colored by cluster;
otherwise clusters are discovered with HDBSCAN by default.

Examples:
    python3 analysis/visualize_cluster.py

    python3 analysis/visualize_cluster.py input.json output.png \
        --cluster-field cluster_id --title "Query clusters"

    python3 analysis/visualize_cluster.py --no-show
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "hard_queries_embedded_umap_2d.json"
DEFAULT_OUTPUT = HERE / "hard_queries_embedded_umap_2d.png"
DEFAULT_INPUT_3D = HERE / "hard_queries_embedded_umap_3d.json"
DEFAULT_OUTPUT_3D = HERE / "hard_queries_embedded_umap_3d.png"
DEFAULT_CLUSTER_FIELDS = ("cluster_id", "cluster", "label")
DEFAULT_MIN_CLUSTER_SIZE = 10


def load_records(
    path: Path, dimensions: int
) -> tuple[list[dict[str, Any]], list[tuple[float, ...]]]:
    """Load records from *path* and validate their embeddings."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise ValueError("Input JSON must be a non-empty top-level array.")

    points: list[tuple[float, ...]] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a JSON object.")

        embedding = record.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != dimensions:
            actual_dimensions = len(embedding) if isinstance(embedding, list) else 0
            raise ValueError(
                f"Record {index} has {actual_dimensions} embedding dimensions; "
                f"expected {dimensions}."
            )

        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in embedding
        ):
            raise ValueError(f"Record {index} contains a non-numeric embedding value.")

        point = tuple(float(value) for value in embedding)
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"Record {index} contains a non-finite embedding value.")
        points.append(point)

    return data, points


def get_cluster_labels(
    records: Sequence[dict[str, Any]], cluster_field: str | None
) -> tuple[str | None, list[Any] | None]:
    """Return a usable cluster field and its labels.

    When no field is requested, a common cluster field is detected only when
    it is present in every record.  An explicitly requested missing or invalid
    field is reported as an error instead of silently producing an uncolored
    plot.
    """
    selected_field = cluster_field
    if selected_field is None:
        selected_field = next(
            (
                field
                for field in DEFAULT_CLUSTER_FIELDS
                if all(field in record for record in records)
            ),
            None,
        )

    if selected_field is None:
        return None, None

    labels: list[Any] = []
    for index, record in enumerate(records):
        if selected_field not in record:
            raise ValueError(
                f"Record {index} does not contain cluster field {selected_field!r}."
            )
        label = record[selected_field]
        if label is None or isinstance(label, (dict, list)):
            raise ValueError(
                f"Record {index} has an invalid value for cluster field "
                f"{selected_field!r}."
            )
        labels.append(label)

    return selected_field, labels


def cluster_points(
    points: Sequence[tuple[float, ...]],
    *,
    min_cluster_size: int,
    min_samples: int | None,
) -> list[int]:
    """Discover clusters in embedding points with HDBSCAN."""
    if min_cluster_size < 2:
        raise ValueError("--min-cluster-size must be at least 2.")
    if min_cluster_size > len(points):
        raise ValueError(
            "--min-cluster-size cannot be greater than the number of points."
        )
    if min_samples is not None and min_samples < 1:
        raise ValueError("--min-samples must be at least 1.")

    try:
        import hdbscan
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'hdbscan'. Install it with: "
            "python3 -m pip install hdbscan"
        ) from exc

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return [int(label) for label in model.fit_predict(np.asarray(points))]


def _label_sort_key(label: Any) -> tuple[int, float | str]:
    """Sort numeric labels numerically and all other labels by display text."""
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        return (0, float(label))
    return (1, str(label))


def plot_points(
    points: Sequence[tuple[float, float]],
    output_path: Path,
    *,
    labels: Sequence[Any] | None = None,
    hover_labels: Sequence[str] | None = None,
    cluster_field: str | None = None,
    title: str = "2D query embedding",
    point_size: float = 22.0,
    alpha: float = 0.75,
    dpi: int = 180,
    show: bool = True,
) -> None:
    """Create, save, and optionally display a scatter plot of *points*."""
    if point_size <= 0:
        raise ValueError("--point-size must be greater than zero.")
    if not 0 < alpha <= 1:
        raise ValueError("--alpha must be greater than zero and at most one.")
    if dpi <= 0:
        raise ValueError("--dpi must be greater than zero.")
    if labels is not None and len(labels) != len(points):
        raise ValueError("The number of cluster labels must match the number of points.")
    if hover_labels is not None and len(hover_labels) != len(points):
        raise ValueError("The number of hover labels must match the number of points.")

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'matplotlib'. Install it with: "
            "python3 -m pip install matplotlib"
        ) from exc

    figure, axis = plt.subplots(figsize=(10, 7))
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    plotted_groups: list[tuple[Any, list[int]]] = []

    if labels is None:
        collection = axis.scatter(
            x_values,
            y_values,
            s=point_size,
            alpha=alpha,
            color="#2563eb",
            edgecolors="none",
        )
        plotted_groups.append((collection, list(range(len(points)))))
    else:
        unique_labels = sorted(set(labels), key=_label_sort_key)
        color_map = plt.get_cmap("tab20", max(len(unique_labels), 1))
        for color_index, label in enumerate(unique_labels):
            indices = [index for index, value in enumerate(labels) if value == label]
            color = "#9ca3af" if label == -1 else color_map(color_index)
            display_label = "noise (-1)" if label == -1 else str(label)
            collection = axis.scatter(
                [x_values[index] for index in indices],
                [y_values[index] for index in indices],
                s=point_size,
                alpha=alpha,
                color=color,
                edgecolors="none",
                label=display_label,
            )
            plotted_groups.append((collection, indices))
        axis.legend(
            title=cluster_field or "cluster",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=False,
        )

    axis.set_title(title)
    axis.set_xlabel("Embedding dimension 1")
    axis.set_ylabel("Embedding dimension 2")
    axis.grid(color="#d1d5db", linewidth=0.6, alpha=0.5)
    axis.set_axisbelow(True)

    if hover_labels is not None:
        annotation = axis.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "#4b5563"},
            fontsize=9,
            zorder=10,
        )
        annotation.set_visible(False)

        def update_hover(event: Any) -> None:
            if event.inaxes is not axis:
                if annotation.get_visible():
                    annotation.set_visible(False)
                    figure.canvas.draw_idle()
                return

            for collection, original_indices in plotted_groups:
                contains, details = collection.contains(event)
                if contains and len(details.get("ind", [])) > 0:
                    local_index = int(details["ind"][0])
                    point_index = original_indices[local_index]
                    annotation.xy = points[point_index]
                    annotation.set_text(
                        textwrap.fill(str(hover_labels[point_index]), width=55)
                    )
                    annotation.set_visible(True)
                    figure.canvas.draw_idle()
                    return

            if annotation.get_visible():
                annotation.set_visible(False)
                figure.canvas.draw_idle()

        figure.canvas.mpl_connect("motion_notify_event", update_hover)

    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_points_3d(
    points: Sequence[tuple[float, ...]],
    output_path: Path,
    *,
    labels: Sequence[Any] | None = None,
    hover_labels: Sequence[str] | None = None,
    cluster_field: str | None = None,
    title: str = "3D query embedding",
    point_size: float = 22.0,
    alpha: float = 0.75,
    dpi: int = 180,
    show: bool = True,
) -> None:
    """Create, save, and optionally display a 3D scatter plot of *points*."""
    if point_size <= 0:
        raise ValueError("--point-size must be greater than zero.")
    if not 0 < alpha <= 1:
        raise ValueError("--alpha must be greater than zero and at most one.")
    if dpi <= 0:
        raise ValueError("--dpi must be greater than zero.")
    if any(len(point) != 3 for point in points):
        raise ValueError("Every point in the 3D plot must have three dimensions.")
    if labels is not None and len(labels) != len(points):
        raise ValueError("The number of cluster labels must match the number of points.")
    if hover_labels is not None and len(hover_labels) != len(points):
        raise ValueError("The number of hover labels must match the number of points.")

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'matplotlib'. Install it with: "
            "python3 -m pip install matplotlib"
        ) from exc

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    plotted_groups: list[tuple[Any, list[int]]] = []

    def scatter(indices: list[int], **kwargs: Any) -> Any:
        return axis.scatter(
            [points[index][0] for index in indices],
            [points[index][1] for index in indices],
            [points[index][2] for index in indices],
            s=point_size,
            alpha=alpha,
            depthshade=True,
            **kwargs,
        )

    if labels is None:
        indices = list(range(len(points)))
        collection = scatter(indices, color="#2563eb", edgecolors="none")
        plotted_groups.append((collection, indices))
    else:
        unique_labels = sorted(set(labels), key=_label_sort_key)
        color_map = plt.get_cmap("tab20", max(len(unique_labels), 1))
        for color_index, label in enumerate(unique_labels):
            indices = [index for index, value in enumerate(labels) if value == label]
            color = "#9ca3af" if label == -1 else color_map(color_index)
            display_label = "noise (-1)" if label == -1 else str(label)
            collection = scatter(
                indices,
                color=color,
                edgecolors="none",
                label=display_label,
            )
            plotted_groups.append((collection, indices))
        axis.legend(
            title=cluster_field or "cluster",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=False,
        )

    axis.set_title(title)
    axis.set_xlabel("Embedding dimension 1")
    axis.set_ylabel("Embedding dimension 2")
    axis.set_zlabel("Embedding dimension 3")
    axis.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.5)

    if hover_labels is not None:
        annotation = axis.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "#4b5563"},
            fontsize=9,
            zorder=10,
        )
        annotation.set_visible(False)

        def update_hover(event: Any) -> None:
            if event.inaxes is not axis:
                if annotation.get_visible():
                    annotation.set_visible(False)
                    figure.canvas.draw_idle()
                return

            for collection, original_indices in plotted_groups:
                contains, details = collection.contains(event)
                if contains and len(details.get("ind", [])) > 0:
                    local_index = int(details["ind"][0])
                    point_index = original_indices[local_index]
                    annotation.xy = (event.xdata, event.ydata)
                    annotation.set_text(
                        textwrap.fill(str(hover_labels[point_index]), width=55)
                    )
                    annotation.set_visible(True)
                    figure.canvas.draw_idle()
                    return

            if annotation.get_visible():
                annotation.set_visible(False)
                figure.canvas.draw_idle()

        figure.canvas.mpl_connect("motion_notify_event", update_hover)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 2D and 3D embeddings from JSON as scatter plots."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Input JSON file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Output image file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--input-3d",
        type=Path,
        default=DEFAULT_INPUT_3D,
        help=f"3D input JSON file. Default: {DEFAULT_INPUT_3D}",
    )
    parser.add_argument(
        "--output-3d",
        type=Path,
        default=DEFAULT_OUTPUT_3D,
        help=f"3D output image file. Default: {DEFAULT_OUTPUT_3D}",
    )
    parser.add_argument(
        "--cluster-field",
        help=(
            "Record field containing cluster labels. If omitted, cluster_id, "
            "cluster, and label are detected automatically."
        ),
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help=(
            "Minimum HDBSCAN cluster size when labels are absent. "
            f"Default: {DEFAULT_MIN_CLUSTER_SIZE}."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN density threshold; defaults to the minimum cluster size.",
    )
    parser.add_argument(
        "--no-auto-cluster",
        action="store_true",
        help="Do not discover clusters when the input has no cluster labels.",
    )
    parser.add_argument("--title", default="2D query embedding", help="Plot title.")
    parser.add_argument(
        "--title-3d", default="3D query embedding", help="3D plot title."
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=22.0,
        help="Scatter point size. Default: 22.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Point opacity from greater than 0 through 1. Default: 0.75.",
    )
    parser.add_argument(
        "--dpi", type=int, default=180, help="Saved image resolution. Default: 180."
    )
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Open the interactive plot window (the default).",
    )
    display_group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Only save the image without opening a window.",
    )
    parser.set_defaults(show=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    records, points = load_records(args.input, dimensions=2)
    cluster_field, labels = get_cluster_labels(records, args.cluster_field)
    if labels is None and not args.no_auto_cluster:
        labels = cluster_points(
            points,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
        cluster_field = "HDBSCAN cluster"

    hover_labels = [
        (
            f"Cluster {labels[index]}\n"
            f"{record.get('query') or record.get('id') or f'Point {index}'}"
            if labels is not None
            else str(record.get("query") or record.get("id") or f"Point {index}")
        )
        for index, record in enumerate(records)
    ]
    plot_points(
        points,
        args.output,
        labels=labels,
        hover_labels=hover_labels,
        cluster_field=cluster_field,
        title=args.title,
        point_size=args.point_size,
        alpha=args.alpha,
        dpi=args.dpi,
        show=args.show,
    )

    cluster_message = (
        f" colored by {cluster_field!r}" if cluster_field is not None else ""
    )
    print(f"Saved {len(points):,} points{cluster_message} to {args.output}")

    records_3d, points_3d = load_records(args.input_3d, dimensions=3)
    cluster_field_3d, labels_3d = get_cluster_labels(records_3d, args.cluster_field)
    if labels_3d is None and not args.no_auto_cluster:
        labels_3d = cluster_points(
            points_3d,
            min_cluster_size=5,
            min_samples=args.min_samples,
        )
        cluster_field_3d = "HDBSCAN cluster"

    hover_labels_3d = [
        (
            f"Cluster {labels_3d[index]}\n"
            f"{record.get('query') or record.get('id') or f'Point {index}'}"
            if labels_3d is not None
            else str(record.get("query") or record.get("id") or f"Point {index}")
        )
        for index, record in enumerate(records_3d)
    ]
    plot_points_3d(
        points_3d,
        args.output_3d,
        labels=labels_3d,
        hover_labels=hover_labels_3d,
        cluster_field=cluster_field_3d,
        title=args.title_3d,
        point_size=args.point_size,
        alpha=args.alpha,
        dpi=args.dpi,
        show=args.show,
    )

    cluster_message_3d = (
        f" colored by {cluster_field_3d!r}" if cluster_field_3d is not None else ""
    )
    print(
        f"Saved {len(points_3d):,} points{cluster_message_3d} "
        f"to {args.output_3d}"
    )


if __name__ == "__main__":
    main()
