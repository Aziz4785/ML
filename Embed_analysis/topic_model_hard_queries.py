#!/usr/bin/env python3
"""Cluster ``hard_queries.json`` with the requested BERTopic pipeline.

The full pipeline is:

1. Validate English-tagged records when a language field is supplied and remove
   normalized duplicate prompts.
2. Embed prompts with SentenceTransformers ``all-mpnet-base-v2``.
3. Reduce the embeddings to five dimensions with UMAP.
4. Cluster the reduced vectors with HDBSCAN (minimum cluster size 20).
5. Select representative prompts from the top HDBSCAN-probability fraction of
   every original cluster, subject to the requested word limit.
6. Optionally ask GPT-4o for a narrow topic name and description.
7. Reassign HDBSCAN outliers first by soft-clustering probabilities and then by
   embedding similarity, updating BERTopic's final topic representations.

Examples:
    python3 topic_model_hard_queries.py --validate-only
    python3 topic_model_hard_queries.py --skip-labeling
    python3 topic_model_hard_queries.py --output hard_query_topics.json
    python3 topic_model_hard_queries.py --language-field language

The expensive ML imports are lazy. This means ``--help`` and
``--validate-only`` work before the topic-modeling dependencies are installed.
See ``TOPIC_MODELING.md`` for setup and interpretation notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "hard_queries.json"
DEFAULT_OUTPUT = HERE / "hard_query_topics.json"
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
DEFAULT_LABEL_MODEL = "gpt-4o"
DEFAULT_ENGLISH_VALUES = ("en", "eng", "english")


@dataclass(frozen=True)
class PreparedRecord:
    """One validated, deduplicated prompt plus its original metadata."""

    source_index: int
    record_id: str
    text: str
    original: dict[str, Any]


@dataclass(frozen=True)
class OutlierReductionResult:
    """Assignments captured after each outlier-reduction stage."""

    probability_topics: list[int]
    embedding_topics: list[int]
    final_topics: list[int]
    methods: list[str]
    forced_centroid_assignments: int


def read_json_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array or JSONL file and return object records."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    if not text:
        raise ValueError(f"{path}: file is empty")

    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a top-level JSON array")
        rows = data
    else:
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSONL: {exc.msg}"
                ) from exc

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: record {index} must be a JSON object")
    return rows


def normalized_prompt(text: str) -> str:
    """Normalize case, Unicode compatibility forms, and whitespace for deduping."""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).casefold()


def word_count(text: str) -> int:
    """Count whitespace-delimited words, matching the paper recipe's cutoff."""
    return len(text.split())


def describe_numbers(values: Sequence[int]) -> dict[str, float | int]:
    """Return compact descriptive statistics for an integer sequence."""
    if not values:
        return {}
    return {
        "min": min(values),
        "median": float(statistics.median(values)),
        "mean": round(float(statistics.fmean(values)), 3),
        "max": max(values),
    }


def prepare_records(
    rows: Sequence[dict[str, Any]],
    *,
    id_field: str,
    text_field: str,
    language_field: str | None = None,
    english_values: Sequence[str] = DEFAULT_ENGLISH_VALUES,
) -> tuple[list[PreparedRecord], dict[str, Any]]:
    """Validate, optionally filter by language tag, and deduplicate prompts."""
    allowed_languages = {value.strip().casefold() for value in english_values if value.strip()}
    if language_field and not allowed_languages:
        raise ValueError("at least one --english-value is required with --language-field")

    prepared: list[PreparedRecord] = []
    seen_prompts: dict[str, int] = {}
    seen_ids: dict[str, int] = {}
    duplicate_count = 0
    non_english_count = 0
    missing_language_count = 0

    for source_index, row in enumerate(rows):
        raw_id = row.get(id_field)
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(
                f"record {source_index}: {id_field!r} must be a non-empty string"
            )
        record_id = raw_id.strip()

        raw_text = row.get(text_field)
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(
                f"record {source_index} ({record_id!r}): {text_field!r} "
                "must be a non-empty string"
            )
        prompt = raw_text.strip()

        if language_field:
            raw_language = row.get(language_field)
            if not isinstance(raw_language, str) or not raw_language.strip():
                missing_language_count += 1
                continue
            if raw_language.strip().casefold() not in allowed_languages:
                non_english_count += 1
                continue

        prompt_key = normalized_prompt(prompt)
        if prompt_key in seen_prompts:
            duplicate_count += 1
            continue

        if record_id in seen_ids:
            first = seen_ids[record_id]
            raise ValueError(
                f"record {source_index}: duplicate {id_field} {record_id!r}; "
                f"first seen at record {first}"
            )

        original = dict(row)
        original[id_field] = record_id
        original[text_field] = prompt
        seen_prompts[prompt_key] = source_index
        seen_ids[record_id] = source_index
        prepared.append(
            PreparedRecord(
                source_index=source_index,
                record_id=record_id,
                text=prompt,
                original=original,
            )
        )

    if not prepared:
        suffix = " after language filtering and deduplication" if rows else ""
        raise ValueError(f"no usable prompts were found{suffix}")

    lengths = [word_count(record.text) for record in prepared]
    stats = {
        "records_loaded": len(rows),
        "records_kept": len(prepared),
        "duplicates_removed": duplicate_count,
        "language_filter": {
            "applied": bool(language_field),
            "field": language_field,
            "accepted_values": sorted(allowed_languages) if language_field else [],
            "non_english_removed": non_english_count,
            "missing_tag_removed": missing_language_count,
            "assumed_english": not bool(language_field),
        },
        "word_counts": describe_numbers(lengths),
    }
    return prepared, stats


def select_representative_indices(
    records: Sequence[PreparedRecord],
    topics: Sequence[int],
    hdbscan_probabilities: Sequence[float],
    *,
    examples_per_topic: int,
    high_probability_fraction: float,
    max_words: int,
) -> dict[int, dict[str, Any]]:
    """Select examples strictly from each cluster's top probability fraction.

    The probability percentile is applied before the word-count filter. A small
    cluster can therefore yield fewer than ``examples_per_topic`` examples; the
    returned metadata makes that shortfall explicit instead of silently relaxing
    the requested top-fraction rule.
    """
    if not (
        len(records) == len(topics) == len(hdbscan_probabilities)
    ):
        raise ValueError("records, topics, and HDBSCAN probabilities must align")
    if examples_per_topic < 1:
        raise ValueError("examples_per_topic must be at least 1")
    if not 0 < high_probability_fraction <= 1:
        raise ValueError("high_probability_fraction must be in (0, 1]")
    if max_words < 1:
        raise ValueError("max_words must be at least 1")

    result: dict[int, dict[str, Any]] = {}
    for topic_id in sorted({int(topic) for topic in topics if int(topic) >= 0}):
        cluster_indices = [
            index for index, topic in enumerate(topics) if int(topic) == topic_id
        ]
        ordered = sorted(
            cluster_indices,
            key=lambda index: (
                -float(hdbscan_probabilities[index]),
                records[index].source_index,
            ),
        )
        probability_pool_size = max(
            1, math.ceil(len(cluster_indices) * high_probability_fraction)
        )
        probability_pool = ordered[:probability_pool_size]
        eligible = [
            index
            for index in probability_pool
            if word_count(records[index].text) < max_words
        ]
        selected = eligible[:examples_per_topic]
        result[topic_id] = {
            "indices": selected,
            "cluster_size": len(cluster_indices),
            "probability_pool_size": probability_pool_size,
            "eligible_in_probability_pool": len(eligible),
            "selected_count": len(selected),
            "requested_count": examples_per_topic,
            "shortfall": max(0, examples_per_topic - len(selected)),
            "strict_top_fraction": True,
        }
    return result


def force_assign_by_embedding_centroids(
    embeddings: Any,
    core_topics: Sequence[int],
    current_topics: Sequence[int],
) -> tuple[list[int], int]:
    """Assign any remaining outliers to the nearest original cluster centroid."""
    import numpy as np

    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(current_topics):
        raise ValueError("embeddings must be a 2-D matrix aligned with topics")
    if len(core_topics) != len(current_topics):
        raise ValueError("core_topics and current_topics must align")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings contain non-finite values")

    topic_ids = sorted({int(topic) for topic in core_topics if int(topic) >= 0})
    if not topic_ids:
        raise ValueError(
            "HDBSCAN produced no clusters, so outliers cannot be reassigned; "
            "lower --min-cluster-size or tune UMAP/HDBSCAN"
        )

    centroids = []
    for topic_id in topic_ids:
        indices = [
            index for index, topic in enumerate(core_topics) if int(topic) == topic_id
        ]
        centroids.append(matrix[indices].mean(axis=0))
    centroid_matrix = np.vstack(centroids)
    centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    if np.any(centroid_norms == 0):
        raise ValueError("a topic centroid has zero magnitude")
    centroid_matrix /= centroid_norms

    final_topics = [int(topic) for topic in current_topics]
    forced = 0
    for index, topic in enumerate(final_topics):
        if topic != -1:
            continue
        vector = matrix[index]
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError(f"embedding {index} has zero magnitude")
        similarities = centroid_matrix @ (vector / norm)
        final_topics[index] = topic_ids[int(np.argmax(similarities))]
        forced += 1
    return final_topics, forced


def reduce_outliers_two_stage(
    topic_model: Any,
    documents: Sequence[str],
    initial_topics: Sequence[int],
    topic_probabilities: Any,
    embeddings: Any,
    *,
    probability_threshold: float,
    embedding_threshold: float,
    force_all: bool = True,
) -> OutlierReductionResult:
    """Chain probability and embedding outlier reduction, preserving each stage."""
    original = [int(topic) for topic in initial_topics]
    if -1 not in original:
        methods = ["original_cluster" for _ in original]
        return OutlierReductionResult(original, original, original, methods, 0)

    probability_topics = [
        int(topic)
        for topic in topic_model.reduce_outliers(
            list(documents),
            original,
            probabilities=topic_probabilities,
            strategy="probabilities",
            threshold=probability_threshold,
        )
    ]

    embedding_topics = list(probability_topics)
    if -1 in probability_topics:
        embedding_topics = [
            int(topic)
            for topic in topic_model.reduce_outliers(
                list(documents),
                probability_topics,
                embeddings=embeddings,
                strategy="embeddings",
                threshold=embedding_threshold,
            )
        ]

    final_topics = list(embedding_topics)
    forced = 0
    if force_all and -1 in final_topics:
        final_topics, forced = force_assign_by_embedding_centroids(
            embeddings, original, final_topics
        )

    methods: list[str] = []
    for initial, after_probability, after_embedding, final in zip(
        original, probability_topics, embedding_topics, final_topics
    ):
        if initial != -1:
            methods.append("original_cluster")
        elif after_probability != -1:
            methods.append("hdbscan_probability")
        elif after_embedding != -1:
            methods.append("embedding_similarity")
        elif final != -1:
            methods.append("embedding_centroid_fallback")
        else:
            methods.append("unassigned")

    return OutlierReductionResult(
        probability_topics=probability_topics,
        embedding_topics=embedding_topics,
        final_topics=final_topics,
        methods=methods,
        forced_centroid_assignments=forced,
    )


def package_version(distribution: str) -> str | None:
    """Return an installed distribution version without failing the analysis."""
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON through a sibling temporary file and atomically replace output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_label_cache(path: Path) -> dict[str, Any]:
    """Load the resumable OpenAI label cache, if present."""
    if not path.exists():
        return {"schema_version": 1, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read label cache {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        raise ValueError(f"{path}: invalid label-cache structure")
    return data


def label_fingerprint(
    *, topic_id: int, model: str, examples: Sequence[str], keywords: Sequence[str]
) -> str:
    """Fingerprint the exact evidence and model used for a topic label."""
    material = json.dumps(
        {
            "topic_id": topic_id,
            "model": model,
            "examples": list(examples),
            "keywords": list(keywords),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def request_topic_label(
    client: Any,
    *,
    model: str,
    topic_id: int,
    examples: Sequence[str],
    keywords: Sequence[str],
) -> dict[str, str]:
    """Ask an OpenAI model for one narrow cluster name and description."""
    numbered_examples = "\n".join(
        f"{number}. {example}" for number, example in enumerate(examples, start=1)
    )
    keyword_text = ", ".join(keywords) if keywords else "(none)"
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You label narrow clusters of user prompts. Infer the shared "
                    "request or subject, distinguish it from nearby plausible topics, "
                    "and ignore generic boilerplate such as 'text where' unless that "
                    "form is itself the cluster's defining intent. Return a concise "
                    "category name and a one-sentence description."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic id: {topic_id}\n"
                    f"BERTopic keywords: {keyword_text}\n\n"
                    f"Representative prompts:\n{numbered_examples}"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "topic_label",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "description"],
                    "additionalProperties": False,
                },
            }
        },
    )
    if getattr(response, "status", None) != "completed":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        suffix = f" ({reason})" if reason else ""
        raise RuntimeError(
            f"OpenAI labeling for topic {topic_id} did not complete{suffix}"
        )
    try:
        label = json.loads(response.output_text)
    except (AttributeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"OpenAI returned invalid structured output for topic {topic_id}"
        ) from exc
    name = label.get("name")
    description = label.get("description")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError(f"OpenAI returned an empty name for topic {topic_id}")
    if not isinstance(description, str) or not description.strip():
        raise RuntimeError(f"OpenAI returned an empty description for topic {topic_id}")
    return {"name": name.strip(), "description": description.strip()}


def fallback_label(topic_id: int, keywords: Sequence[str]) -> dict[str, str]:
    """Create a deterministic local label when OpenAI labeling is disabled."""
    useful = [keyword for keyword in keywords if keyword]
    if useful:
        name = " / ".join(useful[:3])
        description = f"BERTopic keyword cluster: {', '.join(useful[:8])}."
    else:
        name = f"Topic {topic_id}"
        description = "No BERTopic keywords were available for this cluster."
    return {"name": name, "description": description}


def create_topic_labels(
    *,
    topic_evidence: Mapping[int, dict[str, Any]],
    model: str,
    skip_labeling: bool,
    cache_path: Path,
    verbose: bool,
) -> dict[int, dict[str, str]]:
    """Create fallback or cached/resumable OpenAI labels for all topics."""
    labels: dict[int, dict[str, str]] = {}
    if skip_labeling:
        for topic_id, evidence in topic_evidence.items():
            fallback = fallback_label(topic_id, evidence["keywords"])
            labels[topic_id] = {**fallback, "source": "bertopic_keywords"}
        return labels

    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI labeling dependencies are missing; install "
            "requirements-topic-modeling.txt or pass --skip-labeling"
        ) from exc

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set; add it to the environment/.env or pass "
            "--skip-labeling"
        )
    client = OpenAI()
    cache = load_label_cache(cache_path)
    entries = cache["entries"]

    for topic_id in sorted(topic_evidence):
        evidence = topic_evidence[topic_id]
        examples = evidence["examples"]
        keywords = evidence["keywords"]
        if not examples:
            fallback = fallback_label(topic_id, keywords)
            labels[topic_id] = {
                **fallback,
                "source": "bertopic_keywords_no_eligible_examples",
            }
            continue

        fingerprint = label_fingerprint(
            topic_id=topic_id,
            model=model,
            examples=examples,
            keywords=keywords,
        )
        cached = entries.get(fingerprint)
        if (
            isinstance(cached, dict)
            and isinstance(cached.get("name"), str)
            and isinstance(cached.get("description"), str)
        ):
            label = {
                "name": cached["name"],
                "description": cached["description"],
            }
            source = f"openai_cache:{model}"
        else:
            if verbose:
                print(
                    f"Labeling topic {topic_id} with {model} ({len(examples)} examples)",
                    file=sys.stderr,
                    flush=True,
                )
            label = request_topic_label(
                client,
                model=model,
                topic_id=topic_id,
                examples=examples,
                keywords=keywords,
            )
            entries[fingerprint] = {
                **label,
                "model": model,
                "topic_id": topic_id,
            }
            atomic_write_json(cache_path, cache)
            source = f"openai:{model}"
        labels[topic_id] = {**label, "source": source}
    return labels


def output_record(
    record: PreparedRecord, *, id_field: str, text_field: str
) -> dict[str, Any]:
    """Represent a source record without duplicating configured id/text metadata."""
    metadata = {
        key: value
        for key, value in record.original.items()
        if key not in {id_field, text_field}
    }
    return {
        "id": record.record_id,
        "prompt": record.text,
        "metadata": metadata,
        "source_index": record.source_index,
    }


def build_analysis(
    records: Sequence[PreparedRecord],
    preprocess_stats: dict[str, Any],
    *,
    input_path: Path,
    id_field: str,
    text_field: str,
    topic_model: Any,
    embeddings: Any,
    initial_topics: Sequence[int],
    topic_probabilities: Any,
    hdbscan_probabilities: Sequence[float],
    representative_selections: Mapping[int, dict[str, Any]],
    reduction: OutlierReductionResult,
    args: argparse.Namespace,
    effective_umap_neighbors: int,
) -> dict[str, Any]:
    """Assemble the complete, JSON-serializable analysis artifact."""
    import numpy as np

    initial = [int(topic) for topic in initial_topics]
    final = [int(topic) for topic in reduction.final_topics]
    initial_counts = Counter(initial)
    final_counts = Counter(final)

    topic_ids = sorted(topic for topic in final_counts if topic >= 0)
    topic_evidence: dict[int, dict[str, Any]] = {}
    keywords_by_topic: dict[int, list[dict[str, float | str]]] = {}
    for topic_id in topic_ids:
        raw_keywords = topic_model.get_topic(topic_id) or []
        keywords = [
            {"term": str(term), "score": round(float(score), 8)}
            for term, score in raw_keywords[: args.top_words]
        ]
        keywords_by_topic[topic_id] = keywords
        selected_indices = representative_selections.get(topic_id, {}).get(
            "indices", []
        )
        topic_evidence[topic_id] = {
            "keywords": [item["term"] for item in keywords],
            "examples": [records[index].text for index in selected_indices],
        }

    labels = create_topic_labels(
        topic_evidence=topic_evidence,
        model=args.label_model,
        skip_labeling=args.skip_labeling,
        cache_path=args.label_cache,
        verbose=not args.quiet,
    )

    topics_output = []
    for topic_id in topic_ids:
        selection = representative_selections.get(
            topic_id,
            {
                "indices": [],
                "cluster_size": initial_counts.get(topic_id, 0),
                "probability_pool_size": 0,
                "eligible_in_probability_pool": 0,
                "selected_count": 0,
                "requested_count": args.examples_per_topic,
                "shortfall": args.examples_per_topic,
                "strict_top_fraction": True,
            },
        )
        examples = []
        for index in selection["indices"]:
            examples.append(
                {
                    **output_record(
                        records[index], id_field=id_field, text_field=text_field
                    ),
                    "word_count": word_count(records[index].text),
                    "hdbscan_probability": round(
                        float(hdbscan_probabilities[index]), 8
                    ),
                }
            )
        selection_metadata = {
            key: value for key, value in selection.items() if key != "indices"
        }
        topics_output.append(
            {
                "topic_id": topic_id,
                **labels[topic_id],
                "initial_cluster_size": initial_counts.get(topic_id, 0),
                "final_size": final_counts[topic_id],
                "keywords": keywords_by_topic[topic_id],
                "representative_selection": selection_metadata,
                "examples": examples,
            }
        )

    probabilities_array = (
        np.asarray(topic_probabilities) if topic_probabilities is not None else None
    )
    assignments = []
    for index, record in enumerate(records):
        if probabilities_array is None or probabilities_array.size == 0:
            best_soft_probability = None
        elif probabilities_array.ndim == 1:
            best_soft_probability = round(float(probabilities_array[index]), 8)
        else:
            best_soft_probability = round(
                float(np.max(probabilities_array[index])), 8
            )
        assignments.append(
            {
                **output_record(record, id_field=id_field, text_field=text_field),
                "initial_topic": initial[index],
                "final_topic": final[index],
                "hdbscan_probability": round(
                    float(hdbscan_probabilities[index]), 8
                ),
                "best_soft_topic_probability": best_soft_probability,
                "was_outlier": initial[index] == -1,
                "assignment_method": reduction.methods[index],
            }
        )

    initial_outliers = initial_counts.get(-1, 0)
    after_probability_outliers = reduction.probability_topics.count(-1)
    after_embedding_outliers = reduction.embedding_topics.count(-1)
    final_outliers = final_counts.get(-1, 0)
    embedding_shape = list(np.asarray(embeddings).shape)

    return {
        "schema_version": 1,
        "input": {
            "path": str(input_path.resolve()),
            "id_field": id_field,
            "text_field": text_field,
            **preprocess_stats,
        },
        "summary": {
            "documents_clustered": len(records),
            "topic_count": len(topic_ids),
            "initial_outliers": initial_outliers,
            "initial_outlier_fraction": round(initial_outliers / len(records), 6),
            "outliers_after_probability_pass": after_probability_outliers,
            "outliers_after_embedding_pass": after_embedding_outliers,
            "forced_centroid_assignments": reduction.forced_centroid_assignments,
            "final_outliers": final_outliers,
        },
        "pipeline": {
            "method": "quoted BERTopic pipeline adapted to the supplied dataset",
            "embedding": {
                "model": args.embedding_model,
                "normalize_embeddings": False,
                "shape": embedding_shape,
                "batch_size": args.embedding_batch_size,
            },
            "umap": {
                "n_neighbors": effective_umap_neighbors,
                "n_components": args.umap_components,
                "min_dist": args.umap_min_dist,
                "metric": "cosine",
                "random_state": args.random_state,
            },
            "hdbscan": {
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
                "prediction_data": True,
            },
            "representatives": {
                "examples_per_topic": args.examples_per_topic,
                "high_probability_fraction": args.representative_fraction,
                "maximum_words_exclusive": args.max_example_words,
                "probability_source": "HDBSCAN probabilities_ membership strength",
                "small_cluster_policy": "return fewer examples and report shortfall",
            },
            "labeling": {
                "enabled": not args.skip_labeling,
                "model": args.label_model if not args.skip_labeling else None,
                "cache_path": (
                    str(args.label_cache.resolve()) if not args.skip_labeling else None
                ),
            },
            "outlier_reduction": {
                "first_pass": "HDBSCAN topic probabilities",
                "probability_threshold": args.outlier_probability_threshold,
                "second_pass": "prompt/topic embedding cosine similarity",
                "embedding_threshold": args.outlier_embedding_threshold,
                "force_all_with_original_cluster_centroids": not args.allow_outliers,
            },
            "software": {
                "python": sys.version.split()[0],
                "bertopic": package_version("bertopic"),
                "sentence_transformers": package_version("sentence-transformers"),
                "umap_learn": package_version("umap-learn"),
                "hdbscan": package_version("hdbscan"),
                "scikit_learn": package_version("scikit-learn"),
            },
        },
        "topics": topics_output,
        "assignments": assignments,
    }


def run_pipeline(
    records: Sequence[PreparedRecord],
    preprocess_stats: dict[str, Any],
    *,
    input_path: Path,
    id_field: str,
    text_field: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run embeddings, UMAP, HDBSCAN, BERTopic, labeling, and reporting."""
    if sys.version_info < (3, 11):
        raise RuntimeError(
            "the current BERTopic dependency stack requires Python 3.11+; "
            "create the environment described in TOPIC_MODELING.md"
        )
    try:
        import numpy as np
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError(
            "topic-modeling dependencies are missing; install them with "
            "`python -m pip install -r requirements-topic-modeling.txt`"
        ) from exc

    documents = [record.text for record in records]
    if len(documents) < args.min_cluster_size:
        raise ValueError(
            f"only {len(documents)} prompts remain, fewer than --min-cluster-size "
            f"{args.min_cluster_size}"
        )
    if len(documents) < 3:
        raise ValueError("at least three prompts are required")

    effective_neighbors = min(args.umap_neighbors, len(documents) - 1)
    if not args.quiet:
        print(
            f"Embedding {len(documents)} prompts with {args.embedding_model}",
            file=sys.stderr,
            flush=True,
        )
    sentence_model = SentenceTransformer(args.embedding_model)
    embeddings = sentence_model.encode(
        documents,
        batch_size=args.embedding_batch_size,
        show_progress_bar=not args.quiet,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
        raise RuntimeError("SentenceTransformer returned an invalid embedding matrix")

    umap_model = UMAP(
        n_neighbors=effective_neighbors,
        n_components=args.umap_components,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.random_state,
        low_memory=True,
    )
    hdbscan_kwargs: dict[str, Any] = {
        "min_cluster_size": args.min_cluster_size,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": True,
    }
    if args.min_samples is not None:
        hdbscan_kwargs["min_samples"] = args.min_samples
    hdbscan_model = HDBSCAN(**hdbscan_kwargs)
    vectorizer_model = CountVectorizer(
        stop_words="english", ngram_range=(1, args.max_ngram)
    )
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=not args.quiet,
        top_n_words=max(args.top_words, 10),
    )

    initial_topics, topic_probabilities = topic_model.fit_transform(
        documents, embeddings
    )
    initial_topics = [int(topic) for topic in initial_topics]
    if not any(topic >= 0 for topic in initial_topics):
        raise RuntimeError(
            "HDBSCAN classified every prompt as an outlier; lower "
            "--min-cluster-size/--min-samples or tune UMAP"
        )

    hdbscan_probabilities = np.asarray(
        topic_model.hdbscan_model.probabilities_, dtype=np.float64
    )
    if hdbscan_probabilities.shape != (len(documents),):
        raise RuntimeError("HDBSCAN returned invalid membership probabilities")

    representative_selections = select_representative_indices(
        records,
        initial_topics,
        hdbscan_probabilities,
        examples_per_topic=args.examples_per_topic,
        high_probability_fraction=args.representative_fraction,
        max_words=args.max_example_words,
    )

    reduction = reduce_outliers_two_stage(
        topic_model,
        documents,
        initial_topics,
        topic_probabilities,
        embeddings,
        probability_threshold=args.outlier_probability_threshold,
        embedding_threshold=args.outlier_embedding_threshold,
        force_all=not args.allow_outliers,
    )
    topic_model.update_topics(documents, topics=reduction.final_topics)

    return build_analysis(
        records,
        preprocess_stats,
        input_path=input_path,
        id_field=id_field,
        text_field=text_field,
        topic_model=topic_model,
        embeddings=embeddings,
        initial_topics=initial_topics,
        topic_probabilities=topic_probabilities,
        hdbscan_probabilities=hdbscan_probabilities,
        representative_selections=representative_selections,
        reduction=reduction,
        args=args,
        effective_umap_neighbors=effective_neighbors,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"JSON array or JSONL input (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"analysis JSON (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-field", default="query")
    parser.add_argument(
        "--language-field",
        help=(
            "optional metadata field used to keep English-tagged rows; without it, "
            "the input is assumed to be English"
        ),
    )
    parser.add_argument(
        "--english-value",
        dest="english_values",
        action="append",
        help="accepted English tag; repeatable (defaults: en, eng, english)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate/filter/deduplicate and print counts without loading ML models",
    )

    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-components", type=int, default=5)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument(
        "--min-samples",
        type=int,
        help="HDBSCAN min_samples (default: same as min_cluster_size)",
    )
    parser.add_argument("--max-ngram", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--top-words", type=int, default=10)

    parser.add_argument("--examples-per-topic", type=int, default=20)
    parser.add_argument(
        "--representative-fraction",
        type=float,
        default=0.20,
        help=(
            "top HDBSCAN-probability fraction eligible as examples (default: 0.20; "
            "use 1.0 to allow up to 20 examples from small clusters)"
        ),
    )
    parser.add_argument(
        "--max-example-words",
        type=int,
        default=100,
        help="examples must contain fewer than this many words (default: 100)",
    )

    parser.add_argument("--label-model", default=DEFAULT_LABEL_MODEL)
    parser.add_argument(
        "--skip-labeling",
        action="store_true",
        help="do not call OpenAI; use deterministic BERTopic keyword labels",
    )
    parser.add_argument(
        "--label-cache",
        type=Path,
        help="resumable label cache (default: <output stem>.labels.json)",
    )
    parser.add_argument(
        "--outlier-probability-threshold",
        type=float,
        default=0.05,
        help="minimum HDBSCAN soft probability for first-pass assignment",
    )
    parser.add_argument(
        "--outlier-embedding-threshold",
        type=float,
        default=0.0,
        help="minimum embedding cosine similarity for second-pass assignment",
    )
    parser.add_argument(
        "--allow-outliers",
        action="store_true",
        help="leave any outliers that survive both passes instead of forcing a centroid match",
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"input file not found: {args.input}")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be at least 1")
    if args.umap_neighbors < 2:
        parser.error("--umap-neighbors must be at least 2")
    if args.umap_components < 2:
        parser.error("--umap-components must be at least 2")
    if args.umap_min_dist < 0:
        parser.error("--umap-min-dist cannot be negative")
    if args.min_cluster_size < 2:
        parser.error("--min-cluster-size must be at least 2")
    if args.min_samples is not None and args.min_samples < 1:
        parser.error("--min-samples must be at least 1")
    if args.top_words < 1:
        parser.error("--top-words must be at least 1")
    if args.examples_per_topic < 1:
        parser.error("--examples-per-topic must be at least 1")
    if not 0 < args.representative_fraction <= 1:
        parser.error("--representative-fraction must be in (0, 1]")
    if args.max_example_words < 1:
        parser.error("--max-example-words must be at least 1")
    if not 0 <= args.outlier_probability_threshold <= 1:
        parser.error("--outlier-probability-threshold must be between 0 and 1")
    if not -1 <= args.outlier_embedding_threshold <= 1:
        parser.error("--outlier-embedding-threshold must be between -1 and 1")
    if args.english_values is None:
        args.english_values = list(DEFAULT_ENGLISH_VALUES)
    if args.label_cache is None:
        args.label_cache = args.output.with_name(
            f"{args.output.stem}.labels.json"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = read_json_records(args.input)
        records, preprocess_stats = prepare_records(
            rows,
            id_field=args.id_field,
            text_field=args.text_field,
            language_field=args.language_field,
            english_values=args.english_values,
        )
        if args.validate_only:
            print(json.dumps(preprocess_stats, indent=2, ensure_ascii=False))
            return 0

        analysis = run_pipeline(
            records,
            preprocess_stats,
            input_path=args.input,
            id_field=args.id_field,
            text_field=args.text_field,
            args=args,
        )
        atomic_write_json(args.output, analysis)
        print(
            f"Wrote {args.output} with {analysis['summary']['topic_count']} topics "
            f"and {analysis['summary']['documents_clustered']} prompts"
        )
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
