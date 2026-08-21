#!/usr/bin/env python3
"""Group expressions that can occupy the same grammatical slot in a query.

The expensive model imports are intentionally lazy, so the extraction and JSON
validation helpers can be tested without loading PyTorch.

run it with python analysis/words_semantic_clustering.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parent / "hard_queries.json"
DEFAULT_EASY_INPUT = HERE.parent / "easy_queries.json"
DEFAULT_OUTPUT = HERE.parent / "analytics" / "word_semantic_groups.json"
DEFAULT_MODEL = "all-mpnet-base-v2"

# Keep this list deliberately small: these are query scaffolding words rather
# than concepts the report is intended to compare.
DEFAULT_STOP_WORDS = frozenset(
    {"text", "a", "the", "to", "of", "is", "in", "with", "an", "that"}
)
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
PROGRESSIVE_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+"
    r"[a-z]+ing\b",
    re.IGNORECASE,
)
# Orthographic ``-ing`` nouns which otherwise look like progressive verbs.
NON_VERBAL_ING_WORDS = frozenset(
    {
        "anything",
        "ceiling",
        "darling",
        "everything",
        "king",
        "morning",
        "nothing",
        "offspring",
        "something",
        "spring",
        "thing",
    }
)

# These phrases capture discourse units used by the hard-query templates, such
# as "an aside", "a long appeal", and "an enumeration". They are retained as
# phrases even though their leading article is normally a stop word.
CLAUSE_PHRASE_RE = re.compile(
    r"\b(?:a|an)\s+[a-z][a-z'-]*"
    r"(?:\s+(?!(?:of|with|and|then|followed)\b)[a-z][a-z'-]*){0,2}"
    r"(?=\s*(?:,|;|:|\bthen\b|\bfollowed\b|\band\b|\bof\b|\bwith\b|$))",
    re.IGNORECASE,
)
SEMANTIC_PREFIX = "Meaning of the replacement expression: "


def load_queries(paths: Sequence[Path]) -> list[str]:
    """Load and validate query strings from one or more JSON arrays."""
    queries: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a JSON array")
        for index, row in enumerate(payload):
            if not isinstance(row, dict) or not isinstance(row.get("query"), str):
                raise ValueError(f"{path}: item {index} must contain a string 'query'")
            queries.append(row["query"])
    return queries


def normalize_term(value: str) -> str:
    """Return a lowercase, whitespace-normalized term."""
    return " ".join(value.casefold().split())


def extract_terms(
    queries: Iterable[str],
    *,
    stop_words: frozenset[str] = DEFAULT_STOP_WORDS,
    include_clause_phrases: bool = True,
) -> Counter[str]:
    """Extract unique word/phrase candidates and their occurrence counts."""
    counts: Counter[str] = Counter()
    for query in queries:
        lowered = query.casefold()
        counts.update(
            token
            for token in WORD_RE.findall(lowered)
            if token not in stop_words
        )
        if include_clause_phrases:
            counts.update(normalize_term(match.group()) for match in CLAUSE_PHRASE_RE.finditer(lowered))
        # An auxiliary is part of the replaceable unit.  Extracting only
        # ``starting`` from ``is starting`` loses the tense and agreement that
        # make it interchangeable with ``starts``.
        for match in PROGRESSIVE_RE.finditer(lowered):
            phrase = normalize_term(match.group())
            if phrase.rsplit(" ", 1)[-1] not in NON_VERBAL_ING_WORDS:
                counts[phrase] += 1
    return counts


def grammatical_signature(term: str) -> str:
    """Return the coarse sentence slot in which *term* can be substituted.

    This intentionally describes surface grammar, not meaning.  Semantic
    similarity is still supplied by the embedding model.  In particular,
    third-person present progressives and simple-present verbs share a slot,
    while bare, participial, and past forms do not.
    """
    words = normalize_term(term).split()
    if len(words) == 2 and words[1].endswith("ing"):
        auxiliary = words[0]
        if auxiliary == "is":
            return "finite-present-third-singular"
        if auxiliary == "am":
            return "finite-present-first-singular"
        if auxiliary == "are":
            return "finite-present-non-third-singular"
        if auxiliary == "was":
            return "finite-past-singular"
        if auxiliary == "were":
            return "finite-past-non-singular"
        if auxiliary in {"be", "been", "being"}:
            return f"auxiliary-{auxiliary}-participle"

    if len(words) > 1:
        return "multiword-expression"

    word = words[0]
    if word.endswith("ing"):
        return "non-finite-ing"
    if word.endswith("ed"):
        return "past-or-past-participle"
    if word.endswith("s") and not word.endswith("ss"):
        # This also separates plural nouns from singular nouns, which is the
        # desired behavior for literal in-sentence replacement.
        return "finite-third-singular-or-plural"
    return "uninflected"


def signatures_are_compatible(left: str, right: str) -> bool:
    """Whether two grammatical signatures permit literal substitution."""
    if left == right:
        return True
    third_person = {
        "finite-present-third-singular",
        "finite-third-singular-or-plural",
    }
    return left in third_person and right in third_person


def semantic_root(term: str) -> str:
    """Reduce inflection so embeddings compare meaning rather than spelling.

    This is a deliberately small English surface stemmer.  Grammar is checked
    separately, so mapping ``starts`` and ``is starting`` to ``start`` cannot
    incorrectly put bare ``start`` in their cluster.
    """
    words = normalize_term(term).split()
    word = words[-1] if len(words) == 2 and words[0] in {
        "am", "is", "are", "was", "were", "be", "been", "being"
    } else normalize_term(term)
    if " " in word:
        return word

    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
        if len(word) >= 2 and word[-1] == word[-2] and word[-1] not in "lsz":
            word = word[:-1]
    elif word.endswith("ied") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
        if len(word) >= 2 and word[-1] == word[-2] and word[-1] not in "lsz":
            word = word[:-1]
    elif word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    return word


def semantic_communities(
    terms: Sequence[str],
    *,
    model_name: str,
    threshold: float,
    batch_size: int,
    device: str | None,
) -> list[list[int]]:
    """Embed terms and return non-overlapping, high-similarity communities.

    Searching only the nearest neighbors keeps memory bounded for large
    vocabularies. A cluster is an anchor plus its qualifying neighbors, rather
    than a transitive graph component, which also prevents loose similarity
    chains from joining otherwise unrelated words.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required; install requirements-topic-modeling.txt"
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        [SEMANTIC_PREFIX + semantic_root(term) for term in terms],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    neighbors = util.semantic_search(
        embeddings,
        embeddings,
        query_chunk_size=max(64, batch_size * 2),
        corpus_chunk_size=5_000,
        top_k=min(64, len(terms)),
        score_function=util.cos_sim,
    )
    signatures = [grammatical_signature(term) for term in terms]
    candidates = []
    for query_index, hits in enumerate(neighbors):
        candidates.append(
            [
                int(hit["corpus_id"])
                for hit in hits
                if float(hit["score"]) >= threshold
                and signatures_are_compatible(
                    signatures[query_index], signatures[int(hit["corpus_id"])]
                )
            ]
        )
    anchor_order = sorted(range(len(terms)), key=lambda index: (-len(candidates[index]), index))
    communities: list[list[int]] = []
    assigned: set[int] = set()
    for anchor in anchor_order:
        community = [index for index in candidates[anchor] if index not in assigned]
        if anchor not in assigned and anchor not in community:
            community.insert(0, anchor)
        if len(community) < 2:
            continue
        community.sort()
        communities.append(community)
        assigned.update(community)
    return communities


def build_report(
    terms: Sequence[str],
    counts: Counter[str],
    communities: Sequence[Sequence[int]],
    *,
    input_paths: Sequence[Path],
    query_count: int,
    model_name: str,
    threshold: float,
) -> dict[str, Any]:
    """Create the JSON-serializable clustering report."""
    groups: list[dict[str, Any]] = []
    grouped: set[str] = set()
    normalized_communities = sorted(
        ({terms[index] for index in community} for community in communities),
        key=lambda members: (-len(members), sorted(members)),
    )
    for number, members in enumerate(normalized_communities, start=1):
        words = sorted(members, key=lambda term: (-counts[term], term))
        grouped.update(words)
        groups.append(
            {
                "id": f"cluster{number}",
                "words": words,
                "occurrences": {term: counts[term] for term in words},
            }
        )

    return {
        "metadata": {
            "inputs": [str(path) for path in input_paths],
            "query_count": query_count,
            "candidate_count": len(terms),
            "grouped_candidate_count": len(grouped),
            "group_count": len(groups),
            "model": model_name,
            "cosine_similarity_threshold": threshold,
            "semantic_prefix": SEMANTIC_PREFIX,
            "grammar_filter": "surface substitution signature",
            "stop_words": sorted(DEFAULT_STOP_WORDS),
        },
        "groups": groups,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group synonyms and very closely related terms from query JSON files."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[DEFAULT_INPUT, DEFAULT_EASY_INPUT],
        help="JSON arrays containing a query field (default: hard and easy queries)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="minimum cosine similarity; raise this to make groups stricter (default: 0.90)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", help="sentence-transformers device, e.g. cpu, mps, or cuda")
    parser.add_argument(
        "--no-phrases",
        action="store_true",
        help="extract individual words only",
    )
    args = parser.parse_args(argv)
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be greater than 0 and at most 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    queries = load_queries(args.inputs)
    counts = extract_terms(queries, include_clause_phrases=not args.no_phrases)
    terms = sorted(counts)
    if len(terms) < 2:
        raise ValueError("the input must contain at least two non-stop-word terms")

    print(f"Loaded {len(queries):,} queries and extracted {len(terms):,} candidates")
    communities = semantic_communities(
        terms,
        model_name=args.model,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
    )
    report = build_report(
        terms,
        counts,
        communities,
        input_paths=args.inputs,
        query_count=len(queries),
        model_name=args.model,
        threshold=args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {report['metadata']['group_count']:,} groups "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
