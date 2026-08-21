#!/usr/bin/env python3
"""Basic statistics (length, vocabulary size, frequency) for a query file.

Reads hard_queries.json (a JSON list of {"id", "query", "rank"} objects, or a
JSONL file with the same records) and reports:

  * length distributions, in characters and in words
  * vocabulary size, type/token ratio and hapax legomena
  * the most frequent tokens, bigrams and query-initial words
  * the same length/vocabulary summary broken down per corpus (the id prefix)

Examples:
    python3 query_stats.py
    python3 query_stats.py --top 40
    python3 query_stats.py hard_queries.json --output query_stats.json
    python3 query_stats.py --no-stopwords

"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
DEFAULT_QUERY_PATH = HERE / "hard_queries.json"

# Words are lowercased alphanumeric runs; internal apostrophes/hyphens are kept
# so "don't" and "wire-service" stay single tokens.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*")

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can do does for from had has have he her his
    i if in into is it its me my no not of on or our she so some such than that the
    their them then there these they this to too was we were what when where which
    who will with would you your
    """.split()
)


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list or a JSONL file of query records."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of records")
    return [row for row in rows if isinstance(row, dict) and row.get("query")]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def corpus_of(query_id: str) -> str:
    """"msmarco-019723_1" -> "msmarco"; ids without a prefix fall back to "other"."""
    prefix = str(query_id).split("-", 1)[0]
    return prefix if prefix and prefix != str(query_id) else "other"


def describe(values: Sequence[float]) -> dict[str, float]:
    """Min/max/mean/median/stdev plus a few percentiles."""
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        # Nearest-rank percentile; exact enough for a descriptive summary.
        idx = min(len(ordered) - 1, max(0, round(p / 100 * len(ordered) + 0.5) - 1))
        return float(ordered[idx])

    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "p25": pct(25),
        "median": float(statistics.median(ordered)),
        "mean": round(statistics.fmean(ordered), 2),
        "p75": pct(75),
        "p90": pct(90),
        "max": float(ordered[-1]),
        "stdev": round(statistics.stdev(ordered), 2) if len(ordered) > 1 else 0.0,
    }


def vocabulary_stats(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    hapax = sum(1 for n in counts.values() if n == 1)
    return {
        "tokens": total,
        "vocabulary_size": len(counts),
        "type_token_ratio": round(len(counts) / total, 4) if total else 0.0,
        "hapax_legomena": hapax,
        "hapax_share_of_vocab": round(hapax / len(counts), 4) if counts else 0.0,
    }


def analyse(rows: Sequence[dict[str, Any]], top: int, keep_stopwords: bool) -> dict[str, Any]:
    char_lengths: list[int] = []
    word_lengths: list[int] = []
    unigrams: Counter[str] = Counter()
    content_words: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    first_words: Counter[str] = Counter()
    per_corpus: dict[str, dict[str, Any]] = {}
    ranks: list[int] = []
    duplicate_queries: Counter[str] = Counter()

    for row in rows:
        query = str(row["query"]).strip()
        tokens = tokenize(query)
        char_lengths.append(len(query))
        word_lengths.append(len(tokens))
        unigrams.update(tokens)
        content_words.update(t for t in tokens if keep_stopwords or t not in STOPWORDS)
        bigrams.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
        if tokens:
            first_words[tokens[0]] += 1
        duplicate_queries[query.lower()] += 1

        if isinstance(row.get("rank"), (int, float)):
            ranks.append(int(row["rank"]))

        bucket = per_corpus.setdefault(
            corpus_of(row.get("id", "")),
            {"queries": 0, "chars": [], "words": [], "counts": Counter()},
        )
        bucket["queries"] += 1
        bucket["chars"].append(len(query))
        bucket["words"].append(len(tokens))
        bucket["counts"].update(tokens)

    report: dict[str, Any] = {
        "queries": len(rows),
        "duplicate_queries": sum(n - 1 for n in duplicate_queries.values() if n > 1),
        "length": {
            "characters": describe(char_lengths),
            "words": describe(word_lengths),
        },
        "vocabulary": vocabulary_stats(unigrams),
        "frequency": {
            "top_words": unigrams.most_common(top),
            "top_content_words": content_words.most_common(top),
            "top_bigrams": bigrams.most_common(top),
            "top_first_words": first_words.most_common(min(top, 15)),
        },
        "by_corpus": {
            name: {
                "queries": bucket["queries"],
                "characters": describe(bucket["chars"]),
                "words": describe(bucket["words"]),
                "vocabulary": vocabulary_stats(bucket["counts"]),
                "top_content_words": Counter(
                    {
                        w: n
                        for w, n in bucket["counts"].items()
                        if keep_stopwords or w not in STOPWORDS
                    }
                ).most_common(10),
            }
            for name, bucket in sorted(per_corpus.items(), key=lambda kv: -kv[1]["queries"])
        },
    }
    if ranks:
        report["rank"] = describe(ranks)
    return report


def fmt_row(label: str, stats: dict[str, float]) -> str:
    if not stats:
        return f"  {label:<12} (no data)"
    return (
        f"  {label:<12} min {stats['min']:>7.0f}  p25 {stats['p25']:>7.0f}  "
        f"median {stats['median']:>7.0f}  mean {stats['mean']:>9.2f}  "
        f"p75 {stats['p75']:>7.0f}  p90 {stats['p90']:>7.0f}  max {stats['max']:>7.0f}"
    )


def fmt_pairs(pairs: Iterable[tuple[str, int]], per_line: int = 5) -> list[str]:
    items = [f"{word} ({n})" for word, n in pairs]
    return [
        "  " + "   ".join(items[i : i + per_line]) for i in range(0, len(items), per_line)
    ]


def print_report(report: dict[str, Any], path: Path) -> None:
    print(f"=== Query statistics: {path.name} ===")
    print(f"queries: {report['queries']}   duplicate texts: {report['duplicate_queries']}")

    print("\n-- Length --")
    print(fmt_row("characters", report["length"]["characters"]))
    print(fmt_row("words", report["length"]["words"]))
    if "rank" in report:
        print(fmt_row("rank", report["rank"]))

    vocab = report["vocabulary"]
    print("\n-- Vocabulary --")
    print(f"  tokens: {vocab['tokens']}   vocabulary size: {vocab['vocabulary_size']}")
    print(
        f"  type/token ratio: {vocab['type_token_ratio']}   "
        f"hapax legomena: {vocab['hapax_legomena']} "
        f"({vocab['hapax_share_of_vocab']:.1%} of vocab)"
    )

    freq = report["frequency"]
    print("\n-- Most frequent words (all) --")
    print("\n".join(fmt_pairs(freq["top_words"])))
    print("\n-- Most frequent content words --")
    print("\n".join(fmt_pairs(freq["top_content_words"])))
    print("\n-- Most frequent bigrams --")
    print("\n".join(fmt_pairs(freq["top_bigrams"], per_line=3)))
    print("\n-- Most frequent opening words --")
    print("\n".join(fmt_pairs(freq["top_first_words"])))

    print("\n-- By corpus --")
    header = f"  {'corpus':<10} {'queries':>7} {'med.words':>10} {'mean words':>11} {'vocab':>7} {'TTR':>7}"
    print(header)
    for name, stats in report["by_corpus"].items():
        print(
            f"  {name:<10} {stats['queries']:>7} {stats['words']['median']:>10.0f} "
            f"{stats['words']['mean']:>11.2f} {stats['vocabulary']['vocabulary_size']:>7} "
            f"{stats['vocabulary']['type_token_ratio']:>7.3f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "queries",
        nargs="?",
        type=Path,
        default=DEFAULT_QUERY_PATH,
        help=f"query file, JSON list or JSONL (default: {DEFAULT_QUERY_PATH.name})",
    )
    parser.add_argument("--top", type=int, default=25, help="how many frequency entries to show (default: 25)")
    parser.add_argument(
        "--no-stopwords",
        dest="keep_stopwords",
        action="store_false",
        help="drop English stopwords from the content-word counts (default)",
    )
    parser.add_argument(
        "--stopwords",
        dest="keep_stopwords",
        action="store_true",
        help="keep stopwords in the content-word counts",
    )
    parser.add_argument("--output", type=Path, help="also write the full report as JSON")
    parser.set_defaults(keep_stopwords=False)
    args = parser.parse_args(argv)

    if not args.queries.exists():
        print(f"error: {args.queries} not found", file=sys.stderr)
        return 1

    rows = load_queries(args.queries)
    if not rows:
        print(f"error: no query records in {args.queries}", file=sys.stderr)
        return 1

    report = analyse(rows, top=args.top, keep_stopwords=args.keep_stopwords)
    print_report(report, args.queries)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
