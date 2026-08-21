#!/usr/bin/env python3
"""Compare hard/easy query vocabulary and display basic query statistics.

By default the script reads ``hard_queries.json`` and ``easy_queries.json``
from the repository root. Alternative files can be supplied on the command
line. Each input must be an array of objects with a string ``query`` field.

run it with :
python analysis/basic_analytics.py
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parent / "hard_queries.json"
DEFAULT_EASY_INPUT = HERE.parent / "easy_queries.json"
TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
PREFIX_LENGTHS = (2, 3, 4, 5, 6)
COMMON_WORD_LIMIT = 30
WORD_CLOUD_LIMIT = 150
EXCLUDED_COMMON_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "will", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves", "text","directly","indirectly"
}
EXCLUDED_COMMON_TWO_WORDS = {"a text"}
FROMAL_WRITING_STYLE_WORDS = {	"formal", "academic", "dense", "technical", "objective", "cautious", "scientific","encyclopedic", "journalistic", "factual","academic","objective", "analytical", "expository", "scholarly", "technical", "clinical", "authoritative", "matter-of-fact", "neutral", "didactic"}
TONE_WORDS = {"tone","register","style","diction","mood", "objective", "analytical", "cautious"}
STYLE_ADJECTIVES = {"formal","informal","colloquial","conversational","casual","academic","literary","journalistic","technical","poetic","persuasive","descriptive","rhetorical","promotional","narrative","expository","objective","subjective","analytical","critical","reflective","humorous","sarcastic","ironic","satirical","playful","serious","dramatic","emotional","passionate","enthusiastic","optimistic","pessimistic"}
LENGTH_WORDS = {"concise","succinct","brief","short","lengthy"}
TEXT_TYPE= {"research abstract", "abstract" , "paper", "journal", "prose", "passage", "text", "article", "essay", "story", "report", "review"}
{"a claim","a review","a result","a correction","a comparison","a risk", "a distribution"}

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"input JSON array (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of opening phrases to display per length (default: 10)",
    )
    parser.add_argument(
        "--easy-input",
        type=Path,
        default=DEFAULT_EASY_INPUT,
        help=f"easy-query JSON array (default: {DEFAULT_EASY_INPUT})",
    )
    return parser.parse_args(argv)


def load_queries(path: Path) -> list[str]:
    """Load and validate query strings from a JSON array."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array")

    queries: list[str] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be an object")
        query = record.get("query")
        if not isinstance(query, str):
            raise ValueError(
                f"{path}: record {index} has no string 'query' field"
            )
        queries.append(query)

    if not queries:
        raise ValueError(f"{path}: expected at least one query")
    return queries


def tokenize(query: str) -> list[str]:
    """Return case-normalized words, ignoring surrounding punctuation."""
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(query)]


def bag_of_words(
    tokenized_queries: Sequence[Sequence[str]],
) -> Counter[str]:
    """Count every occurrence of each non-excluded word."""
    return Counter(
        word
        for tokens in tokenized_queries
        for word in tokens
        if word not in EXCLUDED_COMMON_WORDS
    )


def two_word_phrases(tokens: Sequence[str]) -> set[str]:
    """Return the distinct adjacent two-word phrases in one query."""
    phrases = {" ".join(pair) for pair in zip(tokens, tokens[1:])}
    return phrases - EXCLUDED_COMMON_TWO_WORDS


def format_odds(numerator: int, denominator: int) -> str:
    """Format odds, including the possible zero-denominator cases."""
    if denominator == 0:
        return "inf" if numerator > 0 else "undefined"
    return f"{numerator / denominator:.2f}"


def format_odds_ratio(
    hard_with: int,
    easy_with: int,
    hard_without: int,
    easy_without: int,
) -> str:
    """Format the hard-label odds ratio for queries with vs. without a word."""
    ratio_numerator = hard_with * easy_without
    ratio_denominator = easy_with * hard_without
    if ratio_denominator == 0:
        return "inf" if ratio_numerator > 0 else "undefined"
    return f"{ratio_numerator / ratio_denominator:.2f}"


def describe_word_counts(counts: Sequence[int]) -> dict[str, int | float]:
    """Compute descriptive statistics for per-query word counts."""
    if not counts:
        raise ValueError("cannot describe an empty collection of word counts")
    ordered = sorted(counts)
    return {
        "count": len(ordered),
        "total": sum(ordered),
        "min": ordered[0],
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "max": ordered[-1],
        "stdev": float(statistics.pstdev(ordered)),
    }


def common_openings(
    tokenized_queries: Sequence[Sequence[str]],
    length: int,
    top: int = 10,
) -> list[tuple[str, int]]:
    """Return the most common exact ``length``-word query prefixes."""
    if length < 1:
        raise ValueError("opening length must be at least 1")
    if top < 1:
        raise ValueError("top must be at least 1")

    return count_openings(tokenized_queries, length).most_common(top)


def count_openings(
    tokenized_queries: Sequence[Sequence[str]], length: int
) -> Counter[str]:
    """Count all exact ``length``-word query prefixes."""
    return Counter(
        " ".join(tokens[:length])
        for tokens in tokenized_queries
        if len(tokens) >= length
    )


def analyze_queries(
    queries: Sequence[str],
    top: int = 10,
    easy_queries: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the word-count and query-opening report."""
    if top < 1:
        raise ValueError("top must be at least 1")
    if not queries:
        raise ValueError("expected at least one query")

    tokenized = [tokenize(query) for query in queries]
    hard_word_counts = [len(tokens) for tokens in tokenized]
    hard_bag = bag_of_words(tokenized)
    word_frequencies = Counter(
        word
        for tokens in tokenized
        for word in set(tokens)
        if word not in EXCLUDED_COMMON_WORDS
    )
    two_word_frequencies = Counter(
        phrase
        for tokens in tokenized
        for phrase in two_word_phrases(tokens)
    )
    report = {
        "bag_of_words": {
            "hard": {
                "total": sum(hard_bag.values()),
                "vocabulary_size": len(hard_bag),
                "most_common": hard_bag.most_common(COMMON_WORD_LIMIT),
                "frequencies": hard_bag,
            }
        },
        "word_counts": describe_word_counts(hard_word_counts),
        "word_count_values": {"hard": hard_word_counts},
        "analyzed_word_total": sum(word_frequencies.values()),
        "common_words": word_frequencies.most_common(COMMON_WORD_LIMIT),
        "common_two_words": two_word_frequencies.most_common(COMMON_WORD_LIMIT),
        "common_openings": {
            length: common_openings(tokenized, length, top)
            for length in PREFIX_LENGTHS
        },
    }

    if easy_queries is not None:
        easy_tokenized = [tokenize(query) for query in easy_queries]
        easy_bag = bag_of_words(easy_tokenized)
        report["bag_of_words"]["easy"] = {
            "total": sum(easy_bag.values()),
            "vocabulary_size": len(easy_bag),
            "most_common": easy_bag.most_common(COMMON_WORD_LIMIT),
            "frequencies": easy_bag,
        }
        report["word_count_values"]["easy"] = [
            len(tokens) for tokens in easy_tokenized
        ]
        report["easy_word_counts"] = describe_word_counts(
            report["word_count_values"]["easy"]
        )
        hard_query_count = len(queries)
        easy_query_count = len(easy_queries)
        total_query_count = hard_query_count + easy_query_count
        report["query_difficulty_summary"] = {
            "hard": hard_query_count,
            "easy": easy_query_count,
            "total": total_query_count,
            "hard_ratio": hard_query_count / total_query_count,
        }
        hard_formal_writing_count = sum(
            bool(set(tokens) & FROMAL_WRITING_STYLE_WORDS) for tokens in tokenized
        )
        easy_formal_writing_count = sum(
            bool(set(tokens) & FROMAL_WRITING_STYLE_WORDS)
            for tokens in easy_tokenized
        )
        report["formal_writing_style_summary"] = {
            "hard_with": hard_formal_writing_count,
            "easy_with": easy_formal_writing_count,
            "hard_without": hard_query_count - hard_formal_writing_count,
            "easy_without": easy_query_count - easy_formal_writing_count,
        }
        hard_tone_count = sum(
            bool(set(tokens) & TONE_WORDS) for tokens in tokenized
        )
        easy_tone_count = sum(
            bool(set(tokens) & TONE_WORDS) for tokens in easy_tokenized
        )
        report["tone_words_summary"] = {
            "hard_with": hard_tone_count,
            "easy_with": easy_tone_count,
            "hard_without": hard_query_count - hard_tone_count,
            "easy_without": easy_query_count - easy_tone_count,
        }
        hard_style_adjective_count = sum(
            bool(set(tokens) & STYLE_ADJECTIVES) for tokens in tokenized
        )
        easy_style_adjective_count = sum(
            bool(set(tokens) & STYLE_ADJECTIVES) for tokens in easy_tokenized
        )
        report["style_adjectives_summary"] = {
            "hard_with": hard_style_adjective_count,
            "easy_with": easy_style_adjective_count,
            "hard_without": hard_query_count - hard_style_adjective_count,
            "easy_without": easy_query_count - easy_style_adjective_count,
        }
        easy_word_frequencies = Counter(
            word
            for tokens in easy_tokenized
            for word in set(tokens)
            if word not in EXCLUDED_COMMON_WORDS
        )
        report["word_difficulty_counts"] = {
            word: (hard_count, easy_word_frequencies[word])
            for word, hard_count in report["common_words"]
        }
        easy_two_word_frequencies = Counter(
            phrase
            for tokens in easy_tokenized
            for phrase in two_word_phrases(tokens)
        )
        report["two_word_difficulty_counts"] = {
            phrase: (hard_count, easy_two_word_frequencies[phrase])
            for phrase, hard_count in report["common_two_words"]
        }
        hard_word_query_counts = Counter(
            word
            for tokens in tokenized
            for word in set(tokens)
            if word not in EXCLUDED_COMMON_WORDS
        )
        easy_word_query_counts = Counter(
            word
            for tokens in easy_tokenized
            for word in set(tokens)
            if word not in EXCLUDED_COMMON_WORDS
        )
        report["word_query_independence"] = {
            word: {
                "hard": hard_word_query_counts[word],
                "easy": easy_word_query_counts[word],
                "expected_hard": (
                    (hard_word_query_counts[word] + easy_word_query_counts[word])
                    * report["query_difficulty_summary"]["hard_ratio"]
                ),
            }
            for word, _ in report["common_words"]
        }
        report["two_word_query_independence"] = {
            phrase: {
                "hard": hard_count,
                "easy": easy_two_word_frequencies[phrase],
                "expected_hard": (
                    (hard_count + easy_two_word_frequencies[phrase])
                    * report["query_difficulty_summary"]["hard_ratio"]
                ),
            }
            for phrase, hard_count in report["common_two_words"]
        }
        difficulty_counts = {}
        for length in PREFIX_LENGTHS:
            easy_counts = count_openings(easy_tokenized, length)
            difficulty_counts[length] = {
                phrase: (hard_count, easy_counts[phrase])
                for phrase, hard_count in report["common_openings"][length]
            }
        report["difficulty_counts"] = difficulty_counts

    return report


def plot_bag_of_words(report: dict[str, Any]) -> None:
    """Display side-by-side word clouds for hard and easy vocabulary."""
    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib and wordcloud are required to display the word clouds. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    bags = report["bag_of_words"]
    panels = [
        (
            "Hard queries",
            bags["hard"],
            ("#222222", "#3f3f46", "#991b1b", "#dc2626", "#ef4444"),
        ),
    ]
    if "easy" in bags:
        panels.append(
            (
                "Easy queries",
                bags["easy"],
                ("#222222", "#3f3f46", "#1e3a8a", "#2563eb", "#3b82f6"),
            )
        )

    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(16 if len(panels) == 2 else 8, 7),
        squeeze=False,
    )
    figure.patch.set_facecolor("white")

    for axis, (title, bag, palette) in zip(axes[0], panels):
        def color_from_palette(
            word: str,
            font_size: int,
            position: tuple[int, int],
            orientation: int | None,
            random_state: Any = None,
            **_kwargs: Any,
        ) -> str:
            del word, font_size, position, orientation
            if random_state is None:
                return palette[0]
            return palette[random_state.randint(0, len(palette) - 1)]

        cloud = WordCloud(
            width=1000,
            height=700,
            background_color="white",
            color_func=color_from_palette,
            max_words=WORD_CLOUD_LIMIT,
            prefer_horizontal=0.9,
            relative_scaling=0.5,
            collocations=False,
            random_state=42,
            margin=3,
        ).generate_from_frequencies(bag["frequencies"])
        axis.imshow(cloud, interpolation="bilinear")
        axis.axis("off")
        axis.set_title(
            f"{title}\n{bag['total']:,} tokens · "
            f"{bag['vocabulary_size']:,} unique words",
            fontsize=15,
            fontweight="bold",
            color="#252832",
            pad=16,
        )

    figure.suptitle(
        "Vocabulary word clouds: hard vs. easy queries",
        y=0.98,
        fontsize=18,
        fontweight="bold",
        color="#20232c",
    )
    figure.text(
        0.5,
        0.935,
        "Larger words occur more often · common excluded words are omitted",
        ha="center",
        fontsize=10,
        color="#686d79",
    )
    figure.tight_layout(rect=(0.015, 0.02, 0.985, 0.91), w_pad=2)
    try:
        figure.canvas.manager.set_window_title("Word-cloud comparison")
    except AttributeError:
        pass
    plt.show()


def plot_word_count_histogram(report: dict[str, Any]) -> None:
    """Display overlaid word-count histograms for hard and easy queries."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to display the word-count histogram. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    counts = report["word_count_values"]
    hard_counts = counts["hard"]
    easy_counts = counts.get("easy")
    all_counts = hard_counts + (easy_counts or [])
    # Center each bin on an integer word count. Density normalization keeps the
    # distributions comparable even when the datasets have different sizes.
    bins = [value - 0.5 for value in range(min(all_counts), max(all_counts) + 2)]

    plt.figure(figsize=(10, 6))
    plt.hist(
        hard_counts,
        bins=bins,
        alpha=0.6,
        density=True,
        label="Hard queries",
        color="tab:red",
        edgecolor="white",
    )
    if easy_counts is not None:
        plt.hist(
            easy_counts,
            bins=bins,
            alpha=0.6,
            density=True,
            label="Easy queries",
            color="tab:blue",
            edgecolor="white",
        )
    plt.title("Query word-count distribution")
    plt.xlabel("Words per query")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def print_report(report: dict[str, Any], path: Path) -> None:
    stats = report["word_counts"]
    if "query_difficulty_summary" in report:
        summary = report["query_difficulty_summary"]
        print("Query counts")
        print(f"  Hard queries: {summary['hard']:,}")
        print(f"  Easy queries: {summary['easy']:,}")
        print(f"  Total queries: {summary['total']:,}")
        print(
            f"  Hard/total: {summary['hard_ratio']:.4f} "
            f"({summary['hard_ratio']:.2%})"
        )
        print()

    if "formal_writing_style_summary" in report:
        formal = report["formal_writing_style_summary"]
        formal_total = formal["hard_with"] + formal["easy_with"]
        hard_percentage = (
            formal["hard_with"] / formal_total * 100 if formal_total else 0.0
        )
        easy_percentage = (
            formal["easy_with"] / formal_total * 100 if formal_total else 0.0
        )
        odds_with = format_odds(formal["hard_with"], formal["easy_with"])
        odds_without = format_odds(
            formal["hard_without"], formal["easy_without"]
        )
        odds_ratio = format_odds_ratio(
            formal["hard_with"],
            formal["easy_with"],
            formal["hard_without"],
            formal["easy_without"],
        )
        print("Formal writing style words")
        print(
            "  Hard queries containing at least one formal writing style word: "
            f"{formal['hard_with']:,}"
        )
        print(
            "  Easy queries containing at least one formal writing style word: "
            f"{formal['easy_with']:,}"
        )
        print(f"  Total matching queries: {formal_total:,}")
        print(
            f"  Among matching queries: hard {hard_percentage:.2f}%, "
            f"easy {easy_percentage:.2f}%"
        )
        print(f"  Hard odds with a formal word: {odds_with}")
        print(f"  Hard odds without a formal word: {odds_without}")
        print(
            "  Odds ratio (hard, formal word present vs. absent): "
            f"{odds_ratio}"
        )
        print()

    if "tone_words_summary" in report:
        tone = report["tone_words_summary"]
        tone_total = tone["hard_with"] + tone["easy_with"]
        hard_percentage = (
            tone["hard_with"] / tone_total * 100 if tone_total else 0.0
        )
        easy_percentage = (
            tone["easy_with"] / tone_total * 100 if tone_total else 0.0
        )
        odds_with = format_odds(tone["hard_with"], tone["easy_with"])
        odds_without = format_odds(tone["hard_without"], tone["easy_without"])
        odds_ratio = format_odds_ratio(
            tone["hard_with"],
            tone["easy_with"],
            tone["hard_without"],
            tone["easy_without"],
        )
        print("Tone words")
        print(
            "  Hard queries containing at least one tone word: "
            f"{tone['hard_with']:,}"
        )
        print(
            "  Easy queries containing at least one tone word: "
            f"{tone['easy_with']:,}"
        )
        print(f"  Total matching queries: {tone_total:,}")
        print(
            f"  Among matching queries: hard {hard_percentage:.2f}%, "
            f"easy {easy_percentage:.2f}%"
        )
        print(f"  Hard odds with a tone word: {odds_with}")
        print(f"  Hard odds without a tone word: {odds_without}")
        print(
            "  Odds ratio (hard, tone word present vs. absent): "
            f"{odds_ratio}"
        )
        print()

    if "style_adjectives_summary" in report:
        style_adjectives = report["style_adjectives_summary"]
        style_adjectives_total = (
            style_adjectives["hard_with"] + style_adjectives["easy_with"]
        )
        hard_percentage = (
            style_adjectives["hard_with"] / style_adjectives_total * 100
            if style_adjectives_total
            else 0.0
        )
        easy_percentage = (
            style_adjectives["easy_with"] / style_adjectives_total * 100
            if style_adjectives_total
            else 0.0
        )
        odds_with = format_odds(
            style_adjectives["hard_with"], style_adjectives["easy_with"]
        )
        odds_without = format_odds(
            style_adjectives["hard_without"], style_adjectives["easy_without"]
        )
        odds_ratio = format_odds_ratio(
            style_adjectives["hard_with"],
            style_adjectives["easy_with"],
            style_adjectives["hard_without"],
            style_adjectives["easy_without"],
        )
        print("Style adjectives")
        print(
            "  Hard queries containing at least one style adjective: "
            f"{style_adjectives['hard_with']:,}"
        )
        print(
            "  Easy queries containing at least one style adjective: "
            f"{style_adjectives['easy_with']:,}"
        )
        print(f"  Total matching queries: {style_adjectives_total:,}")
        print(
            f"  Among matching queries: hard {hard_percentage:.2f}%, "
            f"easy {easy_percentage:.2f}%"
        )
        print(f"  Hard odds with a style adjective: {odds_with}")
        print(f"  Hard odds without a style adjective: {odds_without}")
        print(
            "  Odds ratio (hard, style adjective present vs. absent): "
            f"{odds_ratio}"
        )
        print()

    print(f"Query analytics: {path}")
    print("\nWord-count statistics")
    word_count_groups = [("Hard queries", stats)]
    if "easy_word_counts" in report:
        word_count_groups.append(("Easy queries", report["easy_word_counts"]))

    for label, group_stats in word_count_groups:
        print(f"  {label}")
        print(f"    Queries: {group_stats['count']:,}")
        print(f"    Minimum: {group_stats['min']}")
        print(f"    Mean: {group_stats['mean']:.2f}")
        print(f"    Median: {group_stats['median']:.2f}")
        print(f"    Maximum: {group_stats['max']}")
        print(f"    Standard deviation: {group_stats['stdev']:.2f}")

    print(f"\nTop {COMMON_WORD_LIMIT} most common words")
    print(
        "  Counts and overall percentages are for hard queries "
        "containing the word."
    )
    if "word_difficulty_counts" in report:
        print(
            "  Hard/easy percentages are among all hard + easy queries "
            "containing the word."
        )
    if "word_query_independence" in report:
        print(
            "  Independence hypothesis: whether a query contains the word is "
            "independent of its easy/hard label."
        )
        print(
            "  Expected hard is the number of hard queries expected, under that "
            "hypothesis, among all queries containing the word."
        )
        print(
            "  Hard odds are hard/easy; the odds ratio compares queries with "
            "the word to queries without it."
        )
    common_words = report["common_words"]
    if not common_words:
        print("  (none)")
    else:
        width = max(len(word) for word, _ in common_words)
        for word, frequency in common_words:
            print()
            total = stats["count"]
            percentage = frequency / total * 100 if total else 0.0
            difficulty = ""
            hard_percentage = 0.0
            if "word_difficulty_counts" in report:
                hard_count, easy_count = report["word_difficulty_counts"][word]
                combined_count = hard_count + easy_count
                hard_percentage = hard_count / combined_count * 100
                easy_percentage = easy_count / combined_count * 100
                difficulty = (
                    f" [hard: {hard_count:,} ({hard_percentage:.2f}%), "
                    f"easy: {easy_count:,} ({easy_percentage:.2f}%)]"
                )
            expected = ""
            if "word_query_independence" in report and hard_percentage >70:
                query_counts = report["word_query_independence"][word]
                containing_query_count = query_counts["hard"] + query_counts["easy"]
                summary = report["query_difficulty_summary"]
                hard_without = summary["hard"] - query_counts["hard"]
                easy_without = summary["easy"] - query_counts["easy"]
                odds_with = format_odds(query_counts["hard"], query_counts["easy"])
                odds_without = format_odds(hard_without, easy_without)
                odds_ratio = format_odds_ratio(
                    query_counts["hard"],
                    query_counts["easy"],
                    hard_without,
                    easy_without,
                )
                expected = (
                    f" [expected hard: {query_counts['expected_hard']:.2f} "
                    f"of {containing_query_count:,} queries containing the word; "
                    f"hard odds with: {odds_with}, without: {odds_without}, "
                    f"odds ratio: odds of being hard is {odds_ratio}x higher when this word appears]"
                )
            print(
                f"  {word:<{width}}  {frequency:,} "
                f"({percentage:.2f}%){difficulty}{expected}"
            )

    print(f"\nTop {COMMON_WORD_LIMIT} most common two-word phrases")
    print(
        "  Counts and overall percentages are for hard queries "
        "containing the phrase."
    )
    if "two_word_difficulty_counts" in report:
        print(
            "  Hard/easy percentages are among all hard + easy queries "
            "containing the phrase."
        )
    if "two_word_query_independence" in report:
        print(
            "  Independence hypothesis: whether a query contains the phrase is "
            "independent of its easy/hard label."
        )
        print(
            "  Expected hard is the number of hard queries expected, under that "
            "hypothesis, among all queries containing the phrase."
        )
        print(
            "  Hard odds are hard/easy; the odds ratio compares queries with "
            "the phrase to queries without it."
        )
    common_two_words = report["common_two_words"]
    if not common_two_words:
        print("  (none)")
    else:
        width = max(len(phrase) for phrase, _ in common_two_words)
        for phrase, frequency in common_two_words:
            print()
            total = stats["count"]
            percentage = frequency / total * 100 if total else 0.0
            difficulty = ""
            hard_percentage = 0.0
            if "two_word_difficulty_counts" in report:
                hard_count, easy_count = report["two_word_difficulty_counts"][
                    phrase
                ]
                combined_count = hard_count + easy_count
                hard_percentage = hard_count / combined_count * 100
                easy_percentage = easy_count / combined_count * 100
                difficulty = (
                    f" [hard: {hard_count:,} ({hard_percentage:.2f}%), "
                    f"easy: {easy_count:,} ({easy_percentage:.2f}%)]"
                )
            expected = ""
            if "two_word_query_independence" in report and hard_percentage > 50:
                query_counts = report["two_word_query_independence"][phrase]
                containing_query_count = query_counts["hard"] + query_counts["easy"]
                summary = report["query_difficulty_summary"]
                hard_without = summary["hard"] - query_counts["hard"]
                easy_without = summary["easy"] - query_counts["easy"]
                odds_with = format_odds(query_counts["hard"], query_counts["easy"])
                odds_without = format_odds(hard_without, easy_without)
                odds_ratio = format_odds_ratio(
                    query_counts["hard"],
                    query_counts["easy"],
                    hard_without,
                    easy_without,
                )
                expected = (
                    f" [expected hard: {query_counts['expected_hard']:.2f} "
                    f"of {containing_query_count:,} queries containing the phrase; "
                    f"hard odds with: {odds_with}, without: {odds_without}, "
                    "odds ratio: odds of being hard is "
                    f"{odds_ratio}x higher when this phrase appears]"
                )
            print(
                f"  {phrase:<{width}}  {frequency:,} "
                f"({percentage:.2f}%){difficulty}{expected}"
            )

    if "difficulty_counts" in report:
        print(
            "\nDifficulty split: percentage of each prefix's combined "
            "hard + easy occurrences"
        )

    for length in PREFIX_LENGTHS:
        print(f"\nTop {length}-word openings")
        openings = report["common_openings"][length]
        if "difficulty_counts" in report:
            filtered_openings = []
            for phrase, frequency in openings:
                hard_count, easy_count = report["difficulty_counts"][length][
                    phrase
                ]
                combined_count = hard_count + easy_count
                if combined_count and hard_count / combined_count * 100 > 90:
                    filtered_openings.append((phrase, frequency))
            openings = filtered_openings
        if not openings:
            print("  (none)")
            continue
        width = max(len(phrase) for phrase, _ in openings)
        for phrase, frequency in openings:
            percentage = frequency / stats["count"] * 100
            difficulty = ""
            if "difficulty_counts" in report:
                hard_count, easy_count = report["difficulty_counts"][length][phrase]
                combined_count = hard_count + easy_count
                hard_percentage = hard_count / combined_count * 100
                easy_percentage = easy_count / combined_count * 100
                difficulty = (
                    f" [hard: {hard_percentage:.2f}%, "
                    f"easy: {easy_percentage:.2f}%]"
                )
            print(
                f"  {phrase:<{width}}  {frequency:,} "
                f"({percentage:.2f}%){difficulty}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top < 1:
        print("error: --top must be at least 1", file=sys.stderr)
        return 2

    try:
        queries = load_queries(args.input)
        easy_queries = load_queries(args.easy_input)
        report = analyze_queries(queries, top=args.top, easy_queries=easy_queries)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    plot_bag_of_words(report)
    print_report(report, args.input)
    plot_word_count_histogram(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
