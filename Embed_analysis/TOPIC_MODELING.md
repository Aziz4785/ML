# Hard-query topic modeling

`topic_model_hard_queries.py` implements the pipeline quoted in the request:

1. keep English-tagged prompts when such a field exists and remove normalized
   duplicate prompt text;
2. embed prompts with SentenceTransformers `all-mpnet-base-v2`;
3. reduce embeddings to five dimensions with UMAP;
4. cluster with HDBSCAN using a minimum cluster size of 20;
5. choose examples from the top 20% of each original cluster by HDBSCAN
   membership probability, keeping prompts under 100 words;
6. ask GPT-4o for a narrow name and description (optional);
7. reassign outliers first with HDBSCAN soft probabilities, then with embedding
   similarity, and update the final BERTopic representations.

## Setup

The existing project environment uses Python 3.9, while the current BERTopic
stack needs a newer interpreter. A Python 3.11 executable is already available
on this machine:

```bash
/Users/aziz/.local/bin/python3.11 -m venv .venv-topic
source .venv-topic/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-topic-modeling.txt
```

The first full run downloads the MPNet model and its PyTorch dependencies.

For GPT labels, set `OPENAI_API_KEY` in the environment or the repository's
`.env` file. Label responses use the Responses API with a strict JSON schema and
are cached after every topic, so an interrupted run can resume without paying
to label completed topics again. Use `--skip-labeling` for a fully local run.

## Run

Validate and profile preprocessing without installing the ML dependencies:

```bash
python3 topic_model_hard_queries.py hard_queries.json --validate-only
```

Run the complete quoted pipeline:

```bash
source .venv-topic/bin/activate
python topic_model_hard_queries.py hard_queries.json \
  --output hard_query_topics.json
```

Run clustering without OpenAI API calls:

```bash
python topic_model_hard_queries.py hard_queries.json \
  --skip-labeling \
  --output hard_query_topics.json
```

Useful controls:

- `--min-cluster-size 20`: requested HDBSCAN minimum.
- `--min-samples N`: optionally tune HDBSCAN's noise sensitivity; by default it
  follows `min_cluster_size`.
- `--representative-fraction 0.20`: strict high-probability example pool.
- `--representative-fraction 1.0`: relax the pool for this small dataset so a
  cluster can contribute up to 20 examples.
- `--outlier-probability-threshold 0.05`: first-pass minimum soft probability.
- `--outlier-embedding-threshold 0.0`: second-pass minimum cosine similarity.
- `--allow-outliers`: do not force the last unmatched prompts to their nearest
  original cluster centroid.
- `--language-field language`: only keep values tagged `en`, `eng`, or
  `english`; repeat `--english-value VALUE` to configure other tags.

Run `python topic_model_hard_queries.py --help` for every option.

## Output

The output is one JSON document containing:

- preprocessing counts and the complete model configuration;
- overall topic/outlier counts after each reduction stage;
- each topic's name, description, BERTopic keywords, initial/final size, and
  representative prompts with exact HDBSCAN membership probabilities;
- one assignment per retained source prompt, including the initial topic, final
  topic, whether it was an outlier, and which reassignment method was used;
- installed package versions for reproducibility.

The separate `<output stem>.labels.json` file is only an API-response cache. It
is safe to retain between equivalent runs.

## Important interpretation notes

`hard_queries.json` has 300 unique English prompts and no language tag, so the
script records that English was assumed rather than pretending a tag-based
filter occurred. All 300 prompts are already under 100 words.

The source recipe's two example requirements cannot both hold for small
clusters: the top 20% of a 20-document cluster contains only four prompts. The
default keeps the top-20% rule exact, returns fewer than 20 where necessary, and
records the shortfall in every topic. Pass `--representative-fraction 1.0` if
having up to 20 examples is more important than the percentile restriction.

This dataset is also dominated by short, templated phrases such as "text where"
and "text beginning with". Topic labels may therefore describe rhetorical form
or retrieval-query construction rather than ordinary end-user subject areas.

Finally, the supplied `chatbotArena-1-8.pdf` is an older method. Its section 6.1
uses OpenAI `text-embedding-3-small`, HDBSCAN minimum size 32, ten examples, and
GPT-4-Turbo. The implementation intentionally follows the later quoted recipe,
not those conflicting PDF parameters.

## References

- BERTopic outlier reduction and chained strategies:
  https://maartengr.github.io/BERTopic/getting_started/outlier_reduction/outlier_reduction.html
- BERTopic parameter tuning:
  https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html
- OpenAI Structured Outputs:
  https://developers.openai.com/api/docs/guides/structured-outputs
