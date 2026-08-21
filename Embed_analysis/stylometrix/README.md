# Query feature extraction with StyloMetrix

Install StyloMetrix and its English spaCy model:

```bash
python -m pip install -r stylometrix/requirements.txt
python -m spacy download en_core_web_trf
```

Extract all features from `all_queries.json`:

```bash
python stylometrix/extract_query_features.py
```

The default output is `stylometrix/all_queries_stylometrix.json`. It contains
the ordered feature names and, for each input query, its ID (when present),
text, label, and feature vector. StyloMetrix 0.1.9.1 exposes 196 English
features; the script stops with an error if 190 or fewer are returned.

For a quick installation check without processing the full dataset:

```bash
python stylometrix/extract_query_features.py --limit 5
```

Use `--input`, `--output`, and `--batch-size` to override the defaults.

## Decision-tree classification and SHAP explanations

Run the class-balanced decision-tree analysis with:

```bash
python stylometrix/classify_decision_tree.py
```

In addition to cross-validation scores and the fitted tree, the script prints
the 10 features with the largest mean absolute SHAP values and creates:

- `stylometrix/decision_tree.png`
- `stylometrix/shap_feature_importance.png`, a global feature-impact ranking
- `stylometrix/shap_beeswarm.png`, showing the size and direction of each
  feature's impact on predictions for label 1

In the beeswarm plot, positive SHAP values push a prediction toward label 1 and
negative values push it away. Red points are high feature values and blue
points are low feature values. SHAP explains the final tree fitted on all data;
the cross-validation scores remain the estimate of performance on unseen data.
