import spacy

from CLARIN.feature_extraction import count_features, count_features_list


def _word_feature_scheme():
    return {
        "features": [10],
        "max_features": None,
        "n_grams_word": (1, 3),
        "n_grams_pos": (1, 3),
        "n_grams_dep": (1, 3),
        "n_grams_morph": (1, 1),
        "min_cull_word": 0.0,
        "max_cull_word": 1.0,
        "min_cull_d2": 0.0,
        "max_cull_d2": 1.0,
        "remove_duplicates": False,
    }


def test_count_features_builds_one_shared_unigram_to_trigram_vocabulary():
    nlp = spacy.blank("en")
    docs = [nlp("A cat sat cat"), nlp("A dog sat")]

    counts = count_features(docs, _word_feature_scheme(), verbose=2)

    assert counts.loc[0, "cat_10"] == 2
    assert counts.loc[0, "a cat_10"] == 1
    assert counts.loc[0, "cat sat cat_10"] == 1
    assert counts.loc[1, "cat_10"] == 0
    assert counts.loc[1, "a dog sat_10"] == 1
    assert all(dtype.kind in "iu" for dtype in counts.dtypes)


def test_count_features_list_reuses_generated_vocabulary():
    nlp = spacy.blank("en")
    doc = nlp("A cat sat cat")
    vocabulary = count_features(doc, _word_feature_scheme(), verbose=2).columns.tolist()

    counts = count_features_list(doc, vocabulary, verbose=2)

    assert counts.shape == (1, len(vocabulary))
    assert counts.loc[0, "cat_10"] == 2
    assert counts.loc[0, "cat sat cat_10"] == 1


def test_multiple_word_feature_types_each_get_the_requested_ngrams():
    nlp = spacy.blank("en")
    scheme = _word_feature_scheme()
    scheme["features"] = [10, 12]

    counts = count_features(nlp("A cat sat"), scheme, verbose=2)

    assert "a cat sat_10" in counts.columns
    assert "a cat sat_12" in counts.columns


def test_count_features_stores_model_and_waterfall_feature_names():
    nlp = spacy.blank("en")
    counts = count_features(nlp("A cat"), _word_feature_scheme(), verbose=2)

    assert counts.attrs["feature_names"] == counts.columns.tolist()
    assert counts.attrs["feature_names_waterfall"] == ["a", "a cat", "cat"]


def test_count_features_list_stores_model_and_waterfall_feature_names():
    nlp = spacy.blank("en")
    vocabulary = ["Case=Nom_50", "cat_10"]

    counts = count_features_list(nlp("cat"), vocabulary, verbose=2)

    assert counts.attrs["feature_names"] == vocabulary
    assert counts.attrs["feature_names_waterfall"] == ["Case:Nom", "cat"]
