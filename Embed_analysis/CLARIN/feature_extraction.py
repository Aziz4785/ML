from collections import Counter
import re

import pandas as pd
from spacy.tokens import Doc, Span
from sklearn.feature_extraction.text import CountVectorizer


VALID_FEATURES = {10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 40, 50, 51, 52, 60, 61}


def _ngrams(tokens, n):
    """Return the contiguous n-grams in an already-tokenized sequence."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _ngram_labels(tokens, ngram_range):
    """Represent n-grams as strings without re-tokenizing symbolic features."""
    # Match CountVectorizer's historical behavior: empty annotations (for
    # example, non-entity tokens in feature 60) are separators, not features.
    tokens = [token for token in tokens if token]
    min_n, max_n = ngram_range
    return [
        " ".join(ngram)
        for n in range(min_n, max_n + 1)
        for ngram in _ngrams(tokens, n)
    ]


def _store_feature_names(frame):
    """Store model and display feature names in DataFrame metadata."""
    feature_names = frame.columns.to_list()
    feature_names_waterfall = [
        re.sub(r"_\d+$", "", name.replace("=", ":"))
        for name in feature_names
    ]
    frame.attrs["feature_names"] = feature_names
    frame.attrs["feature_names_waterfall"] = feature_names_waterfall
    return frame


def choose_features(doc, feature=10, verbose=0, string=True):
    """
    Extracts selected type of features from Doc.

    Parameters
    ----------
    doc : spacy.tokens.doc.Doc or spacy.tokens.span.Span
        The input document preprocessed by Spacy.
    feature : int
        The numeral for selecting the feature type.

    Returns
    -------
    str
        A string containing the selected space-separated features.

    Notes
    -----
    The function supports counting different types of features based on the provided feature IDs.
    Available feature IDs and their corresponding types:
    - Tokens:
        - 10: Select lowercased words without punctuation.
        - 11: Select lemmas without punctuation.
        - 12: Select lowercased words without punctuation, replacing named entities with their entity type.
        - 13: Select lemmas without punctuation, replacing named entities with their entity type.
    - Token N-grams, dependency-based:
        - 20: Select lowercased dependency-based word bigrams, including punctuation and excluding numeral children.
        - 21: Select dependency-based lemma bigrams, including punctuation and excluding numeral children.
        - 22: Select lowercased dependency-based word bigrams, including punctuation, excluding numeral children, and replacing named entities with their entity type.
        - 23: Select dependency-based lemma bigrams, including punctuation, excluding numeral children, and replacing named entities with their entity type.
    - Part-of-speech tags:
        - 30: Select all parts of speech.
        - 31: Select parts of speech without punctuation.
        - 32: Select parts of speech without whitespace tokens.
    - Dependency-based tags:
        - 40: Select dependency labels without punctuation.
    - Morphology annotation:
        - 50: Select morphology annotations with punctuation.
        - 51: Select morphology annotations without punctuation.
        - 52: Select morphology annotations with punctuation, replacing named entities with their entity type.
    - Named entities:
        - 60: Select the named-entity type for every token (empty for non-entity tokens).
        - 61: Select all named entities.

    """
    if not isinstance(doc, (Doc, Span)):
        if verbose<2:
            print("Warning: Invalid document format. Please provide a single spacy.tokens.doc.Doc or spacy.tokens.span.Span.\nSee Notes in help(choose_features).")
        return
    if not isinstance(feature, int):
        if verbose<2:
            print("Warning: Invalid features format. Please provide a single integer.\nSee Notes in help(choose_features).")
        return
    elif feature not in VALID_FEATURES:
        if verbose<2:
            print("Warning: Invalid feature provided: {} \nSee Notes in `help(choose_features)' for available options.".format(feature))
        return
    # TO DO: nie wszystkie dane mają tytuły tekstów! Zastąpić czymś ogólniejszym.
    # if verbose<1:
    #         print("From {}, {}:".format(doc.doc._.author,doc.doc._.title))

    
    # VERSION 1:
    # ---------lowercased words without punctuation
    if(feature==10):
        if verbose<1:
            print("-- Extracting all lowercased words (no punctuation).")
        words = [token.lower_ for token in doc if not token.is_punct] # tokens that arent punctuations; wystarczy mniej cech
    # noun tokens that arent stop words or punctuations
    # if (not token.is_stop and not token.is_punct and token.pos_ == "NOUN")

    # ---------lemmas without punctuation
    elif(feature==11):
        if verbose<1:
            print("-- Extracting all lemmas (no punctuation).")
        words = [token.lemma_ for token in doc if not token.is_punct]    
    # ---------lowercased words without punctuation, replacing named entities
        # tokens that arent punctuations
        # replaces a token with its named entity type if it is a part of an entity ("San Francisco" -> "placeName placeName")    
    elif(feature==12):
        if verbose<1:
            print("-- Extracting lowercased words (no punctuation, replacing named entities with their entity type).")
        words = [token.text.lower() if token.ent_type_ == '' 
                 else token.ent_type_ for token in doc if not token.is_punct]
    # ---------lemmas without punctuation, replacing named entities
    elif(feature==13):
        if verbose<1:
            print("-- Extracting lemmas (no punctuation, replacing named entities with their entity type).")
        words = [token.lemma_ if token.ent_type_ == '' 
                 else token.ent_type_ for token in doc if not token.is_punct]
    
    # dependency-based bigrams: ancestor_child
    # ---------lowercased words and punctuation, no numeral children
    elif(feature==20):
        if verbose<1:
            print("-- Extracting lowercased dependency-based word bigrams (including punctuation, excluding numeral children).")
        words = ['_'.join([token.lower_, f.lower_]) for token in doc
         for f in token.children if not f.like_num]
    # ---------lemmas and punctuation, no numeral children
    elif(feature==21):
        if verbose<1:
            print("-- Extracting dependency-based lemma bigrams (including punctuation, excluding numeral children).")
        words = ['_'.join([token.lemma_, f.lemma_]) for token in doc
         for f in token.children if not f.like_num]
    # ---------lowercased words and punctuation, replacing named entities, no numeral children
    elif(feature==22):
        if verbose<1:
            print("-- Extracting lowercased dependency-based word bigrams (including punctuation, excluding numeral children, replacing named entities with their entity type).")
        words = ['_'.join([token.lower_ if token.ent_type_ == '' else token.ent_type_,
                           f.lower_  if f.ent_type_ == '' else f.ent_type_])
                 for token in doc for f in token.children if not f.like_num]
    # ---------lemmas and punctuation, replacing named entities, no numeral children
    elif(feature==23):
        if verbose<1:
            print("-- Extracting dependency-based lemma bigrams (including punctuation, excluding numeral children, replacing named entities with their entity type).")
        words = ['_'.join([token.lemma_ if token.ent_type_ == '' else token.ent_type_,
                           f.lemma_  if f.ent_type_ == '' else f.ent_type_])
                 for token in doc for f in token.children if not f.like_num]    
    
    # VERSION 3: POS
    elif(feature==30):
        if verbose<1:
            print("-- Extracting all parts of speech.")
        words = [token.pos_ for token in doc]
    # ---------without punctuation
    elif(feature==31):
        if verbose<1:
            print("-- Extracting all parts of speech (no punctuation).")
        words = [token.pos_ for token in doc if not token.is_punct]
    # ---------without whitespace tokens
    elif(feature==32):
        if verbose<1:
            print("-- Extracting all parts of speech (no whitespace tokens).")
        words = [token.pos_ for token in doc if not token.pos_ == 'SPACE']
    
    # VERSION 4: dependency without punctuation
    elif(feature==40):
        if verbose<1:
            print("-- Extracting dependency labels without punctuation.")
        words = [token.dep_ for token in doc if not token.is_punct]
        
    # VERSION 5: morphology annotation with punctuation
    elif(feature==50):
        if verbose<1:
            print("-- Extracting morphology annotations with punctuation.")
        words = [str(token.morph) for token in doc]

    # ---------without punctuation
    elif(feature==51):
        if verbose<1:
            print("-- Extracting morphology annotations without punctuation.")
        words = [str(token.morph) for token in doc if not token.is_punct] 
        
    # ---------morphology with punctuation, replacing named entities
    elif(feature==52):
        if verbose<1:
            print("-- Extracting morphology annotations with punctuation (replacing named entities with their entity type).")
        words = [str(token.morph) if token.ent_type_ == '' else token.ent_type_ for token in doc]

    # VERSION 6: same NER-y
    # ---------NER type
    elif(feature==60):
        if verbose<1:
            print("-- Extracting the named-entity type for every token (empty for non-entity tokens).")
        words = [token.ent_type_ for token in doc]
    # ---------all NERs
    elif(feature==61):
        if verbose<1:
            print("-- Extracting all named entities.")
        words = [token.text for token in doc.ents]

    if string:
        words = ' '.join(words)
        
    return words


def count_features_list(docs,
                        feature_list,
                        verbose=0):
    """
    Count the given features in each document, supporting varying n-gram sizes,
    efficient n-gram generation, and integer counts.

    Parameters
    ----------
    docs : list of spacy Doc or Span
        The input documents for feature extraction.
    feature_list : list of str
        Features in the form "<feature_string>_<feature_type_id>".
    verbose : int
        Verbosity level; prints progress if > 0.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (len(docs), len(feature_list)) with integer counts.
    """
    #---- START Check input types
    if isinstance(docs, (Doc, Span)):
        docs = [docs] # Make a list if only one Doc is given.
    elif not isinstance(docs, list) or not all(isinstance(doc, (Doc, Span)) for doc in docs):
        if verbose<2:
            print("Warning: Invalid document format. Please provide a single spacy.tokens.doc.Doc, spacy.tokens.span.Span or a list of such objects.")
        return
    if verbose<1:
        print("Number of documents provided: {}.".format(len(docs)))


    # Map each feature to its column index
    feature_positions = {feat: idx for idx, feat in enumerate(feature_list)}

    # Organize requested columns by feature type and n-gram size.
    suffix_map = {}
    for feat in feature_list:
        try:
            feat_text, suffix = feat.rsplit("_", 1)
            feature_type = int(suffix)
        except (AttributeError, ValueError):
            raise ValueError(
                f"Invalid feature name {feat!r}; expected '<feature>_<feature_type_id>'."
            ) from None
        if feature_type not in VALID_FEATURES:
            raise ValueError(f"Invalid feature type in {feat!r}: {feature_type}")
        tokens = tuple(feat_text.split())
        suffix_map.setdefault(feature_type, {}).setdefault(len(tokens), []).append(tokens)

    # Prepare result storage
    results = [[0] * len(feature_list) for _ in docs]

    # Process each document
    for i, doc in enumerate(docs):
        if verbose:
            print(f"Processing document {i+1}/{len(docs)}")

        # For each feature type suffix
        for feature_type, ngram_groups in suffix_map.items():
            tokens = choose_features(doc, feature=feature_type, string=False, verbose=2)

            # Generate and count n-grams per size
            counters = {}
            for n, feature_tuples in ngram_groups.items():
                counters[n] = Counter(_ngrams(tokens, n))

            # Fill counts for each feature in this suffix
            for n, feature_tuples in ngram_groups.items():
                counter_n = counters[n]
                for feat_tuple in feature_tuples:
                    count = counter_n.get(feat_tuple, 0)
                    feat_name = f"{' '.join(feat_tuple)}_{feature_type}"
                    col_idx = feature_positions[feat_name]
                    results[i][col_idx] = count

    # Build DataFrame with integer dtype
    df = pd.DataFrame(results, columns=feature_list, dtype="int64")
    
    return _store_feature_names(df)


def count_features(docs,
                   feature_scheme = {'features':[13,20, 21,23,30,52, 51, 60, 61],
                                     'max_features':1000,
                                     'n_grams_word':(1,3), #useful for feature 0 to 20
                                     'n_grams_pos':(1,3),
                                     'n_grams_dep':(1,3),
                                     'n_grams_morph':(1,1),
                                     'min_cull_word':0., # ignore terms that have a document frequency strictly lower than the given threshold
                                     'max_cull_word':1., # ignore terms that have a document frequency strictly higher than the given threshold
                                     'min_cull_d2':0.,
                                     'max_cull_d2':1.,
                                     'remove_duplicates':True},
                   verbose = 0,
                   features = None):
    """
    Build a vocabulary from symbolic feature sequences and count it.

    Each requested feature type is extracted with :func:`choose_features`. Its
    contiguous n-grams become vocabulary entries, and the resulting count
    matrices are concatenated into one numeric row per document. Column names
    have the form ``<symbolic n-gram>_<feature id>``. ``features`` can override
    only the feature IDs while retaining every other setting in
    ``feature_scheme``.
    """
    selected_features = feature_scheme['features'] if features is None else features
    print(" max number of features that can be extracted : ", feature_scheme['max_features'])
    print(" and we will extract features using only the methods: ", selected_features)
    #---- START Check input types
    if isinstance(docs, (Doc, Span)):
        docs = [docs] # Make a list if only one Doc is given.
    elif not isinstance(docs, list) or not all(isinstance(doc, (Doc, Span)) for doc in docs):
        if verbose<2:
            print("Warning: Invalid document format. Please provide a single spacy.tokens.doc.Doc, spacy.tokens.span.Span or a list of such objects.")
        return
    if verbose<1:
        print("Number of documents provided: {}.".format(len(docs)))
    

    features = selected_features
    max_features = feature_scheme['max_features']
    n_grams_word = feature_scheme['n_grams_word']
    n_grams_pos = feature_scheme['n_grams_pos']
    n_grams_dep = feature_scheme['n_grams_dep']
    n_grams_morph = feature_scheme['n_grams_morph']
    min_cull_word = feature_scheme['min_cull_word']
    max_cull_word = feature_scheme['max_cull_word']
    min_cull_d2 = feature_scheme['min_cull_d2']
    max_cull_d2 = feature_scheme['max_cull_d2']
    # ``remove_duplicates`` may still be present in older schemes, but the
    # complete vocabulary is intentionally retained.
    
    if isinstance(features, int):
        features = [features] # Make a list if only one int is given.
    elif not isinstance(features, list) or not all(isinstance(f, int) for f in features):
        if verbose<2:
            print("Warning: Invalid features format. Please provide a single integer or a list of integers.")
        return
    
    invalid_features = [f for f in features if f not in VALID_FEATURES]
    features = [f for f in features if f in VALID_FEATURES]
    
    if invalid_features:
        if verbose<2:
            print("Warning: Invalid features were provided and will be dropped: {} \nSee Notes in `help(choose_features)' for available options.".format(", ".join(str(f) for f in invalid_features)))

    if verbose<1:
            print("Features to be extracted: {}.".format(features))
    
#     TO DO: add metadata checking from preprocess_spacy.spacy_save_docs()
    
    #---- END Check input types
     

    feature_params = {
        range(0, 20): {
            'max_features': max_features,
            'ngram_range': n_grams_word,
            'min_df': min_cull_word,
            'max_df': max_cull_word
        },
        range(20, 30): {
            'max_features': max_features,
            'min_df': min_cull_d2,
            'max_df': max_cull_d2
        },
        range(30, 40): {
            'max_features': max_features,
            'ngram_range': n_grams_pos,
        },
        range(40, 50): {
            'max_features': max_features,
            'ngram_range': n_grams_dep,
        },
        range(50, 60): {
            'max_features': max_features,
            'ngram_range': n_grams_morph,
        },
        range(60, 70): {
            'max_features': max_features,
        }
    }
    print(" feature_params: ", feature_params)

    
    #---- START Counting features depending on their type
    frames = []
    for lt in features:
        #find the first bucket that contains this value"
        params = next((params for rng, params in feature_params.items() if lt in rng), None)
        if params is None:
            if verbose < 2:
                print("Warning: Invalid feature {}. Skipping...".format(lt))
            continue
        print(" for feature {} we will use params: {}".format(lt, params))
        # More than one requested feature can belong to the same category.
        # Copy its settings before removing our n-gram-only option.
        params = dict(params)
        ngram_range = params.pop('ngram_range', (1, 1))
        print(f" now we iterate over all the queries and apply feature {lt}")
        symbolic_sequences = [
            choose_features(doc, feature=lt, verbose=1 if i > 0 else verbose, string=False)
            for i, doc in enumerate(docs)
        ]
        print(" with that we get a lit of sequences of length: ", len(symbolic_sequences))
        print(" First element type:", type(symbolic_sequences[0]))
        print(" First inner length:", len(symbolic_sequences[0]))
        print(" now for each element of that list we will generate all N-gram with N from {} to {}".format(ngram_range[0], ngram_range[1]))
        ngram_documents = [
            _ngram_labels(sequence, ngram_range)
            for sequence in symbolic_sequences
        ]
        print(" we put that in another list of course, and it will ook like this (first 2 elements) : ", ngram_documents[:2])
        #so ngram_documents is a list of lists of n-grams (1, 2 and 3 grams depending on the parameters).
        #for example the first element of ngram_documents could be : ['short', 'text', 'contain', 'the', 'answer', 'for', 'what', 'be', 'rba', 'short text', 'text contain', 'contain the', 'the answer', 'answer for', 'for what', 'what be', 'be rba', 'short text contain', 'text contain the', 'contain the answer', 'the answer for', 'answer for what', 'for what be', 'what be rba']
        # The identity analyzer is intentional: choose_features has already
        # tokenized the input. This preserves punctuation, case, morphology,
        # one-character tokens, and dependency symbols exactly.
        print(" now each n-gram will be considered as a column and we will count how many times it appears in the document")
        count_vect = CountVectorizer(analyzer=lambda items: items, **params)
        try:
            counts = count_vect.fit_transform(ngram_documents)
        except ValueError as exc:
            if "empty vocabulary" in str(exc).lower() or "no terms remain" in str(exc).lower():
                continue
            raise
        print(" it will produce a. matrix and this matrix will have shape: ", counts.shape)
        names = [f"{name}_{lt}" for name in count_vect.get_feature_names_out()]
        print(f" it has {len(names)} columns, and the first 5 are: {names[:5]}. (yes we appended feature code to each column)")
        print(" we then trnasform that matrix to a panda df and append that matrix to the list of matrices we will concatenate at the end")
        frames.append(pd.DataFrame(counts.toarray(), columns=names, dtype="int64"))
            
    #---- END Counting features depending on their type
    
    if not frames:
        return _store_feature_names(pd.DataFrame(index=range(len(docs))))

    # Feature ids make column names unambiguous across symbolic sequence types.
    # Keeping all columns is required for a complete count-vector vocabulary;
    # remove_duplicates remains accepted in old schemes for compatibility.
    return _store_feature_names(pd.concat(frames, axis=1))



if __name__ == "__main__":
    import spacy

    # A blank pipeline only tokenizes text. The trained pipeline also supplies
    # the lemmas, dependency parse, POS tags, morphology, and named entities
    # needed by the feature modes below.
    nlp = spacy.load("en_core_web_sm")
    example_doc = nlp("Choose useful features from this sentence.")
    print(choose_features(example_doc, feature=11))
    print()
    print(choose_features(example_doc, feature=12))
    print()
    print(choose_features(example_doc, feature=13))
    print()
    print(choose_features(example_doc, feature=20))
    print()
    print(choose_features(example_doc, feature=21))
    print()
    print(choose_features(example_doc, feature=22))
    print()
    print(choose_features(example_doc, feature=23))
    print()
    print(choose_features(example_doc, feature=30))
    print()
    print(choose_features(example_doc, feature=31))
    print()
    print(choose_features(example_doc, feature=32))
    print(choose_features(example_doc, feature=40))
    print(choose_features(example_doc, feature=51))
    print(choose_features(example_doc, feature=52))
    print(choose_features(example_doc, feature=60))
    print(choose_features(example_doc, feature=61))
    print()
    print()
    count_vector = count_features(example_doc, verbose=0)
    print(f"Count vector size: {count_vector.iloc[0].size}")
    print(f"Number of columns added: {count_vector.shape[1]}")
