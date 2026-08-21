"""Turn the queries in all_queries.json into annotated spaCy Doc objects."""

import json
from pathlib import Path

import spacy
from spacy.tokens import Doc


QUERIES_PATH = Path(__file__).resolve().parent.parent / "all_queries_fewMS.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "preprocessed_queries_fewMS.json"


def preprocess_queries(json_path: Path = QUERIES_PATH) -> tuple[list[dict], list[Doc]]:
    """Load query texts and return one linguistically annotated Doc per query."""
    with json_path.open(encoding="utf-8") as file:
        records = json.load(file)

    queries = [record["query"] for record in records]
    nlp = spacy.load("en_core_web_sm")

    # nlp.pipe is the efficient way to process many texts at once.
    return records, list(nlp.pipe(queries))


def save_docs(records: list[dict], docs: list[Doc], output_path: Path = OUTPUT_PATH) -> None:
    """Save the useful annotations from the Doc objects as JSON."""
    annotated_queries = []

    for record, doc in zip(records, docs):
        annotated_queries.append(
            {
                "id": record["id"],
                "query": doc.text,
                "tokens": [
                    {
                        "text": token.text,
                        "lemma": token.lemma_,
                        "pos": token.pos_,
                        "dependency": token.dep_,
                        "morphology": str(token.morph),
                        "entity": token.ent_type_,
                    }
                    for token in doc
                ],
                "entities": [
                    {"text": entity.text, "label": entity.label_}
                    for entity in doc.ents
                ],
            }
        )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(annotated_queries, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    records, docs = preprocess_queries()
    save_docs(records, docs)
    print(f"Created {len(docs)} annotated spaCy Doc objects.")
    print(f"Saved annotations to {OUTPUT_PATH}.")

    # Example: inspect the annotations of the first query.
    if docs:
        print("\nToken\tLemma\tPOS\tDependency\tMorphology\tEntity")
        for token in docs[0]:
            print(
                f"{token.text}\t{token.lemma_}\t{token.pos_}\t"
                f"{token.dep_}\t{token.morph}\t{token.ent_type_}"
            )
