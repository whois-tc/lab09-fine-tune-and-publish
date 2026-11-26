"""Build the Hugging Face model card from the held-out test results."""

import json
from pathlib import Path

MODEL_DIR = Path("model")
REPORT = Path("reports/final.json")


def write_model_card():
    results = json.loads(REPORT.read_text(encoding="utf-8"))
    settings = results["settings"]
    test = results["test"]

    methods = {
        "most_common": "Most common label",
        "tfidf_logistic": "TF-IDF + logistic regression",
        "full": "Full DistilBERT",
    }
    rows = []
    for key, name in methods.items():
        scores = test[key]
        rows.append(f"| {name} | {scores['accuracy']:.2%} | {scores['macro_f1']:.4f} |")

    card = f"""---
license: apache-2.0
base_model: distilbert/distilbert-base-uncased
pipeline_tag: text-classification
---

# Lakeshore Bike Share Ticket Classifier

## Intended use

Suggest one of four queues for a fictional bike-share support message:
`billing`, `bike_fault`, `station`, or `account`. This is a classroom demo.

## Training data

The model was fine-tuned on {settings["train_size"]} stratified tickets from
the generated training split, using seed {settings["seed"]},
{settings["epochs"]} epochs, and a learning rate of {settings["learning_rate"]}.

## Results

All three methods used the same {settings["train_size"]} training tickets
and were scored on the same {test["n_rows"]} held-out test tickets.

| Method | Accuracy | Macro F1 |
|---|---:|---:|
{chr(10).join(rows)}

## Limitations

The training text was invented. Real tickets may use different language,
contain multiple issues, or lack enough detail for reliable routing.
Do not enter personal information or use this demo to route real tickets.
"""
    (MODEL_DIR / "README.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    write_model_card()
