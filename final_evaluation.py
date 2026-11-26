"""Evaluate the locked model and both baselines on held-out tickets."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

from baselines import fit_baselines
from finetune import make_loader, predict_labels, train_distilbert
from prepare import SPLIT_SEED, sample_training_rows
from tickets_data import LABELS

FINAL_SIZE = 400
FINAL_MODE = "full"
MODEL_DIR = Path("model")
OWN_MESSAGES_PATH = Path("data/own_messages.csv")


def load_own_messages():
    if not OWN_MESSAGES_PATH.is_file():
        raise SystemExit(f"Create {OWN_MESSAGES_PATH} before final evaluation.")

    tickets = pd.read_csv(OWN_MESSAGES_PATH, keep_default_na=False)
    if list(tickets.columns) != ["text", "label"]:
        raise SystemExit("The original-message CSV must have columns text,label.")
    if tickets["text"].str.strip().eq("").any():
        raise SystemExit("The original-message CSV contains an empty message.")
    if tickets["label"].value_counts().to_dict() != dict.fromkeys(LABELS, 10):
        raise SystemExit("Include exactly 10 messages for each of the four queues.")

    return tickets


def score(labels, predictions):
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(
            float(
                f1_score(
                    labels,
                    predictions,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
    }


def run_final_evaluation(train, validation, test, report_dir, learning_rate):
    own_tickets = load_own_messages()

    history = pd.read_csv(report_dir / "epoch_history.csv")
    largest = history[history["train_size"] == FINAL_SIZE]
    selected_epoch = int(
        largest.groupby("epoch")["validation_accuracy"].mean().idxmax()
    )

    sample = sample_training_rows(train, size=FINAL_SIZE, seed=SPLIT_SEED)
    most_common, tfidf_model = fit_baselines(sample)
    model, tokenizer, _ = train_distilbert(
        sample,
        validation,
        mode=FINAL_MODE,
        seed=SPLIT_SEED,
        epochs=selected_epoch,
        learning_rate=learning_rate,
    )

    results = {
        "settings": {
            "method": FINAL_MODE,
            "train_size": FINAL_SIZE,
            "seed": SPLIT_SEED,
            "epochs": selected_epoch,
            "learning_rate": learning_rate,
        }
    }

    for name, rows in [("test", test), ("own_messages", own_tickets)]:
        predictions = predict_labels(model, make_loader(rows, tokenizer, 16))
        results[name] = {
            "n_rows": len(rows),
            "most_common": score(rows["label"], [most_common] * len(rows)),
            "tfidf_logistic": score(rows["label"], tfidf_model.predict(rows["text"])),
            FINAL_MODE: score(rows["label"], predictions),
        }

        if name == "test":
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay.from_predictions(
                rows["label"],
                predictions,
                labels=LABELS,
                xticks_rotation=30,
                colorbar=False,
                ax=ax,
            )
            ax.set_title("Full DistilBERT: held-out test tickets")
            fig.tight_layout()
            fig.savefig(report_dir / "confusion_matrix.png", dpi=150)
            plt.close(fig)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    output = json.dumps(results, indent=2)
    (report_dir / "final.json").write_text(output + "\n", encoding="utf-8")
    print(output)
