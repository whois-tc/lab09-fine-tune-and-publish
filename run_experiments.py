"""Compare four methods on matched training samples."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from baselines import fit_baselines
from evaluate import plot_learning_curve
from final_evaluation import run_final_evaluation
from finetune import train_distilbert
from prepare import SPLIT_SEED, sample_training_rows, split_tickets
from tickets_data import make_tickets

SIZES = (100, 200, 400)
SEEDS = (2026, 2027, 2028)
MODES = ("head_only", "full")
MAX_EPOCHS = 3
LEARNING_RATE = 5e-5
REPORT_DIR = Path("reports")


def run_grid(train, validation):
    baseline_rows = []
    epoch_rows = []
    labels = validation["label"]

    for size in SIZES:
        for seed in SEEDS:
            print(f"\nTraining rows: {size} | Seed: {seed}", flush=True)
            sample = sample_training_rows(train, size=size, seed=seed)
            run = {"train_size": size, "seed": seed}

            most_common, tfidf_model = fit_baselines(sample)
            baseline_predictions = {
                "most_common": [most_common] * len(validation),
                "tfidf_logistic": tfidf_model.predict(validation["text"]),
            }
            for method, predictions in baseline_predictions.items():
                baseline_rows.append(
                    {
                        **run,
                        "method": method,
                        "validation_accuracy": accuracy_score(labels, predictions),
                    }
                )

            for mode in MODES:
                model, _, history = train_distilbert(
                    sample,
                    validation,
                    mode=mode,
                    seed=seed,
                    epochs=MAX_EPOCHS,
                    learning_rate=LEARNING_RATE,
                )
                for epoch in history:
                    epoch_rows.append({**run, "method": mode, **epoch})
                del model

    return pd.DataFrame(baseline_rows), pd.DataFrame(epoch_rows)


def main():
    parser = argparse.ArgumentParser()
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--smoke", action="store_true")
    run_mode.add_argument("--final-only", action="store_true")
    args = parser.parse_args()

    train, validation, test = split_tickets(make_tickets())

    if args.final_only:
        run_final_evaluation(train, validation, test, REPORT_DIR, LEARNING_RATE)
        return

    if args.smoke:
        sample = sample_training_rows(train, size=24, seed=SPLIT_SEED)
        train_distilbert(
            sample,
            validation,
            mode="full",
            seed=SPLIT_SEED,
            epochs=1,
            batch_size=8,
            learning_rate=LEARNING_RATE,
        )
        return

    baselines, epochs = run_grid(train, validation)

    # Choose one epoch for both DistilBERT modes using validation only.
    largest = epochs[epochs["train_size"] == max(SIZES)]
    mean_by_epoch = largest.groupby("epoch")["validation_accuracy"].mean()
    selected_epoch = int(mean_by_epoch.idxmax())

    selected = epochs[epochs["epoch"] == selected_epoch]
    columns = ["train_size", "seed", "method", "validation_accuracy"]
    results = pd.concat(
        [baselines[columns], selected[columns]], ignore_index=True
    ).sort_values(["train_size", "seed", "method"])

    summary = (
        results.groupby(["train_size", "method"])["validation_accuracy"]
        .agg(mean_accuracy="mean", min_accuracy="min", max_accuracy="max")
        .reset_index()
    )

    REPORT_DIR.mkdir(exist_ok=True)
    epochs.to_csv(REPORT_DIR / "epoch_history.csv", index=False)
    results.to_csv(REPORT_DIR / "results.csv", index=False)
    summary.to_csv(REPORT_DIR / "validation_summary.csv", index=False)
    plot_learning_curve(summary, REPORT_DIR / "learning_curve.png")

    print(f"\nSelected epoch for both DistilBERT modes: {selected_epoch}")
    print(summary.to_string(index=False))
    print(f"\nSaved {len(results)} comparison rows to {REPORT_DIR / 'results.csv'}")

    run_final_evaluation(train, validation, test, REPORT_DIR, LEARNING_RATE)


if __name__ == "__main__":
    main()
