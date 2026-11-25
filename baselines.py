"""Fit and check the two ticket classification baselines."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from prepare import SPLIT_SEED, sample_training_rows, split_tickets
from tickets_data import make_tickets


def fit_baselines(train):
    most_common = train["label"].value_counts().idxmax()
    tfidf_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000),
    )
    tfidf_model.fit(train["text"], train["label"])
    return most_common, tfidf_model


def main():
    train, validation, _ = split_tickets(make_tickets())
    sample = sample_training_rows(train, size=400, seed=SPLIT_SEED)
    most_common, tfidf_model = fit_baselines(sample)

    labels = validation["label"]
    majority_score = accuracy_score(labels, [most_common] * len(labels))
    tfidf_score = accuracy_score(labels, tfidf_model.predict(validation["text"]))

    print(f"Training sample: {len(sample)} | Validation rows: {len(validation)}")
    print(f"Most-common-label accuracy: {majority_score:.4f}")
    print(f"TF-IDF + logistic regression accuracy: {tfidf_score:.4f}")


if __name__ == "__main__":
    main()
