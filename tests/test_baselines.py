"""Check the shared split, training samples, and validation baselines."""

import pytest
from sklearn.metrics import accuracy_score

from baselines import fit_baselines
from prepare import SPLIT_SEED, sample_training_rows, split_tickets
from tickets_data import make_tickets


@pytest.mark.parametrize("seed", [2026, 2027, 2028])
def test_split_sizes_and_no_overlap(seed):
    train, validation, test = split_tickets(make_tickets(), seed=seed)

    assert (len(train), len(validation), len(test)) == (720, 240, 240)
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)


def test_sample_contains_only_training_rows_and_all_labels():
    train, _, _ = split_tickets(make_tickets())
    sample = sample_training_rows(train, size=100, seed=SPLIT_SEED)

    assert len(sample) == 100
    assert set(sample.index).issubset(train.index)
    assert set(sample["label"]) == {"billing", "bike_fault", "station", "account"}


def test_tfidf_beats_most_common_label_on_validation():
    train, validation, _ = split_tickets(make_tickets())
    sample = sample_training_rows(train, size=400, seed=SPLIT_SEED)
    most_common, tfidf_model = fit_baselines(sample)

    labels = validation["label"]
    majority_score = accuracy_score(labels, [most_common] * len(labels))
    tfidf_score = accuracy_score(labels, tfidf_model.predict(validation["text"]))

    assert tfidf_score > majority_score
