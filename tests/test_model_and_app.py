"""Test DistilBERT training, model export, and the Gradio prediction function."""

import json

import gradio as gr
import pandas as pd
import pytest
import torch
from transformers import (
    AutoTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
)

import finetune
import share
from space.app import classify_ticket, load_classifier
from tickets_data import LABELS


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(finetune.MODEL_ID)


@pytest.fixture
def tiny_tickets():
    return pd.DataFrame(
        {
            "text": ["Refund please", "Broken pedal", "Dock stuck", "Password reset"],
            "label": LABELS,
        }
    )


def make_tiny_model(tokenizer):
    config = DistilBertConfig(
        vocab_size=len(tokenizer),
        n_layers=1,
        n_heads=2,
        dim=16,
        hidden_dim=32,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
        num_labels=len(LABELS),
        id2label=finetune.ID_TO_LABEL,
        label2id=finetune.LABEL_TO_ID,
    )
    return DistilBertForSequenceClassification(config)


def test_tokenization_truncates_and_keeps_labels(tokenizer):
    tickets = pd.DataFrame({"text": ["payment " * 300], "label": ["billing"]})
    encoded = finetune.tokenize_tickets(tickets, tokenizer)[0]

    assert len(encoded["input_ids"]) == finetune.MAX_LENGTH
    assert encoded["labels"] == finetune.LABEL_TO_ID["billing"]


@pytest.mark.parametrize("mode", ["head_only", "full"])
def test_one_epoch_updates_only_expected_weights(
    mode, tokenizer, tiny_tickets, monkeypatch
):
    model = make_tiny_model(tokenizer)
    initial_body = [p.detach().clone() for p in model.distilbert.parameters()]
    initial_head = model.classifier.weight.detach().clone()

    monkeypatch.setattr(
        finetune.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )
    trained, _, history = finetune.train_distilbert(
        tiny_tickets,
        tiny_tickets,
        mode=mode,
        seed=2026,
        epochs=1,
        batch_size=4,
        learning_rate=0.01,
    )

    body_parameters = list(trained.distilbert.parameters())
    body_changed = any(
        not torch.equal(before, after)
        for before, after in zip(initial_body, body_parameters, strict=True)
    )

    assert body_changed == (mode == "full")
    assert all(p.requires_grad == (mode == "full") for p in body_parameters)
    assert not torch.equal(initial_head, trained.classifier.weight)
    assert len(history) == 1


def test_same_seed_repeats_training(tokenizer, tiny_tickets, monkeypatch):
    monkeypatch.setattr(
        finetune.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *_args, **_kwargs: make_tiny_model(tokenizer),
    )
    settings = {"mode": "full", "seed": 2026, "epochs": 1, "batch_size": 2}

    first, _, _ = finetune.train_distilbert(tiny_tickets, tiny_tickets, **settings)
    second, _, _ = finetune.train_distilbert(tiny_tickets, tiny_tickets, **settings)

    torch.testing.assert_close(first.state_dict(), second.state_dict())


def test_tiny_model_can_memorize_a_batch(tokenizer, tiny_tickets):
    torch.manual_seed(2026)
    model = make_tiny_model(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    loader = finetune.make_loader(tiny_tickets, tokenizer, batch_size=4)
    batch = next(iter(loader))

    for _ in range(60):
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()

    assert finetune.predict_labels(model, loader) == tiny_tickets["label"].tolist()


@pytest.fixture
def saved_model(tokenizer, tmp_path):
    model = make_tiny_model(tokenizer).eval()
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    return model, str(tmp_path)


def test_saved_model_reloads_with_matching_predictions(
    tokenizer, tiny_tickets, saved_model
):
    original, path = saved_model
    loaded_tokenizer, reloaded = load_classifier(path)

    before = finetune.make_loader(tiny_tickets, tokenizer, batch_size=4)
    after = finetune.make_loader(tiny_tickets, loaded_tokenizer, batch_size=4)

    assert finetune.predict_labels(original, before) == finetune.predict_labels(
        reloaded, after
    )


def test_app_returns_four_scores_that_sum_to_one(saved_model):
    _, path = saved_model
    scores = classify_ticket("My pedal is broken.", path)

    assert set(scores) == set(LABELS)
    assert sum(scores.values()) == pytest.approx(1.0)


def test_app_rejects_blank_input():
    with pytest.raises(gr.Error, match="Enter a support ticket"):
        classify_ticket("   ", "unused-model-path")


def test_model_card_has_required_headings(tmp_path, monkeypatch):
    report = tmp_path / "final.json"
    scores = {"accuracy": 0.9, "macro_f1": 0.9}
    report.write_text(
        json.dumps(
            {
                "settings": {
                    "train_size": 400,
                    "seed": 2026,
                    "epochs": 3,
                    "learning_rate": 5e-5,
                },
                "test": {
                    "n_rows": 240,
                    "most_common": scores,
                    "tfidf_logistic": scores,
                    "full": scores,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(share, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(share, "REPORT", report)

    share.write_model_card()

    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n")
    for heading in ["Intended use", "Training data", "Results", "Limitations"]:
        assert f"## {heading}" in card
