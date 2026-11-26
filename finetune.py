"""Train DistilBERT to classify Lakeshore support tickets."""

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from prepare import SPLIT_SEED, sample_training_rows, split_tickets
from tickets_data import LABELS, make_tickets

MODEL_ID = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 128
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = dict(enumerate(LABELS))


def build_model(mode):
    if mode not in {"head_only", "full"}:
        raise ValueError("Mode must be 'head_only' or 'full'.")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(LABELS),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
        use_safetensors=True,
    )
    if mode == "head_only":
        model.distilbert.requires_grad_(False)
    return model


def tokenize_tickets(frame, tokenizer):
    rows = []
    for text, label in zip(frame["text"], frame["label"], strict=True):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_token_type_ids=False,
        )
        encoded["labels"] = LABEL_TO_ID[label]
        rows.append(encoded)
    return rows


def make_loader(frame, tokenizer, batch_size, *, shuffle=False, seed=SPLIT_SEED):
    return DataLoader(
        tokenize_tickets(frame, tokenizer),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


@torch.no_grad()
def predict_labels(model, loader):
    model.eval()
    predictions = []

    for batch in loader:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        predicted_ids = model(**inputs).logits.argmax(dim=1).tolist()
        predictions.extend(ID_TO_LABEL[index] for index in predicted_ids)

    return predictions


def train_distilbert(
    train,
    validation,
    *,
    mode,
    seed,
    epochs=3,
    batch_size=16,
    learning_rate=5e-5,
):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = build_model(mode)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )

    train_loader = make_loader(train, tokenizer, batch_size, shuffle=True, seed=seed)
    validation_loader = make_loader(validation, tokenizer, batch_size)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        if mode == "head_only":
            model.distilbert.eval()

        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch["labels"])

        validation_predictions = predict_labels(model, validation_loader)
        result = {
            "epoch": epoch,
            "training_loss": total_loss / len(train),
            "validation_accuracy": accuracy_score(
                validation["label"], validation_predictions
            ),
        }
        history.append(result)
        print(
            f"{mode} | epoch {epoch}/{epochs} | "
            f"loss: {result['training_loss']:.4f} | "
            f"validation accuracy: {result['validation_accuracy']:.4f}",
            flush=True,
        )

    return model, tokenizer, history


def main():
    """Check the training path before running the full experiment."""
    train, validation, _ = split_tickets(make_tickets())
    sample = sample_training_rows(train, size=24, seed=SPLIT_SEED)

    model, _, _ = train_distilbert(
        sample,
        validation,
        mode="head_only",
        seed=SPLIT_SEED,
        epochs=1,
        batch_size=8,
    )

    print("Trainable parameters:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name)


if __name__ == "__main__":
    main()
