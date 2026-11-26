"""Show the predicted support queue and scores for a ticket."""

import os
from functools import cache

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@cache
def load_classifier(model_source):
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def classify_ticket(text, model_source):
    if not text or not text.strip():
        raise gr.Error("Enter a support ticket before submitting.")

    tokenizer, model = load_classifier(model_source)
    inputs = tokenizer(
        text.strip(),
        truncation=True,
        max_length=128,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    scores = model(**inputs).logits.softmax(dim=1)[0].tolist()

    return {model.config.id2label[index]: score for index, score in enumerate(scores)}


def make_demo(model_source):
    def predict(text):
        return classify_ticket(text, model_source)

    return gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="Support ticket", lines=4),
        outputs=gr.Label(label="Suggested queue", num_top_classes=4),
        title="Lakeshore Bike Share Ticket Classifier",
        description="Enter a fictional ticket. Do not include personal information.",
    )


if __name__ == "__main__":
    make_demo(os.environ["MODEL_ID"]).launch()
