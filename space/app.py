"""Show the predicted support queue and scores for a ticket."""

import os
from functools import cache

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@cache
def load_classifier(model_source, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def classify_ticket(text, model_source, device="cpu"):
    if not text or not text.strip():
        raise gr.Error("Enter a support ticket before submitting.")

    tokenizer, model = load_classifier(model_source, device)
    inputs = tokenizer(
        text.strip(),
        truncation=True,
        max_length=128,
        return_token_type_ids=False,
        return_tensors="pt",
    ).to(device)
    scores = model(**inputs).logits.softmax(dim=1)[0].tolist()

    return {model.config.id2label[index]: score for index, score in enumerate(scores)}


def make_demo(model_source, predict=None):
    def predict_on_cpu(text):
        return classify_ticket(text, model_source)

    return gr.Interface(
        fn=predict if predict is not None else predict_on_cpu,
        inputs=gr.Textbox(label="Support ticket", lines=4),
        outputs=gr.Label(label="Suggested queue", num_top_classes=4),
        title="Lakeshore Bike Share Ticket Classifier",
        description="Enter a fictional ticket. Do not include personal information.",
    )


hub_model_id = os.getenv("MODEL_ID")
if hub_model_id:
    import spaces

    load_classifier(hub_model_id, "cuda")

    @spaces.GPU
    def predict_on_gpu(text):
        return classify_ticket(text, hub_model_id, "cuda")

    demo = make_demo(hub_model_id, predict=predict_on_gpu)

    if __name__ == "__main__":
        demo.launch()
