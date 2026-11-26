"""Publish the trained classifier and Gradio Space to Hugging Face."""

from pathlib import Path

from huggingface_hub import HfApi

from share import write_model_card


def main():
    if not Path("model/model.safetensors").is_file():
        raise SystemExit("Saved model not found. Run this from the Lab 9 folder.")

    write_model_card()
    api = HfApi()
    username = api.whoami()["name"]
    model_id = f"{username}/lakeshore-ticket-distilbert"
    space_id = f"{username}/lakeshore-ticket-sorter"

    api.create_repo(model_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=model_id,
        repo_type="model",
        folder_path="model",
        allow_patterns=["*.json", "*.safetensors", "vocab.txt", "README.md"],
    )

    api.add_space_variable(space_id, "MODEL_ID", model_id)
    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path="space",
        allow_patterns=["app.py", "README.md", "requirements.txt"],
    )

    print(f"Model: https://huggingface.co/{model_id}")
    print(f"Space: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
