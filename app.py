"""Launch the ticket classifier with the locally saved model."""

from pathlib import Path

from space.app import make_demo

if __name__ == "__main__":
    model_path = Path(__file__).resolve().parent / "model"
    make_demo(str(model_path)).launch()
