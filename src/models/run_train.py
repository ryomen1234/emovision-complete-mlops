from pathlib import Path
from src.models.train import train

if __name__ == "__main__":
    train(
        processed_data_dir=Path("data/processed"),
        epochs=5,
        batch_size=32,
        lr=0.001
    )
