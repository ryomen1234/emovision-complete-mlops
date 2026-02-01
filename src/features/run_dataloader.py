from pathlib import Path
from src.features.dataloader import get_dataloader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloader(
        Path("data/processed"),
        batch_size=8
    )

    images, labels = next(iter(train_loader))
    print(images.shape)  # [8, 3, 224, 224]
    print(labels)
