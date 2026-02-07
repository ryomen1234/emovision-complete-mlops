import torch 
import torch.optim as optim 
import torch.nn as nn 
from pathlib import Path 
import mlflow 
from mlflow import pytorch as mlflow_pytorch
from tqdm import tqdm
import logging

from src.features.dataloader import get_dataloader
from src.models.model import SimpleCNN
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train(
    processed_data_dir: Path,
    epochs=5,
    batch_size=32,
    lr=1e-3,
    model_dir=Path("artifacts"),
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, _ = get_dataloader(
        processed_data_dir,
        batch_size=batch_size
    )

    num_classes = len(train_loader.dataset.class_to_idx)
    model = SimpleCNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("emotion_detection")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "model": model.__class__.__name__,
        })

        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                unit="batch",
                leave=False,
                disable=not logger.isEnabledFor(logging.DEBUG),
            ):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / len(train_loader)
            train_acc = correct / total

            mlflow.log_metrics(
                {"train_loss": train_loss, "train_accuracy": train_acc},
                step=epoch,
            )

            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss={train_loss:.4f}, Acc={train_acc:.4f}"
            )

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pth"
        torch.save(model.state_dict(), model_path)

        mlflow.log_artifact(str(model_path))
        mlflow_pytorch.log_model(model, "model")

        logger.info("Training completed successfully")
