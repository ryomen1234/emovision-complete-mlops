import torch 
import torch.optim as optim 
import torch.nn as nn 
from pathlib import Path 
import mlflow 
from mlflow import pytorch as mlflow_pytorch

from src.features.dataloader import get_dataloader
from src.models.model import Model_0
from src.utils.logger import get_logger

logger = get_logger(__name__)

def _train(
    processed_data_dir: Path,
    epochs=5,
    batch_size=32,
    lr=0.001,
    model_dir=Path("artifacts")
):
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    

    train_loader, test_loader = get_dataloader(
        Path("data/processed"),
        batch_size=8
    )

    num_classes = len(train_loader.dataset.class_to_idx)
    model = Model_0(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("Emotion Detection")

    with mlflow.start_run():

        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "model": "SimpleCNN"
        })

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            train_loss = running_loss / len(train_loader)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc
            }, step=epoch)

            logger.info(
                f"Epoch {epoch+1}/{epoch}"
                f"Loss={train_loss:.4f} Acc={train_acc:.4f}"
            )

           
        
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model.pth"
        torch.save(model.state_dict(), model_path)

        mlflow.log_artifact(str(model_path))
        mlflow_pytorch.log_model(model, "model")

        logger.info("Training Completed Successfully.")
