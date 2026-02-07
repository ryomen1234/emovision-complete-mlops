from pathlib import Path
from torch.utils.data import DataLoader
from src.features.transforms import get_transform
from src.features.dataset import EmovisionDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_dataloader(processed_data_dir: Path, batch_size: int = 32):

    logger.info("Initializing dataloaders")
    logger.info(f"Processed data directory: {processed_data_dir}")
    logger.info(f"Batch size: {batch_size}")

    train_dataset = EmovisionDataset(
        processed_data_dir / "train",
        transform=get_transform(train=True)
    )

    test_dataset = EmovisionDataset(
        processed_data_dir / "test",
        transform=get_transform(train=False)
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    logger.info("Dataloaders created successfully")

    return train_loader, test_loader
