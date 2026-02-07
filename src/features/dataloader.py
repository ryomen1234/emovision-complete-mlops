from pathlib import Path
from torch.utils.data import DataLoader, Subset
from src.features.transforms import get_transform
from src.features.dataset import EmovisionDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


import random

def get_dataloaders(
    processed_data_dir: Path,
    batch_size=32,
    train_fraction=1.0   
):
    train_dataset = EmovisionDataset(
        processed_data_dir / "train",
        transform=get_transform(train=True)
    )

    test_dataset = EmovisionDataset(
        processed_data_dir / "test",
        transform=get_transform(train=False)
    )

    # 🔽 Use only a fraction of training data
    if train_fraction < 1.0:
        total_size = len(train_dataset)
        subset_size = int(total_size * train_fraction)

        indices = random.sample(range(total_size), subset_size)
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader
