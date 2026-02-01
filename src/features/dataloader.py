from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch 
from src.features.transforms import get_transform 
from src.features.dataset import EmovisionDataset

def get_dataloader(processed_data_dir: Path, batch_size=32):

    train_dataset = EmovisionDataset(
        processed_data_dir / "train",
        transform=get_transform(train=True)

    )

    test_dataset = EmovisionDataset(
        processed_data_dir / "test",
        transform=get_transform(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader