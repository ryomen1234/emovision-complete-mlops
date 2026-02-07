from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
import torch
from PIL import Image

class EmovisionDataset(Dataset):

    def __init__(self, data_dir: Path, transform=None) -> None:
        self.data_dir = data_dir
        self.transform = transform

        self.class_to_idx = {}
        self.samples: List[Tuple[Path, int]] = []

        self._load_samples()

    def _load_samples(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        classes = sorted(
            d.name for d in self.data_dir.iterdir() if d.is_dir()
        )

        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            class_path = self.data_dir / cls

            for img_path in class_path.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                self.samples.append((img_path, self.class_to_idx[cls]))

        if not self.samples:
            raise ValueError(f"No valid images found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}") from e

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
