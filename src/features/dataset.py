from pathlib import Path 
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image


class EmovisionDataset(Dataset):

    def __init__(self, data_dir: Path, transform=None) -> None:
        self.data_dir = data_dir
        self.transform = transform

        self.class_to_idx = {}
        self.samples = []

        self._load_samples()


    def _load_samples(self):
        classes = [d.stem for d in self.data_dir.iterdir() if self.data_dir.exists()]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for cls in classes:
            class_path = self.data_dir / cls
            for img_path in class_path.iterdir():
                self.samples.append((img_path, self.class_to_idx[cls]))
  
    def __len__(self)->int:
        return len(self.samples)
        

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        imgage = Image.open(img_path).convert("RGB")

        if self.transform:
            imgage = self.transform(imgage)

        return imgage, torch.tensor(label, dtype=torch.long)
    