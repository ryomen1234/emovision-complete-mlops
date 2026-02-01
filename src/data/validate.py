# data validation

import os
from pathlib import Path 
from collections import Counter
from typing import List, Optional, Set

class DataValidation:
    
    def __init__(
        self,
        raw_data_path: Path,
        allowed_extensions: Optional[Set[str]] = None,
        splits: Optional[List[str]] = None
    ) -> None:
        self.raw_data_dir = raw_data_path
        self.allowed_extensions = allowed_extensions or {".jpg", ".png", ".jpeg"}
        self.splits = splits or ["train", "test"]
    
    def validate_splits(self):
        for split in self.splits:
            split_path = self.raw_data_dir / split 
            if not split_path.exists():
                raise FileNotFoundError(f"Missing split folder: {split}")

    def get_class(self, split) -> List[str]:
        split_path = self.raw_data_dir / split
        class_dirs = [d.stem for d in split_path.iterdir()]

        return class_dirs
    
    def validate_class_consistency(self):
        base_class = self.get_class(self.splits[0])
        for split in self.splits[1:]:
            if self.get_class(split) != base_class:
                raise ValueError("Class Mismatch between train and test.")
        
        return base_class
    
    def validate_image(self, split, class_name) -> int:
        class_path = self.raw_data_dir / split / class_name
        images = list(class_path.iterdir())

        if not images:
            raise ValueError(f"Empty folder: {class_path}")
        
        for img in images:
           if img.suffix.lower() not in self.allowed_extensions:
               raise ValueError(f"Invalid imgage format: {img}")
        
        return len(images)
    
    def run_validation(self):
        self.validate_splits()
        classes = self.validate_class_consistency()

        report = {}
        for split in self.splits:
            report[split] = {}
            for cls in classes:
                count = self.validate_image(split, cls)
                report[split][cls] = count 
        
        return report

