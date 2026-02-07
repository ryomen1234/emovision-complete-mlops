from pathlib import Path
from typing import List, Optional, Set, Dict
import logging
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataValidation:

    def __init__(
        self,
        raw_data_path: Path,
        allowed_extensions: Optional[Set[str]] = None,
        splits: Optional[List[str]] = None,
    ) -> None:
        self.raw_data_dir = raw_data_path
        self.allowed_extensions = allowed_extensions or {".jpg", ".png", ".jpeg"}
        self.splits = splits or ["train", "test"]

    def validate_splits(self) -> None:
        logger.info("Validating dataset splits")

        for split in self.splits:
            split_path = self.raw_data_dir / split
            if not split_path.exists():
                logger.error(f"Missing split folder: {split_path}")
                raise FileNotFoundError(f"Missing split folder: {split}")

        logger.info("All split folders are present")

    def get_classes(self, split: str) -> List[str]:
        split_path = self.raw_data_dir / split
        return sorted(
            d.name for d in split_path.iterdir() if d.is_dir()
        )

    def validate_class_consistency(self) -> List[str]:
        logger.info("Validating class consistency across splits")

        base_classes = self.get_classes(self.splits[0])

        for split in self.splits[1:]:
            classes = self.get_classes(split)
            if classes != base_classes:
                logger.error(
                    f"Class mismatch detected in split: {split}"
                )
                raise ValueError("Class mismatch between dataset splits")

        logger.info(f"Classes validated: {base_classes}")
        return base_classes

    def validate_images(self, split: str, class_name: str) -> int:
        class_path = self.raw_data_dir / split / class_name
        images = list(class_path.iterdir())

        if not images:
            logger.error(f"Empty class folder: {class_path}")
            raise ValueError(f"Empty folder: {class_path}")

        use_tqdm = logger.isEnabledFor(logging.DEBUG)

        for img in tqdm(
            images,
            desc=f"Validating {split}/{class_name}",
            unit="img",
            leave=False,
            disable=not use_tqdm,
        ):
            if img.suffix.lower() not in self.allowed_extensions:
                logger.error(f"Invalid image format: {img}")
                raise ValueError(f"Invalid image format: {img}")

        return len(images)

    def run_validation(self) -> Dict[str, Dict[str, int]]:
        logger.info("Starting data validation")

        self.validate_splits()
        classes = self.validate_class_consistency()

        report: Dict[str, Dict[str, int]] = {}

        for split in self.splits:
            logger.info(f"Validating split: {split}")
            report[split] = {}

            for cls in classes:
                count = self.validate_images(split, cls)
                report[split][cls] = count

        logger.info("Data validation completed successfully")
        logger.debug(f"Validation report: {report}")

        return report
