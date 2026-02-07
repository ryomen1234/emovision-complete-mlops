from pathlib import Path
from PIL import Image
from typing import Tuple
from tqdm import tqdm
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataPreprocess:

    def __init__(
        self,
        raw_data_dir: Path,
        processed_data_dir: Path,
        image_size: Tuple = (224, 224),
    ):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.image_size = image_size

    def _process_img(self, image_path: Path) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.image_size)
        return img

    def run(self):
        logger.info("Preprocessing started")
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"Processed data directory: {self.processed_data_dir}")
        logger.info(f"Image size: {self.image_size}")

        use_tqdm = logger.isEnabledFor(logging.DEBUG)

        for split in ["test", "train"]:
            split_dir = self.raw_data_dir / split

            if not split_dir.exists():
                logger.warning(f"Split directory missing: {split_dir}")
                continue

            logger.info(f"Processing split: {split}")

            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue

                images = list(class_dir.iterdir())

                if not images:
                    logger.warning(f"No images found in {class_dir}")
                    continue

                logger.info(
                    f"Processing class '{class_dir.name}' "
                    f"({len(images)} images)"
                )

                for img_path in tqdm(
                    images,
                    desc=f"{split}/{class_dir.name}",
                    unit="img",
                    leave=False,
                    disable=not use_tqdm,
                ):
                    try:
                        proc_img = self._process_img(img_path)

                        save_path = (
                            self.processed_data_dir
                            / split
                            / class_dir.name
                            / img_path.name
                        )

                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        proc_img.save(save_path)

                    except Exception:
                        logger.exception(
                            f"Failed to process image: {img_path}"
                        )

        logger.info("Preprocessing completed successfully")
