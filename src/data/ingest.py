import os 
from pathlib import Path 
import zipfile
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataIngestion:

    def __init__(self, zip_path: Path, extract_path: Path) -> None:
        self.zip_path = zip_path 
        self.extract_dir = extract_path
    
    def ingest(self) -> Path:
        logger.info("Starting data ingestion")
        logger.info(f"Zip path: {self.zip_path}")
        logger.info(f"Extract directory: {self.extract_dir}")

        if not self.zip_path.exists():
            logger.error(f"Source data not found in {self.zip_path}")
            raise FileNotFoundError(f"Source data not found in {self.zip_path}")
        
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extract dir created")
        
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                files = zip_ref.namelist()
                for file in tqdm(
                    files,
                    desc="Extracting files",
                    unit="file",
                    ncols=100
                ):
                 logger.info(f"Extracting {len(files)} files from zip")
                 zip_ref.extractall(self.extract_dir)
        except zipfile.BadZipFile:
            logger.exception(f"Invalid or corrupted zip file")
            raise

        except Exception:
            logger.exception(f"Unexpected error during data ingestion.")
            raise

        logger.info("Data ingestion completed successfully")
        return self.extract_dir