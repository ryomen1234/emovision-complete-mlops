import os 
from pathlib import Path 
import zipfile

class DataIngestion:

    def __init__(self, zip_path: str, extract_path: str) -> None:
        self.zip_path = Path(zip_path) 
        self.extract_dir = Path(extract_path)
    
    def ingest(self) -> Path:
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Source data not found in {self.zip_path}")
        
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_dir)
        
        return self.extract_dir