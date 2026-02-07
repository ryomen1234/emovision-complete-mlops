from src.data.ingest import DataIngestion
from pathlib import Path 

if __name__ == "__main__":
    src_dir = Path(r"data\sources\archive.zip")
    extract_dir = Path(r"data\raw")
    
    d = DataIngestion(src_dir, extract_dir)
    d.ingest()
    