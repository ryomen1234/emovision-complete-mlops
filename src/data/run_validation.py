from src.data.validate import DataValidation
from pprint import pprint
from pathlib import Path

if __name__ == "__main__":
    raw_data_path = Path("data/raw")
    
    dv = DataValidation(raw_data_path)
    report = dv.run_validation()

    pprint(report)
