# process data
from pathlib import Path 
from PIL import Image 
import numpy as np
from typing import Tuple

class DataPreprocess:

    def __init__(self,
                  raw_data_dir: Path,
                  processed_data_dir: Path,
                  image_size: Tuple = (224, 224)):
        
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.image_size = image_size
    
    def _process_img(self, image_path: Path) -> Image.Image:
        # create a PIL object of the image and convert into "rgb".
        img = Image.open(image_path).convert("RGB")  
        # resize
        img = img.resize(self.image_size)
        
        return img
        
    
    def run(self):
        for split in ["test","train"]:
            split_dir = self.raw_data_dir / split 
            for class_dir in split_dir.iterdir():
                for img in class_dir.iterdir():

                    proc_img = self._process_img(img)

                    save_path = (
                        self.processed_data_dir 
                        / split 
                        / class_dir.name
                        / img.name
                        )
                    
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    proc_img.save(save_path)
    