import torch
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
print(f"torch: {torch.__version__}")
print(f"detectron2: {detectron2.__version__}")
print("Detectron2 imported successfully!")
