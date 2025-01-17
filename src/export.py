from ultralytics import YOLO

from src.utils import logger

# Load the YOLO11 model
model = YOLO("yolo11x.pt")


# Export the model
model.export(format="engine", half=True, batch=32)

logger.info("Model exported successfully")
