from ultralytics import YOLOWorld
from ultralytics.engine.results import Boxes

from src.utils import save_detection_results

# Initialize a YOLO-World model
model = YOLOWorld("yolov8x-world.pt")

# Define custom classes
model.set_classes(
    ["phone", "mask", "glasses"]
)  # <--------- Change this to the class you want to detect

# Execute prediction on an image
results: Boxes = model.predict(
    "samples/vietnam-3.jpg", max_det=100, iou=0.01, conf=0.01
)

# Save detection results as images
save_detection_results(results)
