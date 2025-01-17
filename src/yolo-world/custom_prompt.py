from ultralytics import YOLOWorld
from ultralytics.engine.results import Boxes

from src.utils import save_detection_results

# Initialize a YOLO-World model
model = YOLOWorld("yolov8s-world.pt")

# Define custom classes
model.set_classes(["bus"])  # <--------- Change this to the class you want to detect

# Execute prediction on an image
results: Boxes = model.predict("samples/bus.jpg")

# Save detection results as images
save_detection_results(results)
