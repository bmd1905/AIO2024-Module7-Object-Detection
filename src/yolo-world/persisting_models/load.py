from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO("yolov8s-world.pt")  # or select yolov8m/l-world.pt

# Define custom classes
model.set_classes(["person", "bus"])

# Save the model with the defined offline vocabulary
model.save("custom_yolov8s.pt")
