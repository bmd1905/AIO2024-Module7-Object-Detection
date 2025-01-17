from ultralytics import YOLO

# Load your custom model
model = YOLO("custom_yolov8s.pt")

# Run inference to detect your custom classes
results = model.predict("path/to/image.jpg")

# Show results
results[0].show()
