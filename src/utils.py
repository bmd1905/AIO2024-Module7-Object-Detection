import sys
import uuid
from pathlib import Path

import cv2
from loguru import logger
from ultralytics.engine.results import Boxes

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("run/log.log", level="INFO", rotation="100kb")

# Create run directory if it doesn't exist
Path("./run").mkdir(parents=True, exist_ok=True)


def save_detection_results(results: Boxes) -> list[str]:
    """
    Save detection results as images if detections were found.

    :param results: Detection results from YOLO model prediction, containing bounding boxes and other metadata
    :return: List of paths where annotated images were saved as strings
    """
    # Initialize empty list to store paths of saved images
    saved_paths = []

    # Iterate through each detection result
    for i, result in enumerate(results):
        # Check if any detections were made by looking at number of bounding boxes
        if len(result.boxes) > 0:
            # Plot the detection results with bounding boxes and labels on the image
            annotated_image = result.plot()

            # Generate unique filename using UUID to avoid overwrites
            output_path = f"./run/img_{uuid.uuid4()}.jpg"

            # Save the annotated image to disk using OpenCV
            cv2.imwrite(output_path, annotated_image)

            # Get absolute path and convert to string for consistency
            saved_path = Path(output_path).resolve()
            print(f"Image saved to {saved_path}")
            saved_paths.append(str(saved_path))

    return saved_paths
