import cv2
from ultralytics import solutions

from src.utils import logger

video_path = "samples/highway-2.mp4"

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

logger.info(f"Processing video with {fps} fps, {w}x{h} resolution")

# Video writer
file_name = f"./run/{video_path.split('/')[-1].split('.')[0]}_speed_estimated.mp4"
video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Define speed region points
speed_region = [
    (int(w * 0.2), int(h * 0.3)),  # Top left
    (int(w * 0.9), int(h * 0.3)),  # Top right
    (int(w * 0.9), h),  # Bottom right
    (int(w * 0.2), h),  # Bottom left
]

speed = solutions.SpeedEstimator(
    show=True,
    model="yolo11n.pt",
    region=speed_region,
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        logger.info(
            "Video frame is empty or video processing has been successfully completed."
        )
        break
    out = speed.estimate_speed(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

logger.info(f"Speed estimation video saved to {file_name}")
