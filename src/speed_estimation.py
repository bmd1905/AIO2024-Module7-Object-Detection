import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("samples/fruit_and_vegetable.gif")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# Video writer
video_writer = cv2.VideoWriter(
    "speed_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Define speed region points
speed_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

speed = solutions.SpeedEstimator(
    show=True,  # Display the output
    model="yolo11n.pt",  # Path to the YOLO11 model file.
    region=speed_region,  # Pass region points
    # classes=[0, 2],  # If you want to estimate speed of specific classes.
    # line_width=2,  # Adjust the line width for bounding boxes and text display
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break
    out = speed.estimate_speed(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
