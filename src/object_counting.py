import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("samples/highway.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# Define region points
region_points = [(960, 850), (1920, 850)]

# For rectangle region counting: top left, top right, bottom right, bottom left
# region_points = [(20, 400), (1080, 400), (1080, 850), (20, 850)]

# Video writer
video_writer = cv2.VideoWriter(
    "./run/highway_counted.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=False,  # Display the output
    region=region_points,  # Pass region points
    model="yolo11x.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break
    im0 = counter.count(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
