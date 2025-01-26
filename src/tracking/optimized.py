import argparse
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from src.utils import logger


def load_config():
    """Load and return configuration settings"""
    return {
        "model_path": "yolo11x.pt",
        "track_history_length": 120,
        "batch_size": 64,
        "line_thickness": 4,
        "track_color": (230, 230, 230),
    }


def initialize_video(video_path):
    """Initialize video capture and writer objects"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_name = video_path.split("/")[-1]
    output_path = f"run/{video_name.split('.')[0]}_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    return cap, out, output_path, fps


def update_track_history(
    track_history,
    last_seen,
    track_ids,
    frame_count,
    batch_size,
    frame_idx,
    history_length,
):
    """Update tracking history and remove old tracks"""
    current_tracks = set(track_ids)
    for track_id in list(track_history.keys()):
        if track_id in current_tracks:
            last_seen[track_id] = frame_count - (batch_size - frame_idx - 1)
        elif frame_count - last_seen[track_id] > history_length:
            del track_history[track_id]
            del last_seen[track_id]


def draw_tracks(frame, boxes, track_ids, track_history, config):
    """Draw tracking lines on frame"""
    if not track_ids:
        return frame

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > config["track_history_length"]:
            track.pop(0)

        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            frame,
            [points],
            isClosed=False,
            color=config["track_color"],
            thickness=config["line_thickness"],
        )
    return frame


def process_batch(model, batch_frames, track_history, last_seen, frame_count, config):
    """Process a batch of frames through YOLO model"""
    results = model.track(
        batch_frames,
        persist=True,
        tracker="botsort.yaml",
        show=False,
        verbose=False,
        iou=0.5,
    )

    processed_frames = []
    for frame_idx, result in enumerate(results):
        boxes = result.boxes.xywh.cpu()
        track_ids = (
            result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []
        )

        update_track_history(
            track_history,
            last_seen,
            track_ids,
            frame_count,
            len(batch_frames),
            frame_idx,
            config["track_history_length"],
        )

        annotated_frame = result.plot(font_size=4, line_width=2)
        annotated_frame = draw_tracks(
            annotated_frame, boxes, track_ids, track_history, config
        )
        processed_frames.append(annotated_frame)

    return processed_frames


def main(video_path):
    """Main function to process video"""
    CONFIG = load_config()
    model = YOLO(CONFIG.get("model_path", "yolo11x.pt"))

    cap, out, output_path, fps = initialize_video(video_path)
    track_history = defaultdict(lambda: [])
    last_seen = defaultdict(int)

    # Calculate frames for 3 seconds
    frames_to_process = int(fps * 3)

    with tqdm(
        total=frames_to_process,
        desc="Processing frames",
        colour="green",
    ) as pbar:
        frame_count = 0
        batch_frames = []

        while cap.isOpened() and frame_count < frames_to_process:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            batch_frames.append(frame)

            if (
                len(batch_frames) == CONFIG["batch_size"]
                or frame_count == frames_to_process
            ):
                try:
                    processed_frames = process_batch(
                        model,
                        batch_frames,
                        track_history,
                        last_seen,
                        frame_count,
                        CONFIG,
                    )
                    for frame in processed_frames:
                        out.write(frame)
                        pbar.update(1)
                    batch_frames = []

                except Exception as e:
                    logger.error(
                        f"Error processing frames {frame_count - len(batch_frames) + 1} to {frame_count}: {str(e)}"
                    )
                    batch_frames = []
                    continue

    try:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"\nVideo has been saved to {output_path}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, default="samples/vietnam-2.mp4")
    args = parser.parse_args()

    main(args.video_path)
