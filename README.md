# AIO2024-Module7-Object-Detection

This is a repository for the AIO2024 Module 7 Object Detection part II. This is a part of the AIO2024 course.

![Theme](./assets/theme.png)

## Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx

# Use conda
conda create -n object-detection python=3.11 -y
conda activate object-detection
conda install ffmpeg -c conda-forge -y
pip install -r requirements.txt

# Use venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Object Tracking


### Online Tracking

Online tracking is the process of tracking objects in a video stream in real-time. In this case, we will use the batch size of 1, which means that we will process one frame at a time.

```bash
make tracking_simple
```

### Offline Tracking

Offline tracking is the process of tracking objects in a video file. In this case, we will increase the batch size for better performance. As a result, the processing time will be much faster.

```bash
make tracking_optimized
```

## Object Counting

Object counting with Ultralytics YOLO11 involves accurate identification and counting of specific objects in videos and camera streams. YOLO11 excels in real-time applications, providing efficient and precise object counting for various scenarios like crowd analysis and surveillance, thanks to its state-of-the-art algorithms and deep learning capabilities.

```bash
make object_counting
```

## YOLO World

The YOLO-World Model introduces an advanced, real-time Ultralytics YOLOv8-based approach for Open-Vocabulary Detection tasks. This innovation enables the detection of any object within an image based on descriptive texts. By significantly lowering computational demands while preserving competitive performance, YOLO-World emerges as a versatile tool for numerous vision-based applications.

Key Features:

- Real-time Solution: Harnessing the computational speed of CNNs, YOLO-World delivers a swift open-vocabulary detection solution, catering to industries in need of immediate results.

- Efficiency and Performance: YOLO-World slashes computational and resource requirements without sacrificing performance, offering a robust alternative to models like SAM but at a fraction of the computational cost, enabling real-time applications.

- Inference with Offline Vocabulary: YOLO-World introduces a "prompt-then-detect" strategy, employing an offline vocabulary to enhance efficiency further. This approach enables the use of custom prompts computed apriori, including captions or categories, to be encoded and stored as offline vocabulary embeddings, streamlining the detection process.

- Powered by YOLOv8: Built upon Ultralytics YOLOv8, YOLO-World leverages the latest advancements in real-time object detection to facilitate open-vocabulary detection with unparalleled accuracy and speed.

- Benchmark Excellence: YOLO-World outperforms existing open-vocabulary detectors, including MDETR and GLIP series, in terms of speed and efficiency on standard benchmarks, showcasing YOLOv8's superior capability on a single NVIDIA V100 GPU.

- Versatile Applications: YOLO-World's innovative approach unlocks new possibilities for a multitude of vision tasks, delivering speed improvements by orders of magnitude over existing methods.

```bash
make yolo-world
```


## References

- [Object Counting](https://docs.ultralytics.com/guides/object-counting/)
- [YOLO-World](https://docs.ultralytics.com/models/yolo-world/)
