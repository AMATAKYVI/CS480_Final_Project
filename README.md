# Object Detection Model Comparison

This project compares two state-of-the-art object detection models: **YOLOv8** and **DETR (Detection Transformer)**.

## Project Structure

```
final_project/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── models/                 # Model loading functions
│   ├── yolov8.py          # YOLOv8 model loader
│   └── detr.py            # DETR model loader
├── evalulation/           # Evaluation utilities
│   ├── benchmark.py       # Inference benchmarking
│   ├── measure_fps.py     # FPS measurement
│   └── compare_params.py  # Parameter counting
├── visualize/             # Visualization utilities
│   └── draw_boxes.py      # Bounding box drawing
└── datasets/              # Dataset storage
    └── coco128/           # COCO128 dataset
```

## Installation

1. Clone or download the project
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main comparison script:

```bash
python main.py
```

This will:

1. Load both YOLOv8 and DETR models
2. Run inference on a sample image
3. Count model parameters
4. Measure inference speed (FPS)

## Models

### YOLOv8 (Nano)

- **Speed**: Very fast inference
- **Parameters**: ~3.3M
- **Framework**: Ultralytics
- **Strengths**: Real-time detection, small model size

### DETR (Detection Transformer)

- **Speed**: Slower but accurate
- **Parameters**: ~41M (ResNet50 backbone)
- **Framework**: PyTorch torchvision
- **Strengths**: Transformer-based, no post-processing needed

## Features

- **Inference Benchmarking**: Compare detection outputs
- **FPS Measurement**: Evaluate model speed
- **Parameter Counting**: Compare model complexity
- **Visualization**: Draw bounding boxes on detections

## Dataset

The project uses the COCO128 dataset (128 images from COCO) for evaluation.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- ultralytics (YOLOv8)
- OpenCV
- Pillow
- NumPy

## Notes

- Models are automatically downloaded on first use
- Requires sufficient GPU memory for best performance
- DETR uses normalized coordinates (0-1), YOLOv8 uses pixel coordinates

## Author

Final Project for CS480 - George Mason University
