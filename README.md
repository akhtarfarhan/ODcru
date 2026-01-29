# YOLO Object Detection Project

A comprehensive object detection project using YOLOv8 and YOLOv11 models with training, inference, and API capabilities.

## Project Overview

This project implements car detection using YOLO models with the following features:
- Model training and evaluation
- Real-time inference on images and videos
- REST API for predictions
- Data loading and preprocessing
- Visualization tools
- Performance evaluation metrics

## Project Structure

```
.
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── yolo11n.pt             # YOLOv11 nano pre-trained weights
├── yolov8n.pt             # YOLOv8 nano pre-trained weights
│
├── api/
│   └── app.py             # FastAPI application for inference
│
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── data_loader.py     # Data loading utilities
│   ├── evaluate.py        # Model evaluation
│   ├── inference.py       # Inference functions
│   ├── model.py           # Model initialization and management
│   ├── train.py           # Training script
│   ├── utils.py           # Utility functions
│   └── visualize.py       # Visualization tools
│
├── data/
│   ├── data.yaml          # Dataset configuration
│   ├── train/
│   │   ├── images/        # Training images
│   │   └── labels/        # Training annotations (YOLO format)
│   └── test/
│       ├── images/        # Test images
│       └── labels/        # Test annotations (YOLO format)
│
├── results/
│   └── train/
│       ├── args.yaml      # Training arguments
│       ├── results.csv    # Training metrics
│       └── weights/       # Saved model weights
│
└── runs/
    └── detect/
        ├── predict/       # Inference results
        ├── val/           # Validation results
        └── ...
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone or download the project:
```bash
cd ODCUR
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- ultralytics (YOLO models)
- opencv-python (image processing)
- torch/torchvision (deep learning framework)
- fastapi (API framework)
- uvicorn (ASGI server)
- numpy, pandas, pyyaml (utilities)

## Usage

### Training

Train the YOLO model on your dataset:

```bash
python main.py --train --model yolo11n --epochs 100 --batch-size 16
```

Or use the training module directly:

```python
from src.train import train_model
train_model(model_name='yolo11n', epochs=100, batch_size=16)
```

### Inference

Run inference on images or videos:

```bash
python main.py --predict --source image.jpg --model yolo11n
```

Or programmatically:

```python
from src.inference import run_inference
results = run_inference('image.jpg', model_name='yolo11n')
```

### Evaluation

Evaluate model performance on test set:

```bash
python main.py --evaluate --model yolo11n
```

### API Server

Start the FastAPI server for real-time predictions:

```bash
python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `POST /predict` - Send image for detection
  - Request: multipart/form-data with `file` (image)
  - Response: JSON with detected objects and bounding boxes

- `GET /health` - Health check endpoint

- `GET /docs` - Interactive API documentation (Swagger UI)

## Data Format

The project uses YOLO format for annotations. Each image has a corresponding `.txt` file with annotations:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized (0-1) relative to image dimensions.

### Dataset Configuration

Edit `data/data.yaml` to configure your dataset:

```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images
nc: 1  # number of classes
names: ['car']  # class names
```

## Models

### Supported Models

- **YOLOv8 (nano)**: `yolov8n.pt` - Lightweight, fast inference
- **YOLOv11 (nano)**: `yolo11n.pt` - Latest YOLO version, improved accuracy

## Key Modules

### config.py
Central configuration for model, training, and inference parameters.

### model.py
Handles model loading, initialization, and management.

### train.py
Training loop with logging, validation, and checkpointing.

### evaluate.py
Evaluation metrics: precision, recall, mAP, confusion matrix.

### inference.py
Batch and single-image inference with confidence filtering.

### data_loader.py
Data loading with augmentation and preprocessing.

### utils.py
Helper functions for path management, visualization, etc.

### visualize.py
Visualization of predictions, bounding boxes, and metrics.

## Results

Training results are saved in `results/train/`:
- `results.csv` - Epoch-wise metrics
- `weights/` - Model checkpoints
- `args.yaml` - Training configuration

Prediction results are saved in `runs/detect/`:
- `predict/` - Latest inference results
- `val/` - Validation results

## Performance

Expected performance metrics on test set:
- Precision, Recall, mAP50, mAP50-95 (calculated during evaluation)
- See `results/train/results.csv` for training history

## Requirements

See `requirements.txt` for complete list of dependencies:

```
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
fastapi>=0.100.0
uvicorn>=0.23.0
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
```

## Configuration

Edit `src/config.py` to customize:
- Model parameters
- Training hyperparameters
- Inference settings
- Paths and directories

## Troubleshooting

### CUDA/GPU Issues
If you don't have GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Missing Dependencies
Reinstall all requirements:
```bash
pip install -r requirements.txt --upgrade
```

### Model Not Found
Ensure model weights are in the project root:
- `yolo11n.pt`
- `yolov8n.pt`

## Author

Farhan Akhtar

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [YOLO Format Guide](https://docs.ultralytics.com/datasets/detect/)
