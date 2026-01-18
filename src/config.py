from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_YAML = DATA_DIR / "data.yaml"

MODEL_NAME = "yolov8n.pt"

IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 10
CONF_THRESHOLD = 0.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = BASE_DIR / "results"
