from ultralytics import YOLO
from src.config import *

def train_model():
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(RESULTS_DIR),
        name="train",
        seed=42,
        deterministic=True,
        exist_ok=True
    )

if __name__ == "__main__":
    train_model()
