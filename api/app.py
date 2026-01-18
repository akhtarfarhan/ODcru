from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
from ultralytics import YOLO

import numpy as np
import cv2
from pathlib import Path

# App configuration
app = FastAPI(
    title="ODCRU Car Detection API",
    description="YOLOv8-based car detection service",
    version="1.0.0"
)

# Load YOLO model ONCE
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "results/train/weights/best.pt"

model = YOLO(MODEL_PATH)

CONF_THRESHOLD = 0.4   

# Root endpoint
@app.get("/")
def root():
    return {
        "API is running and good to go!"
    }

# Health check
# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "model_loaded": True
#     }

# JSON prediction endpoint
# @app.post("/predict")
# async def predict_json(file: UploadFile = File(...)):
#     contents = await file.read()
#     np_img = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#     if img is None:
#         return JSONResponse(
#             status_code=400,
#             content={"error": "Invalid image file"}
#         )

#     results = model(img, conf=CONF_THRESHOLD)

#     detections = []

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             detections.append({
#                 "class_id": int(box.cls),
#                 "confidence": float(box.conf),
#                 "bbox": [x1, y1, x2, y2]
#             })

#     return {
#         "num_detections": len(detections),
#         "detections": detections
#     }

# Annotated image endpoint
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file"}
        )

    results = model(img, conf=CONF_THRESHOLD)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"car {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    success, encoded_img = cv2.imencode(".png", img)
    if not success:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to encode image"}
        )

    return Response(
        content=encoded_img.tobytes(),
        media_type="image/png"
    )