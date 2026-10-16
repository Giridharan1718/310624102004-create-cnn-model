#!/usr/bin/env python
"""Object Detection API using Ultralytics YOLO11."""
import base64
import io
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from validation_utils import read_upload_bytes, decode_b64_image, parse_conf, cors_origins

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "8388608"))

model = None
MODEL_ERROR = None


def load():
    global model, MODEL_ERROR
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL_PATH)
        MODEL_ERROR = None
    except Exception:
        model = None
        MODEL_ERROR = "YOLO model could not be loaded; check YOLO_MODEL_PATH (and network) then set env var."


load()

app = FastAPI(title="Object Detection API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run(img, conf):
    if model is None:
        raise HTTPException(status_code=503, detail=MODEL_ERROR or "Model unavailable.")
    try:
        results = model.predict(img, conf=conf, verbose=False)
    except Exception:
        raise HTTPException(status_code=500, detail="Detection failed.")
    out = []
    for r in results:
        for b in r.boxes:
            cls_id = int(b.cls.item())
            out.append(
                {
                    "class_id": cls_id,
                    "class_name": model.names[cls_id],
                    "confidence": float(b.conf.item()),
                    "bbox_xyxy": [float(x) for x in b.xyxy[0].tolist()],
                }
            )
    return {"n": len(out), "detections": out}


@app.get("/health")
def health():
    if model is None:
        return {"status": "unavailable", "reason": "model not loaded"}
    return {"status": "healthy", "classes": list(model.names.values()), "model_path": YOLO_MODEL_PATH}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Query(0.35)):
    raw = await read_upload_bytes(file, MAX_UPLOAD_BYTES)
    c = parse_conf(conf)
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return _run(img, c)


@app.post("/detect/base64")
@app.post("/detect_base64")
async def detect_b64(payload: dict):
    img = decode_b64_image((payload or {}).get("image", ""))
    c = parse_conf((payload or {}).get("conf", 0.35))
    return _run(img, c)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
