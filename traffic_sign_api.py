#!/usr/bin/env python
"""Traffic Sign Classification API (VGG16-transfer on GTSRB)."""
import base64
import io
import json
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

from validation_utils import read_upload_bytes, decode_b64_image, cors_origins

MODEL_PATH = os.getenv("TS_MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gtsrb_model.keras"))
model = None
MODEL_ERROR = None
IMG = 32
NUM_CLASSES = 43

LABELS_PATH = os.path.join(os.path.dirname(MODEL_PATH), "gtsrb_labels.json")
CLASS_NAMES = None
try:
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as fh:
            CLASS_NAMES = json.load(fh).get("classes")
except Exception:
    CLASS_NAMES = None


def load():
    global model, MODEL_ERROR
    if not os.path.exists(MODEL_PATH):
        model = None
        MODEL_ERROR = f"Model not found at {MODEL_PATH}. Run train_traffic_sign.py first."
        return
    try:
        model = keras.models.load_model(MODEL_PATH)
        MODEL_ERROR = None
    except Exception:
        model = None
        MODEL_ERROR = "Could not load traffic sign model."


load()

app = FastAPI(title="Traffic Sign Classification API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(img):
    img = img.convert("RGB").resize((IMG, IMG))
    arr = np.array(img).astype("float32")
    return vgg_preprocess(arr)[None, ...]


def classify(img):
    if model is None:
        raise HTTPException(status_code=503, detail=MODEL_ERROR or "Model unavailable.")
    logits = model.predict(preprocess(img), verbose=0)[0]
    cls_id = int(np.argmax(logits))
    top5_idx = np.argsort(logits)[::-1][:5]
    return {
        "class_id": cls_id,
        "confidence": float(logits[cls_id]),
        "probabilities": [float(p) for p in logits],
        "top5": [{"class_id": int(i), "confidence": float(logits[i])} for i in top5_idx],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None,
        "classes": NUM_CLASSES,
        "model_path": MODEL_PATH,
        "reason": MODEL_ERROR,
    }


@app.post("/classify")
async def classify_file(file: UploadFile = File(...)):
    raw = await read_upload_bytes(file)
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return classify(img)


@app.post("/classify_base64")
@app.post("/classify_base64")
async def classify_b64(payload: dict):
    img = decode_b64_image((payload or {}).get("image", ""))
    return classify(img)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
