#!/usr/bin/env python
"""Traffic Light detection API (YOLO11 + HSV color/state heuristic)."""
import base64
import io
import os

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from PIL import ImageChops

from validation_utils import read_upload_bytes, decode_b64_image, parse_conf, cors_origins

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "8388608"))
TL_CLASS_ID = 9  # COCO class "traffic light"

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
        MODEL_ERROR = "YOLO model could not be loaded; check YOLO_MODEL_PATH and network access."


load()

app = FastAPI(title="Traffic Light API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _classify_color_hsv(img_pil: Image.Image):
    """Region-based HSV analysis returning red/yellow/green/unknown.

    Operates on the cropped light region and returns unknown when no single
    color clearly dominates.
    """
    arr = np.asarray(img_pil.convert("RGB"))
    if arr.size == 0:
        return "unknown", 0.0
    # Convert RGB -> HSV using OpenCV style
    # Pillow-based quick approximation via numpy
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    v = mx
    denom = mx - mn
    # hue
    h = np.zeros_like(mx)
    mask = denom != 0
    rmask = (mx == r) & mask
    gmask = (mx == g) & mask
    bmask = (mx == b) & mask
    h[rmask] = (60 * ((g - b) / denom))[rmask] % 360
    h[gmask] = (60 * ((b - r) / denom + 2))[gmask]
    h[bmask] = (60 * ((r - g) / denom + 4))[bmask]
    # use pixels that are saturated enough to be meaningful
    sat = (denom / (mx + 1e-6))
    bright = v > 40
    saturated = sat > 0.25
    valid = bright & saturated

    total = int(valid.sum())
    if total == 0:
        return "unknown", 0.0
    red = int((( (h < 15) | (h > 345)) & valid).sum())
    yellow = int(((h >= 15) & (h < 50) & valid).sum())
    green = int(((h >= 50) & (h < 160) & valid).sum())
    total_color = red + yellow + green
    if total_color == 0:
        return "unknown", 0.0
    shares = {"red": red, "yellow": yellow, "green": green}
    state = max(shares, key=lambda k: shares[k])
    share = shares[state] / total_color
    if share < 0.5:
        return "unknown", float(share)
    return state, float(share)


def _run(img, conf):
    if model is None:
        raise HTTPException(status_code=503, detail=MODEL_ERROR or "Model unavailable.")
    try:
        results = model.predict(img, conf=conf, classes=[TL_CLASS_ID], verbose=False)
    except Exception:
        raise HTTPException(status_code=500, detail="Detection failed.")
    out = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = [int(x) for x in b.xyxy[0].tolist()]
            crop = img.crop((x1, y1, x2, y2))
            state, share = _classify_color_hsv(crop)
            out.append(
                {
                    "state": state,
                    "confidence": float(b.conf.item()),
                    "color_share": float(share),
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            )
    return {"count": len(out), "lights": out}


@app.get("/health")
def health():
    if model is None:
        return {"status": "unavailable", "reason": "model not loaded"}
    return {"status": "healthy", "traffic_light_class_id": TL_CLASS_ID, "model_path": YOLO_MODEL_PATH}


@app.post("/detect_traffic_light")
async def detect(file: UploadFile = File(...), conf: float = Query(0.3)):
    raw = await read_upload_bytes(file, MAX_UPLOAD_BYTES)
    c = parse_conf(conf)
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return _run(img, c)


@app.post("/detect_traffic_light_base64")
async def detect_tl_b64(payload: dict):
    img = decode_b64_image((payload or {}).get("image", ""))
    c = parse_conf((payload or {}).get("conf", 0.3))
    return _run(img, c)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
