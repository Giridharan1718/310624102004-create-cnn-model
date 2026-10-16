#!/usr/bin/env python
"""Driver Drowsiness Detection API using MediaPipe Face Landmarker (EAR+MAR).

Experimental: each POST analyzes a single frame. Sustained drowsiness ideally
requires temporal logic across a video stream, which is not implemented here.
"""
import base64
import io
import os

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from validation_utils import read_upload_bytes, decode_b64_image, cors_origins

MP_MODEL_PATH = os.getenv(
    "MP_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpmodels", "face_landmarker.task"),
)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "8388608"))
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", "0.25"))
MAR_THRESHOLD = float(os.getenv("MAR_THRESHOLD", "0.6"))

mp = None
FACE = None
MODEL_ERROR = None
try:
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as taskspython  # noqa: F401
    from mediapipe.tasks.python import vision as visions  # type: ignore

    if not os.path.exists(MP_MODEL_PATH):
        raise FileNotFoundError(f"face_landmarker.task not found at {MP_MODEL_PATH}")
    options = visions.FaceLandmarkerOptions(
        base_options=taskspython.BaseOptions(model_asset_path=MP_MODEL_PATH),
        output_face_blendshapes=False,
        num_faces=1,
    )
    FACE = visions.FaceLandmarker.create_from_options(options)
except Exception as exc:  # do not expose stack trace; keep model unavailable
    FACE = None
    MODEL_ERROR = f"MediaPipe model unavailable ({exc})" if exc else "MediaPipe model unavailable."

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
M_UPPER = 13
M_LOWER = 14
M_LEFT = 61
M_RIGHT = 291


def _EAR(indices, lm):
    p = [(lm[i].x, lm[i].y) for i in indices]
    A = float(np.linalg.norm(np.array(p[1]) - np.array(p[5])))
    B = float(np.linalg.norm(np.array(p[2]) - np.array(p[4])))
    C = float(np.linalg.norm(np.array(p[0]) - np.array(p[3])))
    return (A + B) / (2.0 * C + 1e-9)


def _MAR(lm):
    pp = lambda i: np.array((lm[i].x, lm[i].y))
    h = float(np.linalg.norm(pp(M_LEFT) - pp(M_RIGHT)))
    v = float(np.linalg.norm(pp(M_UPPER) - pp(M_LOWER)))
    return v / (h + 1e-9)


def analyze_image(img: Image.Image):
    if FACE is None:
        return {"face_detected": False, "state": "unavailable", "reason": MODEL_ERROR or "model not loaded"}
    rgb = np.array(img.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = FACE.detect(mp_image)
    except Exception:
        return {"face_detected": False, "state": "error", "reason": "detection failure"}
    faces = getattr(result, "face_landmarks", None) or []
    if not faces:
        return {"face_detected": False, "state": "unknown", "reason": "no face detected"}
    lm = faces[0]
    ear = (_EAR(LEFT_EYE, lm) + _EAR(RIGHT_EYE, lm)) / 2.0
    mar = _MAR(lm)
    eyes_closed = bool(ear < EAR_THRESHOLD)
    yawning = bool(mar > MAR_THRESHOLD)
    drowsy = eyes_closed or yawning
    return {
        "face_detected": True,
        "ear": round(float(ear), 4),
        "mar": round(float(mar), 4),
        "eyes_closed": eyes_closed,
        "yawning": yawning,
        "state": "DROWSY" if drowsy else "ALERT",
        "note": "Single-frame heuristic; sustained drowsiness requires temporal analysis (not implemented).",
    }


app = FastAPI(title="Driver Drowsiness Detection API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    if FACE is None:
        return {"status": "unavailable", "reason": MODEL_ERROR}
    return {"status": "healthy", "mediapipe": "landmarker+float16", "model_path": MP_MODEL_PATH}


@app.post("/detect_drowsiness")
async def detect_endpoint(file: UploadFile = File(...)):
    raw = await read_upload_bytes(file, MAX_UPLOAD_BYTES)
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return analyze_image(img)


@app.post("/detect_drowsiness_base64")
async def detect_b64(payload: dict):
    img = decode_b64_image((payload or {}).get("image", ""))
    return analyze_image(img)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
