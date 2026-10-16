"""Shared input-validation helpers for all FastAPI services."""
import base64
import io
import os
from typing import Optional

from fastapi import HTTPException, UploadFile
from PIL import Image

DEFAULT_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "8388608"))  # 8 MB
ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "4096"))


async def read_upload_bytes(upload: UploadFile, max_bytes: Optional[int] = None) -> bytes:
    limit = max_bytes or DEFAULT_MAX_UPLOAD_BYTES
    name = getattr(upload, "filename", None) or "<upload>"
    if upload.content_type and upload.content_type.lower() not in ALLOWED_IMAGE_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported content type '{upload.content_type}' for file {name}.")
    chunks = []
    total = 0
    while True:
        chunk = await upload.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"Upload for {name} exceeds {limit} bytes.")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty upload for {name}.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.load()
            if max(probe.size) > MAX_IMAGE_DIMENSION:
                raise HTTPException(status_code=413, detail=f"Image dimensions exceed {MAX_IMAGE_DIMENSION}px.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail=f"Could not decode image for {name}.")
    return data


def decode_b64_image(payload: str, max_bytes: Optional[int] = None) -> Image.Image:
    limit = max_bytes or DEFAULT_MAX_UPLOAD_BYTES
    if not isinstance(payload, str) or not payload.strip():
        raise HTTPException(status_code=400, detail="Missing or invalid 'image' base64 payload.")
    # Accept data: URL form.
    raw_str = payload.split(",", 1)[-1] if "," in payload else payload
    try:
        data = base64.b64decode(raw_str, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty decoded image payload.")
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Decoded image exceeds {limit} bytes.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.load()
            if max(probe.size) > MAX_IMAGE_DIMENSION:
                raise HTTPException(status_code=413, detail=f"Image dimensions exceed {MAX_IMAGE_DIMENSION}px.")
            return probe.copy().convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode base64 image.")


def parse_conf(value, default: float = 0.35) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid confidence value.")
    if not (0.0 <= conf <= 1.0):
        raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0.")
    return conf


def cors_origins() -> list:
    env = os.getenv("CORS_ORIGINS")
    if env:
        return [o.strip() for o in env.split(",") if o.strip()]
    return ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]
