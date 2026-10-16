import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force offline / missing models so tests never trigger downloads.
os.environ.setdefault("YOLO_MODEL_PATH", "/does/not/exist/yolo11n.pt")
os.environ.setdefault("MP_MODEL_PATH", "/does/not/exist/face_landmarker.task")
os.environ.setdefault("TS_MODEL_PATH", str(ROOT / "models" / "does_not_exist.keras"))
os.environ.setdefault("DRIVER_MODELS_DIR", str(ROOT / "models" / "does_not_exist_dir"))
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")
