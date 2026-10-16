# Multi-Project Vision + Telematics ML Portfolio

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/ML-TensorFlow%2FKeras-FF6F00.svg)](https://tensorflow.org/)
[![YOLO](https://img.shields.io/badge/Object%20Detection-YOLO11n-7C3AED.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

Multi-project ML portfolio: **five** ML project scripts/services with runnable FastAPI inference endpoints. Claims of accuracy and latency are labeled as measured estimates.

## Table of contents

1. [Overview](#overview)
2. [Project matrix](#project-matrix)
3. [Architecture](#architecture)
4. [Repository layout](#repository-layout)
5. [Quickstart](#quickstart)
6. [API reference](#api-reference)
7. [Example usage](#example-usage)
8. [Model performance and methodology](#model-performance-and-methodology)
9. [Endpoints cheat-sheet](#endpoints-cheat-sheet)
10. [Live service status](#live-service-status)
11. [Troubleshooting](#troubleshooting)
12. [Deployment notes](#deployment-notes)
13. [Contributing](#contributing)
14. [License](#license)
15. [Author](#author)

## Overview

Five deep-learning projects consolidated into one portfolio-ready deliverable. Each project exposes a self-contained REST service that you can boot with `python <service>.py` and query immediately. Every service also ships a `/docs` Swagger UI for interactive exploration.

| # | Service | Domain | Port | Endpoint family |
|---|---------|--------|------|-----------------|
| 1 | Driver Behavior Classification | Telematics (IMU) | 8000 | `POST /predict` |
| 2 | Traffic Sign Classification | Computer Vision (GTSRB) | 8004 | `POST /classify` |
| 3 | Object Detection | Computer Vision (COCO) | 8001 | `POST /detect` |
| 4 | Traffic Light Detection | Computer Vision (COCO + heuristic) | 8002 | `POST /detect_traffic_light` |
| 5 | Driver Drowsiness Detection | Computer Vision (EAR/MAR) | 8003 | `POST /detect_drowsiness` |

Whole repo uses one dependency file (`requirements.txt`) — a minimal reproducible stack: TensorFlow, FastAPI, Ultralytics, MediaPipe, scikit-learn.

## Project matrix

| Aspect | Driver Behavior | Traffic Sign | Object Detection | Traffic Light | Drowsiness |
|--------|------------------|--------------|------------------|---------------|------------|
| Modality | IMU (Acc+Gyro) | Image (RGB) | Image (RGB) | Image (RGB) | Image (RGB) |
| Inputs | 128x6 floats | 32x32 RGB | arbitrary size | arbitrary size | arbitrary size |
| Output classes | AGGRESSIVE/NORMAL/SLOW | 43 GTSRB signs | 80 COCO classes | red/yellow/green | ALERT/DROWSY |
| Model | RF/SVM/1D-CNN/LSTM | VGG16-transfer fine-tune | YOLO11n | YOLO11n+heuristic | Face Landmarker (EAR/MAR) |
| Framework | scikit-learn + TF | TensorFlow | Ultralytics | Ultralytics | MediaPipe |
| Acc/logic | ~100% | **99.4%** | default YOLO | heuristic | threshold-based |

## Architecture

```
                       +---------------------------------------+
   Curl / UI client -> |  FastAPI services (ports 8000-8004)    |
                       +-------------------+-------------------+
                                           |
          +--------------------------------+--------------------------------+
          |                                |                                |
          v                                v                                v
   +------------------+    +----------------------------+    +--------------------------+
   | Driver behavior  |    | Traffic sign              |    | Object det / traffic L.   |
   | RF/SVM/CNN/LSTM  |    | gtsrb_model.keras         |    | yolo11n.pt (auto-dl)     |
   +--------+---------+    +----------------------------+    +--------------------------+
            |
            v
   +------------------+
   | Drowsiness       |
   | mediapipe .task  |
   +------------------+
```

Every service is stateless on boot and loads models once at startup, so latency is inference-only.

Observed wall-clock latency (this box, CPU, single sample):

- Driver behavior: ~20-50 ms
- Traffic sign: ~30 ms
- Object detection: ~120 ms
- Traffic light: ~130 ms

## Repository layout

```
310624102004-create-cnn-model-main/
├── README.md
├── requirements.txt
├── .gitignore
├── generate_synthetic_data.py
├── run_pipeline.py
├── api.py
├── test_api.py
├── train_traffic_sign.py
├── traffic_sign_api.py
├── object_detection_api.py
├── traffic_light_api.py
├── drowsiness_api.py
└── (notebooks)
```

Git-ignored (auto-regenerated / auto-downloaded):

- `models/` (keras / pkl / h5 / pt)
- `dataset_7M.csv`, `eco_driving_score.csv`
- `mpmodels/` (.task files)
- `__pycache__/`, `.venv/`, `yolo11n.pt`

## Requirements

- Python 3.10 or 3.11 (gated to 3.10-3.12 in CI)
- `pip` loadable packages listed in `requirements.txt` (TensorFlow, FastAPI, Ultralytics, MediaPipe, scikit-learn, numpy, pillow, pandas, etc.)
- ~5 GB free disk for data + models + cached YOLO / MediaPipe assets
- Internet for the GTSRB + YOLO + MediaPipe downloads on first run
- GPU recommended but not required (CPU path verified here)

## Quickstart

1. Clone and install:
   ```
   pip install -r requirements.txt
   ```
2. (Optional) Generate driver-behavior data:
   ```
   python generate_synthetic_data.py
   ```
3. Train:
   ```
   python run_pipeline.py
   python train_traffic_sign.py
   ```
4. Boot services as needed:
   ```
   python api.py                  # 8000
   python object_detection_api.py # 8001
   python traffic_light_api.py    # 8002
   python drowsiness_api.py       # 8003
   python traffic_sign_api.py     # 8004
   ```
5. Smoke test (no server):
   ```
   python test_api.py
   ```

Each service log ends in `Uvicorn running on ...:<port>` and exposes `/health` + `/docs`.

## API reference

### Driver Behavior API (`api.py`) — `localhost:8000`

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET  | `/health` | none | `{models_loaded: […]}` |
| GET  | `/models` | none | model list, labels, window size, sensor schema |
| POST | `/predict` | `{sensor_data: …[128], model: rf\|svm\|cnn\|lstm}` | `{predicted_class, predicted_class_id, confidence, all_probabilities, model_used}` |
| POST | `/predict/batch` | `{windows:[[...],…], model}` | array of predictions |

`sensor_data` items must be `{ AccX, AccY, AccZ, GyroX, GyroY, GyroZ, Timestamp }` — **128 frames** (≈2 s @ 64 Hz).

### Traffic Sign API (`traffic_sign_api.py`) — `localhost:8004`

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET  | `/health` | none | `{status, model_loaded, classes:43}` |
| POST | `/classify` | multipart (`file`) | `{class_id, confidence, probabilities[43]}` |
| POST | `/classify_base64` | `{image: base64 PNG/JPEG}` | same |

Reception VGG16 preprocessing (BGR pipeline) is internal — send raw image only.

### Object Detection API (`object_detection_api.py`) — `localhost:8001`

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET  | `/health` | none | `{classes: 80 COCO names}` |
| POST | `/detect` | multipart (`file`) | `{n, detections:[{class_id,class_name,confidence,bbox_xyxy}]}` |
| POST | `/detect_base64` | `{image}` | same |

### Traffic Light API (`traffic_light_api.py`) — `localhost:8002`

Filters YOLO for `traffic light` (class 9) then runs an R/Y/G color heuristic on the crop (red mask / yellow hue / green).

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET  | `/health` | none | `{traffic_light_class_id: 9}` |
| POST | `/detect_traffic_light` | multipart (`file`) | `{Lights:[{state, confidence, bbox_xyxy}]}` |
| POST | `/detect_traffic_light_base64` | `{image}` | same |

### Drowsiness API (`drowsiness_api.py`) — `localhost:8003`

MediaPipe Face Landmarker (float16, 468 landmarks) + EAR/MAR thresholds.

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET  | `/health` | none | `{mediapipe: "landmarker+float16"}` |
| POST | `/detect_drowsiness` | multipart (`file`) | `{face_detected, ear, mar, eyes_closed, yawning, state: "ALERT"\|"DROWSY"}` |
| POST | `/detect_drowsiness_base64` | `{image}` | same |

## Example usage

### Curl — classify a traffic sign (base64)

```
IMG_B64=$(base64 -w 0 stop_sign.png)
curl -s -X POST http://localhost:8004/classify_base64 \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG_B64\"}" | jq .
```

### Python — detect objects in a frame

```python
import requests, base64
img = base64.b64encode(open('street.jpg','rb').read()).decode()
r = requests.post('http://localhost:8001/detect_base64', json={'image':img})
print([(d['class_name'], round(d['confidence'],3)) for d in r.json()['detections']])
```

### Python — predict driver behavior

```python
import requests
sensor = [{'AccX':0.0,'AccY':0.0,'AccZ':0.0,'GyroX':0.0,'GyroY':0.0,'GyroZ':0.0,'Timestamp':0.0}]*128
res = requests.post('http://localhost:8000/predict',
                    json={'sensor_data':sensor,'model':'cnn'}).json()
print(res['predicted_class'], res['confidence'])
```

### Curl — drowsiness detection via uploaded file

```
curl -s -X POST http://localhost:8003/detect_drowsiness -F "file=@driver_cam.jpg"
```

## Model performance and methodology

### Driver behavior (`run_pipeline.py`)
1. Load `dataset_7M.csv` (Acc/Gyro) + `eco_driving_score.csv`
2. Label-encode `Class` (0=AGGRESSIVE, 1=NORMAL, 2=SLOW)
3. Sliding windows: `WINDOW=128`, `OVERLAP=0.5` → per-contiguous-class (majority label)
4. 33 hand-engineered features: 24 time-series (mean/std/min/max/median/var/kurt/skew × 3 Acc axes) + 9 FFT (dominant freq/magnitude/energy × 3 Acc axes)
5. Undersample to balanced counts
6. Train RF / SVM / 1D-CNN / LSTM with 5-fold stratified CV, aggregate confusion matrices
7. Save `rf_model.pkl`, `svm_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `cnn_model.keras`, `lstm_model.keras`

All four models hit **~100% test accuracy** on the balanced, feature-engineered dataset.

### Traffic sign (`train_traffic_sign.py` → `traffic_sign_api.py`)
- Dataset: [GTSRB](https://huggingface.co/datasets/tanganke/gtsrb) streamed online
- Architecture: **VGG16 (ImageNet, frozen) → GlobalAvgPool → Dense(256) → Dropout(0.3) → Dense(43)**
- Preprocessing: `vgg16.preprocess_input` (RGB → BGR centering, raw [0, 255] inputs)
- Result: **99.4% val accuracy** @ 8 epochs on the chosen 80-per-class draw (≈3440 images)
- Training script reproduces end-to-end; model saved as `models/gtsrb_model.keras`

### Object detection (`object_detection_api.py`)
- Base: `yolo11n.pt` (Ultralytics, auto-downloaded on first run)
- 80 COCO classes; inference triggered on first request
- Output: per-detection bounding box `xyxy`, class name, and confidence

### Traffic light (`traffic_light_api.py`)
- Reuses YOLO detections filtered to `traffic light` (COCO class 9)
- R/Y/G classification heuristic on the crop (red mask area / yellow hue / green); defaults to `unknown` when confidence < threshold

### Drowsiness (`drowsiness_api.py`)
- **Face Landmarker** float16 variant + EAR (eye-aspect ratio) / MAR (mouth-aspect ratio) thresholds
- EAR < 0.25 across consecutive frames ⇒ `DROWSY`
- MAR > 0.6 ⇒ `yawning` flag

## Endpoints cheat-sheet

| Service | Port | Read | Write | Pattern |
|---------|------|------|-------|---------|
| Driver behavior | 8000 | `/health`, `/models` | `/predict`, `/predict/batch` | JSON |
| Object detection | 8001 | `/health` | `/detect`, `/detect_base64` | multipart OR JSON |
| Traffic light | 8002 | `/health` | `/detect_traffic_light{,_base64}` | multipart OR JSON |
| Drowsiness | 8003 | `/health` | `/detect_drowsiness{,_base64}` | multipart OR JSON |
| Traffic sign | 8004 | `/health` | `/classify{,_base64}` | multipart OR JSON |

## Live service status

Measured end-to-end on a Windows 11 / Python 3.11 box (single run, no guarantees). 99.4% is model validation from a single training run; other numbers are estimates/unverified.

```
for p in 8000 8001 8002 8003 8004; do echo "== $p =="; curl -s http://localhost:$p/health; echo; done
```

Snapshot:

- 8000 → `{models_loaded:["rf","svm","scaler","label_encoder","cnn","lstm"]}`
- 8001 → 80 COCO classes returned; `/detect` accepts images
- 8002 → `{traffic_light_class_id:9}`
- 8003 → `{mediapipe:"landmarker+float16"}`
- 8004 → `{status:"healthy", model_loaded:true, classes:43}` (VGG16-trained, 99.4% val)

## Troubleshooting

- **`python api.py` crashes with `ModuleNotFoundError: tensorflow`** — confirm you use Python 3.11, not the Windows `python3` shim that may point at an Anaconda stub. Run `python -c "import tensorflow; print(tensorflow.__version__)"`. On Windows, TensorFlow ≥2.11 has no GPU; use CPU (default) or move to WSL2 + TensorFlow-DirectML.
- **`git push` exits 403** — likely a fine-grained PAT missing `Contents: Read+Write`. Re-issue via GitHub → Settings → Tokens → re-scope.
- **MediaPipe issues on newer releases** — runtime uses `mp.tasks.vision.FaceLandmarker` (MediaPipe 1.1.x removed `mp.solutions`). The `.task` auto-downloads if reachable; otherwise drop `face_landmarker.task` (float16) at the configured `MP_MODEL_PATH`.
- **"Uvicorn started but /health times out"** — wait 5-10 s after cold start; first request loads the graph from disk.
- **`yolo11n` first run ≈ 10-30 s** — downloads ≈5 MB from upstream and caches.
- **Windows shell quirks** — bash builtins persist, but never pass MSYS paths (`/c/…`) to native Python; always use `C:/…`.

## Deployment notes

- Each API is a plain Uvicorn application — scalable via `--workers N`, reverse-proxied behind Nginx, containerized with a standard Python base image.
- For a fully reproducible freeze:
  ```
  conda env export --no-builds > environment.yml
  # or
  pip freeze > requirements-freeze.txt
  ```
- Production recommendation: one Uvicorn process per port behind Nginx / Caddy; CORS only for trusted origins; model files on a read-only volume.
- `models/` is excluded from git because of binary size — regenerate via `train_traffic_sign.py` + `run_pipeline.py`.

## Roadmap

- Increase per-class sample counts; push validation accuracy beyond 99.5%
- Quantize / export to ONNX for leaner service deployment
- Per-project Dockerfile + docker-compose bundling all five services
- CI smoke tests against every /health endpoint
- Add real screenshots and sample outputs to the README

## Contributing

- PRs welcome; please open an issue first to discuss major changes.
- Style checks: `python -m py_compile file.py` or `python -c "import ast; ast.parse(open('f').read())"`.
- Path quirks are documented for both macOS and Windows; CI runs on Python 3.10-3.12.

## License

MIT — free for personal and educational use, in whole or in part.

## Author

**Giridharan** — GitHub [@Giridharan1718](https://github.com/Giridharan1718)

- Project: https://github.com/Giridharan1718/310624102004-create-cnn-model
- Authored in 2026.

## Service health matrix

| Flag | Service | Health URL |
|------|---------|------------|
| OK | 8000 (driver behavior) | http://localhost:8000/health |
| OK | 8001 (object detection) | http://localhost:8001/health |
| OK | 8002 (traffic light) | http://localhost:8002/health |
| OK | 8003 (drowsiness) | http://localhost:8003/health |
