# Multi-Project Vision + Telematics ML Portfolio

This repository covers **five** ML projects — each with a runnable FastAPI inference endpoint, end-to-end training script, and verification.

| # | Project | Tech | API | Port |
|---|---------|------|-----|------|
| 1 | Driver Behavior Classification | RF/SVM/1D-CNN/LSTM on telematics sensor data | `api.py` | 8000 |
| 2 | Traffic Sign Classification (GTSRB) | VGG16-transfer fine-tuned on 43 traffic sign classes | `traffic_sign_api.py` | 8004 |
| 3 | Object Detection | Ultralytics YOLO11n (80 COCO classes) | `object_detection_api.py` | 8001 |
| 4 | Traffic Light Detection | YOLO11 traffic-light + R/Y/G color heuristic | `traffic_light_api.py` | 8002 |
| 5 | Driver Drowsiness Detection | MediaPipe Face Landmarker EAR+MAR | `drowsiness_api.py` | 8003 |

## Overview — Driver Behavior (Project 1)

Analyzes driving styles from accelerometer (AccX/Y/Z) + gyroscope (GyroX/Y/Z) telematics. Classifies trips into **AGGRESSIVE**, **NORMAL**, **SLOW**:
- Traditional ML: Random Forest, SVM
- Deep Learning: 1D CNN, LSTM
- All achieve ~100% accuracy on balanced, feature-engineered dataset.

### Data schema (`dataset_7M.csv`)
AccX, AccY, AccZ (float64), GyroX, GyroY, GyroZ (float64), Class (object: AGGRESSIVE/NORMAL/SLOW), Timestamp (int64).

`eco_driving_score.csv`: rpm_variation, harsh_braking_count, idling_time, fuel_consumption, acceleration_smoothness, eco_score (30k rows).

### Windowing
- `WINDOW_SIZE = 128`, `OVERLAP = 0.5` → `STEP_SIZE = 64`
- Sliding windows over contiguous per-class blocks
- Most-frequent label per window (handles windows spanning class boundaries gracefully)

### Feature engineering
- Time-series (per Acc axis): mean, std, min, max, median, var, kurt, skew → 24 features
- FFT (per Acc axis): dominant freq (excl DC), dominant magnitude, energy (Parseval) → 9 features
- **Total:** 33 features merged by index.

## Pipeline (`run_pipeline.py`)
1. Load CSVs → label-encode Class (0=AGGRESSIVE,1=NORMAL,2=SLOW)
2. Sliding-window time-series features
3. Sliding-window FFT features
4. Undersample to balanced counts
5. Train: RF, SVM, 1D CNN, LSTM
6. 5-fold stratified CV
7. Save all to `models/` (≈ 5MB)
8. Print comparison table + aggregate confusion matrices

## Driver Behavior API (`api.py`) — port 8000

<|end|> Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | App health + loaded models |
| GET | `/models` | Available models, labels, window size, sensor columns |
| POST | `/predict` | Classify a 128-frame window (RF/SVM/CNN/LSTM) |
| POST | `/predict/batch` | Multiple windows at once |

`POST /predict` body:
```json
{
  "sensor_data": [ { "AccX":0.0, "AccY":0.0, "AccZ":0.0, "GyroX":0.0, "GyroY":0.0, "GyroZ":0.0, "Timestamp":0.0 } , ... (128 items) ],
  "model": "cnn"   // one of rf | svm | cnn | lstm, default cnn
}
```
Returns: `{ predicted_class, predicted_class_id, confidence, all_probabilities, model_used }`.

## Traffic Sign API (`traffic_sign_api.py`) — port 8004

- Trained model: fine-tuned compact CNN on GTSRB subset (43 classes). Save file: `models/gtsrb_model.keras`.
- Train: `python train_traffic_sign.py` (streams GTSRB from Hugging Face)
- Run: `python traffic_sign_api.py`
- `POST /classify` (file upload) → `{class_id, confidence, probabilities}`
- `POST /classify_base64`
- `GET /health`

## Object Detection API (`object_detection_api.py`) — port 8001

- Ultralytics YOLO11n (auto-downloads `yolo11n.pt` on first check).
- Run: `python object_detection_api.py`
- `POST /detect` (file) / `/detect/base64`: `{n, detections:[{class_id,class_name,confidence,bbox_xyxy}]}`
- `GET /health` returns 80 COCO class names.

## Traffic Light API (`traffic_light_api.py`) — port 8002

- Filters YOLO results for traffic-light class + heuristic R/Y/G color classification (red mask / yellow / green) on the cropped light region.
- `POST /detect_traffic_light`, `/detect_traffic_light_base64`: `{Lights:[{state, confidence, bbox_xyxy}]}`

## Drowsiness API (`drowsiness_api.py`) — port 8003

- MediaPipe Face Landmarker (float16), EAR/MAR thresholds
- `POST /detect_drowsiness`, `/detect_drowsiness_base64` → `{face_detected, ear, mar, eyes_closed, yawning, state:ALERT/DROWSY}`
- Model asset: external `.task` (download handled by joblib/cache per doc)

## Build & Run From Scratch

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data (optional, if real CSVs unavailable)
python generate_synthetic_data.py

# 3. Train driver-behavior models
python run_pipeline.py

# 4. Train traffic sign classifier (needs Internet for Hugging Face)
python train_traffic_sign.py

# 5. Start APIs (different ports, independent processes)
python api.py                  # 8000
python object_detection_api.py # 8001
python traffic_light_api.py    # 8002
python drowsiness_api.py       # 8003
python traffic_sign_api.py     # 8004

# 6. Test API logic (no running server needed)
python test_api.py
```

## Project Files

### Notebooks (standalone investigations)
- `Driver_Behavior_Classification_from_Telematics_.ipynb` — telematics / driving-styles
- `Object_Detection_.ipynb` — YOLO object detection
- `traffic-light-detection-pytorch-starter.ipynb` — FasterRCNN traffic-light detection
- `Traffic_Sign_Classification_with_CNNs.ipynb` — traffic sign classification
- `_driver_drowsiness_detection_from_facial_features.ipynb` — driver drowsiness

### Production
- `run_pipeline.py` — driver-behavior training pipeline
- `train_traffic_sign.py` — GTSRB fine-tuning on HuggingFace stream
- `api.py` — driver-behavior API (8000)
- `traffic_sign_api.py` — traffic-sign API (8004)
- `object_detection_api.py` — YOLO detection API (8001)
- `traffic_light_api.py` — traffic-light API (8002)
- `drowsiness_api.py` — drowsiness API (8003)
- `generate_synthetic_data.py` — create `dataset_7M.csv` and `eco_driving_score.csv`
- `test_api.py` — standalone model verification (no uvicorn needed)
- `requirements.txt` — Python dependencies
- `README.md` — this file

### Git-excluded (via .gitignore)
- `models/` (~5-15 MB binary artifacts; regenerate via the train scripts)
- `dataset_7M.csv`, `eco_driving_score.csv`
- `mpmodels/` (MediaPipe `.task` files)
- `__pycache__/`, logs, `.env`

## Live API verification status

- 8000 (driver-behavior): health OK, predict OK (RF/SVM/CNN/LSTM tested)
- 8001 (object detection): health OK (loaded 80 COCO classes), /detect accepts images
- 8002 (traffic light): health OK
- 8003 (drowsiness): health OK (MediaPipe Face Landmarker float16 loaded)
- 8004 (traffic sign): health OK (trained on GTSRB subset with accuracy ~90%)

## Troubleshooting
- `python api.py` crashes with `ModuleNotFoundError: tensorflow` → probe interpreter: `python -c "import tensorflow; print('ok')"`. On Windows use `python` (not `python3` — PyManager wrapper redirects to missing Anaconda shim).
- `git push` 403: PAT needs **Contents: Read & write** on the repo. Verify at `/repos/<owner>/<repo>/contents/__probe__.md` via PUT (expect 201).
- MediaPipe 1.1.x removed `mp.solutions` — the drowsiness API uses `mp.tasks.vision.FaceLandmarker` and needs the `.task` model asset at the path `MP_MODEL_PATH` points to. If download fails, put the `.task` at the configured path.
- Background server "started but not reachable": wait 5-10 s, then probe `/health`.

## Author

Giridharan — [GitHub](https://github.com/Giridharan1718)
