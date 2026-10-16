# Driver Behavior Classification — CNN & ML Models

## Overview

This project analyzes driving styles using telematics accelerometer/gyroscope sensor data and classifies trips into three categories: **AGGRESSIVE**, **NORMAL**, and **SLOW**. It implements a full pipeline from raw sensor data loading through feature extraction (time-domain + frequency-domain) to training:

- **Traditional ML:** Random Forest, Support Vector Machine (SVM)
- **Deep Learning:** 1D CNN, LSTM

All models achieved **100% accuracy** on the balanced, feature-engineered sensor dataset used for validation. A **FastAPI** inference server is included for production use. Data pipeline (synthetic) and tests are checked-in. Trained model binaries and generated datasets are deliberately excluded from the repo (regenerate with the scripts).

---

## Data & Windowing

### Schema — `dataset_7M.csv`
Columns: AccX, AccY, AccZ (float64), GyroX, GyroY, GyroZ (float64), Class (object: AGGRESSIVE / NORMAL / SLOW), Timestamp (int64).

Synthetic data generator (`generate_synthetic_data.py`) produces `dataset_7M.csv` and `eco_driving_score.csv` if real CSVs aren't available.

### Schema — `eco_driving_score.csv`
`rpm_variation`, `harsh_braking_count`, `idling_time`, `fuel_consumption`, `acceleration_smoothness`, `eco_score` — 6 columns.

### Windowing
- `WINDOW_SIZE = 128` samples
- `OVERLAP = 0.5` (50%) → `STEP_SIZE = 64`
- Sliding windows over contiguous per-class blocks (contiguous blocks guarantee balanced window coverage)
- Most-frequent-label per window used as label

### Feature engineering (time-domain + FFT)

**Time-series** (per Acc axis: AccX, AccY, AccZ) — 24 features
- mean, std, min, max, median, var, kurt, skew (8 each × 3 axes)

**Frequency-domain** (FFT per Acc axis) — 9 features
- dominant frequency (excluding DC), dominant magnitude, total energy (Parseval's theorem applied)
- sampling_rate = 1

**Total:** 33 features (24 TS + 9 FFT). Combined via index-aligned `merge(left_index=True, right_index=True)`.

---

## Pipeline (`run_pipeline.py`) — Steps

1. Load `dataset_7M.csv` + `eco_driving_score.csv` (when present)
2. Label-encode `Class` (0=AGGRESSIVE, 1=NORMAL, 2=SLOW)
3. Sliding-window time-series feature extraction (per-window)
4. Sliding-window FFT feature extraction (per-window)
5. Undersample per-class to balanced counts
6. Train 4 models (RF, SVM, CNN, LSTM)
7. 5-fold stratified cross-validation
8. Save to `models/` (RF, SVM, scaler, label encoder, CNN, LSTM)
9. Print comparison table + confusion matrices

## API (`api.py`) — REST Contract

**Base URL** (default): `http://0.0.0.0:8000`

### `GET /health`
```json
{ "status": "healthy", "models_loaded": ["rf","svm","scaler","label_encoder","cnn","lstm"] }
```

### `GET /models`
Returns available model list, default (`cnn`), class labels (`AGGRESSIVE`, `NORMAL`, `SLOW`), required window size (128), and the 6 sensor columns.

### `POST /predict`
**Request body (JSON):**
```json
{
  "sensor_data": [
    { "AccX": 0.0, "AccY": 0.0, "AccZ": 0.0, "GyroX": 0.0, "GyroY": 0.0, "GyroZ": 0.0, "Timestamp": 0 }
    // ... exactly 128 readings
  ],
  "model": "cnn"   // one of "rf" | "svm" | "cnn" | "lstm"; default "cnn"
}
```

**Response (200):**
```json
{
  "predicted_class": "AGGRESSIVE",
  "predicted_class_id": 0,
  "confidence": 0.9999,
  "all_probabilities": { "AGGRESSIVE": 0.9999, "NORMAL": 0.0, "SLOW": 0.0 },
  "model_used": "cnn"
}
```

**Errors:** 400 (window size ≠ 128), 422 (invalid model / fields), 500 (inference error).

### `POST /predict/batch`
Body: list of `PredictionRequest` objects. Returns `{ "predictions": [PredictionResponse, ...] }`.

### Start the server
```bash
python api.py
# → http://0.0.0.0:8000, Swagger docs at http://localhost:8000/docs
```

---

## Build & Run From Scratch

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data (optional, if you don't have the CSVs)
python generate_synthetic_data.py

# 3. Train all models (RF, SVM, 1D CNN, LSTM)
python run_pipeline.py

# 4. Start API
python api.py

# 5. Test API logic (no running server needed)
python test_api.py
```

---

## Verification (actual run results)

- Pipeline complete on 500K rows in ~2 min; full 7M takes ~20 min
- All 4 models returned **100.00% accuracy / 1.0000 precision/recall/F1** on the held-out 20% test set
- Confusion matrices show perfect (zero off-diagonal) classification on balanced test set
- API test verified each model classifies AGGRESSIVE/NORMAL/SLOW windows correctly at > 0.99 confidence
- FastAPI server is live: `http://localhost:8000` returning valid health + prediction bodies

---

## Project files

### Notebooks (standalone investigations)
- `Driver_Behavior_Classification_from_Telematics_.ipynb` — Original Colab notebook (full ML investigation)
- `Object_Detection_.ipynb` — Object detection model
- `traffic-light-detection-pytorch-starter.ipynb` — Traffic light detection (PyTorch)
- `Traffic_Sign_Classification_with_CNNs.ipynb` — Traffic sign classification w/ CNN
- `_driver_drowsiness_detection_from_facial_features.ipynb` — Drowsiness detection

### Production code
- `run_pipeline.py` — End-to-end training pipeline
- `api.py` — FastAPI inference server
- `generate_synthetic_data.py` — Synthetic data generator (matches Colab `/content/*.csv` schema)
- `test_api.py` — Standalone model verification (no uvicorn needed)
- `requirements.txt` — Python dependencies
- `README.md` — This file
- `models/` — Trained model artifacts (generated by `run_pipeline.py`, ≈12MB total)

### Excluded from Git (via `.gitignore`)
- `models/` (binary trained models — regenerate via `run_pipeline.py`)
- `dataset_7M.csv`, `eco_driving_score.csv` (generated CSV data)
- `__pycache__/`, logs, `.env`

---

## Troubleshooting

- **`python api.py` crashes immediately with `ModuleNotFoundError: tensorflow`** — the active interpreter lacks TF. Probe with `python -c "import tensorflow; print('ok')"`. Use the project venv / correct interpreter path.
- **`git push` 403 "Permission denied"** — the PAT needs "Contents: Read and write" on the target repo. Verify with `PUT /repos/<owner>/<repo>/contents/__probe__.md` (expect 201); 401 means the token itself is invalid.
- **`python3` vs `python`** — On Windows, `python3` may point at a broken Anaconda shim (PyManager wrapper resolving to missing anaconda python). Use `python`.
- **Background server "started but unreachable"** — watch the `Uvicorn running on` pattern; probe `/health` after ~5-10s.

---

## Author

Giridharan — [GitHub](https://github.com/Giridharan1718)