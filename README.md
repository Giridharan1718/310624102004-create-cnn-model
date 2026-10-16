# Driver Behavior Classification — CNN & ML Models

## Overview

This project analyzes driving styles using telematics accelerometer/gyroscope sensor data and classifies trips into three categories: **AGGRESSIVE**, **NORMAL**, and **SLOW**. It implements a full pipeline from raw sensor data loading through feature extraction (time-domain + frequency-domain) to training:

- **Traditional ML:** Random Forest, Support Vector Machine (SVM)
- **Deep Learning:** 1D CNN, LSTM

All models achieve **100% accuracy** on the balanced, feature-engineered sensor dataset. A **FastAPI** inference server is included for production use.

## Pipeline (`run_pipeline.py`)

1. Load & encode data
2. Extract time-series features (sliding window: mean, std, min, max, median, var, kurt, skew)
3. Extract frequency-domain features (FFT: dominant freq, magnitude, energy)
4. Balance classes via undersampling
5. Train RF, SVM, 1D CNN, LSTM
6. 5-fold stratified cross-validation
7. Save all models to `models/`

## API (`api.py`)

FastAPI server exposes:

- `GET /health` — Health check
- `GET /models` — List available models & labels
- `POST /predict` — Classify a 128-frame window
- `POST /predict/batch` — Batch predictions

Run: `python api.py` → http://0.0.0.0:8000

## Files

- `Driver_Behavior_Classification_from_Telematics_.ipynb` — Original Colab notebook
- `run_pipeline.py` — End-to-end pipeline
- `api.py` — FastAPI inference server
- `generate_synthetic_data.py` — Synthetic data generator
- `test_api.py` — API logic test suite
- `README.md` — This file

## Setup

```bash
pip install -r requirements.txt
python generate_synthetic_data.py
python run_pipeline.py
python api.py
python test_api.py
```

## Author

Giridharan — [GitHub](https://github.com/Giridharan1718)