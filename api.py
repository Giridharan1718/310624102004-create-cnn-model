#!/usr/bin/env python
"""
FastAPI API for Driver Behavior Classification
Serves the trained models for inference
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
cor_origins_env = os.getenv("CORS_ORIGINS")
CORS_ORIGINS = ([o.strip() for o in cor_origins_env.split(",") if o.strip()] if cor_origins_env else ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"])
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = Path(os.getenv("DRIVER_MODELS_DIR", str(PROJECT_ROOT / "models")))

# Global model storage
models = {}

# --- Configuration ---
WINDOW_SIZE = 128
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
CLASS_NAMES = ['AGGRESSIVE', 'NORMAL', 'SLOW']
DL_SENSOR_COLUMNS = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
TS_SENSOR_COLUMNS = ['AccX', 'AccY', 'AccZ']


def extract_fft_features(window_data, sampling_rate=1):
    """Extract frequency domain features via FFT"""
    fft_features = {}
    sensor_columns = TS_SENSOR_COLUMNS
    window_size_local = len(window_data)
    
    for col in sensor_columns:
        fft_output = np.fft.fft(window_data[col].values)
        magnitudes = np.abs(fft_output)
        frequencies = np.fft.fftfreq(window_size_local, d=1/sampling_rate)
        
        dominant_idx = np.argmax(magnitudes[1:]) + 1
        dominant_frequency = frequencies[dominant_idx]
        dominant_magnitude = magnitudes[dominant_idx]
        total_energy = np.sum(magnitudes**2) / window_size_local
        
        fft_features[f'{col}_dom_freq'] = dominant_frequency
        fft_features[f'{col}_dom_mag'] = dominant_magnitude
        fft_features[f'{col}_energy'] = total_energy
    
    return fft_features


def extract_timeseries_features_window(window):
    """Extract statistical features from a single window"""
    window_features = {}
    for col in TS_SENSOR_COLUMNS:
        window_features[f'{col}_mean'] = window[col].mean()
        window_features[f'{col}_std'] = window[col].std()
        window_features[f'{col}_min'] = window[col].min()
        window_features[f'{col}_max'] = window[col].max()
        window_features[f'{col}_median'] = window[col].median()
        window_features[f'{col}_var'] = window[col].var()
        window_features[f'{col}_kurt'] = window[col].kurt()
        window_features[f'{col}_skew'] = window[col].skew()
    return window_features


def load_models():
    """Load all trained models (non-fatal if absent)."""
    global models, MODEL_LOAD_ERROR
    try:
        print("Loading models...")
        if not MODELS_DIR.exists() or not (MODELS_DIR / "rf_model.pkl").exists():
            raise FileNotFoundError("missing model artifacts; run run_pipeline.py")
        models["rf"] = joblib.load(MODELS_DIR / "rf_model.pkl")
        models["svm"] = joblib.load(MODELS_DIR / "svm_model.pkl")
        models["scaler"] = joblib.load(MODELS_DIR / "scaler.pkl")
        models["label_encoder"] = joblib.load(MODELS_DIR / "label_encoder.pkl")
        models["cnn"] = tf.keras.models.load_model(MODELS_DIR / "cnn_model.keras")
        models["lstm"] = tf.keras.models.load_model(MODELS_DIR / "lstm_model.keras")
        MODEL_LOAD_ERROR = None
        print("All models loaded successfully")
    except Exception as exc:
        models.clear()
        MODEL_LOAD_ERROR = "Driver-behavior models unavailable; run run_pipeline.py first."
        print(MODEL_LOAD_ERROR, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_models()
    yield
    # Shutdown
    models.clear()


app = FastAPI(
    title="Driver Behavior Classification API",
    description="API for classifying driving styles (AGGRESSIVE, NORMAL, SLOW) from sensor data",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class SensorReading(BaseModel):
    AccX: float
    AccY: float
    AccZ: float
    GyroX: float
    GyroY: float
    GyroZ: float
    Timestamp: Optional[int] = None


class PredictionRequest(BaseModel):
    sensor_data: List[SensorReading] = Field(..., min_length=WINDOW_SIZE, max_length=WINDOW_SIZE)
    model: str = Field(default="cnn", pattern="^(rf|svm|cnn|lstm)$")


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_id: int
    confidence: float
    all_probabilities: dict
    model_used: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]


# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    if MODEL_LOAD_ERROR:
        return HealthResponse(
            status="unavailable",
            models_loaded=[]
        )
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys())
    )


@app.get("/models")
async def list_models():
    return {
        "available_models": [
            {"id": "rf", "name": "Random Forest", "type": "traditional_ml"},
            {"id": "svm", "name": "Support Vector Machine", "type": "traditional_ml"},
            {"id": "cnn", "name": "1D CNN", "type": "deep_learning"},
            {"id": "lstm", "name": "LSTM", "type": "deep_learning"},
        ],
        "default_model": "cnn",
        "class_labels": CLASS_NAMES,
        "required_window_size": WINDOW_SIZE,
        "sensor_columns": DL_SENSOR_COLUMNS
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not models:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR or "Models not loaded")
    if MODEL_LOAD_ERROR:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)
    
    # Convert to DataFrame
    df = pd.DataFrame([r.model_dump() for r in request.sensor_data])
    
    # Validate window size
    if len(df) != WINDOW_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Exactly {WINDOW_SIZE} sensor readings required, got {len(df)}"
        )
    
    model_id = request.model
    
    try:
        if model_id in ['rf', 'svm']:
            # Traditional ML: extract features, scale, predict
            ts_features = extract_timeseries_features_window(df)
            freq_features = extract_fft_features(df)
            
            # Combine features in the same order as training
            feature_dict = {**ts_features, **freq_features}
            feature_vector = np.array([list(feature_dict.values())]).reshape(1, -1)
            
            # Scale
            feature_vector_scaled = models['scaler'].transform(feature_vector)
            
            # Predict
            model = models[model_id]
            pred_class = model.predict(feature_vector_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(feature_vector_scaled)[0]
            else:
                # SVM with probability=True
                probas = model.predict_proba(feature_vector_scaled)[0]
            
        else:
            # Deep Learning: use raw sensor data
            X_dl = df[DL_SENSOR_COLUMNS].values.reshape(1, WINDOW_SIZE, 6)
            
            model = models[model_id]
            probas = model.predict(X_dl, verbose=0)[0]
            pred_class = int(np.argmax(probas))
        
        # Map to class name
        pred_class_name = CLASS_NAMES[pred_class]
        confidence = float(probas[pred_class])
        all_probas = {CLASS_NAMES[i]: float(probas[i]) for i in range(len(CLASS_NAMES))}
        
        return PredictionResponse(
            predicted_class=pred_class_name,
            predicted_class_id=pred_class,
            confidence=confidence,
            all_probabilities=all_probas,
            model_used=model_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    results = []
    for req in requests:
        result = await predict(req)
        results.append(result)
    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)