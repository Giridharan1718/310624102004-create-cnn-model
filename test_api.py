#!/usr/bin/env python
"""
Test the API by directly calling the prediction logic
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import tensorflow as tf

MODELS_DIR = PROJECT_ROOT / 'models'

WINDOW_SIZE = 128
CLASS_NAMES = ['AGGRESSIVE', 'NORMAL', 'SLOW']
DL_SENSOR_COLUMNS = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
TS_SENSOR_COLUMNS = ['AccX', 'AccY', 'AccZ']

def extract_fft_features(window_data, sampling_rate=1):
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

# Load models
print("Loading models...")
rf = joblib.load(MODELS_DIR / 'rf_model.pkl')
svm = joblib.load(MODELS_DIR / 'svm_model.pkl')
scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
le = joblib.load(MODELS_DIR / 'label_encoder.pkl')
cnn = tf.keras.models.load_model(MODELS_DIR / 'cnn_model.keras')
lstm = tf.keras.models.load_model(MODELS_DIR / 'lstm_model.keras')
print("Models loaded!")

# Generate a test window for each class
np.random.seed(123)

for cls_name in CLASS_NAMES:
    if cls_name == 'AGGRESSIVE':
        acc_scale, gyro_scale = 0.5, 5.0
    elif cls_name == 'NORMAL':
        acc_scale, gyro_scale = 0.1, 2.0
    else:  # SLOW
        acc_scale, gyro_scale = 0.03, 0.8
    
    # Generate a window
    test_data = pd.DataFrame({
        'AccX': np.random.normal(0, acc_scale, WINDOW_SIZE),
        'AccY': np.random.normal(0, acc_scale, WINDOW_SIZE),
        'AccZ': np.random.normal(0, acc_scale, WINDOW_SIZE),
        'GyroX': np.random.normal(0, gyro_scale, WINDOW_SIZE),
        'GyroY': np.random.normal(0, gyro_scale, WINDOW_SIZE),
        'GyroZ': np.random.normal(0, gyro_scale, WINDOW_SIZE),
    })
    
    print(f"\n{'='*60}")
    print(f"Testing with TRUE CLASS: {cls_name}")
    print(f"{'='*60}")
    
    # Test each model
    for model_name, model in [('RandomForest', rf), ('SVM', svm), ('1D_CNN', cnn), ('LSTM', lstm)]:
        if model_name in ['RandomForest', 'SVM']:
            ts_feat = extract_timeseries_features_window(test_data)
            freq_feat = extract_fft_features(test_data)
            feature_dict = {**ts_feat, **freq_feat}
            feature_vector = np.array([list(feature_dict.values())]).reshape(1, -1)
            feature_vector_scaled = scaler.transform(feature_vector)
            
            pred = model.predict(feature_vector_scaled)[0]
            proba = model.predict_proba(feature_vector_scaled)[0]
        else:
            X_dl = test_data[DL_SENSOR_COLUMNS].values.reshape(1, WINDOW_SIZE, 6)
            proba = model.predict(X_dl, verbose=0)[0]
            pred = int(np.argmax(proba))
        
        pred_name = CLASS_NAMES[pred]
        confidence = proba[pred]
        correct = "✓" if pred_name == cls_name else "✗"
        print(f"  {model_name:15} -> {pred_name:10} (conf: {confidence:.4f}) {correct}")

print("\n✓ API logic test complete!")