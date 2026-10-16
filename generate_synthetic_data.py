#!/usr/bin/env python
"""
Generate synthetic driving behavior data matching the notebook's expected schema.
Ensures balanced representation in windowed features.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Output directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# ---- dataset_7M.csv ----
# 7M rows, columns: AccX, AccY, AccZ, GyroX, GyroY, GyroZ, Class, Timestamp
# Classes: SLOW, NORMAL, AGGRESSIVE (matching notebook)
n_samples = 500_000  # Reduced for faster pipeline execution

# Generate data in class-balanced blocks to ensure each class gets enough window samples
# Use equal class sizes for better window representation
n_per_class = n_samples // 3
classes = ['AGGRESSIVE', 'NORMAL', 'SLOW']

# Pre-allocate arrays
data = {
    'AccX': np.zeros(n_samples),
    'AccY': np.zeros(n_samples),
    'AccZ': np.zeros(n_samples),
    'GyroX': np.zeros(n_samples),
    'GyroY': np.zeros(n_samples),
    'GyroZ': np.zeros(n_samples),
    'Class': np.empty(n_samples, dtype=object),
    'Timestamp': np.arange(n_samples, dtype=np.int64)
}

# Generate each class in contiguous blocks (easier for windowing)
idx = 0
for cls in classes:
    if cls == 'AGGRESSIVE':
        acc_scale = 0.5
        gyro_scale = 5.0
    elif cls == 'NORMAL':
        acc_scale = 0.1
        gyro_scale = 2.0
    else:  # SLOW
        acc_scale = 0.03
        gyro_scale = 0.8
    
    end_idx = idx + n_per_class
    data['AccX'][idx:end_idx] = np.random.normal(0, acc_scale, n_per_class)
    data['AccY'][idx:end_idx] = np.random.normal(0, acc_scale, n_per_class)
    data['AccZ'][idx:end_idx] = np.random.normal(0, acc_scale, n_per_class)
    data['GyroX'][idx:end_idx] = np.random.normal(0, gyro_scale, n_per_class)
    data['GyroY'][idx:end_idx] = np.random.normal(0, gyro_scale, n_per_class)
    data['GyroZ'][idx:end_idx] = np.random.normal(0, gyro_scale, n_per_class)
    data['Class'][idx:end_idx] = cls
    idx = end_idx

# Handle remainder
remainder = n_samples - idx
if remainder > 0:
    cls = 'NORMAL'
    data['AccX'][idx:] = np.random.normal(0, 0.1, remainder)
    data['AccY'][idx:] = np.random.normal(0, 0.1, remainder)
    data['AccZ'][idx:] = np.random.normal(0, 0.1, remainder)
    data['GyroX'][idx:] = np.random.normal(0, 2.0, remainder)
    data['GyroY'][idx:] = np.random.normal(0, 2.0, remainder)
    data['GyroZ'][idx:] = np.random.normal(0, 2.0, remainder)
    data['Class'][idx:] = cls

df_7m = pd.DataFrame(data)
df_7m.to_csv(os.path.join(output_dir, 'dataset_7M.csv'), index=False)
print(f"Generated dataset_7M.csv: {df_7m.shape}")

# ---- eco_driving_score.csv ----
# 30000 rows, columns: rpm_variation, harsh_braking_count, idling_time, 
# fuel_consumption, acceleration_smoothness, eco_score
n_eco = 30_000
eco_data = {
    'rpm_variation': np.random.randint(1000, 3000, n_eco),
    'harsh_braking_count': np.random.randint(0, 10, n_eco),
    'idling_time': np.random.uniform(0, 30, n_eco).round(1),
    'fuel_consumption': np.random.uniform(5, 12, n_eco).round(2),
    'acceleration_smoothness': np.random.uniform(0.3, 1.0, n_eco).round(2),
    'eco_score': np.random.uniform(20, 100, n_eco).round(1),
}
df_eco = pd.DataFrame(eco_data)
df_eco.to_csv(os.path.join(output_dir, 'eco_driving_score.csv'), index=False)
print(f"Generated eco_driving_score.csv: {df_eco.shape}")

print("Synthetic data generation complete.")