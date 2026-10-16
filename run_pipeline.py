#!/usr/bin/env python
"""
Driver Behavior Classification Pipeline - Extracted from notebook
Runs end-to-end: data loading, feature extraction, balancing, training, evaluation
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports ---
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
DATA_DIR = PROJECT_ROOT
WINDOW_SIZE = 128
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
N_CLASSES = 3  # AGGRESSIVE, NORMAL, SLOW
CLASS_NAMES = ['AGGRESSIVE', 'NORMAL', 'SLOW']

# Deep Learning params
DL_EPOCHS = 10
DL_BATCH_SIZE = 32

def load_data():
    """Load the synthetic datasets"""
    print("Loading data...")
    df_7m = pd.read_csv(DATA_DIR / 'dataset_7M.csv')
    df_score = pd.read_csv(DATA_DIR / 'eco_driving_score.csv')
    print(f"  df_7m: {df_7m.shape}")
    print(f"  df_score: {df_score.shape}")
    return df_7m, df_score

def preprocess_labels(df_7m):
    """Encode class labels"""
    print("Encoding labels...")
    le = LabelEncoder()
    df_7m['Class_encoded'] = le.fit_transform(df_7m['Class'])
    
    print(f"  Classes: {le.classes_}")
    print(f"  Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df_7m, le

def extract_timeseries_features(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Extract statistical features from accelerometer data using sliding windows"""
    print("Extracting time-series features...")
    sensor_columns = ['AccX', 'AccY', 'AccZ']
    features_list = []
    labels_list = []
    
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        if len(window) != window_size:
            continue
        
        window_features = {}
        for col in sensor_columns:
            window_features[f'{col}_mean'] = window[col].mean()
            window_features[f'{col}_std'] = window[col].std()
            window_features[f'{col}_min'] = window[col].min()
            window_features[f'{col}_max'] = window[col].max()
            window_features[f'{col}_median'] = window[col].median()
            window_features[f'{col}_var'] = window[col].var()
            window_features[f'{col}_kurt'] = window[col].kurt()
            window_features[f'{col}_skew'] = window[col].skew()
        
        window_label = window['Class_encoded'].mode()[0]
        features_list.append(window_features)
        labels_list.append(window_label)
    
    features_df = pd.DataFrame(features_list)
    features_df['Class_encoded'] = labels_list
    print(f"  Extracted {len(features_df)} windows")
    return features_df

def extract_fft_features(window_data, sampling_rate=1):
    """Extract frequency domain features via FFT"""
    fft_features = {}
    sensor_columns = ['AccX', 'AccY', 'AccZ']
    window_size_local = len(window_data)
    
    for col in sensor_columns:
        fft_output = np.fft.fft(window_data[col].values)
        magnitudes = np.abs(fft_output)
        frequencies = np.fft.fftfreq(window_size_local, d=1/sampling_rate)
        
        # Exclude DC component
        dominant_idx = np.argmax(magnitudes[1:]) + 1
        dominant_frequency = frequencies[dominant_idx]
        dominant_magnitude = magnitudes[dominant_idx]
        total_energy = np.sum(magnitudes**2) / window_size_local
        
        fft_features[f'{col}_dom_freq'] = dominant_frequency
        fft_features[f'{col}_dom_mag'] = dominant_magnitude
        fft_features[f'{col}_energy'] = total_energy
    
    return fft_features

def extract_frequency_features(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Extract FFT features using sliding windows"""
    print("Extracting frequency features...")
    freq_features_list = []
    freq_labels_list = []
    
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        if len(window) != window_size:
            continue
        
        window_fft_features = extract_fft_features(window, sampling_rate=1)
        window_label = window['Class_encoded'].mode()[0]
        
        freq_features_list.append(window_fft_features)
        freq_labels_list.append(window_label)
    
    freq_df = pd.DataFrame(freq_features_list)
    freq_df['Class_encoded'] = freq_labels_list
    print(f"  Extracted {len(freq_df)} windows")
    return freq_df

def balance_data(ts_features_df, raw_df, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Balance dataset by undersampling each class to minimum count"""
    print("Balancing data...")
    min_count = ts_features_df['Class_encoded'].value_counts().min()
    print(f"  Min class count: {min_count}")
    
    # First, collect the raw window start indices for each feature row
    # Each feature row corresponds to a window starting at index i*step_size in raw_df
    window_start_indices = []
    for i in range(0, len(raw_df) - window_size + 1, step_size):
        if i + window_size <= len(raw_df):
            window_start_indices.append(i)
    
    # Map feature indices to raw window start indices
    feature_to_window = {idx: window_start_indices[idx] for idx in range(len(window_start_indices))}
    
    balanced_ts = []
    balanced_labels = []
    balanced_windows = []
    
    for class_label in ts_features_df['Class_encoded'].unique():
        class_indices = ts_features_df[ts_features_df['Class_encoded'] == class_label].index
        sampled = np.random.choice(class_indices, min_count, replace=False)
        
        balanced_ts.extend(ts_features_df.loc[sampled].drop(columns=['Class_encoded']).to_dict('records'))
        balanced_labels.extend(ts_features_df.loc[sampled, 'Class_encoded'].tolist())
        
        # Get the actual window data from raw_df
        for feat_idx in sampled:
            win_start = feature_to_window[feat_idx]
            balanced_windows.append(raw_df.iloc[win_start:win_start + window_size])
    
    balanced_ts_df = pd.DataFrame(balanced_ts)
    balanced_ts_df['Class_encoded'] = balanced_labels
    print(f"  Balanced to {len(balanced_ts_df)} samples ({len(balanced_ts_df)//N_CLASSES} per class)")
    return balanced_ts_df, balanced_labels, balanced_windows

def prepare_dl_data(balanced_windows, labels):
    """Prepare raw sensor windows for deep learning"""
    dl_sensor_columns = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
    X_dl = np.array([w[dl_sensor_columns].values for w in balanced_windows])
    y_dl = np.array(labels)
    return X_dl, y_dl

def train_traditional_models(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest and SVM"""
    print("\n--- Traditional ML Models ---")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    
    rf_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'cm': confusion_matrix(y_test, y_pred_rf)
    }
    print(f"  RF Accuracy: {rf_metrics['accuracy']:.4f}")
    
    # SVM
    print("Training SVM...")
    svm = SVC(kernel='linear', random_state=42, max_iter=100000, probability=True)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    
    svm_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'precision': precision_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'cm': confusion_matrix(y_test, y_pred_svm)
    }
    print(f"  SVM Accuracy: {svm_metrics['accuracy']:.4f}")
    
    return rf, svm, scaler, rf_metrics, svm_metrics

def train_dl_models(X_train, X_test, y_train, y_test, input_shape, n_classes):
    """Train and evaluate 1D CNN and LSTM"""
    print("\n--- Deep Learning Models ---")
    
    # 1D CNN
    print("Training 1D CNN...")
    cnn_model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train, y_train, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE, 
                  validation_split=0.2, verbose=0)
    
    y_pred_cnn = np.argmax(cnn_model.predict(X_test, verbose=0), axis=-1)
    cnn_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_cnn),
        'precision': precision_score(y_test, y_pred_cnn, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_cnn, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred_cnn, average='weighted', zero_division=0),
        'cm': confusion_matrix(y_test, y_pred_cnn)
    }
    print(f"  CNN Accuracy: {cnn_metrics['accuracy']:.4f}")
    
    # LSTM
    print("Training LSTM...")
    lstm_model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(100, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train, y_train, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                   validation_split=0.2, verbose=0)
    
    y_pred_lstm = np.argmax(lstm_model.predict(X_test, verbose=0), axis=-1)
    lstm_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_lstm),
        'precision': precision_score(y_test, y_pred_lstm, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_lstm, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred_lstm, average='weighted', zero_division=0),
        'cm': confusion_matrix(y_test, y_pred_lstm)
    }
    print(f"  LSTM Accuracy: {lstm_metrics['accuracy']:.4f}")
    
    return cnn_model, lstm_model, cnn_metrics, lstm_metrics

def cross_validate_traditional(X, y, n_splits=5):
    """Stratified k-fold cross-validation for traditional models"""
    print(f"\n--- {n_splits}-Fold Cross-Validation ---")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    rf_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    svm_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    rf_cms = []
    svm_cms = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/{n_splits}")
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        
        # RF
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_tr_scaled, y_tr)
        y_pred = rf.predict(X_te_scaled)
        rf_scores['accuracy'].append(accuracy_score(y_te, y_pred))
        rf_scores['precision'].append(precision_score(y_te, y_pred, average='weighted', zero_division=0))
        rf_scores['recall'].append(recall_score(y_te, y_pred, average='weighted', zero_division=0))
        rf_scores['f1'].append(f1_score(y_te, y_pred, average='weighted', zero_division=0))
        rf_cms.append(confusion_matrix(y_te, y_pred))
        
        # SVM
        svm = SVC(kernel='linear', random_state=42, max_iter=100000, probability=True)
        svm.fit(X_tr_scaled, y_tr)
        y_pred = svm.predict(X_te_scaled)
        svm_scores['accuracy'].append(accuracy_score(y_te, y_pred))
        svm_scores['precision'].append(precision_score(y_te, y_pred, average='weighted', zero_division=0))
        svm_scores['recall'].append(recall_score(y_te, y_pred, average='weighted', zero_division=0))
        svm_scores['f1'].append(f1_score(y_te, y_pred, average='weighted', zero_division=0))
        svm_cms.append(confusion_matrix(y_te, y_pred))
    
    # Aggregate
    for model_name, scores in [('RF', rf_scores), ('SVM', svm_scores)]:
        print(f"\n  {model_name} CV Results:")
        for metric, vals in scores.items():
            print(f"    {metric}: {np.mean(vals):.4f} (+/- {np.std(vals):.4f})")
    
    return rf_scores, svm_scores, rf_cms, svm_cms

def print_summary(all_metrics, class_names):
    """Print final comparison table"""
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON (balanced test split; note overlap may leak info between folds)")
    print("="*70)
    print(f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-"*70)
    for name, m in all_metrics.items():
        print(f"{name:<15} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print("="*70)
    
    print("\nConfusion Matrices:")
    for name, m in all_metrics.items():
        print(f"\n{name}:")
        cm = m['cm']
        print(f"  True\\Pred  " + "  ".join(f"{c:>8}" for c in class_names))
        for i, row in enumerate(cm):
            print(f"  {class_names[i]:<10} " + "  ".join(f"{v:>8}" for v in row))

def save_models(rf, svm, scaler, cnn_model, lstm_model, le):
    """Save trained models for API serving"""
    import joblib
    
    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(rf, models_dir / 'rf_model.pkl')
    joblib.dump(svm, models_dir / 'svm_model.pkl')
    joblib.dump(scaler, models_dir / 'scaler.pkl')
    joblib.dump(le, models_dir / 'label_encoder.pkl')
    
    cnn_model.save(models_dir / 'cnn_model.keras')
    lstm_model.save(models_dir / 'lstm_model.keras')
    
    print(f"\nModels saved to {models_dir}")

def main():
    print("="*70)
    print("DRIVER BEHAVIOR CLASSIFICATION PIPELINE")
    print("="*70)
    
    # 1. Load data
    df_7m, df_score = load_data()
    
    # 2. Preprocess labels
    df_7m, le = preprocess_labels(df_7m)
    
    # 3. Extract time-series features (on full dataset for balancing)
    ts_features = extract_timeseries_features(df_7m)
    
    # 4. Extract frequency features
    freq_features = extract_frequency_features(df_7m)
    
    # 5. Combine features
    print("Combining features...")
    combined = ts_features.merge(freq_features.drop(columns=['Class_encoded']), 
                                  left_index=True, right_index=True)
    print(f"  Combined shape: {combined.shape}")
    
    # 6. Balance data
    balanced_ts, balanced_labels, balanced_windows = balance_data(ts_features, df_7m)
    
    # Re-extract frequency features on balanced windows
    balanced_freq_features = []
    for w in balanced_windows:
        balanced_freq_features.append(extract_fft_features(w))
    balanced_freq_df = pd.DataFrame(balanced_freq_features)
    balanced_freq_df['Class_encoded'] = balanced_labels
    
    # Combine balanced features
    balanced_combined = balanced_ts.merge(balanced_freq_df.drop(columns=['Class_encoded']),
                                           left_index=True, right_index=True)
    print(f"  Balanced combined shape: {balanced_combined.shape}")
    
    # 7. Prepare features and labels
    X = balanced_combined.drop(columns=['Class_encoded'])
    y = balanced_combined['Class_encoded']
    
    # 8. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    
    # 9. Cross-validation on balanced data
    cross_validate_traditional(X, y)
    
    # 10. Train traditional models on train split
    rf, svm, scaler, rf_metrics, svm_metrics = train_traditional_models(
        X_train, X_test, y_train, y_test
    )
    
    # 11. Prepare DL data
    X_dl, y_dl = prepare_dl_data(balanced_windows, balanced_labels)
    X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
        X_dl, y_dl, test_size=0.2, random_state=42, stratify=y_dl
    )
    input_shape = (X_train_dl.shape[1], X_train_dl.shape[2])
    
    # 12. Train DL models
    cnn_model, lstm_model, cnn_metrics, lstm_metrics = train_dl_models(
        X_train_dl, X_test_dl, y_train_dl, y_test_dl, input_shape, N_CLASSES
    )
    
    # 13. Summary
    all_metrics = {
        'RandomForest': rf_metrics,
        'SVM': svm_metrics,
        '1D_CNN': cnn_metrics,
        'LSTM': lstm_metrics
    }
    print_summary(all_metrics, le.classes_)
    
    # 14. Save models
    save_models(rf, svm, scaler, cnn_model, lstm_model, le)
    
    print("\n✓ Pipeline completed successfully!")
    return all_metrics

if __name__ == '__main__':
    main()