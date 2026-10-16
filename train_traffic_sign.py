"""
Train a traffic sign classifier on GTSRB via HuggingFace streaming.
Saves a model to ../models/ts_classifier.keras
"""
import os, numpy as np, io
from PIL import Image
import time
from datasets import load_dataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib, pickle, json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

N_SAMPLES = 2500
IMG_SIZE = 32
NUM_CLASSES = 43

# --- Stream-collect from GTSRB with per-class stratification ---
print('Streaming GTSRB from HuggingFace (stratified)...')
X, y = [], []
t0 = time.time()
ds = load_dataset('tanganke/gtsrb', split='train', streaming=True)
cnt = 0
class_counts = {}
MAX_PER_CLASS = 25
TARGET_CLASSES_SEEN = 43
seen_classes = 0
for ex in ds:
    img = ex['image']
    label = ex['label']
    if class_counts.get(label, 0) >= MAX_PER_CLASS:
        continue
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    X.append(np.array(img))
    y.append(label)
    class_counts[label] = class_counts.get(label, 0) + 1
    cnt += 1
    if label not in class_counts or (class_counts[label] == 1):
        seen_classes = len(class_counts)
    if cnt % 500 == 0:
        print(f'  collected {cnt}, classes seen so far: {len(class_counts)}', flush=True)
    if len(class_counts) >= TARGET_CLASSES_SEEN and all(c >= MAX_PER_CLASS for c in class_counts.values()):
        break

X = np.stack(X).astype('float32')
y = np.array(y)
from tensorflow.keras.utils import to_categorical
y_one = to_categorical(y, NUM_CLASSES)
print(f'Collected X={X.shape}, y={y.shape} in {time.time()-t0:.0f}s; classes={len(class_counts)}; counts min={min(class_counts.values())}')

# Train/val split
Xtr, Xva, ytr, yva = train_test_split(X, y_one, test_size=0.2, random_state=42, stratify=y)
Xtrn = Xtr.astype("float32")
Xvan = Xva.astype("float32")
# vgg16 preprocess uses VGG16 preprocessing centered inputs
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
Xtrn = vgg_preprocess(Xtrn.copy())
Xvan = vgg_preprocess(Xvan.copy())
print('split: train', Xtrn.shape, Xvan.shape)

# --- Build VGG16-transfer CNN for GTSRB (VGG16 preprocess + frozen transfer head) ---
base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False
m = keras.Sequential([base, layers.GlobalAveragePooling2D(), layers.Dense(256, activation='relu'), layers.Dropout(0.3), layers.Dense(NUM_CLASSES, activation='softmax')])
m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('training VGG16-transfer head ...')
hist = m.fit(Xtrn, ytr, epochs=8, batch_size=32, validation_data=(Xvan, yva), verbose=1)
# quick eval
pred = m.predict(Xvan, verbose=0)
acc = accuracy_score(np.argmax(yva, axis=1), np.argmax(pred, axis=-1))
print('eval acc on val:', acc)
m.save(os.path.join(MODELS_DIR, 'gtsrb_model.keras'))
# metadata classes
with open(os.path.join(MODELS_DIR, 'gtsrb_labels.json'), 'w') as f:
    json.dump({'classes': sorted(list(set(y.tolist()))) , 'num_classes':43}, f)
print('saved:', os.path.join(MODELS_DIR, 'gtsrb_model.keras'))