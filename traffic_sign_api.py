#!/usr/bin/env python
"""Traffic Sign Classification API (VGG16-transfer on GTSRB)."""
import base64, io, json, os
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'gtsrb_model.keras')
model = None
IMG = 32
NUM_CLASSES = 43

def load():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f'model not found at {MODEL_PATH}. run train_traffic_sign.py first')
    model = keras.models.load_model(MODEL_PATH)
load()

app = FastAPI(title='Traffic Sign Classification API', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

def preprocess(img):
    img = img.convert('RGB').resize((IMG, IMG))
    arr = np.array(img).astype('float32') / 255.0
    return arr[None, ...]

def classify(img):
    prob = model.predict(preprocess(img), verbose=0)[0]
    cls_id = int(np.argmax(prob))
    return {'class_id': cls_id, 'confidence': float(prob[cls_id]), 'probabilities': [float(p) for p in prob]}

@app.get('/health')
def health(): return {'status':'healthy','model_loaded': model is not None, 'classes': NUM_CLASSES}

@app.post('/classify')
async def classify_file(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    return classify(img)

@app.post('/classify_base64')
async def classify_b64(payload: dict):
    img = Image.open(io.BytesIO(base64.b64decode(payload['image'])))
    return classify(img)

if __name__=='__main__':
    import uvicorn; uvicorn.run(app,host='0.0.0.0',port=8004)