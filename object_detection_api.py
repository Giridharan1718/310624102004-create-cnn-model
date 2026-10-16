#!/usr/bin/env python
"""Object Detection API using Ultralytics YOLO11."""
import base64, io, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

model = None

def load():
    global model
    # Auto-downloads yolo11n on first inference
    model = YOLO('yolo11n.pt')

load()

app = FastAPI(title='Object Detection API', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.get('/health')
def health(): return {'status':'healthy','classes':list(model.names.values())}

@app.post('/detect')
async def detect(file: UploadFile = File(...), conf: float = 0.35):
    img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    res = model.predict(img, conf=conf, verbose=False)
    out=[]
    for r in res:
        for b in r.boxes:
            out.append({
                'class_id': int(b.cls.item()),
                'class_name': model.names[int(b.cls.item())],
                'confidence': float(b.conf.item()),
                'bbox_xyxy': [float(x) for x in b.xyxy[0].tolist()]
            })
    return {'n': len(out), 'detections': out}

@app.post('/detect/base64')
async def detect_b64(payload: dict):
    img = Image.open(io.BytesIO(base64.b64decode(payload['image']))).convert('RGB')
    conf = payload.get('conf',0.35)
    res = model.predict(img, conf=conf, verbose=False)
    out=[]
    for r in res:
        for b in r.boxes:
            out.append({
                'class_id': int(b.cls.item()),
                'class_name': model.names[int(b.cls.item())],
                'confidence': float(b.conf.item()),
                'bbox_xyxy': [float(x) for x in b.xyxy[0].tolist()]
            })
    return {'n': len(out), 'detections': out}

if __name__=='__main__':
    import uvicorn; uvicorn.run(app,host='0.0.0.0',port=8001)