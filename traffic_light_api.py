#!/usr/bin/env python
"""Traffic Light detection API (YOLO11 + color heuristic for red/yellow/green)."""
import base64, io, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

model = None
def load():
    global model
    model = YOLO('yolo11n.pt')
load()

app = FastAPI(title='Traffic Light API', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

TL_CLASS_ID = 9  # COCO class 'traffic light'

def classify_color(img_pil):
    import numpy as np
    arr = np.array(img_pil.convert('RGB'))
    H, W = arr.shape[:2]
    h = arr[:,:,0].astype(int); s = arr[:,:,1].astype(int); v = arr[:,:,2].astype(int)
    # simple per-channel dominance regions
    red_mask = (h > 190) | (h < 25)
    yellow_mask = (h >= 25) & (h < 55)
    green_mask = (h >= 55) & (h < 110)
    total = arr.shape[0]*arr.shape[1]
    r = red_mask.sum(); g = green_mask.sum(); y = yellow_mask.sum()
    if r>total*0.01: return 'red'
    if y>total*0.01: return 'yellow'
    if g>total*0.01: return 'green'
    return 'unknown'

@app.get('/health')
def health(): return {'status':'healthy','traffic_light_class_id':TL_CLASS_ID}

@app.post('/detect_traffic_light')
async def detect(file: UploadFile = File(...), conf: float = 0.3):
    img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    res = model.predict(img, conf=conf, classes=[TL_CLASS_ID], verbose=False)
    out=[]
    for r in res:
        for b in r.boxes:
            x1,y1,x2,y2 = [int(x) for x in b.xyxy[0].tolist()]
            crop = img.crop((x1,y1,x2,y2))
            state = classify_color(crop)
            out.append({'state':state, 'confidence': float(b.conf.item()), 'bbox_xyxy':[x1,y1,x2,y2]})
    return {'count': len(out), 'lights': out}

@app.post('/detect_traffic_light_base64')
async def detect_tl_b64(payload: dict):
    import numpy as np
    img = Image.open(io.BytesIO(base64.b64decode(payload['image']))).convert('RGB')
    conf = payload.get('conf',0.3)
    res = model.predict(img, conf=conf, classes=[TL_CLASS_ID], verbose=False)
    out=[]
    for r in res:
        for b in r.boxes:
            x1,y1,x2,y2 = [int(x) for x in b.xyxy[0].tolist()]
            crop = img.crop((x1,y1,x2,y2))
            state = classify_color(crop)
            out.append({'state':state,'confidence':float(b.conf.item()),'bbox_xyxy':[x1,y1,x2,y2]})
    return {'count':len(out),'lights':out}

if __name__=='__main__':
    import uvicorn; uvicorn.run(app,host='0.0.0.0',port=8002)