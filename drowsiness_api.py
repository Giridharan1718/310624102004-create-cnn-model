#!/usr/bin/env python
"""Driver Drowsiness Detection API using MediaPipe Face Landmarker (EAR+MAR)."""
import base64, io, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as taskspython
from mediapipe.tasks.python import vision as visions

MP_MODEL_PATH = "C:/Users/girid/Downloads/scratchpad/mpmodels/face_landmarker.task"
options = visions.FaceLandmarkerOptions(
    base_options=taskspython.BaseOptions(model_asset_path=MP_MODEL_PATH),
    output_face_blendshapes=False,
    num_faces=1
)
FACE = visions.FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
M_UPPER = 13
M_LOWER = 14
M_LEFT = 61
M_RIGHT = 291

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6

def EAR(indices, lm):
    p = [(lm[i].x, lm[i].y) for i in indices]
    A = np.linalg.norm(np.array(p[1])-np.array(p[5]))
    B = np.linalg.norm(np.array(p[2])-np.array(p[4]))
    C = np.linalg.norm(np.array(p[0])-np.array(p[3]))
    return (A+B)/(2.0*C+1e-9)

def MAR(lm):
    p = lambda i: np.array((lm[i].x, lm[i].y))
    h = np.linalg.norm(p(M_LEFT)-p(M_RIGHT))
    v = np.linalg.norm(p(M_UPPER)-p(M_LOWER))
    return v/(h+1e-9)

app = FastAPI(title='Driver Drowsiness Detection API', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

def analyze_image(img: Image.Image):
    img_rgb = np.array(img.convert('RGB'))
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = FACE.detect(mpimg)
    faces = result.face_landmarks if result.face_landmarks else []
    if not faces:
        return {'face_detected': False, 'state':'unknown','reason':'no face'}
    lm = faces[0]
    ear_l = EAR(LEFT_EYE, lm)
    ear_r = EAR(RIGHT_EYE, lm)
    ear = (ear_l+ear_r)/2.0
    mar = MAR(lm)
    eyes_closed = bool(ear < EAR_THRESHOLD)
    yawning = bool(mar > MAR_THRESHOLD)
    drowsy = eyes_closed or yawning
    return {
        'face_detected': True,
        'ear': round(float(ear),4),
        'mar': round(float(mar),4),
        'eyes_closed': eyes_closed,
        'yawning': yawning,
        'state': 'DROWSY' if drowsy else 'ALERT'
    }

@app.get('/health')
def health(): return {'status':'healthy', 'mediapipe':'landmarker+float16'}

@app.post('/detect_drowsiness')
async def drowsy_endpoint(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    return analyze_image(img)

@app.post('/detect_drowsiness_base64')
async def drowsy_b64(payload: dict):
    img = Image.open(io.BytesIO(base64.b64decode(payload['image'])))
    return analyze_image(img)

if __name__=='__main__':
    import uvicorn; uvicorn.run(app,host='0.0.0.0',port=8003)
