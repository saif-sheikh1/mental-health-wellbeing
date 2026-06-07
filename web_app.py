from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np

from database import db
from inference import engine
from sensors import ArduinoReader, EEGReader, SensorData
import threading

app = FastAPI(title="MindSense AI Web Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

import os
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Global Sensors
sensor_data = SensorData()
arduino = ArduinoReader()
arduino.data_ref = sensor_data
eeg = EEGReader()
eeg.data_ref = sensor_data

@app.on_event("startup")
def startup_event():
    engine.load_models()
    arduino.start()
    eeg.start()

@app.on_event("shutdown")
def shutdown_event():
    arduino.stop()
    eeg.stop()

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.post("/api/register")
async def register(name: str = Form(...), phone: str = Form(""), email: str = Form("")):
    user = db.register_user(name, phone, email)
    return user

@app.get("/api/live_data")
def get_live_data(user_id: str = ""):
    vals = sensor_data.to_list()
    data_dict = sensor_data.to_dict()
    
    # Run RNN + Predictor
    state_pred = engine.predict_sensor(vals)
    future_pred = engine.predict_future(vals)
    recs = engine.get_recommendations(state_pred.get("state", "NORMAL"))
    
    # We save sensor data in DB periodically, let's do it in the background if user_id is given
    if user_id:
        def _save():
            predictions = {
                "state": state_pred.get("state"),
                "state_confidence": state_pred.get("confidence"),
                "emotion": "Unknown",  # Emotion comes from separate facial endpoint
                "emotion_confidence": 0.0
            }
            db.store_reading(user_id, data_dict, predictions)
        threading.Thread(target=_save, daemon=True).start()
    
    return {
        "sensors": data_dict,
        "state": state_pred,
        "future": future_pred,
        "recommendations": recs
    }

@app.post("/api/predict_facial")
async def predict_facial(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is not None:
        emo_pred = engine.predict_facial(frame)
        return emo_pred
    return {"emotion": "Unknown", "confidence": 0.0}

if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=9000, reload=False)
