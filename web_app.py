from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np

from database import db
from sensors import ArduinoReader, EEGReader, SensorData
import threading

app = FastAPI(title="MindSense AI Web Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

import os
os.makedirs("templates", exist_ok=True)
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"

# Global Sensors
sensor_data = SensorData()
arduino = ArduinoReader()
arduino.data_ref = sensor_data
eeg = EEGReader()
eeg.data_ref = sensor_data
engine = None
engine_loading = False
engine_error = None

STATE_ORDER = [
    "NORMAL",
    "LOW_STRESS",
    "MODERATE_STRESS",
    "HIGH_ANXIETY",
    "PANIC_STATE",
    "DEPRESSION",
]

def build_status_payload():
    engine_status = getattr(engine, "status", {}) if engine is not None else {}
    return {
        "status": "ok",
        "models": {
            "facial": bool(engine_status.get("facial")),
            "vit_facial": bool(engine_status.get("facial")),
            "rnn": bool(engine_status.get("rnn")),
            "rnn_sensor": bool(engine_status.get("rnn")),
            "predictor": bool(engine_status.get("predictor")),
            "future_predictor": bool(engine_status.get("predictor")),
            "scaler": bool(engine_status.get("scaler")),
            "loading": bool(engine_loading),
            "error": engine_error,
        },
        "sensors": {
            "arduino": bool(getattr(arduino, "connected", False)),
            "eeg": bool(getattr(eeg, "connected", False)),
        },
    }

def build_recommendation_details(state: str, emotion: str, future_states: list[str]) -> dict:
    normalized_state = (state or "NORMAL").upper()
    normalized_emotion = (emotion or "neutral").lower()
    state_nums = [
        STATE_ORDER.index(s) if s in STATE_ORDER else 0
        for s in future_states
    ]
    trend = (
        "worsening" if len(state_nums) >= 2 and state_nums[-1] > state_nums[0] else
        "improving" if len(state_nums) >= 2 and state_nums[-1] < state_nums[0] else
        "stable"
    )

    if normalized_state in {"HIGH_ANXIETY", "PANIC_STATE"}:
        first_aid = [
            "Box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s. Repeat 4 times.",
            "5-4-3-2-1 grounding: name 5 things you see, 4 you feel, and 3 you hear.",
            "Drink a glass of cold water slowly.",
        ]
        lifestyle = [
            "Reduce screen stimulation for the next two hours.",
            "Aim for 7 to 9 hours of sleep tonight.",
        ]
        professional = [
            "If high stress persists for more than three days, consult a licensed therapist.",
        ]
    elif normalized_state == "MODERATE_STRESS":
        first_aid = [
            "Step outside for two to three minutes if possible.",
            "Put on calm music at low volume.",
        ]
        lifestyle = [
            "Schedule 30 minutes of moderate exercise tomorrow.",
            "Journal your top stressors for five minutes before bed.",
        ]
        professional = [
            "Consider a routine check-in with your counselor if symptoms continue.",
        ]
    elif normalized_state in {"DEPRESSION", "LOW_STRESS"}:
        first_aid = [
            "Increase ambient light by opening blinds or turning on lights.",
            "Do a two-minute stretch or short movement reset.",
            "Hydrate with 250 ml of water.",
        ]
        lifestyle = [
            "Keep a consistent wake-up time to regulate your daily rhythm.",
            "Review diet and include complex carbs and iron-rich foods.",
        ]
        professional = [
            "Persistent low energy may merit a clinical check-in or bloodwork discussion.",
        ]
    else:
        first_aid = ["Current state is balanced. Maintain your routine."]
        lifestyle = ["Keep a short mindfulness or light meditation practice daily."]
        professional = ["Use routine health check-ins to stay ahead of changes."]

    emotion_tips = {
        "angry": "Anger spike detected. Try a cold-water reset and pause before responding.",
        "sad": "Sadness noted. Reach out to someone you trust today.",
        "fear": "Visible anxiety detected. Try 4-7-8 breathing for one minute.",
        "disgust": "Negative affect detected. A brief walk or stretch can help reset attention.",
        "happy": "Positive emotional state detected. Anchor it with a short gratitude note.",
        "surprise": "Heightened arousal detected. Pause and reassess your surroundings.",
        "neutral": "Calm baseline detected. Good moment for focused work or planning.",
    }

    trend_messages = {
        "worsening": "Sensor forecast shows a worsening trajectory. Proactive intervention is recommended.",
        "improving": "Forecast shows improving physiological trends.",
        "stable": "Physiological forecast is stable over the next readings.",
    }

    return {
        "trend": trend,
        "trend_message": trend_messages[trend],
        "first_aid": first_aid,
        "emotion_tip": emotion_tips.get(normalized_emotion, ""),
        "lifestyle": lifestyle,
        "professional": professional,
    }

def load_engine_background():
    global engine, engine_loading, engine_error
    if engine is not None or engine_loading:
        return

    engine_loading = True
    engine_error = None
    try:
        from inference import engine as loaded_engine
        loaded_engine.load_models()
        engine = loaded_engine
    except Exception as exc:
        engine_error = str(exc)
        print(f"Model loading error: {exc}")
    finally:
        engine_loading = False

def fallback_sensor_prediction(vals: list[float]) -> dict:
    gsr = vals[0] if len(vals) > 0 else 0
    ppg = vals[1] if len(vals) > 1 else 0
    if gsr <= 0 and ppg <= 0:
        return {"state": "NO_SIGNAL", "confidence": 0.0, "probabilities": {}}
    if gsr > 30000 or ppg > 130:
        state = "PANIC_STATE"
    elif gsr > 20000 or ppg > 115:
        state = "HIGH_ANXIETY"
    elif gsr > 12000 or ppg > 100:
        state = "MODERATE_STRESS"
    elif gsr > 6000 or ppg > 88:
        state = "LOW_STRESS"
    else:
        state = "NORMAL"
    return {
        "state": state,
        "confidence": 0.62,
        "probabilities": {state: 0.62},
    }

def fallback_recommendations(state: str) -> list[str]:
    details = build_recommendation_details(state, "Neutral", [])
    return details.get("first_aid", [])

@app.on_event("startup")
def startup_event():
    threading.Thread(target=load_engine_background, daemon=True).start()
    arduino.start()
    eeg.start()

@app.on_event("shutdown")
def shutdown_event():
    arduino.stop()
    eeg.stop()

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))

@app.get("/health")
def health():
    return build_status_payload()

@app.get("/api/status")
def api_status():
    return build_status_payload()

@app.post("/api/register")
async def register(name: str = Form(...), phone: str = Form(""), email: str = Form("")):
    user = db.register_user(name, phone, email)
    return user

@app.get("/api/live_data")
def get_live_data(user_id: str = "", emotion: str = "Neutral", emotion_confidence: float = 0.0):
    vals = sensor_data.to_list()
    data_dict = sensor_data.to_dict()
    active_engine = engine
    
    # Run RNN + Predictor
    if active_engine is None:
        state_pred = fallback_sensor_prediction(vals)
        future_pred = []
        recs = fallback_recommendations(state_pred.get("state", "NORMAL"))
    else:
        state_pred = active_engine.predict_sensor(vals)
        future_pred = active_engine.predict_future(vals)
        recs = active_engine.get_recommendations(state_pred.get("state", "NORMAL"))
    future_states = [step.get("state", "") for step in future_pred]
    rec_details = build_recommendation_details(
        state_pred.get("state", "NORMAL"),
        emotion,
        future_states,
    )
    
    # We save sensor data in DB periodically, let's do it in the background if user_id is given
    if user_id:
        def _save():
            predictions = {
                "state": state_pred.get("state"),
                "state_confidence": state_pred.get("confidence"),
                "emotion": emotion,
                "emotion_confidence": emotion_confidence,
            }
            db.store_reading(user_id, data_dict, predictions)
        threading.Thread(target=_save, daemon=True).start()
    
    return {
        "sensors": data_dict,
        "state": state_pred,
        "future": future_pred,
        "recommendations": recs,
        "recommendation_details": rec_details,
    }

@app.post("/api/predict_facial")
async def predict_facial(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is not None and engine is not None:
        emo_pred = engine.predict_facial(frame)
        return emo_pred
    return {
        "emotion": "Neutral",
        "confidence": 0.0,
        "probabilities": {"Neutral": 0.0},
        "face_detected": False,
    }

if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=9000, reload=False)
