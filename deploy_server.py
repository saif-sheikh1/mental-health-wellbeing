"""
MENTAL HEALTH ASSESSMENT - FastAPI Deployment Server
=====================================================
Dataset : mental_health_dataset_50000.csv  (10 sensor features, 6 mental-state classes)

Endpoints:
  POST /api/predict/sensor      - classify current mental state from 10 sensor features
  POST /api/predict/facial      - classify emotion from a grayscale face image
  POST /api/predict/future      - forecast future sensor values + mental states (via RNN)
  POST /api/explain/facial      - GradCAM heatmap for ViT facial prediction
  GET  /api/sensors/live        - Get live sensor readings
  POST /api/predict/complete    - Run full inference logic for the dashboard
  GET  /                        - Main UI (Login + Dashboard)
  GET  /health                  - liveness check
"""

import os
import io
import json
import pickle
import warnings
import collections
import asyncio
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image

from sensors import SensorData, ArduinoReader, EEGReader

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ── Config ───────────────────────────────────────────────────────────────
# Use directory relative to this script so it works cross-platform
BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_PATH, "models")
IMG_SIZE   = 48
SEQ_LEN    = 10

SENSOR_COLUMNS = [
    "GSR", "PPG",
    "Delta", "Theta",
    "LowAlpha", "HighAlpha",
    "LowBeta",  "HighBeta",
    "LowGamma", "MidGamma",
]
N_FEATURES = len(SENSOR_COLUMNS)

STATE_NAMES   = [
    "NORMAL", "LOW_STRESS", "MODERATE_STRESS",
    "HIGH_ANXIETY", "PANIC_STATE", "DEPRESSION",
]
EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ── Custom ViT layers (required to load saved model) ─────────────────────
class Patches(keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch   = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        dim = patches.shape[-1]
        return tf.reshape(patches, [batch, -1, dim])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        self.projection     = keras.layers.Dense(projection_dim)
        self.pos_embedding  = keras.layers.Embedding(num_patches, projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.pos_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_patches":    self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return cfg


CUSTOM_OBJECTS = {"Patches": Patches, "PatchEncoder": PatchEncoder}

# ── Globals for live sensors ──────────────────────────────────────────────
sensor_data = SensorData()
arduino_reader = ArduinoReader()
arduino_reader.data_ref = sensor_data
eeg_reader = EEGReader()
eeg_reader.data_ref = sensor_data

sensor_history = collections.deque(maxlen=SEQ_LEN)

# ── Load artefacts ────────────────────────────────────────────────────────
print("Loading models...")
try:
    vit_model = keras.models.load_model(
        os.path.join(MODEL_DIR, "vit_facial_model.keras"),
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )
except Exception as e:
    print(f"Warning: Failed to load vit_model. {e}")
    vit_model = None

try:
    rnn_model = keras.models.load_model(
        os.path.join(MODEL_DIR, "rnn_sensor_model.keras"),
        compile=False,
    )
except Exception as e:
    print(f"Warning: Failed to load rnn_model. {e}")
    rnn_model = None

try:
    predictor_model = keras.models.load_model(
        os.path.join(MODEL_DIR, "future_predictor_model.keras"),
        compile=False,
    )
except Exception as e:
    print(f"Warning: Failed to load predictor_model. {e}")
    predictor_model = None

try:
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Warning: Failed to load scaler. {e}")
    scaler = None

try:
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
except Exception:
    le = None

print("Models loading attempt finished.")

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(title="MindSense AI", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def sensor_history_loop():
    while True:
        sensor_history.append(sensor_data.to_list())
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup():
    arduino_reader.start()
    eeg_reader.start()
    asyncio.create_task(sensor_history_loop())

@app.on_event("shutdown")
def shutdown():
    arduino_reader.stop()
    eeg_reader.stop()

# ── Helpers ───────────────────────────────────────────────────────────────
def get_sensor_history_array() -> np.ndarray:
    if len(sensor_history) < SEQ_LEN:
        curr = sensor_data.to_list()
        history = list(sensor_history)
        while len(history) < SEQ_LEN:
            history.append(curr)
    else:
        history = list(sensor_history)
    return np.array(history, dtype=np.float32)

def build_sequence_input(arr):
    """Scale a (SEQ_LEN, N_FEATURES) array and add the batch dimension."""
    if scaler is None:
        return np.expand_dims(arr, axis=0)
    scaled = scaler.transform(arr)
    return scaled.reshape(1, len(arr), arr.shape[1])

def get_recommendations(state: str, emotion: str, future_states: List[str]) -> Dict:
    state_nums = [STATE_NAMES.index(s) if s in STATE_NAMES else 1 for s in future_states]
    trend = (
        "worsening" if (len(state_nums) >= 2 and state_nums[-1] > state_nums[0]) else
        "improving" if (len(state_nums) >= 2 and state_nums[-1] < state_nums[0]) else
        "stable"
    )

    first_aid, lifestyle, professional = [], [], []

    if state in ["HIGH_ANXIETY", "PANIC_STATE"]:
        first_aid = [
            "🫁 Box breathing: inhale 4s → hold 4s → exhale 4s → hold 4s. Repeat ×4",
            "🖐️ 5-4-3-2-1 grounding — name 5 things you see, 4 you feel, 3 you hear",
            "💧 Drink a glass of cold water slowly"
        ]
        lifestyle = [
            "📵 Set a screen-time limit of 30 min for the next 2 hours",
            "🛏️ Aim for 7–9 hours of sleep tonight",
        ]
        professional = [
            "🩺 If High Stress persists > 3 days, consult a licensed therapist",
        ]
    elif state in ["MODERATE_STRESS"]:
        first_aid = [
            "🚶 Step outside for 2–3 minutes if possible",
            "🎧 Put on calm music at low volume",
        ]
        lifestyle = [
            "🏃 30 min moderate exercise tomorrow morning",
            "📓 Journal your stressors for 5 minutes before bed",
        ]
        professional = [
            "📋 Consider a routine check-in with your counselor if needed"
        ]
    elif state in ["DEPRESSION", "LOW_STRESS"]:
        first_aid = [
            "💡 Increase ambient light (open blinds or turn on lights)",
            "🕺 Do 20 jumping jacks or a 2-min stretch",
            "☕ Hydrate — drink 250 ml water right now",
        ]
        lifestyle = [
            "🌅 Try a consistent wake-up time to regulate circadian rhythm",
            "🥗 Review diet — increase complex carbs and iron-rich foods",
        ]
        professional = [
            "🩺 Persistent low energy may indicate thyroid issues — consider bloodwork",
        ]
    else:
        first_aid = ["✅ Current state is balanced — maintain your routine!"]
        lifestyle = [
            "🧘 Keep up mindfulness or light meditation daily",
        ]
        professional = ["💬 Routine check-in with your healthcare provider every 6 months"]

    emotion_tips = {
        "angry":    "😤 Anger spike detected — try splashing cold water on your face",
        "sad":      "💙 Sadness noted — reach out to a trusted friend or family member",
        "fear":     "🛡️ Anxiety visible — 4-7-8 breathing can calm the nervous system quickly",
        "disgust":  "😣 Negative affect detected — brief physical reset (walk/stretch) helps",
        "happy":    "😊 Great emotional state! Anchor this feeling with a short gratitude note",
        "surprise": "😲 Heightened arousal — pause and assess your environment",
        "neutral":  "😐 Calm baseline — good time for focused work or planning",
    }

    trend_msg = {
        "worsening": "⚠️ Sensor forecast shows a worsening trajectory — proactive intervention recommended",
        "improving": "📈 Good news: forecast shows improving physiological trends",
        "stable":    "📊 Physiological forecast is stable over the next 5 readings",
    }

    return {
        "trend":         trend,
        "trend_message": trend_msg[trend],
        "first_aid":     first_aid,
        "emotion_tip":   emotion_tips.get(emotion.lower(), ""),
        "lifestyle":     lifestyle,
        "professional":  professional,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "vit_facial": vit_model is not None,
            "rnn_sensor": rnn_model is not None,
            "future_predictor": predictor_model is not None
        }
    }

@app.get("/api/sensors/live")
def get_live_sensors():
    return sensor_data.to_dict()

@app.post("/api/predict/complete")
async def api_complete(file: Optional[UploadFile] = File(None)):
    # 1. Sensor Prediction
    sensor_res = {"state": "UNKNOWN", "confidence": 0.0, "probabilities": {}}
    arr = get_sensor_history_array()
    
    if rnn_model is not None:
        x = build_sequence_input(arr)
        probs = rnn_model.predict(x, verbose=0)[0]
        idx   = int(np.argmax(probs))
        sensor_res = {
            "state":   STATE_NAMES[idx],
            "state_index": idx,
            "confidence":    float(probs[idx]),
            "probabilities": {STATE_NAMES[i]: float(p) for i, p in enumerate(probs)},
        }
        
    # 2. Future Prediction
    future_res = []
    future_states = []
    if predictor_model is not None and rnn_model is not None:
        x = build_sequence_input(arr)
        forecast_scaled = predictor_model.predict(x, verbose=0)[0]
        if scaler:
            forecast_raw = scaler.inverse_transform(forecast_scaled)
        else:
            forecast_raw = forecast_scaled
            
        for step_idx, step_vals in enumerate(forecast_raw):
            window = np.vstack([arr[1:], step_vals.reshape(1, -1)])
            x_window = build_sequence_input(window)
            probs = rnn_model.predict(x_window, verbose=0)[0]
            s_idx = int(np.argmax(probs))
            s_label = STATE_NAMES[s_idx]
            
            entry = {col: round(float(v), 4) for col, v in zip(SENSOR_COLUMNS, step_vals)}
            entry["predicted_state"] = s_label
            entry["state_index"] = s_idx
            entry["state_confidence"] = round(float(probs[s_idx]), 4)
            entry["step"] = step_idx + 1
            future_res.append(entry)
            future_states.append(s_label)
            
    # 3. Facial Prediction & GradCAM
    facial_res = None
    gradcam_res = None
    if file and file.filename and vit_model is not None:
        content = await file.read()
        if content:
            try:
                # Preprocessing fix (added equalizeHist)
                img = Image.open(io.BytesIO(content)).convert("L").resize((IMG_SIZE, IMG_SIZE))
                img_arr = np.array(img, dtype=np.float32) / 255.0
                img_arr = cv2.equalizeHist((img_arr * 255).astype(np.uint8)).astype(np.float32) / 255.0
                inp = img_arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
                
                # Predict
                probs = vit_model.predict(inp, verbose=0)[0]
                f_idx = int(np.argmax(probs))
                facial_res = {
                    "emotion": EMOTION_NAMES[f_idx],
                    "confidence": float(probs[f_idx]),
                    "probabilities": {EMOTION_NAMES[i]: float(p) for i, p in enumerate(probs)}
                }
                
                # GradCAM
                inp_tensor = tf.constant(inp)
                with tf.GradientTape() as tape:
                    tape.watch(inp_tensor)
                    preds = vit_model(inp_tensor, training=False)
                    score = preds[:, f_idx]
                grads = tape.gradient(score, inp_tensor)[0, :, :, 0].numpy()
                heatmap = np.maximum(grads, 0)
                heatmap /= (heatmap.max() + 1e-8)
                
                gradcam_res = {
                    "heatmap": heatmap.flatten().tolist(),
                    "shape": list(heatmap.shape)
                }
            except Exception as e:
                print(f"Facial prediction error: {e}")

    emotion = facial_res["emotion"] if facial_res else "neutral"
    recs = get_recommendations(sensor_res.get("state", "UNKNOWN"), emotion, future_states)

    return {
        "sensor": sensor_res,
        "facial": facial_res,
        "future": future_res,
        "gradcam": gradcam_res,
        "recommendations": recs,
        "timestamp": datetime.now().isoformat(),
    }


# ── HTML UI ────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MindSense AI — Mental Health Assessment</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#0a0e1a;--paper:#f4f1eb;--teal:#00b4a6;--coral:#ff6b5b;
  --amber:#f5a623;--violet:#7c3aed;--green:#10b981;
  --card:#ffffff;--border:#e2ddd4;--muted:#6b7280;
  --shadow:0 4px 24px rgba(10,14,26,.08);
  --r:16px;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:'DM Sans',sans-serif;background:var(--paper);color:var(--ink);min-height:100vh}
header{
  background:var(--ink);color:var(--paper);
  padding:24px 40px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;
}
.logo{font-family:'Syne',sans-serif;font-weight:800;font-size:1.5rem;letter-spacing:-.02em}
.logo span{color:var(--teal)}
.status-chips{display:flex;gap:10px;flex-wrap:wrap}
.chip{
  display:flex;align-items:center;gap:6px;
  padding:5px 12px;border-radius:999px;font-size:.75rem;font-weight:500;
  background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12)
}
.dot{width:7px;height:7px;border-radius:50%;background:#ef4444}
.dot.ok{background:var(--teal)}
.dot.pulse{animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
main{max-width:1440px;margin:0 auto;padding:40px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}
@media(max-width:1100px){.grid-2{grid-template-columns:1fr}.grid-3{grid-template-columns:1fr 1fr}}
@media(max-width:680px){.grid-3{grid-template-columns:1fr}}
section{margin-bottom:36px}
.card{
  background:var(--card);border-radius:var(--r);
  border:1.5px solid var(--border);box-shadow:var(--shadow);overflow:hidden
}
.card-head{
  padding:20px 24px;border-bottom:1.5px solid var(--border);
  display:flex;align-items:center;gap:12px
}
.card-head h2{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700}
.card-icon{
  width:36px;height:36px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0
}
.ci-teal{background:#e0faf8}
.ci-coral{background:#fff0ee}
.ci-amber{background:#fef8ec}
.ci-violet{background:#f3f0ff}
.ci-green{background:#ecfdf5}
.card-body{padding:24px}
#video{width:100%;border-radius:10px;background:#0a0e1a;display:none;aspect-ratio:4/3;object-fit:cover}
#canvas{display:none}
#snap-preview{width:100%;border-radius:10px;display:none;border:2px solid var(--teal)}
.cam-overlay{
  aspect-ratio:4/3;background:var(--ink);border-radius:10px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:10px;color:var(--paper);font-size:.9rem;cursor:pointer
}
.cam-overlay svg{width:48px;opacity:.4}
.field{margin-bottom:18px}
.field label{
  display:flex;justify-content:space-between;align-items:center;
  font-size:.8rem;font-weight:500;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em
}
.field label span{color:var(--ink);font-size:.9rem;font-weight:600;text-transform:none;letter-spacing:0}
.btn{
  padding:12px 24px;border:none;border-radius:10px;
  font-family:'Syne',sans-serif;font-size:.9rem;font-weight:700;cursor:pointer;
  transition:all .2s;display:inline-flex;align-items:center;gap:8px
}
.btn-primary{background:var(--ink);color:var(--paper)}
.btn-primary:hover{background:#1e2640;transform:translateY(-1px)}
.btn-teal{background:var(--teal);color:white}
.btn-teal:hover{filter:brightness(1.1);transform:translateY(-1px)}
.btn-ghost{background:transparent;color:var(--muted);border:1.5px solid var(--border)}
.btn-ghost:hover{border-color:var(--teal);color:var(--teal)}
.btn:disabled{opacity:.4;pointer-events:none}
.btn-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
.btn-full{width:100%;justify-content:center;margin-top:8px}
.spinner-wrap{padding:60px;display:flex;flex-direction:column;align-items:center;gap:16px;display:none}
.spinner{width:40px;height:40px;border:3px solid var(--border);border-top-color:var(--teal);border-radius:50%;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.result-hero{
  display:flex;align-items:center;gap:20px;
  padding:24px;background:var(--ink);border-radius:12px;color:var(--paper);margin-bottom:20px
}
.hero-label{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;line-height:1.1}
.hero-sub{font-size:.85rem;opacity:.6;margin-top:4px}
.hero-conf{
  margin-left:auto;text-align:right;
  font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;color:var(--teal)
}
.hero-conf small{display:block;font-size:.7rem;font-family:'DM Sans',sans-serif;opacity:.6;margin-top:-2px}
.prob-row{display:flex;align-items:center;gap:12px;margin-bottom:10px;font-size:.85rem}
.prob-name{width:140px;flex-shrink:0;font-weight:500}
.prob-track{flex:1;height:8px;background:#f0ede7;border-radius:4px;overflow:hidden}
.prob-fill{height:100%;border-radius:4px;transition:width .8s cubic-bezier(.22,1,.36,1)}
.fill-teal{background:var(--teal)}
.fill-coral{background:var(--coral)}
.fill-amber{background:var(--amber)}
.fill-violet{background:var(--violet)}
.prob-pct{width:40px;text-align:right;font-weight:600;color:var(--muted)}
.sensor-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:16px}
@media(max-width:780px){.sensor-grid{grid-template-columns:repeat(3,1fr)}}
.sensor-tile{
  background:var(--paper);border-radius:12px;padding:14px;text-align:center;
  border:1.5px solid var(--border)
}
.sensor-tile .val{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:var(--ink)}
.sensor-tile .lbl{font-size:.72rem;color:var(--muted);margin-top:2px;font-weight:500}
.timeline{display:flex;gap:0;overflow-x:auto;padding-bottom:8px}
.tl-step{
  flex:1;min-width:140px;padding:16px 12px;text-align:center;
  border-right:1.5px solid var(--border);position:relative
}
.tl-step:last-child{border-right:none}
.tl-dot{width:12px;height:12px;border-radius:50%;margin:0 auto 8px;}
.dot-low{background:#3b82f6}
.dot-normal{background:var(--green)}
.dot-stress{background:var(--coral)}
.tl-state{font-family:'Syne',sans-serif;font-size:.75rem;font-weight:700}
.tl-vals{font-size:.7rem;color:var(--muted);margin-top:4px;line-height:1.6}
#gradcam-canvas{border-radius:10px;width:100%;display:none}
.rec-section{margin-bottom:20px}
.rec-label{
  font-family:'Syne',sans-serif;font-size:.75rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;
  display:flex;align-items:center;gap:6px
}
.rec-tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.7rem;font-weight:600;margin-left:6px}
.tag-red{background:#fee2e2;color:#991b1b}
.tag-green{background:#d1fae5;color:#065f46}
.tag-blue{background:#dbeafe;color:#1e40af}
.rec-list{list-style:none;display:flex;flex-direction:column;gap:8px}
.rec-list li{
  padding:10px 14px;border-radius:10px;font-size:.88rem;line-height:1.5;
  background:var(--paper);border-left:3px solid var(--border)
}
.rec-first-aid li{border-left-color:var(--coral)}
.rec-lifestyle li{border-left-color:var(--teal)}
.rec-professional li{border-left-color:var(--violet)}
.trend-badge{
  display:inline-flex;align-items:center;gap:8px;
  padding:10px 16px;border-radius:10px;font-size:.88rem;font-weight:500;margin-bottom:16px;width:100%
}
.trend-worsen{background:#fff0ee;color:#c0392b;border:1.5px solid #fca99b}
.trend-improve{background:#ecfdf5;color:#065f46;border:1.5px solid #6ee7b7}
.trend-stable{background:#f3f4f6;color:#374151;border:1.5px solid #d1d5db}
.sec-title{
  font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
  margin-bottom:20px;display:flex;align-items:center;gap:10px
}
.sec-title::after{content:'';flex:1;height:2px;background:var(--border)}
.hidden{display:none!important}
.fade-in{animation:fadeIn .5s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.notice{
  padding:12px 16px;border-radius:10px;font-size:.85rem;
  background:#fef8ec;border:1.5px solid #fcd34d;color:#92400e;margin-bottom:16px
}
.login-wrap {
    max-width: 400px;
    margin: 100px auto;
}
.login-card {
    background: var(--card);
    padding: 30px;
    border-radius: var(--r);
    border: 1.5px solid var(--border);
    box-shadow: var(--shadow);
    text-align: center;
}
.login-card input {
    width: 100%;
    padding: 12px;
    margin-bottom: 20px;
    border: 1.5px solid var(--border);
    border-radius: 8px;
    font-family: inherit;
    font-size: 1rem;
    outline: none;
}
.login-card input:focus {
    border-color: var(--teal);
}
</style>
</head>
<body>
<header>
  <div class="logo">Mind<span>Sense</span> AI</div>
  <div class="status-chips" id="status-chips">
    <div class="chip"><div class="dot" id="d-vit"></div>ViT Facial</div>
    <div class="chip"><div class="dot" id="d-rnn"></div>RNN Sensor</div>
    <div class="chip"><div class="dot" id="d-pred"></div>Predictor</div>
  </div>
</header>

<main id="login-view">
    <div class="login-wrap">
        <div class="login-card fade-in">
            <div class="card-icon ci-teal" style="margin: 0 auto 16px auto; width: 60px; height: 60px; font-size: 2rem;">👤</div>
            <h2 style="font-family:'Syne'; margin-bottom: 8px; font-size: 1.5rem;">Welcome Back</h2>
            <p style="color: var(--muted); font-size: 0.9rem; margin-bottom: 24px;">Please enter your name to access the dashboard.</p>
            <input type="text" id="username" placeholder="Your Name" />
            <button class="btn btn-primary btn-full" onclick="login()">Enter Dashboard</button>
        </div>
    </div>
</main>

<main id="dashboard-view" class="hidden">
<section>
  <div class="grid-2">
    <div class="card">
      <div class="card-head">
        <div class="card-icon ci-teal">📸</div>
        <h2>Facial Expression (Optional)</h2>
      </div>
      <div class="card-body">
        <div id="cam-placeholder" class="cam-overlay" onclick="startCam()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
            <circle cx="12" cy="13" r="4"/>
          </svg>
          <span>Click to enable webcam</span>
        </div>
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas"></canvas>
        <img id="snap-preview" alt="Captured frame">
        <div class="btn-row">
          <button class="btn btn-teal hidden" id="btn-capture" onclick="capture()">📸 Capture</button>
          <button class="btn btn-ghost hidden" id="btn-retake" onclick="retake()">↺ Retake</button>
          <button class="btn btn-ghost hidden" id="btn-stop-cam" onclick="stopCam()">✕ Stop camera</button>
        </div>
      </div>
    </div>
    
    <div class="card">
      <div class="card-head">
        <div class="card-icon ci-coral">📡</div>
        <h2>Live Physiological Sensors</h2>
      </div>
      <div class="card-body">
        <div class="notice">
            Sensors are being read automatically. If physical hardware is not connected, dummy values (normal/moderate stress) are simulated.
        </div>
        <div class="sensor-grid" id="live-sensor-tiles">
            <!-- Populated via JS -->
        </div>
        
        <button onclick="analyze(event)" class="btn btn-primary btn-full" style="margin-top:20px;">
          🔬 Run Full AI Assessment
        </button>
      </div>
    </div>
  </div>
</section>

<div id="results" class="hidden">
  <div class="spinner-wrap" id="spinner" style="display:none">
    <div class="spinner"></div>
    <span style="color:var(--muted);font-size:.9rem">Analysing…</span>
  </div>
  <div id="res-content" class="hidden">
    <section>
      <div class="sec-title">Current Assessment</div>
      <div class="grid-2">
        <div class="card fade-in" id="card-sensor">
          <div class="card-head"><div class="card-icon ci-coral">🧠</div><h2>Mental State</h2></div>
          <div class="card-body">
            <div class="result-hero">
              <div>
                <div class="hero-label" id="state-label">—</div>
                <div class="hero-sub">Sensor-based classification</div>
              </div>
              <div class="hero-conf"><span id="state-conf">—</span>%<small>confidence</small></div>
            </div>
            <div id="state-probs"></div>
            <div style="margin-top: 16px;">
                <h4 style="font-size:0.8rem;color:var(--muted);margin-bottom:8px">Readings at prediction time</h4>
                <div class="sensor-grid" id="sensor-tiles"></div>
            </div>
          </div>
        </div>
        <div class="card fade-in" id="card-facial">
          <div class="card-head"><div class="card-icon ci-teal">😊</div><h2>Facial Emotion</h2></div>
          <div class="card-body">
            <div id="no-facial" class="notice">No image captured — sensor-only assessment</div>
            <div id="yes-facial" class="hidden">
              <div class="result-hero">
                <div>
                  <div class="hero-label" id="emotion-label">—</div>
                  <div class="hero-sub">Vision Transformer (ViT)</div>
                </div>
                <div class="hero-conf"><span id="emotion-conf">—</span>%<small>confidence</small></div>
              </div>
              <div id="emotion-probs"></div>
            </div>
            <div id="gradcam-wrap" class="hidden" style="margin-top:16px">
              <p style="font-size:.78rem;color:var(--muted);margin-bottom:8px">GradCAM — attention heatmap</p>
              <canvas id="gradcam-canvas"></canvas>
            </div>
          </div>
        </div>
      </div>
    </section>
    <section>
      <div class="sec-title">Future Prediction</div>
      <div class="card fade-in">
        <div class="card-head"><div class="card-icon ci-amber">🔮</div><h2>5-Step Physiological Forecast (Seq2Seq LSTM)</h2></div>
        <div class="card-body">
          <div class="timeline" id="timeline"></div>
        </div>
      </div>
    </section>
    <section>
      <div class="sec-title">AI Recommendations</div>
      <div class="card fade-in">
        <div class="card-head"><div class="card-icon ci-green">💡</div><h2>Personalised Health Guidance</h2></div>
        <div class="card-body" id="rec-body"></div>
      </div>
    </section>
  </div>
</div>
</main>

<script>
// UI Management
function login() {
    const name = document.getElementById("username").value.trim();
    if (!name) return alert("Please enter your name");
    document.getElementById("login-view").classList.add("hidden");
    document.getElementById("dashboard-view").classList.remove("hidden");
    startLiveSensors();
}

// Camera Management
let stream=null,blob=null;
function startCam(){
  navigator.mediaDevices.getUserMedia({video:{width:640,height:480}}).then(s=>{
    stream=s;
    const v=document.getElementById('video');
    v.srcObject=s;v.style.display='block';
    document.getElementById('cam-placeholder').classList.add('hidden');
    show('btn-capture');show('btn-stop-cam');
  }).catch(e=>alert('Camera error: '+e.message));
}
function stopCam(){
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  document.getElementById('video').style.display='none';
  hide('btn-capture');hide('btn-stop-cam');
  document.getElementById('cam-placeholder').classList.remove('hidden');
}
function capture(){
  const v=document.getElementById('video');
  const c=document.getElementById('canvas');
  c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  c.toBlob(b=>{
    blob=b;
    const img=document.getElementById('snap-preview');
    img.src=URL.createObjectURL(b);img.style.display='block';
    v.style.display='none';
    hide('btn-capture');hide('btn-stop-cam');show('btn-retake');
    if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  },'image/jpeg',.95);
}
function retake(){
  blob=null;
  document.getElementById('snap-preview').style.display='none';
  hide('btn-retake');
  document.getElementById('cam-placeholder').classList.remove('hidden');
}

// Live Sensors Sync
let liveSensorData = {};
function startLiveSensors() {
    setInterval(async () => {
        try {
            const res = await fetch('/api/sensors/live');
            if (res.ok) {
                liveSensorData = await res.json();
                renderLiveSensors();
            }
        } catch (e) {}
    }, 1000);
}

function renderLiveSensors() {
    const el = document.getElementById('live-sensor-tiles');
    const keys = ["gsr", "ppg", "delta", "theta", "low_alpha", "high_alpha", "low_beta", "high_beta", "low_gamma", "mid_gamma"];
    const names = ["GSR", "PPG", "Delta", "Theta", "L-Alpha", "H-Alpha", "L-Beta", "H-Beta", "L-Gamma", "M-Gamma"];
    
    el.innerHTML = keys.map((k, i) => {
        let val = liveSensorData[k] || 0;
        let displayVal = val.toFixed(1);
        if (k !== 'gsr' && k !== 'ppg') {
            displayVal = val.toFixed(3); // eeg bands are 0.0 - 1.0
        }
        return `<div class="sensor-tile"><div class="val">${displayVal}</div><div class="lbl">${names[i]}</div></div>`;
    }).join('');
}

// Utils
function show(id){document.getElementById(id)?.classList.remove('hidden')}
function hide(id){document.getElementById(id)?.classList.add('hidden')}
function pct(v){return(v*100).toFixed(1)}
const FILL_COLORS=['fill-teal','fill-coral','fill-amber','fill-violet','fill-teal','fill-coral','fill-amber'];

function probBars(containerId,probs,colors){
  const c=document.getElementById(containerId);
  c.innerHTML='';
  if(!probs) return;
  Object.entries(probs).sort((a,b)=>b[1]-a[1]).forEach(([k,v],i)=>{
    const col=colors?colors[i%colors.length]:FILL_COLORS[i%FILL_COLORS.length];
    c.innerHTML+=`<div class="prob-row">
      <span class="prob-name">${k}</span>
      <div class="prob-track"><div class="prob-fill ${col}" style="width:${pct(v)}%"></div></div>
      <span class="prob-pct">${pct(v)}%</span>
    </div>`;
  });
}

// App Status
async function pollStatus(){
  try{
    const d=await(await fetch('/health')).json();
    const map={vit_facial:'d-vit',rnn_sensor:'d-rnn',future_predictor:'d-pred'};
    for(const[k,id] of Object.entries(map)){
      const dot=document.getElementById(id);
      dot.className='dot pulse '+(d.models[k]?'ok':'');
    }
  }catch(e){}
}
pollStatus();setInterval(pollStatus,30000);

// Analysis
async function analyze(e){
  e.preventDefault();
  show('results');
  document.getElementById('spinner').style.display='flex';
  hide('res-content');
  
  const fd=new FormData();
  if(blob) fd.append('file',blob,'frame.jpg');
  
  try{
    const res=await fetch('/api/predict/complete',{method:'POST',body:fd});
    if(!res.ok){const err=await res.json();throw new Error(err.detail||'Server error');}
    const data=await res.json();
    renderResults(data);
  }catch(err){
    alert('Analysis failed: '+err.message);
  }finally{
    document.getElementById('spinner').style.display='none';
  }
}

function renderResults(d){
  const s=d.sensor;
  document.getElementById('state-label').textContent=s.state;
  document.getElementById('state-conf').textContent=pct(s.confidence);
  probBars('state-probs',s.probabilities,['fill-coral','fill-teal','fill-amber','fill-violet']);
  
  // Also show the values that were used
  const keys = ["gsr", "ppg", "delta", "theta", "low_alpha", "high_alpha", "low_beta", "high_beta", "low_gamma", "mid_gamma"];
  const names = ["GSR", "PPG", "Delta", "Theta", "L-Alpha", "H-Alpha", "L-Beta", "H-Beta", "L-Gamma", "M-Gamma"];
  document.getElementById('sensor-tiles').innerHTML=keys.map((k,i)=> {
        let val = liveSensorData[k] || 0;
        let displayVal = val.toFixed(1);
        if (k !== 'gsr' && k !== 'ppg') displayVal = val.toFixed(3);
        return `<div class="sensor-tile"><div class="val">${displayVal}</div><div class="lbl">${names[i]}</div></div>`;
  }).join('');
  
  if(d.facial){
    hide('no-facial');show('yes-facial');
    document.getElementById('emotion-label').textContent=d.facial.emotion;
    document.getElementById('emotion-conf').textContent=pct(d.facial.confidence);
    probBars('emotion-probs',d.facial.probabilities);
    if(d.gradcam&&d.gradcam.heatmap){show('gradcam-wrap');renderGradCam(d.gradcam.heatmap,d.gradcam.shape);}
  }else{
    show('no-facial');hide('yes-facial');hide('gradcam-wrap');
  }
  
  if(d.future) renderTimeline(d.future);
  if(d.recommendations) renderRec(d.recommendations);
  
  show('res-content');
  document.getElementById('results').scrollIntoView({behavior:'smooth',block:'start'});
}

function renderTimeline(steps){
  const el=document.getElementById('timeline');
  el.innerHTML='';
  steps.forEach(step=>{
    const dotClass=step.state_index<=1?'dot-low':step.state_index<=2?'dot-normal':'dot-stress';
    el.innerHTML+=`<div class="tl-step">
      <div class="tl-dot ${dotClass}"></div>
      <div class="tl-state" style="color:${step.state_index>2?'var(--coral)':step.state_index==0?'#3b82f6':'var(--green)'}">${step.predicted_state}</div>
      <div class="tl-vals">GSR ${step.GSR}<br>PPG ${step.PPG}<br>α(H) ${step.HighAlpha}<br>β(H) ${step.HighBeta}<br>γ(H) ${step.MidGamma}</div>
      <div style="font-size:.68rem;color:var(--muted);margin-top:6px">Step ${step.step}</div>
    </div>`;
  });
}

function renderGradCam(heatmap,shape){
  const canvas=document.getElementById('gradcam-canvas');
  canvas.style.display='block';
  const[H,W]=shape;
  canvas.width=W;canvas.height=H;
  const ctx=canvas.getContext('2d');
  const img=ctx.createImageData(W,H);
  for(let i=0;i<heatmap.length;i++){
    const v=heatmap[i];
    const r=Math.round(Math.min(255,v*2*255));
    const g=Math.round(Math.min(255,(1-Math.abs(v-.5)*2)*255));
    const b=Math.round(Math.min(255,(1-v)*2*255));
    img.data[i*4]=r;img.data[i*4+1]=g;img.data[i*4+2]=b;img.data[i*4+3]=180;
  }
  ctx.putImageData(img,0,0);
}

function renderRec(r){
  const body=document.getElementById('rec-body');
  const trendClass=r.trend==='worsening'?'trend-worsen':r.trend==='improving'?'trend-improve':'trend-stable';
  const trendIcon=r.trend==='worsening'?'📉':r.trend==='improving'?'📈':'📊';
  let html=`<div class="trend-badge ${trendClass}">${trendIcon} ${r.trend_message}</div>`;
  if(r.emotion_tip){
    html+=`<div style="padding:10px 14px;border-radius:10px;background:#f3f0ff;border-left:3px solid var(--violet);font-size:.88rem;margin-bottom:16px">${r.emotion_tip}</div>`;
  }
  html+=`<div class="rec-section">
    <div class="rec-label">🚨 First Aid <span class="rec-tag tag-red">Immediate</span></div>
    <ul class="rec-list rec-first-aid">${r.first_aid.map(i=>`<li>${i}</li>`).join('')}</ul>
  </div>`;
  html+=`<div class="rec-section">
    <div class="rec-label">🌿 Lifestyle <span class="rec-tag tag-green">Daily</span></div>
    <ul class="rec-list rec-lifestyle">${r.lifestyle.map(i=>`<li>${i}</li>`).join('')}</ul>
  </div>`;
  html+=`<div class="rec-section">
    <div class="rec-label">🩺 Professional <span class="rec-tag tag-blue">When needed</span></div>
    <ul class="rec-list rec-professional">${r.professional.map(i=>`<li>${i}</li>`).join('')}</ul>
  </div>`;
  body.innerHTML=html;
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  MindSense AI — Mental Health Assessment Server")
    print("═" * 60)
    print(f"  Models : {os.path.abspath(MODEL_DIR)}")
    print("  UI     : http://localhost:8000")
    print("  Docs   : http://localhost:8000/docs")
    print("═" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
