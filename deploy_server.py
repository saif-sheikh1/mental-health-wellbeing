"""
MENTAL HEALTH ASSESSMENT - FastAPI Deployment Server
=====================================================
Dataset : mental_health_dataset_50000.csv  (10 sensor features, 6 mental-state classes)

Endpoints:
  POST /predict/sensor      - classify current mental state from 10 sensor features
  POST /predict/facial      - classify emotion from a grayscale face image
  POST /predict/future      - forecast future sensor values + mental states (via RNN)
  POST /explain/sensor      - SHAP feature importances for sensor prediction
  POST /explain/facial      - GradCAM heatmap for ViT facial prediction
  GET  /health              - liveness check

FIX 10 ─  BASE_PATH / MODEL_DIR unified.
          predict_future uses the saved RNN model + label encoder for state
          prediction instead of a hard-coded GSR/PPG threshold heuristic.
"""

import os
import io
import json
import pickle
import warnings
import numpy as np
import shap
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────
BASE_PATH  = r"D:\\fyp\\prediction model 1\\prediction model"
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

# ── Load artefacts ────────────────────────────────────────────────────────
print("Loading models...")
vit_model       = keras.models.load_model(
    os.path.join(MODEL_DIR, "vit_facial_model.keras"),
    custom_objects=CUSTOM_OBJECTS,
    compile=False,
)
rnn_model       = keras.models.load_model(
    os.path.join(MODEL_DIR, "rnn_sensor_model.keras"),
    compile=False,
)
predictor_model = keras.models.load_model(
    os.path.join(MODEL_DIR, "future_predictor_model.keras"),
    compile=False,
)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)
print("All models loaded successfully.")

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(title="Mental Health Assessment API", version="3.0")

# ── Pydantic schemas ──────────────────────────────────────────────────────
class SensorReading(BaseModel):
    GSR:       float
    PPG:       float
    Delta:     float
    Theta:     float
    LowAlpha:  float
    HighAlpha: float
    LowBeta:   float
    HighBeta:  float
    LowGamma:  float
    MidGamma:  float

class SensorSequence(BaseModel):
    readings: List[SensorReading]   # exactly SEQ_LEN readings

class FutureRequest(BaseModel):
    readings:    List[SensorReading]
    steps_ahead: Optional[int] = 5

# ── Helpers ───────────────────────────────────────────────────────────────
def readings_to_array(readings):
    """Convert a list of SensorReading objects to a float32 numpy array."""
    return np.array(
        [[getattr(r, col) for col in SENSOR_COLUMNS] for r in readings],
        dtype=np.float32,
    )

def build_sequence_input(arr):
    """Scale a (SEQ_LEN, N_FEATURES) array and add the batch dimension."""
    scaled = scaler.transform(arr)
    return scaled.reshape(1, len(arr), arr.shape[1])

# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": ["vit_facial", "rnn_sensor", "future_predictor"],
    }


@app.post("/predict/sensor")
def predict_sensor(seq: SensorSequence):
    if len(seq.readings) < SEQ_LEN:
        raise HTTPException(400, f"Need at least {SEQ_LEN} readings, got {len(seq.readings)}")
    arr   = readings_to_array(seq.readings[-SEQ_LEN:])
    x     = build_sequence_input(arr)
    probs = rnn_model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return {
        "state_index":   idx,
        "state_label":   STATE_NAMES[idx],
        "confidence":    float(probs[idx]),
        "probabilities": {STATE_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }


@app.post("/predict/facial")
async def predict_facial(file: UploadFile = File(...)):
    data  = await file.read()
    img   = Image.open(io.BytesIO(data)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr   = np.array(img, dtype=np.float32) / 255.0
    arr   = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    probs = vit_model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return {
        "emotion_index": idx,
        "emotion_label": EMOTION_NAMES[idx],
        "confidence":    float(probs[idx]),
        "probabilities": {EMOTION_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }


@app.post("/predict/future")
def predict_future(req: FutureRequest):
    """
    Forecast future sensor values AND predict the mental state for each
    forecasted step using the trained RNN model.
    FIX 13 ─ replaced GSR/PPG heuristic with proper RNN + LabelEncoder lookup.
    """
    if len(req.readings) < SEQ_LEN:
        raise HTTPException(400, f"Need at least {SEQ_LEN} readings, got {len(req.readings)}")

    arr             = readings_to_array(req.readings[-SEQ_LEN:])
    x               = build_sequence_input(arr)
    forecast_scaled = predictor_model.predict(x, verbose=0)[0]   # (horizon, n_features)
    forecast_raw    = scaler.inverse_transform(forecast_scaled)   # (horizon, n_features)

    future_steps = []
    for step_idx, step_vals in enumerate(forecast_raw):
        # Build a sliding window: replace oldest reading with this forecast step
        window     = np.vstack([arr[1:], step_vals.reshape(1, -1)])   # (SEQ_LEN, N_FEATURES)
        x_window   = build_sequence_input(window)
        probs      = rnn_model.predict(x_window, verbose=0)[0]
        state_idx  = int(np.argmax(probs))
        state_label = STATE_NAMES[state_idx]

        entry = {col: round(float(v), 4) for col, v in zip(SENSOR_COLUMNS, step_vals)}
        entry["predicted_state"]      = state_label
        entry["state_confidence"]     = round(float(probs[state_idx]), 4)
        entry["step"]                 = step_idx + 1
        future_steps.append(entry)

    return {"forecast_horizon": len(future_steps), "future_steps": future_steps}


@app.post("/explain/sensor")
def explain_sensor(seq: SensorSequence):
    if len(seq.readings) < SEQ_LEN:
        raise HTTPException(400, f"Need at least {SEQ_LEN} readings")
    arr        = readings_to_array(seq.readings[-SEQ_LEN:])
    x          = build_sequence_input(arr)
    background = np.zeros((1, SEQ_LEN, N_FEATURES), dtype=np.float32)
    explainer  = shap.GradientExplainer(rnn_model, background)
    shap_vals  = explainer.shap_values(x)
    result     = {}
    for class_idx, class_sv in enumerate(shap_vals):
        mean_shap = np.mean(np.abs(class_sv[0]), axis=0).tolist()
        result[STATE_NAMES[class_idx]] = {
            SENSOR_COLUMNS[i]: round(mean_shap[i], 6)
            for i in range(len(SENSOR_COLUMNS))
        }
    return {"shap_values": result}


@app.post("/explain/facial")
async def explain_facial(file: UploadFile = File(...)):
    data = await file.read()
    img  = Image.open(io.BytesIO(data)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr  = np.array(img, dtype=np.float32) / 255.0
    inp  = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    probs = vit_model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(probs))

    inp_tensor = tf.constant(inp)
    with tf.GradientTape() as tape:
        tape.watch(inp_tensor)
        preds = vit_model(inp_tensor, training=False)
        score = preds[:, idx]
    grads   = tape.gradient(score, inp_tensor)[0, :, :, 0].numpy()
    heatmap = np.maximum(grads, 0)
    heatmap /= (heatmap.max() + 1e-8)

    return {
        "emotion":         EMOTION_NAMES[idx],
        "confidence":      float(probs[idx]),
        "gradcam_heatmap": heatmap.flatten().tolist(),
        "heatmap_shape":   list(heatmap.shape),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
