import os
import pickle
import collections
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_PATH, "models")

IMG_SIZE = 48
SEQ_LEN = 10
FORECAST_HORIZON = 5
N_FEATURES = 10

SENSOR_COLUMNS = [
    "GSR", "PPG", "Delta", "Theta", 
    "LowAlpha", "HighAlpha", "LowBeta", "HighBeta", 
    "LowGamma", "MidGamma"
]

STATE_NAMES = [
    "NORMAL", "LOW_STRESS", "MODERATE_STRESS",
    "HIGH_ANXIETY", "PANIC_STATE", "DEPRESSION",
]

EMOTION_DISPLAY = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

class InferenceEngine:
    def __init__(self):
        self.rnn_model = None
        self.facial_model = None
        self.predictor_model = None
        self.scaler = None
        self.le = None
        self.face_cascades = []
        self.emotion_prob_history = collections.deque(maxlen=5)
        
        self.status = {
            "rnn": False,
            "facial": False,
            "predictor": False,
            "scaler": False
        }
        self._load_face_detectors()

    def _load_face_detectors(self):
        cascade_names = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
        ]
        for name in cascade_names:
            path = os.path.join(cv2.data.haarcascades, name)
            detector = cv2.CascadeClassifier(path)
            if not detector.empty():
                self.face_cascades.append(detector)

    def _largest_face(self, gray: np.ndarray):
        h, w = gray.shape[:2]
        min_size = max(36, min(h, w) // 8)
        for detector in self.face_cascades:
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=5,
                minSize=(min_size, min_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(faces):
                return max(faces, key=lambda box: box[2] * box[3])
        return None

    def _crop_face(self, gray: np.ndarray, face_box):
        h, w = gray.shape[:2]
        x, y, fw, fh = [int(v) for v in face_box]
        pad = int(max(fw, fh) * 0.22)
        cx, cy = x + fw // 2, y + fh // 2
        side = int(max(fw, fh) + pad * 2)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        x1 = max(0, x2 - side)
        y1 = max(0, y2 - side)
        return gray[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

    def _preprocess_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        face_box = self._largest_face(gray)
        if face_box is None:
            return None, None

        face, expanded_box = self._crop_face(gray, face_box)
        if face.size == 0:
            return None, None

        resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        resized = cv2.equalizeHist(resized)
        arr = resized.astype(np.float32) / 255.0
        return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1), expanded_box
        
    def load_models(self):
        print("Loading ML models...")
        
        # 1. Scaler
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            self.status["scaler"] = True
            print("  [OK] Scaler")
            
        # 2. Label Encoder
        le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        if os.path.exists(le_path):
            with open(le_path, "rb") as f:
                self.le = pickle.load(f)
            print("  [OK] Label Encoder")

        # 3. RNN Sensor Model
        rnn_path = os.path.join(MODEL_DIR, "rnn_sensor_model.keras")
        if not os.path.exists(rnn_path):
            rnn_path = os.path.join(MODEL_DIR, "rnn_sensor_best.keras")
        
        if os.path.exists(rnn_path):
            try:
                self.rnn_model = keras.models.load_model(rnn_path, compile=False)
                self.status["rnn"] = True
                print("  [OK] RNN Sensor Model")
            except Exception as e:
                print(f"  [FAIL] RNN Sensor Model: {e}")

        # 4. Facial CNN Model
        facial_path = os.path.join(MODEL_DIR, "facial_cnn_model.keras")
        if not os.path.exists(facial_path):
            facial_path = os.path.join(MODEL_DIR, "facial_cnn_best.keras")
            
        if os.path.exists(facial_path):
            try:
                self.facial_model = keras.models.load_model(facial_path, compile=False)
                self.status["facial"] = True
                print("  [OK] Facial CNN Model")
            except Exception as e:
                print(f"  [FAIL] Facial CNN Model: {e}")
                
        # 5. Future Predictor
        pred_path = os.path.join(MODEL_DIR, "future_predictor_model.keras")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(MODEL_DIR, "predictor_best.keras")
            
        if os.path.exists(pred_path):
            try:
                self.predictor_model = keras.models.load_model(pred_path, compile=False)
                self.status["predictor"] = True
                print("  [OK] Future Predictor")
            except Exception as e:
                print(f"  [FAIL] Future Predictor: {e}")
                
    def _vals_to_seq(self, vals: list[float]) -> np.ndarray:
        arr = np.array([vals] * SEQ_LEN, dtype=np.float32)
        if self.scaler:
            scaled = self.scaler.transform(arr)
        else:
            scaled = arr
        return scaled.reshape(1, SEQ_LEN, N_FEATURES)

    def predict_sensor(self, vals: list[float]) -> dict:
        if not self.status["rnn"] or not self.status["scaler"]:
            return {"state": "Unknown", "confidence": 0.0}
            
        seq = self._vals_to_seq(vals)
        probs = self.rnn_model.predict(seq, verbose=0)[0]
        idx = int(np.argmax(probs))
        
        state_name = self.le.classes_[idx] if self.le else STATE_NAMES[idx]
        
        return {
            "state": state_name,
            "confidence": float(probs[idx])
        }
        
    def predict_facial(self, frame) -> dict:
        if not self.status["facial"] or frame is None:
            return {
                "emotion": "Neutral",
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in EMOTION_DISPLAY},
                "face_detected": False,
            }

        arr, face_box = self._preprocess_face(frame)
        if arr is None:
            self.emotion_prob_history.clear()
            return {
                "emotion": "No face detected",
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in EMOTION_DISPLAY},
                "face_detected": False,
            }

        probs = self.facial_model.predict(arr, verbose=0)[0].astype(np.float32)
        probs = probs / max(float(probs.sum()), 1e-8)
        self.emotion_prob_history.append(probs)
        smoothed_probs = np.mean(np.array(self.emotion_prob_history), axis=0)
        smoothed_probs = smoothed_probs / max(float(smoothed_probs.sum()), 1e-8)
        idx = int(np.argmax(smoothed_probs))

        x, y, w, h = face_box
        margin = float(smoothed_probs[idx] - np.partition(smoothed_probs, -2)[-2])
        probabilities = {
            EMOTION_DISPLAY[i]: float(smoothed_probs[i])
            for i in range(len(EMOTION_DISPLAY))
        }

        return {
            "emotion": EMOTION_DISPLAY[idx],
            "confidence": float(smoothed_probs[idx]),
            "probabilities": probabilities,
            "face_detected": True,
            "face_box": {"x": x, "y": y, "w": w, "h": h},
            "margin": margin,
        }

    def predict_future(self, vals: list[float]) -> list:
        if not self.status["predictor"] or not self.status["scaler"]:
            return []
            
        seq = self._vals_to_seq(vals)
        fc_scaled = self.predictor_model.predict(seq, verbose=0)[0]
        fc_raw = self.scaler.inverse_transform(fc_scaled)
        
        out = []
        for i, raw in enumerate(fc_raw):
            gsr, ppg = raw[0], raw[1]
            if gsr > 30000 or ppg > 130: state = "PANIC_STATE"
            elif gsr > 20000 or ppg > 115: state = "HIGH_ANXIETY"
            elif gsr > 12000 or ppg > 100: state = "MODERATE_STRESS"
            elif gsr > 6000 or ppg > 88: state = "LOW_STRESS"
            elif gsr < 2000 and ppg < 60: state = "DEPRESSION"
            else: state = "NORMAL"
            
            entry = {"step": i + 1, "state": state}
            for fi, col in enumerate(SENSOR_COLUMNS):
                entry[col.lower()] = float(raw[fi])
            out.append(entry)
        return out

    def get_recommendations(self, state: str) -> dict:
        recs = {
            "PANIC_STATE": [
                "Find a safe, quiet space immediately.",
                "Practice 4-7-8 Breathing (inhale 4s, hold 7s, exhale 8s).",
                "Splash cold water on your face.",
                "Call a trusted person."
            ],
            "HIGH_ANXIETY": [
                "Box breathing: 4s in, 4s hold, 4s out, 4s hold.",
                "Listen to calm instrumental music.",
                "Step outside for a 5-minute slow walk."
            ],
            "MODERATE_STRESS": [
                "Take 3 deep diaphragmatic breaths.",
                "Drink 250 ml of water slowly.",
                "Do shoulder rolls and neck stretches."
            ],
            "LOW_STRESS": [
                "You are doing well, take a mindful moment to appreciate this.",
                "Try a 5-min gratitude meditation."
            ],
            "DEPRESSION": [
                "Get sunlight exposure for at least 15 minutes.",
                "Hydrate — drink water right now.",
                "Reach out to one person you trust today.",
                "Play upbeat music you enjoy."
            ],
            "NORMAL": [
                "You are in a balanced state — keep it up!",
                "Daily mindfulness or meditation (even 5 min helps)."
            ]
        }
        return recs.get(state, recs["NORMAL"])

engine = InferenceEngine()
