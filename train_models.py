import os
import sys
import json
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import cv2

# ========================= CONFIGURATION =========================

BASE_PATH        = "/Users/fakhirbaig/Documents/GitHub/mental-health-wellbeing"
SENSOR_DATA_PATH = os.path.join(BASE_PATH, "mental_health_dataset_50000.csv")

FACIAL_PATHS = {
    'angry':    "/Users/fakhirbaig/Desktop/train/angry",
    'disgust':  "/Users/fakhirbaig/Desktop/train/disgust",
    'fear':     "/Users/fakhirbaig/Desktop/train/fear",
    'happy':    "/Users/fakhirbaig/Desktop/train/happy",
    'neutral':  "/Users/fakhirbaig/Desktop/train/neutral",
    'sad':      "/Users/fakhirbaig/Desktop/train/sad",
    'surprise': "/Users/fakhirbaig/Desktop/train/surprise",
}

MODEL_DIR = os.path.join(BASE_PATH, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyper-parameters - set to 3 epochs for extremely fast verification of architecture
VIT_EPOCHS        = 3
RNN_EPOCHS        = 3
PRED_EPOCHS       = 3
BATCH_SIZE        = 64
LEARNING_RATE     = 0.0002
IMG_SIZE          = 48
SEQUENCE_LEN      = 10
FORECAST_HORIZON  = 5

SENSOR_COLUMNS = [
    'GSR', 'PPG',
    'Delta', 'Theta',
    'LowAlpha', 'HighAlpha',
    'LowBeta',  'HighBeta',
    'LowGamma', 'MidGamma',
]
STATE_NAMES = [
    'NORMAL', 'LOW_STRESS', 'MODERATE_STRESS',
    'HIGH_ANXIETY', 'PANIC_STATE', 'DEPRESSION',
]
NUM_SENSOR_CLASSES = len(STATE_NAMES)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("MENTAL HEALTH ASSESSMENT -- QUICK MODEL TRAINING (macOS)")
print("=" * 80 + "\n")


# ======================== DATA LOADING ========================

def load_sensor_data():
    print("Loading sensor dataset...")
    if not os.path.exists(SENSOR_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {SENSOR_DATA_PATH}")

    df = None
    # 1. Try reading as Excel first because the file signature indicates Excel format
    try:
        df = pd.read_excel(SENSOR_DATA_PATH)
        print(f"  Loaded successfully as Excel: {len(df)} rows")
    except Exception as exc:
        print(f"  Excel loading failed/skipped: {exc}. Trying CSV reader...")
        
    # 2. Try reading as CSV with various encodings
    if df is None:
        for enc in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
            try:
                df = pd.read_csv(SENSOR_DATA_PATH, encoding=enc)
                print(f"  Loaded with CSV encoding={enc}: {len(df)} rows")
                break
            except Exception as e:
                continue

    if df is None:
        raise ValueError("Could not decode dataset with Excel or CSV parsers.")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    print(f"  Columns found: {list(df.columns)}")

    required = SENSOR_COLUMNS + ['Label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required)

    # Correct ranges for GSR and PPG to avoid filtering out actual values
    df = df[df['GSR'].between(0, 40952)]
    df = df[df['PPG'].between(40, 200)]
    for col in ['Delta', 'Theta', 'LowAlpha', 'HighAlpha',
                'LowBeta', 'HighBeta', 'LowGamma', 'MidGamma']:
        df = df[df[col] >= 0.0]

    df['Label'] = df['Label'].str.strip()

    print(f"  After cleaning: {len(df)} rows")
    print(f"  Class distribution:\n{df['Label'].value_counts().to_string()}\n")
    return df.reset_index(drop=True)


def encode_labels(df):
    le = LabelEncoder()
    all_labels = list(dict.fromkeys(STATE_NAMES + df['Label'].unique().tolist()))
    le.fit(all_labels)
    y = le.transform(df['Label'].values)
    return y, le


# ======================== SENSOR RNN MODEL ========================

def build_sequences(X, y, seq_len):
    if len(X) <= seq_len:
        raise ValueError(f"Dataset has {len(X)} rows; need > {seq_len}.")
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys)


def create_rnn_sensor_model(seq_len, n_features, num_classes):
    inputs = keras.Input(shape=(seq_len, n_features))

    # BiLSTM branch
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))
    )(x)
    x = layers.BatchNormalization()(x)

    # GRU branch
    g = layers.GRU(32, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(inputs)
    g = layers.BatchNormalization()(g)
    g = layers.Dropout(0.2)(g)

    combined = layers.Concatenate()([x, g])

    # Attention
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16, dropout=0.1)(combined, combined)
    attn = layers.LayerNormalization(epsilon=1e-6)(attn + combined)
    pooled = layers.GlobalAveragePooling1D()(attn)

    z = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(pooled)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)
    outputs = layers.Dense(num_classes, activation='softmax')(z)

    model = keras.Model(inputs, outputs, name='RNN_Sensor')
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_rnn_sensor():
    print("\n" + "=" * 80)
    print("STEP 1 -- SENSOR MODEL TRAINING")
    print("=" * 80 + "\n")

    df        = load_sensor_data()
    y_raw, le = encode_labels(df)
    X_raw     = df[SENSOR_COLUMNS].values.astype(np.float32)

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_seq = build_sequences(X_scaled, y_raw, SEQUENCE_LEN)
    n_samples, seq_len, n_feat = X_seq.shape
    print(f"  Sequences: {X_seq.shape}")

    # SMOTE
    X_flat    = X_seq.reshape(n_samples, seq_len * n_feat)
    counts    = np.bincount(y_seq)
    min_count = counts.min()

    if min_count < 2:
        print("  WARNING: Skipping SMOTE due to class size.")
        X_bal, y_bal = X_flat, y_seq
    else:
        k = min(5, min_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_bal, y_bal = smote.fit_resample(X_flat, y_seq)
        print(f"  After SMOTE: {len(X_bal)} samples")

    X_bal = X_bal.reshape(-1, seq_len, n_feat)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.15, random_state=42, stratify=y_bal
    )

    model = create_rnn_sensor_model(SEQUENCE_LEN, len(SENSOR_COLUMNS), NUM_SENSOR_CLASSES)

    # Compile with learning rate decay
    total_steps = (len(X_train) // BATCH_SIZE) * RNN_EPOCHS
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=total_steps,
        alpha=1e-5,
    )
    model.optimizer.learning_rate = lr_schedule

    callbacks = [
        ModelCheckpoint(os.path.join(MODEL_DIR, 'rnn_sensor_best.keras'), monitor='val_accuracy', save_best_only=True),
    ]

    print(f"  Training Sensor RNN for {RNN_EPOCHS} epochs...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=RNN_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(os.path.join(MODEL_DIR, 'rnn_sensor_model.keras'))
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    print(f"  Saved Sensor Model and Scalers.")
    return model, scaler, le


# ======================== FACIAL CNN MODEL ========================

def residual_block(x, filters, kernel_size=3, strides=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or strides != 1:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def create_facial_cnn(num_classes=7):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Conv2D(16, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 32)
    x = layers.MaxPooling2D(2)(x)
    
    x = residual_block(x, 64)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='Facial_ResNet')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_facial_cnn():
    print("\n" + "=" * 80)
    print("STEP 2 -- FACIAL MODEL TRAINING")
    print("=" * 80 + "\n")

    images, labels = [], []
    emotion_names  = list(FACIAL_PATHS.keys())
    for idx, (emotion, path) in enumerate(FACIAL_PATHS.items()):
        if not os.path.exists(path):
            print(f"  WARNING: Missing path: {path}")
            continue
        count = 0
        img_files = glob.glob(os.path.join(path, "*.*"))
        # Sample max 200 images per class for quick proof-of-concept training
        sampled_files = img_files[:200]
        for img_file in sampled_files:
            try:
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.equalizeHist(img)
                images.append(img.astype(np.float32) / 255.0)
                labels.append(idx)
                count += 1
            except Exception:
                continue
        print(f"  {emotion}: loaded {count} images")

    if len(images) < 20:
        print("  ERROR: Not enough images found – skipping facial training.")
        return None

    X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels)
    print(f"\n  Total dataset: {len(X)} images")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    model = create_facial_cnn(num_classes=len(FACIAL_PATHS))

    print(f"  Training Facial ResNet for {VIT_EPOCHS} epochs...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=VIT_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    save_path = os.path.join(MODEL_DIR, 'facial_cnn_model.keras')
    model.save(save_path)
    print(f"  Saved Facial CNN Model.")
    return model


# ======================== FUTURE PREDICTION MODEL ========================

def create_prediction_model(seq_len, n_features, forecast_horizon):
    encoder_inputs = keras.Input(shape=(seq_len, n_features), name='encoder_input')
    e = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))
    )(encoder_inputs)
    e = layers.BatchNormalization()(e)

    enc_out_full = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, return_state=True, kernel_regularizer=regularizers.l2(0.001))
    )(e)
    enc_seq = enc_out_full[0]
    fwd_h, fwd_c = enc_out_full[1], enc_out_full[2]
    bwd_h, bwd_c = enc_out_full[3], enc_out_full[4]

    state_h = layers.Concatenate()([fwd_h, bwd_h])
    state_c = layers.Concatenate()([fwd_c, bwd_c])

    decoder_seed = layers.RepeatVector(forecast_horizon)(state_h)
    d = layers.LSTM(64, return_sequences=True)(decoder_seed, initial_state=[state_h, state_c])
    d = layers.BatchNormalization()(d)

    e_proj = layers.TimeDistributed(layers.Dense(64))(enc_seq)
    attn   = layers.MultiHeadAttention(num_heads=2, key_dim=16)(d, e_proj)
    d      = layers.Add()([d, attn])
    d      = layers.LayerNormalization(epsilon=1e-6)(d)

    outputs = layers.TimeDistributed(layers.Dense(n_features))(d)
    model = keras.Model(encoder_inputs, outputs, name='FuturePredictor')
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss='mse', metrics=['mae'])
    return model


def train_future_prediction(scaler):
    print("\n" + "=" * 80)
    print("STEP 3 -- FUTURE PREDICTION MODEL TRAINING")
    print("=" * 80 + "\n")

    df       = load_sensor_data()
    X_scaled = scaler.transform(df[SENSOR_COLUMNS].values.astype(np.float32))
    n_feat   = len(SENSOR_COLUMNS)
    total    = SEQUENCE_LEN + FORECAST_HORIZON

    Xp, yp = [], []
    for i in range(len(X_scaled) - total):
        Xp.append(X_scaled[i:i + SEQUENCE_LEN])
        yp.append(X_scaled[i + SEQUENCE_LEN:i + total])

    Xp = np.array(Xp, dtype=np.float32)
    yp = np.array(yp, dtype=np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(Xp, yp, test_size=0.15, random_state=42)
    model = create_prediction_model(SEQUENCE_LEN, n_feat, FORECAST_HORIZON)

    print(f"  Training Future Seq2Seq for {PRED_EPOCHS} epochs...")
    model.fit(
        X_tr, y_tr, validation_data=(X_te, y_te),
        epochs=PRED_EPOCHS, batch_size=BATCH_SIZE, verbose=1
    )

    save_path = os.path.join(MODEL_DIR, 'future_predictor_model.keras')
    model.save(save_path)
    print(f"  Saved Future Predictor Model.")
    return model


# ======================== METADATA & STATE INFO ========================

def setup_xai(sensor_model, facial_model):
    print("\n" + "=" * 80)
    print("STEP 4 -- SETUP XAI INFO & METADATA")
    print("=" * 80 + "\n")

    xai_info = {
        'sensor_xai': {
            'type': 'SHAP GradientExplainer',
            'feature_names': SENSOR_COLUMNS,
            'sequence_len': SEQUENCE_LEN,
            'class_names': STATE_NAMES,
            'model_available': sensor_model is not None,
        },
        'facial_xai': {
            'type': 'GradCAM',
            'class_names': list(FACIAL_PATHS.keys()),
            'image_size': IMG_SIZE,
            'model_available': facial_model is not None,
        },
    }
    with open(os.path.join(MODEL_DIR, 'xai_info.json'), 'w', encoding='utf-8') as f:
        json.dump(xai_info, f, indent=2)

    meta = {
        'training_date':  datetime.now().isoformat(),
        'dataset':        'mental_health_dataset_50000.csv',
        'sensor_features': SENSOR_COLUMNS,
        'sensor_classes':  STATE_NAMES,
        'models': {
            'facial':    'facial_cnn_model.keras (Residual CNN)',
            'sensor':    'rnn_sensor_model.keras (BiLSTM+GRU+Attention)',
            'predictor': 'future_predictor_model.keras (Seq2Seq BiLSTM)',
        },
        'sequence_len':     SEQUENCE_LEN,
        'forecast_horizon': FORECAST_HORIZON,
    }
    with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(MODEL_DIR, 'state_info.pkl'), 'wb') as f:
        pickle.dump({
            'state_names': STATE_NAMES,
            'num_classes': NUM_SENSOR_CLASSES,
            'class_labels': list(range(NUM_SENSOR_CLASSES))
        }, f)
    print("  Saved XAI info and metadata successfully.")


# ======================== MAIN ========================

def main():
    start = datetime.now()
    rnn_model, scaler, le = train_rnn_sensor()
    facial_model = train_facial_cnn()
    train_future_prediction(scaler)
    setup_xai(rnn_model, facial_model)
    dur = (datetime.now() - start).total_seconds()
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE IN {dur:.1f} SECONDS!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
