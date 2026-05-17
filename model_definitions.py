"""
model_definitions.py
====================
Single source of truth for ALL model architectures.
Both train_models.py and deploy_server.py import from here.

WHY THIS MATTERS
----------------
The root cause of "RNN model not loading" is that keras.models.load_model()
tries to reconstruct the graph from the saved JSON config.  Any layer that
uses a Python operator (+, *, etc.) between tensors, or any kwarg that
changed between TF versions (e.g. MultiHeadAttention dropout=), silently
corrupts the saved config and raises cryptic errors on load.

THE FIX
-------
We NEVER rely on load_model() to rebuild the graph.
Instead:
  1. Save only the weights:      model.save_weights("rnn.weights.h5")
  2. At load time, rebuild the   model = create_rnn_sensor_model(...)
     exact same graph in Python,  model.load_weights("rnn.weights.h5")
     then load the weights.

This approach is 100% immune to serialization bugs because no graph JSON
is ever written or read.  The architecture lives here in Python.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ── shared constants (must match training config) ────────────────────────────
IMG_SIZE          = 48
PATCH_SIZE        = 8
NUM_HEADS         = 4
TRANSFORMER_UNITS = [128, 64]
SEQUENCE_LEN      = 10
N_FEATURES        = 5
FORECAST_HORIZON  = 5
NUM_CLASSES       = 4          # Stable / Anxiety / Stress / Depression
NUM_EMOTIONS      = 7          # angry disgust fear happy neutral sad surprise
LEARNING_RATE     = 1e-4


# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOM LAYERS  (ViT only)
# ═══════════════════════════════════════════════════════════════════════════

class Patches(layers.Layer):
    """Extract non-overlapping patches from a batch of images."""

    def __init__(self, patch_size: int, **kw):
        super().__init__(**kw)
        self.patch_size = patch_size

    def call(self, images):
        b = tf.shape(images)[0]
        p = tf.image.extract_patches(
            images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        return tf.reshape(p, [b, -1, p.shape[-1]])

    def get_config(self):
        cfg = super().get_config()
        cfg["patch_size"] = self.patch_size
        return cfg


class PatchEncoder(layers.Layer):
    """Linear projection of patches + learnable positional embedding."""

    def __init__(self, num_patches: int, projection_dim: int, **kw):
        super().__init__(**kw)
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        self.projection     = layers.Dense(projection_dim)
        self.pos_embedding  = layers.Embedding(num_patches, projection_dim)

    def call(self, patches):
        pos = tf.range(0, self.num_patches, 1)
        return self.projection(patches) + self.pos_embedding(pos)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches,
                    "projection_dim": self.projection_dim})
        return cfg


# Registry passed to load_model() when loading the ViT .keras file.
# The RNN and Predictor do NOT need this.
VIT_CUSTOM_OBJECTS = {"Patches": Patches, "PatchEncoder": PatchEncoder}


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 1 — Vision Transformer  (facial emotion, 7 classes)
# ═══════════════════════════════════════════════════════════════════════════

def _mlp(x, units_list, dropout_rate):
    for u in units_list:
        x = layers.Dense(u, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_vit_model(num_classes: int = NUM_EMOTIONS) -> keras.Model:
    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    proj_dim    = 64

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="vit_input")
    x   = layers.Conv2D(3, 1, padding="same", name="patch_conv")(inp)
    x   = Patches(PATCH_SIZE, name="patches")(x)
    x   = PatchEncoder(num_patches, proj_dim, name="patch_encoder")(x)

    for i in range(6):
        x1   = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x)
        attn = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=proj_dim // NUM_HEADS,
            name=f"mha_{i}")(x1, x1)
        x2   = layers.Add(name=f"add1_{i}")([attn, x])
        x3   = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x2)
        x3   = _mlp(x3, TRANSFORMER_UNITS, 0.1)
        x    = layers.Add(name=f"add2_{i}")([x3, x2])

    x   = layers.LayerNormalization(epsilon=1e-6, name="ln_final")(x)
    x   = layers.Flatten(name="flatten")(x)
    x   = layers.Dropout(0.4, name="drop_head")(x)
    x   = _mlp(x, [256, 128], 0.3)
    out = layers.Dense(num_classes, activation="softmax", name="vit_output")(x)

    model = keras.Model(inp, out, name="ViT_Facial")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 2 — Bidirectional LSTM + GRU + Attention  (sensor, 4 classes)
#
#  KEY RULES (prevent serialization / load failures):
#   • NO dropout= kwarg in MultiHeadAttention
#   • NO inline tensor arithmetic (use layers.Add, layers.Multiply etc.)
#   • Every layer has a unique string name  →  weights load by name
#   • No Lambda layers anywhere
# ═══════════════════════════════════════════════════════════════════════════

def create_rnn_sensor_model(
    seq_len:     int = SEQUENCE_LEN,
    n_features:  int = N_FEATURES,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """
    Input  : (batch, seq_len, n_features)
    Output : (batch, num_classes)  — softmax probabilities
             [Stable, Anxiety, Stress, Depression]
    """
    inp = keras.Input(shape=(seq_len, n_features), name="rnn_input")

    # ── BiLSTM branch ────────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-3)),
        name="bilstm_1",
    )(inp)
    x = layers.BatchNormalization(name="bn_bilstm1")(x)
    x = layers.Dropout(0.3, name="drop_bilstm1")(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-3)),
        name="bilstm_2",
    )(x)
    x = layers.BatchNormalization(name="bn_bilstm2")(x)
    # x shape: (batch, seq_len, 128)

    # ── GRU branch ───────────────────────────────────────────────────────
    g = layers.GRU(64, return_sequences=True,
                   kernel_regularizer=regularizers.l2(1e-3),
                   name="gru_1")(inp)
    g = layers.BatchNormalization(name="bn_gru1")(g)
    g = layers.Dropout(0.3, name="drop_gru1")(g)
    # g shape: (batch, seq_len, 64)

    # ── Concatenate both branches ─────────────────────────────────────────
    c = layers.Concatenate(name="concat_branches")([x, g])
    # c shape: (batch, seq_len, 192)

    # ── Self-attention  (NO dropout kwarg!) ──────────────────────────────
    attn = layers.MultiHeadAttention(
        num_heads=4, key_dim=48, name="self_attn"
    )(c, c)
    # residual: use layers.Add(), NOT inline +
    attn = layers.Add(name="attn_residual")([attn, c])
    attn = layers.LayerNormalization(epsilon=1e-6, name="attn_ln")(attn)

    # ── Pooling + classifier ──────────────────────────────────────────────
    p = layers.GlobalAveragePooling1D(name="gap")(attn)

    p = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3), name="dense_1")(p)
    p = layers.BatchNormalization(name="bn_dense1")(p)
    p = layers.Dropout(0.4, name="drop_dense1")(p)

    p = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3), name="dense_2")(p)
    p = layers.Dropout(0.3, name="drop_dense2")(p)

    out = layers.Dense(num_classes, activation="softmax", name="rnn_output")(p)

    model = keras.Model(inp, out, name="RNN_Sensor")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 3 — Seq2Seq LSTM  (future sensor value forecasting)
#
#  Same serialization rules as RNN model.
# ═══════════════════════════════════════════════════════════════════════════

def create_predictor_model(
    seq_len:          int = SEQUENCE_LEN,
    n_features:       int = N_FEATURES,
    forecast_horizon: int = FORECAST_HORIZON,
) -> keras.Model:
    """
    Input  : (batch, seq_len, n_features)
    Output : (batch, forecast_horizon, n_features)
    """
    enc_inp = keras.Input(shape=(seq_len, n_features), name="pred_input")

    # ── Encoder ──────────────────────────────────────────────────────────
    e = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-3)),
        name="enc_bilstm1",
    )(enc_inp)
    e = layers.BatchNormalization(name="enc_bn1")(e)
    e = layers.Dropout(0.3, name="enc_drop1")(e)
    # e shape: (batch, seq_len, 256)

    # Second BiLSTM — return_state=True gives us the hidden states
    bi_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, return_state=True,
                    kernel_regularizer=regularizers.l2(1e-3)),
        name="enc_bilstm2",
    )(e)
    # bi_out = [output, fwd_h, fwd_c, bwd_h, bwd_c]
    # output is unused; we use the states
    fwd_h, fwd_c, bwd_h, bwd_c = bi_out[1], bi_out[2], bi_out[3], bi_out[4]
    state_h = layers.Concatenate(name="state_h")([fwd_h, bwd_h])  # (batch, 128)
    state_c = layers.Concatenate(name="state_c")([fwd_c, bwd_c])  # (batch, 128)

    # ── Decoder ──────────────────────────────────────────────────────────
    d = layers.RepeatVector(forecast_horizon, name="repeat")(state_h)
    # Decoder LSTM units must equal state dimension = 128
    d = layers.LSTM(128, return_sequences=True, name="dec_lstm1")(
        d, initial_state=[state_h, state_c])
    d = layers.BatchNormalization(name="dec_bn1")(d)
    d = layers.Dropout(0.3, name="dec_drop1")(d)
    d = layers.LSTM(64, return_sequences=True, name="dec_lstm2")(d)
    # d shape: (batch, forecast_horizon, 64)

    # ── Cross-attention: decoder queries encoder ──────────────────────────
    # Project e (256-wide) down to 64 to match d
    e_proj = layers.TimeDistributed(layers.Dense(64), name="enc_proj")(e)
    # (NO dropout kwarg!)
    ca = layers.MultiHeadAttention(
        num_heads=4, key_dim=16, name="cross_attn")(d, e_proj)
    # residual: use layers.Add()
    d = layers.Add(name="ca_residual")([d, ca])
    d = layers.LayerNormalization(epsilon=1e-6, name="ca_ln")(d)

    out = layers.TimeDistributed(layers.Dense(n_features), name="pred_output")(d)

    model = keras.Model(enc_inp, out, name="FuturePredictor")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model