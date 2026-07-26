"""
MindSense AI — Late Fusion Model Training
===========================================
Combines sensor probabilities (20-dim) with facial emotion probabilities
projected into the mental-state space (7→20 via mapping matrix) to produce
unified mental health predictions.

Generates 10,000 synthetic paired samples using actual XGBoost predictions
for sensor probs and Dirichlet-sampled facial probs correlated with the
true mental state.
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.callbacks import EarlyStopping, TensorBoardLogger

logger = logging.getLogger(__name__)


# =========================================================================
# Mental State / Emotion Definitions
# =========================================================================

MENTAL_STATES = [
    'anxiety_high', 'anxiety_low', 'anxiety_moderate',
    'calm', 'cognitive_overload_high', 'cognitive_overload_low',
    'depression_high', 'depression_low', 'depression_moderate',
    'fatigue_high', 'fatigue_low', 'fatigue_moderate',
    'mixed_depression_fatigue', 'mixed_stress_anxiety',
    'normal', 'panic_state', 'relaxed',
    'stress_high', 'stress_low', 'stress_moderate',
]

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Mapping: mental_state → dominant facial emotion(s)
STATE_TO_DOMINANT_EMOTION = {
    'anxiety_high':              ['fear'],
    'anxiety_low':               ['neutral', 'fear'],
    'anxiety_moderate':          ['fear', 'surprise'],
    'calm':                      ['happy', 'neutral'],
    'cognitive_overload_high':   ['surprise', 'angry'],
    'cognitive_overload_low':    ['surprise', 'neutral'],
    'depression_high':           ['sad'],
    'depression_low':            ['sad', 'neutral'],
    'depression_moderate':       ['sad'],
    'fatigue_high':              ['sad', 'neutral'],
    'fatigue_low':               ['neutral'],
    'fatigue_moderate':          ['neutral', 'sad'],
    'mixed_depression_fatigue':  ['sad', 'neutral'],
    'mixed_stress_anxiety':      ['angry', 'fear'],
    'normal':                    ['happy', 'neutral'],
    'panic_state':               ['fear', 'surprise'],
    'relaxed':                   ['happy', 'neutral'],
    'stress_high':               ['angry'],
    'stress_low':                ['neutral', 'angry'],
    'stress_moderate':           ['angry', 'disgust'],
}


def build_emotion_mapping_matrix():
    """Build and row-normalize the 7×20 emotion→mental state mapping matrix.

    Returns
    -------
    M : ndarray (7, 20) — row-normalized
    """
    state_idx = {s: i for i, s in enumerate(MENTAL_STATES)}
    emo_idx = {e: i for i, e in enumerate(EMOTIONS)}

    M = np.zeros((7, 20), dtype=np.float64)

    # angry → stress_high(0.8), anxiety_moderate(0.6), stress_moderate(0.4)
    M[emo_idx['angry'], state_idx['stress_high']] = 0.8
    M[emo_idx['angry'], state_idx['anxiety_moderate']] = 0.6
    M[emo_idx['angry'], state_idx['stress_moderate']] = 0.4

    # disgust → stress_moderate(0.5), depression_low(0.4), anxiety_low(0.3)
    M[emo_idx['disgust'], state_idx['stress_moderate']] = 0.5
    M[emo_idx['disgust'], state_idx['depression_low']] = 0.4
    M[emo_idx['disgust'], state_idx['anxiety_low']] = 0.3

    # fear → anxiety_high(0.9), panic_state(0.7), anxiety_moderate(0.5)
    M[emo_idx['fear'], state_idx['anxiety_high']] = 0.9
    M[emo_idx['fear'], state_idx['panic_state']] = 0.7
    M[emo_idx['fear'], state_idx['anxiety_moderate']] = 0.5

    # happy → normal(0.9), relaxed(0.8), calm(0.7)
    M[emo_idx['happy'], state_idx['normal']] = 0.9
    M[emo_idx['happy'], state_idx['relaxed']] = 0.8
    M[emo_idx['happy'], state_idx['calm']] = 0.7

    # neutral → normal(0.6), calm(0.5), fatigue_low(0.3)
    M[emo_idx['neutral'], state_idx['normal']] = 0.6
    M[emo_idx['neutral'], state_idx['calm']] = 0.5
    M[emo_idx['neutral'], state_idx['fatigue_low']] = 0.3

    # sad → depression_high(0.8), depression_moderate(0.6), fatigue_high(0.4)
    M[emo_idx['sad'], state_idx['depression_high']] = 0.8
    M[emo_idx['sad'], state_idx['depression_moderate']] = 0.6
    M[emo_idx['sad'], state_idx['fatigue_high']] = 0.4

    # surprise → panic_state(0.5), anxiety_moderate(0.4),
    #            cognitive_overload_low(0.3)
    M[emo_idx['surprise'], state_idx['panic_state']] = 0.5
    M[emo_idx['surprise'], state_idx['anxiety_moderate']] = 0.4
    M[emo_idx['surprise'], state_idx['cognitive_overload_low']] = 0.3

    # Row-normalize
    row_sums = M.sum(axis=1, keepdims=True)
    M = M / np.maximum(row_sums, 1e-10)

    return M


# =========================================================================
# Fusion Network
# =========================================================================

class FusionNet(nn.Module):
    """Late fusion network: sensor_probs(20) + projected_facial(20) → 20 classes."""

    def __init__(self, mapping_matrix: np.ndarray):
        super().__init__()
        # Register mapping matrix as buffer (not a parameter)
        self.register_buffer(
            'M', torch.tensor(mapping_matrix, dtype=torch.float32)
        )

        self.layers = nn.Sequential(
            nn.Linear(40, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 20),
        )

    def forward(self, sensor_probs: torch.Tensor,
                facial_probs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sensor_probs : Tensor (batch, 20)
        facial_probs : Tensor (batch, 7)

        Returns
        -------
        logits : Tensor (batch, 20)
        """
        # Project facial probs to mental state space
        facial_projected = facial_probs @ self.M  # (batch, 20)
        x = torch.cat([sensor_probs, facial_projected], dim=1)  # (batch, 40)
        return self.layers(x)


# =========================================================================
# Synthetic Training Data Generation
# =========================================================================

def generate_fusion_data(xgb_model, X_val, y_val,
                         n_samples: int = 10000, seed: int = 42):
    """Generate synthetic paired sensor/facial probability samples.

    Parameters
    ----------
    xgb_model : trained XGBoost model
    X_val : ndarray — validation features
    y_val : ndarray — validation labels
    n_samples : int

    Returns
    -------
    sensor_probs : ndarray (n_samples, 20)
    facial_probs : ndarray (n_samples, 7)
    labels : ndarray (n_samples,)
    """
    rng = np.random.RandomState(seed)
    emo_idx = {e: i for i, e in enumerate(EMOTIONS)}

    # Use actual XGB predictions (cycle through val set if needed)
    n_val = len(X_val)
    indices = np.arange(n_samples) % n_val
    sensor_probs = xgb_model.predict_proba(X_val[indices])
    labels = y_val[indices].copy()

    # Generate correlated facial probs via Dirichlet
    facial_probs = np.zeros((n_samples, 7), dtype=np.float32)

    for i in range(n_samples):
        state_name = MENTAL_STATES[labels[i]]
        dominant_emotions = STATE_TO_DOMINANT_EMOTION.get(
            state_name, ['neutral']
        )

        # Build Dirichlet concentration vector
        alpha = np.ones(7) * 0.5  # low base concentration
        for emo in dominant_emotions:
            idx = emo_idx[emo]
            alpha[idx] += 5.0  # concentrate on dominant emotions

        facial_probs[i] = rng.dirichlet(alpha)

    logger.info(f"  Generated {n_samples} fusion training samples")
    return sensor_probs.astype(np.float32), facial_probs, labels


# =========================================================================
# Training
# =========================================================================

def train_fusion_model(config: dict, device: torch.device,
                       xgb_model=None, sensor_data: dict = None):
    """Train the late fusion network.

    Parameters
    ----------
    config : dict
    device : torch.device
    xgb_model : trained XGBoost model (for generating sensor probs)
    sensor_data : dict from SensorPreprocessor.run()

    Returns
    -------
    model : FusionNet
    """
    logger.info("=" * 60)
    logger.info("FUSION MODEL TRAINING")
    logger.info("=" * 60)

    fusion_cfg = config['fusion']
    models_dir = os.path.join(config['paths']['models_output_dir'],
                               'fusion_model')
    os.makedirs(models_dir, exist_ok=True)

    # Build and save mapping matrix
    M = build_emotion_mapping_matrix()
    np.save(os.path.join(models_dir, 'emotion_mapping_matrix.npy'), M)
    logger.info(f"  Emotion mapping matrix saved ({M.shape})")

    # Load XGB model if not provided
    if xgb_model is None:
        import xgboost as xgb_lib
        xgb_model = xgb_lib.XGBClassifier()
        xgb_path = os.path.join(config['paths']['models_output_dir'],
                                 'sensor_model', 'xgb_model.json')
        xgb_model.load_model(xgb_path)
        logger.info(f"  Loaded XGBoost from {xgb_path}")

    # Load sensor data if not provided
    if sensor_data is None:
        from preprocessing.sensor_preprocessor import SensorPreprocessor
        preprocessor = SensorPreprocessor(config)
        sensor_data = preprocessor.run()

    # Generate training data
    sensor_probs, facial_probs, labels = generate_fusion_data(
        xgb_model,
        sensor_data['X_val'],
        sensor_data['y_val'],
        n_samples=fusion_cfg['n_synthetic_samples'],
    )

    # Train/Val split
    n_train = int(0.8 * len(labels))
    train_ds = TensorDataset(
        torch.tensor(sensor_probs[:n_train], dtype=torch.float32),
        torch.tensor(facial_probs[:n_train], dtype=torch.float32),
        torch.tensor(labels[:n_train], dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(sensor_probs[n_train:], dtype=torch.float32),
        torch.tensor(facial_probs[n_train:], dtype=torch.float32),
        torch.tensor(labels[n_train:], dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds, batch_size=fusion_cfg['batch_size'],
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=fusion_cfg['batch_size'],
        shuffle=False, num_workers=0,
    )

    # Model
    model = FusionNet(M).to(device)
    logger.info(f"  FusionNet params: "
                 f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=fusion_cfg['learning_rate']
    )
    criterion = nn.CrossEntropyLoss()

    early_stop = EarlyStopping(patience=15, mode='min', restore_best=True)
    tb_logger = TensorBoardLogger(
        os.path.join(config['paths']['logs_dir'], 'fusion')
    )

    epochs = fusion_cfg['epochs']

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for sp, fp, y in train_loader:
            sp, fp, y = sp.to(device), fp.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(sp, fp)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sp.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += sp.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for sp, fp, y in val_loader:
                sp, fp, y = sp.to(device), fp.to(device), y.to(device)
                logits = model(sp, fp)
                loss = criterion(logits, y)
                val_loss += loss.item() * sp.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += sp.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        tb_logger.log_scalars('loss', {'train': train_loss, 'val': val_loss},
                               epoch)
        tb_logger.log_scalars('accuracy', {'train': train_acc, 'val': val_acc},
                               epoch)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

        early_stop(epoch, val_loss, model)
        if early_stop.should_stop:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    early_stop.restore(model)
    tb_logger.close()

    # Save
    torch.save(model.state_dict(),
               os.path.join(models_dir, 'fusion_net.pth'))
    fusion_config = {
        'input_dim': 40,
        'output_dim': 20,
        'architecture': '40->256->128->64->20',
        'emotion_classes': EMOTIONS,
        'mental_state_classes': MENTAL_STATES,
    }
    with open(os.path.join(models_dir, 'fusion_config.json'), 'w') as f:
        json.dump(fusion_config, f, indent=2)

    logger.info(f"  ✅ Fusion model saved → {models_dir}")
    return model


if __name__ == '__main__':
    import yaml
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    with open('configs/training_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_fusion_model(cfg, dev)
