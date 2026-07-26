"""
MindSense AI — Sensor Model Training
======================================
Trains a 3-model ensemble (XGBoost + DNN + Random Forest) on preprocessed
multi-sensor mental health data.

Models:
  A) XGBoost   — gradient-boosted trees, best for tabular mixed-type features
  B) SensorDNN — 4-block DNN with residual connections for non-linear EEG ratios
  C) RandomForest — uncorrelated errors complement gradient boosting

Ensemble: soft-voting with weights XGB=0.45, DNN=0.35, RF=0.20
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
from torch.cuda.amp import GradScaler, autocast
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.sensor_preprocessor import SensorPreprocessor
from utils.callbacks import EarlyStopping, TensorBoardLogger

logger = logging.getLogger(__name__)


# =========================================================================
# Model B: Deep Neural Network
# =========================================================================

class SensorDNN(nn.Module):
    """4-block DNN with residual connections for sensor-based classification.

    Architecture:
        Input → BatchNorm → Block1(512) → Block2(256) → Block3(128) → Block4(64) → Output(20)
    Residuals:
        Block1 → Block3 (via linear projection 512→128)
        Block2 → Block4 (via linear projection 256→64)
    """

    def __init__(self, n_features: int, n_classes: int = 20):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        # Input normalization
        self.input_bn = nn.BatchNorm1d(n_features)

        # Block 1: n_features → 512
        self.fc1 = nn.Linear(n_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)

        # Block 2: 512 → 256
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.25)

        # Block 3: 256 → 128 + residual from Block 1
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)
        self.res1_proj = nn.Linear(512, 128)  # project Block1 → Block3

        # Block 4: 128 → 64 + residual from Block 2
        self.fc4 = nn.Linear(128, 64)
        self.drop4 = nn.Dropout(0.15)
        self.res2_proj = nn.Linear(256, 64)   # project Block2 → Block4

        # Output
        self.fc_out = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.input_bn(x)

        # Block 1
        b1 = self.drop1(F.gelu(self.bn1(self.fc1(x))))

        # Block 2
        b2 = self.drop2(F.gelu(self.bn2(self.fc2(b1))))

        # Block 3 + residual from Block 1
        b3 = F.gelu(self.bn3(self.fc3(b2)))
        b3 = b3 + self.res1_proj(b1)  # residual connection
        b3 = self.drop3(b3)

        # Block 4 + residual from Block 2
        b4 = F.gelu(self.fc4(b3))
        b4 = b4 + self.res2_proj(b2)  # residual connection
        b4 = self.drop4(b4)

        return self.fc_out(b4)


# =========================================================================
# Sensor Ensemble
# =========================================================================

class SensorEnsemble:
    """Weighted soft-voting ensemble of XGBoost, DNN, and Random Forest."""

    def __init__(self, xgb_model, dnn_model, rf_model, device='cpu',
                 weights=None):
        self.xgb_model = xgb_model
        self.dnn_model = dnn_model
        self.rf_model = rf_model
        self.device = device
        self.weights = weights or {"xgb": 0.45, "dnn": 0.35, "rf": 0.20}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of predicted probabilities."""
        # XGBoost probabilities
        xgb_probs = self.xgb_model.predict_proba(X)

        # DNN probabilities
        self.dnn_model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            dnn_logits = self.dnn_model(x_tensor)
            dnn_probs = F.softmax(dnn_logits, dim=1).cpu().numpy()

        # Random Forest probabilities
        rf_probs = self.rf_model.predict_proba(X)

        # Weighted combination
        combined = (
            self.weights['xgb'] * xgb_probs
            + self.weights['dnn'] * dnn_probs
            + self.weights['rf'] * rf_probs
        )
        return combined

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# =========================================================================
# Training Functions
# =========================================================================

def train_xgboost(X_train, y_train, X_val, y_val, config, models_dir, device):
    """Train XGBoost classifier with early stopping and sample weights."""
    logger.info("─── Training Model A: XGBoost ───")

    sensor_cfg = config['sensor']['xgboost']
    use_gpu = (device.type == 'cuda')

    params = {
        'n_estimators': sensor_cfg['n_estimators'],
        'max_depth': sensor_cfg['max_depth'],
        'learning_rate': sensor_cfg['learning_rate'],
        'subsample': sensor_cfg['subsample'],
        'colsample_bytree': sensor_cfg['colsample_bytree'],
        'min_child_weight': sensor_cfg['min_child_weight'],
        'gamma': sensor_cfg['gamma'],
        'reg_alpha': sensor_cfg['reg_alpha'],
        'reg_lambda': sensor_cfg['reg_lambda'],
        'objective': 'multi:softprob',
        'num_class': 20,
        'eval_metric': 'mlogloss',
        'tree_method': 'gpu_hist' if use_gpu else 'hist',
        'use_label_encoder': False,
        'random_state': 42,
        'verbosity': 0,
    }

    # Handle newer XGBoost versions that may not support use_label_encoder
    try:
        model = xgb.XGBClassifier(**params)
    except TypeError:
        params.pop('use_label_encoder', None)
        model = xgb.XGBClassifier(**params)

    # Compute balanced sample weights
    sample_weights = compute_sample_weight('balanced', y_train)

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Save model
    model.save_model(os.path.join(models_dir, 'xgb_model.json'))
    logger.info(f"  XGBoost saved → xgb_model.json")

    # Save feature importance plot
    _plot_xgb_importance(model, models_dir)

    # Evaluate on validation set
    val_acc = (model.predict(X_val) == y_val).mean()
    logger.info(f"  XGBoost val accuracy: {val_acc:.4f}")

    return model, val_acc


def _plot_xgb_importance(model, models_dir):
    """Generate and save XGBoost feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(12, 8))
    importance = model.feature_importances_
    n_features = len(importance)

    # Try to load feature names
    feat_path = os.path.join(models_dir, 'feature_names.json')
    if os.path.exists(feat_path):
        with open(feat_path, 'r') as f:
            names = json.load(f)
    else:
        names = [f'f{i}' for i in range(n_features)]

    # Sort by importance
    indices = np.argsort(importance)[::-1][:30]  # top 30
    ax.barh(
        range(len(indices)),
        importance[indices],
        align='center',
        color='steelblue'
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Feature Importance')
    ax.set_title('XGBoost — Top Feature Importances')
    ax.invert_yaxis()
    plt.tight_layout()
    save_path = os.path.join(models_dir, 'xgb_feature_importance.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Feature importance plot saved -> {save_path}")


def train_dnn(X_train, y_train, X_val, y_val, config, models_dir, device):
    """Train the SensorDNN with mixed precision, cosine restarts, and
    early stopping."""
    logger.info("--- Training Model B: DNN (PyTorch) ---")

    dnn_cfg = config['sensor']['dnn']
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # Build model
    model = SensorDNN(n_features=n_features, n_classes=n_classes).to(device)
    logger.info(f"  DNN params: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights for loss
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=dnn_cfg['learning_rate'],
        weight_decay=dnn_cfg['weight_decay'],
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=dnn_cfg['T_0'],
        T_mult=dnn_cfg['T_mult'],
        eta_min=dnn_cfg['eta_min'],
    )

    # Mixed precision
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds, batch_size=dnn_cfg['batch_size'],
        shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=dnn_cfg['batch_size'],
        shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'),
    )

    # TensorBoard
    tb_logger = TensorBoardLogger(
        os.path.join(config['paths']['logs_dir'], 'dnn_sensor')
    )

    # Early stopping
    early_stop = EarlyStopping(
        patience=dnn_cfg['patience'], mode='min', restore_best=True
    )

    # Training loop
    epochs = dnn_cfg['epochs']
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"  DNN Epoch {epoch}/{epochs}",
                     leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=dnn_cfg['max_grad_norm']
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)

        # Logging
        tb_logger.log_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        tb_logger.log_scalars('accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        tb_logger.log_lr(optimizer, epoch)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d} — train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        early_stop(epoch, val_loss, model)
        if early_stop.should_stop:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights
    early_stop.restore(model)
    tb_logger.close()

    # Save model
    torch.save(model.state_dict(), os.path.join(models_dir, 'dnn_model.pth'))
    dnn_config = {
        'n_features': n_features,
        'n_classes': n_classes,
        'architecture': 'SensorDNN_4block_residual',
    }
    with open(os.path.join(models_dir, 'dnn_config.json'), 'w') as f:
        json.dump(dnn_config, f, indent=2)
    logger.info(f"  DNN saved → dnn_model.pth, dnn_config.json")
    logger.info(f"  DNN best val accuracy: {best_val_acc:.4f}")

    return model, best_val_acc


def train_random_forest(X_train, y_train, X_val, y_val, config, models_dir):
    """Train Random Forest classifier."""
    logger.info("─── Training Model C: Random Forest ───")

    rf_cfg = config['sensor']['random_forest']
    model = RandomForestClassifier(
        n_estimators=rf_cfg['n_estimators'],
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=rf_cfg['class_weight'],
        n_jobs=-1,
        random_state=42,
        oob_score=True,
    )

    logger.info("  Fitting Random Forest (this may take a while) …")
    model.fit(X_train, y_train)

    oob = model.oob_score_
    val_acc = (model.predict(X_val) == y_val).mean()
    logger.info(f"  OOB score:     {oob:.4f}")
    logger.info(f"  Val accuracy:  {val_acc:.4f}")

    joblib.dump(model, os.path.join(models_dir, 'rf_model.pkl'))
    logger.info(f"  Random Forest saved → rf_model.pkl")

    return model, val_acc


# =========================================================================
# Main entry point
# =========================================================================

def train_sensor_model(config: dict, device: torch.device):
    """Full sensor model training pipeline.

    Returns
    -------
    ensemble : SensorEnsemble
    data : dict from SensorPreprocessor.run()
    """
    models_dir = os.path.join(config['paths']['models_output_dir'], 'sensor_model')
    os.makedirs(models_dir, exist_ok=True)

    # ── Preprocessing ──
    preprocessor = SensorPreprocessor(config)
    data = preprocessor.run()

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    # ── Model A: XGBoost ──
    xgb_model, xgb_acc = train_xgboost(
        X_train, y_train, X_val, y_val, config, models_dir, device
    )

    # ── Model B: DNN ──
    dnn_model, dnn_acc = train_dnn(
        X_train, y_train, X_val, y_val, config, models_dir, device
    )

    # ── Model C: Random Forest ──
    rf_model, rf_acc = train_random_forest(
        X_train, y_train, X_val, y_val, config, models_dir
    )

    # ── Ensemble ──
    weights = config['sensor']['ensemble_weights']
    ensemble = SensorEnsemble(
        xgb_model, dnn_model, rf_model,
        device=device, weights=weights,
    )

    # Save ensemble weights
    with open(os.path.join(models_dir, 'ensemble_weights.json'), 'w') as f:
        json.dump(weights, f, indent=2)

    # Ensemble validation accuracy
    ens_preds = ensemble.predict(X_val)
    ens_acc = (ens_preds == y_val).mean()
    logger.info(f"\n  ╔══════════════════════════════════════╗")
    logger.info(f"  ║  SENSOR ENSEMBLE VAL ACCURACY: {ens_acc:.4f} ║")
    logger.info(f"  ╚══════════════════════════════════════╝")
    logger.info(f"  Individual: XGB={xgb_acc:.4f}  DNN={dnn_acc:.4f}  "
                 f"RF={rf_acc:.4f}")

    return ensemble, data


if __name__ == '__main__':
    import yaml
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    with open('configs/training_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_sensor_model(cfg, dev)
