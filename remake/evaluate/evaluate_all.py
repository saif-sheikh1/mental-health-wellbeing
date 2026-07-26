"""
MindSense AI — Full Evaluation Pipeline
=========================================
Evaluates all trained models and generates comprehensive reports:
  1. Sensor Ensemble — accuracy, F1, ROC-AUC, Cohen's Kappa, confusion matrix
  2. Facial Model   — Top-1/Top-2 accuracy with TTA, confusion matrix
  3. Future Predictor — step-wise accuracy at t+1 through t+5
  4. Summary HTML report
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import (
    plot_confusion_matrix, compute_sensor_metrics, compute_facial_metrics,
    compute_stepwise_accuracy, save_metrics_json,
)
from utils.augmentation import TestTimeAugmentation, get_val_transforms
from training.train_sensor_model import SensorDNN, SensorEnsemble
from training.train_facial_model import build_facial_model
from training.train_future_predictor import (
    Seq2SeqPredictor, generate_synthetic_sequences,
    build_transition_matrix, MENTAL_STATES,
)
from training.train_fusion_model import FusionNet, build_emotion_mapping_matrix

logger = logging.getLogger(__name__)


# =========================================================================
# Sensor Ensemble Evaluation
# =========================================================================

def evaluate_sensor(config: dict, device: torch.device, sensor_data: dict):
    """Evaluate the sensor ensemble on test data."""
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING SENSOR ENSEMBLE")
    logger.info("=" * 60)

    models_dir = os.path.join(config['paths']['models_output_dir'],
                               'sensor_model')
    X_test = sensor_data['X_test']
    y_test = sensor_data['y_test']
    label_encoder = sensor_data['label_encoder']
    class_names = list(label_encoder.classes_)

    # Load XGBoost
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(models_dir, 'xgb_model.json'))

    # Load DNN
    with open(os.path.join(models_dir, 'dnn_config.json'), 'r') as f:
        dnn_cfg = json.load(f)
    dnn_model = SensorDNN(
        n_features=dnn_cfg['n_features'],
        n_classes=dnn_cfg['n_classes'],
    ).to(device)
    dnn_model.load_state_dict(
        torch.load(os.path.join(models_dir, 'dnn_model.pth'),
                    map_location=device, weights_only=True)
    )
    dnn_model.eval()

    # Load Random Forest
    rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))

    # Load ensemble weights
    with open(os.path.join(models_dir, 'ensemble_weights.json'), 'r') as f:
        weights = json.load(f)

    ensemble = SensorEnsemble(xgb_model, dnn_model, rf_model,
                               device=device, weights=weights)

    # Predictions
    y_proba = ensemble.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)

    # Metrics
    metrics = compute_sensor_metrics(y_test, y_pred, y_proba, class_names)

    # Confusion matrix
    cm_path = os.path.join(models_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, class_names, cm_path)

    # Save report
    save_metrics_json(metrics,
                       os.path.join(models_dir, 'evaluation_report.json'))

    return metrics


# =========================================================================
# Facial Model Evaluation (with TTA)
# =========================================================================

def evaluate_facial(config: dict, device: torch.device):
    """Evaluate the facial emotion model with 5-view TTA."""
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING FACIAL MODEL (with TTA)")
    logger.info("=" * 60)

    models_dir = os.path.join(config['paths']['models_output_dir'],
                               'facial_model')

    # Load class names
    with open(os.path.join(models_dir, 'facial_class_names.json'), 'r') as f:
        class_names = json.load(f)

    # Load model
    model = build_facial_model(num_classes=len(class_names))
    model.load_state_dict(
        torch.load(os.path.join(models_dir, 'efficientnet_b2_facial.pth'),
                    map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    # Dataset (validation set, without augmentation for TTA)
    image_size = config['facial']['image_size']
    val_dir = os.path.join(config['paths']['facial_dataset_root'], 'test')

    # Use plain PIL loading for TTA
    from torchvision import transforms
    plain_transform = transforms.Compose([])  # no transform, keep PIL
    val_dataset = ImageFolder(val_dir, transform=None)

    tta = TestTimeAugmentation(image_size=image_size)

    all_preds = []
    all_probs = []
    all_labels = []

    logger.info(f"  Running TTA on {len(val_dataset)} validation images …")
    for idx in tqdm(range(len(val_dataset)), desc="  TTA Evaluation",
                     leave=False):
        img, label = val_dataset[idx]
        # img is a PIL Image (no transform applied)

        # Get 5 TTA views
        views = tta(img)

        # Predict
        avg_probs = TestTimeAugmentation.predict_with_tta(
            model, views, device=device
        )

        all_probs.append(avg_probs.cpu().numpy())
        all_preds.append(avg_probs.argmax().item())
        all_labels.append(label)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    # Metrics
    metrics = compute_facial_metrics(y_true, y_pred, y_proba, class_names)

    # Confusion matrix
    cm_path = os.path.join(models_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path,
                           figsize=(10, 8))

    # Save report
    save_metrics_json(metrics,
                       os.path.join(models_dir, 'evaluation_report.json'))

    return metrics


# =========================================================================
# Future Predictor Evaluation
# =========================================================================

def evaluate_future_predictor(config: dict, device: torch.device,
                               sensor_data: dict):
    """Evaluate the Seq2Seq future state predictor step by step."""
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING FUTURE STATE PREDICTOR")
    logger.info("=" * 60)

    models_dir = os.path.join(config['paths']['models_output_dir'],
                               'future_predictor')
    fp_cfg = config['future_predictor']

    # Load config
    with open(os.path.join(models_dir, 'predictor_config.json'), 'r') as f:
        pred_cfg = json.load(f)

    # Load model
    model = Seq2SeqPredictor(
        input_size=pred_cfg['input_size'],
        output_size=pred_cfg['output_size'],
        hidden_size=pred_cfg['hidden_size'],
        num_layers=pred_cfg['num_layers'],
        dropout=pred_cfg['dropout'],
        output_len=pred_cfg['output_len'],
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(models_dir, 'seq2seq_bilstm.pth'),
                    map_location=device, weights_only=True)
    )
    model.eval()

    # Generate test sequences
    transition_matrix = np.load(
        os.path.join(models_dir, 'transition_matrix.npy')
    )
    X_seq, Y_seq = generate_synthetic_sequences(
        sensor_data, transition_matrix,
        n_sequences=5000,  # smaller test set
        seq_len=pred_cfg['input_len'] + pred_cfg['output_len'],
        input_len=pred_cfg['input_len'],
        seed=123,  # different seed than training
    )

    # Predict
    all_preds = []
    batch_size = fp_cfg['batch_size']

    with torch.no_grad():
        for start in range(0, len(X_seq), batch_size):
            end = min(start + batch_size, len(X_seq))
            xb = torch.tensor(X_seq[start:end], dtype=torch.float32).to(device)
            logits = model(xb, teacher_forcing_ratio=0.0)
            preds = logits.argmax(dim=2).cpu().numpy()
            all_preds.append(preds)

    y_pred_seq = np.concatenate(all_preds, axis=0)

    # Step-wise accuracy
    metrics = compute_stepwise_accuracy(Y_seq, y_pred_seq,
                                         n_steps=pred_cfg['output_len'])

    # Save report
    save_metrics_json(metrics,
                       os.path.join(models_dir, 'evaluation_report.json'))

    return metrics


# =========================================================================
# Summary HTML Report
# =========================================================================

def generate_html_report(sensor_metrics, facial_metrics, future_metrics,
                          config):
    """Generate a summary HTML report of all evaluations."""
    models_dir = config['paths']['models_output_dir']
    html_path = os.path.join(models_dir, 'evaluation_summary.html')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MindSense AI — Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 960px; margin: 40px auto; padding: 0 20px;
            background: #0f0f23; color: #e0e0e0;
        }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff;
              padding-bottom: 10px; }}
        h2 {{ color: #ff6b6b; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #333; padding: 10px 14px; text-align: left; }}
        th {{ background: #1a1a3e; color: #00d4ff; }}
        td {{ background: #1a1a2e; }}
        .metric-value {{ font-weight: bold; color: #4ade80; font-size: 1.1em; }}
        .section {{ background: #151530; border-radius: 8px; padding: 20px;
                    margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }}
        .footer {{ text-align: center; margin-top: 40px; color: #666;
                   font-size: 0.85em; }}
    </style>
</head>
<body>
    <h1>🧠 MindSense AI — Evaluation Summary</h1>

    <div class="section">
    <h2>1. Sensor Ensemble</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Accuracy</td>
            <td class="metric-value">{sensor_metrics.get('accuracy', 'N/A'):.4f}</td></tr>
        <tr><td>Macro F1</td>
            <td class="metric-value">{sensor_metrics.get('macro_f1', 'N/A'):.4f}</td></tr>
        <tr><td>Weighted F1</td>
            <td class="metric-value">{sensor_metrics.get('weighted_f1', 'N/A'):.4f}</td></tr>
        <tr><td>Cohen's Kappa</td>
            <td class="metric-value">{sensor_metrics.get('cohen_kappa', 'N/A'):.4f}</td></tr>
        <tr><td>ROC-AUC (macro)</td>
            <td class="metric-value">{sensor_metrics.get('roc_auc_macro', 'N/A'):.4f}</td></tr>
    </table>
    </div>

    <div class="section">
    <h2>2. Facial Emotion Model (with TTA)</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Top-1 Accuracy</td>
            <td class="metric-value">{facial_metrics.get('top1_accuracy', 'N/A'):.4f}</td></tr>
        <tr><td>Top-2 Accuracy</td>
            <td class="metric-value">{facial_metrics.get('top2_accuracy', 'N/A'):.4f}</td></tr>
    </table>
    </div>

    <div class="section">
    <h2>3. Future State Predictor</h2>
    <table>
        <tr><th>Step</th><th>Accuracy</th></tr>"""

    for step in ['t+1', 't+2', 't+3', 't+4', 't+5']:
        val = future_metrics.get(step, 'N/A')
        val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
        html += f"""
        <tr><td>{step}</td>
            <td class="metric-value">{val_str}</td></tr>"""

    mean_acc = future_metrics.get('mean_accuracy', 'N/A')
    mean_str = f"{mean_acc:.4f}" if isinstance(mean_acc, (int, float)) else str(mean_acc)
    html += f"""
        <tr><td><strong>Mean</strong></td>
            <td class="metric-value"><strong>{mean_str}</strong></td></tr>
    </table>
    </div>

    <div class="footer">
        <p>Generated by MindSense AI Training Pipeline</p>
    </div>
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"\n  📄 HTML report saved → {html_path}")


# =========================================================================
# Main entry point
# =========================================================================

def evaluate_all(config: dict, device: torch.device, sensor_data: dict = None):
    """Run full evaluation pipeline for all models.

    Parameters
    ----------
    config : dict
    device : torch.device
    sensor_data : dict — output from SensorPreprocessor (optional, will
                  re-run if not provided)
    """
    logger.info("\n" + "╔" + "═" * 58 + "╗")
    logger.info("║" + " MINDSENSE AI — FULL EVALUATION ".center(58) + "║")
    logger.info("╚" + "═" * 58 + "╝")

    # Reload sensor data if not provided
    if sensor_data is None:
        from preprocessing.sensor_preprocessor import SensorPreprocessor
        preprocessor = SensorPreprocessor(config)
        sensor_data = preprocessor.run()

    # 1. Sensor Ensemble
    sensor_metrics = evaluate_sensor(config, device, sensor_data)

    # 2. Facial Model
    facial_models_dir = os.path.join(config['paths']['models_output_dir'],
                                      'facial_model')
    if os.path.exists(os.path.join(facial_models_dir,
                                    'efficientnet_b2_facial.pth')):
        facial_metrics = evaluate_facial(config, device)
    else:
        logger.warning("  Facial model not found — skipping evaluation.")
        facial_metrics = {'top1_accuracy': 0.0, 'top2_accuracy': 0.0}

    # 3. Future Predictor
    fp_models_dir = os.path.join(config['paths']['models_output_dir'],
                                  'future_predictor')
    if os.path.exists(os.path.join(fp_models_dir, 'seq2seq_bilstm.pth')):
        future_metrics = evaluate_future_predictor(config, device, sensor_data)
    else:
        logger.warning("  Future predictor not found — skipping evaluation.")
        future_metrics = {'mean_accuracy': 0.0}

    # 4. HTML Report
    generate_html_report(sensor_metrics, facial_metrics, future_metrics,
                          config)

    # Print summary
    logger.info("\n" + "═" * 60)
    logger.info("EVALUATION COMPLETE — SUMMARY")
    logger.info("═" * 60)
    logger.info(f"  Sensor Ensemble Accuracy:   "
                 f"{sensor_metrics.get('accuracy', 'N/A'):.4f}")
    logger.info(f"  Facial Top-1 Accuracy:      "
                 f"{facial_metrics.get('top1_accuracy', 'N/A'):.4f}")
    logger.info(f"  Future Predictor Mean Acc:   "
                 f"{future_metrics.get('mean_accuracy', 'N/A'):.4f}")

    return sensor_metrics, facial_metrics, future_metrics


if __name__ == '__main__':
    import yaml
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    with open('configs/training_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_all(cfg, dev)
