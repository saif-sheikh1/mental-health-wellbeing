"""
MindSense AI — Evaluation Metrics Utilities
=============================================
Confusion matrix plotting, classification reports, ROC-AUC, Cohen's Kappa,
and step-wise accuracy computation.
"""

import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
    top_k_accuracy_score,
)

logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path,
                          figsize=(20, 16), cmap='Blues'):
    """Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like of true labels (int)
    y_pred : array-like of predicted labels (int)
    class_names : list of str
    save_path : str — path to save the PNG
    figsize : tuple
    cmap : str — seaborn colormap
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Confusion matrix saved → {save_path}")


def compute_sensor_metrics(y_true, y_pred, y_proba, class_names):
    """Compute full evaluation metrics for the sensor ensemble.

    Returns
    -------
    dict with accuracy, macro_f1, weighted_f1, cohen_kappa,
    roc_auc_macro, classification_report (dict)
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)

    # ROC-AUC (one-vs-rest, macro)
    try:
        roc_auc = roc_auc_score(
            y_true, y_proba, multi_class='ovr', average='macro'
        )
    except ValueError:
        roc_auc = float('nan')
        logger.warning("  ROC-AUC could not be computed (missing classes?).")

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    metrics = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'cohen_kappa': float(kappa),
        'roc_auc_macro': float(roc_auc),
        'classification_report': report,
    }

    logger.info(f"  Accuracy:     {acc:.4f}")
    logger.info(f"  Macro-F1:     {macro_f1:.4f}")
    logger.info(f"  Weighted-F1:  {weighted_f1:.4f}")
    logger.info(f"  Cohen's κ:    {kappa:.4f}")
    logger.info(f"  ROC-AUC:      {roc_auc:.4f}")

    return metrics


def compute_facial_metrics(y_true, y_pred, y_proba, class_names):
    """Compute metrics for the facial emotion model.

    Returns
    -------
    dict with top1_accuracy, top2_accuracy, classification_report
    """
    top1 = accuracy_score(y_true, y_pred)

    try:
        top2 = top_k_accuracy_score(y_true, y_proba, k=2)
    except ValueError:
        top2 = float('nan')

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    metrics = {
        'top1_accuracy': float(top1),
        'top2_accuracy': float(top2),
        'classification_report': report,
    }

    logger.info(f"  Top-1 Accuracy: {top1:.4f}")
    logger.info(f"  Top-2 Accuracy: {top2:.4f}")

    return metrics


def compute_stepwise_accuracy(y_true_seq, y_pred_seq, n_steps: int = 5):
    """Compute step-wise accuracy for the future predictor.

    Parameters
    ----------
    y_true_seq : ndarray of shape (N, n_steps) — true labels per step
    y_pred_seq : ndarray of shape (N, n_steps) — predicted labels per step
    n_steps : int

    Returns
    -------
    dict mapping 't+k' → accuracy for k=1..n_steps, plus 'mean_accuracy'.
    """
    step_accs = {}
    for step in range(n_steps):
        acc = accuracy_score(y_true_seq[:, step], y_pred_seq[:, step])
        step_accs[f't+{step+1}'] = float(acc)
        logger.info(f"  Step t+{step+1} accuracy: {acc:.4f}")

    mean_acc = float(np.mean(list(step_accs.values())))
    step_accs['mean_accuracy'] = mean_acc
    logger.info(f"  Mean accuracy:    {mean_acc:.4f}")

    return step_accs


def save_metrics_json(metrics: dict, save_path: str):
    """Save metrics dictionary to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"  Metrics saved → {save_path}")
