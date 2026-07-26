"""
MindSense AI — Facial Emotion Model Training
==============================================
Two-phase fine-tuning of EfficientNet-B2 (via timm) for 7-class facial
emotion recognition.

Phase 1: Feature extraction — train custom head only, backbone frozen.
Phase 2: Fine-tuning — unfreeze last 3 MBConv blocks with layer-wise LR,
         label smoothing, OneCycleLR, and MixUp augmentation.
"""

import os
import sys
import json
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import timm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.facial_preprocessor import (
    get_facial_datasets, get_facial_dataloaders, compute_facial_class_weights,
)
from utils.augmentation import (
    get_train_transforms, get_val_transforms,
    mixup_data, mixup_criterion,
)
from utils.callbacks import EarlyStopping, TensorBoardLogger

logger = logging.getLogger(__name__)


# =========================================================================
# Label Smoothing Cross-Entropy Loss
# =========================================================================

class LabelSmoothingCE(nn.Module):
    """Cross-entropy loss with label smoothing.

    Parameters
    ----------
    smoothing : float
        Label smoothing factor (default 0.1).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)
        smooth_val = self.smoothing / (n_classes - 1)
        with torch.no_grad():
            smooth_target = torch.full_like(pred, smooth_val)
            smooth_target.scatter_(1, target.unsqueeze(1),
                                   1.0 - self.smoothing)
        log_prob = F.log_softmax(pred, dim=1)
        return -(smooth_target * log_prob).sum(dim=1).mean()


# =========================================================================
# Model Builder
# =========================================================================

def build_facial_model(num_classes: int = 7):
    """Create EfficientNet-B2 with custom classification head.

    Returns
    -------
    model : nn.Module
    """
    model = timm.create_model('efficientnet_b2', pretrained=True,
                               num_classes=0)
    num_features = model.num_features  # typically 1408 for B2

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 512),
        nn.GELU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )

    # Override forward to use our classifier
    original_forward_features = model.forward_features

    def custom_forward(x):
        features = original_forward_features(x)
        features = model.global_pool(features)
        if isinstance(features, tuple):
            features = features[0]
        # Flatten if needed
        if features.dim() > 2:
            features = features.flatten(1)
        return model.classifier(features)

    model.forward = custom_forward

    logger.info(f"  EfficientNet-B2 loaded — backbone features: {num_features}")
    logger.info(f"  Total params: "
                 f"{sum(p.numel() for p in model.parameters()):,}")
    return model


# =========================================================================
# Phase 1: Feature Extraction
# =========================================================================

def train_phase1(model, train_loader, val_loader, config, device,
                 class_weights, tb_logger):
    """Freeze backbone, train classifier head only."""
    logger.info("\n═══ PHASE 1: Feature Extraction (Head Only) ═══")

    p1_cfg = config['facial']['phase1']

    # Freeze all parameters except classifier
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable params (head only): {trainable:,}")

    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=p1_cfg['learning_rate']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=p1_cfg['step_size'], gamma=p1_cfg['gamma']
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(1, p1_cfg['epochs'] + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader,
                     desc=f"  P1 Epoch {epoch}/{p1_cfg['epochs']}",
                     leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        train_loss /= total
        train_acc = correct / total

        # Validation
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        tb_logger.log_scalars('phase1/loss',
                               {'train': train_loss, 'val': val_loss}, epoch)
        tb_logger.log_scalars('phase1/accuracy',
                               {'train': train_acc, 'val': val_acc}, epoch)

        logger.info(
            f"  P1 Epoch {epoch}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

    # Unfreeze for Phase 2
    for param in model.parameters():
        param.requires_grad = True

    return model


# =========================================================================
# Phase 2: Fine-Tuning
# =========================================================================

def train_phase2(model, train_loader, val_loader, config, device,
                 class_weights, tb_logger):
    """Unfreeze last 3 MBConv blocks, apply layer-wise LR, label smoothing,
    MixUp, and OneCycleLR."""
    logger.info("\n═══ PHASE 2: Fine-Tuning (Unfreeze Last 3 Blocks) ═══")

    p2_cfg = config['facial']['phase2']

    # Re-freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 blocks + classifier
    if hasattr(model, 'blocks'):
        n_blocks = len(model.blocks)
        for i in range(max(0, n_blocks - 3), n_blocks):
            for param in model.blocks[i].parameters():
                param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    # Also unfreeze batch norm in conv_head / bn2 if they exist
    if hasattr(model, 'conv_head'):
        for param in model.conv_head.parameters():
            param.requires_grad = True
    if hasattr(model, 'bn2'):
        for param in model.bn2.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable params (fine-tune): {trainable:,}")

    # Separate parameter groups
    unfrozen_backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier' in name:
            head_params.append(param)
        else:
            unfrozen_backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': unfrozen_backbone_params, 'lr': p2_cfg['backbone_lr'],
         'weight_decay': p2_cfg['backbone_wd']},
        {'params': head_params, 'lr': p2_cfg['head_lr'],
         'weight_decay': p2_cfg['head_wd']},
    ])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[p2_cfg['backbone_lr'] * 10, p2_cfg['head_lr']],
        epochs=p2_cfg['epochs'],
        steps_per_epoch=len(train_loader),
    )

    criterion = LabelSmoothingCE(smoothing=p2_cfg['label_smoothing'])
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)  # for validation
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    early_stop = EarlyStopping(
        patience=p2_cfg['patience'], mode='max', restore_best=True
    )

    mixup_alpha = p2_cfg['mixup_alpha']
    mixup_prob = p2_cfg['mixup_prob']
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, p2_cfg['epochs'] + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader,
                     desc=f"  P2 Epoch {epoch}/{p2_cfg['epochs']}",
                     leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Apply MixUp to ~50% of batches
            use_mixup = random.random() < mixup_prob

            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                if use_mixup:
                    mixed_x, y_a, y_b, lam = mixup_data(
                        images, labels, alpha=mixup_alpha
                    )
                    outputs = model(mixed_x)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                    # For accuracy tracking, use original labels
                    preds = outputs.argmax(1)
                    correct += (lam * (preds == y_a).float()
                                + (1 - lam) * (preds == y_b).float()
                                ).sum().item()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    correct += (outputs.argmax(1) == labels).sum().item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=p2_cfg['max_grad_norm']
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * images.size(0)
            total += images.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss /= total
        train_acc = correct / total

        # Validation (no MixUp, standard CE)
        val_loss, val_acc = _evaluate(model, val_loader, ce_criterion, device)
        best_val_acc = max(best_val_acc, val_acc)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        p1_epochs = config['facial']['phase1']['epochs']
        global_epoch = p1_epochs + epoch
        tb_logger.log_scalars('phase2/loss',
                               {'train': train_loss, 'val': val_loss},
                               global_epoch)
        tb_logger.log_scalars('phase2/accuracy',
                               {'train': train_acc, 'val': val_acc},
                               global_epoch)
        tb_logger.log_lr(optimizer, global_epoch)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"  P2 Epoch {epoch}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

        early_stop(epoch, val_acc, model)
        if early_stop.should_stop:
            logger.info(f"  Early stopping at phase 2 epoch {epoch}")
            break

    early_stop.restore(model)
    return model, best_val_acc, history


# =========================================================================
# Helpers
# =========================================================================

@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    """Run evaluation pass, return avg loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


# =========================================================================
# Main entry point
# =========================================================================

def train_facial_model(config: dict, device: torch.device):
    """Full facial emotion model training pipeline.

    Returns
    -------
    model : EfficientNet-B2 (fine-tuned)
    class_names : list of 7 emotion names
    history : dict of training metrics
    """
    models_dir = os.path.join(config['paths']['models_output_dir'],
                               'facial_model')
    os.makedirs(models_dir, exist_ok=True)

    image_size = config['facial']['image_size']

    # ── Data ──
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    train_dataset, val_dataset, class_names = get_facial_datasets(
        config, train_transform, val_transform
    )
    train_loader, val_loader = get_facial_dataloaders(
        train_dataset, val_dataset, config
    )
    class_weights = compute_facial_class_weights(train_dataset, device)

    # ── Model ──
    model = build_facial_model(num_classes=len(class_names))
    model = model.to(device)

    # ── TensorBoard ──
    tb_logger = TensorBoardLogger(
        os.path.join(config['paths']['logs_dir'], 'facial_efficientnet')
    )

    # ── Phase 1 ──
    model = train_phase1(
        model, train_loader, val_loader, config, device,
        class_weights, tb_logger
    )

    # ── Phase 2 ──
    model, best_acc, history = train_phase2(
        model, train_loader, val_loader, config, device,
        class_weights, tb_logger
    )

    tb_logger.close()

    # ── Save ──
    torch.save(model.state_dict(),
               os.path.join(models_dir, 'efficientnet_b2_facial.pth'))

    with open(os.path.join(models_dir, 'facial_class_names.json'), 'w') as f:
        json.dump(class_names, f, indent=2)

    with open(os.path.join(models_dir,
                            'facial_training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    arch_info = {
        'backbone': 'efficientnet_b2',
        'pretrained': True,
        'num_classes': len(class_names),
        'image_size': image_size,
        'classifier': 'Dropout(0.4)->512->GELU->BN->Dropout(0.3)->256->GELU->Dropout(0.2)->7',
    }
    with open(os.path.join(models_dir,
                            'efficientnet_b2_architecture.json'), 'w') as f:
        json.dump(arch_info, f, indent=2)

    logger.info(f"\n  ╔══════════════════════════════════════════╗")
    logger.info(f"  ║  FACIAL MODEL BEST VAL ACCURACY: {best_acc:.4f}  ║")
    logger.info(f"  ╚══════════════════════════════════════════╝")

    return model, class_names, history


if __name__ == '__main__':
    import yaml
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    with open('configs/training_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_facial_model(cfg, dev)
