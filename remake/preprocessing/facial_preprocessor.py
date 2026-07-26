"""
MindSense AI — Facial Data Preprocessor
========================================
Helpers for loading the facial emotion dataset using torchvision ImageFolder,
computing class weights, and building DataLoaders.
"""

import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


def get_facial_datasets(config: dict, train_transform, val_transform):
    """Load train and validation ImageFolder datasets.

    Parameters
    ----------
    config : dict
        Full training config with paths.facial_dataset_root.
    train_transform : torchvision.transforms.Compose
        Transform pipeline for training images.
    val_transform : torchvision.transforms.Compose
        Transform pipeline for validation images.

    Returns
    -------
    train_dataset, val_dataset : ImageFolder instances
    class_names : list of str
    """
    root = config['paths']['facial_dataset_root']
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'test')  # archive uses 'test' as val

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            f"Expected structure: {root}/train/<class_name>/images"
        )
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}\n"
            f"Expected structure: {root}/test/<class_name>/images"
        )

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    class_names = train_dataset.classes
    logger.info(f"Facial classes: {class_names}")
    logger.info(f"Train samples: {len(train_dataset)}, "
                 f"Val samples: {len(val_dataset)}")

    return train_dataset, val_dataset, class_names


def get_facial_dataloaders(train_dataset, val_dataset, config: dict):
    """Build DataLoaders for facial emotion datasets.

    Parameters
    ----------
    train_dataset, val_dataset : ImageFolder
    config : dict
        Must contain facial.batch_size and training.num_workers.

    Returns
    -------
    train_loader, val_loader : DataLoader instances
    """
    batch_size = config['facial']['batch_size']
    num_workers = config['training']['num_workers']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def compute_facial_class_weights(dataset, device='cpu'):
    """Compute balanced class weights from an ImageFolder dataset.

    Returns
    -------
    class_weights : torch.Tensor of shape (n_classes,)
    """
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    weights = compute_class_weight('balanced', classes=classes, y=targets)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Facial class weights: {weights.round(3)}")
    return weights_tensor
