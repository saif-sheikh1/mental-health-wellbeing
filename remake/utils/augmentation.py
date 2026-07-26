"""
MindSense AI — Data Augmentation Utilities
============================================
Train/Val transforms for facial emotion recognition, MixUp augmentation,
and Test-Time Augmentation (TTA).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


# =========================================================================
# Transform Pipelines
# =========================================================================

def get_train_transforms(image_size: int = 260):
    """Training augmentation pipeline for facial emotion images.

    Applies heavy augmentation including spatial transforms, color jitter,
    random erasing (simulates occlusions), and normalization for
    ImageNet-pretrained backbones.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size + 20, image_size + 20)),   # 280×280
        transforms.RandomCrop(image_size),                        # 260×260
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        ),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1),
            scale=(0.9, 1.1), shear=5
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms(image_size: int = 260):
    """Validation / test transform pipeline — deterministic."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# =========================================================================
# MixUp Augmentation
# =========================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Apply MixUp augmentation to a batch.

    Parameters
    ----------
    x : Tensor of shape (B, C, H, W)
    y : Tensor of shape (B,)   — integer class labels
    alpha : float
        Beta distribution parameter (default 0.2).

    Returns
    -------
    mixed_x : Tensor of shape (B, C, H, W)
    y_a, y_b : original and shuffled targets
    lam : float — mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a = y
    y_b = y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor,
                    lam: float) -> torch.Tensor:
    """Compute MixUp loss.

    Parameters
    ----------
    criterion : loss function (e.g. CrossEntropyLoss)
    pred : model output logits
    y_a, y_b : original and shuffled targets
    lam : mixing coefficient
    """
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# =========================================================================
# Test-Time Augmentation (TTA)
# =========================================================================

class TestTimeAugmentation:
    """5-view TTA for facial emotion inference.

    For each image, produces 5 forward-pass variants:
      1. Original
      2. Horizontal flip
      3. Rotate +8°
      4. Rotate −8°
      5. Center crop from resize(288)
    Averages the softmax outputs across all 5 views.
    """

    def __init__(self, image_size: int = 260):
        self.image_size = image_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Base transform (applied to all views before specific aug)
        self.base = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
        ])

        # 5 TTA views
        self.views = [
            # View 1: Original
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                self.normalize,
            ]),
            # View 2: Horizontal flip
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                self.normalize,
            ]),
            # View 3: Rotate +8°
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=(8, 8)),
                transforms.ToTensor(),
                self.normalize,
            ]),
            # View 4: Rotate −8°
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=(-8, -8)),
                transforms.ToTensor(),
                self.normalize,
            ]),
            # View 5: Center crop from resize(288)
            transforms.Compose([
                transforms.Resize((288, 288)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]),
        ]

    def __call__(self, pil_image):
        """Apply TTA to a single PIL image.

        Returns
        -------
        list of 5 tensors, each of shape (C, H, W)
        """
        img = self.base(pil_image)
        return [view(img) for view in self.views]

    @staticmethod
    @torch.no_grad()
    def predict_with_tta(model, images_list, device='cpu'):
        """Run TTA prediction on a list of 5 augmented tensors.

        Parameters
        ----------
        model : nn.Module in eval mode
        images_list : list of 5 tensors, each (C, H, W)
        device : str

        Returns
        -------
        avg_probs : Tensor of shape (n_classes,)
        """
        model.eval()
        all_probs = []
        for img_tensor in images_list:
            x = img_tensor.unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            all_probs.append(probs)
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs
