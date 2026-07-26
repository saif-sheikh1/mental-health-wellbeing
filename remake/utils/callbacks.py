"""
MindSense AI — Training Callbacks
===================================
EarlyStopping with best-weight restore and TensorBoard logging helpers.
"""

import copy
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping with patience and best-weight restoration.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    mode : str
        'min' to minimize monitored metric (e.g. loss),
        'max' to maximize (e.g. accuracy).
    min_delta : float
        Minimum change to qualify as an improvement.
    restore_best : bool
        If True, restore model weights to the best checkpoint on stop.
    """

    def __init__(self, patience: int = 20, mode: str = 'min',
                 min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.best_state_dict = None
        self.should_stop = False

        if mode == 'min':
            self._is_better = lambda current, best: current < best - min_delta
            self.best_score = np.inf
        else:
            self._is_better = lambda current, best: current > best + min_delta
            self.best_score = -np.inf

    def __call__(self, epoch: int, score: float, model: torch.nn.Module):
        """Check if training should stop.

        Parameters
        ----------
        epoch : int — current epoch number
        score : float — metric value to monitor
        model : nn.Module — model to snapshot
        """
        if self._is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                self.best_state_dict = copy.deepcopy(model.state_dict())
            logger.debug(f"  EarlyStopping: improved to {score:.6f} "
                          f"at epoch {epoch}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"  EarlyStopping: no improvement for {self.patience} "
                    f"epochs. Best={self.best_score:.6f} at epoch "
                    f"{self.best_epoch}."
                )

    def restore(self, model: torch.nn.Module):
        """Load the best weights back into the model."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            logger.info(f"  Restored best weights from epoch {self.best_epoch}")
        else:
            logger.warning("  No best state_dict to restore.")


class TensorBoardLogger:
    """Thin wrapper around SummaryWriter for structured experiment logging.

    Parameters
    ----------
    log_dir : str
        Directory for TensorBoard log files.
    """

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"  TensorBoard logging → {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        self.writer.add_histogram(tag, values, step)

    def log_lr(self, optimizer, step: int):
        for i, pg in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f'lr/group_{i}', pg['lr'], step)

    def close(self):
        self.writer.flush()
        self.writer.close()
