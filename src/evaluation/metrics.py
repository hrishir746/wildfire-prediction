"""
Evaluation metrics for wildfire prediction.

- IoU (Intersection over Union) for fire extent
- RoS error (rate of spread) vs baseline
- Baseline comparison (simulator vs model)
"""

import numpy as np
import torch
from typing import Tuple, Optional


def iou_binary(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    IoU for binary fire presence. pred and target shape (H, W) or (1, H, W).

    pred: probability or binary; thresholded at threshold.
    """
    if pred.ndim == 3:
        pred = pred.squeeze(0)
    if target.ndim == 3:
        target = target.squeeze(0)
    pred_bin = (pred >= threshold).astype(np.float32)
    target_bin = (target > 0).astype(np.float32)
    intersection = (pred_bin * target_bin).sum()
    union = (pred_bin + target_bin > 0).astype(np.float32).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def iou_batch_torch(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """IoU per sample in batch. pred/target (B, 1, H, W). Returns (B,) IoU."""
    pred_bin = (pred >= threshold).float()
    target_bin = (target > 0).float()
    inter = (pred_bin * target_bin).flatten(1).sum(1)
    union = (pred_bin + target_bin > 0).float().flatten(1).sum(1)
    union = torch.clamp(union, min=1e-8)
    return (inter / union)


def bce_with_logits_or_sigmoid(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE loss. If pred is logits (unbounded), use BCEWithLogitsLoss; else treat as probs and use BCE."""
    if pred.min() < 0 or pred.max() > 1 or (pred.min() >= 0 and pred.max() <= 1 and pred.dim() == 4):
        # Heuristic: if values outside [0,1], treat as logits
        return torch.nn.functional.binary_cross_entropy_with_logits(pred, target.clamp(0, 1))
    return torch.nn.functional.binary_cross_entropy(pred.clamp(1e-7, 1 - 1e-7), target.clamp(0, 1))


def ros_error_cell_based(
    pred_fire: np.ndarray,
    sim_fire: np.ndarray,
    cell_size_m: float = 30.0,
    time_steps: int = 10,
    time_step_s: float = 60.0,
) -> float:
    """
    Approximate RoS error: compare predicted vs simulated fire extent.

    RoS (m/min) ~ (new cells * cell_size_m) / (time_steps * time_step_s / 60).
    Returns absolute difference in "effective RoS" (simplified).
    """
    pred_bin = (pred_fire.squeeze() >= 0.5).astype(np.float32)
    sim_bin = (sim_fire.squeeze() > 0).astype(np.float32)
    pred_area = pred_bin.sum() * (cell_size_m ** 2)
    sim_area = sim_bin.sum() * (cell_size_m ** 2)
    total_time_min = time_steps * time_step_s / 60.0
    if total_time_min <= 0:
        return 0.0
    # Effective spread rate (area/time as proxy)
    pred_ross = pred_area / total_time_min
    sim_ross = sim_area / total_time_min
    return float(np.abs(pred_ross - sim_ross))


def dice_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Dice (F1) for binary segmentation."""
    if pred.ndim == 3:
        pred = pred.squeeze(0)
    if target.ndim == 3:
        target = target.squeeze(0)
    pred_bin = (pred >= threshold).astype(np.float32)
    target_bin = (target > 0).astype(np.float32)
    inter = (pred_bin * target_bin).sum()
    total = pred_bin.sum() + target_bin.sum()
    if total == 0:
        return 1.0
    return float(2 * inter / total)
